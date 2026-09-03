#!/usr/bin/env python3
"""Resumable local trainer for the DDM CL1/RX2 HPAC rate ladders.

This is an owned lift of PR130's ``train_hpac_self_compress.py``.  The model,
loss, optimizer, scheduler, and selection cadence are intentionally kept the
same.  The lift adds the execution properties the intake trainer lacks,
including the repository-mandated EMA deployment policy:

* immutable source/input identities;
* SSD-only storage preflight;
* atomic periodic and phase-boundary checkpoints;
* optimizer, scheduler, best-state, and all used RNG state in every checkpoint;
* exact ``--resume-from`` continuation; and
* a success manifest that certifies every retained artifact.

It is a training tool, not a byte verdict.  Its reported token/model estimates
remain advisory until the intake packer and codec produce real serialized
bytes.  The default CL1 profile admits only MPS.  The sealed RX2 profile uses
CPU-torch, as explicitly admitted by its charter, and keeps the HB2 reference
architecture, context, schedule, and training budget unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.pr130_lift.train_semantic_quantized_resumable import (  # noqa: E402
    _ema_training_state,
    _restore_ema_training_state,
    ema_eval_scope,
    resolve_ema_policy,
)
from tac.training import EMA  # noqa: E402

PRIMARY_SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
FALLBACK_SSD_ROOT = Path("/Volumes/APDataStore/pact")
SSD_ROOTS = (PRIMARY_SSD_ROOT, FALLBACK_SSD_ROOT)
JF1_LOCAL_ROOT = (
    REPO_ROOT
    / ".omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit"
)
DEFAULT_INTAKE_CODE = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code")
EXPECTED_INTAKE_SHA256 = {
    "hpac_integer.py": "6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f",
    "hpac_self_compress.py": "d63d67945a0719ebbe72da36e6c99909557219360d50e74f20577d68d678beec",
    "train_hpac_self_compress.py": "2a16d537cd4cf2233f41fc5f4046f12b798aba841990ab3b0863afbef23a587e",
}
LOCAL_CAUSAL_FILES = (
    Path("src/tac/training.py"),
    Path("src/tac/pr130_lift/train_semantic_quantized_resumable.py"),
    Path("src/tac/witness_dsl/lawref.py"),
    Path("src/tac/canonical_equations/evaluators.py"),
    Path("src/tac/canonical_equations/ema_decay_run_geometry_20260717.py"),
)
PREREGISTERED_CONFIG = {
    "epochs": 60,
    "batch_size": 8,
    "eval_batch_size": 4,
    "eval_every": 2,
    "lr": 0.003,
    "lr_exponent": 0.0002,
    "lr_bits": 0.01,
    "bit_eps": 1e-6,
    "qat_fraction": 0.5,
    "init_bits": 8.0,
    "channels": 64,
    "patch": 64,
    "delta": 2,
    "frame_dim": 8,
    "norm_mode": "none",
    "activation": "relu",
    "frame_scale": True,
    "weight_bound": 127,
    "activation_bound": 127,
    "weight_scales": True,
    "weight_exponent_min": -6,
    "spm": True,
    "norm_gates": False,
    "target_mode": "raw",
    "seed": 20260716,
    "device": "mps",
    "ema_target_seed_fraction": 0.01,
}
RX2_PREREGISTERED_CONFIG = {
    **PREREGISTERED_CONFIG,
    "device": "cpu",
}
JF1_PREREGISTERED_CONFIG = dict(RX2_PREREGISTERED_CONFIG)
PREREGISTERED_CONFIG_BY_PROFILE = {
    "cl1": PREREGISTERED_CONFIG,
    "rx2_mc36": RX2_PREREGISTERED_CONFIG,
    "jf1_joint_refit": JF1_PREREGISTERED_CONFIG,
}
PREREGISTERED_RATE_LAMBDAS_BY_PROFILE = {
    "cl1": frozenset({1.0, 0.5, 0.25}),
    "rx2_mc36": frozenset({1.0}),
    "jf1_joint_refit": frozenset({1.0}),
}
EXPECTED_CACHE_SHA256 = "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"
EXPECTED_INIT_SHA256 = "0e6c30cef6b36c4e530779c92c56e9128c1d86c62e85e9fc5358a7e9f40ec985"
EXPECTED_RX2_SPATIAL_TOKEN_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
DEFAULT_MIN_FREE_BYTES = 1 << 30
CHECKPOINT_SCHEMA = "ddm_cl1_hpac_capacity_checkpoint.v2"
MANIFEST_SCHEMA = "ddm_cl1_hpac_capacity_artifact_manifest.v1"


class CL1TrainingError(RuntimeError):
    """Fail-closed error for a violated CL1 execution contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_hash_update(digest: Any, value: Any) -> None:
    """Add a typed, deterministic Python/Torch tree encoding to ``digest``."""

    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"I" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, float):
        digest.update(b"F" + struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y" + len(value).to_bytes(8, "big") + value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(
            b"T"
            + len(metadata).to_bytes(8, "big")
            + metadata
            + len(raw).to_bytes(8, "big")  # MEASURE_ONLY_OK:atomic checkpoint retains this tensor payload
            + raw  # MEASURE_ONLY_OK:atomic checkpoint retains this tensor payload
        )
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        metadata = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = array.tobytes(order="C")
        digest.update(
            b"A"
            + len(metadata).to_bytes(8, "big")
            + metadata
            + len(raw).to_bytes(8, "big")  # MEASURE_ONLY_OK:atomic checkpoint retains this array payload
            + raw  # MEASURE_ONLY_OK:atomic checkpoint retains this array payload
        )
    elif isinstance(value, np.generic):
        _stable_hash_update(digest, value.item())
    elif isinstance(value, dict):
        digest.update(b"D" + len(value).to_bytes(8, "big"))
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _stable_hash_update(digest, key)
            _stable_hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update((b"L" if isinstance(value, list) else b"Q") + len(value).to_bytes(8, "big"))
        for item in value:
            _stable_hash_update(digest, item)
    else:
        raise CL1TrainingError(f"causal-state hash does not support {type(value).__name__}")


def _causal_state_sha256(payload: dict[str, Any]) -> str:
    causal = {key: value for key, value in payload.items() if key not in {"causal_state_sha256", "resume_lineage"}}
    digest = hashlib.sha256()
    _stable_hash_update(digest, causal)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        with tmp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_torch_save(path: Path, value: Any, *, immutable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        if immutable and path.exists():
            if _sha256_file(tmp) != _sha256_file(path):
                raise CL1TrainingError(f"immutable checkpoint differs on replay: {path}")
            tmp.unlink()
            return
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _preserve_exact_file(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve(strict=False)
    if source_resolved == destination_resolved:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(source)
    if destination.exists():
        if _sha256_file(destination) != source_sha256:
            raise CL1TrainingError(f"preserved resume parent differs: {destination}")
        return
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, tmp.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        if _sha256_file(tmp) != source_sha256:
            raise CL1TrainingError("resume parent copy changed bytes")
        os.replace(tmp, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def _validate_output_layout(args: argparse.Namespace) -> None:
    checkpoint_root = args.save.with_name(args.save.stem + ".checkpoints")
    named_paths = {
        "save": args.save,
        "out": args.out,
        "manifest": args.save.with_suffix(".artifacts.json"),
        "pause_receipt": args.out.with_suffix(".paused.json"),
        "latest": checkpoint_root / "latest.pt",
        "initial_stage": checkpoint_root / "initial_stage_start.pt",
    }
    resolved: dict[Path, str] = {}
    for label, path in named_paths.items():
        identity = path.expanduser().resolve(strict=False)
        if identity in resolved:
            raise CL1TrainingError(f"output path collision: {label} and {resolved[identity]} -> {identity}")
        resolved[identity] = label
    out_resolved = args.out.expanduser().resolve(strict=False)
    checkpoint_resolved = checkpoint_root.expanduser().resolve(strict=False)
    try:
        out_resolved.relative_to(checkpoint_resolved)
    except ValueError:
        pass
    else:
        raise CL1TrainingError("--out may not live inside the checkpoint tree")
    if args.resume_from is None:
        existing = [
            str(path)
            for path in (
                args.save,
                args.out,
                args.save.with_suffix(".artifacts.json"),
                args.out.with_suffix(".paused.json"),
                checkpoint_root,
            )
            if path.exists()
        ]
        if existing:
            raise CL1TrainingError(
                "fresh run output already exists; resume or choose a new root: " + json.dumps(existing)
            )


def _read_lineage(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ddm_cl1_hpac_capacity_resume_lineage.v1":
        raise CL1TrainingError(f"resume lineage has the wrong schema: {path}")
    return _validate_lineage_entries(payload.get("entries"), label=str(path))


def _validate_lineage_entries(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CL1TrainingError(f"resume lineage entries are malformed: {label}")
    required = {"source_path", "preserved_path", "bytes", "sha256", "epoch"}
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, dict):
            raise CL1TrainingError(f"resume lineage entry {index} is not an object: {label}")
        entry = cast("dict[str, Any]", raw_entry)
        if not required.issubset(entry):
            raise CL1TrainingError(f"resume lineage entry {index} lacks required custody fields: {label}")
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CL1TrainingError(f"resume lineage entry {index} has malformed sha256: {label}")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 1:
            raise CL1TrainingError(f"resume lineage entry {index} has malformed byte count: {label}")
        if not isinstance(entry["epoch"], int) or entry["epoch"] < 0:
            raise CL1TrainingError(f"resume lineage entry {index} has malformed epoch: {label}")
        entries.append(entry)
    ids = [entry["sha256"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise CL1TrainingError(f"resume lineage repeats a parent: {label}")
    return [dict(entry) for entry in entries]


def _checkpoint_lineage(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "resume_lineage" not in payload:
        raise CL1TrainingError("resume checkpoint has no embedded authoritative lineage")
    return _validate_lineage_entries(payload["resume_lineage"], label="embedded checkpoint lineage")


def _preserve_lineage_parents(entries: list[dict[str, Any]], checkpoint_root: Path) -> list[dict[str, Any]]:
    preserved: list[dict[str, Any]] = []
    for entry in entries:
        expected_sha256 = entry["sha256"]
        candidates = (Path(entry["preserved_path"]), Path(entry["source_path"]))
        source = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file()
                and candidate.stat().st_size == entry["bytes"]
                and _sha256_file(candidate) == expected_sha256
            ),
            None,
        )
        if source is None:
            raise CL1TrainingError(f"resume ancestor bytes are unavailable or changed: {expected_sha256}")
        destination = checkpoint_root / "resume_parents" / f"{expected_sha256}.pt"
        _preserve_exact_file(source, destination)
        preserved.append({**entry, "preserved_path": str(destination)})
    return preserved


def _ssd_root_for_path(path: Path, label: str) -> Path:
    resolved_parent = path.expanduser().resolve(strict=False).parent
    for root in SSD_ROOTS:
        resolved_root = root.resolve()
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError:
            continue
        return resolved_root
    roots = ", ".join(str(root.resolve()) for root in SSD_ROOTS)
    raise CL1TrainingError(f"{label} must live under an admitted SSD tier ({roots}): {path}")


def _require_ssd_path(path: Path, label: str) -> Path:
    _ssd_root_for_path(path, label)
    return path


def _storage_root_for_args(path: Path, label: str, args: argparse.Namespace) -> Path:
    if args.profile == "jf1_joint_refit" and args.explicit_local_output_opt_in:
        resolved_parent = path.expanduser().resolve(strict=False).parent
        resolved_root = JF1_LOCAL_ROOT.resolve()
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError as exc:
            raise CL1TrainingError(
                f"{label} must live under the JF1 local receipt root {resolved_root}: {path}"
            ) from exc
        return resolved_root
    return _ssd_root_for_path(path, label)


def _storage_preflight(
    save: Path,
    out: Path,
    *,
    min_free_bytes: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    save_root = _storage_root_for_args(save, "--save", args)
    out_root = _storage_root_for_args(out, "--out", args)
    if save_root != out_root:
        raise CL1TrainingError("--save and --out must use the same admitted storage tier")
    if min_free_bytes < 0:
        raise CL1TrainingError("--min-free-bytes must be nonnegative")
    free = shutil.disk_usage(save_root).free
    if free < min_free_bytes:
        raise CL1TrainingError(f"SSD storage preflight failed: required {min_free_bytes} free bytes, observed {free}")
    return {
        "schema": "ddm_cl1_storage_preflight.v1",
        "tier": str(save_root),
        "save": str(save),
        "out": str(out),
        "required_free_bytes": min_free_bytes,
        "observed_free_bytes": free,
        "status": "PASS",
        "explicit_local_output_opt_in": bool(args.explicit_local_output_opt_in),
        "cleanup_policy": (
            "atomic temporary files are removed automatically; periodic and "
            "stage checkpoints are preserved as measurement evidence on the admitted tier"
        ),
    }


def _verify_intake(code_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_INTAKE_SHA256.items():
        path = code_root / name
        if not path.is_file():
            raise CL1TrainingError(f"pinned intake source is absent: {path}")
        digest = _sha256_file(path)
        if digest != expected:
            raise CL1TrainingError(f"pinned intake source changed for {name}: expected {expected}, observed {digest}")
        observed[name] = digest
    return observed


def _local_causal_sha256() -> dict[str, str]:
    return {path.as_posix(): _sha256_file(REPO_ROOT / path) for path in LOCAL_CAUSAL_FILES}


def _assert_preregistered_config(args: argparse.Namespace) -> None:
    expected_config = PREREGISTERED_CONFIG_BY_PROFILE[args.profile]
    differences = {
        key: {"expected": expected, "observed": getattr(args, key)}
        for key, expected in expected_config.items()
        if getattr(args, key) != expected
    }
    admitted_lambdas = PREREGISTERED_RATE_LAMBDAS_BY_PROFILE[args.profile]
    if args.rate_lambda not in admitted_lambdas:
        differences["rate_lambda"] = {
            "expected": sorted(admitted_lambdas),
            "observed": args.rate_lambda,
        }
    if differences:
        raise CL1TrainingError(
            "invocation differs from the receiver-closed preregistration: " + json.dumps(differences, sort_keys=True)
        )


def _import_pr130(code_root: Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    code = str(code_root)
    if code not in sys.path:
        sys.path.insert(0, code)
    from hpac_integer import IntegerHPAC
    from hpac_self_compress import (
        bit_depth_histogram,
        enable_self_compression,
        estimated_model_bits,
        set_deployed_bit_depths,
        variable_weight_bits,
    )

    return (
        IntegerHPAC,
        bit_depth_histogram,
        enable_self_compression,
        estimated_model_bits,
        set_deployed_bit_depths,
        variable_weight_bits,
    )


def _residuals(tokens: torch.Tensor) -> torch.Tensor:
    output = tokens.clone()
    output[1:] = (tokens[1:] - tokens[:-1]) % 5
    return output


def _device_rng_state(device: torch.device) -> dict[str, Any]:
    if device.type == "mps":
        return {"kind": "mps", "state": torch.mps.get_rng_state().cpu()}
    if device.type == "cpu":
        return {"kind": "cpu", "state": torch.random.get_rng_state()}
    raise CL1TrainingError(f"unsupported CL1 device for RNG custody: {device}")


def _restore_device_rng_state(device: torch.device, payload: dict[str, Any]) -> None:
    if device.type == "mps" and payload.get("kind") == "mps":
        torch.mps.set_rng_state(payload["state"])
        return
    if device.type == "cpu" and payload.get("kind") == "cpu":
        torch.random.set_rng_state(payload["state"])
        return
    raise CL1TrainingError(f"checkpoint device RNG kind {payload.get('kind')!r} does not match {device.type!r}")


def _rng_payload(device: torch.device, generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "device": _device_rng_state(device),
        "shuffle_generator": generator.get_state(),
    }


def _restore_rng_payload(
    device: torch.device,
    generator: torch.Generator,
    payload: dict[str, Any],
) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    _restore_device_rng_state(device, payload["device"])
    generator.set_state(payload["shuffle_generator"])


def _clone_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sysctl_value(name: str) -> str | None:
    completed = subprocess.run(
        ["sysctl", "-n", name],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _hardware_identity() -> dict[str, Any]:
    relevant_env = (
        "PYTHONHASHSEED",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "PYTORCH_MPS_PREFER_METAL",
        "TAC_ADMISSION_ENFORCE",
        "TAC_GOVERNED_ADMISSION",
    )
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "host": platform.node(),
        "hw_model": _sysctl_value("hw.model"),
        "cpu_brand": _sysctl_value("machdep.cpu.brand_string"),
        "memory_bytes": _sysctl_value("hw.memsize"),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "environment": {name: os.environ.get(name) for name in relevant_env},
    }


def _run_identity(
    args: argparse.Namespace,
    *,
    cache_sha256: str,
    init_sha256: str,
    source_sha256: dict[str, str],
    ema_policy: dict[str, Any],
) -> dict[str, Any]:
    training_keys = (
        "profile",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "eval_every",
        "lr",
        "lr_exponent",
        "lr_bits",
        "bit_eps",
        "rate_lambda",
        "qat_fraction",
        "init_bits",
        "channels",
        "patch",
        "delta",
        "frame_dim",
        "norm_mode",
        "activation",
        "frame_scale",
        "weight_bound",
        "activation_bound",
        "weight_scales",
        "weight_exponent_min",
        "spm",
        "norm_gates",
        "target_mode",
        "seed",
        "device",
        "ema_target_seed_fraction",
    )
    trainer_sha256 = _sha256_file(Path(__file__).resolve())
    local_causal_source_sha256 = _local_causal_sha256()
    training_config = {key: getattr(args, key) for key in training_keys}
    schedule_config = {key: value for key, value in training_config.items() if key != "rate_lambda"}
    source_identity = {
        "trainer_sha256": trainer_sha256,
        "intake_source_sha256": source_sha256,
        "local_causal_source_sha256": local_causal_source_sha256,
    }
    return {
        "schema": "ddm_cl1_hpac_capacity_run_identity.v1",
        "launch_git_sha": _git_sha(),
        "trainer": str(Path(__file__).resolve()),
        "trainer_sha256": trainer_sha256,
        "training_config": training_config,
        "seed_schedule_identity_sha256": _canonical_json_sha256(
            {"schedule_config": schedule_config, "ema_policy": ema_policy}
        ),
        "trainer_source_identity_sha256": _canonical_json_sha256(source_identity),
        "cache_path": str(args.cache.resolve()),
        "cache_sha256": cache_sha256,
        "init_path": str(args.init.resolve()),
        "init_sha256": init_sha256,
        "intake_code_root": str(args.intake_code.resolve()),
        "intake_source_sha256": source_sha256,
        "local_causal_source_sha256": local_causal_source_sha256,
        "ema_policy": ema_policy,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "hardware": _hardware_identity(),
    }


def _checkpoint_payload(
    *,
    epoch: int,
    phase: str,
    model: torch.nn.Module,
    ema: EMA,
    ema_policy: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    device: torch.device,
    best: dict[str, Any] | None,
    history: list[dict[str, Any]],
    run_identity: dict[str, Any],
    resume_lineage: list[dict[str, Any]],
    qat_start: int,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "phase": phase,
        "qat_start": qat_start,
        "state_dict": _cpu_tree(ema.state_dict()),
        "deployment_weights": "ema_shadow",
        "live_state_dict": _clone_cpu_state_dict(model),
        "ema": _ema_training_state(ema),
        "ema_policy": ema_policy,
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "rng": _rng_payload(device, generator),
        "best": best,
        "history": history,
        "run_identity": run_identity,
        "run_identity_sha256": _canonical_json_sha256(run_identity),
        "resume_lineage": list(resume_lineage),
    }
    payload["causal_state_sha256"] = _causal_state_sha256(payload)
    return payload


def _artifact_row(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _epoch_controls(*, epoch: int, epochs: int, qat_start: int, eval_every: int) -> dict[str, bool]:
    discrete = epoch >= qat_start
    should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
    continuous_stage_end = epoch == qat_start - 1 and not discrete
    qat_stage_end = epoch == epochs and discrete
    return {
        "discrete": discrete,
        "should_evaluate": should_evaluate,
        "continuous_stage_end": continuous_stage_end,
        "qat_stage_end": qat_stage_end,
        "should_checkpoint": (should_evaluate or continuous_stage_end or qat_stage_end),
    }


def _write_success_manifest(
    *,
    args: argparse.Namespace,
    storage_preflight: dict[str, Any],
    run_identity: dict[str, Any],
    checkpoint_root: Path,
    result_path: Path,
    final_path: Path,
) -> Path:
    artifacts = [
        _artifact_row(final_path, role="trainer_surrogate_best_checkpoint"),
        _artifact_row(result_path, role="trainer_advisory_result"),
    ]
    for path in sorted(checkpoint_root.rglob("*.pt")):
        role = "resume_parent" if path.parent.name == "resume_parents" else "resumable_checkpoint"
        artifacts.append(_artifact_row(path, role=role))
    lineage_path = checkpoint_root / "resume_lineage.json"
    resume_lineage: list[dict[str, Any]] = []
    if lineage_path.is_file():
        resume_lineage = _read_lineage(lineage_path)
        artifacts.append(_artifact_row(lineage_path, role="resume_lineage"))
    manifest_path = final_path.with_suffix(".artifacts.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "axis": _training_axis(args.device),
        "argv": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "git_sha": run_identity["launch_git_sha"],
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "run_identity": run_identity,
        "resume_lineage": resume_lineage,
        "storage_preflight": storage_preflight,
        "artifacts": artifacts,
        "rebuildability": {
            "status": "CERTIFIED",
            "reason": (
                "causal training state and EMA deployment weights are bound to "
                "the recorded config, source hashes, input hashes, and RNG state; "
                "outer checkpoint/manifest bytes are not claimed identical across "
                "pause/resume because lineage and argv telemetry legitimately differ"
            ),
            "scratch_cleanup": "atomic .tmp files removed automatically",
        },
        "authority_boundary": (
            "trainer estimates are not packed model bytes or coded token bytes; "
            "pack and encode every selected checkpoint before any ladder verdict"
        ),
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P0-resumable local lift of PR130 HPAC self-compression")
    parser.add_argument("--profile", choices=tuple(PREREGISTERED_CONFIG_BY_PROFILE), default="cl1")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-exponent", type=float, default=2e-4)
    parser.add_argument("--lr-bits", type=float, default=0.1)
    parser.add_argument("--bit-eps", type=float, default=1e-3)
    parser.add_argument("--rate-lambda", type=float, default=1.0)
    parser.add_argument("--qat-fraction", type=float, default=0.25)
    parser.add_argument("--init-bits", type=float, default=8.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--frame-dim", type=int, default=8)
    parser.add_argument("--norm-mode", choices=("none", "center", "power"), default="none")
    parser.add_argument("--activation", choices=("relu", "leaky"), default="relu")
    parser.add_argument("--frame-scale", action="store_true")
    parser.add_argument("--weight-bound", type=int, default=127)
    parser.add_argument("--activation-bound", type=int, default=127)
    parser.add_argument("--weight-scales", action="store_true")
    parser.add_argument("--weight-exponent-min", type=int, default=-6)
    parser.add_argument("--spm", action="store_true")
    parser.add_argument("--norm-gates", action="store_true")
    parser.add_argument("--target-mode", choices=("raw", "residual"), default="raw")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--ema-target-seed-fraction", type=float, default=0.01)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--explicit-local-output-opt-in",
        action="store_true",
        help=(
            "JF1 only: admit output under the exact local arm receipt root while both SSD "
            "tiers are full; all other local paths and profiles remain refused"
        ),
    )
    parser.add_argument(
        "--expected-cache-content-sha256",
        help="JF1 only: required SHA-256 of the raw uint8 n600 field inside --cache",
    )
    parser.add_argument(
        "--expected-init-sha256",
        help="JF1 only: required SHA-256 of the exact warm-start file passed as --init",
    )
    parser.add_argument(
        "--resume-allow-trainer-drift",
        action="store_true",
        help=(
            "Allow resume when ONLY the trainer file's own hash drifted from the "
            "checkpoint (e.g. a resume-gate fix). All cache/init/intake/config/"
            "hardware identity guards stay strict; the drift is printed loudly."
        ),
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="test-only governed pause after publishing this epoch checkpoint",
    )
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--intake-code", type=Path, default=DEFAULT_INTAKE_CODE)
    return parser


def _training_axis(device: str) -> str:
    if device == "cpu":
        return "[macOS-CPU training-signal; byte measurement pending]"
    return "[macOS-MPS research-signal; byte measurement pending]"


def _verify_rx2_cache_payload(cache_payload: Any) -> str:
    if not isinstance(cache_payload, dict) or "seg" not in cache_payload:
        raise CL1TrainingError("RX2 cache must be a mapping containing seg")
    seg = cache_payload["seg"]
    if not isinstance(seg, torch.Tensor):
        raise CL1TrainingError("RX2 cache seg must be a torch tensor")
    if seg.device.type != "cpu" or seg.dtype != torch.uint8 or tuple(seg.shape) != (600, 384, 512):
        raise CL1TrainingError("RX2 cache seg must be CPU uint8 with shape (600,384,512)")
    if int(seg.min()) < 0 or int(seg.max()) > 4:
        raise CL1TrainingError("RX2 cache seg values must be in [0,4]")
    digest = hashlib.sha256(seg.contiguous().numpy().tobytes(order="C")).hexdigest()
    if digest != EXPECTED_RX2_SPATIAL_TOKEN_SHA256:
        raise CL1TrainingError(
            f"RX2 cache token SHA differs: expected {EXPECTED_RX2_SPATIAL_TOKEN_SHA256}, observed {digest}"
        )
    declared = cache_payload.get("spatial_token_sha256")
    if declared != digest:
        raise CL1TrainingError("RX2 cache spatial_token_sha256 metadata is absent or differs")
    return digest


def _verify_jf1_cache_payload(cache_payload: Any, expected_sha256: str | None) -> str:
    if expected_sha256 is None:
        raise CL1TrainingError("JF1 requires --expected-cache-content-sha256")
    if not isinstance(cache_payload, dict) or "seg" not in cache_payload:
        raise CL1TrainingError("JF1 cache must be a mapping containing seg")
    seg = cache_payload["seg"]
    if not isinstance(seg, torch.Tensor):
        raise CL1TrainingError("JF1 cache seg must be a torch tensor")
    if seg.device.type != "cpu" or seg.dtype != torch.uint8 or tuple(seg.shape) != (600, 384, 512):
        raise CL1TrainingError("JF1 cache seg must be CPU uint8 with shape (600,384,512)")
    if int(seg.min()) < 0 or int(seg.max()) > 4:
        raise CL1TrainingError("JF1 cache seg values must be in [0,4]")
    digest = hashlib.sha256(seg.contiguous().numpy().tobytes(order="C")).hexdigest()
    if digest != expected_sha256:
        raise CL1TrainingError(
            f"JF1 cache token SHA differs: expected {expected_sha256}, observed {digest}"
        )
    if cache_payload.get("spatial_token_sha256") != digest:
        raise CL1TrainingError("JF1 cache spatial_token_sha256 metadata is absent or differs")
    return digest


def main() -> None:
    assert_governed_admission("train_ddm_cl1_hpac_capacity")
    args = _build_parser().parse_args()
    if args.epochs < 1:
        raise CL1TrainingError("--epochs must be positive")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise CL1TrainingError("batch sizes must be positive")
    if args.eval_every < 1:
        raise CL1TrainingError("--eval-every must be positive")
    if not 0.0 <= args.qat_fraction <= 1.0:
        raise CL1TrainingError("--qat-fraction must be in [0, 1]")
    if args.rate_lambda <= 0.0:
        raise CL1TrainingError("--rate-lambda must be positive")
    if not 0.0 < args.ema_target_seed_fraction < 1.0:
        raise CL1TrainingError("--ema-target-seed-fraction must be in (0, 1)")
    _assert_preregistered_config(args)
    if args.stop_after_epoch is not None and not (1 <= args.stop_after_epoch <= args.epochs):
        raise CL1TrainingError("--stop-after-epoch must be within the run")
    if args.stop_after_epoch is not None:
        qat_start_for_validation = max(1, math.floor(args.epochs * (1.0 - args.qat_fraction)) + 1)
        stop_controls = _epoch_controls(
            epoch=args.stop_after_epoch,
            epochs=args.epochs,
            qat_start=qat_start_for_validation,
            eval_every=args.eval_every,
        )
        if not stop_controls["should_checkpoint"]:
            raise CL1TrainingError("--stop-after-epoch must be an evaluation or stage-boundary epoch")
    if args.resume_from is not None and not args.resume_from.is_file():
        raise CL1TrainingError(f"--resume-from is not a file: {args.resume_from}")
    _validate_output_layout(args)
    if args.profile != "jf1_joint_refit" and args.explicit_local_output_opt_in:
        raise CL1TrainingError("--explicit-local-output-opt-in is admitted only for jf1_joint_refit")
    if args.profile == "jf1_joint_refit" and not args.explicit_local_output_opt_in:
        raise CL1TrainingError("jf1_joint_refit requires --explicit-local-output-opt-in")
    if args.profile != "jf1_joint_refit" and (
        args.expected_cache_content_sha256 is not None or args.expected_init_sha256 is not None
    ):
        raise CL1TrainingError("JF1 input pins are admitted only for jf1_joint_refit")
    if args.resume_from is not None:
        _storage_root_for_args(args.resume_from, "--resume-from", args)
    if platform.system() != "Darwin":
        raise CL1TrainingError("CL1 local-Metal training requires macOS")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise CL1TrainingError("set PYTHONHASHSEED=0 before launch")
    if os.environ.get("TAC_ADMISSION_ENFORCE") != "1":
        raise CL1TrainingError("set TAC_ADMISSION_ENFORCE=1 for a hard governor gate")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise CL1TrainingError("set PYTORCH_ENABLE_MPS_FALLBACK=0; CPU fallback is forbidden")

    storage_preflight = _storage_preflight(
        args.save,
        args.out,
        min_free_bytes=args.min_free_bytes,
        args=args,
    )
    source_sha256 = _verify_intake(args.intake_code)
    if not args.cache.is_file() or not args.init.is_file():
        raise CL1TrainingError("--cache and --init must be existing files")
    cache_sha256 = _sha256_file(args.cache)
    init_sha256 = _sha256_file(args.init)
    cache_payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if args.profile == "cl1" and cache_sha256 != EXPECTED_CACHE_SHA256:
        raise CL1TrainingError(f"cache SHA differs: expected {EXPECTED_CACHE_SHA256}, observed {cache_sha256}")
    if args.profile == "rx2_mc36":
        _verify_rx2_cache_payload(cache_payload)
    elif args.profile == "jf1_joint_refit":
        _verify_jf1_cache_payload(cache_payload, args.expected_cache_content_sha256)
    expected_init_sha256 = (
        args.expected_init_sha256 if args.profile == "jf1_joint_refit" else EXPECTED_INIT_SHA256
    )
    if expected_init_sha256 is None:
        raise CL1TrainingError("JF1 requires --expected-init-sha256")
    if init_sha256 != expected_init_sha256:
        raise CL1TrainingError(f"init SHA differs: expected {expected_init_sha256}, observed {init_sha256}")

    if args.device == "mps" and (not torch.backends.mps.is_built() or not torch.backends.mps.is_available()):
        raise CL1TrainingError("local Metal is unavailable in this process; CPU substitution is forbidden")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    (
        IntegerHPAC,
        bit_depth_histogram,
        enable_self_compression,
        estimated_model_bits,
        set_deployed_bit_depths,
        variable_weight_bits,
    ) = _import_pr130(args.intake_code)

    raw_tokens = cache_payload["seg"].long().to(device)
    if tuple(raw_tokens.shape) != (600, 384, 512):
        raise CL1TrainingError("CL1 requires the exact n600 segmentation field shape (600,384,512)")
    if 384 % args.patch or 512 % args.patch:
        raise CL1TrainingError("--patch must divide both 384 and 512 exactly")
    tokens = raw_tokens if args.target_mode == "raw" else _residuals(raw_tokens)
    previous = torch.zeros_like(raw_tokens)
    previous[1:] = raw_tokens[:-1]
    model = IntegerHPAC(
        channels=args.channels,
        patch=args.patch,
        delta=args.delta,
        frame_dim=args.frame_dim,
        norm_mode=args.norm_mode,
        activation=args.activation,
        use_frame_scale=args.frame_scale,
        weight_bound=args.weight_bound,
        activation_bound=args.activation_bound,
        use_weight_scales=args.weight_scales,
        weight_exponent_min=args.weight_exponent_min,
        use_spm=args.spm,
        use_norm_gates=args.norm_gates,
    ).to(device)
    enable_self_compression(model, args.init_bits)

    resume: dict[str, Any] | None = None
    if args.resume_from is None:
        initial = torch.load(args.init, map_location="cpu", weights_only=False)
        try:
            incompatible = model.load_state_dict(initial["state_dict"], strict=False)
        except RuntimeError as exc:
            raise CL1TrainingError(
                "initial checkpoint tensor shapes do not match the requested "
                "topology; structural rungs require an explicit owned initializer"
            ) from exc
        allowed_missing = {name for name in incompatible.missing_keys if name.endswith(".bit_depth")}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if incompatible.unexpected_keys or unexpected_missing:
            raise CL1TrainingError(f"incompatible initialization checkpoint: {incompatible}")
    else:
        resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume.get("schema") != CHECKPOINT_SCHEMA:
            raise CL1TrainingError("--resume-from has the wrong checkpoint schema")
        observed_causal_sha256 = resume.get("causal_state_sha256")
        recomputed_causal_sha256 = _causal_state_sha256(resume)
        if observed_causal_sha256 != recomputed_causal_sha256:
            raise CL1TrainingError("--resume-from causal state hash is absent or does not verify")
        model.load_state_dict(resume["live_state_dict"])

    named_parameters = dict(model.named_parameters())
    bit_names = {name for name in named_parameters if name.endswith(".bit_depth")}
    exponent_names = {name for name in named_parameters if name.endswith(".exponent")}
    other_names = set(named_parameters) - bit_names - exponent_names
    groups = [
        {
            "params": [named_parameters[name] for name in sorted(other_names)],
            "lr": args.lr,
            "eps": 1e-8,
        },
        {
            "params": [named_parameters[name] for name in sorted(bit_names)],
            "lr": args.lr_bits,
            "eps": args.bit_eps,
            "weight_decay": 0.0,
        },
    ]
    if exponent_names:
        groups.append(
            {
                "params": [named_parameters[name] for name in sorted(exponent_names)],
                "lr": args.lr_exponent,
                "eps": 1e-8,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pixels = tokens.numel()
    frame_count = tokens.shape[0]
    qat_start = max(1, math.floor(args.epochs * (1.0 - args.qat_fraction)) + 1)
    updates_per_run = args.epochs * math.ceil(frame_count / args.batch_size)
    ema_policy = resolve_ema_policy(
        updates_per_run,
        target_seed_fraction=args.ema_target_seed_fraction,
    )
    ema = EMA(model, decay=float(ema_policy["decay"]), warmup=True)
    run_identity = _run_identity(
        args,
        cache_sha256=cache_sha256,
        init_sha256=init_sha256,
        source_sha256=source_sha256,
        ema_policy=ema_policy,
    )

    @torch.no_grad()
    def evaluate() -> dict[str, Any]:
        with ema_eval_scope(model, ema):
            set_deployed_bit_depths(model, True)
            nats = 0.0
            misses = 0
            for start in range(0, frame_count, args.eval_batch_size):
                end = min(start + args.eval_batch_size, frame_count)
                idx = torch.arange(start, end, device=device)
                target = tokens[start:end]
                logits = model(target, idx, previous[start:end])
                nats += float(F.cross_entropy(logits, target, reduction="sum"))
                misses += int((logits.argmax(dim=1) != target).sum().item())
            bpp = nats / math.log(2) / pixels
            model_bits = estimated_model_bits(model)
            histogram = bit_depth_histogram(model)
        token_bytes = math.ceil(bpp * pixels / 8)
        model_bytes = math.ceil(model_bits / 8)
        return {
            "bpp": bpp,
            "top1_error": misses / pixels,
            "estimated_token_bytes": token_bytes,
            "estimated_model_bytes": model_bytes,
            "estimated_joint_bytes": token_bytes + model_bytes,
            "bit_depth_histogram": histogram,
            "byte_authority": "ADVISORY_ESTIMATE_NOT_SERIALIZED",
            "evaluated_weights": "ema_shadow",
        }

    checkpoint_root = args.save.with_name(args.save.stem + ".checkpoints")
    latest_path = checkpoint_root / "latest.pt"
    lineage_path = checkpoint_root / "resume_lineage.json"
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    resume_lineage: list[dict[str, Any]] = []
    start_epoch = 0
    if resume is None:
        _atomic_json(
            lineage_path,
            {
                "schema": "ddm_cl1_hpac_capacity_resume_lineage.v1",
                "entries": [],
            },
        )
    else:
        observed_identity = resume.get("run_identity")
        # launch_git_sha moves on ANY repo commit (even an unrelated memo), which
        # would brick resume forever; behavioral drift is still fully guarded by
        # trainer_sha256, source/intake/cache/init hashes, config, and hardware.
        _volatile = {"launch_git_sha"}
        if args.resume_allow_trainer_drift:
            # Explicit, recorded override: the trainer file changed since the
            # checkpoint (e.g. a resume-gate fix). Cache/init/intake/config/
            # hardware guards stay strict; the drift is printed, never silent.
            _volatile |= {"trainer_sha256", "trainer_source_identity_sha256"}
            print(
                "[resume] trainer drift EXPLICITLY allowed: "
                f"ckpt={str((observed_identity or {}).get('trainer_sha256'))[:16]} "
                f"now={str(run_identity.get('trainer_sha256'))[:16]}",
                flush=True,
            )
        # Hardware: compare only the BEHAVIORAL subset (thread geometry, arch,
        # OS family, relevant env). Probe metadata (cpu_brand/hw_model/memory/
        # mps flags/host/release) varies with the OBSERVER, not the machine —
        # a sandboxed arm launch records sysctl=None and mps_available=False on
        # the very same M5 Max, which must not brick resume from MAIN.
        _hw_behavioral = (
            "system",
            "machine",
            "torch_num_threads",
            "torch_num_interop_threads",
            "environment",
        )

        def _cmp_view(identity: dict[str, Any]) -> dict[str, Any]:
            view = {k: v for k, v in identity.items() if k not in _volatile}
            hw = view.get("hardware")
            if isinstance(hw, dict):
                view["hardware"] = {k: hw.get(k) for k in _hw_behavioral}
            return view

        observed_cmp = _cmp_view(observed_identity or {})
        current_cmp = _cmp_view(run_identity)
        if observed_cmp != current_cmp:
            drift = sorted(
                k
                for k in set(observed_cmp) | set(current_cmp)
                if observed_cmp.get(k) != current_cmp.get(k)
            )
            raise CL1TrainingError(
                "--resume-from run identity differs; config/input/source drift refused; "
                f"drifting keys: {drift}"
            )
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        if resume.get("ema_policy") != ema_policy:
            raise CL1TrainingError("--resume-from EMA policy differs")
        _restore_ema_training_state(ema, resume["ema"], device=device)
        best = resume["best"]
        history = resume["history"]
        start_epoch = int(resume["epoch"])
        if start_epoch > args.epochs:
            raise CL1TrainingError("resume checkpoint is past --epochs")
        _restore_rng_payload(device, generator, resume["rng"])
        parent_sha256 = _sha256_file(args.resume_from)
        preserved_parent = checkpoint_root / "resume_parents" / f"{parent_sha256}.pt"
        _preserve_exact_file(args.resume_from, preserved_parent)
        # The immutable checkpoint is authoritative.  The adjacent sidecar is
        # only a convenience view of the most recently continued branch and
        # can legitimately be newer than an older checkpoint forked elsewhere.
        resume_lineage = _preserve_lineage_parents(_checkpoint_lineage(resume), checkpoint_root)
        parent_row = {
            "source_path": str(args.resume_from.resolve()),
            "preserved_path": str(preserved_parent),
            "bytes": args.resume_from.stat().st_size,
            "sha256": parent_sha256,
            "epoch": start_epoch,
        }
        if not any(row.get("sha256") == parent_sha256 for row in resume_lineage):
            resume_lineage.append(parent_row)
        _atomic_json(
            lineage_path,
            {
                "schema": "ddm_cl1_hpac_capacity_resume_lineage.v1",
                "entries": resume_lineage,
            },
        )
    started = time.time()
    initial_metrics = evaluate()
    print(
        json.dumps(
            {
                "epoch": start_epoch,
                "resume": args.resume_from is not None,
                **initial_metrics,
            }
        ),
        flush=True,
    )
    if resume is None:
        initial_payload = _checkpoint_payload(
            epoch=0,
            phase="initial",
            model=model,
            ema=ema,
            ema_policy=ema_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            device=device,
            best=best,
            history=history,
            run_identity=run_identity,
            resume_lineage=resume_lineage,
            qat_start=qat_start,
        )
        _atomic_torch_save(
            checkpoint_root / "initial_stage_start.pt",
            initial_payload,
            immutable=True,
        )
        _atomic_torch_save(latest_path, initial_payload, immutable=False)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        controls = _epoch_controls(
            epoch=epoch,
            epochs=args.epochs,
            qat_start=qat_start,
            eval_every=args.eval_every,
        )
        discrete_bits = controls["discrete"]
        set_deployed_bit_depths(model, discrete_bits)
        permutation = torch.randperm(frame_count, generator=generator, device=device)
        for start in range(0, frame_count, args.batch_size):
            idx = permutation[start : start + args.batch_size]
            target = tokens[idx]
            logits = model(target, idx, previous[idx])
            task_loss = F.cross_entropy(logits, target)
            rate_loss = args.rate_lambda * math.log(2) * variable_weight_bits(model, deployed=False) / pixels
            loss = task_loss + rate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ema.update(model)
        scheduler.step()

        phase = "discrete_qat" if discrete_bits else "continuous"
        should_evaluate = controls["should_evaluate"]
        continuous_stage_end = controls["continuous_stage_end"]
        qat_stage_end = controls["qat_stage_end"]
        if should_evaluate:
            metrics = evaluate()
            record = {
                "epoch": epoch,
                "phase": phase,
                **metrics,
            }
            history.append(record)
            print(
                json.dumps(
                    {
                        **record,
                        "telemetry_process_elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
            if best is None or metrics["estimated_joint_bytes"] < best["estimated_joint_bytes"]:
                best = {
                    **metrics,
                    "epoch": epoch,
                    "state_dict": _cpu_tree(ema.state_dict()),
                    "deployment_weights": "ema_shadow",
                }
        should_checkpoint = controls["should_checkpoint"]
        resume_checkpoint_path: Path | None = None
        if should_checkpoint:
            payload = _checkpoint_payload(
                epoch=epoch,
                phase=phase,
                model=model,
                ema=ema,
                ema_policy=ema_policy,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                device=device,
                best=best,
                history=history,
                run_identity=run_identity,
                resume_lineage=resume_lineage,
                qat_start=qat_start,
            )
            if continuous_stage_end:
                continuous_path = checkpoint_root / f"continuous_stage_end_epoch_{epoch:04d}.pt"
                _atomic_torch_save(
                    continuous_path,
                    payload,
                    immutable=True,
                )
                resume_checkpoint_path = continuous_path
            if qat_stage_end:
                qat_path = checkpoint_root / f"qat_stage_end_epoch_{epoch:04d}.pt"
                _atomic_torch_save(
                    qat_path,
                    payload,
                    immutable=True,
                )
                resume_checkpoint_path = qat_path
            if should_evaluate:
                periodic_path = checkpoint_root / "periodic" / f"epoch_{epoch:04d}.pt"
                _atomic_torch_save(
                    periodic_path,
                    payload,
                    immutable=True,
                )
                resume_checkpoint_path = periodic_path
            _atomic_torch_save(latest_path, payload, immutable=False)

        if args.stop_after_epoch == epoch:
            if not should_checkpoint:
                raise CL1TrainingError("--stop-after-epoch must coincide with a checkpoint epoch")
            if resume_checkpoint_path is None:
                raise CL1TrainingError("checkpoint publication path is absent")
            pause_path = args.out.with_suffix(".paused.json")
            _atomic_json(
                pause_path,
                {
                    "schema": "ddm_cl1_hpac_capacity_pause.v1",
                    "epoch": epoch,
                    "resume_from": str(resume_checkpoint_path),
                    "resume_from_sha256": _sha256_file(resume_checkpoint_path),
                    "run_identity_sha256": _canonical_json_sha256(run_identity),
                    "score_claim": False,
                },
            )
            print(json.dumps({"status": "PAUSED", "receipt": str(pause_path)}))
            return

    if best is None:
        raise CL1TrainingError("training completed without an evaluated checkpoint")
    result = {
        "schema": "ddm_cl1_hpac_capacity_trainer_result.v1",
        "best_epoch_by_advisory_surrogate": best["epoch"],
        "best_bpp": best["bpp"],
        "best_top1_error": best["top1_error"],
        "estimated_token_bytes": best["estimated_token_bytes"],
        "estimated_model_bytes": best["estimated_model_bytes"],
        "estimated_joint_bytes": best["estimated_joint_bytes"],
        "bit_depth_histogram": best["bit_depth_histogram"],
        "history": history,
        "config": _jsonable_args(args),
        "run_identity": run_identity,
        "resume_lineage": resume_lineage,
        "score_claim": False,
        "axis": _training_axis(args.device),
        "selection_warning": (
            "best is selected by the intake trainer's ideal-token plus theoretical "
            "model-bit surrogate and is excluded from the CL1 ladder; the "
            "preregistered row is the full-state terminal epoch-60 QAT checkpoint, "
            "which must be real-packed, real-encoded, and exactly decoded"
        ),
    }
    # PAYLOAD WRITE ORDER (ddm_pl1 incident A2, cured by ddm_ql2): the cheap and
    # IRREPLACEABLE `result` lands before the multi-MB checkpoint. A2 finished 379 s
    # of compute, then `_atomic_torch_save` raised `IsADirectoryError` and the
    # `result` write two lines later never ran -- the whole run's readable product
    # was lost. The checkpoint is rebuildable; the evaluated scalars are not.
    _atomic_json(args.out, result)
    _atomic_torch_save(
        args.save,
        {"state_dict": best["state_dict"], "result": result},
        immutable=False,
    )
    manifest_path = _write_success_manifest(
        args=args,
        storage_preflight=storage_preflight,
        run_identity=run_identity,
        checkpoint_root=checkpoint_root,
        result_path=args.out,
        final_path=args.save,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_TRAINING_ONLY",
                "result": str(args.out),
                "manifest": str(manifest_path),
                "score_claim": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
