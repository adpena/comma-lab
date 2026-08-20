#!/usr/bin/env python3
"""Build and run the DDM XI2 full-scale xi-context promotion.

XI2 keeps the proven CL1 model topology and 60-epoch lambda-1 schedule, but
replaces CL1's unwarped previous decoded partition with XI1's deterministic
xi-warped previous partition.  The comparison control is never retrained: its
attested 116,716-byte n600 Range stream remains the byte authority.

The encoder and decoder derive every context causally from the previous exact
partition and the already-counted pose row.  The retained n600 context arrays
are training/debug evidence only; they are not a decoder sidecar or counted
payload.  No scorer is imported or executed by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import lzma
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import constriction
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.admission_guard import assert_governed_admission  # noqa: E402

OUTPUT = Path("/Volumes/APDataStore/pact/ddm_xi2_20260812")
RETAINED = OUTPUT / "retained"
INPUTS = RETAINED / "inputs"
CONTEXTS = RETAINED / "contexts"
TRAINING = RETAINED / "training"
SERIALIZED = RETAINED / "serialized"
QUEUE = OUTPUT / "queue"
STATE = OUTPUT / "state.json"
BUILD_RECEIPT = OUTPUT / "BUILD_RECEIPT.json"
READY_TO_FIRE = OUTPUT / "READY_TO_FIRE.json"
FULL_SCALE_RESULT = OUTPUT / "FULL_SCALE_RESULT.json"

XI1_PATH = ROOT / "tools/run_ddm_xi1_screw_conditioned_learned_prior.py"
CL1_TRAINER_PATH = ROOT / "tools/train_ddm_cl1_hpac_capacity.py"
HP3_PATH = ROOT / "experiments/ddm_hp3_hpac_section_and_zip_frame.py"
CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt")  # GT_LINEAGE_OK: bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
INITIALIZER = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt")
POSE_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2/retained/inputs/pose_targets_n600_f16.bin")
CALIBRATION_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2/retained/inputs/warp_calibration_f64.bin"
)
CONTROL_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_uninterrupted_twin/training")
CONTROL_RANGE = CONTROL_ROOT / "serialized/terminal.range.bin"
CONTROL_RAW = CONTROL_ROOT / "serialized/terminal.raw.u8"
CONTROL_MODEL = CONTROL_ROOT / "serialized/terminal.model.bin.xz"
CONTROL_REPORT = CONTROL_ROOT / "reports/trainer.json"
CONTROL_TRAIN_RECEIPT = CONTROL_ROOT / "run/training.safe_run.json"
CONTROL_PACK_RECEIPT = CONTROL_ROOT / "run/terminal.pack.safe_run.json"
CONTROL_ENCODE_RECEIPT = CONTROL_ROOT / "run/terminal.encode.safe_run.json"
CONTROL_DECODE_RECEIPT = CONTROL_ROOT / "run/terminal.decode.safe_run.json"

SCHEMA = "ddm_xi2_xi_context_full_scale.v1"
CHECKPOINT_SCHEMA = "ddm_xi2_xi_context_checkpoint.v1"
AXIS = "[macOS-MPS research-signal training; real Range bytes; scorer-free]"
SEED = 20260716
FRAME_COUNT = 600
H, W, CLASSES = 384, 512, 5
PIXELS = FRAME_COUNT * H * W
EPOCHS = 60
RATE_LAMBDA = 1.0
CONTROL_RANGE_BYTES = 116_716
PROMOTION_RATIO = 0.98
PROMOTION_MAX_BYTES = 114_381  # largest integer strictly below 0.98 * 116,716
CONTROL_TRAIN_SECONDS = 2_894.155
CONTROL_SECONDS_PER_EPOCH = CONTROL_TRAIN_SECONDS / EPOCHS
CONTROL_PACK_SECONDS = 2.225
CONTROL_ENCODE_SECONDS = 679.825
CONTROL_DECODE_SECONDS = 688.595
EXPECTED_PIPELINE_SECONDS = (
    CONTROL_TRAIN_SECONDS + CONTROL_PACK_SECONDS + 2 * CONTROL_ENCODE_SECONDS + CONTROL_DECODE_SECONDS
)
CONTROL_PEAK_RSS_MIB = 1_673.391

EXPECTED_SHA256 = {
    XI1_PATH: "f8ddbd9bb9479950148364d504484bc2fc278150b2eb48c3789a431c42d78882",
    CL1_TRAINER_PATH: "0c1e6464173d61c5a585450310977c13822ea662bf0bf9b59548491209f3d423",
    CACHE: "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195",
    INITIALIZER: "0e6c30cef6b36c4e530779c92c56e9128c1d86c62e85e9fc5358a7e9f40ec985",
    POSE_RAW: "4c14c3195f676888a8f9511e1ab8ac5a6d621d58c16791c3ae2e9648cfa5c29e",
    CALIBRATION_RAW: "3c6db7263465151cd744b4be40eb0a949059613b34881d7e7afeacfe32d92b42",
    CONTROL_RANGE: "ac2c549c1f48756ad33c6c99af8563f2170db1de61cd50d0615d4c1a0cdd7b87",
    CONTROL_RAW: "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece",
    CONTROL_MODEL: "b74be4d5f4c8f7f1aec37577f2277d43ca44ef6d53e5b0138a8ce5e7d7e02325",
    CONTROL_REPORT: "9382d0736a3e001a342f180d7c04cbab0ff0efc7bf3fcf90c642270cdf3a9d7a",
    CONTROL_TRAIN_RECEIPT: "ae18315234d63d145fabddf18bad16cf1c93930f476eec9c50a21288108ec8a4",
    CONTROL_PACK_RECEIPT: "e32cd015c1ce2edc9a70c69898878fcdef7ba31b68e659f79de0d47138d01b21",
    CONTROL_ENCODE_RECEIPT: "24130f327d9f2962ba3f0168b9d47b3e9b3ce58867cc165855e4f4de10fe5c0f",
    CONTROL_DECODE_RECEIPT: "e23d4bad83eeeecf09b645b914e008a47759e2de5cbf66bfad29c651228bfb13",
    ROOT / "tools/measure_pose_warp_dseg.py": "d2d1ea77780748641dcef50b470fec0385f79426417eaa2abdfb6b9d9510e0ec",
    ROOT / "src/tac/lie/_se3_numpy.py": "b0f7a07b96926f6a579aa5c1e1fb5a05d63d183abe2e9eaa9070cb80bf11edee",
    ROOT
    / "src/tac/pr130_runtime/fx1_runtime_tree/inflate.py": "9e9aad19a038eca07dc22d09f251b192bb9faf62729f3cff179a7c6d31610559",
    ROOT
    / "src/tac/pr130_lift/train_semantic_quantized_resumable.py": "9e9eac2ef826830d3dfc86a2a015c5a09149e4af775ec6634c73fc80b0abc148",
    ROOT / "src/tac/training.py": "adac372a319fd37bdda39336e9998f7d459b4a321d91b5ccd51e24d6ddc8c4cf",
    ROOT
    / "src/tac/canonical_equations/ema_decay_run_geometry_20260717.py": "470b44b8fff5c507878b2a2228987f472397a51f620082d55b515815d9014498",
    ROOT
    / "src/tac/canonical_equations/evaluators.py": "a62d382809db423efebda663050743f54c3ab79e0497a26b55096fa1c391ef1f",
    ROOT / "src/tac/witness_dsl/lawref.py": "4b59c8756c74a06b9141e2da94308e38b33c25620244a5451b57fd772f767bde",
}
EXPECTED_INTAKE_SHA256 = {
    "hpac_integer.py": "6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f",
    "hpac_integer_sparse.py": "2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c",
    "hpac_self_compress.py": "d63d67945a0719ebbe72da36e6c99909557219360d50e74f20577d68d678beec",
    "pack_hpac_self_compress.py": "e796d9249926f8c7dcc45a7cdf1f39e33d0b4409ffee275fbb9cd481a6f5f099",
}

TRAIN_CONFIG = {
    "epochs": EPOCHS,
    "batch_size": 8,
    "eval_batch_size": 4,
    "eval_every": 2,
    "lr": 0.003,
    "lr_exponent": 0.0002,
    "lr_bits": 0.01,
    "bit_eps": 1e-6,
    "rate_lambda": RATE_LAMBDA,
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
    "seed": SEED,
    "ema_target_seed_fraction": 0.01,
    "device": "mps",
    "context_mode": "xi_warped_previous_decoded_partition",
}


class XI2Error(RuntimeError):
    """Fail-closed XI2 custody, resume, or decode error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise XI2Error(f"required artifact is absent: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise XI2Error(f"refusing to overwrite a different retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_record(path)


def retain_payload(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist a materialized payload immediately and return its custody row."""

    return _atomic_bytes(path, payload)


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return _atomic_bytes(path, payload, replace=replace)


def atomic_torch(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    import io

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return _atomic_bytes(path, buffer.getvalue(), replace=replace)


def import_path(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise XI2Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_xi1() -> ModuleType:
    observed = sha256_file(XI1_PATH)
    if observed != EXPECTED_SHA256[XI1_PATH]:
        raise XI2Error(f"XI1 source changed: {observed}")
    return import_path(XI1_PATH, "ddm_xi2_pinned_xi1")


def pin_inputs(xi1: ModuleType) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path, expected in EXPECTED_SHA256.items():
        observed = file_record(path)
        if observed["sha256"] != expected:
            raise XI2Error(f"input pin changed for {path}: {observed['sha256']}")
        records[str(path)] = observed
    for name, expected in EXPECTED_INTAKE_SHA256.items():
        path = xi1.INTAKE_CODE / name
        observed = file_record(path)
        if observed["sha256"] != expected:
            raise XI2Error(f"PR130 source pin changed for {name}: {observed['sha256']}")
        records[str(path)] = observed
    expected_xi1_paths = {
        "HPAC_CACHE": CACHE,
        "HPAC_INIT": INITIALIZER,
        "POSE_RAW": POSE_RAW,
        "CALIBRATION_RAW": CALIBRATION_RAW,
        "POSE_WARP_PATH": ROOT / "tools/measure_pose_warp_dseg.py",
        "FX1_RUNTIME": ROOT / "src/tac/pr130_runtime/fx1_runtime_tree",
    }
    for name, expected in expected_xi1_paths.items():
        observed = Path(getattr(xi1, name))
        if observed.resolve() != expected.resolve():
            raise XI2Error(f"XI1 {name} path changed: {observed}")
    report = json.loads(CONTROL_REPORT.read_text(encoding="utf-8"))
    control_config = report.get("run_identity", {}).get("training_config")
    expected_control_config = {key: value for key, value in TRAIN_CONFIG.items() if key != "context_mode"}
    if control_config != expected_control_config:
        raise XI2Error("banked CL1 control training config changed")
    if CONTROL_RANGE.stat().st_size != CONTROL_RANGE_BYTES:
        raise XI2Error("banked CL1 control byte count changed")
    if CONTROL_RAW.stat().st_size != PIXELS:
        raise XI2Error("banked CL1 control raw-token geometry changed")
    return records


def storage_preflight(required_free_bytes: int = 4 << 30) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(OUTPUT)
    if usage.free < required_free_bytes:
        raise XI2Error(f"APData needs {required_free_bytes} free bytes; found {usage.free}")
    return {
        "path": str(OUTPUT),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": usage.free,
        "status": "PASS",
    }


def _poses_and_calibration() -> tuple[np.ndarray, np.ndarray]:
    poses = np.frombuffer(POSE_RAW.read_bytes(), dtype="<f2").astype(np.float64).reshape(FRAME_COUNT, 6)
    calibration = np.frombuffer(CALIBRATION_RAW.read_bytes(), dtype="<f8").copy()
    if calibration.shape != (3,):
        raise XI2Error("calibration must contain three float64 values")
    return poses, calibration


def derive_xi_context(
    previous_decoded: np.ndarray,
    pose6: np.ndarray,
    calibration: np.ndarray,
    *,
    xi1: ModuleType,
    pose_warp: ModuleType,
) -> np.ndarray:
    """Derive one legal context from receiver-known state only."""

    previous = np.asarray(previous_decoded, dtype=np.uint8)
    if previous.shape != (H, W):
        raise XI2Error(f"previous decoded partition has wrong shape: {previous.shape}")
    geometry = (
        pose_warp.NATIVE_W,
        pose_warp.NATIVE_H,
        pose_warp.NATIVE_FX,
        pose_warp.NATIVE_FY,
        pose_warp.NATIVE_CX,
        pose_warp.NATIVE_CY,
        pose_warp.CAMERA_HEIGHT_M,
    )
    if geometry != (1164, 874, 910.0, 910.0, 582.0, 437.0, 1.22):
        raise XI2Error(f"pose-warp receiver geometry changed: {geometry}")
    return xi1.warp_previous_partition(previous, pose6, calibration, pose_warp)


def causal_context_for_frame(
    frame: int,
    previous_decoded: np.ndarray | None,
    poses: np.ndarray,
    calibration: np.ndarray,
    *,
    xi1: ModuleType,
    pose_warp: ModuleType,
) -> np.ndarray:
    if frame == 0:
        return np.zeros((H, W), dtype=np.uint8)
    if previous_decoded is None:
        raise XI2Error("nonzero frame requires the previous decoded partition")
    return derive_xi_context(
        previous_decoded,
        poses[frame],
        calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )


def _array_payload_sha256(path: Path) -> str:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    for start in range(0, len(array), 12):
        digest.update(np.ascontiguousarray(array[start : start + 12]).tobytes(order="C"))
    return digest.hexdigest()


def _materialize_context(
    destination: Path,
    *,
    raw_tokens: np.ndarray,
    poses: np.ndarray,
    calibration: np.ndarray,
    xi1: ModuleType,
    pose_warp: ModuleType,
) -> dict[str, Any]:
    if destination.is_file():
        array = np.load(destination, mmap_mode="r", allow_pickle=False)
        if array.shape != (FRAME_COUNT, H, W) or array.dtype != np.uint8:
            raise XI2Error(f"retained context geometry changed: {destination}")
        return {**file_record(destination), "payload_sha256": _array_payload_sha256(destination)}
    partial = destination.with_name(destination.name + ".partial.npy")
    progress = destination.with_name(destination.name + ".progress.json")
    if partial.is_file() and progress.is_file():
        state = json.loads(progress.read_text(encoding="utf-8"))
        next_frame = int(state["next_frame"])
        array = np.lib.format.open_memmap(partial, mode="r+")
        if array.shape != (FRAME_COUNT, H, W) or array.dtype != np.uint8:
            raise XI2Error("partial context has incompatible geometry")
    elif partial.exists() or progress.exists():
        raise XI2Error("context resume requires both partial payload and progress receipt")
    else:
        partial.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.uint8,
            shape=(FRAME_COUNT, H, W),
        )
        array[0] = 0
        array.flush()
        next_frame = 1
        atomic_json(
            progress,
            {"schema": "ddm_xi2_context_progress.v1", "partial_path": str(partial), "next_frame": 1},
        )
    for frame in range(next_frame, FRAME_COUNT):
        array[frame] = derive_xi_context(
            raw_tokens[frame - 1],
            poses[frame],
            calibration,
            xi1=xi1,
            pose_warp=pose_warp,
        )
        if (frame + 1) % 10 == 0 or frame + 1 == FRAME_COUNT:
            array.flush()
            atomic_json(
                progress,
                {
                    "schema": "ddm_xi2_context_progress.v1",
                    "partial_path": str(partial),
                    "partial_bytes": partial.stat().st_size,
                    "next_frame": frame + 1,
                },
            )
    array.flush()
    del array
    os.replace(partial, destination)
    atomic_json(
        progress,
        {
            "schema": "ddm_xi2_context_progress.v1",
            "complete": True,
            "artifact": file_record(destination),
            "next_frame": FRAME_COUNT,
        },
    )
    return {**file_record(destination), "payload_sha256": _array_payload_sha256(destination)}


def prepare_contexts(xi1: ModuleType) -> dict[str, Any]:
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)["seg"].to(torch.uint8).numpy()
    if cache.shape != (FRAME_COUNT, H, W):
        raise XI2Error("canonical cache geometry changed")
    poses, calibration = _poses_and_calibration()
    pose_warp = xi1.import_path(xi1.POSE_WARP_PATH, "ddm_xi2_pose_warp_prepare")
    primary = _materialize_context(
        CONTEXTS / "xi_warped_previous_n600.npy",
        raw_tokens=cache,
        poses=poses,
        calibration=calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    repeat = _materialize_context(
        CONTEXTS / "xi_warped_previous_n600.repeat.npy",
        raw_tokens=cache,
        poses=poses,
        calibration=calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    if primary["sha256"] != repeat["sha256"] or primary["payload_sha256"] != repeat["payload_sha256"]:
        raise XI2Error("full n600 xi context repeat differs")
    return {"primary": primary, "repeat": repeat, "repeat_exact": True}


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


def _source_identity(pins: dict[str, Any], context_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "runner": file_record(Path(__file__).resolve()),
        "pinned_inputs": pins,
        "context": context_record,
        "train_config": TRAIN_CONFIG,
    }


def _checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    ema: Any,
    ema_policy: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    history: list[dict[str, Any]],
    source_identity: dict[str, Any],
    resume_lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    phase = "initial" if epoch == 0 else ("continuous" if epoch <= EPOCHS // 2 else "discrete_qat")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "phase": phase,
        "train_config": TRAIN_CONFIG,
        "source_identity": source_identity,
        "live_state_dict": _cpu_tree(model.state_dict()),
        "ema_shadow": _cpu_tree(ema.state_dict()),
        "ema_policy": ema_policy,
        "ema_decay": ema.decay,
        "ema_updates": ema._num_updates,
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "shuffle_generator_state": generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "mps_rng_state": torch.mps.get_rng_state().cpu(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "history": history,
        "resume_lineage": resume_lineage,
        "deployment_weights": "terminal ema_shadow",
    }
    payload["causal_state_sha256"] = _checkpoint_digest(payload)
    return payload


def _json_without_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": sha256_bytes(value.contiguous().numpy().tobytes()),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": sha256_bytes(array.tobytes()),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_without_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_without_tensors(item) for item in value]
    return value


def _checkpoint_digest(payload: dict[str, Any]) -> str:
    causal = {key: value for key, value in payload.items() if key not in {"causal_state_sha256", "resume_lineage"}}
    encoded = json.dumps(
        _json_without_tensors(causal),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(encoded)


def _periodic_paths() -> list[Path]:
    return sorted((TRAINING / "checkpoints/periodic").glob("epoch_*.pt"))


def resolve_resume(value: str) -> Path | None:
    if value == "auto":
        paths = _periodic_paths()
        return paths[-1] if paths else None
    path = Path(value)
    if not path.is_file():
        raise XI2Error(f"--resume-from is absent: {path}")
    try:
        path.resolve().relative_to(OUTPUT.resolve())
    except ValueError as exc:
        raise XI2Error("--resume-from must remain under the XI2 custody root") from exc
    return path


def _restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    source_identity: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise XI2Error("resume checkpoint schema changed")
    if checkpoint.get("causal_state_sha256") != _checkpoint_digest(checkpoint):
        raise XI2Error("resume checkpoint causal state hash is absent or does not verify")
    if checkpoint.get("train_config") != TRAIN_CONFIG or checkpoint.get("source_identity") != source_identity:
        raise XI2Error("resume checkpoint config/input/source identity changed")
    model.load_state_dict(checkpoint["live_state_dict"], strict=True)
    ema.shadow = {name: tensor.to(next(model.parameters()).device) for name, tensor in checkpoint["ema_shadow"].items()}
    ema._num_updates = int(checkpoint["ema_updates"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    generator.set_state(checkpoint["shuffle_generator_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"])
    torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    return (
        int(checkpoint["epoch"]),
        list(checkpoint["history"]),
        list(checkpoint.get("resume_lineage", [])),
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    ema: Any,
    compression: ModuleType,
    target: torch.Tensor,
    ids: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, Any]:
    live = _cpu_tree(model.state_dict())
    model.load_state_dict(ema.shadow, strict=True)
    compression.set_deployed_bit_depths(model, True)
    model.eval()
    nats = 0.0
    misses = 0
    for start in range(0, FRAME_COUNT, TRAIN_CONFIG["eval_batch_size"]):
        end = min(start + TRAIN_CONFIG["eval_batch_size"], FRAME_COUNT)
        logits = model(target[start:end], ids[start:end], context[start:end])
        nats += float(F.cross_entropy(logits, target[start:end], reduction="sum"))
        misses += int((logits.argmax(dim=1) != target[start:end]).sum().item())
    model.load_state_dict(live, strict=True)
    return {
        "bpp": nats / math.log(2) / PIXELS,
        "top1_error": misses / PIXELS,
        "estimated_token_bytes": math.ceil(nats / math.log(2) / 8),
        "estimated_model_bytes": math.ceil(compression.estimated_model_bits(model) / 8),
        "bit_depth_histogram": compression.bit_depth_histogram(model),
        "byte_authority": "ADVISORY_ESTIMATE_NOT_SERIALIZED",
        "evaluated_weights": "ema_shadow",
    }


def train_stage(args: argparse.Namespace, xi1: ModuleType, pins: dict[str, Any]) -> dict[str, Any]:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise XI2Error("XI2 training requires live Metal; CPU substitution is forbidden")
    context_path = CONTEXTS / "xi_warped_previous_n600.npy"
    context_record = {**file_record(context_path), "payload_sha256": _array_payload_sha256(context_path)}
    source_identity = _source_identity(pins, context_record)
    integer, compression, _, _ = xi1.configure_hpac()
    device = torch.device("mps")
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.load(CACHE, map_location="cpu", weights_only=False)["seg"].long().to(device)
    context = torch.from_numpy(np.asarray(np.load(context_path, mmap_mode="r"), dtype=np.int64)).to(device)
    ids = torch.arange(FRAME_COUNT, device=device)
    model = xi1.build_train_model(integer, compression, device)
    optimizer = xi1.optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.003 * 0.02)
    generator = torch.Generator(device=device).manual_seed(SEED)
    updates = EPOCHS * math.ceil(FRAME_COUNT / TRAIN_CONFIG["batch_size"])
    ema_policy = xi1.resolve_ema_policy(updates, target_seed_fraction=0.01)
    ema = xi1.EMA(model, decay=float(ema_policy["decay"]), warmup=True)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    resume_lineage: list[dict[str, Any]] = []
    resume_from = resolve_resume(args.resume_from)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        start_epoch, history, resume_lineage = _restore_checkpoint(
            checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            source_identity=source_identity,
        )
        parent = file_record(resume_from)
        if not any(row.get("sha256") == parent["sha256"] for row in resume_lineage):
            resume_lineage.append({**parent, "epoch": start_epoch})
    checkpoint_root = TRAINING / "checkpoints"
    latest = checkpoint_root / "latest.pt"
    if start_epoch == 0 and not latest.exists():
        initial = _checkpoint_payload(
            epoch=0,
            model=model,
            ema=ema,
            ema_policy=ema_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            history=history,
            source_identity=source_identity,
            resume_lineage=resume_lineage,
        )
        atomic_torch(checkpoint_root / "initial_stage_start.pt", initial)
        atomic_torch(latest, initial, replace=True)
    started = time.time()
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        discrete = epoch > EPOCHS // 2
        compression.set_deployed_bit_depths(model, discrete)
        permutation = torch.randperm(FRAME_COUNT, generator=generator, device=device)
        for start in range(0, FRAME_COUNT, TRAIN_CONFIG["batch_size"]):
            index = permutation[start : start + TRAIN_CONFIG["batch_size"]]
            logits = model(target[index], ids[index], context[index])
            task_loss = F.cross_entropy(logits, target[index])
            rate_loss = RATE_LAMBDA * math.log(2) * compression.variable_weight_bits(model, deployed=False) / PIXELS
            loss = task_loss + rate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ema.update(model)
        scheduler.step()
        if epoch == 1 or epoch % TRAIN_CONFIG["eval_every"] == 0 or epoch == EPOCHS:
            metrics = _evaluate(model, ema, compression, target, ids, context)
            record = {
                "epoch": epoch,
                "phase": "discrete_qat" if discrete else "continuous",
                **metrics,
                "telemetry_process_elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            ema=ema,
            ema_policy=ema_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            history=history,
            source_identity=source_identity,
            resume_lineage=resume_lineage,
        )
        periodic = checkpoint_root / f"periodic/epoch_{epoch:04d}.pt"
        atomic_torch(periodic, payload)
        if epoch == EPOCHS // 2:
            atomic_torch(checkpoint_root / f"continuous_stage_end_epoch_{epoch:04d}.pt", payload)
        if epoch == EPOCHS:
            atomic_torch(checkpoint_root / f"qat_stage_end_epoch_{epoch:04d}.pt", payload)
        atomic_torch(latest, payload, replace=True)
    terminal = checkpoint_root / f"qat_stage_end_epoch_{EPOCHS:04d}.pt"
    result = {
        "schema": "ddm_xi2_training_result.v1",
        "status": "COMPLETE_TRAINING_ONLY",
        "terminal_checkpoint": file_record(terminal),
        "history": history,
        "train_config": TRAIN_CONFIG,
        "source_identity": source_identity,
        "resume_lineage": resume_lineage,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(TRAINING / "TRAINING_RESULT.json", result)
    return result


def pack_stage(xi1: ModuleType) -> dict[str, Any]:
    terminal_path = TRAINING / f"checkpoints/qat_stage_end_epoch_{EPOCHS:04d}.pt"
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    if terminal.get("schema") != CHECKPOINT_SCHEMA or terminal.get("epoch") != EPOCHS:
        raise XI2Error("terminal checkpoint is absent or wrong")
    if terminal.get("causal_state_sha256") != _checkpoint_digest(terminal):
        raise XI2Error("terminal checkpoint causal state hash does not verify")
    _, _, packer, _ = xi1.configure_hpac()
    source = packer.model_from_args(xi1.model_args(), True).eval()
    xi1._require_bit_depth_schema(terminal["ema_shadow"], label="XI2 terminal EMA")
    source.load_state_dict(terminal["ema_shadow"], strict=True)
    packer.set_deployed_bit_depths(source, True)
    raw = packer.serialize_self_compressed(source)
    raw_record = retain_payload(SERIALIZED / "terminal.hpac.raw", raw)
    hp3 = import_path(HP3_PATH, "ddm_xi2_hp3")
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=hp3.LZMA_FILTERS)
    compressed_record = retain_payload(SERIALIZED / "terminal.model.bin.xz", compressed)
    repeat_record = retain_payload(
        SERIALIZED / "terminal.model.repeat.bin.xz", lzma.compress(raw, format=lzma.FORMAT_XZ, filters=hp3.LZMA_FILTERS)
    )
    if compressed_record["sha256"] != repeat_record["sha256"]:
        raise XI2Error("packed model repeat differs")
    decoded_raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if decoded_raw != raw:
        raise XI2Error("XZ model payload changed the packed HPAC bytes")
    restored = packer.model_from_args(xi1.model_args(), False).eval()
    packer.deserialize_self_compressed(restored, decoded_raw)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    current = torch.randint(0, CLASSES, (2, H, W), generator=generator)
    previous = torch.randint(0, CLASSES, (2, H, W), generator=generator)
    ids = torch.tensor([0, FRAME_COUNT - 1])
    max_diff = float((source(current, ids, previous) - restored(current, ids, previous)).abs().max())
    if max_diff != 0.0:
        raise XI2Error(f"packed terminal changed logits: {max_diff}")
    result = {
        "schema": "ddm_xi2_pack_result.v1",
        "terminal_checkpoint": file_record(terminal_path),
        "hpac_raw": raw_record,
        "model_xz": compressed_record,
        "model_xz_repeat": repeat_record,
        "repeat_exact": True,
        "xz_roundtrip_exact": True,
        "max_logit_abs_diff": max_diff,
        "deployment_weights": "terminal ema_shadow",
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.pack.json", result)
    return result


def probability_table(selected: torch.Tensor, digest: Any) -> np.ndarray:
    codes = selected.mul(8).round().clamp(-32768, 32767).to(torch.int16).cpu().numpy()
    digest.update(codes.tobytes(order="C"))
    logits = codes.astype(np.float64) / 8
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _codec_components(xi1: ModuleType, device: torch.device) -> tuple[Any, Any, Any]:
    _, _, packer, inflate = xi1.configure_hpac()
    model = packer.model_from_args(xi1.model_args(), False).eval()
    packer.deserialize_self_compressed(model, (SERIALIZED / "terminal.hpac.raw").read_bytes())
    model = model.to(device)
    masks = inflate.group_masks(device)
    sparse = inflate.SparseIntegerHPAC(model, H, W)
    return model, masks, sparse


@torch.no_grad()
def _encode_once(destination: Path, *, xi1: ModuleType) -> dict[str, Any]:
    device = torch.device("mps")
    model, masks, sparse = _codec_components(xi1, device)
    raw = torch.load(CACHE, map_location="cpu", weights_only=False)["seg"].to(torch.uint8).numpy()
    poses, calibration = _poses_and_calibration()
    pose_warp = xi1.import_path(xi1.POSE_WARP_PATH, f"ddm_xi2_pose_warp_encode_{destination.stem}")
    encoder = constriction.stream.queue.RangeEncoder()
    family = constriction.stream.model.Categorical(perfect=False)
    logit_digest = hashlib.sha256()
    context_digest = hashlib.sha256()
    previous: np.ndarray | None = None
    ideal_bits = 0.0
    started = time.time()
    for frame in range(FRAME_COUNT):
        context_np = causal_context_for_frame(
            frame,
            previous,
            poses,
            calibration,
            xi1=xi1,
            pose_warp=pose_warp,
        )
        context_digest.update(context_np.tobytes(order="C"))
        context = torch.from_numpy(context_np.astype(np.int64)).view(1, H, W).to(device)
        current = torch.zeros_like(context)
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        prepared = model.prepare_frame_context(idx, context)
        target = torch.from_numpy(raw[frame].astype(np.int64)).to(device)
        for group, mask in enumerate(masks):
            selected = sparse.selected_logits(current, prepared, group)
            table = probability_table(selected, logit_digest)
            symbols = target[mask].cpu().numpy().astype(np.int32)
            ideal_bits += float(-np.log2(table[np.arange(len(symbols)), symbols].astype(np.float64)).sum())
            encoder.encode(symbols, family, table)
            current[0, mask] = target[mask]
        previous = raw[frame]
        if frame == 0 or (frame + 1) % 25 == 0:
            atomic_json(
                SERIALIZED / f"{destination.name}.progress.json",
                {
                    "schema": "ddm_xi2_encode_progress.v1",
                    "destination": str(destination),
                    "encoded_frames": frame + 1,
                    "elapsed_seconds": time.time() - started,
                },
            )
    payload = encoder.get_compressed().tobytes()
    payload_record = retain_payload(destination, payload)
    return {
        "range": payload_record,
        "frames": FRAME_COUNT,
        "ideal_bpp": ideal_bits / PIXELS,
        "token_bpp": payload_record["bytes"] * 8 / PIXELS,
        "logit_sha256": logit_digest.hexdigest(),
        "context_payload_sha256": context_digest.hexdigest(),
        "elapsed_seconds": time.time() - started,
    }


def encode_stage(xi1: ModuleType) -> dict[str, Any]:
    primary = _encode_once(SERIALIZED / "terminal.range.bin", xi1=xi1)
    repeat = _encode_once(SERIALIZED / "terminal.repeat.range.bin", xi1=xi1)
    if primary["range"]["sha256"] != repeat["range"]["sha256"]:
        raise XI2Error("full-scale Range repeat differs")
    if (
        primary["logit_sha256"] != repeat["logit_sha256"]
        or primary["context_payload_sha256"] != repeat["context_payload_sha256"]
    ):
        raise XI2Error("repeat encode changed logits or causal contexts")
    prepared_sha = _array_payload_sha256(CONTEXTS / "xi_warped_previous_n600.npy")
    if primary["context_payload_sha256"] != prepared_sha:
        raise XI2Error("causal encoder context differs from retained training context")
    result = {
        "schema": "ddm_xi2_encode_result.v1",
        "primary": primary,
        "repeat": repeat,
        "repeat_exact": True,
        "context_matches_training_payload": True,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.encode.json", result)
    return result


@torch.no_grad()
def decode_stage(xi1: ModuleType) -> dict[str, Any]:
    device = torch.device("mps")
    model, masks, sparse = _codec_components(xi1, device)
    payload = (SERIALIZED / "terminal.range.bin").read_bytes()
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(payload, dtype=np.uint32))
    family = constriction.stream.model.Categorical(perfect=False)
    poses, calibration = _poses_and_calibration()
    pose_warp = xi1.import_path(xi1.POSE_WARP_PATH, "ddm_xi2_pose_warp_decode")
    logit_digest = hashlib.sha256()
    context_digest = hashlib.sha256()
    partial = SERIALIZED / f"terminal.raw.attempt_{os.getpid()}.partial.u8"
    output = np.memmap(partial, mode="w+", dtype=np.uint8, shape=(FRAME_COUNT, H, W))
    previous: np.ndarray | None = None
    started = time.time()
    for frame in range(FRAME_COUNT):
        context_np = causal_context_for_frame(
            frame,
            previous,
            poses,
            calibration,
            xi1=xi1,
            pose_warp=pose_warp,
        )
        context_digest.update(context_np.tobytes(order="C"))
        context = torch.from_numpy(context_np.astype(np.int64)).view(1, H, W).to(device)
        current = torch.zeros_like(context)
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        prepared = model.prepare_frame_context(idx, context)
        for group, mask in enumerate(masks):
            selected = sparse.selected_logits(current, prepared, group)
            table = probability_table(selected, logit_digest)
            symbols = decoder.decode(family, table).astype(np.int64)
            current[0, mask] = torch.from_numpy(symbols).to(device)
        decoded = current[0].to(torch.uint8).cpu().numpy()
        output[frame] = decoded
        previous = np.asarray(output[frame])
        if frame == 0 or (frame + 1) % 25 == 0:
            output.flush()
            atomic_json(
                SERIALIZED / "terminal.decode.progress.json",
                {
                    "schema": "ddm_xi2_decode_progress.v1",
                    "partial_path": str(partial),
                    "partial_bytes": partial.stat().st_size,
                    "decoded_frames": frame + 1,
                    "elapsed_seconds": time.time() - started,
                },
            )
    output.flush()
    del output
    destination = SERIALIZED / "terminal.raw.u8"
    replay_record: dict[str, Any] | None = None
    if destination.exists():
        if sha256_file(destination) != sha256_file(partial):
            raise XI2Error("existing decoded raw differs from the new exact decode")
        replay = SERIALIZED / "terminal.raw.repeat.u8"
        if replay.exists() and sha256_file(replay) != sha256_file(partial):
            raise XI2Error("existing decoded repeat differs from the new exact decode")
        if replay.exists():
            partial.unlink()
        else:
            os.replace(partial, replay)
        replay_record = file_record(replay)
    else:
        os.replace(partial, destination)
    decoded_record = file_record(destination)
    encode_result = json.loads((SERIALIZED / "terminal.encode.json").read_text(encoding="utf-8"))
    expected_raw = EXPECTED_SHA256[CONTROL_RAW]
    if decoded_record["bytes"] != PIXELS or decoded_record["sha256"] != expected_raw:
        raise XI2Error("XI2 decoded tokens differ from the canonical raw partition")
    primary = encode_result["primary"]
    if logit_digest.hexdigest() != primary["logit_sha256"]:
        raise XI2Error("encoder and decoder logit hashes differ")
    if context_digest.hexdigest() != primary["context_payload_sha256"]:
        raise XI2Error("encoder and decoder derived-context hashes differ")
    result = {
        "schema": "ddm_xi2_decode_result.v1",
        "decoded_raw": decoded_record,
        "decoded_raw_repeat": replay_record,
        "verified_exact": True,
        "logit_hash_encode_decode_equal": True,
        "derived_context_hash_encode_decode_equal": True,
        "decode_side_derivation": (
            "previous exact decoded partition + already-counted pose row -> tac.lie SE3 -> "
            "XI1 class composite; frame 0 zero; no retained context sidecar consumed"
        ),
        "elapsed_seconds": time.time() - started,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.decode.json", result)
    return result


def promotion_passes(token_bytes: int) -> bool:
    """Return the preregistered strict-more-than-2% byte verdict."""

    return int(token_bytes) <= PROMOTION_MAX_BYTES


def finalize_stage() -> dict[str, Any]:
    pack = json.loads((SERIALIZED / "terminal.pack.json").read_text(encoding="utf-8"))
    encode = json.loads((SERIALIZED / "terminal.encode.json").read_text(encoding="utf-8"))
    decode = json.loads((SERIALIZED / "terminal.decode.json").read_text(encoding="utf-8"))
    actual = int(encode["primary"]["range"]["bytes"])
    ratio = actual / CONTROL_RANGE_BYTES
    passes = promotion_passes(actual)
    result = {
        "schema": "ddm_xi2_full_scale_result.v1",
        "status": "PROMOTION_PASS_BYTE_ONLY" if passes else "FORMULATION_CLOSED_FULL_SCALE",
        "deliverable_table": [
            {
                "arm": "banked_cl1_unwarped_previous_partition_control",
                "range_token_bytes": CONTROL_RANGE_BYTES,
                "payload": file_record(CONTROL_RANGE),
                "decode_exact": True,
                "retrained_by_xi2": False,
            },
            {
                "arm": "xi_warped_previous_decoded_partition",
                "range_token_bytes": actual,
                "payload": encode["primary"]["range"],
                "decode_exact": decode["verified_exact"],
                "model_xz_bytes": pack["model_xz"]["bytes"],
            },
        ],
        "xi_over_control": ratio,
        "delta_token_bytes": actual - CONTROL_RANGE_BYTES,
        "falsifier": {
            "rule": "xi must beat banked full-scale control Range token bytes by more than 2%",
            "control_bytes": CONTROL_RANGE_BYTES,
            "strict_ratio_limit": PROMOTION_RATIO,
            "largest_passing_integer_bytes": PROMOTION_MAX_BYTES,
            "fires": not passes,
            "verdict_scope": "FORMULATION",
        },
        "projection_transfer_refused": "XI1's 14.6x n120 ratio is not transferred to this result",
        "context_legality": decode["decode_side_derivation"],
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(FULL_SCALE_RESULT, result)
    return result


def _host_memory_bytes() -> int | None:
    completed = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
    value = completed.stdout.strip()
    return int(value) if value.isdigit() else None


def pinned_fire_command() -> str:
    return (
        "PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 "
        ".venv/bin/python tools/safe_run.py --rss-mb 4096 --projected-gib 4 --timeout 7200 "
        "--label ddm_xi2_xi_context_n600 --status-receipt "
        "/Volumes/APDataStore/pact/ddm_xi2_20260812/run/main.safe_run.json -- "
        ".venv/bin/python tools/run_ddm_xi2_xi_context_full_scale.py --leg all --resume-from auto"
    )


def prepare_stage(xi1: ModuleType, pins: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    INPUTS.mkdir(parents=True, exist_ok=True)
    retained_inputs = {
        "pose": retain_payload(INPUTS / "pose_targets_n600_f16.bin", POSE_RAW.read_bytes()),
        "calibration": retain_payload(INPUTS / "warp_calibration_f64.bin", CALIBRATION_RAW.read_bytes()),
    }
    context_records = prepare_contexts(xi1)
    memory = {
        "status": "PASS_DERIVED_FROM_MATCHED_REAL_CONFIG",
        "authority": "derived from banked CL1 real-config safe-run plus exact tensor geometry; not a live XI2 MPS measurement",
        "banked_cl1_observed_peak_rss_mib": CONTROL_PEAK_RSS_MIB,
        "context_uint8_bytes": PIXELS,
        "target_long_bytes": PIXELS * 8,
        "context_long_bytes": PIXELS * 8,
        "safe_run_rss_limit_mib": 4096,
        "safe_run_projected_gib": 4,
        "host_memory_bytes": _host_memory_bytes(),
    }
    receipt = {
        "schema": "ddm_xi2_build_receipt.v1",
        "status": "READY_TO_FIRE",
        "runner_source": file_record(Path(__file__).resolve()),
        "architectural_choice": (
            "new XI2 runner SHA-pinned to unchanged XI1 and CL1 sources; preserves banked-control attestation "
            "while reusing CL1 topology/schedule and XI1 xi context"
        ),
        "banked_control": {
            "range": file_record(CONTROL_RANGE),
            "decoded_raw": file_record(CONTROL_RAW),
            "model_xz": file_record(CONTROL_MODEL),
            "range_token_bytes": CONTROL_RANGE_BYTES,
            "training_elapsed_seconds": CONTROL_TRAIN_SECONDS,
            "measured_seconds_per_epoch": CONTROL_SECONDS_PER_EPOCH,
        },
        "pre_fire_deliverable_table": [
            {
                "arm": "banked_cl1_unwarped_previous_partition_control",
                "range_token_bytes": CONTROL_RANGE_BYTES,
                "evidence": "MEASURED_REAL_RANGE_FULL_N600",
                "retrained_by_xi2": False,
            },
            {
                "arm": "xi_warped_previous_decoded_partition",
                "range_token_bytes": None,
                "evidence": "PENDING_MAIN_METAL",
                "xi1_selected_n120_projection_bytes": 121_100,
                "projection_warning": (
                    "5x selected-token projection from the weak n120 screen; excluded from promotion"
                ),
                "promotion_requires_at_most_bytes": PROMOTION_MAX_BYTES,
            },
        ],
        "expected_wall_clock": {
            "derived_seconds": EXPECTED_PIPELINE_SECONDS,
            "derived_minutes": EXPECTED_PIPELINE_SECONDS / 60,
            "components": {
                "training_60_epochs": CONTROL_TRAIN_SECONDS,
                "pack": CONTROL_PACK_SECONDS,
                "two_determinism_encodes": 2 * CONTROL_ENCODE_SECONDS,
                "decode": CONTROL_DECODE_SECONDS,
            },
            "boundary": "derived from banked CL1 safe-run receipts; XI2 warp overhead unmeasured",
        },
        "memory_preflight": memory,
        "storage_preflight": preflight,
        "retained_inputs": retained_inputs,
        "retained_contexts": context_records,
        "context_legality": (
            "previous decoded partition + already-counted pose bytes are receiver-known; lossless decode makes "
            "teacher forcing exact; codec derives context through causal_context_for_frame -> derive_xi_context -> "
            "XI1 warp_previous_partition/tac.lie and never consumes the retained training context"
        ),
        "pinned_fire_command": pinned_fire_command(),
        "falsifier": {
            "control_range_bytes": CONTROL_RANGE_BYTES,
            "xi_must_be_at_most_bytes": PROMOTION_MAX_BYTES,
            "strict_rule": "xi/control < 0.98",
            "failure_disposition": "FORMULATION_CLOSED_FULL_SCALE",
        },
        "input_pins": pins,
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(BUILD_RECEIPT, receipt)
    ready = {
        "schema": "ddm_xi2_ready_to_fire.v1",
        "status": "READY_TO_FIRE",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN Metal executor",
        "consumer_store": str(FULL_SCALE_RESULT),
        "fire_trigger": (
            "MAIN verifies torch.backends.mps.is_available() is true, the local-Metal lane is free, "
            "all pinned hashes and storage/governor preflights pass, then executes pinned_fire_command"
        ),
        "pinned_fire_command": pinned_fire_command(),
        "build_receipt": file_record(BUILD_RECEIPT),
    }
    atomic_json(READY_TO_FIRE, ready)
    atomic_json(QUEUE / "main_metal_fire_order.json", ready)
    return receipt


def update_state(*, leg: str, status: str, pins: dict[str, Any], preflight: dict[str, Any]) -> None:
    atomic_json(
        STATE,
        {
            "schema": "ddm_xi2_state.v1",
            "arm": "ddm_xi2",
            "leg": leg,
            "status": status,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_root": str(OUTPUT),
            "payload_policy": "retain every materialized payload with bytes and sha256",
            "input_pins": pins,
            "storage_preflight": preflight,
            "hardware": {
                "system": platform.system(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            },
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leg",
        choices=("prepare", "train", "pack", "encode", "decode", "finalize", "all"),
        required=True,
    )
    parser.add_argument(
        "--resume-from",
        default="auto",
        help="checkpoint path or 'auto' for the newest immutable per-epoch checkpoint",
    )
    parser.add_argument("--required-free-bytes", type=int, default=4 << 30)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise XI2Error("PYTHONHASHSEED=0 is required")
    if os.environ.get("TAC_ADMISSION_ENFORCE") != "1":
        raise XI2Error("TAC_ADMISSION_ENFORCE=1 is required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise XI2Error("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    if args.leg not in {"prepare", "finalize"}:
        assert_governed_admission("run_ddm_xi2_xi_context_full_scale")
    preflight = storage_preflight(args.required_free_bytes)
    xi1 = load_xi1()
    pins = pin_inputs(xi1)
    update_state(leg=args.leg, status="running", pins=pins, preflight=preflight)
    if args.leg in {"prepare", "all"}:
        prepare_stage(xi1, pins, preflight)
    if args.leg in {"train", "all"}:
        train_stage(args, xi1, pins)
    if args.leg in {"pack", "all"}:
        pack_stage(xi1)
    if args.leg in {"encode", "all"}:
        if not torch.backends.mps.is_available():
            raise XI2Error("XI2 Range encode requires live Metal")
        encode_stage(xi1)
    if args.leg in {"decode", "all"}:
        if not torch.backends.mps.is_available():
            raise XI2Error("XI2 Range decode requires live Metal")
        decode_stage(xi1)
    result: dict[str, Any] | None = None
    if args.leg in {"finalize", "all"}:
        result = finalize_stage()
    update_state(leg=args.leg, status="complete", pins=pins, preflight=preflight)
    print(json.dumps(result if result is not None else {"status": "COMPLETE", "leg": args.leg}, indent=2))


if __name__ == "__main__":
    main()
