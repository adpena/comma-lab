#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixture and bounded real-pair checks for the C2 integer-plane emitter.

This tool is deliberately not an evaluator.  It emits local, untrained receipts
for the two-plane emitter and never computes the contest action, changes a
frontier pointer, launches training, or writes into a retained run directory.

Two local-loop twins are exposed for the real-pair check:

``numpy-torch-authority``
    NumPy-fp32 emitter/lattice reference plus the frozen CPU-Torch hard oracle.

``mlx-metal-iteration``
    Pair-parallel MLX emitter and frozen MLX scorer adapter.  Exact integer
    lattice realization remains the CPU reference.  A host without a usable
    Metal device records ``BLOCKED_ENVIRONMENT_NO_METAL``; timing is never
    inferred from Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.optimization.vjp_custody import (  # noqa: E402
    VJPCustodyError,
    load_vjp_pair_row,
)
from tools.measure_uint8_lattice_feasibility import stored_npy_memmap  # noqa: E402

SCHEMA = "c2_integer_plane_emitter_measurement.v3"
AXIS = "[macOS-CPU advisory, untrained]"
POINTER = "0.1910828242 [contest-CPU Linux x86_64] UNMOVED"
SEED = 20260719
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
SCORER_SHAPE = (*SCORER_HW, 3)
FROZEN_N24 = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24,
    12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25,
)
PRIMARY_N6 = FROZEN_N24[:6]
DEFAULT_CACHE = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)
DEFAULT_UPSTREAM = Path(
    "/Users/adpena/Projects/pact/experiments/results/"
    "public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source"
)
DEFAULT_VJP_MANIFEST = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/"
    "chunk_000_010_024_composed/manifest.json"
)
EXPECTED_VJP_MANIFEST_SHA256 = (
    "3d1218a52ededc4b347ae94c5c2bf58d06d70dd8f530bec67bf9cab36ee00694"
)
DEFAULT_VJP_MANIFEST_SECONDARY = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/"
    "chunk_012_017_019_023_025_composed/manifest.json"
)
EXPECTED_VJP_MANIFEST_SECONDARY_SHA256 = (
    "200e8cfa375cbdb8154777156441ae6adadf33e75668c86cc52b816f79488e94"
)
EXPECTED_CACHE_SHA256 = (
    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
)
EXPECTED_SEGNET_SHA256 = (
    "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
)
EXPECTED_POSENET_SHA256 = (
    "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
)
EXPECTED_MODULES_SHA256 = (
    "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
)
EXPECTED_FRAME_UTILS_SHA256 = (
    "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90"
)
EXPECTED_VJP_SOURCE_HASHES = {
    "cache_sha256": EXPECTED_CACHE_SHA256,
    "frame_utils_sha256": EXPECTED_FRAME_UTILS_SHA256,
    "modules_sha256": EXPECTED_MODULES_SHA256,
    "posenet_weights_sha256": EXPECTED_POSENET_SHA256,
    "segnet_weights_sha256": EXPECTED_SEGNET_SHA256,
}
EXPECTED_VJP_TENSOR_HASH_KEYS = frozenset(
    {
        "cached_margin",
        "head_pair_norms",
        "native_margin",
        "pose_j_x",
        "pose_j_y",
        "rival",
        "seg_g_x",
        "seg_g_y",
        "seg_local_lipschitz",
        "seg_q",
        "winner",
    }
)
SACRED = Path(
    "/Users/adpena/Projects/pact/experiments/results/"
    "levelset_n600_witness_20260717T113932Z"
)
DEFAULT_FIXTURE_RECEIPT = REPO / ".omx/research/c2_integer_plane_emitter_fixture_20260719_v3.json"
DEFAULT_N24_RECEIPT = REPO / ".omx/research/c2_integer_plane_emitter_n24_advisory_20260719_v3.json"
RESEARCH_ROOT = REPO / ".omx/research"


class C2MeasureError(RuntimeError):
    """Fail-closed C2 custody, contract, parity, or oracle error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _durable_output(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if _is_relative_to(resolved, SACRED.resolve()):
        raise C2MeasureError(f"receipt may not mutate sacred run tree: {resolved}")
    research_root = RESEARCH_ROOT.resolve()
    if not _is_relative_to(resolved, research_root):
        raise C2MeasureError(
            f"receipt must stay under repository research root {research_root}: {resolved}"
        )
    return resolved


def _write_once_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    target = _durable_output(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise C2MeasureError(f"receipt already exists; preserve it and choose a new path: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise C2MeasureError(
            f"git custody query failed ({' '.join(args)}): {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _code_custody(*, include_mlx_scorer: bool) -> dict[str, Any]:
    """Bind every implementation surface used by a receipt to exact bytes."""

    paths = {
        "measurement_tool": Path(__file__).resolve(),
        "integer_plane_emitter": REPO / "src/tac/boundary_math/integer_plane_emitter.py",
        "integer_lattice_solver": REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        "npy_memmap_loader": REPO / "tools/measure_uint8_lattice_feasibility.py",
        "uint8_ste": REPO / "src/tac/quantization.py",
    }
    if include_mlx_scorer:
        paths.update(
            {
                "mlx_scorer_adapter": REPO / "src/tac/local_acceleration/mlx_scorer_adapters.py",
                "mlx_scorer_profile": REPO
                / "src/tac/local_acceleration/mlx_renderer_prefilter_profile.py",
                "mlx_roundtrip_primitives": REPO
                / "src/tac/local_acceleration/pr95_hnerv_mlx_training.py",
                "mlx_roundtrip_runtime": REPO
                / "src/tac/local_acceleration/pr95_hnerv_mlx.py",
                "mlx_roundtrip_contract": REPO
                / "src/tac/local_acceleration/pr95_hnerv_mlx_contract.py",
                "canonical_kernels": REPO
                / "src/tac/framework_agnostic/canonical_kernels.py",
                "scorer_loader": REPO / "src/tac/scorer.py",
                "vjp_custody_validator": REPO
                / "src/tac/optimization/vjp_custody.py",
                "mlx_torch_scorer_parity": REPO
                / "src/tac/local_acceleration/mlx_scorer_torch_parity.py",
            }
        )
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise C2MeasureError(f"code custody surface missing: {', '.join(sorted(missing))}")
    head = _git_output("rev-parse", "HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise C2MeasureError("git HEAD custody is not a canonical 40-hex commit")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": head,
        "git_worktree_dirty": bool(status),
        "git_status_sha256": _sha256_bytes(status.encode()),
        "source_files": {
            name: {
                "path": str(path.relative_to(REPO)),
                "sha256": _sha256_file(path),
            }
            for name, path in sorted(paths.items())
        },
    }


def _timing_summary(
    seconds: Sequence[float],
    *,
    setup_seconds: float,
    total_seconds: float,
    setup_scope: str,
    iteration_scope: str,
) -> dict[str, Any]:
    samples = [float(value) for value in seconds]
    if any(not np.isfinite(value) or value < 0.0 for value in samples):
        raise C2MeasureError("timing samples must be finite and nonnegative")
    if not np.isfinite(setup_seconds) or setup_seconds < 0.0:
        raise C2MeasureError("setup_seconds must be finite and nonnegative")
    if not np.isfinite(total_seconds) or total_seconds < 0.0:
        raise C2MeasureError("total_seconds must be finite and nonnegative")
    return {
        "setup_scope": setup_scope,
        "iteration_scope": iteration_scope,
        "setup_seconds": float(setup_seconds),
        "total_seconds": float(total_seconds),
        "completed_iterations": len(samples),
        "median_seconds_per_pair_iteration": statistics.median(samples) if samples else None,
        "p95_seconds_per_pair_iteration": (
            float(np.percentile(np.asarray(samples, dtype=np.float64), 95)) if samples else None
        ),
    }


def _select_n24_rows(requested_rows: int) -> tuple[int, ...]:
    count = int(requested_rows)
    if count < 6 or count > len(FROZEN_N24):
        raise C2MeasureError("n24-advisory requires 6..24 completed hard-oracle rows")
    return FROZEN_N24[:count]


def validate_vjp_custody(
    manifest_path: Path,
    selected_pairs: Sequence[int],
    *,
    expected_sha256: str = EXPECTED_VJP_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Validate one immutable encode-only VJP manifest and selected sidecars."""

    document = manifest_path.expanduser().resolve()
    if not _is_sha256(expected_sha256):
        raise C2MeasureError("expected VJP manifest SHA must be canonical lowercase hex")
    if not document.is_file() or _sha256_file(document) != expected_sha256:
        raise C2MeasureError("VJP custody manifest is missing or hash-mismatched")
    try:
        manifest = json.loads(document.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise C2MeasureError("VJP custody manifest is unreadable") from exc
    if manifest.get("schema") != "vjp_custody_manifest.v1":
        raise C2MeasureError("VJP custody manifest schema mismatch")
    if manifest.get("source_hashes") != EXPECTED_VJP_SOURCE_HASHES:
        raise C2MeasureError("VJP custody source_hashes mismatch")
    raw_pairs = manifest.get("pair_ids")
    if (
        not isinstance(raw_pairs, list)
        or not raw_pairs
        or any(type(value) is not int or value < 0 for value in raw_pairs)
        or len(set(raw_pairs)) != len(raw_pairs)
    ):
        raise C2MeasureError("VJP custody pair_ids must be unique nonnegative integers")
    manifest_pairs = list(raw_pairs)
    if any(type(value) is not int or value < 0 for value in selected_pairs):
        raise C2MeasureError("selected VJP pair IDs must be nonnegative integers")
    requested = list(selected_pairs)
    if len(set(requested)) != len(requested) or any(pair not in manifest_pairs for pair in requested):
        raise C2MeasureError("selected hard-oracle rows are not covered by VJP custody")
    references = manifest.get("sidecars")
    if not isinstance(references, list) or len(references) != len(manifest_pairs):
        raise C2MeasureError("VJP custody sidecar list is incomplete")
    if any(not isinstance(row, Mapping) for row in references):
        raise C2MeasureError("VJP custody sidecar row is not a mapping")
    row_pair_ids = [row.get("pair_id") for row in references]
    if row_pair_ids != manifest_pairs:
        raise C2MeasureError("VJP custody sidecar rows do not bind pair_ids in manifest order")
    for row in references:
        path_text = row.get("path")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise C2MeasureError("VJP custody sidecar path must be absolute")
        if type(row.get("bytes")) is not int or int(row["bytes"]) <= 0:
            raise C2MeasureError("VJP custody sidecar bytes must be a positive integer")
        if not _is_sha256(row.get("sha256")):
            raise C2MeasureError("VJP custody sidecar SHA is invalid")
        tensor_hashes = row.get("tensor_hashes")
        if (
            not isinstance(tensor_hashes, Mapping)
            or set(tensor_hashes) != EXPECTED_VJP_TENSOR_HASH_KEYS
            or any(not _is_sha256(value) for value in tensor_hashes.values())
        ):
            raise C2MeasureError("VJP custody tensor_hashes are incomplete or invalid")
    selected: list[dict[str, Any]] = []
    by_pair = {int(row["pair_id"]): row for row in references}
    for pair_id in requested:
        row = by_pair.get(pair_id)
        if row is None:
            raise C2MeasureError(f"VJP sidecar reference missing for pair {pair_id}")
        sidecar = Path(str(row.get("path", ""))).resolve()
        expected_bytes = int(row.get("bytes", -1))
        expected_hash = str(row.get("sha256", ""))
        if not sidecar.is_file() or sidecar.stat().st_size != expected_bytes:
            raise C2MeasureError(f"VJP sidecar byte custody mismatch for pair {pair_id}")
        if _sha256_file(sidecar) != expected_hash:
            raise C2MeasureError(f"VJP sidecar hash custody mismatch for pair {pair_id}")
        try:
            custodied_pair = load_vjp_pair_row(
                dict(row),
                manifest,
                scorer_hw=SCORER_HW,
                camera_hw=CAMERA_HW,
            )
        except (VJPCustodyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise C2MeasureError(
                f"VJP sidecar canonical tensor custody failed for pair {pair_id}: {exc}"
            ) from exc
        if int(custodied_pair.pair_id) != pair_id:
            raise C2MeasureError(f"VJP canonical loader returned wrong pair {pair_id}")
        selected.append(
            {
                "pair_id": pair_id,
                "path": str(sidecar),
                "bytes": expected_bytes,
                "sha256": expected_hash,
                "hash_rechecked": True,
                "embedded_pair_id_rechecked": True,
                "custody_json_rechecked": True,
                "tensor_bytes_rehashed": True,
                "tensor_hashes": dict(row["tensor_hashes"]),
                "role": "encode-only proposal metadata; never decoder payload or admission",
            }
        )
    return {
        "manifest": str(document),
        "manifest_sha256": expected_sha256,
        "selected_sidecars": selected,
        "decoder_serialization": False,
        "candidate_admission": False,
    }


def validate_vjp_custody_set(
    primary_manifest: Path,
    secondary_manifest: Path,
    selected_pairs: Sequence[int],
) -> dict[str, Any]:
    """Validate the explicit two-manifest contract covering frozen n24."""

    if any(type(value) is not int or value < 0 for value in selected_pairs):
        raise C2MeasureError("selected VJP pair IDs must be nonnegative integers")
    requested = tuple(selected_pairs)
    primary_pairs = tuple(pair for pair in requested if pair in FROZEN_N24[:12])
    secondary_pairs = tuple(pair for pair in requested if pair in FROZEN_N24[12:])
    if len(primary_pairs) + len(secondary_pairs) != len(requested):
        raise C2MeasureError("selected pair lies outside the frozen two-manifest VJP contract")
    manifests = [
        validate_vjp_custody(
            primary_manifest,
            primary_pairs,
            expected_sha256=EXPECTED_VJP_MANIFEST_SHA256,
        )
    ]
    manifests.append(
        validate_vjp_custody(
            secondary_manifest,
            secondary_pairs,
            expected_sha256=EXPECTED_VJP_MANIFEST_SECONDARY_SHA256,
        )
    )
    rows = [row for manifest in manifests for row in manifest["selected_sidecars"]]
    if [row["pair_id"] for row in rows] != list(requested):
        raise C2MeasureError("two-manifest VJP custody did not preserve selected pair order")
    return {
        "contract": "explicit-two-manifest-frozen-n24",
        "manifests": manifests,
        "selected_sidecars": rows,
        "decoder_serialization": False,
        "candidate_admission": False,
    }


def _metal_environment_status() -> dict[str, Any]:
    """Probe actual MLX evaluation, not importability alone."""

    started = time.monotonic()
    try:
        import mlx.core as mx
    except (ImportError, ModuleNotFoundError) as exc:
        return _mlx_probe_blocker(exc, device=None, setup_seconds=time.monotonic() - started)

    device: str | None = None
    try:
        device = str(mx.default_device())
        probe = mx.array([1.0], dtype=mx.float32)
        mx.eval(probe)
        if "gpu" not in device.lower():
            raise RuntimeError(f"MLX default device is not Metal GPU: {device}")
    except Exception as exc:
        return _mlx_probe_blocker(
            exc,
            device=device,
            setup_seconds=time.monotonic() - started,
        )
    return {
        "status": "READY_METAL",
        "device": device,
        "metal_device_ready": True,
        "fast_path_claim": False,
        "exact_numpy_parity": "PENDING_EXECUTION",
        "timing": None,
        "blocker": None,
    }


def _mlx_probe_blocker(
    exc: Exception,
    *,
    device: str | None,
    setup_seconds: float,
) -> dict[str, Any]:
    detail = f"{type(exc).__name__}: {exc}"
    lowered = detail.lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        status = "BLOCKED_ENVIRONMENT_MLX_NOT_IMPORTABLE"
    elif (
        "metal::load_device" in lowered
        or "no metal device" in lowered
        or "not metal gpu" in lowered
    ):
        status = "BLOCKED_ENVIRONMENT_NO_METAL"
    else:
        status = "BLOCKED_EXECUTION_MLX_PROBE"
    return {
        "status": status,
        "device": device,
        "metal_device_ready": False,
        "fast_path_claim": False,
        "exact_numpy_parity": "NOT_MEASURED",
        "timing": {
            "setup_scope": "MLX import, default-device, and one-array evaluation probe",
            "iteration_scope": "NOT_EXECUTED",
            "setup_seconds": float(setup_seconds),
            "total_seconds": None,
            "completed_iterations": 0,
            "median_seconds_per_pair_iteration": None,
            "p95_seconds_per_pair_iteration": None,
        },
        "blocker": detail,
    }


def _mlx_execution_blocker(exc: Exception, *, device: object) -> dict[str, Any]:
    return {
        "status": "BLOCKED_EXECUTION_MLX_METAL",
        "device": str(device),
        "metal_device_ready": True,
        "fast_path_claim": False,
        "exact_numpy_parity": "NOT_CLOSED_EXECUTION_ERROR",
        "scorer_parity": "NOT_MEASURED",
        "timing": None,
        "blocker": f"{type(exc).__name__}: {exc}",
    }


def _load_real_cache(path: Path) -> tuple[dict[str, np.memmap], str]:
    if not path.is_file():
        raise C2MeasureError(f"C2 advisory cache is missing: {path}")
    cache_sha256 = _sha256_file(path)
    if cache_sha256 != EXPECTED_CACHE_SHA256:
        raise C2MeasureError("C2 advisory cache SHA custody mismatch")
    fields = {name: stored_npy_memmap(path, name) for name in ("n_pairs", "gt_f0", "gt_f1")}
    if int(np.asarray(fields["n_pairs"]).reshape(())) != 600:
        raise C2MeasureError("C2 advisory requires the real n600 cache")
    for name in ("gt_f0", "gt_f1"):
        if fields[name].shape != (600, *CAMERA_HW, 3) or fields[name].dtype != np.uint8:
            raise C2MeasureError(f"{name} cache geometry/dtype mismatch")
    return fields, cache_sha256


def _load_distortion_net(upstream: Path, threads: int) -> tuple[Any, Any, dict[str, str]]:
    upstream = upstream.resolve()
    modules_path = upstream / "modules.py"
    hashes = {
        "modules.py": _sha256_file(modules_path),
        "frame_utils.py": _sha256_file(upstream / "frame_utils.py"),
        "posenet.safetensors": _sha256_file(upstream / "models/posenet.safetensors"),
        "segnet.safetensors": _sha256_file(upstream / "models/segnet.safetensors"),
    }
    expected_hashes = {
        "modules.py": EXPECTED_MODULES_SHA256,
        "frame_utils.py": EXPECTED_FRAME_UTILS_SHA256,
        "posenet.safetensors": EXPECTED_POSENET_SHA256,
        "segnet.safetensors": EXPECTED_SEGNET_SHA256,
    }
    mismatched = sorted(name for name, expected in expected_hashes.items() if hashes[name] != expected)
    if mismatched:
        raise C2MeasureError("frozen scorer source/weight SHA custody mismatch: " + ", ".join(mismatched))
    retained = [entry for entry in sys.path if Path(entry or ".").resolve() != upstream]
    sys.path[:] = [str(upstream), *retained]
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    if Path(sys.modules["modules"].__file__).resolve() != modules_path:
        raise C2MeasureError("frozen modules.py import custody mismatch")
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = DistortionNet().eval().to("cpu")
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, torch, hashes


def _distortion_outputs(model: Any, torch: Any, f0: np.ndarray, f1: np.ndarray) -> tuple[Any, Any]:
    pair = torch.from_numpy(np.stack((f0, f1), axis=0)[None]).float()
    with torch.inference_mode():
        return model(pair)


def _distortion(model: Any, candidate: tuple[Any, Any], source: tuple[Any, Any]) -> dict[str, float]:
    return {
        "d_pose": float(model.posenet.compute_distortion(candidate[0], source[0]).item()),
        "d_seg": float(model.segnet.compute_distortion(candidate[1], source[1]).item()),
    }


def _exact_source_plane(operator: DisjointResizeOperator, frame: np.ndarray) -> np.ndarray:
    numerator, denominator = operator.apply_numerators(frame)
    plane = np.clip(np.rint(numerator.astype(np.float64) / denominator), 0.0, 255.0).astype(np.uint8)
    if plane.shape != SCORER_SHAPE:
        raise C2MeasureError("exact source projection returned wrong scorer geometry")
    return plane


def _fresh_untrained_planes(
    base: np.ndarray, *, seed: int
) -> tuple[np.ndarray, dict[str, Any], Any, Any]:
    """Use Unit A's quotient residual and NumPy reference for shape-parallel expansion."""

    from tac.boundary_math.integer_plane_emitter import (
        CapacitySignature,
        QuotientResidualState,
        StructuredEmitterState,
        numpy_uint8,
    )

    structured = StructuredEmitterState.from_base(
        base.astype(np.float32, copy=False), residual_width=4
    )
    residual = QuotientResidualState.fresh(structured, seed=int(seed), scale=0.25)
    planes = numpy_uint8(structured, residual, require_distinct_planes=True)
    capacity = CapacitySignature.from_states(structured, residual)
    return planes, {
        "seed": int(seed),
        "structured_state_sha256": _sha256_bytes(
            _canonical(
                {
                    "base": _sha256_array(structured.base),
                    "coordinate_basis": _sha256_array(structured.coordinate_basis),
                    "topology_id": structured.topology_id,
                }
            )
        ),
        "quotient_residual_sha256": _sha256_bytes(
            _canonical(
                {
                    "pair_plane_codes": _sha256_array(residual.pair_plane_codes),
                    "shared_rgb_head": _sha256_array(residual.shared_rgb_head),
                    "seed": residual.seed,
                }
            )
        ),
        "capacity_signature": capacity.__dict__,
        "capacity_signature_sha256": capacity.sha256,
        "pair_parallel": True,
        "cross_pair_autoregression": False,
        "learned_state": "quotient residual only; fresh deterministic init; untrained",
    }, structured, residual


def _realize_plane_pair(
    operator: DisjointResizeOperator,
    pair_id: int,
    planes: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for plane_index in range(2):
        target = np.ascontiguousarray(planes[plane_index])
        frame = realize_factor2_uint8_scorer_plane(operator, target)
        proof = verify_factor2_uint8_scorer_plane(operator, frame, target)
        if not proof.numerator_exact or not proof.certified_exact:
            raise C2MeasureError(f"pair={pair_id} plane={plane_index} exact numerator verification failed")
        numerator, denominator = operator.apply_numerators(frame)
        target_numerator = target.astype(np.int64) * denominator
        maximum_error = int(np.max(np.abs(numerator.astype(np.int64) - target_numerator), initial=0))
        if maximum_error != 0:
            raise C2MeasureError("factor-2 proof claimed exact with nonzero numerator error")
        frames.append(frame)
        rows.append(
            {
                "pair_id": int(pair_id),
                "plane_index": plane_index,
                "target_plane_sha256": _sha256_array(target),
                "camera_frame_sha256": _sha256_array(frame),
                "target_numerator_sha256": _sha256_array(target_numerator),
                "realized_numerator_sha256": _sha256_array(numerator),
                "denominator": int(denominator),
                "max_abs_numerator_error": maximum_error,
                "exact": True,
            }
        )
    return np.stack(frames, axis=0), rows


def run_fixture(*, output: Path, seed: int = SEED) -> dict[str, Any]:
    started = time.monotonic()
    proof = _fixture_proof_bundle(seed=int(seed))
    required = (
        "numpy_torch_bytes_equal",
        "in_range_gradient_nonzero",
        "saturated_gradient_zero",
        "planes_distinct",
        "all_factor2_numerators_exact",
        "fixed_capacity_basis_ab",
    )
    if not all(bool(proof.get(name)) for name in required):
        raise C2MeasureError("Unit A fixture bundle did not close every required proof")
    mlx = _metal_environment_status()
    if mlx["status"] == "READY_METAL":
        try:
            mlx = _run_mlx_fixture_twin(seed=int(seed))
        except Exception as exc:  # execution failure is distinct from environment absence
            mlx = _mlx_execution_blocker(exc, device=mlx["device"])
    receipt = {
        "schema": SCHEMA,
        "mode": "fixture",
        "authority": {
            "axis": AXIS,
            "score_claim": False,
            "pointer": POINTER,
            "training": False,
            "promotion_authority": False,
        },
        "seed": int(seed),
        "proof": proof,
        "mlx_metal_iteration": mlx,
        "runtime_seconds": float(time.monotonic() - started),
        "code": _code_custody(include_mlx_scorer=False),
        "written_at_utc": datetime.now(UTC).isoformat(),
    }
    receipt["artifact"] = _write_once_json(output, receipt)
    return receipt


def _fixture_proof_bundle(*, seed: int) -> dict[str, Any]:
    """Exercise the production C2 surfaces on one deterministic full-geometry fixture."""

    import torch
    from safetensors.numpy import load_file

    from tac.boundary_math.integer_plane_emitter import (
        CapacitySignature,
        FixedCapacityBasisAB,
        QuotientResidualState,
        SignFixedU4Basis,
        StructuredEmitterState,
        numpy_uint8,
        realize_all_factor2,
        torch_uint8,
        torch_uint8_bytes,
    )

    base = np.empty((1, 2, *SCORER_SHAPE), dtype=np.float32)
    base[:, 0] = np.float32(96.0)
    base[:, 1] = np.float32(160.0)
    structured = StructuredEmitterState.from_base(base, residual_width=4)
    residual = QuotientResidualState.fresh(structured, seed=seed, scale=0.25)
    numpy_bytes = numpy_uint8(structured, residual, require_distinct_planes=True)
    codes = torch.tensor(residual.pair_plane_codes, dtype=torch.float32, requires_grad=True)
    head = torch.tensor(residual.shared_rgb_head, dtype=torch.float32, requires_grad=True)
    torch_bytes = torch_uint8_bytes(
        structured, codes, head, require_distinct_planes=True
    )
    rounded = torch_uint8(structured, codes, head)
    rounded.sum().backward()
    in_range_nonzero = bool(
        codes.grad is not None
        and head.grad is not None
        and torch.count_nonzero(codes.grad).item() > 0
        and torch.count_nonzero(head.grad).item() > 0
    )

    saturated_base = np.empty_like(base)
    saturated_base[:, 0] = np.float32(-10.0)
    saturated_base[:, 1] = np.float32(265.0)
    saturated = StructuredEmitterState.from_base(saturated_base, residual_width=4)
    sat_codes = torch.tensor(residual.pair_plane_codes, dtype=torch.float32, requires_grad=True)
    sat_head = torch.tensor(residual.shared_rgb_head, dtype=torch.float32, requires_grad=True)
    torch_uint8(saturated, sat_codes, sat_head).sum().backward()
    saturated_zero = bool(
        sat_codes.grad is not None
        and sat_head.grad is not None
        and torch.count_nonzero(sat_codes.grad).item() == 0
        and torch.count_nonzero(sat_head.grad).item() == 0
    )

    lattice = realize_all_factor2(numpy_bytes)
    lattice_exact = all(row.numerator_exact and row.certified_exact for row in lattice.rows)
    head_file = DEFAULT_UPSTREAM / "models/segnet.safetensors"
    head_weights = load_file(head_file)["segmentation_head.0.weight"].astype(np.float32)
    u4 = SignFixedU4Basis.from_head_weight(
        head_weights,
        frozen_head_sha256=_sha256_file(head_file),
    )
    # Build the policy arms independently from byte-identical residual-state
    # snapshots.  Objective selection is downstream of emission, so the fixed-
    # capacity invariant is two independent build products, not x == x.copy().
    raw_structured = StructuredEmitterState(
        base=np.array(structured.base, copy=True),
        coordinate_basis=np.array(structured.coordinate_basis, copy=True),
    )
    raw_residual = QuotientResidualState(
        np.array(residual.pair_plane_codes, copy=True),
        np.array(residual.shared_rgb_head, copy=True),
        residual.seed,
    )
    u4_structured = StructuredEmitterState(
        base=np.array(structured.base, copy=True),
        coordinate_basis=np.array(structured.coordinate_basis, copy=True),
    )
    u4_residual = QuotientResidualState(
        np.array(residual.pair_plane_codes, copy=True),
        np.array(residual.shared_rgb_head, copy=True),
        residual.seed,
    )
    raw_harness = FixedCapacityBasisAB(
        raw_structured, raw_residual, u4, hard_oracle_requested=True
    )
    u4_harness = FixedCapacityBasisAB(
        u4_structured, u4_residual, u4, hard_oracle_requested=True
    )
    raw_arm_bytes = numpy_uint8(raw_structured, raw_residual, require_distinct_planes=True)
    u4_arm_bytes = numpy_uint8(u4_structured, u4_residual, require_distinct_planes=True)
    raw_harness.require_same_emitted_bytes(raw_arm_bytes, u4_arm_bytes)
    capacity = CapacitySignature.from_states(structured, residual)
    logits = np.arange(10, dtype=np.float32).reshape(2, 5)
    raw_view = raw_harness.objective_view("raw_centered", logits)
    margin_view = u4_harness.objective_view("sign_fixed_u4_pair_margin", logits)
    raw_capacity = raw_harness.capacity_signature
    u4_capacity = u4_harness.capacity_signature
    return {
        "numpy_torch_bytes_equal": bool(np.array_equal(numpy_bytes, torch_bytes)),
        "in_range_gradient_nonzero": in_range_nonzero,
        "saturated_gradient_zero": saturated_zero,
        "planes_distinct": bool(not np.array_equal(numpy_bytes[0, 0], numpy_bytes[0, 1])),
        "all_factor2_numerators_exact": bool(lattice_exact),
        "fixed_capacity_basis_ab": bool(
            raw_capacity == capacity
            and u4_capacity == capacity
            and np.array_equal(raw_arm_bytes, u4_arm_bytes)
            and raw_view.shape == (2, 4)
            and margin_view.shape == (2, 10)
        ),
        "basis_ab_build_invariant": {
            "raw_arm_objective": "raw_centered",
            "u4_arm_objective": "sign_fixed_u4_pair_margin",
            "independent_state_objects": bool(
                raw_structured is not u4_structured and raw_residual is not u4_residual
            ),
            "raw_arm_emitted_sha256": _sha256_array(raw_arm_bytes),
            "u4_arm_emitted_sha256": _sha256_array(u4_arm_bytes),
            "emitted_bytes_equal": bool(np.array_equal(raw_arm_bytes, u4_arm_bytes)),
            "raw_capacity_sha256": raw_capacity.sha256,
            "u4_capacity_sha256": u4_capacity.sha256,
            "capacity_equal": bool(raw_capacity == u4_capacity),
            "claim_scope": "build invariant only; no objective efficacy or score claim",
        },
        "numpy_bytes_sha256": _sha256_array(numpy_bytes),
        "torch_bytes_sha256": _sha256_array(torch_bytes),
        "camera_pair_sha256": _sha256_array(lattice.camera_planes),
        "lattice_rows": [row.__dict__ for row in lattice.rows],
        "capacity_signature": capacity.__dict__,
        "capacity_signature_sha256": capacity.sha256,
        "frozen_segnet_head_file_sha256": _sha256_file(head_file),
        "u4_singular_values": [float(value) for value in u4.singular_values],
        "all_ten_pair_margins": int(margin_view.shape[-1]),
        "sigma_division": False,
    }


def _run_mlx_fixture_twin(*, seed: int) -> dict[str, Any]:
    """Execute MLX byte and saturation-gradient proofs on an actual Metal device."""

    import mlx.core as mx

    from tac.boundary_math.integer_plane_emitter import (
        QuotientResidualState,
        StructuredEmitterState,
        mlx_uint8,
        numpy_uint8,
    )

    started = time.monotonic()
    device = str(mx.default_device())
    if "gpu" not in device.lower():
        raise C2MeasureError(f"MLX fixture execution left the Metal GPU context: {device}")
    base = np.empty((1, 2, *SCORER_SHAPE), dtype=np.float32)
    base[:, 0] = np.float32(96.0)
    base[:, 1] = np.float32(160.0)
    structured = StructuredEmitterState.from_base(base, residual_width=4)
    residual = QuotientResidualState.fresh(structured, seed=seed, scale=0.25)
    codes = mx.array(residual.pair_plane_codes, dtype=mx.float32)
    head = mx.array(residual.shared_rgb_head, dtype=mx.float32)
    setup_seconds = time.monotonic() - started
    execution_started = time.monotonic()

    def loss_fn(code_value: Any, head_value: Any) -> Any:
        return mx.sum(mlx_uint8(structured, code_value, head_value))

    emitted = mlx_uint8(structured, codes, head)
    loss, gradients = mx.value_and_grad(loss_fn, argnums=(0, 1))(codes, head)
    code_gradient, head_gradient = gradients
    mx.eval(emitted, loss, code_gradient, head_gradient)
    mlx_bytes = np.asarray(emitted, dtype=np.float32).astype(np.uint8)
    numpy_bytes = numpy_uint8(structured, residual, require_distinct_planes=True)
    in_range_nonzero = bool(
        np.count_nonzero(np.asarray(code_gradient, dtype=np.float32)) > 0
        and np.count_nonzero(np.asarray(head_gradient, dtype=np.float32)) > 0
    )

    saturated_base = np.empty_like(base)
    saturated_base[:, 0] = np.float32(-10.0)
    saturated_base[:, 1] = np.float32(265.0)
    saturated = StructuredEmitterState.from_base(saturated_base, residual_width=4)

    def saturated_loss_fn(code_value: Any, head_value: Any) -> Any:
        return mx.sum(mlx_uint8(saturated, code_value, head_value))

    saturated_loss, saturated_gradients = mx.value_and_grad(
        saturated_loss_fn, argnums=(0, 1)
    )(codes, head)
    saturated_code_gradient, saturated_head_gradient = saturated_gradients
    mx.eval(saturated_loss, saturated_code_gradient, saturated_head_gradient)
    saturated_zero = bool(
        np.count_nonzero(np.asarray(saturated_code_gradient, dtype=np.float32)) == 0
        and np.count_nonzero(np.asarray(saturated_head_gradient, dtype=np.float32)) == 0
    )
    bytes_equal = bool(np.array_equal(mlx_bytes, numpy_bytes))
    if not bytes_equal or not in_range_nonzero or not saturated_zero:
        raise C2MeasureError(
            "MLX fixture failed byte parity or saturation-aware gradient proof"
        )
    execution_seconds = time.monotonic() - execution_started
    return {
        "status": "MEASURED_MLX_METAL_FIXTURE_PARITY_PASS",
        "device": device,
        "metal_device_ready": True,
        "fast_path_claim": False,
        "exact_numpy_parity": "PASS_EMITTER_BYTES",
        "gradient_parity": {
            "in_range_nonzero": in_range_nonzero,
            "saturated_zero": saturated_zero,
        },
        "mlx_bytes_sha256": _sha256_array(mlx_bytes),
        "numpy_bytes_sha256": _sha256_array(numpy_bytes),
        "timing": _timing_summary(
            [execution_seconds],
            setup_seconds=setup_seconds,
            total_seconds=setup_seconds + execution_seconds,
            setup_scope="fixture tensors and deterministic residual initialization",
            iteration_scope="MLX emitter bytes plus in-range and saturated gradient transforms",
        ),
        "blocker": None,
    }


def run_n24_advisory(
    *,
    output: Path,
    cache: Path,
    upstream: Path,
    vjp_manifest: Path,
    completed_rows: int,
    cpu_threads: int,
    vjp_manifest_secondary: Path = DEFAULT_VJP_MANIFEST_SECONDARY,
    seed: int = SEED,
) -> dict[str, Any]:
    custody_started = time.monotonic()
    selected = _select_n24_rows(completed_rows)
    if tuple(selected[:6]) != PRIMARY_N6:
        raise C2MeasureError("primary n6 custody/order changed")
    custody = validate_vjp_custody_set(
        vjp_manifest,
        vjp_manifest_secondary,
        selected,
    )
    vjp_custody_seconds = time.monotonic() - custody_started
    cache = cache.expanduser().resolve()
    upstream = upstream.expanduser().resolve()
    setup_started = time.monotonic()
    cache_fields, cache_sha256 = _load_real_cache(cache)
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    model, torch, scorer_hashes = _load_distortion_net(upstream, cpu_threads)
    bases = []
    source_pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pair_id in selected:
        source0 = np.asarray(cache_fields["gt_f0"][pair_id], dtype=np.uint8).copy()
        source1 = np.asarray(cache_fields["gt_f1"][pair_id], dtype=np.uint8).copy()
        source_pairs[pair_id] = (source0, source1)
        bases.append(np.stack((_exact_source_plane(operator, source0), _exact_source_plane(operator, source1))))
    base = np.stack(bases).astype(np.float32)
    emitted, emitter_custody, structured, residual = _fresh_untrained_planes(base, seed=seed)
    if emitted.shape != (len(selected), 2, *SCORER_SHAPE) or emitted.dtype != np.uint8:
        raise C2MeasureError("fresh emitter violated exact n-pair/two-plane uint8 contract")
    setup_seconds = time.monotonic() - setup_started
    loop_started = time.monotonic()
    pair_seconds: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    all_plane_proofs: list[dict[str, Any]] = []
    for row_index, pair_id in enumerate(selected):
        pair_started = time.monotonic()
        source0, source1 = source_pairs[pair_id]
        source_outputs = _distortion_outputs(model, torch, source0, source1)
        realized, plane_proofs = _realize_plane_pair(operator, pair_id, emitted[row_index])
        candidate_outputs = _distortion_outputs(model, torch, realized[0], realized[1])
        hard = _distortion(model, candidate_outputs, source_outputs)
        elapsed = time.monotonic() - pair_started
        pair_seconds.append(elapsed)
        all_plane_proofs.extend(plane_proofs)
        pair_rows.append(
            {
                "pair_id": int(pair_id),
                "source_f0_sha256": _sha256_array(source0),
                "source_f1_sha256": _sha256_array(source1),
                "emitted_pair_sha256": _sha256_array(emitted[row_index]),
                "realized_camera_pair_sha256": _sha256_array(realized),
                "hard_oracle": hard,
                "runtime_seconds": elapsed,
            }
        )
    cpu_timing = _timing_summary(
        pair_seconds,
        setup_seconds=setup_seconds,
        total_seconds=time.monotonic() - loop_started + setup_seconds,
        setup_scope=(
            "cache SHA/geometry validation, frozen CPU-Torch scorer load, exact source-plane "
            "projection, and one pair-parallel untrained emitter expansion"
        ),
        iteration_scope=(
            "per pair: one source-pair frozen DistortionNet forward (PoseNet+SegNet), two exact "
            "factor-2 camera realizations, one candidate-pair DistortionNet forward, and hard "
            "d_seg/d_pose reductions"
        ),
    )
    mlx_status = _metal_environment_status()
    if mlx_status["status"] == "READY_METAL":
        try:
            mlx_status = _run_mlx_metal_twin(
                emitted=emitted,
                structured=structured,
                residual=residual,
                pair_ids=selected,
                source_pairs=[source_pairs[pair] for pair in selected],
                cpu_pair_rows=pair_rows,
                upstream=upstream,
                torch_model=model,
            )
        except Exception as exc:  # retain the environment/execution distinction
            mlx_status = _mlx_execution_blocker(exc, device=mlx_status["device"])
    receipt = {
        "schema": SCHEMA,
        "mode": "n24-advisory",
        "authority": {
            "axis": AXIS,
            "score_claim": False,
            "contest_score_computed": False,
            "pointer": POINTER,
            "training": False,
            "promotion_authority": False,
            "verdict_scope": "fresh untrained C2 emitter on bounded real-cache hard-oracle rows only",
        },
        "selection": {
            "frozen_n24_pair_ids": list(FROZEN_N24),
            "completed_pair_ids": list(selected),
            "minimum_hard_oracle_rows": 6,
            "completed_hard_oracle_rows": len(pair_rows),
        },
        "inputs": {
            "cache": str(cache),
            "cache_sha256": cache_sha256,
            "upstream": str(upstream),
            "scorer_hashes": scorer_hashes,
            "vjp_custody": custody,
            "vjp_custody_timing": {
                "seconds": float(vjp_custody_seconds),
                "scope": (
                    "both frozen-n24 manifest hashes/source hashes plus every selected sidecar "
                    "file hash, embedded pair/custody_json, tensor byte/hash, shape/finiteness, "
                    "active-arrangement, and VJP-factorization recheck"
                ),
            },
        },
        "emitter": emitter_custody,
        "geometry": {
            "camera_hwc": [*CAMERA_HW, 3],
            "scorer_hwc": [*SCORER_HW, 3],
            "planes_per_pair": 2,
        },
        "numpy_torch_authority": {
            "status": "MEASURED_ADVISORY_UNTRAINED",
            "timing": cpu_timing,
            "pairs": pair_rows,
            "factor2_plane_proofs": all_plane_proofs,
        },
        "mlx_metal_iteration": mlx_status,
        "receiver_expansion": {
            "pair_parallel": True,
            "plane_parallel": True,
            "cross_pair_autoregression": False,
            "receiver_targets": ["4-core CPU pool", "contest CUDA/T4"],
        },
        "code": _code_custody(include_mlx_scorer=True),
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_threads": int(cpu_threads),
        },
        "written_at_utc": datetime.now(UTC).isoformat(),
    }
    receipt["artifact"] = _write_once_json(output, receipt)
    return receipt


def _bind_mlx_cpu_distortion_diagnostic(
    mlx_rows: Sequence[Mapping[str, Any]],
    cpu_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind same-pair downstream distortions without minting a parity verdict.

    Scorer parity is decided separately on the scorer outputs by the canonical
    ``mlx_scorer_torch_parity`` gate.  These final-distortion deltas remain a
    useful diagnostic, but they have no independent tolerance or PASS token.
    """

    cpu_by_pair = {int(row["pair_id"]): row for row in cpu_rows}
    if len(cpu_by_pair) != len(cpu_rows):
        raise C2MeasureError("CPU-Torch parity rows contain duplicate pair IDs")
    bound: list[dict[str, Any]] = []
    for mlx_row in mlx_rows:
        pair_id = int(mlx_row["pair_id"])
        cpu_row = cpu_by_pair.get(pair_id)
        if cpu_row is None:
            raise C2MeasureError(f"MLX pair {pair_id} has no CPU-Torch authority row")
        cpu_hard = cpu_row.get("hard_oracle")
        if not isinstance(cpu_hard, Mapping):
            raise C2MeasureError(f"CPU-Torch pair {pair_id} has no hard-oracle mapping")
        d_seg_delta = abs(float(mlx_row["d_seg"]) - float(cpu_hard["d_seg"]))
        d_pose_delta = abs(float(mlx_row["d_pose"]) - float(cpu_hard["d_pose"]))
        bound.append(
            {
                **dict(mlx_row),
                "cpu_torch_d_seg": float(cpu_hard["d_seg"]),
                "cpu_torch_d_pose": float(cpu_hard["d_pose"]),
                "d_seg_abs_delta": d_seg_delta,
                "d_pose_abs_delta": d_pose_delta,
                "comparison_authority": (
                    "diagnostic only; canonical per-output scorer parity manifest "
                    "is the sole MLX/Torch parity verdict"
                ),
            }
        )
    if [row["pair_id"] for row in bound] != [int(row["pair_id"]) for row in cpu_rows]:
        raise C2MeasureError("MLX/CPU-Torch parity rows differ in pair IDs or order")
    return bound


def _run_mlx_metal_twin(
    *,
    emitted: np.ndarray,
    structured: Any,
    residual: Any,
    pair_ids: Sequence[int],
    source_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    cpu_pair_rows: Sequence[Mapping[str, Any]],
    upstream: Path,
    torch_model: Any,
) -> dict[str, Any]:
    """Run the MLX scorer twin only after an actual Metal probe succeeds.

    Exact lattice solve is intentionally not replaced by fused-R: the latter is
    a different render/roundtrip operator.  The current implementation times
    the shape-parallel emitted scorer planes and MLX scorer forward; the CPU
    exact-lattice proof remains the authority row.
    """

    import mlx.core as mx

    from tac.boundary_math.integer_plane_emitter import mlx_uint8
    from tac.local_acceleration.mlx_renderer_prefilter_profile import (
        _pose_distortion,
        _scorer_inputs_from_pair_rgb255,
        _segnet_argmax_distortion_nhwc,
    )
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        nhwc_to_nchw,
        temporary_mlx_device,
    )
    from tac.local_acceleration.mlx_scorer_torch_parity import (
        MLXTorchParityThresholds,
        build_mlx_scorer_torch_parity_manifest_from_outputs,
        run_torch_distortion_scorer_nchw,
    )
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        resize_nhwc_align_corners_false,
        rgb_to_yuv6_mlx,
    )

    if not (
        len(pair_ids) == len(source_pairs) == len(cpu_pair_rows) == int(emitted.shape[0])
    ):
        raise C2MeasureError("MLX twin pair IDs, sources, CPU rows, and emitted rows disagree")
    total_started = time.monotonic()
    setup_started = total_started
    with temporary_mlx_device("gpu"):
        device_in_context = str(mx.default_device())
        if "gpu" not in device_in_context.lower():
            raise C2MeasureError(
                f"MLX twin did not enter requested Metal GPU context: {device_in_context}"
            )
        mlx_emitted = mlx_uint8(
            structured,
            mx.array(residual.pair_plane_codes, dtype=mx.float32),
            mx.array(residual.shared_rgb_head, dtype=mx.float32),
        )
        mx.eval(mlx_emitted)
        mlx_emitted_bytes = np.asarray(mlx_emitted, dtype=np.float32).astype(np.uint8)
        if not np.array_equal(mlx_emitted_bytes, emitted):
            raise C2MeasureError("MLX emitter bytes diverge from NumPy-fp32 authority")
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(upstream, device="cpu")
        operator = DisjointResizeOperator.build(
            camera_h=CAMERA_HW[0],
            camera_w=CAMERA_HW[1],
            scorer_h=SCORER_HW[0],
            scorer_w=SCORER_HW[1],
        )
        setup_seconds = time.monotonic() - setup_started
        samples: list[float] = []
        rows: list[dict[str, Any]] = []
        for index, (pair_id, (source0, source1)) in enumerate(
            zip(pair_ids, source_pairs, strict=True)
        ):
            pair_started = time.monotonic()
            realized_pair, _ = _realize_plane_pair(
                operator, int(pair_id), mlx_emitted_bytes[index]
            )
            candidate = mx.array(realized_pair[None].astype(np.float32))
            reference = mx.array(np.stack((source0, source1))[None].astype(np.float32))
            cand_pose, cand_seg = _scorer_inputs_from_pair_rgb255(
                candidate,
                resize_nhwc_align_corners_false=resize_nhwc_align_corners_false,
                rgb_to_yuv6_mlx=rgb_to_yuv6_mlx,
            )
            ref_pose, ref_seg = _scorer_inputs_from_pair_rgb255(
                reference,
                resize_nhwc_align_corners_false=resize_nhwc_align_corners_false,
                rgb_to_yuv6_mlx=rgb_to_yuv6_mlx,
            )
            candidate_outputs = adapter(cand_pose, cand_seg)
            reference_outputs = adapter(ref_pose, ref_seg)
            mx.eval(
                *candidate_outputs["posenet"].values(),
                candidate_outputs["segnet"],
                *reference_outputs["posenet"].values(),
                reference_outputs["segnet"],
            )
            candidate_mlx_outputs = {
                "posenet": {
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in candidate_outputs["posenet"].items()
                },
                "segnet": nhwc_to_nchw(
                    np.asarray(candidate_outputs["segnet"], dtype=np.float32)
                ),
            }
            reference_mlx_outputs = {
                "posenet": {
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in reference_outputs["posenet"].items()
                },
                "segnet": nhwc_to_nchw(
                    np.asarray(reference_outputs["segnet"], dtype=np.float32)
                ),
            }
            cand_pose_np = candidate_mlx_outputs["posenet"]["pose"]
            ref_pose_np = reference_mlx_outputs["posenet"]["pose"]
            cand_seg_np = np.asarray(candidate_outputs["segnet"], dtype=np.float32)
            ref_seg_np = np.asarray(reference_outputs["segnet"], dtype=np.float32)
            candidate_torch_outputs = run_torch_distortion_scorer_nchw(
                torch_model,
                nhwc_to_nchw(np.asarray(cand_pose, dtype=np.float32)),
                nhwc_to_nchw(np.asarray(cand_seg, dtype=np.float32)),
            )
            reference_torch_outputs = run_torch_distortion_scorer_nchw(
                torch_model,
                nhwc_to_nchw(np.asarray(ref_pose, dtype=np.float32)),
                nhwc_to_nchw(np.asarray(ref_seg, dtype=np.float32)),
            )
            thresholds = MLXTorchParityThresholds()
            candidate_parity = build_mlx_scorer_torch_parity_manifest_from_outputs(
                torch_outputs=candidate_torch_outputs,
                mlx_outputs=candidate_mlx_outputs,
                thresholds=thresholds,
                run_id=f"c2_pair_{int(pair_id)}_candidate",
                device_type="gpu",
                n_samples=1,
                gpu_research_signal_allowed=True,
            )
            reference_parity = build_mlx_scorer_torch_parity_manifest_from_outputs(
                torch_outputs=reference_torch_outputs,
                mlx_outputs=reference_mlx_outputs,
                thresholds=thresholds,
                run_id=f"c2_pair_{int(pair_id)}_reference",
                device_type="gpu",
                n_samples=1,
                gpu_research_signal_allowed=True,
            )
            elapsed = time.monotonic() - pair_started
            samples.append(elapsed)
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "d_pose": float(_pose_distortion(cand_pose_np, ref_pose_np)[0]),
                    "d_seg": float(_segnet_argmax_distortion_nhwc(cand_seg_np, ref_seg_np)[0]),
                    "canonical_output_parity": {
                        "candidate": candidate_parity,
                        "reference": reference_parity,
                    },
                    "runtime_seconds": elapsed,
                }
            )
        scorer_parity_pass = all(
            bool(manifest["passed"])
            for row in rows
            for manifest in row["canonical_output_parity"].values()
        )
        bound_rows = _bind_mlx_cpu_distortion_diagnostic(rows, cpu_pair_rows)
        total_seconds = time.monotonic() - total_started
    return {
        "status": (
            "MEASURED_MLX_METAL_SCORER_PARITY_PASS"
            if scorer_parity_pass
            else "MEASURED_MLX_METAL_SCORER_PARITY_FAIL"
        ),
        "device": device_in_context,
        "metal_device_ready": True,
        "fast_path_claim": False,
        "fast_path_contract": {
            "grouped_conv_adapter": "loaded; fast-path activation not independently verified",
            "metal_fused_r_operator": "NOT_USED_AS_LATTICE_SUBSTITUTE",
        },
        "exact_numpy_parity": "PASS_EMITTER_BYTES",
        "scorer_parity": "PASS" if scorer_parity_pass else "FAIL",
        "scorer_parity_contract": {
            "producer": (
                "tac.local_acceleration.mlx_scorer_torch_parity."
                "build_mlx_scorer_torch_parity_manifest_from_outputs"
            ),
            "thresholds": rows[0]["canonical_output_parity"]["candidate"]["thresholds"],
            "evaluated_surfaces": ["candidate", "reference"],
            "per_pair_manifest_count": 2,
            "authority": (
                "canonical local MLX/Torch scorer-output parity only; not contest promotion"
            ),
        },
        "timing": _timing_summary(
            samples,
            setup_seconds=setup_seconds,
            total_seconds=total_seconds,
            setup_scope=(
                "Metal-context emitter expansion and NumPy byte comparison, frozen Torch-to-MLX "
                "scorer adapter conversion, and exact factor-2 operator construction"
            ),
            iteration_scope=(
                "per actual pair ID: CPU exact factor-2 realization, MLX scorer preprocessing, "
                "candidate/reference MLX and CPU-Torch PoseNet+SegNet forwards, canonical "
                "per-output parity gates, and local distortion reductions"
            ),
        ),
        "pairs": bound_rows,
        "blocker": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--output", type=Path, default=DEFAULT_FIXTURE_RECEIPT)
    fixture.add_argument("--seed", type=int, default=SEED)
    advisory = subparsers.add_parser("n24-advisory")
    advisory.add_argument("--output", type=Path, default=DEFAULT_N24_RECEIPT)
    advisory.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    advisory.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    advisory.add_argument("--vjp-manifest", type=Path, default=DEFAULT_VJP_MANIFEST)
    advisory.add_argument(
        "--vjp-manifest-secondary",
        type=Path,
        default=DEFAULT_VJP_MANIFEST_SECONDARY,
    )
    advisory.add_argument("--completed-rows", type=int, default=6)
    advisory.add_argument("--cpu-threads", type=int, default=1)
    advisory.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "fixture":
        receipt = run_fixture(output=args.output, seed=args.seed)
    else:
        receipt = run_n24_advisory(
            output=args.output,
            cache=args.cache,
            upstream=args.upstream,
            vjp_manifest=args.vjp_manifest,
            vjp_manifest_secondary=args.vjp_manifest_secondary,
            completed_rows=args.completed_rows,
            cpu_threads=args.cpu_threads,
            seed=args.seed,
        )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
