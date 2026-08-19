#!/usr/bin/env python3
"""Receiver-realized frame-1 edit x frame-0 Schur compensation on CP135.

The run is deliberately local and advisory.  It consumes the fourteen JS6B
Q3 diagnostic proposals, renders CP135's actual signed-int12 frame-0 carrier,
and evaluates every integer compensation through the frozen CPU PoseNet.  All
materialized scorer inputs and outputs are retained on the SSD.  A proposal is
admissible only if its optimistic Seg value exceeds a conservative residual
Pose bound even before rate; otherwise the run emits a sealed non-fire order.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
UPSTREAM: Final = REPO / "upstream"
RUN_ID: Final = "ddm_qs1_frame0_schur_coupled_solve_20260813"
OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs1_20260813")
CP135_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"
)
CP135_ARCHIVE: Final = CP135_RUNTIME / "archive.zip"
CP135_ARCHIVE_SHA256: Final = (
    "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
)
CP135_RAW: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/retained/0.raw"
)
CP135_RAW_SHA256: Final = (
    "a641d1ef149f8da8f06af3da9234d6d2f6be9702c3f606b7acf838b4b298ed47"
)
CP135_STATE: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/receiver_state"
)
CP135_BASIS: Final = CP135_STATE / "pose_basis.float32.npy"
CP135_COEFFICIENTS: Final = CP135_STATE / "pose_coefficients.float32.npy"
CP135_BASE_POSE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/"
    "cp135_base_first6_n600.npy"
)
# GT LINEAGE (#1142 cure, 2026-08-19): the shipping axis (contest-CUDA T4) scores against
# DALI-decoded GT (upstream/evaluate.py:31-42 device fork). The PyAV table differs by the
# ADDITIVE constant C = MSE(dali, pyav) = 1.406151e-04 (verified at materialization), so any
# solve against the PyAV table optimizes the wrong objective. GT_POSE is the DALI table;
# the PyAV table stays available ONLY for lineage-labeled advisory comparisons.
GT_POSE: Final = CP135_BASE_POSE.with_name("gt_first6_dali_n600.npy")
GT_POSE_PYAV_ADVISORY: Final = CP135_BASE_POSE.with_name("gt_first6_n600.npy")  # GT_LINEAGE_OK: the PyAV table is bound here under an explicitly lineage-labeled name and is NOT the solve objective -- GT_POSE on the line above is the DALI table (#1142 cure, commit 809199d24f); retained only for labeled advisory comparison
JS6_BANK: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js6_seg_representation_join_20260813/"
    "proposal_bank"
)
JS6B_STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js6b_pose_screened_compile_20260813"
)
EXPERIMENT_BOOK: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book"
)
JOINT_SOLVER_SOURCE: Final = (
    EXPERIMENT_BOOK / "src/cpr1_sub4/joint_pose_solve.py"
)
BOOK_REPACK_SOURCE: Final = EXPERIMENT_BOOK / "src/cpr1_sub4/carrier_repack.py"

PAIR_COUNT: Final = 600
DIMENSIONS: Final = 12
POSE_DIMENSIONS: Final = 6
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
RAW_BYTES: Final = PAIR_COUNT * 2 * CAMERA_H * CAMERA_W * 3
UNCOMPRESSED_BYTES: Final = 37_545_489
POSE_MARGINAL_S_PER_DPOSE: Final = 603.0
RATE_S_PER_BYTE: Final = 25.0 / UNCOMPRESSED_BYTES
GN_DAMPING: Final = 0.01
MAX_CODE_STEP: Final = 32.0
NEIGHBOUR_DIMS: Final = 3
NEIGHBOUR_RADIUS: Final = 2
POSE_BATCH: Final = 8
STORAGE_EXPECTED_BYTES: Final = 32 * 1024**3
STORAGE_RESERVE_BYTES: Final = 8 * 1024**3
AXIS: Final = (
    "[macOS-CPU advisory frozen CPU-torch PoseNet; 14 retained Q3 pairs] "
    "NON-PROMOTABLE"
)


class QS1Error(RuntimeError):
    """A source pin, receiver surface, solve, retention, or resume check failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(
    path: Path, *, expected_bytes: int | None = None, expected_sha256: str | None = None
) -> dict[str, Any]:
    if not path.is_file():
        raise QS1Error(f"required source is missing: {path}")
    record = file_record(path)
    if expected_bytes is not None and record["bytes"] != expected_bytes:
        raise QS1Error(f"source byte count differs: {record}")
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise QS1Error(f"source SHA-256 differs: {record}")
    return record


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.partial")


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    partial = _atomic_path(path)
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def retain_bytes(path: Path, payload: bytes, *, executable: bool = False) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if path.is_file():
        if file_record(path) != expected:
            raise QS1Error(f"refusing to replace different retained payload: {path}")
        return expected
    partial = _atomic_path(path)
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        if executable:
            path.chmod(0o755)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    expected = {"path": str(path.resolve()), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    if path.is_file():
        if file_record(path) != expected:
            raise QS1Error(f"refusing to replace different retained JSON: {path}")
        return expected
    return atomic_json(path, value)


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    """Atomically retain an array without a RAM-sized BytesIO copy."""
    array = np.asarray(value)
    partial = _atomic_path(path)
    try:
        with partial.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        candidate = file_record(partial)
        expected = {
            "path": str(path.resolve()),
            "bytes": candidate["bytes"],
            "sha256": candidate["sha256"],
        }
        if path.is_file():
            if file_record(path) != expected:
                raise QS1Error(f"refusing to replace different retained array: {path}")
            return expected
        os.replace(partial, path)
        return file_record(path)
    finally:
        partial.unlink(missing_ok=True)


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    already = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
    remaining = max(0, STORAGE_EXPECTED_BYTES - already)
    required = remaining + STORAGE_RESERVE_BYTES
    usage = shutil.disk_usage(output)
    result = {
        "schema": "ddm_qs1_storage_preflight.v1",
        "tier": str(output.resolve()),
        "free_bytes": usage.free,
        "already_retained_bytes": already,
        "expected_total_retained_bytes": STORAGE_EXPECTED_BYTES,
        "remaining_expected_bytes": remaining,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; retain all materialized scorer payloads; never delete",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise QS1Error(f"storage preflight failed: free={usage.free}, required={required}")
    return result


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise QS1Error(f"cannot import pinned producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_posenet() -> Any:
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, posenet_sd_path
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().cpu()
    network.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


@dataclass
class CP135Surface:
    """The exact CP135 receiver's signed-int12 frame-0 actuator."""

    codes: np.ndarray
    scales: np.ndarray
    normalized_basis: Any
    selector_modes: Sequence[Any]
    selector_indices: np.ndarray | None

    @classmethod
    def load(cls) -> tuple[CP135Surface, dict[str, Any]]:
        import torch
        import torch.nn.functional as functional

        sys.path.insert(0, str(CP135_RUNTIME))
        try:
            from runtime.carrier_repack import (
                materialize_cpr1,
                split_frame0_selector_carrier,
            )
            from runtime.frame0_selector import decode_selector
            from runtime.residual_archive import read_residual_archive
        finally:
            sys.path.pop(0)
        parts = read_residual_archive(CP135_ARCHIVE)
        _, selector = split_frame0_selector_carrier(parts.carrier_blob)
        canonical = materialize_cpr1(
            parts.carrier_blob,
            SimpleNamespace(N=PAIR_COUNT, CARRIER_DIM=DIMENSIONS),
        )
        sys.path.insert(0, str(EXPERIMENT_BOOK / "src"))
        try:
            from cpr1_sub4.carrier_repack import (
                cpr1_coefficient_scales,
                decode_cpr1_coefficients,
            )
        finally:
            sys.path.pop(0)
        codes = decode_cpr1_coefficients(
            canonical, frames=PAIR_COUNT, dimensions=DIMENSIONS
        ).astype(np.int32, copy=False)
        scales = cpr1_coefficient_scales(canonical, dimensions=DIMENSIONS)
        coefficients = np.load(CP135_COEFFICIENTS, allow_pickle=False)
        if coefficients.shape != codes.shape or not np.allclose(
            coefficients, codes.astype(np.float32) * scales[None], rtol=0.0, atol=1e-7
        ):
            raise QS1Error("CP135 receiver-state coefficients do not match archive int12 codes")
        basis = torch.from_numpy(np.load(CP135_BASIS, allow_pickle=False)).float()
        basis = functional.interpolate(
            basis, size=(384, 512), mode="bicubic", align_corners=False
        )
        basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
        basis = basis / basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
        if selector is None:
            modes: Sequence[Any] = ()
            indices = None
        else:
            modes, indices = decode_selector(selector)
        pins = {
            "canonical_cpr1": {
                "bytes": len(canonical),
                "sha256": hashlib.sha256(canonical).hexdigest(),
            },
            "carrier_codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            "selector_payload": None
            if selector is None
            else {"bytes": len(selector), "sha256": hashlib.sha256(selector).hexdigest()},
        }
        return cls(codes, scales, basis, modes, indices), pins

    def render(self, codes: np.ndarray, pair: int) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        values = np.asarray(codes, dtype=np.int32)
        if values.ndim != 2 or values.shape[1] != DIMENSIONS:
            raise QS1Error(f"carrier code geometry differs: {values.shape}")
        if np.any(values < -2048) or np.any(values > 2047):
            raise QS1Error("carrier candidate exceeds signed-int12")
        coefficient = torch.from_numpy(
            np.ascontiguousarray(values.astype(np.float32) * self.scales[None])
        )
        with torch.inference_mode():
            carrier = torch.einsum("bk,kchw->bchw", coefficient, self.normalized_basis)
            carrier = carrier / math.sqrt(DIMENSIONS)
            slave_eval = (127.5 + 64.0 * carrier).clamp(0.0, 255.0).round()
            slave = functional.interpolate(
                slave_eval,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 255.0).round()
            result = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if self.selector_indices is not None:
            from runtime.frame0_selector import apply_pixel_mode

            mode_index = int(self.selector_indices[pair])
            result = apply_pixel_mode(result, self.selector_modes[mode_index])
        return np.asarray(result, dtype=np.uint8)


def pose_vectors(posenet: Any, pairs: np.ndarray) -> np.ndarray:
    import torch

    value = np.asarray(pairs)
    if value.ndim != 5 or value.shape[1:] != (2, CAMERA_H, CAMERA_W, 3):
        raise QS1Error(f"PoseNet input geometry differs: {value.shape}")
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).float()
        output = posenet(posenet.preprocess_input(tensor))["pose"][..., :POSE_DIMENSIONS]
    return output.cpu().numpy().astype(np.float32, copy=False)


def evaluate_codes(
    *,
    surface: CP135Surface,
    posenet: Any,
    codes: Sequence[np.ndarray],
    master: np.ndarray,
    pair: int,
    stage_root: Path,
) -> np.ndarray:
    """Render and score candidates in retained, resumable batches."""
    code_array = np.stack([np.asarray(item, dtype=np.int32) for item in codes])
    all_vectors: list[np.ndarray] = []
    for first in range(0, len(code_array), POSE_BATCH):
        last = min(first + POSE_BATCH, len(code_array))
        batch_root = stage_root / f"batch_{first:04d}_{last:04d}"
        result_path = batch_root / "RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            vectors_path = Path(result["pose_vectors"]["path"])
            if file_record(vectors_path) != result["pose_vectors"]:
                raise QS1Error(f"resumed pose vectors differ: {vectors_path}")
            all_vectors.append(np.load(vectors_path, allow_pickle=False))
            continue
        batch_codes = code_array[first:last]
        slaves = surface.render(batch_codes, pair)
        masters = np.repeat(master[None], len(batch_codes), axis=0)
        inputs = np.stack((slaves, masters), axis=1)
        codes_record = retain_npy(batch_root / "codes.int32.npy", batch_codes)
        slave_record = retain_npy(batch_root / "slave_camera.uint8.npy", slaves)
        input_record = retain_npy(batch_root / "pose_input.uint8.npy", inputs)
        vectors = pose_vectors(posenet, inputs)
        vector_record = retain_npy(batch_root / "pose_vectors.float32.npy", vectors)
        result = {
            "schema": "ddm_qs1_pose_batch.v1",
            "pair": pair,
            "candidate_first": first,
            "candidate_last_exclusive": last,
            "codes": codes_record,
            "slave_camera": slave_record,
            "pose_input": input_record,
            "pose_vectors": vector_record,
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
        }
        retain_json(result_path, result)
        all_vectors.append(vectors)
    vectors = np.concatenate(all_vectors, axis=0)
    if vectors.shape != (len(code_array), POSE_DIMENSIONS):
        raise QS1Error("retained PoseNet output census differs")
    retain_npy(stage_root / "ALL_CODES.int32.npy", code_array)
    retain_npy(stage_root / "ALL_POSE_VECTORS.float32.npy", vectors)
    return vectors


def cancellation_metrics(leak: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    leak = np.asarray(leak, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if leak.shape != (POSE_DIMENSIONS,) or residual.shape != leak.shape:
        raise QS1Error("cancellation vectors must be six-dimensional")
    leak_energy = float(leak @ leak)
    residual_energy = float(residual @ residual)
    if leak_energy == 0.0:
        energy_fraction = 1.0 if residual_energy == 0.0 else -math.inf
        norm_fraction = 1.0 if residual_energy == 0.0 else -math.inf
    else:
        energy_fraction = 1.0 - residual_energy / leak_energy
        norm_fraction = 1.0 - math.sqrt(residual_energy / leak_energy)
    return {
        "leak_l2": math.sqrt(leak_energy),
        "residual_l2": math.sqrt(residual_energy),
        "cancellation_energy_fraction": energy_fraction,
        "cancellation_norm_fraction": norm_fraction,
    }


def conservative_dpose_increase_bound(
    base_error: np.ndarray, residual: np.ndarray
) -> float:
    """Cauchy bound for one pair's increase in the n600x6 pose MSE."""
    error = np.asarray(base_error, dtype=np.float64)
    delta = np.asarray(residual, dtype=np.float64)
    if error.shape != (POSE_DIMENSIONS,) or delta.shape != error.shape:
        raise QS1Error("pose bound vectors must be six-dimensional")
    numerator = 2.0 * float(np.linalg.norm(error)) * float(np.linalg.norm(delta))
    numerator += float(delta @ delta)
    return numerator / (PAIR_COUNT * POSE_DIMENSIONS)


def admission_screen(
    *, seg_value_s: float, residual_pose_bound_s: float, delta_bytes: int = 0
) -> dict[str, Any]:
    if seg_value_s < 0.0 or residual_pose_bound_s < 0.0 or delta_bytes < 0:
        raise QS1Error("admission screen inputs must be nonnegative")
    rate_s = delta_bytes * RATE_S_PER_BYTE
    margin = seg_value_s - residual_pose_bound_s - rate_s
    return {
        "optimistic_seg_value_s": seg_value_s,
        "conservative_residual_pose_bound_s": residual_pose_bound_s,
        "rate_delta_bytes": delta_bytes,
        "rate_delta_s": rate_s,
        "screen_margin_s": margin,
        "screened_net_delta_s": -margin,
        "admitted": margin > 0.0,
    }


def strict_descent(
    current: np.ndarray,
    objective: float,
    evaluate: Callable[[tuple[np.ndarray, ...], int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Exact integer coordinate descent; stop after one full non-improving pass."""
    codes = np.asarray(current, dtype=np.int32).copy()
    value = float(objective)
    pass_index = 0
    best_vector = np.empty(POSE_DIMENSIONS, dtype=np.float32)
    while True:
        candidates: list[np.ndarray] = [codes.copy()]
        for dimension in range(DIMENSIONS):
            for delta in (-1, 1):
                candidate = codes.copy()
                candidate[dimension] += delta
                if -2048 <= candidate[dimension] <= 2047:
                    candidates.append(candidate)
        vectors, objectives = evaluate(tuple(candidates), pass_index)
        best_index = min(range(len(candidates)), key=lambda index: (float(objectives[index]), index))
        best_value = float(objectives[best_index])
        best_vector = np.asarray(vectors[best_index], dtype=np.float32)
        if not best_value < value:
            return codes, value, pass_index + 1, best_vector
        codes = candidates[best_index]
        value = best_value
        pass_index += 1


def diagnostic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for screen in sorted((JS6B_STORE / "retained/screens").glob("*/SCREEN_ROW.json")):
        value = json.loads(screen.read_text())
        if value.get("q3_null_delta") is None:
            continue
        proposal_id = str(value["proposal_id"])
        proposal = JS6_BANK / "proposals" / proposal_id / "proposal.json"
        row = json.loads(proposal.read_text())
        row["js6b_screen"] = value
        row["proposal_json_record"] = file_record(proposal)
        rows.append(row)
    if len(rows) != 14 or len({str(row["proposal_id"]) for row in rows}) != 14:
        raise QS1Error(f"expected fourteen unique retained Q3 diagnostics, found {len(rows)}")
    return rows


def source_preflight(output: Path) -> tuple[dict[str, Any], CP135Surface]:
    checkpoint = output / "checkpoints/stage_00_source_preflight.json"
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text())
        sources = prior.get("sources", {})
        for name, record in sources.items():
            path = Path(str(record.get("path", "")))
            if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
                raise QS1Error(f"resumed source size differs for {name}: {path}")
        # The complete SHA census was paid once at the sealed preflight stage.
        # Resume validates every path and byte count, while CP135Surface.load
        # independently re-parses the charged archive and lattice each time.
        surface, carrier = CP135Surface.load()
        if carrier != prior.get("cp135_carrier"):
            raise QS1Error("resumed CP135 carrier parse differs from sealed preflight")
        return prior, surface
    sources = {
        "cp135_archive": require_file(
            CP135_ARCHIVE, expected_bytes=186_252, expected_sha256=CP135_ARCHIVE_SHA256
        ),
        "cp135_raw": require_file(
            CP135_RAW, expected_bytes=RAW_BYTES, expected_sha256=CP135_RAW_SHA256
        ),
        "cp135_basis": require_file(CP135_BASIS),
        "cp135_coefficients": require_file(CP135_COEFFICIENTS),
        "cp135_base_pose_vectors": require_file(CP135_BASE_POSE),
        "gt_pose_vectors": require_file(GT_POSE),
        "joint_integer_solver": require_file(JOINT_SOLVER_SOURCE),
        "carrier_repack_producer": require_file(BOOK_REPACK_SOURCE),
        "upstream_modules": require_file(UPSTREAM / "modules.py"),
        "upstream_pose_weights": require_file(UPSTREAM / "models/posenet.safetensors"),
        "upstream_frame_utils": require_file(UPSTREAM / "frame_utils.py"),
        "upstream_evaluate": require_file(UPSTREAM / "evaluate.py"),
        "js6_index": require_file(JS6_BANK / "proposal_index.jsonl"),
        "js6b_final_result": require_file(JS6B_STORE / "FINAL_RESULT.json"),
    }
    surface, carrier = CP135Surface.load()
    result = {
        "schema": "ddm_qs1_source_preflight.v1",
        "run_id": RUN_ID,
        "sources": sources,
        "cp135_carrier": carrier,
        "solver_constants": {
            "damping": GN_DAMPING,
            "max_code_step": MAX_CODE_STEP,
            "neighbour_dimensions": NEIGHBOUR_DIMS,
            "neighbour_radius": NEIGHBOUR_RADIUS,
            "producer": sources["joint_integer_solver"],
        },
        "axis": AXIS,
        "seed": 135,
        "deterministic_algorithms": True,
        "resume_from": str(output.resolve()),
        "passed": True,
    }
    retain_json(checkpoint, result)
    return result, surface


def solve_one(
    *,
    row: dict[str, Any],
    surface: CP135Surface,
    posenet: Any,
    raw: np.memmap,
    base_pose_all: np.ndarray,
    gt_pose_all: np.ndarray,
    solver: ModuleType,
    output: Path,
) -> dict[str, Any]:
    proposal_id = str(row["proposal_id"])
    pair = int(row["pair"])
    root = output / "retained/proposals" / proposal_id
    final_path = root / "RESULT.json"
    if final_path.is_file():
        return json.loads(final_path.read_text())
    proposal_root = JS6_BANK / "proposals" / proposal_id
    base_camera = np.load(proposal_root / "base_camera.uint8.npy", allow_pickle=False)
    candidate_camera = np.load(
        proposal_root / "candidate_camera.uint8.npy", allow_pickle=False
    )
    if base_camera.shape != (CAMERA_H, CAMERA_W, 3) or candidate_camera.shape != base_camera.shape:
        raise QS1Error(f"JS6 camera geometry differs: {proposal_id}")
    master_base = np.asarray(raw[2 * pair + 1])
    slave_base_raw = np.asarray(raw[2 * pair])
    if not np.array_equal(base_camera, master_base):
        raise QS1Error(f"JS6 base camera is not CP135 raw master: {proposal_id}")
    rendered_base = surface.render(surface.codes[pair : pair + 1], pair)[0]
    receiver_mismatch = int(np.count_nonzero(rendered_base != slave_base_raw))
    if receiver_mismatch:
        raise QS1Error(
            f"CP135 rendered frame-0 is not exact retained receiver raw: {proposal_id}, "
            f"mismatched_values={receiver_mismatch}"
        )
    base_codes = surface.codes[pair].copy()
    baseline_vectors = evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=(base_codes,),
        master=master_base,
        pair=pair,
        stage_root=root / "stage_10_baseline",
    )
    event_vectors = evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=(base_codes,),
        master=candidate_camera,
        pair=pair,
        stage_root=root / "stage_20_event_leak",
    )
    base_vector = baseline_vectors[0]
    event_vector = event_vectors[0]
    retained_base_drift = float(np.max(np.abs(base_vector - base_pose_all[pair])))
    leak = event_vector.astype(np.float64) - base_vector.astype(np.float64)
    jacobian_codes: list[np.ndarray] = [base_codes.copy()]
    for dimension in range(DIMENSIONS):
        for delta in (-1, 1):
            candidate = base_codes.copy()
            candidate[dimension] += delta
            if not -2048 <= candidate[dimension] <= 2047:
                raise QS1Error("CP135 diagnostic coefficient is at an int12 endpoint")
            jacobian_codes.append(candidate)
    jacobian_vectors = evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=tuple(jacobian_codes),
        master=candidate_camera,
        pair=pair,
        stage_root=root / "stage_30_jacobian",
    )
    jacobian = np.empty((POSE_DIMENSIONS, DIMENSIONS), dtype=np.float64)
    for dimension in range(DIMENSIONS):
        minus = jacobian_vectors[1 + 2 * dimension]
        plus = jacobian_vectors[2 + 2 * dimension]
        jacobian[:, dimension] = (plus.astype(np.float64) - minus.astype(np.float64)) / 2.0
    retain_npy(root / "stage_30_jacobian/J_POSE0.float64.npy", jacobian)
    solve = solver.solve_damped_least_squares(
        jacobian, -leak, damping=GN_DAMPING, max_code_step=MAX_CODE_STEP
    )
    centre = solver.quantize_int12_update(base_codes, solve.update)
    active = solver.rank_neighbour_dimensions(jacobian, solve.update, NEIGHBOUR_DIMS)
    neighbourhood = solver.nearby_int12_candidates(
        base_codes,
        centre,
        active_dimensions=active,
        radius=NEIGHBOUR_RADIUS,
    )
    neighbourhood_vectors = evaluate_codes(
        surface=surface,
        posenet=posenet,
        codes=neighbourhood,
        master=candidate_camera,
        pair=pair,
        stage_root=root / "stage_40_integer_cube",
    )
    objectives = np.mean(
        np.square(neighbourhood_vectors.astype(np.float64) - base_vector[None]), axis=1
    )
    best_index = min(range(len(neighbourhood)), key=lambda index: (float(objectives[index]), index))
    current_codes = np.asarray(neighbourhood[best_index], dtype=np.int32)
    current_objective = float(objectives[best_index])

    def evaluate_descent(
        candidates: tuple[np.ndarray, ...], pass_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        vectors = evaluate_codes(
            surface=surface,
            posenet=posenet,
            codes=candidates,
            master=candidate_camera,
            pair=pair,
            stage_root=root / f"stage_50_descent/pass_{pass_index:04d}",
        )
        values = np.mean(
            np.square(vectors.astype(np.float64) - base_vector[None]), axis=1
        )
        retain_npy(
            root / f"stage_50_descent/pass_{pass_index:04d}/OBJECTIVES.float64.npy",
            values,
        )
        return vectors, values

    final_codes, final_objective, passes, final_vector = strict_descent(
        current_codes, current_objective, evaluate_descent
    )
    # If the integer cube itself remained the zero-edit point, strict_descent's
    # first current vector is still the realized event vector.
    if not np.all(np.isfinite(final_vector)):
        raise QS1Error("coordinate descent did not return a finite final vector")
    residual = final_vector.astype(np.float64) - base_vector.astype(np.float64)
    metrics = cancellation_metrics(leak, residual)
    base_error = base_vector.astype(np.float64) - gt_pose_all[pair].astype(np.float64)
    residual_dpose_bound = conservative_dpose_increase_bound(base_error, residual)
    residual_pose_bound_s = residual_dpose_bound * POSE_MARGINAL_S_PER_DPOSE
    exact_local_delta_dpose = (
        float(np.sum(np.square(final_vector.astype(np.float64) - gt_pose_all[pair])))
        - float(np.sum(np.square(base_vector.astype(np.float64) - gt_pose_all[pair])))
    ) / (PAIR_COUNT * POSE_DIMENSIONS)
    screen = admission_screen(
        seg_value_s=float(row["js6b_screen"]["screen"]["optimistic_seg_value_s"]),
        residual_pose_bound_s=residual_pose_bound_s,
        delta_bytes=0,
    )
    result = {
        "schema": "ddm_qs1_schur_pair_result.v1",
        "proposal_id": proposal_id,
        "pair": pair,
        "directed_edge": row["directed_edge"],
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "receiver_realization": {
            "base_frame0_mismatched_values": receiver_mismatch,
            "base_frame1_matches_cp135_raw": True,
            "carrier_coordinate_count": DIMENSIONS,
            "signed_int12": True,
            "retained_base_pose_vector_max_abs_drift": retained_base_drift,
        },
        "solve": {
            "jacobian_rank": solve.rank,
            "jacobian_condition": solve.condition,
            "ridge_lambda": solve.ridge_lambda,
            "float_update": solve.update.tolist(),
            "quantized_centre": centre.tolist(),
            "active_dimensions": list(active),
            "integer_cube_candidates": len(neighbourhood),
            "coordinate_descent_full_passes": passes,
            "derived_stop": "one complete signed-int12 singleton pass accepted zero strict improvements",
            "base_codes": base_codes.tolist(),
            "final_codes": final_codes.tolist(),
            "final_code_delta": (final_codes - base_codes).tolist(),
            "final_objective_mse_to_base_pose_vector": final_objective,
        },
        "pose": {
            **metrics,
            "leak_vector": leak.tolist(),
            "residual_vector": residual.tolist(),
            "exact_local_delta_dpose_one_pair_over_n600": exact_local_delta_dpose,
            "conservative_delta_dpose_increase_bound": residual_dpose_bound,
            "conservative_residual_pose_bound_s_at_603": residual_pose_bound_s,
        },
        "screen": screen,
        "disposition": "SCREEN_SURVIVOR" if screen["admitted"] else "HELD",
        "verdict_scope": (
            "INSTANCE: exact CP135 frame-0 int12 actuator x this retained JS6 frame-1 proposal, "
            "local frozen CPU PoseNet advisory; Seg value remains optimistic and rate is zero"
        ),
        "proposal_source": row["proposal_json_record"],
    }
    retain_npy(root / "FINAL_CODES.int32.npy", final_codes)
    retain_npy(root / "FINAL_POSE_VECTOR.float32.npy", final_vector)
    retain_json(final_path, result)
    return result


def _selected_independent_survivors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the strongest screened survivor per independent temporal pair."""
    by_pair: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not row["screen"]["admitted"]:
            continue
        pair = int(row["pair"])
        incumbent = by_pair.get(pair)
        if incumbent is None or (
            float(row["screen"]["screen_margin_s"]), str(row["proposal_id"])
        ) > (
            float(incumbent["screen"]["screen_margin_s"]),
            str(incumbent["proposal_id"]),
        ):
            by_pair[pair] = row
    return [by_pair[pair] for pair in sorted(by_pair)]


def _cp135_group_positions() -> list[np.ndarray]:
    """Reproduce the pinned CP135 receiver's HPAC token order exactly."""
    patch = 64
    delta = 2
    rows = np.arange(patch).reshape(patch, 1)
    columns = np.arange(patch).reshape(1, patch)
    grid = columns + delta * rows
    patch_rows, patch_columns = 384 // patch, 512 // patch
    positions: list[np.ndarray] = []
    for group in range((1 + delta) * patch - delta):
        local = grid == group
        full = np.broadcast_to(local, (patch_rows, patch_columns, patch, patch))
        mask = full.transpose(0, 2, 1, 3).reshape(384, 512)
        positions.append(np.flatnonzero(mask.reshape(-1)))
    combined = np.concatenate(positions)
    if combined.size != 384 * 512 or np.unique(combined).size != combined.size:
        raise QS1Error("CP135 token-order reproduction is not a pixel partition")
    return positions


def _materialize_js6_tokens(
    *, output: Path, name: str, selected: list[dict[str, Any]], repeat: bool
) -> tuple[ModuleType, Path, dict[str, Any]]:
    """Adapt JO1's HP3/RC64 materializer to exact retained JS6 token frames."""
    from experiments import ddm_jo1_joint_probability_object as jo1

    workspace = output / "compile_workspace"
    root = jo1.candidate_root(workspace, name, repeat=repeat)
    result_path = root / "20_MATERIALIZE_RESULT.json"
    if result_path.is_file():
        return jo1, root, json.loads(result_path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    spatial_path = root / "spatial_tokens.u8"
    temporary = spatial_path.with_name(f".{spatial_path.name}.{os.getpid()}.partial")
    shutil.copyfile(jo1.BASE_SPATIAL, temporary)
    spatial = np.memmap(
        temporary,
        mode="r+",
        dtype=np.uint8,
        shape=(jo1.FRAMES, jo1.HEIGHT, jo1.WIDTH),
    )
    applications: list[dict[str, Any]] = []
    touched: set[tuple[int, int]] = set()
    for row in selected:
        proposal_id = str(row["proposal_id"])
        pair = int(row["pair"])
        candidate_path = Path(
            row.get(
                "candidate_tokens_path",
                JS6_BANK / "proposals" / proposal_id / "candidate_tokens.uint8.npy",
            )
        )
        candidate = np.load(candidate_path, allow_pickle=False)
        before = np.asarray(spatial[pair])
        indices = np.flatnonzero(candidate.reshape(-1) != before.reshape(-1))
        expected_sites = row.get("token_site_count")
        if expected_sites is None:
            expected_sites = json.loads(
                (JS6_BANK / "proposals" / proposal_id / "proposal.json").read_text()
            )["token_site_count"]
        if indices.size != int(expected_sites):
            raise QS1Error(f"JS6 token diff count differs: {proposal_id}")
        overlap = {(pair, int(index)) for index in indices.tolist()} & touched
        if overlap:
            raise QS1Error(f"independent survivor selection contains overlapping token sites: {overlap}")
        touched.update((pair, int(index)) for index in indices.tolist())
        flat = spatial[pair].reshape(-1)
        flat[indices] = candidate.reshape(-1)[indices]
        applications.append(
            {
                "proposal_id": proposal_id,
                "frame": pair,
                "indices": indices.astype(int).tolist(),
                "candidate_tokens": file_record(candidate_path),
            }
        )
    spatial.flush()
    os.replace(temporary, spatial_path)
    applications_record = retain_json(
        root / "EVENT_APPLICATIONS.json",
        {"schema": "ddm_qs1_event_applications.v1", "rows": applications},
    )
    spatial = np.memmap(
        spatial_path,
        mode="r",
        dtype=np.uint8,
        shape=(jo1.FRAMES, jo1.HEIGHT, jo1.WIDTH),
    )
    spatial_sha = jo1.raw_array_sha256(spatial)
    group_positions = _cp135_group_positions()
    positions = np.concatenate(group_positions)
    event_path = root / "event_order.npy"
    event_order = np.lib.format.open_memmap(
        event_path, mode="w+", dtype=np.uint8, shape=(jo1.TOTAL_EVENTS,)
    )
    for frame in range(jo1.FRAMES):
        first = frame * jo1.EVENTS_PER_FRAME
        event_order[first : first + jo1.EVENTS_PER_FRAME] = spatial[frame].reshape(-1)[positions]
    event_order.flush()
    event_raw_sha = jo1.raw_array_sha256(event_order)
    manifest = {
        "schema": "ddm_jo1_event_order_manifest.v1",
        "complete": True,
        "chunks": [
            {
                "start_frame": 0,
                "end_frame": jo1.FRAMES,
                "symbols_path": str(event_path.resolve()),
                "symbols_sha256": file_record(event_path)["sha256"],
                "symbols_bytes": event_path.stat().st_size,
                "tokens": jo1.TOTAL_EVENTS,
            }
        ],
    }
    manifest_record = retain_json(root / "chunk_manifest.json", manifest)
    changed_frames = sorted({int(row["frame"]) for row in applications})
    affected_frames = sorted(
        set(changed_frames)
        | {frame + 1 for frame in changed_frames if frame + 1 < jo1.FRAMES}
    )
    result = {
        "schema": "ddm_jo1_materialized_candidate.v1",
        "name": name,
        "repeat": repeat,
        "proposal_count": len(selected),
        "proposal_ids": [str(row["proposal_id"]) for row in selected],
        "changed_sites": len(touched),
        "changed_frames": changed_frames,
        "probability_affected_frames": affected_frames,
        "spatial_tokens": file_record(spatial_path),
        "spatial_raw_sha256": spatial_sha,
        "event_order": file_record(event_path),
        "event_order_raw_sha256": event_raw_sha,
        "source_manifest": manifest_record,
        "event_applications": applications_record,
        "resumable": True,
    }
    retain_json(result_path, result)
    return jo1, root, result


def _load_cp135_carrier_codes() -> np.ndarray:
    sys.path.insert(0, str(CP135_RUNTIME))
    try:
        from runtime.carrier_repack import materialize_cpr1
        from runtime.residual_archive import read_residual_archive
    finally:
        sys.path.pop(0)
    parts = read_residual_archive(CP135_ARCHIVE)
    canonical = materialize_cpr1(
        parts.carrier_blob,
        SimpleNamespace(N=PAIR_COUNT, CARRIER_DIM=DIMENSIONS),
    )
    sys.path.insert(0, str(EXPERIMENT_BOOK / "src"))
    try:
        from cpr1_sub4.carrier_repack import decode_cpr1_coefficients
    finally:
        sys.path.pop(0)
    return decode_cpr1_coefficients(
        canonical, frames=PAIR_COUNT, dimensions=DIMENSIONS
    ).astype(np.int32, copy=False)


def compensation_object_fingerprint(
    *,
    pair: int,
    semantic_tokens: dict[str, Any],
    master_camera: dict[str, Any],
) -> str:
    """Bind a frame-0 compensation solve to one exact frame-1 object.

    Paths are intentionally excluded from the digest.  The binding follows the
    bytes across retained-store copies, while the caller separately verifies
    that the recorded path still contains those bytes.
    """

    if not 0 <= int(pair) < PAIR_COUNT:
        raise QS1Error("compensation object pair exceeds the n600 domain")
    payload = {
        "schema": "ddm_qs1_compensation_object_fingerprint.v1",
        "pair": int(pair),
        "semantic_tokens": {
            "bytes": int(semantic_tokens["bytes"]),
            "sha256": str(semantic_tokens["sha256"]),
        },
        "master_camera": {
            "bytes": int(master_camera["bytes"]),
            "sha256": str(master_camera["sha256"]),
        },
        "cp135_archive_sha256": CP135_ARCHIVE_SHA256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_compensation_matches_compile_object(row: dict[str, Any]) -> dict[str, Any]:
    """Refuse stale frame-0 compensation after a frame-1 object change.

    Original QS1 rows predate explicit object fingerprints, but their solver
    consumed the proposal's canonical ``candidate_tokens.uint8.npy`` directly.
    That one unchanged legacy object remains admissible.  Any content change
    requires an explicit binding emitted by the fresh solve; carrying the old
    ``final_codes`` across the change is a compile error.
    """

    proposal_id = str(row["proposal_id"])
    pair = int(row["pair"])
    canonical_tokens = (
        JS6_BANK / "proposals" / proposal_id / "candidate_tokens.uint8.npy"
    )
    compile_tokens = Path(row.get("candidate_tokens_path", canonical_tokens))
    canonical_record = file_record(canonical_tokens)
    compile_record = file_record(compile_tokens)
    changed = compile_record["sha256"] != canonical_record["sha256"]
    binding = row.get("compensation_object")
    if binding is None:
        if changed:
            raise QS1Error(
                "changed frame-1 edit object lacks an exact-object compensation solve: "
                f"{proposal_id}"
            )
        return {
            "schema": "ddm_qs1_compile_compensation_binding.v1",
            "proposal_id": proposal_id,
            "pair": pair,
            "mode": "LEGACY_CANONICAL_OBJECT_SOLVED_IN_QS1",
            "semantic_tokens": compile_record,
            "object_changed_from_canonical_proposal": False,
            "passed": True,
        }
    if binding.get("schema") != "ddm_qs1_compensation_object_binding.v1":
        raise QS1Error(f"compensation object binding schema differs: {proposal_id}")
    if int(binding.get("pair", -1)) != pair:
        raise QS1Error(f"compensation object binding pair differs: {proposal_id}")
    if binding.get("semantic_tokens") != compile_record:
        raise QS1Error(f"bound semantic-token object differs at compile: {proposal_id}")
    master_record = binding.get("master_camera")
    if not isinstance(master_record, dict):
        raise QS1Error(f"bound master-camera record is absent: {proposal_id}")
    master_path = Path(str(master_record.get("path", "")))
    if file_record(master_path) != master_record:
        raise QS1Error(f"bound master-camera bytes differ at compile: {proposal_id}")
    expected = compensation_object_fingerprint(
        pair=pair,
        semantic_tokens=compile_record,
        master_camera=master_record,
    )
    if binding.get("fingerprint_sha256") != expected:
        raise QS1Error(f"compensation object fingerprint differs: {proposal_id}")
    if row.get("solve", {}).get("compensation_object_fingerprint_sha256") != expected:
        raise QS1Error(f"frame-0 solve is stale for compile object: {proposal_id}")
    if binding.get("exact_master_rendered_from_semantic_tokens") is not True:
        raise QS1Error(f"compensation master lacks exact render proof: {proposal_id}")
    return {
        "schema": "ddm_qs1_compile_compensation_binding.v1",
        "proposal_id": proposal_id,
        "pair": pair,
        "mode": "EXACT_OBJECT_BOUND_FRESH_SOLVE",
        "semantic_tokens": compile_record,
        "master_camera": master_record,
        "fingerprint_sha256": expected,
        "object_changed_from_canonical_proposal": changed,
        "passed": True,
    }


def _candidate_physical_carrier(
    base_codes: np.ndarray, selected: list[dict[str, Any]]
) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    """Encode selected CP135 int12 changes through the banked CPR1/CAP1 producers."""
    sys.path.insert(0, str(CP135_RUNTIME))
    try:
        from runtime.carrier_repack import (
            materialize_cpr1,
            split_frame0_selector_carrier,
        )
        from runtime.residual_archive import read_residual_archive
    finally:
        sys.path.pop(0)
    parts = read_residual_archive(CP135_ARCHIVE)
    _, selector = split_frame0_selector_carrier(parts.carrier_blob)
    if selector is None or not selector.startswith(b"F0E1\x01"):
        raise QS1Error("CP135 selector is absent or not the proven sparse F0E1 form")
    canonical = materialize_cpr1(
        parts.carrier_blob,
        SimpleNamespace(N=PAIR_COUNT, CARRIER_DIM=DIMENSIONS),
    )
    codes = np.asarray(base_codes, dtype=np.int32).copy()
    compensation_bindings = []
    for row in selected:
        compensation_bindings.append(assert_compensation_matches_compile_object(row))
        final_codes = np.asarray(row["solve"]["final_codes"], dtype=np.int32)
        codes[int(row["pair"])] = final_codes
    sys.path.insert(0, str(EXPERIMENT_BOOK / "src"))
    try:
        from cpr1_sub4.carrier_repack import (
            decode_cps3,
            encode_cps3_coefficients,
        )
        from cpr1_sub4.entropy.coefficient_ar1_codec import encode_cap1
    finally:
        sys.path.pop(0)
    cps3 = encode_cps3_coefficients(
        canonical,
        codes,
        frames=PAIR_COUNT,
        dimensions=DIMENSIONS,
        mode=3,
        threshold=16,
    )
    candidate_cpr1 = decode_cps3(cps3, frames=PAIR_COUNT, dimensions=DIMENSIONS)
    cap1, cap1_report = encode_cap1(
        candidate_cpr1, frames=PAIR_COUNT, dimensions=DIMENSIONS
    )
    # F24S stores the CAP1 body without magic/version, with scales before
    # predictor metadata, and the F0E1 body without its fixed five-byte prefix.
    physical = cap1[8:14] + cap1[50:146] + cap1[14:50] + cap1[146:] + selector[5:]
    report = {
        "schema": "ddm_qs1_candidate_carrier.v1",
        "selected_pairs": [int(row["pair"]) for row in selected],
        "changed_coordinates": int(np.count_nonzero(codes != base_codes)),
        "cpr1_bytes": len(candidate_cpr1),
        "cpr1_sha256": hashlib.sha256(candidate_cpr1).hexdigest(),
        "cap1_bytes": len(cap1),
        "cap1_sha256": hashlib.sha256(cap1).hexdigest(),
        "physical_bytes": len(physical),
        "physical_sha256": hashlib.sha256(physical).hexdigest(),
        "cap1_report": cap1_report,
        "compensation_object_bindings": compensation_bindings,
        "all_compensation_solved_for_exact_compile_objects": True,
    }
    return physical, codes, report


def _patch_variable_carrier_runtime(runtime_root: Path) -> dict[str, Any]:
    path = runtime_root / "runtime/residual_archive.py"
    source = path.read_text()
    old = """    if len(carrier) == PACKED_CAP1_SECTION_BYTES:\n        carrier = _restore_packed_cap1_metadata(carrier)\n    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:\n        return None\n"""
    new = """    if len(carrier) == PACKED_CAP1_SECTION_BYTES:\n        carrier = _restore_packed_cap1_metadata(carrier)\n    elif len(carrier) < 7:\n        return None\n"""
    if source.count(old) != 1:
        raise QS1Error("candidate runtime variable-carrier patch surface differs")
    updated = source.replace(old, new).encode()
    partial = _atomic_path(path)
    try:
        partial.write_bytes(updated)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def _compile_one(
    *, output: Path, selected: list[dict[str, Any]], repeat: bool
) -> dict[str, Any]:
    name = "qs1_combined_unique_pairs"
    from experiments import ddm_jo1_joint_probability_object as jo1

    root = jo1.candidate_root(output / "compile_workspace", name, repeat=repeat)
    result_path = root / "QS1_COMPILED_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        expected = [str(row["proposal_id"]) for row in selected]
        if result.get("schema") != "ddm_qs1_compiled_candidate.v1":
            raise QS1Error(f"compiled resume schema differs: {result_path}")
        if result.get("selected_proposal_ids") != expected:
            raise QS1Error(f"compiled resume survivor set differs: {result_path}")
        if bool(result.get("repeat")) is not repeat:
            raise QS1Error(f"compiled resume repeat identity differs: {result_path}")
        require_file(
            Path(result["archive"]["path"]),
            expected_bytes=int(result["archive"]["bytes"]),
            expected_sha256=str(result["archive"]["sha256"]),
        )
        runtime_root = Path(result["runtime_root"])
        if not runtime_root.is_dir():
            raise QS1Error(f"compiled resume runtime is absent: {result_path}")
        from experiments import ddm_cp135_rate_compose as cp135

        if cp135.tree_record(runtime_root) != result.get("runtime_tree"):
            raise QS1Error(f"compiled resume runtime tree differs: {result_path}")
        return result
    jo1, root, materialized = _materialize_js6_tokens(
        output=output, name=name, selected=selected, repeat=repeat
    )
    entropy = jo1.reclose_candidate(output / "compile_workspace", name, repeat=repeat)
    from experiments import ddm_cp135_rate_compose as cp135

    base_codes = _load_cp135_carrier_codes()
    physical_carrier, candidate_codes, carrier_report = _candidate_physical_carrier(
        base_codes, selected
    )
    brotli_binary = shutil.which("brotli")
    if brotli_binary is None:
        raise QS1Error("the pinned Brotli CLI required by CP135 closure is unavailable")
    base_models = Path(entropy["models"]["path"]).read_bytes()
    hpac, semantic, _ = cp135.unpack_split_models(
        base_models, brotli_binary=brotli_binary
    )
    model_payload, model_report = cp135._optimal_split_models(
        (hpac, semantic, physical_carrier),
        variant="hp3_step2",
        representation="qs1_canonical_cap1_variable",
        output=root,
        brotli_binary=brotli_binary,
    )
    residual = Path(entropy["residual"]["path"]).read_bytes()
    token = Path(entropy["token"]["path"]).read_bytes()
    member = model_payload + residual + token
    archive = jo1.deterministic_zip(member)
    objects = root / "qs1_objects"
    model_record = retain_bytes(objects / "models.bin", model_payload)
    residual_record = retain_bytes(objects / "residual.compact.bin", residual)
    token_record = retain_bytes(objects / "tokens.rc64", token)
    member_record = retain_bytes(objects / "p", member)
    archive_record = retain_bytes(objects / "archive.zip", archive)
    archive_repeat = retain_bytes(objects / "archive.repeat.zip", jo1.deterministic_zip(member))
    runtime_root = root / "adapted_runtime"
    runtime = jo1.copy_runtime(runtime_root, archive)
    runtime_patch = _patch_variable_carrier_runtime(runtime_root)
    for module_name in tuple(sys.modules):
        if module_name == "runtime" or module_name.startswith("runtime."):
            sys.modules.pop(module_name, None)
    runtime_module = cp135.load_runtime(runtime_root)
    parsed = runtime_module.read_residual_archive(objects / "archive.zip")
    canonical = runtime_module.materialize_cpr1(
        parsed.carrier_blob,
        SimpleNamespace(N=PAIR_COUNT, CARRIER_DIM=DIMENSIONS),
    )
    sys.path.insert(0, str(EXPERIMENT_BOOK / "src"))
    try:
        from cpr1_sub4.carrier_repack import decode_cpr1_coefficients
    finally:
        sys.path.pop(0)
    parsed_codes = decode_cpr1_coefficients(
        canonical, frames=PAIR_COUNT, dimensions=DIMENSIONS
    )
    if not np.array_equal(parsed_codes, candidate_codes):
        raise QS1Error("candidate runtime parse-back changed the intended int12 lattice")
    if parsed.token_stream != token or parsed.residual_payload != b"RCF1" + residual:
        raise QS1Error("candidate runtime token/residual parse-back differs")
    tree = cp135.tree_record(runtime_root)
    result = {
        "schema": "ddm_qs1_compiled_candidate.v1",
        "repeat": repeat,
        "selected_proposal_ids": [str(row["proposal_id"]) for row in selected],
        "selected_pairs": [int(row["pair"]) for row in selected],
        "materialized": materialized,
        "entropy_reclose": file_record(root / "50_RECLOSE_RESULT.json"),
        "carrier": carrier_report,
        "model_report": model_report,
        "models": model_record,
        "residual": residual_record,
        "token": token_record,
        "member": member_record,
        "archive": archive_record,
        "archive_repeat": archive_repeat,
        "runtime_root": str(runtime_root.resolve()),
        "runtime_archive": runtime["archive"],
        "runtime_inflate": runtime["inflate"],
        "runtime_variable_carrier_patch": runtime_patch,
        "runtime_tree": tree,
        "receiver_parseback": {
            "int12_lattice_exact": True,
            "token_stream_exact": True,
            "residual_exact": True,
            "archive_member_exact": True,
        },
        "delta_bytes_vs_cp135": archive_record["bytes"] - 186_252,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(root / "QS1_COMPILED_RESULT.json", result)
    return result


def _seal_fire_order(
    *, output: Path, compiled: dict[str, Any], screen: dict[str, Any]
) -> dict[str, Any]:
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b

    run_id = "ddm_qs1_dual_axis_20260813_r2"
    fire_root = output / "fire_order"
    input_root = fire_root / "fire_inputs"
    archive_path = Path(compiled["archive"]["path"])
    runtime_root = Path(compiled["runtime_root"])
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        runtime_root, label="ddm_qs1_combined_unique_pairs"
    )
    pose_screen_payload = (json.dumps(screen, indent=2, sort_keys=True) + "\n").encode()
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "POSE_SCREEN_RESULT.json": pose_screen_payload,
    }
    for name, payload in payloads.items():
        retain_bytes(input_root / name, payload)
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": run_id,
        "resume_from": run_id,
        "lane_id": "ddm_qs1_dual_axis_n600_20260813",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "MAIN",
        "seed": 1234,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": runtime_manifest,
        "inputs": {
            name: js1b.payload_record(payload) for name, payload in payloads.items()
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "dispatcher_source_sha256": sha256_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_record = retain_json(fire_root / "SEALED_REQUEST.json", request)
    dispatch_output = output / "dispatch" / run_id
    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        request_record["path"],
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str(dispatch_output.resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_qs1_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "MAIN confirms no active n600 exact-eval/Modal lane, claims lane "
            "ddm_qs1_dual_axis_n600_20260813, verifies the sealed request and all input SHAs, "
            "then executes exact_command_argv"
        ),
        "fresh_run_id": run_id,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "budget_ledger": "#381",
        "remote_scope": (
            "one exact candidate decode plus n600 frozen T4 SegNet argmax field and official "
            "PoseNet first-six vectors with a repeat"
        ),
        "post_harvest_admission": (
            "recompute complete candidate S locally from returned Seg field, Pose vectors, and "
            "exact archive bytes; accept only net realized delta S < 0"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "SEALED_FIRE_ORDER.json", order)
    return order


def compile_survivors(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _selected_independent_survivors(rows)
    if not selected:
        raise QS1Error("compile_survivors requires at least one screened survivor")
    primary = _compile_one(output=output, selected=selected, repeat=False)
    repeated = _compile_one(output=output, selected=selected, repeat=True)
    if primary["archive"]["sha256"] != repeated["archive"]["sha256"]:
        raise QS1Error("independent coupled archive repeat differs")
    seg_value = sum(float(row["screen"]["optimistic_seg_value_s"]) for row in selected)
    pose_bound = sum(
        float(row["screen"]["conservative_residual_pose_bound_s"]) for row in selected
    )
    delta_bytes = int(primary["delta_bytes_vs_cp135"])
    # A rate saving is credited honestly; admission_screen's nonnegative input
    # rule is for conservative precompile screens, so complete arithmetic is explicit here.
    rate_s = delta_bytes * RATE_S_PER_BYTE
    margin = seg_value - pose_bound - rate_s
    screen = {
        "schema": "ddm_qs1_compiled_candidate_screen.v1",
        "selected_proposal_ids": [str(row["proposal_id"]) for row in selected],
        "selected_pairs": [int(row["pair"]) for row in selected],
        "optimistic_seg_value_s": seg_value,
        "conservative_residual_pose_bound_s": pose_bound,
        "exact_archive_delta_bytes": delta_bytes,
        "exact_rate_delta_s": rate_s,
        "screen_margin_s": margin,
        "screened_net_delta_s": -margin,
        "admitted": margin > 0.0,
        "honesty_boundary": (
            "frame-0 pose and rate are receiver-realized; Seg value is still the JS6 optimistic "
            "target-support bound, so T4 dual-axis measurement is the verdict"
        ),
    }
    retain_json(output / "COMPILED_SCREEN.json", screen)
    result = {
        "schema": "ddm_qs1_compile_result.v1",
        "primary": primary,
        "independent_repeat": repeated["archive"],
        "archive_repeat_byte_identical": True,
        "screen": screen,
        "score_claim": False,
        "promotion_eligible": False,
    }
    if screen["admitted"]:
        result["fire_order"] = _seal_fire_order(
            output=output, compiled=primary, screen=screen
        )
        result["disposition"] = "QUEUED-WITH-A-FIRE-ORDER"
    else:
        no_fire = {
            "schema": "ddm_qs1_sealed_no_fire_order.v1",
            "sealed": True,
            "disposition": "FOLDED",
            "owner": "MAIN",
            "consumer_store": str(output.resolve()),
            "reason": "compiled complete candidate failed the actual-rate conservative screen",
            "screen": screen,
            "candidate_archives": [primary["archive"], repeated["archive"]],
            "fire_trigger": (
                "reopen only if receiver-realized pose compensation or candidate-specific Seg "
                "evidence makes this exact compiled object's screen margin positive"
            ),
        }
        result["no_fire_order"] = retain_json(
            output / "SEALED_NO_FIRE_ORDER.json", no_fire
        )
        result["disposition"] = "FOLDED"
    retain_json(output / "COMPILE_RESULT.json", result)
    return result


def finalize(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    survivors = [row for row in rows if row["screen"]["admitted"]]
    best_cancellation = max(
        rows, key=lambda row: row["pose"]["cancellation_energy_fraction"]
    )
    best_screen = max(rows, key=lambda row: row["screen"]["screen_margin_s"])
    result = {
        "schema": "ddm_qs1_final_result.v1",
        "run_id": RUN_ID,
        "axis": AXIS,
        "diagnostic_denominator": {"examined": len(rows), "declared": 14},
        "survivor_count": len(survivors),
        "held_count": len(rows) - len(survivors),
        "max_cancellation_energy_fraction": best_cancellation["pose"][
            "cancellation_energy_fraction"
        ],
        "max_cancellation_proposal_id": best_cancellation["proposal_id"],
        "best_screen_margin_s": best_screen["screen"]["screen_margin_s"],
        "best_screen_proposal_id": best_screen["proposal_id"],
        "proposal_results": [
            {
                "proposal_id": row["proposal_id"],
                "pair": row["pair"],
                "disposition": row["disposition"],
                "cancellation_energy_fraction": row["pose"][
                    "cancellation_energy_fraction"
                ],
                "screen_margin_s": row["screen"]["screen_margin_s"],
                "result": file_record(
                    output / "retained/proposals" / row["proposal_id"] / "RESULT.json"
                ),
            }
            for row in rows
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if survivors:
        compiled = compile_survivors(output, rows)
        result["finalization"] = {
            "disposition": compiled["disposition"],
            "compile_result": file_record(output / "COMPILE_RESULT.json"),
            "fire_order": (
                file_record(output / "SEALED_FIRE_ORDER.json")
                if (output / "SEALED_FIRE_ORDER.json").is_file()
                else None
            ),
        }
    else:
        no_fire = {
            "schema": "ddm_qs1_sealed_no_fire_order.v1",
            "run_id": RUN_ID,
            "sealed": True,
            "disposition": "FOLDED",
            "owner": "MAIN",
            "consumer_store": str(output.resolve()),
            "reason": "0/14 exact integer Schur diagnostics passed the conservative zero-rate screen",
            "max_cancellation_energy_fraction": result[
                "max_cancellation_energy_fraction"
            ],
            "best_screen_margin_s": result["best_screen_margin_s"],
            "candidate_archives": [],
            "fire_command": None,
            "fire_trigger": (
                "reopen only with a different receiver-realizable coupled actuator or a T4-calibrated "
                "residual bound that makes a retained proposal's same-object screen margin positive"
            ),
        }
        result["finalization"] = {
            "disposition": "FOLDED",
            "no_fire_order": retain_json(output / "SEALED_NO_FIRE_ORDER.json", no_fire),
        }
    atomic_json(output / "FINAL_RESULT.json", result)
    atomic_json(output / "checkpoints/stage_90_final.json", result)
    return result


def run(
    output: Path = OUTPUT, *, proposal_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    import torch

    if output.resolve() != OUTPUT.resolve():
        raise QS1Error(f"output must be the governed SSD store: {OUTPUT}")
    torch.manual_seed(135)
    np.random.seed(135)
    torch.use_deterministic_algorithms(True)
    storage_preflight(output)
    _, surface = source_preflight(output)
    solver = _load_module("_ddm_qs1_joint_pose_solve", JOINT_SOLVER_SOURCE)
    rows = diagnostic_rows()
    retain_json(
        output / "checkpoints/stage_05_diagnostic_census.json",
        {
            "schema": "ddm_qs1_diagnostic_census.v1",
            "count": len(rows),
            "proposal_ids": [row["proposal_id"] for row in rows],
            "pairs": [row["pair"] for row in rows],
            "bank_exhaustion_is_derived_stop": True,
        },
    )
    raw = np.memmap(
        CP135_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT * 2, CAMERA_H, CAMERA_W, 3),
    )
    base_pose = np.load(CP135_BASE_POSE, allow_pickle=False)
    gt_pose = np.load(GT_POSE, allow_pickle=False)
    if base_pose.shape != (PAIR_COUNT, POSE_DIMENSIONS) or gt_pose.shape != base_pose.shape:
        raise QS1Error("retained pose-vector geometry differs")
    posenet = load_posenet()
    selected = set(proposal_ids or ())
    if selected:
        known = {str(row["proposal_id"]) for row in rows}
        if not selected <= known:
            raise QS1Error(f"unknown diagnostic proposal ids: {sorted(selected - known)}")
        rows = [row for row in rows if str(row["proposal_id"]) in selected]
    results: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        result = solve_one(
            row=row,
            surface=surface,
            posenet=posenet,
            raw=raw,
            base_pose_all=base_pose,
            gt_pose_all=gt_pose,
            solver=solver,
            output=output,
        )
        results.append(result)
        atomic_json(
            output / "STATE.json",
            {
                "schema": "ddm_qs1_resume_state.v1",
                "completed": ordinal + 1,
                "declared": len(rows),
                "last_proposal_id": row["proposal_id"],
                "resume_from": str(output.resolve()),
            },
        )
    if selected:
        return {
            "schema": "ddm_qs1_partial_resume_result.v1",
            "proposal_results": results,
            "finalized": False,
        }
    return finalize(output, results)


def compile_only(output: Path = OUTPUT) -> dict[str, Any]:
    """Resume after the complete local census without importing either scorer."""
    if output.resolve() != OUTPUT.resolve():
        raise QS1Error(f"output must be the governed SSD store: {OUTPUT}")
    census_path = output / "checkpoints/stage_05_diagnostic_census.json"
    census = json.loads(census_path.read_text())
    proposal_ids = [str(value) for value in census.get("proposal_ids", [])]
    if len(proposal_ids) != 14 or len(set(proposal_ids)) != 14:
        raise QS1Error("compile-only sealed diagnostic census differs")
    rows: list[dict[str, Any]] = []
    for proposal_id in proposal_ids:
        path = output / "retained/proposals" / proposal_id / "RESULT.json"
        if not path.is_file():
            raise QS1Error(f"compile-only requires complete diagnostic result: {path}")
        result = json.loads(path.read_text())
        if result.get("schema") != "ddm_qs1_schur_pair_result.v1":
            raise QS1Error(f"diagnostic result schema differs: {path}")
        rows.append(result)
    if len(rows) != 14:
        raise QS1Error("compile-only diagnostic denominator differs")
    return finalize(output, rows)


def reseal_fire_order(output: Path = OUTPUT) -> dict[str, Any]:
    """Preserve the import-broken r1 receipt and atomically install sealed r2."""
    if output.resolve() != OUTPUT.resolve():
        raise QS1Error(f"output must be the governed SSD store: {OUTPUT}")
    compile_path = output / "COMPILE_RESULT.json"
    final_path = output / "FINAL_RESULT.json"
    order_path = output / "SEALED_FIRE_ORDER.json"
    request_path = output / "fire_order/SEALED_REQUEST.json"
    checkpoint_path = output / "checkpoints/stage_90_final.json"
    required = (compile_path, final_path, order_path, request_path, checkpoint_path)
    superseded_root = output / "superseded_fire_order_r1"
    superseded: dict[str, dict[str, Any]] = {}
    for source in required:
        target = superseded_root / source.relative_to(output)
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if file_record(target)["sha256"] != file_record(source)["sha256"]:
                    raise QS1Error(f"superseded receipt differs: {target}")
            else:
                os.replace(source, target)
        if not target.is_file():
            raise QS1Error(f"required r1 receipt is absent from both locations: {source}")
        superseded[str(source.relative_to(output))] = file_record(target)

    compile_result = json.loads(
        (superseded_root / compile_path.relative_to(output)).read_text()
    )
    final_result = json.loads((superseded_root / final_path.relative_to(output)).read_text())
    old_request = json.loads(
        (superseded_root / request_path.relative_to(output)).read_text()
    )
    dispatcher_sha = sha256_file(REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py")
    if old_request.get("dispatcher_source_sha256") == dispatcher_sha:
        raise QS1Error("r1 dispatcher already matches; supersession premise is false")
    primary = compile_result.get("primary")
    screen = compile_result.get("screen")
    if not isinstance(primary, dict) or not isinstance(screen, dict):
        raise QS1Error("retained compile result lacks primary candidate or screen")
    order = _seal_fire_order(output=output, compiled=primary, screen=screen)
    compile_result["fire_order"] = order
    compile_record = retain_json(compile_path, compile_result)
    final_result["finalization"] = {
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "compile_result": compile_record,
        "fire_order": file_record(order_path),
    }
    final_record = retain_json(final_path, final_result)
    retain_json(checkpoint_path, final_result)
    receipt = {
        "schema": "ddm_qs1_fire_order_supersession.v1",
        "reason": (
            "r1 dispatcher imported function_call_id from call_id_ledger instead of "
            "auth_eval and failed a local sealed-input import smoke"
        ),
        "superseded_disposition": "FOLDED_INVALID_TRANSPORT",
        "superseded": superseded,
        "current_fire_order": file_record(order_path),
        "current_request": file_record(request_path),
        "current_compile_result": compile_record,
        "current_final_result": final_record,
        "payloads_deleted": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(output / "FIRE_ORDER_SUPERSESSION.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    parser.add_argument("--proposal-id", action="append", default=[])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--reseal-fire-order", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_from.resolve() != args.output.resolve():
        raise QS1Error("--resume-from must equal --output for byte-faithful resume")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "RUN.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QS1Error(f"another QS1 process holds {lock_path}") from exc
        if args.compile_only and args.reseal_fire_order:
            raise QS1Error("--compile-only and --reseal-fire-order are mutually exclusive")
        if args.reseal_fire_order:
            if args.proposal_id:
                raise QS1Error("--reseal-fire-order cannot be combined with --proposal-id")
            result = reseal_fire_order(args.output)
        elif args.compile_only:
            if args.proposal_id:
                raise QS1Error("--compile-only cannot be combined with --proposal-id")
            result = compile_only(args.output)
        else:
            result = run(args.output, proposal_ids=args.proposal_id)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
