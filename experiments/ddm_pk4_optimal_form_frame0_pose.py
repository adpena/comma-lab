#!/usr/bin/env python3
"""Optimal-form PK4 frame-0 pose control on the exact CP135 object.

The permitted ``prepare`` phase is scorer-free.  It pins the real CP135
receiver/object, writes a seeded stratified-random sample plan, probes Metal,
and emits resumable local and per-rung orders.  ``measure`` is fail-closed on
both Metal access and an explicit single-flight scorer-ownership receipt.

The measurement phase builds fresh exact-object finite-difference Jacobians,
per-pair Gauss-Newton/int12 solutions, and three temporal representations.  A
rung is compiled only after an exact held-out repeat gate is positive by at
least twice its measured pair-noise RMS.  Every materialized scorer input,
preprocessed tensor, pose vector, code lattice, and archive byte string is
retained under the SSD output root before it is summarized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jo1_joint_probability_object as jo1
from experiments import ddm_pk3_frame0_pose_representation as pk3
from experiments import ddm_pk4_frame0_pose_overlay_runtime as overlay
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_rate_rung as qs2

RUN_ID: Final = "ddm_pk4_optimal_form_frame0_pose_20260813"
OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pk4_20260813")
OWNERSHIP_RECEIPT: Final = OUTPUT / "SCORER_OWNERSHIP_RECEIPT.json"
PAIR_COUNT: Final = 600
DIMENSIONS: Final = 12
POSE_DIMENSIONS: Final = 6
SEED: Final = 20260813
MIN_SAMPLE_COUNT: Final = 64
RUNG_KNOTS: Final = {"rung_42": 6, "rung_250": 40, "rung_1000": 165}
RIDGES: Final = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
GAINS: Final = (0.25, 0.5, 0.75, 1.0)
GN_DAMPING: Final = 1e-3
MAX_GN_STEP: Final = 7.0
POSE_BATCH: Final = 8
EXPECTED_RETAINED_BYTES: Final = 96 * 1024**3
RESERVE_BYTES: Final = 8 * 1024**3
AXIS: Final = "[macOS-MLX research-signal; exact CP135 receiver/R; non-promotable]"


class PK4Error(RuntimeError):
    """A pin, ownership, Metal, retention, generalization, or receiver gate failed."""


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.partial")


def retain_bytes(path: Path, payload: bytes, *, executable: bool = False) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if path.is_file():
        if qs1.file_record(path) != expected:
            raise PK4Error(f"refusing to replace different retained payload: {path}")
        return expected
    partial = _atomic_path(path)
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            partial.chmod(0o755)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return qs1.file_record(path)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return retain_bytes(path, payload)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    """Atomically replace live metadata; retained payload/checkpoint writers stay immutable."""
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
    return qs1.file_record(path)


def retain_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    partial = _atomic_path(path)
    try:
        with partial.open("wb") as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_file():
            existing = qs1.file_record(path)
            candidate = qs1.file_record(partial)
            if existing["bytes"] != candidate["bytes"] or existing["sha256"] != candidate["sha256"]:
                raise PK4Error(f"refusing to replace different retained array: {path}")
            return existing
        os.replace(partial, path)
        return qs1.file_record(path)
    finally:
        partial.unlink(missing_ok=True)


def require_record(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or qs1.file_record(path) != record:
        raise PK4Error(f"retained {label} failed custody: {path}")
    return path


def storage_preflight(output: Path, sample_count: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    expected = EXPECTED_RETAINED_BYTES * int(sample_count) // MIN_SAMPLE_COUNT
    required = max(0, expected - retained) + RESERVE_BYTES
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_pk4_storage_preflight.v1",
        "tier": str(output.resolve()),
        "already_retained_bytes": retained,
        "sample_count": int(sample_count),
        "expected_total_retained_bytes": expected,
        "reserve_bytes": RESERVE_BYTES,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; retain every materialized payload; never delete",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise PK4Error(f"SSD storage preflight failed: free={free}, required={required}")
    return result


def source_preflight(output: Path) -> dict[str, Any]:
    sources = {
        "cp135_archive": qs1.require_file(
            qs1.CP135_ARCHIVE,
            expected_bytes=186_252,
            expected_sha256=qs1.CP135_ARCHIVE_SHA256,
        ),
        "cp135_raw": qs1.require_file(
            qs1.CP135_RAW,
            expected_bytes=qs1.RAW_BYTES,
            expected_sha256=qs1.CP135_RAW_SHA256,
        ),
        "cp135_basis": qs1.require_file(qs1.CP135_BASIS),
        "cp135_coefficients": qs1.require_file(qs1.CP135_COEFFICIENTS),
        "cp135_base_pose": qs1.require_file(qs1.CP135_BASE_POSE),
        "gt_pose": qs1.require_file(qs1.GT_POSE),
        "upstream_modules": qs1.require_file(REPO / "upstream/modules.py"),
        "upstream_pose_weights": qs1.require_file(REPO / "upstream/models/posenet.safetensors"),
        "upstream_frame_utils": qs1.require_file(REPO / "upstream/frame_utils.py"),
        "upstream_evaluate": qs1.require_file(REPO / "upstream/evaluate.py"),
        "runner": qs1.require_file(Path(__file__).resolve()),
        "overlay_runtime": qs1.require_file(
            REPO / "experiments/ddm_pk4_frame0_pose_overlay_runtime.py"
        ),
        "pk3_result": qs1.require_file(
            Path("/Volumes/VertigoDataTier/pact/ddm_pk3_20260813/FINAL_RESULT.json")
        ),
        "dispatcher": qs1.require_file(REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"),
        "worker": qs1.require_file(REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"),
    }
    _, carrier = qs1.CP135Surface.load()
    result = {
        "schema": "ddm_pk4_source_preflight.v1",
        "run_id": RUN_ID,
        "sources": sources,
        "cp135_carrier": carrier,
        "exact_object": {
            "archive_bytes": 186_252,
            "archive_sha256": qs1.CP135_ARCHIVE_SHA256,
            "frame1_partner": "CP135 public-decode raw frame 2*pair+1",
            "frame0_actuator": "CP135 int12 codes through CP135Surface.render uint8+R",
        },
        "seed": SEED,
        "resume_from": str(output.resolve()),
        "deterministic": True,
        "owns_scorer": False,
        "scorer_calls": 0,
        "passed": True,
    }
    runner_sha = str(sources["runner"]["sha256"])
    checkpoint = output / "checkpoints" / f"stage_00_source_preflight_{runner_sha[:16]}.json"
    record = retain_json(checkpoint, result)
    atomic_json(output / "SOURCE_PREFLIGHT_LATEST.json", {
        "schema": "ddm_pk4_source_preflight_pointer.v1",
        "checkpoint": record,
    })
    return {**result, "_checkpoint_record": record}


def sample_plan(sample_count: int = MIN_SAMPLE_COUNT) -> dict[str, Any]:
    """Seeded random/stratified plan; never a contiguous prefix."""
    if sample_count not in (MIN_SAMPLE_COUNT, PAIR_COUNT):
        raise PK4Error("sample count must be the n64 floor or full n600")
    rng = np.random.default_rng(SEED)
    if sample_count == PAIR_COUNT:
        selected = rng.permutation(PAIR_COUNT)
        holdout_count = 120
        strata = 20
    else:
        strata = 16
        selected_parts = []
        for stratum in range(strata):
            lo = stratum * PAIR_COUNT // strata
            hi = (stratum + 1) * PAIR_COUNT // strata
            selected_parts.append(rng.choice(np.arange(lo, hi), size=4, replace=False))
        selected = np.concatenate(selected_parts)
        holdout_count = strata
    holdout = []
    train = []
    if sample_count == MIN_SAMPLE_COUNT:
        for stratum in range(strata):
            block = selected[4 * stratum : 4 * (stratum + 1)].copy()
            rng.shuffle(block)
            holdout.append(int(block[0]))
            train.extend(int(value) for value in block[1:])
    else:
        holdout = [int(value) for value in selected[:holdout_count]]
        train = [int(value) for value in selected[holdout_count:]]
    selected_sorted = sorted(train + holdout)
    if len(selected_sorted) != sample_count or len(set(selected_sorted)) != sample_count:
        raise PK4Error("sample plan census differs")
    if sample_count < PAIR_COUNT and selected_sorted == list(range(sample_count)):
        raise PK4Error("prefix-biased sample is forbidden")
    return {
        "schema": "ddm_pk4_stratified_random_sample_plan.v1",
        "seed": SEED,
        "selection_mode": "seeded temporal-stratified random without replacement",
        "sample_count": sample_count,
        "train_pairs": sorted(train),
        "holdout_pairs": sorted(holdout),
        "train_denominator": len(train),
        "holdout_denominator": len(holdout),
        "strata": strata,
        "prefix_forbidden": True,
    }


def retain_sample_plan(output: Path, sample_count: int) -> dict[str, Any]:
    plan = sample_plan(sample_count)
    retain_npy(
        output / "retained/sample_plan/train_pairs.int16.npy",
        np.asarray(plan["train_pairs"], dtype=np.int16),
    )
    retain_npy(
        output / "retained/sample_plan/holdout_pairs.int16.npy",
        np.asarray(plan["holdout_pairs"], dtype=np.int16),
    )
    retain_json(output / "checkpoints/stage_05_sample_plan.json", plan)
    return plan


def probe_metal() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "ddm_pk4_metal_probe.v1",
        "device_required": "mlx.gpu Metal",
        "passed": False,
    }
    try:
        import mlx.core as mx

        with mx.stream(mx.gpu):
            value = mx.array([1.0, 2.0], dtype=mx.float32)
            reduced = mx.sum(value)
            mx.eval(reduced)
            observed = float(np.asarray(reduced))
        if observed != 3.0:
            raise PK4Error(f"Metal arithmetic differed: {observed}")
        result.update({"passed": True, "observed_sum": observed})
    except Exception as error:  # exact typed blocker is persisted by the caller
        result.update({"error_type": type(error).__name__, "error": str(error)})
    return result


def validate_ownership_receipt(path: Path, output: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PK4Error(f"scorer ownership receipt is absent: {path}")
    value = json.loads(path.read_text())
    expected = {
        "schema": "ddm_pk4_scorer_ownership.v1",
        "owner": "MAIN",
        "active": True,
        "consumer_store": str(output.resolve()),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise PK4Error("scorer ownership receipt does not match the single-flight contract")
    if not str(value.get("lane_id", "")).startswith("ddm_pk4_"):
        raise PK4Error("scorer ownership receipt has no PK4 lane id")
    return value


def temporal_weights(pair: int, knots: int) -> np.ndarray:
    if not 0 <= int(pair) < PAIR_COUNT or not 2 <= int(knots) <= overlay.MAX_KNOTS:
        raise PK4Error("temporal-weight domain differs")
    position = int(pair) * (int(knots) - 1) / (PAIR_COUNT - 1)
    left = min(int(position), int(knots) - 2)
    fraction = position - left
    weights = np.zeros(int(knots), dtype=np.float64)
    weights[left] = 1.0 - fraction
    weights[left + 1] = fraction
    return weights


def fit_temporal_controls(
    pairs: np.ndarray,
    target_deltas: np.ndarray,
    knots: int,
    *,
    ridge: float,
    gain: float,
) -> np.ndarray:
    """Fit only per-pair exact-GN lattice deltas; holdout pairs are never inputs."""
    pair_values = np.asarray(pairs, dtype=np.int64)
    targets = np.asarray(target_deltas, dtype=np.float64)
    if targets.shape != (len(pair_values), DIMENSIONS):
        raise PK4Error("temporal-fit target geometry differs")
    design = np.stack([temporal_weights(int(pair), knots) for pair in pair_values])
    augmented = np.concatenate((design, np.sqrt(ridge) * np.eye(knots)), axis=0)
    controls = np.empty((knots, DIMENSIONS), dtype=np.int32)
    for dimension in range(DIMENSIONS):
        rhs = np.concatenate((targets[:, dimension], np.zeros(knots)))
        solution = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        controls[:, dimension] = np.rint(float(gain) * solution).astype(np.int32)
    return np.clip(controls, overlay.MIN_CONTROL, overlay.MAX_CONTROL)


def generalization_gate(
    base_first: np.ndarray,
    candidate_first: np.ndarray,
    base_repeat: np.ndarray,
    candidate_repeat: np.ndarray,
    gt: np.ndarray,
    *,
    lopo_modeled_pose_mse_reduction: float,
) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (
        base_first, candidate_first, base_repeat, candidate_repeat, gt
    )]
    if not arrays or any(value.ndim != 2 or value.shape[1] != POSE_DIMENSIONS for value in arrays):
        raise PK4Error("generalization-gate vector geometry differs")
    if len({value.shape for value in arrays}) != 1:
        raise PK4Error("generalization-gate vector census differs")
    b1, c1, b2, c2, target = arrays
    improvement_first = np.mean((b1 - target) ** 2, axis=1) - np.mean((c1 - target) ** 2, axis=1)
    improvement_repeat = np.mean((b2 - target) ** 2, axis=1) - np.mean((c2 - target) ** 2, axis=1)
    mean_improvement = float(np.mean(improvement_first))
    pair_noise = improvement_repeat - improvement_first
    pair_noise_rms = float(np.sqrt(np.mean(np.square(pair_noise))))
    threshold = 2.0 * pair_noise_rms
    lopo_positive = float(lopo_modeled_pose_mse_reduction) > 0.0
    passed = mean_improvement > 0.0 and mean_improvement >= threshold and lopo_positive
    return {
        "schema": "ddm_pk4_generalization_gate.v1",
        "holdout_denominator": len(target),
        "selection_mode": "seeded stratified-random heldout pairs; no fit access",
        "heldout_mean_pose_mse_reduction": mean_improvement,
        "pair_noise_rms_from_exact_repeat": pair_noise_rms,
        "two_sigma_threshold": threshold,
        "positive": mean_improvement > 0.0,
        "lopo_modeled_pose_mse_reduction": float(lopo_modeled_pose_mse_reduction),
        "lopo_positive": lopo_positive,
        "passed": passed,
        "disposition": "GATE_PASS" if passed else "GATE_FAIL_NO_COMPILE",
        "score_claim": False,
        "promotion_eligible": False,
    }


def lopo_modeled_reduction(
    output: Path,
    train_pairs: np.ndarray,
    knots: int,
    gt_pose: np.ndarray,
    *,
    ridge: float,
    gain: float,
) -> float:
    """Leave-one-pair-out linear response using only exact-object PK4 bank rows."""
    pairs = np.asarray(train_pairs, dtype=np.int64)
    final_deltas = np.stack([_load_pair_delta(output, int(pair)) for pair in pairs])
    reductions = []
    for heldout in range(len(pairs)):
        keep = np.arange(len(pairs)) != heldout
        controls = fit_temporal_controls(
            pairs[keep], final_deltas[keep], knots, ridge=ridge, gain=gain
        )
        expanded = overlay.expand_pose_controls(controls)
        pair = int(pairs[heldout])
        root = output / "retained/jacobian_bank" / f"pair_{pair:03d}" / "stage_10_jacobian"
        jacobian = np.load(root / "J_POSE0.float64.npy", allow_pickle=False)
        vectors = np.load(root / "ALL_POSE_VECTORS.float32.npy", allow_pickle=False)
        base_error = vectors[0].astype(np.float64) - gt_pose[pair].astype(np.float64)
        predicted = base_error + jacobian @ expanded[pair]
        reductions.append(float(np.mean(base_error**2) - np.mean(predicted**2)))
    return float(np.mean(reductions))


def select_rung_controls(
    output: Path,
    train_pairs: np.ndarray,
    train_deltas: np.ndarray,
    knots: int,
    gt_pose: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune each rung on train-only LOPO; exact heldout vectors stay untouched."""
    rows = []
    for ridge in RIDGES:
        for gain in GAINS:
            controls = fit_temporal_controls(
                train_pairs, train_deltas, knots, ridge=ridge, gain=gain
            )
            reduction = lopo_modeled_reduction(
                output, train_pairs, knots, gt_pose, ridge=ridge, gain=gain
            )
            rows.append({
                "ridge": ridge,
                "gain": gain,
                "lopo_modeled_pose_mse_reduction": reduction,
                "nonzero_controls": int(np.count_nonzero(controls)),
                "controls_sha256": hashlib.sha256(controls.tobytes()).hexdigest(),
            })
    winner = max(
        rows,
        key=lambda row: (
            float(row["lopo_modeled_pose_mse_reduction"]),
            -int(row["nonzero_controls"]),
            -float(row["ridge"]),
            -float(row["gain"]),
        ),
    )
    controls = fit_temporal_controls(
        train_pairs,
        train_deltas,
        knots,
        ridge=float(winner["ridge"]),
        gain=float(winner["gain"]),
    )
    result = {
        "schema": "ddm_pk4_train_only_rung_tuning.v1",
        "candidate_denominator": len(rows),
        "selection_mode": "maximum train-only LOPO modeled pose-MSE reduction; then sparsity",
        "rows": rows,
        "winner": winner,
        "holdout_access_during_selection": False,
    }
    return controls, result


def _mlx_pose_vectors(posenet_cpu: Any, mlx_posenet: Any, inputs: np.ndarray, root: Path) -> np.ndarray:
    """Run the frozen PoseNet after retaining its exact real input.

    Default backend is the torch-CPU AUTHORITY. The MLX forward is opt-in
    (PK4_POSE_BACKEND=mlx): measured 2026-08-13 on retained batch inputs it
    drifts up to 0.0709 abs (~0.55% rel) from torch-CPU, which is what tripped
    the stage-11 parity gate at threshold 0.05 — the retained cp135 vectors
    are CPU-consistent (0.0038), so the drift is the MLX forward itself.
    """
    import torch

    value = np.asarray(inputs, dtype=np.uint8)
    retain_npy(root / "pose_input.uint8.npy", value)
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        yuv = posenet_cpu.preprocess_input(tensor).cpu().numpy().astype(np.float32, copy=False)
    retain_npy(root / "pose_preprocessed_yuv6.float32.npy", yuv)
    if os.environ.get("PK4_POSE_BACKEND", "torch_cpu") == "mlx":
        from tac.local_acceleration.mlx_scorer_adapters import run_mlx_posenet_nchw

        output = run_mlx_posenet_nchw(mlx_posenet, np.ascontiguousarray(yuv))
        pose = output["pose"] if isinstance(output, dict) else output
    else:
        with torch.inference_mode():
            out = posenet_cpu(torch.from_numpy(yuv))
        pose = out["pose"] if isinstance(out, dict) else out
        pose = pose.cpu().numpy()
    vectors = np.asarray(pose, dtype=np.float32)[:, :POSE_DIMENSIONS]
    retain_npy(root / "pose_vectors.float32.npy", vectors)
    return vectors


def evaluate_codes_mlx(
    *,
    surface: qs1.CP135Surface,
    posenet_cpu: Any,
    mlx_posenet: Any,
    codes: Sequence[np.ndarray],
    master: np.ndarray,
    pair: int,
    stage_root: Path,
) -> np.ndarray:
    code_array = np.stack([np.asarray(value, dtype=np.int32) for value in codes])
    vectors: list[np.ndarray] = []
    for first in range(0, len(code_array), POSE_BATCH):
        last = min(first + POSE_BATCH, len(code_array))
        root = stage_root / f"batch_{first:04d}_{last:04d}"
        result_path = root / "RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            vectors.append(np.load(require_record(result["pose_vectors"], "resumed MLX pose vectors"), allow_pickle=False))
            continue
        batch_codes = code_array[first:last]
        slaves = surface.render(batch_codes, pair)
        masters = np.repeat(np.asarray(master, dtype=np.uint8)[None], len(batch_codes), axis=0)
        inputs = np.stack((slaves, masters), axis=1)
        code_record = retain_npy(root / "codes.int32.npy", batch_codes)
        slave_record = retain_npy(root / "frame0_receiver.uint8.npy", slaves)
        observed = _mlx_pose_vectors(posenet_cpu, mlx_posenet, inputs, root)
        vector_record = qs1.file_record(root / "pose_vectors.float32.npy")
        result = {
            "schema": "ddm_pk4_mlx_pose_batch.v1",
            "pair": int(pair),
            "candidate_first": first,
            "candidate_last_exclusive": last,
            "codes": code_record,
            "frame0_receiver": slave_record,
            "pose_input": qs1.file_record(root / "pose_input.uint8.npy"),
            "pose_preprocessed_yuv6": qs1.file_record(root / "pose_preprocessed_yuv6.float32.npy"),
            "pose_vectors": vector_record,
            "axis": AXIS,
            "all_materialized_payloads_retained": True,
            "score_claim": False,
        }
        retain_json(result_path, result)
        vectors.append(observed)
    combined = np.concatenate(vectors, axis=0)
    if combined.shape != (len(code_array), POSE_DIMENSIONS):
        raise PK4Error("MLX pose-vector census differs")
    retain_npy(stage_root / "ALL_CODES.int32.npy", code_array)
    retain_npy(stage_root / "ALL_POSE_VECTORS.float32.npy", combined)
    return combined


def solve_pair(
    *, pair: int, surface: qs1.CP135Surface, posenet_cpu: Any, mlx_posenet: Any,
    raw: np.memmap, gt_pose: np.ndarray, output: Path,
) -> dict[str, Any]:
    root = output / "retained/jacobian_bank" / f"pair_{pair:03d}"
    result_path = root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    base_codes = surface.codes[pair].copy()
    master = np.asarray(raw[2 * pair + 1])
    rendered_base = surface.render(base_codes[None], pair)[0]
    mismatch = int(np.count_nonzero(rendered_base != np.asarray(raw[2 * pair])))
    if mismatch:
        raise PK4Error(f"exact CP135 frame0 surface mismatch at pair {pair}: {mismatch}")
    candidates = [base_codes.copy()]
    for dimension in range(DIMENSIONS):
        for delta in (-1, 1):
            candidate = base_codes.copy()
            candidate[dimension] += delta
            if not -2048 <= candidate[dimension] <= 2047:
                raise PK4Error("Jacobian perturbation reached an int12 endpoint")
            candidates.append(candidate)
    jacobian_vectors = evaluate_codes_mlx(
        surface=surface, posenet_cpu=posenet_cpu, mlx_posenet=mlx_posenet,
        codes=candidates, master=master, pair=pair, stage_root=root / "stage_10_jacobian",
    )
    base_vector = jacobian_vectors[0].astype(np.float64)
    jacobian = np.empty((POSE_DIMENSIONS, DIMENSIONS), dtype=np.float64)
    for dimension in range(DIMENSIONS):
        jacobian[:, dimension] = (
            jacobian_vectors[2 + 2 * dimension] - jacobian_vectors[1 + 2 * dimension]
        ) / 2.0
    retain_npy(root / "stage_10_jacobian/J_POSE0.float64.npy", jacobian)
    error = base_vector - np.asarray(gt_pose[pair], dtype=np.float64)
    gram = jacobian @ jacobian.T
    update = -jacobian.T @ np.linalg.solve(
        gram + GN_DAMPING * max(1.0, float(np.trace(gram))) * np.eye(POSE_DIMENSIONS), error
    )
    centre = base_codes + np.rint(np.clip(update, -MAX_GN_STEP, MAX_GN_STEP)).astype(np.int32)
    centre = np.clip(centre, -2048, 2047)
    retain_npy(root / "stage_20_gauss_newton/FLOAT_UPDATE.float64.npy", update)
    retain_npy(root / "stage_20_gauss_newton/CENTRE_CODES.int32.npy", centre)
    centre_vector = evaluate_codes_mlx(
        surface=surface, posenet_cpu=posenet_cpu, mlx_posenet=mlx_posenet,
        codes=(centre,), master=master, pair=pair, stage_root=root / "stage_30_integer_start",
    )[0]
    base_objective = float(np.mean((base_vector - gt_pose[pair]) ** 2))
    centre_objective = float(
        np.mean((centre_vector.astype(np.float64) - gt_pose[pair]) ** 2)
    )
    if centre_objective < base_objective:
        current = centre.copy()
        current_vector = centre_vector
        current_objective = centre_objective
    else:
        current = base_codes.copy()
        current_vector = jacobian_vectors[0].copy()
        current_objective = base_objective
    passes = 0
    while True:
        descent_codes = [current.copy()]
        for dimension in range(DIMENSIONS):
            for delta in (-1, 1):
                candidate = current.copy()
                candidate[dimension] += delta
                if -2048 <= candidate[dimension] <= 2047:
                    descent_codes.append(candidate)
        descent_vectors = evaluate_codes_mlx(
            surface=surface, posenet_cpu=posenet_cpu, mlx_posenet=mlx_posenet,
            codes=descent_codes, master=master, pair=pair,
            stage_root=root / f"stage_40_int12_descent/pass_{passes:04d}",
        )
        objectives = np.mean((descent_vectors.astype(np.float64) - gt_pose[pair][None]) ** 2, axis=1)
        retain_npy(root / f"stage_40_int12_descent/pass_{passes:04d}/OBJECTIVES.float64.npy", objectives)
        best = min(range(len(descent_codes)), key=lambda index: (float(objectives[index]), index))
        passes += 1
        if not float(objectives[best]) < current_objective:
            break
        current = np.asarray(descent_codes[best], dtype=np.int32)
        current_vector = np.asarray(descent_vectors[best], dtype=np.float32)
        current_objective = float(objectives[best])
    final_delta = current - base_codes
    result = {
        "schema": "ddm_pk4_exact_object_pair_solution.v1",
        "pair": pair,
        "frame1_is_exact_cp135_partner": True,
        "frame0_receiver_mismatch_values": mismatch,
        "base_pose_mse": base_objective,
        "final_pose_mse": current_objective,
        "improvement": base_objective - current_objective,
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "integer_descent_passes": passes,
        "base_codes": retain_npy(root / "BASE_CODES.int32.npy", base_codes),
        "final_codes": retain_npy(root / "FINAL_CODES.int32.npy", current),
        "final_delta": retain_npy(root / "FINAL_DELTA.int32.npy", final_delta),
        "final_pose_vector": retain_npy(root / "FINAL_POSE_VECTOR.float32.npy", current_vector),
        "axis": AXIS,
        "all_materialized_payloads_retained": True,
        "score_claim": False,
    }
    retain_json(result_path, result)
    return result


def build_bank(
    output: Path, plan: dict[str, Any], surface: qs1.CP135Surface,
    posenet_cpu: Any, mlx_posenet: Any,
) -> list[dict[str, Any]]:
    raw = np.memmap(
        qs1.CP135_RAW, mode="r", dtype=np.uint8,
        shape=(PAIR_COUNT * 2, qs1.CAMERA_H, qs1.CAMERA_W, 3),
    )
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
    results: list[dict[str, Any]] = []
    ordered = plan["train_pairs"] + plan["holdout_pairs"]
    for ordinal, pair in enumerate(ordered):
        results.append(solve_pair(
            pair=int(pair), surface=surface, posenet_cpu=posenet_cpu,
            mlx_posenet=mlx_posenet, raw=raw, gt_pose=gt_pose, output=output,
        ))
        atomic_json(output / "STATE.json", {
            "schema": "ddm_pk4_resume_state.v1",
            "stage": "jacobian_bank",
            "completed": ordinal + 1,
            "denominator": len(ordered),
            "last_pair": int(pair),
            "resume_from": str(output.resolve()),
        })
    retain_json(output / "checkpoints/stage_10_jacobian_bank_complete.json", {
        "schema": "ddm_pk4_jacobian_bank_complete.v1",
        "pair_denominator": len(results),
        "pairs": ordered,
        "complete": True,
    })
    return results


def _load_pair_delta(output: Path, pair: int) -> np.ndarray:
    return np.load(
        output / "retained/jacobian_bank" / f"pair_{pair:03d}/FINAL_DELTA.int32.npy",
        allow_pickle=False,
    )


def evaluate_overlay_pairs(
    *, output: Path, label: str, repeat: str, pairs: list[int], controls: np.ndarray,
    surface: qs1.CP135Surface, posenet_cpu: Any, mlx_posenet: Any,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.memmap(
        qs1.CP135_RAW, mode="r", dtype=np.uint8,
        shape=(PAIR_COUNT * 2, qs1.CAMERA_H, qs1.CAMERA_W, 3),
    )
    expanded = overlay.expand_pose_controls(controls)
    base_vectors = []
    candidate_vectors = []
    for pair in pairs:
        master = np.asarray(raw[2 * pair + 1])
        base_codes = surface.codes[pair]
        candidate_codes = base_codes + expanded[pair]
        if np.any(candidate_codes < -2048) or np.any(candidate_codes > 2047):
            raise PK4Error(f"rung {label} violates int12 at heldout pair {pair}")
        vectors = evaluate_codes_mlx(
            surface=surface, posenet_cpu=posenet_cpu, mlx_posenet=mlx_posenet,
            codes=(base_codes, candidate_codes), master=master, pair=pair,
            stage_root=output / "retained/rungs" / label / "heldout" / repeat / f"pair_{pair:03d}",
        )
        base_vectors.append(vectors[0])
        candidate_vectors.append(vectors[1])
    base = np.stack(base_vectors)
    candidate = np.stack(candidate_vectors)
    retain_npy(output / "retained/rungs" / label / "heldout" / repeat / "BASE_VECTORS.float32.npy", base)
    retain_npy(output / "retained/rungs" / label / "heldout" / repeat / "CANDIDATE_VECTORS.float32.npy", candidate)
    return base, candidate


def compile_rung(
    output: Path,
    label: str,
    controls: np.ndarray,
    surface: qs1.CP135Surface,
    gate_record: dict[str, Any],
) -> dict[str, Any]:
    gate_path = require_record(gate_record, f"{label} generalization gate")
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("schema") != "ddm_pk4_generalization_gate.v1"
        or gate.get("rung") != label
        or gate.get("passed") is not True
        or gate.get("lopo_positive") is not True
    ):
        raise PK4Error(f"rung {label} is not compile-eligible")
    root = output / "retained/rungs" / label / "compiled"
    result_path = root / "COMPILE_RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    payload = overlay.encode_pose_overlay(controls)
    expanded = overlay.expand_pose_controls(controls)
    candidate_codes = surface.codes + expanded
    if np.any(candidate_codes < -2048) or np.any(candidate_codes > 2047):
        raise PK4Error(f"rung {label} violates the full-n600 int12 lattice")
    stream_a, stream_b, carrier, suffix = pk3.rate_sources()
    carrier_source = carrier + payload
    stream_c = pk3._brotli(carrier_source, "-q", "11", "-c")
    if max(len(stream_a), len(stream_b), len(stream_c)) >= 1 << 16:
        raise PK4Error("split stream exceeds uint16")
    models = pk3.SPLIT_HEADER.pack(len(stream_a), len(stream_b), len(stream_c)) + stream_a + stream_b + stream_c
    member = models + suffix
    archive = pk3.deterministic_zip(member)
    archive_repeat = pk3.deterministic_zip(member)
    records = {
        "controls": retain_npy(root / "controls.int32.npy", controls),
        "expanded_deltas": retain_npy(root / "expanded_deltas.int32.npy", expanded),
        "overlay": retain_bytes(root / "pose_overlay.p0j2", payload),
        "carrier_source": retain_bytes(root / "carrier_selector_plus_overlay.raw", carrier_source),
        "carrier_stream": retain_bytes(root / "carrier_selector_plus_overlay.q11.br", stream_c),
        "member": retain_bytes(root / "p", member),
        "archive": retain_bytes(root / "archive.zip", archive),
        "archive_repeat": retain_bytes(root / "archive.repeat.zip", archive_repeat),
    }
    if records["archive"]["sha256"] != records["archive_repeat"]["sha256"]:
        raise PK4Error("deterministic archive repeat differs")
    runtime_root = root / "adapted_runtime"
    runtime_copy = jo1.copy_runtime(runtime_root, archive)
    patches = qs2.patch_runtime(runtime_root)
    runtime_overlay = runtime_root / "runtime/compensation_overlay.py"
    if runtime_overlay.read_bytes() != qs2.RUNTIME_OVERLAY_SOURCE.read_bytes():
        raise PK4Error("generic QS2 overlay patch surface differs before P0J2 replacement")
    replacement = (REPO / "experiments/ddm_pk4_frame0_pose_overlay_runtime.py").read_bytes()
    partial = _atomic_path(runtime_overlay)
    try:
        with partial.open("wb") as stream:
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, runtime_overlay)
    finally:
        partial.unlink(missing_ok=True)
    patches["overlay"] = qs1.file_record(runtime_overlay)
    parseback = qs2.runtime_parseback(
        runtime_root=runtime_root,
        archive=runtime_root / "archive.zip",
        expected_overlay=records["overlay"],
    )
    result = {
        "schema": "ddm_pk4_byte_closed_rung.v1",
        "label": label,
        "raw_overlay_bytes": len(payload),
        "archive_delta_bytes_vs_cp135": records["archive"]["bytes"] - 186_252,
        "records": records,
        "runtime_root": str(runtime_root.resolve()),
        "runtime_copy": runtime_copy,
        "runtime_patches": patches,
        "runtime_parseback": parseback,
        "runtime_tree": pk3.cp135.tree_record(runtime_root),
        "receiver_parseback_exact": True,
        "archive_repeat_byte_identical": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(result_path, result)
    return result


def seal_t4_order(output: Path, label: str, compiled: dict[str, Any]) -> dict[str, Any]:
    """Seal one unchanged-worker request; Pose is structurally unknown until T4."""
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b

    root = output / "retained/rungs" / label / "fire_order"
    input_root = root / "fire_inputs"
    archive_path = Path(compiled["records"]["archive"]["path"])
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        Path(compiled["runtime_root"]), label=f"ddm_pk4_{label}_p0j2"
    )
    screen = {
        "schema": "ddm_pk4_pose_screen.v1",
        "rung": label,
        "generalization_gate": qs1.file_record(output / "retained/rungs" / label / "GENERALIZATION_GATE.json"),
        "archive_delta_bytes_vs_cp135": compiled["archive_delta_bytes_vs_cp135"],
        "score_claim": False,
    }
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "POSE_SCREEN_RESULT.json": (json.dumps(screen, indent=2, sort_keys=True) + "\n").encode(),
    }
    for name, payload in payloads.items():
        retain_bytes(input_root / name, payload)
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    run_id = f"ddm_pk4_{label}_dual_axis_20260813_r1"
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": run_id,
        "resume_from": run_id,
        "lane_id": f"ddm_pk4_{label}_dual_axis_n600_20260813",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "MAIN",
        "seed": SEED,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": qs1.file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": runtime_manifest,
        "inputs": {name: js1b.payload_record(payload) for name, payload in payloads.items()},
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "dispatcher_source_sha256": qs1.sha256_file(REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"),
        "worker_source_sha256": qs1.sha256_file(REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"),
        "js1b_worker_source_sha256": qs1.sha256_file(REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"),
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_record = retain_json(root / "SEALED_REQUEST.json", request)
    command = [
        ".venv/bin/modal", "run", "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request", request_record["path"],
        "--fire-input-dir", str(input_root.resolve()),
        "--expected-request-sha256", request_record["sha256"],
        "--output-dir", str((output / "dispatch" / run_id).resolve()),
        "--detach", "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_pk4_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED_WITH_FIRE_ORDER",
        "rung": label,
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": "heldout gate passed; MAIN observes no active n600 scorer/Modal lane, claims this exact lane, then verifies all sealed hashes",
        "request": request_record,
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "unchanged_worker": "experiments/ddm_re1t_t4_sign_gate_worker.py",
        "score_claim": False,
    }
    retain_json(root / "SEALED_FIRE_ORDER.json", order)
    no_fire = root.parent / "SEALED_NO_FIRE_ORDER_PREFLIGHT.json"
    if no_fire.is_file():
        retain_json(root / "NO_FIRE_ORDER_SUPERSEDED.json", {
            "schema": "ddm_pk4_no_fire_order_superseded.v1",
            "prior": qs1.file_record(no_fire),
            "superseded_by": qs1.file_record(root / "SEALED_FIRE_ORDER.json"),
            "reason": "this rung passed the fresh exact-object heldout and LOPO gate",
        })
    return order


def write_no_fire_order(
    output: Path, label: str, reason: str, disposition: str, *, phase: str
) -> None:
    knots = RUNG_KNOTS[label]
    root = output / "retained/rungs" / label
    order = {
        "schema": "ddm_pk4_sealed_no_fire_order.v1",
        "sealed": True,
        "rung": label,
        "raw_overlay_target_bytes": overlay.encoded_bytes_for_knots(knots),
        "disposition": disposition,
        "owner": "MAIN",
        "consumer_store": str(output.resolve()),
        "fire_trigger": "NONE until the local Metal Jacobian bank and this rung's >=2sigma heldout gate pass",
        "reason": reason,
        "score_claim": False,
    }
    retain_json(root / f"SEALED_NO_FIRE_ORDER_{phase.upper()}.json", order)


def write_no_fire_orders(output: Path, reason: str, disposition: str) -> None:
    for label in RUNG_KNOTS:
        write_no_fire_order(output, label, reason, disposition, phase="preflight")


def prepare(output: Path, sample_count: int) -> dict[str, Any]:
    storage = storage_preflight(output, sample_count)
    sources = source_preflight(output)
    plan = retain_sample_plan(output, sample_count)
    metal = probe_metal()
    atomic_json(output / "METAL_PROBE.json", metal)
    retain_json(output / "retained/preflight/METAL_PROBE_AT_PREPARE.json", metal)
    lane_summary = subprocess.run(
        [str(REPO / ".venv/bin/python"), "tools/claim_lane_dispatch.py", "summary"],
        cwd=REPO, check=False, capture_output=True, text=True,
    )
    atomic_json(output / "LANE_SUMMARY.json", {
        "schema": "ddm_pk4_lane_summary.v1",
        "command": [".venv/bin/python", "tools/claim_lane_dispatch.py", "summary"],
        "returncode": lane_summary.returncode,
        "stdout": lane_summary.stdout,
        "stderr": lane_summary.stderr,
    })
    if not metal["passed"]:
        disposition = "BLOCKED_BEFORE_SCORER_LAUNCH_NO_METAL"
        reason = f"local MLX Metal probe failed: {metal.get('error_type')}: {metal.get('error')}"
    else:
        disposition = "BLOCKED_BEFORE_SCORER_LAUNCH_NO_OWNERSHIP_RECEIPT"
        reason = "Metal is available but MAIN has not supplied a current single-flight scorer-ownership receipt"
    write_no_fire_orders(output, reason, disposition)
    local_order = {
        "schema": "ddm_pk4_local_metal_fire_order.v1",
        "sealed": True,
        "disposition": disposition,
        "owner": "MAIN local scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": "Metal probe passes and MAIN owns the only active full-n600 scorer slot with a current receipt at the pinned path",
        "ownership_receipt_path": str(OWNERSHIP_RECEIPT.resolve()),
        "exact_command_argv": [
            str(REPO / ".venv/bin/python"),
            str((REPO / "experiments/ddm_pk4_optimal_form_frame0_pose.py").resolve()),
            "measure", "--output", str(output.resolve()), "--resume-from", str(output.resolve()),
            "--sample-count", str(sample_count), "--scorer-ownership-receipt", str(OWNERSHIP_RECEIPT.resolve()),
        ],
        "resume_from": str(output.resolve()),
        "reason": reason,
        "score_claim": False,
    }
    retain_json(output / "LOCAL_METAL_FIRE_ORDER.json", local_order)
    final = {
        "schema": "ddm_pk4_final_result.v1",
        "run_id": RUN_ID,
        "status": disposition,
        "storage_preflight": storage,
        "source_preflight": sources["_checkpoint_record"],
        "sample_plan": qs1.file_record(output / "checkpoints/stage_05_sample_plan.json"),
        "metal_probe": qs1.file_record(output / "METAL_PROBE.json"),
        "local_fire_order": qs1.file_record(output / "LOCAL_METAL_FIRE_ORDER.json"),
        "sample_denominator": plan["sample_count"],
        "scorer_calls": 0,
        "jacobian_pair_denominator": 0,
        "byte_closed_rung_denominator": 0,
        "generalization_gate_denominator": 0,
        "exact_eval_denominator": 0,
        "all_materialized_payloads_retained": True,
        "reason": reason,
        "frontier_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "honesty_boundary": "preflight and generic code only; no Jacobian, pose curve, generalization verdict, archive rung, d_pose, or score was measured",
        "sources_pinned": len(sources["sources"]),
    }
    atomic_json(output / "FINAL_RESULT.json", final)
    return final


def measure(output: Path, sample_count: int, receipt: Path) -> dict[str, Any]:
    storage_preflight(output, sample_count)
    source_preflight(output)
    plan = retain_sample_plan(output, sample_count)
    metal = probe_metal()
    atomic_json(output / "METAL_PROBE.json", metal)
    retain_json(output / "checkpoints/stage_06_metal_probe.json", metal)
    if not metal["passed"]:
        raise PK4Error(f"Metal is unavailable: {metal}")
    ownership = validate_ownership_receipt(receipt, output)
    retain_json(output / "checkpoints/stage_06_scorer_ownership.json", ownership)
    import mlx.core as mx
    import torch

    from tac.local_acceleration.mlx_scorer_adapters import torch_posenet_to_mlx

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    mx.random.seed(SEED)
    surface, _ = qs1.CP135Surface.load()
    posenet_cpu = qs1.load_posenet()
    mlx_posenet = torch_posenet_to_mlx(posenet_cpu)
    build_bank(output, plan, surface, posenet_cpu, mlx_posenet)
    retained_base = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
    bank_vectors = np.stack([
        np.load(
            output / "retained/jacobian_bank" / f"pair_{int(pair):03d}"
            / "stage_10_jacobian/ALL_POSE_VECTORS.float32.npy",
            allow_pickle=False,
        )[0]
        for pair in plan["train_pairs"] + plan["holdout_pairs"]
    ])
    bank_pair_indices = np.asarray(plan["train_pairs"] + plan["holdout_pairs"], dtype=np.int64)
    parity_max_abs = float(np.max(np.abs(bank_vectors - retained_base[bank_pair_indices])))
    parity = {
        "schema": "ddm_pk4_mlx_posenet_parity.v1",
        "pair_denominator": len(bank_pair_indices),
        "max_abs_vs_retained_cp135_pose_vectors": parity_max_abs,
        "threshold": 5e-2,
        "passed": parity_max_abs <= 5e-2,
        "axis": AXIS,
    }
    retain_json(output / "checkpoints/stage_11_mlx_parity.json", parity)
    if not parity["passed"]:
        raise PK4Error(f"MLX PoseNet parity failed: {parity}")
    train_pairs = np.asarray(plan["train_pairs"], dtype=np.int16)
    train_deltas = np.stack([_load_pair_delta(output, int(pair)) for pair in train_pairs])
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
    rows = []
    for label, knots in RUNG_KNOTS.items():
        controls, tuning = select_rung_controls(
            output, train_pairs, train_deltas, knots, gt_pose
        )
        lopo_reduction = float(tuning["winner"]["lopo_modeled_pose_mse_reduction"])
        root = output / "retained/rungs" / label
        retain_npy(root / "controls.int32.npy", controls)
        retain_json(root / "TRAIN_ONLY_TUNING.json", tuning)
        base_first, candidate_first = evaluate_overlay_pairs(
            output=output, label=label, repeat="first", pairs=plan["holdout_pairs"],
            controls=controls, surface=surface, posenet_cpu=posenet_cpu, mlx_posenet=mlx_posenet,
        )
        base_repeat, candidate_repeat = evaluate_overlay_pairs(
            output=output, label=label, repeat="repeat", pairs=plan["holdout_pairs"],
            controls=controls, surface=surface, posenet_cpu=posenet_cpu, mlx_posenet=mlx_posenet,
        )
        gate = generalization_gate(
            base_first, candidate_first, base_repeat, candidate_repeat,
            gt_pose[np.asarray(plan["holdout_pairs"], dtype=np.int64)],
            lopo_modeled_pose_mse_reduction=lopo_reduction,
        )
        gate.update({
            "rung": label,
            "knots": knots,
            "raw_overlay_bytes": overlay.encoded_bytes_for_knots(knots),
            "train_pair_denominator": len(train_pairs),
            "fit_inputs": "only FINAL_DELTA.int32 from train_pairs",
            "axis": AXIS,
        })
        gate_record = retain_json(root / "GENERALIZATION_GATE.json", gate)
        if gate["passed"]:
            compiled = compile_rung(output, label, controls, surface, gate_record)
            order = seal_t4_order(output, label, compiled)
            row = {"rung": label, "gate": gate, "compiled": compiled, "order": order}
        else:
            write_no_fire_order(
                output, label,
                f"{label} heldout reduction failed the positive >=2sigma gate",
                "GATE_FAIL_NO_COMPILE",
                phase="gate",
            )
            row = {"rung": label, "gate": gate, "compiled": None, "order": None}
        rows.append(row)
        retain_json(output / "checkpoints" / f"stage_20_{label}.json", row)
        atomic_json(output / "STATE.json", {
            "schema": "ddm_pk4_resume_state.v1", "stage": "rungs",
            "completed": len(rows), "denominator": len(RUNG_KNOTS), "last_rung": label,
        })
    curve = {
        "schema": "ddm_pk4_pose_reach_curve.v1",
        "rows": rows,
        "rung_denominator": len(rows),
        "byte_closed_denominator": sum(row["compiled"] is not None for row in rows),
        "selection_mode": "no rung selected locally; every passing rung gets its own sealed T4 request",
        "axis": AXIS,
        "score_claim": False,
    }
    retain_json(output / "POSE_REACH_CURVE.json", curve)
    final = {
        "schema": "ddm_pk4_final_result.v1",
        "run_id": RUN_ID,
        "status": "MEASURED_LOCAL_GATE_COMPLETE",
        "sample_denominator": plan["sample_count"],
        "jacobian_pair_denominator": plan["sample_count"],
        "generalization_gate_denominator": len(rows),
        "byte_closed_rung_denominator": curve["byte_closed_denominator"],
        "exact_eval_denominator": 0,
        "reach_curve": qs1.file_record(output / "POSE_REACH_CURVE.json"),
        "all_materialized_payloads_retained": True,
        "frontier_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(output / "FINAL_RESULT.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "measure"):
        command = subparsers.add_parser(name)
        command.add_argument("--output", type=Path, default=OUTPUT)
        command.add_argument("--resume-from", type=Path, default=OUTPUT)
        command.add_argument("--sample-count", type=int, default=MIN_SAMPLE_COUNT)
        if name == "measure":
            command.add_argument("--scorer-ownership-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() != args.resume_from.resolve():
        raise PK4Error("--resume-from must identify the exact output root")
    if args.command == "prepare":
        result = prepare(args.output, args.sample_count)
    else:
        result = measure(args.output, args.sample_count, args.scorer_ownership_receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
