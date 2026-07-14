#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed-point frozen-SegNet/PoseNet forward feasibility on real n600.

The primary ladder uses fixed calibration-only activation scales.  A distinct
dynamic arm uses the finite runtime tensor's order-invariant absolute maximum
to remove held-out range clipping.  Both use symmetric per-output-channel
weights and fp32 accumulation, measure numerical feasibility/tie-margin
certificates, and claim neither native integer MAC nor placement/speed/score
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "upstream"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
    stored_npy_memmap,
)
from tac.local_acceleration.calibrated_fixedpoint_scorer import (  # noqa: E402
    ActivationAbsMaxCalibrator,
    FixedPointForwardPolicy,
    build_calibrated_qdq_model,
)

SCHEMA = "fixedpoint_scorer_forward_n600.v2"
DYNAMIC_SCHEMA = "dynamic_fixedpoint_scorer_forward_n600.v1"
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "fixedpoint_scorer_forward_n600_v2.json"
)
DEFAULT_DYNAMIC_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_BITS = (8, 10, 12, 14, 16, 18, 20, 22, 24)
DYNAMIC_BITS = (16, 18, 20, 22, 24)
# The real frozen SegNet has maximum Conv2d fan-in 4,248. W26A26 has the
# static worst-case bound 4,782,822,519,189,016,728 < 2**63 and is therefore
# the last uniform signed precision that one exact int64 accumulator can carry.
# W27A27 exceeds int64. This separate ladder is a preregistered ceiling check,
# not an open-ended precision sweep.
DYNAMIC_INT64_CEILING_BITS = (25, 26)
CALIBRATION_START = 0
CALIBRATION_STOP = 120
HELDOUT_START = 120
N600 = 600
TRAINING_TOLERANCE = 3.3e-5
POSE_CACHE_SELFCHECK_ATOL = 1.0e-4
MARGIN_QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.5)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return _hash_bytes(f"{array.dtype.str}:{array.shape}".encode("ascii") + array.tobytes())


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_models(*, include_pose: bool) -> tuple[Any, Any | None, Path, Path | None]:
    import torch
    from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path
    from safetensors.torch import load_file

    if torch.backends.mps.is_available() and torch.tensor(0).device.type != "cpu":
        raise RuntimeError("MPS is forbidden for this authority-feasibility probe")
    seg_weights = Path(segnet_sd_path)
    seg = SegNet().eval().cpu()
    seg.load_state_dict(load_file(str(seg_weights), device="cpu"))
    pose = None
    pose_weights = None
    if include_pose:
        pose_weights = Path(posenet_sd_path)
        pose = PoseNet().eval().cpu()
        pose.load_state_dict(load_file(str(pose_weights), device="cpu"))
    return seg, pose, seg_weights, pose_weights


def _cache_arrays(path: Path) -> dict[str, np.memmap]:
    arrays = {
        name: stored_npy_memmap(path, name)
        for name in ("gt_f0.npy", "gt_f1.npy", "lstars.npy", "margins.npy", "gt_poses.npy")
    }
    expected = {
        "gt_f0.npy": ((600, 874, 1164, 3), np.dtype(np.uint8)),
        "gt_f1.npy": ((600, 874, 1164, 3), np.dtype(np.uint8)),
        "lstars.npy": ((600, 384, 512), np.dtype(np.int64)),
        "margins.npy": ((600, 384, 512), np.dtype(np.float32)),
        "gt_poses.npy": ((600, 6), np.dtype(np.float64)),
    }
    for name, array in arrays.items():
        shape, dtype = expected[name]
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise ValueError(f"{name} custody mismatch: {array.dtype} {array.shape}")
    return arrays


def _pair_inputs(arrays: dict[str, np.memmap], pair_index: int) -> tuple[Any, Any]:
    import torch

    f0 = np.asarray(arrays["gt_f0.npy"][pair_index])
    f1 = np.asarray(arrays["gt_f1.npy"][pair_index])
    pair = np.stack([f0, f1], axis=0)
    btchw = torch.from_numpy(np.ascontiguousarray(pair.transpose(0, 3, 1, 2))).unsqueeze(0)
    return btchw.to(torch.float32), f1


def _calibrate(
    seg: Any, pose: Any | None, arrays: dict[str, np.memmap]
) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch

    seg_observer = ActivationAbsMaxCalibrator(seg, model_kind="SegNet")
    pose_observer = (
        ActivationAbsMaxCalibrator(pose, model_kind="PoseNet") if pose is not None else None
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for pair_index in range(CALIBRATION_START, CALIBRATION_STOP):
            btchw, _ = _pair_inputs(arrays, pair_index)
            seg(seg.preprocess_input(btchw))
            if pose is not None:
                pose(pose.preprocess_input(btchw))
    seg_calibration = seg_observer.freeze()
    pose_calibration = pose_observer.freeze() if pose_observer is not None else None
    manifest = {
        "split": [CALIBRATION_START, CALIBRATION_STOP],
        "selection_labels": "calibration inputs only; no heldout labels or outputs",
        "segnet_digest": seg_calibration.digest(),
        "segnet_operator_absmax": seg_calibration.operator_absmax,
        "segnet_operator_observations": seg_calibration.operator_observations,
        "posenet_digest": pose_calibration.digest() if pose_calibration else None,
        "posenet_operator_absmax": pose_calibration.operator_absmax if pose_calibration else None,
        "posenet_operator_observations": (
            pose_calibration.operator_observations if pose_calibration else None
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return seg_calibration, pose_calibration, manifest


def _arm_specs(bits: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = [
        {"name": f"w{bit}a{bit}", "bits": bit, "mixed_head_fp32": False}
        for bit in bits
    ]
    if 8 in bits:
        rows.append({"name": "w8a8_head_fp32", "bits": 8, "mixed_head_fp32": True})
    return rows


def _build_candidates(
    *,
    seg: Any,
    pose: Any | None,
    seg_calibration: Any,
    pose_calibration: Any | None,
    bits: tuple[int, ...],
    activation_scale_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seg_candidates: dict[str, Any] = {}
    pose_candidates: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for spec in _arm_specs(bits):
        name = str(spec["name"])
        bit = int(spec["bits"])
        mixed = bool(spec["mixed_head_fp32"])
        seg_policy = FixedPointForwardPolicy(
            bits=bit,
            activation_scale_mode=activation_scale_mode,
            skipped_module_prefixes=("segmentation_head",) if mixed else (),
        )
        seg_candidate, seg_manifest = build_calibrated_qdq_model(
            seg, seg_calibration, seg_policy
        )
        seg_candidates[name] = seg_candidate
        pose_manifest = None
        if pose is not None and pose_calibration is not None:
            pose_policy = FixedPointForwardPolicy(
                bits=bit,
                activation_scale_mode=activation_scale_mode,
                skipped_module_prefixes=("hydra.final_layer",) if mixed else (),
            )
            pose_candidate, pose_manifest = build_calibrated_qdq_model(
                pose, pose_calibration, pose_policy
            )
            pose_candidates[name] = pose_candidate
        manifests[name] = {"segnet": seg_manifest, "posenet": pose_manifest}
    return seg_candidates, pose_candidates, manifests


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return None
    quantiles = np.quantile(array, np.asarray(MARGIN_QUANTILES), method="linear")
    return {
        f"q{value:g}": float(result)
        for value, result in zip(MARGIN_QUANTILES, quantiles, strict=True)
    }


def _seg_row(
    *,
    pair_index: int,
    reference_logits: Any,
    reference_argmax: np.ndarray,
    baseline_margin: np.ndarray,
    candidate_logits: Any,
) -> dict[str, Any]:
    import torch

    candidate_np = candidate_logits.detach().cpu().numpy().astype(np.float32, copy=False)
    reference_np = reference_logits.detach().cpu().numpy().astype(np.float32, copy=False)
    candidate_argmax = np.argmax(candidate_np, axis=1)[0]
    flips = candidate_argmax != reference_argmax
    error = np.abs(candidate_np - reference_np)
    class_error = np.max(error, axis=1)[0]
    uncertified = baseline_margin <= (np.float32(2.0) * class_error)
    difference = candidate_np.astype(np.float64) - reference_np.astype(np.float64)
    return {
        "pair_index": int(pair_index),
        "split": "calibration" if pair_index < HELDOUT_START else "heldout",
        "flips": int(np.count_nonzero(flips)),
        "pixels": int(flips.size),
        "flip_fraction": float(np.mean(flips)),
        "candidate_argmax_sha256": _hash_array(candidate_argmax),
        "reference_argmax_sha256": _hash_array(reference_argmax),
        "max_abs_logit_error": float(np.max(error)),
        "sum_squared_logit_error": float(np.sum(difference * difference, dtype=np.float64)),
        "logit_elements": int(difference.size),
        "uncertified_pixels": int(np.count_nonzero(uncertified)),
        "baseline_margin_min": float(np.min(baseline_margin)),
        "baseline_margin_pair_quantiles": _quantiles(baseline_margin),
        "flipped_pixel_margin_quantiles": _quantiles(baseline_margin[flips]),
        "candidate_logits_sha256": _hash_array(candidate_np),
        "reference_logits_sha256": _hash_array(reference_np),
        "torch_reference_device": str(reference_logits.device),
        "candidate_device": str(candidate_logits.device),
        "torch_threads": int(torch.get_num_threads()),
    }


def _pose_row(*, pair_index: int, reference: np.ndarray, candidate: Any) -> dict[str, Any]:
    candidate_np = candidate["pose"][0, :6].detach().cpu().numpy().astype(np.float64)
    difference = candidate_np - np.asarray(reference, dtype=np.float64)
    mse = float(np.mean(difference * difference))
    return {
        "pair_index": int(pair_index),
        "split": "calibration" if pair_index < HELDOUT_START else "heldout",
        "first_six_sha256": _hash_array(candidate_np),
        "max_abs_error": float(np.max(np.abs(difference))),
        "sum_squared_error": float(np.sum(difference * difference)),
        "elements": 6,
        "d_pose": mse,
        "sqrt_10_d_pose": float(np.sqrt(10.0 * mse)),
    }


def _aggregate_rows(rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    selected = rows if split == "full" else [row for row in rows if row["split"] == split]
    if not selected:
        return {
            "pairs": 0,
            "status": "INCOMPLETE",
            "argmax_exact_gate": False,
            "training_tolerance_gate": False,
        }
    flips = sum(int(row["flips"]) for row in selected)
    pixels = sum(int(row["pixels"]) for row in selected)
    elements = sum(int(row["logit_elements"]) for row in selected)
    sum_squared = sum(float(row["sum_squared_logit_error"]) for row in selected)
    worst = max(selected, key=lambda row: (float(row["flip_fraction"]), -int(row["pair_index"])))
    digest = hashlib.sha256()
    for row in sorted(selected, key=lambda item: int(item["pair_index"])):
        digest.update(
            f"{row['pair_index']}:{row['candidate_argmax_sha256']}\n".encode("ascii")
        )
    return {
        "pairs": len(selected),
        "flips": flips,
        "pixels": pixels,
        "aggregate_flip_fraction": float(flips / pixels) if pixels else None,
        "worst_pair_index": int(worst["pair_index"]),
        "worst_pair_flip_fraction": float(worst["flip_fraction"]),
        "argmax_corpus_sha256": digest.hexdigest(),
        "max_abs_logit_error": max(float(row["max_abs_logit_error"]) for row in selected),
        "rmse_logit_error": float(np.sqrt(sum_squared / elements)) if elements else None,
        "uncertified_pixels": sum(int(row["uncertified_pixels"]) for row in selected),
        "uncertified_fraction": (
            float(sum(int(row["uncertified_pixels"]) for row in selected) / pixels)
            if pixels
            else None
        ),
        "baseline_margin_exact_min": min(float(row["baseline_margin_min"]) for row in selected),
        "margin_quantile_semantics": "quantiles of exact per-pair quantiles, not pooled pixels",
        "per_pair_margin_quantile_distribution": {
            key: _quantiles(
                np.asarray(
                    [row["baseline_margin_pair_quantiles"][key] for row in selected],
                    dtype=np.float64,
                )
            )
            for key in selected[0]["baseline_margin_pair_quantiles"]
        },
        "argmax_exact_gate": flips == 0,
        "training_tolerance_gate": bool(
            pixels
            and flips / pixels <= TRAINING_TOLERANCE
            and float(worst["flip_fraction"]) <= TRAINING_TOLERANCE
        ),
    }


def _aggregate_pose(rows: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    selected = rows if split == "full" else [row for row in rows if row["split"] == split]
    if not selected:
        return {"pairs": 0, "status": "INCOMPLETE", "exact_first_six_gate": False}
    elements = sum(int(row["elements"]) for row in selected)
    sum_squared = sum(float(row["sum_squared_error"]) for row in selected)
    d_pose = float(sum_squared / elements) if elements else None
    return {
        "pairs": len(selected),
        "max_abs_error": max(float(row["max_abs_error"]) for row in selected),
        "d_pose": d_pose,
        "sqrt_10_d_pose": float(np.sqrt(10.0 * d_pose)) if d_pose is not None else None,
        "exact_first_six_gate": all(float(row["max_abs_error"]) == 0.0 for row in selected),
        "worst_pair_index": int(
            max(selected, key=lambda row: (float(row["d_pose"]), -int(row["pair_index"])))[
                "pair_index"
            ]
        ),
    }


def _aggregate_cache_custody(
    rows: list[dict[str, Any]], *, expected_indices: set[int]
) -> dict[str, Any]:
    observed_indices = [int(row["pair_index"]) for row in rows]
    if len(rows) != len(expected_indices) or set(observed_indices) != expected_indices:
        return {
            "status": "INCOMPLETE",
            "pairs": len(rows),
            "unique_pair_indices": len(set(observed_indices)),
            "expected_pair_indices_sha256": _hash_array(
                np.asarray(sorted(expected_indices), dtype=np.int64)
            ),
            "observed_pair_indices_sha256": _hash_array(
                np.asarray(sorted(observed_indices), dtype=np.int64)
            ),
        }
    ordered = sorted(rows, key=lambda row: int(row["pair_index"]))
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(
            (
                f"{row['pair_index']}:{row['one_thread_argmax_sha256']}:"
                f"{row['cached_argmax_sha256']}\n"
            ).encode("ascii")
        )
    return {
        "status": "MEASURED",
        "pairs": len(ordered),
        "unique_pair_indices": len(set(observed_indices)),
        "expected_pair_indices_sha256": _hash_array(
            np.asarray(sorted(expected_indices), dtype=np.int64)
        ),
        "observed_pair_indices_sha256": _hash_array(
            np.asarray(sorted(observed_indices), dtype=np.int64)
        ),
        "mismatch_pairs": sum(int(row["argmax_mismatch_pixels"] > 0) for row in ordered),
        "argmax_mismatch_pixels": sum(
            int(row["argmax_mismatch_pixels"]) for row in ordered
        ),
        "worst_pair_mismatch_pixels": max(
            int(row["argmax_mismatch_pixels"]) for row in ordered
        ),
        "maximum_margin_abs_delta": max(
            float(row["margin_max_abs_delta"]) for row in ordered
        ),
        "one_thread_vs_cache_corpus_sha256": digest.hexdigest(),
        "interpretation": (
            "cache was built by the legacy ambient-thread GT path; any mismatch is an explicit "
            "thread/reduction-geometry custody delta, not silently coerced authority"
        ),
    }


def summarize(receipt: dict[str, Any]) -> dict[str, Any]:
    expected_pairs = int(receipt["contract"]["pair_count"])
    expected_indices = set(
        range(
            int(receipt["contract"]["pair_start"]),
            int(receipt["contract"]["pair_start"]) + expected_pairs,
        )
    )
    arms: dict[str, Any] = {}
    for name, state in receipt["arms"].items():
        seg_rows = list(state["segnet_rows"])
        observed_indices = [int(row["pair_index"]) for row in seg_rows]
        if len(seg_rows) != expected_pairs or set(observed_indices) != expected_indices:
            arms[name] = {
                "status": "INCOMPLETE",
                "pairs": len(seg_rows),
                "unique_pair_indices": len(set(observed_indices)),
            }
            continue
        seg_splits = {
            split: _aggregate_rows(seg_rows, split=split)
            for split in ("calibration", "heldout", "full")
        }
        pose_rows = list(state.get("posenet_rows", []))
        pose_splits = (
            {
                split: _aggregate_pose(pose_rows, split=split)
                for split in ("calibration", "heldout", "full")
            }
            if len(pose_rows) == expected_pairs
            else {"status": "OWED" if receipt["contract"]["include_pose"] else "NOT_REQUESTED"}
        )
        arms[name] = {
            "status": "MEASURED",
            "segnet": seg_splits,
            "posenet": pose_splits,
            "argmax_exact_admitted": bool(
                seg_splits["heldout"]["argmax_exact_gate"]
                and seg_splits["full"]["argmax_exact_gate"]
            ),
            "training_tolerance_admitted": bool(
                seg_splits["heldout"]["training_tolerance_gate"]
                and seg_splits["full"]["training_tolerance_gate"]
            ),
        }
    measured = {name: row for name, row in arms.items() if row.get("status") == "MEASURED"}

    def minimum_gate(field: str) -> str | None:
        candidates = []
        for spec in receipt["contract"]["arms"]:
            name = str(spec["name"])
            if name != "fp32_control" and name in measured and measured[name].get(field):
                candidates.append((int(spec["bits"]), bool(spec["mixed_head_fp32"]), name))
        return min(candidates)[2] if candidates else None

    complete = len(measured) == len(receipt["contract"]["arms"])
    cache_custody = _aggregate_cache_custody(
        list(receipt.get("cache_custody", {}).get("segnet_rows", [])),
        expected_indices=expected_indices,
    )
    full_n600 = (
        int(receipt["contract"]["pair_start"]) == 0
        and expected_pairs == N600
        and all(row.get("status") == "MEASURED" for row in arms.values())
        and cache_custody.get("status") == "MEASURED"
    )
    return {
        "status": "MEASURED" if complete else "INCOMPLETE",
        "full_real_n600": full_n600,
        "arms": arms,
        "cache_custody": cache_custody,
        "minimum_argmax_exact_arm": minimum_gate("argmax_exact_admitted"),
        "minimum_training_tolerance_arm": minimum_gate("training_tolerance_admitted"),
        "rung2_verdict": (
            "ARGMAX_FIXEDPOINT_FEASIBLE"
            if full_n600 and minimum_gate("argmax_exact_admitted")
            else "NO_ADMITTED_PRECISION_IN_LADDER"
            if full_n600
            else "INCOMPLETE"
        ),
        "verdict_scope": (
            "n600 INSTANCE: "
            + (
                "label-free dynamic max-absolute-scale"
                if receipt["contract"].get(
                    "activation_scale_mode", "fixed_calibration"
                )
                == "dynamic_exact_absmax"
                else "calibrated fixed-scale"
            )
            + " WnAn QDQ/fp32 accumulation on frozen one-thread CPU-Torch SegNet "
            "control; legacy cache-thread divergence audited separately; PoseNet "
            "continuous first-six reported separately when requested"
        ),
    }


def _contract_fingerprint(payload: dict[str, Any]) -> str:
    return _hash_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def finalize_existing_receipt(path: Path) -> dict[str, Any]:
    """Re-derive summary fields without laundering the original row producer."""

    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") not in {SCHEMA, DYNAMIC_SCHEMA}:
        raise ValueError("cannot finalize an incompatible fixed-point receipt")
    producer_sha = receipt.get("custody", {}).get("probe_sha256")
    if not isinstance(producer_sha, str) or len(producer_sha) != 64:
        raise ValueError("receipt lacks original numerical-row producer custody")
    derived = summarize(receipt)
    if derived.get("status") != "MEASURED":
        raise ValueError("summary-only finalization requires all preregistered rows")
    current_sha = sha256_file(Path(__file__))
    receipt["summary"] = derived
    receipt["completed"] = True
    receipt["summary_finalization"] = {
        "producer_probe_sha256": producer_sha,
        "finalizer_probe_sha256": current_sha,
        "numerical_rows_recomputed": False,
        "summary_rederived_from_persisted_rows": True,
        "reason": (
            "exclude fp32_control from fixed-point admission and add exact measured "
            "pair-index count/hash custody"
        ),
        "authority_boundary": (
            "original producer SHA remains authoritative for numerical rows; current SHA "
            "owns summary logic only"
        ),
    }
    atomic_json(path, receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("one-thread CPU Torch contract failed")
    bits = tuple(sorted({int(value) for value in args.bits.split(",") if value}))
    allowed_bits = (
        (DYNAMIC_BITS, DYNAMIC_INT64_CEILING_BITS)
        if args.activation_scale_mode == "dynamic_exact_absmax"
        else (DEFAULT_BITS,)
    )
    if bits not in allowed_bits:
        raise ValueError(
            f"Task #494 precision ladder must be one of {allowed_bits}, got {bits}"
        )
    schema = (
        DYNAMIC_SCHEMA
        if args.activation_scale_mode == "dynamic_exact_absmax"
        else SCHEMA
    )
    arrays = _cache_arrays(args.gt_cache)
    seg, pose, seg_weights, pose_weights = _load_models(include_pose=args.include_pose)
    seg_calibration, pose_calibration, calibration_manifest = _calibrate(seg, pose, arrays)
    arms = _arm_specs(bits)
    contract = {
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "calibration_split": [CALIBRATION_START, CALIBRATION_STOP],
        "heldout_split": [HELDOUT_START, N600],
        "arms": arms,
        "include_pose": bool(args.include_pose),
        "threads": {"intraop": 1, "interop": 1},
        "accumulation": "QDQ emulation with fp32 Conv2d/Linear accumulation",
        "activation_scale_mode": args.activation_scale_mode,
        "dynamic_scale_order_invariance": (
            "max(abs(x)) is a commutative/idempotent reduction over finite activations"
            if args.activation_scale_mode == "dynamic_exact_absmax"
            else None
        ),
        "native_integer_speed_claim": False,
        "training_tolerance": TRAINING_TOLERANCE,
        "pose_cache_selfcheck_atol": POSE_CACHE_SELFCHECK_ATOL,
        "segnet_reference_control": (
            "recomputed one-thread CPU-Torch fp32 logits/argmax/top1-top2 margin on each real frame"
        ),
        "legacy_cache_role": "frame custody + explicit thread-geometry audit; not argmax authority",
    }
    custody = {
        "probe_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(
            REPO / "src/tac/local_acceleration/calibrated_fixedpoint_scorer.py"
        ),
        "gt_cache_sha256": sha256_file(args.gt_cache),
        "segnet_weights_sha256": sha256_file(seg_weights),
        "posenet_weights_sha256": sha256_file(pose_weights) if pose_weights else None,
        "calibration_digest": _hash_bytes(
            json.dumps(
                {key: value for key, value in calibration_manifest.items() if key != "elapsed_seconds"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
    }
    fingerprint = _contract_fingerprint({"contract": contract, "custody": custody})
    if args.resume and args.output.is_file():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if receipt.get("schema") != schema or receipt.get("fingerprint") != fingerprint:
            raise ValueError("resume receipt fingerprint differs from current contract/custody")
    else:
        receipt = {
            "schema": schema,
            "lane_id": "throughput_authority_ladder",
            "task_id": 494,
            "axis": "[macOS CPU-Torch local derivation; non-promotable MEANS]",
            "score_claim": False,
            "pointer_moved": False,
            "promotion_eligible": False,
            "git_head": _git_head(),
            "host": platform.node(),
            "contract": contract,
            "custody": custody,
            "fingerprint": fingerprint,
            "calibration": calibration_manifest,
            "cache_custody": {"segnet_rows": []},
            "arms": {
                "fp32_control": {"segnet_rows": [], "posenet_rows": []},
                **{
                    str(spec["name"]): {"segnet_rows": [], "posenet_rows": []}
                    for spec in arms
                },
            },
        }
        receipt["contract"]["arms"] = [
            {"name": "fp32_control", "bits": 32, "mixed_head_fp32": False}, *arms
        ]
        atomic_json(args.output, receipt)

    seg_candidates, pose_candidates, instrumentation = _build_candidates(
        seg=seg,
        pose=pose,
        seg_calibration=seg_calibration,
        pose_calibration=pose_calibration,
        bits=bits,
        activation_scale_mode=args.activation_scale_mode,
    )
    receipt["instrumentation"] = instrumentation
    completed = {
        int(row["pair_index"])
        for row in receipt["arms"]["fp32_control"]["segnet_rows"]
    }
    stop = int(args.pair_start) + int(args.pair_count)
    if args.pair_start < 0 or args.pair_count <= 0 or stop > N600:
        raise ValueError(f"invalid pair interval [{args.pair_start},{stop})")
    started = time.perf_counter()
    with torch.inference_mode():
        for pair_index in range(int(args.pair_start), stop):
            if pair_index in completed:
                continue
            btchw, _ = _pair_inputs(arrays, pair_index)
            seg_input = seg.preprocess_input(btchw)
            reference_logits = seg(seg_input)
            reference_argmax = reference_logits.argmax(dim=1)[0].cpu().numpy()
            top2 = reference_logits.topk(2, dim=1).values
            baseline_margin = (
                (top2[:, 0] - top2[:, 1]).clamp_min(0.0)[0].cpu().numpy()
            )
            cached_argmax = np.asarray(arrays["lstars.npy"][pair_index])
            cached_margin = np.asarray(arrays["margins.npy"][pair_index])
            receipt["cache_custody"]["segnet_rows"].append(
                {
                    "pair_index": pair_index,
                    "argmax_mismatch_pixels": int(
                        np.count_nonzero(reference_argmax != cached_argmax)
                    ),
                    "one_thread_argmax_sha256": _hash_array(reference_argmax),
                    "cached_argmax_sha256": _hash_array(cached_argmax),
                    "margin_max_abs_delta": float(
                        np.max(np.abs(baseline_margin - cached_margin))
                    ),
                    "one_thread_margin_min": float(np.min(baseline_margin)),
                    "cached_margin_min": float(np.min(cached_margin)),
                    "threads": 1,
                }
            )
            control = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=baseline_margin,
                candidate_logits=reference_logits,
            )
            receipt["arms"]["fp32_control"]["segnet_rows"].append(control)

            reference_pose = np.asarray(arrays["gt_poses.npy"][pair_index])
            pose_input = pose.preprocess_input(btchw) if pose is not None else None
            if pose is not None:
                recomputed_pose_output = pose(pose_input)
                recomputed_pose = (
                    recomputed_pose_output["pose"][0, :6].cpu().numpy().astype(np.float64)
                )
                baseline_pose_error = float(np.max(np.abs(recomputed_pose - reference_pose)))
                if baseline_pose_error > POSE_CACHE_SELFCHECK_ATOL:
                    raise RuntimeError(
                        f"CPU fp32 PoseNet failed cached authority at pair {pair_index}: "
                        f"max_abs={baseline_pose_error}"
                    )
                receipt["arms"]["fp32_control"]["posenet_rows"].append(
                    _pose_row(
                        pair_index=pair_index,
                        reference=reference_pose,
                        candidate=recomputed_pose_output,
                    )
                )

            for name, candidate in seg_candidates.items():
                receipt["arms"][name]["segnet_rows"].append(
                    _seg_row(
                        pair_index=pair_index,
                        reference_logits=reference_logits,
                        reference_argmax=reference_argmax,
                        baseline_margin=baseline_margin,
                        candidate_logits=candidate(seg_input),
                    )
                )
                if pose is not None:
                    receipt["arms"][name]["posenet_rows"].append(
                        _pose_row(
                            pair_index=pair_index,
                            reference=reference_pose,
                            candidate=pose_candidates[name](pose_input),
                        )
                    )
            completed.add(pair_index)
            if (len(completed) % args.checkpoint_every) == 0:
                receipt["summary"] = summarize(receipt)
                receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
                atomic_json(args.output, receipt)
    receipt["summary"] = summarize(receipt)
    receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
    receipt["completed"] = receipt["summary"]["status"] == "MEASURED"
    atomic_json(args.output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=N600)
    parser.add_argument("--bits", default=",".join(map(str, DEFAULT_BITS)))
    parser.add_argument(
        "--activation-scale-mode",
        choices=("fixed_calibration", "dynamic_exact_absmax"),
        default="fixed_calibration",
    )
    parser.add_argument("--include-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.finalize_only:
        payload = finalize_existing_receipt(args.output)
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        return 0
    if args.activation_scale_mode == "dynamic_exact_absmax":
        if args.bits == ",".join(map(str, DEFAULT_BITS)):
            args.bits = ",".join(map(str, DYNAMIC_BITS))
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_DYNAMIC_OUTPUT
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["status"] == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
