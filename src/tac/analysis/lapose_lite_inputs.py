"""Deterministic LA-POSE-lite inputs from contest pair metrics.

Lane W pair metrics are general scorer telemetry, not Lane-W-only state. This
module turns those CUDA per-pair metrics into compression-time motion/action
features that can route any downstream atom family: HNeRV residuals, pose
repairs, foveated corrections, mask grammars, or future learned sidechannels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from tac.analysis.lapose_motion_atoms import LaposeMotionAtomError
from tac.analysis.lapose_paper_contract import LAPOSE_PAPER_REFERENCE

SCHEMA_VERSION = 1


def inputs_from_pair_metric_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str | None = None,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Build latent-action and opportunity records from per-pair metric metadata."""

    if not source_path:
        raise LaposeMotionAtomError("source_path is required")
    if str(payload.get("device") or "").lower() != "cuda":
        raise LaposeMotionAtomError("pair metric payload must be CUDA-derived")
    n_pairs = int(payload.get("n_pairs", 0))
    pose = _float_array(payload.get("per_pair_pose_dist"), "per_pair_pose_dist")
    seg = _float_array(payload.get("per_pair_seg_dist"), "per_pair_seg_dist")
    contrib = _float_array(payload.get("per_pair_contrib"), "per_pair_contrib")
    if not (n_pairs == len(pose) == len(seg) == len(contrib)):
        raise LaposeMotionAtomError("pair metric arrays must match n_pairs")
    hardest = _int_list(payload.get("hardest_pair_indices"), "hardest_pair_indices")
    if not hardest:
        raise LaposeMotionAtomError("hardest_pair_indices must be nonempty")
    if max_pairs is not None:
        if max_pairs <= 0:
            raise LaposeMotionAtomError("max_pairs must be positive")
        hardest = hardest[:max_pairs]

    pose_stats = _stats(pose)
    seg_stats = _stats(seg)
    contrib_stats = _stats(contrib)
    pose_deltas = _local_delta_array(pose)
    seg_deltas = _local_delta_array(seg)
    contrib_deltas = _local_delta_array(contrib)
    pose_delta_stats = _stats(pose_deltas)
    seg_delta_stats = _stats(seg_deltas)
    contrib_delta_stats = _stats(contrib_deltas)
    source_sha = source_sha256 or _sha256_json(payload)
    payload_openpilot_priors = _str_list(payload.get("openpilot_priors") or [], "openpilot_priors")
    latent_actions = []
    pair_opportunities = []
    for hard_pair_rank, pair_index in enumerate(hardest):
        _check_pair_index(pair_index, n_pairs)
        latent_actions.append(
            {
                "pair_index": pair_index,
                "hard_pair_rank": hard_pair_rank,
                "latent_action": _latent_action(
                    pair_index,
                    n_pairs=n_pairs,
                    pose=pose,
                    seg=seg,
                    contrib=contrib,
                    pose_deltas=pose_deltas,
                    seg_deltas=seg_deltas,
                    contrib_deltas=contrib_deltas,
                    pose_stats=pose_stats,
                    seg_stats=seg_stats,
                    contrib_stats=contrib_stats,
                    pose_delta_stats=pose_delta_stats,
                    seg_delta_stats=seg_delta_stats,
                    contrib_delta_stats=contrib_delta_stats,
                ),
                "feature_contract": "lapose_lite_pair_metric_v1",
                "source_path": source_path,
                "source_sha256": source_sha,
            }
        )
        pair_opportunities.append(
            {
                "pair_index": pair_index,
                "hard_pair_rank": hard_pair_rank,
                "opportunity_mass": max(float(contrib[pair_index]), 0.0),
                "hard_pair_score": float(contrib[pair_index]),
                "hard_pair_support": [pair_index],
                "confidence": _confidence(float(contrib[pair_index]), contrib_stats),
                "class_support": [],
                "geometry_priors": ["scorer_pair_metric", "pair_metric_hardness"],
                "openpilot_priors": list(payload_openpilot_priors),
                "source_path": source_path,
                "source_sha256": source_sha,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.analysis.lapose_lite_inputs.inputs_from_pair_metric_payload",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_path": source_path,
        "source_sha256": source_sha,
        "source_device": payload.get("device"),
        "source_lane": payload.get("lane", ""),
        "evidence_grade": "empirical_cuda_pair_metric_telemetry",
        "paper_reference": LAPOSE_PAPER_REFERENCE,
        "n_pairs": n_pairs,
        "selected_pair_count": len(hardest),
        "feature_contract": {
            "name": "lapose_lite_pair_metric_v1",
            "description": (
                "10-D deterministic compression-time motion/action proxy from pair index, "
                "PoseNet/SegNet/contribution z-scores, local finite differences, and hard flag"
            ),
            "fields": [
                "time_centered",
                "sin_time",
                "cos_time",
                "pose_z",
                "seg_z",
                "contrib_z",
                "pose_local_delta_z",
                "seg_local_delta_z",
                "contrib_local_delta_z",
                "hard_pair_indicator",
            ],
        },
        "latent_actions": latent_actions,
        "pair_opportunities": pair_opportunities,
        "dispatch_blockers": [
            "planning_only_lapose_lite_inputs",
            "lane_w_pair_metrics_are_general_scorer_telemetry_not_lane_local_state",
            "pair_metrics_are_not_score_authority",
            "lapose_lite_is_not_paper_faithful_lapose_model",
            "requires_component_response_allocation",
            "requires_charged_archive_builder",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _latent_action(
    pair_index: int,
    *,
    n_pairs: int,
    pose: np.ndarray,
    seg: np.ndarray,
    contrib: np.ndarray,
    pose_deltas: np.ndarray,
    seg_deltas: np.ndarray,
    contrib_deltas: np.ndarray,
    pose_stats: Mapping[str, float],
    seg_stats: Mapping[str, float],
    contrib_stats: Mapping[str, float],
    pose_delta_stats: Mapping[str, float],
    seg_delta_stats: Mapping[str, float],
    contrib_delta_stats: Mapping[str, float],
) -> list[float]:
    time = pair_index / max(n_pairs - 1, 1)
    theta = 2.0 * math.pi * time
    return [
        round(2.0 * time - 1.0, 12),
        round(math.sin(theta), 12),
        round(math.cos(theta), 12),
        round(_z(float(pose[pair_index]), pose_stats), 12),
        round(_z(float(seg[pair_index]), seg_stats), 12),
        round(_z(float(contrib[pair_index]), contrib_stats), 12),
        round(_z(float(pose_deltas[pair_index]), pose_delta_stats), 12),
        round(_z(float(seg_deltas[pair_index]), seg_delta_stats), 12),
        round(_z(float(contrib_deltas[pair_index]), contrib_delta_stats), 12),
        1.0,
    ]


def _local_delta_array(values: np.ndarray) -> np.ndarray:
    left = np.empty_like(values)
    right = np.empty_like(values)
    left[0] = values[0]
    left[1:] = values[:-1]
    right[-1] = values[-1]
    right[:-1] = values[1:]
    return values - 0.5 * (left + right)


def _stats(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {"mean": mean, "std": std if std > 0 else 1.0}


def _z(value: float, stats: Mapping[str, float]) -> float:
    return (value - float(stats["mean"])) / float(stats["std"])


def _confidence(value: float, stats: Mapping[str, float]) -> float:
    z = max(_z(value, stats), 0.0)
    return round(0.35 + 0.45 * (1.0 - math.exp(-z / 2.0)), 12)


def _float_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, list):
        raise LaposeMotionAtomError(f"{name} must be a list")
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or out.size == 0 or not np.all(np.isfinite(out)):
        raise LaposeMotionAtomError(f"{name} must be nonempty and finite")
    return out


def _int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise LaposeMotionAtomError(f"{name} must be a list")
    out = []
    for item in value:
        if not isinstance(item, int):
            raise LaposeMotionAtomError(f"{name} must contain integers")
        out.append(item)
    return out


def _str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError(f"{name} must be a list")
    return [str(item) for item in value]


def _check_pair_index(pair_index: int, n_pairs: int) -> None:
    if pair_index < 0 or pair_index >= n_pairs:
        raise LaposeMotionAtomError(f"pair_index {pair_index} outside 0..{n_pairs - 1}")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


__all__ = ["SCHEMA_VERSION", "inputs_from_pair_metric_payload"]
