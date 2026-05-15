# SPDX-License-Identifier: MIT
"""Build sparse exact-CUDA probe plans for HDM8 postdecode selectors.

This module consumes a CUDA-prefix postdecode sweep and emits deterministic
selector configs that can be packed into byte-closed HDM8 archives. It is a
planning bridge only: no score claim is made until each emitted packet is run
through exact contest-CUDA auth eval.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

CONFIG_SCHEMA = "hdm8_film_grain_sidecar_postfilter_config_v1"
PLAN_SCHEMA = "hdm8_cuda_selector_probe_plan_v1"
RATE_DENOMINATOR_BYTES = 37_545_489
DEFAULT_PREFIX_SIZES = (1, 2, 4, 8, 16, 32)


class HDM8CudaSelectorProbePlanError(ValueError):
    """Raised when a CUDA selector probe plan cannot be built."""


def _score(avg_pose: float, avg_seg: float, archive_bytes: int) -> float:
    return (
        100.0 * float(avg_seg)
        + math.sqrt(10.0 * max(0.0, float(avg_pose)))
        + 25.0 * int(archive_bytes) / RATE_DENOMINATOR_BYTES
    )


def _modes(sweep: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    modes = sweep.get("modes")
    if not isinstance(modes, list) or not modes:
        raise HDM8CudaSelectorProbePlanError("sweep must contain non-empty modes list")
    return [mode for mode in modes if isinstance(mode, Mapping)]


def _baseline(modes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in modes:
        if row.get("mode") == "none":
            return row
    raise HDM8CudaSelectorProbePlanError("sweep modes must include mode='none'")


def _pair_array(row: Mapping[str, Any], key: str, n_pairs: int) -> list[float]:
    values = row.get(key)
    if not isinstance(values, list):
        raise HDM8CudaSelectorProbePlanError(f"mode {row.get('mode')!r} missing {key}")
    if len(values) != n_pairs:
        raise HDM8CudaSelectorProbePlanError(
            f"mode {row.get('mode')!r} {key} length {len(values)} != {n_pairs}"
        )
    return [float(value) for value in values]


def _compact_json_bytes(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _normalize_prefix_sizes(prefix_sizes: Sequence[int], max_atoms: int) -> list[int]:
    normalized = sorted({int(size) for size in prefix_sizes if int(size) > 0})
    return [size for size in normalized if size <= max_atoms]


def _axis_is_cuda_prefix(axis: object) -> bool:
    text = str(axis or "").lower().replace("_", "-")
    return "cuda" in text and "proxy" in text and "mps" not in text and "macos" not in text


def _require_cuda_prefix_axis(axis: object) -> None:
    if not _axis_is_cuda_prefix(axis):
        raise HDM8CudaSelectorProbePlanError(
            "HDM8 selector probe planning requires a CUDA-prefix proxy axis; "
            f"got {axis!r}"
        )


def build_hdm8_cuda_selector_probe_plan(
    sweep: Mapping[str, Any],
    *,
    evidence_source_path: str = "",
    max_atoms: int = 64,
    prefix_sizes: Sequence[int] = DEFAULT_PREFIX_SIZES,
    min_pair_gain: float = 0.0,
) -> dict[str, Any]:
    """Return sparse selector configs ranked from a CUDA-prefix sweep.

    The planner picks the best non-``none`` mode for each pair using the local
    linearized official score around the sweep baseline, then emits prefix
    policies over the highest-gain unique pairs. Each config leaves all other
    pairs at ``none`` so exact-CUDA probes can measure whether the proxy gain is
    additive, PoseNet-safe, and worth the charged selector bytes.
    """

    modes = _modes(sweep)
    baseline = _baseline(modes)
    n_pairs = int(sweep.get("n_pairs") or baseline.get("n_pairs") or 0)
    if n_pairs <= 0:
        raise HDM8CudaSelectorProbePlanError("sweep must carry positive n_pairs")
    archive_bytes = int(sweep.get("archive_bytes") or 0)
    if archive_bytes <= 0:
        raise HDM8CudaSelectorProbePlanError("sweep must carry positive archive_bytes")
    axis = sweep.get("axis")
    _require_cuda_prefix_axis(axis)

    baseline_pose = _pair_array(baseline, "pair_posenet_dist", n_pairs)
    baseline_seg = _pair_array(baseline, "pair_segnet_dist", n_pairs)
    base_avg_pose = float(baseline.get("avg_posenet_dist"))
    base_avg_seg = float(baseline.get("avg_segnet_dist"))
    baseline_score = _score(base_avg_pose, base_avg_seg, archive_bytes)
    pose_weight = math.sqrt(10.0) / (2.0 * math.sqrt(max(base_avg_pose, 1e-12)))

    best_by_pair: dict[int, dict[str, Any]] = {}
    for row in modes:
        mode = str(row.get("mode"))
        if mode == "none":
            continue
        pose_values = _pair_array(row, "pair_posenet_dist", n_pairs)
        seg_values = _pair_array(row, "pair_segnet_dist", n_pairs)
        for pair_index, (pose, seg) in enumerate(zip(pose_values, seg_values, strict=True)):
            baseline_local = (
                100.0 * baseline_seg[pair_index] + pose_weight * baseline_pose[pair_index]
            )
            candidate_local = 100.0 * seg + pose_weight * pose
            gain = baseline_local - candidate_local
            if gain <= float(min_pair_gain):
                continue
            current = best_by_pair.get(pair_index)
            if current is None or gain > float(current["local_linearized_gain"]):
                best_by_pair[pair_index] = {
                    "pair_index": pair_index,
                    "mode": mode,
                    "local_linearized_gain": gain,
                    "baseline_pair_posenet_dist": baseline_pose[pair_index],
                    "candidate_pair_posenet_dist": pose,
                    "baseline_pair_segnet_dist": baseline_seg[pair_index],
                    "candidate_pair_segnet_dist": seg,
                    "pair_posenet_delta": pose - baseline_pose[pair_index],
                    "pair_segnet_delta": seg - baseline_seg[pair_index],
                }

    ranked_atoms = sorted(
        best_by_pair.values(),
        key=lambda atom: (
            -float(atom["local_linearized_gain"]),
            int(atom["pair_index"]),
            str(atom["mode"]),
        ),
    )[: int(max_atoms)]
    prefixes = _normalize_prefix_sizes(prefix_sizes, len(ranked_atoms))
    configs: list[dict[str, Any]] = []
    for prefix_size in prefixes:
        selected_atoms = ranked_atoms[:prefix_size]
        palette = ["none"]
        palette_index = {"none": 0}
        selector_indices = [0] * n_pairs
        pose_values = list(baseline_pose)
        seg_values = list(baseline_seg)
        for atom in selected_atoms:
            mode = str(atom["mode"])
            if mode not in palette_index:
                palette_index[mode] = len(palette)
                palette.append(mode)
            pair_index = int(atom["pair_index"])
            selector_indices[pair_index] = palette_index[mode]
            pose_values[pair_index] = float(atom["candidate_pair_posenet_dist"])
            seg_values[pair_index] = float(atom["candidate_pair_segnet_dist"])

        avg_pose = sum(pose_values) / n_pairs
        avg_seg = sum(seg_values) / n_pairs
        proxy_score = _score(avg_pose, avg_seg, archive_bytes)
        delta = proxy_score - baseline_score
        mode_counts = Counter(palette[index] for index in selector_indices)
        config: dict[str, Any] = {
            "schema": CONFIG_SCHEMA,
            "mode": "selector",
            "palette": palette,
            "selector_indices": selector_indices,
            "score_claim": False,
            "evidence_axis": sweep.get("axis"),
            "cuda_probe_plan": {
                "schema": PLAN_SCHEMA,
                "evidence_source_path": evidence_source_path,
                "prefix_size": prefix_size,
                "selected_pair_count": prefix_size,
                "ranked_atom_count": len(ranked_atoms),
                "selected_atoms": selected_atoms,
            },
        }
        config["proxy"] = {
            "present": True,
            "path": evidence_source_path,
            "axis": sweep.get("axis"),
            "n_pairs": n_pairs,
            "mode": "selector",
            "baseline_score_proxy": baseline_score,
            "mode_score_proxy": proxy_score,
            "delta_vs_none": delta,
            "avg_posenet_dist": avg_pose,
            "avg_segnet_dist": avg_seg,
            "baseline_avg_posenet_dist": base_avg_pose,
            "baseline_avg_segnet_dist": base_avg_seg,
            "positive": delta < 0.0,
            "selected_mode_counts": dict(sorted(mode_counts.items())),
            "selected_pair_count": prefix_size,
            "selector_config_bytes_if_charged": _compact_json_bytes(config),
            "compliance_risk": None,
        }
        configs.append(
            {
                "name": f"sparse_cuda_prefix_top{prefix_size:03d}",
                "prefix_size": prefix_size,
                "config": config,
                "proxy_delta_vs_none": delta,
                "proxy_score": proxy_score,
                "selected_pair_count": prefix_size,
                "selected_mode_counts": dict(sorted(mode_counts.items())),
                "selector_config_json_bytes": _compact_json_bytes(config),
                "score_claim": False,
                "promotion_eligible": False,
            }
        )

    return {
        "schema": PLAN_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_source_path": evidence_source_path,
        "axis": axis,
        "axis_rankable_for_probe_planning": _axis_is_cuda_prefix(axis),
        "archive_bytes": archive_bytes,
        "archive_sha256": sweep.get("archive_sha256"),
        "n_pairs": n_pairs,
        "mode_count": len(modes),
        "max_atoms": int(max_atoms),
        "min_pair_gain": float(min_pair_gain),
        "candidate_atom_count": len(ranked_atoms),
        "top_atoms": ranked_atoms,
        "probe_configs": configs,
        "dispatch_blockers": [
            "byte_closed_packets_must_be_built_from_selector_configs",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing_for_each_probe",
            *([] if _axis_is_cuda_prefix(axis) else ["source_axis_not_cuda_prefix"]),
        ],
    }


__all__ = [
    "DEFAULT_PREFIX_SIZES",
    "PLAN_SCHEMA",
    "HDM8CudaSelectorProbePlanError",
    "build_hdm8_cuda_selector_probe_plan",
]
