# SPDX-License-Identifier: MIT
"""Distortion-crux extraction for NeRV scorer-input caches.

The upstream scorer is asymmetric: SegNet consumes only the last frame of each
pair, while PoseNet consumes both frames after RGB->YUV6 preprocessing.  This
helper reads existing scorer-input cache arrays and emits a compact,
false-authority artifact that long-run planners can use for hard-pair
curriculum routing without turning a local cache probe into score authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tac.repo_io import write_json
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

NERV_DISTORTION_CRUX_SCHEMA = "nerv_scorer_input_distortion_crux.v1"
NERV_DISTORTION_CRUX_HARD_PAIR_SCHEMA = "nerv_hard_pair_coverage_evidence.v1"
DEFAULT_DISTORTION_CRUX_TOP_K = 16
DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS = 16


def build_nerv_distortion_crux_report(
    *,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    sample_pairs: int = DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS,
    top_k: int = DEFAULT_DISTORTION_CRUX_TOP_K,
    min_routable_pairs: int = DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS,
    max_segnet_last_frame_mae_for_fit_gate: float = 64.0,
    max_posenet_yuv6_pair_mae_for_fit_gate: float = 64.0,
    max_posenet_temporal_delta_mae_for_fit_gate: float = 64.0,
) -> dict[str, Any]:
    """Build a PR95/evaluate.py-shaped local distortion-crux payload.

    The report intentionally works on scorer-input caches rather than rendered
    full-resolution frames.  That keeps the comparison aligned with the exact
    upstream preprocessing contract while staying portable across NumPy, MLX,
    and Torch callers.
    """

    if int(sample_pairs) < 1:
        raise ValueError(f"sample_pairs must be >= 1, got {sample_pairs}")
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if int(min_routable_pairs) < 1:
        raise ValueError(f"min_routable_pairs must be >= 1, got {min_routable_pairs}")
    seg_threshold = _positive_threshold(
        max_segnet_last_frame_mae_for_fit_gate,
        "max_segnet_last_frame_mae_for_fit_gate",
    )
    pose_threshold = _positive_threshold(
        max_posenet_yuv6_pair_mae_for_fit_gate,
        "max_posenet_yuv6_pair_mae_for_fit_gate",
    )
    temporal_threshold = _positive_threshold(
        max_posenet_temporal_delta_mae_for_fit_gate,
        "max_posenet_temporal_delta_mae_for_fit_gate",
    )

    candidate = Path(candidate_cache_dir).expanduser().resolve(strict=False)
    reference = Path(reference_cache_dir).expanduser().resolve(strict=False)
    cand_seg = _load_cache_array(candidate, "segnet_last_rgb.npy")
    ref_seg = _load_cache_array(reference, "segnet_last_rgb.npy")
    cand_pose = _load_cache_array(candidate, "posenet_yuv6_pair.npy")
    ref_pose = _load_cache_array(reference, "posenet_yuv6_pair.npy")
    n = _sample_count(cand_seg, ref_seg, cand_pose, ref_pose, sample_pairs)
    cand_seg = np.asarray(cand_seg[:n], dtype=np.float32)
    ref_seg = np.asarray(ref_seg[:n], dtype=np.float32)
    cand_pose = np.asarray(cand_pose[:n], dtype=np.float32)
    ref_pose = np.asarray(ref_pose[:n], dtype=np.float32)
    _validate_segnet_cache("candidate_segnet_last_rgb", cand_seg)
    _validate_segnet_cache("reference_segnet_last_rgb", ref_seg)
    _validate_pose_cache("candidate_posenet_yuv6_pair", cand_pose)
    _validate_pose_cache("reference_posenet_yuv6_pair", ref_pose)
    if cand_seg.shape != ref_seg.shape:
        raise ValueError(
            "candidate/reference SegNet cache shape mismatch: "
            f"{cand_seg.shape} vs {ref_seg.shape}"
        )
    if cand_pose.shape != ref_pose.shape:
        raise ValueError(
            "candidate/reference PoseNet cache shape mismatch: "
            f"{cand_pose.shape} vs {ref_pose.shape}"
        )

    pair_indices = _pair_ids(candidate, reference, n)
    pose_c = cand_pose.reshape(n, 2, 6, *cand_pose.shape[-2:])
    pose_r = ref_pose.reshape(n, 2, 6, *ref_pose.shape[-2:])
    seg_abs = np.abs(cand_seg - ref_seg)
    pose_abs = np.abs(pose_c - pose_r)
    temporal_abs = np.abs(
        (pose_c[:, 1, ...] - pose_c[:, 0, ...])
        - (pose_r[:, 1, ...] - pose_r[:, 0, ...])
    )

    seg_pair_mae = _mean_per_pair(seg_abs)
    pose_pair_mae = _mean_per_pair(pose_abs.reshape(n, 12, *cand_pose.shape[-2:]))
    pose_frame0_mae = _mean_per_pair(pose_abs[:, 0, ...])
    pose_frame1_mae = _mean_per_pair(pose_abs[:, 1, ...])
    temporal_delta_mae = _mean_per_pair(temporal_abs)
    rows = _hard_pair_rows(
        pair_indices=pair_indices,
        seg_pair_mae=seg_pair_mae,
        pose_pair_mae=pose_pair_mae,
        pose_frame0_mae=pose_frame0_mae,
        pose_frame1_mae=pose_frame1_mae,
        temporal_delta_mae=temporal_delta_mae,
        seg_threshold=seg_threshold,
        pose_threshold=pose_threshold,
        temporal_threshold=temporal_threshold,
    )
    top_rows = rows[: int(top_k)]
    prioritized = _dedupe_ints(row["pair_index"] for row in top_rows)
    coverage_valid = bool(n >= int(min_routable_pairs))
    nonauthority_blockers: list[str] = []
    if float(np.mean(seg_pair_mae)) > seg_threshold:
        nonauthority_blockers.append(
            "nerv_distortion_crux_segnet_last_frame_mae_too_high"
        )
    if float(np.mean(pose_pair_mae)) > pose_threshold:
        nonauthority_blockers.append(
            "nerv_distortion_crux_posenet_yuv6_pair_mae_too_high"
        )
    if float(np.mean(temporal_delta_mae)) > temporal_threshold:
        nonauthority_blockers.append(
            "nerv_distortion_crux_posenet_temporal_delta_mae_too_high"
        )

    routing_blockers = []
    if not coverage_valid:
        routing_blockers.append("nerv_distortion_crux_min16_pairs_missing_for_launch_routing")
    if not prioritized:
        routing_blockers.append("nerv_distortion_crux_prioritized_pair_indices_empty")

    domain_counts: dict[str, int] = {}
    for row in top_rows:
        domain = str(row["dominant_domain"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    dominant_domain = (
        sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if domain_counts
        else None
    )
    hard_pair_coverage = {
        "schema": NERV_DISTORTION_CRUX_HARD_PAIR_SCHEMA,
        "source_schema": NERV_DISTORTION_CRUX_SCHEMA,
        "sample_pairs": int(n),
        "min_routable_pairs": int(min_routable_pairs),
        "hard_pair_count": len(prioritized),
        "prioritized_pair_indices": prioritized,
        "representative_distortion_evidence": coverage_valid,
        "score_axis_hard_pair_coverage": coverage_valid,
        "coverage_valid_for_distortion": coverage_valid,
        "coverage_verdict": (
            "score_axis_hard_pair_coverage_routable"
            if coverage_valid
            else "score_axis_hard_pair_hitlist_not_routable_until_min16_pairs"
        ),
        "dominant_domain": dominant_domain,
        "domain_counts_top_k": domain_counts,
        "blockers": routing_blockers,
        **FALSE_AUTHORITY,
    }

    blockers = ["nerv_distortion_crux_is_false_authority", *nonauthority_blockers]
    return {
        "schema": NERV_DISTORTION_CRUX_SCHEMA,
        "candidate_cache_dir": candidate.as_posix(),
        "reference_cache_dir": reference.as_posix(),
        "sample_pairs": int(n),
        "top_k": int(top_k),
        "min_routable_pairs": int(min_routable_pairs),
        "scorer_contract": {
            "source": "verified_local_upstream_evaluate_py_modules_py",
            "segnet_domain": "last_frame_of_pair_only_after_scorer_resize",
            "posenet_domain": "both_frames_after_rgb_to_yuv6",
            "local_evidence_axis": "scorer_input_cache_false_authority",
            **FALSE_AUTHORITY,
        },
        "thresholds": {
            "max_segnet_last_frame_mae_for_fit_gate": seg_threshold,
            "max_posenet_yuv6_pair_mae_for_fit_gate": pose_threshold,
            "max_posenet_temporal_delta_mae_for_fit_gate": temporal_threshold,
        },
        "aggregate": {
            "segnet_last_frame_mae_255": _stats(seg_pair_mae),
            "posenet_yuv6_pair_mae_255": _stats(pose_pair_mae),
            "posenet_frame0_yuv6_mae_255": _stats(pose_frame0_mae),
            "posenet_frame1_yuv6_mae_255": _stats(pose_frame1_mae),
            "posenet_temporal_delta_mae_255": _stats(temporal_delta_mae),
            "dominant_domain_top_k": dominant_domain,
            "domain_counts_top_k": domain_counts,
        },
        "hard_pair_rows": top_rows,
        "hard_pair_coverage": hard_pair_coverage,
        "fit_gate_passed": not nonauthority_blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def write_nerv_distortion_crux_report(
    *,
    output_json: str | Path,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    sample_pairs: int = DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS,
    top_k: int = DEFAULT_DISTORTION_CRUX_TOP_K,
    min_routable_pairs: int = DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS,
    max_segnet_last_frame_mae_for_fit_gate: float = 64.0,
    max_posenet_yuv6_pair_mae_for_fit_gate: float = 64.0,
    max_posenet_temporal_delta_mae_for_fit_gate: float = 64.0,
) -> dict[str, Any]:
    """Write a distortion-crux report and return its payload."""

    report = build_nerv_distortion_crux_report(
        candidate_cache_dir=candidate_cache_dir,
        reference_cache_dir=reference_cache_dir,
        sample_pairs=sample_pairs,
        top_k=top_k,
        min_routable_pairs=min_routable_pairs,
        max_segnet_last_frame_mae_for_fit_gate=(
            max_segnet_last_frame_mae_for_fit_gate
        ),
        max_posenet_yuv6_pair_mae_for_fit_gate=(
            max_posenet_yuv6_pair_mae_for_fit_gate
        ),
        max_posenet_temporal_delta_mae_for_fit_gate=(
            max_posenet_temporal_delta_mae_for_fit_gate
        ),
    )
    out = Path(output_json).expanduser().resolve(strict=False)
    report["report_path"] = out.as_posix()
    write_json(out, report)
    return report


def _load_cache_array(root: Path, filename: str) -> np.ndarray:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"scorer-input cache array missing: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim < 2:
        raise ValueError(f"invalid scorer-input cache rank for {path}: {arr.shape}")
    return arr


def _sample_count(*arrays_and_pairs: Any) -> int:
    *arrays, sample_pairs = arrays_and_pairs
    n = min(int(arr.shape[0]) for arr in arrays)
    n = min(n, int(sample_pairs))
    if n < 1:
        raise ValueError("scorer-input caches have no sample rows")
    return n


def _validate_segnet_cache(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 4 or int(arr.shape[1]) != 3:
        raise ValueError(f"{name} must have shape (B, 3, H, W), got {arr.shape}")
    if int(arr.shape[2]) < 1 or int(arr.shape[3]) < 1:
        raise ValueError(f"{name} has invalid spatial shape {arr.shape}")


def _validate_pose_cache(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 4 or int(arr.shape[1]) != 12:
        raise ValueError(f"{name} must have shape (B, 12, H, W), got {arr.shape}")
    if int(arr.shape[2]) < 1 or int(arr.shape[3]) < 1:
        raise ValueError(f"{name} has invalid spatial shape {arr.shape}")


def _pair_ids(candidate: Path, reference: Path, n: int) -> np.ndarray:
    for root in (candidate, reference):
        path = root / "pair_indices.npy"
        if not path.is_file():
            continue
        arr = np.asarray(np.load(path, mmap_mode="r")[:n])
        if arr.shape == (n, 2):
            first = arr[:, 0].astype(np.int64)
            second = arr[:, 1].astype(np.int64)
            if np.all(second == first + 1) and np.all(first % 2 == 0):
                return first // 2
    return np.arange(n, dtype=np.int64)


def _mean_per_pair(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr.reshape(arr.shape[0], -1).mean(axis=1), dtype=np.float64)


def _hard_pair_rows(
    *,
    pair_indices: np.ndarray,
    seg_pair_mae: np.ndarray,
    pose_pair_mae: np.ndarray,
    pose_frame0_mae: np.ndarray,
    pose_frame1_mae: np.ndarray,
    temporal_delta_mae: np.ndarray,
    seg_threshold: float,
    pose_threshold: float,
    temporal_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, pair_index in enumerate(pair_indices):
        ratios = {
            "segnet_last_frame": float(seg_pair_mae[i] / seg_threshold),
            "posenet_yuv6_pair": float(pose_pair_mae[i] / pose_threshold),
            "posenet_temporal_delta": float(
                temporal_delta_mae[i] / temporal_threshold
            ),
        }
        dominant = sorted(ratios.items(), key=lambda item: (-item[1], item[0]))[0]
        rows.append(
            {
                "rank": 0,
                "pair_index": int(pair_index),
                "dominant_domain": dominant[0],
                "crux_score": dominant[1],
                "segnet_last_frame_mae_255": float(seg_pair_mae[i]),
                "posenet_yuv6_pair_mae_255": float(pose_pair_mae[i]),
                "posenet_frame0_yuv6_mae_255": float(pose_frame0_mae[i]),
                "posenet_frame1_yuv6_mae_255": float(pose_frame1_mae[i]),
                "posenet_temporal_delta_mae_255": float(temporal_delta_mae[i]),
                "normalized_domain_scores": ratios,
            }
        )
    rows.sort(key=lambda row: (-float(row["crux_score"]), int(row["pair_index"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _dedupe_ints(values: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        value = int(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _positive_threshold(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return out


__all__ = [
    "DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS",
    "DEFAULT_DISTORTION_CRUX_TOP_K",
    "NERV_DISTORTION_CRUX_HARD_PAIR_SCHEMA",
    "NERV_DISTORTION_CRUX_SCHEMA",
    "build_nerv_distortion_crux_report",
    "write_nerv_distortion_crux_report",
]
