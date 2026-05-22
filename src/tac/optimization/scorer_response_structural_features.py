# SPDX-License-Identifier: MIT
"""Attach non-authoritative structural priors to scorer-response rows."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from tac.optimization.scorer_response_dataset import (
    ScorerResponseDatasetError,
    feature_correlations,
    normalize_legacy_response_dataset_authority,
    summarize_rows,
)

STRUCTURAL_FEATURE_SCHEMA = "scorer_response_structural_features.v1"
DIAGNOSTIC_SENSITIVITY_CYCLE = "diagnostic_sensitivity_cycle_v1"


def attach_structural_features(
    dataset: dict[str, Any],
    *,
    frame_axis_l1: Any | None = None,
    frame_axis_l1_source: str | None = None,
    frame_decomposition: dict[str, Any] | None = None,
    frame_decomposition_source: str | None = None,
    decoder_q_mutation_manifest: dict[str, Any] | None = None,
    decoder_q_mutation_manifest_source: str | None = None,
    decoder_q_family: str = "mlx_decoder_q",
) -> dict[str, Any]:
    """Attach pre-response structural features without changing authority.

    The frame-axis input is treated as a diagnostic cyclic prior unless the
    caller provides a full-window decomposition. The current canonical
    16-frame/8-pair master-gradient projection is useful for exposing the
    SegNet-last-frame and PoseNet-pair topology, but it is not exact 300-window
    score evidence.
    """

    normalized = normalize_legacy_response_dataset_authority(dataset)
    rows = copy.deepcopy(normalized["rows"])
    if not rows:
        raise ScorerResponseDatasetError("dataset rows must be non-empty")

    frame_axis = None
    frame_axis_meta: dict[str, Any] | None = None
    if frame_axis_l1 is not None:
        frame_axis, frame_axis_meta = _coerce_frame_axis_l1(
            frame_axis_l1,
            frame_decomposition=frame_decomposition,
            frame_axis_l1_source=frame_axis_l1_source,
            frame_decomposition_source=frame_decomposition_source,
        )

    decoder_q_meta = None
    if decoder_q_mutation_manifest is not None:
        decoder_q_meta = _decoder_q_metadata(
            decoder_q_mutation_manifest,
            source=decoder_q_mutation_manifest_source,
        )

    feature_write_counts: dict[str, int] = {}
    feature_nonzero_counts: dict[str, int] = {}
    for row in rows:
        if frame_axis is not None and frame_axis_meta is not None:
            for name in _attach_diagnostic_frame_axis_features(
                row,
                frame_axis=frame_axis,
                meta=frame_axis_meta,
            ):
                _count_feature(row, name, feature_write_counts, feature_nonzero_counts)
        if decoder_q_meta is not None:
            for name in _attach_decoder_q_features(
                row,
                decoder_q_meta=decoder_q_meta,
                decoder_q_family=decoder_q_family,
            ):
                _count_feature(row, name, feature_write_counts, feature_nonzero_counts)

    out = copy.deepcopy(normalized)
    out["rows"] = rows
    out["summary"] = summarize_rows(rows)
    out["feature_correlations"] = feature_correlations(rows)
    out["structural_features"] = {
        "schema": STRUCTURAL_FEATURE_SCHEMA,
        "producer": "tac.optimization.scorer_response_structural_features",
        "feature_write_counts": feature_write_counts,
        "feature_nonzero_counts": feature_nonzero_counts,
        "frame_axis_l1": frame_axis_meta,
        "decoder_q_mutation_manifest": decoder_q_meta,
        "decoder_q_family": decoder_q_family,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return out


def _count_feature(
    row: dict[str, Any],
    name: str,
    write_counts: dict[str, int],
    nonzero_counts: dict[str, int],
) -> None:
    write_counts[name] = write_counts.get(name, 0) + 1
    value = _finite_float(row.get(name))
    if value is not None and abs(value) > 0.0:
        nonzero_counts[name] = nonzero_counts.get(name, 0) + 1


def _coerce_frame_axis_l1(
    frame_axis_l1: Any,
    *,
    frame_decomposition: dict[str, Any] | None,
    frame_axis_l1_source: str | None,
    frame_decomposition_source: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(frame_axis_l1, dtype=np.float64)
    if arr.ndim != 2:
        raise ScorerResponseDatasetError("frame_axis_l1 must be a 2-D array")
    if arr.shape[0] < 2 or arr.shape[0] % 2 != 0:
        raise ScorerResponseDatasetError("frame_axis_l1 row count must be even and >= 2")
    if arr.shape[1] < 2:
        raise ScorerResponseDatasetError("frame_axis_l1 must include at least seg and pose axes")
    if not np.isfinite(arr).all():
        raise ScorerResponseDatasetError("frame_axis_l1 must contain only finite values")

    labels = _axis_labels(frame_decomposition, width=arr.shape[1])
    seg_index = _axis_index(labels, "seg", default=0)
    pose_index = _axis_index(labels, "pose", default=1)
    rate_index = _axis_index(labels, "rate", default=2 if arr.shape[1] > 2 else None)
    n_pairs = arr.shape[0] // 2
    source_schema = (
        frame_decomposition.get("schema")
        if isinstance(frame_decomposition, dict)
        else None
    )
    return arr, {
        "source_kind": DIAGNOSTIC_SENSITIVITY_CYCLE,
        "frame_axis_l1_source": frame_axis_l1_source,
        "frame_decomposition_source": frame_decomposition_source,
        "frame_decomposition_schema": source_schema,
        "axis_labels": labels,
        "seg_axis_index": seg_index,
        "pose_axis_index": pose_index,
        "rate_axis_index": rate_index,
        "n_frames": int(arr.shape[0]),
        "n_pairs": int(n_pairs),
        "topology": "non_overlapping_pair_cycle",
        "score_claim": False,
    }


def _axis_labels(frame_decomposition: dict[str, Any] | None, *, width: int) -> list[str]:
    if isinstance(frame_decomposition, dict):
        labels = frame_decomposition.get("axis_labels")
        if (
            isinstance(labels, list)
            and len(labels) >= width
            and all(isinstance(item, str) for item in labels[:width])
        ):
            return [str(item) for item in labels[:width]]
    defaults = ["seg", "pose", "rate"]
    return defaults[:width] + [f"axis_{index}" for index in range(len(defaults), width)]


def _axis_index(labels: list[str], name: str, *, default: int | None) -> int | None:
    try:
        return labels.index(name)
    except ValueError:
        return default


def _attach_diagnostic_frame_axis_features(
    row: dict[str, Any],
    *,
    frame_axis: np.ndarray,
    meta: dict[str, Any],
) -> list[str]:
    n_pairs = int(meta["n_pairs"])
    pair_start = _pair_start(row)
    cycle_pair = int(pair_start) % n_pairs
    first = 2 * cycle_pair
    last = first + 1
    seg_index = meta["seg_axis_index"]
    pose_index = meta["pose_axis_index"]
    rate_index = meta["rate_axis_index"]
    seg_last = _axis_value(frame_axis, last, seg_index)
    pose_pair = _axis_value(frame_axis, first, pose_index) + _axis_value(frame_axis, last, pose_index)
    rate_pair = _axis_value(frame_axis, first, rate_index) + _axis_value(frame_axis, last, rate_index)
    total = seg_last + pose_pair + rate_pair
    denom = total if abs(total) > 0.0 else 1.0
    values = {
        "diagnostic_cycle_pair_index": cycle_pair,
        "diagnostic_cycle_pair_index_norm": cycle_pair / max(n_pairs - 1, 1),
        "diagnostic_cycle_first_frame_index": first,
        "diagnostic_cycle_last_frame_index": last,
        "diagnostic_seg_last_l1": seg_last,
        "diagnostic_pose_pair_l1": pose_pair,
        "diagnostic_rate_pair_l1": rate_pair,
        "diagnostic_total_pair_l1": total,
        "diagnostic_seg_share": seg_last / denom,
        "diagnostic_pose_share": pose_pair / denom,
        "diagnostic_feature_source": meta["source_kind"],
    }
    row.update(values)
    return [
        "diagnostic_cycle_pair_index_norm",
        "diagnostic_seg_last_l1",
        "diagnostic_pose_pair_l1",
        "diagnostic_rate_pair_l1",
        "diagnostic_total_pair_l1",
        "diagnostic_seg_share",
        "diagnostic_pose_share",
    ]


def _axis_value(frame_axis: np.ndarray, row_index: int, axis_index: int | None) -> float:
    if axis_index is None:
        return 0.0
    return float(frame_axis[row_index, axis_index])


def _decoder_q_metadata(
    manifest: dict[str, Any],
    *,
    source: str | None,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ScorerResponseDatasetError("decoder_q mutation manifest must be a JSON object")
    mutation_row = manifest.get("mutation_row")
    if not isinstance(mutation_row, dict):
        raise ScorerResponseDatasetError("decoder_q mutation manifest mutation_row missing")
    mutation = mutation_row.get("mutation")
    if not isinstance(mutation, dict):
        raise ScorerResponseDatasetError("decoder_q mutation row mutation missing")
    target = mutation_row.get("op3v3_target_evidence")
    if not isinstance(target, dict):
        raise ScorerResponseDatasetError("decoder_q mutation row op3v3_target_evidence missing")
    approx = target.get("approx_compressed_range")
    if not isinstance(approx, dict):
        approx = {}
    axis_share = target.get("axis_share")
    if not isinstance(axis_share, dict):
        axis_share = {}
    archive_bytes = _finite_float(manifest.get("archive_bin_bytes")) or 1.0
    start = _finite_float(approx.get("start")) or 0.0
    length = _finite_float(approx.get("length")) or 0.0
    return {
        "source": source,
        "mutation_id": str(mutation_row.get("mutation_id") or ""),
        "tensor_name": str(mutation.get("tensor_name") or mutation_row.get("tensor_name") or ""),
        "score_claim": False,
        "decoder_q_delta": _finite_float(mutation.get("delta")) or 0.0,
        "decoder_q_q_offset": _finite_float(mutation.get("q_offset")) or 0.0,
        "decoder_q_score_impact_abs_sum": _finite_float(target.get("score_impact_abs_sum")) or 0.0,
        "decoder_q_axis_share_seg": _finite_float(axis_share.get("seg")) or 0.0,
        "decoder_q_axis_share_pose": _finite_float(axis_share.get("pose")) or 0.0,
        "decoder_q_axis_share_rate": _finite_float(axis_share.get("rate")) or 0.0,
        "decoder_q_top_byte_count": _finite_float(target.get("top_byte_count")) or 0.0,
        "decoder_q_approx_compressed_start_norm": start / archive_bytes,
        "decoder_q_approx_compressed_length_norm": length / archive_bytes,
    }


def _attach_decoder_q_features(
    row: dict[str, Any],
    *,
    decoder_q_meta: dict[str, Any],
    decoder_q_family: str,
) -> list[str]:
    is_decoder_q = str(row.get("family") or "") == decoder_q_family
    names = [
        "decoder_q_delta",
        "decoder_q_q_offset",
        "decoder_q_score_impact_abs_sum",
        "decoder_q_axis_share_seg",
        "decoder_q_axis_share_pose",
        "decoder_q_axis_share_rate",
        "decoder_q_top_byte_count",
        "decoder_q_approx_compressed_start_norm",
        "decoder_q_approx_compressed_length_norm",
    ]
    for name in names:
        row[name] = float(decoder_q_meta[name]) if is_decoder_q else 0.0
    row["decoder_q_feature_source"] = decoder_q_meta["source"] if is_decoder_q else None
    row["decoder_q_mutation_id"] = decoder_q_meta["mutation_id"] if is_decoder_q else None
    row["decoder_q_tensor_name"] = decoder_q_meta["tensor_name"] if is_decoder_q else None
    return names


def _pair_start(row: dict[str, Any]) -> float:
    pair_window = row.get("source_pair_window") or row.get("pair_indices")
    if isinstance(pair_window, list) and pair_window:
        value = _finite_float(pair_window[0])
        if value is not None:
            return value
    value = _finite_float(row.get("source_start_pair"))
    if value is not None:
        return value
    return 0.0


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "DIAGNOSTIC_SENSITIVITY_CYCLE",
    "STRUCTURAL_FEATURE_SCHEMA",
    "attach_structural_features",
]
