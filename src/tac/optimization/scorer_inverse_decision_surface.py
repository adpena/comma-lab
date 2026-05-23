# SPDX-License-Identifier: MIT
"""Planning-only inverse scorer decision surface.

The surface treats scorer-response observations as samples from the receiver's
inverse decision surface: what SegNet/PoseNet/rate can infer for free, what is
a sufficient statistic, and where the fragile boundaries are. It does not
claim score authority; it converts rows into compressed-coordinate planning
units for byte-shaving acquisition.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.scorer_response_dataset import (
    RATE_SCORE_PER_BYTE,
    ScorerResponseDatasetError,
    normalize_legacy_response_dataset_authority,
    scorer_response_planning_value_for_target,
)

SCHEMA = "scorer_inverse_decision_surface.v1"
UNIT_KIND = "scorer_inverse_surface_cell"
OPERATION_PROBE = "probe_inverse_scorer_surface_cell"
OPERATION_MATERIALIZE = "materialize_inverse_scorer_cell_candidate"
DEFAULT_NULL_SCORER_DELTA_EPSILON = 1e-6
DEFAULT_FRAGILE_SCORER_DELTA_THRESHOLD = 0.0


def build_inverse_scorer_decision_surface(
    dataset: Mapping[str, Any],
    *,
    source_label: str = "",
    max_units: int = 16,
    null_scorer_delta_epsilon: float = DEFAULT_NULL_SCORER_DELTA_EPSILON,
    fragile_scorer_delta_threshold: float = DEFAULT_FRAGILE_SCORER_DELTA_THRESHOLD,
    allow_native_mlx_window_objective: bool = False,
) -> dict[str, Any]:
    """Build a false-authority inverse-scorer planning surface."""

    if max_units < 1:
        raise ScorerResponseDatasetError("max_units must be >= 1")
    if null_scorer_delta_epsilon < 0.0 or not math.isfinite(null_scorer_delta_epsilon):
        raise ScorerResponseDatasetError("null_scorer_delta_epsilon must be finite and >= 0")
    if not math.isfinite(fragile_scorer_delta_threshold):
        raise ScorerResponseDatasetError("fragile_scorer_delta_threshold must be finite")

    normalized = normalize_legacy_response_dataset_authority(
        dict(dataset),
        source_label=source_label,
    )
    try:
        require_no_truthy_authority_fields(
            normalized,
            context="scorer_inverse_decision_surface",
        )
    except ValueError as exc:
        raise ScorerResponseDatasetError(str(exc)) from exc

    source_rows = normalized.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ScorerResponseDatasetError("dataset rows must be a non-empty list")

    samples = [
        _sample_from_row(
            row,
            row_index=index,
            source_label=source_label,
            null_scorer_delta_epsilon=null_scorer_delta_epsilon,
            fragile_scorer_delta_threshold=fragile_scorer_delta_threshold,
            allow_native_mlx_window_objective=allow_native_mlx_window_objective,
        )
        for index, row in enumerate(source_rows)
        if isinstance(row, Mapping)
    ]
    if len(samples) != len(source_rows):
        raise ScorerResponseDatasetError("dataset rows must be JSON objects")
    cells = _aggregate_cells(samples)
    ranked_cells = sorted(
        cells.values(),
        key=lambda cell: (
            float(cell["median_projected_delta_vs_baseline_score"]),
            -int(cell["candidate_saved_bytes"]),
            -int(cell["row_count"]),
            str(cell["cell_id"]),
        ),
    )[:max_units]
    units = [_unit_from_cell(cell) for cell in ranked_cells]
    return {
        "schema": SCHEMA,
        "source_label": source_label,
        "source_schema": normalized.get("schema") or normalized.get("schema_version"),
        "source_row_count": len(source_rows),
        "cell_count": len(cells),
        "emitted_unit_count": len(units),
        "null_scorer_delta_epsilon": null_scorer_delta_epsilon,
        "fragile_scorer_delta_threshold": fragile_scorer_delta_threshold,
        "allow_native_mlx_window_objective": allow_native_mlx_window_objective,
        "decision_surface_classes": sorted(
            {str(cell["decision_surface_class"]) for cell in cells.values()}
        ),
        "cells": ranked_cells,
        "units": units,
        "blockers": [
            "inverse_scorer_surface_is_planning_only",
            *(
                ["native_mlx_window_objective_not_full_video_normalized"]
                if allow_native_mlx_window_objective
                else []
            ),
            "requires_byte_closed_materializer_before_candidate_archive",
            "requires_same_runtime_locality_or_inflate_parity_check",
            "requires_exact_auth_eval_before_score_claim",
        ],
        **FALSE_AUTHORITY,
    }


def _sample_from_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source_label: str,
    null_scorer_delta_epsilon: float,
    fragile_scorer_delta_threshold: float,
    allow_native_mlx_window_objective: bool,
) -> dict[str, Any]:
    row_payload = dict(row)
    label = str(row_payload.get("row_id") or row_payload.get("candidate_id") or row_index)
    projected_delta = _planning_value_for_target(
        row_payload,
        "delta_vs_baseline_score",
        label=label,
        allow_native_mlx_window_objective=allow_native_mlx_window_objective,
    )
    scorer_delta = _planning_value_for_target(
        row_payload,
        "scorer_delta_vs_baseline",
        label=label,
        allow_native_mlx_window_objective=allow_native_mlx_window_objective,
    )
    scorer_gain = _planning_value_for_target(
        row_payload,
        "observed_scorer_gain_vs_baseline",
        label=label,
        allow_native_mlx_window_objective=allow_native_mlx_window_objective,
    )
    margin = _planning_value_for_target(
        row_payload,
        "byte_budget_margin_vs_break_even",
        label=label,
        allow_native_mlx_window_objective=allow_native_mlx_window_objective,
    )
    if projected_delta is None:
        raise ScorerResponseDatasetError(f"{label}: missing projected full-video delta")
    added_archive_bytes = _finite_float(row_payload.get("added_archive_bytes")) or 0.0
    candidate_saved_bytes = max(0, round(-added_archive_bytes))
    scorer_delta_value = 0.0 if scorer_delta is None else float(scorer_delta)
    scorer_gain_value = 0.0 if scorer_gain is None else float(scorer_gain)
    decision_class = _decision_class(
        projected_delta=float(projected_delta),
        scorer_delta=scorer_delta_value,
        scorer_gain=scorer_gain_value,
        margin=margin,
        candidate_saved_bytes=candidate_saved_bytes,
        null_scorer_delta_epsilon=null_scorer_delta_epsilon,
        fragile_scorer_delta_threshold=fragile_scorer_delta_threshold,
    )
    dominant_axis = _dominant_axis(row_payload)
    family = str(row_payload.get("family") or "<family_missing>")
    coordinate_key = ":".join(
        (
            family,
            decision_class,
            dominant_axis,
            _pair_bucket(row_payload),
        )
    )
    return {
        "row_id": label,
        "candidate_id": row_payload.get("candidate_id"),
        "family": family,
        "source_label": source_label,
        "source_index": row_index,
        "coordinate_key": coordinate_key,
        "decision_surface_class": decision_class,
        "dominant_receiver_axis": dominant_axis,
        "pair_bucket": _pair_bucket(row_payload),
        "candidate_saved_bytes": candidate_saved_bytes,
        "added_archive_bytes": added_archive_bytes,
        "projected_delta_vs_baseline_score": float(projected_delta),
        "scorer_delta_vs_baseline": scorer_delta_value,
        "observed_scorer_gain_vs_baseline": scorer_gain_value,
        "byte_budget_margin_vs_break_even": margin,
        "native_mlx_window_objective": _is_mlx_scorer_response_row(row_payload)
        and allow_native_mlx_window_objective,
    }


def _planning_value_for_target(
    row: dict[str, Any],
    target: str,
    *,
    label: str,
    allow_native_mlx_window_objective: bool,
) -> float | None:
    try:
        return scorer_response_planning_value_for_target(row, target, label=label)
    except ScorerResponseDatasetError as exc:
        if (
            allow_native_mlx_window_objective
            and _is_mlx_scorer_response_row(row)
            and "missing normalized full-video objective" in str(exc)
        ):
            return _native_window_value(row, target)
        raise


def _native_window_value(row: dict[str, Any], target: str) -> float | None:
    if target in {"delta_vs_baseline_score", "projected_full_video_delta_vs_baseline_score"}:
        return _finite_float(row.get("delta_vs_baseline_score"))
    if target in {
        "observed_scorer_gain_vs_baseline",
        "normalized_full_video_scorer_gain_vs_baseline",
    }:
        return _finite_float(row.get("observed_scorer_gain_vs_baseline"))
    if target in {"scorer_delta_vs_baseline", "scorer_delta"}:
        value = _finite_float(row.get("scorer_delta_vs_baseline"))
        if value is not None:
            return value
        gain = _finite_float(row.get("observed_scorer_gain_vs_baseline"))
        return None if gain is None else -gain
    if target in {
        "byte_budget_margin_vs_break_even",
        "normalized_full_video_byte_budget_margin_vs_break_even",
    }:
        return _finite_float(row.get("byte_budget_margin_vs_break_even"))
    return None


def _decision_class(
    *,
    projected_delta: float,
    scorer_delta: float,
    scorer_gain: float,
    margin: float | None,
    candidate_saved_bytes: int,
    null_scorer_delta_epsilon: float,
    fragile_scorer_delta_threshold: float,
) -> str:
    if abs(scorer_delta) <= null_scorer_delta_epsilon and candidate_saved_bytes > 0:
        return "rate_only_null_space"
    if projected_delta < 0.0 and scorer_gain > 0.0 and (margin is None or margin >= 0.0):
        return "receiver_sufficient_statistic"
    if scorer_delta > fragile_scorer_delta_threshold or projected_delta > 0.0:
        return "fragile_boundary"
    return "decision_surface_sample"


def _dominant_axis(row: Mapping[str, Any]) -> str:
    candidates = {
        "seg": max(
            abs(_finite_float(row.get("diagnostic_seg_share")) or 0.0),
            abs(_finite_float(row.get("decoder_q_axis_share_seg")) or 0.0),
            abs(_finite_float(row.get("seg_delta_vs_baseline")) or 0.0),
        ),
        "pose": max(
            abs(_finite_float(row.get("diagnostic_pose_share")) or 0.0),
            abs(_finite_float(row.get("decoder_q_axis_share_pose")) or 0.0),
            abs(_finite_float(row.get("pose_delta_vs_baseline")) or 0.0),
        ),
        "rate": abs(_finite_float(row.get("decoder_q_axis_share_rate")) or 0.0),
    }
    axis, value = max(candidates.items(), key=lambda item: (item[1], item[0]))
    return axis if value > 0.0 else "mixed"


def _is_mlx_scorer_response_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("family") == "mlx_scorer_response"
        or row.get("source_schema") == "mlx_scorer_response.v1"
    )


def _pair_bucket(row: Mapping[str, Any]) -> str:
    pair_window = row.get("source_pair_window") or row.get("pair_indices")
    if isinstance(pair_window, list) and pair_window:
        parsed = [_finite_int(value) for value in pair_window[:2]]
        if all(value is not None for value in parsed):
            if len(parsed) == 1 or parsed[1] is None:
                return f"pair_{parsed[0]:04d}"
            return f"pair_{parsed[0]:04d}_{parsed[1]:04d}"
    start = _finite_int(row.get("source_start_pair"))
    if start is not None:
        return f"pair_{start:04d}"
    return "pair_all"


def _aggregate_cells(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["coordinate_key"]), []).append(sample)
    cells: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        projected = [float(row["projected_delta_vs_baseline_score"]) for row in rows]
        scorer_delta = [float(row["scorer_delta_vs_baseline"]) for row in rows]
        saved = [int(row["candidate_saved_bytes"]) for row in rows]
        cell_id = _safe_token(key)
        cells[key] = {
            "cell_id": cell_id,
            "coordinate_key": key,
            "decision_surface_class": rows[0]["decision_surface_class"],
            "dominant_receiver_axis": rows[0]["dominant_receiver_axis"],
            "pair_bucket": rows[0]["pair_bucket"],
            "families": ordered_unique(str(row["family"]) for row in rows),
            "row_count": len(rows),
            "source_row_ids": ordered_unique(str(row["row_id"]) for row in rows),
            "source_candidate_ids": ordered_unique(
                str(row["candidate_id"])
                for row in rows
                if row.get("candidate_id") is not None
            ),
            "candidate_saved_bytes": max(saved) if saved else 0,
            "median_projected_delta_vs_baseline_score": float(statistics.median(projected)),
            "median_scorer_delta_vs_baseline": float(statistics.median(scorer_delta)),
            "best_projected_delta_vs_baseline_score": float(min(projected)),
            "worst_projected_delta_vs_baseline_score": float(max(projected)),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    return cells


def _unit_from_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    saved_bytes = int(cell.get("candidate_saved_bytes") or 0)
    projected_delta = float(cell["median_projected_delta_vs_baseline_score"])
    rate_delta = -RATE_SCORE_PER_BYTE * float(saved_bytes)
    quality_delta = projected_delta - rate_delta
    operation = {
        "operation_id": OPERATION_PROBE,
        "operation_family": OPERATION_PROBE,
        "candidate_saved_bytes": saved_bytes,
        "predicted_quality_score_delta": quality_delta,
        "params": {
            "cell_id": cell["cell_id"],
            "coordinate_key": cell["coordinate_key"],
            "decision_surface_class": cell["decision_surface_class"],
            "dominant_receiver_axis": cell["dominant_receiver_axis"],
            "source_row_ids": list(cell.get("source_row_ids") or []),
        },
        "blockers": [
            "inverse_surface_cell_requires_materializer",
            "inverse_surface_cell_requires_runtime_consumption_proof",
            "inverse_surface_cell_requires_exact_auth_eval_before_score_claim",
        ],
    }
    return {
        "unit_id": f"inverse_surface_{cell['cell_id']}",
        "unit_kind": UNIT_KIND,
        "candidate_saved_bytes": saved_bytes,
        "predicted_quality_score_delta": quality_delta,
        "confidence": _confidence(cell),
        "operation_families": [OPERATION_PROBE, OPERATION_MATERIALIZE],
        "operations": [operation],
        "decision_surface_class": cell["decision_surface_class"],
        "dominant_receiver_axis": cell["dominant_receiver_axis"],
        "source_row_ids": list(cell.get("source_row_ids") or []),
        "source_candidate_ids": list(cell.get("source_candidate_ids") or []),
        "evidence_semantics": "inverse_scorer_decision_surface_planning_cell",
        "planning_value_scope": "compressed_scorer_coordinate",
        "projected_full_video_delta_vs_baseline_score": projected_delta,
        "median_scorer_delta_vs_baseline": cell["median_scorer_delta_vs_baseline"],
        "blockers": [
            "inverse_surface_unit_is_planning_only",
            "requires_materializer_before_candidate_archive",
            "requires_exact_auth_eval_before_score_claim",
        ],
        **FALSE_AUTHORITY,
    }


def _confidence(cell: Mapping[str, Any]) -> float:
    row_count = int(cell.get("row_count") or 0)
    base = min(0.95, 0.35 + 0.1 * row_count)
    if cell.get("decision_surface_class") == "rate_only_null_space":
        return min(0.95, base + 0.1)
    if cell.get("decision_surface_class") == "fragile_boundary":
        return max(0.1, base - 0.1)
    return base


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token[:96] or "cell"


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "OPERATION_MATERIALIZE",
    "OPERATION_PROBE",
    "SCHEMA",
    "UNIT_KIND",
    "build_inverse_scorer_decision_surface",
]
