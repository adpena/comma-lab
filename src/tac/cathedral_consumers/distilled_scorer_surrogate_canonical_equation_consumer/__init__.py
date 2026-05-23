# SPDX-License-Identifier: MIT
"""Cathedral consumer for distilled scorer surrogate evidence.

This is the Catalog #335 bridge between the LL scorer-response dataset and the
PACT-NERV-DistilledScorer substrate. It consumes two safe-by-construction
surfaces:

* scorer-response rows with family ``distilled_vs_direct_scorer_paired_smoke``;
* master-gradient anchors that can later provide frame/pair weighting context.

The consumer is Tier-A observability only. It never creates a score claim,
never marks a candidate promotable, and never dispatches work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber
from tac.optimization.scorer_response_dataset import (
    ScorerResponseDatasetError,
    scorer_response_planning_value_for_target,
)

CONSUMER_NAME = "distilled_scorer_surrogate_canonical_equation_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)

CONSUMES_SCORER_RESPONSE_DATASET = True
CONSUMES_MASTER_GRADIENT_ANCHORS = True

CANONICAL_EQUATION_ID = "kl_distillation_scorer_surrogate_compression_savings_v1"
ROW_FAMILY = "distilled_vs_direct_scorer_paired_smoke"

_ROW_CONTAINER_KEYS = (
    "scorer_response_row",
    "ll_scorer_response_row",
    "distilled_vs_direct_scorer_row",
)
_DATASET_CONTAINER_KEYS = (
    "scorer_response_dataset",
    "ll_scorer_response_dataset",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5.

    Future fan-out hooks may call this for either scorer-response rows or
    master-gradient anchors. The consumer has no mutable local cache; it
    re-derives routing signal from each candidate payload.
    """

    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return non-promotional routing metadata for distilled scorer evidence."""

    if not isinstance(candidate, Mapping):
        return _no_signal("candidate is not a mapping")

    rows = _extract_distilled_rows(candidate)
    if not rows:
        return _no_signal("no distilled-vs-direct scorer-response rows")

    unsafe = _first_unsafe_row(rows)
    if unsafe is not None:
        return _blocked_signal(unsafe)
    invalid_planning_row = _first_invalid_planning_row(rows)
    if invalid_planning_row is not None:
        return _blocked_signal(invalid_planning_row)

    best_total = _best_numeric_row(rows, "delta_vs_baseline_score", lower_is_better=True)
    best_scorer = _best_numeric_row(rows, "scorer_delta_vs_baseline", lower_is_better=True)
    row_ids = [str(row.get("row_id") or row.get("candidate_id") or index) for index, row in enumerate(rows)]
    n_improved_total = sum(
        1
        for row in rows
        if (value := _planning_value(row, "delta_vs_baseline_score")) is not None
        and value < 0.0
    )
    n_improved_scorer = sum(
        1
        for row in rows
        if (value := _planning_value(row, "scorer_delta_vs_baseline")) is not None
        and value < 0.0
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "distilled scorer surrogate rows are present; route to "
            "PACT-NERV-DistilledScorer / #523 KL-T=2.0 analysis only "
            "until paired CPU+CUDA anchors exist [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "consumer_signal_kind": "distilled_scorer_surrogate_routing",
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "row_family": ROW_FAMILY,
        "row_count": len(rows),
        "row_ids": row_ids,
        "planning_target_accessor": "scorer_response_planning_value_for_target",
        "improved_total_score_count": n_improved_total,
        "improved_scorer_term_count": n_improved_scorer,
        "best_total_row": best_total,
        "best_scorer_row": best_scorer,
        "consumes_scorer_response_dataset": True,
        "consumes_master_gradient_anchors": True,
    }


def _extract_distilled_rows(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    direct_family = candidate.get("family")
    if direct_family == ROW_FAMILY:
        rows.append(candidate)

    for key in _ROW_CONTAINER_KEYS:
        value = candidate.get(key)
        if isinstance(value, Mapping) and value.get("family") == ROW_FAMILY:
            rows.append(value)

    for key in _DATASET_CONTAINER_KEYS:
        value = candidate.get(key)
        if isinstance(value, Mapping):
            rows.extend(_rows_from_dataset(value))

    rows.extend(_rows_from_dataset(candidate))
    return _dedupe_rows(rows)


def _rows_from_dataset(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        return []
    return [
        row
        for row in raw_rows
        if isinstance(row, Mapping) and row.get("family") == ROW_FAMILY
    ]


def _dedupe_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        key = str(row.get("row_id") or row.get("candidate_id") or f"index:{index}")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _first_unsafe_row(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        for key in (
            "score_claim",
            "score_claim_valid",
            "promotion_eligible",
            "ready_for_exact_eval_dispatch",
            "rank_or_kill_eligible",
            "promotable",
        ):
            if row.get(key) is True:
                return {"row": row, "field": key}
    return None


def _first_invalid_planning_row(
    rows: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for row in rows:
        for field in ("delta_vs_baseline_score", "scorer_delta_vs_baseline"):
            try:
                _planning_value(row, field)
            except ScorerResponseDatasetError as exc:
                return {"row": row, "field": field, "error": str(exc)}
    return None


def _best_numeric_row(
    rows: list[Mapping[str, Any]],
    field: str,
    *,
    lower_is_better: bool,
) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_value: float | None = None
    for row in rows:
        value = _planning_value(row, field)
        if value is None:
            continue
        if best is None or (
            value < best_value if lower_is_better else value > best_value
        ):
            best = {
                "row_id": row.get("row_id"),
                "candidate_id": row.get("candidate_id"),
                "family": row.get("family"),
                field: value,
                "planning_target_accessor": (
                    "scorer_response_planning_value_for_target"
                ),
                "added_archive_bytes": row.get("added_archive_bytes"),
            }
            best_value = value
    return best


def _planning_value(row: Mapping[str, Any], field: str) -> float | None:
    value = scorer_response_planning_value_for_target(
        dict(row),
        field,
        label=str(row.get("row_id") or row.get("candidate_id") or "distilled row"),
    )
    return _finite_float(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _blocked_signal(unsafe: Mapping[str, Any]) -> Mapping[str, Any]:
    field = unsafe.get("field")
    error = unsafe.get("error")
    row = unsafe.get("row") if isinstance(unsafe.get("row"), Mapping) else {}
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "distilled scorer surrogate consumer refused source row with "
            f"{field}={error or 'true'}; keep LL/distillation evidence "
            "fail-closed [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "consumer_signal_kind": "distilled_scorer_surrogate_authority_blocked",
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "row_id": row.get("row_id"),
        "blocked_field": field,
        "blocked_error": error,
    }


def _no_signal(reason: str) -> Mapping[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"distilled scorer surrogate consumer: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "consumer_signal_kind": "distilled_scorer_surrogate_absent",
        "canonical_equation_id": CANONICAL_EQUATION_ID,
    }


__all__ = [
    "CANONICAL_EQUATION_ID",
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMES_MASTER_GRADIENT_ANCHORS",
    "CONSUMES_SCORER_RESPONSE_DATASET",
    "ROW_FAMILY",
    "consume_candidate",
    "update_from_anchor",
]
