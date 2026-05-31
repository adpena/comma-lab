# SPDX-License-Identifier: MIT
"""Contest-space action functional helpers.

This module is intentionally small and reusable: it encodes the hydrated
contest objective over scorer components and archive bytes, while refusing to
act as score authority. Queue planners can use it to compare local acquisition
rows, but exact CPU/CUDA auth eval remains the only promotion authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tac.master_gradient import CONTEST_RATE_DENOM_BYTES
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)

CONTEST_SPACE_ACTION_FUNCTIONAL_SCHEMA = "contest_space_action_functional.v1"
CONTEST_SPACE_ACTION_ROW_SCHEMA = "contest_space_action_row.v1"
CONTEST_SPACE_HYDRATION_SCHEMA = "contest_space_hydration_contract.v1"
CONTEST_SPACE_RATE_DISTORTION_ACTION_KIND = "rate_distortion_candidate"
CONTEST_SPACE_REPAIR_BUDGET_ACTION_KIND = "repair_budget_spend"

SEG_SCORE_MULTIPLIER = 100.0
POSE_SQRT_INNER_MULTIPLIER = 10.0
RATE_SCORE_MULTIPLIER = 25.0
RATE_SCORE_PER_BYTE = RATE_SCORE_MULTIPLIER / CONTEST_RATE_DENOM_BYTES
CONTEST_OBJECTIVE_EQUATION = (
    "S = 100*d_seg + sqrt(10*d_pose) + "
    f"25*archive_bytes/{CONTEST_RATE_DENOM_BYTES}"
)


class ContestSpaceActionError(ValueError):
    """Raised when a contest-space action row is malformed."""


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def _non_empty_string(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ContestSpaceActionError(f"{field} must be non-empty")
    return text


def contest_rate_from_archive_bytes(archive_bytes: int | float) -> float:
    """Return contest rate term input `archive_bytes / denominator`."""

    bytes_f = _number(archive_bytes)
    if bytes_f is None or bytes_f < 0:
        raise ContestSpaceActionError(f"archive_bytes must be finite non-negative, got {archive_bytes!r}")
    return bytes_f / float(CONTEST_RATE_DENOM_BYTES)


def contest_score_from_components(
    *,
    avg_segnet_dist: float,
    avg_posenet_dist: float,
    archive_bytes: int | float | None = None,
    rate: float | None = None,
) -> float:
    """Compute the official scalar objective from component coordinates."""

    seg = _number(avg_segnet_dist)
    pose = _number(avg_posenet_dist)
    if seg is None or seg < 0:
        raise ContestSpaceActionError(f"avg_segnet_dist must be finite non-negative, got {avg_segnet_dist!r}")
    if pose is None or pose < 0:
        raise ContestSpaceActionError(f"avg_posenet_dist must be finite non-negative, got {avg_posenet_dist!r}")
    if rate is None:
        if archive_bytes is None:
            raise ContestSpaceActionError("one of archive_bytes or rate is required")
        rate = contest_rate_from_archive_bytes(archive_bytes)
    rate_f = _number(rate)
    if rate_f is None or rate_f < 0:
        raise ContestSpaceActionError(f"rate must be finite non-negative, got {rate!r}")
    return (
        SEG_SCORE_MULTIPLIER * seg
        + math.sqrt(POSE_SQRT_INNER_MULTIPLIER * pose)
        + RATE_SCORE_MULTIPLIER * rate_f
    )


def rate_score_credit(saved_bytes: int | float) -> float:
    """Score units gained by saving `saved_bytes` archive bytes."""

    saved_f = _number(saved_bytes)
    if saved_f is None:
        raise ContestSpaceActionError(f"saved_bytes must be finite, got {saved_bytes!r}")
    return saved_f * RATE_SCORE_PER_BYTE


def build_hydration_contract(
    *,
    video_scope: str,
    scorer_axis: str,
    archive_axis: str,
    runtime_contract: str,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Describe which contest-space slice a local row hydrates."""

    if sample_count is not None and (
        not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0
    ):
        raise ContestSpaceActionError(
            f"sample_count must be a non-negative integer or None, got {sample_count!r}"
        )
    payload = {
        "schema": CONTEST_SPACE_HYDRATION_SCHEMA,
        "video_scope": _non_empty_string(video_scope, field="video_scope"),
        "scorer_axis": _non_empty_string(scorer_axis, field="scorer_axis"),
        "archive_axis": _non_empty_string(archive_axis, field="archive_axis"),
        "runtime_contract": _non_empty_string(runtime_contract, field="runtime_contract"),
        "sample_count": sample_count,
        "constraints": [
            "full_frame_receiver_output_required_for_candidate_authority",
            "local_cpu_or_mlx_rows_are_planning_signal_only",
            "exact_cpu_cuda_auth_eval_required_before_score_or_promotion_claim",
        ],
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="contest_space_hydration_contract")
    return payload


def build_rate_distortion_action_row(
    *,
    candidate_id: str,
    observed_net_delta_score_units: float | None,
    saved_bytes: int | float | None,
    local_cpu_score: float | None = None,
    local_cpu_avg_segnet_dist: float | None = None,
    local_cpu_avg_posenet_dist: float | None = None,
    hydration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one action row with explicit rate-credit and distortion-spend math."""

    delta = _number(observed_net_delta_score_units)
    saved = _number(saved_bytes) if saved_bytes is not None else 0.0
    if saved is None:
        saved = 0.0
    credit = rate_score_credit(saved)
    distortion_spend = delta + credit if delta is not None else None
    extra_saved = math.ceil(delta / RATE_SCORE_PER_BYTE) if delta is not None and delta > 0.0 else 0
    row = {
        "schema": CONTEST_SPACE_ACTION_ROW_SCHEMA,
        "action_kind": CONTEST_SPACE_RATE_DISTORTION_ACTION_KIND,
        "candidate_id": str(candidate_id),
        "observed_net_delta_score_units": delta,
        "saved_bytes": int(saved),
        "rate_score_credit": credit,
        "estimated_distortion_spend_equation": (
            "observed_net_delta_score_units + saved_bytes*rate_score_per_byte"
        ),
        "estimated_distortion_spend_score_units": distortion_spend,
        "break_even_equation": (
            "ceil(max(observed_net_delta_score_units, 0) / rate_score_per_byte)"
        ),
        "extra_saved_bytes_to_break_even": int(extra_saved),
        "total_saved_bytes_to_break_even": int(saved) + int(extra_saved),
        "local_cpu_score": _number(local_cpu_score),
        "local_cpu_avg_segnet_dist": _number(local_cpu_avg_segnet_dist),
        "local_cpu_avg_posenet_dist": _number(local_cpu_avg_posenet_dist),
        "hydration": dict(hydration) if hydration is not None else None,
        "acceptance_state": (
            "local_gate_passed"
            if delta is not None and delta < 0.0
            else "local_gate_failed"
            if delta is not None
            else "missing_delta"
        ),
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(row, context=f"contest_space_action_row:{candidate_id}")
    return row


def build_repair_budget_action_row(
    *,
    candidate_id: str,
    expected_distortion_delta_score_units: float | None,
    repair_spend_bytes: int | float | None,
    available_rate_credit_bytes: int | float | None = None,
    local_cpu_score: float | None = None,
    local_cpu_avg_segnet_dist: float | None = None,
    local_cpu_avg_posenet_dist: float | None = None,
    hydration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one row for spending receiver-closed rate credit on repair.

    A repair row is the inverse of a pure rate row: spending bytes is a positive
    rate cost, so the local distortion delta must beat that cost before the
    planner can treat the candidate as a local winner.
    """

    candidate = _non_empty_string(candidate_id, field="candidate_id")
    delta = _number(expected_distortion_delta_score_units)
    spend = _number(repair_spend_bytes) if repair_spend_bytes is not None else 0.0
    if spend is None or spend < 0.0:
        raise ContestSpaceActionError(
            f"repair_spend_bytes must be finite non-negative, got {repair_spend_bytes!r}"
        )
    available = (
        _number(available_rate_credit_bytes)
        if available_rate_credit_bytes is not None
        else None
    )
    if available is not None and available < 0.0:
        raise ContestSpaceActionError(
            "available_rate_credit_bytes must be finite non-negative or None, "
            f"got {available_rate_credit_bytes!r}"
        )
    spend_int = int(spend)
    saved_bytes = -spend_int
    rate_cost = spend * RATE_SCORE_PER_BYTE
    net_delta = delta + rate_cost if delta is not None else None
    break_even_spend = (
        math.floor(max(0.0, -delta) / RATE_SCORE_PER_BYTE)
        if delta is not None and delta < 0.0
        else 0
    )
    credit_margin = int(available) - spend_int if available is not None else None
    blockers = []
    if delta is None:
        blockers.append("expected_distortion_delta_missing")
    if available is not None and spend > available:
        blockers.append("repair_spend_exceeds_available_rate_credit")
    if net_delta is not None and net_delta >= 0.0:
        blockers.append("net_delta_after_rate_spend_not_improving")
    acceptance_state = (
        "missing_delta"
        if delta is None
        else "local_gate_passed"
        if not blockers and net_delta is not None and net_delta < 0.0
        else "local_gate_failed"
    )
    row = {
        "schema": CONTEST_SPACE_ACTION_ROW_SCHEMA,
        "action_kind": CONTEST_SPACE_REPAIR_BUDGET_ACTION_KIND,
        "candidate_id": candidate,
        "expected_distortion_delta_score_units": delta,
        "repair_spend_bytes": spend_int,
        "available_rate_credit_bytes": int(available) if available is not None else None,
        "rate_score_cost": rate_cost,
        "rate_score_credit": -rate_cost,
        "saved_bytes": saved_bytes,
        "observed_net_delta_score_units": net_delta,
        "net_delta_after_rate_spend_score_units": net_delta,
        "estimated_distortion_spend_equation": (
            "expected_distortion_delta_score_units + repair_spend_bytes*rate_score_per_byte"
        ),
        "estimated_distortion_spend_score_units": delta,
        "break_even_equation": (
            "floor(max(-expected_distortion_delta_score_units, 0) / rate_score_per_byte)"
        ),
        "max_repair_spend_bytes_to_break_even": int(break_even_spend),
        "rate_credit_margin_bytes_after_spend": credit_margin,
        "local_cpu_score": _number(local_cpu_score),
        "local_cpu_avg_segnet_dist": _number(local_cpu_avg_segnet_dist),
        "local_cpu_avg_posenet_dist": _number(local_cpu_avg_posenet_dist),
        "hydration": dict(hydration) if hydration is not None else None,
        "acceptance_state": acceptance_state,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"contest_space_repair_budget_action_row:{candidate}",
    )
    return row


def build_contest_space_action_functional(
    *,
    rows: Sequence[Mapping[str, Any]],
    hydration: Mapping[str, Any],
    objective: str = "minimize_contest_score_delta_under_receiver_and_auth_constraints",
) -> dict[str, Any]:
    """Aggregate action rows into a planner-owned functional."""

    action_rows = [dict(row) for row in rows]
    deltas = [
        value
        for row in action_rows
        if (value := _number(row.get("observed_net_delta_score_units"))) is not None
    ]
    saved_total = sum(int(row.get("saved_bytes") or 0) for row in action_rows)
    rate_credit_total = sum(float(row.get("rate_score_credit") or 0.0) for row in action_rows)
    distortion_spend_total = sum(
        float(row.get("estimated_distortion_spend_score_units") or 0.0)
        for row in action_rows
        if row.get("estimated_distortion_spend_score_units") is not None
    )
    action_kind_histogram: dict[str, int] = {}
    for row in action_rows:
        action_kind = str(row.get("action_kind") or "unknown_action_kind")
        action_kind_histogram[action_kind] = action_kind_histogram.get(action_kind, 0) + 1
    payload = {
        "schema": CONTEST_SPACE_ACTION_FUNCTIONAL_SCHEMA,
        "row_schema": CONTEST_SPACE_ACTION_ROW_SCHEMA,
        "objective": objective,
        "objective_equation": CONTEST_OBJECTIVE_EQUATION,
        "component_terms": {
            "segnet_multiplier": SEG_SCORE_MULTIPLIER,
            "posenet_sqrt_inner_multiplier": POSE_SQRT_INNER_MULTIPLIER,
            "rate_multiplier": RATE_SCORE_MULTIPLIER,
            "rate_denominator_bytes": CONTEST_RATE_DENOM_BYTES,
        },
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "hydration": dict(hydration),
        "row_count": len(action_rows),
        "local_gate_passed_count": sum(
            1 for row in action_rows if row.get("acceptance_state") == "local_gate_passed"
        ),
        "best_observed_net_delta_score_units": min(deltas) if deltas else None,
        "worst_observed_net_delta_score_units": max(deltas) if deltas else None,
        "saved_bytes_total": saved_total,
        "rate_score_credit_total": rate_credit_total,
        "estimated_distortion_spend_score_units_total": distortion_spend_total,
        "action_kind_histogram": dict(sorted(action_kind_histogram.items())),
        "break_even_extra_saved_bytes_min": min(
            (int(row.get("extra_saved_bytes_to_break_even") or 0) for row in action_rows),
            default=None,
        ),
        "rows": action_rows,
        "blockers": ordered_unique(
            [
                "exact_auth_eval_required_before_score_or_promotion_claim",
                *(
                    ["no_local_gate_passed_rows"]
                    if action_rows
                    and not any(row.get("acceptance_state") == "local_gate_passed" for row in action_rows)
                    else []
                ),
            ]
        ),
        "allowed_use": "local_queue_planning_and_acquisition_policy",
        "forbidden_use": "score_claim_rank_kill_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="contest_space_action_functional")
    return payload


__all__ = [
    "CONTEST_OBJECTIVE_EQUATION",
    "CONTEST_RATE_DENOM_BYTES",
    "CONTEST_SPACE_ACTION_FUNCTIONAL_SCHEMA",
    "CONTEST_SPACE_ACTION_ROW_SCHEMA",
    "CONTEST_SPACE_HYDRATION_SCHEMA",
    "CONTEST_SPACE_RATE_DISTORTION_ACTION_KIND",
    "CONTEST_SPACE_REPAIR_BUDGET_ACTION_KIND",
    "RATE_SCORE_PER_BYTE",
    "ContestSpaceActionError",
    "build_contest_space_action_functional",
    "build_hydration_contract",
    "build_rate_distortion_action_row",
    "build_repair_budget_action_row",
    "contest_rate_from_archive_bytes",
    "contest_score_from_components",
    "rate_score_credit",
]
