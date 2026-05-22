# SPDX-License-Identifier: MIT
"""Pairset component-marginal signal helpers.

This module is the reusable bridge between exact auth-eval component traces,
portfolio planning, xray discovery, and canonical equations. It carries no
score authority: every payload is a planning signal until a separate exact
auth-axis artifact is evaluated and promoted through the normal gates.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

CONTEST_RATE_DENOMINATOR_BYTES = 37_545_489
CONTEST_RATE_MULTIPLIER = 25.0

PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID = (
    "pairset_component_marginal_score_decomposition_v1"
)
PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME = "pairset_component_marginal"
PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA = "pairset_component_marginal_model.v1"
PAIRSET_COMPONENT_MARGINAL_SCORE_DELTA_SCHEMA = (
    "pairset_component_marginal_score_delta.v1"
)
PAIRSET_COMPONENT_MARGINAL_SIGNAL_REFS_SCHEMA = (
    "pairset_component_marginal_canonical_signal_refs.v1"
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return out


def rate_delta_for_archive_byte_delta(
    archive_byte_delta: float,
    *,
    rate_multiplier: float = CONTEST_RATE_MULTIPLIER,
    denominator_bytes: int = CONTEST_RATE_DENOMINATOR_BYTES,
) -> float:
    """Return score-rate delta for an archive byte-count delta.

    Negative ``archive_byte_delta`` means bytes were removed and therefore
    lowers the rate term.
    """

    byte_delta = _finite_float(archive_byte_delta, label="archive_byte_delta")
    multiplier = _finite_float(rate_multiplier, label="rate_multiplier")
    denom = int(denominator_bytes)
    if denom <= 0:
        raise ValueError("denominator_bytes must be positive")
    return multiplier * byte_delta / float(denom)


def component_score_delta(
    *,
    segnet_delta: float = 0.0,
    posenet_delta: float = 0.0,
    rate_delta: float | None = None,
    archive_byte_delta: float | None = None,
) -> float:
    """Compute the component-decomposed score delta.

    Exactly one of ``rate_delta`` or ``archive_byte_delta`` may be supplied.
    The returned value follows contest score direction: negative is better.
    """

    seg = _finite_float(segnet_delta, label="segnet_delta")
    pose = _finite_float(posenet_delta, label="posenet_delta")
    if rate_delta is not None and archive_byte_delta is not None:
        raise ValueError("provide rate_delta OR archive_byte_delta, not both")
    if rate_delta is None:
        rate = 0.0 if archive_byte_delta is None else rate_delta_for_archive_byte_delta(archive_byte_delta)
    else:
        rate = _finite_float(rate_delta, label="rate_delta")
    return seg + pose + rate


def component_marginal_status(
    *,
    segnet_delta: float = 0.0,
    posenet_delta: float = 0.0,
    rate_delta: float | None = None,
    archive_byte_delta: float | None = None,
) -> str:
    """Classify whether rate credit beats scorer penalty."""

    seg = _finite_float(segnet_delta, label="segnet_delta")
    pose = _finite_float(posenet_delta, label="posenet_delta")
    if rate_delta is not None and archive_byte_delta is not None:
        raise ValueError("provide rate_delta OR archive_byte_delta, not both")
    if rate_delta is not None:
        rate = _finite_float(rate_delta, label="rate_delta")
    elif archive_byte_delta is not None:
        rate = rate_delta_for_archive_byte_delta(archive_byte_delta)
    else:
        rate = 0.0
    scorer_penalty = seg + pose
    rate_credit = -rate
    if scorer_penalty < rate_credit:
        return "rate_credit_exceeds_scorer_penalty"
    if scorer_penalty > rate_credit:
        return "scorer_penalty_exceeds_rate_credit"
    return "rate_credit_ties_scorer_penalty"


def build_component_score_delta_payload(
    *,
    segnet_delta: float = 0.0,
    posenet_delta: float = 0.0,
    rate_delta: float | None = None,
    archive_byte_delta: float | None = None,
    axis: str = "",
    candidate_id: str = "",
    pair_index: int | None = None,
    dropped_pair_rank: int | None = None,
) -> dict[str, Any]:
    """Build a machine-readable component marginal payload."""

    seg = _finite_float(segnet_delta, label="segnet_delta")
    pose = _finite_float(posenet_delta, label="posenet_delta")
    if rate_delta is None:
        if archive_byte_delta is None:
            rate = 0.0
            byte_delta = None
        else:
            byte_delta = _finite_float(archive_byte_delta, label="archive_byte_delta")
            rate = rate_delta_for_archive_byte_delta(byte_delta)
    else:
        if archive_byte_delta is not None:
            raise ValueError("provide rate_delta OR archive_byte_delta, not both")
        rate = _finite_float(rate_delta, label="rate_delta")
        byte_delta = None
    scorer_penalty = seg + pose
    rate_credit = -rate
    return {
        "schema": PAIRSET_COMPONENT_MARGINAL_SCORE_DELTA_SCHEMA,
        "candidate_id": candidate_id,
        "axis": axis,
        "pair_index": pair_index,
        "dropped_pair_rank": dropped_pair_rank,
        "segnet_delta": seg,
        "posenet_delta": pose,
        "rate_delta": rate,
        "archive_byte_delta": byte_delta,
        "scorer_penalty": scorer_penalty,
        "rate_credit": rate_credit,
        "net_component_delta": seg + pose + rate,
        "component_marginal_status": component_marginal_status(
            segnet_delta=seg,
            posenet_delta=pose,
            rate_delta=rate,
        ),
        "score_direction": "negative_is_better",
        "allowed_use": "component_marginal_planning_signal_only_no_score_authority",
        **FALSE_AUTHORITY,
    }


def canonical_signal_refs() -> dict[str, Any]:
    """Return canonical discoverability refs for this signal family."""

    return {
        "schema": PAIRSET_COMPONENT_MARGINAL_SIGNAL_REFS_SCHEMA,
        "xray_primitives": [
            PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME,
            "per_pair_score_decomposition",
            "segnet_margin_polytope",
            "posenet_se3_lie_algebra",
            "score_lipschitz",
        ],
        "canonical_equations": [
            PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID,
            "per_pair_master_gradient_score_impact_taylor_v1",
            "canonical_frontier_pointer_v1",
        ],
        "master_gradient_consumers": [
            "tac.master_gradient_consumers.per_pair_difficulty_atlas",
            "tac.master_gradient_consumers.per_pair_pareto_envelope",
            "tac.master_gradient_consumers.per_pair_lagrangian_lambda_bisection",
            "tac.master_gradient_consumers.per_pair_coding_budget_allocation",
            "tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual",
        ],
        "wire_in_hooks": {
            "sensitivity_map": [
                "component_deltas_by_pair_and_axis",
                "master_gradient_pair_norm_reweighting",
            ],
            "pareto_constraint": [
                "rate_credit_vs_scorer_penalty_drop_condition",
            ],
            "bit_allocator": [
                "protect_pairs_where_scorer_penalty_exceeds_rate_credit",
                "prefer_pairs_where_rate_credit_exceeds_scorer_penalty",
            ],
            "cathedral_autopilot": [
                "operator_action_queue_pairset_candidate_feedback",
            ],
            "continual_learning": [
                "exact_axis_component_marginal_empirical_anchors",
            ],
            "probe_disambiguator": [
                "cpu_cuda_transfer_status",
                "selected_pair_identity_required",
            ],
        },
        "identity_policy": "candidate_id_and_selected_pair_indices_required_and_matched",
        "allowed_use": "planning_signal_only_no_score_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def attach_canonical_signal_refs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with canonical discoverability refs attached."""

    out = dict(payload)
    out["canonical_signal_refs"] = canonical_signal_refs()
    return out


__all__ = [
    "CONTEST_RATE_DENOMINATOR_BYTES",
    "CONTEST_RATE_MULTIPLIER",
    "FALSE_AUTHORITY",
    "PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA",
    "PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID",
    "PAIRSET_COMPONENT_MARGINAL_SCORE_DELTA_SCHEMA",
    "PAIRSET_COMPONENT_MARGINAL_SIGNAL_REFS_SCHEMA",
    "PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME",
    "attach_canonical_signal_refs",
    "build_component_score_delta_payload",
    "canonical_signal_refs",
    "component_marginal_status",
    "component_score_delta",
    "rate_delta_for_archive_byte_delta",
]
