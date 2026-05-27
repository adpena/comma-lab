# SPDX-License-Identifier: MIT
"""Cathedral consumer for 5D-canvas extended operator candidates.

This package makes the eight DROP-MANY audit operators auto-discoverable by
the cathedral autopilot. It is intentionally Tier A: it annotates candidate
rows with the extended operator family and equation id, but it never changes
predicted delta, promotion eligibility, rank authority, or score authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import ConsumerTier, HookNumber
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    EXTENDED_OPERATION_CANONICAL_EQUATION_IDS,
    ExtendedOperation,
)

CONSUMER_NAME = "pair_frame_5d_extended_operator_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Accept continual-learning anchors without mutating score authority."""
    _ = anchor


def _candidate_operation(candidate: Mapping[str, Any]) -> ExtendedOperation | None:
    hint = candidate.get("canonical_dispatch_recipe_hint")
    if isinstance(hint, Mapping):
        operation_value = hint.get("operation")
        if isinstance(operation_value, str):
            try:
                return ExtendedOperation(operation_value)
            except ValueError:
                return None
    operation_value = candidate.get("extended_operation") or candidate.get("operation")
    if isinstance(operation_value, str):
        try:
            return ExtendedOperation(operation_value)
        except ValueError:
            return None
    return None


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Annotate extended-operator candidates for cathedral autopilot planning.

    The return payload preserves Catalog #341 false-authority markers. The
    autopilot can route or inspect the candidate family, but exact score,
    rank, kill, and promotion authority remain gated by paired contest-axis
    eval artifacts.
    """
    operation = _candidate_operation(candidate) if isinstance(candidate, Mapping) else None
    if operation is None:
        return {
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "confidence": 0.0,
            "consumer_name": CONSUMER_NAME,
            "readiness_verdict": "NOT_APPLICABLE",
            "rationale": "Candidate has no recognized 5D extended operator hint.",
        }

    equation_id = EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[operation]
    return {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "confidence": 0.0,
        "consumer_name": CONSUMER_NAME,
        "readiness_verdict": "PLANNING_VISIBLE",
        "extended_operation": operation.value,
        "canonical_equation_id_pending": equation_id,
        "rationale": (
            f"5D extended operator {operation.value} is cathedral-visible for "
            f"queue planning under {equation_id}; false-authority markers "
            f"preserved until paired contest-axis eval exists."
        ),
    }
