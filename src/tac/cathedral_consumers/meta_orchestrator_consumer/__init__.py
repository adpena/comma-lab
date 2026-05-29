# SPDX-License-Identifier: MIT
"""Cathedral consumer for the META-orchestrator extension (Wave N+46).

Per ``feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md``
+ Catalog #335 paradigm-shift (canonical contract auto-discovery).

Surfaces the canonical 3-metric trichotomy ranking +
no-ad-hoc/signal-loss/rediscovery/duplicate/drift invariant verdict
at every cap-window for autopilot/operator/main-thread inheritance.

This consumer is observability-only (predicted_delta_adjustment=0.0; per
CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323;
Tier A canonical-routing markers per Catalog #341).

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE PRIMARY (surfaces 3-metric
    trichotomy + invariant verdict per candidate so the autopilot
    ranker can route per-canonical-metric)
  * #5 continual-learning posterior — ACTIVE (every operator binding
    correction registered via register_operator_binding_correction
    auto-fires Catalog #371 recalibrator within same turn per
    operator_correction_canonical_apparatus_mutation_lag_v1 unwind)
  * #6 probe-disambiguator — ACTIVE (canonical 3-metric ranking IS the
    disambiguator between hygiene-EV vs frontier-breaking-EV vs
    highest-EV-shortest-wall-clock per
    meta_orchestrator_three_metric_trichotomy_orthogonality_v1)
  * #1, #2, #3 — N/A (observability-only annotation)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import (
    ConsumerTier,
    HookNumber,
)


CONSUMER_NAME = "meta_orchestrator_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)
# Tier A per Catalog #341 — observability-only at landing. Future Tier B
# promotion path per Catalog #357 when empirical per-axis anchors land.
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Every new empirical anchor that flows into the continual-learning
    posterior is consumed via the canonical equation registry's
    auto-recalibrator per Catalog #371. THIS consumer is structurally
    NO-OP here — the canonical refresh is operator-triggered via
    :func:`tac.cathedral_autopilot.register_operator_binding_correction`
    or directly via ``tools/recalibrate_equation.py``.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — surface 3-metric trichotomy per candidate.

    The consumer computes the 3-metric trichotomy for the candidate +
    emits an observability-only annotation per Catalog #287/#341.

    Returns:
        Mapping with the canonical Tier A markers per Catalog #341:
          - ``predicted_delta_adjustment=0.0`` (Tier A: routing-only)
          - ``rationale`` (operator-facing readable; surfaces all 3 metrics)
          - ``axis_tag="[predicted]"``
          - ``promotable=False``
          - ``confidence=0.0``
          - ``three_metric_trichotomy`` (the typed sub-result for
            downstream autopilot consumption)
    """
    try:
        from tac.cathedral_autopilot.three_metric_trichotomy import (
            CandidateWithThreeMetric,
            HIGHEST_EV_SHORTEST_WALL_CLOCK,
            _compute_frontier_breaking_ev,
            _compute_highest_ev_shortest_wall_clock_ev,
            _compute_hygiene_ev,
        )
    except ImportError as exc:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"tac.cathedral_autopilot.three_metric_trichotomy unavailable: "
                f"{type(exc).__name__}: {exc} [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    try:
        hygiene = _compute_hygiene_ev(candidate)
        frontier = _compute_frontier_breaking_ev(candidate)
        highest = _compute_highest_ev_shortest_wall_clock_ev(candidate)
    except Exception as exc:  # noqa: BLE001 - defensive
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"3-metric trichotomy computation failed: {type(exc).__name__}: "
                f"{exc} [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    cid = candidate.get("candidate_id", "unknown_candidate")
    rationale = (
        f"META-orchestrator 3-metric trichotomy for {cid!r}: "
        f"hygiene_ev={hygiene:.4f}, frontier_breaking_ev={frontier:.6f}, "
        f"highest_ev_shortest_wall_clock_ev={highest:.6f}. "
        f"Operator-canonical routing default={HIGHEST_EV_SHORTEST_WALL_CLOCK}. "
        "Per Catalog #287/#341 + meta_orchestrator_three_metric_trichotomy_"
        "orthogonality_v1 canonical equation: observability-only [predicted]."
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "three_metric_trichotomy": {
            "hygiene_ev": hygiene,
            "frontier_breaking_ev": frontier,
            "highest_ev_shortest_wall_clock_ev": highest,
            "operator_canonical_metric": HIGHEST_EV_SHORTEST_WALL_CLOCK,
        },
    }


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_TIER",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
]
