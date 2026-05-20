# SPDX-License-Identifier: MIT
"""Risk-adjusted ranking consumer — SLOT MG-1 uncertainty quantification.

Per operator NON-NEGOTIABLE 2026-05-19 SLOT MG-1: build uncertainty
quantification into the candidate ranker; rank by ``posterior_mean -
lambda * posterior_std`` so race-mode (lambda=0) and conservative
(lambda=1) operating points are an operator-visible single knob.

This package is auto-discovered by ``tools/cathedral_autopilot_autonomous_loop.
discover_and_register_consumers`` per Catalog #335 canonical contract. The
3 module-level fields + 2 callable surfaces below satisfy the canonical
``CathedralConsumerContract`` (``tac.cathedral.consumer_contract``).

Public API:
    - ``RiskAdjustedRanker`` — the canonical risk-adjusted ranker class
    - ``RankedCandidate`` — typed output row with risk_adjusted_score audit
    - ``UncertaintyAwareCandidateRow`` — typed wrapper carrying the 4 new
      SLOT MG-1 fields (predicted_delta_uncertainty + n_anchors_consumed
      + evidence_grade + last_updated_utc)
    - ``predicted_delta_uncertainty_from_empirical_anchors`` — canonical
      NIG posterior helper consuming EmpiricalAnchor residuals
    - ``UncertaintyEstimate`` — typed return type of the helper
    - ``CANONICAL_EVIDENCE_GRADES`` — canonical evidence-grade vocabulary

Cross-references:
    - Catalog #349 STRICT preflight gate
      ``check_candidate_ranker_consumes_uncertainty`` (refuses ranker code
      paths that ignore predicted_delta_uncertainty when lambda > 0)
    - Catalog #335 canonical consumer contract
    - Catalog #323 canonical Provenance umbrella
    - Catalog #287 placeholder-rationale rejection
    - Catalog #125 6-hook wire-in declaration (see landing memo)

Wire-in hooks per Catalog #125:
    hook #1 sensitivity-map = N/A (consumer-side, not producer)
    hook #2 Pareto constraint = ACTIVE (lambda IS the risk-axis Pareto knob)
    hook #3 bit-allocator = N/A
    hook #4 cathedral autopilot dispatch = ACTIVE PRIMARY (SLOT MG-1)
    hook #5 continual-learning posterior = ACTIVE (uncertainty refits on
      new EmpiricalAnchor via predicted_delta_uncertainty_from_empirical_anchors)
    hook #6 probe-disambiguator = ACTIVE (the lambda value IS the
      operator-visible probe-vs-empirical disambiguator)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber

from .candidate_row import (
    CANONICAL_EVIDENCE_GRADES,
    UncertaintyAwareCandidateRow,
)
from .ranker import (
    RankedCandidate,
    RiskAdjustedRanker,
)
from .uncertainty import (
    UncertaintyEstimate,
    predicted_delta_uncertainty_from_empirical_anchors,
)


__all__ = (
    "CANONICAL_EVIDENCE_GRADES",
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "RankedCandidate",
    "RiskAdjustedRanker",
    "UncertaintyAwareCandidateRow",
    "UncertaintyEstimate",
    "consume_candidate",
    "predicted_delta_uncertainty_from_empirical_anchors",
    "update_from_anchor",
)


# Catalog #335 canonical contract — module-level metadata.
CONSUMER_NAME = "risk_adjusted_ranking_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Module-level singleton ranker — stateless by design, safe for shared use.
_RANKER = RiskAdjustedRanker()


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Refit the per-equation uncertainty when a new EmpiricalAnchor lands.
    This is a stateless consumer (uncertainty is recomputed per
    consume_candidate call from the canonical equation's anchor list),
    so the hook is a no-op here — the next consume_candidate call
    automatically picks up the new anchor via the
    predicted_delta_uncertainty_from_empirical_anchors helper.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the anchor's
    Provenance is honored downstream when the candidate's evidence_grade
    is computed (we do NOT silently promote diagnostic-grade anchors to
    contest-grade).
    """
    _ = anchor  # stateless consumer; helper re-reads anchors on demand


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns a no-op zero-adjustment contribution for the canonical
    auto-discovery cascade because this consumer's primary mode of
    operation is direct ``RiskAdjustedRanker().rank_candidates(...)``
    invocation by callers that want the risk-adjusted ordering as a
    typed list, not a per-candidate adjustment.

    The auto-discovery loop calls ``consume_candidate`` per-candidate
    expecting a bounded ``predicted_delta_adjustment`` float; the
    canonical risk-adjusted ranking happens at the LIST level (full
    sort by ``posterior_mean - lambda * posterior_std``), so the
    per-candidate adjustment is correctly 0.0.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the
    ``axis_tag="[predicted]"`` + ``promotable=False`` defaults reflect
    the non-promotable-by-construction discipline per Catalog #127.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "risk-adjusted ranking operates at LIST level via "
            "RiskAdjustedRanker.rank_candidates(); per-candidate "
            "adjustment is zero by design"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
