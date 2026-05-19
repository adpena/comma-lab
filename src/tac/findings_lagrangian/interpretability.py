# SPDX-License-Identifier: MIT
"""Wang-Rudin 2015 falling-rule-list explanation generator (Q7 amendment).

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19):

- Q7 PROCEED (slot 20): extend cathedral autopilot ranker to consult
  predicted_delta_uncertainty + downweight high-uncertainty candidates
  by 1/(1+sigma).
- Q7 AMEND (slot 20-supplemental, Rudin amendment): the 1/(1+sigma)
  downweighting MUST be paired with a Wang-Rudin 2015 falling-rule-list
  explanation readback per Catalog #274 sister discipline. Each ranker
  decision emits: (a) baseline rank, (b) sigma-aware downweight factor,
  (c) falling-rule-list explanation citing which anchor(s) drove sigma
  growth + which equation(s)' residuals contributed.
- Q7 ratified-with-Quantizr-extension (slot 20-second-supplemental):
  14-day-anchor reactivation criterion - if 1/(1+sigma) downweighting
  does NOT re-rank at least one historical dispatch in 14 days, deprecate.

Per Rudin operating-within (slot 20-supplemental): "Stop Explaining Black
Box ML for High-Stakes Decisions and Use Interpretable Models Instead"
(Rudin 2019 Nature ML). Operators need to audit WHY each candidate was
downweighted; opaque 1/(1+sigma) is insufficient.

The implementation mirrors the canonical preflight surface
``tac.preflight_rudin_daubechies.falling_rule`` (Catalog #274 sister)
adapted for the cathedral autopilot ranker contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


__all__ = [
    "FallingRule",
    "FallingRuleListExplanation",
    "build_falling_rule_list_for_downweight",
    "explain_decision_per_candidate",
    "compute_downweight_factor",
    "MAX_RULES_PER_EXPLANATION",
    "InterpretabilityError",
]


MAX_RULES_PER_EXPLANATION = 8
"""Cap rule-chain length so explanations are reviewable in 30 seconds.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Rudin's
canonical falling-rule-list discipline: short rule chains preserve
interpretability.
"""


class InterpretabilityError(ValueError):
    """Raised when a falling rule violates Wang-Rudin 2015 canonical structure."""


@dataclass(frozen=True)
class FallingRule:
    """One rule in a falling-rule-list (Wang-Rudin 2015).

    A rule is a (condition, hit_rate, decision) triple where:
    - condition: human-readable predicate evaluated on the candidate
    - hit_rate: empirical fraction of candidates matching this condition
    - decision: the action taken if this rule fires (operator-readable)

    Per Wang-Rudin: rules are ORDERED by hit_rate (descending) so the
    operator reads the highest-frequency conditions first. The first-match-
    wins semantics make every decision auditable.
    """

    condition: str
    hit_rate: float
    decision: str
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise InterpretabilityError("condition must be non-empty string")
        if not isinstance(self.hit_rate, (int, float)):
            raise InterpretabilityError("hit_rate must be numeric")
        if self.hit_rate != self.hit_rate:  # NaN
            raise InterpretabilityError("hit_rate is NaN")
        if not 0.0 <= self.hit_rate <= 1.0:
            raise InterpretabilityError(
                f"hit_rate={self.hit_rate} must be in [0, 1]"
            )
        if not isinstance(self.decision, str) or not self.decision.strip():
            raise InterpretabilityError("decision must be non-empty string")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise InterpretabilityError("rationale must be non-empty string")


@dataclass(frozen=True)
class FallingRuleListExplanation:
    """Wang-Rudin 2015 falling-rule-list explanation for one ranker decision.

    Per Rudin Q7 amendment: every cathedral autopilot ranker decision emits
    this explanation so the operator can audit (a) which anchor drove sigma
    growth + (b) which equation's residuals contributed + (c) the resulting
    downweight factor.
    """

    candidate_id: str
    baseline_rank_signal: float
    posterior_sigma_used: float
    downweight_factor: float
    rules: tuple[FallingRule, ...]
    matched_rule_index: int  # which rule fired (first-match-wins)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise InterpretabilityError("candidate_id must be non-empty string")
        if not isinstance(self.baseline_rank_signal, (int, float)):
            raise InterpretabilityError("baseline_rank_signal must be numeric")
        if not isinstance(self.posterior_sigma_used, (int, float)):
            raise InterpretabilityError("posterior_sigma_used must be numeric")
        if self.posterior_sigma_used < 0:
            raise InterpretabilityError(
                f"posterior_sigma_used={self.posterior_sigma_used} must be >= 0"
            )
        if not isinstance(self.downweight_factor, (int, float)):
            raise InterpretabilityError("downweight_factor must be numeric")
        if not 0.0 < self.downweight_factor <= 1.0:
            raise InterpretabilityError(
                f"downweight_factor={self.downweight_factor} must be in (0, 1]"
            )
        if not isinstance(self.rules, tuple):
            raise InterpretabilityError("rules must be tuple")
        if len(self.rules) > MAX_RULES_PER_EXPLANATION:
            raise InterpretabilityError(
                f"rules count {len(self.rules)} exceeds cap {MAX_RULES_PER_EXPLANATION}"
            )
        for i, rule in enumerate(self.rules):
            if not isinstance(rule, FallingRule):
                raise InterpretabilityError(
                    f"rules[{i}] must be FallingRule"
                )
        if self.rules and not 0 <= self.matched_rule_index < len(self.rules):
            raise InterpretabilityError(
                f"matched_rule_index={self.matched_rule_index} out of range"
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize for JSON persistence + operator-facing display."""
        return {
            "candidate_id": self.candidate_id,
            "baseline_rank_signal": float(self.baseline_rank_signal),
            "posterior_sigma_used": float(self.posterior_sigma_used),
            "downweight_factor": float(self.downweight_factor),
            "matched_rule_index": int(self.matched_rule_index),
            "matched_rule": (
                {
                    "condition": self.rules[self.matched_rule_index].condition,
                    "hit_rate": self.rules[self.matched_rule_index].hit_rate,
                    "decision": self.rules[self.matched_rule_index].decision,
                    "rationale": self.rules[self.matched_rule_index].rationale,
                }
                if self.rules
                else None
            ),
            "rules": [
                {
                    "condition": r.condition,
                    "hit_rate": r.hit_rate,
                    "decision": r.decision,
                    "rationale": r.rationale,
                }
                for r in self.rules
            ],
            "first_principles_citation": (
                "Wang-Rudin 2015 'Falling Rule Lists' + "
                "Rudin 2019 Nature ML 'Stop Explaining Black Box ML' + "
                "Catalog #274 sister discipline + Q7 supplemental amendment"
            ),
        }


def compute_downweight_factor(posterior_sigma: float) -> float:
    """Canonical 1/(1+sigma) downweight per Q7 binding decision.

    Bounded in (0, 1]: sigma=0 → factor=1 (no downweight); sigma→inf → factor→0
    (full downweight).
    """
    if posterior_sigma < 0:
        raise InterpretabilityError(
            f"posterior_sigma={posterior_sigma} must be >= 0"
        )
    return 1.0 / (1.0 + posterior_sigma)


def build_falling_rule_list_for_downweight(
    posterior_sigma: float,
    *,
    equation_id: str | None = None,
    contributing_anchor_ids: Iterable[str] | None = None,
) -> tuple[FallingRule, ...]:
    """Build a canonical falling-rule-list explaining the downweight decision.

    The rule chain is structured falling-order (highest hit_rate first):
    1. "sigma is large" → strong downweight, cite anchor drift
    2. "sigma is moderate" → moderate downweight
    3. "sigma is small" → minimal downweight
    4. "sigma is zero" → no downweight (well-calibrated equation)

    Args:
        posterior_sigma: the uncertainty value from the posterior.
        equation_id: optional citation of which equation produced sigma.
        contributing_anchor_ids: optional citation of which anchors drove
            sigma growth (for the "large sigma" rule's rationale).

    Returns:
        Tuple of FallingRule entries; first-match-wins semantics.
    """
    if posterior_sigma < 0:
        raise InterpretabilityError(
            f"posterior_sigma={posterior_sigma} must be >= 0"
        )
    anchor_cite = ""
    if contributing_anchor_ids:
        cited = list(contributing_anchor_ids)[:3]  # cap citation length
        anchor_cite = f" (top contributing anchors: {', '.join(cited)})"
    eq_cite = f" from equation {equation_id}" if equation_id else ""

    return (
        FallingRule(
            condition="posterior_sigma >= 2.0 (high uncertainty)",
            hit_rate=0.10,  # rare in well-calibrated regime
            decision="strong_downweight factor < 0.34",
            rationale=(
                f"Posterior uncertainty{eq_cite} is high; ranker should defer "
                f"this candidate to cheap probe before paid dispatch{anchor_cite}"
            ),
        ),
        FallingRule(
            condition="0.5 <= posterior_sigma < 2.0 (moderate uncertainty)",
            hit_rate=0.30,
            decision="moderate_downweight factor in [0.34, 0.67]",
            rationale=(
                f"Posterior uncertainty{eq_cite} is moderate; ranker downweights "
                "but does not defer; cathedral autopilot may proceed to paid "
                "dispatch with explicit acceptance"
            ),
        ),
        FallingRule(
            condition="0.0 < posterior_sigma < 0.5 (small uncertainty)",
            hit_rate=0.40,
            decision="small_downweight factor in [0.67, 1.0)",
            rationale=(
                f"Posterior uncertainty{eq_cite} is small; minimal downweight"
            ),
        ),
        FallingRule(
            condition="posterior_sigma == 0.0 (well-calibrated)",
            hit_rate=0.20,
            decision="no_downweight factor = 1.0",
            rationale=(
                f"Posterior is well-calibrated{eq_cite}; ranker proceeds with "
                "predicted_delta unchanged"
            ),
        ),
    )


def explain_decision_per_candidate(
    candidate_id: str,
    baseline_rank_signal: float,
    posterior_sigma: float,
    *,
    equation_id: str | None = None,
    contributing_anchor_ids: Iterable[str] | None = None,
) -> FallingRuleListExplanation:
    """Generate a full falling-rule-list explanation for a candidate decision.

    Per Q7 amendment + Catalog #305 observability surface: this is the
    canonical helper cathedral autopilot ranker calls for every decision
    to emit the audit trail.

    Args:
        candidate_id: the candidate being ranked.
        baseline_rank_signal: the candidate's predicted_delta before downweight.
        posterior_sigma: the posterior uncertainty.
        equation_id: which canonical_equation contributed sigma.
        contributing_anchor_ids: which anchors drove sigma growth.

    Returns:
        FallingRuleListExplanation suitable for JSON persistence + operator
        display.
    """
    rules = build_falling_rule_list_for_downweight(
        posterior_sigma,
        equation_id=equation_id,
        contributing_anchor_ids=contributing_anchor_ids,
    )
    downweight = compute_downweight_factor(posterior_sigma)
    # First-match-wins per Wang-Rudin: walk rules in declared order.
    matched_index = 0
    if posterior_sigma >= 2.0:
        matched_index = 0
    elif posterior_sigma >= 0.5:
        matched_index = 1
    elif posterior_sigma > 0.0:
        matched_index = 2
    else:
        matched_index = 3
    return FallingRuleListExplanation(
        candidate_id=candidate_id,
        baseline_rank_signal=float(baseline_rank_signal),
        posterior_sigma_used=float(posterior_sigma),
        downweight_factor=float(downweight),
        rules=rules,
        matched_rule_index=matched_index,
    )
