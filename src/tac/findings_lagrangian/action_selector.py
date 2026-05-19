# SPDX-License-Identifier: MIT
"""Active-inference action selector via expected KL info gain (Lindley 1956 + Foster 2019).

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19) + Schmidhuber operating-within: the
mu_explore term IS the canonical bridge to active inference. The action
selector picks the next experiment whose hypothetical observation yields
maximum expected KL between posterior-after and posterior-before, PER DOLLAR.

The canonical formula (per Lindley 1956 + Foster et al 2019):

    action* = argmax_a E[KL(posterior_after(a) || posterior_before)] / cost(a)

For ≤20-anchor regime + diagonal Gaussian posterior, the closed-form
expected KL is exact (per Q2 binding decision).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from tac.findings_lagrangian.info_gain import expected_information_gain
from tac.findings_lagrangian.posterior import GaussianPosterior


__all__ = [
    "CandidateAction",
    "ActionRecommendation",
    "recommend_next_action_via_expected_information_gain",
    "ActionSelectorError",
]


class ActionSelectorError(ValueError):
    """Raised when action-selector inputs are malformed."""


@dataclass(frozen=True)
class CandidateAction:
    """One candidate experiment / action the selector can recommend.

    Attributes:
        action_id: operator-readable name for the action.
        equation_id: which canonical equation this action would update.
        predicted_hypothetical_residuals: predicted residuals if the action
            were executed (typically derived from sampling the posterior).
        cost_usd: dollar cost of executing the action ($0 for free probe).
        rationale: human-readable description of the action.
    """

    action_id: str
    equation_id: str
    predicted_hypothetical_residuals: tuple[float, ...]
    cost_usd: float
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ActionSelectorError("action_id must be non-empty string")
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise ActionSelectorError("equation_id must be non-empty string")
        if not isinstance(self.predicted_hypothetical_residuals, tuple):
            raise ActionSelectorError(
                "predicted_hypothetical_residuals must be tuple of floats"
            )
        for i, r in enumerate(self.predicted_hypothetical_residuals):
            if not isinstance(r, (int, float)):
                raise ActionSelectorError(
                    f"predicted_hypothetical_residuals[{i}] must be numeric"
                )
            if r != r:
                raise ActionSelectorError(
                    f"predicted_hypothetical_residuals[{i}] is NaN"
                )
        if not isinstance(self.cost_usd, (int, float)) or self.cost_usd < 0:
            raise ActionSelectorError(
                f"cost_usd={self.cost_usd} must be non-negative"
            )
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ActionSelectorError("rationale must be non-empty string")


@dataclass(frozen=True)
class ActionRecommendation:
    """Action-selector recommendation result.

    Per Catalog #305 observability surface: every action emits cite-chain to
    the equations it queried + the budget envelope it satisfied + the
    expected info gain per dollar.
    """

    ranked_actions: tuple[tuple[CandidateAction, float, float], ...]
    # Each entry: (action, expected_info_gain_nats, info_gain_per_dollar)
    budget_usd: float
    recommended_action_id: str | None  # None if no action fits budget
    recommendation_rationale: str
    computation_utc: str

    def __post_init__(self) -> None:
        if not isinstance(self.ranked_actions, tuple):
            raise ActionSelectorError("ranked_actions must be tuple")
        for i, entry in enumerate(self.ranked_actions):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ActionSelectorError(
                    f"ranked_actions[{i}] must be (CandidateAction, float, float)"
                )
            action, ig, igpd = entry
            if not isinstance(action, CandidateAction):
                raise ActionSelectorError(
                    f"ranked_actions[{i}][0] must be CandidateAction"
                )
            if not isinstance(ig, (int, float)):
                raise ActionSelectorError(
                    f"ranked_actions[{i}][1] must be numeric"
                )
            if not isinstance(igpd, (int, float)):
                raise ActionSelectorError(
                    f"ranked_actions[{i}][2] must be numeric"
                )
        if not isinstance(self.budget_usd, (int, float)) or self.budget_usd < 0:
            raise ActionSelectorError(
                f"budget_usd={self.budget_usd} must be non-negative"
            )
        if self.recommended_action_id is not None:
            if not isinstance(self.recommended_action_id, str):
                raise ActionSelectorError(
                    "recommended_action_id must be string or None"
                )

    def as_dict(self) -> dict[str, object]:
        """Serialize for JSON persistence."""
        return {
            "ranked_actions": [
                {
                    "action_id": a.action_id,
                    "equation_id": a.equation_id,
                    "cost_usd": float(a.cost_usd),
                    "expected_info_gain_nats": float(ig),
                    "info_gain_per_dollar": float(igpd),
                    "rationale": a.rationale,
                }
                for a, ig, igpd in self.ranked_actions
            ],
            "budget_usd": float(self.budget_usd),
            "recommended_action_id": self.recommended_action_id,
            "recommendation_rationale": self.recommendation_rationale,
            "computation_utc": self.computation_utc,
            "first_principles_citation": (
                "Lindley 1956 'On a measure of the information provided by an experiment' + "
                "Foster et al 2019 'Variational Bayesian Optimal Experimental Design' + "
                "Schmidhuber compression-as-intelligence + active inference precursor"
            ),
        }


def recommend_next_action_via_expected_information_gain(
    candidate_actions: Iterable[CandidateAction],
    *,
    posteriors_by_equation_id: Mapping[str, GaussianPosterior],
    budget_usd: float,
    sigma_obs: float = 1.0,
) -> ActionRecommendation:
    """Rank candidate actions by E[KL info gain] per dollar; recommend best within budget.

    Per Lindley 1956 + Foster et al 2019: the canonical decision rule for
    asymmetric-cost active learning is to maximize info-gain-per-cost. This
    helper computes the expected KL info gain for each candidate action
    (using the equation's current posterior + the action's predicted
    hypothetical residuals) + ranks descending by info_gain / cost.

    For zero-cost actions (free probes), info_gain_per_dollar is the raw
    info_gain_nats value (treating them as "infinite ROI" relative to paid
    actions); the recommended action is always the highest absolute info
    gain among free actions if any exist.

    Args:
        candidate_actions: actions the selector can choose from.
        posteriors_by_equation_id: dict mapping equation_id -> current GaussianPosterior.
        budget_usd: maximum dollar budget the recommended action may consume.
        sigma_obs: observation noise std-dev for likelihood.

    Returns:
        ActionRecommendation with ranked candidates + recommended action.
    """
    if budget_usd < 0:
        raise ActionSelectorError(f"budget_usd={budget_usd} must be non-negative")

    actions_list = list(candidate_actions)
    if not actions_list:
        return ActionRecommendation(
            ranked_actions=(),
            budget_usd=budget_usd,
            recommended_action_id=None,
            recommendation_rationale="no candidate actions provided",
            computation_utc=_utc_now_iso(),
        )

    scored: list[tuple[CandidateAction, float, float]] = []
    for action in actions_list:
        if action.equation_id not in posteriors_by_equation_id:
            # No posterior for this equation yet; can't compute info gain.
            scored.append((action, 0.0, 0.0))
            continue
        posterior = posteriors_by_equation_id[action.equation_id]
        info_gain = expected_information_gain(
            posterior,
            hypothetical_residuals=list(action.predicted_hypothetical_residuals),
            sigma_obs=sigma_obs,
        )
        # Free actions get the raw info_gain as "per dollar" (infinite ROI sentinel).
        if action.cost_usd <= 1e-9:
            info_gain_per_dollar = info_gain * 1e6  # massive bonus for free actions
        else:
            info_gain_per_dollar = info_gain / action.cost_usd
        scored.append((action, info_gain, info_gain_per_dollar))

    # Rank by info_gain_per_dollar descending.
    scored.sort(key=lambda x: x[2], reverse=True)

    # Recommendation = highest-IGPD action that fits within budget.
    recommended_id: str | None = None
    recommended_rationale = "no action fits the budget"
    for action, ig, igpd in scored:
        if action.cost_usd <= budget_usd:
            recommended_id = action.action_id
            recommended_rationale = (
                f"action_id={action.action_id} maximizes "
                f"E[KL info gain]/$ = {igpd:.4f} "
                f"(absolute info_gain={ig:.4f} nats; cost=${action.cost_usd:.2f}); "
                f"per {action.rationale}"
            )
            break

    return ActionRecommendation(
        ranked_actions=tuple(scored),
        budget_usd=budget_usd,
        recommended_action_id=recommended_id,
        recommendation_rationale=recommended_rationale,
        computation_utc=_utc_now_iso(),
    )


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with Z suffix."""
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
