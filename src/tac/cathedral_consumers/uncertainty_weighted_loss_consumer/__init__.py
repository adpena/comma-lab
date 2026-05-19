# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.uncertainty_weighted_loss`` (Kendall + focal weighting).

Per Catalog #335 + ``tac.cathedral.consumer_contract.CathedralConsumerContract``.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.uncertainty_weighted_loss`` per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE
2026-05-19.

The canonical helper :mod:`tac.uncertainty_weighted_loss` provides:

- ``UncertaintyWeightedScoreLoss`` — Kendall et al 2018 multi-task
  uncertainty weighting (arxiv:1705.07115). Learns log-σ per axis +
  weights ``L_axis / (2σ²) + log(σ)`` per Bayesian-deep-learning
  closed-form.
- ``per_pair_focal_weights`` / ``apply_focal_per_pair_reweighting`` —
  Lin et al 2017 focal loss (arxiv:1708.02002). Reshapes the per-pair
  loss distribution to concentrate gradient on hard pairs.

Combined predicted ΔS ``[-0.012, -0.002]`` per slot 22 codex findings
review TOP-3 + TOP-6 routing. Cost envelope: ``$0`` (training-time only;
3 scalar log-σ params).

Sister of:
- ``_example_consumer`` (canonical reference template)
- ``tac.cathedral_consumers.score_lagrangian_consumer`` (sister TOP-1
  arbitrariness-extinction consumer; complementary — TOP-1 = analytical
  baseline closed-form; TOP-3/TOP-6 = learned perturbations around the
  baseline)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "uncertainty_weighted_loss_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    The 3 log-σ params learn jointly with the substrate model during
    training; the σ values at convergence are the per-substrate
    empirical anchors for the canonical equation
    ``per_pair_loss_weighting_optimal_v1``. As substrates land paired
    contest-CPU+CUDA anchors confirming the predicted ΔS [-0.012, -0.002]
    band, the equation is refit via
    ``tac.canonical_equations.update_equation_with_empirical_anchor``.
    NO-OP here (canonical pattern; posterior fit happens in the equation
    registry, not in the consumer).
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 + #2 — cathedral autopilot ranker + Pareto.

    Returns zero-adjustment observability annotation citing the canonical
    Kendall + focal helpers. Hook #2 Pareto: the Kendall learned weights
    implicitly trace a Pareto-optimal point on seg×pose×rate frontier via
    the dual Lagrangian.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.uncertainty_weighted_loss canonical helper available "
            "(Kendall et al 2018 multi-task uncertainty weighting + Lin et al "
            "2017 focal per-pair weighting; predicted ΔS [-0.012, -0.002] at "
            "$0 cost; complementary to tac.score_lagrangian analytical baseline) "
            "[predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "update_from_anchor",
    "consume_candidate",
]
