# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.early_stopping`` SlopeWatcher.

Per Catalog #335 + ``tac.cathedral.consumer_contract.CathedralConsumerContract``.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.early_stopping`` per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE 2026-05-19.

The canonical helper :mod:`tac.early_stopping` provides
``SlopeWatcher`` per Prechelt 1998 PQ_α canonical slope-based early-
stopping discipline. Tracks val-score slope over a smoothing window;
halts when slope >= ``min_slope_improvement`` for ``patience_windows``
consecutive observations. Per-substrate ``--epochs`` defaults are wildly
arbitrary (1, 100, 200, 1000, 2000); the watcher closes this
arbitrariness at $0 (in fact NET-NEGATIVE — saves paid GPU wall-clock
past convergence-knee).

Predicted ΔS impact: ``[-0.006, -0.001]`` per slot 22 codex findings
review TOP-2 routing. Cost envelope: ``$0`` (NET-NEGATIVE).

Sister of:
- ``_example_consumer`` (canonical reference template)
- ``tac.cathedral_consumers.ema_decay_formula_consumer`` (sister TOP-4
  arbitrariness-extinction consumer; complementary — both close
  training-hyperparameter arbitrariness via closed-form formulas)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "early_stopping_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Substrate convergence-knee anchors (where SlopeWatcher fires) become
    empirical anchors for the canonical equation
    ``convergence_slope_early_stop_v1``. As more substrate trainers wire
    the watcher, the per-substrate optimal stop-epoch posterior refits via
    ``tac.canonical_equations.update_equation_with_empirical_anchor``.
    NO-OP here.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 + #6 — cathedral autopilot ranker + probe-disambiguator.

    Returns zero-adjustment observability annotation citing the canonical
    SlopeWatcher. Hook #6 probe-disambiguator: the watcher IS the
    canonical disambiguator between "continue training" vs "halt - slope
    flat". No score adjustment — the predicted ΔS [-0.006, -0.001] is
    ``[predicted]`` until empirical per-substrate convergence-knee anchors
    land.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.early_stopping.SlopeWatcher canonical helper available "
            "(Prechelt 1998 PQ_α slope-watcher; predicted ΔS [-0.006, -0.001] "
            "at $0 NET-NEGATIVE cost; halts past convergence-knee, saves paid "
            "GPU wall-clock; sister to tac.training.EMA.decay_from_total_steps "
            "TOP-4 closed-form) [predicted]"
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
