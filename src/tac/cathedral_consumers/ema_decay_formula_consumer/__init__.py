# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.training.EMA.decay_from_total_steps`` (TOP-4 EMA formula).

Per Catalog #335 + ``tac.cathedral.consumer_contract.CathedralConsumerContract``.
Wires the orphan-signal-at-cathedral-autopilot bug class for the canonical
TOP-4 EMA-decay-from-total-steps formula per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE
2026-05-19.

The canonical formula ``decay = 1 - 1/(target_window_fraction * total_steps)``
lives at ``tac.training.EMA.decay_from_total_steps`` (already landed; see
``src/tac/training.py:538``). Per Polyak & Juditsky 1992 + Tarvainen-Valpola
2017 + CLAUDE.md "EMA -- NON-NEGOTIABLE, HIGHEST EMPHASIS" the canonical
default ``target_window_fraction=0.2`` (Polyak's "last 20 percent" averaging
window) recovers the Quantizr PR101 empirical anchor ``decay=0.997`` at
``total_steps=1666``.

Predicted ΔS impact: ``[-0.005, -0.001]`` per slot 22 codex findings
review TOP-4 routing. Cost envelope: ``$0`` (training-time only).

This consumer surfaces the formula's availability as a non-promotable
observability annotation; the actual wire-in is in
``tac.training.EMA(decay="auto", total_steps=int)``.

Sister of:
- ``_example_consumer`` (canonical reference template)
- ``tac.cathedral_consumers.early_stopping_consumer`` (sister TOP-2
  arbitrariness-extinction consumer; complementary — both close
  training-hyperparameter arbitrariness via closed-form formulas)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "ema_decay_formula_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-substrate convergence anchors (substrate trained with
    ``decay="auto"`` matched to its ``total_steps`` budget) feed the
    canonical equation ``ema_decay_substrate_stage_aware_v1``. As more
    substrates wire ``decay="auto"``, the per-substrate-stage-aware
    posterior refits via
    ``tac.canonical_equations.update_equation_with_empirical_anchor``.
    NO-OP here.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns zero-adjustment observability annotation citing the canonical
    EMA-decay-from-total-steps formula. No score adjustment — the
    predicted ΔS [-0.005, -0.001] is ``[predicted]`` until per-substrate
    empirical anchors confirm the formula's per-substrate optimal decay.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.training.EMA.decay_from_total_steps canonical helper available "
            "(Polyak-Juditsky 1992 + Tarvainen-Valpola 2017; decay = 1 - 1/"
            "(target_window_fraction * total_steps); predicted ΔS [-0.005, "
            "-0.001] at $0 cost; closes the 0.997-universal-hardcoded arbitrariness "
            "per CLAUDE.md EMA non-negotiable) [predicted]"
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
