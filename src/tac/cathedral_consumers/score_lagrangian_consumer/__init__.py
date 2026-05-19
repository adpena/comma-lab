# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.score_lagrangian`` analytical Lagrangian multipliers.

Per Catalog #335 + ``tac.cathedral.consumer_contract.CathedralConsumerContract``.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.score_lagrangian`` per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE 2026-05-19.

The canonical helper :mod:`tac.score_lagrangian` provides closed-form
Lagrange multipliers ``λ_seg``, ``λ_pose``, ``λ_rate`` for the contest
score ``S = sqrt(10 * d_pose) + 100 * d_seg + 25 * archive_bytes/N``.
At the PR106 frontier operating point (``d_pose=3.4e-5``), the marginal
ratio ``λ_pose / λ_seg = 2.71`` is INVERTED from the old 1.x operating
point's 77× SegNet > PoseNet rule. ~30 substrate trainers currently use
HAND-TUNED multipliers almost certainly NOT reflecting this
operating-point-dependent flip.

This consumer surfaces the canonical helper's operating-point-dependent
multiplier ratio as a non-promotable observability annotation. Per
CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287 the
rationale carries explicit ``[predicted]`` tag; ``promotable=False``
because the predicted ΔS ``[-0.012, -0.003]`` is a closed-form
derivation, not an empirical anchor.

Sister of:
- ``_example_consumer`` (canonical reference template)
- ``tac.cathedral_consumers.atom_consumer`` (sister observability consumer pattern)
- ``tac.cathedral_consumers.uncertainty_weighted_loss_consumer`` (sister
  TOP-3 arbitrariness-extinction consumer; complementary — TOP-1 analytical
  baseline + TOP-3 learned perturbations around)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "score_lagrangian_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    The canonical helper ``tac.score_lagrangian`` is closed-form
    (Boyd-Vandenberghe Ch.5 KKT); no posterior fit is needed. As
    substrates land paired contest-CPU+CUDA anchors that confirm the
    predicted ΔS [-0.012, -0.003] band, the corresponding canonical
    equation ``score_marginal_lagrange_multipliers_v1`` is updated via
    ``tac.canonical_equations.update_equation_with_empirical_anchor``
    (the registry's APPEND-ONLY semantics preserve the historical
    audit trail). NO-OP here by design.
    """
    _ = anchor  # explicit acknowledgment


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns zero-adjustment observability annotation citing the canonical
    helper's closed-form Lagrangian multipliers. No score adjustment —
    the predicted ΔS [-0.012, -0.003] is ``[predicted]`` axis per
    CLAUDE.md "Apples-to-apples evidence discipline" until paired empirical
    anchor confirms.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.score_lagrangian canonical helper available "
            "(closed-form λ_seg/λ_pose/λ_rate Lagrangian multipliers per "
            "Boyd-Vandenberghe Ch.5; predicted ΔS [-0.012, -0.003] at "
            "PR106 frontier operating point) [predicted]"
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
