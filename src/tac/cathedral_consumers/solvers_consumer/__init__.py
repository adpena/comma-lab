# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.solvers`` canonical optimization algorithms.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for ``tac.solvers``
per wiring + integration audit 2026-05-19 (commit 3821cfb6b).

``tac.solvers`` exposes canonical algorithm choices (FISTA / Frank-Wolfe /
Sinkhorn / Riemannian-Newton-Stiefel / numba_jit_water_filling). Per the
grand council T3 deliberation finding #6 (2026-05-18): canonical-algorithm
adoption provides 1.5-2× velocity multiplier per appropriate candidate type.
This consumer surfaces availability + the canonical mapping
(``unified_action.choose_solver``) as a non-promotable observability
annotation per Catalog #287 ``[predicted]`` discipline.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "solvers_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Solver choice is a function of the
    candidate's track-kind; no anchor-driven update.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation citing canonical solver
    availability. Per-candidate solver selection routes through
    ``tac.unified_action.choose_solver`` (sister consumer).
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.solvers canonical optimization algorithm helpers available "
            "(FISTA / Frank-Wolfe / Sinkhorn / Riemannian-Newton-Stiefel / "
            "numba_jit_water_filling) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
