# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.formula_extinctions`` canonical formulas.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.formula_extinctions`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.formula_extinctions`` is the canonical helper for formula-derived
constants (warmup_steps / validation_split / qint_max_grid /
inflate_device_pin_metadata / bayesian_aggregation_quorum /
frontier_threshold_from_state / early_stopping_patience). These constants
are PROOF-grade engineering outputs whose adoption already routes through
canonical training/dispatch paths; this consumer surfaces their availability
as a non-promotable observability annotation per Catalog #287 ``[predicted]``
discipline.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "formula_extinctions_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Canonical formula outputs are
    deterministic and do not require posterior update from anchors.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation citing canonical-formula
    surface. Promotion of formula-derived constants requires per-substrate
    integration; this consumer only confirms availability.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.formula_extinctions canonical formula-derived constants "
            "available (warmup_steps / validation_split / qint_max_grid / "
            "inflate_device_pin / bayesian_quorum / frontier_threshold / "
            "early_stopping) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
