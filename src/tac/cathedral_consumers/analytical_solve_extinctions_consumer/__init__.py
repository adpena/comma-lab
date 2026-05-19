# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.analytical_solve_extinctions`` proofs.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.analytical_solve_extinctions`` per wiring + integration audit
2026-05-19 (commit 3821cfb6b).

``tac.analytical_solve_extinctions`` exposes analytical-solve PROOF-grade
extinctions (VRAM-aware batch size / RD-theoretic VQ codebook K /
optimal block-FP block size / MST frame ordering / ROC-optimal HIGH_PAIR_INVARIANT
threshold / coupling threshold statistical / SGLD t_final Welling-Teh).
These outputs are PROOF-grade engineering inputs that already route through
canonical training/dispatch paths via existing ranker cascades; consumer
surfaces availability as a non-promotable observability annotation per
Catalog #287.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "analytical_solve_extinctions_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Analytical-solve outputs are
    deterministic functions of inputs; no anchor-driven posterior update.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation. Analytical-solve PROOF-grade
    outputs already route through canonical training/dispatch paths via
    sister cascades (e.g. composition_alpha_v2 for VQ K-sweep, sister
    sister-#817 sidecars for SGLD t_final).
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.analytical_solve_extinctions canonical analytical-solve "
            "PROOF-grade extinctions available (VRAM-batch / RD-VQ-K / "
            "block-FP / MST-frame-ordering / ROC-HIGH_PAIR_INVARIANT / "
            "coupling-threshold / SGLD-Welling-Teh) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
