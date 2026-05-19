# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.unified_action`` GR-style action principle.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.unified_action`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.unified_action`` exposes the canonical Action / DualVariables /
OptimizerAnalyticalBoundaries / SolverChoice / SurfaceKind / TrackKind
surfaces with ``choose_solver`` / ``evaluate_with_admm`` /
``evaluate_with_magic_codec`` entry points. Per CLAUDE.md
"Meta-Lagrangian/Pareto solver" non-negotiable: the unified action IS the
canonical end-to-end candidate evaluation surface. Consumer surfaces
availability as a non-promotable observability annotation; per-candidate
unified-action evaluation requires explicit Action construction (the canonical
caller per the meta-Lagrangian wire-in plan).
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "unified_action_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PARETO_CONSTRAINT,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Unified action evaluation is
    deterministic given input dual variables + track-kind; no anchor-driven
    posterior update.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation. Per-candidate unified-action
    evaluation requires explicit Action construction routed through
    ``tac.unified_action.choose_solver`` per the meta-Lagrangian wire-in plan.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.unified_action canonical GR-style action principle surface "
            "available (Action / DualVariables / SolverChoice / TrackKind / "
            "choose_solver / evaluate_with_admm / evaluate_with_magic_codec) "
            "[predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
