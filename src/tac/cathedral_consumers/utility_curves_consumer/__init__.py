# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.utility_curves`` per-byte utility fits.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.utility_curves`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.utility_curves`` exposes per-byte / per-pixel / per-tensor utility
curve fits (master-gradient utility / inverse-variance utility /
rate-distortion utility). Consumer surfaces availability as a non-promotable
observability annotation; per-candidate utility scoring requires explicit
master-gradient or per-byte sensitivity ledger input (see sister
``master_gradient_consumers`` wiring in cathedral autopilot ranker).
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "utility_curves_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.SENSITIVITY_MAP,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Utility curve fits are deterministic
    functions of input arrays; no posterior update from anchors required.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation. Per-candidate utility scoring
    requires explicit master-gradient sidecar at the candidate's archive
    sha (consumed via existing master_gradient_consumers cascade in cathedral
    autopilot ranker; this consumer documents availability for new candidates).
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.utility_curves canonical per-byte / per-pixel / per-tensor "
            "utility fit helpers available (master-gradient utility / "
            "inverse-variance utility / rate-distortion utility) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
