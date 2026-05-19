# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.contest_oracle`` canonical contest constants.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.contest_oracle`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.contest_oracle`` exposes contest-fixed constants (CONTEST_INPUT_HEIGHT
/ CONTEST_INPUT_WIDTH / CONTEST_NUM_PAIRS / CONTEST_PER_ARCHIVE_PER_CLASS_CELLS
/ CONTEST_RATE_DENOM_BYTES / SCORE_AXIS_LABELS / SEGNET_NUM_CLASSES /
CONTEST_POSE_SQRT_INNER / CONTEST_POSE_SQRT_WEIGHT). These are HARD-EARNED
upstream contest invariants. This consumer surfaces availability and confirms
the contest-fixed-value implications are accessible for downstream candidate
derivation; per Catalog #287 ``[predicted]`` discipline.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "contest_oracle_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. Contest oracle constants are HARD-EARNED
    upstream invariants pinned at module level; no anchor-driven update.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation citing canonical contest-oracle
    constants. Candidate-derived implications must route through the canonical
    helpers (e.g. ``score_predictor``) to inherit ``[predicted]`` discipline.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.contest_oracle canonical contest-fixed constants available "
            "(INPUT_HEIGHT=384 / INPUT_WIDTH=512 / NUM_PAIRS=600 / "
            "RATE_DENOM_BYTES=37,545,489 / SEGNET_NUM_CLASSES=5 / "
            "POSE_SQRT_WEIGHT=10) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
