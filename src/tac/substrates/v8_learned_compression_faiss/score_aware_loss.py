# SPDX-License-Identifier: MIT
"""V8 score-aware training authority contract.

The local V8 smoke does not load scorers. This module records the canonical
hooks that a future real trainer must use, and gives tests a single place to
assert that local smoke remains non-promotional.
"""

from __future__ import annotations

from typing import Any


def build_score_aware_roundtrip_contract() -> dict[str, Any]:
    """Return the required future scorer/eval-roundtrip hook contract."""

    return {
        "score_aware_helper": "tac.substrates.score_aware_common.score_pair_components_dispatch",
        "eval_roundtrip": "tac.differentiable_eval_roundtrip",
        "auth_eval_gate": "tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call",
        "local_smoke_loads_scorers": False,
        "real_scorer_training_complete": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "remaining_blockers": [
            "real_contest_video_scorer_training_not_run",
            "exact_cuda_auth_eval_missing",
            "catalog_324_tier_c_validation_missing",
        ],
    }


__all__ = ["build_score_aware_roundtrip_contract"]
