# SPDX-License-Identifier: MIT
"""Compatibility alias for the canonical Time-Traveler substrate.

The canonical implementation is ``tac.substrates.time_traveler_l5_autonomy``.
This package exists only to preserve older import paths without creating a
second archive grammar or duplicate implementation surface.
"""

from tac.substrates.time_traveler_l5_autonomy import *  # noqa: F403
from tac.substrates.time_traveler_l5_autonomy import __all__ as _CANONICAL_ALL

TIME_TRAVELER_METADATA = {
    "canonical_package": "tac.substrates.time_traveler_l5_autonomy",
    "compatibility_alias": True,
    "research_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

__all__ = [*_CANONICAL_ALL, "TIME_TRAVELER_METADATA"]
