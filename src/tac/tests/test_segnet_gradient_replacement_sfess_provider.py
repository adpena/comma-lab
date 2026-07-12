# SPDX-License-Identifier: MIT
"""Boundary-provider registration tests for cached terminal-objective SFESS."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tac.boundary_math.segnet_gradient_replacement import (
    SFESS_CACHED_K_SUBSET_PROVIDER_MODE,
    TerminalObjectiveProviderMode,
)


def test_sfess_terminal_provider_mode_is_non_costate_and_fail_closed() -> None:
    mode = SFESS_CACHED_K_SUBSET_PROVIDER_MODE

    assert mode == TerminalObjectiveProviderMode()
    assert mode.mode == "sfess_cached_k_subset"
    assert mode.objective_surface == "terminal_exact_through_r"
    assert mode.produces_costate is False
    assert mode.research_only is True
    assert mode.score_claim is False
    assert mode.promotion_eligible is False
    assert mode.cache_failure_action == "refuse"
    assert mode.resolve_live_gradient() == "full_teacher"


def test_sfess_terminal_provider_mode_pins_every_live_admission_guard() -> None:
    mode = SFESS_CACHED_K_SUBSET_PROVIDER_MODE

    assert mode.requires_rederived_objective_context_fingerprint is True
    assert mode.requires_rederived_frame_or_state_fingerprint is True
    assert mode.requires_rederived_provider_fingerprint is True
    assert mode.max_evidence_age_queries == 0
    assert mode.requires_finite_objective is True
    assert mode.live_admission_requires_real_teacher_regret_gate is True
    with pytest.raises(FrozenInstanceError):
        mode.produces_costate = True  # type: ignore[misc]
