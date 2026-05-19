# SPDX-License-Identifier: MIT
"""Tests for Row #9 bootstrap-CI Rashomon K solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.bootstrap_ci_rashomon_K import (
    RashomonKInput,
    solve_bootstrap_ci_rashomon_K,
)


def test_canonical_default_basic() -> None:
    res = solve_bootstrap_ci_rashomon_K(
        RashomonKInput(effective_dimensionality_d=2.0),
    )
    assert isinstance(res.solved_value, int)
    # raw = 2.0 * log(20) / 0.01 = ~599 — way above K=64 ceiling
    assert res.solved_value == 64
    assert res.intermediate_values["ceiling_capped"] is True


def test_error_path_invalid_delta_raises() -> None:
    with pytest.raises(ValueError, match="ci_failure_probability_delta"):
        RashomonKInput(effective_dimensionality_d=2.0, ci_failure_probability_delta=2.0)


def test_error_path_invalid_epsilon_raises() -> None:
    with pytest.raises(ValueError, match="ci_target_half_width_epsilon"):
        RashomonKInput(effective_dimensionality_d=2.0, ci_target_half_width_epsilon=0.0)


def test_regression_tight_ci_increases_K() -> None:
    res_loose = solve_bootstrap_ci_rashomon_K(
        RashomonKInput(
            effective_dimensionality_d=1.0,
            ci_target_half_width_epsilon=0.5,
        ),
    )
    res_tight = solve_bootstrap_ci_rashomon_K(
        RashomonKInput(
            effective_dimensionality_d=1.0,
            ci_target_half_width_epsilon=0.1,
        ),
    )
    assert res_tight.solved_value >= res_loose.solved_value


def test_integration_atom_emission_when_requested() -> None:
    res = solve_bootstrap_ci_rashomon_K(
        RashomonKInput(effective_dimensionality_d=2.0),
        emit_arbitrariness_atom=True,
    )
    atom = res.coupled_adjustments["atom"]
    assert atom.atom_id == "rashomon_K_bootstrap_ci_derived"
    assert atom.metadata["current_value"] == 8
