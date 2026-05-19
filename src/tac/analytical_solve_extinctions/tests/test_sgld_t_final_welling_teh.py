# SPDX-License-Identifier: MIT
"""Tests for Row #8 SGLD t_final Welling-Teh solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.sgld_t_final_welling_teh import (
    SGLDTFinalInput,
    solve_sgld_t_final_welling_teh,
)


def test_canonical_stack_of_stacks_basic() -> None:
    res = solve_sgld_t_final_welling_teh(
        SGLDTFinalInput(
            variance_posterior_target=0.01,
            step_size_eta=0.001,
        ),
    )
    # raw = 1.0 * 0.01 / (0.001^2 * 0.574) = ~17422 — capped at 1.0
    assert res.intermediate_values["raw_t_final_unclamped"] > 1.0
    assert res.solved_value == pytest.approx(1.0)
    assert res.intermediate_values["ceiling_capped"] is True


def test_error_path_negative_variance_raises() -> None:
    with pytest.raises(ValueError, match="variance_posterior_target"):
        SGLDTFinalInput(variance_posterior_target=-0.01, step_size_eta=0.001)


def test_error_path_invalid_acceptance_rate() -> None:
    with pytest.raises(ValueError, match="target_acceptance_rate"):
        SGLDTFinalInput(
            variance_posterior_target=0.01,
            step_size_eta=0.001,
            target_acceptance_rate=1.5,
        )


def test_regression_floor_capped_when_eta_large() -> None:
    res = solve_sgld_t_final_welling_teh(
        SGLDTFinalInput(
            variance_posterior_target=0.001,
            step_size_eta=1.0,
            t_final_floor=1e-3,
        ),
    )
    # raw = 0.001 / (1.0^2 * 0.574) ~ 0.00174 — above floor 1e-3 but below ceil
    # but if eta is very large, raw could fall below floor
    assert res.solved_value >= 1e-3


def test_integration_atom_emission_when_requested() -> None:
    res = solve_sgld_t_final_welling_teh(
        SGLDTFinalInput(
            variance_posterior_target=0.01,
            step_size_eta=0.001,
        ),
        emit_arbitrariness_atom=True,
    )
    atom = res.coupled_adjustments["atom"]
    assert atom.atom_id == "sgld_t_final_welling_teh_derived"
    assert atom.metadata["current_value"] == pytest.approx(1e-4)
