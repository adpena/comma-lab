# SPDX-License-Identifier: MIT
"""Tests for Row #7 statistical coupling-threshold solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.coupling_threshold_statistical_derivation import (
    CouplingThresholdInput,
    solve_coupling_threshold_statistical,
)


def test_3x3_symmetric_basic() -> None:
    matrix = [
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]
    res = solve_coupling_threshold_statistical(
        CouplingThresholdInput(pairwise_inner_products=matrix),
    )
    # off-diagonal: [0.2, 0.3, 0.2, 0.4, 0.3, 0.4]
    # mean = 0.3; std ~ 0.0816
    assert res.intermediate_values["mean"] == pytest.approx(0.3, rel=1e-4)
    assert res.solved_value > 0.3  # mean + 1*std > mean


def test_error_path_too_small_matrix_raises() -> None:
    with pytest.raises(ValueError, match="off-diagonal"):
        CouplingThresholdInput(pairwise_inner_products=[[1.0]])


def test_error_path_row_size_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        CouplingThresholdInput(
            pairwise_inner_products=[[1.0, 0.1], [0.1, 1.0, 0.5]],
        )


def test_regression_k_zero_returns_pure_mean() -> None:
    matrix = [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]
    res = solve_coupling_threshold_statistical(
        CouplingThresholdInput(pairwise_inner_products=matrix, k_std_multiplier=0.0),
    )
    assert res.solved_value == pytest.approx(0.5)


def test_integration_atom_emission_when_requested() -> None:
    matrix = [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]]
    res = solve_coupling_threshold_statistical(
        CouplingThresholdInput(pairwise_inner_products=matrix),
        emit_arbitrariness_atom=True,
    )
    atom = res.coupled_adjustments["atom"]
    assert atom.atom_id == "coupling_threshold_statistical_derived"
    assert atom.metadata["current_value"] == 0.5
