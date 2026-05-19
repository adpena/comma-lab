# SPDX-License-Identifier: MIT
"""Tests for Row #4 optimal block-FP block_size solver."""

from __future__ import annotations

import math

import pytest

from tac.analytical_solve_extinctions.optimal_block_fp_block_size import (
    BlockFPSolverInput,
    solve_optimal_block_fp_block_size,
)


def test_balle_canonical_layer_basic() -> None:
    res = solve_optimal_block_fp_block_size(
        BlockFPSolverInput(
            num_elements_N=10_000,
            header_bytes_per_block=4,
            quant_loss_constant=1.0,
        ),
    )
    assert isinstance(res.solved_value, int)
    assert res.solved_value >= 4
    assert res.solved_value <= 4096
    # Power-of-two rounding produces power-of-two output by default
    assert (res.solved_value & (res.solved_value - 1)) == 0
    # Continuous optimum sqrt(10000*4/1) = ~200
    raw = res.intermediate_values["raw_continuous_optimum"]
    assert raw == pytest.approx(math.sqrt(40_000), rel=1e-6)


def test_error_path_zero_quant_loss_raises() -> None:
    with pytest.raises(ValueError, match="quant_loss_constant"):
        BlockFPSolverInput(
            num_elements_N=10_000,
            header_bytes_per_block=4,
            quant_loss_constant=0.0,
        )


def test_regression_no_power_of_two_round_when_disabled() -> None:
    res = solve_optimal_block_fp_block_size(
        BlockFPSolverInput(
            num_elements_N=10_000,
            header_bytes_per_block=4,
            quant_loss_constant=1.0,
            round_to_power_of_two=False,
        ),
    )
    # When rounding disabled, candidate is int(round(raw)) = 200
    assert res.solved_value == 200


def test_regression_clamp_to_max_when_optimal_above_ceiling() -> None:
    res = solve_optimal_block_fp_block_size(
        BlockFPSolverInput(
            num_elements_N=1_000_000_000,
            header_bytes_per_block=100,
            quant_loss_constant=1e-9,
            max_block_size=1024,
        ),
    )
    assert res.solved_value == 1024


def test_integration_atom_emission_when_requested() -> None:
    res = solve_optimal_block_fp_block_size(
        BlockFPSolverInput(
            num_elements_N=10_000,
            header_bytes_per_block=4,
            quant_loss_constant=1.0,
        ),
        emit_arbitrariness_atom=True,
        layer_name="balle_hyperprior_layer_3",
    )
    atom = res.coupled_adjustments["atom"]
    assert "block_fp_block_size_solved_for_balle_hyperprior_layer_3" == atom.atom_id
    assert atom.cost_envelope_usd == pytest.approx(0.0)
