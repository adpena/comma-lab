# SPDX-License-Identifier: MIT
"""Tests for Row #1 VRAM-aware batch_size solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.vram_aware_batch_size import (
    AnalyticalSolveResult,
    BatchSizeSolverInput,
    solve_vram_aware_batch_size,
)


def test_modal_t4_canonical_substrate_batch_solve_basic() -> None:
    res = solve_vram_aware_batch_size(
        BatchSizeSolverInput(
            vram_budget_gb=14.5,
            model_size_gb=1.2,
            activation_overhead_gb=0.5,
            per_sample_activation_gb=0.4,
            base_lr_at_reference_batch=1e-4,
            reference_batch_size=4,
        ),
    )
    assert isinstance(res, AnalyticalSolveResult)
    assert isinstance(res.solved_value, int)
    assert res.solved_value > 0
    # (14.5 - 1.2 - 0.5) / 0.4 = 32 exactly
    assert res.solved_value == 32
    # linear LR scaling: 32/4 = 8x
    assert res.coupled_adjustments["lr_scaling_factor"] == pytest.approx(8.0)


def test_error_path_negative_vram_remaining_raises() -> None:
    with pytest.raises(ValueError, match="VRAM budget"):
        solve_vram_aware_batch_size(
            BatchSizeSolverInput(
                vram_budget_gb=2.0,
                model_size_gb=3.0,
                activation_overhead_gb=0.5,
                per_sample_activation_gb=0.4,
                base_lr_at_reference_batch=1e-4,
            ),
        )


def test_error_path_per_sample_too_large_raises() -> None:
    with pytest.raises(ValueError, match="cannot fit a single sample"):
        solve_vram_aware_batch_size(
            BatchSizeSolverInput(
                vram_budget_gb=4.0,
                model_size_gb=1.0,
                activation_overhead_gb=0.5,
                per_sample_activation_gb=10.0,
                base_lr_at_reference_batch=1e-4,
            ),
        )


def test_regression_min_batch_clamp_active() -> None:
    res = solve_vram_aware_batch_size(
        BatchSizeSolverInput(
            vram_budget_gb=14.5,
            model_size_gb=1.2,
            activation_overhead_gb=0.5,
            per_sample_activation_gb=0.4,
            base_lr_at_reference_batch=1e-4,
            reference_batch_size=4,
            min_batch_size=64,
            max_batch_size=256,
        ),
    )
    assert res.solved_value == 64
    assert res.intermediate_values["floor_capped"] is True


def test_integration_atom_emission_when_requested() -> None:
    res = solve_vram_aware_batch_size(
        BatchSizeSolverInput(
            vram_budget_gb=14.5,
            model_size_gb=1.2,
            activation_overhead_gb=0.5,
            per_sample_activation_gb=0.4,
            base_lr_at_reference_batch=1e-4,
            reference_batch_size=4,
        ),
        emit_arbitrariness_atom=True,
        substrate_id="c6_e4_mdl_ibps",
    )
    atom = res.coupled_adjustments["atom"]
    assert atom.atom_id == "batch_size_solved_for_c6_e4_mdl_ibps"
    assert atom.cost_envelope_usd == pytest.approx(2.0)
    assert atom.resolution_path.value == "analytical_solve"


def test_input_validation_invalid_reference_batch_size() -> None:
    with pytest.raises(ValueError, match="reference_batch_size"):
        BatchSizeSolverInput(
            vram_budget_gb=14.5,
            model_size_gb=1.2,
            activation_overhead_gb=0.5,
            per_sample_activation_gb=0.4,
            base_lr_at_reference_batch=1e-4,
            reference_batch_size=0,
        )
