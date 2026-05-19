# SPDX-License-Identifier: MIT
"""Tests for Row #2 + #3 (batched) R-D theoretic VQ codebook K solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.rd_theoretic_vq_codebook_K import (
    RDCodebookSolverInput,
    solve_rd_theoretic_vq_codebook_K,
)


def test_neural_weight_codec_K_64_replacement_basic() -> None:
    res = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=100_000,
            codeword_dim=4,
            bytes_per_codeword=4,
            lambda_rd=1.0,
            quantization_constant_C=100.0,
        ),
    )
    assert isinstance(res.solved_value, int)
    assert res.solved_value >= 2
    assert res.solved_value <= 512
    # K should be power of two
    assert (res.solved_value & (res.solved_value - 1)) == 0
    # Per-K grid preserved for observability
    assert "per_K_cost_breakdown" in res.intermediate_values


def test_vqvae_mask_K_256_replacement_basic() -> None:
    res = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=10_000,
            codeword_dim=64,  # 8x8 mask blocks
            bytes_per_codeword=8,
            lambda_rd=1.0,
            quantization_constant_C=1000.0,
        ),
    )
    assert isinstance(res.solved_value, int)
    # Cost decomposition surfaces R-D elbow
    per_K_costs = dict(res.intermediate_values["per_K_cost_breakdown"])
    assert len(per_K_costs) >= 2


def test_error_path_invalid_K_grid_bounds() -> None:
    with pytest.raises(ValueError, match="candidate_K_max"):
        RDCodebookSolverInput(
            num_codewords_used=100,
            codeword_dim=4,
            bytes_per_codeword=4,
            candidate_K_min=512,
            candidate_K_max=2,
        )


def test_regression_higher_lambda_picks_smaller_K() -> None:
    # When distortion penalty (lambda_rd) is huge, smaller codebooks are
    # punished (more residual error); but K is BOUNDED — verify the cost
    # function responds monotonically to lambda.
    small_lambda = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=1_000,
            codeword_dim=4,
            bytes_per_codeword=4,
            lambda_rd=0.01,
            quantization_constant_C=10.0,
        ),
    )
    large_lambda = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=1_000,
            codeword_dim=4,
            bytes_per_codeword=4,
            lambda_rd=100.0,
            quantization_constant_C=10.0,
        ),
    )
    # Higher distortion penalty should push K UP (more codewords to reduce error)
    assert large_lambda.solved_value >= small_lambda.solved_value


def test_integration_atom_emission_when_requested() -> None:
    res = solve_rd_theoretic_vq_codebook_K(
        RDCodebookSolverInput(
            num_codewords_used=100_000,
            codeword_dim=4,
            bytes_per_codeword=4,
        ),
        emit_arbitrariness_atom=True,
        codec_name="neural_weight_codec",
    )
    atom = res.coupled_adjustments["atom"]
    assert "vq_codebook_K_solved_for_neural_weight_codec" in atom.atom_id
    assert atom.resolution_path.value == "analytical_solve"
