# SPDX-License-Identifier: MIT
"""Tests for Row #10 greedy NN-TSP per-pair ordering solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.greedy_tsp_per_pair_ordering import (
    PairOrderingInput,
    solve_greedy_tsp_per_pair_ordering,
)


def test_4_pair_chain_basic() -> None:
    dissim = [
        [0.0, 0.1, 0.5, 0.9],
        [0.1, 0.0, 0.1, 0.5],
        [0.5, 0.1, 0.0, 0.1],
        [0.9, 0.5, 0.1, 0.0],
    ]
    res = solve_greedy_tsp_per_pair_ordering(
        PairOrderingInput(num_pairs=4, pairwise_dissimilarity=dissim, start_pair_id=0),
    )
    assert isinstance(res.solved_value, list)
    assert sorted(res.solved_value) == [0, 1, 2, 3]
    assert res.coupled_adjustments["is_permutation"] is True
    # Greedy NN should hit pair 1 next (dissim 0.1)
    assert res.solved_value[1] == 1


def test_error_path_dissim_row_mismatch() -> None:
    with pytest.raises(ValueError, match="!= num_pairs"):
        PairOrderingInput(
            num_pairs=3,
            pairwise_dissimilarity=[[0.0, 0.1], [0.1, 0.0]],
            start_pair_id=0,
        )


def test_error_path_start_pair_out_of_range() -> None:
    with pytest.raises(ValueError, match="start_pair_id"):
        PairOrderingInput(
            num_pairs=3,
            pairwise_dissimilarity=[[0.0, 0.1, 0.5], [0.1, 0.0, 0.1], [0.5, 0.1, 0.0]],
            start_pair_id=10,
        )


def test_regression_starts_at_specified_pair() -> None:
    n = 5
    dissim = [[0.0 if i == j else 0.5 for j in range(n)] for i in range(n)]
    res = solve_greedy_tsp_per_pair_ordering(
        PairOrderingInput(num_pairs=n, pairwise_dissimilarity=dissim, start_pair_id=2),
    )
    assert res.solved_value[0] == 2
    assert sorted(res.solved_value) == list(range(n))


def test_integration_atom_emission_when_requested() -> None:
    dissim = [[0.0, 0.1], [0.1, 0.0]]
    res = solve_greedy_tsp_per_pair_ordering(
        PairOrderingInput(num_pairs=2, pairwise_dissimilarity=dissim, start_pair_id=0),
        emit_arbitrariness_atom=True,
        substrate_id="a1",
    )
    atom = res.coupled_adjustments["atom"]
    assert "pair_ordering_greedy_tsp_solved_for_a1" == atom.atom_id
    assert atom.resolution_path.value == "analytical_solve"
