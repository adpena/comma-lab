# SPDX-License-Identifier: MIT
"""Tests for Row #5 MST-based frame ordering solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.min_spanning_tree_frame_ordering import (
    FrameOrderingInput,
    solve_min_spanning_tree_frame_ordering,
)


def test_4_frame_chain_basic() -> None:
    # Chain pattern: 0-1-2-3 most similar in order
    dissim = [
        [0.0, 0.1, 0.5, 0.9],
        [0.1, 0.0, 0.1, 0.5],
        [0.5, 0.1, 0.0, 0.1],
        [0.9, 0.5, 0.1, 0.0],
    ]
    res = solve_min_spanning_tree_frame_ordering(
        FrameOrderingInput(num_frames=4, pairwise_dissimilarity=dissim, anchor_frame_id=0),
    )
    assert isinstance(res.solved_value, list)
    assert len(res.solved_value) == 4
    assert sorted(res.solved_value) == [0, 1, 2, 3]
    # First should be anchor
    assert res.solved_value[0] == 0


def test_error_path_dissim_row_size_mismatch() -> None:
    with pytest.raises(ValueError, match="!= num_frames"):
        FrameOrderingInput(
            num_frames=3,
            pairwise_dissimilarity=[[0.0, 0.1], [0.1, 0.0]],
            anchor_frame_id=0,
        )


def test_error_path_anchor_out_of_range() -> None:
    with pytest.raises(ValueError, match="anchor_frame_id"):
        FrameOrderingInput(
            num_frames=3,
            pairwise_dissimilarity=[[0.0, 0.1, 0.5], [0.1, 0.0, 0.1], [0.5, 0.1, 0.0]],
            anchor_frame_id=5,
        )


def test_regression_two_frames_trivial() -> None:
    dissim = [[0.0, 1.0], [1.0, 0.0]]
    res = solve_min_spanning_tree_frame_ordering(
        FrameOrderingInput(num_frames=2, pairwise_dissimilarity=dissim, anchor_frame_id=0),
    )
    assert res.solved_value == [0, 1]
    assert res.intermediate_values["mst_total_weight"] == pytest.approx(1.0)


def test_integration_atom_emission_when_requested() -> None:
    dissim = [[0.0, 0.1], [0.1, 0.0]]
    res = solve_min_spanning_tree_frame_ordering(
        FrameOrderingInput(num_frames=2, pairwise_dissimilarity=dissim, anchor_frame_id=0),
        emit_arbitrariness_atom=True,
        substrate_id="a1",
    )
    atom = res.coupled_adjustments["atom"]
    assert "frame_ordering_mst_solved_for_a1" == atom.atom_id
    assert atom.resolution_path.value == "analytical_solve"
