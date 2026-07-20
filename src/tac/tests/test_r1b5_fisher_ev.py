# SPDX-License-Identifier: MIT
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from tac.optimization.r1b5_fisher_ev import (
    R1B5FisherEVError,
    exact_half_pixel_axis_supports,
    fisher_trace_from_margin,
    head_pair_norm_table,
    rank_pdw1_candidates,
    support_overlap_component_histogram,
)


def test_contest_downsample_supports_are_disjoint() -> None:
    rows = exact_half_pixel_axis_supports(874, 384)
    cols = exact_half_pixel_axis_supports(1164, 512)
    assert len(rows) == 384
    assert len(cols) == 512
    assert all(set(a.indices).isdisjoint(b.indices) for a, b in pairwise(rows))
    assert all(set(a.indices).isdisjoint(b.indices) for a, b in pairwise(cols))


def test_support_overlap_partition_keeps_pairs_and_singletons_separate() -> None:
    result = support_overlap_component_histogram([(0, 0, 0), (0, 0, 1), (1, 0, 0)])
    assert result["cell_count"] == 3
    assert result["component_count"] == 3
    assert result["component_size_histogram"] == {"1": 3}
    assert result["non_singleton_component_count"] == 0


def test_fisher_trace_and_head_norm_table() -> None:
    assert fisher_trace_from_margin(0.0) == 0.5
    assert fisher_trace_from_margin(2.0) < 0.5
    weight = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    norms = head_pair_norm_table(weight)
    assert norms.shape == (5, 5)
    assert np.allclose(np.diag(norms), 0.0)
    assert np.allclose(norms, norms.T)
    with pytest.raises(R1B5FisherEVError):
        fisher_trace_from_margin(-1.0)


def test_tiny_rank_orders_edge_then_flip_distance() -> None:
    labels = np.zeros((1, 384, 512), dtype=np.int8)
    labels[0, 0, 0] = 1
    pred = labels.copy()
    pred[0, 0, 0] = 0
    pred[0, 100, 100] = 1
    winner = labels[0].copy()
    rival = np.ones((384, 512), dtype=np.int8)
    rival[0, 0] = 0
    sidecars = {
        0: {
            "winner": winner,
            "rival": rival,
            "cached_margin": np.full((384, 512), 0.2, dtype=np.float32),
            "seg_q": np.ones((384, 512, 3), dtype=np.float32) / np.sqrt(3.0),
            "seg_local_lipschitz": np.ones((384, 512), dtype=np.float32),
        }
    }
    norms = np.ones((5, 5), dtype=np.float64)
    np.fill_diagonal(norms, 0.0)
    rows, summary = rank_pdw1_candidates(
        labels=labels,
        hard_prediction=pred,
        sidecars=sidecars,
        head_pair_norm_table=norms,
        enforce_counts=False,
    )
    assert len(rows) == 2
    assert rows[0][:3] == [0, 0, 0]
    assert rows[0][6] == 0
    assert rows[1][6] == 2
    assert summary["candidate_count"] == 2
