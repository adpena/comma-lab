# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def test_score_weighted_rank_differs_from_raw_norm_when_marginals_differ() -> None:
    from tac.optimization.byte_score_impact import rank_bytes_by_score_impact

    gradient = np.asarray(
        [
            [0.01, 10.0, 0.0],
            [0.09, 0.0, 0.0],
            [0.0, 0.0, 1000.0],
        ],
        dtype=np.float64,
    )
    marginals = {"seg": 100.0, "pose": 0.01, "rate": 0.000001}

    assert rank_bytes_by_score_impact(gradient, marginals, k_top=3) == [1, 0, 2]


def test_summarize_topk_score_impact_emits_axis_shares() -> None:
    from tac.optimization.byte_score_impact import summarize_topk_score_impact

    gradient = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    marginals = {"seg": 100.0, "pose": 10.0, "rate": 1.0}

    summary = summarize_topk_score_impact(gradient, marginals, k_top=1)

    assert summary["top_byte_indices"] == [0]
    assert summary["selected_axis_score_impact_abs_sum"]["seg"] == 200.0
    assert summary["selected_axis_score_impact_abs_sum"]["pose"] == 10.0
    assert summary["selected_axis_share_within_topk"]["seg"] > 0.95


def test_contiguous_byte_runs_groups_neighborhoods() -> None:
    from tac.optimization.byte_score_impact import contiguous_byte_runs

    assert contiguous_byte_runs([5, 3, 4, 10, 12, 11, 11]) == [
        {"start": 3, "end": 5, "length": 3},
        {"start": 10, "end": 12, "length": 3},
    ]
