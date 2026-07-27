# SPDX-License-Identifier: MIT
"""Regression tests for scorer-native population pose attribution."""

from __future__ import annotations

import math

import pytest
import torch

from tac.xray.per_pair_score_decomposition import PerPairScoreDecomposition


def test_global_pose_sqrt_stays_outside_pair_mean() -> None:
    primitive = PerPairScoreDecomposition()
    result = primitive.compute(
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        target_pose=torch.tensor([0.0, 4.0], dtype=torch.float64),
        top_k=2,
    )
    row = result.primitive_value

    exact = math.sqrt(10.0 * 2.0)
    naive_pair_local = (math.sqrt(10.0 * 0.0) + math.sqrt(10.0 * 4.0)) / 2.0
    assert row.exact_global_distortion_score == pytest.approx(exact)
    assert row.mean_per_pair_contribution == pytest.approx(exact)
    assert row.mean_per_pair_contribution != pytest.approx(naive_pair_local)
    assert row.per_pair_score_attribution == pytest.approx((0.0, 2.0 * exact))
    assert row.top_k_pair_indices == (1, 0)
    assert result.metadata["pair_local_sqrt_used"] is False


def test_attribution_recomposes_complete_seg_and_pose_score() -> None:
    primitive = PerPairScoreDecomposition()
    seg = torch.tensor([0.01, 0.03, 0.02], dtype=torch.float64)
    pose = torch.tensor([1.0, 4.0, 7.0], dtype=torch.float64)
    result = primitive.compute(seg, target_pose=pose, top_k=3)
    row = result.primitive_value

    expected = 100.0 * float(seg.mean()) + math.sqrt(10.0 * float(pose.mean()))
    assert row.global_seg_distortion == pytest.approx(0.02)
    assert row.global_pose_distortion == pytest.approx(4.0)
    assert row.global_pose_score_term == pytest.approx(math.sqrt(40.0))
    assert row.exact_global_distortion_score == pytest.approx(expected)
    assert row.total_distortion_sum / row.n_pairs == pytest.approx(expected)
    assert sum(row.per_pair_score_attribution) / row.n_pairs == pytest.approx(expected)
    assert row.pose_pair_mse_vjp_scale == pytest.approx(5.0 / (3.0 * math.sqrt(40.0)))
    assert row.top_k_cumulative_fraction[-1] == pytest.approx(1.0)


def test_zero_pose_has_zero_attribution_and_singular_costate_marker() -> None:
    row = (
        PerPairScoreDecomposition()
        .compute(
            torch.tensor([0.01, 0.02], dtype=torch.float64),
            target_pose=torch.zeros(2, dtype=torch.float64),
            top_k=2,
        )
        .primitive_value
    )

    assert row.global_pose_score_term == 0.0
    assert row.pose_pair_mse_vjp_scale is None
    assert row.per_pair_score_attribution == pytest.approx((1.0, 2.0))
    assert row.exact_global_distortion_score == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("seg", "pose"),
    [
        ([0.0, -1.0], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, -1.0]),
        ([0.0, float("nan")], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, float("inf")]),
    ],
)
def test_invalid_distortions_fail_closed(
    seg: list[float],
    pose: list[float],
) -> None:
    with pytest.raises(ValueError, match=r"finite|non-negative"):
        PerPairScoreDecomposition().compute(
            torch.tensor(seg, dtype=torch.float64),
            target_pose=torch.tensor(pose, dtype=torch.float64),
        )
