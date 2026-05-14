# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from tac.losses import scorer_loss_terms_btchw
from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    ScoreAwareScorerContractError,
    score_pair_components,
    stage_frame_pair,
)


class _SegScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess_shapes: list[tuple[int, ...]] = []
        self.forward_shapes: list[tuple[int, ...]] = []

    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        self.preprocess_shapes.append(tuple(pair.shape))
        assert pair.dim() == 5
        return pair[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(x.shape))
        assert x.dim() == 4
        base = x.mean(dim=1, keepdim=True)
        return torch.cat([base + float(i) for i in range(5)], dim=1)


class _PoseScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess_shapes: list[tuple[int, ...]] = []
        self.forward_shapes: list[tuple[int, ...]] = []

    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        self.preprocess_shapes.append(tuple(pair.shape))
        b, t, c, h, w = pair.shape
        assert c == 3
        y = pair.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        y6 = y.expand(-1, 6, h, w)
        return y6.reshape(b, t * 6, h, w)

    def forward(self, x: torch.Tensor):
        self.forward_shapes.append(tuple(x.shape))
        assert x.dim() == 4
        pose = x.flatten(2).mean(dim=2)
        return {"pose": pose}


def test_stage_frame_pair_requires_matching_4d_rgb():
    rgb = torch.zeros(2, 3, 4, 5)
    pair = stage_frame_pair(rgb, rgb + 1)
    assert pair.shape == (2, 2, 3, 4, 5)

    with pytest.raises(ScoreAwareScorerContractError, match="4D RGB"):
        stage_frame_pair(rgb.unsqueeze(1), rgb)
    with pytest.raises(ScoreAwareScorerContractError, match="identical shapes"):
        stage_frame_pair(rgb, torch.zeros(2, 3, 4, 6))
    with pytest.raises(ScoreAwareScorerContractError, match="C=3"):
        stage_frame_pair(torch.zeros(2, 1, 4, 5), torch.zeros(2, 1, 4, 5))


def test_contest_pose_weight_matches_score_formula():
    assert pytest.approx(10.0**0.5) == CONTEST_POSE_SQRT_WEIGHT


def test_score_pair_components_enforces_preprocess_shapes_and_grad():
    rgb_0 = torch.randn(2, 3, 4, 6, requires_grad=True)
    rgb_1 = torch.randn(2, 3, 4, 6, requires_grad=True)
    gt_0 = torch.randn(2, 3, 4, 6)
    gt_1 = torch.randn(2, 3, 4, 6)
    seg = _SegScorer()
    pose = _PoseScorer()

    seg_term, pose_term = score_pair_components(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
    )
    loss = seg_term + torch.sqrt(pose_term.clamp(min=1e-12))
    loss.backward()

    assert seg.preprocess_shapes == [(2, 2, 3, 4, 6), (2, 2, 3, 4, 6)]
    assert seg.forward_shapes == [(2, 3, 4, 6), (2, 3, 4, 6)]
    assert pose.preprocess_shapes == [(2, 2, 3, 4, 6), (2, 2, 3, 4, 6)]
    assert pose.forward_shapes == [(2, 12, 4, 6), (2, 12, 4, 6)]
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None


def test_score_pair_components_matches_canonical_scorer_loss_terms():
    rgb_0 = torch.randn(2, 3, 4, 6, requires_grad=True)
    rgb_1 = torch.randn(2, 3, 4, 6, requires_grad=True)
    gt_0 = torch.randn(2, 3, 4, 6)
    gt_1 = torch.randn(2, 3, 4, 6)
    seg = _SegScorer()
    pose = _PoseScorer()

    seg_term, pose_term = score_pair_components(
        seg_scorer=seg,
        pose_scorer=pose,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
    )
    _, canonical_pose, canonical_seg = scorer_loss_terms_btchw(
        stage_frame_pair(rgb_0, rgb_1),
        stage_frame_pair(gt_0, gt_1),
        pose,
        seg,
    )

    assert torch.allclose(seg_term, canonical_seg)
    assert torch.allclose(pose_term, canonical_pose)


def test_score_pair_components_rejects_missing_preprocess():
    scorer = torch.nn.Identity()
    with pytest.raises(ScoreAwareScorerContractError, match="preprocess_input"):
        score_pair_components(
            seg_scorer=scorer,
            pose_scorer=_PoseScorer(),
            rgb_0_rt=torch.zeros(1, 3, 2, 2),
            rgb_1_rt=torch.zeros(1, 3, 2, 2),
            gt_rgb_0=torch.zeros(1, 3, 2, 2),
            gt_rgb_1=torch.zeros(1, 3, 2, 2),
        )
