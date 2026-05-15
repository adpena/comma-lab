# SPDX-License-Identifier: MIT
"""NSCS03 score-aware loss tests: weights, gradient path, eval_roundtrip mandate."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.nscs03_end_to_end_balle_joint_codec.score_aware_loss import (
    NSCS03JointScoreAwareLoss,
    NSCS03ScoreAwareLossWeights,
)


class _StubSegScorer(torch.nn.Module):
    """Stub SegNet returning a constant 5-class logit tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C, H, W); return (B, 5, H/2, W/2) logits-shaped tensor
        b = x.shape[0]
        return torch.zeros(b, 5, x.shape[-2] // 2, x.shape[-1] // 2, requires_grad=True)


class _StubPoseScorer(torch.nn.Module):
    """Stub PoseNet returning a constant 6-dim pose tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.zeros(b, 6, requires_grad=True)


class TestNSCS03LossWeights:
    def test_default_weights(self) -> None:
        w = NSCS03ScoreAwareLossWeights()
        assert w.alpha_rate == 25.0
        assert w.beta_seg == 100.0
        assert w.lambda_R == 0.5
        assert w.contest_normalizer == 37_545_489.0


class TestNSCS03LossNonNegotiables:
    def test_eval_roundtrip_must_be_true(self) -> None:
        """Per CLAUDE.md eval_roundtrip non-negotiable."""
        loss_fn = NSCS03JointScoreAwareLoss(
            seg_scorer=_StubSegScorer(),
            pose_scorer=_StubPoseScorer(),
            weights=NSCS03ScoreAwareLossWeights(),
        )
        rgb_0 = torch.rand(1, 3, 384, 512)
        rgb_1 = torch.rand(1, 3, 384, 512)
        gt_0 = torch.rand(1, 3, 384, 512)
        gt_1 = torch.rand(1, 3, 384, 512)
        rate = {
            "main_rate": torch.tensor(1.0, requires_grad=True),
            "hyper_rate": torch.tensor(2.0, requires_grad=True),
            "total_rate": torch.tensor(3.0, requires_grad=True),
        }
        with pytest.raises(ValueError, match="apply_eval_roundtrip=False is forbidden"):
            loss_fn(
                rgb_0, rgb_1, gt_0, gt_1,
                archive_bytes_proxy=torch.tensor(100000.0, requires_grad=True),
                rate_components=rate,
                apply_eval_roundtrip=False,
            )

    def test_negative_noise_std_rejected(self) -> None:
        loss_fn = NSCS03JointScoreAwareLoss(
            seg_scorer=_StubSegScorer(),
            pose_scorer=_StubPoseScorer(),
            weights=NSCS03ScoreAwareLossWeights(),
        )
        rgb_0 = torch.rand(1, 3, 384, 512)
        rgb_1 = torch.rand(1, 3, 384, 512)
        gt_0 = torch.rand(1, 3, 384, 512)
        gt_1 = torch.rand(1, 3, 384, 512)
        rate = {
            "main_rate": torch.tensor(1.0, requires_grad=True),
            "hyper_rate": torch.tensor(2.0, requires_grad=True),
            "total_rate": torch.tensor(3.0, requires_grad=True),
        }
        with pytest.raises(ValueError, match="noise_std must be >= 0"):
            loss_fn(
                rgb_0, rgb_1, gt_0, gt_1,
                archive_bytes_proxy=torch.tensor(100000.0, requires_grad=True),
                rate_components=rate, noise_std=-0.1,
            )

    def test_missing_rate_component_rejected(self) -> None:
        loss_fn = NSCS03JointScoreAwareLoss(
            seg_scorer=_StubSegScorer(),
            pose_scorer=_StubPoseScorer(),
            weights=NSCS03ScoreAwareLossWeights(),
        )
        rgb_0 = torch.rand(1, 3, 384, 512)
        rgb_1 = torch.rand(1, 3, 384, 512)
        gt_0 = torch.rand(1, 3, 384, 512)
        gt_1 = torch.rand(1, 3, 384, 512)
        rate_missing = {"main_rate": torch.tensor(1.0)}  # no hyper_rate / total_rate
        with pytest.raises(ValueError, match="rate_components missing"):
            loss_fn(
                rgb_0, rgb_1, gt_0, gt_1,
                archive_bytes_proxy=torch.tensor(100000.0, requires_grad=True),
                rate_components=rate_missing,
            )
