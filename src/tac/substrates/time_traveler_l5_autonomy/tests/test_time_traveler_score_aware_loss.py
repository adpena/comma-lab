# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler L5 Autonomy score-aware loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.substrates.time_traveler_l5_autonomy.score_aware_loss import (
    TimeTravelerLossWeights,
    TimeTravelerScoreAwareLoss,
)


class _StandinSegScorer(nn.Module):
    """Upstream-contract SegNet stand-in with ``preprocess_input``.

    Matches the contest 5-class softmax / 4D ``(B, 5, H, W)`` output that
    ``scorer_loss_terms_btchw`` consumes (via the canonical
    ``score_pair_components`` helper).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        return x_btchw[:, -1]  # last frame, (B, 3, H, W)

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


class _StandinPoseScorer(nn.Module):
    """Upstream-contract PoseNet stand-in returning pose dict."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        # mean over RGB then mean across HxW -> proxy 12-channel feature.
        flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        flat6 = flat.expand(-1, 6, -1, -1)
        flat12 = flat6.reshape(b, t * 6, h, w)
        # Downsample to (B, 12, H/2, W/2) keeping the 12 channel structure.
        return flat12[:, :, ::2, ::2]

    def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": self.proj(x_b12hw.flatten(2).mean(dim=2))}


def _toy_ctx(batch: int = 1, h: int = 32, w: int = 48):
    rgb_0 = (torch.rand(batch, 3, h, w) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(batch, 3, h, w) * 255.0).requires_grad_(True)
    gt_0 = torch.rand(batch, 3, h, w) * 255.0
    gt_1 = torch.rand(batch, 3, h, w) * 255.0
    bytes_proxy = torch.tensor(100_000.0)
    return rgb_0, rgb_1, gt_0, gt_1, bytes_proxy


def test_default_weights_are_contest_compliant() -> None:
    """Default Lagrangian weights match the contest formula."""
    w = TimeTravelerLossWeights()
    assert w.alpha_rate == 25.0
    assert w.beta_seg == 100.0
    assert abs(w.gamma_pose - (10.0) ** 0.5) < 1e-9
    assert w.pose_weight_scale == 1.0
    assert w.contest_normalizer == 37_545_489.0


def test_default_weights_carry_predictive_term() -> None:
    """delta_predict default is small (0.1) but nonzero — predictive-coding hook."""
    w = TimeTravelerLossWeights()
    assert 0.0 < w.delta_predict <= 1.0


def test_forward_returns_scalar_loss_and_dict() -> None:
    """Loss forward returns ``(scalar_tensor, dict)``."""
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    loss, parts = loss_fn(rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, noise_std=0.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert isinstance(parts, dict)


def test_forward_backward_propagates_gradient_to_rgb_inputs() -> None:
    """Loss is differentiable through eval-roundtrip into RGB inputs."""
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    loss, _ = loss_fn(rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, noise_std=0.0)
    loss.backward()
    assert rgb_0.grad is not None and rgb_0.grad.abs().sum().item() > 0
    assert rgb_1.grad is not None and rgb_1.grad.abs().sum().item() > 0


def test_forward_rejects_eval_roundtrip_false() -> None:
    """apply_eval_roundtrip=False raises ValueError per CLAUDE.md non-negotiable."""
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    with pytest.raises(ValueError, match="apply_eval_roundtrip=False"):
        loss_fn(rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, apply_eval_roundtrip=False)


def test_forward_rejects_negative_noise_std() -> None:
    """noise_std < 0 raises ValueError."""
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    with pytest.raises(ValueError, match="noise_std"):
        loss_fn(rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, noise_std=-0.1)


def test_forward_includes_predictive_term_when_residual_passed() -> None:
    """When predictive_residual is passed, the predictive term is nonzero."""
    weights = TimeTravelerLossWeights(delta_predict=0.5)
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), weights
    )
    loss_fn.eval()  # disable training-only noise so test is deterministic
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    residual = torch.ones(8, 8) * 0.5
    _, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        predictive_residual=residual, noise_std=0.0,
    )
    # mean(0.5^2) = 0.25
    assert torch.isclose(parts["predict_term"], torch.tensor(0.25), atol=1e-6)


def test_forward_predictive_residual_gradient_flows() -> None:
    """The predictive-coding-hierarchy term contributes gradient to the residual."""
    weights = TimeTravelerLossWeights(delta_predict=1.0)
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), weights
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    residual = torch.randn(4, 4, requires_grad=True)
    loss, _ = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        predictive_residual=residual, noise_std=0.0,
    )
    loss.backward()
    assert residual.grad is not None
    assert residual.grad.abs().sum().item() > 0


def test_forward_parts_dict_has_all_canonical_keys() -> None:
    """parts dict carries rate / seg / pose / pose_sqrt / predict / loss_total."""
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    _, parts = loss_fn(rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, noise_std=0.0)
    for key in ("rate_term", "seg_term", "pose_term", "pose_sqrt",
                "predict_term", "loss_total"):
        assert key in parts, f"parts dict missing {key!r}"


def test_forward_rate_term_scales_with_archive_bytes() -> None:
    """Larger archive bytes => larger rate term (linear)."""
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, _ = _toy_ctx()
    small = torch.tensor(50_000.0)
    large = torch.tensor(200_000.0)
    _, parts_small = loss_fn(rgb_0, rgb_1, gt_0, gt_1, small, noise_std=0.0)
    _, parts_large = loss_fn(rgb_0, rgb_1, gt_0, gt_1, large, noise_std=0.0)
    assert parts_large["rate_term"] > parts_small["rate_term"]


def test_score_aware_common_contract_honored() -> None:
    """The loss uses ``score_pair_components`` -> we get seg and pose terms.

    The actual values depend on stand-in scorer numerics; we only require
    finite tensors of dim 0 (scalar).
    """
    loss_fn = TimeTravelerScoreAwareLoss(
        _StandinSegScorer(), _StandinPoseScorer(), TimeTravelerLossWeights()
    )
    rgb_0, rgb_1, gt_0, gt_1, bytes_proxy = _toy_ctx()
    _, parts = loss_fn(rgb_0, rgb_1, gt_0, gt_1, bytes_proxy, noise_std=0.0)
    assert parts["seg_term"].dim() == 0
    assert parts["pose_term"].dim() == 0
    assert torch.isfinite(parts["seg_term"])
    assert torch.isfinite(parts["pose_term"])
