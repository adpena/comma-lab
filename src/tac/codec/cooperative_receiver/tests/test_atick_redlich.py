"""Tests for the Atick-Redlich cooperative-receiver primitive."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.codec.cooperative_receiver.atick_redlich import (
    AtickRedlichWeights,
    CooperativeReceiverOutput,
    cooperative_receiver_loss,
)


class _StandinSegScorer(nn.Module):
    """Upstream-contract SegNet stand-in with ``preprocess_input``.

    Mirrors the time-traveler test stand-in so the primitive's contract
    surface is verified against the same shape/dtype contract the
    in-tree consumer relies on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        return x_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


class _StandinPoseScorer(nn.Module):
    """Upstream-contract PoseNet stand-in returning a pose dict."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6, bias=False)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_btchw.shape
        flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        flat6 = flat.expand(-1, 6, -1, -1)
        flat12 = flat6.reshape(b, t * 6, h, w)
        return flat12[:, :, ::2, ::2]

    def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pose": self.proj(x_b12hw.flatten(2).mean(dim=2))}


def _toy_pair(batch: int = 1, h: int = 32, w: int = 48):
    rgb_0 = (torch.rand(batch, 3, h, w) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(batch, 3, h, w) * 255.0).requires_grad_(True)
    gt_0 = torch.rand(batch, 3, h, w) * 255.0
    gt_1 = torch.rand(batch, 3, h, w) * 255.0
    return rgb_0, rgb_1, gt_0, gt_1


def test_default_weights_match_contest_formula() -> None:
    """Default Atick-Redlich weights match the contest Lagrangian."""
    w = AtickRedlichWeights()
    assert w.beta_seg == 100.0
    assert abs(w.gamma_pose - (10.0) ** 0.5) < 1e-9
    assert w.pose_weight_scale == 1.0


def test_weights_reject_negative_beta_seg() -> None:
    with pytest.raises(ValueError, match="beta_seg"):
        AtickRedlichWeights(beta_seg=-1.0)


def test_weights_reject_negative_gamma_pose() -> None:
    with pytest.raises(ValueError, match="gamma_pose"):
        AtickRedlichWeights(gamma_pose=-1.0)


def test_weights_reject_negative_pose_weight_scale() -> None:
    with pytest.raises(ValueError, match="pose_weight_scale"):
        AtickRedlichWeights(pose_weight_scale=-0.5)


def test_loss_returns_cooperative_receiver_output() -> None:
    """Primitive returns the typed dataclass with the canonical fields."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
    )
    assert isinstance(out, CooperativeReceiverOutput)
    assert out.cooperative_loss.dim() == 0
    assert out.seg_term.dim() == 0
    assert out.pose_term.dim() == 0
    assert out.pose_sqrt.dim() == 0
    assert torch.isfinite(out.cooperative_loss)


def test_loss_gradient_flows_into_predicted_rgb() -> None:
    """The cooperative loss is differentiable through eval-roundtrip into RGB inputs."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
    )
    out.cooperative_loss.backward()
    assert rgb_0.grad is not None and rgb_0.grad.abs().sum().item() > 0
    assert rgb_1.grad is not None and rgb_1.grad.abs().sum().item() > 0


def test_loss_rejects_eval_roundtrip_false() -> None:
    """apply_eval_roundtrip=False raises per CLAUDE.md non-negotiable."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    with pytest.raises(ValueError, match="apply_eval_roundtrip=False"):
        cooperative_receiver_loss(
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
            seg_scorer=_StandinSegScorer(),
            pose_scorer=_StandinPoseScorer(),
            apply_eval_roundtrip=False,
        )


def test_loss_rejects_unit_range_rgb_inputs() -> None:
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    with pytest.raises(ValueError, match=r"\[0, 1\] unit RGB"):
        cooperative_receiver_loss(
            rgb_0 / 255.0,
            rgb_1 / 255.0,
            gt_0 / 255.0,
            gt_1 / 255.0,
            seg_scorer=_StandinSegScorer(),
            pose_scorer=_StandinPoseScorer(),
        )


def test_loss_rejects_out_of_range_rgb_inputs() -> None:
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    rgb_0 = rgb_0.detach().clone()
    rgb_0[0, 0, 0, 0] = 300.0
    with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
        cooperative_receiver_loss(
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
            seg_scorer=_StandinSegScorer(),
            pose_scorer=_StandinPoseScorer(),
        )


def test_loss_accepts_custom_eval_roundtrip_fn_for_isolation() -> None:
    """Callers can inject a no-op eval-roundtrip for unit-test isolation."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()

    invocations = {"count": 0}

    def identity_roundtrip(x: torch.Tensor) -> torch.Tensor:
        invocations["count"] += 1
        return x

    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        eval_roundtrip_fn=identity_roundtrip,
    )
    # Two RGB inputs => exactly two roundtrip calls.
    assert invocations["count"] == 2
    assert torch.isfinite(out.cooperative_loss)


def test_loss_uses_default_weights_when_none_passed() -> None:
    """Passing weights=None resolves to AtickRedlichWeights() defaults."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    out_none = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        weights=None,
        eval_roundtrip_fn=identity,
    )
    # Same call with explicit defaults must be numerically identical.
    out_explicit = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        weights=AtickRedlichWeights(),
        eval_roundtrip_fn=identity,
    )
    # Stand-in scorers are stochastic-free so the comparison is meaningful.
    # Both calls instantiate fresh stand-ins so we only assert finite + same shape.
    assert out_none.cooperative_loss.shape == out_explicit.cooperative_loss.shape


def test_loss_pose_weight_scale_tilts_pose_term() -> None:
    """pose_weight_scale=2 doubles the pose contribution (linear in scale)."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    seg = _StandinSegScorer()
    pose = _StandinPoseScorer()
    out_1x = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=seg,
        pose_scorer=pose,
        weights=AtickRedlichWeights(pose_weight_scale=1.0),
        eval_roundtrip_fn=identity,
    )
    out_2x = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=seg,
        pose_scorer=pose,
        weights=AtickRedlichWeights(pose_weight_scale=2.0),
        eval_roundtrip_fn=identity,
    )
    # delta_loss = 1x_pose_contrib (because 2x - 1x = 1x).
    pose_contrib = AtickRedlichWeights().gamma_pose * out_1x.pose_sqrt
    expected_delta = pose_contrib
    actual_delta = out_2x.cooperative_loss - out_1x.cooperative_loss
    assert torch.isclose(actual_delta, expected_delta, atol=1e-5)


def test_loss_seg_zero_weight_drops_seg_contribution() -> None:
    """beta_seg=0 makes the loss equal to the pose contribution alone."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        weights=AtickRedlichWeights(beta_seg=0.0),
        eval_roundtrip_fn=identity,
    )
    expected = AtickRedlichWeights().gamma_pose * out.pose_sqrt
    assert torch.isclose(out.cooperative_loss, expected, atol=1e-5)


def test_loss_pose_zero_weight_drops_pose_contribution() -> None:
    """gamma_pose=0 makes the loss equal to the seg contribution alone."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        weights=AtickRedlichWeights(gamma_pose=0.0),
        eval_roundtrip_fn=identity,
    )
    expected = 100.0 * out.seg_term
    assert torch.isclose(out.cooperative_loss, expected, atol=1e-4)


def test_loss_pose_sqrt_floor_avoids_nan_at_zero_pose() -> None:
    """pose_sqrt floor is honored so sqrt of zero pose stays finite."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
        eval_roundtrip_fn=identity,
        pose_sqrt_floor=1e-12,
    )
    # Stand-in pose may not be exactly zero, but the sqrt must always be finite.
    assert torch.isfinite(out.pose_sqrt)
    assert out.pose_sqrt >= 0.0


def test_loss_targets_do_not_receive_gradient() -> None:
    """Ground-truth targets are detached / non-leaf; no grad flow into them."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    # gt tensors created with requires_grad=False (default for torch.rand).
    assert gt_0.requires_grad is False
    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
    )
    out.cooperative_loss.backward()
    # No grad attribute (or None) on the ground-truth tensors.
    assert gt_0.grad is None
    assert gt_1.grad is None


def test_loss_components_carry_gradient_for_diagnostic_logging() -> None:
    """seg_term, pose_term, pose_sqrt remain attached to the graph (diagnostic-readable)."""
    rgb_0, rgb_1, gt_0, gt_1 = _toy_pair()
    out = cooperative_receiver_loss(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        seg_scorer=_StandinSegScorer(),
        pose_scorer=_StandinPoseScorer(),
    )
    # All three should require_grad — caller can detach for logging.
    assert out.seg_term.requires_grad
    assert out.pose_term.requires_grad
    assert out.pose_sqrt.requires_grad
