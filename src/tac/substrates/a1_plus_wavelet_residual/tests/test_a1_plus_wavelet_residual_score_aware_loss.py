"""Tests for the A1 + wavelet residual score-aware Lagrangian.

Covers the CLAUDE.md non-negotiables:
* eval_roundtrip=True required (Catalog #5)
* Score-domain Lagrangian (NOT weight-domain proxies; HNeRV lesson L6)
* preprocess_input contract for both scorers (Catalog #164)
"""
from __future__ import annotations

import pytest
import torch

from tac.substrates.a1_plus_wavelet_residual.score_aware_loss import (
    A1PlusWaveletResidualLossWeights,
    A1PlusWaveletResidualScoreAwareLoss,
)
from tac.substrates._shared.score_aware_loss_real_scorer_test_kit import (
    assert_loss_runs_on_real_segnet,
)


def _toy_seg_scorer():
    class _Seg(torch.nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            # Pick last frame, reduce to (B, 5, H, W) - SegNet 5-class output stand-in
            b, t, c, h, w = x_btchw.shape
            last = x_btchw[:, -1]
            return last.mean(dim=1, keepdim=True).expand(b, 5, h, w)

        def forward(self, pre):
            return pre  # 5-class logits

    return _Seg()


def _toy_pose_scorer():
    """Upstream-contract-faithful PoseNet stand-in.

    Mirrors ``score_aware_loss_real_scorer_test_kit._build_pose_standin``:
    accepts (B, T=2, C=3, H, W) RGB pair, emits (B, T*6, H/2, W/2) YUV6-like
    tensor via average-pool downsample.
    """
    class _Pose(torch.nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            b, t, c, h, w = x_btchw.shape
            assert c == 3, f"expected 3 RGB channels, got {c}"
            flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
            flat6 = flat.expand(-1, 6, -1, -1)
            half_h = h // 2
            half_w = w // 2
            flat6_sub = flat6[..., : half_h * 2, : half_w * 2].reshape(
                b * t, 6, half_h, 2, half_w, 2
            ).mean(dim=(3, 5))
            return flat6_sub.reshape(b, t * 6, half_h, half_w)

        def forward(self, x_b12hw: torch.Tensor):
            return {"pose": x_b12hw.flatten(2).mean(dim=2)}

    return _Pose()


def test_loss_runs_with_apply_eval_roundtrip_true() -> None:
    seg = _toy_seg_scorer()
    pose = _toy_pose_scorer()
    weights = A1PlusWaveletResidualLossWeights()
    loss_fn = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights)
    rgb_0 = torch.rand(1, 3, 32, 48, requires_grad=True) * 255.0
    rgb_1 = torch.rand(1, 3, 32, 48, requires_grad=True) * 255.0
    gt_0 = torch.rand(1, 3, 32, 48) * 255.0
    gt_1 = torch.rand(1, 3, 32, 48) * 255.0
    bytes_proxy = torch.tensor(180_000.0)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert "rate_term" in parts
    assert "seg_term" in parts
    assert "pose_term" in parts
    assert "loss_total" in parts


def test_loss_refuses_apply_eval_roundtrip_false() -> None:
    seg = _toy_seg_scorer()
    pose = _toy_pose_scorer()
    weights = A1PlusWaveletResidualLossWeights()
    loss_fn = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights)
    rgb_0 = torch.rand(1, 3, 16, 16)
    rgb_1 = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="eval_roundtrip"):
        loss_fn(
            rgb_0, rgb_1, rgb_0, rgb_1, torch.tensor(100.0),
            apply_eval_roundtrip=False,
        )


def test_loss_propagates_gradient_to_predicted_rgb() -> None:
    seg = _toy_seg_scorer()
    pose = _toy_pose_scorer()
    weights = A1PlusWaveletResidualLossWeights()
    loss_fn = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights)
    rgb_0 = (torch.rand(1, 3, 32, 48) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(1, 3, 32, 48) * 255.0).requires_grad_(True)
    gt_0 = (torch.rand(1, 3, 32, 48) * 255.0)
    gt_1 = (torch.rand(1, 3, 32, 48) * 255.0)
    bytes_proxy = torch.tensor(180_000.0)
    loss, _parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    loss.backward()
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None
    assert rgb_0.grad.abs().sum() > 0
    assert rgb_1.grad.abs().sum() > 0


def test_loss_weights_default_match_contest_formula() -> None:
    w = A1PlusWaveletResidualLossWeights()
    assert w.alpha_rate == 25.0
    assert w.beta_seg == 100.0
    # gamma_pose = sqrt(10); pose_weight_scale default 1.0 (contest formula)
    assert pytest.approx(w.gamma_pose, rel=1e-6) == 10.0 ** 0.5
    assert w.pose_weight_scale == 1.0
    assert w.contest_normalizer == 37_545_489.0


def test_loss_noise_std_rejects_negative() -> None:
    seg = _toy_seg_scorer()
    pose = _toy_pose_scorer()
    weights = A1PlusWaveletResidualLossWeights()
    loss_fn = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights)
    rgb_0 = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="noise_std must be"):
        loss_fn(
            rgb_0, rgb_0, rgb_0, rgb_0, torch.tensor(100.0),
            apply_eval_roundtrip=True, noise_std=-0.1,
        )


def test_loss_rate_term_proportional_to_bytes_proxy() -> None:
    """The rate term must scale linearly with the archive byte count."""
    seg = _toy_seg_scorer()
    pose = _toy_pose_scorer()
    weights = A1PlusWaveletResidualLossWeights()
    loss_fn = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights)
    rgb_0 = torch.rand(1, 3, 32, 48) * 255.0
    rgb_1 = torch.rand(1, 3, 32, 48) * 255.0
    gt_0 = (rgb_0 + 1.0).clamp(0, 255)
    gt_1 = (rgb_1 + 1.0).clamp(0, 255)
    _, parts_small = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, torch.tensor(100_000.0),
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    _, parts_big = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, torch.tensor(200_000.0),
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    # Rate term doubles; seg/pose terms should be equal between calls.
    assert parts_big["rate_term"] > parts_small["rate_term"]
    assert pytest.approx(
        (parts_big["rate_term"] / parts_small["rate_term"]).item(), rel=1e-5
    ) == 2.0


def test_pose_weight_scale_amplifies_pose_term_contribution() -> None:
    seg = _toy_seg_scorer()
    pose = _toy_pose_scorer()
    weights_1 = A1PlusWaveletResidualLossWeights(pose_weight_scale=1.0)
    weights_3 = A1PlusWaveletResidualLossWeights(pose_weight_scale=3.0)
    loss_fn_1 = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights_1)
    loss_fn_3 = A1PlusWaveletResidualScoreAwareLoss(seg, pose, weights_3)
    rgb_0 = torch.rand(1, 3, 16, 16) * 255.0
    rgb_1 = torch.rand(1, 3, 16, 16) * 255.0
    gt_0 = (rgb_0 + 5.0).clamp(0, 255)
    gt_1 = (rgb_1 + 5.0).clamp(0, 255)
    loss_1, _ = loss_fn_1(
        rgb_0, rgb_1, gt_0, gt_1, torch.tensor(100_000.0),
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    loss_3, _ = loss_fn_3(
        rgb_0, rgb_1, gt_0, gt_1, torch.tensor(100_000.0),
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    # Pose-tilted loss must be strictly larger (gt != rgb so pose_term > 0).
    assert loss_3 > loss_1


def test_loss_with_real_segnet_runs() -> None:
    """Regression guard for the Catalog #164 scorer.preprocess_input contract.

    Skipped if SMP/upstream is not installed.
    """

    def loss_factory(seg, pose):
        return A1PlusWaveletResidualScoreAwareLoss(
            seg_scorer=seg,
            pose_scorer=pose,
            weights=A1PlusWaveletResidualLossWeights(),
        )

    def invoke(loss_fn, ctx):
        return loss_fn(
            ctx["rgb_0"], ctx["rgb_1"], ctx["gt_0"], ctx["gt_1"],
            ctx["bytes_proxy"],
            apply_eval_roundtrip=True, noise_std=0.0,
        )

    assert_loss_runs_on_real_segnet(
        loss_factory=loss_factory,
        invoke_loss=invoke,
    )
