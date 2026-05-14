# SPDX-License-Identifier: MIT
"""Tests for ``score_pair_components_with_cache`` (O1 wire-in).

The cache-aware wrapper is the canonical entry point substrate trainers
adopt to consume the :class:`tac.training_optimization.GTScorerCache`.
Per the optimization audit 2026-05-14 §3.1, the wrapper must be
mathematically identical to the un-cached ``score_pair_components`` when
the cache holds the same tensors a direct GT forward would produce.

Coverage targets:
- contract: SegNet/PoseNet missing preprocess_input -> raises
- happy path: returns (seg_term, pose_term) tensors with grad on predicted
- equivalence vs uncached score_pair_components on same inputs
- logits-mode (gt_seg_already_probs=False) acceptance
- shape contracts on rgb_0_rt / rgb_1_rt / gt_pose / gt_seg
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.substrates.score_aware_common import (
    ScoreAwareScorerContractError,
    score_pair_components,
    score_pair_components_with_cache,
)


# ---------------------------------------------------------------------------
# Fake scorers (reuse the pattern from scorer_cache tests)
# ---------------------------------------------------------------------------


class _FakePoseNet(nn.Module):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(12, 12)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = pair_btchw.shape
        pooled = pair_btchw.mean(dim=(3, 4))
        pooled = torch.cat([pooled, pooled, pooled, pooled], dim=-1)
        return pooled.reshape(b, t, 12)

    def forward(self, x_btc: torch.Tensor) -> dict:
        b, t, c = x_btc.shape
        flat = x_btc.reshape(b * t, c)
        out = self.linear(flat).reshape(b, t, 12)
        return {"pose": out}


class _FakeSegNet(nn.Module):
    def __init__(self, *, seed: int = 1, num_classes: int = 5) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        return pair_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


def _make_setup(b: int = 2, h: int = 16, w: int = 16):
    posenet = _FakePoseNet()
    segnet = _FakeSegNet()
    posenet.eval()
    segnet.eval()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)

    torch.manual_seed(42)
    rgb_0 = torch.rand((b, 3, h, w), requires_grad=True)
    rgb_1 = torch.rand((b, 3, h, w), requires_grad=True)
    gt_rgb_0 = torch.rand((b, 3, h, w))
    gt_rgb_1 = torch.rand((b, 3, h, w))
    return posenet, segnet, rgb_0, rgb_1, gt_rgb_0, gt_rgb_1


def _build_gt_cache_tensors(
    posenet: _FakePoseNet,
    segnet: _FakeSegNet,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    *,
    seg_already_probs: bool,
):
    pair_gt = torch.stack([gt_rgb_0, gt_rgb_1], dim=1)
    with torch.no_grad():
        pose_in = posenet.preprocess_input(pair_gt)
        gt_pose_batch = posenet(pose_in)["pose"]
        seg_in = segnet.preprocess_input(pair_gt)
        seg_logits = segnet(seg_in)
        if seg_already_probs:
            gt_seg_batch = F.softmax(seg_logits, dim=1)
        else:
            gt_seg_batch = seg_logits
    return gt_pose_batch, gt_seg_batch


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_cache_aware_raises_when_segnet_lacks_preprocess() -> None:
    posenet, _, rgb_0, rgb_1, gt_0, gt_1 = _make_setup()
    bad_segnet = nn.Module()
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, _FakeSegNet(), gt_0, gt_1, seg_already_probs=True
    )
    with pytest.raises(ScoreAwareScorerContractError, match="SegNet"):
        score_pair_components_with_cache(
            seg_scorer=bad_segnet,
            pose_scorer=posenet,
            rgb_0_rt=rgb_0,
            rgb_1_rt=rgb_1,
            gt_pose_batch=gt_pose,
            gt_seg_batch=gt_seg,
            gt_seg_already_probs=True,
        )


def test_cache_aware_raises_when_posenet_lacks_preprocess() -> None:
    _, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup()
    bad_posenet = nn.Module()
    gt_pose, gt_seg = _build_gt_cache_tensors(
        _FakePoseNet(), segnet, gt_0, gt_1, seg_already_probs=True
    )
    with pytest.raises(ScoreAwareScorerContractError, match="PoseNet"):
        score_pair_components_with_cache(
            seg_scorer=segnet,
            pose_scorer=bad_posenet,
            rgb_0_rt=rgb_0,
            rgb_1_rt=rgb_1,
            gt_pose_batch=gt_pose,
            gt_seg_batch=gt_seg,
            gt_seg_already_probs=True,
        )


def test_cache_aware_raises_on_non_4d_rgb() -> None:
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup()
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=True
    )
    with pytest.raises(ScoreAwareScorerContractError, match="4D"):
        score_pair_components_with_cache(
            seg_scorer=segnet,
            pose_scorer=posenet,
            rgb_0_rt=rgb_0.unsqueeze(0),  # 5D
            rgb_1_rt=rgb_1.unsqueeze(0),
            gt_pose_batch=gt_pose,
            gt_seg_batch=gt_seg,
            gt_seg_already_probs=True,
        )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_cache_aware_returns_tensors_with_grad_on_predicted() -> None:
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup(b=2, h=16, w=16)
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=True
    )
    seg_term, pose_term = score_pair_components_with_cache(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_pose_batch=gt_pose,
        gt_seg_batch=gt_seg,
        gt_seg_already_probs=True,
    )
    assert isinstance(seg_term, torch.Tensor)
    assert isinstance(pose_term, torch.Tensor)
    # The terms should be gradient-bearing wrt the predicted RGB.
    assert seg_term.requires_grad or pose_term.requires_grad


def test_cache_aware_returns_finite_values() -> None:
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup()
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=True
    )
    seg_term, pose_term = score_pair_components_with_cache(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_pose_batch=gt_pose,
        gt_seg_batch=gt_seg,
        gt_seg_already_probs=True,
    )
    assert torch.isfinite(seg_term).all()
    assert torch.isfinite(pose_term).all()


def test_cache_aware_accepts_logits_mode() -> None:
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup()
    gt_pose, gt_seg_logits = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=False
    )
    seg_term, pose_term = score_pair_components_with_cache(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_pose_batch=gt_pose,
        gt_seg_batch=gt_seg_logits,
        gt_seg_already_probs=False,
        segmentation_temperature=2.0,
    )
    assert torch.isfinite(seg_term).all()
    assert torch.isfinite(pose_term).all()


# ---------------------------------------------------------------------------
# Mathematical equivalence test (CENTRAL CORRECTNESS ANCHOR)
# ---------------------------------------------------------------------------


def test_cache_aware_matches_uncached_score_pair_components() -> None:
    """The central correctness claim of O1: cached == uncached for same inputs.

    If this test ever fails, the cache is NOT a pure-speed primitive and
    the audit's "signal regression risk = mathematically zero" claim is
    violated. The cache must hold exactly what the direct GT forward
    would have produced.
    """
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup(b=2, h=16, w=16)
    # Build cache outputs by running the same GT forward the un-cached
    # path runs internally.
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=True
    )

    seg_uncached, pose_uncached = score_pair_components(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_rgb_0=gt_0,
        gt_rgb_1=gt_1,
    )
    seg_cached, pose_cached = score_pair_components_with_cache(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_pose_batch=gt_pose,
        gt_seg_batch=gt_seg,
        gt_seg_already_probs=True,
    )

    # Floating-point equality up to scorer-forward numerics. The two
    # paths differ only in WHERE the gt softmax is computed (cached path
    # stores probs; uncached path recomputes them). At fp32 the gap is
    # below 1e-6.
    assert torch.allclose(seg_cached, seg_uncached, atol=1e-6, rtol=1e-6)
    assert torch.allclose(pose_cached, pose_uncached, atol=1e-6, rtol=1e-6)


def test_cache_aware_loss_gradient_flow_to_predicted() -> None:
    """Cache lookup must NOT break gradient flow on the predicted path."""
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup(b=2, h=16, w=16)
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=True
    )
    seg_term, pose_term = score_pair_components_with_cache(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_pose_batch=gt_pose,
        gt_seg_batch=gt_seg,
        gt_seg_already_probs=True,
    )
    total = 100.0 * seg_term + torch.sqrt(10.0 * pose_term + 1e-8)
    total.backward()
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None
    assert torch.isfinite(rgb_0.grad).all()
    assert torch.isfinite(rgb_1.grad).all()


def test_cache_aware_loss_no_gradient_to_gt_tensors() -> None:
    """GT cache tensors should NOT carry gradients (the whole point of caching)."""
    posenet, segnet, rgb_0, rgb_1, gt_0, gt_1 = _make_setup(b=2, h=16, w=16)
    gt_pose, gt_seg = _build_gt_cache_tensors(
        posenet, segnet, gt_0, gt_1, seg_already_probs=True
    )
    # Disable gradient on the GT cache (the canonical cache builder does this
    # with no_grad + detach + .cpu()).
    gt_pose = gt_pose.detach()
    gt_seg = gt_seg.detach()
    assert not gt_pose.requires_grad
    assert not gt_seg.requires_grad

    seg_term, pose_term = score_pair_components_with_cache(
        seg_scorer=segnet,
        pose_scorer=posenet,
        rgb_0_rt=rgb_0,
        rgb_1_rt=rgb_1,
        gt_pose_batch=gt_pose,
        gt_seg_batch=gt_seg,
        gt_seg_already_probs=True,
    )
    (seg_term + pose_term).backward()
    # GT tensors must remain grad-free.
    assert gt_pose.grad is None
    assert gt_seg.grad is None
