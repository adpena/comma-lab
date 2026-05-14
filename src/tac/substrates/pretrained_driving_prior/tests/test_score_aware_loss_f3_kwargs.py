# SPDX-License-Identifier: MIT
"""F3 kwargs mathematical-equivalence test for PDP score-aware loss.

Per Council omnibus Decision 13 PROCEED Option C 2026-05-14 + Time-Traveler
default-OFF amendment: ``DrivingPriorScoreAwareLoss.forward`` accepts the 3
F3 GTScorerCache kwargs (``gt_pose_batch`` / ``gt_seg_batch`` /
``gt_seg_already_probs``) and routes through
:func:`tac.substrates.score_aware_common.score_pair_components_dispatch`.

This test proves the substrate-side wire-in is mathematically byte-faithful:
loss values from the cache path are within numerical tolerance of the
GT-forward path on identical inputs. The cache stores exactly what direct
GT forward produces; therefore the computed seg/pose terms must match.

The test does NOT exercise the trainer (no GPU, $0). It uses the canonical
fake scorers from the sister suite ``test_score_pair_components_with_cache``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.training_optimization import build_gt_scorer_cache


class _FakePoseNet(nn.Module):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(12, 12)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        b, t, _c, _h, _w = pair_btchw.shape
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


def _make_loss_fn():
    """Build a real DrivingPriorScoreAwareLoss with fake scorers + tiny prior."""
    from tac.substrates.pretrained_driving_prior.score_aware_loss import (
        DrivingPriorLossWeights,
        DrivingPriorScoreAwareLoss,
    )
    from tac.substrates.pretrained_driving_prior.prior_application import (
        DashcamPriorLoss,
        PriorApplicationWeights,
    )
    from tac.substrates.pretrained_driving_prior.codebook import (
        deterministic_zero_codebook,
    )

    posenet = _FakePoseNet()
    segnet = _FakeSegNet()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)
    posenet.eval()
    segnet.eval()

    # Use the canonical deterministic-zero codebook so the prior loss is
    # well-defined and prior contribution is constant per call (cancels out
    # of the equivalence comparison since delta_prior=0).
    book = deterministic_zero_codebook()
    prior_weights = PriorApplicationWeights()
    prior_loss = DashcamPriorLoss(book, prior_weights)

    weights = DrivingPriorLossWeights(
        alpha_rate=25.0,
        beta_seg=100.0,
        gamma_pose=1.0,
        pose_weight_scale=1.0,
        delta_prior=0.0,  # zero out prior so the comparison isolates seg+pose
        contest_normalizer=37_545_489.0,
    )
    loss_fn = DrivingPriorScoreAwareLoss(
        seg_scorer=segnet,
        pose_scorer=posenet,
        prior_loss=prior_loss,
        weights=weights,
    )
    return loss_fn, posenet, segnet


def test_pdp_loss_accepts_f3_kwargs_no_error():
    """Calling forward with the 3 F3 kwargs as None succeeds (default-OFF path)."""
    loss_fn, _pose, _seg = _make_loss_fn()
    torch.manual_seed(7)
    rgb_0 = torch.rand(2, 3, 384, 512) * 255.0
    rgb_1 = torch.rand(2, 3, 384, 512) * 255.0
    gt_0 = torch.rand(2, 3, 384, 512) * 255.0
    gt_1 = torch.rand(2, 3, 384, 512) * 255.0
    rgb_0.requires_grad_(True)
    rgb_1.requires_grad_(True)
    archive_bytes_proxy = torch.tensor(100_000.0)

    loss, parts = loss_fn(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        archive_bytes_proxy,
        apply_eval_roundtrip=True,
        noise_std=0.0,
        gt_pose_batch=None,
        gt_seg_batch=None,
        gt_seg_already_probs=None,
    )
    assert torch.isfinite(loss), "Loss must be finite"
    assert "seg_term" in parts
    assert "pose_term" in parts


def test_pdp_loss_default_off_byte_faithful_to_no_kwargs():
    """Calling without the 3 F3 kwargs == calling with them as None.

    This is the strongest default-OFF guarantee: a trainer that does not
    pass the kwargs at all gets the same loss as one passing None.
    """
    loss_fn, _pose, _seg = _make_loss_fn()
    torch.manual_seed(11)
    rgb_0 = torch.rand(2, 3, 384, 512) * 255.0
    rgb_1 = torch.rand(2, 3, 384, 512) * 255.0
    gt_0 = torch.rand(2, 3, 384, 512) * 255.0
    gt_1 = torch.rand(2, 3, 384, 512) * 255.0
    archive_bytes_proxy = torch.tensor(100_000.0)

    loss_no_kwargs, _ = loss_fn(
        rgb_0.detach().clone(),
        rgb_1.detach().clone(),
        gt_0,
        gt_1,
        archive_bytes_proxy,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )

    loss_none_kwargs, _ = loss_fn(
        rgb_0.detach().clone(),
        rgb_1.detach().clone(),
        gt_0,
        gt_1,
        archive_bytes_proxy,
        apply_eval_roundtrip=True,
        noise_std=0.0,
        gt_pose_batch=None,
        gt_seg_batch=None,
        gt_seg_already_probs=None,
    )

    # Bit-identical (same code path is taken).
    assert torch.equal(loss_no_kwargs, loss_none_kwargs), (
        "Default-OFF (no kwargs) must produce IDENTICAL loss to "
        "explicit None kwargs (both routes hit the GT-forward fallback)"
    )


def test_pdp_loss_cache_path_equivalent_to_gt_forward():
    """The cache path produces seg/pose terms within tolerance of GT-forward.

    Mathematically the cache stores ``posenet.preprocess_input + posenet.forward``
    of the GT pair. The dispatch helper's cache branch consumes this directly;
    the GT-forward branch recomputes it. Both paths must yield identical
    seg+pose terms within float32 numerical tolerance.
    """
    loss_fn, posenet, segnet = _make_loss_fn()
    torch.manual_seed(13)
    rgb_0 = torch.rand(3, 3, 384, 512) * 255.0
    rgb_1 = torch.rand(3, 3, 384, 512) * 255.0
    gt_0 = torch.rand(3, 3, 384, 512) * 255.0
    gt_1 = torch.rand(3, 3, 384, 512) * 255.0
    archive_bytes_proxy = torch.tensor(100_000.0)

    # Build cache from the GT pair (B=3, 2, 3, H, W).
    target_pairs = torch.stack([gt_0, gt_1], dim=1)  # (3, 2, 3, H, W)
    cache = build_gt_scorer_cache(
        target_pixels=target_pairs,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=2,
    )

    # Path A: GT-forward (no cache kwargs).
    loss_gt, parts_gt = loss_fn(
        rgb_0.detach().clone(),
        rgb_1.detach().clone(),
        gt_0,
        gt_1,
        archive_bytes_proxy,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )

    # Path B: cache lookup.
    idx = torch.arange(3, dtype=torch.long)
    gt_pose_batch, gt_seg_batch = cache.lookup(idx, device=torch.device("cpu"))
    loss_cached, parts_cached = loss_fn(
        rgb_0.detach().clone(),
        rgb_1.detach().clone(),
        gt_0,
        gt_1,
        archive_bytes_proxy,
        apply_eval_roundtrip=True,
        noise_std=0.0,
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=cache.seg_already_probs,
    )

    # The two paths should agree on both seg and pose terms.
    assert torch.allclose(parts_gt["seg_term"], parts_cached["seg_term"], atol=1e-5), (
        f"seg_term mismatch: gt={parts_gt['seg_term'].item()} "
        f"vs cached={parts_cached['seg_term'].item()}"
    )
    assert torch.allclose(parts_gt["pose_term"], parts_cached["pose_term"], atol=1e-5), (
        f"pose_term mismatch: gt={parts_gt['pose_term'].item()} "
        f"vs cached={parts_cached['pose_term'].item()}"
    )
    # Total loss within tolerance (same prior contribution; same rate).
    assert torch.allclose(loss_gt, loss_cached, atol=1e-4), (
        f"loss mismatch: gt={loss_gt.item()} vs cached={loss_cached.item()}"
    )


def test_pdp_loss_partial_cache_kwargs_raise():
    """Passing 1-or-2 cache kwargs (not all 3) raises via dispatch helper."""
    loss_fn, _pose, _seg = _make_loss_fn()
    torch.manual_seed(17)
    rgb_0 = torch.rand(2, 3, 384, 512) * 255.0
    rgb_1 = torch.rand(2, 3, 384, 512) * 255.0
    gt_0 = torch.rand(2, 3, 384, 512) * 255.0
    gt_1 = torch.rand(2, 3, 384, 512) * 255.0
    archive_bytes_proxy = torch.tensor(100_000.0)

    fake_pose_batch = torch.zeros(2, 2, 12)

    from tac.substrates.score_aware_common import ScoreAwareScorerContractError

    with pytest.raises(ScoreAwareScorerContractError):
        loss_fn(
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
            archive_bytes_proxy,
            apply_eval_roundtrip=True,
            noise_std=0.0,
            gt_pose_batch=fake_pose_batch,  # alone — partial; should raise
        )


def test_pdp_loss_eval_roundtrip_false_still_raises():
    """The eval_roundtrip=False guard is preserved across the F3 wire-in."""
    loss_fn, _pose, _seg = _make_loss_fn()
    rgb_0 = torch.rand(1, 3, 384, 512) * 255.0
    rgb_1 = torch.rand(1, 3, 384, 512) * 255.0
    gt_0 = torch.rand(1, 3, 384, 512) * 255.0
    gt_1 = torch.rand(1, 3, 384, 512) * 255.0
    archive_bytes_proxy = torch.tensor(100_000.0)
    with pytest.raises(ValueError, match="eval_roundtrip"):
        loss_fn(
            rgb_0, rgb_1, gt_0, gt_1, archive_bytes_proxy,
            apply_eval_roundtrip=False,
        )
