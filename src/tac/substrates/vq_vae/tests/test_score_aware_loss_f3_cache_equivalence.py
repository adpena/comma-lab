# SPDX-License-Identifier: MIT
"""SELFCOMP-2 (R2 MEDIUM, 2026-05-15): vq_vae F3 cache-equivalence test.

Sister-pattern test mirroring
``src/tac/substrates/pretrained_driving_prior/tests/test_score_aware_loss_f3_kwargs.py::
test_pdp_loss_cache_path_equivalent_to_gt_forward``.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
the F3 GTScorerCache wire-in lives on TWO surfaces (PDP + vq_vae) but the
mathematical-equivalence test was only on PDP. R2 SELFCOMP voice flagged the
asymmetry: a cache HIT with the wrong key would silently produce wrong
score on vq_vae and the existing TEXT-MATCHING tests
(``test_f3_backport_vqvae_pdp_wired.py``) would not catch it.

This test pins cache-vs-GT-forward equivalence on vq_vae's
``VqVaeScoreAwareLoss``, asserting:

* The cache lookup path produces seg/pose terms within float32 numerical
  tolerance of the GT-forward path (``atol=1e-5``).
* Total loss is within tolerance (``atol=1e-4``).
* Default-OFF (no cache kwargs) is byte-identical to passing ``None`` for
  all 3 cache kwargs.
* Calling without ``apply_eval_roundtrip`` raises (preserved guard).

Cross-refs: ``feedback_recursive_review_r2_wave_a_*`` SELFCOMP-2 +
``feedback_r2_medium_fix_wave_selfcomp_mackay_landed_20260515.md`` +
PDP sister test (the pattern this mirrors).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.training_optimization import build_gt_scorer_cache


class _FakePoseNet(nn.Module):
    """Tiny PoseNet stub that satisfies the (B, T, 12) -> {'pose': ...} contract."""

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(12, 12)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        b, t, _c, _h, _w = pair_btchw.shape
        pooled = pair_btchw.mean(dim=(3, 4))  # (B, T, 3)
        # Expand 3 channels -> 12 channels via repeat to match canonical PoseNet shape.
        pooled = torch.cat([pooled, pooled, pooled, pooled], dim=-1)
        return pooled.reshape(b, t, 12)

    def forward(self, x_btc: torch.Tensor) -> dict:
        b, t, c = x_btc.shape
        flat = x_btc.reshape(b * t, c)
        out = self.linear(flat).reshape(b, t, 12)
        return {"pose": out}


class _FakeSegNet(nn.Module):
    """Tiny SegNet stub returning per-pixel logits at last-frame resolution."""

    def __init__(self, *, seed: int = 1, num_classes: int = 5) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        # Canonical SegNet: last frame only.
        return pair_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


def _make_loss_fn():
    """Build a real VqVaeScoreAwareLoss with fake scorers."""
    from tac.substrates.vq_vae.score_aware_loss import (
        ScoreAwareLossWeights,
        VqVaeScoreAwareLoss,
    )

    posenet = _FakePoseNet()
    segnet = _FakeSegNet()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)
    posenet.eval()
    segnet.eval()

    weights = ScoreAwareLossWeights()
    loss_fn = VqVaeScoreAwareLoss(
        seg_scorer=segnet,
        pose_scorer=posenet,
        weights=weights,
    )
    return loss_fn, posenet, segnet


def _make_inputs(batch: int, seed: int):
    torch.manual_seed(seed)
    rgb_0 = torch.rand(batch, 3, 384, 512)
    rgb_1 = torch.rand(batch, 3, 384, 512)
    gt_0 = torch.rand(batch, 3, 384, 512) * 255.0
    gt_1 = torch.rand(batch, 3, 384, 512) * 255.0
    archive_bytes_proxy = torch.tensor(100_000.0)
    commitment = torch.tensor(0.1, requires_grad=True)
    return rgb_0, rgb_1, gt_0, gt_1, archive_bytes_proxy, commitment


def test_vq_vae_loss_accepts_f3_kwargs_no_error():
    """Calling forward with the 3 F3 kwargs as None succeeds (default-OFF path)."""
    loss_fn, _pose, _seg = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=2, seed=7)
    rgb_0.requires_grad_(True)
    rgb_1.requires_grad_(True)
    loss, parts = loss_fn(
        rgb_0,
        rgb_1,
        gt_0,
        gt_1,
        bp,
        commit,
        apply_eval_roundtrip=True,
        noise_std=0.0,
        gt_pose_batch=None,
        gt_seg_batch=None,
        gt_seg_already_probs=None,
    )
    assert torch.isfinite(loss), f"loss must be finite; got {loss.item()}"
    assert "seg_term" in parts
    assert "pose_term" in parts
    assert "commitment_term" in parts
    assert "rate_term" in parts


def test_vq_vae_loss_default_off_byte_faithful_to_no_kwargs():
    """Calling without the 3 F3 kwargs == calling with them as None.

    Strongest default-OFF guarantee: a trainer that does not pass the
    kwargs at all gets the same loss as one passing None (both routes
    hit the GT-forward fallback).
    """
    loss_fn, _pose, _seg = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=2, seed=11)

    loss_no_kwargs, _ = loss_fn(
        rgb_0.detach().clone(),
        rgb_1.detach().clone(),
        gt_0,
        gt_1,
        bp,
        commit,
        apply_eval_roundtrip=True,
        noise_std=0.0,
    )
    loss_none_kwargs, _ = loss_fn(
        rgb_0.detach().clone(),
        rgb_1.detach().clone(),
        gt_0,
        gt_1,
        bp,
        commit,
        apply_eval_roundtrip=True,
        noise_std=0.0,
        gt_pose_batch=None,
        gt_seg_batch=None,
        gt_seg_already_probs=None,
    )
    assert torch.equal(loss_no_kwargs, loss_none_kwargs), (
        "Default-OFF (no kwargs) must produce IDENTICAL loss to explicit "
        "None kwargs (both routes hit the GT-forward fallback)"
    )


def test_vq_vae_loss_cache_path_equivalent_to_gt_forward():
    """The cache path produces seg/pose terms within tolerance of GT-forward.

    SELFCOMP-2 sister of PDP's
    ``test_pdp_loss_cache_path_equivalent_to_gt_forward``. Mathematically
    the cache stores ``posenet.preprocess_input + posenet.forward`` of the
    GT pair; the dispatch helper's cache branch consumes this directly;
    the GT-forward branch recomputes it. Both paths MUST yield identical
    seg+pose terms within float32 numerical tolerance.

    A cache HIT with the wrong key would silently produce a non-equivalent
    seg/pose term. This test catches that bug class for vq_vae.
    """
    loss_fn, posenet, segnet = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=3, seed=13)

    # Build cache from the GT pair (B=3, 2, 3, H, W).
    target_pairs = torch.stack([gt_0, gt_1], dim=1)
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
        bp,
        commit,
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
        bp,
        commit,
        apply_eval_roundtrip=True,
        noise_std=0.0,
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=cache.seg_already_probs,
    )

    # The two paths should agree on both seg and pose terms.
    assert torch.allclose(
        parts_gt["seg_term"], parts_cached["seg_term"], atol=1e-5
    ), (
        f"seg_term mismatch: gt={parts_gt['seg_term'].item()} "
        f"vs cached={parts_cached['seg_term'].item()}"
    )
    assert torch.allclose(
        parts_gt["pose_term"], parts_cached["pose_term"], atol=1e-5
    ), (
        f"pose_term mismatch: gt={parts_gt['pose_term'].item()} "
        f"vs cached={parts_cached['pose_term'].item()}"
    )
    # Commitment + rate terms cancel (both calls share inputs); total loss
    # should be within tolerance.
    assert torch.allclose(loss_gt, loss_cached, atol=1e-4), (
        f"loss mismatch: gt={loss_gt.item()} vs cached={loss_cached.item()}"
    )


def test_vq_vae_loss_cache_equivalence_at_batch_size_1():
    """Cache equivalence holds at batch_size=1 (boundary)."""
    loss_fn, posenet, segnet = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=1, seed=17)

    target_pairs = torch.stack([gt_0, gt_1], dim=1)
    cache = build_gt_scorer_cache(
        target_pixels=target_pairs,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=1,
    )

    loss_gt, parts_gt = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    idx = torch.arange(1, dtype=torch.long)
    gt_pose_batch, gt_seg_batch = cache.lookup(idx, device=torch.device("cpu"))
    loss_cached, parts_cached = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=cache.seg_already_probs,
    )
    assert torch.allclose(parts_gt["seg_term"], parts_cached["seg_term"], atol=1e-5)
    assert torch.allclose(parts_gt["pose_term"], parts_cached["pose_term"], atol=1e-5)


def test_vq_vae_loss_cache_equivalence_at_larger_batch():
    """Cache equivalence holds at batch_size=4 (typical training batch)."""
    loss_fn, posenet, segnet = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=4, seed=19)

    target_pairs = torch.stack([gt_0, gt_1], dim=1)
    cache = build_gt_scorer_cache(
        target_pixels=target_pairs,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=2,
    )

    loss_gt, parts_gt = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    idx = torch.arange(4, dtype=torch.long)
    gt_pose_batch, gt_seg_batch = cache.lookup(idx, device=torch.device("cpu"))
    loss_cached, parts_cached = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=cache.seg_already_probs,
    )
    assert torch.allclose(parts_gt["seg_term"], parts_cached["seg_term"], atol=1e-5)
    assert torch.allclose(parts_gt["pose_term"], parts_cached["pose_term"], atol=1e-5)


def test_vq_vae_loss_partial_cache_kwargs_raise():
    """Passing 1-or-2 cache kwargs (not all 3) raises via dispatch helper."""
    from tac.substrates.score_aware_common import ScoreAwareScorerContractError

    loss_fn, _pose, _seg = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=2, seed=23)
    fake_pose_batch = torch.zeros(2, 2, 12)

    with pytest.raises(ScoreAwareScorerContractError):
        loss_fn(
            rgb_0, rgb_1, gt_0, gt_1, bp, commit,
            apply_eval_roundtrip=True, noise_std=0.0,
            gt_pose_batch=fake_pose_batch,  # alone — partial; must raise
        )


def test_vq_vae_loss_eval_roundtrip_false_still_raises():
    """The eval_roundtrip=False guard is preserved across the F3 wire-in."""
    loss_fn, _pose, _seg = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=1, seed=29)
    with pytest.raises(ValueError, match="eval_roundtrip"):
        loss_fn(
            rgb_0, rgb_1, gt_0, gt_1, bp, commit,
            apply_eval_roundtrip=False,
        )


def test_vq_vae_loss_cache_equivalence_seg_term_NOT_zero():
    """Both paths produce non-zero seg_term (sanity guard against silent zeros).

    Without this check the equivalence test could pass trivially with both
    sides returning 0.0 — useless. Confirms the test exercises real scorer
    forward + computes a real seg disagreement signal.
    """
    loss_fn, posenet, segnet = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=2, seed=31)
    target_pairs = torch.stack([gt_0, gt_1], dim=1)
    cache = build_gt_scorer_cache(
        target_pixels=target_pairs,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=2,
    )
    loss_gt, parts_gt = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    idx = torch.arange(2, dtype=torch.long)
    gt_pose_batch, gt_seg_batch = cache.lookup(idx, device=torch.device("cpu"))
    _, parts_cached = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=cache.seg_already_probs,
    )
    assert parts_gt["seg_term"].item() > 0.0, (
        "seg_term must be > 0 (random rgb vs random gt → segmap disagreement)"
    )
    assert parts_cached["seg_term"].item() > 0.0, (
        "cached seg_term must be > 0 (same expectation)"
    )


def test_vq_vae_loss_cache_equivalence_pose_term_NOT_zero():
    """Both paths produce non-zero pose_term (sanity guard)."""
    loss_fn, posenet, segnet = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=2, seed=37)
    target_pairs = torch.stack([gt_0, gt_1], dim=1)
    cache = build_gt_scorer_cache(
        target_pixels=target_pairs,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=2,
    )
    loss_gt, parts_gt = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    idx = torch.arange(2, dtype=torch.long)
    gt_pose_batch, gt_seg_batch = cache.lookup(idx, device=torch.device("cpu"))
    _, parts_cached = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
        gt_pose_batch=gt_pose_batch,
        gt_seg_batch=gt_seg_batch,
        gt_seg_already_probs=cache.seg_already_probs,
    )
    assert parts_gt["pose_term"].item() > 0.0, (
        "pose_term must be > 0 (random rgb vs random gt → posenet disagreement)"
    )
    assert parts_cached["pose_term"].item() > 0.0


def test_vq_vae_loss_cache_lookup_distinct_from_uncached_indices():
    """Cache lookup with PERMUTED indices produces DIFFERENT seg/pose terms
    than the GT-forward path, proving the cache key actually matters.

    If this test fails (i.e., permuted indices still agree with GT-forward),
    the cache mechanism is broken / the kwargs are being ignored — exactly
    the bug class SELFCOMP-2 raised.
    """
    loss_fn, posenet, segnet = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=3, seed=41)

    target_pairs = torch.stack([gt_0, gt_1], dim=1)
    cache = build_gt_scorer_cache(
        target_pixels=target_pairs,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=2,
    )

    _, parts_gt = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
    )

    # Use REVERSED indices — cache returns wrong-key entries; the seg/pose
    # terms should NOT match the GT-forward terms.
    permuted = torch.tensor([2, 1, 0], dtype=torch.long)
    gt_pose_perm, gt_seg_perm = cache.lookup(permuted, device=torch.device("cpu"))
    _, parts_perm = loss_fn(
        rgb_0.detach().clone(), rgb_1.detach().clone(),
        gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
        gt_pose_batch=gt_pose_perm,
        gt_seg_batch=gt_seg_perm,
        gt_seg_already_probs=cache.seg_already_probs,
    )
    # At least ONE of seg/pose should differ — different cache keys means
    # different seg/pose terms unless the inputs happen to be invariant.
    seg_diff = (parts_gt["seg_term"] - parts_perm["seg_term"]).abs().item()
    pose_diff = (parts_gt["pose_term"] - parts_perm["pose_term"]).abs().item()
    assert seg_diff > 1e-6 or pose_diff > 1e-6, (
        f"Permuted cache lookup produced bit-identical seg+pose to GT-forward "
        f"— the cache kwargs are being ignored! seg_diff={seg_diff} "
        f"pose_diff={pose_diff}. This is exactly the bug class SELFCOMP-2 "
        f"(R2 MEDIUM, 2026-05-15) was raised against."
    )


def test_vq_vae_loss_returns_parts_dict_shape():
    """Returned ``parts`` dict has the canonical keys for trainer logging."""
    loss_fn, _pose, _seg = _make_loss_fn()
    rgb_0, rgb_1, gt_0, gt_1, bp, commit = _make_inputs(batch=2, seed=43)
    _, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bp, commit,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert "rate_term" in parts
    assert "seg_term" in parts
    assert "pose_term" in parts
    assert "commitment_term" in parts
    assert "loss_total" in parts
