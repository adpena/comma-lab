# SPDX-License-Identifier: MIT
"""Tests for ``tac.training_optimization.scorer_cache`` (O1).

The GT scorer cache is the highest-EV optimization on the audit: it
removes the GT scorer forward from every training step (the target video
+ scorer weights are invariant across epochs). Reference impl exists in
the T1 Balle endtoend trainer; this module hoists it into a canonical
helper any substrate trainer can adopt.

Coverage targets:
- ``GTScorerCache`` dataclass validation (shape / device / matching N)
- ``GTScorerCache.summary_line`` formatting
- ``GTScorerCache.n_pairs`` / ``total_bytes`` properties
- ``GTScorerCache.lookup`` correctness + non_blocking transfer
- ``GTScorerCache.lookup`` out-of-range index error
- ``GTScorerCache.lookup`` empty-idx edge case
- ``GTScorerCache.clear`` frees cache
- ``build_gt_scorer_cache`` shape validation
- ``build_gt_scorer_cache`` end-to-end with fake scorers
- ``build_gt_scorer_cache`` softmax-probs vs logits mode toggle
- ``build_gt_scorer_cache`` cache_chunk_size variation
- ``build_gt_scorer_cache`` restores scorer training mode
- ``build_gt_scorer_cache`` pin_for_cuda explicit + auto
- mathematical equivalence to ``scorer_loss_terms_btchw`` (the central
  correctness claim of this helper)

Per CLAUDE.md "Apples-to-apples evidence discipline" the cache is a
pure-speed primitive verified mathematically identical to the un-cached
scorer-loss path. The equivalence test is the key correctness anchor.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.training_optimization.scorer_cache import (
    GTScorerCache,
    GTScorerCacheError,
    build_gt_scorer_cache,
)

# ---------------------------------------------------------------------------
# Fake scorers for fast deterministic testing
# ---------------------------------------------------------------------------


class _FakePoseNet(nn.Module):
    """Minimal stand-in for the contest PoseNet: 5D pair -> (B, 2, 12) dict."""

    def __init__(self, *, deterministic_seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(deterministic_seed)
        self.linear = nn.Linear(12, 12)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        # Trainer-skeleton contract: 5D -> 4D (B, T*6, H/2, W/2). For our
        # fake test we collapse to a (B, T*12) for the linear.
        b, t, c, h, w = pair_btchw.shape
        # Reduce spatial dims to a 12-channel-per-frame summary.
        pooled = pair_btchw.mean(dim=(3, 4))  # (B, T, C)
        # Expand to 12 dims per frame
        pooled = torch.cat([pooled, pooled, pooled, pooled], dim=-1)
        return pooled.reshape(b, t, 12)

    def forward(self, x_btc: torch.Tensor) -> dict:
        b, t, c = x_btc.shape
        # Apply per-frame linear; reshape so output looks like (B, T, 12).
        flat = x_btc.reshape(b * t, c)
        out = self.linear(flat).reshape(b, t, 12)
        return {"pose": out}


class _FakeSegNet(nn.Module):
    """Minimal stand-in for the contest SegNet: 5D pair -> (B, K, H, W) logits."""

    def __init__(self, *, deterministic_seed: int = 1, num_classes: int = 5) -> None:
        super().__init__()
        torch.manual_seed(deterministic_seed)
        self.conv = nn.Conv2d(3, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        # Contest segnet uses LAST frame only.
        return pair_btchw[:, -1]  # (B, C, H, W)

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


def _make_fake_scorers_and_pairs(
    *,
    n_pairs: int = 8,
    h: int = 16,
    w: int = 16,
) -> tuple[_FakePoseNet, _FakeSegNet, torch.Tensor]:
    posenet = _FakePoseNet()
    segnet = _FakeSegNet()
    posenet.eval()
    segnet.eval()
    torch.manual_seed(42)
    target_pixels = torch.rand((n_pairs, 2, 3, h, w))
    return posenet, segnet, target_pixels


def _patched_scorer_forward_pair(monkeypatch, posenet, segnet) -> None:
    """Monkeypatch the scorer_forward_pair resolver to use fakes.

    The cache module lazy-imports from tac.losses.core; we override the
    resolver returns to a function calling our fake scorers via the
    standard preprocess_input + forward path.
    """

    def fake_scorer_forward_pair(pair_btchw, pn, sn):
        pose_in = pn.preprocess_input(pair_btchw)
        pose_out = pn(pose_in)
        seg_in = sn.preprocess_input(pair_btchw)
        seg_out = sn(seg_in)
        return pose_out, seg_out

    monkeypatch.setattr(
        "tac.training_optimization.scorer_cache._resolve_scorer_forward_pair",
        lambda: fake_scorer_forward_pair,
    )


# ---------------------------------------------------------------------------
# GTScorerCache constructor / validation tests
# ---------------------------------------------------------------------------


def _make_dummy_cache(n: int = 4, k: int = 5, h: int = 8, w: int = 8) -> GTScorerCache:
    return GTScorerCache(
        gt_pose=torch.zeros((n, 2, 12)),
        gt_seg=torch.zeros((n, k, h, w)),
        seg_already_probs=True,
        segmentation_temperature=1.0,
        is_pinned=False,
    )


def test_gtscorercache_accepts_canonical_shapes() -> None:
    cache = _make_dummy_cache(n=4)
    assert cache.n_pairs == 4
    assert cache.gt_pose.shape == (4, 2, 12)


def test_gtscorercache_accepts_flat_posenet_output_shape() -> None:
    cache = GTScorerCache(
        gt_pose=torch.zeros((4, 12)),
        gt_seg=torch.zeros((4, 5, 8, 8)),
        seg_already_probs=True,
        segmentation_temperature=1.0,
        is_pinned=False,
    )
    pose, seg = cache.lookup(torch.tensor([0, 3]), device=torch.device("cpu"))
    assert cache.n_pairs == 4
    assert pose.shape == (2, 12)
    assert seg.shape == (2, 5, 8, 8)


def test_gtscorercache_refuses_wrong_pose_dim() -> None:
    with pytest.raises(GTScorerCacheError, match="2D or 3D"):
        GTScorerCache(
            gt_pose=torch.zeros((4, 12, 1, 1)),  # 4-D, not a PoseNet output
            gt_seg=torch.zeros((4, 5, 8, 8)),
            seg_already_probs=True,
            segmentation_temperature=1.0,
            is_pinned=False,
        )


def test_gtscorercache_refuses_wrong_seg_dim() -> None:
    with pytest.raises(GTScorerCacheError, match="4D"):
        GTScorerCache(
            gt_pose=torch.zeros((4, 2, 12)),
            gt_seg=torch.zeros((4, 5, 8)),  # 3-D, not 4-D
            seg_already_probs=True,
            segmentation_temperature=1.0,
            is_pinned=False,
        )


def test_gtscorercache_refuses_mismatched_n() -> None:
    with pytest.raises(GTScorerCacheError, match="matching pair count"):
        GTScorerCache(
            gt_pose=torch.zeros((4, 2, 12)),
            gt_seg=torch.zeros((6, 5, 8, 8)),  # 6 != 4
            seg_already_probs=True,
            segmentation_temperature=1.0,
            is_pinned=False,
        )


def test_gtscorercache_refuses_non_cpu_tensors() -> None:
    # cache must live on CPU; fake a meta device to avoid touching cuda
    pose_meta = torch.zeros((4, 2, 12), device="meta")
    seg_meta = torch.zeros((4, 5, 8, 8), device="meta")
    with pytest.raises(GTScorerCacheError, match="CPU"):
        GTScorerCache(
            gt_pose=pose_meta,
            gt_seg=seg_meta,
            seg_already_probs=True,
            segmentation_temperature=1.0,
            is_pinned=False,
        )


def test_gtscorercache_exposes_n_pairs_property() -> None:
    cache = _make_dummy_cache(n=12)
    assert cache.n_pairs == 12


def test_gtscorercache_exposes_total_bytes_property() -> None:
    cache = _make_dummy_cache(n=4, k=5, h=8, w=8)
    # pose: 4 * 2 * 12 * 4 = 384
    # seg: 4 * 5 * 8 * 8 * 4 = 5120
    assert cache.total_bytes == 384 + 5120


def test_gtscorercache_summary_line_includes_canonical_fields() -> None:
    cache = _make_dummy_cache(n=4, k=5, h=8, w=8)
    line = cache.summary_line()
    assert "[gt-scorer-cache]" in line
    assert "N=4" in line
    assert "(2, 12)" in line
    assert "(5, 8, 8)" in line
    assert "probs" in line
    assert "saves one frozen" in line


def test_gtscorercache_summary_line_logits_mode() -> None:
    cache = GTScorerCache(
        gt_pose=torch.zeros((4, 2, 12)),
        gt_seg=torch.zeros((4, 5, 8, 8)),
        seg_already_probs=False,
        segmentation_temperature=2.0,
        is_pinned=False,
    )
    line = cache.summary_line()
    assert "logits" in line
    assert "probs" not in line


# ---------------------------------------------------------------------------
# GTScorerCache.lookup tests
# ---------------------------------------------------------------------------


def test_lookup_returns_indexed_tensors_on_device() -> None:
    cache = _make_dummy_cache(n=6, k=5, h=8, w=8)
    # Set unique values so we can verify indexing semantics
    cache.gt_pose = torch.arange(6 * 2 * 12, dtype=torch.float32).reshape(6, 2, 12)
    cache.gt_seg = torch.arange(6 * 5 * 8 * 8, dtype=torch.float32).reshape(
        6, 5, 8, 8
    )
    idx = torch.tensor([2, 4, 1], dtype=torch.long)
    pose, seg = cache.lookup(idx, device=torch.device("cpu"))
    assert pose.shape == (3, 2, 12)
    assert seg.shape == (3, 5, 8, 8)
    # Verify the rows match
    assert torch.equal(pose[0], cache.gt_pose[2])
    assert torch.equal(pose[1], cache.gt_pose[4])
    assert torch.equal(pose[2], cache.gt_pose[1])


def test_lookup_refuses_non_1d_idx() -> None:
    cache = _make_dummy_cache(n=4)
    idx_2d = torch.tensor([[0, 1], [2, 3]])
    with pytest.raises(GTScorerCacheError, match="1-D"):
        cache.lookup(idx_2d, device=torch.device("cpu"))


def test_lookup_refuses_non_integer_idx() -> None:
    cache = _make_dummy_cache(n=4)
    idx_float = torch.tensor([0.0, 1.0])
    with pytest.raises(GTScorerCacheError, match="integer dtype"):
        cache.lookup(idx_float, device=torch.device("cpu"))


def test_lookup_refuses_out_of_range_idx() -> None:
    cache = _make_dummy_cache(n=4)
    idx_out = torch.tensor([0, 1, 5], dtype=torch.long)  # 5 >= 4
    with pytest.raises(GTScorerCacheError, match="out of range"):
        cache.lookup(idx_out, device=torch.device("cpu"))


def test_lookup_refuses_negative_idx() -> None:
    cache = _make_dummy_cache(n=4)
    idx_neg = torch.tensor([-1, 0], dtype=torch.long)
    with pytest.raises(GTScorerCacheError, match="out of range"):
        cache.lookup(idx_neg, device=torch.device("cpu"))


def test_lookup_handles_empty_idx() -> None:
    cache = _make_dummy_cache(n=4)
    idx_empty = torch.tensor([], dtype=torch.long)
    pose, seg = cache.lookup(idx_empty, device=torch.device("cpu"))
    assert pose.shape == (0, 2, 12)
    assert seg.shape == (0, 5, 8, 8)


def test_lookup_accepts_int32_dtype() -> None:
    cache = _make_dummy_cache(n=4)
    idx_i32 = torch.tensor([0, 2], dtype=torch.int32)
    pose, seg = cache.lookup(idx_i32, device=torch.device("cpu"))
    assert pose.shape == (2, 2, 12)


def test_clear_frees_cache() -> None:
    cache = _make_dummy_cache(n=10)
    assert cache.n_pairs == 10
    cache.clear()
    assert cache.n_pairs == 0
    assert cache.total_bytes == 0


# ---------------------------------------------------------------------------
# build_gt_scorer_cache tests
# ---------------------------------------------------------------------------


def test_build_refuses_non_5d_target() -> None:
    posenet, segnet, _ = _make_fake_scorers_and_pairs()
    target_4d = torch.zeros((4, 3, 16, 16))
    with pytest.raises(GTScorerCacheError, match="5-D"):
        build_gt_scorer_cache(
            target_pixels=target_4d,
            posenet=posenet,
            segnet=segnet,
            device=torch.device("cpu"),
        )


def test_build_refuses_wrong_t_dim() -> None:
    posenet, segnet, _ = _make_fake_scorers_and_pairs()
    target_t3 = torch.zeros((4, 3, 3, 16, 16))  # T=3 != 2
    with pytest.raises(GTScorerCacheError, match="T=2"):
        build_gt_scorer_cache(
            target_pixels=target_t3,
            posenet=posenet,
            segnet=segnet,
            device=torch.device("cpu"),
        )


def test_build_refuses_wrong_c_dim() -> None:
    posenet, segnet, _ = _make_fake_scorers_and_pairs()
    target_c4 = torch.zeros((4, 2, 4, 16, 16))  # C=4 != 3
    with pytest.raises(GTScorerCacheError, match="C=3"):
        build_gt_scorer_cache(
            target_pixels=target_c4,
            posenet=posenet,
            segnet=segnet,
            device=torch.device("cpu"),
        )


def test_build_refuses_zero_chunk_size() -> None:
    posenet, segnet, target = _make_fake_scorers_and_pairs()
    with pytest.raises(GTScorerCacheError, match="cache_chunk_size"):
        build_gt_scorer_cache(
            target_pixels=target,
            posenet=posenet,
            segnet=segnet,
            device=torch.device("cpu"),
            cache_chunk_size=0,
        )


def test_build_end_to_end_with_fake_scorers(monkeypatch) -> None:
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=8)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)

    cache = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=4,
    )
    assert cache.n_pairs == 8
    assert cache.gt_pose.shape == (8, 2, 12)
    assert cache.gt_seg.shape == (8, 5, 16, 16)
    assert cache.seg_already_probs is True
    assert cache.segmentation_temperature == 1.0
    # CPU cache, no pinning (pin_for_cuda auto-False for cpu device)
    assert cache.is_pinned is False


def test_build_with_temperature_not_one_stores_logits(monkeypatch) -> None:
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)

    cache = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=2.0,
        cache_chunk_size=2,
    )
    assert cache.seg_already_probs is False
    assert cache.segmentation_temperature == 2.0


def test_build_chunk_size_independence(monkeypatch) -> None:
    """Cache contents must be identical regardless of chunk_size."""
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=8)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)

    cache_chunk1 = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        cache_chunk_size=1,
    )
    cache_chunk8 = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        cache_chunk_size=8,
    )
    assert torch.allclose(cache_chunk1.gt_pose, cache_chunk8.gt_pose)
    assert torch.allclose(cache_chunk1.gt_seg, cache_chunk8.gt_seg)


def test_build_restores_scorer_training_mode(monkeypatch) -> None:
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)

    # Put both scorers in training mode first
    posenet.train()
    segnet.train()
    assert posenet.training is True
    assert segnet.training is True

    build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
    )
    # After build, scorers should be back in training mode (the build
    # forces eval but restores afterwards).
    assert posenet.training is True
    assert segnet.training is True


def test_build_keeps_eval_mode_when_originally_eval(monkeypatch) -> None:
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)
    posenet.eval()
    segnet.eval()

    build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
    )
    assert posenet.training is False
    assert segnet.training is False


def test_build_pin_for_cuda_explicit_false(monkeypatch) -> None:
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)
    cache = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        pin_for_cuda=False,
    )
    assert cache.is_pinned is False


def test_build_handles_target_already_on_target_device(monkeypatch) -> None:
    # target_pixels already on the build device (cpu in tests) should
    # work without device-copy errors.
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)
    target_cpu = target.to(torch.device("cpu"))
    cache = build_gt_scorer_cache(
        target_pixels=target_cpu,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
    )
    assert cache.n_pairs == 4


def test_lookup_matches_direct_scorer_forward(monkeypatch) -> None:
    """Central correctness anchor: cache lookup == direct scorer forward.

    The whole optimization is that the cache lookup returns the same
    tensors a direct GT scorer forward would return. This test verifies
    that equivalence for both the pose and seg outputs.
    """
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=6)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)

    cache = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=1.0,
        cache_chunk_size=3,
    )

    # Direct scorer forward on a subset.
    idx = torch.tensor([1, 3, 5], dtype=torch.long)
    direct_pairs = target[idx].contiguous()
    pose_in = posenet.preprocess_input(direct_pairs)
    direct_pose = posenet(pose_in)["pose"]
    seg_in = segnet.preprocess_input(direct_pairs)
    direct_seg = F.softmax(segnet(seg_in), dim=1)

    cache_pose, cache_seg = cache.lookup(idx, device=torch.device("cpu"))

    assert torch.allclose(cache_pose, direct_pose, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cache_seg, direct_seg, atol=1e-6, rtol=1e-6)


def test_lookup_seg_logits_match_direct_scorer_forward(monkeypatch) -> None:
    """Logits-mode (temperature != 1.0) cache must match direct logits."""
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)
    _patched_scorer_forward_pair(monkeypatch, posenet, segnet)

    cache = build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
        segmentation_temperature=2.0,
    )

    idx = torch.tensor([0, 2], dtype=torch.long)
    direct_pairs = target[idx].contiguous()
    seg_in = segnet.preprocess_input(direct_pairs)
    direct_logits = segnet(seg_in).float()

    _, cache_seg = cache.lookup(idx, device=torch.device("cpu"))

    assert torch.allclose(cache_seg, direct_logits, atol=1e-6, rtol=1e-6)


def test_build_no_grad_during_forward(monkeypatch) -> None:
    """The build must run under no_grad (the GT forward is gradient-free)."""
    posenet, segnet, target = _make_fake_scorers_and_pairs(n_pairs=4)

    # Sentinel: capture whether grad mode was on during the scorer call.
    grad_seen: list[bool] = []

    def trace_forward_pair(pair_btchw, pn, sn):
        grad_seen.append(torch.is_grad_enabled())
        pose_in = pn.preprocess_input(pair_btchw)
        pose_out = pn(pose_in)
        seg_in = sn.preprocess_input(pair_btchw)
        seg_out = sn(seg_in)
        return pose_out, seg_out

    monkeypatch.setattr(
        "tac.training_optimization.scorer_cache._resolve_scorer_forward_pair",
        lambda: trace_forward_pair,
    )
    build_gt_scorer_cache(
        target_pixels=target,
        posenet=posenet,
        segnet=segnet,
        device=torch.device("cpu"),
    )
    # Every chunk's forward must have run under no_grad.
    assert grad_seen, "scorer_forward_pair should have been called"
    assert all(g is False for g in grad_seen)
