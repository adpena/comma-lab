# SPDX-License-Identifier: MIT
"""Regression tests for `tac.losses` numerical contracts.

Centerpiece: `kl_distill_segnet_only` reduction was `batchmean` on a
(B, 5, H, W) SegNet logit tensor — under-divides by H × W (=196,608 for
384 × 512). Council forensics 2026-04-27 (findings.md
"## 2026-04-27 Council forensics: Lane G — really dead, or bugged?")
identified the bug; this file pins the per-pixel-per-class mean reduction
that mirrors `kl_distill_scorer_loss` (line 622+646).

These tests are the structural gate: any future refactor that re-introduces
the under-divided reduction will fail `test_kl_distill_segnet_only_reduction_is_per_pixel_mean`.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.losses import kl_distill_segnet_only, segnet_uncertainty_weighted_loss


# ── Test stub: minimal SegNet-shaped module ─────────────────────────────────
#
# `kl_distill_segnet_only` calls `segnet.preprocess_input(fx)` then
# `segnet(input)`. The contract: input is (B, T, C, H, W) (per
# `_hwc_to_chw`), preprocess slices `[:, -1, ...]` and bilinearly resizes
# to (512, 384), output is (B, num_classes, 384, 512) logits.
#
# For a deterministic numerical test we replace SegNet with a shape-faithful
# stub that returns reproducible logits for a given input shape. The exact
# logit values are irrelevant — we are pinning the REDUCTION, which is a
# property of the loss helper, not of the scorer.


class _ShapeStubSegNet(nn.Module):
    """Returns deterministic `(B, num_classes, out_h, out_w)` logits.

    `preprocess_input((B, T, C, H, W))` returns `(B, C, out_h, out_w)`
    after taking the last frame and bilinearly resizing — same contract
    as upstream SegNet. The forward pass returns shape-correct logits
    seeded by the input tensor for reproducibility.
    """

    def __init__(self, num_classes: int = 5, out_h: int = 384, out_w: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.out_h = out_h
        self.out_w = out_w
        # A trivial conv so the module has a graph (kl_distill calls
        # backward via the caller; here we only forward).
        self.proj = nn.Conv2d(3, num_classes, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) — take last frame + resize to (out_h, out_w).
        last = x[:, -1, ...]  # (B, C, H, W)
        return F.interpolate(
            last.float(), size=(self.out_h, self.out_w),
            mode="bilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_classes, H, W) logits
        return self.proj(x)


class _RecordingSegNet(nn.Module):
    """Records SegNet input shapes and enforces the 3-channel scorer contract."""

    def __init__(self, num_classes: int = 5, out_h: int = 4, out_w: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.out_h = out_h
        self.out_w = out_w
        self.preprocess_shapes: list[tuple[int, ...]] = []
        self.forward_shapes: list[tuple[int, ...]] = []

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_shapes.append(tuple(x.shape))
        assert x.ndim == 5
        assert x.shape[2] == 3
        last = x[:, -1, ...]
        return F.interpolate(
            last.float(), size=(self.out_h, self.out_w),
            mode="bilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(x.shape))
        assert x.ndim == 4
        assert x.shape[1] == 3
        return x.new_zeros((x.shape[0], self.num_classes, x.shape[2], x.shape[3]))


def _make_pair(B: int, H: int, W: int, *, seed: int = 0) -> torch.Tensor:
    """Synthesize a (B, T=2, H, W, C=3) HWC pair in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, 2, H, W, 3, generator=g)


def test_segnet_uncertainty_weighted_loss_preserves_3_channel_bchw_for_segnet():
    """Regression for H-V3: do not index RGB channel 0 as the time axis."""
    B, H, W = 2, 8, 10
    rendered = torch.rand(B, H, W, 3)
    gt = torch.rand(B, H, W, 3)
    segnet = _RecordingSegNet(out_h=4, out_w=5)

    loss = segnet_uncertainty_weighted_loss(rendered, gt, segnet)

    assert loss.ndim == 0
    assert segnet.preprocess_shapes == [(B, 1, 3, H, W)]
    assert segnet.forward_shapes == [(B, 3, 4, 5)]


# ─── Reduction correctness ──────────────────────────────────────────────────


class TestKlDistillSegnetOnlyReduction:
    """Bug class: `F.kl_div(..., reduction="batchmean")` on (B, C, H, W)
    silently under-divides by H × W. The fix mirrors `kl_distill_scorer_loss`
    line 622+646: `reduction="none" → .sum(dim=1) → .mean()`.
    """

    def test_kl_distill_segnet_only_reduction_is_per_pixel_mean(self):
        """Magnitude must be in the per-pixel-mean range (~1e-2 to ~1.0 nats),
        NOT the per-batch-sum range (~1e3 to ~1e5).

        Pre-fix, `kl.item()` on a (1, 2, 384, 512, 3) → SegNet (1, 5, 384, 512)
        pair sat at ~24,485 nats (real data) or up to ~150,000 nats (random
        logits). Post-fix, the same input must score in the per-pixel range.
        """
        B, H, W = 1, 384, 512
        torch.manual_seed(42)
        filtered = _make_pair(B, H, W, seed=1)
        gt = _make_pair(B, H, W, seed=2)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        kl, kl_val = kl_distill_segnet_only(filtered, gt, segnet, temperature=2.0)
        # Acceptance band: per-pixel-per-class mean × T² (T=2 → ×4) must
        # land in [1e-3, 5.0]. The pre-fix value of ~24,485 is 4-5 orders
        # of magnitude outside this band — so any future regression that
        # accidentally re-introduces `batchmean` will land outside and fail.
        assert 1e-3 < kl_val < 5.0, (
            f"kl_distill_segnet_only returned kl={kl_val:.4f} on a "
            f"(1, 5, 384, 512) SegNet logit tensor. Expected per-pixel-mean "
            f"magnitude (~1e-2 to ~1.0 nats × T²). The pre-2026-04-27 "
            f"`reduction='batchmean'` bug produced values 100,000-200,000× "
            f"larger (the under-division factor of H × W = 196,608). See "
            f"findings.md \"Lane G — really dead, or bugged?\" for the "
            f"math + memory `feedback_unit_error_masquerading_as_small_signal`."
        )
        # Tensor and float must agree.
        assert pytest.approx(kl.item(), rel=1e-6) == kl_val

    @pytest.mark.parametrize("bad_temperature", [0.0, -1.0, math.inf, -math.inf, math.nan, True])
    def test_kl_distill_segnet_only_rejects_invalid_temperature(self, bad_temperature):
        """KL helpers must fail before dividing logits by invalid temperatures."""
        filtered = _make_pair(1, 8, 8, seed=11)
        gt = _make_pair(1, 8, 8, seed=12)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=4, out_w=5)

        with pytest.raises(ValueError, match="temperature must be a finite positive number"):
            kl_distill_segnet_only(filtered, gt, segnet, temperature=bad_temperature)

    def test_kl_distill_segnet_only_matches_kl_distill_scorer_loss_pattern(self):
        """The two helpers compute the same per-position quantity. After
        the fix, both apply per-pixel-per-class mean. So when given the
        same inputs, `kl_distill_segnet_only` should produce a value
        within numerical tolerance of the SegNet-KL portion of
        `kl_distill_scorer_loss`.

        We replicate the canonical pattern inline to avoid coupling to
        kl_distill_scorer_loss's particular weighting (it returns
        `100 * seg_kl + sqrt(10 * pose_dist)`; we only want seg_kl for
        a structural-equivalence check).
        """
        B, H, W = 1, 384, 512
        torch.manual_seed(42)
        filtered = _make_pair(B, H, W, seed=10)
        gt = _make_pair(B, H, W, seed=20)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        T = 2.0
        # Inline canonical reduction (mirrors kl_distill_scorer_loss
        # lines 618-646 exactly):
        fs_in = segnet.preprocess_input(_chw_pair(filtered))
        gs_in = segnet.preprocess_input(_chw_pair(gt))
        fs_logits = segnet(fs_in)
        gs_logits = segnet(gs_in)
        log_p = F.log_softmax(fs_logits / T, dim=1)
        q = F.softmax(gs_logits / T, dim=1)
        kl_canonical = F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)

        kl_helper, _ = kl_distill_segnet_only(filtered, gt, segnet, temperature=T)
        # Allow tiny numerical drift from operation ordering, but the
        # values MUST be in the same magnitude class. Pre-fix the helper
        # value was ~196,608× larger.
        assert pytest.approx(kl_canonical.item(), rel=1e-4) == kl_helper.item(), (
            f"kl_distill_segnet_only={kl_helper.item():.6f} disagrees with "
            f"the canonical per-pixel-per-class mean = {kl_canonical.item():.6f}. "
            f"Both should compute the same reduction (mirror of "
            f"kl_distill_scorer_loss lines 622+646)."
        )

    def test_kl_distill_segnet_only_magnitude_invariant_to_resolution(self):
        """The pre-fix bug made `kl.item()` scale linearly with H × W
        because `batchmean` divides only by B. The fix must produce
        magnitude-stable values across SegNet output resolutions —
        as long as the underlying per-position KL distribution is the
        same, the mean should be (approximately) the same.

        We can't change the upstream SegNet eval resolution (384×512 is
        the contract), but we can swap our stub's `out_h, out_w` and
        verify that the helper's magnitude does NOT scale with resolution.
        Pre-fix: 96×128 → ~9,400; 384×512 → ~150,000 (linear scaling).
        Post-fix: both → ~0.7-0.8.
        """
        B, H, W = 1, 384, 512
        filtered = _make_pair(B, H, W, seed=3)
        gt = _make_pair(B, H, W, seed=4)
        # Use the SAME stub Conv2d weight init (seed) so logits are
        # comparable across resolutions.
        torch.manual_seed(0)
        segnet_low = _ShapeStubSegNet(num_classes=5, out_h=96, out_w=128)
        torch.manual_seed(0)
        segnet_hi = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        kl_low, _ = kl_distill_segnet_only(filtered, gt, segnet_low, temperature=2.0)
        kl_hi, _ = kl_distill_segnet_only(filtered, gt, segnet_hi, temperature=2.0)
        # Both must land in the per-pixel-mean band. Allow a factor of
        # ~3 difference (different downsampling smooths the logit field
        # differently); the bug would produce a factor of ~16 (96×128 vs
        # 384×512 = 16x).
        ratio = max(kl_hi.item(), kl_low.item()) / max(min(kl_hi.item(), kl_low.item()), 1e-9)
        assert ratio < 4.0, (
            f"kl_distill_segnet_only magnitude scales with output "
            f"resolution: low={kl_low.item():.4f}, hi={kl_hi.item():.4f}, "
            f"ratio={ratio:.2f}. Pre-fix this ratio was ~16 (=384×512 / "
            f"96×128), which is the smoking gun for the `batchmean` bug. "
            f"Post-fix both should be in the same magnitude class."
        )
        # And both must be in the per-pixel band.
        assert 1e-3 < kl_low.item() < 5.0, kl_low.item()
        assert 1e-3 < kl_hi.item() < 5.0, kl_hi.item()

    def test_kl_distill_segnet_only_t_squared_scaling_preserved(self):
        """Hinton 2015 §2.1 T² scaling must still be applied (caller
        contract). Doubling T (2 → 4) should multiply the returned
        magnitude by ~4 (not 1, not 16).
        """
        B, H, W = 1, 384, 512
        filtered = _make_pair(B, H, W, seed=5)
        gt = _make_pair(B, H, W, seed=6)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        kl_t2, _ = kl_distill_segnet_only(filtered, gt, segnet, temperature=2.0)
        kl_t4, _ = kl_distill_segnet_only(filtered, gt, segnet, temperature=4.0)
        # KL itself shrinks with T (smoother distributions); T² scaling
        # compensates. Net: kl_t4 / kl_t2 should be roughly 1.0 (per
        # Hinton's derivation), with some drift. Definitely not 0.25
        # (no T² scaling) and not 16 (T⁴ scaling).
        ratio = kl_t4.item() / max(kl_t2.item(), 1e-9)
        assert 0.3 < ratio < 3.0, (
            f"T² scaling appears broken: kl(T=4) / kl(T=2) = {ratio:.3f}. "
            f"Expected roughly ~1.0 (Hinton 2015 §2.1: T² scaling cancels "
            f"the natural KL compression at high T). 0.25 means T² was "
            f"dropped; 16 means T⁴ was applied."
        )

    def test_kl_distill_segnet_only_returns_nonnegative(self):
        """KL divergence is non-negative by definition. Sanity check
        the helper does not accidentally negate or sign-flip."""
        B, H, W = 1, 384, 512
        filtered = _make_pair(B, H, W, seed=7)
        gt = _make_pair(B, H, W, seed=8)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        kl, kl_val = kl_distill_segnet_only(filtered, gt, segnet, temperature=2.0)
        assert kl.item() >= 0, kl.item()
        assert kl_val >= 0, kl_val

    def test_kl_distill_segnet_only_zero_when_inputs_identical(self):
        """KL(P || P) = 0. When filtered == gt, the helper should return
        ~0 (within numerical noise). Confirms the gradient direction is
        toward distribution-matching, not away from it."""
        B, H, W = 1, 384, 512
        pair = _make_pair(B, H, W, seed=9)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        kl, _ = kl_distill_segnet_only(pair, pair, segnet, temperature=2.0)
        # Per-pixel-per-class KL with identical inputs is < 1e-6 typically.
        # The pre-fix bug would still return 0 here (0 × 196,608 = 0), so
        # this test does NOT discriminate the bug — but it does pin the
        # mathematical contract that KL(P||P) = 0.
        assert kl.item() < 1e-4, f"KL(P||P)={kl.item()} should be ~0"

    def test_kl_distill_segnet_only_gradients_flow_to_filtered_only(self):
        """Per docstring: filtered branch is the student (gradients flow);
        gt branch is the teacher (no gradients). Verifies the
        `with torch.no_grad():` wrapper around `gs_logits` is intact.
        """
        B, H, W = 1, 384, 512
        torch.manual_seed(11)
        filtered = _make_pair(B, H, W, seed=11).requires_grad_(True)
        gt = _make_pair(B, H, W, seed=12).requires_grad_(True)
        segnet = _ShapeStubSegNet(num_classes=5, out_h=384, out_w=512)
        kl, _ = kl_distill_segnet_only(filtered, gt, segnet, temperature=2.0)
        kl.backward()
        assert filtered.grad is not None and filtered.grad.abs().sum() > 0, \
            "filtered branch must receive gradient (it is the student)"
        assert gt.grad is None or gt.grad.abs().sum() == 0, \
            "gt branch must NOT receive gradient (it is the teacher)"


def _chw_pair(pair_hwc: torch.Tensor) -> torch.Tensor:
    """Mirror of tac.losses._hwc_to_chw for inline test use."""
    return pair_hwc.permute(0, 1, 4, 2, 3).contiguous().float()
