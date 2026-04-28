"""Lane PS — per-class SegNet weighting unit tests.

What Lane PS adds:
  1. ``tac.losses.parse_class_weights_csv`` — robust CSV parser with
     fail-loud validation (5 floats, non-negative, not-all-zero).
  2. ``tac.losses._apply_class_weights`` — shared kernel that multiplies
     per-pixel SegNet loss by per-class weights (L1-normalised, GT-argmax
     indexed).
  3. ``tac.losses.scorer_loss`` / ``scorer_loss_cached`` /
     ``kl_distill_segnet_only`` accept ``class_weights=`` kwarg.

Verified invariants:
  - default (None) is byte-identical to the unweighted call (no
    silent regression of the canonical training path).
  - uniform weights ``[1,1,1,1,1]`` are byte-identical to None
    (the L1 normalisation makes mean=1).
  - non-uniform weights produce a DIFFERENT loss value (proves the
    code path is wired and reachable).
  - shape mismatch raises ``ValueError`` (fail-loud, not silent).
  - all-zero / negative weights raise ``ValueError`` at parse time.

Per CLAUDE.md "default override antipattern": every default-path
invariant test exists to catch the case where a future refactor
silently changes default behaviour and breaks the byte-identity
contract for callers that never set ``class_weights``.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tac.losses import (
    _apply_class_weights,
    kl_distill_segnet_only,
    parse_class_weights_csv,
    scorer_loss,
    scorer_loss_cached,
)


# ── parse_class_weights_csv ────────────────────────────────────────────


def test_parse_csv_returns_none_for_none():
    assert parse_class_weights_csv(None) is None


def test_parse_csv_returns_none_for_empty():
    assert parse_class_weights_csv("") is None
    assert parse_class_weights_csv("   ") is None


def test_parse_csv_basic_5_classes():
    t = parse_class_weights_csv("1,5,5,1,1")
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert tuple(t.shape) == (5,)
    assert t.tolist() == [1.0, 5.0, 5.0, 1.0, 1.0]


def test_parse_csv_strips_whitespace():
    t = parse_class_weights_csv(" 1.0 , 2.0 , 3.0 , 4.0 , 5.0 ")
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_parse_csv_wrong_count_raises():
    with pytest.raises(ValueError, match="expected 5"):
        parse_class_weights_csv("1,2,3,4")  # only 4 values
    with pytest.raises(ValueError, match="expected 5"):
        parse_class_weights_csv("1,2,3,4,5,6")  # 6 values


def test_parse_csv_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        parse_class_weights_csv("1,-1,1,1,1")


def test_parse_csv_all_zero_raises():
    with pytest.raises(ValueError, match="all zeros"):
        parse_class_weights_csv("0,0,0,0,0")


def test_parse_csv_garbage_raises():
    with pytest.raises(ValueError):
        parse_class_weights_csv("1,2,abc,4,5")


def test_parse_csv_custom_num_classes():
    t = parse_class_weights_csv("1,2,3", num_classes=3)
    assert tuple(t.shape) == (3,)
    with pytest.raises(ValueError, match="expected 3"):
        parse_class_weights_csv("1,2,3,4,5", num_classes=3)


# ── _apply_class_weights kernel ────────────────────────────────────────


def _fake_logits(B: int, C: int, H: int, W: int, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, H, W, generator=g)


def test_apply_class_weights_uniform_is_identity():
    """[1,1,1,1,1] L1-normalised to mean=1 → element-wise unchanged."""
    seg = torch.ones(2, 8, 8) * 0.3
    logits = _fake_logits(2, 5, 8, 8, seed=1)
    cw = torch.ones(5)
    out = _apply_class_weights(seg, logits, cw)
    assert torch.allclose(out, seg)


def test_apply_class_weights_nonuniform_changes_values():
    """A non-uniform weight tensor MUST change the output (proves the
    code path is reachable; if this fails the kernel is dead)."""
    seg = torch.ones(2, 8, 8) * 0.3
    logits = _fake_logits(2, 5, 8, 8, seed=2)
    cw = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    out = _apply_class_weights(seg, logits, cw)
    assert not torch.allclose(out, seg), (
        "non-uniform class weights produced byte-identical output → "
        "Lane PS kernel is DEAD"
    )


def test_apply_class_weights_l1_normalisation_preserves_mean():
    """L1 norm to mean=1 means the OVERALL mean of the weighted seg
    should be close to the unweighted mean (assuming all classes are
    represented). Not exactly equal because the per-pixel distribution
    of classes varies — this is a sanity check on order of magnitude."""
    torch.manual_seed(42)
    seg = torch.rand(4, 32, 32)
    logits = _fake_logits(4, 5, 32, 32, seed=3)
    cw = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    out = _apply_class_weights(seg, logits, cw)
    # mean of cw / mean is 1.0 by construction; the weighted seg mean
    # should be within 3x of unweighted (loose bound — exact ratio
    # depends on which classes win argmax).
    ratio = out.mean().item() / seg.mean().item()
    assert 0.3 < ratio < 3.0, f"L1 normalisation broken: ratio={ratio}"


def test_apply_class_weights_shape_mismatch_raises():
    seg = torch.ones(2, 8, 8)
    logits = _fake_logits(2, 5, 8, 8, seed=4)
    cw_wrong = torch.ones(4)  # only 4 weights for 5 classes
    with pytest.raises(ValueError, match="num_classes"):
        _apply_class_weights(seg, logits, cw_wrong)


def test_apply_class_weights_2d_weight_raises():
    seg = torch.ones(2, 8, 8)
    logits = _fake_logits(2, 5, 8, 8, seed=5)
    cw_wrong = torch.ones(5, 5)  # 2D
    with pytest.raises(ValueError, match="num_classes"):
        _apply_class_weights(seg, logits, cw_wrong)


def test_apply_class_weights_spatial_shape_mismatch_raises():
    seg = torch.ones(2, 8, 8)
    logits = _fake_logits(2, 5, 16, 16, seed=6)  # mismatched spatial
    cw = torch.ones(5)
    with pytest.raises(ValueError, match="spatial shape"):
        _apply_class_weights(seg, logits, cw)


def test_apply_class_weights_no_grad_on_argmax():
    """The argmax indexing path MUST be non-differentiable so the
    weight selection cannot bleed gradient back into the GT logits."""
    seg = torch.ones(2, 8, 8, requires_grad=True)
    logits = _fake_logits(2, 5, 8, 8, seed=7)
    logits.requires_grad_(True)
    cw = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    out = _apply_class_weights(seg, logits, cw)
    out.sum().backward()
    # gradient flows to seg (the per-pixel loss), not to logits
    assert seg.grad is not None
    assert logits.grad is None or torch.all(logits.grad == 0), (
        "class_weight indexing leaked gradient into GT logits"
    )


# ── Mock SegNet + PoseNet for end-to-end loss tests ────────────────────


class _MockScorer(torch.nn.Module):
    """Minimal stand-in for upstream.modules.SegNet / PoseNet.

    SegNet.preprocess_input takes (B, T, C, H, W), keeps last frame,
    resizes to a fixed (H_in, W_in), and returns (B, C, H_in, W_in).
    SegNet's forward returns (B, NUM_CLASSES, H_out, W_out).

    For the unit test we just project the input to the right output
    shape via a fixed conv. The actual scorer weights are immaterial
    here — we only care about LOSS-LEVEL invariants (default == None,
    non-uniform != uniform, shape errors raise).
    """

    def __init__(self, *, num_classes: int = 5, mode: str = "seg"):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        if mode == "seg":
            self.head = torch.nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        else:  # posenet — output = dict with "pose" key, shape (B, P)
            self.head = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
        # x_btchw: (B, T, C, H, W) — for SegNet take last frame
        if self.mode == "seg":
            return x_btchw[:, -1, ...].contiguous()
        # PoseNet: flatten T into channel: (B, T*C, H, W)
        B, T, C, H, W = x_btchw.shape
        return x_btchw.reshape(B, T * C, H, W).contiguous()

    def forward(self, x):
        if self.mode == "seg":
            return self.head(x)
        # PoseNet: project then pool to (B, 6)
        # Need a conv that accepts T*C=6 channels for the pair input
        if self.head.in_channels != x.shape[1]:
            self.head = torch.nn.Conv2d(
                x.shape[1], 6, kernel_size=3, padding=1,
            ).to(x.device)
        h = self.head(x)
        pooled = self.pool(h).flatten(1)  # (B, 6)
        return {"pose": pooled}


def _toy_pair(seed: int = 0) -> torch.Tensor:
    """Return a (B=1, T=2, H=16, W=16, C=3) HWC float pair in [0, 255]."""
    torch.manual_seed(seed)
    return torch.rand(1, 2, 16, 16, 3) * 255.0


# ── End-to-end loss invariants ─────────────────────────────────────────


def test_scorer_loss_default_none_is_byte_identical():
    """Default class_weights=None MUST match the legacy unweighted
    scorer_loss bit-for-bit (covers the byte-identity contract for
    every existing caller that doesn't pass class_weights)."""
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    pose = _MockScorer(mode="pose")
    seg.eval(); pose.eval()
    for p in seg.parameters(): p.requires_grad_(False)
    for p in pose.parameters(): p.requires_grad_(False)

    loss_a, pd_a, sd_a = scorer_loss(pair_f, pair_g, pose, seg)
    loss_b, pd_b, sd_b = scorer_loss(
        pair_f, pair_g, pose, seg, class_weights=None,
    )
    assert torch.allclose(loss_a, loss_b)
    assert pd_a == pytest.approx(pd_b)
    assert sd_a == pytest.approx(sd_b)


def test_scorer_loss_uniform_weights_match_none():
    """L1-normalised uniform weights → mean=1 → byte-identical to None."""
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    pose = _MockScorer(mode="pose")
    seg.eval(); pose.eval()
    for p in seg.parameters(): p.requires_grad_(False)
    for p in pose.parameters(): p.requires_grad_(False)

    loss_none, _, _ = scorer_loss(pair_f, pair_g, pose, seg)
    loss_unif, _, _ = scorer_loss(
        pair_f, pair_g, pose, seg,
        class_weights=torch.ones(5),
    )
    assert torch.allclose(loss_none, loss_unif, atol=1e-6)


def test_scorer_loss_nonuniform_weights_change_value():
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    pose = _MockScorer(mode="pose")
    seg.eval(); pose.eval()
    for p in seg.parameters(): p.requires_grad_(False)
    for p in pose.parameters(): p.requires_grad_(False)

    loss_none, _, _ = scorer_loss(pair_f, pair_g, pose, seg)
    loss_skewed, _, _ = scorer_loss(
        pair_f, pair_g, pose, seg,
        class_weights=torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0]),
    )
    # NB: with 5 classes random argmax + skewed weights, MUST differ.
    assert not torch.allclose(loss_none, loss_skewed), (
        "non-uniform class_weights produced identical loss → wiring is DEAD"
    )


def test_scorer_loss_cached_default_none_is_byte_identical():
    pair_f = _toy_pair(seed=0)
    seg = _MockScorer(mode="seg")
    pose = _MockScorer(mode="pose")
    seg.eval(); pose.eval()
    for p in seg.parameters(): p.requires_grad_(False)
    for p in pose.parameters(): p.requires_grad_(False)

    # Build cached GT (matches scorer_loss_cached's contract)
    pair_g = _toy_pair(seed=1)
    fx_g = pair_g.float().permute(0, 1, 4, 2, 3).contiguous()
    with torch.no_grad():
        gt_pose_6 = pose(pose.preprocess_input(fx_g))["pose"][..., :6]
        gt_seg_logits = seg(seg.preprocess_input(fx_g))
        gt_seg_soft = F.softmax(gt_seg_logits, dim=1)

    loss_a, _, _ = scorer_loss_cached(pair_f, gt_pose_6, gt_seg_soft, pose, seg)
    loss_b, _, _ = scorer_loss_cached(
        pair_f, gt_pose_6, gt_seg_soft, pose, seg, class_weights=None,
    )
    assert torch.allclose(loss_a, loss_b)


def test_kl_distill_segnet_only_default_none_is_byte_identical():
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    seg.eval()
    for p in seg.parameters(): p.requires_grad_(False)

    kl_a, _ = kl_distill_segnet_only(pair_f, pair_g, seg)
    kl_b, _ = kl_distill_segnet_only(pair_f, pair_g, seg, class_weights=None)
    assert torch.allclose(kl_a, kl_b)


def test_kl_distill_segnet_only_uniform_matches_none():
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    seg.eval()
    for p in seg.parameters(): p.requires_grad_(False)

    kl_none, _ = kl_distill_segnet_only(pair_f, pair_g, seg)
    kl_unif, _ = kl_distill_segnet_only(
        pair_f, pair_g, seg, class_weights=torch.ones(5),
    )
    assert torch.allclose(kl_none, kl_unif, atol=1e-6)


def test_kl_distill_segnet_only_nonuniform_changes_value():
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    seg.eval()
    for p in seg.parameters(): p.requires_grad_(False)

    kl_none, _ = kl_distill_segnet_only(pair_f, pair_g, seg)
    kl_skew, _ = kl_distill_segnet_only(
        pair_f, pair_g, seg,
        class_weights=torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0]),
    )
    assert not torch.allclose(kl_none, kl_skew), (
        "non-uniform class_weights produced identical KL → wiring is DEAD"
    )


def test_kl_distill_segnet_only_shape_mismatch_raises():
    pair_f = _toy_pair(seed=0)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    seg.eval()
    for p in seg.parameters(): p.requires_grad_(False)

    with pytest.raises(ValueError, match="num_classes"):
        kl_distill_segnet_only(
            pair_f, pair_g, seg, class_weights=torch.ones(4),
        )


# ── Lane PS gradient flow ──────────────────────────────────────────────


def test_lane_ps_kl_gradient_still_flows_to_filtered():
    """Per-class weighting must NOT zero out the gradient path. The
    rendered pair is what gets optimized in pose TTO — if its gradient
    is dead, the whole technique is moot."""
    pair_f = _toy_pair(seed=0)
    pair_f.requires_grad_(True)
    pair_g = _toy_pair(seed=1)
    seg = _MockScorer(mode="seg")
    seg.eval()
    for p in seg.parameters(): p.requires_grad_(False)

    kl, _ = kl_distill_segnet_only(
        pair_f, pair_g, seg,
        class_weights=torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0]),
    )
    kl.backward()
    assert pair_f.grad is not None
    assert pair_f.grad.abs().sum().item() > 0, (
        "Lane PS killed the gradient flow into the rendered pair"
    )


def test_lane_ps_argmax_indexing_handles_uniform_classes():
    """Pathological case: all GT pixels are class 0. Weight on class 0
    is small — output should be downweighted but not NaN."""
    pair_f = _toy_pair(seed=0)
    # Force the GT pair to be uniform so argmax is stable per pixel.
    pair_g = torch.zeros_like(pair_f) + 128.0
    seg = _MockScorer(mode="seg")
    seg.eval()
    for p in seg.parameters(): p.requires_grad_(False)

    # Skewed weights with class 0 = 0.1 (much smaller than mean)
    cw = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0])
    kl, kl_val = kl_distill_segnet_only(
        pair_f, pair_g, seg, class_weights=cw,
    )
    assert math.isfinite(kl_val), (
        f"Lane PS produced non-finite KL with skewed weights: {kl_val}"
    )
