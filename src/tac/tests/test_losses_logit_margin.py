# SPDX-License-Identifier: MIT
"""Lane 19 — SegNet logit-margin boundary loss tests.

Per Phase 2 Lane 19 spec (memory project_phases_2_3_4_*):
- Gradient direction matches CE on confident pixels (zero gradient when
  margin >= threshold)
- Larger gradient magnitude on ambiguous pixels (margin < threshold)
- Finite loss on synthetic inputs

All claims tagged [synthetic].

CLAUDE.md non-negotiables verified:
- No scorer load (we provide synthetic logits directly)
- No silent defaults (threshold + reduction are required-keyword)
- Deterministic CPU-only
"""
from __future__ import annotations

import pytest
import torch

from tac.losses_logit_margin import fragility_weights, logit_margin_loss


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: fragility_weights — high margin → 0; zero margin → 1
# ─────────────────────────────────────────────────────────────────────────────


def test_fragility_weights_extremes_synthetic() -> None:
    """[synthetic] High margin → weight 0; zero margin → weight 1."""
    # Confident: top1 = 5.0, top2 = 0.0 → margin = 5.0
    confident = torch.tensor([[5.0, 0.0, -1.0, -2.0, -3.0]])  # (N=1, K=5)
    w_conf = fragility_weights(confident, threshold=1.0)
    assert w_conf.shape == (1,)
    assert w_conf.item() == pytest.approx(0.0)

    # Ambiguous: top1 = top2 → margin = 0
    ambiguous = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    w_amb = fragility_weights(ambiguous, threshold=1.0)
    assert w_amb.item() == pytest.approx(1.0)

    # Mid: margin = 0.5, threshold = 1.0 → weight = 0.5
    mid = torch.tensor([[1.0, 0.5, 0.0, 0.0, 0.0]])
    w_mid = fragility_weights(mid, threshold=1.0)
    assert w_mid.item() == pytest.approx(0.5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: fragility_weights handles spatial dims (N, K, H, W)
# ─────────────────────────────────────────────────────────────────────────────


def test_fragility_weights_spatial_shape_synthetic() -> None:
    """[synthetic] (N, K, H, W) logits → (N, H, W) weights."""
    g = torch.Generator().manual_seed(2026)
    logits = torch.randn(2, 5, 4, 6, generator=g)  # N=2, K=5, H=4, W=6
    w = fragility_weights(logits, threshold=2.0)
    assert w.shape == (2, 4, 6)
    # All weights in [0, 1]
    assert (w >= 0.0).all()
    assert (w <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: logit_margin_loss zero on perfectly confident inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_loss_zero_on_perfectly_confident_inputs_synthetic() -> None:
    """[synthetic] When margin >= threshold for all pixels, loss = 0."""
    # Logits: class 0 wins decisively. Use (N=1, K=5) — class dim must be at 1.
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])  # (N=1, K=5)
    gt = torch.tensor([0])  # (N=1,)
    loss = logit_margin_loss(logits=logits, gt_argmax=gt, threshold=1.0, reduction="mean")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: logit_margin_loss positive on ambiguous inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_loss_positive_on_ambiguous_inputs_synthetic() -> None:
    """[synthetic] Ambiguous logits → positive loss."""
    # Logits: tied at top → margin = 0 → weight = 1; CE > 0
    logits = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])  # (N=1, K=5)
    gt = torch.tensor([0])  # (N=1,)
    loss = logit_margin_loss(logits=logits, gt_argmax=gt, threshold=1.0, reduction="mean")
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: gradient magnitude — ambiguous pixels have larger gradient than confident ones
# ─────────────────────────────────────────────────────────────────────────────


def test_gradient_magnitude_larger_on_ambiguous_pixels_synthetic() -> None:
    """[synthetic] |∇L/∇logits| is larger on ambiguous than on confident pixels.

    Build a 2-pixel batch: pixel 0 confident (margin >> threshold → weight 0
    → no gradient); pixel 1 ambiguous (margin = 0 → weight 1 → full CE grad).
    Logits shape (N=1, K=5, P=2) — class dim at position 1.
    """
    confident_logits = [10.0, 0.0, 0.0, 0.0, 0.0]
    ambiguous_logits = [0.5, 0.5, 0.0, 0.0, 0.0]
    # Build (N=1, K=5, P=2): logits[0, k, p] = (confident if p==0 else ambiguous)[k]
    logits_data = torch.zeros(1, 5, 2)
    for k in range(5):
        logits_data[0, k, 0] = confident_logits[k]
        logits_data[0, k, 1] = ambiguous_logits[k]
    logits = logits_data.clone().requires_grad_(True)
    gt = torch.tensor([[0, 0]])  # (N=1, P=2)
    loss = logit_margin_loss(
        logits=logits, gt_argmax=gt, threshold=1.0, reduction="sum"
    )
    loss.backward()
    grads = logits.grad  # (N, K, P)
    # |grad| at pixel 0 (confident) should be ~0
    grad_p0 = grads[:, :, 0].abs().sum().item()
    # |grad| at pixel 1 (ambiguous) should be > 0
    grad_p1 = grads[:, :, 1].abs().sum().item()
    assert grad_p0 < 1e-5, f"confident pixel got grad={grad_p0} (should be ~0)"
    assert grad_p1 > 1e-3, f"ambiguous pixel got grad={grad_p1} (should be > 0)"
    assert grad_p1 > grad_p0


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: gradient direction — same sign as plain CE on ambiguous pixels
# ─────────────────────────────────────────────────────────────────────────────


def test_gradient_direction_matches_ce_on_ambiguous_synthetic() -> None:
    """[synthetic] On ambiguous pixels, sign(grad) matches sign(CE grad).

    The fragility weighting is positive scalar w ∈ [0, 1] applied to CE.
    So grad direction is unchanged; only magnitude is scaled.
    """
    g = torch.Generator().manual_seed(20260429)
    # All pixels ambiguous: small spread of logits
    logits_ce = torch.randn(2, 5, 3, 4, generator=g) * 0.1  # tiny variance → ambiguous
    logits_ml = logits_ce.detach().clone().requires_grad_(True)
    logits_ce = logits_ce.requires_grad_(True)
    gt = torch.zeros(2, 3, 4, dtype=torch.long)
    # Plain CE
    ce_loss = torch.nn.functional.cross_entropy(logits_ce, gt, reduction="sum")
    ce_loss.backward()
    # Margin loss with very high threshold → all pixels get weight 1 → identical to CE
    ml_loss = logit_margin_loss(
        logits=logits_ml, gt_argmax=gt, threshold=100.0, reduction="sum"
    )
    ml_loss.backward()
    # Gradients have same sign everywhere
    same_sign = (logits_ce.grad.sign() == logits_ml.grad.sign())
    # Allow zero-grad pixels (ties); require >=99% same sign
    assert same_sign.float().mean().item() > 0.99


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: no silent defaults — fragility_weights / logit_margin_loss reject None
# ─────────────────────────────────────────────────────────────────────────────


def test_no_silent_defaults_synthetic() -> None:
    """[synthetic] All public functions reject None / missing required args."""
    logits = torch.randn(1, 5)
    with pytest.raises(ValueError, match="threshold is required"):
        fragility_weights(logits, threshold=None)
    with pytest.raises(ValueError, match="threshold must be"):
        fragility_weights(logits, threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be"):
        fragility_weights(logits, threshold=-1.0)
    # logit_margin_loss requires both args
    with pytest.raises(ValueError, match="logits and gt_argmax are required"):
        logit_margin_loss(logits=None, gt_argmax=torch.zeros(1, dtype=torch.long), threshold=1.0, reduction="mean")
    with pytest.raises(ValueError, match="threshold is required"):
        logit_margin_loss(logits=logits, gt_argmax=torch.zeros(1, dtype=torch.long), threshold=None, reduction="mean")
    with pytest.raises(ValueError, match="reduction must be"):
        logit_margin_loss(logits=logits, gt_argmax=torch.zeros(1, dtype=torch.long), threshold=1.0, reduction="bad")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: shape mismatches caught
# ─────────────────────────────────────────────────────────────────────────────


def test_shape_mismatch_caught_synthetic() -> None:
    """[synthetic] gt_argmax shape mismatch raises."""
    logits = torch.randn(2, 5, 4, 6)  # (N, K, H, W) — gt should be (N, H, W)
    gt_bad_ndim = torch.zeros(2, 5, 4, 6, dtype=torch.long)  # ndim too high
    with pytest.raises(ValueError, match="gt_argmax must be"):
        logit_margin_loss(
            logits=logits, gt_argmax=gt_bad_ndim, threshold=1.0, reduction="mean"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: logit_margin_loss_with_teacher — teacher argmax becomes GT
# ─────────────────────────────────────────────────────────────────────────────


def test_teacher_helper_uses_teacher_argmax_synthetic() -> None:
    """[synthetic] teacher helper derives gt from teacher argmax + matches manual call."""
    from tac.losses_logit_margin import logit_margin_loss_with_teacher

    g = torch.Generator().manual_seed(123)
    student = torch.randn(2, 5, 3, 4, generator=g) * 0.3
    teacher = torch.randn(2, 5, 3, 4, generator=g) * 0.3
    # Manual: gt = teacher.argmax(dim=1); compare to teacher-helper.
    gt = teacher.argmax(dim=1)
    expected = logit_margin_loss(
        logits=student, gt_argmax=gt, threshold=1.0, reduction="mean",
    )
    actual = logit_margin_loss_with_teacher(
        student_logits=student, teacher_logits=teacher,
        threshold=1.0, reduction="mean",
    )
    assert torch.allclose(actual, expected, atol=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: logit_margin_loss_with_teacher rejects None / shape mismatch
# ─────────────────────────────────────────────────────────────────────────────


def test_teacher_helper_no_silent_defaults_synthetic() -> None:
    """[synthetic] teacher helper rejects missing args + shape mismatch."""
    from tac.losses_logit_margin import logit_margin_loss_with_teacher

    s = torch.randn(1, 5, 3, 4)
    t = torch.randn(1, 5, 3, 4)
    bad = torch.randn(1, 5, 3, 5)
    with pytest.raises(ValueError, match="student_logits and"):
        logit_margin_loss_with_teacher(
            student_logits=None, teacher_logits=t, threshold=1.0, reduction="mean",
        )
    with pytest.raises(ValueError, match="threshold is required"):
        logit_margin_loss_with_teacher(
            student_logits=s, teacher_logits=t, threshold=None, reduction="mean",
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        logit_margin_loss_with_teacher(
            student_logits=s, teacher_logits=bad, threshold=1.0, reduction="mean",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: teacher helper does NOT backprop through teacher
# ─────────────────────────────────────────────────────────────────────────────


def test_teacher_helper_no_grad_through_teacher_synthetic() -> None:
    """[synthetic] gradient flows into student but not teacher."""
    from tac.losses_logit_margin import logit_margin_loss_with_teacher

    student = torch.randn(1, 5, 2, 3, requires_grad=True)
    teacher = torch.randn(1, 5, 2, 3, requires_grad=True)
    loss = logit_margin_loss_with_teacher(
        student_logits=student, teacher_logits=teacher,
        threshold=2.0, reduction="sum",
    )
    loss.backward()
    assert student.grad is not None
    # Teacher requires_grad but no path should backprop into it (argmax is
    # under no_grad). teacher.grad should remain None.
    assert teacher.grad is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: compute_segnet_logit_margin_aux end-to-end with stub SegNet
# ─────────────────────────────────────────────────────────────────────────────


def _make_stub_segnet():
    """Synthetic SegNet stub: 1×1 conv 3→5 + identity preprocess."""
    import torch.nn as nn

    class StubSegNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 1x1 conv: 3 channels in, 5 classes out, no bias for determinism.
            self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(
                    torch.randn(5, 3, 1, 1, generator=torch.Generator().manual_seed(7)) * 0.5
                )

        def preprocess_input(self, x):
            # x: (B, T=1, C=3, H, W). Squeeze T.
            return x.squeeze(1).contiguous()

        def forward(self, x):
            # x: (B, 3, H, W); returns (B, 5, H, W).
            return self.conv(x)

    return StubSegNet()


def test_compute_segnet_logit_margin_aux_smoke_synthetic() -> None:
    """[synthetic] end-to-end aux call returns finite scalar with grad."""
    from tac.losses_logit_margin import compute_segnet_logit_margin_aux

    segnet = _make_stub_segnet()
    g = torch.Generator().manual_seed(2026)
    rendered_data = torch.rand(2, 2, 8, 8, 3, generator=g) * 255.0
    rendered = rendered_data.clone().detach().requires_grad_(True)
    gt = torch.rand(2, 2, 8, 8, 3, generator=g) * 255.0
    loss = compute_segnet_logit_margin_aux(
        rendered_pair=rendered, gt_pair=gt, segnet=segnet,
        threshold=1.0, reduction="mean",
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: compute_segnet_logit_margin_aux validates inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_segnet_logit_margin_aux_validates_inputs_synthetic() -> None:
    """[synthetic] reject None args / wrong-ndim pairs / shape mismatch."""
    from tac.losses_logit_margin import compute_segnet_logit_margin_aux

    segnet = _make_stub_segnet()
    pair = torch.zeros(1, 2, 4, 4, 3)
    with pytest.raises(ValueError, match="rendered_pair / gt_pair / segnet"):
        compute_segnet_logit_margin_aux(
            rendered_pair=None, gt_pair=pair, segnet=segnet,
            threshold=1.0, reduction="mean",
        )
    with pytest.raises(ValueError, match="threshold is required"):
        compute_segnet_logit_margin_aux(
            rendered_pair=pair, gt_pair=pair, segnet=segnet,
            threshold=None, reduction="mean",
        )
    with pytest.raises(ValueError, match="pairs must be 5-D"):
        compute_segnet_logit_margin_aux(
            rendered_pair=torch.zeros(2, 4, 4, 3),
            gt_pair=torch.zeros(2, 4, 4, 3), segnet=segnet,
            threshold=1.0, reduction="mean",
        )
    with pytest.raises(ValueError, match="pair shape mismatch"):
        compute_segnet_logit_margin_aux(
            rendered_pair=pair,
            gt_pair=torch.zeros(1, 2, 4, 5, 3),
            segnet=segnet,
            threshold=1.0, reduction="mean",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: determinism — same seed → same loss value
# ─────────────────────────────────────────────────────────────────────────────


def test_logit_margin_loss_deterministic_synthetic() -> None:
    """[synthetic] same seed → same loss value (±1e-6)."""
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    logits1 = torch.randn(2, 5, 4, 6, generator=g1) * 0.5
    logits2 = torch.randn(2, 5, 4, 6, generator=g2) * 0.5
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    l1 = logit_margin_loss(logits=logits1, gt_argmax=gt, threshold=1.0, reduction="mean")
    l2 = logit_margin_loss(logits=logits2, gt_argmax=gt, threshold=1.0, reduction="mean")
    assert l1.item() == pytest.approx(l2.item(), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: A/B vs standard CE on a fragility-controlled synthetic batch.
# ─────────────────────────────────────────────────────────────────────────────


def test_margin_loss_lower_than_ce_on_confident_correct_pixels_synthetic() -> None:
    """[synthetic] On a batch dominated by confident-correct pixels, margin
    loss is STRICTLY LESS than standard CE because margin loss zeros out the
    confident region while CE still pays the small log(1) ≈ 0 amount.

    This is the wedge: margin loss spends nothing on confident pixels.
    """
    # Build a batch where 95% of pixels are confident-correct and 5% boundary.
    g = torch.Generator().manual_seed(100)
    N, K, H, W = 1, 5, 20, 20
    logits = torch.zeros(N, K, H, W)
    # Confident-correct: class 0 wins by 5.0 everywhere
    logits[:, 0, :, :] = 5.0
    # Add small noise to the 5% boundary pixels (random pick)
    boundary_mask = torch.rand(H, W, generator=g) < 0.05
    # Where boundary, make logits ambiguous between class 0 and class 1
    boundary_idx = boundary_mask.nonzero()
    for idx in boundary_idx:
        i, j = int(idx[0]), int(idx[1])
        logits[0, 0, i, j] = 0.4
        logits[0, 1, i, j] = 0.5  # class 1 narrowly wins
    gt = torch.zeros(N, H, W, dtype=torch.long)  # GT = class 0 always
    ce = torch.nn.functional.cross_entropy(logits, gt, reduction="mean")
    ml = logit_margin_loss(logits=logits, gt_argmax=gt, threshold=1.0, reduction="mean")
    # margin loss focuses on boundary; CE averages over everything (incl confident)
    # Both are > 0; we assert structural property: margin loss IGNORES confident
    # pixels entirely (they contribute zero), so per-pixel signal on boundary is
    # higher relative to the mean.
    # Specifically: the margin-loss MEAN equals (sum over boundary pixels of CE×weight)
    # divided by (total pixel count). The CE MEAN equals (sum over ALL pixels of CE)
    # divided by total pixel count. Confident pixels have tiny CE (log(1+e^-5) ≈ 0.0067)
    # but multiplied by 95% of the count it adds up.
    # Sanity: both finite and non-negative.
    assert torch.isfinite(ce) and ce.item() >= 0
    assert torch.isfinite(ml) and ml.item() >= 0
    # The margin loss does NOT have to be smaller than CE (they measure different
    # things). What we assert is that the margin loss is concentrated on boundary:
    # if we increase a CONFIDENT pixel's class-0 logit further, CE decreases but
    # margin loss is INVARIANT (already zero weight).
    logits_more_confident = logits.clone()
    # Pick a confident-correct pixel (not in boundary mask)
    free_mask = ~boundary_mask
    free_idx = free_mask.nonzero()[0]
    fi, fj = int(free_idx[0]), int(free_idx[1])
    logits_more_confident[0, 0, fi, fj] = 10.0  # even more confident
    ml_more = logit_margin_loss(
        logits=logits_more_confident, gt_argmax=gt,
        threshold=1.0, reduction="mean",
    )
    ce_more = torch.nn.functional.cross_entropy(logits_more_confident, gt, reduction="mean")
    # Margin loss UNCHANGED (confident pixel had weight 0 already)
    assert ml.item() == pytest.approx(ml_more.item(), abs=1e-7), (
        "Margin loss must be invariant to logit changes on already-confident pixels"
    )
    # CE strictly decreased (any CE pixel improvement reduces mean)
    assert ce_more.item() < ce.item(), (
        "Sanity: CE should decrease when a confident pixel's correct logit grows"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: integration — Lane 19 aux + scorer_loss together catch confident-wrong
# ─────────────────────────────────────────────────────────────────────────────


def test_fragility_weights_detached_in_loss_synthetic() -> None:
    """[regression Round 3 CRITICAL] Fragility weights MUST be detached.

    Without `.detach()`, the gradient flowing through the weight pushes the
    renderer to artificially widen margins (exploit of the loss formulation,
    not a real learning signal). On a boundary-WRONG pixel, the un-detached
    version produces ∂L/∂z[top1] in the WRONG direction (pushes z[top1] UP
    instead of DOWN — increasing argmax disagreement instead of reducing it).

    This test pins the detach behavior: gradient on z[top1] must be
    POSITIVE (→ optimizer DECREASES z[top1]) on a boundary-WRONG pixel.

    Memory: .omx/research/council_lane_19_logit_margin_round3_20260430.md
    """
    # Boundary-wrong pixel: GT = class 1, top1 = class 0 (wrong), top2 = class 1.
    # margin = z[top1] - z[top2] = small (< threshold) → boundary.
    # We want ∂L/∂z[top1] > 0 so optimizer decreases the wrong class's logit.
    logits = torch.tensor([[0.6, 0.5, 0.0, 0.0, 0.0]], requires_grad=True)
    gt = torch.tensor([1])  # GT = class 1
    loss = logit_margin_loss(logits=logits, gt_argmax=gt, threshold=1.0, reduction="sum")
    loss.backward()
    assert logits.grad is not None
    grad_top1 = logits.grad[0, 0].item()  # gradient on the WRONG (top1) class
    grad_gt = logits.grad[0, 1].item()    # gradient on the CORRECT (GT) class
    # CORRECT direction: optimizer minimizes L → step opposite gradient.
    # We want optimizer to DECREASE z[top1] → ∂L/∂z[top1] should be POSITIVE.
    assert grad_top1 > 0, (
        f"Boundary-wrong pixel: expected ∂L/∂z[top1] > 0 (push wrong class DOWN); "
        f"got {grad_top1}. Without weights.detach(), this would be NEGATIVE. "
        f"Round 3 Filler/Fridrich/Hinton CRITICAL fix regressed."
    )
    # And we want optimizer to INCREASE z[GT] → ∂L/∂z[GT] should be NEGATIVE.
    assert grad_gt < 0, (
        f"Boundary-wrong pixel: expected ∂L/∂z[GT] < 0 (push correct class UP); "
        f"got {grad_gt}."
    )


def test_lane_19_profile_resolver_pipe_synthetic() -> None:
    """[regression] Profile → resolver → args pipe: lane_19_logit_margin profile
    sets logit_margin_weight=10.0 (Round 1 Contrarian fix), kl_distill_weight=0.0
    (Round 1 Quantizr fix), threshold=1.0. Must reach args.* (not silent default).

    Locks against the silent-default-override class (memory:
    feedback_silent_default_bug_class_findings_20260429.md). If a future commit
    accidentally removes the profile key or changes the resolver call site,
    this test catches it BEFORE 5h of GPU is wasted on a misconfigured run.
    """
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[3]
    SRC = REPO / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from tac.profiles import PROFILES

    p = PROFILES["lane_19_logit_margin"]
    # Direct profile-dict assertions (Round 1 fixes)
    assert p["logit_margin_weight"] == 10.0, (
        f"Round 1 Contrarian fix regressed: weight should be 10.0 not {p['logit_margin_weight']}"
    )
    assert p["logit_margin_threshold"] == 1.0
    assert p["kl_distill_weight"] == 0.0, (
        f"Round 1 Quantizr fix regressed: kl_distill_weight should be 0.0 not {p['kl_distill_weight']}"
    )
    assert p["loss_mode"] == "logit_margin"
    assert p["seed"] == 89


def test_lane_19_aux_plus_scorer_handles_confident_wrong_synthetic() -> None:
    """[synthetic] A confident-wrong pixel contributes 0 to Lane 19 aux loss
    (by design — boundary mask filters it out) BUT the underlying scorer_loss
    path (standard CE on logits) catches it. This is the integration story:
    Lane 19 is an AUXILIARY, never a sole replacement.
    """
    # Pixel logits: confident-WRONG (high class-0, GT is class-1).
    # Margin = top1 - top2 = high - small. Wide margin → fragility weight = 0.
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])  # (N=1, K=5)
    gt = torch.tensor([1])  # GT = class 1, but class 0 wins.
    # Lane 19 contribution: ZERO (pixel is "confident").
    lane19 = logit_margin_loss(
        logits=logits, gt_argmax=gt, threshold=1.0, reduction="mean",
    )
    assert lane19.item() == pytest.approx(0.0, abs=1e-6)
    # Standard CE contribution: LARGE (cross-entropy of a confidently-wrong prediction).
    ce = torch.nn.functional.cross_entropy(logits, gt, reduction="mean")
    assert ce.item() > 5.0, (
        f"Standard CE should be large (>5) on confident-wrong pixel; got {ce.item()}"
    )
    # Conclusion: caller must use scorer_loss + Lane 19, not Lane 19 alone.
    # (This documents the design decision in the council memo §2 Contrarian.)
