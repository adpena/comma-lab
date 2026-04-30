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
