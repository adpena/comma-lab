# SPDX-License-Identifier: MIT
"""Regression tests for Lane J-JBL (Jaccard Metric Loss + Boundary Label
Smoothing distillation, ``tac.losses_jbl``).

Reference: Wang et al., "Jaccard Metric Losses: Optimizing the Jaccard
Index with Soft Labels" (arXiv 2302.05666, NeurIPS 2023, refined 2024).

Per Round 26 + CLAUDE.md "Forbidden score claims" / "Critical Lessons":
each test asserts a SIGN/VALUE contract, never just finiteness. A
finite-only test would let arithmetic regressions slip through (e.g.
the historical ``F.kl_div(reduction="batchmean")`` bug that under-
divided by H*W and shipped for months).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.losses_jbl import (
    boundary_label_smoothing,
    combined_jbl_distill_loss,
    jaccard_metric_loss,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _toy_logits(B: int = 1, C: int = 5, H: int = 16, W: int = 16,
                seed: int = 0) -> torch.Tensor:
    """Reproducible logits for tests. Small spatial size keeps tests fast."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, H, W, generator=g)


def _hard_mask(B: int = 1, H: int = 16, W: int = 16, num_classes: int = 5,
               seed: int = 1) -> torch.Tensor:
    """Reproducible integer GT mask in [0, num_classes)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_classes, (B, H, W), generator=g)


# ── 1. JML: zero at perfect match ───────────────────────────────────────────


def test_jaccard_metric_loss_zero_for_perfect_match() -> None:
    """JML(softmax(x), softmax(x)) == 0 (intersection == union)."""
    logits = _toy_logits()
    soft = F.softmax(logits, dim=1)
    loss = jaccard_metric_loss(logits, soft, num_classes=5)
    # Perfect match: intersection_c = sum p_c^2; union_c = 2*sum p_c^2 -
    # sum p_c^2 = sum p_c^2; ratio == 1; loss = 1 - 1 = 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-5), (
        f"JML on identical distributions must be 0; got {loss.item()}."
    )


# ── 2. JML strictly increases with class-flip count ────────────────────────


def test_jaccard_metric_loss_increases_with_disagreement() -> None:
    """JML strictly increases as more pixels flip class.

    Construct teacher with all pixels in class 0, then student with k
    pixels flipped to class 1. JML must be monotone non-decreasing in k
    (strictly increasing as long as k changes the soft distribution).
    """
    B, C, H, W = 1, 5, 8, 8
    # Teacher: all class 0 (one-hot).
    teacher_soft = torch.zeros(B, C, H, W)
    teacher_soft[:, 0, :, :] = 1.0

    losses = []
    # Student: k pixels predicted as class 1, rest as class 0. Use very
    # peaked logits so softmax is effectively one-hot.
    for k in (0, 4, 16, 32, 64):
        student_logits = torch.full((B, C, H, W), -10.0)
        student_logits[:, 0, :, :] = 10.0  # default: all class 0
        if k > 0:
            flat_idx = torch.arange(k)
            ys = flat_idx // W
            xs = flat_idx % W
            student_logits[:, 0, ys, xs] = -10.0
            student_logits[:, 1, ys, xs] = 10.0
        loss = jaccard_metric_loss(student_logits, teacher_soft, num_classes=C)
        losses.append(loss.item())

    # Strictly monotone increasing (each more-disagreement step raises
    # the loss). k=0 is the perfect-match case (loss == 0).
    for i in range(len(losses) - 1):
        assert losses[i] < losses[i + 1], (
            f"JML must strictly increase with disagreement; "
            f"losses={losses}, failed at index {i}."
        )
    assert losses[0] == pytest.approx(0.0, abs=1e-4)


# ── 3. JML gradient flow + correct sign ─────────────────────────────────────


def test_jaccard_metric_loss_gradient_flow() -> None:
    """JML gradient is non-trivial AND has the correct descent sign.

    Stronger than a finiteness check: we verify that taking a gradient
    step on student_logits in the -grad direction reduces the loss
    (gradient points uphill, so descent goes downhill toward teacher).
    This is the contract that makes JML useful as a training signal.
    """
    student_logits = _toy_logits(seed=10).requires_grad_(True)
    teacher_soft = F.softmax(_toy_logits(seed=20), dim=1)

    loss_before = jaccard_metric_loss(student_logits, teacher_soft, num_classes=5)
    loss_before.backward()

    grad = student_logits.grad
    assert grad is not None, "JML must produce gradients on student_logits."
    assert grad.abs().sum().item() > 1e-6, (
        f"JML gradient must be non-trivial; got abs-sum={grad.abs().sum().item()}."
    )

    # Take a small step in -grad direction; loss must decrease (correct
    # descent direction).
    with torch.no_grad():
        new_logits = student_logits - 0.1 * grad
    loss_after = jaccard_metric_loss(new_logits, teacher_soft, num_classes=5)
    assert loss_after.item() < loss_before.item(), (
        f"Stepping in -grad direction must decrease JML; "
        f"before={loss_before.item():.6f} after={loss_after.item():.6f}."
    )


# ── 4. BLS: only boundary pixels are smoothed ──────────────────────────────


def test_boundary_label_smoothing_only_affects_boundary() -> None:
    """Interior pixels (no class transition in neighbourhood) keep one-hot;
    boundary pixels (class transition within boundary_width) are smoothed.

    Construct a mask with TWO disjoint regions: a uniform interior block
    far from any class change, and a boundary along the diagonal. Verify
    interior pixels are one-hot exactly and boundary pixels are smoothed.
    """
    B, H, W, C = 1, 32, 32, 5
    target = torch.zeros((B, H, W), dtype=torch.long)  # all class 0
    # Inject a vertical boundary column at x=16 from class 0 to class 2.
    target[:, :, 16:] = 2

    soft, boundary_mask = boundary_label_smoothing(
        target, num_classes=C, smoothing=0.1, boundary_width=2,
    )

    # Interior pixels (far from x=16): one-hot exactly.
    # x=0 is 16 columns from boundary, way beyond width=2.
    interior_pixel = soft[0, :, 0, 0]  # (C,)
    assert torch.allclose(
        interior_pixel,
        F.one_hot(torch.tensor(0), num_classes=C).float(),
    ), f"Interior pixel must be one-hot; got {interior_pixel.tolist()}."

    # Boundary pixel at (0, 16) — on the class transition. Should be smoothed.
    # Expected at boundary for true class 2: (1-0.1)*1 + 0.1/5 = 0.92 for class 2,
    # 0.1/5 = 0.02 for other classes.
    boundary_pixel = soft[0, :, 0, 16]  # (C,)
    expected_target_class = 0.9 + 0.1 / C
    expected_other = 0.1 / C
    assert boundary_pixel[2].item() == pytest.approx(expected_target_class, abs=1e-5), (
        f"Boundary pixel target-class prob must be {expected_target_class}; "
        f"got {boundary_pixel[2].item()}."
    )
    assert boundary_pixel[0].item() == pytest.approx(expected_other, abs=1e-5)
    assert boundary_pixel[1].item() == pytest.approx(expected_other, abs=1e-5)
    # Sums to 1 (probability distribution).
    assert boundary_pixel.sum().item() == pytest.approx(1.0, abs=1e-5)

    # Boundary mask asserts: interior=0, boundary column=1.
    assert boundary_mask[0, 0, 0].item() == 0.0
    assert boundary_mask[0, 0, 16].item() == 1.0

    # Sanity: every boundary pixel sums to 1 (not just the one we sampled).
    sums = soft.sum(dim=1)  # (B, H, W)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        "Soft target rows must sum to 1 everywhere."
    )


# ── 5. End-to-end: combined JBL loss strictly decreases under descent ──────


def test_combined_jbl_loss_decreases_under_descent() -> None:
    """50-step Adam loop on student_logits (frozen teacher); loss must
    monotonically (or at least strictly) decrease across the 50 steps.

    This is the load-bearing contract: combined_jbl_distill_loss has to
    be a usable training signal. If a future refactor breaks the
    gradient direction (e.g. swaps a sign), the loss will plateau or
    increase and this test will catch it.
    """
    B, C, H, W = 1, 5, 16, 16
    teacher_logits = _toy_logits(B=B, C=C, H=H, W=W, seed=100).detach()
    gt_mask = teacher_logits.argmax(dim=1)  # use teacher argmax as GT
    student_logits = _toy_logits(B=B, C=C, H=H, W=W, seed=200).clone()
    student_logits.requires_grad_(True)

    optimizer = torch.optim.Adam([student_logits], lr=0.05)

    initial_loss, _ = combined_jbl_distill_loss(
        student_logits, teacher_logits, gt_mask, num_classes=C,
    )
    initial_value = initial_loss.item()

    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = combined_jbl_distill_loss(
            student_logits, teacher_logits, gt_mask, num_classes=C,
        )
        loss.backward()
        optimizer.step()

    final_loss, _ = combined_jbl_distill_loss(
        student_logits, teacher_logits, gt_mask, num_classes=C,
    )
    final_value = final_loss.item()

    # Must STRICTLY decrease (not just stay finite). Tighten with margin
    # so a sign-flip regression is unambiguously caught.
    assert final_value < initial_value - 0.01, (
        f"Combined JBL loss must strictly decrease under 50 Adam steps; "
        f"initial={initial_value:.6f} final={final_value:.6f} "
        f"delta={final_value - initial_value:.6f}."
    )


# ── 6. Profile registration ─────────────────────────────────────────────────


def test_jbl_profile_is_importable() -> None:
    """``profiles.get_profile('j_jbl_dilated_h64')`` returns a dict with
    ``loss_mode == 'jbl'`` and inherits Lane G v3's hyperparameter spine.
    """
    from tac.profiles import get_profile

    p = get_profile("j_jbl_dilated_h64")
    assert isinstance(p, dict), f"profile must be a dict; got {type(p)}."
    assert p.get("loss_mode") == "jbl", (
        f"loss_mode must be 'jbl'; got {p.get('loss_mode')!r}."
    )
    # Boundary weight must be present (default 3.0 per Wang et al. 3-5×).
    assert p.get("boundary_weight") == pytest.approx(3.0)
    assert p.get("bls_smoothing") == pytest.approx(0.1)
    # Inherits Lane G v3's KL-distill scalar (repurposed as JBL master
    # weight) so the effective wiring stays byte-identical except for
    # the loss family.
    assert p.get("kl_distill_weight") == pytest.approx(0.002)
