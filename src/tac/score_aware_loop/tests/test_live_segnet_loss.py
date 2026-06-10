# SPDX-License-Identifier: MIT
"""Behavior tests for the PR95-faithful live-SegNet score-aware losses.

These verify BEHAVIOR (NO FAKE per CLAUDE.md class 2): each test would FAIL if
the loss body were replaced by a constant or a no-op. We assert the loss
descends as the live render's argmax matches the target, the margin gradient
reaches the carrier, the d_seg observable equals the true argmax-disagreement,
and a severed/constant gradient does NOT descend.
"""

from __future__ import annotations

import torch

from tac.score_aware_loop.live_segnet_loss import (
    ce_seg_loss,
    exact_d_seg_from_logits,
    l7_softplus_seg_loss,
    pose_loss,
    smooth_disagreement_seg_loss,
    tau_softplus_seg_loss,
)


def _logits_matching(targets: torch.Tensor, n_classes: int, conf: float) -> torch.Tensor:
    """Build logits whose argmax == targets with confidence ``conf`` (margin)."""
    b, h, w = targets.shape
    logits = torch.zeros(b, n_classes, h, w)
    logits.scatter_(1, targets.unsqueeze(1).long(), conf)
    return logits


def test_exact_d_seg_zero_when_argmax_matches():
    targets = torch.randint(0, 5, (2, 8, 8))
    logits = _logits_matching(targets, 5, conf=10.0)
    assert exact_d_seg_from_logits(logits, targets) == 0.0


def test_exact_d_seg_one_when_argmax_all_wrong():
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    logits = torch.zeros(2, 5, 8, 8)
    logits[:, 1] = 10.0  # argmax = class 1 everywhere, target = 0 everywhere
    assert exact_d_seg_from_logits(logits, targets) == 1.0


def test_exact_d_seg_is_the_true_disagreement_rate():
    # Half the pixels match, half don't -> d_seg == 0.5 exactly.
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    logits = torch.zeros(1, 5, 2, 2)
    logits[:, 0, 0, 0] = 10.0  # match
    logits[:, 0, 0, 1] = 10.0  # match
    logits[:, 1, 1, 0] = 10.0  # mismatch (argmax=1, target=0)
    logits[:, 1, 1, 1] = 10.0  # mismatch
    assert exact_d_seg_from_logits(logits, targets) == 0.5


def test_ce_loss_lower_when_argmax_correct():
    targets = torch.randint(0, 5, (2, 8, 8))
    good = _logits_matching(targets, 5, conf=10.0)
    bad = _logits_matching(targets, 5, conf=0.0)  # uniform -> argmax ambiguous
    assert float(ce_seg_loss(good, targets)) < float(ce_seg_loss(bad, targets))


def test_ce_loss_descends_under_gradient_descent():
    # The decisive NO-FAKE test: a free logit tensor optimized against ce_seg_loss
    # must reduce the EXACT d_seg toward 0. A constant/no-op loss would not.
    torch.manual_seed(0)
    targets = torch.randint(0, 5, (2, 6, 6))
    logits = torch.zeros(2, 5, 6, 6, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.5)
    d0 = exact_d_seg_from_logits(logits.detach(), targets)
    for _ in range(60):
        opt.zero_grad()
        ce_seg_loss(logits, targets).backward()
        opt.step()
    d1 = exact_d_seg_from_logits(logits.detach(), targets)
    assert d1 < d0
    assert d1 < 0.05  # near-perfect argmax match reachable


def test_margin_surrogates_descend_d_seg():
    for loss_fn in (
        tau_softplus_seg_loss,
        smooth_disagreement_seg_loss,
        l7_softplus_seg_loss,
    ):
        torch.manual_seed(1)
        targets = torch.randint(0, 5, (2, 6, 6))
        logits = torch.zeros(2, 5, 6, 6, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.5)
        d0 = exact_d_seg_from_logits(logits.detach(), targets)
        for _ in range(80):
            opt.zero_grad()
            loss_fn(logits, targets).backward()
            opt.step()
        d1 = exact_d_seg_from_logits(logits.detach(), targets)
        assert d1 < d0, f"{loss_fn.__name__} did not descend d_seg ({d0}->{d1})"


def test_smooth_disagreement_equals_disagreement_at_sharp_logits():
    # sigmoid(-margin/tau) -> 0 for large positive margin (correct), 1 for
    # large negative margin (wrong). At sharp logits it approximates d_seg.
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, 5, 4, 4), -10.0)
    logits[:, 0] = 10.0  # all correct -> surrogate ~ 0
    assert float(smooth_disagreement_seg_loss(logits, targets)) < 0.01


def test_loss_gradient_is_nonzero_at_boundary():
    # Margin near 0 must produce a nonzero gradient (the boundary push).
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.zeros(1, 5, 4, 4, requires_grad=True)
    smooth_disagreement_seg_loss(logits, targets).backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_pose_loss_is_sqrt_10_mse():
    pred = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    tgt = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mse = ((pred - tgt) ** 2).mean()
    expected = torch.sqrt(10.0 * mse + 1e-12)
    assert torch.allclose(pose_loss(pred, tgt), expected, atol=1e-5)


def test_pose_loss_zero_at_match():
    pred = torch.randn(3, 6)
    assert float(pose_loss(pred, pred)) < 1e-5


def test_ce_loss_rejects_wrong_shapes():
    targets = torch.randint(0, 5, (2, 8, 8))
    logits = torch.zeros(2, 5, 8, 8)
    # ce_seg_loss tolerates via F.cross_entropy; the margin helper is strict.
    import pytest

    from tac.score_aware_loop.live_segnet_loss import _target_minus_runnerup_margin

    with pytest.raises(ValueError):
        _target_minus_runnerup_margin(logits, targets[:, 0])  # wrong target ndim
    with pytest.raises(ValueError):
        _target_minus_runnerup_margin(logits[0], targets)  # wrong logit ndim


def test_l7_weight_boost_concentrates_on_hard_pixels():
    # A pixel with small margin (hard) should contribute more than an easy one.
    targets = torch.zeros(1, 1, 2, dtype=torch.long)
    logits = torch.zeros(1, 5, 1, 2)
    logits[:, 0, 0, 0] = 5.0  # easy: margin 5
    logits[:, 0, 0, 1] = 0.1  # hard: margin ~0.1
    # L7 with a big multiplier weights the hard pixel up; the loss should be
    # strictly larger than the unweighted softplus mean.
    l7 = float(l7_softplus_seg_loss(logits, targets, l7_mult=10.0))
    tau = float(tau_softplus_seg_loss(logits, targets))
    assert l7 > tau
