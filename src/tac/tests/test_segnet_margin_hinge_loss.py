# SPDX-License-Identifier: MIT
"""Tests for the ACCELERATOR-PROBE-1 SegNet margin-hinge flip-targeting loss.

The margin-hinge ``relu(margin_target − (logit[GT] − max_{c≠GT} logit[c]))`` is the
flip-targeting seg loss: ZERO gradient on correct-with-margin pixels (no waste),
CONSTANT-magnitude pull on every flip / near-flip (unlike soft-cosine, whose pull
collapses on confident flips — the gradient-vanish root cause Probe C identified).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tac.losses.core import (
    DEFAULT_SEGNET_MARGIN_HINGE_TARGET,
    SEGMENTATION_SURROGATE_MARGIN_HINGE,
    road_lane_emphasis_class_weights,
    segnet_margin_hinge_per_pixel,
    segnet_surrogate_per_pixel,
)


def _const_logits(B, C, H, W, winner_class, winner_value, *, requires_grad=True):
    """Logits where ``winner_class`` channel = ``winner_value``, all others 0."""
    p = torch.zeros(B, C, H, W)
    p[:, winner_class] = winner_value
    return p.requires_grad_(requires_grad)


def test_zero_loss_when_correct_with_margin():
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    # GT class 0 wins by margin 5 >> target 1.0 → hinge zero everywhere.
    p = _const_logits(2, 5, 4, 6, winner_class=0, winner_value=5.0)
    loss = segnet_margin_hinge_per_pixel(p, gt, margin_target=1.0)
    assert loss.max().item() == 0.0


def test_positive_loss_below_target_margin():
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    # GT wins by only 0.4 < target 1.0 → loss = 1.0 - 0.4 = 0.6 (positive, no flip).
    p = _const_logits(2, 5, 4, 6, winner_class=0, winner_value=0.4)
    loss = segnet_margin_hinge_per_pixel(p, gt, margin_target=1.0)
    assert loss.min().item() == pytest.approx(0.6, abs=1e-5)


def test_flip_loss_value_is_target_minus_signed_margin():
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    # class 1 wins by 3 → GT(0) FLIPPED, signed margin = 0 - 3 = -3 → loss = 1-(-3)=4.
    p = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=3.0)
    loss = segnet_margin_hinge_per_pixel(p, gt, margin_target=1.0)
    assert loss.mean().item() == pytest.approx(4.0, abs=1e-5)


def test_gradient_sign_raises_gt_lowers_runner_up():
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    p = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=3.0)
    segnet_margin_hinge_per_pixel(p, gt, margin_target=1.0).sum().backward()
    # d loss / d logit[GT=0] < 0 (gradient descent RAISES it); d/d runner-up(1) > 0.
    assert p.grad[0, 0, 0, 0].item() < 0.0
    assert p.grad[0, 1, 0, 0].item() > 0.0
    # Untouched non-competitor channels carry no gradient.
    assert p.grad[0, 2, 0, 0].item() == 0.0


def test_gradient_does_not_vanish_on_confident_flip_unlike_soft_cosine():
    """THE decisive property: on a confident flip the hinge keeps a constant pull
    while soft-cosine's gradient collapses to ~0 (Probe C's gradient-vanish root cause)."""
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    # Confident flip: class 1 wins by 50.
    p_hinge = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=50.0)
    segnet_margin_hinge_per_pixel(p_hinge, gt, margin_target=1.0).sum().backward()
    g_hinge = abs(p_hinge.grad[0, 0, 0, 0].item())

    gt_logits = F.one_hot(gt, num_classes=5).permute(0, 3, 1, 2).float() * 30.0
    p_sc = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=50.0)
    segnet_surrogate_per_pixel(
        p_sc, gt_logits, surrogate="soft_cosine", temperature=1.0
    ).sum().backward()
    g_sc = abs(p_sc.grad[0, 0, 0, 0].item())

    assert g_hinge >= 0.9  # constant-magnitude pull survives
    assert g_sc < 1e-8  # soft-cosine grad vanished
    assert g_hinge > g_sc * 1e6


def test_dispatcher_routes_margin_hinge_equal_to_direct_call():
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    gt_logits = F.one_hot(gt, num_classes=5).permute(0, 3, 1, 2).float() * 30.0
    p = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=3.0, requires_grad=False)
    via_dispatch = segnet_surrogate_per_pixel(
        p, gt_logits, surrogate=SEGMENTATION_SURROGATE_MARGIN_HINGE, margin_hinge_target=1.0
    )
    direct = segnet_margin_hinge_per_pixel(p, gt, margin_target=1.0)
    assert torch.allclose(via_dispatch, direct)


def test_dispatcher_margin_hinge_ignores_temperature():
    """The hinge is on raw logits — temperature must not change it (hard argmax surface)."""
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    gt_logits = F.one_hot(gt, num_classes=5).permute(0, 3, 1, 2).float() * 30.0
    p = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=3.0, requires_grad=False)
    a = segnet_surrogate_per_pixel(
        p, gt_logits, surrogate="margin_hinge", temperature=0.3, margin_hinge_target=1.0
    )
    b = segnet_surrogate_per_pixel(
        p, gt_logits, surrogate="margin_hinge", temperature=1.0, margin_hinge_target=1.0
    )
    assert torch.allclose(a, b)


def test_nonunit_margin_target_scales_hinge():
    gt = torch.zeros(2, 4, 6, dtype=torch.long)
    p = _const_logits(2, 5, 4, 6, winner_class=1, winner_value=3.0)
    # signed margin -3; target 2 → loss 5.
    loss = segnet_margin_hinge_per_pixel(p, gt, margin_target=2.0)
    assert loss.mean().item() == pytest.approx(5.0, abs=1e-5)


def test_per_pixel_gt_index_is_honored():
    """Different GT classes per pixel → the hinge uses each pixel's own GT."""
    p = torch.zeros(1, 5, 1, 2)
    p[0, 0, 0, 0] = 4.0  # pixel (0,0): class 0 wins by 4
    p[0, 2, 0, 1] = 4.0  # pixel (0,1): class 2 wins by 4
    gt = torch.tensor([[[0, 2]]], dtype=torch.long)  # both correct
    loss = segnet_margin_hinge_per_pixel(p.requires_grad_(True), gt, margin_target=1.0)
    assert loss[0, 0, 0].item() == 0.0
    assert loss[0, 0, 1].item() == 0.0
    # Swap GT → both become flips of margin -4 → loss 5 each.
    gt_swapped = torch.tensor([[[2, 0]]], dtype=torch.long)
    loss2 = segnet_margin_hinge_per_pixel(p.detach().requires_grad_(True), gt_swapped, margin_target=1.0)
    assert loss2[0, 0, 0].item() == pytest.approx(5.0, abs=1e-5)
    assert loss2[0, 0, 1].item() == pytest.approx(5.0, abs=1e-5)


def test_invalid_margin_target_raises():
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    p = torch.zeros(1, 5, 2, 2)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            segnet_margin_hinge_per_pixel(p, gt, margin_target=bad)


def test_shape_validation():
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError):
        segnet_margin_hinge_per_pixel(torch.zeros(1, 3, 2, 2), gt)  # wrong num_classes
    with pytest.raises(ValueError):
        segnet_margin_hinge_per_pixel(torch.zeros(1, 5, 2, 2), torch.zeros(1, 3, 3, dtype=torch.long))


def test_pred_logits_not_mutated_by_scatter():
    """The runner-up mask must be out-of-place — pred_logits stays intact for the GT gather."""
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    p = _const_logits(1, 5, 2, 2, winner_class=1, winner_value=3.0, requires_grad=False)
    snapshot = p.clone()
    segnet_margin_hinge_per_pixel(p.requires_grad_(True), gt, margin_target=1.0)
    assert torch.equal(p.detach(), snapshot)


def test_road_lane_emphasis_weights():
    w = road_lane_emphasis_class_weights(emphasis=2.0)
    assert w.tolist() == [2.0, 2.0, 1.0, 1.0, 1.0]
    w3 = road_lane_emphasis_class_weights(emphasis=3.5)
    assert w3.tolist() == [3.5, 3.5, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        road_lane_emphasis_class_weights(emphasis=0.0)


def test_default_margin_target_constant():
    assert DEFAULT_SEGNET_MARGIN_HINGE_TARGET == 1.0
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    p = _const_logits(1, 5, 2, 2, winner_class=1, winner_value=3.0)
    default = segnet_margin_hinge_per_pixel(p, gt)
    explicit = segnet_margin_hinge_per_pixel(
        p.detach().requires_grad_(True), gt, margin_target=DEFAULT_SEGNET_MARGIN_HINGE_TARGET
    )
    assert torch.allclose(default, explicit)


def test_grad_is_constant_magnitude_across_flip_depths():
    """The hinge gradient magnitude is INDEPENDENT of how confidently wrong the flip is —
    the property that makes it a uniform flip-fixer (CE decays its push, soft-cosine collapses)."""
    gt = torch.zeros(1, 1, 1, dtype=torch.long)
    grads = []
    for depth in (2.0, 10.0, 50.0):
        p = torch.zeros(1, 5, 1, 1)
        p[0, 1] = depth  # flip of increasing depth
        p = p.requires_grad_(True)
        segnet_margin_hinge_per_pixel(p, gt, margin_target=1.0).sum().backward()
        grads.append(abs(p.grad[0, 0, 0, 0].item()))
    # All equal (constant magnitude) — the defining flip-targeting property.
    assert all(math.isclose(g, grads[0], abs_tol=1e-6) for g in grads)
