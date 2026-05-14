# SPDX-License-Identifier: MIT
"""Tests for the opt-in Fisher-Rao SegNet training surrogate."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.losses import (
    SEGMENTATION_SURROGATE_FISHER_RAO,
    SEGMENTATION_SURROGATE_SINKHORN,
    SEGMENTATION_SURROGATE_SOFT_COSINE,
    segnet_fisher_rao_per_pixel,
    segnet_surrogate_per_pixel,
    sinkhorn_w2_mask_distortion_per_pixel,
)


def _one_hot(labels: torch.Tensor, *, classes: int = 5) -> torch.Tensor:
    return F.one_hot(labels, num_classes=classes).permute(0, 3, 1, 2).float()


def test_fisher_rao_identical_distributions_are_zero() -> None:
    probs = torch.rand(2, 5, 3, 4)
    probs = probs / probs.sum(dim=1, keepdim=True)

    out = segnet_fisher_rao_per_pixel(probs, probs)

    assert torch.allclose(out, torch.zeros_like(out))


def test_fisher_rao_disjoint_one_hot_mismatch_is_unit_scaled() -> None:
    labels_a = torch.zeros(1, 2, 2, dtype=torch.long)
    labels_b = torch.ones(1, 2, 2, dtype=torch.long)

    out = segnet_fisher_rao_per_pixel(_one_hot(labels_a), _one_hot(labels_b))

    assert torch.allclose(out, torch.ones_like(out))


def test_fisher_rao_is_symmetric() -> None:
    a = torch.rand(1, 5, 3, 3)
    b = torch.rand(1, 5, 3, 3)
    a = a / a.sum(dim=1, keepdim=True)
    b = b / b.sum(dim=1, keepdim=True)

    ab = segnet_fisher_rao_per_pixel(a, b)
    ba = segnet_fisher_rao_per_pixel(b, a)

    assert torch.allclose(ab, ba)


def test_fisher_rao_surrogate_has_finite_gradients_near_one_hot() -> None:
    pred_logits = torch.tensor(
        [[[[8.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]]]],
        requires_grad=True,
    )
    gt_probs = _one_hot(torch.zeros(1, 1, 1, dtype=torch.long))

    out = segnet_surrogate_per_pixel(
        pred_logits,
        gt_probs,
        surrogate=SEGMENTATION_SURROGATE_FISHER_RAO,
        gt_already_probs=True,
    ).mean()
    out.backward()

    assert pred_logits.grad is not None
    assert torch.isfinite(pred_logits.grad).all()


def test_soft_cosine_surrogate_preserves_legacy_formula() -> None:
    pred_logits = torch.randn(2, 5, 3, 4)
    gt_logits = torch.randn(2, 5, 3, 4)

    expected = 1.0 - (
        F.softmax(pred_logits, dim=1) * F.softmax(gt_logits, dim=1)
    ).sum(dim=1)
    actual = segnet_surrogate_per_pixel(
        pred_logits,
        gt_logits,
        surrogate=SEGMENTATION_SURROGATE_SOFT_COSINE,
    )

    assert torch.equal(actual, expected)


def test_sinkhorn_identical_distributions_are_zero() -> None:
    probs = torch.rand(2, 5, 3, 4)
    probs = probs / probs.sum(dim=1, keepdim=True)

    out = sinkhorn_w2_mask_distortion_per_pixel(probs, probs, n_iters=10)

    assert torch.all(out >= -1e-6)
    assert out.abs().max().item() < 1e-4


def test_sinkhorn_surrogate_dispatches_to_sinkhorn_not_soft_cosine() -> None:
    pred_logits = torch.randn(2, 5, 3, 4)
    gt_probs = torch.softmax(torch.randn(2, 5, 3, 4), dim=1)
    pred_probs = torch.softmax(pred_logits, dim=1)

    expected = sinkhorn_w2_mask_distortion_per_pixel(
        pred_probs,
        gt_probs,
        blur=0.1,
        n_iters=7,
    )
    actual = segnet_surrogate_per_pixel(
        pred_logits,
        gt_probs,
        surrogate=SEGMENTATION_SURROGATE_SINKHORN,
        gt_already_probs=True,
        sinkhorn_blur=0.1,
        sinkhorn_n_iters=7,
    )
    legacy_soft_cosine = 1.0 - (pred_probs * gt_probs).sum(dim=1)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(actual, legacy_soft_cosine, atol=1e-4)


def test_sinkhorn_rejects_invalid_cost_matrix() -> None:
    probs = torch.full((1, 5, 1, 1), 0.2)
    bad = torch.ones(5, 5)

    with pytest.raises(ValueError, match="zero diagonal"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, cost_matrix=bad)


def test_cached_segnet_probabilities_reject_non_unit_temperature() -> None:
    pred_logits = torch.randn(1, 5, 2, 2)
    gt_probs = F.softmax(torch.randn(1, 5, 2, 2), dim=1)

    with pytest.raises(ValueError, match="cached SegNet probabilities"):
        segnet_surrogate_per_pixel(
            pred_logits,
            gt_probs,
            surrogate=SEGMENTATION_SURROGATE_FISHER_RAO,
            temperature=2.0,
            gt_already_probs=True,
        )


@pytest.mark.parametrize("bad_eps", [0.0, -1e-6, 1e-3, float("inf")])
def test_fisher_rao_rejects_invalid_eps(bad_eps: float) -> None:
    probs = torch.full((1, 5, 1, 1), 0.2)

    with pytest.raises(ValueError, match="fisher_rao_eps"):
        segnet_fisher_rao_per_pixel(probs, probs, eps=bad_eps)


def test_fisher_rao_rejects_class_count_mismatch() -> None:
    probs = torch.full((1, 4, 1, 1), 0.25)

    with pytest.raises(ValueError, match="expects 5 classes"):
        segnet_fisher_rao_per_pixel(probs, probs)
