# SPDX-License-Identifier: MIT
"""Tests for Lane 19 β-variant — sensitivity-weighted logit-margin loss."""
from __future__ import annotations

import pytest
import torch

from tac.logit_margin_sensitivity_weighted import (
    SensitivityWeightedLogitMarginError,
    normalize_sensitivity,
    sensitivity_weighted_logit_margin_loss,
)


def _make_logits(N: int = 2, K: int = 5, H: int = 4, W: int = 4) -> torch.Tensor:
    torch.manual_seed(2026)
    return torch.randn(N, K, H, W)


def _make_gt(N: int = 2, K: int = 5, H: int = 4, W: int = 4) -> torch.Tensor:
    torch.manual_seed(2027)
    return torch.randint(0, K, (N, H, W))


def test_normalize_sensitivity_max_mode_into_zero_one() -> None:
    s = torch.tensor([1.0, 4.0, 0.5, 8.0])
    n = normalize_sensitivity(s, mode="max")
    assert torch.allclose(n, s / 8.0)
    assert n.max().item() == pytest.approx(1.0)
    assert n.min().item() == pytest.approx(0.0625)


def test_normalize_sensitivity_sum_mode_sums_to_one() -> None:
    s = torch.tensor([1.0, 2.0, 3.0])
    n = normalize_sensitivity(s, mode="sum")
    assert n.sum().item() == pytest.approx(1.0)


def test_normalize_sensitivity_none_mode_unchanged() -> None:
    s = torch.tensor([1.0, 2.0, 3.0])
    n = normalize_sensitivity(s, mode="none")
    assert torch.equal(n, s)


def test_normalize_sensitivity_rejects_nan_and_negative() -> None:
    with pytest.raises(SensitivityWeightedLogitMarginError, match="NaN/Inf"):
        normalize_sensitivity(torch.tensor([1.0, float("nan")]))
    with pytest.raises(SensitivityWeightedLogitMarginError, match="non-negative"):
        normalize_sensitivity(torch.tensor([1.0, -1.0]))
    with pytest.raises(SensitivityWeightedLogitMarginError, match="mode must be"):
        normalize_sensitivity(torch.tensor([1.0]), mode="bogus")  # type: ignore[arg-type]


def test_loss_requires_all_kwargs_no_silent_default() -> None:
    logits = _make_logits()
    gt = _make_gt()
    sens = torch.ones_like(gt, dtype=torch.float32)
    with pytest.raises(SensitivityWeightedLogitMarginError, match="logits"):
        sensitivity_weighted_logit_margin_loss(
            None,
            gt,
            pixel_sensitivity=sens,
            threshold=1.0,
            reduction="mean",
        )
    with pytest.raises(SensitivityWeightedLogitMarginError, match="pixel_sensitivity"):
        sensitivity_weighted_logit_margin_loss(
            logits,
            gt,
            pixel_sensitivity=None,
            threshold=1.0,
            reduction="mean",
        )
    with pytest.raises(SensitivityWeightedLogitMarginError, match="threshold"):
        sensitivity_weighted_logit_margin_loss(
            logits,
            gt,
            pixel_sensitivity=sens,
            threshold=None,
            reduction="mean",
        )


def test_loss_reduction_modes() -> None:
    logits = _make_logits()
    gt = _make_gt()
    sens = torch.ones_like(gt, dtype=torch.float32)

    mean_loss = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=sens,
        threshold=2.0,
        reduction="mean",
    )
    sum_loss = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=sens,
        threshold=2.0,
        reduction="sum",
    )
    none_loss = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=sens,
        threshold=2.0,
        reduction="none",
    )
    assert mean_loss.dim() == 0
    assert sum_loss.dim() == 0
    assert none_loss.shape == gt.shape
    # sum / mean = N elements
    assert sum_loss.item() == pytest.approx(mean_loss.item() * gt.numel())


def test_loss_zero_when_all_sensitivity_zero() -> None:
    logits = _make_logits()
    gt = _make_gt()
    sens = torch.zeros_like(gt, dtype=torch.float32)
    # When using "max" normalization, divisor=eps and result is 0 (empty mass).
    loss = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=sens,
        threshold=2.0,
        reduction="sum",
        sensitivity_norm="max",
    )
    assert loss.item() == pytest.approx(0.0)


def test_loss_concentrates_on_high_sensitivity_region() -> None:
    logits = _make_logits()
    gt = _make_gt()
    # Sensitivity = 1 in top-left corner, 0 elsewhere
    sens = torch.zeros_like(gt, dtype=torch.float32)
    sens[..., :2, :2] = 1.0

    loss_concentrated = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=sens,
        threshold=2.0,
        reduction="sum",
    )
    # Compare to flat sensitivity (all 1)
    flat = torch.ones_like(gt, dtype=torch.float32)
    loss_flat = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=flat,
        threshold=2.0,
        reduction="sum",
    )
    # Concentrated loss should be smaller (only some pixels contribute)
    # but per-pixel contribution from top-left should be EQUAL since
    # max-normalization makes both pixel weights = 1.
    assert loss_concentrated.item() <= loss_flat.item()


def test_loss_rejects_negative_sensitivity() -> None:
    logits = _make_logits()
    gt = _make_gt()
    sens = torch.ones_like(gt, dtype=torch.float32)
    sens[0, 0, 0] = -1.0
    with pytest.raises(SensitivityWeightedLogitMarginError, match="non-negative"):
        sensitivity_weighted_logit_margin_loss(
            logits,
            gt,
            pixel_sensitivity=sens,
            threshold=2.0,
            reduction="mean",
        )


def test_loss_rejects_shape_mismatch() -> None:
    logits = _make_logits()
    gt = _make_gt()
    bad_sens = torch.ones(8, 8, dtype=torch.float32)
    with pytest.raises(SensitivityWeightedLogitMarginError, match="shape"):
        sensitivity_weighted_logit_margin_loss(
            logits,
            gt,
            pixel_sensitivity=bad_sens,
            threshold=2.0,
            reduction="mean",
        )


def test_loss_invalid_threshold_raises() -> None:
    logits = _make_logits()
    gt = _make_gt()
    sens = torch.ones_like(gt, dtype=torch.float32)
    with pytest.raises(ValueError):
        sensitivity_weighted_logit_margin_loss(
            logits,
            gt,
            pixel_sensitivity=sens,
            threshold=-1.0,
            reduction="mean",
        )


def test_loss_grad_flows_through_logits_only() -> None:
    """Sensitivity is detached; gradients only flow through logits."""
    logits = _make_logits().requires_grad_(True)
    gt = _make_gt()
    sens = torch.ones_like(gt, dtype=torch.float32).requires_grad_(False)
    loss = sensitivity_weighted_logit_margin_loss(
        logits,
        gt,
        pixel_sensitivity=sens,
        threshold=2.0,
        reduction="sum",
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
