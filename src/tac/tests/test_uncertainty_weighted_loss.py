# SPDX-License-Identifier: MIT
"""Tests for ``tac.uncertainty_weighted_loss`` per TOP-3 + TOP-6 arbitrariness extinction.

Per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE 2026-05-19. Pins the Kendall
multi-task uncertainty weighting + Lin focal loss mathematical contracts.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.uncertainty_weighted_loss import (
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_INITIAL_LOG_SIGMA,
    FOCAL_GAMMA_PAPER_DEFAULT,
    UncertaintyWeightedLossConfig,
    UncertaintyWeightedLossError,
    UncertaintyWeightedScoreLoss,
    apply_focal_per_pair_reweighting,
    per_pair_focal_weights,
)


# -----------------------------------------------------------------------------
# Config tests
# -----------------------------------------------------------------------------

def test_default_config_constructs() -> None:
    cfg = UncertaintyWeightedLossConfig()
    assert cfg.initial_log_sigma_seg == DEFAULT_INITIAL_LOG_SIGMA
    assert cfg.initial_log_sigma_pose == DEFAULT_INITIAL_LOG_SIGMA
    assert cfg.initial_log_sigma_rate == DEFAULT_INITIAL_LOG_SIGMA
    assert cfg.include_rate_axis is True


def test_config_rejects_nan_log_sigma() -> None:
    with pytest.raises(UncertaintyWeightedLossError, match="NaN"):
        UncertaintyWeightedLossConfig(initial_log_sigma_seg=float("nan"))


def test_config_rejects_non_bool_include_rate() -> None:
    with pytest.raises(UncertaintyWeightedLossError):
        UncertaintyWeightedLossConfig(include_rate_axis="yes")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# UncertaintyWeightedScoreLoss module tests
# -----------------------------------------------------------------------------

def test_module_constructs_with_three_axis_params() -> None:
    loss = UncertaintyWeightedScoreLoss()
    params = list(loss.parameters())
    # 3 log-σ params: seg + pose + rate
    assert len(params) == 3


def test_module_constructs_two_axis_when_rate_excluded() -> None:
    cfg = UncertaintyWeightedLossConfig(include_rate_axis=False)
    loss = UncertaintyWeightedScoreLoss(cfg)
    params = list(loss.parameters())
    # 2 log-σ params: seg + pose only
    assert len(params) == 2


def test_forward_at_initial_log_sigma_zero_recovers_half_sum() -> None:
    """At σ=1 (log_sigma=0): L = 0.5*L_axis + 0 for each axis."""
    loss = UncertaintyWeightedScoreLoss()
    seg = torch.tensor(1.0)
    pose = torch.tensor(2.0)
    rate = torch.tensor(3.0)
    out = loss(seg, pose, rate)
    # 0.5*1 + 0 + 0.5*2 + 0 + 0.5*3 + 0 = 3.0
    assert out.item() == pytest.approx(3.0)


def test_forward_increases_loss_when_log_sigma_negative() -> None:
    """Smaller σ (negative log_σ) → larger weighted-loss contribution."""
    cfg = UncertaintyWeightedLossConfig(
        initial_log_sigma_seg=-1.0,  # σ ≈ 0.37
        initial_log_sigma_pose=0.0,
        initial_log_sigma_rate=0.0,
    )
    loss = UncertaintyWeightedScoreLoss(cfg)
    seg = torch.tensor(1.0)
    pose = torch.tensor(1.0)
    rate = torch.tensor(1.0)
    out = loss(seg, pose, rate)
    # seg term: 0.5*e^2*1 + (-1) ≈ 0.5*7.389 - 1 ≈ 2.694
    # pose: 0.5; rate: 0.5; total ≈ 3.694
    expected = 0.5 * math.exp(2.0) * 1.0 + (-1.0) + 0.5 + 0.5
    assert out.item() == pytest.approx(expected, rel=1e-4)


def test_forward_rejects_non_tensor_inputs() -> None:
    loss = UncertaintyWeightedScoreLoss()
    with pytest.raises(UncertaintyWeightedLossError, match="seg_term"):
        loss(1.0, torch.tensor(0.0), torch.tensor(0.0))  # type: ignore[arg-type]


def test_forward_requires_rate_term_when_axis_included() -> None:
    loss = UncertaintyWeightedScoreLoss()
    with pytest.raises(UncertaintyWeightedLossError, match="rate_term"):
        loss(torch.tensor(1.0), torch.tensor(1.0))


def test_forward_ignores_rate_when_axis_excluded() -> None:
    cfg = UncertaintyWeightedLossConfig(include_rate_axis=False)
    loss = UncertaintyWeightedScoreLoss(cfg)
    out = loss(torch.tensor(1.0), torch.tensor(1.0))
    # 0.5*1 + 0 + 0.5*1 + 0 = 1.0
    assert out.item() == pytest.approx(1.0)


def test_log_sigma_is_learnable() -> None:
    loss = UncertaintyWeightedScoreLoss()
    # Backprop should accumulate gradients in log_sigma params.
    seg = torch.tensor(1.0, requires_grad=True)
    pose = torch.tensor(2.0, requires_grad=True)
    rate = torch.tensor(3.0, requires_grad=True)
    out = loss(seg, pose, rate)
    out.backward()
    assert loss.log_sigma_seg.grad is not None
    assert loss.log_sigma_pose.grad is not None
    assert loss.log_sigma_rate.grad is not None


def test_current_sigmas_reports_canonical_initial() -> None:
    loss = UncertaintyWeightedScoreLoss()
    sigmas = loss.current_sigmas()
    assert sigmas["sigma_seg"] == pytest.approx(1.0)
    assert sigmas["sigma_pose"] == pytest.approx(1.0)
    assert sigmas["sigma_rate"] == pytest.approx(1.0)


def test_current_axis_weights_reports_canonical_initial() -> None:
    loss = UncertaintyWeightedScoreLoss()
    weights = loss.current_axis_weights()
    # At log_sigma=0: weight = 0.5 * exp(0) = 0.5
    assert weights["weight_seg"] == pytest.approx(0.5)
    assert weights["weight_pose"] == pytest.approx(0.5)
    assert weights["weight_rate"] == pytest.approx(0.5)


def test_current_sigmas_excludes_rate_when_axis_excluded() -> None:
    cfg = UncertaintyWeightedLossConfig(include_rate_axis=False)
    loss = UncertaintyWeightedScoreLoss(cfg)
    sigmas = loss.current_sigmas()
    assert "sigma_rate" not in sigmas
    assert "sigma_seg" in sigmas
    assert "sigma_pose" in sigmas


def test_optimizer_step_reduces_loss() -> None:
    """Canonical Kendall test: SGD on (model, log_sigma) lowers total."""
    torch.manual_seed(42)
    loss_module = UncertaintyWeightedScoreLoss()
    optim = torch.optim.SGD(loss_module.parameters(), lr=0.1)
    initial_total = None
    final_total = None
    seg = torch.tensor(0.5)
    pose = torch.tensor(2.0)
    rate = torch.tensor(0.1)
    for step in range(50):
        optim.zero_grad()
        total = loss_module(seg, pose, rate)
        if step == 0:
            initial_total = total.item()
        total.backward()
        optim.step()
        if step == 49:
            final_total = total.item()
    assert final_total < initial_total


# -----------------------------------------------------------------------------
# Focal-loss helper tests
# -----------------------------------------------------------------------------

def test_focal_gamma_zero_returns_uniform_weights() -> None:
    losses = torch.tensor([0.1, 1.0, 5.0])
    w = per_pair_focal_weights(losses, gamma=0.0)
    assert torch.allclose(w, torch.ones_like(losses))


def test_focal_gamma_positive_emphasizes_hard_pairs() -> None:
    # p = exp(-loss); hard pair (large loss) → small p → large (1-p)^γ
    losses = torch.tensor([0.1, 1.0, 5.0])
    w = per_pair_focal_weights(losses, gamma=FOCAL_GAMMA_PAPER_DEFAULT)
    assert w[0] < w[1] < w[2]


def test_focal_weights_bounded_zero_to_one() -> None:
    losses = torch.linspace(0.0, 10.0, 100)
    w = per_pair_focal_weights(losses, gamma=FOCAL_GAMMA_PAPER_DEFAULT)
    assert (w >= 0.0).all()
    assert (w <= 1.0).all()


def test_focal_weights_rejects_negative_gamma() -> None:
    losses = torch.tensor([1.0])
    with pytest.raises(UncertaintyWeightedLossError, match=">= 0"):
        per_pair_focal_weights(losses, gamma=-1.0)


def test_focal_weights_rejects_nan_gamma() -> None:
    losses = torch.tensor([1.0])
    with pytest.raises(UncertaintyWeightedLossError):
        per_pair_focal_weights(losses, gamma=float("nan"))


def test_focal_weights_rejects_non_tensor() -> None:
    with pytest.raises(UncertaintyWeightedLossError):
        per_pair_focal_weights([1.0, 2.0])  # type: ignore[arg-type]


def test_focal_weights_detach_by_default() -> None:
    losses = torch.tensor([1.0, 2.0], requires_grad=True)
    w = per_pair_focal_weights(losses, gamma=2.0, detach_weights=True)
    assert not w.requires_grad


def test_focal_weights_can_flow_gradient() -> None:
    losses = torch.tensor([1.0, 2.0], requires_grad=True)
    w = per_pair_focal_weights(losses, gamma=2.0, detach_weights=False)
    assert w.requires_grad


def test_apply_focal_reweighting_uniform_recovers_mean() -> None:
    losses = torch.tensor([0.5, 1.0, 1.5])
    weighted_mean = apply_focal_per_pair_reweighting(losses, gamma=0.0)
    plain_mean = losses.mean()
    assert weighted_mean.item() == pytest.approx(plain_mean.item())


def test_apply_focal_reweighting_focuses_hard_pairs() -> None:
    losses = torch.tensor([0.01, 0.01, 5.0])
    plain_mean = losses.mean()  # ≈ 1.67
    focal_mean = apply_focal_per_pair_reweighting(
        losses, gamma=FOCAL_GAMMA_PAPER_DEFAULT
    )
    # Focal weight on hard pair (loss=5.0) is much higher → focal_mean > plain_mean
    assert focal_mean.item() > plain_mean.item()


# -----------------------------------------------------------------------------
# Canonical constants pinning
# -----------------------------------------------------------------------------

def test_canonical_defaults_match_kendall_2018() -> None:
    """Pin canonical defaults per Kendall et al 2018 + codex directive TOP-6."""
    assert DEFAULT_INITIAL_LOG_SIGMA == 0.0  # σ = 1 = uniform initial weighting
    assert DEFAULT_FOCAL_GAMMA == 0.0  # uniform per-pair by default; opt-in focal
    assert FOCAL_GAMMA_PAPER_DEFAULT == 2.0  # Lin et al 2017 paper default
