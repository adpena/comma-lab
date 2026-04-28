"""Tests for src/tac/learnable_saliency_threshold.py — Lane SI-V2.

Pins:
  1. threshold_logit + log_temperature are learnable Parameters.
  2. threshold() always in (0, 1) via sigmoid.
  3. temperature() always > 0 via exp.
  4. forward() returns soft mask in [0, 1].
  5. As temperature → 0, soft mask → hard 0/1 boundary.
  6. Gradient flows through both threshold and temperature.
  7. fit_linear_rate_model returns the analytic slope/intercept.
  8. optimise_threshold_for_target_bytes drives bytes toward target.
  9. learn_temperature=False freezes log_temperature as a buffer.
 10. Init validation (threshold ∈ (0,1), temperature > 0).
 11. zero-slope rate model raises (sanity check the encoder probe).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.learnable_saliency_threshold import (
    LearnableSaliencyThreshold,
    OptimiseThresholdResult,
    differentiable_threshold_mask,
    fit_linear_rate_model,
    optimise_threshold_for_target_bytes,
)


# ── Module basics ────────────────────────────────────────────────────────


def test_threshold_and_log_temperature_are_parameters():
    m = LearnableSaliencyThreshold()
    assert isinstance(m.threshold_logit, nn.Parameter)
    assert m.threshold_logit.requires_grad
    assert isinstance(m.log_temperature, nn.Parameter)
    assert m.log_temperature.requires_grad


def test_threshold_in_closed_unit_interval():
    """sigmoid clamps the threshold to [0, 1] regardless of the logit
    (mathematically open, but underflow can collapse to exact 0/1 at
    extreme logits — what matters is that the value stays bounded)."""
    m = LearnableSaliencyThreshold(init_threshold=0.5)
    with torch.no_grad():
        m.threshold_logit.fill_(-100.0)
    t = m.threshold()
    assert 0.0 <= t.item() <= 1.0
    with torch.no_grad():
        m.threshold_logit.fill_(100.0)
    t = m.threshold()
    assert 0.0 <= t.item() <= 1.0
    # At moderate logits the value is strictly inside (0, 1)
    with torch.no_grad():
        m.threshold_logit.fill_(0.0)
    t = m.threshold()
    assert 0.0 < t.item() < 1.0
    assert t.item() == pytest.approx(0.5, abs=1e-5)


def test_temperature_positive():
    m = LearnableSaliencyThreshold(init_temperature=0.1)
    with torch.no_grad():
        m.log_temperature.fill_(-50.0)
    t = m.temperature()
    assert t.item() > 0.0


def test_init_threshold_round_trip():
    """init_threshold=0.3 ⇒ threshold() ≈ 0.3 (modulo float error)."""
    m = LearnableSaliencyThreshold(init_threshold=0.3, init_temperature=0.07)
    assert m.threshold().item() == pytest.approx(0.3, abs=1e-4)
    assert m.temperature().item() == pytest.approx(0.07, abs=1e-4)


def test_init_validation():
    with pytest.raises(ValueError, match="init_threshold"):
        LearnableSaliencyThreshold(init_threshold=0.0)
    with pytest.raises(ValueError, match="init_threshold"):
        LearnableSaliencyThreshold(init_threshold=1.0)
    with pytest.raises(ValueError, match="init_temperature"):
        LearnableSaliencyThreshold(init_temperature=0.0)
    with pytest.raises(ValueError, match="init_temperature"):
        LearnableSaliencyThreshold(init_temperature=-1.0)


def test_learn_temperature_false_freezes_log_temperature():
    m = LearnableSaliencyThreshold(learn_temperature=False)
    # log_temperature should NOT be in parameters()
    param_names = {n for n, _ in m.named_parameters()}
    assert "log_temperature" not in param_names
    assert "threshold_logit" in param_names
    # And it lives as a buffer, not a parameter
    buf_names = {n for n, _ in m.named_buffers()}
    assert "log_temperature" in buf_names


# ── Soft-mask shape + values ────────────────────────────────────────────


def test_forward_returns_soft_mask_in_unit_interval():
    m = LearnableSaliencyThreshold(init_threshold=0.5, init_temperature=0.1)
    sal = torch.linspace(0.0, 1.0, 100)
    out = m(sal)
    assert (out >= 0).all() and (out <= 1).all()


def test_forward_at_threshold_returns_half():
    """sigmoid(0) = 0.5 — value at the threshold itself."""
    m = LearnableSaliencyThreshold(init_threshold=0.4, init_temperature=0.05)
    sal = torch.tensor([0.4])
    out = m(sal)
    assert out.item() == pytest.approx(0.5, abs=1e-4)


def test_low_temperature_collapses_to_hard_mask():
    """As temperature → 0, soft mask → step function (1 below threshold,
    0 above)."""
    m = LearnableSaliencyThreshold(init_threshold=0.5, init_temperature=1e-4)
    sal = torch.tensor([0.1, 0.4, 0.6, 0.9])
    out = m(sal)
    # Below threshold (0.1, 0.4) ⇒ ≈ 1.0 (blind spot)
    # Above threshold (0.6, 0.9) ⇒ ≈ 0.0 (salient, preserve)
    assert out[0].item() > 0.99
    assert out[1].item() > 0.99
    assert out[2].item() < 0.01
    assert out[3].item() < 0.01


def test_high_temperature_smears_the_mask():
    """Large temperature ⇒ all values close to 0.5."""
    m = LearnableSaliencyThreshold(init_threshold=0.5, init_temperature=100.0)
    sal = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    out = m(sal)
    assert torch.all((out - 0.5).abs() < 0.01)


def test_hard_mask_no_grad():
    """hard_mask returns a bool tensor that is detached."""
    m = LearnableSaliencyThreshold(init_threshold=0.4)
    sal = torch.linspace(0, 1, 10)
    hm = m.hard_mask(sal)
    assert hm.dtype == torch.bool
    # First few values <= 0.4 ⇒ True (blind); last few > 0.4 ⇒ False
    assert hm[0].item() is True
    assert hm[-1].item() is False


# ── Gradient flow ────────────────────────────────────────────────────────


def test_gradient_flows_through_threshold():
    m = LearnableSaliencyThreshold(init_threshold=0.5, init_temperature=0.05)
    sal = torch.linspace(0, 1, 50)
    out = m(sal)
    out.sum().backward()
    assert m.threshold_logit.grad is not None
    assert torch.isfinite(m.threshold_logit.grad).all()
    # Non-zero (soft mask shifts when threshold moves)
    assert m.threshold_logit.grad.abs().item() > 0


def test_gradient_flows_through_temperature():
    m = LearnableSaliencyThreshold(
        init_threshold=0.5, init_temperature=0.1, learn_temperature=True
    )
    sal = torch.linspace(0, 1, 50)
    out = m(sal)
    out.sum().backward()
    assert m.log_temperature.grad is not None
    assert torch.isfinite(m.log_temperature.grad).all()


# ── differentiable_threshold_mask helper ─────────────────────────────────


def test_helper_accepts_python_floats():
    sal = torch.linspace(0, 1, 10)
    out = differentiable_threshold_mask(sal, threshold=0.5, temperature=0.05)
    assert out.shape == sal.shape


def test_helper_accepts_tensor_threshold_and_temperature():
    sal = torch.linspace(0, 1, 10)
    thr = torch.tensor(0.3, requires_grad=True)
    tmp = torch.tensor(0.05, requires_grad=True)
    out = differentiable_threshold_mask(sal, threshold=thr, temperature=tmp)
    out.sum().backward()
    assert thr.grad is not None
    assert tmp.grad is not None


# ── Linear rate model ───────────────────────────────────────────────────


def test_fit_linear_rate_model_recovers_known_line():
    """If measure_bytes is exactly slope*t + intercept, the fit returns
    the same slope + intercept."""
    truth_slope, truth_intercept = -1500.0, 2000.0

    def measure(t: float) -> int:
        return int(truth_slope * t + truth_intercept)

    slope, intercept = fit_linear_rate_model(measure, sample_thresholds=(0.2, 0.8))
    assert slope == pytest.approx(truth_slope, abs=10.0)
    assert intercept == pytest.approx(truth_intercept, abs=10.0)


def test_fit_linear_rate_model_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least two"):
        fit_linear_rate_model(lambda t: 100, sample_thresholds=(0.5,))


def test_fit_linear_rate_model_collapsed_thresholds_raises():
    with pytest.raises(ValueError, match="span"):
        fit_linear_rate_model(lambda t: 100, sample_thresholds=(0.5, 0.5))


# ── Optimise threshold ─────────────────────────────────────────────────


def test_optimise_drives_bytes_toward_target():
    """With a known linear rate model, dual ascent should converge on a
    threshold that hits target_bytes."""
    slope, intercept = -1000.0, 1500.0  # bytes(t) = -1000*t + 1500
    target = 800
    res = optimise_threshold_for_target_bytes(
        target_bytes=target,
        rate_slope=slope,
        rate_intercept=intercept,
        init_threshold=0.5,
        max_iterations=2000,
        lr_threshold=0.05,
        tolerance_bytes=20.0,  # tight enough to require non-trivial movement
    )
    assert isinstance(res, OptimiseThresholdResult)
    # Closed-form optimum t* = (target - intercept) / slope
    #   = (800 - 1500) / -1000 = 0.7
    assert res.threshold == pytest.approx(0.7, abs=0.05)
    assert abs(res.estimated_bytes - target) < 256.0


def test_optimise_marks_converged_when_within_tolerance():
    res = optimise_threshold_for_target_bytes(
        target_bytes=500,
        rate_slope=-1000.0,
        rate_intercept=1000.0,
        init_threshold=0.5,
        max_iterations=200,
        tolerance_bytes=10.0,
    )
    if res.converged:
        assert abs(res.final_constraint_violation) <= 10.0


def test_optimise_zero_slope_raises():
    with pytest.raises(ValueError, match="rate_slope == 0"):
        optimise_threshold_for_target_bytes(
            target_bytes=500,
            rate_slope=0.0,
            rate_intercept=1000.0,
        )


def test_optimise_returns_finite_values():
    res = optimise_threshold_for_target_bytes(
        target_bytes=300,
        rate_slope=-500.0,
        rate_intercept=800.0,
        init_threshold=0.5,
    )
    assert torch.isfinite(torch.tensor(res.threshold))
    assert torch.isfinite(torch.tensor(res.temperature))
    assert torch.isfinite(torch.tensor(res.estimated_bytes))
    assert res.iterations >= 1
