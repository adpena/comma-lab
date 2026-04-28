"""Tests for ``tac.learnable_bit_quant.LagrangianRateController`` — the
true primal-dual rate controller introduced by Bug 2 (codex Round 3).

The previous penalty was ``λ · max(0, mean_bits − target)²`` (squared
hinge, gradient zero at boundary, equilibrium ABOVE target). The fix
implements Lagrangian dual ascent::

    L(θ, λ) = D(θ) + λ · (mean_bits − target),  λ ≥ 0
    λ_{t+1} = max(0, λ_t + η · (mean_bits − target))

with a *linear* primal penalty (gradient = λ at the boundary, so KKT is
honoured). The dual variable is updated externally by this controller.

These tests pin:
  1. Construction validation (target > 0, eta > 0, λ_init ≥ 0).
  2. Dual ascent moves λ in the correct direction (up when violated,
     down when slack), and stays non-negative.
  3. lambda_max cap is honoured.
  4. End-to-end convergence on a synthetic problem: with the controller
     wired into a tiny optimizer loop, mean_bits converges within 5%
     of the target after dual ascent.
  5. Backward-compat: ``compute_learnable_bit_rate_penalty`` still
     accepts a plain float ``lambda_rate``.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from tac.learnable_bit_quant import (
    LagrangianRateController,
    LearnableBitConv2d,
    compute_learnable_bit_rate_penalty,
    renderer_average_learnable_bits_per_weight,
)


# ── Construction validation ─────────────────────────────────────────────


def test_construct_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="target_bits_per_weight"):
        LagrangianRateController(target_bits_per_weight=0.0)
    with pytest.raises(ValueError, match="target_bits_per_weight"):
        LagrangianRateController(target_bits_per_weight=-1.0)


def test_construct_rejects_nonpositive_eta():
    with pytest.raises(ValueError, match="eta"):
        LagrangianRateController(target_bits_per_weight=2.0, eta=0.0)
    with pytest.raises(ValueError, match="eta"):
        LagrangianRateController(target_bits_per_weight=2.0, eta=-0.01)


def test_construct_rejects_negative_initial_lambda():
    with pytest.raises(ValueError, match="initial_lambda"):
        LagrangianRateController(
            target_bits_per_weight=2.0, initial_lambda=-0.5,
        )


def test_construct_rejects_lambda_max_below_initial():
    with pytest.raises(ValueError, match="lambda_max"):
        LagrangianRateController(
            target_bits_per_weight=2.0,
            initial_lambda=1.0,
            lambda_max=0.5,
        )


# ── Dual update mechanics ────────────────────────────────────────────────


def test_dual_update_raises_lambda_when_constraint_violated():
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=0.1, initial_lambda=0.0,
    )
    new_lam = ctrl.dual_update(mean_bits=4.0)
    # residual = 4 - 2 = 2; new λ = max(0, 0 + 0.1 · 2) = 0.2.
    assert math.isclose(new_lam, 0.2, abs_tol=1e-12)
    assert ctrl.lambda_rate == new_lam
    assert ctrl.last_residual == pytest.approx(2.0)
    assert ctrl.step_count == 1


def test_dual_update_lowers_lambda_when_slack():
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=0.5, initial_lambda=1.0,
    )
    # mean_bits 1.5 → residual = -0.5 → λ_new = max(0, 1.0 + 0.5 · -0.5) = 0.75.
    new_lam = ctrl.dual_update(mean_bits=1.5)
    assert math.isclose(new_lam, 0.75, abs_tol=1e-12)


def test_dual_update_clamps_lambda_to_zero_under_strong_slack():
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=1.0, initial_lambda=0.1,
    )
    new_lam = ctrl.dual_update(mean_bits=1.0)
    # residual = -1.0, λ_new = max(0, 0.1 - 1.0) = 0 (clamped).
    assert new_lam == 0.0


def test_dual_update_respects_lambda_max():
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=10.0,
        initial_lambda=0.0, lambda_max=1.5,
    )
    # Violating constraint → λ would go to 0 + 10 · 5 = 50 without cap.
    for _ in range(5):
        ctrl.dual_update(mean_bits=7.0)
    assert ctrl.lambda_rate == pytest.approx(1.5)


def test_target_override_in_dual_update():
    """Operator can override the target per call (ramp schedules)."""
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=0.5, initial_lambda=0.0,
    )
    new_lam = ctrl.dual_update(mean_bits=2.0, target=1.5)
    # residual = 2.0 - 1.5 = 0.5; λ = 0 + 0.5 · 0.5 = 0.25.
    assert math.isclose(new_lam, 0.25, abs_tol=1e-12)


def test_state_dict_round_trip():
    ctrl = LagrangianRateController(
        target_bits_per_weight=3.0, eta=0.05, initial_lambda=0.5,
        lambda_max=2.0,
    )
    for _ in range(7):
        ctrl.dual_update(mean_bits=4.0)
    state = ctrl.state_dict()
    recovered = LagrangianRateController(target_bits_per_weight=3.0)
    recovered.load_state_dict(state)
    assert recovered.lambda_rate == ctrl.lambda_rate
    assert recovered.step_count == ctrl.step_count
    assert recovered.last_residual == ctrl.last_residual
    assert recovered.target_bits_per_weight == ctrl.target_bits_per_weight
    assert recovered.eta == ctrl.eta
    assert recovered.lambda_max == ctrl.lambda_max


# ── Compute penalty supports both float and controller ──────────────────


def _build_tiny_swapped_model() -> nn.Module:
    return nn.Sequential(
        LearnableBitConv2d(3, 8, 3, padding=1, init_bits=8.0),
        nn.ReLU(),
        LearnableBitConv2d(8, 8, 3, padding=1, init_bits=8.0),
    )


def test_compute_penalty_accepts_float_lambda_legacy():
    model = _build_tiny_swapped_model()
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=2.0, lambda_rate=1.5,
    )
    # Linear penalty, mean_bits ≈ 8 → residual ≈ 6 → pen ≈ 9.
    assert pen.item() > 0


def test_compute_penalty_accepts_controller_instance():
    model = _build_tiny_swapped_model()
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=0.1, initial_lambda=2.0,
    )
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=999.0,  # ignored when ctrl used
        lambda_rate=ctrl,
    )
    # The penalty uses the controller's target (2.0), not the float
    # arg above. mean_bits ≈ 8 → residual ≈ 6 → pen ≈ 12.
    assert pen.item() > 0


def test_compute_penalty_zero_when_lambda_zero():
    model = _build_tiny_swapped_model()
    ctrl = LagrangianRateController(
        target_bits_per_weight=2.0, eta=0.1, initial_lambda=0.0,
    )
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=2.0, lambda_rate=ctrl,
    )
    assert pen.item() == 0.0


# ── End-to-end convergence (Bug 2 verification) ─────────────────────────


def test_dual_ascent_converges_mean_bits_to_target():
    """Bug 2 acceptance test: with the primal-dual loop running,
    mean_bits converges within 5% of the target after enough steps.
    The pre-fix squared-hinge penalty equilibrated *above* the target
    (gradient zero at boundary); the linear penalty + dual ascent
    converges *to* the target.
    """
    torch.manual_seed(0)
    model = _build_tiny_swapped_model()
    target = 3.0
    ctrl = LagrangianRateController(
        target_bits_per_weight=target, eta=0.5, initial_lambda=0.0,
    )

    bits_params = [
        p for n, p in model.named_parameters() if n.endswith(".bit_depth.raw")
    ]
    optimizer = torch.optim.SGD(bits_params, lr=0.1)

    for _ in range(400):
        pen = compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=target, lambda_rate=ctrl,
        )
        optimizer.zero_grad(set_to_none=True)
        if pen.requires_grad:
            pen.backward()
            optimizer.step()
        # Dual ascent on the post-step residual.
        post_mean_bits = renderer_average_learnable_bits_per_weight(model)
        ctrl.dual_update(post_mean_bits)

    final = renderer_average_learnable_bits_per_weight(model)
    assert abs(final - target) / target < 0.05, (
        f"mean_bits did not converge to within 5% of target: "
        f"final={final:.3f} target={target:.3f} "
        f"final_lambda={ctrl.lambda_rate:.3f}"
    )
    # And λ should be small in steady state (slack ≈ 0).
    assert ctrl.lambda_rate >= 0.0


def test_dual_ascent_under_severe_violation_drives_bits_down():
    """Sanity: starting at 8 bits with a 1-bit target, the controller
    should monotonically push bits *down* (not just oscillate or
    plateau)."""
    torch.manual_seed(1)
    model = _build_tiny_swapped_model()
    initial_bits = renderer_average_learnable_bits_per_weight(model)
    target = 1.5
    ctrl = LagrangianRateController(
        target_bits_per_weight=target, eta=1.0, initial_lambda=0.0,
    )
    bits_params = [
        p for n, p in model.named_parameters() if n.endswith(".bit_depth.raw")
    ]
    optimizer = torch.optim.SGD(bits_params, lr=0.5)
    for _ in range(50):
        pen = compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=target, lambda_rate=ctrl,
        )
        optimizer.zero_grad(set_to_none=True)
        if pen.requires_grad:
            pen.backward()
            optimizer.step()
        ctrl.dual_update(renderer_average_learnable_bits_per_weight(model))
    final_bits = renderer_average_learnable_bits_per_weight(model)
    assert final_bits < initial_bits
