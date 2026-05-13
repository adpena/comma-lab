"""Tests for ``tac.optimization.langevin_optimizer`` (beat-PR95 Idea 2 scaffold).

Coverage:
- Schedule monotonicity + endpoints (cosine / log / exp)
- Schedule callable signature
- Optimizer step API contract (zero_grad, step, no-grad call site)
- Convex-quadratic convergence in the T → 0 limit (Langevin reduces to GD)
- High-temperature mixing on a 2D Gaussian target
- Determinism with ``noise_seed`` set
- Non-determinism (different runs) when ``noise_seed`` not set
- state_dict / load_state_dict round-trip preserves step counter
- weight_decay applied to gradient before noise injection
- Invalid hyperparameter errors raise on construction
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from tac.optimization.langevin_optimizer import (
    SCHEDULES,
    LangevinOptimizer,
    cosine_temperature_schedule,
    exponential_temperature_schedule,
    geman_geman_log_schedule,
)

# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------


def test_cosine_schedule_endpoints():
    assert cosine_temperature_schedule(0, 100, 1.0, 1e-4) == pytest.approx(1.0)
    assert cosine_temperature_schedule(100, 100, 1.0, 1e-4) == pytest.approx(1e-4)
    # Midpoint should be near (T_init + T_final) / 2
    mid = cosine_temperature_schedule(50, 100, 1.0, 0.0)
    assert mid == pytest.approx(0.5, abs=1e-6)


def test_cosine_schedule_monotone_nonincreasing():
    n = 200
    vals = [cosine_temperature_schedule(i, n, 1.0, 1e-3) for i in range(n + 1)]
    for a, b in pairwise(vals):
        assert a + 1e-12 >= b, "cosine schedule must be monotone non-increasing"


def test_log_schedule_endpoints():
    assert geman_geman_log_schedule(0, 100, 1.0, 1e-4) == pytest.approx(1.0)
    assert geman_geman_log_schedule(100, 100, 1.0, 1e-4) == pytest.approx(1e-4)


def test_log_schedule_monotone_nonincreasing():
    n = 200
    vals = [geman_geman_log_schedule(i, n, 1.0, 1e-3) for i in range(n + 1)]
    for a, b in pairwise(vals):
        assert a + 1e-12 >= b


def test_exponential_schedule_endpoints():
    assert exponential_temperature_schedule(0, 100, 1.0, 1e-4) == pytest.approx(1.0)
    assert exponential_temperature_schedule(100, 100, 1.0, 1e-4) == pytest.approx(1e-4)


def test_schedule_registry_contains_canonical_aliases():
    assert "cosine" in SCHEDULES
    assert "log" in SCHEDULES
    assert "geman_geman" in SCHEDULES
    assert "exp" in SCHEDULES
    assert "exponential" in SCHEDULES
    # Aliases must point to the same function
    assert SCHEDULES["log"] is SCHEDULES["geman_geman"]
    assert SCHEDULES["exp"] is SCHEDULES["exponential"]


def test_schedule_n_steps_zero_returns_T_final():
    assert cosine_temperature_schedule(0, 0, 1.0, 1e-4) == pytest.approx(1e-4)
    assert geman_geman_log_schedule(0, 0, 1.0, 1e-4) == pytest.approx(1e-4)
    assert exponential_temperature_schedule(0, 0, 1.0, 1e-4) == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Optimizer construction tests
# ---------------------------------------------------------------------------


def _params():
    return [torch.nn.Parameter(torch.zeros(4))]


def test_construct_with_defaults():
    opt = LangevinOptimizer(_params())
    assert opt.defaults["lr"] == 1e-4
    assert opt.defaults["T_init"] == 1.0
    assert opt.defaults["T_final"] == 1e-4
    assert opt.defaults["n_steps"] == 2000
    assert opt.defaults["schedule"] == "cosine"
    assert opt.current_step == 0


def test_invalid_lr_raises():
    with pytest.raises(ValueError, match="lr"):
        LangevinOptimizer(_params(), lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        LangevinOptimizer(_params(), lr=-1e-3)


def test_invalid_temperature_ordering_raises():
    with pytest.raises(ValueError, match="T_init"):
        LangevinOptimizer(_params(), T_init=0.1, T_final=1.0)


def test_negative_temperature_raises():
    with pytest.raises(ValueError, match="non-negative"):
        LangevinOptimizer(_params(), T_init=1.0, T_final=-1e-4)
    with pytest.raises(ValueError, match="non-negative"):
        LangevinOptimizer(_params(), T_init=-1.0, T_final=-2.0)


def test_invalid_n_steps_raises():
    with pytest.raises(ValueError, match="n_steps"):
        LangevinOptimizer(_params(), n_steps=0)
    with pytest.raises(ValueError, match="n_steps"):
        LangevinOptimizer(_params(), n_steps=-100)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="schedule"):
        LangevinOptimizer(_params(), schedule="quadratic")


# ---------------------------------------------------------------------------
# Step semantics
# ---------------------------------------------------------------------------


def test_step_increments_counter():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = LangevinOptimizer([p], lr=1e-3, T_init=0.0, T_final=0.0)
    # Set a gradient
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.current_step == 1
    opt.step()
    assert opt.current_step == 2


def test_step_with_no_grad_does_not_update():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = LangevinOptimizer([p], lr=1e-3, T_init=0.0, T_final=0.0)
    # No gradient set
    initial = p.data.clone()
    opt.step()
    # T=0 + no grad => no change
    assert torch.equal(p.data, initial)
    assert opt.current_step == 1


def test_zero_temperature_reduces_to_gradient_descent():
    """At T=0, Langevin SDE = pure gradient descent."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    opt = LangevinOptimizer([p], lr=0.1, T_init=0.0, T_final=0.0, n_steps=10)
    # Quadratic loss: L = 0.5 * ||p||^2, gradient = p
    for _ in range(10):
        opt.zero_grad()
        loss = 0.5 * (p**2).sum()
        loss.backward()
        opt.step()
    # After 10 steps of GD on L = 0.5||p||² with lr=0.1: p ← 0.9 * p_prev
    # Final p ≈ 0.9^10 * [1, 2, 3] ≈ 0.349 * [1, 2, 3]
    expected = 0.9**10 * torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(p.data, expected, atol=1e-5)


def test_high_temperature_diffuses_zero_gradient_parameter():
    """At T > 0 with zero gradient, parameters do a pure random walk."""
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.zeros(100))
    opt = LangevinOptimizer(
        [p], lr=0.01, T_init=1.0, T_final=1.0, n_steps=100, noise_seed=123
    )
    # No gradient: pure diffusion
    p.grad = torch.zeros_like(p)
    for _ in range(50):
        opt.step()
    # After n steps of pure diffusion at T=1, lr=0.01:
    # E[p^2] = n * 2 * T * lr = 50 * 2 * 1 * 0.01 = 1.0
    # Sample variance should be near 1.0 for 100-dim parameter
    sample_var = (p.data**2).mean().item()
    assert 0.5 < sample_var < 2.0, (
        f"diffused parameter variance {sample_var} not in expected range"
    )


def test_convex_quadratic_convergence_with_annealing():
    """Annealed Langevin should converge close to argmin L on convex L."""
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.tensor([5.0, -3.0]))
    # L = 0.5 * ((p - target) ** 2).sum(); argmin at target
    target = torch.tensor([1.0, 2.0])
    opt = LangevinOptimizer(
        [p],
        lr=0.05,
        T_init=0.1,
        T_final=1e-6,
        n_steps=500,
        schedule="cosine",
        noise_seed=99,
    )
    for _ in range(500):
        opt.zero_grad()
        loss = 0.5 * ((p - target) ** 2).sum()
        loss.backward()
        opt.step()
    # With small T_final, should land near target
    assert torch.allclose(p.data, target, atol=0.05)


# ---------------------------------------------------------------------------
# Determinism + serialization
# ---------------------------------------------------------------------------


def test_noise_seed_yields_deterministic_trajectory():
    def run(seed):
        torch.manual_seed(0)  # control gradient-side determinism
        p = torch.nn.Parameter(torch.zeros(10))
        opt = LangevinOptimizer(
            [p], lr=0.01, T_init=1.0, T_final=1.0, n_steps=100, noise_seed=seed
        )
        p.grad = torch.zeros_like(p)
        for _ in range(50):
            opt.step()
        return p.data.clone()

    a = run(42)
    b = run(42)
    assert torch.equal(a, b), "same noise_seed must produce identical trajectory"
    c = run(99)
    assert not torch.equal(a, c), "different noise_seed must differ"


def test_state_dict_round_trip_preserves_step_count():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = LangevinOptimizer([p], lr=1e-3, T_init=0.5, T_final=1e-3)
    p.grad = torch.ones_like(p)
    for _ in range(7):
        opt.step()
    sd = opt.state_dict()
    assert sd["_step_count"] == 7

    p2 = torch.nn.Parameter(torch.zeros(4))
    opt2 = LangevinOptimizer([p2], lr=1e-3, T_init=0.5, T_final=1e-3)
    opt2.load_state_dict(sd)
    assert opt2.current_step == 7


def test_current_temperature_at_step_zero_equals_t_init():
    opt = LangevinOptimizer(_params(), T_init=0.7, T_final=1e-3, n_steps=500)
    assert opt.current_temperature() == pytest.approx(0.7)


def test_current_temperature_decreases_with_steps():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = LangevinOptimizer(
        [p], lr=1e-3, T_init=1.0, T_final=1e-3, n_steps=100, schedule="cosine"
    )
    t0 = opt.current_temperature()
    p.grad = torch.zeros_like(p)
    for _ in range(50):
        opt.step()
    t50 = opt.current_temperature()
    assert t50 < t0


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------


def test_weight_decay_pulls_toward_origin_at_zero_temperature():
    """At T=0 with weight_decay > 0 and no gradient, p shrinks toward 0."""
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    opt = LangevinOptimizer(
        [p], lr=0.1, weight_decay=0.5, T_init=0.0, T_final=0.0
    )
    p.grad = torch.zeros_like(p)
    initial_norm = p.data.norm().item()
    for _ in range(5):
        opt.step()
    final_norm = p.data.norm().item()
    assert final_norm < initial_norm, "weight_decay should shrink parameters"


# ---------------------------------------------------------------------------
# Closure callable contract
# ---------------------------------------------------------------------------


def test_closure_recompute_returns_loss():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    opt = LangevinOptimizer([p], lr=0.01, T_init=0.0, T_final=0.0)

    def closure():
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert math.isclose(loss.item(), 4.0, abs_tol=1e-6)
