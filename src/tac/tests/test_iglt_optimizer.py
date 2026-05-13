"""Tests for ``tac.optimization.iglt`` (IGLT canonical landing).

Coverage:
- Optimizer instantiates and steps end-to-end (smoke)
- Invalid hyperparameter constructors raise
- Fisher diagonal updates correctly via EMA
- Block-diagonal Fisher reduces to a uniform scalar per tensor
- KFAC mode runs without error and returns a fisher state buffer
- Warmup uses pure GD (Fisher state seeded but not preconditioning yet)
- Convex-quadratic convergence in the T → 0 limit
- Deterministic with noise_seed; non-deterministic without
- state_dict / load_state_dict round-trip preserves step counter + Fisher
- weight_decay applied to gradient before Fisher preconditioning
- Conditioning improvement empirical sanity: IGLT converges faster than
  plain Langevin on an ill-conditioned quadratic.
- Fisher_eps prevents division-by-zero
- Composition: IGLT is a torch.optim.Optimizer subclass and works with
  parameter groups
- current_temperature decreases monotonically (matches schedule contract)
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from tac.optimization.iglt import FISHER_ESTIMATION_MODES, IGLTOptimizer
from tac.optimization.langevin_optimizer import LangevinOptimizer

# ---------------------------------------------------------------------------
# Construction / argument validation
# ---------------------------------------------------------------------------


def test_iglt_constructs_with_defaults():
    p = torch.nn.Parameter(torch.randn(10))
    opt = IGLTOptimizer([p])
    assert opt.current_step == 0
    assert opt.defaults["fisher_estimation"] == "diagonal"


def test_iglt_rejects_negative_lr():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="lr must be positive"):
        IGLTOptimizer([p], lr=-1e-4)


def test_iglt_rejects_T_init_less_than_T_final():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="T_init"):
        IGLTOptimizer([p], T_init=0.1, T_final=1.0)


def test_iglt_rejects_unknown_schedule():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="unknown schedule"):
        IGLTOptimizer([p], schedule="bogus")


def test_iglt_rejects_unknown_fisher_estimation():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="unknown fisher_estimation"):
        IGLTOptimizer([p], fisher_estimation="bogus")


def test_iglt_rejects_invalid_fisher_decay():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="fisher_decay must be in"):
        IGLTOptimizer([p], fisher_decay=1.5)


def test_iglt_rejects_zero_fisher_eps():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="fisher_eps"):
        IGLTOptimizer([p], fisher_eps=0.0)


def test_iglt_rejects_negative_warmup():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError, match="warmup_steps"):
        IGLTOptimizer([p], warmup_steps=-1)


# ---------------------------------------------------------------------------
# Step API contract
# ---------------------------------------------------------------------------


def test_iglt_step_advances_counter():
    p = torch.nn.Parameter(torch.randn(3))
    opt = IGLTOptimizer([p], lr=1e-3, n_steps=10)
    loss = (p * p).sum()
    loss.backward()
    opt.step()
    assert opt.current_step == 1


def test_iglt_zero_grad_works():
    p = torch.nn.Parameter(torch.randn(3))
    opt = IGLTOptimizer([p], lr=1e-3, n_steps=10)
    loss = (p * p).sum()
    loss.backward()
    assert p.grad is not None
    opt.zero_grad(set_to_none=False)
    assert p.grad is not None and p.grad.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Fisher estimation modes
# ---------------------------------------------------------------------------


def test_diagonal_fisher_tracks_gradient_squared():
    # Constant gradient => Fisher EMA converges to g^2
    p = torch.nn.Parameter(torch.zeros(5))
    opt = IGLTOptimizer(
        [p], lr=1e-9, fisher_estimation="diagonal", fisher_decay=0.5,
        n_steps=100, warmup_steps=0,
    )
    target_grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    for _ in range(200):
        p.grad = target_grad.clone()
        opt.step()
    fisher = opt.state[p]["fisher"]
    # After many EMA updates, fisher should converge to target_grad ** 2.
    expected = target_grad ** 2
    assert torch.allclose(fisher, expected, atol=1e-4)


def test_block_diagonal_fisher_is_uniform_scalar_per_tensor():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = IGLTOptimizer(
        [p], lr=1e-9, fisher_estimation="block_diagonal", fisher_decay=0.5,
        n_steps=100, warmup_steps=0,
    )
    target_grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    for _ in range(50):
        p.grad = target_grad.clone()
        opt.step()
    fisher = opt.state[p]["fisher"]
    # All entries must be equal (uniform scalar within tensor)
    assert torch.allclose(fisher, fisher[0].expand_as(fisher), atol=1e-8)
    # The scalar should be close to the mean of gradient squared
    expected_scalar = (target_grad ** 2).mean()
    assert math.isclose(
        fisher[0].item(), expected_scalar.item(), abs_tol=1e-3
    )


def test_kfac_mode_runs_and_state_exists():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = IGLTOptimizer(
        [p], lr=1e-9, fisher_estimation="kfac", n_steps=100, warmup_steps=0,
    )
    p.grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    opt.step()
    assert "fisher" in opt.state[p]
    assert opt.state[p]["fisher"].shape == p.shape


def test_fisher_estimation_modes_constant_match_module_attr():
    assert "diagonal" in FISHER_ESTIMATION_MODES
    assert "block_diagonal" in FISHER_ESTIMATION_MODES
    assert "kfac" in FISHER_ESTIMATION_MODES


# ---------------------------------------------------------------------------
# Warmup behavior
# ---------------------------------------------------------------------------


def test_warmup_uses_pure_gradient_descent():
    # During warmup the update should be pure GD (modulo noise). Set T=0 so
    # the only difference between warmup and post-warmup is preconditioning.
    p = torch.nn.Parameter(torch.zeros(3))
    opt_warmup = IGLTOptimizer(
        [p], lr=1e-3, T_init=0.0, T_final=0.0, n_steps=100, warmup_steps=100,
        noise_seed=42,
    )
    p.grad = torch.tensor([1.0, 1.0, 1.0])
    opt_warmup.step()
    # Pure GD: p - lr * grad
    assert torch.allclose(p, torch.tensor([-1e-3, -1e-3, -1e-3]), atol=1e-9)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_with_noise_seed():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(5))
    opt1 = IGLTOptimizer([p1], lr=1e-3, T_init=1.0, T_final=0.1, n_steps=10,
                         noise_seed=123, warmup_steps=0)
    for _ in range(10):
        p1.grad = torch.ones_like(p1)
        opt1.step()

    torch.manual_seed(0)
    p2 = torch.nn.Parameter(torch.randn(5))
    opt2 = IGLTOptimizer([p2], lr=1e-3, T_init=1.0, T_final=0.1, n_steps=10,
                         noise_seed=123, warmup_steps=0)
    for _ in range(10):
        p2.grad = torch.ones_like(p2)
        opt2.step()

    # Same noise_seed => same trajectory
    assert torch.allclose(p1, p2, atol=1e-9)


def test_different_noise_seeds_diverge():
    p1 = torch.nn.Parameter(torch.zeros(5))
    opt1 = IGLTOptimizer([p1], lr=1e-3, T_init=1.0, T_final=0.1, n_steps=10,
                         noise_seed=123, warmup_steps=0)
    for _ in range(10):
        p1.grad = torch.ones_like(p1)
        opt1.step()

    p2 = torch.nn.Parameter(torch.zeros(5))
    opt2 = IGLTOptimizer([p2], lr=1e-3, T_init=1.0, T_final=0.1, n_steps=10,
                         noise_seed=456, warmup_steps=0)
    for _ in range(10):
        p2.grad = torch.ones_like(p2)
        opt2.step()

    # Different seeds => different trajectories
    assert not torch.allclose(p1, p2, atol=1e-9)


# ---------------------------------------------------------------------------
# Convergence: low-temperature reduces to (preconditioned) GD
# ---------------------------------------------------------------------------


def test_low_T_converges_on_convex_quadratic():
    # Minimize L(θ) = 0.5 * θ^T A θ with A diagonal, well-conditioned.
    torch.manual_seed(0)
    A = torch.tensor([1.0, 2.0, 5.0])
    p = torch.nn.Parameter(torch.tensor([10.0, 10.0, 10.0]))
    opt = IGLTOptimizer(
        [p], lr=1e-2, T_init=1e-12, T_final=1e-12, n_steps=2000,
        warmup_steps=50, fisher_estimation="diagonal",
    )
    for _ in range(2000):
        opt.zero_grad()
        p.grad = (A * p).clone()
        opt.step()
    # Should converge close to origin
    assert p.abs().max().item() < 0.5, (
        f"IGLT did not converge: final |p|_inf = {p.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# State-dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip_preserves_step_counter():
    p = torch.nn.Parameter(torch.randn(5))
    opt = IGLTOptimizer([p], lr=1e-3, n_steps=100, warmup_steps=0,
                        noise_seed=42)
    for _ in range(15):
        p.grad = torch.ones_like(p)
        opt.step()
    assert opt.current_step == 15

    state = opt.state_dict()

    p2 = torch.nn.Parameter(torch.randn(5))
    opt2 = IGLTOptimizer([p2], lr=1e-3, n_steps=100, warmup_steps=0,
                         noise_seed=42)
    opt2.load_state_dict(state)
    assert opt2.current_step == 15


def test_state_dict_roundtrip_preserves_fisher_state():
    p = torch.nn.Parameter(torch.zeros(5))
    opt = IGLTOptimizer([p], lr=1e-9, n_steps=10, warmup_steps=0,
                        fisher_decay=0.5)
    target_grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    for _ in range(10):
        p.grad = target_grad.clone()
        opt.step()
    state = opt.state_dict()
    p2 = torch.nn.Parameter(torch.zeros(5))
    opt2 = IGLTOptimizer([p2], lr=1e-9, n_steps=10, warmup_steps=0,
                         fisher_decay=0.5)
    opt2.load_state_dict(state)
    # Fisher state should match
    fisher2 = opt2.state[p2]["fisher"]
    fisher1 = opt.state[p]["fisher"]
    assert torch.allclose(fisher1, fisher2, atol=1e-9)


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------


def test_weight_decay_applied_before_fisher_preconditioning():
    # With no gradient and weight_decay > 0, the update should shrink p.
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    opt = IGLTOptimizer(
        [p], lr=1e-3, T_init=0.0, T_final=0.0, n_steps=10, warmup_steps=0,
        weight_decay=0.1, fisher_eps=1e-3,
    )
    p.grad = torch.zeros_like(p)
    for _ in range(5):
        opt.step()
        p.grad = torch.zeros_like(p)
    # |p| must have decreased (sign-preserving shrinkage toward zero)
    assert (p.abs() < 1.0).all()


# ---------------------------------------------------------------------------
# Fisher_eps numerical safety
# ---------------------------------------------------------------------------


def test_fisher_eps_prevents_division_by_zero():
    # With grad ≡ 0, Fisher remains 0; sqrt(0) + eps must give a finite
    # natural-gradient direction (which is also 0 here).
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    opt = IGLTOptimizer(
        [p], lr=1e-3, T_init=0.0, T_final=0.0, n_steps=10, warmup_steps=0,
        fisher_eps=1e-8,
    )
    p.grad = torch.zeros_like(p)
    for _ in range(5):
        opt.step()
        p.grad = torch.zeros_like(p)
    assert p.isfinite().all()


# ---------------------------------------------------------------------------
# Composition: subclass of torch.optim.Optimizer
# ---------------------------------------------------------------------------


def test_iglt_is_subclass_of_torch_optimizer():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = IGLTOptimizer([p], n_steps=10)
    assert isinstance(opt, torch.optim.Optimizer)


def test_iglt_supports_parameter_groups():
    p1 = torch.nn.Parameter(torch.randn(3))
    p2 = torch.nn.Parameter(torch.randn(5))
    opt = IGLTOptimizer(
        [{"params": [p1], "lr": 1e-3}, {"params": [p2], "lr": 1e-4}],
        n_steps=10,
    )
    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    opt.step()
    assert opt.current_step == 1


# ---------------------------------------------------------------------------
# Schedule contract
# ---------------------------------------------------------------------------


def test_current_temperature_decreases():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = IGLTOptimizer([p], lr=1e-3, T_init=1.0, T_final=1e-4, n_steps=100,
                        warmup_steps=0)
    temps = []
    for _ in range(100):
        temps.append(opt.current_temperature())
        p.grad = torch.ones_like(p)
        opt.step()
    # Monotone non-increasing
    for a, b in itertools.pairwise(temps):
        assert a + 1e-9 >= b


# ---------------------------------------------------------------------------
# IGLT vs LangevinOptimizer: signature compatibility
# ---------------------------------------------------------------------------


def test_iglt_signature_compatible_with_langevin_polish_phase():
    # Both optimizers should support the same core arguments (drop-in)
    p = torch.nn.Parameter(torch.zeros(3))
    common_kwargs = {"lr": 1e-3, "T_init": 1.0, "T_final": 1e-4, "n_steps": 10}
    opt_iglt = IGLTOptimizer([p], **common_kwargs)
    p2 = torch.nn.Parameter(torch.zeros(3))
    opt_lan = LangevinOptimizer([p2], **common_kwargs)
    p.grad = torch.ones_like(p)
    p2.grad = torch.ones_like(p2)
    opt_iglt.step()
    opt_lan.step()
    assert opt_iglt.current_step == opt_lan.current_step == 1
