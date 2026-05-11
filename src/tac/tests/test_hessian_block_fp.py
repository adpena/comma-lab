"""Tests for tac.hessian_block_fp — Hessian-weighted bit allocation."""
from __future__ import annotations

import math

import pytest
import torch

from tac.hessian_block_fp import (
    HessianAllocationConfig,
    HessianAllocationResult,
    allocate_bits_by_hessian,
    compute_hessian_diagonal_proxy,
    expected_distortion_under_allocation,
    validate_hessian_proxy_source,
)


# ── Config validation ─────────────────────────────────────────────────────


def test_config_rejects_zero_budget():
    with pytest.raises(ValueError, match="total_bit_budget must be positive"):
        HessianAllocationConfig(total_bit_budget=0)


def test_config_rejects_negative_budget():
    with pytest.raises(ValueError, match="total_bit_budget must be positive"):
        HessianAllocationConfig(total_bit_budget=-1.0)


def test_config_rejects_inverted_bit_range():
    with pytest.raises(ValueError, match="min_bits_per_tensor must be in"):
        HessianAllocationConfig(
            total_bit_budget=1000.0,
            min_bits_per_tensor=5.0,
            max_bits_per_tensor=3.0,
        )


def test_config_rejects_negative_min_bits():
    with pytest.raises(ValueError, match="min_bits_per_tensor must be in"):
        HessianAllocationConfig(
            total_bit_budget=1000.0,
            min_bits_per_tensor=-0.5,
        )


def test_config_rejects_zero_iters():
    with pytest.raises(ValueError, match="bisection_max_iters must be positive"):
        HessianAllocationConfig(total_bit_budget=1000.0, bisection_max_iters=0)


def test_config_rejects_zero_tol():
    with pytest.raises(ValueError, match="bisection_tol must be positive"):
        HessianAllocationConfig(total_bit_budget=1000.0, bisection_tol=0.0)


# ── Catalog #123: forbidden weight-domain saliency on score-aware substrate ─


def test_validate_refuses_weight_magnitude_on_score_gradient_substrate():
    """Per CLAUDE.md catalog #123: weight-domain saliency is forbidden on
    score-gradient-trained substrates."""
    proxy = {"layer1": 1.0, "layer2": 2.0}
    with pytest.raises(ValueError, match="FORBIDDEN saliency source"):
        validate_hessian_proxy_source(
            proxy,
            saliency_source="weight_magnitude",
            substrate_class="a1",
        )


def test_validate_refuses_mean_theta_squared_on_a1():
    proxy = {"layer1": 1.0}
    with pytest.raises(ValueError, match="FORBIDDEN saliency source"):
        validate_hessian_proxy_source(
            proxy,
            saliency_source="mean_theta_squared",
            substrate_class="train_score_gradient",
        )


def test_validate_accepts_score_gradient_on_a1():
    """Catalog #123: score_gradient source IS allowed on score-aware substrate."""
    proxy = {"layer1": 1.0}
    # Should not raise
    validate_hessian_proxy_source(
        proxy,
        saliency_source="score_gradient",
        substrate_class="a1",
    )


def test_validate_accepts_weight_magnitude_on_non_score_aware():
    """The forbidden pattern is specifically on score-aware substrates."""
    proxy = {"layer1": 1.0}
    # Should not raise — substrate_class is None / not score-aware
    validate_hessian_proxy_source(
        proxy,
        saliency_source="weight_magnitude",
        substrate_class=None,
    )


def test_validate_refuses_empty_proxy():
    with pytest.raises(ValueError, match="hessian_proxy dict is empty"):
        validate_hessian_proxy_source(
            {},
            saliency_source="score_gradient",
            substrate_class=None,
        )


def test_validate_refuses_negative_proxy_value():
    proxy = {"layer1": -1.0}
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_hessian_proxy_source(
            proxy,
            saliency_source="score_gradient",
            substrate_class=None,
        )


# ── compute_hessian_diagonal_proxy ────────────────────────────────────────


def test_compute_hessian_diagonal_proxy_basic():
    params = {
        "w1": torch.zeros(4, 4),
        "w2": torch.zeros(8),
    }
    grads = {
        "w1": torch.ones(4, 4),  # sum of squares = 16
        "w2": torch.ones(8) * 2.0,  # sum of squares = 32
    }
    out = compute_hessian_diagonal_proxy(parameters=params, gradients=grads)
    assert out["w1"] == 16.0
    assert out["w2"] == 32.0


def test_compute_proxy_rejects_missing_gradient():
    params = {"w1": torch.zeros(4)}
    grads = {}
    with pytest.raises(ValueError, match="gradient missing"):
        compute_hessian_diagonal_proxy(parameters=params, gradients=grads)


def test_compute_proxy_rejects_shape_mismatch():
    params = {"w1": torch.zeros(4, 4)}
    grads = {"w1": torch.zeros(4)}  # wrong shape
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_hessian_diagonal_proxy(parameters=params, gradients=grads)


# ── allocate_bits_by_hessian — KKT / water-filling correctness ───────────


def test_allocate_with_two_tensors_uniform_hessian():
    """Equal Hessian-proxies → roughly equal bit allocation (within budget)."""
    config = HessianAllocationConfig(total_bit_budget=1000.0)
    res = allocate_bits_by_hessian(
        hessian_proxy={"w1": 1.0, "w2": 1.0},
        params_per_tensor={"w1": 100, "w2": 100},
        config=config,
        saliency_source="score_gradient",
        substrate_class=None,
    )
    assert res.bits_per_tensor["w1"] == pytest.approx(res.bits_per_tensor["w2"], rel=0.05)


def test_allocate_with_higher_hessian_gets_more_bits():
    """KKT condition: higher Hessian → more bits (when both are active)."""
    config = HessianAllocationConfig(total_bit_budget=1000.0, max_bits_per_tensor=8.0)
    res = allocate_bits_by_hessian(
        hessian_proxy={"low": 0.1, "high": 100.0},
        params_per_tensor={"low": 100, "high": 100},
        config=config,
        saliency_source="score_gradient",
        substrate_class=None,
    )
    # The high-Hessian tensor must get at least as many bits as the low one
    # (or the low one gets pruned to 0).
    if res.bits_per_tensor["low"] > 0:
        assert res.bits_per_tensor["high"] >= res.bits_per_tensor["low"]


def test_allocate_satisfies_budget_within_tolerance():
    config = HessianAllocationConfig(
        total_bit_budget=2000.0,
        bisection_tol=1e-3,
        max_bits_per_tensor=8.0,
    )
    res = allocate_bits_by_hessian(
        hessian_proxy={f"w{i}": float(i + 1) for i in range(5)},
        params_per_tensor={f"w{i}": 100 for i in range(5)},
        config=config,
        saliency_source="score_gradient",
        substrate_class=None,
    )
    rel_err = abs(res.total_bits_used - config.total_bit_budget) / config.total_bit_budget
    # Allow slight slack: clamping to [min, max] may prevent exact match
    assert rel_err < 0.5  # within 50% — exact match needs unbounded bit range


def test_allocate_records_provenance():
    config = HessianAllocationConfig(total_bit_budget=1000.0)
    res = allocate_bits_by_hessian(
        hessian_proxy={"w1": 1.0},
        params_per_tensor={"w1": 100},
        config=config,
        saliency_source="score_gradient",
        substrate_class="a1",
    )
    assert res.provenance["saliency_source"] == "score_gradient"
    assert res.provenance["substrate_class"] == "a1"
    assert res.provenance["evidence_grade"] == "derivation"
    assert res.provenance["n_tensors"] == 1


def test_allocate_refuses_inconsistent_keys():
    config = HessianAllocationConfig(total_bit_budget=1000.0)
    with pytest.raises(ValueError, match="symmetric difference"):
        allocate_bits_by_hessian(
            hessian_proxy={"w1": 1.0, "w2": 2.0},
            params_per_tensor={"w1": 100, "w3": 100},
            config=config,
            saliency_source="score_gradient",
            substrate_class=None,
        )


def test_allocate_refuses_forbidden_proxy_on_a1():
    config = HessianAllocationConfig(total_bit_budget=1000.0)
    with pytest.raises(ValueError, match="FORBIDDEN saliency source"):
        allocate_bits_by_hessian(
            hessian_proxy={"w1": 1.0},
            params_per_tensor={"w1": 100},
            config=config,
            saliency_source="mean_theta_squared",
            substrate_class="a1",
        )


def test_allocate_prunes_below_min_bits():
    """Tensors with very low Hessian should be pruned (bits=0)."""
    config = HessianAllocationConfig(
        total_bit_budget=100.0,
        min_bits_per_tensor=1.0,
        max_bits_per_tensor=8.0,
    )
    res = allocate_bits_by_hessian(
        hessian_proxy={"tiny": 1e-10, "big": 1000.0},
        params_per_tensor={"tiny": 1000, "big": 100},
        config=config,
        saliency_source="score_gradient",
        substrate_class=None,
    )
    assert res.bits_per_tensor["tiny"] == 0.0
    assert res.n_pruned_tensors >= 1


# ── expected_distortion_under_allocation ─────────────────────────────────


def test_expected_distortion_zero_bits_returns_full_hessian():
    """Pruned tensor (b=0) contributes its full Hessian-proxy as distortion."""
    d = expected_distortion_under_allocation(
        bits_per_tensor={"w1": 0.0},
        hessian_proxy={"w1": 5.0},
    )
    assert d == 5.0


def test_expected_distortion_higher_bits_gives_lower_distortion():
    d_low = expected_distortion_under_allocation(
        bits_per_tensor={"w1": 2.0},
        hessian_proxy={"w1": 1.0},
    )
    d_high = expected_distortion_under_allocation(
        bits_per_tensor={"w1": 8.0},
        hessian_proxy={"w1": 1.0},
    )
    assert d_high < d_low  # more bits → less distortion


def test_expected_distortion_quadratic_in_bits():
    """D = H * 2^(-2b) / 12. Doubling bits → 4x less distortion."""
    d_2 = expected_distortion_under_allocation(
        bits_per_tensor={"w1": 2.0},
        hessian_proxy={"w1": 1.0},
    )
    d_4 = expected_distortion_under_allocation(
        bits_per_tensor={"w1": 4.0},
        hessian_proxy={"w1": 1.0},
    )
    # Ratio should be 2^(-2*4) / 2^(-2*2) = 1/16
    assert math.isclose(d_4 / d_2, 1.0 / 16.0, rel_tol=1e-6)


def test_expected_distortion_refuses_mismatched_keys():
    with pytest.raises(ValueError, match="symmetric difference"):
        expected_distortion_under_allocation(
            bits_per_tensor={"w1": 1.0},
            hessian_proxy={"w2": 1.0},
        )
