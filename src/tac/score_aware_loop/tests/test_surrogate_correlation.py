# SPDX-License-Identifier: MIT
"""Behavior tests for the surrogate<->exact-d_seg correlation harness.

NO FAKE (CLAUDE.md class 2): each test would FAIL if the correlation logic were a
constant/no-op. The heavy real-frame measurement is NOT exercised here (it needs
the EfficientNet-B2 SegNet forward and the video); these test the
correlation/perturbation/segnet-input plumbing on tiny tensors so CI is fast and
deterministic. The real measurement is the durable JSON artifact.
"""

from __future__ import annotations

import math

import torch

from tac.score_aware_loop.surrogate_correlation import (
    DEFAULT_PERTURBATIONS,
    CorrelationResult,
    _pearson_and_ols,
    _spearman,
    _uint8_requant,
)


def test_spearman_perfect_monotone_is_one():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 0.5, 1.5, 1.6, 9.0]  # strictly increasing -> rho == 1
    assert math.isclose(_spearman(x, y), 1.0, abs_tol=1e-9)


def test_spearman_anti_monotone_is_minus_one():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [9.0, 8.0, 7.0, 6.0]
    assert math.isclose(_spearman(x, y), -1.0, abs_tol=1e-9)


def test_spearman_constant_y_is_nan():
    x = [0.0, 1.0, 2.0]
    y = [3.0, 3.0, 3.0]
    assert math.isnan(_spearman(x, y))


def test_pearson_and_ols_recovers_known_line():
    # y = 2*x + 1 exactly -> r=1, slope=2, intercept=1.
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [1.0, 3.0, 5.0, 7.0, 9.0]
    r, slope, intercept = _pearson_and_ols(x, y)
    assert math.isclose(r, 1.0, abs_tol=1e-9)
    assert math.isclose(slope, 2.0, abs_tol=1e-9)
    assert math.isclose(intercept, 1.0, abs_tol=1e-9)


def test_pearson_constant_returns_nan():
    r, slope, intercept = _pearson_and_ols([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
    assert math.isnan(r) and math.isnan(slope)


def test_uint8_requant_coarsens_grid():
    # At s=0 it is (near) identity; at s>0 the distinct-value count drops.
    frame = torch.arange(0, 256, dtype=torch.float32).reshape(1, 256, 1).repeat(1, 1, 3)
    clean = _uint8_requant(frame, 0.0)
    coarse = _uint8_requant(frame, 0.9)
    n_clean = len(torch.unique(clean))
    n_coarse = len(torch.unique(coarse))
    assert n_coarse < n_clean, (n_coarse, n_clean)
    assert coarse.max() <= 255.0 and coarse.min() >= 0.0


def test_perturbation_strength_monotone_increases_l2_error():
    # Larger strength -> larger deviation from the clean frame, for every
    # perturbation. This is the property that makes the population a graded
    # fidelity sweep (so the correlation measurement is meaningful).
    torch.manual_seed(0)
    frame = torch.rand(32, 40, 3) * 255.0
    for pert in DEFAULT_PERTURBATIONS:
        e_low = (pert.fn(frame, 0.05) - frame).pow(2).mean().item()
        e_high = (pert.fn(frame, 0.45) - frame).pow(2).mean().item()
        assert e_high > e_low, (pert.name, e_low, e_high)


def test_is_strong_positive_uses_suprafloor_when_available():
    # Full-cloud rho dampened by floor noise, but suprafloor rho strong -> strong.
    r = CorrelationResult(
        surrogate_name="x",
        spearman_rho=0.55,
        spearman_rho_suprafloor=0.97,
        n_suprafloor=20,
        floor=5e-3,
        pearson_r=0.99,
        ols_slope=1.0,
        ols_intercept=0.0,
        n_points=100,
    )
    assert r.is_strong_positive()


def test_is_strong_positive_requires_positive_slope():
    r = CorrelationResult(
        surrogate_name="x",
        spearman_rho=0.99,
        spearman_rho_suprafloor=0.99,
        n_suprafloor=20,
        floor=5e-3,
        pearson_r=0.99,
        ols_slope=-1.0,  # negative slope: a lower surrogate does NOT lower d_seg
        ols_intercept=0.0,
        n_points=100,
    )
    assert not r.is_strong_positive()


def test_is_strong_positive_falls_back_to_full_when_few_suprafloor():
    r = CorrelationResult(
        surrogate_name="x",
        spearman_rho=0.95,
        spearman_rho_suprafloor=float("nan"),
        n_suprafloor=3,  # < 8 -> use full-cloud rho
        floor=5e-3,
        pearson_r=0.99,
        ols_slope=1.0,
        ols_intercept=0.0,
        n_points=100,
    )
    assert r.is_strong_positive()
