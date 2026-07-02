# SPDX-License-Identifier: MIT
"""Tests for the margin-field HEAD levers (#218 facets 1 & 3)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.laguerre_logit_offset import (
    LaguerreLogitOffsetError,
    additive_margin_logits,
    apply_offset_to_sdf_bias,
    etf_gram_offdiag,
    laguerre_offset_sweep,
    menon_logit_adjustment_offsets,
    per_class_disagreement,
    power_diagram_argmax,
    simplex_etf,
)


# ---- power_diagram_argmax --------------------------------------------------
def test_power_diagram_zero_offset_is_plain_argmax():
    phi = np.random.default_rng(0).standard_normal((7, 11, 5))
    assert np.array_equal(power_diagram_argmax(phi, np.zeros(5)), np.argmax(phi, axis=-1))


def test_power_diagram_offset_flips_toward_boosted_class():
    phi = np.array([[1.0, 0.9, 0.0, 0.0, 0.0]])  # class 0 wins by 0.1 over class 1
    assert power_diagram_argmax(phi, np.zeros(5))[0] == 0
    b = np.array([0.0, 0.2, 0.0, 0.0, 0.0])  # boost class 1 by 0.2 > 0.1 margin
    assert power_diagram_argmax(phi, b)[0] == 1


def test_power_diagram_offset_K_mismatch_raises():
    with pytest.raises(LaguerreLogitOffsetError):
        power_diagram_argmax(np.zeros((3, 5)), np.zeros(4))


# ---- apply_offset_to_sdf_bias ---------------------------------------------
def test_apply_offset_shifts_bias_only_and_is_copy():
    params = {"out_sdf.bias": np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32),
              "out_sdf.weight": np.ones((5, 3), np.float32)}
    b = np.array([0.1, -0.2, 0.0, 0.3, 0.0])
    out = apply_offset_to_sdf_bias(params, b)
    assert np.allclose(out["out_sdf.bias"], params["out_sdf.bias"] + b)
    # input not mutated
    assert np.allclose(params["out_sdf.bias"], [1, 2, 3, 4, 5])
    # weight passed through unchanged (same object ok — shallow copy)
    assert out["out_sdf.weight"] is params["out_sdf.weight"]


def test_apply_offset_bias_shift_equals_phi_shift():
    # phi = h @ W.T + bias ; shifting bias by b shifts phi by b exactly.
    rng = np.random.default_rng(3)
    W = rng.standard_normal((5, 8)); bias = rng.standard_normal(5); h = rng.standard_normal((20, 8))
    phi = h @ W.T + bias
    b = np.array([0.5, -0.3, 0.1, 0.0, 0.2])
    phi_shift = h @ W.T + (bias + b)
    assert np.allclose(phi_shift, phi + b)


def test_apply_offset_missing_key_raises():
    with pytest.raises(LaguerreLogitOffsetError):
        apply_offset_to_sdf_bias({"x": np.zeros(5)}, np.zeros(5))


def test_apply_offset_K_mismatch_raises():
    with pytest.raises(LaguerreLogitOffsetError):
        apply_offset_to_sdf_bias({"out_sdf.bias": np.zeros(5)}, np.zeros(4))


# ---- Menon logit adjustment -----------------------------------------------
def test_menon_rare_class_gets_larger_offset():
    priors = np.array([0.5, 0.006, 0.4, 0.016, 0.078])  # Lane (idx1) rarest
    b = menon_logit_adjustment_offsets(priors)
    assert np.argmax(b) == 1  # rarest class -> largest positive offset
    assert b[0] < b[3] < b[1]  # Road < Movable < Lane monotone in rarity


def test_menon_zero_sum_and_uniform_prior():
    b = menon_logit_adjustment_offsets(np.ones(5))
    assert np.allclose(b, 0.0)  # uniform prior -> no adjustment
    b2 = menon_logit_adjustment_offsets(np.array([0.5, 0.006, 0.4, 0.016, 0.078]))
    assert abs(float(b2.sum())) < 1e-9  # zero-sum (argmax-invariant to global const)


def test_menon_tau_scales_linearly():
    priors = np.array([0.5, 0.006, 0.4, 0.016, 0.078])
    assert np.allclose(menon_logit_adjustment_offsets(priors, tau=2.0),
                       2.0 * menon_logit_adjustment_offsets(priors, tau=1.0))


def test_menon_rejects_bad_priors():
    with pytest.raises(LaguerreLogitOffsetError):
        menon_logit_adjustment_offsets(np.array([1.0]))
    with pytest.raises(LaguerreLogitOffsetError):
        menon_logit_adjustment_offsets(np.array([-1.0, 2.0]))


# ---- simplex ETF ----------------------------------------------------------
def test_simplex_etf_shape_and_equiangular():
    K, d = 5, 96
    W = simplex_etf(K, d)
    assert W.shape == (K, d)
    norms = np.linalg.norm(W, axis=1)
    assert np.allclose(norms, norms[0])  # equal-norm rows
    assert abs(etf_gram_offdiag(W) - (-1.0 / (K - 1))) < 1e-9  # cos == -1/(K-1)


def test_simplex_etf_deterministic():
    assert np.allclose(simplex_etf(5, 96), simplex_etf(5, 96))  # regenerable (FREE at inflate)


def test_simplex_etf_scale():
    W = simplex_etf(5, 96, scale=2.5)
    assert np.allclose(np.linalg.norm(W, axis=1), 2.5)


def test_simplex_etf_requires_dim_ge_k():
    with pytest.raises(LaguerreLogitOffsetError):
        simplex_etf(5, 4)


# ---- additive margin -------------------------------------------------------
def test_additive_margin_subtracts_target_only():
    logits = np.array([[2.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0]])
    target = np.array([0, 2])
    out = additive_margin_logits(logits, target, margin=0.5)
    assert out[0, 0] == 1.5 and out[1, 2] == 2.5  # target reduced
    # non-target unchanged
    assert out[0, 1] == 1.0 and out[1, 0] == 0.0


def test_additive_margin_bad_target_shape_raises():
    with pytest.raises(LaguerreLogitOffsetError):
        additive_margin_logits(np.zeros((4, 5)), np.zeros(3, int), 0.1)


def test_additive_margin_out_of_range_raises():
    with pytest.raises(LaguerreLogitOffsetError):
        additive_margin_logits(np.zeros((2, 5)), np.array([0, 5]), 0.1)


# ---- per_class_disagreement -----------------------------------------------
def test_per_class_disagreement_counts():
    gt = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 0, 2])  # class0: 1/2 wrong, class1: 1/2 wrong, class2: 0/1
    d = per_class_disagreement(pred, gt, 3)
    assert d[0] == 0.5 and d[1] == 0.5 and d[2] == 0.0


# ---- laguerre_offset_sweep (the fire-first engine) ------------------------
def test_sweep_baseline_matches_zero_offset():
    rng = np.random.default_rng(1)
    phi = rng.standard_normal((50, 50, 5))
    gt = np.argmax(phi, axis=-1)  # witness == gt -> d_seg 0 at b=0
    res = laguerre_offset_sweep(phi, gt, focus_classes=(1, 3), offset_grid=(0.0, 0.2, 0.5))
    assert res.baseline_d_seg == 0.0
    assert res.best_d_seg == 0.0
    assert res.delta == 0.0


def test_sweep_recovers_systematically_erased_minority_class():
    # Construct a field where class 1 is systematically under-predicted: pixels whose
    # TRUE (gt) class is 1 have phi_1 just barely below phi_0 (Lane erased to Road).
    rng = np.random.default_rng(2)
    N = 4000
    phi = rng.standard_normal((N, 5)) * 0.1
    gt = np.zeros(N, dtype=np.int64)
    # half are true-Road (class 0), half true-Lane (class 1) but erased.
    lane = np.arange(N) % 2 == 0
    gt[lane] = 1
    # true-Lane pixels: phi_0 slightly beats phi_1 by ~0.15 (erasure) -> argmax says Road
    phi[lane, 0] = 1.0
    phi[lane, 1] = 0.85
    # true-Road pixels: phi_0 clearly wins
    phi[~lane, 0] = 1.0
    phi[~lane, 1] = 0.2
    res = laguerre_offset_sweep(phi, gt, focus_classes=(1,), offset_grid=(0.0, 0.1, 0.2, 0.3, 0.5))
    # baseline: all lane pixels wrong -> d_seg ~0.5
    assert res.baseline_d_seg > 0.4
    # a positive Lane offset in (0.15, 0.8) recovers lane WITHOUT breaking road
    assert res.best_offsets[1] > 0.1
    assert res.best_d_seg < 0.05
    assert res.delta < -0.4
    # per-class: Lane disagreement collapses
    assert res.per_class_disagree_best[1] < 0.1


def test_sweep_focus_classes_only_moves_those():
    rng = np.random.default_rng(4)
    phi = rng.standard_normal((30, 30, 5))
    gt = np.argmax(phi, axis=-1)
    res = laguerre_offset_sweep(phi, gt, focus_classes=(2,), offset_grid=(0.0, 0.3))
    # non-focus classes stay 0
    assert res.best_offsets[0] == 0.0 and res.best_offsets[1] == 0.0
    assert res.best_offsets[3] == 0.0 and res.best_offsets[4] == 0.0


def test_sweep_table_covers_full_grid():
    rng = np.random.default_rng(5)
    phi = rng.standard_normal((10, 10, 5))
    gt = np.argmax(phi, axis=-1)
    grid = (0.0, 0.2, 0.5)
    res = laguerre_offset_sweep(phi, gt, focus_classes=(1, 3), offset_grid=grid)
    assert len(res.table) == len(grid) ** 2  # cartesian product


def test_sweep_base_offsets_compose():
    rng = np.random.default_rng(6)
    phi = rng.standard_normal((20, 20, 5))
    gt = np.argmax(phi, axis=-1)
    base = np.array([0.0, 0.05, 0.0, 0.0, 0.0])
    res = laguerre_offset_sweep(phi, gt, focus_classes=(1,), offset_grid=(0.0, 0.1),
                                base_offsets=base)
    # best offset for class 1 = base + grid value (>= base)
    assert res.best_offsets[1] >= 0.05
