# SPDX-License-Identifier: MIT
"""Tests: closed-form categorical-Fisher pseudo-inverse law (arm A, SPEC_v10 §13.4(2))."""
from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.fisher_actuation_arm_a_20260717 import (
    build_categorical_fisher_pseudoinverse_cotangent_precondition_v1,
    categorical_fisher_pseudoinverse_precondition_law as law,
)


def _rand_p(seed=0, n=6, k=5):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, k))
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _rand_v(seed=1, n=6, k=5):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1.0, (n, k))
    return v - v.mean(axis=-1, keepdims=True)


def test_pseudoinverse_identity_exact_at_eps_zero():
    p, v = _rand_p(), _rand_v()
    u = law(p, v, eps=0.0)
    for i in range(p.shape[0]):
        g = np.diag(p[i]) - np.outer(p[i], p[i])
        np.testing.assert_allclose(g @ u[i], v[i], rtol=1e-10, atol=1e-12)


def test_output_is_zero_sum_min_norm_branch():
    u = law(_rand_p(2), _rand_v(3), eps=0.0)
    np.testing.assert_allclose(u.sum(axis=-1), 0.0, atol=1e-10)


def test_gauge_component_projected_out():
    p = _rand_p(4)
    v = _rand_v(5)
    u0 = law(p, v, eps=1e-3)
    u1 = law(p, v + 7.3, eps=1e-3)  # add a pure-gauge (constant) component
    np.testing.assert_allclose(u0, u1, rtol=1e-12, atol=1e-12)


def test_matches_sister_quotient_solver_at_zero_damping():
    from tac.information_geometry.fisher_natural_solver import (
        solve_categorical_fisher_natural_step_numpy_fp32,
    )
    p = _rand_p(7, n=4).astype(np.float32)
    v = _rand_v(8, n=4).astype(np.float32)
    res = solve_categorical_fisher_natural_step_numpy_fp32(
        p, v, delta=1e12, damping=0.0)  # huge trust ball => raw quotient solve
    got = law(p, v, eps=0.0)
    # solver returns the DESCENT step -g+ g; the law returns g+ v.
    np.testing.assert_allclose(got, -np.asarray(res.step, np.float64), rtol=2e-3, atol=2e-3)


def test_damping_bounds_corner_blowup():
    p = np.array([[1.0 - 4e-9, 1e-9, 1e-9, 1e-9, 1e-9]])
    v = _rand_v(9, n=1)
    u = law(p, v, eps=1e-2)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) <= (np.max(np.abs(v - v.mean())) * 2) / 1e-2 + 1e-9


def test_fail_closed_shape_mismatch_and_bad_eps():
    with pytest.raises(ValueError, match="shape"):
        law(_rand_p(), _rand_v(n=3), eps=1e-3)
    with pytest.raises(ValueError, match="eps"):
        law(_rand_p(), _rand_v(), eps=-1.0)


def test_fail_closed_zero_probability_at_exact_mode():
    p = _rand_p().copy()
    p[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        law(p, _rand_v(), eps=0.0)


def test_build_equation_validates_and_carries_non_promotable_domain():
    eq = build_categorical_fisher_pseudoinverse_cotangent_precondition_v1()
    assert eq.equation_id == "categorical_fisher_pseudoinverse_cotangent_precondition_v1"
    dom = eq.domain_of_validity
    assert dom["research_only"] is True
    assert dom["score_claim"] is False
    assert "OWED" in dom["measurement_status"]
    assert eq.canonical_consumers and eq.canonical_producers


def test_law_is_linear_in_cotangent():
    p = _rand_p(11)
    v1, v2 = _rand_v(12), _rand_v(13)
    u = law(p, 2.0 * v1 + 3.0 * v2, eps=1e-3)
    np.testing.assert_allclose(
        u, 2.0 * law(p, v1, eps=1e-3) + 3.0 * law(p, v2, eps=1e-3), rtol=1e-10)


def test_ce_natural_gradient_special_case():
    # v = p - y (CE logit gradient): at eps=0, g+ (p-y) = 1 - y/p - const; check via identity.
    p = _rand_p(14, n=3)
    y = np.eye(5)[[0, 2, 4]]
    v = p - y
    u = law(p, v, eps=0.0)
    for i in range(3):
        g = np.diag(p[i]) - np.outer(p[i], p[i])
        np.testing.assert_allclose(g @ u[i], v[i] - v[i].mean() * 0, rtol=1e-9, atol=1e-10)
