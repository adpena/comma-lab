"""Tests for ddm_pb3 — the parametric blind-set pose field.

The load-bearing claims are (a) the linearized floor is a genuine BOUND, (b) the
alignment metric returns exactly 1 on its own optimum, and (c) the direct maximizer
never reports less than the least-squares heuristic.  Each is tested against a
constructed case where the answer is known analytically, not against a golden value
this module produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.ddm_pb3_parametric_blind_field import (
    CONTEST_RATE_DENOMINATOR,
    alignment_efficiency,
    delta_s_rate,
    fit_basis_coefficients,
    fit_max_alignment,
    ground_inverse_depth,
    interaction_matrix,
    linearized_pose_floor,
    normalized_grid,
    payload_bytes,
    pose_contribution,
    pullback_to_blind,
    random_polynomial_saliency_fields,
    subset_index_bytes,
    vo_saliency_fields,
)

K = np.array([[910.0, 0.0, 582.0], [0.0, 910.0, 437.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------- arithmetic
def test_pose_contribution_matches_score_formula():
    # bp2's n600 base: mean d_pose 0.007642467 -> pose term 0.276450.
    assert pose_contribution(0.007642467374153057) == pytest.approx(0.27645, abs=1e-5)


def test_pose_contribution_rejects_negative():
    with pytest.raises(ValueError):
        pose_contribution(-1e-9)


def test_delta_s_rate_is_the_contest_rate_term():
    assert delta_s_rate(CONTEST_RATE_DENOMINATOR) == pytest.approx(25.0)
    assert delta_s_rate(0.0) == 0.0


def test_payload_bytes_and_index_endpoints():
    assert payload_bytes(600, 7, 8) == 4200.0
    assert payload_bytes(30, 7, 8, index_bytes=21.0) == pytest.approx(231.0)
    assert subset_index_bytes(0) == 0.0
    assert subset_index_bytes(600) == 0.0
    # log2 C(600,1) = log2 600 bits.
    assert subset_index_bytes(1) == pytest.approx(np.log2(600) / 8)
    with pytest.raises(ValueError):
        subset_index_bytes(601)


# ---------------------------------------------------------------------- floor
def test_floor_formula_matches_closed_form():
    d = np.array([1.0e-3])
    g1 = np.array([1.0e-3])  # gamma = 0.5
    assert linearized_pose_floor(d, g1)[0] == pytest.approx(1.0e-3 * 0.25)


def test_floor_clips_at_full_cancellation():
    # gamma >= 1 makes the bound vacuous (0), never negative.
    assert linearized_pose_floor(np.array([1e-3]), np.array([5e-3]))[0] == 0.0


def test_floor_capture_scales_gamma_linearly():
    d, g1 = np.array([1.0e-3]), np.array([1.0e-3])
    half = linearized_pose_floor(d, g1, capture=0.5)[0]
    assert half == pytest.approx(1.0e-3 * (1 - 0.25) ** 2)
    assert linearized_pose_floor(d, g1, capture=0.0)[0] == pytest.approx(1.0e-3)


def test_floor_rejects_negative_inputs():
    with pytest.raises(ValueError):
        linearized_pose_floor(np.array([-1.0]), np.array([1.0]))


def test_floor_is_a_bound_on_the_real_bp2_n600_receipt():
    """Regression on the arm's central claim, against the live receipt."""
    p = Path(__file__).resolve().parents[4] / "reports" / "ddm_bp2" / "reach_n600.jsonl"
    if not p.exists():  # pragma: no cover - receipt lives with the arm
        pytest.skip("bp2 receipt not present")
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    d0 = np.array([r["d_pose_base"] for r in rows])
    g1 = np.array([r["grad_blind_l1"] for r in rows])
    floor = linearized_pose_floor(d0, g1)
    keys = [f"d_pose_t{t}" for t in (0.002, 0.01, 0.05, 0.15, 0.35, 0.7, 1.0)]
    keys += ["d_pose_full_desc", "d_pose_full_asc", "d_pose_random_sign_same_k"]
    arms = np.stack([np.array([r[k] for r in rows]) for k in keys])
    assert int((arms < floor[None, :] - 1e-12).sum()) == 0
    # and the bound must be non-vacuous on a real fraction of pairs
    assert (g1 < 2.0 * d0).sum() > 100


# -------------------------------------------------------------------- geometry
def test_normalized_grid_zero_at_principal_point():
    x, y = normalized_grid(874, 1164, K)
    assert x[0, 582] == pytest.approx(0.0)
    assert y[437, 0] == pytest.approx(0.0)


def test_ground_inverse_depth_is_zero_above_horizon():
    y = np.array([[-0.5, 0.0, 0.25]])
    assert ground_inverse_depth(y).tolist() == [[0.0, 0.0, 0.25]]


def test_interaction_matrix_analytic_entries():
    x = np.array([[0.3]])
    y = np.array([[0.2]])
    z = np.array([[0.5]])
    lm = interaction_matrix(x, y, z)
    assert lm.shape == (2, 6, 1, 1)
    assert lm[0, 0, 0, 0] == pytest.approx(-0.5)  # -1/Z
    assert lm[0, 2, 0, 0] == pytest.approx(0.3 * 0.5)  # x/Z
    assert lm[0, 4, 0, 0] == pytest.approx(-(1 + 0.09))  # -(1+x^2)
    assert lm[1, 3, 0, 0] == pytest.approx(1 + 0.04)  # 1+y^2
    assert lm[1, 5, 0, 0] == pytest.approx(-0.3)  # -x


def test_interaction_matrix_rotation_columns_are_depth_free():
    x = np.array([[0.1]])
    y = np.array([[0.2]])
    a = interaction_matrix(x, y, np.array([[0.0]]))
    b = interaction_matrix(x, y, np.array([[9.0]]))
    assert np.allclose(a[:, 3:], b[:, 3:])
    assert not np.allclose(a[:, :3], b[:, :3])


def test_vo_saliency_fields_vanish_on_a_flat_image():
    flat = np.full((16, 20, 3), 128.0)
    fields = vo_saliency_fields(flat, K)
    assert fields.shape == (6, 16, 20, 3)
    assert np.allclose(fields, 0.0)


def test_vo_saliency_fields_reject_bad_shape():
    with pytest.raises(ValueError):
        vo_saliency_fields(np.zeros((4, 4)), K)


def test_random_polynomial_control_is_deterministic_and_distinct():
    img = np.random.default_rng(0).integers(0, 255, (12, 14, 3)).astype(float)
    a = random_polynomial_saliency_fields(img, K, np.random.default_rng(7))
    b = random_polynomial_saliency_fields(img, K, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert a.shape == (6, 12, 14, 3)
    assert not np.allclose(a, vo_saliency_fields(img, K))


# ------------------------------------------------------------------- the fit
def test_pullback_is_the_adjoint_of_the_forward_taps():
    from tac.optimization.ddm_bp2_blind_pose_actuator import apply_taps

    rng = np.random.default_rng(3)
    h, w, c = 5, 6, 2
    n = h * w
    idx = rng.integers(0, n, (4, n))
    wt = rng.random((4, n))
    field = rng.standard_normal((1, h, w, c))
    src = rng.standard_normal((h, w, c))
    blind = np.ones((h, w), dtype=bool)
    pulled = pullback_to_blind(idx, wt, 1.0, field, blind).ravel()
    lhs = float((apply_taps(idx, wt, src) * field[0]).sum())
    rhs = float(pulled @ src[blind].ravel())
    assert lhs == pytest.approx(rhs)


def test_fit_recovers_a_gradient_that_lies_in_the_span():
    rng = np.random.default_rng(11)
    basis = rng.standard_normal((6, 500))
    truth = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
    g = basis.T @ truth
    _, r2 = fit_basis_coefficients(g, basis)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_r2_is_near_zero_for_an_orthogonal_gradient():
    basis = np.zeros((2, 4))
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    g = np.array([0.0, 0.0, 1.0, 1.0])
    _, r2 = fit_basis_coefficients(g, basis)
    assert r2 == pytest.approx(0.0, abs=1e-12)


def test_fit_rejects_shape_mismatch_and_non_finite_basis():
    with pytest.raises(ValueError):
        fit_basis_coefficients(np.zeros(5), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        fit_basis_coefficients(np.zeros(4), np.full((2, 4), np.inf))


# ------------------------------------------------------------------ alignment
def test_alignment_oracle_is_exactly_one_on_its_own_optimum():
    """The control that caught a sign error on this module's first real run."""
    g = np.random.default_rng(5).standard_normal(1000)
    eta, delta = alignment_efficiency(g, g)
    assert eta == pytest.approx(1.0)
    assert np.array_equal(delta, -np.sign(g))


def test_alignment_is_minus_one_on_the_ascent_field():
    g = np.random.default_rng(6).standard_normal(500)
    eta, _ = alignment_efficiency(g, -g)
    assert eta == pytest.approx(-1.0)


def test_alignment_of_an_independent_field_is_near_zero():
    rng = np.random.default_rng(9)
    g = rng.standard_normal(200_000)
    eta, _ = alignment_efficiency(g, rng.standard_normal(200_000))
    assert abs(eta) < 0.01


def test_alignment_density_selects_the_largest_magnitudes():
    g = np.array([1.0, 1.0, 1.0, 1.0])
    phi = np.array([10.0, -0.1, 0.2, -9.0])
    _, delta = alignment_efficiency(g, phi, density=0.5)
    assert delta.tolist() == [-1.0, 0.0, 0.0, 1.0]


def test_alignment_rejects_bad_density_and_non_finite_phi():
    g = np.ones(4)
    with pytest.raises(ValueError):
        alignment_efficiency(g, g, density=0.0)
    with pytest.raises(ValueError):
        alignment_efficiency(g, np.array([np.nan, 1.0, 1.0, 1.0]))


def test_max_alignment_never_below_the_least_squares_heuristic():
    rng = np.random.default_rng(13)
    basis = rng.standard_normal((6, 40_000))
    g = basis.T @ rng.standard_normal(6) + 0.5 * rng.standard_normal(40_000)
    c_ls, _ = fit_basis_coefficients(g, basis)
    eta_ls, _ = alignment_efficiency(g, basis.T @ c_ls)
    _, eta_max = fit_max_alignment(g, basis, n_restarts=64, n_search=5000, seed=1)
    assert eta_max >= eta_ls - 1e-12


def test_max_alignment_recovers_unity_when_the_gradient_is_in_the_span():
    rng = np.random.default_rng(17)
    basis = rng.standard_normal((3, 20_000))
    g = basis.T @ np.array([2.0, -1.0, 0.5])
    _, eta = fit_max_alignment(g, basis, n_restarts=128, n_search=4000, seed=2)
    assert eta == pytest.approx(1.0, abs=1e-9)


def test_max_alignment_handles_a_zero_gradient():
    c, eta = fit_max_alignment(np.zeros(100), np.ones((2, 100)), n_restarts=4, n_search=50)
    assert eta == 0.0
    assert c.shape == (2,)
