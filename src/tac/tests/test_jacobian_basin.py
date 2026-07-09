"""Unit tests for the ξ→PoseNet Jacobian BASIN telemetry (OBSERVER).

Covers the build-contract correctness gates:
  * T1 ``jacobian_xi`` vs FINITE-DIFFERENCE parity (the Jacobian must be RIGHT before it informs anything).
  * ``conditioning`` SVD math (σ_min / cond / r_eff / rank), incl the degenerate (rank-deficient) case
    that IS the basin-empty regime, and the fail-safe on a non-finite Jacobian.
  * ``stratified_indices_by_motion`` includes the low + high |ego-t| tails (deterministic, no RNG).
  * T0 ``render_grad_energy`` is ~0 on a flat cartoon interior and rises with an edge (the σ_min proxy).
  * config validation.

Pure numpy + a tiny MLX function for the vjp parity — no trainer import, no GT cache, $0.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.witness_control import jacobian_basin as jb


# --------------------------------------------------------------------------- #
# T1 Jacobian correctness — vjp vs finite difference.
# --------------------------------------------------------------------------- #
def _mx_or_skip():
    try:
        import mlx.core as mx  # noqa: PLC0415
    except Exception:  # pragma: no cover - environment without MLX
        pytest.skip("mlx not available")
    return mx


def test_jacobian_xi_linear_exact():
    """For a LINEAR p(ξ)=Aξ the Jacobian is EXACTLY A (row i = A[i]); the vjp assembly must reproduce it."""
    mx = _mx_or_skip()
    rng = np.random.RandomState(0)
    A = rng.randn(6, 6).astype(np.float32)
    A_mx = mx.array(A)

    def p_of_xi(xi):
        return A_mx @ xi

    xi0 = rng.randn(6).astype(np.float32)
    J = jb.jacobian_xi(p_of_xi, xi0, mx=mx)
    assert J.shape == (6, 6)
    np.testing.assert_allclose(J, A.astype(np.float64), atol=1e-5)


def test_jacobian_xi_nonlinear_finite_diff_parity():
    """For a NONLINEAR p(ξ) the analytic vjp Jacobian must match the central finite-difference Jacobian.

    This is the build-contract T1 correctness gate: the σ_min the sensor reports is only trustworthy if
    the Jacobian it comes from is RIGHT."""
    mx = _mx_or_skip()
    rng = np.random.RandomState(1)
    A = mx.array(rng.randn(6, 6).astype(np.float32))
    b = mx.array(rng.randn(6).astype(np.float32))

    def p_of_xi(xi):
        # a genuinely nonlinear map: tanh(A@xi + b) + 0.3*xi^2 (elementwise), all MLX-differentiable.
        return mx.tanh(A @ xi + b) + 0.3 * (xi * xi)

    xi0 = (0.2 * rng.randn(6)).astype(np.float32)
    J = jb.jacobian_xi(p_of_xi, xi0, mx=mx)

    # central finite difference reference (numpy driving the MLX forward).
    def fwd(v):
        out = p_of_xi(mx.array(v.astype(np.float32)))
        mx.eval(out)
        return np.asarray(out, dtype=np.float64)

    eps = 1e-3
    J_fd = np.zeros((6, 6), dtype=np.float64)
    for j in range(6):
        d = np.zeros(6)
        d[j] = eps
        J_fd[:, j] = (fwd(xi0 + d) - fwd(xi0 - d)) / (2 * eps)
    # J[:, j] = ∂p/∂ξ_j ; jacobian_xi returns rows = ∂p_i/∂ξ, so J[i,j] must match J_fd[i,j].
    np.testing.assert_allclose(J, J_fd, atol=2e-3)


# --------------------------------------------------------------------------- #
# Conditioning SVD math.
# --------------------------------------------------------------------------- #
def test_conditioning_identity():
    c = jb.conditioning(np.eye(6))
    assert c["sigma_min"] == pytest.approx(1.0)
    assert c["sigma_max"] == pytest.approx(1.0)
    assert c["cond"] == pytest.approx(1.0)
    assert c["rank"] == 6
    assert c["r_eff"] == pytest.approx(6.0, rel=1e-6)
    assert not c["degenerate"]


def test_conditioning_rank_deficient_basin_empty():
    """A rank-deficient J_ξ (a zero singular value) => σ_min==0, cond==inf, rank<6 — the basin-empty
    regime (some ego-motion direction is INVISIBLE)."""
    d = np.diag([3.0, 2.0, 1.0, 0.5, 0.1, 0.0])
    c = jb.conditioning(d)
    assert c["sigma_min"] == pytest.approx(0.0)
    assert math.isinf(c["cond"])
    assert c["rank"] == 5
    assert c["r_eff"] < 6.0


def test_conditioning_non_finite_failsafe():
    j = np.full((6, 6), np.nan)
    c = jb.conditioning(j)
    assert c["degenerate"] is True
    assert c["sigma_min"] == 0.0
    assert c["rank"] == 0


# --------------------------------------------------------------------------- #
# Stratification + aggregation + the observer criterion.
# --------------------------------------------------------------------------- #
def test_stratified_includes_both_tails():
    motions = list(np.linspace(0.0, 10.0, 100))  # index i has motion ~ i/10
    idx = jb.stratified_indices_by_motion(motions, 8)
    assert len(idx) == 8
    # the lowest-motion (index 0) and highest-motion (index 99) pairs MUST be represented.
    assert 0 in idx
    assert 99 in idx


def test_stratified_k_ge_n_returns_all():
    assert jb.stratified_indices_by_motion([1.0, 2.0, 3.0], 10) == [0, 1, 2]


def test_aggregate_and_would_have_fired():
    conds = [{"sigma_min": s, "cond": 1.0 / max(s, 1e-9), "r_eff": 6.0} for s in (0.1, 0.2, 0.3, 0.4)]
    agg = jb.aggregate_conditioning(conds, sigma_floor=0.15)
    assert agg["median_sigma_min"] == pytest.approx(0.25)
    assert agg["p10_sigma_min"] == pytest.approx(0.13, abs=0.02)
    assert agg["basin_frac"] == pytest.approx(0.75)  # 3 of 4 >= 0.15
    # would_have_fired: None before a plateau estimate exists.
    assert jb.would_have_fired(agg, plateau_est=0.0, f_basin=1.0, quorum_q=0.8) is None
    # f_basin=1.0 vs plateau 0.25 => median 0.25 >= 0.25 True, but basin_frac 0.75 < 0.8 => False.
    assert jb.would_have_fired(agg, plateau_est=0.25, f_basin=1.0, quorum_q=0.8) is False
    # earlier engage (f_basin 0.5) + relaxed quorum => fires.
    assert jb.would_have_fired(agg, plateau_est=0.25, f_basin=0.5, quorum_q=0.7) is True


# --------------------------------------------------------------------------- #
# T0 render-boundary-gradient energy (the near-free σ_min proxy).
# --------------------------------------------------------------------------- #
def test_t0_flat_interior_near_zero_edge_rises():
    flat = np.full((16, 16, 3), 100.0)
    assert jb.render_grad_energy(flat) == pytest.approx(0.0)
    edged = flat.copy()
    edged[:, 8:, :] = 200.0  # a vertical edge at col 8
    assert jb.render_grad_energy(edged) > 0.0
    # per-class: class 1 sits on the edge => higher energy than the flat class 0.
    argmax = np.zeros((16, 16), dtype=np.int64)
    argmax[:, 7:9] = 1
    pc = jb.render_grad_energy_per_class(edged, argmax, n_classes=2)
    assert pc[1] > pc[0]


def test_t0_fields_shape():
    frames = [np.full((8, 8, 3), 50.0), np.random.RandomState(0).rand(8, 8, 3) * 255.0]
    out = jb.t0_fields(frames)
    assert out["n_frames"] == 2
    assert "render_grad_energy" in out and out["render_grad_energy"] >= 0.0


def test_config_validation():
    assert jb.JacobianBasinConfig().validate() == []
    bad = jb.JacobianBasinConfig(k_pairs=0, every=0, f_basin=2.0, quorum_q=-1.0, sigma_floor=-1.0)
    problems = bad.validate()
    assert len(problems) == 5
