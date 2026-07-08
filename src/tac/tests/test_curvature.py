"""Tests for the HVP-Lanczos curvature spectrum (task #312 Phase B).

Correctness is proven on KNOWN matrices (the matvec is A@v): Lanczos must recover the true
extreme eigenvalues, Hutchinson must approach the true trace, and the MLX HVP adapter must
match a hand-differentiated quadratic Hessian.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control import curvature as cv


def _sym_matvec(A):
    A = np.asarray(A, dtype=np.float64)
    return lambda v: A @ np.asarray(v, dtype=np.float64)


def test_lanczos_recovers_known_eigenvalues_diagonal():
    eig = np.array([10.0, 6.0, 3.0, 1.0, 0.5, -2.0])
    A = np.diag(eig)
    sp = cv.compute_spectrum(_sym_matvec(A), dim=6, k=3, n_iter=6, seed=0)
    # top-3 largest algebraic
    assert sp.lambda_max == pytest.approx(10.0, abs=1e-6)
    assert sp.top_k[0] == pytest.approx(10.0, abs=1e-6)
    assert sp.top_k[1] == pytest.approx(6.0, abs=1e-6)
    assert sp.top_k[2] == pytest.approx(3.0, abs=1e-6)
    assert sp.negative_curvature is True  # -2 eigenvalue present


def test_lanczos_dense_random_symmetric_matches_numpy_eigh():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((20, 20))
    A = (M + M.T) / 2.0
    true = np.sort(np.linalg.eigvalsh(A))[::-1]
    sp = cv.compute_spectrum(_sym_matvec(A), dim=20, k=4, n_iter=20, seed=1)
    assert sp.lambda_max == pytest.approx(true[0], abs=1e-5)
    for got, want in zip(sp.top_k, true[:4]):
        assert got == pytest.approx(want, abs=1e-4)


def test_anisotropy_and_sharpness_definitions():
    A = np.diag([8.0, 4.0, 2.0, 1.0])
    sp = cv.compute_spectrum(_sym_matvec(A), dim=4, k=3, n_iter=4, seed=0)
    # anisotropy = λ1/λk over top-3 = 8/2 = 4
    assert sp.anisotropy == pytest.approx(4.0, abs=1e-4)
    # sharpness = λ_max / |mean(top_k)| = 8 / mean(8,4,2)=8/4.667
    assert sp.sharpness_ratio == pytest.approx(8.0 / ((8 + 4 + 2) / 3), abs=1e-3)


def test_hutchinson_trace_approaches_true_trace():
    rng = np.random.default_rng(5)
    M = rng.standard_normal((30, 30))
    A = (M + M.T) / 2.0
    true_tr = float(np.trace(A))
    est, n_mv = cv.hutchinson_trace(_sym_matvec(A), dim=30, n_probes=200, seed=2)
    assert n_mv == 200
    assert est == pytest.approx(true_tr, rel=0.15)


def test_ritz_values_sorted_ascending():
    alpha = np.array([2.0, 3.0, 1.0])
    beta = np.array([0.5, 0.5])
    rv = cv.ritz_values(alpha, beta)
    assert list(rv) == sorted(rv)
    assert rv.size == 3


def test_lanczos_early_breakdown_invariant_subspace():
    # rank-1 operator: A = e0 e0^T scaled -> Lanczos breaks down after 1-2 steps
    A = np.zeros((5, 5))
    A[0, 0] = 7.0
    alpha, beta, steps = cv.lanczos_tridiag(_sym_matvec(A), dim=5, n_iter=5, seed=0)
    assert steps >= 1
    rv = cv.ritz_values(alpha, beta)
    assert float(rv[-1]) == pytest.approx(7.0, abs=1e-6) or 7.0 in np.round(rv, 6)


def test_spectrum_row_schema():
    A = np.diag([5.0, 2.0, 1.0])
    sp = cv.compute_spectrum(_sym_matvec(A), dim=3, k=2, n_iter=3, seed=0)
    row = sp.to_row(stage="tau_softplus", ep=650, k_pairs=16, source="mod32cap")
    assert row["stage"] == "curvature_spectrum"
    assert row["seg_stage"] == "tau_softplus"
    assert row["ep"] == 650
    assert len(row["top_k_eigs"]) == 2
    assert row["score_neutral"] is True
    assert "NON-PROMOTABLE" in row["axis"]
    assert row["source"] == "mod32cap"


def test_compute_spectrum_k_clamped_to_dim():
    A = np.diag([3.0, 1.0])
    sp = cv.compute_spectrum(_sym_matvec(A), dim=2, k=8, n_iter=8, seed=0)
    assert len(sp.top_k) == 2  # clamped to dim


def test_mlx_hvp_matches_quadratic_hessian():
    """The MLX HVP adapter on a known quadratic loss(x) = 0.5 x^T A x must give matvec(v)=A v."""
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(7)
    M = rng.standard_normal((6, 6)).astype(np.float32)
    A = (M + M.T) / 2.0
    Amx = mx.array(A)

    def loss_of_vector(x):  # 0.5 x^T A x  -> grad = A x -> Hessian = A
        return 0.5 * mx.sum(x * (Amx @ x))

    x0 = rng.standard_normal(6).astype(np.float32)
    matvec = cv.make_mlx_hvp(loss_of_vector, x0)
    v = rng.standard_normal(6)
    got = matvec(v)
    want = A.astype(np.float64) @ v
    assert np.allclose(got, want, atol=1e-4)


def test_mlx_hvp_spectrum_matches_quadratic_eigs():
    mx = pytest.importorskip("mlx.core")
    eig = np.array([9.0, 5.0, 2.0, 0.5], dtype=np.float32)
    A = np.diag(eig).astype(np.float32)
    Amx = mx.array(A)

    def loss_of_vector(x):
        return 0.5 * mx.sum(x * (Amx @ x))

    x0 = np.zeros(4, dtype=np.float32)
    matvec = cv.make_mlx_hvp(loss_of_vector, x0)
    sp = cv.compute_spectrum(matvec, dim=4, k=3, n_iter=4, seed=0)
    assert sp.lambda_max == pytest.approx(9.0, abs=1e-3)
    assert sp.top_k[1] == pytest.approx(5.0, abs=1e-3)


def test_mlx_model_hvp_on_tiny_module_matches_known_hessian_and_restores():
    """The model-param HVP (used by the in-trainer hook + standalone tool) on a tiny MLX module
    with a quadratic-in-weights loss must recover the true Hessian eigenvalues AND leave the
    module's parameters unchanged (score-neutral)."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            # single 1x1 linear (weight w, bias b): loss = 0.5*(a*w^2 + c*b^2) -> Hessian diag(a,c)
            self.lin = nn.Linear(1, 1)

        def __call__(self):
            w = self.lin.weight.reshape(())
            b = self.lin.bias.reshape(())
            return 0.5 * (9.0 * w * w + 4.0 * b * b)

    model = Lin()
    w_before = np.asarray(model.lin.weight).copy()
    matvec, dim = cv.mlx_model_hvp(lambda m: m(), model)
    assert dim == 2  # weight + bias
    sp = cv.compute_spectrum(matvec, dim=dim, k=2, n_iter=2, seed=0)
    got = sorted(sp.top_k, reverse=True)
    assert got[0] == pytest.approx(9.0, abs=1e-3)
    assert got[1] == pytest.approx(4.0, abs=1e-3)
    # module restored after the measurement
    assert np.allclose(np.asarray(model.lin.weight), w_before)
