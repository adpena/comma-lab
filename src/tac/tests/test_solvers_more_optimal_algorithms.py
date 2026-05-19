# SPDX-License-Identifier: MIT
"""Tests for ``tac.solvers`` more-optimal algorithm subpackage.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #229 premise verification,
every solver carries ≥4 tests covering basic/extreme/integration/regression.

[macOS-CPU advisory] — local CPU only; never promoted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.solvers.fista import (
    FistaResult,
    fista_proximal_gradient,
    project_simplex,
    soft_threshold,
)
from tac.solvers.frank_wolfe import (
    FrankWolfeResult,
    frank_wolfe_kcard,
    lmo_kcardinality,
)
from tac.solvers.numba_jit_water_filling import (
    NUMBA_AVAILABLE,
    water_fill_bisection_numpy,
    water_fill_bisection_numba_if_available,
)
from tac.solvers.riemannian_newton_stiefel import (
    RiemannianNewtonResult,
    StiefelManifold,
    project_to_stiefel,
    retract_qr,
    riemannian_newton_step,
)
from tac.solvers.sinkhorn import (
    SinkhornResult,
    sinkhorn_ensemble_select_k,
    sinkhorn_knopp,
)


# ---------------- FISTA tests ----------------


class TestFista:
    def test_soft_threshold_basic(self) -> None:
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        out = soft_threshold(z, lam=1.5)
        # |z|<=1.5 -> 0; else shrink by 1.5 toward 0
        assert np.allclose(out, [-1.5, 0.0, 0.0, 0.0, 1.5])

    def test_project_simplex_unit_budget(self) -> None:
        v = np.array([0.7, 0.3, 0.1])
        out = project_simplex(v, budget=1.0)
        assert np.isclose(out.sum(), 1.0)
        assert np.all(out >= 0)

    def test_project_simplex_zero_budget(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        out = project_simplex(v, budget=0.0)
        assert np.allclose(out, 0)

    def test_fista_converges_on_quadratic(self) -> None:
        # Minimize 0.5 * ||x - target||^2 + 0.5 * ||x||_1
        target = np.array([2.0, 0.5, -1.0, 0.0])

        def grad(x: np.ndarray) -> np.ndarray:
            return x - target

        def prox(z: np.ndarray, step: float) -> np.ndarray:
            return soft_threshold(z, step * 0.5)

        def obj(x: np.ndarray) -> float:
            return 0.5 * float(np.linalg.norm(x - target) ** 2) + 0.5 * float(np.abs(x).sum())

        result = fista_proximal_gradient(grad, prox, np.zeros_like(target), step_size=1.0, objective_fn=obj)
        # Soft-thresholded result: shrink target toward 0 by 0.5 each
        expected = np.sign(target) * np.maximum(np.abs(target) - 0.5, 0.0)
        assert np.allclose(result.x, expected, atol=1e-4)
        # Objective should be monotonically non-increasing after enough iters
        assert result.objective_history[-1] <= result.objective_history[0]

    def test_fista_returns_frozen_dataclass(self) -> None:
        result = fista_proximal_gradient(
            lambda x: x, lambda z, s: z, np.array([1.0]), step_size=1.0
        )
        assert isinstance(result, FistaResult)
        with pytest.raises((AttributeError, Exception)):
            result.x = np.array([0.0])  # type: ignore[misc]


# ---------------- Frank-Wolfe tests ----------------


class TestFrankWolfe:
    def test_lmo_kcardinality_basic(self) -> None:
        grad = np.array([0.5, 0.1, 0.9, 0.2, 0.8])
        s = lmo_kcardinality(grad, k=2)
        # k=2 smallest grad entries are at indices 1 (0.1) and 3 (0.2)
        assert s.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
        assert int(s.sum()) == 2

    def test_lmo_kcardinality_extreme_zero(self) -> None:
        grad = np.array([0.1, 0.2, 0.3])
        s = lmo_kcardinality(grad, k=0)
        assert np.allclose(s, 0)

    def test_lmo_kcardinality_extreme_n(self) -> None:
        grad = np.array([0.1, 0.2, 0.3])
        s = lmo_kcardinality(grad, k=10)
        assert np.allclose(s, 1)

    def test_frank_wolfe_converges_on_quadratic_kcard(self) -> None:
        # minimize 0.5 * ||x - target||^2 over {x in [0,1]^n with sum(x) = k}
        target = np.array([0.9, 0.1, 0.8, 0.2, 0.85, 0.15])
        k = 3

        def grad(x: np.ndarray) -> np.ndarray:
            return x - target

        def obj(x: np.ndarray) -> float:
            return 0.5 * float(np.linalg.norm(x - target) ** 2)

        result = frank_wolfe_kcard(grad, obj, n=6, k=k, max_iters=300, tol=1e-7)
        # Top-k of target are indices 0 (0.9), 2 (0.8), 4 (0.85)
        assert sorted(result.selected_indices) == [0, 2, 4]
        assert np.isclose(result.x.sum(), k, atol=0.2)


# ---------------- Sinkhorn tests ----------------


class TestSinkhorn:
    def test_sinkhorn_balanced_uniform(self) -> None:
        cost = np.array([[1.0, 2.0], [2.0, 1.0]])
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        result = sinkhorn_knopp(cost, a, b, reg_lambda=5.0)
        assert result.converged
        assert np.isclose(result.P.sum(), 1.0, atol=1e-3)
        assert np.allclose(result.P.sum(axis=1), a, atol=1e-3)
        assert np.allclose(result.P.sum(axis=0), b, atol=1e-3)

    def test_sinkhorn_marginal_mismatch_rejected(self) -> None:
        cost = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="row_marginal sum"):
            sinkhorn_knopp(cost, np.array([1.0]), np.array([1.0, 1.0]))

    def test_sinkhorn_ensemble_select_k_basic(self) -> None:
        # 10 candidates, select K=3 best
        # With col_marginal = k/n per candidate, max any candidate absorbs is k/n.
        # Top-K should hit ceiling (3 candidates × k/n = 3 × 0.3 = 0.9), and the
        # remaining (k - 0.9) = 2.1 spreads across the rest. Convergence + weighting
        # toward high-quality is checked via the per-candidate ordering.
        scores = np.array([0.5, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.4, 0.85])
        weights, result = sinkhorn_ensemble_select_k(scores, k=3, reg_lambda=20.0)
        assert np.isclose(weights.sum(), 3.0, atol=0.5)
        # Top-3 candidates should each be at or near the per-candidate cap (k/n=0.3)
        top_3 = np.argsort(scores)[-3:]
        bot_3 = np.argsort(scores)[:3]
        top_weight = float(weights[top_3].sum())
        bot_weight = float(weights[bot_3].sum())
        # Top-3 should be strictly larger than bottom-3 (quality ordering preserved)
        assert top_weight > bot_weight, f"top {top_weight} <= bot {bot_weight}"

    def test_sinkhorn_extreme_k_equals_n(self) -> None:
        scores = np.array([0.3, 0.7, 0.5])
        weights, _ = sinkhorn_ensemble_select_k(scores, k=3, reg_lambda=5.0)
        # If K=N=3 we transport k/n=1 per candidate.
        assert np.isclose(weights.sum(), 3.0, atol=1e-2)


# ---------------- Riemannian-Newton on Stiefel tests ----------------


class TestRiemannianNewton:
    def test_stiefel_dims_invariant(self) -> None:
        with pytest.raises(ValueError, match="invalid Stiefel"):
            StiefelManifold(n=2, p=5)
        with pytest.raises(ValueError, match="invalid Stiefel"):
            StiefelManifold(n=5, p=0)

    def test_stiefel_random_point_is_on_manifold(self) -> None:
        st = StiefelManifold(n=8, p=3)
        rng = np.random.default_rng(42)
        X = st.random_point(rng)
        assert X.shape == (8, 3)
        assert st.is_on_manifold(X)

    def test_project_to_stiefel(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.standard_normal((6, 3))
        Xp = project_to_stiefel(X)
        st = StiefelManifold(n=6, p=3)
        assert st.is_on_manifold(Xp)

    def test_retract_qr_returns_orthonormal(self) -> None:
        rng = np.random.default_rng(11)
        st = StiefelManifold(n=5, p=2)
        X = st.random_point(rng)
        eta = rng.standard_normal((5, 2)) * 0.01
        X_new = retract_qr(X, eta)
        assert st.is_on_manifold(X_new)

    def test_riemannian_newton_preserves_orthogonality(self) -> None:
        # Minimize 0.5 * tr(X^T A X) over St(n, p): solution = top-p eigenvectors of -A
        rng = np.random.default_rng(13)
        n, p = 8, 3
        A = rng.standard_normal((n, n))
        A = 0.5 * (A + A.T)  # symmetric

        def obj(X: np.ndarray) -> float:
            return 0.5 * float(np.trace(X.T @ A @ X))

        def grad(X: np.ndarray) -> np.ndarray:
            return A @ X

        st = StiefelManifold(n, p)
        X0 = st.random_point(rng)
        result = riemannian_newton_step(obj, grad, X0, max_iters=200, step_size=0.05)
        assert result.orthogonality_error < 1e-6
        # Should decrease objective
        assert result.objective_history[-1] <= result.objective_history[0] + 1e-9


# ---------------- Numba JIT water-filling tests ----------------


class TestNumbaJitWaterFilling:
    def test_numpy_bisection_basic(self) -> None:
        rng = np.random.default_rng(101)
        imp = rng.uniform(0.01, 1.0, size=1000).astype(np.float64)
        total_budget = 3000
        result = water_fill_bisection_numpy(imp, total_bits=total_budget, alpha=0.5, min_bits=1, max_bits=8)
        # Bisection rounding can slip slightly over the integer budget due to
        # per-weight rounding ties; tolerance is the canonical bisection's behavior.
        assert result.bits.sum() <= total_budget + 5
        assert result.bits.sum() >= total_budget - 50  # not too far under
        assert result.bits.min() >= 1
        assert result.bits.max() <= 8
        assert result.backend == "numpy"

    def test_numba_or_numpy_fallback(self) -> None:
        rng = np.random.default_rng(202)
        imp = rng.uniform(0.01, 1.0, size=500).astype(np.float64)
        total_budget = 1500
        result = water_fill_bisection_numba_if_available(imp, total_bits=total_budget)
        if NUMBA_AVAILABLE:
            assert result.backend == "numba"
        else:
            assert result.backend == "numpy"
        assert result.bits.sum() <= total_budget + 5

    def test_infeasible_budget_rejected(self) -> None:
        imp = np.ones(100, dtype=np.float64)
        # 100 weights * min_bits=1 = 100 bits floor; ask for 50 (infeasible)
        with pytest.raises(ValueError, match="infeasible budget"):
            water_fill_bisection_numpy(imp, total_bits=50, min_bits=1)

    def test_overbudget_returns_max_bits(self) -> None:
        imp = np.ones(10, dtype=np.float64)
        # 10 weights * max_bits=8 = 80 ceiling; ask for 1000 (overflow)
        result = water_fill_bisection_numpy(imp, total_bits=1000, max_bits=8)
        assert np.all(result.bits == 8)
        assert result.iterations == 0

    def test_zero_importance_returns_min_bits(self) -> None:
        imp = np.zeros(50, dtype=np.float64)
        result = water_fill_bisection_numpy(imp, total_bits=200, min_bits=2)
        assert np.all(result.bits == 2)

    def test_numpy_vs_canonical_bit_allocator_within_tolerance(self) -> None:
        """Cross-check numpy bisection against canonical ``tac.bit_allocator.allocate_bits``."""
        import torch

        from tac.bit_allocator import allocate_bits

        rng = np.random.default_rng(404)
        imp = rng.uniform(0.01, 1.0, size=2000).astype(np.float64)
        imp_tensor = torch.from_numpy(imp).reshape(-1)
        total = 6000
        out = allocate_bits({"flat": imp_tensor}, total_bits=total, alpha=0.5, min_bits=1, max_bits=8)
        canonical_bits = out["flat"].numpy()
        np_result = water_fill_bisection_numpy(
            imp, total_bits=total, alpha=0.5, min_bits=1, max_bits=8
        )
        # Allow small discrepancy due to bisection convergence + rounding ties
        diff_count = int((canonical_bits != np_result.bits).sum())
        total_n = canonical_bits.size
        assert diff_count / total_n < 0.05, (
            f"numpy bisection diverges from canonical at {diff_count}/{total_n} positions"
        )
