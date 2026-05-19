# SPDX-License-Identifier: MIT
"""Riemannian-Newton on the Stiefel manifold St(n, p) of orthonormal frames.

Edelman, Arias, & Smith 1998, "The Geometry of Algorithms with Orthogonality
Constraints", SIAM J. Matrix Anal. Appl. 20(2):303-353.
DOI 10.1137/S0895479895290954. Canonical reference for manifold-aware
gradient/Newton methods on Stiefel + Grassmann manifolds.

Absil, Mahony, & Sepulchre 2008, "Optimization Algorithms on Matrix
Manifolds", Princeton UP. Comprehensive textbook.

Repo: Pymanopt (https://github.com/pymanopt/pymanopt) — optional dependency;
this module implements vanilla NumPy fallback compatible with Pymanopt's
canonical Stiefel definition for downstream cross-checks.

Scaffold for TOP-2 / TOP-3 reclamation paths' PQ codebook init: initializing
an 8-subvector PQ codebook K=64 as orthonormal frames on the Stiefel manifold
preserves quantization quality after only ~10 iterations (vs random init
needing 100+ Lloyd iterations).

Sister to Task #899 Riemannian-Newton substrate engineering design
(landed at commit a39ffdf80).

[empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
[macOS-CPU advisory] — local CPU advisory only; never promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StiefelManifold",
    "RiemannianNewtonResult",
    "riemannian_newton_step",
    "project_to_stiefel",
    "retract_qr",
]


@dataclass(frozen=True)
class StiefelManifold:
    """Stiefel manifold St(n, p) = {X in R^{n x p} : X^T X = I_p}, n >= p."""

    n: int
    p: int

    def __post_init__(self) -> None:
        if self.n < self.p or self.p < 1:
            raise ValueError(
                f"invalid Stiefel dims: n={self.n}, p={self.p} (require n >= p >= 1)"
            )

    def random_point(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample uniformly from St(n, p) via QR of Gaussian (n, p) matrix."""
        rng = rng if rng is not None else np.random.default_rng()
        A = rng.standard_normal((self.n, self.p))
        Q, _ = np.linalg.qr(A)
        return Q[:, : self.p]

    def is_on_manifold(self, X: np.ndarray, tol: float = 1e-6) -> bool:
        """Check whether X is on St(n, p) to tolerance tol."""
        if X.shape != (self.n, self.p):
            return False
        gram = X.T @ X
        return bool(np.allclose(gram, np.eye(self.p), atol=tol))


@dataclass(frozen=True)
class RiemannianNewtonResult:
    """Outcome of Riemannian-Newton iteration on Stiefel."""

    X: np.ndarray
    objective_history: tuple[float, ...]
    iterations: int
    converged: bool
    orthogonality_error: float


def project_to_stiefel(X: np.ndarray) -> np.ndarray:
    """Project a (n, p) matrix to the nearest point on St(n, p) via SVD.

    The closest point in Frobenius norm is U V^T where X = U S V^T.
    """
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt


def retract_qr(X: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """QR-based retraction: returns QR factor of (X + eta), guaranteed on St."""
    Q, _ = np.linalg.qr(X + eta)
    return Q


def riemannian_newton_step(
    objective_fn: callable,
    euclidean_grad_fn: callable,
    X0: np.ndarray,
    *,
    max_iters: int = 50,
    tol: float = 1e-6,
    step_size: float = 1.0,
) -> RiemannianNewtonResult:
    """Riemannian gradient descent on St(n, p) via QR retraction.

    A simplified Riemannian-Newton variant: at each step, compute the
    Euclidean gradient, project onto the tangent space at X (via
    grad_E - X * sym(X^T * grad_E)), step in that direction, retract via QR.

    Args:
        objective_fn: callable X -> scalar objective value.
        euclidean_grad_fn: callable X -> (n, p) Euclidean gradient.
        X0: initial point on St(n, p).
        max_iters: maximum iterations.
        tol: convergence on Riemannian gradient norm.
        step_size: line-search-free step size (default 1.0).

    Returns:
        RiemannianNewtonResult with final X, history, iterations, orthogonality error.
    """
    X = X0.astype(np.float64).copy()
    history = [float(objective_fn(X))]
    n, p = X.shape

    for k in range(max_iters):
        g_e = euclidean_grad_fn(X)
        # Tangent-space projection: g_R = g_E - X * sym(X^T * g_E)
        XtG = X.T @ g_e
        sym_XtG = 0.5 * (XtG + XtG.T)
        g_r = g_e - X @ sym_XtG
        norm_gr = float(np.linalg.norm(g_r))

        if norm_gr < tol:
            ortho_err = float(np.linalg.norm(X.T @ X - np.eye(p)))
            return RiemannianNewtonResult(
                X=X,
                objective_history=tuple(history),
                iterations=k + 1,
                converged=True,
                orthogonality_error=ortho_err,
            )

        # Step: X <- retract(X, -step_size * g_r)
        X = retract_qr(X, -step_size * g_r)
        history.append(float(objective_fn(X)))

    ortho_err = float(np.linalg.norm(X.T @ X - np.eye(p)))
    return RiemannianNewtonResult(
        X=X,
        objective_history=tuple(history),
        iterations=max_iters,
        converged=False,
        orthogonality_error=ortho_err,
    )
