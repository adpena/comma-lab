# SPDX-License-Identifier: MIT
"""FISTA: Fast Iterative Shrinkage-Thresholding Algorithm.

Beck & Teboulle 2009, "A Fast Iterative Shrinkage-Thresholding Algorithm for
Linear Inverse Problems", SIAM J. Imag. Sci. 2(1):183-202.
DOI 10.1137/080716542. Algorithm 4.1 (FISTA with constant step).

O(1/k^2) convergence vs vanilla proximal gradient's O(1/k).

Paired-comparison contract with ``tac.water_filling_codec``:
* Both solve the per-tensor bit-allocation problem under a total budget.
* Water-filling closed-form via Lagrangian KKT bisection (Boyd & Vandenberghe
  §5.5.3) — optimal for the L1-relaxed problem with non-negative variables.
* FISTA solves a smooth proximal step + non-negativity projection — converges
  to the same optimum but with explicit accelerated descent. Hypothesis:
  FISTA wins on smooth quality curves (continuous bits); water-filling wins
  when the bit grid is sparse / heavily clamped.

[empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
[macOS-CPU advisory] — local CPU advisory only; never promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FistaResult",
    "fista_proximal_gradient",
    "soft_threshold",
    "project_simplex",
]


@dataclass(frozen=True)
class FistaResult:
    """Outcome of FISTA: solution, objective trajectory, iterations to converge."""

    x: np.ndarray
    objective_history: tuple[float, ...]
    iterations: int
    converged: bool


def soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Proximal operator of lambda * ||x||_1: shrink magnitudes toward 0."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def project_simplex(v: np.ndarray, budget: float) -> np.ndarray:
    """Project v onto the simplex {x : x >= 0, sum(x) = budget}.

    Duchi et al. 2008 O(n log n) algorithm via sorting.
    """
    n = v.size
    if budget <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - budget
    rho = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        return np.zeros_like(v)
    rho_star = rho[-1]
    theta = cssv[rho_star] / (rho_star + 1)
    return np.maximum(v - theta, 0.0)


def fista_proximal_gradient(
    grad_smooth: callable,
    prox_operator: callable,
    x0: np.ndarray,
    *,
    step_size: float,
    max_iters: int = 500,
    tol: float = 1e-7,
    objective_fn: callable = None,
) -> FistaResult:
    """Beck-Teboulle 2009 FISTA: accelerated proximal gradient.

    Args:
        grad_smooth: callable mapping x -> grad of the smooth part f(x).
        prox_operator: callable mapping (z, step) -> prox of step * g at z.
        x0: starting point.
        step_size: 1/L where L is the Lipschitz constant of grad_smooth.
        max_iters: maximum iterations.
        tol: convergence tolerance on ||x_{k+1} - x_k||_2.
        objective_fn: optional callable for objective trajectory logging.

    Returns:
        FistaResult with final x, history, iteration count, convergence flag.
    """
    x = x0.astype(np.float64).copy()
    y = x.copy()
    t = 1.0
    history: list[float] = []
    if objective_fn is not None:
        history.append(float(objective_fn(x)))

    for k in range(max_iters):
        grad_y = grad_smooth(y)
        z = y - step_size * grad_y
        x_new = prox_operator(z, step_size)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)

        diff = float(np.linalg.norm(x_new - x))
        x = x_new
        t = t_new

        if objective_fn is not None:
            history.append(float(objective_fn(x)))

        if diff < tol:
            return FistaResult(
                x=x, objective_history=tuple(history), iterations=k + 1, converged=True
            )

    return FistaResult(
        x=x, objective_history=tuple(history), iterations=max_iters, converged=False
    )
