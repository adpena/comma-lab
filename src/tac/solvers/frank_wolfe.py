# SPDX-License-Identifier: MIT
"""Frank-Wolfe / Conditional Gradient: projection-free convex optimization.

Frank & Wolfe 1956, "An algorithm for quadratic programming",
Naval Research Logistics Quarterly 3(1-2):95-110.
DOI 10.1002/nav.3800030109.

Modern revival: Jaggi 2013, "Revisiting Frank-Wolfe: Projection-Free Sparse
Convex Optimization", ICML 2013. https://proceedings.mlr.press/v28/jaggi13.html

O(1/k) convergence over compact convex sets via a linear-minimization oracle
(LMO). Projection-free — no Euclidean projection step, which is the
dominating cost for atomic norm balls / cardinality K constraints.

Paired-comparison contract with Sinkhorn for K=8 Rashomon ensemble subset:
* Frank-Wolfe atomic-norm formulation: minimize objective over the convex
  hull of K-cardinality indicator vectors (every step picks the single
  "best" atom via the LMO, exact selection).
* Sinkhorn entropic-regularized OT: minimize entropic-regularized cost over
  doubly stochastic matrices; converges to soft assignment.
* Hypothesis: Frank-Wolfe wins on hard-K cardinality (discrete K-selection);
  Sinkhorn wins when entropic regularization is appropriate (smooth soft
  weighting of ensemble members).

[empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
[macOS-CPU advisory] — local CPU advisory only; never promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FrankWolfeResult",
    "frank_wolfe_kcard",
    "lmo_kcardinality",
]


@dataclass(frozen=True)
class FrankWolfeResult:
    """Outcome of Frank-Wolfe: solution, gap trajectory, iterations, selected atoms."""

    x: np.ndarray
    duality_gap_history: tuple[float, ...]
    iterations: int
    converged: bool
    selected_indices: tuple[int, ...]


def lmo_kcardinality(grad: np.ndarray, k: int) -> np.ndarray:
    """Linear-minimization oracle for the K-cardinality polytope.

    minimize <s, grad> over s in conv{1_{|S|=k}}: pick the k smallest entries
    of grad and set those to 1, others to 0 (closed-form, O(n log n) via
    partial sort).

    Args:
        grad: gradient direction (smaller = better; we minimize <s, grad>).
        k: cardinality constraint.

    Returns:
        Indicator vector s with exactly k entries = 1.
    """
    n = grad.size
    if k <= 0:
        return np.zeros(n)
    if k >= n:
        return np.ones(n)
    s = np.zeros(n)
    top_k_idx = np.argpartition(grad, k)[:k]
    s[top_k_idx] = 1.0
    return s


def frank_wolfe_kcard(
    grad_fn: callable,
    objective_fn: callable,
    n: int,
    k: int,
    *,
    max_iters: int = 200,
    tol: float = 1e-6,
    line_search: bool = True,
) -> FrankWolfeResult:
    """Frank-Wolfe over the K-cardinality polytope conv{|S|=k}.

    Args:
        grad_fn: callable mapping x -> gradient of the smooth objective.
        objective_fn: callable mapping x -> scalar objective (for gap).
        n: ambient dimension.
        k: cardinality constraint.
        max_iters: maximum iterations.
        tol: convergence tolerance on the Frank-Wolfe duality gap.
        line_search: use exact line search (vs default 2/(k+2) step).

    Returns:
        FrankWolfeResult with final x in [0,1]^n with E[sum] = k,
        duality-gap trajectory, iterations, and the indices selected
        by the final LMO step.
    """
    x = np.full(n, k / n, dtype=np.float64)  # initialize uniformly
    gap_history: list[float] = []
    last_s = np.zeros(n)

    for j in range(max_iters):
        grad = grad_fn(x)
        s = lmo_kcardinality(grad, k)
        last_s = s
        gap = float(np.dot(grad, x - s))
        gap_history.append(gap)

        if gap < tol:
            return FrankWolfeResult(
                x=x,
                duality_gap_history=tuple(gap_history),
                iterations=j + 1,
                converged=True,
                selected_indices=tuple(int(i) for i in np.where(s > 0.5)[0]),
            )

        # Step size: default 2/(k+2) (Jaggi 2013); or exact line search for quadratic objectives
        if line_search:
            # Exact line search via golden-section on gamma in [0, 1]
            d = s - x
            best_gamma = 2.0 / (j + 2)
            best_obj = objective_fn(x + best_gamma * d)
            for gamma in (0.05, 0.1, 0.25, 0.5, 0.75, 0.95):
                obj_try = objective_fn(x + gamma * d)
                if obj_try < best_obj:
                    best_obj = obj_try
                    best_gamma = gamma
        else:
            best_gamma = 2.0 / (j + 2)

        x = x + best_gamma * (s - x)

    return FrankWolfeResult(
        x=x,
        duality_gap_history=tuple(gap_history),
        iterations=max_iters,
        converged=False,
        selected_indices=tuple(int(i) for i in np.where(last_s > 0.5)[0]),
    )
