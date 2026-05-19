# SPDX-License-Identifier: MIT
"""Sinkhorn-Knopp: entropic-regularized optimal transport.

Cuturi 2013, "Sinkhorn Distances: Lightspeed Computation of Optimal
Transport", NeurIPS 2013. https://papers.nips.cc/paper/4927.

Peyré & Cuturi 2019, "Computational Optimal Transport",
https://arxiv.org/abs/1803.00567 (canonical OT reference).

Solves min <C, P> + (1/lambda) * H(P) over doubly-stochastic P with row
sums = a and column sums = b. Closed form via matrix scaling: alternately
normalize K = exp(-lambda * C) by row and column sums.

Paired-comparison contract with Frank-Wolfe K-cardinality:
* Sinkhorn produces soft assignments (any row sums to its target marginal);
  natural fit for entropic-regularized ensemble selection where you want
  weighted averaging of K members.
* Frank-Wolfe over the K-cardinality polytope produces a discrete K-subset
  in the limit (extremal vertex), better for hard cardinality constraints.

[empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
[macOS-CPU advisory] — local CPU advisory only; never promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SinkhornResult",
    "sinkhorn_knopp",
    "sinkhorn_ensemble_select_k",
]


@dataclass(frozen=True)
class SinkhornResult:
    """Outcome of Sinkhorn matrix-scaling iteration."""

    P: np.ndarray
    iterations: int
    converged: bool
    marginal_error: float


def sinkhorn_knopp(
    cost: np.ndarray,
    row_marginal: np.ndarray,
    col_marginal: np.ndarray,
    *,
    reg_lambda: float = 10.0,
    max_iters: int = 200,
    tol: float = 1e-7,
) -> SinkhornResult:
    """Sinkhorn-Knopp matrix scaling for entropic-regularized OT.

    Args:
        cost: (m, n) cost matrix (>= 0).
        row_marginal: (m,) target row sums (sum to total mass).
        col_marginal: (n,) target column sums (must sum to same total).
        reg_lambda: regularization strength (larger => harder OT, less smooth).
        max_iters: maximum Sinkhorn iterations.
        tol: convergence tolerance on marginal L1 error.

    Returns:
        SinkhornResult with transport plan P, iteration count, convergence.
    """
    total_row = float(row_marginal.sum())
    total_col = float(col_marginal.sum())
    if not np.isclose(total_row, total_col, atol=1e-6):
        raise ValueError(
            f"row_marginal sum {total_row:.6f} != col_marginal sum {total_col:.6f}"
        )

    K = np.exp(-reg_lambda * cost)
    K = np.maximum(K, 1e-300)  # numerical floor to avoid 0/0

    u = np.ones(cost.shape[0]) / cost.shape[0]
    v = np.ones(cost.shape[1]) / cost.shape[1]

    for j in range(max_iters):
        u_new = row_marginal / (K @ v + 1e-300)
        v_new = col_marginal / (K.T @ u_new + 1e-300)
        err = float(np.max(np.abs(u_new - u)) + np.max(np.abs(v_new - v)))
        u, v = u_new, v_new
        if err < tol:
            P = np.diag(u) @ K @ np.diag(v)
            row_err = float(np.max(np.abs(P.sum(axis=1) - row_marginal)))
            col_err = float(np.max(np.abs(P.sum(axis=0) - col_marginal)))
            return SinkhornResult(
                P=P,
                iterations=j + 1,
                converged=True,
                marginal_error=row_err + col_err,
            )

    P = np.diag(u) @ K @ np.diag(v)
    row_err = float(np.max(np.abs(P.sum(axis=1) - row_marginal)))
    col_err = float(np.max(np.abs(P.sum(axis=0) - col_marginal)))
    return SinkhornResult(
        P=P, iterations=max_iters, converged=False, marginal_error=row_err + col_err
    )


def sinkhorn_ensemble_select_k(
    member_quality_scores: np.ndarray, k: int, *, reg_lambda: float = 5.0
) -> tuple[np.ndarray, SinkhornResult]:
    """Use Sinkhorn entropic-OT to softly select K-member subset from N.

    Formulation: each of K "slot" rows has marginal = 1. The N candidate
    columns have proportional marginals = k * exp(reg_lambda * quality) /
    sum(exp(reg_lambda * quality)) — i.e. the candidate "capacity" is itself
    quality-weighted. With reg_lambda = 0 we get uniform spread (no concentration);
    with large reg_lambda the top-K candidates dominate the column marginal and
    Sinkhorn produces nearly-hard K-of-N selection. The cost matrix is all-zeros
    inside this helper because the candidate ranking is already encoded in the
    column marginals — the entropic OT then just respects marginals.

    Returns: (per-member weights summing to k, SinkhornResult).
    """
    n = member_quality_scores.size
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    # Candidate marginal: quality-weighted softmax scaled to total mass k
    shifted = member_quality_scores - member_quality_scores.min()
    weights_unnorm = np.exp(reg_lambda * shifted)
    col_marginal = k * weights_unnorm / weights_unnorm.sum()
    row_marginal = np.ones(k, dtype=np.float64)
    # All-zero cost: respect marginals only; quality lives in the column marginal
    cost = np.zeros((k, n), dtype=np.float64)
    # reg_lambda inside sinkhorn_knopp doesn't matter when cost is uniform; pick small
    result = sinkhorn_knopp(
        cost, row_marginal, col_marginal, reg_lambda=1.0
    )
    weights = result.P.sum(axis=0)
    return weights, result
