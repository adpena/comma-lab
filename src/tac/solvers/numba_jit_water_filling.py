# SPDX-License-Identifier: MIT
"""Numba-JIT bisection water-filling: 50-200x speedup over Python overhead.

Numba: https://github.com/numba/numba (BSD-licensed).

Paired-comparison contract with ``tac.bit_allocator.allocate_bits``:
* The canonical ``allocate_bits`` runs a 64-iteration binary search on the
  Lagrangian multiplier c, scoring each candidate via torch tensor ops over
  the full flattened importance vector. For N=100K weights this is dominated
  by Python/PyTorch interpreter overhead, not the actual arithmetic.
* This module replaces the inner bisection scoring with a NumPy / Numba-JIT
  callable that operates on plain np.ndarray. The bisection logic is
  identical; only the per-iteration scoring kernel is JIT'd.

Hypothesis: 50-200x wall-clock speedup at $0 GPU cost; bit allocations
byte-identical to the canonical bisection (same Lagrangian, same rounding).

[empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
[macOS-CPU advisory] — local CPU advisory only; never promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NumbaWaterFillingResult",
    "water_fill_bisection_numpy",
    "water_fill_bisection_numba_if_available",
    "NUMBA_AVAILABLE",
]

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class NumbaWaterFillingResult:
    """Outcome of bisection water-filling on per-weight importance."""

    bits: np.ndarray
    lagrangian_c: float
    iterations: int
    backend: str  # "numba" or "numpy"


def _score_at_c_numpy(
    flat_imp: np.ndarray, c: float, alpha: float, min_bits: int, max_bits: int
) -> tuple[int, np.ndarray]:
    """Compute total-bits and per-weight allocation for a given Lagrangian c."""
    raw = c * np.power(flat_imp, alpha)
    rounded = np.clip(np.round(raw), min_bits, max_bits).astype(np.int64)
    return int(rounded.sum()), rounded


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _score_at_c_numba(
        flat_imp: np.ndarray, c: float, alpha: float, min_bits: int, max_bits: int
    ) -> tuple[int, np.ndarray]:  # pragma: no cover - JIT wrapper
        n = flat_imp.size
        rounded = np.empty(n, dtype=np.int64)
        total = 0
        for i in range(n):
            raw = c * (flat_imp[i] ** alpha)
            r = int(round(raw))
            if r < min_bits:
                r = min_bits
            elif r > max_bits:
                r = max_bits
            rounded[i] = r
            total += r
        return total, rounded
else:
    _score_at_c_numba = None


def _bisect(
    score_fn: callable,
    flat_imp: np.ndarray,
    total_bits: int,
    alpha: float,
    min_bits: int,
    max_bits: int,
    iters: int,
) -> tuple[np.ndarray, float, int]:
    n = flat_imp.size
    floor_total = n * min_bits
    ceiling_total = n * max_bits
    if total_bits < floor_total:
        raise ValueError("infeasible budget (below floor)")
    if total_bits >= ceiling_total:
        return np.full(n, max_bits, dtype=np.int64), float("inf"), 0
    max_imp = float(flat_imp.max())
    if max_imp == 0.0:
        return np.full(n, min_bits, dtype=np.int64), 0.0, 0

    c_lo = 0.0
    c_hi = max_bits / max(max_imp**alpha, 1e-30) * 2.0
    last_bits = None
    for k in range(iters):
        c_mid = 0.5 * (c_lo + c_hi)
        total, bits = score_fn(flat_imp, c_mid, alpha, min_bits, max_bits)
        last_bits = bits
        if total > total_bits:
            c_hi = c_mid
        else:
            c_lo = c_mid
    assert last_bits is not None
    return last_bits, 0.5 * (c_lo + c_hi), iters


def water_fill_bisection_numpy(
    flat_imp: np.ndarray,
    total_bits: int,
    *,
    alpha: float = 0.5,
    min_bits: int = 1,
    max_bits: int = 8,
    iters: int = 64,
) -> NumbaWaterFillingResult:
    """Pure-NumPy bisection water-filling (no Numba required)."""
    bits, c, k = _bisect(
        _score_at_c_numpy, flat_imp, total_bits, alpha, min_bits, max_bits, iters
    )
    return NumbaWaterFillingResult(bits=bits, lagrangian_c=c, iterations=k, backend="numpy")


def water_fill_bisection_numba_if_available(
    flat_imp: np.ndarray,
    total_bits: int,
    *,
    alpha: float = 0.5,
    min_bits: int = 1,
    max_bits: int = 8,
    iters: int = 64,
) -> NumbaWaterFillingResult:
    """Numba-JIT bisection water-filling; falls back to NumPy if numba missing."""
    if not NUMBA_AVAILABLE:
        return water_fill_bisection_numpy(
            flat_imp, total_bits, alpha=alpha, min_bits=min_bits, max_bits=max_bits, iters=iters
        )
    bits, c, k = _bisect(
        _score_at_c_numba, flat_imp, total_bits, alpha, min_bits, max_bits, iters
    )
    return NumbaWaterFillingResult(bits=bits, lagrangian_c=c, iterations=k, backend="numba")
