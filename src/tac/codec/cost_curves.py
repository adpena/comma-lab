"""Per-tensor cost-curve precompute and budget-greedy selectors.

Curve types
-----------

* **Discrete sparsity curves** — for each tensor a list of
  ``{"alpha": float, "bytes": int, "rel_err": float}`` rows over a brotli-only
  baseline plus ``encode_sparsity_alpha(symbols, alpha)`` for each ``alpha``.
  See :func:`precompute_per_tensor_sparsity_curves`.

* **Continuous K curves** — for each tensor a list of
  ``{"K": int, "rel_err": float, "byte_proxy": int}`` rows over a K range. See
  :func:`precompute_per_tensor_K_curves`.

Both curve forms are consumed by :class:`tac.optimization.lagrangian_per_tensor_allocation.LagrangianPerTensorAllocator`.
The cost-curve and selector signatures are kept identical so the allocator
runs over either curve type unchanged.

Selectors
---------

* :func:`greedy_uniform_per_tensor_budget_sparsity` — each tensor picks the
  smallest-byte sparsity codec satisfying ``rel_err <= budget``.
* :func:`greedy_uniform_per_tensor_K` — each tensor finds the largest K such
  that per-tensor ``rel_err <= budget``.

Score-relevance: this module emits byte/rel_err per-tensor curves used by the
allocator. Identity of byte values matters; do not log-round, do not skip
points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .per_tensor_codecs import (
    REL_ERR_FORM_LOSSY_K,
    REL_ERR_FORM_SPARSITY,
    encode_brotli_only,
    encode_lossy_K_coarsen,
    encode_sparsity_alpha,
)
from .rel_err import REL_ERR_FORM_KEY


# ---------------------------------------------------------------------------
# Tensor blob (re-exported for callers; canonical type lives here)
# ---------------------------------------------------------------------------


@dataclass
class TensorBlob:
    """Named int symbol vector for per-tensor codec evaluation.

    Attributes:
        name: the FIXED_STATE_SCHEMA tensor name.
        raw: 1-D ``np.ndarray`` of int symbols in ``[-127, 127]``.
    """
    name: str
    raw: np.ndarray


# ---------------------------------------------------------------------------
# Discrete sparsity-α curves
# ---------------------------------------------------------------------------


def precompute_per_tensor_sparsity_curves(
    quantized: Sequence[tuple[str, np.ndarray]] | Sequence[TensorBlob],
    alphas: Iterable[float],
) -> list[list[dict]]:
    """For each tensor return ``[{alpha, bytes, rel_err}, ...]``.

    Includes the lossless brotli-only row at ``alpha=0.0`` plus one row per
    alpha in ``alphas``.

    Args:
        quantized: sequence of ``(name, symbols)`` tuples or
            :class:`TensorBlob` instances.
        alphas: iterable of drop fractions in ``[0.0, 1.0]``.

    Returns:
        Per-tensor list of curve rows.
    """
    alphas_list = list(alphas)
    out: list[list[dict]] = []
    for entry in quantized:
        if isinstance(entry, TensorBlob):
            syms = entry.raw
        else:
            _, syms = entry
        rows: list[dict] = []
        b0, e0 = encode_brotli_only(syms)
        rows.append({
            "alpha": 0.0,
            "bytes": b0,
            "rel_err": e0,
            REL_ERR_FORM_KEY: REL_ERR_FORM_SPARSITY,
        })
        for alpha in alphas_list:
            b, e = encode_sparsity_alpha(syms, alpha)
            rows.append({
                "alpha": alpha,
                "bytes": b,
                "rel_err": e,
                REL_ERR_FORM_KEY: REL_ERR_FORM_SPARSITY,
            })
        out.append(rows)
    return out


def greedy_uniform_per_tensor_budget_sparsity(
    curves: Sequence[Sequence[dict]], budget: float
) -> tuple[int, float]:
    """For each tensor pick smallest-byte codec with ``rel_err <= budget``.

    The aggregate ``rel_err`` is reported as the RMS of per-tensor rel_errs
    (matching the historical Path-B step-4 implementation). When the input
    curves were emitted by :func:`precompute_per_tensor_sparsity_curves` the
    per-tensor inputs are L2 ratios and the RMS aggregate is well-formed
    (RMS-of-L2-ratios is itself an L2-ratio over the concatenated symbol
    space). When the input curves are mixed-form, prefer
    :func:`tac.codec.rel_err.aggregate_rel_err` instead and tag downstream
    evidence rows with ``rel_err_form`` explicitly.

    Args:
        curves: per-tensor sparsity curves.
        budget: per-tensor ``rel_err`` cap.

    Returns:
        ``(total_bytes, achieved_rms_rel_err)``.
    """
    total_bytes = 0
    rel_errs: list[float] = []
    for tensor_rows in curves:
        valid = [r for r in tensor_rows if r["rel_err"] <= budget]
        best = min(valid, key=lambda r: r["bytes"])
        total_bytes += int(best["bytes"])
        rel_errs.append(float(best["rel_err"]))
    rms = float(np.sqrt(np.mean([e ** 2 for e in rel_errs])))
    return total_bytes, rms


# ---------------------------------------------------------------------------
# Continuous K curves
# ---------------------------------------------------------------------------


DEFAULT_K_RANGE: tuple[int, ...] = tuple(range(1, 65))
"""Canonical K range for lossy_coarsening curves (matches Path-B step-6)."""


def precompute_per_tensor_K_curves(
    tensors: Sequence[TensorBlob],
    K_range: Iterable[int] = DEFAULT_K_RANGE,
) -> list[list[dict]]:
    """For each tensor compute ``[{K, rel_err, byte_proxy}, ...]``.

    ``byte_proxy`` is per-tensor brotli over the K-rounded int8 — an
    over-estimate of the joint contribution but monotone in K, which is all
    the Lagrangian selector requires.

    Args:
        tensors: list of :class:`TensorBlob`.
        K_range: iterable of K values (``K >= 1``).

    Returns:
        Per-tensor curves.
    """
    K_list = list(K_range)
    curves: list[list[dict]] = []
    for tb in tensors:
        rows: list[dict] = []
        for K in K_list:
            byte_proxy, rel_err = encode_lossy_K_coarsen(tb.raw, K)
            rows.append({
                "K": int(K),
                "rel_err": rel_err,
                "byte_proxy": byte_proxy,
                REL_ERR_FORM_KEY: REL_ERR_FORM_LOSSY_K,
            })
        curves.append(rows)
    return curves


def find_best_K_for_tensor(
    symbols: np.ndarray, budget: float, *, K_max: int = 256
) -> tuple[int, float]:
    """Find the largest K such that per-tensor ``rel_err <= budget``.

    Args:
        symbols: 1-D int symbol vector.
        budget: per-tensor ``rel_err`` cap.
        K_max: exclusive upper bound on the K search.

    Returns:
        ``(best_K, achieved_rel_err)``. If even ``K=1`` exceeds ``budget``,
        ``(1, achieved_rel_err_at_K=1)`` is returned (which may exceed budget
        — caller's responsibility to validate).
    """
    abs_sum = float(np.abs(symbols).astype(np.float64).sum())
    if abs_sum < 1e-9:
        return 1, 0.0
    best_K = 1
    best_re = 0.0
    for K in range(1, K_max):
        rounded = np.round(symbols / K) * K
        err = float(np.abs(rounded - symbols).astype(np.float64).sum())
        re = err / abs_sum
        if re <= budget:
            best_K = K
            best_re = re
        else:
            break
    return best_K, best_re


__all__ = [
    "TensorBlob",
    "DEFAULT_K_RANGE",
    "precompute_per_tensor_sparsity_curves",
    "precompute_per_tensor_K_curves",
    "greedy_uniform_per_tensor_budget_sparsity",
    "find_best_K_for_tensor",
]
