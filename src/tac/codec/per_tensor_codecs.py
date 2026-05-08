"""Canonical per-tensor codec encoders for PR101 Path-B and cross-paradigm tools.

Three primitive encoders share the same call contract ``(symbols, ...) ->
(bytes_used, rel_err)``:

* :func:`encode_brotli_only` — direct brotli over int8 symbols (lossless).
* :func:`encode_sparsity_alpha` — top-(1-α) magnitude keep + delta-coded CSR
  payload, brotli-compressed.
* :func:`encode_lossy_K_coarsen` — K-step uniform quantization over the int
  range, brotli-compressed.

Historically these implementations were duplicated across
``tools/pr101_omega_opt_*.py`` and the UNIWARD-weighted lane. Centralizing them
here removes drift risk (the Lagrangian allocator and budget-greedy selectors
all assume identical encoder semantics) and gives the brotli parameter pin one
canonical home.

The brotli parameters ``(quality=11, lgwin=16, lgblock=19)`` are pinned to match
the PR101 substrate empirically validated on the public frontier; see
``feedback_pr101_lgwin13_q10_8byte_savings_20260507.md`` for the optimization
sweep that justified locking them at this point.

Public API:
    BROTLI_PARAMS — ``dict`` of pinned brotli kwargs.
    encode_brotli_only(symbols)
    encode_sparsity_alpha(symbols, alpha)
    encode_lossy_K_coarsen(symbols, K)

Score-relevance: this module computes byte counts for the PR101 substrate.
Any change to ``BROTLI_PARAMS`` or encoder framing changes the bytes the
allocator sees and therefore changes Lagrangian solutions. Treat as a contest
contract.
"""
from __future__ import annotations

import struct
from typing import Final

import brotli
import numpy as np


BROTLI_PARAMS: Final[dict[str, int]] = {
    "quality": 11,
    "lgwin": 16,
    "lgblock": 19,
}
"""Pinned brotli parameters for the PR101 substrate.

Modifying these values silently changes byte budgets used by the Lagrangian
allocator. Any modification must be accompanied by a sweep memo + the ledger
seed update in ``tools/cathedral_autopilot.py``.
"""


def _brotli_compress(payload: bytes) -> bytes:
    """Compress ``payload`` under :data:`BROTLI_PARAMS` (single canonical call)."""
    return brotli.compress(payload, **BROTLI_PARAMS)


def encode_brotli_only(symbols: np.ndarray) -> tuple[int, float]:
    """Brotli-encode int8 symbols losslessly.

    Args:
        symbols: 1-D ``np.ndarray`` of int-valued symbols in ``[-127, 127]``.

    Returns:
        ``(bytes_used, rel_err)`` where ``rel_err == 0.0`` (lossless).
    """
    return len(_brotli_compress(symbols.tobytes())), 0.0


def encode_sparsity_alpha(symbols: np.ndarray, alpha: float) -> tuple[int, float]:
    """Drop the smallest-magnitude ``alpha`` fraction; encode the rest as CSR.

    The reconstruction is the kept-values vector with zeros elsewhere; bytes
    are counted on the brotli compression of a ``<II>`` header + delta-coded
    indices + int8 values.

    Args:
        symbols: 1-D int symbols (any int dtype).
        alpha: drop fraction in ``[0.0, 1.0]``. ``0.0`` → fall through to
            :func:`encode_brotli_only`. ``>= 1.0`` (or rounding equivalent) →
            also fall through to lossless.

    Returns:
        ``(bytes_used, rel_err)`` where ``rel_err`` is the L2 relative error
        between the symbol vector and the sparsified reconstruction.
    """
    n = symbols.size
    if alpha <= 0:
        return encode_brotli_only(symbols)
    n_keep = max(1, int(round((1.0 - alpha) * n)))
    if n_keep >= n:
        return encode_brotli_only(symbols)

    flat = symbols.flatten()
    abs_vals = np.abs(flat.astype(np.int32))
    top_idx = np.argpartition(abs_vals, n - n_keep)[n - n_keep:]
    top_idx_sorted = np.sort(top_idx)
    nz_values = flat[top_idx_sorted].astype(np.int8)

    recon = np.zeros_like(flat, dtype=np.int8)
    recon[top_idx_sorted] = nz_values

    diff = flat.astype(np.float64) - recon.astype(np.float64)
    orig_l2 = float(np.linalg.norm(flat.astype(np.float64))) + 1e-12
    rel_err = float(np.linalg.norm(diff)) / orig_l2

    deltas = np.diff(
        np.concatenate([np.array([0], dtype=np.uint32), top_idx_sorted.astype(np.uint32)])
    ).astype(np.uint32)
    payload = struct.pack("<II", n, nz_values.size) + deltas.tobytes() + nz_values.tobytes()
    return len(_brotli_compress(payload)), rel_err


def encode_lossy_K_coarsen(
    symbols: np.ndarray, K: int, *, clip_lo: int = -127, clip_hi: int = 127
) -> tuple[int, float]:
    """Round symbols to multiples of ``K``, brotli-encode the int8 result.

    This mirrors the per-tensor "byte_proxy" used by the K-curve precompute
    paths (``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py``).
    The byte count is over the per-tensor brotli of the rounded int8 stream;
    callers that want joint-encoded bytes should use
    :func:`tac.codec.cost_curves.encode_with_per_tensor_K` (which is the
    canonical joint encoder in ``tools/pr101_lossy_coarsening_analytical.py``).

    Args:
        symbols: 1-D int symbol vector.
        K: rounding step (``>= 1``).
        clip_lo: lower clip bound before int8 cast.
        clip_hi: upper clip bound before int8 cast.

    Returns:
        ``(bytes_used, rel_err)`` where ``rel_err`` is the L1 relative error
        ``sum|round-orig| / sum|orig|``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")
    rounded = np.round(symbols / K) * K
    abs_orig = float(np.abs(symbols).astype(np.float64).sum())
    rel_err = (
        float(np.abs(rounded - symbols).astype(np.float64).sum()) / abs_orig
        if abs_orig > 1e-12
        else 0.0
    )
    rounded_i8 = rounded.clip(clip_lo, clip_hi).astype(np.int8)
    return len(_brotli_compress(rounded_i8.tobytes())), rel_err


__all__ = [
    "BROTLI_PARAMS",
    "encode_brotli_only",
    "encode_sparsity_alpha",
    "encode_lossy_K_coarsen",
]
