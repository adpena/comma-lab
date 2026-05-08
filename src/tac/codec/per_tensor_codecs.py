"""Canonical per-tensor codec encoders for PR101 Path-B and cross-paradigm tools.

Three primitive encoders share the same call contract ``(symbols, ...) ->
(bytes_used, rel_err)``:

* :func:`encode_brotli_only` — direct brotli over int8 symbols (lossless).
* :func:`encode_sparsity_alpha` — top-(1-α) magnitude keep + delta-coded CSR
  payload, brotli-compressed.
* :func:`encode_lossy_K_coarsen` — K-step uniform quantization over the int
  range, brotli-compressed.

``rel_err`` semantics
---------------------

Per the 2026-05-08 council deliberation
(``.omx/research/rel_err_inconsistency_audit_20260508_claude.md``) two of the
three encoders historically used DIFFERENT ``rel_err`` definitions:

* :func:`encode_sparsity_alpha` returns an **L2 ratio** (Euclidean).
* :func:`encode_lossy_K_coarsen` returns an **L1 ratio** (sum of absolutes).

These numerical behaviors are PRESERVED so every shipped empirical anchor
(0.0386 / 0.0415 PR101 Path-B, 0.0019 et al.) stays comparable to its
historical evidence row. New callers should use the canonical
:func:`tac.codec.rel_err.compute_rel_err` directly instead of relying on the
encoder's reported ``rel_err`` field; the canonical default is RMS.

Centralizing the encoders here removes drift risk (the Lagrangian allocator
and budget-greedy selectors all assume identical encoder semantics) and gives
the brotli parameter pin one canonical home. The ``rel_err_form`` tag emitted
by :mod:`tac.codec.cost_curves` lets the allocator assert form-uniformity at
its bisection entry.

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


REL_ERR_FORM_BROTLI_ONLY: Final[str] = "rms"
"""Lossless: ``rel_err == 0`` is form-agnostic; we declare ``rms`` to keep
curves form-uniform with downstream RMS-aggregating selectors."""

REL_ERR_FORM_SPARSITY: Final[str] = "l2_ratio"
"""``encode_sparsity_alpha`` returns ``‖diff‖₂ / (‖orig‖₂ + ε)`` — L2 ratio.

This number equals RMS when both numerator and denominator share the same
``sqrt(N)`` normalization (which they do, elementwise over the same vector).
We keep the legacy form tag to be explicit that the encoder reports the
ratio of L2 norms, not the RMS expression directly.
"""

REL_ERR_FORM_LOSSY_K: Final[str] = "l1_ratio"
"""``encode_lossy_K_coarsen`` returns ``Σ|diff| / Σ|orig|`` — global L1 ratio.

Preserved verbatim to keep PR101 lossy_coarsening / PR106 UNIWARD-packet
empirical anchors valid. New callers wanting RMS should pass the encoder's
intermediate arrays through :func:`tac.codec.rel_err.compute_rel_err` with
``mode=RelErrForm.RMS`` instead of consuming this field.
"""


def encode_brotli_only(symbols: np.ndarray) -> tuple[int, float]:
    """Brotli-encode int8 symbols losslessly.

    The returned ``rel_err`` is exactly ``0.0`` (lossless); for curve-row
    tagging the form is canonically ``"rms"`` (form-agnostic at zero).

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
        ``(bytes_used, rel_err)`` where ``rel_err`` is the **L2 relative
        error** ``‖diff‖₂ / (‖orig‖₂ + ε)`` between the symbol vector and
        the sparsified reconstruction. The corresponding curve-row tag is
        :data:`REL_ERR_FORM_SPARSITY`.
    """  # REL_ERR_NON_CANONICAL_OK: L2 ratio preserved for sparsity-curve anchors
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
        ``(bytes_used, rel_err)`` where ``rel_err`` is the **L1 relative
        error** ``sum|round-orig| / sum|orig|`` (global L1 ratio). The
        corresponding curve-row tag is :data:`REL_ERR_FORM_LOSSY_K`.
        New callers wanting the canonical RMS form should call
        :func:`tac.codec.rel_err.compute_rel_err` with
        ``mode=RelErrForm.RMS`` on the returned reconstruction directly.
    """  # REL_ERR_NON_CANONICAL_OK: L1 ratio preserved for PR101 lossy_coarsening / PR106 UNIWARD anchors
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
    "REL_ERR_FORM_BROTLI_ONLY",
    "REL_ERR_FORM_LOSSY_K",
    "REL_ERR_FORM_SPARSITY",
    "encode_brotli_only",
    "encode_sparsity_alpha",
    "encode_lossy_K_coarsen",
]
