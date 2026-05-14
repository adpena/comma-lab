# SPDX-License-Identifier: MIT
"""PR84 ``adaptive_range_mask`` adaptive-context primitive.

PR84's ``adaptive_range_mask/inflate.py`` upgrades PR81's fixed-context
range-coded mask to a per-position adaptive context. The C++ codec
(``range_mask_codec.cpp``) used by PR84 maintains per-context categorical
frequency tables that are updated row-by-row in raster order using
neighbouring decoded pixels as the context key. The reusable primitive
here is the PYTHON-LEVEL ANALOG: a small typed wrapper that accepts a
caller-supplied context-id function and a per-context categorical model,
and range-codes / decodes the symbol stream against it.

This is intentionally SUBSTRATE-FREE: the primitive does not commit to the
raster-scan context PR84 uses (up / left / prev / diagonal). The caller
supplies the per-position context ids; the primitive routes each symbol
through the corresponding categorical. This makes the primitive directly
applicable to:

* PR84-style 2D raster contexts (mask logits with neighbour context),
* 1D temporal contexts (frame-to-frame action streams with prev-frame),
* Sequential-prior contexts (HPACMini-style sequence models),
* Any per-position conditional cdf system.

Two primitives land here:

1. **AdaptiveContextSpec** — a frozen dataclass naming the per-context
   alphabet and the (n_contexts, alphabet) categorical table.

2. **encode_adaptive_context_stream / decode_adaptive_context_stream** —
   pure-numpy + constriction range coder over a flat ``(n_symbols,)``
   stream paired with a flat ``(n_symbols,)`` context-id stream.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr84_intake_20260505_auto/source/submissions/adaptive_range_mask/{inflate.py,range_mask_codec.cpp}``.

Why this differs from PR91's per-symbol probs (in ``pr91_hpac_grammar``)
================================================================

PR91's :func:`tac.packet_compiler.encode_categorical_stream` takes a FULL
``(n_symbols, alphabet)`` probability matrix — one distribution PER
symbol. PR84's adaptive context is a more compact representation: a small
``(n_contexts, alphabet)`` table where ``n_contexts << n_symbols``, and
the caller declares the context id of each symbol. This is the right
primitive when the caller's model has a small finite set of distinct
distributions (e.g. raster neighbour contexts: 32 contexts × 5 classes
versus 600 × 384 × 5 = 1.15M individual distributions).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + constriction.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; ``encode → decode`` is bit-exact on the
  ``pr84_adaptive_mask_context_v1`` golden vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import constriction
import numpy as np


# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdaptiveContextSpec:
    """Per-context categorical model for an adaptive-context range coder.

    Attributes
    ----------
    alphabet_size:
        Number of distinct symbol values (each symbol is in
        ``[0, alphabet_size)``).
    cdf_table:
        Categorical frequency table of shape ``(n_contexts, alphabet_size)``.
        Float-valued; the coder normalises internally by clipping at
        ``1e-10`` and rescaling to sum=1 per row.
    n_contexts:
        Convenience accessor; equals ``cdf_table.shape[0]``.
    """

    alphabet_size: int
    cdf_table: np.ndarray

    def __post_init__(self) -> None:
        if self.alphabet_size < 2:
            raise ValueError(
                f"alphabet_size must be >= 2; got {self.alphabet_size}"
            )
        if self.cdf_table.ndim != 2:
            raise ValueError(
                f"cdf_table must be 2D; got shape {self.cdf_table.shape}"
            )
        n_ctx, alphabet = self.cdf_table.shape
        if alphabet != self.alphabet_size:
            raise ValueError(
                f"cdf_table.shape[1] = {alphabet} must equal "
                f"alphabet_size = {self.alphabet_size}"
            )
        if n_ctx < 1:
            raise ValueError(
                f"cdf_table.shape[0] = {n_ctx} must be >= 1"
            )

    @property
    def n_contexts(self) -> int:
        return int(self.cdf_table.shape[0])


# ── Internal helpers ────────────────────────────────────────────────────────


def _normalise_row(row: np.ndarray, *, floor: float = 1e-10) -> np.ndarray:
    p = np.asarray(row, dtype=np.float64)
    p = np.maximum(p, floor)
    p /= p.sum()
    return p


def _build_categoricals(spec: AdaptiveContextSpec):
    """Build one Categorical per context id (cached)."""
    cats = []
    for ctx in range(spec.n_contexts):
        row = _normalise_row(spec.cdf_table[ctx])
        cats.append(
            constriction.stream.model.Categorical(probabilities=row, perfect=False)
        )
    return cats


# ── Public encode / decode ──────────────────────────────────────────────────


def encode_adaptive_context_stream(
    symbols: np.ndarray,
    context_ids: np.ndarray,
    spec: AdaptiveContextSpec,
) -> bytes:
    """Range-encode a stream where each symbol's distribution is selected
    by a per-position context id.

    Parameters
    ----------
    symbols:
        1D ``int``-castable array of symbol values. Each must satisfy
        ``0 <= s < spec.alphabet_size``.
    context_ids:
        1D ``int``-castable array PARALLEL to ``symbols``. Each id must
        satisfy ``0 <= id < spec.n_contexts``.
    spec:
        :class:`AdaptiveContextSpec` carrying the per-context cdf table.

    Returns
    -------
    bytes
        Big-endian byte serialisation of the constriction uint32 word
        stream. Matches PR103's wire format.
    """
    sym = np.asarray(symbols, dtype=np.int64).reshape(-1)
    ctx = np.asarray(context_ids, dtype=np.int64).reshape(-1)
    if sym.size != ctx.size:
        raise ValueError(
            f"symbols ({sym.size}) and context_ids ({ctx.size}) must have "
            "the same length"
        )
    if sym.size == 0:
        raise ValueError("must encode at least one symbol")
    if int(sym.min()) < 0 or int(sym.max()) >= spec.alphabet_size:
        raise ValueError(
            f"symbols out of range [0, {spec.alphabet_size}); "
            f"min={int(sym.min())} max={int(sym.max())}"
        )
    if int(ctx.min()) < 0 or int(ctx.max()) >= spec.n_contexts:
        raise ValueError(
            f"context_ids out of range [0, {spec.n_contexts}); "
            f"min={int(ctx.min())} max={int(ctx.max())}"
        )
    cats = _build_categoricals(spec)
    encoder = constriction.stream.queue.RangeEncoder()
    for s, c in zip(sym.tolist(), ctx.tolist(), strict=True):
        encoder.encode(np.array([int(s)], dtype=np.int32), cats[int(c)])
    words = encoder.get_compressed()
    return np.asarray(words, dtype=">u4").tobytes()


def decode_adaptive_context_stream(
    payload: bytes,
    context_ids: np.ndarray,
    spec: AdaptiveContextSpec,
) -> np.ndarray:
    """Inverse of :func:`encode_adaptive_context_stream`.

    Parameters
    ----------
    payload:
        Big-endian uint32 stream bytes.
    context_ids:
        Same per-position context-id stream used at encode time.
    spec:
        Same :class:`AdaptiveContextSpec`.

    Returns
    -------
    np.ndarray
        ``int32`` array of length ``len(context_ids)`` carrying the decoded
        symbols.
    """
    if len(payload) % 4:
        raise ValueError(
            f"payload size {len(payload)} is not a multiple of 4"
        )
    ctx = np.asarray(context_ids, dtype=np.int64).reshape(-1)
    if ctx.size == 0:
        raise ValueError("context_ids must declare at least one symbol")
    if int(ctx.min()) < 0 or int(ctx.max()) >= spec.n_contexts:
        raise ValueError(
            f"context_ids out of range [0, {spec.n_contexts}); "
            f"min={int(ctx.min())} max={int(ctx.max())}"
        )
    cats = _build_categoricals(spec)
    words = np.frombuffer(payload, dtype=">u4").astype(np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    out = np.empty(ctx.size, dtype=np.int32)
    for i, c in enumerate(ctx.tolist()):
        out[i] = decoder.decode(cats[int(c)])
    return out


__all__ = [
    "AdaptiveContextSpec",
    "decode_adaptive_context_stream",
    "encode_adaptive_context_stream",
]
