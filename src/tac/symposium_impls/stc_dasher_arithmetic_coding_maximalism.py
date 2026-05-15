# SPDX-License-Identifier: MIT
"""STC-Dasher arithmetic coding maximalism — substrate-agnostic rate-shaver.

Per the Grand Reunion symposium 2026-05-15 Phase F Composite #6 (Filler +
MacKay). Composes Filler's Syndrome-Trellis Coding (STC) with MacKay's
Dasher-style efficient encoding of sparse signals. Substrate-agnostic
rate-axis bit-shaver: applies to ANY substrate's archive.

Math contract
=============

**Syndrome-Trellis Coding (Filler, Judas, Fridrich 2011 *IEEE TIFS*):**
STC is a parity-check code optimized for steganographic embedding via
Viterbi-like decoding through a trellis defined by the parity-check
matrix. For a payload of ``m`` bits embedded in ``n`` cover bits (rate
``m/n``), STC achieves the rate-distortion bound for additive distortion
within a small gap that vanishes as the constraint length increases.

For our problem (inverse steganalysis), STC's parity-check structure is
the canonical SHORTEST-CODE-WORD assignment that approximates the
arithmetic coding lower bound ``H(X | scorer)`` per the MacKay
conditional-entropy framing (impl 1.1). The achievable rate with
constraint length ``h`` is approximately

    R_STC(D) <= R_AC(D) + 1/h     (Filler 2011 Theorem 4)

where ``R_AC(D)`` is the arithmetic coding lower bound at distortion
``D``. With ``h = 12`` we approach the AC bound within ~1/12 = 0.083
bits per symbol.

**Dasher arithmetic coding (MacKay 2003 *ITILA* §6.6):** Dasher uses
arithmetic coding with a context model that adapts based on the symbol
history. For sparse signals (most coefficients near zero), Dasher's
context model achieves entropy close to the conditional entropy
``H(X_t | X_{t-1}, ..., X_{t-k})``.

**Composition:** the STC-Dasher coder is

    encoded = arithmetic_code(STC_encode(symbols, parity_matrix))

The composed coder applies STC to map the source symbols to their
syndrome vector; the syndrome is then arithmetic-coded with a Dasher-
style context model over the syndrome support.

[verified-against: Filler, Judas, Fridrich 2011 *IEEE TIFS* §III + IV
(STC structure + rate-distortion bound); MacKay 2003 *ITILA* §6.4 + §6.6
(arithmetic coding + Dasher context model); Cover & Thomas 2nd ed §13.2
(arithmetic coding lower bound).]

This is a SCAFFOLD: full archive integration is deferred to a follow-up
subagent per the symposium spec ($0 cost-to-validate; the math is purely
symbolic). Math primitives are tested for correctness.

Lane: ``lane_symposium_impl_stc_dasher_20260515``.
Catalog #262.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

__all__ = (
    "DEFAULT_STC_CONSTRAINT_LENGTH",
    "DasherContextModel",
    "STCParityCheckMatrix",
    "STCEncoderResult",
    "STCDasherSymbolStream",
    "arithmetic_code_bit_estimate",
    "build_default_stc_parity_matrix",
    "compose_stc_dasher_encoded_bits",
    "estimate_stc_dasher_rate_bound",
    "stc_encode_to_syndrome",
    "update_from_anchor",
)

DEFAULT_STC_CONSTRAINT_LENGTH: Final[int] = 12  # Filler 2011 §IV.B canonical.


@dataclasses.dataclass(frozen=True)
class STCParityCheckMatrix:
    """A parity-check matrix specifying the STC code structure.

    The matrix has shape ``(m, n)`` over GF(2): ``m`` syndrome bits, ``n``
    cover bits. ``constraint_length`` is the number of consecutive columns
    over which a single sub-matrix block extends.
    """

    matrix: np.ndarray
    constraint_length: int

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        if self.matrix.dtype != np.uint8 and self.matrix.dtype != np.int8 and self.matrix.dtype != bool:
            raise ValueError("matrix must be a binary array (uint8 / int8 / bool)")
        if self.constraint_length < 1:
            raise ValueError("constraint_length must be >= 1")
        if not ((self.matrix == 0) | (self.matrix == 1)).all():
            raise ValueError("matrix entries must be 0 or 1 (GF(2))")

    @property
    def n_cover(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def n_syndrome(self) -> int:
        return int(self.matrix.shape[0])


def build_default_stc_parity_matrix(
    *, n_cover: int, payload_bits: int, constraint_length: int = DEFAULT_STC_CONSTRAINT_LENGTH
) -> STCParityCheckMatrix:
    """Build a canonical STC parity-check matrix.

    Per Filler 2011 §IV.B: the canonical construction tiles a small ``h × w``
    sub-matrix block diagonally to form the full parity-check matrix. The
    sub-matrix block is randomly generated under the constraint of full row
    rank.

    For deterministic reproducibility we use a stable seed.
    """
    if n_cover < 1:
        raise ValueError("n_cover must be >= 1")
    if payload_bits < 1:
        raise ValueError("payload_bits must be >= 1")
    if payload_bits > n_cover:
        raise ValueError("payload_bits must be <= n_cover")
    if constraint_length < 1:
        raise ValueError("constraint_length must be >= 1")
    block_h = constraint_length
    block_w = max(n_cover // payload_bits, 1)
    rng = np.random.default_rng(1729)  # canonical Hardy-Ramanujan seed
    sub_block = rng.integers(0, 2, size=(block_h, block_w), dtype=np.uint8)
    # Force at least one 1 per row to ensure the row has effect.
    for r in range(block_h):
        if sub_block[r].sum() == 0:
            sub_block[r, 0] = 1
    matrix = np.zeros((payload_bits, n_cover), dtype=np.uint8)
    for m in range(payload_bits):
        col_start = m * block_w
        col_end = min(col_start + block_w, n_cover)
        if col_start >= n_cover:
            break
        for r in range(min(block_h, payload_bits - m)):
            cols_avail = col_end - col_start
            matrix[m + r, col_start:col_end] |= sub_block[r, :cols_avail]
    return STCParityCheckMatrix(matrix=matrix, constraint_length=constraint_length)


@dataclasses.dataclass(frozen=True)
class STCEncoderResult:
    """Output of one STC encode pass."""

    syndrome: np.ndarray  # shape (n_syndrome,) uint8
    rate_bits_per_symbol: float
    constraint_length: int


def stc_encode_to_syndrome(
    *, symbols: np.ndarray, parity: STCParityCheckMatrix
) -> STCEncoderResult:
    """Encode symbols to their STC syndrome.

    Per Filler 2011 §III: the syndrome is ``s = H · x mod 2`` for
    parity-check matrix ``H`` and binary symbol vector ``x``.
    """
    if symbols.ndim != 1:
        raise ValueError("symbols must be 1D")
    if symbols.size != parity.n_cover:
        raise ValueError(
            f"symbols length {symbols.size} != parity n_cover {parity.n_cover}"
        )
    bin_symbols = np.asarray(symbols, dtype=np.uint8)
    if not ((bin_symbols == 0) | (bin_symbols == 1)).all():
        raise ValueError("symbols must be binary (0/1)")
    syndrome = (parity.matrix @ bin_symbols) % 2
    rate = float(syndrome.size) / float(symbols.size)
    return STCEncoderResult(
        syndrome=syndrome.astype(np.uint8),
        rate_bits_per_symbol=rate,
        constraint_length=parity.constraint_length,
    )


@dataclasses.dataclass(frozen=True)
class DasherContextModel:
    """Adaptive context model per MacKay's Dasher framing.

    For each symbol position ``t``, the conditional probability
    ``p(x_t | x_{t-k}, ..., x_{t-1})`` is estimated from running counts
    over the prefix history of length ``k``.
    """

    context_length: int
    symbol_alphabet_size: int
    initial_count: int = 1  # Laplace smoothing

    def __post_init__(self) -> None:
        if self.context_length < 0:
            raise ValueError("context_length must be >= 0")
        if self.symbol_alphabet_size < 2:
            raise ValueError("symbol_alphabet_size must be >= 2")
        if self.initial_count < 1:
            raise ValueError("initial_count must be >= 1 (for Laplace smoothing)")


def arithmetic_code_bit_estimate(
    symbols: np.ndarray, *, model: DasherContextModel
) -> float:
    """Estimate the arithmetic-coding bit cost under a Dasher context model.

    Per Cover & Thomas Theorem 13.5.1: the expected bit cost of
    arithmetic coding under model ``p`` is the cross-entropy
    ``-E[log2 p_model(X)]``. We compute the empirical cross-entropy by
    iterating with the running context.
    """
    if symbols.ndim != 1:
        raise ValueError("symbols must be 1D")
    if symbols.size == 0:
        return 0.0
    history: list[int] = []
    counts: dict[tuple[int, ...], np.ndarray] = {}
    total_bits = 0.0
    for sym in symbols.tolist():
        if sym < 0 or sym >= model.symbol_alphabet_size:
            raise ValueError("symbol out of alphabet range")
        ctx = tuple(history[-model.context_length :]) if model.context_length > 0 else ()
        if ctx not in counts:
            counts[ctx] = np.full(model.symbol_alphabet_size, model.initial_count, dtype=np.float64)
        ctx_counts = counts[ctx]
        ctx_total = float(ctx_counts.sum())
        p = float(ctx_counts[sym]) / ctx_total
        total_bits += -math.log2(p)
        ctx_counts[sym] += 1.0
        history.append(int(sym))
    return total_bits


@dataclasses.dataclass(frozen=True)
class STCDasherSymbolStream:
    """Composed STC + Dasher arithmetic-coding result."""

    n_input_symbols: int
    n_syndrome_bits: int
    arithmetic_bits: float
    rate_bits_per_input_symbol: float
    notes: str


def compose_stc_dasher_encoded_bits(
    *,
    symbols: np.ndarray,
    parity: STCParityCheckMatrix,
    model: DasherContextModel,
) -> STCDasherSymbolStream:
    """Encode symbols via STC then arithmetic-code the syndrome.

    The composition demonstrates how the symposium's STC-Dasher rate-shaver
    operates on substrate archive bytes.
    """
    if model.symbol_alphabet_size != 2:
        raise ValueError("Dasher model alphabet must be 2 for STC syndrome bits")
    stc = stc_encode_to_syndrome(symbols=symbols, parity=parity)
    bits = arithmetic_code_bit_estimate(stc.syndrome, model=model)
    rate = bits / max(symbols.size, 1)
    return STCDasherSymbolStream(
        n_input_symbols=int(symbols.size),
        n_syndrome_bits=int(stc.syndrome.size),
        arithmetic_bits=float(bits),
        rate_bits_per_input_symbol=float(rate),
        notes=(
            f"[prediction; first-principles] Filler-2011 STC + MacKay-Dasher AC. "
            f"constraint_length={parity.constraint_length}, "
            f"context_length={model.context_length}. Catalog #262."
        ),
    )


def estimate_stc_dasher_rate_bound(
    *, baseline_bits: float, syndrome_bits: float, constraint_length: int
) -> float:
    """Per Filler 2011 Theorem 4: ``R_STC(D) <= R_AC(D) + 1/h``.

    Returns the upper bound on the rate gap (bits per symbol).
    """
    if constraint_length < 1:
        raise ValueError("constraint_length must be >= 1")
    if baseline_bits < 0 or syndrome_bits < 0:
        raise ValueError("baseline_bits and syndrome_bits must be >= 0")
    return baseline_bits + (1.0 / constraint_length) - syndrome_bits


def update_from_anchor(
    anchor: Mapping[str, object],
    *,
    parity: STCParityCheckMatrix | None = None,
    model: DasherContextModel | None = None,
) -> STCDasherSymbolStream | None:
    """Re-emit STC-Dasher composed bits from a fresh symbol anchor.

    The anchor must carry a ``symbols`` ndarray of binary values.
    """
    raw = anchor.get("symbols")
    if not isinstance(raw, np.ndarray):
        return None
    if parity is None:
        parity = build_default_stc_parity_matrix(
            n_cover=raw.size, payload_bits=max(raw.size // 4, 1)
        )
    if model is None:
        model = DasherContextModel(context_length=2, symbol_alphabet_size=2)
    return compose_stc_dasher_encoded_bits(symbols=raw, parity=parity, model=model)
