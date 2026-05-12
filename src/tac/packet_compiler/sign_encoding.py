"""Unified 5-strategy sign-encoding taxonomy — PR96 / PR101 / PR103.

This module unifies the sign-encoding strategies used across PR96
(``rem2_HNeRV``), PR101 (``hnerv_ft_microcodec``), and PR103
(``hnerv_lc_ac``) into a single, typed, golden-vector-backed taxonomy.

Range coders (``constriction.stream.queue.RangeDecoder``, ANS, arithmetic
coders) operate on **non-negative integer symbols**. INT8 weight tensors
contain values in ``[-128, 127]``. To feed them to a range coder, we must
bijectively map INT8 -> UINT8. The choice of mapping is **non-arbitrary**:
different mappings reshape the symbol histogram, and the entropy coder's
compressed length is ``~ -sum p(s) log2 p(s)``. Choosing the wrong mapping
inflates the compressed bytes.

This module is the SUPERSET of ``pr101_decoder_byte_maps`` (the 4-strategy
PR101 polymorphic codec port). It adds the fifth strategy (``raw_uint8``)
used by PR103's lo/hi byte streams and PR105's latent uint8 streams.

The 5 strategies
================

1. ``negzig`` — negated-zigzag (negative-skew optimum). PR101 source
   ``hnerv_ft_microcodec/src/codec.py:233-234``.
2. ``zig`` — zigzag (positive-skew optimum). PR101 source
   ``hnerv_ft_microcodec/src/codec.py:225-227``.
3. ``twos`` — two's-complement raw reinterpret. PR101 source
   ``hnerv_ft_microcodec/src/codec.py:237-238``.
4. ``off`` — signed-byte-offset ``x + 128``. PR96 source
   ``rem2_HNeRV/inflate.py:90``; PR101 source
   ``hnerv_ft_microcodec/src/codec.py:235-236``; PR103 source
   ``hnerv_lc_ac/inflate.py:147``.
5. ``raw_uint8`` — already-unsigned passthrough. PR103 source
   ``hnerv_lc_ac/inflate.py:164-183`` (the lo/hi byte stream); PR105
   latent uint8 streams.

Design memo: ``.omx/research/sign_encoding_unified_taxonomy_20260512.md``.

Critical precondition for ``negzig``
====================================

``negzig`` is NOT a bijection over the full int8 range. The values
``-128`` and ``0`` both encode to byte ``0`` (because zigzag of ``-(-128)
mod 256`` collapses to zero under int8 wrap-around). PR101's training
script never produces ``-128`` in the int8 weight quantisation output,
which is why this is admissible at PR101's anchor operating point.

Downstream consumers using ``negzig`` MUST bound their post-quantisation
distribution to ``[-127, 127]``. This module enforces that contract via a
runtime guard in :func:`encode_sign` (it raises ``ValueError`` if any
``-128`` is present and the strategy is ``negzig``).

API contract
============

* ``encode_sign(tensor_int, strategy)`` -> ``bytes``. Bijective on
  ``raw_uint8`` (uint8 input) / ``zig``/``twos``/``off`` (int8 input);
  bijective on ``negzig`` only when input is bounded to ``[-127, 127]``.
* ``decode_sign(bytes_in, shape, dtype, strategy)`` -> ``np.ndarray``.
  Reconstructs the original tensor exactly from the encoded bytes.
* ``select_optimal_strategy(tensor_int)`` -> ``(strategy, shannon_entropy_bits)``.
  Entropy-based per-tensor selector — for each candidate strategy
  available given the input dtype/range, compute the Shannon entropy of
  the resulting UINT8 histogram and return the argmin strategy.

Composition
===========

Strategies are **mutually exclusive per tensor** (a single tensor uses
exactly one mapping). The choice is **per-tensor** (PR101's
``DECODER_BYTE_MAPS = {9: "negzig", 14: "negzig", 20: "twos", 27: "off"}``
explicitly assigns one strategy per state-dict index). Strategies CAN be
**composed across tensors**: each tensor independently selects its own
mapping, with the encoder storing the per-tensor selection table either
as a hardcoded source constant (PR101) or as an in-archive header.

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* OSS-friendly: public surface is the 5 names re-exported from
  ``tac.packet_compiler``.
* Pure functional transducers — no global mutable state.
* No archive bytes mutated by this module — it is byte-grammar plumbing
  only.
* ``score_claim=false``, ``promotion_eligible=false``,
  ``ready_for_exact_eval_dispatch=false`` per Catalog #100 (no naked
  bytes without inflate-consumption proof).

[empirical:src/tac/packet_compiler/golden_vectors/sign_encoding_*_v1.json]

target_substrate_hint
=====================

``any_packet_with_int_tensors`` — these are tensor-level codecs, more
portable than PR101 GOLD's HNeRV-LC-specific anchor tables. The
SPECIFIC per-tensor strategy table is substrate-dependent (must be
re-derived offline via :func:`select_optimal_strategy`); the MECHANISM
(5-way per-tensor dispatch) is reusable across substrates.

predicted_ev_per_byte_basis
===========================

``[predicted; PR96/PR101/PR103 multi-PR triangulation; not yet measured
on internal substrate]`` — sign-encoding-attributable bytes savings are
substrate-dependent. PR101's empirical per-tensor table saves O(50-150)
bytes/archive vs PR103's uniform ``off``; the actual savings on our
internal substrates is unknown until the
``tools/probe_sign_encoding_disambiguator.py`` probe runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np


# ── Strategy enum (string-typed for wire-format compatibility with PR101) ──


class SignEncodingStrategy(str, Enum):
    """The 5 sign-encoding strategies.

    Inherits from ``str`` so members compare equal to their wire-format
    string values (matching PR101 / PR103 source conventions and the
    pre-existing :data:`tac.packet_compiler.VALID_BYTE_MAP_STRATEGIES`
    frozenset).
    """

    NEGZIG = "negzig"
    ZIG = "zig"
    TWOS = "twos"
    OFF = "off"
    RAW_UINT8 = "raw_uint8"


VALID_SIGN_ENCODING_STRATEGIES: Final[frozenset[str]] = frozenset(
    {s.value for s in SignEncodingStrategy}
)
"""The 5 sign-encoding strategy wire-format names.

These match the design-memo taxonomy exactly. The set is frozen at
module-import time to prevent runtime mutation.
"""


# Strategies that require int8 input (negzig, zig, twos, off).
_INT8_INPUT_STRATEGIES: Final[frozenset[SignEncodingStrategy]] = frozenset(
    {
        SignEncodingStrategy.NEGZIG,
        SignEncodingStrategy.ZIG,
        SignEncodingStrategy.TWOS,
        SignEncodingStrategy.OFF,
    }
)


# ── Internal zigzag helpers (shared with pr101_decoder_byte_maps) ──────────


def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """Zigzag encode int8 -> uint8."""
    arr32 = arr_i8.astype(np.int32)
    enc = (arr32 << 1) ^ (arr32 >> 31)
    return (enc & 0xFF).astype(np.uint8)


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    """Zigzag decode uint8 -> int8."""
    arr32 = arr_u8.astype(np.int32)
    return np.where(arr32 % 2 == 0, arr32 // 2, -(arr32 // 2) - 1).astype(
        np.int8
    )


def _coerce_strategy(strategy: SignEncodingStrategy | str) -> SignEncodingStrategy:
    """Validate + coerce a strategy spec to the typed enum."""
    if isinstance(strategy, SignEncodingStrategy):
        return strategy
    if isinstance(strategy, str):
        if strategy not in VALID_SIGN_ENCODING_STRATEGIES:
            raise ValueError(
                f"unknown sign-encoding strategy: {strategy!r} "
                f"(valid: {sorted(VALID_SIGN_ENCODING_STRATEGIES)})"
            )
        return SignEncodingStrategy(strategy)
    raise TypeError(
        f"strategy must be SignEncodingStrategy or str; got {type(strategy)!r}"
    )


# ── Public API ──────────────────────────────────────────────────────────────


def encode_sign(
    tensor_int: np.ndarray,
    strategy: SignEncodingStrategy | str,
) -> bytes:
    """Bijectively encode an integer array to a UINT8 byte stream.

    The input must be a numpy array of:

    * ``dtype=int8`` for the strategies ``zig``, ``negzig``, ``twos``,
      ``off``.
    * ``dtype=uint8`` for the strategy ``raw_uint8``.

    The output is a UINT8 byte stream in C-order. The input shape is not
    preserved in the bytes (the decoder must be given the shape
    separately via :func:`decode_sign`).

    Parameters
    ----------
    tensor_int:
        N-dimensional numpy array. For all 4 signed strategies the dtype
        MUST be ``int8``. For ``raw_uint8`` the dtype MUST be ``uint8``.
    strategy:
        One of the 5 sign-encoding strategy names (or
        :class:`SignEncodingStrategy` enum value).

    Returns
    -------
    bytes
        The encoded UINT8 byte stream. Length is ``tensor_int.size``.

    Raises
    ------
    ValueError
        On unknown strategy, wrong input dtype, or on a ``negzig`` input
        that contains ``-128`` (which is not bijective under negzig).
    TypeError
        On non-array input.

    Notes
    -----
    The ``negzig`` -128 guard is a CLAUDE.md "Beauty, simplicity, and
    developer experience" non-negotiable: failures must surface at
    encode-time (so the bug is loud and adjacent to the offending tensor)
    rather than at decode-time (where the bug is silent and far from the
    cause).
    """
    if not isinstance(tensor_int, np.ndarray):
        raise TypeError(
            f"tensor_int must be np.ndarray; got {type(tensor_int)!r}"
        )
    strat = _coerce_strategy(strategy)

    if strat is SignEncodingStrategy.RAW_UINT8:
        if tensor_int.dtype != np.uint8:
            raise ValueError(
                f"raw_uint8 requires dtype=uint8; got {tensor_int.dtype}"
            )
        return np.ascontiguousarray(tensor_int).tobytes()

    # All 4 signed strategies require int8.
    if tensor_int.dtype != np.int8:
        raise ValueError(
            f"sign-encoding strategy {strat.value!r} requires dtype=int8; "
            f"got {tensor_int.dtype}"
        )

    arr = np.ascontiguousarray(tensor_int).reshape(-1)

    if strat is SignEncodingStrategy.ZIG:
        return _zigzag_encode_i8(arr).tobytes()

    if strat is SignEncodingStrategy.NEGZIG:
        # NEGZIG -128 guard (per design memo §1.2): negzig is NOT
        # bijective on the full int8 range because -(-128) wraps to
        # +128 in int8 arithmetic, collapsing to zigzag(0) under int8
        # wrap-around. We surface this at encode-time so downstream
        # consumers see the cause, not the effect.
        if np.any(arr == -128):
            n_violations = int(np.sum(arr == -128))
            raise ValueError(
                "negzig is not bijective over the full int8 range: "
                f"input contains {n_violations} occurrences of -128 "
                "which collide with 0 under zigzag(-x). NEGZIG requires "
                "the input distribution to be bounded to [-127, 127]. "
                "Per CLAUDE.md 'Beauty, simplicity, and developer "
                "experience' — surface this at encode-time, not silently "
                "corrupt decode-time output."
            )
        # Encode -x via zigzag (per PR101 source line 234).
        neg = (-arr.astype(np.int16)).astype(np.int32)
        enc = (neg << 1) ^ (neg >> 31)
        return (enc & 0xFF).astype(np.uint8).tobytes()

    if strat is SignEncodingStrategy.OFF:
        # Encode x + 128 (per PR101 source line 236).
        return ((arr.astype(np.int16) + 128) & 0xFF).astype(np.uint8).tobytes()

    # strat is SignEncodingStrategy.TWOS — raw int8 reinterpret as uint8
    # (per PR101 source line 238).
    return arr.view(np.uint8).tobytes()


def decode_sign(
    bytes_in: bytes,
    shape: tuple[int, ...] | Iterable[int],
    dtype: np.dtype | type | str,
    strategy: SignEncodingStrategy | str,
) -> np.ndarray:
    """Decode a UINT8 byte stream into an N-dimensional integer array.

    Parameters
    ----------
    bytes_in:
        UINT8 byte stream produced by :func:`encode_sign`.
    shape:
        Target tensor shape (the encoder erases this; the decoder caller
        must provide it).
    dtype:
        Target tensor dtype. MUST be ``int8`` for the 4 signed strategies
        and ``uint8`` for ``raw_uint8``.
    strategy:
        Must match the encoder strategy.

    Returns
    -------
    np.ndarray
        Decoded N-dimensional array of the given shape + dtype.

    Raises
    ------
    ValueError
        On dtype mismatch, unknown strategy, or byte-count vs shape
        mismatch.
    TypeError
        On non-bytes input.
    """
    if not isinstance(bytes_in, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"bytes_in must be bytes-like; got {type(bytes_in)!r}"
        )
    strat = _coerce_strategy(strategy)
    shape_t = tuple(int(d) for d in shape)
    expected_n = 1
    for d in shape_t:
        if d < 0:
            raise ValueError(f"shape contains negative dim: {shape_t}")
        expected_n *= d
    actual_n = len(bytes_in)
    if actual_n != expected_n:
        raise ValueError(
            f"byte count {actual_n} does not match shape {shape_t} "
            f"(expected {expected_n})"
        )

    target_dtype = np.dtype(dtype)
    if strat is SignEncodingStrategy.RAW_UINT8:
        if target_dtype != np.uint8:
            raise ValueError(
                f"raw_uint8 requires target dtype=uint8; got {target_dtype}"
            )
        return np.frombuffer(bytes(bytes_in), dtype=np.uint8).reshape(shape_t).copy()

    if target_dtype != np.int8:
        raise ValueError(
            f"sign-encoding strategy {strat.value!r} requires target "
            f"dtype=int8; got {target_dtype}"
        )

    arr_u8 = np.frombuffer(bytes(bytes_in), dtype=np.uint8)

    if strat is SignEncodingStrategy.ZIG:
        decoded = _zigzag_decode_u8(arr_u8)
    elif strat is SignEncodingStrategy.NEGZIG:
        # PR101 source line 234: (-zigzag_decode_u8(arr).astype(int16)).astype(int8)
        decoded = (-_zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    elif strat is SignEncodingStrategy.OFF:
        # PR101 source line 236.
        decoded = (arr_u8.astype(np.int16) - 128).astype(np.int8)
    else:  # TWOS
        # PR101 source line 238.
        decoded = arr_u8.view(np.int8).copy()

    return decoded.reshape(shape_t)


@dataclass(frozen=True)
class StrategySelection:
    """Per-tensor sign-encoding strategy selection result.

    Attributes
    ----------
    strategy:
        The selected strategy (Shannon-entropy-minimal under the input
        histogram).
    entropy_bits:
        Shannon entropy of the encoded UINT8 histogram (lower is better;
        proportional to the entropy coder's lower-bound compressed
        length).
    per_strategy_entropy_bits:
        Dict mapping strategy -> shannon entropy for every candidate
        evaluated. The minimum value matches ``entropy_bits``.
    """

    strategy: SignEncodingStrategy
    entropy_bits: float
    per_strategy_entropy_bits: dict[SignEncodingStrategy, float]


def _shannon_entropy_bits(arr_u8: np.ndarray) -> float:
    """Shannon entropy of a UINT8 histogram in bits per symbol.

    Returns 0.0 for empty inputs (vacuous entropy convention).
    """
    n = arr_u8.size
    if n == 0:
        return 0.0
    counts = np.bincount(arr_u8.astype(np.int64), minlength=256)
    nonzero = counts > 0
    p = counts[nonzero].astype(np.float64) / float(n)
    return float(-np.sum(p * np.log2(p)))


def select_optimal_strategy(
    tensor_int: np.ndarray,
    *,
    candidates: Iterable[SignEncodingStrategy | str] | None = None,
) -> StrategySelection:
    """Pick the entropy-minimal sign-encoding strategy for ``tensor_int``.

    For each candidate strategy compatible with ``tensor_int.dtype``,
    encode + compute the Shannon entropy of the resulting UINT8
    histogram, then return the argmin strategy.

    This is the offline per-tensor entropy-comparison that PR101's
    encoder uses to derive its ``DECODER_BYTE_MAPS = {9: "negzig", 14:
    "negzig", 20: "twos", 27: "off"}`` constant.

    Parameters
    ----------
    tensor_int:
        Input array. If ``dtype=int8`` the candidate set defaults to
        ``{negzig, zig, twos, off}``; if ``dtype=uint8`` the candidate
        set defaults to ``{raw_uint8}``.
    candidates:
        Optional override for the candidate set. Strategies that require
        a dtype other than ``tensor_int.dtype`` are silently skipped.

    Returns
    -------
    StrategySelection
        The entropy-minimal strategy + the per-strategy entropy map.

    Raises
    ------
    ValueError
        On unsupported dtype, empty candidate set after filtering, or
        zero-length input.
    """
    if not isinstance(tensor_int, np.ndarray):
        raise TypeError(
            f"tensor_int must be np.ndarray; got {type(tensor_int)!r}"
        )
    if tensor_int.size == 0:
        raise ValueError("tensor_int must be non-empty")

    if candidates is None:
        if tensor_int.dtype == np.int8:
            candidates_iter: Iterable[SignEncodingStrategy | str] = (
                SignEncodingStrategy.NEGZIG,
                SignEncodingStrategy.ZIG,
                SignEncodingStrategy.TWOS,
                SignEncodingStrategy.OFF,
            )
        elif tensor_int.dtype == np.uint8:
            candidates_iter = (SignEncodingStrategy.RAW_UINT8,)
        else:
            raise ValueError(
                f"tensor_int dtype must be int8 or uint8; got {tensor_int.dtype}"
            )
    else:
        candidates_iter = candidates

    per_strategy: dict[SignEncodingStrategy, float] = {}
    flat = np.ascontiguousarray(tensor_int).reshape(-1)
    for cand in candidates_iter:
        strat = _coerce_strategy(cand)
        # Skip strategies whose dtype doesn't match the input.
        if strat is SignEncodingStrategy.RAW_UINT8 and flat.dtype != np.uint8:
            continue
        if strat in _INT8_INPUT_STRATEGIES and flat.dtype != np.int8:
            continue
        # Skip negzig if -128 is present (encode would raise).
        if strat is SignEncodingStrategy.NEGZIG and np.any(flat == -128):
            continue
        encoded = encode_sign(flat, strat)
        arr_u8 = np.frombuffer(encoded, dtype=np.uint8)
        per_strategy[strat] = _shannon_entropy_bits(arr_u8)

    if not per_strategy:
        raise ValueError(
            "no compatible candidate strategies for input "
            f"dtype={tensor_int.dtype}"
        )

    # Pick the entropy-minimal strategy. Ties broken by a deterministic
    # tiebreaker (declared enum order) so the result is reproducible
    # across runs.
    enum_order = list(SignEncodingStrategy)

    def _tiebreak(item: tuple[SignEncodingStrategy, float]) -> tuple[float, int]:
        return (item[1], enum_order.index(item[0]))

    winner, winner_bits = min(per_strategy.items(), key=_tiebreak)
    return StrategySelection(
        strategy=winner,
        entropy_bits=winner_bits,
        per_strategy_entropy_bits=dict(per_strategy),
    )


__all__ = [
    "SignEncodingStrategy",
    "StrategySelection",
    "VALID_SIGN_ENCODING_STRATEGIES",
    "decode_sign",
    "encode_sign",
    "select_optimal_strategy",
]
