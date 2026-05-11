"""PR91 ``hpac_coder_hybrid`` grammar primitives — universal AC wrapper +
QM0/QH0 weight-block grammar.

Two primitives land here, extracted from PR91:

1. **Universal constriction arithmetic coder wrapper**
   (:func:`encode_categorical_stream` / :func:`decode_categorical_stream`)

   PR91's ``pr86_hpac.py`` uses ``constriction.stream.queue.{RangeEncoder,
   RangeDecoder}`` to range-code a categorical token stream. The HPACMini
   NN-context-model side is research-only (it ships with the archive), but
   the AC wrapper itself is a thin, universally-applicable primitive:
   given a per-symbol probability matrix ``(n_symbols, alphabet)`` and a
   parallel symbol array of length ``n_symbols``, encode and decode using
   one categorical per symbol.

   This generalises PR103's :func:`encode_merged_range_stream` (which uses
   ONE histogram per tensor) and PR103's :func:`encode_latent_hi_arithmetic`
   (one histogram for the whole stream) to the per-symbol case where each
   position has its own distribution — e.g. an HPACMini-style sequential
   context model OR a learned per-pair latent decoder OR anything that
   emits a per-symbol cdf.

2. **PR91 QM0 / QH0 magic grammar** (:class:`QMQHHeader` +
   :func:`parse_qmqh_header` / :func:`emit_qmqh_header`)

   PR91 frames a model payload with a 3-byte magic ``b"QM0"`` or
   ``b"QH0"``. The choice signals whether the hi-lo byte split is used for
   each FP4-quantised weight block; the rest of the body is opaque to this
   primitive. We surface the magic constants and the header
   encode/decode helpers so any caller wiring a Quantizr-family inflate can
   reuse them.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr91_intake_20260505_auto/source/submissions/hpac_coder_hybrid/{inflate,pr86_hpac}.py``

CLAUDE.md compliance
====================

* No scorer load — pure numpy + constriction.
* No MPS / torch import (the HPACMini NN context model is research-only).
* No ``/tmp`` paths.
* Frozen dataclass; ``encode → decode`` is bit-exact on the
  ``pr91_arithmetic_coder_constriction_v1`` and ``pr91_qmqh_grammar_v1``
  golden vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import constriction
import numpy as np

#: 3-byte magic for plain QM0 (single-byte FP4 packed) layout.
MAGIC_QM0: bytes = b"QM0"
#: 3-byte magic for QH0 (hi-lo split FP4 packed) layout.
MAGIC_QH0: bytes = b"QH0"


# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class QMQHHeader:
    """QM0 / QH0 magic-prefix header.

    Attributes
    ----------
    magic:
        Either :data:`MAGIC_QM0` or :data:`MAGIC_QH0`.
    hilo_split:
        ``True`` iff ``magic == MAGIC_QH0`` (convenience boolean).
    body_offset:
        Byte offset at which the per-tensor body begins (= ``len(magic)``).
    """

    magic: bytes
    hilo_split: bool
    body_offset: int


# ── Universal constriction arithmetic coder wrapper ─────────────────────────


def _normalise_probs(probs: np.ndarray, *, floor: float = 1e-10) -> np.ndarray:
    """Floor + re-normalise rows so each row sums to 1 (PR86 convention)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(
            f"probs must be 2D (n_symbols, alphabet); got shape {p.shape}"
        )
    if floor <= 0:
        raise ValueError(f"floor must be > 0; got {floor}")
    p = np.maximum(p, floor)
    p /= p.sum(axis=1, keepdims=True)
    return p


def encode_categorical_stream(
    symbols: np.ndarray,
    probs: np.ndarray,
) -> bytes:
    """Range-encode a symbol stream against a per-symbol categorical model.

    Parameters
    ----------
    symbols:
        1D ``int32``/``int64``-castable array of length ``n_symbols``. Each
        entry must satisfy ``0 <= symbols[i] < probs.shape[1]``.
    probs:
        Probability matrix of shape ``(n_symbols, alphabet)``. Negative or
        zero entries are floored at ``1e-10`` then re-normalised per row,
        matching the PR86 ``decompress_tokens_hpac`` convention.

    Returns
    -------
    bytes
        Big-endian byte serialisation of the constriction uint32 word
        stream. The wire format matches PR103's
        :func:`tac.packet_compiler.encode_merged_range_stream`.
    """
    sym = np.asarray(symbols, dtype=np.int64).reshape(-1)
    p = _normalise_probs(probs)
    n_symbols, alphabet = p.shape
    if sym.size != n_symbols:
        raise ValueError(
            f"symbol count {sym.size} != probs.shape[0] {n_symbols}"
        )
    if n_symbols == 0:
        raise ValueError("must encode at least one symbol")
    if int(sym.min()) < 0 or int(sym.max()) >= alphabet:
        raise ValueError(
            f"symbols out of range [0, {alphabet}); "
            f"min={int(sym.min())} max={int(sym.max())}"
        )
    encoder = constriction.stream.queue.RangeEncoder()
    for i in range(n_symbols):
        cat = constriction.stream.model.Categorical(
            probabilities=p[i], perfect=False
        )
        encoder.encode(np.array([int(sym[i])], dtype=np.int32), cat)
    words = encoder.get_compressed()
    return np.asarray(words, dtype=">u4").tobytes()


def decode_categorical_stream(
    payload: bytes,
    probs: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`encode_categorical_stream`.

    Parameters
    ----------
    payload:
        Big-endian bytes of the constriction uint32 stream.
    probs:
        Same probability matrix used at encode time. The decoder rebuilds
        per-symbol categoricals deterministically; the caller MUST supply
        the same matrix.

    Returns
    -------
    np.ndarray
        ``int32`` array of length ``probs.shape[0]``.
    """
    if len(payload) % 4:
        raise ValueError(
            f"payload size {len(payload)} is not a multiple of 4"
        )
    p = _normalise_probs(probs)
    n_symbols, _alphabet = p.shape
    if n_symbols == 0:
        raise ValueError("probs must declare at least one symbol")
    words = np.frombuffer(payload, dtype=">u4").astype(np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    out = np.empty(n_symbols, dtype=np.int32)
    for i in range(n_symbols):
        cat = constriction.stream.model.Categorical(
            probabilities=p[i], perfect=False
        )
        out[i] = decoder.decode(cat)
    return out


# ── QM0 / QH0 magic grammar ─────────────────────────────────────────────────


def emit_qmqh_header(*, hilo_split: bool) -> bytes:
    """Emit a 3-byte QMQH header (the magic alone — no body)."""
    return MAGIC_QH0 if hilo_split else MAGIC_QM0


def parse_qmqh_header(payload: bytes) -> QMQHHeader:
    """Parse the leading 3-byte QM0 / QH0 magic into a typed header.

    Raises
    ------
    ValueError
        If ``payload`` does not begin with ``b"QM0"`` or ``b"QH0"``.
    """
    if len(payload) < 3:
        raise ValueError(
            f"QMQH payload too short ({len(payload)} bytes); need at least 3"
        )
    magic = bytes(payload[:3])
    if magic == MAGIC_QM0:
        return QMQHHeader(magic=MAGIC_QM0, hilo_split=False, body_offset=3)
    if magic == MAGIC_QH0:
        return QMQHHeader(magic=MAGIC_QH0, hilo_split=True, body_offset=3)
    raise ValueError(
        f"unknown QMQH magic {magic!r}; expected {MAGIC_QM0!r} or {MAGIC_QH0!r}"
    )


def pack_hi_lo_split(packed_nibbles: bytes) -> bytes:
    """Split a hi-nibble-low-nibble packed byte stream into two byte streams.

    PR91's ``QH0`` layout stores the hi-nibbles of every byte first (run-
    length-friendly under Brotli), then the lo-nibbles. This is a pure byte
    permutation; the inverse is :func:`unpack_hi_lo_split`.

    Parameters
    ----------
    packed_nibbles:
        Bytes where each byte holds two nibbles ``hi << 4 | lo``.

    Returns
    -------
    bytes
        Concatenation ``[hi_byte for hi_byte in packed] + [lo_byte for lo_byte
        in packed]`` where each ``hi_byte``/``lo_byte`` packs the next two
        nibbles of its kind.
    """
    flat = np.frombuffer(packed_nibbles, dtype=np.uint8)
    if flat.size == 0:
        return b""
    if flat.size & 1:
        raise ValueError(
            f"hi-lo split requires even byte count; got {flat.size}"
        )
    hi_nibbles = (flat >> 4) & 0xF
    lo_nibbles = flat & 0xF
    hi_packed = ((hi_nibbles[0::2] << 4) | hi_nibbles[1::2]).astype(np.uint8)
    lo_packed = ((lo_nibbles[0::2] << 4) | lo_nibbles[1::2]).astype(np.uint8)
    return hi_packed.tobytes() + lo_packed.tobytes()


def unpack_hi_lo_split(split_payload: bytes) -> bytes:
    """Inverse of :func:`pack_hi_lo_split`.

    Parameters
    ----------
    split_payload:
        Bytes produced by :func:`pack_hi_lo_split`. Must have even length
        (the two halves are equal-length).

    Returns
    -------
    bytes
        Re-interleaved hi/lo nibble packed bytes.
    """
    flat = np.frombuffer(split_payload, dtype=np.uint8)
    if flat.size == 0:
        return b""
    if flat.size & 1:
        raise ValueError(
            f"split payload must have even length; got {flat.size}"
        )
    half = flat.size // 2
    hi_packed = flat[:half]
    lo_packed = flat[half:]
    hi_nibbles = np.empty(half * 2, dtype=np.uint8)
    lo_nibbles = np.empty(half * 2, dtype=np.uint8)
    hi_nibbles[0::2] = (hi_packed >> 4) & 0xF
    hi_nibbles[1::2] = hi_packed & 0xF
    lo_nibbles[0::2] = (lo_packed >> 4) & 0xF
    lo_nibbles[1::2] = lo_packed & 0xF
    interleaved = ((hi_nibbles << 4) | lo_nibbles).astype(np.uint8)
    return interleaved.tobytes()


__all__ = [
    "MAGIC_QH0",
    "MAGIC_QM0",
    "QMQHHeader",
    "decode_categorical_stream",
    "emit_qmqh_header",
    "encode_categorical_stream",
    "pack_hi_lo_split",
    "parse_qmqh_header",
    "unpack_hi_lo_split",
]
