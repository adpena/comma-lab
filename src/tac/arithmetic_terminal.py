"""PARADIGM-γ Module γ — Arithmetic terminal layer (canonical).

The "arithmetic terminal" is the FINAL layer in the canonical codec
composition order:

    representation → prediction → quantization → hyperprior → arithmetic → pack
                                                              ─────────
                                                              this module

It takes the OUTPUT of the hyperprior layer (a per-block σ prediction) and
arithmetic-codes the integer stream under that conditional Gaussian prior.
This is the Ballé 2018 §3.2 wire-format step: encode each y_i under
``p(y|σ_b) = ∫_{y-0.5}^{y+0.5} N(0, σ_b²) dy``.

Two arithmetic-terminal modes are supported (matching ``arithmetic_qint_codec``):
- **STATIC prior**: histogram-based frequency table (the Lane SH baseline).
- **HYPERPRIOR (BALLE)**: per-block discretized Gaussian under an external
  σ-predictor (the Ballé hyperprior decoder).

This module is a thin VIEW over the existing primitives — it consolidates
the three "arithmetic encode" callsites in the repo so the canonical
"arithmetic terminal" interface is testable in isolation:

- ``encode_static(qints, num_symbols, offset)`` — wraps
  ``arithmetic_qint_codec.encode_qints_arithmetic``.
- ``encode_hyperprior(qints, num_symbols, offset, sigma_per_block,
  block_size)`` — direct PMF construction from a precomputed σ array
  (skips the hyper-encoder/-decoder neural step that ``balle_hyperprior_codec``
  uses; useful when the orchestrator already has σ in hand).

Wire format
-----------
- STATIC mode: AQv1 (re-exported from arithmetic_qint_codec).
- HYPERPRIOR mode: ATBHv1 — a SLIM wire format that omits the hyper-decoder
  weights (the orchestrator transmits σ via Ballé's BHv1 path; this terminal
  only carries the y-stream payload).

::

    magic              : 4 bytes  = b"ATBH"  (Arithmetic Terminal under Ballé Hyperprior)
    version            : 2 bytes  uint16  = 1
    num_symbols        : 2 bytes  uint16
    offset             : 4 bytes  int32
    n_total            : 8 bytes  uint64
    block_size         : 4 bytes  uint32
    n_blocks           : 4 bytes  uint32
    sigma_q8_per_block : n_blocks bytes uint8 (σ × 8, clamped [SIGMA_MIN, SIGMA_MAX])
    payload_len        : 8 bytes  uint64
    payload            : payload_len bytes

The ``sigma_q8_per_block`` is the side-info: 1 byte/block for σ. This is
the lossy quantization of the σ-prior to integer; the encoder and decoder
both round to the same integer step so the discretized-Gaussian PMFs match
bit-deterministically.

CLAUDE.md compliance
--------------------
* COMPRESS-time + INFLATE-time. Inflate path uses no scorer load.
* Public encoders verify decoder roundtrip on every call.
* Bit-deterministic on all platforms (CPython int math).
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np

from tac.arithmetic_qint_codec import (
    _ArithmeticDecoder,
    _ArithmeticEncoder,
    _cumulative_table,
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)
from tac.balle_hyperprior_codec import (
    _SIGMA_MAX,
    _SIGMA_MIN,
    _pmf_to_int_freq,
    discretized_gaussian_pmf,
)


_ATBH_MAGIC: bytes = b"ATBH"
_ATBH_VERSION: int = 1
_ATBH_FIXED_HEADER_SIZE: int = 4 + 2 + 2 + 4 + 8 + 4 + 4

# σ-step for the q8 quantization grid. The σ range is [SIGMA_MIN, SIGMA_MAX]
# = [0.05, 32]; with 256 levels the step is (32 - 0.05) / 255 ≈ 0.1253 per
# level. q8 = round((σ - SIGMA_MIN) / step).
_SIGMA_Q8_STEP: float = (_SIGMA_MAX - _SIGMA_MIN) / 255.0


def _sigma_to_q8(sigma: float) -> int:
    """Quantize σ to a uint8 ∈ [0, 255]."""
    sigma_clipped = float(np.clip(sigma, _SIGMA_MIN, _SIGMA_MAX))
    q = int(round((sigma_clipped - _SIGMA_MIN) / _SIGMA_Q8_STEP))
    return max(0, min(255, q))


def _q8_to_sigma(q: int) -> float:
    """Inverse of ``_sigma_to_q8``."""
    return float(_SIGMA_MIN + int(q) * _SIGMA_Q8_STEP)


# ────────────────────────────────────────────────────────────────────────────
# STATIC mode — wraps the existing AQv1 codec
# ────────────────────────────────────────────────────────────────────────────


def encode_static(
    *,
    qints: np.ndarray,
    num_symbols: int,
    offset: int,
) -> bytes:
    """STATIC arithmetic terminal: histogram-based frequency table.

    Equivalent to ``encode_qints_arithmetic`` but with required-keyword args
    matching the rest of the arithmetic_terminal interface.
    """
    if qints is None:
        raise ValueError("encode_static: qints is required")
    if num_symbols is None or num_symbols < 2:
        raise ValueError(
            f"encode_static: num_symbols must be >= 2, got {num_symbols}"
        )
    if offset is None:
        raise ValueError("encode_static: offset is required")
    blob = encode_qints_arithmetic(qints, num_symbols=num_symbols, offset=offset)
    decoded = decode_static(blob=blob, expected_dtype=np.ascontiguousarray(qints).dtype)
    if not np.array_equal(decoded, np.ascontiguousarray(qints).ravel()):
        raise ValueError("encode_static: decoder roundtrip verification failed")
    return blob


def decode_static(
    *,
    blob: bytes,
    expected_dtype: np.dtype = np.int8,
) -> np.ndarray:
    """STATIC arithmetic terminal decoder."""
    return decode_qints_arithmetic(blob, expected_dtype=expected_dtype)


# ────────────────────────────────────────────────────────────────────────────
# HYPERPRIOR mode — encode under a precomputed σ-per-block prior
# ────────────────────────────────────────────────────────────────────────────


def encode_hyperprior(
    *,
    qints: np.ndarray,
    num_symbols: int,
    offset: int,
    sigma_per_block: np.ndarray,
    block_size: int,
) -> bytes:
    """Arithmetic-code ``qints`` under a per-block σ prior.

    The σ stream is the OUTPUT of a hyperprior model (e.g., Ballé's
    ``HyperDecoder`` applied to the quantized z latent). This function does
    NOT run the hyperprior — it consumes the precomputed σ array.

    Args:
        qints: 1-D integer array. Required keyword.
        num_symbols: alphabet size. Required keyword.
        offset: integer added before symbol indexing. Required keyword.
        sigma_per_block: ``(n_blocks,)`` float array, σ ∈ [SIGMA_MIN, SIGMA_MAX].
            Required keyword.
        block_size: number of qints per block. Required keyword.

    Returns:
        ATBHv1 bytes.
    """
    if qints is None:
        raise ValueError("encode_hyperprior: qints is required")
    if num_symbols is None or num_symbols < 2:
        raise ValueError(
            f"encode_hyperprior: num_symbols must be >= 2, got {num_symbols}"
        )
    if offset is None:
        raise ValueError("encode_hyperprior: offset is required")
    if sigma_per_block is None:
        raise ValueError("encode_hyperprior: sigma_per_block is required")
    if block_size is None or block_size < 1:
        raise ValueError(
            f"encode_hyperprior: block_size must be >= 1, got {block_size}"
        )
    flat = np.ascontiguousarray(qints).ravel()
    if flat.size == 0:
        raise ValueError("encode_hyperprior: qints is empty")
    symbols = flat.astype(np.int64) + int(offset)
    if symbols.min() < 0 or symbols.max() >= num_symbols:
        raise ValueError(
            f"encode_hyperprior: symbols out of range [0, {num_symbols}): "
            f"min={int(symbols.min())}, max={int(symbols.max())}"
        )
    n_total = int(flat.size)
    n_blocks = (n_total + block_size - 1) // block_size
    sigma_arr = np.asarray(sigma_per_block, dtype=np.float64).ravel()
    if sigma_arr.size != n_blocks:
        raise ValueError(
            f"encode_hyperprior: sigma_per_block has {sigma_arr.size} "
            f"entries; expected {n_blocks} blocks"
        )

    # Quantize σ to q8 (matches encoder/decoder).
    sigma_q8 = np.fromiter(
        (_sigma_to_q8(s) for s in sigma_arr),
        dtype=np.uint8,
        count=n_blocks,
    )
    # Encoder uses the SAME q8-roundtripped σ as the decoder will.
    sigma_dec = np.fromiter(
        (_q8_to_sigma(int(q)) for q in sigma_q8),
        dtype=np.float64,
        count=n_blocks,
    )

    encoder = _ArithmeticEncoder()
    for b in range(n_blocks):
        sig = float(np.clip(sigma_dec[b], _SIGMA_MIN, _SIGMA_MAX))
        pmf = discretized_gaussian_pmf(
            sigma=sig, num_symbols=num_symbols, offset=offset
        )
        freq = _pmf_to_int_freq(pmf, total_freq=1 << 16)
        cum, total = _cumulative_table(freq)
        start = b * block_size
        end = min(start + block_size, n_total)
        for i in range(start, end):
            s = int(symbols[i])
            encoder.encode(int(cum[s]), int(cum[s + 1]), int(total))
    payload = encoder.finish()

    out = io.BytesIO()
    out.write(_ATBH_MAGIC)
    out.write(struct.pack("<H", _ATBH_VERSION))
    out.write(struct.pack("<H", int(num_symbols)))
    out.write(struct.pack("<i", int(offset)))
    out.write(struct.pack("<Q", int(n_total)))
    out.write(struct.pack("<I", int(block_size)))
    out.write(struct.pack("<I", int(n_blocks)))
    out.write(sigma_q8.tobytes())
    out.write(struct.pack("<Q", len(payload)))
    out.write(payload)
    blob = out.getvalue()
    decoded = decode_hyperprior(blob=blob, expected_dtype=flat.dtype)
    if not np.array_equal(decoded, flat):
        raise ValueError("encode_hyperprior: decoder roundtrip verification failed")
    return blob


def decode_hyperprior(
    *,
    blob: bytes,
    expected_dtype: np.dtype = np.int8,
) -> np.ndarray:
    """ATBHv1 arithmetic terminal decoder (matched to ``encode_hyperprior``)."""
    if blob is None:
        raise ValueError("decode_hyperprior: blob is required")
    if len(blob) < _ATBH_FIXED_HEADER_SIZE:
        raise ValueError(
            f"decode_hyperprior: truncated fixed header "
            f"(got {len(blob)} bytes, need at least {_ATBH_FIXED_HEADER_SIZE})"
        )
    if blob[:4] != _ATBH_MAGIC:
        raise ValueError(
            f"decode_hyperprior: bad magic {blob[:4]!r}, expected {_ATBH_MAGIC!r}"
        )
    cursor = 4
    (version,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if version != _ATBH_VERSION:
        raise ValueError(
            f"decode_hyperprior: unsupported version {version}"
        )
    (num_symbols,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if num_symbols < 2:
        raise ValueError(f"decode_hyperprior: num_symbols must be >= 2, got {num_symbols}")
    (offset,) = struct.unpack_from("<i", blob, cursor)
    cursor += 4
    (n_total,) = struct.unpack_from("<Q", blob, cursor)
    cursor += 8
    (block_size,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    if block_size < 1:
        raise ValueError(f"decode_hyperprior: block_size must be >= 1, got {block_size}")
    (n_blocks,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    expected_blocks = (int(n_total) + int(block_size) - 1) // int(block_size)
    if n_blocks != expected_blocks:
        raise ValueError(
            f"decode_hyperprior: n_blocks {n_blocks} does not match "
            f"ceil(n_total/block_size) {expected_blocks}"
        )
    if len(blob) < cursor + n_blocks + 8:
        raise ValueError(
            "decode_hyperprior: truncated sigma/header payload "
            f"(got {len(blob)} bytes, need at least {cursor + n_blocks + 8})"
        )
    sigma_q8 = np.frombuffer(blob[cursor : cursor + n_blocks], dtype=np.uint8).copy()
    cursor += n_blocks
    (payload_len,) = struct.unpack_from("<Q", blob, cursor)
    cursor += 8
    expected_end = cursor + int(payload_len)
    if expected_end > len(blob):
        raise ValueError(
            f"decode_hyperprior: payload truncated "
            f"(got {len(blob) - cursor}, declared {payload_len})"
        )
    if expected_end != len(blob):
        raise ValueError(
            f"decode_hyperprior: trailing bytes after payload "
            f"(payload_end={expected_end}, blob_len={len(blob)})"
        )
    payload = blob[cursor : cursor + payload_len]

    sigma_dec = np.fromiter(
        (_q8_to_sigma(int(q)) for q in sigma_q8),
        dtype=np.float64,
        count=n_blocks,
    )

    dec = _ArithmeticDecoder(payload)
    out = np.empty(n_total, dtype=np.int64)
    out_idx = 0
    for b in range(n_blocks):
        sig = float(np.clip(sigma_dec[b], _SIGMA_MIN, _SIGMA_MAX))
        pmf = discretized_gaussian_pmf(
            sigma=sig, num_symbols=num_symbols, offset=offset
        )
        freq = _pmf_to_int_freq(pmf, total_freq=1 << 16)
        cum, total = _cumulative_table(freq)
        n_in_block = min(block_size, n_total - out_idx)
        for _ in range(n_in_block):
            target = dec.get_target(int(total))
            s = int(np.searchsorted(cum, target, side="right") - 1)
            if s < 0 or s >= num_symbols:
                raise ValueError(
                    f"decode_hyperprior: symbol index {s} out of range "
                    f"[0, {num_symbols})"
                )
            dec.remove(int(cum[s]), int(cum[s + 1]), int(total))
            out[out_idx] = s
            out_idx += 1
        if out_idx >= n_total:
            break
    if out_idx != n_total:
        raise ValueError(
            f"decode_hyperprior: decoded {out_idx} symbols, expected {n_total}"
        )
    out -= int(offset)
    return out.astype(expected_dtype)


# ────────────────────────────────────────────────────────────────────────────
# Shannon-entropy diagnostic (theoretical floor for the arithmetic terminal)
# ────────────────────────────────────────────────────────────────────────────


def shannon_entropy_bits(
    *,
    qints: np.ndarray,
    num_symbols: int,
    offset: int,
) -> float:
    """Return the empirical Shannon entropy of the qint stream in bits/symbol.

    H(X) = -Σ_x p(x) log2 p(x). This is the LOWER BOUND on the arithmetic
    terminal's bits/symbol; the actual codec rate exceeds this by the
    finite-precision overhead (typically <0.5%).

    Used by the orchestrator's empirical report to express each codec's
    achieved rate vs the Shannon floor.
    """
    flat = np.ascontiguousarray(qints).ravel().astype(np.int64)
    syms = flat + int(offset)
    if syms.min() < 0 or syms.max() >= num_symbols:
        raise ValueError(
            f"shannon_entropy_bits: symbols out of range [0, {num_symbols})"
        )
    counts = np.bincount(syms, minlength=num_symbols).astype(np.float64)
    n = float(counts.sum())
    if n <= 0:
        return 0.0
    p = counts / n
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())
    return H


__all__ = [
    "decode_hyperprior",
    "decode_static",
    "encode_hyperprior",
    "encode_static",
    "shannon_entropy_bits",
]
