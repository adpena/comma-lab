# SPDX-License-Identifier: MIT
"""Sparse PacketIR codec — three orthogonal-composable sparse residual primitives.

This module closes the wire-format ceiling identified by O's 2026-05-11 L2
score-aware encoder landing (memory:
``feedback_l2_score_aware_encoders_wavelet_c3_cool_chic_landed_20260511.md``):
the L2 encoders correctly REFUSE to emit dense 3.66 GB / 228 MB / 4.81 GB
residual blobs because the L1 inflate format is dense (zero-padded skipped
frames are still charged). To make L2 encoders dispatch-eligible, the L1
inflate format must accept SPARSE bytes.

The three primitives below are the canonical sparse residual representations:

1. **RLE-of-zeros** — natural residuals are sparse after quantization. The
   majority of post-quantization int8 coefficients are zero. RLE encodes only
   the non-zero indices + values. Best for: dense-grid residuals that are
   uniformly sparse (wavelet bands after a low-magnitude threshold).

2. **Arithmetic-coded coefficient stream** — biased coefficient distributions
   (Laplacian, peaked-at-zero) are compressed well by arithmetic coding with
   a per-stream histogram. Best for: per-band wavelet coefficients with
   nontrivial magnitude distributions; c3 first-difference deltas.

3. **Temporal-subsampled indicator vector** — for per-frame residuals where
   only K of N frames carry signal, encode a K-of-N indicator bitmap +
   only the K signal-carrying residuals (densely packed). Best for: temporal
   redundancy where some frame-pairs do not need any residual at all (the
   PR106 r2 score-aware Lagrangian + Pareto frontier finding that pose
   marginal value concentrates on a frame subset).

The primitives **compose orthogonally**:

  wavelet residual → (per-band) → encode_rle_of_zeros (most coeffs near-zero)
                                → encode_arithmetic_coefficients (the non-zero coeffs)
                                → optional encode_temporal_subsampled
  c3 residual      → encode_temporal_subsampled (only K-of-N first-diffs nonzero)
                   → encode_rle_of_zeros (per kept frame)
  cool_chic        → encode_rle_of_zeros (per level after quantization)
                   → encode_arithmetic_coefficients (the level coefficients)

Wire format design philosophy (mirrors PR101/PR103/PR93):

- **byte-stable + endian-fixed** — uint32 lengths in little-endian (matches
  PR106_RESIDUAL_MAGIC framing and the wider tac.packet_compiler suite).
- **typed frozen dataclasses** for in-memory transport; ``encode_*`` returns
  the dataclass plus byte serialization helpers ``serialize_*`` /
  ``deserialize_*``.
- **fail-closed contract** — every primitive raises ``SparsePacketIRError``
  on malformed input rather than silently producing degenerate bytes.
- **8 archive-grammar fields** declared per HNeRV parity discipline lesson 3:
  archive_grammar="sparse_packet_ir_v1", parser_section_manifest declared in
  ``serialize_*`` docstring, inflate_runtime_loc_budget=200 (substrate
  engineering waiver), runtime_dep_closure=("numpy", "constriction"),
  export_format declared per primitive, score_aware_loss=False (proxy-grade
  research tool), bolt_on_loc_budget=400 (this file), no_op_detector_planned
  via round-trip property tests.

CLAUDE.md compliance:

* No scorer load (no PoseNet/SegNet/FastViT/EfficientNet imports).
* No MPS / torch dependency — pure numpy + constriction + stdlib.
* No ``/tmp`` paths.
* Frozen dataclasses; ``encode → decode`` is bit-exact on golden vectors.
* Permanent ``score_claim`` / ``promotion_eligible`` /
  ``ready_for_exact_eval_dispatch`` False invariants per Catalog #100
  ``check_gate2_no_naked_bytes``.

Closes O's L2 wire-format ceiling. Sister of PR101/PR103/PR93 primitives.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import constriction
import numpy as np

# ── Wire-format constants ───────────────────────────────────────────────────

#: Schema label embedded in golden-vector manifests.
SPARSE_RLE_SCHEMA: Final[str] = "sparse_rle_of_zeros.v1"
SPARSE_AC_SCHEMA: Final[str] = "sparse_arithmetic_coefficients.v1"
SPARSE_TEMPORAL_SCHEMA: Final[str] = "sparse_temporal_subsampled.v1"

#: Magic bytes for self-delimiting wire framing (4-byte ASCII).
SPARSE_RLE_MAGIC: Final[bytes] = b"SRL1"
SPARSE_AC_MAGIC: Final[bytes] = b"SAC1"
SPARSE_TEMPORAL_MAGIC: Final[bytes] = b"STS1"

#: Supported nonzero-value dtypes for RLE primitive.
SUPPORTED_RLE_NONZERO_DTYPES: Final[tuple[np.dtype, ...]] = (
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
)


class SparsePacketIRError(ValueError):
    """Raised on wire-format / contract violations."""


# ── Primitive 1: RLE-of-zeros ──────────────────────────────────────────────


@dataclass(frozen=True)
class RleOfZerosStream:
    """Sparse representation of a dense array via non-zero (index, value) pairs.

    Wire format (when serialised via :func:`serialize_rle_of_zeros`):

    ::

        magic(4) = b"SRL1"
        total_length(4, LE u32)
        n_nonzero(4, LE u32)
        nonzero_dtype_code(1, u8) — 0=int8, 1=int16, 2=int32
        nonzero_indices(n_nonzero * 4 bytes, LE u32)
        nonzero_values(n_nonzero * itemsize bytes, LE signed)

    Attributes
    ----------
    nonzero_indices:
        ``uint32`` indices into the dense layout (C-order flat).
    nonzero_values:
        Non-zero values aligned with ``nonzero_indices``. Dtype is one of
        :data:`SUPPORTED_RLE_NONZERO_DTYPES`.
    total_length:
        Length of the dense array the stream represents. Required by the
        decoder to allocate the zero-filled output.
    """

    nonzero_indices: np.ndarray
    nonzero_values: np.ndarray
    total_length: int

    def __post_init__(self) -> None:
        if self.total_length < 0:
            raise SparsePacketIRError(
                f"total_length must be >= 0; got {self.total_length}"
            )
        if self.nonzero_indices.ndim != 1:
            raise SparsePacketIRError(
                f"nonzero_indices must be 1D; got shape {self.nonzero_indices.shape}"
            )
        if self.nonzero_indices.dtype != np.uint32:
            raise SparsePacketIRError(
                f"nonzero_indices dtype must be uint32; got {self.nonzero_indices.dtype}"
            )
        if self.nonzero_values.ndim != 1:
            raise SparsePacketIRError(
                f"nonzero_values must be 1D; got shape {self.nonzero_values.shape}"
            )
        if self.nonzero_indices.size != self.nonzero_values.size:
            raise SparsePacketIRError(
                f"nonzero_indices size {self.nonzero_indices.size} != "
                f"nonzero_values size {self.nonzero_values.size}"
            )
        if self.nonzero_values.dtype not in SUPPORTED_RLE_NONZERO_DTYPES:
            raise SparsePacketIRError(
                f"nonzero_values dtype {self.nonzero_values.dtype} not in "
                f"supported set {SUPPORTED_RLE_NONZERO_DTYPES}"
            )
        if self.nonzero_indices.size > 0:
            if int(self.nonzero_indices.max()) >= self.total_length:
                raise SparsePacketIRError(
                    f"nonzero_indices contains index >= total_length "
                    f"({int(self.nonzero_indices.max())} >= {self.total_length})"
                )
            if np.any(np.diff(self.nonzero_indices.astype(np.int64)) <= 0):
                raise SparsePacketIRError(
                    "nonzero_indices must be strictly increasing; duplicate "
                    "or out-of-order indices are not canonical"
                )
            if np.any(self.nonzero_values == 0):
                raise SparsePacketIRError(
                    "nonzero_values contains a zero entry; the contract is "
                    "non-zero only (a zero entry inflates byte cost for no gain)"
                )

    @property
    def sparsity_ratio(self) -> float:
        """Fraction of zero entries in the represented dense array."""
        if self.total_length == 0:
            return 0.0
        return 1.0 - (self.nonzero_indices.size / self.total_length)


def encode_rle_of_zeros(
    dense: np.ndarray,
    *,
    dtype_nonzero: np.dtype | None = None,
) -> RleOfZerosStream:
    """Build an :class:`RleOfZerosStream` from a dense array.

    Parameters
    ----------
    dense:
        1D dense array of integer values. Non-integer dtypes are refused;
        the caller is responsible for quantising floats first (the L2
        encoders already do this).
    dtype_nonzero:
        Optional override for the stored value dtype. Defaults to the
        smallest signed dtype in :data:`SUPPORTED_RLE_NONZERO_DTYPES` that
        contains all non-zero values.
    """
    if dense.ndim != 1:
        raise SparsePacketIRError(
            f"dense must be 1D (flatten first); got shape {dense.shape}"
        )
    if not np.issubdtype(dense.dtype, np.integer):
        raise SparsePacketIRError(
            f"dense dtype must be integer; got {dense.dtype} (quantise first)"
        )
    indices = np.flatnonzero(dense).astype(np.uint32, copy=False)
    raw_values = dense[indices]
    if dtype_nonzero is None:
        if indices.size == 0:
            target = np.dtype(np.int8)
        else:
            v_min = int(raw_values.min())
            v_max = int(raw_values.max())
            if v_min >= -128 and v_max <= 127:
                target = np.dtype(np.int8)
            elif v_min >= -32768 and v_max <= 32767:
                target = np.dtype(np.int16)
            else:
                target = np.dtype(np.int32)
    else:
        target = np.dtype(dtype_nonzero)
        if target not in SUPPORTED_RLE_NONZERO_DTYPES:
            raise SparsePacketIRError(
                f"dtype_nonzero {target} not in {SUPPORTED_RLE_NONZERO_DTYPES}"
            )
        if indices.size > 0:
            info = np.iinfo(target)
            if int(raw_values.min()) < info.min or int(raw_values.max()) > info.max:
                raise SparsePacketIRError(
                    f"nonzero values out of range for dtype_nonzero={target}; "
                    f"min={int(raw_values.min())} max={int(raw_values.max())}"
                )
    values = raw_values.astype(target, copy=False)
    return RleOfZerosStream(
        nonzero_indices=indices,
        nonzero_values=values,
        total_length=int(dense.size),
    )


def decode_rle_of_zeros(stream: RleOfZerosStream) -> np.ndarray:
    """Inverse of :func:`encode_rle_of_zeros`.

    Returns a dense ``np.ndarray`` of length ``stream.total_length`` and
    dtype matching ``stream.nonzero_values.dtype`` (the caller cast the
    quantised dense array; the round-trip preserves the dtype).
    """
    out = np.zeros(stream.total_length, dtype=stream.nonzero_values.dtype)
    if stream.nonzero_indices.size > 0:
        out[stream.nonzero_indices] = stream.nonzero_values
    return out


def _dtype_code(dtype: np.dtype) -> int:
    if dtype == np.int8:
        return 0
    if dtype == np.int16:
        return 1
    if dtype == np.int32:
        return 2
    raise SparsePacketIRError(f"unsupported nonzero dtype {dtype}")


def _dtype_from_code(code: int) -> np.dtype:
    if code == 0:
        return np.dtype(np.int8)
    if code == 1:
        return np.dtype(np.int16)
    if code == 2:
        return np.dtype(np.int32)
    raise SparsePacketIRError(f"unknown nonzero dtype code {code}")


def serialize_rle_of_zeros(stream: RleOfZerosStream) -> bytes:
    """Serialize :class:`RleOfZerosStream` to self-delimiting bytes."""
    dtype_code = _dtype_code(stream.nonzero_values.dtype)
    n_nonzero = int(stream.nonzero_indices.size)
    header = struct.pack(
        "<4sIIB", SPARSE_RLE_MAGIC, stream.total_length, n_nonzero, dtype_code
    )
    indices_bytes = stream.nonzero_indices.astype("<u4", copy=False).tobytes()
    # Values: little-endian signed of declared dtype.
    if stream.nonzero_values.dtype == np.int8:
        values_bytes = stream.nonzero_values.tobytes()
    else:
        wire_dtype = (
            "<i2" if stream.nonzero_values.dtype == np.int16 else "<i4"
        )
        values_bytes = stream.nonzero_values.astype(wire_dtype, copy=False).tobytes()
    return header + indices_bytes + values_bytes


def deserialize_rle_of_zeros(blob: bytes) -> RleOfZerosStream:
    """Inverse of :func:`serialize_rle_of_zeros`."""
    if len(blob) < 13:
        raise SparsePacketIRError(
            f"blob too short for RLE header: {len(blob)} < 13"
        )
    magic, total_length, n_nonzero, dtype_code = struct.unpack_from(
        "<4sIIB", blob, 0
    )
    if magic != SPARSE_RLE_MAGIC:
        raise SparsePacketIRError(
            f"RLE magic mismatch: got {magic!r} expected {SPARSE_RLE_MAGIC!r}"
        )
    target_dtype = _dtype_from_code(int(dtype_code))
    pos = 13
    idx_bytes = 4 * n_nonzero
    val_bytes = target_dtype.itemsize * n_nonzero
    if pos + idx_bytes + val_bytes > len(blob):
        raise SparsePacketIRError(
            f"RLE blob truncated: need {pos + idx_bytes + val_bytes}, have {len(blob)}"
        )
    indices = np.frombuffer(
        blob, dtype="<u4", count=n_nonzero, offset=pos
    ).astype(np.uint32, copy=True)
    pos += idx_bytes
    if target_dtype == np.dtype(np.int8):
        values = np.frombuffer(
            blob, dtype=np.int8, count=n_nonzero, offset=pos
        ).copy()
    else:
        wire_dtype = "<i2" if target_dtype == np.int16 else "<i4"
        values = np.frombuffer(
            blob, dtype=wire_dtype, count=n_nonzero, offset=pos
        ).astype(target_dtype, copy=True)
    pos += val_bytes
    if pos != len(blob):
        raise SparsePacketIRError(
            f"RLE blob has trailing bytes: pos={pos} total={len(blob)}"
        )
    return RleOfZerosStream(
        nonzero_indices=indices,
        nonzero_values=values,
        total_length=int(total_length),
    )


# ── Primitive 2: Arithmetic-coded coefficient stream ───────────────────────


@dataclass(frozen=True)
class ArithmeticCodedCoefficientStream:
    """AC-coded coefficient stream with explicit per-stream histogram.

    Wire format (when serialised via :func:`serialize_arithmetic_coefficients`):

    ::

        magic(4)             = b"SAC1"
        n_symbols(4, LE u32)
        alphabet_size(4, LE u32)
        symbol_offset(4, LE i32) — added to decoded symbols to recover signed values
        histogram(alphabet_size * 4 bytes, LE f32)
        word_count(4, LE u32)
        encoded(word_count * 4 bytes, BE u32 — constriction big-endian convention)

    Attributes
    ----------
    encoded_bytes:
        Range-coded uint32 stream (constriction, big-endian bytes).
    histogram:
        Per-symbol categorical distribution, length ``alphabet_size``.
    n_symbols:
        Number of symbols encoded (required by the decoder).
    alphabet_size:
        Symbol alphabet cardinality.
    symbol_offset:
        Integer offset to add when decoding to recover signed values
        (e.g. -128 if input was int8 mapped to [0, 256)).
    """

    encoded_bytes: bytes
    histogram: np.ndarray
    n_symbols: int
    alphabet_size: int
    symbol_offset: int

    def __post_init__(self) -> None:
        if self.n_symbols < 0:
            raise SparsePacketIRError(f"n_symbols must be >= 0; got {self.n_symbols}")
        if self.alphabet_size < 2:
            raise SparsePacketIRError(
                f"alphabet_size must be >= 2; got {self.alphabet_size}"
            )
        if self.histogram.shape != (self.alphabet_size,):
            raise SparsePacketIRError(
                f"histogram shape {self.histogram.shape} != "
                f"({self.alphabet_size},)"
            )
        if not np.all(np.isfinite(self.histogram)):
            raise SparsePacketIRError("histogram contains non-finite entries")
        if np.any(self.histogram < 0):
            raise SparsePacketIRError("histogram contains negative entries")


def _make_categorical(weights: np.ndarray) -> constriction.stream.model.Categorical:
    """Build a Categorical distribution from raw weights (mirrors PR103)."""
    p = np.asarray(weights, dtype=np.float64)
    p = np.maximum(p, 1e-10)
    p /= p.sum()
    return constriction.stream.model.Categorical(p, perfect=False)


def encode_arithmetic_coefficients(
    values: np.ndarray,
    *,
    histogram: np.ndarray | None = None,
    symbol_offset: int | None = None,
    alphabet_size: int | None = None,
) -> ArithmeticCodedCoefficientStream:
    """Range-code a 1D integer array with a per-stream histogram.

    Parameters
    ----------
    values:
        1D integer array. May be signed; ``symbol_offset`` is auto-computed
        to map to the unsigned alphabet [0, ``alphabet_size``) when not
        provided.
    histogram:
        Optional categorical distribution over the unsigned alphabet. When
        omitted, the empirical histogram of ``values - symbol_offset`` is
        used (+1 Laplace smoothing).
    symbol_offset:
        Optional explicit offset. When omitted, ``-min(values)`` is used.
    alphabet_size:
        Optional explicit alphabet cardinality. When omitted,
        ``max(values) - min(values) + 1`` is used.
    """
    if values.ndim != 1:
        raise SparsePacketIRError(
            f"values must be 1D; got shape {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise SparsePacketIRError(
            f"values must be integer dtype; got {values.dtype}"
        )
    if values.size == 0:
        # Empty stream: short-circuit. constriction refuses to encode zero
        # symbols.
        return ArithmeticCodedCoefficientStream(
            encoded_bytes=b"",
            histogram=np.array([1.0, 1.0], dtype=np.float32),
            n_symbols=0,
            alphabet_size=2,
            symbol_offset=0,
        )

    v_min = int(values.min())
    v_max = int(values.max())
    if symbol_offset is None:
        symbol_offset = -v_min
    auto_alphabet = v_max - v_min + 1
    if alphabet_size is None:
        alphabet_size = auto_alphabet
    if alphabet_size < auto_alphabet:
        raise SparsePacketIRError(
            f"alphabet_size {alphabet_size} too small for value range "
            f"[{v_min}, {v_max}] (need >= {auto_alphabet})"
        )
    if alphabet_size < 2:
        # constriction Categorical requires at least 2 symbols; pad.
        alphabet_size = 2
    symbols = (values.astype(np.int64) + symbol_offset).astype(np.int32)
    if symbols.min() < 0 or symbols.max() >= alphabet_size:
        raise SparsePacketIRError(
            f"mapped symbols out of [0, {alphabet_size}); "
            f"min={int(symbols.min())} max={int(symbols.max())}"
        )
    if histogram is None:
        hist = np.bincount(symbols, minlength=alphabet_size).astype(np.float32)
        hist += 1.0  # Laplace smoothing
    else:
        hist = np.asarray(histogram, dtype=np.float32)
        if hist.shape != (alphabet_size,):
            raise SparsePacketIRError(
                f"histogram shape {hist.shape} != ({alphabet_size},)"
            )

    cat = _make_categorical(hist)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, cat)
    words = encoder.get_compressed()
    encoded_bytes = np.asarray(words, dtype=">u4").tobytes()
    return ArithmeticCodedCoefficientStream(
        encoded_bytes=encoded_bytes,
        histogram=hist,
        n_symbols=int(values.size),
        alphabet_size=int(alphabet_size),
        symbol_offset=int(symbol_offset),
    )


def decode_arithmetic_coefficients(
    stream: ArithmeticCodedCoefficientStream,
    *,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Inverse of :func:`encode_arithmetic_coefficients`.

    Returns a 1D ``np.ndarray`` of length ``stream.n_symbols``. The dtype
    defaults to int32 (always safe); the caller may downcast.
    """
    if stream.n_symbols == 0:
        return np.zeros(0, dtype=np.int32 if dtype is None else dtype)
    if len(stream.encoded_bytes) % 4 != 0:
        raise SparsePacketIRError(
            f"encoded_bytes len {len(stream.encoded_bytes)} not multiple of 4"
        )
    words = np.frombuffer(stream.encoded_bytes, dtype=">u4").astype(np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    cat = _make_categorical(stream.histogram)
    out = np.zeros(stream.n_symbols, dtype=np.int32)
    for i in range(stream.n_symbols):
        out[i] = decoder.decode(cat)
    out -= stream.symbol_offset
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def serialize_arithmetic_coefficients(
    stream: ArithmeticCodedCoefficientStream,
) -> bytes:
    """Serialize :class:`ArithmeticCodedCoefficientStream` to self-delimiting bytes."""
    if len(stream.encoded_bytes) % 4 != 0:
        raise SparsePacketIRError(
            f"encoded_bytes len {len(stream.encoded_bytes)} not multiple of 4"
        )
    word_count = len(stream.encoded_bytes) // 4
    header = struct.pack(
        "<4sIIi",
        SPARSE_AC_MAGIC,
        stream.n_symbols,
        stream.alphabet_size,
        stream.symbol_offset,
    )
    hist_bytes = stream.histogram.astype("<f4", copy=False).tobytes()
    wc_bytes = struct.pack("<I", word_count)
    return header + hist_bytes + wc_bytes + stream.encoded_bytes


def deserialize_arithmetic_coefficients(
    blob: bytes,
) -> ArithmeticCodedCoefficientStream:
    """Inverse of :func:`serialize_arithmetic_coefficients`."""
    if len(blob) < 16:
        raise SparsePacketIRError(
            f"blob too short for AC header: {len(blob)} < 16"
        )
    magic, n_symbols, alphabet_size, symbol_offset = struct.unpack_from(
        "<4sIIi", blob, 0
    )
    if magic != SPARSE_AC_MAGIC:
        raise SparsePacketIRError(
            f"AC magic mismatch: got {magic!r} expected {SPARSE_AC_MAGIC!r}"
        )
    pos = 16
    hist_bytes = alphabet_size * 4
    if pos + hist_bytes + 4 > len(blob):
        raise SparsePacketIRError(
            f"AC blob truncated: need {pos + hist_bytes + 4}, have {len(blob)}"
        )
    histogram = np.frombuffer(
        blob, dtype="<f4", count=alphabet_size, offset=pos
    ).astype(np.float32, copy=True)
    pos += hist_bytes
    (word_count,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    encoded_len = word_count * 4
    if pos + encoded_len != len(blob):
        raise SparsePacketIRError(
            f"AC blob length mismatch: pos+encoded={pos + encoded_len}, "
            f"total={len(blob)}"
        )
    encoded_bytes = bytes(blob[pos : pos + encoded_len])
    return ArithmeticCodedCoefficientStream(
        encoded_bytes=encoded_bytes,
        histogram=histogram,
        n_symbols=int(n_symbols),
        alphabet_size=int(alphabet_size),
        symbol_offset=int(symbol_offset),
    )


# ── Primitive 3: Temporal-subsampling indicator vector ─────────────────────


@dataclass(frozen=True)
class TemporalSubsampledResidualStream:
    """K-of-N temporal subsampling stream.

    For per-frame residuals where only K of N frames carry signal, encode an
    N-bit indicator bitmap + only the K residuals' raw bytes (concatenated).

    Wire format (when serialised via :func:`serialize_temporal_subsampled`):

    ::

        magic(4)               = b"STS1"
        N(4, LE u32)
        K(4, LE u32)
        per_frame_bytes(4, LE u32)
        indicator_bitmap(ceil(N/8) bytes — bit i = 1 iff frame i carries signal)
        residuals_packed(K * per_frame_bytes bytes)

    Attributes
    ----------
    indicator_bitmap:
        Packed ``ceil(N/8)`` bytes; bit ``i`` (LSB-first within each byte)
        indicates whether frame ``i`` carries a residual.
    residuals_packed:
        Concatenated bytes of the K signal-carrying residuals. The decoder
        unpacks them into a list of length ``N`` with ``None`` for skipped
        frames.
    N:
        Total frame count.
    K:
        Number of signal-carrying frames (must equal popcount of bitmap).
    per_frame_bytes:
        Byte count of each signal-carrying frame's residual. The contract
        is uniform-size frames; non-uniform sizes are out-of-scope (the
        caller must zero-pad).
    """

    indicator_bitmap: bytes
    residuals_packed: bytes
    N: int
    K: int
    per_frame_bytes: int

    def __post_init__(self) -> None:
        if self.N < 0:
            raise SparsePacketIRError(f"N must be >= 0; got {self.N}")
        if self.K < 0 or self.K > self.N:
            raise SparsePacketIRError(
                f"K must be in [0, N={self.N}]; got {self.K}"
            )
        if self.per_frame_bytes < 0:
            raise SparsePacketIRError(
                f"per_frame_bytes must be >= 0; got {self.per_frame_bytes}"
            )
        expected_bitmap_bytes = (self.N + 7) // 8
        if len(self.indicator_bitmap) != expected_bitmap_bytes:
            raise SparsePacketIRError(
                f"indicator_bitmap len {len(self.indicator_bitmap)} != "
                f"ceil(N/8) = {expected_bitmap_bytes}"
            )
        if self.N % 8 and self.indicator_bitmap:
            used_bits = self.N % 8
            padding_mask = (~((1 << used_bits) - 1)) & 0xFF
            if self.indicator_bitmap[-1] & padding_mask:
                raise SparsePacketIRError(
                    "indicator_bitmap has non-zero padding bits beyond N"
                )
        if len(self.residuals_packed) != self.K * self.per_frame_bytes:
            raise SparsePacketIRError(
                f"residuals_packed len {len(self.residuals_packed)} != "
                f"K * per_frame_bytes = {self.K * self.per_frame_bytes}"
            )
        # Verify popcount(bitmap) == K
        popcount = sum(bin(b).count("1") for b in self.indicator_bitmap)
        if popcount != self.K:
            raise SparsePacketIRError(
                f"indicator_bitmap popcount {popcount} != K {self.K}"
            )


def _pack_indicator_bitmap(indicators: Sequence[bool]) -> bytes:
    """Pack a sequence of bool indicators into LSB-first byte stream."""
    n = len(indicators)
    out = bytearray((n + 7) // 8)
    for i, flag in enumerate(indicators):
        if flag:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _unpack_indicator_bitmap(bitmap: bytes, N: int) -> list[bool]:
    """Inverse of :func:`_pack_indicator_bitmap`."""
    out = [False] * N
    for i in range(N):
        if bitmap[i // 8] & (1 << (i % 8)):
            out[i] = True
    return out


def pad_per_frame_to_uniform_size_with_length_prefix(
    per_frame_payloads: Sequence[bytes | None],
) -> list[np.ndarray | None]:
    """Zero-pad variable-length per-frame byte payloads to a uniform size.

    Each non-None payload is wrapped as ``<I (LE u32 length) | payload | zero-pad>``
    so the payload's true byte length survives the uniform-size contract. The
    decoder side (e.g. ``decode_wavelet_residual_sparse`` in family inflate.py)
    reads the leading ``<I`` to recover the actual payload length and slices
    off the zero padding.

    This is the canonical adapter between variable-size encoder outputs (RLE,
    AC, score-aware sparse) and the temporal-subsampled wire format's strict
    uniform-per-frame-bytes contract. The padding bytes are pure zeros so any
    downstream entropy coder (RLE-of-zeros, brotli, etc.) compresses them to
    near-zero overhead.

    Parameters
    ----------
    per_frame_payloads:
        Sequence of length ``N``; each entry is either ``bytes`` (signal-
        carrying frame) or ``None`` (skipped frame). Variable-size payloads
        are accepted; all non-None payloads are padded to ``max(len(p))``.

    Returns
    -------
    list[np.ndarray | None]
        Sequence of length ``N``; non-None entries are uint8 numpy arrays of
        identical size ``4 + max_payload_len``. Suitable as input to
        :func:`encode_temporal_subsampled`.
    """
    non_empty = [payload for payload in per_frame_payloads if payload is not None]
    if not non_empty:
        return [None] * len(per_frame_payloads)
    max_payload = max(len(payload) for payload in non_empty)
    out: list[np.ndarray | None] = []
    for payload in per_frame_payloads:
        if payload is None:
            out.append(None)
            continue
        framed = (
            struct.pack("<I", len(payload))
            + payload
            + b"\x00" * (max_payload - len(payload))
        )
        out.append(np.frombuffer(framed, dtype=np.uint8).copy())
    return out


def encode_temporal_subsampled(
    per_frame_residuals: Sequence[np.ndarray | None],
    *,
    pad_to_uniform_size: bool = False,
) -> TemporalSubsampledResidualStream:
    """Build a :class:`TemporalSubsampledResidualStream` from per-frame residuals.

    Parameters
    ----------
    per_frame_residuals:
        Sequence of length ``N``; each entry is either a uniform-size numpy
        array (signal-carrying frame) or ``None`` (skipped frame). All
        signal-carrying entries must share dtype + byte-size unless
        ``pad_to_uniform_size=True``.
    pad_to_uniform_size:
        When True, accept variable-size numpy arrays. The implementation
        zero-pads each non-None array (after a 4-byte LE u32 length prefix
        capturing the original size) so the encoded stream still satisfies
        the strict uniform-per-frame-bytes wire contract. Each non-None
        entry must be a ``np.uint8`` array (caller is responsible for the
        casting); the dtype-uniformity check still fires on the post-pad
        arrays. Defaults to False for back-compat.
    """
    N = len(per_frame_residuals)
    if pad_to_uniform_size:
        # Convert each non-None numpy array to a length-prefixed uint8 frame.
        adapted: list[np.ndarray | None] = []
        for r in per_frame_residuals:
            if r is None:
                adapted.append(None)
                continue
            if r.dtype != np.uint8:
                raise SparsePacketIRError(
                    f"pad_to_uniform_size=True requires uint8 arrays "
                    f"(caller must cast); got dtype {r.dtype}"
                )
            adapted.append(r.tobytes())
        adapted_padded = pad_per_frame_to_uniform_size_with_length_prefix(adapted)
        per_frame_residuals = adapted_padded
    indicators = [r is not None for r in per_frame_residuals]
    signal_arrays = [r for r in per_frame_residuals if r is not None]
    K = len(signal_arrays)
    if K == 0:
        per_frame_bytes = 0
    else:
        first = signal_arrays[0]
        per_frame_bytes = int(first.nbytes)
        for r in signal_arrays[1:]:
            if int(r.nbytes) != per_frame_bytes:
                raise SparsePacketIRError(
                    f"non-uniform per_frame_bytes: {int(r.nbytes)} != {per_frame_bytes} "
                    f"(caller must zero-pad to a common size, e.g. via "
                    f"pad_per_frame_to_uniform_size_with_length_prefix or "
                    f"pad_to_uniform_size=True)"
                )
            if r.dtype != first.dtype:
                raise SparsePacketIRError(
                    f"non-uniform dtype: {r.dtype} != {first.dtype}"
                )
    bitmap = _pack_indicator_bitmap(indicators)
    if K == 0:
        packed = b""
    else:
        packed = b"".join(r.tobytes() for r in signal_arrays)
    return TemporalSubsampledResidualStream(
        indicator_bitmap=bitmap,
        residuals_packed=packed,
        N=N,
        K=K,
        per_frame_bytes=per_frame_bytes,
    )


def decode_temporal_subsampled(
    stream: TemporalSubsampledResidualStream,
    *,
    dtype: np.dtype = np.dtype(np.int8),
    frame_shape: tuple[int, ...] | None = None,
) -> list[np.ndarray | None]:
    """Inverse of :func:`encode_temporal_subsampled`.

    Returns a list of length ``N`` with the K signal-carrying frames
    reconstructed as numpy arrays (dtype ``dtype``, shape ``frame_shape`` if
    provided else flat 1D) and ``None`` for skipped frames.
    """
    indicators = _unpack_indicator_bitmap(stream.indicator_bitmap, stream.N)
    out: list[np.ndarray | None] = [None] * stream.N
    if stream.K == 0:
        return out
    target = np.dtype(dtype)
    items_per_frame = stream.per_frame_bytes // target.itemsize
    if items_per_frame * target.itemsize != stream.per_frame_bytes:
        raise SparsePacketIRError(
            f"per_frame_bytes {stream.per_frame_bytes} not divisible by "
            f"dtype itemsize {target.itemsize}"
        )
    if frame_shape is not None:
        expected_items = int(np.prod(frame_shape))
        if expected_items != items_per_frame:
            raise SparsePacketIRError(
                f"frame_shape prod {expected_items} != items_per_frame {items_per_frame}"
            )
    cursor = 0
    k_index = 0
    for i, flag in enumerate(indicators):
        if not flag:
            continue
        start = k_index * stream.per_frame_bytes
        end = start + stream.per_frame_bytes
        chunk = stream.residuals_packed[start:end]
        arr = np.frombuffer(chunk, dtype=target).copy()
        if frame_shape is not None:
            arr = arr.reshape(frame_shape)
        out[i] = arr
        k_index += 1
        cursor = end
    if cursor != len(stream.residuals_packed):
        raise SparsePacketIRError(
            f"residuals_packed not fully consumed: cursor={cursor} "
            f"total={len(stream.residuals_packed)}"
        )
    return out


def serialize_temporal_subsampled(
    stream: TemporalSubsampledResidualStream,
) -> bytes:
    """Serialize :class:`TemporalSubsampledResidualStream` to self-delimiting bytes."""
    header = struct.pack(
        "<4sIII",
        SPARSE_TEMPORAL_MAGIC,
        stream.N,
        stream.K,
        stream.per_frame_bytes,
    )
    return header + stream.indicator_bitmap + stream.residuals_packed


def deserialize_temporal_subsampled(
    blob: bytes,
) -> TemporalSubsampledResidualStream:
    """Inverse of :func:`serialize_temporal_subsampled`."""
    if len(blob) < 16:
        raise SparsePacketIRError(
            f"blob too short for temporal header: {len(blob)} < 16"
        )
    magic, N, K, per_frame_bytes = struct.unpack_from("<4sIII", blob, 0)
    if magic != SPARSE_TEMPORAL_MAGIC:
        raise SparsePacketIRError(
            f"temporal magic mismatch: got {magic!r} expected {SPARSE_TEMPORAL_MAGIC!r}"
        )
    pos = 16
    bitmap_bytes = (N + 7) // 8
    if pos + bitmap_bytes > len(blob):
        raise SparsePacketIRError(
            f"temporal blob truncated at bitmap: need {pos + bitmap_bytes}, "
            f"have {len(blob)}"
        )
    bitmap = bytes(blob[pos : pos + bitmap_bytes])
    pos += bitmap_bytes
    expected_packed_len = K * per_frame_bytes
    if pos + expected_packed_len != len(blob):
        raise SparsePacketIRError(
            f"temporal blob length mismatch: pos+packed={pos + expected_packed_len}, "
            f"total={len(blob)}"
        )
    packed = bytes(blob[pos : pos + expected_packed_len])
    return TemporalSubsampledResidualStream(
        indicator_bitmap=bitmap,
        residuals_packed=packed,
        N=int(N),
        K=int(K),
        per_frame_bytes=int(per_frame_bytes),
    )


__all__ = [
    "ArithmeticCodedCoefficientStream",
    "RleOfZerosStream",
    "SPARSE_AC_MAGIC",
    "SPARSE_AC_SCHEMA",
    "SPARSE_RLE_MAGIC",
    "SPARSE_RLE_SCHEMA",
    "SPARSE_TEMPORAL_MAGIC",
    "SPARSE_TEMPORAL_SCHEMA",
    "SUPPORTED_RLE_NONZERO_DTYPES",
    "SparsePacketIRError",
    "TemporalSubsampledResidualStream",
    "decode_arithmetic_coefficients",
    "decode_rle_of_zeros",
    "decode_temporal_subsampled",
    "deserialize_arithmetic_coefficients",
    "deserialize_rle_of_zeros",
    "deserialize_temporal_subsampled",
    "encode_arithmetic_coefficients",
    "encode_rle_of_zeros",
    "encode_temporal_subsampled",
    "pad_per_frame_to_uniform_size_with_length_prefix",
    "serialize_arithmetic_coefficients",
    "serialize_rle_of_zeros",
    "serialize_temporal_subsampled",
]
