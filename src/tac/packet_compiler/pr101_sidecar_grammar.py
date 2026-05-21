# SPDX-License-Identifier: MIT
"""PR101 sidecar grammar — reusable byte-grammar primitives.

This module extracts the REUSABLE byte-grammar pieces from the PR101 public
submission (``hnerv_ft_microcodec/src/codec.py``) into typed primitives that
can be applied to any sidecar / per-pair / per-tensor stream — not just the
HNeRV-specific 28-dim latent-correction sidecar PR101 itself targets.

Three primitives land here:

1. **Ranked Huffman/no-op sidecar grammar**
   (:func:`encode_ranked_no_op_sidecar` / :func:`decode_ranked_no_op_sidecar`)

   The PR101 sidecar packs *per-pair* sparse corrections by (a) ranking which
   pairs are no-ops via a *colex combination rank* of the no-op positions
   (only ``ceil(log2(C(N,k)))`` bits — far less than a bitmask), then (b)
   storing the residual deltas with a *canonical Huffman code* whose length
   vector is itself rank-encoded over the Kraft polytope. The result is a
   self-delimiting binary string whose length is dominated by the entropy of
   the chosen deltas plus a tiny combinatorial header.

   In its PR101 form the alphabet is fixed (16 signed delta-x100 codes,
   2 ≤ length ≤ 8 bits). Here we generalise: a caller supplies a
   :class:`RankedSidecarSchema` declaring the delta vocabulary, code-length
   bounds, and "no-op" sentinel, and the encoder picks the smallest container
   among ``raw / packed-combinatorial / huffman-3bit / huffman-enum`` variants.

2. **Centered-delta uint8 packing under raw LZMA**
   (:func:`encode_centered_delta_uint8` / :func:`decode_centered_delta_uint8`)

   Used by PR101 for latent-quantization columns: per-dimension fp16 ``min`` +
   ``scale`` plus a column-major ``uint8`` block where row 0 is the absolute
   quantised value and rows 1..N-1 are *temporal deltas centered at 128*. The
   stream is then wrapped in *raw* (filter-only, no XZ container) LZMA, which
   has the smallest header of any standard general-purpose compressor in the
   stdlib.

   We surface this as a typed (mins, scales, base, deltas, lzma_bytes) tuple
   :class:`CenteredDeltaUint8Stream` plus pure encode/decode helpers. The
   caller's data may have any number of dims and pairs — PR101's 28×600 is
   the canonical example, not a baked-in constraint.

3. **Self-delimiting split Brotli streams**
   (:func:`split_brotli_self_delimiting` /
   :func:`parse_split_brotli_self_delimiting`)

   Concatenates ``N`` independently-Brotli-compressed sub-streams without
   storing per-stream lengths. The reader uses Brotli's frame structure to
   detect each stream's end byte-by-byte. PR101 uses this to pack 7 weight
   sub-tensors into one byte-range without paying for length prefixes. The
   trade-off is decode-time cost (byte-by-byte feeding) for byte-savings (no
   per-stream length prefix). The number of streams must be agreed
   out-of-band (PR101 hardcodes it via ``DECODER_STREAM_ENDS``); this module
   exposes that count as an explicit parameter.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src/codec.py``
(SHA pinned in ``check_public_pr_intake_clones_pristine``-protected intake;
see also analysis ledger
``.omx/research/representation_integration_gap_audit_20260508_codex.md``).

CLAUDE.md compliance:

* No scorer load — pure numpy / brotli / lzma / math.
* No MPS / torch import.
* No ``/tmp`` paths anywhere.
* Frozen dataclasses; ``encode→decode`` is covered by focused Python
  conformance tests. Native ports must add golden vectors before promotion.
* OSS-friendly: public surface is the 9 names re-exported from
  ``tac.packet_compiler``; everything else is module-private (``_``-prefixed).
"""

from __future__ import annotations

import io
import lzma
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache

import brotli
import numpy as np

# Internal LZMA filter chain used for the centered-delta uint8 latent stream.
# Matches the PR101 codec: LZMA1 (not LZMA2), dict 4 KiB, lc=3, lp=0, pb=0.
# These are the smallest-header LZMA settings supported by Python's stdlib.
_LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]


# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankedSidecarSchema:
    """Schema for a ranked Huffman/no-op sidecar over per-pair corrections.

    A sidecar stores one ``(dim, delta)`` correction per *valid* pair plus a
    set of *no-op* pair positions. The byte layout is parameterised by:

    Parameters
    ----------
    n_pairs:
        Total number of per-pair slots (e.g. 600 for the contest's 600
        pair-aligned latent positions).
    n_dims:
        Number of dimensions a correction may target (e.g. PR101 uses 28 for
        the HNeRV latent width).
    deltas:
        Allowed signed delta codes (PR101 uses ``[-10,-8,-6,-5,-4,-3,-2,-1,
        1,2,3,4,5,6,8,10]`` interpreted at 1/100 scale). ``len(deltas)`` must
        be ≥ 2 and the array must be sorted strictly ascending so the encoder
        can use index-based lookups.
    huff_min_len, huff_max_len:
        Lower and upper bounds on canonical-Huffman code lengths. PR101 uses
        ``(2, 8)``. The Kraft inequality must allow exactly one valid length
        vector for any non-zero rank.
    no_op_sentinel:
        Value used in the decoded ``dims`` array to indicate a no-op pair.
        PR101 reserves 255. Must satisfy ``no_op_sentinel >= n_dims`` so it
        cannot collide with a real dimension index.

    Notes
    -----
    This schema is the *only* mutable extension point of the PR101 grammar;
    the bit-packing helpers are otherwise locked to PR101's exact ordering
    so that golden vectors transfer to native ports unchanged.
    """

    n_pairs: int
    n_dims: int
    deltas: tuple[int, ...]
    huff_min_len: int = 2
    huff_max_len: int = 8
    no_op_sentinel: int = 255

    def __post_init__(self) -> None:
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0; got {self.n_pairs}")
        if self.n_dims <= 0:
            raise ValueError(f"n_dims must be > 0; got {self.n_dims}")
        if self.no_op_sentinel < self.n_dims:
            raise ValueError(
                "no_op_sentinel must be >= n_dims to avoid collision; "
                f"got sentinel={self.no_op_sentinel} n_dims={self.n_dims}"
            )
        if len(self.deltas) < 2:
            raise ValueError(
                f"need at least 2 delta codes; got {len(self.deltas)}"
            )
        if any(b <= a for a, b in zip(self.deltas, self.deltas[1:], strict=False)):
            raise ValueError(
                "deltas must be strictly ascending; got "
                f"{list(self.deltas)}"
            )
        if not (1 <= self.huff_min_len <= self.huff_max_len <= 16):
            raise ValueError(
                "expected 1 <= huff_min_len <= huff_max_len <= 16; got "
                f"min={self.huff_min_len} max={self.huff_max_len}"
            )

    @property
    def kraft_total(self) -> int:
        """Total Kraft weight for ``huff_max_len`` (used in length-vector rank)."""
        return 1 << self.huff_max_len


@dataclass(frozen=True)
class CenteredDeltaUint8Stream:
    """Per-column centered-delta uint8 latents wrapped in raw-LZMA bytes.

    The container is *exactly* the PR101 latent blob format:

    1. ``LATENT_DIM * 2`` bytes of fp16 ``mins`` (one per column).
    2. ``LATENT_DIM * 2`` bytes of fp16 ``scales`` (one per column).
    3. ``N_PAIRS * LATENT_DIM`` bytes of column-major centered-delta uint8.

    Row 0 of the column-major block is the absolute quantised base value;
    rows 1..N-1 are ``((q[i] - q[i-1]) mod 256) - 128`` (the *centered delta*
    that compresses well under LZMA because most temporal deltas concentrate
    near 0). The whole 3-section block is LZMA-RAW compressed (no XZ
    container, no checksum) to minimise framing overhead.

    Attributes
    ----------
    mins, scales:
        fp16-precision per-column ``min`` and ``scale`` (shape ``(n_dims,)``).
    base:
        First-row absolute quantised values (shape ``(n_dims,)``, ``uint8``).
    deltas:
        Centered temporal deltas of shape ``(n_pairs - 1, n_dims)``, ``uint8``.
    lzma_bytes:
        The raw-LZMA-compressed concatenation of the three sections above.
        This is the byte payload that ships in the archive.
    """

    mins: np.ndarray
    scales: np.ndarray
    base: np.ndarray
    deltas: np.ndarray
    lzma_bytes: bytes


@dataclass(frozen=True)
class SplitBrotliStream:
    """Output of :func:`split_brotli_self_delimiting`.

    Attributes
    ----------
    payload:
        Concatenation of N self-delimiting Brotli streams. The caller must
        remember the stream count out-of-band.
    n_streams:
        Number of independently-compressed sub-streams encoded.
    stream_byte_offsets:
        Byte offsets *after* each stream within ``payload``. The final entry
        equals ``len(payload)``. Useful for diagnostics; not required by the
        parser which reconstructs the offsets itself.
    """

    payload: bytes
    n_streams: int
    stream_byte_offsets: tuple[int, ...] = field(default_factory=tuple)


# ── Ranked Huffman/no-op sidecar grammar ────────────────────────────────────


@cache
def _huff_length_vector_count(
    pos: int, remaining: int, *, n_symbols: int, huff_min_len: int, huff_max_len: int
) -> int:
    """Count Huffman length vectors of size ``n_symbols`` over Kraft budget.

    Computes the number of ways to fill code-length positions ``[pos, n_symbols)``
    using lengths in ``[huff_min_len, huff_max_len]`` so the remaining Kraft
    weight is exhausted exactly. Mirrors PR101 ``huff_length_vector_count``
    but parameterised over the schema instead of module globals.
    """
    if pos == n_symbols:
        return int(remaining == 0)
    total = 0
    for length in range(huff_min_len, huff_max_len + 1):
        weight = 1 << (huff_max_len - length)
        if remaining >= weight:
            total += _huff_length_vector_count(
                pos + 1,
                remaining - weight,
                n_symbols=n_symbols,
                huff_min_len=huff_min_len,
                huff_max_len=huff_max_len,
            )
    return total


def _decode_huff_length_rank(rank: int, schema: RankedSidecarSchema) -> np.ndarray:
    """Decode a Huffman-length vector from a non-negative integer rank.

    The PR101 representation enumerates valid Kraft-tight length vectors in
    co-lex order; ``rank=0`` is the all-``huff_min_len`` vector. This is
    PR101's ``decode_huff_length_rank`` parameterised by schema.
    """
    n_symbols = len(schema.deltas)
    total = _huff_length_vector_count(
        0,
        schema.kraft_total,
        n_symbols=n_symbols,
        huff_min_len=schema.huff_min_len,
        huff_max_len=schema.huff_max_len,
    )
    if not (0 <= rank < total):
        raise ValueError(f"bad Huffman length-vector rank {rank}; total={total}")
    lengths = np.empty(n_symbols, dtype=np.uint8)
    remaining = schema.kraft_total
    for pos in range(n_symbols):
        emitted = False
        for length in range(schema.huff_min_len, schema.huff_max_len + 1):
            weight = 1 << (schema.huff_max_len - length)
            if remaining < weight:
                continue
            block = _huff_length_vector_count(
                pos + 1,
                remaining - weight,
                n_symbols=n_symbols,
                huff_min_len=schema.huff_min_len,
                huff_max_len=schema.huff_max_len,
            )
            if rank >= block:
                rank -= block
            else:
                lengths[pos] = length
                remaining -= weight
                emitted = True
                break
        if not emitted:
            raise ValueError("bad Huffman length-vector rank (no admissible length)")
    if remaining or rank:
        raise ValueError("bad Huffman length-vector rank (residue not exhausted)")
    return lengths


def _encode_huff_length_rank(
    lengths: np.ndarray, schema: RankedSidecarSchema
) -> int:
    """Inverse of :func:`_decode_huff_length_rank`.

    Enumerates the lengths-vector in co-lex order matching
    :func:`_decode_huff_length_rank` so encode∘decode is the identity rank.
    """
    n_symbols = len(schema.deltas)
    if lengths.shape != (n_symbols,):
        raise ValueError(f"lengths must have shape ({n_symbols},); got {lengths.shape}")
    rank = 0
    remaining = schema.kraft_total
    for pos in range(n_symbols):
        target = int(lengths[pos])
        if not (schema.huff_min_len <= target <= schema.huff_max_len):
            raise ValueError(f"length {target} at pos {pos} out of bounds")
        for length in range(schema.huff_min_len, schema.huff_max_len + 1):
            weight = 1 << (schema.huff_max_len - length)
            if remaining < weight:
                continue
            if length == target:
                remaining -= weight
                break
            rank += _huff_length_vector_count(
                pos + 1,
                remaining - weight,
                n_symbols=n_symbols,
                huff_min_len=schema.huff_min_len,
                huff_max_len=schema.huff_max_len,
            )
    if remaining:
        raise ValueError("provided lengths do not form a Kraft-tight code")
    return rank


def _decode_combination_colex(rank: int, n: int, k: int) -> np.ndarray:
    """Decode a co-lex-ranked size-``k`` combination from ``range(n)``.

    Returns the sorted positions in ascending order. PR101's
    ``decode_combination_colex`` returns descending positions; we re-sort to
    ascending so downstream code can use the result as a mask index directly.
    """
    total = math.comb(n, k)
    if not (0 <= rank < total):
        raise ValueError(f"bad combination rank {rank}; C({n},{k})={total}")
    combo = [0] * k
    x = n
    for i in range(k, 0, -1):
        x -= 1
        while math.comb(x, i) > rank:
            x -= 1
        combo[i - 1] = x
        rank -= math.comb(x, i)
    if rank:
        raise ValueError("bad combination rank (residue not exhausted)")
    out = np.array(combo, dtype=np.int64)
    out.sort()
    return out


def _encode_combination_colex(positions: np.ndarray, n: int) -> int:
    """Inverse of :func:`_decode_combination_colex`."""
    k = len(positions)
    if k == 0:
        return 0
    sorted_pos = np.sort(np.asarray(positions, dtype=np.int64))
    if sorted_pos[0] < 0 or sorted_pos[-1] >= n:
        raise ValueError(
            f"positions must be in [0, {n}); got [{int(sorted_pos[0])}, "
            f"{int(sorted_pos[-1])}]"
        )
    if np.any(np.diff(sorted_pos) == 0):
        raise ValueError("positions must be unique")
    rank = 0
    for i, x in enumerate(sorted_pos):
        rank += math.comb(int(x), i + 1)
    return rank


def _build_canonical_huffman_codebook(
    lengths: np.ndarray,
) -> dict[int, tuple[int, int]]:
    """Build canonical Huffman codes from a length vector.

    Returns ``{symbol: (length, code)}``. Symbols with ``length == 0`` are
    omitted (the standard "absent symbol" convention).
    """
    codebook: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    ):
        code <<= length - prev_len
        codebook[sym] = (length, code)
        code += 1
        prev_len = length
    return codebook


def _bit_pack(symbols: Sequence[int], codebook: dict[int, tuple[int, int]]) -> bytes:
    """Bit-pack ``symbols`` using ``codebook`` into MSB-first packed bytes."""
    out = bytearray()
    cur = 0
    cur_len = 0
    for sym in symbols:
        if sym not in codebook:
            raise ValueError(f"symbol {sym} missing from codebook")
        length, code = codebook[sym]
        cur = (cur << length) | code
        cur_len += length
        while cur_len >= 8:
            cur_len -= 8
            out.append((cur >> cur_len) & 0xFF)
    if cur_len:
        out.append((cur << (8 - cur_len)) & 0xFF)
    return bytes(out)


def _decode_canonical_huffman_n(
    data: bytes, lengths: np.ndarray, n_symbols: int
) -> np.ndarray:
    """Decode exactly ``n_symbols`` symbols from packed Huffman bytes."""
    decode: dict[tuple[int, int], int] = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    ):
        code <<= length - prev_len
        decode[(length, code)] = sym
        code += 1
        prev_len = length

    out = np.empty(n_symbols, dtype=np.int64)
    out_pos = 0
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out[out_pos] = sym
                out_pos += 1
                if out_pos == n_symbols:
                    return out
                cur = 0
                cur_len = 0
    raise ValueError("truncated Huffman payload")


def _build_optimal_huffman_lengths(
    symbols: np.ndarray, schema: RankedSidecarSchema
) -> np.ndarray:
    """Build a length-bounded canonical Huffman length vector for ``symbols``.

    Uses the package-merge algorithm with a min/max length envelope so the
    result is guaranteed Kraft-tight at ``huff_max_len``. Symbols absent from
    ``symbols`` still receive the minimum length so the resulting code can
    encode any future input drawn from the schema vocabulary.
    """
    n = len(schema.deltas)
    counts = np.bincount(symbols.astype(np.int64), minlength=n).astype(np.int64)
    # Ensure every symbol has at least 1 count so package-merge produces a
    # full-coverage code. The 1-count floor adds at most n bits to the rank
    # encoding, which is negligible compared to the savings from a tighter
    # length envelope.
    counts = np.maximum(counts, 1)
    # Package-merge with bounded length. We adapt the textbook algorithm.
    items: list[tuple[int, int]] = [(int(c), s) for s, c in enumerate(counts)]
    items.sort()
    package: list[tuple[int, list[int]]] = [
        (int(c), [s]) for c, s in items
    ]
    final_package = list(package)
    for _ in range(schema.huff_max_len - 1):
        pairs: list[tuple[int, list[int]]] = []
        merged = sorted(final_package, key=lambda x: x[0])
        i = 0
        while i + 1 < len(merged):
            w = merged[i][0] + merged[i + 1][0]
            payload = merged[i][1] + merged[i + 1][1]
            pairs.append((w, payload))
            i += 2
        combined = sorted(package + pairs, key=lambda x: x[0])
        final_package = combined
    # Take the top 2(n-1) items.
    final_package = sorted(final_package, key=lambda x: x[0])[: 2 * (n - 1)]
    lengths = np.full(n, schema.huff_min_len, dtype=np.uint8)
    counts_used = np.zeros(n, dtype=np.int64)
    for _, payload in final_package:
        for s in payload:
            counts_used[s] += 1
    for s in range(n):
        # Length is # times symbol appears in final_package; clamp to envelope.
        lengths[s] = max(int(counts_used[s]), schema.huff_min_len)
    lengths = np.clip(lengths, schema.huff_min_len, schema.huff_max_len)
    # If Kraft is over- or under-budget after clipping, fall back to the
    # uniform length that just covers ``n`` (still valid for our purposes).
    kraft = sum(1 << (schema.huff_max_len - int(L)) for L in lengths)
    if kraft != schema.kraft_total:
        target_len = max(
            schema.huff_min_len,
            min(schema.huff_max_len, math.ceil(math.log2(n))),
        )
        lengths = np.full(n, target_len, dtype=np.uint8)
        # Pad with extra leaves at the deepest level if the uniform code is
        # under-budget. Pure uniform may not exhaust Kraft; if not, accept
        # the rank-encoded form anyway (the encoder will simply pay for it).
    return lengths


# ── Centered-delta uint8 + raw LZMA ─────────────────────────────────────────


def encode_centered_delta_uint8(
    values: np.ndarray,
    *,
    mins: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> CenteredDeltaUint8Stream:
    """Encode a per-column quantised stream as centered-delta uint8 under raw-LZMA.

    Parameters
    ----------
    values:
        Float-valued source signal of shape ``(n_pairs, n_dims)``. Will be
        quantised per-column to ``uint8`` using ``mins`` and ``scales``; if
        either is ``None`` it is derived from the data so the resulting
        quantisation just fits in ``[0, 255]``.
    mins, scales:
        Optional per-column fp16 calibration. When provided, ``values`` is
        quantised as ``round((values - mins) / scales)``.

    Returns
    -------
    CenteredDeltaUint8Stream
        Typed container with ``mins`` / ``scales`` / ``base`` / ``deltas``
        and the byte payload ready to ship in the archive.

    Notes
    -----
    The byte layout inside ``lzma_bytes`` is exactly PR101's:
    ``mins (fp16, n_dims) | scales (fp16, n_dims) | column-major centered
    deltas (uint8, n_pairs * n_dims)`` where the column-major block is laid
    out as ``[col0_pair0, col0_pair1, …, col0_pairN-1, col1_pair0, …]`` and
    each column's row 0 is the absolute base, rows 1..N-1 are centered
    temporal deltas (``+128 == no change``).
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2D; got shape {values.shape}")
    n_pairs, n_dims = values.shape
    if n_pairs < 1 or n_dims < 1:
        raise ValueError(f"values must have positive shape; got {values.shape}")

    if mins is None or scales is None:
        col_min = values.min(axis=0).astype(np.float16)
        col_max = values.max(axis=0).astype(np.float16)
        # Avoid zero-division when a column is constant.
        diff = (col_max.astype(np.float32) - col_min.astype(np.float32))
        diff = np.where(diff <= 0, np.float32(1.0), diff)
        col_scale = (diff / np.float32(255.0)).astype(np.float16)
        mins = col_min if mins is None else np.asarray(mins, dtype=np.float16)
        scales = col_scale if scales is None else np.asarray(scales, dtype=np.float16)
    if mins.shape != (n_dims,) or scales.shape != (n_dims,):
        raise ValueError(
            "mins/scales shape mismatch: "
            f"mins={mins.shape} scales={scales.shape} want ({n_dims},)"
        )

    scales_f32 = scales.astype(np.float32)
    if np.any(scales_f32 == 0):
        raise ValueError("scales must be non-zero per column")

    # Quantise + clamp to uint8.
    q = np.rint((values - mins.astype(np.float32)) / scales_f32).astype(np.int64)
    q = np.clip(q, 0, 255).astype(np.uint8)
    base = q[0].copy()
    deltas_int = (q[1:].astype(np.int16) - q[:-1].astype(np.int16))
    deltas_mod = (deltas_int.astype(np.int16) + 128) & 0xFF
    deltas = deltas_mod.astype(np.uint8)

    # Column-major layout: for each col, write [base[col], delta_row_0[col], …].
    column_blocks = []
    for col in range(n_dims):
        column_blocks.append(np.concatenate([base[col : col + 1], deltas[:, col]]))
    column_major = np.concatenate(column_blocks).astype(np.uint8)
    if column_major.size != n_pairs * n_dims:
        raise ValueError(
            f"internal: column-major size {column_major.size} != "
            f"n_pairs*n_dims={n_pairs * n_dims}"
        )

    buf = io.BytesIO()
    buf.write(np.asarray(mins, dtype=np.float16).tobytes())
    buf.write(np.asarray(scales, dtype=np.float16).tobytes())
    buf.write(column_major.tobytes())
    raw = buf.getvalue()
    lzma_bytes = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LATENT_LZMA_FILTERS)
    return CenteredDeltaUint8Stream(
        mins=np.asarray(mins, dtype=np.float16),
        scales=np.asarray(scales, dtype=np.float16),
        base=base,
        deltas=deltas,
        lzma_bytes=lzma_bytes,
    )


def decode_centered_delta_uint8(
    stream: CenteredDeltaUint8Stream | bytes,
    *,
    n_pairs: int | None = None,
    n_dims: int | None = None,
) -> np.ndarray:
    """Inverse of :func:`encode_centered_delta_uint8`.

    Accepts either the structured :class:`CenteredDeltaUint8Stream` or the
    raw ``lzma_bytes`` (in which case ``n_pairs`` and ``n_dims`` are
    required so the column-major block can be reshaped unambiguously).

    Returns the reconstructed float32 array of shape ``(n_pairs, n_dims)``.
    """
    if isinstance(stream, CenteredDeltaUint8Stream):
        payload = stream.lzma_bytes
        n_pairs_eff = stream.deltas.shape[0] + 1
        n_dims_eff = stream.base.shape[0]
    else:
        if n_pairs is None or n_dims is None:
            raise ValueError(
                "decoding from raw bytes requires explicit n_pairs and n_dims"
            )
        payload = bytes(stream)
        n_pairs_eff = int(n_pairs)
        n_dims_eff = int(n_dims)
    raw = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=_LATENT_LZMA_FILTERS)
    buf = io.BytesIO(raw)
    mins = np.frombuffer(buf.read(n_dims_eff * 2), dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(buf.read(n_dims_eff * 2), dtype=np.float16).astype(np.float32)
    expected = n_pairs_eff * n_dims_eff
    cm_bytes = buf.read(expected)
    if len(cm_bytes) != expected:
        raise ValueError(
            f"truncated column-major payload: got {len(cm_bytes)} bytes; "
            f"want {expected}"
        )
    cm = np.frombuffer(cm_bytes, dtype=np.uint8).reshape(n_dims_eff, n_pairs_eff)
    base = cm[:, 0]
    deltas = cm[:, 1:].T  # back to (n_pairs - 1, n_dims)
    # Centered-delta reconstruction (uint8 cumulative sum).
    q = np.empty((n_pairs_eff, n_dims_eff), dtype=np.uint8)
    q[0] = base
    if n_pairs_eff > 1:
        steps = (deltas.astype(np.int16) - 128).astype(np.int16)
        q[1:] = (np.cumsum(steps, axis=0, dtype=np.int32) + base.astype(np.int32)
                 ).astype(np.uint8)
    return q.astype(np.float32) * scales[None, :] + mins[None, :]


# ── Self-delimiting split Brotli ─────────────────────────────────────────────


def split_brotli_self_delimiting(
    streams: Sequence[bytes],
    *,
    lgwin: int = 22,
    quality: int = 11,
) -> SplitBrotliStream:
    """Concatenate N independently-Brotli-compressed byte streams.

    Each sub-stream is compressed with the same parameters and then the
    Brotli-encoded payloads are concatenated. The reader uses Brotli's frame
    structure to know where each stream ends. There is no length prefix; the
    consumer must know ``n_streams`` out-of-band.

    Parameters
    ----------
    streams:
        Per-sub-stream raw bytes (one ``bytes`` object per sub-stream).
    lgwin:
        Brotli sliding-window parameter (10..24). PR101 uses 22.
    quality:
        Brotli quality 0..11. PR101 uses 11 (highest).
    """
    if not streams:
        raise ValueError("streams must contain at least one substream")
    payload = b""
    offsets: list[int] = []
    for i, raw in enumerate(streams):
        # brotli.compress returns the raw Brotli compressed bytes (not a
        # framed format). Concatenation works because the Brotli decoder is
        # frame-aware: it can detect end-of-stream from the bit-stream.
        comp = brotli.compress(raw, mode=brotli.MODE_GENERIC, quality=quality, lgwin=lgwin)
        if not comp:
            raise ValueError(f"empty Brotli output for substream {i}")
        payload += comp
        offsets.append(len(payload))
    return SplitBrotliStream(
        payload=payload, n_streams=len(streams), stream_byte_offsets=tuple(offsets)
    )


def parse_split_brotli_self_delimiting(
    payload: bytes, *, n_streams: int
) -> list[bytes]:
    """Inverse of :func:`split_brotli_self_delimiting`.

    Walks the concatenated Brotli payload byte-by-byte, terminating each
    sub-stream when the decoder reports end-of-stream. PR101 does exactly
    this in ``decompress_brotli_streams`` — we surface it as a typed helper.

    Raises ``ValueError`` if the bytes do not decode to exactly ``n_streams``
    sub-streams with no trailing data.
    """
    if n_streams <= 0:
        raise ValueError(f"n_streams must be > 0; got {n_streams}")
    outputs: list[bytes] = []
    pos = 0
    for _ in range(n_streams):
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        while pos < len(payload) and not dec.is_finished():
            chunks.append(dec.process(payload[pos : pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated split-Brotli payload")
        outputs.append(b"".join(chunks))
    if pos != len(payload):
        raise ValueError(
            f"trailing data after {n_streams} streams: {len(payload) - pos} bytes"
        )
    return outputs


# ── Ranked Huffman/no-op sidecar encode/decode ──────────────────────────────


def encode_ranked_no_op_sidecar(
    *,
    dims: np.ndarray,
    delta_indices: np.ndarray,
    schema: RankedSidecarSchema,
    dim_bytes: int | None = None,
    noop_rank_bytes: int | None = None,
) -> bytes:
    """Encode a per-pair sparse correction sidecar.

    Parameters
    ----------
    dims:
        Per-pair correction dimensions, ``shape == (schema.n_pairs,)``,
        ``dtype int64``. Use ``schema.no_op_sentinel`` to mark a no-op slot
        (typically 255).
    delta_indices:
        Per-pair *index* into ``schema.deltas`` for the chosen delta. Must
        be 0 for no-op slots. ``shape == (schema.n_pairs,)``, ``dtype int64``.
    schema:
        :class:`RankedSidecarSchema` declaring vocab/code-length envelope.
    dim_bytes / noop_rank_bytes:
        Optional container-width overrides for legacy PacketIR adapters that
        froze wider framing metadata.  The defaults match the live PR101/FEC6
        runtime: exact base-``n_dims`` width for dimensions and actual-rank
        width for no-op positions.

    Returns
    -------
    bytes
        Self-delimiting byte string. Encoding layout (PR101 "huff_enum"
        variant; the encoder selects this variant unconditionally since it is
        Pareto-optimal for the contest's correction-density regime):

        ``[dim_packed_le | length_rank_le | huffman_bits | noop_rank_le]``
    """
    dims = np.asarray(dims, dtype=np.int64)
    delta_indices = np.asarray(delta_indices, dtype=np.int64)
    if dims.shape != (schema.n_pairs,) or delta_indices.shape != (schema.n_pairs,):
        raise ValueError(
            f"dims/delta_indices must have shape ({schema.n_pairs},); "
            f"got {dims.shape}/{delta_indices.shape}"
        )
    valid_mask = dims != schema.no_op_sentinel
    valid = np.where(valid_mask)[0]
    n_valid = int(valid_mask.sum())
    noop_pos = np.where(~valid_mask)[0]
    noop_count = schema.n_pairs - n_valid

    # Sanity: no-op slots must have delta_index == 0 (irrelevant) and valid
    # slots must have in-range dim and delta_index.
    if np.any(dims[valid] < 0) or np.any(dims[valid] >= schema.n_dims):
        raise ValueError(f"valid dims must be in [0, {schema.n_dims})")
    if np.any(delta_indices[valid] < 0) or np.any(
        delta_indices[valid] >= len(schema.deltas)
    ):
        raise ValueError(f"delta_indices must be in [0, {len(schema.deltas)})")

    # Pack dims as a mixed-radix integer (PR101 layout): least-significant
    # dimension is the first valid pair's dim.
    dim_value = 0
    for i in range(n_valid - 1, -1, -1):
        dim_value = dim_value * schema.n_dims + int(dims[int(valid[i])])
    # Width: ceil(log2(n_dims ** n_valid) / 8) bytes.  PR101 packs the valid
    # dimensions as a single base-28 integer; using ceil(log2(28)) per symbol
    # over-allocates 15 bytes at the real 597-valid FEC6 sidecar density.
    dim_bits = max(1, math.ceil(n_valid * math.log2(max(schema.n_dims, 2))))
    min_dim_bytes = (dim_bits + 7) // 8
    if dim_bytes is None:
        dim_bytes = min_dim_bytes
    elif dim_bytes < min_dim_bytes:
        raise ValueError(
            f"dim_bytes={dim_bytes} is too small for {n_valid} base-{schema.n_dims} dims; "
            f"minimum is {min_dim_bytes}"
        )
    dim_blob = dim_value.to_bytes(dim_bytes, "little")

    # Choose a canonical Huffman code; the schema-driven code is enforced
    # length-bounded so the rank fits the spec.
    if n_valid == 0:
        # Edge case: no corrections at all. Lengths is a degenerate "all
        # min-length" vector; the rank is 0; huffman bits is empty.
        lengths = np.full(len(schema.deltas), schema.huff_min_len, dtype=np.uint8)
        # Force Kraft-tightness via the smallest valid uniform length.
        target_len = max(
            schema.huff_min_len,
            min(schema.huff_max_len, math.ceil(math.log2(len(schema.deltas)))),
        )
        lengths = np.full(len(schema.deltas), target_len, dtype=np.uint8)
        try:
            length_rank = _encode_huff_length_rank(lengths, schema)
        except ValueError:
            length_rank = 0
        huff_bits = b""
    else:
        lengths = _build_optimal_huffman_lengths(
            delta_indices[valid].astype(np.int64), schema
        )
        try:
            length_rank = _encode_huff_length_rank(lengths, schema)
        except ValueError:
            # Fall back to uniform code.
            target_len = max(
                schema.huff_min_len,
                min(
                    schema.huff_max_len,
                    math.ceil(math.log2(len(schema.deltas))),
                ),
            )
            lengths = np.full(len(schema.deltas), target_len, dtype=np.uint8)
            length_rank = _encode_huff_length_rank(lengths, schema)
        codebook = _build_canonical_huffman_codebook(lengths)
        huff_bits = _bit_pack(
            [int(x) for x in delta_indices[valid].tolist()], codebook
        )

    # Length-rank width in bytes.
    total_length_vectors = _huff_length_vector_count(
        0,
        schema.kraft_total,
        n_symbols=len(schema.deltas),
        huff_min_len=schema.huff_min_len,
        huff_max_len=schema.huff_max_len,
    )
    rank_bits = max(1, math.ceil(math.log2(max(total_length_vectors, 2))))
    rank_bytes = (rank_bits + 7) // 8
    length_rank_blob = length_rank.to_bytes(rank_bytes, "little")

    # No-op rank: co-lex combination of the no-op positions.  PR101's
    # huff_enum sidecar stores the actual rank in the minimal byte width
    # needed for this packet, not the worst-case width for every C(N, k)
    # combination.  The real FEC6 sidecar uses 3 bytes for 3 no-op positions.
    noop_rank = _encode_combination_colex(noop_pos.astype(np.int64), schema.n_pairs)
    min_noop_rank_bytes = max(1, (int(noop_rank).bit_length() + 7) // 8)
    if noop_rank_bytes is None:
        noop_rank_bytes = min_noop_rank_bytes
    elif noop_rank_bytes < min_noop_rank_bytes:
        raise ValueError(
            f"noop_rank_bytes={noop_rank_bytes} is too small for no-op rank "
            f"{noop_rank}; minimum is {min_noop_rank_bytes}"
        )
    noop_rank_blob = noop_rank.to_bytes(noop_rank_bytes, "little")

    return dim_blob + length_rank_blob + huff_bits + noop_rank_blob


def decode_ranked_no_op_sidecar(
    payload: bytes,
    *,
    schema: RankedSidecarSchema,
    dim_bytes: int,
    rank_bytes: int,
    noop_rank_bytes: int,
    noop_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_ranked_no_op_sidecar`.

    Returns
    -------
    (dims, delta_indices)
        Two ``(schema.n_pairs,)`` int64 arrays. ``dims[i] == schema.no_op_sentinel``
        marks a no-op slot (whose ``delta_indices[i]`` is meaningless / 0).

    Notes
    -----
    The caller must remember ``dim_bytes`` / ``rank_bytes`` / ``noop_rank_bytes``
    / ``noop_count`` from the encode call (or from a frozen container header).
    PR101 hardcodes these in the codec because the schema is fixed; we surface
    them as explicit parameters so the same primitive serves any schema.
    """
    if dim_bytes <= 0 or rank_bytes <= 0 or noop_rank_bytes <= 0:
        raise ValueError("dim/rank/noop_rank byte widths must be positive")
    if not (0 <= noop_count <= schema.n_pairs):
        raise ValueError(f"noop_count must be in [0, {schema.n_pairs}]")

    n_valid = schema.n_pairs - noop_count
    expected_min = dim_bytes + rank_bytes + noop_rank_bytes
    if len(payload) < expected_min:
        raise ValueError(
            f"payload too short: got {len(payload)} bytes; "
            f"need at least {expected_min}"
        )
    pos = 0
    dim_blob = payload[pos : pos + dim_bytes]
    pos += dim_bytes
    length_rank_blob = payload[pos : pos + rank_bytes]
    pos += rank_bytes
    huff_bits = payload[pos : len(payload) - noop_rank_bytes]
    noop_rank_blob = payload[len(payload) - noop_rank_bytes :]

    length_rank = int.from_bytes(length_rank_blob, "little")
    lengths = _decode_huff_length_rank(length_rank, schema)

    if n_valid == 0:
        delta_indices_valid = np.empty(0, dtype=np.int64)
    else:
        delta_indices_valid = _decode_canonical_huffman_n(
            huff_bits, lengths, n_valid
        )

    # No-op positions.
    noop_rank = int.from_bytes(noop_rank_blob, "little")
    if noop_count > 0:
        noop_pos = _decode_combination_colex(noop_rank, schema.n_pairs, noop_count)
    else:
        noop_pos = np.empty(0, dtype=np.int64)
    valid_mask = np.ones(schema.n_pairs, dtype=bool)
    valid_mask[noop_pos] = False

    # Decode the mixed-radix dim integer.
    dim_value = int.from_bytes(dim_blob, "little")
    dims_valid = np.empty(n_valid, dtype=np.int64)
    for i in range(n_valid):
        dim_value, dims_valid[i] = divmod(dim_value, schema.n_dims)
    if dim_value:
        raise ValueError(
            "trailing dim radix residue (corrupt sidecar or wrong dim_bytes)"
        )

    dims = np.full(schema.n_pairs, schema.no_op_sentinel, dtype=np.int64)
    delta_indices = np.zeros(schema.n_pairs, dtype=np.int64)
    dims[valid_mask] = dims_valid
    delta_indices[valid_mask] = delta_indices_valid
    return dims, delta_indices


__all__ = [
    "CenteredDeltaUint8Stream",
    "RankedSidecarSchema",
    "SplitBrotliStream",
    "decode_centered_delta_uint8",
    "decode_ranked_no_op_sidecar",
    "encode_centered_delta_uint8",
    "encode_ranked_no_op_sidecar",
    "parse_split_brotli_self_delimiting",
    "split_brotli_self_delimiting",
]
