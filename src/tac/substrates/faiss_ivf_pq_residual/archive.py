# SPDX-License-Identifier: MIT
"""faiss_ivf_pq_residual.archive — FAISSPQ1 monolithic single-file ``0.bin`` grammar.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic
0.bin) + L4 (≤200 LOC inflate substrate-engineering waiver) + L8
(deterministic).

L0 SCAFFOLD scope: the grammar defined here is the canonical contract;
the PyTorch port for inflate-time consumption is in ``inflate.py``. The
archive grammar is byte-deterministic from MLX/numpy-trained codebooks
via the canonical write/parse round-trip implemented below.

Grammar:

::

    MAGIC(5)              b"FQP1\\x00"
    VERSION(1)            u8       schema version (currently 1)
    M_SUB(1)              u8       PQ M parameter (1-16)
    KSUB(2)               u16      PQ ksub parameter (2-65535)
    TILE_H(2)             u16      cfg.tile_h
    TILE_W(2)             u16      cfg.tile_w
    TILES_PER_PAIR(2)     u16      grid_h × grid_w
    NUM_PAIRS(2)          u16      cfg.num_pairs (600 canonical)
    CODEBOOK_BLOB_LEN(4)  u32      brotli(q=9) of float32 codebook
    CODEWORD_BLOB_LEN(4)  u32      brotli(q=9) of uint16 codeword stream
    META_BLOB_LEN(4)      u32      sorted-keys JSON utf-8 bytes
    CODEBOOK_BLOB         ...      brotli of M × ksub × sub_dim float32
    CODEWORD_BLOB         ...      brotli of NUM_PAIRS × TILES_PER_PAIR × M uint16
    META_BLOB             ...      sorted-keys JSON utf-8

The codebook + codeword stream are BOTH inflate-time consumers per HNeRV
parity L1 (per-pair RGB residual reconstruction adds to PR110 fec6
frontier output). The codebook is shared across all 600 pairs (NOT per-pair).

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load at inflate
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli quality;
  float32 codebook IEEE-754 stable; uint16 codewords raw little-endian)

Per Catalog #139 + #105 + #272 distinguishing-feature contract:
- codebook bytes ARE frame-affecting at inflate (per-tile PQ centroid gather
  consumed; THIS IS THE DISTINGUISHING FEATURE vs single-decoder NeRV-family)
- codeword bytes ARE frame-affecting at inflate (per-pair PQ codeword indices
  consumed)
- header + meta bytes are parse/config gates (control_or_metadata role)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np


FAISSPQ1_MAGIC: bytes = b"FQP1\x00"
"""Faiss IVF-PQ residual codec variant 1 archive magic."""

FAISSPQ1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout per docstring:
# MAGIC(5) + VERSION(1) + M_SUB(1) + KSUB(2) + TILE_H(2) + TILE_W(2) +
# TILES_PER_PAIR(2) + NUM_PAIRS(2) + CODEBOOK_LEN(4) + CODEWORD_LEN(4) +
# META_LEN(4)
FAISSPQ1_HEADER_FMT: str = "<5sBBHHHHHIII"
FAISSPQ1_HEADER_SIZE: int = struct.calcsize(FAISSPQ1_HEADER_FMT)
# Layout: 5sBBHHHHHIII = 5+1+1+2+2+2+2+2+4+4+4 = 29 bytes
assert FAISSPQ1_HEADER_SIZE == 29, (
    f"FAISSPQ1 header size invariant: expected 29, got {FAISSPQ1_HEADER_SIZE}"
)

# Deterministic brotli quality (matches sister substrate canonical pattern:
# C6 IBPS / sane_hnerv / WZF01 / boost_nerv_pr110_residual / dreamer_v3_rssm /
# nirvana_cascading_nerv).
_BROTLI_QUALITY: int = 9


class FaissIVFPQResidualArchiveError(ValueError):
    """Raised when a FAISSPQ1 archive fails to parse or validate."""


@dataclass(frozen=True)
class FaissIVFPQResidualArchive:
    """Parsed FAISSPQ1 archive — the inflate-time data contract."""

    codebook: np.ndarray
    """Shape (M, ksub, sub_dim) float32 — PQ codebook centroids."""

    per_pair_codewords: np.ndarray
    """Shape (num_pairs, tiles_per_pair, M) uint16 — per-pair PQ codeword indices."""

    meta: dict[str, Any]
    """Sidecar JSON meta: residual_scale, num_pairs, eval_hw, tile_size, ..."""

    schema_version: int
    m_sub_quantizers: int
    ksub_codebook_size: int
    tile_h: int
    tile_w: int
    tiles_per_pair: int
    num_pairs: int


def build_archive_bytes(
    codebook: np.ndarray,
    per_pair_codewords: np.ndarray,
    *,
    tile_h: int,
    tile_w: int,
    meta: dict[str, Any] | None = None,
) -> bytes:
    """Build deterministic FAISSPQ1 archive bytes.

    Args:
        codebook: shape (M, ksub, sub_dim) float32
        per_pair_codewords: shape (num_pairs, tiles_per_pair, M) uint16
        tile_h: per-tile height
        tile_w: per-tile width
        meta: optional sidecar JSON dict; auto-populated with canonical keys

    Returns:
        Byte-deterministic FAISSPQ1 archive bytes.
    """
    if codebook.dtype != np.float32:
        raise FaissIVFPQResidualArchiveError(
            f"codebook dtype must be float32; got {codebook.dtype}"
        )
    if codebook.ndim != 3:
        raise FaissIVFPQResidualArchiveError(
            f"codebook ndim must be 3 (M, ksub, sub_dim); got {codebook.ndim}"
        )
    if per_pair_codewords.dtype != np.uint16:
        raise FaissIVFPQResidualArchiveError(
            f"per_pair_codewords dtype must be uint16; got {per_pair_codewords.dtype}"
        )
    if per_pair_codewords.ndim != 3:
        raise FaissIVFPQResidualArchiveError(
            f"per_pair_codewords ndim must be 3 (num_pairs, tiles_per_pair, M); "
            f"got {per_pair_codewords.ndim}"
        )
    M, ksub, sub_dim = codebook.shape
    num_pairs, tiles_per_pair, M_check = per_pair_codewords.shape
    if M != M_check:
        raise FaissIVFPQResidualArchiveError(
            f"codebook M={M} != codewords M={M_check}"
        )
    if M > 255 or ksub > 65535 or tile_h > 65535 or tile_w > 65535:
        raise FaissIVFPQResidualArchiveError(
            f"params out of range: M={M} ksub={ksub} tile_h={tile_h} tile_w={tile_w}"
        )
    if tiles_per_pair > 65535 or num_pairs > 65535:
        raise FaissIVFPQResidualArchiveError(
            f"counts out of range: tiles_per_pair={tiles_per_pair} num_pairs={num_pairs}"
        )

    # Serialize codebook (float32 little-endian raw bytes; brotli compressed)
    codebook_raw = codebook.astype(np.float32).tobytes(order="C")
    codebook_blob = brotli.compress(codebook_raw, quality=_BROTLI_QUALITY)

    # Serialize codeword stream (uint16 little-endian raw bytes; brotli compressed)
    codeword_raw = per_pair_codewords.astype(np.uint16).tobytes(order="C")
    codeword_blob = brotli.compress(codeword_raw, quality=_BROTLI_QUALITY)

    # Serialize meta JSON with deterministic sorted keys
    canonical_meta: dict[str, Any] = {
        "codec": "faiss_ivf_pq_residual_v1",
        "eval_hw": [int(tile_h * (per_pair_codewords.shape[1] // (tile_w // tile_w if tile_w else 1))), 0],
        "m_sub_quantizers": int(M),
        "ksub_codebook_size": int(ksub),
        "num_pairs": int(num_pairs),
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "research_only": True,
        "dispatch_enabled": False,
        "score_claim": False,
        "sub_dim": int(sub_dim),
        "tile_h": int(tile_h),
        "tile_w": int(tile_w),
        "tiles_per_pair": int(tiles_per_pair),
    }
    if meta:
        # Caller-provided meta overrides canonical defaults (allows residual_scale etc.)
        canonical_meta.update(meta)
    meta_blob = json.dumps(
        canonical_meta, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    header = struct.pack(
        FAISSPQ1_HEADER_FMT,
        FAISSPQ1_MAGIC,
        FAISSPQ1_SCHEMA_VERSION,
        int(M),
        int(ksub),
        int(tile_h),
        int(tile_w),
        int(tiles_per_pair),
        int(num_pairs),
        len(codebook_blob),
        len(codeword_blob),
        len(meta_blob),
    )

    return header + codebook_blob + codeword_blob + meta_blob


def parse_archive(data: bytes) -> FaissIVFPQResidualArchive:
    """Parse a complete FAISSPQ1 archive byte stream.

    Args:
        data: complete archive bytes (header + 3 blobs)

    Returns:
        Parsed FaissIVFPQResidualArchive dataclass.

    Raises FaissIVFPQResidualArchiveError on magic/version/length mismatch.
    """
    if len(data) < FAISSPQ1_HEADER_SIZE:
        raise FaissIVFPQResidualArchiveError(
            f"archive too short for FAISSPQ1 header: {len(data)} < {FAISSPQ1_HEADER_SIZE}"
        )
    (
        magic,
        version,
        m_sub,
        ksub,
        tile_h,
        tile_w,
        tiles_per_pair,
        num_pairs,
        codebook_len,
        codeword_len,
        meta_len,
    ) = struct.unpack(FAISSPQ1_HEADER_FMT, data[:FAISSPQ1_HEADER_SIZE])
    if magic != FAISSPQ1_MAGIC:
        raise FaissIVFPQResidualArchiveError(
            f"FAISSPQ1 magic mismatch: got {magic!r}, expected {FAISSPQ1_MAGIC!r}"
        )
    if version != FAISSPQ1_SCHEMA_VERSION:
        raise FaissIVFPQResidualArchiveError(
            f"FAISSPQ1 version mismatch: got {version}, expected {FAISSPQ1_SCHEMA_VERSION}"
        )
    expected_total = (
        FAISSPQ1_HEADER_SIZE + codebook_len + codeword_len + meta_len
    )
    if len(data) != expected_total:
        raise FaissIVFPQResidualArchiveError(
            f"FAISSPQ1 archive length mismatch: {len(data)} != {expected_total}"
        )

    cursor = FAISSPQ1_HEADER_SIZE
    codebook_blob = data[cursor:cursor + codebook_len]
    cursor += codebook_len
    codeword_blob = data[cursor:cursor + codeword_len]
    cursor += codeword_len
    meta_blob = data[cursor:cursor + meta_len]

    # Decompress + reshape codebook
    codebook_raw = brotli.decompress(codebook_blob)
    sub_dim_raw = len(codebook_raw) // (4 * m_sub * ksub)  # 4 bytes/float32
    if sub_dim_raw * 4 * m_sub * ksub != len(codebook_raw):
        raise FaissIVFPQResidualArchiveError(
            f"codebook decompressed length {len(codebook_raw)} doesn't divide "
            f"into M={m_sub} × ksub={ksub} × float32"
        )
    codebook = np.frombuffer(codebook_raw, dtype=np.float32).reshape(
        m_sub, ksub, sub_dim_raw
    ).copy()  # copy from read-only buffer

    # Decompress + reshape codeword stream
    codeword_raw = brotli.decompress(codeword_blob)
    expected_codeword_bytes = num_pairs * tiles_per_pair * m_sub * 2  # 2 bytes/uint16
    if len(codeword_raw) != expected_codeword_bytes:
        raise FaissIVFPQResidualArchiveError(
            f"codeword decompressed length {len(codeword_raw)} != expected "
            f"{expected_codeword_bytes} for ({num_pairs}, {tiles_per_pair}, {m_sub})"
        )
    per_pair_codewords = np.frombuffer(codeword_raw, dtype=np.uint16).reshape(
        num_pairs, tiles_per_pair, m_sub
    ).copy()

    # Parse meta JSON
    meta = json.loads(meta_blob.decode("utf-8"))

    return FaissIVFPQResidualArchive(
        codebook=codebook,
        per_pair_codewords=per_pair_codewords,
        meta=meta,
        schema_version=version,
        m_sub_quantizers=m_sub,
        ksub_codebook_size=ksub,
        tile_h=tile_h,
        tile_w=tile_w,
        tiles_per_pair=tiles_per_pair,
        num_pairs=num_pairs,
    )


__all__ = [
    "FAISSPQ1_HEADER_FMT",
    "FAISSPQ1_HEADER_SIZE",
    "FAISSPQ1_MAGIC",
    "FAISSPQ1_SCHEMA_VERSION",
    "FaissIVFPQResidualArchive",
    "FaissIVFPQResidualArchiveError",
    "build_archive_bytes",
    "parse_archive",
]
