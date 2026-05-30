# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding archive grammar — Z8HPC1 monolithic ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (≤200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

L0 SCAFFOLD scope: the grammar defined here is the canonical contract; the
PyTorch port for inflate-time consumption is queued per Path 3 cascade. The
archive grammar is byte-deterministic from MLX-trained weights via the
canonical write/parse round-trip implemented below.

Grammar (per design memo Section 8):

::

    HEADER (62 bytes; fixed-layout struct):
      MAGIC(8)              b"Z8HPC1\\x00\\x00"  # Z8 Hierarchical Predictive Coding v1
      VERSION(1)            u8       schema version (currently 1)
      NUM_LEVELS(1)         u8       hierarchy depth (canonical 3)
      G_PER_LEVEL(3)        u8[3]    categorical groups per level (e.g. 24/16/8)
      K_PER_LEVEL(6)        u16[3]   alphabet size per level (e.g. 256/128/64)
      NUM_PAIRS(2)          u16      contest pair count (e.g. 600)
      DECODER_LATENT(2)     u16      decoder input latent dim (e.g. 28)
      BASE_CHANNELS(2)      u16      decoder base channels (e.g. 24)
      WAVELET_BASIS_ID(1)   u8       wavelet basis (0=Daubechies-4)
      DECODER_BLOB_LEN(4)   u32      brotli-compressed decoder weights len
      INDICES_BLOB_LEN(4)   u32      packed per-pair per-level cat indices len
      WAVELET_BLOB_LEN(4)   u32      per-level Mallat detail-band coeffs len
      WYNER_ZIV_BLOB_LEN(4) u32      Wyner-Ziv top-level coded len
      DREAMER_STATE_LEN(4)  u32      DreamerV3 GRU deterministic state init len
      META_BLOB_LEN(4)      u32      sorted-keys JSON utf-8 bytes len
      RESERVED(12)          u8[12]   zero-padded for future expansion

    DECODER_BLOB        brotli(quality=9) of multi-level decoder + cat_proj weights (fp16)
    INDICES_BLOB        packed per-pair per-level categorical indices (u8 for K<=256)
    WAVELET_BLOB        per-level Mallat detail-band Daubechies-CDF coded (L0: brotli-only)
    WYNER_ZIV_BLOB      Wyner-Ziv top-level coded (frame_0 side-info conditional; L0: brotli-only)
    DREAMER_STATE_BLOB  DreamerV3 GRU deterministic + stochastic state init bytes (fp16+brotli)
    META_BLOB           sorted-keys JSON: {"wavelet_basis", "categorical_bits_per_level", ...}

The multi-level decoder + cat_to_continuous projections state_dict is the
inflate-time consumer. The per-pair per-level learned logits (training only)
are REDUCED to argmax indices for archive serialization (canonical sister A
DreamerV3 pattern; HNeRV parity L4 budget discipline).

Per Catalog #139 + #105 + #272 distinguishing-feature contract:
- decoder_blob bytes ARE frame-affecting at inflate (multi-level decoder consumed)
- indices_blob bytes ARE frame-affecting at inflate (per-level cat samples consumed;
  DISTINGUISHING FEATURE #1 — DreamerV3 RSSM per-level)
- wavelet_blob bytes ARE frame-affecting at inflate (per-level Mallat inverse;
  DISTINGUISHING FEATURE #2 — Mallat wavelet codec)
- wyner_ziv_blob bytes ARE frame-affecting at inflate (top-level WZ decode consumed;
  DISTINGUISHING FEATURE #3 — Wyner-Ziv side-info coder)
- dreamer_state_blob bytes ARE frame-affecting at inflate (GRU state init consumed;
  DISTINGUISHING FEATURE #4 — DreamerV3 latent dynamics)
- header + meta bytes are parse/config gates (control_or_metadata role)

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load at inflate
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli quality;
  fp16 state_dict cast on CPU; raw bytes for category indices)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np


Z8HPC1_MAGIC: bytes = b"Z8HPC1\x00\x00"
"""Z8 Hierarchical Predictive Coding variant 1 archive magic (8 bytes)."""

Z8HPC1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout per docstring:
# MAGIC(8) + VERSION(1) + NUM_LEVELS(1) + G_PER_LEVEL(3)
# + K_PER_LEVEL(6: 3xH) + NUM_PAIRS(2) + DECODER_LATENT(2) + BASE_CHANNELS(2)
# + WAVELET_BASIS_ID(1) + DECODER_BLOB_LEN(4) + INDICES_BLOB_LEN(4)
# + WAVELET_BLOB_LEN(4) + WYNER_ZIV_BLOB_LEN(4) + DREAMER_STATE_LEN(4)
# + META_BLOB_LEN(4) + RESERVED(12)
# = 8+1+1+3+(2+2+2)+2+2+2+1+4+4+4+4+4+4+12 = 62 bytes
#
# Format: 6 H fields total (k0/k1/k2 + num_pairs + decoder_latent_dim + base_channels)
# We fix NUM_LEVELS=3 in the grammar (canonical Rao-Ballard); higher counts
# would require schema_version bump.
Z8HPC1_HEADER_FMT: str = "<8sBB3sHHHHHHBIIIIII12s"
Z8HPC1_HEADER_SIZE: int = struct.calcsize(Z8HPC1_HEADER_FMT)
assert Z8HPC1_HEADER_SIZE == 62, (
    f"Z8HPC1 header size invariant: expected 62, got {Z8HPC1_HEADER_SIZE}"
)

# Deterministic brotli quality. Bumped from q=9 to canonical PR95-family L32
# (``pr95_family_l32_brotli_quality_11_max_v1``) value q=11 per the canonical
# PR-or-greater binding-depth standing directive 2026-05-30 + the operator-routed
# Yousfi-cascade TOP-1 quick-wins bolt-on cascade. Per CLAUDE.md HNeRV parity
# discipline L32: "5-10% per-pair byte savings; compression time is offline
# overhead so quality=11 is free at deploy time."
#
# Companion brotli q=11 bump lives in
# ``canonical_quadruple_binding._serialize_pair_wavelet_pyramid`` (covers
# 99.5% of the archive bytes — per-pair wavelet pyramid blobs). The q=11
# bump here covers the decoder_blob + dreamer_state_blob sub-surfaces
# (~0.07% of total archive bytes at L1 SCAFFOLD scale but the bump is
# canonical and applies uniformly per PR101 L32 precedent).
_BROTLI_QUALITY: int = 11

# Canonical hierarchy depth at this schema version. Schema bump required to
# change. Per design memo Section 8.
_CANONICAL_NUM_LEVELS: int = 3


@dataclass(frozen=True)
class Z8HierarchicalArchive:
    """Parsed Z8HPC1 archive — the inflate-time data contract."""

    decoder_state_dict: dict[str, Any]
    """Multi-level decoder + cat_to_continuous_per_level state_dict (inflate consumer)."""

    per_level_category_indices: list[np.ndarray]
    """List of ``(num_pairs, num_groups_l)`` int32 per level (decoded from u8/u16
    archive bytes; DISTINGUISHING FEATURE per Catalog #272)."""

    wavelet_coeffs_blob: bytes
    """Per-level Mallat wavelet detail-band coefficient bytes (DISTINGUISHING
    FEATURE per Catalog #272; L0: opaque brotli-coded bytes; Phase 2 parses to
    typed per-level wavelet coefficient arrays)."""

    wyner_ziv_top_blob: bytes
    """Wyner-Ziv top-level coded bytes against frame_0 side-info (DISTINGUISHING
    FEATURE per Catalog #272; L0: opaque brotli-coded bytes; Phase 2 parses to
    typed top-level Wyner-Ziv conditional probability rows)."""

    dreamer_state_blob: dict[str, Any]
    """DreamerV3 GRU deterministic + stochastic state init state_dict
    (DISTINGUISHING FEATURE per Catalog #272)."""

    meta: dict[str, Any]
    """Sidecar JSON meta: wavelet_basis, categorical_bits_per_level, ..."""

    schema_version: int
    num_levels: int
    num_groups_per_level: tuple[int, ...]
    num_categories_per_level: tuple[int, ...]
    num_pairs: int
    decoder_latent_dim: int
    base_channels: int
    wavelet_basis_id: int


def _serialize_state_dict_fp16(sd: dict[str, Any]) -> bytes:
    """Serialize a state_dict deterministically as length-prefixed records.

    Sister A=DreamerV3 canonical pattern reuse per Catalog #290
    ADOPT_CANONICAL_BECAUSE_SERVES decision; identical mathematical structure
    at the state_dict serialization layer (the substrate-class shift is at
    the latent/grammar layer, not the weights-serialization layer).

    Format per entry (sorted by key for determinism):

        u16 key_len + key_bytes (utf-8)
        u8 ndim
        u32 shape[0..ndim-1] (little-endian)
        fp16 tensor bytes (row-major contiguous, numpy)

    Returns brotli-compressed concatenation.
    """
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key]
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(tensor, dtype=np.float32)
        arr16 = arr.astype(np.float16)
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} too long for u16 length")
        shape = tuple(int(s) for s in arr16.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} has too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(header_fmt, len(key_bytes), key_bytes, len(shape), *shape)
        parts.append(header)
        parts.append(arr16.tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_state_dict_fp16(blob: bytes) -> dict[str, np.ndarray]:
    """Inverse of ``_serialize_state_dict_fp16``. Returns numpy fp16 arrays.

    Caller can cast to torch.Tensor or mlx.array as needed.
    """
    raw = brotli.decompress(blob)
    sd: dict[str, np.ndarray] = {}
    pos = 0
    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + ndim * 4 > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + ndim * 4])
        pos += ndim * 4
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor_bytes = numel * 2  # fp16
        if pos + tensor_bytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        arr = np.frombuffer(
            raw[pos : pos + tensor_bytes], dtype=np.float16
        ).reshape(shape).astype(np.float16, copy=True)
        sd[key] = arr
        pos += tensor_bytes
    if pos != len(raw):
        raise ValueError(
            f"state_dict blob has trailing bytes (pos={pos} len={len(raw)})"
        )
    return sd


def _pack_per_level_indices(
    per_level_indices: list[np.ndarray],
    *,
    num_groups_per_level: tuple[int, ...],
    num_categories_per_level: tuple[int, ...],
    num_pairs: int,
) -> bytes:
    """Pack per-level categorical indices into a single concatenated blob.

    Per level, indices shape ``(num_pairs, num_groups_l)`` are packed as u8
    (K<=256) or u16 (K<=65536). Levels are concatenated in order [level_0,
    level_1, ..., level_{L-1}].
    """
    if len(per_level_indices) != len(num_groups_per_level):
        raise ValueError(
            f"per_level_indices length {len(per_level_indices)} != "
            f"num_levels {len(num_groups_per_level)}"
        )
    parts: list[bytes] = []
    for level_idx, indices in enumerate(per_level_indices):
        G = int(num_groups_per_level[level_idx])
        K = int(num_categories_per_level[level_idx])
        if K <= 0:
            raise ValueError(
                f"level {level_idx}: num_categories {K} must be positive"
            )
        if indices.ndim != 2:
            raise ValueError(
                f"level {level_idx}: indices must be 2-D (num_pairs, num_groups_l); "
                f"got shape {indices.shape}"
            )
        if indices.shape != (num_pairs, G):
            raise ValueError(
                f"level {level_idx}: indices shape {indices.shape} != "
                f"expected ({num_pairs}, {G})"
            )
        if indices.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"level {level_idx}: indices must be integer dtype; got {indices.dtype}"
            )
        if indices.min() < 0 or indices.max() >= K:
            raise ValueError(
                f"level {level_idx}: indices out of [0, {K}) range; "
                f"min={int(indices.min())}, max={int(indices.max())}"
            )
        if K <= 256:
            parts.append(indices.astype(np.uint8).tobytes(order="C"))
        elif K <= 65536:
            parts.append(indices.astype(np.uint16).tobytes(order="C"))
        else:
            raise ValueError(
                f"level {level_idx}: num_categories={K} exceeds u16 packing limit"
            )
    return b"".join(parts)


def _unpack_per_level_indices(
    raw: bytes,
    *,
    num_pairs: int,
    num_groups_per_level: tuple[int, ...],
    num_categories_per_level: tuple[int, ...],
) -> list[np.ndarray]:
    """Inverse of ``_pack_per_level_indices``."""
    L = len(num_groups_per_level)
    if len(num_categories_per_level) != L:
        raise ValueError(
            "num_groups_per_level and num_categories_per_level must have same length"
        )
    expected_bytes = 0
    for level_idx in range(L):
        G = int(num_groups_per_level[level_idx])
        K = int(num_categories_per_level[level_idx])
        bytes_per_idx = 1 if K <= 256 else 2
        expected_bytes += num_pairs * G * bytes_per_idx
    if len(raw) != expected_bytes:
        raise ValueError(
            f"indices_blob: got {len(raw)} bytes, expected {expected_bytes}"
        )
    out: list[np.ndarray] = []
    pos = 0
    for level_idx in range(L):
        G = int(num_groups_per_level[level_idx])
        K = int(num_categories_per_level[level_idx])
        if K <= 256:
            dtype = np.uint8
            bytes_per_idx = 1
        else:
            dtype = np.uint16
            bytes_per_idx = 2
        level_bytes = num_pairs * G * bytes_per_idx
        arr = (
            np.frombuffer(raw[pos : pos + level_bytes], dtype=dtype)
            .reshape(num_pairs, G)
            .copy()
        )
        out.append(arr.astype(np.int32, copy=False))
        pos += level_bytes
    return out


def pack_archive(
    decoder_state_dict: dict[str, Any],
    per_level_category_indices: list[np.ndarray],
    wavelet_coeffs_blob: bytes,
    wyner_ziv_top_blob: bytes,
    dreamer_state_dict: dict[str, Any],
    meta: dict[str, Any],
    *,
    num_levels: int,
    num_groups_per_level: tuple[int, ...],
    num_categories_per_level: tuple[int, ...],
    num_pairs: int,
    decoder_latent_dim: int,
    base_channels: int,
    wavelet_basis_id: int,
    schema_version: int = Z8HPC1_SCHEMA_VERSION,
) -> bytes:
    """Serialize Z8 substrate weights + per-level cat indices + wavelet +
    Wyner-Ziv + DreamerV3 state + meta to Z8HPC1 bytes."""
    if schema_version != Z8HPC1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if num_levels != _CANONICAL_NUM_LEVELS:
        raise ValueError(
            f"num_levels {num_levels} != canonical {_CANONICAL_NUM_LEVELS}; "
            f"schema_version bump required to change hierarchy depth"
        )
    if len(num_groups_per_level) != num_levels:
        raise ValueError(
            f"num_groups_per_level length {len(num_groups_per_level)} != "
            f"num_levels {num_levels}"
        )
    if len(num_categories_per_level) != num_levels:
        raise ValueError(
            f"num_categories_per_level length {len(num_categories_per_level)} != "
            f"num_levels {num_levels}"
        )
    for level_idx, g in enumerate(num_groups_per_level):
        if g <= 0 or g > 0xFF:
            raise ValueError(
                f"num_groups_per_level[{level_idx}]={g} out of u8 range [1, 255]"
            )
    for level_idx, k in enumerate(num_categories_per_level):
        if k <= 0 or k > 0xFFFF:
            raise ValueError(
                f"num_categories_per_level[{level_idx}]={k} out of u16 range [1, 65535]"
            )
    for name, value in [
        ("num_pairs", num_pairs),
        ("decoder_latent_dim", decoder_latent_dim),
        ("base_channels", base_channels),
    ]:
        if value <= 0 or value > 0xFFFF:
            raise ValueError(f"{name} {value} out of u16 range [1, 65535]")
    if wavelet_basis_id < 0 or wavelet_basis_id > 0xFF:
        raise ValueError(
            f"wavelet_basis_id {wavelet_basis_id} out of u8 range [0, 255]"
        )

    decoder_blob = _serialize_state_dict_fp16(decoder_state_dict)
    indices_blob = _pack_per_level_indices(
        per_level_category_indices,
        num_groups_per_level=num_groups_per_level,
        num_categories_per_level=num_categories_per_level,
        num_pairs=num_pairs,
    )
    wavelet_blob = bytes(wavelet_coeffs_blob)
    wz_blob = bytes(wyner_ziv_top_blob)
    dreamer_state_blob = _serialize_state_dict_fp16(dreamer_state_dict)

    meta_bytes = json.dumps(
        meta, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    # Pack groups + categories into fixed-width fields
    g_bytes = bytes(int(g) for g in num_groups_per_level)
    k_bytes_list = []
    for k in num_categories_per_level:
        k_bytes_list.append(struct.pack("<H", int(k)))
    # k_per_level packed inline below via the 3 H slots

    header = struct.pack(
        Z8HPC1_HEADER_FMT,
        Z8HPC1_MAGIC,                       # 8s
        schema_version,                     # B
        num_levels,                         # B
        g_bytes,                            # 3s (3 bytes for 3 levels)
        int(num_categories_per_level[0]),   # H
        int(num_categories_per_level[1]),   # H
        int(num_categories_per_level[2]),   # H
        num_pairs,                          # H
        decoder_latent_dim,                 # H
        base_channels,                      # H
        wavelet_basis_id,                   # B
        len(decoder_blob),                  # I
        len(indices_blob),                  # I
        len(wavelet_blob),                  # I
        len(wz_blob),                       # I
        len(dreamer_state_blob),            # I
        len(meta_bytes),                    # I
        b"\x00" * 12,                       # 12s reserved
    )
    return (
        header
        + decoder_blob
        + indices_blob
        + wavelet_blob
        + wz_blob
        + dreamer_state_blob
        + meta_bytes
    )


def parse_z8hpc1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for Z8HPC1 grammar.

    Canonical section-offset parser. The returned mapping is the data contract
    consumed by ``tac.analysis.scorer_conditional_mdl`` (Tier A density
    estimation) + ``tools.mdl_scorer_conditional_ablation`` + per-section
    byte-mutation testing per Catalog #139 + #272 + #105.

    Returned sections:
    - ``z8hpc1_header`` — 62-byte header (control_or_metadata; fixed layout)
    - ``decoder_blob`` — brotli q=9 compressed multi-level decoder + cat_proj weights
      (decoder_weight_stream; inflate-time consumer)
    - ``indices_blob`` — packed per-pair per-level categorical indices
      (DISTINGUISHING FEATURE #1 — DreamerV3 RSSM per-level)
    - ``wavelet_blob`` — per-level Mallat detail-band coeffs
      (DISTINGUISHING FEATURE #2 — Mallat wavelet codec)
    - ``wyner_ziv_blob`` — Wyner-Ziv top-level coded
      (DISTINGUISHING FEATURE #3 — Wyner-Ziv side-info coder)
    - ``dreamer_state_blob`` — DreamerV3 GRU state init
      (DISTINGUISHING FEATURE #4 — DreamerV3 latent dynamics)
    - ``meta_blob`` — sorted-keys JSON sidecar (control_or_metadata)
    """
    if len(archive_bytes) < Z8HPC1_HEADER_SIZE:
        raise ValueError(
            f"Z8HPC1 archive too short: {len(archive_bytes)} < {Z8HPC1_HEADER_SIZE}"
        )
    header_fields = struct.unpack(
        Z8HPC1_HEADER_FMT, archive_bytes[:Z8HPC1_HEADER_SIZE]
    )
    (
        magic,
        version,
        num_levels,
        g_bytes,
        k0, k1, k2,
        num_pairs,
        decoder_latent_dim,
        base_channels,
        wavelet_basis_id,
        decoder_blob_len,
        indices_blob_len,
        wavelet_blob_len,
        wz_blob_len,
        dreamer_state_blob_len,
        meta_blob_len,
        _reserved,
    ) = header_fields
    if magic != Z8HPC1_MAGIC:
        raise ValueError(
            f"Z8HPC1 magic mismatch: got {magic!r}, expected {Z8HPC1_MAGIC!r}"
        )
    if version != Z8HPC1_SCHEMA_VERSION:
        raise ValueError(
            f"Z8HPC1 schema_version {version} != canonical {Z8HPC1_SCHEMA_VERSION}"
        )
    if num_levels != _CANONICAL_NUM_LEVELS:
        raise ValueError(
            f"Z8HPC1 num_levels {num_levels} != canonical {_CANONICAL_NUM_LEVELS}"
        )

    pos = Z8HPC1_HEADER_SIZE
    decoder_start = pos
    pos += decoder_blob_len
    indices_start = pos
    pos += indices_blob_len
    wavelet_start = pos
    pos += wavelet_blob_len
    wz_start = pos
    pos += wz_blob_len
    dreamer_state_start = pos
    pos += dreamer_state_blob_len
    meta_start = pos
    pos += meta_blob_len

    if pos != len(archive_bytes):
        raise ValueError(
            f"Z8HPC1 declared end {pos} != archive bytes len {len(archive_bytes)}"
        )

    return {
        "z8hpc1_header": (0, Z8HPC1_HEADER_SIZE),
        "decoder_blob": (decoder_start, decoder_blob_len),
        "indices_blob": (indices_start, indices_blob_len),
        "wavelet_blob": (wavelet_start, wavelet_blob_len),
        "wyner_ziv_blob": (wz_start, wz_blob_len),
        "dreamer_state_blob": (dreamer_state_start, dreamer_state_blob_len),
        "meta_blob": (meta_start, meta_blob_len),
    }


def parse_archive(archive_bytes: bytes) -> Z8HierarchicalArchive:
    """Parse Z8HPC1 archive bytes into typed dataclass."""
    sections = parse_z8hpc1_archive_bytes(archive_bytes)

    header_fields = struct.unpack(
        Z8HPC1_HEADER_FMT, archive_bytes[:Z8HPC1_HEADER_SIZE]
    )
    (
        _magic,
        version,
        num_levels,
        g_bytes,
        k0, k1, k2,
        num_pairs,
        decoder_latent_dim,
        base_channels,
        wavelet_basis_id,
        _decoder_len,
        _indices_len,
        _wavelet_len,
        _wz_len,
        _dreamer_state_len,
        _meta_len,
        _reserved,
    ) = header_fields

    num_groups_per_level = tuple(int(b) for b in g_bytes)
    num_categories_per_level = (int(k0), int(k1), int(k2))

    dec_start, dec_len = sections["decoder_blob"]
    decoder_state_dict = _deserialize_state_dict_fp16(
        archive_bytes[dec_start : dec_start + dec_len]
    )

    idx_start, idx_len = sections["indices_blob"]
    per_level_category_indices = _unpack_per_level_indices(
        archive_bytes[idx_start : idx_start + idx_len],
        num_pairs=int(num_pairs),
        num_groups_per_level=num_groups_per_level,
        num_categories_per_level=num_categories_per_level,
    )

    wav_start, wav_len = sections["wavelet_blob"]
    wavelet_coeffs_blob = bytes(archive_bytes[wav_start : wav_start + wav_len])

    wz_start, wz_len = sections["wyner_ziv_blob"]
    wyner_ziv_top_blob = bytes(archive_bytes[wz_start : wz_start + wz_len])

    ds_start, ds_len = sections["dreamer_state_blob"]
    dreamer_state_dict = _deserialize_state_dict_fp16(
        archive_bytes[ds_start : ds_start + ds_len]
    )

    meta_start, meta_len = sections["meta_blob"]
    meta_raw = bytes(archive_bytes[meta_start : meta_start + meta_len])
    meta = json.loads(meta_raw.decode("utf-8")) if meta_len else {}

    return Z8HierarchicalArchive(
        decoder_state_dict=decoder_state_dict,
        per_level_category_indices=per_level_category_indices,
        wavelet_coeffs_blob=wavelet_coeffs_blob,
        wyner_ziv_top_blob=wyner_ziv_top_blob,
        dreamer_state_blob=dreamer_state_dict,
        meta=meta,
        schema_version=int(version),
        num_levels=int(num_levels),
        num_groups_per_level=num_groups_per_level,
        num_categories_per_level=num_categories_per_level,
        num_pairs=int(num_pairs),
        decoder_latent_dim=int(decoder_latent_dim),
        base_channels=int(base_channels),
        wavelet_basis_id=int(wavelet_basis_id),
    )


__all__ = [
    "Z8HPC1_HEADER_FMT",
    "Z8HPC1_HEADER_SIZE",
    "Z8HPC1_MAGIC",
    "Z8HPC1_SCHEMA_VERSION",
    "Z8HierarchicalArchive",
    "pack_archive",
    "parse_archive",
    "parse_z8hpc1_archive_bytes",
]
