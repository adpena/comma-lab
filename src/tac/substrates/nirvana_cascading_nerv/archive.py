# SPDX-License-Identifier: MIT
"""nirvana_cascading_nerv.archive — NIRVANA1 monolithic single-file ``0.bin`` grammar.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic
0.bin) + L4 (≤200 LOC inflate substrate-engineering waiver) + L8
(deterministic).

L0 SCAFFOLD scope: the grammar defined here is the canonical contract;
the PyTorch port for inflate-time consumption is in ``inflate.py``. The
archive grammar is byte-deterministic from MLX-trained weights via the
canonical write/parse round-trip implemented below.

Grammar:

::

    MAGIC(5)             b"NIR1\\x00"
    VERSION(1)           u8       schema version (currently 1)
    NUM_LEVELS(1)        u8       cfg.num_levels (e.g. 4)
    PER_PAIR_LATENT(1)   u8       cfg.per_pair_latent_dim (e.g. 16)
    BASE_H(2)            u16      cfg.base_h (e.g. 48)
    BASE_W(2)            u16      cfg.base_w (e.g. 64)
    DECODER_BLOB_LEN(4)  u32      brotli(q=9) of per-level decoder state_dict
    RESIDUAL_BLOB_LEN(4) u32      brotli(q=9) of per-level int8 residuals
    LATENTS_BLOB_LEN(4)  u32      brotli(q=9) of per-pair int16 latents
    META_BLOB_LEN(4)     u32      sorted-keys JSON utf-8 bytes
    DECODER_BLOB         ...      brotli of state_dict bytes (fp16)
    RESIDUAL_BLOB        ...      brotli of length-prefixed per-level residuals
    LATENTS_BLOB         ...      brotli of int16 latents bytes
    META_BLOB            ...      sorted-keys JSON utf-8

The per-level decoder state_dict + per-level int8 residuals + per-pair
int16 latents are ALL inflate-time consumers. The hierarchical residual
cascade reconstructs final RGB per ``inflate.py::inflate_one_video``.

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load at inflate
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli quality;
  fp16 state_dict cast on CPU; int8/int16 raw bytes for residuals/latents)

Per Catalog #139 + #105 + #272 distinguishing-feature contract:
- decoder_blob bytes ARE frame-affecting at inflate (per-level decoder weights consumed)
- residual_blob bytes ARE frame-affecting at inflate (per-level int8 residuals added
  to upsampled coarse estimate; THIS IS THE DISTINGUISHING FEATURE vs single-decoder
  NeRV-family substrates)
- latents_blob bytes ARE frame-affecting at inflate (per-pair latent z consumed)
- header + meta bytes are parse/config gates (control_or_metadata role)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np


NIRVANA1_MAGIC: bytes = b"NIR1\x00"
"""NIRVANA cascading NeRV variant 1 archive magic."""

NIRVANA1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout per docstring:
# MAGIC(5) + VERSION(1) + NUM_LEVELS(1) + PER_PAIR_LATENT(1) + BASE_H(2) +
# BASE_W(2) + DECODER_LEN(4) + RESIDUAL_LEN(4) + LATENTS_LEN(4) + META_LEN(4)
NIRVANA1_HEADER_FMT: str = "<5sBBBHHIIII"
NIRVANA1_HEADER_SIZE: int = struct.calcsize(NIRVANA1_HEADER_FMT)
assert NIRVANA1_HEADER_SIZE == 28, (
    f"NIRVANA1 header size invariant: expected 28, got {NIRVANA1_HEADER_SIZE}"
)

# Deterministic brotli quality (matches C6 IBPS / sane_hnerv / WZF01 /
# boost_nerv_pr110_residual / dreamer_v3_rssm canonical sister pattern).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class NirvanaCascadingNervArchive:
    """Parsed NIRVANA1 archive — the inflate-time data contract."""

    decoder_state_dict: dict[str, np.ndarray]
    """Per-level decoder + cat_to_continuous state_dict (inflate-time consumer)."""

    per_level_residuals: list[np.ndarray]
    """List of (H_i, W_i, 3) int8 residual arrays; ascending level order."""

    per_pair_latents: np.ndarray
    """(num_pairs, per_pair_latent_dim) int16 latent z array."""

    meta: dict[str, Any]
    """Sidecar JSON meta: num_pairs, gumbel_temperature, residual_scale, ..."""

    schema_version: int
    num_levels: int
    per_pair_latent_dim: int
    base_h: int
    base_w: int


def _serialize_state_dict_fp16(sd: dict[str, Any]) -> bytes:
    """Serialize a state_dict deterministically as length-prefixed records.

    Mirrors the canonical sister substrate dreamer_v3_rssm pattern.

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
        # Normalize to numpy float32 first (MLX arrays, torch tensors, numpy)
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
    """Inverse of ``_serialize_state_dict_fp16``."""
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


def _pack_per_level_residuals(residuals: list[np.ndarray]) -> bytes:
    """Pack per-level int8 residuals with per-level u32 length prefix.

    Format per level (in ascending order):
        u32 len (bytes of int8 residual data)
        int8 residual data (H * W * 3 bytes)

    Returns brotli-compressed concatenation.
    """
    parts: list[bytes] = []
    for residual in residuals:
        if residual.dtype != np.int8:
            raise ValueError(
                f"residual must be int8; got {residual.dtype}"
            )
        if residual.ndim != 3 or residual.shape[-1] != 3:
            raise ValueError(
                f"residual must be (H, W, 3); got shape {residual.shape}"
            )
        data = residual.tobytes(order="C")
        parts.append(struct.pack("<I", len(data)))
        parts.append(data)
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _unpack_per_level_residuals(
    blob: bytes,
    *,
    per_level_shapes: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Inverse of ``_pack_per_level_residuals``.

    Args:
        blob: brotli-compressed concatenation
        per_level_shapes: ordered list of (H, W) per level
    """
    raw = brotli.decompress(blob)
    residuals: list[np.ndarray] = []
    pos = 0
    for (h, w) in per_level_shapes:
        if pos + 4 > len(raw):
            raise ValueError("residual blob truncated reading level length")
        (level_len,) = struct.unpack("<I", raw[pos : pos + 4])
        pos += 4
        expected_bytes = h * w * 3
        if level_len != expected_bytes:
            raise ValueError(
                f"residual level length mismatch: got {level_len}, "
                f"expected {expected_bytes} for shape ({h}, {w})"
            )
        if pos + level_len > len(raw):
            raise ValueError("residual blob truncated reading level data")
        arr = np.frombuffer(
            raw[pos : pos + level_len], dtype=np.int8
        ).reshape(h, w, 3).copy()
        residuals.append(arr)
        pos += level_len
    if pos != len(raw):
        raise ValueError(
            f"residual blob has trailing bytes (pos={pos} len={len(raw)})"
        )
    return residuals


def _pack_latents(latents: np.ndarray) -> bytes:
    """Pack (num_pairs, latent_dim) int16 latents into brotli-compressed bytes."""
    if latents.dtype != np.int16:
        raise ValueError(f"latents must be int16; got {latents.dtype}")
    if latents.ndim != 2:
        raise ValueError(
            f"latents must be (num_pairs, latent_dim); got shape {latents.shape}"
        )
    return bytes(brotli.compress(latents.tobytes(order="C"), quality=_BROTLI_QUALITY))


def _unpack_latents(
    blob: bytes, *, num_pairs: int, latent_dim: int
) -> np.ndarray:
    """Inverse of ``_pack_latents``."""
    raw = brotli.decompress(blob)
    expected_bytes = num_pairs * latent_dim * 2  # int16
    if len(raw) != expected_bytes:
        raise ValueError(
            f"latents blob: got {len(raw)} bytes, expected {expected_bytes} "
            f"(num_pairs={num_pairs}, latent_dim={latent_dim})"
        )
    return np.frombuffer(raw, dtype=np.int16).reshape(num_pairs, latent_dim).copy()


def pack_archive(
    decoder_state_dict: dict[str, Any],
    per_level_residuals: list[np.ndarray],
    per_pair_latents: np.ndarray,
    meta: dict[str, Any],
    *,
    num_levels: int,
    per_pair_latent_dim: int,
    base_h: int,
    base_w: int,
    schema_version: int = NIRVANA1_SCHEMA_VERSION,
) -> bytes:
    """Serialize substrate weights + per-level residuals + per-pair latents + meta to NIRVANA1 bytes."""
    if schema_version != NIRVANA1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if len(per_level_residuals) != num_levels:
        raise ValueError(
            f"expected {num_levels} per-level residuals; got {len(per_level_residuals)}"
        )
    for name, value, max_val in [
        ("num_levels", num_levels, 0xFF),
        ("per_pair_latent_dim", per_pair_latent_dim, 0xFF),
        ("base_h", base_h, 0xFFFF),
        ("base_w", base_w, 0xFFFF),
    ]:
        if value <= 0 or value > max_val:
            raise ValueError(f"{name} {value} out of range [1, {max_val}]")

    decoder_blob = _serialize_state_dict_fp16(decoder_state_dict)
    residual_blob = _pack_per_level_residuals(per_level_residuals)
    latents_blob = _pack_latents(per_pair_latents)
    meta_bytes = json.dumps(
        meta, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        NIRVANA1_HEADER_FMT,
        NIRVANA1_MAGIC,
        schema_version,
        num_levels,
        per_pair_latent_dim,
        base_h,
        base_w,
        len(decoder_blob),
        len(residual_blob),
        len(latents_blob),
        len(meta_bytes),
    )
    return header + decoder_blob + residual_blob + latents_blob + meta_bytes


def parse_archive(archive_bytes: bytes) -> NirvanaCascadingNervArchive:
    """Parse NIRVANA1 archive bytes into the inflate-time data contract."""
    if len(archive_bytes) < NIRVANA1_HEADER_SIZE:
        raise ValueError(
            f"archive too short: {len(archive_bytes)} < {NIRVANA1_HEADER_SIZE} header bytes"
        )
    header = struct.unpack(
        NIRVANA1_HEADER_FMT, archive_bytes[:NIRVANA1_HEADER_SIZE]
    )
    (
        magic,
        version,
        num_levels,
        per_pair_latent_dim,
        base_h,
        base_w,
        decoder_len,
        residual_len,
        latents_len,
        meta_len,
    ) = header

    if magic != NIRVANA1_MAGIC:
        raise ValueError(f"unexpected magic {magic!r}; expected {NIRVANA1_MAGIC!r}")
    if version != NIRVANA1_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {version}; expected {NIRVANA1_SCHEMA_VERSION}"
        )

    total_expected = (
        NIRVANA1_HEADER_SIZE + decoder_len + residual_len + latents_len + meta_len
    )
    if len(archive_bytes) != total_expected:
        raise ValueError(
            f"archive size mismatch: got {len(archive_bytes)}, expected {total_expected}"
        )

    pos = NIRVANA1_HEADER_SIZE
    decoder_blob = archive_bytes[pos : pos + decoder_len]
    pos += decoder_len
    residual_blob = archive_bytes[pos : pos + residual_len]
    pos += residual_len
    latents_blob = archive_bytes[pos : pos + latents_len]
    pos += latents_len
    meta_bytes = archive_bytes[pos : pos + meta_len]

    decoder_sd = _deserialize_state_dict_fp16(decoder_blob)

    # Compute per-level shapes from base_h, base_w, num_levels
    per_level_shapes = [
        (base_h * (2 ** i), base_w * (2 ** i)) for i in range(num_levels)
    ]
    residuals = _unpack_per_level_residuals(
        residual_blob, per_level_shapes=per_level_shapes
    )

    meta = json.loads(meta_bytes.decode("utf-8"))
    num_pairs = int(meta.get("num_pairs", 600))
    latents = _unpack_latents(
        latents_blob, num_pairs=num_pairs, latent_dim=per_pair_latent_dim
    )

    return NirvanaCascadingNervArchive(
        decoder_state_dict=decoder_sd,
        per_level_residuals=residuals,
        per_pair_latents=latents,
        meta=meta,
        schema_version=version,
        num_levels=num_levels,
        per_pair_latent_dim=per_pair_latent_dim,
        base_h=base_h,
        base_w=base_w,
    )


__all__ = [
    "NIRVANA1_HEADER_FMT",
    "NIRVANA1_HEADER_SIZE",
    "NIRVANA1_MAGIC",
    "NIRVANA1_SCHEMA_VERSION",
    "NirvanaCascadingNervArchive",
    "pack_archive",
    "parse_archive",
]
