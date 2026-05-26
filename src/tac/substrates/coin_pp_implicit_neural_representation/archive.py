# SPDX-License-Identifier: MIT
"""coin_pp_implicit_neural_representation.archive — COINPP1 monolithic single-file ``0.bin`` grammar.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic
0.bin) + L4 (≤200 LOC inflate substrate-engineering waiver) + L8
(deterministic).

L0 SCAFFOLD scope: the grammar defined here is the canonical contract;
the PyTorch port for inflate-time consumption is in ``inflate.py``. The
archive grammar is byte-deterministic from MLX-trained weights via the
canonical write/parse round-trip implemented below.

Grammar:

::

    MAGIC(5)             b"CPP1\\x00"
    VERSION(1)           u8       schema version (currently 1)
    MOD_DIM(1)           u8       cfg.mod_dim (e.g. 64)
    POS_DIM(1)           u8       cfg.pos_dim (e.g. 32)
    HIDDEN_DIM(2)        u16      cfg.hidden_dim (e.g. 64)
    NUM_HIDDEN_LAYERS(1) u8       cfg.num_hidden_layers (e.g. 3)
    NUM_PAIRS(2)         u16      cfg.num_pairs (e.g. 600)
    EVAL_H(2)            u16      cfg.eval_h (must be 384)
    EVAL_W(2)            u16      cfg.eval_w (must be 512)
    BASE_BLOB_LEN(4)     u32      brotli(q=9) of base coord-MLP state_dict
    MOD_BLOB_LEN(4)      u32      brotli(q=9) of per-pair int8 modulations
    META_BLOB_LEN(4)     u32      sorted-keys JSON utf-8 bytes
    RESERVED(3)          u8 x 3   reserved for future use (zero-filled)
    BASE_BLOB            ...      brotli of state_dict bytes (fp16)
    MOD_BLOB             ...      brotli of int8 modulation bytes
    META_BLOB            ...      sorted-keys JSON utf-8

Total header = 32 bytes.

Per Catalog #139 + #105 + #272 distinguishing-feature contract:
- base_blob bytes ARE frame-affecting at inflate (shared coord-MLP weights
  consumed across ALL pairs)
- modulation_blob bytes ARE frame-affecting at inflate (per-pair modulation
  vectors consumed; THIS IS THE DISTINGUISHING FEATURE vs per-pair-latent
  NeRV-family substrates — modulation has DIFFERENT semantic role)
- header + meta bytes are parse/config gates (control_or_metadata role)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np


COINPP1_MAGIC: bytes = b"CPP1\x00"
"""COIN++ implicit neural representation variant 1 archive magic."""

COINPP1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout per docstring:
# MAGIC(5) + VERSION(1) + MOD_DIM(1) + POS_DIM(1) + HIDDEN_DIM(2) +
# NUM_HIDDEN_LAYERS(1) + NUM_PAIRS(2) + EVAL_H(2) + EVAL_W(2) +
# BASE_LEN(4) + MOD_LEN(4) + META_LEN(4) + reserved(3)
COINPP1_HEADER_FMT: str = "<5sBBBHBHHHIII3s"
COINPP1_HEADER_SIZE: int = struct.calcsize(COINPP1_HEADER_FMT)
assert COINPP1_HEADER_SIZE == 32, (
    f"COINPP1 header size invariant: expected 32, got {COINPP1_HEADER_SIZE}"
)

# Deterministic brotli quality (matches C6 IBPS / sane_hnerv / NIRVANA /
# boost_nerv_pr110_residual / dreamer_v3_rssm canonical sister pattern).
_BROTLI_QUALITY: int = 9
_RESERVED_BYTES: bytes = b"\x00\x00\x00"


@dataclass(frozen=True)
class CoinPPArchive:
    """Parsed COINPP1 archive — the inflate-time data contract."""

    base_state_dict: dict[str, np.ndarray]
    """Shared coord-MLP base state_dict (inflate-time consumer)."""

    per_pair_modulations: np.ndarray
    """(num_pairs, mod_dim) int8 per-pair modulation array."""

    meta: dict[str, Any]
    """Sidecar JSON meta: num_pairs, modulation_scale, eval_hw, ..."""

    schema_version: int
    mod_dim: int
    pos_dim: int
    hidden_dim: int
    num_hidden_layers: int
    num_pairs: int
    eval_h: int
    eval_w: int


def _serialize_state_dict_fp16(sd: dict[str, Any]) -> bytes:
    """Serialize a state_dict deterministically as length-prefixed records.

    Mirrors the canonical sister substrate dreamer_v3_rssm / NIRVANA pattern.

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


def _pack_modulations(modulations: np.ndarray) -> bytes:
    """Pack (num_pairs, mod_dim) int8 modulations into brotli-compressed bytes."""
    if modulations.dtype != np.int8:
        raise ValueError(f"modulations must be int8; got {modulations.dtype}")
    if modulations.ndim != 2:
        raise ValueError(
            f"modulations must be (num_pairs, mod_dim); got shape {modulations.shape}"
        )
    return bytes(brotli.compress(modulations.tobytes(order="C"), quality=_BROTLI_QUALITY))


def _unpack_modulations(
    blob: bytes, *, num_pairs: int, mod_dim: int
) -> np.ndarray:
    """Inverse of ``_pack_modulations``."""
    raw = brotli.decompress(blob)
    expected_bytes = num_pairs * mod_dim  # int8 = 1 byte each
    if len(raw) != expected_bytes:
        raise ValueError(
            f"modulation blob: got {len(raw)} bytes, expected {expected_bytes} "
            f"(num_pairs={num_pairs}, mod_dim={mod_dim})"
        )
    return np.frombuffer(raw, dtype=np.int8).reshape(num_pairs, mod_dim).copy()


def pack_archive(
    base_state_dict: dict[str, Any],
    per_pair_modulations: np.ndarray,
    meta: dict[str, Any],
    *,
    mod_dim: int,
    pos_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
    schema_version: int = COINPP1_SCHEMA_VERSION,
) -> bytes:
    """Serialize base coord-MLP + per-pair modulations + meta to COINPP1 bytes."""
    if schema_version != COINPP1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if per_pair_modulations.shape != (num_pairs, mod_dim):
        raise ValueError(
            f"modulations shape {per_pair_modulations.shape} != ({num_pairs}, {mod_dim})"
        )
    for name, value, max_val in [
        ("mod_dim", mod_dim, 0xFF),
        ("pos_dim", pos_dim, 0xFF),
        ("hidden_dim", hidden_dim, 0xFFFF),
        ("num_hidden_layers", num_hidden_layers, 0xFF),
        ("num_pairs", num_pairs, 0xFFFF),
        ("eval_h", eval_h, 0xFFFF),
        ("eval_w", eval_w, 0xFFFF),
    ]:
        if value <= 0 or value > max_val:
            raise ValueError(f"{name} {value} out of range [1, {max_val}]")

    base_blob = _serialize_state_dict_fp16(base_state_dict)
    mod_blob = _pack_modulations(per_pair_modulations)
    meta_bytes = json.dumps(
        meta, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        COINPP1_HEADER_FMT,
        COINPP1_MAGIC,
        schema_version,
        mod_dim,
        pos_dim,
        hidden_dim,
        num_hidden_layers,
        num_pairs,
        eval_h,
        eval_w,
        len(base_blob),
        len(mod_blob),
        len(meta_bytes),
        _RESERVED_BYTES,
    )
    return header + base_blob + mod_blob + meta_bytes


def parse_archive(archive_bytes: bytes) -> CoinPPArchive:
    """Parse COINPP1 archive bytes into the inflate-time data contract."""
    if len(archive_bytes) < COINPP1_HEADER_SIZE:
        raise ValueError(
            f"archive too short: {len(archive_bytes)} < {COINPP1_HEADER_SIZE} header bytes"
        )
    header = struct.unpack(
        COINPP1_HEADER_FMT, archive_bytes[:COINPP1_HEADER_SIZE]
    )
    (
        magic,
        version,
        mod_dim,
        pos_dim,
        hidden_dim,
        num_hidden_layers,
        num_pairs,
        eval_h,
        eval_w,
        base_len,
        mod_len,
        meta_len,
        _reserved,
    ) = header

    if magic != COINPP1_MAGIC:
        raise ValueError(f"unexpected magic {magic!r}; expected {COINPP1_MAGIC!r}")
    if version != COINPP1_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {version}; expected {COINPP1_SCHEMA_VERSION}"
        )

    total_expected = (
        COINPP1_HEADER_SIZE + base_len + mod_len + meta_len
    )
    if len(archive_bytes) != total_expected:
        raise ValueError(
            f"archive size mismatch: got {len(archive_bytes)}, expected {total_expected}"
        )

    pos = COINPP1_HEADER_SIZE
    base_blob = archive_bytes[pos : pos + base_len]
    pos += base_len
    mod_blob = archive_bytes[pos : pos + mod_len]
    pos += mod_len
    meta_bytes = archive_bytes[pos : pos + meta_len]

    base_sd = _deserialize_state_dict_fp16(base_blob)
    modulations = _unpack_modulations(
        mod_blob, num_pairs=num_pairs, mod_dim=mod_dim
    )
    meta = json.loads(meta_bytes.decode("utf-8"))

    return CoinPPArchive(
        base_state_dict=base_sd,
        per_pair_modulations=modulations,
        meta=meta,
        schema_version=version,
        mod_dim=mod_dim,
        pos_dim=pos_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_pairs=num_pairs,
        eval_h=eval_h,
        eval_w=eval_w,
    )


__all__ = [
    "COINPP1_HEADER_FMT",
    "COINPP1_HEADER_SIZE",
    "COINPP1_MAGIC",
    "COINPP1_SCHEMA_VERSION",
    "CoinPPArchive",
    "pack_archive",
    "parse_archive",
]
