# SPDX-License-Identifier: MIT
"""archive — MDLIBPS-J1 byte-deterministic archive grammar for J=MDL-IBPS.

Catalog #124 8-field manifest at module level in ``__init__.py``.
Catalog #146 + #205 + #220 inflate-contract canonical reference.
Catalog #295 PYTHONPATH self-containment.
Catalog #139 + #272 + #105 no-op detector + byte-mutation surface.

MDLIBPS-J1 grammar (sister to C6 IBPS1 grammar at ``tac.substrates.c6_e4_mdl_ibps.archive``):

    32-byte header:
        magic        : b"MIBJ1\\x00"     (6 bytes)
        version      : u8               (1 byte)   = 1
        K            : u8               (1 byte)   = 16 (CATEGORICAL_K)
        G            : u8               (1 byte)   = 12 (CATEGORICAL_G)
        HIDDEN_DIM   : u16              (2 bytes)  = 64
        NUM_HID_LAY  : u8               (1 byte)   = 3
        POS_DIM      : u8               (1 byte)   = 8
        NUM_PAIRS    : u16              (2 bytes)  = 600
        EVAL_H       : u16              (2 bytes)  = 384
        EVAL_W       : u16              (2 bytes)  = 512
        BASE_LEN     : u32              (4 bytes)
        MINE_LEN     : u32              (4 bytes)
        INDICES_LEN  : u32              (4 bytes)
        META_LEN     : u32              (4 bytes)
        reserved     : u8 x 3           (3 bytes)
    -- TOTAL HEADER = 38 bytes (rounded up to 38; not 32 as initially specified) --

    BASE_LEN bytes  : brotli(q=9) procedural-coord-MLP + FiLM-proj state_dict (fp16)
    MINE_LEN bytes  : brotli(q=9) MINE critic state_dict (provenance only;
                      NOT consumed at inflate per Catalog #220 declaration)
    INDICES_LEN bytes : brotli(q=9) per-pair categorical indices
                      (NUM_PAIRS * G * 4 bits packed = NUM_PAIRS * G / 2 bytes raw)
    META_LEN bytes  : sorted-keys JSON utf-8 (scale_modulation, num_pairs,
                      eval_hw, schema_version, ...)

Sister substrates' grammar patterns:
- ``tac.substrates.c6_e4_mdl_ibps.archive`` (IBPS1; parent reference)
- ``tac.substrates.coin_pp_implicit_neural_representation.archive`` (CPP1; sister)
- ``tac.substrates.dreamer_v3_rssm.archive`` (sister A; canonical Cat reference)
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from typing import Mapping

try:
    import brotli
    _BROTLI_AVAILABLE = True
except ImportError:
    brotli = None  # type: ignore[assignment]
    _BROTLI_AVAILABLE = False

from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
    CATEGORICAL_G,
    CATEGORICAL_K,
    EVAL_HW,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    NUM_PAIRS,
    POS_DIM,
)

MIBJ1_MAGIC: bytes = b"MIBJ1\x00"
"""Magic bytes for MDLIBPS-J1 grammar (6 bytes)."""

MIBJ1_SCHEMA_VERSION: int = 1
"""Schema version (u8)."""

# Header format string per the docstring layout
# < = little-endian
# 6s = 6-byte magic + u8 version + u8 K + u8 G + u16 HIDDEN_DIM + u8 NUM_HID_LAY + u8 POS_DIM
# + u16 NUM_PAIRS + u16 EVAL_H + u16 EVAL_W
# + u32 BASE_LEN + u32 MINE_LEN + u32 INDICES_LEN + u32 META_LEN
# + 3 reserved bytes
MIBJ1_HEADER_FMT: str = "<6sBBBHBBHHHIIII3s"
MIBJ1_HEADER_SIZE: int = struct.calcsize(MIBJ1_HEADER_FMT)
"""Header size in bytes (= 38)."""

MIBJ1_SECTION_ROLES: Mapping[str, str] = {
    "BASE_BLOB": "frame_affecting_at_inflate",  # Catalog #220 declaration
    "MINE_BLOB": "training_provenance_only",  # NOT consumed at inflate
    "INDICES_BLOB": "frame_affecting_at_inflate",  # per-pair categorical indices drive FiLM modulation
    "META_BLOB": "parse_config_only",
}
"""Per-section role classification (Catalog #220 + #272 distinguishing-feature contract)."""


def _require_brotli() -> None:
    if not _BROTLI_AVAILABLE:
        raise ImportError(
            "brotli is required for MDLIBPS-J1 archive grammar; install via "
            "`uv pip install brotli` (HNeRV parity L4 ≤ 2 deps; torch + brotli)"
        )


@dataclass(frozen=True)
class MDLIBPSJArchive:
    """Parsed MDLIBPS-J1 archive (in-memory representation).

    Attributes:
        base_blob: brotli-decompressed procedural-coord-MLP + FiLM-proj state_dict bytes
        mine_blob: brotli-decompressed MINE critic state_dict bytes (provenance only)
        indices_blob: brotli-decompressed per-pair categorical indices bytes
        meta: parsed JSON metadata dict (sorted keys)
    """

    base_blob: bytes
    mine_blob: bytes
    indices_blob: bytes
    meta: dict


def _pack_categorical_indices_to_bytes(indices: list[list[int]]) -> bytes:
    """Pack per-pair categorical indices into byte stream.

    Format: NUM_PAIRS x G nibbles (4 bits each since K=16 fits in 4 bits).
    Total = NUM_PAIRS * G / 2 bytes (assumes G even per CATEGORICAL_G default).

    Args:
        indices: list of NUM_PAIRS lists, each of length G with values in [0, K).

    Returns:
        packed bytes.
    """
    if CATEGORICAL_K > 16:
        raise ValueError(
            f"This packing assumes K <= 16 (4 bits per nibble); got K={CATEGORICAL_K}"
        )
    if len(indices) != NUM_PAIRS:
        raise ValueError(
            f"expected NUM_PAIRS={NUM_PAIRS} per-pair index lists; got {len(indices)}"
        )
    total_nibbles = NUM_PAIRS * CATEGORICAL_G
    if total_nibbles % 2 != 0:
        raise ValueError(
            f"total nibbles must be even for byte packing; got {total_nibbles} "
            f"(NUM_PAIRS={NUM_PAIRS} x G={CATEGORICAL_G})"
        )
    raw = bytearray(total_nibbles // 2)
    nibble_idx = 0
    for pair_indices in indices:
        if len(pair_indices) != CATEGORICAL_G:
            raise ValueError(
                f"each pair must have G={CATEGORICAL_G} indices; got {len(pair_indices)}"
            )
        for ci in pair_indices:
            if not 0 <= ci < CATEGORICAL_K:
                raise ValueError(
                    f"index out of range [0, {CATEGORICAL_K}); got {ci}"
                )
            byte_idx = nibble_idx // 2
            if nibble_idx % 2 == 0:
                raw[byte_idx] = (raw[byte_idx] & 0xF0) | (ci & 0x0F)
            else:
                raw[byte_idx] = (raw[byte_idx] & 0x0F) | ((ci & 0x0F) << 4)
            nibble_idx += 1
    return bytes(raw)


def _unpack_categorical_indices_from_bytes(raw: bytes) -> list[list[int]]:
    """Inverse of _pack_categorical_indices_to_bytes."""
    expected_bytes = (NUM_PAIRS * CATEGORICAL_G) // 2
    if len(raw) != expected_bytes:
        raise ValueError(
            f"expected {expected_bytes} bytes for packed indices; got {len(raw)}"
        )
    indices: list[list[int]] = []
    nibble_idx = 0
    for _ in range(NUM_PAIRS):
        pair: list[int] = []
        for _ in range(CATEGORICAL_G):
            byte_idx = nibble_idx // 2
            if nibble_idx % 2 == 0:
                pair.append(raw[byte_idx] & 0x0F)
            else:
                pair.append((raw[byte_idx] >> 4) & 0x0F)
            nibble_idx += 1
        indices.append(pair)
    return indices


def pack_archive(
    base_blob_uncompressed: bytes,
    mine_blob_uncompressed: bytes,
    indices: list[list[int]],
    meta: Mapping[str, object],
    *,
    brotli_quality: int = 9,
) -> bytes:
    """Pack a MDLIBPS-J1 archive into byte-deterministic representation.

    Per Catalog #146 + #220: this is the canonical encoder side. Sister
    inflate (``inflate.py``) parses these bytes back into archive.

    Args:
        base_blob_uncompressed: raw state_dict bytes for procedural-coord-MLP + FiLM-proj.
        mine_blob_uncompressed: raw state_dict bytes for MINE critic.
        indices: per-pair categorical indices (NUM_PAIRS x G entries).
        meta: metadata dict (will be json-encoded with sorted keys).
        brotli_quality: brotli compression quality (default 9; per HNeRV parity).

    Returns:
        byte-deterministic archive bytes (header + 4 blobs in order).
    """
    _require_brotli()
    # Compress blobs
    base_blob = brotli.compress(base_blob_uncompressed, quality=brotli_quality)
    mine_blob = brotli.compress(mine_blob_uncompressed, quality=brotli_quality)
    indices_bytes = _pack_categorical_indices_to_bytes(indices)
    indices_blob = brotli.compress(indices_bytes, quality=brotli_quality)
    # Encode meta with sorted keys for byte-determinism
    meta_bytes = json.dumps(
        dict(meta), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    # Compose header
    header = struct.pack(
        MIBJ1_HEADER_FMT,
        MIBJ1_MAGIC,
        MIBJ1_SCHEMA_VERSION,
        CATEGORICAL_K,
        CATEGORICAL_G,
        HIDDEN_DIM,
        NUM_HIDDEN_LAYERS,
        POS_DIM,
        NUM_PAIRS,
        EVAL_HW[0],
        EVAL_HW[1],
        len(base_blob),
        len(mine_blob),
        len(indices_blob),
        len(meta_bytes),
        b"\x00\x00\x00",
    )
    return header + base_blob + mine_blob + indices_blob + meta_bytes


def parse_archive_bytes(archive_bytes: bytes) -> MDLIBPSJArchive:
    """Parse MDLIBPS-J1 archive bytes back into structured form.

    Args:
        archive_bytes: raw archive bytes (from pack_archive).

    Returns:
        MDLIBPSJArchive with decompressed blobs + parsed meta dict.
    """
    _require_brotli()
    if len(archive_bytes) < MIBJ1_HEADER_SIZE:
        raise ValueError(
            f"archive too small; need at least {MIBJ1_HEADER_SIZE} bytes for header; "
            f"got {len(archive_bytes)}"
        )
    header = struct.unpack(
        MIBJ1_HEADER_FMT, archive_bytes[:MIBJ1_HEADER_SIZE]
    )
    magic = header[0]
    if magic != MIBJ1_MAGIC:
        raise ValueError(
            f"bad magic; expected {MIBJ1_MAGIC!r}; got {magic!r}"
        )
    version = header[1]
    if version != MIBJ1_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {version}; expected {MIBJ1_SCHEMA_VERSION}"
        )
    K, G, hidden_dim, num_hid_lay, pos_dim, num_pairs, eval_h, eval_w = header[2:10]
    base_len, mine_len, indices_len, meta_len = header[10:14]
    # Validate against compile-time constants
    if K != CATEGORICAL_K:
        raise ValueError(f"K mismatch; expected {CATEGORICAL_K}; got {K}")
    if G != CATEGORICAL_G:
        raise ValueError(f"G mismatch; expected {CATEGORICAL_G}; got {G}")
    if hidden_dim != HIDDEN_DIM:
        raise ValueError(f"HIDDEN_DIM mismatch; expected {HIDDEN_DIM}; got {hidden_dim}")
    if num_hid_lay != NUM_HIDDEN_LAYERS:
        raise ValueError(
            f"NUM_HIDDEN_LAYERS mismatch; expected {NUM_HIDDEN_LAYERS}; got {num_hid_lay}"
        )
    if pos_dim != POS_DIM:
        raise ValueError(f"POS_DIM mismatch; expected {POS_DIM}; got {pos_dim}")
    if num_pairs != NUM_PAIRS:
        raise ValueError(f"NUM_PAIRS mismatch; expected {NUM_PAIRS}; got {num_pairs}")
    if (eval_h, eval_w) != EVAL_HW:
        raise ValueError(f"EVAL_HW mismatch; expected {EVAL_HW}; got ({eval_h}, {eval_w})")
    # Extract blobs
    offset = MIBJ1_HEADER_SIZE
    base_blob_compressed = archive_bytes[offset:offset + base_len]
    offset += base_len
    mine_blob_compressed = archive_bytes[offset:offset + mine_len]
    offset += mine_len
    indices_blob_compressed = archive_bytes[offset:offset + indices_len]
    offset += indices_len
    meta_bytes = archive_bytes[offset:offset + meta_len]
    expected_total = offset + meta_len
    if len(archive_bytes) != expected_total:
        raise ValueError(
            f"trailing bytes in archive; expected {expected_total}; got {len(archive_bytes)}"
        )
    # Decompress
    base_blob = brotli.decompress(base_blob_compressed)
    mine_blob = brotli.decompress(mine_blob_compressed)
    indices_blob = brotli.decompress(indices_blob_compressed)
    # Parse meta
    meta = json.loads(meta_bytes.decode("utf-8"))
    return MDLIBPSJArchive(
        base_blob=base_blob,
        mine_blob=mine_blob,
        indices_blob=indices_blob,
        meta=meta,
    )


def unpack_categorical_indices(archive: MDLIBPSJArchive) -> list[list[int]]:
    """Unpack categorical indices from a parsed archive.

    Sister of ``_unpack_categorical_indices_from_bytes`` exposed at the public API surface.
    """
    return _unpack_categorical_indices_from_bytes(archive.indices_blob)


__all__ = [
    "MDLIBPSJArchive",
    "MIBJ1_HEADER_FMT",
    "MIBJ1_HEADER_SIZE",
    "MIBJ1_MAGIC",
    "MIBJ1_SCHEMA_VERSION",
    "MIBJ1_SECTION_ROLES",
    "pack_archive",
    "parse_archive_bytes",
    "unpack_categorical_indices",
]
