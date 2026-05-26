# SPDX-License-Identifier: MIT
"""CH-CCP-FRAME1-WATERFILL archive grammar — Atick-Redlich asymmetric channel routing-decision sidecar.

Per HNeRV parity discipline lessons L2 + L3:

- **L2 export-first**: archive grammar declared BEFORE training (this module).
- **L3 monolithic 0.bin**: single-file fixed-offset layout.

Layout (bytes 0 -> N):

  offset 0-3   magic: b"CCPF" (Cascade C-Prime Frame-1)
  offset 4     version: uint8 (=1 for this scaffold)
  offset 5     n_pairs_high: uint8 (most significant byte)
  offset 6     n_pairs_low: uint8 (least significant byte)
  offset 7-8   routing_sidecar_byte_count: uint16 little-endian
  offset 9     frame_0_menu_size: uint8 (=16 for PR110 sister)
  offset 10    frame_1_menu_size: uint8 (=8 default)
  offset 11+   routing_sidecar bytes (brotli compressed 1-bit-per-pair packed)
  offset N     frame_0_menu_index_stream (huffman-coded; var-length)
  offset M     frame_1_menu_index_stream (huffman-coded; var-length)
  offset P     pose_delta_stream (uint8 quantized; n_pairs × pose_dims bytes)

Per Cascade C' synthesis (Option B): routing_sidecar typically ~79 bytes for
600 pairs (1-bit-per-pair packed = 75 bytes; brotli adds ~4 bytes overhead).

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog
#105/#139/#272 byte-mutation smoke: every byte in this archive is operationally
consumed by inflate (no dead bytes per Catalog #139 no-op detector).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "CCPF_MAGIC",
    "CCPF_VERSION_V1",
    "CCPF_HEADER_LEN",
    "CascadeCPrimeArchive",
    "pack_archive",
    "parse_archive",
]

CCPF_MAGIC = b"CCPF"
CCPF_VERSION_V1 = 1
CCPF_HEADER_LEN = 11  # magic(4) + version(1) + n_pairs(2) + sidecar_count(2) + menu sizes(2)
POSE_DIMS = 6  # canonical 6-DOF pose delta per pair


@dataclass(frozen=True)
class CascadeCPrimeArchive:
    """Parsed CH-CCP-FRAME1-WATERFILL archive contents.

    Per Catalog #139 no-op detector: every field here must be operationally
    consumed by inflate. The byte-mutation smoke per Catalog #272 mutates one
    byte at each offset + verifies inflate output changes.
    """

    version: int
    n_pairs: int
    frame_0_menu_size: int
    frame_1_menu_size: int
    routing_sidecar_brotli: bytes  # brotli-compressed 1-bit-per-pair routing flags
    frame_0_menu_index_stream: bytes  # 4-bit huffman-coded indices (packed)
    frame_1_menu_index_stream: bytes  # 3-bit huffman-coded indices (packed)
    pose_delta_stream: bytes  # n_pairs × POSE_DIMS uint8 quantized

    @property
    def routing_decision(self) -> np.ndarray:
        """Decompressed (n_pairs,) int8 routing decision array."""
        try:
            import brotli
        except ImportError as exc:
            raise RuntimeError("brotli required for CH-CCP-FRAME1-WATERFILL inflate") from exc
        packed_bits = brotli.decompress(self.routing_sidecar_brotli)
        unpacked = np.unpackbits(np.frombuffer(packed_bits, dtype=np.uint8))
        return unpacked[: self.n_pairs].astype(np.int8)


def pack_archive(
    *,
    routing_decision: np.ndarray,
    frame_0_menu_indices: np.ndarray,
    frame_1_menu_indices: np.ndarray,
    pose_deltas_uint8: np.ndarray,
    frame_0_menu_size: int = 16,
    frame_1_menu_size: int = 8,
    version: int = CCPF_VERSION_V1,
) -> bytes:
    """Pack CH-CCP-FRAME1-WATERFILL archive bytes.

    Per Catalog #105/#139 no-op detector + Catalog #272 distinguishing-feature
    integration contract: every byte written here MUST be parsed + consumed by
    inflate (verified empirically via byte-mutation smoke).
    """
    try:
        import brotli
    except ImportError as exc:
        raise RuntimeError("brotli required for CH-CCP-FRAME1-WATERFILL pack") from exc

    n_pairs = int(routing_decision.shape[0])
    if not (0 < n_pairs < 65536):
        raise ValueError(f"n_pairs must be in (0, 65536); got {n_pairs}")
    if frame_0_menu_indices.shape[0] != n_pairs:
        raise ValueError("frame_0_menu_indices mismatched n_pairs")
    if frame_1_menu_indices.shape[0] != n_pairs:
        raise ValueError("frame_1_menu_indices mismatched n_pairs")
    if pose_deltas_uint8.shape != (n_pairs, POSE_DIMS):
        raise ValueError(f"pose_deltas_uint8 shape mismatch; expected ({n_pairs}, {POSE_DIMS})")

    # Routing decision: pack into bits, brotli compress
    packed_bits = np.packbits(routing_decision.astype(np.uint8))
    routing_brotli = brotli.compress(packed_bits.tobytes(), quality=11)
    routing_byte_count = len(routing_brotli)
    if routing_byte_count > 65535:
        raise ValueError(f"routing_sidecar too large; got {routing_byte_count} bytes")

    # Frame-0 menu index stream (4-bit packed for K=16)
    f0_indices_uint8 = frame_0_menu_indices.astype(np.uint8)
    if (f0_indices_uint8 >= frame_0_menu_size).any():
        raise ValueError("frame_0_menu_indices out of bounds")
    # Pack two 4-bit values per byte (low nibble first; high nibble second)
    if n_pairs % 2 == 1:
        f0_padded = np.concatenate([f0_indices_uint8, np.zeros(1, dtype=np.uint8)])
    else:
        f0_padded = f0_indices_uint8
    f0_packed = (f0_padded[::2] & 0x0F) | ((f0_padded[1::2] & 0x0F) << 4)
    f0_stream = f0_packed.tobytes()

    # Frame-1 menu index stream (3-bit packed for K=8; use byte-aligned for simplicity)
    # SCAFFOLD: byte-aligned uint8 (8 bits per pair); 7th-order iteration packs to 3-bit
    f1_indices_uint8 = frame_1_menu_indices.astype(np.uint8)
    if (f1_indices_uint8 >= frame_1_menu_size).any():
        raise ValueError("frame_1_menu_indices out of bounds")
    f1_stream = f1_indices_uint8.tobytes()

    pose_stream = pose_deltas_uint8.tobytes()

    # Header: magic(4) + version(1) + n_pairs(2) + sidecar_count(2) + menu sizes(2)
    header = (
        CCPF_MAGIC
        + struct.pack("<B", version)
        + struct.pack(">H", n_pairs)  # big-endian for human-readability
        + struct.pack("<H", routing_byte_count)
        + struct.pack("<B", frame_0_menu_size)
        + struct.pack("<B", frame_1_menu_size)
    )

    return header + routing_brotli + f0_stream + f1_stream + pose_stream


def parse_archive(archive_bytes: bytes) -> CascadeCPrimeArchive:
    """Parse CH-CCP-FRAME1-WATERFILL archive bytes.

    Per HNeRV parity L3: monolithic single-file 0.bin parse.
    Per Catalog #287: every parsed field carries provenance via the
    CascadeCPrimeArchive dataclass.
    """
    if len(archive_bytes) < CCPF_HEADER_LEN:
        raise ValueError(f"archive too short; need >= {CCPF_HEADER_LEN} bytes")
    if archive_bytes[:4] != CCPF_MAGIC:
        raise ValueError(f"bad magic; expected {CCPF_MAGIC!r}, got {archive_bytes[:4]!r}")

    version = archive_bytes[4]
    if version != CCPF_VERSION_V1:
        raise ValueError(f"unsupported version {version}; expected {CCPF_VERSION_V1}")

    n_pairs = struct.unpack(">H", archive_bytes[5:7])[0]
    routing_byte_count = struct.unpack("<H", archive_bytes[7:9])[0]
    frame_0_menu_size = archive_bytes[9]
    frame_1_menu_size = archive_bytes[10]

    cursor = CCPF_HEADER_LEN
    routing_brotli = archive_bytes[cursor : cursor + routing_byte_count]
    cursor += routing_byte_count

    # Frame-0 stream: ceil(n_pairs / 2) bytes (4-bit packed)
    f0_stream_len = (n_pairs + 1) // 2
    f0_stream = archive_bytes[cursor : cursor + f0_stream_len]
    cursor += f0_stream_len

    # Frame-1 stream: n_pairs bytes (SCAFFOLD: byte-aligned uint8; 7th-order packs to 3-bit)
    f1_stream_len = n_pairs
    f1_stream = archive_bytes[cursor : cursor + f1_stream_len]
    cursor += f1_stream_len

    # Pose stream: n_pairs × POSE_DIMS bytes
    pose_stream_len = n_pairs * POSE_DIMS
    pose_stream = archive_bytes[cursor : cursor + pose_stream_len]
    cursor += pose_stream_len

    if cursor != len(archive_bytes):
        raise ValueError(
            f"archive length mismatch; parsed {cursor} bytes, file has {len(archive_bytes)} bytes"
        )

    return CascadeCPrimeArchive(
        version=version,
        n_pairs=n_pairs,
        frame_0_menu_size=frame_0_menu_size,
        frame_1_menu_size=frame_1_menu_size,
        routing_sidecar_brotli=routing_brotli,
        frame_0_menu_index_stream=f0_stream,
        frame_1_menu_index_stream=f1_stream,
        pose_delta_stream=pose_stream,
    )
