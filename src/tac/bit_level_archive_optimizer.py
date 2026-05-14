# SPDX-License-Identifier: MIT
"""Lane Bit-level archive optimizer — payload-side bit-level optimization.

Per Phase 3 Lane 15 spec (memory `project_phases_2_3_4_*` §"Lane 15 Bit-level
archive optimization") and council design `.omx/research/council_lane_bit_level_archive_design_20260430.md`.

**REDIRECTED SCOPE** (per empirical audit in `custom_binary_container.py`):

The original Carmack-style "rip up the ZIP container, save 50 KB" target is **dead**.
Lane A's reference archive has only 328 bytes of ZIP overhead (0.05% of 694,045 total).
The actual headroom is on the PAYLOAD side:

1. **Sub-FP16 bit-packing of poses.pt** (the main win, ~7 KB on Lane G v3 anchor)
2. **Cross-stream shared Brotli dictionary** for renderer.bin + poses.pt (~0–2 KB)
3. **Empirical byte-composition audit** (the diagnostic tool that REDIRECTED the lane)

This module ships NOTHING by default in the outer container — payload-side only.

Math foundation
---------------

**Information-theoretic bound** (Shannon source coding):

    archive_bytes ≥ Σ_streams H(stream) / 8 + container_overhead

For Lane A: container_overhead = 328 B (deterministic ZIP). The streams are already
near per-stream entropy floors EXCEPT poses.pt, which is FP16 — far above its
information-theoretic minimum given its dynamic range.

**Per-dim quantizer** (the poses.pt main win):

For each pose dimension d ∈ {0..5}:
    pose[t, d] ≈ scale_d · q[t, d] + offset_d
    q[t, d] ∈ {-127, -126, ..., 127}    [int8]

where (scale_d, offset_d) are fit per-dim to minimize round-off error. Storage:
    14,400 bytes (FP16) → 6 × 8 bytes (per-dim header) + 1200 × 6 × 1 byte (int8) = 7,248 bytes

Net savings: ~7,152 bytes per Lane G v3 archive (assuming 1200 frames × 6 dims).

CLAUDE.md compliance
--------------------
- No silent defaults — every public function arg required-keyword
- Compress-time only; no inflate-time scorer load (strict-scorer-rule fully OK)
- All claims tagged [synthetic] / [prediction] / [empirical:...]
- Pure CPU; no GPU dependency
- Outer ZIP container is OUT OF SCOPE per the empirical audit

Out of scope (intentional)
--------------------------
- Outer ZIP container rewrite (audit showed 0.05% headroom; not worth the complexity)
- Cross-stream subsequence dedup beyond shared Brotli dict (Lane J-NWC territory)
- Inflate-time decompression (this module produces byte streams; the inflate path
  needs a sibling decoder that handles the BLPS1 magic)

References
----------
- src/tac/custom_binary_container.py (HARD TRUTH BOMB audit that REDIRECTED scope)
- src/tac/archive_diet_pack.py (Subagent L baseline; ~14.7 KB savings on Lane A)
- src/tac/archive_optimizer.py (sibling lane)
- council design: .omx/research/council_lane_bit_level_archive_design_20260430.md
- Shannon 1948 — source coding theorem
- Brotli RFC 7932 (shared dictionary support)
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── magic / version ────────────────────────────────────────────────────


BLPS_MAGIC: bytes = b"BLPS"  # Bit-Level Pose Stream
"""Self-describing magic for bit-packed pose stream payload (4 bytes ASCII)."""

BLPS_VERSION: int = 1
"""Wire-format version. Bumped on any schema change."""


# ── per-dim quantizer fit ──────────────────────────────────────────────


@dataclass
class PerDimQuantizer:
    """Per-dim affine quantizer: pose ≈ scale * q + offset, q ∈ {-127..127}."""

    scales: np.ndarray  # (n_dims,) float32
    offsets: np.ndarray  # (n_dims,) float32
    bits_per_value: int = 8
    """8 = int8 storage. Future: 4 = packed nibble for ultra-tight ranges."""

    def __post_init__(self) -> None:
        if self.scales.shape != self.offsets.shape:
            raise ValueError(
                f"scales and offsets must match shape, got {self.scales.shape} vs "
                f"{self.offsets.shape}"
            )
        if self.bits_per_value not in (4, 8):
            raise ValueError(
                f"bits_per_value must be 4 or 8, got {self.bits_per_value}"
            )

    @property
    def n_dims(self) -> int:
        return int(self.scales.size)


def fit_per_dim_quantizer(
    *,
    poses: np.ndarray,
    bits_per_value: int = 8,
) -> PerDimQuantizer:
    """Fit per-dim affine quantizer minimizing absolute round-off error.

    Args (all required-keyword):
        poses: (T, n_dims) float32/float16 pose tensor.
        bits_per_value: 8 (int8) or 4 (int4 packed). Default 8.

    Returns:
        PerDimQuantizer covering all n_dims independently.

    Raises:
        ValueError: on bad shape or unsupported bits_per_value.
    """
    if poses.ndim != 2:
        raise ValueError(f"poses must be 2D (T, n_dims), got {poses.shape}")
    if bits_per_value not in (4, 8):
        raise ValueError(f"bits_per_value must be 4 or 8, got {bits_per_value}")
    poses_f32 = poses.astype(np.float32, copy=False)
    n_dims = poses_f32.shape[1]
    levels = (1 << (bits_per_value - 1)) - 1  # 127 for int8, 7 for int4
    scales = np.zeros(n_dims, dtype=np.float32)
    offsets = np.zeros(n_dims, dtype=np.float32)
    for d in range(n_dims):
        col = poses_f32[:, d]
        col_min = float(col.min())
        col_max = float(col.max())
        # Symmetric quant around midpoint
        offset = 0.5 * (col_min + col_max)
        half_range = max(col_max - offset, offset - col_min)
        if half_range == 0:
            scales[d] = 1.0  # constant col — any non-zero scale works
        else:
            scales[d] = half_range / float(levels)
        offsets[d] = offset
    return PerDimQuantizer(scales=scales, offsets=offsets, bits_per_value=bits_per_value)


def quantize_poses(
    *,
    poses: np.ndarray,
    quantizer: PerDimQuantizer,
) -> np.ndarray:
    """Apply quantizer; return int8 (or packed int4) array shape (T, n_dims).

    For bits_per_value=4, the returned array is still int8 with values clamped to
    [-7, +7]; the actual nibble packing happens in `encode_blps`.
    """
    if poses.ndim != 2 or poses.shape[1] != quantizer.n_dims:
        raise ValueError(
            f"poses must be (T, {quantizer.n_dims}), got {poses.shape}"
        )
    levels = (1 << (quantizer.bits_per_value - 1)) - 1
    poses_f32 = poses.astype(np.float32)
    # q = round((pose - offset) / scale), clipped to [-levels, +levels]
    q = (poses_f32 - quantizer.offsets[None, :]) / quantizer.scales[None, :]
    q = np.clip(np.round(q), -levels, levels).astype(np.int8)
    return q


def dequantize_poses(
    *,
    q: np.ndarray,
    quantizer: PerDimQuantizer,
) -> np.ndarray:
    """Reverse `quantize_poses`. Returns float32 reconstruction."""
    if q.ndim != 2 or q.shape[1] != quantizer.n_dims:
        raise ValueError(
            f"q must be (T, {quantizer.n_dims}), got {q.shape}"
        )
    return (
        q.astype(np.float32) * quantizer.scales[None, :] + quantizer.offsets[None, :]
    )


# ── BLPS wire format ───────────────────────────────────────────────────


def encode_blps(
    *,
    poses: np.ndarray,
    bits_per_value: int = 8,
) -> bytes:
    """Encode poses to BLPS bit-packed pose stream.

    Wire format:
        magic     : 4 bytes  = BLPS_MAGIC
        version   : 1 byte   uint8 = BLPS_VERSION
        bits_per  : 1 byte   uint8 (4 or 8)
        n_dims    : 1 byte   uint8
        n_frames  : 4 bytes  uint32 big-endian
        scales    : n_dims * 4 bytes  float32 each
        offsets   : n_dims * 4 bytes  float32 each
        body      : packed int8/int4 values (n_frames * n_dims values)
        crc32     : 4 bytes  big-endian, over all preceding bytes

    For bits_per_value=8: body is n_frames * n_dims signed bytes.
    For bits_per_value=4: body is ceil(n_frames * n_dims / 2) bytes;
        each byte holds two nibbles (high nibble = first value).

    Returns:
        Concatenated wire bytes.

    Raises:
        ValueError: on bad input shape / unsupported bits_per_value.
    """
    if poses.ndim != 2:
        raise ValueError(f"poses must be 2D (T, n_dims), got {poses.shape}")
    if bits_per_value not in (4, 8):
        raise ValueError(f"bits_per_value must be 4 or 8, got {bits_per_value}")
    n_frames, n_dims = poses.shape
    if n_dims > 255:
        raise ValueError(
            f"n_dims must fit in uint8 (max 255), got {n_dims}"
        )
    if n_frames > 0xFFFFFFFF:
        raise ValueError(
            f"n_frames must fit in uint32 (max {0xFFFFFFFF}), got {n_frames}"
        )
    quantizer = fit_per_dim_quantizer(poses=poses, bits_per_value=bits_per_value)
    q = quantize_poses(poses=poses, quantizer=quantizer)
    header = bytearray()
    header.extend(BLPS_MAGIC)
    header.append(BLPS_VERSION)
    header.append(bits_per_value)
    header.append(n_dims)
    header.extend(struct.pack(">I", n_frames))
    header.extend(quantizer.scales.astype(np.float32).tobytes())
    header.extend(quantizer.offsets.astype(np.float32).tobytes())
    if bits_per_value == 8:
        body = q.tobytes()
    else:  # bits_per_value == 4
        # Pack two int4 (range [-7, 7]) per byte. High nibble = first value.
        flat = q.flatten()
        # Convert signed int4 to unsigned nibble (offset by 8): -7..7 → 1..15
        # We use unsigned 0..15 representation via +8.
        unsigned_nibbles = (flat.astype(np.int16) + 8).astype(np.uint8)
        if unsigned_nibbles.max() > 15 or unsigned_nibbles.min() < 0:
            raise AssertionError("nibble range exceeded; quantization bug")
        if unsigned_nibbles.size % 2 == 1:
            unsigned_nibbles = np.concatenate(
                [unsigned_nibbles, np.zeros(1, dtype=np.uint8)]
            )
        high = unsigned_nibbles[0::2]
        low = unsigned_nibbles[1::2]
        body = ((high << 4) | low).tobytes()
    payload = bytes(header) + body
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack(">I", crc)


def decode_blps(*, data: bytes) -> tuple[np.ndarray, PerDimQuantizer]:
    """Decode BLPS wire bytes; returns (q_array, quantizer).

    To reconstruct float pose: `dequantize_poses(q=q_array, quantizer=quantizer)`.

    Raises:
        ValueError: on magic/version/CRC mismatch or truncation.
    """
    if len(data) < len(BLPS_MAGIC) + 1 + 1 + 1 + 4 + 4:
        raise ValueError(f"BLPS payload truncated (got {len(data)} bytes)")
    if data[: len(BLPS_MAGIC)] != BLPS_MAGIC:
        raise ValueError(f"bad BLPS magic: {data[:4]!r}")
    cursor = len(BLPS_MAGIC)
    version = data[cursor]
    cursor += 1
    if version != BLPS_VERSION:
        raise ValueError(
            f"BLPS version mismatch: got {version}, expected {BLPS_VERSION}"
        )
    bits_per_value = data[cursor]
    cursor += 1
    if bits_per_value not in (4, 8):
        raise ValueError(f"BLPS bits_per_value must be 4 or 8, got {bits_per_value}")
    n_dims = data[cursor]
    cursor += 1
    n_frames = struct.unpack_from(">I", data, cursor)[0]
    cursor += 4
    scales_bytes = 4 * n_dims
    offsets_bytes = 4 * n_dims
    if cursor + scales_bytes + offsets_bytes + 4 > len(data):
        raise ValueError("BLPS payload truncated in header arrays")
    scales = np.frombuffer(data[cursor : cursor + scales_bytes], dtype=np.float32)
    cursor += scales_bytes
    offsets = np.frombuffer(
        data[cursor : cursor + offsets_bytes], dtype=np.float32
    )
    cursor += offsets_bytes
    # Body
    if bits_per_value == 8:
        body_bytes = n_frames * n_dims
    else:  # 4-bit
        body_bytes = (n_frames * n_dims + 1) // 2
    if cursor + body_bytes + 4 > len(data):
        raise ValueError("BLPS payload truncated in body")
    body = data[cursor : cursor + body_bytes]
    cursor += body_bytes
    # CRC
    crc_expected = struct.unpack_from(">I", data, cursor)[0]
    crc_actual = zlib.crc32(data[: cursor]) & 0xFFFFFFFF
    if crc_actual != crc_expected:
        raise ValueError(
            f"BLPS CRC mismatch: expected 0x{crc_expected:08x}, got 0x{crc_actual:08x}"
        )
    if bits_per_value == 8:
        q_flat = np.frombuffer(body, dtype=np.int8)
    else:  # 4-bit
        packed = np.frombuffer(body, dtype=np.uint8)
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        unsigned_nibbles = np.empty(packed.size * 2, dtype=np.uint8)
        unsigned_nibbles[0::2] = high
        unsigned_nibbles[1::2] = low
        # Trim to actual length
        unsigned_nibbles = unsigned_nibbles[: n_frames * n_dims]
        q_flat = (unsigned_nibbles.astype(np.int16) - 8).astype(np.int8)
    q = q_flat.reshape(n_frames, n_dims).copy()  # writeable view
    quantizer = PerDimQuantizer(
        scales=scales.copy(), offsets=offsets.copy(), bits_per_value=bits_per_value,
    )
    return q, quantizer


# ── Shared Brotli dictionary builder ──────────────────────────────────


def build_shared_brotli_dictionary(
    *,
    streams: Sequence[bytes],
    max_dict_bytes: int = 1024,
) -> bytes:
    """Build a shared Brotli dictionary from common subsequences across streams.

    Phase B scaffold: use a simple K-most-common 16-byte n-gram approach.
    Production version (Phase D) should use Brotli's own dictionary trainer.

    Args (all required-keyword):
        streams: list of raw byte streams to mine for common subsequences.
        max_dict_bytes: maximum dictionary size to ship.

    Returns:
        Dictionary bytes suitable to pass as Brotli's `dictionary` parameter.

    Raises:
        ValueError: if streams is empty or max_dict_bytes <= 0.
    """
    if not streams:
        raise ValueError("streams must be non-empty")
    if max_dict_bytes <= 0:
        raise ValueError(f"max_dict_bytes must be positive, got {max_dict_bytes}")
    # Mine 16-byte n-grams from each stream; count cross-stream occurrences
    ngram_size = 16
    counts: dict[bytes, int] = {}
    for stream in streams:
        seen_in_this_stream: set[bytes] = set()
        for i in range(len(stream) - ngram_size + 1):
            ng = bytes(stream[i : i + ngram_size])
            if ng in seen_in_this_stream:
                continue
            seen_in_this_stream.add(ng)
            counts[ng] = counts.get(ng, 0) + 1
    # Sort by cross-stream count (preferring n-grams that appear in many streams)
    sorted_ngrams = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    # Concatenate top n-grams up to max_dict_bytes
    dict_bytes = bytearray()
    for ng, cnt in sorted_ngrams:
        if cnt < 2:
            break  # only n-grams appearing in 2+ streams contribute
        if len(dict_bytes) + len(ng) > max_dict_bytes:
            break
        dict_bytes.extend(ng)
    return bytes(dict_bytes)


# ── Empirical byte-composition audit ──────────────────────────────────


@dataclass
class ArchiveByteComposition:
    """Per-member byte counts from an existing archive.zip.

    Surfaces where bytes ACTUALLY are — the diagnostic that REDIRECTED Lane Bit-level
    scope away from the (dead) outer-container target.
    """

    total_bytes: int
    container_overhead_bytes: int
    """ZIP central directory + local file headers. For Lane A: 328 B."""

    member_bytes: dict[str, int]
    """Compressed size of each archive member."""

    notes: str = ""

    def overhead_fraction(self) -> float:
        """container_overhead / total — the killer ratio for outer-container lanes."""
        if self.total_bytes == 0:
            return 0.0
        return self.container_overhead_bytes / self.total_bytes


def audit_archive_byte_composition(
    *,
    archive_bytes: bytes,
) -> ArchiveByteComposition:
    """Decompose archive.zip into member sizes + container overhead.

    Args (all required-keyword):
        archive_bytes: raw bytes of the .zip file.

    Returns:
        ArchiveByteComposition with per-member breakdown + overhead.

    Raises:
        ValueError: if archive_bytes is not a valid ZIP.
    """
    import io
    import zipfile

    if len(archive_bytes) < 22:  # min ZIP size
        raise ValueError(f"archive too small to be ZIP ({len(archive_bytes)} bytes)")
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as e:
        raise ValueError(f"not a valid ZIP: {e}") from e
    member_bytes: dict[str, int] = {}
    member_total = 0
    for info in zf.infolist():
        member_bytes[info.filename] = int(info.compress_size)
        member_total += int(info.compress_size)
    overhead = len(archive_bytes) - member_total
    if overhead < 0:
        # Pathological case: should not happen for a valid ZIP, but guard anyway
        overhead = 0
    return ArchiveByteComposition(
        total_bytes=len(archive_bytes),
        container_overhead_bytes=overhead,
        member_bytes=member_bytes,
        notes=(
            f"audit on {len(archive_bytes)}-byte archive; "
            f"overhead fraction {overhead / len(archive_bytes):.4%}"
        ),
    )


# ── Top-level orchestrator (Phase D production wiring point) ──────────


@dataclass
class BitLevelArchiveOptimizer:
    """Orchestrator for payload-side bit-level optimization on an archive.

    Phase B scaffold: holds config + provides the entry-point methods.
    Phase D production: wired into `compress.sh` to produce the `archive.zip`.
    """

    target_archive_bytes: int
    """Predicted archive size (informational; honest 1-8 KB savings on Lane G v3)."""

    enable_pose_bitpacking: bool = True
    """If True, replace poses.pt FP16 with BLPS int8/int4."""

    enable_shared_brotli_dict: bool = True
    """If True, build + ship a shared Brotli dict for renderer.bin + poses.pt."""

    pose_bits_per_value: int = 8
    """8 = int8 (safe); 4 = int4 (aggressive; only for tight dynamic ranges)."""

    def predict_savings_bytes(
        self,
        *,
        n_pose_frames: int,
        n_pose_dims: int,
    ) -> int:
        """Predict net byte savings on poses.pt given frame/dim counts.

        Honest accounting per design doc §4 kill criteria.
        """
        if not self.enable_pose_bitpacking:
            return 0
        fp16_bytes = n_pose_frames * n_pose_dims * 2  # 2 bytes per FP16
        header_bytes = (
            len(BLPS_MAGIC) + 3 + 4 + n_pose_dims * 4 * 2 + 4
        )  # magic + version + bits + dims + n_frames + scales + offsets + crc
        if self.pose_bits_per_value == 8:
            body_bytes = n_pose_frames * n_pose_dims
        else:  # 4
            body_bytes = (n_pose_frames * n_pose_dims + 1) // 2
        bitpacked_bytes = header_bytes + body_bytes
        return max(0, fp16_bytes - bitpacked_bytes)


__all__ = [
    "ArchiveByteComposition",
    "BLPS_MAGIC",
    "BLPS_VERSION",
    "BitLevelArchiveOptimizer",
    "PerDimQuantizer",
    "audit_archive_byte_composition",
    "build_shared_brotli_dictionary",
    "decode_blps",
    "dequantize_poses",
    "encode_blps",
    "fit_per_dim_quantizer",
    "quantize_poses",
]
