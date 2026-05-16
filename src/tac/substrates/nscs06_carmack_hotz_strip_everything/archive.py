# SPDX-License-Identifier: MIT
"""CH06 archive grammar — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields declared in package ``__init__``.
This file IS the export-first grammar (HNeRV parity discipline lesson L2)::

    MAGIC(4)             b"CH06"   Carmack-Hotz strip-everything substrate v06
    VERSION(1)           u8        schema version (currently 1)
    NUM_PAIRS(2)         u16       number of (frame_0, frame_1) pairs
    PALETTE_SIZE(1)      u8        grayscale palette size (default 16)
    GRAYSCALE_H(2)       u16       low-res grayscale field height (e.g. H/4)
    GRAYSCALE_W(2)       u16       low-res grayscale field width  (e.g. W/4)
    OUTPUT_HEIGHT(2)     u16       contest eval height (384)
    OUTPUT_WIDTH(2)      u16       contest eval width  (512)
    PALETTE_LEN(2)       u16       PALETTE_SIZE bytes
    CDF_LEN(2)           u16       NUM_SEGNET_CLASSES * (PALETTE_SIZE+1) * 2 bytes
    GRAYSCALE_LEN(4)     u32       arith-coded grayscale-indices stream length
    POSE_LEN(4)          u32       per-pair pose-delta stream length (uint8 * 6 * NUM_PAIRS)
    META_LEN(2)          u16       utf-8 json meta length
    PALETTE              ...       PALETTE_SIZE bytes (uint8 grayscale levels)
    CDF                  ...       class-conditional CDF table (uint16 row-major)
    GRAYSCALE_STREAM     ...       arith-coded palette indices for odd-frame grayscale
    POSE_STREAM          ...       per-pair pose deltas (uint8 quantized; 6 dims per pair)
    META_BLOB            ...       json: {grayscale_downsample, pose_quant_scale, ...}

Total header: 4+1+2+1+2+2+2+2+2+2+4+4+2 = 30 bytes.

The grammar is FIXED at design-time. CH06 = "Carmack-Hotz substrate v06"
(the "06" matches the package name suffix ``nscs06_*``).

Compress side writes everything; inflate side reads + decodes purely with
numpy + Pillow. NO torch, NO scorer, NO learned weights.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON meta, fixed-precision arith state)
- No /tmp paths
- No scorer load
- Reviewable in 30 seconds per L12 (~120 LOC including doc + assertions)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .codec import (
    CDF_MAX,
    NUM_SEGNET_CLASSES,
    ArithmeticCoder,
    ClassConditionalCDF,
    GrayscalePalette,
)

CH06_MAGIC: bytes = b"CH06"
"""Carmack-Hotz substrate magic. Matches the ``nscs06`` package suffix."""

CH06_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout per docstring above. Format string:
#   < 4s B H B H H H H H H I I H  ->  4+1+2+1+2+2+2+2+2+2+4+4+2 = 30 bytes
CH06_HEADER_FMT: str = "<4sBHBHHHHHHIIH"
CH06_HEADER_SIZE: int = struct.calcsize(CH06_HEADER_FMT)
assert CH06_HEADER_SIZE == 30, (
    f"CH06 header size invariant violated: {CH06_HEADER_SIZE} != 30"
)

# Pose deltas are quantized to uint8 over [-pose_quant_range, +pose_quant_range].
# Stored as raw bytes (NO arithmetic coding); only ~3.6 KB total for 600 pairs.
POSE_DIMS: int = 6
"""Per-pair pose vector dimensionality (PoseNet's first 6 dims per CLAUDE.md)."""


@dataclass(frozen=True)
class CarmackHotzArchive:
    """Parsed CH06 archive — the inflate-time data contract."""

    palette: GrayscalePalette
    cdf: ClassConditionalCDF
    grayscale_arith_bytes: bytes
    """Arith-coded palette indices for the odd-frame grayscale stream."""
    pose_bytes: bytes
    """Per-pair pose deltas, uint8-quantized (NUM_PAIRS * POSE_DIMS bytes)."""
    meta: dict[str, object]
    schema_version: int
    num_pairs: int
    grayscale_h: int
    grayscale_w: int
    output_height: int
    output_width: int

    @property
    def palette_size(self) -> int:
        return self.palette.size

    @property
    def grayscale_downsample(self) -> int:
        return int(self.meta.get("grayscale_downsample", 4))

    @property
    def pose_quant_scale(self) -> float:
        return float(self.meta.get("pose_quant_scale", 1.0))

    @property
    def pose_quant_zero(self) -> int:
        return int(self.meta.get("pose_quant_zero", 128))


def pack_archive(
    *,
    palette: GrayscalePalette,
    cdf: ClassConditionalCDF,
    grayscale_arith_bytes: bytes,
    pose_bytes: bytes,
    meta: dict[str, object],
    num_pairs: int,
    grayscale_h: int,
    grayscale_w: int,
    output_height: int,
    output_width: int,
    schema_version: int = CH06_SCHEMA_VERSION,
) -> bytes:
    """Serialize CH06 archive into monolithic 0.bin bytes."""
    if schema_version != CH06_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if palette.size != cdf.palette_size:
        raise ValueError(
            f"palette.size={palette.size} != cdf.palette_size={cdf.palette_size}"
        )
    if len(pose_bytes) != num_pairs * POSE_DIMS:
        raise ValueError(
            f"pose_bytes length {len(pose_bytes)} != num_pairs * POSE_DIMS = "
            f"{num_pairs * POSE_DIMS}"
        )

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("palette_size", palette.size, 0xFF),
        ("grayscale_h", grayscale_h, 0xFFFF),
        ("grayscale_w", grayscale_w, 0xFFFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of range (max {max_v})")

    palette_bytes = bytes(palette.levels.tobytes())
    cdf_bytes = bytes(cdf.cdf.tobytes())
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    if len(palette_bytes) > 0xFFFF:
        raise ValueError(f"palette too large: {len(palette_bytes)} > {0xFFFF}")
    if len(cdf_bytes) > 0xFFFF:
        raise ValueError(f"cdf too large: {len(cdf_bytes)} > {0xFFFF}")
    if len(grayscale_arith_bytes) > 0xFFFFFFFF:
        raise ValueError(f"grayscale stream too large: {len(grayscale_arith_bytes)}")
    if len(pose_bytes) > 0xFFFFFFFF:
        raise ValueError(f"pose stream too large: {len(pose_bytes)}")
    if len(meta_bytes) > 0xFFFF:
        raise ValueError(f"meta too large: {len(meta_bytes)} > {0xFFFF}")

    header = struct.pack(
        CH06_HEADER_FMT,
        CH06_MAGIC,
        schema_version,
        num_pairs,
        palette.size,
        grayscale_h,
        grayscale_w,
        output_height,
        output_width,
        len(palette_bytes),
        len(cdf_bytes),
        len(grayscale_arith_bytes),
        len(pose_bytes),
        len(meta_bytes),
    )
    return (
        header
        + palette_bytes
        + cdf_bytes
        + grayscale_arith_bytes
        + pose_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> CarmackHotzArchive:
    """Parse 0.bin bytes back into a typed CarmackHotzArchive."""
    if len(blob) < CH06_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {CH06_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        palette_size,
        grayscale_h,
        grayscale_w,
        output_height,
        output_width,
        palette_len,
        cdf_len,
        grayscale_len,
        pose_len,
        meta_len,
    ) = struct.unpack(CH06_HEADER_FMT, blob[:CH06_HEADER_SIZE])
    if magic != CH06_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {CH06_MAGIC!r})")
    if version != CH06_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_palette = palette_size * 1
    if palette_len != expected_palette:
        raise ValueError(
            f"palette_len {palette_len} != expected {expected_palette}"
        )
    expected_cdf = NUM_SEGNET_CLASSES * (palette_size + 1) * 2
    if cdf_len != expected_cdf:
        raise ValueError(f"cdf_len {cdf_len} != expected {expected_cdf}")
    expected_pose = num_pairs * POSE_DIMS
    if pose_len != expected_pose:
        raise ValueError(f"pose_len {pose_len} != expected {expected_pose}")

    pos = CH06_HEADER_SIZE
    palette_bytes = blob[pos : pos + palette_len]
    pos += palette_len
    cdf_bytes = blob[pos : pos + cdf_len]
    pos += cdf_len
    grayscale_arith_bytes = blob[pos : pos + grayscale_len]
    pos += grayscale_len
    pose_bytes = blob[pos : pos + pose_len]
    pos += pose_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    palette_levels = np.frombuffer(palette_bytes, dtype=np.uint8).copy()
    palette = GrayscalePalette(levels=palette_levels)

    cdf_arr = np.frombuffer(cdf_bytes, dtype=np.uint16).copy().reshape(
        NUM_SEGNET_CLASSES, palette_size + 1
    )
    # Defensive: enforce monotonicity + endpoints (in case bytes were tampered;
    # this also doubles as the Catalog #139 byte-mutation acceptance criterion).
    cdf_arr[:, 0] = 0
    cdf_arr[:, -1] = CDF_MAX
    for c in range(NUM_SEGNET_CLASSES):
        for i in range(1, cdf_arr.shape[1]):
            if cdf_arr[c, i] < cdf_arr[c, i - 1]:
                cdf_arr[c, i] = cdf_arr[c, i - 1]
    cdf = ClassConditionalCDF(cdf=cdf_arr.astype(np.uint16))

    meta = json.loads(meta_blob.decode("utf-8"))
    return CarmackHotzArchive(
        palette=palette,
        cdf=cdf,
        grayscale_arith_bytes=bytes(grayscale_arith_bytes),
        pose_bytes=bytes(pose_bytes),
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        grayscale_h=int(grayscale_h),
        grayscale_w=int(grayscale_w),
        output_height=int(output_height),
        output_width=int(output_width),
    )


def decode_grayscale_stream(
    arc: CarmackHotzArchive,
    *,
    class_labels_lowres: np.ndarray,
) -> np.ndarray:
    """Arithmetic-decode the per-cell palette indices for odd-frame grayscale.

    The inflate runtime calls this with the per-cell class-label map, which
    is derived from the archived pose deltas + the previous frame's decoded
    grayscale (a deterministic chicken-and-egg break: we use a uniform
    class-prior for the FIRST decode pass, then refine via the per-class CDF).

    For the L1 SCAFFOLD path we use class label 0 uniformly (CDF row 0 acts
    as the canonical prior). Future iteration: chain the prior through pose.

    Returns:
        uint8 palette indices, shape ``(num_pairs, grayscale_h, grayscale_w)``.
    """
    if class_labels_lowres.dtype != np.uint8:
        raise ValueError("class_labels_lowres must be uint8")
    expected_shape = (arc.num_pairs, arc.grayscale_h, arc.grayscale_w)
    if class_labels_lowres.shape != expected_shape:
        raise ValueError(
            f"class_labels_lowres shape {class_labels_lowres.shape} != {expected_shape}"
        )
    coder = ArithmeticCoder.from_bytes(arc.grayscale_arith_bytes)
    out = np.zeros(expected_shape, dtype=np.uint8)
    cdf_arr = arc.cdf.cdf
    for p in range(arc.num_pairs):
        for y in range(arc.grayscale_h):
            for x in range(arc.grayscale_w):
                cls = int(class_labels_lowres[p, y, x])
                row = cdf_arr[cls]
                sym = coder.decode_symbol(row)
                out[p, y, x] = np.uint8(sym)
    return out


def encode_grayscale_stream(
    *,
    palette_indices: np.ndarray,
    class_labels: np.ndarray,
    cdf: ClassConditionalCDF,
) -> bytes:
    """Arithmetic-encode palette indices using per-pixel class-conditional CDF.

    Mirror of :func:`decode_grayscale_stream`. Compress-side helper.
    """
    if palette_indices.shape != class_labels.shape:
        raise ValueError("shape mismatch palette_indices vs class_labels")
    coder = ArithmeticCoder()
    flat_pi = palette_indices.ravel()
    flat_cls = class_labels.ravel()
    cdf_arr = cdf.cdf
    for sym, cls in zip(flat_pi, flat_cls, strict=True):
        row = cdf_arr[int(cls)]
        coder.encode_symbol(int(sym), row)
    return coder.finish_encoding()


def quantize_pose_deltas(pose: np.ndarray, *, scale: float = 1.0) -> tuple[bytes, int]:
    """Quantize float pose deltas to uint8 around a zero of 128.

    Returns (bytes, zero_point). The reverse operation is
    ``(int(byte) - zero_point) / scale``.
    """
    if pose.ndim != 2 or pose.shape[1] != POSE_DIMS:
        raise ValueError(
            f"pose must be (N, {POSE_DIMS}); got {pose.shape}"
        )
    zero = 128
    q = np.clip(np.round(pose * scale + zero), 0, 255).astype(np.uint8)
    return q.tobytes(), zero


def dequantize_pose_deltas(
    pose_bytes: bytes, *, num_pairs: int, scale: float, zero: int
) -> np.ndarray:
    """Inverse of :func:`quantize_pose_deltas`."""
    arr = np.frombuffer(pose_bytes, dtype=np.uint8).copy().reshape(num_pairs, POSE_DIMS)
    return (arr.astype(np.float32) - float(zero)) / float(scale)
