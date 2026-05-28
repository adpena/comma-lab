# SPDX-License-Identifier: MIT
"""NSCS06 v8 chroma_lut + cls_stream PR-95-parity codec.

Per HNeRV parity discipline lessons L2 (export-first design) + L3 (monolithic
single-file ``0.bin`` archive grammar with fixed offsets declared in source)
+ L7 (substrate engineering exceeds bolt-on size; tagged ``lane_class=
substrate_engineering``) + L12 (single-LOC-per-LOC review discipline).

This file IS the canonical archive grammar. CH09 = "nscs06 V8 chroma-LUT
+ cls_stream PR-95-parity substrate" (the suffix mirrors the package name).

**Archive grammar (declared HERE per L2, NOT in a sister doc)**::

    MAGIC(4)                    b"CH09"
    SCHEMA_VERSION(1)           u8   2 = procedural seed + cls_stream
    NUM_PAIRS(2)                u16
    GRAYSCALE_H(2)              u16
    GRAYSCALE_W(2)              u16
    OUTPUT_HEIGHT(2)            u16  contest eval height (384 advisory; 874 contest)
    OUTPUT_WIDTH(2)             u16  contest eval width  (512 advisory; 1164 contest)
    GRAYSCALE_LEVELS(1)         u8   LUT level dimension (default 16)
    NUM_SEGNET_CLASSES(1)       u8   LUT class dimension (default 5)
    CHROMA_LUT_BYTES(2)         u16  declared chroma LUT footprint (4096)
    SEED_LEN(2)                 u16  procedural seed bytes (32)
    POSE_QUANT_SCALE_MICRO(4)   u32  pose quantization scale * 1e6
    POSE_LEN(4)                 u32  per-pair pose-delta stream length
    GRAYSCALE_LEN(4)            u32  compressed grayscale stream length
    CLS_LEN(4)                  u32  compressed cls_stream length
    HEADER_SIZE                 = 4+1+2+2+2+2+2+1+1+2+2+4+4+4+4 = 37 bytes

    [CHROMA_SEED]               SEED_LEN  bytes (PCG64 32-byte seed)
    [POSE_STREAM]               POSE_LEN  bytes (per-pair pose deltas)
    [GRAYSCALE_STREAM]          GRAYSCALE_LEN bytes (low-res uint8 grayscale)
    [CLS_STREAM]                CLS_LEN bytes (low-res uint8 SegNet class)

The grammar is FIXED at design-time. Per L3 the archive is a single ``0.bin``
member; per Catalog #205 the inflate runtime selects device via canonical
helper; per L9 runtime closure is numpy + Pillow + canonical
``tac.procedural_codebook_generator`` only.

CLAUDE.md compliance:
- No scorer load at inflate time (strict-scorer-rule)
- No /tmp paths in persisted artifacts (transient-evidence trap)
- No silent device defaults (Catalog #205 select_inflate_device)
- Deterministic byte layout (fixed struct format)
- Sorted-keys N/A (binary archive)

# AUTOCAST_FP16_WAIVED:numpy_codec_no_training_no_autocast_needed
# TF32_WAIVED:numpy_codec_no_cuda_matmul
# TORCH_COMPILE_WAIVED:numpy_codec_no_torch
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = [
    "CH09_HEADER_FMT",
    "CH09_HEADER_SIZE",
    "CH09_MAGIC",
    "CH09_SCHEMA_VERSION_PR95_PARITY",
    "CHROMA_LUT_BYTES_DEFAULT",
    "DECODER_BLOB_LEN",
    "GRAYSCALE_LEVELS_DEFAULT",
    "LATENT_BLOB_LEN",
    "NUM_SEGNET_CLASSES",
    "Nscs06V8Pr95ParityArchive",
    "Nscs06V8Pr95ParityConfig",
    "POSE_DIMS",
    "POSE_QUANT_SCALE_DEFAULT",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "build_chroma_lut_from_ground_truth",
    "derive_chroma_lut_bytes_from_seed",
    "lookup_rgb_via_chroma_lut",
    "pack_archive",
    "parse_archive",
]


# ---------------------------------------------------------------------------
# Canonical constants per HNeRV parity L3 (fixed offsets declared in source)
# ---------------------------------------------------------------------------

CH09_MAGIC: Final[bytes] = b"CH09"
"""nscs06 V8 chroma-LUT + cls_stream PR-95-parity substrate magic."""

CH09_SCHEMA_VERSION_PR95_PARITY: Final[int] = 2
"""PR-95-parity schema: 32-byte PCG64 seed + cls_stream consumption mandatory."""

CH09_HEADER_FMT: Final[str] = "<4sBHHHHHBBHHIIII"
"""Fixed struct format. Little-endian. Total = 37 bytes."""

CH09_HEADER_SIZE: Final[int] = struct.calcsize(CH09_HEADER_FMT)
assert CH09_HEADER_SIZE == 37, f"CH09 header size {CH09_HEADER_SIZE} != 37"

POSE_DIMS: Final[int] = 6
"""6-DOF pose: (tx, ty, tz, rx, ry, rz)."""

POSE_QUANT_SCALE_DEFAULT: Final[float] = 32.0
"""Pose quantization scale: uint8 = (float * SCALE) + 128, clipped [0, 255]."""

NUM_SEGNET_CLASSES: Final[int] = 5
"""Matches upstream/modules.py SegNet classes=5."""

GRAYSCALE_LEVELS_DEFAULT: Final[int] = 16
"""4-bit luma quantization for chroma-LUT indexing."""

CHROMA_LUT_BYTES_DEFAULT: Final[int] = 4096
"""Canonical chroma LUT footprint per canonical equation #26."""

PROCEDURAL_SEED_SIZE_BYTES: Final[int] = 32
"""Canonical PCG64 32-byte seed (sister DP1 + VQ-VAE + grayscale_lut pattern)."""

# Per HNeRV parity L3 mandate: fixed offsets declared at module level.
DECODER_BLOB_LEN: Final[int] = PROCEDURAL_SEED_SIZE_BYTES
"""Decoder blob = procedural seed (32 bytes). Equivalent to PR101's DECODER_BLOB_LEN."""

LATENT_BLOB_LEN: Final[int] = 0
"""LATENT_BLOB_LEN = 0: NSCS06 v8 has NO learned latents (closed-form LUT)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Nscs06V8Pr95ParityConfig:
    """Frozen configuration for the PR-95-parity packet.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #290.
    """

    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT
    num_segnet_classes: int = NUM_SEGNET_CLASSES
    chroma_lut_bytes: int = CHROMA_LUT_BYTES_DEFAULT
    seed_size_bytes: int = PROCEDURAL_SEED_SIZE_BYTES
    pose_quant_scale: float = POSE_QUANT_SCALE_DEFAULT
    generator_kind: str = "pcg64"
    # SCAFFOLD: cls_stream consumption mandatory at L1 per Wave N+22 wire-in
    require_cls_stream: bool = True

    def __post_init__(self) -> None:
        """Validate canonical invariants per Catalog #287."""
        if self.grayscale_levels < 1 or self.grayscale_levels > 256:
            raise ValueError(f"grayscale_levels={self.grayscale_levels} outside [1, 256]")
        if self.num_segnet_classes < 1 or self.num_segnet_classes > 32:
            raise ValueError(f"num_segnet_classes={self.num_segnet_classes} outside [1, 32]")
        if self.chroma_lut_bytes < 1 or self.chroma_lut_bytes > 65535:
            raise ValueError(f"chroma_lut_bytes={self.chroma_lut_bytes} outside [1, 65535]")
        if self.seed_size_bytes != PROCEDURAL_SEED_SIZE_BYTES:
            raise ValueError(f"seed_size_bytes={self.seed_size_bytes} != canonical {PROCEDURAL_SEED_SIZE_BYTES}")
        if self.generator_kind not in {"pcg64", "xorshift", "lcg"}:
            raise ValueError(f"generator_kind={self.generator_kind!r} not in {{pcg64,xorshift,lcg}}")
        # Validate dense (levels, classes, 3) fits in chroma_lut_bytes budget
        min_required = self.grayscale_levels * self.num_segnet_classes * 3
        if self.chroma_lut_bytes < min_required:
            raise ValueError(
                f"chroma_lut_bytes={self.chroma_lut_bytes} < minimum required "
                f"{min_required} (levels * classes * 3)"
            )

    @property
    def chroma_lut_shape(self) -> tuple[int, int, int]:
        return (self.grayscale_levels, self.num_segnet_classes, 3)


# ---------------------------------------------------------------------------
# Parsed archive container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Nscs06V8Pr95ParityArchive:
    """Parsed CH09 PR-95-parity archive (returned by ``parse_archive``)."""

    schema_version: int
    num_pairs: int
    grayscale_h: int
    grayscale_w: int
    output_height: int
    output_width: int
    grayscale_levels: int
    num_segnet_classes: int
    chroma_lut_bytes: int
    chroma_seed: bytes
    pose_quant_scale: float
    pose_bytes: bytes
    grayscale_bytes: bytes
    cls_bytes: bytes
    generator_kind: str = "pcg64"

    def __post_init__(self) -> None:
        if self.schema_version != CH09_SCHEMA_VERSION_PR95_PARITY:
            raise ValueError(
                f"unsupported schema_version {self.schema_version}; expected {CH09_SCHEMA_VERSION_PR95_PARITY}"
            )
        if len(self.chroma_seed) != PROCEDURAL_SEED_SIZE_BYTES:
            raise ValueError(f"chroma_seed len={len(self.chroma_seed)} != {PROCEDURAL_SEED_SIZE_BYTES}")
        expected_pose_len = self.num_pairs * POSE_DIMS
        if len(self.pose_bytes) != expected_pose_len:
            raise ValueError(f"pose_bytes len={len(self.pose_bytes)} != {expected_pose_len}")
        expected_gray_len = self.num_pairs * self.grayscale_h * self.grayscale_w
        if len(self.grayscale_bytes) != expected_gray_len:
            raise ValueError(f"grayscale_bytes len={len(self.grayscale_bytes)} != {expected_gray_len}")
        expected_cls_len = expected_gray_len  # cls_stream at same low-res as grayscale
        if len(self.cls_bytes) != expected_cls_len:
            raise ValueError(f"cls_bytes len={len(self.cls_bytes)} != {expected_cls_len}")


# ---------------------------------------------------------------------------
# Pack / parse (canonical archive grammar per L2 + L3)
# ---------------------------------------------------------------------------


def pack_archive(
    *,
    chroma_seed: bytes,
    pose_quantized_u8: np.ndarray,
    grayscale_lowres_u8: np.ndarray,
    cls_lowres_u8: np.ndarray,
    output_height: int = 384,
    output_width: int = 512,
    config: Nscs06V8Pr95ParityConfig | None = None,
) -> bytes:
    """Pack the PR-95-parity archive to canonical CH09 byte layout.

    Args:
        chroma_seed: 32-byte PCG64 seed for chroma LUT derivation
        pose_quantized_u8: (N, POSE_DIMS) uint8 quantized pose deltas
        grayscale_lowres_u8: (N, H_low, W_low) uint8 low-res luma stream
        cls_lowres_u8: (N, H_low, W_low) uint8 low-res SegNet class stream
        output_height: contest eval output height
        output_width: contest eval output width
        config: optional config (defaults to canonical)

    Returns:
        bytes: canonical CH09 archive payload (single 0.bin member)
    """
    cfg = config or Nscs06V8Pr95ParityConfig()
    if len(chroma_seed) != cfg.seed_size_bytes:
        raise ValueError(f"chroma_seed len={len(chroma_seed)} != {cfg.seed_size_bytes}")
    if pose_quantized_u8.dtype != np.uint8 or pose_quantized_u8.ndim != 2:
        raise ValueError(f"pose_quantized_u8 must be 2D uint8; got {pose_quantized_u8.shape} {pose_quantized_u8.dtype}")
    if pose_quantized_u8.shape[1] != POSE_DIMS:
        raise ValueError(f"pose_quantized_u8 second dim {pose_quantized_u8.shape[1]} != POSE_DIMS={POSE_DIMS}")
    if grayscale_lowres_u8.dtype != np.uint8 or grayscale_lowres_u8.ndim != 3:
        raise ValueError(f"grayscale_lowres_u8 must be 3D uint8; got {grayscale_lowres_u8.shape} {grayscale_lowres_u8.dtype}")
    if cls_lowres_u8.dtype != np.uint8 or cls_lowres_u8.shape != grayscale_lowres_u8.shape:
        raise ValueError(f"cls_lowres_u8 shape {cls_lowres_u8.shape} != grayscale_lowres_u8 shape {grayscale_lowres_u8.shape}")
    num_pairs, gray_h, gray_w = grayscale_lowres_u8.shape
    if num_pairs != pose_quantized_u8.shape[0]:
        raise ValueError(f"num_pairs mismatch: pose={pose_quantized_u8.shape[0]} vs grayscale={num_pairs}")
    pose_bytes = pose_quantized_u8.tobytes()
    grayscale_bytes = grayscale_lowres_u8.tobytes()
    cls_bytes = cls_lowres_u8.tobytes()
    pose_quant_micro = int(round(cfg.pose_quant_scale * 1e6))
    header = struct.pack(
        CH09_HEADER_FMT,
        CH09_MAGIC,
        CH09_SCHEMA_VERSION_PR95_PARITY,
        num_pairs,
        gray_h,
        gray_w,
        output_height,
        output_width,
        cfg.grayscale_levels,
        cfg.num_segnet_classes,
        cfg.chroma_lut_bytes,
        cfg.seed_size_bytes,
        pose_quant_micro,
        len(pose_bytes),
        len(grayscale_bytes),
        len(cls_bytes),
    )
    return header + chroma_seed + pose_bytes + grayscale_bytes + cls_bytes


def parse_archive(archive_bytes: bytes) -> Nscs06V8Pr95ParityArchive:
    """Parse CH09 PR-95-parity archive bytes (single 0.bin member)."""
    if len(archive_bytes) < CH09_HEADER_SIZE:
        raise ValueError(f"archive too short: {len(archive_bytes)} < {CH09_HEADER_SIZE}")
    header = struct.unpack_from(CH09_HEADER_FMT, archive_bytes, 0)
    (magic, schema_version, num_pairs, gray_h, gray_w, out_h, out_w, gray_levels,
     num_classes, chroma_lut_bytes, seed_len, pose_quant_micro, pose_len, gray_len, cls_len) = header
    if magic != CH09_MAGIC:
        raise ValueError(f"bad magic {magic!r} != {CH09_MAGIC!r}")
    pose_quant_scale = pose_quant_micro / 1e6
    offset = CH09_HEADER_SIZE
    chroma_seed = archive_bytes[offset:offset + seed_len]
    offset += seed_len
    pose_bytes = archive_bytes[offset:offset + pose_len]
    offset += pose_len
    grayscale_bytes = archive_bytes[offset:offset + gray_len]
    offset += gray_len
    cls_bytes = archive_bytes[offset:offset + cls_len]
    return Nscs06V8Pr95ParityArchive(
        schema_version=schema_version,
        num_pairs=num_pairs,
        grayscale_h=gray_h,
        grayscale_w=gray_w,
        output_height=out_h,
        output_width=out_w,
        grayscale_levels=gray_levels,
        num_segnet_classes=num_classes,
        chroma_lut_bytes=chroma_lut_bytes,
        chroma_seed=chroma_seed,
        pose_quant_scale=pose_quant_scale,
        pose_bytes=pose_bytes,
        grayscale_bytes=grayscale_bytes,
        cls_bytes=cls_bytes,
    )


# ---------------------------------------------------------------------------
# Chroma LUT derivation (compress side + inflate side both call this)
# ---------------------------------------------------------------------------


def derive_chroma_lut_bytes_from_seed(
    seed_bytes: bytes,
    *,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
    generator_kind: str = "pcg64",
) -> np.ndarray:
    """Deterministically derive (levels, classes, 3) uint8 chroma LUT from seed.

    Routes through the canonical helper at
    ``tac.procedural_codebook_generator.derive_codebook_from_seed`` per
    Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES decision.
    """
    if len(seed_bytes) != PROCEDURAL_SEED_SIZE_BYTES:
        raise ValueError(f"seed_bytes len={len(seed_bytes)} != {PROCEDURAL_SEED_SIZE_BYTES}")
    from tac.procedural_codebook_generator import derive_codebook_from_seed
    flat = derive_codebook_from_seed(
        seed_bytes=seed_bytes,
        output_shape=(grayscale_levels * num_segnet_classes * 3,),
        dtype=np.uint8,
        generator_kind=generator_kind,
    )
    return flat.reshape(grayscale_levels, num_segnet_classes, 3)


def build_chroma_lut_from_ground_truth(
    rgb_pairs: np.ndarray,
    class_labels: np.ndarray,
    *,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
) -> np.ndarray:
    """Derive (levels, classes, 3) chroma LUT via per-bin median over GT.

    Compress-side derivation (HARD-EARNED case). For each (level, class) bin,
    compute median (R, G, B) across compress-time pixels whose luma quantizes
    to that level AND whose SegNet argmax equals that class. Empty bins fall
    back to per-class global median (v7 anchor).

    Per Catalog #290 UNIQUE-BECAUSE-PRINCIPLED-MISMATCH: this derivation is
    v8-specific (sister v7 has per-class only; v8 adds per-level conditioning).
    """
    if rgb_pairs.dtype != np.uint8 or rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise ValueError(f"rgb_pairs must be (N, 3, H, W) uint8; got {rgb_pairs.shape} {rgb_pairs.dtype}")
    if class_labels.dtype != np.uint8 or class_labels.ndim != 3:
        raise ValueError(f"class_labels must be (N, H, W) uint8; got {class_labels.shape} {class_labels.dtype}")
    n, _, h, w = rgb_pairs.shape
    if class_labels.shape != (n, h, w):
        raise ValueError(f"class_labels shape {class_labels.shape} != ({n}, {h}, {w})")
    r = rgb_pairs[:, 0].astype(np.float32)
    g = rgb_pairs[:, 1].astype(np.float32)
    b = rgb_pairs[:, 2].astype(np.float32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b).clip(0.0, 255.0)
    level_step = max(1, 256 // grayscale_levels)
    level_idx = np.clip((luma // level_step).astype(np.int64), 0, grayscale_levels - 1)
    lut = np.zeros((grayscale_levels, num_segnet_classes, 3), dtype=np.uint8)
    rgb_flat = rgb_pairs.transpose(1, 0, 2, 3).reshape(3, -1)
    cls_flat = class_labels.reshape(-1).astype(np.int64)
    level_flat = level_idx.reshape(-1)
    for c in range(num_segnet_classes):
        cls_mask = cls_flat == c
        if cls_mask.any():
            global_median = np.array(
                [np.median(rgb_flat[ch][cls_mask]) for ch in range(3)], dtype=np.uint8,
            )
        else:
            global_median = np.array([128, 128, 128], dtype=np.uint8)
        for lvl in range(grayscale_levels):
            bin_mask = cls_mask & (level_flat == lvl)
            if bin_mask.any():
                for ch in range(3):
                    lut[lvl, c, ch] = np.uint8(np.median(rgb_flat[ch][bin_mask]))
            else:
                lut[lvl, c, :] = global_median
    return lut


def lookup_rgb_via_chroma_lut(
    gray_u8: np.ndarray,
    cls_u8: np.ndarray,
    chroma_lut: np.ndarray,
) -> np.ndarray:
    """Per-pixel RGB lookup via (level, class) chroma LUT.

    Per HNeRV parity L5: returns FULL RGB frame (NOT mask only).
    Cargo-cult #2 (Y=R=G=B chroma destruction) stays UNWOUND; v8 strengthens
    by indexing chroma on BOTH (level, class) rather than (class) alone.
    """
    if gray_u8.dtype != np.uint8 or cls_u8.dtype != np.uint8:
        raise ValueError("gray_u8 and cls_u8 must be uint8")
    if gray_u8.shape != cls_u8.shape:
        raise ValueError(f"shape mismatch {gray_u8.shape} vs {cls_u8.shape}")
    if chroma_lut.ndim != 3 or chroma_lut.shape[2] != 3:
        raise ValueError(f"chroma_lut must be (levels, classes, 3); got {chroma_lut.shape}")
    grayscale_levels, num_segnet_classes, _ = chroma_lut.shape
    level_step = max(1, 256 // grayscale_levels)
    level_idx = np.clip((gray_u8.astype(np.int64) // level_step), 0, grayscale_levels - 1)
    cls_idx = np.clip(cls_u8.astype(np.int64), 0, num_segnet_classes - 1)
    out = chroma_lut[level_idx, cls_idx]
    return np.ascontiguousarray(out, dtype=np.uint8)
