# SPDX-License-Identifier: MIT
"""CH08 archive grammar — monolithic single-file ``0.bin`` for nscs06 v8.

Catalog #124 STRICT archive-grammar 8 fields declared in package ``__init__``.
This file IS the export-first grammar (HNeRV parity discipline lesson L2).

**Schema v1 + v2** of the CH08 family:

  - **v1 (canonical)** carries a FULL ``(grayscale_levels, num_segnet_classes, 3)``
    uint8 chroma LUT inline at the canonical 4096-byte budget per canonical
    equation #26 ``_NSCS06_V8_BYTES_SAVED``.
  - **v2 (procedural seed)** replaces the 4096-byte LUT slot with a 32-byte
    PCG64 seed. The inflate runtime re-derives the LUT bytes deterministically
    via ``tac.procedural_codebook_generator.derive_codebook_from_seed`` per
    sister grayscale_lut + DP1 + VQ-VAE canonical PROCEDURAL VARIANT pattern.

    rate-axis savings (predicted-only at landing 2026-05-21):
    ``ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706`` [prediction;
    canonical-equation-26-grounded; per-substrate-symposium-pending].

Schema-version layout::

    MAGIC(4)                  b"CH08"   nscs06 V8 chroma-LUT substrate
    VERSION(1)                u8        1 = inline LUT; 2 = procedural seed
    NUM_PAIRS(2)              u16       number of (frame_0, frame_1) pairs
    GRAYSCALE_H(2)            u16       low-res grayscale field height
    GRAYSCALE_W(2)            u16       low-res grayscale field width
    OUTPUT_HEIGHT(2)          u16       contest eval height (384)
    OUTPUT_WIDTH(2)           u16       contest eval width  (512)
    GRAYSCALE_LEVELS(1)       u8        LUT level dimension (default 16)
    NUM_SEGNET_CLASSES(1)     u8        LUT class dimension (default 5)
    CHROMA_LUT_BYTES(2)       u16       declared chroma LUT footprint (4096)
    LUT_PAYLOAD_LEN(2)        u16       inline LUT bytes OR seed bytes (32)
    POSE_QUANT_SCALE_MICRO(4) u32       pose quantization scale * 1e6 (fixed)
    POSE_LEN(4)               u32       per-pair pose-delta stream length
    GRAYSCALE_LEN(4)          u32       compressed grayscale stream length
    GENERATOR_KIND_TAG(1)     u8        0=xorshift / 1=lcg / 2=pcg64 (v2 only)
    RESERVED(1)               u8        zero-padded reserved byte
    HEADER_SIZE                          = 4+1+2+2+2+2+2+1+1+2+2+4+4+4+1+1 = 35 bytes

    LUT_PAYLOAD             ...     v1: ``CHROMA_LUT_BYTES`` uint8 dense LUT
                                    v2: ``PROCEDURAL_SEED_SIZE_BYTES`` seed
    POSE_STREAM             ...     per-pair pose deltas (uint8 quantized)
    GRAYSCALE_STREAM        ...     raw uint8 quantized grayscale (low-res)

The grammar is FIXED at design-time. CH08 = "nscs06 V8 chroma-LUT substrate"
(the "08" matches the package name suffix ``nscs06_v8_*``).

Compress side writes everything; inflate side reads + decodes purely with
numpy + Pillow + ``tac.procedural_codebook_generator`` (for v2 seed
expansion). NO torch, NO scorer, NO learned weights.

CLAUDE.md compliance:
- Deterministic byte layout (fixed-precision struct format, sorted-keys JSON N/A)
- No /tmp paths
- No scorer load
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .architecture import (
    CHROMA_LUT_BYTES_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
)

__all__ = [
    "CH08_HEADER_FMT",
    "CH08_HEADER_SIZE",
    "CH08_MAGIC",
    "CH08_SCHEMA_VERSION_INLINE_LUT",
    "CH08_SCHEMA_VERSION_PROCEDURAL_SEED",
    "GENERATOR_KIND_TAG",
    "GENERATOR_KIND_TAG_INVERSE",
    "Nscs06V8Archive",
    "POSE_DIMS",
    "POSE_QUANT_SCALE_DEFAULT",
    "pack_archive",
    "parse_archive",
]


CH08_MAGIC: bytes = b"CH08"
"""nscs06 V8 chroma-LUT substrate magic. Matches the ``nscs06_v8_*`` suffix."""

CH08_SCHEMA_VERSION_INLINE_LUT: int = 1
"""v1: full 4096-byte LUT inline (NO procedural seed)."""

CH08_SCHEMA_VERSION_PROCEDURAL_SEED: int = 2
"""v2: 32-byte PCG64 seed replaces the 4096-byte LUT slot.

The inflate runtime re-derives the LUT bytes deterministically via
``tac.procedural_codebook_generator.derive_codebook_from_seed``. Per canonical
equation #26 ``_INCLUDED_CONTEXTS['nscs06_v8_chroma_lut']`` closed-form
``ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706``.
"""

POSE_DIMS: int = 6
"""Per-pair pose vector dimensionality; sister of v7 POSE_DIMS."""

POSE_QUANT_SCALE_DEFAULT: float = 1.0
"""Default pose quantization scale (uint8 around zero=128 over [-127.5/scale, +127.5/scale])."""

GENERATOR_KIND_TAG: dict[str, int] = {"xorshift": 0, "lcg": 1, "pcg64": 2}
"""Sister to grayscale_lut PROCEDURAL_VARIANT `_GENERATOR_KIND_TAG` (canonical)."""

GENERATOR_KIND_TAG_INVERSE: dict[int, str] = {v: k for k, v in GENERATOR_KIND_TAG.items()}

# Header layout per docstring. Format string:
#   < 4s B H H H H H B B H H I I I B B  ->  4+1+2+2+2+2+2+1+1+2+2+4+4+4+1+1 = 35 bytes
CH08_HEADER_FMT: str = "<4sBHHHHHBBHHIIIBB"
CH08_HEADER_SIZE: int = struct.calcsize(CH08_HEADER_FMT)
assert CH08_HEADER_SIZE == 35, (
    f"CH08 header size invariant violated: {CH08_HEADER_SIZE} != 35"
)


@dataclass(frozen=True)
class Nscs06V8Archive:
    """Parsed CH08 archive — the inflate-time data contract.

    Either ``chroma_lut`` (v1) OR ``chroma_seed`` (v2) carries the chroma table
    payload; the inflate runtime resolves which by inspecting ``schema_version``.
    """

    schema_version: int
    num_pairs: int
    grayscale_h: int
    grayscale_w: int
    output_height: int
    output_width: int
    grayscale_levels: int
    num_segnet_classes: int
    chroma_lut_bytes: int
    """Declared chroma-LUT footprint (canonical 4096) — both v1 and v2 declare
    the SAME value so the canonical equation #26 bytes-saved prediction stays
    byte-stable across versions."""
    pose_quant_scale: float
    pose_bytes: bytes
    """Per-pair pose deltas, uint8-quantized (NUM_PAIRS * POSE_DIMS bytes)."""
    grayscale_bytes: bytes
    """Raw uint8 quantized grayscale stream (low-res). One byte per cell."""
    chroma_lut: np.ndarray | None
    """v1: dense (grayscale_levels, num_segnet_classes, 3) uint8 LUT. v2: None."""
    chroma_seed: bytes | None
    """v2: PROCEDURAL_SEED_SIZE_BYTES PCG64 seed. v1: None."""
    generator_kind: str | None
    """v2: 'xorshift' / 'lcg' / 'pcg64'. v1: None."""


def _validate_header_extents(
    *,
    num_pairs: int,
    grayscale_h: int,
    grayscale_w: int,
    output_height: int,
    output_width: int,
    grayscale_levels: int,
    num_segnet_classes: int,
    chroma_lut_bytes: int,
) -> None:
    """Range-validate every header field against its struct format max."""
    for name, value, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("grayscale_h", grayscale_h, 0xFFFF),
        ("grayscale_w", grayscale_w, 0xFFFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
        ("grayscale_levels", grayscale_levels, 0xFF),
        ("num_segnet_classes", num_segnet_classes, 0xFF),
        ("chroma_lut_bytes", chroma_lut_bytes, 0xFFFF),
    ):
        if value <= 0 or value > max_v:
            raise ValueError(f"{name}={value} out of range (max {max_v})")


def pack_archive(
    *,
    num_pairs: int,
    grayscale_h: int,
    grayscale_w: int,
    output_height: int,
    output_width: int,
    pose_bytes: bytes,
    grayscale_bytes: bytes,
    pose_quant_scale: float = POSE_QUANT_SCALE_DEFAULT,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
    chroma_lut_bytes: int = CHROMA_LUT_BYTES_DEFAULT,
    chroma_lut: np.ndarray | None = None,
    chroma_seed: bytes | None = None,
    generator_kind: str = "pcg64",
) -> bytes:
    """Serialize CH08 v1 (inline LUT) or v2 (procedural seed) archive into 0.bin.

    Exactly one of ``chroma_lut`` (v1) or ``chroma_seed`` (v2) MUST be supplied.

    Args:
        num_pairs: Number of (frame_0, frame_1) pairs in the contest video.
        grayscale_h, grayscale_w: Low-res grayscale field dimensions.
        output_height, output_width: Contest eval frame dimensions (384, 512).
        pose_bytes: Quantized per-pair pose deltas (NUM_PAIRS * POSE_DIMS bytes).
        grayscale_bytes: Raw uint8 quantized grayscale stream
            (NUM_PAIRS * grayscale_h * grayscale_w bytes).
        pose_quant_scale: Pose quantization scale (default 1.0).
        grayscale_levels: LUT level dimension (default 16).
        num_segnet_classes: LUT class dimension (default 5).
        chroma_lut_bytes: Declared chroma-LUT footprint (default 4096).
        chroma_lut: v1 payload — dense (grayscale_levels, num_segnet_classes, 3) uint8
            LUT. The dense portion is the first ``min(dense_bytes, chroma_lut_bytes)``
            bytes; remainder is zero-padded to ``chroma_lut_bytes``.
        chroma_seed: v2 payload — PROCEDURAL_SEED_SIZE_BYTES PCG64 seed bytes.
        generator_kind: v2 generator kind (xorshift / lcg / pcg64; default pcg64).

    Returns:
        Monolithic 0.bin bytes per CH08 grammar.
    """
    if (chroma_lut is None) == (chroma_seed is None):
        raise ValueError(
            "pack_archive requires EXACTLY ONE of chroma_lut (v1) or "
            "chroma_seed (v2); got both-None or both-set"
        )
    _validate_header_extents(
        num_pairs=num_pairs,
        grayscale_h=grayscale_h,
        grayscale_w=grayscale_w,
        output_height=output_height,
        output_width=output_width,
        grayscale_levels=grayscale_levels,
        num_segnet_classes=num_segnet_classes,
        chroma_lut_bytes=chroma_lut_bytes,
    )

    expected_pose = num_pairs * POSE_DIMS
    if len(pose_bytes) != expected_pose:
        raise ValueError(
            f"pose_bytes length {len(pose_bytes)} != num_pairs * POSE_DIMS = {expected_pose}"
        )
    expected_grayscale = num_pairs * grayscale_h * grayscale_w
    if len(grayscale_bytes) != expected_grayscale:
        raise ValueError(
            f"grayscale_bytes length {len(grayscale_bytes)} != expected {expected_grayscale}"
        )

    if chroma_lut is not None:
        # v1: pack the dense LUT and zero-pad to chroma_lut_bytes.
        if chroma_lut.dtype != np.uint8:
            raise ValueError(f"chroma_lut must be uint8; got {chroma_lut.dtype}")
        expected_shape = (grayscale_levels, num_segnet_classes, 3)
        if chroma_lut.shape != expected_shape:
            raise ValueError(
                f"chroma_lut shape {chroma_lut.shape} != {expected_shape}"
            )
        dense_bytes = bytes(chroma_lut.tobytes())
        if len(dense_bytes) > chroma_lut_bytes:
            raise ValueError(
                f"chroma_lut dense bytes {len(dense_bytes)} > chroma_lut_bytes {chroma_lut_bytes}"
            )
        lut_payload = dense_bytes + b"\x00" * (chroma_lut_bytes - len(dense_bytes))
        if len(lut_payload) > 0xFFFF:
            raise ValueError(
                f"lut_payload {len(lut_payload)} exceeds u16 LUT_PAYLOAD_LEN field max"
            )
        schema_version = CH08_SCHEMA_VERSION_INLINE_LUT
        generator_kind_tag = 0  # unused in v1
    else:
        # v2: pack the seed.
        seed_view = bytes(chroma_seed)  # type: ignore[arg-type]
        if len(seed_view) != PROCEDURAL_SEED_SIZE_BYTES:
            raise ValueError(
                f"chroma_seed length {len(seed_view)} != PROCEDURAL_SEED_SIZE_BYTES "
                f"{PROCEDURAL_SEED_SIZE_BYTES}"
            )
        if generator_kind not in GENERATOR_KIND_TAG:
            raise ValueError(
                f"generator_kind={generator_kind!r} not in {sorted(GENERATOR_KIND_TAG)}"
            )
        lut_payload = seed_view
        schema_version = CH08_SCHEMA_VERSION_PROCEDURAL_SEED
        generator_kind_tag = GENERATOR_KIND_TAG[generator_kind]

    # Quantize pose_quant_scale to micro-precision uint32 for byte-stable header.
    pose_quant_scale_micro = int(round(pose_quant_scale * 1_000_000))
    if pose_quant_scale_micro < 0 or pose_quant_scale_micro > 0xFFFFFFFF:
        raise ValueError(
            f"pose_quant_scale {pose_quant_scale} -> micro {pose_quant_scale_micro} "
            f"out of u32 range"
        )

    if len(pose_bytes) > 0xFFFFFFFF:
        raise ValueError(f"pose_bytes too large: {len(pose_bytes)}")
    if len(grayscale_bytes) > 0xFFFFFFFF:
        raise ValueError(f"grayscale_bytes too large: {len(grayscale_bytes)}")

    header = struct.pack(
        CH08_HEADER_FMT,
        CH08_MAGIC,
        schema_version,
        num_pairs,
        grayscale_h,
        grayscale_w,
        output_height,
        output_width,
        grayscale_levels,
        num_segnet_classes,
        chroma_lut_bytes,
        len(lut_payload),
        pose_quant_scale_micro,
        len(pose_bytes),
        len(grayscale_bytes),
        generator_kind_tag,
        0,  # RESERVED
    )
    return header + lut_payload + pose_bytes + grayscale_bytes


def parse_archive(blob: bytes) -> Nscs06V8Archive:
    """Parse 0.bin bytes back into a typed Nscs06V8Archive."""
    if len(blob) < CH08_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {CH08_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        grayscale_h,
        grayscale_w,
        output_height,
        output_width,
        grayscale_levels,
        num_segnet_classes,
        chroma_lut_bytes,
        lut_payload_len,
        pose_quant_scale_micro,
        pose_len,
        grayscale_len,
        generator_kind_tag,
        _reserved,
    ) = struct.unpack(CH08_HEADER_FMT, blob[:CH08_HEADER_SIZE])
    if magic != CH08_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {CH08_MAGIC!r})")
    if version not in (
        CH08_SCHEMA_VERSION_INLINE_LUT,
        CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
    ):
        raise ValueError(f"unsupported schema version: {version}")

    expected_pose = num_pairs * POSE_DIMS
    if pose_len != expected_pose:
        raise ValueError(f"pose_len {pose_len} != expected {expected_pose}")
    expected_grayscale = num_pairs * grayscale_h * grayscale_w
    if grayscale_len != expected_grayscale:
        raise ValueError(
            f"grayscale_len {grayscale_len} != expected {expected_grayscale}"
        )

    pos = CH08_HEADER_SIZE
    lut_payload = blob[pos : pos + lut_payload_len]
    pos += lut_payload_len
    pose_bytes = blob[pos : pos + pose_len]
    pos += pose_len
    grayscale_bytes = blob[pos : pos + grayscale_len]
    pos += grayscale_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    pose_quant_scale = pose_quant_scale_micro / 1_000_000

    if version == CH08_SCHEMA_VERSION_INLINE_LUT:
        if lut_payload_len != chroma_lut_bytes:
            raise ValueError(
                f"v1 inline-LUT: lut_payload_len {lut_payload_len} != "
                f"chroma_lut_bytes {chroma_lut_bytes}"
            )
        dense_bytes = grayscale_levels * num_segnet_classes * 3
        dense_view = lut_payload[:dense_bytes]
        chroma_lut = np.frombuffer(dense_view, dtype=np.uint8).copy().reshape(
            grayscale_levels, num_segnet_classes, 3
        )
        chroma_seed: bytes | None = None
        generator_kind: str | None = None
    else:
        if lut_payload_len != PROCEDURAL_SEED_SIZE_BYTES:
            raise ValueError(
                f"v2 procedural-seed: lut_payload_len {lut_payload_len} != "
                f"PROCEDURAL_SEED_SIZE_BYTES {PROCEDURAL_SEED_SIZE_BYTES}"
            )
        chroma_lut = None
        chroma_seed = bytes(lut_payload)
        if generator_kind_tag not in GENERATOR_KIND_TAG_INVERSE:
            raise ValueError(f"unknown generator_kind_tag {generator_kind_tag}")
        generator_kind = GENERATOR_KIND_TAG_INVERSE[generator_kind_tag]

    return Nscs06V8Archive(
        schema_version=int(version),
        num_pairs=int(num_pairs),
        grayscale_h=int(grayscale_h),
        grayscale_w=int(grayscale_w),
        output_height=int(output_height),
        output_width=int(output_width),
        grayscale_levels=int(grayscale_levels),
        num_segnet_classes=int(num_segnet_classes),
        chroma_lut_bytes=int(chroma_lut_bytes),
        pose_quant_scale=float(pose_quant_scale),
        pose_bytes=bytes(pose_bytes),
        grayscale_bytes=bytes(grayscale_bytes),
        chroma_lut=chroma_lut,
        chroma_seed=chroma_seed,
        generator_kind=generator_kind,
    )
