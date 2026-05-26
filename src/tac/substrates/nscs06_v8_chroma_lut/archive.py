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
    "CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM",
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
"""v1: full 4096-byte LUT inline (NO procedural seed). cls_stream NOT carried;
inflate uses cls=0 uniform per L0 SCAFFOLD legacy behavior."""

CH08_SCHEMA_VERSION_PROCEDURAL_SEED: int = 2
"""v2: 32-byte PCG64 seed replaces the 4096-byte LUT slot.

The inflate runtime re-derives the LUT bytes deterministically via
``tac.procedural_codebook_generator.derive_codebook_from_seed``. Per canonical
equation #26 ``_INCLUDED_CONTEXTS['nscs06_v8_chroma_lut']`` closed-form
``ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706``. cls_stream NOT carried;
inflate uses cls=0 uniform per L0 SCAFFOLD legacy behavior.
"""

CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM: int = 3
"""v3: 32-byte PCG64 seed + per-cell uint8 class-label stream (CLS_STREAM).

Per T3 council #1335 PROCEED_WITH_REVISIONS REVISION #2 (Yousfi BLOCKER):
the cls_stream wire-in at L0 inflate is the L1→L2 promotion canonical 4-gate
unblocker per Catalog #233. Cargo-cult #5 unwinding (`distinguishing_feature_smoke`
verdict transitions FAIL_AT_CLASS_1 → PASS_PER_CLASS): the per-cell class
labels at low-res are stored AS-IS (raw uint8, one byte per cell) AFTER the
GRAYSCALE_STREAM. Inflate re-binds ``cls_full`` from upsample(cls_lowres,
NEAREST) instead of np.zeros_like(gray_full).

Sister of NSCS06 strip_everything CH06 v2 (commit 4292c8ce2 symposium) but
without arith-coding (raw uint8 minimum-LOC wire-in scope per pre-execution
gate report 2026-05-26; arith-coded variant is a downstream bytes-saving
optimization DEFERRED-pending-bytes-budget-analysis).

cls_stream rate-axis cost: ``num_pairs * grayscale_h * grayscale_w`` bytes
(ADDITIVE to canonical equation #26 REPLACEMENT savings). The total stacked
rate-axis ΔS = -0.002706 (canonical equation #26 LUT replacement, unchanged)
+ 25 * (num_pairs * gh * gw) / 37_545_489 (cls_stream ADDITIVE byte cost).
At realistic shapes (num_pairs=600, gh=96, gw=128) this is ~+0.0049 rate-axis
cost which would dominate; the EXPECTATION is that seg+pose-axis IMPROVEMENT
from correct per-class chroma reconstruction more than offsets the cls_stream
rate cost. The full-axis trade-off is the empirical question per the paired
Modal T4 dispatch decision per Catalog #246.
"""

POSE_DIMS: int = 6
"""Per-pair pose vector dimensionality; sister of v7 POSE_DIMS."""

POSE_QUANT_SCALE_DEFAULT: float = 1.0
"""Default pose quantization scale (uint8 around zero=128 over [-127.5/scale, +127.5/scale])."""

GENERATOR_KIND_TAG: dict[str, int] = {"xorshift": 0, "lcg": 1, "pcg64": 2}
"""Sister to grayscale_lut PROCEDURAL_VARIANT `_GENERATOR_KIND_TAG` (canonical)."""

GENERATOR_KIND_TAG_INVERSE: dict[int, str] = {v: k for k, v in GENERATOR_KIND_TAG.items()}

# Header layout per docstring. Format string (v1/v2):
#   < 4s B H H H H H B B H H I I I B B  ->  4+1+2+2+2+2+2+1+1+2+2+4+4+4+1+1 = 35 bytes
CH08_HEADER_FMT: str = "<4sBHHHHHBBHHIIIBB"
CH08_HEADER_SIZE: int = struct.calcsize(CH08_HEADER_FMT)
assert CH08_HEADER_SIZE == 35, (
    f"CH08 header size invariant violated: {CH08_HEADER_SIZE} != 35"
)

# v3 (procedural seed + cls_stream) header layout — appends a CLS_LEN u32 to
# the canonical v1/v2 header so v1/v2 byte stability is PRESERVED (parsers
# dispatch on the version byte at struct position 4).
#
#   v1/v2 35-byte header || CLS_LEN(4) u32
#
# Total: 35 + 4 = 39 bytes.
CH08_HEADER_FMT_V3: str = CH08_HEADER_FMT + "I"
CH08_HEADER_SIZE_V3: int = struct.calcsize(CH08_HEADER_FMT_V3)
assert CH08_HEADER_SIZE_V3 == 39, (
    f"CH08 v3 header size invariant violated: {CH08_HEADER_SIZE_V3} != 39"
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
    """v2/v3: PROCEDURAL_SEED_SIZE_BYTES PCG64 seed. v1: None."""
    generator_kind: str | None
    """v2/v3: 'xorshift' / 'lcg' / 'pcg64'. v1: None."""
    cls_lowres: np.ndarray | None = None
    """v3: per-cell uint8 SegNet class labels at low-res, shape
    ``(num_pairs, grayscale_h, grayscale_w)``. v1/v2: None (inflate falls back
    to cls=0 uniform per L0 SCAFFOLD legacy behavior).

    Per T3 council #1335 PROCEED_WITH_REVISIONS REVISION #2 (Yousfi BLOCKER):
    the cls_stream wire-in unblocks Catalog #233 L1→L2 promotion canonical
    4-gate by transitioning the `distinguishing_feature_smoke` verdict from
    FAIL_AT_CLASS_1 to PASS_PER_CLASS at the cargo-cult #5 inflate site.
    """


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
    cls_bytes: bytes | None = None,
) -> bytes:
    """Serialize CH08 v1 (inline LUT) / v2 (procedural seed) / v3 (procedural
    seed + cls_stream) archive into 0.bin.

    Exactly one of ``chroma_lut`` (v1) or ``chroma_seed`` (v2/v3) MUST be supplied.
    If ``cls_bytes`` is supplied AND ``chroma_seed`` is supplied, schema_version
    is v3; otherwise v1 (if chroma_lut) or v2 (if chroma_seed only).

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
        # v2/v3: pack the seed. v3 is selected when cls_bytes is supplied.
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
        generator_kind_tag = GENERATOR_KIND_TAG[generator_kind]
        if cls_bytes is None:
            schema_version = CH08_SCHEMA_VERSION_PROCEDURAL_SEED
        else:
            schema_version = CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM

    # v3 cls_stream validation (length must match the low-res grayscale shape;
    # one uint8 class label per cell per pair). cls_bytes is only consumed in
    # v3; v1/v2 reject it with an explicit error so caller mistakes surface
    # at compress time rather than silently dropped.
    if cls_bytes is not None and schema_version != CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM:
        raise ValueError(
            "cls_bytes supplied but schema_version resolved to v1/v2 "
            "(only v3 carries CLS_STREAM); supply chroma_seed for v3 dispatch"
        )
    expected_cls = num_pairs * grayscale_h * grayscale_w
    cls_len = 0
    if schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM:
        if cls_bytes is None:
            raise ValueError(
                "v3 archive requires cls_bytes (per-cell uint8 class labels)"
            )
        if len(cls_bytes) != expected_cls:
            raise ValueError(
                f"cls_bytes length {len(cls_bytes)} != expected {expected_cls} "
                f"(num_pairs * grayscale_h * grayscale_w)"
            )
        cls_len = len(cls_bytes)
        if cls_len > 0xFFFFFFFF:
            raise ValueError(f"cls_bytes too large: {cls_len}")

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

    if schema_version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM:
        header = struct.pack(
            CH08_HEADER_FMT_V3,
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
            cls_len,
        )
        return header + lut_payload + pose_bytes + grayscale_bytes + bytes(cls_bytes)
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
    """Parse 0.bin bytes back into a typed Nscs06V8Archive.

    Dispatches on the version byte at struct position 4 (single u8 right after
    the 4-byte magic): v1/v2 use the 35-byte canonical header; v3 uses the
    39-byte extended header with appended CLS_LEN u32 + trailing CLS_STREAM
    section.
    """
    if len(blob) < CH08_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {CH08_HEADER_SIZE})"
        )
    # Peek the version byte (struct position 4: right after b"CH08" magic).
    version_peek = int(blob[4])
    is_v3 = version_peek == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM
    if is_v3:
        if len(blob) < CH08_HEADER_SIZE_V3:
            raise ValueError(
                f"v3 archive too short ({len(blob)} bytes; need >= {CH08_HEADER_SIZE_V3})"
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
            cls_len,
        ) = struct.unpack(CH08_HEADER_FMT_V3, blob[:CH08_HEADER_SIZE_V3])
        header_size_local = CH08_HEADER_SIZE_V3
    else:
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
        cls_len = 0
        header_size_local = CH08_HEADER_SIZE
    if magic != CH08_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {CH08_MAGIC!r})")
    if version not in (
        CH08_SCHEMA_VERSION_INLINE_LUT,
        CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
        CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM,
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

    pos = header_size_local
    lut_payload = blob[pos : pos + lut_payload_len]
    pos += lut_payload_len
    pose_bytes = blob[pos : pos + pose_len]
    pos += pose_len
    grayscale_bytes = blob[pos : pos + grayscale_len]
    pos += grayscale_len
    cls_stream_bytes: bytes | None = None
    if version == CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM:
        expected_cls = num_pairs * grayscale_h * grayscale_w
        if cls_len != expected_cls:
            raise ValueError(
                f"v3 cls_len {cls_len} != expected {expected_cls} "
                f"(num_pairs * grayscale_h * grayscale_w)"
            )
        cls_stream_bytes = blob[pos : pos + cls_len]
        pos += cls_len
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
        # v2 or v3: both carry procedural seed in lut_payload.
        if lut_payload_len != PROCEDURAL_SEED_SIZE_BYTES:
            raise ValueError(
                f"v2/v3 procedural-seed: lut_payload_len {lut_payload_len} != "
                f"PROCEDURAL_SEED_SIZE_BYTES {PROCEDURAL_SEED_SIZE_BYTES}"
            )
        chroma_lut = None
        chroma_seed = bytes(lut_payload)
        if generator_kind_tag not in GENERATOR_KIND_TAG_INVERSE:
            raise ValueError(f"unknown generator_kind_tag {generator_kind_tag}")
        generator_kind = GENERATOR_KIND_TAG_INVERSE[generator_kind_tag]

    cls_lowres: np.ndarray | None = None
    if cls_stream_bytes is not None:
        cls_lowres = (
            np.frombuffer(cls_stream_bytes, dtype=np.uint8)
            .copy()
            .reshape(num_pairs, grayscale_h, grayscale_w)
        )
        if int(cls_lowres.max(initial=0)) >= num_segnet_classes:
            raise ValueError(
                f"cls_stream label {int(cls_lowres.max())} >= num_segnet_classes "
                f"{num_segnet_classes}"
            )

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
        cls_lowres=cls_lowres,
    )
