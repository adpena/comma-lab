# SPDX-License-Identifier: MIT
"""CH06 archive grammar — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields declared in package ``__init__``.
This file IS the export-first grammar (HNeRV parity discipline lesson L2).

**Schema v2** (Path A redesign 2026-05-16; symposium commit 4292c8ce2): adds
per-class CHROMA palette + per-cell CLASS-LABEL stream so the inflate runtime
can synthesize RGB from grayscale+class instead of Y=R=G=B replication.::  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:docstring-references-the-cargo-cult-that-was-already-unwound-by-v7-path-A-per-class-RGB-anchors-and-v8-DB4-DWT-decorrelation-per-symposium-commit-4292c8ce2

    MAGIC(4)             b"CH06"   Carmack-Hotz strip-everything substrate v06
    VERSION(1)           u8        schema version (2 = Path A chroma+optical-flow;
                                    3 = v2 plus seeded chroma palette)
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
    CHROMA_LEN(2)        u16       NUM_SEGNET_CLASSES * 3 bytes  (v2; per-class RGB anchors)
                                    or 8 bytes (v3; archived chroma seed)
    CLS_LEN(4)           u32       arith-coded per-cell class-label stream length (v2)
    PALETTE              ...       PALETTE_SIZE bytes (uint8 grayscale levels)
    CDF                  ...       class-conditional CDF table (uint16 row-major)
    GRAYSCALE_STREAM     ...       arith-coded palette indices for odd-frame grayscale
    POSE_STREAM          ...       per-pair pose deltas (uint8 quantized; 6 dims per pair)
    META_BLOB            ...       json: {grayscale_downsample, pose_quant_scale, ...}
    CHROMA_BLOB          ...       NUM_SEGNET_CLASSES*3 uint8 RGB anchors (v2)
                                    or 8-byte seed expanded at inflate (v3)
    CLS_STREAM           ...       arith-coded uniform-CDF class labels per cell (v2)

Total header v2: 30 + 2 + 4 = 36 bytes.

The grammar is FIXED at design-time. CH06 = "Carmack-Hotz substrate v06"
(the "06" matches the package name suffix ``nscs06_*``).

Compress side writes everything; inflate side reads + decodes purely with
numpy + Pillow. NO torch, NO scorer, NO learned weights.

Path A cargo-cult unwinding (per grand council symposium 2026-05-16 commit
4292c8ce2; Assumption-Adversary VETO satisfied):
- Cargo-cult #1 (closed-form-argmax-allocator) UNWOUND: per-class CDF is now
  ACTUALLY consumed at inflate via the CLS_STREAM; previously the inflate
  used class=0 uniformly making the CDF effectively dead.
- Cargo-cult #2 (Y=R=G=B chroma destruction) UNWOUND: per-class RGB anchors  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:docstring-references-the-cargo-cult-that-was-already-unwound-by-v7-path-A-per-class-RGB-anchors-and-v8-DB4-DWT-decorrelation-per-symposium-commit-4292c8ce2
  provide chroma at inflate; SegNet now sees coloured frames.
- Cargo-cult #4 (2-of-6-pose-warp) UNWOUND in inflate.py via 6-dim affine warp.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON meta, fixed-precision arith state)
- No /tmp paths
- No scorer load
- Reviewable in 30 seconds per L12 (~140 LOC including doc + assertions)
"""

from __future__ import annotations

import hashlib
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

CH06_SCHEMA_VERSION: int = 2
"""Schema version byte. v2 (Path A 2026-05-16): adds chroma palette + class
labels stream so the inflate runtime can synthesize per-pixel RGB instead of
Y=R=G=B replication. Symposium commit 4292c8ce2 ratifies the bump."""  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:docstring-references-the-cargo-cult-that-was-already-unwound-by-v7-path-A-per-class-RGB-anchors-and-v8-DB4-DWT-decorrelation-per-symposium-commit-4292c8ce2

CH06_SCHEMA_VERSION_SEEDED_CHROMA: int = 3
"""Schema v3 stores an 8-byte seed and expands the per-class chroma palette."""

# Header layout per docstring above. v2 format string:
#   < 4s B H B H H H H H H I I H H I  ->  4+1+2+1+2+2+2+2+2+2+4+4+2+2+4 = 36 bytes
CH06_HEADER_FMT: str = "<4sBHBHHHHHHIIHHI"
CH06_HEADER_SIZE: int = struct.calcsize(CH06_HEADER_FMT)
assert CH06_HEADER_SIZE == 36, (
    f"CH06 v2 header size invariant violated: {CH06_HEADER_SIZE} != 36"
)

# Pose deltas are quantized to uint8 over [-pose_quant_range, +pose_quant_range].
# Stored as raw bytes (NO arithmetic coding); only ~3.6 KB total for 600 pairs.
POSE_DIMS: int = 6
"""Per-pair pose vector dimensionality (PoseNet's first 6 dims per CLAUDE.md).
ALL 6 dims are consumed by the inflate runtime's affine warp per Path A
(symposium commit 4292c8ce2; cargo-cult #4 unwound)."""

CHROMA_BYTES_PER_CLASS: int = 3
"""Per-SegNet-class RGB anchor stored as uint8 R/G/B (3 bytes per class)."""

CHROMA_SEED_BYTES: int = 8
"""Archive-charged bytes for procedural per-class chroma palette generation."""

_CHROMA_SEED_DOMAIN = b"tac.procedural_codebook_generator.seed.v1\0"


def emit_chroma_palette_seed() -> bytes:
    """Emit the canonical archive-member seed for CH06 v3 chroma palettes."""
    descriptor = json.dumps(
        {
            "distribution": "uniform_int8",
            "shape": (NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_CHROMA_SEED_DOMAIN + descriptor).digest()[
        :CHROMA_SEED_BYTES
    ]


def expand_chroma_seed_to_palette(seed: bytes) -> np.ndarray:
    """Expand an archive-charged seed into a deterministic uint8 chroma palette."""
    seed_bytes = _validate_chroma_seed(seed)
    rng = np.random.Generator(
        np.random.PCG64(int.from_bytes(seed_bytes, byteorder="big"))
    )
    return rng.integers(
        0,
        256,
        size=(NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS),
        dtype=np.uint8,
    )


def _validate_chroma_seed(seed: bytes | None) -> bytes:
    if seed is None:
        raise ValueError("chroma_seed is required for CH06 seeded-chroma archives")
    seed_bytes = bytes(seed)
    if len(seed_bytes) != CHROMA_SEED_BYTES:
        raise ValueError(f"chroma_seed must be exactly {CHROMA_SEED_BYTES} bytes")
    return seed_bytes


@dataclass(frozen=True)
class CarmackHotzArchive:
    """Parsed CH06 archive — the inflate-time data contract.

    Path A v2 fields (`chroma_rgb`, `cls_arith_bytes`) carry the per-class
    chroma anchors + per-cell class labels so the inflate runtime can render
    per-pixel RGB instead of Y=R=G=B replication.  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:docstring-references-the-cargo-cult-that-was-already-unwound-by-v7-path-A-per-class-RGB-anchors-and-v8-DB4-DWT-decorrelation-per-symposium-commit-4292c8ce2
    """

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
    chroma_rgb: np.ndarray  # (NUM_SEGNET_CLASSES, 3) uint8 per-class RGB anchors (v2)
    chroma_seed: bytes | None
    """Archive-charged seed for v3 procedural chroma palettes."""

    cls_arith_bytes: bytes  # arith-coded per-cell class labels stream (v2)

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
    chroma_rgb: np.ndarray,
    cls_arith_bytes: bytes,
    schema_version: int = CH06_SCHEMA_VERSION,
    chroma_seed: bytes | None = None,
) -> bytes:
    """Serialize CH06 v2 archive into monolithic 0.bin bytes.

    Path A redesign 2026-05-16: `chroma_rgb` (NUM_SEGNET_CLASSES, 3) uint8 +
    `cls_arith_bytes` (arith-coded per-cell class labels) are now REQUIRED.
    If `chroma_seed` is supplied, CH06 v3 stores that seed instead of the raw
    15-byte palette and requires `chroma_rgb` to be exactly seed-derived.
    """
    effective_schema_version = (
        CH06_SCHEMA_VERSION_SEEDED_CHROMA
        if chroma_seed is not None and schema_version == CH06_SCHEMA_VERSION
        else schema_version
    )
    if effective_schema_version not in (
        CH06_SCHEMA_VERSION,
        CH06_SCHEMA_VERSION_SEEDED_CHROMA,
    ):
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
    if chroma_rgb.dtype != np.uint8:
        raise ValueError(f"chroma_rgb must be uint8; got {chroma_rgb.dtype}")
    if chroma_rgb.shape != (NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS):
        raise ValueError(
            f"chroma_rgb must be ({NUM_SEGNET_CLASSES}, {CHROMA_BYTES_PER_CLASS}); "
            f"got {chroma_rgb.shape}"
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
    if effective_schema_version == CH06_SCHEMA_VERSION_SEEDED_CHROMA:
        seed = _validate_chroma_seed(chroma_seed)
        generated_chroma = expand_chroma_seed_to_palette(seed)
        if not np.array_equal(chroma_rgb, generated_chroma):
            raise ValueError(
                "seeded chroma archive requires chroma_rgb to equal "
                "expand_chroma_seed_to_palette(chroma_seed)"
            )
        chroma_bytes = seed
    else:
        chroma_bytes = bytes(chroma_rgb.tobytes())

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
    if len(chroma_bytes) > 0xFFFF:
        raise ValueError(f"chroma too large: {len(chroma_bytes)} > {0xFFFF}")
    if len(cls_arith_bytes) > 0xFFFFFFFF:
        raise ValueError(f"cls stream too large: {len(cls_arith_bytes)}")

    header = struct.pack(
        CH06_HEADER_FMT,
        CH06_MAGIC,
        effective_schema_version,
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
        len(chroma_bytes),
        len(cls_arith_bytes),
    )
    return (
        header
        + palette_bytes
        + cdf_bytes
        + grayscale_arith_bytes
        + pose_bytes
        + meta_bytes
        + chroma_bytes
        + cls_arith_bytes
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
        chroma_len,
        cls_len,
    ) = struct.unpack(CH06_HEADER_FMT, blob[:CH06_HEADER_SIZE])
    if magic != CH06_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {CH06_MAGIC!r})")
    if version not in (CH06_SCHEMA_VERSION, CH06_SCHEMA_VERSION_SEEDED_CHROMA):
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
    expected_chroma = (
        NUM_SEGNET_CLASSES * CHROMA_BYTES_PER_CLASS
        if version == CH06_SCHEMA_VERSION
        else CHROMA_SEED_BYTES
    )
    if chroma_len != expected_chroma:
        raise ValueError(f"chroma_len {chroma_len} != expected {expected_chroma}")

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
    chroma_blob = blob[pos : pos + chroma_len]
    pos += chroma_len
    cls_arith_bytes = blob[pos : pos + cls_len]
    pos += cls_len
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
    if version == CH06_SCHEMA_VERSION_SEEDED_CHROMA:
        chroma_seed = bytes(chroma_blob)
        chroma_rgb = expand_chroma_seed_to_palette(chroma_seed)
    else:
        chroma_seed = None
        chroma_rgb = (
            np.frombuffer(chroma_blob, dtype=np.uint8)
            .copy()
            .reshape(NUM_SEGNET_CLASSES, CHROMA_BYTES_PER_CLASS)
        )
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
        chroma_rgb=chroma_rgb,
        chroma_seed=chroma_seed,
        cls_arith_bytes=bytes(cls_arith_bytes),
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


def _uniform_class_cdf() -> np.ndarray:
    """Uniform CDF over NUM_SEGNET_CLASSES symbols (uint16, in [0, CDF_MAX])."""
    edges = np.linspace(0, CDF_MAX, NUM_SEGNET_CLASSES + 1, dtype=np.int64)
    edges[-1] = CDF_MAX
    return edges.astype(np.uint16)


def encode_class_label_stream(class_labels: np.ndarray) -> bytes:
    """Arith-encode per-cell SegNet class labels with a uniform CDF (Path A).

    The per-cell class labels are decoded BEFORE the grayscale stream at
    inflate time so the per-class CDF can be consumed. Uniform CDF is the
    minimum-assumption coder; future passes can swap in a temporal/spatial
    predictor (Schmidhuber predictive coding) for tighter compression.
    """
    if class_labels.dtype != np.uint8:
        raise ValueError("class_labels must be uint8")
    if int(class_labels.max(initial=0)) >= NUM_SEGNET_CLASSES:
        raise ValueError(
            f"class label {int(class_labels.max())} >= NUM_SEGNET_CLASSES"
        )
    coder = ArithmeticCoder()
    row = _uniform_class_cdf()
    for sym in class_labels.ravel():
        coder.encode_symbol(int(sym), row)
    return coder.finish_encoding()


def decode_class_label_stream(
    arith_bytes: bytes, *, shape: tuple[int, ...]
) -> np.ndarray:
    """Inverse of :func:`encode_class_label_stream`; returns shape uint8 cells."""
    coder = ArithmeticCoder.from_bytes(arith_bytes)
    row = _uniform_class_cdf()
    n = 1
    for s in shape:
        n *= int(s)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out[i] = np.uint8(coder.decode_symbol(row))
    return out.reshape(shape)


def build_chroma_palette(
    rgb_pairs: np.ndarray, class_labels: np.ndarray
) -> np.ndarray:
    """Derive per-class (R, G, B) anchor from compress-time GT video.

    For each SegNet class c in [0, NUM_SEGNET_CLASSES), compute the median
    (R, G, B) across every pixel labelled class c. Median is robust to
    outliers in noisy class regions. Returns (NUM_SEGNET_CLASSES, 3) uint8.

    Args:
        rgb_pairs: (N, 3, H, W) uint8 RGB frames (compress-time GT).
        class_labels: (N, H, W) uint8 SegNet argmax labels (same N, H, W).
    """
    if rgb_pairs.dtype != np.uint8:
        raise ValueError("rgb_pairs must be uint8")
    if class_labels.dtype != np.uint8:
        raise ValueError("class_labels must be uint8")
    if rgb_pairs.shape[1] != 3:
        raise ValueError(f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}")
    n, _, h, w = rgb_pairs.shape
    if class_labels.shape != (n, h, w):
        raise ValueError(
            f"class_labels shape {class_labels.shape} != ({n}, {h}, {w})"
        )
    # Reshape RGB to (3, N*H*W) for masked median per channel.
    rgb_flat = rgb_pairs.transpose(1, 0, 2, 3).reshape(3, -1)
    cls_flat = class_labels.reshape(-1)
    out = np.zeros((NUM_SEGNET_CLASSES, 3), dtype=np.uint8)
    for c in range(NUM_SEGNET_CLASSES):
        mask = cls_flat == c
        if mask.any():
            for ch in range(3):
                out[c, ch] = np.uint8(np.median(rgb_flat[ch][mask]))
        else:
            # Mid-grey fallback for absent classes — anchor still in range.
            out[c, :] = 128
    return out


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
