# SPDX-License-Identifier: MIT
"""SABOR archive grammar SBO1 — monolithic single-file 0.bin.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (HNeRV parity discipline
lesson L2):

::

    MAGIC(4)               b"SBO1"   Stable-argmax Boundary-Only renderer V1
    VERSION(1)             u8        schema version (currently 1)
    NUM_PAIRS(2)           u16       cfg.num_pairs (e.g. 600)
    OUTPUT_HEIGHT(2)       u16       output_height (e.g. 384)
    OUTPUT_WIDTH(2)        u16       output_width (e.g. 512)
    NUM_SEG_CLASSES(1)     u8        SegNet class count (5)
    REFINEMENT_HIDDEN(2)   u16       FiLM block hidden channels
    REFINEMENT_BLOCKS(1)   u8        Number of FiLM blocks
    EMBEDDING_DIM(2)       u16       FiLM embedding dim
    BIAS_DIM(1)            u8        per-frame bias channels
    EDGE_THRESHOLD_Q(2)    u16       edge_threshold * 65535 (quantized; for replay)
    BOUNDARY_PIXEL_COUNT(4) u32      number of True bits across all frames (audit)
    DECODER_BLOB_LEN(4)    u32       brotli(fp16 decoder state_dict) bytes len
    CLASS_MEANS_BLOB_LEN(4) u32      brotli(uint8 class_means table) bytes len
    BOUNDARY_MASK_BLOB_LEN(4) u32    brotli(packbits boundary mask) bytes len
    BOUNDARY_RGB_BLOB_LEN(4) u32     brotli(int8 boundary RGB values) bytes len
    SEG_ARGMAX_BLOB_LEN(4) u32       brotli(packed SegNet argmax indices) bytes len
    META_BLOB_LEN(4)       u32       utf-8 json meta bytes len
    DECODER_BLOB           ...       brotli(pickle of stem+blocks+head+pair_embedding+pair_bias fp16)
    CLASS_MEANS_BLOB       ...       brotli(uint8 class_means flat: num_seg_classes * 3 bytes)
    BOUNDARY_MASK_BLOB     ...       brotli(np.packbits over (num_frames, H, W) bool)
    BOUNDARY_RGB_BLOB      ...       brotli(uint8 RGB values for boundary pixels in
                                     row-major order across frames; length =
                                     boundary_pixel_count * 3 bytes)
    SEG_ARGMAX_BLOB        ...       brotli(uint8 segnet_argmax over (num_frames, H, W))
                                     — at most 8 classes so uint8 suffices.
    META_BLOB              ...       json: {edge_threshold, num_pairs, ...}

SBO1 = "Stable-argmax Boundary-Only renderer V1". The boundary RGB blob
contains the high-fidelity int8 RGB at every boundary pixel (packed row-major
across frames). At inflate time the decoder reconstructs:

* The boundary mask via packbits unpack.
* The boundary RGB by indexing the flat blob via the cumulative sum of
  boundary-True bits per frame.

The grammar is fixed at design-time; mutating it changes the schema VERSION
and requires a new inflate.py.

CLAUDE.md compliance:

* Deterministic (sorted-keys JSON, fp16 CPU state_dict, fixed brotli quality).
* No /tmp paths.
* No scorer load.
* `forbidden_score_claim_with_byte_change_unless_inflate_consumes`: this is
  a representation substrate; inflate.py consumes ALL bytes. The
  byte-mutation no-op detector (Catalog #139) lands at trainer-emission time.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

SBO1_MAGIC: bytes = b"SBO1"
"""SABOR archive magic."""

SBO1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout (48 bytes total):
#   MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + OUT_H(2) + OUT_W(2)
#   + NUM_SEG_CLASSES(1) + REFINEMENT_HIDDEN(2) + REFINEMENT_BLOCKS(1)
#   + EMBEDDING_DIM(2) + BIAS_DIM(1) + EDGE_THRESHOLD_Q(2)
#   + BOUNDARY_PIXEL_COUNT(4) + 6 u32 lens (24)
# = 4 + 1 + 12 + 4 + 4 + 24 = 48 bytes (with 4 u8s and 6 u16s in the
#   middle-cluster).
SBO1_HEADER_FMT: str = "<4sBHHHBHBHBHIIIIIII"
SBO1_HEADER_SIZE: int = struct.calcsize(SBO1_HEADER_FMT)
assert SBO1_HEADER_SIZE == 48, (
    f"SBO1 header size invariant (got {SBO1_HEADER_SIZE}; expected 48)"
)

BROTLI_QUALITY: int = 9

_SBO1_REQUIRED_RUNTIME_PREFIXES: tuple[str, ...] = (
    "pair_embedding",
    "pair_bias",
    "stem.",
    "blocks.",
    "head_rgb.",
)


@dataclass(frozen=True)
class SaborArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Refinement decoder + pair_embedding + pair_bias state_dict."""

    class_means: torch.Tensor
    """``(num_seg_classes, 3)`` uint8 class-mean RGB table."""

    boundary_mask: torch.Tensor
    """``(num_frames, H, W)`` boolean boundary mask."""

    boundary_rgb_flat: torch.Tensor
    """``(boundary_pixel_count, 3)`` uint8 RGB values at boundary pixels,
    row-major across frames. Indexed at inflate time by the cumulative sum
    of True bits per frame."""

    segnet_argmax: torch.Tensor
    """``(num_frames, H, W)`` uint8 SegNet argmax class indices for inflate-time
    texture-fill. Captured at GT time (the scorer runs once during archive
    emission; inflate-time decoder does NOT re-run the scorer per CLAUDE.md
    strict-scorer-rule)."""

    meta: dict[str, object]
    """Sidecar JSON meta with arch hparams and audit counters."""

    schema_version: int
    num_pairs: int
    output_height: int
    output_width: int
    num_seg_classes: int
    refinement_hidden: int
    refinement_blocks: int
    embedding_dim: int
    bias_dim: int
    edge_threshold: float
    boundary_pixel_count: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu)."""
    _validate_runtime_state_dict(sd)
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _validate_runtime_state_dict(sd: dict[str, torch.Tensor]) -> None:
    """Refuse class_means bytes in the decoder state_dict section."""
    if "class_means" in sd:
        raise ValueError(
            "SBO1 archive state_dict contains class_means, but class_means has "
            "its own uint8 section. Use "
            "SaborBoundaryOnlyRenderer.runtime_state_dict_for_archive()."
        )
    missing_prefixes = [
        prefix
        for prefix in _SBO1_REQUIRED_RUNTIME_PREFIXES
        if not any(
            key == prefix.rstrip(".") or key.startswith(prefix) for key in sd
        )
    ]
    if missing_prefixes:
        raise ValueError(
            f"SBO1 runtime state_dict missing prefixes: {missing_prefixes}"
        )


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _pack_class_means(class_means: torch.Tensor) -> bytes:
    if class_means.dtype != torch.uint8:
        raise ValueError(f"class_means must be uint8; got {class_means.dtype}")
    if class_means.dim() != 2 or class_means.shape[1] != 3:
        raise ValueError(
            f"class_means must be (num_seg_classes, 3); got {tuple(class_means.shape)}"
        )
    raw = class_means.contiguous().cpu().numpy().tobytes()
    return bytes(brotli.compress(raw, quality=BROTLI_QUALITY))


def _unpack_class_means(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    import numpy as np

    raw = brotli.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(*shape)


def _pack_boundary_mask(mask: torch.Tensor) -> bytes:
    """Pack boundary mask via np.packbits then brotli."""
    import numpy as np

    if mask.dtype != torch.bool:
        raise ValueError(f"boundary_mask must be bool; got {mask.dtype}")
    arr = mask.contiguous().cpu().numpy().astype(np.bool_)
    packed = np.packbits(arr.flatten(), bitorder="little")
    return bytes(brotli.compress(packed.tobytes(), quality=BROTLI_QUALITY))


def _unpack_boundary_mask(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    import numpy as np

    raw = brotli.decompress(blob)
    packed = np.frombuffer(raw, dtype=np.uint8).copy()
    n_bits = 1
    for dim in shape:
        n_bits *= int(dim)
    flat = np.unpackbits(packed, bitorder="little")[:n_bits].astype(np.bool_)
    return torch.from_numpy(flat.reshape(*shape).copy())


def _pack_boundary_rgb(boundary_rgb_flat: torch.Tensor) -> bytes:
    if boundary_rgb_flat.dtype != torch.uint8:
        raise ValueError(
            f"boundary_rgb_flat must be uint8; got {boundary_rgb_flat.dtype}"
        )
    if boundary_rgb_flat.dim() != 2 or boundary_rgb_flat.shape[1] != 3:
        raise ValueError(
            f"boundary_rgb_flat must be (N, 3); got {tuple(boundary_rgb_flat.shape)}"
        )
    raw = boundary_rgb_flat.contiguous().cpu().numpy().tobytes()
    return bytes(brotli.compress(raw, quality=BROTLI_QUALITY))


def _unpack_boundary_rgb(blob: bytes, n_pixels: int) -> torch.Tensor:
    import numpy as np

    raw = brotli.decompress(blob)
    expected = n_pixels * 3
    if len(raw) != expected:
        raise ValueError(
            f"boundary_rgb byte length mismatch: got {len(raw)} expected {expected}"
        )
    arr = np.frombuffer(raw, dtype=np.uint8).copy().reshape(n_pixels, 3)
    return torch.from_numpy(arr)


def _pack_segnet_argmax(seg: torch.Tensor) -> bytes:
    if seg.dtype not in (torch.uint8, torch.int8, torch.int32, torch.int64):
        raise ValueError(f"segnet_argmax must be integer; got {seg.dtype}")
    if seg.dim() != 3:
        raise ValueError(
            f"segnet_argmax must be (num_frames, H, W); got {tuple(seg.shape)}"
        )
    if int(seg.max().item()) > 255 or int(seg.min().item()) < 0:
        raise ValueError("segnet_argmax must fit in uint8 (class index 0..255)")
    raw = seg.to(torch.uint8).contiguous().cpu().numpy().tobytes()
    return bytes(brotli.compress(raw, quality=BROTLI_QUALITY))


def _unpack_segnet_argmax(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    import numpy as np

    raw = brotli.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.uint8).copy().reshape(*shape)
    return torch.from_numpy(arr)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    class_means: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_rgb_flat: torch.Tensor,
    segnet_argmax: torch.Tensor,
    meta: dict[str, object],
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
    num_seg_classes: int,
    refinement_hidden: int,
    refinement_blocks: int,
    embedding_dim: int,
    bias_dim: int,
    edge_threshold: float,
    schema_version: int = SBO1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained SABOR state into the monolithic 0.bin bytes."""
    if schema_version != SBO1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if not (0.0 < edge_threshold < 1.0):
        raise ValueError("edge_threshold must be in (0, 1)")
    edge_threshold_q = round(edge_threshold * 65535.0)
    edge_threshold_q = max(1, min(65535, edge_threshold_q))

    expected_frames = num_pairs * 2
    if tuple(boundary_mask.shape) != (expected_frames, output_height, output_width):
        raise ValueError(
            f"boundary_mask shape {tuple(boundary_mask.shape)} != "
            f"({expected_frames}, {output_height}, {output_width})"
        )
    if tuple(segnet_argmax.shape) != (expected_frames, output_height, output_width):
        raise ValueError(
            f"segnet_argmax shape {tuple(segnet_argmax.shape)} != "
            f"({expected_frames}, {output_height}, {output_width})"
        )
    boundary_pixel_count = int(boundary_mask.sum().item())
    if int(boundary_rgb_flat.shape[0]) != boundary_pixel_count:
        raise ValueError(
            f"boundary_rgb_flat rows {int(boundary_rgb_flat.shape[0])} != "
            f"boundary mask count {boundary_pixel_count}"
        )

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
        ("num_seg_classes", num_seg_classes, 0xFF),
        ("refinement_hidden", refinement_hidden, 0xFFFF),
        ("refinement_blocks", refinement_blocks, 0xFF),
        ("embedding_dim", embedding_dim, 0xFFFF),
        ("bias_dim", bias_dim, 0xFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of u8/u16 range (max {max_v})")

    decoder_blob = _serialize_state_dict(decoder_state_dict)
    class_means_blob = _pack_class_means(class_means)
    boundary_mask_blob = _pack_boundary_mask(boundary_mask)
    boundary_rgb_blob = _pack_boundary_rgb(boundary_rgb_flat)
    segnet_argmax_blob = _pack_segnet_argmax(segnet_argmax)
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )

    header = struct.pack(
        SBO1_HEADER_FMT,
        SBO1_MAGIC,
        schema_version,
        num_pairs,
        output_height,
        output_width,
        num_seg_classes,
        refinement_hidden,
        refinement_blocks,
        embedding_dim,
        bias_dim,
        edge_threshold_q,
        boundary_pixel_count,
        len(decoder_blob),
        len(class_means_blob),
        len(boundary_mask_blob),
        len(boundary_rgb_blob),
        len(segnet_argmax_blob),
        len(meta_bytes),
    )
    return (
        header
        + decoder_blob
        + class_means_blob
        + boundary_mask_blob
        + boundary_rgb_blob
        + segnet_argmax_blob
        + meta_bytes
    )


def parse_archive(blob: bytes) -> SaborArchive:
    """Parse 0.bin bytes back into a typed SaborArchive."""
    if len(blob) < SBO1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {SBO1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        output_height,
        output_width,
        num_seg_classes,
        refinement_hidden,
        refinement_blocks,
        embedding_dim,
        bias_dim,
        edge_threshold_q,
        boundary_pixel_count,
        decoder_len,
        class_means_len,
        boundary_mask_len,
        boundary_rgb_len,
        segnet_argmax_len,
        meta_len,
    ) = struct.unpack(SBO1_HEADER_FMT, blob[:SBO1_HEADER_SIZE])
    if magic != SBO1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {SBO1_MAGIC!r})")
    if version != SBO1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    edge_threshold = float(edge_threshold_q) / 65535.0

    pos = SBO1_HEADER_SIZE
    decoder_blob = blob[pos : pos + decoder_len]
    pos += decoder_len
    class_means_blob = blob[pos : pos + class_means_len]
    pos += class_means_len
    boundary_mask_blob = blob[pos : pos + boundary_mask_len]
    pos += boundary_mask_len
    boundary_rgb_blob = blob[pos : pos + boundary_rgb_len]
    pos += boundary_rgb_len
    segnet_argmax_blob = blob[pos : pos + segnet_argmax_len]
    pos += segnet_argmax_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    sd = _deserialize_state_dict(decoder_blob)
    class_means = _unpack_class_means(class_means_blob, (num_seg_classes, 3))
    expected_frames = int(num_pairs) * 2
    boundary_mask = _unpack_boundary_mask(
        boundary_mask_blob, (expected_frames, output_height, output_width)
    )
    actual_count = int(boundary_mask.sum().item())
    if actual_count != int(boundary_pixel_count):
        raise ValueError(
            f"boundary_mask True count {actual_count} != header {boundary_pixel_count}"
        )
    boundary_rgb_flat = _unpack_boundary_rgb(
        boundary_rgb_blob, int(boundary_pixel_count)
    )
    segnet_argmax = _unpack_segnet_argmax(
        segnet_argmax_blob, (expected_frames, output_height, output_width)
    )
    meta = json.loads(meta_blob.decode("utf-8"))

    return SaborArchive(
        decoder_state_dict=sd,
        class_means=class_means,
        boundary_mask=boundary_mask,
        boundary_rgb_flat=boundary_rgb_flat,
        segnet_argmax=segnet_argmax,
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        output_height=int(output_height),
        output_width=int(output_width),
        num_seg_classes=int(num_seg_classes),
        refinement_hidden=int(refinement_hidden),
        refinement_blocks=int(refinement_blocks),
        embedding_dim=int(embedding_dim),
        bias_dim=int(bias_dim),
        edge_threshold=edge_threshold,
        boundary_pixel_count=int(boundary_pixel_count),
    )


__all__ = [
    "BROTLI_QUALITY",
    "SBO1_HEADER_FMT",
    "SBO1_HEADER_SIZE",
    "SBO1_MAGIC",
    "SBO1_SCHEMA_VERSION",
    "SaborArchive",
    "pack_archive",
    "parse_archive",
]
