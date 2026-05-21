# SPDX-License-Identifier: MIT
"""grayscale_lut archive grammar GLV1 — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (HNeRV parity discipline
lesson L2):

::

    MAGIC(4)             b"GLV1"   GrayscaLe-lut Variant 1
    VERSION(1)           u8        schema version (currently 1)
    NUM_PAIRS(2)         u16       cfg.num_pairs (e.g. 600)
    GRAYSCALE_H(2)       u16       grayscale field height (= H/grayscale_downsample)
    GRAYSCALE_W(2)       u16       grayscale field width (= W/grayscale_downsample)
    GRAYSCALE_DOWNSAMPLE(1) u8     grayscale_downsample (e.g. 4)
    EMBEDDING_DIM(2)     u16       embedding_dim
    OUTPUT_HEIGHT(2)     u16       output_height
    OUTPUT_WIDTH(2)      u16       output_width
    DECODER_BLOB_LEN(4)  u32       brotli(state_dict bytes) of stem + blocks + heads + pair_embedding
    GRAYSCALE_BLOB_LEN(4) u32      brotli(uint8 grayscale stream) bytes len
    META_BLOB_LEN(4)     u32       utf-8 json meta bytes len
    DECODER_BLOB         ...       brotli(pickled state_dict, fp16 cpu)
    GRAYSCALE_BLOB       ...       brotli(uint8 grayscale row-major: num_pairs * H/D * W/D bytes)
    META_BLOB            ...       json: {decoder_hidden, decoder_blocks, ...}

GLV1 = "GrayscaLe-lut Variant 1". The grayscale stream is the dominant rate
term; storing it as uint8 + brotli exploits the high spatial+temporal
redundancy of natural-video grayscale fields.

The grammar is fixed at design-time; mutating it changes the schema VERSION
and requires a new inflate.py.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 CPU state_dict, fixed brotli quality)
- No /tmp paths
- No scorer load
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

GLV1_MAGIC: bytes = b"GLV1"
"""grayscale_lut variant 1 archive magic."""

GLV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + GRAYSCALE_H(2) + GRAYSCALE_W(2)
#                + GRAYSCALE_DOWNSAMPLE(1) + EMBEDDING_DIM(2) + OUTPUT_H(2) + OUTPUT_W(2)
#                + 3 u32 (12) = 30 bytes
GLV1_HEADER_FMT: str = "<4sBHHHBHHHIII"
GLV1_HEADER_SIZE: int = struct.calcsize(GLV1_HEADER_FMT)
assert GLV1_HEADER_SIZE == 30, "GLV1 header size invariant (4+1+2+2+2+1+2+2+2+12 = 30)"

# Brotli quality
BROTLI_QUALITY: int = 9

_GLV1_REQUIRED_RUNTIME_PREFIXES: tuple[str, ...] = (
    "pair_embedding",
    "stem.",
    "blocks.",
    "head_rgb_0.",
    "head_rgb_1.",
)


@dataclass(frozen=True)
class GrayscaleLutArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Stem + FiLM blocks + heads + pair_embedding state_dict."""

    grayscale: torch.Tensor
    """``(num_pairs, 1, H/D, W/D)`` uint8 quantized analog grayscale stream."""

    meta: dict[str, object]
    """Sidecar JSON meta with arch hparams."""

    schema_version: int
    num_pairs: int
    grayscale_h: int
    grayscale_w: int
    grayscale_downsample: int
    embedding_dim: int
    output_height: int
    output_width: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu)."""
    _validate_runtime_state_dict(sd)
    buf = io.BytesIO()
    sd_cpu = {k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()}
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _validate_runtime_state_dict(sd: dict[str, torch.Tensor]) -> None:
    """Refuse grayscale bytes in the decoder state_dict section."""

    if "grayscale" in sd:
        raise ValueError(
            "GLV1 archive state_dict contains grayscale, but grayscale has its "
            "own uint8 section. Use "
            "GrayscaleLutSubstrate.runtime_state_dict_for_archive()."
        )
    missing_prefixes = [
        prefix
        for prefix in _GLV1_REQUIRED_RUNTIME_PREFIXES
        if not any(key == prefix.rstrip(".") or key.startswith(prefix) for key in sd)
    ]
    if missing_prefixes:
        raise ValueError(f"GLV1 runtime state_dict missing prefixes: {missing_prefixes}")


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _pack_grayscale(gs: torch.Tensor) -> bytes:
    """Pack uint8 grayscale stream with brotli for max compression.

    Natural-video grayscale fields have high spatial+temporal redundancy;
    brotli exploits both. NO additional quantization here — caller is
    responsible for producing uint8 from float via
    ``GrayscaleLutSubstrate.quantize_grayscale_for_archive``.
    """
    if gs.dtype != torch.uint8:
        raise ValueError(f"grayscale must be uint8; got {gs.dtype}")
    raw = gs.contiguous().cpu().numpy().tobytes()
    return bytes(brotli.compress(raw, quality=BROTLI_QUALITY))


def _unpack_grayscale(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    import numpy as np  # local

    raw = brotli.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(arr).view(*shape)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    grayscale: torch.Tensor,
    meta: dict[str, object],
    *,
    num_pairs: int,
    grayscale_downsample: int,
    embedding_dim: int,
    output_height: int,
    output_width: int,
    schema_version: int = GLV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained grayscale_lut state into the monolithic 0.bin bytes."""
    if schema_version != GLV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if grayscale.dim() != 4:
        raise ValueError(
            f"grayscale must be 4-D (num_pairs, 1, H, W); got {tuple(grayscale.shape)}"
        )
    if grayscale.shape[1] != 1:
        raise ValueError(f"grayscale.shape[1] must be 1; got {grayscale.shape[1]}")
    if grayscale.shape[0] != num_pairs:
        raise ValueError(
            f"grayscale.shape[0]={grayscale.shape[0]} != num_pairs={num_pairs}"
        )

    grayscale_h = int(grayscale.shape[2])
    grayscale_w = int(grayscale.shape[3])

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("grayscale_h", grayscale_h, 0xFFFF),
        ("grayscale_w", grayscale_w, 0xFFFF),
        ("grayscale_downsample", grayscale_downsample, 0xFF),
        ("embedding_dim", embedding_dim, 0xFFFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of u8/u16 range (max {max_v})")

    decoder_blob = _serialize_state_dict(decoder_state_dict)
    grayscale_blob = _pack_grayscale(grayscale)
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        GLV1_HEADER_FMT,
        GLV1_MAGIC,
        schema_version,
        num_pairs,
        grayscale_h,
        grayscale_w,
        grayscale_downsample,
        embedding_dim,
        output_height,
        output_width,
        len(decoder_blob),
        len(grayscale_blob),
        len(meta_bytes),
    )
    return header + decoder_blob + grayscale_blob + meta_bytes


def parse_archive(blob: bytes) -> GrayscaleLutArchive:
    """Parse 0.bin bytes back into typed GrayscaleLutArchive."""
    if len(blob) < GLV1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes; need >= {GLV1_HEADER_SIZE})")
    (
        magic,
        version,
        num_pairs,
        grayscale_h,
        grayscale_w,
        grayscale_downsample,
        embedding_dim,
        output_height,
        output_width,
        decoder_len,
        grayscale_len,
        meta_len,
    ) = struct.unpack(GLV1_HEADER_FMT, blob[:GLV1_HEADER_SIZE])
    if magic != GLV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {GLV1_MAGIC!r})")
    if version != GLV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    pos = GLV1_HEADER_SIZE
    decoder_blob = blob[pos : pos + decoder_len]
    pos += decoder_len
    grayscale_blob = blob[pos : pos + grayscale_len]
    pos += grayscale_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    sd = _deserialize_state_dict(decoder_blob)
    grayscale = _unpack_grayscale(grayscale_blob, (num_pairs, 1, grayscale_h, grayscale_w))
    meta = json.loads(meta_blob.decode("utf-8"))

    return GrayscaleLutArchive(
        decoder_state_dict=sd,
        grayscale=grayscale,
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        grayscale_h=int(grayscale_h),
        grayscale_w=int(grayscale_w),
        grayscale_downsample=int(grayscale_downsample),
        embedding_dim=int(embedding_dim),
        output_height=int(output_height),
        output_width=int(output_width),
    )


def compose_procedural_archive(
    original_archive_bytes: bytes,
    seed_bytes: bytes,
) -> bytes:
    """Thin convenience wrapper for grayscale_lut procedural-LUT archive composition.

    Per WAVE-3-GRAYSCALE-LUT-PROCEDURAL-TRAINER-BUILD 2026-05-20 + sister
    DP1 canonical pattern landing commit ``9cbfa471c`` + sister VQ-VAE
    canonical pattern landing commit ``6fea30f22``: delegates to
    :func:`tac.substrates.grayscale_lut.distillation_procedural_variant.compose_with_procedural_lut`
    using canonical defaults (32-byte seed, PCG64, 256-byte chroma LUT
    target matching the canonical ``chroma_lut_replacement`` context per
    canonical equation #26 ``_INCLUDED_CONTEXTS``).

    Sister of :func:`pack_archive` (canonical builder for the trained
    grayscale_lut variant). The procedural variant appends a sentinel-
    prefixed seed envelope to the existing GLV1 archive; a future
    LUT-aware grayscale_lut variant will REPLACE in-archive chroma LUT
    bytes with the envelope (matching the canonical DP1 + VQ-VAE
    REPLACE-IN-PLACE pattern).

    Args:
        original_archive_bytes: Existing GLV1 archive bytes (parseable
            via :func:`parse_archive`).
        seed_bytes: Procedural seed (8-256 bytes; canonical 32 bytes).

    Returns:
        Procedural-variant archive bytes with the canonical GLV1
        sections preserved + the procedural seed envelope appended.
    """
    # Lazy import to avoid cyclic-import friction; the variant module
    # imports GLV1_HEADER_FMT / GLV1_HEADER_SIZE / GLV1_MAGIC /
    # GLV1_SCHEMA_VERSION / parse_archive from this module.
    from tac.substrates.grayscale_lut.distillation_procedural_variant import (
        compose_with_procedural_lut,
    )

    return compose_with_procedural_lut(
        original_archive_bytes=original_archive_bytes,
        seed_bytes=seed_bytes,
    )
