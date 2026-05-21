# SPDX-License-Identifier: MIT
"""vq_vae archive grammar VQV1 — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (HNeRV parity discipline
lesson L2):

::

    MAGIC(4)             b"VQV1"   VQ-VAE Variant 1
    VERSION(1)           u8        schema version (currently 1)
    CODEBOOK_SIZE(2)     u16       K (number of codebook entries; e.g. 512)
    EMBEDDING_DIM(2)     u16       D (per-entry dim; e.g. 8)
    NUM_PAIRS(2)         u16       cfg.num_pairs (e.g. 600)
    H_GRID(2)            u16       index grid height (= H/grid_downsample)
    W_GRID(2)            u16       index grid width (= W/grid_downsample)
    DECODER_BLOB_LEN(4)  u32       brotli(state_dict bytes) of decoder + codebook
    INDICES_BLOB_LEN(4)  u32       packed int16 codebook indices bytes len
    META_BLOB_LEN(4)     u32       utf-8 json meta bytes len
    DECODER_BLOB         ...       brotli(pickled runtime state_dict, fp16 cpu)
    INDICES_BLOB         ...       int16 indices row-major (num_pairs, 2, H_GRID, W_GRID)
    META_BLOB            ...       json: {encoder_hidden, decoder_hidden, grid_downsample, ...}

VQV1 = "VQ-VAE Variant 1". Codebook indices are stored as int16 (signed
2's-complement) because K=512 fits in 9 bits but we use 16-bit aligned storage
for trivial parser correctness. A K=65536-and-beyond variant would need a u32
storage type and a new schema VERSION.

The runtime state_dict must contain only inflate-time tensors: ``codebook`` and
``decoder.*``. Encoder weights and per-pair feature grids are training-only once
the index grid has been exported; packing them would pay bytes that the runtime
never consumes.

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
import numpy as np
import torch

VQV1_MAGIC: bytes = b"VQV1"
"""vq_vae variant 1 archive magic."""

VQV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

VQV1_PROCEDURAL_DECODER_SENTINEL: bytes = b"VQVP"
"""Decoder-section sentinel for procedural-codebook VQV1 archives."""

# Header layout: MAGIC(4) + VERSION(1) + 5 u16 (10) + 3 u32 (12) = 27 bytes
VQV1_HEADER_FMT: str = "<4sBHHHHHIII"
VQV1_HEADER_SIZE: int = struct.calcsize(VQV1_HEADER_FMT)
assert VQV1_HEADER_SIZE == 27, "VQV1 header size invariant (4+1+10+12 = 27)"

# Brotli quality
BROTLI_QUALITY: int = 9

_VQV1_REQUIRED_RUNTIME_KEYS: tuple[str, ...] = ("codebook",)
_VQV1_ALLOWED_RUNTIME_PREFIXES: tuple[str, ...] = ("decoder.",)
_VQV1_FORBIDDEN_TRAINING_PREFIXES: tuple[str, ...] = (
    "per_pair_features",
    "encoder_refine.",
)


@dataclass(frozen=True)
class VqVaeArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Inflate-time state_dict: codebook plus decoder tensors."""

    indices: torch.Tensor
    """``(num_pairs, 2, H_GRID, W_GRID)`` int64 codebook indices."""

    meta: dict[str, object]
    """Sidecar JSON meta with arch hparams."""

    schema_version: int
    codebook_size: int
    embedding_dim: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu)."""
    _validate_runtime_state_dict(sd)
    buf = io.BytesIO()
    sd_cpu = {k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()}
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _is_runtime_key(key: str) -> bool:
    return key in _VQV1_REQUIRED_RUNTIME_KEYS or key.startswith(
        _VQV1_ALLOWED_RUNTIME_PREFIXES
    )


def _validate_runtime_state_dict(sd: dict[str, torch.Tensor]) -> None:
    """Refuse training-only tensors in the VQV1 archive payload."""

    missing = [key for key in _VQV1_REQUIRED_RUNTIME_KEYS if key not in sd]
    if missing:
        raise ValueError(f"VQV1 runtime state_dict missing required keys: {missing}")
    forbidden = [
        key
        for key in sd
        if key.startswith(_VQV1_FORBIDDEN_TRAINING_PREFIXES) or not _is_runtime_key(key)
    ]
    if forbidden:
        sample = ", ".join(sorted(forbidden)[:5])
        raise ValueError(
            "VQV1 archive state_dict contains training-only or unknown keys: "
            f"{sample}. Use VqVaeSubstrate.runtime_state_dict_for_archive()."
        )


def _derive_procedural_codebook_tensor(
    *,
    seed_bytes: bytes,
    codebook_size: int,
    embedding_dim: int,
    generator_kind: str,
) -> torch.Tensor:
    """Derive a bounded fp16 codebook tensor from an in-archive seed."""

    from tac.procedural_codebook_generator import derive_codebook_from_seed

    raw = derive_codebook_from_seed(
        seed_bytes=seed_bytes,
        output_shape=(codebook_size, embedding_dim),
        dtype=np.uint16,
        generator_kind=generator_kind,
    )
    # Map deterministic uint16 values to the same small initialization scale
    # used by VqVaeSubstrate.__init__. This avoids arbitrary fp16 bit-patterns
    # such as NaN/Inf while preserving a byte-mutation-traceable codebook.
    scale = max(float(codebook_size), 1.0)
    values = ((raw.astype(np.float32) / 65535.0) * 2.0 - 1.0) / scale
    return torch.from_numpy(values.astype(np.float16, copy=True))


def _deserialize_procedural_state_dict(
    blob: bytes,
    *,
    codebook_size: int,
    embedding_dim: int,
) -> dict[str, torch.Tensor]:
    """Deserialize a VQVP procedural decoder section and inject codebook."""

    if len(blob) < 13:
        raise ValueError("procedural decoder section too short")

    pos = len(VQV1_PROCEDURAL_DECODER_SENTINEL)
    stored_codebook_size, stored_embedding_dim, generator_kind_tag = struct.unpack(
        "<HHB", blob[pos : pos + 5]
    )
    pos += 5
    seed_len = struct.unpack("<I", blob[pos : pos + 4])[0]
    pos += 4
    if seed_len <= 0 or seed_len > 256:
        raise ValueError(f"procedural seed length {seed_len} outside [1, 256]")
    seed_bytes = blob[pos : pos + seed_len]
    pos += seed_len
    if len(seed_bytes) != seed_len:
        raise ValueError("procedural decoder section truncated before seed bytes")
    if stored_codebook_size != codebook_size or stored_embedding_dim != embedding_dim:
        raise ValueError(
            "procedural envelope codebook shape "
            f"({stored_codebook_size}, {stored_embedding_dim}) does not match "
            f"VQV1 header ({codebook_size}, {embedding_dim})"
        )
    generator_kind_by_tag = {0: "xorshift", 1: "lcg", 2: "pcg64"}
    try:
        generator_kind = generator_kind_by_tag[int(generator_kind_tag)]
    except KeyError as exc:
        raise ValueError(f"unknown procedural generator tag: {generator_kind_tag}") from exc

    raw = brotli.decompress(blob[pos:])
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("procedural decoder blob did not unpickle to a dict")
    sd["codebook"] = _derive_procedural_codebook_tensor(
        seed_bytes=seed_bytes,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        generator_kind=generator_kind,
    )
    _validate_runtime_state_dict(sd)
    return sd


def _deserialize_state_dict(
    blob: bytes,
    *,
    codebook_size: int,
    embedding_dim: int,
) -> dict[str, torch.Tensor]:
    if blob.startswith(VQV1_PROCEDURAL_DECODER_SENTINEL):
        return _deserialize_procedural_state_dict(
            blob,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
        )
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _pack_indices_int16(indices: torch.Tensor, codebook_size: int) -> bytes:
    """Pack int64 codebook indices into int16 bytes.

    Indices in [0, K) where K <= 65536. We store them as signed int16 by
    offsetting: ``raw_int16 = idx - 32768`` so the int16 range [-32768, 32767]
    can represent the full [0, 65535] range. Unpack via inverse offset.
    """
    if indices.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"indices must be int; got {indices.dtype}")
    if codebook_size <= 0 or codebook_size > 65536:
        raise ValueError(f"codebook_size {codebook_size} not in (0, 65536]")
    if int(indices.min()) < 0 or int(indices.max()) >= codebook_size:
        raise ValueError(
            f"indices range [{int(indices.min())}, {int(indices.max())}] "
            f"out of [0, {codebook_size})"
        )
    # Offset and cast to int16
    shifted = (indices.to(torch.int64) - 32768).to(torch.int16)
    return shifted.contiguous().cpu().numpy().tobytes()


def _unpack_indices_int16(blob: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    """Unpack int16 bytes back to int64 codebook indices (inverse offset)."""
    import numpy as np  # local

    arr = np.frombuffer(blob, dtype=np.int16).copy()
    t = torch.from_numpy(arr).view(*shape).to(torch.int64) + 32768
    return t


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    indices: torch.Tensor,
    meta: dict[str, object],
    *,
    codebook_size: int,
    embedding_dim: int,
    schema_version: int = VQV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained vq_vae state into monolithic 0.bin bytes."""
    if schema_version != VQV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if indices.dim() != 4:
        raise ValueError(
            f"indices must be 4-D (num_pairs, 2, H_GRID, W_GRID); got {tuple(indices.shape)}"
        )
    if indices.shape[1] != 2:
        raise ValueError(f"indices.shape[1] must be 2 (frame_0, frame_1); got {indices.shape[1]}")

    num_pairs = int(indices.shape[0])
    h_grid = int(indices.shape[2])
    w_grid = int(indices.shape[3])

    for name, v in (
        ("codebook_size", codebook_size),
        ("embedding_dim", embedding_dim),
        ("num_pairs", num_pairs),
        ("h_grid", h_grid),
        ("w_grid", w_grid),
    ):
        if v <= 0 or v > 0xFFFF:
            raise ValueError(f"{name}={v} out of u16 range")

    decoder_blob = _serialize_state_dict(decoder_state_dict)
    indices_bytes = _pack_indices_int16(indices, codebook_size)

    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        VQV1_HEADER_FMT,
        VQV1_MAGIC,
        schema_version,
        codebook_size,
        embedding_dim,
        num_pairs,
        h_grid,
        w_grid,
        len(decoder_blob),
        len(indices_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + indices_bytes + meta_bytes


def parse_archive(blob: bytes) -> VqVaeArchive:
    """Parse 0.bin bytes back into typed VqVaeArchive."""
    if len(blob) < VQV1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes; need >= {VQV1_HEADER_SIZE})")
    (
        magic,
        version,
        codebook_size,
        embedding_dim,
        num_pairs,
        h_grid,
        w_grid,
        decoder_len,
        indices_len,
        meta_len,
    ) = struct.unpack(VQV1_HEADER_FMT, blob[:VQV1_HEADER_SIZE])
    if magic != VQV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {VQV1_MAGIC!r})")
    if version != VQV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_indices_bytes = num_pairs * 2 * h_grid * w_grid * 2  # int16 = 2 bytes
    if indices_len != expected_indices_bytes:
        raise ValueError(
            f"indices_len {indices_len} != expected {expected_indices_bytes}"
        )

    pos = VQV1_HEADER_SIZE
    decoder_blob = blob[pos : pos + decoder_len]
    pos += decoder_len
    indices_blob = blob[pos : pos + indices_len]
    pos += indices_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    sd = _deserialize_state_dict(
        decoder_blob,
        codebook_size=int(codebook_size),
        embedding_dim=int(embedding_dim),
    )
    indices = _unpack_indices_int16(indices_blob, (num_pairs, 2, h_grid, w_grid))
    meta = json.loads(meta_blob.decode("utf-8"))

    return VqVaeArchive(
        decoder_state_dict=sd,
        indices=indices,
        meta=meta,
        schema_version=int(version),
        codebook_size=int(codebook_size),
        embedding_dim=int(embedding_dim),
    )


def compose_procedural_archive(
    original_archive_bytes: bytes,
    seed_bytes: bytes,
) -> bytes:
    """Thin convenience wrapper for VQ-VAE procedural-codebook archive composition.

    Per WAVE-3-VQ-VAE-PROCEDURAL-TRAINER-BUILD 2026-05-20 + sister DP1
    canonical pattern landing commit ``9cbfa471c``: delegates to
    :func:`tac.substrates.vq_vae.distillation_procedural_variant.compose_with_procedural_codebook`
    using canonical defaults (32-byte seed, PCG64, output shape
    ``(8192,)`` uint8 matching K=512 × D=8 × fp16 codebook footprint).

    Sister of :func:`pack_archive` (canonical builder for the trained-
    codebook variant). The procedural variant replaces the codebook tensor
    INSIDE the decoder state_dict with a 32-byte PCG64 seed; indices /
    meta sections are preserved byte-for-byte from the original archive.

    Args:
        original_archive_bytes: Existing VQV1 archive bytes (parseable
            via :func:`parse_archive`).
        seed_bytes: Procedural seed (8-256 bytes; canonical 32 bytes).

    Returns:
        Procedural-variant archive bytes with the codebook tensor REMOVED
        from the decoder state_dict + a procedural seed envelope prepended
        to the decoder blob.
    """
    # Lazy import to avoid cyclic-import friction; the variant module
    # imports VQV1_HEADER_FMT / VQV1_HEADER_SIZE / VQV1_MAGIC /
    # VQV1_SCHEMA_VERSION / parse_archive from this module.
    from tac.substrates.vq_vae.distillation_procedural_variant import (
        compose_with_procedural_codebook,
    )

    return compose_with_procedural_codebook(
        original_archive_bytes=original_archive_bytes,
        seed_bytes=seed_bytes,
    )
