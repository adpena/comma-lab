# SPDX-License-Identifier: MIT
"""Torch-free NRV1-v2 archive reader for Nirvana numpy-portable inflate.

This module is the runtime-safe archive surface. It owns the fixed NRV1 header
grammar, section splitting, numpy-native decoder weight decode, and numpy
latent dequantization. It intentionally imports no training framework so it can
be vendored directly into a contest inflate tree.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.substrates._shared.numpy_portable_inflate import unpack_state_dict_numpy

NRV1_MAGIC: bytes = b"NRV1"
NRV1_SCHEMA_VERSION: int = 2

NRV1_HEADER_FMT: str = "<4sBHHBBHIII"
NRV1_HEADER_SIZE: int = struct.calcsize(NRV1_HEADER_FMT)
assert NRV1_HEADER_SIZE == 25, "NRV1 header size invariant (patch grid + embed dim)"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class NirvanaArchiveSections:
    """Validated NRV1 section offsets and payload slices."""

    schema_version: int
    latent_dim: int
    num_pairs: int
    patch_grid_h: int
    patch_grid_w: int
    patch_embed_dim: int
    decoder_blob: bytes
    latent_blob: bytes
    meta_blob: bytes


@dataclass(frozen=True)
class NirvanaArchiveNumpy:
    """Torch-free parsed NRV1 archive: numpy weights, latents, and metadata."""

    decoder_state_dict: dict[str, np.ndarray]
    latents: np.ndarray
    meta: dict[str, object]
    schema_version: int
    patch_grid_h: int
    patch_grid_w: int
    patch_embed_dim: int


def split_archive_sections(blob: bytes) -> NirvanaArchiveSections:
    """Validate an NRV1 blob and return its typed section slices."""
    if len(blob) < NRV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {NRV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        patch_grid_h,
        patch_grid_w,
        patch_embed_dim,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(NRV1_HEADER_FMT, blob[:NRV1_HEADER_SIZE])
    if magic != NRV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {NRV1_MAGIC!r})")
    if version != NRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = NRV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_meta = end_latents + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    return NirvanaArchiveSections(
        schema_version=int(version),
        latent_dim=int(latent_dim),
        num_pairs=int(num_pairs),
        patch_grid_h=int(patch_grid_h),
        patch_grid_w=int(patch_grid_w),
        patch_embed_dim=int(patch_embed_dim),
        decoder_blob=blob[end_header:end_decoder],
        latent_blob=blob[end_decoder:end_latents],
        meta_blob=blob[end_latents:end_meta],
    )


def deserialize_numpy_state_dict(blob: bytes) -> dict[str, np.ndarray]:
    """Deserialize the brotli-wrapped numpy-native NRV1-v2 decoder state dict."""
    np_sd = unpack_state_dict_numpy(brotli.decompress(blob))
    return {k: v.astype(np.float32) for k, v in np_sd.items()}


def parse_archive_numpy(blob: bytes) -> NirvanaArchiveNumpy:
    """Parse an NRV1-v2 archive with no torch/MLX dependency."""
    sections = split_archive_sections(blob)
    sd = deserialize_numpy_state_dict(sections.decoder_blob)
    meta = json.loads(sections.meta_blob.decode("utf-8"))

    q = np.frombuffer(sections.latent_blob, dtype=np.int16).reshape(
        sections.num_pairs, sections.latent_dim
    )
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    latents = (q.astype(np.float32) + 32767.0) * scale + zp

    return NirvanaArchiveNumpy(
        decoder_state_dict=sd,
        latents=latents,
        meta=meta,
        schema_version=sections.schema_version,
        patch_grid_h=sections.patch_grid_h,
        patch_grid_w=sections.patch_grid_w,
        patch_embed_dim=sections.patch_embed_dim,
    )


__all__ = [
    "BROTLI_QUALITY",
    "NRV1_HEADER_FMT",
    "NRV1_HEADER_SIZE",
    "NRV1_MAGIC",
    "NRV1_SCHEMA_VERSION",
    "NirvanaArchiveNumpy",
    "NirvanaArchiveSections",
    "deserialize_numpy_state_dict",
    "parse_archive_numpy",
    "split_archive_sections",
]
