# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard archive grammar — Z5RB1 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (<=200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

Z5RB1 distinguishes Z5 hierarchical (z_low + z_high + predictor) from sister
Z6PCWM1 (single-latent + FiLM) and Z7MCM2 (state-space) via the ``Z5RB`` magic
prefix. The 2-level Rao-Ballard hierarchy is encoded in the archive grammar
itself: TWO latent blobs (low + high) + the predictor blob that maps
``(z_high, ego) -> z_low_pred``.

Grammar::

    MAGIC(4)             b"Z5RB"
    VERSION(1)           u8       schema version (currently 1)
    LOW_LATENT_DIM(2)    u16      cfg.low_latent_dim (e.g. 24)
    HIGH_LATENT_DIM(2)   u16      cfg.high_latent_dim (e.g. 16)
    EGO_DIM(2)           u16      cfg.ego_dim (e.g. 6)
    NUM_PAIRS(2)         u16      cfg.num_pairs (e.g. 600)
    DECODER_BLOB_LEN(4)  u32      brotli-compressed decoder state_dict bytes len
    PREDICTOR_BLOB_LEN(4) u32     brotli-compressed predictor state_dict bytes len
    LOW_LAT_BLOB_LEN(4)  u32      brotli-compressed low_latents fp16 bytes len
    HIGH_LAT_BLOB_LEN(4) u32      brotli-compressed high_latents fp16 bytes len
    EGO_BLOB_LEN(4)      u32      brotli-compressed ego_vecs fp16 bytes len
    META_BLOB_LEN(4)     u32      sorted-keys JSON utf-8 bytes len
    DECODER_BLOB         ...      brotli(quality=9) decoder state_dict fp16
    PREDICTOR_BLOB       ...      brotli(quality=9) predictor state_dict fp16
    LOW_LAT_BLOB         ...      brotli(quality=9) low_latents fp16
    HIGH_LAT_BLOB        ...      brotli(quality=9) high_latents fp16
    EGO_BLOB             ...      brotli(quality=9) ego_vecs fp16
    META_BLOB            ...      sorted-keys JSON with z5_rao_ballard_meta tag

Header total: 4 + 1 + 2 + 2 + 2 + 2 + 6*4 = 37 bytes.

CLAUDE.md compliance:
- No silent defaults
- No /tmp paths
- No scorer load
- Deterministic: same input -> same bytes (sorted-keys JSON; fixed brotli q=9;
  fp16 cast on CPU)
"""
# AUTOCAST_FP16_WAIVED:archive_serializer_uses_torch_fp16_cast_for_byte_determinism_not_cuda_autocast
# NO_GRAD_WAIVED:archive_serializer_path_is_export_not_training
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

Z5RB1_MAGIC: bytes = b"Z5RB"
Z5RB1_SCHEMA_VERSION: int = 1
Z5RB1_HEADER_FMT: str = "<4sBHHHHIIIIII"
Z5RB1_HEADER_SIZE: int = struct.calcsize(Z5RB1_HEADER_FMT)
assert Z5RB1_HEADER_SIZE == 37, f"header size invariant: expected 37, got {Z5RB1_HEADER_SIZE}"

Z5RB1_SECTION_ROLES: tuple[str, ...] = (
    "decoder_state_dict",
    "predictor_state_dict",
    "low_latents",
    "high_latents",
    "ego_vecs",
    "meta_json",
)

_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class Z5RaoBallardArchive:
    """Parsed Z5RB1 archive — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    predictor_state_dict: dict[str, torch.Tensor]
    low_latents: torch.Tensor    # (num_pairs, low_latent_dim) float32
    high_latents: torch.Tensor   # (num_pairs, high_latent_dim) float32
    ego_vecs: torch.Tensor       # (num_pairs, ego_dim) float32
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Deterministic state_dict serialization to brotli'd fp16 bytes."""
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} too long for u16")
        shape = tuple(int(s) for s in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(
            header_fmt, len(key_bytes), key_bytes, len(shape), *shape
        )
        parts.append(header)
        parts.append(tensor.numpy().tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
    pos = 0
    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("truncated state_dict key-len header")
        key_len = struct.unpack_from("<H", raw, pos)[0]
        pos += 2
        if pos + key_len > len(raw):
            raise ValueError("truncated state_dict key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        if pos + 1 > len(raw):
            raise ValueError("truncated state_dict ndim byte")
        ndim = raw[pos]
        pos += 1
        if pos + 4 * ndim > len(raw):
            raise ValueError("truncated state_dict shape bytes")
        shape = struct.unpack_from(f"<{ndim}I", raw, pos)
        pos += 4 * ndim
        numel = int(np.prod(shape)) if ndim else 1
        nbytes = numel * 2  # fp16
        if pos + nbytes > len(raw):
            raise ValueError(f"truncated tensor {key!r} bytes")
        arr = np.frombuffer(raw, dtype=np.float16, count=numel, offset=pos).reshape(
            shape
        ).copy()
        pos += nbytes
        sd[key] = torch.from_numpy(arr).to(dtype=torch.float32)
    return sd


def _serialize_latents_fp16(latents: torch.Tensor) -> bytes:
    """Serialize a (N, D) float tensor to brotli'd fp16 bytes."""
    arr = (
        latents.detach()
        .to("cpu", dtype=torch.float16)
        .contiguous()
        .numpy()
        .tobytes(order="C")
    )
    return bytes(brotli.compress(arr, quality=_BROTLI_QUALITY))


def _deserialize_latents_fp16(
    blob: bytes, num_pairs: int, dim: int
) -> torch.Tensor:
    raw = brotli.decompress(blob)
    expected = num_pairs * dim * 2  # fp16
    if len(raw) != expected:
        raise ValueError(
            f"latents blob size mismatch: got {len(raw)} bytes, expected {expected}"
        )
    arr = np.frombuffer(raw, dtype=np.float16).reshape(num_pairs, dim).copy()
    return torch.from_numpy(arr).to(dtype=torch.float32)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    predictor_state_dict: dict[str, torch.Tensor],
    low_latents: torch.Tensor,
    high_latents: torch.Tensor,
    ego_vecs: torch.Tensor,
    meta: dict[str, object],
) -> bytes:
    """Pack the canonical Z5RB1 ``0.bin`` bytes deterministically."""
    if low_latents.dim() != 2:
        raise ValueError(f"low_latents must be 2D; got {tuple(low_latents.shape)}")
    if high_latents.dim() != 2:
        raise ValueError(f"high_latents must be 2D; got {tuple(high_latents.shape)}")
    if ego_vecs.dim() != 2:
        raise ValueError(f"ego_vecs must be 2D; got {tuple(ego_vecs.shape)}")
    num_pairs = int(low_latents.shape[0])
    if int(high_latents.shape[0]) != num_pairs:
        raise ValueError("high_latents num_pairs mismatch low_latents")
    if int(ego_vecs.shape[0]) != num_pairs:
        raise ValueError("ego_vecs num_pairs mismatch low_latents")
    low_latent_dim = int(low_latents.shape[1])
    high_latent_dim = int(high_latents.shape[1])
    ego_dim = int(ego_vecs.shape[1])
    for v, name in (
        (low_latent_dim, "low_latent_dim"),
        (high_latent_dim, "high_latent_dim"),
        (ego_dim, "ego_dim"),
        (num_pairs, "num_pairs"),
    ):
        if v < 1 or v > 0xFFFF:
            raise ValueError(f"{name}={v} out of u16 range")

    decoder_blob = _serialize_state_dict(decoder_state_dict)
    predictor_blob = _serialize_state_dict(predictor_state_dict)
    low_blob = _serialize_latents_fp16(low_latents)
    high_blob = _serialize_latents_fp16(high_latents)
    ego_blob = _serialize_latents_fp16(ego_vecs)
    meta_blob = json.dumps(
        {**meta, "z5_rao_ballard_meta": True}, sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z5RB1_HEADER_FMT,
        Z5RB1_MAGIC,
        Z5RB1_SCHEMA_VERSION,
        low_latent_dim,
        high_latent_dim,
        ego_dim,
        num_pairs,
        len(decoder_blob),
        len(predictor_blob),
        len(low_blob),
        len(high_blob),
        len(ego_blob),
        len(meta_blob),
    )
    return (
        header
        + decoder_blob
        + predictor_blob
        + low_blob
        + high_blob
        + ego_blob
        + meta_blob
    )


def parse_archive(blob: bytes) -> Z5RaoBallardArchive:
    """Inverse of ``pack_archive``."""
    if len(blob) < Z5RB1_HEADER_SIZE:
        raise ValueError(
            f"archive too short for header: {len(blob)} < {Z5RB1_HEADER_SIZE}"
        )
    (
        magic,
        schema_version,
        low_latent_dim,
        high_latent_dim,
        ego_dim,
        num_pairs,
        decoder_len,
        predictor_len,
        low_len,
        high_len,
        ego_len,
        meta_len,
    ) = struct.unpack(Z5RB1_HEADER_FMT, blob[:Z5RB1_HEADER_SIZE])
    if magic != Z5RB1_MAGIC:
        raise ValueError(f"unexpected magic: got {magic!r}, want {Z5RB1_MAGIC!r}")
    if schema_version != Z5RB1_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {schema_version}; want {Z5RB1_SCHEMA_VERSION}"
        )

    pos = Z5RB1_HEADER_SIZE
    decoder_blob = blob[pos : pos + decoder_len]
    pos += decoder_len
    predictor_blob = blob[pos : pos + predictor_len]
    pos += predictor_len
    low_blob = blob[pos : pos + low_len]
    pos += low_len
    high_blob = blob[pos : pos + high_len]
    pos += high_len
    ego_blob = blob[pos : pos + ego_len]
    pos += ego_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(
            f"trailing bytes after meta section: {len(blob) - pos} extra"
        )

    return Z5RaoBallardArchive(
        decoder_state_dict=_deserialize_state_dict(decoder_blob),
        predictor_state_dict=_deserialize_state_dict(predictor_blob),
        low_latents=_deserialize_latents_fp16(low_blob, num_pairs, low_latent_dim),
        high_latents=_deserialize_latents_fp16(high_blob, num_pairs, high_latent_dim),
        ego_vecs=_deserialize_latents_fp16(ego_blob, num_pairs, ego_dim),
        meta=json.loads(meta_blob.decode("utf-8")),
        schema_version=int(schema_version),
    )


__all__ = [
    "Z5RB1_HEADER_FMT",
    "Z5RB1_HEADER_SIZE",
    "Z5RB1_MAGIC",
    "Z5RB1_SCHEMA_VERSION",
    "Z5RB1_SECTION_ROLES",
    "Z5RaoBallardArchive",
    "pack_archive",
    "parse_archive",
]
