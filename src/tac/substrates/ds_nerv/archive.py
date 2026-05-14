# SPDX-License-Identifier: MIT
"""ds_nerv archive grammar — monolithic single-file ``0.bin`` (DSV1).

Catalog #124 STRICT archive-grammar 8 fields declared in package
``__init__``. Export-first grammar (L2):

::

    MAGIC(4)            b"DSV1"  Depth-Separable Variant 1
    VERSION(1)          u8       schema version (currently 1)
    LATENT_DIM(2)       u16      cfg.latent_dim
    NUM_PAIRS(2)        u16      cfg.num_pairs
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder state_dict bytes len
    LATENT_BLOB_LEN(4)  u32      raw int16 latents bytes len
    META_BLOB_LEN(4)    u32      utf-8 json meta bytes len
    DECODER_BLOB        ...      brotli(quality=9) of pickled state_dict
    LATENT_BLOB         ...      int16 latents row-major
    META_BLOB           ...      json: {"sin_freq": ..., "decoder_channels": [...]}

Header: 4+1+2+2+4+4+4 = 21 bytes (same as SHV1 — DSV1's distinctive
section is the "depthwise_separable_params" subset inside DECODER_BLOB).

Catalog #124 parser-section manifest enumerates 5 logical sections:
HEADER + DECODER_BLOB + LATENT_BLOB + META_BLOB + (implicit
"depthwise_separable_params" subset). The 5th section is logical-only —
the depthwise + pointwise weight keys are co-located in DECODER_BLOB but
the design declares them as a separately-addressable group so future
rate-axis work can quantize them independently.

CLAUDE.md compliance: deterministic, no /tmp, no scorer load.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

DSV1_MAGIC: bytes = b"DSV1"
DSV1_SCHEMA_VERSION: int = 1

DSV1_HEADER_FMT: str = "<4sBHHIII"
DSV1_HEADER_SIZE: int = struct.calcsize(DSV1_HEADER_FMT)
assert DSV1_HEADER_SIZE == 21, "DSV1 header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class DsnervArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (all model weights except per-pair latents).

    Includes the depthwise + pointwise weights as a logically-grouped subset
    ("depthwise_separable_params" section per Catalog #124).
    """

    latents: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("decoder_state_dict blob did not unpickle to a dict")
    return sd


def _quantize_latents_to_int16(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    if latents.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"latents must be float; got {latents.dtype}")
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        # FFFF Catalog #158 fix: -32767 fill so dequant = 0*scale + lo = lo
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_latents(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = DSV1_SCHEMA_VERSION,
) -> bytes:
    if schema_version != DSV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_latents, scale, zero_point = _quantize_latents_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()
    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_quant_scale"] = float(scale)
    meta_with_quant["_quant_zero_point"] = float(zero_point)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        DSV1_HEADER_FMT,
        DSV1_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        len(decoder_blob),
        len(latent_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + meta_bytes


def parse_archive(blob: bytes) -> DsnervArchive:
    if len(blob) < DSV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {DSV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(DSV1_HEADER_FMT, blob[:DSV1_HEADER_SIZE])
    if magic != DSV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {DSV1_MAGIC!r})")
    if version != DSV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = DSV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_meta = end_latents + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    meta_blob = blob[end_latents:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    latents = _dequantize_latents(q_latents, scale, zp)

    return DsnervArchive(
        decoder_state_dict=sd,
        latents=latents,
        meta=meta,
        schema_version=int(version),
    )
