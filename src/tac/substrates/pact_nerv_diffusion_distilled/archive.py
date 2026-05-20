# SPDX-License-Identifier: MIT
"""pact_nerv_diffusion_distilled archive grammar — monolithic 0.bin (PNDD).

Catalog #124 STRICT archive-grammar 8 fields declared in package ``__init__``.
Export-first grammar (HNeRV parity L2):

::

    MAGIC(4)              b"PNDD"     Pact-NeRV Diffusion-Distilled
    VERSION(1)            u8          schema version (currently 1)
    LATENT_DIM(2)         u16         cfg.latent_dim
    NUM_PAIRS(2)          u16         cfg.num_pairs
    NOISE_COND_DIM(2)     u16         cfg.noise_conditioning_dim
    STUDENT_BLOB_LEN(4)   u32         brotli student decoder state_dict len
    LATENT_BLOB_LEN(4)    u32         raw int16 latents bytes len
    META_BLOB_LEN(4)      u32         utf-8 json meta bytes len

Header: 4+1+2+2+2+4+4+4 = 23 bytes.

The teacher network is NOT shipped (per HNeRV parity L4 ≤200 LOC inflate
budget; distillation contract: teacher is compress-time only).

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

PNDD_MAGIC: bytes = b"PNDD"
PNDD_SCHEMA_VERSION: int = 1

PNDD_HEADER_FMT: str = "<4sBHHHIII"
PNDD_HEADER_SIZE: int = struct.calcsize(PNDD_HEADER_FMT)
assert PNDD_HEADER_SIZE == 23, "PNDD header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervDiffusionDistilledArchive:
    """Parsed archive structure — the inflate-time data contract."""

    student_state_dict: dict[str, torch.Tensor]
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
        raise ValueError("student_state_dict blob did not unpickle to a dict")
    return sd


def _quantize_tensor_to_int16(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    if tensor.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {tensor.dtype}")
    f = tensor.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_tensor(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    student_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = PNDD_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PNDD_SCHEMA_VERSION:
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

    noise_cond_dim = int(meta.get("noise_conditioning_dim", 16))
    if noise_cond_dim <= 0 or noise_cond_dim > 0xFFFF:
        raise ValueError(f"noise_conditioning_dim {noise_cond_dim} out of u16 range")

    q_latents, lat_scale, lat_zp = _quantize_tensor_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    student_blob = _serialize_state_dict(student_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PNDD_HEADER_FMT,
        PNDD_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        noise_cond_dim,
        len(student_blob),
        len(latent_bytes),
        len(meta_bytes),
    )
    return header + student_blob + latent_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervDiffusionDistilledArchive:
    if len(blob) < PNDD_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PNDD_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        noise_cond_dim,
        student_len,
        latent_len,
        meta_len,
    ) = struct.unpack(PNDD_HEADER_FMT, blob[:PNDD_HEADER_SIZE])
    if magic != PNDD_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PNDD_MAGIC!r})")
    if version != PNDD_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = PNDD_HEADER_SIZE
    end_student = end_header + student_len
    end_latents = end_student + latent_len
    end_meta = end_latents + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    student_blob = blob[end_header:end_student]
    latent_blob = blob[end_student:end_latents]
    meta_blob = blob[end_latents:end_meta]

    sd = _deserialize_state_dict(student_blob)
    meta = json.loads(meta_blob.decode("utf-8"))
    meta["noise_conditioning_dim"] = int(noise_cond_dim)

    import numpy as np
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)

    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequantize_tensor(q_latents, lat_scale, lat_zp)

    return PactNervDiffusionDistilledArchive(
        student_state_dict=sd,
        latents=latents,
        meta=meta,
        schema_version=int(version),
    )
