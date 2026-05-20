# SPDX-License-Identifier: MIT
"""pact_nerv_diffusion_trajectory archive grammar — monolithic 0.bin (PDT).

::

    MAGIC(4)              b"PDT\\x00"
    VERSION(1)            u8
    LATENT_DIM(2)         u16
    NUM_PAIRS(2)          u16
    NUM_TIMESTEPS(1)      u8
    DECODER_BLOB_LEN(4)   u32  brotli decoder + predictor state_dict
    SEEDS_BLOB_LEN(4)     u32  raw int16 seeds
    META_BLOB_LEN(4)      u32

Header: 4+1+2+2+1+4+4+4 = 22 bytes.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PDT_MAGIC: bytes = b"PDT\x00"
PDT_SCHEMA_VERSION: int = 1

PDT_HEADER_FMT: str = "<4sBHHBIII"
PDT_HEADER_SIZE: int = struct.calcsize(PDT_HEADER_FMT)
assert PDT_HEADER_SIZE == 22, "PDT header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervDiffusionTrajectoryArchive:
    decoder_state_dict: dict[str, torch.Tensor]
    seeds: torch.Tensor
    meta: dict[str, object]
    schema_version: int
    num_timesteps: int


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
    decoder_state_dict: dict[str, torch.Tensor],
    seeds: torch.Tensor,
    meta: dict[str, object],
    *,
    num_timesteps: int,
    schema_version: int = PDT_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PDT_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if seeds.dim() != 2:
        raise ValueError(f"seeds must be 2-D; got {tuple(seeds.shape)}")
    if num_timesteps <= 0 or num_timesteps > 255:
        raise ValueError(f"num_timesteps {num_timesteps} out of u8 range")

    num_pairs, latent_dim = int(seeds.shape[0]), int(seeds.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_seeds, seed_scale, seed_zp = _quantize_tensor_to_int16(seeds)
    seed_bytes = q_seeds.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_seed_quant_scale"] = float(seed_scale)
    meta_with_quant["_seed_quant_zero_point"] = float(seed_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PDT_HEADER_FMT,
        PDT_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        num_timesteps,
        len(decoder_blob),
        len(seed_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + seed_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervDiffusionTrajectoryArchive:
    if len(blob) < PDT_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PDT_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        num_timesteps,
        decoder_len,
        seed_len,
        meta_len,
    ) = struct.unpack(PDT_HEADER_FMT, blob[:PDT_HEADER_SIZE])
    if magic != PDT_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PDT_MAGIC!r})")
    if version != PDT_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_seed_bytes = num_pairs * latent_dim * 2
    if seed_len != expected_seed_bytes:
        raise ValueError(
            f"seed_len {seed_len} != num_pairs*latent_dim*2 = {expected_seed_bytes}"
        )

    end_header = PDT_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_seeds = end_decoder + seed_len
    end_meta = end_seeds + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    seed_blob = blob[end_decoder:end_seeds]
    meta_blob = blob[end_seeds:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_seeds = torch.from_numpy(
        np.frombuffer(seed_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)

    seed_scale = float(meta.pop("_seed_quant_scale"))
    seed_zp = float(meta.pop("_seed_quant_zero_point"))
    seeds = _dequantize_tensor(q_seeds, seed_scale, seed_zp)

    return PactNervDiffusionTrajectoryArchive(
        decoder_state_dict=sd,
        seeds=seeds,
        meta=meta,
        schema_version=int(version),
        num_timesteps=int(num_timesteps),
    )
