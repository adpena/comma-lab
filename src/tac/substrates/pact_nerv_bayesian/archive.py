# SPDX-License-Identifier: MIT
"""pact_nerv_bayesian archive grammar — monolithic 0.bin (PBN).

::

    MAGIC(4)              b"PBN\\x00"
    VERSION(1)            u8
    LATENT_DIM(2)         u16
    NUM_PAIRS(2)          u16
    DECODER_BLOB_LEN(4)   u32  brotli decoder + bayesian mean state_dict
    LATENT_BLOB_LEN(4)    u32  raw int16 latents
    META_BLOB_LEN(4)      u32  utf-8 json meta

Header: 21 bytes. Bayesian posterior MEAN-ONLY (not sigma) is shipped per
Blundell §4 canonical inflate-time discipline (sigma is for training-time
sampling; only mean is needed at inflate).
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PBN_MAGIC: bytes = b"PBN\x00"
PBN_SCHEMA_VERSION: int = 1

PBN_HEADER_FMT: str = "<4sBHHIII"
PBN_HEADER_SIZE: int = struct.calcsize(PBN_HEADER_FMT)
assert PBN_HEADER_SIZE == 21, "PBN header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervBayesianArchive:
    decoder_state_dict: dict[str, torch.Tensor]
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
    latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = PBN_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PBN_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_latents, lat_scale, lat_zp = _quantize_tensor_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PBN_HEADER_FMT,
        PBN_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        len(decoder_blob),
        len(latent_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervBayesianArchive:
    if len(blob) < PBN_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PBN_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(PBN_HEADER_FMT, blob[:PBN_HEADER_SIZE])
    if magic != PBN_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PBN_MAGIC!r})")
    if version != PBN_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = PBN_HEADER_SIZE
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

    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequantize_tensor(q_latents, lat_scale, lat_zp)

    return PactNervBayesianArchive(
        decoder_state_dict=sd,
        latents=latents,
        meta=meta,
        schema_version=int(version),
    )
