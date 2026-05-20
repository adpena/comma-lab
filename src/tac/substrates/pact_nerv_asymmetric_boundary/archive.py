# SPDX-License-Identifier: MIT
"""pact_nerv_asymmetric_boundary archive grammar - PAB1 monolithic 0.bin."""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PAB1_MAGIC: bytes = b"PAB1"
PAB1_SCHEMA_VERSION: int = 1
# Header: magic(4) + version(1) + latent_dim(2) + num_pairs(2)
#       + boundary_signal_dim(1) + decoder_len(4) + latent_len(4)
#       + boundary_len(4) + meta_len(4) = 26 bytes
PAB1_HEADER_FMT: str = "<4sBHHBIIII"
PAB1_HEADER_SIZE: int = struct.calcsize(PAB1_HEADER_FMT)
assert PAB1_HEADER_SIZE == 26
BROTLI_QUALITY = 9


@dataclass(frozen=True)
class PactNervAsymmetricBoundaryArchive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    boundary_signals: torch.Tensor
    meta: dict[str, object]
    schema_version: int
    boundary_signal_dim: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    sd = pickle.loads(brotli.decompress(blob))
    if not isinstance(sd, dict):
        raise ValueError("decoder_state_dict blob did not unpickle to a dict")
    return sd


def _quantize_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    qu = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (qu - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequant_int16(q: torch.Tensor, scale: float, zp: float) -> torch.Tensor:
    qu = q.to(torch.float32) + 32767.0
    return qu * float(scale) + float(zp)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    boundary_signals: torch.Tensor,
    meta: dict[str, object],
    *,
    boundary_signal_dim: int,
    schema_version: int = PAB1_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PAB1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if boundary_signals.dim() != 2:
        raise ValueError(f"boundary_signals must be 2-D; got {tuple(boundary_signals.shape)}")
    if boundary_signal_dim < 0 or boundary_signal_dim > 255:
        raise ValueError(f"boundary_signal_dim {boundary_signal_dim} out of u8 range")
    if boundary_signals.shape[1] != boundary_signal_dim:
        raise ValueError(
            f"boundary_signals dim {boundary_signals.shape[1]} != {boundary_signal_dim}"
        )
    if boundary_signals.shape[0] != latents.shape[0]:
        raise ValueError("boundary_signals num_pairs != latents num_pairs")
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    q_b, b_scale, b_zp = _quantize_int16(boundary_signals)
    lat_bytes = q_lat.contiguous().numpy().tobytes()
    b_bytes = q_b.contiguous().numpy().tobytes()
    dec_blob = _serialize_state_dict(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_q["_b_quant_scale"] = float(b_scale)
    meta_q["_b_quant_zero_point"] = float(b_zp)
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        PAB1_HEADER_FMT, PAB1_MAGIC, schema_version, latent_dim, num_pairs,
        boundary_signal_dim, len(dec_blob), len(lat_bytes), len(b_bytes), len(meta_bytes),
    )
    return header + dec_blob + lat_bytes + b_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervAsymmetricBoundaryArchive:
    if len(blob) < PAB1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, num_pairs, boundary_signal_dim,
        dec_len, lat_len, b_len, meta_len,
    ) = struct.unpack(PAB1_HEADER_FMT, blob[:PAB1_HEADER_SIZE])
    if magic != PAB1_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != PAB1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2
    expected_b = num_pairs * boundary_signal_dim * 2
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != {expected_lat}")
    if b_len != expected_b:
        raise ValueError(f"b_len {b_len} != {expected_b}")
    end_hdr = PAB1_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_b = end_lat + b_len
    end_meta = end_b + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")
    sd = _deserialize_state_dict(blob[end_hdr:end_dec])
    meta = json.loads(blob[end_b:end_meta].decode("utf-8"))
    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    q_b = torch.from_numpy(
        np.frombuffer(blob[end_lat:end_b], dtype=np.int16).copy()
    ).view(num_pairs, boundary_signal_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    b_scale = float(meta.pop("_b_quant_scale"))
    b_zp = float(meta.pop("_b_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)
    boundary_signals = _dequant_int16(q_b, b_scale, b_zp)
    return PactNervAsymmetricBoundaryArchive(
        decoder_state_dict=sd,
        latents=latents,
        boundary_signals=boundary_signals,
        meta=meta,
        schema_version=int(version),
        boundary_signal_dim=int(boundary_signal_dim),
    )
