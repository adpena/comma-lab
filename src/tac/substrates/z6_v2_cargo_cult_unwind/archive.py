# SPDX-License-Identifier: MIT
"""z6_v2_cargo_cult_unwind Z6V2CU1 archive grammar — monolithic single-file 0.bin.

Per Catalog #124 archive-grammar 8 fields + Catalog #146 contest-compliant
runtime contract:

Header (28 bytes):
    MAGIC(4)               b"Z6V2"
    VERSION(1)             u8
    LATENT_DIM(2)          u16
    EGO_DIM(2)             u16   FoE ego-motion vector dim (6)
    NUM_PAIRS(2)           u16
    DECODER_BLOB_LEN(4)    u32   level0_micro_film_predictor + level1_meso_film_predictor + heads
    LATENT_BLOB_LEN(4)     u32   per-pair latents (int16)
    EGO_BLOB_LEN(4)        u32   per-pair ego-motion vectors (int16)
    META_BLOB_LEN(4)       u32   json meta (cooperative_receiver_beta, hierarchy boundary, etc.)
    RESERVED(1)            u8    padding

Total header size: 28 bytes per struct calc below.

Per Catalog #272 distinguishing-feature integration contract: the archive
sections that distinguish Z6-v2 from sister substrates are:
1. ``hierarchy_weights_level0_blob`` — first 3 FiLM-conditioned blocks (micro)
2. ``hierarchy_weights_level1_blob`` — remaining 4 FiLM-conditioned blocks (meso)
3. ``ego_motion_focus_of_expansion_blob`` — per-pair (tx,ty,tz,rx,ry,rz) FoE prior
4. ``predictor_latents_blob`` — per-pair residual stream

All 4 sections are packed into the canonical 0.bin blob; the Rao-Ballard
hierarchy boundary + cooperative_receiver_beta + FiLM generator depth +
film_hidden_width are in the meta blob so the inflate runtime can re-instantiate
the canonical Z6V2Substrate without ambient state.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

Z6V2_MAGIC: bytes = b"Z6V2"
Z6V2_SCHEMA_VERSION: int = 1
# Header format: 4s (magic) + B (version) + H (latent_dim) + H (ego_dim) +
# H (num_pairs) + I (dec_blob_len) + I (lat_blob_len) + I (ego_blob_len) +
# I (meta_blob_len) + B (reserved)
Z6V2_HEADER_FMT: str = "<4sBHHHIIIIB"
Z6V2_HEADER_SIZE: int = struct.calcsize(Z6V2_HEADER_FMT)
assert Z6V2_HEADER_SIZE == 28
BROTLI_QUALITY = 9


@dataclass(frozen=True)
class Z6V2Archive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    ego_vecs: torch.Tensor
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
    ego_vecs: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = Z6V2_SCHEMA_VERSION,
) -> bytes:
    if schema_version != Z6V2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if ego_vecs.dim() != 2:
        raise ValueError(f"ego_vecs must be 2-D; got {tuple(ego_vecs.shape)}")
    if latents.shape[0] != ego_vecs.shape[0]:
        raise ValueError(
            f"latents.shape[0]={latents.shape[0]} != "
            f"ego_vecs.shape[0]={ego_vecs.shape[0]}"
        )
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    ego_dim = int(ego_vecs.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if ego_dim <= 0 or ego_dim > 0xFFFF:
        raise ValueError(f"ego_dim {ego_dim} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    lat_bytes = q_lat.contiguous().numpy().tobytes()

    q_ego, ego_scale, ego_zp = _quantize_int16(ego_vecs)
    ego_bytes = q_ego.contiguous().numpy().tobytes()

    dec_blob = _serialize_state_dict(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_q["_ego_quant_scale"] = float(ego_scale)
    meta_q["_ego_quant_zero_point"] = float(ego_zp)
    meta_bytes = json.dumps(
        meta_q, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z6V2_HEADER_FMT,
        Z6V2_MAGIC,
        schema_version,
        latent_dim,
        ego_dim,
        num_pairs,
        len(dec_blob),
        len(lat_bytes),
        len(ego_bytes),
        len(meta_bytes),
        0,  # reserved
    )
    return header + dec_blob + lat_bytes + ego_bytes + meta_bytes


def parse_archive(blob: bytes) -> Z6V2Archive:
    if len(blob) < Z6V2_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, ego_dim, num_pairs,
        dec_len, lat_len, ego_len, meta_len, _reserved,
    ) = struct.unpack(Z6V2_HEADER_FMT, blob[:Z6V2_HEADER_SIZE])
    if magic != Z6V2_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != Z6V2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2  # int16
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != expected {expected_lat}")
    expected_ego = num_pairs * ego_dim * 2  # int16
    if ego_len != expected_ego:
        raise ValueError(f"ego_len {ego_len} != expected {expected_ego}")
    end_hdr = Z6V2_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_ego = end_lat + ego_len
    end_meta = end_ego + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")

    sd = _deserialize_state_dict(blob[end_hdr:end_dec])
    meta = json.loads(blob[end_ego:end_meta].decode("utf-8"))

    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)

    q_ego = torch.from_numpy(
        np.frombuffer(blob[end_lat:end_ego], dtype=np.int16).copy()
    ).view(num_pairs, ego_dim)
    ego_scale = float(meta.pop("_ego_quant_scale"))
    ego_zp = float(meta.pop("_ego_quant_zero_point"))
    ego_vecs = _dequant_int16(q_ego, ego_scale, ego_zp)

    return Z6V2Archive(
        decoder_state_dict=sd,
        latents=latents,
        ego_vecs=ego_vecs,
        meta=meta,
        schema_version=int(version),
    )


__all__ = [
    "BROTLI_QUALITY",
    "Z6V2Archive",
    "Z6V2_HEADER_FMT",
    "Z6V2_HEADER_SIZE",
    "Z6V2_MAGIC",
    "Z6V2_SCHEMA_VERSION",
    "pack_archive",
    "parse_archive",
]
