# SPDX-License-Identifier: MIT
"""pact_nerv_cross_codec_b archive grammar — CC_B monolithic single-file 0.bin.

Header (34 bytes):
    MAGIC(4)                  b"CCBB"
    VERSION(1)                u8
    LATENT_DIM(2)             u16
    NUM_PAIRS(2)              u16
    POSE_DIM(1)               u8     PoseNet first 6 dims
    SCORE_TABLE_SIZE(1)       u8     PR106 latent-score-table
    PR106_BASE_BLOB_LEN(4)    u32    PR106 base codec bytes (placeholder at L0)
    DECODER_BLOB_LEN(4)       u32    Pact-NeRV-IA3 side-info decoder bytes
    LATENT_BLOB_LEN(4)        u32    Per-pair latents
    EGO_POSE_BLOB_LEN(4)      u32    Per-pair ego_pose (float32)
    SCORE_INDEX_BLOB_LEN(4)   u32    Per-pair PR106 score-table indices (u8)
    META_BLOB_LEN(4)          u32    utf-8 JSON meta
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

CC_B_MAGIC: bytes = b"CCBB"
CC_B_SCHEMA_VERSION: int = 1
CC_B_HEADER_FMT: str = "<4sBHHBBIIIIII"
CC_B_HEADER_SIZE: int = struct.calcsize(CC_B_HEADER_FMT)
assert CC_B_HEADER_SIZE == 35
BROTLI_QUALITY = 9


@dataclass(frozen=True)
class PactNervCrossCodecBArchive:
    pr106_base_bytes: bytes
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    ego_poses: torch.Tensor
    score_index_bytes: bytes
    meta: dict[str, object]
    schema_version: int
    score_table_size: int
    pose_dim: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
        if k not in ("ego_poses", "score_indices") and not k.startswith("base_codec.")
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
    pr106_base_bytes: bytes,
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    ego_poses: torch.Tensor,
    score_index_bytes: bytes,
    meta: dict[str, object],
    *,
    score_table_size: int,
    pose_dim: int,
    schema_version: int = CC_B_SCHEMA_VERSION,
) -> bytes:
    if schema_version != CC_B_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if ego_poses.dim() != 2:
        raise ValueError(f"ego_poses must be 2-D; got {tuple(ego_poses.shape)}")
    if score_table_size < 0 or score_table_size > 255:
        raise ValueError(f"score_table_size {score_table_size} out of u8 range")
    if pose_dim < 0 or pose_dim > 255:
        raise ValueError(f"pose_dim {pose_dim} out of u8 range")
    if not isinstance(pr106_base_bytes, (bytes, bytearray)):
        raise ValueError("pr106_base_bytes must be bytes")
    if not isinstance(score_index_bytes, (bytes, bytearray)):
        raise ValueError("score_index_bytes must be bytes")
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if int(ego_poses.shape[0]) != num_pairs or int(ego_poses.shape[1]) != pose_dim:
        raise ValueError(
            f"ego_poses shape {tuple(ego_poses.shape)} mismatch "
            f"({num_pairs}, {pose_dim})"
        )

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    lat_bytes = q_lat.contiguous().numpy().tobytes()
    ego_bytes = ego_poses.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()
    dec_blob = _serialize_state_dict(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        CC_B_HEADER_FMT, CC_B_MAGIC, schema_version, latent_dim, num_pairs,
        pose_dim, score_table_size,
        len(pr106_base_bytes), len(dec_blob), len(lat_bytes),
        len(ego_bytes), len(score_index_bytes), len(meta_bytes),
    )
    return (
        header
        + bytes(pr106_base_bytes)
        + dec_blob
        + lat_bytes
        + ego_bytes
        + bytes(score_index_bytes)
        + meta_bytes
    )


def parse_archive(blob: bytes) -> PactNervCrossCodecBArchive:
    if len(blob) < CC_B_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, num_pairs, pose_dim, score_table_size,
        pr106_len, dec_len, lat_len, ego_len, score_idx_len, meta_len,
    ) = struct.unpack(CC_B_HEADER_FMT, blob[:CC_B_HEADER_SIZE])
    if magic != CC_B_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != CC_B_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != {expected_lat}")
    expected_ego = num_pairs * pose_dim * 4
    if ego_len != expected_ego:
        raise ValueError(f"ego_len {ego_len} != {expected_ego}")
    end_hdr = CC_B_HEADER_SIZE
    end_pr106 = end_hdr + pr106_len
    end_dec = end_pr106 + dec_len
    end_lat = end_dec + lat_len
    end_ego = end_lat + ego_len
    end_score = end_ego + score_idx_len
    end_meta = end_score + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")
    pr106_base = bytes(blob[end_hdr:end_pr106])
    sd = _deserialize_state_dict(blob[end_pr106:end_dec])
    meta = json.loads(blob[end_score:end_meta].decode("utf-8"))
    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    ego_poses = torch.from_numpy(
        np.frombuffer(blob[end_lat:end_ego], dtype=np.float32).copy()
    ).view(num_pairs, pose_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)
    return PactNervCrossCodecBArchive(
        pr106_base_bytes=pr106_base,
        decoder_state_dict=sd,
        latents=latents,
        ego_poses=ego_poses,
        score_index_bytes=bytes(blob[end_ego:end_score]),
        meta=meta,
        schema_version=int(version),
        score_table_size=int(score_table_size),
        pose_dim=int(pose_dim),
    )
