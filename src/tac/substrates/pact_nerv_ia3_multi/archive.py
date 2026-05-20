# SPDX-License-Identifier: MIT
"""pact_nerv_ia3_multi archive grammar - PIM1 monolithic 0.bin (multi-block IA3)."""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PIM1_MAGIC: bytes = b"PIM1"
PIM1_SCHEMA_VERSION: int = 1
# Header: magic(4) + version(1) + latent_dim(2) + num_pairs(2) + pose_dim(1)
#       + difficulty_dim(1) + decoder_len(4) + latent_len(4) + pose_len(4)
#       + diff_len(4) + meta_len(4) = 31 bytes
PIM1_HEADER_FMT: str = "<4sBHHBBIIIII"
PIM1_HEADER_SIZE: int = struct.calcsize(PIM1_HEADER_FMT)
assert PIM1_HEADER_SIZE == 31
BROTLI_QUALITY = 9


@dataclass(frozen=True)
class PactNervIa3MultiArchive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    ego_poses: torch.Tensor
    difficulties: torch.Tensor
    meta: dict[str, object]
    schema_version: int
    pose_dim: int
    difficulty_dim: int


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
    ego_poses: torch.Tensor,
    difficulties: torch.Tensor,
    meta: dict[str, object],
    *,
    pose_dim: int,
    difficulty_dim: int,
    schema_version: int = PIM1_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PIM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if ego_poses.dim() != 2:
        raise ValueError(f"ego_poses must be 2-D; got {tuple(ego_poses.shape)}")
    if difficulties.dim() != 2:
        raise ValueError(f"difficulties must be 2-D; got {tuple(difficulties.shape)}")
    if pose_dim < 0 or pose_dim > 255:
        raise ValueError(f"pose_dim {pose_dim} out of u8 range")
    if difficulty_dim < 0 or difficulty_dim > 255:
        raise ValueError(f"difficulty_dim {difficulty_dim} out of u8 range")
    if ego_poses.shape[1] != pose_dim:
        raise ValueError(f"ego_poses dim {ego_poses.shape[1]} != {pose_dim}")
    if difficulties.shape[1] != difficulty_dim:
        raise ValueError(f"difficulties dim {difficulties.shape[1]} != {difficulty_dim}")
    if ego_poses.shape[0] != latents.shape[0]:
        raise ValueError("ego_poses num_pairs != latents num_pairs")
    if difficulties.shape[0] != latents.shape[0]:
        raise ValueError("difficulties num_pairs != latents num_pairs")

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    q_pose, pose_scale, pose_zp = _quantize_int16(ego_poses)
    q_diff, diff_scale, diff_zp = _quantize_int16(difficulties)
    lat_bytes = q_lat.contiguous().numpy().tobytes()
    pose_bytes = q_pose.contiguous().numpy().tobytes()
    diff_bytes = q_diff.contiguous().numpy().tobytes()
    dec_blob = _serialize_state_dict(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_q["_pose_quant_scale"] = float(pose_scale)
    meta_q["_pose_quant_zero_point"] = float(pose_zp)
    meta_q["_diff_quant_scale"] = float(diff_scale)
    meta_q["_diff_quant_zero_point"] = float(diff_zp)
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        PIM1_HEADER_FMT, PIM1_MAGIC, schema_version, latent_dim, num_pairs,
        pose_dim, difficulty_dim, len(dec_blob), len(lat_bytes),
        len(pose_bytes), len(diff_bytes), len(meta_bytes),
    )
    return header + dec_blob + lat_bytes + pose_bytes + diff_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervIa3MultiArchive:
    if len(blob) < PIM1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, num_pairs, pose_dim, difficulty_dim,
        dec_len, lat_len, pose_len, diff_len, meta_len,
    ) = struct.unpack(PIM1_HEADER_FMT, blob[:PIM1_HEADER_SIZE])
    if magic != PIM1_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != PIM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2
    expected_pose = num_pairs * pose_dim * 2
    expected_diff = num_pairs * difficulty_dim * 2
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != {expected_lat}")
    if pose_len != expected_pose:
        raise ValueError(f"pose_len {pose_len} != {expected_pose}")
    if diff_len != expected_diff:
        raise ValueError(f"diff_len {diff_len} != {expected_diff}")
    end_hdr = PIM1_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_pose = end_lat + pose_len
    end_diff = end_pose + diff_len
    end_meta = end_diff + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")
    sd = _deserialize_state_dict(blob[end_hdr:end_dec])
    meta = json.loads(blob[end_diff:end_meta].decode("utf-8"))
    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    q_pose = torch.from_numpy(
        np.frombuffer(blob[end_lat:end_pose], dtype=np.int16).copy()
    ).view(num_pairs, pose_dim)
    q_diff = torch.from_numpy(
        np.frombuffer(blob[end_pose:end_diff], dtype=np.int16).copy()
    ).view(num_pairs, difficulty_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    pose_scale = float(meta.pop("_pose_quant_scale"))
    pose_zp = float(meta.pop("_pose_quant_zero_point"))
    diff_scale = float(meta.pop("_diff_quant_scale"))
    diff_zp = float(meta.pop("_diff_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)
    ego_poses = _dequant_int16(q_pose, pose_scale, pose_zp)
    difficulties = _dequant_int16(q_diff, diff_scale, diff_zp)
    return PactNervIa3MultiArchive(
        decoder_state_dict=sd,
        latents=latents,
        ego_poses=ego_poses,
        difficulties=difficulties,
        meta=meta,
        schema_version=int(version),
        pose_dim=int(pose_dim),
        difficulty_dim=int(difficulty_dim),
    )
