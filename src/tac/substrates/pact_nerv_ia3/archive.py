# SPDX-License-Identifier: MIT
"""pact_nerv_ia3 archive grammar — monolithic single-file ``0.bin`` (PIA3).

Catalog #124 STRICT archive-grammar 8 fields declared in package
``__init__``. Export-first grammar (HNeRV parity L2):

::

    MAGIC(4)              b"PIA3"  Pact-NeRV IA3
    VERSION(1)            u8       schema version (currently 1)
    LATENT_DIM(2)         u16      cfg.latent_dim
    NUM_PAIRS(2)          u16      cfg.num_pairs
    POSE_DIM(1)           u8       cfg.pose_dim (0..255; canonical 6)
    DECODER_BLOB_LEN(4)   u32      brotli base+IA3 decoder state_dict len
    LATENT_BLOB_LEN(4)    u32      raw int16 latents bytes len
    EGO_POSE_BLOB_LEN(4)  u32      raw int16 ego-pose bytes len
    META_BLOB_LEN(4)      u32      utf-8 json meta bytes len

    DECODER_BLOB          ...      brotli(quality=9) of pickled state_dict
                                   (includes base decoder + IA3 γ_proj heads)
    LATENT_BLOB           ...      int16 latents row-major
    EGO_POSE_BLOB         ...      int16 ego_poses row-major
    META_BLOB             ...      json: {"sin_freq": ..., "decoder_channels": [...],
                                          "ia3_init_delta_std": ...}

Header: 4+1+2+2+1+4+4+4+4 = 26 bytes (POSE_DIM is the distinctive header
field vs sister BSV1 / DSV1 — operator-visible knob declaring the ego-pose
conditioning dimensionality the inflate-time consumer must use).

Catalog #124 parser-section manifest enumerates 6 logical sections:
HEADER + DECODER_BLOB + LATENT_BLOB + EGO_POSE_BLOB + META_BLOB + (implicit
"ia3_gamma_proj_heads" subset inside DECODER_BLOB).

Per HNeRV parity L4: the decoder_state_dict ships base + IA3 γ_proj weights
in a SINGLE brotli blob; the IA3 γ_proj heads are logically-grouped (so future
rate-axis work can quantize them independently per Catalog #303 cargo-cult
audit's "per-block modulation" alternative path).

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

PIA3_MAGIC: bytes = b"PIA3"
PIA3_SCHEMA_VERSION: int = 1

PIA3_HEADER_FMT: str = "<4sBHHBIIII"
PIA3_HEADER_SIZE: int = struct.calcsize(PIA3_HEADER_FMT)
assert PIA3_HEADER_SIZE == 26, "PIA3 header size invariant (1-byte pose-dim + 4-byte ego-pose-blob-len fields)"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervIa3Archive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (base decoder + IA3 γ_proj heads; all model weights
    except per-pair latents + ego_poses). The IA3 γ_proj head keys are
    logically-grouped as "ia3_gamma_proj_heads" per Catalog #124 manifest."""

    latents: torch.Tensor
    ego_poses: torch.Tensor
    meta: dict[str, object]
    schema_version: int
    pose_dim: int


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
        # FFFF Catalog #158 fix: -32767 fill so dequant = 0*scale + lo = lo
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
    ego_poses: torch.Tensor,
    meta: dict[str, object],
    *,
    pose_dim: int,
    schema_version: int = PIA3_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PIA3_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
    if ego_poses.dim() != 2:
        raise ValueError(
            f"ego_poses must be 2-D (num_pairs, pose_dim); got {tuple(ego_poses.shape)}"
        )
    if pose_dim < 0 or pose_dim > 255:
        raise ValueError(f"pose_dim {pose_dim} out of u8 range [0, 255]")
    if ego_poses.shape[1] != pose_dim:
        raise ValueError(
            f"ego_poses pose_dim {ego_poses.shape[1]} != declared pose_dim {pose_dim}"
        )
    if ego_poses.shape[0] != latents.shape[0]:
        raise ValueError(
            f"ego_poses num_pairs {ego_poses.shape[0]} != latents "
            f"num_pairs {latents.shape[0]}"
        )

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_latents, lat_scale, lat_zp = _quantize_tensor_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    q_pose, pose_scale, pose_zp = _quantize_tensor_to_int16(ego_poses)
    pose_bytes = q_pose.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_with_quant["_pose_quant_scale"] = float(pose_scale)
    meta_with_quant["_pose_quant_zero_point"] = float(pose_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PIA3_HEADER_FMT,
        PIA3_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        pose_dim,
        len(decoder_blob),
        len(latent_bytes),
        len(pose_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + pose_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervIa3Archive:
    if len(blob) < PIA3_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PIA3_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        pose_dim,
        decoder_len,
        latent_len,
        pose_len,
        meta_len,
    ) = struct.unpack(PIA3_HEADER_FMT, blob[:PIA3_HEADER_SIZE])
    if magic != PIA3_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PIA3_MAGIC!r})")
    if version != PIA3_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )
    expected_pose_bytes = num_pairs * pose_dim * 2
    if pose_len != expected_pose_bytes:
        raise ValueError(
            f"pose_len {pose_len} != num_pairs*pose_dim*2 = {expected_pose_bytes}"
        )

    end_header = PIA3_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_poses = end_latents + pose_len
    end_meta = end_poses + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    pose_blob = blob[end_latents:end_poses]
    meta_blob = blob[end_poses:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    q_poses = torch.from_numpy(
        np.frombuffer(pose_blob, dtype=np.int16).copy()
    ).view(num_pairs, pose_dim)

    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    pose_scale = float(meta.pop("_pose_quant_scale"))
    pose_zp = float(meta.pop("_pose_quant_zero_point"))
    latents = _dequantize_tensor(q_latents, lat_scale, lat_zp)
    ego_poses = _dequantize_tensor(q_poses, pose_scale, pose_zp)

    return PactNervIa3Archive(
        decoder_state_dict=sd,
        latents=latents,
        ego_poses=ego_poses,
        meta=meta,
        schema_version=int(version),
        pose_dim=int(pose_dim),
    )
