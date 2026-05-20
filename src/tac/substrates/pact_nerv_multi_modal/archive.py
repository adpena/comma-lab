# SPDX-License-Identifier: MIT
"""pact_nerv_multi_modal archive grammar — monolithic 0.bin (PMM).

::

    MAGIC(4)              b"PMM\\x00"
    VERSION(1)            u8
    LATENT_DIM(2)         u16
    NUM_PAIRS(2)          u16
    POSE_DIM(1)           u8
    CLASS_PRIOR_DIM(1)    u8
    ODOMETRY_DIM(1)       u8
    DECODER_BLOB_LEN(4)   u32
    LATENT_BLOB_LEN(4)    u32
    POSE_BLOB_LEN(4)      u32
    CLASS_PRIOR_BLOB_LEN(4) u32
    ODOMETRY_BLOB_LEN(4)  u32
    META_BLOB_LEN(4)      u32

Header: 4+1+2+2+1+1+1+4+4+4+4+4+4 = 36 bytes. POSE_DIM + CLASS_PRIOR_DIM +
ODOMETRY_DIM are u8 header fields so the inflate-time consumer reconstructs
the fusion-tower dimensionalities exactly.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PMM_MAGIC: bytes = b"PMM\x00"
PMM_SCHEMA_VERSION: int = 1

PMM_HEADER_FMT: str = "<4sBHHBBBIIIIII"
PMM_HEADER_SIZE: int = struct.calcsize(PMM_HEADER_FMT)
assert PMM_HEADER_SIZE == 36, "PMM header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervMultiModalArchive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    pose: torch.Tensor
    class_prior: torch.Tensor
    odometry: torch.Tensor
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
    pose: torch.Tensor,
    class_prior: torch.Tensor,
    odometry: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = PMM_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PMM_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    for name, t in (
        ("latents", latents), ("pose", pose),
        ("class_prior", class_prior), ("odometry", odometry),
    ):
        if t.dim() != 2:
            raise ValueError(f"{name} must be 2-D; got {tuple(t.shape)}")

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    pose_dim = int(pose.shape[1])
    class_prior_dim = int(class_prior.shape[1])
    odometry_dim = int(odometry.shape[1])

    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    for d_name, d_val in (
        ("pose_dim", pose_dim), ("class_prior_dim", class_prior_dim),
        ("odometry_dim", odometry_dim),
    ):
        if d_val <= 0 or d_val > 0xFF:
            raise ValueError(f"{d_name} {d_val} out of u8 range")

    if pose.shape[0] != num_pairs or class_prior.shape[0] != num_pairs or odometry.shape[0] != num_pairs:
        raise ValueError(
            "pose / class_prior / odometry batch must match latents num_pairs"
        )

    q_lat, lat_s, lat_z = _quantize_tensor_to_int16(latents)
    q_pose, pose_s, pose_z = _quantize_tensor_to_int16(pose)
    q_cls, cls_s, cls_z = _quantize_tensor_to_int16(class_prior)
    q_odo, odo_s, odo_z = _quantize_tensor_to_int16(odometry)

    latent_bytes = q_lat.contiguous().numpy().tobytes()
    pose_bytes = q_pose.contiguous().numpy().tobytes()
    class_bytes = q_cls.contiguous().numpy().tobytes()
    odometry_bytes = q_odo.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_s)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_z)
    meta_with_quant["_pose_quant_scale"] = float(pose_s)
    meta_with_quant["_pose_quant_zero_point"] = float(pose_z)
    meta_with_quant["_class_quant_scale"] = float(cls_s)
    meta_with_quant["_class_quant_zero_point"] = float(cls_z)
    meta_with_quant["_odo_quant_scale"] = float(odo_s)
    meta_with_quant["_odo_quant_zero_point"] = float(odo_z)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PMM_HEADER_FMT,
        PMM_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        pose_dim,
        class_prior_dim,
        odometry_dim,
        len(decoder_blob),
        len(latent_bytes),
        len(pose_bytes),
        len(class_bytes),
        len(odometry_bytes),
        len(meta_bytes),
    )
    return (
        header + decoder_blob + latent_bytes + pose_bytes
        + class_bytes + odometry_bytes + meta_bytes
    )


def parse_archive(blob: bytes) -> PactNervMultiModalArchive:
    if len(blob) < PMM_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PMM_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        pose_dim,
        class_prior_dim,
        odometry_dim,
        decoder_len,
        latent_len,
        pose_len,
        class_len,
        odo_len,
        meta_len,
    ) = struct.unpack(PMM_HEADER_FMT, blob[:PMM_HEADER_SIZE])
    if magic != PMM_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PMM_MAGIC!r})")
    if version != PMM_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent = num_pairs * latent_dim * 2
    expected_pose = num_pairs * pose_dim * 2
    expected_class = num_pairs * class_prior_dim * 2
    expected_odo = num_pairs * odometry_dim * 2
    if latent_len != expected_latent:
        raise ValueError(f"latent_len mismatch: {latent_len} != {expected_latent}")
    if pose_len != expected_pose:
        raise ValueError(f"pose_len mismatch: {pose_len} != {expected_pose}")
    if class_len != expected_class:
        raise ValueError(f"class_len mismatch: {class_len} != {expected_class}")
    if odo_len != expected_odo:
        raise ValueError(f"odo_len mismatch: {odo_len} != {expected_odo}")

    end_header = PMM_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_pose = end_latents + pose_len
    end_class = end_pose + class_len
    end_odo = end_class + odo_len
    end_meta = end_odo + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    pose_blob = blob[end_latents:end_pose]
    class_blob = blob[end_pose:end_class]
    odo_blob = blob[end_class:end_odo]
    meta_blob = blob[end_odo:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_lat = torch.from_numpy(np.frombuffer(latent_blob, dtype=np.int16).copy()).view(num_pairs, latent_dim)
    q_pose = torch.from_numpy(np.frombuffer(pose_blob, dtype=np.int16).copy()).view(num_pairs, pose_dim)
    q_cls = torch.from_numpy(np.frombuffer(class_blob, dtype=np.int16).copy()).view(num_pairs, class_prior_dim)
    q_odo = torch.from_numpy(np.frombuffer(odo_blob, dtype=np.int16).copy()).view(num_pairs, odometry_dim)

    lat_s = float(meta.pop("_lat_quant_scale")); lat_z = float(meta.pop("_lat_quant_zero_point"))
    pose_s = float(meta.pop("_pose_quant_scale")); pose_z = float(meta.pop("_pose_quant_zero_point"))
    cls_s = float(meta.pop("_class_quant_scale")); cls_z = float(meta.pop("_class_quant_zero_point"))
    odo_s = float(meta.pop("_odo_quant_scale")); odo_z = float(meta.pop("_odo_quant_zero_point"))

    latents = _dequantize_tensor(q_lat, lat_s, lat_z)
    pose = _dequantize_tensor(q_pose, pose_s, pose_z)
    class_prior = _dequantize_tensor(q_cls, cls_s, cls_z)
    odometry = _dequantize_tensor(q_odo, odo_s, odo_z)

    return PactNervMultiModalArchive(
        decoder_state_dict=sd,
        latents=latents,
        pose=pose,
        class_prior=class_prior,
        odometry=odometry,
        meta=meta,
        schema_version=int(version),
    )
