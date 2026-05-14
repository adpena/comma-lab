"""Z5 predictive-coding world-model archive grammar — Z5PCWM1 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (≤200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

Z5PCWM1 extends Z4CR1 by adding a predictor section and replacing the
per-pair-latent section with (latent_init + per-pair-residuals + ego_motion).

Grammar:

::

    MAGIC(4)             b"Z5WM"
    VERSION(1)           u8       schema version (currently 1)
    LATENT_DIM(2)        u16      cfg.latent_dim (e.g. 24)
    EGO_MOTION_DIM(2)    u16      cfg.predictor_ego_motion_dim (e.g. 8)
    NUM_PAIRS(2)         u16      cfg.num_pairs (e.g. 600)
    ENCODER_BLOB_LEN(4)  u32      brotli-compressed encoder state_dict bytes len
    DECODER_BLOB_LEN(4)  u32      brotli-compressed decoder state_dict bytes len
    PREDICTOR_BLOB_LEN(4)u32      brotli-compressed predictor state_dict bytes len
    LATENT_INIT_LEN(4)   u32      int8 z_0 bytes (= latent_dim)
    RESIDUALS_LEN(4)     u32      int8 residuals bytes (= num_pairs * latent_dim)
    EGO_MOTION_LEN(4)    u32      int8 ego_motion bytes (= num_pairs * ego_motion_dim)
    META_BLOB_LEN(4)     u32      sorted-keys JSON utf-8 bytes len
    ENCODER_BLOB         ...      brotli(quality=9) encoder state_dict fp16
    DECODER_BLOB         ...      brotli(quality=9) decoder state_dict fp16
    PREDICTOR_BLOB       ...      brotli(quality=9) predictor state_dict fp16
    LATENT_INIT_BLOB     ...      int8 z_0 row-major (latent_dim,)
    RESIDUALS_BLOB       ...      int8 residuals row-major (num_pairs, latent_dim)
    EGO_MOTION_BLOB      ...      int8 ego_motion row-major (num_pairs, ego_motion_dim)
    META_BLOB            ...      sorted-keys JSON with predictive_coding_world_model_meta tag

Round-trip contract:

    bytes → parse_archive → (encoder_sd, decoder_sd, predictor_sd, latent_init,
                              residuals, ego_motion, meta)
    inverse → pack_archive → bytes

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli;
  fp16 state_dict cast on CPU)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

Z5PCWM1_MAGIC: bytes = b"Z5WM"
"""Z5 predictive-coding world-model variant 1 archive magic."""

Z5PCWM1_SCHEMA_VERSION: int = 1
"""Schema version byte."""

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + EGO_MOTION_DIM(2)
#                + NUM_PAIRS(2) + 7 × u32 section lengths
# = 4+1+2+2+2+7*4 = 39 bytes
Z5PCWM1_HEADER_FMT: str = "<4sBHHHIIIIIII"
Z5PCWM1_HEADER_SIZE: int = struct.calcsize(Z5PCWM1_HEADER_FMT)
assert Z5PCWM1_HEADER_SIZE == 39, f"header size invariant: expected 39, got {Z5PCWM1_HEADER_SIZE}"

_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PredictiveCodingArchive:
    """Parsed Z5PCWM1 archive — the inflate-time data contract."""

    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]
    predictor_state_dict: dict[str, torch.Tensor]
    latent_init: torch.Tensor  # (latent_dim,) float32 dequantized
    residuals: torch.Tensor    # (num_pairs, latent_dim) float32 dequantized
    ego_motion: torch.Tensor   # (num_pairs, ego_motion_dim) float32 dequantized
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize state_dict deterministically (matches Z4 pattern)."""
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} too long for u16")
        shape = tuple(int(s) for s in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(header_fmt, len(key_bytes), key_bytes, len(shape), *shape)
        parts.append(header)
        parts.append(tensor.numpy().tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
    pos = 0
    import numpy as np

    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + ndim * 4 > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + ndim * 4])
        pos += ndim * 4
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor_bytes = numel * 2
        if pos + tensor_bytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        arr = np.frombuffer(
            raw[pos : pos + tensor_bytes], dtype=np.float16
        ).reshape(shape).astype(np.float16, copy=True)
        sd[key] = torch.from_numpy(arr)
        pos += tensor_bytes
    if pos != len(raw):
        raise ValueError(f"state_dict blob has trailing bytes (pos={pos} len={len(raw)})")
    return sd


def _quantize_to_int8(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize a float tensor to int8 via min/max range.

    Per Catalog #161 the degenerate-range case (hi <= lo) clamps to -127
    so the decoded value is exactly ``lo``.
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return torch.full_like(f, -127, dtype=torch.int8), 1.0, lo
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (q_unsigned - 127.0).to(torch.int8)
    return q, scale, lo


def _dequantize_from_int8(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 127.0
    return q_unsigned * float(scale) + float(zero_point)


def _make_predictive_coding_world_model_meta_block(
    base_meta: dict[str, object],
    *,
    lambda_residual_entropy: float,
    predictor_num_layers: int,
    identity_predictor: bool,
) -> dict[str, object]:
    """Inject the predictive-coding-world-model provenance tag."""
    meta = dict(base_meta)
    meta["predictive_coding_world_model_meta"] = {
        "lambda_residual_entropy": float(lambda_residual_entropy),
        "predictor_num_layers": int(predictor_num_layers),
        "identity_predictor": bool(identity_predictor),
        "literature_anchor": "Rao-Ballard1999",
        "staircase_step": 3,
        "predicted_band_lo": 0.155,
        "predicted_band_hi": 0.180,
    }
    return meta


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    predictor_state_dict: dict[str, torch.Tensor],
    latent_init: torch.Tensor,
    residuals: torch.Tensor,
    ego_motion: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = Z5PCWM1_SCHEMA_VERSION,
    lambda_residual_entropy: float = 1.0,
    predictor_num_layers: int = 2,
    identity_predictor: bool = False,
) -> bytes:
    """Serialize trained substrate into Z5PCWM1 0.bin bytes."""
    if schema_version != Z5PCWM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latent_init.dim() != 1:
        raise ValueError(f"latent_init must be 1-D (latent_dim,); got {tuple(latent_init.shape)}")
    if residuals.dim() != 2:
        raise ValueError(f"residuals must be 2-D; got {tuple(residuals.shape)}")
    if ego_motion.dim() != 2:
        raise ValueError(f"ego_motion must be 2-D; got {tuple(ego_motion.shape)}")

    latent_dim = int(latent_init.shape[0])
    num_pairs = int(residuals.shape[0])
    ego_motion_dim = int(ego_motion.shape[1])
    if int(residuals.shape[1]) != latent_dim:
        raise ValueError(
            f"residuals second dim {residuals.shape[1]} != latent_dim {latent_dim}"
        )
    if int(ego_motion.shape[0]) != num_pairs:
        raise ValueError(
            f"ego_motion first dim {ego_motion.shape[0]} != num_pairs {num_pairs}"
        )
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if ego_motion_dim <= 0 or ego_motion_dim > 0xFFFF:
        raise ValueError(f"ego_motion_dim {ego_motion_dim} out of u16 range")

    q_latent_init, scale_li, zp_li = _quantize_to_int8(latent_init)
    q_residuals, scale_r, zp_r = _quantize_to_int8(residuals)
    q_ego, scale_e, zp_e = _quantize_to_int8(ego_motion)
    latent_init_bytes = q_latent_init.contiguous().numpy().tobytes()
    residuals_bytes = q_residuals.contiguous().numpy().tobytes()
    ego_motion_bytes = q_ego.contiguous().numpy().tobytes()

    encoder_blob = _serialize_state_dict(encoder_state_dict)
    decoder_blob = _serialize_state_dict(decoder_state_dict)
    predictor_blob = _serialize_state_dict(predictor_state_dict)

    meta_with_tag = _make_predictive_coding_world_model_meta_block(
        meta,
        lambda_residual_entropy=lambda_residual_entropy,
        predictor_num_layers=predictor_num_layers,
        identity_predictor=identity_predictor,
    )
    meta_with_tag["_latent_init_scale"] = float(scale_li)
    meta_with_tag["_latent_init_zp"] = float(zp_li)
    meta_with_tag["_residuals_scale"] = float(scale_r)
    meta_with_tag["_residuals_zp"] = float(zp_r)
    meta_with_tag["_ego_motion_scale"] = float(scale_e)
    meta_with_tag["_ego_motion_zp"] = float(zp_e)
    meta_bytes = json.dumps(
        meta_with_tag, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z5PCWM1_HEADER_FMT,
        Z5PCWM1_MAGIC,
        schema_version,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        len(encoder_blob),
        len(decoder_blob),
        len(predictor_blob),
        len(latent_init_bytes),
        len(residuals_bytes),
        len(ego_motion_bytes),
        len(meta_bytes),
    )
    return (
        header
        + encoder_blob
        + decoder_blob
        + predictor_blob
        + latent_init_bytes
        + residuals_bytes
        + ego_motion_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> PredictiveCodingArchive:
    """Parse Z5PCWM1 0.bin bytes back into all sections."""
    if len(blob) < Z5PCWM1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {Z5PCWM1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        ego_motion_dim,
        num_pairs,
        encoder_len,
        decoder_len,
        predictor_len,
        latent_init_len,
        residuals_len,
        ego_motion_len,
        meta_len,
    ) = struct.unpack(Z5PCWM1_HEADER_FMT, blob[:Z5PCWM1_HEADER_SIZE])
    if magic != Z5PCWM1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {Z5PCWM1_MAGIC!r})")
    if version != Z5PCWM1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    if latent_init_len != latent_dim:
        raise ValueError(
            f"latent_init_len {latent_init_len} != latent_dim {latent_dim}"
        )
    expected_residuals = num_pairs * latent_dim
    if residuals_len != expected_residuals:
        raise ValueError(
            f"residuals_len {residuals_len} != num_pairs*latent_dim = {expected_residuals}"
        )
    expected_ego = num_pairs * ego_motion_dim
    if ego_motion_len != expected_ego:
        raise ValueError(
            f"ego_motion_len {ego_motion_len} != num_pairs*ego_motion_dim = {expected_ego}"
        )

    pos = Z5PCWM1_HEADER_SIZE
    end_encoder = pos + encoder_len
    end_decoder = end_encoder + decoder_len
    end_predictor = end_decoder + predictor_len
    end_latent_init = end_predictor + latent_init_len
    end_residuals = end_latent_init + residuals_len
    end_ego = end_residuals + ego_motion_len
    end_meta = end_ego + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_blob = blob[pos:end_encoder]
    decoder_blob = blob[end_encoder:end_decoder]
    predictor_blob = blob[end_decoder:end_predictor]
    latent_init_blob = blob[end_predictor:end_latent_init]
    residuals_blob = blob[end_latent_init:end_residuals]
    ego_motion_blob = blob[end_residuals:end_ego]
    meta_blob = blob[end_ego:end_meta]

    encoder_sd = _deserialize_state_dict(encoder_blob)
    decoder_sd = _deserialize_state_dict(decoder_blob)
    predictor_sd = _deserialize_state_dict(predictor_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np

    q_li = torch.from_numpy(
        np.frombuffer(latent_init_blob, dtype=np.int8).copy()
    ).view(latent_dim)
    q_r = torch.from_numpy(
        np.frombuffer(residuals_blob, dtype=np.int8).copy()
    ).view(num_pairs, latent_dim)
    q_e = torch.from_numpy(
        np.frombuffer(ego_motion_blob, dtype=np.int8).copy()
    ).view(num_pairs, ego_motion_dim)

    scale_li = float(meta.pop("_latent_init_scale"))
    zp_li = float(meta.pop("_latent_init_zp"))
    scale_r = float(meta.pop("_residuals_scale"))
    zp_r = float(meta.pop("_residuals_zp"))
    scale_e = float(meta.pop("_ego_motion_scale"))
    zp_e = float(meta.pop("_ego_motion_zp"))

    latent_init = _dequantize_from_int8(q_li, scale_li, zp_li)
    residuals = _dequantize_from_int8(q_r, scale_r, zp_r)
    ego_motion = _dequantize_from_int8(q_e, scale_e, zp_e)

    return PredictiveCodingArchive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        predictor_state_dict=predictor_sd,
        latent_init=latent_init,
        residuals=residuals,
        ego_motion=ego_motion,
        meta=meta,
        schema_version=int(version),
    )


__all__ = [
    "PredictiveCodingArchive",
    "Z5PCWM1_HEADER_FMT",
    "Z5PCWM1_HEADER_SIZE",
    "Z5PCWM1_MAGIC",
    "Z5PCWM1_SCHEMA_VERSION",
    "pack_archive",
    "parse_archive",
]
