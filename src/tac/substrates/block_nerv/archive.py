# SPDX-License-Identifier: MIT
"""block_nerv archive grammar — monolithic single-file ``0.bin`` (BNV1).

L0 SKETCH archive grammar; Catalog #124 8-field declaration is the
``__init__`` module docstring. This file IS the export-first grammar (L2):

::

    MAGIC(4)              b"BNV1"  Block-NeRV Variant 1
    VERSION(1)            u8       schema version (currently 1)
    LATENT_DIM(2)         u16      cfg.latent_dim (e.g., 28)
    EMBED_DIM(2)          u16      cfg.embed_dim (e.g., 36)
    NUM_PAIRS(2)          u16      cfg.num_pairs (e.g., 600)
    DECODER_BLOB_LEN(4)   u32      brotli-compressed base-decoder state_dict
    LATENT_BLOB_LEN(4)    u32      raw int16 latents bytes len
    LORA_BIAS_BLOB_LEN(4) u32      raw int16 lora_latent_bias bytes len
    LORA_GAIN_BLOB_LEN(4) u32      raw int8 lora_embed_gain bytes len
    META_BLOB_LEN(4)      u32      utf-8 json meta bytes len
    DECODER_BLOB          ...      brotli(quality=9, mode=GENERIC) pickled state_dict
    LATENT_BLOB           ...      int16 latents row-major (num_pairs, latent_dim)
    LORA_BIAS_BLOB        ...      int16 lora_latent_bias row-major (num_pairs, latent_dim)
    LORA_GAIN_BLOB        ...      int8 lora_embed_gain row-major (num_pairs, embed_dim)
    META_BLOB             ...      json: {"sin_freq": ..., "decoder_channels": [...]}

Header layout: 31 bytes (vs SHV1's 21).

The base-decoder state_dict EXCLUDES the three per-pair tensors
(``latents``, ``lora_latent_bias``, ``lora_embed_gain``); those are stored
in their own quantized sections.

CLAUDE.md compliance: same as sane_hnerv.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch


BNV1_MAGIC: bytes = b"BNV1"
BNV1_SCHEMA_VERSION: int = 1

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + EMBED_DIM(2) +
#                NUM_PAIRS(2) + DECODER_LEN(4) + LATENT_LEN(4) +
#                LORA_BIAS_LEN(4) + LORA_GAIN_LEN(4) + META_LEN(4)
#                = 31 bytes
BNV1_HEADER_FMT: str = "<4sBHHHIIIII"
BNV1_HEADER_SIZE: int = struct.calcsize(BNV1_HEADER_FMT)
assert BNV1_HEADER_SIZE == 31, "header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class BlockNervArchive:
    """Parsed archive structure — the inflate-time data contract."""

    base_decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    """``(num_pairs, latent_dim)`` int16-then-dequantized base latents."""

    lora_latent_bias: torch.Tensor
    """``(num_pairs, latent_dim)`` int16-then-dequantized LoRA latent bias."""

    lora_embed_gain: torch.Tensor
    """``(num_pairs, embed_dim)`` int8-then-dequantized LoRA channel gain."""

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


def _quantize_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize a float tensor to int16. Returns (q, scale, zero_point).

    Degenerate (all-equal) case: q is filled with -32767 so that
    ``dequant(q) = (q + 32767) * scale + zero_point = 0 * scale + lo = lo``.
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -32767, dtype=torch.int16),
            1.0,
            lo,
        )
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_int16(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def _quantize_int8(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize a float tensor to int8. Returns (q, scale, zero_point).

    Degenerate (all-equal) case: q is filled with -127 so that
    ``dequant(q) = (q + 127) * scale + zero_point = 0 + lo = lo``.
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -127, dtype=torch.int8),
            1.0,
            lo,
        )
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (q_unsigned - 127.0).to(torch.int8)
    return (q, scale, lo)


def _dequantize_int8(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 127.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    base_decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    lora_latent_bias: torch.Tensor,
    lora_embed_gain: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = BNV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained base-decoder + latents + LoRA + meta into 0.bin bytes."""
    if schema_version != BNV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2 or lora_latent_bias.dim() != 2 or lora_embed_gain.dim() != 2:
        raise ValueError("latents, lora_latent_bias, lora_embed_gain must all be 2-D")

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if lora_latent_bias.shape != latents.shape:
        raise ValueError(
            f"lora_latent_bias shape {tuple(lora_latent_bias.shape)} != "
            f"latents shape {tuple(latents.shape)}"
        )
    embed_dim = int(lora_embed_gain.shape[1])
    if int(lora_embed_gain.shape[0]) != num_pairs:
        raise ValueError(
            f"lora_embed_gain num_pairs {int(lora_embed_gain.shape[0])} != "
            f"{num_pairs}"
        )
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if embed_dim <= 0 or embed_dim > 0xFFFF:
        raise ValueError(f"embed_dim {embed_dim} out of u16 range")

    q_latents, lat_scale, lat_zp = _quantize_int16(latents)
    q_bias, bias_scale, bias_zp = _quantize_int16(lora_latent_bias)
    q_gain, gain_scale, gain_zp = _quantize_int8(lora_embed_gain)

    latent_bytes = q_latents.contiguous().numpy().tobytes()
    bias_bytes = q_bias.contiguous().numpy().tobytes()
    gain_bytes = q_gain.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(base_decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_scale"] = float(lat_scale)
    meta_with_quant["_lat_zero_point"] = float(lat_zp)
    meta_with_quant["_bias_scale"] = float(bias_scale)
    meta_with_quant["_bias_zero_point"] = float(bias_zp)
    meta_with_quant["_gain_scale"] = float(gain_scale)
    meta_with_quant["_gain_zero_point"] = float(gain_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        BNV1_HEADER_FMT,
        BNV1_MAGIC,
        schema_version,
        latent_dim,
        embed_dim,
        num_pairs,
        len(decoder_blob),
        len(latent_bytes),
        len(bias_bytes),
        len(gain_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + bias_bytes + gain_bytes + meta_bytes


def parse_archive(blob: bytes) -> BlockNervArchive:
    """Parse 0.bin bytes back into base-decoder + latents + LoRA + meta."""
    if len(blob) < BNV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {BNV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        embed_dim,
        num_pairs,
        decoder_len,
        latent_len,
        bias_len,
        gain_len,
        meta_len,
    ) = struct.unpack(BNV1_HEADER_FMT, blob[:BNV1_HEADER_SIZE])
    if magic != BNV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {BNV1_MAGIC!r})")
    if version != BNV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    expected_bias_bytes = num_pairs * latent_dim * 2
    expected_gain_bytes = num_pairs * embed_dim
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != expected {expected_latent_bytes}"
        )
    if bias_len != expected_bias_bytes:
        raise ValueError(
            f"lora_bias_len {bias_len} != expected {expected_bias_bytes}"
        )
    if gain_len != expected_gain_bytes:
        raise ValueError(
            f"lora_gain_len {gain_len} != expected {expected_gain_bytes}"
        )

    end_header = BNV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_bias = end_latents + bias_len
    end_gain = end_bias + gain_len
    end_meta = end_gain + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    bias_blob = blob[end_latents:end_bias]
    gain_blob = blob[end_bias:end_gain]
    meta_blob = blob[end_gain:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import per sane_hnerv pattern
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    q_bias = torch.from_numpy(
        np.frombuffer(bias_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    q_gain = torch.from_numpy(
        np.frombuffer(gain_blob, dtype=np.int8).copy()
    ).view(num_pairs, embed_dim)

    lat_scale = float(meta.pop("_lat_scale"))
    lat_zp = float(meta.pop("_lat_zero_point"))
    bias_scale = float(meta.pop("_bias_scale"))
    bias_zp = float(meta.pop("_bias_zero_point"))
    gain_scale = float(meta.pop("_gain_scale"))
    gain_zp = float(meta.pop("_gain_zero_point"))

    latents = _dequantize_int16(q_latents, lat_scale, lat_zp)
    lora_latent_bias = _dequantize_int16(q_bias, bias_scale, bias_zp)
    lora_embed_gain = _dequantize_int8(q_gain, gain_scale, gain_zp)

    return BlockNervArchive(
        base_decoder_state_dict=sd,
        latents=latents,
        lora_latent_bias=lora_latent_bias,
        lora_embed_gain=lora_embed_gain,
        meta=meta,
        schema_version=int(version),
    )
