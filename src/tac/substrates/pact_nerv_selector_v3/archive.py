# SPDX-License-Identifier: MIT
"""pact_nerv_selector_v3 archive grammar - PSV3 monolithic single-file 0.bin.

Header (26 bytes):
    MAGIC(4)               b"PSV3"
    VERSION(1)             u8
    LATENT_DIM(2)          u16
    NUM_PAIRS(2)           u16
    PALETTE_SIZE(1)        u8     FEC6 k=16
    DECODER_BLOB_LEN(4)    u32
    LATENT_BLOB_LEN(4)     u32
    SELECTOR_BLOB_LEN(4)   u32    Rice-Golomb coded
    META_BLOB_LEN(4)       u32    meta includes "rice_golomb_k"
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

PSV3_MAGIC: bytes = b"PSV3"
PSV3_SCHEMA_VERSION: int = 1
PSV3_HEADER_FMT: str = "<4sBHHBIIII"
PSV3_HEADER_SIZE: int = struct.calcsize(PSV3_HEADER_FMT)
assert PSV3_HEADER_SIZE == 26
BROTLI_QUALITY = 9
DECODER_QUANT_FP16_BROTLI_Q9 = "fp16_brotli_q9"
DECODER_QUANT_FP16_BROTLI_Q11 = "fp16_brotli_q11"
DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11 = "int8_per_channel_brotli_q11"
DECODER_QUANTIZATION_KINDS = frozenset(
    {
        DECODER_QUANT_FP16_BROTLI_Q9,
        DECODER_QUANT_FP16_BROTLI_Q11,
        DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11,
    }
)


@dataclass(frozen=True)
class PactNervSelectorV3Archive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    selector_bytes: bytes
    meta: dict[str, object]
    schema_version: int
    palette_size: int


def _quantize_decoder_state_dict_int8_per_channel(
    sd: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Quantize decoder tensors with symmetric int8 + per-output-channel scales."""

    q_state: dict[str, object] = {}
    for name, tensor in sd.items():
        if name == "selectors":
            continue
        t = tensor.detach().to("cpu")
        if not torch.is_floating_point(t):
            q_state[name] = t.contiguous().numpy()
            continue
        f = t.to(dtype=torch.float32).contiguous()
        if f.ndim >= 2:
            channels = int(f.shape[0])
            flat = f.reshape(channels, -1)
            scales = flat.abs().amax(dim=1) / 127.0
            scales = torch.where(scales < 1e-10, torch.ones_like(scales), scales)
            scale_view = scales.view(channels, *([1] * (f.ndim - 1)))
            q = (f / scale_view).round().clamp(-128, 127).to(torch.int8)
            q_state[name] = {
                "q": q.contiguous().numpy(),
                "scale": scales.to(dtype=torch.float32).contiguous().numpy(),
                "per_channel": True,
            }
        else:
            scale = f.abs().max() / 127.0
            if float(scale) < 1e-10:
                scale = torch.tensor(1.0, dtype=torch.float32)
            q = (f / scale).round().clamp(-128, 127).to(torch.int8)
            q_state[name] = {
                "q": q.contiguous().numpy(),
                "scale": np.asarray(scale.to(dtype=torch.float32).item(), dtype=np.float32),
                "per_channel": False,
            }
    return q_state


def _dequantize_decoder_state_dict_int8_per_channel(
    q_state: dict[str, object],
) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for name, record in q_state.items():
        if isinstance(record, torch.Tensor):
            sd[name] = record
            continue
        if isinstance(record, np.ndarray):
            sd[name] = torch.from_numpy(record.copy())
            continue
        if not isinstance(record, dict):
            raise ValueError(f"bad int8 decoder record for {name!r}")
        q = record.get("q")
        scale = record.get("scale")
        if q is None or scale is None:
            raise ValueError(f"bad int8 decoder tensors for {name!r}")
        qf = torch.from_numpy(np.asarray(q).copy()).to(dtype=torch.float32)
        sf = torch.from_numpy(np.asarray(scale).copy()).to(dtype=torch.float32)
        if bool(record.get("per_channel", False)):
            if qf.ndim < 2:
                raise ValueError(f"per-channel record {name!r} has ndim={qf.ndim}")
            view_shape = [int(sf.shape[0])] + [1] * (qf.ndim - 1)
            sd[name] = qf * sf.view(*view_shape)
        else:
            sd[name] = qf * sf
    return sd


def _serialize_state_dict(
    sd: dict[str, torch.Tensor],
    *,
    decoder_quantization: str = DECODER_QUANT_FP16_BROTLI_Q9,
) -> bytes:
    if decoder_quantization not in DECODER_QUANTIZATION_KINDS:
        raise ValueError(
            f"unsupported decoder_quantization {decoder_quantization!r}; "
            f"expected one of {sorted(DECODER_QUANTIZATION_KINDS)}"
        )
    buf = io.BytesIO()
    if decoder_quantization == DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11:
        payload: object = {
            "__pact_decoder_quantization__": {
                "kind": DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11,
                "scale_axis": 0,
                "scale_dtype": "float32",
            },
            "state": _quantize_decoder_state_dict_int8_per_channel(sd),
        }
        quality = 11
    else:
        payload = {
            k: v.detach().to("cpu", dtype=torch.float16).contiguous()
            for k, v in sd.items()
            if k != "selectors"
        }
        quality = (
            11 if decoder_quantization == DECODER_QUANT_FP16_BROTLI_Q11
            else BROTLI_QUALITY
        )
    pickle.dump(payload, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=quality))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    sd = pickle.loads(brotli.decompress(blob))
    if not isinstance(sd, dict):
        raise ValueError("decoder_state_dict blob did not unpickle to a dict")
    quant_meta = sd.get("__pact_decoder_quantization__")
    if isinstance(quant_meta, dict):
        kind = quant_meta.get("kind")
        if kind != DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11:
            raise ValueError(f"unsupported decoder quantization kind: {kind!r}")
        q_state = sd.get("state")
        if not isinstance(q_state, dict):
            raise ValueError("int8 decoder payload missing state dict")
        return _dequantize_decoder_state_dict_int8_per_channel(q_state)
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
    selector_bytes: bytes,
    meta: dict[str, object],
    *,
    palette_size: int,
    schema_version: int = PSV3_SCHEMA_VERSION,
    decoder_quantization: str = DECODER_QUANT_FP16_BROTLI_Q9,
) -> bytes:
    if schema_version != PSV3_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if palette_size < 0 or palette_size > 255:
        raise ValueError(f"palette_size {palette_size} out of u8 range")
    if not isinstance(selector_bytes, (bytes, bytearray)):
        raise ValueError("selector_bytes must be bytes")
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    lat_bytes = q_lat.contiguous().numpy().tobytes()
    dec_blob = _serialize_state_dict(
        decoder_state_dict, decoder_quantization=decoder_quantization
    )

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_q["decoder_quantization"] = decoder_quantization
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        PSV3_HEADER_FMT, PSV3_MAGIC, schema_version, latent_dim, num_pairs,
        palette_size, len(dec_blob), len(lat_bytes), len(selector_bytes), len(meta_bytes),
    )
    return header + dec_blob + lat_bytes + bytes(selector_bytes) + meta_bytes


def parse_archive(blob: bytes) -> PactNervSelectorV3Archive:
    if len(blob) < PSV3_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, num_pairs, palette_size,
        dec_len, lat_len, sel_len, meta_len,
    ) = struct.unpack(PSV3_HEADER_FMT, blob[:PSV3_HEADER_SIZE])
    if magic != PSV3_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != PSV3_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != {expected_lat}")
    end_hdr = PSV3_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_sel = end_lat + sel_len
    end_meta = end_sel + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")
    sd = _deserialize_state_dict(blob[end_hdr:end_dec])
    meta = json.loads(blob[end_sel:end_meta].decode("utf-8"))
    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)
    return PactNervSelectorV3Archive(
        decoder_state_dict=sd,
        latents=latents,
        selector_bytes=bytes(blob[end_lat:end_sel]),
        meta=meta,
        schema_version=int(version),
        palette_size=int(palette_size),
    )
