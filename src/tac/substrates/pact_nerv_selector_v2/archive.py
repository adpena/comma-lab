# SPDX-License-Identifier: MIT
"""pact_nerv_selector_v2 archive grammar - monolithic single-file ``0.bin`` (PSV2).

Catalog #124 STRICT archive-grammar 8 fields. Export-first grammar (HNeRV
parity L2):

::

    MAGIC(4)               b"PSV2"  Pact-NeRV Selector V2
    VERSION(1)             u8       schema version (currently 1)
    LATENT_DIM(2)          u16      cfg.latent_dim
    NUM_PAIRS(2)           u16      cfg.num_pairs
    PALETTE_SIZE(1)        u8       cfg.selector_palette_size (FEC6 k=16)
    DECODER_BLOB_LEN(4)    u32      brotli base decoder state_dict len
    LATENT_BLOB_LEN(4)     u32      raw int16 latents bytes len
    SELECTOR_BLOB_LEN(4)   u32      arithmetic-coded selector bytes len
    META_BLOB_LEN(4)       u32      utf-8 json meta bytes len

    DECODER_BLOB           ...      brotli(quality=9) of pickled state_dict
    LATENT_BLOB            ...      int16 latents row-major
    SELECTOR_BLOB          ...      arithmetic-coded selector indices over
                                    FEC6 k=16 palette (Witten 1987 §3.2)
    META_BLOB              ...      json: {"cum_freq": [...], decoder_channels}

Header: 4+1+2+2+1+4+4+4+4 = 26 bytes.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PSV2_MAGIC: bytes = b"PSV2"
PSV2_SCHEMA_VERSION: int = 1

PSV2_HEADER_FMT: str = "<4sBHHBIIII"
PSV2_HEADER_SIZE: int = struct.calcsize(PSV2_HEADER_FMT)
assert PSV2_HEADER_SIZE == 26, "PSV2 header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervSelectorV2Archive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    selector_bytes: bytes
    meta: dict[str, object]
    schema_version: int
    palette_size: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
        if k != "selectors"  # buffer; transported via SELECTOR_BLOB
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
    schema_version: int = PSV2_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PSV2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
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
    dec_blob = _serialize_state_dict(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )

    header = struct.pack(
        PSV2_HEADER_FMT,
        PSV2_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        palette_size,
        len(dec_blob),
        len(lat_bytes),
        len(selector_bytes),
        len(meta_bytes),
    )
    return header + dec_blob + lat_bytes + bytes(selector_bytes) + meta_bytes


def parse_archive(blob: bytes) -> PactNervSelectorV2Archive:
    if len(blob) < PSV2_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PSV2_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        palette_size,
        dec_len,
        lat_len,
        sel_len,
        meta_len,
    ) = struct.unpack(PSV2_HEADER_FMT, blob[:PSV2_HEADER_SIZE])
    if magic != PSV2_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PSV2_MAGIC!r})")
    if version != PSV2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2
    if lat_len != expected_lat:
        raise ValueError(
            f"lat_len {lat_len} != num_pairs*latent_dim*2 = {expected_lat}"
        )

    end_hdr = PSV2_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_sel = end_lat + sel_len
    end_meta = end_sel + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )
    dec_blob = blob[end_hdr:end_dec]
    lat_blob = blob[end_dec:end_lat]
    sel_blob = blob[end_lat:end_sel]
    meta_blob = blob[end_sel:end_meta]
    sd = _deserialize_state_dict(dec_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(lat_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)

    return PactNervSelectorV2Archive(
        decoder_state_dict=sd,
        latents=latents,
        selector_bytes=bytes(sel_blob),
        meta=meta,
        schema_version=int(version),
        palette_size=int(palette_size),
    )
