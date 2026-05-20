# SPDX-License-Identifier: MIT
"""pact_nerv_cross_codec_a archive grammar — CC_A monolithic single-file 0.bin.

Header (30 bytes):
    MAGIC(4)                  b"CCAA"
    VERSION(1)                u8
    LATENT_DIM(2)             u16
    NUM_PAIRS(2)              u16
    PALETTE_SIZE(1)           u8     FEC6 k=16
    FEC6_BASE_BLOB_LEN(4)     u32    fec6 base codec bytes (placeholder at L0)
    DECODER_BLOB_LEN(4)       u32    Pact-NeRV side-info decoder bytes
    LATENT_BLOB_LEN(4)        u32    Pact-NeRV per-pair latents
    SELECTOR_BLOB_LEN(4)      u32    Per-pair fec6 selectors (u8)
    META_BLOB_LEN(4)          u32    utf-8 JSON meta (composition_alpha + cfg)

Catalog #139 no-op-detector planning:
    Both fec6_base_blob AND pact_nerv_decoder_blob must be byte-mutation-
    sensitive (changing either bytes class must change rendered output).
    At L0 SCAFFOLD this is verified by the test pack-then-parse round-trip
    invariant.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

CC_A_MAGIC: bytes = b"CCAA"
CC_A_SCHEMA_VERSION: int = 1
CC_A_HEADER_FMT: str = "<4sBHHBIIIII"
CC_A_HEADER_SIZE: int = struct.calcsize(CC_A_HEADER_FMT)
assert CC_A_HEADER_SIZE == 30
BROTLI_QUALITY = 9


@dataclass(frozen=True)
class PactNervCrossCodecAArchive:
    fec6_base_bytes: bytes
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
        if k != "selectors" and not k.startswith("base_codec.")
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
    fec6_base_bytes: bytes,
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    selector_bytes: bytes,
    meta: dict[str, object],
    *,
    palette_size: int,
    schema_version: int = CC_A_SCHEMA_VERSION,
) -> bytes:
    if schema_version != CC_A_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if palette_size < 0 or palette_size > 255:
        raise ValueError(f"palette_size {palette_size} out of u8 range")
    if not isinstance(fec6_base_bytes, (bytes, bytearray)):
        raise ValueError("fec6_base_bytes must be bytes")
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
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        CC_A_HEADER_FMT, CC_A_MAGIC, schema_version, latent_dim, num_pairs,
        palette_size, len(fec6_base_bytes), len(dec_blob), len(lat_bytes),
        len(selector_bytes), len(meta_bytes),
    )
    return (
        header
        + bytes(fec6_base_bytes)
        + dec_blob
        + lat_bytes
        + bytes(selector_bytes)
        + meta_bytes
    )


def parse_archive(blob: bytes) -> PactNervCrossCodecAArchive:
    if len(blob) < CC_A_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, num_pairs, palette_size,
        fec6_len, dec_len, lat_len, sel_len, meta_len,
    ) = struct.unpack(CC_A_HEADER_FMT, blob[:CC_A_HEADER_SIZE])
    if magic != CC_A_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != CC_A_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat = num_pairs * latent_dim * 2
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != {expected_lat}")
    end_hdr = CC_A_HEADER_SIZE
    end_fec6 = end_hdr + fec6_len
    end_dec = end_fec6 + dec_len
    end_lat = end_dec + lat_len
    end_sel = end_lat + sel_len
    end_meta = end_sel + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")
    fec6_base = bytes(blob[end_hdr:end_fec6])
    sd = _deserialize_state_dict(blob[end_fec6:end_dec])
    meta = json.loads(blob[end_sel:end_meta].decode("utf-8"))
    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)
    return PactNervCrossCodecAArchive(
        fec6_base_bytes=fec6_base,
        decoder_state_dict=sd,
        latents=latents,
        selector_bytes=bytes(blob[end_lat:end_sel]),
        meta=meta,
        schema_version=int(version),
        palette_size=int(palette_size),
    )
