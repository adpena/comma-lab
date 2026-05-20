# SPDX-License-Identifier: MIT
"""pact_nerv_neural_codec_e2e_cross archive grammar — NCEC monolithic single-file 0.bin.

Header (35 bytes):
    MAGIC(4)                  b"NCEC"
    VERSION(1)                u8
    LATENT_DIM_A(2)           u16
    LATENT_DIM_B(2)           u16
    NUM_PAIRS(2)              u16
    DECODER_A_BLOB_LEN(4)     u32    HNeRV branch A weights
    DECODER_B_BLOB_LEN(4)     u32    HNeRV branch B weights
    HYPERPRIOR_BLOB_LEN(4)    u32    Hyperprior gate weights
    LATENTS_A_BLOB_LEN(4)     u32    Per-pair latents A (int16)
    LATENTS_B_BLOB_LEN(4)     u32    Per-pair latents B (int16)
    META_BLOB_LEN(4)          u32    utf-8 JSON meta

Catalog #139 no-op-detector planning:
    All three weight blobs (decoder_a, decoder_b, hyperprior) must be
    byte-mutation-sensitive (changing any class must change rendered output).
    Hyperprior gate sensitivity is the canonical SUPER_ADDITIVE proof per
    Catalog #322: if gate bytes do NOT affect score, the cross-codec
    composition has degenerated to single-branch selection.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

NCEC_MAGIC: bytes = b"NCEC"
NCEC_SCHEMA_VERSION: int = 1
NCEC_HEADER_FMT: str = "<4sBHHHIIIIII"
NCEC_HEADER_SIZE: int = struct.calcsize(NCEC_HEADER_FMT)
assert NCEC_HEADER_SIZE == 35
BROTLI_QUALITY = 9


@dataclass(frozen=True)
class PactNervNeuralCodecE2ECrossArchive:
    decoder_a_state_dict: dict[str, torch.Tensor]
    decoder_b_state_dict: dict[str, torch.Tensor]
    hyperprior_state_dict: dict[str, torch.Tensor]
    latents_a: torch.Tensor
    latents_b: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_branch_sd(sd: dict[str, torch.Tensor], prefix: str) -> bytes:
    """Serialize one branch's state_dict (filtered by branch prefix)."""
    buf = io.BytesIO()
    sd_cpu = {
        k[len(prefix):]: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
        if k.startswith(prefix)
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _serialize_module_sd(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_sd(blob: bytes) -> dict[str, torch.Tensor]:
    sd = pickle.loads(brotli.decompress(blob))
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
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
    branch_a_state_dict: dict[str, torch.Tensor],
    branch_b_state_dict: dict[str, torch.Tensor],
    hyperprior_state_dict: dict[str, torch.Tensor],
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = NCEC_SCHEMA_VERSION,
) -> bytes:
    if schema_version != NCEC_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents_a.dim() != 2 or latents_b.dim() != 2:
        raise ValueError("latents must be 2-D")
    if latents_a.shape[0] != latents_b.shape[0]:
        raise ValueError(
            f"num_pairs mismatch: latents_a={latents_a.shape[0]} "
            f"latents_b={latents_b.shape[0]}"
        )
    num_pairs = int(latents_a.shape[0])
    latent_dim_a = int(latents_a.shape[1])
    latent_dim_b = int(latents_b.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim_a <= 0 or latent_dim_a > 0xFFFF:
        raise ValueError(f"latent_dim_a {latent_dim_a} out of u16 range")
    if latent_dim_b <= 0 or latent_dim_b > 0xFFFF:
        raise ValueError(f"latent_dim_b {latent_dim_b} out of u16 range")

    q_a, sa, za = _quantize_int16(latents_a)
    q_b, sb, zb = _quantize_int16(latents_b)
    lat_a_bytes = q_a.contiguous().numpy().tobytes()
    lat_b_bytes = q_b.contiguous().numpy().tobytes()
    dec_a_blob = _serialize_module_sd(branch_a_state_dict)
    dec_b_blob = _serialize_module_sd(branch_b_state_dict)
    hp_blob = _serialize_module_sd(hyperprior_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_a_quant_scale"] = float(sa)
    meta_q["_lat_a_quant_zero_point"] = float(za)
    meta_q["_lat_b_quant_scale"] = float(sb)
    meta_q["_lat_b_quant_zero_point"] = float(zb)
    meta_bytes = json.dumps(meta_q, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        NCEC_HEADER_FMT, NCEC_MAGIC, schema_version, latent_dim_a, latent_dim_b,
        num_pairs, len(dec_a_blob), len(dec_b_blob), len(hp_blob),
        len(lat_a_bytes), len(lat_b_bytes), len(meta_bytes),
    )
    return (
        header
        + dec_a_blob + dec_b_blob + hp_blob
        + lat_a_bytes + lat_b_bytes + meta_bytes
    )


def parse_archive(blob: bytes) -> PactNervNeuralCodecE2ECrossArchive:
    if len(blob) < NCEC_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim_a, latent_dim_b, num_pairs,
        dec_a_len, dec_b_len, hp_len, lat_a_len, lat_b_len, meta_len,
    ) = struct.unpack(NCEC_HEADER_FMT, blob[:NCEC_HEADER_SIZE])
    if magic != NCEC_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != NCEC_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    expected_lat_a = num_pairs * latent_dim_a * 2
    expected_lat_b = num_pairs * latent_dim_b * 2
    if lat_a_len != expected_lat_a:
        raise ValueError(f"lat_a_len {lat_a_len} != {expected_lat_a}")
    if lat_b_len != expected_lat_b:
        raise ValueError(f"lat_b_len {lat_b_len} != {expected_lat_b}")
    end_hdr = NCEC_HEADER_SIZE
    end_dec_a = end_hdr + dec_a_len
    end_dec_b = end_dec_a + dec_b_len
    end_hp = end_dec_b + hp_len
    end_lat_a = end_hp + lat_a_len
    end_lat_b = end_lat_a + lat_b_len
    end_meta = end_lat_b + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")
    sd_a = _deserialize_sd(blob[end_hdr:end_dec_a])
    sd_b = _deserialize_sd(blob[end_dec_a:end_dec_b])
    sd_hp = _deserialize_sd(blob[end_dec_b:end_hp])
    meta = json.loads(blob[end_lat_b:end_meta].decode("utf-8"))
    import numpy as np
    q_a = torch.from_numpy(
        np.frombuffer(blob[end_hp:end_lat_a], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim_a)
    q_b = torch.from_numpy(
        np.frombuffer(blob[end_lat_a:end_lat_b], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim_b)
    sa = float(meta.pop("_lat_a_quant_scale"))
    za = float(meta.pop("_lat_a_quant_zero_point"))
    sb = float(meta.pop("_lat_b_quant_scale"))
    zb = float(meta.pop("_lat_b_quant_zero_point"))
    latents_a = _dequant_int16(q_a, sa, za)
    latents_b = _dequant_int16(q_b, sb, zb)
    return PactNervNeuralCodecE2ECrossArchive(
        decoder_a_state_dict=sd_a,
        decoder_b_state_dict=sd_b,
        hyperprior_state_dict=sd_hp,
        latents_a=latents_a,
        latents_b=latents_b,
        meta=meta,
        schema_version=int(version),
    )
