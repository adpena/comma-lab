"""Vendored factorized-HNeRV decoder for the submission archive.

Self-contained mirror of :mod:`tac.codec.factorized_hnerv_codec` decoder
path plus PR101-style latent decoder. Inflate-time only loads the float
weights via SVD reconstruction; it does NOT load any scorers (per CLAUDE.md
strict-scorer-rule).

Wire format (factorized section): "FHN1" magic + section header + index
table + brotli(factorized records) + brotli(non-factorized records).
"""
from __future__ import annotations

import struct

import brotli
import numpy as np
import torch


SECTION_MAGIC = b"FHN1"
N_QUANT = 127

FIXED_STATE_SCHEMA = (
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
)
_SCHEMA_INDEX = {n: i for i, (n, _) in enumerate(FIXED_STATE_SCHEMA)}

_SECTION_HEADER = struct.Struct("<4sHHII")
_PER_TENSOR_HEADER = struct.Struct("<HHBHHHHeee")


def _decode_one_factorized_record(buf, pos):
    fields = _PER_TENSOR_HEADER.unpack_from(buf, pos)
    idx, rank, ndim, s0, s1, s2, s3, sU, sS, sV = fields
    pos += _PER_TENSOR_HEADER.size
    name, schema_shape = FIXED_STATE_SCHEMA[idx]
    if ndim == 2:
        original_shape = (s0, s1)
        M, N = s0, s1
    else:
        original_shape = (s0, s1, s2, s3)
        M = s0
        N = int(s1 * s2 * s3)
    if original_shape != schema_shape:
        raise ValueError(
            f"shape mismatch idx={idx}: schema {schema_shape}, on-disk {original_shape}"
        )
    u_size = M * rank
    s_size = rank
    v_size = rank * N
    u_i8 = np.frombuffer(bytes(buf[pos:pos + u_size]), dtype=np.int8).reshape(M, rank).copy()
    pos += u_size
    s_i8 = np.frombuffer(bytes(buf[pos:pos + s_size]), dtype=np.int8).reshape(rank).copy()
    pos += s_size
    v_i8 = np.frombuffer(bytes(buf[pos:pos + v_size]), dtype=np.int8).reshape(rank, N).copy()
    pos += v_size
    # Reconstruct the float tensor: (U_i8 * sU) @ diag(S_i8 * sS) @ (V_i8 * sV)
    u_f = u_i8.astype(np.float64) * float(sU)
    s_f = s_i8.astype(np.float64) * float(sS)
    v_f = v_i8.astype(np.float64) * float(sV)
    m2d = (u_f * s_f[None, :]) @ v_f
    return name, m2d.reshape(original_shape).astype(np.float32), pos


def _decode_one_non_factorized(buf, pos):
    (idx,) = struct.unpack_from("<H", buf, pos); pos += 2
    name, shape = FIXED_STATE_SCHEMA[idx]
    scale = float(np.frombuffer(bytes(buf[pos:pos + 2]), dtype=np.float16)[0])
    pos += 2
    n = int(np.prod(shape))
    q = np.frombuffer(bytes(buf[pos:pos + n]), dtype=np.int8).reshape(shape).copy()
    pos += n
    return name, q.astype(np.float32) * scale, pos


def decode_factorized_section(data):
    """Decode a factorized-HNeRV section to a torch state_dict."""
    if data[:4] != SECTION_MAGIC:
        raise ValueError(f"bad section magic: {data[:4]!r}")
    magic, n_fact, n_non_fact, fact_len, non_fact_len = _SECTION_HEADER.unpack_from(data, 0)
    pos = _SECTION_HEADER.size
    idx_table = np.frombuffer(data, dtype=np.uint16, count=(n_fact + n_non_fact), offset=pos)
    pos += (n_fact + n_non_fact) * 2
    factorized_brotli = data[pos:pos + fact_len]; pos += fact_len
    non_factorized_brotli = data[pos:pos + non_fact_len]; pos += non_fact_len
    if pos != len(data):
        raise ValueError(f"section length mismatch: {pos} != {len(data)}")

    sd = {}
    if factorized_brotli:
        raw = brotli.decompress(factorized_brotli)
        buf = memoryview(raw)
        p = 0
        for _ in range(n_fact):
            name, recon, p = _decode_one_factorized_record(buf, p)
            sd[name] = torch.from_numpy(recon)
        if p != len(raw):
            raise ValueError("trailing bytes in factorized payload")
    if non_factorized_brotli:
        raw = brotli.decompress(non_factorized_brotli)
        buf = memoryview(raw)
        p = 0
        for _ in range(n_non_fact):
            name, recon, p = _decode_one_non_factorized(buf, p)
            sd[name] = torch.from_numpy(recon)
        if p != len(raw):
            raise ValueError("trailing bytes in non-factorized payload")

    # Sanity: schema ordering completeness
    missing = [name for name, _ in FIXED_STATE_SCHEMA if name not in sd]
    if missing:
        raise ValueError(f"decoded state_dict missing tensors: {missing}")
    return sd


def decode_fixed_latents(data):
    """Decode fixed 600x28 latent payload (verbatim port of PR107/PR106 path)."""
    raw = brotli.decompress(data)
    n, d = 600, 28
    meta_len = d * 4
    total = n * d
    lo = np.frombuffer(raw[:total], dtype=np.uint8).astype(np.uint16)
    mins = torch.from_numpy(np.frombuffer(raw[total:total + d * 2], dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(raw[total + d * 2:total + meta_len], dtype=np.float16).copy()).float()
    hi = np.frombuffer(raw[total + meta_len:total + meta_len + total], dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2,
                     -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


# ---------------------------------------------------------------------------
# Top-level archive parse
# ---------------------------------------------------------------------------

# Archive layout (factorized_hnerv_v1):
#   magic byte 0xF1 (factorized HNeRV v1)
#   uint32 decoder_section_len  (LE)
#   <decoder_section_len bytes>  (the factorized section above)
#   uint32 latent_section_len  (LE)
#   <latent_section_len bytes>  (PR101/PR106 brotli'd latent payload)

ARCHIVE_MAGIC = 0xF1


def parse_archive(archive_bytes):
    if len(archive_bytes) < 1 or archive_bytes[0] != ARCHIVE_MAGIC:
        raise ValueError(
            f"factorized_hnerv_v1: bad archive magic, expected 0x{ARCHIVE_MAGIC:02X}, "
            f"got 0x{archive_bytes[0] if archive_bytes else 0:02X}"
        )
    pos = 1
    (dec_len,) = struct.unpack_from("<I", archive_bytes, pos); pos += 4
    decoder_sd = decode_factorized_section(archive_bytes[pos:pos + dec_len])
    pos += dec_len
    (lat_len,) = struct.unpack_from("<I", archive_bytes, pos); pos += 4
    latents = decode_fixed_latents(archive_bytes[pos:pos + lat_len])
    pos += lat_len
    if pos != len(archive_bytes):
        raise ValueError(
            f"factorized_hnerv_v1: trailing bytes "
            f"({len(archive_bytes) - pos} unread)"
        )
    meta = {
        "n_pairs": 600, "latent_dim": 28, "base_channels": 36,
        "eval_size": [384, 512],
    }
    return decoder_sd, latents, meta
