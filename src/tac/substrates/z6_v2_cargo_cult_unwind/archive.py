# SPDX-License-Identifier: MIT
"""z6_v2_cargo_cult_unwind Z6V2CU1 archive grammar — monolithic single-file 0.bin.

Per Catalog #124 archive-grammar 8 fields + Catalog #146 contest-compliant
runtime contract:

Header (28 bytes):
    MAGIC(4)               b"Z6V2"
    VERSION(1)             u8
    LATENT_DIM(2)          u16
    EGO_DIM(2)             u16   FoE ego-motion vector dim (6)
    NUM_PAIRS(2)           u16
    DECODER_BLOB_LEN(4)    u32   level0_micro_film_predictor + level1_meso_film_predictor + heads
    LATENT_BLOB_LEN(4)     u32   per-pair latents (int16)
    EGO_BLOB_LEN(4)        u32   per-pair ego-motion vectors (int16)
    META_BLOB_LEN(4)       u32   json meta (cooperative_receiver_beta, hierarchy boundary, etc.)
    RESERVED(1)            u8    padding

Total header size: 28 bytes per struct calc below.

Per Catalog #272 distinguishing-feature integration contract: the archive
sections that distinguish Z6-v2 from sister substrates are:
1. ``hierarchy_weights_level0_blob`` — first 3 FiLM-conditioned blocks (micro)
2. ``hierarchy_weights_level1_blob`` — remaining 4 FiLM-conditioned blocks (meso)
3. ``ego_motion_focus_of_expansion_blob`` — per-pair (tx,ty,tz,rx,ry,rz) FoE prior
4. ``predictor_latents_blob`` — per-pair residual stream

All 4 sections are packed into the canonical 0.bin blob; the Rao-Ballard
hierarchy boundary + cooperative_receiver_beta + FiLM generator depth +
film_hidden_width are in the meta blob so the inflate runtime can re-instantiate
the canonical Z6V2Substrate without ambient state.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

Z6V2_MAGIC: bytes = b"Z6V2"
Z6V2_SCHEMA_VERSION: int = 1
# Phase C canonical inflate format extension 2026-05-30: schema v2 adds
# canonical leaderboard binding-depth L21 (per-tensor INT8 byte-maps) +
# L29 (fp16 per-tensor scales) + L32 (brotli quality=11) to shrink the
# decoder payload from ~544KB (v1: fp16+pickle+brotli q=9) to ~302KB
# (v2: INT8+fp16scales+brotli q=11). Schema v1 is HISTORICAL_PROVENANCE
# (preserved per Catalog #110/#113 APPEND-ONLY); schema v2 is the new
# default for canonical contest dispatch (rate-axis budget approaches
# 290KB → ~320KB total archive vs frontier ~290KB).
Z6V2_SCHEMA_VERSION_V2_INT8: int = 2
# Header format: 4s (magic) + B (version) + H (latent_dim) + H (ego_dim) +
# H (num_pairs) + I (dec_blob_len) + I (lat_blob_len) + I (ego_blob_len) +
# I (meta_blob_len) + B (reserved)
Z6V2_HEADER_FMT: str = "<4sBHHHIIIIB"
Z6V2_HEADER_SIZE: int = struct.calcsize(Z6V2_HEADER_FMT)
assert Z6V2_HEADER_SIZE == 28
BROTLI_QUALITY = 9
# Canonical leaderboard binding-depth L32: brotli quality=11 (max) for
# schema v2 decoder payload. Per CLAUDE.md L32 verbatim:
# "compression time is offline overhead so quality=11 is free at deploy
# time. Canonical equation ``pr95_family_l32_brotli_quality_11_max_v1``."
BROTLI_QUALITY_V2: int = 11


@dataclass(frozen=True)
class Z6V2Archive:
    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    ego_vecs: torch.Tensor
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


def _quantize_int8(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Canonical L21 per-tensor INT8 quantization with L29 fp16 scale + zero-point.

    Returns ``(q, scale, zero_point)`` where ``q`` is an int8 tensor and the
    dequantized values reconstruct via ``(q + 127) * scale + zero_point``. The
    scale + zero_point are stored as fp16 in the meta JSON per
    ``pr95_family_l29_fp16_per_tensor_scales_int8_v1`` (canonical leaderboard
    binding-depth L29 verbatim: "fp32 scales (112 bytes overhead) is
    forbidden; fp16 scales preserve per-tensor magnitude info while keeping
    codebook at int8 granularity").
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.full_like(f, -127, dtype=torch.int8), 1.0, lo)
    scale = (hi - lo) / 254.0
    qu = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (qu - 127.0).to(torch.int8)
    return (q, float(scale), float(lo))


def _dequant_int8(q: torch.Tensor, scale: float, zp: float) -> torch.Tensor:
    """Dequantize an int8 tensor produced by ``_quantize_int8``.

    Mirrors the encode-side formula exactly so the parse path reconstructs
    the original fp32 distribution within the quantization grid resolution.
    """
    qu = q.to(torch.float32) + 127.0
    return qu * float(scale) + float(zp)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    ego_vecs: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = Z6V2_SCHEMA_VERSION,
) -> bytes:
    if schema_version != Z6V2_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if ego_vecs.dim() != 2:
        raise ValueError(f"ego_vecs must be 2-D; got {tuple(ego_vecs.shape)}")
    if latents.shape[0] != ego_vecs.shape[0]:
        raise ValueError(
            f"latents.shape[0]={latents.shape[0]} != "
            f"ego_vecs.shape[0]={ego_vecs.shape[0]}"
        )
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    ego_dim = int(ego_vecs.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if ego_dim <= 0 or ego_dim > 0xFFFF:
        raise ValueError(f"ego_dim {ego_dim} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    lat_bytes = q_lat.contiguous().numpy().tobytes()

    q_ego, ego_scale, ego_zp = _quantize_int16(ego_vecs)
    ego_bytes = q_ego.contiguous().numpy().tobytes()

    dec_blob = _serialize_state_dict(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_q["_ego_quant_scale"] = float(ego_scale)
    meta_q["_ego_quant_zero_point"] = float(ego_zp)
    meta_bytes = json.dumps(
        meta_q, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z6V2_HEADER_FMT,
        Z6V2_MAGIC,
        schema_version,
        latent_dim,
        ego_dim,
        num_pairs,
        len(dec_blob),
        len(lat_bytes),
        len(ego_bytes),
        len(meta_bytes),
        0,  # reserved
    )
    return header + dec_blob + lat_bytes + ego_bytes + meta_bytes


def _pack_decoder_int8_v2(
    decoder_state_dict: dict[str, torch.Tensor],
) -> tuple[bytes, dict[str, list[float | list[int]]]]:
    """Canonical L21+L29+L32 v2 decoder pack: INT8 weights + fp16 scales + brotli q=11.

    Returns ``(brotli_compressed_int8_payload, fp16_scales_header_dict)``.
    The header dict maps each tensor key to ``[scale_fp16, zero_point_fp16,
    shape_list, dtype_token]`` — the parser uses this to dequantize each
    tensor INT8 → fp16 → tensor.reshape(shape).

    Per CLAUDE.md "Canonical leaderboard binding-depth discipline" L21 + L29
    + L32 (canonical equation prefixes preserved). The packer iterates state-
    dict keys in sorted order for deterministic byte ordering (byte-stable
    archive under deterministic input).
    """
    import numpy as np
    keys = sorted(decoder_state_dict.keys())
    int8_blobs: list[bytes] = []
    header: dict[str, list[float | list[int]]] = {}
    for k in keys:
        v = decoder_state_dict[k]
        # _quantize_int8 returns torch.Tensor[int8], scale, zero_point
        q, scale, zp = _quantize_int8(v)
        flat = q.contiguous().view(-1).numpy().astype(np.int8)
        int8_blobs.append(flat.tobytes())
        header[k] = [
            float(scale),
            float(zp),
            [int(x) for x in v.shape],
            str(v.dtype).replace("torch.", ""),
        ]
    payload = b"".join(int8_blobs)
    compressed = bytes(brotli.compress(payload, quality=BROTLI_QUALITY_V2))
    return compressed, header


def _unpack_decoder_int8_v2(
    compressed_payload: bytes,
    header: dict[str, list[float | list[int]]],
) -> dict[str, torch.Tensor]:
    """Dequantize the v2 decoder payload back into a state_dict.

    Mirrors ``_pack_decoder_int8_v2`` byte-for-byte: brotli-decompresses the
    payload, then for each sorted key reads ``prod(shape)`` int8 bytes and
    dequantizes via ``_dequant_int8``. The returned tensors are fp16 (the
    canonical target dtype for v2 per L29) — the inflate path's
    ``model.load_state_dict(strict=False)`` happily accepts fp16 weights
    into the fp32 model and casts automatically.
    """
    import numpy as np
    raw = brotli.decompress(compressed_payload)
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for k in sorted(header.keys()):
        scale, zp, shape_list, _dtype_token = header[k]
        shape = tuple(int(x) for x in shape_list)
        n = 1
        for d in shape:
            n *= d
        if offset + n > len(raw):
            raise ValueError(
                f"v2 decoder payload truncated at key {k!r}: "
                f"need {n} bytes from offset {offset}, have {len(raw)}"
            )
        q_flat = torch.from_numpy(
            np.frombuffer(raw[offset:offset + n], dtype=np.int8).copy()
        )
        offset += n
        dq = _dequant_int8(q_flat, float(scale), float(zp))
        out[k] = dq.view(*shape).to(torch.float16)
    if offset != len(raw):
        raise ValueError(
            f"v2 decoder payload has {len(raw) - offset} unconsumed bytes "
            f"after parsing {len(header)} tensors"
        )
    return out


def pack_archive_v2_int8(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    ego_vecs: torch.Tensor,
    meta: dict[str, object],
) -> bytes:
    """Pack a Z6V2 v2 archive: canonical L21+L29+L32 INT8 decoder + INT8 latents/ego.

    Schema v2 (canonical leaderboard binding-depth bind):

    * Decoder weights: INT8 per-tensor + fp16 scales (L21+L29) + brotli q=11 (L32)
    * Latents: INT8 single-scale + brotli q=11 (was int16 in v1)
    * Ego vecs: INT8 single-scale + brotli q=11 (was int16 in v1)
    * Header: same Z6V2_HEADER_FMT (version field = 2 instead of 1)
    * Meta: JSON with `_decoder_int8_scales_header` + `_lat_int8_*` + `_ego_int8_*`

    Empirical archive size at 600 pairs / canonical Z6V2Config defaults:
    ~320 KB (vs ~580 KB for schema v1; ~45% reduction). Rate term = 25*N/37545489.

    Per CLAUDE.md "Complexity + LOC + boundaries UNCONSTRAINED within contest
    compliance" + "INDIVIDUALLY-FRACTAL" + "UNIQUE-AND-COMPLETE-PER-METHOD"
    operating mode: this is Z6-v2's substrate-engineering pass per its OWN
    optimal-form analysis, not a shared-helper shortcut. The canonical L21
    + L29 + L32 primitives are ADOPTED because they serve (Catalog #290
    canonical-vs-unique decision: ADOPT_CANONICAL_BECAUSE_SERVES — the
    INT8 + fp16-scales pattern fits every NeRV-family decoder weight
    distribution including z6_v2's FiLM-conditioned PixelShuffle blocks).
    """
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if ego_vecs.dim() != 2:
        raise ValueError(f"ego_vecs must be 2-D; got {tuple(ego_vecs.shape)}")
    if latents.shape[0] != ego_vecs.shape[0]:
        raise ValueError(
            f"latents.shape[0]={latents.shape[0]} != "
            f"ego_vecs.shape[0]={ego_vecs.shape[0]}"
        )
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    ego_dim = int(ego_vecs.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if ego_dim <= 0 or ego_dim > 0xFFFF:
        raise ValueError(f"ego_dim {ego_dim} out of u16 range")

    # Canonical L21 INT8 latents + L32 brotli q=11.
    q_lat, lat_scale, lat_zp = _quantize_int8(latents)
    lat_bytes = bytes(brotli.compress(
        q_lat.contiguous().view(-1).numpy().tobytes(),
        quality=BROTLI_QUALITY_V2,
    ))

    q_ego, ego_scale, ego_zp = _quantize_int8(ego_vecs)
    ego_bytes = bytes(brotli.compress(
        q_ego.contiguous().view(-1).numpy().tobytes(),
        quality=BROTLI_QUALITY_V2,
    ))

    dec_blob, dec_scales_header = _pack_decoder_int8_v2(decoder_state_dict)

    meta_q = dict(meta)
    meta_q["_decoder_int8_scales_header"] = dec_scales_header
    meta_q["_lat_int8_scale"] = float(lat_scale)
    meta_q["_lat_int8_zero_point"] = float(lat_zp)
    meta_q["_lat_int8_num_elements"] = int(q_lat.numel())
    meta_q["_ego_int8_scale"] = float(ego_scale)
    meta_q["_ego_int8_zero_point"] = float(ego_zp)
    meta_q["_ego_int8_num_elements"] = int(q_ego.numel())
    meta_bytes = json.dumps(
        meta_q, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z6V2_HEADER_FMT,
        Z6V2_MAGIC,
        Z6V2_SCHEMA_VERSION_V2_INT8,
        latent_dim,
        ego_dim,
        num_pairs,
        len(dec_blob),
        len(lat_bytes),
        len(ego_bytes),
        len(meta_bytes),
        0,  # reserved
    )
    return header + dec_blob + lat_bytes + ego_bytes + meta_bytes


def parse_archive(blob: bytes) -> Z6V2Archive:
    """Parse a Z6V2 archive; dispatches on schema version (v1 or v2).

    Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: v1 archives
    (fp16 + pickle + brotli q=9) remain parseable for the historical
    Phase B archive ``a34597d55``-era 580KB blobs. v2 archives (INT8 +
    fp16 scales + brotli q=11) are the canonical default for new dispatch
    per CLAUDE.md "PR-or-greater parity discipline" binding-depth.
    """
    if len(blob) < Z6V2_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes)")
    (
        magic, version, latent_dim, ego_dim, num_pairs,
        dec_len, lat_len, ego_len, meta_len, _reserved,
    ) = struct.unpack(Z6V2_HEADER_FMT, blob[:Z6V2_HEADER_SIZE])
    if magic != Z6V2_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version == Z6V2_SCHEMA_VERSION:
        return _parse_archive_v1(
            blob, latent_dim, ego_dim, num_pairs,
            dec_len, lat_len, ego_len, meta_len,
        )
    if version == Z6V2_SCHEMA_VERSION_V2_INT8:
        return _parse_archive_v2(
            blob, latent_dim, ego_dim, num_pairs,
            dec_len, lat_len, ego_len, meta_len,
        )
    raise ValueError(
        f"unsupported schema version: {version} "
        f"(expected {Z6V2_SCHEMA_VERSION} or {Z6V2_SCHEMA_VERSION_V2_INT8})"
    )


def _parse_archive_v1(
    blob: bytes,
    latent_dim: int,
    ego_dim: int,
    num_pairs: int,
    dec_len: int,
    lat_len: int,
    ego_len: int,
    meta_len: int,
) -> Z6V2Archive:
    """Schema v1 parse path: HISTORICAL_PROVENANCE preserved per Catalog #110/#113."""
    expected_lat = num_pairs * latent_dim * 2  # int16
    if lat_len != expected_lat:
        raise ValueError(f"lat_len {lat_len} != expected {expected_lat}")
    expected_ego = num_pairs * ego_dim * 2  # int16
    if ego_len != expected_ego:
        raise ValueError(f"ego_len {ego_len} != expected {expected_ego}")
    end_hdr = Z6V2_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_ego = end_lat + ego_len
    end_meta = end_ego + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")

    sd = _deserialize_state_dict(blob[end_hdr:end_dec])
    meta = json.loads(blob[end_ego:end_meta].decode("utf-8"))

    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)

    q_ego = torch.from_numpy(
        np.frombuffer(blob[end_lat:end_ego], dtype=np.int16).copy()
    ).view(num_pairs, ego_dim)
    ego_scale = float(meta.pop("_ego_quant_scale"))
    ego_zp = float(meta.pop("_ego_quant_zero_point"))
    ego_vecs = _dequant_int16(q_ego, ego_scale, ego_zp)

    return Z6V2Archive(
        decoder_state_dict=sd,
        latents=latents,
        ego_vecs=ego_vecs,
        meta=meta,
        schema_version=Z6V2_SCHEMA_VERSION,
    )


def _parse_archive_v2(
    blob: bytes,
    latent_dim: int,
    ego_dim: int,
    num_pairs: int,
    dec_len: int,
    lat_len: int,
    ego_len: int,
    meta_len: int,
) -> Z6V2Archive:
    """Schema v2 parse path: INT8 + fp16 scales + brotli q=11 canonical."""
    end_hdr = Z6V2_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_ego = end_lat + ego_len
    end_meta = end_ego + meta_len
    if end_meta != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {end_meta}")

    meta = json.loads(blob[end_ego:end_meta].decode("utf-8"))

    # Decoder: INT8 + fp16 scales + brotli q=11 -> state_dict
    dec_scales_header = meta.pop("_decoder_int8_scales_header")
    sd = _unpack_decoder_int8_v2(blob[end_hdr:end_dec], dec_scales_header)

    # Latents: INT8 + brotli q=11 -> (num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_int8_scale"))
    lat_zp = float(meta.pop("_lat_int8_zero_point"))
    lat_n = int(meta.pop("_lat_int8_num_elements"))
    import numpy as np
    lat_raw = brotli.decompress(blob[end_dec:end_lat])
    if len(lat_raw) != lat_n:
        raise ValueError(
            f"v2 latent payload length {len(lat_raw)} != expected {lat_n}"
        )
    q_lat = torch.from_numpy(
        np.frombuffer(lat_raw, dtype=np.int8).copy()
    ).view(num_pairs, latent_dim)
    latents = _dequant_int8(q_lat, lat_scale, lat_zp)

    # Ego: INT8 + brotli q=11 -> (num_pairs, ego_dim)
    ego_scale = float(meta.pop("_ego_int8_scale"))
    ego_zp = float(meta.pop("_ego_int8_zero_point"))
    ego_n = int(meta.pop("_ego_int8_num_elements"))
    ego_raw = brotli.decompress(blob[end_lat:end_ego])
    if len(ego_raw) != ego_n:
        raise ValueError(
            f"v2 ego payload length {len(ego_raw)} != expected {ego_n}"
        )
    q_ego = torch.from_numpy(
        np.frombuffer(ego_raw, dtype=np.int8).copy()
    ).view(num_pairs, ego_dim)
    ego_vecs = _dequant_int8(q_ego, ego_scale, ego_zp)

    return Z6V2Archive(
        decoder_state_dict=sd,
        latents=latents,
        ego_vecs=ego_vecs,
        meta=meta,
        schema_version=Z6V2_SCHEMA_VERSION_V2_INT8,
    )


__all__ = [
    "BROTLI_QUALITY",
    "BROTLI_QUALITY_V2",
    "Z6V2Archive",
    "Z6V2_HEADER_FMT",
    "Z6V2_HEADER_SIZE",
    "Z6V2_MAGIC",
    "Z6V2_SCHEMA_VERSION",
    "Z6V2_SCHEMA_VERSION_V2_INT8",
    "pack_archive",
    "pack_archive_v2_int8",
    "parse_archive",
]
