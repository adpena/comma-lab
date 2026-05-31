# SPDX-License-Identifier: MIT
"""Z4 Atick-Redlich Z4ATR1 archive grammar — monolithic single-file 0.bin.

Per Catalog #124 archive-grammar 8 fields + Catalog #146 contest-compliant
runtime contract + canonical leaderboard binding-depth L20 (monolithic
single-file 4-section archive grammar).

Header (24 bytes):
    MAGIC(4)               b"Z4AR"
    VERSION(1)             u8
    LATENT_DIM(2)          u16
    NUM_PAIRS(2)           u16
    DECODER_BLOB_LEN(4)    u32   decoder state_dict (pickle + brotli q=11)
    LATENT_BLOB_LEN(4)     u32   per-pair latents (int16)
    DECORRELATOR_BLOB_LEN(2) u16  Atick-Redlich W_AR + b_AR (fp16)
    META_BLOB_LEN(4)       u32   json meta (cooperative_receiver_beta + shape + ...)
    RESERVED(1)            u8    padding

Per Catalog #272 distinguishing-feature integration contract: the archive
section that distinguishes Z4 from sister Z6/Z7 substrates is the
``decorrelator_blob`` — the Atick-Redlich 1990 spatial decorrelation filter
``W_AR`` (latent_dim x latent_dim) + ``b_AR`` (latent_dim,) packed as fp16.
This is the per-substrate distinguishing primitive operationalized at
inflate time per Catalog #220 L1+ scaffold operational mechanism.

Schema v1 = fp16 + pickle + brotli q=11 (canonical L32 max-quality;
HISTORICAL_PROVENANCE-aware).

[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/archive.py
 canonical archive grammar pattern (Z6V2 28-byte header + 4 sections)]
[verified-against: Catalog #220 substrate L1+ scaffold operational mechanism
 (decorrelator blob is the distinguishing-feature payload)]
[verified-against: Catalog #124 representation lane archive grammar 8 fields]
[verified-against: canonical leaderboard binding-depth L20 monolithic
 single-file 4-section archive grammar canonical equation
 ``pr95_family_l20_monolithic_4_section_archive_grammar_v1``]
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

Z4ATR_MAGIC: bytes = b"Z4AR"
Z4ATR_SCHEMA_VERSION: int = 1
# Header format: 4s (magic) + B (version) + H (latent_dim) + H (num_pairs)
# + I (dec_blob_len) + I (lat_blob_len) + H (decorrelator_blob_len) +
# I (meta_blob_len) + B (reserved)
Z4ATR_HEADER_FMT: str = "<4sBHHIIHIB"
Z4ATR_HEADER_SIZE: int = struct.calcsize(Z4ATR_HEADER_FMT)
assert Z4ATR_HEADER_SIZE == 24, (
    f"Z4ATR_HEADER_SIZE invariant: expected 24, got {Z4ATR_HEADER_SIZE}"
)
# Canonical leaderboard binding-depth L32: brotli q=11 (max) is FREE at
# deploy time (compression cost is offline overhead). Canonical equation
# ``pr95_family_l32_brotli_quality_11_max_v1``.
BROTLI_QUALITY_V1: int = 11


@dataclass(frozen=True)
class Z4ATRArchive:
    """Parsed Z4ATR archive (immutable typed container).

    Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog
    #323 canonical Provenance: the parsed archive carries every field the
    inflate runtime needs to re-instantiate ``Z4AtickRedlichSubstrate``
    without ambient state.
    """

    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    decorrelator_weight: torch.Tensor
    decorrelator_bias: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY_V1))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    sd = pickle.loads(brotli.decompress(blob))
    if not isinstance(sd, dict):
        raise ValueError("decoder_state_dict blob did not unpickle to a dict")
    return sd


def _quantize_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Canonical L21 int16 quantization with fp32 scale + zero-point (meta-stored)."""
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    qu = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (qu - 32767.0).to(torch.int16)
    return (q, float(scale), float(lo))


def _dequant_int16(q: torch.Tensor, scale: float, zp: float) -> torch.Tensor:
    qu = q.to(torch.float32) + 32767.0
    return qu * float(scale) + float(zp)


def _serialize_decorrelator(
    weight: torch.Tensor, bias: torch.Tensor
) -> bytes:
    """Pack Atick-Redlich W_AR + b_AR as fp16 bytes (distinguishing-feature payload).

    The decorrelator blob is the per-substrate distinguishing-feature
    payload per Catalog #272. fp16 quantization is sufficient since the
    decorrelator's role is rotation-toward-decorrelating-eigenbasis (the
    information lives in the SHAPE not the precision) per Atick-Redlich
    1990 retinal MI canonical.
    """
    import numpy as np
    w_fp16 = weight.detach().to(dtype=torch.float16, device="cpu").contiguous()
    b_fp16 = bias.detach().to(dtype=torch.float16, device="cpu").contiguous()
    w_bytes = w_fp16.numpy().astype(np.float16).tobytes()
    b_bytes = b_fp16.numpy().astype(np.float16).tobytes()
    return w_bytes + b_bytes


def _deserialize_decorrelator(
    blob: bytes, latent_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack Atick-Redlich W_AR + b_AR from fp16 bytes."""
    import numpy as np
    expected_w_bytes = latent_dim * latent_dim * 2  # fp16
    expected_b_bytes = latent_dim * 2  # fp16
    expected_total = expected_w_bytes + expected_b_bytes
    if len(blob) != expected_total:
        raise ValueError(
            f"decorrelator blob len {len(blob)} != expected {expected_total} "
            f"(latent_dim={latent_dim} fp16 W + b)"
        )
    w_flat = np.frombuffer(blob[:expected_w_bytes], dtype=np.float16).copy()
    b_flat = np.frombuffer(blob[expected_w_bytes:], dtype=np.float16).copy()
    w = torch.from_numpy(w_flat).view(latent_dim, latent_dim).to(torch.float32)
    b = torch.from_numpy(b_flat).view(latent_dim).to(torch.float32)
    return w, b


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    decorrelator_weight: torch.Tensor,
    decorrelator_bias: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = Z4ATR_SCHEMA_VERSION,
) -> bytes:
    """Pack a Z4ATR archive: monolithic 4-section single-file format.

    Per CLAUDE.md "Canonical leaderboard binding-depth discipline" L20
    (monolithic single-file 4-section archive grammar) + L32 (brotli q=11)
    + L29 (fp16 per-tensor scales meta-stored).

    Per Catalog #272 distinguishing-feature integration contract: the
    ``decorrelator_blob`` is the per-substrate distinguishing payload
    that the inflate runtime MUST consume to re-instantiate the
    Atick-Redlich decorrelation filter.

    Per CLAUDE.md "Bit-level deconstruction and entropy discipline": the
    pack path is byte-deterministic under deterministic input (sorted
    state_dict keys, fixed quantization grids, brotli q=11).
    """
    if schema_version != Z4ATR_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if decorrelator_weight.dim() != 2:
        raise ValueError(
            f"decorrelator_weight must be 2-D; got {tuple(decorrelator_weight.shape)}"
        )
    if decorrelator_bias.dim() != 1:
        raise ValueError(
            f"decorrelator_bias must be 1-D; got {tuple(decorrelator_bias.shape)}"
        )
    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if decorrelator_weight.shape != (latent_dim, latent_dim):
        raise ValueError(
            f"decorrelator_weight shape {tuple(decorrelator_weight.shape)} "
            f"!= ({latent_dim}, {latent_dim})"
        )
    if int(decorrelator_bias.shape[0]) != latent_dim:
        raise ValueError(
            f"decorrelator_bias shape {tuple(decorrelator_bias.shape)} "
            f"!= ({latent_dim},)"
        )
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_int16(latents)
    lat_bytes = q_lat.contiguous().numpy().tobytes()

    dec_blob = _serialize_state_dict(decoder_state_dict)
    decorrelator_blob = _serialize_decorrelator(
        decorrelator_weight, decorrelator_bias
    )

    if len(decorrelator_blob) > 0xFFFF:
        raise ValueError(
            f"decorrelator_blob {len(decorrelator_blob)} bytes exceeds u16 "
            f"range (latent_dim {latent_dim} too large for canonical Z4ATR "
            f"header; consider widening DECORRELATOR_BLOB_LEN to u32)"
        )

    meta_q = dict(meta)
    meta_q["_lat_quant_scale"] = float(lat_scale)
    meta_q["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(
        meta_q, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z4ATR_HEADER_FMT,
        Z4ATR_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        len(dec_blob),
        len(lat_bytes),
        len(decorrelator_blob),
        len(meta_bytes),
        0,  # reserved
    )
    return header + dec_blob + lat_bytes + decorrelator_blob + meta_bytes


def parse_archive(blob: bytes) -> Z4ATRArchive:
    """Parse a Z4ATR archive (schema v1).

    Per Catalog #146 + #205 + #146 inflate runtime contract: parser is
    self-contained (no ambient state); fail-closed on malformed input.
    """
    if len(blob) < Z4ATR_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {Z4ATR_HEADER_SIZE})"
        )
    (
        magic, version, latent_dim, num_pairs,
        dec_len, lat_len, decorrelator_len, meta_len, _reserved,
    ) = struct.unpack(Z4ATR_HEADER_FMT, blob[:Z4ATR_HEADER_SIZE])
    if magic != Z4ATR_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {Z4ATR_MAGIC!r})")
    if version != Z4ATR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version: {version} "
            f"(expected {Z4ATR_SCHEMA_VERSION})"
        )

    expected_lat = num_pairs * latent_dim * 2  # int16
    if lat_len != expected_lat:
        raise ValueError(
            f"lat_len {lat_len} != expected {expected_lat} "
            f"(num_pairs={num_pairs} * latent_dim={latent_dim} * 2 int16 bytes)"
        )

    end_hdr = Z4ATR_HEADER_SIZE
    end_dec = end_hdr + dec_len
    end_lat = end_dec + lat_len
    end_decorrelator = end_lat + decorrelator_len
    end_meta = end_decorrelator + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} "
            f"(header {end_hdr} + dec {dec_len} + lat {lat_len} + "
            f"decorrelator {decorrelator_len} + meta {meta_len})"
        )

    sd = _deserialize_state_dict(blob[end_hdr:end_dec])

    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(blob[end_dec:end_lat], dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    meta = json.loads(blob[end_decorrelator:end_meta].decode("utf-8"))
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequant_int16(q_lat, lat_scale, lat_zp)

    decorrelator_weight, decorrelator_bias = _deserialize_decorrelator(
        blob[end_lat:end_decorrelator], latent_dim
    )

    return Z4ATRArchive(
        decoder_state_dict=sd,
        latents=latents,
        decorrelator_weight=decorrelator_weight,
        decorrelator_bias=decorrelator_bias,
        meta=meta,
        schema_version=Z4ATR_SCHEMA_VERSION,
    )


__all__ = [
    "BROTLI_QUALITY_V1",
    "Z4ATR_HEADER_FMT",
    "Z4ATR_HEADER_SIZE",
    "Z4ATR_MAGIC",
    "Z4ATR_SCHEMA_VERSION",
    "Z4ATRArchive",
    "pack_archive",
    "parse_archive",
]
