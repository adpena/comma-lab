# SPDX-License-Identifier: MIT
"""tc_nerv archive grammar — monolithic single-file ``0.bin`` (TCV1).

L0 SKETCH archive grammar; Catalog #124 8-field declaration is the
``__init__`` module docstring. This file IS the export-first grammar (L2):

::

    MAGIC(4)            b"TCV1"  Temporal-Consistency NeRV Variant 1
    VERSION(1)          u8       schema version (currently 1)
    LATENT_DIM(2)       u16      cfg.latent_dim (e.g., 24)
    NUM_PAIRS(2)        u16      cfg.num_pairs (e.g., 600)
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder state_dict bytes
    LATENT_BLOB_LEN(4)  u32      raw int16 latents bytes len
    TC_TABLE_BLOB_LEN(4) u32     raw int8 temporal correlation table bytes len
    META_BLOB_LEN(4)    u32      utf-8 json meta bytes len
    DECODER_BLOB        ...      brotli(quality=9, mode=GENERIC) pickled state_dict
    LATENT_BLOB         ...      int16 latents row-major (num_pairs, latent_dim)
    TC_TABLE_BLOB       ...      int8 per-pair adjacency strengths (num_pairs,)
    META_BLOB           ...      json: {"sin_freq": ..., "decoder_channels": [...]}

The new section vs SHV1 is **TC_TABLE_BLOB**: a 600-byte (one int8 per pair)
table giving the per-pair empirical adjacency strength
``mean(||rgb_t+1 - rgb_t||^2)`` after EMA shadow consolidation. The inflate
runtime does NOT consume this table (it's purely a training-time provenance
record + a continuation-anchor for L1 re-tuning); for byte-mutation
no_op_proof purposes the table IS consumed by the parser to confirm round-
trip integrity, so a mutated byte does propagate to the parsed archive.

Header layout: 23 bytes total (vs SHV1's 21 — the extra 2 bytes from the new
TC_TABLE_BLOB_LEN u32 minus the shared layout).

Wait — recompute:
    SHV1 header = MAGIC(4) + VERSION(1) + LATENT_DIM(2) + NUM_PAIRS(2)
                 + DECODER_LEN(4) + LATENT_LEN(4) + META_LEN(4) = 21 bytes
    TCV1 header = SHV1 + TC_TABLE_LEN(4) = 25 bytes

CLAUDE.md compliance:
- No silent defaults
- No /tmp paths
- No scorer load
- Deterministic
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch


TCV1_MAGIC: bytes = b"TCV1"
"""tc_nerv variant 1 archive magic."""

TCV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + NUM_PAIRS(2)
#                + DECODER_LEN(4) + LATENT_LEN(4) + TC_TABLE_LEN(4) + META_LEN(4)
#                = 25 bytes
TCV1_HEADER_FMT: str = "<4sBHHIIII"
TCV1_HEADER_SIZE: int = struct.calcsize(TCV1_HEADER_FMT)
assert TCV1_HEADER_SIZE == 25, "header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class TCNervArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (all model weights except per-pair latents)."""

    latents: torch.Tensor
    """``(num_pairs, latent_dim)`` int16-then-dequantized latents."""

    tc_table: torch.Tensor
    """``(num_pairs,)`` int8 per-pair adjacency strengths (training-only)."""

    meta: dict[str, object]
    """Sidecar JSON meta: sin_freq, decoder_channels, output_h/w, etc."""

    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (mirrors sane_hnerv)."""
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


def _quantize_latents_to_int16(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize latents to int16. Returns (q, scale, zero_point)."""
    if latents.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"latents must be float; got {latents.dtype}")
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        # FFFF Catalog #158 fix: -32767 fill so dequant = 0*scale + lo = lo
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_latents(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    tc_table: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = TCV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained weights + latents + tc_table + meta into 0.bin bytes."""
    if schema_version != TCV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
    if tc_table.dim() != 1:
        raise ValueError(
            f"tc_table must be 1-D (num_pairs,); got {tuple(tc_table.shape)}"
        )
    if tc_table.shape[0] != latents.shape[0]:
        raise ValueError(
            f"tc_table length {tc_table.shape[0]} != num_pairs {latents.shape[0]}"
        )

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_latents, scale, zero_point = _quantize_latents_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    # tc_table is stored as raw int8 bytes (clamped/cast by caller)
    if tc_table.dtype != torch.int8:
        raise ValueError(f"tc_table must be int8; got {tc_table.dtype}")
    tc_bytes = tc_table.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_quant_scale"] = float(scale)
    meta_with_quant["_quant_zero_point"] = float(zero_point)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        TCV1_HEADER_FMT,
        TCV1_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        len(decoder_blob),
        len(latent_bytes),
        len(tc_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + tc_bytes + meta_bytes


def parse_archive(blob: bytes) -> TCNervArchive:
    """Parse 0.bin bytes back into trained-weight + latents + tc_table + meta."""
    if len(blob) < TCV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {TCV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        decoder_len,
        latent_len,
        tc_len,
        meta_len,
    ) = struct.unpack(TCV1_HEADER_FMT, blob[:TCV1_HEADER_SIZE])
    if magic != TCV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {TCV1_MAGIC!r})")
    if version != TCV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )
    if tc_len != num_pairs:
        raise ValueError(
            f"tc_len {tc_len} != num_pairs {num_pairs} (one int8 per pair)"
        )

    end_header = TCV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_tc = end_latents + tc_len
    end_meta = end_tc + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    tc_blob = blob[end_latents:end_tc]
    meta_blob = blob[end_tc:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import per sane_hnerv pattern
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    tc_table = torch.from_numpy(
        np.frombuffer(tc_blob, dtype=np.int8).copy()
    )
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    latents = _dequantize_latents(q_latents, scale, zp)

    return TCNervArchive(
        decoder_state_dict=sd,
        latents=latents,
        tc_table=tc_table,
        meta=meta,
        schema_version=int(version),
    )
