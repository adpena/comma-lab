# SPDX-License-Identifier: MIT
"""sane_hnerv archive grammar — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (lesson L2):

::

    MAGIC(4)            b"SHV1"  Sane-Hnerv Variant 1
    VERSION(1)          u8       schema version (currently 1)
    LATENT_DIM(2)       u16      cfg.latent_dim (e.g., 28)
    NUM_PAIRS(2)        u16      cfg.num_pairs (e.g., 600)
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder state_dict bytes len
    LATENT_BLOB_LEN(4)  u32      raw int16 latents bytes len (= num_pairs * latent_dim * 2)
    META_BLOB_LEN(4)    u32      utf-8 json meta bytes len
    DECODER_BLOB        ...      brotli(quality=9, mode=GENERIC) of pickled state_dict
    LATENT_BLOB         ...      int16 latents row-major (num_pairs, latent_dim)
    META_BLOB           ...      json: {"sin_freq": ..., "decoder_channels": [...]}

The grammar is FIXED at design-time; mutating it changes the schema VERSION
and requires a new inflate.py.

Round-trip contract (tested in tests/test_sane_hnerv_roundtrip.py per
Catalog #91):

    bytes -> parse_archive -> (decoder_state_dict, latents_tensor, meta_dict)
    (decoder_state_dict, latents_tensor, meta_dict) -> pack_archive -> bytes

The parse_archive() return type IS the inflate-time API. inflate.py imports
parse_archive() and the substrate forward; nothing else.

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load
- Deterministic: same input -> same bytes (no timestamps, no host info)
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch


SHV1_MAGIC: bytes = b"SHV1"
"""sane_hnerv variant 1 archive magic."""

SHV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + NUM_PAIRS(2)
#                + DECODER_LEN(4) + LATENT_LEN(4) + META_LEN(4)  = 21 bytes
SHV1_HEADER_FMT: str = "<4sBHHIII"
SHV1_HEADER_SIZE: int = struct.calcsize(SHV1_HEADER_FMT)
assert SHV1_HEADER_SIZE == 21, "header size invariant"

# Brotli quality (PR101 uses 9-11; 9 is the safe default for archive grammar)
BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class SaneHnervArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (all model weights except per-pair latents)."""

    latents: torch.Tensor
    """``(num_pairs, latent_dim)`` int16 quantized latents."""

    meta: dict[str, object]
    """Sidecar JSON meta: sin_freq, decoder_channels, output_h/w, ..."""

    schema_version: int
    """Archive schema version (must match SHV1_SCHEMA_VERSION)."""


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically.

    torch.save is non-deterministic across pytorch versions in some edge cases;
    using pickle on cpu-tensored fp16 state_dict avoids the torch.save header.
    """
    buf = io.BytesIO()
    # Cast all tensors to fp16 on CPU for stable bytes
    sd_cpu = {k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()}
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("decoder_state_dict blob did not unpickle to a dict")
    return sd


def _quantize_latents_to_int16(latents: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize latents to int16. Returns (q, scale, zero_point) so the
    inflate-time path can dequantize without consulting the model.

    The scale + zero_point are stored in META_BLOB; the int16 stream
    (interpreted as signed two's-complement int16) goes in LATENT_BLOB.

    Quantization: ``q_int16 = round((f - zero_point) / scale) - 32767``
    such that ``f = (q_int16 + 32767) * scale + zero_point``. This gives
    the full 65535 levels of int16 storage.

    Degenerate (all-equal) case: ``hi <= lo`` (single unique value or
    NaN/inf). q is filled with ``-32767`` so that
    ``dequant(q) = (q + 32767) * scale + zero_point = 0 * scale + lo = lo``.
    The earlier ``zeros_like`` fill produced
    ``(0 + 32767) * 1.0 + lo = 32767 + lo``, off by 32767. Sister bug
    NNN flagged: same pattern was fixed in
    ``src/tac/substrates/block_nerv/archive.py``; Catalog #158 STRICT
    preflight refuses re-introduction of this class.
    """
    if latents.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"latents must be float; got {latents.dtype}")
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -32767, dtype=torch.int16),
            1.0,
            lo,
        )
    # Map [lo, hi] -> [0, 65534] -> [-32767, 32767]
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_latents(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = SHV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize a substrate's trained weights + latents + meta into the
    monolithic ``0.bin`` archive bytes.

    This is the export-first contract: the trainer ONLY calls this; everything
    else (CRC, framing, padding) is the codec's responsibility, not the
    training loop's.
    """
    if schema_version != SHV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}")

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_latents, scale, zero_point = _quantize_latents_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_quant_scale"] = float(scale)
    meta_with_quant["_quant_zero_point"] = float(zero_point)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        SHV1_HEADER_FMT,
        SHV1_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        len(decoder_blob),
        len(latent_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + meta_bytes


def parse_archive(blob: bytes) -> SaneHnervArchive:
    """Parse the ``0.bin`` bytes back into trained-weight + latents + meta.

    Pure-bytes function — no model class needed. inflate.py imports this +
    the model class + builds + loads + renders, in ~75 LOC total.
    """
    if len(blob) < SHV1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes; need >= {SHV1_HEADER_SIZE})")
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(SHV1_HEADER_FMT, blob[:SHV1_HEADER_SIZE])
    if magic != SHV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {SHV1_MAGIC!r})")
    if version != SHV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2  # int16 = 2 bytes
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = SHV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_meta = end_latents + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    meta_blob = blob[end_latents:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import to keep the module's import-time hot path light
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    latents = _dequantize_latents(q_latents, scale, zp)

    return SaneHnervArchive(
        decoder_state_dict=sd,
        latents=latents,
        meta=meta,
        schema_version=int(version),
    )
