"""ff_nerv archive grammar — monolithic single-file ``0.bin`` (FFV1).

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (lesson L2):

::

    MAGIC(4)            b"FFV1"  Frequency-domain NeRV Variant 1
    VERSION(1)          u8       schema version (currently 1)
    LATENT_DIM(2)       u16      cfg.latent_dim (e.g., 16)
    NUM_PAIRS(2)        u16      cfg.num_pairs (e.g., 600)
    FREQ_GRID_H(2)      u16      cfg.freq_grid_h (e.g., 64)
    FREQ_GRID_W(2)      u16      cfg.freq_grid_w (e.g., 64)
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder state_dict bytes len
    LATENT_BLOB_LEN(4)  u32      raw int16 latents bytes len (= num_pairs*latent_dim*2)
    META_BLOB_LEN(4)    u32      utf-8 json meta bytes len
    DECODER_BLOB        ...      brotli(quality=9) of pickled state_dict
    LATENT_BLOB         ...      int16 latents row-major (num_pairs, latent_dim)
    META_BLOB           ...      json: {"sin_freq": ..., "decoder_channels": [...]}

Header: 4+1+2+2+2+2+4+4+4 = 25 bytes (FFV1 = 4 more bytes than SHV1 because
FFV1 carries explicit freq_grid_h/w in the header rather than meta).

NOTE — the FFV1 grammar has 5 logical sections per Catalog #124 parser
manifest: HEADER + DECODER_BLOB + LATENT_BLOB + META_BLOB + (implicit
"frequency_band_coefficients" section, which IS the dct_coeff_decoder
state_dict bytes inside DECODER_BLOB). The 5th section is logical-only —
the inflate path reads it from DECODER_BLOB but the design declares it
explicitly so a future zero-redundant rate split can address it directly.

Round-trip contract (tests):
    bytes -> parse_archive -> (decoder_state_dict, latents_tensor, meta_dict)
    (decoder_state_dict, latents_tensor, meta_dict) -> pack_archive -> bytes

CLAUDE.md compliance:
- Deterministic: same input -> same bytes
- No /tmp paths
- No scorer load
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

FFV1_MAGIC: bytes = b"FFV1"
"""ff_nerv variant 1 archive magic."""

FFV1_SCHEMA_VERSION: int = 1

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + NUM_PAIRS(2)
#                + FREQ_GRID_H(2) + FREQ_GRID_W(2)
#                + DECODER_LEN(4) + LATENT_LEN(4) + META_LEN(4)  = 25 bytes
FFV1_HEADER_FMT: str = "<4sBHHHHIII"
FFV1_HEADER_SIZE: int = struct.calcsize(FFV1_HEADER_FMT)
assert FFV1_HEADER_SIZE == 25, "FFV1 header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class FfnervArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (model weights minus per-pair latents)."""

    latents: torch.Tensor
    """``(num_pairs, latent_dim)`` dequantized latents."""

    meta: dict[str, object]
    """Sidecar JSON: sin_freq, decoder_channels, output_h/w, ..."""

    schema_version: int
    """Archive schema version."""

    freq_grid_h: int
    """Vertical DCT basis count (low-freq band height)."""

    freq_grid_w: int
    """Horizontal DCT basis count."""


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


def _quantize_latents_to_int16(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize float latents to int16 with scale + zero_point sidecar."""
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
    meta: dict[str, object],
    *,
    freq_grid_h: int,
    freq_grid_w: int,
    schema_version: int = FFV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained weights + latents + meta into the monolithic 0.bin bytes."""
    if schema_version != FFV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    for name, v in (
        ("num_pairs", num_pairs),
        ("latent_dim", latent_dim),
        ("freq_grid_h", freq_grid_h),
        ("freq_grid_w", freq_grid_w),
    ):
        if v <= 0 or v > 0xFFFF:
            raise ValueError(f"{name} {v} out of u16 range")

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
        FFV1_HEADER_FMT,
        FFV1_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        int(freq_grid_h),
        int(freq_grid_w),
        len(decoder_blob),
        len(latent_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + meta_bytes


def parse_archive(blob: bytes) -> FfnervArchive:
    """Parse the 0.bin bytes back into trained-weight + latents + meta."""
    if len(blob) < FFV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {FFV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        freq_grid_h,
        freq_grid_w,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(FFV1_HEADER_FMT, blob[:FFV1_HEADER_SIZE])
    if magic != FFV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {FFV1_MAGIC!r})")
    if version != FFV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = FFV1_HEADER_SIZE
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

    import numpy as np
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    latents = _dequantize_latents(q_latents, scale, zp)

    return FfnervArchive(
        decoder_state_dict=sd,
        latents=latents,
        meta=meta,
        schema_version=int(version),
        freq_grid_h=int(freq_grid_h),
        freq_grid_w=int(freq_grid_w),
    )
