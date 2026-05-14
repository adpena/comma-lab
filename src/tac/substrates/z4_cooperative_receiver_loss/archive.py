# SPDX-License-Identifier: MIT
"""Z4 cooperative-receiver archive grammar — Z4CR1 monolithic single-file ``0.bin``.

Per CLAUDE.md HNeRV parity discipline L2 (export-first) + L3 (monolithic 0.bin)
+ L4 (≤200 LOC inflate substrate-engineering waiver) + L8 (deterministic).

Z4CR1 is BYTEWISE-COMPATIBLE with the sister C6 IBPS1 grammar (encoder +
decoder + per-pair z) with one differentiating addition: a
``cooperative_receiver_meta`` field in the JSON meta blob that carries
the cooperative-receiver provenance tag (Atick-Redlich form, λ_pixel,
literature anchor) for forensic distinction at audit time. This enables
the bit-level deconstruction tool (CLAUDE.md "Bit-level deconstruction
and entropy discipline") to distinguish Z3 archives from Z4 archives at
section-hash level without reading the bytes.

Grammar:

::

    MAGIC(4)            b"Z4CR"
    VERSION(1)          u8       schema version (currently 1)
    LATENT_DIM(2)       u16      cfg.latent_dim (e.g. 24)
    NUM_PAIRS(2)        u16      cfg.num_pairs (e.g. 600)
    ENCODER_BLOB_LEN(4) u32      brotli-compressed encoder state_dict bytes len
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder state_dict bytes len
    LATENT_BLOB_LEN(4)  u32      int8 latents bytes len (= num_pairs * latent_dim)
    META_BLOB_LEN(4)    u32      sorted-keys JSON utf-8 bytes len
    ENCODER_BLOB        ...      brotli(quality=9) of pickled encoder state_dict (fp16)
    DECODER_BLOB        ...      brotli(quality=9) of pickled decoder state_dict (fp16)
    LATENT_BLOB         ...      int8 latents row-major (num_pairs, latent_dim)
    META_BLOB           ...      sorted-keys JSON with cooperative_receiver_meta tag

Round-trip contract:

    bytes → parse_archive → (encoder_sd, decoder_sd, latents, meta)
    (encoder_sd, decoder_sd, latents, meta) → pack_archive → bytes

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load
- Deterministic: same input → same bytes (sorted-keys JSON; fixed brotli quality;
  fp16 state_dict cast on CPU)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

Z4CR1_MAGIC: bytes = b"Z4CR"
"""Z4 cooperative-receiver variant 1 archive magic."""

Z4CR1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + LATENT_DIM(2) + NUM_PAIRS(2)
#                + ENCODER_LEN(4) + DECODER_LEN(4) + LATENT_LEN(4) + META_LEN(4)
# = 4+1+2+2+4+4+4+4 = 25 bytes
Z4CR1_HEADER_FMT: str = "<4sBHHIIII"
Z4CR1_HEADER_SIZE: int = struct.calcsize(Z4CR1_HEADER_FMT)
assert Z4CR1_HEADER_SIZE == 25, f"header size invariant: expected 25, got {Z4CR1_HEADER_SIZE}"

# Deterministic brotli quality (matches sister substrates).
_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class CooperativeReceiverArchive:
    """Parsed Z4CR1 archive — the inflate-time data contract."""

    encoder_state_dict: dict[str, torch.Tensor]
    """Encoder state_dict (provenance; not strictly required at inflate)."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (the inflate-time consumer)."""

    latents: torch.Tensor
    """``(num_pairs, latent_dim)`` float32 dequantized per-pair z."""

    meta: dict[str, object]
    """Sidecar JSON meta with cooperative_receiver_meta provenance fields."""

    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize state_dict deterministically as length-prefixed records.

    Format per entry (sorted by key for determinism):

        u16 key_len + key_bytes (utf-8)
        u8 ndim
        u32 shape[0..ndim-1] (little-endian)
        fp16 tensor bytes (row-major contiguous)

    Returns brotli-compressed concatenation. Pickle is avoided because
    pickle's tensor memo IDs vary across instantiations.
    """
    parts: list[bytes] = []
    for key in sorted(sd.keys()):
        tensor = sd[key].detach().to("cpu", dtype=torch.float16).contiguous()
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise ValueError(f"key {key!r} is too long for u16 length")
        shape = tuple(int(s) for s in tensor.shape)
        if len(shape) > 0xFF:
            raise ValueError(f"tensor {key!r} has too many dims for u8 ndim")
        for dim in shape:
            if dim < 0 or dim > 0xFFFFFFFF:
                raise ValueError(f"tensor {key!r} dim {dim} out of u32 range")
        header_fmt = f"<H{len(key_bytes)}sB" + "I" * len(shape)
        header = struct.pack(header_fmt, len(key_bytes), key_bytes, len(shape), *shape)
        parts.append(header)
        parts.append(tensor.numpy().tobytes(order="C"))
    raw = b"".join(parts)
    return bytes(brotli.compress(raw, quality=_BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
    pos = 0
    import numpy as np  # local import to keep hot path light

    while pos < len(raw):
        if pos + 2 > len(raw):
            raise ValueError("state_dict blob truncated reading key length")
        (key_len,) = struct.unpack("<H", raw[pos : pos + 2])
        pos += 2
        if pos + key_len + 1 > len(raw):
            raise ValueError("state_dict blob truncated reading key bytes")
        key = raw[pos : pos + key_len].decode("utf-8")
        pos += key_len
        (ndim,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        if pos + ndim * 4 > len(raw):
            raise ValueError("state_dict blob truncated reading shape")
        shape = struct.unpack("<" + "I" * ndim, raw[pos : pos + ndim * 4])
        pos += ndim * 4
        numel = 1
        for dim in shape:
            numel *= int(dim)
        tensor_bytes = numel * 2  # fp16
        if pos + tensor_bytes > len(raw):
            raise ValueError(f"state_dict blob truncated reading tensor {key!r}")
        arr = np.frombuffer(
            raw[pos : pos + tensor_bytes], dtype=np.float16
        ).reshape(shape).astype(np.float16, copy=True)
        sd[key] = torch.from_numpy(arr)
        pos += tensor_bytes
    if pos != len(raw):
        raise ValueError(f"state_dict blob has trailing bytes (pos={pos} len={len(raw)})")
    return sd


def _quantize_latents_to_int8(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize latents to int8 (saves 50% vs int16; total ~24 B/pair × 600 = 14.4 KB max).

    Returns (q_int8, scale, zero_point) so dequant is: ``f = (q + 127) * scale + zero_point``.

    Degenerate (all-equal) case: ``hi <= lo``. Per Catalog #161 (the canonical
    degenerate-range fix landed via sane_hnerv + block_nerv), q is filled
    with ``-(MAX_LEVELS // 2) = -127`` so ``dequant(-127) = 0 * scale + lo = lo``.
    """
    if latents.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"latents must be float; got {latents.dtype}")
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (
            torch.full_like(f, -127, dtype=torch.int8),
            1.0,
            lo,
        )
    # Map [lo, hi] → [0, 254] → [-127, 127]
    scale = (hi - lo) / 254.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 254.0)
    q = (q_unsigned - 127.0).to(torch.int8)
    return (q, scale, lo)


def _dequantize_latents(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 127.0
    return q_unsigned * float(scale) + float(zero_point)


def _make_cooperative_receiver_meta_block(
    base_meta: dict[str, object],
    *,
    cooperative_receiver_lambda_pixel: float,
    cooperative_receiver_atick_redlich_form: bool,
) -> dict[str, object]:
    """Inject the cooperative-receiver provenance tag into the meta dict.

    The tag is structured so future audit tools can distinguish a Z4 archive
    from a Z3 archive at meta-blob hash level even if encoder/decoder/latent
    blobs happen to coincide.
    """
    meta = dict(base_meta)
    meta["cooperative_receiver_meta"] = {
        "lambda_pixel": float(cooperative_receiver_lambda_pixel),
        "atick_redlich_form": bool(cooperative_receiver_atick_redlich_form),
        "literature_anchor": "Atick-Redlich1990",
        "staircase_step": 2,
        "predicted_band_lo": 0.180,
        "predicted_band_hi": 0.188,
    }
    return meta


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = Z4CR1_SCHEMA_VERSION,
    cooperative_receiver_lambda_pixel: float = 0.0,
    cooperative_receiver_atick_redlich_form: bool = True,
) -> bytes:
    """Serialize trained substrate weights + latents + meta into Z4CR1 0.bin bytes."""
    if schema_version != Z4CR1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")

    q_latents, scale, zero_point = _quantize_latents_to_int8(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    encoder_blob = _serialize_state_dict(encoder_state_dict)
    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_tag = _make_cooperative_receiver_meta_block(
        meta,
        cooperative_receiver_lambda_pixel=cooperative_receiver_lambda_pixel,
        cooperative_receiver_atick_redlich_form=cooperative_receiver_atick_redlich_form,
    )
    meta_with_tag["_lat_scale"] = float(scale)
    meta_with_tag["_lat_zero_point"] = float(zero_point)
    meta_bytes = json.dumps(
        meta_with_tag, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        Z4CR1_HEADER_FMT,
        Z4CR1_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        len(encoder_blob),
        len(decoder_blob),
        len(latent_bytes),
        len(meta_bytes),
    )
    return header + encoder_blob + decoder_blob + latent_bytes + meta_bytes


def parse_archive(blob: bytes) -> CooperativeReceiverArchive:
    """Parse Z4CR1 0.bin bytes back into encoder + decoder + latents + meta."""
    if len(blob) < Z4CR1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {Z4CR1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        encoder_len,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(Z4CR1_HEADER_FMT, blob[:Z4CR1_HEADER_SIZE])
    if magic != Z4CR1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {Z4CR1_MAGIC!r})")
    if version != Z4CR1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim  # int8 = 1 byte
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim = {expected_latent_bytes}"
        )

    end_header = Z4CR1_HEADER_SIZE
    end_encoder = end_header + encoder_len
    end_decoder = end_encoder + decoder_len
    end_latents = end_decoder + latent_len
    end_meta = end_latents + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    encoder_blob = blob[end_header:end_encoder]
    decoder_blob = blob[end_encoder:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    meta_blob = blob[end_latents:end_meta]

    encoder_sd = _deserialize_state_dict(encoder_blob)
    decoder_sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import — keep import-time hot path light

    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int8).copy()
    ).view(num_pairs, latent_dim)
    scale = float(meta.pop("_lat_scale"))
    zp = float(meta.pop("_lat_zero_point"))
    latents = _dequantize_latents(q_latents, scale, zp)

    return CooperativeReceiverArchive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        latents=latents,
        meta=meta,
        schema_version=int(version),
    )


__all__ = [
    "CooperativeReceiverArchive",
    "Z4CR1_HEADER_FMT",
    "Z4CR1_HEADER_SIZE",
    "Z4CR1_MAGIC",
    "Z4CR1_SCHEMA_VERSION",
    "pack_archive",
    "parse_archive",
]
