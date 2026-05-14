# SPDX-License-Identifier: MIT
"""balle_renderer archive grammar — monolithic single-file ``0.bin`` (β).

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (lesson L2) for β.

Per the council §4.2 β candidate archive-grammar declaration:

::

    MAGIC(4)              b"BRV1"   Balle-Renderer Variant 1
    VERSION(1)            u8        schema version (currently 1)
    LATENT_DIM(2)         u16       cfg.latent_dim
    HYPER_DIM(2)          u16       cfg.hyper_latent_dim
    NUM_PAIRS(2)          u16       cfg.num_pairs
    ENC_BLOB_LEN(4)       u32       brotli-compressed hyper-analysis state_dict bytes len
    DEC_BLOB_LEN(4)       u32       brotli-compressed decoder + heads state_dict bytes len
    HP_BLOB_LEN(4)        u32       brotli-compressed hyper-synthesis + factorized prior bytes len
    LATENTS_BLOB_LEN(4)   u32       raw int16 main-latent bytes len (= num_pairs * latent_dim * 2)
    SCALES_BLOB_LEN(4)    u32       raw int16 hyper-latent w_hat bytes len (= num_pairs * hyper_dim * 2)
    META_BLOB_LEN(4)      u32       utf-8 json meta bytes len
    ENC_BLOB              ...
    DEC_BLOB              ...
    HP_BLOB               ...
    LATENTS_BLOB          ...       int16 main-latent row-major (num_pairs, latent_dim)
    SCALES_BLOB           ...       int16 hyper-latent row-major (num_pairs, hyper_dim)
    META_BLOB             ...

The grammar is FIXED at design-time; mutating it changes the schema VERSION
and requires a new inflate.py.

Round-trip contract (tested in tests/test_balle_renderer_roundtrip.py per
Catalog #91):

    bytes -> parse_archive -> BalleRendererArchive
    BalleRendererArchive components -> pack_archive -> bytes

The parse_archive() return type IS the inflate-time API.

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

BRV1_MAGIC: bytes = b"BRV1"
"""balle_renderer variant 1 archive magic."""

BRV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
#   MAGIC(4) + VERSION(1) + LATENT_DIM(2) + HYPER_DIM(2) + NUM_PAIRS(2)
#   + ENC_LEN(4) + DEC_LEN(4) + HP_LEN(4) + LATENTS_LEN(4) + SCALES_LEN(4)
#   + META_LEN(4)
# Total = 4 + 1 + 2 + 2 + 2 + 4*6 = 35 bytes
BRV1_HEADER_FMT: str = "<4sBHHHIIIIII"
BRV1_HEADER_SIZE: int = struct.calcsize(BRV1_HEADER_FMT)
assert BRV1_HEADER_SIZE == 35, f"header size invariant; got {BRV1_HEADER_SIZE}"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class BalleRendererArchive:
    """Parsed archive structure — the inflate-time data contract.

    Attributes:
        encoder_state_dict: hyper-analysis state_dict (encoder side).
        decoder_state_dict: decoder + RGB heads state_dict.
        hyperprior_state_dict: hyper-synthesis + factorized-prior params
            (w_prior_mean, w_prior_log_scale).
        latents: ``(num_pairs, latent_dim)`` float main latents
            (dequantized from int16 archive bytes).
        scales: ``(num_pairs, hyper_dim)`` float hyper latents w_hat
            (dequantized from int16 archive bytes).
        meta: sidecar JSON meta (config, quant scales/zero-points,
            sin_frequency, decoder_channels, output_h/w, ...).
        schema_version: archive schema version (must == BRV1_SCHEMA_VERSION).
    """

    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]
    hyperprior_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    scales: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 CPU)."""
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _quantize_to_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize ``t`` to int16. Returns ``(q, scale, zero_point)``.

    ``f = (q_int16 + 32767) * scale + zero_point``
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        # FFFF Catalog #158 fix: -32767 fill so dequant = 0*scale + lo = lo
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_from_int16(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    hyperprior_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    scales: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = BRV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize all β substrate components into the monolithic ``0.bin`` bytes.

    The trainer ONLY calls this; everything else (framing, padding,
    section CRCs if added later) is the codec's responsibility, not the
    training loop's.
    """
    if schema_version != BRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
    if scales.dim() != 2:
        raise ValueError(
            f"scales must be 2-D (num_pairs, hyper_dim); got {tuple(scales.shape)}"
        )
    if latents.shape[0] != scales.shape[0]:
        raise ValueError(
            f"num_pairs mismatch: latents {latents.shape[0]} vs scales {scales.shape[0]}"
        )

    num_pairs = int(latents.shape[0])
    latent_dim = int(latents.shape[1])
    hyper_dim = int(scales.shape[1])

    for name, val in (("num_pairs", num_pairs), ("latent_dim", latent_dim), ("hyper_dim", hyper_dim)):
        if val <= 0 or val > 0xFFFF:
            raise ValueError(f"{name}={val} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_to_int16(latents)
    q_sca, sca_scale, sca_zp = _quantize_to_int16(scales)
    latent_bytes = q_lat.contiguous().numpy().tobytes()
    scale_bytes = q_sca.contiguous().numpy().tobytes()

    enc_blob = _serialize_state_dict(encoder_state_dict)
    dec_blob = _serialize_state_dict(decoder_state_dict)
    hp_blob = _serialize_state_dict(hyperprior_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_with_quant["_sca_quant_scale"] = float(sca_scale)
    meta_with_quant["_sca_quant_zero_point"] = float(sca_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        BRV1_HEADER_FMT,
        BRV1_MAGIC,
        schema_version,
        latent_dim,
        hyper_dim,
        num_pairs,
        len(enc_blob),
        len(dec_blob),
        len(hp_blob),
        len(latent_bytes),
        len(scale_bytes),
        len(meta_bytes),
    )
    return header + enc_blob + dec_blob + hp_blob + latent_bytes + scale_bytes + meta_bytes


def parse_archive(blob: bytes) -> BalleRendererArchive:
    """Parse the ``0.bin`` bytes back into the substrate components.

    Pure-bytes function — no model class needed. inflate.py imports this +
    the model class + builds + loads + renders, in ~150 LOC total.
    """
    if len(blob) < BRV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {BRV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        hyper_dim,
        num_pairs,
        enc_len,
        dec_len,
        hp_len,
        latent_len,
        scale_len,
        meta_len,
    ) = struct.unpack(BRV1_HEADER_FMT, blob[:BRV1_HEADER_SIZE])
    if magic != BRV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {BRV1_MAGIC!r})")
    if version != BRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2  # int16 = 2 bytes
    expected_scale_bytes = num_pairs * hyper_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )
    if scale_len != expected_scale_bytes:
        raise ValueError(
            f"scale_len {scale_len} != num_pairs*hyper_dim*2 = {expected_scale_bytes}"
        )

    end_header = BRV1_HEADER_SIZE
    end_enc = end_header + enc_len
    end_dec = end_enc + dec_len
    end_hp = end_dec + hp_len
    end_latents = end_hp + latent_len
    end_scales = end_latents + scale_len
    end_meta = end_scales + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    enc_blob = blob[end_header:end_enc]
    dec_blob = blob[end_enc:end_dec]
    hp_blob = blob[end_dec:end_hp]
    latent_blob = blob[end_hp:end_latents]
    scale_blob = blob[end_latents:end_scales]
    meta_blob = blob[end_scales:end_meta]

    enc_sd = _deserialize_state_dict(enc_blob)
    dec_sd = _deserialize_state_dict(dec_blob)
    hp_sd = _deserialize_state_dict(hp_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import; keep module's import-time light
    q_lat = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    q_sca = torch.from_numpy(
        np.frombuffer(scale_blob, dtype=np.int16).copy()
    ).view(num_pairs, hyper_dim)
    lat_scale = float(meta["_lat_quant_scale"])
    lat_zp = float(meta["_lat_quant_zero_point"])
    sca_scale = float(meta["_sca_quant_scale"])
    sca_zp = float(meta["_sca_quant_zero_point"])
    latents = _dequantize_from_int16(q_lat, lat_scale, lat_zp)
    scales = _dequantize_from_int16(q_sca, sca_scale, sca_zp)

    return BalleRendererArchive(
        encoder_state_dict=enc_sd,
        decoder_state_dict=dec_sd,
        hyperprior_state_dict=hp_sd,
        latents=latents,
        scales=scales,
        meta=meta,
        schema_version=int(version),
    )
