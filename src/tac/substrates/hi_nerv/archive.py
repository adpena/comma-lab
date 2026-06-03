# SPDX-License-Identifier: MIT
"""hi_nerv archive grammar — monolithic single-file ``0.bin`` (HIV1).

Catalog #124 STRICT archive-grammar 8 fields declared in package
``__init__``. Export-first grammar (L2):

::

    MAGIC(4)            b"HIV1"  Hierarchical Variant 1
    VERSION(1)          u8       schema version (currently 1)
    LATENT_DIM_C(2)     u16      cfg.latent_dim_coarse
    LATENT_DIM_M(2)     u16      cfg.latent_dim_mid
    LATENT_DIM_F(2)     u16      cfg.latent_dim_fine
    NUM_PAIRS(2)        u16      cfg.num_pairs
    DECODER_BLOB_LEN(4) u32      brotli-compressed decoder state_dict bytes len
    LATENT_C_LEN(4)     u32      int16 coarse latents bytes len
    LATENT_M_LEN(4)     u32      int16 mid latents bytes len
    LATENT_F_LEN(4)     u32      int16 fine latents bytes len
    META_BLOB_LEN(4)    u32      utf-8 json meta bytes len
    DECODER_BLOB        ...      brotli(quality=9) of pickled state_dict
    LATENT_C_BLOB       ...      int16 coarse latents (num_pairs, latent_dim_coarse)
    LATENT_M_BLOB       ...      int16 mid latents
    LATENT_F_BLOB       ...      int16 fine latents
    META_BLOB           ...      json: {"sin_freq": ..., "decoder_channels": [...], ...}

Header: 4+1+2+2+2+2+4+4+4+4+4 = 33 bytes.

Wait — the design says 26-byte header. The TIGHT version is to pack the 3
latent lengths via a single LATENT_BLOB_LEN that covers all 3 sequentially
+ per-scale meta JSON. Keep the 3 distinct length fields here to honor the
"3 latent pyramid sections" parser-manifest declaration in Catalog #124.
The 33-byte header is the trade-off; the Catalog #124 declaration in the
package docstring is the design declaration that this file IS the
authoritative section grammar.

Catalog #124 parser-section manifest enumerates 7 sections:
- HEADER (33 bytes)
- DECODER_BLOB (brotli decoder)
- LATENT_C_BLOB (coarse latents)
- LATENT_M_BLOB (mid latents)
- LATENT_F_BLOB (fine latents)
- META_BLOB (utf-8 json)
- (implicit 7th: per-quant-scale sidecar inside META)

CLAUDE.md compliance: deterministic, no /tmp, no scorer load.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

from tac.substrates._shared.decoder_state_codec import (
    decoder_state_codec_stats,
    deserialize_decoder_state_dict,
    serialize_decoder_state_dict,
)

HIV1_MAGIC: bytes = b"HIV1"
HIV1_SCHEMA_VERSION: int = 1

# 4+1+2+2+2+2+4+4+4+4+4 = 33 bytes (3 latent scales + decoder + meta lengths)
HIV1_HEADER_FMT: str = "<4sBHHHHIIIII"
HIV1_HEADER_SIZE: int = struct.calcsize(HIV1_HEADER_FMT)
assert HIV1_HEADER_SIZE == 33, "HIV1 header size invariant"

BROTLI_QUALITY: int = 9
LATENT_CODEC_RAW_INT16: str = "int16_raw"
LATENT_CODEC_BROTLI_INT16_Q11: str = "int16_brotli_q11"
SUPPORTED_LATENT_CODECS: tuple[str, ...] = (
    LATENT_CODEC_RAW_INT16,
    LATENT_CODEC_BROTLI_INT16_Q11,
)


@dataclass(frozen=True)
class HinervArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (model weights minus per-pair latents)."""

    latents_coarse: torch.Tensor
    """``(num_pairs, latent_dim_coarse)`` dequantized coarse-scale latents."""

    latents_mid: torch.Tensor
    """``(num_pairs, latent_dim_mid)`` dequantized mid-scale latents."""

    latents_fine: torch.Tensor
    """``(num_pairs, latent_dim_fine)`` dequantized fine-scale latents."""

    meta: dict[str, object]
    schema_version: int


@dataclass(frozen=True)
class HinervArchiveSections:
    """Raw HIV1 sections used for decoder-codec-only repacks."""

    decoder_blob: bytes
    latents_coarse_blob: bytes
    latents_mid_blob: bytes
    latents_fine_blob: bytes
    meta: dict[str, object]
    schema_version: int
    latent_dim_coarse: int
    latent_dim_mid: int
    latent_dim_fine: int
    num_pairs: int


def _serialize_state_dict(
    sd: dict[str, torch.Tensor],
    *,
    codec: str = "fp16_brotli_legacy",
) -> bytes:
    return serialize_decoder_state_dict(sd, codec=codec)


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    return deserialize_decoder_state_dict(blob)


def _quantize_latents_to_int16(
    latents: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
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
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = HIV1_SCHEMA_VERSION,
    decoder_codec: str = "fp16_brotli_legacy",
    latent_codec: str = LATENT_CODEC_RAW_INT16,
) -> bytes:
    """Serialize trained weights + 3-scale latents + meta into 0.bin bytes."""
    if schema_version != HIV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    for name, lat in (
        ("latents_coarse", latents_coarse),
        ("latents_mid", latents_mid),
        ("latents_fine", latents_fine),
    ):
        if lat.dim() != 2:
            raise ValueError(
                f"{name} must be 2-D (num_pairs, latent_dim); got {tuple(lat.shape)}"
            )
    num_pairs = int(latents_coarse.shape[0])
    if not (
        latents_mid.shape[0] == num_pairs and latents_fine.shape[0] == num_pairs
    ):
        raise ValueError("all 3 latent scales must share num_pairs")

    dim_c = int(latents_coarse.shape[1])
    dim_m = int(latents_mid.shape[1])
    dim_f = int(latents_fine.shape[1])
    for name, v in (
        ("num_pairs", num_pairs),
        ("latent_dim_coarse", dim_c),
        ("latent_dim_mid", dim_m),
        ("latent_dim_fine", dim_f),
    ):
        if v <= 0 or v > 0xFFFF:
            raise ValueError(f"{name} {v} out of u16 range")

    qc, sc_c, zp_c = _quantize_latents_to_int16(latents_coarse)
    qm, sc_m, zp_m = _quantize_latents_to_int16(latents_mid)
    qf, sc_f, zp_f = _quantize_latents_to_int16(latents_fine)

    raw_c = qc.contiguous().numpy().tobytes()
    raw_m = qm.contiguous().numpy().tobytes()
    raw_f = qf.contiguous().numpy().tobytes()
    bytes_c = _encode_latent_blob(raw_c, codec=latent_codec)
    bytes_m = _encode_latent_blob(raw_m, codec=latent_codec)
    bytes_f = _encode_latent_blob(raw_f, codec=latent_codec)

    decoder_blob = _serialize_state_dict(decoder_state_dict, codec=decoder_codec)

    meta_with_quant = dict(meta)
    meta_with_quant["_quant_scale_coarse"] = float(sc_c)
    meta_with_quant["_quant_zero_point_coarse"] = float(zp_c)
    meta_with_quant["_quant_scale_mid"] = float(sc_m)
    meta_with_quant["_quant_zero_point_mid"] = float(zp_m)
    meta_with_quant["_quant_scale_fine"] = float(sc_f)
    meta_with_quant["_quant_zero_point_fine"] = float(zp_f)
    meta_with_quant["_latent_codec"] = str(latent_codec)
    meta_with_quant["_latent_codec_lossless"] = True
    meta_with_quant["_latent_raw_bytes_coarse"] = len(raw_c)
    meta_with_quant["_latent_raw_bytes_mid"] = len(raw_m)
    meta_with_quant["_latent_raw_bytes_fine"] = len(raw_f)
    meta_with_quant["_latent_coded_bytes_coarse"] = len(bytes_c)
    meta_with_quant["_latent_coded_bytes_mid"] = len(bytes_m)
    meta_with_quant["_latent_coded_bytes_fine"] = len(bytes_f)
    meta_with_quant["_decoder_state_codec"] = decoder_state_codec_stats(
        decoder_blob
    ).as_dict()
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        HIV1_HEADER_FMT,
        HIV1_MAGIC,
        schema_version,
        dim_c,
        dim_m,
        dim_f,
        num_pairs,
        len(decoder_blob),
        len(bytes_c),
        len(bytes_m),
        len(bytes_f),
        len(meta_bytes),
    )
    return header + decoder_blob + bytes_c + bytes_m + bytes_f + meta_bytes


def split_archive_sections(blob: bytes) -> HinervArchiveSections:
    """Return raw HIV1 sections without dequantizing latent payloads."""

    if len(blob) < HIV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {HIV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        dim_c,
        dim_m,
        dim_f,
        num_pairs,
        decoder_len,
        lat_c_len,
        lat_m_len,
        lat_f_len,
        meta_len,
    ) = struct.unpack(HIV1_HEADER_FMT, blob[:HIV1_HEADER_SIZE])
    if magic != HIV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {HIV1_MAGIC!r})")
    if version != HIV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")
    end_header = HIV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_lat_c = end_decoder + lat_c_len
    end_lat_m = end_lat_c + lat_m_len
    end_lat_f = end_lat_m + lat_f_len
    end_meta = end_lat_f + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )
    return HinervArchiveSections(
        decoder_blob=blob[end_header:end_decoder],
        latents_coarse_blob=blob[end_decoder:end_lat_c],
        latents_mid_blob=blob[end_lat_c:end_lat_m],
        latents_fine_blob=blob[end_lat_m:end_lat_f],
        meta=json.loads(blob[end_lat_f:end_meta].decode("utf-8")),
        schema_version=int(version),
        latent_dim_coarse=int(dim_c),
        latent_dim_mid=int(dim_m),
        latent_dim_fine=int(dim_f),
        num_pairs=int(num_pairs),
    )


def repack_archive_decoder_codec(
    blob: bytes,
    *,
    decoder_codec: str,
    decoder_state_dict: dict[str, torch.Tensor] | None = None,
    extra_meta: dict[str, object] | None = None,
) -> bytes:
    """Rebuild HIV1 bytes with a different decoder codec.

    Latent int16 blobs are copied byte-for-byte so codec sweeps isolate the
    decoder-state bitstream instead of silently requantizing per-pair latents.
    """

    sections = split_archive_sections(blob)
    state = (
        decoder_state_dict
        if decoder_state_dict is not None
        else _deserialize_state_dict(sections.decoder_blob)
    )
    decoder_blob = _serialize_state_dict(state, codec=decoder_codec)
    meta = dict(sections.meta)
    meta["_decoder_state_codec"] = decoder_state_codec_stats(decoder_blob).as_dict()
    if extra_meta:
        meta.update(extra_meta)
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")
    header = struct.pack(
        HIV1_HEADER_FMT,
        HIV1_MAGIC,
        sections.schema_version,
        sections.latent_dim_coarse,
        sections.latent_dim_mid,
        sections.latent_dim_fine,
        sections.num_pairs,
        len(decoder_blob),
        len(sections.latents_coarse_blob),
        len(sections.latents_mid_blob),
        len(sections.latents_fine_blob),
        len(meta_bytes),
    )
    return (
        header
        + decoder_blob
        + sections.latents_coarse_blob
        + sections.latents_mid_blob
        + sections.latents_fine_blob
        + meta_bytes
    )


def parse_archive(blob: bytes) -> HinervArchive:
    if len(blob) < HIV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {HIV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        dim_c,
        dim_m,
        dim_f,
        num_pairs,
        decoder_len,
        lat_c_len,
        lat_m_len,
        lat_f_len,
        meta_len,
    ) = struct.unpack(HIV1_HEADER_FMT, blob[:HIV1_HEADER_SIZE])
    if magic != HIV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {HIV1_MAGIC!r})")
    if version != HIV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    end_header = HIV1_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_lat_c = end_decoder + lat_c_len
    end_lat_m = end_lat_c + lat_m_len
    end_lat_f = end_lat_m + lat_f_len
    end_meta = end_lat_f + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    lat_c_blob = blob[end_decoder:end_lat_c]
    lat_m_blob = blob[end_lat_c:end_lat_m]
    lat_f_blob = blob[end_lat_m:end_lat_f]
    meta_blob = blob[end_lat_f:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np

    latent_codec = str(meta.get("_latent_codec", LATENT_CODEC_RAW_INT16))

    def _decode_latent(buf: bytes, np_dim: int, lat_dim: int, name: str) -> torch.Tensor:
        raw = _decode_latent_blob(
            buf,
            codec=latent_codec,
            expected_raw_bytes=int(np_dim) * int(lat_dim) * 2,
            name=name,
        )
        return torch.from_numpy(
            np.frombuffer(raw, dtype=np.int16).copy()
        ).view(np_dim, lat_dim)

    qc = _decode_latent(lat_c_blob, num_pairs, dim_c, "latents_coarse")
    qm = _decode_latent(lat_m_blob, num_pairs, dim_m, "latents_mid")
    qf = _decode_latent(lat_f_blob, num_pairs, dim_f, "latents_fine")

    sc_c = float(meta.pop("_quant_scale_coarse"))
    zp_c = float(meta.pop("_quant_zero_point_coarse"))
    sc_m = float(meta.pop("_quant_scale_mid"))
    zp_m = float(meta.pop("_quant_zero_point_mid"))
    sc_f = float(meta.pop("_quant_scale_fine"))
    zp_f = float(meta.pop("_quant_zero_point_fine"))
    meta.pop("_latent_codec", None)
    meta.pop("_latent_codec_lossless", None)
    meta.pop("_latent_raw_bytes_coarse", None)
    meta.pop("_latent_raw_bytes_mid", None)
    meta.pop("_latent_raw_bytes_fine", None)
    meta.pop("_latent_coded_bytes_coarse", None)
    meta.pop("_latent_coded_bytes_mid", None)
    meta.pop("_latent_coded_bytes_fine", None)

    return HinervArchive(
        decoder_state_dict=sd,
        latents_coarse=_dequantize_latents(qc, sc_c, zp_c),
        latents_mid=_dequantize_latents(qm, sc_m, zp_m),
        latents_fine=_dequantize_latents(qf, sc_f, zp_f),
        meta=meta,
        schema_version=int(version),
    )


def _encode_latent_blob(raw: bytes, *, codec: str) -> bytes:
    normalized = str(codec)
    if normalized == LATENT_CODEC_RAW_INT16:
        return bytes(raw)
    if normalized == LATENT_CODEC_BROTLI_INT16_Q11:
        return bytes(brotli.compress(raw, quality=11))
    valid = ", ".join(SUPPORTED_LATENT_CODECS)
    raise ValueError(f"unsupported HiNeRV latent codec {normalized!r}; expected one of {valid}")


def _decode_latent_blob(
    blob: bytes,
    *,
    codec: str,
    expected_raw_bytes: int,
    name: str,
) -> bytes:
    normalized = str(codec)
    if normalized == LATENT_CODEC_RAW_INT16:
        raw = bytes(blob)
    elif normalized == LATENT_CODEC_BROTLI_INT16_Q11:
        raw = bytes(brotli.decompress(blob))
    else:
        valid = ", ".join(SUPPORTED_LATENT_CODECS)
        raise ValueError(
            f"unsupported HiNeRV latent codec {normalized!r}; expected one of {valid}"
        )
    if len(raw) != int(expected_raw_bytes):
        raise ValueError(
            f"{name} decoded latent bytes {len(raw)} != expected {int(expected_raw_bytes)} "
            f"for codec {normalized}"
        )
    return raw
