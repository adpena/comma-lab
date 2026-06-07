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
    LATENT_C_LEN(4)     u32      encoded coarse latents bytes len
    LATENT_M_LEN(4)     u32      encoded mid latents bytes len
    LATENT_F_LEN(4)     u32      encoded fine latents bytes len
    META_BLOB_LEN(4)    u32      utf-8 json meta bytes len
    DECODER_BLOB        ...      brotli(quality=9) of pickled state_dict
    LATENT_C_BLOB       ...      quantized coarse latents
    LATENT_M_BLOB       ...      quantized mid latents
    LATENT_F_BLOB       ...      quantized fine latents
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

import base64
import hashlib
import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

from tac.substrates._shared.decoder_state_codec import (
    decoder_state_codec_stats,
    deserialize_decoder_state_dict,
    serialize_decoder_state_dict,
)
from tac.substrates.hi_nerv.target_region_actions import (
    TARGET_REGION_ACTION_META_KEY,
    decode_target_region_actions_from_meta,
    target_region_action_payload_codec,
    target_region_action_section_telemetry,
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
LATENT_CODEC_HI_AC_INT16_Q11: str = "int16_hi_ac_brotli_q11"
LATENT_CODEC_RAW_INT8: str = "int8_raw"
LATENT_CODEC_BROTLI_INT8_Q11: str = "int8_brotli_q11"
LATENT_CODEC_PACKED_INT4: str = "int4_packed"
LATENT_CODEC_PACKED_BROTLI_INT4_Q11: str = "int4_packed_brotli_q11"
LATENT_CODEC_PACKED_INT2: str = "int2_packed"
LATENT_CODEC_PACKED_BROTLI_INT2_Q11: str = "int2_packed_brotli_q11"
SUPPORTED_LATENT_CODECS: tuple[str, ...] = (
    LATENT_CODEC_RAW_INT16,
    LATENT_CODEC_BROTLI_INT16_Q11,
    LATENT_CODEC_HI_AC_INT16_Q11,
    LATENT_CODEC_RAW_INT8,
    LATENT_CODEC_BROTLI_INT8_Q11,
    LATENT_CODEC_PACKED_INT4,
    LATENT_CODEC_PACKED_BROTLI_INT4_Q11,
    LATENT_CODEC_PACKED_INT2,
    LATENT_CODEC_PACKED_BROTLI_INT2_Q11,
)
LATENT_HI_AC_MAGIC: bytes = b"HILA1"
LATENT_HI_AC_HEADER_FMT: str = "<5sIIII"
LATENT_HI_AC_HEADER_SIZE: int = struct.calcsize(LATENT_HI_AC_HEADER_FMT)
assert LATENT_HI_AC_HEADER_SIZE == 21, "HiNeRV latent hi-ac header invariant"
HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA: str = (
    "hinerv_archive_section_telemetry.v1"
)
_RECEIVER_DERIVABLE_META_KEYS: tuple[str, ...] = (
    "_latent_codec_lossless",
    "_latent_raw_bytes_coarse",
    "_latent_raw_bytes_mid",
    "_latent_raw_bytes_fine",
    "_latent_coded_bytes_coarse",
    "_latent_coded_bytes_mid",
    "_latent_coded_bytes_fine",
    "_decoder_state_codec",
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


@dataclass(frozen=True)
class _Hiv1Layout:
    schema_version: int
    latent_dim_coarse: int
    latent_dim_mid: int
    latent_dim_fine: int
    num_pairs: int
    decoder_range: tuple[int, int]
    latents_coarse_range: tuple[int, int]
    latents_mid_range: tuple[int, int]
    latents_fine_range: tuple[int, int]
    meta_range: tuple[int, int]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strip_receiver_derivable_meta(meta: dict[str, object]) -> dict[str, object]:
    """Drop report fields the receiver can derive from header/section bytes."""

    compact = dict(meta)
    for key in _RECEIVER_DERIVABLE_META_KEYS:
        compact.pop(key, None)
    return compact


def _read_hiv1_layout(blob: bytes) -> _Hiv1Layout:
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
    return _Hiv1Layout(
        schema_version=int(version),
        latent_dim_coarse=int(dim_c),
        latent_dim_mid=int(dim_m),
        latent_dim_fine=int(dim_f),
        num_pairs=int(num_pairs),
        decoder_range=(end_header, end_decoder),
        latents_coarse_range=(end_decoder, end_lat_c),
        latents_mid_range=(end_lat_c, end_lat_m),
        latents_fine_range=(end_lat_m, end_lat_f),
        meta_range=(end_lat_f, end_meta),
    )


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


def _latent_codec_quant_bits(codec: str) -> int:
    normalized = str(codec)
    if normalized in (
        LATENT_CODEC_RAW_INT16,
        LATENT_CODEC_BROTLI_INT16_Q11,
        LATENT_CODEC_HI_AC_INT16_Q11,
    ):
        return 16
    if normalized in (LATENT_CODEC_RAW_INT8, LATENT_CODEC_BROTLI_INT8_Q11):
        return 8
    if normalized in (
        LATENT_CODEC_PACKED_INT4,
        LATENT_CODEC_PACKED_BROTLI_INT4_Q11,
    ):
        return 4
    if normalized in (
        LATENT_CODEC_PACKED_INT2,
        LATENT_CODEC_PACKED_BROTLI_INT2_Q11,
    ):
        return 2
    valid = ", ".join(SUPPORTED_LATENT_CODECS)
    raise ValueError(
        f"unsupported HiNeRV latent codec {normalized!r}; expected one of {valid}"
    )


def _latent_unsigned_max(bits: int) -> int:
    if int(bits) not in (2, 4, 8, 16):
        raise ValueError(f"unsupported latent quant bits {bits}")
    return (1 << int(bits)) - 2


def _latent_unwrapped_byte_count(n_symbols: int, *, codec: str) -> int:
    bits = _latent_codec_quant_bits(codec)
    if bits == 16:
        return int(n_symbols) * 2
    if bits == 8:
        return int(n_symbols)
    return (int(n_symbols) * bits + 7) // 8


def _quantize_latents_to_unsigned(
    latents: torch.Tensor,
    *,
    bits: int,
) -> tuple[np.ndarray, float, float]:
    if latents.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"latents must be float; got {latents.dtype}")
    quant_bits = int(bits)
    max_unsigned = _latent_unsigned_max(quant_bits)
    f = latents.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        q = torch.zeros_like(f, dtype=torch.int32)
        return q.numpy().astype(np.uint16 if quant_bits == 16 else np.uint8), 1.0, lo
    scale = (hi - lo) / float(max_unsigned)
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, float(max_unsigned))
    q_np = q_unsigned.to(torch.int32).numpy()
    dtype = np.uint16 if quant_bits == 16 else np.uint8
    return q_np.astype(dtype, copy=False), scale, lo


def _dequantize_unsigned_latents(
    q_unsigned: torch.Tensor,
    scale: float,
    zero_point: float,
) -> torch.Tensor:
    q_unsigned = q_unsigned.to(torch.float32)
    return q_unsigned * float(scale) + float(zero_point)


def _pack_unsigned_values(values: np.ndarray, *, bits: int) -> bytes:
    quant_bits = int(bits)
    flat = np.asarray(values).reshape(-1).astype(np.uint16, copy=False)
    max_unsigned = _latent_unsigned_max(quant_bits)
    if flat.size == 0:
        raise ValueError("HiNeRV latent quant stream must be non-empty")
    if int(flat.max(initial=0)) > max_unsigned:
        raise ValueError(
            f"latent quant value exceeds {quant_bits}-bit unsigned max {max_unsigned}"
        )
    if quant_bits == 16:
        signed = flat.astype(np.int32) - 32767
        return signed.astype("<i2", copy=False).tobytes()
    if quant_bits == 8:
        return flat.astype(np.uint8, copy=False).tobytes()
    per_byte = 8 // quant_bits
    pad = (-int(flat.size)) % per_byte
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    packed = np.zeros(int(flat.size) // per_byte, dtype=np.uint8)
    mask = (1 << quant_bits) - 1
    lanes = flat.reshape(-1, per_byte).astype(np.uint8, copy=False)
    for lane in range(per_byte):
        packed |= (lanes[:, lane] & mask) << (lane * quant_bits)
    return packed.tobytes()


def _unpack_unsigned_values(
    payload: bytes,
    *,
    bits: int,
    n_symbols: int,
    name: str,
) -> np.ndarray:
    quant_bits = int(bits)
    expected_symbols = int(n_symbols)
    if expected_symbols <= 0:
        raise ValueError(f"{name} expected symbol count must be positive")
    if quant_bits == 16:
        expected = expected_symbols * 2
        if len(payload) != expected:
            raise ValueError(f"{name} decoded int16 bytes {len(payload)} != {expected}")
        signed = np.frombuffer(payload, dtype="<i2").astype(np.int32)
        return (signed + 32767).astype(np.uint16, copy=False)
    if quant_bits == 8:
        if len(payload) != expected_symbols:
            raise ValueError(
                f"{name} decoded int8 bytes {len(payload)} != {expected_symbols}"
            )
        return np.frombuffer(payload, dtype=np.uint8).copy()
    per_byte = 8 // quant_bits
    expected_bytes = (expected_symbols * quant_bits + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{name} decoded packed int{quant_bits} bytes {len(payload)} != "
            f"{expected_bytes}"
        )
    packed = np.frombuffer(payload, dtype=np.uint8)
    mask = (1 << quant_bits) - 1
    out = np.empty(len(packed) * per_byte, dtype=np.uint8)
    for lane in range(per_byte):
        out[lane::per_byte] = (packed >> (lane * quant_bits)) & mask
    return out[:expected_symbols].copy()


def _encode_quantized_latents(
    latents: torch.Tensor,
    *,
    codec: str,
) -> tuple[bytes, float, float]:
    bits = _latent_codec_quant_bits(codec)
    q_unsigned, scale, zero_point = _quantize_latents_to_unsigned(latents, bits=bits)
    raw = _pack_unsigned_values(q_unsigned, bits=bits)
    return _encode_latent_blob(raw, codec=codec), scale, zero_point


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

    bytes_c, sc_c, zp_c = _encode_quantized_latents(
        latents_coarse,
        codec=latent_codec,
    )
    bytes_m, sc_m, zp_m = _encode_quantized_latents(
        latents_mid,
        codec=latent_codec,
    )
    bytes_f, sc_f, zp_f = _encode_quantized_latents(
        latents_fine,
        codec=latent_codec,
    )

    decoder_blob = _serialize_state_dict(decoder_state_dict, codec=decoder_codec)

    meta_with_quant = dict(meta)
    meta_with_quant["_quant_scale_coarse"] = float(sc_c)
    meta_with_quant["_quant_zero_point_coarse"] = float(zp_c)
    meta_with_quant["_quant_scale_mid"] = float(sc_m)
    meta_with_quant["_quant_zero_point_mid"] = float(zp_m)
    meta_with_quant["_quant_scale_fine"] = float(sc_f)
    meta_with_quant["_quant_zero_point_fine"] = float(zp_f)
    meta_with_quant["_latent_codec"] = str(latent_codec)
    meta_bytes = json.dumps(
        _strip_receiver_derivable_meta(meta_with_quant),
        separators=(",", ":"),
        sort_keys=True,
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

    layout = _read_hiv1_layout(blob)
    decoder_start, decoder_end = layout.decoder_range
    lat_c_start, lat_c_end = layout.latents_coarse_range
    lat_m_start, lat_m_end = layout.latents_mid_range
    lat_f_start, lat_f_end = layout.latents_fine_range
    meta_start, meta_end = layout.meta_range
    return HinervArchiveSections(
        decoder_blob=blob[decoder_start:decoder_end],
        latents_coarse_blob=blob[lat_c_start:lat_c_end],
        latents_mid_blob=blob[lat_m_start:lat_m_end],
        latents_fine_blob=blob[lat_f_start:lat_f_end],
        meta=json.loads(blob[meta_start:meta_end].decode("utf-8")),
        schema_version=layout.schema_version,
        latent_dim_coarse=layout.latent_dim_coarse,
        latent_dim_mid=layout.latent_dim_mid,
        latent_dim_fine=layout.latent_dim_fine,
        num_pairs=layout.num_pairs,
    )


def build_archive_section_telemetry(
    blob: bytes,
    *,
    archive_zip_bytes: int | None = None,
) -> dict[str, object]:
    """Return exact HIV1 section bytes for byte-cap/modelsize controllers."""

    layout = _read_hiv1_layout(blob)
    meta_payload = blob[layout.meta_range[0] : layout.meta_range[1]]
    meta = json.loads(meta_payload.decode("utf-8"))
    latent_codec = str(meta.get("_latent_codec", LATENT_CODEC_RAW_INT16))
    decoder_codec_name = decoder_state_codec_stats(
        blob[layout.decoder_range[0] : layout.decoder_range[1]]
    ).codec
    blockers: list[str] = []
    target_region_actions: dict[str, object] | None = None
    if TARGET_REGION_ACTION_META_KEY in meta:
        try:
            actions = decode_target_region_actions_from_meta(meta)
            target_region_actions = target_region_action_section_telemetry(actions)
            raw_b64 = meta.get(TARGET_REGION_ACTION_META_KEY)
            if isinstance(raw_b64, str):
                stored_payload = base64.b64decode(raw_b64.encode("ascii"), validate=True)
                target_region_actions["payload_bytes"] = len(stored_payload)
                target_region_actions["payload_codec"] = target_region_action_payload_codec(
                    stored_payload
                )
                target_region_actions["base64_text_bytes"] = len(raw_b64.encode("ascii"))
            else:
                target_region_actions["base64_text_bytes"] = None
            target_region_actions["charged_meta_json_bytes"] = len(meta_payload)
        except ValueError as exc:
            blockers.append(f"target_region_action_meta_invalid:{exc}")

    def _section_row(
        *,
        name: str,
        role: str,
        byte_range: tuple[int, int],
        codec: str | None = None,
        scale: str | None = None,
        raw_bytes: int | None = None,
    ) -> dict[str, object]:
        start, end = byte_range
        payload = blob[start:end]
        row: dict[str, object] = {
            "name": name,
            "role": role,
            "offset": int(start),
            "end_offset": int(end),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        if codec is not None:
            row["codec"] = codec
        if scale is not None:
            row["scale"] = scale
        if raw_bytes is not None:
            row["raw_bytes"] = int(raw_bytes)
            row["coded_to_raw_ratio"] = (
                None
                if int(raw_bytes) <= 0
                else float(len(payload)) / float(raw_bytes)
            )
        return row

    latent_rows = [
        _section_row(
            name="latents_coarse",
            role="latent",
            byte_range=layout.latents_coarse_range,
            codec=latent_codec,
            scale="coarse",
            raw_bytes=_latent_unwrapped_byte_count(
                int(layout.num_pairs) * int(layout.latent_dim_coarse),
                codec=latent_codec,
            ),
        ),
        _section_row(
            name="latents_mid",
            role="latent",
            byte_range=layout.latents_mid_range,
            codec=latent_codec,
            scale="mid",
            raw_bytes=_latent_unwrapped_byte_count(
                int(layout.num_pairs) * int(layout.latent_dim_mid),
                codec=latent_codec,
            ),
        ),
        _section_row(
            name="latents_fine",
            role="latent",
            byte_range=layout.latents_fine_range,
            codec=latent_codec,
            scale="fine",
            raw_bytes=_latent_unwrapped_byte_count(
                int(layout.num_pairs) * int(layout.latent_dim_fine),
                codec=latent_codec,
            ),
        ),
    ]
    latent_quant_bits = _latent_codec_quant_bits(latent_codec)
    for row in latent_rows:
        row["quant_bits"] = int(latent_quant_bits)
    sections: list[dict[str, object]] = [
        _section_row(
            name="hiv1_header",
            role="header",
            byte_range=(0, HIV1_HEADER_SIZE),
        ),
        _section_row(
            name="decoder_state",
            role="decoder",
            byte_range=layout.decoder_range,
            codec=decoder_codec_name,
        ),
        *latent_rows,
        _section_row(
            name="meta_json",
            role="metadata",
            byte_range=layout.meta_range,
            codec="utf8_json",
        ),
    ]
    section_payload_bytes = sum(int(row["bytes"]) for row in sections)
    dominant_sections = sorted(
        sections,
        key=lambda row: int(row["bytes"]),
        reverse=True,
    )
    payload: dict[str, object] = {
        "schema": HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA,
        "profile_ready": True,
        "archive_payload_kind": "hiv1_monolithic_0_bin",
        "hiv1_schema_version": layout.schema_version,
        "num_pairs": layout.num_pairs,
        "latent_dims": {
            "coarse": layout.latent_dim_coarse,
            "mid": layout.latent_dim_mid,
            "fine": layout.latent_dim_fine,
        },
        "decoder_codec": decoder_codec_name,
        "latent_codec": latent_codec,
        "hprc_bin_bytes": len(blob),
        "inner_payload_bytes": len(blob),
        "section_payload_bytes": int(section_payload_bytes),
        "sections": sections,
        "dominant_sections": dominant_sections[:4],
        "blockers": blockers,
    }
    if target_region_actions is not None:
        payload["target_region_actions"] = target_region_actions
    if archive_zip_bytes is not None:
        overhead = int(archive_zip_bytes) - len(blob)
        payload["archive_zip_bytes"] = int(archive_zip_bytes)
        payload["archive_zip_overhead_bytes"] = int(overhead)
        payload["archive_zip_overhead_fraction"] = (
            None
            if int(archive_zip_bytes) <= 0
            else float(overhead) / float(archive_zip_bytes)
        )
        payload["sections_with_zip_overhead"] = [
            *sections,
            {
                "name": "archive_zip_overhead",
                "role": "container_overhead",
                "bytes": int(overhead),
                "codec": "zip_runtime_container",
            },
        ]
    return payload


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
    if extra_meta:
        meta.update(extra_meta)
    meta_bytes = json.dumps(
        _strip_receiver_derivable_meta(meta),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
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

    latent_codec = str(meta.get("_latent_codec", LATENT_CODEC_RAW_INT16))

    def _decode_latent(buf: bytes, np_dim: int, lat_dim: int, name: str) -> torch.Tensor:
        expected_symbols = int(np_dim) * int(lat_dim)
        raw = _decode_latent_blob(
            buf,
            codec=latent_codec,
            expected_raw_bytes=_latent_unwrapped_byte_count(
                expected_symbols,
                codec=latent_codec,
            ),
            name=name,
        )
        quant_bits = _latent_codec_quant_bits(latent_codec)
        q_unsigned = _unpack_unsigned_values(
            raw,
            bits=quant_bits,
            n_symbols=expected_symbols,
            name=name,
        )
        return torch.from_numpy(q_unsigned.astype(np.float32, copy=False)).view(
            np_dim,
            lat_dim,
        )

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
    meta.pop("_decoder_state_codec", None)

    return HinervArchive(
        decoder_state_dict=sd,
        latents_coarse=_dequantize_unsigned_latents(qc, sc_c, zp_c),
        latents_mid=_dequantize_unsigned_latents(qm, sc_m, zp_m),
        latents_fine=_dequantize_unsigned_latents(qf, sc_f, zp_f),
        meta=meta,
        schema_version=int(version),
    )


def _encode_latent_blob(raw: bytes, *, codec: str) -> bytes:
    normalized = str(codec)
    if normalized in (
        LATENT_CODEC_RAW_INT16,
        LATENT_CODEC_RAW_INT8,
        LATENT_CODEC_PACKED_INT4,
        LATENT_CODEC_PACKED_INT2,
    ):
        return bytes(raw)
    if normalized in (
        LATENT_CODEC_BROTLI_INT16_Q11,
        LATENT_CODEC_BROTLI_INT8_Q11,
        LATENT_CODEC_PACKED_BROTLI_INT4_Q11,
        LATENT_CODEC_PACKED_BROTLI_INT2_Q11,
    ):
        return bytes(brotli.compress(raw, quality=11))
    if normalized == LATENT_CODEC_HI_AC_INT16_Q11:
        return _encode_latent_hi_ac_blob(raw)
    valid = ", ".join(SUPPORTED_LATENT_CODECS)
    raise ValueError(
        f"unsupported HiNeRV latent codec {normalized!r}; expected one of {valid}"
    )


def _decode_latent_blob(
    blob: bytes,
    *,
    codec: str,
    expected_raw_bytes: int,
    name: str,
) -> bytes:
    normalized = str(codec)
    if normalized in (
        LATENT_CODEC_RAW_INT16,
        LATENT_CODEC_RAW_INT8,
        LATENT_CODEC_PACKED_INT4,
        LATENT_CODEC_PACKED_INT2,
    ):
        raw = bytes(blob)
    elif normalized in (
        LATENT_CODEC_BROTLI_INT16_Q11,
        LATENT_CODEC_BROTLI_INT8_Q11,
        LATENT_CODEC_PACKED_BROTLI_INT4_Q11,
        LATENT_CODEC_PACKED_BROTLI_INT2_Q11,
    ):
        raw = bytes(brotli.decompress(blob))
    elif normalized == LATENT_CODEC_HI_AC_INT16_Q11:
        raw = _decode_latent_hi_ac_blob(
            blob,
            expected_raw_bytes=expected_raw_bytes,
            name=name,
        )
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


def _encode_latent_hi_ac_blob(raw: bytes) -> bytes:
    """Encode int16 latent bytes as Brotli low byte + arithmetic high byte."""

    raw_bytes = bytes(raw)
    if not raw_bytes or len(raw_bytes) % 2:
        raise ValueError("HiNeRV latent hi-ac codec requires non-empty int16 bytes")
    words = np.frombuffer(raw_bytes, dtype="<u2")
    lo = (words & 0x00FF).astype(np.uint8)
    hi = ((words >> 8) & 0x00FF).astype(np.uint8)
    hist = np.bincount(hi.astype(np.int32), minlength=256).astype("<u4")
    lo_payload = bytes(brotli.compress(lo.tobytes(), quality=11))
    hist_payload = bytes(brotli.compress(hist.tobytes(), quality=11))
    hi_payload = _encode_high_bytes_arithmetic(hi, hist.astype(np.float64))
    header = struct.pack(
        LATENT_HI_AC_HEADER_FMT,
        LATENT_HI_AC_MAGIC,
        int(words.size),
        len(lo_payload),
        len(hist_payload),
        len(hi_payload),
    )
    return header + lo_payload + hist_payload + hi_payload


def _decode_latent_hi_ac_blob(
    blob: bytes,
    *,
    expected_raw_bytes: int,
    name: str,
) -> bytes:
    if expected_raw_bytes <= 0 or int(expected_raw_bytes) % 2:
        raise ValueError(f"{name} expected_raw_bytes must be positive even int16 bytes")
    if len(blob) < LATENT_HI_AC_HEADER_SIZE:
        raise ValueError(f"{name} latent hi-ac blob too short")
    magic, n_symbols, lo_len, hist_len, hi_len = struct.unpack(
        LATENT_HI_AC_HEADER_FMT,
        blob[:LATENT_HI_AC_HEADER_SIZE],
    )
    if magic != LATENT_HI_AC_MAGIC:
        raise ValueError(f"{name} bad latent hi-ac magic: {magic!r}")
    if int(n_symbols) * 2 != int(expected_raw_bytes):
        raise ValueError(
            f"{name} latent hi-ac symbol count {int(n_symbols)} does not match "
            f"expected raw bytes {int(expected_raw_bytes)}"
        )
    cursor = LATENT_HI_AC_HEADER_SIZE
    end_lo = cursor + int(lo_len)
    end_hist = end_lo + int(hist_len)
    end_hi = end_hist + int(hi_len)
    if end_hi != len(blob):
        raise ValueError(f"{name} latent hi-ac section lengths do not cover blob")
    lo = np.frombuffer(brotli.decompress(blob[cursor:end_lo]), dtype=np.uint8)
    hist_raw = brotli.decompress(blob[end_lo:end_hist])
    if len(hist_raw) != 256 * np.dtype("<u4").itemsize:
        raise ValueError(f"{name} latent hi-ac histogram length mismatch")
    hist = np.frombuffer(hist_raw, dtype="<u4").astype(np.float64)
    if lo.size != int(n_symbols):
        raise ValueError(f"{name} latent hi-ac low-byte count mismatch")
    hi = _decode_high_bytes_arithmetic(
        blob[end_hist:end_hi],
        histogram=hist,
        n_symbols=int(n_symbols),
    )
    words = (
        hi.astype(np.uint16).astype("<u2") << np.uint16(8)
    ) | lo.astype(np.uint16).astype("<u2")
    return words.astype("<u2", copy=False).tobytes()


def _make_latent_categorical(histogram: np.ndarray):
    import constriction

    weights = np.asarray(histogram, dtype=np.float64)
    if weights.shape != (256,):
        raise ValueError(f"latent histogram must have shape (256,), got {weights.shape}")
    weights = np.maximum(weights, 1e-10)
    weights /= weights.sum()
    return constriction.stream.model.Categorical(weights, perfect=False)


def _encode_high_bytes_arithmetic(hi: np.ndarray, histogram: np.ndarray) -> bytes:
    import constriction

    symbols = np.asarray(hi, dtype=np.uint8).astype(np.int32)
    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError("latent hi-byte stream must be non-empty 1D")
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, _make_latent_categorical(histogram))
    return np.asarray(encoder.get_compressed(), dtype=">u4").tobytes()


def _decode_high_bytes_arithmetic(
    payload: bytes,
    *,
    histogram: np.ndarray,
    n_symbols: int,
) -> np.ndarray:
    import constriction

    if n_symbols <= 0:
        raise ValueError(f"n_symbols must be > 0; got {n_symbols}")
    if len(payload) % 4:
        raise ValueError("latent hi-byte arithmetic payload is not uint32 aligned")
    words = np.frombuffer(payload, dtype=">u4").astype(np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    categorical = _make_latent_categorical(histogram)
    hi = np.zeros(int(n_symbols), dtype=np.int32)
    for index in range(int(n_symbols)):
        hi[index] = decoder.decode(categorical)
    return hi.astype(np.uint8)
