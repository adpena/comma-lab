# SPDX-License-Identifier: MIT
"""Receiver-visible SNeRV archive packet bundling.

This module is intentionally small: it gives SNeRV a deterministic archive
section grammar for the receiver-facing byte streams the advisory already
charges. It does not load scorers and it does not claim full inflate readiness.

Sections are bundled under one header so downstream work can stop treating LF
codes, decoder bytes, and compact L-infinity step maps as disconnected blobs.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.analysis.snerv_step_map_coder import decode_step_maps
from tac.codec.receiver_integer_plane_codec import (
    SPATIAL_DELTA_ZIGZAG_LEB128_CODEC,
    ReceiverIntegerPlaneCodecError,
    canonical_int64_raw,
    decode_spatial_delta_zigzag_leb128_planes,
    encode_spatial_delta_zigzag_leb128_planes,
)
from tac.substrates._shared.int_stream_codec import (
    pack_fixed_width_uints,
    unpack_fixed_width_uints,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    DEFAULT_SNERV_MODEL_SIZE,
    HfGenerationDecoder,
    SnervFrameCode,
    SnervModelSizeConfig,
    decode_frame,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    SNERV_LF_PAYLOAD_INTN_CODEC_PROOF as _SNERV_LF_PAYLOAD_INTN_CODEC_PROOF,
)

SNERV_ARCHIVE_SCHEMA = "snerv_inverse_steg_archive.v1"
SNERV_ARCHIVE_MAGIC = b"SNAR1"
SNERV_LF_QUANT_MAGIC = b"SNQL1"
SNERV_DECODER_MAGIC = b"SNDC1"
SNERV_LF_PAYLOAD_INTN_CODEC_PROOF = _SNERV_LF_PAYLOAD_INTN_CODEC_PROOF
HEADER_LEN_FMT = "<I"
SECTION_ORDER = ("metadata_payload", "lf_payload", "decoder_payload", "step_map_packet")
DECODER_SUBBANDS = ("LH", "HL", "HH")
LF_QUANT_PAYLOAD_SCHEMA_V1 = "snerv_lf_quant_payload.v1"
LF_QUANT_PAYLOAD_SCHEMA_V2 = "snerv_lf_quant_payload.v2"
LF_QUANT_CODEC_INT64_LZMA = "int64_lzma"
LF_QUANT_CODEC_SPATIAL_DELTA_LEB128_LZMA = SPATIAL_DELTA_ZIGZAG_LEB128_CODEC
DECODER_PAYLOAD_V1_SCHEMA = "snerv_decoder_payload.v1"
DECODER_PAYLOAD_V2_SCHEMA = "snerv_decoder_payload.v2"
DECODER_PAYLOAD_V3_SCHEMA = "snerv_decoder_payload.v3"
DECODER_PAYLOAD_LEGACY_CODEC = "float32_lzma"
DECODER_PAYLOAD_MIXED_CODEC = "mixed_magnitude_symmetric"
DECODER_PAYLOAD_QUANTIZED_CODECS = {
    "int8_symmetric": 8,
    "int4_symmetric": 4,
    "int2_symmetric": 2,
}
DECODER_PAYLOAD_MIXED_MODE_TO_CODE = {
    "zero": 0,
    "int2": 1,
    "int4": 2,
    "int8": 3,
    "fp16": 4,
}
DECODER_PAYLOAD_MIXED_CODE_TO_MODE = {
    code: mode for mode, code in DECODER_PAYLOAD_MIXED_MODE_TO_CODE.items()
}


class SnervArchiveError(ValueError):
    """Raised when the SNeRV receiver archive packet is malformed."""


@dataclass(frozen=True)
class SnervArchivePacket:
    """A bundled receiver-visible SNeRV archive packet."""

    packet: bytes
    schema: str
    section_order: tuple[str, ...]
    section_bytes: dict[str, int]
    section_sha256: dict[str, str]
    metadata: dict[str, Any]
    header_bytes: int
    total_bytes: int
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["packet"] = {"bytes": len(self.packet), "sha256": _sha256(self.packet)}
        return d


@dataclass(frozen=True)
class DecodedSnervArchive:
    """Decoded SNeRV archive sections and metadata."""

    schema: str
    section_order: tuple[str, ...]
    sections: dict[str, bytes]
    metadata: dict[str, Any]
    packet_sha256: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def decode_step_maps(self) -> list[np.ndarray]:
        """Decode compact step maps from the bundled receiver packet."""

        return decode_step_maps(self.sections["step_map_packet"])

    def decode_lf_zero_points(self) -> np.ndarray:
        """Decode LF zero-point metadata from the bundled receiver packet."""

        expected = self.metadata.get("lf_plane_count")
        return decode_lf_metadata_payload(
            self.sections["metadata_payload"],
            expected_count=int(expected) if expected is not None else None,
        )

    def decode_lf_quant_planes(self) -> list[np.ndarray]:
        """Decode LF quantized coefficient planes from the bundled receiver packet."""

        return decode_lf_quant_payload(self.sections["lf_payload"])

    def decode_decoder(self) -> HfGenerationDecoder:
        """Decode the shared HF generator from the bundled receiver packet."""

        return decode_decoder_payload(self.sections["decoder_payload"])

    def decode_frame_planes(self, *, clip_to_uint8_range: bool = True) -> list[np.ndarray]:
        """Decode receiver-visible LF planes into ordered reconstructed frames.

        This is the scorer-free inflate primitive for the SNAR1 packet: it consumes
        only archived LF quant planes, archived zero-points, archived compact step
        maps, and the archived HF decoder. The return value is a flat plane list in
        archive order: pair-major, frame-major, channel-major when metadata includes
        the full-frame grouping fields.
        """

        return decode_snerv_archive_frame_planes_from_decoded(
            self,
            clip_to_uint8_range=clip_to_uint8_range,
        )

    def decode_frames(self, *, clip_to_uint8_range: bool = True) -> np.ndarray:
        """Decode a full receiver frame tensor ``(pairs, 2, 3, H, W)`` from SNAR1."""

        return decode_snerv_archive_frames_from_decoded(
            self,
            clip_to_uint8_range=clip_to_uint8_range,
        )


def pack_snerv_archive(
    *,
    metadata_payload: bytes,
    lf_payload: bytes,
    decoder_payload: bytes,
    step_map_packet: bytes,
    metadata: dict[str, Any] | None = None,
) -> SnervArchivePacket:
    """Bundle receiver-visible SNeRV sections into one deterministic packet."""

    sections = {
        "metadata_payload": bytes(metadata_payload),
        "lf_payload": bytes(lf_payload),
        "decoder_payload": bytes(decoder_payload),
        "step_map_packet": bytes(step_map_packet),
    }
    _validate_sections(sections)
    clean_metadata = _jsonable_metadata(metadata or {})
    cursor = 0
    section_headers = []
    payload_parts = []
    section_bytes: dict[str, int] = {}
    section_sha256: dict[str, str] = {}
    for name in SECTION_ORDER:
        blob = sections[name]
        section_headers.append(
            {
                "name": name,
                "offset": cursor,
                "bytes": len(blob),
                "sha256": _sha256(blob),
            }
        )
        payload_parts.append(blob)
        section_bytes[name] = len(blob)
        section_sha256[name] = _sha256(blob)
        cursor += len(blob)
    header = {
        "schema": SNERV_ARCHIVE_SCHEMA,
        "section_order": list(SECTION_ORDER),
        "sections": section_headers,
        "metadata": clean_metadata,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    header_bytes_raw = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    packet = (
        SNERV_ARCHIVE_MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_bytes_raw))
        + header_bytes_raw
        + b"".join(payload_parts)
    )
    return SnervArchivePacket(
        packet=packet,
        schema=SNERV_ARCHIVE_SCHEMA,
        section_order=SECTION_ORDER,
        section_bytes=section_bytes,
        section_sha256=section_sha256,
        metadata=clean_metadata,
        header_bytes=len(SNERV_ARCHIVE_MAGIC)
        + struct.calcsize(HEADER_LEN_FMT)
        + len(header_bytes_raw),
        total_bytes=len(packet),
    )


def unpack_snerv_archive(packet: bytes) -> DecodedSnervArchive:
    """Decode and validate a bundled SNeRV archive packet."""

    packet = bytes(packet)
    if not packet.startswith(SNERV_ARCHIVE_MAGIC):
        raise SnervArchiveError("bad SNeRV archive magic")
    offset = len(SNERV_ARCHIVE_MAGIC)
    if len(packet) < offset + struct.calcsize(HEADER_LEN_FMT):
        raise SnervArchiveError("truncated SNeRV archive header")
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT,
        packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)],
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    if header_end > len(packet):
        raise SnervArchiveError("declared SNeRV archive header exceeds packet size")
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    if header.get("schema") != SNERV_ARCHIVE_SCHEMA:
        raise SnervArchiveError(f"unsupported SNeRV archive schema: {header.get('schema')!r}")
    section_order = tuple(str(v) for v in header.get("section_order", []))
    if section_order != SECTION_ORDER:
        raise SnervArchiveError(f"unexpected section order: {section_order!r}")
    payload = packet[header_end:]
    sections: dict[str, bytes] = {}
    expected_offset = 0
    seen: set[str] = set()
    for row in header.get("sections", []):
        name = str(row["name"])
        if name not in SECTION_ORDER:
            raise SnervArchiveError(f"unknown SNeRV archive section: {name!r}")
        if name in seen:
            raise SnervArchiveError(f"duplicate SNeRV archive section: {name!r}")
        start = int(row["offset"])
        end = start + int(row["bytes"])
        if start != expected_offset:
            raise SnervArchiveError(
                f"SNeRV archive section {name!r} offset {start} != expected {expected_offset}"
            )
        if start < 0 or end > len(payload):
            raise SnervArchiveError(f"SNeRV archive section {name!r} out of range")
        blob = payload[start:end]
        expected_sha = str(row["sha256"])
        if _sha256(blob) != expected_sha:
            raise SnervArchiveError(f"SNeRV archive section {name!r} sha256 mismatch")
        sections[name] = blob
        seen.add(name)
        expected_offset = end
    if expected_offset != len(payload):
        raise SnervArchiveError("SNeRV archive has unreferenced trailing payload bytes")
    if tuple(sections.keys()) != SECTION_ORDER:
        raise SnervArchiveError("SNeRV archive missing required sections")
    return DecodedSnervArchive(
        schema=SNERV_ARCHIVE_SCHEMA,
        section_order=SECTION_ORDER,
        sections=sections,
        metadata=dict(header.get("metadata", {})),
        packet_sha256=_sha256(packet),
    )


def decode_snerv_archive_step_maps(packet: bytes) -> list[np.ndarray]:
    """Convenience helper for receiver-side step-map decode proof."""

    return unpack_snerv_archive(packet).decode_step_maps()


def decode_snerv_archive_frame_planes(
    packet: bytes,
    *,
    clip_to_uint8_range: bool = True,
) -> list[np.ndarray]:
    """Decode all archived SNeRV frame planes without scorer/torch imports."""

    return unpack_snerv_archive(packet).decode_frame_planes(
        clip_to_uint8_range=clip_to_uint8_range,
    )


def decode_snerv_archive_frames(
    packet: bytes,
    *,
    clip_to_uint8_range: bool = True,
) -> np.ndarray:
    """Decode a full ``(n_pairs, 2, 3, H, W)`` receiver tensor from SNAR1 bytes."""

    return unpack_snerv_archive(packet).decode_frames(
        clip_to_uint8_range=clip_to_uint8_range,
    )


def decode_snerv_archive_frame_planes_from_decoded(
    decoded: DecodedSnervArchive,
    *,
    clip_to_uint8_range: bool = True,
) -> list[np.ndarray]:
    """Decode archived LF planes into receiver frames from an unpacked archive."""

    metadata = decoded.metadata
    levels = _metadata_int(metadata, "levels", minimum=1)
    wavelet = _metadata_str(metadata, "wavelet")
    orig_hw = _metadata_hw(metadata)
    lf_planes = decoded.decode_lf_quant_planes()
    zeros = decoded.decode_lf_zero_points()
    step_maps = decoded.decode_step_maps()
    decoder = decoded.decode_decoder()
    _validate_replay_counts(lf_planes, zeros, step_maps)

    out: list[np.ndarray] = []
    for idx, (q, zero, steps) in enumerate(zip(lf_planes, zeros, step_maps, strict=True)):
        if q.shape != steps.shape:
            raise SnervArchiveError(
                f"receiver replay plane {idx} LF shape {q.shape} != step shape {steps.shape}"
            )
        code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=tuple(int(v) for v in q.shape),
            levels=levels,
            wavelet=wavelet,
            orig_hw=orig_hw,
            per_element_steps=steps,
        )
        frame = decode_frame(code, decoder)
        if clip_to_uint8_range:
            frame = np.clip(frame, 0.0, 255.0)
        out.append(np.asarray(frame, dtype=np.float32))
    return out


def decode_snerv_archive_frames_from_decoded(
    decoded: DecodedSnervArchive,
    *,
    clip_to_uint8_range: bool = True,
) -> np.ndarray:
    """Decode and group receiver frames as ``(n_pairs, 2, 3, H, W)``."""

    metadata = decoded.metadata
    n_pairs = _metadata_int(metadata, "n_pairs", minimum=1)
    frames_per_pair = _metadata_int(metadata, "frames_per_pair", default=2, minimum=1)
    channels = _metadata_int(metadata, "channels", default=3, minimum=1)
    h, w = _metadata_hw(metadata)
    planes = decode_snerv_archive_frame_planes_from_decoded(
        decoded,
        clip_to_uint8_range=clip_to_uint8_range,
    )
    expected = n_pairs * frames_per_pair * channels
    if len(planes) != expected:
        raise SnervArchiveError(
            f"receiver replay decoded {len(planes)} planes, expected {expected} "
            f"from n_pairs={n_pairs}, frames_per_pair={frames_per_pair}, channels={channels}"
        )
    arr = np.stack(planes, axis=0)
    return arr.reshape(n_pairs, frames_per_pair, channels, h, w).astype(np.float32)


def encode_lf_metadata_payload(
    *,
    lf_zero_points: list[float] | np.ndarray,
) -> bytes:
    """Encode LF dequant metadata as compact receiver-visible bytes."""

    zeros = np.asarray(lf_zero_points, dtype="<f4").reshape(-1)
    if zeros.size == 0:
        raise SnervArchiveError("lf_zero_points must be non-empty")
    if not np.all(np.isfinite(zeros)):
        raise SnervArchiveError("lf_zero_points must be finite")
    return zeros.tobytes()


def encode_lf_quant_payload(
    lf_quant_planes: list[np.ndarray],
    *,
    codec: str = LF_QUANT_CODEC_INT64_LZMA,
) -> bytes:
    """Encode quantized LF planes as deterministic scorer-free receiver bytes."""

    arrays = [_validate_lf_quant_plane(a) for a in lf_quant_planes]
    if not arrays:
        raise SnervArchiveError("lf_quant_planes must be non-empty")
    normalized = str(codec).strip().lower()
    if normalized in {"v1", "legacy", "int64", "int64_lzma"}:
        return _encode_lf_quant_payload_int64_lzma(arrays)
    if normalized in {
        "v2",
        "spatial_delta",
        "spatial_delta_zigzag_leb128",
        LF_QUANT_CODEC_SPATIAL_DELTA_LEB128_LZMA,
    }:
        return _encode_lf_quant_payload_spatial_delta_leb128_lzma(arrays)
    if normalized == "auto":
        candidates = (
            _encode_lf_quant_payload_int64_lzma(arrays),
            _encode_lf_quant_payload_spatial_delta_leb128_lzma(arrays),
        )
        return min(candidates, key=len)
    raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")


def _encode_lf_quant_payload_int64_lzma(arrays: list[np.ndarray]) -> bytes:
    raw = canonical_int64_raw(arrays)
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": LF_QUANT_PAYLOAD_SCHEMA_V1,
        "codec": LF_QUANT_CODEC_INT64_LZMA,
        "dtype": "int64_le",
        "shapes": [list(a.shape) for a in arrays],
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_LF_QUANT_MAGIC, header, compressed)


def _encode_lf_quant_payload_spatial_delta_leb128_lzma(
    arrays: list[np.ndarray],
) -> bytes:
    payload = encode_spatial_delta_zigzag_leb128_planes(arrays)
    compressed = lzma.compress(
        payload.raw,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )
    header = {
        "schema": LF_QUANT_PAYLOAD_SCHEMA_V2,
        **payload.header,
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_LF_QUANT_MAGIC, header, compressed)


def decode_lf_quant_payload(payload: bytes) -> list[np.ndarray]:
    """Decode LF quantized coefficient planes from receiver payload bytes."""

    header, compressed = _unpack_lf_quant_subpacket(payload)
    schema = str(header.get("schema"))
    codec = str(header.get("codec", LF_QUANT_CODEC_INT64_LZMA))
    if schema == LF_QUANT_PAYLOAD_SCHEMA_V2:
        if codec != LF_QUANT_CODEC_SPATIAL_DELTA_LEB128_LZMA:
            raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")
        return _decode_lf_quant_payload_spatial_delta_leb128_lzma(header, compressed)
    if schema != LF_QUANT_PAYLOAD_SCHEMA_V1:
        raise SnervArchiveError(f"unsupported subpacket schema: {schema!r}")
    if codec != LF_QUANT_CODEC_INT64_LZMA:
        raise SnervArchiveError(f"unsupported LF quant payload codec: {codec!r}")
    raw = lzma.decompress(compressed)
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("LF quant payload raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervArchiveError("LF quant payload raw sha256 mismatch")
    out = []
    cursor = 0
    for shape in header["shapes"]:
        out_shape = tuple(int(v) for v in shape)
        count = int(np.prod(out_shape))
        nbytes = count * np.dtype("<i8").itemsize
        arr = np.frombuffer(raw[cursor : cursor + nbytes], dtype="<i8").copy()
        if arr.size != count:
            raise SnervArchiveError("LF quant payload ended inside a plane")
        out.append(arr.reshape(out_shape))
        cursor += nbytes
    if cursor != len(raw):
        raise SnervArchiveError("LF quant payload has unused raw bytes")
    return out


def inspect_lf_quant_payload_header(payload: bytes) -> dict[str, Any]:
    """Return validated LF payload header metadata without decoding planes."""

    header, body = _unpack_lf_quant_subpacket(payload)
    out = dict(header)
    out["payload_bytes"] = len(body)
    out["section_bytes"] = len(payload)
    return out


def _decode_lf_quant_payload_spatial_delta_leb128_lzma(
    header: dict[str, Any],
    compressed: bytes,
) -> list[np.ndarray]:
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervArchiveError("LF quant payload decompression failed") from exc
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("LF quant payload raw byte count mismatch")
    try:
        return decode_spatial_delta_zigzag_leb128_planes(raw, header=header)
    except ReceiverIntegerPlaneCodecError as exc:
        raise SnervArchiveError(str(exc)) from exc


def encode_decoder_payload(
    decoder: HfGenerationDecoder,
    *,
    codec: str = DECODER_PAYLOAD_LEGACY_CODEC,
    mixed_modes: Sequence[str] | None = None,
) -> bytes:
    """Encode the shared HF decoder as deterministic scorer-free receiver bytes."""

    levels, values, model_size = _decoder_to_flat_values(decoder)
    raw = values.astype("<f4").tobytes()
    if not raw:
        raise SnervArchiveError("decoder payload must be non-empty")
    normalized = str(codec).strip().lower()
    if normalized in {
        DECODER_PAYLOAD_LEGACY_CODEC,
        "fp32_lzma",
        "float32",
        "legacy",
    }:
        if mixed_modes is not None:
            raise SnervArchiveError("mixed decoder modes require mixed codec")
        return _encode_decoder_payload_v1(
            levels=levels,
            raw=raw,
            model_size=model_size,
        )
    if normalized in DECODER_PAYLOAD_QUANTIZED_CODECS:
        if mixed_modes is not None:
            raise SnervArchiveError("mixed decoder modes require mixed codec")
        return _encode_decoder_payload_quantized(
            levels=levels,
            values=values,
            model_size=model_size,
            bits=DECODER_PAYLOAD_QUANTIZED_CODECS[normalized],
            codec=normalized,
            raw_reference=raw,
        )
    if normalized in {
        DECODER_PAYLOAD_MIXED_CODEC,
        "mixed_per_kernel_symmetric",
        "mixed_symmetric",
    }:
        return _encode_decoder_payload_mixed(
            levels=levels,
            values=values,
            model_size=model_size,
            raw_reference=raw,
            explicit_modes=mixed_modes,
        )
    raise SnervArchiveError(f"unsupported decoder payload codec: {codec!r}")


def _encode_decoder_payload_v1(
    *,
    levels: int,
    raw: bytes,
    model_size: SnervModelSizeConfig,
) -> bytes:
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": DECODER_PAYLOAD_V1_SCHEMA,
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": _decoder_kernel_shape_header(model_size),
        "feature_count": int(model_size.feature_count),
        "model_size_config": model_size.as_jsonable(),
        "dtype": "float32_le",
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, compressed)


def _encode_decoder_payload_quantized(
    *,
    levels: int,
    values: np.ndarray,
    model_size: SnervModelSizeConfig,
    bits: int,
    codec: str,
    raw_reference: bytes,
) -> bytes:
    qmax = (1 << (int(bits) - 1)) - 1
    if qmax < 1:
        raise SnervArchiveError(f"invalid decoder quantizer bits: {bits}")
    feature_count = int(model_size.feature_count)
    value_groups = values.reshape(levels * len(DECODER_SUBBANDS), feature_count)
    scales = []
    unsigned_parts = []
    max_abs_error = 0.0
    mean_abs_errors = []
    for group in value_groups:
        max_abs = float(np.max(np.abs(group))) if group.size else 0.0
        scale = 1.0 if max_abs == 0.0 else max_abs / float(qmax)
        q_signed = np.round(group / scale).clip(-qmax, qmax).astype(np.int64)
        dequant = q_signed.astype(np.float64) * scale
        err = np.abs(dequant - group)
        max_abs_error = max(max_abs_error, float(np.max(err)) if err.size else 0.0)
        mean_abs_errors.append(float(np.mean(err)) if err.size else 0.0)
        scales.append(scale)
        unsigned_parts.append((q_signed + qmax).astype(np.int64))
    q_unsigned = np.concatenate(unsigned_parts) if unsigned_parts else np.zeros(0)
    packed_q = pack_fixed_width_uints(q_unsigned, bits=bits)
    scale_payload = np.asarray(scales, dtype="<f2").tobytes()
    raw_payload = scale_payload + packed_q
    header = {
        "schema": DECODER_PAYLOAD_V2_SCHEMA,
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": _decoder_kernel_shape_header(model_size),
        "feature_count": feature_count,
        "model_size_config": model_size.as_jsonable(),
        "codec": codec,
        "bits_per_weight": int(bits),
        "quantizer": "symmetric_per_kernel_fp16_scale",
        "q_offset": int(qmax),
        "scale_dtype": "float16_le",
        "scale_count": len(scales),
        "scale_bytes": len(scale_payload),
        "packed_q_bytes": len(packed_q),
        "value_count": int(values.size),
        "raw_reference_sha256": _sha256(raw_reference),
        "raw_reference_bytes": len(raw_reference),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else 0.0,
        "payload_sha256": _sha256(raw_payload),
        "payload_bytes": len(raw_payload),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, raw_payload)


def _encode_decoder_payload_mixed(
    *,
    levels: int,
    values: np.ndarray,
    model_size: SnervModelSizeConfig,
    raw_reference: bytes,
    explicit_modes: Sequence[str] | None = None,
) -> bytes:
    feature_count = int(model_size.feature_count)
    value_groups = values.reshape(levels * len(DECODER_SUBBANDS), feature_count)
    mode_plan: tuple[str, ...] | None = None
    if explicit_modes is not None:
        mode_plan = tuple(_normalize_mixed_decoder_kernel_mode(v) for v in explicit_modes)
        if len(mode_plan) != len(value_groups):
            raise SnervArchiveError(
                f"decoder mixed mode count {len(mode_plan)} != expected "
                f"{len(value_groups)}"
            )
    mode_codes: list[int] = []
    scales = []
    q_parts: list[bytes] = []
    fp16_parts: list[bytes] = []
    max_abs_error = 0.0
    mean_abs_errors = []
    histogram = dict.fromkeys(DECODER_PAYLOAD_MIXED_MODE_TO_CODE, 0)
    for idx, group in enumerate(value_groups):
        mode = (
            mode_plan[idx]
            if mode_plan is not None
            else _select_mixed_decoder_kernel_mode(group)
        )
        histogram[mode] += 1
        mode_codes.append(DECODER_PAYLOAD_MIXED_MODE_TO_CODE[mode])
        if mode == "zero":
            dequant = np.zeros_like(group, dtype=np.float64)
        elif mode == "fp16":
            payload = np.asarray(group, dtype="<f2").tobytes()
            fp16_parts.append(payload)
            dequant = np.frombuffer(payload, dtype="<f2").astype(np.float64)
        else:
            bits = int(mode.removeprefix("int"))
            qmax = (1 << (bits - 1)) - 1
            max_abs = float(np.max(np.abs(group))) if group.size else 0.0
            scale = 1.0 if max_abs == 0.0 else max_abs / float(qmax)
            q_signed = np.round(group / scale).clip(-qmax, qmax).astype(np.int64)
            q_parts.append(pack_fixed_width_uints(q_signed + qmax, bits=bits))
            scales.append(scale)
            dequant = q_signed.astype(np.float64) * scale
        err = np.abs(dequant - group)
        max_abs_error = max(max_abs_error, float(np.max(err)) if err.size else 0.0)
        mean_abs_errors.append(float(np.mean(err)) if err.size else 0.0)
    mode_code_payload = pack_fixed_width_uints(mode_codes, bits=3)
    scale_payload = np.asarray(scales, dtype="<f2").tobytes()
    q_payload = b"".join(q_parts)
    fp16_payload = b"".join(fp16_parts)
    raw_payload = mode_code_payload + scale_payload + q_payload + fp16_payload
    header = {
        "schema": DECODER_PAYLOAD_V3_SCHEMA,
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": _decoder_kernel_shape_header(model_size),
        "feature_count": feature_count,
        "model_size_config": model_size.as_jsonable(),
        "codec": DECODER_PAYLOAD_MIXED_CODEC,
        "quantizer": "mixed_per_kernel_zero_int2_int4_int8_fp16",
        "mode_assignment_source": (
            "explicit" if mode_plan is not None else "magnitude_heuristic"
        ),
        "mode_code_bits": 3,
        "mode_codebook": dict(DECODER_PAYLOAD_MIXED_MODE_TO_CODE),
        "mode_histogram": histogram,
        "mode_count": len(mode_codes),
        "mode_code_bytes": len(mode_code_payload),
        "scale_dtype": "float16_le",
        "scale_count": len(scales),
        "scale_bytes": len(scale_payload),
        "packed_q_bytes": len(q_payload),
        "fp16_value_bytes": len(fp16_payload),
        "value_count": int(values.size),
        "raw_reference_sha256": _sha256(raw_reference),
        "raw_reference_bytes": len(raw_reference),
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else 0.0,
        "payload_sha256": _sha256(raw_payload),
        "payload_bytes": len(raw_payload),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, raw_payload)


def decode_decoder_payload(payload: bytes) -> HfGenerationDecoder:
    """Decode the shared HF decoder from receiver payload bytes."""

    header, compressed = _unpack_subpacket(
        payload,
        magic=SNERV_DECODER_MAGIC,
        schema=(
            DECODER_PAYLOAD_V1_SCHEMA,
            DECODER_PAYLOAD_V2_SCHEMA,
            DECODER_PAYLOAD_V3_SCHEMA,
        ),
    )
    levels = int(header["levels"])
    if header["schema"] == DECODER_PAYLOAD_V3_SCHEMA:
        return _decode_decoder_payload_mixed(header, compressed)
    if header["schema"] == DECODER_PAYLOAD_V2_SCHEMA:
        return _decode_decoder_payload_quantized(header, compressed)
    raw = lzma.decompress(compressed)
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("decoder payload raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervArchiveError("decoder payload raw sha256 mismatch")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    return _decoder_from_flat_values(
        levels=levels,
        values=values,
        model_size=_model_size_from_decoder_header(header),
    )


def inspect_decoder_payload_header(payload: bytes) -> dict[str, Any]:
    """Return validated decoder payload header metadata without decoding weights."""

    header, _payload = _unpack_subpacket(
        payload,
        magic=SNERV_DECODER_MAGIC,
        schema=(
            DECODER_PAYLOAD_V1_SCHEMA,
            DECODER_PAYLOAD_V2_SCHEMA,
            DECODER_PAYLOAD_V3_SCHEMA,
        ),
    )
    return dict(header)


def _decode_decoder_payload_mixed(
    header: dict[str, Any],
    payload: bytes,
) -> HfGenerationDecoder:
    if _sha256(payload) != str(header["payload_sha256"]):
        raise SnervArchiveError("decoder mixed payload sha256 mismatch")
    levels = int(header["levels"])
    model_size = _model_size_from_decoder_header(header)
    feature_count = int(model_size.feature_count)
    group_count = levels * len(DECODER_SUBBANDS)
    if int(header["mode_count"]) != group_count:
        raise SnervArchiveError("decoder mixed mode count mismatch")
    mode_code_bytes = int(header["mode_code_bytes"])
    scale_bytes = int(header["scale_bytes"])
    q_bytes = int(header["packed_q_bytes"])
    fp16_bytes = int(header["fp16_value_bytes"])
    if len(payload) != mode_code_bytes + scale_bytes + q_bytes + fp16_bytes:
        raise SnervArchiveError("decoder mixed payload byte count mismatch")
    mode_code_payload = payload[:mode_code_bytes]
    scale_payload = payload[mode_code_bytes : mode_code_bytes + scale_bytes]
    q_payload = payload[
        mode_code_bytes + scale_bytes : mode_code_bytes + scale_bytes + q_bytes
    ]
    fp16_payload = payload[mode_code_bytes + scale_bytes + q_bytes :]
    mode_codes = unpack_fixed_width_uints(
        mode_code_payload,
        bits=int(header["mode_code_bits"]),
        count=group_count,
    )
    scales = np.frombuffer(scale_payload, dtype="<f2").astype(np.float64)
    values = np.zeros(int(header["value_count"]), dtype=np.float64)
    scale_cursor = 0
    q_cursor = 0
    fp16_cursor = 0
    for group_idx, raw_code in enumerate(mode_codes.tolist()):
        mode = DECODER_PAYLOAD_MIXED_CODE_TO_MODE.get(int(raw_code))
        if mode is None:
            raise SnervArchiveError(f"unknown decoder mixed mode code: {raw_code}")
        start = group_idx * feature_count
        stop = start + feature_count
        if mode == "zero":
            continue
        if mode == "fp16":
            nbytes = feature_count * np.dtype("<f2").itemsize
            segment = fp16_payload[fp16_cursor : fp16_cursor + nbytes]
            if len(segment) != nbytes:
                raise SnervArchiveError("decoder mixed fp16 payload truncated")
            values[start:stop] = np.frombuffer(segment, dtype="<f2").astype(np.float64)
            fp16_cursor += nbytes
            continue
        bits = int(mode.removeprefix("int"))
        qmax = (1 << (bits - 1)) - 1
        nbytes = (feature_count * bits + 7) // 8
        segment = q_payload[q_cursor : q_cursor + nbytes]
        if len(segment) != nbytes:
            raise SnervArchiveError("decoder mixed q payload truncated")
        if scale_cursor >= scales.size:
            raise SnervArchiveError("decoder mixed scale payload truncated")
        q_unsigned = unpack_fixed_width_uints(
            segment,
            bits=bits,
            count=feature_count,
        )
        q_signed = q_unsigned.astype(np.int64) - qmax
        values[start:stop] = q_signed.astype(np.float64) * float(scales[scale_cursor])
        scale_cursor += 1
        q_cursor += nbytes
    if scale_cursor != scales.size:
        raise SnervArchiveError("decoder mixed payload has unused scales")
    if q_cursor != len(q_payload):
        raise SnervArchiveError("decoder mixed payload has unused q bytes")
    if fp16_cursor != len(fp16_payload):
        raise SnervArchiveError("decoder mixed payload has unused fp16 bytes")
    return _decoder_from_flat_values(
        levels=levels,
        values=values,
        model_size=model_size,
    )


def _select_mixed_decoder_kernel_mode(group: np.ndarray) -> str:
    max_abs = float(np.max(np.abs(group))) if group.size else 0.0
    if max_abs <= 1e-12:
        return "zero"
    if max_abs >= 0.125:
        return "fp16"
    if max_abs >= 0.05:
        return "int8"
    if max_abs >= 0.015:
        return "int4"
    return "int2"


def _normalize_mixed_decoder_kernel_mode(raw: str) -> str:
    mode = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "0": "zero",
        "none": "zero",
        "z": "zero",
        "i2": "int2",
        "2": "int2",
        "int2_symmetric": "int2",
        "i4": "int4",
        "4": "int4",
        "int4_symmetric": "int4",
        "i8": "int8",
        "8": "int8",
        "int8_symmetric": "int8",
        "float16": "fp16",
        "f16": "fp16",
        "half": "fp16",
    }
    mode = aliases.get(mode, mode)
    if mode not in DECODER_PAYLOAD_MIXED_MODE_TO_CODE:
        raise SnervArchiveError(f"unsupported decoder mixed mode: {raw!r}")
    return mode


def _decode_decoder_payload_quantized(
    header: dict[str, Any],
    payload: bytes,
) -> HfGenerationDecoder:
    if _sha256(payload) != str(header["payload_sha256"]):
        raise SnervArchiveError("decoder quantized payload sha256 mismatch")
    levels = int(header["levels"])
    model_size = _model_size_from_decoder_header(header)
    feature_count = int(model_size.feature_count)
    bits = int(header["bits_per_weight"])
    offset = int(header["q_offset"])
    scale_bytes = int(header["scale_bytes"])
    scale_count = int(header["scale_count"])
    value_count = int(header["value_count"])
    if scale_count != levels * len(DECODER_SUBBANDS):
        raise SnervArchiveError("decoder quantized scale count mismatch")
    if scale_bytes != scale_count * np.dtype("<f2").itemsize:
        raise SnervArchiveError("decoder quantized scale byte count mismatch")
    if len(payload) < scale_bytes:
        raise SnervArchiveError("decoder quantized payload too short")
    scale_payload = payload[:scale_bytes]
    packed_q = payload[scale_bytes:]
    if len(packed_q) != int(header["packed_q_bytes"]):
        raise SnervArchiveError("decoder quantized q byte count mismatch")
    scales = np.frombuffer(scale_payload, dtype="<f2").astype(np.float64)
    q_unsigned = unpack_fixed_width_uints(packed_q, bits=bits, count=value_count)
    q_signed = q_unsigned.astype(np.int64) - offset
    values = q_signed.astype(np.float64)
    for idx, scale in enumerate(scales):
        start = idx * feature_count
        stop = start + feature_count
        values[start:stop] *= float(scale)
    return _decoder_from_flat_values(
        levels=levels,
        values=values,
        model_size=model_size,
    )


def _decoder_to_flat_values(
    decoder: HfGenerationDecoder,
) -> tuple[int, np.ndarray, SnervModelSizeConfig]:
    levels = int(decoder.levels)
    model_size = decoder.model_size
    feature_count = int(model_size.feature_count)
    arrays = []
    for lvl in range(levels):
        level = decoder.kernels.get(lvl)
        if not isinstance(level, dict):
            raise SnervArchiveError(f"decoder missing level {lvl}")
        for subband in DECODER_SUBBANDS:
            kernel = np.asarray(level.get(subband), dtype=np.float64)
            if kernel.size != feature_count:
                raise SnervArchiveError(
                    f"decoder kernel {lvl}/{subband} has {kernel.size} values, "
                    f"expected {feature_count}"
                )
            if not np.all(np.isfinite(kernel)):
                raise SnervArchiveError(f"decoder kernel {lvl}/{subband} is non-finite")
            arrays.append(kernel.reshape(-1))
    values = np.concatenate(arrays).astype(np.float64) if arrays else np.zeros(0)
    return levels, values, model_size


def _decoder_from_flat_values(
    *,
    levels: int,
    values: np.ndarray,
    model_size: SnervModelSizeConfig | None = None,
) -> HfGenerationDecoder:
    model_size = model_size or DEFAULT_SNERV_MODEL_SIZE
    feature_count = int(model_size.feature_count)
    expected = levels * len(DECODER_SUBBANDS) * feature_count
    if values.size != expected:
        raise SnervArchiveError(
            f"decoder payload has {values.size} values, expected {expected}"
        )
    kernels: dict[int, dict[str, np.ndarray]] = {}
    cursor = 0
    for lvl in range(levels):
        kernels[lvl] = {}
        for subband in DECODER_SUBBANDS:
            kernels[lvl][subband] = values[
                cursor : cursor + feature_count
            ].reshape(_decoder_kernel_storage_shape(model_size))
            cursor += feature_count
    return HfGenerationDecoder(
        kernels=kernels,
        levels=levels,
        model_size=model_size,
    )


def _decoder_kernel_shape_header(model_size: SnervModelSizeConfig) -> list[int]:
    if model_size == DEFAULT_SNERV_MODEL_SIZE:
        return [3, 3]
    return [int(model_size.feature_count)]


def _decoder_kernel_storage_shape(model_size: SnervModelSizeConfig) -> tuple[int, ...]:
    if model_size == DEFAULT_SNERV_MODEL_SIZE:
        return (3, 3)
    return (int(model_size.feature_count),)


def _model_size_from_decoder_header(header: dict[str, Any]) -> SnervModelSizeConfig:
    raw = header.get("model_size_config")
    if isinstance(raw, dict):
        return SnervModelSizeConfig(
            fc_dim=int(raw.get("fc_dim", raw.get("feature_count", 9))),
            emb_size=int(raw.get("emb_size", 0)),
            patch_radius=int(raw.get("patch_radius", 1)),
            adapter=str(raw.get("adapter", "snerv_fc_dim_emb_size_adapter_v1")),
        )
    feature_count = int(header.get("feature_count", 9))
    if feature_count != 9:
        return SnervModelSizeConfig(
            fc_dim=feature_count,
            emb_size=0,
            patch_radius=1,
        )
    return DEFAULT_SNERV_MODEL_SIZE


def decode_lf_metadata_payload(
    payload: bytes,
    *,
    expected_count: int | None = None,
) -> np.ndarray:
    """Decode compact LF zero-point metadata payload."""

    if len(payload) % 4:
        raise SnervArchiveError("LF metadata payload byte count is not float32-aligned")
    zeros = np.frombuffer(payload, dtype="<f4").copy()
    if expected_count is not None and zeros.size != expected_count:
        raise SnervArchiveError(
            f"decoded {zeros.size} LF zero-points, expected {expected_count}"
        )
    if zeros.size == 0:
        raise SnervArchiveError("LF metadata payload is empty")
    if not np.all(np.isfinite(zeros)):
        raise SnervArchiveError("LF metadata payload contains non-finite values")
    return zeros


def _validate_sections(sections: dict[str, bytes]) -> None:
    for name in SECTION_ORDER:
        blob = sections.get(name)
        if not isinstance(blob, bytes) or not blob:
            raise SnervArchiveError(f"section {name!r} must be non-empty bytes")


def _validate_lf_quant_plane(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane)
    if arr.size == 0:
        raise SnervArchiveError("LF quant planes must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise SnervArchiveError("LF quant planes must contain integers")
    return arr.astype("<i8", copy=False)


def _validate_replay_counts(
    lf_planes: list[np.ndarray],
    zeros: np.ndarray,
    step_maps: list[np.ndarray],
) -> None:
    if len(lf_planes) != int(zeros.size):
        raise SnervArchiveError(
            f"receiver replay LF plane count {len(lf_planes)} != zero-point count {zeros.size}"
        )
    if len(lf_planes) != len(step_maps):
        raise SnervArchiveError(
            f"receiver replay LF plane count {len(lf_planes)} != step-map count {len(step_maps)}"
        )
    if not lf_planes:
        raise SnervArchiveError("receiver replay requires at least one LF plane")


def _metadata_int(
    metadata: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key not in metadata:
        if default is None:
            raise SnervArchiveError(f"receiver replay metadata missing {key!r}")
        value = default
    else:
        value = metadata[key]
    if isinstance(value, bool):
        raise SnervArchiveError(f"receiver replay metadata {key!r} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise SnervArchiveError(
            f"receiver replay metadata {key!r} must be an integer"
        ) from exc
    if minimum is not None and out < minimum:
        raise SnervArchiveError(
            f"receiver replay metadata {key!r}={out} must be >= {minimum}"
        )
    return out


def _metadata_str(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise SnervArchiveError(f"receiver replay metadata missing string {key!r}")
    return value


def _metadata_hw(metadata: dict[str, Any]) -> tuple[int, int]:
    value = metadata.get("carrier_hw", metadata.get("orig_hw"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SnervArchiveError(
            "receiver replay metadata missing 2-element 'carrier_hw'/'orig_hw'"
        )
    h, w = int(value[0]), int(value[1])
    if h <= 0 or w <= 0:
        raise SnervArchiveError("receiver replay metadata height/width must be positive")
    return h, w


def _pack_subpacket(magic: bytes, header: dict[str, Any], payload: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return magic + struct.pack(HEADER_LEN_FMT, len(header_bytes)) + header_bytes + payload


def _unpack_subpacket(
    packet: bytes,
    *,
    magic: bytes,
    schema: str | tuple[str, ...],
) -> tuple[dict[str, Any], bytes]:
    packet = bytes(packet)
    if not packet.startswith(magic):
        raise SnervArchiveError(f"bad subpacket magic for {schema}")
    offset = len(magic)
    if len(packet) < offset + struct.calcsize(HEADER_LEN_FMT):
        raise SnervArchiveError(f"truncated subpacket header for {schema}")
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT,
        packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)],
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    if header_end > len(packet):
        raise SnervArchiveError(f"declared subpacket header exceeds bytes for {schema}")
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    allowed = (schema,) if isinstance(schema, str) else tuple(schema)
    if header.get("schema") not in allowed:
        raise SnervArchiveError(f"unsupported subpacket schema: {header.get('schema')!r}")
    return dict(header), packet[header_end:]


def _unpack_lf_quant_subpacket(packet: bytes) -> tuple[dict[str, Any], bytes]:
    return _unpack_subpacket(
        packet,
        magic=SNERV_LF_QUANT_MAGIC,
        schema=(LF_QUANT_PAYLOAD_SCHEMA_V1, LF_QUANT_PAYLOAD_SCHEMA_V2),
    )


def _jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(metadata, sort_keys=True)
    except TypeError as exc:
        raise SnervArchiveError("metadata must be JSON-serializable") from exc
    return dict(metadata)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
