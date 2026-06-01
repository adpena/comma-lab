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
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.analysis.snerv_step_map_coder import decode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    HfGenerationDecoder,
    SnervFrameCode,
    decode_frame,
)

SNERV_ARCHIVE_SCHEMA = "snerv_inverse_steg_archive.v1"
SNERV_ARCHIVE_MAGIC = b"SNAR1"
SNERV_LF_QUANT_MAGIC = b"SNQL1"
SNERV_DECODER_MAGIC = b"SNDC1"
HEADER_LEN_FMT = "<I"
SECTION_ORDER = ("metadata_payload", "lf_payload", "decoder_payload", "step_map_packet")
DECODER_SUBBANDS = ("LH", "HL", "HH")


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


def encode_lf_quant_payload(lf_quant_planes: list[np.ndarray]) -> bytes:
    """Encode quantized LF planes as deterministic scorer-free receiver bytes."""

    arrays = [_validate_lf_quant_plane(a) for a in lf_quant_planes]
    if not arrays:
        raise SnervArchiveError("lf_quant_planes must be non-empty")
    raw = b"".join(np.asarray(a, dtype="<i8").reshape(-1).tobytes() for a in arrays)
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": "snerv_lf_quant_payload.v1",
        "dtype": "int64_le",
        "shapes": [list(a.shape) for a in arrays],
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_LF_QUANT_MAGIC, header, compressed)


def decode_lf_quant_payload(payload: bytes) -> list[np.ndarray]:
    """Decode LF quantized coefficient planes from receiver payload bytes."""

    header, compressed = _unpack_subpacket(
        payload,
        magic=SNERV_LF_QUANT_MAGIC,
        schema="snerv_lf_quant_payload.v1",
    )
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


def encode_decoder_payload(decoder: HfGenerationDecoder) -> bytes:
    """Encode the shared HF decoder as deterministic scorer-free receiver bytes."""

    levels = int(decoder.levels)
    arrays = []
    for lvl in range(levels):
        level = decoder.kernels.get(lvl)
        if not isinstance(level, dict):
            raise SnervArchiveError(f"decoder missing level {lvl}")
        for subband in DECODER_SUBBANDS:
            kernel = np.asarray(level.get(subband), dtype="<f4")
            if kernel.shape != (3, 3):
                raise SnervArchiveError(
                    f"decoder kernel {lvl}/{subband} shape {kernel.shape} != (3, 3)"
                )
            if not np.all(np.isfinite(kernel)):
                raise SnervArchiveError(f"decoder kernel {lvl}/{subband} is non-finite")
            arrays.append(kernel.reshape(-1))
    raw = np.concatenate(arrays).astype("<f4").tobytes() if arrays else b""
    if not raw:
        raise SnervArchiveError("decoder payload must be non-empty")
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": "snerv_decoder_payload.v1",
        "levels": levels,
        "subbands": list(DECODER_SUBBANDS),
        "kernel_shape": [3, 3],
        "dtype": "float32_le",
        "raw_sha256": _sha256(raw),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    return _pack_subpacket(SNERV_DECODER_MAGIC, header, compressed)


def decode_decoder_payload(payload: bytes) -> HfGenerationDecoder:
    """Decode the shared HF decoder from receiver payload bytes."""

    header, compressed = _unpack_subpacket(
        payload,
        magic=SNERV_DECODER_MAGIC,
        schema="snerv_decoder_payload.v1",
    )
    levels = int(header["levels"])
    raw = lzma.decompress(compressed)
    if len(raw) != int(header["raw_bytes"]):
        raise SnervArchiveError("decoder payload raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervArchiveError("decoder payload raw sha256 mismatch")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    expected = levels * len(DECODER_SUBBANDS) * 9
    if values.size != expected:
        raise SnervArchiveError(
            f"decoder payload has {values.size} values, expected {expected}"
        )
    kernels: dict[int, dict[str, np.ndarray]] = {}
    cursor = 0
    for lvl in range(levels):
        kernels[lvl] = {}
        for subband in DECODER_SUBBANDS:
            kernels[lvl][subband] = values[cursor : cursor + 9].reshape(3, 3)
            cursor += 9
    return HfGenerationDecoder(kernels=kernels, levels=levels)


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
    schema: str,
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
    if header.get("schema") != schema:
        raise SnervArchiveError(f"unsupported subpacket schema: {header.get('schema')!r}")
    return dict(header), packet[header_end:]


def _jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(metadata, sort_keys=True)
    except TypeError as exc:
        raise SnervArchiveError("metadata must be JSON-serializable") from exc
    return dict(metadata)


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
