# SPDX-License-Identifier: MIT
"""Production V10 scorer-plane archive and deterministic factor-2 receiver.

The archive carries only video-derived scorer-plane bytes (and an optional
camera-lattice residual).  Decode is scorer-free: it validates the declared
native-f32/first-max selection policy, realizes the uint8 scorer plane with the
integer-only disjoint-support construction, and writes contest raw frames.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tac.codec.v10_predictor_residual import (
    CODEC_ID as PREDICTOR_RESIDUAL_Y_CODEC_ID,
)
from tac.codec.v10_predictor_residual import (
    PredictorMode,
    PredictorResidualError,
    decode_predictor_residual,
    encode_predictor_residual,
)
from tac.codec.v10_jxl_plane_codec import (
    CODEC_ID as JXL_PLANE_Y_CODEC_ID,
)
from tac.codec.v10_jxl_plane_codec import (
    JxlPlaneCodecError,
    decode_payload as decode_jxl_plane_payload,
    encode_pairs as encode_jxl_plane_pairs,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

MAGIC = b"TACV10R\x00"
VERSION = 1
PREFIX = struct.Struct(">8sHI")
SECTION_LENGTH = struct.Struct(">Q")
RESIDUAL_RECORD = struct.Struct("<IIIBh")
MEMBER_NAME = "0.bin"
PACKET_SCHEMA = "v10_production_archive.v1"
MANIFEST_SCHEMA = "v10_production_archive_manifest.v1"
INFLATE_MANIFEST_SCHEMA = "v10_production_inflate_manifest.v1"
PAIR_STATE_SCHEMA = "v10_production_pair_state.v1"
RECEIVER_CONTRACT_ID = "factor2-disjoint-half-pixel-uint8.v1"
TIE_POLICY_ID = "native-cpu-torch-f32-first-max-class-index.v1"
ARITHMETIC_ID = "integer-only-support-fill-after-native-f32-encode-selection.v1"
FRAME0_POLICY_ID = "repeat-frame1"
DESCRIPTION_FRAME0_POLICY_ID = "description-frame0.v1"
FRAME0_POLICY_IDS = frozenset({FRAME0_POLICY_ID, DESCRIPTION_FRAME0_POLICY_ID})
RESIDUAL_CODEC_ID = "sparse-int16-le.v1"
Y_CODEC_IDS = frozenset(
    {"raw-uint8-y", "brotli-y", "witness-y-stub", PREDICTOR_RESIDUAL_Y_CODEC_ID, JXL_PLANE_Y_CODEC_ID}
)
SECTION_ORDER = ("y_description", "frame0_policy", "quotient_residual")

MAX_HEADER_BYTES = 1 << 20
MAX_PACKET_BYTES = 1 << 30
MAX_SECTION_BYTES = 1 << 30
MAX_DECODED_PLANE_BYTES = 1 << 30
MAX_PAIRS = 10_000
MAX_CAMERA_DIMENSION = 4096
MAX_SCORER_DIMENSION = 2048

_HEADER_FIELDS = frozenset(
    {
        "schema",
        "version",
        "geometry",
        "pair_count",
        "section_count",
        "sections",
        "counted_section_payload_bytes",
        "video_derived_payload_bytes",
        "section_framing_bytes",
        "packet_bytes",
        "receiver_contract_id",
        "tie_policy_id",
        "arithmetic_id",
        "frame0_policy_id",
        "y_codec_id",
        "residual_codec_id",
        "launch_ready",
        "score_claim",
        "promotion_eligible",
    }
)
_SECTION_FIELDS = frozenset(
    {
        "section_id",
        "codec_id",
        "byte_length",
        "sha256",
        "decoded_byte_length",
        "decoded_sha256",
        "video_derived",
    }
)


class ProductionReceiverError(ValueError):
    """Fail-closed malformed packet, archive, state, or output error."""


@dataclass(frozen=True)
class ProductionSection:
    section_id: str
    codec_id: str
    payload: bytes
    decoded_byte_length: int
    decoded_sha256: str
    video_derived: bool

    def header_row(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "codec_id": self.codec_id,
            "byte_length": len(self.payload),
            "sha256": _sha256(self.payload),
            "decoded_byte_length": self.decoded_byte_length,
            "decoded_sha256": self.decoded_sha256,
            "video_derived": self.video_derived,
        }


@dataclass(frozen=True)
class ParsedProductionPacket:
    packet_bytes: bytes
    packet_sha256: str
    header: Mapping[str, Any]
    sections: tuple[ProductionSection, ...]

    def section(self, section_id: str) -> ProductionSection:
        matches = [section for section in self.sections if section.section_id == section_id]
        if len(matches) != 1:
            raise ProductionReceiverError(f"packet requires exactly one {section_id} section")
        return matches[0]


@dataclass(frozen=True)
class ArchiveBuildResult:
    archive_path: Path
    manifest_path: Path
    archive_sha256: str
    archive_bytes: int
    packet_sha256: str
    packet_bytes: int
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class InflateResult:
    completed: bool
    raw_path: Path | None
    raw_sha256: str | None
    raw_bytes: int
    pair_stages_preserved: int
    numerator_values_verified: int
    tree_sha256: str | None
    storage_preflight: Mapping[str, Any]


@dataclass(frozen=True)
class DecodedYPlanePair:
    """Typed scorer-plane pair returned by the closed production grammar."""

    frame0: np.ndarray
    frame1: np.ndarray


@dataclass(frozen=True, order=True)
class QuotientResidualUpdate:
    pair_index: int
    row: int
    col: int
    channel: int
    delta: int


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionReceiverError("value is not canonical-JSON encodable") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not (
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    ):
        raise ProductionReceiverError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProductionReceiverError(f"{label} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ProductionReceiverError(f"{label} is outside its admitted bounds")
    return value


def _checked_product(values: Sequence[int], label: str, *, maximum: int) -> int:
    result = 1
    for value in values:
        result *= _exact_int(value, label, minimum=1)
        if result > maximum:
            raise ProductionReceiverError(f"{label} exceeds its byte cap")
    return result


def _brotli() -> Any:
    try:
        import brotli  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProductionReceiverError("brotli-y codec dependency is unavailable") from exc
    return brotli


def _encode_y(raw: bytes, codec_id: str) -> bytes:
    if codec_id == "raw-uint8-y":
        return raw
    if codec_id == "brotli-y":
        return bytes(_brotli().compress(raw, quality=11))
    if codec_id == "witness-y-stub":
        return raw
    if codec_id == PREDICTOR_RESIDUAL_Y_CODEC_ID:
        raise ProductionReceiverError("predictor-residual requires typed two-plane encoding")
    if codec_id == JXL_PLANE_Y_CODEC_ID:
        raise ProductionReceiverError("jxl-lossless-plane requires typed two-plane encoding")
    raise ProductionReceiverError(f"unknown y codec id {codec_id!r}")


def _decode_y(section: ProductionSection) -> bytes:
    if section.codec_id == "witness-y-stub":
        raise ProductionReceiverError("witness-y-stub is a typed refusal, not a decoder")
    if section.codec_id == "raw-uint8-y":
        decoded = section.payload
    elif section.codec_id == "brotli-y":
        try:
            decoded = bytes(_brotli().decompress(section.payload))
        except Exception as exc:
            raise ProductionReceiverError("brotli-y decompression failed") from exc
    elif section.codec_id == PREDICTOR_RESIDUAL_Y_CODEC_ID:
        try:
            pair = decode_predictor_residual(section.payload)
        except PredictorResidualError as exc:
            raise ProductionReceiverError("predictor-residual y payload refused") from exc
        decoded = pair.frame0.tobytes(order="C") + pair.frame1.tobytes(order="C")
    elif section.codec_id == JXL_PLANE_Y_CODEC_ID:
        try:
            planes = decode_jxl_plane_payload(section.payload)
        except JxlPlaneCodecError as exc:
            raise ProductionReceiverError("jxl-lossless-plane y payload refused") from exc
        decoded = planes.frame0.tobytes(order="C") + planes.frame1.tobytes(order="C")
    else:
        raise ProductionReceiverError(f"unknown y codec id {section.codec_id!r}")
    if len(decoded) != section.decoded_byte_length or _sha256(decoded) != section.decoded_sha256:
        raise ProductionReceiverError("decoded y section length/hash custody failure")
    return decoded


def _validate_geometry(header: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    geometry = header.get("geometry")
    if not isinstance(geometry, dict) or set(geometry) != {
        "camera_height",
        "camera_width",
        "scorer_height",
        "scorer_width",
        "channels",
    }:
        raise ProductionReceiverError("header geometry has an unknown or missing field")
    camera_h = _exact_int(geometry["camera_height"], "camera_height", minimum=1, maximum=MAX_CAMERA_DIMENSION)
    camera_w = _exact_int(geometry["camera_width"], "camera_width", minimum=1, maximum=MAX_CAMERA_DIMENSION)
    scorer_h = _exact_int(geometry["scorer_height"], "scorer_height", minimum=1, maximum=MAX_SCORER_DIMENSION)
    scorer_w = _exact_int(geometry["scorer_width"], "scorer_width", minimum=1, maximum=MAX_SCORER_DIMENSION)
    channels = _exact_int(geometry["channels"], "channels", minimum=3, maximum=3)
    try:
        DisjointResizeOperator.build(
            camera_h=camera_h,
            camera_w=camera_w,
            scorer_h=scorer_h,
            scorer_w=scorer_w,
        )
    except Uint8LatticeError as exc:
        raise ProductionReceiverError("geometry is not the certified disjoint half-pixel lattice") from exc
    return camera_h, camera_w, scorer_h, scorer_w, channels


def _validate_residual_update(
    update: QuotientResidualUpdate,
    *,
    pair_count: int,
    camera_h: int,
    camera_w: int,
    channels: int,
) -> QuotientResidualUpdate:
    if not isinstance(update, QuotientResidualUpdate):
        raise ProductionReceiverError("quotient residual rows must be typed updates")
    pair_index = _exact_int(update.pair_index, "residual pair_index", maximum=pair_count - 1)
    row = _exact_int(update.row, "residual row", maximum=camera_h - 1)
    col = _exact_int(update.col, "residual col", maximum=camera_w - 1)
    channel = _exact_int(update.channel, "residual channel", maximum=channels - 1)
    delta = _exact_int(update.delta, "residual delta", minimum=-32768, maximum=32767)
    if delta == 0:
        raise ProductionReceiverError("quotient residual refuses paid zero updates")
    return QuotientResidualUpdate(pair_index, row, col, channel, delta)


def _encode_residual_updates(
    residual: np.ndarray | Sequence[QuotientResidualUpdate],
    *,
    pair_count: int,
    camera_h: int,
    camera_w: int,
    channels: int,
) -> bytes:
    if isinstance(residual, np.ndarray):
        expected_shape = (pair_count, camera_h, camera_w, channels)
        if residual.dtype != np.dtype("<i2") or residual.shape != expected_shape:
            raise ProductionReceiverError("quotient_residual must be little-endian int16 [pairs,camera_H,camera_W,3]")
        rows = tuple(
            QuotientResidualUpdate(
                int(index[0]),
                int(index[1]),
                int(index[2]),
                int(index[3]),
                int(residual[tuple(index)]),
            )
            for index in np.argwhere(residual != 0)
        )
    elif isinstance(residual, Sequence) and not isinstance(residual, (bytes, bytearray, str)):
        rows = tuple(residual)
    else:
        raise ProductionReceiverError("quotient_residual must be an int16 tensor or typed update sequence")
    if not rows:
        raise ProductionReceiverError("quotient residual section cannot be empty")
    validated = tuple(
        _validate_residual_update(
            update,
            pair_count=pair_count,
            camera_h=camera_h,
            camera_w=camera_w,
            channels=channels,
        )
        for update in rows
    )
    ordered = tuple(sorted(validated))
    coordinates = [(item.pair_index, item.row, item.col, item.channel) for item in ordered]
    if len(set(coordinates)) != len(coordinates):
        raise ProductionReceiverError("quotient residual has duplicate camera ownership")
    payload = b"".join(
        RESIDUAL_RECORD.pack(item.pair_index, item.row, item.col, item.channel, item.delta) for item in ordered
    )
    if len(payload) > MAX_SECTION_BYTES:
        raise ProductionReceiverError("quotient residual exceeds the section cap")
    return payload


def _parse_residual_updates(
    payload: bytes,
    *,
    pair_count: int,
    camera_h: int,
    camera_w: int,
    channels: int,
) -> tuple[QuotientResidualUpdate, ...]:
    if len(payload) % RESIDUAL_RECORD.size:
        raise ProductionReceiverError("quotient residual has a truncated record")
    updates = tuple(
        _validate_residual_update(
            QuotientResidualUpdate(*RESIDUAL_RECORD.unpack_from(payload, offset)),
            pair_count=pair_count,
            camera_h=camera_h,
            camera_w=camera_w,
            channels=channels,
        )
        for offset in range(0, len(payload), RESIDUAL_RECORD.size)
    )
    if tuple(sorted(updates)) != updates:
        raise ProductionReceiverError("quotient residual records are not canonically ordered")
    coordinates = [(item.pair_index, item.row, item.col, item.channel) for item in updates]
    if len(set(coordinates)) != len(coordinates):
        raise ProductionReceiverError("quotient residual records duplicate camera ownership")
    return updates


def build_packet(
    y_planes: np.ndarray,
    *,
    camera_height: int,
    camera_width: int,
    y_codec_id: str = "raw-uint8-y",
    frame0_y_planes: np.ndarray | None = None,
    predictor_modes: PredictorMode | str | int | Sequence[PredictorMode | str | int] | None = None,
    predictor_descriptors: Sequence[bytes] | None = None,
    predictor_pair_ids: Sequence[int] | None = None,
    quotient_residual: np.ndarray | Sequence[QuotientResidualUpdate] | None = None,
    jxl_effort: int = 9,
    jxl_workers: int = 1,
) -> bytes:
    """Build canonical ``0.bin`` bytes from exact uint8 scorer planes."""

    y = np.asarray(y_planes)
    if y.dtype != np.uint8 or y.ndim != 4 or y.shape[-1] != 3:
        raise ProductionReceiverError("y_planes must be exact uint8 [pairs,H,W,3]")
    if not y.flags.c_contiguous:
        y = np.ascontiguousarray(y)
    pair_count, scorer_h, scorer_w, channels = map(int, y.shape)
    _exact_int(pair_count, "pair_count", minimum=1, maximum=MAX_PAIRS)
    camera_h = _exact_int(camera_height, "camera_height", minimum=1, maximum=MAX_CAMERA_DIMENSION)
    camera_w = _exact_int(camera_width, "camera_width", minimum=1, maximum=MAX_CAMERA_DIMENSION)
    if y_codec_id not in Y_CODEC_IDS:
        raise ProductionReceiverError(f"unknown y codec id {y_codec_id!r}")
    _validate_geometry(
        {
            "geometry": {
                "camera_height": camera_h,
                "camera_width": camera_w,
                "scorer_height": scorer_h,
                "scorer_width": scorer_w,
                "channels": channels,
            }
        }
    )

    if y_codec_id == PREDICTOR_RESIDUAL_Y_CODEC_ID:
        if frame0_y_planes is None:
            raise ProductionReceiverError("predictor-residual y codec requires frame0_y_planes")
        frame0 = np.asarray(frame0_y_planes)
        if frame0.dtype != np.uint8 or frame0.shape != y.shape:
            raise ProductionReceiverError("frame0_y_planes must match exact uint8 frame1 geometry")
        frame0 = np.ascontiguousarray(frame0)
        try:
            y_payload = encode_predictor_residual(
                frame0,
                y,
                modes=PredictorMode.PREVIOUS_PLANE_COPY if predictor_modes is None else predictor_modes,
                descriptors=predictor_descriptors,
                pair_ids=predictor_pair_ids,
            )
        except PredictorResidualError as exc:
            raise ProductionReceiverError("predictor-residual y encoding refused") from exc
        y_raw = frame0.tobytes(order="C") + y.tobytes(order="C")
        frame0_policy_id = DESCRIPTION_FRAME0_POLICY_ID
    elif y_codec_id == JXL_PLANE_Y_CODEC_ID:
        if frame0_y_planes is None:
            raise ProductionReceiverError("jxl-lossless-plane y codec requires frame0_y_planes")
        if predictor_modes is not None or predictor_descriptors is not None:
            raise ProductionReceiverError("jxl-lossless-plane refuses predictor mode/descriptor arguments")
        frame0 = np.asarray(frame0_y_planes)
        if frame0.dtype != np.uint8 or frame0.shape != y.shape:
            raise ProductionReceiverError("frame0_y_planes must match exact uint8 frame1 geometry")
        frame0 = np.ascontiguousarray(frame0)
        try:
            y_payload = encode_jxl_plane_pairs(
                frame0,
                y,
                pair_ids=None if predictor_pair_ids is None else [int(p) for p in predictor_pair_ids],
                effort=jxl_effort,
                workers=jxl_workers,
            )
        except JxlPlaneCodecError as exc:
            raise ProductionReceiverError("jxl-lossless-plane y encoding refused") from exc
        y_raw = frame0.tobytes(order="C") + y.tobytes(order="C")
        frame0_policy_id = DESCRIPTION_FRAME0_POLICY_ID
    else:
        if (
            frame0_y_planes is not None
            or predictor_modes is not None
            or predictor_descriptors is not None
            or predictor_pair_ids is not None
        ):
            raise ProductionReceiverError("legacy y codecs refuse predictor-only two-plane arguments")
        y_raw = y.tobytes(order="C")
        y_payload = _encode_y(y_raw, y_codec_id)
        frame0_policy_id = FRAME0_POLICY_ID
    if len(y_raw) > MAX_DECODED_PLANE_BYTES:
        raise ProductionReceiverError("decoded y plane bytes exceed the cap")
    sections = [
        ProductionSection(
            "y_description",
            y_codec_id,
            y_payload,
            len(y_raw),
            _sha256(y_raw),
            True,
        ),
        ProductionSection(
            "frame0_policy",
            frame0_policy_id,
            b"",
            0,
            _sha256(b""),
            False,
        ),
    ]
    residual_codec_id: str | None = None
    if quotient_residual is not None:
        residual_raw = _encode_residual_updates(
            quotient_residual,
            pair_count=pair_count,
            camera_h=camera_h,
            camera_w=camera_w,
            channels=channels,
        )
        residual_codec_id = RESIDUAL_CODEC_ID
        sections.append(
            ProductionSection(
                "quotient_residual",
                RESIDUAL_CODEC_ID,
                residual_raw,
                len(residual_raw),
                _sha256(residual_raw),
                True,
            )
        )
    rows = [section.header_row() for section in sections]
    payload_bytes = sum(len(section.payload) for section in sections)
    video_bytes = sum(len(section.payload) for section in sections if section.video_derived)
    header: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "version": VERSION,
        "geometry": {
            "camera_height": camera_h,
            "camera_width": camera_w,
            "scorer_height": scorer_h,
            "scorer_width": scorer_w,
            "channels": channels,
        },
        "pair_count": pair_count,
        "section_count": len(sections),
        "sections": rows,
        "counted_section_payload_bytes": payload_bytes,
        "video_derived_payload_bytes": video_bytes,
        "section_framing_bytes": SECTION_LENGTH.size * len(sections),
        "packet_bytes": 0,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "tie_policy_id": TIE_POLICY_ID,
        "arithmetic_id": ARITHMETIC_ID,
        "frame0_policy_id": frame0_policy_id,
        "y_codec_id": y_codec_id,
        "residual_codec_id": residual_codec_id,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    for _ in range(8):
        header_bytes = _canonical_json(header)
        packet_bytes = PREFIX.size + len(header_bytes) + header["section_framing_bytes"] + payload_bytes
        if header["packet_bytes"] == packet_bytes:
            break
        header["packet_bytes"] = packet_bytes
    else:  # pragma: no cover - integer digit count converges immediately
        raise ProductionReceiverError("packet byte-count fixed point did not converge")
    header_bytes = _canonical_json(header)
    if len(header_bytes) > MAX_HEADER_BYTES or header["packet_bytes"] > MAX_PACKET_BYTES:
        raise ProductionReceiverError("production packet exceeds its size cap")
    packet = bytearray(PREFIX.pack(MAGIC, VERSION, len(header_bytes)))
    packet.extend(header_bytes)
    for section in sections:
        packet.extend(SECTION_LENGTH.pack(len(section.payload)))
        packet.extend(section.payload)
    if len(packet) != header["packet_bytes"]:
        raise ProductionReceiverError("packet byte accounting drift")
    return bytes(packet)


def parse_packet(packet_bytes: bytes, *, max_packet_bytes: int = MAX_PACKET_BYTES) -> ParsedProductionPacket:
    """Parse one packet with exact stream consumption and size/hash custody."""

    if not isinstance(packet_bytes, bytes):
        raise ProductionReceiverError("packet must be immutable bytes")
    max_packet_bytes = _exact_int(
        max_packet_bytes,
        "max_packet_bytes",
        minimum=1,
        maximum=MAX_PACKET_BYTES,
    )
    if len(packet_bytes) > max_packet_bytes or len(packet_bytes) < PREFIX.size:
        raise ProductionReceiverError("packet is truncated or exceeds its size cap")
    magic, version, header_length = PREFIX.unpack_from(packet_bytes)
    if magic != MAGIC or version != VERSION:
        raise ProductionReceiverError("production packet magic/version mismatch")
    if not 0 < header_length <= MAX_HEADER_BYTES:
        raise ProductionReceiverError("production header length is outside its cap")
    header_end = PREFIX.size + header_length
    if header_end > len(packet_bytes):
        raise ProductionReceiverError("truncated production header")
    header_bytes = packet_bytes[PREFIX.size : header_end]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionReceiverError("production header is not valid JSON") from exc
    if not isinstance(header, dict) or _canonical_json(header) != header_bytes:
        raise ProductionReceiverError("production header is not canonical JSON")
    if set(header) != _HEADER_FIELDS:
        raise ProductionReceiverError("production header has an unknown or missing field")
    if header["schema"] != PACKET_SCHEMA or type(header["version"]) is not int or header["version"] != VERSION:
        raise ProductionReceiverError("production header schema/version mismatch")
    for field in ("launch_ready", "score_claim", "promotion_eligible"):
        if header[field] is not False:
            raise ProductionReceiverError(f"production packet cannot authorize {field}")
    if (
        header["receiver_contract_id"] != RECEIVER_CONTRACT_ID
        or header["tie_policy_id"] != TIE_POLICY_ID
        or header["arithmetic_id"] != ARITHMETIC_ID
        or header["frame0_policy_id"] not in FRAME0_POLICY_IDS
    ):
        raise ProductionReceiverError("receiver/arithmetic/tie/frame0 policy declaration drift")
    if header["y_codec_id"] not in Y_CODEC_IDS:
        raise ProductionReceiverError("header declares an unknown y codec")
    if header["residual_codec_id"] not in (None, RESIDUAL_CODEC_ID):
        raise ProductionReceiverError("header declares an unknown residual codec")
    camera_h, camera_w, scorer_h, scorer_w, channels = _validate_geometry(header)
    pair_count = _exact_int(header["pair_count"], "pair_count", minimum=1, maximum=MAX_PAIRS)
    expected_y_bytes = _checked_product(
        (pair_count, scorer_h, scorer_w, channels),
        "decoded y bytes",
        maximum=MAX_DECODED_PLANE_BYTES,
    )
    if header["y_codec_id"] in (PREDICTOR_RESIDUAL_Y_CODEC_ID, JXL_PLANE_Y_CODEC_ID):
        if expected_y_bytes > MAX_DECODED_PLANE_BYTES // 2:
            raise ProductionReceiverError("decoded two-plane y bytes exceed the cap")
        expected_y_bytes *= 2
        expected_frame0_policy_id = DESCRIPTION_FRAME0_POLICY_ID
    else:
        expected_frame0_policy_id = FRAME0_POLICY_ID
    if header["frame0_policy_id"] != expected_frame0_policy_id:
        raise ProductionReceiverError("y codec and frame0 policy declaration disagree")
    rows = header["sections"]
    section_count = _exact_int(header["section_count"], "section_count", minimum=2, maximum=3)
    if not isinstance(rows, list) or len(rows) != section_count:
        raise ProductionReceiverError("section count/header rows mismatch")
    expected_ids = list(SECTION_ORDER[:section_count])
    if [row.get("section_id") if isinstance(row, dict) else None for row in rows] != expected_ids:
        raise ProductionReceiverError("sections are duplicate, unknown, missing, or out of order")
    if (section_count == 3) != (header["residual_codec_id"] is not None):
        raise ProductionReceiverError("residual declaration and section presence disagree")

    cursor = header_end
    sections: list[ProductionSection] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _SECTION_FIELDS:
            raise ProductionReceiverError(f"section row {index} has an unknown or missing field")
        if cursor + SECTION_LENGTH.size > len(packet_bytes):
            raise ProductionReceiverError(f"truncated length prefix for section {index}")
        framed_length = SECTION_LENGTH.unpack_from(packet_bytes, cursor)[0]
        cursor += SECTION_LENGTH.size
        declared_length = _exact_int(row["byte_length"], f"section {index} byte_length", maximum=MAX_SECTION_BYTES)
        if framed_length != declared_length:
            raise ProductionReceiverError(f"section {index} framed/header length mismatch")
        end = cursor + framed_length
        if end > len(packet_bytes):
            raise ProductionReceiverError(f"truncated payload for section {index}")
        payload = packet_bytes[cursor:end]
        if _sha256(payload) != _require_sha256(row["sha256"], f"section {index} sha256"):
            raise ProductionReceiverError(f"section {index} payload hash mismatch")
        decoded_length = _exact_int(
            row["decoded_byte_length"],
            f"section {index} decoded_byte_length",
            maximum=MAX_DECODED_PLANE_BYTES,
        )
        decoded_sha = _require_sha256(row["decoded_sha256"], f"section {index} decoded_sha256")
        if not isinstance(row["codec_id"], str) or type(row["video_derived"]) is not bool:
            raise ProductionReceiverError(f"section {index} codec/video-derived type drift")
        section = ProductionSection(
            row["section_id"], row["codec_id"], payload, decoded_length, decoded_sha, row["video_derived"]
        )
        if section.header_row() != row:
            raise ProductionReceiverError(f"section {index} row disagrees with reopened bytes")
        sections.append(section)
        cursor = end
    if cursor != len(packet_bytes):
        raise ProductionReceiverError("production packet has trailing data")
    packet_total = _exact_int(header["packet_bytes"], "packet_bytes", minimum=1)
    if packet_total != len(packet_bytes):
        raise ProductionReceiverError("packet total byte declaration drift")
    payload_total = sum(len(section.payload) for section in sections)
    video_total = sum(len(section.payload) for section in sections if section.video_derived)
    counted_payload = _exact_int(header["counted_section_payload_bytes"], "counted_section_payload_bytes")
    video_payload = _exact_int(header["video_derived_payload_bytes"], "video_derived_payload_bytes")
    framing_bytes = _exact_int(header["section_framing_bytes"], "section_framing_bytes")
    if (
        counted_payload != payload_total
        or video_payload != video_total
        or framing_bytes != SECTION_LENGTH.size * section_count
    ):
        raise ProductionReceiverError("packet counted-byte totals drift")
    y_section, policy_section = sections[:2]
    if (
        y_section.codec_id != header["y_codec_id"]
        or y_section.decoded_byte_length != expected_y_bytes
        or y_section.video_derived is not True
    ):
        raise ProductionReceiverError("y section codec/shape/authority drift")
    if (
        policy_section.codec_id != expected_frame0_policy_id
        or policy_section.payload != b""
        or policy_section.decoded_byte_length != 0
        or policy_section.decoded_sha256 != _sha256(b"")
        or policy_section.video_derived is not False
    ):
        raise ProductionReceiverError("frame0 policy must be the generic zero-byte policy")
    if y_section.codec_id == PREDICTOR_RESIDUAL_Y_CODEC_ID:
        try:
            decoded_pair = decode_predictor_residual(y_section.payload)
        except PredictorResidualError as exc:
            raise ProductionReceiverError("predictor-residual y payload parse-back refused") from exc
        expected_shape = (pair_count, scorer_h, scorer_w, channels)
        if decoded_pair.frame0.shape != expected_shape or decoded_pair.frame1.shape != expected_shape:
            raise ProductionReceiverError("predictor-residual payload/header geometry drift")
        _decode_y(y_section)
    elif y_section.codec_id == JXL_PLANE_Y_CODEC_ID:
        try:
            decoded_planes = decode_jxl_plane_payload(y_section.payload)
        except JxlPlaneCodecError as exc:
            raise ProductionReceiverError("jxl-lossless-plane y payload parse-back refused") from exc
        expected_shape = (pair_count, scorer_h, scorer_w, channels)
        if decoded_planes.frame0.shape != expected_shape or decoded_planes.frame1.shape != expected_shape:
            raise ProductionReceiverError("jxl-lossless-plane payload/header geometry drift")
        _decode_y(y_section)
    if section_count == 3:
        residual = sections[2]
        if (
            residual.codec_id != RESIDUAL_CODEC_ID
            or not residual.payload
            or residual.decoded_byte_length != len(residual.payload)
            or residual.video_derived is not True
            or residual.decoded_sha256 != _sha256(residual.payload)
            or len(residual.payload) % RESIDUAL_RECORD.size
        ):
            raise ProductionReceiverError("quotient residual codec/shape/hash drift")
        _parse_residual_updates(
            residual.payload,
            pair_count=pair_count,
            camera_h=camera_h,
            camera_w=camera_w,
            channels=channels,
        )
    return ParsedProductionPacket(packet_bytes, _sha256(packet_bytes), header, tuple(sections))


def decode_y_plane_pair(packet: ParsedProductionPacket) -> DecodedYPlanePair:
    """Expand both charged scorer planes without importing a scorer."""

    if not isinstance(packet, ParsedProductionPacket):
        raise ProductionReceiverError("decode_y_plane_pair requires a parsed production packet")
    _, _, scorer_h, scorer_w, channels = _validate_geometry(packet.header)
    pair_count = packet.header["pair_count"]
    section = packet.section("y_description")
    decoded = _decode_y(section)
    shape = (pair_count, scorer_h, scorer_w, channels)
    plane_bytes = pair_count * scorer_h * scorer_w * channels
    if section.codec_id in (PREDICTOR_RESIDUAL_Y_CODEC_ID, JXL_PLANE_Y_CODEC_ID):
        frame0 = np.frombuffer(decoded[:plane_bytes], dtype=np.uint8).reshape(shape).copy()
        frame1 = np.frombuffer(decoded[plane_bytes:], dtype=np.uint8).reshape(shape).copy()
    else:
        frame1 = np.frombuffer(decoded, dtype=np.uint8).reshape(shape).copy()
        frame0 = frame1.copy()
    return DecodedYPlanePair(frame0=frame0, frame1=frame1)


def decode_y_planes(packet: ParsedProductionPacket) -> np.ndarray:
    """Return frame-1 scorer planes, preserving the legacy public contract."""

    return decode_y_plane_pair(packet).frame1


def _decode_residuals(
    packet: ParsedProductionPacket,
) -> tuple[tuple[QuotientResidualUpdate, ...], ...] | None:
    if packet.header["residual_codec_id"] is None:
        return None
    camera_h, camera_w, _, _, channels = _validate_geometry(packet.header)
    section = packet.section("quotient_residual")
    if _sha256(section.payload) != section.decoded_sha256:
        raise ProductionReceiverError("decoded residual hash custody failure")
    updates = _parse_residual_updates(
        section.payload,
        pair_count=packet.header["pair_count"],
        camera_h=camera_h,
        camera_w=camera_w,
        channels=channels,
    )
    grouped: list[list[QuotientResidualUpdate]] = [[] for _ in range(packet.header["pair_count"])]
    for update in updates:
        grouped[update.pair_index].append(update)
    return tuple(tuple(group) for group in grouped)


def realize_pair_frame1(
    packet: ParsedProductionPacket,
    y_plane: np.ndarray,
    *,
    residual: np.ndarray | Sequence[QuotientResidualUpdate] | None = None,
) -> tuple[np.ndarray, int]:
    """Realize and re-verify one scorer plane, then apply a nullspace residual."""

    camera_h, camera_w, scorer_h, scorer_w, channels = _validate_geometry(packet.header)
    operator = DisjointResizeOperator.build(
        camera_h=camera_h,
        camera_w=camera_w,
        scorer_h=scorer_h,
        scorer_w=scorer_w,
    )
    try:
        frame = realize_factor2_uint8_scorer_plane(operator, y_plane)
    except Uint8LatticeError as exc:
        raise ProductionReceiverError("factor-2 canonical realization refused") from exc
    if residual is not None:
        if isinstance(residual, np.ndarray):
            if residual.dtype != np.dtype("<i2") or residual.shape != (
                camera_h,
                camera_w,
                channels,
            ):
                raise ProductionReceiverError("pair residual has wrong int16 camera geometry")
            frame = np.clip(frame.astype(np.int32) + residual.astype(np.int32), 0, 255).astype(np.uint8)
        else:
            mutable = frame.astype(np.int32)
            for update in residual:
                validated = _validate_residual_update(
                    update,
                    pair_count=packet.header["pair_count"],
                    camera_h=camera_h,
                    camera_w=camera_w,
                    channels=channels,
                )
                mutable[validated.row, validated.col, validated.channel] = np.clip(
                    mutable[validated.row, validated.col, validated.channel] + validated.delta,
                    0,
                    255,
                )
            frame = mutable.astype(np.uint8)
    try:
        proof = verify_factor2_uint8_scorer_plane(operator, frame, y_plane)
    except Uint8LatticeError as exc:
        raise ProductionReceiverError("factor-2 exact verification refused") from exc
    if not proof.numerator_exact or (residual is None and not proof.certified_exact):
        raise ProductionReceiverError("quotient residual left the exact scorer-plane nullspace")
    return np.ascontiguousarray(frame), proof.numerator_equal_values


def _zip_bytes(packet_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as archive:
        info = zipfile.ZipInfo(MEMBER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet_bytes)
    return buffer.getvalue()


def _read_archive_packet(archive_path: Path) -> tuple[bytes, str, int]:
    try:
        archive_bytes = archive_path.read_bytes()
    except OSError as exc:
        raise ProductionReceiverError(f"cannot read archive {archive_path}") from exc
    if len(archive_bytes) > MAX_PACKET_BYTES + MAX_HEADER_BYTES:
        raise ProductionReceiverError("archive.zip exceeds the production size cap")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME or infos[0].is_dir():
                raise ProductionReceiverError("archive.zip must contain exactly one 0.bin member")
            info = infos[0]
            if (
                info.file_size > MAX_PACKET_BYTES
                or info.compress_type != zipfile.ZIP_STORED
                or info.compress_size != info.file_size
                or info.flag_bits & 0x1
            ):
                raise ProductionReceiverError("0.bin must be bounded, unencrypted, and stored without ZIP compression")
            packet_bytes = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ProductionReceiverError("archive.zip cannot be reopened exactly") from exc
    return packet_bytes, _sha256(archive_bytes), len(archive_bytes)


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProductionReceiverError(f"write-once path already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_write_once_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ProductionReceiverError(f"preserved write-once bytes drifted: {path}")
        return
    try:
        _atomic_write_once(path, payload)
    except ProductionReceiverError:
        if not path.is_file() or path.read_bytes() != payload:
            raise


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_production_archive(
    y_planes: np.ndarray,
    *,
    archive_path: Path | str,
    camera_height: int,
    camera_width: int,
    y_codec_id: str = "raw-uint8-y",
    frame0_y_planes: np.ndarray | None = None,
    predictor_modes: PredictorMode | str | int | Sequence[PredictorMode | str | int] | None = None,
    predictor_descriptors: Sequence[bytes] | None = None,
    predictor_pair_ids: Sequence[int] | None = None,
    quotient_residual: np.ndarray | Sequence[QuotientResidualUpdate] | None = None,
    manifest_path: Path | str | None = None,
    jxl_effort: int = 9,
    jxl_workers: int = 1,
) -> ArchiveBuildResult:
    """Build deterministic ``archive.zip`` plus a write-once byte manifest."""

    target = Path(archive_path)
    manifest_target = (
        Path(manifest_path) if manifest_path is not None else target.with_name(f"{target.name}.manifest.json")
    )
    if manifest_target.exists():
        raise ProductionReceiverError(f"write-once manifest path already exists: {manifest_target}")
    packet_bytes = build_packet(
        y_planes,
        camera_height=camera_height,
        camera_width=camera_width,
        y_codec_id=y_codec_id,
        frame0_y_planes=frame0_y_planes,
        predictor_modes=predictor_modes,
        predictor_descriptors=predictor_descriptors,
        predictor_pair_ids=predictor_pair_ids,
        quotient_residual=quotient_residual,
        jxl_effort=jxl_effort,
        jxl_workers=jxl_workers,
    )
    parsed = parse_packet(packet_bytes)
    archive_bytes = _zip_bytes(packet_bytes)
    archive_sha = _sha256(archive_bytes)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "archive_member": MEMBER_NAME,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "packet_bytes": len(packet_bytes),
        "packet_sha256": parsed.packet_sha256,
        "sections": [section.header_row() for section in parsed.sections],
        "counted_section_payload_bytes": parsed.header["counted_section_payload_bytes"],
        "video_derived_payload_bytes": parsed.header["video_derived_payload_bytes"],
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "tie_policy_id": TIE_POLICY_ID,
        "arithmetic_id": ARITHMETIC_ID,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    required_bytes = len(archive_bytes) + len(_canonical_json(manifest)) + (1 << 20)
    preflight = storage_preflight(target.parent, required_bytes)
    manifest["storage_preflight"] = {
        "schema": preflight["schema"],
        "required_bytes": required_bytes,
        "passed": True,
    }
    _ensure_write_once_exact(target, archive_bytes)
    _atomic_write_once(manifest_target, _canonical_json(manifest))
    reopened, reopened_archive_sha, reopened_bytes = _read_archive_packet(target)
    if reopened != packet_bytes or reopened_archive_sha != archive_sha or reopened_bytes != len(archive_bytes):
        raise ProductionReceiverError("archive parse-back differs from built bytes")
    parse_packet(reopened)
    return ArchiveBuildResult(
        target,
        manifest_target,
        archive_sha,
        len(archive_bytes),
        parsed.packet_sha256,
        len(packet_bytes),
        manifest,
    )


def _safe_output_path(output_root: Path, video_name: str) -> Path:
    if not isinstance(video_name, str) or not video_name or video_name.strip() != video_name:
        raise ProductionReceiverError("video name must be a non-empty trimmed relative path")
    pure = PurePosixPath(video_name)
    if pure.is_absolute() or ".." in pure.parts or pure.name in {"", ".", ".."}:
        raise ProductionReceiverError("video name cannot escape the output root")
    relative = Path(*pure.parts).with_suffix(".raw")
    candidate = output_root / relative
    root_resolved = output_root.resolve()
    candidate_parent = candidate.parent.resolve()
    if candidate_parent != root_resolved and root_resolved not in candidate_parent.parents:
        raise ProductionReceiverError("resolved video output escapes the output root")
    return candidate


def _read_single_video_name(video_names_file: Path) -> str:
    try:
        names = [line for line in video_names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise ProductionReceiverError("cannot read video names file") from exc
    if len(names) != 1:
        raise ProductionReceiverError("production archive currently requires exactly one video name")
    return names[0]


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ProductionReceiverError("cannot resolve a storage-preflight filesystem")
        candidate = candidate.parent
    return candidate


def storage_preflight(output_root: Path | str, required_bytes: int) -> Mapping[str, Any]:
    required = _exact_int(required_bytes, "required_bytes", minimum=0)
    filesystem_root = _nearest_existing_parent(Path(output_root))
    free = int(shutil.disk_usage(filesystem_root).free)
    if free < required:
        raise ProductionReceiverError(f"storage preflight refused: need {required} bytes, only {free} free")
    return {
        "schema": "v10_production_storage_preflight.v1",
        "required_bytes": required,
        "free_bytes_at_check": free,
        "passed": True,
    }


def tree_sha256(root: Path | str) -> str:
    """Hash a complete output tree by canonical relative paths and file bytes."""

    base = Path(root)
    if not base.is_dir():
        raise ProductionReceiverError("tree hash root must be a directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ProductionReceiverError("tree hash refuses symlinks and special files")
        relative = path.relative_to(base).as_posix()
        if path.is_dir():
            rows.append({"path": relative, "type": "dir"})
        else:
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return _sha256(_canonical_json(rows))


def _pair_state(
    *,
    packet_sha256: str,
    archive_sha256: str,
    pair_index: int,
    stage_payload: bytes,
    frame0_y_plane: np.ndarray,
    frame1_y_plane: np.ndarray,
    numerator_values: int,
    frame0_is_described: bool,
) -> bytes:
    state = {
        "schema": PAIR_STATE_SCHEMA,
        "packet_sha256": packet_sha256,
        "archive_sha256": archive_sha256,
        "pair_index": pair_index,
        "stage_bytes": len(stage_payload),
        "stage_sha256": _sha256(stage_payload),
        "y_plane_sha256": _sha256(np.ascontiguousarray(frame1_y_plane).tobytes()),
        "numerator_values_verified": numerator_values,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
    }
    # Preserve byte-identical legacy resume state.  The additive two-plane
    # policy binds both independent targets because frame 0 is no longer an
    # implied copy of frame 1.
    if frame0_is_described:
        state["frame0_y_plane_sha256"] = _sha256(np.ascontiguousarray(frame0_y_plane).tobytes())
        state["frame1_y_plane_sha256"] = _sha256(np.ascontiguousarray(frame1_y_plane).tobytes())
    return _canonical_json(state)


def inflate_archive(
    archive_dir: Path | str,
    output_dir: Path | str,
    video_names_file: Path | str,
    *,
    stop_after_pairs: int | None = None,
) -> InflateResult:
    """Inflate a production archive with write-once per-pair crash recovery."""

    archive_root = Path(archive_dir)
    output_root = Path(output_dir)
    video_name = _read_single_video_name(Path(video_names_file))
    raw_path = _safe_output_path(output_root, video_name)
    packet_bytes, archive_sha, _archive_bytes = _read_archive_packet(archive_root / "archive.zip")
    packet = parse_packet(packet_bytes)
    y_pair = decode_y_plane_pair(packet)
    residuals = _decode_residuals(packet)
    pair_count = int(packet.header["pair_count"])
    if stop_after_pairs is None:
        limit = pair_count
    else:
        limit = _exact_int(stop_after_pairs, "stop_after_pairs", minimum=0, maximum=pair_count)
    camera_h, camera_w, _, _, channels = _validate_geometry(packet.header)
    frame_bytes = _checked_product((camera_h, camera_w, channels), "camera frame bytes", maximum=MAX_PACKET_BYTES)
    pair_stage_bytes = frame_bytes * 2
    stage_root = output_root / ".v10-production-receiver" / raw_path.relative_to(output_root).with_suffix("")
    missing_stage_bytes = sum(
        pair_stage_bytes
        for pair_index in range(pair_count)
        if not (stage_root / f"pair-{pair_index:06d}.bin").is_file()
    )
    required = missing_stage_bytes + (0 if raw_path.is_file() else pair_count * pair_stage_bytes) + (1 << 20)
    deterministic_capacity_requirement = pair_count * pair_stage_bytes * 2 + (1 << 20)
    preflight = storage_preflight(output_root, required)

    numerator_values_verified = 0
    preserved = 0
    for pair_index in range(limit):
        residual = None if residuals is None else residuals[pair_index]
        frame1, frame1_numerator_values = realize_pair_frame1(packet, y_pair.frame1[pair_index], residual=residual)
        if packet.header["frame0_policy_id"] == FRAME0_POLICY_ID:
            frame0 = frame1.copy()
            numerator_values = frame1_numerator_values * 2
            state_numerator_values = frame1_numerator_values
        elif packet.header["frame0_policy_id"] == DESCRIPTION_FRAME0_POLICY_ID:
            frame0, frame0_numerator_values = realize_pair_frame1(packet, y_pair.frame0[pair_index])
            numerator_values = frame0_numerator_values + frame1_numerator_values
            state_numerator_values = numerator_values
        else:  # parse_packet already closes this registry; retain local defense.
            raise ProductionReceiverError("unsupported frame0 policy")
        stage_payload = frame0.tobytes(order="C") + frame1.tobytes(order="C")
        if len(stage_payload) != pair_stage_bytes:
            raise ProductionReceiverError("pair stage raw byte count drift")
        state_payload = _pair_state(
            packet_sha256=packet.packet_sha256,
            archive_sha256=archive_sha,
            pair_index=pair_index,
            stage_payload=stage_payload,
            frame0_y_plane=y_pair.frame0[pair_index],
            frame1_y_plane=y_pair.frame1[pair_index],
            numerator_values=state_numerator_values,
            frame0_is_described=packet.header["frame0_policy_id"] == DESCRIPTION_FRAME0_POLICY_ID,
        )
        stage_path = stage_root / f"pair-{pair_index:06d}.bin"
        state_path = stage_root / f"pair-{pair_index:06d}.json"
        _ensure_write_once_exact(stage_path, stage_payload)
        _ensure_write_once_exact(state_path, state_payload)
        # Reopen both preserved legs before allowing progress to the next pair.
        if stage_path.read_bytes() != stage_payload or state_path.read_bytes() != state_payload:
            raise ProductionReceiverError("pair stage/state failed immediate parse-back")
        numerator_values_verified += numerator_values
        preserved += 1

    if limit < pair_count:
        return InflateResult(
            False,
            None,
            None,
            pair_count * pair_stage_bytes,
            preserved,
            numerator_values_verified,
            None,
            preflight,
        )

    hasher = hashlib.sha256()
    partial_path = raw_path.with_name(f"{raw_path.name}.partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{partial_path.name}.", suffix=".tmp", dir=partial_path.parent
    )
    temporary = Path(temporary_name)
    total_bytes = 0
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            for pair_index in range(pair_count):
                stage_path = stage_root / f"pair-{pair_index:06d}.bin"
                state_path = stage_root / f"pair-{pair_index:06d}.json"
                if not stage_path.is_file() or not state_path.is_file():
                    raise ProductionReceiverError("final assembly found a missing preserved pair")
                stage_payload = stage_path.read_bytes()
                state = json.loads(state_path.read_bytes())
                if (
                    _canonical_json(state) != state_path.read_bytes()
                    or state.get("stage_sha256") != _sha256(stage_payload)
                    or state.get("stage_bytes") != len(stage_payload)
                    or state.get("packet_sha256") != packet.packet_sha256
                    or state.get("archive_sha256") != archive_sha
                    or state.get("pair_index") != pair_index
                ):
                    raise ProductionReceiverError("preserved pair state/archive custody drift")
                handle.write(stage_payload)
                hasher.update(stage_payload)
                total_bytes += len(stage_payload)
            handle.flush()
            os.fsync(handle.fileno())
        expected_bytes = pair_count * pair_stage_bytes
        if total_bytes != expected_bytes or temporary.stat().st_size != expected_bytes:
            raise ProductionReceiverError("final raw size assertion failed")
        expected_sha = hasher.hexdigest()
        os.replace(temporary, partial_path)
        if partial_path.stat().st_size != expected_bytes or _sha256_file(partial_path) != expected_sha:
            raise ProductionReceiverError(".partial raw size/hash assertion failed")
        if raw_path.exists():
            if (
                not raw_path.is_file()
                or raw_path.stat().st_size != expected_bytes
                or _sha256_file(raw_path) != expected_sha
            ):
                raise ProductionReceiverError("existing final raw bytes drifted")
            partial_path.unlink()
        else:
            os.replace(partial_path, raw_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema": INFLATE_MANIFEST_SCHEMA,
        "video_name": video_name,
        "raw_relative_path": raw_path.relative_to(output_root).as_posix(),
        "raw_bytes": total_bytes,
        "raw_sha256": expected_sha,
        "archive_sha256": archive_sha,
        "packet_sha256": packet.packet_sha256,
        "pair_count": pair_count,
        "pair_stages_preserved": pair_count,
        "numerator_values_verified": numerator_values_verified,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "tie_policy_id": TIE_POLICY_ID,
        "arithmetic_id": ARITHMETIC_ID,
        "storage_preflight": {
            "schema": preflight["schema"],
            "required_bytes": deterministic_capacity_requirement,
            "passed": True,
        },
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    manifest_path = stage_root / "inflate-manifest.json"
    _ensure_write_once_exact(manifest_path, _canonical_json(manifest))
    tree_hash = tree_sha256(output_root)
    return InflateResult(
        True,
        raw_path,
        expected_sha,
        total_bytes,
        pair_count,
        numerator_values_verified,
        tree_hash,
        preflight,
    )


__all__ = [
    "ARITHMETIC_ID",
    "DESCRIPTION_FRAME0_POLICY_ID",
    "FRAME0_POLICY_ID",
    "FRAME0_POLICY_IDS",
    "INFLATE_MANIFEST_SCHEMA",
    "MAGIC",
    "MANIFEST_SCHEMA",
    "MEMBER_NAME",
    "PACKET_SCHEMA",
    "PAIR_STATE_SCHEMA",
    "PREDICTOR_RESIDUAL_Y_CODEC_ID",
    "PREFIX",
    "RECEIVER_CONTRACT_ID",
    "RESIDUAL_RECORD",
    "SECTION_LENGTH",
    "TIE_POLICY_ID",
    "VERSION",
    "ArchiveBuildResult",
    "DecodedYPlanePair",
    "InflateResult",
    "ParsedProductionPacket",
    "ProductionReceiverError",
    "ProductionSection",
    "QuotientResidualUpdate",
    "build_packet",
    "build_production_archive",
    "decode_y_plane_pair",
    "decode_y_planes",
    "inflate_archive",
    "parse_packet",
    "realize_pair_frame1",
    "storage_preflight",
    "tree_sha256",
]
