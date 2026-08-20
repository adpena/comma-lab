# SPDX-License-Identifier: MIT
"""Counted XIP2 chronological-A receiver for the production PASS-G surface.

``TACX2A3`` closes the receiver gap left by the encoder-only XIP2 guidance in
``taskspace_chronological_a3_encoder``.  The packet contains only protocol
custody plus an optional counted XIP2 trajectory.  Its decoder is generic
NumPy/SE(3) geometry: it never accepts or loads a scorer, target, target-label,
GT table, dense Y0, or dense Y1.

V1 is deliberately source-bound only to
``PassConditionalASurfaceV1``/``PassConditionalASourceBindingV1``.  The older
exact-semantic ``PostG8ConditionalA`` universe is not translated or accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from functools import lru_cache
from typing import Any, Final, Literal

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    WarpRealLumaFrame0Error,
    warp_frame0_native_numpy,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    MAX_BOUNDED_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
)
from tac.witness_dsl.taskspace_chronological_a3_encoder import (
    EncoderOnlyXIP2GuidanceV1,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    PassConditionalAError,
    PassConditionalASourceBindingV1,
    PassConditionalASurfaceV1,
    derive_pass_conditional_a_source_binding,
)
from tac.witness_dsl.taskspace_pass_semantic_g import PassSemanticGEnvelopeMode
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    SE3XiTransportV2,
    TaskspacePredictorStateV2Error,
)

PACKET_MAGIC: Final = b"TACX2A3\x00"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct(">8sBBBBBHHHHHHf32s32sII")
PACKET_FOOTER: Final = struct.Struct(">I")
SOURCE_SCHEMA: Final = "tac.taskspace_counted_xip2_chronological_a3_source.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_counted_xip2_chronological_a3_receipt.v1"
NUMERIC_REFERENCE_ID: Final = "NUMPY_FP32_SAMPLE_FP64_EON_RNE_U8_V1"
ABSENT_GUIDANCE_SHA256: Final = "0" * 64
MAX_XIP2_PAYLOAD_BYTES: Final = 1_048_576


class CountedXIP2ChronologicalA3Error(ValueError):
    """A packet, source foreign key, nested XIP2 body, or receipt failed closed."""


class CountedXIP2A3SourceDomain(StrEnum):
    PASS_G_CONDITIONAL_V1 = "PASS_G_CONDITIONAL_V1"


class CountedXIP2A3Mode(StrEnum):
    PASS_P0_V1 = "PASS_P0_V1"
    COPY_CONDITIONAL_Y1_V1 = "COPY_CONDITIONAL_Y1_V1"
    XIP2_WARP_V1 = "XIP2_WARP_V1"


class CountedXIP2A3Interpretation(StrEnum):
    CAMERA_THEN_R = "CAMERA_THEN_R"
    SCORER_THEN_FACTOR2 = "SCORER_THEN_FACTOR2"


class CountedXIP2A3GeometryProfile(StrEnum):
    EON_GROUND_874X1164_TO_384X512_V1 = "EON_GROUND_874X1164_TO_384X512_V1"


_SOURCE_TO_WIRE: Final = {CountedXIP2A3SourceDomain.PASS_G_CONDITIONAL_V1: 1}
_WIRE_TO_SOURCE: Final = {value: key for key, value in _SOURCE_TO_WIRE.items()}
_MODE_TO_WIRE: Final = {
    CountedXIP2A3Mode.PASS_P0_V1: 0,
    CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1: 1,
    CountedXIP2A3Mode.XIP2_WARP_V1: 2,
}
_WIRE_TO_MODE: Final = {value: key for key, value in _MODE_TO_WIRE.items()}
_INTERPRETATION_TO_WIRE: Final = {
    CountedXIP2A3Interpretation.CAMERA_THEN_R: 0,
    CountedXIP2A3Interpretation.SCORER_THEN_FACTOR2: 1,
}
_WIRE_TO_INTERPRETATION: Final = {value: key for key, value in _INTERPRETATION_TO_WIRE.items()}
_GEOMETRY_TO_WIRE: Final = {CountedXIP2A3GeometryProfile.EON_GROUND_874X1164_TO_384X512_V1: 1}
_WIRE_TO_GEOMETRY: Final = {value: key for key, value in _GEOMETRY_TO_WIRE.items()}
_POSITIVE_ZERO_F32: Final = b"\x00\x00\x00\x00"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CountedXIP2ChronologicalA3Error("counted XIP2 A3 record is not finite canonical ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise CountedXIP2ChronologicalA3Error(f"{label} must be exact bytes")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CountedXIP2ChronologicalA3Error(f"{label} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=no_duplicates)
    except CountedXIP2ChronologicalA3Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CountedXIP2ChronologicalA3Error(f"{label} is not canonical ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise CountedXIP2ChronologicalA3Error(f"{label} is not canonical or changed on re-emit")
    return value


def _require_sha256(value: object, field: str, *, allow_absent: bool = True) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_absent and value == ABSENT_GUIDANCE_SHA256)
    ):
        raise CountedXIP2ChronologicalA3Error(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CountedXIP2ChronologicalA3Error(f"{field} must be an exact integer >= {minimum}")
    return value


def _require_pair_ids(value: object, *, field: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(type(item) is not int for item in value):
        raise CountedXIP2ChronologicalA3Error(f"{field} must be one nonempty exact integer tuple")
    pair_ids = value
    expected = tuple(range(pair_ids[0], pair_ids[0] + len(pair_ids)))
    if pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise CountedXIP2ChronologicalA3Error(f"{field} must be contiguous inside [0,600)")
    if len(pair_ids) > MAX_BOUNDED_PAIRS:
        raise CountedXIP2ChronologicalA3Error(f"{field} exceeds the bounded pair ABI")
    return pair_ids


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _canonical_pitch(value: object) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise CountedXIP2ChronologicalA3Error("pitch must be one finite real scalar")
    packed = struct.pack(">f", float(value))
    pitch = struct.unpack(">f", packed)[0]
    if not math.isfinite(pitch):
        raise CountedXIP2ChronologicalA3Error("pitch escaped finite fp32 range")
    if pitch == 0.0 and packed != _POSITIVE_ZERO_F32:
        raise CountedXIP2ChronologicalA3Error("negative-zero pitch is a forbidden packet alias")
    return pitch


@lru_cache(maxsize=1)
def _resize_operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )


@dataclass(frozen=True, slots=True)
class CountedXIP2ChronologicalA3ProgramV1:
    """Closed V1 program; dense evidence is intentionally not representable."""

    mode: CountedXIP2A3Mode
    interpretation: CountedXIP2A3Interpretation = CountedXIP2A3Interpretation.CAMERA_THEN_R
    geometry_profile: CountedXIP2A3GeometryProfile = CountedXIP2A3GeometryProfile.EON_GROUND_874X1164_TO_384X512_V1
    pitch: float = 0.0
    xip2_payload: bytes = b""
    guidance_source_binding_sha256: str = ABSENT_GUIDANCE_SHA256

    def __post_init__(self) -> None:
        if type(self.mode) is not CountedXIP2A3Mode:
            raise CountedXIP2ChronologicalA3Error("program mode escaped its closed enum")
        if type(self.interpretation) is not CountedXIP2A3Interpretation:
            raise CountedXIP2ChronologicalA3Error("program interpretation escaped its closed enum")
        if type(self.geometry_profile) is not CountedXIP2A3GeometryProfile:
            raise CountedXIP2ChronologicalA3Error("program geometry escaped its closed enum")
        pitch = _canonical_pitch(self.pitch)
        if type(self.xip2_payload) is not bytes:
            raise CountedXIP2ChronologicalA3Error("program XIP2 payload must be exact bytes")
        _require_sha256(self.guidance_source_binding_sha256, "guidance source binding")
        if self.mode in (
            CountedXIP2A3Mode.PASS_P0_V1,
            CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
        ):
            if (
                self.interpretation is not CountedXIP2A3Interpretation.CAMERA_THEN_R
                or struct.pack(">f", pitch) != _POSITIVE_ZERO_F32
                or self.xip2_payload
                or self.guidance_source_binding_sha256 != ABSENT_GUIDANCE_SHA256
            ):
                raise CountedXIP2ChronologicalA3Error(
                    "PASS/COPY require canonical camera-domain, +0 pitch, empty body, and absent guidance"
                )
        else:
            if not self.xip2_payload.startswith(b"XIP2"):
                raise CountedXIP2ChronologicalA3Error("active program requires one exact XIP2 payload")
            if len(self.xip2_payload) > MAX_XIP2_PAYLOAD_BYTES:
                raise CountedXIP2ChronologicalA3Error("active XIP2 payload exceeds the bounded V1 byte ABI")
            _require_sha256(
                self.guidance_source_binding_sha256,
                "active guidance source binding",
                allow_absent=False,
            )
        object.__setattr__(self, "pitch", pitch)


@dataclass(frozen=True, slots=True)
class ParsedCountedXIP2ChronologicalA3PacketV1:
    packet: bytes
    source_domain: CountedXIP2A3SourceDomain
    mode: CountedXIP2A3Mode
    interpretation: CountedXIP2A3Interpretation
    geometry_profile: CountedXIP2A3GeometryProfile
    source_pair_ids: tuple[int, ...]
    pitch: float
    source_binding_sha256: str
    guidance_source_binding_sha256: str
    xip2_payload: bytes
    xip2_crc32: int
    packet_crc32: int
    transport: SE3XiTransportV2 | None

    def __post_init__(self) -> None:
        if type(self.packet) is not bytes or len(self.packet) < PACKET_HEADER.size + PACKET_FOOTER.size:
            raise CountedXIP2ChronologicalA3Error("parsed packet bytes are truncated")
        if type(self.source_domain) is not CountedXIP2A3SourceDomain:
            raise CountedXIP2ChronologicalA3Error("parsed source domain changed")
        if type(self.mode) is not CountedXIP2A3Mode:
            raise CountedXIP2ChronologicalA3Error("parsed mode changed")
        if type(self.interpretation) is not CountedXIP2A3Interpretation:
            raise CountedXIP2ChronologicalA3Error("parsed interpretation changed")
        if type(self.geometry_profile) is not CountedXIP2A3GeometryProfile:
            raise CountedXIP2ChronologicalA3Error("parsed geometry changed")
        _require_pair_ids(self.source_pair_ids, field="parsed source_pair_ids")
        _canonical_pitch(self.pitch)
        _require_sha256(self.source_binding_sha256, "parsed source binding")
        _require_sha256(self.guidance_source_binding_sha256, "parsed guidance binding")
        if type(self.xip2_payload) is not bytes:
            raise CountedXIP2ChronologicalA3Error("parsed XIP2 body changed exact type")
        for name in ("xip2_crc32", "packet_crc32"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                raise CountedXIP2ChronologicalA3Error(f"{name} escaped uint32")
        try:
            header = PACKET_HEADER.unpack_from(self.packet, 0)
            (footer_crc,) = PACKET_FOOTER.unpack_from(
                self.packet,
                len(self.packet) - PACKET_FOOTER.size,
            )
        except struct.error as exc:
            raise CountedXIP2ChronologicalA3Error("parsed packet fixed fields are malformed") from exc
        expected_header = (
            PACKET_MAGIC,
            PACKET_VERSION,
            _SOURCE_TO_WIRE[self.source_domain],
            _MODE_TO_WIRE[self.mode],
            _INTERPRETATION_TO_WIRE[self.interpretation],
            _GEOMETRY_TO_WIRE[self.geometry_profile],
            self.source_pair_ids[0],
            len(self.source_pair_ids),
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            SCORER_HEIGHT,
            SCORER_WIDTH,
            self.pitch,
            bytes.fromhex(self.source_binding_sha256),
            bytes.fromhex(self.guidance_source_binding_sha256),
            len(self.xip2_payload),
            self.xip2_crc32,
        )
        if header != expected_header:
            raise CountedXIP2ChronologicalA3Error("parsed fields differ from exact packet header")
        if (
            len(self.packet) != PACKET_HEADER.size + len(self.xip2_payload) + PACKET_FOOTER.size
            or self.packet[PACKET_HEADER.size : -PACKET_FOOTER.size] != self.xip2_payload
            or zlib.crc32(self.xip2_payload) & 0xFFFFFFFF != self.xip2_crc32
            or footer_crc != self.packet_crc32
            or zlib.crc32(self.packet[: -PACKET_FOOTER.size]) & 0xFFFFFFFF != self.packet_crc32
        ):
            raise CountedXIP2ChronologicalA3Error("parsed packet/body/CRC custody differs")
        if self.mode is CountedXIP2A3Mode.XIP2_WARP_V1:
            if type(self.transport) is not SE3XiTransportV2:
                raise CountedXIP2ChronologicalA3Error("active parsed packet omitted exact XIP2 transport")
            if (
                self.guidance_source_binding_sha256 == ABSENT_GUIDANCE_SHA256
                or not self.xip2_payload.startswith(b"XIP2")
                or len(self.xip2_payload) > MAX_XIP2_PAYLOAD_BYTES
                or self.transport.counted_payload != self.xip2_payload
                or self.transport.source_pair_ids != self.source_pair_ids
            ):
                raise CountedXIP2ChronologicalA3Error("active parsed transport differs from packet custody")
        elif (
            self.transport is not None
            or self.interpretation is not CountedXIP2A3Interpretation.CAMERA_THEN_R
            or struct.pack(">f", self.pitch) != _POSITIVE_ZERO_F32
            or self.xip2_payload
            or self.xip2_crc32 != 0
            or self.guidance_source_binding_sha256 != ABSENT_GUIDANCE_SHA256
        ):
            raise CountedXIP2ChronologicalA3Error("empty parsed mode is not its sole canonical encoding")

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet)

    @property
    def packet_bytes(self) -> int:
        return len(self.packet)

    @property
    def header_bytes(self) -> int:
        return PACKET_HEADER.size

    @property
    def xip2_payload_bytes(self) -> int:
        return len(self.xip2_payload)

    @property
    def footer_bytes(self) -> int:
        return PACKET_FOOTER.size

    @property
    def protocol_overhead_bytes(self) -> int:
        return self.header_bytes + self.footer_bytes


def _pack_packet(
    *,
    source: PassConditionalASourceBindingV1,
    program: CountedXIP2ChronologicalA3ProgramV1,
) -> bytes:
    if type(source) is not PassConditionalASourceBindingV1:
        raise CountedXIP2ChronologicalA3Error("packet source must be exact PASS conditional binding")
    if type(program) is not CountedXIP2ChronologicalA3ProgramV1:
        raise CountedXIP2ChronologicalA3Error("packet program must be exact V1 program")
    body = program.xip2_payload
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        _SOURCE_TO_WIRE[CountedXIP2A3SourceDomain.PASS_G_CONDITIONAL_V1],
        _MODE_TO_WIRE[program.mode],
        _INTERPRETATION_TO_WIRE[program.interpretation],
        _GEOMETRY_TO_WIRE[program.geometry_profile],
        source.pair_start,
        source.pair_count,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        SCORER_HEIGHT,
        SCORER_WIDTH,
        program.pitch,
        bytes.fromhex(source.binding_sha256),
        bytes.fromhex(program.guidance_source_binding_sha256),
        len(body),
        body_crc,
    )
    packet_without_footer = header + body
    packet_crc = zlib.crc32(packet_without_footer) & 0xFFFFFFFF
    return packet_without_footer + PACKET_FOOTER.pack(packet_crc)


def parse_counted_xip2_chronological_a3_packet(
    packet: bytes,
    *,
    expected_source_binding: PassConditionalASourceBindingV1,
) -> ParsedCountedXIP2ChronologicalA3PacketV1:
    """Strict parse with exact source, nested EOF, CRC, and re-encode closure."""

    if type(packet) is not bytes:
        raise CountedXIP2ChronologicalA3Error("packet must be exact bytes")
    if type(expected_source_binding) is not PassConditionalASourceBindingV1:
        raise CountedXIP2ChronologicalA3Error("expected source must be exact PassConditionalASourceBindingV1")
    minimum = PACKET_HEADER.size + PACKET_FOOTER.size
    if len(packet) < minimum:
        raise CountedXIP2ChronologicalA3Error("packet is truncated before fixed header/footer")
    try:
        (
            magic,
            version,
            source_wire,
            mode_wire,
            interpretation_wire,
            geometry_wire,
            pair_start,
            pair_count,
            camera_h,
            camera_w,
            scorer_h,
            scorer_w,
            pitch,
            source_digest,
            guidance_digest,
            body_bytes,
            body_crc,
        ) = PACKET_HEADER.unpack_from(packet, 0)
    except struct.error as exc:
        raise CountedXIP2ChronologicalA3Error("packet header is malformed") from exc
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise CountedXIP2ChronologicalA3Error("packet magic/version changed")
    try:
        source_domain = _WIRE_TO_SOURCE[source_wire]
        mode = _WIRE_TO_MODE[mode_wire]
        interpretation = _WIRE_TO_INTERPRETATION[interpretation_wire]
        geometry_profile = _WIRE_TO_GEOMETRY[geometry_wire]
    except KeyError as exc:
        raise CountedXIP2ChronologicalA3Error("packet contains an unknown closed discriminator") from exc
    if source_domain is not CountedXIP2A3SourceDomain.PASS_G_CONDITIONAL_V1:
        raise CountedXIP2ChronologicalA3Error("packet source domain is not production PASS-G conditional")
    if (camera_h, camera_w, scorer_h, scorer_w) != (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        SCORER_HEIGHT,
        SCORER_WIDTH,
    ):
        raise CountedXIP2ChronologicalA3Error("packet geometry dimensions changed")
    source_pair_ids = tuple(range(pair_start, pair_start + pair_count))
    _require_pair_ids(source_pair_ids, field="packet source_pair_ids")
    if source_pair_ids != expected_source_binding.source_pair_ids:
        raise CountedXIP2ChronologicalA3Error("packet pair window differs from expected source")
    source_hash = source_digest.hex()
    guidance_hash = guidance_digest.hex()
    if source_hash != expected_source_binding.binding_sha256:
        raise CountedXIP2ChronologicalA3Error("packet source binding differs from exact PASS-G source")
    total = PACKET_HEADER.size + body_bytes + PACKET_FOOTER.size
    if len(packet) != total:
        raise CountedXIP2ChronologicalA3Error("packet length/body length is not exact EOF")
    if body_bytes > MAX_XIP2_PAYLOAD_BYTES:
        raise CountedXIP2ChronologicalA3Error("packet XIP2 body exceeds the bounded V1 byte ABI")
    body = packet[PACKET_HEADER.size : PACKET_HEADER.size + body_bytes]
    if zlib.crc32(body) & 0xFFFFFFFF != body_crc:
        raise CountedXIP2ChronologicalA3Error("nested XIP2 body CRC mismatch")
    (packet_crc,) = PACKET_FOOTER.unpack_from(packet, len(packet) - PACKET_FOOTER.size)
    if zlib.crc32(packet[: -PACKET_FOOTER.size]) & 0xFFFFFFFF != packet_crc:
        raise CountedXIP2ChronologicalA3Error("packet CRC mismatch")
    pitch = _canonical_pitch(pitch)
    transport: SE3XiTransportV2 | None = None
    if mode in (CountedXIP2A3Mode.PASS_P0_V1, CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1):
        if (
            interpretation is not CountedXIP2A3Interpretation.CAMERA_THEN_R
            or struct.pack(">f", pitch) != _POSITIVE_ZERO_F32
            or body
            or body_crc != 0
            or guidance_hash != ABSENT_GUIDANCE_SHA256
        ):
            raise CountedXIP2ChronologicalA3Error("PASS/COPY packet has a noncanonical empty-body encoding")
    else:
        _require_sha256(guidance_hash, "packet guidance binding", allow_absent=False)
        try:
            transport = SE3XiTransportV2(
                counted_payload=body,
                source_pair_ids=source_pair_ids,
                predictor_program_sha256=expected_source_binding.predictor_program_sha256,
            )
        except TaskspacePredictorStateV2Error as exc:
            raise CountedXIP2ChronologicalA3Error("nested XIP2 payload failed exact EOF/shape parse") from exc
    program = CountedXIP2ChronologicalA3ProgramV1(
        mode=mode,
        interpretation=interpretation,
        geometry_profile=geometry_profile,
        pitch=pitch,
        xip2_payload=body,
        guidance_source_binding_sha256=guidance_hash,
    )
    canonical = _pack_packet(source=expected_source_binding, program=program)
    if canonical != packet:
        raise CountedXIP2ChronologicalA3Error("packet changed on strict parse/re-encode")
    return ParsedCountedXIP2ChronologicalA3PacketV1(
        packet=packet,
        source_domain=source_domain,
        mode=mode,
        interpretation=interpretation,
        geometry_profile=geometry_profile,
        source_pair_ids=source_pair_ids,
        pitch=pitch,
        source_binding_sha256=source_hash,
        guidance_source_binding_sha256=guidance_hash,
        xip2_payload=body,
        xip2_crc32=body_crc,
        packet_crc32=packet_crc,
        transport=transport,
    )


def reencode_counted_xip2_chronological_a3_packet(
    parsed: ParsedCountedXIP2ChronologicalA3PacketV1,
    *,
    expected_source_binding: PassConditionalASourceBindingV1,
) -> bytes:
    if type(parsed) is not ParsedCountedXIP2ChronologicalA3PacketV1:
        raise CountedXIP2ChronologicalA3Error("re-encode requires exact parsed packet")
    reparsed = parse_counted_xip2_chronological_a3_packet(
        parsed.packet,
        expected_source_binding=expected_source_binding,
    )
    if reparsed.packet != parsed.packet:
        raise CountedXIP2ChronologicalA3Error("parsed packet changed under strict reparse")
    return parsed.packet


@dataclass(frozen=True, slots=True)
class CountedXIP2ChronologicalA3ReceiptV1:
    schema: Literal["tac.taskspace_counted_xip2_chronological_a3_receipt.v1"]
    source_domain: str
    mode: str
    interpretation: str
    geometry_profile: str
    numeric_reference_id: str
    pass_g_mode: str
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    guidance_source_binding_sha256: str
    packet_sha256: str
    packet_bytes: int
    raw_counted_bytes: int
    header_bytes: int
    xip2_payload_bytes: int
    footer_bytes: int
    protocol_overhead_bytes: int
    xip2_payload_sha256: str
    decoded_q_sha256: str
    xip2_crc32: int
    packet_crc32: int
    output_camera_y0_sha256: str
    output_camera_y1_sha256: str
    output_chronology_sha256: str
    guidance_binding_role: Literal["OPAQUE_ENCODER_LINEAGE_ONLY"] = "OPAQUE_ENCODER_LINEAGE_ONLY"
    guidance_binding_receiver_verified: Literal[False] = False
    guidance_binding_source_authority: Literal[False] = False
    strict_eof_crc_parse_reencode: Literal[True] = True
    exact_source_binding_closed: Literal[True] = True
    pass_g_mode_inside_source_binding: Literal[True] = True
    conditional_camera_y1_preserved: Literal[True] = True
    generic_decoder_only: Literal[True] = True
    counted_sufficient_statistics_only: Literal[True] = True
    global_copy_has_empty_body: Literal[True] = True
    numpy_fp32_u8_reference: Literal[True] = True
    dense_target_serialized: Literal[False] = False
    dense_camera_y0_serialized: Literal[False] = False
    dense_camera_y1_serialized: Literal[False] = False
    per_cell_coordinates_serialized: Literal[False] = False
    scorer_serialized: Literal[False] = False
    scorer_output_serialized: Literal[False] = False
    target_labels_serialized: Literal[False] = False
    gt_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_claim: Literal[False] = False
    through_r_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise CountedXIP2ChronologicalA3Error("receipt schema changed")
        try:
            source_domain = CountedXIP2A3SourceDomain(self.source_domain)
            mode = CountedXIP2A3Mode(self.mode)
            CountedXIP2A3Interpretation(self.interpretation)
            CountedXIP2A3GeometryProfile(self.geometry_profile)
        except ValueError as exc:
            raise CountedXIP2ChronologicalA3Error("receipt closed discriminator changed") from exc
        if source_domain is not CountedXIP2A3SourceDomain.PASS_G_CONDITIONAL_V1:
            raise CountedXIP2ChronologicalA3Error("receipt escaped production source domain")
        if self.numeric_reference_id != NUMERIC_REFERENCE_ID:
            raise CountedXIP2ChronologicalA3Error("receipt numerical reference changed")
        if type(self.pass_g_mode) is not str:
            raise CountedXIP2ChronologicalA3Error("receipt PASS-G mode changed")
        try:
            PassSemanticGEnvelopeMode(self.pass_g_mode)
        except ValueError as exc:
            raise CountedXIP2ChronologicalA3Error("receipt PASS-G mode changed") from exc
        _require_pair_ids(self.source_pair_ids, field="receipt source_pair_ids")
        for name in (
            "source_binding_sha256",
            "guidance_source_binding_sha256",
            "packet_sha256",
            "xip2_payload_sha256",
            "decoded_q_sha256",
            "output_camera_y0_sha256",
            "output_camera_y1_sha256",
            "output_chronology_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "packet_bytes",
            "raw_counted_bytes",
            "header_bytes",
            "xip2_payload_bytes",
            "footer_bytes",
            "protocol_overhead_bytes",
            "xip2_crc32",
            "packet_crc32",
        ):
            _require_exact_int(getattr(self, name), name)
        if self.xip2_crc32 > 0xFFFFFFFF or self.packet_crc32 > 0xFFFFFFFF:
            raise CountedXIP2ChronologicalA3Error("receipt CRC escaped uint32")
        if (
            self.header_bytes != PACKET_HEADER.size
            or self.footer_bytes != PACKET_FOOTER.size
            or self.protocol_overhead_bytes != self.header_bytes + self.footer_bytes
            or self.packet_bytes != self.header_bytes + self.xip2_payload_bytes + self.footer_bytes
            or self.raw_counted_bytes != self.packet_bytes
        ):
            raise CountedXIP2ChronologicalA3Error("receipt raw counted-byte arithmetic changed")
        if mode is CountedXIP2A3Mode.XIP2_WARP_V1:
            if (
                self.xip2_payload_bytes < 1
                or self.xip2_payload_sha256 == ABSENT_GUIDANCE_SHA256
                or self.decoded_q_sha256 == ABSENT_GUIDANCE_SHA256
                or self.guidance_source_binding_sha256 == ABSENT_GUIDANCE_SHA256
            ):
                raise CountedXIP2ChronologicalA3Error("active receipt omitted counted XIP2 custody")
        elif (
            self.xip2_payload_bytes != 0
            or self.xip2_payload_sha256 != _sha256(b"")
            or self.decoded_q_sha256 != ABSENT_GUIDANCE_SHA256
            or self.guidance_source_binding_sha256 != ABSENT_GUIDANCE_SHA256
            or self.xip2_crc32 != 0
        ):
            raise CountedXIP2ChronologicalA3Error("empty-mode receipt fabricated XIP2 bytes")
        truth = {
            "guidance_binding_role": "OPAQUE_ENCODER_LINEAGE_ONLY",
            "guidance_binding_receiver_verified": False,
            "guidance_binding_source_authority": False,
            "strict_eof_crc_parse_reencode": True,
            "exact_source_binding_closed": True,
            "pass_g_mode_inside_source_binding": True,
            "conditional_camera_y1_preserved": True,
            "generic_decoder_only": True,
            "counted_sufficient_statistics_only": True,
            "global_copy_has_empty_body": True,
            "numpy_fp32_u8_reference": True,
            "dense_target_serialized": False,
            "dense_camera_y0_serialized": False,
            "dense_camera_y1_serialized": False,
            "per_cell_coordinates_serialized": False,
            "scorer_serialized": False,
            "scorer_output_serialized": False,
            "target_labels_serialized": False,
            "gt_serialized": False,
            "scorer_invoked": False,
            "n600_claim": False,
            "through_r_claim": False,
            "exact_score_claim": False,
            "candidate_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(
            type(getattr(self, name)) is not type(expected) or getattr(self, name) != expected
            for name, expected in truth.items()
        ):
            raise CountedXIP2ChronologicalA3Error("receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_counted_xip2_chronological_a3_receipt(
    payload: bytes,
) -> CountedXIP2ChronologicalA3ReceiptV1:
    value = _decode_canonical_json(payload, label="counted XIP2 A3 receipt")
    expected_fields = {field.name for field in fields(CountedXIP2ChronologicalA3ReceiptV1)}
    if set(value) != expected_fields:
        raise CountedXIP2ChronologicalA3Error("receipt fields differ from the exact V1 schema")
    if type(value.get("source_pair_ids")) is not list:
        raise CountedXIP2ChronologicalA3Error("receipt source_pair_ids must be one JSON list")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        receipt = CountedXIP2ChronologicalA3ReceiptV1(**value)
    except TypeError as exc:
        raise CountedXIP2ChronologicalA3Error("receipt shape is malformed") from exc
    if receipt.to_receipt_bytes() != payload:
        raise CountedXIP2ChronologicalA3Error("receipt changed on strict parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class DecodedCountedXIP2ChronologicalA3V1:
    parsed_packet: ParsedCountedXIP2ChronologicalA3PacketV1
    camera_y0: np.ndarray
    camera_y1: np.ndarray
    chronological_camera_frames: np.ndarray
    receipt: CountedXIP2ChronologicalA3ReceiptV1

    def __post_init__(self) -> None:
        if type(self.parsed_packet) is not ParsedCountedXIP2ChronologicalA3PacketV1:
            raise CountedXIP2ChronologicalA3Error("decoded result requires exact parsed packet")
        if type(self.receipt) is not CountedXIP2ChronologicalA3ReceiptV1:
            raise CountedXIP2ChronologicalA3Error("decoded result requires exact receipt")
        if (
            self.receipt.packet_sha256 != self.parsed_packet.packet_sha256
            or self.receipt.source_domain != self.parsed_packet.source_domain.value
            or self.receipt.mode != self.parsed_packet.mode.value
            or self.receipt.interpretation != self.parsed_packet.interpretation.value
            or self.receipt.geometry_profile != self.parsed_packet.geometry_profile.value
            or self.receipt.source_pair_ids != self.parsed_packet.source_pair_ids
            or self.receipt.source_binding_sha256 != self.parsed_packet.source_binding_sha256
            or self.receipt.guidance_source_binding_sha256 != self.parsed_packet.guidance_source_binding_sha256
            or self.receipt.xip2_payload_bytes != self.parsed_packet.xip2_payload_bytes
            or self.receipt.xip2_payload_sha256 != _sha256(self.parsed_packet.xip2_payload)
            or self.receipt.xip2_crc32 != self.parsed_packet.xip2_crc32
            or self.receipt.packet_crc32 != self.parsed_packet.packet_crc32
        ):
            raise CountedXIP2ChronologicalA3Error("decoded receipt differs from parsed packet custody")
        pair_count = len(self.parsed_packet.source_pair_ids)
        y0 = np.asarray(self.camera_y0)
        y1 = np.asarray(self.camera_y1)
        chronology = np.asarray(self.chronological_camera_frames)
        expected_frame_shape = (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        if y0.dtype != np.uint8 or y0.shape != expected_frame_shape:
            raise CountedXIP2ChronologicalA3Error("decoded Y0 ABI changed")
        if y1.dtype != np.uint8 or y1.shape != expected_frame_shape:
            raise CountedXIP2ChronologicalA3Error("decoded Y1 ABI changed")
        if chronology.dtype != np.uint8 or chronology.shape != (
            pair_count,
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        ):
            raise CountedXIP2ChronologicalA3Error("decoded chronology ABI changed")
        if (
            not np.array_equal(chronology[:, 0], y0)
            or not np.array_equal(chronology[:, 1], y1)
            or _array_sha256(y0) != self.receipt.output_camera_y0_sha256
            or _array_sha256(y1) != self.receipt.output_camera_y1_sha256
            or _array_sha256(chronology) != self.receipt.output_chronology_sha256
        ):
            raise CountedXIP2ChronologicalA3Error("decoded arrays differ from chronology/receipt custody")
        object.__setattr__(self, "camera_y0", _immutable(y0, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "camera_y1", _immutable(y1, dtype=np.dtype(np.uint8)))
        object.__setattr__(
            self,
            "chronological_camera_frames",
            _immutable(chronology, dtype=np.dtype(np.uint8)),
        )


def _realize_y0(
    parsed: ParsedCountedXIP2ChronologicalA3PacketV1,
    surface: PassConditionalASurfaceV1,
) -> np.ndarray:
    if parsed.mode is CountedXIP2A3Mode.PASS_P0_V1:
        return np.ascontiguousarray(surface.predictor_camera_p0, dtype=np.uint8).copy()
    if parsed.mode is CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1:
        return np.ascontiguousarray(surface.conditional_camera_y1, dtype=np.uint8).copy()
    assert parsed.transport is not None
    xi = parsed.transport.xi
    output: list[np.ndarray] = []
    try:
        if parsed.interpretation is CountedXIP2A3Interpretation.CAMERA_THEN_R:
            geometry = GroundHomographyGeom.eon(
                (CAMERA_HEIGHT, CAMERA_WIDTH),
                pitch=parsed.pitch,
            )
            for pair_index in range(len(parsed.source_pair_ids)):
                warped = warp_frame0_native_numpy(
                    surface.conditional_camera_y1[pair_index],
                    xi[pair_index],
                    geometry,
                    compute_dtype=np.float32,
                )
                output.append(np.clip(np.round(warped), 0.0, 255.0).astype(np.uint8))
        else:
            operator = _resize_operator()
            geometry = GroundHomographyGeom.eon(
                (SCORER_HEIGHT, SCORER_WIDTH),
                pitch=parsed.pitch,
            )
            for pair_index in range(len(parsed.source_pair_ids)):
                scorer_y1 = operator.apply(surface.conditional_camera_y1[pair_index])
                warped = warp_frame0_native_numpy(
                    scorer_y1,
                    xi[pair_index],
                    geometry,
                    compute_dtype=np.float32,
                )
                scorer_u8 = np.clip(np.round(warped), 0.0, 255.0).astype(np.uint8)
                output.append(operator.realize_factor2_uint8(scorer_u8))
    except (WarpRealLumaFrame0Error, Uint8LatticeError, np.linalg.LinAlgError) as exc:
        raise CountedXIP2ChronologicalA3Error("generic XIP2 warp receiver failed") from exc
    return np.stack(output, axis=0)


def _receipt_for(
    *,
    parsed: ParsedCountedXIP2ChronologicalA3PacketV1,
    source: PassConditionalASourceBindingV1,
    camera_y0: np.ndarray,
    camera_y1: np.ndarray,
    chronology: np.ndarray,
) -> CountedXIP2ChronologicalA3ReceiptV1:
    decoded_q_sha256 = (
        ABSENT_GUIDANCE_SHA256 if parsed.transport is None else _sha256(memoryview(parsed.transport.q_codes).cast("B"))
    )
    return CountedXIP2ChronologicalA3ReceiptV1(
        schema=RECEIPT_SCHEMA,
        source_domain=parsed.source_domain.value,
        mode=parsed.mode.value,
        interpretation=parsed.interpretation.value,
        geometry_profile=parsed.geometry_profile.value,
        numeric_reference_id=NUMERIC_REFERENCE_ID,
        pass_g_mode=source.pass_g_mode,
        source_pair_ids=parsed.source_pair_ids,
        source_binding_sha256=parsed.source_binding_sha256,
        guidance_source_binding_sha256=parsed.guidance_source_binding_sha256,
        packet_sha256=parsed.packet_sha256,
        packet_bytes=parsed.packet_bytes,
        raw_counted_bytes=parsed.packet_bytes,
        header_bytes=parsed.header_bytes,
        xip2_payload_bytes=parsed.xip2_payload_bytes,
        footer_bytes=parsed.footer_bytes,
        protocol_overhead_bytes=parsed.protocol_overhead_bytes,
        xip2_payload_sha256=_sha256(parsed.xip2_payload),
        decoded_q_sha256=decoded_q_sha256,
        xip2_crc32=parsed.xip2_crc32,
        packet_crc32=parsed.packet_crc32,
        output_camera_y0_sha256=_array_sha256(camera_y0),
        output_camera_y1_sha256=_array_sha256(camera_y1),
        output_chronology_sha256=_array_sha256(chronology),
    )


def _decode_once(
    packet: bytes,
    *,
    surface: PassConditionalASurfaceV1,
) -> DecodedCountedXIP2ChronologicalA3V1:
    if type(surface) is not PassConditionalASurfaceV1:
        raise CountedXIP2ChronologicalA3Error("decode requires exact PassConditionalASurfaceV1")
    try:
        source = derive_pass_conditional_a_source_binding(surface)
    except PassConditionalAError as exc:
        raise CountedXIP2ChronologicalA3Error("PASS-G conditional source derivation failed") from exc
    parsed = parse_counted_xip2_chronological_a3_packet(
        packet,
        expected_source_binding=source,
    )
    camera_y1 = np.ascontiguousarray(surface.conditional_camera_y1, dtype=np.uint8).copy()
    if _array_sha256(camera_y1) != source.conditional_camera_y1_sha256:
        raise CountedXIP2ChronologicalA3Error("supplied conditional Y1 differs from exact source binding")
    camera_y0 = _realize_y0(parsed, surface)
    chronology = np.stack((camera_y0, camera_y1), axis=1)
    receipt = _receipt_for(
        parsed=parsed,
        source=source,
        camera_y0=camera_y0,
        camera_y1=camera_y1,
        chronology=chronology,
    )
    return DecodedCountedXIP2ChronologicalA3V1(
        parsed_packet=parsed,
        camera_y0=camera_y0,
        camera_y1=camera_y1,
        chronological_camera_frames=chronology,
        receipt=receipt,
    )


def decode_counted_xip2_chronological_a3_packet(
    packet: bytes,
    *,
    surface: PassConditionalASurfaceV1,
) -> DecodedCountedXIP2ChronologicalA3V1:
    """Deterministically decode twice and return the exact chronological pair stack."""

    first = _decode_once(packet, surface=surface)
    second = _decode_once(packet, surface=surface)
    if (
        not np.array_equal(first.camera_y0, second.camera_y0)
        or not np.array_equal(first.camera_y1, second.camera_y1)
        or not np.array_equal(first.chronological_camera_frames, second.chronological_camera_frames)
        or first.receipt.to_receipt_bytes() != second.receipt.to_receipt_bytes()
    ):
        raise CountedXIP2ChronologicalA3Error("receiver failed deterministic double decode")
    return first


@dataclass(frozen=True, slots=True)
class CompiledCountedXIP2ChronologicalA3V1:
    program: CountedXIP2ChronologicalA3ProgramV1
    source_binding: PassConditionalASourceBindingV1
    packet: bytes
    parsed_packet: ParsedCountedXIP2ChronologicalA3PacketV1
    decoded: DecodedCountedXIP2ChronologicalA3V1

    def __post_init__(self) -> None:
        if type(self.program) is not CountedXIP2ChronologicalA3ProgramV1:
            raise CountedXIP2ChronologicalA3Error("compiled result requires exact program")
        if type(self.source_binding) is not PassConditionalASourceBindingV1:
            raise CountedXIP2ChronologicalA3Error("compiled result requires exact source binding")
        if type(self.packet) is not bytes or self.packet != self.parsed_packet.packet:
            raise CountedXIP2ChronologicalA3Error("compiled packet differs from parsed custody")
        if type(self.decoded) is not DecodedCountedXIP2ChronologicalA3V1:
            raise CountedXIP2ChronologicalA3Error("compiled result requires exact decoded output")
        if self.decoded.parsed_packet.packet != self.parsed_packet.packet:
            raise CountedXIP2ChronologicalA3Error("compiled parsed/decoded packets differ")
        if (
            self.source_binding.binding_sha256 != self.parsed_packet.source_binding_sha256
            or self.source_binding.source_pair_ids != self.parsed_packet.source_pair_ids
            or self.program.mode is not self.parsed_packet.mode
            or self.program.interpretation is not self.parsed_packet.interpretation
            or self.program.geometry_profile is not self.parsed_packet.geometry_profile
            or struct.pack(">f", self.program.pitch) != struct.pack(">f", self.parsed_packet.pitch)
            or self.program.xip2_payload != self.parsed_packet.xip2_payload
            or self.program.guidance_source_binding_sha256 != self.parsed_packet.guidance_source_binding_sha256
        ):
            raise CountedXIP2ChronologicalA3Error("compiled program/source differ from parsed custody")

    @property
    def receipt(self) -> CountedXIP2ChronologicalA3ReceiptV1:
        return self.decoded.receipt


def compile_counted_xip2_chronological_a3(
    program: CountedXIP2ChronologicalA3ProgramV1,
    *,
    surface: PassConditionalASurfaceV1,
) -> CompiledCountedXIP2ChronologicalA3V1:
    """Compile, strict-parse, re-encode, and execute one bounded V1 program."""

    if type(program) is not CountedXIP2ChronologicalA3ProgramV1:
        raise CountedXIP2ChronologicalA3Error("compile requires exact V1 program")
    if type(surface) is not PassConditionalASurfaceV1:
        raise CountedXIP2ChronologicalA3Error("compile requires exact PassConditionalASurfaceV1")
    try:
        source = derive_pass_conditional_a_source_binding(surface)
    except PassConditionalAError as exc:
        raise CountedXIP2ChronologicalA3Error("PASS-G conditional source derivation failed") from exc
    packet = _pack_packet(source=source, program=program)
    parsed = parse_counted_xip2_chronological_a3_packet(packet, expected_source_binding=source)
    if (
        reencode_counted_xip2_chronological_a3_packet(
            parsed,
            expected_source_binding=source,
        )
        != packet
    ):
        raise CountedXIP2ChronologicalA3Error("compiled packet failed parse/re-encode closure")
    decoded = decode_counted_xip2_chronological_a3_packet(packet, surface=surface)
    strict_receipt = parse_counted_xip2_chronological_a3_receipt(decoded.receipt.to_receipt_bytes())
    if strict_receipt != decoded.receipt:
        raise CountedXIP2ChronologicalA3Error("compiled receipt failed strict parse/re-emit")
    return CompiledCountedXIP2ChronologicalA3V1(
        program=program,
        source_binding=source,
        packet=packet,
        parsed_packet=parsed,
        decoded=decoded,
    )


def compile_counted_xip2_chronological_a3_from_guidance(
    guidance: EncoderOnlyXIP2GuidanceV1,
    *,
    interpretation: CountedXIP2A3Interpretation,
    surface: PassConditionalASurfaceV1,
) -> CompiledCountedXIP2ChronologicalA3V1:
    """Turn G9's exact XIP2 bytes into a real counted receiver program.

    Dense encoder-only guidance targets are deliberately not read; only the
    counted XIP2 sufficient statistic, pitch, pair IDs, and lineage digest are
    consumed.
    """

    if type(guidance) is not EncoderOnlyXIP2GuidanceV1:
        raise CountedXIP2ChronologicalA3Error("guidance compile requires exact G9 guidance type")
    if type(interpretation) is not CountedXIP2A3Interpretation:
        raise CountedXIP2ChronologicalA3Error("guidance compile requires exact interpretation enum")
    if type(surface) is not PassConditionalASurfaceV1:
        raise CountedXIP2ChronologicalA3Error("guidance compile requires exact PASS conditional surface")
    if guidance.source_pair_ids != surface.source_pair_ids:
        raise CountedXIP2ChronologicalA3Error("guidance pair window differs from PASS-G source")
    program = CountedXIP2ChronologicalA3ProgramV1(
        mode=CountedXIP2A3Mode.XIP2_WARP_V1,
        interpretation=interpretation,
        pitch=guidance.pitch,
        xip2_payload=guidance.xip2_payload,
        guidance_source_binding_sha256=guidance.source_binding_sha256,
    )
    return compile_counted_xip2_chronological_a3(program, surface=surface)


def compile_counted_xip2_chronological_a3_pass(
    *,
    surface: PassConditionalASurfaceV1,
    mode: CountedXIP2A3Mode = CountedXIP2A3Mode.PASS_P0_V1,
) -> CompiledCountedXIP2ChronologicalA3V1:
    """Compile canonical empty-body PASS or global conditional-Y1 copy."""

    if type(mode) is not CountedXIP2A3Mode or mode not in (
        CountedXIP2A3Mode.PASS_P0_V1,
        CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    ):
        raise CountedXIP2ChronologicalA3Error("pass helper accepts only PASS_P0 or COPY_CONDITIONAL_Y1")
    return compile_counted_xip2_chronological_a3(
        CountedXIP2ChronologicalA3ProgramV1(mode=mode),
        surface=surface,
    )


def resume_counted_xip2_chronological_a3(
    packet: bytes,
    *,
    expected_packet_sha256: str,
    surface: PassConditionalASurfaceV1,
) -> DecodedCountedXIP2ChronologicalA3V1:
    """Resume solely from exact packet bytes after verifying their expected hash."""

    _require_sha256(expected_packet_sha256, "resume expected packet SHA-256")
    if type(packet) is not bytes or _sha256(packet) != expected_packet_sha256:
        raise CountedXIP2ChronologicalA3Error("resume checkpoint packet hash mismatch")
    return decode_counted_xip2_chronological_a3_packet(packet, surface=surface)


__all__ = [
    "ABSENT_GUIDANCE_SHA256",
    "MAX_XIP2_PAYLOAD_BYTES",
    "NUMERIC_REFERENCE_ID",
    "PACKET_FOOTER",
    "PACKET_HEADER",
    "PACKET_MAGIC",
    "PACKET_VERSION",
    "CompiledCountedXIP2ChronologicalA3V1",
    "CountedXIP2A3GeometryProfile",
    "CountedXIP2A3Interpretation",
    "CountedXIP2A3Mode",
    "CountedXIP2A3SourceDomain",
    "CountedXIP2ChronologicalA3Error",
    "CountedXIP2ChronologicalA3ProgramV1",
    "CountedXIP2ChronologicalA3ReceiptV1",
    "DecodedCountedXIP2ChronologicalA3V1",
    "ParsedCountedXIP2ChronologicalA3PacketV1",
    "compile_counted_xip2_chronological_a3",
    "compile_counted_xip2_chronological_a3_from_guidance",
    "compile_counted_xip2_chronological_a3_pass",
    "decode_counted_xip2_chronological_a3_packet",
    "parse_counted_xip2_chronological_a3_packet",
    "parse_counted_xip2_chronological_a3_receipt",
    "reencode_counted_xip2_chronological_a3_packet",
    "resume_counted_xip2_chronological_a3",
]
