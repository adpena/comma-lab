# SPDX-License-Identifier: MIT
"""Nonempty counted PASS_SEMANTIC_G envelope with optional G8 realization.

This is the production identity semantic coordinate.  It is deliberately not an
empty legacy G and not a forged ``TACG1C`` packet: ``TACGPS1`` is source-bound to
the exact predictor semantics and emits P labels/P1 unchanged.  ``TACPG81`` then
wraps that pass coordinate and either no realization packet or one exact G8R1
repair.  Conditional A can therefore distinguish the lawful base/no-repair
surface from the post-G8 surface.

No scorer or evaluator is invoked and no score/candidate/promotion claim is
made.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Final, Literal

import numpy as np

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    MAX_BOUNDED_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PredictorCameraPairSurfaceV1,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    PACKET_MAGIC as G8_REPAIR_PACKET_MAGIC,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    CompiledSameClassRealizationRepairV1,
    CountedTaskspaceSectionIdentityV1,
    DecodedSameClassRealizationRepairV1,
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairError,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairReceiptV1,
    SameClassRealizationRepairSurfaceV1,
    TaskspaceSectionRoleV1,
    compile_same_class_realization_repair,
    decode_same_class_realization_repair_packet,
    parse_same_class_realization_repair_receipt,
)

PASS_PACKET_MAGIC: Final = b"TACGPS1\x00"
PASS_PACKET_VERSION: Final = 1
PASS_PACKET_HEADER: Final = struct.Struct(">8sBBHH32s32s32s32sII")
ENVELOPE_MAGIC: Final = b"TACPG81\x00"
ENVELOPE_VERSION: Final = 1
ENVELOPE_HEADER: Final = struct.Struct(">8sBBHII32s32sII")
PASS_RECEIPT_SCHEMA: Final = "tac.taskspace_pass_semantic_g_receipt.v1"
ENVELOPE_RECEIPT_SCHEMA: Final = "tac.taskspace_pass_semantic_g_envelope_receipt.v1"
PASS_SEMANTIC_BINDING_SCHEMA: Final = "tac.taskspace_pass_semantic_g_binding.v1"
_ZERO_SHA256: Final = "0" * 64


class PassSemanticGError(ValueError):
    """The pass-semantic packet/envelope or optional repair failed closed."""


class PassSemanticGEnvelopeMode(StrEnum):
    PASS_NO_G8_V1 = "PASS_NO_G8_V1"
    PASS_THEN_G8_V1 = "PASS_THEN_G8_V1"


_MODE_TO_WIRE: Final = {
    PassSemanticGEnvelopeMode.PASS_NO_G8_V1: 0,
    PassSemanticGEnvelopeMode.PASS_THEN_G8_V1: 1,
}
_WIRE_TO_MODE: Final = {wire: mode for mode, wire in _MODE_TO_WIRE.items()}


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256(memoryview(contiguous).cast("B"))


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
        raise PassSemanticGError("pass-semantic record is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PassSemanticGError(f"{field} must be canonical lowercase SHA-256")
    return value


def _validate_pair_ids(value: object, *, field: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(type(pair_id) is not int for pair_id in value):
        raise PassSemanticGError(f"{field} must be one nonempty exact integer tuple")
    pair_ids = value
    expected = tuple(range(pair_ids[0], pair_ids[0] + len(pair_ids)))
    if pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise PassSemanticGError(f"{field} must be contiguous inside [0,600)")
    if len(pair_ids) > MAX_BOUNDED_PAIRS:
        raise PassSemanticGError(f"{field} exceeds bounded pair scope")
    return pair_ids


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class PassSemanticGReceiptV1:
    schema: Literal["tac.taskspace_pass_semantic_g_receipt.v1"]
    source_pair_ids: tuple[int, ...]
    packet_sha256: str
    packet_bytes: int
    packet_header_bytes: int
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str
    predictor_labels_sha256: str
    predictor_camera_p1_sha256: str
    output_labels_sha256: str
    output_camera_y1_sha256: str
    semantic_mode: Literal["PASS_SEMANTIC_G_V1"] = "PASS_SEMANTIC_G_V1"
    semantic_labels_unchanged: Literal[True] = True
    camera_y1_unchanged: Literal[True] = True
    semantic_mutation_count: Literal[0] = 0
    camera_mutation_count: Literal[0] = 0
    packet_nonempty_and_source_bound: Literal[True] = True
    packet_parse_reencode_exact: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    dense_labels_serialized: Literal[False] = False
    dense_camera_y1_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != PASS_RECEIPT_SCHEMA:
            raise PassSemanticGError("pass-semantic receipt schema changed")
        _validate_pair_ids(self.source_pair_ids, field="receipt source_pair_ids")
        for field in (
            "packet_sha256",
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "predictor_surface_binding_sha256",
            "upstream_decode_receipt_sha256",
            "predictor_labels_sha256",
            "predictor_camera_p1_sha256",
            "output_labels_sha256",
            "output_camera_y1_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if (
            type(self.packet_bytes) is not int
            or type(self.packet_header_bytes) is not int
            or self.packet_bytes != self.packet_header_bytes
            or self.packet_header_bytes != PASS_PACKET_HEADER.size
        ):
            raise PassSemanticGError("pass-semantic packet byte accounting changed")
        if (
            self.output_labels_sha256 != self.predictor_labels_sha256
            or self.output_camera_y1_sha256 != self.predictor_camera_p1_sha256
        ):
            raise PassSemanticGError("pass-semantic receipt changed labels or P1")
        truth = {
            "semantic_mode": "PASS_SEMANTIC_G_V1",
            "semantic_labels_unchanged": True,
            "camera_y1_unchanged": True,
            "semantic_mutation_count": 0,
            "camera_mutation_count": 0,
            "packet_nonempty_and_source_bound": True,
            "packet_parse_reencode_exact": True,
            "deterministic_double_decode": True,
            "dense_labels_serialized": False,
            "dense_camera_y1_serialized": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(
            type(getattr(self, field)) is not type(expected) or getattr(self, field) != expected
            for field, expected in truth.items()
        ):
            raise PassSemanticGError("pass-semantic receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class DecodedPassSemanticGV1:
    semantic_labels: np.ndarray
    camera_y1: np.ndarray
    receipt: PassSemanticGReceiptV1

    def __post_init__(self) -> None:
        if type(self.receipt) is not PassSemanticGReceiptV1:
            raise PassSemanticGError("decoded pass-semantic output requires exact receipt type")
        pair_count = len(self.receipt.source_pair_ids)
        labels = np.asarray(self.semantic_labels)
        camera = np.asarray(self.camera_y1)
        if labels.dtype != np.uint8 or labels.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH):
            raise PassSemanticGError("decoded pass-semantic labels ABI changed")
        if camera.dtype != np.uint8 or camera.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
            raise PassSemanticGError("decoded pass-semantic camera Y1 ABI changed")
        if (
            _array_sha256(labels) != self.receipt.output_labels_sha256
            or _array_sha256(camera) != self.receipt.output_camera_y1_sha256
        ):
            raise PassSemanticGError("decoded pass-semantic arrays differ from receipt")
        object.__setattr__(self, "semantic_labels", _immutable(labels, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "camera_y1", _immutable(camera, dtype=np.dtype(np.uint8)))


def _encode_pass_packet(predictor_surface: PredictorCameraPairSurfaceV1) -> bytes:
    if type(predictor_surface) is not PredictorCameraPairSurfaceV1:
        raise PassSemanticGError("PASS_SEMANTIC_G requires exact predictor surface")
    state = predictor_surface.predictor_state
    if predictor_surface.upstream_decode_receipt_sha256 is None:
        raise PassSemanticGError("PASS_SEMANTIC_G requires strict P decode receipt custody")
    fixed = (
        PASS_PACKET_MAGIC,
        PASS_PACKET_VERSION,
        0,
        state.source_pair_ids[0],
        state.pair_count,
        bytes.fromhex(state.semantic_binding_sha256),
        bytes.fromhex(state.predictor_program_sha256),
        bytes.fromhex(state.labels_sha256),
        bytes.fromhex(predictor_surface.binding_sha256),
        PASS_PACKET_HEADER.size,
    )
    zero = PASS_PACKET_HEADER.pack(*fixed, 0)
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    return PASS_PACKET_HEADER.pack(*fixed, crc)


def _parse_pass_packet(
    packet: bytes,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
) -> None:
    if type(packet) is not bytes or len(packet) < PASS_PACKET_HEADER.size:
        raise PassSemanticGError("PASS_SEMANTIC_G packet is shorter than its exact header")
    if type(predictor_surface) is not PredictorCameraPairSurfaceV1:
        raise PassSemanticGError("PASS_SEMANTIC_G parse requires exact predictor surface")
    try:
        (
            magic,
            version,
            flags,
            pair_start,
            pair_count,
            semantic_raw,
            program_raw,
            labels_raw,
            surface_raw,
            packet_bytes,
            crc,
        ) = PASS_PACKET_HEADER.unpack_from(packet)
    except struct.error as exc:
        raise PassSemanticGError("PASS_SEMANTIC_G packet length/trailing bytes changed") from exc
    state = predictor_surface.predictor_state
    if magic != PASS_PACKET_MAGIC or version != PASS_PACKET_VERSION or flags != 0:
        raise PassSemanticGError("PASS_SEMANTIC_G packet magic/version/flags changed")
    if (
        pair_start != state.source_pair_ids[0]
        or pair_count != state.pair_count
        or semantic_raw.hex() != state.semantic_binding_sha256
        or program_raw.hex() != state.predictor_program_sha256
        or labels_raw.hex() != state.labels_sha256
        or surface_raw.hex() != predictor_surface.binding_sha256
    ):
        raise PassSemanticGError("PASS_SEMANTIC_G packet differs from exact live predictor source")
    if packet_bytes != len(packet) or packet_bytes != PASS_PACKET_HEADER.size:
        raise PassSemanticGError("PASS_SEMANTIC_G packet length/trailing bytes changed")
    zero = PASS_PACKET_HEADER.pack(
        magic,
        version,
        flags,
        pair_start,
        pair_count,
        semantic_raw,
        program_raw,
        labels_raw,
        surface_raw,
        packet_bytes,
        0,
    )
    if zlib.crc32(zero) & 0xFFFFFFFF != crc:
        raise PassSemanticGError("PASS_SEMANTIC_G packet CRC differs from exact bytes")
    if _encode_pass_packet(predictor_surface) != packet:
        raise PassSemanticGError("PASS_SEMANTIC_G packet changed on strict parse/re-encode")


def _decode_pass_once(
    packet: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
) -> DecodedPassSemanticGV1:
    _parse_pass_packet(packet, predictor_surface=predictor_surface)
    state = predictor_surface.predictor_state
    receipt = PassSemanticGReceiptV1(
        schema=PASS_RECEIPT_SCHEMA,
        source_pair_ids=state.source_pair_ids,
        packet_sha256=_sha256(packet),
        packet_bytes=len(packet),
        packet_header_bytes=PASS_PACKET_HEADER.size,
        predictor_state_binding_sha256=state.binding_sha256,
        predictor_semantic_binding_sha256=state.semantic_binding_sha256,
        predictor_program_sha256=state.predictor_program_sha256,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        predictor_surface_binding_sha256=predictor_surface.binding_sha256,
        upstream_decode_receipt_sha256=predictor_surface.upstream_decode_receipt_sha256,  # type: ignore[arg-type]
        predictor_labels_sha256=state.labels_sha256,
        predictor_camera_p1_sha256=predictor_surface.camera_p1_sha256,
        output_labels_sha256=state.labels_sha256,
        output_camera_y1_sha256=predictor_surface.camera_p1_sha256,
    )
    return DecodedPassSemanticGV1(
        semantic_labels=state.labels,
        camera_y1=predictor_surface.camera_p1,
        receipt=receipt,
    )


def decode_pass_semantic_g_packet(
    packet: bytes,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
) -> DecodedPassSemanticGV1:
    first = _decode_pass_once(packet, predictor_surface)
    second = _decode_pass_once(packet, predictor_surface)
    if first.receipt != second.receipt:
        raise PassSemanticGError("PASS_SEMANTIC_G deterministic double decode changed receipt")
    if not np.array_equal(first.semantic_labels, second.semantic_labels) or not np.array_equal(
        first.camera_y1, second.camera_y1
    ):
        raise PassSemanticGError("PASS_SEMANTIC_G deterministic double decode changed arrays")
    if parse_pass_semantic_g_receipt(first.receipt.to_receipt_bytes()) != first.receipt:
        raise PassSemanticGError("PASS_SEMANTIC_G receipt failed strict parse/re-emit")
    return first


def compile_pass_semantic_g_packet(
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
) -> tuple[bytes, DecodedPassSemanticGV1]:
    packet = _encode_pass_packet(predictor_surface)
    return packet, decode_pass_semantic_g_packet(packet, predictor_surface=predictor_surface)


def parse_pass_semantic_g_receipt(payload: bytes) -> PassSemanticGReceiptV1:
    if type(payload) is not bytes or not payload:
        raise PassSemanticGError("PASS_SEMANTIC_G receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PassSemanticGError(f"PASS_SEMANTIC_G receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except PassSemanticGError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PassSemanticGError("PASS_SEMANTIC_G receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(PassSemanticGReceiptV1.__dataclass_fields__):
        raise PassSemanticGError("PASS_SEMANTIC_G receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise PassSemanticGError("PASS_SEMANTIC_G receipt is not canonical JSON")
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise PassSemanticGError("PASS_SEMANTIC_G receipt pair IDs must be exact integers")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        receipt = PassSemanticGReceiptV1(**value)
    except TypeError as exc:
        raise PassSemanticGError("PASS_SEMANTIC_G receipt contains invalid typed values") from exc
    if receipt.to_receipt_bytes() != payload:
        raise PassSemanticGError("PASS_SEMANTIC_G receipt changed on strict parse-back")
    return receipt


@dataclass(frozen=True, slots=True)
class ParsedPassSemanticGEnvelopeV1:
    envelope_bytes: bytes
    mode: PassSemanticGEnvelopeMode
    pass_packet: bytes
    repair_packet: bytes | None
    envelope_sha256: str
    pass_packet_sha256: str
    repair_packet_sha256: str | None
    header_bytes: int


def compose_pass_semantic_g_envelope(
    pass_packet: bytes,
    repair_packet: bytes | None = None,
) -> bytes:
    if type(pass_packet) is not bytes or not pass_packet.startswith(PASS_PACKET_MAGIC):
        raise PassSemanticGError("pass-semantic envelope requires nonempty exact TACGPS1 bytes")
    if repair_packet is None:
        mode = PassSemanticGEnvelopeMode.PASS_NO_G8_V1
        repair = b""
        repair_sha = _ZERO_SHA256
    else:
        if type(repair_packet) is not bytes or not repair_packet.startswith(G8_REPAIR_PACKET_MAGIC):
            raise PassSemanticGError("pass-semantic G8 envelope requires exact nonempty G8R1 bytes")
        mode = PassSemanticGEnvelopeMode.PASS_THEN_G8_V1
        repair = repair_packet
        repair_sha = _sha256(repair)
    body = pass_packet + repair
    envelope_bytes = ENVELOPE_HEADER.size + len(body)
    if envelope_bytes > 0xFFFFFFFF:
        raise PassSemanticGError("pass-semantic envelope exceeds the version-1 byte ABI")
    fixed = (
        ENVELOPE_MAGIC,
        ENVELOPE_VERSION,
        _MODE_TO_WIRE[mode],
        0,
        len(pass_packet),
        len(repair),
        bytes.fromhex(_sha256(pass_packet)),
        bytes.fromhex(repair_sha),
        envelope_bytes,
    )
    zero = ENVELOPE_HEADER.pack(*fixed, 0)
    crc = zlib.crc32(zero + body) & 0xFFFFFFFF
    return ENVELOPE_HEADER.pack(*fixed, crc) + body


def parse_pass_semantic_g_envelope(envelope: bytes) -> ParsedPassSemanticGEnvelopeV1:
    if type(envelope) is not bytes or len(envelope) < ENVELOPE_HEADER.size:
        raise PassSemanticGError("pass-semantic envelope is shorter than its exact header")
    try:
        (
            magic,
            version,
            mode_wire,
            reserved,
            pass_bytes,
            repair_bytes,
            pass_sha_raw,
            repair_sha_raw,
            envelope_bytes,
            crc,
        ) = ENVELOPE_HEADER.unpack_from(envelope)
    except struct.error as exc:  # pragma: no cover - guarded length
        raise PassSemanticGError("pass-semantic envelope header is malformed") from exc
    if magic != ENVELOPE_MAGIC or version != ENVELOPE_VERSION or reserved != 0:
        raise PassSemanticGError("pass-semantic envelope magic/version/flags changed")
    try:
        mode = _WIRE_TO_MODE[mode_wire]
    except KeyError as exc:
        raise PassSemanticGError("pass-semantic envelope mode is outside the closed universe") from exc
    if (
        pass_bytes < 1
        or envelope_bytes != len(envelope)
        or envelope_bytes != ENVELOPE_HEADER.size + pass_bytes + repair_bytes
    ):
        raise PassSemanticGError("pass-semantic envelope length accounting or trailing bytes changed")
    if mode is PassSemanticGEnvelopeMode.PASS_NO_G8_V1:
        if repair_bytes != 0 or repair_sha_raw.hex() != _ZERO_SHA256:
            raise PassSemanticGError("PASS_NO_G8 envelope contains fake repair bytes or hash")
    elif repair_bytes < 1 or repair_sha_raw.hex() == _ZERO_SHA256:
        raise PassSemanticGError("PASS_THEN_G8 envelope omitted its exact repair bytes")
    body = envelope[ENVELOPE_HEADER.size :]
    pass_packet = body[:pass_bytes]
    repair_raw = body[pass_bytes:]
    if not pass_packet.startswith(PASS_PACKET_MAGIC) or _sha256(pass_packet) != pass_sha_raw.hex():
        raise PassSemanticGError("pass-semantic envelope inner pass packet hash/domain changed")
    repair_packet: bytes | None = None
    repair_sha: str | None = None
    if mode is PassSemanticGEnvelopeMode.PASS_THEN_G8_V1:
        if not repair_raw.startswith(G8_REPAIR_PACKET_MAGIC) or _sha256(repair_raw) != repair_sha_raw.hex():
            raise PassSemanticGError("pass-semantic envelope inner G8R1 hash/domain changed")
        repair_packet = repair_raw
        repair_sha = repair_sha_raw.hex()
    zero = ENVELOPE_HEADER.pack(
        magic,
        version,
        mode_wire,
        reserved,
        pass_bytes,
        repair_bytes,
        pass_sha_raw,
        repair_sha_raw,
        envelope_bytes,
        0,
    )
    if zlib.crc32(zero + body) & 0xFFFFFFFF != crc:
        raise PassSemanticGError("pass-semantic envelope CRC differs from exact bytes")
    if compose_pass_semantic_g_envelope(pass_packet, repair_packet) != envelope:
        raise PassSemanticGError("pass-semantic envelope changed on strict parse/re-encode")
    return ParsedPassSemanticGEnvelopeV1(
        envelope_bytes=envelope,
        mode=mode,
        pass_packet=pass_packet,
        repair_packet=repair_packet,
        envelope_sha256=_sha256(envelope),
        pass_packet_sha256=_sha256(pass_packet),
        repair_packet_sha256=repair_sha,
        header_bytes=ENVELOPE_HEADER.size,
    )


def _pass_binding_sha256(decoded: DecodedPassSemanticGV1) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema": PASS_SEMANTIC_BINDING_SCHEMA,
                "pass_packet_sha256": decoded.receipt.packet_sha256,
                "pass_receipt_sha256": decoded.receipt.receipt_sha256,
                "predictor_semantic_binding_sha256": decoded.receipt.predictor_semantic_binding_sha256,
                "output_labels_sha256": decoded.receipt.output_labels_sha256,
                "output_camera_y1_sha256": decoded.receipt.output_camera_y1_sha256,
                "source_pair_ids": list(decoded.receipt.source_pair_ids),
            }
        )
    )


def build_pass_semantic_g8_surface(
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_semantic: DecodedPassSemanticGV1,
    predictor_codec_id: str,
) -> SameClassRealizationRepairSurfaceV1:
    if type(predictor_section_payload) is not bytes or predictor_section_payload != (
        predictor_surface.predictor_state.predictor_program
    ):
        raise PassSemanticGError("pass-semantic G8 surface predictor bytes differ from typed P")
    if type(pass_semantic) is not DecodedPassSemanticGV1:
        raise PassSemanticGError("pass-semantic G8 surface requires exact pass decode")
    state = predictor_surface.predictor_state
    receipt = pass_semantic.receipt
    if (
        receipt.source_pair_ids != state.source_pair_ids
        or receipt.predictor_state_binding_sha256 != state.binding_sha256
        or receipt.predictor_semantic_binding_sha256 != state.semantic_binding_sha256
        or receipt.predictor_program_sha256 != state.predictor_program_sha256
        or receipt.predictor_renderer_sha256 != state.predictor_renderer_sha256
        or receipt.predictor_surface_binding_sha256 != predictor_surface.binding_sha256
        or receipt.upstream_decode_receipt_sha256 != predictor_surface.upstream_decode_receipt_sha256
        or receipt.predictor_labels_sha256 != state.labels_sha256
        or receipt.predictor_camera_p1_sha256 != predictor_surface.camera_p1_sha256
    ):
        raise PassSemanticGError("pass-semantic G8 surface differs from exact live predictor custody")
    return SameClassRealizationRepairSurfaceV1(
        source_pair_ids=pass_semantic.receipt.source_pair_ids,
        corrected_y1_camera=pass_semantic.camera_y1,
        g_semantic_labels=pass_semantic.semantic_labels,
        g_semantic_binding_sha256=_pass_binding_sha256(pass_semantic),
        p_section_identity=CountedTaskspaceSectionIdentityV1(
            role=TaskspaceSectionRoleV1.P,
            section_index=0,
            codec_id=predictor_codec_id,
            payload_bytes=len(predictor_section_payload),
            payload_sha256=_sha256(predictor_section_payload),
        ),
        g_section_identity=CountedTaskspaceSectionIdentityV1(
            role=TaskspaceSectionRoleV1.G,
            section_index=1,
            codec_id="TACGPS1.v1",
            payload_bytes=pass_semantic.receipt.packet_bytes,
            payload_sha256=pass_semantic.receipt.packet_sha256,
        ),
    )


@dataclass(frozen=True, slots=True)
class PassSemanticGEnvelopeReceiptV1:
    schema: Literal["tac.taskspace_pass_semantic_g_envelope_receipt.v1"]
    mode: str
    source_pair_ids: tuple[int, ...]
    envelope_sha256: str
    envelope_bytes: int
    envelope_header_bytes: int
    pass_packet_sha256: str
    pass_packet_bytes: int
    pass_receipt: PassSemanticGReceiptV1
    pass_receipt_sha256: str
    repair_packet_sha256: str | None
    repair_packet_bytes: int
    repair_receipt: SameClassRealizationRepairReceiptV1 | None
    repair_receipt_sha256: str | None
    semantic_labels_sha256: str
    pre_repair_camera_y1_sha256: str
    output_camera_y1_sha256: str
    pass_semantic_packet_nonempty: Literal[True] = True
    pass_semantic_source_bound: Literal[True] = True
    semantic_labels_unchanged: Literal[True] = True
    optional_g8_mode_explicit: Literal[True] = True
    no_empty_or_fabricated_legacy_g: Literal[True] = True
    envelope_parse_reencode_exact: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    dense_labels_serialized: Literal[False] = False
    dense_camera_y1_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    through_r_target_realization_debt_closed: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != ENVELOPE_RECEIPT_SCHEMA:
            raise PassSemanticGError("pass-semantic envelope receipt schema changed")
        try:
            mode = PassSemanticGEnvelopeMode(self.mode)
        except ValueError as exc:
            raise PassSemanticGError("pass-semantic envelope receipt mode changed") from exc
        _validate_pair_ids(self.source_pair_ids, field="envelope receipt source_pair_ids")
        for field in (
            "envelope_sha256",
            "pass_packet_sha256",
            "pass_receipt_sha256",
            "semantic_labels_sha256",
            "pre_repair_camera_y1_sha256",
            "output_camera_y1_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        for field in ("envelope_bytes", "envelope_header_bytes", "pass_packet_bytes", "repair_packet_bytes"):
            if type(getattr(self, field)) is not int:
                raise PassSemanticGError("pass-semantic envelope receipt counts must be exact integers")
        if (
            self.envelope_header_bytes != ENVELOPE_HEADER.size
            or self.pass_packet_bytes != PASS_PACKET_HEADER.size
            or self.envelope_bytes != self.envelope_header_bytes + self.pass_packet_bytes + self.repair_packet_bytes
        ):
            raise PassSemanticGError("pass-semantic envelope receipt byte accounting changed")
        if type(self.pass_receipt) is not PassSemanticGReceiptV1:
            raise PassSemanticGError("pass-semantic envelope requires exact pass receipt")
        if (
            self.pass_receipt.receipt_sha256 != self.pass_receipt_sha256
            or self.pass_receipt.packet_sha256 != self.pass_packet_sha256
            or self.pass_receipt.packet_bytes != self.pass_packet_bytes
            or self.pass_receipt.source_pair_ids != self.source_pair_ids
            or self.pass_receipt.output_labels_sha256 != self.semantic_labels_sha256
            or self.pass_receipt.output_camera_y1_sha256 != self.pre_repair_camera_y1_sha256
        ):
            raise PassSemanticGError("pass-semantic envelope pass custody does not close")
        if mode is PassSemanticGEnvelopeMode.PASS_NO_G8_V1:
            if (
                self.repair_packet_sha256 is not None
                or self.repair_packet_bytes != 0
                or self.repair_receipt is not None
                or self.repair_receipt_sha256 is not None
                or self.output_camera_y1_sha256 != self.pre_repair_camera_y1_sha256
            ):
                raise PassSemanticGError("PASS_NO_G8 receipt contains a fake repair or changed Y1")
        else:
            if self.repair_packet_sha256 is None or self.repair_receipt_sha256 is None:
                raise PassSemanticGError("PASS_THEN_G8 receipt omitted repair hashes")
            _require_sha256(self.repair_packet_sha256, "repair_packet_sha256")
            _require_sha256(self.repair_receipt_sha256, "repair_receipt_sha256")
            if type(self.repair_receipt) is not SameClassRealizationRepairReceiptV1:
                raise PassSemanticGError("PASS_THEN_G8 receipt requires exact G8 receipt")
            if (
                self.repair_packet_bytes < 1
                or self.output_camera_y1_sha256 == self.pre_repair_camera_y1_sha256
                or self.repair_receipt.packet_sha256 != self.repair_packet_sha256
                or self.repair_receipt.packet_bytes != self.repair_packet_bytes
                or self.repair_receipt.receipt_sha256 != self.repair_receipt_sha256
                or self.repair_receipt.source_pair_ids != self.source_pair_ids
                or self.repair_receipt.corrected_y1_camera_sha256 != self.pre_repair_camera_y1_sha256
                or self.repair_receipt.output_camera_y1_sha256 != self.output_camera_y1_sha256
                or self.repair_receipt.output_g_labels_sha256 != self.semantic_labels_sha256
            ):
                raise PassSemanticGError("PASS_THEN_G8 receipt repair custody does not close")
        truth = {
            "pass_semantic_packet_nonempty": True,
            "pass_semantic_source_bound": True,
            "semantic_labels_unchanged": True,
            "optional_g8_mode_explicit": True,
            "no_empty_or_fabricated_legacy_g": True,
            "envelope_parse_reencode_exact": True,
            "deterministic_double_decode": True,
            "dense_labels_serialized": False,
            "dense_camera_y1_serialized": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "through_r_target_realization_debt_closed": False,
            "research_only": True,
        }
        if any(
            type(getattr(self, field)) is not type(expected) or getattr(self, field) != expected
            for field, expected in truth.items()
        ):
            raise PassSemanticGError("pass-semantic envelope receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["pass_receipt"] = self.pass_receipt.as_dict()
        value["repair_receipt"] = self.repair_receipt.as_dict() if self.repair_receipt is not None else None
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class DecodedPassSemanticGEnvelopeV1:
    parsed_envelope: ParsedPassSemanticGEnvelopeV1
    pass_semantic: DecodedPassSemanticGV1
    repair_surface: SameClassRealizationRepairSurfaceV1
    decoded_repair: DecodedSameClassRealizationRepairV1 | None
    receipt: PassSemanticGEnvelopeReceiptV1

    def __post_init__(self) -> None:
        if type(self.parsed_envelope) is not ParsedPassSemanticGEnvelopeV1:
            raise PassSemanticGError("decoded pass envelope requires exact parsed type")
        if type(self.pass_semantic) is not DecodedPassSemanticGV1:
            raise PassSemanticGError("decoded pass envelope requires exact pass output")
        if type(self.repair_surface) is not SameClassRealizationRepairSurfaceV1:
            raise PassSemanticGError("decoded pass envelope requires exact repair surface")
        if type(self.receipt) is not PassSemanticGEnvelopeReceiptV1:
            raise PassSemanticGError("decoded pass envelope requires exact receipt type")
        if self.receipt.envelope_sha256 != self.parsed_envelope.envelope_sha256:
            raise PassSemanticGError("decoded pass envelope differs from receipt")
        if self.parsed_envelope.mode is PassSemanticGEnvelopeMode.PASS_NO_G8_V1:
            if self.decoded_repair is not None:
                raise PassSemanticGError("decoded PASS_NO_G8 unexpectedly contains a repair")
        elif type(self.decoded_repair) is not DecodedSameClassRealizationRepairV1:
            raise PassSemanticGError("decoded PASS_THEN_G8 omitted exact repair output")

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return self.pass_semantic.receipt.source_pair_ids

    @property
    def semantic_labels(self) -> np.ndarray:
        return self.pass_semantic.semantic_labels

    @property
    def pre_repair_camera_y1(self) -> np.ndarray:
        return self.pass_semantic.camera_y1

    @property
    def conditional_camera_y1(self) -> np.ndarray:
        if self.decoded_repair is None:
            return self.pass_semantic.camera_y1
        return self.decoded_repair.camera_y1


def _decode_envelope_once(
    envelope: bytes,
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    predictor_codec_id: str,
) -> DecodedPassSemanticGEnvelopeV1:
    parsed = parse_pass_semantic_g_envelope(envelope)
    pass_semantic = decode_pass_semantic_g_packet(
        parsed.pass_packet,
        predictor_surface=predictor_surface,
    )
    surface = build_pass_semantic_g8_surface(
        predictor_section_payload=predictor_section_payload,
        predictor_surface=predictor_surface,
        pass_semantic=pass_semantic,
        predictor_codec_id=predictor_codec_id,
    )
    decoded_repair: DecodedSameClassRealizationRepairV1 | None = None
    if parsed.mode is PassSemanticGEnvelopeMode.PASS_THEN_G8_V1:
        assert parsed.repair_packet is not None
        try:
            decoded_repair = decode_same_class_realization_repair_packet(
                parsed.repair_packet,
                surface=surface,
            )
        except SameClassRealizationRepairError as exc:
            raise PassSemanticGError("pass-semantic envelope G8 repair failed exact decode") from exc
    output_y1 = pass_semantic.camera_y1 if decoded_repair is None else decoded_repair.camera_y1
    repair_receipt = None if decoded_repair is None else decoded_repair.receipt
    receipt = PassSemanticGEnvelopeReceiptV1(
        schema=ENVELOPE_RECEIPT_SCHEMA,
        mode=parsed.mode.value,
        source_pair_ids=pass_semantic.receipt.source_pair_ids,
        envelope_sha256=parsed.envelope_sha256,
        envelope_bytes=len(envelope),
        envelope_header_bytes=parsed.header_bytes,
        pass_packet_sha256=parsed.pass_packet_sha256,
        pass_packet_bytes=len(parsed.pass_packet),
        pass_receipt=pass_semantic.receipt,
        pass_receipt_sha256=pass_semantic.receipt.receipt_sha256,
        repair_packet_sha256=parsed.repair_packet_sha256,
        repair_packet_bytes=0 if parsed.repair_packet is None else len(parsed.repair_packet),
        repair_receipt=repair_receipt,
        repair_receipt_sha256=None if repair_receipt is None else repair_receipt.receipt_sha256,
        semantic_labels_sha256=_array_sha256(pass_semantic.semantic_labels),
        pre_repair_camera_y1_sha256=_array_sha256(pass_semantic.camera_y1),
        output_camera_y1_sha256=_array_sha256(output_y1),
    )
    return DecodedPassSemanticGEnvelopeV1(
        parsed_envelope=parsed,
        pass_semantic=pass_semantic,
        repair_surface=surface,
        decoded_repair=decoded_repair,
        receipt=receipt,
    )


def decode_pass_semantic_g_envelope(
    envelope: bytes,
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    predictor_codec_id: str = "LVLS1.v1",
) -> DecodedPassSemanticGEnvelopeV1:
    first = _decode_envelope_once(
        envelope,
        predictor_section_payload=predictor_section_payload,
        predictor_surface=predictor_surface,
        predictor_codec_id=predictor_codec_id,
    )
    second = _decode_envelope_once(
        envelope,
        predictor_section_payload=predictor_section_payload,
        predictor_surface=predictor_surface,
        predictor_codec_id=predictor_codec_id,
    )
    if first.receipt != second.receipt:
        raise PassSemanticGError("pass-semantic envelope deterministic double decode changed receipt")
    for field in ("semantic_labels", "pre_repair_camera_y1", "conditional_camera_y1"):
        if not np.array_equal(getattr(first, field), getattr(second, field)):
            raise PassSemanticGError(f"pass-semantic envelope deterministic replay changed {field}")
    if parse_pass_semantic_g_envelope_receipt(first.receipt.to_receipt_bytes()) != first.receipt:
        raise PassSemanticGError("pass-semantic envelope receipt failed strict parse/re-emit")
    return first


@dataclass(frozen=True, slots=True)
class CompiledPassSemanticGEnvelopeV1:
    envelope: bytes
    mode: PassSemanticGEnvelopeMode
    pass_packet: bytes
    pass_semantic: DecodedPassSemanticGV1
    compiled_repair: CompiledSameClassRealizationRepairV1 | None
    decoded: DecodedPassSemanticGEnvelopeV1

    def __post_init__(self) -> None:
        if type(self.envelope) is not bytes or not self.envelope.startswith(ENVELOPE_MAGIC):
            raise PassSemanticGError("compiled pass-semantic envelope changed exact domain")
        if type(self.mode) is not PassSemanticGEnvelopeMode:
            raise PassSemanticGError("compiled pass-semantic mode changed exact type")
        if type(self.pass_packet) is not bytes or not self.pass_packet.startswith(PASS_PACKET_MAGIC):
            raise PassSemanticGError("compiled pass-semantic inner packet changed exact domain")
        if type(self.pass_semantic) is not DecodedPassSemanticGV1:
            raise PassSemanticGError("compiled pass-semantic output changed exact type")
        if type(self.decoded) is not DecodedPassSemanticGEnvelopeV1:
            raise PassSemanticGError("compiled pass-semantic envelope decode changed exact type")
        if self.decoded.receipt.envelope_sha256 != _sha256(self.envelope):
            raise PassSemanticGError("compiled pass-semantic envelope differs from receipt")
        if self.mode is PassSemanticGEnvelopeMode.PASS_NO_G8_V1:
            if self.compiled_repair is not None:
                raise PassSemanticGError("compiled PASS_NO_G8 contains a fake repair")
        elif type(self.compiled_repair) is not CompiledSameClassRealizationRepairV1:
            raise PassSemanticGError("compiled PASS_THEN_G8 omitted exact repair")


def compile_pass_semantic_g_envelope(
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    repair_program: SameClassRealizationRepairProgramV1 | None = None,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1 | None = None,
    predictor_codec_id: str = "LVLS1.v1",
) -> CompiledPassSemanticGEnvelopeV1:
    pass_packet, pass_semantic = compile_pass_semantic_g_packet(
        predictor_surface=predictor_surface,
    )
    surface = build_pass_semantic_g8_surface(
        predictor_section_payload=predictor_section_payload,
        predictor_surface=predictor_surface,
        pass_semantic=pass_semantic,
        predictor_codec_id=predictor_codec_id,
    )
    compiled_repair: CompiledSameClassRealizationRepairV1 | None = None
    if repair_program is None:
        if target_custody is not None:
            raise PassSemanticGError("PASS_NO_G8 compile must not consume target custody")
        envelope = compose_pass_semantic_g_envelope(pass_packet)
        mode = PassSemanticGEnvelopeMode.PASS_NO_G8_V1
    else:
        if type(repair_program) is not SameClassRealizationRepairProgramV1:
            raise PassSemanticGError("PASS_THEN_G8 compile requires exact repair program")
        if type(target_custody) is not EncoderOnlyExactTargetLabelCustodyV1:
            raise PassSemanticGError("PASS_THEN_G8 compile requires exact encoder-only target custody")
        try:
            compiled_repair = compile_same_class_realization_repair(
                repair_program,
                surface=surface,
                target_custody=target_custody,
            )
        except SameClassRealizationRepairError as exc:
            raise PassSemanticGError("PASS_THEN_G8 repair compiler failed closed") from exc
        envelope = compose_pass_semantic_g_envelope(pass_packet, compiled_repair.packet)
        mode = PassSemanticGEnvelopeMode.PASS_THEN_G8_V1
    decoded = decode_pass_semantic_g_envelope(
        envelope,
        predictor_section_payload=predictor_section_payload,
        predictor_surface=predictor_surface,
        predictor_codec_id=predictor_codec_id,
    )
    return CompiledPassSemanticGEnvelopeV1(
        envelope=envelope,
        mode=mode,
        pass_packet=pass_packet,
        pass_semantic=pass_semantic,
        compiled_repair=compiled_repair,
        decoded=decoded,
    )


def parse_pass_semantic_g_envelope_receipt(payload: bytes) -> PassSemanticGEnvelopeReceiptV1:
    if type(payload) is not bytes or not payload:
        raise PassSemanticGError("pass-semantic envelope receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PassSemanticGError(f"pass-semantic envelope receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except PassSemanticGError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PassSemanticGError("pass-semantic envelope receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(PassSemanticGEnvelopeReceiptV1.__dataclass_fields__):
        raise PassSemanticGError("pass-semantic envelope receipt fields differ from closed schema")
    if _canonical_json(value) != payload:
        raise PassSemanticGError("pass-semantic envelope receipt is not canonical JSON")
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise PassSemanticGError("pass-semantic envelope receipt pair IDs must be exact integers")
    if type(value["pass_receipt"]) is not dict:
        raise PassSemanticGError("pass-semantic envelope nested pass receipt must be one object")
    try:
        pass_receipt = parse_pass_semantic_g_receipt(_canonical_json(value["pass_receipt"]))
        repair_receipt = None
        if value["repair_receipt"] is not None:
            if type(value["repair_receipt"]) is not dict:
                raise PassSemanticGError("pass-semantic envelope nested G8 receipt must be object/null")
            repair_receipt = parse_same_class_realization_repair_receipt(_canonical_json(value["repair_receipt"]))
    except SameClassRealizationRepairError as exc:
        raise PassSemanticGError("pass-semantic envelope nested G8 receipt is invalid") from exc
    arguments = dict(value)
    arguments["source_pair_ids"] = tuple(value["source_pair_ids"])
    arguments["pass_receipt"] = pass_receipt
    arguments["repair_receipt"] = repair_receipt
    try:
        receipt = PassSemanticGEnvelopeReceiptV1(**arguments)
    except TypeError as exc:
        raise PassSemanticGError("pass-semantic envelope receipt has invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise PassSemanticGError("pass-semantic envelope receipt changed on strict parse-back")
    return receipt


__all__ = [
    "ENVELOPE_HEADER",
    "ENVELOPE_MAGIC",
    "PASS_PACKET_HEADER",
    "PASS_PACKET_MAGIC",
    "CompiledPassSemanticGEnvelopeV1",
    "DecodedPassSemanticGEnvelopeV1",
    "DecodedPassSemanticGV1",
    "ParsedPassSemanticGEnvelopeV1",
    "PassSemanticGEnvelopeMode",
    "PassSemanticGEnvelopeReceiptV1",
    "PassSemanticGError",
    "PassSemanticGReceiptV1",
    "build_pass_semantic_g8_surface",
    "compile_pass_semantic_g_envelope",
    "compile_pass_semantic_g_packet",
    "compose_pass_semantic_g_envelope",
    "decode_pass_semantic_g_envelope",
    "decode_pass_semantic_g_packet",
    "parse_pass_semantic_g_envelope",
    "parse_pass_semantic_g_envelope_receipt",
    "parse_pass_semantic_g_receipt",
]
