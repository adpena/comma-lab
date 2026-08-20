# SPDX-License-Identifier: MIT
"""Strict conditional-A packet for a counted PASS semantic-G envelope.

``TACAPG1`` is the production A coordinate for ``TACPG81``.  It consumes P0
and the Y1 emitted by the exact pass envelope after its explicit optional-G8
mode.  A no-G8 control is therefore a real source-bound packet, not an empty G,
a fabricated legacy packet, or a no-op repair represented as G8.

No scorer/evaluator is loaded and no score, candidate, or promotion claim is
made.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import numpy as np

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    COPY_ROW,
    MAX_BOUNDED_PAIRS,
    RGB_ROW,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    CorrectedY1SupportCopyCellV1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Error,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    SparseConstantRGBCellV1,
    apply_disjoint_camera_cell_reconstruction_v1,
)
from tac.witness_dsl.taskspace_pass_semantic_g import (
    DecodedPassSemanticGEnvelopeV1,
    PassSemanticGEnvelopeMode,
    PassSemanticGError,
    parse_pass_semantic_g_envelope_receipt,
)

PACKET_MAGIC: Final = b"TACAPG1\x00"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct(">8sBBHH32sIIII")
SOURCE_SCHEMA: Final = "tac.taskspace_pass_conditional_a_source.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_pass_conditional_a_receipt.v1"
_MODE_TO_WIRE: Final = {
    PredictorPreservingA3Mode.PASS_P0_V1: 0,
    PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: 1,
    PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: 2,
}
_WIRE_TO_MODE: Final = {wire: mode for mode, wire in _MODE_TO_WIRE.items()}


class PassConditionalAError(ValueError):
    """The PASS-G conditional-A packet or one of its foreign keys failed."""


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
        raise PassConditionalAError("PASS-G conditional-A record is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PassConditionalAError(f"{field} must be canonical lowercase SHA-256")
    return value


def _pair_ids(value: object, *, field: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(type(pair_id) is not int for pair_id in value):
        raise PassConditionalAError(f"{field} must be one nonempty exact integer tuple")
    pair_ids = value
    expected = tuple(range(pair_ids[0], pair_ids[0] + len(pair_ids)))
    if pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise PassConditionalAError(f"{field} must be contiguous inside [0,600)")
    if len(pair_ids) > MAX_BOUNDED_PAIRS:
        raise PassConditionalAError(f"{field} exceeds the bounded pair ABI")
    return pair_ids


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class PassConditionalASurfaceV1:
    predictor_surface: PredictorCameraPairSurfaceV1
    pass_g: DecodedPassSemanticGEnvelopeV1

    def __post_init__(self) -> None:
        if type(self.predictor_surface) is not PredictorCameraPairSurfaceV1:
            raise PassConditionalAError("conditional A requires exact predictor surface")
        if type(self.pass_g) is not DecodedPassSemanticGEnvelopeV1:
            raise PassConditionalAError("conditional A requires exact PASS-G envelope decode")
        predictor = self.predictor_surface
        state = predictor.predictor_state
        receipt = self.pass_g.receipt
        if predictor.upstream_decode_receipt_sha256 is None:
            raise PassConditionalAError("conditional A requires strict P decode receipt custody")
        try:
            strict = parse_pass_semantic_g_envelope_receipt(receipt.to_receipt_bytes())
        except PassSemanticGError as exc:
            raise PassConditionalAError("conditional A could not close strict PASS-G receipt") from exc
        if strict != receipt:
            raise PassConditionalAError("conditional A PASS-G receipt changed on strict parse/re-emit")
        pass_receipt = receipt.pass_receipt
        if (
            receipt.source_pair_ids != state.source_pair_ids
            or pass_receipt.predictor_state_binding_sha256 != state.binding_sha256
            or pass_receipt.predictor_semantic_binding_sha256 != state.semantic_binding_sha256
            or pass_receipt.predictor_program_sha256 != state.predictor_program_sha256
            or pass_receipt.predictor_renderer_sha256 != state.predictor_renderer_sha256
            or pass_receipt.predictor_surface_binding_sha256 != predictor.binding_sha256
            or pass_receipt.upstream_decode_receipt_sha256 != predictor.upstream_decode_receipt_sha256
            or pass_receipt.predictor_camera_p1_sha256 != predictor.camera_p1_sha256
            or receipt.semantic_labels_sha256 != state.labels_sha256
            or receipt.output_camera_y1_sha256 != _array_sha256(self.pass_g.conditional_camera_y1)
        ):
            raise PassConditionalAError("conditional A live P/PASS-G surfaces differ from exact custody")

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return self.predictor_surface.predictor_state.source_pair_ids

    @property
    def predictor_camera_p0(self) -> np.ndarray:
        return self.predictor_surface.camera_p0

    @property
    def conditional_camera_y1(self) -> np.ndarray:
        return self.pass_g.conditional_camera_y1


def build_pass_conditional_a_surface(
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_g: DecodedPassSemanticGEnvelopeV1,
) -> PassConditionalASurfaceV1:
    return PassConditionalASurfaceV1(predictor_surface=predictor_surface, pass_g=pass_g)


@dataclass(frozen=True, slots=True)
class PassConditionalASourceBindingV1:
    schema: Literal["tac.taskspace_pass_conditional_a_source.v1"]
    source_pair_ids: tuple[int, ...]
    pass_g_mode: str
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str
    predictor_labels_sha256: str
    predictor_camera_p0_sha256: str
    predictor_camera_p1_sha256: str
    pass_g_envelope_sha256: str
    pass_g_receipt_sha256: str
    pass_semantic_packet_sha256: str
    pass_semantic_receipt_sha256: str
    semantic_labels_sha256: str
    pre_repair_camera_y1_sha256: str
    repair_packet_sha256: str | None
    repair_receipt_sha256: str | None
    conditional_camera_y1_sha256: str

    def __post_init__(self) -> None:
        if self.schema != SOURCE_SCHEMA:
            raise PassConditionalAError("conditional-A source schema changed")
        _pair_ids(self.source_pair_ids, field="source_pair_ids")
        try:
            mode = PassSemanticGEnvelopeMode(self.pass_g_mode)
        except ValueError as exc:
            raise PassConditionalAError("conditional-A PASS-G mode changed") from exc
        for field in (
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "predictor_surface_binding_sha256",
            "upstream_decode_receipt_sha256",
            "predictor_labels_sha256",
            "predictor_camera_p0_sha256",
            "predictor_camera_p1_sha256",
            "pass_g_envelope_sha256",
            "pass_g_receipt_sha256",
            "pass_semantic_packet_sha256",
            "pass_semantic_receipt_sha256",
            "semantic_labels_sha256",
            "pre_repair_camera_y1_sha256",
            "conditional_camera_y1_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if mode is PassSemanticGEnvelopeMode.PASS_NO_G8_V1:
            if (
                self.repair_packet_sha256 is not None
                or self.repair_receipt_sha256 is not None
                or self.conditional_camera_y1_sha256 != self.pre_repair_camera_y1_sha256
            ):
                raise PassConditionalAError("PASS_NO_G8 conditional-A source forged a repair or changed Y1")
        else:
            if self.repair_packet_sha256 is None or self.repair_receipt_sha256 is None:
                raise PassConditionalAError("PASS_THEN_G8 conditional-A source omitted repair custody")
            _require_sha256(self.repair_packet_sha256, "repair_packet_sha256")
            _require_sha256(self.repair_receipt_sha256, "repair_receipt_sha256")
            if self.conditional_camera_y1_sha256 == self.pre_repair_camera_y1_sha256:
                raise PassConditionalAError("PASS_THEN_G8 conditional-A source contains a no-op repair")

    @property
    def pair_start(self) -> int:
        return self.source_pair_ids[0]

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    @property
    def binding_sha256(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


def derive_pass_conditional_a_source_binding(
    surface: PassConditionalASurfaceV1,
) -> PassConditionalASourceBindingV1:
    if type(surface) is not PassConditionalASurfaceV1:
        raise PassConditionalAError("conditional-A source derivation requires exact surface")
    predictor = surface.predictor_surface
    state = predictor.predictor_state
    receipt = surface.pass_g.receipt
    return PassConditionalASourceBindingV1(
        schema=SOURCE_SCHEMA,
        source_pair_ids=surface.source_pair_ids,
        pass_g_mode=receipt.mode,
        predictor_state_binding_sha256=state.binding_sha256,
        predictor_semantic_binding_sha256=state.semantic_binding_sha256,
        predictor_program_sha256=state.predictor_program_sha256,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        predictor_surface_binding_sha256=predictor.binding_sha256,
        upstream_decode_receipt_sha256=predictor.upstream_decode_receipt_sha256,  # type: ignore[arg-type]
        predictor_labels_sha256=state.labels_sha256,
        predictor_camera_p0_sha256=predictor.camera_p0_sha256,
        predictor_camera_p1_sha256=predictor.camera_p1_sha256,
        pass_g_envelope_sha256=receipt.envelope_sha256,
        pass_g_receipt_sha256=receipt.receipt_sha256,
        pass_semantic_packet_sha256=receipt.pass_packet_sha256,
        pass_semantic_receipt_sha256=receipt.pass_receipt_sha256,
        semantic_labels_sha256=receipt.semantic_labels_sha256,
        pre_repair_camera_y1_sha256=receipt.pre_repair_camera_y1_sha256,
        repair_packet_sha256=receipt.repair_packet_sha256,
        repair_receipt_sha256=receipt.repair_receipt_sha256,
        conditional_camera_y1_sha256=receipt.output_camera_y1_sha256,
    )


@dataclass(frozen=True, slots=True)
class ParsedPassConditionalAPacketV1:
    packet: bytes
    source_binding_sha256: str
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    coordinate_bytes: int
    rgb_value_bytes: int
    program: PredictorPreservingA3ProgramV1


def _encode_packet(program: PredictorPreservingA3ProgramV1, source: PassConditionalASourceBindingV1) -> bytes:
    if type(program) is not PredictorPreservingA3ProgramV1:
        raise PassConditionalAError("conditional-A compile requires exact A3 program")
    if type(source) is not PassConditionalASourceBindingV1:
        raise PassConditionalAError("conditional-A packet requires exact source binding")
    addressed = tuple(row.source_pair_id for row in program.constant_rgb_cells) + tuple(
        row.source_pair_id for row in program.corrected_y1_copy_cells
    )
    if any(pair_id not in source.source_pair_ids for pair_id in addressed):
        raise PassConditionalAError("conditional-A row addresses outside the source window")
    body = bytearray()
    if program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
        for row in program.constant_rgb_cells:
            body.extend(RGB_ROW.pack(row.source_pair_id, row.scorer_row, row.scorer_col, *row.rgb_u8))
    elif program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1:
        for row in program.corrected_y1_copy_cells:
            body.extend(COPY_ROW.pack(row.source_pair_id, row.scorer_row, row.scorer_col))
    if len(body) != program.body_bytes:
        raise PassConditionalAError("conditional-A body accounting changed")
    packet_bytes = PACKET_HEADER.size + len(body)
    fixed = (
        PACKET_MAGIC,
        PACKET_VERSION,
        _MODE_TO_WIRE[program.mode],
        source.pair_start,
        source.pair_count,
        bytes.fromhex(source.binding_sha256),
        program.control_row_count,
        len(body),
        packet_bytes,
    )
    zero = PACKET_HEADER.pack(*fixed, 0)
    crc = zlib.crc32(zero + body) & 0xFFFFFFFF
    return PACKET_HEADER.pack(*fixed, crc) + bytes(body)


def parse_pass_conditional_a_packet(
    packet: bytes,
    *,
    expected_source_binding: PassConditionalASourceBindingV1,
) -> ParsedPassConditionalAPacketV1:
    if type(packet) is not bytes or len(packet) < PACKET_HEADER.size:
        raise PassConditionalAError("conditional-A packet is shorter than its exact header")
    if type(expected_source_binding) is not PassConditionalASourceBindingV1:
        raise PassConditionalAError("conditional-A parse requires exact source binding")
    try:
        magic, version, mode_wire, pair_start, pair_count, source_raw, row_count, body_bytes, packet_bytes, crc = (
            PACKET_HEADER.unpack_from(packet)
        )
    except struct.error as exc:  # pragma: no cover
        raise PassConditionalAError("conditional-A header is malformed") from exc
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise PassConditionalAError("conditional-A packet magic/version changed")
    try:
        mode = _WIRE_TO_MODE[mode_wire]
    except KeyError as exc:
        raise PassConditionalAError("conditional-A mode is outside the closed universe") from exc
    source = expected_source_binding
    if pair_start != source.pair_start or pair_count != source.pair_count:
        raise PassConditionalAError("conditional-A pair window differs from live source")
    if source_raw.hex() != source.binding_sha256:
        raise PassConditionalAError("conditional-A source binding differs from live P/PASS-G/Y1")
    if packet_bytes != len(packet) or body_bytes != len(packet) - PACKET_HEADER.size:
        raise PassConditionalAError("conditional-A length accounting or trailing bytes changed")
    row_size = {
        PredictorPreservingA3Mode.PASS_P0_V1: 0,
        PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: RGB_ROW.size,
        PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: COPY_ROW.size,
    }[mode]
    if mode is PredictorPreservingA3Mode.PASS_P0_V1:
        if row_count != 0 or body_bytes != 0:
            raise PassConditionalAError("conditional PASS_P0 must have zero rows/body")
    elif row_count < 1 or body_bytes != row_count * row_size:
        raise PassConditionalAError("conditional-A row/body count differs from mode ABI")
    body = packet[PACKET_HEADER.size :]
    zero = PACKET_HEADER.pack(
        magic,
        version,
        mode_wire,
        pair_start,
        pair_count,
        source_raw,
        row_count,
        body_bytes,
        packet_bytes,
        0,
    )
    if zlib.crc32(zero + body) & 0xFFFFFFFF != crc:
        raise PassConditionalAError("conditional-A CRC differs from exact bytes")
    rgb_rows: list[SparseConstantRGBCellV1] = []
    copy_rows: list[CorrectedY1SupportCopyCellV1] = []
    offset = 0
    try:
        if mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
            for _ in range(row_count):
                pair_id, row, col, red, green, blue = RGB_ROW.unpack_from(body, offset)
                offset += RGB_ROW.size
                rgb_rows.append(SparseConstantRGBCellV1(pair_id, row, col, (red, green, blue)))
        elif mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1:
            for _ in range(row_count):
                pair_id, row, col = COPY_ROW.unpack_from(body, offset)
                offset += COPY_ROW.size
                copy_rows.append(CorrectedY1SupportCopyCellV1(pair_id, row, col))
    except (struct.error, PredictorPreservingA3Error) as exc:
        raise PassConditionalAError("conditional-A control row is malformed") from exc
    if offset != len(body):
        raise PassConditionalAError("conditional-A body was not consumed exactly")
    try:
        program = PredictorPreservingA3ProgramV1(
            mode,
            constant_rgb_cells=tuple(rgb_rows),
            corrected_y1_copy_cells=tuple(copy_rows),
        )
    except PredictorPreservingA3Error as exc:
        raise PassConditionalAError("conditional-A program is noncanonical") from exc
    addressed = tuple(row.source_pair_id for row in program.constant_rgb_cells) + tuple(
        row.source_pair_id for row in program.corrected_y1_copy_cells
    )
    if any(pair_id not in source.source_pair_ids for pair_id in addressed):
        raise PassConditionalAError("conditional-A row addresses outside the source window")
    if _encode_packet(program, source) != packet:
        raise PassConditionalAError("conditional-A packet changed on strict parse/re-encode")
    return ParsedPassConditionalAPacketV1(
        packet=packet,
        source_binding_sha256=source.binding_sha256,
        packet_bytes=packet_bytes,
        packet_header_bytes=PACKET_HEADER.size,
        packet_body_bytes=body_bytes,
        coordinate_bytes=program.coordinate_bytes,
        rgb_value_bytes=program.rgb_value_bytes,
        program=program,
    )


@dataclass(frozen=True, slots=True)
class PassConditionalAReceiptV1:
    schema: Literal["tac.taskspace_pass_conditional_a_receipt.v1"]
    mode: str
    source: PassConditionalASourceBindingV1
    source_binding_sha256: str
    packet_sha256: str
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    serialized_coordinate_bytes: int
    serialized_rgb_value_bytes: int
    control_row_count: int
    output_camera_y0_sha256: str
    output_camera_y1_sha256: str
    chronological_camera_frames_sha256: str
    owned_scorer_mask_sha256: str
    owned_camera_mask_sha256: str
    predictor_p0_scorer_numerators_sha256: str
    output_y0_scorer_numerators_sha256: str
    owned_scorer_cells: int
    owned_camera_values: int
    actually_changed_camera_values: int
    scorer_denominator: int
    p0_pass_through_zero_mutation: bool
    constant_rgb_target_numerators_exact: bool
    conditional_y1_support_numerators_exact: bool
    counted_pass_semantic_g_nonempty: Literal[True] = True
    optional_g8_mode_explicit: Literal[True] = True
    conditional_y1_unchanged_by_a: Literal[True] = True
    canonical_disjoint_reconstruction_reused: Literal[True] = True
    outside_a_owned_p0_camera_values_preserved: Literal[True] = True
    outside_a_owned_p0_scorer_numerators_preserved: Literal[True] = True
    packet_parse_reencode_exact: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    dense_conditional_y1_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    promotion_eligible: Literal[False] = False
    through_r_target_realization_debt_closed: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise PassConditionalAError("conditional-A receipt schema changed")
        if type(self.source) is not PassConditionalASourceBindingV1:
            raise PassConditionalAError("conditional-A receipt requires exact source")
        if self.source.binding_sha256 != self.source_binding_sha256:
            raise PassConditionalAError("conditional-A receipt source binding does not recompose")
        try:
            mode = PredictorPreservingA3Mode(self.mode)
        except ValueError as exc:
            raise PassConditionalAError("conditional-A receipt mode changed") from exc
        for field in (
            "source_binding_sha256",
            "packet_sha256",
            "output_camera_y0_sha256",
            "output_camera_y1_sha256",
            "chronological_camera_frames_sha256",
            "owned_scorer_mask_sha256",
            "owned_camera_mask_sha256",
            "predictor_p0_scorer_numerators_sha256",
            "output_y0_scorer_numerators_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        counts = (
            "packet_bytes",
            "packet_header_bytes",
            "packet_body_bytes",
            "serialized_coordinate_bytes",
            "serialized_rgb_value_bytes",
            "control_row_count",
            "owned_scorer_cells",
            "owned_camera_values",
            "actually_changed_camera_values",
            "scorer_denominator",
        )
        if any(type(getattr(self, field)) is not int for field in counts):
            raise PassConditionalAError("conditional-A receipt counts must be exact integers")
        if (
            self.packet_header_bytes != PACKET_HEADER.size
            or self.packet_bytes != self.packet_header_bytes + self.packet_body_bytes
            or self.packet_body_bytes != self.serialized_coordinate_bytes + self.serialized_rgb_value_bytes
            or self.control_row_count < 0
            or self.owned_scorer_cells < 0
            or self.owned_camera_values < 0
            or self.actually_changed_camera_values < 0
            or self.scorer_denominator <= 0
            or self.output_camera_y1_sha256 != self.source.conditional_camera_y1_sha256
        ):
            raise PassConditionalAError("conditional-A receipt accounting or Y1 custody changed")
        expected_mode_truth = {
            PredictorPreservingA3Mode.PASS_P0_V1: (True, False, False),
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: (False, True, False),
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: (False, False, True),
        }[mode]
        actual_mode_truth = (
            self.p0_pass_through_zero_mutation,
            self.constant_rgb_target_numerators_exact,
            self.conditional_y1_support_numerators_exact,
        )
        if any(type(value) is not bool for value in actual_mode_truth) or actual_mode_truth != expected_mode_truth:
            raise PassConditionalAError("conditional-A receipt mode-specific proof flags disagree")
        if mode is PredictorPreservingA3Mode.PASS_P0_V1:
            if self.control_row_count != 0 or self.actually_changed_camera_values != 0:
                raise PassConditionalAError("conditional PASS_P0 receipt contains a mutation")
        elif self.control_row_count < 1 or self.owned_scorer_cells != self.control_row_count:
            raise PassConditionalAError("conditional-A receipt ownership differs from rows")
        truth = {
            "counted_pass_semantic_g_nonempty": True,
            "optional_g8_mode_explicit": True,
            "conditional_y1_unchanged_by_a": True,
            "canonical_disjoint_reconstruction_reused": True,
            "outside_a_owned_p0_camera_values_preserved": True,
            "outside_a_owned_p0_scorer_numerators_preserved": True,
            "packet_parse_reencode_exact": True,
            "deterministic_double_decode": True,
            "dense_conditional_y1_serialized": False,
            "scorer_invoked": False,
            "n600_evidence_claim": False,
            "exact_score_claim": False,
            "candidate_archive_eligible": False,
            "promotion_eligible": False,
            "through_r_target_realization_debt_closed": False,
            "research_only": True,
        }
        if any(
            type(getattr(self, field)) is not bool or getattr(self, field) != expected
            for field, expected in truth.items()
        ):
            raise PassConditionalAError("conditional-A receipt truth labels became permissive")

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return self.source.source_pair_ids

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = self.source.as_dict()
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class DecodedPassConditionalAV1:
    camera_y0: np.ndarray
    camera_y1: np.ndarray
    chronological_camera_frames: np.ndarray
    owned_scorer_mask: np.ndarray
    owned_camera_mask: np.ndarray
    receipt: PassConditionalAReceiptV1

    def __post_init__(self) -> None:
        if type(self.receipt) is not PassConditionalAReceiptV1:
            raise PassConditionalAError("decoded conditional A requires exact receipt")
        pair_count = len(self.receipt.source_pair_ids)
        y0 = np.asarray(self.camera_y0)
        y1 = np.asarray(self.camera_y1)
        chronology = np.asarray(self.chronological_camera_frames)
        scorer_mask = np.asarray(self.owned_scorer_mask)
        camera_mask = np.asarray(self.owned_camera_mask)
        if y0.dtype != np.uint8 or y0.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
            raise PassConditionalAError("decoded conditional-A Y0 ABI changed")
        if y1.dtype != np.uint8 or y1.shape != y0.shape:
            raise PassConditionalAError("decoded conditional-A Y1 ABI changed")
        if chronology.dtype != np.uint8 or chronology.shape != (pair_count, 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
            raise PassConditionalAError("decoded conditional-A chronology ABI changed")
        if scorer_mask.dtype != np.bool_ or scorer_mask.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH):
            raise PassConditionalAError("decoded conditional-A scorer-mask ABI changed")
        if camera_mask.dtype != np.bool_ or camera_mask.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH):
            raise PassConditionalAError("decoded conditional-A camera-mask ABI changed")
        expected = {
            "output_camera_y0_sha256": _array_sha256(y0),
            "output_camera_y1_sha256": _array_sha256(y1),
            "chronological_camera_frames_sha256": _array_sha256(chronology),
            "owned_scorer_mask_sha256": _array_sha256(scorer_mask),
            "owned_camera_mask_sha256": _array_sha256(camera_mask),
        }
        if any(getattr(self.receipt, field) != digest for field, digest in expected.items()):
            raise PassConditionalAError("decoded conditional-A arrays differ from receipt")
        if not np.array_equal(chronology[:, 0], y0) or not np.array_equal(chronology[:, 1], y1):
            raise PassConditionalAError("decoded conditional-A chronology is not [Y0,Y1]")
        object.__setattr__(self, "camera_y0", _immutable(y0, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "camera_y1", _immutable(y1, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "chronological_camera_frames", _immutable(chronology, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "owned_scorer_mask", _immutable(scorer_mask, dtype=np.dtype(np.bool_)))
        object.__setattr__(self, "owned_camera_mask", _immutable(camera_mask, dtype=np.dtype(np.bool_)))


def _decode_once(
    parsed: ParsedPassConditionalAPacketV1,
    source: PassConditionalASourceBindingV1,
    surface: PassConditionalASurfaceV1,
) -> DecodedPassConditionalAV1:
    program = parsed.program
    y1 = surface.conditional_camera_y1
    try:
        reconstruction = apply_disjoint_camera_cell_reconstruction_v1(
            surface.predictor_camera_p0,
            source_pair_ids=source.source_pair_ids,
            constant_rgb_cells=program.constant_rgb_cells,
            copy_source_camera=(y1 if program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1 else None),
            copy_source_cells=program.corrected_y1_copy_cells,
        )
    except PredictorPreservingA3Error as exc:
        raise PassConditionalAError("conditional-A canonical reconstruction failed") from exc
    y0 = reconstruction.output_camera
    if program.mode is PredictorPreservingA3Mode.PASS_P0_V1:
        if not np.array_equal(y0, surface.predictor_camera_p0) or np.any(reconstruction.owned_scorer_mask):
            raise PassConditionalAError("conditional PASS_P0 failed exact zero mutation")
    elif program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
        if not reconstruction.constant_rgb_target_numerators_exact:
            raise PassConditionalAError("conditional sparse RGB lacked numerator proof")
    elif not reconstruction.copy_source_target_numerators_exact:
        raise PassConditionalAError("conditional Y1 copy lacked numerator proof")
    if _array_sha256(y1) != source.conditional_camera_y1_sha256:
        raise PassConditionalAError("conditional Y1 changed after source binding")
    chronology = np.stack((y0, y1), axis=1)
    receipt = PassConditionalAReceiptV1(
        schema=RECEIPT_SCHEMA,
        mode=program.mode.value,
        source=source,
        source_binding_sha256=source.binding_sha256,
        packet_sha256=_sha256(parsed.packet),
        packet_bytes=parsed.packet_bytes,
        packet_header_bytes=parsed.packet_header_bytes,
        packet_body_bytes=parsed.packet_body_bytes,
        serialized_coordinate_bytes=parsed.coordinate_bytes,
        serialized_rgb_value_bytes=parsed.rgb_value_bytes,
        control_row_count=program.control_row_count,
        output_camera_y0_sha256=_array_sha256(y0),
        output_camera_y1_sha256=_array_sha256(y1),
        chronological_camera_frames_sha256=_array_sha256(chronology),
        owned_scorer_mask_sha256=_array_sha256(reconstruction.owned_scorer_mask),
        owned_camera_mask_sha256=_array_sha256(reconstruction.owned_camera_mask),
        predictor_p0_scorer_numerators_sha256=_array_sha256(reconstruction.base_scorer_numerators),
        output_y0_scorer_numerators_sha256=_array_sha256(reconstruction.output_scorer_numerators),
        owned_scorer_cells=int(np.count_nonzero(reconstruction.owned_scorer_mask)),
        owned_camera_values=int(np.count_nonzero(reconstruction.owned_camera_mask)) * CHANNELS,
        actually_changed_camera_values=reconstruction.actually_changed_camera_values,
        scorer_denominator=reconstruction.scorer_denominator,
        p0_pass_through_zero_mutation=program.mode is PredictorPreservingA3Mode.PASS_P0_V1,
        constant_rgb_target_numerators_exact=program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
        conditional_y1_support_numerators_exact=(
            program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1
        ),
    )
    return DecodedPassConditionalAV1(
        camera_y0=y0,
        camera_y1=y1,
        chronological_camera_frames=chronology,
        owned_scorer_mask=reconstruction.owned_scorer_mask,
        owned_camera_mask=reconstruction.owned_camera_mask,
        receipt=receipt,
    )


def decode_pass_conditional_a_packet(
    packet: bytes,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_g: DecodedPassSemanticGEnvelopeV1,
) -> DecodedPassConditionalAV1:
    surface = build_pass_conditional_a_surface(predictor_surface=predictor_surface, pass_g=pass_g)
    source = derive_pass_conditional_a_source_binding(surface)
    parsed = parse_pass_conditional_a_packet(packet, expected_source_binding=source)
    first = _decode_once(parsed, source, surface)
    second = _decode_once(parsed, source, surface)
    if first.receipt != second.receipt:
        raise PassConditionalAError("conditional-A double decode changed receipt")
    for field in ("camera_y0", "camera_y1", "chronological_camera_frames", "owned_scorer_mask", "owned_camera_mask"):
        if not np.array_equal(getattr(first, field), getattr(second, field)):
            raise PassConditionalAError(f"conditional-A double decode changed {field}")
    if parse_pass_conditional_a_receipt(first.receipt.to_receipt_bytes()) != first.receipt:
        raise PassConditionalAError("conditional-A receipt failed strict parse/re-emit")
    return first


def parse_pass_conditional_a_receipt(payload: bytes) -> PassConditionalAReceiptV1:
    if type(payload) is not bytes or not payload:
        raise PassConditionalAError("conditional-A receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PassConditionalAError(f"conditional-A receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PassConditionalAError(f"conditional-A receipt contains non-finite constant {token}")
            ),
        )
    except PassConditionalAError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PassConditionalAError("conditional-A receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(PassConditionalAReceiptV1.__dataclass_fields__):
        raise PassConditionalAError("conditional-A receipt fields differ from closed schema")
    if _canonical_json(value) != payload:
        raise PassConditionalAError("conditional-A receipt is not canonical JSON")
    if type(value["source"]) is not dict or set(value["source"]) != set(
        PassConditionalASourceBindingV1.__dataclass_fields__
    ):
        raise PassConditionalAError("conditional-A nested source fields changed")
    source_value = dict(value["source"])
    if type(source_value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in source_value["source_pair_ids"]
    ):
        raise PassConditionalAError("conditional-A source pair IDs must be exact integers")
    source_value["source_pair_ids"] = tuple(source_value["source_pair_ids"])
    try:
        source = PassConditionalASourceBindingV1(**source_value)
    except TypeError as exc:
        raise PassConditionalAError("conditional-A nested source has invalid typed fields") from exc
    arguments = dict(value)
    arguments["source"] = source
    try:
        receipt = PassConditionalAReceiptV1(**arguments)
    except TypeError as exc:
        raise PassConditionalAError("conditional-A receipt has invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise PassConditionalAError("conditional-A receipt changed on strict parse-back")
    return receipt


@dataclass(frozen=True, slots=True)
class CompiledPassConditionalAV1:
    packet: bytes
    program: PredictorPreservingA3ProgramV1
    source_binding: PassConditionalASourceBindingV1
    decoded: DecodedPassConditionalAV1

    def __post_init__(self) -> None:
        if type(self.packet) is not bytes or not self.packet.startswith(PACKET_MAGIC):
            raise PassConditionalAError("compiled conditional-A packet changed domain")
        if type(self.program) is not PredictorPreservingA3ProgramV1:
            raise PassConditionalAError("compiled conditional-A program changed type")
        if type(self.source_binding) is not PassConditionalASourceBindingV1:
            raise PassConditionalAError("compiled conditional-A source changed type")
        if type(self.decoded) is not DecodedPassConditionalAV1:
            raise PassConditionalAError("compiled conditional-A decode changed type")
        if (
            self.decoded.receipt.packet_sha256 != _sha256(self.packet)
            or self.decoded.receipt.source_binding_sha256 != self.source_binding.binding_sha256
        ):
            raise PassConditionalAError("compiled conditional-A packet/source differs from receipt")


def compile_pass_conditional_a(
    program: PredictorPreservingA3ProgramV1,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    pass_g: DecodedPassSemanticGEnvelopeV1,
) -> CompiledPassConditionalAV1:
    if type(program) is not PredictorPreservingA3ProgramV1:
        raise PassConditionalAError("conditional-A compile requires exact A3 program")
    surface = build_pass_conditional_a_surface(predictor_surface=predictor_surface, pass_g=pass_g)
    source = derive_pass_conditional_a_source_binding(surface)
    packet = _encode_packet(program, source)
    decoded = decode_pass_conditional_a_packet(
        packet,
        predictor_surface=predictor_surface,
        pass_g=pass_g,
    )
    return CompiledPassConditionalAV1(packet, program, source, decoded)


__all__ = [
    "PACKET_HEADER",
    "PACKET_MAGIC",
    "CompiledPassConditionalAV1",
    "DecodedPassConditionalAV1",
    "ParsedPassConditionalAPacketV1",
    "PassConditionalAError",
    "PassConditionalAReceiptV1",
    "PassConditionalASourceBindingV1",
    "PassConditionalASurfaceV1",
    "build_pass_conditional_a_surface",
    "compile_pass_conditional_a",
    "decode_pass_conditional_a_packet",
    "derive_pass_conditional_a_source_binding",
    "parse_pass_conditional_a_packet",
    "parse_pass_conditional_a_receipt",
]
