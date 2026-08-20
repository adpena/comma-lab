# SPDX-License-Identifier: MIT
"""Strict counted conditional-A packet bound to the post-G8 corrected Y1.

The legacy ``TACA3P1`` packet is foreign-keyed to the semantic-G overlay.  A
``TACG8S1`` section changes that camera surface after the overlay, so reusing a
legacy packet (or forging its overlay receipt) would erase the actual causal
order.  This module gives the post-G8 surface its own packet domain while
reusing the canonical disjoint camera-cell reconstruction exactly.

No scorer is loaded here and no score, target-closure, candidate, originality,
or promotion claim is made.
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
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    DecodedSameClassRealizationRepairPGAIntegrationV1,
    SameClassRealizationRepairError,
    parse_same_class_realization_repair_receipt,
)

PACKET_MAGIC: Final = b"TACA8P1\x00"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct(">8sBBHH32sIIII")
SOURCE_SCHEMA: Final = "tac.taskspace_post_g8_conditional_a_source.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_post_g8_conditional_a_receipt.v1"
MAX_BOUNDED_PAIRS: Final = 4

_MODE_TO_WIRE: Final = {
    PredictorPreservingA3Mode.PASS_P0_V1: 0,
    PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: 1,
    PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: 2,
}
_WIRE_TO_MODE: Final = {wire: mode for mode, wire in _MODE_TO_WIRE.items()}


class PostG8ConditionalAError(ValueError):
    """The post-G8 A packet, foreign keys, or reconstruction failed closed."""


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
        raise PostG8ConditionalAError("post-G8 A record is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PostG8ConditionalAError(f"{field} must be canonical lowercase SHA-256")
    return value


def _validate_pair_ids(source_pair_ids: object, *, field: str) -> tuple[int, ...]:
    if type(source_pair_ids) is not tuple or not source_pair_ids:
        raise PostG8ConditionalAError(f"{field} must be a nonempty exact tuple")
    if any(type(value) is not int for value in source_pair_ids):
        raise PostG8ConditionalAError(f"{field} must contain exact integers")
    pair_ids = source_pair_ids
    expected = tuple(range(pair_ids[0], pair_ids[0] + len(pair_ids)))
    if pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise PostG8ConditionalAError(f"{field} must be contiguous inside [0,600)")
    if len(pair_ids) > MAX_BOUNDED_PAIRS:
        raise PostG8ConditionalAError(f"{field} exceeds the bounded pair ABI")
    return pair_ids


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class PostG8ConditionalASurfaceV1:
    """Typed ephemeral P0/post-G8-Y1 surface; deliberately not serializable."""

    predictor_surface: PredictorCameraPairSurfaceV1
    g8_integration: DecodedSameClassRealizationRepairPGAIntegrationV1

    def __post_init__(self) -> None:
        if type(self.predictor_surface) is not PredictorCameraPairSurfaceV1:
            raise PostG8ConditionalAError("post-G8 A requires exact PredictorCameraPairSurfaceV1")
        if type(self.g8_integration) is not DecodedSameClassRealizationRepairPGAIntegrationV1:
            raise PostG8ConditionalAError("post-G8 A requires exact DecodedSameClassRealizationRepairPGAIntegrationV1")
        predictor = self.predictor_surface
        integration = self.g8_integration
        state = predictor.predictor_state
        repair_surface = integration.surface
        if predictor.upstream_decode_receipt_sha256 is None:
            raise PostG8ConditionalAError("post-G8 A requires strict upstream P decode receipt custody")
        if state.source_pair_ids != repair_surface.source_pair_ids:
            raise PostG8ConditionalAError("post-G8 A P and G8 pair windows differ")
        if (
            repair_surface.p_section_identity.payload_sha256 != state.predictor_program_sha256
            or repair_surface.p_section_identity.payload_bytes != len(state.predictor_program)
        ):
            raise PostG8ConditionalAError("post-G8 A G8 surface differs from exact predictor P bytes")
        parsed = integration.parsed_composite_g_section
        if (
            repair_surface.g_section_identity.payload_sha256 != parsed.semantic_g_packet_sha256
            or repair_surface.g_section_identity.payload_bytes != len(parsed.semantic_g_packet)
        ):
            raise PostG8ConditionalAError("post-G8 A G8 surface differs from exact inner semantic G")
        repair_receipt = integration.decoded_repair.receipt
        try:
            strict_receipt = parse_same_class_realization_repair_receipt(repair_receipt.to_receipt_bytes())
        except SameClassRealizationRepairError as exc:
            raise PostG8ConditionalAError("post-G8 A could not close the strict G8 receipt") from exc
        if strict_receipt != repair_receipt:
            raise PostG8ConditionalAError("post-G8 A G8 receipt changed on strict parse/re-emit")
        if (
            repair_receipt.packet_sha256 != parsed.repair_packet_sha256
            or repair_receipt.output_camera_y1_sha256 != _array_sha256(integration.post_g8_corrected_y1)
            or repair_receipt.output_g_labels_sha256 != _array_sha256(integration.unchanged_g_semantic_labels)
            or not np.array_equal(
                integration.unchanged_g_semantic_labels,
                repair_surface.g_semantic_labels,
            )
        ):
            raise PostG8ConditionalAError("post-G8 A live G8 arrays differ from exact repair custody")

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return self.predictor_surface.predictor_state.source_pair_ids

    @property
    def predictor_camera_p0(self) -> np.ndarray:
        return self.predictor_surface.camera_p0

    @property
    def pre_g8_corrected_y1(self) -> np.ndarray:
        return self.g8_integration.surface.corrected_y1_camera

    @property
    def post_g8_corrected_y1(self) -> np.ndarray:
        return self.g8_integration.post_g8_corrected_y1


def build_post_g8_conditional_a_surface(
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    g8_integration: DecodedSameClassRealizationRepairPGAIntegrationV1,
) -> PostG8ConditionalASurfaceV1:
    """Construct the sole admitted live source type for post-G8 conditional A."""

    return PostG8ConditionalASurfaceV1(
        predictor_surface=predictor_surface,
        g8_integration=g8_integration,
    )


@dataclass(frozen=True, slots=True)
class PostG8ConditionalASourceBindingV1:
    """Every foreign key embedded in one ``TACA8P1`` packet."""

    schema: Literal["tac.taskspace_post_g8_conditional_a_source.v1"]
    source_pair_ids: tuple[int, ...]
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str
    predictor_labels_sha256: str
    predictor_camera_p0_sha256: str
    predictor_camera_p1_sha256: str
    predictor_section_identity_sha256: str
    composite_g_section_sha256: str
    semantic_g_packet_sha256: str
    semantic_g_labels_sha256: str
    repair_packet_sha256: str
    repair_receipt_sha256: str
    pre_g8_corrected_y1_sha256: str
    post_g8_corrected_y1_sha256: str

    def __post_init__(self) -> None:
        if self.schema != SOURCE_SCHEMA:
            raise PostG8ConditionalAError("post-G8 A source schema changed")
        _validate_pair_ids(self.source_pair_ids, field="source_pair_ids")
        for field in PostG8ConditionalASourceBindingV1.__dataclass_fields__:
            if field not in {"schema", "source_pair_ids"}:
                _require_sha256(getattr(self, field), field)
        if self.pre_g8_corrected_y1_sha256 == self.post_g8_corrected_y1_sha256:
            raise PostG8ConditionalAError("post-G8 A source requires a realized non-no-op G8 Y1")
        if self.semantic_g_packet_sha256 == self.repair_packet_sha256:
            raise PostG8ConditionalAError("post-G8 A inner semantic G and G8R1 identities aliased")

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


def derive_post_g8_conditional_a_source_binding(
    surface: PostG8ConditionalASurfaceV1,
) -> PostG8ConditionalASourceBindingV1:
    """Derive the packet binding from the live typed P/G8 surface only."""

    if type(surface) is not PostG8ConditionalASurfaceV1:
        raise PostG8ConditionalAError("post-G8 A source derivation requires exact surface type")
    predictor = surface.predictor_surface
    state = predictor.predictor_state
    integration = surface.g8_integration
    parsed = integration.parsed_composite_g_section
    repair = integration.decoded_repair.receipt
    return PostG8ConditionalASourceBindingV1(
        schema=SOURCE_SCHEMA,
        source_pair_ids=surface.source_pair_ids,
        predictor_state_binding_sha256=state.binding_sha256,
        predictor_semantic_binding_sha256=state.semantic_binding_sha256,
        predictor_program_sha256=state.predictor_program_sha256,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        predictor_surface_binding_sha256=predictor.binding_sha256,
        upstream_decode_receipt_sha256=predictor.upstream_decode_receipt_sha256,  # type: ignore[arg-type]
        predictor_labels_sha256=state.labels_sha256,
        predictor_camera_p0_sha256=predictor.camera_p0_sha256,
        predictor_camera_p1_sha256=predictor.camera_p1_sha256,
        predictor_section_identity_sha256=integration.surface.p_section_identity.identity_sha256,
        composite_g_section_sha256=parsed.section_sha256,
        semantic_g_packet_sha256=parsed.semantic_g_packet_sha256,
        semantic_g_labels_sha256=integration.surface.g_labels_sha256,
        repair_packet_sha256=parsed.repair_packet_sha256,
        repair_receipt_sha256=repair.receipt_sha256,
        pre_g8_corrected_y1_sha256=integration.surface.corrected_y1_camera_sha256,
        post_g8_corrected_y1_sha256=repair.output_camera_y1_sha256,
    )


def _encode_packet(
    program: PredictorPreservingA3ProgramV1,
    source: PostG8ConditionalASourceBindingV1,
) -> bytes:
    if type(program) is not PredictorPreservingA3ProgramV1:
        raise PostG8ConditionalAError("post-G8 A compile requires exact A3 program type")
    if type(source) is not PostG8ConditionalASourceBindingV1:
        raise PostG8ConditionalAError("post-G8 A packet requires exact source binding")
    addressed = tuple(row.source_pair_id for row in program.constant_rgb_cells) + tuple(
        row.source_pair_id for row in program.corrected_y1_copy_cells
    )
    if any(pair_id not in source.source_pair_ids for pair_id in addressed):
        raise PostG8ConditionalAError("post-G8 A row addresses a pair outside the bound source window")

    body = bytearray()
    if program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
        for row in program.constant_rgb_cells:
            body.extend(
                RGB_ROW.pack(
                    row.source_pair_id,
                    row.scorer_row,
                    row.scorer_col,
                    *row.rgb_u8,
                )
            )
    elif program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1:
        for row in program.corrected_y1_copy_cells:
            body.extend(COPY_ROW.pack(row.source_pair_id, row.scorer_row, row.scorer_col))
    if len(body) != program.body_bytes:
        raise PostG8ConditionalAError("post-G8 A packet body accounting changed")
    if program.control_row_count > 0xFFFFFFFF or len(body) > 0xFFFFFFFF:
        raise PostG8ConditionalAError("post-G8 A packet exceeds the version-1 integer ABI")
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
    header_zero_crc = PACKET_HEADER.pack(*fixed, 0)
    crc = zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF
    return PACKET_HEADER.pack(*fixed, crc) + bytes(body)


@dataclass(frozen=True, slots=True)
class ParsedPostG8ConditionalAPacketV1:
    packet: bytes
    source_binding_sha256: str
    pair_start: int
    pair_count: int
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    coordinate_bytes: int
    rgb_value_bytes: int
    program: PredictorPreservingA3ProgramV1


def parse_post_g8_conditional_a_packet(
    packet: bytes,
    *,
    expected_source_binding: PostG8ConditionalASourceBindingV1,
) -> ParsedPostG8ConditionalAPacketV1:
    """Strict parse with exact EOF, CRC, source-FK, and re-encode closure."""

    if type(packet) is not bytes or len(packet) < PACKET_HEADER.size:
        raise PostG8ConditionalAError("post-G8 A packet is shorter than its exact header")
    if type(expected_source_binding) is not PostG8ConditionalASourceBindingV1:
        raise PostG8ConditionalAError("post-G8 A parse requires exact expected source binding")
    try:
        (
            magic,
            version,
            mode_wire,
            pair_start,
            pair_count,
            source_binding_raw,
            row_count,
            body_bytes,
            packet_bytes,
            crc,
        ) = PACKET_HEADER.unpack_from(packet, 0)
    except struct.error as exc:  # pragma: no cover - guarded length
        raise PostG8ConditionalAError("post-G8 A packet header is malformed") from exc
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise PostG8ConditionalAError("post-G8 A packet magic/version is unsupported")
    try:
        mode = _WIRE_TO_MODE[mode_wire]
    except KeyError as exc:
        raise PostG8ConditionalAError("post-G8 A packet mode is outside the closed universe") from exc
    if pair_start != expected_source_binding.pair_start or pair_count != expected_source_binding.pair_count:
        raise PostG8ConditionalAError("post-G8 A packet pair window differs from exact source binding")
    if source_binding_raw.hex() != expected_source_binding.binding_sha256:
        raise PostG8ConditionalAError("post-G8 A packet source binding differs from live P/G8/P0/Y1")
    if packet_bytes != len(packet) or body_bytes != len(packet) - PACKET_HEADER.size:
        raise PostG8ConditionalAError("post-G8 A packet length accounting or trailing bytes changed")
    row_size = {
        PredictorPreservingA3Mode.PASS_P0_V1: 0,
        PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: RGB_ROW.size,
        PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: COPY_ROW.size,
    }[mode]
    if mode is PredictorPreservingA3Mode.PASS_P0_V1:
        if row_count != 0 or body_bytes != 0:
            raise PostG8ConditionalAError("post-G8 PASS_P0 packet must have zero rows and body bytes")
    elif row_count == 0 or body_bytes != row_count * row_size:
        raise PostG8ConditionalAError("post-G8 A row count/body length differs from its mode ABI")
    body = packet[PACKET_HEADER.size :]
    header_zero_crc = PACKET_HEADER.pack(
        magic,
        version,
        mode_wire,
        pair_start,
        pair_count,
        source_binding_raw,
        row_count,
        body_bytes,
        packet_bytes,
        0,
    )
    if zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF != crc:
        raise PostG8ConditionalAError("post-G8 A packet CRC differs from exact bytes")

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
        raise PostG8ConditionalAError("post-G8 A control row is malformed") from exc
    if offset != len(body):
        raise PostG8ConditionalAError("post-G8 A packet body was not consumed exactly")
    try:
        program = PredictorPreservingA3ProgramV1(
            mode=mode,
            constant_rgb_cells=tuple(rgb_rows),
            corrected_y1_copy_cells=tuple(copy_rows),
        )
    except PredictorPreservingA3Error as exc:
        raise PostG8ConditionalAError("post-G8 A packet program is noncanonical") from exc
    addressed = tuple(row.source_pair_id for row in program.constant_rgb_cells) + tuple(
        row.source_pair_id for row in program.corrected_y1_copy_cells
    )
    if any(pair_id not in expected_source_binding.source_pair_ids for pair_id in addressed):
        raise PostG8ConditionalAError("post-G8 A packet row addresses a foreign pair")
    if _encode_packet(program, expected_source_binding) != packet:
        raise PostG8ConditionalAError("post-G8 A packet changed on strict typed parse/re-encode")
    return ParsedPostG8ConditionalAPacketV1(
        packet=packet,
        source_binding_sha256=expected_source_binding.binding_sha256,
        pair_start=pair_start,
        pair_count=pair_count,
        packet_bytes=packet_bytes,
        packet_header_bytes=PACKET_HEADER.size,
        packet_body_bytes=body_bytes,
        coordinate_bytes=program.coordinate_bytes,
        rgb_value_bytes=program.rgb_value_bytes,
        program=program,
    )


@dataclass(frozen=True, slots=True)
class PostG8ConditionalAReceiptV1:
    """Strict proof for one counted post-G8 conditional-A reconstruction."""

    schema: Literal["tac.taskspace_post_g8_conditional_a_receipt.v1"]
    mode: str
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str
    predictor_labels_sha256: str
    predictor_camera_p0_sha256: str
    predictor_camera_p1_sha256: str
    predictor_section_identity_sha256: str
    composite_g_section_sha256: str
    semantic_g_packet_sha256: str
    semantic_g_labels_sha256: str
    repair_packet_sha256: str
    repair_receipt_sha256: str
    pre_g8_corrected_y1_sha256: str
    post_g8_corrected_y1_sha256: str
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
    post_g8_y1_support_numerators_exact: bool
    post_g8_y1_unchanged: Literal[True] = True
    g8_semantic_labels_unchanged: Literal[True] = True
    canonical_disjoint_reconstruction_reused: Literal[True] = True
    outside_a_owned_p0_camera_values_preserved: Literal[True] = True
    outside_a_owned_p0_scorer_numerators_preserved: Literal[True] = True
    packet_parse_reencode_exact: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    dense_post_g8_y1_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    through_r_target_realization_debt_closed: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise PostG8ConditionalAError("post-G8 A receipt schema changed")
        try:
            mode = PredictorPreservingA3Mode(self.mode)
        except ValueError as exc:
            raise PostG8ConditionalAError("post-G8 A receipt mode is outside the closed universe") from exc
        _validate_pair_ids(self.source_pair_ids, field="receipt source_pair_ids")
        source_fields = {
            field: getattr(self, field)
            for field in PostG8ConditionalASourceBindingV1.__dataclass_fields__
            if field not in {"schema", "source_pair_ids"}
        }
        source = PostG8ConditionalASourceBindingV1(
            schema=SOURCE_SCHEMA,
            source_pair_ids=self.source_pair_ids,
            **source_fields,
        )
        if source.binding_sha256 != self.source_binding_sha256:
            raise PostG8ConditionalAError("post-G8 A receipt source binding does not recompose")
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
        exact_ints = (
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
        if any(type(getattr(self, field)) is not int for field in exact_ints):
            raise PostG8ConditionalAError("post-G8 A receipt byte/count fields must be exact integers")
        if (
            self.packet_header_bytes != PACKET_HEADER.size
            or self.packet_bytes != self.packet_header_bytes + self.packet_body_bytes
            or self.packet_body_bytes != self.serialized_coordinate_bytes + self.serialized_rgb_value_bytes
            or self.control_row_count < 0
            or self.owned_scorer_cells < 0
            or self.owned_camera_values < 0
            or self.actually_changed_camera_values < 0
            or self.scorer_denominator <= 0
        ):
            raise PostG8ConditionalAError("post-G8 A receipt byte/count accounting changed")
        if self.output_camera_y1_sha256 != self.post_g8_corrected_y1_sha256:
            raise PostG8ConditionalAError("post-G8 A receipt changed the bound post-G8 Y1")
        expected_mode_truth = {
            PredictorPreservingA3Mode.PASS_P0_V1: (True, False, False),
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: (False, True, False),
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: (False, False, True),
        }[mode]
        actual_mode_truth = (
            self.p0_pass_through_zero_mutation,
            self.constant_rgb_target_numerators_exact,
            self.post_g8_y1_support_numerators_exact,
        )
        if actual_mode_truth != expected_mode_truth:
            raise PostG8ConditionalAError("post-G8 A receipt mode-specific proof flags disagree")
        if mode is PredictorPreservingA3Mode.PASS_P0_V1:
            if self.control_row_count != 0 or self.actually_changed_camera_values != 0:
                raise PostG8ConditionalAError("post-G8 PASS_P0 receipt contains a mutation")
        elif self.control_row_count < 1 or self.owned_scorer_cells != self.control_row_count:
            raise PostG8ConditionalAError("post-G8 A receipt ownership differs from control rows")
        truth = {
            "post_g8_y1_unchanged": True,
            "g8_semantic_labels_unchanged": True,
            "canonical_disjoint_reconstruction_reused": True,
            "outside_a_owned_p0_camera_values_preserved": True,
            "outside_a_owned_p0_scorer_numerators_preserved": True,
            "packet_parse_reencode_exact": True,
            "deterministic_double_decode": True,
            "dense_post_g8_y1_serialized": False,
            "scorer_invoked": False,
            "n600_evidence_claim": False,
            "exact_score_claim": False,
            "candidate_archive_eligible": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "through_r_target_realization_debt_closed": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise PostG8ConditionalAError("post-G8 A receipt truth labels became permissive")

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
class DecodedPostG8ConditionalAV1:
    camera_y0: np.ndarray
    camera_y1: np.ndarray
    chronological_camera_frames: np.ndarray
    owned_scorer_mask: np.ndarray
    owned_camera_mask: np.ndarray
    receipt: PostG8ConditionalAReceiptV1

    def __post_init__(self) -> None:
        if type(self.receipt) is not PostG8ConditionalAReceiptV1:
            raise PostG8ConditionalAError("decoded post-G8 A requires exact receipt type")
        pair_count = len(self.receipt.source_pair_ids)
        y0 = np.asarray(self.camera_y0)
        y1 = np.asarray(self.camera_y1)
        chronological = np.asarray(self.chronological_camera_frames)
        scorer_mask = np.asarray(self.owned_scorer_mask)
        camera_mask = np.asarray(self.owned_camera_mask)
        if y0.dtype != np.uint8 or y0.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
            raise PostG8ConditionalAError("decoded post-G8 A Y0 camera ABI changed")
        if y1.dtype != np.uint8 or y1.shape != y0.shape:
            raise PostG8ConditionalAError("decoded post-G8 A Y1 camera ABI changed")
        if chronological.dtype != np.uint8 or chronological.shape != (
            pair_count,
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        ):
            raise PostG8ConditionalAError("decoded post-G8 A chronology ABI changed")
        if scorer_mask.dtype != np.bool_ or scorer_mask.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH):
            raise PostG8ConditionalAError("decoded post-G8 A scorer mask ABI changed")
        if camera_mask.dtype != np.bool_ or camera_mask.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH):
            raise PostG8ConditionalAError("decoded post-G8 A camera mask ABI changed")
        expected = {
            "output_camera_y0_sha256": _array_sha256(y0),
            "output_camera_y1_sha256": _array_sha256(y1),
            "chronological_camera_frames_sha256": _array_sha256(chronological),
            "owned_scorer_mask_sha256": _array_sha256(scorer_mask),
            "owned_camera_mask_sha256": _array_sha256(camera_mask),
        }
        if any(getattr(self.receipt, field) != digest for field, digest in expected.items()):
            raise PostG8ConditionalAError("decoded post-G8 A arrays differ from receipt hashes")
        if not np.array_equal(chronological[:, 0], y0) or not np.array_equal(chronological[:, 1], y1):
            raise PostG8ConditionalAError("decoded post-G8 A chronology is not exact [Y0,Y1]")
        object.__setattr__(self, "camera_y0", _immutable_array(y0, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "camera_y1", _immutable_array(y1, dtype=np.dtype(np.uint8)))
        object.__setattr__(
            self,
            "chronological_camera_frames",
            _immutable_array(chronological, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "owned_scorer_mask",
            _immutable_array(scorer_mask, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "owned_camera_mask",
            _immutable_array(camera_mask, dtype=np.dtype(np.bool_)),
        )


def _decode_parsed_packet(
    parsed: ParsedPostG8ConditionalAPacketV1,
    source: PostG8ConditionalASourceBindingV1,
    surface: PostG8ConditionalASurfaceV1,
) -> DecodedPostG8ConditionalAV1:
    program = parsed.program
    p0 = surface.predictor_camera_p0
    post_g8_y1 = surface.post_g8_corrected_y1
    try:
        reconstruction = apply_disjoint_camera_cell_reconstruction_v1(
            p0,
            source_pair_ids=source.source_pair_ids,
            constant_rgb_cells=program.constant_rgb_cells,
            copy_source_camera=(
                post_g8_y1 if program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1 else None
            ),
            copy_source_cells=program.corrected_y1_copy_cells,
        )
    except PredictorPreservingA3Error as exc:
        raise PostG8ConditionalAError("post-G8 A canonical disjoint reconstruction failed") from exc
    y0 = reconstruction.output_camera
    if program.mode is PredictorPreservingA3Mode.PASS_P0_V1:
        if not np.array_equal(y0, p0) or np.any(reconstruction.owned_scorer_mask):
            raise PostG8ConditionalAError("post-G8 PASS_P0 failed exact zero-mutation pass-through")
    elif program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
        if not reconstruction.constant_rgb_target_numerators_exact:
            raise PostG8ConditionalAError("post-G8 sparse RGB lacked exact numerator proof")
    elif not reconstruction.copy_source_target_numerators_exact:
        raise PostG8ConditionalAError("post-G8 Y1 support copy lacked exact numerator proof")
    if _array_sha256(post_g8_y1) != source.post_g8_corrected_y1_sha256:
        raise PostG8ConditionalAError("post-G8 Y1 changed after source binding")
    chronological = np.stack((y0, post_g8_y1), axis=1)
    receipt = PostG8ConditionalAReceiptV1(
        schema=RECEIPT_SCHEMA,
        mode=program.mode.value,
        source_pair_ids=source.source_pair_ids,
        source_binding_sha256=source.binding_sha256,
        **{
            field: getattr(source, field)
            for field in PostG8ConditionalASourceBindingV1.__dataclass_fields__
            if field not in {"schema", "source_pair_ids"}
        },
        packet_sha256=_sha256(parsed.packet),
        packet_bytes=parsed.packet_bytes,
        packet_header_bytes=parsed.packet_header_bytes,
        packet_body_bytes=parsed.packet_body_bytes,
        serialized_coordinate_bytes=parsed.coordinate_bytes,
        serialized_rgb_value_bytes=parsed.rgb_value_bytes,
        control_row_count=program.control_row_count,
        output_camera_y0_sha256=_array_sha256(y0),
        output_camera_y1_sha256=_array_sha256(post_g8_y1),
        chronological_camera_frames_sha256=_array_sha256(chronological),
        owned_scorer_mask_sha256=_array_sha256(reconstruction.owned_scorer_mask),
        owned_camera_mask_sha256=_array_sha256(reconstruction.owned_camera_mask),
        predictor_p0_scorer_numerators_sha256=_array_sha256(reconstruction.base_scorer_numerators),
        output_y0_scorer_numerators_sha256=_array_sha256(reconstruction.output_scorer_numerators),
        owned_scorer_cells=int(np.count_nonzero(reconstruction.owned_scorer_mask)),
        owned_camera_values=int(np.count_nonzero(reconstruction.owned_camera_mask)) * CHANNELS,
        actually_changed_camera_values=reconstruction.actually_changed_camera_values,
        scorer_denominator=reconstruction.scorer_denominator,
        p0_pass_through_zero_mutation=program.mode is PredictorPreservingA3Mode.PASS_P0_V1,
        constant_rgb_target_numerators_exact=(program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1),
        post_g8_y1_support_numerators_exact=(program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1),
    )
    return DecodedPostG8ConditionalAV1(
        camera_y0=y0,
        camera_y1=post_g8_y1,
        chronological_camera_frames=chronological,
        owned_scorer_mask=reconstruction.owned_scorer_mask,
        owned_camera_mask=reconstruction.owned_camera_mask,
        receipt=receipt,
    )


def decode_post_g8_conditional_a_packet(
    packet: bytes,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    g8_integration: DecodedSameClassRealizationRepairPGAIntegrationV1,
) -> DecodedPostG8ConditionalAV1:
    """Decode twice against exact P0 and post-G8 Y1, with strict receipt closure."""

    surface = build_post_g8_conditional_a_surface(
        predictor_surface=predictor_surface,
        g8_integration=g8_integration,
    )
    source = derive_post_g8_conditional_a_source_binding(surface)
    parsed = parse_post_g8_conditional_a_packet(packet, expected_source_binding=source)
    first = _decode_parsed_packet(parsed, source, surface)
    second = _decode_parsed_packet(parsed, source, surface)
    if first.receipt != second.receipt:
        raise PostG8ConditionalAError("post-G8 A deterministic double decode changed receipt")
    for field in (
        "camera_y0",
        "camera_y1",
        "chronological_camera_frames",
        "owned_scorer_mask",
        "owned_camera_mask",
    ):
        if not np.array_equal(getattr(first, field), getattr(second, field)):
            raise PostG8ConditionalAError(f"post-G8 A deterministic double decode changed {field}")
    if parse_post_g8_conditional_a_receipt(first.receipt.to_receipt_bytes()) != first.receipt:
        raise PostG8ConditionalAError("post-G8 A receipt failed strict parse/re-emit closure")
    return first


def parse_post_g8_conditional_a_receipt(payload: bytes) -> PostG8ConditionalAReceiptV1:
    """Strict canonical JSON parser with duplicate-key and exact-type rejection."""

    if type(payload) is not bytes or not payload:
        raise PostG8ConditionalAError("post-G8 A receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostG8ConditionalAError(f"post-G8 A receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PostG8ConditionalAError(f"post-G8 A receipt contains non-finite constant {token}")
            ),
        )
    except PostG8ConditionalAError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostG8ConditionalAError("post-G8 A receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(PostG8ConditionalAReceiptV1.__dataclass_fields__):
        raise PostG8ConditionalAError("post-G8 A receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise PostG8ConditionalAError("post-G8 A receipt is not canonical JSON")
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise PostG8ConditionalAError("post-G8 A receipt pair IDs must be exact JSON integers")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        receipt = PostG8ConditionalAReceiptV1(**value)
    except TypeError as exc:
        raise PostG8ConditionalAError("post-G8 A receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise PostG8ConditionalAError("post-G8 A receipt changed on typed parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class CompiledPostG8ConditionalAV1:
    packet: bytes
    program: PredictorPreservingA3ProgramV1
    source_binding: PostG8ConditionalASourceBindingV1
    decoded: DecodedPostG8ConditionalAV1
    receipt: PostG8ConditionalAReceiptV1

    def __post_init__(self) -> None:
        if type(self.packet) is not bytes or not self.packet.startswith(PACKET_MAGIC):
            raise PostG8ConditionalAError("compiled post-G8 A packet changed exact domain")
        if type(self.program) is not PredictorPreservingA3ProgramV1:
            raise PostG8ConditionalAError("compiled post-G8 A program changed exact type")
        if type(self.source_binding) is not PostG8ConditionalASourceBindingV1:
            raise PostG8ConditionalAError("compiled post-G8 A source changed exact type")
        if type(self.decoded) is not DecodedPostG8ConditionalAV1 or self.decoded.receipt != self.receipt:
            raise PostG8ConditionalAError("compiled post-G8 A decode and receipt diverged")
        if (
            self.receipt.packet_sha256 != _sha256(self.packet)
            or self.receipt.packet_bytes != len(self.packet)
            or self.receipt.source_binding_sha256 != self.source_binding.binding_sha256
        ):
            raise PostG8ConditionalAError("compiled post-G8 A packet/source differs from receipt")


def compile_post_g8_conditional_a(
    program: PredictorPreservingA3ProgramV1,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    g8_integration: DecodedSameClassRealizationRepairPGAIntegrationV1,
) -> CompiledPostG8ConditionalAV1:
    """Compile canonical A3 controls into the distinct post-G8 packet domain."""

    if type(program) is not PredictorPreservingA3ProgramV1:
        raise PostG8ConditionalAError("post-G8 A compile requires exact A3 program type")
    surface = build_post_g8_conditional_a_surface(
        predictor_surface=predictor_surface,
        g8_integration=g8_integration,
    )
    source = derive_post_g8_conditional_a_source_binding(surface)
    packet = _encode_packet(program, source)
    decoded = decode_post_g8_conditional_a_packet(
        packet,
        predictor_surface=predictor_surface,
        g8_integration=g8_integration,
    )
    return CompiledPostG8ConditionalAV1(
        packet=packet,
        program=program,
        source_binding=source,
        decoded=decoded,
        receipt=decoded.receipt,
    )


__all__ = [
    "PACKET_HEADER",
    "PACKET_MAGIC",
    "PACKET_VERSION",
    "CompiledPostG8ConditionalAV1",
    "DecodedPostG8ConditionalAV1",
    "ParsedPostG8ConditionalAPacketV1",
    "PostG8ConditionalAError",
    "PostG8ConditionalAReceiptV1",
    "PostG8ConditionalASourceBindingV1",
    "PostG8ConditionalASurfaceV1",
    "build_post_g8_conditional_a_surface",
    "compile_post_g8_conditional_a",
    "decode_post_g8_conditional_a_packet",
    "derive_post_g8_conditional_a_source_binding",
    "parse_post_g8_conditional_a_packet",
    "parse_post_g8_conditional_a_receipt",
]
