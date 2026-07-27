# SPDX-License-Identifier: MIT
"""Class-complete, population-addressed Y1 semantic operand over exact V15 P.

This is the first executable semantic wire in the task-space line that admits
all five frozen SegNet classes without falling back to a dense label table:

* Road and UndrivableBoundary use donor-exact directional shearlet atoms;
* Lane uses the inherited coherent lane chart and periodic/drift programs;
* Movable uses island atoms and persistent worldsheet tracks; and
* MyCar uses the immutable P static field plus sparse topology exceptions.

Topology geometry is dictionary factored across semantic roles.  The operand
contains no scorer weights, target labels, costates, thresholds, RGB residual,
or learned dense quotient.  Costates and whole-state archive arbitration are
encoder-side admission authorities and intentionally do not cross this wire.

The receiver reopens the exact counted semantic P, applies the analytic
program, renders camera-resolution uint8, preserves Y0 exactly, and emits the
mutated Y1.  Exact integer R diagnostics and a strict outer archive builder are
provided so a structural receipt can prove parse/re-encode and double-decode
identity.  This module does not claim a score, public ``inflate.sh`` closure,
or a full-n600 selected program.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    _ROLE_TO_WIRE,
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    BoundaryShearletAtomV1,
    CarrierComposeReceiverV1,
    DirectDescriptionError,
    IslandShapeAtomV1,
    LaneDriftKnotV1,
    LanePeriodicProgramV1,
    MovableWorldsheetKnotV1,
    MovableWorldsheetTrackV1,
    TopologyEventV1,
    _apply_lane_predictor_programs,
    _decode_boundary_shearlet_atoms,
    _decode_island_shape_atoms,
    _decode_lane_knots,
    _decode_lane_programs,
    _decode_worldsheet_knots,
    _decode_worldsheet_tracks,
    _encode_boundary_shearlet_atoms,
    _encode_island_shape_atoms,
    _encode_lane_knots,
    _encode_lane_programs,
    _encode_topology_events,
    _encode_worldsheet_knots,
    _encode_worldsheet_tracks,
)
from tac.witness_dsl.c0b_semantic_quotient import exact_resize_round_u8
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    V15RoleAwareOverlayDecoderV1,
    V15RoleAwareOverlayError,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    TaskspaceOuterArchiveError,
    build_taskspace_outer_archive,
    parse_taskspace_outer_archive,
)

OPERAND_MAGIC: Final = b"G89CCS1\x00"
OPERAND_VERSION: Final = 1
PRODUCT_MAGIC: Final = b"G89CCP1\x00"
PRODUCT_VERSION: Final = 1

OPERAND_CONTRACT_ID: Final = "tac.g89.class_complete_y1_semantic_operand.v1"
RECEIVER_CONTRACT_ID: Final = "tac.g89.exact_v15_p_class_complete_y1_receiver.v1"
G83_DECODER_TRANSITION_ID: Final = "g89:1:CLASS_COMPLETE_Y1_SEMANTIC"
SELECTION_CONTRACT_ID: Final = "EXTERNAL_G83_COSTATE_WHOLE_STATE_SCORE_ARBITRATION_NO_FIXED_THRESHOLDS"
IRREDUCIBLE_QUOTIENT_ID: Final = "TARGET_LABELS_MINUS_CLASS_COMPLETE_ANALYTIC_SPAN_AFTER_EXACT_R_AND_ARGMAX"

OPEN_PRODUCT_BLOCKERS: Final = (
    "G89_CURRENT_BASE_COSTATE_PROJECTED_GLOBAL_PROGRAM_NOT_MATERIALIZED",
    "G89_FULL_N600_CLASS_COMPLETE_OPERAND_NOT_MATERIALIZED",
    "G89_CONDITIONAL_Y0_GIVEN_EXACT_Y1_NOT_COMPOSED",
    "G89_PUBLIC_INFLATE_SH_RUNTIME_INTEGRATION_OWED",
    "G89_UPSTREAM_EVALUATE_PY_N600_AUTHORITY_OWED",
)

_WIRE_TO_ROLE: Final = {value: key for key, value in _ROLE_TO_WIRE.items()}
_ACTION_TO_WIRE: Final = {"birth": 1, "death": 2}
_WIRE_TO_ACTION: Final = {value: key for key, value in _ACTION_TO_WIRE.items()}
_SHAPE_TO_WIRE: Final = {"ellipse": 1, "box": 2}
_WIRE_TO_SHAPE: Final = {value: key for key, value in _SHAPE_TO_WIRE.items()}

# Header: magic/version/semantic SHA, eight length-delimited physical streams.
_OPERAND_HEADER: Final = struct.Struct(">8sB32s8I")
_TEMPLATE_HEADER: Final = struct.Struct(">5sBHI")
_TEMPLATE_ROW: Final = struct.Struct(">HBBBHHbb")
_APPLICATION_HEADER: Final = struct.Struct(">5sBHI")
_APPLICATION_ROW: Final = struct.Struct(">HBHHH")
_TEMPLATE_MAGIC: Final = b"G89TT"
_APPLICATION_MAGIC: Final = b"G89TA"
_PRODUCT_HEADER: Final = struct.Struct(">8sBII")

_MAX_STREAM_BYTES: Final = 64 * 1024 * 1024
_MAX_ROWS: Final = 0xFFFF


class ClassCompleteSemanticError(ValueError):
    """Malformed program, custody mismatch, or receiver execution failure."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ClassCompleteSemanticError(f"{label} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ClassCompleteSemanticError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


@dataclass(frozen=True, order=True, slots=True)
class SharedTopologyTemplateV1:
    """One reusable topology shape/lifecycle row with no pair or role address."""

    template_id: int
    action: str
    shape: str
    lifetime: int
    height: int
    width: int
    transport_gain_x_q4: int = 0
    transport_gain_y_q4: int = 0

    def __post_init__(self) -> None:
        _require_exact_int(self.template_id, "template_id", minimum=0, maximum=0xFFFF)
        if self.action not in _ACTION_TO_WIRE or self.shape not in _SHAPE_TO_WIRE:
            raise ClassCompleteSemanticError("shared topology action/shape is unknown")
        _require_exact_int(self.lifetime, "lifetime", minimum=1, maximum=255)
        _require_exact_int(self.height, "height", minimum=1, maximum=SCORER_HEIGHT)
        _require_exact_int(self.width, "width", minimum=1, maximum=SCORER_WIDTH)
        for field in ("transport_gain_x_q4", "transport_gain_y_q4"):
            _require_exact_int(getattr(self, field), field, minimum=-128, maximum=127)
        if self.lifetime == 1 and (self.transport_gain_x_q4 or self.transport_gain_y_q4):
            raise ClassCompleteSemanticError("one-pair shared topology template cannot carry inert transport")


@dataclass(frozen=True, order=True, slots=True)
class SharedTopologyApplicationV1:
    """One role/pair placement of a shared topology template."""

    pair_index: int
    role: str
    template_id: int
    y0: int
    x0: int

    def __post_init__(self) -> None:
        _require_exact_int(self.pair_index, "pair_index", minimum=0, maximum=599)
        if self.role not in _ROLE_TO_WIRE:
            raise ClassCompleteSemanticError("shared topology application role is unknown")
        _require_exact_int(self.template_id, "template_id", minimum=0, maximum=0xFFFF)
        _require_exact_int(self.y0, "y0", minimum=0, maximum=SCORER_HEIGHT - 1)
        _require_exact_int(self.x0, "x0", minimum=0, maximum=SCORER_WIDTH - 1)


def _encode_templates(rows: tuple[SharedTopologyTemplateV1, ...]) -> bytes:
    if not rows:
        raise ClassCompleteSemanticError("class-complete operand requires topology templates")
    keys = [row.template_id for row in rows]
    if len(rows) > _MAX_ROWS or keys != sorted(set(keys)):
        raise ClassCompleteSemanticError("topology templates must be unique canonical order")
    body = b"".join(
        _TEMPLATE_ROW.pack(
            row.template_id,
            _ACTION_TO_WIRE[row.action],
            _SHAPE_TO_WIRE[row.shape],
            row.lifetime,
            row.height,
            row.width,
            row.transport_gain_x_q4,
            row.transport_gain_y_q4,
        )
        for row in rows
    )
    return (
        _TEMPLATE_HEADER.pack(
            _TEMPLATE_MAGIC,
            OPERAND_VERSION,
            len(rows),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        + body
    )


def _decode_templates(payload: bytes) -> tuple[SharedTopologyTemplateV1, ...]:
    if len(payload) < _TEMPLATE_HEADER.size:
        raise ClassCompleteSemanticError("topology-template stream is truncated")
    magic, version, count, checksum = _TEMPLATE_HEADER.unpack_from(payload)
    body = payload[_TEMPLATE_HEADER.size :]
    if (
        magic != _TEMPLATE_MAGIC
        or version != OPERAND_VERSION
        or count == 0
        or len(body) != count * _TEMPLATE_ROW.size
        or (zlib.crc32(body) & 0xFFFFFFFF) != checksum
    ):
        raise ClassCompleteSemanticError("topology-template stream header/CRC is invalid")
    rows: list[SharedTopologyTemplateV1] = []
    for index in range(count):
        values = _TEMPLATE_ROW.unpack_from(body, index * _TEMPLATE_ROW.size)
        template_id, action, shape, lifetime, height, width, gain_x, gain_y = values
        if action not in _WIRE_TO_ACTION or shape not in _WIRE_TO_SHAPE:
            raise ClassCompleteSemanticError("topology-template stream has unknown enum")
        rows.append(
            SharedTopologyTemplateV1(
                template_id,
                _WIRE_TO_ACTION[action],
                _WIRE_TO_SHAPE[shape],
                lifetime,
                height,
                width,
                gain_x,
                gain_y,
            )
        )
    result = tuple(rows)
    if _encode_templates(result) != payload:
        raise ClassCompleteSemanticError("topology-template parse/re-encode changed bytes")
    return result


def _encode_applications(rows: tuple[SharedTopologyApplicationV1, ...]) -> bytes:
    if not rows:
        raise ClassCompleteSemanticError("class-complete operand requires topology applications")
    keys = [(row.pair_index, _ROLE_TO_WIRE[row.role], row.template_id, row.y0, row.x0) for row in rows]
    if len(rows) > _MAX_ROWS or keys != sorted(set(keys)):
        raise ClassCompleteSemanticError("topology applications must be unique canonical order")
    body = b"".join(
        _APPLICATION_ROW.pack(
            row.pair_index,
            _ROLE_TO_WIRE[row.role],
            row.template_id,
            row.y0,
            row.x0,
        )
        for row in rows
    )
    return (
        _APPLICATION_HEADER.pack(
            _APPLICATION_MAGIC,
            OPERAND_VERSION,
            len(rows),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        + body
    )


def _decode_applications(payload: bytes) -> tuple[SharedTopologyApplicationV1, ...]:
    if len(payload) < _APPLICATION_HEADER.size:
        raise ClassCompleteSemanticError("topology-application stream is truncated")
    magic, version, count, checksum = _APPLICATION_HEADER.unpack_from(payload)
    body = payload[_APPLICATION_HEADER.size :]
    if (
        magic != _APPLICATION_MAGIC
        or version != OPERAND_VERSION
        or count == 0
        or len(body) != count * _APPLICATION_ROW.size
        or (zlib.crc32(body) & 0xFFFFFFFF) != checksum
    ):
        raise ClassCompleteSemanticError("topology-application stream header/CRC is invalid")
    rows: list[SharedTopologyApplicationV1] = []
    for index in range(count):
        pair, role_wire, template_id, y0, x0 = _APPLICATION_ROW.unpack_from(body, index * _APPLICATION_ROW.size)
        if role_wire not in _WIRE_TO_ROLE:
            raise ClassCompleteSemanticError("topology-application stream has unknown role")
        rows.append(
            SharedTopologyApplicationV1(
                pair,
                _WIRE_TO_ROLE[role_wire],
                template_id,
                y0,
                x0,
            )
        )
    result = tuple(rows)
    if _encode_applications(result) != payload:
        raise ClassCompleteSemanticError("topology-application parse/re-encode changed bytes")
    return result


@dataclass(frozen=True, slots=True)
class ClassCompleteSemanticProgramV1:
    """The complete analytic semantic operand; no target or costate payload."""

    semantic_archive_sha256: str
    topology_templates: tuple[SharedTopologyTemplateV1, ...]
    topology_applications: tuple[SharedTopologyApplicationV1, ...]
    boundary_shearlets: tuple[BoundaryShearletAtomV1, ...]
    island_shapes: tuple[IslandShapeAtomV1, ...]
    worldsheet_tracks: tuple[MovableWorldsheetTrackV1, ...]
    worldsheet_knots: tuple[MovableWorldsheetKnotV1, ...]
    lane_programs: tuple[LanePeriodicProgramV1, ...]
    lane_knots: tuple[LaneDriftKnotV1, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.semantic_archive_sha256, "semantic_archive_sha256")
        exact_types = (
            (self.topology_templates, SharedTopologyTemplateV1),
            (self.topology_applications, SharedTopologyApplicationV1),
            (self.boundary_shearlets, BoundaryShearletAtomV1),
            (self.island_shapes, IslandShapeAtomV1),
            (self.worldsheet_tracks, MovableWorldsheetTrackV1),
            (self.worldsheet_knots, MovableWorldsheetKnotV1),
            (self.lane_programs, LanePeriodicProgramV1),
            (self.lane_knots, LaneDriftKnotV1),
        )
        for rows, expected_type in exact_types:
            if type(rows) is not tuple or any(type(row) is not expected_type for row in rows):
                raise ClassCompleteSemanticError(f"program stream must be exact tuple[{expected_type.__name__}]")

        template_ids = {row.template_id for row in self.topology_templates}
        if any(row.template_id not in template_ids for row in self.topology_applications):
            raise ClassCompleteSemanticError("topology application references an absent template")
        represented_roles = {row.role for row in self.topology_applications}
        if represented_roles != set(REALIZATION_PAINT_ORDER):
            raise ClassCompleteSemanticError(
                "class-complete program requires a topology application for all five roles"
            )
        shearlet_roles = {row.role for row in self.boundary_shearlets}
        if shearlet_roles != {"Road", "UndrivableBoundary"}:
            raise ClassCompleteSemanticError(
                "class-complete program requires both Road and UndrivableBoundary shearlets"
            )
        if not self.lane_programs:
            raise ClassCompleteSemanticError("class-complete program requires a coherent Lane program")
        if not self.worldsheet_tracks:
            raise ClassCompleteSemanticError("class-complete program requires a Movable worldsheet track")

        # Every donor encoder enforces canonical sort, uniqueness, and wire
        # bounds.  Round-tripping here makes constructor validity executable.
        streams = self._streams()
        decoders = (
            _decode_templates,
            _decode_applications,
            _decode_boundary_shearlet_atoms,
            _decode_island_shape_atoms,
            _decode_worldsheet_tracks,
            _decode_worldsheet_knots,
            _decode_lane_programs,
            _decode_lane_knots,
        )
        expected = (
            self.topology_templates,
            self.topology_applications,
            self.boundary_shearlets,
            self.island_shapes,
            self.worldsheet_tracks,
            self.worldsheet_knots,
            self.lane_programs,
            self.lane_knots,
        )
        try:
            if tuple(decoder(payload) for decoder, payload in zip(decoders, streams, strict=True)) != expected:
                raise ClassCompleteSemanticError("program stream round trip changed values")
        except DirectDescriptionError as exc:
            raise ClassCompleteSemanticError("donor stream validation refused program") from exc

        lowered = self.lower_topology_events()
        factored = len(streams[0]) + len(streams[1])
        direct = len(_encode_topology_events(lowered))
        if len(self.topology_applications) > len(self.topology_templates) and factored >= direct:
            raise ClassCompleteSemanticError("shared physical topology stream failed to save bytes over direct rows")

    def _streams(self) -> tuple[bytes, ...]:
        try:
            return (
                _encode_templates(self.topology_templates),
                _encode_applications(self.topology_applications),
                _encode_boundary_shearlet_atoms(self.boundary_shearlets),
                _encode_island_shape_atoms(self.island_shapes),
                _encode_worldsheet_tracks(self.worldsheet_tracks),
                _encode_worldsheet_knots(self.worldsheet_knots),
                _encode_lane_programs(self.lane_programs),
                _encode_lane_knots(self.lane_knots),
            )
        except DirectDescriptionError as exc:
            raise ClassCompleteSemanticError("donor stream encoding refused program") from exc

    def lower_topology_events(self) -> tuple[TopologyEventV1, ...]:
        template_by_id = {row.template_id: row for row in self.topology_templates}
        events: list[TopologyEventV1] = []
        for application in self.topology_applications:
            template = template_by_id[application.template_id]
            y1 = application.y0 + template.height
            x1 = application.x0 + template.width
            if y1 > SCORER_HEIGHT or x1 > SCORER_WIDTH:
                raise ClassCompleteSemanticError("topology template placement escapes scorer geometry")
            try:
                events.append(
                    TopologyEventV1(
                        pair_index=application.pair_index,
                        role=application.role,
                        action=template.action,
                        shape=template.shape,
                        lifetime=template.lifetime,
                        y0=application.y0,
                        x0=application.x0,
                        y1=y1,
                        x1=x1,
                        transport_gain_x_q4=template.transport_gain_x_q4,
                        transport_gain_y_q4=template.transport_gain_y_q4,
                    )
                )
            except DirectDescriptionError as exc:
                raise ClassCompleteSemanticError("lowered topology event is invalid") from exc
        return tuple(
            sorted(
                events,
                key=lambda row: (
                    row.pair_index,
                    _ROLE_TO_WIRE[row.role],
                    _ACTION_TO_WIRE[row.action],
                    _SHAPE_TO_WIRE[row.shape],
                    row.y0,
                    row.x0,
                    row.y1,
                    row.x1,
                ),
            )
        )

    @property
    def factored_topology_bytes(self) -> int:
        streams = self._streams()
        return len(streams[0]) + len(streams[1])

    @property
    def unfactored_topology_bytes(self) -> int:
        return len(_encode_topology_events(self.lower_topology_events()))

    def to_bytes(self) -> bytes:
        streams = self._streams()
        if any(len(stream) > _MAX_STREAM_BYTES for stream in streams):
            raise ClassCompleteSemanticError("program stream exceeds byte ceiling")
        return _OPERAND_HEADER.pack(
            OPERAND_MAGIC,
            OPERAND_VERSION,
            bytes.fromhex(self.semantic_archive_sha256),
            *(len(stream) for stream in streams),
        ) + b"".join(streams)

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())


def parse_class_complete_semantic_program(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    maximum_operand_bytes: int = _MAX_STREAM_BYTES,
) -> ClassCompleteSemanticProgramV1:
    """Strictly parse and re-emit one class-complete operand."""

    if type(payload) is not bytes or len(payload) < _OPERAND_HEADER.size:
        raise ClassCompleteSemanticError("class-complete operand is truncated/non-byte")
    _require_exact_int(
        maximum_operand_bytes,
        "maximum_operand_bytes",
        minimum=_OPERAND_HEADER.size,
        maximum=(1 << 32) - 1,
    )
    if len(payload) > maximum_operand_bytes:
        raise ClassCompleteSemanticError("class-complete operand exceeds byte ceiling")
    unpacked = _OPERAND_HEADER.unpack_from(payload)
    magic, version, semantic_sha_raw, *lengths = unpacked
    if magic != OPERAND_MAGIC or version != OPERAND_VERSION:
        raise ClassCompleteSemanticError("class-complete operand magic/version mismatch")
    if sum(lengths) != len(payload) - _OPERAND_HEADER.size:
        raise ClassCompleteSemanticError("class-complete operand length/EOF mismatch")
    if expected_sha256 is not None and _sha256(payload) != _require_sha256(expected_sha256, "expected_sha256"):
        raise ClassCompleteSemanticError("class-complete operand SHA mismatch")
    cursor = _OPERAND_HEADER.size
    streams: list[bytes] = []
    for length in lengths:
        streams.append(payload[cursor : cursor + length])
        cursor += length
    try:
        program = ClassCompleteSemanticProgramV1(
            semantic_archive_sha256=semantic_sha_raw.hex(),
            topology_templates=_decode_templates(streams[0]),
            topology_applications=_decode_applications(streams[1]),
            boundary_shearlets=_decode_boundary_shearlet_atoms(streams[2]),
            island_shapes=_decode_island_shape_atoms(streams[3]),
            worldsheet_tracks=_decode_worldsheet_tracks(streams[4]),
            worldsheet_knots=_decode_worldsheet_knots(streams[5]),
            lane_programs=_decode_lane_programs(streams[6]),
            lane_knots=_decode_lane_knots(streams[7]),
        )
    except DirectDescriptionError as exc:
        raise ClassCompleteSemanticError("donor stream parser refused operand") from exc
    if program.to_bytes() != payload:
        raise ClassCompleteSemanticError("operand parse/re-encode changed bytes")
    return program


def _collision_key_for_event(row: TopologyEventV1) -> tuple[object, ...]:
    return (
        row.pair_index,
        row.role,
        row.action,
        row.shape,
        row.y0,
        row.x0,
        row.y1,
        row.x1,
    )


def _merge_unique(
    existing: tuple[object, ...],
    added: tuple[object, ...],
    *,
    key,
    label: str,
) -> tuple[object, ...]:
    existing_keys = {key(row) for row in existing}
    added_keys = [key(row) for row in added]
    if len(added_keys) != len(set(added_keys)) or existing_keys.intersection(added_keys):
        raise ClassCompleteSemanticError(f"{label} address conflicts with immutable P")
    return tuple(sorted((*existing, *added), key=key))


def _semantic_cells(
    receiver: CarrierComposeReceiverV1,
    local_pair_ids: tuple[int, ...],
) -> np.ndarray:
    result = np.full(
        (len(local_pair_ids), SCORER_HEIGHT, SCORER_WIDTH),
        -1,
        dtype=np.int16,
    )
    layer_by_role = {row.role: row for row in receiver.layers}
    for role in REALIZATION_PAINT_ORDER:
        layer = layer_by_role[role]
        for local_index, pair_id in enumerate(local_pair_ids):
            mask = receiver._mask_for_layer(layer, pair_id, replace_g1_movable=True)
            result[local_index, mask] = ROLE_CLASS_IDS[role]
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class ClassCompleteDecodeV1:
    local_pair_ids: tuple[int, ...]
    source_pair_ids: tuple[int, ...]
    camera_pairs: np.ndarray
    exact_r_pairs: np.ndarray
    semantic_cells: np.ndarray
    base_camera_pairs: np.ndarray
    base_exact_r_pairs: np.ndarray
    base_semantic_cells: np.ndarray
    changed_camera_values: int
    changed_exact_r_values: int
    changed_semantic_cells: int
    changed_role_mask_cells: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True, init=False)
class ClassCompleteSemanticReceiverV1:
    """Exact P receiver plus one immutable class-complete Y1 program."""

    semantic_archive: bytes
    program_payload: bytes
    program: ClassCompleteSemanticProgramV1
    base: V15RoleAwareOverlayDecoderV1
    mutated: CarrierComposeReceiverV1

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ClassCompleteSemanticReceiverV1 must be constructed by .open()")

    @classmethod
    def open(
        cls,
        semantic_archive: bytes,
        program_payload: bytes,
        *,
        expected_semantic_archive_sha256: str,
        expected_program_sha256: str,
        verify_member_effects: bool = True,
    ) -> ClassCompleteSemanticReceiverV1:
        semantic_sha = _require_sha256(
            expected_semantic_archive_sha256,
            "expected_semantic_archive_sha256",
        )
        if type(semantic_archive) is not bytes or _sha256(semantic_archive) != semantic_sha:
            raise ClassCompleteSemanticError("semantic P exact custody mismatch")
        program = parse_class_complete_semantic_program(
            program_payload,
            expected_sha256=expected_program_sha256,
            maximum_operand_bytes=max(len(program_payload), _OPERAND_HEADER.size),
        )
        if program.semantic_archive_sha256 != semantic_sha:
            raise ClassCompleteSemanticError("program is bound to a different semantic P")
        try:
            base = V15RoleAwareOverlayDecoderV1.open(
                semantic_archive,
                expected_archive_bytes=len(semantic_archive),
                expected_archive_sha256=semantic_sha,
                verify_member_effects=verify_member_effects,
            )
        except V15RoleAwareOverlayError as exc:
            raise ClassCompleteSemanticError("strict semantic P reopen failed") from exc
        receiver = base.receiver
        source_start = receiver.predictor.source_pair_start
        source_stop = source_start + receiver.z.n_pairs

        addressed_pairs: list[int] = [
            *(row.pair_index for row in program.topology_applications),
            *(row.pair_index for row in program.boundary_shearlets),
            *(row.pair_index for row in program.island_shapes),
            *(row.birth_pair for row in program.worldsheet_tracks),
            *(row.pair_index for row in program.worldsheet_knots),
            *(row.birth_pair for row in program.lane_programs),
            *(row.pair_index for row in program.lane_knots),
        ]
        if any(value < source_start or value >= source_stop for value in addressed_pairs):
            raise ClassCompleteSemanticError("program address escaped semantic P window")

        events = _merge_unique(
            receiver.topology_events,
            program.lower_topology_events(),
            key=_collision_key_for_event,
            label="topology event",
        )
        shearlets = _merge_unique(
            receiver.boundary_shearlets,
            program.boundary_shearlets,
            key=lambda row: (row.pair_index, row.role, row.center_y, row.center_x),
            label="boundary shearlet",
        )
        islands = _merge_unique(
            receiver.island_shapes,
            program.island_shapes,
            key=lambda row: (row.pair_index, row.action, row.center_y, row.center_x),
            label="island shape",
        )
        tracks = _merge_unique(
            receiver.worldsheet_tracks,
            program.worldsheet_tracks,
            key=lambda row: row.object_id,
            label="worldsheet track",
        )
        knots = _merge_unique(
            receiver.worldsheet_knots,
            program.worldsheet_knots,
            key=lambda row: (row.object_id, row.pair_index),
            label="worldsheet knot",
        )
        lane_programs = _merge_unique(
            receiver.lane_programs,
            program.lane_programs,
            key=lambda row: row.line_index,
            label="lane program",
        )
        lane_knots = _merge_unique(
            receiver.lane_knots,
            program.lane_knots,
            key=lambda row: (row.line_index, row.pair_index),
            label="lane knot",
        )
        try:
            layers = _apply_lane_predictor_programs(
                receiver.layers,
                program.lane_programs,
                program.lane_knots,
                pose6_codes=receiver.pose6_codes,
                source_pair_start=source_start,
            )
        except DirectDescriptionError as exc:
            raise ClassCompleteSemanticError("Lane program execution failed") from exc
        mutated = replace(
            receiver,
            layers=layers,
            topology_events=events,
            boundary_shearlets=shearlets,
            island_shapes=islands,
            worldsheet_tracks=tracks,
            worldsheet_knots=knots,
            lane_programs=lane_programs,
            lane_knots=lane_knots,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "semantic_archive", semantic_archive)
        object.__setattr__(instance, "program_payload", program_payload)
        object.__setattr__(instance, "program", program)
        object.__setattr__(instance, "base", base)
        object.__setattr__(instance, "mutated", mutated)
        return instance

    def decode(self, local_pair_ids: tuple[int, ...]) -> ClassCompleteDecodeV1:
        if (
            type(local_pair_ids) is not tuple
            or not local_pair_ids
            or local_pair_ids != tuple(sorted(set(local_pair_ids)))
            or len(local_pair_ids) > 16
            or any(
                type(value) is not int or value < 0 or value >= self.base.receiver.z.n_pairs for value in local_pair_ids
            )
        ):
            raise ClassCompleteSemanticError("decode pair IDs must be unique canonical batch <=16 in P window")
        try:
            base_camera = self.base.receiver.render_camera_pairs(local_pair_ids)
            mutated_camera = self.mutated.render_camera_pairs(local_pair_ids)
        except (DirectDescriptionError, ValueError) as exc:
            raise ClassCompleteSemanticError("camera receiver execution failed") from exc
        expected_shape = (
            len(local_pair_ids),
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        if (
            base_camera.dtype != np.uint8
            or mutated_camera.dtype != np.uint8
            or base_camera.shape != expected_shape
            or mutated_camera.shape != expected_shape
        ):
            raise ClassCompleteSemanticError("camera receiver ABI drifted")
        output = base_camera.copy()
        output[:, 1] = mutated_camera[:, 1]
        if not np.array_equal(output[:, 0], base_camera[:, 0]):
            raise ClassCompleteSemanticError("Y1 semantic receiver mutated Y0")

        base_r = np.empty(
            (len(local_pair_ids), 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS),
            dtype=np.uint8,
        )
        output_r = np.empty_like(base_r)
        for local_index in range(len(local_pair_ids)):
            for frame_index in range(2):
                base_r[local_index, frame_index] = exact_resize_round_u8(
                    self.base.operator, base_camera[local_index, frame_index]
                )
                output_r[local_index, frame_index] = exact_resize_round_u8(
                    self.base.operator, output[local_index, frame_index]
                )
        base_semantic = _semantic_cells(self.base.receiver, local_pair_ids)
        mutated_semantic = _semantic_cells(self.mutated, local_pair_ids)
        layer_by_role_base = {row.role: row for row in self.base.receiver.layers}
        layer_by_role_mutated = {row.role: row for row in self.mutated.layers}
        role_changes: list[tuple[str, int]] = []
        for role in REALIZATION_PAINT_ORDER:
            changed = 0
            for pair_id in local_pair_ids:
                old = self.base.receiver._mask_for_layer(layer_by_role_base[role], pair_id, replace_g1_movable=True)
                new = self.mutated._mask_for_layer(layer_by_role_mutated[role], pair_id, replace_g1_movable=True)
                changed += int(np.count_nonzero(old != new))
            role_changes.append((role, changed))

        return ClassCompleteDecodeV1(
            local_pair_ids=local_pair_ids,
            source_pair_ids=tuple(self.base.receiver.predictor.source_pair_start + value for value in local_pair_ids),
            camera_pairs=np.ascontiguousarray(output),
            exact_r_pairs=np.ascontiguousarray(output_r),
            semantic_cells=mutated_semantic,
            base_camera_pairs=np.ascontiguousarray(base_camera),
            base_exact_r_pairs=np.ascontiguousarray(base_r),
            base_semantic_cells=base_semantic,
            changed_camera_values=int(np.count_nonzero(output != base_camera)),
            changed_exact_r_values=int(np.count_nonzero(output_r != base_r)),
            changed_semantic_cells=int(np.count_nonzero(mutated_semantic != base_semantic)),
            changed_role_mask_cells=tuple(role_changes),
        )


def derive_irreducible_learned_quotient(
    target_labels: np.ndarray,
    analytic_labels: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    """Return the exact encoder-only remainder after the analytic semantic span.

    This remainder is a diagnostic coordinate, not an admitted wire type.
    Shipping it densely would defeat the codec; a later learned quotient must
    prove byte value through the same whole-state score arbitration.
    """

    target = np.asarray(target_labels)
    analytic = np.asarray(analytic_labels)
    if (
        target.dtype != np.uint8
        or analytic.dtype not in (np.uint8, np.int16)
        or target.shape != analytic.shape
        or target.ndim != 3
        or target.shape[1:] != (SCORER_HEIGHT, SCORER_WIDTH)
        or np.any(target > 4)
    ):
        raise ClassCompleteSemanticError("learned quotient requires exact n-by-384-by-512 class labels")
    residual = np.ascontiguousarray(target.astype(np.int16) != analytic.astype(np.int16))
    residual.setflags(write=False)
    return residual, int(np.count_nonzero(residual)), _sha256(memoryview(residual).cast("B"))


def encode_class_complete_product(
    semantic_archive: bytes,
    program_payload: bytes,
) -> bytes:
    if (
        type(semantic_archive) is not bytes
        or not semantic_archive
        or type(program_payload) is not bytes
        or not program_payload
    ):
        raise ClassCompleteSemanticError("product requires exact nonempty P and operand bytes")
    if len(semantic_archive) > _MAX_STREAM_BYTES or len(program_payload) > _MAX_STREAM_BYTES:
        raise ClassCompleteSemanticError("product section exceeds byte ceiling")
    return (
        _PRODUCT_HEADER.pack(
            PRODUCT_MAGIC,
            PRODUCT_VERSION,
            len(semantic_archive),
            len(program_payload),
        )
        + semantic_archive
        + program_payload
    )


def parse_class_complete_product(payload: bytes) -> tuple[bytes, bytes]:
    if type(payload) is not bytes or len(payload) < _PRODUCT_HEADER.size:
        raise ClassCompleteSemanticError("class-complete product is truncated/non-byte")
    magic, version, semantic_bytes, operand_bytes = _PRODUCT_HEADER.unpack_from(payload)
    if (
        magic != PRODUCT_MAGIC
        or version != PRODUCT_VERSION
        or semantic_bytes == 0
        or operand_bytes == 0
        or semantic_bytes > _MAX_STREAM_BYTES
        or operand_bytes > _MAX_STREAM_BYTES
        or len(payload) != _PRODUCT_HEADER.size + semantic_bytes + operand_bytes
    ):
        raise ClassCompleteSemanticError("class-complete product header/EOF is invalid")
    cursor = _PRODUCT_HEADER.size
    semantic = payload[cursor : cursor + semantic_bytes]
    operand = payload[cursor + semantic_bytes :]
    if encode_class_complete_product(semantic, operand) != payload:
        raise ClassCompleteSemanticError("product parse/re-encode changed bytes")
    return semantic, operand


@dataclass(frozen=True, slots=True)
class ClassCompleteArchiveBuildV1:
    outer: TaskspaceOuterArchiveBuild
    semantic_archive_bytes: int
    semantic_archive_sha256: str
    operand_bytes: int
    operand_sha256: str
    product_bytes: int
    product_sha256: str


def build_class_complete_archive(
    semantic_archive: bytes,
    program_payload: bytes,
) -> ClassCompleteArchiveBuildV1:
    """Build and strictly parse the exact counted P+operand archive."""

    product = encode_class_complete_product(semantic_archive, program_payload)
    try:
        outer = build_taskspace_outer_archive(
            product,
            max_member_bytes=max(len(product), 1),
        )
        parsed = parse_taskspace_outer_archive(
            outer.selected.archive_bytes,
            expected_encoding=outer.selected.encoding,
            expected_archive_sha256=outer.selected.archive_sha256,
            expected_member_sha256=outer.selected.member_sha256,
            max_member_bytes=max(len(product), 1),
        )
    except TaskspaceOuterArchiveError as exc:
        raise ClassCompleteSemanticError("outer archive build/parse failed") from exc
    semantic_back, operand_back = parse_class_complete_product(parsed.member_bytes)
    if semantic_back != semantic_archive or operand_back != program_payload:
        raise ClassCompleteSemanticError("outer archive parse-back changed product sections")
    return ClassCompleteArchiveBuildV1(
        outer=outer,
        semantic_archive_bytes=len(semantic_archive),
        semantic_archive_sha256=_sha256(semantic_archive),
        operand_bytes=len(program_payload),
        operand_sha256=_sha256(program_payload),
        product_bytes=len(product),
        product_sha256=_sha256(product),
    )


def receive_class_complete_archive(
    archive: bytes,
    *,
    expected_archive_sha256: str,
    expected_semantic_archive_sha256: str,
    expected_program_sha256: str,
    verify_member_effects: bool = True,
) -> ClassCompleteSemanticReceiverV1:
    """Strictly parse the counted archive and construct the executable receiver."""

    archive_sha = _require_sha256(expected_archive_sha256, "expected_archive_sha256")
    if type(archive) is not bytes or _sha256(archive) != archive_sha:
        raise ClassCompleteSemanticError("counted archive exact custody mismatch")
    parsed: ParsedTaskspaceOuterArchive | None = None
    for encoding in ("zip_stored", "zip_deflated"):
        try:
            parsed = parse_taskspace_outer_archive(
                archive,
                expected_encoding=encoding,
                expected_archive_sha256=archive_sha,
                max_member_bytes=_MAX_STREAM_BYTES,
            )
            break
        except TaskspaceOuterArchiveError:
            continue
    if parsed is None:
        raise ClassCompleteSemanticError("counted archive strict parse refused both encodings")
    semantic, operand = parse_class_complete_product(parsed.member_bytes)
    return ClassCompleteSemanticReceiverV1.open(
        semantic,
        operand,
        expected_semantic_archive_sha256=expected_semantic_archive_sha256,
        expected_program_sha256=expected_program_sha256,
        verify_member_effects=verify_member_effects,
    )
