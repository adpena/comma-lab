# SPDX-License-Identifier: MIT
"""Additive receiver grammar for isolated Lane and G2CS1 coordinates.

The sealed V13/V15 carrier grammar intentionally forbids mixing natural
worldsheet productions with post-solve G2CS1 corrections.  RG1 adds that
missing composition as a new outer grammar.  Empty RG1 streams are represented
by *no wrapper at all*, so the inactive compiler returns the input V13/V19C
carrier bytes exactly.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.direct_description_carrier_compose import (
    CarrierComposeReceiverV1,
    LanePeriodicProgramV1,
    _apply_chart_symbols,
    _apply_lane_predictor_programs,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_entropy_priced_member import (
    _zip_stored,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.predictor_upgrade_xi_chart import (
    LaneCoefficientDelta,
    decode_lane_coefficient_deltas,
    encode_lane_coefficient_deltas,
)

ARCHIVE_SCHEMA: Final = "ddm_rg1_joint_receiver_grammar_archive.v1"
RECEIVER_SCHEMA: Final = "ddm_rg1_joint_receiver_grammar_receiver.v1"
MAGIC: Final = "DDRG1"
BASE_MEMBER: Final = "base/v13_v19c_carrier.zip"
LANE_PROGRAM_MEMBER: Final = "production/lane_program_coordinates.rg1lp"
CORRECTION_MEMBER: Final = "correction/lane_chart_symbols.g2cs2"
LANE_PROGRAM_PACKET_MAGIC: Final = b"RG1LP"
LANE_PROGRAM_PACKET_VERSION: Final = 1
_LANE_HEADER: Final = struct.Struct(">5sBHI")
_LANE_ROW: Final = struct.Struct(">BBh")

LANE_FIELDS: Final = (
    "dash_phase_origin_q8",
    "dash_phase_xi_gain_q8",
    "width_bias_q8",
    "width_slope_q12",
)
_FIELD_TO_WIRE: Final = {name: index for index, name in enumerate(LANE_FIELDS)}
_WIRE_TO_FIELD: Final = {value: key for key, value in _FIELD_TO_WIRE.items()}


@dataclass(frozen=True, order=True, slots=True)
class LaneProgramCoordinateV1:
    """One isolated signed quantum in the six-line Lane program."""

    line_index: int
    field: Literal[
        "dash_phase_origin_q8",
        "dash_phase_xi_gain_q8",
        "width_bias_q8",
        "width_slope_q12",
    ]
    signed_quanta: int

    def __post_init__(self) -> None:
        if isinstance(self.line_index, bool) or not 0 <= self.line_index < 6:
            raise DirectDescriptionError("RG1 Lane coordinate line index must be in [0,6)")
        if self.field not in _FIELD_TO_WIRE:
            raise DirectDescriptionError("RG1 Lane coordinate field is outside the sealed vocabulary")
        if (
            isinstance(self.signed_quanta, bool)
            or not isinstance(self.signed_quanta, int)
            or not -32768 <= self.signed_quanta <= 32767
            or self.signed_quanta == 0
        ):
            raise DirectDescriptionError("RG1 Lane coordinate must carry a nonzero int16 quantum")

    @property
    def actuator_id(self) -> str:
        return f"j2.lane.line{self.line_index}.{self.field}"


def encode_lane_program_coordinates(rows: Sequence[LaneProgramCoordinateV1]) -> bytes:
    """Encode canonical, address-unique Lane program coordinates."""

    ordered = tuple(rows)
    if not ordered:
        return b""
    keys = [(row.line_index, _FIELD_TO_WIRE[row.field]) for row in ordered]
    if len(ordered) > 24 or keys != sorted(set(keys)):
        raise DirectDescriptionError("RG1 Lane coordinates must be sorted, unique, and at most 24")
    body = b"".join(
        _LANE_ROW.pack(row.line_index, _FIELD_TO_WIRE[row.field], row.signed_quanta)
        for row in ordered
    )
    return _LANE_HEADER.pack(
        LANE_PROGRAM_PACKET_MAGIC,
        LANE_PROGRAM_PACKET_VERSION,
        len(ordered),
        zlib.crc32(body) & 0xFFFFFFFF,
    ) + body


def decode_lane_program_coordinates(payload: bytes) -> tuple[LaneProgramCoordinateV1, ...]:
    """Fail closed on length, CRC, vocabulary, ordering, and parse-back."""

    if not isinstance(payload, bytes):
        raise DirectDescriptionError("RG1 Lane program payload must be bytes")
    if not payload:
        return ()
    if len(payload) < _LANE_HEADER.size:
        raise DirectDescriptionError("RG1 Lane program payload is truncated")
    magic, version, count, checksum = _LANE_HEADER.unpack_from(payload)
    expected = _LANE_HEADER.size + count * _LANE_ROW.size
    if (
        magic != LANE_PROGRAM_PACKET_MAGIC
        or version != LANE_PROGRAM_PACKET_VERSION
        or not 1 <= count <= 24
        or len(payload) != expected
    ):
        raise DirectDescriptionError("RG1 Lane program header or length is invalid")
    body = payload[_LANE_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != checksum:
        raise DirectDescriptionError("RG1 Lane program CRC differs")
    try:
        rows = tuple(
            LaneProgramCoordinateV1(
                line_index=line_index,
                field=_WIRE_TO_FIELD[field_id],
                signed_quanta=signed_quanta,
            )
            for line_index, field_id, signed_quanta in (
                _LANE_ROW.unpack_from(body, index * _LANE_ROW.size)
                for index in range(count)
            )
        )
    except KeyError as exc:
        raise DirectDescriptionError("RG1 Lane program field tag is unknown") from exc
    if encode_lane_program_coordinates(rows) != payload:
        raise DirectDescriptionError("RG1 Lane program is not canonical on parse-back")
    return rows


def _programs(rows: Sequence[LaneProgramCoordinateV1]) -> tuple[LanePeriodicProgramV1, ...]:
    by_line: dict[int, dict[str, int]] = {}
    for row in rows:
        by_line.setdefault(row.line_index, {})[row.field] = row.signed_quanta
    return tuple(
        LanePeriodicProgramV1(
            line_index=line_index,
            birth_pair=0,
            death_pair_exclusive=600,
            dash_phase_origin_delta_q8=values.get("dash_phase_origin_q8", 0),
            dash_phase_xi_gain_q8=values.get("dash_phase_xi_gain_q8", 0),
            width_bias_q8=values.get("width_bias_q8", 0),
            width_slope_q12=values.get("width_slope_q12", 0),
        )
        for line_index, values in sorted(by_line.items())
    )


def project_polygon_center(
    center: int,
    relative_coordinates: Sequence[int],
    extent: int,
) -> int:
    """Project a polygon center onto the exact integer interval that keeps it in-grid."""

    if (
        isinstance(center, bool)
        or not isinstance(center, int)
        or isinstance(extent, bool)
        or not isinstance(extent, int)
        or extent <= 0
        or not relative_coordinates
        or any(isinstance(value, bool) or not isinstance(value, int) for value in relative_coordinates)
    ):
        raise DirectDescriptionError("RG1 polygon projection requires integer geometry")
    lower = -min(relative_coordinates)
    upper = extent - 1 - max(relative_coordinates)
    if lower > upper:
        raise DirectDescriptionError("RG1 polygon cannot fit inside the scorer extent")
    return min(max(center, lower), upper)


def _typed_tag(stream_type: StreamType, layer: LayerHome, counted_bytes: int) -> dict[str, Any]:
    return TypedStreamTag(
        type=stream_type,
        layer_home=layer,
        evaluate_py_recursion_level_cited=(
            f"{layer.value} counted RG1 stream -> L3_raster -> L4_scorer_feature"
        ),
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    ).to_dict()


def _manifest(
    base_archive: bytes,
    lane_payload: bytes,
    correction_payload: bytes,
) -> dict[str, Any]:
    return {
        "schema": ARCHIVE_SCHEMA,
        "magic": MAGIC,
        "base": {
            "member": BASE_MEMBER,
            "bytes": len(base_archive),
            "sha256": hashlib.sha256(base_archive).hexdigest(),
            "sealed_v13_v19c_mutated": False,
            "typed_stream_tag": _typed_tag(StreamType.SKELETON, LayerHome.L1_PROGRAM, len(base_archive)),
        },
        "worldsheet_productions": {
            "member": LANE_PROGRAM_MEMBER if lane_payload else None,
            "bytes": len(lane_payload),
            "sha256": hashlib.sha256(lane_payload).hexdigest(),
            "coordinate_count": len(decode_lane_program_coordinates(lane_payload)),
            "coordinate_vocabulary": [
                f"j2.lane.line{line}.{field}" for line in range(6) for field in LANE_FIELDS
            ],
            "typed_stream_tag": _typed_tag(StreamType.SKELETON, LayerHome.L1_PROGRAM, len(lane_payload)),
        },
        "post_solve_corrections": {
            "member": CORRECTION_MEMBER if correction_payload else None,
            "bytes": len(correction_payload),
            "sha256": hashlib.sha256(correction_payload).hexdigest(),
            "symbol_count": len(decode_lane_coefficient_deltas(correction_payload)),
            "typed_stream_tag": _typed_tag(StreamType.RESIDUAL, LayerHome.L2_CHART, len(correction_payload)),
        },
        "composition_order": [
            "parse sealed V13/V19C worldsheet base",
            "apply counted Lane program coordinates",
            "apply counted post-solve G2CS2 chart corrections",
            "execute inherited region-coherent raster and exact R",
        ],
        "five_type_compatible": [member.value for member in StreamType],
        "empty_stream_identity_policy": "no RG1 wrapper emitted",
        "pixel_coordinate_or_rgb_patch_present": False,
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
    }


def compile_rg1_receiver_grammar(
    base_archive: bytes,
    *,
    lane_coordinates: Sequence[LaneProgramCoordinateV1] = (),
    corrections: Sequence[LaneCoefficientDelta] = (),
) -> bytes:
    """Compile RG1, returning the sealed base bytes for the inactive default."""

    if not isinstance(base_archive, bytes) or not base_archive:
        raise DirectDescriptionError("RG1 requires nonempty sealed base carrier bytes")
    lane_payload = encode_lane_program_coordinates(tuple(lane_coordinates))
    correction_payload = encode_lane_coefficient_deltas(tuple(corrections))
    if not lane_payload and not correction_payload:
        return base_archive
    # Strictly admit the base before wrapping it.
    receive_carrier_compose_archive(base_archive, verify_member_effects=False)
    manifest = _manifest(base_archive, lane_payload, correction_payload)
    members = {
        "manifest.json": rfc8785_canonicalize(manifest),
        BASE_MEMBER: base_archive,
    }
    if lane_payload:
        members[LANE_PROGRAM_MEMBER] = lane_payload
    if correction_payload:
        members[CORRECTION_MEMBER] = correction_payload
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("RG1 compiler is nondeterministic")
    parsed = parse_rg1_receiver_grammar(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("RG1 parse/re-encode changed bytes")
    return first


def parse_rg1_receiver_grammar(archive: bytes) -> dict[str, bytes]:
    """Strictly parse a nonempty RG1 wrapper."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if [row.filename for row in infos[:2]] != ["manifest.json", BASE_MEMBER]:
                raise DirectDescriptionError("RG1 member prefix is invalid")
            if not 3 <= len(infos) <= 4:
                raise DirectDescriptionError("RG1 wrapper must carry at least one extension stream")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("RG1 ZIP metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
    except DirectDescriptionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DirectDescriptionError("RG1 ZIP is malformed") from exc
    try:
        manifest = json.loads(members["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("RG1 manifest is malformed") from exc
    lane_payload = members.get(LANE_PROGRAM_MEMBER, b"")
    correction_payload = members.get(CORRECTION_MEMBER, b"")
    expected_members = {
        "manifest.json",
        BASE_MEMBER,
        *([LANE_PROGRAM_MEMBER] if lane_payload else []),
        *([CORRECTION_MEMBER] if correction_payload else []),
    }
    expected_manifest = _manifest(members[BASE_MEMBER], lane_payload, correction_payload)
    if (
        set(members) != expected_members
        or (not lane_payload and not correction_payload)
        or manifest != expected_manifest
        or members["manifest.json"] != rfc8785_canonicalize(manifest)
    ):
        raise DirectDescriptionError("RG1 manifest or member binding differs")
    decode_lane_program_coordinates(lane_payload)
    decode_lane_coefficient_deltas(correction_payload)
    return members


def receive_rg1_receiver_grammar(
    archive: bytes,
    *,
    verify_member_effects: bool = False,
) -> CarrierComposeReceiverV1:
    """Receive either a sealed base or a nonempty RG1 extension."""

    try:
        members = parse_rg1_receiver_grammar(archive)
    except DirectDescriptionError:
        # A valid legacy carrier is the byte-identical inactive RG1 form.
        return receive_carrier_compose_archive(
            archive,
            verify_member_effects=verify_member_effects,
        )
    base_archive = members[BASE_MEMBER]
    base = receive_carrier_compose_archive(base_archive, verify_member_effects=False)
    coordinates = decode_lane_program_coordinates(members.get(LANE_PROGRAM_MEMBER, b""))
    corrections = decode_lane_coefficient_deltas(members.get(CORRECTION_MEMBER, b""))
    programs = _programs(coordinates)
    layers = _apply_lane_predictor_programs(
        base.layers,
        programs,
        (),
        pose6_codes=base.predictor.pose6_codes,
        source_pair_start=base.predictor.source_pair_start,
    )
    layers = _apply_chart_symbols(layers, corrections, coefficient_limit=4)
    receiver = replace(
        base,
        archive=archive,
        layers=layers,
        symbols=corrections,
        lane_programs=programs,
        custody={
            "schema": RECEIVER_SCHEMA,
            "archive_bytes": len(archive),
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "base_archive_sha256": hashlib.sha256(base_archive).hexdigest(),
            "lane_coordinate_count": len(coordinates),
            "lane_coordinate_ids": [row.actuator_id for row in coordinates],
            "correction_symbol_count": len(corrections),
            "composition_order_enforced": True,
            "typed_stream_tags_validated": True,
            "sealed_v13_v19c_mutated": False,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        },
    )
    if verify_member_effects:
        probe_ids = sorted(
            {
                *(range(600) if coordinates else ()),
                *(row.pair_index for row in corrections),
            }
        )
        if not probe_ids:
            raise DirectDescriptionError("RG1 wrapper has no effective coordinate")
        if all(
            (
                base.render_camera_pairs((pair_id,)) == receiver.render_camera_pairs((pair_id,))
            ).all()
            for pair_id in probe_ids
        ):
            raise DirectDescriptionError("RG1 extension is a receiver-output no-op")
    return receiver


__all__ = [
    "ARCHIVE_SCHEMA",
    "BASE_MEMBER",
    "CORRECTION_MEMBER",
    "LANE_FIELDS",
    "LANE_PROGRAM_MEMBER",
    "LaneProgramCoordinateV1",
    "compile_rg1_receiver_grammar",
    "decode_lane_program_coordinates",
    "encode_lane_program_coordinates",
    "parse_rg1_receiver_grammar",
    "project_polygon_center",
    "receive_rg1_receiver_grammar",
]
