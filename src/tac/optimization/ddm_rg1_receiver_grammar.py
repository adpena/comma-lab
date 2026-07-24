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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
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
ARCHIVE_SCHEMA_RG2: Final = "ddm_rg2_joint_receiver_grammar_archive.v2"
RECEIVER_SCHEMA_RG2: Final = "ddm_rg2_joint_receiver_grammar_receiver.v2"
MAGIC_RG2: Final = "DDRG2"
ARCHIVE_SCHEMA_RG3: Final = "ddm_rg3_joint_receiver_grammar_archive.v3"
RECEIVER_SCHEMA_RG3: Final = "ddm_rg3_joint_receiver_grammar_receiver.v3"
MAGIC_RG3: Final = "DDRG3"
BASE_MEMBER: Final = "base/v13_v19c_carrier.zip"
LANE_PROGRAM_MEMBER: Final = "production/lane_program_coordinates.rg1lp"
CORRECTION_MEMBER: Final = "correction/lane_chart_symbols.g2cs2"
SKELETON_AMPLITUDE_MEMBER: Final = "production/skeleton_amplitude_coordinates.rg2sa"
RG3_RESIDUAL_MEMBER: Final = "production/residual_family_coordinates.rg3rf"
LANE_PROGRAM_PACKET_MAGIC: Final = b"RG1LP"
LANE_PROGRAM_PACKET_VERSION: Final = 1
_LANE_HEADER: Final = struct.Struct(">5sBHI")
_LANE_ROW: Final = struct.Struct(">BBh")
SKELETON_AMPLITUDE_PACKET_MAGIC: Final = b"RG2SA"
SKELETON_AMPLITUDE_PACKET_VERSION: Final = 1
_AMPLITUDE_HEADER: Final = struct.Struct(">5sBHI")
_AMPLITUDE_ROW: Final = struct.Struct(">HBBBBBb")
_AMPLITUDE_FAMILY_TO_WIRE: Final = {
    "EVENT_LOCAL_BOUNDARY": 0,
    "PER_STRATUM_ROW_BAND": 1,
}
_AMPLITUDE_WIRE_TO_FAMILY: Final = {value: key for key, value in _AMPLITUDE_FAMILY_TO_WIRE.items()}
RG3_RESIDUAL_PACKET_MAGIC: Final = b"RG3RF"
RG3_RESIDUAL_PACKET_VERSION: Final = 1
_RG3_HEADER: Final = struct.Struct(">5sBHI")
_RG3_ROW: Final = struct.Struct(">HBBBBBBb")
_RG3_FAMILY_TO_WIRE: Final = {
    "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION": 0,
    "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK": 1,
    "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK": 2,
}
_RG3_WIRE_TO_FAMILY: Final = {value: key for key, value in _RG3_FAMILY_TO_WIRE.items()}
_TEMPORAL_TO_WIRE: Final = {"STATIC_IN_IMAGE": 0, "TRANSIENT": 1}
_WIRE_TO_TEMPORAL: Final = {value: key for key, value in _TEMPORAL_TO_WIRE.items()}
_CLASS_TO_ROLE: Final = {value: key for key, value in ROLE_CLASS_IDS.items()}
_ROW_BAND_HEIGHT: Final = 64
_ROW_BAND_COUNT: Final = 6
_FINE_BAND_HEIGHT: Final = 16
_FINE_BAND_COUNT: Final = 4

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


@dataclass(frozen=True, order=True, slots=True)
class SkeletonAmplitudeCoordinateV1:
    """One isolated RG2 amplitude quantum at a typed pair/stratum address."""

    pair_index: int
    class_a: int
    class_b: int
    family: Literal["EVENT_LOCAL_BOUNDARY", "PER_STRATUM_ROW_BAND"]
    temporal_class: Literal["STATIC_IN_IMAGE", "TRANSIENT"]
    row_band: int
    signed_quanta: Literal[-1, 1]

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("RG2 SKELETON pair index must be in [0,600)")
        if isinstance(self.class_a, bool) or isinstance(self.class_b, bool) or not 0 <= self.class_a < self.class_b < 5:
            raise DirectDescriptionError("RG2 SKELETON class address must be canonical 0<=a<b<5")
        if self.family not in _AMPLITUDE_FAMILY_TO_WIRE:
            raise DirectDescriptionError("RG2 SKELETON production family is unknown")
        if self.temporal_class not in _TEMPORAL_TO_WIRE:
            raise DirectDescriptionError("RG2 SKELETON temporal class is unknown")
        if isinstance(self.row_band, bool) or not 0 <= self.row_band < _ROW_BAND_COUNT:
            raise DirectDescriptionError("RG2 SKELETON row band must be in [0,6)")
        if self.signed_quanta not in (-1, 1):
            raise DirectDescriptionError("RG2 SKELETON coordinate must be exactly one signed quantum")

    @property
    def stratum(self) -> str:
        return "boundary" if self.family == "EVENT_LOCAL_BOUNDARY" else "cell"

    @property
    def actuator_id(self) -> str:
        temporal = self.temporal_class.lower()
        return (
            f"rg2.skeleton.pair{self.pair_index:03d}.class{self.class_a}_{self.class_b}."
            f"{self.stratum}.{temporal}.band{self.row_band:02d}"
        )


@dataclass(frozen=True, order=True, slots=True)
class RG3ResidualCoordinateV1:
    """One typed RG3 class-birth or refined amplitude-codebook symbol."""

    pair_index: int
    class_a: int
    class_b: int
    family: Literal[
        "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION",
        "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK",
        "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK",
    ]
    temporal_class: Literal["STATIC_IN_IMAGE", "TRANSIENT"]
    row_band: int
    fine_band: int
    signed_quanta: Literal[-2, -1, 1, 2]

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("RG3 SKELETON pair index must be in [0,600)")
        if isinstance(self.class_a, bool) or isinstance(self.class_b, bool) or not 0 <= self.class_a < self.class_b < 5:
            raise DirectDescriptionError("RG3 SKELETON class address must be canonical 0<=a<b<5")
        if self.family not in _RG3_FAMILY_TO_WIRE:
            raise DirectDescriptionError("RG3 SKELETON residual family is unknown")
        if self.temporal_class not in _TEMPORAL_TO_WIRE:
            raise DirectDescriptionError("RG3 SKELETON temporal class is unknown")
        if isinstance(self.row_band, bool) or not 0 <= self.row_band < _ROW_BAND_COUNT:
            raise DirectDescriptionError("RG3 SKELETON row band must be in [0,6)")
        if isinstance(self.fine_band, bool) or not 0 <= self.fine_band < _FINE_BAND_COUNT:
            raise DirectDescriptionError("RG3 SKELETON fine band must be in [0,4)")
        if self.signed_quanta not in (-2, -1, 1, 2):
            raise DirectDescriptionError("RG3 SKELETON codebook quantum must be one of {-2,-1,1,2}")
        if (
            self.family == "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION"
            and abs(self.signed_quanta) != 1
        ):
            raise DirectDescriptionError("RG3 class-birth seed is exactly one signed quantum")

    @property
    def stratum(self) -> str:
        if self.family == "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK":
            return "cell"
        return "boundary"

    @property
    def actuator_id(self) -> str:
        family_slug = {
            "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION": "class_birth",
            "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK": "finer_event",
            "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK": "fisher_stratum",
        }[self.family]
        return (
            f"rg3.{family_slug}.pair{self.pair_index:03d}.class{self.class_a}_{self.class_b}."
            f"{self.stratum}.{self.temporal_class.lower()}.band{self.row_band:02d}."
            f"fine{self.fine_band:02d}.q{self.signed_quanta:+d}"
        )


def encode_rg3_residual_coordinates(rows: Sequence[RG3ResidualCoordinateV1]) -> bytes:
    """Encode canonical, counted, sorted-unique RG3 residual coordinates."""

    ordered = tuple(rows)
    if not ordered:
        return b""
    keys = [
        (
            row.pair_index,
            row.class_a,
            row.class_b,
            _RG3_FAMILY_TO_WIRE[row.family],
            _TEMPORAL_TO_WIRE[row.temporal_class],
            row.row_band,
            row.fine_band,
        )
        for row in ordered
    ]
    if len(ordered) > 36 or keys != sorted(set(keys)):
        raise DirectDescriptionError("RG3 SKELETON coordinates must be sorted, unique, and at most 36")
    body = b"".join(
        _RG3_ROW.pack(
            row.pair_index,
            row.class_a,
            row.class_b,
            _RG3_FAMILY_TO_WIRE[row.family],
            _TEMPORAL_TO_WIRE[row.temporal_class],
            row.row_band,
            row.fine_band,
            row.signed_quanta,
        )
        for row in ordered
    )
    return (
        _RG3_HEADER.pack(
            RG3_RESIDUAL_PACKET_MAGIC,
            RG3_RESIDUAL_PACKET_VERSION,
            len(ordered),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        + body
    )


def decode_rg3_residual_coordinates(payload: bytes) -> tuple[RG3ResidualCoordinateV1, ...]:
    """Fail closed on RG3 length, CRC, enums, address, and canonical parse-back."""

    if not isinstance(payload, bytes):
        raise DirectDescriptionError("RG3 SKELETON residual payload must be bytes")
    if not payload:
        return ()
    if len(payload) < _RG3_HEADER.size:
        raise DirectDescriptionError("RG3 SKELETON residual payload is truncated")
    magic, version, count, checksum = _RG3_HEADER.unpack_from(payload)
    expected = _RG3_HEADER.size + count * _RG3_ROW.size
    if (
        magic != RG3_RESIDUAL_PACKET_MAGIC
        or version != RG3_RESIDUAL_PACKET_VERSION
        or not 1 <= count <= 36
        or len(payload) != expected
    ):
        raise DirectDescriptionError("RG3 SKELETON residual header or length is invalid")
    body = payload[_RG3_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != checksum:
        raise DirectDescriptionError("RG3 SKELETON residual CRC differs")
    try:
        rows = tuple(
            RG3ResidualCoordinateV1(
                pair_index=pair_index,
                class_a=class_a,
                class_b=class_b,
                family=_RG3_WIRE_TO_FAMILY[family_id],
                temporal_class=_WIRE_TO_TEMPORAL[temporal_id],
                row_band=row_band,
                fine_band=fine_band,
                signed_quanta=signed_quanta,
            )
            for (
                pair_index,
                class_a,
                class_b,
                family_id,
                temporal_id,
                row_band,
                fine_band,
                signed_quanta,
            ) in (_RG3_ROW.unpack_from(body, index * _RG3_ROW.size) for index in range(count))
        )
    except KeyError as exc:
        raise DirectDescriptionError("RG3 SKELETON residual enum tag is unknown") from exc
    if encode_rg3_residual_coordinates(rows) != payload:
        raise DirectDescriptionError("RG3 SKELETON residual packet is not canonical on parse-back")
    return rows


def encode_skeleton_amplitude_coordinates(
    rows: Sequence[SkeletonAmplitudeCoordinateV1],
) -> bytes:
    """Encode canonical, counted, address-unique RG2 amplitude coordinates."""

    ordered = tuple(rows)
    if not ordered:
        return b""
    keys = [
        (
            row.pair_index,
            row.class_a,
            row.class_b,
            _AMPLITUDE_FAMILY_TO_WIRE[row.family],
            _TEMPORAL_TO_WIRE[row.temporal_class],
            row.row_band,
        )
        for row in ordered
    ]
    if len(ordered) > 64 or keys != sorted(set(keys)):
        raise DirectDescriptionError("RG2 SKELETON coordinates must be sorted, unique, and at most 64")
    body = b"".join(
        _AMPLITUDE_ROW.pack(
            row.pair_index,
            row.class_a,
            row.class_b,
            _AMPLITUDE_FAMILY_TO_WIRE[row.family],
            _TEMPORAL_TO_WIRE[row.temporal_class],
            row.row_band,
            row.signed_quanta,
        )
        for row in ordered
    )
    return (
        _AMPLITUDE_HEADER.pack(
            SKELETON_AMPLITUDE_PACKET_MAGIC,
            SKELETON_AMPLITUDE_PACKET_VERSION,
            len(ordered),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        + body
    )


def decode_skeleton_amplitude_coordinates(
    payload: bytes,
) -> tuple[SkeletonAmplitudeCoordinateV1, ...]:
    """Fail closed on RG2 length, CRC, enum vocabulary, address, and parse-back."""

    if not isinstance(payload, bytes):
        raise DirectDescriptionError("RG2 SKELETON amplitude payload must be bytes")
    if not payload:
        return ()
    if len(payload) < _AMPLITUDE_HEADER.size:
        raise DirectDescriptionError("RG2 SKELETON amplitude payload is truncated")
    magic, version, count, checksum = _AMPLITUDE_HEADER.unpack_from(payload)
    expected = _AMPLITUDE_HEADER.size + count * _AMPLITUDE_ROW.size
    if (
        magic != SKELETON_AMPLITUDE_PACKET_MAGIC
        or version != SKELETON_AMPLITUDE_PACKET_VERSION
        or not 1 <= count <= 64
        or len(payload) != expected
    ):
        raise DirectDescriptionError("RG2 SKELETON amplitude header or length is invalid")
    body = payload[_AMPLITUDE_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != checksum:
        raise DirectDescriptionError("RG2 SKELETON amplitude CRC differs")
    try:
        rows = tuple(
            SkeletonAmplitudeCoordinateV1(
                pair_index=pair_index,
                class_a=class_a,
                class_b=class_b,
                family=_AMPLITUDE_WIRE_TO_FAMILY[family_id],
                temporal_class=_WIRE_TO_TEMPORAL[temporal_id],
                row_band=row_band,
                signed_quanta=signed_quanta,
            )
            for (
                pair_index,
                class_a,
                class_b,
                family_id,
                temporal_id,
                row_band,
                signed_quanta,
            ) in (_AMPLITUDE_ROW.unpack_from(body, index * _AMPLITUDE_ROW.size) for index in range(count))
        )
    except KeyError as exc:
        raise DirectDescriptionError("RG2 SKELETON amplitude enum tag is unknown") from exc
    if encode_skeleton_amplitude_coordinates(rows) != payload:
        raise DirectDescriptionError("RG2 SKELETON amplitude packet is not canonical on parse-back")
    return rows


def encode_lane_program_coordinates(rows: Sequence[LaneProgramCoordinateV1]) -> bytes:
    """Encode canonical, address-unique Lane program coordinates."""

    ordered = tuple(rows)
    if not ordered:
        return b""
    keys = [(row.line_index, _FIELD_TO_WIRE[row.field]) for row in ordered]
    if len(ordered) > 24 or keys != sorted(set(keys)):
        raise DirectDescriptionError("RG1 Lane coordinates must be sorted, unique, and at most 24")
    body = b"".join(_LANE_ROW.pack(row.line_index, _FIELD_TO_WIRE[row.field], row.signed_quanta) for row in ordered)
    return (
        _LANE_HEADER.pack(
            LANE_PROGRAM_PACKET_MAGIC,
            LANE_PROGRAM_PACKET_VERSION,
            len(ordered),
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        + body
    )


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
                _LANE_ROW.unpack_from(body, index * _LANE_ROW.size) for index in range(count)
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


def _typed_tag(
    stream_type: StreamType,
    layer: LayerHome,
    counted_bytes: int,
    *,
    grammar: str = "RG1",
) -> dict[str, Any]:
    return TypedStreamTag(
        type=stream_type,
        layer_home=layer,
        evaluate_py_recursion_level_cited=(f"{layer.value} counted {grammar} stream -> L3_raster -> L4_scorer_feature"),
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    ).to_dict()


def _manifest(
    base_archive: bytes,
    lane_payload: bytes,
    correction_payload: bytes,
    amplitude_payload: bytes = b"",
    rg3_payload: bytes = b"",
) -> dict[str, Any]:
    grammar = "RG3" if rg3_payload else "RG2" if amplitude_payload else "RG1"
    manifest = {
        "schema": (
            ARCHIVE_SCHEMA_RG3
            if rg3_payload
            else ARCHIVE_SCHEMA_RG2
            if amplitude_payload
            else ARCHIVE_SCHEMA
        ),
        "magic": MAGIC_RG3 if rg3_payload else MAGIC_RG2 if amplitude_payload else MAGIC,
        "base": {
            "member": BASE_MEMBER,
            "bytes": len(base_archive),
            "sha256": hashlib.sha256(base_archive).hexdigest(),
            "sealed_v13_v19c_mutated": False,
            "typed_stream_tag": _typed_tag(
                StreamType.SKELETON,
                LayerHome.L1_PROGRAM,
                len(base_archive),
                grammar=grammar,
            ),
        },
        "worldsheet_productions": {
            "member": LANE_PROGRAM_MEMBER if lane_payload else None,
            "bytes": len(lane_payload),
            "sha256": hashlib.sha256(lane_payload).hexdigest(),
            "coordinate_count": len(decode_lane_program_coordinates(lane_payload)),
            "coordinate_vocabulary": [f"j2.lane.line{line}.{field}" for line in range(6) for field in LANE_FIELDS],
            "typed_stream_tag": _typed_tag(
                StreamType.SKELETON,
                LayerHome.L1_PROGRAM,
                len(lane_payload),
                grammar=grammar,
            ),
        },
        "post_solve_corrections": {
            "member": CORRECTION_MEMBER if correction_payload else None,
            "bytes": len(correction_payload),
            "sha256": hashlib.sha256(correction_payload).hexdigest(),
            "symbol_count": len(decode_lane_coefficient_deltas(correction_payload)),
            "typed_stream_tag": _typed_tag(
                StreamType.RESIDUAL,
                LayerHome.L2_CHART,
                len(correction_payload),
                grammar=grammar,
            ),
        },
        "composition_order": [
            "parse sealed V13/V19C worldsheet base",
            "apply counted Lane program coordinates",
            "apply counted post-solve G2CS2 chart corrections",
            "execute inherited region-coherent raster and exact R",
        ],
        "five_type_compatible": [member.value for member in StreamType],
        "empty_stream_identity_policy": f"no {grammar} wrapper emitted",
        "pixel_coordinate_or_rgb_patch_present": False,
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
    }
    if amplitude_payload:
        manifest["skeleton_amplitude_productions"] = {
            "member": SKELETON_AMPLITUDE_MEMBER,
            "bytes": len(amplitude_payload),
            "sha256": hashlib.sha256(amplitude_payload).hexdigest(),
            "coordinate_count": len(decode_skeleton_amplitude_coordinates(amplitude_payload)),
            "addressing": (
                "pair_index x ordered_class_pair x boundary_or_cell_family x temporal_class x receiver_derived_row_band"
            ),
            "typed_stream_tag": _typed_tag(
                StreamType.SKELETON,
                LayerHome.L3_RASTER,
                len(amplitude_payload),
                grammar=grammar,
            ),
        }
        manifest["composition_order"] = [
            "parse sealed V13/V19C worldsheet base",
            "apply counted Lane program coordinates",
            "apply counted post-solve G2CS2 chart corrections",
            "execute inherited region-coherent raster to the camera surface",
            "apply counted receiver-derived SKELETON amplitude masks",
            "execute evaluator-owned exact R",
        ]
    if rg3_payload:
        manifest["rg3_residual_family_productions"] = {
            "member": RG3_RESIDUAL_MEMBER,
            "bytes": len(rg3_payload),
            "sha256": hashlib.sha256(rg3_payload).hexdigest(),
            "coordinate_count": len(decode_rg3_residual_coordinates(rg3_payload)),
            "addressing": (
                "pair_index x ordered_class_pair x derived_family x temporal_class "
                "x receiver_row_band x event_local_fine_band x signed_codebook_quantum"
            ),
            "fisher_source_policy": (
                "measured top1-top2 margin selects only the counted fine-band symbol offline; "
                "no scorer label, logit, margin map, or ground-truth field ships"
            ),
            "typed_stream_tag": _typed_tag(
                StreamType.SKELETON,
                LayerHome.L3_RASTER,
                len(rg3_payload),
                grammar=grammar,
            ),
        }
        manifest["composition_order"] = [
            "parse sealed V13/V19C worldsheet base",
            "apply counted Lane program coordinates",
            "apply counted post-solve G2CS2 chart corrections",
            "execute inherited region-coherent raster to the camera surface",
            "apply optional counted RG2 SKELETON amplitude masks",
            "apply counted RG3 class-birth and refined SKELETON codebook masks",
            "execute evaluator-owned exact R",
        ]
    return manifest


def _compile_receiver_grammar(
    base_archive: bytes,
    *,
    lane_coordinates: Sequence[LaneProgramCoordinateV1],
    corrections: Sequence[LaneCoefficientDelta],
    skeleton_amplitudes: Sequence[SkeletonAmplitudeCoordinateV1],
    rg3_residuals: Sequence[RG3ResidualCoordinateV1],
) -> bytes:
    if not isinstance(base_archive, bytes) or not base_archive:
        raise DirectDescriptionError("RG1/RG2 requires nonempty sealed base carrier bytes")
    lane_payload = encode_lane_program_coordinates(tuple(lane_coordinates))
    correction_payload = encode_lane_coefficient_deltas(tuple(corrections))
    amplitude_payload = encode_skeleton_amplitude_coordinates(tuple(skeleton_amplitudes))
    rg3_payload = encode_rg3_residual_coordinates(tuple(rg3_residuals))
    if not lane_payload and not correction_payload and not amplitude_payload and not rg3_payload:
        return base_archive
    receive_carrier_compose_archive(base_archive, verify_member_effects=False)
    manifest = _manifest(
        base_archive,
        lane_payload,
        correction_payload,
        amplitude_payload,
        rg3_payload,
    )
    members = {
        "manifest.json": rfc8785_canonicalize(manifest),
        BASE_MEMBER: base_archive,
    }
    if lane_payload:
        members[LANE_PROGRAM_MEMBER] = lane_payload
    if amplitude_payload:
        members[SKELETON_AMPLITUDE_MEMBER] = amplitude_payload
    if rg3_payload:
        members[RG3_RESIDUAL_MEMBER] = rg3_payload
    if correction_payload:
        members[CORRECTION_MEMBER] = correction_payload
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("RG1/RG2 compiler is nondeterministic")
    parsed = parse_rg1_receiver_grammar(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("RG1/RG2 parse/re-encode changed bytes")
    return first


def compile_rg1_receiver_grammar(
    base_archive: bytes,
    *,
    lane_coordinates: Sequence[LaneProgramCoordinateV1] = (),
    corrections: Sequence[LaneCoefficientDelta] = (),
) -> bytes:
    """Compile RG1, returning the sealed base bytes for the inactive default."""

    return _compile_receiver_grammar(
        base_archive,
        lane_coordinates=lane_coordinates,
        corrections=corrections,
        skeleton_amplitudes=(),
        rg3_residuals=(),
    )


def compile_rg2_receiver_grammar(
    base_archive: bytes,
    *,
    lane_coordinates: Sequence[LaneProgramCoordinateV1] = (),
    skeleton_amplitudes: Sequence[SkeletonAmplitudeCoordinateV1] = (),
    corrections: Sequence[LaneCoefficientDelta] = (),
) -> bytes:
    """Compile the additive RG2 version of the same RG1 outer grammar."""

    return _compile_receiver_grammar(
        base_archive,
        lane_coordinates=lane_coordinates,
        corrections=corrections,
        skeleton_amplitudes=skeleton_amplitudes,
        rg3_residuals=(),
    )


def compile_rg3_receiver_grammar(
    base_archive: bytes,
    *,
    lane_coordinates: Sequence[LaneProgramCoordinateV1] = (),
    skeleton_amplitudes: Sequence[SkeletonAmplitudeCoordinateV1] = (),
    rg3_residuals: Sequence[RG3ResidualCoordinateV1] = (),
    corrections: Sequence[LaneCoefficientDelta] = (),
) -> bytes:
    """Compile additive RG3 around the same sealed RG1/RG2 base."""

    return _compile_receiver_grammar(
        base_archive,
        lane_coordinates=lane_coordinates,
        corrections=corrections,
        skeleton_amplitudes=skeleton_amplitudes,
        rg3_residuals=rg3_residuals,
    )


def parse_rg1_receiver_grammar(archive: bytes) -> dict[str, bytes]:
    """Strictly parse a nonempty RG1 wrapper."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if [row.filename for row in infos[:2]] != ["manifest.json", BASE_MEMBER]:
                raise DirectDescriptionError("RG1 member prefix is invalid")
            if not 3 <= len(infos) <= 6:
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
    amplitude_payload = members.get(SKELETON_AMPLITUDE_MEMBER, b"")
    rg3_payload = members.get(RG3_RESIDUAL_MEMBER, b"")
    correction_payload = members.get(CORRECTION_MEMBER, b"")
    expected_members = {
        "manifest.json",
        BASE_MEMBER,
        *([LANE_PROGRAM_MEMBER] if lane_payload else []),
        *([SKELETON_AMPLITUDE_MEMBER] if amplitude_payload else []),
        *([RG3_RESIDUAL_MEMBER] if rg3_payload else []),
        *([CORRECTION_MEMBER] if correction_payload else []),
    }
    expected_manifest = _manifest(
        members[BASE_MEMBER],
        lane_payload,
        correction_payload,
        amplitude_payload,
        rg3_payload,
    )
    if (
        set(members) != expected_members
        or (not lane_payload and not amplitude_payload and not rg3_payload and not correction_payload)
        or manifest != expected_manifest
        or members["manifest.json"] != rfc8785_canonicalize(manifest)
    ):
        raise DirectDescriptionError("RG1 manifest or member binding differs")
    decode_lane_program_coordinates(lane_payload)
    decode_skeleton_amplitude_coordinates(amplitude_payload)
    decode_rg3_residual_coordinates(rg3_payload)
    decode_lane_coefficient_deltas(correction_payload)
    return members


def _dilate_four(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant")
    return padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]


def _erode_four(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant")
    return padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]


def _base_masks_for_classes(
    base: CarrierComposeReceiverV1,
    *,
    source_pair_id: int,
    class_a: int,
    class_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_pair_id = source_pair_id - base.predictor.source_pair_start
    if not 0 <= local_pair_id < base.z.n_pairs:
        raise DirectDescriptionError("RG2 SKELETON pair address escapes the SHA-bound base")
    layer_by_role = {row.role: row for row in base.layers}
    masks = []
    for class_id in (class_a, class_b):
        role = _CLASS_TO_ROLE[class_id]
        layer = layer_by_role.get(role)
        if layer is None:
            raise DirectDescriptionError("RG2 SKELETON class role is absent from the base")
        masks.append(
            base._mask_for_layer(
                layer,
                local_pair_id,
                replace_g1_movable=True,
            )
        )
    return masks[0], masks[1]


def derive_skeleton_amplitude_row_band(
    base: CarrierComposeReceiverV1,
    *,
    pair_index: int,
    class_a: int,
    class_b: int,
    family: Literal["EVENT_LOCAL_BOUNDARY", "PER_STRATUM_ROW_BAND"],
) -> int:
    """Choose the highest-mass receiver-derived band without scorer/label input."""

    mask_a, mask_b = _base_masks_for_classes(
        base,
        source_pair_id=pair_index,
        class_a=class_a,
        class_b=class_b,
    )
    if family == "EVENT_LOCAL_BOUNDARY":
        support = (_dilate_four(mask_a) & mask_b) | (_dilate_four(mask_b) & mask_a)
    elif family == "PER_STRATUM_ROW_BAND":
        support = mask_a | mask_b
    else:
        raise DirectDescriptionError("RG2 SKELETON row-band family is unknown")
    masses = [
        int(np.count_nonzero(support[start : start + _ROW_BAND_HEIGHT])) for start in range(0, 384, _ROW_BAND_HEIGHT)
    ]
    if not any(masses):
        raise DirectDescriptionError("RG2 SKELETON typed class pair has no receiver support")
    return int(np.argmax(np.asarray(masses, dtype=np.int64)))


def _fine_band_argmax(mass: np.ndarray, *, row_band: int) -> int:
    if mass.shape != (384, 512):
        raise DirectDescriptionError("RG3 fine-band mass must live on the 384x512 scorer grid")
    start = row_band * _ROW_BAND_HEIGHT
    values = [
        float(np.asarray(mass[start + index * _FINE_BAND_HEIGHT : start + (index + 1) * _FINE_BAND_HEIGHT]).sum())
        for index in range(_FINE_BAND_COUNT)
    ]
    if not any(value > 0.0 for value in values):
        raise DirectDescriptionError("RG3 fine-band address has no receiver support")
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def _all_receiver_masks(
    base: CarrierComposeReceiverV1,
    *,
    pair_index: int,
) -> dict[int, np.ndarray]:
    layer_by_role = {row.role: row for row in base.layers}
    local_pair_id = pair_index - base.predictor.source_pair_start
    if not 0 <= local_pair_id < base.z.n_pairs:
        raise DirectDescriptionError("RG3 SKELETON pair address escapes the SHA-bound base")
    return {
        class_id: base._mask_for_layer(
            layer_by_role[role],
            local_pair_id,
            replace_g1_movable=True,
        )
        for class_id, role in _CLASS_TO_ROLE.items()
        if role in layer_by_role
    }


def derive_rg3_class_birth_address(
    base: CarrierComposeReceiverV1,
    *,
    pair_index: int,
) -> tuple[int, int]:
    """Derive a seed band from class-agnostic receiver geometry, never labels."""

    masks = _all_receiver_masks(base, pair_index=pair_index)
    if not masks:
        raise DirectDescriptionError("RG3 class-birth receiver has no class geometry")
    occupied = np.logical_or.reduce(tuple(masks.values()))
    support = occupied & ~_erode_four(occupied)
    if not support.any():
        support = occupied
    if not support.any():
        raise DirectDescriptionError("RG3 class-birth receiver geometry is empty")
    best = (0, 0)
    best_mass = -1
    for row_band in range(_ROW_BAND_COUNT):
        for fine_band in range(_FINE_BAND_COUNT):
            start = row_band * _ROW_BAND_HEIGHT + fine_band * _FINE_BAND_HEIGHT
            mass = int(np.count_nonzero(support[start : start + _FINE_BAND_HEIGHT]))
            if mass > best_mass:
                best_mass = mass
                best = (row_band, fine_band)
    return best


def derive_rg3_finer_event_local_band(
    base: CarrierComposeReceiverV1,
    *,
    pair_index: int,
    class_a: int,
    class_b: int,
    row_band: int,
) -> int:
    """Refine only one RG2 boundary band into four event-local subbands."""

    mask_a, mask_b = _base_masks_for_classes(
        base,
        source_pair_id=pair_index,
        class_a=class_a,
        class_b=class_b,
    )
    support = (_dilate_four(mask_a) & mask_b) | (_dilate_four(mask_b) & mask_a)
    return _fine_band_argmax(support.astype(np.float32), row_band=row_band)


def derive_rg3_fisher_margin_band(
    base: CarrierComposeReceiverV1,
    *,
    pair_index: int,
    class_a: int,
    class_b: int,
    row_band: int,
    margin_map: np.ndarray,
) -> int:
    """Select one per-stratum subband by categorical-Fisher margin weight."""

    margin = np.asarray(margin_map, dtype=np.float32)
    if margin.shape != (384, 512) or not np.isfinite(margin).all() or (margin < 0).any():
        raise DirectDescriptionError("RG3 Fisher margin map must be finite nonnegative 384x512")
    mask_a, mask_b = _base_masks_for_classes(
        base,
        source_pair_id=pair_index,
        class_a=class_a,
        class_b=class_b,
    )
    support = mask_a | mask_b
    clipped = np.minimum(margin, np.float32(40.0))
    fisher_trace = np.float32(0.5) / np.cosh(clipped * np.float32(0.5)) ** np.float32(2.0)
    return _fine_band_argmax(np.where(support, fisher_trace, np.float32(0.0)), row_band=row_band)


@dataclass(frozen=True, slots=True)
class RG2ReceiverV1:
    """Receiver adapter applying RG2 amplitudes before inherited outer corrections."""

    archive: bytes
    base: CarrierComposeReceiverV1
    skeleton_amplitudes: tuple[SkeletonAmplitudeCoordinateV1, ...]
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.base.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.base.pose6_codes

    @property
    def predictor(self) -> Any:
        return self.base.predictor

    @property
    def layers(self) -> Any:
        return self.base.layers

    @property
    def scorer_solved_templates(self) -> Any:
        return self.base.scorer_solved_templates

    def template_camera_masks(self, pair_ids: Sequence[int], template: Any) -> np.ndarray:
        return self.base.template_camera_masks(pair_ids, template)

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        output = self.base.render_camera_pairs(indexes)
        if self.base.realization_profile is None:
            raise DirectDescriptionError("RG2 SKELETON receiver lost its realization profile")
        from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

        ys = (np.arange(CAMERA_H) * 384 // CAMERA_H).clip(0, 383)
        xs = (np.arange(CAMERA_W) * 512 // CAMERA_W).clip(0, 511)
        yy = np.arange(384, dtype=np.int64)[:, None]
        xx = np.arange(512, dtype=np.int64)[None, :]
        local_by_source = {
            self.base.predictor.source_pair_start + local_pair: output_index
            for output_index, local_pair in enumerate(indexes)
        }
        for row in self.skeleton_amplitudes:
            output_index = local_by_source.get(row.pair_index)
            if output_index is None:
                continue
            expected_band = derive_skeleton_amplitude_row_band(
                self.base,
                pair_index=row.pair_index,
                class_a=row.class_a,
                class_b=row.class_b,
                family=row.family,
            )
            if row.row_band != expected_band:
                raise DirectDescriptionError("RG2 SKELETON row-band/base binding differs")
            mask_a, mask_b = _base_masks_for_classes(
                self.base,
                source_pair_id=row.pair_index,
                class_a=row.class_a,
                class_b=row.class_b,
            )
            role_a, role_b = _CLASS_TO_ROLE[row.class_a], _CLASS_TO_ROLE[row.class_b]
            if REALIZATION_PAINT_ORDER.index(role_a) < REALIZATION_PAINT_ORDER.index(role_b):
                early_role, late_role = role_a, role_b
                early_mask, late_mask = mask_a, mask_b
            else:
                early_role, late_role = role_b, role_a
                early_mask, late_mask = mask_b, mask_a
            band = np.zeros((384, 512), dtype=bool)
            start = row.row_band * _ROW_BAND_HEIGHT
            band[start : start + _ROW_BAND_HEIGHT] = True
            if row.family == "EVENT_LOCAL_BOUNDARY":
                if row.signed_quanta > 0:
                    sites = _dilate_four(late_mask) & early_mask & ~late_mask & band
                    paint_role = late_role
                else:
                    sites = _dilate_four(early_mask) & late_mask & ~early_mask & band
                    paint_role = early_role
            else:
                phase = (yy * 3 + xx * 5 + row.pair_index + row.class_a * 7 + row.class_b * 11) % 16 == 0
                sites = (early_mask if row.signed_quanta > 0 else late_mask) & band & phase
                paint_role = late_role if row.signed_quanta > 0 else early_role
            camera_sites = sites[np.ix_(ys, xs)]
            colour = self.base.realization_profile.colour_for(paint_role)
            output[output_index, 0, camera_sites] = colour
            output[output_index, 1, camera_sites] = colour
        return np.ascontiguousarray(output)


@dataclass(frozen=True, slots=True)
class RG3ReceiverV1:
    """Receiver adapter applying counted RG3 family symbols at L3 raster."""

    archive: bytes
    base: CarrierComposeReceiverV1 | RG2ReceiverV1
    residuals: tuple[RG3ResidualCoordinateV1, ...]
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.base.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.base.pose6_codes

    @property
    def predictor(self) -> Any:
        return self.base.predictor

    @property
    def layers(self) -> Any:
        return self.base.layers

    @property
    def scorer_solved_templates(self) -> Any:
        return self.base.scorer_solved_templates

    def template_camera_masks(self, pair_ids: Sequence[int], template: Any) -> np.ndarray:
        return self.base.template_camera_masks(pair_ids, template)

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        output = self.base.render_camera_pairs(indexes)
        geometry = self.base.base if isinstance(self.base, RG2ReceiverV1) else self.base
        if geometry.realization_profile is None:
            raise DirectDescriptionError("RG3 SKELETON receiver lost its realization profile")
        from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

        ys = (np.arange(CAMERA_H) * 384 // CAMERA_H).clip(0, 383)
        xs = (np.arange(CAMERA_W) * 512 // CAMERA_W).clip(0, 511)
        yy = np.arange(384, dtype=np.int64)[:, None]
        xx = np.arange(512, dtype=np.int64)[None, :]
        local_by_source = {
            geometry.predictor.source_pair_start + local_pair: output_index
            for output_index, local_pair in enumerate(indexes)
        }

        def paint(output_index: int, sites: np.ndarray, role: str) -> None:
            camera_sites = sites[np.ix_(ys, xs)]
            colour = geometry.realization_profile.colour_for(role)
            output[output_index, 0, camera_sites] = colour
            output[output_index, 1, camera_sites] = colour

        for row in self.residuals:
            output_index = local_by_source.get(row.pair_index)
            if output_index is None:
                continue
            start = row.row_band * _ROW_BAND_HEIGHT + row.fine_band * _FINE_BAND_HEIGHT
            band = np.zeros((384, 512), dtype=bool)
            band[start : start + _FINE_BAND_HEIGHT] = True
            if row.family == "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION":
                if (row.row_band, row.fine_band) != derive_rg3_class_birth_address(
                    geometry,
                    pair_index=row.pair_index,
                ):
                    raise DirectDescriptionError("RG3 class-birth address/base binding differs")
                masks = _all_receiver_masks(geometry, pair_index=row.pair_index)
                occupied = np.logical_or.reduce(tuple(masks.values()))
                support = (occupied & ~_erode_four(occupied)) & band
                if not support.any():
                    support = occupied & band
                candidates = np.argwhere(support)
                if not candidates.size:
                    raise DirectDescriptionError("RG3 class-birth seed has no receiver-derived anchor")
                order = np.lexsort((candidates[:, 0], np.abs(candidates[:, 1] - 256)))
                y, x = (int(value) for value in candidates[order[0]])
                neighbor_x = x + 1 if x < 511 else x - 1
                first = np.zeros((384, 512), dtype=bool)
                second = np.zeros((384, 512), dtype=bool)
                first[y, x] = True
                second[y, neighbor_x] = True
                role_a, role_b = _CLASS_TO_ROLE[row.class_a], _CLASS_TO_ROLE[row.class_b]
                if row.signed_quanta < 0:
                    role_a, role_b = role_b, role_a
                paint(output_index, first, role_a)
                paint(output_index, second, role_b)
                continue

            mask_a, mask_b = _base_masks_for_classes(
                geometry,
                source_pair_id=row.pair_index,
                class_a=row.class_a,
                class_b=row.class_b,
            )
            role_a, role_b = _CLASS_TO_ROLE[row.class_a], _CLASS_TO_ROLE[row.class_b]
            if REALIZATION_PAINT_ORDER.index(role_a) < REALIZATION_PAINT_ORDER.index(role_b):
                early_role, late_role = role_a, role_b
                early_mask, late_mask = mask_a, mask_b
            else:
                early_role, late_role = role_b, role_a
                early_mask, late_mask = mask_b, mask_a
            if row.family == "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK":
                expected_fine = derive_rg3_finer_event_local_band(
                    geometry,
                    pair_index=row.pair_index,
                    class_a=row.class_a,
                    class_b=row.class_b,
                    row_band=row.row_band,
                )
                if row.fine_band != expected_fine:
                    raise DirectDescriptionError("RG3 finer-event address/base binding differs")
                dilation = _dilate_four(late_mask if row.signed_quanta > 0 else early_mask)
                if abs(row.signed_quanta) == 2:
                    dilation = _dilate_four(dilation)
                if row.signed_quanta > 0:
                    sites = dilation & early_mask & ~late_mask & band
                    paint_role = late_role
                else:
                    sites = dilation & late_mask & ~early_mask & band
                    paint_role = early_role
            else:
                modulus = 16 if abs(row.signed_quanta) == 1 else 8
                phase = (
                    yy * 3 + xx * 5 + row.pair_index + row.class_a * 7 + row.class_b * 11
                ) % modulus == 0
                sites = (early_mask if row.signed_quanta > 0 else late_mask) & band & phase
                paint_role = late_role if row.signed_quanta > 0 else early_role
            paint(output_index, sites, paint_role)
        return np.ascontiguousarray(output)


def receive_rg1_receiver_grammar(
    archive: bytes,
    *,
    verify_member_effects: bool = False,
) -> CarrierComposeReceiverV1 | RG2ReceiverV1 | RG3ReceiverV1:
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
    amplitudes = decode_skeleton_amplitude_coordinates(members.get(SKELETON_AMPLITUDE_MEMBER, b""))
    rg3_residuals = decode_rg3_residual_coordinates(members.get(RG3_RESIDUAL_MEMBER, b""))
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
    receiver: CarrierComposeReceiverV1 | RG2ReceiverV1 | RG3ReceiverV1 = replace(
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
            "skeleton_amplitude_coordinate_count": len(amplitudes),
            "rg3_residual_coordinate_count": len(rg3_residuals),
            "composition_order_enforced": True,
            "typed_stream_tags_validated": True,
            "sealed_v13_v19c_mutated": False,
            "score_claim": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        },
    )
    if amplitudes:
        for row in amplitudes:
            if row.row_band != derive_skeleton_amplitude_row_band(
                receiver,
                pair_index=row.pair_index,
                class_a=row.class_a,
                class_b=row.class_b,
                family=row.family,
            ):
                raise DirectDescriptionError("RG2 SKELETON row-band/base binding differs")
        receiver = RG2ReceiverV1(
            archive=archive,
            base=receiver,
            skeleton_amplitudes=amplitudes,
            custody={
                **dict(receiver.custody),
                "schema": RECEIVER_SCHEMA_RG2,
                "skeleton_amplitude_coordinate_ids": [row.actuator_id for row in amplitudes],
                "skeleton_amplitude_typed_stream": "SKELETON/L3_raster",
            },
        )
    if rg3_residuals:
        geometry = receiver.base if isinstance(receiver, RG2ReceiverV1) else receiver
        for row in rg3_residuals:
            if row.family == "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION":
                expected = derive_rg3_class_birth_address(
                    geometry,
                    pair_index=row.pair_index,
                )
                if (row.row_band, row.fine_band) != expected:
                    raise DirectDescriptionError("RG3 class-birth address/base binding differs")
            elif row.family == "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK":
                expected_fine = derive_rg3_finer_event_local_band(
                    geometry,
                    pair_index=row.pair_index,
                    class_a=row.class_a,
                    class_b=row.class_b,
                    row_band=row.row_band,
                )
                if row.fine_band != expected_fine:
                    raise DirectDescriptionError("RG3 finer-event address/base binding differs")
        receiver = RG3ReceiverV1(
            archive=archive,
            base=receiver,
            residuals=rg3_residuals,
            custody={
                **dict(receiver.custody),
                "schema": RECEIVER_SCHEMA_RG3,
                "rg3_residual_coordinate_ids": [row.actuator_id for row in rg3_residuals],
                "rg3_residual_typed_stream": "SKELETON/L3_raster",
                "fisher_margin_field_shipped": False,
            },
        )
    if verify_member_effects:
        probe_ids = sorted(
            {
                *(range(600) if coordinates else ()),
                *(row.pair_index for row in amplitudes),
                *(row.pair_index for row in rg3_residuals),
                *(row.pair_index for row in corrections),
            }
        )
        if not probe_ids:
            raise DirectDescriptionError("RG1 wrapper has no effective coordinate")
        if all(
            (base.render_camera_pairs((pair_id,)) == receiver.render_camera_pairs((pair_id,))).all()
            for pair_id in probe_ids
        ):
            raise DirectDescriptionError("RG1 extension is a receiver-output no-op")
    return receiver


__all__ = [
    "ARCHIVE_SCHEMA",
    "ARCHIVE_SCHEMA_RG2",
    "ARCHIVE_SCHEMA_RG3",
    "BASE_MEMBER",
    "CORRECTION_MEMBER",
    "LANE_FIELDS",
    "LANE_PROGRAM_MEMBER",
    "RG3_RESIDUAL_MEMBER",
    "SKELETON_AMPLITUDE_MEMBER",
    "LaneProgramCoordinateV1",
    "RG2ReceiverV1",
    "RG3ReceiverV1",
    "RG3ResidualCoordinateV1",
    "SkeletonAmplitudeCoordinateV1",
    "compile_rg1_receiver_grammar",
    "compile_rg2_receiver_grammar",
    "compile_rg3_receiver_grammar",
    "decode_lane_program_coordinates",
    "decode_rg3_residual_coordinates",
    "decode_skeleton_amplitude_coordinates",
    "derive_rg3_class_birth_address",
    "derive_rg3_finer_event_local_band",
    "derive_rg3_fisher_margin_band",
    "derive_skeleton_amplitude_row_band",
    "encode_lane_program_coordinates",
    "encode_rg3_residual_coordinates",
    "encode_skeleton_amplitude_coordinates",
    "parse_rg1_receiver_grammar",
    "project_polygon_center",
    "receive_rg1_receiver_grammar",
]
