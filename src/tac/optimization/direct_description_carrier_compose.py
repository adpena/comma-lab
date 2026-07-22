# SPDX-License-Identifier: MIT
"""Receiver-closed V9 carrier composition over a custodied DDM predictor.

The counted predictor already owns the five class-carrier payloads and the one
Pose6 stream.  This module adds only a strict outer grammar and the solved G2CS1
chart-symbol refinement surface.  Refinement changes Lane chart coefficients
before generic region-coherent rasterization; pixel-coordinate/value patches
are deliberately not part of this grammar.
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
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.optimization.direct_description_entropy_priced_member import (
    COMPOSED_ROLE_ORDER,
    ComposedStructuredMemberReceiverV1,
    StructuredRoleLayerV1,
    _sha256,
    _zip_stored,
    parse_structured_member_archive,
    receive_structured_member_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_entropy_streams import parse_entropy_chart_archive
from tac.optimization.direct_description_minimizer import SEED, DirectDescriptionError, _require_sha256
from tac.optimization.predictor_upgrade_xi_chart import (
    LaneCoefficientDelta,
    decode_lane_coefficient_deltas,
    encode_lane_coefficient_deltas,
)

CONFIG_SCHEMA: Final = "DirectDescriptionV9CarrierComposeConfigV1"
ARCHIVE_SCHEMA: Final = "direct_description_v9_carrier_compose_archive.v1"
ARCHIVE_SCHEMA_V2: Final = "direct_description_v10_fisher_event_archive.v1"
ARCHIVE_SCHEMA_V3: Final = "direct_description_v11_obligation_archive.v1"
RECEIVER_SCHEMA: Final = "direct_description_v9_carrier_compose_receiver.v1"
RECEIVER_SCHEMA_V2: Final = "direct_description_v10_fisher_event_receiver.v1"
RECEIVER_SCHEMA_V3: Final = "direct_description_v11_obligation_receiver.v1"
RESULT_SCHEMA: Final = "direct_description_v9_carrier_compose_receipt.v1"
RESULT_SCHEMA_V2: Final = "direct_description_v10_fisher_event_search_receipt.v1"
RESULT_SCHEMA_V3: Final = "direct_description_v11_obligation_search_receipt.v1"
RESULT_SCHEMA_V4: Final = "direct_description_v12_obligation_drain_receipt.v1"
MAGIC: Final = "DDV9C1"
MAGIC_V2: Final = "DDV10C1"
MAGIC_V3: Final = "DDV11C1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
CLASS_ORDER: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
ROLE_CLASS_IDS: Final = {
    "Road": 0,
    "Lane": 1,
    "UndrivableBoundary": 2,
    "Movable": 3,
    "MyCar": 4,
}
CORRECTION_MEMBER: Final = "correction/lane_chart_symbols.g2cs"
BOUNDARY_CORRECTION_MEMBER: Final = "correction/road_boundary_coefficients.g2bc"
EVENT_CORRECTION_MEMBER: Final = "correction/topology_events.g2ev"
BOUNDARY_SHEARLET_MEMBER: Final = "correction/boundary_shearlet_atoms.g2sh"
ISLAND_SHAPE_MEMBER: Final = "correction/movable_shape_atoms.g2is"

_ROLE_TO_WIRE: Final = {name: index for index, name in enumerate(COMPOSED_ROLE_ORDER)}
_WIRE_TO_ROLE: Final = {value: key for key, value in _ROLE_TO_WIRE.items()}
_BOUNDARY_MAGIC: Final = b"G2BC1"
_BOUNDARY_VERSION: Final = 1
_BOUNDARY_HEADER: Final = struct.Struct(">5sBHI")
_BOUNDARY_ROW: Final = struct.Struct(">HBBf")
_EVENT_MAGIC: Final = b"G2EV1"
_EVENT_VERSION: Final = 1
_EVENT_HEADER: Final = struct.Struct(">5sBHI")
_EVENT_ROW: Final = struct.Struct(">HBBBBHHHHbb")
_EVENT_ACTION_TO_WIRE: Final = {"birth": 1, "death": 2}
_WIRE_TO_EVENT_ACTION: Final = {value: key for key, value in _EVENT_ACTION_TO_WIRE.items()}
_EVENT_SHAPE_TO_WIRE: Final = {"ellipse": 1, "box": 2}
_WIRE_TO_EVENT_SHAPE: Final = {value: key for key, value in _EVENT_SHAPE_TO_WIRE.items()}
_SHEARLET_MAGIC: Final = b"G2SH1"
_SHEARLET_VERSION: Final = 1
_SHEARLET_HEADER: Final = struct.Struct(">5sBHI")
_SHEARLET_ROW: Final = struct.Struct(">HBHHBHbh")
_ISLAND_MAGIC: Final = b"G2IS1"
_ISLAND_VERSION: Final = 1
_ISLAND_HEADER: Final = struct.Struct(">5sBHI")
_ISLAND_ROW: Final = struct.Struct(">HBBHHBHBbbbbb")


@dataclass(frozen=True, order=True, slots=True)
class BoundaryCoefficientDelta:
    """One counted coefficient of a Road-carrier boundary displacement chart.

    The receiver evaluates four powers of normalized image ``x`` and vertically
    advects the already-counted Road mask.  It never carries a raster or pixel
    value.  ``pair_index`` is the global source-pair address, matching G2CS1.
    """

    pair_index: int
    role: str
    coefficient_index: int
    coefficient_delta: float

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("boundary coefficient pair index is outside [0,600)")
        if self.role != "Road":
            raise DirectDescriptionError("v10 boundary coefficients currently address the Road carrier only")
        if isinstance(self.coefficient_index, bool) or not 0 <= self.coefficient_index < 4:
            raise DirectDescriptionError("boundary coefficient index is outside the cubic chart")
        quantized = float(np.float32(self.coefficient_delta))
        if not np.isfinite(quantized) or quantized == 0.0:
            raise DirectDescriptionError("boundary coefficient delta must be finite and nonzero after fp32")
        object.__setattr__(self, "coefficient_delta", quantized)


@dataclass(frozen=True, order=True, slots=True)
class TopologyEventV1:
    """Parametric birth/death event transported by the counted Pose6 path.

    The payload stores only a semantic role, action, primitive shape, birth
    bounding box, lifetime, and two small gains.  For lifetimes greater than
    one, the receiver displaces the primitive from the already-counted Pose6
    ordinal-code differences; there is no coordinate list or RGB payload.
    """

    pair_index: int
    role: str
    action: str
    shape: str
    lifetime: int
    y0: int
    x0: int
    y1: int
    x1: int
    transport_gain_x_q4: int = 0
    transport_gain_y_q4: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("topology-event pair index is outside [0,600)")
        if self.role not in _ROLE_TO_WIRE:
            raise DirectDescriptionError("topology-event role is unknown")
        if self.action not in _EVENT_ACTION_TO_WIRE or self.shape not in _EVENT_SHAPE_TO_WIRE:
            raise DirectDescriptionError("topology-event action or shape is unknown")
        if isinstance(self.lifetime, bool) or not 1 <= self.lifetime <= 255:
            raise DirectDescriptionError("topology-event lifetime is outside [1,255]")
        if not (0 <= self.y0 < self.y1 <= 384 and 0 <= self.x0 < self.x1 <= 512):
            raise DirectDescriptionError("topology-event bbox is outside scorer geometry")
        for name in ("transport_gain_x_q4", "transport_gain_y_q4"):
            value = getattr(self, name)
            if isinstance(value, bool) or not -128 <= value <= 127:
                raise DirectDescriptionError(f"{name} is outside int8")
        if self.lifetime == 1 and (self.transport_gain_x_q4 or self.transport_gain_y_q4):
            raise DirectDescriptionError("one-pair topology events must not carry inert transport gains")


@dataclass(frozen=True, order=True, slots=True)
class BoundaryShearletAtomV1:
    """One localized, directional boundary-displacement obligation atom.

    The compact parabolic atom is parameterized in chart space.  It is not a
    pixel/value stream: the receiver synthesizes its displacement field from
    center, anisotropic support, shear, and one quantized coefficient.
    """

    pair_index: int
    role: str
    center_y: int
    center_x: int
    scale_y: int
    scale_x: int
    shear_q4: int
    amplitude_q4: int

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("boundary-shearlet pair index is outside [0,600)")
        if self.role not in {"Road", "UndrivableBoundary"}:
            raise DirectDescriptionError("boundary-shearlet role must be Road or UndrivableBoundary")
        if not (0 <= self.center_y < 384 and 0 <= self.center_x < 512):
            raise DirectDescriptionError("boundary-shearlet center is outside scorer geometry")
        if not (2 <= self.scale_y <= 96 and 4 <= self.scale_x <= 256):
            raise DirectDescriptionError("boundary-shearlet support scale is outside the governed bank")
        if self.scale_x < 2 * self.scale_y:
            raise DirectDescriptionError("boundary-shearlet support must preserve parabolic anisotropy")
        if not -64 <= self.shear_q4 <= 64:
            raise DirectDescriptionError("boundary-shearlet shear is outside q4 range")
        if self.amplitude_q4 == 0 or not -512 <= self.amplitude_q4 <= 512:
            raise DirectDescriptionError("boundary-shearlet amplitude is zero or outside q4 range")


@dataclass(frozen=True, order=True, slots=True)
class IslandShapeAtomV1:
    """Low-order Movable island shape plus one Fourier-free curvelet lobe."""

    pair_index: int
    action: str
    lifetime: int
    center_y: int
    center_x: int
    radius_y: int
    radius_x: int
    angle_u8: int
    skew_q6: int
    taper_q6: int
    curvelet_q6: int
    transport_gain_x_q4: int = 0
    transport_gain_y_q4: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("island-shape pair index is outside [0,600)")
        if self.action not in _EVENT_ACTION_TO_WIRE:
            raise DirectDescriptionError("island-shape action is unknown")
        if not 1 <= self.lifetime <= 255:
            raise DirectDescriptionError("island-shape lifetime is outside [1,255]")
        if not (0 <= self.center_y < 384 and 0 <= self.center_x < 512):
            raise DirectDescriptionError("island-shape center is outside scorer geometry")
        if not (1 <= self.radius_y <= 191 and 1 <= self.radius_x <= 255):
            raise DirectDescriptionError("island-shape radius is outside scorer geometry")
        if not 0 <= self.angle_u8 <= 255:
            raise DirectDescriptionError("island-shape angle is outside uint8")
        for name in ("skew_q6", "taper_q6", "curvelet_q6"):
            if not -96 <= getattr(self, name) <= 96:
                raise DirectDescriptionError(f"island-shape {name} is outside the stable q6 range")
        for name in ("transport_gain_x_q4", "transport_gain_y_q4"):
            if not -128 <= getattr(self, name) <= 127:
                raise DirectDescriptionError(f"island-shape {name} is outside int8")
        if self.lifetime == 1 and (self.transport_gain_x_q4 or self.transport_gain_y_q4):
            raise DirectDescriptionError("one-pair island shapes must not carry inert transport gains")


def _encode_boundary_coefficient_deltas(symbols: Sequence[BoundaryCoefficientDelta]) -> bytes:
    rows = tuple(symbols)
    if not rows:
        return b""
    keys = [(row.pair_index, _ROLE_TO_WIRE[row.role], row.coefficient_index) for row in rows]
    if len(rows) > 0xFFFF or keys != sorted(set(keys)):
        raise DirectDescriptionError("boundary coefficient rows must be sorted, unique, and uint16-bounded")
    body = b"".join(
        _BOUNDARY_ROW.pack(
            row.pair_index,
            _ROLE_TO_WIRE[row.role],
            row.coefficient_index,
            row.coefficient_delta,
        )
        for row in rows
    )
    return _BOUNDARY_HEADER.pack(
        _BOUNDARY_MAGIC, _BOUNDARY_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF
    ) + body


def _decode_boundary_coefficient_deltas(payload: bytes) -> tuple[BoundaryCoefficientDelta, ...]:
    if not payload:
        return ()
    if len(payload) < _BOUNDARY_HEADER.size:
        raise DirectDescriptionError("boundary coefficient packet is truncated")
    magic, version, count, checksum = _BOUNDARY_HEADER.unpack_from(payload)
    expected = _BOUNDARY_HEADER.size + count * _BOUNDARY_ROW.size
    if magic != _BOUNDARY_MAGIC or version != _BOUNDARY_VERSION or count == 0 or len(payload) != expected:
        raise DirectDescriptionError("boundary coefficient packet header/length is invalid")
    body = payload[_BOUNDARY_HEADER.size :]
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise DirectDescriptionError("boundary coefficient packet CRC mismatch")
    rows: list[BoundaryCoefficientDelta] = []
    for index in range(count):
        pair_index, role_id, coefficient_index, delta = _BOUNDARY_ROW.unpack_from(
            body, index * _BOUNDARY_ROW.size
        )
        if role_id not in _WIRE_TO_ROLE:
            raise DirectDescriptionError("boundary coefficient packet contains an unknown role")
        rows.append(BoundaryCoefficientDelta(pair_index, _WIRE_TO_ROLE[role_id], coefficient_index, delta))
    result = tuple(rows)
    if _encode_boundary_coefficient_deltas(result) != payload:
        raise DirectDescriptionError("boundary coefficient packet is not canonical on parse-back")
    return result


def _encode_topology_events(events: Sequence[TopologyEventV1]) -> bytes:
    rows = tuple(events)
    if not rows:
        return b""
    keys = [
        (
            row.pair_index,
            _ROLE_TO_WIRE[row.role],
            _EVENT_ACTION_TO_WIRE[row.action],
            _EVENT_SHAPE_TO_WIRE[row.shape],
            row.y0,
            row.x0,
            row.y1,
            row.x1,
        )
        for row in rows
    ]
    if len(rows) > 0xFFFF or keys != sorted(set(keys)):
        raise DirectDescriptionError("topology events must be sorted, unique, and uint16-bounded")
    body = b"".join(
        _EVENT_ROW.pack(
            row.pair_index,
            _ROLE_TO_WIRE[row.role],
            _EVENT_ACTION_TO_WIRE[row.action],
            _EVENT_SHAPE_TO_WIRE[row.shape],
            row.lifetime,
            row.y0,
            row.x0,
            row.y1,
            row.x1,
            row.transport_gain_x_q4,
            row.transport_gain_y_q4,
        )
        for row in rows
    )
    return _EVENT_HEADER.pack(_EVENT_MAGIC, _EVENT_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF) + body


def _decode_topology_events(payload: bytes) -> tuple[TopologyEventV1, ...]:
    if not payload:
        return ()
    if len(payload) < _EVENT_HEADER.size:
        raise DirectDescriptionError("topology-event packet is truncated")
    magic, version, count, checksum = _EVENT_HEADER.unpack_from(payload)
    expected = _EVENT_HEADER.size + count * _EVENT_ROW.size
    if magic != _EVENT_MAGIC or version != _EVENT_VERSION or count == 0 or len(payload) != expected:
        raise DirectDescriptionError("topology-event packet header/length is invalid")
    body = payload[_EVENT_HEADER.size :]
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise DirectDescriptionError("topology-event packet CRC mismatch")
    rows: list[TopologyEventV1] = []
    for index in range(count):
        values = _EVENT_ROW.unpack_from(body, index * _EVENT_ROW.size)
        pair_index, role_id, action_id, shape_id, lifetime, y0, x0, y1, x1, gain_x, gain_y = values
        if role_id not in _WIRE_TO_ROLE or action_id not in _WIRE_TO_EVENT_ACTION or shape_id not in _WIRE_TO_EVENT_SHAPE:
            raise DirectDescriptionError("topology-event packet contains an unknown enum value")
        rows.append(
            TopologyEventV1(
                pair_index,
                _WIRE_TO_ROLE[role_id],
                _WIRE_TO_EVENT_ACTION[action_id],
                _WIRE_TO_EVENT_SHAPE[shape_id],
                lifetime,
                y0,
                x0,
                y1,
                x1,
                gain_x,
                gain_y,
            )
        )
    result = tuple(rows)
    if _encode_topology_events(result) != payload:
        raise DirectDescriptionError("topology-event packet is not canonical on parse-back")
    return result


def _encode_boundary_shearlet_atoms(atoms: Sequence[BoundaryShearletAtomV1]) -> bytes:
    rows = tuple(atoms)
    if not rows:
        return b""
    keys = [(row.pair_index, _ROLE_TO_WIRE[row.role], row.center_y, row.center_x) for row in rows]
    if len(rows) > 0xFFFF or keys != sorted(set(keys)):
        raise DirectDescriptionError("boundary-shearlet atoms must be sorted, unique, and uint16-bounded")
    body = b"".join(
        _SHEARLET_ROW.pack(
            row.pair_index,
            _ROLE_TO_WIRE[row.role],
            row.center_y,
            row.center_x,
            row.scale_y,
            row.scale_x,
            row.shear_q4,
            row.amplitude_q4,
        )
        for row in rows
    )
    return _SHEARLET_HEADER.pack(
        _SHEARLET_MAGIC, _SHEARLET_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF
    ) + body


def _decode_boundary_shearlet_atoms(payload: bytes) -> tuple[BoundaryShearletAtomV1, ...]:
    if not payload:
        return ()
    if len(payload) < _SHEARLET_HEADER.size:
        raise DirectDescriptionError("boundary-shearlet packet is truncated")
    magic, version, count, checksum = _SHEARLET_HEADER.unpack_from(payload)
    expected = _SHEARLET_HEADER.size + count * _SHEARLET_ROW.size
    if magic != _SHEARLET_MAGIC or version != _SHEARLET_VERSION or count == 0 or len(payload) != expected:
        raise DirectDescriptionError("boundary-shearlet packet header/length is invalid")
    body = payload[_SHEARLET_HEADER.size :]
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise DirectDescriptionError("boundary-shearlet packet CRC mismatch")
    rows: list[BoundaryShearletAtomV1] = []
    for index in range(count):
        pair, role_id, cy, cx, sy, sx, shear, amplitude = _SHEARLET_ROW.unpack_from(
            body, index * _SHEARLET_ROW.size
        )
        if role_id not in _WIRE_TO_ROLE:
            raise DirectDescriptionError("boundary-shearlet packet contains an unknown role")
        rows.append(BoundaryShearletAtomV1(pair, _WIRE_TO_ROLE[role_id], cy, cx, sy, sx, shear, amplitude))
    result = tuple(rows)
    if _encode_boundary_shearlet_atoms(result) != payload:
        raise DirectDescriptionError("boundary-shearlet packet is not canonical on parse-back")
    return result


def _encode_island_shape_atoms(atoms: Sequence[IslandShapeAtomV1]) -> bytes:
    rows = tuple(atoms)
    if not rows:
        return b""
    keys = [(row.pair_index, _EVENT_ACTION_TO_WIRE[row.action], row.center_y, row.center_x) for row in rows]
    if len(rows) > 0xFFFF or keys != sorted(set(keys)):
        raise DirectDescriptionError("island-shape atoms must be sorted, unique, and uint16-bounded")
    body = b"".join(
        _ISLAND_ROW.pack(
            row.pair_index,
            _EVENT_ACTION_TO_WIRE[row.action],
            row.lifetime,
            row.center_y,
            row.center_x,
            row.radius_y,
            row.radius_x,
            row.angle_u8,
            row.skew_q6,
            row.taper_q6,
            row.curvelet_q6,
            row.transport_gain_x_q4,
            row.transport_gain_y_q4,
        )
        for row in rows
    )
    return _ISLAND_HEADER.pack(
        _ISLAND_MAGIC, _ISLAND_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF
    ) + body


def _decode_island_shape_atoms(payload: bytes) -> tuple[IslandShapeAtomV1, ...]:
    if not payload:
        return ()
    if len(payload) < _ISLAND_HEADER.size:
        raise DirectDescriptionError("island-shape packet is truncated")
    magic, version, count, checksum = _ISLAND_HEADER.unpack_from(payload)
    expected = _ISLAND_HEADER.size + count * _ISLAND_ROW.size
    if magic != _ISLAND_MAGIC or version != _ISLAND_VERSION or count == 0 or len(payload) != expected:
        raise DirectDescriptionError("island-shape packet header/length is invalid")
    body = payload[_ISLAND_HEADER.size :]
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise DirectDescriptionError("island-shape packet CRC mismatch")
    rows: list[IslandShapeAtomV1] = []
    for index in range(count):
        values = _ISLAND_ROW.unpack_from(body, index * _ISLAND_ROW.size)
        pair, action_id, lifetime, cy, cx, ry, rx, angle, skew, taper, curvelet, gain_x, gain_y = values
        if action_id not in _WIRE_TO_EVENT_ACTION:
            raise DirectDescriptionError("island-shape packet contains an unknown action")
        rows.append(
            IslandShapeAtomV1(
                pair,
                _WIRE_TO_EVENT_ACTION[action_id],
                lifetime,
                cy,
                cx,
                ry,
                rx,
                angle,
                skew,
                taper,
                curvelet,
                gain_x,
                gain_y,
            )
        )
    result = tuple(rows)
    if _encode_island_shape_atoms(result) != payload:
        raise DirectDescriptionError("island-shape packet is not canonical on parse-back")
    return result


class DirectDescriptionV9CarrierComposeConfigV1(BaseModel):
    """Typed local-only measurement contract for one bridge window."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionV9CarrierComposeConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_v9_carrier_compose_seed1234"] = "ddm_v9_carrier_compose_seed1234"
    seed: Literal[1234] = SEED
    pair_start: Literal[344, 448]
    pair_count: Literal[64, 256]
    v6_receipt_path: StrictStr
    v6_receipt_sha256: StrictStr
    predictor_archive_path: StrictStr
    predictor_archive_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16, 32] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    correction_policy: Literal["g2cs1_chart_coefficients_only_fisher_margin_ranked_no_pixel_residual"] = (
        "g2cs1_chart_coefficients_only_fisher_margin_ranked_no_pixel_residual"
    )
    correction_symbols: tuple[tuple[StrictInt, StrictInt, StrictInt, float], ...] = ()
    checkpoint_policy: Literal["atomic_preserve_build_then_measure"] = "atomic_preserve_build_then_measure"
    rate_authority: Literal["exact_len_receiver_closed_v9_zip"] = "exact_len_receiver_closed_v9_zip"
    class_order: tuple[StrictStr, ...] = CLASS_ORDER
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionV9CarrierComposeConfigV1:
        for name in ("v6_receipt_sha256", "predictor_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if (self.pair_start, self.pair_count) not in {(448, 64), (344, 256)}:
            raise ValueError("bridge windows must be exactly [448,512) or [344,600)")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute scorer custody")
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be canonical {CLASS_ORDER!r}")
        symbols = tuple(LaneCoefficientDelta(*row) for row in self.correction_symbols)
        if tuple((s.pair_index, s.line_index, s.coefficient_index) for s in symbols) != tuple(
            sorted({(s.pair_index, s.line_index, s.coefficient_index) for s in symbols})
        ):
            raise ValueError("correction symbols must be sorted and address-unique")
        return self

    def symbols(self) -> tuple[LaneCoefficientDelta, ...]:
        return tuple(LaneCoefficientDelta(*row) for row in self.correction_symbols)

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


class DirectDescriptionV10FisherEventSearchConfigV1(BaseModel):
    """Typed, local-only greedy candidate-search contract.

    This is deliberately not named a closed-form solver.  Candidates are
    derived encoder-side from frozen-scorer error cells and Fisher/margin
    geometry, then admitted only by measured receiver replay.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionV10FisherEventSearchConfigV1"] = Field(
        default="DirectDescriptionV10FisherEventSearchConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: StrictStr
    seed: Literal[1234] = SEED
    pair_start: Literal[344, 448]
    pair_count: Literal[64, 256]
    v6_receipt_path: StrictStr
    v6_receipt_sha256: StrictStr
    predictor_archive_path: StrictStr
    predictor_archive_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16, 32] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    added_budget_bytes: tuple[StrictInt, ...] = (0, 5120, 15360, 40960, 102400)
    max_candidates: StrictInt = Field(default=192, ge=1, le=2048)
    minimum_candidates_per_family: StrictInt = Field(default=2, ge=1, le=64)
    max_components_per_pair_role: StrictInt = Field(default=3, ge=1, le=16)
    min_component_sites: StrictInt = Field(default=12, ge=2, le=4096)
    pose_dpose_increase_limit: float = Field(default=0.0, ge=0.0, le=1.0)
    correction_policy: Literal[
        "greedy_measured_fisher_margin_candidate_search_g2cs1_boundary_xi_events_no_pixels"
    ] = "greedy_measured_fisher_margin_candidate_search_g2cs1_boundary_xi_events_no_pixels"
    checkpoint_policy: Literal[
        "atomic_preserve_inventory_every_candidate_every_budget"
    ] = "atomic_preserve_inventory_every_candidate_every_budget"
    rate_authority: Literal["exact_len_receiver_closed_v10_zip"] = "exact_len_receiver_closed_v10_zip"
    class_order: tuple[StrictStr, ...] = CLASS_ORDER
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionV10FisherEventSearchConfigV1:
        for name in ("v6_receipt_sha256", "predictor_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if (self.pair_start, self.pair_count) not in {(448, 64), (344, 256)}:
            raise ValueError("v10 bridge windows must be exactly [448,512) or [344,600)")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute scorer custody")
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be canonical {CLASS_ORDER!r}")
        budgets = tuple(int(value) for value in self.added_budget_bytes)
        if budgets != tuple(sorted(set(budgets))) or not budgets or budgets[0] != 0:
            raise ValueError("added budgets must be sorted, unique, nonempty, and begin at zero")
        if budgets[-1] > 102400:
            raise ValueError("local v10 correction budget is capped at 100 KiB")
        if self.minimum_candidates_per_family * 4 > self.max_candidates:
            raise ValueError("max_candidates must cover four mechanism-family minima")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


class DirectDescriptionV11ObligationSearchConfigV1(BaseModel):
    """Typed scorer-obligation vocabulary and measured joint-objective search."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionV11ObligationSearchConfigV1"] = Field(
        default="DirectDescriptionV11ObligationSearchConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: StrictStr
    seed: Literal[1234] = SEED
    pair_start: Literal[0, 344, 448]
    pair_count: Literal[64, 256, 600]
    v6_receipt_path: StrictStr
    v6_receipt_sha256: StrictStr
    predictor_archive_path: StrictStr
    predictor_archive_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    added_budget_bytes: tuple[StrictInt, ...] = (0, 16384, 49152, 98304, 147456)
    total_archive_ceiling_bytes: Literal[200000] = 200000
    max_generated_candidates: StrictInt = Field(default=4096, ge=128, le=16384)
    max_measured_candidates: StrictInt = Field(default=32, ge=8, le=128)
    max_atoms_per_measured_bundle: StrictInt = Field(default=128, ge=16, le=512)
    minimum_candidates_per_family: StrictInt = Field(default=1, ge=1, le=8)
    max_components_per_pair_role: StrictInt = Field(default=2, ge=1, le=8)
    min_component_sites: StrictInt = Field(default=8, ge=2, le=4096)
    pose_tube_dpose_radius: float = Field(default=1.0, gt=0.0, le=10.0)
    correction_policy: Literal[
        "obligation_derived_lane_full_curvelet_boundary_island_moments_no_pixels"
    ] = "obligation_derived_lane_full_curvelet_boundary_island_moments_no_pixels"
    admission_policy: Literal[
        "greedy_measured_joint_contest_objective_pose_tube_safety_only"
    ] = "greedy_measured_joint_contest_objective_pose_tube_safety_only"
    boundary_basis: Literal[
        "governed_compact_parabolic_shearlet_bank_fourier_free"
    ] = "governed_compact_parabolic_shearlet_bank_fourier_free"
    checkpoint_policy: Literal[
        "atomic_preserve_inventory_every_candidate_every_budget"
    ] = "atomic_preserve_inventory_every_candidate_every_budget"
    rate_authority: Literal["exact_len_receiver_closed_v11_zip"] = "exact_len_receiver_closed_v11_zip"
    class_order: tuple[StrictStr, ...] = CLASS_ORDER
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionV11ObligationSearchConfigV1:
        for name in ("v6_receipt_sha256", "predictor_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if (self.pair_start, self.pair_count) not in {(448, 64), (344, 256), (0, 600)}:
            raise ValueError("v11 windows must be exactly [448,512), [344,600), or [0,600)")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute scorer custody")
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be canonical {CLASS_ORDER!r}")
        budgets = tuple(int(value) for value in self.added_budget_bytes)
        if budgets != tuple(sorted(set(budgets))) or not budgets or budgets[0] != 0:
            raise ValueError("v11 added budgets must be sorted, unique, nonempty, and begin at zero")
        if budgets[-1] > 147456:
            raise ValueError("v11 added-byte request exceeds the preregistered near-200KB ladder")
        if self.minimum_candidates_per_family * 6 > self.max_measured_candidates:
            raise ValueError("max_measured_candidates must cover all six obligation families")
        if self.max_measured_candidates > self.max_generated_candidates:
            raise ValueError("measured candidate cap cannot exceed generated candidate cap")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


class DirectDescriptionV12ObligationDrainConfigV1(DirectDescriptionV11ObligationSearchConfigV1):
    """Typed exhaustive, resumable drain of the bounded V11 obligation pool."""

    schema_: Literal["DirectDescriptionV12ObligationDrainConfigV1"] = Field(
        default="DirectDescriptionV12ObligationDrainConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    max_measured_candidates: StrictInt = Field(default=512, ge=32, le=4096)
    max_bundles_per_invocation: StrictInt = Field(default=64, ge=1, le=512)
    max_atoms_per_measured_bundle: StrictInt = Field(default=16, ge=1, le=64)
    drain_policy: Literal[
        "exhaustive_conflict_free_canonical_batch_family_partition"
    ] = "exhaustive_conflict_free_canonical_batch_family_partition"
    ev_order_policy: Literal[
        "flip_distance_x_margin_band_x_stratum_mass_movable_lane_first"
    ] = "flip_distance_x_margin_band_x_stratum_mass_movable_lane_first"
    base_cache_policy: Literal[
        "immutable_zlib_argmax_pose_per_canonical_batch"
    ] = "immutable_zlib_argmax_pose_per_canonical_batch"
    checkpoint_policy: Literal[
        "atomic_preserve_inventory_base_batches_every_candidate_every_budget"
    ] = "atomic_preserve_inventory_base_batches_every_candidate_every_budget"

    @model_validator(mode="after")
    def _valid_v12(self) -> DirectDescriptionV12ObligationDrainConfigV1:
        if (self.pair_start, self.pair_count) != (0, 600):
            raise ValueError("v12 obligation drain is the exact full n600 [0,600) window")
        if self.max_atoms_per_measured_bundle > 64:
            raise ValueError("v12 bundles are capped at 64 atomic chart/event obligations")
        return self


def _manifest_for(
    predictor_archive: bytes,
    predictor: ComposedStructuredMemberReceiverV1,
    correction_payload: bytes,
    boundary_payload: bytes = b"",
    event_payload: bytes = b"",
    shearlet_payload: bytes = b"",
    island_payload: bytes = b"",
    obligation_vocabulary: bool = False,
) -> dict[str, Any]:
    if shearlet_payload or island_payload or obligation_vocabulary:
        return {
            "schema": ARCHIVE_SCHEMA_V3,
            "magic": MAGIC_V3,
            "pair_count": predictor.z.n_pairs,
            "source_pair_start": predictor.source_pair_start,
            "class_order": list(CLASS_ORDER),
            "role_order": list(COMPOSED_ROLE_ORDER),
            "role_class_ids": ROLE_CLASS_IDS,
            "predictor": {"bytes": len(predictor_archive), "sha256": _sha256(predictor_archive)},
            "corrections": {
                "lane_full_coefficients": {
                    "member": CORRECTION_MEMBER if correction_payload else None,
                    "bytes": len(correction_payload),
                    "sha256": _sha256(correction_payload),
                    "symbol_count": len(decode_lane_coefficient_deltas(correction_payload)),
                    "coefficient_roles": "centerline_c0_c3_width_c4_c5_dash_phase_c7",
                },
                "boundary_shearlet_atoms": {
                    "member": BOUNDARY_SHEARLET_MEMBER if shearlet_payload else None,
                    "bytes": len(shearlet_payload),
                    "sha256": _sha256(shearlet_payload),
                    "symbol_count": len(_decode_boundary_shearlet_atoms(shearlet_payload)),
                    "basis": "compact parabolic shearlet displacement atoms; Fourier-free",
                },
                "movable_shape_atoms": {
                    "member": ISLAND_SHAPE_MEMBER if island_payload else None,
                    "bytes": len(island_payload),
                    "sha256": _sha256(island_payload),
                    "symbol_count": len(_decode_island_shape_atoms(island_payload)),
                    "basis": "birth/death plus low-order moments and one compact curvelet lobe; Fourier-free",
                },
                "pixel_coordinate_or_rgb_patch_present": False,
                "admission": "measured receiver replay under the exact joint contest objective",
            },
            "merge_diff_correct": {
                "merge": "five nested semantic carrier masks in canonical role order",
                "diff": "scorer-obligation clusters ranked by rank-4 flip distance, margin band, and curvature",
                "correct": "chart and parametric atoms rerasterize before canonical semantic paint",
            },
            "xi_pose6": {
                "home": "predictor.zip::chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin",
                "ownership": "sole counted Pose6 owner; Lane phase and island transport consume it without duplication",
            },
            "scorer_weights_present": False,
            "ground_truth_argmax_present": False,
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        }
    if boundary_payload or event_payload:
        return {
            "schema": ARCHIVE_SCHEMA_V2,
            "magic": MAGIC_V2,
            "pair_count": predictor.z.n_pairs,
            "source_pair_start": predictor.source_pair_start,
            "class_order": list(CLASS_ORDER),
            "role_order": list(COMPOSED_ROLE_ORDER),
            "role_class_ids": ROLE_CLASS_IDS,
            "predictor": {"bytes": len(predictor_archive), "sha256": _sha256(predictor_archive)},
            "corrections": {
                "lane_g2cs1": {
                    "member": CORRECTION_MEMBER if correction_payload else None,
                    "bytes": len(correction_payload),
                    "sha256": _sha256(correction_payload),
                    "symbol_count": len(decode_lane_coefficient_deltas(correction_payload)),
                },
                "road_boundary_coefficients": {
                    "member": BOUNDARY_CORRECTION_MEMBER if boundary_payload else None,
                    "bytes": len(boundary_payload),
                    "sha256": _sha256(boundary_payload),
                    "symbol_count": len(_decode_boundary_coefficient_deltas(boundary_payload)),
                    "basis": "cubic normalized-x displacement of the counted Road mask boundary",
                },
                "topology_events": {
                    "member": EVENT_CORRECTION_MEMBER if event_payload else None,
                    "bytes": len(event_payload),
                    "sha256": _sha256(event_payload),
                    "symbol_count": len(_decode_topology_events(event_payload)),
                    "alphabet": "semantic birth/death x ellipse/box x counted Pose6-code transport",
                },
                "pixel_coordinate_or_rgb_patch_present": False,
                "admission": "nonempty only after encoder-side Fisher/margin candidate screening",
            },
            "merge_diff_correct": {
                "merge": "five nested semantic carrier masks in canonical role order",
                "diff": "Lane coefficients, Road boundary chart coefficients, and parametric topology events",
                "correct": "receiver rerasterizes semantic masks before canonical layer paint",
            },
            "xi_pose6": {
                "home": "predictor.zip::chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin",
                "ownership": "inherited sole counted Pose6 owner; event transport consumes it without duplication",
            },
            "scorer_weights_present": False,
            "ground_truth_argmax_present": False,
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        }
    return {
        "schema": ARCHIVE_SCHEMA,
        "magic": MAGIC,
        "pair_count": predictor.z.n_pairs,
        "source_pair_start": predictor.source_pair_start,
        "class_order": list(CLASS_ORDER),
        "role_order": list(COMPOSED_ROLE_ORDER),
        "role_class_ids": ROLE_CLASS_IDS,
        "predictor": {"bytes": len(predictor_archive), "sha256": _sha256(predictor_archive)},
        "correction": {
            "member": CORRECTION_MEMBER if correction_payload else None,
            "bytes": len(correction_payload),
            "sha256": _sha256(correction_payload),
            "symbol_count": len(decode_lane_coefficient_deltas(correction_payload)),
            "policy": "G2CS1 counted Lane coefficient deltas before region-coherent rerasterization",
            "pixel_coordinate_or_rgb_patch_present": False,
            "admission": "nonempty only after caller-owned hard-oracle selection",
        },
        "merge_diff_correct": {
            "merge": "five nested semantic carrier masks in canonical role order",
            "diff": "G2CS1 addresses chart coefficients, never pixels",
            "correct": "generic Lane chart rerasterization then canonical layer merge",
        },
        "xi_pose6": {
            "home": "predictor.zip::chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin",
            "ownership": "inherited sole counted Pose6 owner; no duplicate stream",
        },
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def compile_carrier_compose_archive(
    predictor_archive: bytes,
    symbols: Sequence[LaneCoefficientDelta] = (),
    boundary_symbols: Sequence[BoundaryCoefficientDelta] = (),
    topology_events: Sequence[TopologyEventV1] = (),
    boundary_shearlets: Sequence[BoundaryShearletAtomV1] = (),
    island_shapes: Sequence[IslandShapeAtomV1] = (),
    obligation_vocabulary: bool = False,
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    """Compile a byte-canonical outer archive around the five-carrier predictor.

    With only Lane symbols this emits the byte-compatible V9 grammar.  Boundary
    or topology symbols opt into V10; obligation atoms opt into V11 while
    preserving the same nested predictor and receiver lineage.
    """

    predictor = receive_structured_member_archive(predictor_archive)
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v9 carrier composition requires the composed structured predictor")
    if [layer.role for layer in predictor.layers] != list(COMPOSED_ROLE_ORDER):
        raise DirectDescriptionError("predictor role order differs from the canonical merge order")
    if {layer.role: layer.class_id for layer in predictor.layers} != ROLE_CLASS_IDS:
        raise DirectDescriptionError("predictor role/class self-detection differs from canonical IDs")
    window_start = predictor.source_pair_start
    window_stop = window_start + predictor.z.n_pairs
    addressed = [*symbols, *boundary_symbols, *topology_events, *boundary_shearlets, *island_shapes]
    if any(row.pair_index < window_start or row.pair_index >= window_stop for row in addressed):
        raise DirectDescriptionError("correction symbol is outside the nested predictor source window")
    if any(row.pair_index + row.lifetime > window_stop for row in topology_events):
        raise DirectDescriptionError("topology-event lifetime escapes the nested predictor source window")
    if any(row.pair_index + row.lifetime > window_stop for row in island_shapes):
        raise DirectDescriptionError("island-shape lifetime escapes the nested predictor source window")
    v11_requested = bool(boundary_shearlets or island_shapes or obligation_vocabulary)
    if v11_requested and (boundary_symbols or topology_events):
        raise DirectDescriptionError("V10 and V11 correction vocabularies cannot be mixed in one archive")
    correction_payload = encode_lane_coefficient_deltas(tuple(symbols))
    boundary_payload = _encode_boundary_coefficient_deltas(tuple(boundary_symbols))
    event_payload = _encode_topology_events(tuple(topology_events))
    shearlet_payload = _encode_boundary_shearlet_atoms(tuple(boundary_shearlets))
    island_payload = _encode_island_shape_atoms(tuple(island_shapes))
    members = {
        "manifest.json": rfc8785_canonicalize(
            _manifest_for(
                predictor_archive,
                predictor,
                correction_payload,
                boundary_payload,
                event_payload,
                shearlet_payload,
                island_payload,
                v11_requested,
            )
        ),
        "predictor.zip": predictor_archive,
    }
    if correction_payload:
        members[CORRECTION_MEMBER] = correction_payload
    if boundary_payload:
        members[BOUNDARY_CORRECTION_MEMBER] = boundary_payload
    if event_payload:
        members[EVENT_CORRECTION_MEMBER] = event_payload
    if shearlet_payload:
        members[BOUNDARY_SHEARLET_MEMBER] = shearlet_payload
    if island_payload:
        members[ISLAND_SHAPE_MEMBER] = island_payload
    first = _zip_stored(members)
    second = _zip_stored(members)
    if first != second:
        raise DirectDescriptionError("carrier compiler is nondeterministic")
    parsed, homes = parse_carrier_compose_archive(first)
    if parsed != members or _zip_stored(parsed) != first:
        raise DirectDescriptionError("carrier archive parse/re-encode identity failed")
    return first, homes


def parse_carrier_compose_archive(archive: bytes) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    """Strictly parse V9/V10 outer ZIPs and close exact unique-home accounting."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            expected_prefix = ["manifest.json", "predictor.zip"]
            if [row.filename for row in infos[:2]] != expected_prefix or not 2 <= len(infos) <= 7:
                raise DirectDescriptionError("carrier archive member order/cardinality is invalid")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in Path(row.filename).parts
                for row in infos
            ):
                raise DirectDescriptionError("carrier archive metadata is noncanonical")
            members = {row.filename: reader.read(row) for row in infos}
            member_order = [row.filename for row in infos]
            start_dir = reader.start_dir
    except DirectDescriptionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise DirectDescriptionError("carrier archive ZIP is malformed") from exc
    try:
        manifest = json.loads(members["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError("carrier manifest is malformed") from exc
    correction = members.get(CORRECTION_MEMBER, b"")
    boundary = members.get(BOUNDARY_CORRECTION_MEMBER, b"")
    events = members.get(EVENT_CORRECTION_MEMBER, b"")
    shearlets = members.get(BOUNDARY_SHEARLET_MEMBER, b"")
    islands = members.get(ISLAND_SHAPE_MEMBER, b"")
    common_invalid = (
        rfc8785_canonicalize(manifest) != members["manifest.json"]
        or manifest.get("class_order") != list(CLASS_ORDER)
        or manifest.get("role_order") != list(COMPOSED_ROLE_ORDER)
        or manifest.get("role_class_ids") != ROLE_CLASS_IDS
        or manifest.get("predictor")
        != {"bytes": len(members["predictor.zip"]), "sha256": _sha256(members["predictor.zip"])}
    )
    schema = manifest.get("schema")
    if schema == ARCHIVE_SCHEMA:
        invalid = (
            common_invalid
            or manifest.get("magic") != MAGIC
            or boundary
            or events
            or shearlets
            or islands
            or manifest.get("correction", {}).get("member") != (CORRECTION_MEMBER if correction else None)
            or manifest.get("correction", {}).get("bytes") != len(correction)
            or manifest.get("correction", {}).get("sha256") != _sha256(correction)
            or set(members) != {"manifest.json", "predictor.zip", *([CORRECTION_MEMBER] if correction else [])}
        )
    elif schema == ARCHIVE_SCHEMA_V2:
        correction_rows = manifest.get("corrections", {})
        expected_members = {
            "manifest.json",
            "predictor.zip",
            *([CORRECTION_MEMBER] if correction else []),
            *([BOUNDARY_CORRECTION_MEMBER] if boundary else []),
            *([EVENT_CORRECTION_MEMBER] if events else []),
        }
        payloads = {
            "lane_g2cs1": (CORRECTION_MEMBER, correction),
            "road_boundary_coefficients": (BOUNDARY_CORRECTION_MEMBER, boundary),
            "topology_events": (EVENT_CORRECTION_MEMBER, events),
        }
        invalid = (
            common_invalid
            or manifest.get("magic") != MAGIC_V2
            or shearlets
            or islands
            or set(members) != expected_members
        )
        for key, (member_name, payload) in payloads.items():
            row = correction_rows.get(key, {})
            invalid = invalid or row.get("member") != (member_name if payload else None)
            invalid = invalid or row.get("bytes") != len(payload) or row.get("sha256") != _sha256(payload)
    elif schema == ARCHIVE_SCHEMA_V3:
        correction_rows = manifest.get("corrections", {})
        expected_members = {
            "manifest.json",
            "predictor.zip",
            *([CORRECTION_MEMBER] if correction else []),
            *([BOUNDARY_SHEARLET_MEMBER] if shearlets else []),
            *([ISLAND_SHAPE_MEMBER] if islands else []),
        }
        payloads = {
            "lane_full_coefficients": (CORRECTION_MEMBER, correction),
            "boundary_shearlet_atoms": (BOUNDARY_SHEARLET_MEMBER, shearlets),
            "movable_shape_atoms": (ISLAND_SHAPE_MEMBER, islands),
        }
        invalid = (
            common_invalid
            or manifest.get("magic") != MAGIC_V3
            or boundary
            or events
            or set(members) != expected_members
        )
        for key, (member_name, payload) in payloads.items():
            row = correction_rows.get(key, {})
            invalid = invalid or row.get("member") != (member_name if payload else None)
            invalid = invalid or row.get("bytes") != len(payload) or row.get("sha256") != _sha256(payload)
    else:
        invalid = True
    if invalid:
        raise DirectDescriptionError("carrier manifest identity/custody is invalid")
    canonical_member_order = [
        "manifest.json",
        "predictor.zip",
        *([CORRECTION_MEMBER] if correction else []),
        *([BOUNDARY_CORRECTION_MEMBER] if boundary else []),
        *([EVENT_CORRECTION_MEMBER] if events else []),
        *([BOUNDARY_SHEARLET_MEMBER] if shearlets else []),
        *([ISLAND_SHAPE_MEMBER] if islands else []),
    ]
    if member_order != canonical_member_order:
        raise DirectDescriptionError("carrier correction member order is noncanonical")
    symbols = decode_lane_coefficient_deltas(correction)
    boundary_symbols = _decode_boundary_coefficient_deltas(boundary)
    topology_events = _decode_topology_events(events)
    boundary_shearlets = _decode_boundary_shearlet_atoms(shearlets)
    island_shapes = _decode_island_shape_atoms(islands)
    if schema == ARCHIVE_SCHEMA:
        counts_valid = manifest["correction"]["symbol_count"] == len(symbols)
    elif schema == ARCHIVE_SCHEMA_V2:
        counts_valid = (
            manifest["corrections"]["lane_g2cs1"]["symbol_count"] == len(symbols)
            and manifest["corrections"]["road_boundary_coefficients"]["symbol_count"]
            == len(boundary_symbols)
            and manifest["corrections"]["topology_events"]["symbol_count"] == len(topology_events)
        )
    else:
        counts_valid = (
            manifest["corrections"]["lane_full_coefficients"]["symbol_count"] == len(symbols)
            and manifest["corrections"]["boundary_shearlet_atoms"]["symbol_count"]
            == len(boundary_shearlets)
            and manifest["corrections"]["movable_shape_atoms"]["symbol_count"] == len(island_shapes)
        )
    if not counts_valid:
        raise DirectDescriptionError("carrier correction count differs after parse-back")
    predictor = receive_structured_member_archive(members["predictor.zip"])
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("v9 nested predictor is not composed")
    if predictor.z.n_pairs != manifest["pair_count"] or predictor.source_pair_start != manifest["source_pair_start"]:
        raise DirectDescriptionError("v9 nested predictor window differs from manifest")
    if _zip_stored(members) != archive:
        raise DirectDescriptionError("v9 carrier archive is not byte-canonical")
    homes: list[dict[str, Any]] = []
    for index, info in enumerate(infos):
        next_offset = infos[index + 1].header_offset if index + 1 < len(infos) else start_dir
        homes.append(
            {
                "name": info.filename,
                "payload_bytes": info.file_size,
                "zip_home_bytes": next_offset - info.header_offset,
                "payload_sha256": _sha256(members[info.filename]),
            }
        )
    homes.append(
        {
            "name": "__central_directory_and_eocd__",
            "payload_bytes": 0,
            "zip_home_bytes": len(archive) - sum(row["zip_home_bytes"] for row in homes),
        }
    )
    if sum(row["zip_home_bytes"] for row in homes) != len(archive):
        raise DirectDescriptionError("v9 carrier unique-home accounting does not close")
    return members, tuple(homes)


def _apply_chart_symbols(
    layers: Sequence[StructuredRoleLayerV1],
    symbols: Sequence[LaneCoefficientDelta],
    *,
    coefficient_limit: int = 4,
) -> tuple[StructuredRoleLayerV1, ...]:
    copied = list(layers)
    lane_index = next((index for index, layer in enumerate(copied) if layer.role == "Lane"), None)
    if lane_index is None or copied[lane_index].lane_lines is None:
        raise DirectDescriptionError("v9 receiver lacks a decoded Lane chart")
    lines = [[np.asarray(value, dtype=np.float64).copy() for value in pair] for pair in copied[lane_index].lane_lines]
    for symbol in symbols:
        if symbol.pair_index >= len(lines) or symbol.line_index >= len(lines[symbol.pair_index]):
            raise DirectDescriptionError("G2CS1 address is absent from nested Lane chart")
        vector = lines[symbol.pair_index][symbol.line_index]
        if symbol.coefficient_index >= min(coefficient_limit, vector.size):
            raise DirectDescriptionError(
                "G2CS1 correction addresses a coefficient outside the permitted Lane centerline/width/phase set"
            )
        vector[symbol.coefficient_index] += symbol.coefficient_delta
        if not np.isfinite(vector).all():
            raise DirectDescriptionError("G2CS1 application produced nonfinite Lane coefficients")
    copied[lane_index] = replace(
        copied[lane_index],
        lane_lines=tuple(tuple(value for value in pair) for pair in lines),
    )
    return tuple(copied)


def _apply_boundary_coefficients(
    mask: np.ndarray,
    symbols: Sequence[BoundaryCoefficientDelta],
) -> np.ndarray:
    """Warp a counted carrier mask by a cubic normalized-x boundary chart."""

    if not symbols:
        return np.asarray(mask, dtype=bool)
    coefficients = np.zeros(4, dtype=np.float64)
    for symbol in symbols:
        coefficients[symbol.coefficient_index] += symbol.coefficient_delta
    height, width = mask.shape
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    displacement = np.rint(sum(coefficients[index] * x**index for index in range(4))).astype(np.int64)
    source_y = np.arange(height, dtype=np.int64)[:, None] - displacement[None, :]
    valid = (source_y >= 0) & (source_y < height)
    clipped_y = np.clip(source_y, 0, height - 1)
    source_x = np.broadcast_to(np.arange(width, dtype=np.int64)[None, :], (height, width))
    shifted = np.asarray(mask, dtype=bool)[clipped_y, source_x]
    shifted[~valid] = False
    return shifted


def _apply_boundary_shearlet_atoms(
    mask: np.ndarray,
    atoms: Sequence[BoundaryShearletAtomV1],
) -> np.ndarray:
    """Apply compact parabolic shearlet atoms as a synthesized displacement field."""

    result = np.asarray(mask, dtype=bool)
    if not atoms:
        return result
    height, width = result.shape
    yy = np.arange(height, dtype=np.float64)[:, None]
    xx = np.arange(width, dtype=np.float64)[None, :]
    source_x = np.broadcast_to(np.arange(width, dtype=np.int64)[None, :], (height, width))
    for atom in atoms:
        u = (xx - atom.center_x) / float(atom.scale_x)
        ridge = atom.center_y + (atom.shear_q4 / 16.0) * (xx - atom.center_x)
        v = (yy - ridge) / float(atom.scale_y)
        window = np.square(np.maximum(1.0 - np.square(u), 0.0)) * np.square(
            np.maximum(1.0 - np.square(v), 0.0)
        )
        displacement = np.rint((atom.amplitude_q4 / 16.0) * window).astype(np.int64)
        source_y = np.arange(height, dtype=np.int64)[:, None] - displacement
        valid = (source_y >= 0) & (source_y < height)
        shifted = result[np.clip(source_y, 0, height - 1), source_x]
        shifted[~valid] = False
        result = shifted
    return result


def _event_mask(
    event: TopologyEventV1,
    *,
    source_pair_id: int,
    source_pair_start: int,
    pose6_codes: np.ndarray,
) -> np.ndarray:
    """Rasterize one parametric topology event; target cells never enter."""

    if not event.pair_index <= source_pair_id < event.pair_index + event.lifetime:
        return np.zeros((384, 512), dtype=bool)
    birth_local = event.pair_index - source_pair_start
    current_local = source_pair_id - source_pair_start
    if not (0 <= birth_local < len(pose6_codes) and 0 <= current_local < len(pose6_codes)):
        raise DirectDescriptionError("topology-event Pose6 transport address escaped the local window")
    pose_delta = pose6_codes[current_local].astype(np.int16) - pose6_codes[birth_local].astype(np.int16)
    dx = int(np.rint(float(pose_delta[0]) * event.transport_gain_x_q4 / 16.0))
    dy = int(np.rint(float(pose_delta[1]) * event.transport_gain_y_q4 / 16.0))
    y0, y1 = event.y0 + dy, event.y1 + dy
    x0, x1 = event.x0 + dx, event.x1 + dx
    clipped_y0, clipped_y1 = max(0, y0), min(384, y1)
    clipped_x0, clipped_x1 = max(0, x0), min(512, x1)
    result = np.zeros((384, 512), dtype=bool)
    if clipped_y0 >= clipped_y1 or clipped_x0 >= clipped_x1:
        return result
    if event.shape == "box":
        result[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = True
        return result
    cy = (y0 + y1 - 1) / 2.0
    cx = (x0 + x1 - 1) / 2.0
    ry = max((y1 - y0) / 2.0, 0.5)
    rx = max((x1 - x0) / 2.0, 0.5)
    ys = np.arange(clipped_y0, clipped_y1, dtype=np.float64)[:, None]
    xs = np.arange(clipped_x0, clipped_x1, dtype=np.float64)[None, :]
    result[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = (
        np.square((ys - cy) / ry) + np.square((xs - cx) / rx) <= 1.0
    )
    return result


def _island_shape_mask(
    atom: IslandShapeAtomV1,
    *,
    source_pair_id: int,
    source_pair_start: int,
    pose6_codes: np.ndarray,
) -> np.ndarray:
    """Synthesize a moment-shaped island with one compact curvelet lobe."""

    if not atom.pair_index <= source_pair_id < atom.pair_index + atom.lifetime:
        return np.zeros((384, 512), dtype=bool)
    birth_local = atom.pair_index - source_pair_start
    current_local = source_pair_id - source_pair_start
    if not (0 <= birth_local < len(pose6_codes) and 0 <= current_local < len(pose6_codes)):
        raise DirectDescriptionError("island-shape Pose6 transport address escaped the local window")
    pose_delta = pose6_codes[current_local].astype(np.int16) - pose6_codes[birth_local].astype(np.int16)
    center_x = atom.center_x + int(np.rint(float(pose_delta[0]) * atom.transport_gain_x_q4 / 16.0))
    center_y = atom.center_y + int(np.rint(float(pose_delta[1]) * atom.transport_gain_y_q4 / 16.0))
    extent = max(atom.radius_x, atom.radius_y) * 2
    y0, y1 = max(0, center_y - extent), min(384, center_y + extent + 1)
    x0, x1 = max(0, center_x - extent), min(512, center_x + extent + 1)
    result = np.zeros((384, 512), dtype=bool)
    if y0 >= y1 or x0 >= x1:
        return result
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] - center_y
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] - center_x
    angle = atom.angle_u8 * np.pi / 256.0
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    u = (cosine * xs + sine * ys) / float(atom.radius_x)
    v = (-sine * xs + cosine * ys) / float(atom.radius_y)
    skew = atom.skew_q6 / 64.0
    taper = atom.taper_q6 / 64.0
    curvelet = atom.curvelet_q6 / 64.0
    tapered_radius = np.clip(1.0 + taper * v, 0.25, 2.0)
    shaped_u = (u - skew * v * np.maximum(1.0 - np.square(v), 0.0)) / tapered_radius
    compact_lobe = np.maximum(1.0 - 4.0 * np.square(u - 0.5), 0.0) * np.maximum(
        1.0 - np.square(v), 0.0
    )
    threshold = np.maximum(0.25, 1.0 + curvelet * compact_lobe)
    result[y0:y1, x0:x1] = np.square(shaped_u) + np.square(v) <= threshold
    return result


@dataclass(frozen=True, slots=True)
class CarrierComposeReceiverV1:
    archive: bytes
    predictor: ComposedStructuredMemberReceiverV1
    layers: tuple[StructuredRoleLayerV1, ...]
    symbols: tuple[LaneCoefficientDelta, ...]
    boundary_symbols: tuple[BoundaryCoefficientDelta, ...]
    topology_events: tuple[TopologyEventV1, ...]
    boundary_shearlets: tuple[BoundaryShearletAtomV1, ...]
    island_shapes: tuple[IslandShapeAtomV1, ...]
    custody: Mapping[str, Any]

    @property
    def z(self) -> Any:
        return self.predictor.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.predictor.pose6_codes

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        indexes = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in indexes):
            raise DirectDescriptionError("v9 receiver pair ID is outside its local window")
        output = self.predictor.baseline.render_pairs(indexes)
        for layer in self.layers:
            for local_index, pair_id in enumerate(indexes):
                source_pair_id = self.predictor.source_pair_start + pair_id
                mask = layer.mask(
                    local_pair_id=pair_id,
                    source_pair_id=source_pair_id,
                    camera=self.predictor.camera,
                )
                boundary = tuple(
                    row
                    for row in self.boundary_symbols
                    if row.pair_index == source_pair_id and row.role == layer.role
                )
                if boundary:
                    mask = _apply_boundary_coefficients(mask, boundary)
                shearlets = tuple(
                    row
                    for row in self.boundary_shearlets
                    if row.pair_index == source_pair_id and row.role == layer.role
                )
                if shearlets:
                    mask = _apply_boundary_shearlet_atoms(mask, shearlets)
                for event in self.topology_events:
                    if event.role != layer.role:
                        continue
                    event_sites = _event_mask(
                        event,
                        source_pair_id=source_pair_id,
                        source_pair_start=self.predictor.source_pair_start,
                        pose6_codes=self.pose6_codes,
                    )
                    if event.action == "birth":
                        mask |= event_sites
                    else:
                        mask &= ~event_sites
                if layer.role == "Movable":
                    for atom in self.island_shapes:
                        atom_sites = _island_shape_mask(
                            atom,
                            source_pair_id=source_pair_id,
                            source_pair_start=self.predictor.source_pair_start,
                            pose6_codes=self.pose6_codes,
                        )
                        if atom.action == "birth":
                            mask |= atom_sites
                        else:
                            mask &= ~atom_sites
                output[local_index, 0, mask] = layer.paint_rgb_u8
                output[local_index, 1, mask] = layer.paint_rgb_u8
        return np.ascontiguousarray(output)


def receive_carrier_compose_archive(archive: bytes) -> CarrierComposeReceiverV1:
    members, homes = parse_carrier_compose_archive(archive)
    manifest = json.loads(members["manifest.json"])
    predictor = receive_structured_member_archive(members["predictor.zip"])
    if not isinstance(predictor, ComposedStructuredMemberReceiverV1):
        raise DirectDescriptionError("nested predictor changed type after strict parse")
    symbols = decode_lane_coefficient_deltas(members.get(CORRECTION_MEMBER, b""))
    boundary_symbols = _decode_boundary_coefficient_deltas(members.get(BOUNDARY_CORRECTION_MEMBER, b""))
    topology_events = _decode_topology_events(members.get(EVENT_CORRECTION_MEMBER, b""))
    boundary_shearlets = _decode_boundary_shearlet_atoms(members.get(BOUNDARY_SHEARLET_MEMBER, b""))
    island_shapes = _decode_island_shape_atoms(members.get(ISLAND_SHAPE_MEMBER, b""))
    start = predictor.source_pair_start
    stop = start + predictor.z.n_pairs
    addressed = [*symbols, *boundary_symbols, *topology_events, *boundary_shearlets, *island_shapes]
    if any(row.pair_index < start or row.pair_index >= stop for row in addressed):
        raise DirectDescriptionError("correction symbol is outside the nested predictor source window")
    if any(row.pair_index + row.lifetime > stop for row in topology_events):
        raise DirectDescriptionError("topology-event lifetime escapes the nested predictor source window")
    if any(row.pair_index + row.lifetime > stop for row in island_shapes):
        raise DirectDescriptionError("island-shape lifetime escapes the nested predictor source window")
    coefficient_limit = 9 if manifest["schema"] == ARCHIVE_SCHEMA_V3 else 4
    layers = _apply_chart_symbols(predictor.layers, symbols, coefficient_limit=coefficient_limit)
    first = CarrierComposeReceiverV1(
        archive=archive,
        predictor=predictor,
        layers=layers,
        symbols=symbols,
        boundary_symbols=boundary_symbols,
        topology_events=topology_events,
        boundary_shearlets=boundary_shearlets,
        island_shapes=island_shapes,
        custody={},
    )

    lane_groups: dict[tuple[int, int], list[LaneCoefficientDelta]] = {}
    for symbol in symbols:
        lane_groups.setdefault((symbol.pair_index, symbol.line_index), []).append(symbol)
    for (pair_index, _line_index), group in lane_groups.items():
        local_pair_id = pair_index - predictor.source_pair_start
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=_apply_chart_symbols(predictor.layers, tuple(group), coefficient_limit=coefficient_limit),
            symbols=tuple(group),
            boundary_symbols=(),
            topology_events=(),
            boundary_shearlets=(),
            island_shapes=(),
            custody={},
        )
        if np.array_equal(predictor.render_pairs((local_pair_id,)), isolated.render_pairs((local_pair_id,))):
            raise DirectDescriptionError("G2CS1 line-coefficient group is a receiver-output no-op")

    boundary_groups: dict[tuple[int, str], list[BoundaryCoefficientDelta]] = {}
    for symbol in boundary_symbols:
        boundary_groups.setdefault((symbol.pair_index, symbol.role), []).append(symbol)
    for (pair_index, _role), group in boundary_groups.items():
        local_pair_id = pair_index - predictor.source_pair_start
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=predictor.layers,
            symbols=(),
            boundary_symbols=tuple(group),
            topology_events=(),
            boundary_shearlets=(),
            island_shapes=(),
            custody={},
        )
        if np.array_equal(predictor.render_pairs((local_pair_id,)), isolated.render_pairs((local_pair_id,))):
            raise DirectDescriptionError("boundary coefficient group is a receiver-output no-op")

    for event in topology_events:
        local_ids = tuple(range(event.pair_index - start, event.pair_index - start + event.lifetime))
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=predictor.layers,
            symbols=(),
            boundary_symbols=(),
            topology_events=(event,),
            boundary_shearlets=(),
            island_shapes=(),
            custody={},
        )
        if np.array_equal(predictor.render_pairs(local_ids), isolated.render_pairs(local_ids)):
            raise DirectDescriptionError("topology event is a receiver-output no-op")

    shearlet_groups: dict[tuple[int, str], list[BoundaryShearletAtomV1]] = {}
    for atom in boundary_shearlets:
        shearlet_groups.setdefault((atom.pair_index, atom.role), []).append(atom)
    for (pair_index, _role), group in shearlet_groups.items():
        local_pair_id = pair_index - predictor.source_pair_start
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=predictor.layers,
            symbols=(),
            boundary_symbols=(),
            topology_events=(),
            boundary_shearlets=tuple(group),
            island_shapes=(),
            custody={},
        )
        if np.array_equal(predictor.render_pairs((local_pair_id,)), isolated.render_pairs((local_pair_id,))):
            raise DirectDescriptionError("boundary-shearlet atom group is a receiver-output no-op")

    for atom in island_shapes:
        local_ids = tuple(range(atom.pair_index - start, atom.pair_index - start + atom.lifetime))
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=predictor.layers,
            symbols=(),
            boundary_symbols=(),
            topology_events=(),
            boundary_shearlets=(),
            island_shapes=(atom,),
            custody={},
        )
        if np.array_equal(predictor.render_pairs(local_ids), isolated.render_pairs(local_ids)):
            raise DirectDescriptionError("island-shape atom is a receiver-output no-op")
    probes = tuple(sorted({0, predictor.z.n_pairs - 1}))
    a = first.render_pairs(probes)
    b = first.render_pairs(probes)
    if not np.array_equal(a, b):
        raise DirectDescriptionError("carrier receiver replay is nondeterministic")
    custody = {
        "schema": (
            RECEIVER_SCHEMA_V3
            if manifest["schema"] == ARCHIVE_SCHEMA_V3
            else RECEIVER_SCHEMA_V2
            if manifest["schema"] == ARCHIVE_SCHEMA_V2
            else RECEIVER_SCHEMA
        ),
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "member_homes": list(homes),
        "all_archive_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
        "all_five_roles_consumed": [layer.role for layer in layers] == list(COMPOSED_ROLE_ORDER),
        "chart_symbol_count": len(symbols),
        "chart_symbol_parse_reencode_identical": encode_lane_coefficient_deltas(symbols)
        == members.get(CORRECTION_MEMBER, b""),
        "boundary_symbol_count": len(boundary_symbols),
        "boundary_symbol_parse_reencode_identical": _encode_boundary_coefficient_deltas(boundary_symbols)
        == members.get(BOUNDARY_CORRECTION_MEMBER, b""),
        "topology_event_count": len(topology_events),
        "topology_event_parse_reencode_identical": _encode_topology_events(topology_events)
        == members.get(EVENT_CORRECTION_MEMBER, b""),
        "topology_events_consume_counted_pose6_transport": any(row.lifetime > 1 for row in topology_events),
        "boundary_shearlet_count": len(boundary_shearlets),
        "boundary_shearlet_parse_reencode_identical": _encode_boundary_shearlet_atoms(boundary_shearlets)
        == members.get(BOUNDARY_SHEARLET_MEMBER, b""),
        "island_shape_count": len(island_shapes),
        "island_shape_parse_reencode_identical": _encode_island_shape_atoms(island_shapes)
        == members.get(ISLAND_SHAPE_MEMBER, b""),
        "island_shapes_consume_counted_pose6_transport": any(row.lifetime > 1 for row in island_shapes),
        "region_coherent_chart_rerasterization": True,
        "pixel_coordinate_or_rgb_patch_present": False,
        "nested_pose6_owner_reused": True,
        "deterministic_probe_replay": True,
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    return replace(first, custody=custody)


def recursive_carrier_byte_rows(archive: bytes) -> list[dict[str, Any]]:
    """Attribute nested semantic payload homes without double counting bytes."""

    members, outer_homes = parse_carrier_compose_archive(archive)
    predictor_members, predictor_homes = parse_structured_member_archive(members["predictor.zip"])
    home_by_name = {row["name"]: row for row in predictor_homes}
    chart = parse_entropy_chart_archive(predictor_members["chart.zip"])
    pose_row = next(row for row in chart.stream_byte_rows() if row["stream"] == "pose6_pair_codes")
    groups = {
        "Road": ("structure/road_pxq1_mask.br", "structure/road_events.lz", "structure/road_components.br"),
        "Lane": ("structure/lane_lbnd2.lz", "structure/lane_events.lz", "structure/lane_components.br"),
        "Undrivable": ("structure/undrivable_events.lz", "structure/undrivable_components.br"),
        "Movable": ("structure/movable_events.lz",),
        "MyCar": ("structure/mycar_static_hood.br",),
    }
    rows = [
        {
            "stratum": name,
            "nested_members": list(names),
            "nested_unique_home_bytes": sum(int(home_by_name[item]["zip_home_bytes"]) for item in names),
            "byte_authority": "exact nested ZIP home bytes; part of predictor.zip outer home",
        }
        for name, names in groups.items()
    ]
    rows.append(
        {
            "stratum": "xi/Pose6",
            "nested_members": ["chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin"],
            "nested_unique_home_bytes": int(pose_row["unique_final_zip_home_bytes"]),
            "byte_authority": "exact nested entropy-chart ZIP home bytes; sole Pose6 owner",
        }
    )
    correction_home = next((row for row in outer_homes if row["name"] == CORRECTION_MEMBER), None)
    rows.append(
        {
            "stratum": "chart_symbol_refinement",
            "nested_members": [] if correction_home is None else [CORRECTION_MEMBER],
            "nested_unique_home_bytes": 0 if correction_home is None else int(correction_home["zip_home_bytes"]),
            "byte_authority": "exact outer ZIP home bytes",
        }
    )
    for stratum, member_name in (
        ("road_boundary_coefficients", BOUNDARY_CORRECTION_MEMBER),
        ("xi_topology_events", EVENT_CORRECTION_MEMBER),
        ("boundary_shearlet_obligations", BOUNDARY_SHEARLET_MEMBER),
        ("movable_shape_obligations", ISLAND_SHAPE_MEMBER),
    ):
        correction_home = next((row for row in outer_homes if row["name"] == member_name), None)
        if correction_home is None:
            continue
        rows.append(
            {
                "stratum": stratum,
                "nested_members": [member_name],
                "nested_unique_home_bytes": int(correction_home["zip_home_bytes"]),
                "byte_authority": "exact outer ZIP home bytes",
            }
        )
    return rows


def prove_carrier_archive_fail_closed(archive: bytes) -> dict[str, Any]:
    """Sample every outer home: a mutation must refuse or alter decoded RGB."""

    baseline = receive_carrier_compose_archive(archive)
    probe_ids = tuple(sorted({0, baseline.z.n_pairs - 1}))
    digest = hashlib.sha256(baseline.render_pairs(probe_ids).tobytes()).hexdigest()
    _members, homes = parse_carrier_compose_archive(archive)
    positions: list[int] = []
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        infos = reader.infolist()
        for info in infos:
            if info.file_size:
                positions.append(
                    info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + info.file_size // 2
                )
    refused = changed = 0
    for position in positions:
        altered = bytearray(archive)
        altered[position] ^= 1
        try:
            candidate = receive_carrier_compose_archive(bytes(altered))
        except (DirectDescriptionError, OSError, ValueError, zipfile.BadZipFile):
            refused += 1
            continue
        candidate_digest = hashlib.sha256(candidate.render_pairs(probe_ids).tobytes()).hexdigest()
        if candidate_digest == digest:
            raise DirectDescriptionError("sampled archive mutation was accepted as a receiver no-op")
        changed += 1
    return {
        "sampled_member_payload_homes": len(positions),
        "refused": refused,
        "changed_decode": changed,
        "all_samples_refused_or_changed_decode": refused + changed == len(positions),
        "unique_home_coverage_bytes": sum(row["zip_home_bytes"] for row in homes),
    }


__all__ = [
    "ARCHIVE_SCHEMA",
    "ARCHIVE_SCHEMA_V2",
    "ARCHIVE_SCHEMA_V3",
    "BOUNDARY_SHEARLET_MEMBER",
    "ISLAND_SHAPE_MEMBER",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_V2",
    "RESULT_SCHEMA_V3",
    "RESULT_SCHEMA_V4",
    "BoundaryCoefficientDelta",
    "BoundaryShearletAtomV1",
    "CarrierComposeReceiverV1",
    "DirectDescriptionV9CarrierComposeConfigV1",
    "DirectDescriptionV10FisherEventSearchConfigV1",
    "DirectDescriptionV11ObligationSearchConfigV1",
    "DirectDescriptionV12ObligationDrainConfigV1",
    "IslandShapeAtomV1",
    "TopologyEventV1",
    "compile_carrier_compose_archive",
    "parse_carrier_compose_archive",
    "prove_carrier_archive_fail_closed",
    "receive_carrier_compose_archive",
    "recursive_carrier_byte_rows",
]
