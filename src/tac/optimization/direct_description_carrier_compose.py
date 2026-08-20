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
import lzma
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
from tac.optimization.direct_description_g1_worldsheet import decode_g1_movable_worldsheet
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
ARCHIVE_SCHEMA_V4: Final = "direct_description_v13_worldsheet_predictor_archive.v1"
ARCHIVE_SCHEMA_V5: Final = "direct_description_v14_realization_fidelity_archive.v1"
ARCHIVE_SCHEMA_V6: Final = "direct_description_v15_scorer_solved_template_archive.v1"
RECEIVER_SCHEMA: Final = "direct_description_v9_carrier_compose_receiver.v1"
RECEIVER_SCHEMA_V2: Final = "direct_description_v10_fisher_event_receiver.v1"
RECEIVER_SCHEMA_V3: Final = "direct_description_v11_obligation_receiver.v1"
RECEIVER_SCHEMA_V4: Final = "direct_description_v13_worldsheet_predictor_receiver.v1"
RECEIVER_SCHEMA_V5: Final = "direct_description_v14_realization_fidelity_receiver.v1"
RECEIVER_SCHEMA_V6: Final = "direct_description_v15_scorer_solved_template_receiver.v1"
RESULT_SCHEMA: Final = "direct_description_v9_carrier_compose_receipt.v1"
RESULT_SCHEMA_V2: Final = "direct_description_v10_fisher_event_search_receipt.v1"
RESULT_SCHEMA_V3: Final = "direct_description_v11_obligation_search_receipt.v1"
RESULT_SCHEMA_V4: Final = "direct_description_v12_obligation_drain_receipt.v1"
RESULT_SCHEMA_V5: Final = "direct_description_v13_worldsheet_predictor_receipt.v1"
MAGIC: Final = "DDV9C1"
MAGIC_V2: Final = "DDV10C1"
MAGIC_V3: Final = "DDV11C1"
MAGIC_V4: Final = "DDV13P1"
MAGIC_V5: Final = "DDV14R1"
MAGIC_V6: Final = "DDV15S1"
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
WORLDSHEET_TRACK_MEMBER: Final = "predict/movable_worldsheet_tracks.ddwt"
WORLDSHEET_KNOT_MEMBER: Final = "predict/movable_worldsheet_knots.ddwk"
WORLDSHEET_G1_MEMBER: Final = "predict/movable_polygon_worldsheet.g1s"
LANE_PROGRAM_MEMBER: Final = "predict/lane_periodic_programs.ddlp"
LANE_KNOT_MEMBER: Final = "predict/lane_drift_knots.ddlk"
REALIZATION_PROFILE_MEMBER: Final = "render/receiver_realization.ddrp"
REALIZATION_STATIC_RULE_MEMBER: Final = "render/static_cell_rule.g4sr"
SCORER_SOLVED_TEMPLATE_MEMBER: Final = "render/scorer_solved_templates.ddst"
REALIZATION_PAINT_ORDER: Final = ("UndrivableBoundary", "Road", "Lane", "Movable", "MyCar")
_REALIZATION_MAGIC: Final = b"DDRP1"
_REALIZATION_VERSION: Final = 1
_STATIC_RULE_CODEC_RAW: Final = 0
_STATIC_RULE_CODEC_LZMA_RAW: Final = 1
_STATIC_RULE_IDS: Final = {
    "movable_midband_parametric": b"G4MB",
    "horizon_row_parametric": b"G4HR",
    "static_image_sparse_all": b"G4SR",
    "mycar_static_mask": b"G4HM",
}
_STATIC_RULE_LZMA_FILTERS: Final = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]

_SOLVED_TEMPLATE_MAGIC: Final = b"DDST1"
_SOLVED_TEMPLATE_VERSION: Final = 1
_SOLVED_TEMPLATE_HEADER: Final = struct.Struct(">5sBH")
_SOLVED_TEMPLATE_ROW: Final = struct.Struct(">BBHHBBH")
_SOLVED_TEMPLATE_APPLICATION_TO_WIRE: Final = {"fill": 0, "inner_boundary": 1}
_SOLVED_TEMPLATE_WIRE_TO_APPLICATION: Final = {
    value: key for key, value in _SOLVED_TEMPLATE_APPLICATION_TO_WIRE.items()
}
_MAX_SOLVED_TEMPLATE_PAYLOAD_BYTES: Final = 65_536

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
_WORLDSHEET_TRACK_MAGIC: Final = b"DDWT1"
_WORLDSHEET_TRACK_HEADER: Final = struct.Struct(">5sBHI")
_WORLDSHEET_TRACK_ROW: Final = struct.Struct(">HHHHHBHBbbbbb")
_WORLDSHEET_KNOT_MAGIC: Final = b"DDWK1"
_WORLDSHEET_KNOT_HEADER: Final = struct.Struct(">5sBHI")
_WORLDSHEET_KNOT_ROW: Final = struct.Struct(">HHhhhhhhbbb")
_LANE_PROGRAM_MAGIC: Final = b"DDLP1"
_LANE_PROGRAM_HEADER: Final = struct.Struct(">5sBHI")
_LANE_PROGRAM_ROW: Final = struct.Struct(">BHHhhhh")
_LANE_KNOT_MAGIC: Final = b"DDLK1"
_LANE_KNOT_HEADER: Final = struct.Struct(">5sBHI")
_LANE_KNOT_ROW: Final = struct.Struct(">BHhhhhhh")


@dataclass(frozen=True, slots=True)
class ReceiverRealizationProfileV1:
    """Counted camera-resolution semantic-paint contract.

    Colours are ordered by :data:`REALIZATION_PAINT_ORDER`.  The profile is a
    few-byte template, never a pixel/RGB patch: generic nearest coverage and
    semantic compositing remain free receiver logic.
    """

    role_rgb_u8: tuple[tuple[int, int, int], ...]
    coverage_radius: int = 0
    amplitude_u8: int = 255

    def __post_init__(self) -> None:
        if len(self.role_rgb_u8) != len(REALIZATION_PAINT_ORDER):
            raise DirectDescriptionError("realization profile must bind exactly five role colours")
        if any(
            len(row) != 3 or any(isinstance(value, bool) or not 0 <= value <= 255 for value in row)
            for row in self.role_rgb_u8
        ):
            raise DirectDescriptionError("realization profile colours must be uint8 RGB triples")
        if self.coverage_radius != 0:
            raise DirectDescriptionError("measured v14 realization profile requires zero coverage expansion")
        if self.amplitude_u8 != 255:
            raise DirectDescriptionError("measured v14 realization profile requires the full uint8 amplitude floor")

    def colour_for(self, role: str) -> np.ndarray:
        try:
            index = REALIZATION_PAINT_ORDER.index(role)
        except ValueError as exc:
            raise DirectDescriptionError(f"realization profile role is unknown: {role!r}") from exc
        return np.asarray(self.role_rgb_u8[index], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class RowBandScorerTemplateV1:
    """One counted RGB patch shared by a semantic role and scorer-row band.

    The template is solved encode-side against the frozen scorer. Decode only
    tiles these explicit uint8 bytes over a grammar-derived semantic mask; no
    scorer weights, logits, gradients, or ground-truth table cross the wire.
    """

    role: str
    application: str
    scorer_row_start: int
    scorer_row_stop: int
    patch_height: int
    patch_width: int
    rgb_u8: bytes

    def __post_init__(self) -> None:
        if self.role not in REALIZATION_PAINT_ORDER:
            raise DirectDescriptionError(f"scorer template role is unknown: {self.role!r}")
        if self.application not in _SOLVED_TEMPLATE_APPLICATION_TO_WIRE:
            raise DirectDescriptionError(f"scorer template application is unsupported: {self.application!r}")
        if not 0 <= self.scorer_row_start < self.scorer_row_stop <= 384:
            raise DirectDescriptionError("scorer template row band must be inside [0,384]")
        if not 1 <= self.patch_height <= 8 or not 1 <= self.patch_width <= 8:
            raise DirectDescriptionError("scorer template patch dimensions must be in [1,8]")
        expected = self.patch_height * self.patch_width * 3
        if len(self.rgb_u8) != expected:
            raise DirectDescriptionError(
                f"scorer template RGB payload expected {expected} bytes, got {len(self.rgb_u8)}"
            )

    def patch(self) -> np.ndarray:
        return np.frombuffer(self.rgb_u8, dtype=np.uint8).reshape(self.patch_height, self.patch_width, 3)


def _solved_template_sort_key(row: RowBandScorerTemplateV1) -> tuple[int, int, int, int, int, int]:
    return (
        REALIZATION_PAINT_ORDER.index(row.role),
        _SOLVED_TEMPLATE_APPLICATION_TO_WIRE[row.application],
        row.scorer_row_start,
        row.scorer_row_stop,
        row.patch_height,
        row.patch_width,
    )


@dataclass(frozen=True, slots=True)
class ScorerSolvedTemplateBankV1:
    """Canonical row-band template bank consumed by the existing V14 renderer."""

    templates: tuple[RowBandScorerTemplateV1, ...]

    def __post_init__(self) -> None:
        if not self.templates:
            raise DirectDescriptionError("scorer-solved template bank must not be empty")
        if len(self.templates) > 64:
            raise DirectDescriptionError("scorer-solved template bank exceeds 64 records")
        ordered = tuple(sorted(self.templates, key=_solved_template_sort_key))
        if ordered != self.templates:
            raise DirectDescriptionError("scorer-solved template records are not canonical-order")
        previous: dict[tuple[str, str], int] = {}
        for row in self.templates:
            key = (row.role, row.application)
            if row.scorer_row_start < previous.get(key, 0):
                raise DirectDescriptionError("scorer template row bands overlap")
            previous[key] = row.scorer_row_stop

    def for_role(self, role: str) -> tuple[RowBandScorerTemplateV1, ...]:
        return tuple(row for row in self.templates if row.role == role)


def _encode_realization_profile(profile: ReceiverRealizationProfileV1 | None) -> bytes:
    if profile is None:
        return b""
    channels = [value for row in profile.role_rgb_u8 for value in row]
    return _REALIZATION_MAGIC + bytes([_REALIZATION_VERSION, *channels, profile.coverage_radius, profile.amplitude_u8])


def _decode_realization_profile(payload: bytes) -> ReceiverRealizationProfileV1 | None:
    if not payload:
        return None
    expected = len(_REALIZATION_MAGIC) + 1 + 15 + 2
    if len(payload) != expected or payload[: len(_REALIZATION_MAGIC)] != _REALIZATION_MAGIC:
        raise DirectDescriptionError("receiver realization profile header/length is invalid")
    body = payload[len(_REALIZATION_MAGIC) :]
    if body[0] != _REALIZATION_VERSION:
        raise DirectDescriptionError("receiver realization profile version is unsupported")
    channels = body[1:16]
    colours = tuple(tuple(channels[3 * index : 3 * index + 3]) for index in range(5))
    profile = ReceiverRealizationProfileV1(
        role_rgb_u8=colours,
        coverage_radius=body[16],
        amplitude_u8=body[17],
    )
    if _encode_realization_profile(profile) != payload:
        raise DirectDescriptionError("receiver realization profile parse/re-encode changed bytes")
    return profile


def encode_scorer_solved_template_bank(bank: ScorerSolvedTemplateBankV1 | None) -> bytes:
    """Encode a bounded canonical template bank for counted archive storage."""

    if bank is None:
        return b""
    body = bytearray(
        _SOLVED_TEMPLATE_HEADER.pack(_SOLVED_TEMPLATE_MAGIC, _SOLVED_TEMPLATE_VERSION, len(bank.templates))
    )
    for row in bank.templates:
        rgb = bytes(row.rgb_u8)
        body.extend(
            _SOLVED_TEMPLATE_ROW.pack(
                _ROLE_TO_WIRE[row.role],
                _SOLVED_TEMPLATE_APPLICATION_TO_WIRE[row.application],
                row.scorer_row_start,
                row.scorer_row_stop,
                row.patch_height,
                row.patch_width,
                len(rgb),
            )
        )
        body.extend(rgb)
    if len(body) > _MAX_SOLVED_TEMPLATE_PAYLOAD_BYTES:
        raise DirectDescriptionError("scorer-solved template payload exceeds 65536 bytes")
    return bytes(body)


def decode_scorer_solved_template_bank(payload: bytes) -> ScorerSolvedTemplateBankV1 | None:
    """Strictly decode and byte-roundtrip a counted template bank."""

    if not payload:
        return None
    if len(payload) > _MAX_SOLVED_TEMPLATE_PAYLOAD_BYTES or len(payload) < _SOLVED_TEMPLATE_HEADER.size:
        raise DirectDescriptionError("scorer-solved template payload length is invalid")
    magic, version, count = _SOLVED_TEMPLATE_HEADER.unpack_from(payload)
    if magic != _SOLVED_TEMPLATE_MAGIC or version != _SOLVED_TEMPLATE_VERSION or not 1 <= count <= 64:
        raise DirectDescriptionError("scorer-solved template header is invalid")
    cursor = _SOLVED_TEMPLATE_HEADER.size
    rows: list[RowBandScorerTemplateV1] = []
    for _ in range(count):
        if cursor + _SOLVED_TEMPLATE_ROW.size > len(payload):
            raise DirectDescriptionError("scorer-solved template record is truncated")
        role_wire, application_wire, y0, y1, patch_h, patch_w, rgb_bytes = _SOLVED_TEMPLATE_ROW.unpack_from(
            payload, cursor
        )
        cursor += _SOLVED_TEMPLATE_ROW.size
        if cursor + rgb_bytes > len(payload):
            raise DirectDescriptionError("scorer-solved template RGB bytes are truncated")
        try:
            role = _WIRE_TO_ROLE[role_wire]
            application = _SOLVED_TEMPLATE_WIRE_TO_APPLICATION[application_wire]
        except KeyError as exc:
            raise DirectDescriptionError("scorer-solved template enum is invalid") from exc
        rows.append(
            RowBandScorerTemplateV1(
                role=role,
                application=application,
                scorer_row_start=y0,
                scorer_row_stop=y1,
                patch_height=patch_h,
                patch_width=patch_w,
                rgb_u8=payload[cursor : cursor + rgb_bytes],
            )
        )
        cursor += rgb_bytes
    if cursor != len(payload):
        raise DirectDescriptionError("scorer-solved template payload has trailing bytes")
    bank = ScorerSolvedTemplateBankV1(tuple(rows))
    if encode_scorer_solved_template_bank(bank) != payload:
        raise DirectDescriptionError("scorer-solved template parse/re-encode changed bytes")
    return bank


def _read_uleb128(payload: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise DirectDescriptionError("static rule ULEB128 is truncated")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise DirectDescriptionError("static rule ULEB128 is overlong")


def encode_static_class_mask_rule(
    mask: np.ndarray,
    *,
    target_class: int,
) -> bytes:
    """Encode one frame-shared class mask as a counted receiver rule.

    The target class is an encode-side, self-detected result and is carried in
    the payload; the decoder never assumes a fixed class index. Generic
    unpacking and painting stay in the free runtime.
    """

    value = np.asarray(mask, dtype=bool)
    if value.shape != (384, 512):
        raise DirectDescriptionError("static class mask geometry differs")
    if (
        isinstance(target_class, (bool, np.bool_))
        or not isinstance(target_class, (int, np.integer))
        or not 0 <= int(target_class) < len(CLASS_ORDER)
    ):
        raise DirectDescriptionError("static class mask target is invalid")
    packed = np.packbits(value.reshape(-1), bitorder="little").tobytes()
    raw = (
        struct.pack(
            ">4sBBHH",
            _STATIC_RULE_IDS["mycar_static_mask"],
            1,
            int(target_class),
            384,
            512,
        )
        + packed
    )
    coded = lzma.compress(
        raw,
        format=lzma.FORMAT_RAW,
        filters=_STATIC_RULE_LZMA_FILTERS,
    )
    payload = bytes([_STATIC_RULE_CODEC_LZMA_RAW]) + coded
    if _decode_realization_static_rule(payload, "mycar_static_mask") is None:
        raise DirectDescriptionError("static class mask self-check failed")
    return payload


def _decode_realization_static_rule(
    payload: bytes,
    opportunity_id: str | None,
) -> np.ndarray | None:
    """Decode one G4 one-time rule into a 384x512 source-to-target code field."""

    if not payload:
        if opportunity_id is not None:
            raise DirectDescriptionError("static rule identifier has no payload")
        return None
    if opportunity_id not in _STATIC_RULE_IDS:
        raise DirectDescriptionError("static rule opportunity identifier is unsupported")
    codec, coded = payload[0], payload[1:]
    if codec == _STATIC_RULE_CODEC_RAW:
        raw = coded
    elif codec == _STATIC_RULE_CODEC_LZMA_RAW:
        try:
            raw = lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=_STATIC_RULE_LZMA_FILTERS)
        except lzma.LZMAError as exc:
            raise DirectDescriptionError("static rule raw-LZMA payload is invalid") from exc
    else:
        raise DirectDescriptionError("static rule codec tag is unsupported")
    if raw[:4] != _STATIC_RULE_IDS[opportunity_id]:
        raise DirectDescriptionError("static rule magic differs from its opportunity identifier")
    codes = np.full((384, 512), -1, dtype=np.int16)
    if opportunity_id == "mycar_static_mask":
        header = struct.Struct(">4sBBHH")
        packed_bytes = (384 * 512 + 7) // 8
        if len(raw) != header.size + packed_bytes:
            raise DirectDescriptionError("MyCar static-mask rule length is invalid")
        _magic, version, target_class, height, width = header.unpack_from(raw)
        if (
            version != 1
            or (height, width) != (384, 512)
            or target_class >= len(CLASS_ORDER)
        ):
            raise DirectDescriptionError("MyCar static-mask rule geometry is invalid")
        mask = np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8, offset=header.size),
            bitorder="little",
            count=384 * 512,
        ).reshape(384, 512)
        codes[np.asarray(mask, dtype=bool)] = 25 + int(target_class)
    elif opportunity_id == "movable_midband_parametric":
        row = struct.Struct(">4sBHHBB")
        if len(raw) != row.size:
            raise DirectDescriptionError("Movable-midband static rule length is invalid")
        _magic, version, start, stop, source, target = row.unpack(raw)
        if version != 1 or (start, stop, source, target) != (174, 215, 1, 0):
            raise DirectDescriptionError("Movable-midband static rule geometry is invalid")
        codes[start : stop + 1] = source * 5 + target
    elif opportunity_id == "horizon_row_parametric":
        row = struct.Struct(">4sBHBBBB")
        if len(raw) != row.size:
            raise DirectDescriptionError("horizon static rule length is invalid")
        _magic, version, center, halfwidth, source, target, reserved = row.unpack(raw)
        if version != 1 or (center, halfwidth, source, target, reserved) != (212, 4, 2, 0, 0):
            raise DirectDescriptionError("horizon static rule geometry is invalid")
        codes[max(0, center - halfwidth) : min(384, center + halfwidth + 1)] = source * 5 + target
    else:
        header = struct.Struct(">4sBHHI")
        if len(raw) < header.size:
            raise DirectDescriptionError("sparse static rule header is truncated")
        _magic, version, height, width, count = header.unpack_from(raw)
        if version != 1 or (height, width, count) != (384, 512, 19_661):
            raise DirectDescriptionError("sparse static rule header is invalid")
        offset = header.size
        previous = -1
        flat = codes.reshape(-1)
        for _ in range(count):
            delta, offset = _read_uleb128(raw, offset)
            index = previous + 1 + delta
            if index >= flat.size or offset >= len(raw):
                raise DirectDescriptionError("sparse static rule row is out of bounds")
            flat[index] = raw[offset]
            offset += 1
            previous = index
        if offset != len(raw):
            raise DirectDescriptionError("sparse static rule has trailing bytes")
    active = codes >= 0
    transition = active & (codes < 25)
    wildcard = active & (codes >= 25) & (codes < 30)
    if (
        np.any(codes[active] >= 30)
        or np.any(codes[transition] // 5 == codes[transition] % 5)
        or np.any(active & ~(transition | wildcard))
    ):
        raise DirectDescriptionError("static rule transition codes are invalid")
    return np.ascontiguousarray(codes)


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


def requires_pose6_transport(atom: TopologyEventV1 | IslandShapeAtomV1) -> bool:
    """Return whether an event/island receiver output can depend on Pose6.

    Transport dependence is carried by the two quantized gains, not by the
    nominal primitive family or its lifetime.  Keeping this predicate next to
    the primitive definitions gives admission and rasterization one shared
    source of truth.
    """

    if type(atom) not in {TopologyEventV1, IslandShapeAtomV1}:
        raise DirectDescriptionError("Pose6 dependency requires an exact topology-event or island-shape atom")
    return atom.transport_gain_x_q4 != 0 or atom.transport_gain_y_q4 != 0


@dataclass(frozen=True, order=True, slots=True)
class MovableWorldsheetTrackV1:
    """One persist-unless-event Movable object in the PREDICT grammar.

    Birth and death are lifecycle productions.  The initial contour uses the
    already-governed island-carrier moment/curvelet coordinates.  Between
    events the receiver rides the sole counted Pose6/xi stream; instance bytes
    store only sparse deviations and low-order morphs in
    :class:`MovableWorldsheetKnotV1`.
    """

    object_id: int
    birth_pair: int
    death_pair_exclusive: int
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
        if isinstance(self.object_id, bool) or not 0 <= self.object_id <= 0xFFFF:
            raise DirectDescriptionError("worldsheet object ID is outside uint16")
        if not 0 <= self.birth_pair < self.death_pair_exclusive <= 600:
            raise DirectDescriptionError("worldsheet birth/death interval is invalid")
        if not (0 <= self.center_y < 384 and 0 <= self.center_x < 512):
            raise DirectDescriptionError("worldsheet initial center is outside scorer geometry")
        if not (1 <= self.radius_y <= 191 and 1 <= self.radius_x <= 255):
            raise DirectDescriptionError("worldsheet initial radius is outside scorer geometry")
        if not 0 <= self.angle_u8 <= 255:
            raise DirectDescriptionError("worldsheet angle is outside uint8")
        for name in ("skew_q6", "taper_q6", "curvelet_q6"):
            if not -96 <= getattr(self, name) <= 96:
                raise DirectDescriptionError(f"worldsheet {name} is outside the stable q6 range")
        for name in ("transport_gain_x_q4", "transport_gain_y_q4"):
            if not -128 <= getattr(self, name) <= 127:
                raise DirectDescriptionError(f"worldsheet {name} is outside int8")


@dataclass(frozen=True, order=True, slots=True)
class MovableWorldsheetKnotV1:
    """Sparse deviation from xi transport plus low-order contour morph."""

    object_id: int
    pair_index: int
    delta_center_y_q4: int = 0
    delta_center_x_q4: int = 0
    delta_radius_y_q4: int = 0
    delta_radius_x_q4: int = 0
    delta_angle_q4: int = 0
    reserved_q4: int = 0
    delta_skew_q6: int = 0
    delta_taper_q6: int = 0
    delta_curvelet_q6: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.object_id, bool) or not 0 <= self.object_id <= 0xFFFF:
            raise DirectDescriptionError("worldsheet-knot object ID is outside uint16")
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("worldsheet-knot pair index is outside [0,600)")
        for name in (
            "delta_center_y_q4",
            "delta_center_x_q4",
            "delta_radius_y_q4",
            "delta_radius_x_q4",
            "delta_angle_q4",
        ):
            if not -32768 <= getattr(self, name) <= 32767:
                raise DirectDescriptionError(f"worldsheet-knot {name} is outside int16")
        if self.reserved_q4 != 0:
            raise DirectDescriptionError("worldsheet-knot reserved field must be zero")
        for name in ("delta_skew_q6", "delta_taper_q6", "delta_curvelet_q6"):
            if not -128 <= getattr(self, name) <= 127:
                raise DirectDescriptionError(f"worldsheet-knot {name} is outside int8")
        if not any(
            getattr(self, name)
            for name in (
                "delta_center_y_q4",
                "delta_center_x_q4",
                "delta_radius_y_q4",
                "delta_radius_x_q4",
                "delta_angle_q4",
                "delta_skew_q6",
                "delta_taper_q6",
                "delta_curvelet_q6",
            )
        ):
            raise DirectDescriptionError("worldsheet knot must carry a nonzero deviation or morph")


@dataclass(frozen=True, order=True, slots=True)
class LanePeriodicProgramV1:
    """One Lane object with a single xi-advanced dash phase production."""

    line_index: int
    birth_pair: int
    death_pair_exclusive: int
    dash_phase_origin_delta_q8: int
    dash_phase_xi_gain_q8: int
    width_bias_q8: int
    width_slope_q12: int

    def __post_init__(self) -> None:
        if isinstance(self.line_index, bool) or not 0 <= self.line_index <= 255:
            raise DirectDescriptionError("lane-program line index is outside uint8")
        if not 0 <= self.birth_pair < self.death_pair_exclusive <= 600:
            raise DirectDescriptionError("lane-program visibility interval is invalid")
        for name in (
            "dash_phase_origin_delta_q8",
            "dash_phase_xi_gain_q8",
            "width_bias_q8",
            "width_slope_q12",
        ):
            if not -32768 <= getattr(self, name) <= 32767:
                raise DirectDescriptionError(f"lane-program {name} is outside int16")


@dataclass(frozen=True, order=True, slots=True)
class LaneDriftKnotV1:
    """Sparse polynomial/width/phase deviation for one coherent Lane object."""

    line_index: int
    pair_index: int
    center_c0_delta_q24: int = 0
    center_c1_delta_q18: int = 0
    center_c2_delta_q12: int = 0
    center_c3_delta_q8: int = 0
    width_delta_q8: int = 0
    phase_delta_q8: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.line_index, bool) or not 0 <= self.line_index <= 255:
            raise DirectDescriptionError("lane-knot line index is outside uint8")
        if isinstance(self.pair_index, bool) or not 0 <= self.pair_index < 600:
            raise DirectDescriptionError("lane-knot pair index is outside [0,600)")
        for name in (
            "center_c0_delta_q24",
            "center_c1_delta_q18",
            "center_c2_delta_q12",
            "center_c3_delta_q8",
            "width_delta_q8",
            "phase_delta_q8",
        ):
            if not -32768 <= getattr(self, name) <= 32767:
                raise DirectDescriptionError(f"lane-knot {name} is outside int16")
        if not any(
            getattr(self, name)
            for name in (
                "center_c0_delta_q24",
                "center_c1_delta_q18",
                "center_c2_delta_q12",
                "center_c3_delta_q8",
                "width_delta_q8",
                "phase_delta_q8",
            )
        ):
            raise DirectDescriptionError("lane drift knot must carry a nonzero derivation")


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
    return _BOUNDARY_HEADER.pack(_BOUNDARY_MAGIC, _BOUNDARY_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF) + body


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
        pair_index, role_id, coefficient_index, delta = _BOUNDARY_ROW.unpack_from(body, index * _BOUNDARY_ROW.size)
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
        if (
            role_id not in _WIRE_TO_ROLE
            or action_id not in _WIRE_TO_EVENT_ACTION
            or shape_id not in _WIRE_TO_EVENT_SHAPE
        ):
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
    return _SHEARLET_HEADER.pack(_SHEARLET_MAGIC, _SHEARLET_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF) + body


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
        pair, role_id, cy, cx, sy, sx, shear, amplitude = _SHEARLET_ROW.unpack_from(body, index * _SHEARLET_ROW.size)
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
    return _ISLAND_HEADER.pack(_ISLAND_MAGIC, _ISLAND_VERSION, len(rows), zlib.crc32(body) & 0xFFFFFFFF) + body


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


def _encode_fixed_rows(
    rows: Sequence[Any],
    *,
    magic: bytes,
    header: struct.Struct,
    row_struct: struct.Struct,
    values: Any,
    keys: Any,
    label: str,
) -> bytes:
    ordered = tuple(rows)
    if not ordered:
        return b""
    addresses = [keys(row) for row in ordered]
    if len(ordered) > 0xFFFF or addresses != sorted(set(addresses)):
        raise DirectDescriptionError(f"{label} rows must be sorted, unique, and uint16-bounded")
    body = b"".join(row_struct.pack(*values(row)) for row in ordered)
    return header.pack(magic, 1, len(ordered), zlib.crc32(body) & 0xFFFFFFFF) + body


def _decode_fixed_rows(
    payload: bytes,
    *,
    magic: bytes,
    header: struct.Struct,
    row_struct: struct.Struct,
    factory: Any,
    encoder: Any,
    label: str,
) -> tuple[Any, ...]:
    if not payload:
        return ()
    if len(payload) < header.size:
        raise DirectDescriptionError(f"{label} packet is truncated")
    found_magic, version, count, checksum = header.unpack_from(payload)
    expected = header.size + count * row_struct.size
    if found_magic != magic or version != 1 or count == 0 or len(payload) != expected:
        raise DirectDescriptionError(f"{label} packet header/length is invalid")
    body = payload[header.size :]
    if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
        raise DirectDescriptionError(f"{label} packet CRC mismatch")
    rows = tuple(factory(*row_struct.unpack_from(body, index * row_struct.size)) for index in range(count))
    if encoder(rows) != payload:
        raise DirectDescriptionError(f"{label} packet is not canonical on parse-back")
    return rows


def _encode_worldsheet_tracks(rows: Sequence[MovableWorldsheetTrackV1]) -> bytes:
    return _encode_fixed_rows(
        rows,
        magic=_WORLDSHEET_TRACK_MAGIC,
        header=_WORLDSHEET_TRACK_HEADER,
        row_struct=_WORLDSHEET_TRACK_ROW,
        values=lambda row: (
            row.object_id,
            row.birth_pair,
            row.death_pair_exclusive,
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
        ),
        keys=lambda row: row.object_id,
        label="worldsheet-track",
    )


def _decode_worldsheet_tracks(payload: bytes) -> tuple[MovableWorldsheetTrackV1, ...]:
    return _decode_fixed_rows(
        payload,
        magic=_WORLDSHEET_TRACK_MAGIC,
        header=_WORLDSHEET_TRACK_HEADER,
        row_struct=_WORLDSHEET_TRACK_ROW,
        factory=MovableWorldsheetTrackV1,
        encoder=_encode_worldsheet_tracks,
        label="worldsheet-track",
    )


def _encode_worldsheet_knots(rows: Sequence[MovableWorldsheetKnotV1]) -> bytes:
    return _encode_fixed_rows(
        rows,
        magic=_WORLDSHEET_KNOT_MAGIC,
        header=_WORLDSHEET_KNOT_HEADER,
        row_struct=_WORLDSHEET_KNOT_ROW,
        values=lambda row: (
            row.object_id,
            row.pair_index,
            row.delta_center_y_q4,
            row.delta_center_x_q4,
            row.delta_radius_y_q4,
            row.delta_radius_x_q4,
            row.delta_angle_q4,
            row.reserved_q4,
            row.delta_skew_q6,
            row.delta_taper_q6,
            row.delta_curvelet_q6,
        ),
        keys=lambda row: (row.object_id, row.pair_index),
        label="worldsheet-knot",
    )


def _decode_worldsheet_knots(payload: bytes) -> tuple[MovableWorldsheetKnotV1, ...]:
    return _decode_fixed_rows(
        payload,
        magic=_WORLDSHEET_KNOT_MAGIC,
        header=_WORLDSHEET_KNOT_HEADER,
        row_struct=_WORLDSHEET_KNOT_ROW,
        factory=MovableWorldsheetKnotV1,
        encoder=_encode_worldsheet_knots,
        label="worldsheet-knot",
    )


def _encode_lane_programs(rows: Sequence[LanePeriodicProgramV1]) -> bytes:
    return _encode_fixed_rows(
        rows,
        magic=_LANE_PROGRAM_MAGIC,
        header=_LANE_PROGRAM_HEADER,
        row_struct=_LANE_PROGRAM_ROW,
        values=lambda row: (
            row.line_index,
            row.birth_pair,
            row.death_pair_exclusive,
            row.dash_phase_origin_delta_q8,
            row.dash_phase_xi_gain_q8,
            row.width_bias_q8,
            row.width_slope_q12,
        ),
        keys=lambda row: row.line_index,
        label="lane-program",
    )


def _decode_lane_programs(payload: bytes) -> tuple[LanePeriodicProgramV1, ...]:
    return _decode_fixed_rows(
        payload,
        magic=_LANE_PROGRAM_MAGIC,
        header=_LANE_PROGRAM_HEADER,
        row_struct=_LANE_PROGRAM_ROW,
        factory=LanePeriodicProgramV1,
        encoder=_encode_lane_programs,
        label="lane-program",
    )


def _encode_lane_knots(rows: Sequence[LaneDriftKnotV1]) -> bytes:
    return _encode_fixed_rows(
        rows,
        magic=_LANE_KNOT_MAGIC,
        header=_LANE_KNOT_HEADER,
        row_struct=_LANE_KNOT_ROW,
        values=lambda row: (
            row.line_index,
            row.pair_index,
            row.center_c0_delta_q24,
            row.center_c1_delta_q18,
            row.center_c2_delta_q12,
            row.center_c3_delta_q8,
            row.width_delta_q8,
            row.phase_delta_q8,
        ),
        keys=lambda row: (row.line_index, row.pair_index),
        label="lane-knot",
    )


def _decode_lane_knots(payload: bytes) -> tuple[LaneDriftKnotV1, ...]:
    return _decode_fixed_rows(
        payload,
        magic=_LANE_KNOT_MAGIC,
        header=_LANE_KNOT_HEADER,
        row_struct=_LANE_KNOT_ROW,
        factory=LaneDriftKnotV1,
        encoder=_encode_lane_knots,
        label="lane-knot",
    )


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
    correction_policy: Literal["greedy_measured_fisher_margin_candidate_search_g2cs1_boundary_xi_events_no_pixels"] = (
        "greedy_measured_fisher_margin_candidate_search_g2cs1_boundary_xi_events_no_pixels"
    )
    checkpoint_policy: Literal["atomic_preserve_inventory_every_candidate_every_budget"] = (
        "atomic_preserve_inventory_every_candidate_every_budget"
    )
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
    correction_policy: Literal["obligation_derived_lane_full_curvelet_boundary_island_moments_no_pixels"] = (
        "obligation_derived_lane_full_curvelet_boundary_island_moments_no_pixels"
    )
    admission_policy: Literal["greedy_measured_joint_contest_objective_pose_tube_safety_only"] = (
        "greedy_measured_joint_contest_objective_pose_tube_safety_only"
    )
    boundary_basis: Literal["governed_compact_parabolic_shearlet_bank_fourier_free"] = (
        "governed_compact_parabolic_shearlet_bank_fourier_free"
    )
    checkpoint_policy: Literal["atomic_preserve_inventory_every_candidate_every_budget"] = (
        "atomic_preserve_inventory_every_candidate_every_budget"
    )
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
    drain_policy: Literal["exhaustive_conflict_free_canonical_batch_family_partition"] = (
        "exhaustive_conflict_free_canonical_batch_family_partition"
    )
    ev_order_policy: Literal["flip_distance_x_margin_band_x_stratum_mass_movable_lane_first"] = (
        "flip_distance_x_margin_band_x_stratum_mass_movable_lane_first"
    )
    base_cache_policy: Literal["immutable_zlib_argmax_pose_per_canonical_batch"] = (
        "immutable_zlib_argmax_pose_per_canonical_batch"
    )
    checkpoint_policy: Literal["atomic_preserve_inventory_base_batches_every_candidate_every_budget"] = (
        "atomic_preserve_inventory_base_batches_every_candidate_every_budget"
    )

    @model_validator(mode="after")
    def _valid_v12(self) -> DirectDescriptionV12ObligationDrainConfigV1:
        if (self.pair_start, self.pair_count) != (0, 600):
            raise ValueError("v12 obligation drain is the exact full n600 [0,600) window")
        if self.max_atoms_per_measured_bundle > 64:
            raise ValueError("v12 bundles are capped at 64 atomic chart/event obligations")
        return self


class DirectDescriptionV13WorldsheetPredictorConfigV1(BaseModel):
    """Typed local-only natural-production PREDICT successor contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionV13WorldsheetPredictorConfigV1"] = Field(
        default="DirectDescriptionV13WorldsheetPredictorConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: StrictStr
    seed: Literal[1234] = SEED
    pair_start: Literal[0, 448]
    pair_count: Literal[64, 600]
    v6_receipt_path: StrictStr
    v6_receipt_sha256: StrictStr
    predictor_archive_path: StrictStr
    predictor_archive_sha256: StrictStr
    upstream_root: StrictStr
    scorer_batch_size: Literal[16] = 16
    scorer_threads: StrictInt = Field(ge=1, le=16)
    total_archive_ceiling_bytes: Literal[200000] = 200000
    minimum_component_sites: StrictInt = Field(default=8, ge=2, le=4096)
    maximum_worldsheet_tracks: StrictInt = Field(default=2048, ge=1, le=8192)
    maximum_knots_per_track: StrictInt = Field(default=16, ge=1, le=64)
    worldsheet_match_radius_pixels: StrictInt = Field(default=48, ge=4, le=192)
    worldsheet_knot_error_pixels_q4: StrictInt = Field(default=64, ge=8, le=512)
    lane_knot_stride: StrictInt = Field(default=24, ge=4, le=128)
    max_ladder_rungs_per_invocation: Literal[1] = 1
    composition_ladder: tuple[Literal["base", "islands", "lane", "both"], ...] = (
        "base",
        "islands",
        "lane",
        "both",
    )
    worldsheet_policy: Literal["g1_eps1_persist_unless_event_delta_centroid_absolute_relative_polygon_shape"] = (
        "g1_eps1_persist_unless_event_delta_centroid_absolute_relative_polygon_shape"
    )
    lane_policy: Literal["coherent_multiframe_slot_periodic_dash_phase_xi_polynomial_drift_width_visibility"] = (
        "coherent_multiframe_slot_periodic_dash_phase_xi_polynomial_drift_width_visibility"
    )
    lane_policy_status: Literal["measured_pre_20260722T1916Z_operator_addenda_baseline_only"] = (
        "measured_pre_20260722T1916Z_operator_addenda_baseline_only"
    )
    lane_successor_required: Literal[
        "bev_curvature_dash_comb_range_gate_anisotropic_ar1_whitened_innovations_road_polytope"
    ] = "bev_curvature_dash_comb_range_gate_anisotropic_ar1_whitened_innovations_road_polytope"
    movable_successor_required: Literal[
        "projective_flow_depth_magnification_shared_template_aspect_rotation_sparse_events"
    ] = "projective_flow_depth_magnification_shared_template_aspect_rotation_sparse_events"
    grammar_hierarchy: Literal["stratum_then_object_then_production_with_cross_stratum_lane_road_adjacency"] = (
        "stratum_then_object_then_production_with_cross_stratum_lane_road_adjacency"
    )
    rate_authority: Literal["exact_len_receiver_closed_v13_zip_two_part_derivation_code"] = (
        "exact_len_receiver_closed_v13_zip_two_part_derivation_code"
    )
    checkpoint_policy: Literal["atomic_preserve_extraction_then_each_composition_measurement"] = (
        "atomic_preserve_extraction_then_each_composition_measurement"
    )
    class_order: tuple[StrictStr, ...] = CLASS_ORDER
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid_v13(self) -> DirectDescriptionV13WorldsheetPredictorConfigV1:
        for name in ("v6_receipt_sha256", "predictor_archive_sha256"):
            _require_sha256(getattr(self, name), name)
        if (self.pair_start, self.pair_count) not in {(448, 64), (0, 600)}:
            raise ValueError("v13 windows must be exactly [448,512) or [0,600)")
        if not Path(self.upstream_root).is_absolute():
            raise ValueError("upstream_root must be absolute scorer custody")
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be canonical {CLASS_ORDER!r}")
        if self.composition_ladder != ("base", "islands", "lane", "both"):
            raise ValueError("v13 composition ladder is sealed base -> islands -> lane -> both")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _manifest_for(
    predictor_archive: bytes,
    predictor: ComposedStructuredMemberReceiverV1,
    correction_payload: bytes,
    boundary_payload: bytes = b"",
    event_payload: bytes = b"",
    shearlet_payload: bytes = b"",
    island_payload: bytes = b"",
    obligation_vocabulary: bool = False,
    worldsheet_track_payload: bytes = b"",
    worldsheet_knot_payload: bytes = b"",
    worldsheet_g1_payload: bytes = b"",
    lane_program_payload: bytes = b"",
    lane_knot_payload: bytes = b"",
) -> dict[str, Any]:
    if worldsheet_track_payload or worldsheet_g1_payload or lane_program_payload:
        tracks = _decode_worldsheet_tracks(worldsheet_track_payload)
        worldsheet_knots = _decode_worldsheet_knots(worldsheet_knot_payload)
        _worldsheet_mask, worldsheet_g1 = (
            decode_g1_movable_worldsheet(worldsheet_g1_payload, expected_pairs=predictor.z.n_pairs)
            if worldsheet_g1_payload
            else (None, None)
        )
        lane_programs = _decode_lane_programs(lane_program_payload)
        lane_knots = _decode_lane_knots(lane_knot_payload)
        return {
            "schema": ARCHIVE_SCHEMA_V4,
            "magic": MAGIC_V4,
            "pair_count": predictor.z.n_pairs,
            "source_pair_start": predictor.source_pair_start,
            "class_order": list(CLASS_ORDER),
            "role_order": list(COMPOSED_ROLE_ORDER),
            "role_class_ids": ROLE_CLASS_IDS,
            "predictor": {"bytes": len(predictor_archive), "sha256": _sha256(predictor_archive)},
            "grammar": {
                "hierarchy": "stratum -> object -> production",
                "temporal_default": "persist-unless-birth-or-death-event",
                "cross_stratum_constraint": "Lane remains adjacent to the inherited Road boundary",
                "free_generic_logic": "production interpreter and rasterizer only; no instance tables",
                "counted_derivations_only": True,
                "movable": {
                    "g1_polygon_worldsheet": {
                        "member": WORLDSHEET_G1_MEMBER if worldsheet_g1_payload else None,
                        "bytes": len(worldsheet_g1_payload),
                        "sha256": _sha256(worldsheet_g1_payload),
                        "pair_count": 0 if worldsheet_g1 is None else worldsheet_g1.pair_count,
                        "object_slots": 0 if worldsheet_g1 is None else worldsheet_g1.max_slots,
                        "alphabet": "G1 eps1 EVENT + delta CENTROID + absolute relative polygon SHAPE",
                        "g1_reference_payload_sha256": "1066081727229e605462e67b8fdd26937d5e3552c13cb66a7444ea3b7360366f",
                    },
                    "tracks": {
                        "member": WORLDSHEET_TRACK_MEMBER if worldsheet_track_payload else None,
                        "bytes": len(worldsheet_track_payload),
                        "sha256": _sha256(worldsheet_track_payload),
                        "symbol_count": len(tracks),
                        "alphabet": "birth/death + xi ride + moment/curvelet initial contour",
                    },
                    "deviation_morph_knots": {
                        "member": WORLDSHEET_KNOT_MEMBER if worldsheet_knot_payload else None,
                        "bytes": len(worldsheet_knot_payload),
                        "sha256": _sha256(worldsheet_knot_payload),
                        "symbol_count": len(worldsheet_knots),
                        "alphabet": "sparse deviations from xi prediction + low-order contour morph",
                    },
                },
                "lane": {
                    "periodic_programs": {
                        "member": LANE_PROGRAM_MEMBER if lane_program_payload else None,
                        "bytes": len(lane_program_payload),
                        "sha256": _sha256(lane_program_payload),
                        "symbol_count": len(lane_programs),
                        "alphabet": "one dash phase advanced by xi + width profile + appear/disappear",
                    },
                    "drift_knots": {
                        "member": LANE_KNOT_MEMBER if lane_knot_payload else None,
                        "bytes": len(lane_knot_payload),
                        "sha256": _sha256(lane_knot_payload),
                        "symbol_count": len(lane_knots),
                        "alphabet": "coherent polynomial drift/width/phase deviations; never per-dash events",
                    },
                },
            },
            "xi_pose6": {
                "home": "predictor.zip::chart.zip::ddm_chart_v3/05_pose6_pair_codes.bin",
                "ownership": "sole counted xi/Pose6 stream; both natural-production families consume it",
            },
            "pixel_coordinate_or_rgb_patch_present": False,
            "scorer_weights_present": False,
            "ground_truth_argmax_present": False,
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        }
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
    worldsheet_tracks: Sequence[MovableWorldsheetTrackV1] = (),
    worldsheet_knots: Sequence[MovableWorldsheetKnotV1] = (),
    worldsheet_g1_payload: bytes = b"",
    lane_programs: Sequence[LanePeriodicProgramV1] = (),
    lane_knots: Sequence[LaneDriftKnotV1] = (),
    realization_profile: ReceiverRealizationProfileV1 | None = None,
    realization_static_rule_payload: bytes = b"",
    realization_static_rule_id: str | None = None,
    scorer_solved_templates: ScorerSolvedTemplateBankV1 | None = None,
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
    addressed = [
        *symbols,
        *boundary_symbols,
        *topology_events,
        *boundary_shearlets,
        *island_shapes,
    ]
    if any(row.pair_index < window_start or row.pair_index >= window_stop for row in addressed):
        raise DirectDescriptionError("correction symbol is outside the nested predictor source window")
    if any(row.pair_index + row.lifetime > window_stop for row in topology_events):
        raise DirectDescriptionError("topology-event lifetime escapes the nested predictor source window")
    if any(row.pair_index + row.lifetime > window_stop for row in island_shapes):
        raise DirectDescriptionError("island-shape lifetime escapes the nested predictor source window")
    if any(
        row.birth_pair < window_start or row.death_pair_exclusive > window_stop
        for row in (*worldsheet_tracks, *lane_programs)
    ):
        raise DirectDescriptionError("v13 production lifecycle escapes the nested predictor source window")
    track_by_id = {row.object_id: row for row in worldsheet_tracks}
    if len(track_by_id) != len(worldsheet_tracks):
        raise DirectDescriptionError("v13 worldsheet object IDs must be unique")
    if any(
        row.object_id not in track_by_id
        or not track_by_id[row.object_id].birth_pair <= row.pair_index < track_by_id[row.object_id].death_pair_exclusive
        for row in worldsheet_knots
    ):
        raise DirectDescriptionError("v13 worldsheet knot is outside its declared object lifecycle")
    lane_by_id = {row.line_index: row for row in lane_programs}
    if len(lane_by_id) != len(lane_programs):
        raise DirectDescriptionError("v13 lane object IDs must be unique")
    if any(
        row.line_index not in lane_by_id
        or not lane_by_id[row.line_index].birth_pair <= row.pair_index < lane_by_id[row.line_index].death_pair_exclusive
        for row in lane_knots
    ):
        raise DirectDescriptionError("v13 lane knot is outside its declared object lifecycle")
    if worldsheet_g1_payload and (worldsheet_tracks or worldsheet_knots):
        raise DirectDescriptionError("G1 polygon worldsheet cannot be mixed with the superseded moment worldsheet")
    if worldsheet_g1_payload:
        decode_g1_movable_worldsheet(worldsheet_g1_payload, expected_pairs=predictor.z.n_pairs)
    v13_requested = bool(worldsheet_tracks or worldsheet_knots or worldsheet_g1_payload or lane_programs or lane_knots)
    if realization_profile is not None and not worldsheet_g1_payload:
        raise DirectDescriptionError("receiver realization profile requires the exact G1 worldsheet")
    if realization_static_rule_payload and realization_profile is None:
        raise DirectDescriptionError("static realization rule requires the counted realization profile")
    if scorer_solved_templates is not None and realization_profile is None:
        raise DirectDescriptionError("scorer-solved templates require the counted V14 realization profile")
    _decode_realization_static_rule(realization_static_rule_payload, realization_static_rule_id)
    v11_requested = bool(boundary_shearlets or island_shapes or obligation_vocabulary)
    if v13_requested and addressed:
        raise DirectDescriptionError("v13 PREDICT productions cannot be mixed with post-solve correction vocabularies")
    if v11_requested and (boundary_symbols or topology_events):
        raise DirectDescriptionError("V10 and V11 correction vocabularies cannot be mixed in one archive")
    correction_payload = encode_lane_coefficient_deltas(tuple(symbols))
    boundary_payload = _encode_boundary_coefficient_deltas(tuple(boundary_symbols))
    event_payload = _encode_topology_events(tuple(topology_events))
    shearlet_payload = _encode_boundary_shearlet_atoms(tuple(boundary_shearlets))
    island_payload = _encode_island_shape_atoms(tuple(island_shapes))
    worldsheet_track_payload = _encode_worldsheet_tracks(tuple(worldsheet_tracks))
    worldsheet_knot_payload = _encode_worldsheet_knots(tuple(worldsheet_knots))
    lane_program_payload = _encode_lane_programs(tuple(lane_programs))
    lane_knot_payload = _encode_lane_knots(tuple(lane_knots))
    realization_payload = _encode_realization_profile(realization_profile)
    solved_template_payload = encode_scorer_solved_template_bank(scorer_solved_templates)
    manifest = _manifest_for(
        predictor_archive,
        predictor,
        correction_payload,
        boundary_payload,
        event_payload,
        shearlet_payload,
        island_payload,
        v11_requested,
        worldsheet_track_payload,
        worldsheet_knot_payload,
        worldsheet_g1_payload,
        lane_program_payload,
        lane_knot_payload,
    )
    if realization_payload:
        manifest["schema"] = ARCHIVE_SCHEMA_V6 if solved_template_payload else ARCHIVE_SCHEMA_V5
        manifest["magic"] = MAGIC_V6 if solved_template_payload else MAGIC_V5
        manifest["realization_profile"] = {
            "member": REALIZATION_PROFILE_MEMBER,
            "bytes": len(realization_payload),
            "sha256": _sha256(realization_payload),
            "paint_order": list(REALIZATION_PAINT_ORDER),
            "placement": "hard_semantic_coverage_at_camera_874x1164_before_scorer_R_down",
            "worldsheet_policy": "replace_inherited_movable_mask",
            "edge_policy": "nearest_camera_coverage_no_expansion",
            "amplitude_floor_u8": realization_profile.amplitude_u8,
            "pixel_coordinate_or_rgb_patch_present": bool(solved_template_payload),
        }
        if realization_static_rule_payload:
            manifest["realization_static_rule"] = {
                "member": REALIZATION_STATIC_RULE_MEMBER,
                "opportunity_id": realization_static_rule_id,
                "bytes": len(realization_static_rule_payload),
                "sha256": _sha256(realization_static_rule_payload),
                "placement": "decoder_derived_source_semantic_mask_then_camera_resolution_target_prototype",
                "target_custody": "G4_static_cell_rule_only_no_scorer_or_ground_truth_table",
            }
        if solved_template_payload:
            manifest["scorer_solved_templates"] = {
                "member": SCORER_SOLVED_TEMPLATE_MEMBER,
                "bytes": len(solved_template_payload),
                "sha256": _sha256(solved_template_payload),
                "record_count": len(scorer_solved_templates.templates),
                "placement": "grammar_semantic_mask_x_scorer_row_band_x_periodic_uint8_patch",
                "solve_boundary": "encode_side_only_frozen_scorer_through_exact_R",
                "decode_boundary": "deterministic_template_tiling_no_scorer",
                "target_custody": "counted_video_derived_shared_template_no_ground_truth_table",
            }
    members = {
        "manifest.json": rfc8785_canonicalize(manifest),
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
    if worldsheet_track_payload:
        members[WORLDSHEET_TRACK_MEMBER] = worldsheet_track_payload
    if worldsheet_knot_payload:
        members[WORLDSHEET_KNOT_MEMBER] = worldsheet_knot_payload
    if worldsheet_g1_payload:
        members[WORLDSHEET_G1_MEMBER] = worldsheet_g1_payload
    if realization_payload:
        members[REALIZATION_PROFILE_MEMBER] = realization_payload
    if realization_static_rule_payload:
        members[REALIZATION_STATIC_RULE_MEMBER] = realization_static_rule_payload
    if solved_template_payload:
        members[SCORER_SOLVED_TEMPLATE_MEMBER] = solved_template_payload
    if lane_program_payload:
        members[LANE_PROGRAM_MEMBER] = lane_program_payload
    if lane_knot_payload:
        members[LANE_KNOT_MEMBER] = lane_knot_payload
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
            if [row.filename for row in infos[:2]] != expected_prefix or not 2 <= len(infos) <= 11:
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
    worldsheet_tracks_payload = members.get(WORLDSHEET_TRACK_MEMBER, b"")
    worldsheet_knots_payload = members.get(WORLDSHEET_KNOT_MEMBER, b"")
    worldsheet_g1_payload = members.get(WORLDSHEET_G1_MEMBER, b"")
    realization_payload = members.get(REALIZATION_PROFILE_MEMBER, b"")
    realization_static_rule_payload = members.get(REALIZATION_STATIC_RULE_MEMBER, b"")
    solved_template_payload = members.get(SCORER_SOLVED_TEMPLATE_MEMBER, b"")
    lane_programs_payload = members.get(LANE_PROGRAM_MEMBER, b"")
    lane_knots_payload = members.get(LANE_KNOT_MEMBER, b"")
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
    elif schema in {ARCHIVE_SCHEMA_V4, ARCHIVE_SCHEMA_V5, ARCHIVE_SCHEMA_V6}:
        grammar = manifest.get("grammar", {})
        movable = grammar.get("movable", {})
        lane = grammar.get("lane", {})
        expected_members = {
            "manifest.json",
            "predictor.zip",
            *([WORLDSHEET_TRACK_MEMBER] if worldsheet_tracks_payload else []),
            *([WORLDSHEET_KNOT_MEMBER] if worldsheet_knots_payload else []),
            *([WORLDSHEET_G1_MEMBER] if worldsheet_g1_payload else []),
            *([REALIZATION_PROFILE_MEMBER] if realization_payload else []),
            *([REALIZATION_STATIC_RULE_MEMBER] if realization_static_rule_payload else []),
            *([SCORER_SOLVED_TEMPLATE_MEMBER] if solved_template_payload else []),
            *([LANE_PROGRAM_MEMBER] if lane_programs_payload else []),
            *([LANE_KNOT_MEMBER] if lane_knots_payload else []),
        }
        payloads = {
            "worldsheet_tracks": (movable.get("tracks", {}), WORLDSHEET_TRACK_MEMBER, worldsheet_tracks_payload),
            "worldsheet_knots": (
                movable.get("deviation_morph_knots", {}),
                WORLDSHEET_KNOT_MEMBER,
                worldsheet_knots_payload,
            ),
            "worldsheet_g1": (
                movable.get("g1_polygon_worldsheet", {}),
                WORLDSHEET_G1_MEMBER,
                worldsheet_g1_payload,
            ),
            "lane_programs": (lane.get("periodic_programs", {}), LANE_PROGRAM_MEMBER, lane_programs_payload),
            "lane_knots": (lane.get("drift_knots", {}), LANE_KNOT_MEMBER, lane_knots_payload),
        }
        invalid = (
            common_invalid
            or manifest.get("magic")
            != ({ARCHIVE_SCHEMA_V4: MAGIC_V4, ARCHIVE_SCHEMA_V5: MAGIC_V5, ARCHIVE_SCHEMA_V6: MAGIC_V6}[schema])
            or correction
            or boundary
            or events
            or shearlets
            or islands
            or set(members) != expected_members
            or (not worldsheet_tracks_payload and not worldsheet_g1_payload and not lane_programs_payload)
            or (bool(worldsheet_g1_payload) and bool(worldsheet_tracks_payload or worldsheet_knots_payload))
            or bool(realization_payload) != (schema in {ARCHIVE_SCHEMA_V5, ARCHIVE_SCHEMA_V6})
            or bool(solved_template_payload) != (schema == ARCHIVE_SCHEMA_V6)
        )
        realization_row = manifest.get("realization_profile", {})
        static_rule_row = manifest.get("realization_static_rule", {})
        solved_template_row = manifest.get("scorer_solved_templates", {})
        if schema in {ARCHIVE_SCHEMA_V5, ARCHIVE_SCHEMA_V6}:
            invalid = (
                invalid
                or not worldsheet_g1_payload
                or realization_row.get("member") != REALIZATION_PROFILE_MEMBER
                or realization_row.get("bytes") != len(realization_payload)
                or realization_row.get("sha256") != _sha256(realization_payload)
                or realization_row.get("paint_order") != list(REALIZATION_PAINT_ORDER)
                or realization_row.get("pixel_coordinate_or_rgb_patch_present") != (schema == ARCHIVE_SCHEMA_V6)
            )
            _decode_realization_profile(realization_payload)
            if realization_static_rule_payload:
                invalid = (
                    invalid
                    or static_rule_row.get("member") != REALIZATION_STATIC_RULE_MEMBER
                    or static_rule_row.get("bytes") != len(realization_static_rule_payload)
                    or static_rule_row.get("sha256") != _sha256(realization_static_rule_payload)
                )
                _decode_realization_static_rule(
                    realization_static_rule_payload,
                    static_rule_row.get("opportunity_id"),
                )
            else:
                invalid = invalid or bool(static_rule_row)
            if schema == ARCHIVE_SCHEMA_V6:
                bank = decode_scorer_solved_template_bank(solved_template_payload)
                invalid = (
                    invalid
                    or bank is None
                    or solved_template_row.get("member") != SCORER_SOLVED_TEMPLATE_MEMBER
                    or solved_template_row.get("bytes") != len(solved_template_payload)
                    or solved_template_row.get("sha256") != _sha256(solved_template_payload)
                    or solved_template_row.get("record_count") != len(bank.templates)
                    or solved_template_row.get("solve_boundary") != "encode_side_only_frozen_scorer_through_exact_R"
                    or solved_template_row.get("decode_boundary") != "deterministic_template_tiling_no_scorer"
                )
            else:
                invalid = invalid or bool(solved_template_row)
        else:
            invalid = (
                invalid
                or bool(realization_row)
                or bool(static_rule_row)
                or bool(realization_static_rule_payload)
                or bool(solved_template_row)
                or bool(solved_template_payload)
            )
        for row, member_name, payload in payloads.values():
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
        *([WORLDSHEET_TRACK_MEMBER] if worldsheet_tracks_payload else []),
        *([WORLDSHEET_KNOT_MEMBER] if worldsheet_knots_payload else []),
        *([WORLDSHEET_G1_MEMBER] if worldsheet_g1_payload else []),
        *([REALIZATION_PROFILE_MEMBER] if realization_payload else []),
        *([REALIZATION_STATIC_RULE_MEMBER] if realization_static_rule_payload else []),
        *([SCORER_SOLVED_TEMPLATE_MEMBER] if solved_template_payload else []),
        *([LANE_PROGRAM_MEMBER] if lane_programs_payload else []),
        *([LANE_KNOT_MEMBER] if lane_knots_payload else []),
    ]
    if member_order != canonical_member_order:
        raise DirectDescriptionError("carrier correction member order is noncanonical")
    symbols = decode_lane_coefficient_deltas(correction)
    boundary_symbols = _decode_boundary_coefficient_deltas(boundary)
    topology_events = _decode_topology_events(events)
    boundary_shearlets = _decode_boundary_shearlet_atoms(shearlets)
    island_shapes = _decode_island_shape_atoms(islands)
    worldsheet_tracks = _decode_worldsheet_tracks(worldsheet_tracks_payload)
    worldsheet_knots = _decode_worldsheet_knots(worldsheet_knots_payload)
    worldsheet_g1_metadata = (
        decode_g1_movable_worldsheet(worldsheet_g1_payload, expected_pairs=manifest["pair_count"])[1]
        if worldsheet_g1_payload
        else None
    )
    lane_programs = _decode_lane_programs(lane_programs_payload)
    lane_knots = _decode_lane_knots(lane_knots_payload)
    if schema == ARCHIVE_SCHEMA:
        counts_valid = manifest["correction"]["symbol_count"] == len(symbols)
    elif schema == ARCHIVE_SCHEMA_V2:
        counts_valid = (
            manifest["corrections"]["lane_g2cs1"]["symbol_count"] == len(symbols)
            and manifest["corrections"]["road_boundary_coefficients"]["symbol_count"] == len(boundary_symbols)
            and manifest["corrections"]["topology_events"]["symbol_count"] == len(topology_events)
        )
    elif schema == ARCHIVE_SCHEMA_V3:
        counts_valid = (
            manifest["corrections"]["lane_full_coefficients"]["symbol_count"] == len(symbols)
            and manifest["corrections"]["boundary_shearlet_atoms"]["symbol_count"] == len(boundary_shearlets)
            and manifest["corrections"]["movable_shape_atoms"]["symbol_count"] == len(island_shapes)
        )
    else:
        counts_valid = (
            manifest["grammar"]["movable"]["g1_polygon_worldsheet"]["pair_count"]
            == (0 if worldsheet_g1_metadata is None else worldsheet_g1_metadata.pair_count)
            and manifest["grammar"]["movable"]["g1_polygon_worldsheet"]["object_slots"]
            == (0 if worldsheet_g1_metadata is None else worldsheet_g1_metadata.max_slots)
            and manifest["grammar"]["movable"]["tracks"]["symbol_count"] == len(worldsheet_tracks)
            and manifest["grammar"]["movable"]["deviation_morph_knots"]["symbol_count"] == len(worldsheet_knots)
            and manifest["grammar"]["lane"]["periodic_programs"]["symbol_count"] == len(lane_programs)
            and manifest["grammar"]["lane"]["drift_knots"]["symbol_count"] == len(lane_knots)
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
        window = np.square(np.maximum(1.0 - np.square(u), 0.0)) * np.square(np.maximum(1.0 - np.square(v), 0.0))
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
    pose6_codes: np.ndarray | None,
) -> np.ndarray:
    """Rasterize one parametric topology event; target cells never enter."""

    transport_required = requires_pose6_transport(event)
    if transport_required and pose6_codes is None:
        raise DirectDescriptionError("nonzero-gain topology event requires Pose6 transport")
    if not event.pair_index <= source_pair_id < event.pair_index + event.lifetime:
        return np.zeros((384, 512), dtype=bool)
    dx = 0
    dy = 0
    if transport_required:
        assert pose6_codes is not None
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
    result[clipped_y0:clipped_y1, clipped_x0:clipped_x1] = np.square((ys - cy) / ry) + np.square((xs - cx) / rx) <= 1.0
    return result


def _island_shape_mask(
    atom: IslandShapeAtomV1,
    *,
    source_pair_id: int,
    source_pair_start: int,
    pose6_codes: np.ndarray | None,
) -> np.ndarray:
    """Synthesize a moment-shaped island with one compact curvelet lobe."""

    transport_required = requires_pose6_transport(atom)
    if transport_required and pose6_codes is None:
        raise DirectDescriptionError("nonzero-gain island shape requires Pose6 transport")
    if not atom.pair_index <= source_pair_id < atom.pair_index + atom.lifetime:
        return np.zeros((384, 512), dtype=bool)
    dx = 0
    dy = 0
    if transport_required:
        assert pose6_codes is not None
        birth_local = atom.pair_index - source_pair_start
        current_local = source_pair_id - source_pair_start
        if not (0 <= birth_local < len(pose6_codes) and 0 <= current_local < len(pose6_codes)):
            raise DirectDescriptionError("island-shape Pose6 transport address escaped the local window")
        pose_delta = pose6_codes[current_local].astype(np.int16) - pose6_codes[birth_local].astype(np.int16)
        dx = int(np.rint(float(pose_delta[0]) * atom.transport_gain_x_q4 / 16.0))
        dy = int(np.rint(float(pose_delta[1]) * atom.transport_gain_y_q4 / 16.0))
    center_x = atom.center_x + dx
    center_y = atom.center_y + dy
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
    compact_lobe = np.maximum(1.0 - 4.0 * np.square(u - 0.5), 0.0) * np.maximum(1.0 - np.square(v), 0.0)
    threshold = np.maximum(0.25, 1.0 + curvelet * compact_lobe)
    result[y0:y1, x0:x1] = np.square(shaped_u) + np.square(v) <= threshold
    return result


def _interpolate_sparse_rows(rows: Sequence[Any], pair_index: int, field: str) -> float:
    if not rows:
        return 0.0
    ordered = tuple(sorted(rows, key=lambda row: row.pair_index))
    if pair_index <= ordered[0].pair_index:
        return float(getattr(ordered[0], field))
    if pair_index >= ordered[-1].pair_index:
        return float(getattr(ordered[-1], field))
    right_index = next(index for index, row in enumerate(ordered) if row.pair_index >= pair_index)
    left, right = ordered[right_index - 1], ordered[right_index]
    if right.pair_index == left.pair_index:
        return float(getattr(right, field))
    alpha = (pair_index - left.pair_index) / float(right.pair_index - left.pair_index)
    return (1.0 - alpha) * float(getattr(left, field)) + alpha * float(getattr(right, field))


def _apply_lane_predictor_programs(
    layers: Sequence[StructuredRoleLayerV1],
    programs: Sequence[LanePeriodicProgramV1],
    knots: Sequence[LaneDriftKnotV1],
    *,
    pose6_codes: np.ndarray,
    source_pair_start: int,
) -> tuple[StructuredRoleLayerV1, ...]:
    if not programs:
        return tuple(layers)
    copied = list(layers)
    lane_index = next((index for index, layer in enumerate(copied) if layer.role == "Lane"), None)
    if lane_index is None or copied[lane_index].lane_lines is None:
        raise DirectDescriptionError("v13 receiver lacks the inherited coherent Lane chart")
    lines = [[np.asarray(value, dtype=np.float64).copy() for value in pair] for pair in copied[lane_index].lane_lines]
    stop = source_pair_start + len(pose6_codes)
    for program in programs:
        relevant = tuple(row for row in knots if row.line_index == program.line_index)
        birth_local = program.birth_pair - source_pair_start
        if not 0 <= birth_local < len(pose6_codes):
            raise DirectDescriptionError("v13 lane-program birth lacks xi/Pose6 custody")
        template_pair = next(
            (
                pair_index
                for pair_index in range(program.birth_pair, program.death_pair_exclusive)
                if program.line_index < len(lines[pair_index])
            ),
            None,
        )
        if template_pair is None:
            raise DirectDescriptionError("v13 lane-program object has no inherited coherent slot")
        template_phase = float(lines[template_pair][program.line_index][7])
        for pair_index in range(max(program.birth_pair, source_pair_start), min(program.death_pair_exclusive, stop)):
            if program.line_index >= len(lines[pair_index]):
                continue
            vector = lines[pair_index][program.line_index]
            if vector.size < 11:
                raise DirectDescriptionError("v13 lane-program addressed an incomplete Lane vector")
            for coefficient_index, (field, scale) in enumerate(
                (
                    ("center_c0_delta_q24", 1 << 24),
                    ("center_c1_delta_q18", 1 << 18),
                    ("center_c2_delta_q12", 1 << 12),
                    ("center_c3_delta_q8", 1 << 8),
                )
            ):
                vector[coefficient_index] += _interpolate_sparse_rows(relevant, pair_index, field) / scale
            vector[4] += (
                program.width_bias_q8 + _interpolate_sparse_rows(relevant, pair_index, "width_delta_q8")
            ) / 256.0
            vector[5] += program.width_slope_q12 / 4096.0
            current_local = pair_index - source_pair_start
            xi_delta = int(pose6_codes[current_local, 0]) - int(pose6_codes[birth_local, 0])
            vector[7] = (
                template_phase
                + (
                    program.dash_phase_origin_delta_q8
                    + xi_delta * program.dash_phase_xi_gain_q8
                    + _interpolate_sparse_rows(relevant, pair_index, "phase_delta_q8")
                )
                / 256.0
            )
            if not np.isfinite(vector).all():
                raise DirectDescriptionError("v13 Lane production produced nonfinite coefficients")
    copied[lane_index] = replace(copied[lane_index], lane_lines=tuple(tuple(value for value in pair) for pair in lines))
    return tuple(copied)


def _worldsheet_track_mask(
    track: MovableWorldsheetTrackV1,
    knots: Sequence[MovableWorldsheetKnotV1],
    *,
    source_pair_id: int,
    source_pair_start: int,
    pose6_codes: np.ndarray,
) -> np.ndarray:
    if not track.birth_pair <= source_pair_id < track.death_pair_exclusive:
        return np.zeros((384, 512), dtype=bool)
    birth_local = track.birth_pair - source_pair_start
    current_local = source_pair_id - source_pair_start
    if not (0 <= birth_local < len(pose6_codes) and 0 <= current_local < len(pose6_codes)):
        raise DirectDescriptionError("v13 worldsheet track escaped xi/Pose6 custody")
    pose_delta = pose6_codes[current_local].astype(np.int16) - pose6_codes[birth_local].astype(np.int16)
    center_y = track.center_y + float(pose_delta[1]) * track.transport_gain_y_q4 / 16.0
    center_x = track.center_x + float(pose_delta[0]) * track.transport_gain_x_q4 / 16.0
    center_y += _interpolate_sparse_rows(knots, source_pair_id, "delta_center_y_q4") / 16.0
    center_x += _interpolate_sparse_rows(knots, source_pair_id, "delta_center_x_q4") / 16.0
    radius_y = track.radius_y + _interpolate_sparse_rows(knots, source_pair_id, "delta_radius_y_q4") / 16.0
    radius_x = track.radius_x + _interpolate_sparse_rows(knots, source_pair_id, "delta_radius_x_q4") / 16.0
    extent = max(radius_y, radius_x) * 2.0
    if center_y + extent < 0 or center_y - extent >= 384 or center_x + extent < 0 or center_x - extent >= 512:
        return np.zeros((384, 512), dtype=bool)
    atom = IslandShapeAtomV1(
        pair_index=source_pair_id,
        action="birth",
        lifetime=1,
        center_y=int(np.clip(np.rint(center_y), 0, 383)),
        center_x=int(np.clip(np.rint(center_x), 0, 511)),
        radius_y=int(np.clip(np.rint(radius_y), 1, 191)),
        radius_x=int(np.clip(np.rint(radius_x), 1, 255)),
        angle_u8=int(np.rint(track.angle_u8 + _interpolate_sparse_rows(knots, source_pair_id, "delta_angle_q4") / 16.0))
        % 256,
        skew_q6=int(
            np.clip(
                np.rint(track.skew_q6 + _interpolate_sparse_rows(knots, source_pair_id, "delta_skew_q6")),
                -96,
                96,
            )
        ),
        taper_q6=int(
            np.clip(
                np.rint(track.taper_q6 + _interpolate_sparse_rows(knots, source_pair_id, "delta_taper_q6")),
                -96,
                96,
            )
        ),
        curvelet_q6=int(
            np.clip(
                np.rint(track.curvelet_q6 + _interpolate_sparse_rows(knots, source_pair_id, "delta_curvelet_q6")),
                -96,
                96,
            )
        ),
    )
    return _island_shape_mask(
        atom,
        source_pair_id=source_pair_id,
        source_pair_start=source_pair_start,
        pose6_codes=pose6_codes,
    )


def _dilate_one_pixel(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    return np.logical_or.reduce(
        tuple(
            padded[1 + dy : 1 + dy + source.shape[0], 1 + dx : 1 + dx + source.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )
    )


def _inner_boundary_one_cell(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    eroded = source.copy()
    eroded &= padded[:-2, 1:-1]
    eroded &= padded[2:, 1:-1]
    eroded &= padded[1:-1, :-2]
    eroded &= padded[1:-1, 2:]
    return source & ~eroded


def _template_camera_field(template: RowBandScorerTemplateV1, camera_h: int, camera_w: int) -> np.ndarray:
    patch = template.patch()
    rows = np.arange(camera_h, dtype=np.intp) % template.patch_height
    cols = np.arange(camera_w, dtype=np.intp) % template.patch_width
    return np.ascontiguousarray(patch[rows[:, None], cols[None, :]])


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
    worldsheet_tracks: tuple[MovableWorldsheetTrackV1, ...] = ()
    worldsheet_knots: tuple[MovableWorldsheetKnotV1, ...] = ()
    worldsheet_g1_mask: np.ndarray | None = None
    lane_programs: tuple[LanePeriodicProgramV1, ...] = ()
    lane_knots: tuple[LaneDriftKnotV1, ...] = ()
    realization_profile: ReceiverRealizationProfileV1 | None = None
    realization_static_rule_codes: np.ndarray | None = None
    realization_static_rule_id: str | None = None
    scorer_solved_templates: ScorerSolvedTemplateBankV1 | None = None

    @property
    def z(self) -> Any:
        return self.predictor.z

    @property
    def pose6_codes(self) -> np.ndarray:
        return self.predictor.pose6_codes

    def _mask_for_layer(
        self,
        layer: StructuredRoleLayerV1,
        pair_id: int,
        *,
        replace_g1_movable: bool,
    ) -> np.ndarray:
        source_pair_id = self.predictor.source_pair_start + pair_id
        mask = layer.mask(
            local_pair_id=pair_id,
            source_pair_id=source_pair_id,
            camera=self.predictor.camera,
        )
        if layer.role == "Lane" and self.lane_programs:
            road_layer = next(row for row in self.layers if row.role == "Road")
            road_mask = road_layer.mask(
                local_pair_id=pair_id,
                source_pair_id=source_pair_id,
                camera=self.predictor.camera,
            )
            mask &= _dilate_one_pixel(road_mask)
        boundary = tuple(
            row for row in self.boundary_symbols if row.pair_index == source_pair_id and row.role == layer.role
        )
        if boundary:
            mask = _apply_boundary_coefficients(mask, boundary)
        shearlets = tuple(
            row for row in self.boundary_shearlets if row.pair_index == source_pair_id and row.role == layer.role
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
            if self.worldsheet_g1_mask is not None:
                if replace_g1_movable:
                    mask = self.worldsheet_g1_mask[pair_id].copy()
                else:
                    mask |= self.worldsheet_g1_mask[pair_id]
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
            for track in self.worldsheet_tracks:
                track_sites = _worldsheet_track_mask(
                    track,
                    tuple(row for row in self.worldsheet_knots if row.object_id == track.object_id),
                    source_pair_id=source_pair_id,
                    source_pair_start=self.predictor.source_pair_start,
                    pose6_codes=self.pose6_codes,
                )
                mask |= track_sites
        return mask

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        """Legacy scorer-grid render retained byte-for-byte for V9--V13."""

        indexes = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in indexes):
            raise DirectDescriptionError("v9 receiver pair ID is outside its local window")
        output = self.predictor.baseline.render_pairs(indexes)
        for layer in self.layers:
            for local_index, pair_id in enumerate(indexes):
                mask = self._mask_for_layer(layer, pair_id, replace_g1_movable=False)
                output[local_index, 0, mask] = layer.paint_rgb_u8
                output[local_index, 1, mask] = layer.paint_rgb_u8
        return np.ascontiguousarray(output)

    def template_camera_masks(
        self,
        pair_ids: Sequence[int],
        template: RowBandScorerTemplateV1,
    ) -> np.ndarray:
        """Return the exact grammar-derived camera mask consumed by one template."""

        indexes = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in indexes):
            raise DirectDescriptionError("v15 template receiver pair ID is outside its local window")
        from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

        ys = (np.arange(CAMERA_H) * 384 // CAMERA_H).clip(0, 383)
        xs = (np.arange(CAMERA_W) * 512 // CAMERA_W).clip(0, 511)
        layer = next(row for row in self.layers if row.role == template.role)
        row_band = (np.arange(384) >= template.scorer_row_start) & (np.arange(384) < template.scorer_row_stop)
        output = np.empty((len(indexes), CAMERA_H, CAMERA_W), dtype=bool)
        for local_index, pair_id in enumerate(indexes):
            mask = self._mask_for_layer(layer, pair_id, replace_g1_movable=True)
            if template.application == "inner_boundary":
                mask = _inner_boundary_one_cell(mask)
            mask &= row_band[:, None]
            role_index = REALIZATION_PAINT_ORDER.index(template.role)
            for later_role in REALIZATION_PAINT_ORDER[role_index + 1 :]:
                later_layer = next(row for row in self.layers if row.role == later_role)
                mask &= ~self._mask_for_layer(
                    later_layer,
                    pair_id,
                    replace_g1_movable=True,
                )
            output[local_index] = mask[np.ix_(ys, xs)]
        return np.ascontiguousarray(output)

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        """V14 camera-res placement before the evaluator-owned R-down stage."""

        if self.realization_profile is None:
            raise DirectDescriptionError("camera-resolution render requires a counted realization profile")
        indexes = tuple(int(value) for value in pair_ids)
        if any(value < 0 or value >= self.z.n_pairs for value in indexes):
            raise DirectDescriptionError("v14 receiver pair ID is outside its local window")
        from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W, render_grid_to_camera_uint8

        render_grid = self.predictor.baseline.render_pairs(indexes)
        output = np.empty((len(indexes), 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        for local_index in range(len(indexes)):
            for frame_index in range(2):
                output[local_index, frame_index] = render_grid_to_camera_uint8(render_grid[local_index, frame_index])
        ys = (np.arange(CAMERA_H) * 384 // CAMERA_H).clip(0, 383)
        xs = (np.arange(CAMERA_W) * 512 // CAMERA_W).clip(0, 511)
        layer_by_role = {row.role: row for row in self.layers}
        semantic_cells = np.full((len(indexes), 384, 512), -1, dtype=np.int16)
        for role in REALIZATION_PAINT_ORDER:
            layer = layer_by_role[role]
            colour = self.realization_profile.colour_for(role)
            for local_index, pair_id in enumerate(indexes):
                mask = self._mask_for_layer(layer, pair_id, replace_g1_movable=True)
                semantic_cells[local_index, mask] = ROLE_CLASS_IDS[role]
                camera_mask = mask[np.ix_(ys, xs)]
                output[local_index, 0, camera_mask] = colour
                output[local_index, 1, camera_mask] = colour
            for template in (
                ()
                if self.scorer_solved_templates is None
                else self.scorer_solved_templates.for_role(role)
            ):
                camera_masks = self.template_camera_masks(indexes, template)
                field = _template_camera_field(template, CAMERA_H, CAMERA_W)
                for local_index, camera_mask in enumerate(camera_masks):
                    output[local_index, 0, camera_mask] = field[camera_mask]
                    output[local_index, 1, camera_mask] = field[camera_mask]
        if self.realization_static_rule_codes is not None:
            rules = self.realization_static_rule_codes
            source = rules // 5
            target = rules % 5
            for local_index in range(len(indexes)):
                wildcard = (rules >= 25) & (rules < 30)
                admitted = wildcard | (
                    (rules >= 0)
                    & (rules < 25)
                    & (semantic_cells[local_index] == source)
                )
                for target_id, role in enumerate(CLASS_ORDER):
                    target_mask = admitted & (target == target_id)
                    if not np.any(target_mask):
                        continue
                    camera_mask = target_mask[np.ix_(ys, xs)]
                    colour = self.realization_profile.colour_for("UndrivableBoundary" if role == "Undrivable" else role)
                    output[local_index, 0, camera_mask] = colour
                    output[local_index, 1, camera_mask] = colour
        return np.ascontiguousarray(output)


def receive_carrier_compose_archive(
    archive: bytes,
    *,
    verify_member_effects: bool = True,
) -> CarrierComposeReceiverV1:
    """Parse a carrier archive, strictly proving member effects by default.

    ``verify_member_effects=False`` exists only for optimizer inner-loop finite
    secants after a strict source admission.  It still validates the complete
    typed wire grammar and lifecycle geometry, but skips the expensive isolated
    no-op renders and returns empty custody.  Such a receiver is never evidence;
    stage exits and published receipts must call the strict default.
    """
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
    worldsheet_tracks = _decode_worldsheet_tracks(members.get(WORLDSHEET_TRACK_MEMBER, b""))
    worldsheet_knots = _decode_worldsheet_knots(members.get(WORLDSHEET_KNOT_MEMBER, b""))
    worldsheet_g1_payload = members.get(WORLDSHEET_G1_MEMBER, b"")
    worldsheet_g1_mask, worldsheet_g1_metadata = (
        decode_g1_movable_worldsheet(worldsheet_g1_payload, expected_pairs=predictor.z.n_pairs)
        if worldsheet_g1_payload
        else (None, None)
    )
    realization_profile = _decode_realization_profile(members.get(REALIZATION_PROFILE_MEMBER, b""))
    static_rule_row = manifest.get("realization_static_rule", {})
    realization_static_rule_id = static_rule_row.get("opportunity_id")
    realization_static_rule_codes = _decode_realization_static_rule(
        members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
        realization_static_rule_id,
    )
    scorer_solved_templates = decode_scorer_solved_template_bank(members.get(SCORER_SOLVED_TEMPLATE_MEMBER, b""))
    lane_programs = _decode_lane_programs(members.get(LANE_PROGRAM_MEMBER, b""))
    lane_knots = _decode_lane_knots(members.get(LANE_KNOT_MEMBER, b""))
    start = predictor.source_pair_start
    stop = start + predictor.z.n_pairs
    addressed = [*symbols, *boundary_symbols, *topology_events, *boundary_shearlets, *island_shapes]
    if any(row.pair_index < start or row.pair_index >= stop for row in addressed):
        raise DirectDescriptionError("correction symbol is outside the nested predictor source window")
    if any(row.pair_index + row.lifetime > stop for row in topology_events):
        raise DirectDescriptionError("topology-event lifetime escapes the nested predictor source window")
    if any(row.pair_index + row.lifetime > stop for row in island_shapes):
        raise DirectDescriptionError("island-shape lifetime escapes the nested predictor source window")
    track_by_id = {row.object_id: row for row in worldsheet_tracks}
    if len(track_by_id) != len(worldsheet_tracks) or any(
        row.object_id not in track_by_id
        or not track_by_id[row.object_id].birth_pair <= row.pair_index < track_by_id[row.object_id].death_pair_exclusive
        for row in worldsheet_knots
    ):
        raise DirectDescriptionError("worldsheet track/knot lifecycle custody is invalid")
    lane_by_id = {row.line_index: row for row in lane_programs}
    if len(lane_by_id) != len(lane_programs) or any(
        row.line_index not in lane_by_id
        or not lane_by_id[row.line_index].birth_pair <= row.pair_index < lane_by_id[row.line_index].death_pair_exclusive
        for row in lane_knots
    ):
        raise DirectDescriptionError("lane program/knot lifecycle custody is invalid")
    coefficient_limit = 9 if manifest["schema"] == ARCHIVE_SCHEMA_V3 else 4
    layers = _apply_chart_symbols(predictor.layers, symbols, coefficient_limit=coefficient_limit)
    layers = _apply_lane_predictor_programs(
        layers,
        lane_programs,
        lane_knots,
        pose6_codes=predictor.pose6_codes,
        source_pair_start=predictor.source_pair_start,
    )
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
        worldsheet_tracks=worldsheet_tracks,
        worldsheet_knots=worldsheet_knots,
        worldsheet_g1_mask=worldsheet_g1_mask,
        lane_programs=lane_programs,
        lane_knots=lane_knots,
        realization_profile=realization_profile,
        realization_static_rule_codes=realization_static_rule_codes,
        realization_static_rule_id=realization_static_rule_id,
        scorer_solved_templates=scorer_solved_templates,
    )

    if not verify_member_effects:
        return first

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
    for track in worldsheet_tracks:
        local_ids = tuple(range(track.birth_pair - start, track.death_pair_exclusive - start))
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=predictor.layers,
            symbols=(),
            boundary_symbols=(),
            topology_events=(),
            boundary_shearlets=(),
            island_shapes=(),
            custody={},
            worldsheet_tracks=(track,),
            worldsheet_knots=tuple(row for row in worldsheet_knots if row.object_id == track.object_id),
        )
        if np.array_equal(predictor.render_pairs(local_ids), isolated.render_pairs(local_ids)):
            raise DirectDescriptionError("worldsheet track production is a receiver-output no-op")
    if worldsheet_g1_mask is not None:
        movable_layer = next(row for row in predictor.layers if row.role == "Movable")
        changes_output = False
        for local_pair_id in range(predictor.z.n_pairs):
            source_pair_id = start + local_pair_id
            inherited = movable_layer.mask(
                local_pair_id=local_pair_id,
                source_pair_id=source_pair_id,
                camera=predictor.camera,
            )
            if np.any(worldsheet_g1_mask[local_pair_id] & ~inherited):
                changes_output = True
                break
        if not changes_output:
            raise DirectDescriptionError("G1 Movable polygon worldsheet is a receiver-output no-op")
    for program in lane_programs:
        local_ids = tuple(range(program.birth_pair - start, program.death_pair_exclusive - start))
        isolated_layers = _apply_lane_predictor_programs(
            predictor.layers,
            (program,),
            tuple(row for row in lane_knots if row.line_index == program.line_index),
            pose6_codes=predictor.pose6_codes,
            source_pair_start=start,
        )
        isolated = CarrierComposeReceiverV1(
            archive=archive,
            predictor=predictor,
            layers=isolated_layers,
            symbols=(),
            boundary_symbols=(),
            topology_events=(),
            boundary_shearlets=(),
            island_shapes=(),
            custody={},
        )
        if np.array_equal(predictor.render_pairs(local_ids), isolated.render_pairs(local_ids)):
            raise DirectDescriptionError("lane periodic production is a receiver-output no-op")
    probes = tuple(sorted({0, predictor.z.n_pairs - 1}))
    render_probe = first.render_camera_pairs if realization_profile is not None else first.render_pairs
    a = render_probe(probes)
    b = render_probe(probes)
    if not np.array_equal(a, b):
        raise DirectDescriptionError("carrier receiver replay is nondeterministic")
    custody = {
        "schema": (
            RECEIVER_SCHEMA_V6
            if manifest["schema"] == ARCHIVE_SCHEMA_V6
            else RECEIVER_SCHEMA_V5
            if manifest["schema"] == ARCHIVE_SCHEMA_V5
            else RECEIVER_SCHEMA_V4
            if manifest["schema"] == ARCHIVE_SCHEMA_V4
            else RECEIVER_SCHEMA_V3
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
        "topology_events_consume_counted_pose6_transport": any(
            requires_pose6_transport(row) for row in topology_events
        ),
        "boundary_shearlet_count": len(boundary_shearlets),
        "boundary_shearlet_parse_reencode_identical": _encode_boundary_shearlet_atoms(boundary_shearlets)
        == members.get(BOUNDARY_SHEARLET_MEMBER, b""),
        "island_shape_count": len(island_shapes),
        "island_shape_parse_reencode_identical": _encode_island_shape_atoms(island_shapes)
        == members.get(ISLAND_SHAPE_MEMBER, b""),
        "island_shapes_consume_counted_pose6_transport": any(
            requires_pose6_transport(row) for row in island_shapes
        ),
        "worldsheet_track_count": len(worldsheet_tracks),
        "worldsheet_track_parse_reencode_identical": _encode_worldsheet_tracks(worldsheet_tracks)
        == members.get(WORLDSHEET_TRACK_MEMBER, b""),
        "worldsheet_knot_count": len(worldsheet_knots),
        "worldsheet_knot_parse_reencode_identical": _encode_worldsheet_knots(worldsheet_knots)
        == members.get(WORLDSHEET_KNOT_MEMBER, b""),
        "worldsheet_g1_payload_bytes": len(worldsheet_g1_payload),
        "worldsheet_g1_payload_sha256": _sha256(worldsheet_g1_payload),
        "worldsheet_g1_pair_count": 0 if worldsheet_g1_metadata is None else worldsheet_g1_metadata.pair_count,
        "worldsheet_g1_object_slots": 0 if worldsheet_g1_metadata is None else worldsheet_g1_metadata.max_slots,
        "worldsheet_g1_semantic_parseback": worldsheet_g1_metadata is not None,
        "realization_profile_bytes": len(members.get(REALIZATION_PROFILE_MEMBER, b"")),
        "realization_profile_sha256": _sha256(members.get(REALIZATION_PROFILE_MEMBER, b"")),
        "realization_static_rule_bytes": len(members.get(REALIZATION_STATIC_RULE_MEMBER, b"")),
        "realization_static_rule_sha256": _sha256(members.get(REALIZATION_STATIC_RULE_MEMBER, b"")),
        "realization_static_rule_id": realization_static_rule_id,
        "realization_static_rule_active_sites": (
            0 if realization_static_rule_codes is None else int(np.count_nonzero(realization_static_rule_codes >= 0))
        ),
        "scorer_solved_template_bytes": len(members.get(SCORER_SOLVED_TEMPLATE_MEMBER, b"")),
        "scorer_solved_template_sha256": _sha256(members.get(SCORER_SOLVED_TEMPLATE_MEMBER, b"")),
        "scorer_solved_template_count": (
            0 if scorer_solved_templates is None else len(scorer_solved_templates.templates)
        ),
        "scorer_solved_template_parse_reencode_identical": (
            encode_scorer_solved_template_bank(scorer_solved_templates)
            == members.get(SCORER_SOLVED_TEMPLATE_MEMBER, b"")
        ),
        "scorer_solve_boundary": (
            None if scorer_solved_templates is None else "encode_side_only_frozen_scorer_through_exact_R"
        ),
        "decode_scorer_dependency": False,
        "camera_resolution_placement": realization_profile is not None,
        "movable_g1_replaces_inherited_mask": realization_profile is not None,
        "semantic_paint_order": (
            list(REALIZATION_PAINT_ORDER) if realization_profile is not None else list(COMPOSED_ROLE_ORDER)
        ),
        "worldsheet_persist_unless_event": bool(worldsheet_tracks or worldsheet_g1_payload),
        "worldsheet_stores_only_xi_deviations": bool(worldsheet_tracks),
        "lane_program_count": len(lane_programs),
        "lane_program_parse_reencode_identical": _encode_lane_programs(lane_programs)
        == members.get(LANE_PROGRAM_MEMBER, b""),
        "lane_knot_count": len(lane_knots),
        "lane_knot_parse_reencode_identical": _encode_lane_knots(lane_knots) == members.get(LANE_KNOT_MEMBER, b""),
        "lane_one_dash_phase_per_object_not_per_dash": bool(lane_programs),
        "lane_road_adjacency_constraint": "receiver intersects Lane with one-pixel dilation of inherited Road support",
        "region_coherent_chart_rerasterization": True,
        "pixel_coordinate_or_rgb_patch_present": scorer_solved_templates is not None,
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
        ("movable_worldsheet_lifecycle_shape", WORLDSHEET_TRACK_MEMBER),
        ("movable_worldsheet_xi_deviation_morph_knots", WORLDSHEET_KNOT_MEMBER),
        ("movable_g1_polygon_worldsheet_derivation", WORLDSHEET_G1_MEMBER),
        ("receiver_realization_profile", REALIZATION_PROFILE_MEMBER),
        ("receiver_static_cell_rule", REALIZATION_STATIC_RULE_MEMBER),
        ("encode_side_scorer_solved_shared_templates", SCORER_SOLVED_TEMPLATE_MEMBER),
        ("lane_periodic_phase_width_visibility", LANE_PROGRAM_MEMBER),
        ("lane_polynomial_drift_knots", LANE_KNOT_MEMBER),
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


def _sampled_member_payload_positions(archive: bytes) -> tuple[list[int], int]:
    """Byte offset inside each non-empty member payload, plus the denominator.

    Returned separately from the proof so the sample count can be checked
    against the member census: an offset or member-walk regression that yields
    zero positions must be a refusal, never a vacuous pass.
    """

    positions: list[int] = []
    payload_members = 0
    with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
        infos = reader.infolist()
        for info in infos:
            if info.file_size:
                payload_members += 1
                positions.append(
                    info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + info.file_size // 2
                )
    return positions, payload_members


def prove_carrier_archive_fail_closed(archive: bytes) -> dict[str, Any]:
    """Sample every outer home: a mutation must refuse or alter decoded RGB."""

    baseline = receive_carrier_compose_archive(archive)
    probe_ids = tuple(sorted({0, baseline.z.n_pairs - 1}))
    render_probe = baseline.render_camera_pairs if baseline.realization_profile is not None else baseline.render_pairs
    digest = hashlib.sha256(render_probe(probe_ids).tobytes()).hexdigest()
    _members, homes = parse_carrier_compose_archive(archive)
    positions, payload_members = _sampled_member_payload_positions(archive)
    # `refused + changed == len(positions)` is `0 == 0` on an empty walk, so the
    # whole proof would report True having mutated nothing.  Refuse instead: a
    # carrier archive always has at least the manifest and the predictor, and
    # this value is published as `fail_closed_mutation_proof` by 6 measurement
    # tools.
    if not positions or len(positions) != payload_members or payload_members < 2:
        raise DirectDescriptionError(
            "fail-closed mutation proof sampled no member payload homes: "
            f"{len(positions)} positions over {payload_members} non-empty members"
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
        candidate_render = (
            candidate.render_camera_pairs if candidate.realization_profile is not None else candidate.render_pairs
        )
        candidate_digest = hashlib.sha256(candidate_render(probe_ids).tobytes()).hexdigest()
        if candidate_digest == digest:
            raise DirectDescriptionError("sampled archive mutation was accepted as a receiver no-op")
        changed += 1
    return {
        "sampled_member_payload_homes": len(positions),
        "non_empty_member_payload_count": payload_members,
        "refused": refused,
        "changed_decode": changed,
        "all_samples_refused_or_changed_decode": refused + changed == len(positions),
        "unique_home_coverage_bytes": sum(row["zip_home_bytes"] for row in homes),
    }


__all__ = [
    "ARCHIVE_SCHEMA",
    "ARCHIVE_SCHEMA_V2",
    "ARCHIVE_SCHEMA_V3",
    "ARCHIVE_SCHEMA_V4",
    "ARCHIVE_SCHEMA_V5",
    "BOUNDARY_SHEARLET_MEMBER",
    "ISLAND_SHAPE_MEMBER",
    "REALIZATION_PROFILE_MEMBER",
    "REALIZATION_STATIC_RULE_MEMBER",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_V2",
    "RESULT_SCHEMA_V3",
    "RESULT_SCHEMA_V4",
    "RESULT_SCHEMA_V5",
    "WORLDSHEET_G1_MEMBER",
    "BoundaryCoefficientDelta",
    "BoundaryShearletAtomV1",
    "CarrierComposeReceiverV1",
    "DirectDescriptionV9CarrierComposeConfigV1",
    "DirectDescriptionV10FisherEventSearchConfigV1",
    "DirectDescriptionV11ObligationSearchConfigV1",
    "DirectDescriptionV12ObligationDrainConfigV1",
    "DirectDescriptionV13WorldsheetPredictorConfigV1",
    "IslandShapeAtomV1",
    "LaneDriftKnotV1",
    "LanePeriodicProgramV1",
    "MovableWorldsheetKnotV1",
    "MovableWorldsheetTrackV1",
    "ReceiverRealizationProfileV1",
    "TopologyEventV1",
    "compile_carrier_compose_archive",
    "encode_static_class_mask_rule",
    "parse_carrier_compose_archive",
    "prove_carrier_archive_fail_closed",
    "receive_carrier_compose_archive",
    "recursive_carrier_byte_rows",
    "requires_pose6_transport",
]
