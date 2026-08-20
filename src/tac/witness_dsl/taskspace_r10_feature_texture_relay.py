# SPDX-License-Identifier: MIT
"""Deterministic receiver-executable R10 feature/texture/luma/edge relay.

This module is a cross-level operator over an already-realized pair population.
It does not contain a scorer and it does not store target frames.  Generic image
operations are decoder code; every source-fitted scalar, phase, knot, selector,
trajectory, and spatial pullback is present in the counted packet.

The bounded receipts produced here are mechanism evidence only.  Changed pixels
are not d_seg/d_pose evidence, and only a complete n600 archive evaluation can
price this relay scientifically.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import IntEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.boundary_math.dash_phase_carrier import (
    DASH_PHASE_MAGIC,
    DecodedDash,
    decode_dash_phase_carrier,
)
from tac.boundary_math.stratified_depth_warp import (
    affine_extra_flow,
    stratified_warp_uint8,
)
from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    warp_frame0_uint8_numpy,
)
from tac.boundary_math.xi_pose_coder import (
    decode_xi_payload,
    parse_xi_payload,
    serialize_xi_payload,
)

R10_PACKET_SCHEMA: Final = "taskspace_r10_feature_texture_relay_packet.v1"
R10_RECEIPT_SCHEMA: Final = "taskspace_r10_feature_texture_relay_decode_receipt.v1"
R10_ADAPTER_SCHEMA: Final = "taskspace_r10_selected_solution_adapter.v1"
R10_MAGIC: Final = b"TSR10R1\x00"
R10_VERSION: Final = 1
R10_RATE_DENOMINATOR: Final = 37_545_489.0
R10_RECEIVER_OPERATION: Final = (
    "tac.witness_dsl.taskspace_r10_feature_texture_relay.decode_r10_packet"
)

R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED: Final = (
    "R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED"
)
R10_G23_PHYSICAL_GROUP_AND_LIFECYCLE_INTEGRATION_OWED: Final = (
    "R10_G23_PHYSICAL_GROUP_AND_LIFECYCLE_INTEGRATION_OWED"
)
R10_CROSS_HOST_BIT_IDENTITY_PROOF_OWED: Final = "R10_CROSS_HOST_BIT_IDENTITY_PROOF_OWED"
R10_CURRENT_SELECTED_SOLUTION_FITTED_OPERANDS_OWED: Final = (
    "R10_CURRENT_SELECTED_SOLUTION_FITTED_OPERANDS_OWED"
)
R10_EXACT_CONTEST_ROW_OWED: Final = "R10_EXACT_CONTEST_ROW_OWED"
R10_INFLATE_PUBLIC_RECEIVER_BUNDLE_CLOSURE_OWED: Final = (
    "R10_INFLATE_PUBLIC_RECEIVER_BUNDLE_CLOSURE_OWED"
)
R10_MAXIMUM_INVERSE_SOLVE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED: Final = (
    "R10_MAXIMUM_INVERSE_SOLVE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED"
)
R10_JOINT_DESCENT_EXACT_REALIZED_SCORE_ZIP_LINKER_OWED: Final = (
    "R10_JOINT_DESCENT_EXACT_REALIZED_SCORE_ZIP_LINKER_OWED"
)
R10_INFLATE_RUNTIME_30MIN_AND_MEMORY_16GB_PROOF_OWED: Final = (
    "R10_INFLATE_RUNTIME_30MIN_AND_MEMORY_16GB_PROOF_OWED"
)
R10_CPU_GPU_DECODE_PARITY_OWED: Final = "R10_CPU_GPU_DECODE_PARITY_OWED"
R10_PUBLIC_VIDEO_EMISSION_AND_UPSTREAM_EVALUATE_REPLAY_OWED: Final = (
    "R10_PUBLIC_VIDEO_EMISSION_AND_UPSTREAM_EVALUATE_REPLAY_OWED"
)

R10_BLOCKERS: Final = (
    R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED,
    R10_G23_PHYSICAL_GROUP_AND_LIFECYCLE_INTEGRATION_OWED,
    R10_CROSS_HOST_BIT_IDENTITY_PROOF_OWED,
    R10_CURRENT_SELECTED_SOLUTION_FITTED_OPERANDS_OWED,
    R10_INFLATE_PUBLIC_RECEIVER_BUNDLE_CLOSURE_OWED,
    R10_MAXIMUM_INVERSE_SOLVE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED,
    R10_JOINT_DESCENT_EXACT_REALIZED_SCORE_ZIP_LINKER_OWED,
    R10_INFLATE_RUNTIME_30MIN_AND_MEMORY_16GB_PROOF_OWED,
    R10_CPU_GPU_DECODE_PARITY_OWED,
    R10_PUBLIC_VIDEO_EMISSION_AND_UPSTREAM_EVALUATE_REPLAY_OWED,
    R10_EXACT_CONTEST_ROW_OWED,
)

R10_CONSTRAINTS: Final = (
    "AMPLITUDE",
    "FREQUENCY",
    "PHASE",
    "CONTRAST",
    "CHANNEL_ENERGY",
    "TEXTURE",
    "MULTIPLE_SHOOTING",
    "FROZEN_FEATURE",
)

_HEADER = struct.Struct("<8sBBHHHHH32s32s32s")
_DIRECTORY = struct.Struct("<HHIII")
_BASE_FEATURE = struct.Struct("<hhhhhhh")
_TEXTURE = struct.Struct("<hHHh")
_SHOOTING_KNOT = struct.Struct("<Hhhhhhh")
_FLOW = struct.Struct("<Hhhhhhh")
_GEOMETRY = struct.Struct("<i")

# angle k*pi/16, represented as (cos, sin) in Q14.  This static generic table is
# decoder mechanism, not source-fitted data.
_DIRECTION_Q14: Final = (
    (16384, 0),
    (16069, 3196),
    (15137, 6270),
    (13623, 9102),
    (11585, 11585),
    (9102, 13623),
    (6270, 15137),
    (3196, 16069),
    (0, 16384),
    (-3196, 16069),
    (-6270, 15137),
    (-9102, 13623),
    (-11585, 11585),
    (-13623, 9102),
    (-15137, 6270),
    (-16069, 3196),
)


class R10RelayError(ValueError):
    """A packet, identity, spatial-support, or receiver invariant failed."""


class R10RelayMode(IntEnum):
    RELAY_ONLY = 0
    JOINT = 1


class R10Section(IntEnum):
    PAIR_INDEX = 1
    GEOMETRY = 2
    BASE_FEATURE = 3
    TEXTURE = 4
    SHOOTING_KNOT = 5
    XIP2 = 6
    DASH1 = 7
    PULLBACK_POLYGON = 8
    STRATIFIED_FLOW = 9


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_sha256(value: str, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise R10RelayError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise R10RelayError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    if decoded.hex() != value:
        raise R10RelayError(f"{name} must be canonical lowercase hex")
    return value


def _i16(value: int, *, name: str) -> int:
    if type(value) is not int or not (-32768 <= value <= 32767):
        raise R10RelayError(f"{name} must fit signed int16")
    return value


def _u16(value: int, *, name: str) -> int:
    if type(value) is not int or not (0 <= value <= 65535):
        raise R10RelayError(f"{name} must fit uint16")
    return value


def _hex_to_raw(value: str, *, name: str) -> bytes:
    return bytes.fromhex(_validate_sha256(value, name=name))


def realization_sha256(pairs: np.ndarray) -> str:
    """Hash one exact uint8 realized-pair population, including shape."""

    arr = np.asarray(pairs)
    if arr.dtype != np.uint8 or arr.ndim != 5 or arr.shape[1] != 2 or arr.shape[-1] != 3:
        raise R10RelayError("realized pairs must be uint8 [P,2,H,W,3]")
    shape = struct.pack("<5I", *(int(v) for v in arr.shape))
    digest = hashlib.sha256()
    digest.update(b"R10_REALIZED_PAIR_V1\x00")
    digest.update(shape)
    digest.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return digest.hexdigest()


def pair_population_sha256(source_sha256: str, pair_indices: Sequence[int]) -> str:
    """Canonical identity for one ordered source-bound pair population."""

    source = _validate_sha256(source_sha256, name="source_sha256")
    indices = tuple(int(v) for v in pair_indices)
    if not indices or indices != tuple(range(len(indices))) or len(indices) > 600:
        raise R10RelayError("pair indices must be canonical contiguous order in [0,600)")
    return _sha256(
        _canonical_json(
            {
                "schema": "taskspace_r10_pair_population.v1",
                "source_sha256": source,
                "pair_indices": list(indices),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class R10IdentityV1:
    source_sha256: str
    pair_population_sha256: str
    base_realization_sha256: str

    def __post_init__(self) -> None:
        _validate_sha256(self.source_sha256, name="source_sha256")
        _validate_sha256(self.pair_population_sha256, name="pair_population_sha256")
        _validate_sha256(self.base_realization_sha256, name="base_realization_sha256")


@dataclass(frozen=True, slots=True)
class R10BaseFeatureRecordV1:
    luma_bias_q8: int
    contrast_q8: int
    edge_gain_q8: int
    feature_gain_q8: int
    channel_weights_q8: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name in ("luma_bias_q8", "contrast_q8", "edge_gain_q8", "feature_gain_q8"):
            _i16(getattr(self, name), name=name)
        if type(self.channel_weights_q8) is not tuple or len(self.channel_weights_q8) != 3:
            raise R10RelayError("channel_weights_q8 must be an exact 3-tuple")
        for value in self.channel_weights_q8:
            _i16(value, name="channel weight")
        if not any(self.channel_weights_q8):
            raise R10RelayError("at least one channel-energy weight must be nonzero")


@dataclass(frozen=True, slots=True)
class R10TextureRecordV1:
    amplitude_q8: int
    frequency_q8: int
    phase_q10: int
    texture_gain_q8: int

    def __post_init__(self) -> None:
        _i16(self.amplitude_q8, name="amplitude_q8")
        _u16(self.frequency_q8, name="frequency_q8")
        _u16(self.phase_q10, name="phase_q10")
        _i16(self.texture_gain_q8, name="texture_gain_q8")
        if self.frequency_q8 == 0:
            raise R10RelayError("texture frequency must be nonzero")
        if self.phase_q10 >= 1024:
            raise R10RelayError("phase_q10 must be in [0,1024)")


@dataclass(frozen=True, slots=True)
class R10ShootingKnotV1:
    pair_index: int
    luma_bias_delta_q8: int
    contrast_delta_q8: int
    edge_delta_q8: int
    amplitude_delta_q8: int
    phase_delta_q10: int
    texture_delta_q8: int

    def __post_init__(self) -> None:
        _u16(self.pair_index, name="shooting pair index")
        for name in (
            "luma_bias_delta_q8",
            "contrast_delta_q8",
            "edge_delta_q8",
            "amplitude_delta_q8",
            "phase_delta_q10",
            "texture_delta_q8",
        ):
            _i16(getattr(self, name), name=name)

    @property
    def deltas(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.luma_bias_delta_q8,
            self.contrast_delta_q8,
            self.edge_delta_q8,
            self.amplitude_delta_q8,
            self.phase_delta_q10,
            self.texture_delta_q8,
        )


@dataclass(frozen=True, slots=True)
class R10PullbackPolygonV1:
    pair_index: int
    vertices_q15: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _u16(self.pair_index, name="polygon pair index")
        if type(self.vertices_q15) is not tuple or not (3 <= len(self.vertices_q15) <= 32):
            raise R10RelayError("pullback polygon needs 3..32 exact vertices")
        for point in self.vertices_q15:
            if type(point) is not tuple or len(point) != 2:
                raise R10RelayError("pullback vertex must be an exact (x_q15,y_q15) tuple")
            if not all(type(v) is int and 0 <= v <= 32767 for v in point):
                raise R10RelayError("pullback coordinates must be uint15 normalized values")


@dataclass(frozen=True, slots=True)
class R10StratifiedFlowV1:
    pair_index: int
    affine_q8: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        _u16(self.pair_index, name="flow pair index")
        if type(self.affine_q8) is not tuple or len(self.affine_q8) != 6:
            raise R10RelayError("stratified flow must be an exact 6-tuple")
        for value in self.affine_q8:
            _i16(value, name="stratified flow coefficient")


def _read_dash_header(section: bytes) -> Mapping[str, Any]:
    if not section.startswith(DASH_PHASE_MAGIC):
        raise R10RelayError("DASH1 section has bad magic")
    off = len(DASH_PHASE_MAGIC)
    if len(section) < off + 4:
        raise R10RelayError("DASH1 section is truncated before header length")
    (size,) = struct.unpack_from("<I", section, off)
    off += 4
    if size > len(section) - off:
        raise R10RelayError("DASH1 header length exceeds section")
    try:
        header = json.loads(section[off : off + size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R10RelayError("DASH1 header is not valid UTF-8 JSON") from exc
    if type(header) is not dict:
        raise R10RelayError("DASH1 header must be a JSON object")
    return header


def _canonical_xip2(payload: bytes, *, n_pairs: int) -> tuple[bytes, np.ndarray]:
    if type(payload) is not bytes or len(payload) < 8 or payload[:4] != b"XIP2":
        raise R10RelayError("XIP2 section is missing or malformed")
    coder_id = payload[4]
    coder_by_id = {0: "none", 1: "delta_ar", 3: "delta_res"}
    coder = coder_by_id.get(coder_id)
    if coder is None:
        raise R10RelayError("R10 accepts canonical XIP2 none/delta_ar/delta_res only")
    try:
        q, scales = parse_xi_payload(payload)
        canonical = serialize_xi_payload(q, scales, coder=coder)
        xi = decode_xi_payload(payload)
    except (ValueError, struct.error, zlib.error) as exc:
        raise R10RelayError(f"XIP2 strict decode failed: {exc}") from exc
    if canonical != payload:
        raise R10RelayError("XIP2 section is not canonical parse/re-emitted bytes")
    if q.shape != (n_pairs, 6) or xi.shape != (n_pairs, 6):
        raise R10RelayError(f"XIP2 population must be ({n_pairs},6)")
    if not np.isfinite(xi).all():
        raise R10RelayError("XIP2 decoded non-finite twist")
    return canonical, xi


@dataclass(frozen=True, slots=True)
class R10PacketV1:
    mode: R10RelayMode
    identity: R10IdentityV1
    pair_indices: tuple[int, ...]
    height: int
    width: int
    pitch_q20: int
    base_features: tuple[R10BaseFeatureRecordV1, ...]
    xip2_payload: bytes
    textures: tuple[R10TextureRecordV1, ...] | None = None
    shooting_knots: tuple[R10ShootingKnotV1, ...] = ()
    dash1_payload: bytes | None = None
    pullback_polygons: tuple[R10PullbackPolygonV1, ...] = ()
    stratified_flows: tuple[R10StratifiedFlowV1, ...] = ()

    def __post_init__(self) -> None:
        if type(self.mode) is not R10RelayMode:
            raise R10RelayError("relay mode must be typed")
        if type(self.identity) is not R10IdentityV1:
            raise R10RelayError("packet identity must be typed")
        if type(self.pair_indices) is not tuple or not self.pair_indices:
            raise R10RelayError("packet needs an exact nonempty pair tuple")
        n_pairs = len(self.pair_indices)
        if n_pairs > 600 or self.pair_indices != tuple(range(n_pairs)):
            raise R10RelayError("pair coordinates must be canonical contiguous order, max 600")
        if not (1 <= self.height <= 65535 and 1 <= self.width <= 65535):
            raise R10RelayError("receiver dimensions must fit nonzero uint16")
        if type(self.pitch_q20) is not int or not (-(1 << 31) <= self.pitch_q20 < (1 << 31)):
            raise R10RelayError("pitch_q20 must fit signed int32")
        if type(self.base_features) is not tuple or len(self.base_features) != n_pairs:
            raise R10RelayError("BASE_FEATURE needs exactly one typed record per pair")
        if any(type(row) is not R10BaseFeatureRecordV1 for row in self.base_features):
            raise R10RelayError("BASE_FEATURE contains an untyped record")
        if type(self.xip2_payload) is not bytes:
            raise R10RelayError("XIP2 payload must be exact bytes")
        _, xi = _canonical_xip2(self.xip2_payload, n_pairs=n_pairs)
        if self.textures is not None:
            if type(self.textures) is not tuple or len(self.textures) != n_pairs:
                raise R10RelayError("TEXTURE needs exactly one typed record per pair")
            if any(type(row) is not R10TextureRecordV1 for row in self.textures):
                raise R10RelayError("TEXTURE contains an untyped record")
            if type(self.dash1_payload) is not bytes or not self.dash1_payload:
                raise R10RelayError(
                    "coefficient-only texture is spatially nonidentifiable without counted DASH1 pullback"
                )
        elif self.dash1_payload is not None:
            raise R10RelayError("DASH1 without TEXTURE would be an unconsumed counted operand")
        if type(self.shooting_knots) is not tuple or any(
            type(row) is not R10ShootingKnotV1 for row in self.shooting_knots
        ):
            raise R10RelayError("SHOOTING_KNOT must be an exact typed tuple")
        knot_indices = tuple(row.pair_index for row in self.shooting_knots)
        if knot_indices != tuple(sorted(set(knot_indices))) or any(v >= n_pairs for v in knot_indices):
            raise R10RelayError("shooting knots must be unique, ordered, and inside the population")
        if self.textures is None and any(
            row.amplitude_delta_q8 or row.phase_delta_q10 or row.texture_delta_q8
            for row in self.shooting_knots
        ):
            raise R10RelayError("texture-free packet retained live texture shooting operands")
        if self.dash1_payload is not None:
            header = _read_dash_header(self.dash1_payload)
            if bool(header.get("include_xi", True)):
                raise R10RelayError("DASH1 must consume the single XIP2 owner, not duplicate xi bytes")
            if (
                int(header.get("n_frames", -1)) != n_pairs
                or int(header.get("height", -1)) != self.height
                or int(header.get("width", -1)) != self.width
            ):
                raise R10RelayError("DASH1 population/geometry differs from the R10 packet")
            expected_pitch = self.pitch_q20 / float(1 << 20)
            if float(header.get("pitch", math.nan)) != expected_pitch:
                raise R10RelayError("DASH1 pitch differs from the counted R10 geometry operand")
            try:
                decode_dash_phase_carrier(
                    self.dash1_payload,
                    xi_twists_external=xi,
                    geom=GroundHomographyGeom.eon(
                        native_hw=(self.height, self.width), pitch=expected_pitch
                    ),
                )
            except (ValueError, struct.error, IndexError, KeyError) as exc:
                raise R10RelayError(f"DASH1 strict decode failed: {exc}") from exc
        if type(self.pullback_polygons) is not tuple or any(
            type(row) is not R10PullbackPolygonV1 for row in self.pullback_polygons
        ):
            raise R10RelayError("PULLBACK_POLYGON must be an exact typed tuple")
        if type(self.stratified_flows) is not tuple or any(
            type(row) is not R10StratifiedFlowV1 for row in self.stratified_flows
        ):
            raise R10RelayError("STRATIFIED_FLOW must be an exact typed tuple")
        polygon_indices = tuple(row.pair_index for row in self.pullback_polygons)
        flow_indices = tuple(row.pair_index for row in self.stratified_flows)
        if polygon_indices != tuple(sorted(set(polygon_indices))) or any(v >= n_pairs for v in polygon_indices):
            raise R10RelayError("pullback polygons must be unique, ordered, and inside population")
        if flow_indices != tuple(sorted(set(flow_indices))) or any(v >= n_pairs for v in flow_indices):
            raise R10RelayError("stratified flows must be unique, ordered, and inside population")
        if set(flow_indices) != set(polygon_indices):
            raise R10RelayError(
                "stratified coefficients are spatially nonidentifiable without one counted polygon per flow"
            )

    @property
    def n_pairs(self) -> int:
        return len(self.pair_indices)

    @property
    def pitch(self) -> float:
        return self.pitch_q20 / float(1 << 20)


def _encode_pair_indices(packet: R10PacketV1) -> bytes:
    return struct.pack(f"<{packet.n_pairs}H", *packet.pair_indices)


def _encode_base_features(packet: R10PacketV1) -> bytes:
    return b"".join(
        _BASE_FEATURE.pack(
            row.luma_bias_q8,
            row.contrast_q8,
            row.edge_gain_q8,
            row.feature_gain_q8,
            *row.channel_weights_q8,
        )
        for row in packet.base_features
    )


def _encode_textures(packet: R10PacketV1) -> bytes:
    if packet.textures is None:
        return b""
    return b"".join(
        _TEXTURE.pack(
            row.amplitude_q8,
            row.frequency_q8,
            row.phase_q10,
            row.texture_gain_q8,
        )
        for row in packet.textures
    )


def _encode_shooting(packet: R10PacketV1) -> bytes:
    return b"".join(
        _SHOOTING_KNOT.pack(row.pair_index, *row.deltas) for row in packet.shooting_knots
    )


def _encode_polygons(packet: R10PacketV1) -> bytes:
    if not packet.pullback_polygons:
        return b""
    out = bytearray(struct.pack("<H", len(packet.pullback_polygons)))
    for row in packet.pullback_polygons:
        out += struct.pack("<HBB", row.pair_index, len(row.vertices_q15), 0)
        for x_q15, y_q15 in row.vertices_q15:
            out += struct.pack("<HH", x_q15, y_q15)
    return bytes(out)


def _encode_flows(packet: R10PacketV1) -> bytes:
    if not packet.stratified_flows:
        return b""
    return struct.pack("<H", len(packet.stratified_flows)) + b"".join(
        _FLOW.pack(row.pair_index, *row.affine_q8) for row in packet.stratified_flows
    )


def _section_payloads(packet: R10PacketV1) -> dict[R10Section, bytes]:
    sections = {
        R10Section.PAIR_INDEX: _encode_pair_indices(packet),
        R10Section.GEOMETRY: _GEOMETRY.pack(packet.pitch_q20),
        R10Section.BASE_FEATURE: _encode_base_features(packet),
        R10Section.XIP2: packet.xip2_payload,
    }
    if packet.textures is not None:
        sections[R10Section.TEXTURE] = _encode_textures(packet)
        sections[R10Section.DASH1] = packet.dash1_payload or b""
    if packet.shooting_knots:
        sections[R10Section.SHOOTING_KNOT] = _encode_shooting(packet)
    if packet.pullback_polygons:
        sections[R10Section.PULLBACK_POLYGON] = _encode_polygons(packet)
        sections[R10Section.STRATIFIED_FLOW] = _encode_flows(packet)
    return dict(sorted(sections.items(), key=lambda item: int(item[0])))


def serialize_r10_packet(packet: R10PacketV1) -> bytes:
    """Emit canonical counted packet bytes."""

    if type(packet) is not R10PacketV1:
        raise R10RelayError("serialize requires the exact R10PacketV1 type")
    sections = _section_payloads(packet)
    count = len(sections)
    payload_offset = _HEADER.size + count * _DIRECTORY.size
    directory = bytearray()
    payload = bytearray()
    cursor = payload_offset
    for section, blob in sections.items():
        directory += _DIRECTORY.pack(
            int(section),
            0,
            cursor,
            len(blob),
            zlib.crc32(blob) & 0xFFFFFFFF,
        )
        payload += blob
        cursor += len(blob)
    header = _HEADER.pack(
        R10_MAGIC,
        R10_VERSION,
        int(packet.mode),
        0,
        packet.n_pairs,
        packet.height,
        packet.width,
        count,
        _hex_to_raw(packet.identity.source_sha256, name="source_sha256"),
        _hex_to_raw(packet.identity.pair_population_sha256, name="pair_population_sha256"),
        _hex_to_raw(packet.identity.base_realization_sha256, name="base_realization_sha256"),
    )
    return header + bytes(directory) + bytes(payload)


@dataclass(frozen=True, slots=True)
class R10PhysicalSpanV1:
    section: str
    byte_offset: int
    byte_length: int
    bit_start: int
    bit_stop: int
    range_sha256: str


def packet_physical_spans(payload: bytes) -> tuple[R10PhysicalSpanV1, ...]:
    """Return exact counted section ranges after strict parsing."""

    _packet, directory = _parse_r10_packet_with_directory(payload)
    return tuple(
        R10PhysicalSpanV1(
            section=section.name,
            byte_offset=offset,
            byte_length=length,
            bit_start=offset * 8,
            bit_stop=(offset + length) * 8,
            range_sha256=_sha256(payload[offset : offset + length]),
        )
        for section, offset, length in directory
    )


def _parse_polygons(blob: bytes, *, n_pairs: int) -> tuple[R10PullbackPolygonV1, ...]:
    if len(blob) < 2:
        raise R10RelayError("PULLBACK_POLYGON is truncated")
    (count,) = struct.unpack_from("<H", blob, 0)
    off = 2
    rows: list[R10PullbackPolygonV1] = []
    for _ in range(count):
        if off + 4 > len(blob):
            raise R10RelayError("PULLBACK_POLYGON record header is truncated")
        pair_index, n_vertices, reserved = struct.unpack_from("<HBB", blob, off)
        off += 4
        if reserved != 0 or not (3 <= n_vertices <= 32):
            raise R10RelayError("PULLBACK_POLYGON record has invalid flags/count")
        end = off + n_vertices * 4
        if end > len(blob):
            raise R10RelayError("PULLBACK_POLYGON vertices are truncated")
        vertices = tuple(struct.unpack_from("<HH", blob, off + i * 4) for i in range(n_vertices))
        off = end
        rows.append(R10PullbackPolygonV1(int(pair_index), vertices))
    if off != len(blob) or count > n_pairs:
        raise R10RelayError("PULLBACK_POLYGON has trailing bytes or excessive records")
    return tuple(rows)


def _parse_r10_packet_with_directory(
    payload: bytes,
) -> tuple[R10PacketV1, tuple[tuple[R10Section, int, int], ...]]:
    if type(payload) is not bytes or len(payload) < _HEADER.size:
        raise R10RelayError("R10 packet must be nonempty exact bytes")
    try:
        (
            magic,
            version,
            mode_raw,
            flags,
            n_pairs,
            height,
            width,
            section_count,
            source_raw,
            population_raw,
            base_raw,
        ) = _HEADER.unpack_from(payload, 0)
    except struct.error as exc:
        raise R10RelayError("R10 fixed header is truncated") from exc
    if magic != R10_MAGIC or version != R10_VERSION or flags != 0:
        raise R10RelayError("R10 magic/version/flags are not canonical")
    try:
        mode = R10RelayMode(mode_raw)
    except ValueError as exc:
        raise R10RelayError("R10 relay mode is unknown") from exc
    if not (1 <= n_pairs <= 600 and height > 0 and width > 0):
        raise R10RelayError("R10 header geometry/population is invalid")
    directory_end = _HEADER.size + section_count * _DIRECTORY.size
    if directory_end > len(payload):
        raise R10RelayError("R10 section directory is truncated")
    directory: list[tuple[R10Section, int, int]] = []
    section_blobs: dict[R10Section, bytes] = {}
    cursor = directory_end
    previous_id = 0
    for index in range(section_count):
        sid, reserved, offset, length, crc = _DIRECTORY.unpack_from(
            payload, _HEADER.size + index * _DIRECTORY.size
        )
        try:
            section = R10Section(sid)
        except ValueError as exc:
            raise R10RelayError(f"R10 packet has unknown section id {sid}") from exc
        if reserved != 0 or sid <= previous_id or offset != cursor or offset + length > len(payload):
            raise R10RelayError("R10 directory is unsorted, overlapping, gapped, or out of range")
        blob = payload[offset : offset + length]
        if zlib.crc32(blob) & 0xFFFFFFFF != crc:
            raise R10RelayError(f"R10 {section.name} CRC mismatch")
        previous_id = sid
        cursor += length
        directory.append((section, offset, length))
        section_blobs[section] = blob
    if cursor != len(payload):
        raise R10RelayError("R10 packet has trailing bytes")
    required = {
        R10Section.PAIR_INDEX,
        R10Section.GEOMETRY,
        R10Section.BASE_FEATURE,
        R10Section.XIP2,
    }
    if not required.issubset(section_blobs):
        raise R10RelayError("R10 packet is missing a required physical section")
    pair_blob = section_blobs[R10Section.PAIR_INDEX]
    if len(pair_blob) != n_pairs * 2:
        raise R10RelayError("PAIR_INDEX length differs from population")
    pair_indices = tuple(int(v) for v in struct.unpack(f"<{n_pairs}H", pair_blob))
    geometry_blob = section_blobs[R10Section.GEOMETRY]
    if len(geometry_blob) != _GEOMETRY.size:
        raise R10RelayError("GEOMETRY has noncanonical length")
    (pitch_q20,) = _GEOMETRY.unpack(geometry_blob)
    base_blob = section_blobs[R10Section.BASE_FEATURE]
    if len(base_blob) != n_pairs * _BASE_FEATURE.size:
        raise R10RelayError("BASE_FEATURE length differs from population")
    base_features = tuple(
        R10BaseFeatureRecordV1(
            *values[:4],
            channel_weights_q8=(values[4], values[5], values[6]),
        )
        for values in (
            _BASE_FEATURE.unpack_from(base_blob, index * _BASE_FEATURE.size)
            for index in range(n_pairs)
        )
    )
    texture_blob = section_blobs.get(R10Section.TEXTURE)
    textures: tuple[R10TextureRecordV1, ...] | None = None
    if texture_blob is not None:
        if len(texture_blob) != n_pairs * _TEXTURE.size:
            raise R10RelayError("TEXTURE length differs from population")
        textures = tuple(
            R10TextureRecordV1(*_TEXTURE.unpack_from(texture_blob, index * _TEXTURE.size))
            for index in range(n_pairs)
        )
    shooting_blob = section_blobs.get(R10Section.SHOOTING_KNOT, b"")
    if len(shooting_blob) % _SHOOTING_KNOT.size:
        raise R10RelayError("SHOOTING_KNOT has a partial record")
    shooting_knots = tuple(
        R10ShootingKnotV1(*_SHOOTING_KNOT.unpack_from(shooting_blob, off))
        for off in range(0, len(shooting_blob), _SHOOTING_KNOT.size)
    )
    polygon_blob = section_blobs.get(R10Section.PULLBACK_POLYGON, b"")
    polygons = _parse_polygons(polygon_blob, n_pairs=n_pairs) if polygon_blob else ()
    flow_blob = section_blobs.get(R10Section.STRATIFIED_FLOW, b"")
    flows: tuple[R10StratifiedFlowV1, ...] = ()
    if flow_blob:
        if len(flow_blob) < 2:
            raise R10RelayError("STRATIFIED_FLOW is truncated")
        (flow_count,) = struct.unpack_from("<H", flow_blob, 0)
        if len(flow_blob) != 2 + flow_count * _FLOW.size:
            raise R10RelayError("STRATIFIED_FLOW length/count mismatch")
        flows = tuple(
            R10StratifiedFlowV1(
                values[0],
                tuple(values[1:]),
            )
            for values in (
                _FLOW.unpack_from(flow_blob, 2 + index * _FLOW.size)
                for index in range(flow_count)
            )
        )
    packet = R10PacketV1(
        mode=mode,
        identity=R10IdentityV1(source_raw.hex(), population_raw.hex(), base_raw.hex()),
        pair_indices=pair_indices,
        height=int(height),
        width=int(width),
        pitch_q20=int(pitch_q20),
        base_features=base_features,
        xip2_payload=section_blobs[R10Section.XIP2],
        textures=textures,
        shooting_knots=shooting_knots,
        dash1_payload=section_blobs.get(R10Section.DASH1),
        pullback_polygons=polygons,
        stratified_flows=flows,
    )
    if serialize_r10_packet(packet) != payload:
        raise R10RelayError("R10 packet is not canonical parse/re-emitted bytes")
    return packet, tuple(directory)


def parse_r10_packet(payload: bytes) -> R10PacketV1:
    """Strict parse, dependency validation, and canonical re-emission."""

    packet, _directory = _parse_r10_packet_with_directory(payload)
    return packet


def _signed_round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise R10RelayError("rounding denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _signed_round_shift(values: np.ndarray, shift: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    bias = 1 << (shift - 1)
    return np.where(arr < 0, -((-arr + bias) >> shift), (arr + bias) >> shift)


def _shooting_delta(knots: tuple[R10ShootingKnotV1, ...], pair_index: int) -> np.ndarray:
    if not knots:
        return np.zeros(6, dtype=np.int64)
    if pair_index <= knots[0].pair_index:
        return np.asarray(knots[0].deltas, dtype=np.int64)
    if pair_index >= knots[-1].pair_index:
        return np.asarray(knots[-1].deltas, dtype=np.int64)
    for left, right in pairwise(knots):
        if left.pair_index <= pair_index <= right.pair_index:
            width = right.pair_index - left.pair_index
            step = pair_index - left.pair_index
            a = np.asarray(left.deltas, dtype=np.int64)
            b = np.asarray(right.deltas, dtype=np.int64)
            return np.asarray(
                [
                    int(a[i]) + _signed_round_div(int((b[i] - a[i]) * step), width)
                    for i in range(6)
                ],
                dtype=np.int64,
            )
    raise AssertionError("unreachable shooting interval")


def _polygon_mask(height: int, width: int, polygon: R10PullbackPolygonV1) -> np.ndarray:
    vertices = [
        (
            (x_q15 * (width - 1) + 16383) // 32767,
            (y_q15 * (height - 1) + 16383) // 32767,
        )
        for x_q15, y_q15 in polygon.vertices_q15
    ]
    xs = np.arange(width, dtype=np.int64)[None, :]
    ys = np.arange(height, dtype=np.int64)[:, None]
    inside = np.zeros((height, width), dtype=bool)
    for (xi, yi), (xj, yj) in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        crosses_y = (yi > ys) != (yj > ys)
        denominator = yj - yi
        if denominator == 0:
            continue
        lhs = (xs - xi) * denominator
        rhs = (xj - xi) * (ys - yi)
        crosses_x = lhs < rhs if denominator > 0 else lhs > rhs
        inside ^= crosses_y & crosses_x
    return inside


def _integer_luma(rgb: np.ndarray) -> np.ndarray:
    work = np.asarray(rgb, dtype=np.int64)
    return (77 * work[..., 0] + 150 * work[..., 1] + 29 * work[..., 2] + 128) >> 8


def _feature_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    luma = _integer_luma(rgb)
    up = np.vstack((luma[:1], luma[:-1]))
    down = np.vstack((luma[1:], luma[-1:]))
    left = np.hstack((luma[:, :1], luma[:, :-1]))
    right = np.hstack((luma[:, 1:], luma[:, -1:]))
    laplacian = 4 * luma - up - down - left - right
    gradient = np.minimum(255, np.abs(right - left) + np.abs(down - up))
    feature = gradient - _signed_round_div(int(gradient.sum()), int(gradient.size))
    return luma, laplacian, feature


def _triangle_q8(phase_q10: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(phase_q10, dtype=np.int64) & 1023
    return 256 - np.abs(wrapped - 512)


def _texture_carrier(
    height: int,
    width: int,
    dashes: Sequence[DecodedDash],
    *,
    frequency_q8: int,
    phase_q10: int,
    frame_role: int,
) -> np.ndarray:
    carrier = np.zeros((height, width), dtype=np.int64)
    frame_phase = int(phase_q10) - (int(frequency_q8) << 1) * (1 - int(frame_role))
    for dash in dashes:
        row = round(dash.centroid_rc[0])
        col = round(dash.centroid_rc[1])
        radius = max(2, min(48, 2 * math.isqrt(max(1, int(dash.area)))))
        tangent_radius = min(96, radius * 2)
        r0, r1 = max(0, row - tangent_radius), min(height, row + tangent_radius + 1)
        c0, c1 = max(0, col - tangent_radius), min(width, col + tangent_radius + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        tilt_bin = round(float(dash.tilt) * 16.0 / math.pi) % 16
        cos_q14, sin_q14 = _DIRECTION_Q14[tilt_bin]
        dr = np.arange(r0, r1, dtype=np.int64)[:, None] - row
        dc = np.arange(c0, c1, dtype=np.int64)[None, :] - col
        along_q14 = dr * sin_q14 + dc * cos_q14
        normal_q14 = dr * cos_q14 - dc * sin_q14
        abs_along = np.abs(along_q14)
        abs_normal = np.abs(normal_q14)
        support = (abs_normal <= radius * 16384) & (abs_along <= tangent_radius * 16384)
        phase = frame_phase + ((along_q14 * int(frequency_q8)) >> 12)
        wave = _triangle_q8(phase)
        taper_normal = np.maximum(0, 256 - (abs_normal * 256) // (radius * 16384 + 1))
        taper_along = np.maximum(
            0,
            256 - (abs_along * 256) // (tangent_radius * 16384 + 1),
        )
        local = _signed_round_shift(wave * taper_normal * taper_along, 16)
        view = carrier[r0:r1, c0:c1]
        view += np.where(support, local, 0)
        np.clip(view, -512, 512, out=view)
    return carrier


def _apply_feature_relay(
    frame: np.ndarray,
    base: R10BaseFeatureRecordV1,
    texture: R10TextureRecordV1 | None,
    dashes: Sequence[DecodedDash],
    knot_delta: np.ndarray,
    *,
    frame_role: int,
) -> np.ndarray:
    rgb = np.asarray(frame, dtype=np.uint8)
    luma, laplacian, feature = _feature_maps(rgb)
    luma_bias = int(base.luma_bias_q8) + int(knot_delta[0])
    contrast = int(base.contrast_q8) + int(knot_delta[1])
    edge = int(base.edge_gain_q8) + int(knot_delta[2])
    scalar_q8 = (
        luma_bias
        + contrast * (luma - 128)
        + edge * laplacian
        + int(base.feature_gain_q8) * feature
    )
    if texture is not None:
        amplitude = int(texture.amplitude_q8) + int(knot_delta[3])
        phase = (int(texture.phase_q10) + int(knot_delta[4])) & 1023
        texture_gain = int(texture.texture_gain_q8) + int(knot_delta[5])
        carrier = _texture_carrier(
            rgb.shape[0],
            rgb.shape[1],
            dashes,
            frequency_q8=texture.frequency_q8,
            phase_q10=phase,
            frame_role=frame_role,
        )
        scalar_q8 = scalar_q8 + _signed_round_shift(
            carrier * amplitude * texture_gain,
            16,
        )
    weights = np.asarray(base.channel_weights_q8, dtype=np.int64)
    delta_q8 = _signed_round_shift(scalar_q8[..., None] * weights[None, None, :], 8)
    delta = _signed_round_shift(delta_q8, 8)
    return np.clip(rgb.astype(np.int64) + delta, 0, 255).astype(np.uint8)


def _validate_decode_identity(
    packet: R10PacketV1,
    base_pairs: np.ndarray,
    *,
    expected_source_sha256: str,
    expected_pair_population_sha256: str,
) -> np.ndarray:
    source = _validate_sha256(expected_source_sha256, name="expected_source_sha256")
    population = _validate_sha256(
        expected_pair_population_sha256,
        name="expected_pair_population_sha256",
    )
    if source != packet.identity.source_sha256:
        raise R10RelayError("source identity differs from counted packet source")
    if population != packet.identity.pair_population_sha256:
        raise R10RelayError("pair-population identity differs from counted packet population")
    base = np.asarray(base_pairs)
    expected_shape = (packet.n_pairs, 2, packet.height, packet.width, 3)
    if base.dtype != np.uint8 or base.shape != expected_shape:
        raise R10RelayError(f"base realized pairs must be uint8 {expected_shape}")
    if realization_sha256(base) != packet.identity.base_realization_sha256:
        raise R10RelayError("base realized-pair bytes drifted from packet dependency")
    return np.ascontiguousarray(base)


def decode_r10_packet(
    packet_bytes: bytes,
    base_pairs: np.ndarray,
    *,
    expected_source_sha256: str,
    expected_pair_population_sha256: str,
) -> np.ndarray:
    """Execute R10 and return actual deterministic uint8 ``[P,2,H,W,3]``."""

    packet = parse_r10_packet(packet_bytes)
    base = _validate_decode_identity(
        packet,
        base_pairs,
        expected_source_sha256=expected_source_sha256,
        expected_pair_population_sha256=expected_pair_population_sha256,
    )
    xi = decode_xi_payload(packet.xip2_payload)
    geom = GroundHomographyGeom.eon(
        native_hw=(packet.height, packet.width),
        pitch=packet.pitch,
    )
    output = (
        np.full_like(base, 128)
        if packet.mode is R10RelayMode.RELAY_ONLY
        else base.copy()
    )
    polygons = {row.pair_index: row for row in packet.pullback_polygons}
    flows = {row.pair_index: row for row in packet.stratified_flows}
    for pair_index in packet.pair_indices:
        if pair_index in flows:
            polygon = polygons[pair_index]
            mask = _polygon_mask(packet.height, packet.width, polygon)
            coefficients = np.asarray(flows[pair_index].affine_q8, dtype=np.float64) / 256.0
            extra_flow = affine_extra_flow((packet.height, packet.width), coefficients)
            output[pair_index, 0] = stratified_warp_uint8(
                output[pair_index, 0],
                xi[pair_index],
                geom,
                offplane_mask=mask,
                extra_flow=extra_flow,
            )
        else:
            output[pair_index, 0] = warp_frame0_uint8_numpy(
                output[pair_index, 0],
                xi[pair_index],
                geom,
            )
    dash_frames: Sequence[Sequence[DecodedDash]]
    if packet.dash1_payload is None:
        dash_frames = tuple(() for _ in packet.pair_indices)
    else:
        dash_frames = decode_dash_phase_carrier(
            packet.dash1_payload,
            xi_twists_external=xi,
            geom=geom,
        )
    for pair_index in packet.pair_indices:
        knot_delta = _shooting_delta(packet.shooting_knots, pair_index)
        texture = packet.textures[pair_index] if packet.textures is not None else None
        for frame_role in (0, 1):
            output[pair_index, frame_role] = _apply_feature_relay(
                output[pair_index, frame_role],
                packet.base_features[pair_index],
                texture,
                dash_frames[pair_index],
                knot_delta,
                frame_role=frame_role,
            )
    if output.dtype != np.uint8 or output.shape != base.shape:
        raise R10RelayError("R10 receiver failed its uint8 realized-pair contract")
    return output


def r10_control_mode(packet_bytes: bytes, mode: R10RelayMode) -> bytes:
    packet = parse_r10_packet(packet_bytes)
    if type(mode) is not R10RelayMode:
        raise R10RelayError("control mode must be typed")
    return serialize_r10_packet(replace(packet, mode=mode))


def r10_control_phase_time_permutation(
    packet_bytes: bytes,
    permutation: Sequence[int],
) -> bytes:
    packet = parse_r10_packet(packet_bytes)
    if packet.textures is None:
        raise R10RelayError("phase/time permutation requires TEXTURE")
    order = tuple(int(v) for v in permutation)
    if tuple(sorted(order)) != tuple(range(packet.n_pairs)):
        raise R10RelayError("phase/time permutation must be a population bijection")
    textures = tuple(packet.textures[index] for index in order)
    return serialize_r10_packet(replace(packet, textures=textures))


def r10_control_channel_energy_shuffle(
    packet_bytes: bytes,
    permutation: Sequence[int],
) -> bytes:
    packet = parse_r10_packet(packet_bytes)
    order = tuple(int(v) for v in permutation)
    if tuple(sorted(order)) != (0, 1, 2):
        raise R10RelayError("channel shuffle must be a permutation of (0,1,2)")
    records = []
    for row in packet.base_features:
        before = sum(value * value for value in row.channel_weights_q8)
        weights = tuple(row.channel_weights_q8[index] for index in order)
        after = sum(value * value for value in weights)
        if before != after:
            raise R10RelayError("channel shuffle failed exact integer energy preservation")
        records.append(replace(row, channel_weights_q8=weights))
    return serialize_r10_packet(replace(packet, base_features=tuple(records)))


def r10_control_delete_texture(packet_bytes: bytes) -> bytes:
    packet = parse_r10_packet(packet_bytes)
    knots = tuple(
        replace(
            row,
            amplitude_delta_q8=0,
            phase_delta_q10=0,
            texture_delta_q8=0,
        )
        for row in packet.shooting_knots
    )
    return serialize_r10_packet(
        replace(packet, textures=None, dash1_payload=None, shooting_knots=knots)
    )


def r10_control_delete_shooting_knots(packet_bytes: bytes) -> bytes:
    packet = parse_r10_packet(packet_bytes)
    return serialize_r10_packet(replace(packet, shooting_knots=()))


def r10_control_merge_shooting_knots(packet_bytes: bytes, left_index: int = 0) -> bytes:
    packet = parse_r10_packet(packet_bytes)
    if len(packet.shooting_knots) < 2:
        raise R10RelayError("knot merge requires at least two shooting knots")
    if not (0 <= left_index < len(packet.shooting_knots) - 1):
        raise R10RelayError("knot merge index is out of range")
    left = packet.shooting_knots[left_index]
    right = packet.shooting_knots[left_index + 1]
    merged_values = tuple(
        _signed_round_div(a + b, 2)
        for a, b in zip(left.deltas, right.deltas, strict=True)
    )
    merged = R10ShootingKnotV1(left.pair_index, *merged_values)
    knots = (
        *packet.shooting_knots[:left_index],
        merged,
        *packet.shooting_knots[left_index + 2 :],
    )
    return serialize_r10_packet(replace(packet, shooting_knots=knots))


def mutate_r10_constraint(packet_bytes: bytes, constraint: str) -> bytes:
    """Deterministically move one counted operand for receiver reachability tests."""

    packet = parse_r10_packet(packet_bytes)
    if constraint not in R10_CONSTRAINTS:
        raise R10RelayError(f"unknown R10 constraint {constraint!r}")
    if constraint in {"AMPLITUDE", "FREQUENCY", "PHASE", "TEXTURE"}:
        if packet.textures is None:
            raise R10RelayError(f"{constraint} mutation requires TEXTURE")
        rows = list(packet.textures)
        row = rows[0]
        if constraint == "AMPLITUDE":
            row = replace(row, amplitude_q8=max(-32768, min(32767, row.amplitude_q8 + 64)))
        elif constraint == "FREQUENCY":
            row = replace(row, frequency_q8=1 + (row.frequency_q8 % 65534))
        elif constraint == "PHASE":
            row = replace(row, phase_q10=(row.phase_q10 + 128) & 1023)
        else:
            row = replace(row, texture_gain_q8=max(-32768, min(32767, row.texture_gain_q8 + 64)))
        rows[0] = row
        return serialize_r10_packet(replace(packet, textures=tuple(rows)))
    if constraint in {"CONTRAST", "CHANNEL_ENERGY", "FROZEN_FEATURE"}:
        rows = list(packet.base_features)
        row = rows[0]
        if constraint == "CONTRAST":
            row = replace(row, contrast_q8=max(-32768, min(32767, row.contrast_q8 + 64)))
        elif constraint == "FROZEN_FEATURE":
            row = replace(row, feature_gain_q8=max(-32768, min(32767, row.feature_gain_q8 + 64)))
        else:
            weights = list(row.channel_weights_q8)
            weights[0] = max(-32768, min(32767, weights[0] + 64))
            row = replace(row, channel_weights_q8=tuple(weights))
        rows[0] = row
        return serialize_r10_packet(replace(packet, base_features=tuple(rows)))
    if constraint == "MULTIPLE_SHOOTING":
        if packet.shooting_knots:
            knots = list(packet.shooting_knots)
            first = knots[0]
            knots[0] = replace(
                first,
                luma_bias_delta_q8=max(-32768, min(32767, first.luma_bias_delta_q8 + 64)),
            )
        else:
            knots = [R10ShootingKnotV1(0, 64, 0, 0, 0, 0, 0)]
        return serialize_r10_packet(replace(packet, shooting_knots=tuple(knots)))
    raise AssertionError("unreachable constraint mutation")


@dataclass(frozen=True, slots=True)
class R10OperandSpanV1:
    constraint: str
    section: str
    packet_offset: int
    byte_length: int
    operand_sha256: str


@dataclass(frozen=True, slots=True)
class R10SelectedSolutionAdapterV1:
    schema: str
    packet_sha256: str
    packet_bytes: int
    source_sha256: str
    pair_population_sha256: str
    base_realization_sha256: str
    receiver_operation: str
    constraints: tuple[str, ...]
    operand_spans: tuple[R10OperandSpanV1, ...]
    frame_role_by_constraint: Mapping[str, str]
    semantic_role_by_constraint: Mapping[str, str]
    scientific_role_by_constraint: Mapping[str, str]
    physical_sections: tuple[R10PhysicalSpanV1, ...]
    public_inflate_receiver_abi: Mapping[str, Any]
    asymmetric_codec_placement: Mapping[str, Any]
    public_video_emission_abi: Mapping[str, Any]
    blockers: tuple[str, ...]


def r10_runtime_source_closure() -> dict[str, Any]:
    """Hash the explicit repo-local source closure needed by the public ABI.

    This is a conservative encoder-side closure manifest, not an assertion that
    an ``inflate.py`` bundle already contains it.  Root/G23 must compare these
    source identities to the actual shipped public receiver and retain the
    named blocker until that public archive path executes under upstream eval.
    """

    module_names = (
        "tac.witness_dsl.taskspace_r10_feature_texture_relay",
        "tac.boundary_math.dash_phase_carrier",
        "tac.boundary_math.phase_primitives",
        "tac.boundary_math.stratified_depth_warp",
        "tac.boundary_math.warp_real_luma_frame0",
        "tac.boundary_math.xi_pose_coder",
        "tac.boundary_math.xi_spline_residual_coder",
        "tac.lie._se3_numpy",
        "tac.lossless.range_coder",
    )
    rows: list[dict[str, Any]] = []
    for module_name in module_names:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise R10RelayError(f"public receiver dependency has no source: {module_name}")
        source_path = Path(spec.origin).resolve()
        source = source_path.read_bytes()
        rows.append(
            {
                "module": module_name,
                "source_path": str(source_path),
                "source_bytes": len(source),
                "source_sha256": _sha256(source),
            }
        )
    closure_payload = _canonical_json(
        [{key: row[key] for key in ("module", "source_bytes", "source_sha256")} for row in rows]
    )
    return {
        "schema": "taskspace_r10_public_receiver_source_closure.v1",
        "entrypoint": R10_RECEIVER_OPERATION,
        "signature": str(inspect.signature(decode_r10_packet)),
        "files": rows,
        "closure_sha256": _sha256(closure_payload),
        "inflate_bundle_executed": False,
        "blocker": R10_INFLATE_PUBLIC_RECEIVER_BUNDLE_CLOSURE_OWED,
    }


def build_r10_selected_solution_adapter(packet_bytes: bytes) -> R10SelectedSolutionAdapterV1:
    """Expose a no-second-schema descriptor for root/G23 placement integration."""

    packet, directory = _parse_r10_packet_with_directory(packet_bytes)
    by_section = {section: (offset, length) for section, offset, length in directory}
    spans: list[R10OperandSpanV1] = []

    def add(constraint: str, section: R10Section, rel_offset: int, length: int) -> None:
        if section not in by_section or length <= 0:
            return
        section_offset, section_length = by_section[section]
        if rel_offset < 0 or rel_offset + length > section_length:
            raise R10RelayError("adapter operand span escaped its exact physical section")
        absolute = section_offset + rel_offset
        operand = packet_bytes[absolute : absolute + length]
        spans.append(
            R10OperandSpanV1(
                constraint=constraint,
                section=section.name,
                packet_offset=absolute,
                byte_length=length,
                operand_sha256=_sha256(operand),
            )
        )

    for pair_index in range(packet.n_pairs):
        base_offset = pair_index * _BASE_FEATURE.size
        add("CONTRAST", R10Section.BASE_FEATURE, base_offset + 2, 2)
        add("FROZEN_FEATURE", R10Section.BASE_FEATURE, base_offset + 6, 2)
        add("CHANNEL_ENERGY", R10Section.BASE_FEATURE, base_offset + 8, 6)
        if packet.textures is not None:
            texture_offset = pair_index * _TEXTURE.size
            add("AMPLITUDE", R10Section.TEXTURE, texture_offset, 2)
            add("FREQUENCY", R10Section.TEXTURE, texture_offset + 2, 2)
            add("PHASE", R10Section.TEXTURE, texture_offset + 4, 2)
            add("TEXTURE", R10Section.TEXTURE, texture_offset + 6, 2)
    if R10Section.DASH1 in by_section:
        add("PHASE", R10Section.DASH1, 0, by_section[R10Section.DASH1][1])
        add("TEXTURE", R10Section.DASH1, 0, by_section[R10Section.DASH1][1])
    if R10Section.SHOOTING_KNOT in by_section:
        add(
            "MULTIPLE_SHOOTING",
            R10Section.SHOOTING_KNOT,
            0,
            by_section[R10Section.SHOOTING_KNOT][1],
        )
    add("MULTIPLE_SHOOTING", R10Section.XIP2, 0, by_section[R10Section.XIP2][1])
    common_frame_roles = {
        "AMPLITUDE": "CHRONOLOGICAL_PAIR",
        "FREQUENCY": "CHRONOLOGICAL_PAIR",
        "PHASE": "CHRONOLOGICAL_PAIR",
        "CONTRAST": "CHRONOLOGICAL_PAIR",
        "CHANNEL_ENERGY": "CHRONOLOGICAL_PAIR",
        "TEXTURE": "CHRONOLOGICAL_PAIR",
        "MULTIPLE_SHOOTING": "CHRONOLOGICAL_PAIR",
        "FROZEN_FEATURE": "CHRONOLOGICAL_PAIR",
    }
    semantic = {
        "AMPLITUDE": "FIBER",
        "FREQUENCY": "FIBER",
        "PHASE": "CONNECTION",
        "CONTRAST": "FIBER",
        "CHANNEL_ENERGY": "FIBER",
        "TEXTURE": "RESIDUAL",
        "MULTIPLE_SHOOTING": "CONNECTION",
        "FROZEN_FEATURE": "RESIDUAL",
    }
    scientific = {
        constraint: (
            "POSE_TRANSPORT_FRAME0"
            if constraint in {"PHASE", "MULTIPLE_SHOOTING"}
            else "CELL_VALUE_PREIMAGE"
        )
        for constraint in R10_CONSTRAINTS
    }
    counted_sections = tuple(span.section for span in packet_physical_spans(packet_bytes))
    placement = {
        "encoder_only_unbounded_compiler": (
            "source_frame_analysis",
            "inverse_solve",
            "operand_fit_and_selection",
            "scorer_queries",
            "costate_estimation",
            "joint_descent_final_linker",
        ),
        "generic_deterministic_decode_vm": (
            "strict_packet_parse",
            "xip2_decode_and_se3_homography",
            "dash1_chronology_decode",
            "counted_polygon_rasterization",
            "stratified_inverse_warp",
            "integer_luma_edge_feature_map",
            "integer_phase_texture_raster",
            "uint8_round_clip",
        ),
        "counted_video_statistic_sections": counted_sections,
        "counted_learned_irreducible_residual_sections": (),
        "learned_residual_policy": (
            "admit_only_after_maximum_inverse_solve_leaves_receiver_measured_debt"
        ),
    }
    video_emission = {
        "receiver_output": "uint8[P,2,H,W,3]",
        "canonical_pair_order": list(packet.pair_indices),
        "video_mux_or_file_emitter_present": False,
        "upstream_evaluate_replay_present": False,
        "blocker": R10_PUBLIC_VIDEO_EMISSION_AND_UPSTREAM_EVALUATE_REPLAY_OWED,
    }
    return R10SelectedSolutionAdapterV1(
        schema=R10_ADAPTER_SCHEMA,
        packet_sha256=_sha256(packet_bytes),
        packet_bytes=len(packet_bytes),
        source_sha256=packet.identity.source_sha256,
        pair_population_sha256=packet.identity.pair_population_sha256,
        base_realization_sha256=packet.identity.base_realization_sha256,
        receiver_operation=R10_RECEIVER_OPERATION,
        constraints=R10_CONSTRAINTS,
        operand_spans=tuple(spans),
        frame_role_by_constraint=common_frame_roles,
        semantic_role_by_constraint=semantic,
        scientific_role_by_constraint=scientific,
        physical_sections=packet_physical_spans(packet_bytes),
        public_inflate_receiver_abi=r10_runtime_source_closure(),
        asymmetric_codec_placement=placement,
        public_video_emission_abi=video_emission,
        blockers=R10_BLOCKERS,
    )


def _changed_output_stats(base: np.ndarray, output: np.ndarray) -> dict[str, int]:
    delta = np.abs(output.astype(np.int16) - base.astype(np.int16))
    return {
        "changed_values": int(np.count_nonzero(delta)),
        "changed_pixels": int(np.count_nonzero(np.any(delta != 0, axis=-1))),
        "l1_rgb_delta": int(delta.astype(np.int64).sum()),
        "max_abs_rgb_delta": int(delta.max(initial=0)),
    }


def build_r10_decode_receipt(
    packet_bytes: bytes,
    base_pairs: np.ndarray,
    *,
    expected_source_sha256: str,
    expected_pair_population_sha256: str,
    include_mutation_canaries: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Double replay plus controls/telemetry; no scorer or score authority."""

    packet = parse_r10_packet(packet_bytes)
    output_a = decode_r10_packet(
        packet_bytes,
        base_pairs,
        expected_source_sha256=expected_source_sha256,
        expected_pair_population_sha256=expected_pair_population_sha256,
    )
    output_b = decode_r10_packet(
        packet_bytes,
        base_pairs,
        expected_source_sha256=expected_source_sha256,
        expected_pair_population_sha256=expected_pair_population_sha256,
    )
    if not np.array_equal(output_a, output_b):
        raise R10RelayError("same packet/base failed deterministic double replay")
    controls: dict[str, Any] = {}

    def run_control(name: str, control_packet: bytes) -> None:
        result = decode_r10_packet(
            control_packet,
            base_pairs,
            expected_source_sha256=expected_source_sha256,
            expected_pair_population_sha256=expected_pair_population_sha256,
        )
        controls[name] = {
            "packet_sha256": _sha256(control_packet),
            "packet_bytes": len(control_packet),
            "output_sha256": realization_sha256(result),
            "changed_from_main": _changed_output_stats(output_a, result),
        }

    alternate_mode = (
        R10RelayMode.RELAY_ONLY
        if packet.mode is R10RelayMode.JOINT
        else R10RelayMode.JOINT
    )
    run_control("relay_only_vs_joint", r10_control_mode(packet_bytes, alternate_mode))
    run_control("channel_energy_preserving_shuffle", r10_control_channel_energy_shuffle(packet_bytes, (1, 2, 0)))
    if packet.textures is not None:
        permutation = tuple(reversed(range(packet.n_pairs)))
        if packet.n_pairs == 1:
            permutation = (0,)
        run_control("phase_time_permutation", r10_control_phase_time_permutation(packet_bytes, permutation))
        run_control("texture_deletion", r10_control_delete_texture(packet_bytes))
    if packet.shooting_knots:
        run_control("shooting_knot_deletion", r10_control_delete_shooting_knots(packet_bytes))
    if len(packet.shooting_knots) >= 2:
        run_control("shooting_knot_merge", r10_control_merge_shooting_knots(packet_bytes))
    canaries: dict[str, Any] = {}
    if include_mutation_canaries:
        for constraint in R10_CONSTRAINTS:
            if packet.textures is None and constraint in {"AMPLITUDE", "FREQUENCY", "PHASE", "TEXTURE"}:
                canaries[constraint] = {"status": "NOT_PRESENT", "output_changed": None}
                continue
            mutant = mutate_r10_constraint(packet_bytes, constraint)
            mutant_output = decode_r10_packet(
                mutant,
                base_pairs,
                expected_source_sha256=expected_source_sha256,
                expected_pair_population_sha256=expected_pair_population_sha256,
            )
            stats = _changed_output_stats(output_a, mutant_output)
            canaries[constraint] = {
                "status": "RECEIVER_REACHABLE" if stats["changed_values"] else "DEAD_OR_SATURATED_OPERAND",
                "output_changed": bool(stats["changed_values"]),
                "mutant_packet_sha256": _sha256(mutant),
                **stats,
            }
    spans = packet_physical_spans(packet_bytes)
    section_bytes = {span.section: span.byte_length for span in spans}
    allocator = [
        {
            "action": "DELETE_SECTION",
            "section": section,
            "exact_section_bytes": size,
            "score_unit_value_per_byte": None,
            "blocker": R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED,
        }
        for section, size in sorted(section_bytes.items(), key=lambda item: (-item[1], item[0]))
        if section not in {R10Section.PAIR_INDEX.name, R10Section.GEOMETRY.name}
    ]
    receipt = {
        "schema": R10_RECEIPT_SCHEMA,
        "authority": "bounded_real_input_receiver_contract_only",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pair_count": packet.n_pairs,
        "mode": packet.mode.name,
        "source_sha256": packet.identity.source_sha256,
        "pair_population_sha256": packet.identity.pair_population_sha256,
        "base_realization_sha256": packet.identity.base_realization_sha256,
        "packet_sha256": _sha256(packet_bytes),
        "packet_bytes": len(packet_bytes),
        "packet_only_nominal_rate_coordinate": (
            25.0 * len(packet_bytes) / R10_RATE_DENOMINATOR
        ),
        "complete_archive_rate_contribution": None,
        "complete_archive_rate_blocker": (
            R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED
        ),
        "output_sha256": realization_sha256(output_a),
        "deterministic_double_replay": True,
        "numpy_cpu_portable": True,
        "cross_host_bit_identity_proven": False,
        "asymmetric_codec_contract": {
            "encoder_compiler": "unbounded_encode_only_inverse_solve_and_selection",
            "decode_vm": "bounded_deterministic_cpu_numpy",
            "gpu_decode_available": False,
            "video_derived_operands_counted": True,
            "learned_irreducible_residual_present": False,
            "training_policy": "irreducible_residual_only_after_inverse_solve",
            "joint_descent_role": "final_exact_realized_score_and_zip_linker",
        },
        "runtime_memory_portability": {
            "bounded_contract_runtime_authority": None,
            "contest_30_min_proven": False,
            "contest_16gb_proven": False,
            "cpu_gpu_parity_proven": False,
            "blockers": [
                R10_INFLATE_RUNTIME_30MIN_AND_MEMORY_16GB_PROOF_OWED,
                R10_CPU_GPU_DECODE_PARITY_OWED,
                R10_CROSS_HOST_BIT_IDENTITY_PROOF_OWED,
            ],
        },
        "output_contract": {
            "dtype": str(output_a.dtype),
            "shape": [int(v) for v in output_a.shape],
            "actual_uint8_receiver_output": True,
        },
        "main_vs_base": _changed_output_stats(np.asarray(base_pairs), output_a),
        "physical_sections": [asdict(span) for span in spans],
        "controls": controls,
        "operand_mutation_canaries": canaries,
        "costates": {
            "lambda_bytes_vs_distortion": None,
            "reason": R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED,
        },
        "sensitivity_hook": {
            "receiver_pixel_secants_only": canaries,
            "scorer_sensitivity": None,
            "reason": R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED,
        },
        "pareto_bit_allocator_hook": allocator,
        "autopilot_hook": {
            "status": "BLOCKED",
            "reason": R10_G23_PHYSICAL_GROUP_AND_LIFECYCLE_INTEGRATION_OWED,
        },
        "maximum_inverse_solve_hook": {
            "status": "OPEN_FOR_ANALYTIC_INVERSE_SOLVE",
            "learned_state_policy": "counted_irreducible_residual_only",
            "reason": R10_MAXIMUM_INVERSE_SOLVE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED,
        },
        "joint_descent_linker_hook": {
            "status": "BLOCKED_UNTIL_COMPLETE_OBJECT",
            "role": "final_exact_realized_score_and_zip_linker_pass",
            "reason": R10_JOINT_DESCENT_EXACT_REALIZED_SCORE_ZIP_LINKER_OWED,
        },
        "public_video_emission_hook": {
            "receiver_uint8_pairs_emitted": True,
            "required_video_file_emitted": False,
            "upstream_evaluate_replayed": False,
            "reason": R10_PUBLIC_VIDEO_EMISSION_AND_UPSTREAM_EVALUATE_REPLAY_OWED,
        },
        "continual_learning_hook": {
            "status": "BLOCKED_NO_AUTHORITY_ROW",
            "posterior_update": None,
            "reason": R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED,
        },
        "strict_proxy_debt": list(R10_BLOCKERS),
        "pointer_moved": False,
    }
    return output_a, receipt


__all__ = [
    "R10_ADAPTER_SCHEMA",
    "R10_BLOCKERS",
    "R10_CONSTRAINTS",
    "R10_PACKET_SCHEMA",
    "R10_RECEIPT_SCHEMA",
    "R10BaseFeatureRecordV1",
    "R10IdentityV1",
    "R10OperandSpanV1",
    "R10PacketV1",
    "R10PhysicalSpanV1",
    "R10PullbackPolygonV1",
    "R10RelayError",
    "R10RelayMode",
    "R10Section",
    "R10SelectedSolutionAdapterV1",
    "R10ShootingKnotV1",
    "R10StratifiedFlowV1",
    "R10TextureRecordV1",
    "build_r10_decode_receipt",
    "build_r10_selected_solution_adapter",
    "decode_r10_packet",
    "mutate_r10_constraint",
    "packet_physical_spans",
    "pair_population_sha256",
    "parse_r10_packet",
    "r10_control_channel_energy_shuffle",
    "r10_control_delete_shooting_knots",
    "r10_control_delete_texture",
    "r10_control_merge_shooting_knots",
    "r10_control_mode",
    "r10_control_phase_time_permutation",
    "r10_runtime_source_closure",
    "realization_sha256",
    "serialize_r10_packet",
]
