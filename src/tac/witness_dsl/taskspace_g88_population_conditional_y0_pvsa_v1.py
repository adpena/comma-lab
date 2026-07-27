# SPDX-License-Identifier: MIT
"""Population-addressed conditional-Y0 successor for compact PVSA V1.

The existing compact PVSA receiver ends after semantic P and an optional G74
role-aware prepaint.  This module adds the next causal transition without
changing that sealed V1 wire:

``PVSA1 -> exact corrected uint8 Y1 -> counted Y0 | exact Y1 -> (Y0, Y1)``.

The successor member contains the exact compact-PVSA member once plus one
strict conditional operand.  The operand addresses the complete n600 domain
with a closed ``PASS_P0`` or ``XIP2_SE3_FRAME0_WARP`` population default and
sparse per-pair overrides.  This avoids repeating 600 identical XIP2 mode rows
while keeping COPY, role-local, and PASS exceptions available.

* ``COPY_CONDITIONAL_Y1`` copies exact corrected Y1 into Y0 and owns the whole
  Y0 frame.
* ``ROLE_TRANSLATE_RGB`` edge-clamp translates exact corrected Y1, applies one
  RGB delta per selected visible semantic role, and copies only the selected
  translated role supports into Y0.
* ``XIP2_SE3_FRAME0_WARP`` consumes one exact n600 XIP2 ``[pair,6]`` trajectory
  and the counted fp32 pitch, then executes the existing NumPy/EON frame-0
  homography reference against exact corrected Y1.

Y1 is never writable.  Y0 outside the declared support is checked byte-for-byte
against the incoming P0.  The decoder is generic and never accepts a scorer,
target labels, GT pose, or a dense learned frame payload.

The XIP2 transition composes the mechanisms in
``taskspace_counted_xip2_chronological_a3.py``,
``taskspace_chronological_a3_encoder.py``, and
``warp_real_luma_frame0.py`` under a new PVSA source foreign key.  It does not
translate or forge the older bounded PASS-G source binding.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Final, Literal

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    WarpRealLumaFrame0Error,
    warp_frame0_native_numpy,
)
from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    CarrierComposeReceiverV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    V15RoleAwareOverlayError,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    OuterArchiveEncoding,
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    TaskspaceOuterArchiveError,
    build_taskspace_outer_archive,
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    SE3XiTransportV2,
    TaskspacePredictorStateV2Error,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    MAX_STREAM_BATCH_PAIRS,
    CompactActuatorTypeV1,
    CompactPVSAError,
    CompactPVSAReceiverV1,
    ParsedCompactPVSAMemberV1,
    parse_compact_pvsa_member,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
PAIR_COUNT: Final = 600
ROLE_COUNT: Final = len(REALIZATION_PAINT_ORDER)
ALL_ROLE_BITS: Final = (1 << ROLE_COUNT) - 1
MAX_MEMBER_BYTES: Final = (1 << 32) - 1
MAX_CONDITIONAL_BYTES: Final = 1 << 20

OPERAND_MAGIC: Final = b"G88CY1\x00\x00"
OPERAND_VERSION: Final = 1
_OPERAND_HEADER: Final = struct.Struct(">8sBBHHH32s32sfI32sI32s")
_CONTROL_ROW: Final = struct.Struct(">HBBhh15h")
_CRC32: Final = struct.Struct(">I")
_POSITIVE_ZERO_F32: Final = b"\x00\x00\x00\x00"
_ABSENT_XIP2_SHA256: Final = "0" * 64

SUCCESSOR_MAGIC: Final = b"PVSC2\x00\x00\x00"
SUCCESSOR_VERSION: Final = 1
_SUCCESSOR_HEADER: Final = struct.Struct(">8sBII32s32s")

RECEIVER_ID: Final = "tac.g88.pvsa_population_conditional_y0_given_exact_y1.v1"
WIRE_POLICY_ID: Final = "EXACT_PVSA1_MEMBER_PLUS_ONE_N600_DEFAULT_SPARSE_OVERRIDE_CONDITIONAL_OPERAND_V1"
CAUSAL_TRANSITION_ID: Final = "PVSA1_TO_EXACT_CORRECTED_Y1_TO_COUNTED_Y0_GIVEN_Y1_V1"
ROLE_SUPPORT_POLICY_ID: Final = "VISIBLE_V15_ROLE_MASK_TRANSLATED_EDGE_CLAMP_CAMERA_SUPPORT_V1"
XIP2_POLICY_ID: Final = "COUNTED_XIP2_SE3_CAMERA_THEN_R_NUMPY_FP32_SAMPLE_RNE_U8_V1"
XIP2_SOURCE_POLICY_ID: Final = "EXACT_CORRECTED_PVSA_Y1_AS_WARP_SOURCE_KEYFRAME_V1"
XIP2_NUMERIC_REFERENCE_ID: Final = "NUMPY_FP32_SAMPLE_FP64_EON_RNE_U8_V1"
PASS_POLICY_ID: Final = "DEFAULT_PASS_OR_XIP2_WITH_SPARSE_TYPED_OVERRIDES_V1"
PUBLIC_RUNTIME_BLOCKER: Final = "G88_PUBLIC_INFLATE_RUNTIME_GRAPH_LINK_OWED"
POSE_AUTHORITY_BLOCKER: Final = "G88_EXACT_POSE_OR_UPSTREAM_EVAL_OF_SUCCESSOR_ARCHIVE_OWED"
FRESH_XIP2_CUSTODY_BLOCKER: Final = "G88_FRESH_V15_N600_XIP2_TRAJECTORY_CUSTODY_OWED"
OPEN_BLOCKERS: Final = (
    PUBLIC_RUNTIME_BLOCKER,
    POSE_AUTHORITY_BLOCKER,
    FRESH_XIP2_CUSTODY_BLOCKER,
)


class PopulationConditionalPVSAError(ValueError):
    """A counted wire, custody binding, or conditional execution failed."""


class ConditionalY0ModeV1(IntEnum):
    """Closed population-default and per-pair override mode universe."""

    PASS_P0 = 0
    COPY_CONDITIONAL_Y1 = 1
    ROLE_TRANSLATE_RGB = 2
    XIP2_SE3_FRAME0_WARP = 3


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PopulationConditionalPVSAError(f"{label} must be canonical lowercase SHA-256")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PopulationConditionalPVSAError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != dtype:
        raise PopulationConditionalPVSAError(f"array dtype must be exactly {dtype}")
    result = np.ascontiguousarray(raw).copy()
    result.setflags(write=False)
    return result


def _zero_role_deltas() -> tuple[tuple[int, int, int], ...]:
    return tuple((0, 0, 0) for _ in REALIZATION_PAINT_ORDER)


def _canonical_pitch(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise PopulationConditionalPVSAError("pitch must be one finite real scalar")
    try:
        packed = struct.pack(">f", float(value))
    except (OverflowError, struct.error) as exc:
        raise PopulationConditionalPVSAError("pitch escaped finite fp32 range") from exc
    pitch = struct.unpack(">f", packed)[0]
    if not math.isfinite(pitch):
        raise PopulationConditionalPVSAError("pitch escaped finite fp32 range")
    if pitch == 0.0 and packed != _POSITIVE_ZERO_F32:
        raise PopulationConditionalPVSAError("negative-zero pitch is a forbidden operand alias")
    return pitch


@dataclass(frozen=True, slots=True)
class ConditionalY0ControlV1:
    """One behavior-canonical sparse control row."""

    source_pair_id: int
    mode: ConditionalY0ModeV1
    role_bits: int = 0
    shift_y: int = 0
    shift_x: int = 0
    role_rgb_deltas: tuple[tuple[int, int, int], ...] = field(default_factory=_zero_role_deltas)

    def __post_init__(self) -> None:
        _exact_int(
            self.source_pair_id,
            "source_pair_id",
            minimum=0,
            maximum=PAIR_COUNT - 1,
        )
        if type(self.mode) is not ConditionalY0ModeV1:
            raise PopulationConditionalPVSAError("mode changed exact closed enum type")
        _exact_int(self.role_bits, "role_bits", minimum=0, maximum=ALL_ROLE_BITS)
        _exact_int(
            self.shift_y,
            "shift_y",
            minimum=-(CAMERA_HEIGHT - 1),
            maximum=CAMERA_HEIGHT - 1,
        )
        _exact_int(
            self.shift_x,
            "shift_x",
            minimum=-(CAMERA_WIDTH - 1),
            maximum=CAMERA_WIDTH - 1,
        )
        if type(self.role_rgb_deltas) is not tuple or len(self.role_rgb_deltas) != ROLE_COUNT:
            raise PopulationConditionalPVSAError("role_rgb_deltas must have one exact RGB tuple per V15 paint role")
        for role_index, delta in enumerate(self.role_rgb_deltas):
            if type(delta) is not tuple or len(delta) != CHANNELS:
                raise PopulationConditionalPVSAError(f"role_rgb_deltas[{role_index}] must be one exact RGB tuple")
            for channel, value in enumerate(delta):
                _exact_int(
                    value,
                    f"role_rgb_deltas[{role_index}][{channel}]",
                    minimum=-255,
                    maximum=255,
                )

        zero_deltas = _zero_role_deltas()
        if self.mode in (
            ConditionalY0ModeV1.PASS_P0,
            ConditionalY0ModeV1.COPY_CONDITIONAL_Y1,
            ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
        ):
            if self.role_bits != 0 or self.shift_y != 0 or self.shift_x != 0 or self.role_rgb_deltas != zero_deltas:
                raise PopulationConditionalPVSAError(f"{self.mode.name} has no role support, shift, or RGB parameters")
            return
        if self.mode is not ConditionalY0ModeV1.ROLE_TRANSLATE_RGB:
            raise PopulationConditionalPVSAError("control mode has no executable decoder")
        if self.role_bits == 0:
            raise PopulationConditionalPVSAError("ROLE_TRANSLATE_RGB requires at least one owned semantic role")
        for role_index, delta in enumerate(self.role_rgb_deltas):
            if not (self.role_bits & (1 << role_index)) and delta != (0, 0, 0):
                raise PopulationConditionalPVSAError("unselected roles must have zero RGB deltas")
        if self.shift_y == 0 and self.shift_x == 0 and self.role_rgb_deltas == zero_deltas:
            raise PopulationConditionalPVSAError(
                "ROLE_TRANSLATE_RGB zero action aliases role-bounded COPY and is noncanonical"
            )

    @classmethod
    def copy_conditional_y1(cls, source_pair_id: int) -> ConditionalY0ControlV1:
        return cls(
            source_pair_id=source_pair_id,
            mode=ConditionalY0ModeV1.COPY_CONDITIONAL_Y1,
        )


@dataclass(frozen=True, slots=True)
class PopulationConditionalOperandV1:
    """Exact n600 sparse operand bound to one PVSA member and semantic P."""

    base_pvsa_member_sha256: str
    semantic_p_sha256: str
    controls: tuple[ConditionalY0ControlV1, ...]
    default_mode: ConditionalY0ModeV1 = ConditionalY0ModeV1.PASS_P0
    xip2_payload: bytes = field(default=b"", repr=False)
    pitch: float = 0.0
    source_pair_start: Literal[0] = 0
    pair_count: Literal[600] = PAIR_COUNT
    transport: SE3XiTransportV2 | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_sha256(self.base_pvsa_member_sha256, "base_pvsa_member_sha256")
        _require_sha256(self.semantic_p_sha256, "semantic_p_sha256")
        if self.source_pair_start != 0 or self.pair_count != PAIR_COUNT:
            raise PopulationConditionalPVSAError("G88 V1 operand must address the complete source pair range [0,600)")
        if (
            type(self.controls) is not tuple
            or len(self.controls) > PAIR_COUNT
            or any(type(row) is not ConditionalY0ControlV1 for row in self.controls)
        ):
            raise PopulationConditionalPVSAError("controls must be a bounded exact ConditionalY0ControlV1 tuple")
        pair_ids = tuple(row.source_pair_id for row in self.controls)
        if pair_ids != tuple(sorted(set(pair_ids))):
            raise PopulationConditionalPVSAError("sparse control rows must have unique ascending source pair IDs")
        if type(self.default_mode) is not ConditionalY0ModeV1 or self.default_mode not in (
            ConditionalY0ModeV1.PASS_P0,
            ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
        ):
            raise PopulationConditionalPVSAError("population default must be exact PASS_P0 or XIP2_SE3_FRAME0_WARP")
        if any(row.mode is self.default_mode for row in self.controls):
            raise PopulationConditionalPVSAError("sparse override cannot alias the population default mode")
        if type(self.xip2_payload) is not bytes:
            raise PopulationConditionalPVSAError("operand XIP2 payload must be exact bytes")
        pitch = _canonical_pitch(self.pitch)
        uses_xip2 = self.default_mode is ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP or any(
            row.mode is ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP for row in self.controls
        )
        transport: SE3XiTransportV2 | None = None
        if uses_xip2:
            if not self.xip2_payload.startswith(b"XIP2"):
                raise PopulationConditionalPVSAError("XIP2 mode requires one exact counted XIP2 payload")
            _exact_int(
                len(self.xip2_payload),
                "xip2_payload bytes",
                minimum=1,
                maximum=MAX_CONDITIONAL_BYTES,
            )
            try:
                transport = SE3XiTransportV2(
                    counted_payload=self.xip2_payload,
                    source_pair_ids=tuple(range(PAIR_COUNT)),
                    predictor_program_sha256=self.base_pvsa_member_sha256,
                )
            except TaskspacePredictorStateV2Error as exc:
                raise PopulationConditionalPVSAError("operand nested XIP2 failed exact EOF/n600 shape parse") from exc
        elif self.xip2_payload or struct.pack(">f", pitch) != _POSITIVE_ZERO_F32:
            raise PopulationConditionalPVSAError("non-XIP2 controls require canonical +0 pitch and empty XIP2 payload")
        object.__setattr__(self, "pitch", pitch)
        object.__setattr__(self, "transport", transport)

    @property
    def active_pair_count(self) -> int:
        if self.default_mode is ConditionalY0ModeV1.PASS_P0:
            return len(self.controls)
        return PAIR_COUNT - sum(row.mode is ConditionalY0ModeV1.PASS_P0 for row in self.controls)

    @property
    def pass_pair_count(self) -> int:
        return self.pair_count - self.active_pair_count

    def control_for_pair(self, source_pair_id: int) -> ConditionalY0ControlV1 | None:
        _exact_int(
            source_pair_id,
            "source_pair_id",
            minimum=self.source_pair_start,
            maximum=self.source_pair_start + self.pair_count - 1,
        )
        for row in self.controls:
            if row.source_pair_id == source_pair_id:
                return row
            if row.source_pair_id > source_pair_id:
                break
        return None

    def mode_for_pair(self, source_pair_id: int) -> ConditionalY0ModeV1:
        override = self.control_for_pair(source_pair_id)
        return self.default_mode if override is None else override.mode

    def to_bytes(self) -> bytes:
        body = b"".join(_encode_control(row) for row in self.controls)
        xip2_sha = _sha256(self.xip2_payload) if self.xip2_payload else _ABSENT_XIP2_SHA256
        header = _OPERAND_HEADER.pack(
            OPERAND_MAGIC,
            OPERAND_VERSION,
            int(self.default_mode),
            self.source_pair_start,
            self.pair_count,
            len(self.controls),
            bytes.fromhex(self.base_pvsa_member_sha256),
            bytes.fromhex(self.semantic_p_sha256),
            self.pitch,
            len(body),
            bytes.fromhex(_sha256(body)),
            len(self.xip2_payload),
            bytes.fromhex(xip2_sha),
        )
        prefix = header + body + self.xip2_payload
        if len(prefix) + _CRC32.size > MAX_CONDITIONAL_BYTES:
            raise PopulationConditionalPVSAError("conditional operand exceeds its bounded V1 byte ABI")
        return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())


def _encode_control(control: ConditionalY0ControlV1) -> bytes:
    flattened = tuple(channel for row in control.role_rgb_deltas for channel in row)
    return _CONTROL_ROW.pack(
        control.source_pair_id,
        int(control.mode),
        control.role_bits,
        control.shift_y,
        control.shift_x,
        *flattened,
    )


def parse_population_conditional_operand(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    maximum_operand_bytes: int = MAX_CONDITIONAL_BYTES,
) -> PopulationConditionalOperandV1:
    """Strictly parse, hash-bind, CRC-check, and re-emit one operand."""

    if type(payload) is not bytes:
        raise PopulationConditionalPVSAError("conditional operand must be exact bytes")
    limit = _exact_int(
        maximum_operand_bytes,
        "maximum_operand_bytes",
        minimum=_OPERAND_HEADER.size + _CRC32.size,
        maximum=MAX_MEMBER_BYTES,
    )
    minimum = _OPERAND_HEADER.size + _CRC32.size
    if not minimum <= len(payload) <= limit:
        raise PopulationConditionalPVSAError("conditional operand is truncated or exceeds caller ceiling")
    (
        magic,
        version,
        default_mode_wire,
        source_pair_start,
        pair_count,
        row_count,
        base_sha,
        semantic_sha,
        pitch,
        body_bytes,
        body_sha,
        xip2_bytes,
        xip2_sha,
    ) = _OPERAND_HEADER.unpack_from(payload)
    if magic != OPERAND_MAGIC or version != OPERAND_VERSION:
        raise PopulationConditionalPVSAError("conditional operand magic/version mismatch")
    try:
        default_mode = ConditionalY0ModeV1(default_mode_wire)
    except ValueError as exc:
        raise PopulationConditionalPVSAError("conditional operand population default mode is unknown") from exc
    if source_pair_start != 0 or pair_count != PAIR_COUNT:
        raise PopulationConditionalPVSAError("conditional operand escaped the complete n600 address space")
    if row_count > PAIR_COUNT or body_bytes != row_count * _CONTROL_ROW.size:
        raise PopulationConditionalPVSAError("conditional operand row count/body length mismatch")
    expected_bytes = _OPERAND_HEADER.size + body_bytes + xip2_bytes + _CRC32.size
    if len(payload) != expected_bytes:
        raise PopulationConditionalPVSAError("conditional operand body length/EOF mismatch")
    prefix = payload[: -_CRC32.size]
    (expected_crc,) = _CRC32.unpack_from(payload, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != expected_crc:
        raise PopulationConditionalPVSAError("conditional operand CRC32 mismatch")
    body_start = _OPERAND_HEADER.size
    xip2_start = body_start + body_bytes
    body = payload[body_start:xip2_start]
    xip2_payload = payload[xip2_start : -_CRC32.size]
    if _sha256(body) != body_sha.hex():
        raise PopulationConditionalPVSAError("conditional operand body SHA-256 mismatch")
    if xip2_payload:
        if _sha256(xip2_payload) != xip2_sha.hex():
            raise PopulationConditionalPVSAError("conditional operand XIP2 SHA-256 mismatch")
    elif xip2_sha.hex() != _ABSENT_XIP2_SHA256:
        raise PopulationConditionalPVSAError("conditional operand empty XIP2 section has noncanonical digest")
    if expected_sha256 is not None and _sha256(payload) != _require_sha256(
        expected_sha256,
        "expected_sha256",
    ):
        raise PopulationConditionalPVSAError("conditional operand SHA-256 mismatch")

    controls: list[ConditionalY0ControlV1] = []
    for index in range(row_count):
        values = _CONTROL_ROW.unpack_from(body, index * _CONTROL_ROW.size)
        source_pair_id, mode_wire, role_bits, shift_y, shift_x, *flat = values
        try:
            mode = ConditionalY0ModeV1(mode_wire)
        except ValueError as exc:
            raise PopulationConditionalPVSAError("conditional operand mode is unknown") from exc
        deltas = tuple(
            tuple(flat[role_index * CHANNELS : (role_index + 1) * CHANNELS]) for role_index in range(ROLE_COUNT)
        )
        controls.append(
            ConditionalY0ControlV1(
                source_pair_id=source_pair_id,
                mode=mode,
                role_bits=role_bits,
                shift_y=shift_y,
                shift_x=shift_x,
                role_rgb_deltas=deltas,
            )
        )
    result = PopulationConditionalOperandV1(
        base_pvsa_member_sha256=base_sha.hex(),
        semantic_p_sha256=semantic_sha.hex(),
        controls=tuple(controls),
        default_mode=default_mode,
        xip2_payload=xip2_payload,
        pitch=pitch,
    )
    if result.to_bytes() != payload:
        raise PopulationConditionalPVSAError("conditional operand changed on strict parse/re-encode")
    return result


@dataclass(frozen=True, slots=True)
class ParsedPopulationConditionalPVSAMemberV1:
    """Exact successor member and both typed decoder operands."""

    member_bytes: bytes = field(repr=False)
    base_pvsa_member_bytes: bytes = field(repr=False)
    base_pvsa: ParsedCompactPVSAMemberV1
    conditional_operand_bytes: bytes = field(repr=False)
    conditional_operand: PopulationConditionalOperandV1
    receiver_id: Literal["tac.g88.pvsa_population_conditional_y0_given_exact_y1.v1"] = RECEIVER_ID
    wire_policy_id: Literal["EXACT_PVSA1_MEMBER_PLUS_ONE_N600_DEFAULT_SPARSE_OVERRIDE_CONDITIONAL_OPERAND_V1"] = (
        WIRE_POLICY_ID
    )
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.member_bytes) is not bytes
            or not self.member_bytes
            or type(self.base_pvsa_member_bytes) is not bytes
            or not self.base_pvsa_member_bytes
            or type(self.base_pvsa) is not ParsedCompactPVSAMemberV1
            or type(self.conditional_operand_bytes) is not bytes
            or not self.conditional_operand_bytes
            or type(self.conditional_operand) is not PopulationConditionalOperandV1
        ):
            raise PopulationConditionalPVSAError("parsed successor lost exact typed byte custody")
        if (
            _sha256(self.base_pvsa_member_bytes) != self.conditional_operand.base_pvsa_member_sha256
            or self.base_pvsa.member_bytes != self.base_pvsa_member_bytes
            or self.base_pvsa.semantic_p_sha256 != self.conditional_operand.semantic_p_sha256
            or self.conditional_operand.to_bytes() != self.conditional_operand_bytes
        ):
            raise PopulationConditionalPVSAError("successor operand foreign keys differ from exact base PVSA")
        if (
            self.receiver_id != RECEIVER_ID
            or self.wire_policy_id != WIRE_POLICY_ID
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise PopulationConditionalPVSAError("successor research-only truth labels became permissive")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def open_blockers(self) -> tuple[str, ...]:
        return OPEN_BLOCKERS

    def open_receiver(
        self,
        *,
        verify_member_effects: bool = True,
    ) -> PopulationConditionalPVSAReceiverV1:
        return PopulationConditionalPVSAReceiverV1.open(
            self,
            verify_member_effects=verify_member_effects,
        )


def encode_population_conditional_pvsa_member(
    *,
    base_pvsa_member_bytes: bytes,
    conditional_operand_bytes: bytes,
) -> bytes:
    """Encode one exact PVSA member plus one exact conditional operand."""

    if (
        type(base_pvsa_member_bytes) is not bytes
        or not base_pvsa_member_bytes
        or type(conditional_operand_bytes) is not bytes
        or not conditional_operand_bytes
    ):
        raise PopulationConditionalPVSAError("successor encode requires two nonempty exact byte operands")
    for payload, label in (
        (base_pvsa_member_bytes, "base_pvsa_member_bytes"),
        (conditional_operand_bytes, "conditional_operand_bytes"),
    ):
        _exact_int(len(payload), label, minimum=1, maximum=MAX_MEMBER_BYTES)
    header = _SUCCESSOR_HEADER.pack(
        SUCCESSOR_MAGIC,
        SUCCESSOR_VERSION,
        len(base_pvsa_member_bytes),
        len(conditional_operand_bytes),
        bytes.fromhex(_sha256(base_pvsa_member_bytes)),
        bytes.fromhex(_sha256(conditional_operand_bytes)),
    )
    prefix = header + base_pvsa_member_bytes + conditional_operand_bytes
    return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


def parse_population_conditional_pvsa_member(
    member_bytes: bytes,
    *,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
    maximum_conditional_bytes: int = MAX_CONDITIONAL_BYTES,
) -> ParsedPopulationConditionalPVSAMemberV1:
    """Strictly parse both byte homes and re-encode the successor member."""

    if type(member_bytes) is not bytes:
        raise PopulationConditionalPVSAError("successor member must be exact bytes")
    conditional_limit = _exact_int(
        maximum_conditional_bytes,
        "maximum_conditional_bytes",
        minimum=_OPERAND_HEADER.size + _CRC32.size,
        maximum=MAX_MEMBER_BYTES,
    )
    member_limit = _exact_int(
        maximum_member_bytes,
        "maximum_member_bytes",
        minimum=_SUCCESSOR_HEADER.size + _CRC32.size + 2,
        maximum=MAX_MEMBER_BYTES,
    )
    minimum = _SUCCESSOR_HEADER.size + _CRC32.size + 2
    if not minimum <= len(member_bytes) <= member_limit:
        raise PopulationConditionalPVSAError("successor member is truncated or exceeds caller ceiling")
    (
        magic,
        version,
        base_bytes,
        conditional_bytes,
        base_sha,
        conditional_sha,
    ) = _SUCCESSOR_HEADER.unpack_from(member_bytes)
    if magic != SUCCESSOR_MAGIC or version != SUCCESSOR_VERSION:
        raise PopulationConditionalPVSAError("successor member magic/version mismatch")
    if (
        base_bytes < 1
        or conditional_bytes < 1
        or conditional_bytes > conditional_limit
        or len(member_bytes) != _SUCCESSOR_HEADER.size + base_bytes + conditional_bytes + _CRC32.size
    ):
        raise PopulationConditionalPVSAError("successor section length/EOF mismatch")
    prefix = member_bytes[: -_CRC32.size]
    (expected_crc,) = _CRC32.unpack_from(member_bytes, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != expected_crc:
        raise PopulationConditionalPVSAError("successor member CRC32 mismatch")
    base_start = _SUCCESSOR_HEADER.size
    conditional_start = base_start + base_bytes
    base_payload = member_bytes[base_start:conditional_start]
    conditional_payload = member_bytes[conditional_start : -_CRC32.size]
    if _sha256(base_payload) != base_sha.hex():
        raise PopulationConditionalPVSAError("successor base PVSA SHA-256 mismatch")
    if _sha256(conditional_payload) != conditional_sha.hex():
        raise PopulationConditionalPVSAError("successor conditional operand SHA-256 mismatch")
    try:
        base = parse_compact_pvsa_member(
            base_payload,
            maximum_member_bytes=maximum_member_bytes,
            maximum_section_bytes=maximum_section_bytes,
        )
    except CompactPVSAError as exc:
        raise PopulationConditionalPVSAError("successor embedded base PVSA strict parse failed") from exc
    conditional = parse_population_conditional_operand(
        conditional_payload,
        expected_sha256=conditional_sha.hex(),
        maximum_operand_bytes=conditional_limit,
    )
    if (
        conditional.base_pvsa_member_sha256 != base.member_sha256
        or conditional.semantic_p_sha256 != base.semantic_p_sha256
    ):
        raise PopulationConditionalPVSAError("conditional operand is bound to a different base PVSA or semantic P")
    if (
        encode_population_conditional_pvsa_member(
            base_pvsa_member_bytes=base_payload,
            conditional_operand_bytes=conditional_payload,
        )
        != member_bytes
    ):
        raise PopulationConditionalPVSAError("successor member changed on strict parse/re-encode")
    return ParsedPopulationConditionalPVSAMemberV1(
        member_bytes=member_bytes,
        base_pvsa_member_bytes=base_payload,
        base_pvsa=base,
        conditional_operand_bytes=conditional_payload,
        conditional_operand=conditional,
    )


@dataclass(frozen=True, slots=True)
class ConditionalBatchResultV1:
    """One exact bounded chronological transition and its ownership proof."""

    camera_pairs: np.ndarray
    base_camera_pairs: np.ndarray
    owned_y0_values: np.ndarray
    local_pair_ids: tuple[int, ...]
    active_pair_ids: tuple[int, ...]
    camera_sha256: str
    base_camera_sha256: str
    exact_y1_sha256: str
    owned_y0_sha256: str
    changed_y0_values: int
    changed_y0_pixels: int
    preserved_unowned_y0_values: int
    deterministic_double_decode: bool
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        count = len(self.local_pair_ids)
        expected_camera = (count, 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        expected_owned = (count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        for value, dtype, shape, label in (
            (
                self.camera_pairs,
                np.dtype(np.uint8),
                expected_camera,
                "camera_pairs",
            ),
            (
                self.base_camera_pairs,
                np.dtype(np.uint8),
                expected_camera,
                "base_camera_pairs",
            ),
            (
                self.owned_y0_values,
                np.dtype(bool),
                expected_owned,
                "owned_y0_values",
            ),
        ):
            raw = np.asarray(value)
            if raw.dtype != dtype or raw.shape != shape:
                raise PopulationConditionalPVSAError(f"{label} changed exact dtype/shape ABI")
        for label in (
            "camera_sha256",
            "base_camera_sha256",
            "exact_y1_sha256",
            "owned_y0_sha256",
        ):
            _require_sha256(getattr(self, label), label)
        if (
            _array_sha256(self.camera_pairs) != self.camera_sha256
            or _array_sha256(self.base_camera_pairs) != self.base_camera_sha256
            or _array_sha256(self.camera_pairs[:, 1]) != self.exact_y1_sha256
            or _array_sha256(self.owned_y0_values) != self.owned_y0_sha256
        ):
            raise PopulationConditionalPVSAError("conditional batch arrays differ from exact receipt hashes")
        if not np.array_equal(self.camera_pairs[:, 1], self.base_camera_pairs[:, 1]):
            raise PopulationConditionalPVSAError("conditional transition changed exact corrected Y1")
        if not np.array_equal(
            self.camera_pairs[:, 0][~self.owned_y0_values],
            self.base_camera_pairs[:, 0][~self.owned_y0_values],
        ):
            raise PopulationConditionalPVSAError("conditional transition changed P0 outside owned support")
        changed = self.camera_pairs[:, 0] != self.base_camera_pairs[:, 0]
        if (
            self.changed_y0_values != int(np.count_nonzero(changed))
            or self.changed_y0_pixels != int(np.count_nonzero(np.any(changed, axis=3)))
            or self.preserved_unowned_y0_values != int(np.count_nonzero(~self.owned_y0_values))
            or type(self.deterministic_double_decode) is not bool
            or self.research_only is not True
            or self.score_claim is not False
        ):
            raise PopulationConditionalPVSAError("conditional batch receipt counts or truth labels differ")
        object.__setattr__(
            self,
            "camera_pairs",
            _immutable(self.camera_pairs, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "base_camera_pairs",
            _immutable(self.base_camera_pairs, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "owned_y0_values",
            _immutable(self.owned_y0_values, dtype=np.dtype(bool)),
        )


def _validate_batch_ids(local_pair_ids: tuple[int, ...]) -> None:
    if (
        type(local_pair_ids) is not tuple
        or not 1 <= len(local_pair_ids) <= MAX_STREAM_BATCH_PAIRS
        or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in local_pair_ids)
        or local_pair_ids != tuple(range(local_pair_ids[0], local_pair_ids[0] + len(local_pair_ids)))
    ):
        raise PopulationConditionalPVSAError("conditional stream batch must be 1..16 contiguous exact n600 pair IDs")


def _translated(value: np.ndarray, *, shift_y: int, shift_x: int) -> np.ndarray:
    source_y = np.clip(
        np.arange(value.shape[0], dtype=np.int64) - shift_y,
        0,
        value.shape[0] - 1,
    )
    source_x = np.clip(
        np.arange(value.shape[1], dtype=np.int64) - shift_x,
        0,
        value.shape[1] - 1,
    )
    return np.ascontiguousarray(value[source_y[:, None], source_x[None, :]])


def _apply_once(
    *,
    operand: PopulationConditionalOperandV1,
    base_camera_pairs: np.ndarray,
    local_pair_ids: tuple[int, ...],
    visible_role_masks: np.ndarray | None,
) -> ConditionalBatchResultV1:
    _validate_batch_ids(local_pair_ids)
    base = np.asarray(base_camera_pairs)
    expected = (len(local_pair_ids), 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
    if base.dtype != np.uint8 or base.shape != expected:
        raise PopulationConditionalPVSAError("base PVSA batch changed exact chronological camera ABI")
    if visible_role_masks is not None:
        masks = np.asarray(visible_role_masks)
        expected_masks = (
            len(local_pair_ids),
            ROLE_COUNT,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
        )
        if masks.dtype != np.dtype(bool) or masks.shape != expected_masks:
            raise PopulationConditionalPVSAError("visible role masks changed exact bool camera ABI")
    else:
        masks = None

    output = np.ascontiguousarray(base).copy()
    owned = np.zeros(
        (len(local_pair_ids), CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS),
        dtype=bool,
    )
    geometry: GroundHomographyGeom | None = None
    if operand.transport is not None:
        try:
            geometry = GroundHomographyGeom.eon(
                (CAMERA_HEIGHT, CAMERA_WIDTH),
                pitch=operand.pitch,
            )
        except np.linalg.LinAlgError as exc:
            raise PopulationConditionalPVSAError("XIP2 EON geometry construction failed") from exc
    active: list[int] = []
    for local_index, pair_id in enumerate(local_pair_ids):
        control = operand.control_for_pair(pair_id)
        mode = operand.default_mode if control is None else control.mode
        if mode is ConditionalY0ModeV1.PASS_P0:
            continue
        active.append(pair_id)
        p0 = base[local_index, 0]
        exact_y1 = base[local_index, 1]
        if mode is ConditionalY0ModeV1.COPY_CONDITIONAL_Y1:
            output[local_index, 0] = exact_y1
            owned[local_index] = True
        elif mode is ConditionalY0ModeV1.ROLE_TRANSLATE_RGB:
            if control is None:  # pragma: no cover - default mode is closed
                raise PopulationConditionalPVSAError("role mode lost its sparse parameter row")
            if masks is None:
                raise PopulationConditionalPVSAError("ROLE_TRANSLATE_RGB requires receiver-derived visible role masks")
            translated_y1 = _translated(
                exact_y1,
                shift_y=control.shift_y,
                shift_x=control.shift_x,
            )
            candidate = translated_y1.astype(np.int32)
            owned_pixels = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=bool)
            for role_index, delta in enumerate(control.role_rgb_deltas):
                if not control.role_bits & (1 << role_index):
                    continue
                role_mask = _translated(
                    masks[local_index, role_index],
                    shift_y=control.shift_y,
                    shift_x=control.shift_x,
                )
                owned_pixels |= role_mask
                candidate[role_mask] += np.asarray(delta, dtype=np.int32)
            if not np.any(owned_pixels):
                raise PopulationConditionalPVSAError("ROLE_TRANSLATE_RGB selected zero visible camera support")
            candidate_u8 = np.clip(candidate, 0, 255).astype(np.uint8)
            output[local_index, 0, owned_pixels] = candidate_u8[owned_pixels]
            owned[local_index, owned_pixels] = True
        elif mode is ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP:
            if operand.transport is None or geometry is None:
                raise PopulationConditionalPVSAError("XIP2 control lost its exact counted transport")
            try:
                warped = warp_frame0_native_numpy(
                    exact_y1,
                    operand.transport.xi[pair_id],
                    geometry,
                    compute_dtype=np.float32,
                )
            except (WarpRealLumaFrame0Error, np.linalg.LinAlgError) as exc:
                raise PopulationConditionalPVSAError("XIP2 camera-then-R frame0 warp failed") from exc
            output[local_index, 0] = np.clip(
                np.round(warped),
                0.0,
                255.0,
            ).astype(np.uint8)
            owned[local_index] = True
        else:  # pragma: no cover - dataclass/parser close the enum
            raise PopulationConditionalPVSAError("conditional control reached an unsupported execution mode")
        if np.array_equal(output[local_index, 0], p0):
            raise PopulationConditionalPVSAError("active conditional control was camera-inert on its source pair")

    if not np.array_equal(output[:, 1], base[:, 1]):
        raise PopulationConditionalPVSAError("conditional transition changed exact corrected Y1")
    if not np.array_equal(output[:, 0][~owned], base[:, 0][~owned]):
        raise PopulationConditionalPVSAError("conditional transition changed P0 outside owned support")
    changed = output[:, 0] != base[:, 0]
    return ConditionalBatchResultV1(
        camera_pairs=output,
        base_camera_pairs=base,
        owned_y0_values=owned,
        local_pair_ids=local_pair_ids,
        active_pair_ids=tuple(active),
        camera_sha256=_array_sha256(output),
        base_camera_sha256=_array_sha256(base),
        exact_y1_sha256=_array_sha256(output[:, 1]),
        owned_y0_sha256=_array_sha256(owned),
        changed_y0_values=int(np.count_nonzero(changed)),
        changed_y0_pixels=int(np.count_nonzero(np.any(changed, axis=3))),
        preserved_unowned_y0_values=int(np.count_nonzero(~owned)),
        deterministic_double_decode=False,
    )


def apply_population_conditional_to_decoded_batch(
    *,
    operand: PopulationConditionalOperandV1,
    first_base_camera_pairs: np.ndarray,
    second_base_camera_pairs: np.ndarray,
    local_pair_ids: tuple[int, ...],
    first_visible_role_masks: np.ndarray | None = None,
    second_visible_role_masks: np.ndarray | None = None,
) -> ConditionalBatchResultV1:
    """Apply twice to two exact base decodes and return only equal results.

    This function is also the low-memory seam for a separately materialized
    public PVSA double decode.  It does not weaken custody: the two base batches
    must be byte-identical before the counted transition is admitted.
    """

    if not np.array_equal(first_base_camera_pairs, second_base_camera_pairs):
        raise PopulationConditionalPVSAError("base PVSA deterministic double decode differs")
    if (first_visible_role_masks is None) != (second_visible_role_masks is None):
        raise PopulationConditionalPVSAError("double decode supplied only one visible-role mask tensor")
    if first_visible_role_masks is not None and not np.array_equal(first_visible_role_masks, second_visible_role_masks):
        raise PopulationConditionalPVSAError("receiver-derived visible role masks differ across double decode")
    first = _apply_once(
        operand=operand,
        base_camera_pairs=first_base_camera_pairs,
        local_pair_ids=local_pair_ids,
        visible_role_masks=first_visible_role_masks,
    )
    second = _apply_once(
        operand=operand,
        base_camera_pairs=second_base_camera_pairs,
        local_pair_ids=local_pair_ids,
        visible_role_masks=second_visible_role_masks,
    )
    if (
        first.active_pair_ids != second.active_pair_ids
        or not np.array_equal(first.camera_pairs, second.camera_pairs)
        or not np.array_equal(first.owned_y0_values, second.owned_y0_values)
    ):
        raise PopulationConditionalPVSAError("conditional transition deterministic double decode differs")
    return replace(first, deterministic_double_decode=True)


@dataclass(frozen=True, slots=True, init=False)
class PopulationConditionalPVSAReceiverV1:
    """Cached PVSA receiver plus the population conditional transition."""

    parsed: ParsedPopulationConditionalPVSAMemberV1
    base_receiver: CompactPVSAReceiverV1
    role_receiver: CarrierComposeReceiverV1
    _parsed_identity: int
    _base_receiver_identity: int
    _role_receiver_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("PopulationConditionalPVSAReceiverV1 must be constructed through .open()")

    @classmethod
    def open(
        cls,
        parsed: ParsedPopulationConditionalPVSAMemberV1,
        *,
        verify_member_effects: bool = True,
    ) -> PopulationConditionalPVSAReceiverV1:
        if type(parsed) is not ParsedPopulationConditionalPVSAMemberV1:
            raise PopulationConditionalPVSAError("conditional receiver requires an exact parsed successor member")
        try:
            base_receiver = parsed.base_pvsa.open_receiver(verify_member_effects=verify_member_effects)
        except CompactPVSAError as exc:
            raise PopulationConditionalPVSAError("conditional receiver failed to open exact base PVSA") from exc
        role_receiver = _role_receiver_for_exact_y1(base_receiver)
        instance = object.__new__(cls)
        object.__setattr__(instance, "parsed", parsed)
        object.__setattr__(instance, "base_receiver", base_receiver)
        object.__setattr__(instance, "role_receiver", role_receiver)
        object.__setattr__(instance, "_parsed_identity", id(parsed))
        object.__setattr__(
            instance,
            "_base_receiver_identity",
            id(base_receiver),
        )
        object.__setattr__(instance, "_role_receiver_identity", id(role_receiver))
        instance._validate_custody()
        return instance

    def _validate_custody(self) -> None:
        if (
            type(self.parsed) is not ParsedPopulationConditionalPVSAMemberV1
            or type(self.base_receiver) is not CompactPVSAReceiverV1
            or type(self.role_receiver) is not CarrierComposeReceiverV1
            or id(self.parsed) != self._parsed_identity
            or id(self.base_receiver) != self._base_receiver_identity
            or id(self.role_receiver) != self._role_receiver_identity
            or self.base_receiver.parsed is not self.parsed.base_pvsa
        ):
            raise PopulationConditionalPVSAError("conditional cached receiver custody drifted")

    def render_camera_pair_batch(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> ConditionalBatchResultV1:
        """Decode and condition one exact bounded chronological batch twice."""

        self._validate_custody()
        _validate_batch_ids(local_pair_ids)
        first_base = self.base_receiver.render_camera_pair_batch(local_pair_ids)
        second_base = self.base_receiver.render_camera_pair_batch(local_pair_ids)
        needs_masks = any(
            (row.source_pair_id in set(local_pair_ids) and row.mode is ConditionalY0ModeV1.ROLE_TRANSLATE_RGB)
            for row in self.parsed.conditional_operand.controls
        )
        first_masks = _visible_role_camera_masks(self.role_receiver, local_pair_ids) if needs_masks else None
        second_masks = _visible_role_camera_masks(self.role_receiver, local_pair_ids) if needs_masks else None
        return apply_population_conditional_to_decoded_batch(
            operand=self.parsed.conditional_operand,
            first_base_camera_pairs=first_base,
            second_base_camera_pairs=second_base,
            local_pair_ids=local_pair_ids,
            first_visible_role_masks=first_masks,
            second_visible_role_masks=second_masks,
        )

    def decode_pair(self, pair_index: int) -> ConditionalBatchResultV1:
        if type(pair_index) is not int:
            raise PopulationConditionalPVSAError("pair_index must be an exact integer")
        return self.render_camera_pair_batch((pair_index,))

    def iter_camera_pair_batches(
        self,
        *,
        batch_pairs: int = MAX_STREAM_BATCH_PAIRS,
    ) -> Iterator[ConditionalBatchResultV1]:
        if type(batch_pairs) is not int or not 1 <= batch_pairs <= MAX_STREAM_BATCH_PAIRS:
            raise PopulationConditionalPVSAError("batch_pairs must be an exact integer in [1,16]")
        for start in range(0, PAIR_COUNT, batch_pairs):
            stop = min(start + batch_pairs, PAIR_COUNT)
            yield self.render_camera_pair_batch(tuple(range(start, stop)))


def _role_receiver_for_exact_y1(
    base_receiver: CompactPVSAReceiverV1,
) -> CarrierComposeReceiverV1:
    semantic = base_receiver.overlay_decoder.receiver
    if not base_receiver.parsed.actuators:
        return semantic
    if (
        len(base_receiver.parsed.actuators) != 1
        or base_receiver.parsed.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
    ):
        raise PopulationConditionalPVSAError("conditional role support requires zero or one exact G74 base actuator")
    actuator = base_receiver.parsed.actuators[0]
    if actuator.operand.frame_selector not in (
        SelectedPreimageFrameSelectorV1.Y1,
        SelectedPreimageFrameSelectorV1.BOTH,
    ):
        return semantic
    try:
        _, combined = base_receiver.overlay_decoder._validate_operand_and_pairs(
            actuator.operand,
            (0,),
        )
    except V15RoleAwareOverlayError as exc:
        raise PopulationConditionalPVSAError("conditional role support failed exact G74/Y1 geometry binding") from exc
    return replace(semantic, boundary_shearlets=combined)


def _visible_role_camera_masks(
    receiver: CarrierComposeReceiverV1,
    local_pair_ids: tuple[int, ...],
) -> np.ndarray:
    """Return disjoint visible role masks at the exact camera resolution."""

    _validate_batch_ids(local_pair_ids)
    if receiver.predictor.source_pair_start != 0 or receiver.z.n_pairs != PAIR_COUNT:
        raise PopulationConditionalPVSAError("visible role support requires the complete fresh n600 semantic receiver")
    layer_by_role = {row.role: row for row in receiver.layers}
    if not set(REALIZATION_PAINT_ORDER).issubset(layer_by_role):
        raise PopulationConditionalPVSAError("semantic receiver does not expose every V15 paint role")
    ys = (np.arange(CAMERA_HEIGHT) * 384 // CAMERA_HEIGHT).clip(0, 383)
    xs = (np.arange(CAMERA_WIDTH) * 512 // CAMERA_WIDTH).clip(0, 511)
    result = np.zeros(
        (
            len(local_pair_ids),
            ROLE_COUNT,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
        ),
        dtype=bool,
    )
    for local_index, pair_id in enumerate(local_pair_ids):
        for role_index, role in enumerate(REALIZATION_PAINT_ORDER):
            mask = receiver._mask_for_layer(
                layer_by_role[role],
                pair_id,
                replace_g1_movable=True,
            ).copy()
            for later_role in REALIZATION_PAINT_ORDER[role_index + 1 :]:
                mask &= ~receiver._mask_for_layer(
                    layer_by_role[later_role],
                    pair_id,
                    replace_g1_movable=True,
                )
            result[local_index, role_index] = mask[np.ix_(ys, xs)]
    overlap = np.sum(result.astype(np.uint8), axis=1)
    if np.any(overlap > 1):
        raise PopulationConditionalPVSAError("visible role support is not disjoint after paint-order occlusion")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class PopulationConditionalPVSAArchiveBuildV1:
    """Both exact outer prices and their identical parsed successor."""

    outer_build: TaskspaceOuterArchiveBuild
    stored: ParsedPopulationConditionalPVSAMemberV1
    deflated: ParsedPopulationConditionalPVSAMemberV1
    selected: ParsedPopulationConditionalPVSAMemberV1
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.outer_build) is not TaskspaceOuterArchiveBuild:
            raise PopulationConditionalPVSAError("successor outer build changed exact type")
        expected = self.stored if self.outer_build.selected.encoding is OuterArchiveEncoding.STORED else self.deflated
        if (
            self.selected != expected
            or self.stored != self.deflated
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise PopulationConditionalPVSAError("successor outer coding custody or truth labels differ")


def _parse_outer_successor(
    exact: ParsedTaskspaceOuterArchive,
    *,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
    maximum_conditional_bytes: int,
) -> ParsedPopulationConditionalPVSAMemberV1:
    try:
        reopened = parse_taskspace_outer_archive(
            exact.archive_bytes,
            expected_encoding=exact.encoding,
            expected_archive_sha256=exact.archive_sha256,
            expected_member_sha256=exact.member_sha256,
            max_member_bytes=maximum_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise PopulationConditionalPVSAError("successor outer archive strict reopen failed") from exc
    return parse_population_conditional_pvsa_member(
        reopened.member_bytes,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
        maximum_conditional_bytes=maximum_conditional_bytes,
    )


def build_population_conditional_pvsa_archive(
    *,
    base_pvsa_member_bytes: bytes,
    conditional_operand_bytes: bytes,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
    maximum_conditional_bytes: int = MAX_CONDITIONAL_BYTES,
) -> PopulationConditionalPVSAArchiveBuildV1:
    """Build, race STORE/DEFLATE, and parse back the exact successor archive."""

    member = encode_population_conditional_pvsa_member(
        base_pvsa_member_bytes=base_pvsa_member_bytes,
        conditional_operand_bytes=conditional_operand_bytes,
    )
    try:
        outer = build_taskspace_outer_archive(
            member,
            max_member_bytes=maximum_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise PopulationConditionalPVSAError("successor exact outer archive build failed") from exc
    stored = _parse_outer_successor(
        outer.stored,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
        maximum_conditional_bytes=maximum_conditional_bytes,
    )
    deflated = _parse_outer_successor(
        outer.deflated,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
        maximum_conditional_bytes=maximum_conditional_bytes,
    )
    selected = stored if outer.selected.encoding is OuterArchiveEncoding.STORED else deflated
    return PopulationConditionalPVSAArchiveBuildV1(
        outer_build=outer,
        stored=stored,
        deflated=deflated,
        selected=selected,
    )


__all__ = [
    "CAUSAL_TRANSITION_ID",
    "FRESH_XIP2_CUSTODY_BLOCKER",
    "OPEN_BLOCKERS",
    "PASS_POLICY_ID",
    "POSE_AUTHORITY_BLOCKER",
    "PUBLIC_RUNTIME_BLOCKER",
    "RECEIVER_ID",
    "ROLE_SUPPORT_POLICY_ID",
    "SUCCESSOR_MAGIC",
    "WIRE_POLICY_ID",
    "XIP2_NUMERIC_REFERENCE_ID",
    "XIP2_POLICY_ID",
    "XIP2_SOURCE_POLICY_ID",
    "ConditionalBatchResultV1",
    "ConditionalY0ControlV1",
    "ConditionalY0ModeV1",
    "ParsedPopulationConditionalPVSAMemberV1",
    "PopulationConditionalOperandV1",
    "PopulationConditionalPVSAArchiveBuildV1",
    "PopulationConditionalPVSAError",
    "PopulationConditionalPVSAReceiverV1",
    "apply_population_conditional_to_decoded_batch",
    "build_population_conditional_pvsa_archive",
    "encode_population_conditional_pvsa_member",
    "parse_population_conditional_operand",
    "parse_population_conditional_pvsa_member",
]
