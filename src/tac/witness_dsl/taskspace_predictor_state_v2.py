# SPDX-License-Identifier: MIT
"""Typed predictor state for the original task-space inverse witness stack.

Version 1 of the generative-correction receiver intentionally binds the V9
predictor's integer Pose6 table.  A level-set predictor does not own Pose6, so
manufacturing an all-zero table would turn absence into false state.  This
module keeps the old type untouched and introduces a closed, discriminated
transport contract for new predictors.

The state contains only decoder-derived information.  Target labels, scorer
outputs, dense target frames, and oracle observations are not representable.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from tac.boundary_math import xi_pose_coder
from tac.optimization.direct_description_carrier_compose import requires_pose6_transport
from tac.witness_dsl.generative_taskspace_correction import (
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
)

SEMANTIC_HEIGHT = 384
SEMANTIC_WIDTH = 512
_SHA256_HEX_LENGTH = 64
_NONE_TRANSPORT_DOMAIN = b"tac.taskspace_transport.absent.v2"
NONE_TRANSPORT_CONTENT_SHA256 = hashlib.sha256(_NONE_TRANSPORT_DOMAIN).hexdigest()


class TaskspacePredictorStateV2Error(ValueError):
    """The predictor/transport state is malformed or causally inadmissible."""


class TransportKindV2(StrEnum):
    """Closed transport universe; adding a value requires a decoder adapter."""

    NONE = "NONE"
    V9_POSE6 = "V9_POSE6"
    SE3_XI = "SE3_XI"


class GTransportRequirementV2(StrEnum):
    """The transport actually needed by a counted G primitive family."""

    LABEL_LOCAL = "LABEL_LOCAL"
    POSE6_ADVECTED = "POSE6_ADVECTED"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TaskspacePredictorStateV2Error(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TaskspacePredictorStateV2Error("predictor binding is not canonical JSON") from exc


def _validate_pair_ids(source_pair_ids: tuple[int, ...]) -> tuple[int, ...]:
    if type(source_pair_ids) is not tuple or not source_pair_ids:
        raise TaskspacePredictorStateV2Error("source_pair_ids must be a nonempty exact tuple")
    if any(type(value) is not int for value in source_pair_ids):
        raise TaskspacePredictorStateV2Error("source_pair_ids must contain exact integers")
    expected = tuple(range(source_pair_ids[0], source_pair_ids[0] + len(source_pair_ids)))
    if source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise TaskspacePredictorStateV2Error("source_pair_ids must be one ordered contiguous subset of [0,600)")
    return expected


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[object]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class NoTransportV2:
    """Canonical absence marker.  It carries zero counted transport bytes."""

    kind: TransportKindV2 = field(default=TransportKindV2.NONE, init=False)

    @property
    def counted_content_sha256(self) -> str:
        return NONE_TRANSPORT_CONTENT_SHA256

    @property
    def decoded_values_sha256(self) -> str:
        return NONE_TRANSPORT_CONTENT_SHA256

    @property
    def counted_bytes(self) -> int:
        return 0

    def binding_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "counted_bytes": 0,
            "counted_content_sha256": NONE_TRANSPORT_CONTENT_SHA256,
            "decoded_values_sha256": NONE_TRANSPORT_CONTENT_SHA256,
        }


@dataclass(frozen=True, slots=True)
class V9Pose6TransportV2:
    """Exact raw big-endian int16 Pose6 transport owned by a counted V9 P."""

    counted_payload: bytes
    pose6_codes: np.ndarray
    source_pair_ids: tuple[int, ...]
    predictor_program_sha256: str
    kind: TransportKindV2 = field(default=TransportKindV2.V9_POSE6, init=False)

    def __post_init__(self) -> None:
        if type(self.counted_payload) is not bytes or not self.counted_payload:
            raise TaskspacePredictorStateV2Error("V9 Pose6 transport requires exact counted bytes")
        expected_ids = _validate_pair_ids(self.source_pair_ids)
        _require_sha256(self.predictor_program_sha256, "transport.predictor_program_sha256")
        values = np.asarray(self.pose6_codes)
        if values.shape != (len(expected_ids), 6) or values.dtype.kind not in "iu":
            raise TaskspacePredictorStateV2Error("V9 Pose6 must be an integer [pair,6] array")
        if int(values.min()) < -32768 or int(values.max()) > 32767:
            raise TaskspacePredictorStateV2Error("V9 Pose6 escaped canonical int16 range")
        immutable = _immutable_array(values, dtype=np.dtype(">i2"))
        if self.counted_payload != memoryview(immutable).cast("B").tobytes():
            raise TaskspacePredictorStateV2Error(
                "V9 Pose6 counted bytes do not decode to the supplied exact int16 state"
            )
        object.__setattr__(self, "pose6_codes", immutable)

    @property
    def counted_content_sha256(self) -> str:
        return _sha256(self.counted_payload)

    @property
    def decoded_values_sha256(self) -> str:
        return _sha256(memoryview(self.pose6_codes).cast("B"))

    @property
    def counted_bytes(self) -> int:
        return len(self.counted_payload)

    def binding_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "counted_bytes": self.counted_bytes,
            "counted_content_sha256": self.counted_content_sha256,
            "decoded_dtype": ">i2",
            "decoded_shape": list(self.pose6_codes.shape),
            "decoded_values_sha256": self.decoded_values_sha256,
            "source_pair_ids": list(self.source_pair_ids),
            "predictor_program_sha256": self.predictor_program_sha256,
        }


def _parse_xip2_exact(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Use the canonical XIP2 decoders while independently enforcing EOF."""

    if type(payload) is not bytes or len(payload) < 8 or payload[:4] != b"XIP2":
        raise TaskspacePredictorStateV2Error("SE3_XI transport must be an exact XIP2 payload")
    try:
        coder_id, pair_count, dimensions = struct.unpack_from("<BHB", payload, 4)
        offset = 8 + dimensions * 4
        if offset > len(payload):
            raise TaskspacePredictorStateV2Error("XIP2 scales are truncated")
        scales = np.frombuffer(payload[8:offset], dtype=np.float32).copy()
        if coder_id == 0:
            stop = offset + pair_count * dimensions * 2
            if stop > len(payload):
                raise TaskspacePredictorStateV2Error("raw XIP2 integer table is truncated")
            q = np.frombuffer(payload[offset:stop], dtype=np.int16).reshape(pair_count, dimensions).copy()
            offset = stop
        elif coder_id == 1:
            columns: list[np.ndarray] = []
            for _ in range(dimensions):
                column, offset = xi_pose_coder._channel_delta_decode(payload, offset, pair_count)
                columns.append(column)
            q = np.stack(columns, axis=1).astype(np.int16)
        elif coder_id == 2:
            from tac.boundary_math.xi_spline_residual_coder import decode_spline_residual_body

            q, offset = decode_spline_residual_body(payload, offset, pair_count, dimensions, scales)
        elif coder_id == 3:
            from tac.boundary_math.xi_spline_residual_coder import decode_delta_res_body

            q, offset = decode_delta_res_body(payload, offset, pair_count, dimensions)
        else:
            raise TaskspacePredictorStateV2Error(f"unknown XIP2 coder id {coder_id}")
    except TaskspacePredictorStateV2Error:
        raise
    except (IndexError, OverflowError, struct.error, ValueError) as exc:
        raise TaskspacePredictorStateV2Error("XIP2 payload is truncated or malformed") from exc
    if offset != len(payload):
        raise TaskspacePredictorStateV2Error("XIP2 payload has unconsumed trailing bytes")
    try:
        canonical_q, canonical_scales = xi_pose_coder.parse_xi_payload(payload)
    except (ValueError, struct.error) as exc:
        raise TaskspacePredictorStateV2Error("canonical XIP2 parser refused transport") from exc
    if not np.array_equal(q, canonical_q) or not np.array_equal(scales, canonical_scales):
        raise TaskspacePredictorStateV2Error("XIP2 exact parser disagrees with canonical decoder")
    return q, scales


@dataclass(frozen=True, slots=True)
class SE3XiTransportV2:
    """Exact counted XIP2 trajectory, explicitly not a PoseNet/Pose6 claim."""

    counted_payload: bytes
    source_pair_ids: tuple[int, ...]
    predictor_program_sha256: str
    q_codes: np.ndarray = field(init=False, repr=False)
    scales: np.ndarray = field(init=False, repr=False)
    xi: np.ndarray = field(init=False, repr=False)
    kind: TransportKindV2 = field(default=TransportKindV2.SE3_XI, init=False)

    def __post_init__(self) -> None:
        expected_ids = _validate_pair_ids(self.source_pair_ids)
        _require_sha256(self.predictor_program_sha256, "transport.predictor_program_sha256")
        q_codes, scales = _parse_xip2_exact(self.counted_payload)
        if q_codes.shape != (len(expected_ids), 6) or q_codes.dtype != np.int16:
            raise TaskspacePredictorStateV2Error("SE3_XI must decode to int16 [pair,6]")
        if scales.shape != (6,) or scales.dtype != np.float32:
            raise TaskspacePredictorStateV2Error("SE3_XI must carry six fp32 scales")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise TaskspacePredictorStateV2Error("SE3_XI scales must be finite and positive")
        xi = xi_pose_coder.dequantize_xi(q_codes, scales)
        if xi.shape != (len(expected_ids), 6) or not np.all(np.isfinite(xi)):
            raise TaskspacePredictorStateV2Error("SE3_XI decoded trajectory is invalid")
        object.__setattr__(self, "q_codes", _immutable_array(q_codes, dtype=np.dtype(np.int16)))
        object.__setattr__(self, "scales", _immutable_array(scales, dtype=np.dtype(np.float32)))
        object.__setattr__(self, "xi", _immutable_array(xi, dtype=np.dtype(np.float64)))

    @property
    def counted_content_sha256(self) -> str:
        return _sha256(self.counted_payload)

    @property
    def decoded_values_sha256(self) -> str:
        return _sha256(memoryview(self.q_codes).cast("B"))

    @property
    def counted_bytes(self) -> int:
        return len(self.counted_payload)

    def binding_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "counted_bytes": self.counted_bytes,
            "counted_content_sha256": self.counted_content_sha256,
            "decoded_dtype": "int16",
            "decoded_shape": list(self.q_codes.shape),
            "decoded_values_sha256": self.decoded_values_sha256,
            "scales_sha256": _sha256(memoryview(self.scales).cast("B")),
            "source_pair_ids": list(self.source_pair_ids),
            "predictor_program_sha256": self.predictor_program_sha256,
            "semantic_claim": "transport_statistic_not_posenet_or_pose6",
        }


TaskspaceTransportV2 = NoTransportV2 | V9Pose6TransportV2 | SE3XiTransportV2


@runtime_checkable
class PredictorSemanticStateProtocolV2(Protocol):
    """The semantic foreign-key surface consumed by pose-independent A."""

    predictor_program_sha256: str
    predictor_renderer_sha256: str
    source_pair_ids: tuple[int, ...]
    labels: np.ndarray

    @property
    def pair_start(self) -> int: ...

    @property
    def pair_count(self) -> int: ...

    @property
    def labels_sha256(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TaskspacePredictorStateV2:
    """Counted P bytes plus decoder-owned semantics and typed transport."""

    predictor_program: bytes
    predictor_renderer_sha256: str
    source_archive_sha256: str
    source_runtime_sha256: str
    source_member_name: str
    source_pair_ids: tuple[int, ...]
    labels: np.ndarray
    transport: TaskspaceTransportV2

    def __post_init__(self) -> None:
        if type(self.predictor_program) is not bytes or not self.predictor_program:
            raise TaskspacePredictorStateV2Error("predictor_program must be exact nonempty member bytes")
        for field_name in (
            "predictor_renderer_sha256",
            "source_archive_sha256",
            "source_runtime_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.source_member_name != "0.bin":
            raise TaskspacePredictorStateV2Error("source member must be the canonical sole member 0.bin")
        expected_ids = _validate_pair_ids(self.source_pair_ids)
        labels = np.asarray(self.labels)
        if labels.dtype != np.uint8 or labels.shape != (
            len(expected_ids),
            SEMANTIC_HEIGHT,
            SEMANTIC_WIDTH,
        ):
            raise TaskspacePredictorStateV2Error(
                "predictor labels must be decoder-owned uint8 [pair,384,512] semantics"
            )
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise TaskspacePredictorStateV2Error("predictor labels escaped the five-class universe")
        if type(self.transport) not in {NoTransportV2, V9Pose6TransportV2, SE3XiTransportV2}:
            raise TaskspacePredictorStateV2Error("transport must use one exact closed V2 discriminant")
        if type(self.transport) is not NoTransportV2:
            if self.transport.source_pair_ids != expected_ids:
                raise TaskspacePredictorStateV2Error("transport pair IDs differ from predictor state")
            if self.transport.predictor_program_sha256 != self.predictor_program_sha256:
                raise TaskspacePredictorStateV2Error("transport is bound to different predictor bytes")
        object.__setattr__(self, "labels", _immutable_array(labels, dtype=np.dtype(np.uint8)))

    @property
    def predictor_program_sha256(self) -> str:
        return _sha256(self.predictor_program)

    @property
    def predictor_program_bytes(self) -> int:
        return len(self.predictor_program)

    @property
    def source_member_sha256(self) -> str:
        return self.predictor_program_sha256

    @property
    def pair_start(self) -> int:
        return self.source_pair_ids[0]

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    @property
    def labels_sha256(self) -> str:
        return _sha256(memoryview(self.labels).cast("B"))

    @property
    def semantic_binding_sha256(self) -> str:
        """Transport-independent foreign key for the pose-independent A decoder."""

        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.taskspace_predictor_semantic_binding.v2",
                    "predictor_program_sha256": self.predictor_program_sha256,
                    "predictor_renderer_sha256": self.predictor_renderer_sha256,
                    "source_archive_sha256": self.source_archive_sha256,
                    "source_runtime_sha256": self.source_runtime_sha256,
                    "source_member_name": self.source_member_name,
                    "source_pair_ids": list(self.source_pair_ids),
                    "labels_sha256": self.labels_sha256,
                }
            )
        )

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.taskspace_predictor_binding.v2",
                    "semantic_binding_sha256": self.semantic_binding_sha256,
                    "transport": self.transport.binding_record(),
                }
            )
        )

    @classmethod
    def from_v1(
        cls,
        state: PredictorSemanticStateV1,
        *,
        predictor_program: bytes,
        counted_pose6_payload: bytes,
        source_archive_sha256: str,
        source_runtime_sha256: str,
        source_member_name: str = "0.bin",
    ) -> TaskspacePredictorStateV2:
        """Lift V1 without weakening or changing its byte/read contract."""

        if type(state) is not PredictorSemanticStateV1:
            raise TaskspacePredictorStateV2Error("V1 adapter requires exact PredictorSemanticStateV1")
        if _sha256(predictor_program) != state.predictor_program_sha256:
            raise TaskspacePredictorStateV2Error("V1 predictor hash differs from supplied counted bytes")
        transport = V9Pose6TransportV2(
            counted_payload=counted_pose6_payload,
            pose6_codes=state.pose6_codes,
            source_pair_ids=state.source_pair_ids,
            predictor_program_sha256=state.predictor_program_sha256,
        )
        return cls(
            predictor_program=predictor_program,
            predictor_renderer_sha256=state.predictor_renderer_sha256,
            source_archive_sha256=source_archive_sha256,
            source_runtime_sha256=source_runtime_sha256,
            source_member_name=source_member_name,
            source_pair_ids=state.source_pair_ids,
            labels=state.labels,
            transport=transport,
        )

    def as_v1(self) -> PredictorSemanticStateV1:
        """Project only an actual V9 Pose6 state back onto the exact V1 type."""

        if type(self.transport) is not V9Pose6TransportV2:
            raise TaskspacePredictorStateV2Error("NONE/SE3_XI cannot be projected to V1 by fabricating Pose6")
        return PredictorSemanticStateV1(
            predictor_program_sha256=self.predictor_program_sha256,
            predictor_renderer_sha256=self.predictor_renderer_sha256,
            source_pair_ids=self.source_pair_ids,
            labels=self.labels,
            pose6_codes=self.transport.pose6_codes,
        )


def g_transport_requirement(program: GenerativeCorrectionProgramV1) -> GTransportRequirementV2:
    """Derive the exact transport dependency from live G primitive families."""

    if type(program) is not GenerativeCorrectionProgramV1:
        raise TaskspacePredictorStateV2Error("G dependency predicate requires exact program type")
    gained_atoms = (*program.topology_events, *program.island_shapes)
    if (
        any(requires_pose6_transport(row) for row in gained_atoms)
        or program.worldsheet_tracks
        or program.worldsheet_knots
    ):
        return GTransportRequirementV2.POSE6_ADVECTED
    return GTransportRequirementV2.LABEL_LOCAL


def require_g_transport_admission(
    state: TaskspacePredictorStateV2,
    program: GenerativeCorrectionProgramV1,
) -> GTransportRequirementV2:
    """Refuse unsupported transport before any G receiver mutation can run."""

    if type(state) is not TaskspacePredictorStateV2:
        raise TaskspacePredictorStateV2Error("G admission requires exact V2 predictor state")
    requirement = g_transport_requirement(program)
    if requirement is GTransportRequirementV2.LABEL_LOCAL:
        return requirement
    if state.transport.kind is TransportKindV2.NONE:
        raise TaskspacePredictorStateV2Error("transport-dependent G family refuses NONE before receiver mutation")
    if state.transport.kind is TransportKindV2.SE3_XI:
        raise TaskspacePredictorStateV2Error("SE3_XI G event/island/worldsheet parity adapter is not implemented")
    return requirement


__all__ = [
    "NONE_TRANSPORT_CONTENT_SHA256",
    "GTransportRequirementV2",
    "NoTransportV2",
    "PredictorSemanticStateProtocolV2",
    "SE3XiTransportV2",
    "TaskspacePredictorStateV2",
    "TaskspacePredictorStateV2Error",
    "TaskspaceTransportV2",
    "TransportKindV2",
    "V9Pose6TransportV2",
    "g_transport_requirement",
    "require_g_transport_admission",
]
