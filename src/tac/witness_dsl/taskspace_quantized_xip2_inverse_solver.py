# SPDX-License-Identifier: MIT
"""Direct counted-XIP2 integer search through the real G13 uint8 receiver.

G16 is an encoder-side, research-only bounded search.  Its state is the exact
``int16`` table decoded from each counted XIP2 packet; no affine pose model,
Jacobian surrogate, dense target, scorer, or archive builder lives here.  The
external objective callback owns scorer and rebuilt-archive custody and returns
only scalar observations and hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np

from tac.boundary_math.xi_pose_coder import (
    XiPoseCoderError,
    parse_xi_payload,
    quantize_xi,
    serialize_xi_payload,
)
from tac.witness_dsl.taskspace_chronological_a3_encoder import XIP2Coder
from tac.witness_dsl.taskspace_counted_xip2_chronological_a3 import (
    ABSENT_GUIDANCE_SHA256,
    CompiledCountedXIP2ChronologicalA3V1,
    CountedXIP2A3Interpretation,
    CountedXIP2A3Mode,
    CountedXIP2ChronologicalA3Error,
    CountedXIP2ChronologicalA3ProgramV1,
    compile_counted_xip2_chronological_a3,
    compile_counted_xip2_chronological_a3_pass,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    PassConditionalAError,
    PassConditionalASourceBindingV1,
    PassConditionalASurfaceV1,
    derive_pass_conditional_a_source_binding,
)

MANIFEST_SCHEMA: Final = "tac.quantized_xip2_inverse_search_manifest.v1"
EVALUATION_STAGE_SCHEMA: Final = "tac.quantized_xip2_inverse_search_evaluation_stage.v1"
TERMINAL_STAGE_SCHEMA: Final = "tac.quantized_xip2_inverse_search_terminal_stage.v1"
ARM_SCHEMA: Final = "tac.quantized_xip2_inverse_search_arm.v1"
CANDIDATE_SCHEMA: Final = "tac.quantized_xip2_inverse_search_candidate.v1"
ARM_RESULT_SCHEMA: Final = "tac.quantized_xip2_inverse_search_arm_result.v1"
RESULT_SCHEMA: Final = "tac.quantized_xip2_inverse_search_result.v1"
RECEIPT_SCHEMA: Final = "tac.quantized_xip2_inverse_search_receipt.v1"
FORMULATION_SCOPE: Final = "FIXED_SCALE_DIRECT_INT16_COORDINATE_SEARCH_V1"
ROOT_INTEGRATION_BLOCKER: Final = (
    "ROOT_REVIEW_REQUIRED_TO_BIND_G16_CALLBACK_PROVIDER_INTO_FROZEN_G14_N2_ARCHIVE_SCORER_PATH"
)
RATE_DENOMINATOR_BYTES: Final = 37_545_489
RATE_MULTIPLIER: Final = 25.0
_POSITIVE_ZERO_F32: Final = b"\x00\x00\x00\x00"
_STAGE_RE: Final = re.compile(r"stage-(\d{8})-(evaluation|terminal)\.json\Z")
_CONTROL_ORDER: Final = {
    CountedXIP2A3Mode.PASS_P0_V1: 0,
    CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1: 1,
}


class QuantizedXIP2InverseError(ValueError):
    """A G16 input, receiver closure, callback, or durable chain failed closed."""


class QuantizedXIP2ObjectiveAuthorityV1(StrEnum):
    """The only callback authorities representable by G16 V1."""

    SYNTHETIC_TEST_ONLY = "SYNTHETIC_TEST_ONLY"
    FROZEN_POSENET_CALLBACK_ADVISORY = "FROZEN_POSENET_CALLBACK_ADVISORY"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_sha256(value: object, *, field: str, allow_absent: bool = True) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_absent and value == ABSENT_GUIDANCE_SHA256)
    ):
        raise QuantizedXIP2InverseError(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(value: object, *, field: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise QuantizedXIP2InverseError(f"{field} must be an exact integer{suffix}")
    return value


def _require_finite_float(value: object, *, field: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (nonnegative and value < 0.0):
        qualifier = " finite nonnegative" if nonnegative else " finite"
        raise QuantizedXIP2InverseError(f"{field} must be an exact{qualifier} float")
    return value


def _canonical_pitch(value: object) -> float:
    pitch = _require_finite_float(value, field="pitch")
    try:
        packed = struct.pack(">f", pitch)
    except (OverflowError, struct.error) as exc:
        raise QuantizedXIP2InverseError("pitch escaped finite fp32 range") from exc
    canonical = struct.unpack(">f", packed)[0]
    if not math.isfinite(canonical):
        raise QuantizedXIP2InverseError("pitch escaped finite fp32 range")
    if canonical == 0.0 and packed != _POSITIVE_ZERO_F32:
        raise QuantizedXIP2InverseError("negative-zero pitch is a forbidden alias")
    if float(canonical) != pitch:
        raise QuantizedXIP2InverseError("pitch must already be fp32-canonical")
    return canonical


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
        raise QuantizedXIP2InverseError("G16 record is not finite canonical ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise QuantizedXIP2InverseError(f"{label} must be exact bytes")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QuantizedXIP2InverseError(f"{label} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=no_duplicates)
    except QuantizedXIP2InverseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuantizedXIP2InverseError(f"{label} is not canonical ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise QuantizedXIP2InverseError(f"{label} is noncanonical or changed on re-emit")
    return value


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _implementation_sha256() -> str:
    try:
        return _sha256(Path(__file__).read_bytes())
    except OSError as exc:
        raise QuantizedXIP2InverseError("could not bind the G16 implementation bytes") from exc


@dataclass(frozen=True, slots=True)
class QuantizedXIP2SearchConfigV1:
    seed: int
    q_levels: tuple[int, ...]
    coders: tuple[XIP2Coder, ...]
    interpretations: tuple[CountedXIP2A3Interpretation, ...]
    step_schedule: tuple[int, ...]
    sweeps_per_step: int
    max_callback_evaluations: int
    pitch: float

    def __post_init__(self) -> None:
        _require_exact_int(self.seed, field="seed")
        if type(self.q_levels) is not tuple or not self.q_levels:
            raise QuantizedXIP2InverseError("q_levels must be one nonempty ordered tuple")
        for index, value in enumerate(self.q_levels):
            _require_exact_int(value, field=f"q_levels[{index}]", minimum=1)
            if value > 32767:
                raise QuantizedXIP2InverseError("q_levels must stay inside [1,32767]")
        if len(set(self.q_levels)) != len(self.q_levels):
            raise QuantizedXIP2InverseError("q_levels must be unique without implicit reordering")
        if type(self.coders) is not tuple or not self.coders:
            raise QuantizedXIP2InverseError("coders must be one nonempty ordered tuple")
        if any(type(coder) is not XIP2Coder for coder in self.coders):
            raise QuantizedXIP2InverseError("coders escaped the closed G9 XIP2Coder enum")
        if len(set(self.coders)) != len(self.coders):
            raise QuantizedXIP2InverseError("coders must be unique without implicit reordering")
        if type(self.interpretations) is not tuple or not self.interpretations:
            raise QuantizedXIP2InverseError("interpretations must be one nonempty ordered tuple")
        if any(type(interpretation) is not CountedXIP2A3Interpretation for interpretation in self.interpretations):
            raise QuantizedXIP2InverseError("interpretations escaped the active G13 domain enum")
        if len(set(self.interpretations)) != len(self.interpretations):
            raise QuantizedXIP2InverseError("interpretations must be unique without implicit reordering")
        if type(self.step_schedule) is not tuple or not self.step_schedule:
            raise QuantizedXIP2InverseError("step_schedule must be one nonempty ordered tuple")
        for index, value in enumerate(self.step_schedule):
            _require_exact_int(value, field=f"step_schedule[{index}]", minimum=1)
        _require_exact_int(self.sweeps_per_step, field="sweeps_per_step", minimum=1)
        _require_exact_int(
            self.max_callback_evaluations,
            field="max_callback_evaluations",
            minimum=1,
        )
        minimum_evaluations = 2 + len(self.q_levels) * len(self.coders) * len(self.interpretations)
        if self.max_callback_evaluations < minimum_evaluations:
            raise QuantizedXIP2InverseError(
                "max_callback_evaluations cannot enumerate both controls and every configured arm"
            )
        object.__setattr__(self, "pitch", _canonical_pitch(self.pitch))

    def as_dict(self) -> dict[str, Any]:
        return {
            "coders": [coder.value for coder in self.coders],
            "interpretations": [value.value for value in self.interpretations],
            "max_callback_evaluations": self.max_callback_evaluations,
            "pitch": self.pitch,
            "q_levels": list(self.q_levels),
            "seed": self.seed,
            "step_schedule": list(self.step_schedule),
            "sweeps_per_step": self.sweeps_per_step,
        }

    @property
    def config_sha256(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


@dataclass(frozen=True, slots=True)
class QuantizedXIP2EvaluationRequestV1:
    """One exact G13 receive result plus low-dimensional search metadata.

    This type intentionally has no ``as_dict`` or JSON serializer: the compiled
    object and its decoded uint8 frames remain in memory only.
    """

    compiled: CompiledCountedXIP2ChronologicalA3V1
    evaluation_id: str
    arm_index: int | None
    arm_id: str | None
    q_level: int | None
    coder: XIP2Coder | None
    interpretation: CountedXIP2A3Interpretation
    q: np.ndarray | None
    scales: np.ndarray | None
    source_binding_sha256: str
    lineage_sha256: str
    config_sha256: str
    initial_xi_sha256: str
    packet_sha256: str
    q_sha256: str | None
    scales_sha256: str | None
    xip2_payload_sha256: str
    output_camera_y0_sha256: str
    output_camera_y1_sha256: str
    output_chronology_sha256: str

    def __post_init__(self) -> None:
        if type(self.compiled) is not CompiledCountedXIP2ChronologicalA3V1:
            raise QuantizedXIP2InverseError("evaluation request requires the exact compiled G13 object")
        for name in (
            "evaluation_id",
            "source_binding_sha256",
            "lineage_sha256",
            "config_sha256",
            "initial_xi_sha256",
            "packet_sha256",
            "xip2_payload_sha256",
            "output_camera_y0_sha256",
            "output_camera_y1_sha256",
            "output_chronology_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)
        if type(self.interpretation) is not CountedXIP2A3Interpretation:
            raise QuantizedXIP2InverseError("request interpretation escaped the G13 enum")
        compiled = self.compiled
        receipt = compiled.receipt
        if (
            self.source_binding_sha256 != compiled.source_binding.binding_sha256
            or self.packet_sha256 != receipt.packet_sha256
            or self.xip2_payload_sha256 != receipt.xip2_payload_sha256
            or self.output_camera_y0_sha256 != receipt.output_camera_y0_sha256
            or self.output_camera_y1_sha256 != receipt.output_camera_y1_sha256
            or self.output_chronology_sha256 != receipt.output_chronology_sha256
            or self.interpretation is not compiled.program.interpretation
        ):
            raise QuantizedXIP2InverseError("request hashes/metadata differ from the exact G13 compile")
        if compiled.program.mode is CountedXIP2A3Mode.XIP2_WARP_V1:
            if (
                type(self.arm_index) is not int
                or self.arm_index < 0
                or type(self.arm_id) is not str
                or not self.arm_id
                or type(self.q_level) is not int
                or not 1 <= self.q_level <= 32767
                or type(self.coder) is not XIP2Coder
                or type(self.q_sha256) is not str
                or type(self.scales_sha256) is not str
            ):
                raise QuantizedXIP2InverseError("active request omitted exact arm/q/coder metadata")
            _require_sha256(self.q_sha256, field="q_sha256")
            _require_sha256(self.scales_sha256, field="scales_sha256")
            if self.q is None or self.scales is None or compiled.parsed_packet.transport is None:
                raise QuantizedXIP2InverseError("active request omitted decoded XIP2 state")
            q = np.asarray(self.q)
            scales = np.asarray(self.scales)
            transport = compiled.parsed_packet.transport
            if (
                q.dtype != np.int16
                or q.shape != transport.q_codes.shape
                or scales.dtype != np.float32
                or scales.shape != (6,)
                or not np.array_equal(q, transport.q_codes)
                or not np.array_equal(scales, transport.scales)
                or int(np.abs(q.astype(np.int32)).max(initial=0)) > self.q_level
                or _array_sha256(q) != self.q_sha256
                or _array_sha256(scales) != self.scales_sha256
            ):
                raise QuantizedXIP2InverseError("request q/scales differ from actual decoded XIP2 state")
            object.__setattr__(self, "q", _immutable_array(q, dtype=np.dtype(np.int16)))
            object.__setattr__(self, "scales", _immutable_array(scales, dtype=np.dtype(np.float32)))
        else:
            if compiled.program.mode not in _CONTROL_ORDER:
                raise QuantizedXIP2InverseError("request mode escaped controls/active G16 universe")
            if any(
                value is not None
                for value in (
                    self.arm_index,
                    self.arm_id,
                    self.q_level,
                    self.coder,
                    self.q,
                    self.scales,
                    self.q_sha256,
                    self.scales_sha256,
                )
            ):
                raise QuantizedXIP2InverseError("control request fabricated active q metadata")

    @property
    def mode(self) -> CountedXIP2A3Mode:
        return self.compiled.program.mode


@dataclass(frozen=True, slots=True)
class QuantizedXIP2ObjectiveObservationV1:
    per_pair_pose6_mse: tuple[float, ...]
    d_seg: float
    seg_prediction_sha256: str
    archive_bytes: int
    archive_sha256: str
    authority: QuantizedXIP2ObjectiveAuthorityV1
    frozen_scorer_sha256: str
    target_custody_sha256: str
    measurement_custody_sha256: str

    def __post_init__(self) -> None:
        if type(self.per_pair_pose6_mse) is not tuple or not self.per_pair_pose6_mse:
            raise QuantizedXIP2InverseError("per_pair_pose6_mse must be one nonempty exact tuple")
        for index, value in enumerate(self.per_pair_pose6_mse):
            _require_finite_float(
                value,
                field=f"per_pair_pose6_mse[{index}]",
                nonnegative=True,
            )
        _require_finite_float(self.d_seg, field="d_seg", nonnegative=True)
        _require_exact_int(self.archive_bytes, field="archive_bytes", minimum=0)
        if type(self.authority) is not QuantizedXIP2ObjectiveAuthorityV1:
            raise QuantizedXIP2InverseError("callback authority escaped the closed V1 enum")
        for name in (
            "seg_prediction_sha256",
            "archive_sha256",
            "frozen_scorer_sha256",
            "target_custody_sha256",
            "measurement_custody_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)


class QuantizedXIP2ObjectiveCallbackV1(Protocol):
    def __call__(
        self,
        request: QuantizedXIP2EvaluationRequestV1,
    ) -> QuantizedXIP2ObjectiveObservationV1: ...


def quantized_xip2_complete_score(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    _require_finite_float(d_seg, field="d_seg", nonnegative=True)
    _require_finite_float(d_pose, field="d_pose", nonnegative=True)
    _require_exact_int(archive_bytes, field="archive_bytes", minimum=0)
    score = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + (RATE_MULTIPLIER * archive_bytes / RATE_DENOMINATOR_BYTES)
    if not math.isfinite(score):
        raise QuantizedXIP2InverseError("complete score escaped finite binary64")
    return score


def quantized_xip2_strict_archive_ceiling(
    *,
    reference_archive_bytes: int,
    delta_distortion: float,
) -> int | None:
    _require_exact_int(reference_archive_bytes, field="reference_archive_bytes", minimum=0)
    _require_finite_float(delta_distortion, field="delta_distortion")
    if delta_distortion >= 0.0:
        return None
    boundary = reference_archive_bytes - (RATE_DENOMINATOR_BYTES / RATE_MULTIPLIER) * delta_distortion
    if not math.isfinite(boundary):
        raise QuantizedXIP2InverseError("strict archive ceiling escaped finite binary64")
    return math.ceil(boundary) - 1


@dataclass(frozen=True, slots=True)
class QuantizedXIP2ArmV1:
    arm_index: int
    arm_id: str
    q_level: int
    coder: XIP2Coder
    interpretation: CountedXIP2A3Interpretation
    frozen_scales: tuple[float, ...]
    initial_q_sha256: str
    coordinate_order_sha256: str
    schema: Literal["tac.quantized_xip2_inverse_search_arm.v1"] = ARM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ARM_SCHEMA:
            raise QuantizedXIP2InverseError("arm schema changed")
        _require_exact_int(self.arm_index, field="arm_index", minimum=0)
        if type(self.arm_id) is not str or not self.arm_id:
            raise QuantizedXIP2InverseError("arm_id must be one nonempty exact string")
        _require_exact_int(self.q_level, field="q_level", minimum=1)
        if self.q_level > 32767 or type(self.coder) is not XIP2Coder:
            raise QuantizedXIP2InverseError("arm q/coder escaped the configured universe")
        if type(self.interpretation) is not CountedXIP2A3Interpretation:
            raise QuantizedXIP2InverseError("arm interpretation escaped G13")
        if type(self.frozen_scales) is not tuple or len(self.frozen_scales) != 6:
            raise QuantizedXIP2InverseError("arm must freeze exactly six fp32 scales")
        for index, value in enumerate(self.frozen_scales):
            _require_finite_float(value, field=f"frozen_scales[{index}]")
            if value <= 0.0 or float(np.float32(value)) != value:
                raise QuantizedXIP2InverseError("arm scales must be positive fp32-canonical values")
        _require_sha256(self.initial_q_sha256, field="initial_q_sha256")
        _require_sha256(self.coordinate_order_sha256, field="coordinate_order_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_index": self.arm_index,
            "coder": self.coder.value,
            "coordinate_order_sha256": self.coordinate_order_sha256,
            "frozen_scales": list(self.frozen_scales),
            "initial_q_sha256": self.initial_q_sha256,
            "interpretation": self.interpretation.value,
            "q_level": self.q_level,
            "schema": self.schema,
        }

    def to_canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class QuantizedXIP2CandidateV1:
    callback_evaluation_index: int
    evaluation_id: str
    mode: CountedXIP2A3Mode
    arm_index: int | None
    arm_id: str | None
    q_level: int | None
    coder: XIP2Coder | None
    interpretation: CountedXIP2A3Interpretation
    q: tuple[tuple[int, ...], ...]
    scales: tuple[float, ...]
    source_binding_sha256: str
    lineage_sha256: str
    packet_sha256: str
    packet_bytes: int
    q_sha256: str | None
    scales_sha256: str | None
    xip2_payload_sha256: str
    xip2_payload_bytes: int
    output_camera_y0_sha256: str
    output_camera_y1_sha256: str
    output_chronology_sha256: str
    per_pair_pose6_mse: tuple[float, ...]
    d_pose: float
    d_seg: float
    seg_prediction_sha256: str
    archive_bytes: int
    archive_sha256: str
    authority: QuantizedXIP2ObjectiveAuthorityV1
    frozen_scorer_sha256: str
    target_custody_sha256: str
    measurement_custody_sha256: str
    complete_score: float
    reference_control_packet_sha256: str | None
    reference_d_pose: float | None
    reference_d_seg: float | None
    reference_archive_bytes: int | None
    delta_distortion: float | None
    delta_s: float | None
    strict_archive_ceiling: int | None
    admitted: bool | None
    admission_reason: str
    schema: Literal["tac.quantized_xip2_inverse_search_candidate.v1"] = CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_SCHEMA:
            raise QuantizedXIP2InverseError("candidate schema changed")
        _require_exact_int(
            self.callback_evaluation_index,
            field="callback_evaluation_index",
            minimum=1,
        )
        for name in (
            "evaluation_id",
            "source_binding_sha256",
            "lineage_sha256",
            "packet_sha256",
            "xip2_payload_sha256",
            "output_camera_y0_sha256",
            "output_camera_y1_sha256",
            "output_chronology_sha256",
            "seg_prediction_sha256",
            "archive_sha256",
            "frozen_scorer_sha256",
            "target_custody_sha256",
            "measurement_custody_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)
        if type(self.mode) is not CountedXIP2A3Mode or type(self.interpretation) is not CountedXIP2A3Interpretation:
            raise QuantizedXIP2InverseError("candidate mode/interpretation escaped G13")
        _require_exact_int(self.packet_bytes, field="packet_bytes", minimum=0)
        _require_exact_int(self.xip2_payload_bytes, field="xip2_payload_bytes", minimum=0)
        _require_exact_int(self.archive_bytes, field="archive_bytes", minimum=0)
        if type(self.per_pair_pose6_mse) is not tuple or not self.per_pair_pose6_mse:
            raise QuantizedXIP2InverseError("candidate pose losses must be a nonempty tuple")
        for index, value in enumerate(self.per_pair_pose6_mse):
            _require_finite_float(value, field=f"per_pair_pose6_mse[{index}]", nonnegative=True)
        for name in ("d_pose", "d_seg", "complete_score"):
            _require_finite_float(getattr(self, name), field=name, nonnegative=True)
        if type(self.authority) is not QuantizedXIP2ObjectiveAuthorityV1:
            raise QuantizedXIP2InverseError("candidate callback authority changed")
        derived_pose = math.fsum(self.per_pair_pose6_mse) / len(self.per_pair_pose6_mse)
        if self.d_pose != derived_pose or self.complete_score != quantized_xip2_complete_score(
            d_seg=self.d_seg,
            d_pose=self.d_pose,
            archive_bytes=self.archive_bytes,
        ):
            raise QuantizedXIP2InverseError("candidate score is not exactly derived from callback scalars")
        if self.mode is CountedXIP2A3Mode.XIP2_WARP_V1:
            if (
                type(self.arm_index) is not int
                or self.arm_index < 0
                or type(self.arm_id) is not str
                or not self.arm_id
                or type(self.q_level) is not int
                or not 1 <= self.q_level <= 32767
                or type(self.coder) is not XIP2Coder
                or type(self.q) is not tuple
                or len(self.q) != len(self.per_pair_pose6_mse)
                or any(type(row) is not tuple or len(row) != 6 for row in self.q)
                or type(self.scales) is not tuple
                or len(self.scales) != 6
                or type(self.q_sha256) is not str
                or type(self.scales_sha256) is not str
            ):
                raise QuantizedXIP2InverseError("active candidate omitted exact q/arm metadata")
            q_values: list[list[int]] = []
            for row in self.q:
                q_row: list[int] = []
                for value in row:
                    _require_exact_int(value, field="candidate q")
                    if abs(value) > self.q_level:
                        raise QuantizedXIP2InverseError("candidate q escaped its arm lattice")
                    q_row.append(value)
                q_values.append(q_row)
            scales = np.asarray(self.scales)
            if scales.dtype != np.float64:
                raise QuantizedXIP2InverseError("candidate scale JSON values must decode as float")
            for value in self.scales:
                _require_finite_float(value, field="candidate scale")
                if value <= 0.0 or float(np.float32(value)) != value:
                    raise QuantizedXIP2InverseError("candidate scales escaped frozen fp32 values")
            _require_sha256(self.q_sha256, field="q_sha256")
            _require_sha256(self.scales_sha256, field="scales_sha256")
            if (
                _array_sha256(np.asarray(q_values, dtype=np.int16)) != self.q_sha256
                or _array_sha256(np.asarray(self.scales, dtype=np.float32)) != self.scales_sha256
            ):
                raise QuantizedXIP2InverseError("candidate q/scales hashes are inconsistent")
            reference_values = (
                self.reference_control_packet_sha256,
                self.reference_d_pose,
                self.reference_d_seg,
                self.reference_archive_bytes,
                self.delta_distortion,
                self.delta_s,
                self.admitted,
            )
            if any(value is None for value in reference_values):
                raise QuantizedXIP2InverseError("active candidate omitted control-relative objective fields")
            assert self.reference_control_packet_sha256 is not None
            assert self.reference_d_pose is not None
            assert self.reference_d_seg is not None
            assert self.reference_archive_bytes is not None
            assert self.delta_distortion is not None
            assert self.delta_s is not None
            assert self.admitted is not None
            _require_sha256(self.reference_control_packet_sha256, field="reference_control_packet_sha256")
            _require_finite_float(self.reference_d_pose, field="reference_d_pose", nonnegative=True)
            _require_finite_float(self.reference_d_seg, field="reference_d_seg", nonnegative=True)
            _require_exact_int(self.reference_archive_bytes, field="reference_archive_bytes", minimum=0)
            _require_finite_float(self.delta_distortion, field="delta_distortion")
            _require_finite_float(self.delta_s, field="delta_s")
            expected_distortion = (
                100.0 * (self.d_seg - self.reference_d_seg)
                + math.sqrt(10.0 * self.d_pose)
                - math.sqrt(10.0 * self.reference_d_pose)
            )
            expected_delta = (
                expected_distortion
                + RATE_MULTIPLIER * (self.archive_bytes - self.reference_archive_bytes) / RATE_DENOMINATOR_BYTES
            )
            expected_ceiling = quantized_xip2_strict_archive_ceiling(
                reference_archive_bytes=self.reference_archive_bytes,
                delta_distortion=expected_distortion,
            )
            if (
                self.delta_distortion != expected_distortion
                or self.delta_s != expected_delta
                or self.strict_archive_ceiling != expected_ceiling
                or self.admitted is not (expected_delta < 0.0)
                or self.admission_reason
                != ("DELTA_S_STRICTLY_NEGATIVE" if expected_delta < 0.0 else "DELTA_S_NOT_STRICTLY_NEGATIVE")
            ):
                raise QuantizedXIP2InverseError("candidate control-relative objective arithmetic changed")
        else:
            if self.mode not in _CONTROL_ORDER:
                raise QuantizedXIP2InverseError("candidate mode escaped G16")
            if (
                any(
                    value is not None
                    for value in (
                        self.arm_index,
                        self.arm_id,
                        self.q_level,
                        self.coder,
                        self.q_sha256,
                        self.scales_sha256,
                        self.reference_control_packet_sha256,
                        self.reference_d_pose,
                        self.reference_d_seg,
                        self.reference_archive_bytes,
                        self.delta_distortion,
                        self.delta_s,
                        self.strict_archive_ceiling,
                        self.admitted,
                    )
                )
                or self.q
                or self.scales
                or self.xip2_payload_bytes != 0
                or self.admission_reason != "CONTROL_EVALUATION"
            ):
                raise QuantizedXIP2InverseError("control candidate fabricated active/delta state")

    def as_dict(self) -> dict[str, Any]:
        return {
            "admission_reason": self.admission_reason,
            "admitted": self.admitted,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "arm_id": self.arm_id,
            "arm_index": self.arm_index,
            "authority": self.authority.value,
            "callback_evaluation_index": self.callback_evaluation_index,
            "coder": None if self.coder is None else self.coder.value,
            "complete_score": self.complete_score,
            "d_pose": self.d_pose,
            "d_seg": self.d_seg,
            "delta_distortion": self.delta_distortion,
            "delta_s": self.delta_s,
            "evaluation_id": self.evaluation_id,
            "frozen_scorer_sha256": self.frozen_scorer_sha256,
            "interpretation": self.interpretation.value,
            "lineage_sha256": self.lineage_sha256,
            "measurement_custody_sha256": self.measurement_custody_sha256,
            "mode": self.mode.value,
            "output_camera_y0_sha256": self.output_camera_y0_sha256,
            "output_camera_y1_sha256": self.output_camera_y1_sha256,
            "output_chronology_sha256": self.output_chronology_sha256,
            "packet_bytes": self.packet_bytes,
            "packet_sha256": self.packet_sha256,
            "per_pair_pose6_mse": list(self.per_pair_pose6_mse),
            "q": [list(row) for row in self.q],
            "q_level": self.q_level,
            "q_sha256": self.q_sha256,
            "reference_archive_bytes": self.reference_archive_bytes,
            "reference_control_packet_sha256": self.reference_control_packet_sha256,
            "reference_d_pose": self.reference_d_pose,
            "reference_d_seg": self.reference_d_seg,
            "scales": list(self.scales),
            "scales_sha256": self.scales_sha256,
            "schema": self.schema,
            "seg_prediction_sha256": self.seg_prediction_sha256,
            "source_binding_sha256": self.source_binding_sha256,
            "strict_archive_ceiling": self.strict_archive_ceiling,
            "target_custody_sha256": self.target_custody_sha256,
            "xip2_payload_bytes": self.xip2_payload_bytes,
            "xip2_payload_sha256": self.xip2_payload_sha256,
        }

    def to_canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def candidate_sha256(self) -> str:
        return _sha256(self.to_canonical_json_bytes())


@dataclass(frozen=True, slots=True)
class QuantizedXIP2ArmResultV1:
    arm: QuantizedXIP2ArmV1
    initial_candidate: QuantizedXIP2CandidateV1
    final_candidate: QuantizedXIP2CandidateV1
    best_candidate: QuantizedXIP2CandidateV1
    evaluated_neighbors: int
    search_completed: bool
    schema: Literal["tac.quantized_xip2_inverse_search_arm_result.v1"] = ARM_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ARM_RESULT_SCHEMA or type(self.arm) is not QuantizedXIP2ArmV1:
            raise QuantizedXIP2InverseError("arm-result schema/type changed")
        for candidate in (self.initial_candidate, self.final_candidate, self.best_candidate):
            if type(candidate) is not QuantizedXIP2CandidateV1 or candidate.arm_id != self.arm.arm_id:
                raise QuantizedXIP2InverseError("arm result candidate differs from its arm")
        _require_exact_int(self.evaluated_neighbors, field="evaluated_neighbors", minimum=0)
        if type(self.search_completed) is not bool:
            raise QuantizedXIP2InverseError("arm search_completed must be exact bool")
        if (self.best_candidate.complete_score, self.best_candidate.packet_sha256) > (
            self.initial_candidate.complete_score,
            self.initial_candidate.packet_sha256,
        ):
            raise QuantizedXIP2InverseError("arm best candidate is worse than its initializer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.as_dict(),
            "best_candidate": self.best_candidate.as_dict(),
            "evaluated_neighbors": self.evaluated_neighbors,
            "final_candidate": self.final_candidate.as_dict(),
            "initial_candidate": self.initial_candidate.as_dict(),
            "schema": self.schema,
            "search_completed": self.search_completed,
        }

    def to_canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class QuantizedXIP2SearchReceiptV1:
    lineage_sha256: str
    config_sha256: str
    implementation_sha256: str
    source_binding_sha256: str
    initial_xi_sha256: str
    callback_authority: QuantizedXIP2ObjectiveAuthorityV1
    frozen_scorer_sha256: str
    target_custody_sha256: str
    measurement_custody_sha256: str
    callback_evaluations: int
    evaluated_active_packets: int
    bounded_search_completed: bool
    bounded_search_negative: bool
    through_g13_uint8_receiver: bool
    verdict_scope: str = FORMULATION_SCOPE
    integration_blocker: str = ROOT_INTEGRATION_BLOCKER
    research_only: Literal[True] = True
    callback_external: Literal[True] = True
    direct_quantized_receiver_search: Literal[True] = True
    affine_xi_pose_model_used: Literal[False] = False
    public_or_historical_payload_used: Literal[False] = False
    dense_target_serialized: Literal[False] = False
    scorer_or_weights_serialized: Literal[False] = False
    actual_scorer_invoked_by_core: Literal[False] = False
    n2_claim: Literal[False] = False
    n600_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    schema: Literal["tac.quantized_xip2_inverse_search_receipt.v1"] = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise QuantizedXIP2InverseError("search receipt schema changed")
        for name in (
            "lineage_sha256",
            "config_sha256",
            "implementation_sha256",
            "source_binding_sha256",
            "initial_xi_sha256",
            "frozen_scorer_sha256",
            "target_custody_sha256",
            "measurement_custody_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)
        if type(self.callback_authority) is not QuantizedXIP2ObjectiveAuthorityV1:
            raise QuantizedXIP2InverseError("receipt authority escaped callback authority")
        _require_exact_int(self.callback_evaluations, field="callback_evaluations", minimum=2)
        _require_exact_int(self.evaluated_active_packets, field="evaluated_active_packets", minimum=1)
        if type(self.bounded_search_completed) is not bool or type(self.bounded_search_negative) is not bool:
            raise QuantizedXIP2InverseError("receipt search verdicts must be exact bools")
        if self.through_g13_uint8_receiver is not True or self.verdict_scope != FORMULATION_SCOPE:
            raise QuantizedXIP2InverseError("receipt weakened G13/formulation scope")
        if self.integration_blocker != ROOT_INTEGRATION_BLOCKER:
            raise QuantizedXIP2InverseError("receipt changed the root integration blocker")
        expected_truth = {
            "research_only": True,
            "callback_external": True,
            "direct_quantized_receiver_search": True,
            "affine_xi_pose_model_used": False,
            "public_or_historical_payload_used": False,
            "dense_target_serialized": False,
            "scorer_or_weights_serialized": False,
            "actual_scorer_invoked_by_core": False,
            "n2_claim": False,
            "n600_claim": False,
            "exact_score_claim": False,
            "candidate_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
        }
        if any(
            type(getattr(self, name)) is not type(expected) or getattr(self, name) != expected
            for name, expected in expected_truth.items()
        ):
            raise QuantizedXIP2InverseError("receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "actual_scorer_invoked_by_core": self.actual_scorer_invoked_by_core,
            "affine_xi_pose_model_used": self.affine_xi_pose_model_used,
            "bounded_search_completed": self.bounded_search_completed,
            "bounded_search_negative": self.bounded_search_negative,
            "callback_authority": self.callback_authority.value,
            "callback_evaluations": self.callback_evaluations,
            "callback_external": self.callback_external,
            "candidate_claim": self.candidate_claim,
            "config_sha256": self.config_sha256,
            "dense_target_serialized": self.dense_target_serialized,
            "direct_quantized_receiver_search": self.direct_quantized_receiver_search,
            "evaluated_active_packets": self.evaluated_active_packets,
            "exact_score_claim": self.exact_score_claim,
            "frozen_scorer_sha256": self.frozen_scorer_sha256,
            "implementation_sha256": self.implementation_sha256,
            "initial_xi_sha256": self.initial_xi_sha256,
            "integration_blocker": self.integration_blocker,
            "lineage_sha256": self.lineage_sha256,
            "measurement_custody_sha256": self.measurement_custody_sha256,
            "n2_claim": self.n2_claim,
            "n600_claim": self.n600_claim,
            "originality_claim": self.originality_claim,
            "promotion_eligible": self.promotion_eligible,
            "public_or_historical_payload_used": self.public_or_historical_payload_used,
            "research_only": self.research_only,
            "schema": self.schema,
            "scorer_or_weights_serialized": self.scorer_or_weights_serialized,
            "source_binding_sha256": self.source_binding_sha256,
            "target_custody_sha256": self.target_custody_sha256,
            "through_g13_uint8_receiver": self.through_g13_uint8_receiver,
            "verdict_scope": self.verdict_scope,
        }

    def to_canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_canonical_json_bytes())


@dataclass(frozen=True, slots=True)
class QuantizedXIP2SearchResultV1:
    lineage_sha256: str
    selected_control: QuantizedXIP2CandidateV1
    best_active_candidate: QuantizedXIP2CandidateV1
    arm_results: tuple[QuantizedXIP2ArmResultV1, ...]
    improved_control: bool
    bounded_search_completed: bool
    stop_reason: str
    receipt: QuantizedXIP2SearchReceiptV1
    schema: Literal["tac.quantized_xip2_inverse_search_result.v1"] = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise QuantizedXIP2InverseError("search result schema changed")
        _require_sha256(self.lineage_sha256, field="lineage_sha256")
        if (
            type(self.selected_control) is not QuantizedXIP2CandidateV1
            or self.selected_control.mode not in _CONTROL_ORDER
            or type(self.best_active_candidate) is not QuantizedXIP2CandidateV1
            or self.best_active_candidate.mode is not CountedXIP2A3Mode.XIP2_WARP_V1
        ):
            raise QuantizedXIP2InverseError("result control/active selection changed")
        if type(self.arm_results) is not tuple or not self.arm_results:
            raise QuantizedXIP2InverseError("result must contain every configured arm")
        if any(type(value) is not QuantizedXIP2ArmResultV1 for value in self.arm_results):
            raise QuantizedXIP2InverseError("result arm record type changed")
        if type(self.improved_control) is not bool or type(self.bounded_search_completed) is not bool:
            raise QuantizedXIP2InverseError("result verdicts must be exact bools")
        if self.improved_control is not (self.best_active_candidate.delta_s < 0.0):
            raise QuantizedXIP2InverseError("result admission differs from strict finite delta_S")
        expected_stop = "SCHEDULE_COMPLETED" if self.bounded_search_completed else "CALLBACK_BUDGET_EXHAUSTED"
        if self.stop_reason != expected_stop:
            raise QuantizedXIP2InverseError("result stop reason differs from bounded search state")
        if type(self.receipt) is not QuantizedXIP2SearchReceiptV1:
            raise QuantizedXIP2InverseError("result omitted exact receipt")
        if (
            self.receipt.lineage_sha256 != self.lineage_sha256
            or self.receipt.bounded_search_completed is not self.bounded_search_completed
            or self.receipt.bounded_search_negative is not (not self.improved_control)
        ):
            raise QuantizedXIP2InverseError("result and receipt differ")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_results": [value.as_dict() for value in self.arm_results],
            "best_active_candidate": self.best_active_candidate.as_dict(),
            "bounded_search_completed": self.bounded_search_completed,
            "improved_control": self.improved_control,
            "lineage_sha256": self.lineage_sha256,
            "receipt": self.receipt.as_dict(),
            "schema": self.schema,
            "selected_control": self.selected_control.as_dict(),
            "stop_reason": self.stop_reason,
        }

    def to_canonical_json_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def result_sha256(self) -> str:
        return _sha256(self.to_canonical_json_bytes())


def _candidate_from_dict(value: object) -> QuantizedXIP2CandidateV1:
    if type(value) is not dict:
        raise QuantizedXIP2InverseError("evaluation stage candidate must be one JSON object")
    raw = value
    list_fields = ("q", "scales", "per_pair_pose6_mse")
    if any(type(raw.get(name)) is not list for name in list_fields):
        raise QuantizedXIP2InverseError("evaluation candidate tuple fields must be JSON lists")
    q_raw = raw["q"]
    if any(type(row) is not list for row in q_raw):
        raise QuantizedXIP2InverseError("evaluation candidate q rows must be JSON lists")
    try:
        candidate = QuantizedXIP2CandidateV1(
            callback_evaluation_index=raw["callback_evaluation_index"],
            evaluation_id=raw["evaluation_id"],
            mode=CountedXIP2A3Mode(raw["mode"]),
            arm_index=raw["arm_index"],
            arm_id=raw["arm_id"],
            q_level=raw["q_level"],
            coder=None if raw["coder"] is None else XIP2Coder(raw["coder"]),
            interpretation=CountedXIP2A3Interpretation(raw["interpretation"]),
            q=tuple(tuple(row) for row in q_raw),
            scales=tuple(raw["scales"]),
            source_binding_sha256=raw["source_binding_sha256"],
            lineage_sha256=raw["lineage_sha256"],
            packet_sha256=raw["packet_sha256"],
            packet_bytes=raw["packet_bytes"],
            q_sha256=raw["q_sha256"],
            scales_sha256=raw["scales_sha256"],
            xip2_payload_sha256=raw["xip2_payload_sha256"],
            xip2_payload_bytes=raw["xip2_payload_bytes"],
            output_camera_y0_sha256=raw["output_camera_y0_sha256"],
            output_camera_y1_sha256=raw["output_camera_y1_sha256"],
            output_chronology_sha256=raw["output_chronology_sha256"],
            per_pair_pose6_mse=tuple(raw["per_pair_pose6_mse"]),
            d_pose=raw["d_pose"],
            d_seg=raw["d_seg"],
            seg_prediction_sha256=raw["seg_prediction_sha256"],
            archive_bytes=raw["archive_bytes"],
            archive_sha256=raw["archive_sha256"],
            authority=QuantizedXIP2ObjectiveAuthorityV1(raw["authority"]),
            frozen_scorer_sha256=raw["frozen_scorer_sha256"],
            target_custody_sha256=raw["target_custody_sha256"],
            measurement_custody_sha256=raw["measurement_custody_sha256"],
            complete_score=raw["complete_score"],
            reference_control_packet_sha256=raw["reference_control_packet_sha256"],
            reference_d_pose=raw["reference_d_pose"],
            reference_d_seg=raw["reference_d_seg"],
            reference_archive_bytes=raw["reference_archive_bytes"],
            delta_distortion=raw["delta_distortion"],
            delta_s=raw["delta_s"],
            strict_archive_ceiling=raw["strict_archive_ceiling"],
            admitted=raw["admitted"],
            admission_reason=raw["admission_reason"],
            schema=raw["schema"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QuantizedXIP2InverseError("evaluation candidate shape/discriminator is malformed") from exc
    if _canonical_json(candidate.as_dict()) != _canonical_json(raw):
        raise QuantizedXIP2InverseError("evaluation candidate fields differ from the exact V1 schema")
    return candidate


def _observation_fingerprint(candidate: QuantizedXIP2CandidateV1) -> bytes:
    return _canonical_json(
        {
            "archive_bytes": candidate.archive_bytes,
            "archive_sha256": candidate.archive_sha256,
            "authority": candidate.authority.value,
            "d_seg": candidate.d_seg,
            "frozen_scorer_sha256": candidate.frozen_scorer_sha256,
            "measurement_custody_sha256": candidate.measurement_custody_sha256,
            "per_pair_pose6_mse": list(candidate.per_pair_pose6_mse),
            "seg_prediction_sha256": candidate.seg_prediction_sha256,
            "target_custody_sha256": candidate.target_custody_sha256,
        }
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise QuantizedXIP2InverseError(f"{label} must be one regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise QuantizedXIP2InverseError(f"could not read {label}") from exc


def _atomic_publish(path: Path, payload: bytes) -> None:
    """Publish bytes without overwriting evidence; identical existing bytes are accepted."""

    if type(payload) is not bytes:
        raise QuantizedXIP2InverseError("atomic payload must be exact bytes")
    if path.exists() or path.is_symlink():
        if _read_regular_file(path, label=path.name) != payload:
            raise QuantizedXIP2InverseError(f"existing {path.name} is not byte-identical")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise QuantizedXIP2InverseError(f"unexpected partial file {temporary.name}")
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, path)
            published = True
        except FileExistsError:
            if _read_regular_file(path, label=path.name) != payload:
                raise QuantizedXIP2InverseError(f"concurrent {path.name} differs from exact bytes") from None
        _fsync_directory(path.parent)
        os.unlink(temporary)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise QuantizedXIP2InverseError(f"atomic publish failed for {path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published and temporary.exists() and path.exists():
            # Only the successfully published/identical-target case may remove its own temp.
            try:
                if _read_regular_file(path, label=path.name) == payload:
                    os.unlink(temporary)
                    _fsync_directory(path.parent)
            except (OSError, QuantizedXIP2InverseError):
                pass


@dataclass(slots=True)
class _StageStore:
    run_dir: Path
    manifest_sha256: str
    candidates_by_packet: dict[str, QuantizedXIP2CandidateV1]
    evaluation_candidates: list[QuantizedXIP2CandidateV1]
    stage_sha256s: list[str]
    terminal_payload: bytes | None
    terminal_index: int | None
    custody: tuple[QuantizedXIP2ObjectiveAuthorityV1, str, str, str] | None
    replay_next_index: int = 1
    consumed_packets: set[str] | None = None

    def __post_init__(self) -> None:
        if self.consumed_packets is None:
            self.consumed_packets = set()

    @property
    def callback_count(self) -> int:
        return len(self.evaluation_candidates)

    @property
    def previous_stage_sha256(self) -> str:
        if not self.stage_sha256s:
            return self.manifest_sha256
        if self.has_terminal:
            return self.stage_sha256s[-2] if len(self.stage_sha256s) > 1 else self.manifest_sha256
        return self.stage_sha256s[-1]

    @property
    def has_terminal(self) -> bool:
        return self.terminal_payload is not None

    def consume_cached(self, candidate: QuantizedXIP2CandidateV1) -> None:
        assert self.consumed_packets is not None
        if candidate.packet_sha256 in self.consumed_packets:
            return
        if candidate.callback_evaluation_index != self.replay_next_index:
            raise QuantizedXIP2InverseError("evaluation stage order differs from deterministic replay")
        self.consumed_packets.add(candidate.packet_sha256)
        self.replay_next_index += 1

    def append_evaluation(self, candidate: QuantizedXIP2CandidateV1) -> None:
        expected_index = self.callback_count + 1
        if candidate.callback_evaluation_index != expected_index or expected_index != self.replay_next_index:
            raise QuantizedXIP2InverseError("new callback stage is not the next replay/chain index")
        if expected_index > 99_999_999:
            raise QuantizedXIP2InverseError("evaluation stage index escaped the eight-digit V1 ABI")
        record = {
            "candidate": candidate.as_dict(),
            "manifest_sha256": self.manifest_sha256,
            "previous_stage_sha256": self.previous_stage_sha256,
            "schema": EVALUATION_STAGE_SCHEMA,
            "stage_index": expected_index,
            "stage_kind": "evaluation",
        }
        payload = _canonical_json(record)
        path = self.run_dir / f"stage-{expected_index:08d}-evaluation.json"
        _atomic_publish(path, payload)
        self.evaluation_candidates.append(candidate)
        self.candidates_by_packet[candidate.packet_sha256] = candidate
        self.stage_sha256s.append(_sha256(payload))
        assert self.consumed_packets is not None
        self.consumed_packets.add(candidate.packet_sha256)
        self.replay_next_index += 1


def _load_stage_store(run_dir: Path, *, manifest_payload: bytes) -> _StageStore:
    manifest_sha = _sha256(manifest_payload)
    indexed: dict[int, tuple[str, Path]] = {}
    try:
        entries = tuple(run_dir.iterdir())
    except OSError as exc:
        raise QuantizedXIP2InverseError("could not enumerate the durable run directory") from exc
    for entry in entries:
        if entry.name == "manifest.json":
            continue
        match = _STAGE_RE.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_file():
            raise QuantizedXIP2InverseError(f"unexpected temporary/partial/run file: {entry.name}")
        index = int(match.group(1))
        if index < 1 or index in indexed:
            raise QuantizedXIP2InverseError("duplicate or invalid stage index")
        indexed[index] = (match.group(2), entry)
    if indexed and tuple(sorted(indexed)) != tuple(range(1, len(indexed) + 1)):
        raise QuantizedXIP2InverseError("durable stage sequence contains a gap")
    candidates: list[QuantizedXIP2CandidateV1] = []
    by_packet: dict[str, QuantizedXIP2CandidateV1] = {}
    stage_hashes: list[str] = []
    terminal_payload: bytes | None = None
    terminal_index: int | None = None
    custody: tuple[QuantizedXIP2ObjectiveAuthorityV1, str, str, str] | None = None
    expected_previous = manifest_sha
    for index in sorted(indexed):
        kind, path = indexed[index]
        payload = _read_regular_file(path, label=path.name)
        record = _decode_canonical_json(payload, label=path.name)
        common = {
            "manifest_sha256",
            "previous_stage_sha256",
            "schema",
            "stage_index",
            "stage_kind",
        }
        expected_keys = common | ({"candidate"} if kind == "evaluation" else {"result"})
        if set(record) != expected_keys:
            raise QuantizedXIP2InverseError(f"{path.name} fields differ from exact V1 stage schema")
        if (
            record["stage_index"] != index
            or record["stage_kind"] != kind
            or record["manifest_sha256"] != manifest_sha
            or record["previous_stage_sha256"] != expected_previous
        ):
            raise QuantizedXIP2InverseError("durable stage chain/index/manifest binding broke")
        expected_schema = EVALUATION_STAGE_SCHEMA if kind == "evaluation" else TERMINAL_STAGE_SCHEMA
        if record["schema"] != expected_schema:
            raise QuantizedXIP2InverseError("durable stage schema changed")
        if kind == "terminal":
            if index != len(indexed) or terminal_payload is not None:
                raise QuantizedXIP2InverseError("terminal stage must be unique and last")
            terminal_payload = payload
            terminal_index = index
        else:
            if terminal_payload is not None:
                raise QuantizedXIP2InverseError("evaluation stage appears after terminal")
            candidate = _candidate_from_dict(record["candidate"])
            if candidate.callback_evaluation_index != index:
                raise QuantizedXIP2InverseError("candidate callback index differs from stage index")
            observed_custody = (
                candidate.authority,
                candidate.frozen_scorer_sha256,
                candidate.target_custody_sha256,
                candidate.measurement_custody_sha256,
            )
            if custody is None:
                custody = observed_custody
            elif custody != observed_custody:
                raise QuantizedXIP2InverseError("callback authority/custody changed inside stage chain")
            prior = by_packet.get(candidate.packet_sha256)
            if prior is not None and _observation_fingerprint(prior) != _observation_fingerprint(candidate):
                raise QuantizedXIP2InverseError("duplicate packet has differing callback observations")
            if prior is None:
                by_packet[candidate.packet_sha256] = candidate
            candidates.append(candidate)
        stage_digest = _sha256(payload)
        stage_hashes.append(stage_digest)
        expected_previous = stage_digest
    return _StageStore(
        run_dir=run_dir,
        manifest_sha256=manifest_sha,
        candidates_by_packet=by_packet,
        evaluation_candidates=candidates,
        stage_sha256s=stage_hashes,
        terminal_payload=terminal_payload,
        terminal_index=terminal_index,
        custody=custody,
    )


def _fresh_initial_xi(initial_xi: np.ndarray, *, pair_count: int) -> np.ndarray:
    if type(initial_xi) is not np.ndarray or initial_xi.dtype.kind != "f":
        raise QuantizedXIP2InverseError("initial_xi must be a fresh floating NumPy array")
    if initial_xi.shape != (pair_count, 6) or not np.all(np.isfinite(initial_xi)):
        raise QuantizedXIP2InverseError("initial_xi must be finite with exact live shape [pair,6]")
    return _immutable_array(initial_xi, dtype=np.dtype(np.float64))


def _manifest_record(
    *,
    config: QuantizedXIP2SearchConfigV1,
    implementation_sha256: str,
    source: PassConditionalASourceBindingV1,
    initial_xi_sha256: str,
    lineage_sha256: str,
) -> dict[str, Any]:
    return {
        "allowed_callback_authorities": [value.value for value in QuantizedXIP2ObjectiveAuthorityV1],
        "callback_custody_fields": [
            "frozen_scorer_sha256",
            "target_custody_sha256",
            "measurement_custody_sha256",
        ],
        "config": config.as_dict(),
        "config_sha256": config.config_sha256,
        "conditional_camera_y1_sha256": source.conditional_camera_y1_sha256,
        "implementation_sha256": implementation_sha256,
        "initial_xi_dtype": "float64-canonical",
        "initial_xi_sha256": initial_xi_sha256,
        "initial_xi_shape": [source.pair_count, 6],
        "lineage_sha256": lineage_sha256,
        "research_only": True,
        "schema": MANIFEST_SCHEMA,
        "seed": config.seed,
        "source_binding_sha256": source.binding_sha256,
        "source_pair_ids": list(source.source_pair_ids),
    }


def _prepare_run_directory(
    *,
    run_dir: Path,
    resume_from: Path | None,
    manifest_payload: bytes,
) -> _StageStore:
    if not isinstance(run_dir, Path) or not run_dir.is_absolute():
        raise QuantizedXIP2InverseError("run_dir must be one explicit absolute pathlib.Path")
    if resume_from is not None and (not isinstance(resume_from, Path) or resume_from != run_dir):
        raise QuantizedXIP2InverseError("resume_from must be absent or exactly equal run_dir")
    if resume_from is None:
        if run_dir.exists() or run_dir.is_symlink():
            raise QuantizedXIP2InverseError("new run requires an absent run_dir")
        if not run_dir.parent.is_dir() or run_dir.parent.is_symlink():
            raise QuantizedXIP2InverseError("run_dir parent must be one existing durable directory")
        try:
            run_dir.mkdir(mode=0o700)
            _fsync_directory(run_dir.parent)
        except OSError as exc:
            raise QuantizedXIP2InverseError("could not create durable run_dir") from exc
        _atomic_publish(run_dir / "manifest.json", manifest_payload)
    else:
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise QuantizedXIP2InverseError("resume requires the exact existing non-symlink run_dir")
        existing = _read_regular_file(run_dir / "manifest.json", label="manifest.json")
        _decode_canonical_json(existing, label="manifest.json")
        if existing != manifest_payload:
            raise QuantizedXIP2InverseError("resume manifest differs in config/source/initializer/implementation")
    manifest_on_disk = _read_regular_file(run_dir / "manifest.json", label="manifest.json")
    if manifest_on_disk != manifest_payload:
        raise QuantizedXIP2InverseError("manifest bytes differ after atomic publication")
    return _load_stage_store(run_dir, manifest_payload=manifest_payload)


def _lineage_sha256(
    *,
    source_binding_sha256: str,
    initial_xi_sha256: str,
    config_sha256: str,
    implementation_sha256: str,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "config_sha256": config_sha256,
                "implementation_sha256": implementation_sha256,
                "initial_xi_sha256": initial_xi_sha256,
                "schema": "tac.quantized_xip2_inverse_lineage.v1",
                "source_binding_sha256": source_binding_sha256,
            }
        )
    )


def _evaluation_id(
    *,
    lineage_sha256: str,
    mode: CountedXIP2A3Mode,
    arm_id: str | None,
    packet_sha256: str,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "arm_id": arm_id,
                "lineage_sha256": lineage_sha256,
                "mode": mode.value,
                "packet_sha256": packet_sha256,
                "schema": "tac.quantized_xip2_inverse_evaluation_identity.v1",
            }
        )
    )


def _request_from_compiled(
    *,
    compiled: CompiledCountedXIP2ChronologicalA3V1,
    arm_index: int | None,
    arm_id: str | None,
    q_level: int | None,
    coder: XIP2Coder | None,
    lineage_sha256: str,
    config_sha256: str,
    initial_xi_sha256: str,
) -> QuantizedXIP2EvaluationRequestV1:
    transport = compiled.parsed_packet.transport
    q = None if transport is None else transport.q_codes
    scales = None if transport is None else transport.scales
    receipt = compiled.receipt
    return QuantizedXIP2EvaluationRequestV1(
        compiled=compiled,
        evaluation_id=_evaluation_id(
            lineage_sha256=lineage_sha256,
            mode=compiled.program.mode,
            arm_id=arm_id,
            packet_sha256=receipt.packet_sha256,
        ),
        arm_index=arm_index,
        arm_id=arm_id,
        q_level=q_level,
        coder=coder,
        interpretation=compiled.program.interpretation,
        q=q,
        scales=scales,
        source_binding_sha256=compiled.source_binding.binding_sha256,
        lineage_sha256=lineage_sha256,
        config_sha256=config_sha256,
        initial_xi_sha256=initial_xi_sha256,
        packet_sha256=receipt.packet_sha256,
        q_sha256=None if q is None else _array_sha256(q),
        scales_sha256=None if scales is None else _array_sha256(scales),
        xip2_payload_sha256=receipt.xip2_payload_sha256,
        output_camera_y0_sha256=receipt.output_camera_y0_sha256,
        output_camera_y1_sha256=receipt.output_camera_y1_sha256,
        output_chronology_sha256=receipt.output_chronology_sha256,
    )


def _compile_control_request(
    *,
    mode: CountedXIP2A3Mode,
    surface: PassConditionalASurfaceV1,
    lineage_sha256: str,
    config_sha256: str,
    initial_xi_sha256: str,
) -> QuantizedXIP2EvaluationRequestV1:
    try:
        compiled = compile_counted_xip2_chronological_a3_pass(surface=surface, mode=mode)
    except CountedXIP2ChronologicalA3Error as exc:
        raise QuantizedXIP2InverseError("G13 refused a canonical empty-body control") from exc
    return _request_from_compiled(
        compiled=compiled,
        arm_index=None,
        arm_id=None,
        q_level=None,
        coder=None,
        lineage_sha256=lineage_sha256,
        config_sha256=config_sha256,
        initial_xi_sha256=initial_xi_sha256,
    )


def _compile_active_request(
    *,
    q: np.ndarray,
    scales: np.ndarray,
    arm_index: int,
    arm_id: str,
    q_level: int,
    coder: XIP2Coder,
    interpretation: CountedXIP2A3Interpretation,
    pitch: float,
    surface: PassConditionalASurfaceV1,
    lineage_sha256: str,
    config_sha256: str,
    initial_xi_sha256: str,
) -> QuantizedXIP2EvaluationRequestV1:
    q_input = np.asarray(q)
    scales_input = np.asarray(scales)
    if (
        q_input.dtype != np.int16
        or q_input.shape != (len(surface.source_pair_ids), 6)
        or scales_input.dtype != np.float32
        or scales_input.shape != (6,)
        or not np.all(np.isfinite(scales_input))
        or np.any(scales_input <= 0.0)
        or int(np.abs(q_input.astype(np.int32)).max(initial=0)) > q_level
    ):
        raise QuantizedXIP2InverseError("active arm q/scales escaped its fixed decoded lattice")
    serializer_kwargs: dict[str, Any] = {"coder": coder.value}
    if coder is XIP2Coder.SPLINE_RESIDUAL:
        if q_input.shape[0] < 3:
            raise QuantizedXIP2InverseError("spline_residual requires at least three live pairs for its cubic path")
        serializer_kwargs["spline_knots"] = min(16, q_input.shape[0] + 1)
    try:
        payload = serialize_xi_payload(q_input, scales_input, **serializer_kwargs)
        parsed_q, parsed_scales = parse_xi_payload(payload)
        reencoded = serialize_xi_payload(parsed_q, parsed_scales, **serializer_kwargs)
    except (XiPoseCoderError, ValueError, struct.error) as exc:
        raise QuantizedXIP2InverseError("XIP2 serialization/strict round trip failed") from exc
    if not np.array_equal(parsed_q, q_input) or not np.array_equal(parsed_scales, scales_input) or reencoded != payload:
        raise QuantizedXIP2InverseError("XIP2 q/scales changed on exact parse/re-encode")
    program = CountedXIP2ChronologicalA3ProgramV1(
        mode=CountedXIP2A3Mode.XIP2_WARP_V1,
        interpretation=interpretation,
        pitch=pitch,
        xip2_payload=payload,
        guidance_source_binding_sha256=lineage_sha256,
    )
    try:
        compiled = compile_counted_xip2_chronological_a3(program, surface=surface)
    except CountedXIP2ChronologicalA3Error as exc:
        raise QuantizedXIP2InverseError("active packet failed the real G13 uint8 receiver chain") from exc
    transport = compiled.parsed_packet.transport
    if (
        transport is None
        or not np.array_equal(transport.q_codes, q_input)
        or not np.array_equal(transport.scales, scales_input)
        or compiled.program.xip2_payload != payload
    ):
        raise QuantizedXIP2InverseError("G13 decoded q/scales differ from the serialized search state")
    return _request_from_compiled(
        compiled=compiled,
        arm_index=arm_index,
        arm_id=arm_id,
        q_level=q_level,
        coder=coder,
        lineage_sha256=lineage_sha256,
        config_sha256=config_sha256,
        initial_xi_sha256=initial_xi_sha256,
    )


def _candidate_matches_request(
    candidate: QuantizedXIP2CandidateV1,
    request: QuantizedXIP2EvaluationRequestV1,
) -> bool:
    q = () if request.q is None else tuple(tuple(int(value) for value in row) for row in request.q)
    scales = () if request.scales is None else tuple(float(value) for value in request.scales)
    receipt = request.compiled.receipt
    return (
        candidate.evaluation_id == request.evaluation_id
        and candidate.mode is request.mode
        and candidate.arm_index == request.arm_index
        and candidate.arm_id == request.arm_id
        and candidate.q_level == request.q_level
        and candidate.coder is request.coder
        and candidate.interpretation is request.interpretation
        and candidate.q == q
        and candidate.scales == scales
        and candidate.source_binding_sha256 == request.source_binding_sha256
        and candidate.lineage_sha256 == request.lineage_sha256
        and candidate.packet_sha256 == request.packet_sha256
        and candidate.packet_bytes == receipt.packet_bytes
        and candidate.q_sha256 == request.q_sha256
        and candidate.scales_sha256 == request.scales_sha256
        and candidate.xip2_payload_sha256 == request.xip2_payload_sha256
        and candidate.xip2_payload_bytes == receipt.xip2_payload_bytes
        and candidate.output_camera_y0_sha256 == request.output_camera_y0_sha256
        and candidate.output_camera_y1_sha256 == request.output_camera_y1_sha256
        and candidate.output_chronology_sha256 == request.output_chronology_sha256
    )


def _candidate_from_observation(
    *,
    request: QuantizedXIP2EvaluationRequestV1,
    observation: QuantizedXIP2ObjectiveObservationV1,
    callback_evaluation_index: int,
    reference_control: QuantizedXIP2CandidateV1 | None,
) -> QuantizedXIP2CandidateV1:
    d_pose = math.fsum(observation.per_pair_pose6_mse) / len(observation.per_pair_pose6_mse)
    complete_score = quantized_xip2_complete_score(
        d_seg=observation.d_seg,
        d_pose=d_pose,
        archive_bytes=observation.archive_bytes,
    )
    if reference_control is None:
        reference_packet = None
        reference_d_pose = None
        reference_d_seg = None
        reference_bytes = None
        delta_distortion = None
        delta_s = None
        ceiling = None
        admitted = None
        reason = "CONTROL_EVALUATION"
    else:
        reference_packet = reference_control.packet_sha256
        reference_d_pose = reference_control.d_pose
        reference_d_seg = reference_control.d_seg
        reference_bytes = reference_control.archive_bytes
        delta_distortion = (
            100.0 * (observation.d_seg - reference_d_seg)
            + math.sqrt(10.0 * d_pose)
            - math.sqrt(10.0 * reference_d_pose)
        )
        delta_s = (
            delta_distortion + RATE_MULTIPLIER * (observation.archive_bytes - reference_bytes) / RATE_DENOMINATOR_BYTES
        )
        ceiling = quantized_xip2_strict_archive_ceiling(
            reference_archive_bytes=reference_bytes,
            delta_distortion=delta_distortion,
        )
        admitted = delta_s < 0.0
        reason = "DELTA_S_STRICTLY_NEGATIVE" if admitted else "DELTA_S_NOT_STRICTLY_NEGATIVE"
    q = () if request.q is None else tuple(tuple(int(value) for value in row) for row in request.q)
    scales = () if request.scales is None else tuple(float(value) for value in request.scales)
    receipt = request.compiled.receipt
    return QuantizedXIP2CandidateV1(
        callback_evaluation_index=callback_evaluation_index,
        evaluation_id=request.evaluation_id,
        mode=request.mode,
        arm_index=request.arm_index,
        arm_id=request.arm_id,
        q_level=request.q_level,
        coder=request.coder,
        interpretation=request.interpretation,
        q=q,
        scales=scales,
        source_binding_sha256=request.source_binding_sha256,
        lineage_sha256=request.lineage_sha256,
        packet_sha256=request.packet_sha256,
        packet_bytes=receipt.packet_bytes,
        q_sha256=request.q_sha256,
        scales_sha256=request.scales_sha256,
        xip2_payload_sha256=request.xip2_payload_sha256,
        xip2_payload_bytes=receipt.xip2_payload_bytes,
        output_camera_y0_sha256=request.output_camera_y0_sha256,
        output_camera_y1_sha256=request.output_camera_y1_sha256,
        output_chronology_sha256=request.output_chronology_sha256,
        per_pair_pose6_mse=observation.per_pair_pose6_mse,
        d_pose=d_pose,
        d_seg=observation.d_seg,
        seg_prediction_sha256=observation.seg_prediction_sha256,
        archive_bytes=observation.archive_bytes,
        archive_sha256=observation.archive_sha256,
        authority=observation.authority,
        frozen_scorer_sha256=observation.frozen_scorer_sha256,
        target_custody_sha256=observation.target_custody_sha256,
        measurement_custody_sha256=observation.measurement_custody_sha256,
        complete_score=complete_score,
        reference_control_packet_sha256=reference_packet,
        reference_d_pose=reference_d_pose,
        reference_d_seg=reference_d_seg,
        reference_archive_bytes=reference_bytes,
        delta_distortion=delta_distortion,
        delta_s=delta_s,
        strict_archive_ceiling=ceiling,
        admitted=admitted,
        admission_reason=reason,
    )


def _evaluate(
    *,
    request: QuantizedXIP2EvaluationRequestV1,
    objective: QuantizedXIP2ObjectiveCallbackV1,
    store: _StageStore,
    config: QuantizedXIP2SearchConfigV1,
    source: PassConditionalASourceBindingV1,
    reference_control: QuantizedXIP2CandidateV1 | None,
) -> QuantizedXIP2CandidateV1:
    receipt = request.compiled.receipt
    if (
        request.source_binding_sha256 != source.binding_sha256
        or request.output_camera_y1_sha256 != source.conditional_camera_y1_sha256
        or receipt.output_camera_y1_sha256 != source.conditional_camera_y1_sha256
        or request.compiled.source_binding != source
    ):
        raise QuantizedXIP2InverseError("G13 source/decoded conditional Y1 changed before callback")
    cached = store.candidates_by_packet.get(request.packet_sha256)
    if cached is not None:
        if not _candidate_matches_request(cached, request):
            raise QuantizedXIP2InverseError("cached packet metadata differs from deterministic G13 replay")
        if reference_control is None:
            if cached.mode not in _CONTROL_ORDER:
                raise QuantizedXIP2InverseError("cached active candidate replayed without control")
        else:
            if (
                cached.reference_control_packet_sha256 != reference_control.packet_sha256
                or cached.reference_d_pose != reference_control.d_pose
                or cached.reference_d_seg != reference_control.d_seg
                or cached.reference_archive_bytes != reference_control.archive_bytes
            ):
                raise QuantizedXIP2InverseError("cached candidate differs from selected control")
        store.consume_cached(cached)
        return cached
    if store.has_terminal:
        raise QuantizedXIP2InverseError("terminal chain is missing a deterministic cached evaluation")
    if store.callback_count >= config.max_callback_evaluations:
        raise QuantizedXIP2InverseError("internal callback budget overrun")
    try:
        observation = objective(request)
    except Exception as exc:
        raise QuantizedXIP2InverseError("objective callback raised before a complete observation") from exc
    if type(observation) is not QuantizedXIP2ObjectiveObservationV1:
        raise QuantizedXIP2InverseError("objective callback must return exact QuantizedXIP2ObjectiveObservationV1")
    observation.__post_init__()
    if len(observation.per_pair_pose6_mse) != source.pair_count:
        raise QuantizedXIP2InverseError("callback per-pair Pose6 loss count differs from live pair count")
    observed_custody = (
        observation.authority,
        observation.frozen_scorer_sha256,
        observation.target_custody_sha256,
        observation.measurement_custody_sha256,
    )
    if store.custody is None:
        store.custody = observed_custody
    elif store.custody != observed_custody:
        raise QuantizedXIP2InverseError("callback authority/custody changed during the run")
    candidate = _candidate_from_observation(
        request=request,
        observation=observation,
        callback_evaluation_index=store.callback_count + 1,
        reference_control=reference_control,
    )
    store.append_evaluation(candidate)
    return candidate


def _coordinate_order(
    *,
    seed: int,
    arm_id: str,
    step: int,
    sweep: int,
    coordinate_count: int,
) -> tuple[int, ...]:
    ranked: list[tuple[str, int]] = []
    for coordinate in range(coordinate_count):
        rank = _sha256(
            _canonical_json(
                {
                    "arm_id": arm_id,
                    "coordinate": coordinate,
                    "schema": "tac.quantized_xip2_coordinate_rank.v1",
                    "seed": seed,
                    "step": step,
                    "sweep": sweep,
                }
            )
        )
        ranked.append((rank, coordinate))
    return tuple(coordinate for _rank, coordinate in sorted(ranked))


def _coordinate_order_sha256(
    *,
    config: QuantizedXIP2SearchConfigV1,
    arm_id: str,
    coordinate_count: int,
) -> str:
    orders = [
        {
            "order": list(
                _coordinate_order(
                    seed=config.seed,
                    arm_id=arm_id,
                    step=step,
                    sweep=sweep,
                    coordinate_count=coordinate_count,
                )
            ),
            "step": step,
            "sweep": sweep,
        }
        for step in config.step_schedule
        for sweep in range(config.sweeps_per_step)
    ]
    return _sha256(_canonical_json(orders))


@dataclass(slots=True)
class _ArmState:
    arm: QuantizedXIP2ArmV1
    initial: QuantizedXIP2CandidateV1
    current: QuantizedXIP2CandidateV1
    best: QuantizedXIP2CandidateV1
    evaluated_neighbors: int = 0
    search_completed: bool = False


def _candidate_rank(candidate: QuantizedXIP2CandidateV1) -> tuple[float, str]:
    return candidate.complete_score, candidate.packet_sha256


def _append_terminal(
    *,
    store: _StageStore,
    result: QuantizedXIP2SearchResultV1,
) -> None:
    index = store.callback_count + 1
    if index > 99_999_999:
        raise QuantizedXIP2InverseError("terminal stage index escaped the eight-digit V1 ABI")
    record = {
        "manifest_sha256": store.manifest_sha256,
        "previous_stage_sha256": store.previous_stage_sha256,
        "result": result.as_dict(),
        "schema": TERMINAL_STAGE_SCHEMA,
        "stage_index": index,
        "stage_kind": "terminal",
    }
    payload = _canonical_json(record)
    if store.terminal_payload is not None:
        if store.terminal_index != index or store.terminal_payload != payload:
            raise QuantizedXIP2InverseError("terminal result differs from deterministic replay")
        return
    _atomic_publish(store.run_dir / f"stage-{index:08d}-terminal.json", payload)
    store.terminal_payload = payload
    store.terminal_index = index


def run_quantized_xip2_inverse_search(
    *,
    surface: PassConditionalASurfaceV1,
    initial_xi: np.ndarray,
    config: QuantizedXIP2SearchConfigV1,
    objective: QuantizedXIP2ObjectiveCallbackV1,
    run_dir: Path,
    resume_from: Path | None,
) -> QuantizedXIP2SearchResultV1:
    """Run or resume deterministic direct-q search through G13 and an external callback."""

    if type(surface) is not PassConditionalASurfaceV1:
        raise QuantizedXIP2InverseError("surface must be exact live PassConditionalASurfaceV1")
    if type(config) is not QuantizedXIP2SearchConfigV1:
        raise QuantizedXIP2InverseError("config must be exact QuantizedXIP2SearchConfigV1")
    if not callable(objective):
        raise QuantizedXIP2InverseError("objective must be one callable accepting exactly one request")
    try:
        source = derive_pass_conditional_a_source_binding(surface)
    except PassConditionalAError as exc:
        raise QuantizedXIP2InverseError("could not derive the exact live G13 source binding") from exc
    xi = _fresh_initial_xi(initial_xi, pair_count=source.pair_count)
    initial_xi_sha = _array_sha256(xi)
    implementation_sha = _implementation_sha256()
    lineage_sha = _lineage_sha256(
        source_binding_sha256=source.binding_sha256,
        initial_xi_sha256=initial_xi_sha,
        config_sha256=config.config_sha256,
        implementation_sha256=implementation_sha,
    )
    manifest = _manifest_record(
        config=config,
        implementation_sha256=implementation_sha,
        source=source,
        initial_xi_sha256=initial_xi_sha,
        lineage_sha256=lineage_sha,
    )
    manifest_payload = _canonical_json(manifest)
    store = _prepare_run_directory(
        run_dir=run_dir,
        resume_from=resume_from,
        manifest_payload=manifest_payload,
    )

    controls: list[QuantizedXIP2CandidateV1] = []
    for mode in (
        CountedXIP2A3Mode.PASS_P0_V1,
        CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    ):
        request = _compile_control_request(
            mode=mode,
            surface=surface,
            lineage_sha256=lineage_sha,
            config_sha256=config.config_sha256,
            initial_xi_sha256=initial_xi_sha,
        )
        controls.append(
            _evaluate(
                request=request,
                objective=objective,
                store=store,
                config=config,
                source=source,
                reference_control=None,
            )
        )
    first_control, second_control = controls
    if (
        first_control.output_camera_y1_sha256 != source.conditional_camera_y1_sha256
        or second_control.output_camera_y1_sha256 != source.conditional_camera_y1_sha256
        or first_control.output_camera_y1_sha256 != second_control.output_camera_y1_sha256
    ):
        raise QuantizedXIP2InverseError("controls did not preserve exact live conditional Y1")
    if (
        first_control.seg_prediction_sha256 != second_control.seg_prediction_sha256
        or first_control.d_seg != second_control.d_seg
    ):
        raise QuantizedXIP2InverseError("PASS/global-copy controls did not establish exact Seg invariance")
    selected_control = min(
        controls,
        key=lambda candidate: (candidate.complete_score, _CONTROL_ORDER[candidate.mode]),
    )

    arm_states: list[_ArmState] = []
    arm_index = 0
    for q_level in config.q_levels:
        for coder in config.coders:
            for interpretation in config.interpretations:
                arm_id = f"arm-{arm_index:04d}-q{q_level}-{coder.value}-{interpretation.value}"
                try:
                    quantized_q, frozen_scales = quantize_xi(xi, q_levels=q_level)
                except (XiPoseCoderError, ValueError) as exc:
                    raise QuantizedXIP2InverseError("initial xi quantization failed") from exc
                initial_request = _compile_active_request(
                    q=quantized_q,
                    scales=frozen_scales,
                    arm_index=arm_index,
                    arm_id=arm_id,
                    q_level=q_level,
                    coder=coder,
                    interpretation=interpretation,
                    pitch=config.pitch,
                    surface=surface,
                    lineage_sha256=lineage_sha,
                    config_sha256=config.config_sha256,
                    initial_xi_sha256=initial_xi_sha,
                )
                initial_candidate = _evaluate(
                    request=initial_request,
                    objective=objective,
                    store=store,
                    config=config,
                    source=source,
                    reference_control=selected_control,
                )
                if (
                    initial_candidate.seg_prediction_sha256 != first_control.seg_prediction_sha256
                    or initial_candidate.d_seg != first_control.d_seg
                ):
                    raise QuantizedXIP2InverseError("active callback Seg prediction/d_seg drifted from controls")
                assert initial_request.q is not None
                assert initial_request.scales is not None
                arm = QuantizedXIP2ArmV1(
                    arm_index=arm_index,
                    arm_id=arm_id,
                    q_level=q_level,
                    coder=coder,
                    interpretation=interpretation,
                    frozen_scales=tuple(float(value) for value in initial_request.scales),
                    initial_q_sha256=_array_sha256(initial_request.q),
                    coordinate_order_sha256=_coordinate_order_sha256(
                        config=config,
                        arm_id=arm_id,
                        coordinate_count=initial_request.q.size,
                    ),
                )
                arm_states.append(
                    _ArmState(
                        arm=arm,
                        initial=initial_candidate,
                        current=initial_candidate,
                        best=initial_candidate,
                    )
                )
                arm_index += 1

    budget_stopped = False
    for state in arm_states:
        if budget_stopped:
            break
        completed = True
        for step in config.step_schedule:
            if not completed:
                break
            for sweep in range(config.sweeps_per_step):
                if not completed:
                    break
                current_q = np.asarray(state.current.q, dtype=np.int16)
                order = _coordinate_order(
                    seed=config.seed,
                    arm_id=state.arm.arm_id,
                    step=step,
                    sweep=sweep,
                    coordinate_count=current_q.size,
                )
                for coordinate in order:
                    current_q = np.asarray(state.current.q, dtype=np.int16)
                    flat = current_q.reshape(-1)
                    neighbor_requests: list[QuantizedXIP2EvaluationRequestV1] = []
                    for direction in (1, -1):
                        proposed = int(flat[coordinate]) + direction * step
                        if not -state.arm.q_level <= proposed <= state.arm.q_level:
                            continue
                        neighbor_q = current_q.copy()
                        neighbor_q.reshape(-1)[coordinate] = proposed
                        neighbor_requests.append(
                            _compile_active_request(
                                q=neighbor_q,
                                scales=np.asarray(state.arm.frozen_scales, dtype=np.float32),
                                arm_index=state.arm.arm_index,
                                arm_id=state.arm.arm_id,
                                q_level=state.arm.q_level,
                                coder=state.arm.coder,
                                interpretation=state.arm.interpretation,
                                pitch=config.pitch,
                                surface=surface,
                                lineage_sha256=lineage_sha,
                                config_sha256=config.config_sha256,
                                initial_xi_sha256=initial_xi_sha,
                            )
                        )
                    choices = [state.current]
                    coordinate_truncated = False
                    for request in neighbor_requests:
                        if (
                            request.packet_sha256 not in store.candidates_by_packet
                            and store.callback_count >= config.max_callback_evaluations
                        ):
                            completed = False
                            budget_stopped = True
                            coordinate_truncated = True
                            break
                        candidate = _evaluate(
                            request=request,
                            objective=objective,
                            store=store,
                            config=config,
                            source=source,
                            reference_control=selected_control,
                        )
                        if (
                            candidate.seg_prediction_sha256 != first_control.seg_prediction_sha256
                            or candidate.d_seg != first_control.d_seg
                        ):
                            raise QuantizedXIP2InverseError(
                                "active callback Seg prediction/d_seg drifted from controls"
                            )
                        choices.append(candidate)
                        state.evaluated_neighbors += 1
                        if _candidate_rank(candidate) < _candidate_rank(state.best):
                            state.best = candidate
                    state.current = min(choices, key=_candidate_rank)
                    if _candidate_rank(state.current) < _candidate_rank(state.best):
                        state.best = state.current
                    if coordinate_truncated:
                        break
        state.search_completed = completed

    arm_results = tuple(
        QuantizedXIP2ArmResultV1(
            arm=state.arm,
            initial_candidate=state.initial,
            final_candidate=state.current,
            best_candidate=state.best,
            evaluated_neighbors=state.evaluated_neighbors,
            search_completed=state.search_completed,
        )
        for state in arm_states
    )
    best_active = min((state.best for state in arm_states), key=_candidate_rank)
    bounded_completed = all(state.search_completed for state in arm_states)
    if not bounded_completed and store.callback_count != config.max_callback_evaluations:
        raise QuantizedXIP2InverseError("incomplete search did not consume the exact callback budget")
    if len(store.candidates_by_packet) != len(store.evaluation_candidates):
        raise QuantizedXIP2InverseError("duplicate/orphan evaluation stages are not replayable")
    assert store.consumed_packets is not None
    if len(store.consumed_packets) != len(store.evaluation_candidates):
        raise QuantizedXIP2InverseError("durable evaluation stage was not consumed by deterministic replay")
    if store.custody is None:
        raise QuantizedXIP2InverseError("callback custody was never established")
    authority, scorer_sha, target_sha, measurement_sha = store.custody
    active_evaluations = sum(
        candidate.mode is CountedXIP2A3Mode.XIP2_WARP_V1 for candidate in store.evaluation_candidates
    )
    receipt = QuantizedXIP2SearchReceiptV1(
        lineage_sha256=lineage_sha,
        config_sha256=config.config_sha256,
        implementation_sha256=implementation_sha,
        source_binding_sha256=source.binding_sha256,
        initial_xi_sha256=initial_xi_sha,
        callback_authority=authority,
        frozen_scorer_sha256=scorer_sha,
        target_custody_sha256=target_sha,
        measurement_custody_sha256=measurement_sha,
        callback_evaluations=store.callback_count,
        evaluated_active_packets=active_evaluations,
        bounded_search_completed=bounded_completed,
        bounded_search_negative=not (best_active.delta_s < 0.0),
        through_g13_uint8_receiver=True,
    )
    result = QuantizedXIP2SearchResultV1(
        lineage_sha256=lineage_sha,
        selected_control=selected_control,
        best_active_candidate=best_active,
        arm_results=arm_results,
        improved_control=best_active.delta_s < 0.0,
        bounded_search_completed=bounded_completed,
        stop_reason="SCHEDULE_COMPLETED" if bounded_completed else "CALLBACK_BUDGET_EXHAUSTED",
        receipt=receipt,
    )
    _append_terminal(store=store, result=result)
    return result


# Descriptive aliases keep the immutable record roles obvious to callers.
QuantizedXIP2InverseSearchReceiptV1 = QuantizedXIP2SearchReceiptV1
QuantizedXIP2InverseSearchResultV1 = QuantizedXIP2SearchResultV1
QuantizedXIP2CandidateEvaluationV1 = QuantizedXIP2CandidateV1


__all__ = [
    "FORMULATION_SCOPE",
    "RATE_DENOMINATOR_BYTES",
    "ROOT_INTEGRATION_BLOCKER",
    "QuantizedXIP2ArmResultV1",
    "QuantizedXIP2ArmV1",
    "QuantizedXIP2CandidateEvaluationV1",
    "QuantizedXIP2CandidateV1",
    "QuantizedXIP2EvaluationRequestV1",
    "QuantizedXIP2InverseError",
    "QuantizedXIP2InverseSearchReceiptV1",
    "QuantizedXIP2InverseSearchResultV1",
    "QuantizedXIP2ObjectiveAuthorityV1",
    "QuantizedXIP2ObjectiveCallbackV1",
    "QuantizedXIP2ObjectiveObservationV1",
    "QuantizedXIP2SearchConfigV1",
    "QuantizedXIP2SearchReceiptV1",
    "QuantizedXIP2SearchResultV1",
    "quantized_xip2_complete_score",
    "quantized_xip2_strict_archive_ceiling",
    "run_quantized_xip2_inverse_search",
]
