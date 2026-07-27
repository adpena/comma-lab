# SPDX-License-Identifier: MIT
"""Closed G17 actuator IR for the first ep725 semantic state transition.

This module connects an exact counted label-local ``G`` page to the existing
V2 semantic decoder and the predictor-preserving camera overlay.  It does not
compile from teacher data, invoke a scorer, claim n600 execution, or construct
a public archive.  Its bounded executor consumes the exact ep725 runtime
surface and proves that Y0 and every G-unowned Y1 byte survive unchanged.

The program is deliberately not a byte VM.  Its only V1 actuator kind and
receiver operation are closed enums, and dispatch is a direct audited call to
the existing typed donors.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

import tac.witness_dsl.predictor_preserving_taskspace_overlay as _overlay_module
import tac.witness_dsl.taskspace_predictor_v2_consumer_seam as _seam_module
from tac.witness_dsl.ep725_levelset_predictor_adapter import Ep725EphemeralRuntimeSurfaceV2
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayResultV1,
    overlay_g_on_predictor_camera_y1,
    parse_predictor_preserving_overlay_receipt,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    GTransportRequirementV2,
    NoTransportV2,
    TaskspacePredictorStateV2,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    GCorrectionOwnershipV2,
    apply_generative_taskspace_correction_v2,
    derive_g_correction_ownership_v2,
    parse_generative_taskspace_correction_v2,
)

if TYPE_CHECKING:
    from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1

PROGRAM_SCHEMA: Final = "tac.g17_actuator_program.v1"
EXECUTION_SCHEMA: Final = "tac.g17_actuator_execution_receipt.v1"
CHECKPOINT_SCHEMA: Final = "tac.g17_actuator_checkpoint_receipt.v1"
OPERAND_PACKET_SCHEMA: Final = "tac.generative_taskspace_correction.v1"
STATE_CONTRACT_ID: Final = "TaskspacePredictorStateV2+NoTransportV2"
PHYSICAL_SECTION_NAME: Final = "G_LABEL_LOCAL_PAGE"
SEMANTIC_RECEIVER_OPERATION: Final = (
    "tac.witness_dsl.taskspace_predictor_v2_consumer_seam.parse+apply_generative_taskspace_correction_v2"
)
OVERLAY_RECEIVER_OPERATION: Final = (
    "tac.witness_dsl.predictor_preserving_taskspace_overlay.overlay_g_on_predictor_camera_y1"
)
MAX_BOUNDED_EXECUTION_PAIRS: Final = 2
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ASCII_ID_RE: Final = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class G17ActuatorIRError(ValueError):
    """A counted span, state binding, execution, or receipt failed closed."""


class G17ActuatorKindV1(StrEnum):
    """The only physically admitted actuator in V1."""

    EP725_LABEL_LOCAL_SEMANTIC_G = "EP725_LABEL_LOCAL_SEMANTIC_G"


class G17ActuatorReceiverOperationV1(StrEnum):
    """Closed dispatch keys; never import or execute a caller-provided name."""

    EP725_V2_G_THEN_PREDICTOR_PRESERVING_OVERLAY = "EP725_V2_G_THEN_PREDICTOR_PRESERVING_OVERLAY"


class G17ActuatorCheckpointStageV1(StrEnum):
    """Crash-resume stages retained one pair-page at a time."""

    OPERAND_VERIFIED = "OPERAND_VERIFIED"
    PAIR_OUTPUT_REALIZED = "PAIR_OUTPUT_REALIZED"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


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
        raise G17ActuatorIRError("actuator receipt is not finite canonical ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, expected_fields: set[str], schema: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise G17ActuatorIRError("receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G17ActuatorIRError(f"receipt repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except G17ActuatorIRError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G17ActuatorIRError("receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != expected_fields:
        raise G17ActuatorIRError("receipt fields differ from the closed schema")
    if value.get("schema") != schema or _canonical_json(value) != payload:
        raise G17ActuatorIRError("receipt schema or canonical parse/re-emit identity changed")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise G17ActuatorIRError(f"{name} must be canonical lowercase SHA-256")
    return value


def _require_id(value: object, *, name: str) -> str:
    if type(value) is not str or _ASCII_ID_RE.fullmatch(value) is None:
        raise G17ActuatorIRError(f"{name} must be a closed printable ASCII identifier")
    return value


def _require_member_name(value: object) -> str:
    result = _require_id(value, name="member_name")
    if result.startswith(".") or "/" in result or "\\" in result:
        raise G17ActuatorIRError("member_name must be one exact archive basename")
    return result


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _source_sha256(module: object) -> str:
    source_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if source_path.suffix != ".py" or not source_path.is_file():
        raise G17ActuatorIRError("receiver identity is not backed by exact Python source bytes")
    return _sha256(source_path.read_bytes())


def _slice_predictor_state(
    state: TaskspacePredictorStateV2,
    *,
    pair_start: int,
    pair_count: int,
) -> TaskspacePredictorStateV2:
    if type(state) is not TaskspacePredictorStateV2 or type(state.transport) is not NoTransportV2:
        raise G17ActuatorIRError("EP725_LABEL_LOCAL_SEMANTIC_G requires exact V2 NONE-transport state")
    stop = pair_start + pair_count
    if type(pair_start) is not int or type(pair_count) is not int or pair_count < 1:
        raise G17ActuatorIRError("operand pair window must use positive exact integers")
    if pair_start not in state.source_pair_ids or stop - 1 not in state.source_pair_ids:
        raise G17ActuatorIRError("operand pair window escaped the predictor population")
    local_start = pair_start - state.pair_start
    local_stop = local_start + pair_count
    expected_ids = tuple(range(pair_start, stop))
    if state.source_pair_ids[local_start:local_stop] != expected_ids:
        raise G17ActuatorIRError("operand pair window is not one chronological predictor slice")
    return TaskspacePredictorStateV2(
        predictor_program=state.predictor_program,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        source_archive_sha256=state.source_archive_sha256,
        source_runtime_sha256=state.source_runtime_sha256,
        source_member_name=state.source_member_name,
        source_pair_ids=expected_ids,
        labels=state.labels[local_start:local_stop],
        transport=NoTransportV2(),
    )


def _state_content_sha256(
    state: TaskspacePredictorStateV2,
    chronology: np.ndarray,
    labels: np.ndarray,
) -> str:
    return _sha256(
        b"G17-ACTUATOR-REALIZED-STATE-V1\0"
        + bytes.fromhex(state.binding_sha256)
        + memoryview(np.ascontiguousarray(chronology)).cast("B")
        + memoryview(np.ascontiguousarray(labels)).cast("B")
    )


@dataclass(frozen=True, slots=True)
class G17ActuatorPhysicalSpanGroupV1:
    """Actuator-local member interval owned by declared operand spans.

    This is deliberately not the canonical archive-level
    ``G17PhysicalCodingGroupV1``.  The latter owns exact outer-archive and
    member custody in ``taskspace_selected_solution_compiler``; this bounded
    type is only the member-relative execution view used by the actuator.
    """

    physical_coding_group_id: str
    member_name: str
    member_offset: int
    byte_length: int

    def __post_init__(self) -> None:
        _require_id(self.physical_coding_group_id, name="physical_coding_group_id")
        _require_member_name(self.member_name)
        if type(self.member_offset) is not int or self.member_offset < 0:
            raise G17ActuatorIRError("coding-group member_offset must be a nonnegative exact integer")
        if type(self.byte_length) is not int or self.byte_length < 1:
            raise G17ActuatorIRError("coding-group byte_length must be a positive exact integer")

    @property
    def stop(self) -> int:
        return self.member_offset + self.byte_length


@dataclass(frozen=True, slots=True)
class G17ActuatorOperandRefV1:
    """Exact counted archive-member span for one predictor-bound G page."""

    operand_id: str
    kind: G17ActuatorKindV1
    physical_coding_group_id: str
    member_name: str
    member_offset: int
    byte_length: int
    operand_sha256: str
    packet_schema: Literal["tac.generative_taskspace_correction.v1"]
    section_name: Literal["G_LABEL_LOCAL_PAGE"]
    pair_start: int
    pair_count: int
    predictor_slice_binding_sha256: str
    counted: Literal[True] = True

    def __post_init__(self) -> None:
        _require_id(self.operand_id, name="operand_id")
        if type(self.kind) is not G17ActuatorKindV1 or self.kind is not G17ActuatorKindV1.EP725_LABEL_LOCAL_SEMANTIC_G:
            raise G17ActuatorIRError("operand kind is not the closed V1 semantic-G actuator")
        _require_id(self.physical_coding_group_id, name="physical_coding_group_id")
        _require_member_name(self.member_name)
        if type(self.member_offset) is not int or self.member_offset < 0:
            raise G17ActuatorIRError("operand member_offset must be a nonnegative exact integer")
        if type(self.byte_length) is not int or self.byte_length < 1:
            raise G17ActuatorIRError("operand byte_length must be a positive exact integer")
        _require_sha256(self.operand_sha256, name="operand_sha256")
        _require_sha256(self.predictor_slice_binding_sha256, name="predictor_slice_binding_sha256")
        if self.packet_schema != OPERAND_PACKET_SCHEMA or self.section_name != PHYSICAL_SECTION_NAME:
            raise G17ActuatorIRError("operand packet schema or physical section changed")
        if type(self.pair_start) is not int or type(self.pair_count) is not int or self.pair_count < 1:
            raise G17ActuatorIRError("operand pair window must use positive exact integers")
        if self.counted is not True:
            raise G17ActuatorIRError("every actuator operand must be counted")

    @property
    def stop(self) -> int:
        return self.member_offset + self.byte_length

    @property
    def pair_stop(self) -> int:
        return self.pair_start + self.pair_count

    def reopen(self, member_bytes: bytes) -> bytes:
        if type(member_bytes) is not bytes:
            raise G17ActuatorIRError("counted member must be exact immutable bytes")
        if self.stop > len(member_bytes):
            raise G17ActuatorIRError("operand span escaped the counted member")
        payload = member_bytes[self.member_offset : self.stop]
        if len(payload) != self.byte_length or _sha256(payload) != self.operand_sha256:
            raise G17ActuatorIRError("reopened operand bytes differ from exact span hash")
        return payload


@dataclass(frozen=True, slots=True)
class G17ActuatorProgramReceiptV1:
    schema: Literal["tac.g17_actuator_program.v1"]
    state_contract_id: Literal["TaskspacePredictorStateV2+NoTransportV2"]
    receiver_operation: G17ActuatorReceiverOperationV1
    counted_member_name: str
    counted_member_bytes: int
    counted_member_sha256: str
    counted_actuator_bytes: int
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_program_bytes: int
    predictor_renderer_sha256: str
    source_archive_sha256: str
    source_runtime_sha256: str
    source_member_name: str
    source_pair_ids: tuple[int, ...]
    chronological_pair_order: tuple[int, ...]
    physical_coding_groups: tuple[dict[str, Any], ...]
    operands: tuple[dict[str, Any], ...]
    operand_manifest_sha256: str
    semantic_receiver_operation: str
    overlay_receiver_operation: str
    semantic_receiver_source_sha256: str
    overlay_receiver_source_sha256: str
    generic_receiver_dispatch_only: Literal[True]
    byte_vm_impersonation: Literal[False]
    teacher_or_gt_bytes_counted_as_operands: Literal[0]
    scorer_weight_bytes_counted_as_operands: Literal[0]
    public_archive_proven: Literal[False]
    n600_execution_proven: Literal[False]
    research_only: Literal[True]

    def __post_init__(self) -> None:
        if self.schema != PROGRAM_SCHEMA or self.state_contract_id != STATE_CONTRACT_ID:
            raise G17ActuatorIRError("program receipt schema or state contract changed")
        if (
            type(self.receiver_operation) is not G17ActuatorReceiverOperationV1
            or self.receiver_operation
            is not G17ActuatorReceiverOperationV1.EP725_V2_G_THEN_PREDICTOR_PRESERVING_OVERLAY
        ):
            raise G17ActuatorIRError("program receipt receiver dispatch is not closed")
        _require_member_name(self.counted_member_name)
        for name in (
            "counted_member_sha256",
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "source_archive_sha256",
            "source_runtime_sha256",
            "operand_manifest_sha256",
            "semantic_receiver_source_sha256",
            "overlay_receiver_source_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_member_name(self.source_member_name)
        if any(
            type(value) is not int or value < 1
            for value in (self.counted_member_bytes, self.counted_actuator_bytes, self.predictor_program_bytes)
        ):
            raise G17ActuatorIRError("program byte counts must be positive exact integers")
        if (
            type(self.source_pair_ids) is not tuple
            or not self.source_pair_ids
            or self.chronological_pair_order != self.source_pair_ids
            or self.source_pair_ids != tuple(range(self.source_pair_ids[0], self.source_pair_ids[-1] + 1))
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids[0] < 0
            or self.source_pair_ids[-1] >= 600
        ):
            raise G17ActuatorIRError("program receipt pair population is not exact chronological [0,600) subset")
        if type(self.physical_coding_groups) is not tuple or not self.physical_coding_groups:
            raise G17ActuatorIRError("program receipt lacks physical coding groups")
        if type(self.operands) is not tuple or not self.operands:
            raise G17ActuatorIRError("program receipt lacks counted operands")
        group_fields = {row.name for row in fields(G17ActuatorPhysicalSpanGroupV1)}
        operand_fields = {row.name for row in fields(G17ActuatorOperandRefV1)}
        typed_groups: list[G17ActuatorPhysicalSpanGroupV1] = []
        typed_operands: list[G17ActuatorOperandRefV1] = []
        try:
            for row in self.physical_coding_groups:
                if type(row) is not dict or set(row) != group_fields:
                    raise G17ActuatorIRError("program receipt coding-group fields are not exact")
                typed_groups.append(G17ActuatorPhysicalSpanGroupV1(**row))
            for row in self.operands:
                if type(row) is not dict or set(row) != operand_fields:
                    raise G17ActuatorIRError("program receipt operand fields are not exact")
                converted = dict(row)
                converted["kind"] = G17ActuatorKindV1(converted["kind"])
                typed_operands.append(G17ActuatorOperandRefV1(**converted))
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, G17ActuatorIRError):
                raise
            raise G17ActuatorIRError("program receipt nested manifest failed exact typing") from exc
        groups_by_id = {row.physical_coding_group_id: row for row in typed_groups}
        if len(groups_by_id) != len(typed_groups) or len({row.operand_id for row in typed_operands}) != len(
            typed_operands
        ):
            raise G17ActuatorIRError("program receipt group or operand identifiers are not unique")
        if any(
            row.member_name != self.counted_member_name or row.stop > self.counted_member_bytes for row in typed_groups
        ):
            raise G17ActuatorIRError("program receipt coding group escaped its counted member")
        ordered_groups = sorted(typed_groups, key=lambda row: row.member_offset)
        if any(left.stop > right.member_offset for left, right in pairwise(ordered_groups)):
            raise G17ActuatorIRError("program receipt coding groups overlap and double-own counted bytes")
        if any(
            row.member_name != self.counted_member_name or row.physical_coding_group_id not in groups_by_id
            for row in typed_operands
        ):
            raise G17ActuatorIRError("program receipt operand names a foreign member or group")
        for group_id, group in groups_by_id.items():
            rows = sorted(
                (row for row in typed_operands if row.physical_coding_group_id == group_id),
                key=lambda row: row.member_offset,
            )
            if not rows:
                raise G17ActuatorIRError("program receipt coding group has no operand")
            cursor = group.member_offset
            for row in rows:
                if row.member_offset != cursor:
                    raise G17ActuatorIRError("program receipt coding group has overlapping or gapped bytes")
                cursor = row.stop
            if cursor != group.stop:
                raise G17ActuatorIRError("program receipt coding group has trailing unowned bytes")
        if sum(row.byte_length for row in typed_operands) != self.counted_actuator_bytes:
            raise G17ActuatorIRError("program receipt counted actuator byte total differs from operand spans")
        ordered_operands = tuple(sorted(typed_operands, key=lambda row: (row.pair_start, row.member_offset)))
        if tuple(typed_operands) != ordered_operands:
            raise G17ActuatorIRError("program receipt operands are not chronological")
        pair_cursor = self.source_pair_ids[0]
        for row in ordered_operands:
            if row.pair_start != pair_cursor:
                raise G17ActuatorIRError("program receipt pair population has overlap or gap")
            pair_cursor = row.pair_stop
        if pair_cursor != self.source_pair_ids[-1] + 1:
            raise G17ActuatorIRError("program receipt operands do not cover its pair population")
        expected_manifest_sha256 = _sha256(
            b"G17-ACTUATOR-OPERAND-MANIFEST-V1\0"
            + _canonical_json(
                {
                    "groups": self.physical_coding_groups,
                    "operands": self.operands,
                }
            )
        )
        if self.operand_manifest_sha256 != expected_manifest_sha256:
            raise G17ActuatorIRError("program receipt operand manifest hash is not recomposable")
        if (
            self.semantic_receiver_operation != SEMANTIC_RECEIVER_OPERATION
            or self.overlay_receiver_operation != OVERLAY_RECEIVER_OPERATION
        ):
            raise G17ActuatorIRError("program receipt receiver identity strings changed")
        truth = {
            "generic_receiver_dispatch_only": True,
            "byte_vm_impersonation": False,
            "teacher_or_gt_bytes_counted_as_operands": 0,
            "scorer_weight_bytes_counted_as_operands": 0,
            "public_archive_proven": False,
            "n600_execution_proven": False,
            "research_only": True,
        }
        if any(getattr(self, name) != expected for name, expected in truth.items()):
            raise G17ActuatorIRError("program receipt truth contract became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["receiver_operation"] = self.receiver_operation.value
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["chronological_pair_order"] = list(self.chronological_pair_order)
        value["physical_coding_groups"] = list(self.physical_coding_groups)
        value["operands"] = list(self.operands)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class G17ActuatorProgramV1:
    """Exact counted member plus a complete, non-overlapping operand directory."""

    predictor_state: TaskspacePredictorStateV2
    counted_member_name: str
    counted_member_bytes: bytes = field(repr=False)
    physical_coding_groups: tuple[G17ActuatorPhysicalSpanGroupV1, ...]
    operands: tuple[G17ActuatorOperandRefV1, ...]
    receiver_operation: G17ActuatorReceiverOperationV1 = field(
        default=G17ActuatorReceiverOperationV1.EP725_V2_G_THEN_PREDICTOR_PRESERVING_OVERLAY,
        init=False,
    )
    receipt: G17ActuatorProgramReceiptV1 = field(init=False)

    def __post_init__(self) -> None:
        state = self.predictor_state
        if type(state) is not TaskspacePredictorStateV2 or type(state.transport) is not NoTransportV2:
            raise G17ActuatorIRError("actuator program requires exact TaskspacePredictorStateV2 + NoTransportV2")
        member_name = _require_member_name(self.counted_member_name)
        if type(self.counted_member_bytes) is not bytes or not self.counted_member_bytes:
            raise G17ActuatorIRError("actuator program requires nonempty exact counted member bytes")
        if type(self.physical_coding_groups) is not tuple or not self.physical_coding_groups:
            raise G17ActuatorIRError("actuator program requires exact physical coding groups")
        if type(self.operands) is not tuple or not self.operands:
            raise G17ActuatorIRError("actuator program requires exact counted operands")
        if any(type(row) is not G17ActuatorPhysicalSpanGroupV1 for row in self.physical_coding_groups):
            raise G17ActuatorIRError("physical coding groups changed exact type")
        if any(type(row) is not G17ActuatorOperandRefV1 for row in self.operands):
            raise G17ActuatorIRError("actuator operands changed exact type")
        if len({row.physical_coding_group_id for row in self.physical_coding_groups}) != len(
            self.physical_coding_groups
        ):
            raise G17ActuatorIRError("physical coding group IDs are not unique")
        if len({row.operand_id for row in self.operands}) != len(self.operands):
            raise G17ActuatorIRError("operand IDs are not unique")
        groups = {row.physical_coding_group_id: row for row in self.physical_coding_groups}
        if any(row.member_name != member_name or row.stop > len(self.counted_member_bytes) for row in groups.values()):
            raise G17ActuatorIRError("coding group member identity or span differs from counted member")
        ordered_groups = sorted(groups.values(), key=lambda row: row.member_offset)
        if any(left.stop > right.member_offset for left, right in pairwise(ordered_groups)):
            raise G17ActuatorIRError("physical coding groups overlap and double-own counted bytes")
        if any(row.member_name != member_name or row.physical_coding_group_id not in groups for row in self.operands):
            raise G17ActuatorIRError("operand names a foreign member or undeclared coding group")

        for group_id, group in groups.items():
            rows = sorted(
                (row for row in self.operands if row.physical_coding_group_id == group_id),
                key=lambda row: row.member_offset,
            )
            if not rows:
                raise G17ActuatorIRError("declared physical coding group has no owned operand bytes")
            cursor = group.member_offset
            for row in rows:
                if row.member_offset != cursor:
                    relation = "overlap" if row.member_offset < cursor else "gap"
                    raise G17ActuatorIRError(f"physical coding group has an operand {relation}")
                cursor = row.stop
            if cursor != group.stop:
                raise G17ActuatorIRError("physical coding group has trailing unowned bytes")

        ordered = tuple(sorted(self.operands, key=lambda row: (row.pair_start, row.member_offset)))
        if ordered != self.operands:
            raise G17ActuatorIRError("actuator operands must be in chronological pair/span order")
        pair_cursor = state.pair_start
        for operand in ordered:
            if operand.pair_start != pair_cursor:
                relation = "overlap" if operand.pair_start < pair_cursor else "gap"
                raise G17ActuatorIRError(f"actuator pair population has an operand {relation}")
            pair_cursor = operand.pair_stop
            slice_state = _slice_predictor_state(
                state,
                pair_start=operand.pair_start,
                pair_count=operand.pair_count,
            )
            if operand.predictor_slice_binding_sha256 != slice_state.binding_sha256:
                raise G17ActuatorIRError("operand predictor-slice foreign key differs from exact V2 state")
            packet = operand.reopen(self.counted_member_bytes)
            parsed = parse_generative_taskspace_correction_v2(packet, predictor_state=slice_state)
            if (
                parsed.pair_start != operand.pair_start
                or parsed.pair_count != operand.pair_count
                or parsed.packet_bytes != operand.byte_length
            ):
                raise G17ActuatorIRError("operand span or pair declaration differs from parsed G packet")
        if pair_cursor != state.pair_start + state.pair_count:
            raise G17ActuatorIRError("actuator operands do not cover the predictor pair population exactly")

        groups_payload = tuple(asdict(row) for row in self.physical_coding_groups)
        operands_payload = tuple(
            {
                **asdict(row),
                "kind": row.kind.value,
            }
            for row in self.operands
        )
        manifest_sha = _sha256(
            b"G17-ACTUATOR-OPERAND-MANIFEST-V1\0"
            + _canonical_json({"groups": groups_payload, "operands": operands_payload})
        )
        receipt = G17ActuatorProgramReceiptV1(
            schema=PROGRAM_SCHEMA,
            state_contract_id=STATE_CONTRACT_ID,
            receiver_operation=self.receiver_operation,
            counted_member_name=member_name,
            counted_member_bytes=len(self.counted_member_bytes),
            counted_member_sha256=_sha256(self.counted_member_bytes),
            counted_actuator_bytes=sum(row.byte_length for row in self.operands),
            predictor_state_binding_sha256=state.binding_sha256,
            predictor_semantic_binding_sha256=state.semantic_binding_sha256,
            predictor_program_sha256=state.predictor_program_sha256,
            predictor_program_bytes=state.predictor_program_bytes,
            predictor_renderer_sha256=state.predictor_renderer_sha256,
            source_archive_sha256=state.source_archive_sha256,
            source_runtime_sha256=state.source_runtime_sha256,
            source_member_name=state.source_member_name,
            source_pair_ids=state.source_pair_ids,
            chronological_pair_order=state.source_pair_ids,
            physical_coding_groups=groups_payload,
            operands=operands_payload,
            operand_manifest_sha256=manifest_sha,
            semantic_receiver_operation=SEMANTIC_RECEIVER_OPERATION,
            overlay_receiver_operation=OVERLAY_RECEIVER_OPERATION,
            semantic_receiver_source_sha256=_source_sha256(_seam_module),
            overlay_receiver_source_sha256=_source_sha256(_overlay_module),
            generic_receiver_dispatch_only=True,
            byte_vm_impersonation=False,
            teacher_or_gt_bytes_counted_as_operands=0,
            scorer_weight_bytes_counted_as_operands=0,
            public_archive_proven=False,
            n600_execution_proven=False,
            research_only=True,
        )
        object.__setattr__(self, "receipt", receipt)

    @property
    def program_sha256(self) -> str:
        return _sha256(b"G17-ACTUATOR-PROGRAM-V1\0" + self.counted_member_bytes + self.receipt.to_receipt_bytes())


def parse_g17_actuator_program_receipt(payload: bytes) -> G17ActuatorProgramReceiptV1:
    value = _decode_canonical_json(
        payload,
        expected_fields={row.name for row in fields(G17ActuatorProgramReceiptV1)},
        schema=PROGRAM_SCHEMA,
    )
    try:
        value["receiver_operation"] = G17ActuatorReceiverOperationV1(value["receiver_operation"])
        value["source_pair_ids"] = tuple(value["source_pair_ids"])
        value["chronological_pair_order"] = tuple(value["chronological_pair_order"])
        value["physical_coding_groups"] = tuple(value["physical_coding_groups"])
        value["operands"] = tuple(value["operands"])
        receipt = G17ActuatorProgramReceiptV1(**value)
    except (KeyError, TypeError, ValueError) as exc:
        raise G17ActuatorIRError("program receipt values failed exact typed validation") from exc
    if receipt.to_receipt_bytes() != payload:
        raise G17ActuatorIRError("program receipt changed on strict typed parse-back")
    return receipt


def reverify_g17_actuator_program_receipt(
    payload: bytes,
    *,
    program: G17ActuatorProgramV1,
) -> G17ActuatorProgramReceiptV1:
    if type(program) is not G17ActuatorProgramV1:
        raise G17ActuatorIRError("program reverify requires exact G17ActuatorProgramV1")
    receipt = parse_g17_actuator_program_receipt(payload)
    if receipt != program.receipt or payload != program.receipt.to_receipt_bytes():
        raise G17ActuatorIRError("program receipt differs from reopened member/state/span graph")
    return receipt


@dataclass(frozen=True, slots=True)
class G17ActuatorStepExecutionV1:
    operand_id: str
    pair_start: int
    pair_count: int
    operand_sha256: str
    predictor_slice_binding_sha256: str
    input_state_sha256: str
    output_state_sha256: str
    corrected_labels_sha256: str
    ownership_mask_sha256: str
    overlay_receipt_sha256: str
    changed_semantic_cells: int
    owned_camera_values: int
    actually_changed_camera_values: int

    def __post_init__(self) -> None:
        _require_id(self.operand_id, name="step.operand_id")
        for name in (
            "operand_sha256",
            "predictor_slice_binding_sha256",
            "input_state_sha256",
            "output_state_sha256",
            "corrected_labels_sha256",
            "ownership_mask_sha256",
            "overlay_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name=f"step.{name}")
        if type(self.pair_start) is not int or type(self.pair_count) is not int or self.pair_count < 1:
            raise G17ActuatorIRError("step pair window is invalid")
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 1
            for name in (
                "changed_semantic_cells",
                "owned_camera_values",
                "actually_changed_camera_values",
            )
        ):
            raise G17ActuatorIRError("step must record positive exact realized-change counts")


@dataclass(frozen=True, slots=True)
class G17ActuatorExecutionReceiptV1:
    schema: Literal["tac.g17_actuator_execution_receipt.v1"]
    program_sha256: str
    program_receipt_sha256: str
    counted_member_sha256: str
    counted_actuator_bytes: int
    predictor_state_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    source_archive_sha256: str
    source_runtime_sha256: str
    source_pair_ids: tuple[int, ...]
    chronological_pair_order: tuple[int, ...]
    input_chronology_sha256: str
    input_y0_sha256: str
    input_y1_sha256: str
    output_chronology_sha256: str
    output_y0_sha256: str
    output_y1_sha256: str
    output_raw_bytes: int
    step_receipts: tuple[G17ActuatorStepExecutionV1, ...]
    semantic_receiver_source_sha256: str
    overlay_receiver_source_sha256: str
    transport_requirement: Literal["LABEL_LOCAL"]
    y0_preserved: Literal[True]
    unowned_y1_preserved: Literal[True]
    deterministic_double_replay: Literal[True]
    scorer_invoked: Literal[False]
    score_claim: Literal[False]
    candidate_claim: Literal[False]
    public_rgb_output_proven: Literal[False]
    n600_execution_proven: Literal[False]
    teacher_or_gt_payload_bytes: Literal[0]
    scorer_weight_payload_bytes: Literal[0]
    byte_vm_impersonation: Literal[False]
    research_only: Literal[True]

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_SCHEMA:
            raise G17ActuatorIRError("execution receipt schema changed")
        for name in (
            "program_sha256",
            "program_receipt_sha256",
            "counted_member_sha256",
            "predictor_state_binding_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "source_archive_sha256",
            "source_runtime_sha256",
            "input_chronology_sha256",
            "input_y0_sha256",
            "input_y1_sha256",
            "output_chronology_sha256",
            "output_y0_sha256",
            "output_y1_sha256",
            "semantic_receiver_source_sha256",
            "overlay_receiver_source_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if (
            type(self.source_pair_ids) is not tuple
            or not self.source_pair_ids
            or self.chronological_pair_order != self.source_pair_ids
            or len(self.source_pair_ids) > MAX_BOUNDED_EXECUTION_PAIRS
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids != tuple(range(self.source_pair_ids[0], self.source_pair_ids[-1] + 1))
        ):
            raise G17ActuatorIRError("execution receipt is not a bounded chronological population")
        if type(self.counted_actuator_bytes) is not int or self.counted_actuator_bytes < 1:
            raise G17ActuatorIRError("execution counted actuator bytes are invalid")
        expected_raw = len(self.source_pair_ids) * 2 * 874 * 1164 * 3
        if type(self.output_raw_bytes) is not int or self.output_raw_bytes != expected_raw:
            raise G17ActuatorIRError("execution output raw byte count differs from exact chronology geometry")
        if (
            type(self.step_receipts) is not tuple
            or not self.step_receipts
            or any(type(row) is not G17ActuatorStepExecutionV1 for row in self.step_receipts)
        ):
            raise G17ActuatorIRError("execution receipt lacks exact step receipts")
        pair_cursor = self.source_pair_ids[0]
        for row in self.step_receipts:
            if row.pair_start != pair_cursor:
                raise G17ActuatorIRError("execution step population has overlap or gap")
            pair_cursor += row.pair_count
        if pair_cursor != self.source_pair_ids[-1] + 1:
            raise G17ActuatorIRError("execution steps do not cover the source population")
        if self.output_y0_sha256 != self.input_y0_sha256:
            raise G17ActuatorIRError("execution receipt does not self-prove exact Y0 preservation")
        if self.transport_requirement != GTransportRequirementV2.LABEL_LOCAL.value:
            raise G17ActuatorIRError("execution transport requirement is not label-local")
        truth = {
            "y0_preserved": True,
            "unowned_y1_preserved": True,
            "deterministic_double_replay": True,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "public_rgb_output_proven": False,
            "n600_execution_proven": False,
            "teacher_or_gt_payload_bytes": 0,
            "scorer_weight_payload_bytes": 0,
            "byte_vm_impersonation": False,
            "research_only": True,
        }
        if any(getattr(self, name) != expected for name, expected in truth.items()):
            raise G17ActuatorIRError("execution receipt truth contract became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["chronological_pair_order"] = list(self.chronological_pair_order)
        value["step_receipts"] = [asdict(row) for row in self.step_receipts]
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class G17ActuatorExecutionResultV1:
    chronological_frames: np.ndarray
    decoded_g_by_operand: tuple[DecodedGenerativeCorrectionV1, ...]
    ownership_by_operand: tuple[GCorrectionOwnershipV2, ...]
    overlay_by_operand: tuple[PredictorPreservingOverlayResultV1, ...]
    receipt: G17ActuatorExecutionReceiptV1

    def __post_init__(self) -> None:
        frames_array = np.asarray(self.chronological_frames)
        expected_shape = (len(self.receipt.source_pair_ids), 2, 874, 1164, 3)
        if frames_array.dtype != np.uint8 or frames_array.shape != expected_shape:
            raise G17ActuatorIRError("execution result chronology changed exact camera ABI")
        if _array_sha256(frames_array) != self.receipt.output_chronology_sha256:
            raise G17ActuatorIRError("execution result chronology differs from receipt")
        count = len(self.receipt.step_receipts)
        if not (
            len(self.decoded_g_by_operand) == len(self.ownership_by_operand) == len(self.overlay_by_operand) == count
        ):
            raise G17ActuatorIRError("execution result typed donor outputs differ from step count")
        immutable = np.ascontiguousarray(frames_array, dtype=np.uint8).copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "chronological_frames", immutable)


def _execute_once(
    program: G17ActuatorProgramV1,
    surface: Ep725EphemeralRuntimeSurfaceV2,
) -> G17ActuatorExecutionResultV1:
    if type(program) is not G17ActuatorProgramV1:
        raise G17ActuatorIRError("execution requires exact G17ActuatorProgramV1")
    if type(surface) is not Ep725EphemeralRuntimeSurfaceV2:
        raise G17ActuatorIRError("execution requires exact ep725 ephemeral runtime surface")
    state = surface.predictor_state
    if state.binding_sha256 != program.predictor_state.binding_sha256:
        raise G17ActuatorIRError("execution ep725 surface differs from program predictor state")
    if type(state.transport) is not NoTransportV2:
        raise G17ActuatorIRError("execution refuses non-NONE ep725 transport before mutation")
    chronology = np.asarray(surface.chronological_camera_frames)
    if chronology.shape[0] != state.pair_count or state.pair_count > MAX_BOUNDED_EXECUTION_PAIRS:
        raise G17ActuatorIRError("execution surface escaped bounded n1/n2 mechanism scope")
    output = np.ascontiguousarray(chronology.copy())
    decoded_rows: list[DecodedGenerativeCorrectionV1] = []
    ownership_rows: list[GCorrectionOwnershipV2] = []
    overlay_rows: list[PredictorPreservingOverlayResultV1] = []
    step_rows: list[G17ActuatorStepExecutionV1] = []

    for operand in program.operands:
        local_start = operand.pair_start - state.pair_start
        local_stop = local_start + operand.pair_count
        slice_state = _slice_predictor_state(
            state,
            pair_start=operand.pair_start,
            pair_count=operand.pair_count,
        )
        packet = operand.reopen(program.counted_member_bytes)
        parsed = parse_generative_taskspace_correction_v2(packet, predictor_state=slice_state)
        if parsed.predictor_binding_sha256 != slice_state.binding_sha256:
            raise G17ActuatorIRError("semantic receiver parse lost exact predictor-slice foreign key")
        decoded_first = apply_generative_taskspace_correction_v2(packet, predictor_state=slice_state)
        decoded_second = apply_generative_taskspace_correction_v2(packet, predictor_state=slice_state)
        if not np.array_equal(decoded_first.labels, decoded_second.labels):
            raise G17ActuatorIRError("semantic G receiver is nondeterministic")
        ownership = derive_g_correction_ownership_v2(slice_state, decoded_first)
        input_slice = np.ascontiguousarray(chronology[local_start:local_stop])
        overlay_first = overlay_g_on_predictor_camera_y1(
            input_slice[:, 1],
            slice_state.labels,
            decoded_first,
        )
        overlay_second = overlay_g_on_predictor_camera_y1(
            input_slice[:, 1],
            slice_state.labels,
            decoded_second,
        )
        if overlay_first.receipt != overlay_second.receipt or not np.array_equal(
            overlay_first.camera_y1, overlay_second.camera_y1
        ):
            raise G17ActuatorIRError("predictor-preserving overlay is nondeterministic")
        if (
            parse_predictor_preserving_overlay_receipt(overlay_first.receipt.to_receipt_bytes())
            != overlay_first.receipt
        ):
            raise G17ActuatorIRError("overlay receipt failed strict parse-back")
        output[local_start:local_stop, 1] = overlay_first.camera_y1
        output_slice = np.ascontiguousarray(output[local_start:local_stop])
        if not np.array_equal(output_slice[:, 0], input_slice[:, 0]):
            raise G17ActuatorIRError("actuator changed Y0")
        if not np.array_equal(
            output_slice[:, 1][~overlay_first.owned_camera_mask],
            input_slice[:, 1][~overlay_first.owned_camera_mask],
        ):
            raise G17ActuatorIRError("actuator changed G-unowned Y1 values")
        step_rows.append(
            G17ActuatorStepExecutionV1(
                operand_id=operand.operand_id,
                pair_start=operand.pair_start,
                pair_count=operand.pair_count,
                operand_sha256=operand.operand_sha256,
                predictor_slice_binding_sha256=slice_state.binding_sha256,
                input_state_sha256=_state_content_sha256(slice_state, input_slice, slice_state.labels),
                output_state_sha256=_state_content_sha256(slice_state, output_slice, decoded_first.labels),
                corrected_labels_sha256=_array_sha256(decoded_first.labels),
                ownership_mask_sha256=ownership.ownership_mask_sha256,
                overlay_receipt_sha256=overlay_first.receipt.receipt_sha256,
                changed_semantic_cells=ownership.changed_cells,
                owned_camera_values=overlay_first.receipt.owned_camera_values,
                actually_changed_camera_values=overlay_first.receipt.actually_changed_camera_values,
            )
        )
        decoded_rows.append(decoded_first)
        ownership_rows.append(ownership)
        overlay_rows.append(overlay_first)

    input_y0 = np.ascontiguousarray(chronology[:, 0])
    input_y1 = np.ascontiguousarray(chronology[:, 1])
    output_y0 = np.ascontiguousarray(output[:, 0])
    output_y1 = np.ascontiguousarray(output[:, 1])
    if not np.array_equal(input_y0, output_y0):
        raise G17ActuatorIRError("composed actuator program changed chronological Y0")
    receipt = G17ActuatorExecutionReceiptV1(
        schema=EXECUTION_SCHEMA,
        program_sha256=program.program_sha256,
        program_receipt_sha256=program.receipt.receipt_sha256,
        counted_member_sha256=_sha256(program.counted_member_bytes),
        counted_actuator_bytes=program.receipt.counted_actuator_bytes,
        predictor_state_binding_sha256=state.binding_sha256,
        predictor_program_sha256=state.predictor_program_sha256,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        source_archive_sha256=state.source_archive_sha256,
        source_runtime_sha256=state.source_runtime_sha256,
        source_pair_ids=state.source_pair_ids,
        chronological_pair_order=state.source_pair_ids,
        input_chronology_sha256=_array_sha256(chronology),
        input_y0_sha256=_array_sha256(input_y0),
        input_y1_sha256=_array_sha256(input_y1),
        output_chronology_sha256=_array_sha256(output),
        output_y0_sha256=_array_sha256(output_y0),
        output_y1_sha256=_array_sha256(output_y1),
        output_raw_bytes=int(output.nbytes),
        step_receipts=tuple(step_rows),
        semantic_receiver_source_sha256=_source_sha256(_seam_module),
        overlay_receiver_source_sha256=_source_sha256(_overlay_module),
        transport_requirement=GTransportRequirementV2.LABEL_LOCAL.value,
        y0_preserved=True,
        unowned_y1_preserved=True,
        deterministic_double_replay=True,
        scorer_invoked=False,
        score_claim=False,
        candidate_claim=False,
        public_rgb_output_proven=False,
        n600_execution_proven=False,
        teacher_or_gt_payload_bytes=0,
        scorer_weight_payload_bytes=0,
        byte_vm_impersonation=False,
        research_only=True,
    )
    return G17ActuatorExecutionResultV1(
        chronological_frames=output,
        decoded_g_by_operand=tuple(decoded_rows),
        ownership_by_operand=tuple(ownership_rows),
        overlay_by_operand=tuple(overlay_rows),
        receipt=receipt,
    )


def execute_g17_actuator_program_v1(
    program: G17ActuatorProgramV1,
    *,
    ep725_surface: Ep725EphemeralRuntimeSurfaceV2,
) -> G17ActuatorExecutionResultV1:
    """Execute twice and return the first exact bounded ep725 state transition."""

    first = _execute_once(program, ep725_surface)
    second = _execute_once(program, ep725_surface)
    if first.receipt != second.receipt or not np.array_equal(
        first.chronological_frames,
        second.chronological_frames,
    ):
        raise G17ActuatorIRError("whole actuator execution is nondeterministic")
    return first


def parse_g17_actuator_execution_receipt(payload: bytes) -> G17ActuatorExecutionReceiptV1:
    value = _decode_canonical_json(
        payload,
        expected_fields={row.name for row in fields(G17ActuatorExecutionReceiptV1)},
        schema=EXECUTION_SCHEMA,
    )
    try:
        value["source_pair_ids"] = tuple(value["source_pair_ids"])
        value["chronological_pair_order"] = tuple(value["chronological_pair_order"])
        value["step_receipts"] = tuple(G17ActuatorStepExecutionV1(**row) for row in value["step_receipts"])
        receipt = G17ActuatorExecutionReceiptV1(**value)
    except (KeyError, TypeError, ValueError) as exc:
        raise G17ActuatorIRError("execution receipt values failed exact typed validation") from exc
    if receipt.to_receipt_bytes() != payload:
        raise G17ActuatorIRError("execution receipt changed on strict typed parse-back")
    return receipt


def reverify_g17_actuator_execution_receipt(
    payload: bytes,
    *,
    program: G17ActuatorProgramV1,
    ep725_surface: Ep725EphemeralRuntimeSurfaceV2,
) -> G17ActuatorExecutionReceiptV1:
    parsed = parse_g17_actuator_execution_receipt(payload)
    replay = execute_g17_actuator_program_v1(program, ep725_surface=ep725_surface)
    if parsed != replay.receipt or payload != replay.receipt.to_receipt_bytes():
        raise G17ActuatorIRError("execution receipt differs from exact receiver replay")
    return parsed


@dataclass(frozen=True, slots=True)
class G17ActuatorCheckpointReceiptV1:
    schema: Literal["tac.g17_actuator_checkpoint_receipt.v1"]
    stage: G17ActuatorCheckpointStageV1
    sequence_index: int
    operand_id: str
    completed_pair_ids: tuple[int, ...]
    program_sha256: str
    program_receipt_sha256: str
    execution_receipt_sha256: str
    input_state_sha256: str
    output_state_sha256: str
    previous_checkpoint_sha256: str | None
    atomic_write_required: Literal[True]
    distinct_stage_filename_required: Literal[True]
    resumable_from_disk_contract: Literal[True]
    research_only: Literal[True]

    def __post_init__(self) -> None:
        if self.schema != CHECKPOINT_SCHEMA or type(self.stage) is not G17ActuatorCheckpointStageV1:
            raise G17ActuatorIRError("checkpoint schema or stage changed")
        if type(self.sequence_index) is not int or self.sequence_index < 0:
            raise G17ActuatorIRError("checkpoint sequence index must be nonnegative exact integer")
        _require_id(self.operand_id, name="checkpoint.operand_id")
        if (
            type(self.completed_pair_ids) is not tuple
            or not self.completed_pair_ids
            or any(type(value) is not int for value in self.completed_pair_ids)
            or self.completed_pair_ids != tuple(range(self.completed_pair_ids[0], self.completed_pair_ids[-1] + 1))
        ):
            raise G17ActuatorIRError("checkpoint completed pairs must be one exact chronological prefix")
        for name in (
            "program_sha256",
            "program_receipt_sha256",
            "execution_receipt_sha256",
            "input_state_sha256",
            "output_state_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.previous_checkpoint_sha256 is not None:
            _require_sha256(self.previous_checkpoint_sha256, name="previous_checkpoint_sha256")
        if any(
            getattr(self, name) is not True
            for name in (
                "atomic_write_required",
                "distinct_stage_filename_required",
                "resumable_from_disk_contract",
                "research_only",
            )
        ):
            raise G17ActuatorIRError("checkpoint durability truth contract changed")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stage"] = self.stage.value
        value["completed_pair_ids"] = list(self.completed_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def build_g17_actuator_checkpoints(
    result: G17ActuatorExecutionResultV1,
) -> tuple[G17ActuatorCheckpointReceiptV1, ...]:
    if type(result) is not G17ActuatorExecutionResultV1:
        raise G17ActuatorIRError("checkpoint builder requires exact actuator execution result")
    checkpoints: list[G17ActuatorCheckpointReceiptV1] = []
    completed: list[int] = []
    previous: str | None = None
    for index, step in enumerate(result.receipt.step_receipts):
        completed.extend(range(step.pair_start, step.pair_start + step.pair_count))
        checkpoint = G17ActuatorCheckpointReceiptV1(
            schema=CHECKPOINT_SCHEMA,
            stage=G17ActuatorCheckpointStageV1.PAIR_OUTPUT_REALIZED,
            sequence_index=index,
            operand_id=step.operand_id,
            completed_pair_ids=tuple(completed),
            program_sha256=result.receipt.program_sha256,
            program_receipt_sha256=result.receipt.program_receipt_sha256,
            execution_receipt_sha256=result.receipt.receipt_sha256,
            input_state_sha256=step.input_state_sha256,
            output_state_sha256=step.output_state_sha256,
            previous_checkpoint_sha256=previous,
            atomic_write_required=True,
            distinct_stage_filename_required=True,
            resumable_from_disk_contract=True,
            research_only=True,
        )
        checkpoints.append(checkpoint)
        previous = checkpoint.receipt_sha256
    return tuple(checkpoints)


def parse_g17_actuator_checkpoint_receipt(payload: bytes) -> G17ActuatorCheckpointReceiptV1:
    value = _decode_canonical_json(
        payload,
        expected_fields={row.name for row in fields(G17ActuatorCheckpointReceiptV1)},
        schema=CHECKPOINT_SCHEMA,
    )
    try:
        value["stage"] = G17ActuatorCheckpointStageV1(value["stage"])
        value["completed_pair_ids"] = tuple(value["completed_pair_ids"])
        receipt = G17ActuatorCheckpointReceiptV1(**value)
    except (KeyError, TypeError, ValueError) as exc:
        raise G17ActuatorIRError("checkpoint receipt values failed exact typed validation") from exc
    if receipt.to_receipt_bytes() != payload:
        raise G17ActuatorIRError("checkpoint receipt changed on strict typed parse-back")
    return receipt


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXECUTION_SCHEMA",
    "PROGRAM_SCHEMA",
    "G17ActuatorCheckpointReceiptV1",
    "G17ActuatorCheckpointStageV1",
    "G17ActuatorExecutionReceiptV1",
    "G17ActuatorExecutionResultV1",
    "G17ActuatorIRError",
    "G17ActuatorKindV1",
    "G17ActuatorOperandRefV1",
    "G17ActuatorPhysicalSpanGroupV1",
    "G17ActuatorProgramReceiptV1",
    "G17ActuatorProgramV1",
    "G17ActuatorReceiverOperationV1",
    "G17ActuatorStepExecutionV1",
    "build_g17_actuator_checkpoints",
    "execute_g17_actuator_program_v1",
    "parse_g17_actuator_checkpoint_receipt",
    "parse_g17_actuator_execution_receipt",
    "parse_g17_actuator_program_receipt",
    "reverify_g17_actuator_execution_receipt",
    "reverify_g17_actuator_program_receipt",
]
