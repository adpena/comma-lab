# SPDX-License-Identifier: MIT
"""Deterministic bounded encoder from frozen target debt to counted G atoms.

The decoder-facing object remains the existing finite G grammar.  Frozen
``lstars`` labels, target tables, PBR streams, and dense RGB witnesses are
encoder-only inputs.  This module supplies the missing acquisition seam: it
turns a bounded predictor/target difference into one-pixel-high topology-event
row spans, compiles those spans through the strict G codec, and proves exact
semantic reconstruction by decoding the counted packet.

The row-span construction is an exact, intentionally unsophisticated control.
It is not claimed to be rate optimal, scorer closed, or candidate promotable.
Its purpose is to make the residual compiler real and measurable so richer V9
and V10 atoms can be compared against the same exact debt and byte baseline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl.generative_taskspace_correction import (
    CompiledGenerativeCorrectionV1,
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    GenerativeTaskspaceCorrectionError,
    PredictorSemanticStateV1,
    compile_generative_taskspace_correction,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    TaskspacePredictorStateV2,
    TaskspacePredictorStateV2Error,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    TaskspacePredictorV2ConsumerSeamError,
    compile_generative_taskspace_correction_v2,
)

SCHEMA: Final = "tac.bounded_target_g_encoder_receipt.v1"
ENCODER_ID: Final = "exact_target_debt_to_one_row_topology_spans.v1"
MAX_TOPOLOGY_EVENTS: Final = 0xFFFF
_ROLE_BY_CLASS: Final = {class_id: role for role, class_id in ROLE_CLASS_IDS.items()}
_PRECEDENCE: Final = {role: index for index, role in enumerate(REALIZATION_PAINT_ORDER)}


class BoundedTargetGEncoderError(ValueError):
    """Target custody, finite plan, or exact compile proof failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BoundedTargetGEncoderError(f"{field} must be canonical lowercase SHA-256 hex")
    return value


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
        raise BoundedTargetGEncoderError("encoder evidence is not finite canonical ASCII JSON") from exc


def _evidence_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json({"schema": schema, **dict(payload)}))


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


@dataclass(frozen=True, slots=True)
class FrozenTargetSliceCustodyV1:
    """Identity of the encoder-only frozen target slice used for one fit."""

    cache_sha256: str
    member_name: str
    source_pair_ids: tuple[int, ...]
    target_labels_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.cache_sha256, field="cache_sha256")
        _require_sha256(self.target_labels_sha256, field="target_labels_sha256")
        if type(self.member_name) is not str or not self.member_name or self.member_name != "lstars":
            raise BoundedTargetGEncoderError("frozen target member must be exact lstars")
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise BoundedTargetGEncoderError("target custody requires a nonempty exact pair tuple")
        if any(type(value) is not int for value in self.source_pair_ids):
            raise BoundedTargetGEncoderError("target custody pair IDs must be exact integers")
        expected = tuple(range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids)))
        if self.source_pair_ids != expected:
            raise BoundedTargetGEncoderError("target custody pair IDs must be one contiguous window")

    @property
    def binding_sha256(self) -> str:
        return _evidence_sha256(
            "tac.frozen_target_slice_custody.v1",
            {
                "cache_sha256": self.cache_sha256,
                "member_name": self.member_name,
                "source_pair_ids": list(self.source_pair_ids),
                "target_labels_sha256": self.target_labels_sha256,
            },
        )


@dataclass(frozen=True, slots=True)
class TargetTransitionCountV1:
    """Exact encoder-side class-transition census; never candidate payload."""

    source_class: int
    target_class: int
    cells: int

    def __post_init__(self) -> None:
        if (
            type(self.source_class) is not int
            or type(self.target_class) is not int
            or self.source_class not in _ROLE_BY_CLASS
            or self.target_class not in _ROLE_BY_CLASS
            or self.source_class == self.target_class
            or type(self.cells) is not int
            or self.cells < 1
        ):
            raise BoundedTargetGEncoderError("target transition census row is invalid")

    def as_dict(self) -> dict[str, int]:
        return {
            "cells": self.cells,
            "source_class": self.source_class,
            "target_class": self.target_class,
        }


@dataclass(frozen=True, slots=True)
class BoundedTargetGEncoderReceiptV1:
    """Exact control-plane receipt for one bounded real-target G compile."""

    schema: Literal["tac.bounded_target_g_encoder_receipt.v1"]
    encoder_id: Literal["exact_target_debt_to_one_row_topology_spans.v1"]
    predictor_binding_sha256: str
    target_custody_binding_sha256: str
    target_labels_sha256: str
    decoded_labels_sha256: str
    packet_sha256: str
    compile_receipt_binding_sha256: str
    event_plan_sha256: str
    source_pair_ids: tuple[int, ...]
    transition_counts: tuple[TargetTransitionCountV1, ...]
    debt_before_cells: int
    debt_after_cells: Literal[0]
    birth_event_count: int
    death_event_count: int
    total_topology_events: int
    packet_bytes: int
    serialized_target_table_bytes: Literal[0] = 0
    serialized_pbr_bytes: Literal[0] = 0
    serialized_dense_y_bytes: Literal[0] = 0
    acquisition_lineage: Literal["exact_target_derived_row_span_control_encoder_only.v1"] = (
        "exact_target_derived_row_span_control_encoder_only.v1"
    )
    counted_payload_kind: Literal["finite_topology_event_parameters_plus_realization_profile"] = (
        "finite_topology_event_parameters_plus_realization_profile"
    )
    exact_semantic_target_reconstructed: Literal[True] = True
    deterministic_strict_decode_verified: Literal[True] = True
    arbitrary_segment_pose_rate_caps_applied: Literal[False] = False
    scorer_invoked: Literal[False] = False
    through_r_realization_verified: Literal[False] = False
    exact_score_claim: Literal[False] = False
    rate_optimality_claim: Literal[False] = False
    compiled_g_generic_eligibility_superseded_by_acquisition_lineage: Literal[True] = True
    candidate_archive_eligible: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.encoder_id != ENCODER_ID:
            raise BoundedTargetGEncoderError("encoder receipt schema or algorithm identity changed")
        for field in (
            "predictor_binding_sha256",
            "target_custody_binding_sha256",
            "target_labels_sha256",
            "decoded_labels_sha256",
            "packet_sha256",
            "compile_receipt_binding_sha256",
            "event_plan_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise BoundedTargetGEncoderError("encoder receipt lost its exact source-pair window")
        if type(self.transition_counts) is not tuple or any(
            type(row) is not TargetTransitionCountV1 for row in self.transition_counts
        ):
            raise BoundedTargetGEncoderError("encoder receipt transition census is not typed")
        integers = (
            self.debt_before_cells,
            self.debt_after_cells,
            self.birth_event_count,
            self.death_event_count,
            self.total_topology_events,
            self.packet_bytes,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise BoundedTargetGEncoderError("encoder receipt counts must be exact nonnegative integers")
        if (
            self.debt_before_cells < 1
            or self.debt_after_cells != 0
            or sum(row.cells for row in self.transition_counts) != self.debt_before_cells
            or self.total_topology_events != self.birth_event_count + self.death_event_count
            or not 1 <= self.total_topology_events <= MAX_TOPOLOGY_EVENTS
            or self.packet_bytes < 1
        ):
            raise BoundedTargetGEncoderError("encoder receipt arithmetic does not close")
        fixed_truth = (
            self.serialized_target_table_bytes == 0
            and self.serialized_pbr_bytes == 0
            and self.serialized_dense_y_bytes == 0
            and self.acquisition_lineage == "exact_target_derived_row_span_control_encoder_only.v1"
            and self.counted_payload_kind == "finite_topology_event_parameters_plus_realization_profile"
            and self.exact_semantic_target_reconstructed is True
            and self.deterministic_strict_decode_verified is True
            and self.arbitrary_segment_pose_rate_caps_applied is False
            and self.scorer_invoked is False
            and self.through_r_realization_verified is False
            and self.exact_score_claim is False
            and self.rate_optimality_claim is False
            and self.compiled_g_generic_eligibility_superseded_by_acquisition_lineage is True
            and self.candidate_archive_eligible is False
            and self.promotion_eligible is False
            and self.research_only is True
        )
        if not fixed_truth:
            raise BoundedTargetGEncoderError("encoder receipt authority labels were relaxed")

    def as_dict(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["source_pair_ids"] = list(self.source_pair_ids)
        result["transition_counts"] = [row.as_dict() for row in self.transition_counts]
        return result

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_bounded_target_g_encoder_receipt(payload: bytes) -> BoundedTargetGEncoderReceiptV1:
    """Strictly parse and byte-reemit one canonical encoder-control receipt."""

    if type(payload) is not bytes or not payload:
        raise BoundedTargetGEncoderError("bounded target G receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BoundedTargetGEncoderError(f"bounded target G receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except BoundedTargetGEncoderError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundedTargetGEncoderError("bounded target G receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(BoundedTargetGEncoderReceiptV1.__dataclass_fields__):
        raise BoundedTargetGEncoderError("bounded target G receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise BoundedTargetGEncoderError("bounded target G receipt is not canonical JSON")
    pair_ids = value["source_pair_ids"]
    raw_transitions = value["transition_counts"]
    if type(pair_ids) is not list or any(type(pair_id) is not int for pair_id in pair_ids):
        raise BoundedTargetGEncoderError("bounded target G receipt pair IDs are not exact integers")
    if type(raw_transitions) is not list:
        raise BoundedTargetGEncoderError("bounded target G transition census must be one JSON list")
    transition_fields = set(TargetTransitionCountV1.__dataclass_fields__)
    transitions: list[TargetTransitionCountV1] = []
    for index, row in enumerate(raw_transitions):
        if type(row) is not dict or set(row) != transition_fields:
            raise BoundedTargetGEncoderError(f"bounded target G transition[{index}] fields changed")
        if any(type(row[field]) is not int for field in transition_fields):
            raise BoundedTargetGEncoderError(f"bounded target G transition[{index}] values are not exact integers")
        transitions.append(TargetTransitionCountV1(**row))
    arguments = dict(value)
    arguments["source_pair_ids"] = tuple(pair_ids)
    arguments["transition_counts"] = tuple(transitions)
    try:
        receipt = BoundedTargetGEncoderReceiptV1(**arguments)
    except TypeError as exc:
        raise BoundedTargetGEncoderError("bounded target G receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise BoundedTargetGEncoderError("bounded target G receipt changed on typed parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class CompiledBoundedTargetGV1:
    """Counted G bytes plus the encoder-only exact-control receipt."""

    compiled: CompiledGenerativeCorrectionV1
    receipt: BoundedTargetGEncoderReceiptV1

    def __post_init__(self) -> None:
        if type(self.compiled) is not CompiledGenerativeCorrectionV1:
            raise BoundedTargetGEncoderError("bounded target compile result lost its strict G object")
        if type(self.receipt) is not BoundedTargetGEncoderReceiptV1:
            raise BoundedTargetGEncoderError("bounded target compile result lost its typed receipt")
        if self.compiled.receipt.packet_sha256 != self.receipt.packet_sha256:
            raise BoundedTargetGEncoderError("bounded target receipt differs from exact compiled G bytes")


def _row_runs(mask: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    """Return canonical ``(y,x0,x1)`` runs for one exact boolean mask."""

    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != (384, 512):
        raise BoundedTargetGEncoderError("row-run input changed scorer-grid boolean geometry")
    rows: list[tuple[int, int, int]] = []
    for y in range(array.shape[0]):
        padded = np.empty(array.shape[1] + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = array[y]
        edges = np.flatnonzero(np.diff(padded))
        for x0, x1 in edges.reshape(-1, 2):
            rows.append((y, int(x0), int(x1)))
    return tuple(rows)


def _transition_counts(predictor: np.ndarray, target: np.ndarray) -> tuple[TargetTransitionCountV1, ...]:
    rows: list[TargetTransitionCountV1] = []
    for source_class in sorted(_ROLE_BY_CLASS):
        for target_class in sorted(_ROLE_BY_CLASS):
            if source_class == target_class:
                continue
            count = int(np.count_nonzero((predictor == source_class) & (target == target_class)))
            if count:
                rows.append(TargetTransitionCountV1(source_class, target_class, count))
    return tuple(rows)


def _event_plan(
    predictor_state: PredictorSemanticStateV1 | TaskspacePredictorStateV2,
    target_labels: np.ndarray,
) -> tuple[tuple[TopologyEventV1, ...], int, int]:
    events: list[TopologyEventV1] = []
    births = 0
    deaths = 0
    for local_pair, source_pair_id in enumerate(predictor_state.source_pair_ids):
        predictor = predictor_state.labels[local_pair]
        target = target_labels[local_pair]
        mismatch = predictor != target

        # One target-role birth run can cover adjacent cells regardless of the
        # original class.  This is the smallest exact horizontal-span control.
        for target_class in sorted(_ROLE_BY_CLASS):
            target_role = _ROLE_BY_CLASS[target_class]
            for y, x0, x1 in _row_runs(mismatch & (target == target_class)):
                events.append(
                    TopologyEventV1(
                        pair_index=source_pair_id,
                        role=target_role,
                        action="birth",
                        shape="box",
                        lifetime=1,
                        y0=y,
                        x0=x0,
                        y1=y + 1,
                        x1=x1,
                    )
                )
                births += 1

        # The receiver composites role masks in a fixed precedence order.  A
        # source class above the target would hide that target birth, so clear
        # exactly those source-role spans.  Lower-precedence source masks may
        # remain because the target birth deterministically wins compositing.
        for source_class in sorted(_ROLE_BY_CLASS):
            source_role = _ROLE_BY_CLASS[source_class]
            for target_class in sorted(_ROLE_BY_CLASS):
                if source_class == target_class:
                    continue
                target_role = _ROLE_BY_CLASS[target_class]
                if _PRECEDENCE[source_role] <= _PRECEDENCE[target_role]:
                    continue
                clear_mask = (predictor == source_class) & (target == target_class)
                for y, x0, x1 in _row_runs(clear_mask):
                    events.append(
                        TopologyEventV1(
                            pair_index=source_pair_id,
                            role=source_role,
                            action="death",
                            shape="box",
                            lifetime=1,
                            y0=y,
                            x0=x0,
                            y1=y + 1,
                            x1=x1,
                        )
                    )
                    deaths += 1
        if len(events) > MAX_TOPOLOGY_EVENTS:
            raise BoundedTargetGEncoderError(
                "exact row-span control exceeds the uint16 G topology-event ABI; a richer atom family is required"
            )
    return tuple(events), births, deaths


def _event_plan_sha256(events: Sequence[TopologyEventV1]) -> str:
    return _evidence_sha256(
        "tac.bounded_target_g_event_plan.v1",
        {
            "events": [
                {
                    "action": row.action,
                    "lifetime": row.lifetime,
                    "pair_index": row.pair_index,
                    "role": row.role,
                    "shape": row.shape,
                    "transport_gain_x_q4": row.transport_gain_x_q4,
                    "transport_gain_y_q4": row.transport_gain_y_q4,
                    "x0": row.x0,
                    "x1": row.x1,
                    "y0": row.y0,
                    "y1": row.y1,
                }
                for row in events
            ]
        },
    )


def _compile_bounded_target_g_impl(
    predictor_state: PredictorSemanticStateV1 | TaskspacePredictorStateV2,
    target_labels: np.ndarray,
    *,
    target_custody: FrozenTargetSliceCustodyV1,
    realization_profile: ReceiverRealizationProfileV1,
    state_version: Literal["v1", "v2"],
) -> CompiledBoundedTargetGV1:
    if state_version == "v1":
        if type(predictor_state) is not PredictorSemanticStateV1:
            raise BoundedTargetGEncoderError("V1 bounded compile requires exact PredictorSemanticStateV1")
    elif state_version == "v2":
        if type(predictor_state) is not TaskspacePredictorStateV2:
            raise BoundedTargetGEncoderError("V2 bounded compile requires exact TaskspacePredictorStateV2")
    else:
        raise BoundedTargetGEncoderError("bounded target compiler state version escaped the closed dispatch")
    if type(target_custody) is not FrozenTargetSliceCustodyV1:
        raise BoundedTargetGEncoderError("target_custody must be one typed frozen slice identity")
    if type(realization_profile) is not ReceiverRealizationProfileV1:
        raise BoundedTargetGEncoderError("realization_profile must use the counted receiver type")
    target = np.ascontiguousarray(target_labels)
    if target.dtype != np.uint8 or target.shape != predictor_state.labels.shape:
        raise BoundedTargetGEncoderError("target labels differ from the predictor pair/scorer geometry")
    if int(target.min()) < 0 or int(target.max()) > 4:
        raise BoundedTargetGEncoderError("target labels escaped the five-class universe")
    target_sha = _array_sha256(target)
    if (
        target_custody.source_pair_ids != predictor_state.source_pair_ids
        or target_custody.target_labels_sha256 != target_sha
    ):
        raise BoundedTargetGEncoderError("frozen target custody does not bind the exact predictor window/bytes")
    debt = int(np.count_nonzero(predictor_state.labels != target))
    if debt == 0:
        raise BoundedTargetGEncoderError("bounded target G control refuses a zero-debt no-op")

    transitions = _transition_counts(predictor_state.labels, target)
    events, births, deaths = _event_plan(predictor_state, target)
    if not events or len(events) > MAX_TOPOLOGY_EVENTS:
        raise BoundedTargetGEncoderError("bounded target event plan is empty or outside the strict G ABI")
    event_plan_sha = _event_plan_sha256(events)
    transition_payload = [row.as_dict() for row in transitions]

    evidence = EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256=_evidence_sha256(
            "tac.bounded_target_g_predictor_debt.v1",
            {
                "debt_cells": debt,
                "predictor_binding_sha256": predictor_state.binding_sha256,
                "target_custody_binding_sha256": target_custody.binding_sha256,
            },
        ),
        pbr2_sha256=_evidence_sha256(
            "tac.bounded_target_g_row_span_teacher.v1",
            {
                "birth_event_count": births,
                "death_event_count": deaths,
                "event_plan_sha256": event_plan_sha,
            },
        ),
        target_labels_sha256=target_sha,
        obligation_ir_sha256=_evidence_sha256(
            "tac.bounded_target_g_transition_obligations.v1",
            {"transition_counts": transition_payload},
        ),
        oracle_evidence_sha256=_evidence_sha256(
            "tac.bounded_target_g_exact_semantic_oracle.v1",
            {
                "expected_debt_after_cells": 0,
                "event_plan_sha256": event_plan_sha,
                "target_labels_sha256": target_sha,
            },
        ),
        dense_y_sha256=_evidence_sha256(
            "tac.bounded_target_g_dense_y_absence.v1",
            {
                "dense_y_present": False,
                "reason": "semantic_row_span_control_requires_no_dense_rgb_teacher",
            },
        ),
        target_labels=target,
        teacher_event_count=debt,
    )
    program = GenerativeCorrectionProgramV1(
        topology_events=events,
        realization_profile=realization_profile,
    )
    try:
        if state_version == "v1":
            assert type(predictor_state) is PredictorSemanticStateV1
            compiled = compile_generative_taskspace_correction(
                predictor_state,
                program,
                teacher_evidence=evidence,
            )
        else:
            assert type(predictor_state) is TaskspacePredictorStateV2
            compiled = compile_generative_taskspace_correction_v2(
                predictor_state,
                program,
                teacher_evidence=evidence,
            )
    except (
        GenerativeTaskspaceCorrectionError,
        TaskspacePredictorStateV2Error,
        TaskspacePredictorV2ConsumerSeamError,
    ) as exc:
        raise BoundedTargetGEncoderError("strict G compiler refused the bounded target plan") from exc
    decoded_sha = _array_sha256(compiled.decoded.labels)
    if (
        compiled.receipt.debt_before_cells != debt
        or compiled.receipt.debt_after_cells != 0
        or not compiled.receipt.exact_semantic_target_reconstructed
        or decoded_sha != target_sha
        or not np.array_equal(compiled.decoded.labels, target)
    ):
        raise BoundedTargetGEncoderError("counted G row-span control did not reconstruct exact target semantics")

    receipt = BoundedTargetGEncoderReceiptV1(
        schema=SCHEMA,
        encoder_id=ENCODER_ID,
        predictor_binding_sha256=predictor_state.binding_sha256,
        target_custody_binding_sha256=target_custody.binding_sha256,
        target_labels_sha256=target_sha,
        decoded_labels_sha256=decoded_sha,
        packet_sha256=compiled.receipt.packet_sha256,
        compile_receipt_binding_sha256=compiled.receipt_binding_sha256,
        event_plan_sha256=event_plan_sha,
        source_pair_ids=predictor_state.source_pair_ids,
        transition_counts=transitions,
        debt_before_cells=debt,
        debt_after_cells=0,
        birth_event_count=births,
        death_event_count=deaths,
        total_topology_events=len(events),
        packet_bytes=len(compiled.packet),
    )
    return CompiledBoundedTargetGV1(compiled=compiled, receipt=receipt)


def compile_bounded_target_g(
    predictor_state: PredictorSemanticStateV1,
    target_labels: np.ndarray,
    *,
    target_custody: FrozenTargetSliceCustodyV1,
    realization_profile: ReceiverRealizationProfileV1,
) -> CompiledBoundedTargetGV1:
    """Compile exact bounded target debt from an exact legacy V1 predictor.

    This performs no scorer call and applies no independent Seg/Pose/rate cap.
    The result is an exact semantic upper-bound/control for subsequent
    rate-distortion atom selection, not an admitted archive.
    """

    return _compile_bounded_target_g_impl(
        predictor_state,
        target_labels,
        target_custody=target_custody,
        realization_profile=realization_profile,
        state_version="v1",
    )


def compile_bounded_target_g_v2(
    predictor_state: TaskspacePredictorStateV2,
    target_labels: np.ndarray,
    *,
    target_custody: FrozenTargetSliceCustodyV1,
    realization_profile: ReceiverRealizationProfileV1,
) -> CompiledBoundedTargetGV1:
    """Compile exact bounded target debt from V2 without fabricating Pose6.

    The row-span event plan has zero transport gains, so the V2 G seam admits
    it from exact decoder-owned labels under ``NoTransportV2``.  This function
    never projects through ``as_v1`` and never manufactures a zero Pose6 table.
    """

    return _compile_bounded_target_g_impl(
        predictor_state,
        target_labels,
        target_custody=target_custody,
        realization_profile=realization_profile,
        state_version="v2",
    )


__all__ = [
    "ENCODER_ID",
    "MAX_TOPOLOGY_EVENTS",
    "SCHEMA",
    "BoundedTargetGEncoderError",
    "BoundedTargetGEncoderReceiptV1",
    "CompiledBoundedTargetGV1",
    "FrozenTargetSliceCustodyV1",
    "TargetTransitionCountV1",
    "compile_bounded_target_g",
    "compile_bounded_target_g_v2",
    "parse_bounded_target_g_encoder_receipt",
]
