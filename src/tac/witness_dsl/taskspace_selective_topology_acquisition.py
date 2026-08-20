# SPDX-License-Identifier: MIT
"""Encoder-only selective acquisition of canonical row-3 topology debt.

This module compiles bounded subsets of ``Z != T, H != T`` through the
existing counted ``TACG1C`` grammar.  Dense labels and post-forward arrays are
encoder evidence only.  The module never invokes a scorer, receiver, archive
builder, evaluator, or whole-object allocator and makes no score or candidate
claim.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from functools import partial
from typing import Any, Final, Literal, TypeAlias

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
    apply_generative_taskspace_correction,
    compile_generative_taskspace_correction,
    parse_generative_taskspace_correction,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    MAX_BOUNDED_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    COPY_ROW as A3_COPY_ROW,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_HEADER as A3_PACKET_HEADER,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_MAGIC as A3_PACKET_MAGIC,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_VERSION as A3_PACKET_VERSION,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    RGB_ROW as A3_RGB_ROW,
)
from tac.witness_dsl.taskspace_pass_conditional_a import PACKET_MAGIC as PASS_A_PACKET_MAGIC
from tac.witness_dsl.taskspace_pass_semantic_g import ENVELOPE_MAGIC as PASS_G_ENVELOPE_MAGIC
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_HEADER as POST_G8_A_PACKET_HEADER,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_MAGIC as POST_G8_A_PACKET_MAGIC,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_VERSION as POST_G8_A_PACKET_VERSION,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    apply_generative_taskspace_correction_v2,
    compile_generative_taskspace_correction_v2,
    parse_generative_taskspace_correction_v2,
)
from tac.witness_dsl.taskspace_same_class_realization_encoder import (
    SameClassRealizationCellPartitionTelemetryV1,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    COMPOSITE_G_SECTION_MAGIC,
    SEMANTIC_G_PACKET_MAGIC,
    SameClassRealizationRepairError,
    parse_same_class_realization_repair_g_section,
)
from tac.witness_dsl.taskspace_whole_archive_allocator import (
    TaskspaceSectionBundleV1,
    TaskspaceWholeArchiveProposalV1,
)

EVIDENCE_SCHEMA: Final = "tac.encoder_only_selective_topology_evidence.v1"
PROPOSAL_RECEIPT_SCHEMA: Final = "tac.selective_topology_program_proposal_receipt.v1"
ACQUISITION_RECEIPT_SCHEMA: Final = "tac.selective_topology_acquisition_receipt.v1"
FRESH_G8_RECEIPT_SCHEMA: Final = "tac.selective_topology_fresh_g8_request_receipt.v1"
ACQUISITION_SCHEMA: Final = "tac.selective_topology_acquisition.v1"
MAX_TOPOLOGY_EVENTS: Final = 0xFFFF
# Every singleton may require one birth and one precedence death.  All emitted
# families/orders therefore remain representable under this universal bound.
MAX_SINGLETON_PREFIX_CELLS: Final = MAX_TOPOLOGY_EVENTS // 2
MAX_PREFIX_REQUESTS: Final = 64
MAX_PREFIX_REQUEST_VALUE: Final = 0xFFFFFFFF

_CLASS_COUNT: Final = 5
_ROLE_BY_CLASS: Final = {class_id: role for role, class_id in ROLE_CLASS_IDS.items()}
_PRECEDENCE: Final = {role: index for index, role in enumerate(REALIZATION_PAINT_ORDER)}


class SelectiveTopologyBlockerCodeV1(StrEnum):
    """Typed fail-closed reasons required by the frozen acquisition contract."""

    EMPTY_TOPOLOGY_DEBT = "EMPTY_TOPOLOGY_DEBT"
    EMPTY_RESOLVED_PREFIX = "EMPTY_RESOLVED_PREFIX"
    PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH = "PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH"
    TACG1C_EVENT_CARDINALITY_OVERFLOW = "TACG1C_EVENT_CARDINALITY_OVERFLOW"
    TACG1C_COMPILE_PARSE_APPLY_REFUSAL = "TACG1C_COMPILE_PARSE_APPLY_REFUSAL"
    DECODED_MUTATION_OUTSIDE_SELECTED_ROW3 = "DECODED_MUTATION_OUTSIDE_SELECTED_ROW3"
    BASELINE_FORWARD_RECEIPT_REUSED = "BASELINE_FORWARD_RECEIPT_REUSED"
    POST_FORWARD_CUSTODY_MISMATCH = "POST_FORWARD_CUSTODY_MISMATCH"
    G7_WHOLE_BUNDLE_REFUSAL = "G7_WHOLE_BUNDLE_REFUSAL"


class SelectiveTopologyAcquisitionError(ValueError):
    """One typed selective-acquisition or lazy whole-bundle proof failed."""

    def __init__(self, code: SelectiveTopologyBlockerCodeV1, message: str) -> None:
        if type(code) is not SelectiveTopologyBlockerCodeV1:
            raise TypeError("selective topology blocker code must use the closed V1 enum")
        self.code = code
        super().__init__(f"{code.value}: {message}")


class SelectiveTopologyPrefixOrderV1(StrEnum):
    """Closed bounded-prefix ordering priors; none is an admission rule."""

    CANONICAL_ADDRESS_V1 = "CANONICAL_ADDRESS_V1"
    FORTUNATE_CLEARANCE_DESCENDING_V1 = "FORTUNATE_CLEARANCE_DESCENDING_V1"
    TRANSITION_MASS_DESCENDING_V1 = "TRANSITION_MASS_DESCENDING_V1"


class SelectiveTopologyGrammarFamilyV1(StrEnum):
    """Three exact TACG1C box grammars for one selected semantic support."""

    SINGLETON_BOX_CONTROL_V1 = "SINGLETON_BOX_CONTROL_V1"
    SOURCE_TARGET_ROW_RUN_BOX_V1 = "SOURCE_TARGET_ROW_RUN_BOX_V1"
    TARGET_BIRTH_SOURCE_DEATH_ROW_RUN_BOX_V1 = "TARGET_BIRTH_SOURCE_DEATH_ROW_RUN_BOX_V1"


class SelectiveTopologyG12SeamStatusV1(StrEnum):
    """Frozen G12 cannot yet name the selective semantic-G base honestly."""

    G12_SELECTIVE_TACG1C_BASE_INTERPRETATION_OWED = "G12_SELECTIVE_TACG1C_BASE_INTERPRETATION_OWED"


class SelectiveTopologyProductionEnvelopeStatusV1(StrEnum):
    """Candidate-ineligibility status of the current diagnostic G/A domain."""

    PRODUCTION_SELECTIVE_TOPOLOGY_ENVELOPE_OWED = "PRODUCTION_SELECTIVE_TOPOLOGY_ENVELOPE_OWED"


class SelectiveTopologyG7DomainV1(StrEnum):
    """Existing G/A domains accepted by the lazy structural hook."""

    PLAIN_TACG1C_TACA3P1 = "PLAIN_TACG1C_TACA3P1"
    COMPOSITE_TACG8S1_TACA8P1 = "COMPOSITE_TACG8S1_TACA8P1"


_PREFIX_ORDERS: Final = tuple(SelectiveTopologyPrefixOrderV1)
_GRAMMAR_FAMILIES: Final = tuple(SelectiveTopologyGrammarFamilyV1)
_PRODUCTION_BLOCKER: Final = SelectiveTopologyProductionEnvelopeStatusV1.PRODUCTION_SELECTIVE_TOPOLOGY_ENVELOPE_OWED
_G12_BLOCKER: Final = SelectiveTopologyG12SeamStatusV1.G12_SELECTIVE_TACG1C_BASE_INTERPRETATION_OWED

PredictorStateV1OrV2: TypeAlias = PredictorSemanticStateV1 | TaskspacePredictorStateV2
_PREDICTOR_STATE_V1: Final = "PREDICTOR_SEMANTIC_STATE_V1"
_PREDICTOR_STATE_V2: Final = "TASKSPACE_PREDICTOR_STATE_V2"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


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
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "record is not finite canonical ASCII JSON",
        ) from exc


def _require_sha256(value: object, field: str, *, code: SelectiveTopologyBlockerCodeV1) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectiveTopologyAcquisitionError(code, f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int,
    code: SelectiveTopologyBlockerCodeV1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SelectiveTopologyAcquisitionError(code, f"{field} must be an exact integer in [{minimum},{maximum}]")
    return value


def _validate_pair_ids(source_pair_ids: tuple[int, ...]) -> tuple[int, ...]:
    code = SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
    if (
        type(source_pair_ids) is not tuple
        or not 1 <= len(source_pair_ids) <= MAX_BOUNDED_PAIRS
        or any(type(pair_id) is not int for pair_id in source_pair_ids)
    ):
        raise SelectiveTopologyAcquisitionError(code, "source_pair_ids must be an exact nonempty tuple of at most 4")
    expected = tuple(range(source_pair_ids[0], source_pair_ids[0] + len(source_pair_ids)))
    if source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise SelectiveTopologyAcquisitionError(code, "source_pair_ids must be contiguous inside [0,600)")
    return expected


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _five_counts(mask: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int, int]:
    values = tuple(int(np.count_nonzero(mask & (target == class_id))) for class_id in range(_CLASS_COUNT))
    return values[0], values[1], values[2], values[3], values[4]


def _partition_masks(
    semantic: np.ndarray,
    target: np.ndarray,
    realized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    semantic_matches = semantic == target
    realized_matches = realized == target
    return (
        np.ascontiguousarray(semantic_matches & realized_matches, dtype=np.bool_),
        np.ascontiguousarray(semantic_matches & ~realized_matches, dtype=np.bool_),
        np.ascontiguousarray(~semantic_matches & ~realized_matches, dtype=np.bool_),
        np.ascontiguousarray(~semantic_matches & realized_matches, dtype=np.bool_),
    )


def _partition_telemetry(
    source_pair_ids: tuple[int, ...],
    semantic: np.ndarray,
    target: np.ndarray,
    realized: np.ndarray,
) -> SameClassRealizationCellPartitionTelemetryV1:
    closed, realization, topology, fortunate = _partition_masks(semantic, target, realized)
    return SameClassRealizationCellPartitionTelemetryV1(
        source_pair_ids=source_pair_ids,
        label_shape=(len(source_pair_ids), SCORER_HEIGHT, SCORER_WIDTH),
        current_semantic_labels_sha256=_array_sha256(semantic),
        target_labels_sha256=_array_sha256(target),
        realized_labels_sha256=_array_sha256(realized),
        closed_mask_sha256=_array_sha256(closed),
        realization_debt_mask_sha256=_array_sha256(realization),
        topology_debt_mask_sha256=_array_sha256(topology),
        fortunate_semantic_mismatch_mask_sha256=_array_sha256(fortunate),
        closed_cell_count=int(np.count_nonzero(closed)),
        realization_debt_cell_count=int(np.count_nonzero(realization)),
        topology_debt_cell_count=int(np.count_nonzero(topology)),
        fortunate_semantic_mismatch_cell_count=int(np.count_nonzero(fortunate)),
        closed_count_by_target_class=_five_counts(closed, target),
        realization_debt_count_by_target_class=_five_counts(realization, target),
        topology_debt_count_by_target_class=_five_counts(topology, target),
        fortunate_semantic_mismatch_count_by_target_class=_five_counts(fortunate, target),
    )


@dataclass(frozen=True, slots=True)
class EncoderOnlySelectiveTopologyEvidenceV1:
    """Hashed immutable Z/T/baseline-H evidence with no dense serializer."""

    source_pair_ids: tuple[int, ...]
    predictor_state: PredictorStateV1OrV2
    base_p_section_sha256: str
    base_p_section_bytes: int
    current_camera_y1_sha256: str
    target_scorer_rgb_sha256: str
    frozen_scorer_sha256: str
    current_realization_forward_receipt_sha256: str
    target_forward_receipt_sha256: str
    current_semantic_labels_sha256: str
    target_labels_sha256: str
    baseline_realized_labels_sha256: str
    current_semantic_labels: np.ndarray
    target_labels: np.ndarray
    baseline_realized_labels: np.ndarray
    schema: Literal["tac.encoder_only_selective_topology_evidence.v1"] = EVIDENCE_SCHEMA
    serialized_dense_evidence_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
        if self.schema != EVIDENCE_SCHEMA or self.serialized_dense_evidence_bytes != 0:
            raise SelectiveTopologyAcquisitionError(code, "selective topology dense evidence must stay encoder-only")
        pair_ids = _validate_pair_ids(self.source_pair_ids)
        if type(self.predictor_state) not in {PredictorSemanticStateV1, TaskspacePredictorStateV2}:
            raise SelectiveTopologyAcquisitionError(code, "predictor_state must use the exact V1 or V2 semantic type")
        if self.predictor_state.source_pair_ids != pair_ids:
            raise SelectiveTopologyAcquisitionError(code, "predictor and evidence pair windows differ")
        for field in (
            "base_p_section_sha256",
            "current_camera_y1_sha256",
            "target_scorer_rgb_sha256",
            "frozen_scorer_sha256",
            "current_realization_forward_receipt_sha256",
            "target_forward_receipt_sha256",
            "current_semantic_labels_sha256",
            "target_labels_sha256",
            "baseline_realized_labels_sha256",
        ):
            _require_sha256(getattr(self, field), field, code=code)
        _require_exact_int(
            self.base_p_section_bytes,
            "base_p_section_bytes",
            minimum=1,
            maximum=0xFFFFFFFF,
            code=code,
        )
        if self.base_p_section_sha256 != self.predictor_state.predictor_program_sha256:
            raise SelectiveTopologyAcquisitionError(code, "counted P hash differs from predictor-state program bytes")
        if type(self.predictor_state) is TaskspacePredictorStateV2 and self.base_p_section_bytes != len(
            self.predictor_state.predictor_program
        ):
            raise SelectiveTopologyAcquisitionError(code, "counted P bytes differ from the exact V2 predictor program")
        expected_shape = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
        for field, expected_hash in (
            ("current_semantic_labels", self.current_semantic_labels_sha256),
            ("target_labels", self.target_labels_sha256),
            ("baseline_realized_labels", self.baseline_realized_labels_sha256),
        ):
            raw = np.asarray(getattr(self, field))
            if raw.dtype != np.uint8 or raw.shape != expected_shape or int(raw.max()) >= _CLASS_COUNT:
                raise SelectiveTopologyAcquisitionError(code, f"{field} must be uint8 [pair,384,512] classes 0..4")
            immutable = _immutable_array(raw, dtype=np.dtype(np.uint8))
            if _array_sha256(immutable) != expected_hash:
                raise SelectiveTopologyAcquisitionError(code, f"{field} content hash mismatch")
            object.__setattr__(self, field, immutable)
        if not np.array_equal(self.current_semantic_labels, self.predictor_state.labels):
            raise SelectiveTopologyAcquisitionError(code, "Z must equal decoder-owned predictor labels byte-for-byte")
        if self.current_semantic_labels_sha256 != self.predictor_state.labels_sha256:
            raise SelectiveTopologyAcquisitionError(code, "Z hash differs from decoder-owned predictor labels")

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": self.schema,
                    "source_pair_ids": list(self.source_pair_ids),
                    "predictor_state_kind": self.predictor_state_kind,
                    "predictor_binding_sha256": self.predictor_state.binding_sha256,
                    "base_p_section_sha256": self.base_p_section_sha256,
                    "base_p_section_bytes": self.base_p_section_bytes,
                    "current_camera_y1_sha256": self.current_camera_y1_sha256,
                    "target_scorer_rgb_sha256": self.target_scorer_rgb_sha256,
                    "frozen_scorer_sha256": self.frozen_scorer_sha256,
                    "current_realization_forward_receipt_sha256": self.current_realization_forward_receipt_sha256,
                    "target_forward_receipt_sha256": self.target_forward_receipt_sha256,
                    "current_semantic_labels_sha256": self.current_semantic_labels_sha256,
                    "target_labels_sha256": self.target_labels_sha256,
                    "baseline_realized_labels_sha256": self.baseline_realized_labels_sha256,
                    "label_shape": list(self.current_semantic_labels.shape),
                    "serialized_dense_evidence_bytes": 0,
                }
            )
        )

    @property
    def predictor_state_kind(self) -> Literal["PREDICTOR_SEMANTIC_STATE_V1", "TASKSPACE_PREDICTOR_STATE_V2"]:
        if type(self.predictor_state) is PredictorSemanticStateV1:
            return _PREDICTOR_STATE_V1
        return _PREDICTOR_STATE_V2

    @property
    def cell_partition_telemetry(self) -> SameClassRealizationCellPartitionTelemetryV1:
        return _partition_telemetry(
            self.source_pair_ids,
            self.current_semantic_labels,
            self.target_labels,
            self.baseline_realized_labels,
        )

    @property
    def topology_debt_mask(self) -> np.ndarray:
        result = _partition_masks(
            self.current_semantic_labels,
            self.target_labels,
            self.baseline_realized_labels,
        )[2]
        result.setflags(write=False)
        return result

    @property
    def fortunate_semantic_mismatch_mask(self) -> np.ndarray:
        result = _partition_masks(
            self.current_semantic_labels,
            self.target_labels,
            self.baseline_realized_labels,
        )[3]
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class SelectiveTopologyAcquisitionPlanV1:
    """Explicit singleton prefix requests; values are coordinates, not gates."""

    singleton_prefix_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.EMPTY_RESOLVED_PREFIX
        values = self.singleton_prefix_sizes
        if type(values) is not tuple or not 1 <= len(values) <= MAX_PREFIX_REQUESTS:
            raise SelectiveTopologyAcquisitionError(code, "singleton_prefix_sizes must be nonempty")
        if any(type(value) is not int or not 1 <= value <= MAX_PREFIX_REQUEST_VALUE for value in values):
            raise SelectiveTopologyAcquisitionError(
                code,
                "singleton prefix sizes must be bounded exact positive integers",
            )
        if values != tuple(sorted(set(values))):
            raise SelectiveTopologyAcquisitionError(code, "singleton prefix sizes must be sorted and unique")

    def resolve_with_requests(self, topology_debt_cell_count: int) -> tuple[tuple[int, int], ...]:
        if type(topology_debt_cell_count) is not int or topology_debt_cell_count < 0:
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.EMPTY_TOPOLOGY_DEBT,
                "topology debt count must be an exact nonnegative integer",
            )
        if topology_debt_cell_count == 0:
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.EMPTY_TOPOLOGY_DEBT,
                "selective topology acquisition refuses empty row-3 debt",
            )
        abi_limit = min(topology_debt_cell_count, MAX_SINGLETON_PREFIX_CELLS)
        resolved: list[tuple[int, int]] = []
        seen: set[int] = set()
        for requested in self.singleton_prefix_sizes:
            size = min(requested, abi_limit)
            if size > 0 and size not in seen:
                resolved.append((requested, size))
                seen.add(size)
        if not resolved:
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.EMPTY_RESOLVED_PREFIX,
                "prefix resolution produced no representable selection",
            )
        return tuple(resolved)

    def resolved_prefix_sizes(self, topology_debt_cell_count: int) -> tuple[int, ...]:
        return tuple(size for _requested, size in self.resolve_with_requests(topology_debt_cell_count))


@dataclass(frozen=True, slots=True, order=True)
class SelectiveTopologyTransitionCountV1:
    """Exact selected source-to-target transition census."""

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
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3,
                "selected transition census row is invalid",
            )

    def as_dict(self) -> dict[str, int]:
        return {"cells": self.cells, "source_class": self.source_class, "target_class": self.target_class}


def _parse_partition_telemetry(value: object) -> SameClassRealizationCellPartitionTelemetryV1:
    if type(value) is not dict or set(value) != {
        field.name for field in fields(SameClassRealizationCellPartitionTelemetryV1)
    }:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "cell-partition telemetry fields changed",
        )
    arguments = dict(value)
    for field in (
        "source_pair_ids",
        "label_shape",
        "closed_count_by_target_class",
        "realization_debt_count_by_target_class",
        "topology_debt_count_by_target_class",
        "fortunate_semantic_mismatch_count_by_target_class",
    ):
        if type(arguments[field]) is not list:
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
                f"cell-partition {field} must encode as a JSON list",
            )
        arguments[field] = tuple(arguments[field])
    try:
        return SameClassRealizationCellPartitionTelemetryV1(**arguments)
    except (TypeError, ValueError) as exc:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "cell-partition telemetry failed typed reconstruction",
        ) from exc


def _strict_json_object(payload: bytes, *, expected_fields: set[str], kind: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            f"{kind} must be nonempty exact bytes",
        )

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectiveTopologyAcquisitionError(
                    SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
                    f"{kind} repeats key {key!r}",
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except SelectiveTopologyAcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            f"{kind} is not strict ASCII JSON",
        ) from exc
    if type(value) is not dict or set(value) != expected_fields or _canonical_json(value) != payload:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            f"{kind} fields or canonical encoding changed",
        )
    return value


@dataclass(frozen=True, slots=True)
class SelectiveTopologyProgramProposalReceiptV1:
    """Canonical dense-free receipt for one exact compiled TACG1C program."""

    schema: Literal["tac.selective_topology_program_proposal_receipt.v1"]
    proposal_id: str
    evidence_binding_sha256: str
    teacher_evidence_binding_sha256: str
    predictor_binding_sha256: str
    predictor_state_kind: Literal["PREDICTOR_SEMANTIC_STATE_V1", "TASKSPACE_PREDICTOR_STATE_V2"]
    base_p_section_sha256: str
    base_p_section_bytes: int
    current_camera_y1_sha256: str
    target_scorer_rgb_sha256: str
    frozen_scorer_sha256: str
    current_realization_forward_receipt_sha256: str
    target_forward_receipt_sha256: str
    source_pair_ids: tuple[int, ...]
    family: SelectiveTopologyGrammarFamilyV1
    prefix_order: SelectiveTopologyPrefixOrderV1
    requested_prefix_cell_count: int
    resolved_prefix_cell_count: int
    selected_mask_sha256: str
    selected_count_by_target_class: tuple[int, int, int, int, int]
    transition_counts: tuple[SelectiveTopologyTransitionCountV1, ...]
    birth_event_count: int
    death_event_count: int
    total_topology_events: int
    packet_bytes: int
    packet_sha256: str
    compile_receipt_binding_sha256: str
    baseline_cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    baseline_semantic_debt_cell_count: int
    post_topology_semantic_sha256: str
    residual_semantic_debt_cell_count: int
    production_envelope_status: SelectiveTopologyProductionEnvelopeStatusV1
    selected_support_is_exact_row3_subset: Literal[True] = True
    decoded_changed_support_equals_selected_row3: Literal[True] = True
    selected_cells_decode_to_exact_target: Literal[True] = True
    row4_semantics_preserved: Literal[True] = True
    row4_excluded_from_g_owned_support: Literal[True] = True
    unselected_semantics_preserved: Literal[True] = True
    row4_post_topology_h_preservation_claim: Literal[False] = False
    deterministic_strict_parse_apply_verified: Literal[True] = True
    serialized_dense_evidence_bytes: Literal[0] = 0
    serialized_target_table_bytes: Literal[0] = 0
    serialized_scorer_rgb_bytes: Literal[0] = 0
    arbitrary_component_or_byte_thresholds_applied: Literal[False] = False
    scorer_invoked: Literal[False] = False
    through_r_realization_verified: Literal[False] = False
    whole_object_allocator_invoked: Literal[False] = False
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_candidate_eligible: Literal[False] = False
    compiled_g_generic_candidate_eligibility_superseded_by_selective_acquisition: Literal[True] = True
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
        if self.schema != PROPOSAL_RECEIPT_SCHEMA:
            raise SelectiveTopologyAcquisitionError(code, "proposal receipt schema changed")
        if (
            type(self.proposal_id) is not str
            or not self.proposal_id
            or len(self.proposal_id) > 128
            or not self.proposal_id.isascii()
        ):
            raise SelectiveTopologyAcquisitionError(code, "proposal_id must be bounded nonempty ASCII")
        for field in (
            "evidence_binding_sha256",
            "teacher_evidence_binding_sha256",
            "predictor_binding_sha256",
            "base_p_section_sha256",
            "current_camera_y1_sha256",
            "target_scorer_rgb_sha256",
            "frozen_scorer_sha256",
            "current_realization_forward_receipt_sha256",
            "target_forward_receipt_sha256",
            "selected_mask_sha256",
            "packet_sha256",
            "compile_receipt_binding_sha256",
            "post_topology_semantic_sha256",
        ):
            _require_sha256(getattr(self, field), field, code=code)
        if self.predictor_state_kind not in {_PREDICTOR_STATE_V1, _PREDICTOR_STATE_V2}:
            raise SelectiveTopologyAcquisitionError(code, "proposal predictor-state kind escaped the closed seam")
        _validate_pair_ids(self.source_pair_ids)
        _require_exact_int(self.base_p_section_bytes, "base_p_section_bytes", minimum=1, maximum=0xFFFFFFFF, code=code)
        if (
            type(self.family) is not SelectiveTopologyGrammarFamilyV1
            or type(self.prefix_order) is not SelectiveTopologyPrefixOrderV1
        ):
            raise SelectiveTopologyAcquisitionError(code, "proposal family/order escaped the closed universe")
        if type(self.baseline_cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1:
            raise SelectiveTopologyAcquisitionError(code, "proposal must nest exact canonical four-way telemetry")
        for field in (
            "requested_prefix_cell_count",
            "resolved_prefix_cell_count",
            "birth_event_count",
            "death_event_count",
            "total_topology_events",
            "packet_bytes",
            "baseline_semantic_debt_cell_count",
            "residual_semantic_debt_cell_count",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise SelectiveTopologyAcquisitionError(code, f"{field} must be an exact nonnegative integer")
        if not 1 <= self.resolved_prefix_cell_count <= self.requested_prefix_cell_count:
            raise SelectiveTopologyAcquisitionError(code, "resolved prefix does not close against its request")
        if (
            type(self.selected_count_by_target_class) is not tuple
            or len(self.selected_count_by_target_class) != _CLASS_COUNT
            or any(type(value) is not int or value < 0 for value in self.selected_count_by_target_class)
            or sum(self.selected_count_by_target_class) != self.resolved_prefix_cell_count
        ):
            raise SelectiveTopologyAcquisitionError(code, "selected target-class counts do not close")
        if (
            type(self.transition_counts) is not tuple
            or not self.transition_counts
            or any(type(row) is not SelectiveTopologyTransitionCountV1 for row in self.transition_counts)
            or self.transition_counts != tuple(sorted(self.transition_counts))
            or sum(row.cells for row in self.transition_counts) != self.resolved_prefix_cell_count
        ):
            raise SelectiveTopologyAcquisitionError(code, "selected transition census does not close canonically")
        if (
            self.total_topology_events != self.birth_event_count + self.death_event_count
            or not 1 <= self.total_topology_events <= MAX_TOPOLOGY_EVENTS
            or self.packet_bytes < 1
        ):
            raise SelectiveTopologyAcquisitionError(code, "event and packet accounting does not close")
        telemetry = self.baseline_cell_partition_telemetry
        if (
            telemetry.source_pair_ids != self.source_pair_ids
            or self.resolved_prefix_cell_count > telemetry.topology_debt_cell_count
        ):
            raise SelectiveTopologyAcquisitionError(code, "selected prefix escaped baseline row 3")
        expected_semantic_debt = telemetry.topology_debt_cell_count + telemetry.fortunate_semantic_mismatch_cell_count
        if (
            self.baseline_semantic_debt_cell_count != expected_semantic_debt
            or self.residual_semantic_debt_cell_count != expected_semantic_debt - self.resolved_prefix_cell_count
        ):
            raise SelectiveTopologyAcquisitionError(code, "semantic-debt arithmetic does not close")
        if self.production_envelope_status is not _PRODUCTION_BLOCKER:
            raise SelectiveTopologyAcquisitionError(code, "production selective-topology blocker was relaxed")
        truths = {
            "selected_support_is_exact_row3_subset": True,
            "decoded_changed_support_equals_selected_row3": True,
            "selected_cells_decode_to_exact_target": True,
            "row4_semantics_preserved": True,
            "row4_excluded_from_g_owned_support": True,
            "unselected_semantics_preserved": True,
            "row4_post_topology_h_preservation_claim": False,
            "deterministic_strict_parse_apply_verified": True,
            "serialized_dense_evidence_bytes": 0,
            "serialized_target_table_bytes": 0,
            "serialized_scorer_rgb_bytes": 0,
            "arbitrary_component_or_byte_thresholds_applied": False,
            "scorer_invoked": False,
            "through_r_realization_verified": False,
            "whole_object_allocator_invoked": False,
            "candidate_claim": False,
            "score_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "production_candidate_eligible": False,
            "compiled_g_generic_candidate_eligibility_superseded_by_selective_acquisition": True,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truths.items()):
            raise SelectiveTopologyAcquisitionError(code, "proposal authority or preservation labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["family"] = self.family.value
        value["prefix_order"] = self.prefix_order.value
        value["production_envelope_status"] = self.production_envelope_status.value
        value["selected_count_by_target_class"] = list(self.selected_count_by_target_class)
        value["transition_counts"] = [row.as_dict() for row in self.transition_counts]
        value["baseline_cell_partition_telemetry"] = self.baseline_cell_partition_telemetry.as_dict()
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_selective_topology_program_proposal_receipt(
    payload: bytes,
) -> SelectiveTopologyProgramProposalReceiptV1:
    """Strictly parse and canonically re-emit one proposal receipt."""

    value = _strict_json_object(
        payload,
        expected_fields={field.name for field in fields(SelectiveTopologyProgramProposalReceiptV1)},
        kind="selective topology proposal receipt",
    )
    arguments = dict(value)
    try:
        for field in ("source_pair_ids", "selected_count_by_target_class"):
            if type(arguments[field]) is not list:
                raise TypeError(field)
            arguments[field] = tuple(arguments[field])
        raw_transitions = arguments["transition_counts"]
        if type(raw_transitions) is not list:
            raise TypeError("transition_counts")
        transition_fields = {field.name for field in fields(SelectiveTopologyTransitionCountV1)}
        arguments["transition_counts"] = tuple(
            SelectiveTopologyTransitionCountV1(**row)
            for row in raw_transitions
            if type(row) is dict and set(row) == transition_fields
        )
        if len(arguments["transition_counts"]) != len(raw_transitions):
            raise TypeError("transition_counts")
        arguments["baseline_cell_partition_telemetry"] = _parse_partition_telemetry(
            arguments["baseline_cell_partition_telemetry"]
        )
        arguments["family"] = SelectiveTopologyGrammarFamilyV1(arguments["family"])
        arguments["prefix_order"] = SelectiveTopologyPrefixOrderV1(arguments["prefix_order"])
        arguments["production_envelope_status"] = SelectiveTopologyProductionEnvelopeStatusV1(
            arguments["production_envelope_status"]
        )
        receipt = SelectiveTopologyProgramProposalReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SelectiveTopologyAcquisitionError):
            raise
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "proposal receipt contains invalid typed fields",
        ) from exc
    if receipt.to_receipt_bytes() != payload:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "proposal receipt changed on canonical parse/re-emit",
        )
    return receipt


@dataclass(frozen=True, slots=True)
class SelectiveTopologyProgramProposalV1:
    """One compiled TACG1C packet, immutable selection, and exact receipt."""

    proposal_id: str
    compiled: CompiledGenerativeCorrectionV1
    selected_row3_mask: np.ndarray
    receipt: SelectiveTopologyProgramProposalReceiptV1

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3
        if (
            type(self.compiled) is not CompiledGenerativeCorrectionV1
            or type(self.receipt) is not SelectiveTopologyProgramProposalReceiptV1
        ):
            raise SelectiveTopologyAcquisitionError(code, "proposal lost its exact compiled/receipt type")
        if self.proposal_id != self.receipt.proposal_id:
            raise SelectiveTopologyAcquisitionError(code, "proposal ID differs from receipt")
        mask = np.asarray(self.selected_row3_mask)
        expected_shape = (
            len(self.receipt.source_pair_ids),
            SCORER_HEIGHT,
            SCORER_WIDTH,
        )
        if mask.dtype != np.bool_ or mask.shape != expected_shape:
            raise SelectiveTopologyAcquisitionError(code, "selected row-3 mask changed boolean scorer geometry")
        immutable = _immutable_array(mask, dtype=np.dtype(np.bool_))
        if (
            _array_sha256(immutable) != self.receipt.selected_mask_sha256
            or int(np.count_nonzero(immutable)) != self.receipt.resolved_prefix_cell_count
        ):
            raise SelectiveTopologyAcquisitionError(code, "selected mask differs from receipt")
        object.__setattr__(self, "selected_row3_mask", immutable)
        if (
            len(self.compiled.packet) != self.receipt.packet_bytes
            or _sha256(self.compiled.packet) != self.receipt.packet_sha256
            or self.compiled.receipt_binding_sha256 != self.receipt.compile_receipt_binding_sha256
            or _array_sha256(self.compiled.decoded.labels) != self.receipt.post_topology_semantic_sha256
        ):
            raise SelectiveTopologyAcquisitionError(code, "compiled TACG1C bytes differ from proposal receipt")


@dataclass(frozen=True, slots=True)
class SelectiveTopologyAcquisitionReceiptV1:
    """Canonical summary retaining full telemetry and every interpretation."""

    schema: Literal["tac.selective_topology_acquisition_receipt.v1"]
    evidence_binding_sha256: str
    teacher_evidence_binding_sha256: str
    requested_prefix_sizes: tuple[int, ...]
    resolved_prefix_sizes: tuple[int, ...]
    baseline_cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    topology_debt_cell_count: int
    proposal_count: int
    unique_packet_count: int
    proposal_receipt_sha256s: tuple[str, ...]
    unique_packet_sha256s: tuple[str, ...]
    production_envelope_status: SelectiveTopologyProductionEnvelopeStatusV1
    all_orders_emitted: Literal[True] = True
    all_grammar_families_emitted: Literal[True] = True
    exact_packet_byte_dedupe: Literal[True] = True
    arbitrary_component_or_byte_thresholds_applied: Literal[False] = False
    scorer_invoked: Literal[False] = False
    whole_object_allocator_invoked: Literal[False] = False
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_candidate_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
        if self.schema != ACQUISITION_RECEIPT_SCHEMA:
            raise SelectiveTopologyAcquisitionError(code, "acquisition receipt schema changed")
        for field in ("evidence_binding_sha256", "teacher_evidence_binding_sha256"):
            _require_sha256(getattr(self, field), field, code=code)
        if type(self.baseline_cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1:
            raise SelectiveTopologyAcquisitionError(code, "acquisition receipt lost complete four-way telemetry")
        if (
            type(self.requested_prefix_sizes) is not tuple
            or not self.requested_prefix_sizes
            or self.requested_prefix_sizes != tuple(sorted(set(self.requested_prefix_sizes)))
            or type(self.resolved_prefix_sizes) is not tuple
            or not self.resolved_prefix_sizes
            or len(set(self.resolved_prefix_sizes)) != len(self.resolved_prefix_sizes)
        ):
            raise SelectiveTopologyAcquisitionError(code, "acquisition prefix summaries are noncanonical")
        if self.topology_debt_cell_count != self.baseline_cell_partition_telemetry.topology_debt_cell_count:
            raise SelectiveTopologyAcquisitionError(code, "acquisition row-3 count differs from complete telemetry")
        expected_proposals = len(self.resolved_prefix_sizes) * len(_PREFIX_ORDERS) * len(_GRAMMAR_FAMILIES)
        if self.proposal_count != expected_proposals or not 1 <= self.unique_packet_count <= self.proposal_count:
            raise SelectiveTopologyAcquisitionError(code, "acquisition proposal/unique counts do not close")
        if (
            len(self.proposal_receipt_sha256s) != self.proposal_count
            or len(self.unique_packet_sha256s) != self.unique_packet_count
        ):
            raise SelectiveTopologyAcquisitionError(code, "acquisition digest tuple lengths do not close")
        for field in ("proposal_receipt_sha256s", "unique_packet_sha256s"):
            values = getattr(self, field)
            if type(values) is not tuple:
                raise SelectiveTopologyAcquisitionError(code, f"{field} must be an exact tuple")
            for index, value in enumerate(values):
                _require_sha256(value, f"{field}[{index}]", code=code)
        if len(set(self.unique_packet_sha256s)) != len(self.unique_packet_sha256s):
            raise SelectiveTopologyAcquisitionError(code, "unique packet digest view contains aliases")
        if self.production_envelope_status is not _PRODUCTION_BLOCKER:
            raise SelectiveTopologyAcquisitionError(code, "production blocker was relaxed")
        truths = {
            "all_orders_emitted": True,
            "all_grammar_families_emitted": True,
            "exact_packet_byte_dedupe": True,
            "arbitrary_component_or_byte_thresholds_applied": False,
            "scorer_invoked": False,
            "whole_object_allocator_invoked": False,
            "candidate_claim": False,
            "score_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "production_candidate_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truths.items()):
            raise SelectiveTopologyAcquisitionError(code, "acquisition authority labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "requested_prefix_sizes",
            "resolved_prefix_sizes",
            "proposal_receipt_sha256s",
            "unique_packet_sha256s",
        ):
            value[field] = list(getattr(self, field))
        value["baseline_cell_partition_telemetry"] = self.baseline_cell_partition_telemetry.as_dict()
        value["production_envelope_status"] = self.production_envelope_status.value
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_selective_topology_acquisition_receipt(payload: bytes) -> SelectiveTopologyAcquisitionReceiptV1:
    """Strictly parse and canonically re-emit an acquisition summary."""

    value = _strict_json_object(
        payload,
        expected_fields={field.name for field in fields(SelectiveTopologyAcquisitionReceiptV1)},
        kind="selective topology acquisition receipt",
    )
    arguments = dict(value)
    try:
        for field in (
            "requested_prefix_sizes",
            "resolved_prefix_sizes",
            "proposal_receipt_sha256s",
            "unique_packet_sha256s",
        ):
            if type(arguments[field]) is not list:
                raise TypeError(field)
            arguments[field] = tuple(arguments[field])
        arguments["baseline_cell_partition_telemetry"] = _parse_partition_telemetry(
            arguments["baseline_cell_partition_telemetry"]
        )
        arguments["production_envelope_status"] = SelectiveTopologyProductionEnvelopeStatusV1(
            arguments["production_envelope_status"]
        )
        receipt = SelectiveTopologyAcquisitionReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SelectiveTopologyAcquisitionError):
            raise
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "acquisition receipt contains invalid typed fields",
        ) from exc
    if receipt.to_receipt_bytes() != payload:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH,
            "acquisition receipt changed on canonical parse/re-emit",
        )
    return receipt


@dataclass(frozen=True, slots=True)
class SelectiveTopologyAcquisitionV1:
    """Complete interpretation lattice plus exact-wire-unique measurement view."""

    schema: Literal["tac.selective_topology_acquisition.v1"]
    evidence_binding_sha256: str
    teacher_evidence_binding_sha256: str
    baseline_cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    resolved_prefix_sizes: tuple[int, ...]
    proposals: tuple[SelectiveTopologyProgramProposalV1, ...]
    unique_program_proposals: tuple[SelectiveTopologyProgramProposalV1, ...]
    receipt: SelectiveTopologyAcquisitionReceiptV1
    production_envelope_status: SelectiveTopologyProductionEnvelopeStatusV1 = _PRODUCTION_BLOCKER

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3
        if self.schema != ACQUISITION_SCHEMA:
            raise SelectiveTopologyAcquisitionError(code, "acquisition schema changed")
        if type(self.baseline_cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1:
            raise SelectiveTopologyAcquisitionError(code, "acquisition lost canonical telemetry")
        if type(self.receipt) is not SelectiveTopologyAcquisitionReceiptV1:
            raise SelectiveTopologyAcquisitionError(code, "acquisition lost exact receipt")
        if (
            type(self.proposals) is not tuple
            or not self.proposals
            or any(type(row) is not SelectiveTopologyProgramProposalV1 for row in self.proposals)
        ):
            raise SelectiveTopologyAcquisitionError(code, "acquisition proposals changed exact type")
        if type(self.unique_program_proposals) is not tuple or not self.unique_program_proposals:
            raise SelectiveTopologyAcquisitionError(code, "acquisition exact-wire unique view is empty")
        expected_unique: list[SelectiveTopologyProgramProposalV1] = []
        seen: set[tuple[bytes, str]] = set()
        for proposal in self.proposals:
            key = (proposal.compiled.packet, proposal.receipt.packet_sha256)
            if key not in seen:
                expected_unique.append(proposal)
                seen.add(key)
        if self.unique_program_proposals != tuple(expected_unique):
            raise SelectiveTopologyAcquisitionError(code, "unique view is not deterministic exact packet-byte dedupe")
        if (
            self.receipt.evidence_binding_sha256 != self.evidence_binding_sha256
            or self.receipt.teacher_evidence_binding_sha256 != self.teacher_evidence_binding_sha256
            or self.receipt.resolved_prefix_sizes != self.resolved_prefix_sizes
            or self.receipt.baseline_cell_partition_telemetry != self.baseline_cell_partition_telemetry
            or self.receipt.proposal_receipt_sha256s != tuple(row.receipt.receipt_sha256 for row in self.proposals)
            or self.receipt.unique_packet_sha256s
            != tuple(row.receipt.packet_sha256 for row in self.unique_program_proposals)
        ):
            raise SelectiveTopologyAcquisitionError(code, "acquisition object differs from its canonical receipt")
        if self.production_envelope_status is not _PRODUCTION_BLOCKER:
            raise SelectiveTopologyAcquisitionError(code, "G7 structural success cannot relax production blocker")


def _row_runs(mask: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != (SCORER_HEIGHT, SCORER_WIDTH):
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3,
            "row-run input changed scorer-grid boolean geometry",
        )
    runs: list[tuple[int, int, int]] = []
    for row in range(SCORER_HEIGHT):
        padded = np.empty(SCORER_WIDTH + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = array[row]
        edges = np.flatnonzero(np.diff(padded))
        for start, stop in edges.reshape(-1, 2):
            runs.append((row, int(start), int(stop)))
    return tuple(runs)


def _manhattan_clearance_from_fortunate(fortunate: np.ndarray) -> np.ndarray:
    """Exact within-pair city-block distance; empty rows tie canonically."""

    if fortunate.dtype != np.bool_ or fortunate.ndim != 3:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3,
            "fortunate mask changed boolean scorer geometry",
        )
    infinity = SCORER_HEIGHT + SCORER_WIDTH + 1
    result = np.empty(fortunate.shape, dtype=np.int32)
    for pair_index in range(fortunate.shape[0]):
        mask = fortunate[pair_index]
        distance = np.where(mask, 0, infinity).astype(np.int32)
        for row in range(SCORER_HEIGHT):
            for column in range(SCORER_WIDTH):
                if row:
                    distance[row, column] = min(distance[row, column], distance[row - 1, column] + 1)
                if column:
                    distance[row, column] = min(distance[row, column], distance[row, column - 1] + 1)
        for row in range(SCORER_HEIGHT - 1, -1, -1):
            for column in range(SCORER_WIDTH - 1, -1, -1):
                if row + 1 < SCORER_HEIGHT:
                    distance[row, column] = min(distance[row, column], distance[row + 1, column] + 1)
                if column + 1 < SCORER_WIDTH:
                    distance[row, column] = min(distance[row, column], distance[row, column + 1] + 1)
        result[pair_index] = distance
    return result


def _ordered_row3_coordinates(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    order: SelectiveTopologyPrefixOrderV1,
) -> tuple[tuple[int, int, int], ...]:
    topology = evidence.topology_debt_mask
    coordinates = [tuple(int(value) for value in row) for row in np.argwhere(topology)]
    if order is SelectiveTopologyPrefixOrderV1.CANONICAL_ADDRESS_V1:
        return tuple(coordinates)
    if order is SelectiveTopologyPrefixOrderV1.FORTUNATE_CLEARANCE_DESCENDING_V1:
        clearance = _manhattan_clearance_from_fortunate(evidence.fortunate_semantic_mismatch_mask)
        return tuple(sorted(coordinates, key=lambda row: (-int(clearance[row]), row)))
    if order is SelectiveTopologyPrefixOrderV1.TRANSITION_MASS_DESCENDING_V1:
        masses: dict[tuple[int, int], int] = {}
        for pair, row, column in coordinates:
            transition = (
                int(evidence.current_semantic_labels[pair, row, column]),
                int(evidence.target_labels[pair, row, column]),
            )
            masses[transition] = masses.get(transition, 0) + 1
        transition_rank = {
            transition: rank for rank, transition in enumerate(sorted(masses, key=lambda key: (-masses[key], key)))
        }
        return tuple(
            sorted(
                coordinates,
                key=lambda cell: (
                    transition_rank[
                        (
                            int(evidence.current_semantic_labels[cell]),
                            int(evidence.target_labels[cell]),
                        )
                    ],
                    cell,
                ),
            )
        )
    raise SelectiveTopologyAcquisitionError(
        SelectiveTopologyBlockerCodeV1.EMPTY_RESOLVED_PREFIX,
        "prefix order escaped the closed universe",
    )


def _selected_mask(
    shape: tuple[int, ...],
    ordered_coordinates: tuple[tuple[int, int, int], ...],
    size: int,
) -> np.ndarray:
    selected = np.zeros(shape, dtype=np.bool_)
    for coordinate in ordered_coordinates[:size]:
        selected[coordinate] = True
    if int(np.count_nonzero(selected)) != size:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.EMPTY_RESOLVED_PREFIX,
            "resolved prefix did not select its exact cardinality",
        )
    return selected


def _transition_census(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    selected: np.ndarray,
) -> tuple[SelectiveTopologyTransitionCountV1, ...]:
    rows: list[SelectiveTopologyTransitionCountV1] = []
    for source_class in range(_CLASS_COUNT):
        for target_class in range(_CLASS_COUNT):
            if source_class == target_class:
                continue
            count = int(
                np.count_nonzero(
                    selected
                    & (evidence.current_semantic_labels == source_class)
                    & (evidence.target_labels == target_class)
                )
            )
            if count:
                rows.append(SelectiveTopologyTransitionCountV1(source_class, target_class, count))
    return tuple(rows)


def _event(
    *,
    pair_id: int,
    role: str,
    action: Literal["birth", "death"],
    row: int,
    start: int,
    stop: int,
) -> TopologyEventV1:
    return TopologyEventV1(
        pair_index=pair_id,
        role=role,
        action=action,
        shape="box",
        lifetime=1,
        y0=row,
        x0=start,
        y1=row + 1,
        x1=stop,
        transport_gain_x_q4=0,
        transport_gain_y_q4=0,
    )


def _event_plan(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    selected: np.ndarray,
    family: SelectiveTopologyGrammarFamilyV1,
) -> tuple[tuple[TopologyEventV1, ...], int, int]:
    events: list[TopologyEventV1] = []
    births = 0
    deaths = 0

    def append_transition_box(
        pair_id: int,
        source_class: int,
        target_class: int,
        row: int,
        start: int,
        stop: int,
    ) -> None:
        nonlocal births, deaths
        source_role = _ROLE_BY_CLASS[source_class]
        target_role = _ROLE_BY_CLASS[target_class]
        events.append(_event(pair_id=pair_id, role=target_role, action="birth", row=row, start=start, stop=stop))
        births += 1
        if _PRECEDENCE[source_role] > _PRECEDENCE[target_role]:
            events.append(_event(pair_id=pair_id, role=source_role, action="death", row=row, start=start, stop=stop))
            deaths += 1

    for local_pair, pair_id in enumerate(evidence.source_pair_ids):
        source = evidence.current_semantic_labels[local_pair]
        target = evidence.target_labels[local_pair]
        chosen = selected[local_pair]
        if family is SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1:
            for row, column in np.argwhere(chosen):
                source_class = int(source[row, column])
                target_class = int(target[row, column])
                append_transition_box(pair_id, source_class, target_class, int(row), int(column), int(column) + 1)
        elif family is SelectiveTopologyGrammarFamilyV1.SOURCE_TARGET_ROW_RUN_BOX_V1:
            for source_class in range(_CLASS_COUNT):
                for target_class in range(_CLASS_COUNT):
                    if source_class == target_class:
                        continue
                    mask = chosen & (source == source_class) & (target == target_class)
                    for row, start, stop in _row_runs(mask):
                        append_transition_box(pair_id, source_class, target_class, row, start, stop)
        elif family is SelectiveTopologyGrammarFamilyV1.TARGET_BIRTH_SOURCE_DEATH_ROW_RUN_BOX_V1:
            for target_class in range(_CLASS_COUNT):
                for row, start, stop in _row_runs(chosen & (target == target_class)):
                    events.append(
                        _event(
                            pair_id=pair_id,
                            role=_ROLE_BY_CLASS[target_class],
                            action="birth",
                            row=row,
                            start=start,
                            stop=stop,
                        )
                    )
                    births += 1
            for source_class in range(_CLASS_COUNT):
                source_role = _ROLE_BY_CLASS[source_class]
                for target_class in range(_CLASS_COUNT):
                    if (
                        source_class == target_class
                        or _PRECEDENCE[source_role] <= _PRECEDENCE[_ROLE_BY_CLASS[target_class]]
                    ):
                        continue
                    mask = chosen & (source == source_class) & (target == target_class)
                    for row, start, stop in _row_runs(mask):
                        events.append(
                            _event(
                                pair_id=pair_id,
                                role=source_role,
                                action="death",
                                row=row,
                                start=start,
                                stop=stop,
                            )
                        )
                        deaths += 1
        else:  # pragma: no cover - exact enum dispatch above
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.TACG1C_COMPILE_PARSE_APPLY_REFUSAL,
                "grammar family escaped the closed universe",
            )
        if len(events) > MAX_TOPOLOGY_EVENTS:
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.TACG1C_EVENT_CARDINALITY_OVERFLOW,
                "selective topology event plan exceeds the uint16 TACG1C ABI",
            )
    if not events:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.EMPTY_RESOLVED_PREFIX,
            "selected row-3 prefix produced an empty topology event plan",
        )
    return tuple(events), births, deaths


def _validate_teacher(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
) -> None:
    code = SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
    if type(teacher_evidence) is not EncoderOnlyTeacherEvidenceV1:
        raise SelectiveTopologyAcquisitionError(code, "teacher_evidence must be actual EncoderOnlyTeacherEvidenceV1")
    semantic_debt = int(np.count_nonzero(evidence.current_semantic_labels != evidence.target_labels))
    if (
        teacher_evidence.target_labels_sha256 != evidence.target_labels_sha256
        or not np.array_equal(teacher_evidence.target_labels, evidence.target_labels)
        or teacher_evidence.teacher_event_count != semantic_debt
    ):
        raise SelectiveTopologyAcquisitionError(
            code, "teacher target bytes/count do not bind complete baseline semantic debt"
        )


def _compile_proposal(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
    realization_profile: ReceiverRealizationProfileV1,
    telemetry: SameClassRealizationCellPartitionTelemetryV1,
    *,
    selected: np.ndarray,
    requested: int,
    resolved: int,
    order: SelectiveTopologyPrefixOrderV1,
    family: SelectiveTopologyGrammarFamilyV1,
) -> SelectiveTopologyProgramProposalV1:
    events, births, deaths = _event_plan(evidence, selected, family)
    if len(events) > MAX_TOPOLOGY_EVENTS:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.TACG1C_EVENT_CARDINALITY_OVERFLOW,
            "selective topology event plan exceeds the uint16 TACG1C ABI",
        )
    try:
        program = GenerativeCorrectionProgramV1(
            topology_events=events,
            realization_profile=realization_profile,
        )
        if type(evidence.predictor_state) is PredictorSemanticStateV1:
            compiled = compile_generative_taskspace_correction(
                evidence.predictor_state,
                program,
                teacher_evidence=teacher_evidence,
            )
            parsed = parse_generative_taskspace_correction(
                compiled.packet,
                predictor_state=evidence.predictor_state,
            )
            decoded = apply_generative_taskspace_correction(
                compiled.packet,
                predictor_state=evidence.predictor_state,
            )
            replay = apply_generative_taskspace_correction(
                compiled.packet,
                predictor_state=evidence.predictor_state,
            )
        else:
            assert type(evidence.predictor_state) is TaskspacePredictorStateV2
            compiled = compile_generative_taskspace_correction_v2(
                evidence.predictor_state,
                program,
                teacher_evidence=teacher_evidence,
            )
            parsed = parse_generative_taskspace_correction_v2(
                compiled.packet,
                predictor_state=evidence.predictor_state,
            )
            decoded = apply_generative_taskspace_correction_v2(
                compiled.packet,
                predictor_state=evidence.predictor_state,
            )
            replay = apply_generative_taskspace_correction_v2(
                compiled.packet,
                predictor_state=evidence.predictor_state,
            )
    except (GenerativeTaskspaceCorrectionError, ValueError) as exc:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.TACG1C_COMPILE_PARSE_APPLY_REFUSAL,
            "strict TACG1C compile/parse/apply refused the selective event plan",
        ) from exc
    if parsed.packet != compiled.packet or not np.array_equal(decoded.labels, replay.labels):
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.TACG1C_COMPILE_PARSE_APPLY_REFUSAL,
            "strict TACG1C parse/apply replay changed exact bytes or labels",
        )
    changed = np.ascontiguousarray(decoded.labels != evidence.current_semantic_labels, dtype=np.bool_)
    row4 = evidence.fortunate_semantic_mismatch_mask
    outside = ~selected
    exact = (
        np.array_equal(changed, selected)
        and np.array_equal(decoded.labels[selected], evidence.target_labels[selected])
        and np.array_equal(decoded.labels[row4], evidence.current_semantic_labels[row4])
        and np.array_equal(decoded.labels[outside], evidence.current_semantic_labels[outside])
    )
    if not exact:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3,
            "decoded TACG1C mutation differs from the exact selected row-3 support",
        )
    transitions = _transition_census(evidence, selected)
    packet_hash = _sha256(compiled.packet)
    proposal_id = f"sta:{resolved}:{_PREFIX_ORDERS.index(order)}:{_GRAMMAR_FAMILIES.index(family)}:{packet_hash[:12]}"
    baseline_semantic_debt = int(np.count_nonzero(evidence.current_semantic_labels != evidence.target_labels))
    receipt = SelectiveTopologyProgramProposalReceiptV1(
        schema=PROPOSAL_RECEIPT_SCHEMA,
        proposal_id=proposal_id,
        evidence_binding_sha256=evidence.binding_sha256,
        teacher_evidence_binding_sha256=teacher_evidence.binding_sha256,
        predictor_binding_sha256=evidence.predictor_state.binding_sha256,
        predictor_state_kind=evidence.predictor_state_kind,
        base_p_section_sha256=evidence.base_p_section_sha256,
        base_p_section_bytes=evidence.base_p_section_bytes,
        current_camera_y1_sha256=evidence.current_camera_y1_sha256,
        target_scorer_rgb_sha256=evidence.target_scorer_rgb_sha256,
        frozen_scorer_sha256=evidence.frozen_scorer_sha256,
        current_realization_forward_receipt_sha256=evidence.current_realization_forward_receipt_sha256,
        target_forward_receipt_sha256=evidence.target_forward_receipt_sha256,
        source_pair_ids=evidence.source_pair_ids,
        family=family,
        prefix_order=order,
        requested_prefix_cell_count=requested,
        resolved_prefix_cell_count=resolved,
        selected_mask_sha256=_array_sha256(selected),
        selected_count_by_target_class=_five_counts(selected, evidence.target_labels),
        transition_counts=transitions,
        birth_event_count=births,
        death_event_count=deaths,
        total_topology_events=len(events),
        packet_bytes=len(compiled.packet),
        packet_sha256=packet_hash,
        compile_receipt_binding_sha256=compiled.receipt_binding_sha256,
        baseline_cell_partition_telemetry=telemetry,
        baseline_semantic_debt_cell_count=baseline_semantic_debt,
        post_topology_semantic_sha256=_array_sha256(decoded.labels),
        residual_semantic_debt_cell_count=int(np.count_nonzero(decoded.labels != evidence.target_labels)),
        production_envelope_status=_PRODUCTION_BLOCKER,
    )
    if parse_selective_topology_program_proposal_receipt(receipt.to_receipt_bytes()) != receipt:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.TACG1C_COMPILE_PARSE_APPLY_REFUSAL,
            "proposal receipt failed strict parse/re-emit",
        )
    return SelectiveTopologyProgramProposalV1(
        proposal_id=proposal_id,
        compiled=compiled,
        selected_row3_mask=selected,
        receipt=receipt,
    )


def acquire_selective_topology_programs(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    *,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
    realization_profile: ReceiverRealizationProfileV1,
    plan: SelectiveTopologyAcquisitionPlanV1,
) -> SelectiveTopologyAcquisitionV1:
    """Compile every bounded prefix/order/family through the real TACG1C API."""

    code = SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
    if type(evidence) is not EncoderOnlySelectiveTopologyEvidenceV1:
        raise SelectiveTopologyAcquisitionError(code, "evidence must use exact selective topology type")
    if type(plan) is not SelectiveTopologyAcquisitionPlanV1:
        raise SelectiveTopologyAcquisitionError(code, "plan must use exact selective topology type")
    if type(realization_profile) is not ReceiverRealizationProfileV1:
        raise SelectiveTopologyAcquisitionError(code, "realization_profile must use the counted receiver type")
    _validate_teacher(evidence, teacher_evidence)
    telemetry = evidence.cell_partition_telemetry
    topology_count = telemetry.topology_debt_cell_count
    resolved_with_requests = plan.resolve_with_requests(topology_count)
    ordered = {order: _ordered_row3_coordinates(evidence, order) for order in _PREFIX_ORDERS}
    proposals: list[SelectiveTopologyProgramProposalV1] = []
    for requested, resolved in resolved_with_requests:
        for order in _PREFIX_ORDERS:
            selected = _selected_mask(evidence.current_semantic_labels.shape, ordered[order], resolved)
            if not np.all(evidence.topology_debt_mask[selected]):
                raise SelectiveTopologyAcquisitionError(
                    SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3,
                    "prefix ordering selected a cell outside canonical row 3",
                )
            for family in _GRAMMAR_FAMILIES:
                proposals.append(
                    _compile_proposal(
                        evidence,
                        teacher_evidence,
                        realization_profile,
                        telemetry,
                        selected=selected,
                        requested=requested,
                        resolved=resolved,
                        order=order,
                        family=family,
                    )
                )
    unique: list[SelectiveTopologyProgramProposalV1] = []
    seen: set[tuple[bytes, str]] = set()
    for proposal in proposals:
        key = (proposal.compiled.packet, proposal.receipt.packet_sha256)
        if key not in seen:
            unique.append(proposal)
            seen.add(key)
    resolved_sizes = tuple(size for _requested, size in resolved_with_requests)
    receipt = SelectiveTopologyAcquisitionReceiptV1(
        schema=ACQUISITION_RECEIPT_SCHEMA,
        evidence_binding_sha256=evidence.binding_sha256,
        teacher_evidence_binding_sha256=teacher_evidence.binding_sha256,
        requested_prefix_sizes=plan.singleton_prefix_sizes,
        resolved_prefix_sizes=resolved_sizes,
        baseline_cell_partition_telemetry=telemetry,
        topology_debt_cell_count=topology_count,
        proposal_count=len(proposals),
        unique_packet_count=len(unique),
        proposal_receipt_sha256s=tuple(row.receipt.receipt_sha256 for row in proposals),
        unique_packet_sha256s=tuple(row.receipt.packet_sha256 for row in unique),
        production_envelope_status=_PRODUCTION_BLOCKER,
    )
    if parse_selective_topology_acquisition_receipt(receipt.to_receipt_bytes()) != receipt:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.TACG1C_COMPILE_PARSE_APPLY_REFUSAL,
            "acquisition receipt failed strict parse/re-emit",
        )
    return SelectiveTopologyAcquisitionV1(
        schema=ACQUISITION_SCHEMA,
        evidence_binding_sha256=evidence.binding_sha256,
        teacher_evidence_binding_sha256=teacher_evidence.binding_sha256,
        baseline_cell_partition_telemetry=telemetry,
        resolved_prefix_sizes=resolved_sizes,
        proposals=tuple(proposals),
        unique_program_proposals=tuple(unique),
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class SelectiveTopologyFreshG8RequestReceiptV1:
    """Dense-free custody receipt for a freshly scored post-topology state."""

    schema: Literal["tac.selective_topology_fresh_g8_request_receipt.v1"]
    evidence_binding_sha256: str
    proposal_id: str
    selective_g_packet_sha256: str
    selective_g_packet_bytes: int
    selective_g_compile_receipt_binding_sha256: str
    source_pair_ids: tuple[int, ...]
    post_topology_semantic_labels_sha256: str
    target_labels_sha256: str
    post_topology_realized_labels_sha256: str
    post_topology_scorer_rgb_sha256: str
    target_scorer_rgb_sha256: str
    post_topology_camera_y1_sha256: str
    frozen_scorer_sha256: str
    baseline_forward_receipt_sha256: str
    post_topology_forward_receipt_sha256: str
    target_forward_receipt_sha256: str
    baseline_realization_debt_mask_sha256: str
    fresh_realization_debt_mask_sha256: str
    fresh_realization_debt_cell_count: int
    baseline_cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    fresh_cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    g12_selective_base_status: SelectiveTopologyG12SeamStatusV1
    production_envelope_status: SelectiveTopologyProductionEnvelopeStatusV1
    fresh_debt_derived_from_post_topology_h_prime: Literal[True] = True
    baseline_g8_debt_reused: Literal[False] = False
    baseline_g8_program_reused: Literal[False] = False
    diagnostic_g12_base_renamed: Literal[False] = False
    row4_post_topology_h_preservation_claim_inherited: Literal[False] = False
    serialized_dense_evidence_bytes: Literal[0] = 0
    scorer_invoked_by_this_module: Literal[False] = False
    whole_object_allocator_invoked: Literal[False] = False
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_candidate_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.POST_FORWARD_CUSTODY_MISMATCH
        if self.schema != FRESH_G8_RECEIPT_SCHEMA:
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 receipt schema changed")
        if (
            type(self.proposal_id) is not str
            or not self.proposal_id
            or len(self.proposal_id) > 128
            or not self.proposal_id.isascii()
        ):
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 proposal ID must be bounded nonempty ASCII")
        for field in (
            "evidence_binding_sha256",
            "selective_g_packet_sha256",
            "selective_g_compile_receipt_binding_sha256",
            "post_topology_semantic_labels_sha256",
            "target_labels_sha256",
            "post_topology_realized_labels_sha256",
            "post_topology_scorer_rgb_sha256",
            "target_scorer_rgb_sha256",
            "post_topology_camera_y1_sha256",
            "frozen_scorer_sha256",
            "baseline_forward_receipt_sha256",
            "post_topology_forward_receipt_sha256",
            "target_forward_receipt_sha256",
            "baseline_realization_debt_mask_sha256",
            "fresh_realization_debt_mask_sha256",
        ):
            _require_sha256(getattr(self, field), field, code=code)
        _validate_pair_ids(self.source_pair_ids)
        _require_exact_int(
            self.selective_g_packet_bytes,
            "selective_g_packet_bytes",
            minimum=1,
            maximum=0xFFFFFFFF,
            code=code,
        )
        _require_exact_int(
            self.fresh_realization_debt_cell_count,
            "fresh_realization_debt_cell_count",
            minimum=0,
            maximum=MAX_BOUNDED_PAIRS * SCORER_HEIGHT * SCORER_WIDTH,
            code=code,
        )
        if self.baseline_forward_receipt_sha256 == self.post_topology_forward_receipt_sha256:
            raise SelectiveTopologyAcquisitionError(
                SelectiveTopologyBlockerCodeV1.BASELINE_FORWARD_RECEIPT_REUSED,
                "post-topology forward receipt reuses the baseline identity",
            )
        if (
            type(self.baseline_cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1
            or type(self.fresh_cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1
        ):
            raise SelectiveTopologyAcquisitionError(
                code, "fresh-G8 receipt must retain both complete four-way telemetry rows"
            )
        if (
            self.baseline_cell_partition_telemetry.source_pair_ids != self.source_pair_ids
            or self.fresh_cell_partition_telemetry.source_pair_ids != self.source_pair_ids
            or self.baseline_cell_partition_telemetry.realization_debt_mask_sha256
            != self.baseline_realization_debt_mask_sha256
            or self.fresh_cell_partition_telemetry.realization_debt_mask_sha256
            != self.fresh_realization_debt_mask_sha256
            or self.fresh_cell_partition_telemetry.realization_debt_cell_count != self.fresh_realization_debt_cell_count
            or self.fresh_cell_partition_telemetry.current_semantic_labels_sha256
            != self.post_topology_semantic_labels_sha256
            or self.fresh_cell_partition_telemetry.target_labels_sha256 != self.target_labels_sha256
            or self.fresh_cell_partition_telemetry.realized_labels_sha256 != self.post_topology_realized_labels_sha256
        ):
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 mask/telemetry custody does not close")
        if (
            self.g12_selective_base_status is not _G12_BLOCKER
            or self.production_envelope_status is not _PRODUCTION_BLOCKER
        ):
            raise SelectiveTopologyAcquisitionError(code, "selective G12 or production blocker was relaxed")
        truths = {
            "fresh_debt_derived_from_post_topology_h_prime": True,
            "baseline_g8_debt_reused": False,
            "baseline_g8_program_reused": False,
            "diagnostic_g12_base_renamed": False,
            "row4_post_topology_h_preservation_claim_inherited": False,
            "serialized_dense_evidence_bytes": 0,
            "scorer_invoked_by_this_module": False,
            "whole_object_allocator_invoked": False,
            "candidate_claim": False,
            "score_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "production_candidate_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truths.items()):
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 authority or causal labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["baseline_cell_partition_telemetry"] = self.baseline_cell_partition_telemetry.as_dict()
        value["fresh_cell_partition_telemetry"] = self.fresh_cell_partition_telemetry.as_dict()
        value["g12_selective_base_status"] = self.g12_selective_base_status.value
        value["production_envelope_status"] = self.production_envelope_status.value
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_selective_topology_fresh_g8_request_receipt(
    payload: bytes,
) -> SelectiveTopologyFreshG8RequestReceiptV1:
    """Strictly parse and canonically re-emit a fresh-cascade receipt."""

    value = _strict_json_object(
        payload,
        expected_fields={field.name for field in fields(SelectiveTopologyFreshG8RequestReceiptV1)},
        kind="selective topology fresh-G8 request receipt",
    )
    arguments = dict(value)
    try:
        if type(arguments["source_pair_ids"]) is not list:
            raise TypeError("source_pair_ids")
        arguments["source_pair_ids"] = tuple(arguments["source_pair_ids"])
        arguments["baseline_cell_partition_telemetry"] = _parse_partition_telemetry(
            arguments["baseline_cell_partition_telemetry"]
        )
        arguments["fresh_cell_partition_telemetry"] = _parse_partition_telemetry(
            arguments["fresh_cell_partition_telemetry"]
        )
        arguments["g12_selective_base_status"] = SelectiveTopologyG12SeamStatusV1(
            arguments["g12_selective_base_status"]
        )
        arguments["production_envelope_status"] = SelectiveTopologyProductionEnvelopeStatusV1(
            arguments["production_envelope_status"]
        )
        receipt = SelectiveTopologyFreshG8RequestReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SelectiveTopologyAcquisitionError):
            raise
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.POST_FORWARD_CUSTODY_MISMATCH,
            "fresh-G8 receipt contains invalid typed fields",
        ) from exc
    if receipt.to_receipt_bytes() != payload:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.POST_FORWARD_CUSTODY_MISMATCH,
            "fresh-G8 receipt changed on canonical parse/re-emit",
        )
    return receipt


@dataclass(frozen=True, slots=True)
class SelectiveTopologyFreshG8RequestV1:
    """Actual post-forward arrays for a future honest selective-base G12 path."""

    source_pair_ids: tuple[int, ...]
    selective_g_program_proposal: SelectiveTopologyProgramProposalV1
    post_topology_semantic_labels: np.ndarray
    target_labels: np.ndarray
    post_topology_realized_labels: np.ndarray
    post_topology_scorer_rgb: np.ndarray
    target_scorer_rgb: np.ndarray
    post_topology_camera_y1: np.ndarray
    fresh_realization_debt_mask: np.ndarray
    receipt: SelectiveTopologyFreshG8RequestReceiptV1
    g12_selective_base_status: SelectiveTopologyG12SeamStatusV1 = _G12_BLOCKER
    production_envelope_status: SelectiveTopologyProductionEnvelopeStatusV1 = _PRODUCTION_BLOCKER
    serialized_dense_evidence_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        code = SelectiveTopologyBlockerCodeV1.POST_FORWARD_CUSTODY_MISMATCH
        pair_ids = _validate_pair_ids(self.source_pair_ids)
        if type(self.selective_g_program_proposal) is not SelectiveTopologyProgramProposalV1:
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 request lost the exact selective-G proposal")
        if type(self.receipt) is not SelectiveTopologyFreshG8RequestReceiptV1:
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 request lost its exact receipt")
        proposal = self.selective_g_program_proposal
        if (
            proposal.receipt.evidence_binding_sha256 != self.receipt.evidence_binding_sha256
            or proposal.proposal_id != self.receipt.proposal_id
            or proposal.receipt.packet_sha256 != self.receipt.selective_g_packet_sha256
            or proposal.receipt.packet_bytes != self.receipt.selective_g_packet_bytes
            or proposal.compiled.receipt_binding_sha256 != self.receipt.selective_g_compile_receipt_binding_sha256
            or proposal.receipt.target_scorer_rgb_sha256 != self.receipt.target_scorer_rgb_sha256
            or proposal.receipt.target_forward_receipt_sha256 != self.receipt.target_forward_receipt_sha256
        ):
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 request selective-G bytes differ from receipt")
        label_shape = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
        rgb_shape = (*label_shape, CHANNELS)
        camera_shape = (len(pair_ids), CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        for field, shape, dtype, expected_hash in (
            (
                "post_topology_semantic_labels",
                label_shape,
                np.dtype(np.uint8),
                self.receipt.post_topology_semantic_labels_sha256,
            ),
            ("target_labels", label_shape, np.dtype(np.uint8), self.receipt.target_labels_sha256),
            (
                "post_topology_realized_labels",
                label_shape,
                np.dtype(np.uint8),
                self.receipt.post_topology_realized_labels_sha256,
            ),
            (
                "post_topology_scorer_rgb",
                rgb_shape,
                np.dtype(np.uint8),
                self.receipt.post_topology_scorer_rgb_sha256,
            ),
            (
                "target_scorer_rgb",
                rgb_shape,
                np.dtype(np.uint8),
                self.receipt.target_scorer_rgb_sha256,
            ),
            (
                "post_topology_camera_y1",
                camera_shape,
                np.dtype(np.uint8),
                self.receipt.post_topology_camera_y1_sha256,
            ),
            (
                "fresh_realization_debt_mask",
                label_shape,
                np.dtype(np.bool_),
                self.receipt.fresh_realization_debt_mask_sha256,
            ),
        ):
            raw = np.asarray(getattr(self, field))
            if raw.dtype != dtype or raw.shape != shape:
                raise SelectiveTopologyAcquisitionError(code, f"{field} changed exact post-forward geometry/dtype")
            if dtype == np.dtype(np.uint8) and raw.ndim == 3 and int(raw.max()) >= _CLASS_COUNT:
                raise SelectiveTopologyAcquisitionError(code, f"{field} escaped the five-class universe")
            immutable = _immutable_array(raw, dtype=dtype)
            if _array_sha256(immutable) != expected_hash:
                raise SelectiveTopologyAcquisitionError(code, f"{field} differs from fresh-G8 receipt")
            object.__setattr__(self, field, immutable)
        expected_debt = (self.post_topology_semantic_labels == self.target_labels) & (
            self.post_topology_realized_labels != self.target_labels
        )
        if not np.array_equal(self.fresh_realization_debt_mask, expected_debt):
            raise SelectiveTopologyAcquisitionError(code, "fresh row-2 mask was not derived from actual H-prime")
        if not np.array_equal(
            self.post_topology_semantic_labels,
            proposal.compiled.decoded.labels,
        ):
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 Z-prime differs from exact selective-G decode")
        if (
            self.receipt.source_pair_ids != pair_ids
            or self.g12_selective_base_status is not _G12_BLOCKER
            or self.production_envelope_status is not _PRODUCTION_BLOCKER
            or self.serialized_dense_evidence_bytes != 0
        ):
            raise SelectiveTopologyAcquisitionError(code, "fresh-G8 typed blockers or dense-evidence invariant changed")

    @property
    def binding_sha256(self) -> str:
        return self.receipt.receipt_sha256

    @property
    def selective_g_packet(self) -> bytes:
        return self.selective_g_program_proposal.compiled.packet


def bind_selective_topology_post_forward(
    evidence: EncoderOnlySelectiveTopologyEvidenceV1,
    proposal: SelectiveTopologyProgramProposalV1,
    *,
    post_topology_realized_labels: np.ndarray,
    post_topology_realized_labels_sha256: str,
    post_topology_scorer_rgb: np.ndarray,
    post_topology_scorer_rgb_sha256: str,
    target_scorer_rgb: np.ndarray,
    target_scorer_rgb_sha256: str,
    post_topology_camera_y1: np.ndarray,
    post_topology_camera_y1_sha256: str,
    post_topology_forward_receipt_sha256: str,
) -> SelectiveTopologyFreshG8RequestV1:
    """Bind actual H-prime/RGB/Y1 and recompute all Z-prime/T/H-prime rows."""

    code = SelectiveTopologyBlockerCodeV1.POST_FORWARD_CUSTODY_MISMATCH
    if (
        type(evidence) is not EncoderOnlySelectiveTopologyEvidenceV1
        or type(proposal) is not SelectiveTopologyProgramProposalV1
    ):
        raise SelectiveTopologyAcquisitionError(code, "post-forward bind requires exact evidence and proposal types")
    if (
        proposal.receipt.evidence_binding_sha256 != evidence.binding_sha256
        or proposal.receipt.source_pair_ids != evidence.source_pair_ids
    ):
        raise SelectiveTopologyAcquisitionError(code, "post-forward proposal belongs to foreign baseline evidence")
    for field, value in (
        ("post_topology_realized_labels_sha256", post_topology_realized_labels_sha256),
        ("post_topology_scorer_rgb_sha256", post_topology_scorer_rgb_sha256),
        ("target_scorer_rgb_sha256", target_scorer_rgb_sha256),
        ("post_topology_camera_y1_sha256", post_topology_camera_y1_sha256),
        ("post_topology_forward_receipt_sha256", post_topology_forward_receipt_sha256),
    ):
        _require_sha256(value, field, code=code)
    if post_topology_forward_receipt_sha256 == evidence.current_realization_forward_receipt_sha256:
        raise SelectiveTopologyAcquisitionError(
            SelectiveTopologyBlockerCodeV1.BASELINE_FORWARD_RECEIPT_REUSED,
            "post-topology forward receipt must differ from baseline forward identity",
        )
    post_semantic = proposal.compiled.decoded.labels
    labels = np.asarray(post_topology_realized_labels)
    scorer_rgb = np.asarray(post_topology_scorer_rgb)
    target_rgb = np.asarray(target_scorer_rgb)
    camera_y1 = np.asarray(post_topology_camera_y1)
    pair_count = len(evidence.source_pair_ids)
    if (
        labels.dtype != np.uint8
        or labels.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH)
        or int(labels.max()) >= _CLASS_COUNT
    ):
        raise SelectiveTopologyAcquisitionError(code, "post-topology H-prime changed exact label ABI")
    if scorer_rgb.dtype != np.uint8 or scorer_rgb.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS):
        raise SelectiveTopologyAcquisitionError(code, "post-topology scorer RGB changed exact ABI")
    if target_rgb.dtype != np.uint8 or target_rgb.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS):
        raise SelectiveTopologyAcquisitionError(code, "target scorer RGB changed exact ABI")
    if camera_y1.dtype != np.uint8 or camera_y1.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
        raise SelectiveTopologyAcquisitionError(code, "post-topology camera Y1 changed exact ABI")
    if (
        _array_sha256(labels) != post_topology_realized_labels_sha256
        or _array_sha256(scorer_rgb) != post_topology_scorer_rgb_sha256
        or _array_sha256(target_rgb) != target_scorer_rgb_sha256
        or _array_sha256(camera_y1) != post_topology_camera_y1_sha256
    ):
        raise SelectiveTopologyAcquisitionError(code, "post-topology H-prime/RGB/Y1 content hash mismatch")
    if target_scorer_rgb_sha256 != evidence.target_scorer_rgb_sha256:
        raise SelectiveTopologyAcquisitionError(code, "target scorer RGB differs from baseline target custody")
    baseline_telemetry = evidence.cell_partition_telemetry
    fresh_telemetry = _partition_telemetry(
        evidence.source_pair_ids,
        post_semantic,
        evidence.target_labels,
        labels,
    )
    fresh_mask = np.ascontiguousarray(
        (post_semantic == evidence.target_labels) & (labels != evidence.target_labels),
        dtype=np.bool_,
    )
    receipt = SelectiveTopologyFreshG8RequestReceiptV1(
        schema=FRESH_G8_RECEIPT_SCHEMA,
        evidence_binding_sha256=evidence.binding_sha256,
        proposal_id=proposal.proposal_id,
        selective_g_packet_sha256=proposal.receipt.packet_sha256,
        selective_g_packet_bytes=proposal.receipt.packet_bytes,
        selective_g_compile_receipt_binding_sha256=proposal.compiled.receipt_binding_sha256,
        source_pair_ids=evidence.source_pair_ids,
        post_topology_semantic_labels_sha256=_array_sha256(post_semantic),
        target_labels_sha256=evidence.target_labels_sha256,
        post_topology_realized_labels_sha256=post_topology_realized_labels_sha256,
        post_topology_scorer_rgb_sha256=post_topology_scorer_rgb_sha256,
        target_scorer_rgb_sha256=target_scorer_rgb_sha256,
        post_topology_camera_y1_sha256=post_topology_camera_y1_sha256,
        frozen_scorer_sha256=evidence.frozen_scorer_sha256,
        baseline_forward_receipt_sha256=evidence.current_realization_forward_receipt_sha256,
        post_topology_forward_receipt_sha256=post_topology_forward_receipt_sha256,
        target_forward_receipt_sha256=evidence.target_forward_receipt_sha256,
        baseline_realization_debt_mask_sha256=baseline_telemetry.realization_debt_mask_sha256,
        fresh_realization_debt_mask_sha256=_array_sha256(fresh_mask),
        fresh_realization_debt_cell_count=int(np.count_nonzero(fresh_mask)),
        baseline_cell_partition_telemetry=baseline_telemetry,
        fresh_cell_partition_telemetry=fresh_telemetry,
        g12_selective_base_status=_G12_BLOCKER,
        production_envelope_status=_PRODUCTION_BLOCKER,
    )
    if parse_selective_topology_fresh_g8_request_receipt(receipt.to_receipt_bytes()) != receipt:
        raise SelectiveTopologyAcquisitionError(code, "fresh-G8 receipt failed strict parse/re-emit")
    return SelectiveTopologyFreshG8RequestV1(
        source_pair_ids=evidence.source_pair_ids,
        selective_g_program_proposal=proposal,
        post_topology_semantic_labels=post_semantic,
        target_labels=evidence.target_labels,
        post_topology_realized_labels=labels,
        post_topology_scorer_rgb=scorer_rgb,
        target_scorer_rgb=target_rgb,
        post_topology_camera_y1=camera_y1,
        fresh_realization_debt_mask=fresh_mask,
        receipt=receipt,
    )


SelectiveTopologyWholeBundleBuilderV1: TypeAlias = Callable[
    [TaskspaceSectionBundleV1, SelectiveTopologyProgramProposalV1],
    TaskspaceSectionBundleV1,
]


def _validate_structural_a_packet(
    packet: bytes,
    *,
    expected_magic: bytes,
    expected_version: int,
    header: struct.Struct,
    expected_source_pair_ids: tuple[int, ...],
) -> None:
    """Validate the source-independent closed A envelope and exact CRC/EOF."""

    code = SelectiveTopologyBlockerCodeV1.G7_WHOLE_BUNDLE_REFUSAL
    if type(packet) is not bytes or len(packet) < header.size:
        raise SelectiveTopologyAcquisitionError(code, "rebuilt A packet is truncated or mutable")
    try:
        values = list(header.unpack_from(packet, 0))
    except struct.error as exc:  # pragma: no cover - guarded length
        raise SelectiveTopologyAcquisitionError(code, "rebuilt A header is malformed") from exc
    magic, version, mode_wire, pair_start, pair_count, _source_binding, row_count, body_bytes, packet_bytes, crc = (
        values
    )
    if magic != expected_magic or version != expected_version or mode_wire not in {0, 1, 2}:
        raise SelectiveTopologyAcquisitionError(code, "rebuilt A packet escaped its closed magic/version/mode")
    if (
        pair_start != expected_source_pair_ids[0]
        or pair_count != len(expected_source_pair_ids)
        or packet_bytes != len(packet)
        or body_bytes != len(packet) - header.size
    ):
        raise SelectiveTopologyAcquisitionError(code, "rebuilt A pair/length accounting changed")
    expected_row_size = {0: 0, 1: A3_RGB_ROW.size, 2: A3_COPY_ROW.size}[mode_wire]
    if (mode_wire == 0 and (row_count != 0 or body_bytes != 0)) or (
        mode_wire != 0 and (row_count < 1 or body_bytes != row_count * expected_row_size)
    ):
        raise SelectiveTopologyAcquisitionError(code, "rebuilt A body differs from its closed row ABI")
    values[-1] = 0
    if zlib.crc32(header.pack(*values) + packet[header.size :]) & 0xFFFFFFFF != crc:
        raise SelectiveTopologyAcquisitionError(code, "rebuilt A packet CRC differs from exact bytes")


def validate_selective_topology_g7_bundle(
    baseline_bundle: TaskspaceSectionBundleV1,
    rebuilt_bundle: TaskspaceSectionBundleV1,
    proposal: SelectiveTopologyProgramProposalV1,
) -> SelectiveTopologyG7DomainV1:
    """Validate one lazily rebuilt P/G/A bundle without running its receiver."""

    code = SelectiveTopologyBlockerCodeV1.G7_WHOLE_BUNDLE_REFUSAL
    if type(baseline_bundle) is not TaskspaceSectionBundleV1 or type(rebuilt_bundle) is not TaskspaceSectionBundleV1:
        raise SelectiveTopologyAcquisitionError(code, "G7 hook requires new exact TaskspaceSectionBundleV1 objects")
    if type(proposal) is not SelectiveTopologyProgramProposalV1:
        raise SelectiveTopologyAcquisitionError(code, "G7 hook proposal changed exact type")
    if rebuilt_bundle is baseline_bundle or rebuilt_bundle.bundle_sha256 == baseline_bundle.bundle_sha256:
        raise SelectiveTopologyAcquisitionError(code, "whole-bundle builder returned the baseline object/no-op")
    if (
        _sha256(baseline_bundle.predictor_packet) != proposal.receipt.base_p_section_sha256
        or len(baseline_bundle.predictor_packet) != proposal.receipt.base_p_section_bytes
    ):
        raise SelectiveTopologyAcquisitionError(code, "G7 baseline P differs from selective proposal custody")
    if rebuilt_bundle.predictor_packet != baseline_bundle.predictor_packet:
        raise SelectiveTopologyAcquisitionError(code, "whole-bundle builder mutated exact counted P")
    if rebuilt_bundle.coupled_preimage_packet == baseline_bundle.coupled_preimage_packet:
        raise SelectiveTopologyAcquisitionError(code, "whole-bundle builder retained stale A bytes after G selection")
    if baseline_bundle.terminal_quotient_packet is not None or rebuilt_bundle.terminal_quotient_packet is not None:
        raise SelectiveTopologyAcquisitionError(code, "selective topology G7 hook does not admit optional T")
    g_packet = rebuilt_bundle.generative_correction_packet
    a_packet = rebuilt_bundle.coupled_preimage_packet
    if g_packet.startswith(PASS_G_ENVELOPE_MAGIC) or a_packet.startswith(PASS_A_PACKET_MAGIC):
        raise SelectiveTopologyAcquisitionError(
            code, "PASS production G/A domains are not selective semantic-G results"
        )
    if g_packet == proposal.compiled.packet:
        if not g_packet.startswith(SEMANTIC_G_PACKET_MAGIC):
            raise SelectiveTopologyAcquisitionError(code, "plain selective G lost exact TACG1C magic")
        _validate_structural_a_packet(
            a_packet,
            expected_magic=A3_PACKET_MAGIC,
            expected_version=A3_PACKET_VERSION,
            header=A3_PACKET_HEADER,
            expected_source_pair_ids=proposal.receipt.source_pair_ids,
        )
        return SelectiveTopologyG7DomainV1.PLAIN_TACG1C_TACA3P1
    if g_packet.startswith(COMPOSITE_G_SECTION_MAGIC):
        try:
            parsed = parse_same_class_realization_repair_g_section(g_packet)
        except SameClassRealizationRepairError as exc:
            raise SelectiveTopologyAcquisitionError(code, "composite selective G failed strict TACG8S1 parse") from exc
        if parsed.semantic_g_packet != proposal.compiled.packet:
            raise SelectiveTopologyAcquisitionError(code, "TACG8S1 substituted a foreign inner selective TACG1C packet")
        _validate_structural_a_packet(
            a_packet,
            expected_magic=POST_G8_A_PACKET_MAGIC,
            expected_version=POST_G8_A_PACKET_VERSION,
            header=POST_G8_A_PACKET_HEADER,
            expected_source_pair_ids=proposal.receipt.source_pair_ids,
        )
        return SelectiveTopologyG7DomainV1.COMPOSITE_TACG8S1_TACA8P1
    raise SelectiveTopologyAcquisitionError(code, "rebuilt G is neither exact plain TACG1C nor strict matching TACG8S1")


def _build_and_validate_g7_bundle(
    baseline_bundle: TaskspaceSectionBundleV1,
    *,
    proposal: SelectiveTopologyProgramProposalV1,
    whole_bundle_builder: SelectiveTopologyWholeBundleBuilderV1,
) -> TaskspaceSectionBundleV1:
    code = SelectiveTopologyBlockerCodeV1.G7_WHOLE_BUNDLE_REFUSAL
    if type(baseline_bundle) is not TaskspaceSectionBundleV1:
        raise SelectiveTopologyAcquisitionError(code, "lazy G7 transform received a foreign baseline bundle type")
    try:
        rebuilt = whole_bundle_builder(baseline_bundle, proposal)
    except SelectiveTopologyAcquisitionError:
        raise
    except Exception as exc:
        raise SelectiveTopologyAcquisitionError(code, "typed whole-bundle builder refused selective proposal") from exc
    validate_selective_topology_g7_bundle(baseline_bundle, rebuilt, proposal)
    return rebuilt


def make_selective_topology_g7_proposals(
    acquisition: SelectiveTopologyAcquisitionV1,
    *,
    whole_bundle_builder: SelectiveTopologyWholeBundleBuilderV1,
) -> tuple[TaskspaceWholeArchiveProposalV1, ...]:
    """Create lazy G7 proposals only for exact compiled-packet-byte uniques."""

    code = SelectiveTopologyBlockerCodeV1.G7_WHOLE_BUNDLE_REFUSAL
    if type(acquisition) is not SelectiveTopologyAcquisitionV1:
        raise SelectiveTopologyAcquisitionError(code, "G7 hook requires exact selective acquisition")
    if not callable(whole_bundle_builder):
        raise SelectiveTopologyAcquisitionError(code, "whole_bundle_builder must be callable")
    if acquisition.production_envelope_status is not _PRODUCTION_BLOCKER:
        raise SelectiveTopologyAcquisitionError(code, "G7 hook cannot relax production-envelope blocker")
    return tuple(
        TaskspaceWholeArchiveProposalV1(
            proposal_id=f"stg7:{index}:{proposal.receipt.packet_sha256[:16]}",
            transform=partial(
                _build_and_validate_g7_bundle,
                proposal=proposal,
                whole_bundle_builder=whole_bundle_builder,
            ),
        )
        for index, proposal in enumerate(acquisition.unique_program_proposals)
    )


__all__ = [
    "MAX_PREFIX_REQUESTS",
    "MAX_SINGLETON_PREFIX_CELLS",
    "MAX_TOPOLOGY_EVENTS",
    "EncoderOnlySelectiveTopologyEvidenceV1",
    "PredictorStateV1OrV2",
    "SelectiveTopologyAcquisitionError",
    "SelectiveTopologyAcquisitionPlanV1",
    "SelectiveTopologyAcquisitionReceiptV1",
    "SelectiveTopologyAcquisitionV1",
    "SelectiveTopologyBlockerCodeV1",
    "SelectiveTopologyFreshG8RequestReceiptV1",
    "SelectiveTopologyFreshG8RequestV1",
    "SelectiveTopologyG7DomainV1",
    "SelectiveTopologyG12SeamStatusV1",
    "SelectiveTopologyGrammarFamilyV1",
    "SelectiveTopologyPrefixOrderV1",
    "SelectiveTopologyProductionEnvelopeStatusV1",
    "SelectiveTopologyProgramProposalReceiptV1",
    "SelectiveTopologyProgramProposalV1",
    "SelectiveTopologyTransitionCountV1",
    "SelectiveTopologyWholeBundleBuilderV1",
    "acquire_selective_topology_programs",
    "bind_selective_topology_post_forward",
    "make_selective_topology_g7_proposals",
    "parse_selective_topology_acquisition_receipt",
    "parse_selective_topology_fresh_g8_request_receipt",
    "parse_selective_topology_program_proposal_receipt",
    "validate_selective_topology_g7_bundle",
]
