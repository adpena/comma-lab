# SPDX-License-Identifier: MIT
"""Encoder-only acquisition for G8 same-class realization repair programs.

The G8 receiver owns a counted class-preserving camera coordinate.  This
module owns the inverse/acquisition side: it identifies cells whose semantic G
label already equals the target while the frozen-scorer realization does not,
then emits deterministic families of concrete G8 run programs for whole-object
arbitration.

Dense labels and scorer-plane RGB are encoder-only evidence.  They are hashed
and copied into an immutable custody object, but never serialized by this
module.  The only values eligible for a later counted G8 payload are the run
coordinates and RGB triples contained in ``SameClassRealizationRepairProgramV1``.

No function here invokes a scorer, closes realization through R, selects a
candidate, computes a contest score, or claims frontier movement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Final, Literal

import numpy as np

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    MAX_BOUNDED_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    MAX_RUNS,
    PACKET_HEADER,
    RUN_ROW,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
)

EVIDENCE_SCHEMA: Final = "tac.encoder_only_same_class_realization_evidence.v1"
RECEIPT_SCHEMA: Final = "tac.same_class_realization_encoder_proposal_receipt.v1"
ACQUISITION_SCHEMA: Final = "tac.same_class_realization_encoder_acquisition.v1"
CELL_PARTITION_SCHEMA: Final = "tac.same_class_realization_cell_partition_telemetry.v1"
N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256: Final = "2af9b50d70f342224aa438e95b4d53a05be3f253709c1fa1835da089f37e0f61"
N2_STAGE_ABLATION_PRIORITY_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
N2_STAGE_ABLATION_PRIORITY_SCOPE: Final = "n2_stage_ablation_ordering_only.v1"

_CLASS_TO_ROLE: Final = (
    SameClassSemanticRoleV1.ROAD,
    SameClassSemanticRoleV1.LANE,
    SameClassSemanticRoleV1.UNDRIVABLE_BOUNDARY,
    SameClassSemanticRoleV1.MOVABLE,
    SameClassSemanticRoleV1.MY_CAR,
)
_CLASS_COUNT: Final = len(_CLASS_TO_ROLE)


class SameClassRealizationEncoderError(ValueError):
    """Encoder custody, debt acquisition, or proposal proof failed."""


class SameClassRealizationProposalFamilyV1(StrEnum):
    """Distinct target-derived RGB program families."""

    CLASS_SHARED_TARGET_MEDOID_V1 = "CLASS_SHARED_TARGET_MEDOID_V1"
    CLASS_BOUNDED_TARGET_MEDOIDS_V1 = "CLASS_BOUNDED_TARGET_MEDOIDS_V1"
    TARGET_PIXEL_RGB_ORACLE_CONTROL_V1 = "TARGET_PIXEL_RGB_ORACLE_CONTROL_V1"


class SameClassRealizationPrefixOrderV1(StrEnum):
    """Two non-equivalent, always-emitted prefix interpretations."""

    CANONICAL_ADDRESS_V1 = "CANONICAL_ADDRESS_V1"
    TARGET_RGB_SSE_DESCENDING_V1 = "TARGET_RGB_SSE_DESCENDING_V1"


class SameClassRealizationBaseInterpretationV1(StrEnum):
    """The two source-bound semantic/Y1 bases; never silently conflated."""

    EXACT_SEMANTIC_G_CONTROL_V1 = "EXACT_SEMANTIC_G_CONTROL_V1"
    PASS_PREDICTOR_SEMANTIC_TOPOLOGY_V1 = "PASS_PREDICTOR_SEMANTIC_TOPOLOGY_V1"


class SameClassRealizationG8CompileSeamV1(StrEnum):
    """Exact status of the frozen G8 compiler for one base interpretation."""

    EXISTING_COUNTED_SEMANTIC_G_V1 = "EXISTING_COUNTED_SEMANTIC_G_V1"
    PASS_BASE_REQUIRES_NONEMPTY_COUNTED_G8_BASE_ENVELOPE_V1 = "PASS_BASE_REQUIRES_NONEMPTY_COUNTED_G8_BASE_ENVELOPE_V1"


_PREFIX_ORDERS: Final = tuple(SameClassRealizationPrefixOrderV1)
_FAMILY_PRIORITY: Final = {
    SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1: 0,
    SameClassRealizationProposalFamilyV1.CLASS_BOUNDED_TARGET_MEDOIDS_V1: 1,
    SameClassRealizationProposalFamilyV1.CLASS_SHARED_TARGET_MEDOID_V1: 2,
}


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
        raise SameClassRealizationEncoderError(
            "same-class realization encoder record is not finite canonical ASCII JSON"
        ) from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SameClassRealizationEncoderError(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SameClassRealizationEncoderError(f"{field} must be an exact integer in [{minimum},{maximum}]")
    return value


def _validate_pair_ids(source_pair_ids: tuple[int, ...]) -> tuple[int, ...]:
    if (
        type(source_pair_ids) is not tuple
        or not 1 <= len(source_pair_ids) <= MAX_BOUNDED_PAIRS
        or any(type(pair_id) is not int for pair_id in source_pair_ids)
    ):
        raise SameClassRealizationEncoderError("source_pair_ids must be a bounded nonempty exact tuple")
    expected = tuple(range(source_pair_ids[0], source_pair_ids[0] + len(source_pair_ids)))
    if source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise SameClassRealizationEncoderError("source_pair_ids must be contiguous inside [0,600)")
    return expected


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _rgb_tuple(value: np.ndarray) -> tuple[int, int, int]:
    array = np.asarray(value)
    if array.shape != (3,):
        raise SameClassRealizationEncoderError("RGB tuple source must contain exactly three channels")
    return int(array[0]), int(array[1]), int(array[2])


def _five_int_tuple(values: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    if len(values) != _CLASS_COUNT:
        raise SameClassRealizationEncoderError("class-count tuple must contain exactly five integers")
    return values[0], values[1], values[2], values[3], values[4]


@dataclass(frozen=True, slots=True)
class SameClassRealizationCellPartitionTelemetryV1:
    """Dense-free exhaustive Z/T/H cell-partition telemetry.

    ``Z`` is the current semantic field, ``T`` is the target label field, and
    ``H`` is the realized frozen-scorer label field.  Counts by class always
    use the target class.
    """

    source_pair_ids: tuple[int, ...]
    label_shape: tuple[int, int, int]
    current_semantic_labels_sha256: str
    target_labels_sha256: str
    realized_labels_sha256: str
    closed_mask_sha256: str
    realization_debt_mask_sha256: str
    topology_debt_mask_sha256: str
    fortunate_semantic_mismatch_mask_sha256: str
    closed_cell_count: int
    realization_debt_cell_count: int
    topology_debt_cell_count: int
    fortunate_semantic_mismatch_cell_count: int
    closed_count_by_target_class: tuple[int, int, int, int, int]
    realization_debt_count_by_target_class: tuple[int, int, int, int, int]
    topology_debt_count_by_target_class: tuple[int, int, int, int, int]
    fortunate_semantic_mismatch_count_by_target_class: tuple[int, int, int, int, int]
    schema: Literal["tac.same_class_realization_cell_partition_telemetry.v1"] = CELL_PARTITION_SCHEMA
    exact_partition_definition: Literal[
        "closed_z_eq_t_h_eq_t__realization_z_eq_t_h_ne_t__topology_z_ne_t_h_ne_t__fortunate_z_ne_t_h_eq_t.v1"
    ] = "closed_z_eq_t_h_eq_t__realization_z_eq_t_h_ne_t__topology_z_ne_t_h_ne_t__fortunate_z_ne_t_h_eq_t.v1"
    dense_masks_serialized: Literal[False] = False
    counts_by_target_class: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != CELL_PARTITION_SCHEMA:
            raise SameClassRealizationEncoderError("cell-partition telemetry schema changed")
        pair_ids = _validate_pair_ids(self.source_pair_ids)
        expected_shape = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
        if type(self.label_shape) is not tuple or self.label_shape != expected_shape:
            raise SameClassRealizationEncoderError("cell-partition label shape differs from source-pair window")
        for field in (
            "current_semantic_labels_sha256",
            "target_labels_sha256",
            "realized_labels_sha256",
            "closed_mask_sha256",
            "realization_debt_mask_sha256",
            "topology_debt_mask_sha256",
            "fortunate_semantic_mismatch_mask_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        count_fields = (
            "closed_cell_count",
            "realization_debt_cell_count",
            "topology_debt_cell_count",
            "fortunate_semantic_mismatch_cell_count",
        )
        for field in count_fields:
            _require_exact_int(
                getattr(self, field),
                field,
                minimum=0,
                maximum=MAX_BOUNDED_PAIRS * SCORER_HEIGHT * SCORER_WIDTH,
            )
        by_class_fields = (
            ("closed_count_by_target_class", self.closed_cell_count),
            ("realization_debt_count_by_target_class", self.realization_debt_cell_count),
            ("topology_debt_count_by_target_class", self.topology_debt_cell_count),
            (
                "fortunate_semantic_mismatch_count_by_target_class",
                self.fortunate_semantic_mismatch_cell_count,
            ),
        )
        for field, expected_count in by_class_fields:
            values = getattr(self, field)
            if (
                type(values) is not tuple
                or len(values) != _CLASS_COUNT
                or any(type(value) is not int or value < 0 for value in values)
                or sum(values) != expected_count
            ):
                raise SameClassRealizationEncoderError(f"{field} must close as five target-class counts")
        expected_cells = len(pair_ids) * SCORER_HEIGHT * SCORER_WIDTH
        if sum(getattr(self, field) for field in count_fields) != expected_cells:
            raise SameClassRealizationEncoderError("four-way cell partition is not exhaustive")
        if self.dense_masks_serialized is not False or self.counts_by_target_class is not True:
            raise SameClassRealizationEncoderError("cell-partition telemetry truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["label_shape"] = list(self.label_shape)
        for field in (
            "closed_count_by_target_class",
            "realization_debt_count_by_target_class",
            "topology_debt_count_by_target_class",
            "fortunate_semantic_mismatch_count_by_target_class",
        ):
            value[field] = list(getattr(self, field))
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def telemetry_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def _cell_partition_masks(
    current_semantic: np.ndarray,
    target: np.ndarray,
    realized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    semantic_matches = current_semantic == target
    realized_matches = realized == target
    return (
        np.ascontiguousarray(semantic_matches & realized_matches, dtype=np.bool_),
        np.ascontiguousarray(semantic_matches & ~realized_matches, dtype=np.bool_),
        np.ascontiguousarray(~semantic_matches & ~realized_matches, dtype=np.bool_),
        np.ascontiguousarray(~semantic_matches & realized_matches, dtype=np.bool_),
    )


def _mask_counts_by_target_class(
    mask: np.ndarray,
    target: np.ndarray,
) -> tuple[int, int, int, int, int]:
    return _five_int_tuple(
        tuple(int(np.count_nonzero(mask & (target == class_id))) for class_id in range(_CLASS_COUNT))
    )


@dataclass(frozen=True, slots=True)
class EncoderOnlySameClassRealizationEvidenceV1:
    """Hashed dense encoder evidence with deliberately no serializer."""

    source_pair_ids: tuple[int, ...]
    base_interpretation: SameClassRealizationBaseInterpretationV1
    base_semantic_binding_sha256: str
    base_p_section_sha256: str
    base_p_section_bytes: int
    base_g_section_sha256: str | None
    base_g_section_bytes: int | None
    base_camera_y1_sha256: str
    frozen_scorer_sha256: str
    candidate_forward_receipt_sha256: str
    target_forward_receipt_sha256: str
    priority_ordering_receipt_sha256: str
    priority_ordering_axis: Literal["[macOS-CPU frozen-scorer advisory]"]
    priority_ordering_scope: Literal["n2_stage_ablation_ordering_only.v1"]
    current_semantic_labels_sha256: str
    target_labels_sha256: str
    realized_labels_sha256: str
    candidate_scorer_rgb_sha256: str
    target_scorer_rgb_sha256: str
    current_semantic_labels: np.ndarray
    target_labels: np.ndarray
    realized_labels: np.ndarray
    candidate_scorer_rgb: np.ndarray
    target_scorer_rgb: np.ndarray
    schema: Literal["tac.encoder_only_same_class_realization_evidence.v1"] = EVIDENCE_SCHEMA
    serialized_dense_evidence_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        if self.schema != EVIDENCE_SCHEMA or self.serialized_dense_evidence_bytes != 0:
            raise SameClassRealizationEncoderError("dense realization evidence must remain encoder-only")
        pair_ids = _validate_pair_ids(self.source_pair_ids)
        if type(self.base_interpretation) is not SameClassRealizationBaseInterpretationV1:
            raise SameClassRealizationEncoderError("base interpretation escaped the closed universe")
        for field in (
            "base_semantic_binding_sha256",
            "base_p_section_sha256",
            "base_camera_y1_sha256",
            "frozen_scorer_sha256",
            "candidate_forward_receipt_sha256",
            "target_forward_receipt_sha256",
            "priority_ordering_receipt_sha256",
            "current_semantic_labels_sha256",
            "target_labels_sha256",
            "realized_labels_sha256",
            "candidate_scorer_rgb_sha256",
            "target_scorer_rgb_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        _require_exact_int(
            self.base_p_section_bytes,
            "base_p_section_bytes",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
        if self.base_interpretation is SameClassRealizationBaseInterpretationV1.EXACT_SEMANTIC_G_CONTROL_V1:
            _require_sha256(self.base_g_section_sha256, "base_g_section_sha256")
            _require_exact_int(
                self.base_g_section_bytes,
                "base_g_section_bytes",
                minimum=1,
                maximum=0xFFFFFFFF,
            )
            if self.base_g_section_sha256 == self.base_p_section_sha256:
                raise SameClassRealizationEncoderError("exact semantic G and P identities must differ")
        elif self.base_g_section_sha256 is not None or self.base_g_section_bytes is not None:
            raise SameClassRealizationEncoderError(
                "PASS predictor-semantic base must not invent an empty or synthetic G section"
            )
        if (
            self.priority_ordering_receipt_sha256 != N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256
            or self.priority_ordering_axis != N2_STAGE_ABLATION_PRIORITY_AXIS
            or self.priority_ordering_scope != N2_STAGE_ABLATION_PRIORITY_SCOPE
        ):
            raise SameClassRealizationEncoderError("priority receipt may authorize only n2 macOS-CPU advisory ordering")

        label_shape = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
        rgb_shape = (*label_shape, 3)
        label_fields = (
            ("current_semantic_labels", self.current_semantic_labels, self.current_semantic_labels_sha256),
            ("target_labels", self.target_labels, self.target_labels_sha256),
            ("realized_labels", self.realized_labels, self.realized_labels_sha256),
        )
        for field, raw, expected_sha256 in label_fields:
            array = np.asarray(raw)
            if array.dtype != np.uint8 or array.shape != label_shape:
                raise SameClassRealizationEncoderError(f"{field} must be exact uint8 [pair,384,512]")
            if int(array.min()) < 0 or int(array.max()) >= _CLASS_COUNT:
                raise SameClassRealizationEncoderError(f"{field} escaped the closed five-class universe")
            immutable = _immutable_array(array, dtype=np.dtype(np.uint8))
            if _array_sha256(immutable) != expected_sha256:
                raise SameClassRealizationEncoderError(f"{field} content hash mismatch")
            object.__setattr__(self, field, immutable)

        rgb_fields = (
            ("candidate_scorer_rgb", self.candidate_scorer_rgb, self.candidate_scorer_rgb_sha256),
            ("target_scorer_rgb", self.target_scorer_rgb, self.target_scorer_rgb_sha256),
        )
        for field, raw, expected_sha256 in rgb_fields:
            array = np.asarray(raw)
            if array.dtype != np.uint8 or array.shape != rgb_shape:
                raise SameClassRealizationEncoderError(f"{field} must be exact uint8 [pair,384,512,3]")
            immutable = _immutable_array(array, dtype=np.dtype(np.uint8))
            if _array_sha256(immutable) != expected_sha256:
                raise SameClassRealizationEncoderError(f"{field} content hash mismatch")
            object.__setattr__(self, field, immutable)

    @property
    def binding_sha256(self) -> str:
        """Hash custody without exposing dense evidence bytes."""

        return _sha256(
            _canonical_json(
                {
                    "schema": self.schema,
                    "source_pair_ids": list(self.source_pair_ids),
                    "base_interpretation": self.base_interpretation.value,
                    "base_semantic_binding_sha256": self.base_semantic_binding_sha256,
                    "base_p_section_sha256": self.base_p_section_sha256,
                    "base_p_section_bytes": self.base_p_section_bytes,
                    "base_g_section_sha256": self.base_g_section_sha256,
                    "base_g_section_bytes": self.base_g_section_bytes,
                    "base_camera_y1_sha256": self.base_camera_y1_sha256,
                    "frozen_scorer_sha256": self.frozen_scorer_sha256,
                    "candidate_forward_receipt_sha256": self.candidate_forward_receipt_sha256,
                    "target_forward_receipt_sha256": self.target_forward_receipt_sha256,
                    "priority_ordering_receipt_sha256": self.priority_ordering_receipt_sha256,
                    "priority_ordering_axis": self.priority_ordering_axis,
                    "priority_ordering_scope": self.priority_ordering_scope,
                    "current_semantic_labels_sha256": self.current_semantic_labels_sha256,
                    "target_labels_sha256": self.target_labels_sha256,
                    "realized_labels_sha256": self.realized_labels_sha256,
                    "candidate_scorer_rgb_sha256": self.candidate_scorer_rgb_sha256,
                    "target_scorer_rgb_sha256": self.target_scorer_rgb_sha256,
                    "label_shape": list(self.current_semantic_labels.shape),
                    "rgb_shape": list(self.candidate_scorer_rgb.shape),
                    "serialized_dense_evidence_bytes": 0,
                }
            )
        )

    @property
    def g8_compile_seam(self) -> SameClassRealizationG8CompileSeamV1:
        if self.base_interpretation is SameClassRealizationBaseInterpretationV1.EXACT_SEMANTIC_G_CONTROL_V1:
            return SameClassRealizationG8CompileSeamV1.EXISTING_COUNTED_SEMANTIC_G_V1
        return SameClassRealizationG8CompileSeamV1.PASS_BASE_REQUIRES_NONEMPTY_COUNTED_G8_BASE_ENVELOPE_V1

    @property
    def base_interpretation_has_frozen_g8_compile_path(self) -> bool:
        """Whether frozen G8 already accepts this base shape; no compile proof."""

        return self.base_interpretation is SameClassRealizationBaseInterpretationV1.EXACT_SEMANTIC_G_CONTROL_V1

    @property
    def cell_partition_telemetry(self) -> SameClassRealizationCellPartitionTelemetryV1:
        closed, realization, topology, fortunate = _cell_partition_masks(
            self.current_semantic_labels,
            self.target_labels,
            self.realized_labels,
        )
        return SameClassRealizationCellPartitionTelemetryV1(
            source_pair_ids=self.source_pair_ids,
            label_shape=(len(self.source_pair_ids), SCORER_HEIGHT, SCORER_WIDTH),
            current_semantic_labels_sha256=self.current_semantic_labels_sha256,
            target_labels_sha256=self.target_labels_sha256,
            realized_labels_sha256=self.realized_labels_sha256,
            closed_mask_sha256=_array_sha256(closed),
            realization_debt_mask_sha256=_array_sha256(realization),
            topology_debt_mask_sha256=_array_sha256(topology),
            fortunate_semantic_mismatch_mask_sha256=_array_sha256(fortunate),
            closed_cell_count=int(np.count_nonzero(closed)),
            realization_debt_cell_count=int(np.count_nonzero(realization)),
            topology_debt_cell_count=int(np.count_nonzero(topology)),
            fortunate_semantic_mismatch_cell_count=int(np.count_nonzero(fortunate)),
            closed_count_by_target_class=_mask_counts_by_target_class(
                closed,
                self.target_labels,
            ),
            realization_debt_count_by_target_class=_mask_counts_by_target_class(
                realization,
                self.target_labels,
            ),
            topology_debt_count_by_target_class=_mask_counts_by_target_class(
                topology,
                self.target_labels,
            ),
            fortunate_semantic_mismatch_count_by_target_class=_mask_counts_by_target_class(
                fortunate,
                self.target_labels,
            ),
        )

    @property
    def debt_mask(self) -> np.ndarray:
        return _cell_partition_masks(
            self.current_semantic_labels,
            self.target_labels,
            self.realized_labels,
        )[1]

    @property
    def debt_count(self) -> int:
        return int(np.count_nonzero(self.debt_mask))

    @property
    def debt_count_by_class(self) -> tuple[int, int, int, int, int]:
        debt = self.debt_mask
        return _five_int_tuple(
            tuple(int(np.count_nonzero(debt & (self.target_labels == class_id))) for class_id in range(_CLASS_COUNT))
        )


@dataclass(frozen=True, slots=True)
class SameClassRealizationAcquisitionPlanV1:
    """Caller-enumerated palette bounds and geometric prefix ratios."""

    palette_sizes_per_class: tuple[int, ...]
    geometric_prefix_ratios: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.palette_sizes_per_class) is not tuple or not self.palette_sizes_per_class:
            raise SameClassRealizationEncoderError("palette_sizes_per_class must be a nonempty exact tuple")
        if any(type(value) is not int or not 2 <= value <= MAX_RUNS for value in self.palette_sizes_per_class):
            raise SameClassRealizationEncoderError("bounded palette sizes must be exact integers in [2,MAX_RUNS]")
        if self.palette_sizes_per_class != tuple(sorted(set(self.palette_sizes_per_class))):
            raise SameClassRealizationEncoderError("bounded palette sizes must be sorted and unique")
        if type(self.geometric_prefix_ratios) is not tuple or not self.geometric_prefix_ratios:
            raise SameClassRealizationEncoderError("geometric_prefix_ratios must be a nonempty exact tuple")
        if any(type(value) is not int or not 2 <= value <= MAX_RUNS for value in self.geometric_prefix_ratios):
            raise SameClassRealizationEncoderError("geometric prefix ratios must be exact integers in [2,MAX_RUNS]")
        if self.geometric_prefix_ratios != tuple(sorted(set(self.geometric_prefix_ratios))):
            raise SameClassRealizationEncoderError("geometric prefix ratios must be sorted and unique")

    def prefix_cell_counts(self, debt_count: int) -> tuple[int, ...]:
        _require_exact_int(
            debt_count,
            "debt_count",
            minimum=1,
            maximum=MAX_BOUNDED_PAIRS * SCORER_HEIGHT * SCORER_WIDTH,
        )
        counts = {debt_count}
        for ratio in self.geometric_prefix_ratios:
            current = 1
            while current < debt_count:
                counts.add(current)
                current = min(debt_count, current * ratio)
        return tuple(sorted(counts))


@dataclass(frozen=True, slots=True)
class SameClassRealizationProposalReceiptV1:
    """Dense-free exact acquisition receipt for one concrete G8 program."""

    schema: Literal["tac.same_class_realization_encoder_proposal_receipt.v1"]
    proposal_id: str
    evidence_binding_sha256: str
    base_interpretation: SameClassRealizationBaseInterpretationV1
    base_semantic_binding_sha256: str
    base_p_section_sha256: str
    base_p_section_bytes: int
    base_g_section_sha256: str | None
    base_g_section_bytes: int | None
    base_camera_y1_sha256: str
    g8_compile_seam: SameClassRealizationG8CompileSeamV1
    base_interpretation_has_frozen_g8_compile_path: bool
    current_semantic_labels_sha256: str
    target_labels_sha256: str
    realized_labels_sha256: str
    candidate_scorer_rgb_sha256: str
    target_scorer_rgb_sha256: str
    priority_ordering_receipt_sha256: str
    priority_ordering_axis: Literal["[macOS-CPU frozen-scorer advisory]"]
    priority_ordering_scope: Literal["n2_stage_ablation_ordering_only.v1"]
    cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    family: SameClassRealizationProposalFamilyV1
    prefix_order: SameClassRealizationPrefixOrderV1
    palette_bound_per_class: int | None
    actual_palette_count_by_class: tuple[int, int, int, int, int]
    geometric_prefix_ratios: tuple[int, ...]
    requested_prefix_cell_count: int
    total_debt_cell_count: int
    covered_debt_cell_count: int
    omitted_debt_cell_count: int
    total_debt_count_by_class: tuple[int, int, int, int, int]
    covered_debt_count_by_class: tuple[int, int, int, int, int]
    omitted_debt_count_by_class: tuple[int, int, int, int, int]
    run_count: int
    expanded_cell_count: int
    program_sha256: str
    palette_sha256: str
    estimated_standalone_g8_raw_bytes: int
    selected_target_rgb_sse_before: int
    selected_target_rgb_sse_after: int
    exact_debt_definition: Literal["current_semantic_equals_target_and_realized_label_differs_from_target.v1"] = (
        "current_semantic_equals_target_and_realized_label_differs_from_target.v1"
    )
    program_cells_subset_of_exact_debt: Literal[True] = True
    no_fake_empty_g_section: Literal[True] = True
    target_pixel_oracle_exact_on_selected_cells: bool = False
    priority_ordering_is_advisory_only: Literal[True] = True
    dense_evidence_serialized: Literal[False] = False
    selected_video_derived_rgb_and_coordinates_counted_if_compiled: Literal[True] = True
    public_archive_payload_reused: Literal[False] = False
    g8_compile_invoked: Literal[False] = False
    g8_packet_emitted: Literal[False] = False
    scorer_invoked: Literal[False] = False
    through_r_realization_closed: Literal[False] = False
    requires_through_r_scorer_admission: Literal[True] = True
    rgb_sse_is_encoder_coordinate_only: Literal[True] = True
    whole_object_allocator_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise SameClassRealizationEncoderError("proposal receipt schema changed")
        if (
            type(self.proposal_id) is not str
            or not self.proposal_id
            or len(self.proposal_id) > 192
            or not self.proposal_id.isascii()
        ):
            raise SameClassRealizationEncoderError("proposal_id must be bounded nonempty ASCII")
        for field in (
            "evidence_binding_sha256",
            "base_semantic_binding_sha256",
            "base_p_section_sha256",
            "base_camera_y1_sha256",
            "current_semantic_labels_sha256",
            "target_labels_sha256",
            "realized_labels_sha256",
            "candidate_scorer_rgb_sha256",
            "target_scorer_rgb_sha256",
            "priority_ordering_receipt_sha256",
            "program_sha256",
            "palette_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if type(self.base_interpretation) is not SameClassRealizationBaseInterpretationV1:
            raise SameClassRealizationEncoderError("receipt base interpretation escaped the closed universe")
        if type(self.g8_compile_seam) is not SameClassRealizationG8CompileSeamV1:
            raise SameClassRealizationEncoderError("receipt G8 compile seam escaped the closed universe")
        _require_exact_int(
            self.base_p_section_bytes,
            "base_p_section_bytes",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
        exact_base = self.base_interpretation is SameClassRealizationBaseInterpretationV1.EXACT_SEMANTIC_G_CONTROL_V1
        if exact_base:
            _require_sha256(self.base_g_section_sha256, "base_g_section_sha256")
            _require_exact_int(
                self.base_g_section_bytes,
                "base_g_section_bytes",
                minimum=1,
                maximum=0xFFFFFFFF,
            )
            if (
                self.g8_compile_seam is not SameClassRealizationG8CompileSeamV1.EXISTING_COUNTED_SEMANTIC_G_V1
                or self.base_interpretation_has_frozen_g8_compile_path is not True
            ):
                raise SameClassRealizationEncoderError("exact-G receipt compile seam became inconsistent")
        elif (
            self.base_g_section_sha256 is not None
            or self.base_g_section_bytes is not None
            or self.g8_compile_seam
            is not SameClassRealizationG8CompileSeamV1.PASS_BASE_REQUIRES_NONEMPTY_COUNTED_G8_BASE_ENVELOPE_V1
            or self.base_interpretation_has_frozen_g8_compile_path is not False
        ):
            raise SameClassRealizationEncoderError("PASS receipt must preserve the missing nonempty base-envelope seam")
        if (
            self.priority_ordering_receipt_sha256 != N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256
            or self.priority_ordering_axis != N2_STAGE_ABLATION_PRIORITY_AXIS
            or self.priority_ordering_scope != N2_STAGE_ABLATION_PRIORITY_SCOPE
        ):
            raise SameClassRealizationEncoderError("proposal priority binding escaped advisory ordering scope")
        if type(self.cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1:
            raise SameClassRealizationEncoderError("proposal cell-partition telemetry changed exact type")
        partition = self.cell_partition_telemetry
        if (
            partition.current_semantic_labels_sha256 != self.current_semantic_labels_sha256
            or partition.target_labels_sha256 != self.target_labels_sha256
            or partition.realized_labels_sha256 != self.realized_labels_sha256
            or partition.realization_debt_cell_count != self.total_debt_cell_count
            or partition.realization_debt_count_by_target_class != self.total_debt_count_by_class
        ):
            raise SameClassRealizationEncoderError("proposal debt differs from canonical four-way partition row 2")
        if type(self.family) is not SameClassRealizationProposalFamilyV1:
            raise SameClassRealizationEncoderError("proposal family escaped the closed universe")
        if type(self.prefix_order) is not SameClassRealizationPrefixOrderV1:
            raise SameClassRealizationEncoderError("prefix order escaped the closed universe")
        if self.palette_bound_per_class is not None:
            _require_exact_int(
                self.palette_bound_per_class,
                "palette_bound_per_class",
                minimum=1,
                maximum=MAX_RUNS,
            )
        for field in (
            "actual_palette_count_by_class",
            "total_debt_count_by_class",
            "covered_debt_count_by_class",
            "omitted_debt_count_by_class",
        ):
            values = getattr(self, field)
            if (
                type(values) is not tuple
                or len(values) != _CLASS_COUNT
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise SameClassRealizationEncoderError(f"{field} must be five nonnegative integers")
        if (
            type(self.geometric_prefix_ratios) is not tuple
            or not self.geometric_prefix_ratios
            or any(type(value) is not int or value < 2 for value in self.geometric_prefix_ratios)
        ):
            raise SameClassRealizationEncoderError("receipt geometric_prefix_ratios must remain exact")
        for field in (
            "requested_prefix_cell_count",
            "total_debt_cell_count",
            "covered_debt_cell_count",
            "omitted_debt_cell_count",
            "run_count",
            "expanded_cell_count",
            "estimated_standalone_g8_raw_bytes",
            "selected_target_rgb_sse_before",
            "selected_target_rgb_sse_after",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise SameClassRealizationEncoderError(f"{field} must be an exact nonnegative integer")
        if not (self.requested_prefix_cell_count == self.covered_debt_cell_count == self.expanded_cell_count):
            raise SameClassRealizationEncoderError("proposal prefix/coverage/cell counts disagree")
        if self.total_debt_cell_count != self.covered_debt_cell_count + self.omitted_debt_cell_count:
            raise SameClassRealizationEncoderError("proposal global debt accounting does not close")
        if sum(self.total_debt_count_by_class) != self.total_debt_cell_count:
            raise SameClassRealizationEncoderError("per-class total debt does not close")
        if sum(self.covered_debt_count_by_class) != self.covered_debt_cell_count:
            raise SameClassRealizationEncoderError("per-class covered debt does not close")
        if sum(self.omitted_debt_count_by_class) != self.omitted_debt_cell_count:
            raise SameClassRealizationEncoderError("per-class omitted debt does not close")
        if (
            tuple(
                covered + omitted
                for covered, omitted in zip(
                    self.covered_debt_count_by_class,
                    self.omitted_debt_count_by_class,
                    strict=True,
                )
            )
            != self.total_debt_count_by_class
        ):
            raise SameClassRealizationEncoderError("per-class debt accounting differs cellwise")
        if self.estimated_standalone_g8_raw_bytes != PACKET_HEADER.size + self.run_count * RUN_ROW.size:
            raise SameClassRealizationEncoderError("raw G8 byte estimate differs from frozen wire ABI")
        oracle_expected = self.family is SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1
        if self.target_pixel_oracle_exact_on_selected_cells is not oracle_expected:
            raise SameClassRealizationEncoderError("oracle truth label disagrees with proposal family")
        truth = {
            "program_cells_subset_of_exact_debt": True,
            "no_fake_empty_g_section": True,
            "priority_ordering_is_advisory_only": True,
            "dense_evidence_serialized": False,
            "selected_video_derived_rgb_and_coordinates_counted_if_compiled": True,
            "public_archive_payload_reused": False,
            "g8_compile_invoked": False,
            "g8_packet_emitted": False,
            "scorer_invoked": False,
            "through_r_realization_closed": False,
            "requires_through_r_scorer_admission": True,
            "rgb_sse_is_encoder_coordinate_only": True,
            "whole_object_allocator_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise SameClassRealizationEncoderError("proposal receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["base_interpretation"] = self.base_interpretation.value
        value["g8_compile_seam"] = self.g8_compile_seam.value
        value["family"] = self.family.value
        value["prefix_order"] = self.prefix_order.value
        value["cell_partition_telemetry"] = self.cell_partition_telemetry.as_dict()
        for field in (
            "actual_palette_count_by_class",
            "geometric_prefix_ratios",
            "total_debt_count_by_class",
            "covered_debt_count_by_class",
            "omitted_debt_count_by_class",
        ):
            value[field] = list(value[field])
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def _parse_cell_partition_telemetry_dict(
    value: object,
) -> SameClassRealizationCellPartitionTelemetryV1:
    if type(value) is not dict:
        raise SameClassRealizationEncoderError("cell-partition telemetry must encode as an object")
    raw = dict(value)
    expected = {field.name for field in SameClassRealizationCellPartitionTelemetryV1.__dataclass_fields__.values()}
    if set(raw) != expected:
        raise SameClassRealizationEncoderError("cell-partition telemetry keys changed")
    for field in (
        "source_pair_ids",
        "label_shape",
        "closed_count_by_target_class",
        "realization_debt_count_by_target_class",
        "topology_debt_count_by_target_class",
        "fortunate_semantic_mismatch_count_by_target_class",
    ):
        if type(raw[field]) is not list:
            raise SameClassRealizationEncoderError(f"cell-partition {field} must encode as a JSON list")
        raw[field] = tuple(raw[field])
    try:
        return SameClassRealizationCellPartitionTelemetryV1(**raw)
    except TypeError as exc:
        raise SameClassRealizationEncoderError("cell-partition telemetry contains invalid typed fields") from exc


def parse_same_class_realization_proposal_receipt(
    payload: bytes,
) -> SameClassRealizationProposalReceiptV1:
    """Strictly parse and canonically re-emit one dense-free proposal receipt."""

    if type(payload) is not bytes or not payload:
        raise SameClassRealizationEncoderError("proposal receipt payload must be nonempty exact bytes")
    try:
        raw = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SameClassRealizationEncoderError("proposal receipt is not canonical ASCII JSON") from exc
    if type(raw) is not dict:
        raise SameClassRealizationEncoderError("proposal receipt root must be an object")
    expected = {field.name for field in SameClassRealizationProposalReceiptV1.__dataclass_fields__.values()}
    if set(raw) != expected:
        raise SameClassRealizationEncoderError("proposal receipt keys changed")
    tuple_fields = (
        "actual_palette_count_by_class",
        "geometric_prefix_ratios",
        "total_debt_count_by_class",
        "covered_debt_count_by_class",
        "omitted_debt_count_by_class",
    )
    try:
        for field in tuple_fields:
            if type(raw[field]) is not list:
                raise SameClassRealizationEncoderError(f"{field} must encode as a JSON list")
            raw[field] = tuple(raw[field])
        raw["cell_partition_telemetry"] = _parse_cell_partition_telemetry_dict(raw["cell_partition_telemetry"])
        raw["family"] = SameClassRealizationProposalFamilyV1(raw["family"])
        raw["prefix_order"] = SameClassRealizationPrefixOrderV1(raw["prefix_order"])
        raw["base_interpretation"] = SameClassRealizationBaseInterpretationV1(raw["base_interpretation"])
        raw["g8_compile_seam"] = SameClassRealizationG8CompileSeamV1(raw["g8_compile_seam"])
        receipt = SameClassRealizationProposalReceiptV1(**raw)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SameClassRealizationEncoderError):
            raise
        raise SameClassRealizationEncoderError("proposal receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise SameClassRealizationEncoderError("proposal receipt changed on canonical parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class SameClassRealizationProgramProposalV1:
    """One exact G8 program plus its acquisition receipt."""

    proposal_id: str
    program: SameClassRealizationRepairProgramV1
    receipt: SameClassRealizationProposalReceiptV1

    def __post_init__(self) -> None:
        if type(self.program) is not SameClassRealizationRepairProgramV1:
            raise SameClassRealizationEncoderError("proposal program changed exact G8 type")
        if type(self.receipt) is not SameClassRealizationProposalReceiptV1:
            raise SameClassRealizationEncoderError("proposal receipt changed exact type")
        if self.proposal_id != self.receipt.proposal_id:
            raise SameClassRealizationEncoderError("proposal ID differs from receipt")
        if _program_sha256(self.program) != self.receipt.program_sha256:
            raise SameClassRealizationEncoderError("proposal program differs from receipt digest")
        if self.program.run_count != self.receipt.run_count:
            raise SameClassRealizationEncoderError("proposal run count differs from receipt")
        if self.program.expanded_cell_count != self.receipt.expanded_cell_count:
            raise SameClassRealizationEncoderError("proposal cell count differs from receipt")


@dataclass(frozen=True, slots=True)
class SameClassRealizationAcquisitionV1:
    """Deterministic family of concrete G8 programs for downstream G7."""

    schema: Literal["tac.same_class_realization_encoder_acquisition.v1"]
    evidence_binding_sha256: str
    cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1
    total_debt_cell_count: int
    total_debt_count_by_class: tuple[int, int, int, int, int]
    prefix_cell_counts: tuple[int, ...]
    proposals: tuple[SameClassRealizationProgramProposalV1, ...]
    scorer_invoked: Literal[False] = False
    whole_object_allocator_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != ACQUISITION_SCHEMA:
            raise SameClassRealizationEncoderError("acquisition schema changed")
        _require_sha256(self.evidence_binding_sha256, "evidence_binding_sha256")
        if type(self.cell_partition_telemetry) is not SameClassRealizationCellPartitionTelemetryV1:
            raise SameClassRealizationEncoderError("acquisition cell-partition telemetry changed exact type")
        if type(self.total_debt_cell_count) is not int or self.total_debt_cell_count < 1:
            raise SameClassRealizationEncoderError("acquisition requires nonempty exact realization debt")
        if (
            type(self.total_debt_count_by_class) is not tuple
            or len(self.total_debt_count_by_class) != _CLASS_COUNT
            or sum(self.total_debt_count_by_class) != self.total_debt_cell_count
        ):
            raise SameClassRealizationEncoderError("acquisition per-class debt does not close")
        if (
            self.cell_partition_telemetry.realization_debt_cell_count != self.total_debt_cell_count
            or self.cell_partition_telemetry.realization_debt_count_by_target_class != self.total_debt_count_by_class
        ):
            raise SameClassRealizationEncoderError("acquisition debt differs from canonical four-way partition row 2")
        if (
            type(self.prefix_cell_counts) is not tuple
            or not self.prefix_cell_counts
            or self.prefix_cell_counts != tuple(sorted(set(self.prefix_cell_counts)))
            or self.prefix_cell_counts[-1] != self.total_debt_cell_count
        ):
            raise SameClassRealizationEncoderError("acquisition prefixes must be canonical and complete")
        if type(self.proposals) is not tuple or not self.proposals:
            raise SameClassRealizationEncoderError("acquisition must emit concrete proposals")
        if any(type(proposal) is not SameClassRealizationProgramProposalV1 for proposal in self.proposals):
            raise SameClassRealizationEncoderError("acquisition proposal changed exact type")
        proposal_ids = tuple(proposal.proposal_id for proposal in self.proposals)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise SameClassRealizationEncoderError("acquisition proposal IDs must be unique")
        if self.proposals != tuple(sorted(self.proposals, key=_proposal_priority_key)):
            raise SameClassRealizationEncoderError(
                "acquisition proposals must follow advisory fidelity-first canonical order"
            )
        families = {proposal.receipt.family for proposal in self.proposals}
        if families != set(SameClassRealizationProposalFamilyV1):
            raise SameClassRealizationEncoderError("acquisition omitted a required proposal family")
        orders = {proposal.receipt.prefix_order for proposal in self.proposals}
        if orders != set(SameClassRealizationPrefixOrderV1):
            raise SameClassRealizationEncoderError("acquisition omitted a required prefix order")
        truth = {
            "scorer_invoked": False,
            "whole_object_allocator_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise SameClassRealizationEncoderError("acquisition truth labels became permissive")

    @property
    def unique_program_proposals(self) -> tuple[SameClassRealizationProgramProposalV1, ...]:
        """Return one fidelity-first representative per exact G8 program digest.

        The complete proposal tuple preserves every acquisition interpretation.
        Downstream G7/G10 measurement should use this view so equivalent full
        prefixes or palette interpretations do not spend duplicate scorer rows.
        """

        seen: set[str] = set()
        unique: list[SameClassRealizationProgramProposalV1] = []
        for proposal in self.proposals:
            if proposal.receipt.program_sha256 in seen:
                continue
            seen.add(proposal.receipt.program_sha256)
            unique.append(proposal)
        return tuple(unique)

    @property
    def duplicate_program_count(self) -> int:
        """Number of interpretation receipts sharing already-emitted bytes."""

        return len(self.proposals) - len(self.unique_program_proposals)


def _program_sha256(program: SameClassRealizationRepairProgramV1) -> str:
    if type(program) is not SameClassRealizationRepairProgramV1:
        raise SameClassRealizationEncoderError("program digest requires exact G8 program type")
    return _sha256(
        _canonical_json(
            {
                "schema": "tac.same_class_realization_repair_program_identity.v1",
                "runs": [
                    {
                        "source_pair_id": run.source_pair_id,
                        "scorer_row": run.scorer_row,
                        "scorer_col_start": run.scorer_col_start,
                        "scorer_col_stop": run.scorer_col_stop,
                        "semantic_class": run.semantic_class,
                        "semantic_role": run.semantic_role.value,
                        "rgb_u8": list(run.rgb_u8),
                    }
                    for run in program.runs
                ],
            }
        )
    )


def _palette_sha256(palettes: dict[int, tuple[tuple[int, int, int], ...]]) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema": "tac.same_class_realization_target_palette.v1",
                "classes": [
                    {
                        "semantic_class": class_id,
                        "rgb_u8": [list(rgb) for rgb in palettes.get(class_id, ())],
                    }
                    for class_id in range(_CLASS_COUNT)
                ],
            }
        )
    )


def _squared_distances(colours: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    delta = colours.astype(np.int64, copy=False) - prototype.astype(np.int64, copy=False)
    return np.sum(delta * delta, axis=1, dtype=np.int64)


def _exact_weighted_medoid(colours: np.ndarray, weights: np.ndarray) -> tuple[int, int, int]:
    """Observed RGB minimizing exact weighted squared Euclidean distance."""

    if colours.dtype != np.uint8 or colours.ndim != 2 or colours.shape[1] != 3 or len(colours) < 1:
        raise SameClassRealizationEncoderError("medoid colours must be nonempty uint8 RGB rows")
    if weights.dtype != np.int64 or weights.shape != (len(colours),) or np.any(weights <= 0):
        raise SameClassRealizationEncoderError("medoid weights must be positive exact int64 rows")
    colour_i64 = colours.astype(np.int64)
    total_weight = int(np.sum(weights, dtype=np.int64))
    weighted_sum = np.sum(colour_i64 * weights[:, None], axis=0, dtype=np.int64)
    scores = total_weight * np.sum(colour_i64 * colour_i64, axis=1, dtype=np.int64)
    scores -= 2 * np.sum(colour_i64 * weighted_sum[None, :], axis=1, dtype=np.int64)
    index = int(np.argmin(scores))
    return _rgb_tuple(colours[index])


def _palette_objective(
    colours: np.ndarray,
    weights: np.ndarray,
    palette: tuple[tuple[int, int, int], ...],
) -> int:
    best: np.ndarray | None = None
    for rgb in palette:
        distance = _squared_distances(colours, np.asarray(rgb, dtype=np.uint8))
        best = distance if best is None else np.minimum(best, distance)
    if best is None:  # pragma: no cover - every palette is nonempty
        raise SameClassRealizationEncoderError("empty palette has no objective")
    return int(np.sum(best * weights, dtype=np.int64))


def _assign_to_palette(
    colours: np.ndarray,
    palette: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    if not palette:
        raise SameClassRealizationEncoderError("cannot assign colours to empty palette")
    best_distance: np.ndarray | None = None
    assignment = np.zeros(len(colours), dtype=np.int64)
    for index, rgb in enumerate(palette):
        distance = _squared_distances(colours, np.asarray(rgb, dtype=np.uint8))
        if best_distance is None:
            best_distance = distance
            continue
        better = distance < best_distance
        assignment[better] = index
        best_distance = np.minimum(best_distance, distance)
    return assignment


def _bounded_target_medoids(rgb_rows: np.ndarray, palette_bound: int) -> tuple[tuple[int, int, int], ...]:
    """Deterministic observed-colour medoids with finite monotone refinement."""

    _require_exact_int(palette_bound, "palette_bound", minimum=1, maximum=MAX_RUNS)
    rows = np.ascontiguousarray(rgb_rows, dtype=np.uint8).reshape(-1, 3)
    if len(rows) < 1:
        raise SameClassRealizationEncoderError("cannot fit a target palette without RGB rows")
    colours, raw_weights = np.unique(rows, axis=0, return_counts=True)
    colours = np.ascontiguousarray(colours, dtype=np.uint8)
    weights = np.ascontiguousarray(raw_weights, dtype=np.int64)
    bound = min(palette_bound, len(colours))

    palette: list[tuple[int, int, int]] = [_exact_weighted_medoid(colours, weights)]
    while len(palette) < bound:
        current = tuple(sorted(palette))
        assignment = _assign_to_palette(colours, current)
        nearest = np.empty(len(colours), dtype=np.int64)
        for index, rgb in enumerate(current):
            mask = assignment == index
            nearest[mask] = _squared_distances(
                colours[mask],
                np.asarray(rgb, dtype=np.uint8),
            )
        contribution = nearest * weights
        for index in np.argsort(-contribution, kind="stable"):
            candidate = _rgb_tuple(colours[int(index)])
            if candidate not in palette:
                palette.append(candidate)
                break
        else:  # pragma: no cover - bound is no greater than unique colours
            raise SameClassRealizationEncoderError("palette seeding failed to add a distinct medoid")

    current = tuple(sorted(palette))
    current_objective = _palette_objective(colours, weights, current)
    while True:
        assignment = _assign_to_palette(colours, current)
        updated = tuple(
            sorted(
                _exact_weighted_medoid(colours[assignment == index], weights[assignment == index])
                for index in range(len(current))
            )
        )
        updated_objective = _palette_objective(colours, weights, updated)
        if updated == current:
            return current
        if updated_objective < current_objective or (updated_objective == current_objective and updated < current):
            current = updated
            current_objective = updated_objective
            continue
        return current


def _debt_coordinates(evidence: EncoderOnlySameClassRealizationEvidenceV1) -> np.ndarray:
    coordinates = np.argwhere(evidence.debt_mask)
    return np.ascontiguousarray(coordinates, dtype=np.int64)


def _ordered_coordinate_indices(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    coordinates: np.ndarray,
    order: SameClassRealizationPrefixOrderV1,
) -> np.ndarray:
    if order is SameClassRealizationPrefixOrderV1.CANONICAL_ADDRESS_V1:
        return np.arange(len(coordinates), dtype=np.int64)
    candidate = evidence.candidate_scorer_rgb[tuple(coordinates.T)]
    target = evidence.target_scorer_rgb[tuple(coordinates.T)]
    delta = candidate.astype(np.int64) - target.astype(np.int64)
    sse = np.sum(delta * delta, axis=1, dtype=np.int64)
    return np.lexsort(
        (
            coordinates[:, 2],
            coordinates[:, 1],
            coordinates[:, 0],
            -sse,
        )
    ).astype(np.int64, copy=False)


def _target_palettes(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    coordinates: np.ndarray,
    *,
    palette_bound: int,
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    labels = evidence.target_labels[tuple(coordinates.T)]
    target_rgb = evidence.target_scorer_rgb[tuple(coordinates.T)]
    result: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for class_id in range(_CLASS_COUNT):
        class_rgb = target_rgb[labels == class_id]
        if len(class_rgb):
            result[class_id] = _bounded_target_medoids(class_rgb, palette_bound)
    return result


def _rgb_for_coordinates(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    selected: np.ndarray,
    *,
    family: SameClassRealizationProposalFamilyV1,
    palettes: dict[int, tuple[tuple[int, int, int], ...]],
) -> np.ndarray:
    target = np.ascontiguousarray(evidence.target_scorer_rgb[tuple(selected.T)], dtype=np.uint8)
    if family is SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1:
        return target
    labels = evidence.target_labels[tuple(selected.T)]
    output = np.empty_like(target)
    for class_id in range(_CLASS_COUNT):
        mask = labels == class_id
        if not np.any(mask):
            continue
        palette = palettes.get(class_id)
        if not palette:
            raise SameClassRealizationEncoderError("target-derived palette omitted an active class")
        assignment = _assign_to_palette(target[mask], palette)
        palette_array = np.asarray(palette, dtype=np.uint8)
        output[mask] = palette_array[assignment]
    return output


def _build_program(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    selected: np.ndarray,
    selected_rgb: np.ndarray,
) -> SameClassRealizationRepairProgramV1:
    records: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for coordinate, rgb in zip(selected, selected_rgb, strict=True):
        local_pair, row, col = (int(value) for value in coordinate)
        semantic_class = int(evidence.current_semantic_labels[local_pair, row, col])
        records.append(
            (
                evidence.source_pair_ids[local_pair],
                row,
                col,
                semantic_class,
                _rgb_tuple(rgb),
            )
        )
    records.sort()
    runs: list[SameClassRealizationRepairRunV1] = []
    for source_pair_id, row, col, semantic_class, rgb in records:
        role = _CLASS_TO_ROLE[semantic_class]
        if runs:
            previous = runs[-1]
            if (
                previous.source_pair_id == source_pair_id
                and previous.scorer_row == row
                and previous.scorer_col_stop == col
                and previous.semantic_class == semantic_class
                and previous.semantic_role is role
                and previous.rgb_u8 == rgb
            ):
                runs[-1] = SameClassRealizationRepairRunV1(
                    source_pair_id=source_pair_id,
                    scorer_row=row,
                    scorer_col_start=previous.scorer_col_start,
                    scorer_col_stop=col + 1,
                    semantic_class=semantic_class,
                    semantic_role=role,
                    rgb_u8=rgb,
                )
                continue
        runs.append(
            SameClassRealizationRepairRunV1(
                source_pair_id=source_pair_id,
                scorer_row=row,
                scorer_col_start=col,
                scorer_col_stop=col + 1,
                semantic_class=semantic_class,
                semantic_role=role,
                rgb_u8=rgb,
            )
        )
    try:
        return SameClassRealizationRepairProgramV1(runs=tuple(runs))
    except ValueError as exc:
        raise SameClassRealizationEncoderError(
            "selected realization prefix cannot be represented by the frozen G8 run ABI"
        ) from exc


def validate_same_class_realization_program_against_evidence(
    program: SameClassRealizationRepairProgramV1,
    *,
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
) -> None:
    """Fail closed unless every expanded program cell lies in the exact debt set."""

    if type(program) is not SameClassRealizationRepairProgramV1:
        raise SameClassRealizationEncoderError("program validation requires exact G8 program type")
    if type(evidence) is not EncoderOnlySameClassRealizationEvidenceV1:
        raise SameClassRealizationEncoderError("program validation requires exact encoder evidence")
    pair_to_local = {pair_id: index for index, pair_id in enumerate(evidence.source_pair_ids)}
    debt = evidence.debt_mask
    seen: set[tuple[int, int, int]] = set()
    for run in program.runs:
        local_pair = pair_to_local.get(run.source_pair_id)
        if local_pair is None:
            raise SameClassRealizationEncoderError("program addresses a foreign source pair")
        for col in range(run.scorer_col_start, run.scorer_col_stop):
            address = (local_pair, run.scorer_row, col)
            if address in seen:
                raise SameClassRealizationEncoderError("program repeats a scorer-cell address")
            seen.add(address)
            current_class = int(evidence.current_semantic_labels[address])
            target_class = int(evidence.target_labels[address])
            realized_class = int(evidence.realized_labels[address])
            if (
                not bool(debt[address])
                or run.semantic_class != current_class
                or current_class != target_class
                or realized_class == target_class
            ):
                raise SameClassRealizationEncoderError(
                    "program admitted a cell outside exact same-class realization debt"
                )
    if len(seen) != program.expanded_cell_count:
        raise SameClassRealizationEncoderError("program expanded-cell accounting changed")


def _count_selected_by_class(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    selected: np.ndarray,
) -> tuple[int, int, int, int, int]:
    labels = evidence.target_labels[tuple(selected.T)]
    return _five_int_tuple(tuple(int(np.count_nonzero(labels == class_id)) for class_id in range(_CLASS_COUNT)))


def _proposal_id(
    family: SameClassRealizationProposalFamilyV1,
    *,
    palette_bound: int | None,
    order: SameClassRealizationPrefixOrderV1,
    prefix_count: int,
) -> str:
    palette_token = "oracle" if palette_bound is None else f"k{palette_bound}"
    return f"{family.value.lower()}__{palette_token}__{order.value.lower()}__n{prefix_count}"


def _proposal_priority_key(
    proposal: SameClassRealizationProgramProposalV1,
) -> tuple[int, int, int, int, str]:
    """Advisory n2-derived measurement order; never an admission score."""

    receipt = proposal.receipt
    order_rank = 0 if receipt.prefix_order is SameClassRealizationPrefixOrderV1.TARGET_RGB_SSE_DESCENDING_V1 else 1
    palette_rank = receipt.palette_bound_per_class or (MAX_RUNS + 1)
    return (
        _FAMILY_PRIORITY[receipt.family],
        -receipt.covered_debt_cell_count,
        -palette_rank,
        order_rank,
        proposal.proposal_id,
    )


def _make_proposal(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    *,
    plan: SameClassRealizationAcquisitionPlanV1,
    coordinates: np.ndarray,
    ordered_indices: np.ndarray,
    family: SameClassRealizationProposalFamilyV1,
    palette_bound: int | None,
    palettes: dict[int, tuple[tuple[int, int, int], ...]],
    order: SameClassRealizationPrefixOrderV1,
    prefix_count: int,
    cell_partition_telemetry: SameClassRealizationCellPartitionTelemetryV1,
) -> SameClassRealizationProgramProposalV1:
    selected = np.ascontiguousarray(coordinates[ordered_indices[:prefix_count]], dtype=np.int64)
    selected_rgb = _rgb_for_coordinates(
        evidence,
        selected,
        family=family,
        palettes=palettes,
    )
    program = _build_program(evidence, selected, selected_rgb)
    validate_same_class_realization_program_against_evidence(program, evidence=evidence)

    total_by_class = evidence.debt_count_by_class
    covered_by_class = _count_selected_by_class(evidence, selected)
    omitted_by_class = _five_int_tuple(
        tuple(total - covered for total, covered in zip(total_by_class, covered_by_class, strict=True))
    )
    candidate_rgb = evidence.candidate_scorer_rgb[tuple(selected.T)].astype(np.int64)
    target_rgb = evidence.target_scorer_rgb[tuple(selected.T)].astype(np.int64)
    proposed_rgb = selected_rgb.astype(np.int64)
    before_delta = candidate_rgb - target_rgb
    after_delta = proposed_rgb - target_rgb
    proposal_id = _proposal_id(
        family,
        palette_bound=palette_bound,
        order=order,
        prefix_count=prefix_count,
    )
    receipt = SameClassRealizationProposalReceiptV1(
        schema=RECEIPT_SCHEMA,
        proposal_id=proposal_id,
        evidence_binding_sha256=evidence.binding_sha256,
        base_interpretation=evidence.base_interpretation,
        base_semantic_binding_sha256=evidence.base_semantic_binding_sha256,
        base_p_section_sha256=evidence.base_p_section_sha256,
        base_p_section_bytes=evidence.base_p_section_bytes,
        base_g_section_sha256=evidence.base_g_section_sha256,
        base_g_section_bytes=evidence.base_g_section_bytes,
        base_camera_y1_sha256=evidence.base_camera_y1_sha256,
        g8_compile_seam=evidence.g8_compile_seam,
        base_interpretation_has_frozen_g8_compile_path=(evidence.base_interpretation_has_frozen_g8_compile_path),
        current_semantic_labels_sha256=evidence.current_semantic_labels_sha256,
        target_labels_sha256=evidence.target_labels_sha256,
        realized_labels_sha256=evidence.realized_labels_sha256,
        candidate_scorer_rgb_sha256=evidence.candidate_scorer_rgb_sha256,
        target_scorer_rgb_sha256=evidence.target_scorer_rgb_sha256,
        priority_ordering_receipt_sha256=evidence.priority_ordering_receipt_sha256,
        priority_ordering_axis=evidence.priority_ordering_axis,
        priority_ordering_scope=evidence.priority_ordering_scope,
        cell_partition_telemetry=cell_partition_telemetry,
        family=family,
        prefix_order=order,
        palette_bound_per_class=palette_bound,
        actual_palette_count_by_class=_five_int_tuple(
            tuple(len(palettes.get(class_id, ())) for class_id in range(_CLASS_COUNT))
        ),
        geometric_prefix_ratios=plan.geometric_prefix_ratios,
        requested_prefix_cell_count=prefix_count,
        total_debt_cell_count=len(coordinates),
        covered_debt_cell_count=prefix_count,
        omitted_debt_cell_count=len(coordinates) - prefix_count,
        total_debt_count_by_class=total_by_class,
        covered_debt_count_by_class=covered_by_class,
        omitted_debt_count_by_class=omitted_by_class,
        run_count=program.run_count,
        expanded_cell_count=program.expanded_cell_count,
        program_sha256=_program_sha256(program),
        palette_sha256=_palette_sha256(palettes),
        estimated_standalone_g8_raw_bytes=PACKET_HEADER.size + program.run_count * RUN_ROW.size,
        selected_target_rgb_sse_before=int(np.sum(before_delta * before_delta, dtype=np.int64)),
        selected_target_rgb_sse_after=int(np.sum(after_delta * after_delta, dtype=np.int64)),
        target_pixel_oracle_exact_on_selected_cells=(
            family is SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1
        ),
    )
    return SameClassRealizationProgramProposalV1(
        proposal_id=proposal_id,
        program=program,
        receipt=receipt,
    )


def acquire_same_class_realization_repair_programs(
    evidence: EncoderOnlySameClassRealizationEvidenceV1,
    *,
    plan: SameClassRealizationAcquisitionPlanV1,
) -> SameClassRealizationAcquisitionV1:
    """Emit all target-derived families across geometric prefixes and orders."""

    if type(evidence) is not EncoderOnlySameClassRealizationEvidenceV1:
        raise SameClassRealizationEncoderError("acquisition requires exact encoder evidence type")
    if type(plan) is not SameClassRealizationAcquisitionPlanV1:
        raise SameClassRealizationEncoderError("acquisition requires exact plan type")
    coordinates = _debt_coordinates(evidence)
    if len(coordinates) == 0:
        raise SameClassRealizationEncoderError("same-class realization acquisition requires nonempty exact debt")
    prefixes = plan.prefix_cell_counts(len(coordinates))
    cell_partition_telemetry = evidence.cell_partition_telemetry
    class_shared_palettes = _target_palettes(
        evidence,
        coordinates,
        palette_bound=1,
    )
    bounded_palettes = {
        palette_bound: _target_palettes(
            evidence,
            coordinates,
            palette_bound=palette_bound,
        )
        for palette_bound in plan.palette_sizes_per_class
    }
    oracle_palettes: dict[int, tuple[tuple[int, int, int], ...]] = {}

    family_specs: list[
        tuple[
            SameClassRealizationProposalFamilyV1,
            int | None,
            dict[int, tuple[tuple[int, int, int], ...]],
        ]
    ] = [
        (
            SameClassRealizationProposalFamilyV1.CLASS_SHARED_TARGET_MEDOID_V1,
            1,
            class_shared_palettes,
        ),
        *[
            (
                SameClassRealizationProposalFamilyV1.CLASS_BOUNDED_TARGET_MEDOIDS_V1,
                palette_bound,
                bounded_palettes[palette_bound],
            )
            for palette_bound in plan.palette_sizes_per_class
        ],
        (
            SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1,
            None,
            oracle_palettes,
        ),
    ]

    proposals: list[SameClassRealizationProgramProposalV1] = []
    for family, palette_bound, palettes in family_specs:
        for order in _PREFIX_ORDERS:
            ordered_indices = _ordered_coordinate_indices(evidence, coordinates, order)
            for prefix_count in prefixes:
                proposals.append(
                    _make_proposal(
                        evidence,
                        plan=plan,
                        coordinates=coordinates,
                        ordered_indices=ordered_indices,
                        family=family,
                        palette_bound=palette_bound,
                        palettes=palettes,
                        order=order,
                        prefix_count=prefix_count,
                        cell_partition_telemetry=cell_partition_telemetry,
                    )
                )
    proposals.sort(key=_proposal_priority_key)
    return SameClassRealizationAcquisitionV1(
        schema=ACQUISITION_SCHEMA,
        evidence_binding_sha256=evidence.binding_sha256,
        cell_partition_telemetry=cell_partition_telemetry,
        total_debt_cell_count=len(coordinates),
        total_debt_count_by_class=evidence.debt_count_by_class,
        prefix_cell_counts=prefixes,
        proposals=tuple(proposals),
    )


__all__ = [
    "ACQUISITION_SCHEMA",
    "CELL_PARTITION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "N2_STAGE_ABLATION_PRIORITY_AXIS",
    "N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256",
    "N2_STAGE_ABLATION_PRIORITY_SCOPE",
    "RECEIPT_SCHEMA",
    "EncoderOnlySameClassRealizationEvidenceV1",
    "SameClassRealizationAcquisitionPlanV1",
    "SameClassRealizationAcquisitionV1",
    "SameClassRealizationBaseInterpretationV1",
    "SameClassRealizationCellPartitionTelemetryV1",
    "SameClassRealizationEncoderError",
    "SameClassRealizationG8CompileSeamV1",
    "SameClassRealizationPrefixOrderV1",
    "SameClassRealizationProgramProposalV1",
    "SameClassRealizationProposalFamilyV1",
    "SameClassRealizationProposalReceiptV1",
    "acquire_same_class_realization_repair_programs",
    "parse_same_class_realization_proposal_receipt",
    "validate_same_class_realization_program_against_evidence",
]
