# SPDX-License-Identifier: MIT
"""Strict G72 precompiler for fresh n600 scorer-native analytic factors.

This module closes the part of the G72 path that can be closed honestly today:

* recursively reopen the fresh V15 semantic compile, G46 batch-16 labels, and
  G51 five-stage direct-task scorer-plane custody;
* derive the real V9 ``BoundaryShearletAtomV1`` proposal family from exact
  scorer labels and Fisher margins, preserving each atom's semantic role;
* preserve five immutable 120-pair proposal checkpoints on the SSD tier; and
* admit a proposal only by the exact whole-object contest objective.

It deliberately does not emit a G49 packet while the current ``TSPPV1``
analytic wire drops ``BoundaryShearletAtomV1.role`` and applies its mask after
RGB painting.  V15 applies role-aware atoms before ordered role painting at
camera resolution.  Treating those two semantics as interchangeable would
silently destroy occluded signal.  ``audit_g72_readiness`` proves that blocker
by behavior, rather than trusting a caller-provided flag.

The G51 ``gt_poses`` stream is reopened and retained as source-cache advisory
custody.  It is not promoted to fresh batch-16 Pose authority.  No score,
candidate, learned-quotient, or frontier claim is made here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    HEAD_PAIR_NORMS,
)
from tac.optimization.direct_description_carrier_compose import (
    CLASS_ORDER,
    BoundaryShearletAtomV1,
)
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
    FreshScorerPlaneOperandLoaderV1,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    canonical_json_bytes,
    load_compile_ready_materialization_receipt,
    require_ssd_output_root,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    V15_SEMANTIC_RECEIVER_ID,
    SelectedPreimageFrameSelectorV1,
    build_analytic_shearlet_residual_factor,
    verify_v15_semantic_compile_lineage,
)

READINESS_SCHEMA: Final = "tac.g72_fresh_n600_g49_analytic_factor_readiness.v1"
PROPOSAL_SCHEMA: Final = "tac.g72_v9_boundary_shearlet_proposal.v1"
STAGE_CHECKPOINT_SCHEMA: Final = "tac.g72_analytic_factor_stage_checkpoint.v1"
JOINT_ADMISSION_SCHEMA: Final = "tac.g72_exact_whole_object_joint_admission.v1"

PAIR_COUNT: Final = 600
PAIRS_PER_STAGE: Final = 120
STAGE_COUNT: Final = 5
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
POSE_COORDINATES_PER_PAIR: Final = 6
SOURCE_VIDEO_BYTES: Final = 37_545_489
EXPECTED_SCORER_BATCH_SIZE: Final = 16

FRESH_BATCH16_MARGIN_CUSTODY_OWED: Final = "G72_FRESH_BATCH16_TARGET_MARGIN_CUSTODY_OWED"
FRESH_V15_BASE_SCORER_CACHE_OWED: Final = "G72_FRESH_V15_CAMERA_R_BATCH16_BASE_SCORER_STAGE_CACHE_OWED"
G49_ROLE_WIRE_OWED: Final = "G72_G49_ROLE_PRESERVING_ANALYTIC_WIRE_ABI_OWED"
V15_ROLE_AWARE_DECODER_OWED: Final = "G72_V15_ROLE_AWARE_PREPAINT_ANALYTIC_DECODER_PROOF_OWED"
POSE_AUTHORITY_OR_FINAL_REPLAY_OWED: Final = "G72_FRESH_POSE_TARGET_AUTHORITY_OR_EXACT_UPSTREAM_FINAL_REPLAY_OWED"
EXACT_JOINT_STAGE_ADMISSION_OWED: Final = "G72_FIVE_STAGE_EXACT_WHOLE_OBJECT_JOINT_ADMISSION_OWED"


class G72AnalyticFactorCompilerError(ValueError):
    """Fresh custody, proposal, checkpoint, or joint admission failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise G72AnalyticFactorCompilerError(f"required custody file is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes([int(item) for item in array.shape]))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise G72AnalyticFactorCompilerError(f"{label} is not a lowercase SHA-256")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise G72AnalyticFactorCompilerError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _seal(body: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise G72AnalyticFactorCompilerError(f"{field} is already present")
    return {**body, field: _sha256(canonical_json_bytes(body))}


def _verify_seal(value: dict[str, Any], *, field: str) -> None:
    expected = _require_sha256(value.get(field), label=field)
    body = {key: item for key, item in value.items() if key != field}
    if _sha256(canonical_json_bytes(body)) != expected:
        raise G72AnalyticFactorCompilerError(f"{field} differs from the canonical body")


@dataclass(frozen=True, slots=True)
class G72StagePlanV1:
    """One mandatory immutable 120-pair production stage."""

    stage_index: int
    pair_start: int
    pair_stop_exclusive: int

    def __post_init__(self) -> None:
        index = _require_exact_int(
            self.stage_index,
            label="stage_index",
            minimum=0,
            maximum=STAGE_COUNT - 1,
        )
        if self.pair_start != index * PAIRS_PER_STAGE or self.pair_stop_exclusive != (index + 1) * PAIRS_PER_STAGE:
            raise G72AnalyticFactorCompilerError("G72 stage is not one canonical 120-pair n600 partition")


def g72_stage_plan() -> tuple[G72StagePlanV1, ...]:
    """Return the only production stage partition admitted by G72."""

    return tuple(
        G72StagePlanV1(
            stage_index=index,
            pair_start=index * PAIRS_PER_STAGE,
            pair_stop_exclusive=(index + 1) * PAIRS_PER_STAGE,
        )
        for index in range(STAGE_COUNT)
    )


@dataclass(frozen=True, slots=True)
class G72BoundaryShearletProposalV1:
    """One role-preserving V9 scorer-native analytic proposal."""

    candidate_id: str
    fisher_priority: float
    atom: BoundaryShearletAtomV1

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id or not self.candidate_id.isascii():
            raise G72AnalyticFactorCompilerError("candidate_id must be nonempty ASCII")
        priority = float(self.fisher_priority)
        if not math.isfinite(priority) or priority < 0.0:
            raise G72AnalyticFactorCompilerError("fisher_priority must be finite and nonnegative")
        if type(self.atom) is not BoundaryShearletAtomV1:
            raise G72AnalyticFactorCompilerError("proposal must contain an exact BoundaryShearletAtomV1")

    @property
    def fingerprint(self) -> str:
        return _sha256(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        atom = self.atom
        return {
            "schema": PROPOSAL_SCHEMA,
            "candidate_id": self.candidate_id,
            "fisher_priority": format(float(self.fisher_priority), ".17g"),
            "atom": {
                "pair_index": atom.pair_index,
                "role": atom.role,
                "center_y": atom.center_y,
                "center_x": atom.center_x,
                "scale_y": atom.scale_y,
                "scale_x": atom.scale_x,
                "shear_q4": atom.shear_q4,
                "amplitude_q4": atom.amplitude_q4,
            },
        }


def _proposal_from_dict(value: object) -> G72BoundaryShearletProposalV1:
    if type(value) is not dict or set(value) != {
        "atom",
        "candidate_id",
        "fisher_priority",
        "schema",
    }:
        raise G72AnalyticFactorCompilerError("stage proposal has a noncanonical key set")
    if value["schema"] != PROPOSAL_SCHEMA or type(value["atom"]) is not dict:
        raise G72AnalyticFactorCompilerError("stage proposal schema or atom differs")
    atom = value["atom"]
    if set(atom) != {
        "amplitude_q4",
        "center_x",
        "center_y",
        "pair_index",
        "role",
        "scale_x",
        "scale_y",
        "shear_q4",
    }:
        raise G72AnalyticFactorCompilerError("stage proposal atom has a noncanonical key set")
    try:
        return G72BoundaryShearletProposalV1(
            candidate_id=value["candidate_id"],
            fisher_priority=float(value["fisher_priority"]),
            atom=BoundaryShearletAtomV1(**atom),
        )
    except (TypeError, ValueError) as exc:
        raise G72AnalyticFactorCompilerError("stage proposal failed strict reconstruction") from exc


def _require_stage_field(
    value: np.ndarray,
    *,
    label: str,
    dtype_kind: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (PAIRS_PER_STAGE, SCORER_HEIGHT, SCORER_WIDTH):
        raise G72AnalyticFactorCompilerError(f"{label} must be one exact 120-pair scorer field")
    if dtype_kind == "integer":
        if not np.issubdtype(array.dtype, np.integer):
            raise G72AnalyticFactorCompilerError(f"{label} must be integer")
        if bool(np.any(array < 0)) or bool(np.any(array >= len(CLASS_ORDER))):
            raise G72AnalyticFactorCompilerError(f"{label} contains a class outside the frozen five-class head")
    elif not np.issubdtype(array.dtype, np.floating):
        raise G72AnalyticFactorCompilerError(f"{label} must be floating point")
    if not bool(np.isfinite(array).all()):
        raise G72AnalyticFactorCompilerError(f"{label} contains nonfinite values")
    return array


def _pair_norm(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    result = np.ones(target.shape, dtype=np.float64)
    for left in range(len(CLASS_ORDER)):
        for right in range(left + 1, len(CLASS_ORDER)):
            key = f"{CLASS_ORDER[left]}-{CLASS_ORDER[right]}"
            mask = ((target == left) & (predicted == right)) | ((target == right) & (predicted == left))
            result[mask] = HEAD_PAIR_NORMS[key]
    return result


def _fisher_priority_map(
    target: np.ndarray,
    predicted: np.ndarray,
    margin: np.ndarray,
) -> np.ndarray:
    """The scorer-native V9 flip-distance/Fisher proposal law."""

    absolute_margin = np.abs(np.asarray(margin, dtype=np.float64))
    distance = absolute_margin / _pair_norm(target, predicted)
    curvature = 0.5 / np.square(np.cosh(np.minimum(absolute_margin, 20.0) / 2.0))
    band = np.select(
        [
            absolute_margin < 0.1,
            absolute_margin < 0.5,
            absolute_margin < 1.0,
        ],
        [4.0, 2.0, 1.0],
        default=0.25,
    )
    value = curvature * band / np.maximum(distance, 1.0e-3)
    value[target == predicted] = 0.0
    return value


def _component_boxes(
    mask: np.ndarray,
    *,
    minimum_sites: int,
    maximum: int,
) -> list[tuple[int, int, int, int, int]]:
    if int(np.count_nonzero(mask)) < minimum_sites:
        return []
    labels, _ = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    rows: list[tuple[int, int, int, int, int]] = []
    for component, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        sites = int(np.count_nonzero(labels[slices] == component))
        if sites < minimum_sites:
            continue
        y_slice, x_slice = slices
        rows.append(
            (
                sites,
                int(y_slice.start),
                int(x_slice.start),
                int(y_slice.stop),
                int(x_slice.stop),
            )
        )
    rows.sort(reverse=True)
    return rows[:maximum]


def derive_v9_boundary_shearlet_stage_proposals(
    *,
    stage: G72StagePlanV1,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    described_cells: np.ndarray,
    minimum_component_sites: int,
    maximum_components_per_pair_role: int,
) -> tuple[G72BoundaryShearletProposalV1, ...]:
    """Port the real V9 Road/Undrivable shearlet proposal derivation.

    This is proposal generation only.  Fisher priority orders work; it never
    admits a factor.  Admission requires exact scorer replay and the joint
    whole-object objective via :func:`admit_exact_whole_object_change`.
    """

    if type(stage) is not G72StagePlanV1:
        raise G72AnalyticFactorCompilerError("proposal derivation requires an exact G72 stage")
    target = _require_stage_field(
        target_cells,
        label="target_cells",
        dtype_kind="integer",
    )
    described = _require_stage_field(
        described_cells,
        label="described_cells",
        dtype_kind="integer",
    )
    margins = _require_stage_field(
        target_margins,
        label="target_margins",
        dtype_kind="floating",
    )
    minimum = _require_exact_int(
        minimum_component_sites,
        label="minimum_component_sites",
        minimum=1,
        maximum=SCORER_HEIGHT * SCORER_WIDTH,
    )
    maximum = _require_exact_int(
        maximum_components_per_pair_role,
        label="maximum_components_per_pair_role",
        minimum=1,
        maximum=4096,
    )

    proposals: list[G72BoundaryShearletProposalV1] = []
    for local_pair_id in range(PAIRS_PER_STAGE):
        source_pair_id = stage.pair_start + local_pair_id
        target_pair = np.asarray(target[local_pair_id])
        described_pair = np.asarray(described[local_pair_id])
        priority = _fisher_priority_map(
            target_pair,
            described_pair,
            margins[local_pair_id],
        )
        for role, class_id in (("Road", 0), ("UndrivableBoundary", 2)):
            mismatch = (target_pair == class_id) != (described_pair == class_id)
            boxes = _component_boxes(
                mismatch,
                minimum_sites=minimum,
                maximum=maximum,
            )
            for ordinal, (site_count, y0, x0, y1, x1) in enumerate(boxes):
                local_mismatch = mismatch[y0:y1, x0:x1]
                component_sites = np.argwhere(local_mismatch) + np.asarray((y0, x0))
                if component_sites.size == 0:
                    continue
                center_y, center_x = np.rint(component_sites.mean(axis=0)).astype(np.int64)
                centered = component_sites - component_sites.mean(axis=0)
                var_x = float(np.square(centered[:, 1]).mean()) + 1.0e-6
                shear = float((centered[:, 0] * centered[:, 1]).mean() / var_x)
                shear_q4 = int(np.clip(np.rint(shear * 16.0), -64, 64))
                scale_y = int(np.clip(max(2, (y1 - y0) // 2), 2, 48))
                scale_x = int(
                    np.clip(
                        max(2 * scale_y, (x1 - x0) * 2, 8),
                        4,
                        256,
                    )
                )
                missing = int(np.count_nonzero((target_pair[y0:y1, x0:x1] == class_id) & local_mismatch))
                excess = site_count - missing
                inferred_sign = 1 if missing >= excess else -1
                amplitude = float(np.clip(max(1.0, (y1 - y0) / 2.0), 1.0, 24.0))
                family_name = "Road" if role == "Road" else "Undrivable"
                local_priority = float(priority[y0:y1, x0:x1][local_mismatch].sum())
                for direction_rank, direction in enumerate((inferred_sign, -inferred_sign)):
                    for scale in (0.5, 1.0):
                        amplitude_q4 = int(
                            np.clip(
                                np.rint(direction * amplitude * scale * 16.0),
                                -512,
                                512,
                            )
                        )
                        if amplitude_q4 == 0:
                            continue
                        atom = BoundaryShearletAtomV1(
                            pair_index=source_pair_id,
                            role=role,
                            center_y=int(center_y),
                            center_x=int(center_x),
                            scale_y=scale_y,
                            scale_x=scale_x,
                            shear_q4=shear_q4,
                            amplitude_q4=amplitude_q4,
                        )
                        proposals.append(
                            G72BoundaryShearletProposalV1(
                                candidate_id=(
                                    f"{family_name.lower()}_{source_pair_id}_{ordinal}_sh_d{direction_rank}_a{scale:g}"
                                ),
                                fisher_priority=(local_priority * scale / (1.0 + direction_rank)),
                                atom=atom,
                            )
                        )
    return tuple(
        sorted(
            proposals,
            key=lambda row: (-row.fisher_priority, row.candidate_id),
        )
    )


def prove_current_g49_role_collision() -> dict[str, Any]:
    """Prove that the current G49 analytic payload drops semantic atom role."""

    common_atom = {
        "pair_index": 0,
        "center_y": 160,
        "center_x": 256,
        "scale_y": 24,
        "scale_x": 96,
        "shear_q4": 0,
        "amplitude_q4": 64,
    }
    common_factor = {
        "section_id": "g72.role_collision_proof",
        "source_pair_start": 0,
        "source_pair_stop_exclusive": 1,
        "frame_selector": SelectedPreimageFrameSelectorV1.BOTH,
        "source_rgb_u8": (11, 3, 9),
        "added_rgb_u8": (12, 4, 10),
        "removed_rgb_u8": (10, 2, 8),
        "source_receipt_sha256": _sha256(b"g72 structural role-collision proof; no video payload"),
    }
    road = build_analytic_shearlet_residual_factor(
        atoms=(BoundaryShearletAtomV1(role="Road", **common_atom),),
        **common_factor,
    )
    undrivable = build_analytic_shearlet_residual_factor(
        atoms=(
            BoundaryShearletAtomV1(
                role="UndrivableBoundary",
                **common_atom,
            ),
        ),
        **common_factor,
    )
    collided = road.payload == undrivable.payload
    return {
        "road_payload_sha256": road.payload_sha256,
        "undrivable_payload_sha256": undrivable.payload_sha256,
        "payloads_byte_identical": collided,
        "role_preserved_by_current_g49_wire": not collided,
        "blocker": G49_ROLE_WIRE_OWED if collided else None,
    }


@dataclass(frozen=True, slots=True)
class G72WholeObjectMeasurementV1:
    """One exact n600 same-object measurement used for joint admission."""

    archive_sha256: str
    measurement_receipt_sha256: str
    evaluator_closure_sha256: str
    target_custody_receipt_sha256: str
    evidence_axis: str
    archive_bytes: int
    segmentation_error_count: int
    pose_squared_error_sum: float
    scorer_batch_size: int = EXPECTED_SCORER_BATCH_SIZE
    pair_count: int = PAIR_COUNT
    receiver_closed: bool = True
    parse_back_closed: bool = True
    double_decode_equal: bool = True

    def __post_init__(self) -> None:
        for label in (
            "archive_sha256",
            "measurement_receipt_sha256",
            "evaluator_closure_sha256",
            "target_custody_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if type(self.evidence_axis) is not str or not self.evidence_axis or not self.evidence_axis.isascii():
            raise G72AnalyticFactorCompilerError("evidence_axis must be nonempty ASCII")
        _require_exact_int(
            self.archive_bytes,
            label="archive_bytes",
            minimum=1,
            maximum=1 << 40,
        )
        _require_exact_int(
            self.segmentation_error_count,
            label="segmentation_error_count",
            minimum=0,
            maximum=PAIR_COUNT * SCORER_HEIGHT * SCORER_WIDTH,
        )
        pose_error = float(self.pose_squared_error_sum)
        if not math.isfinite(pose_error) or pose_error < 0.0:
            raise G72AnalyticFactorCompilerError("pose_squared_error_sum must be finite and nonnegative")
        if (
            self.scorer_batch_size != EXPECTED_SCORER_BATCH_SIZE
            or self.pair_count != PAIR_COUNT
            or self.receiver_closed is not True
            or self.parse_back_closed is not True
            or self.double_decode_equal is not True
        ):
            raise G72AnalyticFactorCompilerError(
                "joint admission requires receiver-closed, parse-backed, double-decoded exact n600 batch16 evidence"
            )

    @property
    def d_seg(self) -> float:
        return self.segmentation_error_count / (PAIR_COUNT * SCORER_HEIGHT * SCORER_WIDTH)

    @property
    def d_pose(self) -> float:
        return self.pose_squared_error_sum / (PAIR_COUNT * POSE_COORDINATES_PER_PAIR)

    @property
    def score(self) -> float:
        return 100.0 * self.d_seg + math.sqrt(10.0 * self.d_pose) + 25.0 * self.archive_bytes / SOURCE_VIDEO_BYTES


def admit_exact_whole_object_change(
    *,
    current: G72WholeObjectMeasurementV1,
    proposed: G72WholeObjectMeasurementV1,
) -> dict[str, Any]:
    """Admit iff the exact whole-object score decreases.

    There are intentionally no independent segmentation, pose, or byte
    thresholds.  Every trade-off is decided by the contest equation.
    """

    if type(current) is not G72WholeObjectMeasurementV1 or type(proposed) is not G72WholeObjectMeasurementV1:
        raise G72AnalyticFactorCompilerError("joint admission requires exact whole-object measurements")
    if current.archive_sha256 == proposed.archive_sha256:
        raise G72AnalyticFactorCompilerError("proposed and current archive identities are identical")
    if (
        current.evaluator_closure_sha256 != proposed.evaluator_closure_sha256
        or current.target_custody_receipt_sha256 != proposed.target_custody_receipt_sha256
        or current.evidence_axis != proposed.evidence_axis
    ):
        raise G72AnalyticFactorCompilerError(
            "joint admission measurements differ in evaluator, target, or axis custody"
        )
    delta = proposed.score - current.score
    body = {
        "schema": JOINT_ADMISSION_SCHEMA,
        "current": {
            "archive_sha256": current.archive_sha256,
            "measurement_receipt_sha256": current.measurement_receipt_sha256,
            "archive_bytes": current.archive_bytes,
            "segmentation_error_count": current.segmentation_error_count,
            "pose_squared_error_sum": format(
                current.pose_squared_error_sum,
                ".17g",
            ),
            "d_seg": format(current.d_seg, ".17g"),
            "d_pose": format(current.d_pose, ".17g"),
            "score": format(current.score, ".17g"),
        },
        "proposed": {
            "archive_sha256": proposed.archive_sha256,
            "measurement_receipt_sha256": proposed.measurement_receipt_sha256,
            "archive_bytes": proposed.archive_bytes,
            "segmentation_error_count": proposed.segmentation_error_count,
            "pose_squared_error_sum": format(
                proposed.pose_squared_error_sum,
                ".17g",
            ),
            "d_seg": format(proposed.d_seg, ".17g"),
            "d_pose": format(proposed.d_pose, ".17g"),
            "score": format(proposed.score, ".17g"),
        },
        "joint_score_delta": format(delta, ".17g"),
        "evaluator_closure_sha256": current.evaluator_closure_sha256,
        "target_custody_receipt_sha256": current.target_custody_receipt_sha256,
        "evidence_axis": current.evidence_axis,
        "admitted": delta < 0.0,
        "admission_law": ("100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489 strictly decreases"),
        "independent_component_thresholds_used": False,
    }
    return _seal(body, field="admission_receipt_sha256")


def write_stage_checkpoint(
    *,
    output_root: Path,
    stage: G72StagePlanV1,
    semantic_archive_sha256: str,
    semantic_compile_receipt_sha256: str,
    g46_target_receipt_sha256: str,
    g51_operand_receipt_sha256: str,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    described_cells: np.ndarray,
    minimum_component_sites: int,
    maximum_components_per_pair_role: int,
    proposals: tuple[G72BoundaryShearletProposalV1, ...],
    previous_checkpoint_sha256: str | None,
) -> Path:
    """Atomically preserve one immutable role-aware proposal stage."""

    root = require_ssd_output_root(Path(output_root))
    if type(stage) is not G72StagePlanV1:
        raise G72AnalyticFactorCompilerError("checkpoint requires an exact G72 stage")
    target = _require_stage_field(
        target_cells,
        label="target_cells",
        dtype_kind="integer",
    )
    margins = _require_stage_field(
        target_margins,
        label="target_margins",
        dtype_kind="floating",
    )
    described = _require_stage_field(
        described_cells,
        label="described_cells",
        dtype_kind="integer",
    )
    expected_proposals = derive_v9_boundary_shearlet_stage_proposals(
        stage=stage,
        target_cells=target,
        target_margins=margins,
        described_cells=described,
        minimum_component_sites=minimum_component_sites,
        maximum_components_per_pair_role=maximum_components_per_pair_role,
    )
    if type(proposals) is not tuple or any(type(row) is not G72BoundaryShearletProposalV1 for row in proposals):
        raise G72AnalyticFactorCompilerError("checkpoint proposals must be an exact tuple")
    if proposals != expected_proposals:
        raise G72AnalyticFactorCompilerError("checkpoint proposals differ from the scorer-native V9 derivation")
    if any(not stage.pair_start <= row.atom.pair_index < stage.pair_stop_exclusive for row in proposals):
        raise G72AnalyticFactorCompilerError("checkpoint proposal escapes its 120-pair stage")
    for label, digest in (
        ("semantic_archive_sha256", semantic_archive_sha256),
        (
            "semantic_compile_receipt_sha256",
            semantic_compile_receipt_sha256,
        ),
        ("g46_target_receipt_sha256", g46_target_receipt_sha256),
        ("g51_operand_receipt_sha256", g51_operand_receipt_sha256),
    ):
        _require_sha256(digest, label=label)
    if stage.stage_index == 0:
        if previous_checkpoint_sha256 is not None:
            raise G72AnalyticFactorCompilerError("stage zero must not name a predecessor")
    else:
        _require_sha256(
            previous_checkpoint_sha256,
            label="previous_checkpoint_sha256",
        )

    body = {
        "schema": STAGE_CHECKPOINT_SCHEMA,
        "stage_index": stage.stage_index,
        "pair_range": [
            stage.pair_start,
            stage.pair_stop_exclusive,
        ],
        "semantic_archive_sha256": semantic_archive_sha256,
        "semantic_compile_receipt_sha256": (semantic_compile_receipt_sha256),
        "g46_target_receipt_sha256": g46_target_receipt_sha256,
        "g51_operand_receipt_sha256": g51_operand_receipt_sha256,
        "input_fields": {
            "target_cells": {
                "dtype": target.dtype.str,
                "shape": list(target.shape),
                "sha256": _sha256_array(target),
            },
            "target_margins": {
                "dtype": margins.dtype.str,
                "shape": list(margins.shape),
                "sha256": _sha256_array(margins),
            },
            "described_cells": {
                "dtype": described.dtype.str,
                "shape": list(described.shape),
                "sha256": _sha256_array(described),
            },
        },
        "derivation_config": {
            "minimum_component_sites": minimum_component_sites,
            "maximum_components_per_pair_role": (maximum_components_per_pair_role),
            "proposal_law": ("V9_FISHER_MARGIN_BOUNDARY_SHEARLET_ROLE_PRESERVING"),
        },
        "proposals": [row.to_dict() for row in proposals],
        "proposal_fingerprints": [row.fingerprint for row in proposals],
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "checkpoint_policy": ("immutable_atomic_preserve_every_120_pair_stage"),
        "candidate_claim": False,
        "score_claim": False,
        "research_only": True,
    }
    checkpoint = _seal(body, field="checkpoint_sha256")
    payload = canonical_json_bytes(checkpoint)
    path = (
        root
        / "stage_checkpoints"
        / (f"stage_{stage.stage_index:02d}_{stage.pair_start:04d}_{stage.pair_stop_exclusive:04d}.json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise G72AnalyticFactorCompilerError(f"immutable stage checkpoint differs: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    reopened = json.loads(path.read_bytes())
    if type(reopened) is not dict:
        raise G72AnalyticFactorCompilerError("stage checkpoint did not parse as an object")
    _verify_seal(reopened, field="checkpoint_sha256")
    if canonical_json_bytes(reopened) != payload:
        raise G72AnalyticFactorCompilerError("stage checkpoint changed across parse-back")
    return path


def reopen_stage_checkpoint(
    path: Path,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Reopen one immutable stage before resuming the next stage."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise G72AnalyticFactorCompilerError("stage checkpoint must be a regular file")
    value = json.loads(checkpoint_path.read_bytes())
    if type(value) is not dict:
        raise G72AnalyticFactorCompilerError("stage checkpoint is not one object")
    _verify_seal(value, field="checkpoint_sha256")
    if value["checkpoint_sha256"] != _require_sha256(
        expected_checkpoint_sha256,
        label="expected_checkpoint_sha256",
    ):
        raise G72AnalyticFactorCompilerError("stage checkpoint SHA differs from resume custody")
    if value.get("schema") != STAGE_CHECKPOINT_SCHEMA:
        raise G72AnalyticFactorCompilerError("stage checkpoint schema differs")
    if set(value) != {
        "candidate_claim",
        "checkpoint_policy",
        "checkpoint_sha256",
        "derivation_config",
        "g46_target_receipt_sha256",
        "g51_operand_receipt_sha256",
        "input_fields",
        "pair_range",
        "previous_checkpoint_sha256",
        "proposal_fingerprints",
        "proposals",
        "research_only",
        "schema",
        "score_claim",
        "semantic_archive_sha256",
        "semantic_compile_receipt_sha256",
        "stage_index",
    }:
        raise G72AnalyticFactorCompilerError("stage checkpoint has a noncanonical key set")
    pair_range = value.get("pair_range")
    if type(pair_range) is not list or len(pair_range) != 2:
        raise G72AnalyticFactorCompilerError("stage checkpoint pair range differs")
    stage = G72StagePlanV1(
        stage_index=value.get("stage_index"),
        pair_start=pair_range[0],
        pair_stop_exclusive=pair_range[1],
    )
    if value.get("checkpoint_policy") != ("immutable_atomic_preserve_every_120_pair_stage"):
        raise G72AnalyticFactorCompilerError("stage checkpoint weakened resume policy")
    if (
        value.get("candidate_claim") is not False
        or value.get("score_claim") is not False
        or value.get("research_only") is not True
    ):
        raise G72AnalyticFactorCompilerError("stage checkpoint weakened its authority boundary")
    for label in (
        "semantic_archive_sha256",
        "semantic_compile_receipt_sha256",
        "g46_target_receipt_sha256",
        "g51_operand_receipt_sha256",
    ):
        _require_sha256(value.get(label), label=label)
    previous = value.get("previous_checkpoint_sha256")
    if stage.stage_index == 0:
        if previous is not None:
            raise G72AnalyticFactorCompilerError("stage zero checkpoint names a predecessor")
    else:
        _require_sha256(previous, label="previous_checkpoint_sha256")
    inputs = value.get("input_fields")
    if type(inputs) is not dict or set(inputs) != {
        "described_cells",
        "target_cells",
        "target_margins",
    }:
        raise G72AnalyticFactorCompilerError("stage checkpoint input fields differ")
    for label, row in inputs.items():
        if (
            type(row) is not dict
            or set(row) != {"dtype", "sha256", "shape"}
            or row["shape"] != [PAIRS_PER_STAGE, SCORER_HEIGHT, SCORER_WIDTH]
            or type(row["dtype"]) is not str
        ):
            raise G72AnalyticFactorCompilerError(f"stage checkpoint {label} identity differs")
        _require_sha256(row["sha256"], label=f"{label}.sha256")
    config = value.get("derivation_config")
    if (
        type(config) is not dict
        or set(config)
        != {
            "maximum_components_per_pair_role",
            "minimum_component_sites",
            "proposal_law",
        }
        or config["proposal_law"] != "V9_FISHER_MARGIN_BOUNDARY_SHEARLET_ROLE_PRESERVING"
    ):
        raise G72AnalyticFactorCompilerError("stage checkpoint derivation config differs")
    _require_exact_int(
        config["minimum_component_sites"],
        label="minimum_component_sites",
        minimum=1,
        maximum=SCORER_HEIGHT * SCORER_WIDTH,
    )
    _require_exact_int(
        config["maximum_components_per_pair_role"],
        label="maximum_components_per_pair_role",
        minimum=1,
        maximum=4096,
    )
    raw_proposals = value.get("proposals")
    fingerprints = value.get("proposal_fingerprints")
    if type(raw_proposals) is not list or type(fingerprints) is not list:
        raise G72AnalyticFactorCompilerError("stage checkpoint proposals or fingerprints differ")
    proposals = tuple(_proposal_from_dict(row) for row in raw_proposals)
    if fingerprints != [row.fingerprint for row in proposals]:
        raise G72AnalyticFactorCompilerError("stage checkpoint proposal fingerprints differ")
    if any(not stage.pair_start <= row.atom.pair_index < stage.pair_stop_exclusive for row in proposals):
        raise G72AnalyticFactorCompilerError("stage checkpoint proposal escapes its pair range")
    return value


def _load_g46_primary_receipt(
    audit_path: Path,
) -> tuple[dict[str, Any], str]:
    if audit_path.is_symlink() or not audit_path.is_file():
        raise G72AnalyticFactorCompilerError("G46 audit must be a regular file")
    audit = json.loads(audit_path.read_bytes())
    if type(audit) is not dict:
        raise G72AnalyticFactorCompilerError("G46 audit is not an object")
    _verify_seal(audit, field="audit_sha256")
    primary = audit.get("primary")
    if (
        type(primary) is not dict
        or audit.get("verdict") != "PRIMARY_MATCHES_FROZEN_UPSTREAM_DEFAULT_BATCH_GEOMETRY"
        or audit.get("research_only") is not True
        or audit.get("score_claim") is not False
    ):
        raise G72AnalyticFactorCompilerError("G46 audit does not select the exact batch16 primary")
    receipt_file = primary.get("receipt_file")
    if type(receipt_file) is not dict:
        raise G72AnalyticFactorCompilerError("G46 primary receipt binding is absent")
    receipt_path = Path(str(receipt_file.get("path")))
    observed_sha256 = _sha256_file(receipt_path)
    if observed_sha256 != _require_sha256(
        receipt_file.get("sha256"),
        label="G46 primary receipt file SHA",
    ) or receipt_path.stat().st_size != receipt_file.get("bytes"):
        raise G72AnalyticFactorCompilerError("G46 primary receipt file differs from the sealed audit")
    receipt = load_compile_ready_materialization_receipt(receipt_path)
    if receipt.get("receipt_sha256") != primary.get("receipt_sha256"):
        raise G72AnalyticFactorCompilerError("G46 compile-ready receipt self-hash differs from the audit")
    return receipt, observed_sha256


def audit_g72_readiness(
    *,
    semantic_compile_receipt_path: Path,
    semantic_archive_path: Path,
    semantic_producer_root: Path,
    g46_batch_geometry_audit_path: Path,
    g51_operand_aggregate_path: Path,
) -> dict[str, Any]:
    """Recursively reopen current production custody and emit exact blockers."""

    semantic_receipt_path = Path(semantic_compile_receipt_path)
    semantic_bytes_path = Path(semantic_archive_path)
    if (
        semantic_receipt_path.is_symlink()
        or not semantic_receipt_path.is_file()
        or semantic_bytes_path.is_symlink()
        or not semantic_bytes_path.is_file()
    ):
        raise G72AnalyticFactorCompilerError("fresh semantic receipt/archive must be regular files")
    semantic_receipt_bytes = semantic_receipt_path.read_bytes()
    semantic_archive = semantic_bytes_path.read_bytes()
    identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=semantic_receipt_bytes,
        compiled_semantic_archive=semantic_archive,
        producer_root=semantic_producer_root,
    )
    g46_receipt, g46_receipt_file_sha256 = _load_g46_primary_receipt(Path(g46_batch_geometry_audit_path))
    operand_path = Path(g51_operand_aggregate_path)
    operand_file_sha256 = _sha256_file(operand_path)
    operand = FreshScorerPlaneOperandLoaderV1.open(
        operand_path,
        expected_sha256=operand_file_sha256,
    )
    target_sha256 = _require_sha256(
        g46_receipt["target_labels"]["sha256"],
        label="G46 target labels SHA",
    )
    if (
        operand.pair_count != PAIR_COUNT
        or operand.stage_pairs != PAIRS_PER_STAGE
        or len(operand.receipt.get("stages", ())) != STAGE_COUNT
        or operand.receipt.get("target_labels", {}).get("sha256") != target_sha256
        or operand.receipt.get("fresh_teacher_receipt", {}).get("sealed_receipt_sha256")
        != g46_receipt.get("receipt_sha256")
    ):
        raise G72AnalyticFactorCompilerError("G51 five-stage operands do not bind the reopened G46 target coordinate")
    role_collision = prove_current_g49_role_collision()
    blockers = (
        FRESH_BATCH16_MARGIN_CUSTODY_OWED,
        FRESH_V15_BASE_SCORER_CACHE_OWED,
        G49_ROLE_WIRE_OWED,
        V15_ROLE_AWARE_DECODER_OWED,
        POSE_AUTHORITY_OR_FINAL_REPLAY_OWED,
        EXACT_JOINT_STAGE_ADMISSION_OWED,
    )
    body = {
        "schema": READINESS_SCHEMA,
        "fresh_semantic_lineage": {
            "compile_receipt_file_sha256": _sha256(semantic_receipt_bytes),
            "compile_receipt_self_identity_sha256": (identity.fresh_compile_receipt_sha256),
            "semantic_archive_sha256": (identity.compiled_semantic_archive_sha256),
            "semantic_archive_bytes": (identity.compiled_semantic_archive_bytes),
            "pair_count": identity.pair_count,
            "receiver_contract_id": identity.receiver_contract_id,
            "lineage_reopened": True,
        },
        "g46_target_custody": {
            "receipt_file_sha256": g46_receipt_file_sha256,
            "receipt_self_hash": g46_receipt["receipt_sha256"],
            "target_labels_sha256": target_sha256,
            "pair_count": g46_receipt["pair_count"],
            "scorer_batch_size": g46_receipt["scorer_pair_batch_size"],
            "compile_ready_reopened": True,
            "target_margins_present": False,
        },
        "g51_direct_task_operand_custody": {
            "aggregate_file_sha256": operand_file_sha256,
            "aggregate_self_hash": operand.receipt["aggregate_receipt_sha256"],
            "pair_count": operand.pair_count,
            "pairs_per_stage": operand.stage_pairs,
            "stage_count": len(operand.receipt["stages"]),
            "target_labels_sha256": operand.receipt["target_labels"]["sha256"],
            "y0_y1_source_derived_and_recursively_reopened": True,
            "pose_authority": operand.receipt["pose_authority"],
            "fresh_pose_target_authority": False,
        },
        "current_g49_role_semantics": role_collision,
        "current_semantic_receiver_is_legacy_scorer_grid": (V15_SEMANTIC_RECEIVER_ID.endswith(".render_pairs.v9_v13")),
        "canonical_stage_plan": [
            {
                "stage_index": row.stage_index,
                "pair_range": [
                    row.pair_start,
                    row.pair_stop_exclusive,
                ],
            }
            for row in g72_stage_plan()
        ],
        "production_compile_ready": False,
        "open_blockers": list(blockers),
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
    }
    return _seal(body, field="readiness_receipt_sha256")


__all__ = [
    "EXACT_JOINT_STAGE_ADMISSION_OWED",
    "FRESH_BATCH16_MARGIN_CUSTODY_OWED",
    "FRESH_V15_BASE_SCORER_CACHE_OWED",
    "G49_ROLE_WIRE_OWED",
    "JOINT_ADMISSION_SCHEMA",
    "PAIRS_PER_STAGE",
    "PAIR_COUNT",
    "POSE_AUTHORITY_OR_FINAL_REPLAY_OWED",
    "READINESS_SCHEMA",
    "STAGE_CHECKPOINT_SCHEMA",
    "STAGE_COUNT",
    "V15_ROLE_AWARE_DECODER_OWED",
    "G72AnalyticFactorCompilerError",
    "G72BoundaryShearletProposalV1",
    "G72StagePlanV1",
    "G72WholeObjectMeasurementV1",
    "admit_exact_whole_object_change",
    "audit_g72_readiness",
    "derive_v9_boundary_shearlet_stage_proposals",
    "g72_stage_plan",
    "prove_current_g49_role_collision",
    "reopen_stage_checkpoint",
    "write_stage_checkpoint",
]
