# SPDX-License-Identifier: MIT
"""Fail-closed producers for the DDM MS4 scorer-metric measurement.

The functions in this module deliberately separate *identified inputs* from
semantic PF2 keys.  A PF2 key is not an actuator, receiver input, or scorer
sample assignment.  Producers therefore refuse bucket-specific tensors until
an explicit assignment is present in every atlas row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from tac.optimization.ddm_metric_custody_bundle import (
    HARD_PAIR_ORDER,
    PAIR_COUNT,
    PF2_BUCKET_COUNT,
    POSE_OUTPUT_DIMENSION,
    SCORER_BATCH_SIZE,
)
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)

PRODUCER_SCHEMA: Final = "ddm_ms4_metric_producer_receipt.v1"
BLOCKER_SCHEMA: Final = "ddm_ms4_metric_producer_blocker.v1"
POSE_BLOCK_SCHEMA: Final = "ddm_ms4_pose_batch32_checkpoint.v1"
POSE_TUBE_RADIUS: Final = 0.05
POSE_TUBE_SOURCE: Final = (
    "DDMV16CoupledJointSolveConfigV1.pose_trust_radius; "
    "tools/measure_ddm_v16_coupled_joint_solve.py"
)


class MetricProducerError(ValueError):
    """An input cannot support the requested scorer-metric claim."""


@dataclass(frozen=True, slots=True)
class PF2AssignmentAudit:
    bucket_count: int
    assigned_count: int
    unassigned_bucket_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.bucket_count == PF2_BUCKET_COUNT and self.assigned_count == self.bucket_count


def metric_tag(stream_type: StreamType, layer_home: LayerHome) -> dict[str, Any]:
    """Return the sealed typed-stream declaration shared by producer rows."""

    return TypedStreamTag(
        type=stream_type,
        layer_home=layer_home,
        evaluate_py_recursion_level_cited=f"{layer_home.value} metric measurement -> L5_verdict",
        counted_bytes=0,
        free_receiver_code=True,
    ).to_dict()


def audit_pf2_bucket_assignments(pf2: Mapping[str, Any]) -> PF2AssignmentAudit:
    """Audit explicit bucket-to-input mappings without inferring missing data.

    The accepted mapping is intentionally narrow.  Each row must contain a
    ``measurement_assignment`` object naming the pair IDs and a receiver
    actuator/direction ID.  Merely carrying a bucket ID, zero event count, or
    an occupancy label does not identify a measurement.
    """

    atlas = pf2.get("typed_split_atlas")
    rows = atlas.get("rows") if isinstance(atlas, Mapping) else None
    if not isinstance(rows, list) or len(rows) != PF2_BUCKET_COUNT:
        raise MetricProducerError("PF2 atlas does not contain exactly 1,200 rows")
    assigned = 0
    unassigned: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("bucket_id"), str):
            raise MetricProducerError("PF2 atlas contains an untyped bucket row")
        assignment = row.get("measurement_assignment")
        valid = (
            isinstance(assignment, Mapping)
            and isinstance(assignment.get("pair_ids"), list)
            and len(assignment["pair_ids"]) == PAIR_COUNT
            and assignment["pair_ids"] == list(range(PAIR_COUNT))
            and isinstance(assignment.get("receiver_actuator_id"), str)
            and bool(assignment["receiver_actuator_id"])
            and isinstance(assignment.get("direction_id"), str)
            and bool(assignment["direction_id"])
        )
        if valid:
            assigned += 1
        else:
            unassigned.append(str(row["bucket_id"]))
    return PF2AssignmentAudit(
        bucket_count=len(rows),
        assigned_count=assigned,
        unassigned_bucket_ids=tuple(unassigned),
    )


def validate_hard_pair_schedule(g3: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the preregistered hard/control schedule after strict validation."""

    result: dict[str, tuple[int, ...]] = {}
    for field, count in (("top24", 24), ("top64", 64), ("stratified_control24", 24)):
        rows = g3.get(field)
        if (
            not isinstance(rows, list)
            or len(rows) != count
            or any(isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < PAIR_COUNT for row in rows)
            or len(set(rows)) != count
        ):
            raise MetricProducerError(f"G3 {field} is malformed")
        result[field] = tuple(rows)
    if result["top64"][:24] != result["top24"]:
        raise MetricProducerError("G3 top64 does not preserve top24 prefix")
    result["full_n600"] = tuple(range(PAIR_COUNT))
    return result


def padded_batch32(pair_ids: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pad a smoke subset deterministically while preserving the measured IDs."""

    ids = tuple(int(row) for row in pair_ids)
    if not ids or len(ids) > SCORER_BATCH_SIZE or len(set(ids)) != len(ids):
        raise MetricProducerError("smoke batch IDs must be unique with length 1..32")
    if any(not 0 <= row < PAIR_COUNT for row in ids):
        raise MetricProducerError("smoke batch pair ID is outside 0..599")
    padding = tuple(row for row in range(PAIR_COUNT) if row not in ids)[: SCORER_BATCH_SIZE - len(ids)]
    return ids + padding, padding


def pose_quadratic_row(
    pair_id: int,
    center: Sequence[float],
    *,
    observed_against_registered_center_max_abs: float,
) -> dict[str, Any]:
    """Produce the exact six-dimensional PoseNet output-MSE quadratic.

    For ``mean((pose6 - center)**2)``, a factor ``I/sqrt(6)`` gives
    ``||F.T @ delta||^2 = mean(delta**2)`` exactly.  Hence the analytic solve
    always converges when the fresh center is finite.
    """

    if isinstance(pair_id, bool) or not isinstance(pair_id, int) or not 0 <= pair_id < PAIR_COUNT:
        raise MetricProducerError("pair_id must be an exact integer in 0..599")
    vector = np.asarray(center, dtype=np.float64)
    if vector.shape != (POSE_OUTPUT_DIMENSION,) or not np.isfinite(vector).all():
        raise MetricProducerError("Pose center must be one finite Pose6 vector")
    drift = float(observed_against_registered_center_max_abs)
    if not math.isfinite(drift) or drift < 0.0:
        raise MetricProducerError("registered-center drift must be finite and nonnegative")
    factor = np.eye(POSE_OUTPUT_DIMENSION, dtype=np.float64) / math.sqrt(POSE_OUTPUT_DIMENSION)
    return {
        "pair_id": pair_id,
        "center": vector.tolist(),
        "low_rank_factors": factor.tolist(),
        "rank": POSE_OUTPUT_DIMENSION,
        "tube_radius": POSE_TUBE_RADIUS,
        "tube_radius_epistemic_status": "DERIVED_FROM_SEALED_CONTEST_BUDGET",
        "tube_radius_source": POSE_TUBE_SOURCE,
        "converged": True,
        "convergence_status": "CONVERGED",
        "registered_center_max_abs_delta": drift,
        "typed_stream_tag": metric_tag(StreamType.FIBER, LayerHome.L5_VERDICT),
    }


def non_converged_pose_row(pair_id: int, reason: str) -> dict[str, Any]:
    """Emit an explicit non-convergence record for a failed scorer block."""

    if not reason or not reason.replace("_", "").isalnum():
        raise MetricProducerError("non-convergence reason must be a stable token")
    return {
        "pair_id": pair_id,
        "center": [0.0] * POSE_OUTPUT_DIMENSION,
        "low_rank_factors": np.eye(POSE_OUTPUT_DIMENSION, dtype=np.float64).tolist(),
        "rank": POSE_OUTPUT_DIMENSION,
        "tube_radius": POSE_TUBE_RADIUS,
        "tube_radius_epistemic_status": "DERIVED_FROM_SEALED_CONTEST_BUDGET",
        "tube_radius_source": POSE_TUBE_SOURCE,
        "converged": False,
        "convergence_status": f"NON_CONVERGED_{reason}",
        "registered_center_max_abs_delta": 0.0,
        "typed_stream_tag": metric_tag(StreamType.FIBER, LayerHome.L5_VERDICT),
    }


def seg_margin_fisher_row(
    atlas_row: Mapping[str, Any],
    margin_fisher_rows: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Contract 600 identified rank-4 Fisher rows into one PF2 metric row."""

    values = np.asarray(margin_fisher_rows, dtype=np.float64)
    if values.shape != (PAIR_COUNT, 4) or not np.isfinite(values).all():
        raise MetricProducerError("Seg margin-Fisher source must be one finite n600x4 array")
    required = (
        "bucket_id",
        "class_pair",
        "class_stratum",
        "visibility",
        "g4_temporal_class",
        "representation_type",
    )
    if any(not isinstance(atlas_row.get(field), str) or not atlas_row[field] for field in required):
        raise MetricProducerError("Seg source row does not preserve the six PF2 key fields")
    gram = np.asarray(values.T @ values / PAIR_COUNT, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    spectrum = np.linalg.eigvalsh(gram)
    if float(spectrum[0]) < -1e-10:
        raise MetricProducerError("computed Seg margin-Fisher Gram is not PSD")
    return {
        **{field: str(atlas_row[field]) for field in required},
        "margin_fisher_gram": gram.tolist(),
        "eigenvalues_ascending": spectrum.tolist(),
        "lambda_range": [max(0.0, float(spectrum[0])), max(0.0, float(spectrum[-1]))],
        "sample_count": PAIR_COUNT,
        "typed_stream_tag": metric_tag(StreamType.SKELETON, LayerHome.L4_SCORER_FEATURE),
    }


def composite_r_second_order_row(
    bucket_id: str,
    *,
    model_hessian: Sequence[Sequence[float]],
    adjoint_readback: Sequence[float],
    realized_secant_positive: Sequence[float],
    realized_secant_negative: Sequence[float],
    secant_amplitude: float,
) -> dict[str, Any]:
    """Validate and emit exact-model plus paired-realized composite-R custody."""

    hessian = np.asarray(model_hessian, dtype=np.float64)
    if (
        hessian.ndim != 2
        or hessian.shape[0] == 0
        or hessian.shape[0] != hessian.shape[1]
        or not np.isfinite(hessian).all()
        or not np.allclose(hessian, hessian.T, rtol=1e-9, atol=1e-11)
    ):
        raise MetricProducerError("composite-R Hessian must be one finite symmetric square matrix")
    hessian = 0.5 * (hessian + hessian.T)
    if float(np.linalg.eigvalsh(hessian).min()) < -1e-9:
        raise MetricProducerError("composite-R Hessian must be positive semidefinite")
    dimension = int(hessian.shape[0])
    vectors = [
        np.asarray(value, dtype=np.float64)
        for value in (adjoint_readback, realized_secant_positive, realized_secant_negative)
    ]
    if any(value.shape != (dimension,) or not np.isfinite(value).all() for value in vectors):
        raise MetricProducerError("composite-R adjoint/secants must match the Hessian dimension")
    amplitude = float(secant_amplitude)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise MetricProducerError("composite-R secant amplitude must be positive")
    if not isinstance(bucket_id, str) or not bucket_id:
        raise MetricProducerError("composite-R bucket ID must be nonempty")
    return {
        "bucket_id": bucket_id,
        "dimension": dimension,
        "model_hessian": hessian.tolist(),
        "adjoint_readback": vectors[0].tolist(),
        "realized_secant_positive": vectors[1].tolist(),
        "realized_secant_negative": vectors[2].tolist(),
        "secant_amplitude": amplitude,
        "typed_stream_tag": metric_tag(StreamType.CONNECTION, LayerHome.L4_SCORER_FEATURE),
    }


def dual_metric_diagnostic_row(
    bucket_id: str,
    *,
    fisher_vector: Sequence[float],
    euclidean_control_vector: Sequence[float],
) -> dict[str, Any]:
    """Emit matched signed cosine and relative norm on identical custody."""

    fisher = np.asarray(fisher_vector, dtype=np.float64)
    euclidean = np.asarray(euclidean_control_vector, dtype=np.float64)
    if (
        fisher.ndim != 1
        or fisher.shape[0] == 0
        or fisher.shape != euclidean.shape
        or not np.isfinite(fisher).all()
        or not np.isfinite(euclidean).all()
    ):
        raise MetricProducerError("dual vectors must be matched finite nonempty vectors")
    fisher_norm = float(np.linalg.norm(fisher))
    euclidean_norm = float(np.linalg.norm(euclidean))
    if fisher_norm <= 0.0 or euclidean_norm <= 0.0:
        raise MetricProducerError("dual vectors must both have positive norm")
    if not isinstance(bucket_id, str) or not bucket_id:
        raise MetricProducerError("dual bucket ID must be nonempty")
    return {
        "bucket_id": bucket_id,
        "fisher_euclidean_cosine": float(np.dot(fisher, euclidean) / (fisher_norm * euclidean_norm)),
        "fisher_to_euclidean_rel_norm": fisher_norm / euclidean_norm,
        "euclidean_role": "LABELED_CONTROL_ONLY",
        "typed_stream_tag": metric_tag(StreamType.RESIDUAL, LayerHome.L5_VERDICT),
    }


def measurement_schedule() -> list[str]:
    return list(HARD_PAIR_ORDER)


__all__ = [
    "BLOCKER_SCHEMA",
    "POSE_BLOCK_SCHEMA",
    "POSE_TUBE_RADIUS",
    "POSE_TUBE_SOURCE",
    "PRODUCER_SCHEMA",
    "MetricProducerError",
    "PF2AssignmentAudit",
    "audit_pf2_bucket_assignments",
    "composite_r_second_order_row",
    "dual_metric_diagnostic_row",
    "measurement_schedule",
    "metric_tag",
    "non_converged_pose_row",
    "padded_batch32",
    "pose_quadratic_row",
    "seg_margin_fisher_row",
    "validate_hard_pair_schedule",
]
