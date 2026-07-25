# SPDX-License-Identifier: MIT
"""Canonical laws for the PC1 frame-0 PoseNet quotient carrier.

The frozen evaluator factorization makes frame 0 a SegNet quotient coordinate:
SegNet consumes only frame 1 while PoseNet consumes the ordered pair.  P1
therefore admits a receiver whose only write is frame 0 and whose frame-1
output is the exact parent byte string.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

EQUATION_ID = "ddm_p1_frame0_pose_quotient_rank_law_v1"
POSE_DIMS = 6
DELEGATED_TARGET_D_POSE = 5.0e-5
GC4_STRICT_TARGET_D_POSE = 2.94e-5
MAX_CARRIER_BYTES = 30_000
MIN_FALSIFIER_ROWS = 5
VERDICT_SCOPE = (
    "FORMULATION: shared low-rank, quantized, parent-additive frame-0 actuator "
    "basis with per-pair coefficients; frame 1 is an exact parent-byte identity; "
    "macOS-CPU frozen-scorer rows are advisory and cannot promote the family"
)


def frame0_quotient_law(
    parent_pairs: np.ndarray,
    realized_frame0: np.ndarray,
) -> np.ndarray:
    """Return ``(realized_frame0, parent_frame1)`` with exact frame-1 identity."""

    parent = np.asarray(parent_pairs)
    frame0 = np.asarray(realized_frame0)
    if (
        parent.dtype != np.uint8
        or parent.ndim != 5
        or parent.shape[1] != 2
        or parent.shape[-1] != 3
        or frame0.dtype != np.uint8
        or frame0.shape != parent[:, 0].shape
    ):
        raise ValueError("frame-0 quotient receiver geometry differs")
    result = np.empty_like(parent)
    result[:, 0] = frame0
    result[:, 1] = parent[:, 1]
    if not np.array_equal(result[:, 1], parent[:, 1]):
        raise AssertionError("frame-1 quotient identity failed")
    return result


def pose_targeted_actuator(
    jacobian: np.ndarray,
    target_residual: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Return the minimum-norm damped actuator ``Jᵀ(JJᵀ+λI)⁻¹e``.

    ``J`` is one pair's frozen PoseNet Jacobian from a scorer-recursive,
    receiver-realized frame-0 chart to Pose6.  This construction is targeted
    by the exact target residual and is not a generic spatial menu.
    """

    jac = np.asarray(jacobian, dtype=np.float64)
    residual = np.asarray(target_residual, dtype=np.float64)
    if (
        jac.ndim != 2
        or jac.shape[0] != POSE_DIMS
        or residual.shape != (POSE_DIMS,)
        or not np.all(np.isfinite(jac))
        or not np.all(np.isfinite(residual))
        or not math.isfinite(float(ridge))
        or ridge <= 0.0
    ):
        raise ValueError("pose-targeted actuator inputs differ")
    gram = jac @ jac.T
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = jac.T @ np.linalg.solve(
            gram + float(ridge) * np.eye(POSE_DIMS),
            residual,
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("pose-targeted actuator solve produced nonfinite values")
    return np.ascontiguousarray(result)


def descending_covariance_spectrum(actuators: np.ndarray) -> np.ndarray:
    """Return all nonnegative eigenvalues of the shared actuator covariance."""

    values = np.asarray(actuators, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("actuators must be a finite (pairs,coordinates) matrix")
    centered = values - values.mean(axis=0, keepdims=True)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gram = centered @ centered.T
    if not np.all(np.isfinite(gram)):
        raise ValueError("actuator covariance Gram matrix is nonfinite")
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    return np.ascontiguousarray(eigenvalues)


def linearized_pose_floor(
    *,
    rank: int,
    eigenvalues: Sequence[float],
    baseline_d_pose: float,
) -> float:
    """Preregistered PCA tail model ``D_lin(r)=D0·Σ[j>r]λj/Σjλj``."""

    spectrum = np.asarray(tuple(eigenvalues), dtype=np.float64)
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or spectrum.ndim != 1
        or len(spectrum) == 0
        or np.any(~np.isfinite(spectrum))
        or np.any(spectrum < 0.0)
        or not math.isfinite(float(baseline_d_pose))
        or baseline_d_pose < 0.0
    ):
        raise ValueError("linearized rank-law inputs differ")
    total = float(spectrum.sum(dtype=np.float64))
    if total <= 0.0:
        return float(baseline_d_pose)
    tail = float(spectrum[min(rank, len(spectrum)) :].sum(dtype=np.float64))
    return float(baseline_d_pose) * tail / total


def canonical_rank_law(
    *,
    eigenvalues: Sequence[float],
    baseline_d_pose: float,
    target_d_pose: float = DELEGATED_TARGET_D_POSE,
    maximum_rank: int = POSE_DIMS,
) -> dict[str, Any]:
    """Select the least PCA rank whose preregistered linear floor reaches target."""

    if (
        isinstance(maximum_rank, bool)
        or not isinstance(maximum_rank, int)
        or maximum_rank < 1
        or not math.isfinite(float(target_d_pose))
        or target_d_pose < 0.0
    ):
        raise ValueError("rank-law target or maximum rank differs")
    rows = [
        {
            "rank": rank,
            "predicted_linearized_d_pose": linearized_pose_floor(
                rank=rank,
                eigenvalues=eigenvalues,
                baseline_d_pose=baseline_d_pose,
            ),
        }
        for rank in range(1, maximum_rank + 1)
    ]
    selected = next(
        (row["rank"] for row in rows if row["predicted_linearized_d_pose"] <= target_d_pose),
        None,
    )
    return {
        "equation_id": EQUATION_ID,
        "law": "D_lin(r)=D0*sum_{j>r}(lambda_j)/sum_j(lambda_j)",
        "target_d_pose": float(target_d_pose),
        "maximum_rank": maximum_rank,
        "selected_rank": selected,
        "rows": rows,
        "status": "RANK_SELECTED" if selected is not None else "NO_RANK_REACHES_LINEARIZED_TARGET",
    }


def matched_control_fence(
    *,
    treatment_packet_bytes: int,
    control_packet_bytes: int,
    treatment_frame1_sha256: str,
    control_frame1_sha256: str,
    parent_frame1_sha256: str,
    same_rank: bool,
    same_precision: bool,
    same_solver: bool,
) -> bool:
    """Require exact rate, frame-1, rank, precision, and solver matching."""

    byte_values = (treatment_packet_bytes, control_packet_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in byte_values):
        raise ValueError("matched-control byte counts must be positive integers")
    hashes = (treatment_frame1_sha256, control_frame1_sha256, parent_frame1_sha256)
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("matched-control frame-1 hashes must be lowercase SHA-256")
    return bool(
        treatment_packet_bytes == control_packet_bytes
        and treatment_packet_bytes <= MAX_CARRIER_BYTES
        and treatment_frame1_sha256 == parent_frame1_sha256
        and control_frame1_sha256 == parent_frame1_sha256
        and same_rank
        and same_precision
        and same_solver
    )


def reach_curve_disposition(rows: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Return a treatment disposition without over-generalizing a negative."""

    if len(rows) < MIN_FALSIFIER_ROWS:
        raise ValueError("P1 falsifier requires at least five exact reach rows")
    ranks = [int(row["rank"]) for row in rows]
    if ranks != sorted(set(ranks)):
        raise ValueError("P1 reach rows must have strictly increasing unique ranks")
    if any(
        not math.isfinite(float(row["d_pose"]))
        or float(row["d_pose"]) < 0.0
        or int(row["carrier_bytes"]) <= 0
        for row in rows
    ):
        raise ValueError("P1 reach rows contain invalid metrics")
    if any(
        float(row["d_pose"]) <= DELEGATED_TARGET_D_POSE
        and int(row["carrier_bytes"]) <= MAX_CARRIER_BYTES
        for row in rows
    ):
        return (
            "DELEGATED_P1_TARGET_REACHED",
            VERDICT_SCOPE,
        )
    return (
        "P1_SHARED_LOW_RANK_FRAME0_ACTUATOR_FORMULATION_BLOCKED",
        VERDICT_SCOPE,
    )


__all__ = [
    "DELEGATED_TARGET_D_POSE",
    "EQUATION_ID",
    "GC4_STRICT_TARGET_D_POSE",
    "MAX_CARRIER_BYTES",
    "MIN_FALSIFIER_ROWS",
    "POSE_DIMS",
    "VERDICT_SCOPE",
    "canonical_rank_law",
    "descending_covariance_spectrum",
    "frame0_quotient_law",
    "linearized_pose_floor",
    "matched_control_fence",
    "pose_targeted_actuator",
    "reach_curve_disposition",
]
