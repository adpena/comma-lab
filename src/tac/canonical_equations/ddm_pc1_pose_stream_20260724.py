# SPDX-License-Identifier: MIT
"""Canonical triality equations for the counted DDM PC1 pose-stream owner."""

from __future__ import annotations

import math

import numpy as np

EQUATION_ID = "ddm_pc1_pose_stream_laws_v1"
SOURCE_BYTES = 37_545_489
VERDICT_SCOPE = (
    "PC1 counted pose-stream member composed independently with each exact W parent; "
    "n600 batch32 macOS-CPU frozen-scorer rows are advisory and non-promotional"
)


def two_frame_receiver_law(
    *,
    source_frame: np.ndarray,
    xi: np.ndarray,
    depth: np.ndarray,
    warp: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Φ(qξ): frame0=W(-ξ/2,D)x and frame1=W(ξ,D)frame0.

    ``warp`` is a callable receiver implementation.  The equation stays
    backend-neutral while making the second-frame dependency explicit.
    """

    if not callable(warp):
        raise ValueError("warp must be callable")
    xi_value = np.asarray(xi, dtype=np.float64)
    if xi_value.shape != (6,) or not np.all(np.isfinite(xi_value)):
        raise ValueError("xi must be a finite six-vector")
    depth_value = np.asarray(depth)
    if depth_value.ndim != 2 or not np.all(np.isfinite(depth_value)) or np.any(depth_value <= 0):
        raise ValueError("depth must be a finite positive field")
    frame0 = warp(source_frame, -0.5 * xi_value, depth_value)
    frame1 = warp(frame0, xi_value, depth_value)
    return np.asarray(frame0), np.asarray(frame1)


def ms4d_pose_quadratic(
    candidate_pose6: np.ndarray,
    center_pose6: np.ndarray,
    low_rank_factor: np.ndarray,
) -> float:
    """Q_pose=(p-c)^T L^T L (p-c), without a tube-membership claim."""

    candidate = np.asarray(candidate_pose6, dtype=np.float64)
    center = np.asarray(center_pose6, dtype=np.float64)
    factor = np.asarray(low_rank_factor, dtype=np.float64)
    if candidate.shape != (6,) or center.shape != (6,) or factor.shape != (6, 6):
        raise ValueError("MS4d quadratic geometry differs")
    if not all(np.all(np.isfinite(value)) for value in (candidate, center, factor)):
        raise ValueError("MS4d quadratic inputs must be finite")
    projected = factor @ (candidate - center)
    return float(projected @ projected)


def non_telescoping_conditional_delta_s(
    *,
    parent_dseg: float,
    parent_dpose: float,
    candidate_dseg: float,
    candidate_dpose: float,
    parent_bytes: int,
    candidate_bytes: int,
) -> float:
    """Direct S(W⊕PC1)-S(W), never a sum of intermediate score deltas."""

    metrics = (parent_dseg, parent_dpose, candidate_dseg, candidate_dpose)
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in metrics):
        raise ValueError("metrics must be finite and nonnegative")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (parent_bytes, candidate_bytes)
    ):
        raise ValueError("byte counts must be nonnegative integers")
    return (
        100.0 * (candidate_dseg - parent_dseg)
        + math.sqrt(10.0 * candidate_dpose)
        - math.sqrt(10.0 * parent_dpose)
        + 25.0 * (candidate_bytes - parent_bytes) / SOURCE_BYTES
    )


def admission_fence(
    *,
    exact_parseback: bool,
    inactive_byte_identity: bool,
    nonzero_composite_r_support: bool,
    both_parents_exact_replay: bool,
    unique_effect_owner: bool,
    n600_batch32_measured: bool,
    descent_was_run: bool,
    conditional_delta_s: float | None,
) -> tuple[bool, bool]:
    """Return ``(admitted, tube_claim_allowed)`` with fail-closed semantics."""

    typed = (
        exact_parseback
        and inactive_byte_identity
        and nonzero_composite_r_support
        and both_parents_exact_replay
        and unique_effect_owner
        and n600_batch32_measured
    )
    if conditional_delta_s is not None and not math.isfinite(conditional_delta_s):
        raise ValueError("conditional_delta_s must be finite when present")
    admitted = bool(typed and conditional_delta_s is not None)
    tube_claim_allowed = bool(admitted and descent_was_run)
    return admitted, tube_claim_allowed


__all__ = [
    "EQUATION_ID",
    "SOURCE_BYTES",
    "VERDICT_SCOPE",
    "admission_fence",
    "ms4d_pose_quadratic",
    "non_telescoping_conditional_delta_s",
    "two_frame_receiver_law",
]
