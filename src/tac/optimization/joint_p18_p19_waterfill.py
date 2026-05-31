# SPDX-License-Identifier: MIT
"""Joint P18/P19 scorer water-fill weights for rate-axis attacks.

P18 supplies the dense SegNet argmax-flip surface. P19 supplies the PoseNet
Jacobian under the contest pose normalization. The coupled weight is:

    w_i = 100 * |dL_seg/dx_i| + 5 / sqrt(10 * d_pose) * ||J_pose,i||_{Sigma^-1}

where the pose-null subset is the set of atoms whose Mahalanobis pose norm is
below a configured threshold. For Z8 this is the bit allocator for the
rate-binding wavelet-codec attack: low joint-weight atoms are the safe
dead-zone/coarsening set, while high joint-weight atoms protect the already good
SegNet/PoseNet distortion.

The surface is a LOCAL TANGENT PLANE of the full-video contest action ``S_full``
at the archive it was linearized at (``linearization_archive_sha``). Per the
full-video joint-backprop authority (architecture memo correction footer
2026-05-31): every repair / allocation / quantizer decision must ultimately see
the whole-video score functional, not a frozen local map. After each ACCEPTED
archive mutation the surface is STALE and MUST be recomputed; the full-video
inflate/eval replay ledger -- never the stale tangent plane -- is the authority
path. Two orthogonal budget-spend gates apply:

- ``require_full_video_joint_surface`` enforces full-video COVERAGE
  (``evidence_scope == full_video`` and ``observed_atom_count`` covers the
  whole video) -- refuses crop/window-only gradients.
- ``require_fresh_joint_surface`` / ``surface_is_stale_for_archive`` enforce the
  FRESH-SURFACE (trust-region) discipline -- refuses a tangent plane linearized
  at a different archive than the one a decision is about to mutate.

Budget spend is authoritative only when BOTH gates pass: full-video coverage AND
fresh-vs-current-archive. All surfaces remain ``[macOS-MLX research-signal]`` /
non-promotable; the surface proposes local moves, the full-video replay ratifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.target_modes import (
    CONTEST_VIDEO_OVERFIT_MODE,
    normalize_target_optimization_mode,
    target_mode_declares_overfit_allowed,
    target_mode_requires_corpus_manifest,
)

JOINT_P18_P19_WATERFILL_SCHEMA = "joint_p18_p19_waterfill_surface.v1"
JOINT_P18_SEGNET_CLASS_BOUNDARY_SCHEMA = "joint_p18_segnet_class_boundary_modifier.v1"
JOINT_P18_P19_IMPLICIT_ALLOCATOR_SCHEMA = "joint_p18_p19_implicit_kkt_dykstra_allocator.v1"
JOINT_P18_P19_WEIGHT_FORMULA = (
    "w_i = 100*class_boundary_weight_i*abs(dL_seg/dx_i) "
    "+ 5/sqrt(10*d_pose)*||J_pose_i||_{Sigma^-1}"
)
JOINT_P18_P19_RATE_ATTACK_ROLE = (
    "dead_zone_low_joint_weight_wavelet_detail_atoms_to_reduce_rate_while_"
    "protecting_seg_class_boundary_and_pose_sensitive_atoms"
)
FULL_VIDEO_AUTHORITY_SCOPE = "full_video"
PROPOSAL_ONLY_SCOPE = "proposal_only"
FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION = "full_video_exact_accumulation"
PROPOSAL_OR_SAMPLED_REDUCTION = "proposal_or_sampled"


class JointP18P19WaterfillAuthorityError(ValueError):
    """Raised when a joint P18/P19 surface is used beyond its evidence scope."""


def _looks_like_sha256(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


@dataclass(frozen=True)
class JointP18P19WaterfillConfig:
    """Configuration for joint SegNet/PoseNet water-fill surfaces."""

    d_pose: float
    pose_inverse_variance: tuple[float, ...]
    target_mode: str = CONTEST_VIDEO_OVERFIT_MODE
    pose_null_threshold: float = 1e-8
    d_pose_floor: float = 1e-12
    evidence_scope: str = PROPOSAL_ONLY_SCOPE
    full_video_atom_count: int | None = None
    linearization_archive_sha: str | None = None
    gradient_reduction_semantics: str = PROPOSAL_OR_SAMPLED_REDUCTION
    allocator_budget_fraction: float = 0.5
    allocator_atom_capacity: float = 1.0

    def __post_init__(self) -> None:
        if self.d_pose < 0.0:
            raise ValueError("d_pose must be >= 0")
        if self.d_pose_floor <= 0.0:
            raise ValueError("d_pose_floor must be > 0")
        if self.pose_null_threshold < 0.0:
            raise ValueError("pose_null_threshold must be >= 0")
        if not self.pose_inverse_variance:
            raise ValueError("pose_inverse_variance must not be empty")
        if any(value <= 0.0 for value in self.pose_inverse_variance):
            raise ValueError("pose_inverse_variance entries must be > 0")
        normalize_target_optimization_mode(self.target_mode)
        if self.evidence_scope not in {PROPOSAL_ONLY_SCOPE, FULL_VIDEO_AUTHORITY_SCOPE}:
            raise ValueError(f"evidence_scope must be {PROPOSAL_ONLY_SCOPE!r} or {FULL_VIDEO_AUTHORITY_SCOPE!r}")
        if self.gradient_reduction_semantics not in {
            PROPOSAL_OR_SAMPLED_REDUCTION,
            FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        }:
            raise ValueError(
                "gradient_reduction_semantics must be "
                f"{PROPOSAL_OR_SAMPLED_REDUCTION!r} or {FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION!r}"
            )
        if self.full_video_atom_count is not None and self.full_video_atom_count <= 0:
            raise ValueError("full_video_atom_count must be positive when provided")
        if self.linearization_archive_sha is not None and not _looks_like_sha256(self.linearization_archive_sha):
            raise ValueError("linearization_archive_sha must be a 64-character hex sha256 when provided")
        if not 0.0 <= self.allocator_budget_fraction <= 1.0:
            raise ValueError("allocator_budget_fraction must be in [0, 1]")
        if self.allocator_atom_capacity <= 0.0:
            raise ValueError("allocator_atom_capacity must be > 0")

    @property
    def pose_ail_gain(self) -> float:
        return 5.0 / float(np.sqrt(10.0 * max(self.d_pose, self.d_pose_floor)))

    @property
    def normalized_target_mode(self) -> str:
        return normalize_target_optimization_mode(self.target_mode)


def mahalanobis_pose_jacobian_norm(
    pose_jacobian: np.ndarray,
    pose_inverse_variance: np.ndarray,
) -> np.ndarray:
    """Return ``||J_pose,i||_{Sigma^-1}`` over the last dimension."""

    jac = np.asarray(pose_jacobian, dtype=np.float64)
    inv_var = np.asarray(pose_inverse_variance, dtype=np.float64)
    if jac.ndim < 1:
        raise ValueError("pose_jacobian must have a pose dimension")
    if jac.shape[-1] != inv_var.shape[0]:
        raise ValueError("pose_jacobian last dimension must match pose_inverse_variance length")
    return np.sqrt(np.sum((jac * jac) * inv_var, axis=-1))


def _optional_array_matching(
    value: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    name: str,
    dtype: Any,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != shape:
        raise ValueError(f"{name} shape must match segnet_argmax_gradient; got {arr.shape}, expected {shape}")
    return arr


def _full_video_authority_blockers(
    *,
    evidence_scope: str,
    observed_atom_count: int,
    full_video_atom_count: int | None,
) -> list[str]:
    blockers: list[str] = []
    if evidence_scope != FULL_VIDEO_AUTHORITY_SCOPE:
        blockers.append("joint_p18_p19_surface_not_full_video_scope")
    if full_video_atom_count is None:
        blockers.append("joint_p18_p19_full_video_atom_count_missing")
    elif observed_atom_count != int(full_video_atom_count):
        blockers.append(
            "joint_p18_p19_atom_coverage_incomplete:"
            f"observed={observed_atom_count}:full_video={int(full_video_atom_count)}"
        )
    return blockers


def _fresh_surface_blockers(linearization_archive_sha: str | None) -> list[str]:
    if not _looks_like_sha256(linearization_archive_sha):
        return ["joint_p18_p19_linearization_archive_sha_missing"]
    return []


def _gradient_reduction_blockers(gradient_reduction_semantics: str) -> list[str]:
    if gradient_reduction_semantics != FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION:
        return ["joint_p18_p19_gradient_reduction_not_exact_full_video_accumulation"]
    return []


def require_full_video_joint_surface(
    surface: dict[str, Any],
    *,
    context: str = "joint_p18_p19_waterfill_surface",
) -> None:
    """Refuse budget spend from crop/window-only P18/P19 gradients."""

    blockers = list(surface.get("full_video_authority_blockers") or [])
    if surface.get("full_video_authority") is not True or blockers:
        detail = ",".join(str(blocker) for blocker in blockers) or "full_video_authority_false"
        raise JointP18P19WaterfillAuthorityError(f"{context}: {detail}")


def require_exact_full_video_gradient_reduction(
    surface: dict[str, Any],
    *,
    context: str = "joint_p18_p19_waterfill_surface",
) -> None:
    """Refuse budget spend from sampled gradients or chunk-local updates.

    Chunking is allowed only as deterministic memory scheduling:
    ``grad = sum(pair_chunk_grads)`` over the entire video, then one optimizer
    update. Pair/window shards must never update independently.
    """

    blockers = list(surface.get("gradient_reduction_blockers") or [])
    if surface.get("gradient_reduction_authority") is not True or blockers:
        detail = ",".join(str(blocker) for blocker in blockers) or "gradient_reduction_authority_false"
        raise JointP18P19WaterfillAuthorityError(f"{context}: {detail}")


def _project_box_with_safe_mask(
    values: np.ndarray,
    *,
    safe_mask: np.ndarray,
    atom_capacity: float,
) -> np.ndarray:
    projected = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(atom_capacity))
    return np.where(safe_mask, projected, 0.0)


def _project_budget_halfspace(
    values: np.ndarray,
    *,
    safe_mask: np.ndarray,
    budget: float,
) -> np.ndarray:
    projected = np.asarray(values, dtype=np.float64).copy()
    allowed = np.asarray(safe_mask, dtype=bool)
    allowed_count = int(np.count_nonzero(allowed))
    if allowed_count == 0:
        return np.zeros_like(projected)
    excess = float(np.sum(projected[allowed]) - float(budget))
    if excess > 0.0:
        projected[allowed] -= excess / float(allowed_count)
    projected[~allowed] = 0.0
    return projected


def _dykstra_project_box_budget(
    priority: np.ndarray,
    *,
    safe_mask: np.ndarray,
    budget: float,
    atom_capacity: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, float]:
    """Project ``priority`` onto safe-mask, box, and budget constraints."""

    x = np.asarray(priority, dtype=np.float64)
    p_box = np.zeros_like(x)
    p_budget = np.zeros_like(x)
    last_delta = float("inf")
    for iteration in range(1, int(max_iterations) + 1):
        previous = x
        y = _project_box_with_safe_mask(
            x + p_box,
            safe_mask=safe_mask,
            atom_capacity=atom_capacity,
        )
        p_box = x + p_box - y
        x = _project_budget_halfspace(
            y + p_budget,
            safe_mask=safe_mask,
            budget=budget,
        )
        p_budget = y + p_budget - x
        last_delta = float(np.max(np.abs(x - previous))) if x.size else 0.0
        if last_delta <= tolerance:
            return x, iteration, last_delta
    return x, int(max_iterations), last_delta


def solve_joint_p18_p19_implicit_kkt_dykstra_allocator(
    *,
    coarsening_priority: np.ndarray,
    safe_rate_spend_mask: np.ndarray,
    budget: float,
    atom_capacity: float = 1.0,
    dykstra_max_iterations: int = 256,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Solve the differentiable water-fill allocation layer.

    The layer solves the convex projection

        min_x 0.5 ||x - p||^2
        s.t. 0 <= x_i <= capacity_i, x_i = 0 outside the safe mask,
             sum_i x_i <= budget.

    KKT gives the closed-form water level ``x_i = clip(p_i - tau, 0, cap)``.
    Dykstra is used as an independent projection check for the same feasible
    polytope. The returned implicit derivative is active-set based: for a
    binding budget and active uncapped atoms, ``d x_A = d p_A - mean(d p_A)``
    and ``d x_A / d budget = 1 / |A|``. This is the bridge needed by the
    full-video adjoint without pretending that hard archive projection is
    smooth authority.
    """

    priority = np.asarray(coarsening_priority, dtype=np.float64)
    safe_mask = np.asarray(safe_rate_spend_mask, dtype=bool)
    if priority.shape != safe_mask.shape:
        raise ValueError("coarsening_priority shape must match safe_rate_spend_mask")
    if budget < 0.0:
        raise ValueError("budget must be >= 0")
    if atom_capacity <= 0.0:
        raise ValueError("atom_capacity must be > 0")
    if dykstra_max_iterations <= 0:
        raise ValueError("dykstra_max_iterations must be > 0")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")

    safe_priority = np.where(safe_mask, np.maximum(priority, 0.0), 0.0)
    safe_count = int(np.count_nonzero(safe_mask))
    clipped_without_budget = _project_box_with_safe_mask(
        safe_priority,
        safe_mask=safe_mask,
        atom_capacity=atom_capacity,
    )
    effective_budget = min(float(budget), float(safe_count) * float(atom_capacity))
    budget_binding = bool(float(np.sum(clipped_without_budget)) > effective_budget + tolerance)

    if safe_count == 0 or effective_budget <= 0.0:
        allocation = np.zeros_like(priority)
        tau = float("inf")
    elif not budget_binding:
        allocation = clipped_without_budget
        tau = 0.0
    else:
        low = float(np.min(safe_priority[safe_mask]) - atom_capacity)
        high = float(np.max(safe_priority[safe_mask]))
        for _ in range(96):
            mid = (low + high) / 2.0
            candidate = _project_box_with_safe_mask(
                safe_priority - mid,
                safe_mask=safe_mask,
                atom_capacity=atom_capacity,
            )
            if float(np.sum(candidate)) > effective_budget:
                low = mid
            else:
                high = mid
        tau = high
        allocation = _project_box_with_safe_mask(
            safe_priority - tau,
            safe_mask=safe_mask,
            atom_capacity=atom_capacity,
        )

    dykstra_allocation, dykstra_iterations, dykstra_delta = _dykstra_project_box_budget(
        safe_priority,
        safe_mask=safe_mask,
        budget=effective_budget,
        atom_capacity=atom_capacity,
        max_iterations=dykstra_max_iterations,
        tolerance=tolerance,
    )
    dykstra_residual = float(np.max(np.abs(allocation - dykstra_allocation))) if allocation.size else 0.0
    equality_residual = max(0.0, float(np.sum(allocation)) - effective_budget)
    lower_bound_residual = max(0.0, float(-np.min(allocation))) if allocation.size else 0.0
    upper_bound_residual = max(0.0, float(np.max(allocation[safe_mask] - float(atom_capacity)))) if safe_count else 0.0
    unsafe_residual = float(np.max(np.abs(allocation[~safe_mask]))) if np.any(~safe_mask) else 0.0
    active_mask = safe_mask & (allocation > tolerance) & (allocation < float(atom_capacity) - tolerance)
    active_count = int(np.count_nonzero(active_mask))
    budget_jacobian = np.zeros_like(allocation)
    implicit_diag = np.zeros_like(allocation)
    if budget_binding and active_count:
        budget_jacobian[active_mask] = 1.0 / float(active_count)
        implicit_diag[active_mask] = 1.0 - (1.0 / float(active_count))
    else:
        free_mask = safe_mask & (allocation > tolerance) & (allocation < float(atom_capacity) - tolerance)
        implicit_diag[free_mask] = 1.0

    blockers: list[str] = []
    if equality_residual > tolerance:
        blockers.append("joint_p18_p19_allocator_budget_residual_nonzero")
    if lower_bound_residual > tolerance or upper_bound_residual > tolerance:
        blockers.append("joint_p18_p19_allocator_box_residual_nonzero")
    if unsafe_residual > tolerance:
        blockers.append("joint_p18_p19_allocator_unsafe_atom_residual_nonzero")
    if dykstra_residual > max(1e-7, tolerance * 32.0):
        blockers.append("joint_p18_p19_allocator_dykstra_projection_mismatch")

    return {
        "schema": JOINT_P18_P19_IMPLICIT_ALLOCATOR_SCHEMA,
        "objective": "min_0.5_norm_allocation_minus_priority_squared_subject_to_box_safe_mask_budget",
        "kkt_solution": "allocation_i=clip(priority_i-water_level_tau,0,atom_capacity)",
        "implicit_differentiation_contract": (
            "active_set_jvp:dx_A=dp_A-mean(dp_A);dx_A_dbudget=1/active_count_when_budget_binding"
        ),
        "budget": float(budget),
        "effective_budget": float(effective_budget),
        "atom_capacity": float(atom_capacity),
        "safe_atom_count": safe_count,
        "budget_binding": budget_binding,
        "water_level_tau": float(tau),
        "allocation": allocation,
        "active_mask": active_mask,
        "active_count": active_count,
        "implicit_jacobian_diagonal": implicit_diag,
        "implicit_budget_jacobian": budget_jacobian,
        "dykstra_projection": dykstra_allocation,
        "dykstra_iterations": int(dykstra_iterations),
        "dykstra_final_delta": float(dykstra_delta),
        "kkt_residuals": {
            "budget_excess": float(equality_residual),
            "lower_bound": float(lower_bound_residual),
            "upper_bound": float(upper_bound_residual),
            "unsafe_mask": float(unsafe_residual),
            "dykstra_projection_linf": float(dykstra_residual),
        },
        "solver_blockers": blockers,
        "solver_converged": not blockers,
    }


def surface_is_stale_for_archive(
    surface: dict[str, Any],
    current_archive_sha: str,
) -> bool:
    """Return whether ``surface`` is a stale tangent plane for ``current_archive_sha``.

    A joint P18/P19 surface is a LOCAL TANGENT PLANE of the full-video action
    ``S_full`` at the archive it was linearized at. After an accepted archive
    mutation the tangent plane no longer describes the new archive, so the
    surface is stale and the solver MUST recompute P18 + P19 on the new archive
    before the next decision (the fresh-surface / trust-region discipline).

    Returns ``True`` (stale) when the surface is unpinned or when its pinned
    ``linearization_archive_sha`` differs from ``current_archive_sha``. Unpinned
    surfaces are treated as stale because they cannot prove which archive
    produced their tangent plane.
    """

    if not _looks_like_sha256(current_archive_sha):
        raise ValueError("current_archive_sha must be a 64-character hex sha256")
    pinned = surface.get("linearization_archive_sha")
    return str(pinned or "").strip().lower() != current_archive_sha.strip().lower()


def require_fresh_joint_surface(
    surface: dict[str, Any],
    *,
    current_archive_sha: str,
    context: str = "joint_p18_p19_waterfill_surface",
) -> None:
    """Refuse budget spend from a tangent plane linearized at a different archive.

    This is the FRESH-SURFACE budget-spend gate, orthogonal to
    ``require_full_video_joint_surface`` (which enforces full-video COVERAGE).
    Budget spend is authoritative only when BOTH gates pass: full-video coverage
    AND fresh-vs-current-archive. Per the full-video joint-backprop authority,
    the surface must have a pinned ``linearization_archive_sha`` equal to the
    archive the decision is about to mutate -- otherwise the move is being made
    on a stale (or unprovable) tangent plane instead of the full-video ledger.
    """
    pinned = surface.get("linearization_archive_sha")
    if not _looks_like_sha256(pinned):
        raise JointP18P19WaterfillAuthorityError(f"{context}: joint_p18_p19_surface_linearization_archive_sha_missing")
    if surface_is_stale_for_archive(surface, current_archive_sha):
        raise JointP18P19WaterfillAuthorityError(
            f"{context}: joint_p18_p19_surface_stale_tangent_plane:"
            f"linearized_at={pinned}:"
            f"current_archive={current_archive_sha}"
        )


def build_joint_p18_p19_waterfill_surface(
    *,
    segnet_argmax_gradient: np.ndarray,
    pose_jacobian: np.ndarray,
    config: JointP18P19WaterfillConfig,
    segnet_class_region_weight: np.ndarray | None = None,
    segnet_boundary_protect_mask: np.ndarray | None = None,
    segnet_class_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a coupled P18/P19 acquisition surface.

    ``segnet_argmax_gradient`` and the pose norm must describe the same atom
    grid: pixel, boundary, region, pair, batch, or full-video atoms are all
    valid as long as the leading dimensions match.
    """

    seg = np.abs(np.asarray(segnet_argmax_gradient, dtype=np.float64))
    class_region_weight = _optional_array_matching(
        segnet_class_region_weight,
        shape=seg.shape,
        name="segnet_class_region_weight",
        dtype=np.float64,
    )
    if class_region_weight is None:
        class_region_weight = np.ones_like(seg, dtype=np.float64)
    if not np.all(np.isfinite(class_region_weight)):
        raise ValueError("segnet_class_region_weight must be finite")
    if np.any(class_region_weight <= 0.0):
        raise ValueError("segnet_class_region_weight entries must be > 0")
    boundary_protect_mask = _optional_array_matching(
        segnet_boundary_protect_mask,
        shape=seg.shape,
        name="segnet_boundary_protect_mask",
        dtype=bool,
    )
    if boundary_protect_mask is None:
        boundary_protect_mask = np.zeros_like(seg, dtype=bool)
    class_ids = _optional_array_matching(
        segnet_class_ids,
        shape=seg.shape,
        name="segnet_class_ids",
        dtype=np.int64,
    )
    pose_norm = mahalanobis_pose_jacobian_norm(
        np.asarray(pose_jacobian, dtype=np.float64),
        np.asarray(config.pose_inverse_variance, dtype=np.float64),
    )
    if seg.shape != pose_norm.shape:
        raise ValueError("segnet_argmax_gradient shape must match pose_jacobian leading shape")
    seg_term = 100.0 * seg * class_region_weight
    pose_term = config.pose_ail_gain * pose_norm
    joint = seg_term + pose_term
    coarsening_priority = 1.0 / (1.0 + joint)
    observed_atom_count = int(seg.size)
    authority_blockers = _full_video_authority_blockers(
        evidence_scope=config.evidence_scope,
        observed_atom_count=observed_atom_count,
        full_video_atom_count=config.full_video_atom_count,
    )
    fresh_blockers = _fresh_surface_blockers(config.linearization_archive_sha)
    gradient_reduction_blockers = _gradient_reduction_blockers(config.gradient_reduction_semantics)
    target_mode = config.normalized_target_mode
    pose_null_mask = pose_norm <= float(config.pose_null_threshold)
    safe_rate_spend_mask = pose_null_mask & ~boundary_protect_mask
    ready_for_budget_spend = not authority_blockers and not fresh_blockers and not gradient_reduction_blockers
    allocator_budget = (
        float(np.count_nonzero(safe_rate_spend_mask))
        * float(config.allocator_atom_capacity)
        * float(config.allocator_budget_fraction)
    )
    allocator = solve_joint_p18_p19_implicit_kkt_dykstra_allocator(
        coarsening_priority=coarsening_priority,
        safe_rate_spend_mask=safe_rate_spend_mask,
        budget=allocator_budget,
        atom_capacity=config.allocator_atom_capacity,
    )
    allocator_blockers = list(allocator["solver_blockers"])
    if not ready_for_budget_spend:
        allocator_blockers.append("joint_p18_p19_allocator_budget_spend_authority_missing")
    rate_attack_deadzone_mask = np.asarray(allocator["allocation"], dtype=np.float64) > 0.0
    distortion_protect_mask = ~rate_attack_deadzone_mask | boundary_protect_mask
    class_boundary_report = {
        "schema": JOINT_P18_SEGNET_CLASS_BOUNDARY_SCHEMA,
        "class_region_weight_applied": bool(segnet_class_region_weight is not None),
        "class_ids_present": bool(class_ids is not None),
        "boundary_protect_mask_applied": bool(segnet_boundary_protect_mask is not None),
        "boundary_protect_atom_count": int(np.count_nonzero(boundary_protect_mask)),
        "pose_null_atom_count": int(np.count_nonzero(pose_null_mask)),
        "safe_rate_spend_atom_count": int(np.count_nonzero(safe_rate_spend_mask)),
        "class_region_weight_min": float(np.min(class_region_weight)) if class_region_weight.size else 0.0,
        "class_region_weight_max": float(np.max(class_region_weight)) if class_region_weight.size else 0.0,
        "class_region_weight_mean": float(np.mean(class_region_weight)) if class_region_weight.size else 0.0,
    }
    if class_ids is not None:
        valid_class_ids = class_ids[class_ids >= 0]
        unique_ids, counts = np.unique(valid_class_ids, return_counts=True) if valid_class_ids.size else ([], [])
        class_boundary_report["class_histogram"] = {
            str(int(class_id)): int(count)
            for class_id, count in zip(unique_ids, counts, strict=False)
        }
    return {
        "schema": JOINT_P18_P19_WATERFILL_SCHEMA,
        "formula": JOINT_P18_P19_WEIGHT_FORMULA,
        "rate_axis_attack_role": JOINT_P18_P19_RATE_ATTACK_ROLE,
        "d_pose": float(config.d_pose),
        "pose_ail_gain": float(config.pose_ail_gain),
        "pose_null_threshold": float(config.pose_null_threshold),
        "target_mode": target_mode,
        "declared_overfit_allowed": target_mode_declares_overfit_allowed(target_mode),
        "corpus_manifest_required": target_mode_requires_corpus_manifest(target_mode),
        "objective_mode_contract": {
            "schema": "joint_p18_p19_objective_mode_contract.v1",
            "target_mode": target_mode,
            "contest_mode_uses_declared_video_as_optimization_target": (target_mode == CONTEST_VIDEO_OVERFIT_MODE),
            "production_modes_use_same_surface_contract_with_declared_corpus": (
                target_mode_requires_corpus_manifest(target_mode)
            ),
            "no_hidden_video_or_corpus_binding": True,
        },
        "evidence_scope": config.evidence_scope,
        "observed_atom_count": observed_atom_count,
        "full_video_atom_count": config.full_video_atom_count,
        "full_video_coverage_fraction": (
            None
            if config.full_video_atom_count is None
            else float(observed_atom_count) / float(config.full_video_atom_count)
        ),
        "full_video_authority": not authority_blockers,
        "full_video_authority_blockers": authority_blockers,
        "gradient_reduction_semantics": config.gradient_reduction_semantics,
        "gradient_reduction_authority": not gradient_reduction_blockers,
        "gradient_reduction_blockers": gradient_reduction_blockers,
        "pair_chunk_updates_forbidden": True,
        "optimizer_update_semantics": (
            "single_update_after_all_pair_shards_reduce" if not gradient_reduction_blockers else "no_update_probe_only"
        ),
        "linearization_archive_sha": config.linearization_archive_sha,
        "surface_is_local_tangent_plane": True,
        "recompute_required_after_archive_mutation": True,
        "fresh_surface_authority": not fresh_blockers,
        "fresh_surface_blockers": fresh_blockers,
        "ready_for_budget_spend": ready_for_budget_spend,
        "segnet_term": seg_term,
        "segnet_class_boundary_report": class_boundary_report,
        "segnet_class_region_weight": class_region_weight,
        "segnet_boundary_protect_mask": boundary_protect_mask,
        "segnet_class_ids": class_ids,
        "pose_mahalanobis_norm": pose_norm,
        "pose_term": pose_term,
        "joint_weight": joint,
        "coarsening_priority": coarsening_priority,
        "pose_null_mask": pose_null_mask,
        "rate_attack_allocation": allocator["allocation"],
        "rate_attack_deadzone_mask": rate_attack_deadzone_mask,
        "distortion_protect_mask": distortion_protect_mask,
        "safe_rate_spend_mask": safe_rate_spend_mask,
        "implicit_kkt_dykstra_allocator": {
            key: value
            for key, value in allocator.items()
            if key
            not in {
                "allocation",
                "active_mask",
                "dykstra_projection",
                "implicit_jacobian_diagonal",
                "implicit_budget_jacobian",
            }
        },
        "implicit_allocator_authority": ready_for_budget_spend and not allocator_blockers,
        "implicit_allocator_blockers": allocator_blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "CONTEST_VIDEO_OVERFIT_MODE",
    "FULL_VIDEO_AUTHORITY_SCOPE",
    "FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION",
    "JOINT_P18_P19_IMPLICIT_ALLOCATOR_SCHEMA",
    "JOINT_P18_P19_RATE_ATTACK_ROLE",
    "JOINT_P18_P19_WATERFILL_SCHEMA",
    "JOINT_P18_P19_WEIGHT_FORMULA",
    "JOINT_P18_SEGNET_CLASS_BOUNDARY_SCHEMA",
    "PROPOSAL_ONLY_SCOPE",
    "PROPOSAL_OR_SAMPLED_REDUCTION",
    "JointP18P19WaterfillAuthorityError",
    "JointP18P19WaterfillConfig",
    "build_joint_p18_p19_waterfill_surface",
    "mahalanobis_pose_jacobian_norm",
    "require_exact_full_video_gradient_reduction",
    "require_fresh_joint_surface",
    "require_full_video_joint_surface",
    "solve_joint_p18_p19_implicit_kkt_dykstra_allocator",
    "surface_is_stale_for_archive",
]
