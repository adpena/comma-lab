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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

JOINT_P18_P19_WATERFILL_SCHEMA = "joint_p18_p19_waterfill_surface.v1"
JOINT_P18_P19_WEIGHT_FORMULA = (
    "w_i = 100*abs(dL_seg/dx_i) + 5/sqrt(10*d_pose)*||J_pose_i||_{Sigma^-1}"
)
JOINT_P18_P19_RATE_ATTACK_ROLE = (
    "dead_zone_low_joint_weight_wavelet_detail_atoms_to_reduce_rate_while_"
    "protecting_seg_boundary_and_pose_sensitive_atoms"
)
FULL_VIDEO_AUTHORITY_SCOPE = "full_video"
PROPOSAL_ONLY_SCOPE = "proposal_only"


class JointP18P19WaterfillAuthorityError(ValueError):
    """Raised when a joint P18/P19 surface is used beyond its evidence scope."""


@dataclass(frozen=True)
class JointP18P19WaterfillConfig:
    """Configuration for joint SegNet/PoseNet water-fill surfaces."""

    d_pose: float
    pose_inverse_variance: tuple[float, ...]
    pose_null_threshold: float = 1e-8
    d_pose_floor: float = 1e-12
    evidence_scope: str = PROPOSAL_ONLY_SCOPE
    full_video_atom_count: int | None = None

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
        if self.evidence_scope not in {PROPOSAL_ONLY_SCOPE, FULL_VIDEO_AUTHORITY_SCOPE}:
            raise ValueError(
                f"evidence_scope must be {PROPOSAL_ONLY_SCOPE!r} or {FULL_VIDEO_AUTHORITY_SCOPE!r}"
            )
        if self.full_video_atom_count is not None and self.full_video_atom_count <= 0:
            raise ValueError("full_video_atom_count must be positive when provided")

    @property
    def pose_ail_gain(self) -> float:
        return 5.0 / float(np.sqrt(10.0 * max(self.d_pose, self.d_pose_floor)))


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
        raise ValueError(
            "pose_jacobian last dimension must match pose_inverse_variance length"
        )
    return np.sqrt(np.sum((jac * jac) * inv_var, axis=-1))


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


def build_joint_p18_p19_waterfill_surface(
    *,
    segnet_argmax_gradient: np.ndarray,
    pose_jacobian: np.ndarray,
    config: JointP18P19WaterfillConfig,
) -> dict[str, Any]:
    """Build a coupled P18/P19 acquisition surface.

    ``segnet_argmax_gradient`` and the pose norm must describe the same atom
    grid: pixel, boundary, region, pair, batch, or full-video atoms are all
    valid as long as the leading dimensions match.
    """

    seg = np.abs(np.asarray(segnet_argmax_gradient, dtype=np.float64))
    pose_norm = mahalanobis_pose_jacobian_norm(
        np.asarray(pose_jacobian, dtype=np.float64),
        np.asarray(config.pose_inverse_variance, dtype=np.float64),
    )
    if seg.shape != pose_norm.shape:
        raise ValueError(
            "segnet_argmax_gradient shape must match pose_jacobian leading shape"
        )
    seg_term = 100.0 * seg
    pose_term = config.pose_ail_gain * pose_norm
    joint = seg_term + pose_term
    observed_atom_count = int(seg.size)
    authority_blockers = _full_video_authority_blockers(
        evidence_scope=config.evidence_scope,
        observed_atom_count=observed_atom_count,
        full_video_atom_count=config.full_video_atom_count,
    )
    pose_null_mask = pose_norm <= float(config.pose_null_threshold)
    rate_attack_deadzone_mask = pose_null_mask & (joint <= np.median(joint))
    distortion_protect_mask = ~rate_attack_deadzone_mask
    return {
        "schema": JOINT_P18_P19_WATERFILL_SCHEMA,
        "formula": JOINT_P18_P19_WEIGHT_FORMULA,
        "rate_axis_attack_role": JOINT_P18_P19_RATE_ATTACK_ROLE,
        "d_pose": float(config.d_pose),
        "pose_ail_gain": float(config.pose_ail_gain),
        "pose_null_threshold": float(config.pose_null_threshold),
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
        "ready_for_budget_spend": not authority_blockers,
        "segnet_term": seg_term,
        "pose_mahalanobis_norm": pose_norm,
        "pose_term": pose_term,
        "joint_weight": joint,
        "pose_null_mask": pose_null_mask,
        "rate_attack_deadzone_mask": rate_attack_deadzone_mask,
        "distortion_protect_mask": distortion_protect_mask,
        "safe_rate_spend_mask": rate_attack_deadzone_mask,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "FULL_VIDEO_AUTHORITY_SCOPE",
    "JOINT_P18_P19_RATE_ATTACK_ROLE",
    "JOINT_P18_P19_WATERFILL_SCHEMA",
    "JOINT_P18_P19_WEIGHT_FORMULA",
    "JointP18P19WaterfillAuthorityError",
    "JointP18P19WaterfillConfig",
    "PROPOSAL_ONLY_SCOPE",
    "build_joint_p18_p19_waterfill_surface",
    "mahalanobis_pose_jacobian_norm",
    "require_full_video_joint_surface",
]
