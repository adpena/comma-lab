# SPDX-License-Identifier: MIT
"""Per-pair Lagrangian dual routing decision per Atick-Redlich asymmetric scorer channel.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS":

    min_x  Σ_i (100 × d_seg_i(x) + √(10 × d_pose_i(x)) + 25/37545489 × bytes_i(x))
           s.t. routing_i ∈ {frame_0, frame_1}

Per Catalog #356 per-axis decomposition: the per-pair routing decision IS a
canonical per-axis decomposition emission (seg + pose + bytes) per the
``tac.cathedral.consumer_contract.AxisDecomposition`` schema.

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
at PR106 frontier (pose_avg ~3.4e-5), POSE marginal value is 2.71× SegNet's;
the per-pair Lagrangian formula correctly weights this via the non-linear
``sqrt(10 × pose_avg)`` term.

Per CLAUDE.md "Forbidden score claims": this module produces per-pair routing
decisions for the COMPRESS pass; actual scorer measurements at TRAIN time
require differentiable scorer-preprocess routing per Catalog #164.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "PerPairRoutingDecision",
    "compute_per_pair_lagrangian_dual_routing",
    "SEG_MULTIPLIER",
    "POSE_SQRT_INNER",
    "RATE_MULTIPLIER",
    "RATE_DENOM_BYTES",
    "FRAME_0",
    "FRAME_1",
]

# Canonical contest formula constants per `tac.score_composition` Catalog #356
SEG_MULTIPLIER = 100.0
POSE_SQRT_INNER = 10.0
RATE_MULTIPLIER = 25.0
RATE_DENOM_BYTES = 37_545_489

# Routing decision sentinels
FRAME_0 = np.int8(0)
FRAME_1 = np.int8(1)


@dataclass(frozen=True)
class PerPairRoutingDecision:
    """Per-pair Lagrangian dual routing decision result.

    All arrays have shape (n_pairs,) unless noted.

    Per Catalog #287/#323 canonical Provenance: every score-axis field is
    measured during COMPRESS pass and tagged with axis_tag at the caller.
    """

    selected_mode_idx: np.ndarray  # (n_pairs,) int joint-menu index
    routing_decision: np.ndarray  # (n_pairs,) int8 in {FRAME_0, FRAME_1}
    selected_seg_delta: np.ndarray  # (n_pairs,) float per-pair SegNet delta
    selected_pose_delta: np.ndarray  # (n_pairs,) float per-pair PoseNet delta
    selected_lagrangian: np.ndarray  # (n_pairs,) float per-pair Lagrangian
    frame_0_only_best_lagrangian: np.ndarray  # comparison baseline
    per_pair_improvement: np.ndarray  # frame_0_only - joint; positive = better
    per_pair_score_delta: np.ndarray  # -per_pair_improvement (canonical sign)

    @property
    def n_pairs(self) -> int:
        return int(self.selected_mode_idx.shape[0])

    @property
    def frame_1_count(self) -> int:
        return int(np.sum(self.routing_decision == FRAME_1))

    @property
    def frame_1_pct(self) -> float:
        return 100.0 * self.frame_1_count / max(1, self.n_pairs)

    @property
    def total_score_delta(self) -> float:
        return float(np.sum(self.per_pair_score_delta))


def compute_per_pair_lagrangian_dual_routing(
    frame_0_seg_penalty: np.ndarray,
    frame_0_pose_delta: np.ndarray,
    frame_1_seg_penalty: np.ndarray,
    frame_1_pose_delta: np.ndarray,
    pose_avg_baseline: float,
) -> PerPairRoutingDecision:
    """Per-pair Lagrangian dual routing decision per Atick-Redlich asymmetric channel.

    Inputs (all shape ``(n_pairs, n_modes_per_frame)``):
        frame_0_seg_penalty: SegNet delta per-pair per frame-0 mode (TYPICALLY ALL ZERO per Atick-Redlich)
        frame_0_pose_delta: PoseNet delta per-pair per frame-0 mode
        frame_1_seg_penalty: SegNet delta per-pair per frame-1 mode (POSITIVE; M bytes cost)
        frame_1_pose_delta: PoseNet delta per-pair per frame-1 mode (can be NEGATIVE = savings)
        pose_avg_baseline: PR106 frontier operating point ~3.4e-5

    Returns PerPairRoutingDecision with all per-pair fields populated.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver": closed-form O(n_pairs ×
    n_modes_joint) argmin pass; no iteration required.

    Per Catalog #205 + #356: this routing decision is the canonical
    per-axis-decomposition primitive consumed by the cathedral autopilot
    ranker via the per-axis emission path.
    """
    if frame_0_seg_penalty.shape[0] != frame_0_pose_delta.shape[0]:
        raise ValueError("frame_0 seg+pose mismatched n_pairs")
    if frame_1_seg_penalty.shape[0] != frame_1_pose_delta.shape[0]:
        raise ValueError("frame_1 seg+pose mismatched n_pairs")
    if frame_0_seg_penalty.shape[0] != frame_1_seg_penalty.shape[0]:
        raise ValueError("frame_0 vs frame_1 mismatched n_pairs")
    if pose_avg_baseline <= 0:
        raise ValueError(f"pose_avg_baseline must be positive; got {pose_avg_baseline}")

    n_pairs = frame_0_seg_penalty.shape[0]
    n_frame_0 = frame_0_seg_penalty.shape[1]

    joint_seg = np.hstack([frame_0_seg_penalty, frame_1_seg_penalty])
    joint_pose = np.hstack([frame_0_pose_delta, frame_1_pose_delta])

    # Non-linear pose contribution per canonical formula
    new_pose_total = np.maximum(pose_avg_baseline + joint_pose, 1e-12)
    seg_contrib = SEG_MULTIPLIER * joint_seg
    pose_contrib = (
        np.sqrt(POSE_SQRT_INNER * new_pose_total)
        - np.sqrt(POSE_SQRT_INNER * pose_avg_baseline)
    )
    lagrangian_per_candidate = seg_contrib + pose_contrib

    # Per-pair argmin
    selected_mode_idx = np.argmin(lagrangian_per_candidate, axis=1)
    arange = np.arange(n_pairs)
    selected_lagrangian = lagrangian_per_candidate[arange, selected_mode_idx]
    selected_seg = joint_seg[arange, selected_mode_idx]
    selected_pose = joint_pose[arange, selected_mode_idx]

    # Routing decision: frame_0 if selected_idx < n_frame_0, else frame_1
    routing_decision = (selected_mode_idx >= n_frame_0).astype(np.int8)

    # Comparison: per-pair frame-0-only Lagrangian (PR110 status quo)
    frame_0_only_lagrangian_per = SEG_MULTIPLIER * frame_0_seg_penalty + (
        np.sqrt(POSE_SQRT_INNER * np.maximum(pose_avg_baseline + frame_0_pose_delta, 1e-12))
        - np.sqrt(POSE_SQRT_INNER * pose_avg_baseline)
    )
    frame_0_best_idx = np.argmin(frame_0_only_lagrangian_per, axis=1)
    frame_0_best_lagrangian = frame_0_only_lagrangian_per[arange, frame_0_best_idx]

    # Per-pair improvement: positive = joint menu beats frame-0-only (= lower Lagrangian)
    per_pair_improvement = frame_0_best_lagrangian - selected_lagrangian
    # Canonical score-delta sign convention: NEGATIVE = score reduction = BETTER
    per_pair_score_delta = -per_pair_improvement

    return PerPairRoutingDecision(
        selected_mode_idx=selected_mode_idx,
        routing_decision=routing_decision,
        selected_seg_delta=selected_seg,
        selected_pose_delta=selected_pose,
        selected_lagrangian=selected_lagrangian,
        frame_0_only_best_lagrangian=frame_0_best_lagrangian,
        per_pair_improvement=per_pair_improvement,
        per_pair_score_delta=per_pair_score_delta,
    )
