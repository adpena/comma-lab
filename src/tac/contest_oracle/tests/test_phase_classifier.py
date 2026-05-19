# SPDX-License-Identifier: MIT
"""Tests for ``tac.contest_oracle.phase_classifier`` -- operating-point phases."""
from __future__ import annotations

import math

import pytest

from tac.contest_oracle.phase_classifier import (
    CROSSOVER_POSE_AVG,
    ContestPhase,
    OptimalAttackRecommendation,
    classify_phase,
    recommend_attack,
)


def test_crossover_threshold_is_2_5e_4():
    """Closed-form: dS/d_pose == 100 when d_pose = 2.5e-4."""
    assert CROSSOVER_POSE_AVG == 2.5e-4


def test_old_1x_seg_dominant_phase():
    """pose_avg=0.18 -> SEG_DOMINANT_OLD_1X + SEG_FIRST recommendation."""
    pc = classify_phase(d_pose=0.18)
    assert pc.phase == ContestPhase.SEG_DOMINANT_OLD_1X
    assert pc.optimal_attack == OptimalAttackRecommendation.SEG_FIRST_TERTIARY_POSE


def test_pr106_frontier_pose_dominant():
    """pose_avg=3.4e-5 (PR106 frontier) -> POSE_DOMINANT_FRONTIER."""
    pc = classify_phase(d_pose=3.4e-5)
    assert pc.phase == ContestPhase.POSE_DOMINANT_FRONTIER
    assert pc.optimal_attack == OptimalAttackRecommendation.POSE_FIRST_TERTIARY_SEG
    # Per CLAUDE.md anchor: pose marginal ~2.71x seg marginal at this point
    assert math.isclose(pc.pose_to_seg_marginal_ratio, 2.71, abs_tol=0.05)


def test_mid_transition_phase():
    """pose_avg=0.001 -> MID_TRANSITION."""
    pc = classify_phase(d_pose=0.001)
    assert pc.phase == ContestPhase.MID_TRANSITION


def test_crossover_phase():
    """pose_avg near 2.5e-4 -> CROSSOVER."""
    pc = classify_phase(d_pose=2.5e-4)
    assert pc.phase == ContestPhase.CROSSOVER


def test_classify_rejects_negative():
    with pytest.raises(ValueError):
        classify_phase(d_pose=-0.01)


def test_crossover_distance_signed_correctly():
    """log10(d_pose / 2.5e-4); negative -> frontier side, positive -> seg-dominant side."""
    pc_high = classify_phase(d_pose=2.5e-3)  # 10x above crossover
    assert pc_high.crossover_distance > 0  # positive = above crossover
    pc_low = classify_phase(d_pose=2.5e-5)  # 10x below
    assert pc_low.crossover_distance < 0


def test_recommend_attack_at_frontier_recommends_pose():
    """Operator-facing convenience: at frontier, recommend POSE_FIRST."""
    rec = recommend_attack(d_seg=0.001, d_pose=3.4e-5, archive_bytes=500_000)
    assert rec == OptimalAttackRecommendation.POSE_FIRST_TERTIARY_SEG


def test_recommend_attack_at_old_1x_recommends_seg():
    rec = recommend_attack(d_seg=0.1, d_pose=0.18, archive_bytes=500_000)
    assert rec == OptimalAttackRecommendation.SEG_FIRST_TERTIARY_POSE


def test_phase_classification_for_zero_pose_is_frontier():
    """d_pose = 0 is theoretically below frontier boundary."""
    pc = classify_phase(d_pose=0.0)
    assert pc.phase == ContestPhase.POSE_DOMINANT_FRONTIER
    assert pc.pose_to_seg_marginal_ratio == math.inf
    assert pc.crossover_distance == -math.inf
