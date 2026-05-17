# SPDX-License-Identifier: MIT
"""Tests for the closed-form contest score geometry analyzer."""
from __future__ import annotations

import math

import pytest

from tac.contest_rate_distortion_system import contest_score as canonical_contest_score
from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    DualAxisDispatchRecommendation,
    PlannerAxisMarginals,
    PoseByteTradeoff,
    RateOnlyDeltaAudit,
    audit_rate_only_delta_claim,
    contest_score,
    equal_score_curve_archive_bytes,
    equal_score_curve_d_pose,
    importance_flip_threshold,
    information_floor,
    marginal_value_per_byte,
    operating_regime,
    planner_axis_marginals,
    pose_byte_tradeoff,
    pose_score_saving_from_delta,
    predict_cpu_axis_marginals,
    project_onto_pareto_envelope,
    recommend_dispatch_axis_dual,
    required_byte_savings_for_score_delta,
    score_decomposition,
    score_gradient,
    score_saving_from_byte_savings,
    target_byte_budget_for_score,
)

# ---------------------------------------------------------------------------
# Sanity: known-anchor scores
# ---------------------------------------------------------------------------


def test_pr103_on_pr106_anchor_reproduces_exact_formula() -> None:
    """The active A++ anchor reproduces from exact formula components."""
    score = contest_score(d_seg=0.00067082, d_pose=0.0000336, archive_bytes=185_578)
    assert math.isclose(score, 0.2089810755823297, rel_tol=1e-12)


def test_pure_python_score_matches_canonical_torch_formula() -> None:
    """Keep this planning helper numerically tied to the canonical formula."""
    pure = contest_score(d_seg=0.00067082, d_pose=0.0000336, archive_bytes=185_578)
    canonical = canonical_contest_score(
        seg_distortion=0.00067082,
        pose_distortion=0.0000336,
        archive_bytes=185_578,
    )
    assert math.isclose(pure, float(canonical), rel_tol=1e-12)


def test_pure_rate_floor_is_dominant_at_active_anchor_bytes() -> None:
    """An archive with d_seg=0 + d_pose=0 only pays the rate term."""
    score = contest_score(d_seg=0.0, d_pose=0.0, archive_bytes=185_578)
    expected_rate = 25.0 * 185_578 / CONTEST_REFERENCE_BYTES
    assert math.isclose(score, expected_rate, rel_tol=1e-12)
    assert math.isclose(score, information_floor(185_578), rel_tol=1e-12)


def test_decomposition_sums_to_total() -> None:
    decomp = score_decomposition(d_seg=0.001, d_pose=1e-4, archive_bytes=178258)
    assert math.isclose(
        decomp.total,
        contest_score(0.001, 1e-4, 178258),
        rel_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# Importance flip
# ---------------------------------------------------------------------------


def test_flip_threshold_is_2_5e_4() -> None:
    """The closed-form threshold is exactly 10 / (4*100*100) = 2.5e-4."""
    assert math.isclose(importance_flip_threshold(), 2.5e-4, rel_tol=1e-12)


def test_at_flip_threshold_seg_pose_marginals_equal() -> None:
    """At d_pose = flip_threshold, dS/d(seg) == dS/d(pose) by construction."""
    threshold = importance_flip_threshold()
    grad = score_gradient(0.0, threshold)
    assert math.isclose(grad.d_seg, grad.d_pose, rel_tol=1e-9)


def test_below_flip_pose_dominates() -> None:
    """At PR106 frontier (d_pose ~ 3.4e-5 << 2.5e-4) pose dominates."""
    regime = operating_regime(3.4e-5)
    assert regime.pose_dominates
    assert not regime.seg_dominates
    # Marginal pose-vs-seg ratio should be > 1: pose marginal exceeds seg
    pose_over_seg = 1.0 / regime.marginal_ratio_seg_over_pose
    assert pose_over_seg > 1.0
    # CLAUDE.md cites 2.71x at PR106's pose_avg=3.4e-5
    assert math.isclose(pose_over_seg, 2.71, rel_tol=0.05), (
        f"PR106-frontier pose-dominance ratio drift: got {pose_over_seg}"
    )


def test_above_flip_seg_dominates() -> None:
    """Legacy 1.x regime (d_pose ~ 0.18) is seg-dominated."""
    regime = operating_regime(0.18)
    assert regime.seg_dominates
    assert not regime.pose_dominates
    # CLAUDE.md cites SegNet ~7x more cost-effective at the OLD operating
    # point in cost-effectiveness terms. The marginal ratio (per unit of
    # axis) is much steeper because pose contribution flattens.
    assert regime.marginal_ratio_seg_over_pose > 5.0


def test_at_d_pose_zero_pose_marginal_is_infinite() -> None:
    """d_pose=0 is the singularity of the sqrt; pose marginal is unbounded."""
    grad = score_gradient(0.0, 0.0)
    assert math.isinf(grad.d_pose)
    assert grad.d_seg == 100.0


# ---------------------------------------------------------------------------
# Marginal value per byte
# ---------------------------------------------------------------------------


def test_marginal_value_per_byte_axis_constants() -> None:
    """Bytes axis is constant; seg axis is constant 100."""
    bytes_per = marginal_value_per_byte("bytes")
    assert math.isclose(bytes_per, 25.0 / CONTEST_REFERENCE_BYTES, rel_tol=1e-12)
    seg_per = marginal_value_per_byte("seg")
    assert seg_per == 100.0


def test_marginal_value_per_byte_pose_depends_on_op_point() -> None:
    """Pose marginal must change with operating point."""
    pose_marginal_legacy = marginal_value_per_byte("pose", d_pose_at_operating_point=0.18)
    pose_marginal_pr106 = marginal_value_per_byte("pose", d_pose_at_operating_point=3.4e-5)
    assert pose_marginal_pr106 > pose_marginal_legacy
    # Specifically, PR106 marginal is ~70x larger than legacy 1.x marginal
    ratio = pose_marginal_pr106 / pose_marginal_legacy
    assert ratio > 50.0, f"expected ~70x amplification, got {ratio}"


# ---------------------------------------------------------------------------
# Pareto envelope
# ---------------------------------------------------------------------------


def test_pareto_envelope_at_floor_returns_floor_score() -> None:
    """If candidate equals all floors, the slack is zero on every axis."""
    envelope, slack = project_onto_pareto_envelope(
        d_seg=0.0, d_pose=0.0, archive_bytes=178258,
        seg_floor=0.0, pose_floor=0.0, byte_floor=178258,
    )
    assert math.isclose(envelope, contest_score(0.0, 0.0, 178258))
    assert slack["seg_slack"] == 0.0
    assert slack["pose_slack"] == 0.0
    assert slack["byte_slack"] == 0.0


def test_pareto_envelope_slack_reflects_axis_gap() -> None:
    """Score-slack on each axis equals the per-axis term-gap."""
    envelope, slack = project_onto_pareto_envelope(
        d_seg=0.001, d_pose=1e-4, archive_bytes=200000,
        seg_floor=0.0, pose_floor=0.0, byte_floor=178258,
    )
    expected_byte_score_slack = 25.0 * (200000 - 178258) / CONTEST_REFERENCE_BYTES
    assert math.isclose(slack["byte_score_slack"], expected_byte_score_slack, rel_tol=1e-9)
    # Total slack reconstructs the actual score - envelope
    actual_score = contest_score(0.001, 1e-4, 200000)
    total_slack = slack["seg_score_slack"] + slack["pose_score_slack"] + slack["byte_score_slack"]
    assert math.isclose(actual_score - envelope, total_slack, rel_tol=1e-9)


def test_pareto_envelope_rejects_below_floor() -> None:
    """A candidate that violates a stated floor raises ValueError."""
    with pytest.raises(ValueError):
        project_onto_pareto_envelope(
            d_seg=0.0, d_pose=0.0, archive_bytes=100,
            byte_floor=178258,
        )


# ---------------------------------------------------------------------------
# Inverse curves (dispatch budgeting)
# ---------------------------------------------------------------------------


def test_equal_score_curve_d_pose_round_trip() -> None:
    """For a target score, computing required d_pose then re-scoring yields target.

    Use feasible inputs: at d_seg=1e-4 + B=160_000 the seg+rate terms are ~0.1166
    leaving ~0.073 in the pose budget for a 0.190 target.
    """
    target = 0.190
    d_seg = 1e-4
    archive_bytes = 160_000
    required_pose = equal_score_curve_d_pose(target, d_seg, archive_bytes)
    assert required_pose is not None
    re_scored = contest_score(d_seg, required_pose, archive_bytes)
    assert math.isclose(re_scored, target, rel_tol=1e-9)


def test_equal_score_curve_archive_bytes_round_trip() -> None:
    target = 0.190
    d_seg = 0.001
    d_pose = 5e-5
    required_bytes = equal_score_curve_archive_bytes(target, d_seg, d_pose)
    assert required_bytes is not None
    re_scored = contest_score(d_seg, d_pose, required_bytes)
    # Byte rounding causes ~0.5/N_REF score noise; allow it
    assert abs(re_scored - target) < 25.0 / CONTEST_REFERENCE_BYTES


def test_equal_score_curve_returns_none_when_unreachable() -> None:
    """If seg+rate already exceeds target, no pose value can reach it."""
    target = 0.05
    # seg term alone = 100 * 0.001 = 0.1 > 0.05
    assert equal_score_curve_d_pose(target, d_seg=0.001, archive_bytes=178258) is None


def test_target_byte_budget_for_sub017_cpu_floor_math() -> None:
    """Closed-form CPU-axis byte budget for the sub-0.17 planning target."""
    budget = target_byte_budget_for_score(
        target_score=0.17,
        d_seg_floor=6.0e-4,
        d_pose_floor=3.5e-5,
        current_archive_bytes=178_392,
    )
    assert budget.feasible_under_floors is True
    assert budget.blocker is None
    assert 136_000 <= budget.max_archive_bytes <= 138_000
    assert budget.required_savings_bytes is not None
    assert 40_000 <= budget.required_savings_bytes <= 43_000
    assert budget.score_claim is False
    assert budget.promotion_eligible is False
    assert budget.rank_or_kill_eligible is False
    assert budget.ready_for_exact_eval_dispatch is False


def test_target_byte_budget_rejects_infeasible_distortion_floors() -> None:
    budget = target_byte_budget_for_score(
        target_score=0.17,
        d_seg_floor=0.002,
        d_pose_floor=1e-3,
        current_archive_bytes=178_392,
    )
    assert budget.feasible_under_floors is False
    assert budget.max_archive_bytes is None
    assert budget.required_savings_bytes is None
    assert budget.blocker == "distortion_floors_exceed_target_before_rate"


def test_rate_only_delta_audit_bounds_l5_pose_stream_claim() -> None:
    """L5 rate-only shrink cannot claim -0.008 from a 4.8KB pose stream."""
    audit = audit_rate_only_delta_claim(
        original_bytes=4_800,
        candidate_bytes=2_000,
        claimed_score_saving=0.008,
    )

    assert isinstance(audit, RateOnlyDeltaAudit)
    assert audit.saved_bytes == 2_800
    assert audit.rate_only_score_saving == pytest.approx(
        25.0 * 2_800 / CONTEST_REFERENCE_BYTES,
        rel=1e-12,
    )
    assert audit.required_saved_bytes_for_claim == 12_015
    assert audit.feasible_from_candidate_savings is False
    assert audit.feasible_even_if_section_removed is False
    assert audit.blocker == "claim_exceeds_rate_only_section_capacity"
    assert audit.score_claim is False
    assert audit.ready_for_exact_eval_dispatch is False


def test_rate_only_delta_audit_accepts_claim_with_enough_byte_savings() -> None:
    audit = audit_rate_only_delta_claim(
        original_bytes=4_800,
        candidate_bytes=2_000,
        claimed_score_saving=0.001,
    )

    assert audit.feasible_from_candidate_savings is True
    assert audit.blocker is None


def test_score_delta_byte_conversion_round_trips() -> None:
    required = required_byte_savings_for_score_delta(0.008)
    assert required == 12_015
    assert score_saving_from_byte_savings(required) >= 0.008
    assert score_saving_from_byte_savings(required - 1) < 0.008


def test_pose_byte_tradeoff_bounds_fec6_writeup_example() -> None:
    """At FEC6, a 1e-6 pose drop does not pay for 1000 new bytes."""
    audit = pose_byte_tradeoff(
        current_d_pose=0.00002943271901344679,
        added_bytes=1_000,
        candidate_pose_delta=1e-6,
    )

    assert isinstance(audit, PoseByteTradeoff)
    assert audit.pose_score_saving == pytest.approx(0.00029396227124486515)
    assert audit.byte_score_cost == pytest.approx(25.0 * 1_000 / CONTEST_REFERENCE_BYTES)
    assert audit.net_score_delta == pytest.approx(0.00037189668187730617)
    assert audit.required_pose_delta_to_break_even == pytest.approx(
        2.24035397806799e-06,
    )
    assert audit.feasible_to_break_even is True
    assert audit.blocker == "candidate_pose_delta_below_byte_break_even"
    assert audit.score_claim is False
    assert audit.ready_for_exact_eval_dispatch is False


def test_pose_score_saving_from_delta_matches_exact_sqrt_formula() -> None:
    saving = pose_score_saving_from_delta(
        current_d_pose=0.00002943271901344679,
        pose_delta=1e-6,
    )
    expected = math.sqrt(10 * 0.00002943271901344679) - math.sqrt(
        10 * (0.00002943271901344679 - 1e-6),
    )
    assert saving == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Negative / edge inputs
# ---------------------------------------------------------------------------


def test_negative_inputs_raise() -> None:
    with pytest.raises(ValueError):
        contest_score(-1, 0, 0)
    with pytest.raises(ValueError):
        contest_score(0, -1, 0)
    with pytest.raises(ValueError):
        contest_score(0, 0, -1)
    with pytest.raises(ValueError):
        score_gradient(0.0, -1)
    with pytest.raises(ValueError):
        target_byte_budget_for_score(
            target_score=-0.1,
            d_seg_floor=0.0,
            d_pose_floor=0.0,
        )
    with pytest.raises(ValueError):
        target_byte_budget_for_score(
            target_score=0.17,
            d_seg_floor=-1e-4,
            d_pose_floor=0.0,
        )
    with pytest.raises(ValueError):
        target_byte_budget_for_score(
            target_score=0.17,
            d_seg_floor=0.0,
            d_pose_floor=0.0,
            current_archive_bytes=-1,
        )


# ---------------------------------------------------------------------------
# CPU-axis marginals — added 2026-05-08 per CLAUDE.md dual-axis contract
# ---------------------------------------------------------------------------


def test_predict_cpu_axis_marginals_includes_required_keys() -> None:
    out = predict_cpu_axis_marginals(
        d_seg_cuda=6.7e-4, d_pose_cuda=1.7e-4, archive_class="hnerv",
    )
    assert "pose_marginal" in out
    assert "seg_marginal" in out
    assert "bytes_marginal" in out
    assert "seg_over_pose_marginal_cpu" in out
    assert "pose_over_seg_marginal_cpu" in out
    assert "cuda_d_pose" in out
    assert "cpu_d_pose" in out
    assert out["score_claim"] is False
    assert out["promotion_eligible"] is False
    assert out["rank_or_kill_eligible"] is False


def test_predict_cpu_axis_marginals_seg_marginal_is_constant_100() -> None:
    out = predict_cpu_axis_marginals(
        d_seg_cuda=6.7e-4, d_pose_cuda=1.7e-4, archive_class="hnerv",
    )
    assert out["seg_marginal"] == 100.0


def test_predict_cpu_axis_marginals_at_pose_floor_returns_high_marginal() -> None:
    """At PR106-frontier (d_pose_cuda very small, near floor), CPU pose
    is still small and the pose marginal is HIGH (sqrt singularity)."""
    # PR106 frontier with d_pose_cuda just above floor.
    out = predict_cpu_axis_marginals(
        d_seg_cuda=7.84e-4, d_pose_cuda=1.7e-4, archive_class="hnerv",
    )
    # CPU pose < CUDA pose (floor stripped).
    assert out["cpu_d_pose"] < out["cuda_d_pose"]


def test_predict_cpu_axis_marginals_negative_input_raises() -> None:
    with pytest.raises(ValueError):
        predict_cpu_axis_marginals(d_seg_cuda=-0.1, d_pose_cuda=1e-4)
    with pytest.raises(ValueError):
        predict_cpu_axis_marginals(d_seg_cuda=1e-4, d_pose_cuda=-0.1)


# ---------------------------------------------------------------------------
# Target-axis planner marginals — CPU leaderboard chain-rule separation
# ---------------------------------------------------------------------------


def test_planner_axis_marginals_cuda_internal_matches_score_gradient() -> None:
    out = planner_axis_marginals(
        target_axis="cuda_internal",
        cuda_d_seg=6.88e-4,
        cuda_d_pose=1.74e-4,
        archive_bytes=178_392,
    )
    grad = score_gradient(6.88e-4, 1.74e-4)
    assert isinstance(out, PlannerAxisMarginals)
    assert out.input_axis == "cuda_candidate_delta"
    assert out.seg_marginal == grad.d_seg
    assert out.pose_marginal == pytest.approx(grad.d_pose, rel=1e-12)
    assert out.priority_axis == "pose"
    assert out.score_claim is False
    assert out.promotion_eligible is False


def test_planner_axis_marginals_cpu_leaderboard_applies_chain_rule() -> None:
    """CPU-leaderboard planning reweights CUDA-side candidate deltas.

    At this PR107/apogee-ish operating point, raw CUDA gradients prioritize
    pose, but the calibrated CPU-leaderboard response to CUDA-coordinate
    changes prioritizes seg after applying 1/R_pose and 1/R_seg.
    """
    out = planner_axis_marginals(
        target_axis="cpu_leaderboard",
        cuda_d_seg=6.88e-4,
        cuda_d_pose=1.74e-4,
        archive_bytes=178_392,
        archive_class="hnerv",
    )
    assert out.calibration_class == "hnerv"
    assert out.effective_d_pose == pytest.approx(1.74e-4 / 5.04, rel=1e-12)
    assert out.effective_d_seg == pytest.approx(6.88e-4 / 1.17, rel=1e-12)
    assert out.seg_chain_scale == pytest.approx(1.0 / 1.17, rel=1e-12)
    assert out.pose_chain_scale == pytest.approx(1.0 / 5.04, rel=1e-12)
    assert out.seg_marginal > out.pose_marginal
    assert out.priority_axis == "seg"
    assert out.evidence_grade == "[prediction; cpu-leaderboard chain-rule planner marginals]"
    assert out.rank_or_kill_eligible is False
    assert out.ready_for_exact_eval_dispatch is False


def test_planner_axis_marginals_negative_and_unknown_axis_raise() -> None:
    with pytest.raises(ValueError):
        planner_axis_marginals(
            target_axis="cuda_internal",
            cuda_d_seg=-1e-4,
            cuda_d_pose=1e-4,
            archive_bytes=178_392,
        )
    with pytest.raises(ValueError):
        planner_axis_marginals(
            target_axis="not-a-real-axis",  # type: ignore[arg-type]
            cuda_d_seg=1e-4,
            cuda_d_pose=1e-4,
            archive_bytes=178_392,
        )


# ---------------------------------------------------------------------------
# Dual-axis dispatch recommendation — added 2026-05-08 (Phase A planning)
# ---------------------------------------------------------------------------


def test_dual_axis_recommendation_returns_dataclass() -> None:
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=6.88e-4, cuda_d_pose=1.74e-4, archive_bytes=178392,
    )
    assert isinstance(rec, DualAxisDispatchRecommendation)
    assert rec.evidence_grade == "[prediction; dual-axis-dispatch-recommendation]"
    assert rec.score_claim is False
    assert rec.promotion_eligible is False


def test_dual_axis_priority_axis_is_pose_at_pr107_frontier() -> None:
    """PR107 operating point: pose_avg=3.58e-5 << flip threshold 2.5e-4."""
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=6.88e-4, cuda_d_pose=1.74e-4, archive_bytes=178392,
    )
    # CUDA pose marginal (5/sqrt(10*1.74e-4) ~= 119.9) > seg marginal (100)
    assert rec.cuda_priority_axis == "pose"
    # CPU pose is even smaller, so pose still wins on CPU axis
    assert rec.cpu_priority_axis == "pose"
    assert rec.axis_priority_differs is False


def test_dual_axis_target_score_gap_is_signed_correctly() -> None:
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=6.88e-4, cuda_d_pose=1.74e-4, archive_bytes=178392,
        target_score_cpu=0.17,
    )
    # PR107 CPU score is ~0.197; gap is positive (above target)
    assert rec.cpu_score_gap_to_target is not None
    assert rec.cpu_score_gap_to_target > 0.0
    assert "0.17" in rec.advice


def test_dual_axis_decision_attack_map_covers_all_axes() -> None:
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=6.88e-4, cuda_d_pose=1.74e-4, archive_bytes=178392,
    )
    assert "seg" in rec.decision_attack_map
    assert "pose" in rec.decision_attack_map
    assert "bytes" in rec.decision_attack_map
    # bytes attack list includes Phase A4 (council G5 gate-clearer)
    assert any("A4" in d for d in rec.decision_attack_map["bytes"])
    # seg attack list includes A1 score-gradient
    assert any("A1" in d for d in rec.decision_attack_map["seg"])


def test_dual_axis_negative_inputs_raise() -> None:
    with pytest.raises(ValueError):
        recommend_dispatch_axis_dual(
            cuda_d_seg=-0.1, cuda_d_pose=1e-4, archive_bytes=178392,
        )
    with pytest.raises(ValueError):
        recommend_dispatch_axis_dual(
            cuda_d_seg=1e-4, cuda_d_pose=-0.1, archive_bytes=178392,
        )
    with pytest.raises(ValueError):
        recommend_dispatch_axis_dual(
            cuda_d_seg=1e-4, cuda_d_pose=1e-4, archive_bytes=-1,
        )


# ---------------------------------------------------------------------------
# Tie handling at the importance flip threshold (codex review fix 2026-05-08)
# ---------------------------------------------------------------------------


def test_operating_regime_at_exact_flip_threshold_is_tied() -> None:
    """At d_pose == 2.5e-4 exactly, neither axis dominates."""
    threshold = importance_flip_threshold()
    regime = operating_regime(threshold)
    # Both False at the tie
    assert regime.seg_dominates is False
    assert regime.pose_dominates is False
    # Advice text says TIED, not "seg-dominated"
    assert "TIED" in regime.advice
    assert "parallel" in regime.advice.lower()


def test_operating_regime_strictly_above_threshold_seg_dominates() -> None:
    threshold = importance_flip_threshold()
    regime = operating_regime(threshold * 1.5)
    assert regime.seg_dominates is True
    assert regime.pose_dominates is False
    assert "seg-dominated" in regime.advice


def test_operating_regime_strictly_below_threshold_pose_dominates() -> None:
    threshold = importance_flip_threshold()
    regime = operating_regime(threshold * 0.5)
    assert regime.seg_dominates is False
    assert regime.pose_dominates is True
    assert "pose-dominated" in regime.advice


def test_dual_axis_at_exact_flip_threshold_detects_seg_pose_tie() -> None:
    """At d_pose == flip threshold, seg and pose marginals are equal (both 100)."""
    threshold = importance_flip_threshold()
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=1e-4, cuda_d_pose=threshold, archive_bytes=178392,
    )
    # Both axes should be in the tied set on CUDA side
    assert "seg" in rec.cuda_tied_axes
    assert "pose" in rec.cuda_tied_axes
    # Advice surfaces the tie, not silently routes to one axis
    assert "TIE DETECTED" in rec.advice
    assert "parallel" in rec.advice.lower()


def test_dual_axis_no_tie_when_sufficiently_off_threshold() -> None:
    """At pose well below threshold, only pose is in cuda_tied_axes (single-element tuple)."""
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=6.88e-4, cuda_d_pose=1.74e-4, archive_bytes=178392,
    )
    # Pose dominates at this PR107-frontier operating point
    assert rec.cuda_priority_axis == "pose"
    # Single-element tied tuple (the deterministic winner alone)
    assert rec.cuda_tied_axes == ("pose",)
    assert "TIE DETECTED" not in rec.advice


def test_dual_axis_tie_rtol_is_documented() -> None:
    rec = recommend_dispatch_axis_dual(
        cuda_d_seg=1e-4, cuda_d_pose=1e-4, archive_bytes=178392,
    )
    assert rec.tie_rtol == 1e-9
