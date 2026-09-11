# SPDX-License-Identifier: MIT
"""Behaviour tests for the OBX2 pose law and the pose-weight derivation."""

from __future__ import annotations

import math

import pytest

from tac.canonical_equations import obx2_pose_vs_scorer_plane_rmse_20260911 as law


def test_law_reproduces_every_measured_n600_point_within_six_percent() -> None:
    for name, rmse, measured in law.MEASURED_POINTS:
        predicted = law.predict_d_pose(rmse)
        assert predicted > 0.0, name
        assert abs(predicted / measured - 1.0) < 0.13, name


def test_law_floors_at_the_pointer_own_d_pose_and_rises_monotonically() -> None:
    assert law.predict_d_pose(0.0) == law.POSE_FLOOR
    values = [law.predict_d_pose(r) for r in (0.0, 0.1, 1.0, 5.0, 10.0)]
    assert values == sorted(values)
    with pytest.raises(ValueError):
        law.predict_d_pose(-1.0)
    with pytest.raises(ValueError):
        law.predict_d_pose(float("nan"))


def test_inverting_the_law_round_trips() -> None:
    for rmse in (0.05, 0.24, 1.0, 6.0):
        assert law.admissible_scorer_plane_rmse(law.predict_d_pose(rmse)) == pytest.approx(rmse, rel=1e-9)
    assert law.admissible_scorer_plane_rmse(law.POSE_FLOOR) == 0.0
    assert law.admissible_scorer_plane_rmse(0.0) == 0.0


def test_pose_budget_shrinks_as_seg_eats_the_gate() -> None:
    # sqrt(10 * budget) must exactly fill what d_seg leaves under the gate.
    for d_seg in (0.0, 1.0e-4, 3.0e-4):
        budget = law.pose_budget_at_distortion_gate(d_seg)
        assert 100.0 * d_seg + math.sqrt(10.0 * budget) == pytest.approx(law.DISTORTION_GATE)
    assert law.pose_budget_at_distortion_gate(1.0) == 0.0


def test_the_gate_demands_a_near_lossless_scorer_plane() -> None:
    budget = law.pose_budget_at_distortion_gate(0.000111576)
    admissible = law.admissible_scorer_plane_rmse(budget)
    assert admissible < 0.30
    assert 100.0 * admissible / law.SCORER_PLANE_RMSE_FULL_SCALE < 0.1


def test_pose_weight_is_the_exact_contest_derivative() -> None:
    for d_pose in (1.0e-5, 1.1e-3, 0.05, 0.5):
        analytic = law.pose_weight_at_operating_point(d_pose)
        step = d_pose * 1.0e-6
        numeric = (math.sqrt(10.0 * (d_pose + step)) - math.sqrt(10.0 * (d_pose - step))) / (2.0 * step)
        assert analytic == pytest.approx(numeric, rel=1e-5)
    with pytest.raises(ValueError):
        law.pose_weight_at_operating_point(0.0)
    with pytest.raises(ValueError):
        law.pose_weight_at_operating_point(-1.0)


def test_pose_weight_falls_as_the_operating_point_worsens() -> None:
    assert law.pose_weight_at_operating_point(1.0e-3) > law.pose_weight_at_operating_point(1.0e-1)


def test_derivation_pins_every_input_and_refuses_a_wrong_denominator() -> None:
    derived = law.derive_pose_weight(
        d_pose=0.02,
        d_seg=0.001,
        source_artifact="x/CHECKPOINT_SCORE.json",
        source_sha256="a" * 64,
        receiver="torch",
        pair_denominator=600,
    )
    assert derived["law_ref"] == law.EQUATION_ID
    assert derived["pose_weight"] == pytest.approx(5.0 / math.sqrt(0.2))
    assert derived["operating_point_distortion"] == pytest.approx(0.1 + math.sqrt(0.2))
    assert derived["source_sha256"] == "a" * 64
    assert derived["receiver"] == "torch"
    assert derived["score_claim"] is False
    with pytest.raises(ValueError):
        law.derive_pose_weight(
            d_pose=0.02, d_seg=0.001, source_artifact="x", source_sha256="a", receiver="torch",
            pair_denominator=96,
        )
    with pytest.raises(ValueError):
        law.derive_pose_weight(
            d_pose=0.02, d_seg=0.001, source_artifact="x", source_sha256="a", receiver="mps",
            pair_denominator=600,
        )


def test_canonical_equation_builds_with_every_measured_anchor() -> None:
    equation = law.build_obx2_pose_vs_scorer_plane_rmse_v1()
    assert equation.equation_id == law.EQUATION_ID
    assert len(equation.empirical_anchors) == len(law.MEASURED_POINTS)
    worst = equation.predicted_vs_empirical_residual["worst_relative_error_over_six_n600_points"]
    assert 0.0 < worst < 0.13
    assert equation.domain_of_validity["scorer_plane_rmse_range"] == [0.0, 8.67]


def test_the_small_error_regime_returns_the_same_law_as_the_global_fit() -> None:
    # The gate lives below spRMSE 1; a separate fit there must not be a different law.
    assert abs(law.SMALL_REGIME_EXPONENT / law.POWER_LAW_EXPONENT - 1.0) < 0.05
    small = [p for p in law.MEASURED_POINTS if 0.0 < p[1] < law.SMALL_REGIME_MAX_RMSE]
    assert len(small) >= 3
    for _, rmse, measured in small:
        local = law.POSE_FLOOR + law.SMALL_REGIME_COEFFICIENT * rmse**law.SMALL_REGIME_EXPONENT
        assert abs(local / measured - 1.0) < 0.15


def test_one_lsb_of_render_noise_already_fails_the_gate() -> None:
    rung = {name: (rmse, d_pose) for name, rmse, d_pose in law.MEASURED_POINTS}["sp384_render_noise_1"]
    rmse, d_pose = rung
    assert d_pose > law.pose_budget_at_distortion_gate(0.000138109)
    assert rmse > law.admissible_scorer_plane_rmse(law.pose_budget_at_distortion_gate(0.000111576))
