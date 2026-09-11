# SPDX-License-Identifier: MIT
"""Behaviour tests for the OBX2 pose law and the pose-weight derivation."""

from __future__ import annotations

import math

import pytest

from tac.canonical_equations import obx2_pose_vs_scorer_plane_rmse_20260911 as law


def test_each_family_reproduces_its_own_measured_points() -> None:
    for structure, rungs, bound in (
        ("smooth", law.SMOOTH_RUNGS, 0.13),
        ("noise", law.NOISE_RUNGS, 0.10),
    ):
        for name, rmse, measured in law.MEASURED_POINTS:
            if name not in rungs:
                continue
            predicted = law.predict_d_pose_by_structure(rmse, structure)
            assert abs(predicted / measured - 1.0) < bound, (structure, name)
    with pytest.raises(ValueError):
        law.predict_d_pose_by_structure(1.0, "mystery")


def test_the_relation_is_not_a_function_of_rmse_alone() -> None:
    # The finding that forced the family split: more RMSE, less d_pose.
    assert law.is_monotonic_in_rmse() is False
    points = {name: (rmse, d_pose) for name, rmse, d_pose in law.MEASURED_POINTS}
    noisy_rmse, noisy_pose = points["sp384_render_noise_4"]
    smooth_rmse, smooth_pose = points["grid_384x512"]
    assert smooth_rmse > noisy_rmse
    assert smooth_pose < noisy_pose
    # and noise is materially worse at equal RMSE
    at_equal = noisy_pose / law.predict_d_pose_by_structure(noisy_rmse, "smooth")
    assert at_equal > 2.0


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
    residuals = equation.predicted_vs_empirical_residual
    assert residuals["pooled_worst_relative_error"] > residuals["smooth_family_worst_relative_error"]
    assert residuals["pooled_worst_relative_error"] > residuals["noise_family_worst_relative_error"]
    assert residuals["smooth_family_worst_relative_error"] < 0.15
    assert equation.domain_of_validity["monotonic_in_rmse"] is False
    assert equation.domain_of_validity["scorer_plane_rmse_range"] == [0.0, 8.67]


def test_pooling_the_families_is_worse_than_fitting_them_apart() -> None:
    assert law.WORST_RELATIVE_ERROR > law.SMOOTH_WORST_RELATIVE_ERROR
    assert law.WORST_RELATIVE_ERROR > law.NOISE_WORST_RELATIVE_ERROR
    assert law.NOISE_EXPONENT > law.SMOOTH_EXPONENT


def test_the_measured_bracket_needs_no_model_and_contains_the_fit() -> None:
    budget = law.pose_budget_at_distortion_gate(0.000111576)
    inside, outside = law.measured_bracket(budget)
    assert inside is not None and outside is not None
    assert inside < outside
    # The pooled bracket mixes families: its upper end is a NOISE rung, which is
    # the wrong reference for a trained generator's structured error.
    assert outside in {rmse for name, rmse, _ in law.MEASURED_POINTS if name in law.NOISE_RUNGS}
    # a budget nothing reaches has no inside rung
    assert law.measured_bracket(1.0e-12)[0] is None


def test_one_and_two_lsb_of_render_noise_both_fail_the_gate() -> None:
    rungs = {name: (rmse, d_pose) for name, rmse, d_pose in law.MEASURED_POINTS}
    for name, seg in (("sp384_render_noise_1", 0.000138109), ("sp384_render_noise_2", 0.000156742)):
        rmse, d_pose = rungs[name]
        assert d_pose > law.pose_budget_at_distortion_gate(seg), name
        assert 100.0 * seg + math.sqrt(10.0 * d_pose) > law.DISTORTION_GATE, name
