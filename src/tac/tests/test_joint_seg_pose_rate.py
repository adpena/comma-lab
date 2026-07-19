from __future__ import annotations

import math

import numpy as np
import pytest

from tac.optimization.joint_seg_pose_rate import (
    JointSolveError,
    MarginBandConfig,
    derive_hyperplane_channel_band,
    derive_margin_rgb_band,
    generated_fill_predictor,
    pose_score_derivative,
    solve_interval_frame,
    solve_measured_waterfill,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(camera_h=6, camera_w=8, scorer_h=3, scorer_w=4)


def test_margin_band_is_cached_margin_over_explicit_lipschitz() -> None:
    margin = np.array([[1.0, 2.0]])
    winner = np.array([[0, 1]])
    rival = np.array([[1, 2]])
    pullback = np.array([[[1.0, 2.0, 4.0], [2.0, 1.0, 0.0]]])
    norms = np.array([[2.0, 4.0]])
    got = derive_hyperplane_channel_band(
        margin, winner, rival, pullback, norms,
        MarginBandConfig(scale=1.0, local_lipschitz=1.0, max_rgb_radius=2.0),
    )
    np.testing.assert_allclose(got.feature_flip_distance, [[0.5, 0.5]])
    np.testing.assert_allclose(got.channel_radii[0, 0], [1 / 6, 1 / 12, 1 / 24])
    np.testing.assert_allclose(got.channel_radii[0, 1], [1 / 12, 1 / 6, 2.0])


def test_positive_isotropic_band_is_forbidden() -> None:
    with pytest.raises(JointSolveError, match="isotropic"):
        derive_margin_rgb_band(np.ones((2, 2)), MarginBandConfig(scale=1.0, local_lipschitz=1.0, max_rgb_radius=2.0))


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_margin_band_fails_closed(bad: float) -> None:
    with pytest.raises(JointSolveError):
        derive_margin_rgb_band(np.ones((2, 2)), MarginBandConfig(scale=bad, local_lipschitz=1.0, max_rgb_radius=1.0))


def test_generated_fill_uses_target_only_and_has_camera_geometry() -> None:
    op = _operator()
    target = np.arange(3 * 4 * 3, dtype=np.float64).reshape(3, 4, 3)
    pred = generated_fill_predictor(op, target)
    assert pred.shape == (6, 8, 3)
    assert pred.dtype == np.uint8
    np.testing.assert_allclose(op.apply(pred), np.rint(target), atol=1e-12)


def test_zero_band_joint_frame_is_exact_and_custodied() -> None:
    op = _operator()
    source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    target_num, den = op.apply_numerators(source)
    target = target_num.astype(np.float64) / den
    pred = generated_fill_predictor(op, target)
    solved = solve_interval_frame(op, target_num, den, np.zeros((3, 4)), predictor=pred)
    got_num, got_den = op.apply_numerators(solved.frame)
    assert got_den == den
    np.testing.assert_array_equal(got_num, target_num)
    assert solved.telemetry.maximum_projection_error == 0.0
    assert solved.telemetry.exact_blocks == target_num.size


def test_positive_band_moves_only_inside_interval() -> None:
    op = _operator()
    source = np.random.default_rng(7).integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    target_num, den = op.apply_numerators(source)
    target = target_num.astype(np.float64) / den
    pred = generated_fill_predictor(op, target)
    band = np.full((3, 4), 4.0)
    solved = solve_interval_frame(op, target_num, den, band, predictor=pred)
    delta = np.abs(solved.chosen_numerators - target_num)
    assert np.all(delta <= solved.band_radius_numerators)
    assert solved.telemetry.maximum_projection_error == 0.0


def test_pose_derivative_and_crossover() -> None:
    assert pose_score_derivative(0.0) == math.inf
    assert pose_score_derivative(2.5e-4) == pytest.approx(100.0)


def test_waterfill_refuses_to_force_flat_curves() -> None:
    got = solve_measured_waterfill(
        [{"bytes": 10, "distortion": 0.1}, {"bytes": 20, "distortion": 0.1}],
        [{"bytes": 10, "distortion": 0.01}, {"bytes": 20, "distortion": 0.01}],
    )
    assert got["status"] == "INCONCLUSIVE_FLAT_OR_NOISY"


def test_waterfill_returns_measured_secant_candidate() -> None:
    got = solve_measured_waterfill(
        [{"bytes": 10, "distortion": 0.2}, {"bytes": 20, "distortion": 0.1}],
        [{"bytes": 10, "distortion": 0.001}, {"bytes": 20, "distortion": 0.0005}],
    )
    assert got["status"] == "MEASURED_SECANT_KKT_CANDIDATE"
    assert got["derived_pose_seg_crossover_d_pose"] == 2.5e-4
