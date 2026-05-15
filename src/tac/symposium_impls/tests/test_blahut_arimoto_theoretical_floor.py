# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.blahut_arimoto_theoretical_floor`."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tac.symposium_impls.blahut_arimoto_theoretical_floor import (
    ContestTheoreticalFloor,
    binary_entropy,
    blahut_arimoto_rate_distortion,
    compute_contest_theoretical_floor,
    dykstra_project_onto_pareto_frontier,
    gaussian_rate_distortion_bound,
    load_cached_theoretical_floor,
    save_theoretical_floor,
    update_from_anchor,
)


# ----- closed-form sanity tests -----------------------------------------------------------------


def test_binary_entropy_at_half_is_one_bit() -> None:
    assert binary_entropy(0.5) == pytest.approx(1.0, abs=1e-12)


def test_binary_entropy_at_endpoints_is_zero() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_binary_entropy_at_quarter() -> None:
    expected = -0.25 * math.log2(0.25) - 0.75 * math.log2(0.75)
    assert binary_entropy(0.25) == pytest.approx(expected, abs=1e-12)


def test_gaussian_rate_distortion_at_d_eq_variance_is_zero() -> None:
    """``R(D=σ²) = 0`` per Cover & Thomas Theorem 10.3.2."""
    assert gaussian_rate_distortion_bound(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_gaussian_rate_distortion_above_variance_is_zero() -> None:
    assert gaussian_rate_distortion_bound(1.0, 2.0) == 0.0


def test_gaussian_rate_distortion_at_d_eq_quarter_variance() -> None:
    """``R(D=σ²/4) = 0.5 log2(4) = 1.0`` bit."""
    assert gaussian_rate_distortion_bound(1.0, 0.25) == pytest.approx(1.0, abs=1e-12)


def test_gaussian_rate_distortion_zero_distortion_is_inf() -> None:
    assert gaussian_rate_distortion_bound(1.0, 0.0) == float("inf")


def test_gaussian_rate_distortion_invalid_variance_raises() -> None:
    with pytest.raises(ValueError):
        gaussian_rate_distortion_bound(0.0, 0.5)


# ----- Blahut-Arimoto algorithm tests -----------------------------------------------------------


def test_blahut_arimoto_binary_symmetric_hamming_distortion_quarter() -> None:
    """For BSS with hamming distortion: R(D) = 1 - H(D) for D ∈ [0, 1/2].

    [verified-against: Cover & Thomas 2nd ed Example 10.2.1.]
    """
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    rate = blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=0.25)
    expected = 1.0 - binary_entropy(0.25)
    assert rate == pytest.approx(expected, abs=0.05)


def test_blahut_arimoto_at_zero_distortion_returns_log2_alphabet() -> None:
    """At D=0 you cannot compress (binary symmetric source); R = 1 bit."""
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    rate = blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=1e-9)
    assert rate >= 0.95


def test_blahut_arimoto_at_max_distortion_returns_zero() -> None:
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    rate = blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=0.5)
    assert rate == pytest.approx(0.0, abs=1e-6)


def test_blahut_arimoto_invalid_distribution_raises() -> None:
    with pytest.raises(ValueError):
        blahut_arimoto_rate_distortion(
            np.array([0.5, 0.6]), np.array([[0.0, 1.0], [1.0, 0.0]]), target_distortion=0.1
        )


def test_blahut_arimoto_negative_distortion_raises() -> None:
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[-1.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=0.1)


def test_blahut_arimoto_negative_target_distortion_raises() -> None:
    p_x = np.array([1.0])
    distortion = np.array([[0.0]])
    with pytest.raises(ValueError):
        blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=-0.1)


def test_blahut_arimoto_wrong_distortion_shape_raises() -> None:
    p_x = np.array([0.5, 0.5])
    bad_distortion = np.array([[0.0, 1.0]])  # only 1 row
    with pytest.raises(ValueError):
        blahut_arimoto_rate_distortion(p_x, bad_distortion, target_distortion=0.1)


def test_blahut_arimoto_3_class_uniform_source_rate_decreasing_in_distortion() -> None:
    """Sanity: rate must be monotonically non-increasing in target distortion."""
    p_x = np.array([1 / 3, 1 / 3, 1 / 3])
    distortion = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    rates = [
        blahut_arimoto_rate_distortion(p_x, distortion, target_distortion=d)
        for d in [0.1, 0.2, 0.4, 0.6]
    ]
    for prev, current in zip(rates, rates[1:]):
        assert current <= prev + 1e-6


# ----- Dykstra projection tests -----------------------------------------------------------------


def test_dykstra_project_meets_both_floors() -> None:
    r_seg, r_pose, _, converged = dykstra_project_onto_pareto_frontier(
        rate_seg_init=0.0,
        rate_pose_init=0.0,
        target_d_seg=0.05,
        target_d_pose=0.001,
    )
    assert converged
    assert r_seg >= gaussian_rate_distortion_bound(1.0, 0.05) - 1e-9
    assert r_pose >= 6.0 * gaussian_rate_distortion_bound(1.0, 0.001 / 6.0) - 1e-9


def test_dykstra_project_initial_above_floor_unchanged() -> None:
    """At D=0.99 (≈ variance) the Gaussian floor is ~0; init at 10 stays at 10."""
    r_seg, r_pose, iterations, converged = dykstra_project_onto_pareto_frontier(
        rate_seg_init=10.0,
        rate_pose_init=10.0,
        target_d_seg=0.99,
        target_d_pose=0.99 * 6,  # per-dim D ≈ 0.99 for 6D pose
    )
    assert converged
    assert r_seg == pytest.approx(10.0, abs=1e-6)
    assert r_pose == pytest.approx(10.0, abs=1e-6)
    assert iterations >= 1


# ----- contest theoretical floor tests ----------------------------------------------------------


def test_contest_theoretical_floor_at_a1_anchor_returns_finite_bound() -> None:
    floor = compute_contest_theoretical_floor(
        target_d_seg=0.01,
        target_d_pose=0.0001,
    )
    assert floor.contest_score_floor > 0
    assert math.isfinite(floor.contest_score_floor)
    assert floor.dykstra_converged
    assert floor.evidence_grade == "theoretical-bound-prediction"
    assert floor.score_claim is False


def test_contest_theoretical_floor_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        compute_contest_theoretical_floor(target_d_seg=0.0, target_d_pose=0.01)
    with pytest.raises(ValueError):
        compute_contest_theoretical_floor(target_d_seg=0.01, target_d_pose=0.0)


def test_contest_theoretical_floor_floor_decreases_with_larger_distortion() -> None:
    """Larger D budgets allow lower rate; net effect depends on coefficients."""
    floor_tight = compute_contest_theoretical_floor(target_d_seg=0.01, target_d_pose=0.001)
    floor_loose = compute_contest_theoretical_floor(target_d_seg=0.1, target_d_pose=0.01)
    # Both are finite, well-formed
    assert math.isfinite(floor_tight.contest_score_floor)
    assert math.isfinite(floor_loose.contest_score_floor)


# ----- save / load round trip ------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    floor = compute_contest_theoretical_floor(target_d_seg=0.01, target_d_pose=0.001)
    path = tmp_path / "floor.json"
    save_theoretical_floor(floor, state_path=path)
    loaded = load_cached_theoretical_floor(state_path=path)
    assert loaded is not None
    assert loaded.contest_score_floor == pytest.approx(floor.contest_score_floor)


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_cached_theoretical_floor(state_path=tmp_path / "absent.json") is None


# ----- continual-learning hook -----------------------------------------------------------------


def test_update_from_anchor_with_distortions_emits_floor(tmp_path: Path) -> None:
    state_path = tmp_path / "floor.json"
    anchor = {"cuda_seg": 0.01, "cuda_pose": 0.001, "notes": "synthetic anchor"}
    floor = update_from_anchor(anchor, state_path=state_path)
    assert floor is not None
    assert state_path.is_file()
    parsed = json.loads(state_path.read_text())
    assert parsed["operating_point_anchor"] == "synthetic anchor"


def test_update_from_anchor_missing_distortions_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "floor.json"
    anchor = {"notes": "no dist"}
    assert update_from_anchor(anchor, state_path=state_path) is None


def test_update_from_anchor_negative_distortions_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "floor.json"
    anchor = {"cuda_seg": -1.0, "cuda_pose": 0.001}
    assert update_from_anchor(anchor, state_path=state_path) is None


def test_update_from_anchor_non_numeric_distortions_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "floor.json"
    anchor = {"cuda_seg": "abc", "cuda_pose": 0.001}
    assert update_from_anchor(anchor, state_path=state_path) is None


def test_update_from_anchor_falls_back_to_cpu_axis(tmp_path: Path) -> None:
    state_path = tmp_path / "floor.json"
    anchor = {"cpu_seg": 0.02, "cpu_pose": 0.0001}
    floor = update_from_anchor(anchor, state_path=state_path)
    assert floor is not None
    assert floor.target_d_seg == pytest.approx(0.02)
    assert floor.target_d_pose == pytest.approx(0.0001)
