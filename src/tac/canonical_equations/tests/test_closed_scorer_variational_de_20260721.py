# SPDX-License-Identifier: MIT
from fractions import Fraction

import numpy as np
import pytest

from tac.canonical_equations.closed_scorer_variational_de_20260721 import (
    RATE_PRICE_EXACT,
    bregman_voronoi_labels,
    build_reachability_equation,
    build_stationarity_equation,
    build_taskspace_equation,
    categorical_bregman_debt,
    closed_scorer_action,
    pose_task_quadratic,
    power_laguerre_labels,
    rate_term_exact,
    reachability_certificate,
    stationarity_residual,
)


def test_exact_rate_price_and_cap_arithmetic() -> None:
    assert Fraction(25, 37_545_489) == RATE_PRICE_EXACT
    assert rate_term_exact(154_600) == Fraction(25 * 154_600, 37_545_489)
    receipt = reachability_certificate()
    assert receipt.status == "UNRESOLVED_REQUIRES_BYTE_CLOSED_WITNESS"
    assert receipt.residual_distortion_budget == pytest.approx(0.04705820584731231)


def test_action_uses_frozen_formula_and_rejects_bad_inputs() -> None:
    assert closed_scorer_action(d_seg=0.001, d_pose=0.0001, archive_bytes=100) == pytest.approx(
        0.1 + np.sqrt(0.001) + 2500 / 37_545_489
    )
    with pytest.raises(ValueError):
        closed_scorer_action(d_seg=-1.0, d_pose=0.0, archive_bytes=0)
    with pytest.raises(TypeError):
        rate_term_exact(True)


def test_laguerre_assignment_reproduces_affine_logits() -> None:
    rng = np.random.default_rng(1234)
    q = rng.normal(size=(50, 4))
    affine_weights = rng.normal(size=(5, 4))
    bias = rng.normal(size=5)
    sites = affine_weights / 2.0
    power_weights = bias + np.sum(sites * sites, axis=1)
    expected = np.argmax(q @ affine_weights.T + bias, axis=1)
    assert np.array_equal(power_laguerre_labels(q, sites, power_weights), expected)


def test_negative_entropy_bregman_voronoi_is_argmax() -> None:
    logits = np.array([[2.0, -1.0, 0.5], [-4.0, 3.0, 2.0]])
    assert np.array_equal(bregman_voronoi_labels(logits), np.argmax(logits, axis=1))
    target = np.array([0, 1])
    debt = categorical_bregman_debt(logits, target)
    expected = np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(2), target]
    assert np.allclose(debt, expected)


def test_pose_is_exact_quadratic_only_in_output_coordinate() -> None:
    target = np.zeros((2, 6))
    xi = np.ones((2, 6))
    assert pose_task_quadratic(xi, target) == 1.0
    with pytest.raises(ValueError):
        pose_task_quadratic(np.ones((2, 5)), np.zeros((2, 5)))


def test_stationarity_separates_rate_price_from_cap_multiplier() -> None:
    result = stationarity_residual(
        relaxed_seg_gradient=[0.0, 0.0],
        pose_debt_gradient=[0.0, 0.0],
        code_length_gradient=[1.0, -1.0],
        d_pose=1.0,
        byte_multiplier=0.5,
    )
    assert result["hard_byte_cap_multiplier_mu_B"] == 0.5
    assert result["objective_rate_price_exact"] == "25/37545489"
    assert result["stationarity_vector"] == pytest.approx(
        [0.5 + 25 / 37_545_489, -0.5 - 25 / 37_545_489]
    )


def test_canonical_builders_bind_measured_and_pending_scopes() -> None:
    task, stationarity, reachability = (
        build_taskspace_equation(),
        build_stationarity_equation(),
        build_reachability_equation(),
    )
    assert task.predicted_vs_empirical_residual["real_frozen_segnet_20_tile_max"] == 0.0
    assert task.empirical_anchors[0].empirical_output["axis"].startswith("[macOS-CPU advisory")
    assert stationarity.domain_of_validity["lambda_star"].endswith("distinct mu_B")
    assert reachability.domain_of_validity["numeric_minimum_status"].startswith("UNRESOLVED")
