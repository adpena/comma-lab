# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tac.optimization.ripo_fisher_trust_region import (
    categorical_exact_kl,
    categorical_fisher_quadratic,
    centre_logits,
    clip_categorical_fisher_step_numpy_fp32,
    convert_delta_budget,
    winner_rival_curvature,
    winner_rival_radius,
)


def _clip(
    probabilities: np.ndarray,
    step: np.ndarray,
    *,
    delta: float = 0.01,
    convention: str = "delta_kl",
    mode: str = "local_directional",
):
    return clip_categorical_fisher_step_numpy_fp32(
        probabilities,
        step,
        delta=delta,
        delta_convention=convention,
        mode=mode,
        tolerance=1e-10,
    )


def test_identity_below_bound_and_exact_local_contraction() -> None:
    p = np.array([[0.4, 0.3, 0.2, 0.08, 0.02]], dtype=np.float32)
    small = np.array([[0.01, -0.01, 0.0, 0.0, 0.0]], dtype=np.float32)
    identity = _clip(p, small, delta=0.1)
    np.testing.assert_allclose(identity.centred_output, centre_logits(small), atol=2e-8)
    np.testing.assert_array_equal(identity.clipped, [False])

    large = 100.0 * small
    contracted = _clip(p, large, delta=0.001)
    expected = np.sqrt(contracted.delta_quad / contracted.q_before)
    assert contracted.alpha[0] == pytest.approx(expected[0], rel=2e-5)
    assert contracted.alpha[0] < 1.0
    assert contracted.q_after[0] <= contracted.delta_quad


@pytest.mark.parametrize("mode", ["local_directional", "local_euclidean_ball"])
def test_local_fisher_modes_obey_post_cast_quadratic_bound(mode: str) -> None:
    rng = np.random.default_rng(20260714)
    p = rng.dirichlet(np.ones(5), size=128).astype(np.float32)
    step = rng.normal(size=(128, 5)).astype(np.float32)
    result = _clip(p, step, delta=3e-4, mode=mode)
    assert np.all(result.q_after <= result.delta_quad + 1e-13)
    assert result.centred_output.dtype == np.float32
    np.testing.assert_allclose(
        np.mean(result.centred_output, axis=-1),
        0.0,
        atol=2e-8,
    )


def test_exact_kl_mode_obeys_post_cast_finite_bound() -> None:
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.array([0.1, 0.2, 0.5, 1.0, 2.0]), size=96).astype(np.float32)
    step = (3.0 * rng.normal(size=(96, 5))).astype(np.float32)
    result = _clip(p, step, delta=2e-5, mode="exact_kl")
    assert np.all(result.exact_kl_after <= result.delta_kl + 1e-13)
    assert np.all(result.alpha >= 0.0)
    assert np.all(result.alpha <= 1.0)


def test_gauge_invariance_constant_null_and_quotient_direction() -> None:
    p = np.array(
        [[0.5, 0.2, 0.15, 0.1, 0.05], [0.1, 0.2, 0.3, 0.15, 0.25]],
        dtype=np.float32,
    )
    step = np.array([[4, -2, 1, 3, -7], [-1, 5, 2, -3, 8]], dtype=np.float32)
    first = _clip(p, step, delta=1e-4)
    second = _clip(p, step + np.array([[123.0], [-91.0]], dtype=np.float32), delta=1e-4)
    np.testing.assert_allclose(first.centred_output, second.centred_output, atol=3e-7)

    centred = centre_logits(step)
    for row in range(2):
        nonzero = np.abs(centred[row]) > 1e-7
        ratios = first.centred_output[row, nonzero] / centred[row, nonzero]
        np.testing.assert_allclose(ratios, first.alpha[row], rtol=3e-5, atol=3e-7)

    constant = _clip(p, np.full_like(p, 19.0), delta=0.0, mode="exact_kl")
    np.testing.assert_array_equal(constant.centred_output, np.zeros_like(p))
    np.testing.assert_array_equal(constant.alpha, np.ones(2))
    np.testing.assert_array_equal(constant.q_after, np.zeros(2))


def test_eq10_eq11_factor_two_equivalence() -> None:
    p = np.array([[0.7, 0.1, 0.08, 0.07, 0.05]], dtype=np.float32)
    step = np.array([[2.0, -1.0, 0.5, -0.25, 0.1]], dtype=np.float32)
    eq10 = _clip(p, step, delta=0.003, convention="delta_kl")
    eq11 = _clip(p, step, delta=0.006, convention="delta_quad")
    np.testing.assert_array_equal(eq10.centred_output, eq11.centred_output)
    assert convert_delta_budget(0.003, "delta_kl") == (0.003, 0.006)
    assert convert_delta_budget(0.006, "delta_quad") == (0.003, 0.006)


def test_exact_k2_reduction_and_k5_tail_counterexample() -> None:
    p2 = np.array([[0.8, 0.2]], dtype=np.float64)
    step2 = np.array([[1.5, -1.5]], dtype=np.float64)
    q2 = categorical_fisher_quadratic(p2, step2)[0]
    curvature2 = winner_rival_curvature(p2)[0]
    assert q2 == pytest.approx(0.25 * curvature2 * (step2[0, 0] - step2[0, 1]) ** 2)

    p5 = np.array([[0.45, 0.35, 0.1, 0.07, 0.03]], dtype=np.float64)
    step5 = np.array([[1.5, -1.5, 4.0, -3.0, 2.0]], dtype=np.float64)
    q5 = categorical_fisher_quadratic(p5, step5)[0]
    curvature5 = winner_rival_curvature(p5)[0]
    top_two = 0.25 * curvature5 * (step5[0, 0] - step5[0, 1]) ** 2
    assert abs(q5 - top_two) > 0.1


def test_asymmetric_finite_kl_roots_preserve_sign_and_differ() -> None:
    p = np.array([[0.8, 0.2]], dtype=np.float32)
    positive = _clip(p, np.array([[3.0, -1.0]], dtype=np.float32), delta=0.01, mode="exact_kl")
    negative = _clip(p, np.array([[-3.0, 1.0]], dtype=np.float32), delta=0.01, mode="exact_kl")
    assert positive.alpha[0] != pytest.approx(negative.alpha[0], rel=1e-3)
    assert np.sign(positive.centred_output[0, 0]) == 1
    assert np.sign(negative.centred_output[0, 0]) == -1
    assert positive.exact_kl_after[0] <= 0.01
    assert negative.exact_kl_after[0] <= 0.01


def test_near_tie_vs_confident_regression_reverses_p1_heuristic() -> None:
    p = np.array(
        [[0.45, 0.45, 0.05, 0.03, 0.02], [0.98, 0.01, 0.005, 0.003, 0.002]],
        dtype=np.float64,
    )
    radius = winner_rival_radius(p, delta=0.01, delta_convention="delta_kl")
    p1_heuristic = np.sqrt(0.01 / p[:, 0])
    assert radius[1] > radius[0]
    assert p1_heuristic[1] < p1_heuristic[0]


def test_near_zero_probabilities_zero_update_and_near_bound() -> None:
    p = np.array([[1.0 - 4e-12, 1e-12, 1e-12, 1e-12, 1e-12]], dtype=np.float64)
    zero = _clip(p, np.zeros((1, 5), dtype=np.float64), delta=0.0)
    np.testing.assert_array_equal(zero.centred_output, np.zeros((1, 5), dtype=np.float32))
    step = np.array([[0.0, 1.0, -1.0, 2.0, -2.0]], dtype=np.float64)
    q = categorical_fisher_quadratic(p, centre_logits(step))[0]
    boundary = _clip(p, step, delta=0.5 * q, convention="delta_kl")
    assert boundary.q_after[0] <= q + 1e-13
    assert boundary.alpha[0] == pytest.approx(1.0, abs=2e-6)


def test_tiny_exact_kl_is_cancellation_safe_and_delta_zero_is_strict_null() -> None:
    p = np.array([[0.2, 0.3, 0.1, 0.15, 0.25]], dtype=np.float64)
    step = np.array([[1e-8, -1e-8, 2e-8, -2e-8, 0.0]], dtype=np.float64)
    q = categorical_fisher_quadratic(p, step)[0]
    kl = categorical_exact_kl(p, step)[0]
    assert kl > 0.0
    assert kl == pytest.approx(0.5 * q, rel=1e-7)

    tiny_budget = _clip(p, step, delta=1e-20, mode="exact_kl")
    assert 0.0 < tiny_budget.alpha[0] < 1.0
    assert 0.0 < tiny_budget.exact_kl_after[0] <= 1e-20

    zero_budget = _clip(p, step, delta=0.0, mode="exact_kl")
    assert zero_budget.alpha[0] == 0.0
    np.testing.assert_array_equal(zero_budget.centred_output, np.zeros((1, 5), dtype=np.float32))

    gauge_null = _clip(p, np.full((1, 5), 1e-8), delta=0.0, mode="exact_kl")
    assert gauge_null.alpha[0] == 1.0
    assert gauge_null.exact_kl_after[0] == 0.0


def test_simplex_acceptance_is_not_weakened_by_caller_tolerance() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        clip_categorical_fisher_step_numpy_fp32(
            np.array([[0.2, 0.2, 0.2, 0.2, 0.19]], dtype=np.float64),
            np.zeros((1, 5), dtype=np.float64),
            delta=1.0,
            delta_convention="delta_kl",
            mode="exact_kl",
            tolerance=0.1,
        )


def test_exact_kl_tolerance_controls_conservative_bracket_precision() -> None:
    p = np.array([[0.72, 0.12, 0.08, 0.05, 0.03]], dtype=np.float64)
    step = np.array([[2.0, -1.0, 0.5, -0.75, -0.75]], dtype=np.float64)
    coarse = clip_categorical_fisher_step_numpy_fp32(
        p,
        step,
        delta=0.01,
        delta_convention="delta_kl",
        mode="exact_kl",
        tolerance=0.1,
    )
    fine = clip_categorical_fisher_step_numpy_fp32(
        p,
        step,
        delta=0.01,
        delta_convention="delta_kl",
        mode="exact_kl",
        tolerance=1e-12,
    )
    assert coarse.alpha[0] < fine.alpha[0]
    assert coarse.exact_kl_after[0] <= 0.01
    assert fine.exact_kl_after[0] <= 0.01


def test_uniform_control_is_explicit_and_probability_independent() -> None:
    p = np.array(
        [[0.2] * 5, [0.96, 0.01, 0.01, 0.01, 0.01]],
        dtype=np.float32,
    )
    step = np.broadcast_to(np.array([1, -1, 2, -2, 0], dtype=np.float32), p.shape)
    result = _clip(p, step, delta=0.001, mode="uniform_l2_control")
    assert result.mode == "uniform_l2_control"
    assert result.authority.startswith("numpy-fp32 categorical output-space")
    assert result.alpha[0] == pytest.approx(result.alpha[1])


@pytest.mark.parametrize(
    ("probabilities", "step", "delta", "convention", "mode", "tolerance"),
    [
        ([0.4, 0.4, 0.1], [0.0, 0.0, 0.0], 0.1, "delta_kl", "exact_kl", 1e-9),
        ([0.5, -0.1, 0.6], [0.0, 0.0, 0.0], 0.1, "delta_kl", "exact_kl", 1e-9),
        ([0.5, 0.5, 0.0], [0.0, 0.0, 0.0], 0.1, "delta_kl", "exact_kl", 1e-9),
        ([0.5, 0.5], [0.0, float("nan")], 0.1, "delta_kl", "exact_kl", 1e-9),
        ([0.5, 0.5], [0.0, 0.0, 0.0], -0.1, "delta_kl", "exact_kl", 1e-9),
        ([0.5, 0.5], [0.0, 0.0, 0.0], 0.1, "hidden", "exact_kl", 1e-9),
        ([0.5, 0.5], [0.0, 0.0, 0.0], 0.1, "delta_kl", "scalar_p1", 1e-9),
        ([0.5, 0.5], [0.0, 0.0, 0.0], 0.1, "delta_kl", "exact_kl", 0.0),
    ],
)
def test_invalid_inputs_fail_closed(
    probabilities: object,
    step: object,
    delta: float,
    convention: str,
    mode: str,
    tolerance: float,
) -> None:
    with pytest.raises(ValueError):
        clip_categorical_fisher_step_numpy_fp32(
            probabilities,
            step,
            delta=delta,
            delta_convention=convention,
            mode=mode,
            tolerance=tolerance,
        )


def test_exact_kl_formula_matches_direct_positive_probability_form() -> None:
    p = np.array([[0.7, 0.2, 0.08, 0.015, 0.005]], dtype=np.float64)
    step = centre_logits(np.array([[2.0, -1.0, 0.1, 0.4, -3.0]]))
    q = p * np.exp(step)
    q /= q.sum(axis=-1, keepdims=True)
    direct = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    np.testing.assert_allclose(categorical_exact_kl(p, step), direct, rtol=1e-13, atol=1e-14)


@pytest.mark.skipif(importlib.util.find_spec("mlx.core") is None, reason="MLX not installed")
@pytest.mark.parametrize("mode", ["local_directional", "exact_kl"])
def test_numpy_mlx_parity_when_available(mode: str) -> None:
    from tac.optimization.ripo_fisher_trust_region_mlx import (
        clip_categorical_fisher_step_mlx,
    )

    rng = np.random.default_rng(73)
    p = rng.dirichlet(np.ones(5), size=64).astype(np.float32)
    step = rng.normal(size=(64, 5)).astype(np.float32)
    try:
        result = clip_categorical_fisher_step_mlx(
            p,
            step,
            delta=1e-3,
            delta_convention="delta_kl",
            mode=mode,
            tolerance=1e-10,
        )
    except RuntimeError as error:
        if "No Metal device available" in str(error):
            pytest.skip("MLX is installed but no Metal device is exposed")
        raise
    assert result.parity["passed"] is True
    assert result.parity["q_correlation"] >= 0.9997
    assert result.parity["exact_kl_correlation"] >= 0.9997
    assert result.parity["output_correlation"] >= 0.9997
    if mode == "local_directional":
        assert result.parity["update_backend"] == "mlx_float32_local_directional"
        assert np.any(np.asarray(result.alpha) < 1.0)


def test_mlx_parity_floor_is_non_lowerable_before_runtime_import() -> None:
    from tac.optimization.ripo_fisher_trust_region_mlx import (
        clip_categorical_fisher_step_mlx,
    )

    with pytest.raises(ValueError, match="non-lowerable"):
        clip_categorical_fisher_step_mlx(
            np.array([[0.2] * 5], dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            delta=0.1,
            delta_convention="delta_kl",
            mode="local_directional",
            tolerance=1e-10,
            minimum_parity=0.9996,
        )
