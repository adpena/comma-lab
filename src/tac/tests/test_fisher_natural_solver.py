from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tac.canonical_equations.categorical_fisher_natural_trust_region_20260715 import (
    build_categorical_fisher_natural_trust_region_solve_v1,
)
from tac.information_geometry.fisher_natural_solver import (
    METRIC_ID,
    categorical_fisher_quadratic,
    centre_cotangent,
    helmert_zero_sum_basis,
    solve_categorical_fisher_natural_step_numpy_fp32,
)
from tac.witness_dsl.fisher_natural_solver_policy import (
    canonical_fisher_natural_solver_policy,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    p = np.array(
        [[0.50, 0.20, 0.15, 0.10, 0.05], [0.05, 0.10, 0.20, 0.25, 0.40]],
        dtype=np.float32,
    )
    g = centre_cotangent(
        np.array([[1.2, -0.3, 0.7, -0.4, -1.1], [-0.5, 0.8, -0.2, 1.0, -0.7]])
    )
    return p, g


def test_helmert_basis_is_the_zero_sum_quotient() -> None:
    q = helmert_zero_sum_basis(5)
    np.testing.assert_allclose(q.T @ q, np.eye(4), rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(np.ones(5) @ q, np.zeros(4), rtol=0.0, atol=2e-15)


def test_solver_satisfies_projected_normal_equation_and_trust_radius() -> None:
    p, g = _fixture()
    receipt = solve_categorical_fisher_natural_step_numpy_fp32(
        p,
        g,
        delta=2e-3,
        delta_convention="delta_kl",
        damping=1e-4,
    )
    assert receipt.step.dtype == np.float32
    assert receipt.step.shape == p.shape
    assert np.max(receipt.projected_residual_linf) < 2e-12
    assert np.max(receipt.step_gauge_residual) < 2e-6
    assert np.all(receipt.fisher_quadratic_after <= receipt.delta_quad)
    np.testing.assert_allclose(
        receipt.fisher_quadratic_after,
        categorical_fisher_quadratic(p, receipt.step),
        rtol=0.0,
        atol=5e-11,
    )


def test_zero_damping_matches_independent_kkt_solution() -> None:
    p, g = _fixture()
    receipt = solve_categorical_fisher_natural_step_numpy_fp32(
        p[:1], g[:1], delta=100.0, delta_convention="delta_quad", damping=0.0
    )
    h = np.diag(p[0].astype(np.float64)) - np.outer(p[0], p[0])
    kkt = np.block([[h, np.ones((5, 1))], [np.ones((1, 5)), np.zeros((1, 1))]])
    expected = np.linalg.solve(kkt, np.concatenate([-g[0], [0.0]]))[:5]
    np.testing.assert_allclose(receipt.step[0], expected, rtol=2e-6, atol=2e-6)


def test_gauge_projection_is_explicit_and_fail_closed_by_default() -> None:
    p, g = _fixture()
    bad = g.copy()
    bad[0, 0] += 0.2
    with pytest.raises(ValueError, match="quotient-compatible"):
        solve_categorical_fisher_natural_step_numpy_fp32(p, bad, delta=1e-3)
    projected = solve_categorical_fisher_natural_step_numpy_fp32(
        p, bad, delta=1e-3, project_gauge=True
    )
    assert projected.cotangent_gauge_residual[0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("probability", "cotangent", "message"),
    [
        ([0.6, 0.5], [1.0, -1.0], "sum to one"),
        ([1.0, 0.0], [1.0, -1.0], "lie in"),
        ([0.5, 0.5], [1.0, np.nan], "finite"),
    ],
)
def test_invalid_inputs_fail_closed(probability, cotangent, message) -> None:
    with pytest.raises(ValueError, match=message):
        solve_categorical_fisher_natural_step_numpy_fp32(
            probability, cotangent, delta=1e-3
        )


def test_equation_and_dsl_policy_bind_one_metric_without_fake_argv() -> None:
    equation = build_categorical_fisher_natural_trust_region_solve_v1()
    policy = canonical_fisher_natural_solver_policy()
    assert equation.equation_id == policy.equation_id
    assert policy.metric_id == METRIC_ID
    assert policy.flags() == {}
    assert policy.to_dict()["trainer_argv"] == []
    assert policy.activation == "built_not_activated_measurement_owed"


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="MLX is not installed")
def test_mlx_solver_meets_nonlowerable_parity() -> None:
    from tac.information_geometry.fisher_natural_solver_mlx import (
        solve_categorical_fisher_natural_step_mlx,
    )

    p, g = _fixture()
    try:
        receipt = solve_categorical_fisher_natural_step_mlx(
            p, g, delta=3e-3, damping=1e-4
        )
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("MLX installed but Metal is unavailable in this sandbox")
        raise
    assert receipt.parity["passed"] is True
    assert receipt.parity["step_correlation"] >= 0.9997
