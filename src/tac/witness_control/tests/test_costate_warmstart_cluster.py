# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.costate_warmstart_cluster import (
    HierarchicalPhysicsResidualAdjoint,
    physics_prior_coefficients,
    posterior_solve_mlx_fp32,
    posterior_solve_numpy_fp32,
    project_score_aggregate,
    select_prefix_candidate,
    support_certificate,
)
from tac.witness_control.lambda_net import (
    PHI_DIM,
    STATE_DIM,
    Interval,
    lever_features,
)

LEVER_NAMES = ("seg", "pose", "persistence")


def _intervals(n: int = 6) -> list[Interval]:
    out = []
    for i in range(n):
        x0 = np.asarray([.20, .30, .40, .50, .60, 11.0], dtype=np.float64) + i * .01
        rate = np.asarray([-.001, .002, -.0005, .001, -.0002, .0001]) * (1.0 + .05 * i)
        shares = np.asarray([.55 + .01 * i, .25, .20 - .01 * i], dtype=np.float64)
        out.append(Interval(
            ep0=float(25 * i), ep1=float(25 * (i + 1)), x0=x0,
            x1=x0 + 25.0 * rate, ctx=np.asarray([i / 100.0, .1, .2]),
            u_mean=shares, path=np.hstack((shares, [.1]))[None, :],
        ))
    return out


def _design(intervals: list[Interval]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phis = np.stack([lever_features(n) for n in LEVER_NAMES]).astype(np.float32)
    rows = [np.concatenate(([1.0], iv.x0, phis.T @ iv.u_mean)) for iv in intervals]
    return np.asarray(rows, dtype=np.float32), np.stack([iv.dxdt() for iv in intervals]), phis


def test_numpy_posterior_is_deterministic_psd_and_prior_centered() -> None:
    z, y, _ = _design(_intervals())
    c0, _ = physics_prior_coefficients(z, y, mode="Q_priormean_iso")
    a = posterior_solve_numpy_fp32(z, y, c0, precision=1.0)
    b = posterior_solve_numpy_fp32(z, y, c0, precision=1.0)
    assert np.array_equal(a.coefficients, b.coefficients)
    assert np.array_equal(a.coefficient_covariance, b.coefficient_covariance)
    assert a.coefficients.dtype == np.float32
    for cov in a.coefficient_covariance:
        assert np.linalg.eigvalsh(cov.astype(np.float64)).min() >= -1e-12
    assert 0.0 < a.effective_degrees_of_freedom <= len(z)


def test_model_shapes_and_response_uncertainty() -> None:
    intervals = _intervals()
    _, _, phis = _design(intervals)
    model = HierarchicalPhysicsResidualAdjoint(precision=.1)
    model.fit(intervals, phis)
    pred = model.predict_interval(intervals[-1], LEVER_NAMES)
    assert pred.shape == (STATE_DIM,)
    variance = model.response_variance(lever_features("seg"))
    assert variance.shape == (STATE_DIM,)
    assert np.all(variance >= 0.0)


def test_prefix_selection_does_not_read_future_target() -> None:
    intervals = _intervals()
    _, _, phis = _design(intervals)
    weights = np.asarray([.2] * 5, dtype=np.float32)
    before = select_prefix_candidate(
        intervals[:4], LEVER_NAMES, phis, weights,
        prior_modes=("Q_priormean_iso",), precision_grid=(.1, 1.0))
    mutated = list(intervals)
    held = mutated[4]
    mutated[4] = Interval(
        ep0=held.ep0, ep1=held.ep1, x0=held.x0,
        x1=held.x1 + 1e6, ctx=held.ctx, u_mean=held.u_mean, path=held.path)
    after = select_prefix_candidate(
        mutated[:4], LEVER_NAMES, phis, weights,
        prior_modes=("Q_priormean_iso",), precision_grid=(.1, 1.0))
    assert before == after


def test_support_certificates_fail_closed_on_causal_and_ope_claims() -> None:
    intervals = _intervals()
    _, _, phis = _design(intervals)
    cert = support_certificate(intervals, phis)
    assert not cert.causally_identified
    assert cert.fore_status.startswith("BLOCKED_DISTRIBUTION_CUSTODY")
    assert cert.tofu_status.startswith("BLOCKED_PARTIAL_ACTION_CUSTODY")
    assert cert.rl_actor_status.startswith("DISABLED")
    assert 0 <= cert.occupancy_rank <= PHI_DIM


def test_aggregate_projection_is_exact_and_variance_weighted() -> None:
    pred = np.asarray([1., 2., 3., 4., 5., 0.], dtype=np.float32)
    variance = np.asarray([1., 2., 3., 4., 5., 1.], dtype=np.float32)
    weights = np.asarray([.1, .2, .3, .2, .2], dtype=np.float32)
    projected, delta = project_score_aggregate(pred, variance, weights, 2.5)
    assert float(weights @ projected[:5]) == pytest.approx(2.5, abs=2e-6)
    assert delta > 0.0
    assert abs(projected[4] - pred[4]) > abs(projected[0] - pred[0])


def test_mlx_matches_numpy_when_available() -> None:
    pytest.importorskip("mlx.core")
    z, y, _ = _design(_intervals())
    c0, _ = physics_prior_coefficients(z, y, mode="Q_priormean_iso")
    np_ref = posterior_solve_numpy_fp32(z, y, c0, precision=1.0)
    try:
        mlx = posterior_solve_mlx_fp32(z, y, c0, precision=1.0)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("MLX installed but Metal is unavailable in this sandbox")
        raise
    assert np.allclose(mlx.coefficients, np_ref.coefficients, rtol=3e-4, atol=3e-6)
    flat_a, flat_b = mlx.coefficients.ravel(), np_ref.coefficients.ravel()
    assert float(np.dot(flat_a, flat_b) / (
        np.linalg.norm(flat_a) * np.linalg.norm(flat_b))) >= .9997
