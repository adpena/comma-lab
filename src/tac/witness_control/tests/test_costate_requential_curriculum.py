from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.costate_requential_curriculum import (
    MAX_REPLAY_WEIGHT,
    _capped_mass_allocation,
    gaussian_disagreement_bits,
    weighted_posterior_mlx_fp32,
)
from tac.witness_control.costate_warmstart_cluster import posterior_solve_numpy_fp32


def test_gaussian_disagreement_bits_matches_closed_form() -> None:
    got = gaussian_disagreement_bits(delta=2.0, variance=4.0)
    assert got == pytest.approx(1.0 / (2.0 * np.log(2.0)))


def test_disagreement_allocation_conserves_mass_and_coverage() -> None:
    bits = np.asarray([0.0, 0.0, 0.25, 2.0, 1000.0], dtype=np.float32)
    weights = _capped_mass_allocation(bits)
    assert float(weights.sum()) == pytest.approx(float(len(bits)), abs=2e-6)
    assert np.all(weights >= 0.5)
    assert np.all(weights <= MAX_REPLAY_WEIGHT + 1e-6)
    assert weights[-1] > weights[-2] > weights[0]


def test_uniform_disagreement_is_uniform() -> None:
    np.testing.assert_array_equal(
        _capped_mass_allocation(np.zeros(6, dtype=np.float32)),
        np.ones(6, dtype=np.float32),
    )


def test_weighted_mlx_matches_numpy_when_metal_is_available() -> None:
    mx = pytest.importorskip("mlx.core")
    try:
        mx.eval(mx.array([0.0]))
    except RuntimeError as exc:
        pytest.skip(f"MLX Metal unavailable: {exc}")
    z = np.zeros((3, 17), dtype=np.float32)
    z[:, 0] = 1.0
    z[:, 1] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    y = np.zeros((3, 6), dtype=np.float32)
    y[:, 0] = np.asarray([0.0, 1.0, 1.5], dtype=np.float32)
    prior = np.zeros((17, 6), dtype=np.float32)
    replay = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    root = np.sqrt(replay).astype(np.float32)
    numpy_post = posterior_solve_numpy_fp32(
        z * root[:, None], y * root[:, None], prior, precision=0.1)
    mlx_post = weighted_posterior_mlx_fp32(
        z, y, prior, replay, precision=0.1)
    np.testing.assert_allclose(
        mlx_post.coefficients, numpy_post.coefficients, rtol=2e-4, atol=2e-5)
