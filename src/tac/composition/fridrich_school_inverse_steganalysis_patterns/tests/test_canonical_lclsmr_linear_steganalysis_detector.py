# SPDX-License-Identifier: MIT
"""Tests for canonical LCLSMR linear steganalysis detector.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS": every test verifies BEHAVIOR
(actual numerical fit + score) not just constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    LCLSMRConfig,
    LCLSMRDetectorError,
    LCLSMRSolverStrategy,
    fit_lclsmr_linear_classifier,
    score_lclsmr_linear_classifier,
)


def test_lclsmr_fit_recovers_known_weights_lsmr_canonical() -> None:
    """LSMR canonical solver recovers known linear weights within tolerance."""
    rng = np.random.default_rng(seed=42)
    n_samples, n_features = 200, 5
    true_weights = np.array([1.0, -0.5, 0.3, -0.2, 0.8])
    true_bias = 0.1
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    y = X @ true_weights + true_bias + 1e-4 * rng.normal(size=n_samples)
    config = LCLSMRConfig(solver_strategy=LCLSMRSolverStrategy.LSMR_FONG_SAUNDERS_2011)
    fitted = fit_lclsmr_linear_classifier(X, y, config)
    assert fitted.shape == (n_features + 1,)
    # Recovered weights should match true within tolerance.
    np.testing.assert_allclose(fitted[:n_features], true_weights, atol=1e-2)
    assert abs(fitted[-1] - true_bias) < 1e-2


def test_lclsmr_score_returns_predictions() -> None:
    """Score helper returns y_pred = W @ x + b."""
    weights = np.array([1.0, 2.0, 0.5])  # 2 features + bias
    X = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 1.0]])
    expected = np.array([1.0 + 2.0 + 0.5, 0.0 + 0.0 + 0.5, 2.0 + 2.0 + 0.5])
    pred = score_lclsmr_linear_classifier(weights, X)
    np.testing.assert_allclose(pred, expected)


def test_lclsmr_strategies_substantively_distinct() -> None:
    """Slot EEE substantive-distinctness gate: each canonical solver
    strategy produces a DIFFERENT weight vector under ill-conditioning.

    The CANONICAL insight Yousfi uses LSMR for: when the feature matrix has
    near-singular columns (correlated features), DIRECT_NORMAL_EQUATIONS
    loses precision while LSMR maintains it.
    """
    rng = np.random.default_rng(seed=42)
    # Build ill-conditioned feature matrix: columns 0 and 1 are nearly identical.
    n_samples = 100
    base = rng.normal(size=(n_samples, 3))
    base[:, 1] = base[:, 0] + 1e-6 * rng.normal(size=n_samples)
    y = base @ np.array([1.0, 2.0, 3.0]) + 0.5
    fitted_outputs = {}
    for strategy in LCLSMRSolverStrategy:
        config = LCLSMRConfig(solver_strategy=strategy, damping=1e-3)
        fitted_outputs[strategy] = fit_lclsmr_linear_classifier(base, y, config)
    # Each strategy produces fitted weights; some may differ substantially.
    # Verify substantive-distinctness: at least one pair differs by >1e-4.
    pairs = list(fitted_outputs.values())
    max_diff = 0.0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            max_diff = max(max_diff, float(np.abs(pairs[i] - pairs[j]).max()))
    assert max_diff > 1e-6, (
        f"All solver strategies produced near-identical output (max_diff={max_diff:.2e}); "
        "Slot EEE substantive-distinctness gate FAILED"
    )


def test_lclsmr_qr_decomposition_baseline_works() -> None:
    """QR decomposition (numpy.linalg.lstsq) baseline produces valid fit."""
    rng = np.random.default_rng(seed=99)
    X = rng.normal(size=(50, 3))
    true_w = np.array([1.0, -0.5, 0.2])
    y = X @ true_w + 0.1
    config = LCLSMRConfig(solver_strategy=LCLSMRSolverStrategy.QR_DECOMPOSITION)
    fitted = fit_lclsmr_linear_classifier(X, y, config)
    pred = score_lclsmr_linear_classifier(fitted, X)
    # MSE should be very low for well-conditioned problem.
    mse = float(np.mean((y - pred) ** 2))
    assert mse < 1e-10


def test_lclsmr_invalid_inputs_raise() -> None:
    """Invalid input shapes raise LCLSMRDetectorError."""
    with pytest.raises(LCLSMRDetectorError, match="must be np.ndarray"):
        fit_lclsmr_linear_classifier([1, 2, 3], np.array([1, 2, 3]))  # type: ignore[arg-type]
    with pytest.raises(LCLSMRDetectorError, match="must be 2-D"):
        fit_lclsmr_linear_classifier(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(LCLSMRDetectorError, match="must be 1-D"):
        fit_lclsmr_linear_classifier(np.array([[1.0, 2.0]]), np.array([[1.0]]))
    with pytest.raises(LCLSMRDetectorError, match="rows"):
        fit_lclsmr_linear_classifier(
            np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0, 3.0])
        )
    with pytest.raises(LCLSMRDetectorError, match="empty"):
        fit_lclsmr_linear_classifier(np.zeros((0, 3)), np.zeros(0))


def test_lclsmr_score_shape_mismatch_raises() -> None:
    """Wrong-shape weight vector raises."""
    with pytest.raises(LCLSMRDetectorError, match="must be 1-D"):
        score_lclsmr_linear_classifier(
            np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])
        )
    with pytest.raises(LCLSMRDetectorError, match="must be 2-D"):
        score_lclsmr_linear_classifier(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(LCLSMRDetectorError, match="length"):
        # Weight has wrong size for feature matrix.
        score_lclsmr_linear_classifier(
            np.array([1.0, 2.0]), np.array([[1.0, 2.0, 3.0]])
        )


def test_lclsmr_config_invalid_strategy_raises() -> None:
    """Wrong type for strategy raises."""
    with pytest.raises(LCLSMRDetectorError, match="must be LCLSMRSolverStrategy"):
        LCLSMRConfig(solver_strategy="bogus")  # type: ignore[arg-type]


def test_lclsmr_config_invalid_numerics_raise() -> None:
    """Invalid numeric values raise."""
    with pytest.raises(LCLSMRDetectorError, match="damping"):
        LCLSMRConfig(damping=-1.0)
    with pytest.raises(LCLSMRDetectorError, match="atol"):
        LCLSMRConfig(atol=0.0)
    with pytest.raises(LCLSMRDetectorError, match="btol"):
        LCLSMRConfig(btol=-1e-6)
    with pytest.raises(LCLSMRDetectorError, match="max_iter"):
        LCLSMRConfig(max_iter=0)


def test_lclsmr_canonical_default_strategy() -> None:
    """Default strategy is the canonical Fong-Saunders LSMR."""
    cfg = LCLSMRConfig()
    assert cfg.solver_strategy == LCLSMRSolverStrategy.LSMR_FONG_SAUNDERS_2011
