# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.atw_codec_atick_tishby_wyner_triple`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.symposium_impls.atw_codec_atick_tishby_wyner_triple import (
    ATW_BETA_FROM_CONTEST_FORMULA,
    ATWCodecConfig,
    ATWCompositeLoss,
    atick_redlich_cooperative_loss,
    compose_atw_lagrangian,
    tishby_information_bottleneck_lagrangian,
    update_from_anchor,
    wyner_ziv_conditional_rate,
)


# ----- config validation -----------------------------------------------------------------------


def test_config_default_beta_matches_contest_formula_ratio() -> None:
    """β = 100/25 = 4 per Tao Phase E Eureka #3."""
    cfg = ATWCodecConfig()
    assert cfg.beta_information_bottleneck == pytest.approx(4.0)
    assert ATW_BETA_FROM_CONTEST_FORMULA == pytest.approx(4.0)


def test_config_invalid_beta_raises() -> None:
    with pytest.raises(ValueError):
        ATWCodecConfig(beta_information_bottleneck=0.0)


def test_config_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        ATWCodecConfig(alpha_atick_redlich_weight=-1.0)


def test_config_invalid_gamma_raises() -> None:
    with pytest.raises(ValueError):
        ATWCodecConfig(gamma_wyner_ziv_weight=-0.5)


# ----- Atick-Redlich cooperative loss ----------------------------------------------------------


def test_atick_redlich_zero_for_identical_outputs() -> None:
    """L_AR = 0 when receiver outputs match exactly."""
    a = np.array([1.0, 2.0, 3.0])
    assert atick_redlich_cooperative_loss(a, a) == pytest.approx(0.0)


def test_atick_redlich_mse_form() -> None:
    """L_AR = mean((a-b)^2). Hand-check on a small example."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    expected = float(((a - b) ** 2).mean())
    assert atick_redlich_cooperative_loss(a, b) == pytest.approx(expected)


def test_atick_redlich_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        atick_redlich_cooperative_loss(np.zeros(3), np.zeros(4))


def test_atick_redlich_empty_arrays_returns_zero() -> None:
    a = np.zeros(0)
    assert atick_redlich_cooperative_loss(a, a) == 0.0


# ----- Tishby IB Lagrangian --------------------------------------------------------------------


def test_tishby_ib_lagrangian_canonical_form() -> None:
    """L_IB = I(X;T) - β · I(T;Y); hand-check."""
    ib = tishby_information_bottleneck_lagrangian(
        mutual_information_X_T=1.0, mutual_information_T_Y=0.5, beta=4.0
    )
    # 1.0 - 4.0 * 0.5 = -1.0
    assert ib == pytest.approx(-1.0)


def test_tishby_ib_zero_when_balanced() -> None:
    """β=1 and equal I terms → L_IB=0."""
    ib = tishby_information_bottleneck_lagrangian(
        mutual_information_X_T=2.0, mutual_information_T_Y=2.0, beta=1.0
    )
    assert ib == 0.0


def test_tishby_ib_invalid_beta_raises() -> None:
    with pytest.raises(ValueError):
        tishby_information_bottleneck_lagrangian(
            mutual_information_X_T=1.0, mutual_information_T_Y=1.0, beta=0.0
        )


def test_tishby_ib_negative_mi_raises() -> None:
    with pytest.raises(ValueError):
        tishby_information_bottleneck_lagrangian(
            mutual_information_X_T=-0.1, mutual_information_T_Y=0.5
        )
    with pytest.raises(ValueError):
        tishby_information_bottleneck_lagrangian(
            mutual_information_X_T=0.5, mutual_information_T_Y=-0.1
        )


# ----- Wyner-Ziv conditional rate --------------------------------------------------------------


def test_wyner_ziv_no_side_info_reduces_to_gaussian_rd() -> None:
    """At ρ=0 the Wyner-Ziv rate equals the standard Gaussian R(D)."""
    rate = wyner_ziv_conditional_rate(
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=0.0,
        distortion_target=0.25,
    )
    expected = 0.5 * math.log2(1.0 / 0.25)
    assert rate == pytest.approx(expected, abs=1e-12)


def test_wyner_ziv_perfect_side_info_zero_rate() -> None:
    """At ρ=1 (perfect correlation) the rate is zero (side info gives X)."""
    rate = wyner_ziv_conditional_rate(
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=1.0,
        distortion_target=0.001,
    )
    assert rate == 0.0


def test_wyner_ziv_at_distortion_eq_effective_variance() -> None:
    """At D = σ²(1-ρ²) the rate is exactly zero."""
    rate = wyner_ziv_conditional_rate(
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=0.5,
        distortion_target=1.0 * (1.0 - 0.25),
    )
    assert rate == 0.0


def test_wyner_ziv_zero_distortion_returns_inf() -> None:
    rate = wyner_ziv_conditional_rate(
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=0.5,
        distortion_target=0.0,
    )
    assert rate == float("inf")


def test_wyner_ziv_invalid_variance_raises() -> None:
    with pytest.raises(ValueError):
        wyner_ziv_conditional_rate(
            variance_source=0.0,
            variance_side_info=1.0,
            correlation_source_side=0.0,
            distortion_target=0.1,
        )
    with pytest.raises(ValueError):
        wyner_ziv_conditional_rate(
            variance_source=1.0,
            variance_side_info=-1.0,
            correlation_source_side=0.0,
            distortion_target=0.1,
        )


def test_wyner_ziv_invalid_correlation_raises() -> None:
    with pytest.raises(ValueError):
        wyner_ziv_conditional_rate(
            variance_source=1.0,
            variance_side_info=1.0,
            correlation_source_side=1.5,
            distortion_target=0.1,
        )


def test_wyner_ziv_negative_correlation_handled() -> None:
    """Negative ρ has same effect as positive (only ρ² matters)."""
    rate_pos = wyner_ziv_conditional_rate(
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=0.5,
        distortion_target=0.1,
    )
    rate_neg = wyner_ziv_conditional_rate(
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=-0.5,
        distortion_target=0.1,
    )
    assert rate_pos == pytest.approx(rate_neg, abs=1e-12)


# ----- composition end-to-end ------------------------------------------------------------------


def test_compose_atw_lagrangian_returns_well_formed() -> None:
    cfg = ATWCodecConfig()
    encoded = np.array([0.1, 0.2, 0.3])
    target = np.array([0.0, 0.0, 0.0])
    result = compose_atw_lagrangian(
        config=cfg,
        mutual_information_X_T=1.0,
        mutual_information_T_Y=0.5,
        encoded_receiver_output=encoded,
        target_receiver_output=target,
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=0.3,
        distortion_target=0.2,
    )
    assert isinstance(result, ATWCompositeLoss)
    assert result.beta_used == pytest.approx(4.0)
    assert math.isfinite(result.total)
    assert result.atick_redlich_term > 0
    assert result.wyner_ziv_term >= 0


def test_compose_atw_lagrangian_zero_residual_terms_when_perfect() -> None:
    """Perfect receiver + saturated side-info → ATW collapses to L_IB only."""
    cfg = ATWCodecConfig(alpha_atick_redlich_weight=1.0, gamma_wyner_ziv_weight=1.0)
    same = np.array([0.5, 0.5, 0.5])
    result = compose_atw_lagrangian(
        config=cfg,
        mutual_information_X_T=2.0,
        mutual_information_T_Y=1.0,
        encoded_receiver_output=same,
        target_receiver_output=same,  # zero AR loss
        variance_source=1.0,
        variance_side_info=1.0,
        correlation_source_side=1.0,  # ρ=1 → WZ=0
        distortion_target=0.01,
    )
    assert result.atick_redlich_term == 0.0
    assert result.wyner_ziv_term == 0.0
    # Only L_IB = 2.0 - 4.0 * 1.0 = -2.0 survives.
    assert result.total == pytest.approx(-2.0, abs=1e-12)


# ----- continual learning hook -----------------------------------------------------------------


def test_update_from_anchor_missing_fields_returns_none() -> None:
    assert update_from_anchor({"mutual_information_X_T": 1.0}) is None


def test_update_from_anchor_invalid_scalars_returns_none() -> None:
    anchor = {
        "mutual_information_X_T": "abc",
        "mutual_information_T_Y": 0.5,
        "variance_source": 1.0,
        "variance_side_info": 1.0,
        "correlation_source_side": 0.3,
        "distortion_target": 0.1,
    }
    assert update_from_anchor(anchor) is None


def test_update_from_anchor_with_tensors_returns_loss() -> None:
    anchor = {
        "mutual_information_X_T": 1.0,
        "mutual_information_T_Y": 0.5,
        "variance_source": 1.0,
        "variance_side_info": 1.0,
        "correlation_source_side": 0.3,
        "distortion_target": 0.1,
        "encoded_receiver_output": np.array([0.1, 0.2]),
        "target_receiver_output": np.array([0.0, 0.0]),
    }
    result = update_from_anchor(anchor)
    assert result is not None
    assert isinstance(result, ATWCompositeLoss)


def test_update_from_anchor_scalar_fallback_returns_loss() -> None:
    anchor = {
        "mutual_information_X_T": 1.0,
        "mutual_information_T_Y": 0.5,
        "variance_source": 1.0,
        "variance_side_info": 1.0,
        "correlation_source_side": 0.3,
        "distortion_target": 0.1,
        "atick_redlich_term": 0.05,  # scalar fallback
    }
    result = update_from_anchor(anchor)
    assert result is not None
    assert result.atick_redlich_term == pytest.approx(0.05)
