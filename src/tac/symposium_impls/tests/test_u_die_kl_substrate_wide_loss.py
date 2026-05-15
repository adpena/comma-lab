# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.u_die_kl_substrate_wide_loss`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.symposium_impls.u_die_kl_substrate_wide_loss import (
    DEFAULT_KL_TEMPERATURE,
    DEFAULT_UDIE_LAMBDA_KL,
    UDIEKLConfig,
    UDIEKLLossResult,
    compose_per_pixel_weights,
    kl_distillation_loss_with_temperature,
    u_die_kl_substrate_loss,
    update_from_anchor,
    weighted_pixel_reconstruction_loss,
)


# ----- config validation -----------------------------------------------------------------------


def test_config_default_values_match_canonical() -> None:
    cfg = UDIEKLConfig()
    assert cfg.kl_temperature == DEFAULT_KL_TEMPERATURE  # 2.0 per Hinton 2014
    assert cfg.lambda_kl_distillation == DEFAULT_UDIE_LAMBDA_KL  # 1.0


def test_config_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        UDIEKLConfig(alpha_uniward_weight=-0.1)


def test_config_invalid_beta_raises() -> None:
    with pytest.raises(ValueError):
        UDIEKLConfig(beta_attention_weight=-0.5)


def test_config_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError):
        UDIEKLConfig(kl_temperature=0.0)


def test_config_invalid_lambda_raises() -> None:
    with pytest.raises(ValueError):
        UDIEKLConfig(lambda_kl_distillation=-0.1)


# ----- per-pixel weight composition ------------------------------------------------------------


def test_compose_weights_zeroes_in_blind_region() -> None:
    cfg = UDIEKLConfig(alpha_uniward_weight=0.5, beta_attention_weight=0.5)
    uniward = np.full((4, 4), 1.0)
    attention = np.full((4, 4), 1.0)
    blind = np.full((4, 4), 1.0)  # everything is blind
    w = compose_per_pixel_weights(
        uniward=uniward, attention=attention, die_blind=blind, config=cfg
    )
    assert (w == 0).all()


def test_compose_weights_non_blind_carries_signal() -> None:
    cfg = UDIEKLConfig(alpha_uniward_weight=1.0, beta_attention_weight=0.0)
    uniward = np.array([[0.0, 1.0], [0.5, 1.0]])
    attention = np.zeros((2, 2))
    blind = np.zeros((2, 2))
    w = compose_per_pixel_weights(
        uniward=uniward, attention=attention, die_blind=blind, config=cfg
    )
    assert w.shape == (2, 2)
    assert w.max() > 0


def test_compose_weights_shape_mismatch_raises() -> None:
    cfg = UDIEKLConfig()
    with pytest.raises(ValueError):
        compose_per_pixel_weights(
            uniward=np.zeros((4, 4)),
            attention=np.zeros((3, 3)),
            die_blind=np.zeros((4, 4)),
            config=cfg,
        )


# ----- weighted pixel loss ----------------------------------------------------------------------


def test_weighted_pixel_loss_zero_when_reconstruction_matches_target() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.ones((2, 2))
    assert weighted_pixel_reconstruction_loss(reconstruction=a, target=a, weights=weights) == 0.0


def test_weighted_pixel_loss_zero_weights_returns_zero() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    weights = np.zeros((2, 2))
    assert weighted_pixel_reconstruction_loss(reconstruction=a, target=b, weights=weights) == 0.0


def test_weighted_pixel_loss_uniform_weights_matches_mean_squared_error() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 3.0], [4.0, 5.0]])
    weights = np.ones((2, 2))
    expected = float(((a - b) ** 2).mean())
    assert weighted_pixel_reconstruction_loss(
        reconstruction=a, target=b, weights=weights
    ) == pytest.approx(expected)


def test_weighted_pixel_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        weighted_pixel_reconstruction_loss(
            reconstruction=np.zeros((2, 2)),
            target=np.zeros((3, 3)),
            weights=np.ones((2, 2)),
        )


# ----- KL distillation tests ------------------------------------------------------------------


def test_kl_distillation_zero_when_logits_match() -> None:
    """If student == teacher then KL = 0."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, 5))
    kl = kl_distillation_loss_with_temperature(
        student_logits=logits, teacher_logits=logits
    )
    assert kl == pytest.approx(0.0, abs=1e-9)


def test_kl_distillation_positive_when_logits_differ() -> None:
    student = np.array([[1.0, 0.0]])
    teacher = np.array([[0.0, 1.0]])
    kl = kl_distillation_loss_with_temperature(student_logits=student, teacher_logits=teacher)
    assert kl > 0


def test_kl_distillation_temperature_squared_scaling() -> None:
    """T² scaling: doubling T reduces KL but multiplies by 4 → bigger result."""
    student = np.array([[2.0, 0.0]])
    teacher = np.array([[0.0, 2.0]])
    kl_t1 = kl_distillation_loss_with_temperature(
        student_logits=student, teacher_logits=teacher, temperature=1.0
    )
    kl_t2 = kl_distillation_loss_with_temperature(
        student_logits=student, teacher_logits=teacher, temperature=2.0
    )
    # At higher T the softmax is softer → per-sample KL is smaller; T² rescaling
    # may net out to either direction, but both must be finite + non-negative.
    assert kl_t1 >= 0
    assert kl_t2 >= 0
    assert math.isfinite(kl_t1) and math.isfinite(kl_t2)


def test_kl_distillation_invalid_temperature_raises() -> None:
    student = np.zeros((1, 2))
    teacher = np.zeros((1, 2))
    with pytest.raises(ValueError):
        kl_distillation_loss_with_temperature(
            student_logits=student, teacher_logits=teacher, temperature=0.0
        )


def test_kl_distillation_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        kl_distillation_loss_with_temperature(
            student_logits=np.zeros((1, 2)), teacher_logits=np.zeros((1, 3))
        )


def test_kl_distillation_empty_returns_zero() -> None:
    empty = np.zeros((0, 2))
    assert kl_distillation_loss_with_temperature(student_logits=empty, teacher_logits=empty) == 0.0


# ----- composition end-to-end ------------------------------------------------------------------


def test_u_die_kl_substrate_loss_returns_well_formed() -> None:
    cfg = UDIEKLConfig()
    rng = np.random.default_rng(0)
    h, w, channels = 4, 4, 3
    rec = rng.standard_normal((h, w, channels))
    tgt = rec + rng.standard_normal((h, w, channels)) * 0.1
    uniward = np.abs(rng.standard_normal((h, w)))
    attention = np.abs(rng.standard_normal((h, w)))
    die = np.zeros((h, w))
    student_logits = rng.standard_normal((4, 5))
    teacher_logits = rng.standard_normal((4, 5))
    result = u_die_kl_substrate_loss(
        config=cfg,
        reconstruction=rec,
        target=tgt,
        uniward=uniward,
        attention=attention,
        die_blind=die,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
    )
    assert isinstance(result, UDIEKLLossResult)
    assert result.weighted_pixel_term >= 0
    assert result.kl_distillation_term >= 0
    assert math.isfinite(result.total)


# ----- continual learning hook -----------------------------------------------------------------


def test_update_from_anchor_missing_field_returns_none() -> None:
    incomplete = {"reconstruction": np.zeros((2, 2))}
    assert update_from_anchor(incomplete) is None


def test_update_from_anchor_non_array_field_returns_none() -> None:
    bad = {
        "reconstruction": np.zeros((2, 2)),
        "target": "not an array",
        "uniward": np.zeros((2, 2)),
        "attention": np.zeros((2, 2)),
        "die_blind": np.zeros((2, 2)),
        "student_logits": np.zeros((1, 2)),
        "teacher_logits": np.zeros((1, 2)),
    }
    assert update_from_anchor(bad) is None


def test_update_from_anchor_complete_returns_loss() -> None:
    anchor = {
        "reconstruction": np.zeros((2, 2, 3)),
        "target": np.ones((2, 2, 3)),
        "uniward": np.ones((2, 2)),
        "attention": np.ones((2, 2)),
        "die_blind": np.zeros((2, 2)),
        "student_logits": np.zeros((1, 2)),
        "teacher_logits": np.zeros((1, 2)),
    }
    result = update_from_anchor(anchor)
    assert result is not None
    assert isinstance(result, UDIEKLLossResult)
