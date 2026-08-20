# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.stage_transition_soft_velocity_blend import (
    StageTransitionSoftVelocityBlendConfig,
    StageTransitionSoftVelocityBlendError,
    alpha_at_step,
    blend_momentum,
    replay_custody_issues,
    rms,
    soft_velocity_blend_sequence,
    window_steps_from_beta2,
)


def test_beta2_window_is_derived_in_optimizer_steps() -> None:
    assert window_steps_from_beta2(0.999, c=2.0) == 2000
    assert window_steps_from_beta2(0.999, c=1.0) == 1000
    with pytest.raises(StageTransitionSoftVelocityBlendError, match="beta2"):
        window_steps_from_beta2(1.0)


def test_alpha_schedule_endpoints_and_cosine_midpoint() -> None:
    linear = StageTransitionSoftVelocityBlendConfig(window_steps=5)
    assert alpha_at_step(1, linear) == pytest.approx(0.0)
    assert alpha_at_step(3, linear) == pytest.approx(0.5)
    assert alpha_at_step(99, linear) == pytest.approx(1.0)

    cosine = StageTransitionSoftVelocityBlendConfig(window_steps=5, shape="cosine")
    assert alpha_at_step(3, cosine) == pytest.approx(0.5)


def test_blend_momentum_uses_real_arrays_and_clip() -> None:
    mapped = np.array([2.0, 2.0])
    fresh = np.array([6.0, 10.0])
    out = blend_momentum(mapped, fresh, 0.25)
    np.testing.assert_allclose(out, np.array([3.0, 4.0]))

    clipped = blend_momentum(mapped, fresh, 1.0, clip_rms=2.0)
    assert rms(clipped) == pytest.approx(2.0)
    with pytest.raises(StageTransitionSoftVelocityBlendError, match="shape"):
        blend_momentum(np.ones(2), np.ones(3), 0.5)
    with pytest.raises(StageTransitionSoftVelocityBlendError, match="non-finite"):
        blend_momentum(np.ones(2), np.array([1.0, np.nan]), 0.5)


def test_sequence_consumes_gradient_values_not_markers() -> None:
    cfg = StageTransitionSoftVelocityBlendConfig(window_steps=3)
    mapped = np.array([10.0, 0.0])
    gradients_a = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    gradients_b = [np.array([2.0, 0.0]), np.array([2.0, 0.0]), np.array([2.0, 0.0])]

    rows_a = soft_velocity_blend_sequence(mapped, gradients_a, cfg, beta1=0.5)
    rows_b = soft_velocity_blend_sequence(mapped, gradients_b, cfg, beta1=0.5)
    assert rows_a[-1].alpha == pytest.approx(1.0)
    assert rows_a[-1].fresh_moment[0] == pytest.approx(0.875)
    assert rows_b[-1].fresh_moment[0] == pytest.approx(1.75)
    assert rows_a[-1].blended_moment[0] != rows_b[-1].blended_moment[0]


def test_empty_sequence_refuses_vacuous_backtest() -> None:
    cfg = StageTransitionSoftVelocityBlendConfig(window_steps=3)
    with pytest.raises(StageTransitionSoftVelocityBlendError, match="at least one"):
        soft_velocity_blend_sequence(np.ones(2), [], cfg)


def test_replay_custody_issues_name_missing_inputs() -> None:
    issues = replay_custody_issues(
        previous_mapped_m_available=False,
        gradient_vectors_available=1,
        required_gradient_vectors=4,
        baseline_controls_available=False,
    )
    assert any("previous mapped" in issue for issue in issues)
    assert any("1/4" in issue for issue in issues)
    assert any("baseline-control" in issue for issue in issues)
