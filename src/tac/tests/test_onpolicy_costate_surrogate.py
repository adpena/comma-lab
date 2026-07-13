from __future__ import annotations

import pytest
import torch

from tac.ib_lagrangian_aux_scorer import _EMA
from tac.scorer_surrogate.onpolicy_costate import (
    NonlinearCostateSurrogate,
    OnPolicyCostateError,
    OnPolicyTransition,
    ProviderCustody,
    fit_onpolicy_transitions,
    predict_detached_costate,
    whole_step_economics,
)


def _custody() -> ProviderCustody:
    return ProviderCustody("a" * 64, "test")


def _sample() -> OnPolicyTransition:
    torch.manual_seed(4)
    anchor_frame = torch.rand(1, 3, 12, 16) * 255.0
    anchor_costate = torch.randn(1, 3, 12, 16) * 0.01
    current_frame = (anchor_frame + 4.0 * torch.randn_like(anchor_frame)).clamp(0, 255)
    current_costate = anchor_costate + 0.2 * anchor_costate.tanh() * (
        current_frame - anchor_frame
    ) / 255.0
    return OnPolicyTransition(anchor_frame, anchor_costate, current_frame, current_costate, 1, _custody())


def test_untrained_model_is_exact_anchor_cache_positive_control() -> None:
    sample = _sample()
    model = NonlinearCostateSurrogate(hidden_channels=4)
    predicted = predict_detached_costate(
        model,
        current_frame=sample.current_frame,
        anchor_frame=sample.anchor_frame,
        anchor_costate=sample.anchor_costate,
    )
    torch.testing.assert_close(predicted, sample.anchor_costate, rtol=0, atol=0)


def test_detached_provider_costate_can_drive_renderer_vjp() -> None:
    sample = _sample()
    model = NonlinearCostateSurrogate(hidden_channels=4)
    renderer_output = sample.current_frame.clone().requires_grad_(True)
    predicted = predict_detached_costate(
        model,
        current_frame=renderer_output.detach(),
        anchor_frame=sample.anchor_frame,
        anchor_costate=sample.anchor_costate,
    )
    gradient = torch.autograd.grad((renderer_output * predicted).sum(), renderer_output)[0]
    torch.testing.assert_close(gradient, predicted)


def test_onpolicy_fit_reduces_costate_objective_and_changes_prediction() -> None:
    sample = _sample()
    model = NonlinearCostateSurrogate(hidden_channels=4)
    before = predict_detached_costate(
        model,
        current_frame=sample.current_frame,
        anchor_frame=sample.anchor_frame,
        anchor_costate=sample.anchor_costate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    ema = _EMA(model, decay=0.997)
    result = fit_onpolicy_transitions(model, [sample], optimizer=optimizer, steps=30, ema=ema)
    after = predict_detached_costate(
        model,
        current_frame=sample.current_frame,
        anchor_frame=sample.anchor_frame,
        anchor_costate=sample.anchor_costate,
    )
    assert result.finite and result.improved
    assert not torch.equal(before, after)


def test_offline_source_is_rejected() -> None:
    sample = _sample()
    bad = OnPolicyTransition(
        sample.anchor_frame,
        sample.anchor_costate,
        sample.current_frame,
        sample.current_costate,
        1,
        _custody(),
        source="fixed_dataset",
    )
    with pytest.raises(OnPolicyCostateError, match="offline"):
        bad.validate()


def test_invalid_provider_custody_is_rejected() -> None:
    sample = _sample()
    bad = OnPolicyTransition(
        sample.anchor_frame,
        sample.anchor_costate,
        sample.current_frame,
        sample.current_costate,
        1,
        ProviderCustody("not-a-sha", "test"),
    )
    with pytest.raises(OnPolicyCostateError, match="fingerprint"):
        bad.validate()


def test_whole_step_economics_matches_mission_formula() -> None:
    row = whole_step_economics(cadence=20, t_exact_seconds=2.0, t_surrogate_seconds=0.02)
    assert row["speedup"] == pytest.approx(40.0 / 2.38)
    assert row["exact_teacher_skip_fraction"] == pytest.approx(0.95)


def test_nonfinite_provider_input_fails_closed() -> None:
    sample = _sample()
    model = NonlinearCostateSurrogate(hidden_channels=4)
    bad = sample.current_frame.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(OnPolicyCostateError, match="finite"):
        predict_detached_costate(
            model,
            current_frame=bad,
            anchor_frame=sample.anchor_frame,
            anchor_costate=sample.anchor_costate,
        )
