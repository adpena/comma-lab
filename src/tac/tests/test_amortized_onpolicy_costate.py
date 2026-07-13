from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from tac.scorer_surrogate.amortized_onpolicy_costate import (
    AUTHORITY_SCOPE,
    RESEARCH_ONLY,
    AmortizedCostateConfig,
    AmortizedOnPolicyCostate,
    EMACostateProvider,
    OnPolicyCostateError,
    OnPolicyTransition,
    ProviderCustody,
    checkpoint_payload,
    dense_costate_loss,
    fit_dense_onpolicy_batch,
    inject_detached_costate,
    predict_ema_detached_costate,
    restore_checkpoint_payload,
    validate_dense_onpolicy_batch,
)


def _config(
    *,
    ema_decay: float = 0.0,
    admission_min_relative_improvement: float = 0.01,
) -> AmortizedCostateConfig:
    return AmortizedCostateConfig(
        frame_channels=3,
        hidden_channels=6,
        branch_kernel_sizes=(1, 3),
        frame_value_scale=255.0,
        normalization_floor=1.0e-8,
        mse_weight=1.0,
        cosine_weight=0.1,
        ema_decay=ema_decay,
        admission_min_relative_improvement=admission_min_relative_improvement,
    )


def _samples(count: int = 3) -> list[OnPolicyTransition]:
    generator = torch.Generator().manual_seed(455)
    anchor_frame = torch.rand((1, 3, 6, 8), generator=generator) * 255.0
    anchor_costate = torch.randn((1, 3, 6, 8), generator=generator) * 0.02
    result: list[OnPolicyTransition] = []
    for offset in range(count):
        step = offset + 5
        displacement = torch.randn((1, 3, 6, 8), generator=generator) * float(offset + 1) * 8.0
        current_frame = (anchor_frame + displacement).clamp(0.0, 255.0)
        delta_unit = (current_frame - anchor_frame) / 255.0
        current_costate = anchor_costate + 0.4 * anchor_costate.square().mean().sqrt() * torch.tanh(
            3.0 * delta_unit
        )
        result.append(
            OnPolicyTransition(
                anchor_frame=anchor_frame,
                anchor_costate=anchor_costate,
                current_frame=current_frame,
                current_costate=current_costate,
                trajectory_step=step,
                custody=ProviderCustody("a" * 64, "test"),
            )
        )
    return result


def test_multiscale_model_is_nonlinear_and_current_frame_dependent() -> None:
    torch.manual_seed(7)
    model = AmortizedOnPolicyCostate(_config())
    sample = _samples(1)[0]
    at_anchor = model(sample.anchor_frame, sample.anchor_frame, sample.anchor_costate)
    torch.testing.assert_close(at_anchor, sample.anchor_costate, rtol=0.0, atol=0.0)

    delta = sample.current_frame - sample.anchor_frame
    once = model(sample.anchor_frame + delta, sample.anchor_frame, sample.anchor_costate)
    twice = model(sample.anchor_frame + 2.0 * delta, sample.anchor_frame, sample.anchor_costate)
    assert not torch.equal(once, at_anchor)
    assert not torch.allclose(twice - at_anchor, 2.0 * (once - at_anchor), rtol=1.0e-4, atol=1.0e-7)


def test_exact_anchor_costate_loss_has_nonnegative_fp32_floor() -> None:
    model = AmortizedOnPolicyCostate(_config())
    sample = _samples(1)[0]
    exact_anchor = replace(
        sample,
        current_frame=sample.anchor_frame.clone(),
        current_costate=sample.anchor_costate.clone(),
    )

    loss = dense_costate_loss(model, [exact_anchor], config=model.config)

    assert loss == 0.0


def test_dense_window_and_sealed_offline_rejection_fail_closed() -> None:
    samples = _samples(3)
    validate_dense_onpolicy_batch(samples)
    with pytest.raises(OnPolicyCostateError, match="step-contiguous"):
        validate_dense_onpolicy_batch([samples[0], samples[2]])
    wrong_run = replace(samples[1], custody=ProviderCustody("b" * 64, "test"))
    with pytest.raises(OnPolicyCostateError, match="run/regime provider fingerprint"):
        validate_dense_onpolicy_batch([samples[0], wrong_run])

    offline = OnPolicyTransition(
        anchor_frame=samples[0].anchor_frame,
        anchor_costate=samples[0].anchor_costate,
        current_frame=samples[0].current_frame,
        current_costate=samples[0].current_costate,
        trajectory_step=samples[0].trajectory_step,
        custody=samples[0].custody,
        source="fixed_offline_dataset",
    )
    with pytest.raises(OnPolicyCostateError, match="offline"):
        validate_dense_onpolicy_batch([offline])


def test_chain_rule_injection_detaches_provider_and_returns_exact_costate_vjp() -> None:
    frame = torch.randn(1, 3, 4, 5, requires_grad=True)
    costate = torch.randn_like(frame, requires_grad=True)
    injection = inject_detached_costate(frame, costate)
    injection.backward()
    torch.testing.assert_close(frame.grad, costate.detach(), rtol=0.0, atol=0.0)
    assert costate.grad is None


def test_live_improvement_does_not_admit_when_slow_ema_gate_does_not_improve_enough() -> None:
    torch.manual_seed(12)
    config = _config(ema_decay=0.9999, admission_min_relative_improvement=0.02)
    model = AmortizedOnPolicyCostate(config)
    provider = EMACostateProvider(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    result = fit_dense_onpolicy_batch(
        model,
        _samples(3),
        optimizer=optimizer,
        optimizer_steps=2,
        ema_provider=provider,
    )
    assert result.live_improved
    assert not result.ema_improved
    assert not result.admitted


def test_ema_provider_improvement_is_the_admission_authority() -> None:
    torch.manual_seed(13)
    config = _config(ema_decay=0.0, admission_min_relative_improvement=0.005)
    model = AmortizedOnPolicyCostate(config)
    provider = EMACostateProvider(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    result = fit_dense_onpolicy_batch(
        model,
        _samples(3),
        optimizer=optimizer,
        optimizer_steps=12,
        ema_provider=provider,
    )
    assert result.ema_improved
    assert result.admitted is result.ema_improved
    assert result.authority == "ema_shadow_training_signal_loss_only"
    assert result.ema_updates == 12
    served = predict_ema_detached_costate(
        provider,
        current_frame=_samples(1)[0].current_frame,
        anchor_frame=_samples(1)[0].anchor_frame,
        anchor_costate=_samples(1)[0].anchor_costate,
    )
    assert not served.requires_grad
    assert RESEARCH_ONLY and AUTHORITY_SCOPE == "training_signal_only_not_dseg_or_score"


def test_checkpoint_roundtrip_preserves_live_ema_optimizer_and_position() -> None:
    torch.manual_seed(21)
    config = _config(ema_decay=0.5)
    model = AmortizedOnPolicyCostate(config)
    provider = EMACostateProvider(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    fit_dense_onpolicy_batch(
        model,
        _samples(2),
        optimizer=optimizer,
        optimizer_steps=2,
        ema_provider=provider,
    )
    payload = checkpoint_payload(model, provider, optimizer, next_trajectory_step=7)
    frozen_live = copy.deepcopy(payload["live_model"])
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    for name, tensor in frozen_live.items():
        torch.testing.assert_close(payload["live_model"][name], tensor, rtol=0.0, atol=0.0)
    model.load_state_dict(frozen_live)

    restored_model = AmortizedOnPolicyCostate(config)
    restored_provider = EMACostateProvider(restored_model)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
    assert restore_checkpoint_payload(
        copy.deepcopy(payload), restored_model, restored_provider, restored_optimizer
    ) == 7
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name], tensor, rtol=0.0, atol=0.0)
    for name, tensor in provider.state_dict().items():
        torch.testing.assert_close(restored_provider.state_dict()[name], tensor, rtol=0.0, atol=0.0)


def test_nonfinite_and_shape_errors_fail_closed() -> None:
    sample = _samples(1)[0]
    model = AmortizedOnPolicyCostate(_config())
    bad = sample.current_frame.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(OnPolicyCostateError, match="finite"):
        model(bad, sample.anchor_frame, sample.anchor_costate)
    with pytest.raises(OnPolicyCostateError, match="identical-shape"):
        model(sample.current_frame[:, :, :-1], sample.anchor_frame, sample.anchor_costate)
    with pytest.raises(OnPolicyCostateError, match="shape mismatch"):
        inject_detached_costate(sample.current_frame[:, :, :-1], sample.anchor_costate)

    other_model = AmortizedOnPolicyCostate(_config())
    wrong_optimizer = torch.optim.Adam(other_model.parameters(), lr=0.01)
    with pytest.raises(OnPolicyCostateError, match="optimizer must own exactly"):
        fit_dense_onpolicy_batch(
            model,
            [sample],
            optimizer=wrong_optimizer,
            optimizer_steps=1,
            ema_provider=EMACostateProvider(model),
        )
