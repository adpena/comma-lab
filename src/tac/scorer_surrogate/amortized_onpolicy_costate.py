# SPDX-License-Identifier: MIT
"""Dense on-policy amortization of the exact SegNet input costate.

This module predicts a *training signal*, ``dL_teacher / d(frame)``.  It does
not predict ``d_seg`` and it is never an evaluation or promotion authority.
All supervised samples are :class:`OnPolicyTransition` objects emitted by the
witness's own realized-through-R trajectory.  The sealed transition validator
therefore remains the custody boundary and rejects fixed/offline data.

The provider is deliberately split into a live learner and an EMA shadow.  The
live learner is optimized, while only the EMA shadow is served and used for
admission.  Both are ordinary ``nn.Module`` state dictionaries, so a caller can
atomically preserve them together with the optimizer and trajectory position.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.boundary_math.segnet_gradient_replacement import costate_injection_loss_torch
from tac.scorer_surrogate.onpolicy_costate import (
    OnPolicyCostateError,
    OnPolicyTransition,
    ProviderCustody,
)

RESEARCH_ONLY: Final[bool] = True
AUTHORITY_SCOPE: Final[str] = "training_signal_only_not_dseg_or_score"
CHECKPOINT_SCHEMA: Final[str] = "amortized_onpolicy_costate_checkpoint.v1"
RGB_CHANNELS: Final[int] = len(("red", "green", "blue"))


@dataclass(frozen=True)
class AmortizedCostateConfig:
    """Externally supplied, fully typed architecture and fitting policy.

    No field is defaulted: callers must bind every tunable value through their
    typed DSL/config surface.  ``frame_channels`` is also explicit so the RGB
    structural assumption is recorded instead of hidden in a convolution.
    """

    frame_channels: int
    hidden_channels: int
    branch_kernel_sizes: tuple[int, ...]
    frame_value_scale: float
    normalization_floor: float
    mse_weight: float
    cosine_weight: float
    ema_decay: float
    admission_min_relative_improvement: float

    def validate(self) -> None:
        if isinstance(self.frame_channels, bool) or self.frame_channels != RGB_CHANNELS:
            raise OnPolicyCostateError("frame_channels must record the sealed RGB channel count")
        if (
            isinstance(self.hidden_channels, bool)
            or not isinstance(self.hidden_channels, int)
            or self.hidden_channels < self.frame_channels
        ):
            raise OnPolicyCostateError("hidden_channels must be an integer >= frame_channels")
        kernels = self.branch_kernel_sizes
        if not kernels or any(
            isinstance(kernel, bool)
            or not isinstance(kernel, int)
            or kernel < 1
            or kernel % 2 == 0
            for kernel in kernels
        ):
            raise OnPolicyCostateError("branch_kernel_sizes must contain positive odd integers")
        if len(set(kernels)) != len(kernels):
            raise OnPolicyCostateError("branch_kernel_sizes must be unique")
        if not math.isfinite(self.frame_value_scale) or self.frame_value_scale <= 0.0:
            raise OnPolicyCostateError("frame_value_scale must be finite > 0")
        if not math.isfinite(self.normalization_floor) or self.normalization_floor <= 0.0:
            raise OnPolicyCostateError("normalization_floor must be finite > 0")
        if not math.isfinite(self.mse_weight) or self.mse_weight < 0.0:
            raise OnPolicyCostateError("mse_weight must be finite >= 0")
        if not math.isfinite(self.cosine_weight) or self.cosine_weight < 0.0:
            raise OnPolicyCostateError("cosine_weight must be finite >= 0")
        if self.mse_weight + self.cosine_weight <= 0.0:
            raise OnPolicyCostateError("at least one loss weight must be positive")
        if not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0:
            raise OnPolicyCostateError("ema_decay must be finite in [0, 1)")
        threshold = self.admission_min_relative_improvement
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise OnPolicyCostateError("admission_min_relative_improvement must be finite in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["branch_kernel_sizes"] = list(self.branch_kernel_sizes)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AmortizedCostateConfig:
        required = {
            "frame_channels",
            "hidden_channels",
            "branch_kernel_sizes",
            "frame_value_scale",
            "normalization_floor",
            "mse_weight",
            "cosine_weight",
            "ema_decay",
            "admission_min_relative_improvement",
        }
        if set(value) != required:
            raise OnPolicyCostateError("checkpoint config keys do not match the typed schema")
        try:
            config = cls(
                frame_channels=value["frame_channels"],
                hidden_channels=value["hidden_channels"],
                branch_kernel_sizes=tuple(value["branch_kernel_sizes"]),
                frame_value_scale=value["frame_value_scale"],
                normalization_floor=value["normalization_floor"],
                mse_weight=value["mse_weight"],
                cosine_weight=value["cosine_weight"],
                ema_decay=value["ema_decay"],
                admission_min_relative_improvement=value["admission_min_relative_improvement"],
            )
        except (KeyError, TypeError) as error:
            raise OnPolicyCostateError("checkpoint config values do not match the typed schema") from error
        config.validate()
        return config


class _ScaleBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(F.gelu(self.depthwise(value)))


class AmortizedOnPolicyCostate(nn.Module):
    """Nonlinear multi-receptive-field residual model for a current costate.

    The exact anchor is a control variate.  The network predicts the normalized
    residual ``lambda(x_t) - lambda(x_anchor)`` from current RGB, frame delta,
    and anchor costate.  Multiplication by a bounded delta gate makes the
    reference cancellation structural: when current equals anchor, the output
    is bit-identically the anchored exact costate for every set of weights.
    Branches operate at parallel receptive fields without downsampling; this is
    not a spatial pyramid and preserves a frame-shaped per-pixel costate.
    """

    def __init__(self, config: AmortizedCostateConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        input_channels = config.frame_channels * 3
        self.stem = nn.Conv2d(input_channels, config.hidden_channels, kernel_size=1)
        self.branches = nn.ModuleList(
            [_ScaleBranch(config.hidden_channels, kernel) for kernel in config.branch_kernel_sizes]
        )
        mixed_channels = config.hidden_channels * len(config.branch_kernel_sizes)
        self.mixer = nn.Sequential(
            nn.Conv2d(mixed_channels, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_channels, config.frame_channels, kernel_size=1),
        )

    def _validate_inputs(
        self,
        current_frame: torch.Tensor,
        anchor_frame: torch.Tensor,
        anchor_costate: torch.Tensor,
    ) -> None:
        expected_channels = self.config.frame_channels
        if not (
            current_frame.ndim == 4
            and current_frame.shape[1] == expected_channels
            and current_frame.shape == anchor_frame.shape == anchor_costate.shape
        ):
            raise OnPolicyCostateError(
                "provider inputs must be identical-shape NCHW frame/costate tensors"
            )
        if current_frame.device != anchor_frame.device or current_frame.device != anchor_costate.device:
            raise OnPolicyCostateError("provider inputs must share a device")
        if current_frame.dtype != anchor_frame.dtype or current_frame.dtype != anchor_costate.dtype:
            raise OnPolicyCostateError("provider inputs must share a dtype")
        if not current_frame.is_floating_point():
            raise OnPolicyCostateError("provider inputs must use a floating dtype")
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (current_frame, anchor_frame, anchor_costate)
        ):
            raise OnPolicyCostateError("provider inputs must be finite")

    def costate_scale(self, anchor_costate: torch.Tensor) -> torch.Tensor:
        return (
            anchor_costate.square()
            .mean(dim=(1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(self.config.normalization_floor)
        )

    def forward(
        self,
        current_frame: torch.Tensor,
        anchor_frame: torch.Tensor,
        anchor_costate: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(current_frame, anchor_frame, anchor_costate)
        scale = self.costate_scale(anchor_costate)
        current_unit = current_frame / self.config.frame_value_scale
        delta_unit = (current_frame - anchor_frame) / self.config.frame_value_scale
        anchor_unit = anchor_costate / scale
        hidden = F.gelu(self.stem(torch.cat((current_unit, delta_unit, anchor_unit), dim=1)))
        multiscale = torch.cat([branch(hidden) for branch in self.branches], dim=1)
        residual_unit = self.mixer(multiscale)
        # This gate is data-derived and has no tuned threshold.  It enforces
        # exact anchor cancellation while preserving nonlinear frame response.
        delta_gate = torch.tanh(delta_unit.square().mean(dim=1, keepdim=True).sqrt())
        prediction = anchor_costate + scale * delta_gate * residual_unit
        if not bool(torch.isfinite(prediction).all()):
            raise OnPolicyCostateError("amortized provider emitted a nonfinite costate")
        return prediction


class EMACostateProvider(nn.Module):
    """Stateful EMA shadow that is the only served/admitted provider."""

    def __init__(self, live_model: AmortizedOnPolicyCostate) -> None:
        super().__init__()
        self.config = live_model.config
        self.provider_model = copy.deepcopy(live_model).eval()
        self.provider_model.requires_grad_(False)
        self._num_updates: torch.Tensor
        self.register_buffer("_num_updates", torch.zeros((), dtype=torch.int64))

    @property
    def updates(self) -> int:
        return int(self._num_updates.item())

    @torch.no_grad()
    def update_from_live(self, live_model: AmortizedOnPolicyCostate) -> None:
        if live_model.config != self.config:
            raise OnPolicyCostateError("live and EMA-provider configs differ")
        decay = self.config.ema_decay
        live_state = live_model.state_dict()
        shadow_state = self.provider_model.state_dict()
        if live_state.keys() != shadow_state.keys():
            raise OnPolicyCostateError("live and EMA-provider state schemas differ")
        if any(
            tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
            for tensor in (*live_state.values(), *shadow_state.values())
        ):
            raise OnPolicyCostateError("live or EMA-provider state is nonfinite")
        for name, shadow in shadow_state.items():
            live = live_state[name].detach()
            if live.shape != shadow.shape or live.dtype != shadow.dtype:
                raise OnPolicyCostateError(f"live and EMA tensor metadata differ for {name}")
            if shadow.is_floating_point():
                shadow.mul_(decay).add_(live, alpha=1.0 - decay)
            else:
                shadow.copy_(live)
        self._num_updates.add_(1)

    def forward(
        self,
        current_frame: torch.Tensor,
        anchor_frame: torch.Tensor,
        anchor_costate: torch.Tensor,
    ) -> torch.Tensor:
        return self.provider_model(current_frame, anchor_frame, anchor_costate)


@dataclass(frozen=True)
class DenseFitAdmission:
    """Fit result where EMA-shadow improvement alone controls admission."""

    ema_initial_loss: float
    ema_final_loss: float
    live_initial_loss: float
    live_final_loss: float
    optimizer_steps: int
    dense_trajectory_samples: int
    ema_updates: int
    ema_improved: bool
    live_improved: bool
    admitted: bool
    authority: str = "ema_shadow_training_signal_loss_only"
    research_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dense_onpolicy_batch(samples: Sequence[OnPolicyTransition]) -> None:
    """Require a contiguous, ordered, exact-teacher-labeled trajectory window."""

    if not samples:
        raise OnPolicyCostateError("at least one dense on-policy transition is required")
    for sample in samples:
        sample.validate()
    steps = [sample.trajectory_step for sample in samples]
    first_step = steps[0]
    expected = list(range(first_step, first_step + len(steps)))
    if steps != expected:
        raise OnPolicyCostateError("dense on-policy samples must be ordered and step-contiguous")
    reference = samples[0]
    reference_shape = tuple(reference.current_frame.shape)
    reference_scope = (
        reference.custody.fingerprint_sha256,
        reference.custody.regime,
        reference.custody.pair_index,
        reference.custody.objective,
        reference.custody.schema,
    )
    for sample in samples[1:]:
        scope = (
            sample.custody.fingerprint_sha256,
            sample.custody.regime,
            sample.custody.pair_index,
            sample.custody.objective,
            sample.custody.schema,
        )
        if scope != reference_scope:
            raise OnPolicyCostateError(
                "dense on-policy samples must share one run/regime provider fingerprint and scope"
            )
        if tuple(sample.current_frame.shape) != reference_shape:
            raise OnPolicyCostateError("dense on-policy samples must share one tensor shape")


def _sample_loss(
    provider: nn.Module,
    sample: OnPolicyTransition,
    config: AmortizedCostateConfig,
) -> torch.Tensor:
    prediction = provider(sample.current_frame, sample.anchor_frame, sample.anchor_costate)
    scale = (
        sample.anchor_costate.square()
        .mean(dim=(1, 2, 3), keepdim=True)
        .sqrt()
        .clamp_min(config.normalization_floor)
    )
    residual_error = (prediction - sample.current_costate) / scale
    mse = residual_error.square().mean()
    prediction_flat = (prediction / scale).flatten(1)
    target_flat = (sample.current_costate / scale).flatten(1)
    # The mathematical debt is nonnegative. A large fp32 reduction can round
    # self-similarity slightly above one, so zero is its derived numeric floor.
    cosine_debt = (
        1.0
        - F.cosine_similarity(
            prediction_flat,
            target_flat,
            dim=1,
            eps=config.normalization_floor,
        ).mean()
    ).clamp_min(0.0)
    return config.mse_weight * mse + config.cosine_weight * cosine_debt


def dense_costate_loss(
    provider: nn.Module,
    samples: Sequence[OnPolicyTransition],
    *,
    config: AmortizedCostateConfig,
) -> float:
    """Measure a provider on the exact dense on-policy training-signal window."""

    config.validate()
    validate_dense_onpolicy_batch(samples)
    with torch.no_grad():
        loss = torch.stack([_sample_loss(provider, sample, config) for sample in samples]).mean()
    value = float(loss.item())
    if not math.isfinite(value):
        raise OnPolicyCostateError("costate provider loss is nonfinite")
    return value


def fit_dense_onpolicy_batch(
    live_model: AmortizedOnPolicyCostate,
    samples: Sequence[OnPolicyTransition],
    *,
    optimizer: torch.optim.Optimizer,
    optimizer_steps: int,
    ema_provider: EMACostateProvider,
) -> DenseFitAdmission:
    """Fit a dense student-owned window and gate using the EMA provider.

    Exact teacher labels must already be attached to every trajectory step.
    There is intentionally no API accepting frames/costates as an offline
    tensor dataset.  The caller owns the optimizer and serializes it together
    with both model state dictionaries.
    """

    if not isinstance(optimizer_steps, int) or optimizer_steps < 1:
        raise OnPolicyCostateError("optimizer_steps must be an integer >= 1")
    if live_model.config != ema_provider.config:
        raise OnPolicyCostateError("live and EMA-provider configs differ")
    live_parameter_ids = {id(parameter) for parameter in live_model.parameters() if parameter.requires_grad}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    if optimizer_parameter_ids != live_parameter_ids:
        raise OnPolicyCostateError("optimizer must own exactly the live model trainable parameters")
    validate_dense_onpolicy_batch(samples)
    live_initial = dense_costate_loss(live_model, samples, config=live_model.config)
    ema_initial = dense_costate_loss(ema_provider, samples, config=live_model.config)
    live_model.train()
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [_sample_loss(live_model, sample, live_model.config) for sample in samples]
        ).mean()
        if not bool(torch.isfinite(loss)):
            raise OnPolicyCostateError("live dense on-policy training loss is nonfinite")
        loss.backward()
        optimizer.step()
        ema_provider.update_from_live(live_model)
    live_model.eval()
    live_final = dense_costate_loss(live_model, samples, config=live_model.config)
    ema_final = dense_costate_loss(ema_provider, samples, config=live_model.config)
    minimum_fraction = live_model.config.admission_min_relative_improvement
    ema_improved = ema_final < ema_initial * (1.0 - minimum_fraction)
    live_improved = live_final < live_initial * (1.0 - minimum_fraction)
    return DenseFitAdmission(
        ema_initial_loss=ema_initial,
        ema_final_loss=ema_final,
        live_initial_loss=live_initial,
        live_final_loss=live_final,
        optimizer_steps=optimizer_steps,
        dense_trajectory_samples=len(samples),
        ema_updates=ema_provider.updates,
        ema_improved=ema_improved,
        live_improved=live_improved,
        admitted=ema_improved,
    )


def predict_ema_detached_costate(
    ema_provider: EMACostateProvider,
    *,
    current_frame: torch.Tensor,
    anchor_frame: torch.Tensor,
    anchor_costate: torch.Tensor,
) -> torch.Tensor:
    """Serve only the EMA-shadow costate, detached for renderer injection."""

    ema_provider.eval()
    with torch.no_grad():
        prediction = ema_provider(current_frame, anchor_frame, anchor_costate)
    prediction = prediction.clone().detach()
    if not bool(torch.isfinite(prediction).all()):
        raise OnPolicyCostateError("EMA surrogate emitted a nonfinite costate")
    return prediction


def inject_detached_costate(frame: torch.Tensor, costate: torch.Tensor) -> torch.Tensor:
    """Canonical ``<stop_gradient(lambda_hat), frame>`` chain-rule seam."""

    try:
        return costate_injection_loss_torch(frame, costate)
    except (TypeError, ValueError) as error:
        raise OnPolicyCostateError(str(error)) from error


def checkpoint_payload(
    live_model: AmortizedOnPolicyCostate,
    ema_provider: EMACostateProvider,
    optimizer: torch.optim.Optimizer,
    *,
    next_trajectory_step: int,
) -> dict[str, Any]:
    """Return all state needed for an atomic per-stage checkpoint."""

    if not isinstance(next_trajectory_step, int) or next_trajectory_step < 0:
        raise OnPolicyCostateError("next_trajectory_step must be an integer >= 0")
    if live_model.config != ema_provider.config:
        raise OnPolicyCostateError("live and EMA-provider configs differ")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config": live_model.config.to_dict(),
        # ``state_dict()`` tensors alias module storage.  Deep copies freeze
        # this payload at the stage boundary instead of silently following a
        # subsequent optimizer step before the atomic writer serializes it.
        "live_model": copy.deepcopy(live_model.state_dict()),
        "ema_provider": copy.deepcopy(ema_provider.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "next_trajectory_step": next_trajectory_step,
        "research_only": RESEARCH_ONLY,
        "authority_scope": AUTHORITY_SCOPE,
    }


def restore_checkpoint_payload(
    payload: Mapping[str, Any],
    live_model: AmortizedOnPolicyCostate,
    ema_provider: EMACostateProvider,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Fail-closed restore of live, EMA, optimizer, and trajectory position."""

    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise OnPolicyCostateError("unknown amortized costate checkpoint schema")
    if payload.get("authority_scope") != AUTHORITY_SCOPE or payload.get("research_only") is not True:
        raise OnPolicyCostateError("checkpoint attempts to change research-only authority scope")
    checkpoint_config = AmortizedCostateConfig.from_mapping(payload.get("config", {}))
    if checkpoint_config != live_model.config or checkpoint_config != ema_provider.config:
        raise OnPolicyCostateError("checkpoint config does not match the constructed provider")
    next_step = payload.get("next_trajectory_step")
    if not isinstance(next_step, int) or next_step < 0:
        raise OnPolicyCostateError("checkpoint next_trajectory_step is invalid")
    live_before = copy.deepcopy(live_model.state_dict())
    ema_before = copy.deepcopy(ema_provider.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    try:
        live_model.load_state_dict(payload["live_model"], strict=True)
        ema_provider.load_state_dict(payload["ema_provider"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
    except (KeyError, RuntimeError, ValueError) as error:
        live_model.load_state_dict(live_before, strict=True)
        ema_provider.load_state_dict(ema_before, strict=True)
        optimizer.load_state_dict(optimizer_before)
        raise OnPolicyCostateError("checkpoint state is incomplete or incompatible") from error
    return next_step


__all__ = [
    "AUTHORITY_SCOPE",
    "CHECKPOINT_SCHEMA",
    "RESEARCH_ONLY",
    "RGB_CHANNELS",
    "AmortizedCostateConfig",
    "AmortizedOnPolicyCostate",
    "DenseFitAdmission",
    "EMACostateProvider",
    "OnPolicyCostateError",
    "OnPolicyTransition",
    "ProviderCustody",
    "checkpoint_payload",
    "dense_costate_loss",
    "fit_dense_onpolicy_batch",
    "inject_detached_costate",
    "predict_ema_detached_costate",
    "restore_checkpoint_payload",
    "validate_dense_onpolicy_batch",
]
