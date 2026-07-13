# SPDX-License-Identifier: MIT
"""Nonlinear on-policy input-costate surrogate for frozen SegNet.

The surrogate predicts the *training signal* ``dL_teacher / d(frame)``.  It
does not report d_seg and it is never an evaluation authority.  At an exact
teacher anchor the caller supplies a current through-R frame and exact input
costate.  Between anchors the model uses the current frame together with the
last anchored frame/costate and emits a detached replacement costate for the
existing chain-rule injection seam.

Training samples are transitions produced by the witness trajectory itself;
there is intentionally no fixed/offline dataset API.  This is the minimum
mechanism needed to distinguish this arm from the already-falsified
forward/logit-only distillation formulation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS: Final[float] = 1.0e-12
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
RESEARCH_ONLY: Final[bool] = True
FULL_BUILD_BLOCKER: Final[str] = (
    "task #455 terminated NEEDS-MORE: every K20 arm and seven of nine regime/cadence arms blocked, "
    "so no common early/boundary/late cadence is admissible and no typed live-trainer integration exists"
)


class OnPolicyCostateError(ValueError):
    """Raised when on-policy or tensor-shape custody is violated."""


@dataclass(frozen=True)
class ProviderCustody:
    """Content-addressed authority for one on-policy teacher label."""

    fingerprint_sha256: str
    regime: str
    pair_index: int = 0
    objective: str = "exact_segnet_ce_input_costate_through_R"
    schema: str = "onpolicy_costate_provider.v1"

    def validate(self) -> None:
        if not _SHA256_RE.fullmatch(self.fingerprint_sha256):
            raise OnPolicyCostateError("provider custody fingerprint must be a lowercase SHA-256")
        if not self.regime:
            raise OnPolicyCostateError("provider custody regime is required")
        if self.pair_index != 0:
            raise OnPolicyCostateError("task #455 probe is scoped to pair_index=0")
        if self.objective != "exact_segnet_ce_input_costate_through_R":
            raise OnPolicyCostateError("provider custody objective is not the exact through-R CE costate")
        if self.schema != "onpolicy_costate_provider.v1":
            raise OnPolicyCostateError("unknown provider custody schema")


@dataclass(frozen=True)
class OnPolicyTransition:
    """One teacher-labeled transition from the witness's own trajectory."""

    anchor_frame: torch.Tensor
    anchor_costate: torch.Tensor
    current_frame: torch.Tensor
    current_costate: torch.Tensor
    trajectory_step: int
    custody: ProviderCustody
    source: str = "witness_through_r_exact_teacher_anchor"

    def validate(self) -> None:
        tensors = (self.anchor_frame, self.anchor_costate, self.current_frame, self.current_costate)
        shape = tuple(self.anchor_frame.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise OnPolicyCostateError(f"expected singleton NCHW RGB/costate tensors; got {shape}")
        if any(tuple(tensor.shape) != shape for tensor in tensors):
            raise OnPolicyCostateError("transition tensors must have identical shapes")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise OnPolicyCostateError("transition tensors must be finite")
        if self.trajectory_step < 0:
            raise OnPolicyCostateError("trajectory_step must be >= 0")
        self.custody.validate()
        if self.source != "witness_through_r_exact_teacher_anchor":
            raise OnPolicyCostateError("offline or synthetic transition sources are not admissible")


class NonlinearCostateSurrogate(nn.Module):
    """Tiny residual CNN predicting the current normalized input costate.

    The input has nine channels: current RGB, current-minus-anchor RGB, and
    the anchored exact costate.  A nonlinear residual stack predicts a
    correction to the anchored costate.  This makes the exact anchor a stable
    positive control while retaining nonlinear current-frame dependence.
    """

    def __init__(self, hidden_channels: int = 8) -> None:
        super().__init__()
        if not isinstance(hidden_channels, int) or hidden_channels < 4:
            raise OnPolicyCostateError("hidden_channels must be an integer >= 4")
        self.net = nn.Sequential(
            nn.Conv2d(9, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )
        # The untrained provider is exactly the anchored costate.  Learning is
        # required to move away from this cache baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _scale(anchor_costate: torch.Tensor) -> torch.Tensor:
        return anchor_costate.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(_EPS)

    def forward(
        self,
        current_frame: torch.Tensor,
        anchor_frame: torch.Tensor,
        anchor_costate: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            current_frame.ndim == 4
            and current_frame.shape[1] == 3
            and current_frame.shape == anchor_frame.shape == anchor_costate.shape
        ):
            raise OnPolicyCostateError("provider inputs must be identical-shape NCHW RGB/costate tensors")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in (current_frame, anchor_frame, anchor_costate)):
            raise OnPolicyCostateError("provider inputs must be finite")
        scale = self._scale(anchor_costate)
        current_unit = current_frame / 255.0
        delta_unit = (current_frame - anchor_frame) / 255.0
        anchor_unit = anchor_costate / scale
        correction = self.net(torch.cat((current_unit, delta_unit, anchor_unit), dim=1))
        return anchor_costate + scale * correction


@dataclass(frozen=True)
class FitResult:
    initial_loss: float
    final_loss: float
    steps: int
    finite: bool
    improved: bool
    ema_decay: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "steps": self.steps,
            "finite": self.finite,
            "improved": self.improved,
            "ema_decay": self.ema_decay,
        }


def _transition_loss(model: NonlinearCostateSurrogate, sample: OnPolicyTransition) -> torch.Tensor:
    sample.validate()
    prediction = model(sample.current_frame, sample.anchor_frame, sample.anchor_costate)
    scale = NonlinearCostateSurrogate._scale(sample.current_costate)
    pred_unit = prediction / scale
    target_unit = sample.current_costate / scale
    mse = F.mse_loss(pred_unit, target_unit)
    pred_flat = pred_unit.flatten(1)
    target_flat = target_unit.flatten(1)
    cosine_debt = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1, eps=_EPS).mean()
    return mse + cosine_debt


def fit_onpolicy_transitions(
    model: NonlinearCostateSurrogate,
    samples: Sequence[OnPolicyTransition],
    *,
    optimizer: torch.optim.Optimizer,
    steps: int,
    ema: Any,
) -> FitResult:
    """Fit only teacher-labeled on-trajectory transitions.

    ``steps`` is a caller-controlled stage budget.  The caller owns the
    optimizer state so checkpoints can preserve and resume it exactly.
    """

    if not samples:
        raise OnPolicyCostateError("at least one on-policy transition is required")
    if not isinstance(steps, int) or steps < 1:
        raise OnPolicyCostateError("steps must be an integer >= 1")
    if not hasattr(ema, "update") or not hasattr(ema, "state_dict"):
        raise OnPolicyCostateError("canonical EMA tracker is required")
    for sample in samples:
        sample.validate()
    model.train()
    with torch.no_grad():
        initial = float(torch.stack([_transition_loss(model, sample) for sample in samples]).mean().item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([_transition_loss(model, sample) for sample in samples]).mean()
        if not bool(torch.isfinite(loss)):
            return FitResult(
                initial, float("nan"), steps, finite=False, improved=False, ema_decay=float(ema.decay)
            )
        loss.backward()
        optimizer.step()
        ema.update(model)
    with torch.no_grad():
        final = float(torch.stack([_transition_loss(model, sample) for sample in samples]).mean().item())
    model.eval()
    return FitResult(
        initial_loss=initial,
        final_loss=final,
        steps=steps,
        finite=math.isfinite(final),
        improved=bool(math.isfinite(final) and final < initial),
        ema_decay=float(ema.decay),
    )


def predict_detached_costate(
    model: NonlinearCostateSurrogate,
    *,
    current_frame: torch.Tensor,
    anchor_frame: torch.Tensor,
    anchor_costate: torch.Tensor,
) -> torch.Tensor:
    """Run the cheap provider without retaining a surrogate backward graph."""

    model.eval()
    with torch.inference_mode():
        prediction = model(current_frame, anchor_frame, anchor_costate)
    # A tensor created under inference_mode cannot be saved by the renderer's
    # autograd graph.  Clone after leaving the context so the detached costate
    # remains legal as a constant in the chain-rule injection seam.
    prediction = prediction.clone()
    if not bool(torch.isfinite(prediction).all()):
        raise OnPolicyCostateError("surrogate emitted a nonfinite costate")
    return prediction.detach()


def whole_step_economics(*, cadence: int, t_exact_seconds: float, t_surrogate_seconds: float) -> dict[str, float]:
    """Compute the operator-specified anchored-cycle economics exactly."""

    if not isinstance(cadence, int) or cadence < 1:
        raise OnPolicyCostateError("cadence must be an integer >= 1")
    if not (math.isfinite(t_exact_seconds) and t_exact_seconds > 0.0):
        raise OnPolicyCostateError("t_exact_seconds must be finite > 0")
    if not (math.isfinite(t_surrogate_seconds) and t_surrogate_seconds >= 0.0):
        raise OnPolicyCostateError("t_surrogate_seconds must be finite >= 0")
    denominator = t_exact_seconds + (cadence - 1) * t_surrogate_seconds
    speedup = cadence * t_exact_seconds / denominator
    baseline = cadence * t_exact_seconds
    saved_fraction = 1.0 - denominator / baseline
    return {
        "cadence": float(cadence),
        "t_exact_seconds": t_exact_seconds,
        "t_surrogate_seconds": t_surrogate_seconds,
        "anchored_cycle_seconds": denominator,
        "all_exact_cycle_seconds": baseline,
        "speedup": speedup,
        "saved_fraction": saved_fraction,
        "exact_teacher_skip_fraction": 1.0 - 1.0 / cadence,
    }


__all__ = [
    "FULL_BUILD_BLOCKER",
    "RESEARCH_ONLY",
    "FitResult",
    "NonlinearCostateSurrogate",
    "OnPolicyCostateError",
    "OnPolicyTransition",
    "ProviderCustody",
    "fit_onpolicy_transitions",
    "predict_detached_costate",
    "whole_step_economics",
]
