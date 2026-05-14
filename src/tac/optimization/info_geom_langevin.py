# SPDX-License-Identifier: MIT
"""Canonical information-geometric Langevin primitives.

This module is a small public facade over the existing
``tac.optimization.iglt.IGLTOptimizer`` implementation. It keeps trainer-facing
configuration, pure Fisher-diagonal helpers, and schedule inspection in one
place without duplicating the optimizer state machine.

These primitives do not create score authority. They are optimization tools for
training/export candidates whose archive/runtime packets still require exact
contest evaluation before any score or promotion claim.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import Any, Literal

import torch

from tac.optimization.iglt import FISHER_ESTIMATION_MODES, IGLTOptimizer
from tac.optimization.langevin_optimizer import SCHEDULES

FisherEstimationMode = Literal["diagonal", "block_diagonal", "kfac"]
PreconditionerPower = Literal["inverse_sqrt", "inverse"]


@dataclasses.dataclass(frozen=True)
class InfoGeomLangevinConfig:
    """Typed config for Fisher-preconditioned Langevin training.

    Args mirror :class:`tac.optimization.iglt.IGLTOptimizer`. The default mode
    is diagonal empirical Fisher because it is CPU-testable, deterministic under
    ``noise_seed``, and cheap enough to wire into arbitrary trainers.
    """

    lr: float = 1e-4
    T_init: float = 1.0
    T_final: float = 1e-4
    n_steps: int = 2000
    weight_decay: float = 0.0
    schedule: str = "cosine"
    noise_seed: int | None = None
    fisher_estimation: FisherEstimationMode = "diagonal"
    fisher_decay: float = 0.99
    fisher_eps: float = 1e-8
    warmup_steps: int = 10

    def validate(self) -> None:
        """Raise ``ValueError`` if the config is internally inconsistent."""

        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not math.isfinite(float(self.lr)):
            raise ValueError("lr must be finite")
        if self.T_init < 0.0 or self.T_final < 0.0:
            raise ValueError("temperatures must be non-negative")
        if not math.isfinite(float(self.T_init)) or not math.isfinite(
            float(self.T_final)
        ):
            raise ValueError("temperatures must be finite")
        if self.T_init < self.T_final:
            raise ValueError("T_init must be >= T_final")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.weight_decay < 0.0 or not math.isfinite(float(self.weight_decay)):
            raise ValueError("weight_decay must be non-negative and finite")
        if self.schedule not in SCHEDULES:
            raise ValueError(
                f"unknown schedule {self.schedule!r}; must be one of {sorted(SCHEDULES)}"
            )
        if self.fisher_estimation not in FISHER_ESTIMATION_MODES:
            raise ValueError(
                f"unknown fisher_estimation {self.fisher_estimation!r}; "
                f"must be one of {list(FISHER_ESTIMATION_MODES)}"
            )
        if not 0.0 < self.fisher_decay < 1.0 or not math.isfinite(
            float(self.fisher_decay)
        ):
            raise ValueError("fisher_decay must be in (0, 1)")
        if self.fisher_eps <= 0.0 or not math.isfinite(float(self.fisher_eps)):
            raise ValueError("fisher_eps must be positive and finite")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

    def to_optimizer_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by :class:`IGLTOptimizer`."""

        self.validate()
        return dataclasses.asdict(self)

    def temperature_at(self, step: int) -> float:
        """Evaluate the configured annealing schedule at ``step``."""

        self.validate()
        if step < 0:
            raise ValueError("step must be non-negative")
        return float(
            SCHEDULES[self.schedule](step, self.n_steps, self.T_init, self.T_final)
        )


InfoGeomLangevinOptimizer = IGLTOptimizer
"""Backward-compatible canonical optimizer name for information-geometric SGLD."""


def build_info_geom_langevin_optimizer(
    params: Iterable[Any],
    config: InfoGeomLangevinConfig | None = None,
    **overrides: Any,
) -> IGLTOptimizer:
    """Construct the canonical information-geometric Langevin optimizer.

    ``overrides`` are applied with :func:`dataclasses.replace`, so callers can
    keep one immutable config and specialize a trainer without mutation.
    """

    cfg = config if config is not None else InfoGeomLangevinConfig()
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return IGLTOptimizer(params, **cfg.to_optimizer_kwargs())


def fisher_diagonal_ema(
    previous: torch.Tensor | None,
    gradient: torch.Tensor,
    *,
    decay: float = 0.99,
) -> torch.Tensor:
    """Update an empirical Fisher diagonal with ``EMA[g ** 2]``.

    The returned tensor is detached from autograd and does not mutate
    ``previous``. This makes it suitable for trainer-side diagnostics and unit
    tests outside the optimizer class.
    """

    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")
    if not torch.is_floating_point(gradient):
        raise ValueError("gradient must be a floating-point tensor")
    if not torch.isfinite(gradient).all():
        raise ValueError("gradient must be finite")
    grad = gradient.detach()
    if previous is None:
        prev = torch.zeros_like(grad)
    else:
        if previous.shape != grad.shape:
            raise ValueError("previous and gradient shapes must match")
        if not torch.is_floating_point(previous):
            raise ValueError("previous must be a floating-point tensor")
        if not torch.isfinite(previous).all():
            raise ValueError("previous must be finite")
        prev = previous.detach().to(device=grad.device, dtype=grad.dtype)
    return prev.mul(decay).addcmul(grad, grad, value=1.0 - decay)


def precondition_gradient(
    gradient: torch.Tensor,
    fisher_diagonal: torch.Tensor,
    *,
    eps: float = 1e-8,
    power: PreconditionerPower = "inverse_sqrt",
) -> torch.Tensor:
    """Apply a Fisher-diagonal natural-gradient-like preconditioner.

    ``inverse_sqrt`` matches the canonical IGLT optimizer's RMS/Sophia-style
    direction ``g / (sqrt(F) + eps)``. ``inverse`` is exposed for diagnostics
    that need the stricter natural-gradient direction ``g / (F + eps)``.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if gradient.shape != fisher_diagonal.shape:
        raise ValueError("gradient and fisher_diagonal shapes must match")
    if not torch.is_floating_point(gradient) or not torch.is_floating_point(
        fisher_diagonal
    ):
        raise ValueError("gradient and fisher_diagonal must be floating-point")
    if not torch.isfinite(gradient).all():
        raise ValueError("gradient must be finite")
    if not torch.isfinite(fisher_diagonal).all():
        raise ValueError("fisher_diagonal must be finite")
    if torch.any(fisher_diagonal < 0):
        raise ValueError("fisher_diagonal must be non-negative")
    if power == "inverse_sqrt":
        return gradient / (fisher_diagonal.sqrt() + eps)
    if power == "inverse":
        return gradient / (fisher_diagonal + eps)
    raise ValueError("power must be 'inverse_sqrt' or 'inverse'")


__all__ = [
    "FisherEstimationMode",
    "InfoGeomLangevinConfig",
    "InfoGeomLangevinOptimizer",
    "PreconditionerPower",
    "build_info_geom_langevin_optimizer",
    "fisher_diagonal_ema",
    "precondition_gradient",
]
