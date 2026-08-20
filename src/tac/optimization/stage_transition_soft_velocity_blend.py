# SPDX-License-Identifier: MIT
"""Stage-transition soft velocity blend arithmetic for FA1/FB1.

This module is deliberately pure optimizer-state algebra.  It does not read a
scorer, run a trainer, or claim that a soft blend improves a candidate.  The
only implemented mechanism is the named one:

    m_new(t) = (1 - alpha(t)) * m_mapped + alpha(t) * m_fresh(t)

where ``m_fresh`` is the ordinary Adam first moment accumulated from the recorded
post-boundary gradient sequence.  Callers must provide real arrays; missing
stage-boundary state or missing gradients are fail-closed by the replay custody
helpers rather than synthesized.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "StageTransitionSoftVelocityBlendConfig",
    "StageTransitionSoftVelocityBlendError",
    "SoftVelocityBlendStep",
    "alpha_at_step",
    "blend_momentum",
    "clip_to_rms",
    "fresh_adam_moment",
    "replay_custody_issues",
    "rms",
    "soft_velocity_blend_sequence",
    "window_steps_from_beta2",
]


class StageTransitionSoftVelocityBlendError(ValueError):
    """A fail-closed soft-blend contract violation."""


@dataclass(frozen=True, slots=True)
class StageTransitionSoftVelocityBlendConfig:
    """Configuration for the FA1 soft first-moment blend.

    ``window_steps`` is in optimizer steps, not epochs.  The default construction
    derives it from the Adam beta2 memory length ``ceil(c / (1 - beta2))``.
    """

    window_steps: int
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    shape: str = "linear"
    clip_rms: float | None = None

    def __post_init__(self) -> None:
        if int(self.window_steps) < 1:
            raise StageTransitionSoftVelocityBlendError(
                f"window_steps must be >= 1, got {self.window_steps!r}"
            )
        for name, value in (("alpha_start", self.alpha_start), ("alpha_end", self.alpha_end)):
            if not (0.0 <= float(value) <= 1.0) or not math.isfinite(float(value)):
                raise StageTransitionSoftVelocityBlendError(
                    f"{name} must be finite and in [0, 1], got {value!r}"
                )
        if self.shape not in ("linear", "cosine"):
            raise StageTransitionSoftVelocityBlendError(
                f"shape must be 'linear' or 'cosine', got {self.shape!r}"
            )
        if self.clip_rms is not None and not (math.isfinite(float(self.clip_rms))
                                             and float(self.clip_rms) > 0.0):
            raise StageTransitionSoftVelocityBlendError(
                f"clip_rms must be positive finite when set, got {self.clip_rms!r}"
            )

    @classmethod
    def from_beta2(
        cls,
        beta2: float = 0.999,
        *,
        c: float = 2.0,
        alpha_start: float = 0.0,
        alpha_end: float = 1.0,
        shape: str = "linear",
        clip_rms: float | None = None,
    ) -> "StageTransitionSoftVelocityBlendConfig":
        return cls(
            window_steps=window_steps_from_beta2(beta2, c=c),
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            shape=shape,
            clip_rms=clip_rms,
        )


@dataclass(frozen=True, slots=True)
class SoftVelocityBlendStep:
    """One replayed post-boundary optimizer step."""

    step: int
    alpha: float
    fresh_moment: np.ndarray
    blended_moment: np.ndarray


def window_steps_from_beta2(beta2: float, *, c: float = 2.0) -> int:
    """Return ``ceil(c / (1 - beta2))`` optimizer steps."""

    beta2 = float(beta2)
    c = float(c)
    if not (0.0 < beta2 < 1.0):
        raise StageTransitionSoftVelocityBlendError(f"beta2 must be in (0, 1), got {beta2!r}")
    if not (math.isfinite(c) and c > 0.0):
        raise StageTransitionSoftVelocityBlendError(f"c must be positive finite, got {c!r}")
    return int(math.ceil(c / (1.0 - beta2)))


def alpha_at_step(step: int, cfg: StageTransitionSoftVelocityBlendConfig) -> float:
    """One-indexed alpha schedule over ``cfg.window_steps``."""

    step = int(step)
    if step < 1:
        raise StageTransitionSoftVelocityBlendError(f"step must be >= 1, got {step!r}")
    if cfg.window_steps == 1:
        frac = 1.0
    else:
        frac = min(1.0, max(0.0, (step - 1) / float(cfg.window_steps - 1)))
    if cfg.shape == "cosine":
        frac = 0.5 - 0.5 * math.cos(math.pi * frac)
    return float(cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * frac)


def _finite_array(name: str, value: object, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise StageTransitionSoftVelocityBlendError(
            f"{name} shape {arr.shape} does not match required shape {shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise StageTransitionSoftVelocityBlendError(f"{name} contains non-finite values")
    return arr


def rms(value: object) -> float:
    arr = _finite_array("value", value)
    if arr.size == 0:
        raise StageTransitionSoftVelocityBlendError("rms requires a non-empty array")
    return float(np.sqrt(np.mean(np.square(arr))))


def clip_to_rms(value: object, max_rms: float | None) -> np.ndarray:
    arr = _finite_array("value", value)
    if max_rms is None:
        return arr
    max_rms = float(max_rms)
    if not (math.isfinite(max_rms) and max_rms > 0.0):
        raise StageTransitionSoftVelocityBlendError(
            f"max_rms must be positive finite when set, got {max_rms!r}"
        )
    current = rms(arr)
    if current == 0.0 or current <= max_rms:
        return arr
    return arr * (max_rms / current)


def blend_momentum(
    m_mapped: object,
    m_fresh: object,
    alpha: float,
    *,
    clip_rms: float | None = None,
) -> np.ndarray:
    """Apply ``(1-alpha) * m_mapped + alpha * m_fresh`` on real arrays."""

    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0) or not math.isfinite(alpha):
        raise StageTransitionSoftVelocityBlendError(f"alpha must be finite and in [0, 1], got {alpha!r}")
    mapped = _finite_array("m_mapped", m_mapped)
    fresh = _finite_array("m_fresh", m_fresh, shape=mapped.shape)
    return clip_to_rms((1.0 - alpha) * mapped + alpha * fresh, clip_rms)


def fresh_adam_moment(previous_fresh: object, gradient: object, *, beta1: float = 0.9) -> np.ndarray:
    """One Adam first-moment update using the recorded gradient vector."""

    beta1 = float(beta1)
    if not (0.0 <= beta1 < 1.0):
        raise StageTransitionSoftVelocityBlendError(f"beta1 must be in [0, 1), got {beta1!r}")
    prev = _finite_array("previous_fresh", previous_fresh)
    grad = _finite_array("gradient", gradient, shape=prev.shape)
    return beta1 * prev + (1.0 - beta1) * grad


def soft_velocity_blend_sequence(
    m_mapped: object,
    gradients: Sequence[object],
    cfg: StageTransitionSoftVelocityBlendConfig,
    *,
    beta1: float = 0.9,
    fresh_initial: object | None = None,
) -> tuple[SoftVelocityBlendStep, ...]:
    """Replay a recorded gradient sequence through the FA1 blend.

    This is intentionally useless without real recorded gradients; an empty
    sequence raises instead of returning a vacuous success.
    """

    mapped = _finite_array("m_mapped", m_mapped)
    if not gradients:
        raise StageTransitionSoftVelocityBlendError("gradients must contain at least one vector")
    fresh = (
        np.zeros_like(mapped, dtype=np.float64)
        if fresh_initial is None
        else _finite_array("fresh_initial", fresh_initial, shape=mapped.shape)
    )
    rows: list[SoftVelocityBlendStep] = []
    for step, gradient in enumerate(gradients, start=1):
        fresh = fresh_adam_moment(fresh, gradient, beta1=beta1)
        alpha = alpha_at_step(step, cfg)
        blended = blend_momentum(mapped, fresh, alpha, clip_rms=cfg.clip_rms)
        rows.append(SoftVelocityBlendStep(step, alpha, fresh.copy(), blended))
    return tuple(rows)


def replay_custody_issues(
    *,
    previous_mapped_m_available: bool,
    gradient_vectors_available: int,
    required_gradient_vectors: int,
    baseline_controls_available: bool,
) -> tuple[str, ...]:
    """Return fail-closed reasons that a corpus cannot support the FA1 backtest."""

    issues: list[str] = []
    if not previous_mapped_m_available:
        issues.append("missing previous mapped first-moment state at the stage boundary")
    if int(required_gradient_vectors) < 1:
        issues.append("required_gradient_vectors must be >= 1")
    elif int(gradient_vectors_available) < int(required_gradient_vectors):
        issues.append(
            "missing full recorded post-boundary gradient sequence "
            f"({int(gradient_vectors_available)}/{int(required_gradient_vectors)} vectors)"
        )
    if not baseline_controls_available:
        issues.append("missing matched cold/reset baseline-control arithmetic")
    return tuple(issues)
