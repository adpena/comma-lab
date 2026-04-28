"""Lane SAUG-V2: Cosmos HighSigmaStrategy-style input noise.

Sigmas are raw standard deviations in the same units as the tensor passed to
``apply_sigma_noise_to_input``. Callers with normalized images, uint8-scale
images, or categorical masks must scale sigmas before calling the helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HighSigmaStrategyConfig:
    redraw_fraction: float = 0.05
    high_sigma_min: float = 80.0
    high_sigma_max: float = 2000.0
    normal_sigma_min: float = 0.5
    normal_sigma_max: float = 80.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.redraw_fraction <= 1.0:
            raise ValueError(
                f"redraw_fraction must be in [0, 1], got {self.redraw_fraction}"
            )
        for name in (
            "high_sigma_min",
            "high_sigma_max",
            "normal_sigma_min",
            "normal_sigma_max",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive float, got {value}")
        if self.high_sigma_min > self.high_sigma_max:
            raise ValueError(
                "high_sigma_min must be <= high_sigma_max, got "
                f"{self.high_sigma_min} > {self.high_sigma_max}"
            )
        if self.normal_sigma_min > self.normal_sigma_max:
            raise ValueError(
                "normal_sigma_min must be <= normal_sigma_max, got "
                f"{self.normal_sigma_min} > {self.normal_sigma_max}"
            )


class HighSigmaSampler:
    """Sample per-example sigmas from normal and high-sigma log-uniform bands."""

    def __init__(
        self,
        config: HighSigmaStrategyConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator

    def _rand(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self.generator is None:
            return torch.rand(shape, device=device)
        generator_device = torch.device(getattr(self.generator, "device", "cpu"))
        values = torch.rand(shape, device=generator_device, generator=self.generator)
        return values.to(device=device)

    def sample_sigmas(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Return ``(batch_size,)`` sigmas with independent Bernoulli redraws.

        Each sample independently draws ``Bernoulli(redraw_fraction)``. High
        samples are log-uniform in ``[high_sigma_min, high_sigma_max]``; the
        rest are log-uniform in ``[normal_sigma_min, normal_sigma_max]``.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        device = torch.device(device)
        if not self.config.enabled:
            return torch.zeros(batch_size, device=device)

        high_draw = self._rand((batch_size,), device) < self.config.redraw_fraction
        u = self._rand((batch_size,), device)

        normal = _log_uniform(
            u,
            self.config.normal_sigma_min,
            self.config.normal_sigma_max,
        )
        high = _log_uniform(
            u,
            self.config.high_sigma_min,
            self.config.high_sigma_max,
        )
        return torch.where(high_draw, high, normal)


def _log_uniform(u: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    return torch.exp(u * (log_max - log_min) + log_min)


def apply_sigma_noise_to_input(x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Add per-sample Gaussian noise without in-place mutation.

    ``sigmas`` must have shape ``(B,)`` for an input whose first dimension is
    ``B``. The function preserves autograd through ``x``; the random noise is
    an additive constant for the backward pass.
    """
    if x.ndim == 0:
        raise ValueError("x must have a batch dimension")
    if not torch.is_floating_point(x):
        raise TypeError(
            "apply_sigma_noise_to_input expects a floating point tensor. "
            "Convert or scale categorical inputs at the call site."
        )
    if sigmas.ndim != 1 or sigmas.shape[0] != x.shape[0]:
        raise ValueError(
            f"sigmas must have shape ({x.shape[0]},), got {tuple(sigmas.shape)}"
        )

    sigma_view = sigmas.to(device=x.device, dtype=x.dtype).view(
        x.shape[0], *([1] * (x.ndim - 1))
    )
    noise = torch.randn_like(x) * sigma_view
    return x + noise
