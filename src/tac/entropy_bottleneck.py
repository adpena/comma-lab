# SPDX-License-Identifier: MIT
"""Entropy bottleneck rate model for renderer training.

Implements a lightweight factorized entropy model inspired by Ballé et al.
without depending on CompressAI. The module is intended for differentiable
rate regularization during training; it returns a quantized/noisy tensor and
the mean estimated bits per element.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EntropyBottleneck(nn.Module):
    """Per-channel logistic CDF entropy bottleneck.

    Args:
        num_channels: channel count of the latent tensor at dimension 1.
        init_scale: initial logistic scale. Larger values start with a broad
            prior and avoid over-penalizing early training activations.
    """

    def __init__(self, num_channels: int, init_scale: float = 10.0):
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")

        self.num_channels = int(num_channels)
        self.loc = nn.Parameter(torch.zeros(num_channels))
        raw_scale = math.log(math.expm1(float(init_scale)))
        self.raw_scale = nn.Parameter(torch.full((num_channels,), raw_scale))
        # Initialize raw_shape so softplus(raw_shape) ≈ 1.0 at construction.
        # Previously raw_shape inited to 0 → softplus ≈ 0.693 with the +1e-6
        # epsilon dominating after the (-50.0) collapse path; combined with
        # `((x − loc)/scale) * shape` this saturated the CDF at init and made
        # the rate loss meaningless until shape drifted away from zero.  Pure
        # logistic CDF (Ballé 2018) corresponds to shape = 1, so we anchor the
        # parameterization there: log(expm1(1.0)) ≈ 0.5413 (R17 finding 2).
        raw_shape_init = math.log(math.expm1(1.0))
        self.raw_shape = nn.Parameter(torch.full((num_channels,), raw_shape_init))
        self._last_bits_per_element: Tensor | None = None

    def _channel_params(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if y.ndim < 2:
            raise ValueError(f"expected channel dimension at dim=1, got shape {tuple(y.shape)}")
        if y.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels, got {y.shape[1]}"
            )
        view_shape = (1, self.num_channels) + (1,) * (y.ndim - 2)
        loc = self.loc.view(view_shape).to(dtype=y.dtype, device=y.device)
        scale = (F.softplus(self.raw_scale) + 1e-6).view(view_shape).to(
            dtype=y.dtype, device=y.device,
        )
        shape = (F.softplus(self.raw_shape) + 1e-6).view(view_shape).to(
            dtype=y.dtype, device=y.device,
        )
        return loc, scale, shape

    def _cdf(self, x: Tensor, loc: Tensor, scale: Tensor, shape: Tensor) -> Tensor:
        # Positive shape keeps the CDF monotone while giving each channel a
        # learnable slope correction around its location.
        return torch.sigmoid(((x - loc) / scale) * shape)

    def forward(self, y: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale, shape = self._channel_params(y)
        if self.training:
            noise = torch.empty_like(y).uniform_(-0.5, 0.5)
            y_hat = y + noise
        else:
            y_hat = torch.round(y)

        upper = self._cdf(y_hat + 0.5, loc, scale, shape)
        lower = self._cdf(y_hat - 0.5, loc, scale, shape)
        likelihood = (upper - lower).clamp(min=1e-9)
        bits_per_element = -torch.log2(likelihood).mean()
        self._last_bits_per_element = bits_per_element
        return y_hat, bits_per_element

    def rate_loss(self) -> Tensor:
        """Return the mean bits per element from the most recent forward."""
        if self._last_bits_per_element is None:
            return self.loc.sum() * 0.0
        return self._last_bits_per_element
