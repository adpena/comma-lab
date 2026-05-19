# SPDX-License-Identifier: MIT
"""Per-pixel inverse-local-variance utility (UNIWARD-style).

UNIWARD (Holub, Fridrich, Denemark 2014 *"Universal distortion function for
steganography in an arbitrary domain"*) defines a per-pixel distortion as
the inverse of local image variance — embedding cost is HIGH in smooth
regions (where detectors notice changes) and LOW in textured regions
(where steganalysis blind spots live).

This module exposes one utility callable that maps a 2-D image patch to a
per-pixel cost map suitable as the SegNet-axis contribution to the unified
action — the unified-action SOLVER can then water-fill the cost map under
a per-image bit budget to maximise the score under the UNIWARD distortion.

Per the synthesis memo §"OTHER APPLICATIONS" #10: per-region UNIWARD
perturbation — water-filling on inverse-local-variance.

Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE — the autopilot
ranker consumes this utility when sweeping per-pixel perturbation budgets
via ``evaluate_with_water_filling``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (canonical alias)

_EPS = 1e-6


def per_pixel_inverse_variance_utility(
    theta: torch.Tensor,
    duals: Any = None,
    *,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Sum of per-pixel inverse-local-variance utility (UNIWARD-style).

    Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE.

    Computes the local variance via mean-of-squares minus square-of-mean in
    a sliding ``kernel_size × kernel_size`` window (canonical UNIWARD
    discrimination filter), then returns the sum of ``1/(var + eps)`` over
    the image — high in smooth regions (high cost / low utility), low in
    textured regions (low cost / high utility).

    The utility is sign-flipped relative to UNIWARD distortion: the
    unified action MINIMIZES the action, so a HIGH utility value here
    encodes a HIGH cost (smooth region) the solver wants to avoid; a LOW
    utility encodes a LOW cost (textured region) the solver should prefer.
    This sign convention matches the rest of the action: every
    contribution adds to the action and the variational solver descends.

    Args:
        theta: A 2-D ``torch.Tensor`` ``(H, W)`` of single-channel pixel
          intensities OR a 4-D batch ``(B, C, H, W)``. The function
          flattens batches and channels for the local-variance
          computation, then sums across all spatial positions.
        duals: Optional ``DualVariables``. Reads ``duals.lambda_seg``
          as the canonical per-pixel weight (defaults to 1.0 when absent).
        kernel_size: Sliding-window kernel size for local variance.
          Defaults to 5 (the canonical UNIWARD discrimination filter).
          Must be odd and ≥3 so the centre pixel is well-defined.

    Returns:
        A 0-D ``torch.Tensor`` carrying the sum of per-pixel
        inverse-local-variance utility values. Has autograd ``grad_fn``
        when ``theta.requires_grad``.
    """
    if not (kernel_size >= 3 and kernel_size % 2 == 1):
        raise ValueError(
            f"per_pixel_inverse_variance_utility: kernel_size must be odd "
            f"and >=3; got {kernel_size}"
        )
    if theta.ndim == 2:
        x = theta.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif theta.ndim == 4:
        x = theta
    else:
        raise ValueError(
            "per_pixel_inverse_variance_utility: theta must be 2-D (H, W) "
            f"or 4-D (B, C, H, W); got ndim={theta.ndim} shape={tuple(theta.shape)}"
        )
    if x.numel() == 0:
        return torch.zeros((), device=theta.device, dtype=theta.dtype)

    B, C, H, W = x.shape
    if H < kernel_size or W < kernel_size:
        raise ValueError(
            f"per_pixel_inverse_variance_utility: input ({H}, {W}) smaller "
            f"than kernel_size {kernel_size}; cannot compute local variance."
        )

    # Per-channel local-mean and local-mean-of-squares via avg_pool2d.
    pad = kernel_size // 2
    # Reshape (B, C, H, W) -> (B*C, 1, H, W) so avg_pool2d treats each
    # channel independently.
    x_flat = x.reshape(B * C, 1, H, W)
    local_mean = F.avg_pool2d(
        x_flat, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False
    )
    local_mean_sq = F.avg_pool2d(
        x_flat * x_flat,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
        count_include_pad=False,
    )
    local_var = torch.clamp(local_mean_sq - local_mean * local_mean, min=_EPS)
    utility = 1.0 / (local_var + _EPS)

    weight = 1.0
    if duals is not None and hasattr(duals, "lambda_seg"):
        weight = float(duals.lambda_seg)
        if not math.isfinite(weight):
            raise ValueError(
                f"per_pixel_inverse_variance_utility: duals.lambda_seg must "
                f"be finite; got {weight!r}"
            )
    return weight * utility.sum()


__all__ = ["per_pixel_inverse_variance_utility"]
