# SPDX-License-Identifier: MIT
"""Per-tensor Shannon R(D) utility curve.

The canonical Shannon rate-distortion utility per Cover & Thomas Theorem
10.3.2 (Gaussian source): for a Gaussian source with variance σ², the
rate-distortion function is

    R(D) = 0.5 * log2(σ² / D)    for 0 < D ≤ σ²
         = 0                     for D > σ²

This module exposes one utility callable that returns the concave R(D)
curve evaluated at a per-tensor distortion estimate, which the unified
action consumes as the rate-axis contribution.

Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE — the autopilot
ranker consumes this utility as a candidate rate-axis treatment when
sweeping per-tensor bit budgets via ``evaluate_with_water_filling``.

Per the synthesis memo §"OTHER APPLICATIONS" #1: per-tensor archive bit
budget — water-filling on R(D) curve. This utility IS the R(D) curve in
that pairing.
"""
from __future__ import annotations

import math
from typing import Any

import torch

_EPS = 1e-12


def per_tensor_rate_distortion_utility(
    theta: torch.Tensor,
    duals: Any = None,
) -> torch.Tensor:
    """Concave Shannon R(D) utility curve evaluated on a per-tensor σ²/D.

    Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE.

    Treats ``theta`` as a 1-D tensor of per-tensor distortion estimates D
    and computes the closed-form Gaussian R(D) per Cover & Thomas 10.3.2:

        R(D) = 0.5 * log2(σ² / D)    when 0 < D ≤ σ²
             = 0                     when D > σ²

    with σ² fixed at ``var(theta)`` (the canonical reference variance over
    the input tensor). The returned scalar is the sum over the per-tensor
    rate contributions — concave in each element so the gradient is
    monotone-decreasing in D (high-distortion bytes are cheap; low-distortion
    bytes are expensive).

    Args:
        theta: A 1-D ``torch.Tensor`` of per-tensor distortion estimates D.
          Values must be > 0 (the closed-form is undefined at D=0). The
          function clamps internally to a small epsilon and emits the
          clamp-clipped sum rather than raising.
        duals: Optional ``DualVariables`` from the unified action. The
          per-tensor rate term reads ``duals.lambda_rate`` as a uniform
          scalar multiplier when present; otherwise uses 1.0.

    Returns:
        A 0-D ``torch.Tensor`` carrying the sum of per-tensor R(D)
        contributions. Has autograd ``grad_fn`` when ``theta.requires_grad``.
    """
    if theta.ndim != 1:
        raise ValueError(
            f"per_tensor_rate_distortion_utility: theta must be 1-D; got "
            f"ndim={theta.ndim} shape={tuple(theta.shape)}"
        )
    if theta.numel() == 0:
        return torch.zeros((), device=theta.device, dtype=theta.dtype)

    sigma2 = theta.detach().var(unbiased=False) + _EPS
    D = torch.clamp(theta, min=_EPS)
    # R(D) = 0.5 * log2(sigma^2 / D) when D ≤ sigma^2; clamp to 0 otherwise.
    ratio = sigma2 / D
    raw = 0.5 * torch.log2(torch.clamp(ratio, min=1.0))  # log2(1)=0 at the floor
    weight = 1.0
    if duals is not None and hasattr(duals, "lambda_rate"):
        weight = float(duals.lambda_rate)
        if not math.isfinite(weight):
            raise ValueError(
                f"per_tensor_rate_distortion_utility: duals.lambda_rate must "
                f"be finite; got {weight!r}"
            )
    return weight * raw.sum()


__all__ = ["per_tensor_rate_distortion_utility"]
