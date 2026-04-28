"""Tuna-2 x-prediction reconstruction loss helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


SIGMA_MIN = 1e-3


def x_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    weighting: str = "v",
) -> torch.Tensor:
    """Mean-reduced x-prediction loss.

    ``weighting="x"`` is plain x-space MSE. ``weighting="v"`` applies the
    velocity-objective factor ``(1 + sigma^2) / sigma^2`` with sigma clamped
    away from zero for numerical stability.
    """
    if weighting not in {"v", "x"}:
        raise ValueError(f"unknown x_prediction_loss weighting: {weighting!r}")

    per_elem = F.mse_loss(pred, target, reduction="none")
    if weighting == "x":
        return per_elem.mean()

    sigma_t = torch.as_tensor(sigma, dtype=per_elem.dtype, device=per_elem.device)
    sigma_t = sigma_t.clamp(min=SIGMA_MIN)
    sigma2 = sigma_t * sigma_t
    weight = (1.0 + sigma2) / sigma2
    return (per_elem * weight).mean()
