# SPDX-License-Identifier: MIT
"""Portable loss functions with MLX + PyTorch sister implementations.

Per OVERNIGHT-WW: Selfcomp grayscale_lut uses MSE loss + an L1-style
component; this module provides canonical MSE + cross-entropy + L1
primitives.
"""

from __future__ import annotations

from typing import Any

from tac.portable_primitives.backend import Backend, resolve_backend

__all__ = [
    "mse_loss",
    "l1_loss",
    "cross_entropy_loss",
]


def mse_loss(predictions: Any, targets: Any, *, backend: Backend | str) -> Any:
    """Mean squared error loss: mean((pred - target)^2)."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        diff = predictions - targets
        return mx.mean(diff * diff)
    import torch.nn.functional as F

    return F.mse_loss(predictions, targets)


def l1_loss(predictions: Any, targets: Any, *, backend: Backend | str) -> Any:
    """Mean absolute error loss: mean(|pred - target|)."""
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        return mx.mean(mx.abs(predictions - targets))
    import torch.nn.functional as F

    return F.l1_loss(predictions, targets)


def cross_entropy_loss(
    logits: Any,
    targets: Any,
    *,
    backend: Backend | str,
) -> Any:
    """Cross-entropy loss for classification.

    Args:
        logits: (B, num_classes) tensor of unnormalized scores
        targets: (B,) long tensor of class indices
    """
    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx
        import mlx.nn as nn

        return mx.mean(nn.losses.cross_entropy(logits, targets))
    import torch.nn.functional as F

    return F.cross_entropy(logits, targets)
