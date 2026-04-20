"""Feature matching loss on intermediate scorer activations.

Provides richer gradients than output-only MSE by matching activations at
multiple depths of the PoseNet encoder. The top-3 layers account for ~47%
of activation distance between GT and rendered frames.

Usage::

    from tac.feature_matching import compute_feature_matching_loss

    loss = compute_feature_matching_loss(
        rendered_frames, gt_frames, posenet,
        layer_names=["stages.3.blocks.1.mlp.fc2", ...],
    )
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["compute_feature_matching_loss", "get_top_posenet_layers"]

# Top PoseNet layers by parameter count / activation distance.
# These are the deepest, largest conv layers in the vision encoder that
# capture the most discriminative features for ego-motion estimation.
# Prefix: "vision." because PoseNet wraps the encoder as posenet.vision.*
DEFAULT_POSENET_LAYERS = [
    "vision.stages.3.blocks.1.mlp.fc2",   # 787K params, highest activation distance
    "vision.stages.3.blocks.0.mlp.fc2",   # 787K params, second highest
    "vision.stages.2.blocks.5.mlp.fc2",   # Deep stage-2 features
]

# Default weights: proportional to layer importance (deeper = more semantic)
DEFAULT_POSENET_LAYER_WEIGHTS = [1.0, 0.8, 0.6]


def get_top_posenet_layers() -> list[str]:
    """Return the top-3 PoseNet layer names for feature matching."""
    return DEFAULT_POSENET_LAYERS.copy()


def _resolve_module(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    """Resolve a dotted module path like 'stages.3.blocks.1.mlp.fc2'."""
    module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def compute_feature_matching_loss(
    rendered_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    scorer: torch.nn.Module,
    layer_names: list[str] | None = None,
    weights: list[float] | None = None,
    batch_size: int = 16,
    normalize: bool = True,
) -> torch.Tensor:
    """L2 loss on intermediate scorer activations between rendered and GT frames.

    Hooks the specified layers, runs both rendered and GT through the scorer,
    then computes weighted L2 distance at each hooked layer. This provides
    RICHER gradients than output-only MSE (6D pose -> now matching at multiple
    depths with thousands of dimensions).

    Args:
        rendered_frames: (N, H, W, 3) float [0, 255] tensor, requires_grad.
        gt_frames: (N, H, W, 3) float [0, 255] tensor (ground truth).
        scorer: PoseNet model (must have preprocess_input method).
        layer_names: list of dotted module paths to hook. If None, uses
            the default top-3 PoseNet layers from scorer internals analysis.
        weights: per-layer weight multipliers. If None, uses default weights.
        batch_size: frames per forward pass to avoid OOM.  NOTE: this only
            controls the peak memory of the *forward pass*. All per-batch
            losses are accumulated into ``layer_losses`` tensors before the
            final backward call, so total graph memory still scales with N.
            A true per-batch backward (zero_grad each iteration) would require
            a different API and is not implemented here.
        normalize: if True, normalize activations before computing L2 distance
            (makes the loss scale-invariant across layers).

    Returns:
        Scalar loss (weighted mean of per-layer L2 distances).
    """
    if layer_names is None:
        layer_names = DEFAULT_POSENET_LAYERS
    if weights is None:
        weights = DEFAULT_POSENET_LAYER_WEIGHTS[:len(layer_names)]

    device = rendered_frames.device
    N = rendered_frames.shape[0]
    P = N // 2  # PoseNet operates on pairs

    # Accumulate per-layer losses across batches.
    # WARNING: accumulated tensors retain their computation graphs, so total
    # backward-pass memory is O(N), not O(batch_size).  If OOM occurs during
    # backward, reduce the number of layer_names or call .backward() inside the
    # loop (requires restructuring this function).
    layer_losses = {name: torch.tensor(0.0, device=device) for name in layer_names}
    n_batches = 0

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch_indices = []
        for p in range(start, end):
            batch_indices.extend([2 * p, 2 * p + 1])

        rendered_batch = rendered_frames[batch_indices]  # (2*B, H, W, 3)
        gt_batch = gt_frames[batch_indices]

        # Format for PoseNet: (B, T=2, C, H, W) -- consecutive pairs
        B = end - start
        rendered_pairs = rendered_batch.reshape(B, 2, *rendered_batch.shape[1:])
        rendered_pairs = rendered_pairs.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, 3, H, W)
        gt_pairs = gt_batch.reshape(B, 2, *gt_batch.shape[1:])
        gt_pairs = gt_pairs.permute(0, 1, 4, 2, 3).contiguous()

        # Preprocess (handles YUV conversion etc.)
        rendered_input = scorer.preprocess_input(rendered_pairs)
        gt_input = scorer.preprocess_input(gt_pairs)

        # Register hooks
        hooks = []
        activations: dict[str, torch.Tensor] = {}

        def _make_hook(name: str):
            def hook_fn(module: Any, input: Any, output: torch.Tensor) -> None:
                activations[name] = output
            return hook_fn

        for name in layer_names:
            try:
                module = _resolve_module(scorer, name)
                hooks.append(module.register_forward_hook(_make_hook(name)))
            except (AttributeError, IndexError, TypeError) as e:
                logger.warning("Cannot hook layer %r: %s", name, e)
                continue

        # Forward rendered (gradients flow back through this)
        scorer(rendered_input)
        rendered_acts = {k: v for k, v in activations.items()}
        activations.clear()

        # Forward GT (no grad needed)
        with torch.no_grad():
            scorer(gt_input)
        gt_acts = {k: v for k, v in activations.items()}

        # Remove hooks immediately
        for h in hooks:
            h.remove()

        # Compute per-layer L2 distance
        for name in layer_names:
            if name not in rendered_acts or name not in gt_acts:
                continue
            r_act = rendered_acts[name]
            g_act = gt_acts[name].detach()  # GT acts should not accumulate grad

            if normalize:
                # L2-normalize along channel dim for scale invariance
                r_act = F.normalize(r_act.flatten(1), dim=1)
                g_act = F.normalize(g_act.flatten(1), dim=1)
                layer_losses[name] = layer_losses[name] + F.mse_loss(r_act, g_act)
            else:
                layer_losses[name] = layer_losses[name] + F.mse_loss(r_act, g_act)

        n_batches += 1

    # Weighted sum of per-layer losses
    total_loss = torch.tensor(0.0, device=device)
    for i, name in enumerate(layer_names):
        w = weights[i] if i < len(weights) else 1.0
        total_loss = total_loss + w * layer_losses[name] / max(n_batches, 1)

    return total_loss
