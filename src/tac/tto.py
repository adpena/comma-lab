"""Test-Time Optimization (TTO) for task-aware codec post-filters.

Adapts a pre-trained postfilter model to specific test video content at
inflation time by running a few gradient steps before producing output.
This is a self-supervised technique — no labels or scorer networks needed
at test time.

Loss options:
  - temporal_consistency: minimize frame-to-frame difference after filtering
    (encourages temporally smooth output, helps PoseNet)
  - reconstruction: denoising autoencoder on compressed frames
    (makes the model more robust to compression artifacts)
  - edge_preservation: Sobel edge alignment between input and output
    (preserves structural information both scorers care about)

Safety:
  - Wall-clock budget: aborts early if approaching time limit
  - Gradient clipping: prevents catastrophic weight updates
  - Only adapts a subset of parameters (last layer or BN stats)
  - Saves original weights and can restore if quality degrades

Usage::

    from tac.tto import test_time_optimize
    model = load_postfilter_int8("postfilter_int8.pt")
    model = test_time_optimize(model, frames, n_steps=10)
    # model is now adapted to this specific content
"""
from __future__ import annotations

import copy
import sys
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Loss functions (self-supervised, no scorer needed) ---- #


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for (B, C, H, W) tensor.

    Returns (B, C, H-2, W-2) edge magnitudes.
    """
    # Sobel kernels
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0).unsqueeze(0)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).unsqueeze(0).unsqueeze(0)

    B, C, H, W = x.shape
    # Process each channel independently
    x_flat = x.reshape(B * C, 1, H, W)
    gx = F.conv2d(x_flat, kx, padding=0)
    gy = F.conv2d(x_flat, ky, padding=0)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    return mag.reshape(B, C, H - 2, W - 2)


def temporal_consistency_loss(
    model: nn.Module,
    frames_bchw: torch.Tensor,
) -> torch.Tensor:
    """Minimize difference between filtered consecutive frames.

    For each consecutive pair (t, t+1), we want:
      ||model(frame_t) - model(frame_{t+1})|| to be small
    relative to the input frame difference. This encourages
    temporally smooth corrections — important for PoseNet
    which evaluates frame pairs for ego-motion.

    Args:
        model: postfilter model (B, 3, H, W) -> (B, 3, H, W)
        frames_bchw: (N, 3, H, W) float [0, 255] decoded frames
    """
    N = frames_bchw.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=frames_bchw.device, requires_grad=True)

    filtered = model(frames_bchw)
    # Temporal diff in output vs input
    out_diff = (filtered[1:] - filtered[:-1]).pow(2).mean(dim=(1, 2, 3))
    in_diff = (frames_bchw[1:] - frames_bchw[:-1]).pow(2).mean(dim=(1, 2, 3)).detach()

    # We want the output temporal diff to be <= input temporal diff.
    # Loss = max(0, out_diff - in_diff) averaged over pairs
    # Plus a small L2 on the absolute output diff for smoothness.
    excess = F.relu(out_diff - in_diff)
    smoothness = out_diff * 0.01  # light encouragement for temporal smoothness

    return (excess + smoothness).mean()


def reconstruction_loss(
    model: nn.Module,
    frames_bchw: torch.Tensor,
    noise_std: float = 3.0,
) -> torch.Tensor:
    """Denoising autoencoder loss: model should be robust to small perturbations.

    Apply small Gaussian noise, then minimize:
      ||model(frame + noise) - model(frame)||
    This makes the model more stable and robust to compression artifacts.

    Args:
        model: postfilter model
        frames_bchw: (N, 3, H, W) float [0, 255]
        noise_std: standard deviation of Gaussian noise
    """
    with torch.no_grad():
        clean_out = model(frames_bchw)

    noise = torch.randn_like(frames_bchw) * noise_std
    noisy_in = (frames_bchw + noise).clamp(0, 255)
    noisy_out = model(noisy_in)

    return (noisy_out - clean_out.detach()).pow(2).mean()


def edge_preservation_loss(
    model: nn.Module,
    frames_bchw: torch.Tensor,
) -> torch.Tensor:
    """Minimize Sobel edge difference between input and output.

    Preserves structural information that both PoseNet (ego-motion from
    feature matching) and SegNet (class boundaries) care about.

    Args:
        model: postfilter model
        frames_bchw: (N, 3, H, W) float [0, 255]
    """
    filtered = model(frames_bchw)
    in_edges = _sobel_edges(frames_bchw).detach()
    out_edges = _sobel_edges(filtered)

    return (out_edges - in_edges).pow(2).mean()


# ---- Parameter selection ---- #


def _select_params(
    model: nn.Module,
    param_mode: str = "last_layer",
) -> list[nn.Parameter]:
    """Select which parameters to optimize during TTO.

    Modes:
      - last_layer: only the final conv layer weights + bias
      - bn_only: only BatchNorm parameters (if any)
      - all: all parameters (most aggressive, highest risk)

    Args:
        model: the postfilter model
        param_mode: one of "last_layer", "bn_only", "all"

    Returns:
        list of parameters to optimize
    """
    if param_mode == "all":
        return [p for p in model.parameters() if p.requires_grad]

    if param_mode == "bn_only":
        params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                params.extend(p for p in m.parameters() if p.requires_grad)
        if not params:
            # Fallback: if no BN layers, use last conv
            return _select_params(model, "last_layer")
        return params

    # last_layer: find the last Conv2d or Linear and return its params
    last_conv = None
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            last_conv = m
    if last_conv is not None:
        return [p for p in last_conv.parameters() if p.requires_grad]

    # Fallback: all params
    return [p for p in model.parameters() if p.requires_grad]


# ---- Main TTO entry point ---- #


_LOSS_FNS = {
    "temporal_consistency": temporal_consistency_loss,
    "reconstruction": reconstruction_loss,
    "edge_preservation": edge_preservation_loss,
}


def test_time_optimize(
    model: nn.Module,
    frames: torch.Tensor,
    n_steps: int = 10,
    lr: float = 1e-4,
    loss_type: Literal[
        "temporal_consistency", "reconstruction", "edge_preservation"
    ] = "temporal_consistency",
    param_mode: Literal["last_layer", "bn_only", "all"] = "last_layer",
    grad_clip: float = 0.5,
    time_budget_seconds: float = 60.0,
    batch_size: int = 16,
    quality_check: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """Adapt model to specific test content at inflation time.

    Runs a few gradient steps using self-supervised losses (no scorer needed).
    Must complete within time_budget_seconds to stay within the 30-min eval limit.

    Args:
        model: pre-trained postfilter model (nn.Module)
        frames: decoded compressed frames, either:
            - (N, H, W, 3) uint8/float HWC format, or
            - (N, 3, H, W) float CHW format
        n_steps: number of gradient steps (default 10)
        lr: learning rate for adaptation (default 1e-4)
        loss_type: self-supervised loss function to use
        param_mode: which parameters to adapt ("last_layer", "bn_only", "all")
        grad_clip: max gradient norm (prevents catastrophic updates)
        time_budget_seconds: wall-clock time limit
        batch_size: frames per optimization step
        quality_check: if True, restore original weights if loss increases
        verbose: print progress to stderr

    Returns:
        The adapted model (same object, modified in-place).
        If quality_check detects degradation, original weights are restored.
    """
    t0 = time.monotonic()

    # Convert HWC -> CHW if needed
    if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2).float()
    elif frames.ndim == 4 and frames.shape[1] == 3:
        frames = frames.float()
    else:
        raise ValueError(
            f"Expected frames shape (N, H, W, 3) or (N, 3, H, W), got {frames.shape}"
        )

    frames = frames.clamp(0, 255)
    device = next(model.parameters()).device
    frames = frames.to(device)

    # Save original weights for quality check / rollback
    original_state = copy.deepcopy(model.state_dict())

    # Select parameters and create optimizer
    model.train()
    params = _select_params(model, param_mode)
    if not params:
        if verbose:
            print("TTO: no trainable parameters found, skipping", file=sys.stderr)
        model.eval()
        return model

    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = _LOSS_FNS[loss_type]

    N = frames.shape[0]
    losses = []

    for step in range(n_steps):
        # Check time budget
        elapsed = time.monotonic() - t0
        if elapsed >= time_budget_seconds:
            if verbose:
                print(
                    f"TTO: time budget exhausted at step {step}/{n_steps} "
                    f"({elapsed:.1f}s)",
                    file=sys.stderr,
                )
            break

        # Sample a batch (sliding window for temporal, random for others)
        if loss_type == "temporal_consistency":
            # Use consecutive frames for temporal loss
            start = (step * batch_size) % max(1, N - 1)
            end = min(start + batch_size, N)
            batch = frames[start:end]
            if batch.shape[0] < 2:
                batch = frames[:min(batch_size, N)]
        else:
            # Random sample for other losses
            indices = torch.randperm(N, device="cpu")[:batch_size]
            batch = frames[indices]

        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(params, grad_clip)

        optimizer.step()
        losses.append(loss.item())

        if verbose and (step + 1) % max(1, n_steps // 5) == 0:
            print(
                f"TTO step {step + 1}/{n_steps}: loss={loss.item():.6f} "
                f"({time.monotonic() - t0:.1f}s)",
                file=sys.stderr,
            )

    model.eval()

    # Quality check: ensure adaptation didn't degrade
    if quality_check and len(losses) >= 2:
        # If final loss is worse than initial, roll back
        if losses[-1] > losses[0] * 1.5:
            if verbose:
                print(
                    f"TTO: quality check FAILED (loss {losses[0]:.6f} -> "
                    f"{losses[-1]:.6f}), restoring original weights",
                    file=sys.stderr,
                )
            model.load_state_dict(original_state)

    elapsed = time.monotonic() - t0
    if verbose:
        loss_str = f"{losses[0]:.6f} -> {losses[-1]:.6f}" if losses else "N/A"
        print(
            f"TTO complete: {len(losses)} steps, loss {loss_str}, "
            f"{elapsed:.1f}s elapsed",
            file=sys.stderr,
        )

    return model


# Prevent pytest from collecting this function as a test
test_time_optimize.__test__ = False  # type: ignore[attr-defined]
