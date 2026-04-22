"""Fridrich-inspired loss terms for inverse steganalysis.

Implements techniques from Fridrich's body of work on making changes
undetectable by steganalysis classifiers, adapted for our renderer
training against the SegNet scorer (EfficientNet-B2 U-Net).

References:
    [1] UNIWARD (Holub & Fridrich 2014) — Universal Wavelet Relative Distortion
    [2] Square Root Law (Ker, Filler & Fridrich 2008-2009)
    [3] HUGO (Filler, Judas & Fridrich 2010) — Highly Undetectable steGO
    [4] Yousfi & Fridrich 2020 — "An Intriguing Struggle of CNNs in JPEG
        Steganalysis and the OneHot Solution"
    [5] Yousfi, Dworetzky & Fridrich 2022 — Detector-Informed Batch Steganography
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_texture_complexity(
    frames: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Compute local texture complexity map (UNIWARD-inspired).

    High values = textured regions (errors are undetectable).
    Low values = smooth regions (errors are instantly caught).

    Uses local variance in a sliding window as the complexity measure.
    This is the spatial-domain analog of UNIWARD's wavelet directional
    filter bank distortion cost.

    Args:
        frames: (B, C, H, W) or (B, H, W, C) float tensor
        kernel_size: sliding window size for local variance

    Returns:
        (B, 1, H, W) texture complexity map, higher = more textured
    """
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # HWC format — convert to CHW
        frames = frames.permute(0, 3, 1, 2)

    # Convert to grayscale for texture analysis
    # BT.601 luma: 0.299R + 0.587G + 0.114B
    gray = (
        0.299 * frames[:, 0:1] +
        0.587 * frames[:, 1:2] +
        0.114 * frames[:, 2:3]
    )  # (B, 1, H, W)

    # Local mean via average pooling
    pad = kernel_size // 2
    mean = F.avg_pool2d(
        F.pad(gray, [pad, pad, pad, pad], mode="reflect"),
        kernel_size, stride=1,
    )

    # Local variance = E[X²] - E[X]²
    sq_mean = F.avg_pool2d(
        F.pad(gray ** 2, [pad, pad, pad, pad], mode="reflect"),
        kernel_size, stride=1,
    )
    variance = (sq_mean - mean ** 2).clamp(min=0)

    return variance


def texture_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    texture_map: torch.Tensor | None = None,
    texture_kernel: int = 5,
    smooth_weight: float = 3.0,
    texture_weight: float = 0.3,
) -> torch.Tensor:
    """UNIWARD-inspired texture-weighted pixel loss.

    Concentrates training signal on SMOOTH regions where the scorer
    is most sensitive, and reduces weight on TEXTURED regions where
    errors are undetectable (Fridrich's UNIWARD principle).

    This is the spatial-domain equivalent of UNIWARD's embedding
    distortion cost function adapted for renderer training.

    Args:
        pred: (B, C, H, W) or (B, H, W, C) predicted frames
        target: same shape as pred, GT frames
        texture_map: precomputed texture complexity, or None (computed here)
        texture_kernel: kernel size for texture computation
        smooth_weight: loss multiplier for smooth regions
        texture_weight: loss multiplier for textured regions

    Returns:
        Scalar weighted loss
    """
    if pred.ndim == 4 and pred.shape[-1] == 3:
        pred_chw = pred.permute(0, 3, 1, 2)
        target_chw = target.permute(0, 3, 1, 2)
    else:
        pred_chw = pred
        target_chw = target

    # Compute or use provided texture map
    if texture_map is None:
        texture_map = compute_texture_complexity(target_chw, texture_kernel)

    # Normalize texture map to [0, 1]
    t_min = texture_map.amin(dim=(2, 3), keepdim=True)
    t_max = texture_map.amax(dim=(2, 3), keepdim=True)
    t_norm = (texture_map - t_min) / (t_max - t_min + 1e-8)

    # Weight map: high weight on smooth (t_norm ≈ 0), low on texture (t_norm ≈ 1)
    # Smooth regions get smooth_weight, textured get texture_weight
    weight_map = smooth_weight * (1 - t_norm) + texture_weight * t_norm

    # Per-pixel L1 error, weighted by texture complexity
    error = (pred_chw - target_chw).abs()  # (B, C, H, W)
    weighted_error = error * weight_map  # broadcast (B, 1, H, W) over C

    return weighted_error.mean()


def linf_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    percentile: float = 0.99,
) -> torch.Tensor:
    """Square root law penalty: penalize concentrated peak errors.

    Fridrich's square root law (2008-2009) shows that spreading many
    tiny errors is fundamentally less detectable than concentrating
    large errors. This loss term penalizes the top-percentile errors
    to encourage the renderer to spread errors evenly.

    Uses soft top-k (percentile) instead of hard max for gradient stability.

    Args:
        pred: predicted frames (any shape)
        target: GT frames (same shape)
        percentile: fraction of pixels to consider as "peak" (default 99th)

    Returns:
        Scalar: mean of top-percentile absolute errors
    """
    error = (pred.float() - target.float()).abs().reshape(-1)
    k = max(1, int(error.shape[0] * (1 - percentile)))
    topk_errors = error.topk(k).values
    return topk_errors.mean()


def boundary_sensitive_hinge(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    margin: float = 0.5,
    boundary_weight: float = 5.0,
    boundary_dilation: int = 3,
) -> torch.Tensor:
    """Hinge loss with extra weight on class boundaries.

    SegNet's argmax disagreement is concentrated at class boundaries.
    This loss applies extra weight to boundary pixels (where classes
    change between adjacent pixels), focusing the renderer's capacity
    where it matters most.

    Combines our proven hinge loss with Fridrich's insight that
    boundary regions are the most sensitive to detection.

    Args:
        logits: (B, C, H, W) SegNet output logits
        gt_masks: (B, H, W) long tensor of GT class indices
        margin: hinge margin (gap between correct and runner-up logit)
        boundary_weight: extra weight for boundary pixels
        boundary_dilation: dilation of boundary mask (pixels)

    Returns:
        Scalar hinge loss with boundary weighting
    """
    B, C, H, W = logits.shape

    # Standard hinge loss per pixel
    correct = logits.gather(1, gt_masks.unsqueeze(1)).squeeze(1)  # (B, H, W)
    mask_inf = torch.zeros_like(logits)
    mask_inf.scatter_(1, gt_masks.unsqueeze(1), float("-inf"))
    runner_up = (logits + mask_inf).max(dim=1).values  # (B, H, W)
    hinge = F.relu(margin - (correct - runner_up))  # (B, H, W)

    # Compute boundary mask: pixels where adjacent pixels have different classes
    masks_float = gt_masks.float().unsqueeze(1)  # (B, 1, H, W)
    # Horizontal and vertical gradients
    dx = (masks_float[:, :, :, 1:] - masks_float[:, :, :, :-1]).abs()
    dy = (masks_float[:, :, 1:, :] - masks_float[:, :, :-1, :]).abs()
    # Pad back to original size
    boundary_x = F.pad(dx, [0, 1, 0, 0])  # right-pad
    boundary_y = F.pad(dy, [0, 0, 0, 1])  # bottom-pad
    boundary = ((boundary_x + boundary_y) > 0).float().squeeze(1)  # (B, H, W)

    # Dilate boundary for safety margin
    if boundary_dilation > 1:
        kernel = torch.ones(1, 1, boundary_dilation, boundary_dilation, device=logits.device)
        boundary = F.conv2d(
            boundary.unsqueeze(1), kernel,
            padding=boundary_dilation // 2,
        ).squeeze(1).clamp(0, 1)

    # Weight: boundary_weight on boundaries, 1.0 elsewhere
    weight = 1.0 + (boundary_weight - 1.0) * boundary

    return (hinge * weight).mean()


def markov_chain_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Penalize deviations in local pixel dependency statistics.

    Fridrich's HUGO (2010) showed that preserving local Markov chain
    statistics makes changes undetectable. This loss encourages the
    renderer to preserve the statistical relationship between adjacent
    pixels — not just the pixel values themselves.

    Computes the difference in horizontal and vertical gradients
    between predicted and target frames, which is a first-order
    approximation of the Markov chain transition probabilities.

    Args:
        pred: (B, C, H, W) predicted frames
        target: (B, C, H, W) target frames

    Returns:
        Scalar: mean absolute difference in local gradients
    """
    if pred.ndim == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    # Horizontal gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Vertical gradients
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    # Match gradient statistics
    grad_loss = (
        (pred_dx - target_dx).abs().mean() +
        (pred_dy - target_dy).abs().mean()
    )

    return grad_loss
