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
    [6] JPEG Standard ITU-T T.81 Annex K, Table K.1 — Luminance quantization table
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── JPEG luminance quantization table (ITU-T T.81 Annex K, Table K.1) ──────
#
# This is the canonical quality=50 luminance Q-table that JPEG encoders use
# to scale DCT coefficients before integer rounding. It encodes the human
# visual system's frequency sensitivity curve: low spatial frequencies (top-
# left of the 8x8 block) have small divisors (16-26) — humans see errors
# there clearly; high frequencies (bottom-right) have large divisors (95-99)
# — humans cannot see errors there.
#
# Fridrich's inverse-steganalysis principle: scorer CNNs (EfficientNet-B2 +
# FastViT-T12) likely inherit a similar low-frequency bias from training on
# natural images. By weighting our DCT-domain residual loss by 1/Q, we
# penalize residual energy in low-freq bins much more than high-freq bins
# (a unit coefficient in the (0,0) bin costs (1/16)² ≈ 38× more than in the
# (7,7) bin), concentrating renderer error into directions the scorers
# cannot detect.
JPEG_LUMA_Q_TABLE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]


def _build_dct8_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build the orthonormal 8x8 DCT-II basis matrix.

    Defined as M[k, n] = α(k) · cos(π·(2n+1)·k / 16) where α(0)=√(1/8)
    and α(k>0)=√(2/8). This is the canonical orthonormal type-II DCT used
    by JPEG (after the 1/√N normalisation factor is folded in).

    A 2D DCT of an 8×8 block X is then:  Y = M @ X @ Mᵀ
    The inverse transform is:           X = Mᵀ @ Y @ M

    We construct it explicitly here (rather than via torch.fft) because the
    backward pass through fft on some accelerators is non-deterministic,
    which violates the project's eval_roundtrip determinism contract. A
    plain matmul is bit-exact and trivially differentiable.
    """
    n_idx = torch.arange(8, dtype=dtype)
    k_idx = torch.arange(8, dtype=dtype).unsqueeze(1)  # (8, 1)
    # cos((2n+1) k π / 16)
    basis = torch.cos(math.pi * (2 * n_idx + 1) * k_idx / 16.0)
    # Orthonormal scaling: row 0 multiplied by √(1/8); rows 1-7 by √(2/8).
    alpha = torch.full((8, 1), math.sqrt(2.0 / 8.0), dtype=dtype)
    alpha[0, 0] = math.sqrt(1.0 / 8.0)
    return alpha * basis  # (8, 8)


# Module-level cached basis (CPU, float32). The loss copies it to the
# correct device/dtype on first call (then cached per-device via _DCT_CACHE).
_DCT8_CPU = _build_dct8_matrix(dtype=torch.float32)
_DCT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_QTABLE_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_dct8(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _DCT_CACHE.get(key)
    if cached is None:
        cached = _DCT8_CPU.to(device=device, dtype=dtype)
        _DCT_CACHE[key] = cached
    return cached


def _get_inv_qtable(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _QTABLE_CACHE.get(key)
    if cached is None:
        q = torch.tensor(JPEG_LUMA_Q_TABLE, dtype=dtype, device=device)
        cached = 1.0 / q
        _QTABLE_CACHE[key] = cached
    return cached


def dct_quant_loss(
    rendered_pair: torch.Tensor,
    gt_pair: torch.Tensor,
    weight: float = 1.0,
    channel_mode: str = "luma",
) -> torch.Tensor:
    """JPEG-Q-table-weighted DCT-domain residual loss (Fridrich council #1).

    For each 8×8 block of the residual `rendered - gt` (in luma or per-RGB),
    take the orthonormal 2D DCT-II and weight each frequency coefficient by
    1/Q_jpeg[i,j] where Q_jpeg is the standard ITU-T T.81 luminance table.
    Sum the squared weighted coefficients as the loss.

    The mechanism (see module docstring on JPEG_LUMA_Q_TABLE): scorer CNNs
    inherit human-visual-system-like frequency sensitivity, which JPEG's
    Q-table already encodes. The loss contribution of one unit of DCT
    coefficient energy is (coeff / Q[i,j])², so a unit residual in the
    (0,0) DC bin costs (1/16)² ≈ 3.9e-3 while the same in the (7,7)
    HF bin costs (1/99)² ≈ 1.0e-4 — a ~38× cheaper hiding place. The
    renderer responds by concentrating its inevitable approximation error
    in the DCT directions the scorer cannot see — a direct UNIWARD analog
    in the DCT domain.

    Caveat (Yousfi council audit): the JPEG luminance Q-table encodes the
    HVS frequency-sensitivity curve. Our scorers (EfficientNet-B2 +
    FastViT-T12) likely inherit a similar low-freq bias from training on
    natural images, but that's a hypothesis. Empirical validation of the
    proxy-auth gap on a renderer trained with this loss is the
    next-required experiment.

    Args:
        rendered_pair: (B, 2, H, W, 3) HWC RGB tensor at scorer resolution
            (typically (1, 2, 384, 512, 3) post eval_roundtrip).
        gt_pair: (B, 2, H, W, 3) GT, same shape and device as rendered_pair.
        weight: scalar multiplier applied to the returned loss. Caller may
            also externally scale; included here so the loss can be a self-
            contained term.
        channel_mode: "luma" (BT.601 Y from RGB) or "rgb" (sum over channels).

    Returns:
        Scalar tensor on the input's device/dtype.

    Raises:
        ValueError: on shape, dtype, or channel_mode contract violations.
    """
    if rendered_pair.shape != gt_pair.shape:
        raise ValueError(
            f"dct_quant_loss: rendered_pair shape {tuple(rendered_pair.shape)} "
            f"!= gt_pair shape {tuple(gt_pair.shape)}"
        )
    if rendered_pair.ndim != 5 or rendered_pair.shape[-1] != 3:
        raise ValueError(
            f"dct_quant_loss: expected (B, T, H, W, 3) HWC RGB input, got "
            f"shape {tuple(rendered_pair.shape)}"
        )
    if channel_mode not in ("luma", "rgb"):
        raise ValueError(
            f"dct_quant_loss: channel_mode must be 'luma' or 'rgb', got "
            f"{channel_mode!r}"
        )

    # Residual in HWC RGB. Cast to the renderer's working dtype (typically
    # float32 — but support float64 for tests that probe orthonormality).
    residual = rendered_pair - gt_pair  # (B, T, H, W, 3)
    B, T, H, W, _ = residual.shape

    if channel_mode == "luma":
        # BT.601 luma — matches the convention already used in
        # compute_texture_complexity above and elsewhere in the project.
        r = residual[..., 0]
        g = residual[..., 1]
        b = residual[..., 2]
        chan = (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(-1)  # (B,T,H,W,1)
        n_channels = 1
    else:
        chan = residual  # (B, T, H, W, 3)
        n_channels = 3

    # Reshape to (B*T*C, 1, H, W) for F.pad / blocking. The leading dim
    # collapses pair, batch, and (optionally) channel — every 2D slice gets
    # independent 8x8 block DCT.
    # chan: (B, T, H, W, C) -> (B, T, C, H, W) -> (B*T*C, 1, H, W)
    flat = chan.permute(0, 1, 4, 2, 3).reshape(B * T * n_channels, 1, H, W)

    # Pad H and W to multiples of 8 using replicate (matches the renderer's
    # padding_mode='replicate' convention — a constant or zero pad would
    # introduce a high-freq artifact at the boundary that the loss would
    # then "punish" the renderer for, even though the renderer never
    # produced it).
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        # F.pad signature for 4D: (left, right, top, bottom)
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    H_p = H + pad_h
    W_p = W + pad_w
    h_blocks = H_p // 8
    w_blocks = W_p // 8

    # Block reshape: (N, 1, H_p, W_p) -> (N, h_blocks, 8, w_blocks, 8) ->
    # (N, h_blocks, w_blocks, 8, 8) -> (N * h_blocks * w_blocks, 8, 8)
    N = flat.shape[0]
    blocks = flat.view(N, 1, h_blocks, 8, w_blocks, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()  # (N,1,hb,wb,8,8)
    blocks = blocks.view(-1, 8, 8)  # (N * hb * wb, 8, 8)

    # 2D orthonormal DCT-II per 8x8 block: Y = M @ X @ Mᵀ
    M = _get_dct8(blocks.device, blocks.dtype)  # (8, 8)
    # X @ Mᵀ first → (B*, 8, 8); then M @ (X @ Mᵀ) → (B*, 8, 8)
    coeffs = torch.matmul(M, torch.matmul(blocks, M.t()))  # (B*, 8, 8)

    # Multiply each (8,8) block by the inverse Q table elementwise. Squared
    # weighted coefficients give us per-block per-bin energy in the
    # perceptually weighted basis.
    inv_q = _get_inv_qtable(blocks.device, blocks.dtype)  # (8, 8)
    weighted = coeffs * inv_q  # broadcast over leading dim
    loss = (weighted * weighted).mean()

    if weight != 1.0:
        loss = loss * weight
    return loss


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
    # [derived: Fridrich-style hinge-margin convention (0.5/1.0/2.0 standard);
    # 0.5 reflects ~half-class-logit-spread; sister rationale in fridrich.py
    # docstring; arbitrariness_audit_20260512.md A-4]
    margin: float = 0.5,
    # [derived: 5× boundary-pixel emphasis is half of the kl_distill_scorer_loss
    # boundary_weight=10.0 because hinge here is a per-pixel SECONDARY signal
    # not the primary KL distillation]
    boundary_weight: float = 5.0,
    # [derived: 3-pixel dilation matches contest scorer's class-boundary
    # resolution at 384x512]
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
