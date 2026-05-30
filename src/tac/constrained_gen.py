# SPDX-License-Identifier: MIT
"""Constrained optimization frame generator -- Yousfi GPU breakthrough.

Generate scorer-optimal frames via constrained gradient descent from noise.
No neural renderer weights needed. The archive contains only:
  - masks (entropy-coded, ~239 bytes)
  - expected PoseNet targets (~7KB)
  - noise seed (64 bytes)
Total: ~8KB archive.

At inflate time: run gradient descent from seeded noise to satisfy two
hard constraints:
  1. SegNet(frame) argmax == mask  (semantic preservation)
  2. PoseNet(frame_t, frame_t+1) ~ expected_pose  (temporal consistency)

While minimizing total variation (compressibility).

~500 steps with early stopping (~150 patience), ~100ms/step on T4.
Typical: 200-500 steps per batch, 60 batches for 1200 frames = ~60-100 min.

Example::

    from tac.constrained_gen import ConstrainedFrameGenerator
    gen = ConstrainedFrameGenerator(posenet, segnet, device="cuda")
    frames = gen.constrained_generate(
        masks, expected_pose, noise_seed=42, num_steps=1000,
    )
"""

from __future__ import annotations

import json
import logging
import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tac.camera import (
    CAMERA_H,
    CAMERA_W,
    CLASS_MEAN_COLORS,
    NUM_CLASSES,
    SEGNET_INPUT_H,
    SEGNET_INPUT_W,
)
from tac.mask_codec import SEGNET_H, SEGNET_W  # noqa: F401 — used by downstream

logger = logging.getLogger(__name__)

__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CLASS_MEAN_COLORS",
    "SEGNET_INPUT_H",
    "SEGNET_INPUT_W",
    "ConstrainedFrameGenerator",
    "alternating_projections_optimize",
    "build_constrained_archive",
    "compute_compressibility_loss",
    "compute_posenet_constraint_loss",
    "compute_posenet_embedding_constraint_loss",
    "compute_segnet_class_weights",
    "compute_segnet_constraint_loss",
    "constrained_generate",
    "coupled_trajectory_optimize",
    "estimate_expected_pose",
    "generate_in_scorer_space",
    "generate_initial_frames",
    "gpu_lane_full_pipeline",
    "inflate_constrained",
    "inverse_preprocess_input",
    "load_constrained_archive",
    "newton_step_optimize",
    "rgb_to_yuv6",
    "scorer_as_compressor",
    "validate_yuv_parity",
    "yuv6_to_rgb",
    "yuv_null_space_projection",
]

# Constants are now imported from tac.camera (canonical source of truth).
# The module-level names CAMERA_W, CAMERA_H, SEGNET_INPUT_W, SEGNET_INPUT_H,
# CLASS_MEAN_COLORS, NUM_CLASSES remain available for backward compatibility.

# Temporal smoothness weight for compressibility loss (L1 between consecutive frames)
TEMPORAL_SMOOTHNESS_WEIGHT = 0.5


# ---- YUV420 conversion utilities ----


def rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    """NCHW PyTorch training-time wrapper around the canonical rgb_to_yuv6.

    Canonical math lives in
    :func:`tac.framework_agnostic.canonical_kernels.rgb_to_yuv6` (the
    extracted single source of truth per MLX canonicalization audit
    inventory A.2.6 and canonical equation
    ``mlx_primitive_canonicalization_compounding_savings_v1``).

    Must match upstream ``frame_utils.rgb_to_yuv6`` exactly. Any divergence
    will cause scorer-space optimization to silently produce wrong gradients.
    Use :func:`validate_yuv_parity` to cross-validate against the upstream
    implementation after any modification.

    PRINCIPLED FORK per Catalog #290 falling-rule list: the canonical
    helper hard-requires 4D NCHW ``(N, 3, H, W)``; the training-time
    constrained-generation stack feeds arbitrary leading dims such as
    ``(B, S, 3, H, W)``. This wrapper flattens leading dims, dispatches
    canonical math, and restores leading dims while preserving exact
    byte-stable contest parity per the
    ``tac.differentiable_eval_roundtrip`` non-negotiable.

    Args:
        rgb_chw: (..., 3, H, W) float tensor in [0, 255].

    Returns:
        (..., 6, H//2, W//2) tensor: [y00, y10, y01, y11, U_sub, V_sub].
    """
    from tac.framework_agnostic.canonical_kernels import (
        Backend,
    )
    from tac.framework_agnostic.canonical_kernels import (
        rgb_to_yuv6 as _canonical_rgb_to_yuv6,
    )

    if rgb_chw.shape[-3] != 3:
        raise ValueError(
            f"rgb_to_yuv6 expects NCHW with 3 channels at dim -3; "
            f"got shape {tuple(rgb_chw.shape)}."
        )
    leading_shape = rgb_chw.shape[:-3]
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    flat_nchw = rgb_chw.reshape(-1, 3, H, W)  # (B', 3, H, W)
    yuv6_nchw = _canonical_rgb_to_yuv6(
        flat_nchw, backend=Backend.PYTORCH, value_range=255.0
    )  # (B', 6, H2, W2)
    H2, W2 = yuv6_nchw.shape[-2], yuv6_nchw.shape[-1]
    return yuv6_nchw.reshape(*leading_shape, 6, H2, W2)


def yuv6_to_rgb(yuv6: torch.Tensor) -> torch.Tensor:
    """Invert YUV420 6-channel back to approximate RGB CHW.

    This is the pseudo-inverse of ``rgb_to_yuv6``. The chroma subsampling
    loses spatial resolution, so the reconstruction is approximate.

    Must stay consistent with :func:`rgb_to_yuv6` and the upstream
    ``frame_utils.yuv6_to_rgb``. Use :func:`validate_yuv_parity` to verify.

    Args:
        yuv6: (..., 6, H2, W2) tensor from rgb_to_yuv6.

    Returns:
        (..., 3, H, W) float tensor in [0, 255].
    """
    y00 = yuv6[..., 0, :, :]
    y10 = yuv6[..., 1, :, :]
    y01 = yuv6[..., 2, :, :]
    y11 = yuv6[..., 3, :, :]
    U_sub = yuv6[..., 4, :, :]
    V_sub = yuv6[..., 5, :, :]

    H2, W2 = y00.shape[-2], y00.shape[-1]
    H, W = H2 * 2, W2 * 2

    # Reconstruct full-res Y by placing sub-pixels back
    Y = torch.zeros(*yuv6.shape[:-3], H, W, device=yuv6.device, dtype=yuv6.dtype)
    Y[..., 0::2, 0::2] = y00
    Y[..., 1::2, 0::2] = y10
    Y[..., 0::2, 1::2] = y01
    Y[..., 1::2, 1::2] = y11

    # Upsample chroma via nearest-neighbor (matches 4:2:0 upsampling)
    U_full = U_sub.unsqueeze(-3)  # add channel dim for interpolate
    V_full = V_sub.unsqueeze(-3)
    U_full = F.interpolate(
        U_full.reshape(-1, 1, H2, W2), size=(H, W), mode="nearest"
    ).reshape(*yuv6.shape[:-3], H, W)
    V_full = F.interpolate(
        V_full.reshape(-1, 1, H2, W2), size=(H, W), mode="nearest"
    ).reshape(*yuv6.shape[:-3], H, W)

    # Inverse YUV -> RGB
    # Y = R*0.299 + G*0.587 + B*0.114
    # U = (B - Y)/1.772 + 128  =>  B = (U - 128)*1.772 + Y
    # V = (R - Y)/1.402 + 128  =>  R = (V - 128)*1.402 + Y
    B = (U_full - 128.0) * 1.772 + Y
    R = (V_full - 128.0) * 1.402 + Y
    G = (Y - R * 0.299 - B * 0.114) / 0.587

    R = R.clamp(0.0, 255.0)
    G = G.clamp(0.0, 255.0)
    B = B.clamp(0.0, 255.0)

    return torch.stack([R, G, B], dim=-3)


def validate_yuv_parity(
    upstream_rgb_to_yuv6: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 5,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Cross-validate local rgb_to_yuv6 against the upstream implementation.

    Generates random RGB tensors and compares local vs. upstream YUV6 outputs.
    Returns a summary dict with pass/fail status and max observed error.

    Args:
        upstream_rgb_to_yuv6: the upstream ``frame_utils.rgb_to_yuv6`` function.
        num_samples: number of random test inputs to compare.
        atol: absolute tolerance for floating-point comparison.

    Returns:
        Dict with keys ``passed`` (bool), ``max_abs_error`` (float),
        ``num_samples`` (int), and ``details`` (list of per-sample errors).
    """
    details: list[float] = []
    max_err = 0.0
    passed = True

    for _ in range(num_samples):
        test_rgb = torch.rand(1, 3, 128, 128) * 255.0
        local_out = rgb_to_yuv6(test_rgb)
        upstream_out = upstream_rgb_to_yuv6(test_rgb)

        if local_out.shape != upstream_out.shape:
            passed = False
            details.append(float("inf"))
            continue

        err = (local_out - upstream_out).abs().max().item()
        details.append(err)
        max_err = max(max_err, err)
        if err > atol:
            passed = False

    return {
        "passed": passed,
        "max_abs_error": max_err,
        "num_samples": num_samples,
        "details": details,
    }


# ---- Core functions ----


def generate_initial_frames(
    masks: torch.Tensor,
    noise_seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate deterministic initial frames from masks + seed.

    Uses class-mean colors as the starting point with small additive noise
    from a seeded generator. This gives the optimizer a head start: each
    pixel already has roughly the right color for its semantic class.

    Args:
        masks: (N, H, W) long tensor with class indices in [0, NUM_CLASSES).
        noise_seed: integer seed for deterministic noise generation.
        device: target device for the output tensor.

    Returns:
        (N, H, W, 3) float tensor in [0, 255] on device.
    """
    device = torch.device(device)
    N, H, W = masks.shape
    masks_dev = masks.to(device)

    # Look up class-mean color per pixel: (N, H, W) -> (N, H, W, 3)
    colors = CLASS_MEAN_COLORS.to(device)  # (NUM_CLASSES, 3)
    frames = colors[masks_dev]  # fancy index: (N, H, W, 3)

    # Add small deterministic noise for symmetry breaking.
    # For GPU devices, generate noise directly on device to avoid a CPU->GPU
    # transfer. A CPU generator is used for reproducibility: seeding is
    # deterministic regardless of CUDA non-determinism.
    if device.type == "cuda":
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(noise_seed)
        # Generate on CPU with the seeded generator, then create on device
        # using the same values for deterministic reproducibility.
        cpu_noise = torch.randn(N, H, W, 3, generator=cpu_gen)
        noise = torch.empty(N, H, W, 3, device=device)
        noise.copy_(cpu_noise)
        noise.mul_(5.0)
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(noise_seed)
        noise = torch.randn(N, H, W, 3, generator=gen) * 5.0
        noise = noise.to(device)
    frames = (frames + noise).clamp(0.0, 255.0)

    return frames


def compute_segnet_class_weights(
    masks: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from mask statistics.

    Uses median-frequency balancing: weight[c] = median_freq / freq[c].
    This normalizes weights so the majority class gets ~1.0 and rare classes
    get proportionally higher weights.

    Args:
        masks: (N, H, W) long tensor with class indices.
        num_classes: number of classes (default 5 for comma).

    Returns:
        (num_classes,) float tensor of per-class weights.
    """
    total_pixels = masks.numel()
    weights = torch.ones(num_classes, dtype=torch.float32, device=masks.device)
    freqs = []
    for c in range(num_classes):
        count = (masks == c).sum().item()
        freq = count / max(total_pixels, 1)
        freqs.append(freq)

    # Median-frequency balancing (avoids extreme weights from very rare classes)
    freqs_tensor = torch.tensor(freqs, dtype=torch.float32, device=masks.device)
    # Filter out zero-frequency classes for median computation
    nonzero_freqs = freqs_tensor[freqs_tensor > 0]
    if len(nonzero_freqs) > 0:
        median_freq = nonzero_freqs.median()
        for c in range(num_classes):
            if freqs[c] > 0:
                weights[c] = (median_freq / freqs_tensor[c]).clamp(max=50.0)
            else:
                weights[c] = 1.0  # no pixels of this class, neutral weight

    return weights


def compute_segnet_constraint_loss(
    frames: torch.Tensor,
    masks: torch.Tensor,
    segnet: torch.nn.Module,
    batch_size: int = 32,
    seg_odd_only: bool = False,
    loss_mode: str = "xent",
    hinge_margin: float = 0.5,
    per_class_weights: torch.Tensor | None = None,
    error_boost: float = 1.0,
) -> torch.Tensor:
    """Compute SegNet constraint loss forcing argmax to match target masks.

    Supports two loss modes:

    - ``"xent"`` (default): standard cross-entropy. Differentiable through
      logits via the softmax gradient. Wastes gradient signal on pixels
      that are already correctly classified with high confidence.
    - ``"hinge"``: logit-margin hinge loss. Computes
      ``relu(margin - (target_logit - max_wrong_logit))`` per pixel,
      focusing gradient **only** on boundary pixels at risk of argmax flip.
      2--5x more gradient-efficient for TTO because 95%+ of pixels are
      already correct after renderer warm-start.

    The SegNet preprocessor expects (B, T, C, H, W) input. We feed single
    frames as T=1 sequences using the last-frame-only SegNet interface.

    Mini-batches are used to avoid OOM on T4 with 1200 frames -- gradients
    accumulate across batches (caller must NOT zero_grad between calls).

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], requires_grad.
        masks: (N, H, W) long tensor with target class indices.
        segnet: frozen SegNet model.
        batch_size: frames per mini-batch (reduce if OOM, default 32).
        seg_odd_only: if True, only compute SegNet loss on odd-indexed frames
            (frame 2k+1). Matches the official scorer's ``x[:, -1, ...]``
            behavior: SegNet evaluates only the second frame of each pair.
            Even frames get zero SegNet gradient, freeing them for pure
            PoseNet optimization.
        loss_mode: ``"xent"`` for cross-entropy (default, backward compatible),
            ``"hinge"`` for logit-margin hinge loss.
        hinge_margin: margin for hinge loss. Larger values push the correct
            class logit further above the best wrong class. Default 0.5.
        per_class_weights: (NUM_CLASSES,) optional float tensor of per-class
            weights. When provided, the per-pixel hinge loss is multiplied by
            the weight for that pixel's target class, focusing gradient on rare
            but important classes (e.g., lane markings, vehicles). Compute via
            :func:`compute_segnet_class_weights` or pass manually (e.g.,
            ``[1.0, 15.0, 1.5, 3.0, 7.0]``). Only affects ``"hinge"`` mode.
            Default None (uniform weighting, backward compatible).
        error_boost: per-pixel error magnification factor (Quantizr technique).
            Pixels with higher error get quadratically more gradient weight.
            1.0 = no boost (default), 9.0 = Quantizr anchor, 49.0 = extreme.
            Applied to both hinge and xent modes.

    Returns:
        Scalar loss (mean over all frames and pixels).

    Raises:
        ValueError: if *loss_mode* is not ``"xent"`` or ``"hinge"``.
    """
    if loss_mode not in ("xent", "hinge"):
        raise ValueError(
            f"Unknown segnet loss_mode={loss_mode!r}. Use 'xent' or 'hinge'."
        )

    N, H, W, C = frames.shape
    device = frames.device

    # When seg_odd_only, select only odd-indexed frames and their masks.
    # Official scorer: SegNet sees x[:, -1, ...] = frame_2k+1 of each pair.
    if seg_odd_only:
        odd_indices = torch.arange(1, N, 2, device=device)
        sel_frames = frames[odd_indices]    # (N//2, H, W, 3)
        sel_masks = masks[odd_indices]      # (N//2, H, W)
    else:
        sel_frames = frames
        sel_masks = masks

    N_sel = sel_frames.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    n_batches = 0

    for start in range(0, N_sel, batch_size):
        end = min(start + batch_size, N_sel)
        batch_frames = sel_frames[start:end]  # (B, H, W, 3)
        batch_masks = sel_masks[start:end]    # (B, H, W)

        # SegNet.preprocess_input expects (B, T, C, H, W), T=1 (last frame only).
        frames_btchw = batch_frames.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
        seg_input = segnet.preprocess_input(frames_btchw)
        logits = segnet(seg_input)  # (B, NUM_CLASSES, H_out, W_out)

        # Resize masks to match logit spatial dims
        H_out, W_out = logits.shape[2], logits.shape[3]
        masks_resized = (
            F.interpolate(
                batch_masks.float().unsqueeze(1),
                size=(H_out, W_out),
                mode="nearest",
            )
            .squeeze(1)
            .long()
            .to(device)
        )  # (B, H_out, W_out)

        if loss_mode == "hinge":
            # Hinge loss: maximize margin between correct class and best wrong class.
            # Focuses gradient ONLY on boundary pixels at risk of argmax flip.
            # target_logits: logit for the correct class at each pixel.
            target = masks_resized  # (B, H_out, W_out)
            target_logits = logits.gather(
                1, target.unsqueeze(1),
            )  # (B, 1, H_out, W_out)
            # Mask out the correct class with -inf, then take max over classes
            # to get the highest wrong-class logit at each pixel.
            mask_fill = logits.scatter(
                1, target.unsqueeze(1), float("-inf"),
            )  # (B, C, H_out, W_out)
            max_wrong = mask_fill.max(dim=1, keepdim=True).values  # (B, 1, H_out, W_out)
            # Loss = relu(margin - (correct - best_wrong)). Zero when correct
            # class leads by >= margin, positive when it doesn't.
            hinge_per_pixel = F.relu(
                hinge_margin - (target_logits - max_wrong),
            ).squeeze(1)  # (B, H_out, W_out)

            # error_boost: per-pixel weight proportional to error magnitude.
            # Quantizr uses 9x in anchor, 49x in anchor_boost. Focuses gradient
            # on the hardest pixels, not just boundary pixels.
            # boost_weight = 1 + (error_boost - 1) * normalized_error²
            if error_boost > 1.0:
                with torch.no_grad():
                    error_magnitude = hinge_per_pixel / (hinge_per_pixel.mean() + 1e-8)
                    boost_weights = 1.0 + (error_boost - 1.0) * error_magnitude.pow(2)
                    # Clamp to prevent single-pixel gradient domination
                    # (Yousfi audit: outlier at 10x mean → 4801x weight without clamp)
                    boost_weights = boost_weights.clamp(max=error_boost * 10)
                hinge_per_pixel = hinge_per_pixel * boost_weights

            if per_class_weights is not None:
                per_class_weights = per_class_weights.to(device)
                pixel_weights = per_class_weights[target]  # (B, H_out, W_out)
                batch_loss = (hinge_per_pixel * pixel_weights).mean()
            else:
                batch_loss = hinge_per_pixel.mean()
        else:
            # xent mode: error_boost via focal-style weighting
            if error_boost > 1.0:
                log_probs = F.log_softmax(logits, dim=1)
                nll = F.nll_loss(log_probs, masks_resized, reduction='none')
                with torch.no_grad():
                    error_magnitude = nll / (nll.mean() + 1e-8)
                    boost_weights = 1.0 + (error_boost - 1.0) * error_magnitude.pow(2)
                    boost_weights = boost_weights.clamp(max=error_boost * 10)
                batch_loss = (nll * boost_weights).mean()
            else:
                batch_loss = F.cross_entropy(logits, masks_resized)

        total_loss = total_loss + batch_loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


def compute_posenet_constraint_loss(
    frames: torch.Tensor,
    expected_pose: torch.Tensor,
    posenet: torch.nn.Module,
    batch_size: int = 32,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """L2 loss between PoseNet output and expected ego-motion targets.

    Uses NON-OVERLAPPING pairs (f_{2k}, f_{2k+1}) to match the upstream scorer
    which processes video via DaliVideoDataset with seq_len=2. For N frames,
    there are N//2 non-overlapping pairs.

    PoseNet.preprocess_input expects (B, 2, C, H, W) input.

    Mini-batches are used to avoid OOM on T4 — gradients accumulate across
    batches (caller must NOT zero_grad between calls).

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], requires_grad.
        expected_pose: (P, 6) float tensor of expected pose outputs,
            where P = N//2 (one pose per non-overlapping pair, seq_len=2).
        posenet: frozen PoseNet model.
        batch_size: pairs per mini-batch (reduce if OOM, default 32).
        pair_weights: (P,) optional float tensor of per-pair importance weights.
            Scales per-pair MSE before averaging. Use to focus gradient on hard
            pairs. Default None (uniform weighting, backward compatible).

    Returns:
        Scalar L2 loss (weighted average over all pairs and dimensions).
    """
    N = frames.shape[0]
    num_pairs = expected_pose.shape[0]
    assert num_pairs == N // 2, (
        f"Expected {N // 2} non-overlapping pose targets for {N} frames (seq_len=2), "
        f"got {num_pairs}. Use extract_masks_and_targets() to obtain scorer-matched targets."
    )
    device = frames.device

    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)

        # Non-overlapping pair k = (frames[2k], frames[2k+1])
        # Stride-2 slice selects every other frame from 2*start to 2*end.
        f1 = frames[2 * start:2 * end:2]           # (B, H, W, 3) — even-indexed frames
        f2 = frames[2 * start + 1:2 * end + 1:2]  # (B, H, W, 3) — odd-indexed frames
        pairs_hwc = torch.stack([f1, f2], dim=1)            # (B, 2, H, W, 3)
        pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, C, H, W)

        posenet_in = posenet.preprocess_input(pairs_chw)
        posenet_out = posenet(posenet_in)

        pred_pose = posenet_out["pose"][..., :6]  # (B, 6) — first 6 of 12 outputs
        target = expected_pose[start:end].to(device)  # (B, 6)

        # Per-pair MSE: (B, 6) -> mean over dims -> (B,)
        per_pair_mse = (pred_pose - target).pow(2).mean(dim=1)  # (B,)

        if pair_weights is not None:
            w = pair_weights[start:end].to(device)  # (B,)
            total_loss = total_loss + (per_pair_mse * w).sum()
            total_weight = total_weight + w.sum()
        else:
            total_loss = total_loss + per_pair_mse.sum()
            total_weight = total_weight + per_pair_mse.shape[0]

    return total_loss / total_weight.clamp(min=1e-8)


def compute_posenet_embedding_constraint_loss(
    frames: torch.Tensor,
    expected_pose_embeddings: torch.Tensor,
    posenet: torch.nn.Module,
    batch_size: int = 32,
    layer: str = "summary",
) -> torch.Tensor:
    """Embedding MSE loss between PoseNet intermediate features and GT embeddings.

    Instead of comparing the 6-dim pose output (rank-6 gradient, only 6 pixel-space
    directions), this compares the ~512-dim summarizer embedding (rank ~256, 42x more
    gradient directions). This gives the optimizer a dramatically richer gradient
    landscape for frame refinement.

    Uses the same hook-based feature extraction as ``posenet_embedding_loss`` in
    ``tac.losses``, but adapted for the constrained generation pipeline:
    - Takes (N, H, W, 3) frames and forms non-overlapping pairs
    - Compares against precomputed GT embeddings (not live GT frames)
    - Mini-batched to avoid OOM

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], requires_grad.
        expected_pose_embeddings: (P, D) float tensor of GT embeddings,
            where P = N//2 (one embedding per non-overlapping pair) and
            D is the embedding dimension (512 for summarizer).
        posenet: frozen PoseNet model.
        batch_size: pairs per mini-batch (reduce if OOM, default 32).
        layer: which PoseNet layer to hook -- 'summary' for summarizer (512-d).

    Returns:
        Scalar normalized L2 embedding loss averaged over all pairs.
    """
    N = frames.shape[0]
    num_pairs = expected_pose_embeddings.shape[0]
    assert num_pairs == N // 2, (
        f"Expected {N // 2} embedding targets for {N} frames, got {num_pairs}."
    )
    device = frames.device

    # Resolve target module for hook
    if layer == "summary":
        target_module = posenet.summarizer
    else:
        target_module = posenet.vision
        for part in layer.split("."):
            target_module = target_module[int(part)] if part.isdigit() else getattr(target_module, part)

    total_loss = torch.tensor(0.0, device=device)
    n_batches = 0

    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)

        # Non-overlapping pairs: (frames[2k], frames[2k+1])
        f1 = frames[2 * start:2 * end:2]           # (B, H, W, 3)
        f2 = frames[2 * start + 1:2 * end + 1:2]  # (B, H, W, 3)
        pairs_hwc = torch.stack([f1, f2], dim=1)    # (B, 2, H, W, 3)
        pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, C, H, W)

        # Hook to capture intermediate features
        features = []

        def _hook_fn(module, input, output, *, features=features):
            features.append(output)

        handle = target_module.register_forward_hook(_hook_fn)
        posenet_in = posenet.preprocess_input(pairs_chw)
        posenet(posenet_in)
        handle.remove()

        pred_emb = features[0]  # (B, D) embedding from the hooked layer
        target_emb = expected_pose_embeddings[start:end].to(device)  # (B, D)

        # Normalized L2 (same formula as posenet_embedding_loss in tac.losses)
        emb_diff = pred_emb - target_emb
        emb_norm = target_emb.detach().pow(2).mean().sqrt() + 1e-6
        total_loss = total_loss + emb_diff.pow(2).mean() / emb_norm
        n_batches += 1

    return total_loss / max(n_batches, 1)


def compute_compressibility_loss(
    frames: torch.Tensor,
    antialias_weight: float = 0.0,
) -> torch.Tensor:
    """Total variation + temporal smoothness + anti-aliasing for compressibility.

    Smooth frames compress better under any codec. This loss encourages:
    1. Spatial smoothness: small pixel differences between neighbors.
    2. Temporal smoothness: small differences between consecutive frames.
    3. Anti-aliasing (optional): penalizes sub-2x2 pixel noise that is
       invisible to PoseNet (which operates on 2x-downsampled YUV via
       rgb_to_yuv6) but harms codec compressibility and SegNet.

    The anti-aliasing term computes the variance within each 2x2 block.
    If all pixels in a 2x2 block are identical, PoseNet sees the same
    value regardless — so any within-block variation is wasted bits.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255].
        antialias_weight: weight for the 2x2 block variance penalty.
            0.0 disables it (backward compatible). Recommended: 0.1-0.5.

    Returns:
        Scalar compressibility loss (lower = more compressible).
    """
    # Spatial total variation: sum of absolute horizontal + vertical gradients
    # Normalized by pixel count and channel count for scale invariance.
    tv_h = (frames[:, 1:, :, :] - frames[:, :-1, :, :]).abs().mean()
    tv_w = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs().mean()
    spatial_tv = tv_h + tv_w

    # Temporal smoothness: L1 between consecutive frames
    if frames.shape[0] > 1:
        temporal = (frames[1:] - frames[:-1]).abs().mean()
    else:
        temporal = torch.tensor(0.0, device=frames.device, dtype=frames.dtype)

    loss = spatial_tv + TEMPORAL_SMOOTHNESS_WEIGHT * temporal

    # Anti-aliasing: penalize within-2x2-block variance.
    # PoseNet's rgb_to_yuv6 averages 2x2 blocks, so sub-block detail is
    # invisible to PoseNet but costs bits under any codec.
    if antialias_weight > 0.0:
        N, H, W, C = frames.shape
        # Reshape to (N, H//2, 2, W//2, 2, C) to access 2x2 blocks
        H2, W2 = H // 2 * 2, W // 2 * 2  # ensure even dims
        cropped = frames[:, :H2, :W2, :]
        blocks = cropped.reshape(N, H2 // 2, 2, W2 // 2, 2, C)
        block_mean = blocks.mean(dim=(2, 4), keepdim=True)  # (N, H/2, 1, W/2, 1, C)
        block_var = (blocks - block_mean).pow(2).mean()
        loss = loss + antialias_weight * block_var

    return loss


def estimate_expected_pose(
    masks: torch.Tensor,
    device: torch.device | str = "cpu",
    pose_heuristic_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Estimate expected ego-motion from mask sequence.

    Uses class centroid displacement and vanishing point shift as features
    to estimate the 6-DOF pose output that PoseNet would produce for
    well-formed driving frames matching these masks.

    The model is simple linear: for driving scenes the ego-motion is mostly
    forward translation with small lateral/rotational corrections. The mask
    geometry (where the road is, how it shifts) encodes this implicitly.

    The 6 PoseNet outputs are (tx, ty, tz, rx, ry, rz) in some internal
    representation. For typical comma driving data:
      - tx ~ small lateral (near 0)
      - ty ~ small vertical (near 0)
      - tz ~ forward motion (dominant, ~0.1-1.0)
      - rx, ry, rz ~ small rotations (near 0)

    Args:
        masks: (N, H, W) long tensor with class indices in [0, NUM_CLASSES).
        device: computation device.
        pose_heuristic_weights: optional dict overriding the heuristic
            coefficients used to map centroid displacements to pose
            estimates. These are rough approximations and should be
            overridden by actual PoseNet targets computed from ground-truth
            frames when available. Supported keys and their defaults:

            - ``tx_road_dx`` (0.5): lateral translation from road horizontal shift.
            - ``ty_road_dy`` (0.3): vertical translation from road vertical shift.
            - ``tz_baseline`` (0.1): constant forward motion baseline.
            - ``tz_road_dy`` (0.2): forward motion from road vertical growth.
            - ``rx_my_car_dy`` (0.2): pitch proxy from ego-car-mask vertical shift.
            - ``ry_my_car_dx`` (0.3): yaw proxy from ego-car-mask horizontal shift.
            - ``rz_road_cx`` (0.1): roll from road lateral asymmetry.

            Legacy keys ``rx_sky_dy`` and ``ry_sky_dx`` are accepted as aliases
            for old manifests. Class 4 is the contest ``my_car`` class, not sky.

    Returns:
        (P, 6) float tensor of estimated pose targets, P = N//2.
        Uses non-overlapping pairs (f_{2k}, f_{2k+1}) to match the upstream scorer.
    """
    device = torch.device(device)
    N, H, W = masks.shape
    P = N // 2

    if P == 0:
        return torch.zeros(0, 6, device=device)  # OFF_MANIFOLD_OK: empty-tensor sentinel (P=0); no rows so no off-manifold conditioning possible.

    # Compute class centroids per frame
    # centroid_y, centroid_x for each class in each frame
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    masks_dev = masks.to(device)

    centroids = torch.zeros(N, NUM_CLASSES, 2, device=device)  # (N, C, 2) = (y, x)
    for c in range(NUM_CLASSES):
        class_mask = (masks_dev == c).float()  # (N, H, W)
        area = class_mask.sum(dim=(1, 2)).clamp(min=1.0)  # (N,)
        cy = (class_mask * y_coords).sum(dim=(1, 2)) / area  # (N,)
        cx = (class_mask * x_coords).sum(dim=(1, 2)) / area  # (N,)
        centroids[:, c, 0] = cy / H  # normalize to [0, 1]
        centroids[:, c, 1] = cx / W

    # Non-overlapping centroid deltas: pair k = (frames[2k], frames[2k+1])
    # delta = centroid[odd frames] - centroid[even frames]
    centroid_deltas = centroids[1::2] - centroids[0::2]  # (P=N//2, C, 2)

    # Road class (0) centroid displacement is the strongest ego-motion signal
    road_dy = centroid_deltas[:, 0, 0]  # (P,) vertical shift of road
    road_dx = centroid_deltas[:, 0, 1]  # (P,) horizontal shift of road

    # Class 4 is the contest my_car/ego-car mask, not a sky class.
    my_car_dy = centroid_deltas[:, 4, 0]  # (P,)
    my_car_dx = centroid_deltas[:, 4, 1]  # (P,)

    # Vanishing point proxy: road centroid x position in the first frame of each pair
    road_cx = centroids[0::2, 0, 1]  # (P,) road center-x in even-indexed (first) frames

    # Simple linear model for pose estimation:
    # These are heuristic coefficients tuned to match typical PoseNet output
    # magnitudes on comma driving data. The exact values matter less than
    # providing a reasonable initial target for the optimization to converge from.
    # Override via pose_heuristic_weights with actual PoseNet targets from
    # ground-truth frames for production use.
    _defaults = {
        "tx_road_dx": 0.5,
        "ty_road_dy": 0.3,
        "tz_baseline": 0.1,
        "tz_road_dy": 0.2,
        "rx_my_car_dy": 0.2,
        "ry_my_car_dx": 0.3,
        "rz_road_cx": 0.1,
    }
    overrides = dict(pose_heuristic_weights or {})
    legacy_aliases = {
        "rx_sky_dy": "rx_my_car_dy",
        "ry_sky_dx": "ry_my_car_dx",
    }
    for old_key, new_key in legacy_aliases.items():
        if old_key in overrides and new_key not in overrides:
            overrides[new_key] = overrides[old_key]
    w = {**_defaults, **overrides}

    poses = torch.zeros(P, 6, device=device)  # OFF_MANIFOLD_OK: heuristic constructor — ALL 6 dims populated immediately below from centroid deltas (see lines 812-817).
    poses[:, 0] = road_dx * w["tx_road_dx"]            # tx: lateral from road shift
    poses[:, 1] = road_dy * w["ty_road_dy"]             # ty: vertical from road shift
    poses[:, 2] = w["tz_baseline"] + road_dy * w["tz_road_dy"]  # tz: forward (baseline + road growth)
    poses[:, 3] = my_car_dy * w["rx_my_car_dy"]         # rx: pitch proxy from ego-car-mask shift
    poses[:, 4] = my_car_dx * w["ry_my_car_dx"]         # ry: yaw proxy from ego-car-mask shift
    poses[:, 5] = (road_cx - 0.5) * w["rz_road_cx"]    # rz: roll from road asymmetry

    return poses


def constrained_generate(
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    noise_seed: int = 42,
    num_steps: int = 1000,
    # [empirical: high-LR analytical-step regime per constrained_gen family;
    # arbitrariness_audit_20260512.md A-3 flagged as DEFERRED-pending-council
    # rederivation against current optimizer + step-scheduler]
    lr: float = 0.1,
    # [DEFERRED-pending-council: arbitrariness_audit_20260512.md A-1; chosen
    # at old 1.x score operating point; at PR106 r2 frontier the marginal
    # value FLIPS per CLAUDE.md "SegNet vs PoseNet importance —
    # operating-point dependent". Council deliberation required to
    # re-derive from current frontier marginals. DO NOT silently change.]
    seg_weight: float = 50.0,
    # [DEFERRED-pending-council: sister of seg_weight per A-1; same caveat]
    pose_weight: float = 50.0,
    compress_weight: float = 1.0,
    device: torch.device | str = "cpu",
    log_every: int = 100,
    early_stop_patience: int = 50,
    early_stop_delta: float = 1e-4,
    segnet_batch_size: int = 32,
    posenet_batch_size: int = 32,
    init_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Main constrained optimization loop: generate frames from masks.

    Starting from class-mean-colored noise (or *init_frames* if provided),
    optimize pixel values via gradient descent to simultaneously satisfy:
      1. SegNet constraint: argmax matches target masks
      2. PoseNet constraint: pose output matches expected targets
      3. Compressibility: total variation is minimized

    Mini-batched internally to fit on T4 (16GB) for 1200 frames.
    Early stopping when loss improvement < early_stop_delta for
    early_stop_patience consecutive steps (set to 0 to disable).

    Args:
        masks: (N, H, W) long tensor with class indices.
        expected_pose: (P, 6) float tensor, P = N//2 (non-overlapping pairs, seq_len=2).
        posenet: frozen PoseNet model (on device).
        segnet: frozen SegNet model (on device).
        noise_seed: deterministic seed for initialization (ignored if init_frames set).
        num_steps: maximum Adam optimization steps.
        lr: learning rate for Adam optimizer.
        seg_weight: weight for SegNet cross-entropy constraint.
        pose_weight: weight for PoseNet L2 constraint. Council binding:
            both seg and pose at 50 — PoseNet is 69% of the current score gap.
        compress_weight: weight for total variation loss.
        device: computation device.
        log_every: print loss every N steps (0 to disable).
        early_stop_patience: stop if no improvement for this many steps (0 = disabled).
            Default 50 may be too aggressive for first runs — disable until convergence
            behavior is understood empirically.
        early_stop_delta: minimum loss improvement to reset patience counter.
        segnet_batch_size: frames per SegNet mini-batch (reduce if OOM).
        posenet_batch_size: pairs per PoseNet mini-batch (reduce if OOM).
        init_frames: (N, H, W, 3) float tensor to warm-start from instead of noise.
            Enables two-phase optimization: run N steps, evaluate, continue from result.

    Returns:
        (N, H, W, 3) float tensor in [0, 255], rounded to uint8-compatible values.
    """
    device = torch.device(device)

    # Initialize frames — from provided warm-start or from masks + seed
    if init_frames is not None:
        frames = init_frames.to(device).detach().clone()
    else:
        frames = generate_initial_frames(masks, noise_seed, device=device)
    frames.requires_grad_(True)

    optimizer = torch.optim.Adam([frames], lr=lr)

    best_loss = float("inf")
    no_improve_count = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # Mini-batched constraint losses — gradients accumulate on frames.grad
        seg_loss = compute_segnet_constraint_loss(
            frames, masks, segnet, batch_size=segnet_batch_size,
        )
        compress_loss = compute_compressibility_loss(frames)

        # PoseNet loss only if we have pairs (N > 1)
        if frames.shape[0] > 1 and expected_pose.shape[0] > 0:
            pose_loss = compute_posenet_constraint_loss(
                frames, expected_pose, posenet, batch_size=posenet_batch_size,
            )
        else:
            pose_loss = torch.tensor(0.0, device=device)

        # Weighted combination
        total_loss = (
            seg_weight * seg_loss
            + pose_weight * pose_loss
            + compress_weight * compress_loss
        )

        total_loss.backward()
        optimizer.step()

        # Project back to valid pixel range
        with torch.no_grad():
            frames.data.clamp_(0.0, 255.0)

        loss_val = total_loss.item()

        if log_every > 0 and (step + 1) % log_every == 0:
            logger.info(
                "  step %4d/%d: total=%.4f seg=%.4f pose=%.4f compress=%.4f",
                step + 1, num_steps, loss_val, seg_loss.item(),
                pose_loss.item(), compress_loss.item(),
            )

        # Early stopping (disabled when early_stop_patience == 0)
        if early_stop_patience > 0:
            if best_loss - loss_val > early_stop_delta:
                best_loss = loss_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    if log_every > 0:
                        logger.info(
                            "  Early stop at step %d: no improvement for %d steps (best=%.4f)",
                            step + 1, early_stop_patience, best_loss,
                        )
                    break

    # Final quantization: round to nearest integer, clamp to uint8 range
    with torch.no_grad():
        result = frames.detach().round().clamp(0.0, 255.0)

    return result


def build_constrained_archive(
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    noise_seed: int,
    output_path: str | Path,
) -> Path:
    """Build a minimal archive for constrained-generation inflate.

    The archive contains three files:
      - masks.bin: LZMA-compressed uint8 mask tensor (~239 bytes for typical data)
      - pose_targets.bin: float16 pose targets (~7KB for 1200 frames)
      - seed.bin: 64-byte noise seed

    Total archive size: ~8KB (vs ~100KB+ for neural renderer weights).

    Args:
        masks: (N, H, W) long tensor with class indices.
        expected_pose: (P, 6) float tensor.
        noise_seed: integer seed.
        output_path: directory to write archive files into.

    Returns:
        Path to the output directory.
    """
    import lzma

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Masks: convert to uint8, flatten, LZMA compress
    masks_np = masks.cpu().numpy().astype(np.uint8)
    masks_bytes = masks_np.tobytes()
    # Store shape header (3 x uint32) + compressed data
    shape_header = struct.pack("<III", *masks_np.shape)
    compressed = lzma.compress(masks_bytes, preset=9)
    (out / "masks.bin").write_bytes(shape_header + compressed)

    # 2. Pose targets: float16 for space efficiency
    pose_np = expected_pose.cpu().to(torch.float16).numpy()
    pose_header = struct.pack("<II", *pose_np.shape)
    (out / "pose_targets.bin").write_bytes(pose_header + pose_np.tobytes())

    # 3. Seed: 64 bytes (uint64)
    (out / "seed.bin").write_bytes(struct.pack("<Q", noise_seed))

    # Metadata for reproducibility
    meta = {
        "num_frames": int(masks.shape[0]),
        "mask_shape": list(masks.shape),
        "pose_shape": list(expected_pose.shape),
        "noise_seed": noise_seed,
        "compressed_masks_bytes": len(shape_header + compressed),
        "pose_bytes": len(pose_header + pose_np.tobytes()),
        "total_bytes": (
            len(shape_header + compressed)
            + len(pose_header + pose_np.tobytes())
            + 8  # seed
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    total = meta["total_bytes"]
    logger.info("Constrained archive: %s bytes (%.1f KB)", f"{total:,}", total / 1024)
    logger.info("  masks: %s bytes", f"{meta['compressed_masks_bytes']:,}")
    logger.info("  poses: %s bytes", f"{meta['pose_bytes']:,}")
    logger.info("  seed:  8 bytes")

    return out


def load_constrained_archive(
    archive_dir: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load a constrained-generation archive.

    Args:
        archive_dir: directory containing masks.bin, pose_targets.bin, seed.bin.

    Returns:
        (masks, expected_pose, noise_seed) tuple.
    """
    import lzma

    d = Path(archive_dir)

    # Masks
    masks_data = (d / "masks.bin").read_bytes()
    N, H, W = struct.unpack("<III", masks_data[:12])
    masks_bytes = lzma.decompress(masks_data[12:])
    masks = torch.from_numpy(
        np.frombuffer(masks_bytes, dtype=np.uint8).reshape(N, H, W).copy()
    ).long()

    # Pose targets
    pose_data = (d / "pose_targets.bin").read_bytes()
    P, D = struct.unpack("<II", pose_data[:8])
    pose_np = np.frombuffer(pose_data[8:], dtype=np.float16).reshape(P, D).copy()
    expected_pose = torch.from_numpy(pose_np).float()

    # Seed
    seed_data = (d / "seed.bin").read_bytes()
    noise_seed = struct.unpack("<Q", seed_data)[0]

    return masks, expected_pose, noise_seed


def inflate_constrained(
    archive_dir: str | Path,
    output_dir: str | Path,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    num_steps: int = 1000,
    # [empirical: constrained-gen family canonical LR; same regime as
    # constrained_generate() above; arbitrariness_audit A-3 DEFERRED for
    # operator-conditioned rederivation]
    lr: float = 0.1,
    device: torch.device | str = "cpu",
    log_every: int = 100,
) -> Path:
    """Inflate function: run constrained optimization at test time.

    Loads the archive, runs gradient descent to generate frames, and
    writes the result as uint8 numpy arrays.

    Args:
        archive_dir: directory containing the constrained archive.
        output_dir: directory to write generated frames.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        num_steps: optimization steps (more = better quality, slower).
        lr: Adam learning rate.
        device: computation device.
        log_every: print loss every N steps.

    Returns:
        Path to output directory.
    """
    masks, expected_pose, noise_seed = load_constrained_archive(archive_dir)
    masks = masks.to(device)
    expected_pose = expected_pose.to(device)

    N = masks.shape[0]
    logger.info("Inflating %d frames via constrained optimization...", N)
    logger.info("  steps=%d, lr=%s, seed=%d", num_steps, lr, noise_seed)

    frames = constrained_generate(
        masks=masks,
        expected_pose=expected_pose,
        posenet=posenet,
        segnet=segnet,
        noise_seed=noise_seed,
        num_steps=num_steps,
        lr=lr,
        device=device,
        log_every=log_every,
    )

    # Write frames as raw RGB24 — the format the upstream scorer expects.
    # Each frame is H×W×3 uint8, concatenated with no header/separator.
    # Shape: (N, H, W, 3) -> flatten to bytes in frame-major order.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_np = frames.cpu().numpy().astype(np.uint8)  # (N, H, W, 3)
    raw_path = out / "frames.raw"
    frames_np.tofile(raw_path)
    n_bytes = raw_path.stat().st_size
    logger.info("Wrote %d frames (%s bytes) to %s", N, f"{n_bytes:,}", raw_path)

    return out


# ---- GPU Eureka #1: Generate in scorer space ----


def inverse_preprocess_input(
    scorer_space_tensor: torch.Tensor,
    target_h: int = CAMERA_H,
    target_w: int = CAMERA_W,
) -> torch.Tensor:
    """Invert the scorer preprocessing to recover approximate RGB frames.

    The scorer preprocessing pipeline is:
      RGB (B, C, H, W) -> resize to (384, 512) -> rgb_to_yuv6 -> (B, 6, 192, 256)

    This function inverts that pipeline:
      (B, 6, 192, 256) -> yuv6_to_rgb -> (B, 3, 384, 512) -> resize to (H, W)

    The inversion is approximate because:
      1. Chroma subsampling in YUV420 loses spatial information.
      2. Bilinear resize is not perfectly invertible.

    Args:
        scorer_space_tensor: (B, 6, H2, W2) tensor in scorer's internal
            YUV420 representation (output of rgb_to_yuv6 after resize).
        target_h: target height for output RGB frames.
        target_w: target width for output RGB frames.

    Returns:
        (B, 3, target_h, target_w) float tensor in [0, 255].
    """
    # Step 1: YUV6 -> RGB at half-resolution
    rgb_small = yuv6_to_rgb(scorer_space_tensor)  # (B, 3, H, W) at 384x512 scale

    # Step 2: Resize to target resolution
    if rgb_small.shape[-2] != target_h or rgb_small.shape[-1] != target_w:
        rgb_full = F.interpolate(
            rgb_small,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    else:
        rgb_full = rgb_small

    return rgb_full.clamp(0.0, 255.0)


def generate_in_scorer_space(
    masks: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    noise_seed: int = 42,
    num_steps: int = 1000,
    lr: float = 0.01,
    # [derived: closed-form gradient per src/tac/score_geometry.py:253-257]
    # Legacy (seg=100, pose=10) defaults reflect OLD 1.x operating-point
    # marginals; at PR106 r2 (d_pose ~3.4e-5) the marginal value FLIPS to
    # pose-dominant. Use ``tools/probe_seg_pose_weight_at_operating_point.py``
    # to compute operating-point-aware weights. Council A-1 (OOO commit
    # 328bf2f9, 8/10 FOR Option C, 2026-05-12).
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
    device: torch.device | str = "cpu",
    log_every: int = 100,
    expected_pose: torch.Tensor | None = None,
    use_embedding_loss: bool = False,
    expected_pose_embeddings: torch.Tensor | None = None,
    seg_odd_only: bool = False,
) -> torch.Tensor:
    """Generate scorer-optimal frames via coupled trajectory optimization.

    Delegates to coupled_trajectory_optimize with all improvements:
    - Coupled pair optimization (PoseNet gradient through both frames)
    - Compress weight annealing (prevents PoseNet divergence)
    - PoseNet transient snapshot (returns best PoseNet state)
    - Embedding loss for 42x richer PoseNet gradient (use_embedding_loss)
    - Odd-frame-only SegNet loss matching official scorer (seg_odd_only)

    If expected_pose is not provided, falls back to heuristic estimation
    from mask dynamics (less accurate — prefer GT targets from posenet_targets.bin).

    Key insight from scorer analysis:
    - SegNet uses raw RGB (no YUV). preprocess_input just resizes.
    - PoseNet converts to YUV6 internally via its own preprocess_input.
    - Optimizing in RGB is correct — both scorers handle their own preprocessing.
    - The original YUV6 parameterization was a red herring.

    Args:
        masks: (N, H, W) long tensor with class indices.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        noise_seed: deterministic seed for initialization.
        num_steps: optimization steps.
        lr: Adam learning rate.
        seg_weight: SegNet constraint weight.
        pose_weight: PoseNet constraint weight.
        compress_weight: compressibility weight.
        device: computation device.
        log_every: print loss every N steps.
        expected_pose: (P, 6) GT pose targets. If None, uses heuristic estimation.
        use_embedding_loss: use PoseNet embedding MSE instead of pose output MSE.
        expected_pose_embeddings: (P, D) GT embeddings. Required if use_embedding_loss.
        seg_odd_only: only compute SegNet loss on odd frames (scorer-matched).

    Returns:
        (N, H, W, 3) float tensor in [0, 255] (RGB, HWC format).
    """
    # Use GT pose targets if provided, otherwise fall back to heuristic
    if expected_pose is None:
        logger.warning("  [scorer-space] Using heuristic pose targets (inaccurate). "
                       "Pass expected_pose from GT for best results.")
        expected_pose = estimate_expected_pose(masks, device=device)

    # Delegate to coupled_trajectory_optimize which has all improvements:
    # - Coupled pair optimization (Fix #2)
    # - Compress weight annealing (Fix #4a)
    # - PoseNet transient snapshot (Fix #4b)
    # Fix #1 (lossy round-trip) is moot: SegNet uses raw RGB, PoseNet converts
    # internally. Optimizing in RGB and letting scorers preprocess is correct.
    # Fix #3 (GT targets) is handled by the expected_pose parameter above.
    return coupled_trajectory_optimize(
        masks=masks,
        expected_pose=expected_pose,
        posenet=posenet,
        segnet=segnet,
        num_steps=num_steps,
        lr=lr,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
        compress_weight=compress_weight,
        noise_seed=noise_seed,
        device=str(device),
        log_every=log_every,
        use_embedding_loss=use_embedding_loss,
        expected_pose_embeddings=expected_pose_embeddings,
        seg_odd_only=seg_odd_only,
    )


# ---- High-level interface ----


class ConstrainedFrameGenerator:
    """Generate scorer-optimal frames via constrained optimization from noise.

    No neural network weights needed. Archive contains only:
    - masks (239 bytes via entropy coder)
    - expected PoseNet targets (7KB)
    - noise seed (64 bytes)
    Total: ~8KB archive.

    At inflate time: run gradient descent from seeded noise.
    ~1000 steps, ~50ms/step on T4 = 50 seconds.

    Example::

        gen = ConstrainedFrameGenerator(posenet, segnet, device="cuda")
        frames = gen.generate(masks, noise_seed=42, num_steps=1000)
    """

    def __init__(
        self,
        posenet: torch.nn.Module,
        segnet: torch.nn.Module,
        device: torch.device | str = "cpu",
    ) -> None:
        self.posenet = posenet
        self.segnet = segnet
        self.device = torch.device(device)

    def generate(
        self,
        masks: torch.Tensor,
        noise_seed: int = 42,
        num_steps: int = 1000,
        # [empirical: constrained-gen family canonical LR; arbitrariness A-3]
        lr: float = 0.1,
        # [empirical: 100/10 seg/pose split matches "old 1.x SegNet 77×" regime;
        # arbitrariness A-1 DEFERRED-pending-council for PR106 r2 frontier]
        seg_weight: float = 100.0,
        pose_weight: float = 10.0,
        compress_weight: float = 1.0,
        log_every: int = 100,
        scorer_space: bool = False,
    ) -> torch.Tensor:
        """Generate frames satisfying scorer constraints.

        Args:
            masks: (N, H, W) long tensor with target class indices.
            noise_seed: deterministic seed.
            num_steps: optimization steps.
            lr: Adam learning rate.
            seg_weight: SegNet cross-entropy weight.
            pose_weight: PoseNet L2 weight.
            compress_weight: total variation weight.
            log_every: logging interval (0 to disable).
            scorer_space: if True, optimize in scorer's YUV420 space
                (GPU Eureka #1) instead of RGB space.

        Returns:
            (N, H, W, 3) float tensor in [0, 255].
        """
        expected_pose = estimate_expected_pose(masks, device=self.device)

        if scorer_space:
            return generate_in_scorer_space(
                masks=masks,
                posenet=self.posenet,
                segnet=self.segnet,
                noise_seed=noise_seed,
                num_steps=num_steps,
                lr=lr,
                seg_weight=seg_weight,
                pose_weight=pose_weight,
                compress_weight=compress_weight,
                device=self.device,
                log_every=log_every,
            )
        else:
            return constrained_generate(
                masks=masks,
                expected_pose=expected_pose,
                posenet=self.posenet,
                segnet=self.segnet,
                noise_seed=noise_seed,
                num_steps=num_steps,
                lr=lr,
                seg_weight=seg_weight,
                pose_weight=pose_weight,
                compress_weight=compress_weight,
                device=self.device,
                log_every=log_every,
            )

    def build_archive(
        self,
        masks: torch.Tensor,
        noise_seed: int,
        output_path: str | Path,
    ) -> Path:
        """Extract pose targets from masks and build minimal archive.

        Args:
            masks: (N, H, W) long tensor.
            noise_seed: deterministic seed.
            output_path: output directory.

        Returns:
            Path to archive directory.
        """
        expected_pose = estimate_expected_pose(masks, device=self.device)
        return build_constrained_archive(masks, expected_pose, noise_seed, output_path)

    def inflate(
        self,
        archive_dir: str | Path,
        output_dir: str | Path,
        num_steps: int = 1000,
        # [empirical: constrained-gen family canonical LR; arbitrariness A-3]
        lr: float = 0.1,
        log_every: int = 100,
    ) -> Path:
        """Inflate from archive via constrained optimization.

        Args:
            archive_dir: path to archive directory.
            output_dir: path to write output frames.
            num_steps: optimization steps.
            lr: Adam learning rate.
            log_every: logging interval.

        Returns:
            Path to output directory.
        """
        return inflate_constrained(
            archive_dir=archive_dir,
            output_dir=output_dir,
            posenet=self.posenet,
            segnet=self.segnet,
            num_steps=num_steps,
            lr=lr,
            device=self.device,
            log_every=log_every,
        )

    def generate_full_pipeline(
        self,
        masks: torch.Tensor,
        noise_seed: int = 42,
        cfg: dict | None = None,
        log_every: int = 50,
    ) -> tuple[torch.Tensor, dict]:
        """Full constrained generation pipeline combining all mathematical tools.

        The pipeline integrates all five researcher contributions:

        1. **Initialization** (Yousfi): Class-mean colors + seeded noise, or
           warm-start from mask-based low-rank approximation.
        2. **Variational optimization** (Euler): Solve Euler-Lagrange equations
           via gradient descent with smoothness regularization.
        3. **Lagrangian dual** (Lagrange): Learn optimal rate-distortion lambda
           via primal-dual updates satisfying KKT conditions.
        4. **Manifold projection** (Tao): After each phase, project frames
           onto scorer null-space to reduce rate at zero distortion cost.
        5. **Hamiltonian refinement** (Karpathy): Phase-space dynamics for
           escaping local minima in the final refinement stage.
        6. **Quantization** (Yousfi): Round to uint8 using gradient-directed
           Floyd-Steinberg dithering.

        Args:
            masks: (N, H, W) long tensor with class indices.
            noise_seed: deterministic seed.
            cfg: pipeline configuration dict. Keys:
                - phase1_steps (int): variational steps, default 200
                - phase2_steps (int): Lagrangian dual steps, default 200
                - phase3_steps (int): Hamiltonian refinement steps, default 100
                - rate_budget (float): rate constraint for dual, default 0.01
                - manifold_project (bool): enable null-space projection, default True
                - use_hamiltonian (bool): enable phase-space refinement, default True
                - use_dithering (bool): gradient-directed dithering, default True
            log_every: logging interval.

        Returns:
            (frames, diagnostics) where frames is (N, H, W, 3) in [0, 255]
            and diagnostics is a dict with per-phase metrics.
        """
        from tac.contrib.hamiltonian_dynamics import HamiltonianPixelOptimizer
        from tac.contrib.scorer_manifold import ScorerManifold
        from tac.contrib.variational_gen import LagrangianDualOptimizer, VariationalFrameGenerator

        cfg = cfg or {}
        device = self.device
        N, H_m, W_m = masks.shape

        phase1_steps = cfg.get("phase1_steps", 200)
        phase2_steps = cfg.get("phase2_steps", 200)
        phase3_steps = cfg.get("phase3_steps", 100)
        rate_budget = cfg.get("rate_budget", 0.01)
        do_manifold = cfg.get("manifold_project", True)
        do_hamiltonian = cfg.get("use_hamiltonian", True)
        do_dithering = cfg.get("use_dithering", True)

        diagnostics: dict = {"phases": {}}

        # ---- Phase 0: Initialization ----
        init_frames_hwc = generate_initial_frames(masks, noise_seed, device=device)
        # Convert HWC -> CHW for the pipeline
        frames = init_frames_hwc.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
        if log_every > 0:
            logger.info("  pipeline: initialized %d frames (%dx%d)", N, H_m, W_m)

        # ---- Phase 1: Variational optimization (Euler) ----
        var_gen = VariationalFrameGenerator(cfg={
            "variational_steps": phase1_steps,
            "variational_lr": 0.5,
            "lambda_smooth": cfg.get("lambda_smooth", 0.01),
            "lambda_rate": cfg.get("lambda_rate", 0.1),
            "use_line_search": False,
        })
        frames = var_gen.solve(
            frames, masks, self.posenet, self.segnet, log_every=log_every,
        )
        diagnostics["phases"]["variational"] = {"steps": phase1_steps}
        if log_every > 0:
            logger.info("  pipeline: Phase 1 (variational) complete")

        # ---- Phase 1.5: Manifold projection (Tao) ----
        if do_manifold and N >= 1:
            manifold = ScorerManifold(self.posenet, self.segnet, cfg={
                "max_jacobian_outputs": cfg.get("max_jacobian_outputs", 16),
                "rank_threshold": cfg.get("rank_threshold", 1e-3),
            })
            for i in range(N):
                single = frames[i: i + 1]
                J = manifold.compute_jacobian(single)
                smoothed = F.avg_pool2d(
                    F.pad(single, (1, 1, 1, 1), mode="replicate"), 3, stride=1,
                )
                delta = smoothed - single
                delta_flat = delta.reshape(-1)
                proj_delta = manifold.null_space_project(delta_flat, J)
                frames[i] = (
                    single.reshape(-1) + proj_delta.to(device)
                ).reshape(single.shape).clamp(0.0, 255.0)
            diagnostics["phases"]["manifold_projection"] = {"frames_projected": N}
            if log_every > 0:
                logger.info("  pipeline: Phase 1.5 (manifold projection) complete")

        # ---- Phase 2: Lagrangian dual optimization (Lagrange) ----
        dual_opt = LagrangianDualOptimizer(cfg={
            "dual_steps": phase2_steps,
            "dual_primal_lr": 0.3,
            "dual_dual_lr": 0.01,
            "rate_budget": rate_budget,
            "lambda_init": 25.0,
            "lambda_smooth": cfg.get("lambda_smooth", 0.01),
        })
        frames, dual_diag = dual_opt.optimize(
            frames, masks, self.posenet, self.segnet,
            rate_budget=rate_budget, log_every=log_every,
        )
        diagnostics["phases"]["lagrangian_dual"] = dual_diag
        if log_every > 0:
            logger.info("  pipeline: Phase 2 (Lagrangian dual) complete, lambda=%.3f",
                        dual_diag['final_lambda'])

        # ---- Phase 3: Hamiltonian refinement (Karpathy) ----
        if do_hamiltonian:
            ham_opt = HamiltonianPixelOptimizer(cfg={
                "hamiltonian_steps": phase3_steps,
                "hamiltonian_dt": cfg.get("hamiltonian_dt", 0.05),
                "hamiltonian_mass": cfg.get("hamiltonian_mass", 1.0),
                "anneal_damping": True,
                "anneal_start": 0.001,
                "anneal_end": 0.05,
            })
            frames, ham_diag = ham_opt.optimize_with_scorer(
                frames, masks, self.posenet, self.segnet,
                seg_weight=cfg.get("seg_weight", 100.0),
                pose_weight=cfg.get("pose_weight", 10.0),
                smooth_weight=cfg.get("smooth_weight", 0.01),
                log_every=log_every,
            )
            diagnostics["phases"]["hamiltonian"] = {
                "best_potential": ham_diag["best_potential"],
                "energy_drift": ham_diag["energy_drift"],
                "steps": ham_diag["steps"],
            }
            if log_every > 0:
                logger.info("  pipeline: Phase 3 (Hamiltonian) complete, V=%.4f",
                            ham_diag['best_potential'])

        # ---- Phase 4: Gradient-directed quantization ----
        if do_dithering:
            frames = _gradient_directed_dither(
                frames, masks, self.segnet, device,
            )
            diagnostics["phases"]["dithering"] = {"applied": True}
            if log_every > 0:
                logger.info("  pipeline: Phase 4 (gradient dithering) complete")
        else:
            frames = frames.round().clamp(0.0, 255.0)

        # Convert CHW -> HWC for output
        result = frames.permute(0, 2, 3, 1).contiguous().clamp(0.0, 255.0)
        return result, diagnostics


def scorer_as_compressor(
    frames: torch.Tensor,  # GT frames (N, H, W, 3)
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: str = "cpu",
    topk: int = 2,
    batch_size: int = 4,
) -> dict:
    """Extract the scorer's sufficient statistic as a compressed representation.

    The scorer networks already learned optimal compression for driving video.
    PoseNet: 2 frames -> 6 numbers.  SegNet: 1 frame -> 96x128x5 logits.
    Store scorer OUTPUTS in the archive directly.  At inflate time, find frames
    that decompress to match.

    Returns dict with:
        'posenet_targets': (N//2, 6) float16 PoseNet outputs — non-overlapping pairs
        'segnet_masks': (N, H_out, W_out) uint8 argmax masks
        'segnet_logits_topk': (N, topk, H_out, W_out) float16 top-K logits
        'segnet_logits_topk_idx': (N, topk, H_out, W_out) uint8 top-K class indices

    Total size: ~15KB (14KB pose + 239B masks + optional logits).
    This IS the archive.  The scorer's outputs are the sufficient statistic.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], HWC format.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        topk: number of top SegNet logits to store per pixel (higher = more fidelity).
        batch_size: pairs per forward pass.

    Returns:
        Dict with scorer sufficient statistics.
    """
    device = torch.device(device)
    N = frames.shape[0]

    # --- PoseNet targets: extract for NON-OVERLAPPING pairs (seq_len=2) ---
    # Upstream DaliVideoDataset uses non-overlapping pairs (f0,f1),(f2,f3),...
    # N//2 pairs total — NOT N-1 consecutive pairs.
    all_pose = []
    num_pairs = N // 2
    frames_chw = frames.permute(0, 3, 1, 2).contiguous().to(device)  # (N, 3, H, W)

    with torch.inference_mode():
        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            t0 = frames_chw[2 * start:2 * end:2]       # (B, 3, H, W) — even frames
            t1 = frames_chw[2 * start + 1:2 * end + 1:2]  # (B, 3, H, W) — odd frames
            pairs = torch.stack([t0, t1], dim=1).contiguous()  # (B, 2, 3, H, W)
            posenet_in = posenet.preprocess_input(pairs)
            pose_out = posenet(posenet_in)
            pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            all_pose.append(pose_tensor[..., :6].cpu())

    posenet_targets = torch.cat(all_pose, dim=0).to(torch.float16) if all_pose else torch.empty(0, 6, dtype=torch.float16)

    # --- SegNet: extract masks and top-K logits ---
    all_masks = []
    all_topk_vals = []
    all_topk_idx = []

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch = frames_chw[i:end]  # (B, 3, H, W)
            # SegNet expects (B, T, C, H, W) via preprocess_input
            seg_btchw = batch.unsqueeze(1).contiguous()  # (B, 1, 3, H, W)
            seg_in = segnet.preprocess_input(seg_btchw)
            logits = segnet(seg_in)  # (B, NUM_CLASSES, H_out, W_out)

            masks = logits.argmax(dim=1).to(torch.uint8).cpu()  # (B, H_out, W_out)
            all_masks.append(masks)

            # Top-K logits for higher fidelity reconstruction
            K = min(topk, logits.shape[1])
            topk_vals, topk_indices = logits.topk(K, dim=1)  # (B, K, H_out, W_out)
            all_topk_vals.append(topk_vals.cpu().to(torch.float16))
            all_topk_idx.append(topk_indices.cpu().to(torch.uint8))

    segnet_masks = torch.cat(all_masks, dim=0)         # (N, H_out, W_out)
    segnet_topk_vals = torch.cat(all_topk_vals, dim=0)  # (N, K, H_out, W_out)
    segnet_topk_idx = torch.cat(all_topk_idx, dim=0)    # (N, K, H_out, W_out)

    return {
        "posenet_targets": posenet_targets,
        "segnet_masks": segnet_masks,
        "segnet_logits_topk": segnet_topk_vals,
        "segnet_logits_topk_idx": segnet_topk_idx,
    }


def yuv_null_space_projection(
    frames: torch.Tensor,
    segnet_grad: torch.Tensor,
    step_size: float = 1.0,
) -> torch.Tensor:
    """Project SegNet gradient into PoseNet null space and apply.

    The PoseNet preprocess converts RGB to YUV6 with 4:2:0 chroma subsampling.
    Specifically, U and V channels are averaged over 2x2 blocks. Perturbations
    to the RGB that create zero-mean changes within each 2x2 block in the
    chroma (U, V) channels are invisible to PoseNet but can improve SegNet.

    Null space structure:
        Y = 0.299*R + 0.587*G + 0.114*B  (full resolution, every pixel matters)
        U = (B - Y) / 1.772 + 128  →  averaged over 2x2 blocks
        V = (R - Y) / 1.402 + 128  →  averaged over 2x2 blocks

        Within each 2x2 block, perturbations to U (or V) that sum to zero
        vanish after averaging. That gives 3 DOF per 2x2 block per chroma
        channel (6 null dimensions per 4-pixel block in chroma space).

        Additionally, any change purely in the Y high-frequency component
        (differences within a 2x2 block) that doesn't affect U/V averages
        is also invisible to PoseNet's chroma path.

    Implementation: for each 2x2 block, subtract the block mean from the
    SegNet gradient's U/V components, then convert back to RGB space.
    This residual is in PoseNet's null space by construction.

    Args:
        frames: (N, H, W, 3) current frames in [0, 255]
        segnet_grad: (N, H, W, 3) gradient of SegNet loss w.r.t. frames
        step_size: scaling factor for the null-space step

    Returns:
        (N, H, W, 3) updated frames with null-space perturbation applied
    """
    N, H, W, _ = frames.shape
    H2, W2 = H // 2 * 2, W // 2 * 2  # Ensure even dimensions

    # Work in CHW for convenience
    grad_chw = segnet_grad[:, :H2, :W2, :].permute(0, 3, 1, 2)  # (N, 3, H2, W2)

    # Convert gradient to YUV space
    R_g, G_g, B_g = grad_chw[:, 0], grad_chw[:, 1], grad_chw[:, 2]
    kYR, kYG, kYB = 0.299, 0.587, 0.114

    # dU/dRGB and dV/dRGB (Jacobian of chroma w.r.t. RGB)
    # U = (B - Y) / 1.772, where Y = 0.299R + 0.587G + 0.114B
    # dU/dR = -0.299/1.772, dU/dG = -0.587/1.772, dU/dB = (1-0.114)/1.772
    # V = (R - Y) / 1.402
    # dV/dR = (1-0.299)/1.402, dV/dG = -0.587/1.402, dV/dB = -0.114/1.402

    # Project gradient into U, V components
    U_grad = (-kYR * R_g - kYG * G_g + (1 - kYB) * B_g) / 1.772
    V_grad = ((1 - kYR) * R_g - kYG * G_g - kYB * B_g) / 1.402

    # For each 2x2 block, subtract the block mean (null space = zero-sum within block)
    # Reshape to blocks: (N, H2//2, 2, W2//2, 2)
    U_blocks = U_grad.reshape(N, H2 // 2, 2, W2 // 2, 2)
    V_blocks = V_grad.reshape(N, H2 // 2, 2, W2 // 2, 2)

    # Block means (what PoseNet would see)
    U_mean = U_blocks.mean(dim=(2, 4), keepdim=True)
    V_mean = V_blocks.mean(dim=(2, 4), keepdim=True)

    # Null-space component: gradient minus what PoseNet sees
    U_null = (U_blocks - U_mean).reshape(N, H2, W2)
    V_null = (V_blocks - V_mean).reshape(N, H2, W2)

    # Convert null-space UV gradient back to RGB space
    # Solve: given dU_null and dV_null, find dR, dG, dB
    # Using the pseudoinverse of the YUV Jacobian restricted to chroma:
    # dR = -1.772 * kYR/(1-kYR-kYB) * dU + 1.402 * (1-kYR)/(1-kYR-kYB) * dV
    # ... but simpler: use the direct chroma-to-RGB inverse
    # For null space in U: dB - dY = 1.772 * dU_null, and we want minimal RGB change
    # Simplest correct approach: perturb only B for U-null, only R for V-null
    # (these are the dominant contributors to U and V respectively)

    # Minimal perturbation: B carries U, R carries V (approximate but correct null space)
    dR_null = 1.402 * V_null  # V = (R-Y)/1.402, so dR ≈ 1.402*dV for V-null
    dB_null = 1.772 * U_null  # U = (B-Y)/1.772, so dB ≈ 1.772*dU for U-null
    # Compensate Y to keep Y unchanged: dY = kYR*dR + kYB*dB, solve for dG
    dG_null = -(kYR * dR_null + kYB * dB_null) / kYG

    # Stack and apply (descend along SegNet gradient in PoseNet null space)
    null_step = torch.stack([dR_null, dG_null, dB_null], dim=1)  # (N, 3, H2, W2)
    null_step_hwc = null_step.permute(0, 2, 3, 1)  # (N, H2, W2, 3)

    # Apply step (negative because we're minimizing SegNet loss)
    result = frames.clone()
    result[:, :H2, :W2, :] = result[:, :H2, :W2, :] - step_size * null_step_hwc
    result.clamp_(0.0, 255.0)
    return result


def coupled_trajectory_optimize(
    masks: torch.Tensor,  # (N, H, W)
    expected_pose: torch.Tensor,  # (P, 6) where P = N//2 (non-overlapping pairs)
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    num_steps: int = 1000,
    lr: float = 0.01,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
    noise_seed: int = 42,
    device: str = "cuda",
    log_every: int = 100,
    init_frames: torch.Tensor | None = None,
    early_stop_patience: int = 150,
    use_embedding_loss: bool = False,
    expected_pose_embeddings: torch.Tensor | None = None,
    seg_odd_only: bool = False,
    antialias_weight: float = 0.0,
    lr_schedule: str = "constant",
    segnet_loss_mode: str = "xent",
    hinge_margin: float = 0.5,
    phase2_segnet_only: bool = False,
    phase2_steps: int = 200,
    eval_roundtrip: bool = True,
    roundtrip_noise_std: float = 0.5,
    use_null_space: bool = False,
    null_space_step: float = 0.5,
    null_space_every: int = 10,
    pair_weights: torch.Tensor | None = None,
    segnet_class_weights: torch.Tensor | str | None = None,
    **cfg,
) -> torch.Tensor:
    """Jointly optimize ALL frames to satisfy coupled PoseNet constraints.

    Unlike independent frame optimization, this solves the coupled system:
    for each frame t, minimize seg_loss(t) + pose_loss(t-1,t) + pose_loss(t,t+1)
    simultaneously across all frames.

    This is equivalent to 4D-Var data assimilation with the temporal coupling
    as the forward model and scorer evaluations as observations.

    Key insight: frames are NOT independent -- PoseNet evaluates consecutive
    PAIRS.  Frame t affects pair (t-1,t) AND pair (t,t+1).  This is a Markov
    chain requiring joint optimization.  A single Adam optimizer over ALL
    frame pixels jointly lets the gradient flow through the entire coupled
    system.

    When init_frames is provided (renderer+TTO mode), optimization starts from
    the given frames instead of noise. This is the "test-time optimization"
    path: the renderer produces PoseNet~0.031 frames, then TTO pushes PoseNet
    toward 0.005 by gradient descent against the scorers.

    Args:
        masks: (N, H, W) long tensor with target class indices.
        expected_pose: (P, 6) float tensor, P = N//2 (non-overlapping pairs,
            seq_len=2, matching upstream scorer).
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        num_steps: number of joint optimization steps.
        lr: Adam learning rate.
        seg_weight: weight for SegNet constraint.
        pose_weight: weight for PoseNet constraint.
        compress_weight: weight for spatial/temporal smoothness.
        noise_seed: deterministic seed for initialization (ignored if init_frames).
        device: computation device.
        log_every: print loss every N steps (0 to disable).
        init_frames: (N, H, W, 3) float tensor to warm-start from (renderer+TTO).
            If None, starts from class-mean-colored noise.
        early_stop_patience: stop if PoseNet hasn't improved in this many steps.
        use_embedding_loss: if True, use PoseNet embedding MSE (~512-d) instead of
            pose output MSE (6-d). Provides 42x more gradient directions.
            Requires expected_pose_embeddings to be provided.
        expected_pose_embeddings: (P, D) float tensor of GT PoseNet embeddings,
            precomputed from GT frames. Required when use_embedding_loss=True.
        seg_odd_only: if True, only compute SegNet loss on odd-indexed frames
            (frame 2k+1), matching official scorer behavior. Frees even frames
            for pure PoseNet optimization.
        antialias_weight: weight for 2x2-block variance penalty in the
            compressibility loss. PoseNet operates on 2x-downsampled YUV
            (rgb_to_yuv6 averages 2x2 blocks), so sub-block pixel noise is
            invisible to PoseNet but costs bits under any codec. Recommended
            0.1-0.5 for TTO mode. Default 0.0 (disabled, backward compatible).
        lr_schedule: learning rate schedule. "constant" = fixed LR (default).
            "cosine" = warmup from lr/10 over 50 steps, then cosine decay
            from lr to lr/100 over remaining steps. Fridrich optimization
            theory: coarse alignment early, fine refinement late.
        segnet_loss_mode: SegNet loss function. ``"xent"`` = cross-entropy
            (default, backward compatible). ``"hinge"`` = logit-margin hinge
            loss that focuses gradient on boundary pixels at risk of argmax
            flip, 2--5x more efficient for TTO.
        hinge_margin: margin for hinge loss (only used when
            ``segnet_loss_mode="hinge"``). Default 0.5.
        eval_roundtrip: if True, simulates the contest eval resolution
            roundtrip (scorer res -> camera res -> uint8 -> scorer res) using
            STE before computing PoseNet/SegNet losses. Makes training loss
            match the actual evaluation pipeline. Default False (backward
            compatible).
        phase2_segnet_only: if True, after ``early_stop_patience`` steps of
            Phase 1 (joint optimization), switch to Phase 2 where only odd
            frames (SegNet frames) are optimized with SegNet-only loss. Even
            frames are frozen. This allows fine-tuning SegNet agreement without
            disturbing PoseNet. Default False (single-phase, backward compatible).
        phase2_steps: number of Phase 2 steps (only used when
            ``phase2_segnet_only=True``). Default 200.
        use_null_space: if True, after each optimizer step, apply a YUV null
            space projection that improves SegNet without affecting PoseNet.
            PoseNet operates on 4:2:0 subsampled YUV; perturbations that are
            zero-mean within 2x2 chroma blocks are invisible to PoseNet.
            Default False (backward compatible).
        null_space_step: step size for the null space projection (only used
            when ``use_null_space=True``). Default 0.5.
        null_space_every: apply null space projection every N steps (only used
            when ``use_null_space=True``). Default 10.
        pair_weights: (P,) optional float tensor of per-pair importance weights.
            Scales the per-pair PoseNet loss before averaging. Use to focus TTO
            gradient budget on hard pairs (from pair_difficulty_map). Default None
            (uniform weighting, backward compatible).
        **cfg: additional config (ignored, for profile compatibility).

    Returns:
        (N, H, W, 3) float tensor in [0, 255].
    """
    device = torch.device(device)
    N = masks.shape[0]
    P = N // 2  # non-overlapping pairs, matching upstream scorer seq_len=2

    # Validate embedding loss config
    if use_embedding_loss and expected_pose_embeddings is None:
        raise ValueError(
            "use_embedding_loss=True requires expected_pose_embeddings. "
            "Precompute them with extract_gt_pose_embeddings() in renderer_tto.py."
        )

    # ── Validate PoseNet gradients flow (catch the @torch.no_grad bug) ────
    # The upstream rgb_to_yuv6 has @torch.no_grad, which silently kills all
    # PoseNet gradients. Call make_scorers_differentiable() before this function.
    # This check costs ~1ms and prevents hours of wasted GPU time.
    _test_in = torch.randn(1, 2, 3, 8, 8, device=device, requires_grad=True)
    _test_out = posenet(posenet.preprocess_input(_test_in))
    _grad = torch.autograd.grad(_test_out["pose"].sum(), _test_in, allow_unused=True)[0]
    if _grad is None or _grad.abs().max().item() < 1e-10:
        raise RuntimeError(
            "PoseNet gradients are ZERO through preprocess_input. "
            "Call tac.scorer.make_scorers_differentiable(posenet, segnet) "
            "before optimization. The upstream rgb_to_yuv6 has @torch.no_grad()."
        )
    del _test_in, _test_out, _grad

    # Move inputs to device once (avoids repeated transfers in inner loop)
    masks = masks.to(device)
    expected_pose = expected_pose.to(device)
    if expected_pose_embeddings is not None:
        expected_pose_embeddings = expected_pose_embeddings.to(device)
    if pair_weights is not None:
        pair_weights = pair_weights.to(device)

    # Resolve per-class weights for SegNet hinge loss
    _per_class_weights: torch.Tensor | None = None
    if segnet_class_weights is not None:
        if isinstance(segnet_class_weights, str) and segnet_class_weights == "auto":
            _per_class_weights = compute_segnet_class_weights(masks)
            logger.info("  [coupled-4dvar] Auto class weights: %s",
                        [f"{w:.2f}" for w in _per_class_weights.tolist()])
        elif isinstance(segnet_class_weights, torch.Tensor):
            _per_class_weights = segnet_class_weights.to(device)
        else:
            raise ValueError(
                f"segnet_class_weights must be 'auto', a Tensor, or None. "
                f"Got: {type(segnet_class_weights)}"
            )

    # Initialize ALL frames jointly — from renderer output (TTO) or noise
    warm_start = init_frames is not None
    if warm_start:
        frames = init_frames.to(device).float().detach().clone()
        logger.info("  [coupled-4dvar] Warm-starting from init_frames (TTO mode)")
        if use_embedding_loss:
            logger.info("  [coupled-4dvar] Using embedding loss (~%dd) instead of pose output (6d)",
                        expected_pose_embeddings.shape[-1])
        if seg_odd_only:
            logger.info("  [coupled-4dvar] SegNet loss on odd frames only (scorer-matched)")
    else:
        frames = generate_initial_frames(masks, noise_seed, device=device)
    frames.requires_grad_(True)

    # Single optimizer over ALL frame pixels
    optimizer = torch.optim.Adam([frames], lr=lr)

    # LR scheduler: cosine annealing with warmup for fine-grained control
    scheduler = None
    if lr_schedule == "cosine":
        warmup_steps = min(50, num_steps // 5)

        def _cosine_with_warmup(step: int) -> float:
            """LR multiplier: warmup from 0.1x to 1.0x, then cosine decay to 0.01x.

            Continuous at step=warmup_steps: warmup reaches exactly 1.0,
            cosine starts at exactly 1.0.
            """
            if step < warmup_steps:
                # Linear warmup: lr/10 -> lr (reaches 1.0 at step=warmup_steps-1
                # when warmup_steps > 1, or 0.1 at step=0 when warmup_steps=1)
                return 0.1 + 0.9 * (step / max(warmup_steps - 1, 1))
            # Cosine decay: lr -> lr/100
            progress = (step - warmup_steps) / max(num_steps - warmup_steps - 1, 1)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_with_warmup)
        logger.info("  [coupled-4dvar] LR schedule: cosine with %d-step warmup "
                     "(%.4f -> %.4f -> %.6f)", warmup_steps, lr * 0.1, lr, lr * 0.01)
    elif lr_schedule != "constant":
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}. Use 'constant' or 'cosine'.")

    # Track best PoseNet snapshot (Lagrangian transient strategy: the optimizer
    # hits a PoseNet minimum around step 300-500 before compressibility diverges it).
    best_pose_loss = float("inf")
    best_frames_snapshot: torch.Tensor | None = None
    steps_since_improvement = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # Apply eval-matched roundtrip if enabled (STE for gradients)
        if eval_roundtrip:
            from tac.renderer import simulate_eval_roundtrip
            frames_for_loss = simulate_eval_roundtrip(
                frames.permute(0, 3, 1, 2),  # (N, H, W, 3) -> (N, 3, H, W)
                target_h=CAMERA_H,
                target_w=CAMERA_W,
                noise_std=roundtrip_noise_std,
            ).permute(0, 2, 3, 1)  # (N, 3, H, W) -> (N, H, W, 3)
        else:
            frames_for_loss = frames

        # --- SegNet constraint: all frames (or odd-only if seg_odd_only) ---
        seg_loss = compute_segnet_constraint_loss(
            frames_for_loss, masks, segnet, seg_odd_only=seg_odd_only,
            loss_mode=segnet_loss_mode, hinge_margin=hinge_margin,
            per_class_weights=_per_class_weights,
        )

        # --- PoseNet constraint: all consecutive PAIRS jointly ---
        pose_loss = torch.tensor(0.0, device=device)
        if P > 0 and expected_pose.shape[0] > 0:
            if use_embedding_loss:
                # Embedding MSE: ~512 gradient directions instead of 6.
                # 42x richer gradient landscape for frame refinement.
                pose_loss = compute_posenet_embedding_constraint_loss(
                    frames_for_loss, expected_pose_embeddings, posenet,
                )
            else:
                pose_loss = compute_posenet_constraint_loss(
                    frames_for_loss, expected_pose, posenet,
                    pair_weights=pair_weights,
                )

        # --- Compressibility: anneal weight to prevent PoseNet divergence ---
        compress_loss = compute_compressibility_loss(frames, antialias_weight=antialias_weight)
        progress = step / max(num_steps - 1, 1)

        if warm_start:
            # Fix #3: Skip annealing for warm-start (TTO mode).
            # The annealing schedule was designed for from-noise optimization where
            # compressibility shapes the image before scorers refine it. When warm-
            # starting from renderer output, the image is already well-formed — the
            # annealing keeps compress_weight at full for the first 40% of steps,
            # exactly when PoseNet needs maximum gradient. Start at 10% immediately
            # so PoseNet gets full gradient authority from step 0.
            effective_compress_weight = compress_weight * 0.1
        else:
            # From-noise: full weight for first 40%, then decay to 10%.
            if progress < 0.4:
                effective_compress_weight = compress_weight
            else:
                decay = 1.0 - 0.9 * ((progress - 0.4) / 0.6)  # 1.0 -> 0.1
                effective_compress_weight = compress_weight * decay

        # --- Coupled total loss ---
        total_loss = (
            seg_weight * seg_loss
            + pose_weight * pose_loss
            + effective_compress_weight * compress_loss
        )

        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Project back to valid pixel range
        with torch.no_grad():
            frames.data.clamp_(0.0, 255.0)

        # Adaptive per-pixel YUV null space projection: improve SegNet without
        # touching PoseNet. Uses the empirical null space basis (6 directions in
        # the 12D 2x2 block space that are invisible to PoseNet's YUV420 preprocessing).
        if use_null_space and (step + 1) % null_space_every == 0:
            # Compute SegNet gradient w.r.t. current frames (needs grad enabled)
            frames_ns = frames.detach().clone().requires_grad_(True)
            seg_loss_ns = compute_segnet_constraint_loss(
                frames_ns, masks, segnet, seg_odd_only=seg_odd_only,
                loss_mode=segnet_loss_mode, hinge_margin=hinge_margin,
                per_class_weights=_per_class_weights,
            )
            seg_grad = torch.autograd.grad(seg_loss_ns, frames_ns)[0]
            # Project into PoseNet null space and apply (no grad needed)
            with torch.no_grad():
                from tac.scorer_exploits import (
                    load_null_space_basis,
                    project_segnet_grad_to_posenet_null_space,
                )
                # Load basis once (cached after first call via module-level)
                _cache_key = f"_null_basis_{device}"
                if not hasattr(coupled_trajectory_optimize, _cache_key):
                    setattr(coupled_trajectory_optimize, _cache_key, load_null_space_basis(
                        device=device,
                    ))
                null_basis = getattr(coupled_trajectory_optimize, _cache_key)

                # seg_grad is (N, H, W, 3), need (N, 3, H, W) for the projection
                seg_grad_chw = seg_grad.permute(0, 3, 1, 2)
                projected_grad = project_segnet_grad_to_posenet_null_space(
                    seg_grad_chw, null_basis, max_magnitude=null_space_step * 10.0,
                )
                # Apply: descend in null space direction (negative = minimize loss)
                projected_hwc = projected_grad.permute(0, 2, 3, 1)
                frames.data.sub_(null_space_step * projected_hwc)
                frames.data.clamp_(0.0, 255.0)
            del frames_ns, seg_grad, seg_loss_ns

        # Snapshot best PoseNet state (transient capture)
        pose_val = pose_loss.item()
        if pose_val < best_pose_loss:
            best_pose_loss = pose_val
            best_frames_snapshot = frames.detach().clone()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        # Early stop: if PoseNet hasn't improved in EARLY_STOP_PATIENCE steps,
        # the optimizer is past the transient and drifting. Stop and return snapshot.
        if early_stop_patience > 0 and steps_since_improvement >= early_stop_patience and best_frames_snapshot is not None:
            if log_every > 0:
                logger.info("  [coupled-4dvar] Early stop at step %d: "
                            "no PoseNet improvement in %d steps", step + 1, early_stop_patience)
            break

        if log_every > 0 and (step + 1) % log_every == 0:
            logger.info(
                "  [coupled-4dvar] step %4d/%d: total=%.4f seg=%.4f pose=%.4f "
                "compress=%.4f cw=%.3f best_pose=%.4f",
                step + 1, num_steps, total_loss.item(), seg_loss.item(),
                pose_loss.item(), compress_loss.item(),
                effective_compress_weight, best_pose_loss,
            )

    # ── Phase 2: SegNet-only refinement on odd frames ──────────────────────
    # After Phase 1 converges (early stop or full steps), optionally switch
    # to SegNet-only mode: freeze even frames (PoseNet-only frames), optimize
    # only odd frames (SegNet evaluation frames) with SegNet loss only.
    # This polishes SegNet agreement without disturbing the PoseNet optimum.
    if phase2_segnet_only and best_frames_snapshot is not None and phase2_steps > 0:
        logger.info("  [coupled-4dvar] Phase 2: SegNet-only on odd frames "
                    "(%d steps, loss_mode=%s)", phase2_steps, segnet_loss_mode)

        # Start Phase 2 from the best PoseNet snapshot
        p2_frames = best_frames_snapshot.clone().detach()

        # Create separate optimizable tensor for odd frames only
        N_p2 = p2_frames.shape[0]
        odd_indices = list(range(1, N_p2, 2))
        odd_frames = p2_frames[odd_indices].clone().detach().requires_grad_(True)
        p2_optimizer = torch.optim.Adam([odd_frames], lr=lr * 0.5)  # lower LR for fine-tuning

        odd_masks = masks[odd_indices]

        for p2_step in range(phase2_steps):
            p2_optimizer.zero_grad()

            # Compute SegNet loss on odd frames only (these are the scorer frames)
            seg_loss_p2 = compute_segnet_constraint_loss(
                odd_frames, odd_masks, segnet,
                seg_odd_only=False,  # already selected odd frames
                loss_mode=segnet_loss_mode,
                hinge_margin=hinge_margin,
                per_class_weights=_per_class_weights,
            )

            seg_loss_p2.backward()
            p2_optimizer.step()

            with torch.no_grad():
                odd_frames.data.clamp_(0.0, 255.0)

            if log_every > 0 and (p2_step + 1) % log_every == 0:
                logger.info(
                    "  [coupled-4dvar] phase2 step %4d/%d: seg=%.4f",
                    p2_step + 1, phase2_steps, seg_loss_p2.item(),
                )

        # Write refined odd frames back into the snapshot
        with torch.no_grad():
            for i, idx in enumerate(odd_indices):
                p2_frames[idx] = odd_frames[i].detach()

        best_frames_snapshot = p2_frames
        logger.info("  [coupled-4dvar] Phase 2 complete. Final seg_loss=%.4f",
                    seg_loss_p2.item())

        del odd_frames, p2_optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Return the snapshot with best PoseNet (not the final state which may have drifted)
    if best_frames_snapshot is not None:
        logger.info("  [coupled-4dvar] Returning best PoseNet snapshot (loss=%.4f)", best_pose_loss)
        result = best_frames_snapshot.round().clamp(0.0, 255.0)
    else:
        with torch.no_grad():
            result = frames.detach().round().clamp(0.0, 255.0)
    return result


def gpu_lane_full_pipeline(
    masks: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: str = "cuda",
    **cfg,
) -> dict:
    """Complete GPU lane pipeline with Fridrich integration.

    Orchestrates all stages of the GPU lane in a single call:

    1. Generate initial frames from masks (class-mean colors + noise)
    2. Estimate expected pose from mask dynamics (if not provided)
    3. Run coupled trajectory optimization (4D-Var joint optimization)
    4. Estimate detection boundary (Fridrich: find scorer's blind spot)
    5. Compute pixel cost map (Fridrich hybrid: scorer Jacobian + S-UNIWARD)
    6. Run Fridrich constrained refinement (minimize rate s.t. boundary)
    7. Apply STC quantization (Fridrich: cost-weighted optimal rounding)
    8. Build archive

    Args:
        masks: (N, H, W) long tensor with target class indices.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        **cfg: configuration overrides:
            - expected_pose (Tensor): (P, 6) pose targets. If None, estimated from masks.
            - num_steps (int): coupled trajectory steps (default 1000).
            - fridrich_steps (int): Fridrich refinement steps (default 500).
            - lr (float): learning rate (default 0.01).
            - seg_weight (float): SegNet constraint weight (default 100.0).
            - pose_weight (float): PoseNet constraint weight (default 10.0).
            - compress_weight (float): compressibility weight (default 1.0).
            - noise_seed (int): deterministic seed (default 42).
            - batch_size (int): frames per scorer forward pass (default 8).
            - log_every (int): print progress every N steps (default 100).
            - skip_fridrich (bool): skip Fridrich refinement (default False).
            - skip_stc (bool): skip STC quantization (default False).
            - cost_method (str): pixel cost method (default "hybrid").
            - num_probes (int): boundary probing count (default 20).
            - rate_reduction (float): STC target rate reduction (default 0.1).

    Returns:
        Dict with:
            'optimized_frames': (N, H, W, 3) float tensor in [0, 255]
            'masks': (N, H, W) long tensor
            'expected_pose': (P, 6) float tensor
            'boundary': dict from estimate_detection_boundary (or None)
            'cost_map': (N, H, W) pixel cost map (or None)
            'diagnostics': dict with per-stage timing and metrics
    """
    from tac.fridrich import (
        compute_pixel_cost_map,
        estimate_detection_boundary,
        fridrich_constrained_optimize,
        optimal_quantization_stc,
    )

    num_steps = cfg.get("num_steps", 1000)
    fridrich_steps = cfg.get("fridrich_steps", 500)
    lr = cfg.get("lr", 0.01)
    seg_weight = cfg.get("seg_weight", 100.0)
    pose_weight = cfg.get("pose_weight", 10.0)
    compress_weight = cfg.get("compress_weight", 1.0)
    noise_seed = cfg.get("noise_seed", 42)
    batch_size = cfg.get("batch_size", 8)
    log_every = cfg.get("log_every", 100)
    skip_fridrich = cfg.get("skip_fridrich", False)
    skip_stc = cfg.get("skip_stc", False)
    cost_method = cfg.get("cost_method", "hybrid")
    num_probes = cfg.get("num_probes", 20)
    rate_reduction = cfg.get("rate_reduction", 0.1)

    import time
    diagnostics: dict[str, Any] = {}

    # --- Stage 1: Pose targets ---
    expected_pose = cfg.get("expected_pose")
    if expected_pose is None:
        t0 = time.time()
        logger.info("[gpu_lane_pipeline] Stage 1: Estimating expected pose from masks...")
        expected_pose = estimate_expected_pose(masks, device=device)
        diagnostics["pose_estimation_seconds"] = time.time() - t0
        logger.info("  Pose targets shape: %s", expected_pose.shape)
    else:
        diagnostics["pose_estimation_seconds"] = 0.0
        logger.info("[gpu_lane_pipeline] Stage 1: Using provided pose targets (%s)", expected_pose.shape)

    # --- Stage 2: Coupled trajectory optimization (4D-Var) ---
    t0 = time.time()
    logger.info("[gpu_lane_pipeline] Stage 2: Coupled trajectory (%d steps)...", num_steps)
    frames = coupled_trajectory_optimize(
        masks=masks,
        expected_pose=expected_pose,
        posenet=posenet,
        segnet=segnet,
        num_steps=num_steps,
        lr=lr,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
        compress_weight=compress_weight,
        noise_seed=noise_seed,
        device=device,
        log_every=log_every,
    )
    diagnostics["coupled_trajectory_seconds"] = time.time() - t0
    logger.info("  Coupled trajectory done in %.1fs", diagnostics['coupled_trajectory_seconds'])

    boundary = None
    cost_map = None

    if not skip_fridrich:
        # --- Stage 3: Detection boundary estimation ---
        t0 = time.time()
        logger.info("[gpu_lane_pipeline] Stage 3: Estimating detection boundary...")
        frames_bchw = frames.permute(0, 3, 1, 2).contiguous()
        boundary = estimate_detection_boundary(
            frames_bchw, posenet, segnet,
            num_probes=num_probes,
            max_magnitude=30.0,
            device=device,
            subsample_frames=4,
        )
        seg_boundary = boundary["seg_boundary"]
        pose_boundary = boundary["pose_boundary"]
        diagnostics["boundary_estimation_seconds"] = time.time() - t0
        logger.info("  Boundary: seg=%.6f, pose=%.6f in %.1fs",
                    seg_boundary, pose_boundary, diagnostics['boundary_estimation_seconds'])

        # --- Stage 4: Pixel cost map ---
        t0 = time.time()
        logger.info("[gpu_lane_pipeline] Stage 4: Computing pixel cost map...")
        cost_map = compute_pixel_cost_map(
            frames_bchw, posenet, segnet,
            method=cost_method,
            device=device,
            subsample_frames=1,
        )
        diagnostics["cost_map_seconds"] = time.time() - t0
        logger.info("  Cost map: [%.4f, %.4f] in %.1fs",
                    cost_map.min(), cost_map.max(), diagnostics['cost_map_seconds'])

        # --- Stage 5: Fridrich constrained refinement ---
        t0 = time.time()
        logger.info("[gpu_lane_pipeline] Stage 5: Fridrich constrained refinement (%d steps)...", fridrich_steps)
        refined_bchw = fridrich_constrained_optimize(
            frames,
            posenet, segnet,
            pixel_costs=cost_map,
            seg_boundary=seg_boundary,
            pose_boundary=pose_boundary,
            num_steps=fridrich_steps,
            lr=lr,
            device=device,
            batch_size=batch_size,
        )
        diagnostics["fridrich_seconds"] = time.time() - t0
        logger.info("  Fridrich refinement done in %.1fs", diagnostics['fridrich_seconds'])

        # --- Stage 6: STC quantization ---
        if not skip_stc:
            t0 = time.time()
            logger.info("[gpu_lane_pipeline] Stage 6: STC quantization...")
            quantized_bchw = optimal_quantization_stc(
                refined_bchw.float(),
                cost_map,
                target_rate_reduction=rate_reduction,
            )
            frames = quantized_bchw.permute(0, 2, 3, 1).contiguous().float()
            diagnostics["stc_seconds"] = time.time() - t0
            logger.info("  STC quantization done in %.1fs", diagnostics['stc_seconds'])
        else:
            frames = refined_bchw.permute(0, 2, 3, 1).contiguous().float()
            diagnostics["stc_seconds"] = 0.0
    else:
        diagnostics["boundary_estimation_seconds"] = 0.0
        diagnostics["cost_map_seconds"] = 0.0
        diagnostics["fridrich_seconds"] = 0.0
        diagnostics["stc_seconds"] = 0.0

    return {
        "optimized_frames": frames,
        "masks": masks,
        "expected_pose": expected_pose,
        "boundary": boundary,
        "cost_map": cost_map,
        "diagnostics": diagnostics,
    }


def alternating_projections_optimize(
    masks: torch.Tensor,  # (N, H, W)
    expected_pose: torch.Tensor,  # (P, 6) where P = N//2 (non-overlapping pairs)
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    num_outer_iterations: int = 100,
    num_inner_steps: int = 10,
    # [empirical: alternating-projections regime LR; arbitrariness A-3;
    # half of constrained_generate canonical 0.1 because alternating-projection
    # uses smaller per-inner-step updates]
    lr: float = 0.05,
    noise_seed: int = 42,
    # [empirical: 100/10 split per A-1 DEFERRED context; same caveat]
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    tv_weight: float = 1.0,
    device: str = "cuda",
    log_every: int = 10,
    **cfg,
) -> torch.Tensor:
    """Dykstra's alternating projections for multi-constraint frame generation.

    Each outer iteration cycles through 3 projection operators:
    1. P_seg: gradient descent steps to satisfy argmax(SegNet(f)) == masks
    2. P_pose: gradient descent steps to satisfy PoseNet(f_t, f_{t+1}) == target
    3. P_rate: total variation denoising to minimize rate

    Dykstra's modification accumulates increments to handle non-convex sets.
    Convergence guaranteed for convex constraint sets; empirically effective
    for our quasi-convex scorer constraints.

    Args:
        masks: (N, H, W) long tensor with target class indices.
        expected_pose: (P, 6) float tensor, P = N//2 (non-overlapping pairs,
            seq_len=2, matching upstream scorer).
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        num_outer_iterations: number of full projection cycles.
        num_inner_steps: gradient descent steps per projection.
        lr: Adam learning rate for inner loops.
        noise_seed: deterministic seed for initialization.
        seg_weight: SegNet projection strength.
        pose_weight: PoseNet projection strength.
        tv_weight: TV denoising strength.
        device: computation device.
        log_every: print loss every N outer iterations (0 to disable).
        **cfg: additional config (ignored).

    Returns:
        (N, H, W, 3) float tensor in [0, 255].
    """
    device = torch.device(device)
    N = masks.shape[0]
    P = N // 2  # non-overlapping pairs, matching upstream scorer seq_len=2

    # Initialize
    frames = generate_initial_frames(masks, noise_seed, device=device)

    # Dykstra's increments (accumulated corrections for non-convex sets)
    inc_seg = torch.zeros_like(frames)
    inc_pose = torch.zeros_like(frames)
    inc_rate = torch.zeros_like(frames)

    # Create persistent parameter buffers and optimizers ONCE.
    # Reusing optimizers preserves Adam momentum/variance across outer iters.
    y_seg = torch.zeros_like(frames, requires_grad=True)
    opt_seg = torch.optim.Adam([y_seg], lr=lr)
    y_pose = torch.zeros_like(frames, requires_grad=True)
    opt_pose = torch.optim.Adam([y_pose], lr=lr)
    y_rate = torch.zeros_like(frames, requires_grad=True)
    opt_rate = torch.optim.Adam([y_rate], lr=lr)

    for outer in range(num_outer_iterations):
        # ---- Projection 1: SegNet constraint ----
        with torch.no_grad():
            y_seg.data.copy_(frames + inc_seg)
        for _ in range(num_inner_steps):
            opt_seg.zero_grad()
            loss = seg_weight * compute_segnet_constraint_loss(y_seg, masks, segnet)
            loss.backward()
            opt_seg.step()
            with torch.no_grad():
                y_seg.data.clamp_(0.0, 255.0)
        with torch.no_grad():
            p_seg = y_seg.data.clone()
            inc_seg = frames + inc_seg - p_seg
            frames = p_seg

        # ---- Projection 2: PoseNet constraint ----
        if P > 0 and expected_pose.shape[0] > 0:
            with torch.no_grad():
                y_pose.data.copy_(frames + inc_pose)
            for _ in range(num_inner_steps):
                opt_pose.zero_grad()
                loss = pose_weight * compute_posenet_constraint_loss(
                    y_pose, expected_pose, posenet,
                )
                loss.backward()
                opt_pose.step()
                with torch.no_grad():
                    y_pose.data.clamp_(0.0, 255.0)
            with torch.no_grad():
                p_pose = y_pose.data.clone()
                inc_pose = frames + inc_pose - p_pose
                frames = p_pose

        # ---- Projection 3: Rate minimization (TV denoising) ----
        with torch.no_grad():
            y_rate.data.copy_(frames + inc_rate)
        for _ in range(num_inner_steps):
            opt_rate.zero_grad()
            loss = tv_weight * compute_compressibility_loss(y_rate)
            loss.backward()
            opt_rate.step()
            with torch.no_grad():
                y_rate.data.clamp_(0.0, 255.0)
        with torch.no_grad():
            p_rate = y_rate.data.clone()
            inc_rate = frames + inc_rate - p_rate
            frames = p_rate

        if log_every > 0 and (outer + 1) % log_every == 0:
            with torch.no_grad():
                s = compute_segnet_constraint_loss(frames, masks, segnet).item()
                p = (
                    compute_posenet_constraint_loss(frames, expected_pose, posenet).item()
                    if P > 0
                    else 0.0
                )
                c = compute_compressibility_loss(frames).item()
            logger.info(
                "  [dykstra] outer %3d/%d: seg=%.4f pose=%.4f compress=%.4f",
                outer + 1, num_outer_iterations, s, p, c,
            )

    with torch.no_grad():
        result = frames.detach().round().clamp(0.0, 255.0)
    return result


def newton_step_optimize(
    frames: torch.Tensor,  # (N, H, W, 3) or None
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks: torch.Tensor | None = None,
    expected_pose: torch.Tensor | None = None,
    num_newton_steps: int = 3,
    max_iter_per_step: int = 20,
    lr: float = 1.0,
    history_size: int = 10,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
    noise_seed: int = 42,
    device: str = "cuda",
    log_every: int = 1,
    **cfg,
) -> torch.Tensor:
    """Newton's method: x_{k+1} = x_k - H^{-1} * g.

    For 589K-dimensional pixel space, the full Hessian is intractable.
    Use L-BFGS approximation (rank-m quasi-Newton) which maintains an
    implicit approximation of H^{-1} from the last m gradient pairs.

    torch.optim.LBFGS handles this natively.  We wrap it with:
    - Line search (strong Wolfe conditions)
    - Convergence monitoring (gradient norm, function value change)
    - Max num_newton_steps Newton steps (each requires ~20 scorer evaluations)

    Expected: convergence in 1-3 steps where gradient descent needs 1000.

    Args:
        frames: (N, H, W, 3) initial frames, or None to initialize from masks.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        masks: (N, H, W) long tensor (required if frames is None).
        expected_pose: (P, 6) float tensor (estimated from masks if None).
        num_newton_steps: max L-BFGS steps (each = multiple evaluations).
        max_iter_per_step: max function evaluations per L-BFGS step.
        lr: step size multiplier.
        history_size: L-BFGS memory (rank of Hessian approximation).
        seg_weight: SegNet weight.
        pose_weight: PoseNet weight.
        compress_weight: compressibility weight.
        noise_seed: seed for initialization if frames is None.
        device: computation device.
        log_every: print loss every N steps.
        **cfg: additional config (ignored).

    Returns:
        (N, H, W, 3) float tensor in [0, 255].
    """
    device = torch.device(device)

    if frames is None:
        assert masks is not None, "Must provide either frames or masks"
        frames = generate_initial_frames(masks, noise_seed, device=device)
    else:
        frames = frames.to(device).detach().clone()

    if masks is None:
        # Extract masks from current frames via SegNet argmax
        with torch.inference_mode():
            frames_chw = frames.permute(0, 3, 1, 2).contiguous()
            seg_btchw = frames_chw.unsqueeze(1).contiguous()
            seg_in = segnet.preprocess_input(seg_btchw)
            logits = segnet(seg_in)
            masks = logits.argmax(dim=1).to(device)  # (N, H_out, W_out)

    if expected_pose is None:
        expected_pose = estimate_expected_pose(masks, device=device)

    N = frames.shape[0]
    P = N // 2  # non-overlapping pairs, matching upstream scorer seq_len=2

    frames.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [frames],
        lr=lr,
        max_iter=max_iter_per_step,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    step_losses: list[float] = []

    for newton_step in range(num_newton_steps):
        def closure():
            optimizer.zero_grad()

            seg_loss = compute_segnet_constraint_loss(frames, masks, segnet)

            pose_loss = torch.tensor(0.0, device=device)
            if P > 0 and expected_pose.shape[0] > 0:
                pose_loss = compute_posenet_constraint_loss(
                    frames, expected_pose, posenet,
                )

            compress_loss = compute_compressibility_loss(frames)

            total = (
                seg_weight * seg_loss
                + pose_weight * pose_loss
                + compress_weight * compress_loss
            )
            total.backward()

            # Track for logging
            step_losses.append(total.item())
            return total

        optimizer.step(closure)

        # Project back to valid range
        with torch.no_grad():
            frames.data.clamp_(0.0, 255.0)

        if log_every > 0 and (newton_step + 1) % log_every == 0 and step_losses:
            logger.info(
                "  [newton/lbfgs] step %d/%d: loss=%.6f (evals=%d)",
                newton_step + 1, num_newton_steps, step_losses[-1], len(step_losses),
            )

        # Convergence check: if loss barely changed
        if len(step_losses) >= 2 and abs(step_losses[-1] - step_losses[-2]) < 1e-8:
            if log_every > 0:
                logger.info("  [newton/lbfgs] converged at step %d", newton_step + 1)
            break

    with torch.no_grad():
        result = frames.detach().round().clamp(0.0, 255.0)
    return result


def _gradient_directed_dither(
    frames: torch.Tensor,
    masks: torch.Tensor,
    segnet: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Gradient-directed Floyd-Steinberg dithering for uint8 quantization.

    Standard rounding introduces quantization error uniformly. This method
    uses the scorer gradient to direct the rounding error: round in the
    direction that HELPS the scorer, or at least hurts it least.

    For each pixel:
      1. Compute floor and ceil values
      2. Check which direction the scorer gradient prefers
      3. Round in the preferred direction
      4. Diffuse the remaining error to neighbors (Floyd-Steinberg pattern)

    This is Yousfi's steganalysis insight applied to quantization: place the
    quantization noise where the scorer cannot detect it.

    Args:
        frames: (N, C, H, W) float tensor in [0, 255].
        masks: (N, H, W) long tensor with target masks.
        segnet: frozen SegNet model.
        device: computation device.

    Returns:
        (N, C, H, W) uint8-compatible float tensor.
    """
    N, C, H, W = frames.shape

    # Compute scorer gradient direction for each pixel.
    # Must go through segnet.preprocess_input (expects (B, T, C, H, W)) —
    # bypassing it skips required preprocessing and produces wrong gradients.
    inp = frames.detach().clone().requires_grad_(True)
    resized = F.interpolate(inp, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
    seg_btchw = resized.unsqueeze(1).contiguous()  # (N, 1, C, H, W) — T=1
    seg_input = segnet.preprocess_input(seg_btchw)
    logits = segnet(seg_input)
    H_out, W_out = logits.shape[2], logits.shape[3]
    masks_resized = F.interpolate(
        masks.float().unsqueeze(1).to(device),
        size=(H_out, W_out), mode="nearest",
    ).squeeze(1).long()
    loss = F.cross_entropy(logits, masks_resized)
    loss.backward()

    grad = inp.grad.detach() if inp.grad is not None else torch.zeros_like(frames)

    # Gradient-directed rounding
    with torch.no_grad():
        floor_val = frames.detach().floor()
        ceil_val = frames.detach().ceil()
        frac = frames.detach() - floor_val

        # If gradient points toward lower values, prefer floor (saves rate)
        # If gradient points toward higher values, prefer ceil
        # Tie-breaking at 0.5: use gradient direction
        prefer_ceil = (grad < 0).float()  # negative grad = loss decreases with more
        threshold = 0.5 - 0.1 * (2.0 * prefer_ceil - 1.0)  # bias threshold by gradient
        rounded = torch.where(frac > threshold, ceil_val, floor_val)

    return rounded.clamp(0.0, 255.0)
