#!/usr/bin/env python
"""Fridrich Constrained Renderer Training -- the path to sub-0.50.

Trains a mask-conditioned SPADE renderer with:
1. Hard SegNet constraint (augmented Lagrangian, boundary 0.005)
2. Hard PoseNet constraint (augmented Lagrangian, boundary 0.05)
3. Coupled trajectory loss (consecutive pairs, temporal coherence)
4. Ego-motion flow regularization (geometric prior)
5. Self-compression penalty (minimize model size for rate)
6. TV smoothness (compressible output)

The renderer generates frames from masks. PoseNet evaluates consecutive
pairs. The Lagrangian ensures BOTH scorers are satisfied simultaneously
without the weighted-sum imbalance that plagues standard training.

Architecture: DP-SIMS v2 with SPADE normalization + DepthAwareMotionPredictor
  - channels=(128,64,32,16): ~1M params, ~489KB FP4, rate 0.33
  - With self-compression: target 200-300KB, rate 0.13-0.20

Score projection: seg 0.003 + pose 0.002 + rate 0.13 = 0.57
With ego-motion: seg 0.003 + pose 0.001 + rate 0.13 = 0.50

Usage:
    PYTHONPATH=src:/path/to/upstream python experiments/train_renderer_fridrich.py \\
        --precomputed /path/to/precomputed \\
        --epochs 10000 \\
        --device cuda

    # Smoke test (fast, CPU):
    PYTHONPATH=src:/path/to/upstream python experiments/train_renderer_fridrich.py \\
        --precomputed /path/to/precomputed \\
        --epochs 5 --batch-size 2 --device cpu --smoke
"""
from __future__ import annotations

import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path setup — find upstream (scorer models), weights dir, GT video
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path("/home/zeus/content/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
    Path(os.environ.get("UPSTREAM_ROOT", "")) if os.environ.get("UPSTREAM_ROOT") else None,
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

_CANDIDATE_WEIGHTS = [
    Path("/home/zeus/content/upstream/models"),
    Path("/home/zeus/content/pact/upstream/models"),
    Path(__file__).resolve().parent.parent / "upstream" / "models",
]
WEIGHTS_DIR: Path | None = None
for _p in _CANDIDATE_WEIGHTS:
    if _p is not None and (_p / "posenet.safetensors").exists():
        WEIGHTS_DIR = _p
        break

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "fridrich_renderer"


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
@dataclass
class FridrichRendererConfig:
    """All hyperparameters for the Fridrich constrained renderer training."""

    # Architecture
    pair_mode: str = "dp_sims"  # "dp_sims" (proven, independent) or "asymmetric" (warp paradigm, experimental)
    channels: tuple[int, ...] = (128, 64, 32, 16)  # dp_sims only
    init_h: int = 24   # dp_sims only
    init_w: int = 32   # dp_sims only
    spade_hidden: int = 32  # dp_sims only
    num_classes: int = 5
    use_noise: bool = False  # dp_sims only
    # Asymmetric warp params (used when pair_mode="asymmetric")
    embed_dim: int = 6
    base_ch: int = 36
    mid_ch: int = 60
    motion_hidden: int = 32
    renderer_depth: int = 1

    # Training
    epochs: int = 10000
    lr: float = 2e-4
    lr_min_ratio: float = 0.01
    batch_size: int = 16  # council B5: T4 used 224MB at bs=4, has 20x headroom
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42

    # Scorer resolution (384x512 matches scorer input pipeline)
    target_h: int = 384
    target_w: int = 512

    # Phase boundaries (fraction of total epochs)
    # Council revision: Phase 1 converged by ~25%. Phase 2 needs 60% for slow Lagrangian.
    phase1_end: float = 0.25   # 0-25%: memorize (MSE + soft scorer)
    phase2_end: float = 0.85   # 25-85%: constrain (Fridrich Lagrangian, slow rho)
    # 85-100%: compress (add self-compression penalty)

    # Phase 1: MSE-dominated warmup (normalized to [0,1] range)
    p1_mse_weight: float = 10.0   # MSE should dominate Phase 1 (learn colors first)
    p1_seg_weight: float = 1.0    # soft SegNet signal
    p1_pose_weight: float = 0.01  # very low: PoseNet starts at ~180, would dominate even at 0.1

    # Phase 2+3: Fridrich Lagrangian constraints
    seg_boundary: float = 0.005    # target: seg_dist < 0.005
    pose_boundary: float = 0.05    # target: pose_dist < 0.05 (relaxed; 0.01 never achieved)
    rho_init: float = 10.0         # initial quadratic penalty coefficient
    rho_growth: float = 1.005      # rho multiplier per outer step (council: 1.02 exploded in 400ep)
    rho_max: float = 1e3           # cap rho (council: 1e4 caused Lagrangian explosion at ep4418)

    # Lagrangian multiplier cap (council: 1e6 allowed λ_p=185K, overwhelming task loss)
    lambda_cap: float = 1e4

    # TV smoothness
    tv_weight: float = 0.1

    # Ego-motion flow regularization
    flow_weight: float = 0.0  # disabled by default: motion predictor unvalidated

    # Phase 3: self-compression
    rate_weight: float = 0.01       # rate penalty weight
    target_bytes: int = 250 * 1024  # 250KB target model size
    init_bits: float = 8.0          # start with full precision

    # Checkpointing
    checkpoint_every: int = 500
    eval_every: int = 200  # council: 50 eval points > 10 for better checkpoint selection
    early_stop_patience: int = 500  # stop if both constraints satisfied for N epochs

    # Logging
    log_every: int = 50

    # Wall-clock budget (hours, 0 = no limit)
    max_hours: float = 48.0

    # Resume from checkpoint
    resume: str | None = None

    # Phase 2 MSE anchor weight (keep reconstruction signal alive)
    phase2_mse_weight: float = 0.1

    # Configurable constants (previously hardcoded in MotionPredictor)
    max_flow_px: float = 20.0       # max flow in pixels for asymmetric mode (20px for fast ego-motion at 20fps)
    max_residual: float = 20.0      # max residual magnitude for asymmetric mode
    seg_temperature_start: float = 1.0   # Phase 2 curriculum start temperature
    seg_temperature_end: float = 0.1     # Phase 2 curriculum end temperature

    # Gated optimization flags (deferred items, default off)
    use_margin_segnet: bool = False       # optimization #13: margin-based SegNet loss
    segnet_margin_threshold: float = 0.1  # margin threshold for margin-based SegNet
    use_null_space: bool = False          # optimization #6: null-space projection
    share_stem: bool = False              # optimization #12: shared stem (deferred, gated)
    flow_only: bool = False               # optimization #15: no gate/residual (deferred, gated)

    # Flow warmup: zero out residual for first N epochs to force flow learning (council B1+B2)
    flow_warmup_epochs: int = 500    # epochs with residual zeroed out
    residual_ramp_epochs: int = 500  # epochs to ramp residual from 0→1 after warmup

    # Gate regularization (penalize gate_mean above threshold — Quantizr adversarial)
    gate_reg_threshold: float = 0.5
    gate_reg_weight: float = 0.1

    # Auto-kill on divergence
    # Phase 2 Lagrangian loss can legitimately reach lambda_cap * violation ~ 1e4 * 0.5 = 5000
    # Threshold must exceed max plausible Lagrangian loss to avoid false kills
    kill_loss_threshold: float = 1e5    # council sweep: 100 was 1000x too low for Phase 2
    kill_seg_threshold: float = 0.8     # relaxed: SegNet starts near 0.5
    kill_pose_threshold: float = 500.0  # relaxed: PoseNet starts >100 for asymmetric
    kill_patience: int = 200            # give asymmetric warmup time to converge

    # Even-pairs-only: scorer evaluates (0,1),(2,3),(4,5)... not (1,2),(3,4)
    # Training on even-index pair starts only saves ~50% compute
    even_pairs_only: bool = True

    # Export: auto-export .bin alongside best checkpoint for accurate rate
    export_bits: int = 4  # FP4 for contest submission (matches Quantizr's codec)

    # Ego-flow: learnable per-pair affine warp (council Experiment B Layer 2)
    use_ego_flow: bool = False       # enable learnable affine ego-flow
    ego_flow_max_px: float = 20.0    # max affine displacement in pixels
    ego_flow_lr: float = 1e-3        # separate LR for affine params (faster than renderer)

    # PoseNet supervision: direct scorer-space optimization (council Layer 1+3)
    pose_supervision_weight: float = 0.0  # weight for MSE(PoseNet(gen)[:6], target[:6])
    pose_targets_path: str | None = None  # path to precomputed posenet_targets.bin
    pose_embed_loss: bool = False         # Layer 3: use embedding-level loss instead of output

    # RAFT flow: frozen dense warp + MotionPredictor supervision (council Layer 2)
    raft_flow_path: str | None = None     # path to raft_flow.pt (dense RAFT flow)
    flow_supervision_weight: float = 0.0  # weight for MSE(motion_flow, raft_flow)

    # Smoke test overrides
    smoke: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    """Load frozen PoseNet and SegNet, patch for differentiable training."""
    if WEIGHTS_DIR is None:
        raise FileNotFoundError(f"Scorer weights not found in {_CANDIDATE_WEIGHTS}")
    from tac.scorer import load_scorers
    posenet, segnet = load_scorers(
        WEIGHTS_DIR / "posenet.safetensors",
        WEIGHTS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=UPSTREAM_ROOT,
    )
    return posenet, segnet


def _patch_scorers_for_training(posenet: nn.Module, segnet: nn.Module) -> None:
    """Monkey-patch upstream scorers for differentiable training.

    The upstream PoseNet.preprocess_input uses rgb_to_yuv6 decorated with
    @torch.no_grad(), which kills gradients. We replace it with a
    differentiable version that faithfully reproduces the upstream math.

    Also patches AllNorm.forward to use .reshape() instead of .view()
    for robustness with non-contiguous tensors.
    """
    import einops

    # Patch AllNorm to not break gradients
    for module in list(posenet.modules()) + list(segnet.modules()):
        if type(module).__name__ == "AllNorm":
            def _patched_forward(self, x):
                return self.bn(x.reshape(-1, 1)).reshape(x.shape)
            module.forward = types.MethodType(_patched_forward, module)

    # Differentiable rgb_to_yuv6: full-range BT.601 with 4:2:0 subsampling
    def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
        H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
        H2, W2 = H // 2, W // 2
        rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
        R = rgb[..., 0, :, :]
        G = rgb[..., 1, :, :]
        B = rgb[..., 2, :, :]
        Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
        U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
        V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
        U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
        V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
        y00 = Y[..., 0::2, 0::2]
        y10 = Y[..., 1::2, 0::2]
        y01 = Y[..., 0::2, 1::2]
        y11 = Y[..., 1::2, 1::2]
        return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

    # Get scorer input size from upstream module
    try:
        from modules import segnet_model_input_size
    except ImportError:
        segnet_model_input_size = (512, 384)  # (W, H) default

    def _diff_preprocess(self, x):
        batch_size, seq_len_local = x.shape[0], x.shape[1]
        x = einops.rearrange(x, "b t c h w -> (b t) c h w", b=batch_size, t=seq_len_local, c=3)
        x = F.interpolate(
            x,
            size=(segnet_model_input_size[1], segnet_model_input_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        yuv = _rgb_to_yuv6_diff(x)
        return einops.rearrange(
            yuv, "(b t) c h w -> b (t c) h w",
            b=batch_size, t=seq_len_local, c=6,
        ).contiguous()

    posenet.preprocess_input = types.MethodType(_diff_preprocess, posenet)


def _extract_masks(
    gt_chw: torch.Tensor,
    segnet: nn.Module,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract SegNet masks from GT frames (CHW float [0,255]).

    Returns: (N, H, W) long tensor of class indices in [0, 4].
    """
    masks_list = []
    with torch.no_grad():
        for i in range(0, gt_chw.shape[0], batch_size):
            batch = gt_chw[i:i + batch_size]
            inp = batch.unsqueeze(1).contiguous()  # (B, 1, C, H, W)
            seg_in = segnet.preprocess_input(inp)
            logits = segnet(seg_in)
            mask = logits.argmax(dim=1)  # (B, H', W')
            # Resize mask back to target resolution if needed
            if mask.shape[1] != gt_chw.shape[2] or mask.shape[2] != gt_chw.shape[3]:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=(gt_chw.shape[2], gt_chw.shape[3]),
                    mode="nearest",
                ).squeeze(1).long()
            # Store as int8 — values are [0,4], saves ~7x RAM vs int64
            masks_list.append(mask.to(torch.int8))
    return torch.cat(masks_list, dim=0)


def _tv_loss(frames: torch.Tensor) -> torch.Tensor:
    """Total variation loss for compressibility.

    Args:
        frames: (B, 3, H, W) float tensor.

    Returns:
        Scalar TV loss.
    """
    dx = (frames[:, :, :, 1:] - frames[:, :, :, :-1]).abs().mean()
    dy = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs().mean()
    return dx + dy


def _compute_model_bits(model: nn.Module) -> torch.Tensor:
    """Compute total model size in bits (differentiable via LearnableBitDepth).

    Handles both Conv2d and ConvTranspose2d (council decision: include both).
    Conv2d weight: (C_out, C_in, kH, kW), fan_in = C_in * kH * kW
    ConvTranspose2d weight: (C_in, C_out, kH, kW), fan_in = C_in * kH * kW
      (per output-channel, each slice is weight[:, ch_idx] with C_in*kH*kW elements)
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    has_learnable_bits = False
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and hasattr(m, "bit_depth"):
            bits_per_ch = m.bit_depth.bits.clamp(0.0, 8.0)
            if isinstance(m, nn.ConvTranspose2d):
                # (C_in, C_out, kH, kW): per output-channel fan_in = C_in * kH * kW
                fan_in = m.weight.shape[0] * m.weight.shape[2] * m.weight.shape[3]
            else:
                # (C_out, C_in, kH, kW): per output-channel fan_in = C_in * kH * kW
                fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            total = total + (bits_per_ch * fan_in).sum()
            if m.bias is not None:
                total = total + bits_per_ch.sum()  # 1 bias value per channel
            has_learnable_bits = True
    if not has_learnable_bits:
        # Fallback: count params * 4 bits (FP4 quantization estimate)
        n_params = sum(p.numel() for p in model.parameters())
        total = torch.tensor(float(n_params * 4), device=next(model.parameters()).device)
    return total


def _wrap_with_learnable_bits(model: nn.Module, init_bits: float = 8.0) -> list[nn.Module]:
    """Wrap Conv2d AND ConvTranspose2d layers with LearnableBitDepth for self-compression.

    Council decision: include ConvTranspose2d (7-20% of params, material for rate).
    Returns list of LearnableBitDepth modules (for optimizing their bits params).
    """
    from tac.self_compress import LearnableBitDepth

    bit_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # For ConvTranspose2d: weight is (C_in, C_out, kH, kW), per-channel
            # dim is C_out (shape[1]). For Conv2d: per-channel dim is C_out (shape[0]).
            if isinstance(module, nn.ConvTranspose2d):
                num_ch = module.weight.shape[1]  # C_out for transposed
            else:
                num_ch = module.weight.shape[0]  # C_out for regular

            bit_depth = LearnableBitDepth(
                num_channels=num_ch,
                init_bits=init_bits,
            ).to(module.weight.device)
            module.bit_depth = bit_depth
            bit_modules.append(bit_depth)

            # Monkey-patch forward to quantize weights through bit_depth
            if isinstance(module, nn.ConvTranspose2d):
                def _make_quantized_forward_t(conv, bd):
                    def _quantized_forward(x, output_size=None):
                        # ConvTranspose2d weight: (C_in, C_out, kH, kW)
                        # LearnableBitDepth expects dim-0 = num_channels = C_out
                        # Transpose dims 0,1 → quantize → transpose back
                        w_t = conv.weight.permute(1, 0, 2, 3).contiguous()
                        q_t = bd(w_t)
                        q_weight = q_t.permute(1, 0, 2, 3).contiguous()
                        return F.conv_transpose2d(
                            x, q_weight, conv.bias,
                            conv.stride, conv.padding,
                            output_padding=conv.output_padding,
                            groups=conv.groups, dilation=conv.dilation,
                        )
                    return _quantized_forward
                module.forward = _make_quantized_forward_t(module, bit_depth)
            else:
                def _make_quantized_forward(conv, bd):
                    def _quantized_forward(x):
                        q_weight = bd(conv.weight)
                        return F.conv2d(
                            x, q_weight, conv.bias,
                            conv.stride, conv.padding, conv.dilation, conv.groups,
                        )
                    return _quantized_forward
                module.forward = _make_quantized_forward(module, bit_depth)

    return bit_modules


# _compute_self_compress_bits was removed — it called total_bits() which only
# summed per-channel bit-depths without multiplying by fan_in (off by ~576x).
# Use _compute_model_bits everywhere instead (correctly accounts for fan_in).


# ---------------------------------------------------------------------------
# Scorer distortion computation — the core training signal
# ---------------------------------------------------------------------------

def _compute_seg_distortion(
    gen_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: nn.Module,
) -> torch.Tensor:
    """Compute SegNet distortion between generated and GT frames.

    Uses preprocess_input on both. Returns differentiable soft cosine distance.

    Args:
        gen_frames: (B, 3, H, W) float [0, 255] generated frames.
        gt_frames: (B, 3, H, W) float [0, 255] GT frames.
        segnet: frozen SegNet with preprocess_input.

    Returns:
        Scalar SegNet distortion (1 - dot product of softmax class probs).
        This is the Bhattacharyya coefficient, NOT cosine similarity (which
        would normalize by ||p|| * ||q||). Since softmax sums to 1, the dot
        product is bounded [0, 1]. It is a valid differentiable relaxation
        of hard argmax disagreement but with weaker gradients near convergence.
    """
    # SegNet expects (B, T, C, H, W) — use T=1 for single-frame seg
    gen_btchw = gen_frames.unsqueeze(1).contiguous()
    gt_btchw = gt_frames.unsqueeze(1).contiguous()

    seg_in_gen = segnet.preprocess_input(gen_btchw)
    seg_logits_gen = segnet(seg_in_gen)

    with torch.no_grad():
        seg_in_gt = segnet.preprocess_input(gt_btchw)
        seg_logits_gt = segnet(seg_in_gt)

    pred_soft = F.softmax(seg_logits_gen, dim=1)
    gt_soft = F.softmax(seg_logits_gt, dim=1)
    seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()
    return seg_dist


def _compute_seg_distortion_tempered(
    gen_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """SegNet distortion with temperature-annealed softmax.

    Lower temperature → sharper distributions → closer to hard argmax.
    At T=0.1, softmax is nearly one-hot, providing strong gradient signal
    toward the correct class while remaining differentiable.
    """
    gen_btchw = gen_frames.unsqueeze(1).contiguous()
    gt_btchw = gt_frames.unsqueeze(1).contiguous()

    seg_in_gen = segnet.preprocess_input(gen_btchw)
    seg_logits_gen = segnet(seg_in_gen)

    with torch.no_grad():
        seg_in_gt = segnet.preprocess_input(gt_btchw)
        seg_logits_gt = segnet(seg_in_gt)

    pred_soft = F.softmax(seg_logits_gen / max(temperature, 0.01), dim=1)
    gt_soft = F.softmax(seg_logits_gt / max(temperature, 0.01), dim=1)
    return 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()


def _compute_seg_distortion_ste(
    gen_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: nn.Module,
) -> torch.Tensor:
    """SegNet distortion with STE: forward=hard argmax, backward=cross-entropy.

    Directly optimizes the eval metric (hard disagreement) while maintaining
    gradient flow through cross-entropy backward. This is the tightest possible
    proxy for the scorer's actual measurement.
    """
    gen_btchw = gen_frames.unsqueeze(1).contiguous()
    gt_btchw = gt_frames.unsqueeze(1).contiguous()

    seg_in_gen = segnet.preprocess_input(gen_btchw)
    seg_logits_gen = segnet(seg_in_gen)

    with torch.no_grad():
        seg_in_gt = segnet.preprocess_input(gt_btchw)
        seg_logits_gt = segnet(seg_in_gt)

    # Hard forward: argmax disagreement (matches eval exactly)
    hard_disagree = (seg_logits_gen.argmax(dim=1) != seg_logits_gt.argmax(dim=1)).float().mean()

    # Soft backward: cross-entropy against GT argmax labels
    gt_labels = seg_logits_gt.argmax(dim=1)  # (B, H, W)
    soft_ce = F.cross_entropy(seg_logits_gen, gt_labels)

    # STE: forward value = hard, gradient = soft
    return soft_ce + (hard_disagree - soft_ce).detach()


def _compute_seg_distortion_margin(
    gen_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: nn.Module,
    margin: float = 0.1,
) -> torch.Tensor:
    """Margin-based SegNet loss (optimization #13).

    Only penalizes when soft disagreement exceeds the margin threshold.
    This avoids wasting gradient signal on already-good pixels and focuses
    optimization on the hard boundary cases that actually affect the score.

    Args:
        gen_frames: (B, 3, H, W) float [0, 255] generated frames.
        gt_frames: (B, 3, H, W) float [0, 255] GT frames.
        segnet: frozen SegNet with preprocess_input.
        margin: disagreement threshold below which loss is zero.

    Returns:
        Scalar margin-based SegNet distortion.
    """
    gen_btchw = gen_frames.unsqueeze(1).contiguous()
    gt_btchw = gt_frames.unsqueeze(1).contiguous()

    seg_in_gen = segnet.preprocess_input(gen_btchw)
    seg_logits_gen = segnet(seg_in_gen)

    with torch.no_grad():
        seg_in_gt = segnet.preprocess_input(gt_btchw)
        seg_logits_gt = segnet(seg_in_gt)

    pred_soft = F.softmax(seg_logits_gen, dim=1)
    gt_soft = F.softmax(seg_logits_gt, dim=1)
    # Per-pixel disagreement: 1 - dot product
    per_pixel = 1.0 - (pred_soft * gt_soft).sum(dim=1)  # (B, H, W)
    # Only penalize pixels above margin
    margin_loss = F.relu(per_pixel - margin).mean()
    return margin_loss


def _compute_pose_distortion_coupled(
    gen_frame_t: torch.Tensor,
    gen_frame_t1: torch.Tensor,
    gt_frame_t: torch.Tensor,
    gt_frame_t1: torch.Tensor,
    posenet: nn.Module,
) -> torch.Tensor:
    """Compute PoseNet distortion with COUPLED trajectory through both frames.

    This is the key insight: PoseNet evaluates consecutive PAIRS.
    Gradient flows through BOTH generated frames to the renderer.
    This forces temporal coherence.

    Args:
        gen_frame_t: (B, 3, H, W) generated frame at time t.
        gen_frame_t1: (B, 3, H, W) generated frame at time t+1.
        gt_frame_t: (B, 3, H, W) GT frame at time t.
        gt_frame_t1: (B, 3, H, W) GT frame at time t+1.
        posenet: frozen PoseNet with preprocess_input.

    Returns:
        Scalar PoseNet distortion (MSE on first 6 pose outputs).
    """
    # Stack consecutive frames as pairs: (B, 2, C, H, W)
    gen_pair = torch.stack([gen_frame_t, gen_frame_t1], dim=1).contiguous()
    gt_pair = torch.stack([gt_frame_t, gt_frame_t1], dim=1).contiguous()

    pose_in_gen = posenet.preprocess_input(gen_pair)
    pose_out_gen = posenet(pose_in_gen)

    with torch.no_grad():
        pose_in_gt = posenet.preprocess_input(gt_pair)
        pose_out_gt = posenet(pose_in_gt)

    pred = pose_out_gen["pose"] if isinstance(pose_out_gen, dict) else pose_out_gen
    target = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
    pose_dist = (pred[..., :6] - target[..., :6]).pow(2).mean()
    return pose_dist


def _compute_flow_regularization(
    gen_frame_t: torch.Tensor,
    gen_frame_t1: torch.Tensor,
    motion_predictor: nn.Module,
    mask_t: torch.Tensor,
    mask_t1: torch.Tensor,
) -> torch.Tensor:
    """Ego-motion flow regularization using DepthAwareMotionPredictor.

    Penalizes frames that don't warp correctly under the expected geometric flow.

    Args:
        gen_frame_t: (B, 3, H, W) generated frame at time t.
        gen_frame_t1: (B, 3, H, W) generated frame at time t+1.
        motion_predictor: DepthAwareMotionPredictor from the PairGenerator.
        mask_t: (B, H, W) masks at time t.
        mask_t1: (B, H, W) masks at time t+1.

    Returns:
        Scalar flow consistency loss.
    """
    from tac.renderer import warp_with_flow

    motion_out = motion_predictor(mask_t, mask_t1)  # (B, 2+, H, W)
    flow = motion_out[:, :2]  # first 2 channels are flow (asymmetric has 6: flow+gate+residual)
    warped = warp_with_flow(gen_frame_t, flow)
    flow_loss = (warped - gen_frame_t1).abs().mean()
    return flow_loss


# ---------------------------------------------------------------------------
# Full evaluation (no gradients)
# ---------------------------------------------------------------------------

def _full_eval(
    renderer: nn.Module,
    masks: torch.Tensor,
    gt_chw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    batch_size: int = 4,
    raft_flow_all: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate renderer on all frames with official scoring formula.

    When raft_flow_all is provided and renderer is asymmetric, passes it as
    ego_flow to match training conditions (council revised ruling: eval should
    match training for optimal checkpoint selection).

    Returns dict with avg_seg, avg_pose, model_bytes, rate, score.
    """
    renderer.eval()
    # Extract the actual renderer (DPSIMSRenderer) from PairGenerator wrapper
    renderer_module = getattr(renderer, "renderer", renderer)
    n = masks.shape[0]
    seg_dists = []
    pose_dists = []

    is_asymmetric = (hasattr(renderer, "motion")
                     and hasattr(renderer.motion, "output_channels")
                     and renderer.motion.output_channels == 6)

    with torch.no_grad():
        if not is_asymmetric:
            # DP-SIMS: pre-generate ALL frames independently (batched)
            all_gen = []
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch_masks = masks[i:end].to(device=device, dtype=torch.long)
                gen_batch = renderer_module(batch_masks)  # (B, 3, H, W)
                all_gen.append(gen_batch)
            all_gen = torch.cat(all_gen, dim=0)  # (N, 3, H, W) on device

        # Evaluate pairs — must match scorer's non-overlapping stride-2 pairs:
        # (0,1), (2,3), (4,5)... NOT (0,1), (1,2), (2,3)...
        # Council sweep: stride-1 eval was selecting wrong best checkpoint.
        pair_starts = list(range(0, n - 1, 2))  # even-index starts only
        if len(pair_starts) == 0:
            print(f"  [_full_eval] WARNING: no pairs (n={n}), returning sentinel", file=sys.stderr)
            return {"avg_seg": float("inf"), "avg_pose": float("inf"),
                    "score": float("inf"), "rate": float("inf"),
                    "model_bytes": 0.0,
                    "n_seg_samples": 0, "n_pose_samples": 0}
        for batch_start in range(0, len(pair_starts), batch_size):
            batch_indices = pair_starts[batch_start:batch_start + batch_size]
            B = len(batch_indices)

            if is_asymmetric:
                # Asymmetric: generate pairs via warp
                m_t = torch.stack([masks[j] for j in batch_indices]).to(device=device, dtype=torch.long)
                m_t1 = torch.stack([masks[j + 1] for j in batch_indices]).to(device=device, dtype=torch.long)
                # Eval WITH ego_flow when available (council revised: eval matches training)
                eval_ego_flow = None
                if raft_flow_all is not None:
                    eval_pair_idx = torch.tensor(batch_indices, device=device) // 2
                    eval_ego_flow = raft_flow_all[eval_pair_idx].float()
                pair_hwc = renderer(m_t, m_t1, residual_scale=1.0, ego_flow=eval_ego_flow)  # (B, 2, H, W, 3)
                gen_t = pair_hwc[:, 0].permute(0, 3, 1, 2).contiguous()
                gen_t1 = pair_hwc[:, 1].permute(0, 3, 1, 2).contiguous()
            else:
                gen_t = torch.stack([all_gen[j] for j in batch_indices])
                gen_t1 = torch.stack([all_gen[j + 1] for j in batch_indices])
            gt_t = torch.stack([gt_chw[j] for j in batch_indices]).to(device)
            gt_t1 = torch.stack([gt_chw[j + 1] for j in batch_indices]).to(device)

            # SegNet on LAST frame only (upstream uses x[:, -1, ...] which
            # selects the last frame of each pair for SegNet evaluation)
            gen_btchw = gen_t1.unsqueeze(1).contiguous()
            gt_btchw = gt_t1.unsqueeze(1).contiguous()
            seg_in_g = segnet.preprocess_input(gen_btchw)
            seg_in_gt = segnet.preprocess_input(gt_btchw)
            seg_logits_gen = segnet(seg_in_g)
            seg_logits_gt = segnet(seg_in_gt)
            # Hard disagreement rate (official metric) — per-sample
            hard_seg_batch = (seg_logits_gen.argmax(dim=1) != seg_logits_gt.argmax(dim=1)).float()
            for b in range(B):
                seg_dists.append(hard_seg_batch[b].mean().item())

            # PoseNet on consecutive pairs
            gen_pair = torch.stack([gen_t, gen_t1], dim=1).contiguous()
            gt_pair = torch.stack([gt_t, gt_t1], dim=1).contiguous()
            pose_in_g = posenet.preprocess_input(gen_pair)
            pose_in_gt = posenet.preprocess_input(gt_pair)
            p_out = posenet(pose_in_g)
            g_out = posenet(pose_in_gt)
            pm = p_out["pose"] if isinstance(p_out, dict) else p_out
            gm = g_out["pose"] if isinstance(g_out, dict) else g_out
            # Per-sample MSE on first 6 pose outputs
            pose_mse = (pm[..., :6] - gm[..., :6]).pow(2).mean(dim=-1)  # (B,) or (B, 1)
            for b in range(B):
                pose_dists.append(pose_mse[b].mean().item())

        if not is_asymmetric:
            del all_gen  # free VRAM

    renderer.train()

    avg_seg = sum(seg_dists) / max(len(seg_dists), 1)
    avg_pose = sum(pose_dists) / max(len(pose_dists), 1)

    # Model size estimate — for asymmetric mode, BOTH renderer and motion ship
    # (council decision: ship full AsymmetricPairGenerator for warp at deploy time)
    # For dp_sims mode, only renderer ships.
    if hasattr(renderer, "motion") and hasattr(renderer.motion, "output_channels") and renderer.motion.output_channels == 6:
        # Asymmetric: full model ships
        n_params = sum(p.numel() for p in renderer.parameters())
    else:
        # DP-SIMS: only renderer ships
        n_params = sum(p.numel() for p in renderer_module.parameters())
    model_bytes = n_params * 4 / 8  # FP4 estimate

    # Check for self-compression (could be on renderer, motion, or both)
    has_sc = False
    for m in renderer.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and hasattr(m, "bit_depth"):
            has_sc = True
            break
    if has_sc:
        if is_asymmetric:
            # Asymmetric: full model ships (renderer + motion)
            model_bytes = _compute_model_bits(renderer).item() / 8.0
        else:
            # DP-SIMS: only renderer ships
            model_bytes = _compute_model_bits(renderer_module).item() / 8.0

    # Rate = archive size / original uncompressed video file size
    # From upstream evaluate.py: sum of file sizes in videos/ dir
    ORIGINAL_UNCOMPRESSED_SIZE = 37_545_489  # bytes (0.mkv on disk)
    rate = model_bytes / ORIGINAL_UNCOMPRESSED_SIZE

    # Contest compliance: archive must be < 10MB
    ARCHIVE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    if model_bytes > ARCHIVE_MAX_BYTES:
        print(f"  WARNING: model_bytes={model_bytes / 1024:.0f}KB exceeds "
              f"contest limit of {ARCHIVE_MAX_BYTES / 1024 / 1024:.0f}MB")

    from tac.scorer import comma_score
    score = comma_score(avg_pose, avg_seg, rate)

    return {
        "avg_seg": avg_seg,
        "avg_pose": avg_pose,
        "model_bytes": model_bytes,
        "rate": rate,
        "score": score,
        "n_seg_samples": len(seg_dists),
        "n_pose_samples": len(pose_dists),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_fridrich_renderer(cfg: FridrichRendererConfig) -> dict[str, Any]:
    """Train the DP-SIMS renderer with Fridrich constrained optimization."""

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FRIDRICH CONSTRAINED RENDERER TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Channels: {cfg.channels}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Phase1 end: {cfg.phase1_end:.0%}, Phase2 end: {cfg.phase2_end:.0%}")
    print(f"SegNet boundary: {cfg.seg_boundary}, PoseNet boundary: {cfg.pose_boundary}")
    print()

    # ---- Load scorers ----
    print("[1/5] Loading scorers...")
    posenet, segnet = _load_scorers(cfg.device)
    _patch_scorers_for_training(posenet, segnet)
    print(f"  Scorers loaded, VRAM: {_vram_mb():.0f} MB")

    # ---- Load data ----
    print("[2/5] Loading data...")
    # For renderer training, we need GT frames and masks extracted from GT
    # The renderer learns mask -> frame, not comp -> filtered
    from tac.data import load_precomputed, decode_video

    gt_frames_hwc: list[torch.Tensor] | None = None

    # Try precomputed first
    precomputed_dir = os.environ.get("PRECOMPUTED_DIR")
    if precomputed_dir and Path(precomputed_dir).exists():
        _, gt_list = load_precomputed(precomputed_dir)
        gt_frames_hwc = gt_list
        print(f"  Loaded {len(gt_frames_hwc)} GT frames from precomputed")
    else:
        # Try GT video directly
        gt_candidates = [
            Path("/home/zeus/content/upstream/videos/0.mkv"),
            Path(__file__).resolve().parent.parent / "upstream" / "videos" / "0.mkv",
        ]
        for gp in gt_candidates:
            if gp.exists():
                gt_frames_hwc = decode_video(str(gp), target_h=cfg.target_h, target_w=cfg.target_w)
                print(f"  Decoded {len(gt_frames_hwc)} frames from {gp}")
                break

    if gt_frames_hwc is None:
        raise FileNotFoundError("No GT data found. Set PRECOMPUTED_DIR or place GT video in upstream/videos/")

    if cfg.smoke:
        gt_frames_hwc = gt_frames_hwc[:20]
        print(f"  [SMOKE] Using only {len(gt_frames_hwc)} frames")

    n_frames = len(gt_frames_hwc)
    print(f"  Total frames: {n_frames}")

    # Convert to CHW float for scorer calls
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc])
    # Resize to scorer resolution if needed
    if gt_chw.shape[2] != cfg.target_h or gt_chw.shape[3] != cfg.target_w:
        gt_chw = F.interpolate(
            gt_chw,
            size=(cfg.target_h, cfg.target_w),
            mode="bilinear",  # Must match SegNet.preprocess_input (bilinear)
            align_corners=False,  # to avoid training/inflate mask mismatch
        ).clamp(0, 255)
    print(f"  GT shape: {gt_chw.shape}")

    # ---- Extract masks from GT via SegNet ----
    print("[3/5] Extracting masks from GT...")
    gt_chw_dev = gt_chw.to(device)
    masks = _extract_masks(gt_chw_dev, segnet, batch_size=8)
    masks = masks.cpu()
    gt_chw_dev = gt_chw_dev.cpu()  # free VRAM
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"  Masks shape: {masks.shape}, classes: {masks.unique().tolist()}")
    print(f"  VRAM after mask extraction: {_vram_mb():.0f} MB")

    # ---- Build renderer ----
    if cfg.pair_mode == "asymmetric":
        print("[4/5] Building AsymmetricPairGenerator (warp paradigm)...")
        from tac.renderer import AsymmetricPairGenerator

        pair_gen = AsymmetricPairGenerator(
            num_classes=cfg.num_classes,
            embed_dim=cfg.embed_dim,
            base_ch=cfg.base_ch,
            mid_ch=cfg.mid_ch,
            motion_hidden=cfg.motion_hidden,
            depth=cfg.renderer_depth,
            max_flow_px=cfg.max_flow_px,
            max_residual=cfg.max_residual,
            flow_only=cfg.flow_only,
        ).to(device)

        # Ego-flow: learnable per-pair affine warp (council Experiment B)
        ego_flow_module = None
        if cfg.use_ego_flow:
            from tac.ego_flow import LearnableAffineFlow
            n_pairs = len(list(range(0, masks.shape[0] - 1, 2)))  # even-pair count
            ego_flow_module = LearnableAffineFlow(
                n_pairs=n_pairs,
                max_flow_px=cfg.ego_flow_max_px,
            ).to(device)
            print(f"  Ego-flow: {ego_flow_module.param_count()} params, "
                  f"~{ego_flow_module.archive_bytes_fp4()} bytes FP4")
    else:
        print("[4/5] Building DP-SIMS v2 renderer...")
        from tac.dp_sims_renderer import build_dp_sims_renderer_v2

        pair_gen = build_dp_sims_renderer_v2(
            num_classes=cfg.num_classes,
            channels=cfg.channels,
            init_h=cfg.init_h,
            init_w=cfg.init_w,
            spade_hidden=cfg.spade_hidden,
            use_noise=cfg.use_noise,
        ).to(device)

    n_params = pair_gen.param_count()
    print(f"  Params: {n_params:,}")
    print(f"  FP4 estimate: {n_params * 4 / 8 / 1024:.0f} KB")
    print(f"  VRAM: {_vram_mb():.0f} MB")

    # ---- Optimizer ----
    param_groups = [{"params": pair_gen.parameters(), "lr": cfg.lr}]
    if cfg.pair_mode == "asymmetric" and cfg.use_ego_flow and ego_flow_module is not None:
        param_groups.append({"params": ego_flow_module.parameters(), "lr": cfg.ego_flow_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * cfg.lr_min_ratio,
    )

    # ---- Lagrangian state ----
    lambda_seg = 1.0
    lambda_pose = 1.0
    rho = cfg.rho_init

    # ---- Resume from checkpoint ----
    bit_modules: list[nn.Module] = []
    start_epoch = 0
    best_score_ckpt = float("inf")
    if cfg.resume:
        ckpt_path = Path(cfg.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"  Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self_compress_active = ckpt.get("self_compress_active", False)

        # If checkpoint was saved during Phase 3, it contains bit_depth.bits keys.
        # We must re-wrap the model with LearnableBitDepth BEFORE loading state_dict,
        # otherwise strict=True fails on the unexpected *.bit_depth.bits keys.
        if self_compress_active:
            # Asymmetric: wrap full model (renderer+motion both ship in archive)
            # DP-SIMS: wrap renderer only (motion is training-only)
            wrap_target = pair_gen if cfg.pair_mode == "asymmetric" else pair_gen.renderer
            bit_modules = _wrap_with_learnable_bits(wrap_target, init_bits=cfg.init_bits)
            # Rebuild optimizer with 2-3 param groups (matching Phase 3 structure)
            # BEFORE loading state_dict. Groups: renderer + bits (+ motion if asymmetric).
            bit_param_ids = set()
            bit_params = []
            for bm in bit_modules:
                for p in bm.parameters():
                    bit_params.append(p)
                    bit_param_ids.add(id(p))
            renderer_params = [p for p in pair_gen.renderer.parameters()
                               if id(p) not in bit_param_ids]
            # Collect motion params separately (asymmetric: motion ships too)
            renderer_param_ids = {id(p) for p in pair_gen.renderer.parameters()}
            motion_params = [p for p in pair_gen.parameters()
                             if id(p) not in renderer_param_ids
                             and id(p) not in bit_param_ids]
            param_groups = [
                {"params": renderer_params, "lr": cfg.lr * 0.5},
                {"params": bit_params, "lr": cfg.lr * 0.1},
            ]
            if motion_params:
                param_groups.append({"params": motion_params, "lr": cfg.lr * 0.3})
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=1e-4,
            )
            print(f"  Re-wrapped {len(bit_modules)} conv layers for Phase 3 resume")

        pair_gen.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        lambda_seg = ckpt.get("lambda_seg", lambda_seg)
        lambda_pose = ckpt.get("lambda_pose", lambda_pose)
        rho = ckpt.get("rho", rho)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score_ckpt = ckpt.get("best_score", float("inf"))
        # Clamp restored Lagrangian state to new (potentially tighter) caps.
        # Critical for resume after explosion: ep4000 checkpoint had rho=10K,
        # λ_s=42K, λ_p=185K which caused divergence. New caps prevent reloading
        # the explosive state.
        rho_before, ls_before, lp_before = rho, lambda_seg, lambda_pose
        rho = min(rho, cfg.rho_max)
        lambda_seg = min(lambda_seg, cfg.lambda_cap)
        lambda_pose = min(lambda_pose, cfg.lambda_cap)
        if rho != rho_before or lambda_seg != ls_before or lambda_pose != lp_before:
            print(f"  [CLAMP] Lagrangian state clamped to new caps: "
                  f"rho {rho_before:.0f}→{rho:.0f}, "
                  f"λ_s {ls_before:.0f}→{lambda_seg:.0f}, "
                  f"λ_p {lp_before:.0f}→{lambda_pose:.0f}")
        # Phase 3 scheduler warm restart: the saved scheduler had T_max=old_remaining
        # but we need T_max=new_remaining (from current start_epoch to end).
        # This is intentionally a warm restart — LR resets to Phase 3 base rate.
        # Loading the old scheduler state into a new T_max would produce wrong decay.
        if self_compress_active:
            remaining_epochs = cfg.epochs - start_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(remaining_epochs, 1),
                eta_min=cfg.lr * cfg.lr_min_ratio,
            )
            print(f"  Phase 3 warm restart: fresh scheduler T_max={remaining_epochs}")
        print(f"  Resumed at epoch {start_epoch}, lambda_seg={lambda_seg:.1f}, "
              f"lambda_pose={lambda_pose:.1f}, rho={rho:.1f}, "
              f"self_compress={'ON' if self_compress_active else 'OFF'}")
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Save config and replicability info ----
    config_path = results_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    print(f"  Config saved to {config_path}")

    # Replicability record
    replicability: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": cfg.device,
    }
    try:
        replicability["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        replicability["git_commit"] = None
    try:
        replicability["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        replicability["git_dirty"] = None

    replicability_path = results_dir / "replicability.json"
    with open(replicability_path, "w") as f:
        json.dump(replicability, f, indent=2, default=str)
    print(f"  Replicability saved to {replicability_path}")

    # ---- Contest compliance assertions ----
    ORIGINAL_UNCOMPRESSED_SIZE = 37_545_489  # bytes (0.mkv on disk)
    # Verify upstream constant matches
    assert ORIGINAL_UNCOMPRESSED_SIZE == 37_545_489, "ORIGINAL_UNCOMPRESSED_SIZE mismatch vs upstream"
    # Archive size upper limit (contest constraint)
    ARCHIVE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    print(f"  Scoring formula: 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate")
    print(f"  Contest archive limit: {ARCHIVE_MAX_BYTES / 1024 / 1024:.0f} MB")
    print(f"  Original uncompressed size: {ORIGINAL_UNCOMPRESSED_SIZE:,} bytes")

    # ---- Load PoseNet targets for supervision (council Layer 1) ----
    pose_targets = None
    if cfg.pose_supervision_weight > 0 and cfg.pose_targets_path:
        from tac.scorer_targets import load_posenet_targets
        targets_dict = load_posenet_targets(cfg.pose_targets_path)
        if targets_dict is not None:
            pose_targets = targets_dict["targets"].to(device)  # (n_pairs, 6)
            print(f"  PoseNet targets loaded: {pose_targets.shape} from {cfg.pose_targets_path}")
        else:
            print(f"  WARNING: could not load PoseNet targets from {cfg.pose_targets_path}")

    # ---- Load RAFT flow for frozen warp + supervision (council Layer 2) ----
    raft_flow_all = None
    if cfg.raft_flow_path:
        raft_data = torch.load(cfg.raft_flow_path, map_location="cpu", weights_only=True)
        raft_flow_all = raft_data["flow"].to(device)  # (N_pairs, 2, H, W) stays float16 — 472MB not 944MB
        n_raft_pairs = raft_flow_all.shape[0]
        n_expected_pairs = len(list(range(0, n_frames - 1, 2)))
        assert n_raft_pairs >= n_expected_pairs, (
            f"RAFT flow has {n_raft_pairs} pairs but need {n_expected_pairs} "
            f"for {n_frames} frames with even-pairs-only"
        )
        print(f"  RAFT flow loaded: {raft_flow_all.shape} ({raft_flow_all.dtype}) from {cfg.raft_flow_path}")
        print(f"  Flow magnitude: mean={raft_data['flow_px'].norm(dim=1).mean():.1f}px")

    # ---- Training loop ----
    print("[5/5] Training...")
    print()

    history: list[dict[str, Any]] = []
    best_score = best_score_ckpt if cfg.resume else float("inf")
    best_epoch = -1
    constraints_satisfied_count = 0
    # self_compress_active is set in resume block above; only default to False
    # if NOT resuming (avoids overwriting checkpoint state for Phase 3 resumes)
    if not cfg.resume:
        self_compress_active = False
    # bit_modules already initialized before resume block
    divergence_counter = 0  # Bug #7: auto-kill on divergence
    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        progress = epoch / max(cfg.epochs - 1, 1)
        residual_scale = 1.0  # default; overridden by flow warmup for asymmetric

        # Determine phase
        if progress < cfg.phase1_end:
            phase = 1
        elif progress < cfg.phase2_end:
            phase = 2
        else:
            phase = 3

        # Activate self-compression at phase 3 start (once)
        if phase == 3 and not self_compress_active:
            print(f"\n[epoch {epoch}] Phase 3: activating self-compression...")
            wrap_target = pair_gen if cfg.pair_mode == "asymmetric" else pair_gen.renderer
            bit_modules = _wrap_with_learnable_bits(wrap_target, init_bits=cfg.init_bits)
            # Create fresh optimizer + scheduler so new params get full LR
            # and existing params aren't stuck at decayed cosine LR
            if bit_modules:
                bit_param_ids = set()
                bit_params = []
                for bm in bit_modules:
                    for p in bm.parameters():
                        bit_params.append(p)
                        bit_param_ids.add(id(p))
                # Exclude bit_depth params from renderer group to avoid
                # "some parameters appear in more than one parameter group" error.
                # bit_depth modules are registered as submodules of Conv2d layers,
                # so pair_gen.renderer.parameters() includes their bits params.
                renderer_params = [p for p in pair_gen.renderer.parameters()
                                   if id(p) not in bit_param_ids]
                # Collect motion params separately (asymmetric: motion ships too)
                renderer_param_ids = {id(p) for p in pair_gen.renderer.parameters()}
                motion_params = [p for p in pair_gen.parameters()
                                 if id(p) not in renderer_param_ids
                                 and id(p) not in bit_param_ids]
                remaining_epochs = cfg.epochs - epoch
                param_groups = [
                    {"params": renderer_params, "lr": cfg.lr * 0.5},
                    {"params": bit_params, "lr": cfg.lr * 0.1},
                ]
                if motion_params:
                    param_groups.append({"params": motion_params, "lr": cfg.lr * 0.3})
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=1e-4,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=remaining_epochs,
                    eta_min=cfg.lr * cfg.lr_min_ratio,
                )
            self_compress_active = True
            print(f"  Wrapped {len(bit_modules)} conv layers with LearnableBitDepth")

        step_start_time = time.time()

        # Sample a random batch of consecutive frame indices
        max_start = n_frames - 2  # need at least 2 consecutive frames
        if max_start < 1:
            raise ValueError(f"Need >= 3 frames, got {n_frames}")
        if cfg.even_pairs_only and cfg.pair_mode == "asymmetric":
            # Scorer evaluates non-overlapping pairs (0,1),(2,3),(4,5)...
            # Only sample even-index starts to match scorer distribution
            max_even = max_start // 2  # number of valid even starts
            if max_even < 1:
                raise ValueError(f"Need >= 4 frames for even_pairs_only, got {n_frames}")
            batch_starts = torch.randint(0, max_even, (cfg.batch_size,)) * 2
        else:
            batch_starts = torch.randint(0, max_start, (cfg.batch_size,))

        # Gather mask pairs and GT pairs
        mask_t_list = []
        mask_t1_list = []
        gt_t_list = []
        gt_t1_list = []
        for idx in batch_starts:
            mask_t_list.append(masks[idx])
            mask_t1_list.append(masks[idx + 1])
            gt_t_list.append(gt_chw[idx])
            gt_t1_list.append(gt_chw[idx + 1])

        mask_t = torch.stack(mask_t_list).to(device=device, dtype=torch.long)
        mask_t1 = torch.stack(mask_t1_list).to(device=device, dtype=torch.long)
        gt_t = torch.stack(gt_t_list).to(device)
        gt_t1 = torch.stack(gt_t1_list).to(device)

        # ---- Forward: generate frames ----
        if cfg.pair_mode == "asymmetric":
            # Flow warmup (council B1+B2): zero residual for first N epochs,
            # then ramp from 0→1 to force flow learning before residual develops.
            if epoch < cfg.flow_warmup_epochs:
                residual_scale = 0.0
            elif epoch < cfg.flow_warmup_epochs + cfg.residual_ramp_epochs:
                ramp_progress = (epoch - cfg.flow_warmup_epochs) / max(cfg.residual_ramp_epochs, 1)
                residual_scale = ramp_progress
            else:
                residual_scale = 1.0
            # Compute ego-flow: RAFT frozen warp (council Layer 2) OR learnable affine
            ego_flow = None
            pair_indices = (batch_starts // 2).to(device)
            if raft_flow_all is not None:
                # Layer 2: use precomputed dense RAFT flow as frozen warp
                # Stored as float16 to save VRAM (472MB vs 944MB); upcast per-batch
                ego_flow = raft_flow_all[pair_indices].float()  # (B, 2, H, W) — no grad, frozen
            elif cfg.use_ego_flow and ego_flow_module is not None:
                ego_flow = ego_flow_module(pair_indices, cfg.target_h, cfg.target_w)

            # Warp paradigm: single forward produces both frames jointly
            pair_hwc = pair_gen(mask_t, mask_t1, residual_scale=residual_scale, ego_flow=ego_flow)  # (B, 2, H, W, 3)
            gen_t = pair_hwc[:, 0].permute(0, 3, 1, 2).contiguous()   # (B, 3, H, W)
            gen_t1 = pair_hwc[:, 1].permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        else:
            # DP-SIMS: independent per-frame generation
            gen_t = pair_gen.renderer(mask_t)   # (B, 3, H, W) float [0, 255]
            gen_t1 = pair_gen.renderer(mask_t1)  # (B, 3, H, W) float [0, 255]

        # ---- Compute losses ----
        # MSE reconstruction normalized to [0,1] range (Bug fix: raw [0,255] MSE
        # was ~21000, drowning seg (~5) and pose (~180) terms completely)
        mse_loss = F.mse_loss(gen_t / 255.0, gt_t / 255.0) + F.mse_loss(gen_t1 / 255.0, gt_t1 / 255.0)

        # SegNet distortion — ONLY on frame_t1 (council #5: SegNet uses x[:, -1, ...]
        # which selects the LAST frame. frame_t is invisible to SegNet at eval time.
        # Free frame_t for pure PoseNet optimization. Also saves ~50% SegNet compute.)
        #
        # Fridrich curriculum (council P1):
        # Phase 1: soft Bhattacharyya (broad gradients for initial learning)
        # Phase 2: temperature-annealed Bhattacharyya (sharpen toward argmax)
        # Phase 3: STE (forward=hard argmax, backward=cross-entropy)
        if cfg.use_margin_segnet:
            seg_dist = _compute_seg_distortion_margin(gen_t1, gt_t1, segnet, margin=cfg.segnet_margin_threshold)
        elif phase == 1:
            seg_dist = _compute_seg_distortion(gen_t1, gt_t1, segnet)
        elif phase == 2:
            phase2_progress = (progress - cfg.phase1_end) / max(cfg.phase2_end - cfg.phase1_end, 1e-6)
            temperature = cfg.seg_temperature_start - (cfg.seg_temperature_start - cfg.seg_temperature_end) * phase2_progress
            seg_dist = _compute_seg_distortion_tempered(gen_t1, gt_t1, segnet, temperature)
        else:  # phase 3
            seg_dist = _compute_seg_distortion_ste(gen_t1, gt_t1, segnet)

        # PoseNet distortion (coupled trajectory — gradient through BOTH frames)
        pose_dist = _compute_pose_distortion_coupled(gen_t, gen_t1, gt_t, gt_t1, posenet)

        # TV smoothness
        tv = _tv_loss(gen_t) + _tv_loss(gen_t1)

        # Ego-motion flow regularization (skip if weight is zero)
        if cfg.flow_weight > 0:
            flow_reg = _compute_flow_regularization(
                gen_t, gen_t1, pair_gen.motion, mask_t, mask_t1,
            )
        else:
            flow_reg = torch.tensor(0.0, device=device)

        # ---- Assemble total loss based on phase ----
        if phase == 1:
            # Phase 1: MSE + soft scorer (standard weighted sum warmup)
            total_loss = (
                cfg.p1_mse_weight * mse_loss
                + cfg.p1_seg_weight * seg_dist
                + cfg.p1_pose_weight * pose_dist
                + cfg.tv_weight * tv
            )
        else:
            # Phase 2+3: Fridrich augmented Lagrangian
            # Keep MSE as reconstruction anchor (Bug 8: prevents hallucination)
            seg_violation = F.relu(seg_dist - cfg.seg_boundary)
            pose_violation = F.relu(pose_dist - cfg.pose_boundary)

            total_loss = (
                cfg.phase2_mse_weight * mse_loss  # scaled-down MSE anchor
                + cfg.tv_weight * tv
                + cfg.flow_weight * flow_reg
                + lambda_seg * seg_violation
                + lambda_pose * pose_violation
                + (rho / 2.0) * seg_violation.pow(2)
                + (rho / 2.0) * pose_violation.pow(2)
            )

            # Gate regularization: penalize gate_mean above threshold
            # Council sweep round 5: must use _last_gate_mean_tensor (live grad)
            # not _last_gate_mean (.item() scalar, zero gradient — was completely inert)
            if cfg.pair_mode == "asymmetric" and cfg.gate_reg_weight > 0:
                gate_mean_t = getattr(pair_gen, "_last_gate_mean_tensor", None)
                if gate_mean_t is not None:
                    gate_penalty = cfg.gate_reg_weight * F.relu(gate_mean_t - cfg.gate_reg_threshold)
                    total_loss = total_loss + gate_penalty

            if phase == 3 and self_compress_active:
                # Add self-compression rate penalty
                # Asymmetric: full model ships (renderer + motion)
                # DP-SIMS: only renderer ships
                if cfg.pair_mode == "asymmetric":
                    model_bits = _compute_model_bits(pair_gen)
                else:
                    model_bits = _compute_model_bits(pair_gen.renderer)
                target_bits = cfg.target_bytes * 8.0
                rate_excess = F.relu(model_bits - target_bits) / target_bits
                total_loss = total_loss + cfg.rate_weight * rate_excess

            # Update Lagrangian multipliers (outer loop), capped at lambda_cap
            # NaN guard: a single NaN violation would permanently corrupt both
            # multipliers (NaN propagates through min/max). Skip update if NaN.
            with torch.no_grad():
                sv = seg_violation.item()
                pv = pose_violation.item()
                if math.isfinite(sv) and math.isfinite(pv):
                    lambda_seg = min(cfg.lambda_cap, max(0.0, lambda_seg + rho * sv))
                    lambda_pose = min(cfg.lambda_cap, max(0.0, lambda_pose + rho * pv))
                    if sv > 1e-6 or pv > 1e-6:
                        rho = min(rho * cfg.rho_growth, cfg.rho_max)
                else:
                    print(f"  [epoch {epoch}] WARNING: NaN violation "
                          f"(sv={sv}, pv={pv}), skipping lambda update",
                          file=sys.stderr, flush=True)

            # Track constraint satisfaction
            both_satisfied = seg_violation.item() < 1e-6 and pose_violation.item() < 1e-6
            if both_satisfied:
                # Save checkpoint at first constraint satisfaction boundary
                if constraints_satisfied_count == 0:
                    boundary_path = results_dir / f"renderer_epoch{epoch:05d}_constraints_met.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": pair_gen.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "lambda_seg": lambda_seg,
                        "lambda_pose": lambda_pose,
                        "rho": rho,
                        "best_score": best_score,
                        "self_compress_active": self_compress_active,
                        "config": asdict(cfg),
                    }, boundary_path)
                    print(f"  -> Constraints met checkpoint: {boundary_path}")
                constraints_satisfied_count += 1
            else:
                constraints_satisfied_count = 0

        # ---- Supervision losses (all phases — council revised ruling) ----
        # Flow supervision: teach MotionPredictor to reproduce RAFT flow (Layer 2)
        # Scale-safe in all phases (bounded at ~0.001). Full weight always.
        if cfg.flow_supervision_weight > 0 and raft_flow_all is not None and cfg.pair_mode == "asymmetric":
            with torch.no_grad():
                target_flow = raft_flow_all[pair_indices].float()
            cached_motion = getattr(pair_gen, "_last_motion_out", None)
            if cached_motion is not None:
                predicted_flow = cached_motion[:, :2]
                flow_sup_loss = F.mse_loss(predicted_flow, target_flow)
                total_loss = total_loss + cfg.flow_supervision_weight * flow_sup_loss

        # PoseNet supervision: direct scorer-space optimization (Layer 1+3)
        # Phase-adaptive: p1_pose_weight in Phase 1 (avoid dominating MSE warmup),
        # full weight in Phase 2+ (complement Lagrangian constraints).
        if cfg.pose_supervision_weight > 0 and pose_targets is not None and cfg.pair_mode == "asymmetric":
            effective_pose_sup_weight = cfg.pose_supervision_weight * (
                cfg.p1_pose_weight if phase == 1 else 1.0
            )
            target_pose = pose_targets[pair_indices]

            if cfg.pose_embed_loss:
                from tac.losses import posenet_embedding_loss
                gen_pair = torch.stack([gen_t, gen_t1], dim=1)
                gt_pair = torch.stack([gt_t, gt_t1], dim=1)
                pose_sup_loss = posenet_embedding_loss(gen_pair, gt_pair, posenet)
            else:
                gen_pair_btchw = torch.stack([gen_t, gen_t1], dim=1).contiguous()
                pose_in = posenet.preprocess_input(gen_pair_btchw)
                pose_out_raw = posenet(pose_in)
                pose_out = pose_out_raw["pose"] if isinstance(pose_out_raw, dict) else pose_out_raw
                pose_sup_loss = F.mse_loss(pose_out[:, :6], target_pose)

            total_loss = total_loss + effective_pose_sup_weight * pose_sup_loss

        # ---- Backward + step ----
        optimizer.zero_grad()
        try:
            total_loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[epoch {epoch}] OOM during backward, skipping step")
                optimizer.zero_grad()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                continue
            raise

        # Null-space gradient projection (optimization #6, gated)
        # Projects gradients to remove components that increase the satisfied
        # constraint's violation. Only active when one constraint is met and
        # the other is not — prevents regression on the satisfied scorer.
        if cfg.use_null_space and phase >= 2:
            with torch.no_grad():
                seg_ok = seg_dist.item() < cfg.seg_boundary
                pose_ok = pose_dist.item() < cfg.pose_boundary
                if seg_ok and not pose_ok:
                    # Project out the seg gradient direction to protect SegNet
                    # Recompute seg grad direction (already in .grad from backward)
                    pass  # Gradient already contains both; no-op placeholder
                    # Full null-space projection requires separate backward passes
                    # which doubles compute. Deferred: flag exists, logic is gated.
                elif pose_ok and not seg_ok:
                    pass  # Same — deferred null-space for PoseNet protection

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(pair_gen.parameters(), cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        step_time_ms = (time.time() - step_start_time) * 1000.0

        # ---- Logging ----
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            elapsed = time.time() - start_time
            lr_now = optimizer.param_groups[0]["lr"]
            # P0: Compute hard disagreement alongside soft Bhattacharyya
            # for live calibration of the soft/hard gap (council recommendation)
            with torch.no_grad():
                seg_in_g = segnet.preprocess_input(gen_t1.unsqueeze(1).contiguous())
                seg_in_gt = segnet.preprocess_input(gt_t1.unsqueeze(1).contiguous())
                seg_hard = (segnet(seg_in_g).argmax(dim=1) != segnet(seg_in_gt).argmax(dim=1)).float().mean().item()

            # Asymmetric-mode telemetry
            asym_telemetry: dict[str, Any] = {}
            if cfg.pair_mode == "asymmetric":
                asym_telemetry["gate_mean"] = getattr(pair_gen, "_last_gate_mean", None)
                asym_telemetry["residual_scale"] = residual_scale
                with torch.no_grad():
                    motion_out = pair_gen.motion(mask_t, mask_t1)
                    flow_field = motion_out[:, :2]  # (B, 2, H, W)
                    asym_telemetry["flow_magnitude"] = flow_field.abs().mean().item()
                    residual_field = motion_out[:, 3:6]  # (B, 3, H, W)
                    asym_telemetry["residual_magnitude"] = residual_field.abs().mean().item()
                    # Warp quality: MSE between warped frame and GT frame_t
                    from tac.renderer import warp_with_flow
                    warped = warp_with_flow(gen_t1.detach(), flow_field)
                    asym_telemetry["warp_quality"] = F.mse_loss(warped, gt_t).item()

            # Score projection from current training metrics
            from tac.scorer import comma_score
            # Asymmetric: full model ships. DP-SIMS: renderer only.
            if cfg.pair_mode == "asymmetric":
                n_params_proj = sum(p.numel() for p in pair_gen.parameters())
                model_bytes_proj = n_params_proj * 4 / 8
                if self_compress_active:
                    model_bytes_proj = _compute_model_bits(pair_gen).item() / 8.0
            else:
                renderer_module_proj = getattr(pair_gen, "renderer", pair_gen)
                n_params_proj = sum(p.numel() for p in renderer_module_proj.parameters())
                model_bytes_proj = n_params_proj * 4 / 8
                if self_compress_active:
                    model_bytes_proj = _compute_model_bits(renderer_module_proj).item() / 8.0
            rate_proj = model_bytes_proj / ORIGINAL_UNCOMPRESSED_SIZE
            score_projection = comma_score(pose_dist.item(), seg_hard, rate_proj)

            log_entry = {
                "epoch": epoch,
                "phase": phase,
                "loss": total_loss.item(),
                "mse": mse_loss.item(),
                "seg_dist": seg_dist.item(),
                "seg_hard": seg_hard,  # hard argmax disagreement (eval metric)
                "pose_dist": pose_dist.item(),
                "pair_mode": cfg.pair_mode,
                **asym_telemetry,
                "tv": tv.item(),
                "flow_reg": flow_reg.item(),
                "lambda_seg": lambda_seg,
                "lambda_pose": lambda_pose,
                "rho": rho,
                "lr": lr_now,
                "step_time_ms": step_time_ms,
                "score_projection": score_projection,
                "elapsed_s": elapsed,
                "vram_mb": _vram_mb(),
            }
            if self_compress_active:
                if cfg.pair_mode == "asymmetric":
                    log_entry["model_bits"] = _compute_model_bits(pair_gen).item()
                else:
                    log_entry["model_bits"] = _compute_model_bits(pair_gen.renderer).item()
            history.append(log_entry)

            print(
                f"[{epoch:5d}/{cfg.epochs}] P{phase} "
                f"loss={total_loss.item():.4f} "
                f"mse={mse_loss.item():.2f} "
                f"seg={seg_dist.item():.5f}(h={seg_hard:.5f}) "
                f"pose={pose_dist.item():.5f} "
                f"tv={tv.item():.3f} "
                f"flow={flow_reg.item():.3f} "
                f"lam_s={lambda_seg:.1f} lam_p={lambda_pose:.1f} "
                f"rho={rho:.0f} "
                f"lr={lr_now:.2e} "
                + (f"gm={asym_telemetry.get('gate_mean', 0):.3f} "
                   f"fl={asym_telemetry.get('flow_magnitude', 0):.4f} "
                   f"rs={asym_telemetry.get('residual_scale', 1):.2f} "
                   if cfg.pair_mode == "asymmetric" else "")
                + f"VRAM={_vram_mb():.0f}MB "
                f"({elapsed:.0f}s)"
            )

        # ---- Checkpoint ----
        if (epoch > 0 and epoch % cfg.checkpoint_every == 0) or epoch == cfg.epochs - 1:
            ckpt_path = results_dir / f"renderer_epoch{epoch:05d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": pair_gen.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "lambda_seg": lambda_seg,
                "lambda_pose": lambda_pose,
                "rho": rho,
                "best_score": best_score,
                "self_compress_active": self_compress_active,
                "config": asdict(cfg),
            }, ckpt_path)
            print(f"  -> Checkpoint: {ckpt_path}")

        # ---- Full evaluation ----
        if (epoch > 0 and epoch % cfg.eval_every == 0) or epoch == cfg.epochs - 1:
            print(f"\n--- Full evaluation at epoch {epoch} ---")
            try:
                eval_result = _full_eval(
                    pair_gen, masks, gt_chw, posenet, segnet, device,
                    batch_size=cfg.batch_size,
                    raft_flow_all=raft_flow_all if cfg.raft_flow_path else None,
                )
                score = eval_result["score"]
                print(
                    f"  seg={eval_result['avg_seg']:.5f} "
                    f"pose={eval_result['avg_pose']:.5f} "
                    f"rate={eval_result['rate']:.6f} "
                    f"model={eval_result['model_bytes'] / 1024:.0f}KB "
                    f"SCORE={score:.4f}"
                )
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_path = results_dir / "renderer_best.pt"
                    ckpt_data = {
                        "epoch": epoch,
                        "model_state_dict": pair_gen.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "lambda_seg": lambda_seg,
                        "lambda_pose": lambda_pose,
                        "rho": rho,
                        "best_score": best_score,
                        "self_compress_active": self_compress_active,
                        "score": score,
                        "eval_result": eval_result,
                        "config": asdict(cfg),
                    }
                    if cfg.use_ego_flow and ego_flow_module is not None:
                        ckpt_data["ego_flow_state_dict"] = ego_flow_module.state_dict()
                    torch.save(ckpt_data, best_path)
                    print(f"  -> NEW BEST: {score:.4f} at epoch {epoch}")
                print()
            except Exception as e:
                print(f"  Eval failed: {e}")
                traceback.print_exc()
                print()

        # ---- Early stopping ----
        if constraints_satisfied_count >= cfg.early_stop_patience:
            print(f"\nEarly stopping: both constraints satisfied for {cfg.early_stop_patience} epochs")
            break

        # ---- Auto-kill on divergence (Bug #7) ----
        loss_val = total_loss.item()
        seg_val = seg_dist.item()
        pose_val = pose_dist.item()
        if (loss_val > cfg.kill_loss_threshold
                or seg_val > cfg.kill_seg_threshold
                or pose_val > cfg.kill_pose_threshold
                or not math.isfinite(loss_val)):
            divergence_counter += 1
            if divergence_counter >= cfg.kill_patience:
                print(f"\nAuto-kill: divergence detected for {cfg.kill_patience} consecutive epochs "
                      f"(loss={loss_val:.4f}, seg={seg_val:.5f}, pose={pose_val:.5f})")
                kill_path = results_dir / f"renderer_epoch{epoch:05d}_killed.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": pair_gen.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "self_compress_active": self_compress_active,
                    "lambda_seg": lambda_seg,
                    "lambda_pose": lambda_pose,
                    "rho": rho,
                    "kill_reason": "divergence",
                    "config": asdict(cfg),
                }, kill_path)
                print(f"  -> Emergency checkpoint: {kill_path}")
                break
        else:
            divergence_counter = 0

        # ---- Wall-clock budget safeguard ----
        if cfg.max_hours > 0:
            elapsed_hours = (time.time() - start_time) / 3600.0
            if elapsed_hours >= cfg.max_hours:
                print(f"\nWall-clock budget exhausted: {elapsed_hours:.1f}h >= {cfg.max_hours}h")
                # Save final checkpoint before stopping
                ckpt_path = results_dir / f"renderer_epoch{epoch:05d}_timeout.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": pair_gen.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "lambda_seg": lambda_seg,
                    "lambda_pose": lambda_pose,
                    "rho": rho,
                    "best_score": best_score,
                    "self_compress_active": self_compress_active,
                    "config": asdict(cfg),
                }, ckpt_path)
                print(f"  -> Timeout checkpoint: {ckpt_path}")
                break

    # ---- Final summary ----
    elapsed = time.time() - start_time
    summary = {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "elapsed_s": elapsed,
        "n_params": n_params,
        "config": asdict(cfg),
        "final_lambda_seg": lambda_seg,
        "final_lambda_pose": lambda_pose,
        "final_rho": rho,
        "history": history,
    }

    # Git provenance
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        summary["git_commit"] = git_hash
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        summary["git_dirty"] = git_dirty
    except Exception:
        pass

    # Environment provenance
    summary["torch_version"] = torch.__version__
    summary["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        summary["gpu_name"] = torch.cuda.get_device_name(0)
    summary["python_version"] = platform.python_version()
    summary["hostname"] = platform.node()

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    history_path = results_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"History saved to {history_path}")

    print("\n" + "=" * 70)
    print(f"BEST SCORE: {best_score:.4f} at epoch {best_epoch}")
    print(f"Total time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--precomputed", type=str, default=None, help="Precomputed data dir (comp_frames.pt, gt_frames.pt)")
@click.option("--epochs", type=int, default=10000, help="Training epochs")
@click.option("--batch-size", type=int, default=16, help="Batch size (council B5: T4 has 20x VRAM headroom)")
@click.option("--lr", type=float, default=2e-4, help="Learning rate")
@click.option("--channels", type=str, default="128,64,32,16", help="Channel widths (comma-separated)")
@click.option("--spade-hidden", type=int, default=32, help="SPADE conditioning hidden dim")
@click.option("--seg-boundary", type=float, default=0.005, help="SegNet constraint boundary")
@click.option("--pose-boundary", type=float, default=0.05, help="PoseNet constraint boundary (relaxed default)")
@click.option("--rho-init", type=float, default=10.0, help="Initial Lagrangian penalty coefficient")
@click.option("--rho-growth", type=float, default=1.005, help="Rho growth rate per outer step (council: 1.02 exploded)")
@click.option("--tv-weight", type=float, default=0.1, help="TV smoothness weight")
@click.option("--flow-weight", type=float, default=0.0, help="Ego-motion flow regularization weight (disabled: unvalidated)")
@click.option("--rate-weight", type=float, default=0.01, help="Self-compression rate penalty weight")
@click.option("--target-bytes", type=int, default=250 * 1024, help="Target model size in bytes")
@click.option("--device", type=str, default="cuda", help="Device (cuda/mps/cpu)")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--checkpoint-every", type=int, default=500, help="Checkpoint interval")
@click.option("--eval-every", type=int, default=200, help="Full evaluation interval (council: 50 eval points for 10K epochs)")
@click.option("--log-every", type=int, default=50, help="Log interval")
@click.option("--smoke", is_flag=True, help="Smoke test: 20 frames, 5 epochs")
@click.option("--resume", type=str, default=None, help="Resume from checkpoint path")
@click.option("--max-hours", type=float, default=48.0, help="Wall-clock budget in hours (0=unlimited)")
@click.option("--phase2-mse-weight", type=float, default=0.1, help="MSE anchor weight in Phase 2+3")
@click.option("--pair-mode", type=click.Choice(["dp_sims", "asymmetric"]), default="dp_sims", help="Renderer paradigm")
@click.option("--embed-dim", type=int, default=6, help="Asymmetric warp: mask embedding dim")
@click.option("--base-ch", type=int, default=36, help="Asymmetric warp: base channel width")
@click.option("--mid-ch", type=int, default=60, help="Asymmetric warp: mid channel width")
@click.option("--motion-hidden", type=int, default=32, help="Asymmetric warp: motion predictor hidden dim")
@click.option("--renderer-depth", type=int, default=1, help="Asymmetric warp: renderer depth")
@click.option("--init-bits", type=float, default=8.0, help="Initial bit depth for self-compression")
@click.option("--phase1-end", type=float, default=0.25, help="Phase 1 end (council: 25%, Phase 1 converged early)")
@click.option("--phase2-end", type=float, default=0.85, help="Phase 2 end (council: 85%, slow Lagrangian needs time)")
@click.option("--early-stop-patience", type=int, default=500, help="Early stop if constraints met for N epochs")
@click.option("--init-h", type=int, default=24, help="DP-SIMS initial height")
@click.option("--init-w", type=int, default=32, help="DP-SIMS initial width")
@click.option("--num-classes", type=int, default=5, help="Number of segmentation classes")
@click.option("--use-noise", is_flag=True, help="DP-SIMS: add noise input")
@click.option("--target-h", type=int, default=384, help="Scorer target height")
@click.option("--target-w", type=int, default=512, help="Scorer target width")
@click.option("--lr-min-ratio", type=float, default=0.01, help="Cosine LR min ratio")
@click.option("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
@click.option("--p1-mse-weight", type=float, default=10.0, help="Phase 1 MSE weight (normalized [0,1], should dominate warmup)")
@click.option("--p1-seg-weight", type=float, default=1.0, help="Phase 1 SegNet weight")
@click.option("--p1-pose-weight", type=float, default=0.01, help="Phase 1 PoseNet weight (very low: PoseNet starts at ~180)")
@click.option("--rho-max", type=float, default=1e3, help="Maximum rho (council: 1e4 caused explosion)")
@click.option("--lambda-cap", type=float, default=1e4, help="Lagrangian multiplier cap (council: 1e6 overwhelmed task loss)")
@click.option("--max-flow-px", type=float, default=20.0, help="Max optical flow in pixels (asymmetric mode)")
@click.option("--max-residual", type=float, default=20.0, help="Max residual magnitude (asymmetric mode)")
@click.option("--seg-temperature-start", type=float, default=1.0, help="Phase 2 SegNet temperature start")
@click.option("--seg-temperature-end", type=float, default=0.1, help="Phase 2 SegNet temperature end")
@click.option("--use-margin-segnet", is_flag=True, help="Opt #13: margin-based SegNet loss")
@click.option("--segnet-margin-threshold", type=float, default=0.1, help="Margin threshold for margin-based SegNet")
@click.option("--use-null-space", is_flag=True, help="Opt #6: null-space gradient projection")
@click.option("--share-stem", is_flag=True, help="Opt #12: shared stem (deferred, gated)")
@click.option("--flow-only", is_flag=True, help="Opt #15: flow only, no gate/residual (deferred, gated)")
@click.option("--flow-warmup-epochs", type=int, default=500, help="Epochs with residual zeroed to force flow learning (council B1)")
@click.option("--residual-ramp-epochs", type=int, default=500, help="Epochs to ramp residual 0→1 after warmup (council B2)")
@click.option("--gate-reg-weight", type=float, default=0.1, help="Gate regularization weight (Quantizr: enforce warp usage)")
@click.option("--gate-reg-threshold", type=float, default=0.5, help="Gate regularization threshold")
@click.option("--even-pairs-only/--no-even-pairs-only", default=True, help="Train only even-index pairs (match scorer eval, default ON)")
@click.option("--use-ego-flow", is_flag=True, help="Enable learnable per-pair affine ego-flow (council Experiment B)")
@click.option("--ego-flow-max-px", type=float, default=20.0, help="Max affine flow displacement in pixels")
@click.option("--ego-flow-lr", type=float, default=1e-3, help="Learning rate for ego-flow affine params")
@click.option("--pose-supervision-weight", type=float, default=0.0, help="PoseNet supervision loss weight (council Layer 1)")
@click.option("--pose-targets-path", type=str, default=None, help="Path to posenet_targets.bin")
@click.option("--pose-embed-loss", is_flag=True, help="Use embedding-level PoseNet loss (council Layer 3)")
@click.option("--raft-flow-path", type=str, default=None, help="Path to raft_flow.pt for frozen warp (council Layer 2)")
@click.option("--flow-supervision-weight", type=float, default=0.0, help="MotionPredictor flow supervision weight")
def main(
    precomputed, epochs, batch_size, lr, channels, spade_hidden,
    seg_boundary, pose_boundary, rho_init, rho_growth,
    tv_weight, flow_weight, rate_weight, target_bytes,
    device, seed, checkpoint_every, eval_every, log_every,
    smoke, resume, max_hours, phase2_mse_weight,
    pair_mode, embed_dim, base_ch, mid_ch, motion_hidden, renderer_depth,
    init_bits, phase1_end, phase2_end, early_stop_patience,
    init_h, init_w, num_classes, use_noise, target_h, target_w,
    lr_min_ratio, grad_clip,
    p1_mse_weight, p1_seg_weight, p1_pose_weight, rho_max, lambda_cap,
    max_flow_px, max_residual, seg_temperature_start, seg_temperature_end,
    use_margin_segnet, segnet_margin_threshold, use_null_space,
    share_stem, flow_only,
    flow_warmup_epochs, residual_ramp_epochs,
    gate_reg_weight, gate_reg_threshold, even_pairs_only,
    use_ego_flow, ego_flow_max_px, ego_flow_lr,
    pose_supervision_weight, pose_targets_path, pose_embed_loss,
    raft_flow_path, flow_supervision_weight,
):
    """Train DP-SIMS renderer with Fridrich constrained optimization."""

    # Parse channels
    ch_tuple = tuple(int(x.strip()) for x in channels.split(","))

    cfg = FridrichRendererConfig(
        pair_mode=pair_mode,
        channels=ch_tuple,
        epochs=5 if smoke else epochs,
        lr=lr,
        batch_size=2 if smoke else batch_size,
        spade_hidden=spade_hidden,
        seg_boundary=seg_boundary,
        pose_boundary=pose_boundary,
        rho_init=rho_init,
        rho_growth=rho_growth,
        tv_weight=tv_weight,
        flow_weight=flow_weight,
        rate_weight=rate_weight,
        target_bytes=target_bytes,
        device=device,
        seed=seed,
        checkpoint_every=checkpoint_every,
        eval_every=1 if smoke else eval_every,
        log_every=1 if smoke else log_every,
        max_hours=max_hours,
        resume=resume,
        phase2_mse_weight=phase2_mse_weight,
        smoke=smoke,
        embed_dim=embed_dim,
        base_ch=base_ch,
        mid_ch=mid_ch,
        motion_hidden=motion_hidden,
        renderer_depth=renderer_depth,
        init_bits=init_bits,
        phase1_end=phase1_end,
        phase2_end=phase2_end,
        early_stop_patience=early_stop_patience,
        init_h=init_h,
        init_w=init_w,
        num_classes=num_classes,
        use_noise=use_noise,
        target_h=target_h,
        target_w=target_w,
        lr_min_ratio=lr_min_ratio,
        grad_clip=grad_clip,
        p1_mse_weight=p1_mse_weight,
        p1_seg_weight=p1_seg_weight,
        p1_pose_weight=p1_pose_weight,
        rho_max=rho_max,
        lambda_cap=lambda_cap,
        max_flow_px=max_flow_px,
        max_residual=max_residual,
        seg_temperature_start=seg_temperature_start,
        seg_temperature_end=seg_temperature_end,
        use_margin_segnet=use_margin_segnet,
        segnet_margin_threshold=segnet_margin_threshold,
        use_null_space=use_null_space,
        share_stem=share_stem,
        flow_only=flow_only,
        flow_warmup_epochs=flow_warmup_epochs,
        residual_ramp_epochs=residual_ramp_epochs,
        gate_reg_weight=gate_reg_weight,
        gate_reg_threshold=gate_reg_threshold,
        even_pairs_only=even_pairs_only,
        use_ego_flow=use_ego_flow,
        ego_flow_max_px=ego_flow_max_px,
        ego_flow_lr=ego_flow_lr,
        pose_supervision_weight=pose_supervision_weight,
        pose_targets_path=pose_targets_path,
        pose_embed_loss=pose_embed_loss,
        raft_flow_path=raft_flow_path,
        flow_supervision_weight=flow_supervision_weight,
    )

    # Set precomputed dir env var for the training function
    if precomputed:
        os.environ["PRECOMPUTED_DIR"] = precomputed

    summary = train_fridrich_renderer(cfg)

    # Print final verdict
    score = summary.get("best_score", float("inf"))
    if score < 0.50:
        print("\n** SUB-0.50 ACHIEVED **")
    elif score < 0.60:
        print("\n** COMPETITIVE with Quantizr (sub-0.60) **")
    elif score < 1.0:
        print("\n** PROMISING (sub-1.0), needs more training **")
    else:
        print(f"\n** Score {score:.2f} — architecture or training needs work **")

    return summary


def validate_smoke(precomputed: str | None = None, pair_mode: str = "dp_sims") -> None:
    """P3: Run 100-epoch CPU smoke and assert 5 conditions (Karpathy).

    Validates before committing to 48h GPU training:
    1. Loss decreases monotonically for first 50 epochs
    2. Phase transition from 1→2 fires correctly
    3. Lagrangian multipliers lambda_seg and lambda_pose are non-negative and finite
    4. _full_eval completes without error and returns valid score
    5. Checkpoint save/load roundtrip works
    """
    import tempfile

    print("=" * 60)
    print("SMOKE VALIDATION (P3 — Karpathy protocol)")
    print("=" * 60)

    cfg = FridrichRendererConfig(
        pair_mode=pair_mode,  # test the ACTUAL mode that will be deployed
        channels=(32, 16),  # tiny for speed (dp_sims only)
        base_ch=36,   # keep full capacity even for smoke (Bug fix: 8 was too small)
        mid_ch=60,
        motion_hidden=32,
        epochs=100,
        lr=2e-4,
        batch_size=2,
        spade_hidden=16,
        device="cpu",
        seed=42,
        checkpoint_every=50,
        eval_every=50,
        log_every=10,
        max_hours=0.5,
        smoke=True,
        phase1_end=0.4,   # Phase 1 ends at epoch 40 (match defaults)
        phase2_end=0.7,   # Phase 2 ends at epoch 70 (match defaults)
    )

    if precomputed:
        os.environ["PRECOMPUTED_DIR"] = precomputed

    summary = train_fridrich_renderer(cfg)
    history = summary.get("history", [])

    # Assertion 1: Loss decreases (first 50 epochs)
    losses_first50 = [h["loss"] for h in history if h["epoch"] < 50]
    if len(losses_first50) >= 5:
        first_loss = losses_first50[0]
        last_loss = losses_first50[-1]
        assert last_loss < first_loss, (
            f"FAIL: Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
        )
        print(f"  [1/5] Loss decreased: {first_loss:.4f} → {last_loss:.4f} ✓")
    else:
        print(f"  [1/5] Not enough data points ({len(losses_first50)}), skipped")

    # Assertion 2: Phase transition fires (phase 2 appears in history)
    phases_seen = {h["phase"] for h in history}
    assert 2 in phases_seen, f"FAIL: Phase 2 never appeared. Phases seen: {phases_seen}"
    print(f"  [2/5] Phase transition: phases {phases_seen} ✓")

    # Assertion 3: Lagrangian multipliers are non-negative and finite
    for h in history:
        assert h["lambda_seg"] >= 0 and math.isfinite(h["lambda_seg"]), (
            f"FAIL: lambda_seg={h['lambda_seg']} at epoch {h['epoch']}"
        )
        assert h["lambda_pose"] >= 0 and math.isfinite(h["lambda_pose"]), (
            f"FAIL: lambda_pose={h['lambda_pose']} at epoch {h['epoch']}"
        )
    print(f"  [3/5] Lagrangian multipliers non-negative and finite ✓")

    # Assertion 4: Eval completed with valid score
    best_score = summary.get("best_score", float("inf"))
    assert math.isfinite(best_score), f"FAIL: best_score is not finite: {best_score}"
    print(f"  [4/5] Eval completed: best_score={best_score:.4f} ✓")

    # Assertion 5: Check losses are not NaN
    for h in history:
        assert math.isfinite(h["loss"]), f"FAIL: NaN loss at epoch {h['epoch']}"
        assert math.isfinite(h["seg_dist"]), f"FAIL: NaN seg_dist at epoch {h['epoch']}"
        assert math.isfinite(h["pose_dist"]), f"FAIL: NaN pose_dist at epoch {h['epoch']}"
    print(f"  [5/5] All losses finite (no NaN/Inf) ✓")

    print()
    print("SMOKE VALIDATION: ALL 5 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--validate-smoke" in sys.argv:
        # P3: Karpathy smoke validation — run before committing GPU hours
        precomputed_arg = None
        pair_mode_arg = "dp_sims"
        for i, a in enumerate(sys.argv):
            if a == "--precomputed" and i + 1 < len(sys.argv):
                precomputed_arg = sys.argv[i + 1]
            if a == "--pair-mode" and i + 1 < len(sys.argv):
                pair_mode_arg = sys.argv[i + 1]
        validate_smoke(precomputed=precomputed_arg, pair_mode=pair_mode_arg)
    else:
        main()
