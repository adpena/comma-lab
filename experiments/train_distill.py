#!/usr/bin/env python3
"""Distillation training: teach the renderer to reproduce TTO frames in one forward pass.

This is THE path to contest-compliant sub-0.33. The v7 TTO achieved auth=0.37
with 500 gradient steps per pair at inflate time. Distillation eliminates that
cost by training the renderer to match TTO frames directly, then fine-tuning
with scorer feedback.

Phases:
    Phase 1 (pixel regression): L1 loss against TTO frames. Fast warm-start.
    Phase 2 (scorer-guided): Contest formula loss through frozen PoseNet + SegNet.
    Phase 3 (hard-pair fine-tuning): Weighted loss on hardest 20% of pairs.

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/train_distill.py --smoke --device mps

    # Full run (Vast.ai 4090):
    PYTHONPATH=src:upstream python experiments/train_distill.py \
        --device cuda \
        --tto-frames experiments/results/tto_v7_hinge_500/tto_frames.pt \
        --checkpoint experiments/results/v5_lagrangian_renderer/renderer_best.pt \
        --gt-poses experiments/results/gt_poses.pt

    # Resume from checkpoint:
    PYTHONPATH=src:upstream python experiments/train_distill.py \
        --device cuda --resume experiments/results/distillation/distill_latest.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path("/kaggle/working/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

RESULTS_DIR = (
    Path(os.environ["TAC_RESULTS_DIR"])
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "distillation"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DistillConfig:
    """All hyperparameters for distillation training."""

    def __post_init__(self):
        if not (0.0 <= self.hard_frame_ratio <= 1.0):
            raise ValueError(f"hard_frame_ratio must be in [0, 1], got {self.hard_frame_ratio}")
        if self.loss_mode not in ("standard", "pcgrad", "focal_ste"):
            raise ValueError(f"loss_mode must be standard/pcgrad/focal_ste, got {self.loss_mode!r}")

    # Architecture (must match checkpoint)
    embed_dim: int = 6
    base_ch: int = 36
    mid_ch: int = 60
    motion_hidden: int = 32
    depth: int = 1
    pose_dim: int = 6  # FiLM conditioning on pose vectors
    max_flow_px: float = 20.0
    max_residual: float = 20.0
    use_dsconv: bool = False  # Depthwise-separable convolutions (fewer params, wider channels)
    padding_mode: str = "replicate"  # Yousfi: zeros creates boundary artifacts
    use_dilation: bool = False  # Dilated ResBlocks [1,2,4] cascade (3x receptive field, 0 extra params)

    # Data paths
    tto_frames_path: str = "experiments/results/tto_v7_hinge_500/tto_frames.pt"
    checkpoint_path: str = "experiments/results/v5_lagrangian_renderer/renderer_best.pt"
    gt_poses_path: str = "experiments/results/gt_poses.pt"
    upstream_dir: str = "upstream/"

    # Training
    device: str = "cuda"
    seed: int = 42

    # Phase 1: pixel regression
    phase1_epochs: int = 2000
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 8  # pairs

    # Phase 2: scorer-guided fine-tuning
    phase2_epochs: int = 5000
    phase2_lr: float = 3e-4
    phase2_batch_size: int = 4  # scorer is VRAM-intensive
    phase2_pixel_weight: float = 0.1
    phase2_pose_weight: float = 10.0
    phase2_seg_weight: float = 100.0
    segnet_loss_mode: str = "hinge"  # "hinge" or "xent"
    hinge_margin: float = 0.5
    error_boost: float = 1.0  # Quantizr: 9.0 in anchor, 49.0 in anchor_boost
    error_boost_phase3: float = 1.0  # Quantizr anchor_boost: 49.0

    # SHIRAZ loss mode: selects scorer loss function for Phase 2+3
    loss_mode: str = "standard"  # "standard", "pcgrad", or "focal_ste"
    focal_gamma: float = 2.0  # focal loss gamma (only used when loss_mode="focal_ste")

    # SHIRAZ continuous curriculum: difficulty-weighted sampling in Phase 2
    hard_frame_ratio: float = 0.0  # 0.0 = uniform, 0.3 = weight 30% toward hard pairs
    error_replay_every: int = 0  # 0 = never, N = recompute difficulty weights every N epochs

    # Freeze/unfreeze (Quantizr 5-stage technique — SegNet/PoseNet orthogonal optimization)
    freeze_motion_phase2: bool = False  # Freeze MotionPredictor during SegNet training
    freeze_renderer_phase3: bool = False  # Freeze MaskRenderer during PoseNet training

    # Phase 3: hard-pair fine-tuning
    phase3_epochs: int = 2000
    phase3_lr: float = 1e-4
    phase3_batch_size: int = 4
    hard_pair_fraction: float = 0.2  # top 20% hardest pairs
    hard_pair_weight: float = 5.0  # weight multiplier for hard pairs

    # Eval-matched resize roundtrip
    eval_roundtrip: bool = True

    # Fridrich inverse steganalysis losses (Phase 2+3 only)
    use_texture_loss: bool = False       # UNIWARD: weight errors by texture complexity
    texture_loss_weight: float = 0.5     # weight for texture-aware L1
    use_linf_penalty: bool = False       # Square root law: penalize peak errors
    linf_weight: float = 0.01           # weight for L∞ penalty
    # use_boundary_hinge: REMOVED — was declared but never wired to any loss
    # computation. boundary_sensitive_hinge exists in fridrich_losses.py but
    # was never integrated. Removed to prevent dead-feature trap.
    use_markov_loss: bool = False        # HUGO: preserve local gradient statistics
    markov_weight: float = 0.1          # weight for Markov chain loss

    # Integrated QAT — train through FP4 quantization from the start
    # Post-hoc QAT degrades PoseNet 26x. Training through it adapts weights.
    qat_enabled: bool = False           # wrap model in FP4 fake-quant during training
    qat_from_phase2: bool = False       # legacy: start QAT at Phase 2 only (not Phase 1)
    fp4_block_size: int = 32            # FP4 block size

    # EMA (Quantizr uses decay=0.997, we have the class, never wired it)
    use_ema: bool = True              # EMA on by default — no reason not to
    ema_decay: float = 0.997

    # Optimizer
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Checkpointing and logging
    checkpoint_every: int = 500
    eval_every: int = 200
    log_every: int = 50

    # Export
    export_bits: int = 4  # FP4 for contest

    # Smoke test
    smoke: bool = False

    # Resume
    resume: str | None = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tto_frames(path: str, device: str, n_frames: int = 1200) -> torch.Tensor:
    """Load TTO target frames.

    Expected shape: (N, H, W, 3) float32 in [0, 255].
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        # Handle various save formats
        if "frames" in data:
            frames = data["frames"]
        elif "tto_frames" in data:
            frames = data["tto_frames"]
        else:
            raise KeyError(f"Cannot find frames in {path}. Keys: {list(data.keys())}")
    else:
        frames = data

    # Normalize shape to (N, H, W, 3)
    if frames.ndim == 4 and frames.shape[1] == 3:
        # (N, 3, H, W) -> (N, H, W, 3)
        frames = frames.permute(0, 2, 3, 1).contiguous()

    frames = frames[:n_frames].float()
    if frames.max() <= 1.0:
        frames = frames * 255.0

    print(f"  TTO frames loaded: {frames.shape}, range [{frames.min():.1f}, {frames.max():.1f}]")
    return frames


def load_gt_video(upstream_dir: str, n_frames: int = 1200) -> list[torch.Tensor]:
    """Load GT video frames from upstream."""
    import av

    video_path = Path(upstream_dir) / "videos" / "0.mkv"
    if not video_path.exists():
        raise FileNotFoundError(f"GT video not found: {video_path}")

    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i >= n_frames:
                break
            arr = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(arr))

    print(f"  GT video loaded: {len(frames)} frames, {frames[0].shape}")
    return frames


def load_masks(
    gt_frames: list[torch.Tensor],
    segnet: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Extract SegNet masks from GT frames."""
    from tac.scorer import extract_gt_masks

    masks = extract_gt_masks(gt_frames, segnet, device, batch_size=32)
    print(f"  Masks extracted: {masks.shape}, classes: {masks.unique().tolist()}")
    return masks


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_model(cfg: DistillConfig, device: torch.device) -> nn.Module:
    """Create AsymmetricPairGenerator with FiLM conditioning."""
    from tac.renderer import AsymmetricPairGenerator

    model = AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch,
        mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden,
        depth=cfg.depth,
        max_flow_px=cfg.max_flow_px,
        max_residual=cfg.max_residual,
        pose_dim=cfg.pose_dim,
        use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode,
        use_dilation=cfg.use_dilation,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created: {n_params:,} parameters "
          f"(padding={cfg.padding_mode}, dilation={cfg.use_dilation})")
    return model


def load_checkpoint_weights(model: nn.Module, path: str, device: torch.device) -> None:
    """Load renderer checkpoint, handling shape mismatches from FiLM layers."""
    state = torch.load(path, map_location=device, weights_only=True)

    # Handle wrapped checkpoints
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Filter out mismatched keys (FiLM layers won't exist in old checkpoints)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            compatible[k] = v
        else:
            skipped.append(k)

    # Also handle keys present in model but not in checkpoint (new FiLM layers)
    missing = [k for k in model_state if k not in compatible]

    model.load_state_dict(compatible, strict=False)
    print(f"  Checkpoint loaded: {len(compatible)} params, {len(skipped)} skipped, {len(missing)} new (FiLM)")


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------


def compute_pixel_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 pixel loss between predicted and target frame pairs.

    Args:
        predicted: (B, 2, H, W, 3) predicted pairs from renderer
        target: (B, 2, H, W, 3) TTO target pairs

    Returns:
        Scalar L1 loss averaged over all elements.
    """
    return F.l1_loss(predicted, target)


def compute_scorer_loss(
    predicted_pairs: torch.Tensor,
    gt_pairs: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks_even: torch.Tensor,
    masks_odd: torch.Tensor,
    cfg: DistillConfig,
    eval_roundtrip: bool = True,
    error_boost_override: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute contest-formula scorer loss with eval-matched roundtrip.

    Supports three loss modes (cfg.loss_mode):
      - "standard": hinge/xent SegNet + PoseNet MSE (original behavior)
      - "pcgrad": PCGrad non-opposing gradient from tac.losses (handles conflict)
      - "focal_ste": focal STE for SegNet + PoseNet from tac.losses

    Args:
        predicted_pairs: (B, 2, H, W, 3) predicted frame pairs
        gt_pairs: (B, 2, H, W, 3) GT frame pairs
        posenet: frozen differentiable PoseNet
        segnet: frozen SegNet
        masks_even: (B, H, W) target masks for even frames
        masks_odd: (B, H, W) target masks for odd frames
        cfg: training config
        eval_roundtrip: whether to apply simulate_eval_roundtrip

    Returns:
        (total_loss, metrics_dict)
    """
    from tac.renderer import simulate_eval_roundtrip

    B = predicted_pairs.shape[0]
    device = predicted_pairs.device

    # Flatten pairs to frames: (B, 2, H, W, 3) -> (2B, H, W, 3)
    pred_frames = predicted_pairs.reshape(B * 2, *predicted_pairs.shape[2:])
    gt_frames = gt_pairs.reshape(B * 2, *gt_pairs.shape[2:])

    # Apply eval-matched resize roundtrip (STE for gradients)
    if eval_roundtrip:
        from tac.camera import CAMERA_H, CAMERA_W

        pred_chw = pred_frames.permute(0, 3, 1, 2)  # (2B, 3, H, W)
        pred_chw = simulate_eval_roundtrip(pred_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.5)
        pred_frames_for_loss = pred_chw.permute(0, 2, 3, 1)  # (2B, H, W, 3)

        gt_chw = gt_frames.permute(0, 3, 1, 2)
        gt_chw = simulate_eval_roundtrip(gt_chw, target_h=CAMERA_H, target_w=CAMERA_W)
        gt_frames_for_loss = gt_chw.permute(0, 2, 3, 1)
    else:
        pred_frames_for_loss = pred_frames
        gt_frames_for_loss = gt_frames

    # ── Validate: Fridrich losses and error_boost only work with standard mode ──
    if cfg.loss_mode != "standard":
        fridrich_active = any([cfg.use_texture_loss, cfg.use_linf_penalty,
                               cfg.use_markov_loss])
        if fridrich_active:
            raise ValueError(
                f"Fridrich losses (texture/linf/markov/boundary) are only supported "
                f"with loss_mode='standard', got {cfg.loss_mode!r}. "
                f"PCGrad and focal_ste have their own gradient shaping."
            )
        boost = error_boost_override if error_boost_override is not None else cfg.error_boost
        if boost > 1.0:
            import warnings
            warnings.warn(
                f"error_boost={boost} has no effect with loss_mode={cfg.loss_mode!r} "
                f"(only applies to loss_mode='standard')",
                stacklevel=2,
            )

    # ── Dispatch on loss_mode ────────────────────────────────────────
    if cfg.loss_mode == "pcgrad":
        return _compute_pcgrad_loss(
            pred_frames_for_loss, gt_frames_for_loss, B, posenet, segnet, cfg,
        )
    elif cfg.loss_mode == "focal_ste":
        return _compute_focal_ste_loss(
            pred_frames_for_loss, gt_frames_for_loss, B, posenet, segnet, cfg,
        )

    # ── loss_mode="standard": original hinge/xent path ──────────────
    from tac.constrained_gen import compute_segnet_constraint_loss

    # SegNet loss (hinge or xent)
    # Interleave masks to match frame order: even, odd, even, odd, ...
    all_masks = torch.stack([masks_even, masks_odd], dim=1).reshape(B * 2, *masks_even.shape[1:])
    seg_loss = compute_segnet_constraint_loss(
        pred_frames_for_loss, all_masks, segnet,
        batch_size=B * 2,
        loss_mode=cfg.segnet_loss_mode,
        hinge_margin=cfg.hinge_margin,
        error_boost=error_boost_override if error_boost_override is not None else cfg.error_boost,
    )

    # PoseNet loss: pairs format (B, 2, C, H, W)
    pred_pairs_chw = pred_frames_for_loss.reshape(B, 2, *pred_frames_for_loss.shape[1:])
    pred_pairs_chw = pred_pairs_chw.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, 3, H, W)
    gt_pairs_chw = gt_frames_for_loss.reshape(B, 2, *gt_frames_for_loss.shape[1:])
    gt_pairs_chw = gt_pairs_chw.permute(0, 1, 4, 2, 3).contiguous()

    # PoseNet forward on predicted pairs
    posenet_in_pred = posenet.preprocess_input(pred_pairs_chw)
    posenet_out_pred = posenet(posenet_in_pred)
    pred_pose = posenet_out_pred["pose"][..., :6]

    # PoseNet forward on GT pairs (no grad needed)
    with torch.no_grad():
        posenet_in_gt = posenet.preprocess_input(gt_pairs_chw)
        posenet_out_gt = posenet(posenet_in_gt)
        gt_pose = posenet_out_gt["pose"][..., :6]

    pose_loss = (pred_pose - gt_pose).pow(2).mean()

    # Contest formula: 100 * seg + sqrt(10 * pose) + 25 * rate
    # Rate is fixed (archive size), so we optimize seg + pose only
    total = cfg.phase2_seg_weight * seg_loss + cfg.phase2_pose_weight * pose_loss

    # ── Fridrich inverse steganalysis losses (additive) ──────────────
    # These are gated by config flags and OFF by default.
    # Enable via --use-texture-loss, --use-linf-penalty, etc.
    fridrich_metrics: dict[str, float] = {}

    if cfg.use_texture_loss and cfg.texture_loss_weight > 0:
        from tac.fridrich_losses import texture_weighted_loss
        tex_loss = texture_weighted_loss(
            pred_frames_for_loss.permute(0, 3, 1, 2),
            gt_frames_for_loss.permute(0, 3, 1, 2),
        )
        total = total + cfg.texture_loss_weight * tex_loss
        fridrich_metrics["texture_loss"] = tex_loss.item()

    if cfg.use_linf_penalty and cfg.linf_weight > 0:
        from tac.fridrich_losses import linf_penalty
        linf = linf_penalty(pred_frames_for_loss, gt_frames_for_loss)
        total = total + cfg.linf_weight * linf
        fridrich_metrics["linf_penalty"] = linf.item()

    if cfg.use_markov_loss and cfg.markov_weight > 0:
        from tac.fridrich_losses import markov_chain_loss
        mc_loss = markov_chain_loss(
            pred_frames_for_loss.permute(0, 3, 1, 2),
            gt_frames_for_loss.permute(0, 3, 1, 2),
        )
        total = total + cfg.markov_weight * mc_loss
        fridrich_metrics["markov_loss"] = mc_loss.item()

    metrics = {
        "seg_loss": seg_loss.item(),
        "pose_loss": pose_loss.item(),
        "total_loss": total.item(),
        **fridrich_metrics,
    }
    return total, metrics


def _compute_pcgrad_loss(
    pred_frames_for_loss: torch.Tensor,
    gt_frames_for_loss: torch.Tensor,
    B: int,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    cfg: DistillConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PCGrad non-opposing gradient loss (SHIRAZ loss_mode="pcgrad").

    Reshapes roundtripped frames into (B, T=2, H, W, C) pairs for
    scorer_loss_pcgrad() which handles _hwc_to_chw conversion internally.
    """
    from tac.losses import scorer_loss_pcgrad

    # Reshape (2B, H, W, 3) -> (B, 2, H, W, 3) for scorer_loss_pcgrad
    pred_pairs_hwc = pred_frames_for_loss.reshape(B, 2, *pred_frames_for_loss.shape[1:])
    gt_pairs_hwc = gt_frames_for_loss.reshape(B, 2, *gt_frames_for_loss.shape[1:])

    loss, pose_dist, seg_dist, conflict = scorer_loss_pcgrad(
        pred_pairs_hwc,
        gt_pairs_hwc,
        posenet,
        segnet,
        segnet_weight=cfg.phase2_seg_weight,
        do_projection=True,
    )

    metrics = {
        "seg_loss": seg_dist,
        "pose_loss": pose_dist,
        "total_loss": loss.item(),
        "pcgrad_conflict": float(conflict),
    }
    return loss, metrics


def _compute_focal_ste_loss(
    pred_frames_for_loss: torch.Tensor,
    gt_frames_for_loss: torch.Tensor,
    B: int,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    cfg: DistillConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Focal STE loss (SHIRAZ loss_mode="focal_ste").

    Reshapes roundtripped frames into (B, T=2, H, W, C) pairs for
    focal_segnet_ste_loss() which handles _hwc_to_chw conversion internally.
    The focal loss already applies the contest scoring formula (100*seg + sqrt(10*pose)).
    """
    from tac.losses import focal_segnet_ste_loss

    # Reshape (2B, H, W, 3) -> (B, 2, H, W, 3) for focal_segnet_ste_loss
    pred_pairs_hwc = pred_frames_for_loss.reshape(B, 2, *pred_frames_for_loss.shape[1:])
    gt_pairs_hwc = gt_frames_for_loss.reshape(B, 2, *gt_frames_for_loss.shape[1:])

    loss, pose_dist, seg_dist = focal_segnet_ste_loss(
        pred_pairs_hwc,
        gt_pairs_hwc,
        posenet,
        segnet,
        gamma=cfg.focal_gamma,
    )

    metrics = {
        "seg_loss": seg_dist,
        "pose_loss": pose_dist,
        "total_loss": loss.item(),
    }
    return loss, metrics


def train_phase1(
    model: nn.Module,
    masks: torch.Tensor,
    tto_frames: torch.Tensor,
    poses: torch.Tensor | None,
    cfg: DistillConfig,
    device: torch.device,
    start_epoch: int = 0,
    ema: object | None = None,
) -> int:
    """Phase 1: pixel regression warm-start.

    Returns the epoch number after completion.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Pixel Regression Warm-Start")
    print("=" * 70)

    # Integrated QAT from Phase 1 (full training through FP4)
    if cfg.qat_enabled and not cfg.qat_from_phase2:
        from tac.fp4_quantize import QATRendererFP4
        qat_p1 = QATRendererFP4(model, block_size=cfg.fp4_block_size)
        train_model = qat_p1
        print(f"  QAT from Phase 1: FP4 fake-quant on {len(qat_p1._parametrized_modules)} layers")
    else:
        train_model = model

    train_model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.phase1_epochs, eta_min=cfg.phase1_lr * 0.01
    )

    n_pairs = masks.shape[0] // 2
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + cfg.phase1_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle pair indices
        pair_indices = torch.randperm(n_pairs, device="cpu")

        for batch_start in range(0, n_pairs, cfg.phase1_batch_size):
            batch_end = min(batch_start + cfg.phase1_batch_size, n_pairs)
            batch_idx = pair_indices[batch_start:batch_end]

            # Get masks for this batch of pairs
            even_idx = batch_idx * 2
            odd_idx = batch_idx * 2 + 1
            mask_even = masks[even_idx].to(device)
            mask_odd = masks[odd_idx].to(device)

            # Get TTO target frames
            tto_even = tto_frames[even_idx].to(device)
            tto_odd = tto_frames[odd_idx].to(device)
            target_pairs = torch.stack([tto_even, tto_odd], dim=1)  # (B, 2, H, W, 3)

            # Get pose conditioning
            pose = poses[batch_idx].to(device) if poses is not None else None

            # Forward pass — use train_model (has QAT wrapper if enabled)
            predicted_pairs = train_model(mask_even, mask_odd, pose=pose)  # (B, 2, H, W, 3)

            # L1 loss
            loss = compute_pixel_loss(predicted_pairs, target_pairs)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch - start_epoch) % cfg.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Phase1] epoch {epoch:5d} | loss {avg_loss:.4f} | "
                f"best {best_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
            )

        if (epoch - start_epoch) % cfg.checkpoint_every == 0 and epoch > start_epoch:
            _save_checkpoint(model, optimizer, epoch, "phase1", cfg, ema=ema)

    _save_checkpoint(model, optimizer, start_epoch + cfg.phase1_epochs, "phase1_final", cfg, ema=ema)
    print(f"  Phase 1 complete. Best L1: {best_loss:.4f}")
    return start_epoch + cfg.phase1_epochs


def train_phase2(
    model: nn.Module,
    masks: torch.Tensor,
    tto_frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    cfg: DistillConfig,
    device: torch.device,
    start_epoch: int = 0,
    ema: object | None = None,
) -> tuple[int, torch.Tensor, nn.Module]:
    """Phase 2: scorer-guided fine-tuning.

    Returns (final_epoch, per_pair_difficulty).
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Scorer-Guided Fine-Tuning")
    print(f"  loss_mode={cfg.loss_mode}" + (f" (gamma={cfg.focal_gamma})" if cfg.loss_mode == "focal_ste" else ""))
    print("=" * 70)

    # Integrated QAT: wrap model with FP4 fake quantization BEFORE training
    # This trains the weights to be robust to FP4 noise from the start,
    # instead of post-hoc QAT which degrades PoseNet 26x on large models.
    qat_wrapper = None
    if cfg.qat_enabled or cfg.qat_from_phase2:
        from tac.fp4_quantize import QATRendererFP4
        qat_wrapper = QATRendererFP4(model, block_size=cfg.fp4_block_size)
        train_model = qat_wrapper  # forward through fake-quant weights
        import torch.nn.utils.parametrize as _pm
        n_quant = sum(1 for m in model.modules()
                      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding))
                      and _pm.is_parametrized(m, "weight"))
        n_new = len(qat_wrapper._parametrized_modules)
        print(f"  Integrated QAT: {n_quant} FP4-parametrized layers ({n_new} new, block_size={cfg.fp4_block_size})")
    else:
        train_model = model

    train_model.train()

    # Freeze/unfreeze: Quantizr's key SegNet technique.
    # Freeze MotionPredictor during Phase 2 so ALL gradient goes to MaskRenderer.
    # SegNet sees only frame_t1 (rendered by MaskRenderer). Freezing motion
    # ensures zero cross-contamination between SegNet and PoseNet objectives.
    if cfg.freeze_motion_phase2 and hasattr(model, 'motion'):
        for p in model.motion.parameters():
            p.requires_grad_(False)
        n_frozen = sum(1 for p in model.motion.parameters() if not p.requires_grad)
        n_total = sum(1 for p in model.motion.parameters())
        print(f"  Freeze: MotionPredictor frozen ({n_frozen}/{n_total} params)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.phase2_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.phase2_epochs, eta_min=cfg.phase2_lr * 0.01
    )

    n_pairs = masks.shape[0] // 2
    n_frames = masks.shape[0]
    best_score = float("inf")
    per_pair_loss = torch.zeros(n_pairs)
    t0 = time.time()

    # SHIRAZ continuous curriculum: difficulty-weighted sampling
    # difficulty_weights is a probability distribution over pairs. Updated every
    # error_replay_every epochs. When hard_frame_ratio=0, falls back to uniform.
    difficulty_weights: torch.Tensor | None = None
    if cfg.hard_frame_ratio > 0:
        # Initialize uniform until first replay computes real difficulties
        difficulty_weights = torch.ones(n_pairs) / n_pairs
        print(f"  SHIRAZ curriculum: hard_frame_ratio={cfg.hard_frame_ratio}, "
              f"error_replay_every={cfg.error_replay_every}")

    # Prepare GT frames as tensors at scorer resolution
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    gt_tensor = torch.stack(gt_frames[:n_frames]).float()  # (N, H, W, 3)
    # Resize GT to scorer resolution if needed
    if gt_tensor.shape[1] != SEGNET_INPUT_H or gt_tensor.shape[2] != SEGNET_INPUT_W:
        gt_chw = gt_tensor.permute(0, 3, 1, 2)
        gt_chw = F.interpolate(gt_chw, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
        gt_tensor = gt_chw.permute(0, 2, 3, 1).contiguous()

    for epoch in range(start_epoch, start_epoch + cfg.phase2_epochs):
        epoch_loss = 0.0
        epoch_seg = 0.0
        epoch_pose = 0.0
        epoch_conflicts = 0  # PCGrad conflict count
        n_batches = 0

        # SHIRAZ: recompute difficulty weights periodically
        if (difficulty_weights is not None
                and cfg.error_replay_every > 0
                and (epoch - start_epoch) > 0
                and (epoch - start_epoch) % cfg.error_replay_every == 0):
            # Use per_pair_loss EMA as difficulty signal
            raw = per_pair_loss.clamp(min=1e-8)
            # Blend: hard_frame_ratio of weight goes to difficulty, rest uniform
            uniform = torch.ones_like(raw) / n_pairs
            normed = raw / raw.sum()
            difficulty_weights = (1.0 - cfg.hard_frame_ratio) * uniform + cfg.hard_frame_ratio * normed
            print(f"  [Curriculum] epoch {epoch}: recomputed difficulty weights "
                  f"(top={difficulty_weights.max():.4f}, bot={difficulty_weights.min():.4f})")

        # Sample pair indices: weighted or uniform
        if difficulty_weights is not None and cfg.hard_frame_ratio > 0:
            # Weighted sampling without replacement for one epoch
            # replacement=True allows hard pairs to appear multiple times per epoch
            # (Yousfi audit: replacement=False makes curriculum effectively uniform)
            # Floor weights at 0.5/n to prevent complete starvation of easy pairs
            floored_weights = difficulty_weights.clamp(min=0.5 / n_pairs)
            floored_weights = floored_weights / floored_weights.sum()
            pair_indices = torch.multinomial(floored_weights, n_pairs, replacement=True)
        else:
            pair_indices = torch.randperm(n_pairs, device="cpu")

        for batch_start in range(0, n_pairs, cfg.phase2_batch_size):
            batch_end = min(batch_start + cfg.phase2_batch_size, n_pairs)
            batch_idx = pair_indices[batch_start:batch_end]

            even_idx = batch_idx * 2
            odd_idx = batch_idx * 2 + 1
            mask_even = masks[even_idx].to(device)
            mask_odd = masks[odd_idx].to(device)

            # GT pairs for scorer comparison
            gt_even = gt_tensor[even_idx].to(device)
            gt_odd = gt_tensor[odd_idx].to(device)
            gt_pairs = torch.stack([gt_even, gt_odd], dim=1)

            # Pose conditioning
            pose = poses[batch_idx].to(device) if poses is not None else None

            # Forward pass — use train_model (has QAT wrapper if enabled)
            predicted_pairs = train_model(mask_even, mask_odd, pose=pose)

            # Pixel loss (anchor toward TTO quality)
            tto_even = tto_frames[even_idx].to(device)
            tto_odd = tto_frames[odd_idx].to(device)
            tto_pairs = torch.stack([tto_even, tto_odd], dim=1)
            pixel_loss = compute_pixel_loss(predicted_pairs, tto_pairs)

            # Scorer loss
            scorer_loss, metrics = compute_scorer_loss(
                predicted_pairs, gt_pairs, posenet, segnet,
                mask_even, mask_odd, cfg, eval_roundtrip=cfg.eval_roundtrip,
            )

            # Combined loss
            total = cfg.phase2_pixel_weight * pixel_loss + scorer_loss

            optimizer.zero_grad()
            total.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            epoch_loss += total.item()
            epoch_seg += metrics["seg_loss"]
            epoch_pose += metrics["pose_loss"]
            if metrics.get("pcgrad_conflict", 0.0) > 0:
                epoch_conflicts += 1
            n_batches += 1

            # Track per-pair difficulty via EMA using FULL score contribution
            # (Yousfi audit: PoseNet-only was wrong — SegNet is 77x more important)
            # difficulty = 100*seg_dist + sqrt(10*pose_dist) per the contest formula
            with torch.no_grad():
                pred_chw = predicted_pairs.reshape(-1, *predicted_pairs.shape[2:]).permute(0, 3, 1, 2)
                gt_chw = gt_pairs.reshape(-1, *gt_pairs.shape[2:]).permute(0, 3, 1, 2)
                B_pairs = len(batch_idx)
                # PoseNet: per-pair MSE on 6D pose
                pred_pose_p = pred_chw.reshape(B_pairs, 2, *pred_chw.shape[1:])
                gt_pose_p = gt_chw.reshape(B_pairs, 2, *gt_chw.shape[1:])
                p_in = posenet.preprocess_input(pred_pose_p)
                g_in = posenet.preprocess_input(gt_pose_p)
                p_out = posenet(p_in)["pose"][..., :6]
                g_out = posenet(g_in)["pose"][..., :6]
                per_pair_pose = (p_out - g_out).pow(2).mean(dim=1)  # (B,)
                # SegNet: per-pair argmax disagreement on frame_t1 (odd frame)
                odd_pred = pred_chw.reshape(B_pairs, 2, *pred_chw.shape[1:])[:, 1]  # (B, 3, H, W)
                odd_gt = gt_chw.reshape(B_pairs, 2, *gt_chw.shape[1:])[:, 1]
                seg_in_p = segnet.preprocess_input(odd_pred.unsqueeze(1))
                seg_in_g = segnet.preprocess_input(odd_gt.unsqueeze(1))
                seg_p = segnet(seg_in_p).argmax(dim=1)
                seg_g = segnet(seg_in_g).argmax(dim=1)
                per_pair_seg = (seg_p != seg_g).float().mean(dim=(1, 2))  # (B,)
                # Contest formula: 100*seg + sqrt(10*pose)
                per_pair_difficulty = 100 * per_pair_seg + (10 * per_pair_pose).sqrt()
                for k, idx in enumerate(batch_idx):
                    per_pair_loss[idx.item()] = 0.9 * per_pair_loss[idx.item()] + 0.1 * per_pair_difficulty[k].item()

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_seg = epoch_seg / max(n_batches, 1)
        avg_pose = epoch_pose / max(n_batches, 1)

        # Proxy score estimate
        proxy_score = 100 * avg_seg + math.sqrt(max(10 * avg_pose, 0))
        if proxy_score < best_score:
            best_score = proxy_score
            _save_checkpoint(model, optimizer, epoch, "phase2_best", cfg, ema=ema)

        if (epoch - start_epoch) % cfg.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            extra = ""
            if cfg.loss_mode == "pcgrad":
                extra = f" | conflicts {epoch_conflicts}/{n_batches}"
            print(
                f"  [Phase2] epoch {epoch:5d} | loss {avg_loss:.4f} | "
                f"seg {avg_seg:.5f} | pose {avg_pose:.5f} | "
                f"proxy ~{proxy_score:.3f} | best {best_score:.3f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s{extra}"
            )

        if (epoch - start_epoch) % cfg.checkpoint_every == 0 and epoch > start_epoch:
            _save_checkpoint(model, optimizer, epoch, "phase2", cfg, ema=ema)

    _save_checkpoint(model, optimizer, start_epoch + cfg.phase2_epochs, "phase2_final", cfg, ema=ema)
    print(f"  Phase 2 complete. Best proxy: {best_score:.3f}")
    return start_epoch + cfg.phase2_epochs, per_pair_loss, train_model


def train_phase3(
    model: nn.Module,
    masks: torch.Tensor,
    tto_frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    per_pair_difficulty: torch.Tensor,
    cfg: DistillConfig,
    device: torch.device,
    start_epoch: int = 0,
    train_model: nn.Module | None = None,
    ema: object | None = None,
) -> int:
    """Phase 3: hard-pair fine-tuning.

    Trains on the hardest pairs with increased weight.
    train_model is the QAT wrapper from Phase 2 (or model itself if no QAT).
    Returns the final epoch number.
    """
    if train_model is None:
        train_model = model

    print("\n" + "=" * 70)
    print("PHASE 3: Hard-Pair Fine-Tuning")
    print("=" * 70)

    # Reuse the QAT wrapper from Phase 2 (passed via train_model_p2 in main)
    # Do NOT create a new QATRendererFP4 — that would produce an empty
    # _parametrized_modules list since modules are already parametrized.
    # The train_model variable is set by the caller (main) before this call.

    train_model.train()

    # Freeze/unfreeze: Phase 3 can freeze renderer for pure PoseNet optimization.
    # Unfreeze MotionPredictor if it was frozen in Phase 2.
    if cfg.freeze_motion_phase2 and hasattr(model, 'motion'):
        for p in model.motion.parameters():
            p.requires_grad_(True)
        print("  Unfreeze: MotionPredictor unfrozen for Phase 3")

    if cfg.freeze_renderer_phase3 and hasattr(model, 'renderer'):
        for p in model.renderer.parameters():
            p.requires_grad_(False)
        n_frozen = sum(1 for p in model.renderer.parameters() if not p.requires_grad)
        print(f"  Freeze: MaskRenderer frozen ({n_frozen} params) — pure PoseNet optimization")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.phase3_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.phase3_epochs, eta_min=cfg.phase3_lr * 0.1
    )

    n_pairs = masks.shape[0] // 2
    n_frames = masks.shape[0]

    # Identify hard pairs (top fraction by difficulty)
    n_hard = max(1, int(n_pairs * cfg.hard_pair_fraction))
    hard_indices = per_pair_difficulty.topk(n_hard).indices
    print(f"  Hard pairs: {n_hard} (top {cfg.hard_pair_fraction*100:.0f}%)")
    print(f"  Difficulty range: {per_pair_difficulty[hard_indices].min():.4f} - "
          f"{per_pair_difficulty[hard_indices].max():.4f}")

    # Prepare GT frames
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    gt_tensor = torch.stack(gt_frames[:n_frames]).float()
    if gt_tensor.shape[1] != SEGNET_INPUT_H or gt_tensor.shape[2] != SEGNET_INPUT_W:
        gt_chw = gt_tensor.permute(0, 3, 1, 2)
        gt_chw = F.interpolate(gt_chw, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
        gt_tensor = gt_chw.permute(0, 2, 3, 1).contiguous()

    best_score = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + cfg.phase3_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Sample: 50% hard pairs, 50% random (prevent catastrophic forgetting)
        n_hard_batch = max(1, cfg.phase3_batch_size // 2)
        n_easy_batch = cfg.phase3_batch_size - n_hard_batch

        hard_perm = hard_indices[torch.randperm(len(hard_indices))]
        easy_perm = torch.randperm(n_pairs)

        for batch_start in range(0, n_hard, n_hard_batch):
            batch_hard = hard_perm[batch_start:min(batch_start + n_hard_batch, len(hard_perm))]
            # Random easy pairs (wrap around if needed)
            easy_start = (batch_start * n_easy_batch // max(n_hard_batch, 1)) % n_pairs
            easy_end = min(easy_start + n_easy_batch, n_pairs)
            batch_easy = easy_perm[easy_start:easy_end]
            batch_idx = torch.cat([batch_hard, batch_easy])

            even_idx = batch_idx * 2
            odd_idx = batch_idx * 2 + 1
            mask_even = masks[even_idx].to(device)
            mask_odd = masks[odd_idx].to(device)

            gt_even = gt_tensor[even_idx].to(device)
            gt_odd = gt_tensor[odd_idx].to(device)
            gt_pairs = torch.stack([gt_even, gt_odd], dim=1)

            pose = poses[batch_idx].to(device) if poses is not None else None

            predicted_pairs = train_model(mask_even, mask_odd, pose=pose)

            # Scorer loss (Phase 3 uses error_boost_phase3 for extreme hard mining)
            scorer_loss, metrics = compute_scorer_loss(
                predicted_pairs, gt_pairs, posenet, segnet,
                mask_even, mask_odd, cfg, eval_roundtrip=cfg.eval_roundtrip,
                error_boost_override=cfg.error_boost_phase3,
            )

            # Weight hard pairs more
            # The first n_hard_batch pairs in the batch are hard
            # Simple approach: multiply total loss by weight factor
            # (all pairs in batch get weighted since hard pairs dominate gradients)
            weight = (cfg.hard_pair_weight + 1.0) / 2.0  # average of hard_weight and 1.0
            total = weight * scorer_loss

            optimizer.zero_grad()
            total.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            epoch_loss += total.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_score:
            best_score = avg_loss
            _save_checkpoint(model, optimizer, epoch, "phase3_best", cfg, ema=ema)

        if (epoch - start_epoch) % cfg.log_every == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Phase3] epoch {epoch:5d} | loss {avg_loss:.4f} | "
                f"best {best_score:.4f} | lr {lr:.2e} | {elapsed:.0f}s"
            )

        if (epoch - start_epoch) % cfg.checkpoint_every == 0 and epoch > start_epoch:
            _save_checkpoint(model, optimizer, epoch, "phase3", cfg, ema=ema)

    _save_checkpoint(model, optimizer, start_epoch + cfg.phase3_epochs, "phase3_final", cfg, ema=ema)
    print(f"  Phase 3 complete. Best loss: {best_score:.4f}")
    return start_epoch + cfg.phase3_epochs


# ---------------------------------------------------------------------------
# Checkpointing and export
# ---------------------------------------------------------------------------


def _clean_state_dict(state_dict: dict) -> dict:
    """Strip nn.utils.parametrize keys to plain weight keys for clean resume.

    When QAT is active, state_dict has keys like
    'layer.parametrizations.weight.original' instead of 'layer.weight'.
    This makes load_state_dict crash on a bare model. Strip them.
    """
    clean = {}
    for k, v in state_dict.items():
        if ".parametrizations.weight.original" in k:
            clean[k.replace(".parametrizations.weight.original", ".weight")] = v
        elif ".parametrizations." in k:
            continue  # skip codebook buffers
        else:
            clean[k] = v
    return clean


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    tag: str,
    cfg: DistillConfig,
    ema: object | None = None,
) -> None:
    """Save training checkpoint with clean (non-parametrized) state dict.
    If EMA is provided, saves EMA weights as the model state (better for eval)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"distill_{tag}.pt"
    # Use EMA weights if available (smoother, better for eval/export)
    state = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(
        {
            "model_state_dict": _clean_state_dict(state),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "tag": tag,
            "config": asdict(cfg),
        },
        path,
    )
    # Also save as "latest" for easy resume (copy, not second torch.save)
    import shutil
    latest_path = RESULTS_DIR / "distill_latest.pt"
    shutil.copy2(path, latest_path)


def export_model(model: nn.Module, cfg: DistillConfig) -> int:
    """Export model to ASYM .bin format for inflate_renderer.py.

    Returns archive size in bytes.
    """
    from tac.renderer_export import export_asymmetric_checkpoint

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "renderer_distilled.bin"
    nbytes = export_asymmetric_checkpoint(model, output_path, default_bits=cfg.export_bits)
    print(f"  Exported: {output_path} ({nbytes:,} bytes)")
    return nbytes


def run_proxy_eval(
    model: nn.Module,
    masks: torch.Tensor,
    gt_frames: list[torch.Tensor],
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    n_frames: int = 1200,
) -> dict[str, float]:
    """Run proxy evaluation on the full frame set."""
    from tac.scorer import compute_proxy_score

    import torch.nn.utils.parametrize as parametrize_mod
    is_qat = any(parametrize_mod.is_parametrized(m, "weight")
                 for m in model.modules()
                 if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)))
    model.eval()
    n_pairs = min(n_frames // 2, masks.shape[0] // 2)
    mode_tag = "[QAT-FP4]" if is_qat else "[float]"
    print(f"  Proxy eval {mode_tag}: {n_pairs} pairs")

    # Generate all frames
    all_frames = []
    with torch.no_grad():
        for i in range(0, n_pairs, 8):
            end = min(i + 8, n_pairs)
            even_idx = torch.arange(i * 2, end * 2, 2)
            odd_idx = torch.arange(i * 2 + 1, end * 2 + 1, 2)
            mask_even = masks[even_idx].to(device)
            mask_odd = masks[odd_idx].to(device)
            pose = poses[i:end].to(device) if poses is not None else None

            pairs = model(mask_even, mask_odd, pose=pose)  # (B, 2, H, W, 3)
            # Unpack pairs to frames
            for j in range(pairs.shape[0]):
                all_frames.append(pairs[j, 0].cpu())
                all_frames.append(pairs[j, 1].cpu())

    frames_tensor = torch.stack(all_frames[:n_frames])
    result = compute_proxy_score(
        frames_tensor, gt_frames[:n_frames],
        posenet, segnet, device,
        rate=0.0, eval_roundtrip=True,
    )
    model.train()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distillation: train renderer to reproduce TTO frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data paths
    p.add_argument("--tto-frames", type=str, default="experiments/results/tto_v7_hinge_500/tto_frames.pt")
    p.add_argument("--masks", type=str, default=None,
                   help="Pre-encoded masks (.mkv). Train with THESE masks instead of fresh SegNet. "
                        "CRITICAL: must match the masks that will be in archive.zip.")
    p.add_argument("--checkpoint", type=str, default="experiments/results/v5_lagrangian_renderer/renderer_best.pt")
    p.add_argument("--gt-poses", type=str, default="experiments/results/gt_poses.pt")
    p.add_argument("--upstream", type=str, default="upstream/")
    p.add_argument("--output-dir", type=str, default=None)

    # Architecture
    p.add_argument("--pose-dim", type=int, default=6)
    p.add_argument("--base-ch", type=int, default=36)
    p.add_argument("--mid-ch", type=int, default=60)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--use-dsconv", action="store_true",
                   help="Use depthwise-separable convolutions (fewer params, wider channels)")
    p.add_argument("--motion-hidden", type=int, default=32,
                   help="MotionPredictor hidden channels (wilde: 16, default: 32)")
    p.add_argument("--padding-mode", type=str, default="replicate",
                   choices=["zeros", "reflect", "replicate", "circular"],
                   help="Conv padding mode (Yousfi: zeros creates boundary artifacts)")
    p.add_argument("--use-dilation", action="store_true",
                   help="Dilated ResBlocks [1,2,4] cascade (3x receptive field, 0 extra params)")

    # Training
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-roundtrip", action="store_true", default=True)
    p.add_argument("--no-eval-roundtrip", action="store_true")
    p.add_argument("--segnet-loss-mode", type=str, default="hinge", choices=["hinge", "xent"])
    p.add_argument("--hinge-margin", type=float, default=0.5)
    p.add_argument("--error-boost", type=float, default=1.0,
                   help="Per-pixel error magnification for Phase 2 (Quantizr: 9.0)")
    p.add_argument("--error-boost-phase3", type=float, default=1.0,
                   help="Error boost for Phase 3 (Quantizr anchor_boost: 49.0)")

    # SHIRAZ loss mode
    p.add_argument("--loss-mode", type=str, default="standard",
                   choices=["standard", "pcgrad", "focal_ste"],
                   help="Scorer loss function: standard (hinge/xent), pcgrad (gradient conflict), focal_ste (focal STE)")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma for loss_mode=focal_ste (higher = more focus on hard pixels)")

    # SHIRAZ continuous curriculum
    p.add_argument("--hard-frame-ratio", type=float, default=0.0,
                   help="Difficulty-weighted sampling ratio in Phase 2 (0.0=uniform, 0.3=30%% hard bias)")
    p.add_argument("--error-replay-every", type=int, default=0,
                   help="Recompute difficulty weights every N epochs (0=never)")

    p.add_argument("--freeze-motion-phase2", action="store_true",
                   help="Freeze MotionPredictor during Phase 2 (pure SegNet optimization)")
    p.add_argument("--freeze-renderer-phase3", action="store_true",
                   help="Freeze MaskRenderer during Phase 3 (pure PoseNet optimization)")

    # Integrated QAT
    p.add_argument("--no-ema", action="store_true",
                   help="Disable EMA (on by default, Quantizr uses decay=0.997)")
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--qat", action="store_true", dest="qat_enabled",
                   help="Train with FP4 fake-quant from Phase 1 (full QAT)")
    p.add_argument("--qat-from-phase2", action="store_true",
                   help="Start FP4 fake-quant at Phase 2 only (not Phase 1)")
    p.add_argument("--fp4-block-size", type=int, default=32)

    # Fridrich inverse steganalysis losses
    p.add_argument("--use-texture-loss", action="store_true",
                   help="UNIWARD: weight errors by texture complexity (Holub & Fridrich 2014)")
    p.add_argument("--texture-loss-weight", type=float, default=0.5)
    p.add_argument("--use-linf-penalty", action="store_true",
                   help="Square root law: penalize peak errors (Ker, Filler & Fridrich 2008)")
    p.add_argument("--linf-weight", type=float, default=0.01)
    p.add_argument("--use-markov-loss", action="store_true",
                   help="HUGO: preserve local gradient statistics (Filler, Judas & Fridrich 2010)")
    p.add_argument("--markov-weight", type=float, default=0.1)

    # Phase config
    p.add_argument("--phase1-epochs", type=int, default=2000)
    p.add_argument("--phase2-epochs", type=int, default=5000)
    p.add_argument("--phase3-epochs", type=int, default=2000)
    p.add_argument("--phase1-lr", type=float, default=1e-3)
    p.add_argument("--phase2-lr", type=float, default=3e-4)
    p.add_argument("--phase3-lr", type=float, default=1e-4)
    p.add_argument("--phase1-batch-size", type=int, default=8)
    p.add_argument("--phase2-batch-size", type=int, default=4)
    p.add_argument("--phase3-batch-size", type=int, default=4)

    # Scorer weights
    p.add_argument("--seg-weight", type=float, default=100.0)
    p.add_argument("--pose-weight", type=float, default=10.0)
    p.add_argument("--pixel-weight", type=float, default=0.1)

    # Control
    p.add_argument("--smoke", action="store_true", help="Smoke test: tiny config")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 (resume from Phase 2)")
    p.add_argument("--only-phase1", action="store_true", help="Run only Phase 1")

    # Logging
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build config from args
    cfg = DistillConfig(
        tto_frames_path=args.tto_frames,
        checkpoint_path=args.checkpoint,
        gt_poses_path=args.gt_poses,
        upstream_dir=args.upstream,
        device=args.device,
        seed=args.seed,
        pose_dim=args.pose_dim,
        base_ch=args.base_ch,
        mid_ch=args.mid_ch,
        depth=args.depth,
        use_dsconv=args.use_dsconv,
        motion_hidden=args.motion_hidden,
        padding_mode=args.padding_mode,
        use_dilation=args.use_dilation,
        eval_roundtrip=args.eval_roundtrip and not args.no_eval_roundtrip,
        segnet_loss_mode=args.segnet_loss_mode,
        hinge_margin=args.hinge_margin,
        error_boost=args.error_boost,
        error_boost_phase3=args.error_boost_phase3,
        loss_mode=args.loss_mode,
        focal_gamma=args.focal_gamma,
        hard_frame_ratio=args.hard_frame_ratio,
        error_replay_every=args.error_replay_every,
        freeze_motion_phase2=args.freeze_motion_phase2,
        freeze_renderer_phase3=args.freeze_renderer_phase3,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        phase3_lr=args.phase3_lr,
        phase1_batch_size=args.phase1_batch_size,
        phase2_batch_size=args.phase2_batch_size,
        phase3_batch_size=args.phase3_batch_size,
        phase2_seg_weight=args.seg_weight,
        phase2_pose_weight=args.pose_weight,
        phase2_pixel_weight=args.pixel_weight,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        resume=args.resume,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        qat_enabled=args.qat_enabled,
        qat_from_phase2=args.qat_from_phase2,
        fp4_block_size=args.fp4_block_size,
        use_texture_loss=args.use_texture_loss,
        texture_loss_weight=args.texture_loss_weight,
        use_linf_penalty=args.use_linf_penalty,
        linf_weight=args.linf_weight,
        use_markov_loss=args.use_markov_loss,
        markov_weight=args.markov_weight,
    )

    # Smoke test overrides
    if args.smoke:
        cfg.phase1_epochs = 5
        cfg.phase2_epochs = 5
        cfg.phase3_epochs = 5
        cfg.phase1_batch_size = 2
        cfg.phase2_batch_size = 2
        cfg.phase3_batch_size = 2
        cfg.log_every = 1
        cfg.checkpoint_every = 3
        cfg.eval_every = 3
        cfg.smoke = True

    # Output directory
    if args.output_dir:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 70)
    print("DISTILLATION TRAINING")
    print("=" * 70)
    print(f"  Device: {cfg.device}")
    print(f"  Loss mode: {cfg.loss_mode}" + (f" (gamma={cfg.focal_gamma})" if cfg.loss_mode == "focal_ste" else ""))
    print(f"  Phases: {cfg.phase1_epochs} + {cfg.phase2_epochs} + {cfg.phase3_epochs} epochs")
    if cfg.hard_frame_ratio > 0:
        print(f"  Curriculum: hard_frame_ratio={cfg.hard_frame_ratio}, replay_every={cfg.error_replay_every}")
    print(f"  TTO frames: {cfg.tto_frames_path}")
    print(f"  Checkpoint: {cfg.checkpoint_path}")
    print(f"  Output: {RESULTS_DIR}")
    print()

    # Setup
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Determine number of frames
    n_frames = 1200
    if cfg.smoke:
        n_frames = 20
        print("  [SMOKE TEST] Using 20 frames only")

    # Load TTO frames
    print("Loading TTO target frames...")
    tto_frames = load_tto_frames(cfg.tto_frames_path, cfg.device, n_frames)

    # Load GT poses (optional but recommended for FiLM)
    poses = None
    if cfg.pose_dim > 0 and Path(cfg.gt_poses_path).exists():
        print("Loading GT poses for FiLM conditioning...")
        poses_data = torch.load(cfg.gt_poses_path, map_location="cpu", weights_only=True)
        if isinstance(poses_data, dict):
            poses = poses_data.get("poses", poses_data.get("gt_poses", None))
        else:
            poses = poses_data
        if poses is not None:
            n_pairs = n_frames // 2
            poses = poses[:n_pairs].float()
            print(f"  GT poses loaded: {poses.shape}")
    elif cfg.pose_dim > 0:
        print(f"  WARNING: pose_dim={cfg.pose_dim} but gt_poses not found at {cfg.gt_poses_path}")
        print("  FiLM conditioning will use zero vectors (no benefit)")

    # Load scorers (needed for Phase 2+3 and eval)
    print("Loading scorers...")
    upstream_dir = cfg.upstream_dir
    if UPSTREAM_ROOT is not None:
        upstream_dir = str(UPSTREAM_ROOT)
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)
    print("  Scorers loaded (differentiable PoseNet + frozen SegNet)")

    # Load GT video for scorer eval
    print("Loading GT video...")
    gt_frames = load_gt_video(upstream_dir, n_frames)

    # Extract masks
    if args.masks:
        # Use pre-encoded masks (MUST match archive masks)
        masks_path = Path(args.masks)
        if masks_path.suffix in (".mkv", ".mp4"):
            import subprocess, numpy as np
            cmd = ["ffmpeg", "-v", "quiet", "-i", str(masks_path),
                   "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
            proc = subprocess.run(cmd, capture_output=True)
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", str(masks_path)],
                capture_output=True, text=True)
            w, h = map(int, probe.stdout.strip().split(","))
            pixels = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w)
            scale = 255 // 4
            masks = torch.from_numpy(
                np.clip(np.round(pixels.astype(np.float32) / scale).astype(np.int64), 0, 4)
            ).long()
        else:
            masks = torch.load(str(masks_path), weights_only=True)
        print(f"Loaded archive masks from {args.masks}: {masks.shape}")
    else:
        print("Extracting SegNet masks from GT...")
        masks = load_masks(gt_frames, segnet, device)

    # Resize TTO frames to match mask resolution if needed
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W
    if tto_frames.shape[1] != SEGNET_INPUT_H or tto_frames.shape[2] != SEGNET_INPUT_W:
        print(f"  Resizing TTO frames from {tto_frames.shape[1:3]} to ({SEGNET_INPUT_H}, {SEGNET_INPUT_W})...")
        tto_chw = tto_frames.permute(0, 3, 1, 2)
        tto_chw = F.interpolate(tto_chw, size=(SEGNET_INPUT_H, SEGNET_INPUT_W), mode="bilinear", align_corners=False)
        tto_frames = tto_chw.permute(0, 2, 3, 1).contiguous()

    # Create model
    print("Creating model...")
    model = create_model(cfg, device)

    # Load pretrained weights
    if cfg.resume:
        print(f"Resuming from {cfg.resume}...")
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        resume_tag = ckpt.get("tag", "unknown")
        print(f"  Resumed from epoch {start_epoch}, tag={resume_tag}")
    elif Path(cfg.checkpoint_path).exists():
        print(f"Loading pretrained weights from {cfg.checkpoint_path}...")
        load_checkpoint_weights(model, cfg.checkpoint_path, device)
        start_epoch = 0
    else:
        print("  No checkpoint found, training from scratch")
        start_epoch = 0

    # ── EMA ────────────────────────────────────────────────────────────
    ema = None
    if cfg.use_ema:
        from tac.training import EMA
        ema = EMA(model, decay=cfg.ema_decay)
        print(f"  EMA enabled (decay={cfg.ema_decay})")

    # ── Training phases ───────────────────────────────────────────────
    current_epoch = start_epoch
    per_pair_difficulty = None

    # Phase 1: pixel regression
    # Skip if --skip-phase1 OR if resuming from any checkpoint (resume = continue from where we left off)
    if not args.skip_phase1 and not cfg.resume:
        current_epoch = train_phase1(
            model, masks, tto_frames, poses, cfg, device, start_epoch=current_epoch, ema=ema,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.only_phase1:
        print("\n  --only-phase1 set, stopping after Phase 1.")
        if ema is not None:
            ema.apply(model)
        export_model(model, cfg)
        return

    # Phase 2: scorer-guided (returns train_model which is QAT wrapper if enabled)
    current_epoch, per_pair_difficulty, train_model_p2 = train_phase2(
        model, masks, tto_frames, gt_frames, poses,
        posenet, segnet, cfg, device, start_epoch=current_epoch, ema=ema,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Proxy eval after Phase 2
    print("\nRunning proxy evaluation after Phase 2...")
    result = run_proxy_eval(model, masks, gt_frames, poses, posenet, segnet, device, n_frames)
    print(f"  Proxy score: {result.get('score', 'N/A')}")
    print(f"  PoseNet: {result.get('pose', 'N/A'):.5f}, SegNet: {result.get('seg', 'N/A'):.5f}")

    # Phase 3: hard-pair fine-tuning (pass QAT wrapper from Phase 2)
    if per_pair_difficulty is not None:
        current_epoch = train_phase3(
            model, masks, tto_frames, gt_frames, poses,
            posenet, segnet, per_pair_difficulty, cfg, device,
            start_epoch=current_epoch,
            train_model=train_model_p2,
            ema=ema,
        )

    # Load BEST checkpoint (EMA weights from best scorer epoch) for final eval/export
    best_candidates = ["distill_phase3_best.pt", "distill_phase2_best.pt", "distill_latest.pt"]
    for bc in best_candidates:
        best_path = RESULTS_DIR / bc
        if best_path.exists():
            print(f"\nLoading best checkpoint: {bc}")
            best_ckpt = torch.load(str(best_path), map_location=device, weights_only=True)
            model.load_state_dict(best_ckpt["model_state_dict"])
            break

    # Final eval
    print("\nRunning final proxy evaluation (best checkpoint)...")
    result = run_proxy_eval(model, masks, gt_frames, poses, posenet, segnet, device, n_frames)
    print(f"  FINAL proxy score: {result.get('score', 'N/A')}")
    print(f"  PoseNet: {result.get('pose', 'N/A'):.5f}, SegNet: {result.get('seg', 'N/A'):.5f}")

    # Export
    print("\nExporting model...")
    archive_bytes = export_model(model, cfg)
    uncompressed = n_frames * SEGNET_INPUT_H * SEGNET_INPUT_W * 3
    rate = archive_bytes / uncompressed
    print(f"  Rate: {rate:.6f} (archive {archive_bytes:,} / uncompressed {uncompressed:,})")

    # Save final results
    final_results = {
        "proxy_score": result.get("score"),
        "pose_dist": result.get("pose"),
        "seg_dist": result.get("seg"),
        "rate": rate,
        "archive_bytes": archive_bytes,
        "total_epochs": current_epoch,
        "config": asdict(cfg),
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 70)
    print("DISTILLATION COMPLETE")
    print(f"  Results: {RESULTS_DIR / 'results.json'}")
    print(f"  Best checkpoint: {RESULTS_DIR / 'distill_phase3_best.pt'}")
    print(f"  Export: {RESULTS_DIR / 'renderer_distilled.bin'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
