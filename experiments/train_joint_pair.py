#!/usr/bin/env python
"""Joint Pair Generator Training — mask-conditioned frame pair generation.

Council-approved architecture: Y-shaped U-Net that takes (mask1, mask2) and
produces (frame1, frame2). This is the paradigm Quantizr uses at 0.60 with
386KB archive.

Training signal comes directly from the frozen scorers:
  - SegNet: cross-entropy on logits vs GT masks (last frame only, matching scorer)
  - PoseNet: MSE on pose output[:6] between generated and GT pairs
  - TV: total variation for compressibility
  - Rate: FP4 model size penalty

Score = 100 * seg + sqrt(10 * pose) + 25 * rate

Usage:
    PYTHONPATH=src:upstream python experiments/train_joint_pair.py \\
        --epochs 5000 --device cuda

    # Smoke test (fast, CPU):
    PYTHONPATH=src:upstream python experiments/train_joint_pair.py --smoke
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
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
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path("/kaggle/working/upstream"),
    Path("/home/zeus/content/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

_CANDIDATE_WEIGHTS = [
    Path(os.environ["TAC_MODELS_DIR"]) if os.environ.get("TAC_MODELS_DIR") else None,
    Path("/kaggle/working/upstream/models"),
    Path("/home/zeus/content/upstream/models"),
    Path("/home/zeus/content/pact/upstream/models"),
    Path(__file__).resolve().parent.parent / "upstream" / "models",
]
WEIGHTS_DIR: Path | None = None
for _p in _CANDIDATE_WEIGHTS:
    if _p is not None and _p.exists() and (_p / "posenet.safetensors").exists():
        WEIGHTS_DIR = _p
        break

RESULTS_DIR = (
    Path(os.environ["TAC_RESULTS_DIR"])
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "joint_pair"
)

# Original uncompressed video size in bytes (used for rate calculation)
ORIGINAL_UNCOMPRESSED_SIZE = 37_545_489


# Tier-1 operator-required CLI flags manifest. Catalog #151
# (`check_operator_wrapper_threads_trainer_tier_required_flags`) refuses any
# wrapper that invokes this trainer without threading the env-var ladder for
# each flag below. Per OD-WIRE-3 grand council 2026-05-12 verdict (6/10 PROCEED,
# INVOKED-ONLY at TIER_1).
#
# This Click-based trainer exposes NO `--enable-*` semantic gates (the
# OD-WIRE-3 pattern that catches "landed-but-not-wired" feature flags per the
# design memo `.omx/research/design_trainer_flag_manifest_for_wireup_and_composition_20260512.md`).
# Click flag-pairs (--eval-roundtrip/--no-eval-roundtrip, --fp4-qat/--no-fp4-qat)
# are CLAUDE.md-referenced but default to the safe side and have settled wrapper
# conventions; canonical non-negotiables are enforced trainer-internally
# AND by sister checks (Catalog #5 / #7 / #88).
TIER_1_OPERATOR_REQUIRED_FLAGS = {}  # No operator-tier semantic gates declared.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class JointPairConfig:
    """All hyperparameters for joint pair generator training."""

    # Architecture
    num_classes: int = 5
    base_ch: int = 40       # 731K params, 357KB FP4 — Quantizr-scale capacity

    # Training
    epochs: int = 5000
    lr: float = 2e-4
    lr_min_ratio: float = 0.01
    batch_size: int = 8     # pairs per step (600 total pairs, so ~75 steps/epoch)
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42

    # Scorer resolution (384x512 matches scorer input pipeline)
    target_h: int = 384
    target_w: int = 512

    # Loss weights — Lagrangian-style with council R2 caps
    seg_weight: float = 100.0    # matches score formula: 100 * seg
    pose_weight: float = 1.0     # PoseNet MSE (scaled by sqrt(10*x) in score)
    tv_weight: float = 0.1       # TV smoothness for compressibility
    rate_weight: float = 25.0    # matches score formula: 25 * rate

    # PoseNet supervision (direct targets when available)
    pose_targets_path: str | None = None  # path to posenet_targets.bin

    # Phase boundaries
    warmup_epochs: int = 200     # MSE-only warmup before scorer losses
    warmup_mse_weight: float = 10.0  # MSE weight during warmup

    # Self-compression (Phase 3)
    fp4_qat: bool = True               # enable FP4 QAT from the start

    # Checkpointing
    checkpoint_every: int = 500
    eval_every: int = 200
    log_every: int = 50

    # Wall-clock budget (hours, 0 = no limit)
    max_hours: float = 48.0

    # Auto-kill thresholds
    kill_loss_threshold: float = 1e5
    kill_patience: int = 200

    # EMA — Council D 2026-04-29 PM fix: 0.9995 → 0.997 (Quantizr canonical).
    # The previous default was Polyak-style (200-epoch retention 0.9995^200
    # ≈ 0.905) and out of band with the Quantizr 0.997 default the rest
    # of the pipeline uses (200-epoch retention 0.997^200 ≈ 0.548 — i.e.
    # ours was 1.65× MORE frozen than the leader's). The audit at
    # .omx/research/council_ema_audit_20260429.md §3.6 prescribed this
    # change; CLAUDE.md "EMA — NON-NEGOTIABLE" makes 0.997 the default
    # weight EMA decay everywhere (codebook EMAs, e.g. LCT N_c/m_c, keep
    # their van den Oord 0.99 default).
    ema_decay: float = 0.997

    # Resume from checkpoint
    resume: str | None = None

    # eval_roundtrip: simulate contest eval resize chain
    eval_roundtrip: bool = True

    # Smoke test
    smoke: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    """Load frozen PoseNet and SegNet."""
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
    differentiable version.
    """
    import einops
    import types

    # Patch AllNorm to not break gradients.
    # The upstream AllNorm uses .view() which fails on non-contiguous tensors.
    # On MPS, BN1d backward also uses .view() internally — we replace BN1d
    # with manual mean/var + affine when on MPS to avoid this PyTorch bug.
    is_mps = any(p.device.type == "mps" for p in posenet.parameters())
    for module in list(posenet.modules()) + list(segnet.modules()):
        if type(module).__name__ == "AllNorm":
            if is_mps:
                def _patched_forward_mps(self, x):
                    shape = x.shape
                    flat = x.reshape(-1, 1).contiguous()
                    if self.bn.training:
                        mean = flat.mean(dim=0, keepdim=True)
                        var = flat.var(dim=0, unbiased=False, keepdim=True)
                    else:
                        mean = self.bn.running_mean.unsqueeze(0)
                        var = self.bn.running_var.unsqueeze(0)
                    out = (flat - mean) / (var + self.bn.eps).sqrt()
                    if self.bn.affine:
                        out = out * self.bn.weight.unsqueeze(0) + self.bn.bias.unsqueeze(0)
                    return out.reshape(shape)
                module.forward = types.MethodType(_patched_forward_mps, module)
            else:
                def _patched_forward(self, x):
                    flat = x.reshape(-1, 1).contiguous()
                    return self.bn(flat).reshape(x.shape)
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
            if mask.shape[1] != gt_chw.shape[2] or mask.shape[2] != gt_chw.shape[3]:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=(gt_chw.shape[2], gt_chw.shape[3]),
                    mode="nearest",
                ).squeeze(1).long()
            masks_list.append(mask.to(torch.int8))
    return torch.cat(masks_list, dim=0)


def _tv_loss(frames: torch.Tensor) -> torch.Tensor:
    """Total variation loss for compressibility.

    Args:
        frames: (B, 3, H, W) or (B, 2, 3, H, W) float tensor.
    """
    if frames.ndim == 5:
        # (B, 2, 3, H, W) → flatten to (B*2, 3, H, W)
        frames = frames.reshape(-1, *frames.shape[2:])
    dx = (frames[:, :, :, 1:] - frames[:, :, :, :-1]).abs().mean()
    dy = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs().mean()
    return dx + dy


def _compute_seg_loss_ce(
    gen_frames: torch.Tensor,
    gt_masks: torch.Tensor,
    segnet: nn.Module,
) -> torch.Tensor:
    """SegNet cross-entropy loss: CE(SegNet(generated), GT_mask).

    This directly optimizes the segmentation accuracy of generated frames
    against the ground truth masks extracted from GT video.

    Args:
        gen_frames: (B, 3, H, W) float [0, 255] generated frames.
        gt_masks: (B, H, W) long tensor of GT class indices [0, 4].
        segnet: frozen SegNet.

    Returns:
        Scalar cross-entropy loss.
    """
    gen_btchw = gen_frames.unsqueeze(1).contiguous()  # (B, 1, C, H, W)
    seg_in = segnet.preprocess_input(gen_btchw)
    seg_logits = segnet(seg_in)  # (B, 5, H', W')

    # Resize GT masks to match SegNet output resolution
    if gt_masks.shape[1] != seg_logits.shape[2] or gt_masks.shape[2] != seg_logits.shape[3]:
        gt_resized = F.interpolate(
            gt_masks.unsqueeze(1).float(),
            size=(seg_logits.shape[2], seg_logits.shape[3]),
            mode="nearest",
        ).squeeze(1).long()
    else:
        gt_resized = gt_masks.long()

    return F.cross_entropy(seg_logits, gt_resized)


def _compute_seg_distortion_hard(
    gen_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: nn.Module,
) -> float:
    """Hard SegNet disagreement for eval (non-differentiable)."""
    with torch.no_grad():
        gen_btchw = gen_frames.unsqueeze(1).contiguous()
        gt_btchw = gt_frames.unsqueeze(1).contiguous()
        seg_in_g = segnet.preprocess_input(gen_btchw)
        seg_in_gt = segnet.preprocess_input(gt_btchw)
        logits_g = segnet(seg_in_g)
        logits_gt = segnet(seg_in_gt)
        disagree = (logits_g.argmax(dim=1) != logits_gt.argmax(dim=1)).float().mean()
    return disagree.item()


def _compute_pose_distortion_coupled(
    gen_frame_t: torch.Tensor,
    gen_frame_t1: torch.Tensor,
    gt_frame_t: torch.Tensor,
    gt_frame_t1: torch.Tensor,
    posenet: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PoseNet distortion with COUPLED trajectory through both frames.

    PoseNet evaluates consecutive PAIRS. Gradient flows through BOTH frames.

    Returns:
        (pose_dist, pred) where pred is the raw pose output for optional supervision.
    """
    gen_pair = torch.stack([gen_frame_t, gen_frame_t1], dim=1).contiguous()
    gt_pair = torch.stack([gt_frame_t, gt_frame_t1], dim=1).contiguous()

    pose_in_gen = posenet.preprocess_input(gen_pair)
    pose_out_gen = posenet(pose_in_gen)

    with torch.no_grad():
        pose_in_gt = posenet.preprocess_input(gt_pair)
        pose_out_gt = posenet(pose_in_gt)

    pred = pose_out_gen["pose"] if isinstance(pose_out_gen, dict) else pose_out_gen
    target = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt

    # Bounds assertion (council round 18)
    assert pred.shape[-1] >= 6, f"PoseNet output too small: {pred.shape}"
    pose_dist = (pred[..., :6] - target[..., :6]).pow(2).mean()
    return pose_dist, pred


def _compute_pose_supervision_loss(
    pred_pose: torch.Tensor,
    pose_targets: torch.Tensor,
) -> torch.Tensor:
    """Direct MSE on PoseNet output[:6] vs precomputed GT pose targets.

    Args:
        pred_pose: (B, 12) PoseNet output from generated pairs.
        pose_targets: (B, 6) precomputed GT pose for these pairs.
    """
    return (pred_pose[..., :6] - pose_targets).pow(2).mean()


# ---------------------------------------------------------------------------
# Full evaluation (no gradients)
# ---------------------------------------------------------------------------

def _full_eval(
    model: nn.Module,
    masks: torch.Tensor,
    gt_chw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, float]:
    """Evaluate joint pair generator with official scoring formula."""
    model.eval()
    n = masks.shape[0]
    seg_dists = []
    pose_dists = []

    # Move data to device once at start (H4 fix — avoid per-batch CPU->GPU transfers)
    masks_on_dev = masks.to(device=device)
    gt_on_dev = gt_chw.to(device=device)

    with torch.no_grad():
        # Even-index pairs: (0,1), (2,3), ...
        pair_starts = list(range(0, n - 1, 2))
        if not pair_starts:
            model.train()
            return {"avg_seg": float("inf"), "avg_pose": float("inf"),
                    "score": float("inf"), "rate": 0.0, "model_bytes": 0.0}

        for batch_start in range(0, len(pair_starts), batch_size):
            batch_indices = pair_starts[batch_start:batch_start + batch_size]
            B = len(batch_indices)

            m_t = torch.stack([masks_on_dev[j] for j in batch_indices]).long()
            m_t1 = torch.stack([masks_on_dev[j + 1] for j in batch_indices]).long()

            # Generate pair via model
            pair_hwc = model(m_t, m_t1)  # (B, 2, H, W, 3)
            gen_t = pair_hwc[:, 0].permute(0, 3, 1, 2).contiguous()   # (B, 3, H, W)
            gen_t1 = pair_hwc[:, 1].permute(0, 3, 1, 2).contiguous()

            gt_t = torch.stack([gt_on_dev[j] for j in batch_indices])
            gt_t1 = torch.stack([gt_on_dev[j + 1] for j in batch_indices])

            # SegNet: last frame only (upstream uses x[:, -1, ...])
            gen_btchw = gen_t1.unsqueeze(1).contiguous()
            gt_btchw = gt_t1.unsqueeze(1).contiguous()
            seg_in_g = segnet.preprocess_input(gen_btchw)
            seg_in_gt = segnet.preprocess_input(gt_btchw)
            seg_logits_gen = segnet(seg_in_g)
            seg_logits_gt = segnet(seg_in_gt)
            hard_seg = (seg_logits_gen.argmax(dim=1) != seg_logits_gt.argmax(dim=1)).float()
            for b in range(B):
                seg_dists.append(hard_seg[b].mean().item())

            # PoseNet: consecutive pairs
            gen_pair = torch.stack([gen_t, gen_t1], dim=1).contiguous()
            gt_pair = torch.stack([gt_t, gt_t1], dim=1).contiguous()
            pose_in_g = posenet.preprocess_input(gen_pair)
            pose_in_gt = posenet.preprocess_input(gt_pair)
            p_out = posenet(pose_in_g)
            g_out = posenet(pose_in_gt)
            pm = p_out["pose"] if isinstance(p_out, dict) else p_out
            gm = g_out["pose"] if isinstance(g_out, dict) else g_out
            pose_mse = (pm[..., :6] - gm[..., :6]).pow(2).mean(dim=-1)
            for b in range(B):
                pose_dists.append(pose_mse[b].mean().item())

    # Single empty_cache after entire eval, not per-batch (M3 fix)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.train()

    avg_seg = sum(seg_dists) / max(len(seg_dists), 1)
    avg_pose = sum(pose_dists) / max(len(pose_dists), 1)

    # Model size: FP4 estimate
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = n_params * 4 / 8  # 4 bits per param
    rate = model_bytes / ORIGINAL_UNCOMPRESSED_SIZE

    from tac.scorer import comma_score
    score = comma_score(avg_pose, avg_seg, rate)

    return {
        "avg_seg": avg_seg,
        "avg_pose": avg_pose,
        "model_bytes": model_bytes,
        "rate": rate,
        "score": score,
        "n_pairs": len(seg_dists),
    }


# ---------------------------------------------------------------------------
# EMA — Council D 2026-04-29 PM fix: import canonical class from
# tac.training instead of locally redefining (the local class diverged
# from the canonical one in two ways: (a) lacked the float-buffer guard
# at L356-358 of training.py for late-bound modules added AFTER EMA
# construction; (b) used decay=0.9995 default deviating from Quantizr's
# 0.997 — see .omx/research/council_ema_audit_20260429.md §3.6).
# ---------------------------------------------------------------------------

from tac.training import EMA  # noqa: E402


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_joint_pair(cfg: JointPairConfig) -> dict[str, Any]:
    """Train the joint pair generator."""

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JOINT PAIR GENERATOR TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"base_ch: {cfg.base_ch}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size} pairs")
    print(f"Loss weights: seg={cfg.seg_weight}, pose={cfg.pose_weight}, "
          f"tv={cfg.tv_weight}, rate={cfg.rate_weight}")
    print()

    # ---- Load scorers ----
    print("[1/5] Loading scorers...")
    posenet, segnet = _load_scorers(cfg.device)
    _patch_scorers_for_training(posenet, segnet)
    print(f"  Scorers loaded, VRAM: {_vram_mb():.0f} MB")

    # ---- Load data ----
    print("[2/5] Loading data...")
    from tac.data import load_precomputed, decode_video

    gt_frames_hwc: list[torch.Tensor] | None = None

    precomputed_dir = os.environ.get("PRECOMPUTED_DIR")
    if precomputed_dir and Path(precomputed_dir).exists():
        _, gt_list = load_precomputed(precomputed_dir)
        gt_frames_hwc = gt_list
        print(f"  Loaded {len(gt_frames_hwc)} GT frames from precomputed")
    else:
        gt_candidates = [
            Path(os.environ["TAC_UPSTREAM_DIR"]) / "videos" / "0.mkv" if os.environ.get("TAC_UPSTREAM_DIR") else None,
            Path("/kaggle/working/upstream/videos/0.mkv"),
            Path("/kaggle/input/datasets/adpena/comma-lab-private-assets/0.mkv"),
            Path("/home/zeus/content/upstream/videos/0.mkv"),
            Path(__file__).resolve().parent.parent / "upstream" / "videos" / "0.mkv",
        ]
        for gp in gt_candidates:
            if gp is not None and gp.exists():
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
    if gt_chw.shape[2] != cfg.target_h or gt_chw.shape[3] != cfg.target_w:
        gt_chw = F.interpolate(
            gt_chw,
            size=(cfg.target_h, cfg.target_w),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 255)
    print(f"  GT shape: {gt_chw.shape}")

    # ---- Extract masks from GT via SegNet ----
    print("[3/5] Extracting masks from GT...")
    gt_chw_dev = gt_chw.to(device)
    masks = _extract_masks(gt_chw_dev, segnet, batch_size=8)
    masks = masks.cpu()
    gt_chw_dev = gt_chw_dev.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Masks shape: {masks.shape}, classes: {masks.unique().tolist()}")
    print(f"  VRAM after mask extraction: {_vram_mb():.0f} MB")

    # ---- Build pair indices ----
    # Even-pairs: (0,1), (2,3), (4,5), ... — matches scorer evaluation
    pair_indices = list(range(0, n_frames - 1, 2))
    n_pairs = len(pair_indices)
    print(f"  Training pairs: {n_pairs}")

    # ---- Load precomputed PoseNet targets (optional) ----
    pose_targets_all: torch.Tensor | None = None
    if cfg.pose_targets_path and Path(cfg.pose_targets_path).exists():
        pose_targets_all = torch.load(cfg.pose_targets_path, map_location="cpu", weights_only=True)
        print(f"  Loaded pose targets: {pose_targets_all.shape}")
    else:
        # Precompute PoseNet targets from GT pairs
        print("  Precomputing PoseNet targets from GT...")
        targets_list = []
        with torch.no_grad():
            for pi in pair_indices:
                gt_t = gt_chw[pi:pi + 1].to(device)
                gt_t1 = gt_chw[pi + 1:pi + 2].to(device)
                gt_pair = torch.stack([gt_t, gt_t1], dim=1).contiguous()
                pose_in = posenet.preprocess_input(gt_pair)
                pose_out = posenet(pose_in)
                p = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
                targets_list.append(p[..., :6].cpu())
        pose_targets_all = torch.cat(targets_list, dim=0)  # (n_pairs, 6)
        # Save for reuse
        targets_path = results_dir / "posenet_targets.bin"
        torch.save(pose_targets_all, targets_path)
        print(f"  Saved pose targets to {targets_path}: {pose_targets_all.shape}")

    # ---- Build model ----
    print("[4/5] Building JointPairGenerator...")
    from tac.joint_pair_generator import JointPairGenerator, count_params

    model = JointPairGenerator(
        num_classes=cfg.num_classes,
        base_ch=cfg.base_ch,
    ).to(device)

    n_params = count_params(model)
    fp4_kb = n_params * 4 / 8 / 1024
    print(f"  Params: {n_params:,}")
    print(f"  FP4 estimate: {fp4_kb:.0f} KB")
    print(f"  VRAM: {_vram_mb():.0f} MB")

    # FP4 QAT wrapper (optional)
    if cfg.fp4_qat:
        from tac.fp4_quantize import QATRendererFP4
        qat_model = QATRendererFP4(model)
        print(f"  FP4 QAT enabled ({len(qat_model._parametrized_modules)} layers parametrized)")
    else:
        qat_model = None

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * cfg.lr_min_ratio,
    )

    # ---- EMA ----
    ema = EMA(model, decay=cfg.ema_decay)

    # ---- Training state ----
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    start_epoch = 0
    start_time = time.time()
    history: list[dict] = []
    diverge_count = 0

    # ---- Resume from checkpoint (C2 fix) ----
    if cfg.resume:
        ckpt_path = Path(cfg.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"  Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load model weights
        model.load_state_dict(ckpt["model_state_dict"])
        # Load optimizer state
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Load scheduler state
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # Restore epoch counter
        start_epoch = ckpt.get("epoch", 0) + 1
        # Restore best score
        best_score = ckpt.get("best_score", ckpt.get("score", float("inf")))
        # Restore EMA state
        if "ema_state_dict" in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}
            print("  EMA state restored")
        # Re-wrap QAT if it was active (parametrizations must be re-registered
        # after loading state_dict since load_state_dict replaces weight data)
        if cfg.fp4_qat and qat_model is not None:
            qat_model.remove_hooks()
            from tac.fp4_quantize import QATRendererFP4 as _QAT
            qat_model = _QAT(model)
            print("  FP4 QAT re-registered after resume")
        print(f"  Resumed at epoch {start_epoch}, best_score={best_score:.4f}")

    # Move all GT data to device for speed (fits in VRAM for 600 pairs at 384x512)
    gt_chw_dev = gt_chw.to(device)
    masks_dev = masks.to(device)

    # Precompute param count (constant for fixed architecture) — H3 fix
    n_model_params = sum(p.numel() for p in model.parameters())

    # VRAM warning (M4 fix) — warn if estimated usage exceeds T4 safe limit
    if device.type == "cuda":
        vram_now = _vram_mb()
        if vram_now > 14000:
            print(f"  WARNING: Current VRAM usage ({vram_now:.0f} MB) exceeds T4 safe "
                  f"limit (14 GB). Consider enabling gradient checkpointing or "
                  f"reducing batch_size.")

    print()
    print("[5/5] Training...")
    print("=" * 70)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()

        # Wall-clock budget
        if cfg.max_hours > 0:
            elapsed_h = (time.time() - start_time) / 3600
            if elapsed_h > cfg.max_hours:
                print(f"\n  WALL-CLOCK BUDGET REACHED ({elapsed_h:.1f}h > {cfg.max_hours}h)")
                break

        # Shuffle pair order each epoch
        perm = torch.randperm(n_pairs)
        epoch_seg_loss = 0.0
        epoch_pose_loss = 0.0
        epoch_tv_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_total_loss = 0.0
        n_steps = 0

        for batch_start in range(0, n_pairs, cfg.batch_size):
            batch_perm = perm[batch_start:batch_start + cfg.batch_size]
            B = len(batch_perm)
            if B == 0:
                continue

            # Gather masks and GT for this batch
            idx_t = torch.tensor([pair_indices[p] for p in batch_perm])
            idx_t1 = idx_t + 1

            m_t = masks_dev[idx_t].long()     # (B, H, W)
            m_t1 = masks_dev[idx_t1].long()   # (B, H, W)
            gt_t = gt_chw_dev[idx_t]          # (B, 3, H, W)
            gt_t1 = gt_chw_dev[idx_t1]        # (B, 3, H, W)

            # Forward: generate frame pair
            pair_hwc = model(m_t, m_t1)  # (B, 2, H, W, 3)
            # .reshape avoids view/stride issues that break MPS autograd
            pair_chw = pair_hwc.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, 3, H, W)
            gen_t = pair_chw[:, 0].contiguous()   # (B, 3, H, W)
            gen_t1 = pair_chw[:, 1].contiguous()  # (B, 3, H, W)

            # --- Phase logic ---
            # Warmup phase overlap (M5): During warmup, MSE is the primary signal.
            # After warmup_epochs//2, scorer losses (seg, pose) are blended in at
            # reduced weight to avoid a sharp transition. After warmup_epochs, MSE
            # is dropped entirely and scorer losses take over at full weight.
            is_warmup = epoch < cfg.warmup_epochs

            # Loss 1: MSE warmup (helps model learn basic colors/structure)
            loss_mse = torch.tensor(0.0, device=device)
            if is_warmup:
                loss_mse = (
                    F.mse_loss(gen_t, gt_t) + F.mse_loss(gen_t1, gt_t1)
                ) / 2.0

            # eval_roundtrip: simulate contest eval resize chain before scorer losses
            if cfg.eval_roundtrip:
                from tac.renderer import simulate_eval_roundtrip
                from tac.camera import CAMERA_H, CAMERA_W
                gen_t = simulate_eval_roundtrip(gen_t, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.5)
                gen_t1 = simulate_eval_roundtrip(gen_t1, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.5)
                gt_t = simulate_eval_roundtrip(gt_t, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0)
                gt_t1 = simulate_eval_roundtrip(gt_t1, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0)

            # Loss 2: SegNet cross-entropy (last frame only — matches official
            # scorer's x[:, -1, ...] behavior)
            loss_seg = torch.tensor(0.0, device=device)
            if not is_warmup or epoch > cfg.warmup_epochs // 2:
                gt_masks_t1 = masks_dev[idx_t1].long()
                loss_seg = _compute_seg_loss_ce(gen_t1, gt_masks_t1, segnet)

            # Loss 3: PoseNet coupled distortion
            loss_pose = torch.tensor(0.0, device=device)
            if not is_warmup or epoch > cfg.warmup_epochs // 2:
                pose_dist, pred_pose = _compute_pose_distortion_coupled(
                    gen_t, gen_t1, gt_t, gt_t1, posenet,
                )
                loss_pose = pose_dist

                # Direct pose supervision if targets available
                if pose_targets_all is not None:
                    sup_targets = pose_targets_all[batch_perm].to(device)
                    loss_pose = loss_pose + _compute_pose_supervision_loss(pred_pose, sup_targets)

            # Loss 4: TV smoothness
            loss_tv = _tv_loss(torch.stack([gen_t, gen_t1], dim=1))
            # Guard against NaN (FP4 QAT on MPS can produce NaN outputs)
            if torch.isnan(loss_tv):
                loss_tv = torch.tensor(0.0, device=device)

            # Combine losses
            if is_warmup:
                total_loss = (
                    cfg.warmup_mse_weight * loss_mse
                    + 0.1 * cfg.seg_weight * loss_seg  # gentle seg signal in warmup
                    + 0.01 * cfg.pose_weight * loss_pose  # very gentle pose
                    + cfg.tv_weight * loss_tv
                )
            else:
                total_loss = (
                    cfg.seg_weight * loss_seg
                    + cfg.pose_weight * loss_pose
                    + cfg.tv_weight * loss_tv
                )

            # Skip step if loss is NaN (MPS FP4 QAT edge case)
            if torch.isnan(total_loss):
                optimizer.zero_grad(set_to_none=True)
                n_steps += 1
                continue

            # Rate penalty (always on, proportional to model size)
            model_bytes = n_model_params * 4 / 8  # FP4 estimate
            rate = model_bytes / ORIGINAL_UNCOMPRESSED_SIZE
            # Rate is constant for fixed architecture, but included for
            # self-compression awareness when LearnableBitDepth is added
            # (Phase 3). For now it's a constant — no gradient.

            # Backward + step
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # EMA update
            ema.update(model)

            # Accumulate metrics
            epoch_seg_loss += loss_seg.item()
            epoch_pose_loss += loss_pose.item()
            epoch_tv_loss += loss_tv.item()
            epoch_mse_loss += loss_mse.item()
            epoch_total_loss += total_loss.item()
            n_steps += 1

        scheduler.step()

        # --- Auto-kill check (catches NaN and divergence) ---
        avg_total = epoch_total_loss / max(n_steps, 1)
        if math.isnan(avg_total) or avg_total > cfg.kill_loss_threshold:
            diverge_count += 1
            if diverge_count >= cfg.kill_patience:
                print(f"\n  DIVERGENCE KILL at epoch {epoch}: loss={avg_total}")
                break
        else:
            diverge_count = 0

        # --- Logging ---
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            phase = "warmup" if epoch < cfg.warmup_epochs else "scorer"
            elapsed = time.time() - start_time
            print(
                f"  ep {epoch:5d} | {phase:7s} | "
                f"loss {avg_total:8.4f} | "
                f"seg {epoch_seg_loss / max(n_steps, 1):7.4f} | "
                f"pose {epoch_pose_loss / max(n_steps, 1):7.4f} | "
                f"tv {epoch_tv_loss / max(n_steps, 1):6.4f} | "
                f"mse {epoch_mse_loss / max(n_steps, 1):7.4f} | "
                f"lr {lr_now:.2e} | "
                f"VRAM {_vram_mb():.0f}MB | "
                f"{elapsed / 60:.1f}min"
            )

        # --- Evaluation ---
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            # Use EMA weights for eval
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply(model)

            metrics = _full_eval(
                model, masks, gt_chw, posenet, segnet,
                device=device,
                batch_size=max(1, cfg.batch_size // 2),
            )

            # Restore training weights
            model.load_state_dict(orig_state)

            print(
                f"  EVAL ep {epoch:5d} | "
                f"seg {metrics['avg_seg']:.6f} | "
                f"pose {metrics['avg_pose']:.4f} | "
                f"rate {metrics['rate']:.6f} | "
                f"SCORE {metrics['score']:.4f}"
            )

            history.append({"epoch": epoch, **metrics})

            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best_epoch = epoch
                best_state = ema.state_dict()
                print(f"  *** NEW BEST: {best_score:.4f} at epoch {epoch} ***")

                # Save best checkpoint
                ckpt_path = results_dir / "best_joint_pair.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "ema_state_dict": best_state,  # best_state IS the EMA state
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_score": best_score,
                    "score": best_score,
                    "metrics": metrics,
                    "config": asdict(cfg),
                }, ckpt_path, pickle_protocol=4)
                print(f"  Saved checkpoint: {ckpt_path}")

        # --- Periodic checkpoint ---
        if epoch > 0 and epoch % cfg.checkpoint_every == 0:
            ckpt_path = results_dir / f"joint_pair_ep{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "best_score": best_score,
                "config": asdict(cfg),
            }, ckpt_path, pickle_protocol=4)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ---------------------------------------------------------------------------
    # Final export
    # ---------------------------------------------------------------------------
    elapsed_total = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed_total / 60:.1f} min")
    print(f"  Best score: {best_score:.4f} at epoch {best_epoch}")

    # Export best model in FP4
    if best_state is not None:
        # QAT parametrizations change state_dict keys. Build a fresh model
        # and map parametrized keys back to clean keys for export.
        if qat_model is not None:
            qat_model.remove_hooks()
            qat_model = None

        from tac.joint_pair_generator import JointPairGenerator as JPG_
        export_model = JPG_(num_classes=cfg.num_classes, base_ch=cfg.base_ch).to(device)
        try:
            export_model.load_state_dict(best_state)
        except RuntimeError:
            # Parametrized keys — use canonical strip helper (factored
            # 2026-04-28, hardened Round 11 for root-level keys + multi-
            # original weight_norm + nested chains).
            from tac.parametrize_strip import strip_parametrize_hooks
            export_model.load_state_dict(strip_parametrize_hooks(best_state))

        from tac.fp4_quantize import save_fp4
        fp4_path = results_dir / "joint_pair_best_fp4.pt"
        fp4_bytes = save_fp4(
            export_model, fp4_path,
            meta={
                "architecture": "JointPairGenerator",
                "base_ch": cfg.base_ch,
                "num_classes": cfg.num_classes,
                "best_epoch": best_epoch,
                "best_score": best_score,
            },
        )
        rate = fp4_bytes / ORIGINAL_UNCOMPRESSED_SIZE
        print(f"  FP4 export: {fp4_bytes:,} bytes, rate={rate:.6f}")
        print(f"  Rate contribution to score: {25.0 * rate:.4f}")

    # Save history
    history_path = results_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")

    return {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "elapsed_min": elapsed_total / 60,
        "history": history,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--epochs", default=5000, type=int, help="Training epochs")
@click.option("--batch-size", default=8, type=int, help="Pairs per batch")
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
@click.option("--base-ch", default=40, type=int, help="Base channels (32=472K, 40=731K, 48=1M)")
@click.option("--device", default=None, type=str, help="Device (auto-detected if not set)")
@click.option("--seg-weight", default=100.0, type=float, help="SegNet loss weight")
@click.option("--pose-weight", default=1.0, type=float, help="PoseNet loss weight")
@click.option("--tv-weight", default=0.1, type=float, help="TV loss weight")
@click.option("--rate-weight", default=25.0, type=float, help="Rate penalty weight (eval-only; no gradient for fixed architecture)")
@click.option("--warmup-epochs", default=200, type=int, help="MSE warmup epochs")
@click.option("--eval-every", default=200, type=int, help="Eval interval")
@click.option("--checkpoint-every", default=500, type=int, help="Checkpoint interval")
@click.option("--fp4-qat/--no-fp4-qat", default=True, help="Enable FP4 QAT")
@click.option("--pose-targets", default=None, type=str, help="Path to posenet_targets.bin")
@click.option("--resume", default=None, type=str, help="Resume from checkpoint path")
@click.option("--max-hours", default=48.0, type=float, help="Wall-clock budget in hours")
@click.option("--eval-roundtrip/--no-eval-roundtrip", default=True, help="Simulate contest eval resize chain (default: on)")
@click.option("--smoke", is_flag=True, help="Smoke test: 10 epochs, 20 frames")
def main(
    epochs: int,
    batch_size: int,
    lr: float,
    base_ch: int,
    device: str | None,
    seg_weight: float,
    pose_weight: float,
    tv_weight: float,
    rate_weight: float,
    warmup_epochs: int,
    eval_every: int,
    checkpoint_every: int,
    fp4_qat: bool,
    pose_targets: str | None,
    resume: str | None,
    max_hours: float,
    eval_roundtrip: bool,
    smoke: bool,
):
    """Train the JointPairGenerator: mask pairs -> frame pairs."""

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    cfg = JointPairConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        base_ch=base_ch,
        device=device,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
        tv_weight=tv_weight,
        rate_weight=rate_weight,
        warmup_epochs=warmup_epochs,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        fp4_qat=fp4_qat,
        pose_targets_path=pose_targets,
        resume=resume,
        max_hours=max_hours,
        eval_roundtrip=eval_roundtrip,
        smoke=smoke,
    )

    # Smoke test overrides (L5)
    # NOTE: Smoke test uses only 20 frames / 10 epochs / 3 warmup epochs.
    # This is insufficient to validate training signal quality — it only
    # verifies that the pipeline runs end-to-end without crashes. A negative
    # result on a smoke test CANNOT kill a technique (see CLAUDE.md).
    if smoke:
        cfg.epochs = 10
        cfg.warmup_epochs = 3
        cfg.eval_every = 5
        cfg.checkpoint_every = 100
        cfg.log_every = 1
        cfg.batch_size = min(batch_size, 4)
        cfg.max_hours = 0.5
        print("[SMOKE TEST MODE]")

    results = train_joint_pair(cfg)

    print()
    print(f"Final best score: {results['best_score']:.4f} "
          f"(epoch {results['best_epoch']}) "
          f"in {results['elapsed_min']:.1f} min")


if __name__ == "__main__":
    main()
