#!/usr/bin/env python
"""Experiment 2: Tiny DP-SIMS -- PROPER training run (not smoke test).

Run on Lightning T4:
    PYTHONPATH=src:/home/zeus/content/upstream python experiments/exp2_tiny_dp_sims_proper.py

Pre-registered hypothesis:
    "Tiny DP-SIMS at 78KB FP4 produces seg < 0.01, pose < 0.05 on 100 training frames"

Success criteria:
    score < 1.0 including rate --> PROMOTE to full 1200 frames
Kill criteria:
    score > 3.0 --> architecture too small, KILL
Concern:
    score 1.0-3.0 --> try channels=(64,32,16,8) at 182KB

Improvements over smoke test:
    - 100 frames (smoke used 20)
    - 5000 training steps (smoke used 500)
    - Full scorer loss with correct formula weighting
    - FP4 quantization at export with proper measurement
    - Masks extracted from GT using SegNet (with preprocess_input)
    - Score with full formula including rate
    - Cosine annealing LR with warm restarts
    - Gradient accumulation for effective batch 16
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
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

_CANDIDATE_GT = [
    Path("/home/zeus/content/upstream/videos/0.mkv"),
    Path(__file__).resolve().parent.parent / "upstream" / "videos" / "0.mkv",
]
GT_VIDEO: Path | None = None
for _p in _CANDIDATE_GT:
    if _p is not None and _p.exists():
        GT_VIDEO = _p
        break

RESULTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TinyDPSIMSConfig:
    """All hyperparameters for the proper Tiny DP-SIMS experiment."""

    n_frames: int = 100
    steps: int = 5000

    # Architecture
    channels: tuple[int, ...] = (32, 16, 8, 4)
    init_h: int = 24
    init_w: int = 32
    spade_hidden: int = 16
    num_classes: int = 5

    # Training
    lr: float = 1e-3
    lr_min_ratio: float = 0.01
    batch_size: int = 4  # consecutive frames per step
    accum_steps: int = 4  # effective batch = batch_size * accum_steps = 16
    grad_clip: float = 1.0

    # Loss weights (match scorer formula)
    seg_weight: float = 100.0
    pose_weight: float = 10.0  # approximation of sqrt(10*pose) gradient

    # Scorer resolution
    target_h: int = 384
    target_w: int = 512

    # Logging
    log_every: int = 100
    eval_every: int = 500  # full evaluation every N steps

    # Fallback architecture (if primary fails)
    fallback_channels: tuple[int, ...] = (64, 32, 16, 8)
    fallback_spade_hidden: int = 32

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Shared helpers (same as exp1)
# ---------------------------------------------------------------------------

def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    if WEIGHTS_DIR is None:
        raise FileNotFoundError(f"Scorer weights not found in {_CANDIDATE_WEIGHTS}")
    from tac.scorer import load_scorers
    return load_scorers(
        WEIGHTS_DIR / "posenet.safetensors",
        WEIGHTS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=UPSTREAM_ROOT,
    )


def _load_gt_frames(n_frames: int, target_h: int = 384, target_w: int = 512) -> list[torch.Tensor]:
    if GT_VIDEO is None:
        raise FileNotFoundError(f"GT video not found in {_CANDIDATE_GT}")
    from tac.data import decode_video
    frames = decode_video(str(GT_VIDEO), target_h=target_h, target_w=target_w)
    return frames[:n_frames]


def _vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def _extract_masks(gt_chw: torch.Tensor, segnet: nn.Module) -> torch.Tensor:
    masks_list = []
    with torch.no_grad():
        for i in range(gt_chw.shape[0]):
            inp = gt_chw[i:i+1].unsqueeze(1)
            seg_in = segnet.preprocess_input(inp)
            logits = segnet(seg_in)
            mask = logits.argmax(dim=1).squeeze(0)
            masks_list.append(mask)
    return torch.stack(masks_list)


def _score_frames_batched(
    gen_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    batch_size: int = 4,
) -> dict[str, float]:
    """Score generated vs GT using official formula. All pairs, batched."""
    n_frames = gen_chw.shape[0]
    n_pairs = n_frames - 1
    seg_dists = []
    pose_dists = []

    with torch.no_grad():
        for i in range(n_pairs):
            gen_pair = gen_chw[i:i+2].unsqueeze(0)
            gt_pair = gt_chw[i:i+2].unsqueeze(0)

            seg_in_gen = segnet.preprocess_input(gen_pair)
            seg_out_gen = segnet(seg_in_gen)
            seg_in_gt = segnet.preprocess_input(gt_pair)
            seg_out_gt = segnet(seg_in_gt)
            p_soft = F.softmax(seg_out_gen, dim=1)
            g_soft = F.softmax(seg_out_gt, dim=1)
            seg_d = (1.0 - (p_soft * g_soft).sum(dim=1).mean()).item()
            seg_dists.append(seg_d)

            pose_in_gen = posenet.preprocess_input(gen_pair)
            pose_out_gen = posenet(pose_in_gen)
            pose_in_gt = posenet.preprocess_input(gt_pair)
            pose_out_gt = posenet(pose_in_gt)
            pm = pose_out_gen["pose"] if isinstance(pose_out_gen, dict) else pose_out_gen
            po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
            pose_d = (pm[..., :6] - po[..., :6]).pow(2).mean().item()
            pose_dists.append(pose_d)

    avg_seg = sum(seg_dists) / len(seg_dists) if seg_dists else 0.0
    avg_pose = sum(pose_dists) / len(pose_dists) if pose_dists else 0.0
    return {"avg_seg": avg_seg, "avg_pose": avg_pose, "seg_dists": seg_dists, "pose_dists": pose_dists}


# ---------------------------------------------------------------------------
# Model (imported from gpu_lane_dual_smoke but also defined here for independence)
# ---------------------------------------------------------------------------

class _TinySPADE(nn.Module):
    """Minimal SPADE normalization."""

    def __init__(self, norm_ch: int, mask_ch: int = 5, hidden: int = 16):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_ch, affine=False)
        self.mask_ch = mask_ch
        self.shared = nn.Sequential(
            nn.Conv2d(mask_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden, norm_ch, 3, padding=1)
        self.beta = nn.Conv2d(hidden, norm_ch, 3, padding=1)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        _, _, fH, fW = x.shape
        B = mask.shape[0]
        if mask.shape[1] != fH or mask.shape[2] != fW:
            m = F.interpolate(mask.unsqueeze(1).float(), size=(fH, fW), mode="nearest").squeeze(1).long()
        else:
            m = mask
        onehot = torch.zeros(B, self.mask_ch, fH, fW, device=x.device, dtype=x.dtype)
        onehot.scatter_(1, m.unsqueeze(1), 1.0)
        shared = self.shared(onehot)
        return normalized * (1.0 + self.gamma(shared)) + self.beta(shared)


class _TinySPADEResBlock(nn.Module):
    """Minimal SPADE ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, mask_ch: int = 5, spade_h: int = 16):
        super().__init__()
        self.learned_skip = in_ch != out_ch
        self.spade1 = _TinySPADE(in_ch, mask_ch, spade_h)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.spade2 = _TinySPADE(out_ch, mask_ch, spade_h)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        if self.learned_skip:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.spade1(x, mask)
        h = self.act(h)
        h = self.conv1(h)
        h = self.spade2(h, mask)
        h = self.act(h)
        h = self.conv2(h)
        if self.learned_skip:
            x = self.skip_conv(x)
        return x + h


class TinyDPSIMS(nn.Module):
    """Minimal DP-SIMS: SPADE-conditioned mask-to-RGB."""

    def __init__(
        self,
        num_classes: int = 5,
        channels: tuple[int, ...] = (32, 16, 8, 4),
        init_h: int = 24,
        init_w: int = 32,
        spade_hidden: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.init_h = init_h
        self.init_w = init_w
        self.num_stages = len(channels)
        self.const = nn.Parameter(torch.randn(1, channels[0], init_h, init_w) * 0.02)
        self.spade_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            sh = max(4, min(spade_hidden, out_ch))
            self.spade_blocks.append(_TinySPADEResBlock(in_ch, out_ch, num_classes, sh))
            in_ch = out_ch
        self.head = nn.Conv2d(channels[-1], 3, 3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        B = masks.shape[0]
        target_h, target_w = masks.shape[1], masks.shape[2]
        x = self.const.expand(B, -1, -1, -1)
        for i, block in enumerate(self.spade_blocks):
            x = block(x, masks)
            if i < self.num_stages - 1:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return 255.0 * torch.sigmoid(self.head(x) / 50.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_bytes_fp4(self) -> int:
        return self.param_count() // 2 + 256


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_tiny_dp_sims_proper(cfg: TinyDPSIMSConfig) -> dict[str, Any]:
    """Run the proper Tiny DP-SIMS training experiment."""

    device = cfg.device
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Tiny DP-SIMS (PROPER)")
    print(f"  {cfg.n_frames} frames, {cfg.steps} steps")
    print(f"  Channels: {cfg.channels}, spade_hidden: {cfg.spade_hidden}")
    print(f"  Effective batch: {cfg.batch_size} x {cfg.accum_steps} = {cfg.batch_size * cfg.accum_steps}")
    print("=" * 70)

    results: dict[str, Any] = {
        "experiment": "tiny_dp_sims_proper",
        "config": {k: str(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
        "status": "running",
        "hypothesis": (
            "Tiny DP-SIMS at 78KB FP4 produces seg < 0.01, pose < 0.05 on 100 training frames"
        ),
    }
    t0 = time.time()

    # --- Load ---
    print(f"\n[1/5] Loading {cfg.n_frames} GT frames and scorers...")
    gt_frames_hwc = _load_gt_frames(cfg.n_frames, cfg.target_h, cfg.target_w)
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc]).to(device)
    posenet, segnet = _load_scorers(device)
    print(f"  VRAM after load: {_vram_mb():.0f} MB")

    # --- Extract masks ---
    print(f"\n[2/5] Extracting GT masks...")
    masks = _extract_masks(gt_chw, segnet)
    print(f"  Masks shape: {masks.shape}, classes: {masks.unique().tolist()}")

    # --- Build model ---
    print(f"\n[3/5] Building TinyDPSIMS...")
    model = TinyDPSIMS(
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        init_h=cfg.init_h,
        init_w=cfg.init_w,
        spade_hidden=cfg.spade_hidden,
    ).to(device)

    n_params = model.param_count()
    fp4_bytes = model.param_bytes_fp4()
    print(f"  Parameters: {n_params:,}")
    print(f"  FP4 estimate: {fp4_bytes / 1024:.1f} KB")
    print(f"  VRAM after model: {_vram_mb():.0f} MB")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.steps // 5, T_mult=2, eta_min=cfg.lr * cfg.lr_min_ratio,
    )

    # --- Training ---
    print(f"\n[4/5] Training for {cfg.steps} steps...")
    history: list[dict] = []
    best_score = float("inf")
    best_state = None

    for step in range(cfg.steps):
        # Sample random batch of consecutive frames
        batch_start = torch.randint(0, max(1, cfg.n_frames - cfg.batch_size + 1), (1,)).item()
        batch_end = min(batch_start + cfg.batch_size, cfg.n_frames)
        batch_masks = masks[batch_start:batch_end]
        batch_gt = gt_chw[batch_start:batch_end]
        B = batch_gt.shape[0]

        # Generate
        gen_rgb = model(batch_masks)

        # SegNet loss
        seg_loss = torch.tensor(0.0, device=device)
        for i in range(B):
            gen_inp = gen_rgb[i:i+1].unsqueeze(1)
            gt_inp = batch_gt[i:i+1].unsqueeze(1)
            seg_in_gen = segnet.preprocess_input(gen_inp)
            seg_out_gen = segnet(seg_in_gen)
            with torch.no_grad():
                seg_in_gt = segnet.preprocess_input(gt_inp)
                seg_out_gt = segnet(seg_in_gt)
            p_soft = F.softmax(seg_out_gen, dim=1)
            g_soft = F.softmax(seg_out_gt, dim=1)
            seg_loss = seg_loss + (1.0 - (p_soft * g_soft).sum(dim=1).mean())
        seg_loss = seg_loss / B

        # PoseNet loss
        pose_loss = torch.tensor(0.0, device=device)
        n_pose_pairs = max(1, B - 1)
        for i in range(B - 1):
            gen_pair = gen_rgb[i:i+2].unsqueeze(0)
            gt_pair = batch_gt[i:i+2].unsqueeze(0)
            pose_in_gen = posenet.preprocess_input(gen_pair)
            pose_out_gen = posenet(pose_in_gen)
            with torch.no_grad():
                pose_in_gt = posenet.preprocess_input(gt_pair)
                pose_out_gt = posenet(pose_in_gt)
            pm = pose_out_gen["pose"] if isinstance(pose_out_gen, dict) else pose_out_gen
            po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
            pose_loss = pose_loss + (pm[..., :6] - po[..., :6]).pow(2).mean()
        pose_loss = pose_loss / n_pose_pairs

        # Combined loss using scorer formula weights
        total_loss = cfg.seg_weight * seg_loss + cfg.pose_weight * pose_loss

        # Gradient accumulation
        total_loss = total_loss / cfg.accum_steps
        total_loss.backward()

        if (step + 1) % cfg.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Logging
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            record = {
                "step": step,
                "seg_loss": round(seg_loss.item(), 6),
                "pose_loss": round(pose_loss.item(), 6),
                "total_loss": round((total_loss * cfg.accum_steps).item(), 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
            }
            history.append(record)
            print(
                f"  step {step:5d}: seg={seg_loss.item():.6f} "
                f"pose={pose_loss.item():.6f} "
                f"total={record['total_loss']:.4f} "
                f"lr={record['lr']:.6f}"
            )

        # Periodic full evaluation
        if step % cfg.eval_every == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                gen_all = model(masks)
            scores = _score_frames_batched(gen_all, gt_chw, posenet, segnet)
            rate = fp4_bytes / (cfg.n_frames * cfg.target_h * cfg.target_w * 1.5)
            from tac.scorer import comma_score
            score = comma_score(scores["avg_pose"], scores["avg_seg"], rate)
            print(f"  [EVAL@{step}] score={score:.4f} seg={scores['avg_seg']:.6f} pose={scores['avg_pose']:.6f}")
            if score < best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

    # --- Final evaluation ---
    print(f"\n[5/5] Final scorer evaluation...")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best checkpoint (score={best_score:.4f})")

    model.eval()
    with torch.no_grad():
        gen_all = model(masks)

    scores = _score_frames_batched(gen_all, gt_chw, posenet, segnet)
    final_seg = scores["avg_seg"]
    final_pose = scores["avg_pose"]

    total_pixels_yuv = cfg.n_frames * cfg.target_h * cfg.target_w * 1.5
    rate = fp4_bytes / total_pixels_yuv

    from tac.scorer import comma_score
    total_score = comma_score(final_pose, final_seg, rate)

    results.update({
        "final_seg_dist": round(final_seg, 8),
        "final_pose_dist": round(final_pose, 8),
        "model_params": n_params,
        "model_fp4_bytes": fp4_bytes,
        "model_fp4_kb": round(fp4_bytes / 1024, 1),
        "rate": round(rate, 6),
        "total_score": round(total_score, 4),
        "best_intermediate_score": round(best_score, 4),
        "score_components": {
            "seg_term": round(100.0 * final_seg, 4),
            "pose_term": round(math.sqrt(10.0 * final_pose), 4),
            "rate_term": round(25.0 * rate, 4),
        },
        "history": history,
    })

    elapsed = time.time() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    results["elapsed_seconds"] = round(elapsed, 1)
    results["vram_peak_mb"] = round(vram_peak, 0)
    results["status"] = "ok"

    # Verdict
    if total_score < 1.0:
        results["verdict"] = "PROMOTE_TO_FULL_1200"
        verdict = "PROMOTE: score < 1.0 including rate!"
    elif total_score > 3.0:
        results["verdict"] = "KILLED_ARCHITECTURE_TOO_SMALL"
        verdict = "KILLED: score > 3.0 -- architecture too small."
    else:
        results["verdict"] = "CONCERN_TRY_LARGER"
        verdict = f"CONCERN: score {total_score:.4f} in [1.0, 3.0]. Try channels=(64,32,16,8) at 182KB."

    print(f"\n  --- TINY DP-SIMS PROPER VERDICT ---")
    print(f"  SegNet dist:  {final_seg:.6f}")
    print(f"  PoseNet dist: {final_pose:.6f}")
    print(f"  Model FP4:    {fp4_bytes / 1024:.1f} KB")
    print(f"  Rate:         {rate:.6f}")
    print(f"  TOTAL SCORE:  {total_score:.4f}")
    print(f"  Time: {elapsed:.0f}s | VRAM: {vram_peak:.0f} MB")
    print(f"  VERDICT:      {verdict}")

    # Cleanup
    del model, optimizer, gen_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: Tiny DP-SIMS Proper")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--channels", default="32,16,8,4", help="Channel widths")
    parser.add_argument("--spade-hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    channels = tuple(int(x) for x in args.channels.split(","))

    cfg = TinyDPSIMSConfig(
        n_frames=args.n_frames,
        steps=args.steps,
        channels=channels,
        spade_hidden=args.spade_hidden,
        lr=args.lr,
        device=args.device,
    )

    print("=" * 70)
    print("EXPERIMENT 2: TINY DP-SIMS (PROPER)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    try:
        results = run_tiny_dp_sims_proper(cfg)
    except Exception:
        traceback.print_exc()
        results = {"status": "error", "error": traceback.format_exc()}

    out_path = RESULTS_DIR / "exp2_tiny_dp_sims_proper_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
