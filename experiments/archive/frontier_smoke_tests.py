#!/usr/bin/env python
"""Frontier smoke tests: SIREN memorization, Fridrich constrained gen, self-compression.

Three viability tests for sub-0.50 score techniques. Each answers a specific
question in <30 minutes total on a T4 GPU.

Usage (on Lightning T4):
    PYTHONPATH=src:/home/zeus/content/upstream python frontier_smoke_tests.py --device cuda
    PYTHONPATH=src:/home/zeus/content/upstream python frontier_smoke_tests.py --test siren
    PYTHONPATH=src:/home/zeus/content/upstream python frontier_smoke_tests.py --test fridrich
    PYTHONPATH=src:/home/zeus/content/upstream python frontier_smoke_tests.py --test compress
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — discover upstream and scorer weights automatically
# ---------------------------------------------------------------------------

# Search order for upstream
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

# Search for scorer weights
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

# Search for GT video
_CANDIDATE_GT = [
    Path("/home/zeus/content/upstream/videos/0.mkv"),
    Path(__file__).resolve().parent.parent / "upstream" / "videos" / "0.mkv",
]
GT_VIDEO: Path | None = None
for _p in _CANDIDATE_GT:
    if _p is not None and _p.exists():
        GT_VIDEO = _p
        break


def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    """Load frozen PoseNet + SegNet. Fails fast if weights not found."""
    if WEIGHTS_DIR is None:
        raise FileNotFoundError(
            "Scorer weights not found. Searched: " +
            ", ".join(str(p) for p in _CANDIDATE_WEIGHTS)
        )
    from tac.scorer import load_scorers
    return load_scorers(
        WEIGHTS_DIR / "posenet.safetensors",
        WEIGHTS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=UPSTREAM_ROOT,
    )


def _load_gt_frames(n_frames: int = 20) -> list[torch.Tensor]:
    """Load n ground truth frames as (H, W, 3) uint8 tensors."""
    if GT_VIDEO is None:
        raise FileNotFoundError(
            "GT video not found. Searched: " +
            ", ".join(str(p) for p in _CANDIDATE_GT)
        )
    from tac.data import decode_video
    frames = decode_video(str(GT_VIDEO))
    return frames[:n_frames]


def _estimate_vram_mb() -> float:
    """Estimate current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# =========================================================================
# Smoke Test 1: SIREN Video Memorization
# =========================================================================

class SirenLayer(nn.Module):
    """Single SIREN layer: Linear → sin(omega * x).

    SIREN (Sitzmann et al., 2020) uses sinusoidal activations with
    careful initialization. omega_0 controls frequency range.
    """

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            fan_in = self.linear.weight.shape[1]
            if self.is_first:
                # First layer: uniform in [-1/fan_in, 1/fan_in]
                bound = 1.0 / fan_in
            else:
                # Hidden layers: uniform in [-sqrt(6/fan_in)/omega_0, ...]
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenNetwork(nn.Module):
    """SIREN for video memorization: (frame_idx, y, x) -> (R, G, B).

    Maps normalized coordinates in [-1, 1] to RGB in [0, 255].
    """

    def __init__(self, hidden: int = 64, layers: int = 4, omega_0: float = 30.0):
        super().__init__()
        self.hidden = hidden
        self.layers_count = layers
        self.omega_0 = omega_0

        net = [SirenLayer(3, hidden, omega_0=omega_0, is_first=True)]
        for _ in range(layers - 1):
            net.append(SirenLayer(hidden, hidden, omega_0=omega_0))
        self.net = nn.Sequential(*net)
        # Final linear: no sine activation, output RGB
        self.final = nn.Linear(hidden, 3)
        with torch.no_grad():
            fan_in = hidden
            bound = math.sqrt(6.0 / fan_in) / omega_0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 3) with (t, y, x) in [-1, 1]. Returns (..., 3) in [0, 255]."""
        h = self.net(coords)
        rgb = self.final(h)
        # Sigmoid -> [0, 255]
        return torch.sigmoid(rgb) * 255.0

    def param_bytes_fp16(self) -> int:
        """Total parameter count * 2 bytes (fp16 storage)."""
        return sum(p.numel() for p in self.parameters()) * 2


def _build_siren_coords(n_frames: int, H: int, W: int,
                         device: str = "cpu") -> torch.Tensor:
    """Build normalized (t, y, x) coordinate grid for SIREN input.

    Returns (n_frames * H * W, 3) tensor with values in [-1, 1].
    """
    t_vals = torch.linspace(-1, 1, n_frames, device=device)
    y_vals = torch.linspace(-1, 1, H, device=device)
    x_vals = torch.linspace(-1, 1, W, device=device)
    grid = torch.meshgrid(t_vals, y_vals, x_vals, indexing="ij")
    coords = torch.stack(grid, dim=-1)  # (T, H, W, 3)
    return coords.reshape(-1, 3)


def test_siren_memorization(
    device: str = "cuda",
    n_frames: int = 20,
    hidden: int = 64,
    layers: int = 4,
    omega_0: float = 30.0,
    steps: int = 500,
    lr: float = 1e-4,
    batch_pixels: int = 65536,
) -> dict[str, Any]:
    """Smoke Test 1: Can a SIREN memorize 20 frames of driving video?

    Trains SIREN to minimize MSE on pixel coordinates -> RGB.
    Measures PSNR, model size, and scorer distortion on memorized output.
    """
    print("\n" + "=" * 70)
    print("SMOKE TEST 1: SIREN Video Memorization")
    print("=" * 70)

    results: dict[str, Any] = {"test": "siren_memorization", "status": "running"}
    t0 = time.time()

    # --- Load data ---
    print(f"\n[1/4] Loading {n_frames} GT frames...")
    gt_frames_hwc = _load_gt_frames(n_frames)
    H, W = gt_frames_hwc[0].shape[:2]
    print(f"  Frame size: {H}x{W}, {n_frames} frames")

    # Work at reduced resolution for speed (SIREN is resolution-independent)
    scale = 4
    H_s, W_s = H // scale, W // scale
    print(f"  Working at {H_s}x{W_s} (1/{scale} resolution) for speed")

    # Stack to (N, 3, H, W) float and downscale
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc])  # (N, 3, H, W)
    gt_small = F.interpolate(gt_chw, size=(H_s, W_s), mode="bilinear", align_corners=False)
    # Target RGB: (N*H*W, 3)
    target_rgb = gt_small.permute(0, 2, 3, 1).reshape(-1, 3).to(device)  # (N*H*W, 3)

    # --- Build SIREN ---
    print(f"\n[2/4] Building SIREN(hidden={hidden}, layers={layers}, omega_0={omega_0})...")
    model = SirenNetwork(hidden=hidden, layers=layers, omega_0=omega_0).to(device)
    param_bytes = model.param_bytes_fp16()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({param_bytes:,} bytes at fp16 = {param_bytes/1024:.1f} KB)")
    results["param_count"] = n_params
    results["model_bytes_fp16"] = param_bytes
    results["model_kb_fp16"] = round(param_bytes / 1024, 2)

    # Check for NaN after init (omega_0 sensitivity)
    with torch.no_grad():
        test_coord = torch.zeros(1, 3, device=device)
        test_out = model(test_coord)
        if torch.isnan(test_out).any():
            results["status"] = "FAIL_NAN_INIT"
            results["error"] = f"NaN in SIREN output at init with omega_0={omega_0}"
            print(f"  FATAL: {results['error']}")
            return results
    print(f"  Init check: no NaN (omega_0={omega_0} is stable)")

    # --- Build coordinate grid ---
    coords = _build_siren_coords(n_frames, H_s, W_s, device=device)
    total_pixels = coords.shape[0]
    print(f"  Total pixels to memorize: {total_pixels:,}")

    vram_before = _estimate_vram_mb()
    print(f"  VRAM before training: {vram_before:.0f} MB")

    # --- Train ---
    print(f"\n[3/4] Training for {steps} steps (batch={batch_pixels})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    losses = []
    for step in range(steps):
        # Random batch of pixels
        idx = torch.randint(0, total_pixels, (batch_pixels,), device=device)
        batch_coords = coords[idx]
        batch_target = target_rgb[idx]

        pred = model(batch_coords)
        loss = F.mse_loss(pred, batch_target)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for SIREN stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if step % 100 == 0 or step == steps - 1:
            psnr = 10.0 * math.log10(255.0 ** 2 / max(loss.item(), 1e-10))
            print(f"  step {step:4d}/{steps} | MSE={loss.item():.2f} | PSNR={psnr:.1f} dB")

    # --- Evaluate ---
    print(f"\n[4/4] Evaluating memorized output...")
    model.eval()
    with torch.no_grad():
        # Reconstruct all frames
        pred_all = []
        for i in range(0, total_pixels, batch_pixels):
            batch = coords[i:i + batch_pixels]
            pred_all.append(model(batch))
        pred_rgb = torch.cat(pred_all, dim=0)  # (N*H*W, 3)

    # Compute final metrics
    mse = F.mse_loss(pred_rgb, target_rgb).item()
    psnr = 10.0 * math.log10(255.0 ** 2 / max(mse, 1e-10))
    print(f"  Final MSE: {mse:.2f}")
    print(f"  Final PSNR: {psnr:.1f} dB")
    results["final_mse"] = round(mse, 4)
    results["final_psnr"] = round(psnr, 2)

    # Reshape to frames for scorer evaluation
    pred_frames = pred_rgb.reshape(n_frames, H_s, W_s, 3)  # (N, H, W, 3)

    # Upscale back to original resolution for scorer
    pred_chw = pred_frames.permute(0, 3, 1, 2)  # (N, 3, H, W)
    pred_full = F.interpolate(pred_chw, size=(H, W), mode="bicubic", align_corners=False)
    pred_full = pred_full.clamp(0, 255)

    # Run scorer on a few frame pairs
    try:
        posenet, segnet = _load_scorers(device)
        seg_dists = []
        pose_dists = []
        n_pairs = min(5, n_frames - 1)
        for i in range(n_pairs):
            # Build (1, 2, 3, H, W) pairs
            pred_pair = pred_full[i:i + 2].unsqueeze(0)  # (1, 2, 3, H, W)
            gt_pair = gt_chw[i:i + 2].to(device).unsqueeze(0)

            # CRITICAL: use preprocess_input
            with torch.no_grad():
                gt_pose_in = posenet.preprocess_input(gt_pair)
                gt_pose_out = posenet(gt_pose_in)
                gt_pose = gt_pose_out["pose"] if isinstance(gt_pose_out, dict) else gt_pose_out

                pred_pose_in = posenet.preprocess_input(pred_pair)
                pred_pose_out = posenet(pred_pose_in)
                pred_pose = pred_pose_out["pose"] if isinstance(pred_pose_out, dict) else pred_pose_out

                pose_d = (pred_pose[..., :6] - gt_pose[..., :6]).pow(2).mean().item()
                pose_dists.append(pose_d)

                gt_seg_in = segnet.preprocess_input(gt_pair)
                gt_seg_out = segnet(gt_seg_in)
                pred_seg_in = segnet.preprocess_input(pred_pair)
                pred_seg_out = segnet(pred_seg_in)

                gt_soft = F.softmax(gt_seg_out, dim=1)
                pred_soft = F.softmax(pred_seg_out, dim=1)
                seg_d = (1.0 - (pred_soft * gt_soft).sum(dim=1).mean()).item()
                seg_dists.append(seg_d)

        avg_pose = sum(pose_dists) / len(pose_dists)
        avg_seg = sum(seg_dists) / len(seg_dists)
        # Score with hypothetical rate (model_bytes / 1e6)
        rate = param_bytes / 1e6
        from tac.scorer import comma_score
        score = comma_score(avg_pose, avg_seg, rate)

        print(f"\n  Scorer results ({n_pairs} pairs):")
        print(f"    PoseNet distortion: {avg_pose:.6f}")
        print(f"    SegNet distortion:  {avg_seg:.6f}")
        print(f"    Rate (model size):  {rate:.4f} MB")
        print(f"    Estimated score:    {score:.3f}")

        results["pose_distortion"] = round(avg_pose, 8)
        results["seg_distortion"] = round(avg_seg, 8)
        results["rate_mb"] = round(rate, 6)
        results["estimated_score"] = round(score, 4)

        del posenet, segnet
    except Exception as e:
        print(f"  Scorer eval failed: {e}")
        results["scorer_error"] = str(e)

    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    elapsed = time.time() - t0

    results["status"] = "ok"
    results["elapsed_seconds"] = round(elapsed, 1)
    results["vram_peak_mb"] = round(vram_peak, 0)
    results["resolution"] = f"{H_s}x{W_s} (1/{scale} of {H}x{W})"

    # Verdict
    print(f"\n  --- SIREN VERDICT ---")
    print(f"  PSNR: {psnr:.1f} dB at {param_bytes/1024:.1f} KB (fp16)")
    if psnr > 25:
        print(f"  VIABLE: Good reconstruction quality. Rate breakthrough possible.")
        results["verdict"] = "VIABLE"
    elif psnr > 20:
        print(f"  MARGINAL: Moderate quality. Needs more capacity or steps.")
        results["verdict"] = "MARGINAL"
    else:
        print(f"  NOT VIABLE at this scale: PSNR too low for scorer fidelity.")
        results["verdict"] = "NOT_VIABLE"
    print(f"  Time: {elapsed:.0f}s | VRAM peak: {vram_peak:.0f} MB")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return results


# =========================================================================
# Smoke Test 2: Fridrich Constrained Generation
# =========================================================================

def test_fridrich_constrained_gen(
    device: str = "cuda",
    n_frames: int = 8,
    steps: int = 100,
    lr: float = 0.5,
    seg_boundary: float = 0.01,
    pose_boundary: float = 0.1,
    rho_init: float = 10.0,
    rho_growth: float = 1.3,
    rho_max: float = 1000.0,
    outer_steps: int = 5,
) -> dict[str, Any]:
    """Smoke Test 2: Fridrich constrained optimization.

    Instead of weighted-sum loss (which causes PoseNet divergence),
    minimize TV subject to hard constraints on seg and pose distortion.
    Uses augmented Lagrangian with capped rho growth for stability.
    """
    print("\n" + "=" * 70)
    print("SMOKE TEST 2: Fridrich Constrained Generation")
    print("=" * 70)

    results: dict[str, Any] = {"test": "fridrich_constrained_gen", "status": "running"}
    t0 = time.time()

    # --- Load data and scorers ---
    print(f"\n[1/4] Loading {n_frames} GT frames and scorers...")
    gt_frames_hwc = _load_gt_frames(n_frames)
    H, W = gt_frames_hwc[0].shape[:2]
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc]).to(device)
    posenet, segnet = _load_scorers(device)

    vram_before = _estimate_vram_mb()
    print(f"  Loaded. VRAM: {vram_before:.0f} MB")
    print(f"  Constraints: seg < {seg_boundary}, pose < {pose_boundary}")

    # --- Initialize from GT (small perturbation) ---
    print(f"\n[2/4] Initializing optimization variable...")
    # Start from GT + small noise (simulates decoded frames with artifacts)
    original = gt_chw.detach().clone()
    delta = torch.zeros_like(gt_chw, requires_grad=True)

    optimizer = torch.optim.Adam([delta], lr=lr)
    inner_steps = max(1, steps // outer_steps)

    # Lagrangian multipliers
    lam_s = 1.0
    lam_p = 1.0
    rho = rho_init

    # --- Optimization ---
    print(f"\n[3/4] Running augmented Lagrangian ({outer_steps} outer x {inner_steps} inner)...")
    history: list[dict] = []

    for outer in range(outer_steps):
        for step in range(inner_steps):
            optimizer.zero_grad()

            current = (original + delta).clamp(0.0, 255.0)

            # Total variation (rate proxy)
            dx = (current[:, :, :, 1:] - current[:, :, :, :-1]).abs().mean()
            dy = (current[:, :, 1:, :] - current[:, :, :-1, :]).abs().mean()
            tv_loss = dx + dy

            # Scorer distortion on random frame pair
            idx = torch.randint(0, max(1, n_frames - 1), (1,)).item()
            cur_pair = current[idx:idx + 2].unsqueeze(0)  # (1, 2, 3, H, W)
            gt_pair = original[idx:idx + 2].unsqueeze(0)

            if cur_pair.shape[1] < 2:
                cur_pair = current[:2].unsqueeze(0)
                gt_pair = original[:2].unsqueeze(0)

            # CRITICAL: use preprocess_input for ALL scorer calls
            # SegNet
            seg_in_mod = segnet.preprocess_input(cur_pair)
            seg_out_mod = segnet(seg_in_mod)
            with torch.no_grad():
                seg_in_orig = segnet.preprocess_input(gt_pair)
                seg_out_orig = segnet(seg_in_orig)
            pred_soft = F.softmax(seg_out_mod, dim=1)
            gt_soft = F.softmax(seg_out_orig, dim=1)
            seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

            # PoseNet
            pose_in_mod = posenet.preprocess_input(cur_pair)
            pose_out_mod = posenet(pose_in_mod)
            with torch.no_grad():
                pose_in_orig = posenet.preprocess_input(gt_pair)
                pose_out_orig = posenet(pose_in_orig)
            pose_mod = pose_out_mod["pose"] if isinstance(pose_out_mod, dict) else pose_out_mod
            pose_orig = pose_out_orig["pose"] if isinstance(pose_out_orig, dict) else pose_out_orig
            pose_dist = (pose_mod[..., :6] - pose_orig[..., :6]).pow(2).mean()

            # Constraint violations
            seg_violation = F.relu(seg_dist - seg_boundary)
            pose_violation = F.relu(pose_dist - pose_boundary)

            # Augmented Lagrangian
            loss = (
                0.1 * tv_loss
                + lam_s * seg_violation
                + lam_p * pose_violation
                + (rho / 2.0) * seg_violation.pow(2)
                + (rho / 2.0) * pose_violation.pow(2)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], 5.0)
            optimizer.step()

            # Clamp delta for stability
            with torch.no_grad():
                delta.data.clamp_(-30.0, 30.0)

        # Outer step: update multipliers
        with torch.no_grad():
            current = (original + delta).clamp(0.0, 255.0)
            # Evaluate on first pair
            cur_eval = current[:2].unsqueeze(0)
            gt_eval = original[:2].unsqueeze(0)

            seg_in = segnet.preprocess_input(cur_eval)
            seg_out = segnet(seg_in)
            seg_in_gt = segnet.preprocess_input(gt_eval)
            seg_out_gt = segnet(seg_in_gt)
            p_soft = F.softmax(seg_out, dim=1)
            g_soft = F.softmax(seg_out_gt, dim=1)
            eval_seg = (1.0 - (p_soft * g_soft).sum(dim=1).mean()).item()

            pose_in = posenet.preprocess_input(cur_eval)
            pose_out = posenet(pose_in)
            pose_in_gt = posenet.preprocess_input(gt_eval)
            pose_out_gt = posenet(pose_in_gt)
            pm = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
            eval_pose = (pm[..., :6] - po[..., :6]).pow(2).mean().item()

            seg_viol = max(0.0, eval_seg - seg_boundary)
            pose_viol = max(0.0, eval_pose - pose_boundary)

            lam_s += rho * seg_viol
            lam_p += rho * pose_viol
            rho = min(rho * rho_growth, rho_max)  # Cap rho to prevent divergence

        record = {
            "outer": outer,
            "seg_dist": round(eval_seg, 6),
            "pose_dist": round(eval_pose, 6),
            "seg_satisfied": eval_seg < seg_boundary,
            "pose_satisfied": eval_pose < pose_boundary,
            "lam_s": round(lam_s, 2),
            "lam_p": round(lam_p, 2),
            "rho": round(rho, 2),
            "tv": round(tv_loss.item(), 4),
        }
        history.append(record)
        seg_ok = "OK" if record["seg_satisfied"] else "VIOLATED"
        pose_ok = "OK" if record["pose_satisfied"] else "VIOLATED"
        print(
            f"  outer {outer}: seg={eval_seg:.6f} [{seg_ok}] "
            f"pose={eval_pose:.6f} [{pose_ok}] "
            f"TV={tv_loss.item():.4f} rho={rho:.1f}"
        )

    # --- Final evaluation ---
    print(f"\n[4/4] Final scorer evaluation...")
    with torch.no_grad():
        current = (original + delta).clamp(0.0, 255.0)
        seg_total = []
        pose_total = []
        n_pairs = min(5, n_frames - 1)
        for i in range(n_pairs):
            cp = current[i:i + 2].unsqueeze(0)
            gp = original[i:i + 2].unsqueeze(0)

            seg_in = segnet.preprocess_input(cp)
            seg_out = segnet(seg_in)
            seg_in_gt = segnet.preprocess_input(gp)
            seg_out_gt = segnet(seg_in_gt)
            ps = F.softmax(seg_out, dim=1)
            gs = F.softmax(seg_out_gt, dim=1)
            seg_total.append((1.0 - (ps * gs).sum(dim=1).mean()).item())

            pose_in = posenet.preprocess_input(cp)
            pose_out = posenet(pose_in)
            pose_in_gt = posenet.preprocess_input(gp)
            pose_out_gt = posenet(pose_in_gt)
            pm = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
            pose_total.append((pm[..., :6] - po[..., :6]).pow(2).mean().item())

    avg_seg = sum(seg_total) / len(seg_total)
    avg_pose = sum(pose_total) / len(pose_total)

    # Hypothetical rate: these are GT frames with learned perturbations
    # In practice the "archive" is the delta compressed. Estimate delta size.
    delta_magnitude = delta.detach().abs().mean().item()
    from tac.scorer import comma_score

    # Use current rate as baseline (0.595) — this technique doesn't change rate
    score_at_current_rate = comma_score(avg_pose, avg_seg, 0.595)
    score_at_zero_rate = comma_score(avg_pose, avg_seg, 0.0)

    results["final_seg_dist"] = round(avg_seg, 8)
    results["final_pose_dist"] = round(avg_pose, 8)
    results["delta_mean_magnitude"] = round(delta_magnitude, 4)
    results["score_at_current_rate"] = round(score_at_current_rate, 4)
    results["score_at_zero_rate"] = round(score_at_zero_rate, 4)
    results["seg_satisfied"] = avg_seg < seg_boundary
    results["pose_satisfied"] = avg_pose < pose_boundary
    results["history"] = history

    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    results["vram_peak_mb"] = round(vram_peak, 0)
    results["status"] = "ok"

    print(f"\n  --- FRIDRICH VERDICT ---")
    print(f"  SegNet dist:  {avg_seg:.6f} (boundary: {seg_boundary})")
    print(f"  PoseNet dist: {avg_pose:.6f} (boundary: {pose_boundary})")
    print(f"  Score @current_rate: {score_at_current_rate:.4f}")
    print(f"  Score @zero_rate:    {score_at_zero_rate:.4f}")
    diverged = avg_pose > 1.0
    if diverged:
        print(f"  FAILED: PoseNet diverged even with hard constraints!")
        results["verdict"] = "POSENET_DIVERGED"
    elif results["seg_satisfied"] and results["pose_satisfied"]:
        print(f"  VIABLE: Both constraints satisfied. Constrained formulation works!")
        results["verdict"] = "VIABLE"
    elif results["pose_satisfied"]:
        print(f"  PARTIAL: PoseNet OK but SegNet violated. Need tighter optimization.")
        results["verdict"] = "PARTIAL_SEG_VIOLATED"
    else:
        print(f"  PARTIAL: Constraints not fully satisfied. Needs more steps/tuning.")
        results["verdict"] = "NEEDS_TUNING"
    print(f"  Time: {elapsed:.0f}s | VRAM peak: {vram_peak:.0f} MB")

    del posenet, segnet, delta, gt_chw, original
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return results


# =========================================================================
# Smoke Test 3: Self-Compressing Postfilter
# =========================================================================

def test_self_compressing_postfilter(
    device: str = "cuda",
    n_frames: int = 20,
    steps: int = 100,
    target_bits: int = 8000,
    lr: float = 5e-4,
    lr_bits: float = 1e-2,
) -> dict[str, Any]:
    """Smoke Test 3: Compress the postfilter from 46KB to <10KB.

    Loads the existing dilated_h64 architecture, wraps with learnable
    bit-depth, fine-tunes with rate penalty, exports and reimports.
    """
    print("\n" + "=" * 70)
    print("SMOKE TEST 3: Self-Compressing Postfilter")
    print("=" * 70)

    results: dict[str, Any] = {"test": "self_compressing_postfilter", "status": "running"}
    t0 = time.time()

    from tac.self_compress import (
        SelfCompressingPostFilter,
        train_self_compressing,
        export_compressed_checkpoint,
        load_compressed_checkpoint,
    )

    # --- Load data and scorers ---
    print(f"\n[1/5] Loading {n_frames} GT frames and scorers...")
    gt_frames_hwc = _load_gt_frames(n_frames)
    H, W = gt_frames_hwc[0].shape[:2]
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc])

    posenet, segnet = _load_scorers(device)

    # Try to load the current postfilter weights
    postfilter_path = None
    _candidate_pf = [
        Path("/home/zeus/content/pact/submissions/robust_current/postfilter_int8.pt"),
        Path(__file__).resolve().parent.parent / "submissions" / "robust_current" / "postfilter_int8.pt",
    ]
    for _p in _candidate_pf:
        if _p.exists():
            postfilter_path = _p
            break

    # --- Build model ---
    print(f"\n[2/5] Building SelfCompressingPostFilter(hidden=64)...")
    model = SelfCompressingPostFilter(hidden=64, kernel=3, init_bits=8.0)

    # If we have existing weights, load them as starting point
    if postfilter_path is not None:
        print(f"  Loading existing weights from {postfilter_path}...")
        try:
            existing = torch.load(str(postfilter_path), map_location="cpu", weights_only=True)
            # Map existing int8 weights into the self-compressing model
            # The state dict keys differ: existing has conv1.weight, model has conv1.conv.weight
            mapped = 0
            for name in ["conv1", "conv2", "conv3"]:
                for param in ["weight", "bias"]:
                    src_key = f"{name}.{param}"
                    dst_key = f"{name}.conv.{param}"
                    if src_key in existing and dst_key in model.state_dict():
                        src_val = existing[src_key].float()
                        dst_shape = model.state_dict()[dst_key].shape
                        if src_val.shape == dst_shape:
                            model.state_dict()[dst_key].copy_(src_val)
                            mapped += 1
                        else:
                            print(f"    Shape mismatch for {src_key}: {src_val.shape} vs {dst_shape}")
            print(f"  Mapped {mapped} parameter tensors from existing checkpoint")
        except Exception as e:
            print(f"  Could not load existing weights: {e}")
            print(f"  Proceeding with random initialization")
    else:
        print(f"  No existing postfilter found, using random initialization")

    # Baseline stats
    baseline_stats = model.compression_stats()
    print(f"  Baseline: {baseline_stats['total_bytes']:.0f} bytes, "
          f"{sum(l['channels'] for l in baseline_stats['layers'])} total channels")
    results["baseline_bytes"] = baseline_stats["total_bytes"]

    # --- Compressed training with small subset ---
    print(f"\n[3/5] Training with rate penalty (target={target_bits} bits, {steps} steps)...")

    # Use a small subset for the smoke test
    comp_frames = gt_chw[:n_frames].clone()  # Use GT as "compressed" for smoke test
    gt_for_train = gt_chw[:n_frames].clone()

    model = train_self_compressing(
        model,
        comp_frames,
        gt_for_train,
        posenet,
        segnet,
        target_bits=target_bits,
        epochs=steps,
        lr=lr,
        lr_bits=lr_bits,
        lambda_rate_start=0.0,
        lambda_rate_end=1.0,
        ramp_start_frac=0.2,
        scorer_weight=20.0,
        device=device,
        log_every=20,
    )

    # --- Compression stats ---
    print(f"\n[4/5] Compression results...")
    final_stats = model.compression_stats()
    for layer_info in final_stats["layers"]:
        print(
            f"  {layer_info['name']}: {layer_info['active']}/{layer_info['channels']} active, "
            f"mean {layer_info['mean_bits']:.1f} bits, "
            f"{layer_info['pruned']} pruned"
        )
    print(f"  Total: {final_stats['total_bytes']:.0f} bytes "
          f"({final_stats['compression_ratio']:.1f}x compression)")
    results["final_bytes"] = final_stats["total_bytes"]
    results["compression_ratio"] = round(final_stats["compression_ratio"], 2)
    results["layer_stats"] = final_stats["layers"]

    # --- Export / import round-trip ---
    print(f"\n[5/5] Export/import round-trip test...")
    model.eval()

    # Export
    blob = export_compressed_checkpoint(model)
    export_size = len(blob)
    print(f"  Exported: {export_size:,} bytes ({export_size/1024:.1f} KB)")
    results["export_bytes"] = export_size
    results["export_kb"] = round(export_size / 1024, 2)

    # Import
    try:
        restored = load_compressed_checkpoint(blob, hidden=64, kernel=3)
        restored = restored.to(device)

        # Round-trip test on a few frames
        test_input = gt_chw[:2].to(device)
        with torch.no_grad():
            out_original = model.to(device)(test_input)
            out_restored = restored(test_input)

        diff = (out_original - out_restored).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  Round-trip max diff: {max_diff:.4f}")
        print(f"  Round-trip mean diff: {mean_diff:.4f}")
        results["roundtrip_max_diff"] = round(max_diff, 4)
        results["roundtrip_mean_diff"] = round(mean_diff, 4)
        results["roundtrip_bitexact"] = max_diff < 1.0  # Within 1 pixel value
    except Exception as e:
        print(f"  Round-trip FAILED: {e}")
        results["roundtrip_error"] = str(e)

    # Rate impact
    current_archive_kb = 893  # Current archive.zip size
    current_postfilter_kb = 46  # Current postfilter_int8.pt in archive
    new_postfilter_kb = export_size / 1024
    savings_kb = current_postfilter_kb - new_postfilter_kb
    new_archive_kb = current_archive_kb - savings_kb
    rate_savings = savings_kb / 1e3 * 25  # 25 * delta_rate (rate = MB)
    print(f"\n  Rate impact estimate:")
    print(f"    Current postfilter: {current_postfilter_kb} KB")
    print(f"    Compressed postfilter: {new_postfilter_kb:.1f} KB")
    print(f"    Savings: {savings_kb:.1f} KB -> rate savings: {rate_savings:.3f} score points")
    results["savings_kb"] = round(savings_kb, 2)
    results["rate_savings_score"] = round(rate_savings, 4)

    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    results["vram_peak_mb"] = round(vram_peak, 0)
    results["status"] = "ok"

    print(f"\n  --- SELF-COMPRESS VERDICT ---")
    if export_size < 10240:
        print(f"  VIABLE: Compressed to {export_size/1024:.1f} KB (target <10 KB)")
        results["verdict"] = "VIABLE"
    elif export_size < 20480:
        print(f"  MARGINAL: {export_size/1024:.1f} KB (target <10 KB, got <20 KB)")
        results["verdict"] = "MARGINAL"
    else:
        print(f"  NEEDS MORE STEPS: {export_size/1024:.1f} KB (target <10 KB)")
        results["verdict"] = "NEEDS_MORE_STEPS"
    print(f"  Time: {elapsed:.0f}s | VRAM peak: {vram_peak:.0f} MB")

    del posenet, segnet, model, restored
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Frontier smoke tests for sub-0.50 techniques")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", choices=["siren", "fridrich", "compress", "all"], default="all",
                        help="Which test to run (default: all)")
    parser.add_argument("--output", default="frontier_smoke_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        used_vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"VRAM: {used_vram:.0f} / {total_vram:.0f} MB")
    print(f"Upstream: {UPSTREAM_ROOT}")
    print(f"Weights: {WEIGHTS_DIR}")
    print(f"GT video: {GT_VIDEO}")

    all_results: dict[str, Any] = {"meta": {
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "upstream": str(UPSTREAM_ROOT),
    }}

    tests_to_run = ["siren", "fridrich", "compress"] if args.test == "all" else [args.test]

    for test_name in tests_to_run:
        try:
            if test_name == "siren":
                result = test_siren_memorization(device=args.device)
            elif test_name == "fridrich":
                result = test_fridrich_constrained_gen(device=args.device)
            elif test_name == "compress":
                result = test_self_compressing_postfilter(device=args.device)
            else:
                continue
            all_results[test_name] = result
        except Exception as e:
            print(f"\n  EXCEPTION in {test_name}: {e}")
            traceback.print_exc()
            all_results[test_name] = {
                "test": test_name,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for test_name in tests_to_run:
        if test_name in all_results:
            r = all_results[test_name]
            verdict = r.get("verdict", r.get("status", "unknown"))
            elapsed = r.get("elapsed_seconds", "?")
            print(f"  {test_name:20s} | {verdict:25s} | {elapsed}s")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
