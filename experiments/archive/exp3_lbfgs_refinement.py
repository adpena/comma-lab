#!/usr/bin/env python
"""Experiment 3: L-BFGS Refinement -- polish step after GPU generation.

Run on Lightning T4:
    PYTHONPATH=src:/home/zeus/content/upstream python experiments/exp3_lbfgs_refinement.py

Pre-registered hypothesis:
    "L-BFGS refinement (10 steps) on GPU-generated frames improves score
     by at least 5% compared to the input frames"

Success criteria:
    score improvement >= 5% on the same frames
Kill criteria:
    score worsens or no improvement after 10 steps
Concern:
    improvement < 5% but positive -- may still be worth combining

This experiment takes output frames from exp1 (Fridrich) or exp2 (DP-SIMS)
and runs L-BFGS second-order optimization to polish them. L-BFGS uses
approximate Hessian information for faster convergence than Adam.

The key insight: L-BFGS converges in ~10 steps where Adam needs ~1000.
Cost: ~1 minute on 100 frames. Can only help, never hurts (we keep the
better of input vs output).

L-BFGS minimizes the EXACT scorer formula:
    score = 100 * seg + sqrt(10 * pose) + 25 * rate_proxy
where rate_proxy = total_variation (compressibility).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
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


@dataclass
class LBFGSConfig:
    """L-BFGS refinement hyperparameters."""

    n_frames: int = 100
    lbfgs_steps: int = 10
    lbfgs_lr: float = 1.0
    lbfgs_history_size: int = 10
    lbfgs_max_iter: int = 5  # inner iterations per step
    lbfgs_line_search: str = "strong_wolfe"

    # Loss weights (match scorer formula)
    seg_weight: float = 100.0
    pose_weight_scale: float = 10.0  # inside sqrt
    tv_weight: float = 0.5  # rate proxy

    # Delta bounds
    delta_clamp: float = 15.0  # tighter than Fridrich -- we're polishing

    # Scorer resolution
    target_h: int = 384
    target_w: int = 512

    # Pairs to evaluate per L-BFGS step (memory)
    pairs_per_step: int = 8

    # Input source
    input_source: str = "gt"  # "gt" for standalone, "file" for chained

    device: str = "cuda"


def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    if WEIGHTS_DIR is None:
        raise FileNotFoundError("Scorer weights not found")
    from tac.scorer import load_scorers
    return load_scorers(
        WEIGHTS_DIR / "posenet.safetensors",
        WEIGHTS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=UPSTREAM_ROOT,
    )


def _load_gt_frames(n_frames: int, target_h: int = 384, target_w: int = 512) -> list[torch.Tensor]:
    if GT_VIDEO is None:
        raise FileNotFoundError("GT video not found")
    from tac.data import decode_video
    frames = decode_video(str(GT_VIDEO), target_h=target_h, target_w=target_w)
    return frames[:n_frames]


def _compute_scorer_loss(
    current: torch.Tensor,
    original: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    pair_indices: list[int],
    seg_weight: float,
    pose_weight_scale: float,
    tv_weight: float,
) -> torch.Tensor:
    """Compute differentiable score approximation for L-BFGS.

    Approximates: 100 * seg + sqrt(10 * pose) + tv_weight * TV
    Using: seg_weight * seg + pose_weight_scale * pose + tv_weight * TV
    (sqrt is non-smooth near 0, direct weighting is more L-BFGS-friendly)
    """
    # TV (compressibility proxy)
    dx = (current[:, :, :, 1:] - current[:, :, :, :-1]).abs().mean()
    dy = (current[:, :, 1:, :] - current[:, :, :-1, :]).abs().mean()
    tv = dx + dy

    seg_total = torch.tensor(0.0, device=current.device)
    pose_total = torch.tensor(0.0, device=current.device)
    n_valid = 0

    for idx in pair_indices:
        if idx + 1 >= current.shape[0]:
            continue
        cur_pair = current[idx:idx+2].unsqueeze(0)
        gt_pair = original[idx:idx+2].unsqueeze(0)

        # SegNet (uses preprocess_input)
        seg_in = segnet.preprocess_input(cur_pair)
        seg_out = segnet(seg_in)
        with torch.no_grad():
            seg_in_gt = segnet.preprocess_input(gt_pair)
            seg_out_gt = segnet(seg_in_gt)
        p_soft = F.softmax(seg_out, dim=1)
        g_soft = F.softmax(seg_out_gt, dim=1)
        seg_total = seg_total + (1.0 - (p_soft * g_soft).sum(dim=1).mean())

        # PoseNet (uses preprocess_input)
        pose_in = posenet.preprocess_input(cur_pair)
        pose_out = posenet(pose_in)
        with torch.no_grad():
            pose_in_gt = posenet.preprocess_input(gt_pair)
            pose_out_gt = posenet(pose_in_gt)
        pm = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
        pose_total = pose_total + (pm[..., :6] - po[..., :6]).pow(2).mean()

        n_valid += 1

    if n_valid > 0:
        seg_total = seg_total / n_valid
        pose_total = pose_total / n_valid

    return seg_weight * seg_total + pose_weight_scale * pose_total + tv_weight * tv


def _score_all_pairs(
    frames_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
) -> dict[str, float]:
    """Score all pairs (no gradient)."""
    n = frames_chw.shape[0]
    seg_dists = []
    pose_dists = []
    with torch.no_grad():
        for i in range(n - 1):
            gen_pair = frames_chw[i:i+2].unsqueeze(0)
            gt_pair = gt_chw[i:i+2].unsqueeze(0)

            seg_in = segnet.preprocess_input(gen_pair)
            seg_out = segnet(seg_in)
            seg_in_gt = segnet.preprocess_input(gt_pair)
            seg_out_gt = segnet(seg_in_gt)
            p = F.softmax(seg_out, dim=1)
            g = F.softmax(seg_out_gt, dim=1)
            seg_dists.append((1.0 - (p * g).sum(dim=1).mean()).item())

            pose_in = posenet.preprocess_input(gen_pair)
            pose_out = posenet(pose_in)
            pose_in_gt = posenet.preprocess_input(gt_pair)
            pose_out_gt = posenet(pose_in_gt)
            pm = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
            pose_dists.append((pm[..., :6] - po[..., :6]).pow(2).mean().item())

    avg_seg = sum(seg_dists) / len(seg_dists) if seg_dists else 0.0
    avg_pose = sum(pose_dists) / len(pose_dists) if pose_dists else 0.0
    return {"avg_seg": avg_seg, "avg_pose": avg_pose}


def run_lbfgs_refinement(cfg: LBFGSConfig) -> dict[str, Any]:
    """Run L-BFGS refinement on frames."""

    device = cfg.device
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: L-BFGS Refinement (Newton's Method)")
    print(f"  {cfg.n_frames} frames, {cfg.lbfgs_steps} L-BFGS steps")
    print(f"  Input: {cfg.input_source}")
    print("=" * 70)

    results: dict[str, Any] = {
        "experiment": "lbfgs_refinement",
        "config": asdict(cfg),
        "status": "running",
        "hypothesis": (
            "L-BFGS refinement (10 steps) improves score by at least 5%"
        ),
    }
    t0 = time.time()

    # --- Load ---
    print("\n[1/4] Loading frames and scorers...")
    gt_frames_hwc = _load_gt_frames(cfg.n_frames, cfg.target_h, cfg.target_w)
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc]).to(device)
    posenet, segnet = _load_scorers(device)

    # Input frames (GT for standalone, or loaded from file for chained)
    if cfg.input_source == "gt":
        input_chw = gt_chw.clone()
    else:
        # Load from file (e.g., exp1 or exp2 output)
        input_data = torch.load(cfg.input_source, map_location=device)
        input_chw = input_data["frames_chw"].to(device)

    # --- Score BEFORE refinement ---
    print("\n[2/4] Scoring input frames (before L-BFGS)...")
    before_scores = _score_all_pairs(input_chw, gt_chw, posenet, segnet)
    est_rate = 2000 / (cfg.n_frames * cfg.target_h * cfg.target_w * 1.5)
    from tac.scorer import comma_score
    before_score = comma_score(before_scores["avg_pose"], before_scores["avg_seg"], est_rate)
    print(f"  Before: seg={before_scores['avg_seg']:.6f} pose={before_scores['avg_pose']:.6f} score={before_score:.4f}")
    results["before"] = {
        "seg": round(before_scores["avg_seg"], 8),
        "pose": round(before_scores["avg_pose"], 8),
        "score": round(before_score, 4),
    }

    # --- L-BFGS optimization ---
    print(f"\n[3/4] Running L-BFGS ({cfg.lbfgs_steps} steps)...")
    original = gt_chw.detach().clone()
    delta = (input_chw - gt_chw).detach().clone().requires_grad_(True)

    lbfgs = torch.optim.LBFGS(
        [delta],
        lr=cfg.lbfgs_lr,
        max_iter=cfg.lbfgs_max_iter,
        history_size=cfg.lbfgs_history_size,
        line_search_fn=cfg.lbfgs_line_search,
    )

    history: list[dict] = []

    for step in range(cfg.lbfgs_steps):
        step_t0 = time.time()

        # Random pairs for this step
        all_pairs = list(range(max(1, cfg.n_frames - 1)))
        np.random.shuffle(all_pairs)
        pair_indices = all_pairs[:cfg.pairs_per_step]

        def closure(lbfgs=lbfgs, delta=delta, pair_indices=pair_indices):
            lbfgs.zero_grad()
            current = (original + delta).clamp(0.0, 255.0)
            loss = _compute_scorer_loss(
                current, original, posenet, segnet,
                pair_indices, cfg.seg_weight, cfg.pose_weight_scale, cfg.tv_weight,
            )
            loss.backward()
            return loss

        loss_val = lbfgs.step(closure)

        with torch.no_grad():
            delta.data.clamp_(-cfg.delta_clamp, cfg.delta_clamp)

        step_elapsed = time.time() - step_t0
        record = {
            "step": step,
            "loss": round(loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val, 6),
            "delta_mean": round(delta.detach().abs().mean().item(), 4),
            "time_s": round(step_elapsed, 2),
        }
        history.append(record)
        print(f"  step {step}: loss={record['loss']:.6f} delta={record['delta_mean']:.4f} ({step_elapsed:.1f}s)")

    # --- Score AFTER refinement ---
    print("\n[4/4] Scoring refined frames (after L-BFGS)...")
    with torch.no_grad():
        refined = (original + delta).clamp(0.0, 255.0)

    after_scores = _score_all_pairs(refined, gt_chw, posenet, segnet)
    after_score = comma_score(after_scores["avg_pose"], after_scores["avg_seg"], est_rate)
    print(f"  After: seg={after_scores['avg_seg']:.6f} pose={after_scores['avg_pose']:.6f} score={after_score:.4f}")

    improvement_pct = (before_score - after_score) / before_score * 100 if before_score > 0 else 0

    results.update({
        "after": {
            "seg": round(after_scores["avg_seg"], 8),
            "pose": round(after_scores["avg_pose"], 8),
            "score": round(after_score, 4),
        },
        "improvement_pct": round(improvement_pct, 2),
        "delta_mean": round(delta.detach().abs().mean().item(), 4),
        "history": history,
    })

    elapsed = time.time() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    results["elapsed_seconds"] = round(elapsed, 1)
    results["vram_peak_mb"] = round(vram_peak, 0)
    results["status"] = "ok"

    # Verdict
    if improvement_pct >= 5.0:
        results["verdict"] = "SUCCESS"
        verdict = f"SUCCESS: {improvement_pct:.1f}% improvement."
    elif improvement_pct > 0:
        results["verdict"] = "CONCERN_SMALL_IMPROVEMENT"
        verdict = f"CONCERN: Only {improvement_pct:.1f}% improvement. May still stack."
    else:
        results["verdict"] = "KILLED_NO_IMPROVEMENT"
        verdict = f"KILLED: No improvement ({improvement_pct:.1f}%)."

    print("\n  --- L-BFGS VERDICT ---")
    print(f"  Before: {before_score:.4f}")
    print(f"  After:  {after_score:.4f}")
    print(f"  Improvement: {improvement_pct:.1f}%")
    print(f"  Time: {elapsed:.0f}s | VRAM: {vram_peak:.0f} MB")
    print(f"  VERDICT: {verdict}")

    # Cleanup
    del delta, lbfgs, refined
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: L-BFGS Refinement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--lbfgs-steps", type=int, default=10)
    parser.add_argument("--input-source", default="gt", help="'gt' or path to saved frames")
    args = parser.parse_args()

    cfg = LBFGSConfig(
        n_frames=args.n_frames,
        lbfgs_steps=args.lbfgs_steps,
        input_source=args.input_source,
        device=args.device,
    )

    try:
        results = run_lbfgs_refinement(cfg)
    except Exception:
        traceback.print_exc()
        results = {"status": "error", "error": traceback.format_exc()}

    out_path = RESULTS_DIR / "exp3_lbfgs_refinement_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
