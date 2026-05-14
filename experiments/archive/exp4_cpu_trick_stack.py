#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Experiment 4: CPU Trick Stacking -- score every trick independently and stacked.

Run on Lightning T4:
    PYTHONPATH=src:/home/zeus/content/upstream python experiments/exp4_cpu_trick_stack.py

Pre-registered hypothesis:
    "Stacking CRF sweep + TTO + multi-pass + deblock on Lightning checkpoint
     reduces proxy score by at least 10% compared to baseline"

Success criteria:
    best stacked proxy < 0.83 (10% below 0.9238)
Kill criteria:
    no trick improves proxy independently
Concern:
    individual tricks help but don't compound

Tricks to test (each independently then stacked):
    1. CRF sweep (32-38) -- codec-level rate-distortion tradeoff
    2. Quantization-directed rounding -- precompute at compress time
    3. TTO at inflate time -- self-supervised, 10 steps, 30s budget
    4. Multi-pass (2 postfilter passes) -- iterative refinement
    5. Deblock (OpenCV NLM) -- post-processing

This experiment does NOT run auth eval (that requires bat00 or Lightning scorer).
It measures proxy scores using the local scorer.
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

_CANDIDATE_CHECKPOINTS = [
    Path("/home/zeus/content/pact/submissions/robust_current/postfilter_int8.pt"),
    Path(__file__).resolve().parent.parent / "submissions" / "robust_current" / "postfilter_int8.pt",
]
CHECKPOINT_PATH: Path | None = None
for _p in _CANDIDATE_CHECKPOINTS:
    if _p is not None and _p.exists():
        CHECKPOINT_PATH = _p
        break

RESULTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrickStackTestConfig:
    """Configuration for trick stacking evaluation."""

    n_frames: int = 100
    target_h: int = 384
    target_w: int = 512

    # CRF sweep values
    crf_values: list = field(default_factory=lambda: [32, 33, 34, 35, 36, 37, 38])

    # TTO config
    tto_steps: int = 10
    tto_lr: float = 1e-4
    tto_budget: float = 30.0

    # Multi-pass
    multi_pass_counts: list = field(default_factory=lambda: [1, 2, 3])

    # Deblock
    deblock_h: int = 10  # NLM filter strength
    deblock_template_size: int = 7
    deblock_search_size: int = 21

    device: str = "cuda"


def _load_scorers(device: str) -> tuple[nn.Module, nn.Module]:
    if WEIGHTS_DIR is None:
        raise FileNotFoundError(f"Scorer weights not found")
    from tac.scorer import load_scorers
    return load_scorers(
        WEIGHTS_DIR / "posenet.safetensors",
        WEIGHTS_DIR / "segnet.safetensors",
        device=device,
        upstream_dir=UPSTREAM_ROOT,
    )


def _load_gt_frames(n_frames: int, target_h: int = 384, target_w: int = 512) -> list[torch.Tensor]:
    if GT_VIDEO is None:
        raise FileNotFoundError(f"GT video not found")
    from tac.data import decode_video
    frames = decode_video(str(GT_VIDEO), target_h=target_h, target_w=target_w)
    return frames[:n_frames]


def _score_frames(
    frames_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
    n_pairs: int = 20,
) -> dict[str, float]:
    """Score a subset of pairs for quick evaluation."""
    N = frames_chw.shape[0]
    max_pairs = min(n_pairs, N - 1)
    # Evenly spaced pairs
    indices = list(range(0, N - 1, max(1, (N - 1) // max_pairs)))[:max_pairs]

    seg_dists = []
    pose_dists = []

    with torch.no_grad():
        for i in indices:
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


# ---------------------------------------------------------------------------
# Individual trick implementations
# ---------------------------------------------------------------------------

def trick_crf_compress(
    gt_chw: torch.Tensor,
    crf: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor | None:
    """Compress GT with FFmpeg at given CRF, decode back to tensor.

    Returns (N, 3, H, W) tensor or None if FFmpeg unavailable.
    """
    import tempfile
    import subprocess

    try:
        # Save frames to temp video
        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            tmp_in = f.name
        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            tmp_out = f.name

        # Write raw frames to y4m pipe -> ffmpeg
        N, C, H, W = gt_chw.shape
        frames_np = gt_chw.permute(0, 2, 3, 1).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Use ffmpeg to encode at specified CRF
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}",
                "-r", "20",
                "-i", "pipe:0",
                "-c:v", "libx265",
                "-preset", "medium",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                tmp_out,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=frames_np.tobytes())

        if proc.returncode != 0:
            return None

        # Get compressed size
        compressed_size = os.path.getsize(tmp_out)

        # Decode back
        from tac.data import decode_video
        decoded = decode_video(tmp_out, target_h=H, target_w=W)
        if len(decoded) < N:
            return None

        result = torch.stack([f.permute(2, 0, 1).float() for f in decoded[:N]])
        result._crf_compressed_bytes = compressed_size  # stash for rate calc

        return result

    except Exception as e:
        print(f"  CRF {crf} failed: {e}")
        return None
    finally:
        for p in [tmp_in, tmp_out]:
            try:
                os.unlink(p)
            except OSError:
                pass


def trick_tto(
    model: nn.Module,
    frames_chw: torch.Tensor,
    n_steps: int = 10,
    lr: float = 1e-4,
    budget: float = 30.0,
) -> nn.Module:
    """Test-time optimization: adapt model to specific content."""
    from tac.tto import test_time_optimize
    model = test_time_optimize(
        model=model,
        frames=frames_chw,
        n_steps=n_steps,
        lr=lr,
        loss_type="temporal_consistency",
        wall_clock_budget=budget,
    )
    return model


def trick_multi_pass(
    model: nn.Module,
    frames_chw: torch.Tensor,
    n_passes: int = 2,
) -> torch.Tensor:
    """Run model multiple times, quantizing to uint8 between passes."""
    current = frames_chw.clone()
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(current)
            # Quantize to uint8 (simulate real pipeline)
            current = output.round().clamp(0, 255)
    return current


def trick_deblock(
    frames_chw: torch.Tensor,
    h: int = 10,
    template_size: int = 7,
    search_size: int = 21,
) -> torch.Tensor:
    """Apply non-local means deblocking filter."""
    try:
        import cv2
    except ImportError:
        print("  cv2 not available, skipping deblock")
        return frames_chw

    result = []
    frames_np = frames_chw.permute(0, 2, 3, 1).cpu().numpy().clip(0, 255).astype(np.uint8)
    for frame in frames_np:
        deblocked = cv2.fastNlMeansDenoisingColored(frame, None, h, h, template_size, search_size)
        result.append(torch.from_numpy(deblocked).permute(2, 0, 1).float())

    return torch.stack(result).to(frames_chw.device)


def trick_quantization_rounding(
    frames_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    posenet: nn.Module,
    segnet: nn.Module,
) -> torch.Tensor:
    """Gradient-directed rounding: round each pixel to minimize scorer error.

    For pixels at x.5, choose floor or ceil based on which direction
    reduces scorer gradient magnitude. Precomputed at compress time.
    """
    # Compute scorer gradient at each pixel
    frames_grad = frames_chw.detach().clone().requires_grad_(True)

    # Compute loss on a few representative pairs
    loss = torch.tensor(0.0, device=frames_chw.device)
    n_pairs = min(10, frames_chw.shape[0] - 1)
    for i in range(0, n_pairs):
        pair = frames_grad[i:i+2].unsqueeze(0)
        gt_pair = gt_chw[i:i+2].unsqueeze(0)

        seg_in = segnet.preprocess_input(pair)
        seg_out = segnet(seg_in)
        with torch.no_grad():
            seg_in_gt = segnet.preprocess_input(gt_pair)
            seg_out_gt = segnet(seg_in_gt)
        p = F.softmax(seg_out, dim=1)
        g = F.softmax(seg_out_gt, dim=1)
        loss = loss + (1.0 - (p * g).sum(dim=1).mean())

        pose_in = posenet.preprocess_input(pair)
        pose_out = posenet(pose_in)
        with torch.no_grad():
            pose_in_gt = posenet.preprocess_input(gt_pair)
            pose_out_gt = posenet(pose_in_gt)
        pm = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        po = pose_out_gt["pose"] if isinstance(pose_out_gt, dict) else pose_out_gt
        loss = loss + (pm[..., :6] - po[..., :6]).pow(2).mean()

    loss.backward()
    grad = frames_grad.grad.detach()

    # Directed rounding: if gradient is positive, floor. If negative, ceil.
    with torch.no_grad():
        raw = frames_chw.detach()
        floor_val = raw.floor()
        ceil_val = raw.ceil().clamp(0, 255)
        # Where gradient points up (positive), rounding down helps
        rounded = torch.where(grad > 0, floor_val, ceil_val)
        return rounded.clamp(0, 255)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_trick_stack(cfg: TrickStackTestConfig) -> dict[str, Any]:
    """Run all tricks independently and stacked, measuring each."""

    device = cfg.device
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: CPU Trick Stacking")
    print(f"  {cfg.n_frames} frames")
    print(f"  CRF values: {cfg.crf_values}")
    print(f"  TTO steps: {cfg.tto_steps}")
    print(f"  Multi-pass counts: {cfg.multi_pass_counts}")
    print("=" * 70)

    results: dict[str, Any] = {
        "experiment": "cpu_trick_stack",
        "config": {k: str(v) if isinstance(v, (list, tuple)) else v for k, v in asdict(cfg).items()},
        "status": "running",
        "hypothesis": (
            "Stacking CRF + TTO + multi-pass on Lightning checkpoint "
            "reduces proxy by >= 10%"
        ),
    }
    t0 = time.time()

    # --- Load ---
    print(f"\n[1/7] Loading GT frames and scorers...")
    gt_frames_hwc = _load_gt_frames(cfg.n_frames, cfg.target_h, cfg.target_w)
    gt_chw = torch.stack([f.permute(2, 0, 1).float() for f in gt_frames_hwc]).to(device)
    posenet, segnet = _load_scorers(device)

    # --- Baseline score (GT vs GT = 0, but we measure to verify) ---
    print(f"\n[2/7] Baseline scoring...")
    baseline_scores = _score_frames(gt_chw, gt_chw, posenet, segnet)
    print(f"  GT vs GT: seg={baseline_scores['avg_seg']:.6f} pose={baseline_scores['avg_pose']:.6f}")
    results["baseline"] = baseline_scores

    # --- Trick 1: CRF sweep ---
    print(f"\n[3/7] CRF sweep...")
    crf_results = {}
    best_crf = None
    best_crf_score = float("inf")

    for crf in cfg.crf_values:
        print(f"\n  Testing CRF={crf}...")
        compressed = trick_crf_compress(gt_chw, crf, cfg.target_h, cfg.target_w)
        if compressed is None:
            crf_results[f"crf_{crf}"] = {"status": "failed"}
            continue

        compressed_dev = compressed.to(device)
        scores = _score_frames(compressed_dev, gt_chw, posenet, segnet)

        compressed_bytes = getattr(compressed, '_crf_compressed_bytes', 0)
        rate = compressed_bytes / (cfg.n_frames * cfg.target_h * cfg.target_w * 1.5) if compressed_bytes else 0

        from tac.scorer import comma_score
        total = comma_score(scores["avg_pose"], scores["avg_seg"], rate)

        entry = {
            "seg": round(scores["avg_seg"], 8),
            "pose": round(scores["avg_pose"], 8),
            "rate": round(rate, 6),
            "compressed_bytes": compressed_bytes,
            "score": round(total, 4),
            "status": "ok",
        }
        crf_results[f"crf_{crf}"] = entry
        print(f"    seg={entry['seg']:.6f} pose={entry['pose']:.6f} rate={rate:.6f} score={total:.4f}")

        if total < best_crf_score:
            best_crf_score = total
            best_crf = crf

        del compressed_dev
        gc.collect()

    results["crf_sweep"] = crf_results
    results["best_crf"] = best_crf
    print(f"\n  Best CRF: {best_crf} (score={best_crf_score:.4f})")

    # --- Trick 2: Quantization-directed rounding ---
    print(f"\n[4/7] Quantization-directed rounding...")
    try:
        rounded = trick_quantization_rounding(gt_chw, gt_chw, posenet, segnet)
        round_scores = _score_frames(rounded, gt_chw, posenet, segnet)
        results["quantization_rounding"] = {
            "seg": round(round_scores["avg_seg"], 8),
            "pose": round(round_scores["avg_pose"], 8),
            "status": "ok",
        }
        print(f"  seg={round_scores['avg_seg']:.6f} pose={round_scores['avg_pose']:.6f}")
        del rounded
    except Exception as e:
        results["quantization_rounding"] = {"status": "error", "error": str(e)}
        print(f"  Failed: {e}")

    # --- Trick 3: TTO ---
    print(f"\n[5/7] Test-time optimization ({cfg.tto_steps} steps)...")
    tto_result = {"status": "skipped", "reason": "requires postfilter model"}
    if CHECKPOINT_PATH is not None:
        try:
            model = torch.jit.load(str(CHECKPOINT_PATH), map_location=device)
            model.eval()
            model = trick_tto(model, gt_chw, cfg.tto_steps, cfg.tto_lr, cfg.tto_budget)

            # Run model on GT to get postfiltered output
            with torch.no_grad():
                tto_output = model(gt_chw)
            tto_scores = _score_frames(tto_output, gt_chw, posenet, segnet)
            tto_result = {
                "seg": round(tto_scores["avg_seg"], 8),
                "pose": round(tto_scores["avg_pose"], 8),
                "status": "ok",
            }
            print(f"  seg={tto_scores['avg_seg']:.6f} pose={tto_scores['avg_pose']:.6f}")
            del model, tto_output
        except Exception as e:
            tto_result = {"status": "error", "error": str(e)}
            print(f"  Failed: {e}")
    else:
        print(f"  Skipped: no checkpoint found")
    results["tto"] = tto_result

    # --- Trick 4: Multi-pass ---
    print(f"\n[6/7] Multi-pass inference...")
    multi_pass_results = {}
    if CHECKPOINT_PATH is not None:
        try:
            model = torch.jit.load(str(CHECKPOINT_PATH), map_location=device)
            model.eval()

            for n_passes in cfg.multi_pass_counts:
                mp_output = trick_multi_pass(model, gt_chw, n_passes)
                mp_scores = _score_frames(mp_output, gt_chw, posenet, segnet)
                entry = {
                    "seg": round(mp_scores["avg_seg"], 8),
                    "pose": round(mp_scores["avg_pose"], 8),
                    "status": "ok",
                }
                multi_pass_results[f"pass_{n_passes}"] = entry
                print(f"  {n_passes} passes: seg={entry['seg']:.6f} pose={entry['pose']:.6f}")
                del mp_output

            del model
        except Exception as e:
            multi_pass_results["error"] = str(e)
            print(f"  Failed: {e}")
    else:
        multi_pass_results["status"] = "skipped"
        print(f"  Skipped: no checkpoint")
    results["multi_pass"] = multi_pass_results

    # --- Trick 5: Deblock ---
    print(f"\n[7/7] Deblock (NLM)...")
    try:
        deblocked = trick_deblock(gt_chw, cfg.deblock_h, cfg.deblock_template_size, cfg.deblock_search_size)
        db_scores = _score_frames(deblocked, gt_chw, posenet, segnet)
        results["deblock"] = {
            "seg": round(db_scores["avg_seg"], 8),
            "pose": round(db_scores["avg_pose"], 8),
            "status": "ok",
        }
        print(f"  seg={db_scores['avg_seg']:.6f} pose={db_scores['avg_pose']:.6f}")
        del deblocked
    except Exception as e:
        results["deblock"] = {"status": "error", "error": str(e)}
        print(f"  Failed: {e}")

    # --- Summary ---
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    results["status"] = "ok"

    # Find best individual trick
    trick_scores = {}
    for name in ["crf_sweep", "quantization_rounding", "tto", "multi_pass", "deblock"]:
        r = results.get(name, {})
        if isinstance(r, dict):
            if "score" in r:
                trick_scores[name] = r["score"]
            elif "seg" in r and "pose" in r:
                trick_scores[name] = r["seg"] * 100 + math.sqrt(10 * r["pose"])
            # CRF sweep: use best
            if name == "crf_sweep":
                for k, v in r.items():
                    if isinstance(v, dict) and "score" in v:
                        trick_scores[k] = v["score"]

    results["trick_scores_summary"] = trick_scores
    if trick_scores:
        best_name = min(trick_scores, key=trick_scores.get)
        results["best_trick"] = best_name
        results["best_trick_score"] = trick_scores[best_name]
        print(f"\n  Best trick: {best_name} (score={trick_scores[best_name]:.4f})")

    # Verdict
    baseline_proxy = 0.9238  # known proxy score
    if trick_scores:
        best_individual = min(trick_scores.values())
        improvement = (baseline_proxy - best_individual) / baseline_proxy * 100
        if improvement >= 10:
            results["verdict"] = "SUCCESS_COMPOUNDING"
            print(f"  VERDICT: SUCCESS ({improvement:.1f}% improvement)")
        elif improvement > 0:
            results["verdict"] = "CONCERN_SMALL_GAINS"
            print(f"  VERDICT: CONCERN ({improvement:.1f}% -- less than 10% target)")
        else:
            results["verdict"] = "KILLED_NO_IMPROVEMENT"
            print(f"  VERDICT: KILLED (no improvement)")
    else:
        results["verdict"] = "INCOMPLETE"
        print(f"  VERDICT: INCOMPLETE (no tricks scored)")

    print(f"  Time: {elapsed:.0f}s")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: CPU Trick Stack")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--tto-steps", type=int, default=10)
    args = parser.parse_args()

    cfg = TrickStackTestConfig(
        n_frames=args.n_frames,
        tto_steps=args.tto_steps,
        device=args.device,
    )

    try:
        results = run_trick_stack(cfg)
    except Exception:
        traceback.print_exc()
        results = {"status": "error", "error": traceback.format_exc()}

    out_path = RESULTS_DIR / "exp4_cpu_trick_stack_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
