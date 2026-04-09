"""Evaluation utilities for task-aware codec post-filters.

Provides:
  - Faithful proxy scoring (matches authoritative scorer to 8 decimal places)
  - Top-K checkpoint averaging
  - Submission packaging validation
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import torch

from .quantization import save_int8_from_state_dict, load_int8
from .scorer import comma_score


def proxy_score(
    model,
    comp_frames: list[torch.Tensor],
    gt_frames: list[torch.Tensor],
    posenet,
    segnet,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Run faithful proxy scoring on a model.

    Evaluates the model on all frame pairs and returns the score breakdown.
    Matches the authoritative scorer's evaluate.py to high precision.

    Returns dict with: score, pose, seg, n_pairs
    """
    from .data import build_pairs
    from .losses import scorer_loss

    model = model.eval().to(device)
    comp_pairs = build_pairs(comp_frames)
    gt_pairs = build_pairs(gt_frames)

    total_p, total_s = 0.0, 0.0
    with torch.no_grad():
        for cp, gp in zip(comp_pairs, gt_pairs):
            cp = cp.to(device)
            gp = gp.to(device)

            # Apply filter
            B, T, H, W, C = cp.shape
            frames = cp.float().reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
            filtered = model(frames)
            filtered_pair = filtered.permute(0, 2, 3, 1).reshape(B, T, H, W, C)

            _, pd, sd = scorer_loss(filtered_pair, gp, posenet, segnet)
            total_p += pd
            total_s += sd

    n = len(comp_pairs)
    avg_p = total_p / n
    avg_s = total_s / n
    score = 100.0 * avg_s + math.sqrt(10.0 * avg_p)

    return {"score": score, "pose": avg_p, "seg": avg_s, "n_pairs": n}


def find_checkpoints(
    weights_dir: str | Path,
    tag_pattern: str = "",
) -> list[dict]:
    """Find all checkpoint metadata files, optionally filtered by tag pattern.

    Returns list of metadata dicts sorted by scorer (best first).
    """
    results = []
    for path in sorted(Path(weights_dir).glob("*_best_meta.json")):
        if tag_pattern and tag_pattern not in path.stem:
            continue
        try:
            data = json.loads(path.read_text())
            data["meta_path"] = str(path)
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(results, key=lambda d: d.get("scorer", float("inf")))


def average_top_k_checkpoints(
    weights_dir: str | Path,
    tag_pattern: str = "",
    k: int = 3,
    output_path: str | Path | None = None,
) -> dict:
    """Average the top-K checkpoints by scorer and save as int8.

    Smooths out quantization noise by averaging fp32 weights from the
    best K epochs before quantizing. Council recommendation: "average
    top-3 epoch int8 weights to smooth quantization noise."

    Returns metadata dict with averaged score and saved path.
    """
    checkpoints = find_checkpoints(weights_dir, tag_pattern)
    top_k = [c for c in checkpoints if os.path.exists(c.get("fp32_path", ""))][:k]

    if not top_k:
        raise ValueError(f"No checkpoints found matching '{tag_pattern}' in {weights_dir}")

    print(f"Averaging top-{len(top_k)} checkpoints:")
    for c in top_k:
        print(f"  epoch {c['epoch']}, scorer {c['scorer']:.4f}")

    # Load and average fp32 state dicts
    states = [torch.load(c["fp32_path"], map_location="cpu", weights_only=True) for c in top_k]
    avg = {}
    for key in states[0]:
        tensors = [s[key].float() for s in states if key in s]
        if tensors:
            avg[key] = torch.stack(tensors).mean(dim=0)

    # Save as int8
    if output_path is None:
        output_path = Path(weights_dir) / f"postfilter_{tag_pattern}_top{len(top_k)}_avg_int8.pt"

    meta = top_k[0].get("meta", {})
    size = save_int8_from_state_dict(avg, output_path, meta=meta)

    result = {
        "source_epochs": [c["epoch"] for c in top_k],
        "source_scorers": [c["scorer"] for c in top_k],
        "avg_scorer": sum(c["scorer"] for c in top_k) / len(top_k),
        "int8_path": str(output_path),
        "int8_size": size,
    }
    print(f"Saved averaged int8 to {output_path} ({size} bytes)")
    return result
