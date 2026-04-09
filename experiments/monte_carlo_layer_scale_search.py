#!/usr/bin/env python
"""CMA-style Monte Carlo search over a tiny layer-scale manifold.

Full CMA-ES over all post-filter weights is expensive and awkward to fit into
the lab's current scoring loop. This script keeps the idea but narrows the
search space to a few scale factors around an existing winner:

- one scale per layer weight tensor
- one scale per layer bias tensor

That keeps the search low-dimensional, cheap to rank on a scorer-faithful
subsample, and easy to save as a real int8 artifact when it improves.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "submissions" / "robust_current"))

from train_postfilter_saliency import (  # type: ignore
    ARCHIVE_ZIP,
    DEVICE,
    OUTPUT_DIR,
    VIDEOS_DIR,
    apply_filter_to_pair,
    build_pairs,
    compute_pair_loss,
    decode_archive,
    decode_video,
    load_scorers,
    save_model_int8,
)
from inflate_postfilter import load_postfilter_int8, normalize_postfilter_meta  # type: ignore


LAYER_KEYS = [
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "conv3.weight",
    "conv3.bias",
]


def apply_layer_scales(
    base_state: dict[str, torch.Tensor],
    theta: np.ndarray,
    *,
    scale_width: float = 0.10,
) -> dict[str, torch.Tensor]:
    scaled = {name: tensor.detach().clone() for name, tensor in base_state.items()}
    for idx, key in enumerate(LAYER_KEYS):
        factor = 1.0 + scale_width * float(theta[idx])
        scaled[key] = scaled[key] * factor
    return scaled


def score_model(
    model: torch.nn.Module,
    comp_pairs: list[torch.Tensor],
    gt_pairs: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
) -> tuple[float, float, float]:
    total_pose, total_seg = 0.0, 0.0
    with torch.no_grad():
        for comp_pair, gt_pair in zip(comp_pairs, gt_pairs):
            filtered = apply_filter_to_pair(model, comp_pair, DEVICE)
            _, pose_dist, seg_dist = compute_pair_loss(filtered, gt_pair, posenet, segnet)
            total_pose += pose_dist
            total_seg += seg_dist
    avg_pose = total_pose / len(comp_pairs)
    avg_seg = total_seg / len(comp_pairs)
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose)
    return score, avg_pose, avg_seg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo layer-scale search around the promoted winner")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT / "submissions" / "robust_current" / "postfilter_int8.pt",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.75)
    parser.add_argument("--eval-subsample", type=int, default=20)
    parser.add_argument("--tag", type=str, default="mc_layer_scale")
    parser.add_argument(
        "--init-meta",
        type=Path,
        default=None,
        help="Optional JSON metadata file from a previous mc_layer_scale run. If present, initialize theta from its best payload.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)

    print(f"[mc] device={DEVICE} weights={args.weights} tag={args.tag}")
    model = load_postfilter_int8(str(args.weights), device=DEVICE)
    model.eval()
    base_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    meta = normalize_postfilter_meta(torch.load(args.weights, map_location="cpu").get("__meta__"))
    meta["search_strategy"] = "mc_layer_scale"

    posenet, segnet = load_scorers(DEVICE)
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_pairs_all = build_pairs(comp_frames[:n])
    gt_pairs_all = build_pairs(gt_frames[:n])
    eval_indices = list(range(0, len(comp_pairs_all), args.eval_subsample))
    comp_pairs = [comp_pairs_all[i].to(DEVICE) for i in eval_indices]
    gt_pairs = [gt_pairs_all[i].to(DEVICE) for i in eval_indices]
    print(f"[mc] Evaluating on {len(comp_pairs)}/{len(comp_pairs_all)} pairs")

    theta = np.zeros(len(LAYER_KEYS), dtype=np.float32)
    if args.init_meta is not None:
        payload = json.loads(args.init_meta.read_text())
        if isinstance(payload.get("theta"), list) and len(payload["theta"]) == len(LAYER_KEYS):
            theta = np.array(payload["theta"], dtype=np.float32)
            print(f"[mc] Initialized theta from {args.init_meta}")
    best_theta = theta.copy()
    best_score, best_pose, best_seg = score_model(model, comp_pairs, gt_pairs, posenet, segnet)
    best_payload: dict[str, object] | None = None
    print(f"[mc] Baseline: score={best_score:.4f}, pose={best_pose:.6f}, seg={best_seg:.6f}")

    rng = np.random.default_rng(0)
    for iteration in range(1, args.iterations + 1):
        candidates: list[tuple[float, float, float, np.ndarray]] = []
        for _ in range(args.population):
            noise = rng.normal(0.0, args.sigma, size=len(LAYER_KEYS)).astype(np.float32)
            for sign in (+1.0, -1.0):
                candidate_theta = theta + sign * noise
                model.load_state_dict(apply_layer_scales(base_state, candidate_theta))
                score, pose, seg = score_model(model, comp_pairs, gt_pairs, posenet, segnet)
                candidates.append((score, pose, seg, candidate_theta.copy()))
        candidates.sort(key=lambda item: item[0])
        theta = np.mean([entry[3] for entry in candidates[: max(2, args.population // 2)]], axis=0)
        score, pose, seg, candidate_theta = candidates[0]
        print(
            f"{iteration:>4d} {score:>10.4f} {pose:>12.6f} {seg:>12.6f} "
            f"sigma={args.sigma:>8.4f}"
        )
        if score < best_score:
            best_score, best_pose, best_seg = score, pose, seg
            best_theta = candidate_theta.copy()
            model.load_state_dict(apply_layer_scales(base_state, best_theta))
            fp32_path = OUTPUT_DIR / f"postfilter_{args.tag}_best_fp32.pt"
            int8_path = OUTPUT_DIR / f"postfilter_{args.tag}_best_int8.pt"
            meta_path = OUTPUT_DIR / f"postfilter_{args.tag}_best_meta.json"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), fp32_path)
            int8_size = save_model_int8(model, int8_path, meta=meta)
            best_payload = {
                "iteration": iteration,
                "score": score,
                "pose": pose,
                "seg": seg,
                "theta": best_theta.tolist(),
                "fp32_path": str(fp32_path),
                "int8_path": str(int8_path),
                "int8_size": int8_size,
            }
            meta_path.write_text(json.dumps(best_payload, indent=2))
            args.sigma *= 0.9
        else:
            args.sigma *= 1.05

    model.load_state_dict(apply_layer_scales(base_state, best_theta))
    result = {
        "baseline": {"score": best_score if best_payload is None else None},
        "best": best_payload
        or {
            "iteration": 0,
            "score": best_score,
            "pose": best_pose,
            "seg": best_seg,
            "theta": best_theta.tolist(),
        },
    }
    print("[mc] JSON:")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
