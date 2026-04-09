#!/usr/bin/env python3
"""Average top-K int8 checkpoint weights for smoother quantization.

Instead of picking the single best epoch, this averages the fp32 weights
from the top-K checkpoints (by scorer) and then quantizes the average.
May smooth out quantization noise for a better int8 result.

Usage:
    python experiments/average_top_checkpoints.py \
        --weights-dir experiments/postfilter_weights/ \
        --tag long1000_h64 \
        --top-k 3

Council recommendation: "average top-3 epoch int8 weights to smooth
out quantization noise."
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from src.tac.quantization import save_int8


def find_checkpoints(weights_dir: str, tag_pattern: str) -> list[dict]:
    """Find all checkpoint metadata files matching a pattern."""
    pattern = os.path.join(weights_dir, f"*{tag_pattern}*_meta.json")
    metas = []
    for path in glob.glob(pattern):
        with open(path) as f:
            meta = json.load(f)
            meta["meta_path"] = path
            metas.append(meta)
    return sorted(metas, key=lambda m: m.get("scorer", float("inf")))


def average_fp32_weights(paths: list[str]) -> dict[str, torch.Tensor]:
    """Load and average fp32 state dicts."""
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    avg = {}
    for key in states[0]:
        tensors = [s[key].float() for s in states]
        avg[key] = torch.stack(tensors).mean(dim=0)
    return avg


def main():
    parser = argparse.ArgumentParser(description="Average top-K checkpoints")
    parser.add_argument("--weights-dir", default="experiments/postfilter_weights/")
    parser.add_argument("--tag", required=True, help="Tag pattern to match")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=None, help="Output int8 path")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.weights_dir, args.tag)
    if len(checkpoints) < args.top_k:
        print(f"Only {len(checkpoints)} checkpoints found for '{args.tag}', need {args.top_k}")
        # Still try with what we have
        if not checkpoints:
            print("No checkpoints found.")
            return

    top_k = checkpoints[:args.top_k]
    print(f"Averaging top-{len(top_k)} checkpoints:")
    for c in top_k:
        print(f"  epoch {c['epoch']}, scorer {c['scorer']:.4f}, {c.get('fp32_path', '?')}")

    fp32_paths = [c["fp32_path"] for c in top_k if os.path.exists(c.get("fp32_path", ""))]
    if not fp32_paths:
        print("No fp32 weight files found.")
        return

    print(f"Averaging {len(fp32_paths)} fp32 weight sets...")
    avg_state = average_fp32_weights(fp32_paths)

    # Build a dummy model to save through save_int8
    # Or just quantize directly
    out_path = args.out or os.path.join(
        args.weights_dir,
        f"postfilter_{args.tag}_top{len(fp32_paths)}_avg_int8.pt"
    )

    # Manual int8 quantization of averaged weights
    int8_state = {}
    meta_info = top_k[0].get("meta", {})
    for name, param in avg_state.items():
        p = param.float()
        scale = p.abs().max() / 127.0
        if scale.item() == 0:
            scale = torch.tensor(1.0)
        quantized = (p / scale).round().clamp(-128, 127).to(torch.int8)
        int8_state[name + ".q"] = quantized
        int8_state[name + ".s"] = scale

    int8_state["__meta__"] = meta_info
    torch.save(int8_state, out_path)
    size = os.path.getsize(out_path)
    print(f"Saved averaged int8 to {out_path} ({size} bytes)")
    print(f"Source epochs: {[c['epoch'] for c in top_k]}")
    print("Source scorers:", [round(c["scorer"], 4) for c in top_k])


if __name__ == "__main__":
    main()
