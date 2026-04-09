#!/usr/bin/env python
"""Average fp32 weights across multiple trained post-filter candidates.

The simplest possible ensemble: take the fp32 weights of N candidate
trainers that all use the same architecture and simply average them.

This is Polyak averaging across independent SGD runs — well-known to
reduce variance without changing deployment cost. It works because
different training trajectories find slightly different local minima,
and the average of those minima is often closer to the true optimum
than any individual.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/ensemble_weight_averaging.py \\
        --tag ensemble_h32_avg \\
        experiments/postfilter_weights/postfilter_long1000_qat_ema_alpha20_h32_fp32.pt \\
        experiments/postfilter_weights/postfilter_kalman_long1000_h32_fp32.pt \\
        experiments/postfilter_weights/postfilter_uint8ste_long1000_h32_fp32.pt

Then score the resulting ``postfilter_{tag}_int8.pt`` via the faithful
proxy. If it beats all individual members it is a free improvement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "submissions" / "robust_current"))

from train_postfilter_saliency import (  # type: ignore
    OUTPUT_DIR,
    normalize_postfilter_meta,
    save_model_int8,
)
from inflate_postfilter import PostFilter  # type: ignore


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Average fp32 post-filter weights")
    p.add_argument("inputs", nargs="+", help="paths to fp32 state dicts")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--weights", type=float, nargs="*", default=None,
                   help="optional per-input weight, default uniform")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    n_inputs = len(args.inputs)

    # Normalize weights
    if args.weights:
        assert len(args.weights) == n_inputs
        weights = torch.tensor(args.weights, dtype=torch.float32)
    else:
        weights = torch.ones(n_inputs, dtype=torch.float32)
    weights = weights / weights.sum()

    print(f"[ensemble] averaging {n_inputs} fp32 checkpoints with weights {weights.tolist()}")

    # Load all state dicts
    state_dicts = []
    for path in args.inputs:
        print(f"  loading {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        state_dicts.append(sd)

    # Verify matching keys
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        assert set(sd.keys()) == keys, "state dict keys don't match"

    # Weighted average
    averaged = {}
    for k in keys:
        tensors = [sd[k].float() for sd in state_dicts]
        stack = torch.stack(tensors, dim=0)  # (N, ...)
        # Broadcast weights
        w = weights.view(-1, *([1] * (stack.ndim - 1)))
        avg = (stack * w).sum(dim=0)
        # Cast back to original dtype
        averaged[k] = avg.to(state_dicts[0][k].dtype)

    # Build a model and load the averaged weights
    model = PostFilter(hidden=args.hidden, kernel=args.kernel)
    model.load_state_dict(averaged)
    model.eval()

    # Save fp32 + int8
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUTPUT_DIR / f"postfilter_{args.tag}_fp32.pt"
    int8_path = OUTPUT_DIR / f"postfilter_{args.tag}_int8.pt"
    torch.save(averaged, fp32_path)

    meta = normalize_postfilter_meta(args.hidden, args.kernel, args.alpha)
    int8_size = save_model_int8(model, int8_path, meta=meta)

    print(f"\n[ensemble] Saved fp32: {fp32_path}")
    print(f"[ensemble] Saved int8: {int8_path} ({int8_size} bytes)")
    return {"tag": args.tag, "inputs": args.inputs, "weights": weights.tolist(),
            "fp32_path": str(fp32_path), "int8_path": str(int8_path),
            "int8_size": int8_size}


if __name__ == "__main__":
    main()
