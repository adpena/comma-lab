"""Technique 7: Checkpoint ensemble via weight-space averaging.

Load top N checkpoints from a training run, average their state dicts
(weight-space averaging), quantize the averaged weights to INT8, and
evaluate. Weight-space averaging should be strictly better than any
individual checkpoint (Izmailov et al., 2018 — SWA paper).

Usage:
    python experiments/ensemble_checkpoints.py \\
        --checkpoints path1.pt path2.pt path3.pt \\
        --output averaged_int8.pt \\
        --variant dilated --hidden 64

    # Or discover checkpoints from a directory:
    python experiments/ensemble_checkpoints.py \\
        --checkpoint-dir experiments/postfilter_weights/ \\
        --top-k 5 \\
        --output averaged_int8.pt \\
        --variant dilated --hidden 64
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from tac.architectures import build_postfilter
from tac.quantization import load_int8, save_int8, save_int8_from_state_dict


def load_checkpoint_state_dict(path: str, variant: str, hidden: int, kernel: int = 3) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return its float state dict.

    Handles both int8-quantized (.pt with .q/.s keys) and float checkpoints.
    """
    model = build_postfilter(variant, hidden=hidden, kernel=kernel)
    try:
        model = load_int8(path, model, device="cpu")
        return model.state_dict()
    except Exception:
        # Try loading as float checkpoint
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "__meta__" in state:
            del state["__meta__"]
        # Check if it's a training state with model key
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "ema_shadow" in state:
            model.load_state_dict(state["ema_shadow"])
        else:
            model.load_state_dict(state)
        return model.state_dict()


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average multiple state dicts (weight-space averaging).

    All state dicts must have the same keys and tensor shapes.

    Args:
        state_dicts: list of state dicts to average

    Returns:
        Averaged state dict
    """
    if len(state_dicts) == 0:
        raise ValueError("Need at least one state dict to average")
    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    avg = {}
    keys = state_dicts[0].keys()
    n = len(state_dicts)

    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        avg[key] = sum(tensors) / n

    return avg


def discover_checkpoints(
    directory: str,
    pattern: str = "*.pt",
    top_k: int = 5,
) -> list[str]:
    """Discover checkpoint files in a directory, sorted by modification time.

    Returns the most recent top_k checkpoints.
    """
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    candidates = sorted(d.glob(pattern), key=os.path.getmtime, reverse=True)
    # Filter out training state files
    candidates = [c for c in candidates if "training_state" not in c.name]
    return [str(c) for c in candidates[:top_k]]


def ensemble_and_save(
    checkpoint_paths: list[str],
    output_path: str,
    variant: str = "dilated",
    hidden: int = 64,
    kernel: int = 3,
    per_channel: bool = True,
) -> dict:
    """Load checkpoints, average, quantize to int8, and save.

    Returns dict with metadata about the ensemble.
    """
    print(f"[ensemble] Loading {len(checkpoint_paths)} checkpoints...")
    state_dicts = []
    for i, path in enumerate(checkpoint_paths):
        print(f"  [{i+1}/{len(checkpoint_paths)}] {path}")
        sd = load_checkpoint_state_dict(path, variant, hidden, kernel)
        state_dicts.append(sd)

    print("[ensemble] Averaging state dicts...")
    avg_sd = average_state_dicts(state_dicts)

    # Save as int8 (use save_int8 with per_channel for better fidelity)
    meta = {
        "variant": variant,
        "hidden": hidden,
        "kernel": kernel,
        "ensemble_size": len(checkpoint_paths),
        "source_checkpoints": checkpoint_paths,
        "method": "weight_space_averaging",
        "per_channel": per_channel,
    }

    model = build_postfilter(variant, hidden=hidden, kernel=kernel)
    model.load_state_dict(avg_sd)
    size = save_int8(model, output_path, meta=meta, per_channel=per_channel)
    print(f"[ensemble] Saved averaged int8 checkpoint: {output_path} ({size:,} bytes)")

    return {
        "output_path": output_path,
        "size_bytes": size,
        "num_checkpoints": len(checkpoint_paths),
        "source_paths": checkpoint_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Checkpoint ensemble via weight-space averaging")
    parser.add_argument("--checkpoints", nargs="+", help="Explicit checkpoint paths")
    parser.add_argument("--checkpoint-dir", help="Directory to discover checkpoints from")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top checkpoints to average")
    parser.add_argument("--output", required=True, help="Output int8 checkpoint path")
    parser.add_argument("--variant", default="dilated", help="Architecture variant")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channel width")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size")
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false",
                        help="Disable per-channel quantization (use per-tensor instead)")
    args = parser.parse_args()

    if args.checkpoints:
        paths = args.checkpoints
    elif args.checkpoint_dir:
        paths = discover_checkpoints(args.checkpoint_dir, top_k=args.top_k)
    else:
        parser.error("Must specify --checkpoints or --checkpoint-dir")

    if not paths:
        print("No checkpoints found!")
        return

    result = ensemble_and_save(
        checkpoint_paths=paths,
        output_path=args.output,
        variant=args.variant,
        hidden=args.hidden,
        kernel=args.kernel,
        per_channel=args.per_channel,
    )
    print(f"\nEnsemble complete: {result['num_checkpoints']} checkpoints -> {result['output_path']}")


if __name__ == "__main__":
    main()
