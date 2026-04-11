"""Checkpoint ensemble via weight-space averaging.

Core logic extracted from experiments/ensemble_checkpoints.py for use via ``tac ensemble``.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from tac.architectures import build_postfilter
from tac.quantization import load_int8, save_int8


def load_checkpoint_state_dict(
    path: str, variant: str, hidden: int, kernel: int = 3
) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return its float state dict.

    Handles both int8-quantized (.pt with .q/.s keys) and float checkpoints.
    """
    model = build_postfilter(variant, hidden=hidden, kernel=kernel)
    try:
        model = load_int8(path, model, device="cpu")
        return model.state_dict()
    except Exception as e:
        print(f"  Note: INT8 load failed ({e}), trying float format")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "__meta__" in state:
            del state["__meta__"]
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "ema_shadow" in state:
            model.load_state_dict(state["ema_shadow"])
        else:
            model.load_state_dict(state)
        return model.state_dict()


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average multiple state dicts (weight-space averaging / SWA)."""
    if len(state_dicts) == 0:
        raise ValueError("Need at least one state dict to average")
    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    avg: dict[str, torch.Tensor] = {}
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
    sort_by: str = "mtime",
) -> list[str]:
    """Discover checkpoint files in a directory."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if sort_by == "name":
        candidates = sorted(d.glob(pattern), key=lambda p: p.name, reverse=True)
    else:
        candidates = sorted(d.glob(pattern), key=os.path.getmtime, reverse=True)
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

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 2):
        sd_keys = set(sd.keys())
        if sd_keys != ref_keys:
            missing = ref_keys - sd_keys
            extra = sd_keys - ref_keys
            raise ValueError(
                f"Checkpoint {i} has incompatible architecture: "
                f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )

    print("[ensemble] Averaging state dicts...")
    avg_sd = average_state_dicts(state_dicts)

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
