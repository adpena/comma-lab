"""INT8 dequantization drift measurement.

Audits layerwise fp32->int8 quantization error for saved best-checkpoint
artifacts. Useful for diagnosing whether quantization is silently degrading
model quality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def dequantize_int8_state_file(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Load an int8 state file and dequantize all parameters back to float32.

    Parameters
    ----------
    path : Path
        Path to the int8 ``.pt`` checkpoint.

    Returns
    -------
    tuple[dict, dict[str, torch.Tensor]]
        ``(meta, float_state)`` where *meta* comes from the ``__meta__`` key
        and *float_state* maps parameter names to dequantized fp32 tensors.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    meta = dict(state.get("__meta__", {}))
    float_state: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for raw_key in state.keys():
        if raw_key == "__meta__":
            continue
        if raw_key.endswith(".q") or raw_key.endswith(".s"):
            base = raw_key[:-2]
            if base in seen:
                continue
            seen.add(base)
            q = state[base + ".q"].float()
            s = state[base + ".s"]
            if getattr(s, "ndim", 0) == 0:
                float_state[base] = q * s
            else:
                shape = [s.shape[0]] + [1] * (q.ndim - 1)
                float_state[base] = q * s.view(*shape)
        else:
            float_state[raw_key] = state[raw_key].float()
            seen.add(raw_key)
    return meta, float_state


def _tensor_metrics(name: str, fp32: torch.Tensor, dequant: torch.Tensor) -> dict:
    """Compute per-layer drift metrics between fp32 and dequantized tensors."""
    fp = fp32.detach().float().reshape(-1)
    dq = dequant.detach().float().reshape(-1)
    diff = dq - fp
    denom = max(float(fp.norm().item()), 1e-12)
    cosine = torch.nn.functional.cosine_similarity(fp.unsqueeze(0), dq.unsqueeze(0)).item()
    return {
        "name": name,
        "numel": int(fp.numel()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "relative_l2": float(diff.norm().item() / denom),
        "cosine": float(cosine),
    }


def audit_best_meta(meta_path: Path) -> dict:
    """Audit quantization drift for a single best-checkpoint meta file.

    Parameters
    ----------
    meta_path : Path
        Path to a ``*_best_meta.json`` file containing ``fp32_path`` and
        ``int8_path`` entries.

    Returns
    -------
    dict
        Audit report with per-layer and aggregate drift metrics.
    """
    meta_payload = json.loads(meta_path.read_text())
    fp32_path = Path(meta_payload["fp32_path"])
    int8_path = Path(meta_payload["int8_path"])

    fp32_state = torch.load(fp32_path, map_location="cpu", weights_only=True)
    _, dequant_state = dequantize_int8_state_file(int8_path)

    common = sorted(set(fp32_state.keys()) & set(dequant_state.keys()))
    layers = [_tensor_metrics(name, fp32_state[name], dequant_state[name]) for name in common]
    layers.sort(key=lambda item: item["max_abs"], reverse=True)

    total_numel = sum(item["numel"] for item in layers) or 1
    sum_abs = sum(item["mean_abs"] * item["numel"] for item in layers)
    sum_sq = 0.0
    max_abs = 0.0
    for name in common:
        diff = dequant_state[name].detach().float().reshape(-1) - fp32_state[name].detach().float().reshape(-1)
        sum_sq += float((diff * diff).sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))

    aggregate = {
        "max_abs": max_abs,
        "mean_abs": sum_abs / total_numel,
        "rmse": math.sqrt(sum_sq / total_numel),
    }

    return {
        "meta_path": str(meta_path),
        "epoch": meta_payload.get("epoch"),
        "scorer": meta_payload.get("scorer"),
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "layer_count": len(layers),
        "aggregate": aggregate,
        "layers": layers,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the quantization drift audit."""
    parser = argparse.ArgumentParser(description="Audit fp32->int8 layerwise drift for saved best artifacts.")
    parser.add_argument("meta_paths", nargs="+", type=Path, help="One or more *_best_meta.json paths.")
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    reports = [audit_best_meta(path) for path in args.meta_paths]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
