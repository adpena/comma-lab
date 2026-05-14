#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FP4 robustness audit: measure float→FP4 round-trip error on a trained ckpt.

Direct test of the R-FP4-fix without re-training: take a real, trained float
state dict, quantize it 4 ways, measure the L1 / L∞ reconstruction error and
the fraction of weights that round to ZERO (the failure mode the fix targets).

Usage:
    PYTHONPATH=src:upstream python experiments/fp4_roundtrip_audit.py \\
        --checkpoint experiments/results/.../renderer_*_best_fp32.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from tac.fp4_quantize import (
    DEFAULT_CODEBOOK,
    RESIDUAL_CODEBOOK,
    dequantize_fp4,
    quantize_fp4,
)


def round_trip(state_dict: dict[str, torch.Tensor], *,
               codebook: torch.Tensor, robust_scale: bool) -> dict[str, torch.Tensor]:
    packed = quantize_fp4(state_dict, codebook=codebook, robust_scale=robust_scale)
    return dequantize_fp4(packed)


def measure(name: str,
            original: dict[str, torch.Tensor],
            restored: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-tensor L1/L∞ + zero-collapse fraction summary."""
    l1_total = 0.0
    linf_total = 0.0
    n_total = 0
    n_zero_collapse = 0
    n_nonzero_orig = 0
    for k, w in original.items():
        if not torch.is_floating_point(w):
            continue
        r = restored[k]
        diff = (r.float() - w.float()).abs()
        l1_total += diff.mean().item() * w.numel()
        linf_total = max(linf_total, diff.max().item())
        n_total += w.numel()
        # Weights that were nonzero but got rounded to zero
        was_nonzero = (w.float().abs() > 1e-9)
        is_zero = (r.float().abs() < 1e-9)
        n_zero_collapse += (was_nonzero & is_zero).sum().item()
        n_nonzero_orig += was_nonzero.sum().item()
    return {
        "config": name,
        "l1_per_weight": round(l1_total / max(n_total, 1), 6),
        "linf": round(linf_total, 6),
        "n_weights": n_total,
        "zero_collapse_pct": round(100.0 * n_zero_collapse / max(n_nonzero_orig, 1), 3),
        "n_zero_collapse": n_zero_collapse,
        "n_nonzero_orig": n_nonzero_orig,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt with float state dict (e.g. *_best_fp32.pt)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FATAL: {ckpt_path} not found", file=sys.stderr)
        return 1

    print(f"[audit] Loading {ckpt_path}")
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Accept either a state_dict directly or a wrapping dict
    if isinstance(raw, dict) and any(torch.is_tensor(v) for v in raw.values()):
        state_dict = raw
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict) and "ema_state" in raw:
        state_dict = raw["ema_state"]
    else:
        print(f"FATAL: don't know how to read {type(raw)} keys={list(raw)[:5]}",
              file=sys.stderr)
        return 1

    n_params = sum(v.numel() for v in state_dict.values() if torch.is_floating_point(v))
    print(f"[audit] {n_params:,} float params across {len(state_dict)} tensors")
    print()

    configs = [
        ("DEFAULT codebook  + max-scale     (legacy)",
         dict(codebook=DEFAULT_CODEBOOK,  robust_scale=False)),
        ("DEFAULT codebook  + p99.5-scale  (R-FP4-fix #2)",
         dict(codebook=DEFAULT_CODEBOOK,  robust_scale=True)),
        ("RESIDUAL codebook + max-scale    (R-FP4-fix #1)",
         dict(codebook=RESIDUAL_CODEBOOK, robust_scale=False)),
        ("RESIDUAL codebook + p99.5-scale  (R-FP4-fix #1+#2)",
         dict(codebook=RESIDUAL_CODEBOOK, robust_scale=True)),
    ]

    results = []
    for label, kwargs in configs:
        restored = round_trip(state_dict, **kwargs)
        r = measure(label, state_dict, restored)
        results.append(r)
        print(f"  {label}")
        print(f"    L1/weight: {r['l1_per_weight']:.6f}  L∞: {r['linf']:.6f}  "
              f"zero-collapse: {r['zero_collapse_pct']:.2f}% "
              f"({r['n_zero_collapse']:,}/{r['n_nonzero_orig']:,})")
        print()

    # Pick the winner: smallest L1/weight (or fewest zero-collapses if tied)
    winner = min(results, key=lambda r: (r["l1_per_weight"], r["zero_collapse_pct"]))
    legacy = results[0]
    improvement_l1 = (legacy["l1_per_weight"] - winner["l1_per_weight"]) / legacy["l1_per_weight"] * 100
    improvement_zero = legacy["zero_collapse_pct"] - winner["zero_collapse_pct"]

    print("─" * 70)
    print(f"WINNER: {winner['config']}")
    print(f"  vs legacy: L1/weight ↓ {improvement_l1:.1f}%, "
          f"zero-collapse ↓ {improvement_zero:.2f}pp")
    print("─" * 70)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": str(ckpt_path),
            "n_params": n_params,
            "results": results,
            "winner": winner["config"],
            "improvement_l1_pct": round(improvement_l1, 2),
            "improvement_zero_collapse_pp": round(improvement_zero, 3),
        }, indent=2))
        print(f"[audit] Wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
