#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""NSCS01 head0 architecture disambiguator stub.

This is a fail-closed operator probe surface, not score evidence. It records
the exact measurements required before NSCS01's frame-0 nullspace byte-saving
band can rank a dispatch:

1. frame-0 vs frame-1 PoseNet gradient norms on the same candidate batch;
2. SegNet invariance under frame-0 perturbation;
3. head0 CNN-vs-MLP no-train or short-smoke ablation;
4. paired CPU+CUDA exact-eval custody for any promoted candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_probe_plan() -> dict[str, Any]:
    """Return the deterministic probe plan consumed by operator ledgers."""
    return {
        "probe_id": "nscs01_head0_arch_disambiguator",
        "score_claim": False,
        "promotion_eligible": False,
        "required_measurements": [
            "posenet_gradient_norm_frame0_vs_frame1",
            "segnet_frame0_perturbation_invariance",
            "head0_cnn_vs_mlp_ablation",
            "paired_cpu_cuda_exact_eval_for_promoted_candidate",
        ],
        "dispatch_rule": (
            "Do not rank NSCS01 predicted_delta until this probe emits measured "
            "component deltas and a paired-axis follow-up plan."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    plan = build_probe_plan()
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
