#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit upstream PyTorch-vs-MLX scorer parity on a fixed cache window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_scorer_torch_parity import (
    MLXTorchParityThresholds,
    build_mlx_scorer_torch_parity_manifest,
    write_mlx_scorer_torch_parity_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--allow-gpu-research-signal",
        action="store_true",
        help="Permit --device gpu as local MLX research-signal parity evidence only.",
    )
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-posenet-output-abs-delta", type=float, default=2.0e-3)
    parser.add_argument("--max-segnet-logit-abs-delta", type=float, default=1.0e-2)
    parser.add_argument("--max-posenet-component-abs-delta", type=float, default=2.0e-5)
    parser.add_argument("--max-segnet-argmax-diff-pixels", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_mlx_scorer_torch_parity_manifest(
            cache_dir=args.cache_dir,
            repo_root=args.repo_root,
            device_type=args.device,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            thresholds=MLXTorchParityThresholds(
                max_posenet_output_abs_delta=args.max_posenet_output_abs_delta,
                max_segnet_logit_abs_delta=args.max_segnet_logit_abs_delta,
                max_posenet_component_abs_delta=args.max_posenet_component_abs_delta,
                max_segnet_argmax_diff_pixels=args.max_segnet_argmax_diff_pixels,
            ),
            run_id=args.run_id,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_mlx_scorer_torch_parity_manifest(manifest, args.output)
    print(json.dumps({"passed": manifest["passed"], "verdict": manifest["verdict"]}, sort_keys=True))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
