#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Trace PyTorch-vs-MLX SegNet drift across structural layer boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_scorer_torch_parity import (
    build_mlx_segnet_layer_trace_manifest,
    write_mlx_segnet_layer_trace_manifest,
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
        help="Permit --device gpu as local MLX research-signal trace evidence only.",
    )
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--cliff-threshold", type=float, default=1.0e-4)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_mlx_segnet_layer_trace_manifest(
            cache_dir=args.cache_dir,
            repo_root=args.repo_root,
            device_type=args.device,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            run_id=args.run_id,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
            cliff_threshold=args.cliff_threshold,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_mlx_segnet_layer_trace_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "pair_window": manifest["pair_window"],
                "trace_count": manifest["trace_count"],
                "drift_cliff": manifest["drift_cliff"],
                "segnet_argmax_diff_pixels": manifest["segnet_argmax_diff_pixels"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
