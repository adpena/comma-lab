#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe which stage-0 prefix reset fixes MLX SegNet final argmax drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_segnet_prefix_reset_probe import (
    build_mlx_segnet_stage0_prefix_reset_probe_manifest,
    write_mlx_segnet_stage0_prefix_reset_probe_manifest,
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
        help="Permit --device gpu as local MLX research-signal evidence only.",
    )
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        manifest = build_mlx_segnet_stage0_prefix_reset_probe_manifest(
            cache_dir=args.cache_dir,
            repo_root=args.repo_root,
            device_type=args.device,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            run_id=args.run_id,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_mlx_segnet_stage0_prefix_reset_probe_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "verdict": manifest["verdict"],
                "pair_window": manifest["pair_window"],
                "baseline_segnet_argmax_diff_pixels": (
                    manifest["baseline_segnet_argmax_diff_pixels"]
                ),
                "earliest_zero_argmax_boundary": manifest["earliest_zero_argmax_boundary"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
