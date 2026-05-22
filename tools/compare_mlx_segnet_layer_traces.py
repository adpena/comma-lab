#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare two PyTorch-vs-MLX SegNet layer-trace manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_segnet_trace_compare import (
    MLXSegNetTraceComparisonError,
    compare_mlx_segnet_layer_traces,
    load_trace_manifest,
    write_trace_comparison,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        comparison = compare_mlx_segnet_layer_traces(
            baseline=load_trace_manifest(args.baseline),
            candidate=load_trace_manifest(args.candidate),
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            top_k=args.top_k,
        )
    except (OSError, json.JSONDecodeError, MLXSegNetTraceComparisonError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_trace_comparison(comparison, args.output)
    print(
        json.dumps(
            {
                "verdict": comparison["verdict"],
                "score_claim": False,
                "segnet_argmax_diff_pixels_change": (
                    comparison["segnet_argmax_diff_pixels_change"]
                ),
                "baseline_drift_cliff": comparison["baseline_drift_cliff"],
                "candidate_drift_cliff": comparison["candidate_drift_cliff"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
