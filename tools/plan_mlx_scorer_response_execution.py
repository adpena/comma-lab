#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan safe local MLX scorer-response execution from a stability selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.local_acceleration.mlx_execution_plan import (
    MLXExecutionPlanError,
    build_mlx_scorer_response_execution_plan,
    load_json_object,
    render_execution_plan_markdown,
    write_execution_plan,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--archive-size-bytes", type=int)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--response-output", type=Path)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--allow-gpu-research-signal",
        action="store_true",
        help=(
            "Permit a selected GPU row as local research signal only. The plan "
            "still refuses score/promotion authority."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        plan = build_mlx_scorer_response_execution_plan(
            load_json_object(args.stability_manifest),
            archive_size_bytes=args.archive_size_bytes,
            repo_root=args.repo_root,
            response_output=args.response_output,
            components_dir=args.components_dir,
            progress_every=args.progress_every,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
        )
    except (OSError, ValueError, MLXExecutionPlanError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    write_execution_plan(plan, args.output)
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_execution_plan_markdown(plan), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "md_out": None if args.md_out is None else str(args.md_out),
                "device": plan["recommended_execution"]["device"],
                "batch_pairs": plan["recommended_execution"]["batch_pairs"],
                "score_claim": plan["score_claim"],
                "promotion_eligible": plan["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
