#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe MLX Metal scorer-VJP drift against MLX CPU and PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_scorer_vjp_crux import (  # noqa: E402
    BRANCHES,
    build_mlx_scorer_vjp_crux_manifest,
    write_mlx_scorer_vjp_crux_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--full-video-d-pose", required=True, type=float)
    parser.add_argument("--seg-ce-weight", type=float, default=100.0)
    parser.add_argument("--pose-eps", type=float, default=1.0e-12)
    parser.add_argument("--branch", action="append", choices=BRANCHES)
    parser.add_argument("--max-abs-ratio-warn", type=float, default=1.0e3)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.start_pair < 0:
        raise SystemExit("--start-pair must be >= 0")
    if args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    if args.pose_eps <= 0.0:
        raise SystemExit("--pose-eps must be > 0")
    if args.max_abs_ratio_warn < 1.0:
        raise SystemExit("--max-abs-ratio-warn must be >= 1")
    manifest = build_mlx_scorer_vjp_crux_manifest(
        candidate_cache_dir=args.candidate_cache_dir,
        reference_cache_dir=args.reference_cache_dir,
        repo_root=args.repo_root,
        start_pair=args.start_pair,
        max_pairs=args.max_pairs,
        full_video_d_pose=args.full_video_d_pose,
        seg_ce_weight=args.seg_ce_weight,
        pose_eps=args.pose_eps,
        branches=tuple(args.branch or BRANCHES),
        max_abs_ratio_warn=args.max_abs_ratio_warn,
        run_id=args.run_id,
    )
    write_mlx_scorer_vjp_crux_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": args.output.expanduser().resolve(strict=False).as_posix(),
                "passed": manifest["passed"],
                "verdict": manifest["verdict"],
                "blockers": manifest["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
