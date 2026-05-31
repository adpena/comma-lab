#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate local replay results before any exact auth-eval dispatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from comma_lab.local_exact_auth_gate import (
    LocalExactAuthGateConfig,
    build_local_exact_auth_gate_report,
    load_json_object,
)
from tac.repo_io import ArtifactWriteError, write_text_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-replay-summary-json", type=Path, default=None)
    parser.add_argument("--mlx-prefilter-summary-json", type=Path, default=None)
    parser.add_argument("--auth-frontier-score", type=float, default=None)
    parser.add_argument("--local-baseline-score", type=float, default=None)
    parser.add_argument("--min-local-improvement", type=float, default=0.0)
    parser.add_argument("--exact-auth-axis", default="[contest-CPU]")
    parser.add_argument("--expected-local-axis-tag", default="[macOS-CPU advisory]")
    parser.add_argument("--require-mlx-prefilter", action="store_true")
    parser.add_argument("--mlx-target-action", type=float, default=None)
    parser.add_argument("--expected-mlx-axis-tag", default="[macOS-MLX research-signal]")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--success-on-blocked",
        action="store_true",
        help="Return 0 after writing a fail-closed blocked report; useful for queue-owned classifiers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.local_replay_summary_json is None and args.mlx_prefilter_summary_json is None:
        raise SystemExit("at least one of --local-replay-summary-json or --mlx-prefilter-summary-json is required")
    local_summary = (
        load_json_object(args.local_replay_summary_json)
        if args.local_replay_summary_json is not None
        else None
    )
    mlx_summary = (
        load_json_object(args.mlx_prefilter_summary_json)
        if args.mlx_prefilter_summary_json is not None
        else None
    )
    report = build_local_exact_auth_gate_report(
        local_replay_summary=local_summary,
        mlx_prefilter_summary=mlx_summary,
        local_replay_summary_path=args.local_replay_summary_json,
        mlx_prefilter_summary_path=args.mlx_prefilter_summary_json,
        config=LocalExactAuthGateConfig(
            exact_auth_axis=args.exact_auth_axis,
            auth_target_score=args.auth_frontier_score,
            local_baseline_score=args.local_baseline_score,
            min_local_improvement=args.min_local_improvement,
            expected_local_axis_tag=args.expected_local_axis_tag,
            require_mlx_prefilter=bool(args.require_mlx_prefilter),
            mlx_target_action=args.mlx_target_action,
            expected_mlx_axis_tag=args.expected_mlx_axis_tag,
        ),
    )
    write_text_artifact(args.out_json, report.to_json() + "\n")
    print(report.to_json())
    return 0 if report.exact_auth_dispatch_recommended or args.success_on_blocked else 2


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"gate_local_candidate_for_exact_auth failed: {exc}", file=sys.stderr)
        if isinstance(exc, ArtifactWriteError):
            raise SystemExit(2) from exc
        raise
