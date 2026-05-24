#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest learned-sweep local MLX rows into observation JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.mlx_dynamic_learned_sweep_observation_harvest import (  # noqa: E402
    DEFAULT_OPTIMIZATION_PASS_ID,
    DEFAULT_SWEEP_CONFIG_ID,
    MLXDynamicLearnedSweepObservationHarvestError,
    build_observation_harvest_manifest,
    build_observation_rows_from_learned_sweep_plan,
    load_json_object,
    write_json,
    write_observation_jsonl,
)
from tac.optimization.mlx_dynamic_sweep_observations import json_text  # noqa: E402
from tac.repo_io import sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--jsonl-out", type=Path, required=True)
    parser.add_argument("--summary-json-out", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--sweep-config-id", default=DEFAULT_SWEEP_CONFIG_ID)
    parser.add_argument("--optimization-pass-id", default=DEFAULT_OPTIMIZATION_PASS_ID)
    parser.add_argument(
        "--all-sweep-configs",
        action="store_true",
        help="Do not filter by sweep_config_id.",
    )
    parser.add_argument(
        "--all-optimization-passes",
        action="store_true",
        help="Do not filter by optimization_pass_id.",
    )
    parser.add_argument(
        "--include-exact-gated-rows",
        action="store_true",
        help="Allow harvesting rows not marked ready_for_local_sweep.",
    )
    parser.add_argument(
        "--source-artifact-root",
        type=Path,
        default=REPO_ROOT,
        help="Root used to resolve repo-relative source_path values.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = load_json_object(args.plan)
        selection = load_json_object(args.selection)
        rows = build_observation_rows_from_learned_sweep_plan(
            plan,
            selection,
            planner_artifact_path=args.plan,
            planner_artifact_sha256=sha256_file(args.plan),
            max_rows=args.max_rows,
            sweep_config_id=None if args.all_sweep_configs else args.sweep_config_id,
            optimization_pass_id=(
                None if args.all_optimization_passes else args.optimization_pass_id
            ),
            ready_local_only=not args.include_exact_gated_rows,
            source_artifact_root=args.source_artifact_root,
        )
        write_observation_jsonl(rows, output_path=args.jsonl_out, replace=args.replace)
        manifest = build_observation_harvest_manifest(
            rows,
            plan_path=args.plan,
            selection_path=args.selection,
        )
        if args.summary_json_out is not None:
            write_json(args.summary_json_out, manifest, replace=args.replace)
    except (
        OSError,
        MLXDynamicLearnedSweepObservationHarvestError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "jsonl_out": str(args.jsonl_out),
                "summary_json_out": None
                if args.summary_json_out is None
                else str(args.summary_json_out),
                "row_count": len(rows),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
