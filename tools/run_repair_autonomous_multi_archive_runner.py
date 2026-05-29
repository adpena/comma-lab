#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the repair-family autonomous floor loop over multiple archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.repair_autonomous_multi_archive_runner import (  # noqa: E402
    RepairAutonomousMultiArchiveRunnerError,
    run_repair_autonomous_multi_archive_runner,
)
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", default=[], type=Path)
    parser.add_argument("--archive-glob", action="append", default=[])
    parser.add_argument("--source-label", action="append", default=[])
    parser.add_argument(
        "--source-runtime-dir",
        action="append",
        default=[],
        help="Runtime dir for all rows, or source_label=runtime_dir.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--chain-id", default="repair_multi_archive_autonomous")
    parser.add_argument("--queue-id", default="repair_multi_archive_materialization")
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--close-runtime-custody", action="store_true")
    parser.add_argument("--max-floor-iterations", type=int, default=4)
    parser.add_argument("--max-steps-per-iteration", type=int, default=128)
    parser.add_argument("--worker-max-experiments-per-iteration", type=int)
    parser.add_argument("--byte-credit-budget", type=int)
    parser.add_argument("--posterior-path", type=Path)
    parser.add_argument("--posterior-lock-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-overwrite-existing-historical-provenance",
        action="store_true",
        help=(
            "Opt-in to overwriting an existing .omx/research/<dir>/ that already "
            "contains canonical HISTORICAL_PROVENANCE JSON files. Per Catalog #113 + "
            "anti-pattern "
            "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1, "
            "the default behavior is fail-closed; requires --overwrite-rationale."
        ),
    )
    parser.add_argument(
        "--overwrite-rationale",
        type=str,
        default=None,
        help=(
            "Substantive operator rationale (>=4 chars; non-placeholder per "
            "Catalog #287) required when --allow-overwrite-existing-historical-provenance "
            "is set."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Canonical HISTORICAL_PROVENANCE safety per Catalog #113 + anti-pattern
    # `research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1`
    # (registered 2026-05-28; 50 of 77 violations were on repair_multi_archive_*
    # dirs from re-running with same --output-dir).
    from tac.research_pipeline_output_dir_safety import (
        OutputDirSafetyError,
        enforce_research_pipeline_output_dir,
    )

    try:
        enforce_research_pipeline_output_dir(
            args.output_dir,
            repo_root=REPO_ROOT,
            allow_overwrite_existing_historical_provenance=(
                args.allow_overwrite_existing_historical_provenance
            ),
            waiver_rationale=args.overwrite_rationale,
        )
    except OutputDirSafetyError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 3
    try:
        summary = run_repair_autonomous_multi_archive_runner(
            archives=args.archive,
            archive_globs=args.archive_glob,
            source_labels=args.source_label,
            source_runtime_dirs=args.source_runtime_dir,
            output_dir=args.output_dir,
            repo_root=REPO_ROOT,
            max_archives=args.max_archives,
            chain_id=args.chain_id,
            queue_id=args.queue_id,
            execute_local=args.execute_local,
            close_runtime_custody=args.close_runtime_custody,
            max_floor_iterations=args.max_floor_iterations,
            max_steps_per_iteration=args.max_steps_per_iteration,
            worker_max_experiments_per_iteration=args.worker_max_experiments_per_iteration,
            byte_credit_budget=args.byte_credit_budget,
            posterior_path=args.posterior_path,
            posterior_lock_path=(
                args.posterior_lock_path
                if args.posterior_lock_path is not None
                else ".omx/state/repair_campaign_stackability_posterior.lock"
            ),
            overwrite=args.overwrite,
        )
    except (
        RepairAutonomousMultiArchiveRunnerError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair autonomous multi-archive runner failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_autonomous_multi_archive_runner_cli_result.v1",
                "runner_summary_path": str(args.output_dir / "runner_summary.json"),
                "archive_count": summary["archive_count"],
                "typed_response_count": summary["typed_response_count"],
                "ready_experiment_count": summary["ready_experiment_count"],
                "exact_ready_bridge_candidate_count": summary[
                    "exact_ready_bridge_candidate_count"
                ],
                "exact_ready_bridge_runtime_content_tree_custody_proven_count": summary[
                    "exact_ready_bridge_runtime_content_tree_custody_proven_count"
                ],
                "blocked_exact_dispatch_authorized_candidate_count": summary[
                    "blocked_exact_dispatch_authorized_candidate_count"
                ],
                "blocked_exact_dispatch_blocked_candidate_count": summary[
                    "blocked_exact_dispatch_blocked_candidate_count"
                ],
                "stop_reason": summary["stop_reason"],
                "max_floor_iterations": summary["max_floor_iterations"],
                "bounded_live_archive_loop": summary["bounded_live_archive_loop"],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
