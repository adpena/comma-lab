#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the DQS1 local-first experiment queue from an action summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.dqs1_local_first_queue import (  # noqa: E402
    DEFAULT_QUEUE_ID,
    DEFAULT_RESULTS_ROOT,
    ExperimentQueueError,
    build_queue_from_action_summary,
    find_latest_cross_family_action_summary,
)


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-summary",
        default="latest",
        help="action_summary.json path, or 'latest' to select the newest cross-family summary",
    )
    parser.add_argument(
        "--output",
        default=f"configs/experiment_queues/{DEFAULT_QUEUE_ID}.yaml",
        help="queue definition output path",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write --output; without this flag the generated queue is printed",
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="DQS1 result root used for materialized candidate paths",
    )
    parser.add_argument("--bridge-plan", default=None, help="decoder-q bridge plan path")
    parser.add_argument("--base-submission-dir", default=None, help="parent submission runtime path")
    parser.add_argument(
        "--global-mutated-archive",
        default=None,
        help="global mutated archive used by locality controls",
    )
    parser.add_argument("--upstream-dir", default=None, help="upstream evaluator directory")
    parser.add_argument("--video-names-file", default=None, help="public-test video names file")
    parser.add_argument("--frame-policy", default=None, help="DQS1 frame policy")
    parser.add_argument(
        "--exclude-candidate",
        action="append",
        default=[],
        help="candidate_id to skip when selecting the next queue target",
    )
    parser.add_argument(
        "--include-completed-local-advisory",
        action="store_true",
        help="do not skip candidates whose local_cpu_advisory.json already exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    action_summary = (
        find_latest_cross_family_action_summary(REPO_ROOT)
        if args.action_summary == "latest"
        else Path(args.action_summary)
    )
    result = build_queue_from_action_summary(
        action_summary,
        repo_root=REPO_ROOT,
        results_root=args.results_root,
        **{
            key: value
            for key, value in {
                "bridge_plan": args.bridge_plan,
                "base_submission_dir": args.base_submission_dir,
                "global_mutated_archive": args.global_mutated_archive,
                "upstream_dir": args.upstream_dir,
                "video_names_file": args.video_names_file,
                "frame_policy": args.frame_policy,
            }.items()
            if value is not None
        },
        exclude_candidate_ids=set(args.exclude_candidate),
        skip_completed_local_advisory=not args.include_completed_local_advisory,
    )
    output = Path(args.output)
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.queue, indent=2, allow_nan=False) + "\n")
        _json_print(
            {
                "output": str(output),
                "queue_id": result.queue["queue_id"],
                "selected_candidate_id": result.selection.candidate_id,
                "selected_pair_indices": list(result.selection.selected_pair_indices),
                "action_summary": str(result.selection.action_summary_path),
                "portfolio": str(result.selection.portfolio_path),
                "skipped_candidates": list(result.selection.skipped_candidates),
            }
        )
    else:
        _json_print(result.queue)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentQueueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
