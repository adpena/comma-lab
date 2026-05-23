#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the DQS1 local-first experiment queue from an action summary."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.dqs1_local_first_queue import (  # noqa: E402
    DEFAULT_DRIFT_CALIBRATION_JSON,
    DEFAULT_EUREKA_OUTPUT_DIR,
    DEFAULT_QUEUE_ID,
    DEFAULT_RESULTS_ROOT,
    ExperimentQueueError,
    build_queue_from_action_summary,
    find_latest_cross_family_action_summary,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


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
        "--overwrite-output",
        action="store_true",
        help="allow --write to replace an existing --output only with --expected-output-sha256",
    )
    parser.add_argument(
        "--expected-output-sha256",
        default=None,
        help="required sha256 of existing --output when --overwrite-output replaces it",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help="free-space floor before writing --output (default: 1.0 GiB)",
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
        "--drift-calibration-json",
        default=DEFAULT_DRIFT_CALIBRATION_JSON,
        help="local CPU to contest CPU drift calibration JSON used by the eureka step",
    )
    parser.add_argument(
        "--eureka-output-dir",
        default=DEFAULT_EUREKA_OUTPUT_DIR,
        help="directory for canonical local CPU contest drift eureka signals",
    )
    parser.add_argument(
        "--eureka-run-id",
        default=None,
        help=(
            "unique UTC-ish id for append-only eureka outputs; defaults to the "
            "current UTC timestamp when --write is used, otherwise the summary timestamp"
        ),
    )
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
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=1,
        help="number of independent safe DQS1 candidates to include in the queue",
    )
    parser.add_argument(
        "--local-cpu-concurrency",
        type=int,
        default=1,
        help="local_cpu resource concurrency to write into the generated queue",
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
                "drift_calibration_json": args.drift_calibration_json,
                "eureka_output_dir": args.eureka_output_dir,
                "eureka_run_id": args.eureka_run_id
                or (
                    datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                    if args.write
                    else None
                ),
            }.items()
            if value is not None
        },
        exclude_candidate_ids=set(args.exclude_candidate),
        skip_completed_local_advisory=not args.include_completed_local_advisory,
        candidate_limit=args.candidate_limit,
        local_cpu_concurrency=args.local_cpu_concurrency,
    )
    output = Path(args.output)
    if args.write:
        if args.overwrite_output and output.exists() and args.expected_output_sha256 is None:
            print(
                "--overwrite-output requires --expected-output-sha256 when --output exists",
                file=sys.stderr,
            )
            return 2
        try:
            artifact = write_json_artifact(
                output,
                result.queue,
                allow_overwrite=bool(args.overwrite_output),
                expected_existing_sha256=args.expected_output_sha256,
                min_free_bytes=int(max(args.min_free_gb, 0.0) * (1024**3)),
            )
        except ArtifactWriteError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        _json_print(
            {
                "output": str(output),
                "output_bytes": artifact.bytes_written,
                "output_sha256": artifact.sha256,
                "output_free_bytes_before": artifact.free_bytes_before,
                "output_allow_overwrite": artifact.allow_overwrite,
                "queue_id": result.queue["queue_id"],
                "selected_candidate_id": result.selection.candidate_id,
                "selected_candidate_ids": [
                    selection.candidate_id for selection in result.selections
                ],
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
