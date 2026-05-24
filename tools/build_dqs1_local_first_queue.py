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
    DEFAULT_DECODER_Q_CANDIDATE_MANIFEST,
    DEFAULT_DRIFT_CALIBRATION_JSON,
    DEFAULT_EUREKA_OUTPUT_DIR,
    DEFAULT_MLX_EFFECTIVE_SELECTION,
    DEFAULT_MLX_REFERENCE_CACHE_DIR,
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
    parser.add_argument("--queue-id", default=DEFAULT_QUEUE_ID)
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
    parser.add_argument(
        "--completed-results-root",
        action="append",
        default=[],
        help=(
            "additional result root whose completed local advisories should be "
            "treated as already observed when selecting the next candidate"
        ),
    )
    parser.add_argument(
        "--mlx-effective-selection",
        default=DEFAULT_MLX_EFFECTIVE_SELECTION,
        help=(
            "observed MLX candidate-selection artifact used to rebuild the "
            "decoder-q bridge plan with normalized full-video objective fields"
        ),
    )
    parser.add_argument(
        "--decoder-q-candidate-manifest",
        default=DEFAULT_DECODER_Q_CANDIDATE_MANIFEST,
        help="materialized decoder-q mutation manifest used to rebuild the bridge plan",
    )
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
    parser.add_argument(
        "--local-io-concurrency",
        type=int,
        default=1,
        help="local_io_heavy resource concurrency to write into the generated queue",
    )
    parser.add_argument(
        "--include-mlx-local-advisory-debug",
        action="store_true",
        help=(
            "add local MLX candidate-cache and scorer-response steps after the "
            "local CPU advisory"
        ),
    )
    parser.add_argument(
        "--allow-large-mlx-cache",
        action="store_true",
        help="required with --include-mlx-local-advisory-debug for full DQS1 caches",
    )
    parser.add_argument(
        "--mlx-reference-cache-dir",
        default=DEFAULT_MLX_REFERENCE_CACHE_DIR,
        help="reference scorer-input cache for MLX response runs",
    )
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mlx-batch-pairs", type=int, default=1)
    parser.add_argument("--mlx-cache-batch-pairs", type=int, default=8)
    parser.add_argument(
        "--skip-raw-retention-plan",
        action="store_true",
        help="do not add the post-local-CPU raw/inflated scratch retention planning step",
    )
    parser.add_argument(
        "--skip-mlx-retention-plan",
        action="store_true",
        help="do not add the post-response mlx_delta_cache retention planning step",
    )
    parser.add_argument(
        "--include-scheduler-preflight",
        action="store_true",
        help="add one cross-experiment storage/cleanup preflight node that gates candidate work",
    )
    parser.add_argument("--scheduler-storage-tier", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--scheduler-storage-workload-subdir", default=None)
    parser.add_argument("--scheduler-storage-expected-workload-root", default=None)
    parser.add_argument("--scheduler-storage-reserve-free-gb", type=float, default=40.0)
    parser.add_argument("--scheduler-storage-expected-bytes", type=int, default=0)
    parser.add_argument("--scheduler-proactive-cleanup-root", action="append", default=[])
    parser.add_argument("--scheduler-proactive-cleanup-execute", action="store_true")
    parser.add_argument("--scheduler-proactive-cleanup-action", choices=("move", "delete"), default="move")
    parser.add_argument("--scheduler-proactive-cleanup-min-bytes", default="1")
    parser.add_argument("--scheduler-proactive-cleanup-cold-store-root", action="append", default=[])
    parser.add_argument("--scheduler-proactive-cleanup-cold-store-reserve-gb", type=float, default=40.0)
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
        queue_id=args.queue_id,
        completed_results_roots=tuple(args.completed_results_root),
        **{
            key: value
            for key, value in {
                "mlx_effective_selection": args.mlx_effective_selection,
                "decoder_q_candidate_manifest": args.decoder_q_candidate_manifest,
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
        local_io_concurrency=args.local_io_concurrency,
        include_mlx_local_advisory_debug=args.include_mlx_local_advisory_debug,
        allow_large_mlx_cache=args.allow_large_mlx_cache,
        mlx_reference_cache_dir=args.mlx_reference_cache_dir,
        mlx_device=args.mlx_device,
        mlx_batch_pairs=args.mlx_batch_pairs,
        mlx_cache_batch_pairs=args.mlx_cache_batch_pairs,
        include_raw_retention_plan=not args.skip_raw_retention_plan,
        include_mlx_retention_plan=not args.skip_mlx_retention_plan,
        include_scheduler_preflight=args.include_scheduler_preflight,
        scheduler_storage_tiers=tuple(args.scheduler_storage_tier),
        scheduler_storage_workload_subdir=args.scheduler_storage_workload_subdir,
        scheduler_storage_expected_workload_root=args.scheduler_storage_expected_workload_root,
        scheduler_storage_reserve_free_gb=args.scheduler_storage_reserve_free_gb,
        scheduler_storage_expected_bytes=args.scheduler_storage_expected_bytes,
        scheduler_proactive_cleanup_roots=tuple(args.scheduler_proactive_cleanup_root),
        scheduler_proactive_cleanup_execute=args.scheduler_proactive_cleanup_execute,
        scheduler_proactive_cleanup_action=args.scheduler_proactive_cleanup_action,
        scheduler_proactive_cleanup_min_bytes=args.scheduler_proactive_cleanup_min_bytes,
        scheduler_proactive_cleanup_cold_store_roots=tuple(args.scheduler_proactive_cleanup_cold_store_root),
        scheduler_proactive_cleanup_cold_store_reserve_gb=args.scheduler_proactive_cleanup_cold_store_reserve_gb,
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
