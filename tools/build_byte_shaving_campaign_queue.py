#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile byte-shaving campaign plans into DQS1 local-first queue inputs."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.byte_shaving_campaign_queue import (  # noqa: E402
    build_materializer_execution_queue,
    compile_dqs1_byte_shaving_campaign,
    materializer_contexts_from_payload,
)
from comma_lab.scheduler.dqs1_local_first_queue import (  # noqa: E402
    DEFAULT_QUEUE_ID,
    DEFAULT_RESULTS_ROOT,
    build_queue_from_action_summary,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.final_byte_operation_contexts import (  # noqa: E402
    build_final_byte_operation_contexts,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

AUTO_LOCAL_CPU_CONCURRENCY = "auto"


def _auto_local_cpu_concurrency(*, cpu_count: int | None = None) -> int:
    count = os.cpu_count() if cpu_count is None else cpu_count
    if count is None or count < 1:
        return 1
    return count


def _parse_local_cpu_concurrency(value: str) -> int:
    text = str(value).strip().lower()
    if text == AUTO_LOCAL_CPU_CONCURRENCY:
        return _auto_local_cpu_concurrency()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise SystemExit(
            "--local-cpu-concurrency must be a positive integer or 'auto'"
        ) from exc
    if parsed < 1:
        raise SystemExit("--local-cpu-concurrency must be >= 1 or 'auto'")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
) -> None:
    try:
        write_json_artifact(
            path,
            payload,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_sha256,
        )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc


def _parse_pair_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError as exc:
            raise SystemExit(f"--base-pair-indices contains a non-integer: {item!r}") from exc
    if not out:
        raise SystemExit("--base-pair-indices must contain at least one integer")
    return out


def _parse_resource_concurrency(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(
                "--materializer-resource-concurrency entries must be KIND=LIMIT"
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--materializer-resource-concurrency KIND must be non-empty")
        try:
            limit = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"--materializer-resource-concurrency has non-integer limit: {raw!r}"
            ) from exc
        if limit < 1:
            raise SystemExit(
                f"--materializer-resource-concurrency limit must be >= 1: {raw!r}"
            )
        parsed[key] = limit
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialization-out", type=Path, required=True)
    parser.add_argument("--portfolio-out", type=Path, required=True)
    parser.add_argument("--action-summary-out", type=Path, required=True)
    parser.add_argument("--materializer-backlog-out", type=Path, default=None)
    parser.add_argument(
        "--materializer-contexts",
        type=Path,
        default=None,
        help="JSON byte_shaving_materializer_contexts.v1 file used to unblock proof-chain work rows.",
    )
    parser.add_argument(
        "--materializer-artifact-map",
        type=Path,
        default=None,
        help=(
            "JSON artifact/custody hints used to derive materializer contexts "
            "from the generated backlog before the final queue compile."
        ),
    )
    parser.add_argument(
        "--materializer-contexts-out",
        type=Path,
        default=None,
        help="Write generated byte_shaving_materializer_contexts.v1 when using --materializer-artifact-map.",
    )
    parser.add_argument(
        "--materializer-context-default-output-root",
        type=Path,
        default=None,
        help="Default output root used by the context compiler when artifact hints omit output paths.",
    )
    parser.add_argument(
        "--materializer-contexts-fail-if-blocked",
        action="store_true",
        help="Exit nonzero if generated materializer contexts still carry context_blockers.",
    )
    parser.add_argument("--materializer-work-queue-out", type=Path, default=None)
    parser.add_argument(
        "--materializer-execution-queue-out",
        type=Path,
        default=None,
        help=(
            "Write an experiment_queue.v1 file that runs executable "
            "byte_shaving_materializer_work_queue.v1 rows through the shared worker."
        ),
    )
    parser.add_argument("--queue-out", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--base-pair-indices",
        default=None,
        help="comma-separated DQS1 base pairset; required unless the plan carries dqs1_base_pair_indices",
    )
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--queue-candidate-limit", type=int, default=2)
    parser.add_argument("--allow-partial-materialization", action="store_true")
    parser.add_argument("--partial-materialization-rationale", default=None)
    parser.add_argument("--queue-id", default=DEFAULT_QUEUE_ID)
    parser.add_argument(
        "--include-scheduler-preflight",
        action="store_true",
        help="gate generated DQS1 queue work on storage-tier planning and proactive cleanup",
    )
    parser.add_argument(
        "--scheduler-storage-tier",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--scheduler-storage-workload-subdir", default=None)
    parser.add_argument("--scheduler-storage-expected-workload-root", default=None)
    parser.add_argument("--scheduler-storage-reserve-free-gb", type=float, default=40.0)
    parser.add_argument("--scheduler-storage-expected-bytes", type=int, default=0)
    parser.add_argument("--scheduler-proactive-cleanup-root", action="append", default=[])
    parser.add_argument("--scheduler-proactive-cleanup-execute", action="store_true")
    parser.add_argument(
        "--scheduler-proactive-cleanup-action",
        choices=("move", "delete"),
        default="move",
    )
    parser.add_argument("--scheduler-proactive-cleanup-min-bytes", default="1")
    parser.add_argument("--scheduler-proactive-cleanup-cold-store-root", action="append", default=[])
    parser.add_argument("--scheduler-proactive-cleanup-cold-store-reserve-gb", type=float, default=40.0)
    parser.add_argument(
        "--materializer-execution-queue-id",
        default="byte_shaving_materializer_local_proof_chain",
    )
    parser.add_argument("--materializer-execution-lane-id", default=None)
    parser.add_argument("--materializer-execution-limit", type=int, default=None)
    parser.add_argument("--materializer-execution-timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--materializer-resource-concurrency",
        action="append",
        default=[],
        metavar="KIND=LIMIT",
        help=(
            "Override materializer execution concurrency by resource kind, "
            "for example local_cpu=8 or local_mlx=2."
        ),
    )
    parser.add_argument(
        "--include-materializer-scheduler-preflight",
        action="store_true",
        help="gate materializer execution work on storage-tier planning and proactive cleanup",
    )
    parser.add_argument(
        "--materializer-scheduler-storage-tier",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--materializer-scheduler-storage-workload-subdir", default=None)
    parser.add_argument("--materializer-scheduler-storage-expected-workload-root", default=None)
    parser.add_argument(
        "--materializer-scheduler-storage-reserve-free-gb",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--materializer-scheduler-storage-expected-bytes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--materializer-scheduler-proactive-cleanup-root",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--materializer-scheduler-proactive-cleanup-execute",
        action="store_true",
    )
    parser.add_argument(
        "--materializer-scheduler-proactive-cleanup-action",
        choices=("move", "delete"),
        default="move",
    )
    parser.add_argument("--materializer-scheduler-proactive-cleanup-min-bytes", default="1")
    parser.add_argument(
        "--materializer-scheduler-proactive-cleanup-cold-store-root",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--materializer-scheduler-proactive-cleanup-cold-store-reserve-gb",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--include-materializer-exact-readiness-followup",
        "--include-materializer-exact-eval-handoff",
        dest="include_materializer_exact_readiness_followup",
        action="store_true",
        help=(
            "append per-materializer harvest, exact-readiness bridge, and paused "
            "dry-run dispatch-plan steps under each materializer output root"
        ),
    )
    parser.add_argument(
        "--materializer-exact-readiness-followup-require-ready",
        "--materializer-exact-eval-require-ready",
        dest="materializer_exact_readiness_followup_require_ready",
        action="store_true",
        help=(
            "make follow-up harvest fail if no exact-ready row is emitted; default "
            "keeps blocked readiness as durable signal"
        ),
    )
    parser.add_argument(
        "--materializer-exact-eval-dispatch-require-authorized",
        action="store_true",
        help="make generated dispatch-plan step exit nonzero if no row is authorized",
    )
    parser.add_argument(
        "--materializer-exact-eval-dispatch-provider",
        choices=("lightning", "vastai"),
        default="lightning",
    )
    parser.add_argument(
        "--materializer-exact-eval-dispatch-label-prefix",
        default="materializer_exact_eval",
    )
    parser.add_argument(
        "--materializer-exact-eval-dispatch-estimated-cost-per-dispatch",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--materializer-exact-eval-dispatch-max-total-cost",
        type=float,
        default=5.00,
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--completed-results-root", action="append", default=[])
    parser.add_argument(
        "--local-cpu-concurrency",
        default=AUTO_LOCAL_CPU_CONCURRENCY,
        help=(
            "local_cpu resource cap for generated queues; use 'auto' to use "
            "os.cpu_count() on this machine"
        ),
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--expected-materialization-sha256")
    parser.add_argument("--expected-portfolio-sha256")
    parser.add_argument("--expected-action-summary-sha256")
    parser.add_argument("--expected-materializer-backlog-sha256")
    parser.add_argument("--expected-materializer-contexts-sha256")
    parser.add_argument("--expected-materializer-work-queue-sha256")
    parser.add_argument("--expected-materializer-execution-queue-sha256")
    parser.add_argument("--expected-queue-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be >= 1")
    if args.queue_candidate_limit < 1:
        raise SystemExit("--queue-candidate-limit must be >= 1")
    local_cpu_concurrency = _parse_local_cpu_concurrency(args.local_cpu_concurrency)
    if args.materializer_execution_limit is not None and args.materializer_execution_limit < 1:
        raise SystemExit("--materializer-execution-limit must be >= 1")
    if args.materializer_execution_timeout_seconds < 0:
        raise SystemExit("--materializer-execution-timeout-seconds must be non-negative")
    if args.allow_partial_materialization and not str(
        args.partial_materialization_rationale or ""
    ).strip():
        raise SystemExit(
            "--allow-partial-materialization requires --partial-materialization-rationale"
        )
    if not args.plan.is_file():
        raise SystemExit(f"--plan does not exist: {args.plan}")
    if args.materializer_contexts is not None and args.materializer_artifact_map is not None:
        raise SystemExit(
            "--materializer-contexts and --materializer-artifact-map are mutually exclusive"
        )
    if args.materializer_artifact_map is not None:
        if args.materializer_contexts_out is None:
            raise SystemExit(
                "--materializer-artifact-map requires --materializer-contexts-out"
            )
        if not args.materializer_artifact_map.is_file():
            raise SystemExit(
                f"--materializer-artifact-map does not exist: {args.materializer_artifact_map}"
            )

    plan = _load_json(args.plan)
    materializer_contexts = None
    generated_materializer_contexts_payload = None
    if args.materializer_contexts is not None:
        if not args.materializer_contexts.is_file():
            raise SystemExit(f"--materializer-contexts does not exist: {args.materializer_contexts}")
        try:
            materializer_contexts = materializer_contexts_from_payload(
                _load_json(args.materializer_contexts)
            )
        except ExperimentQueueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.materializer_artifact_map is not None:
        try:
            preliminary = compile_dqs1_byte_shaving_campaign(
                plan,
                repo_root=args.repo_root,
                plan_path=args.plan,
                base_pair_indices=_parse_pair_indices(args.base_pair_indices),
                candidate_limit=args.candidate_limit,
                portfolio_json=str(args.portfolio_out),
                allow_partial_materialization=bool(args.allow_partial_materialization),
                partial_materialization_rationale=(
                    args.partial_materialization_rationale
                ),
                materializer_contexts=None,
            )
            generated_materializer_contexts_payload = (
                build_final_byte_operation_contexts(
                    preliminary["materializer_backlog"],
                    artifact_map=_load_json(args.materializer_artifact_map),
                    repo_root=args.repo_root,
                    default_output_root=(
                        args.materializer_context_default_output_root
                    ),
                )
            )
            if (
                args.materializer_contexts_fail_if_blocked
                and int(
                    generated_materializer_contexts_payload.get(
                        "blocked_context_count"
                    )
                    or 0
                )
                > 0
            ):
                raise ExperimentQueueError(
                    "generated materializer contexts still have blockers"
                )
            assert args.materializer_contexts_out is not None
            _write_json(
                args.materializer_contexts_out,
                generated_materializer_contexts_payload,
                allow_overwrite=bool(args.overwrite_output),
                expected_existing_sha256=args.expected_materializer_contexts_sha256,
            )
            materializer_contexts = materializer_contexts_from_payload(
                generated_materializer_contexts_payload
            )
        except ExperimentQueueError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        compiled = compile_dqs1_byte_shaving_campaign(
            plan,
            repo_root=args.repo_root,
            plan_path=args.plan,
            base_pair_indices=_parse_pair_indices(args.base_pair_indices),
            candidate_limit=args.candidate_limit,
            portfolio_json=str(args.portfolio_out),
            allow_partial_materialization=bool(args.allow_partial_materialization),
            partial_materialization_rationale=args.partial_materialization_rationale,
            materializer_contexts=materializer_contexts,
        )
    except ExperimentQueueError as exc:
        raise SystemExit(str(exc)) from exc

    _write_json(
        args.materialization_out,
        compiled,
        allow_overwrite=bool(args.overwrite_output),
        expected_existing_sha256=args.expected_materialization_sha256,
    )
    _write_json(
        args.portfolio_out,
        compiled["portfolio"],
        allow_overwrite=bool(args.overwrite_output),
        expected_existing_sha256=args.expected_portfolio_sha256,
    )
    _write_json(
        args.action_summary_out,
        compiled["action_summary"],
        allow_overwrite=bool(args.overwrite_output),
        expected_existing_sha256=args.expected_action_summary_sha256,
    )
    if args.materializer_backlog_out is not None:
        _write_json(
            args.materializer_backlog_out,
            compiled["materializer_backlog"],
            allow_overwrite=bool(args.overwrite_output),
            expected_existing_sha256=args.expected_materializer_backlog_sha256,
        )
    if args.materializer_work_queue_out is not None:
        _write_json(
            args.materializer_work_queue_out,
            compiled["materializer_work_queue"],
            allow_overwrite=bool(args.overwrite_output),
            expected_existing_sha256=args.expected_materializer_work_queue_sha256,
        )

    materializer_execution_queue_payload = None
    if args.materializer_execution_queue_out is not None:
        try:
            materializer_execution_queue = build_materializer_execution_queue(
                compiled["materializer_work_queue"],
                queue_id=args.materializer_execution_queue_id,
                repo_root=args.repo_root,
                lane_id=args.materializer_execution_lane_id,
                source_work_queue_path=args.materializer_work_queue_out,
                local_cpu_concurrency=local_cpu_concurrency,
                resource_concurrency=_parse_resource_concurrency(
                    args.materializer_resource_concurrency
                ),
                step_timeout_seconds=args.materializer_execution_timeout_seconds,
                limit=args.materializer_execution_limit,
                include_scheduler_preflight=args.include_materializer_scheduler_preflight,
                scheduler_results_root=args.results_root,
                scheduler_storage_tiers=tuple(args.materializer_scheduler_storage_tier),
                scheduler_storage_workload_subdir=(
                    args.materializer_scheduler_storage_workload_subdir
                ),
                scheduler_storage_expected_workload_root=(
                    args.materializer_scheduler_storage_expected_workload_root
                ),
                scheduler_storage_reserve_free_gb=(
                    args.materializer_scheduler_storage_reserve_free_gb
                ),
                scheduler_storage_expected_bytes=(
                    args.materializer_scheduler_storage_expected_bytes
                ),
                scheduler_proactive_cleanup_roots=tuple(
                    args.materializer_scheduler_proactive_cleanup_root
                ),
                scheduler_proactive_cleanup_execute=(
                    args.materializer_scheduler_proactive_cleanup_execute
                ),
                scheduler_proactive_cleanup_action=(
                    args.materializer_scheduler_proactive_cleanup_action
                ),
                scheduler_proactive_cleanup_min_bytes=(
                    args.materializer_scheduler_proactive_cleanup_min_bytes
                ),
                scheduler_proactive_cleanup_cold_store_roots=tuple(
                    args.materializer_scheduler_proactive_cleanup_cold_store_root
                ),
                scheduler_proactive_cleanup_cold_store_reserve_gb=(
                    args.materializer_scheduler_proactive_cleanup_cold_store_reserve_gb
                ),
                include_exact_readiness_followup=(
                    args.include_materializer_exact_readiness_followup
                ),
                exact_readiness_followup_require_ready=(
                    args.materializer_exact_readiness_followup_require_ready
                ),
                exact_eval_dispatch_require_authorized=(
                    args.materializer_exact_eval_dispatch_require_authorized
                ),
                exact_eval_dispatch_provider=(
                    args.materializer_exact_eval_dispatch_provider
                ),
                exact_eval_dispatch_label_prefix=(
                    args.materializer_exact_eval_dispatch_label_prefix
                ),
                exact_eval_dispatch_estimated_cost_per_dispatch=(
                    args.materializer_exact_eval_dispatch_estimated_cost_per_dispatch
                ),
                exact_eval_dispatch_max_total_cost=(
                    args.materializer_exact_eval_dispatch_max_total_cost
                ),
            )
        except ExperimentQueueError as exc:
            raise SystemExit(str(exc)) from exc
        _write_json(
            args.materializer_execution_queue_out,
            materializer_execution_queue,
            allow_overwrite=bool(args.overwrite_output),
            expected_existing_sha256=args.expected_materializer_execution_queue_sha256,
        )
        materializer_execution_queue_payload = {
            "queue_out": str(args.materializer_execution_queue_out),
            "queue_id": materializer_execution_queue["queue_id"],
            "experiment_count": len(materializer_execution_queue["experiments"]),
            "selected_work_ids": [
                experiment["metadata"]["work_id"]
                for experiment in materializer_execution_queue["experiments"]
                if isinstance(experiment.get("metadata"), dict)
                and "work_id" in experiment["metadata"]
            ],
            "exact_readiness_followup": bool(
                args.include_materializer_exact_readiness_followup
            ),
        }

    queue_payload = None
    if args.queue_out is not None:
        if int(compiled["blocked_row_count"]) > 0 and not args.allow_partial_materialization:
            raise SystemExit(
                "blocked byte-shaving materialization rows present; refusing partial "
                "queue build without --allow-partial-materialization and rationale"
            )
        if int(compiled["executable_row_count"]) <= 0:
            raise SystemExit("no executable DQS1 byte-shaving rows; refusing to build queue")
        if int(compiled["queueable_row_count"]) <= 0:
            raise SystemExit("no queueable DQS1 byte-shaving rows; refusing to build queue")
        queue_result = build_queue_from_action_summary(
            args.action_summary_out,
            repo_root=args.repo_root,
            results_root=args.results_root,
            queue_id=args.queue_id,
            completed_results_roots=tuple(args.completed_results_root),
            candidate_limit=args.queue_candidate_limit,
            local_cpu_concurrency=local_cpu_concurrency,
            include_scheduler_preflight=args.include_scheduler_preflight,
            scheduler_storage_tiers=tuple(args.scheduler_storage_tier),
            scheduler_storage_workload_subdir=args.scheduler_storage_workload_subdir,
            scheduler_storage_expected_workload_root=(
                args.scheduler_storage_expected_workload_root
            ),
            scheduler_storage_reserve_free_gb=args.scheduler_storage_reserve_free_gb,
            scheduler_storage_expected_bytes=args.scheduler_storage_expected_bytes,
            scheduler_proactive_cleanup_roots=tuple(args.scheduler_proactive_cleanup_root),
            scheduler_proactive_cleanup_execute=args.scheduler_proactive_cleanup_execute,
            scheduler_proactive_cleanup_action=args.scheduler_proactive_cleanup_action,
            scheduler_proactive_cleanup_min_bytes=args.scheduler_proactive_cleanup_min_bytes,
            scheduler_proactive_cleanup_cold_store_roots=tuple(
                args.scheduler_proactive_cleanup_cold_store_root
            ),
            scheduler_proactive_cleanup_cold_store_reserve_gb=(
                args.scheduler_proactive_cleanup_cold_store_reserve_gb
            ),
        )
        _write_json(
            args.queue_out,
            queue_result.queue,
            allow_overwrite=bool(args.overwrite_output),
            expected_existing_sha256=args.expected_queue_sha256,
        )
        queue_payload = {
            "queue_out": str(args.queue_out),
            "queue_id": queue_result.queue["queue_id"],
            "experiment_count": len(queue_result.queue["experiments"]),
            "selected_candidate_ids": [
                selection.candidate_id for selection in queue_result.selections
            ],
        }

    print(
        json.dumps(
            {
                "schema": "byte_shaving_campaign_queue_build_result.v1",
                "materialization_out": str(args.materialization_out),
                "portfolio_out": str(args.portfolio_out),
                "action_summary_out": str(args.action_summary_out),
                "materializer_backlog_out": (
                    str(args.materializer_backlog_out)
                    if args.materializer_backlog_out is not None
                    else None
                ),
                "materializer_contexts": (
                    str(args.materializer_contexts)
                    if args.materializer_contexts is not None
                    else (
                        str(args.materializer_contexts_out)
                        if generated_materializer_contexts_payload is not None
                        else None
                    )
                ),
                "materializer_contexts_generated": (
                    generated_materializer_contexts_payload is not None
                ),
                "materializer_contexts_blocked_count": (
                    generated_materializer_contexts_payload.get(
                        "blocked_context_count"
                    )
                    if generated_materializer_contexts_payload is not None
                    else None
                ),
                "materializer_work_queue_out": (
                    str(args.materializer_work_queue_out)
                    if args.materializer_work_queue_out is not None
                    else None
                ),
                "materializer_execution_queue": materializer_execution_queue_payload,
                "local_cpu_concurrency": local_cpu_concurrency,
                "local_cpu_concurrency_requested": str(args.local_cpu_concurrency),
                "executable_row_count": compiled["executable_row_count"],
                "blocked_row_count": compiled["blocked_row_count"],
                "dqs1_executable_row_count": compiled["executable_row_count"],
                "dqs1_blocked_row_count": compiled["blocked_row_count"],
                "materializer_backlog_row_count": compiled["materializer_backlog"][
                    "backlog_row_count"
                ],
                "materializer_work_queue_row_count": compiled[
                    "materializer_work_queue"
                ]["row_count"],
                "materializer_work_queue_executable_row_count": compiled[
                    "materializer_work_queue"
                ]["executable_row_count"],
                "materializer_work_queue_blocked_row_count": compiled[
                    "materializer_work_queue"
                ]["blocked_row_count"],
                "queue": queue_payload,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
