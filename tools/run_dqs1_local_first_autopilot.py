#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the DQS1 local-first queue across harvest/reroute boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.artifact_retention import (  # noqa: E402
    DEFAULT_RETENTION_KINDS,
    ArtifactRetentionError,
    build_retention_plan,
    execute_retention_plan,
)
from comma_lab.scheduler.dqs1_local_first_harvest import (  # noqa: E402
    DEFAULT_QUEUE_PATH,
    DEFAULT_RESULTS_ROOT,
    Dqs1HarvestResult,
    ExperimentQueueError,
    build_dqs1_harvest_result,
    candidate_experiment_ids,
)
from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    assert_canonical_state_for_execution,
    connect_state,
    default_state_path,
    initialize_queue_state,
    load_queue_definition,
    queue_summary,
    run_queue_worker,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

AUTOPILOT_SCHEMA = "dqs1_local_first_autopilot_result.v1"
REROUTE_LEDGER_SCHEMA = "dqs1_local_first_queue_reroute_record.v1"
ARTIFACT_RETENTION_SCHEMA = "dqs1_local_first_artifact_retention.v1"


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)


def _parse_bytes(value: str) -> int:
    raw = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * multiplier)
    return int(raw)


def _write_json_new(path: Path, payload: object) -> None:
    try:
        write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc


def _default_retention_path(candidate_id: str, stamp: str, *, repo_root: Path = REPO_ROOT) -> Path:
    return (
        repo_root
        / ".omx/research"
        / f"dqs1_artifact_retention_{_artifact_token(candidate_id)}_{stamp}.json"
    )


def _candidate_root_from_harvest(result: Dqs1HarvestResult, *, repo_root: Path) -> Path:
    advisory_path = Path(str(result.harvest_record["local_cpu_advisory_path"]))
    if not advisory_path.is_absolute():
        advisory_path = repo_root / advisory_path
    return advisory_path.parent


def _execute_candidate_artifact_retention(
    *,
    candidate_root: Path,
    candidate_id: str,
    stamp: str,
    action: str,
    cold_store_roots: list[Path] | None,
    min_bytes: int,
    include_mlx_cache: bool,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Plan and execute retention for one harvested DQS1 candidate.

    The queue already creates a raw-retention plan after local CPU advisory. This
    helper is the actuator: it re-plans immediately before moving/deleting so
    stale plans cannot authorize a destructive operation.
    """

    include_kinds = set(DEFAULT_RETENTION_KINDS)
    if include_mlx_cache:
        include_kinds.add("mlx_scorer_input_cache")
    output_path = _default_retention_path(candidate_id, stamp, repo_root=repo_root)
    journal_path = output_path.with_suffix(output_path.suffix + ".journal.jsonl")
    plan = build_retention_plan(
        [candidate_root],
        repo_root=repo_root,
        include_kinds=include_kinds,
        min_bytes=min_bytes,
    )
    payload: dict[str, Any] = {
        "schema": ARTIFACT_RETENTION_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_root": str(candidate_root),
        "timestamp_utc": stamp,
        "action": action,
        "cold_store_roots": [] if cold_store_roots is None else [str(path) for path in cold_store_roots],
        "include_mlx_cache": include_mlx_cache,
        "plan": plan.to_dict(),
        "execution": None,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if plan.blocked_candidates:
        _write_json_new(output_path, payload)
        blockers = [
            f"{candidate.path}:{','.join(candidate.blockers)}"
            for candidate in plan.blocked_candidates
        ]
        raise ExperimentQueueError(
            f"{candidate_id}: artifact retention blocked; wrote {output_path}: {blockers}"
        )
    if plan.candidates:
        try:
            payload["execution"] = execute_retention_plan(
                plan,
                action=action,
                cold_store_root=(
                    cold_store_roots[0]
                    if cold_store_roots is not None and len(cold_store_roots) == 1
                    else None
                ),
                cold_store_roots=(
                    cold_store_roots
                    if cold_store_roots is not None and len(cold_store_roots) != 1
                    else None
                ),
                journal_path=journal_path,
            )
        except ArtifactRetentionError as exc:
            _write_json_new(output_path, payload)
            raise ExperimentQueueError(str(exc)) from exc
    _write_json_new(output_path, payload)
    return {
        "path": str(output_path),
        "candidate_count": plan.to_dict()["candidate_count"],
        "blocked_candidate_count": plan.to_dict()["blocked_candidate_count"],
        "total_reclaimable_bytes": plan.total_reclaimable_bytes,
        "executed_count": (
            int(payload["execution"]["executed_count"])
            if isinstance(payload.get("execution"), dict)
            else 0
        ),
        "executed_bytes": (
            int(payload["execution"]["executed_bytes"])
            if isinstance(payload.get("execution"), dict)
            else 0
        ),
        "journal_path": (
            payload["execution"].get("journal_path")
            if isinstance(payload.get("execution"), dict)
            else None
        ),
    }


def _free_disk_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return float(usage.free) / (1024.0**3)


def _default_cleanup_path(stamp: str) -> Path:
    return REPO_ROOT / ".omx/research" / f"dqs1_autopilot_scratch_cleanup_{stamp}.json"


def _default_harvest_path(candidate_id: str, stamp: str) -> Path:
    return (
        REPO_ROOT
        / ".omx/research"
        / f"dqs1_local_first_harvest_{_artifact_token(candidate_id)}_{stamp}.json"
    )


def _default_request_path(candidate_id: str, stamp: str) -> Path:
    return (
        REPO_ROOT
        / ".omx/research"
        / f"exact_auth_anchor_request_{_artifact_token(candidate_id)}_{stamp}.json"
    )


def _default_reroute_path(candidate_id: str, stamp: str) -> Path:
    return (
        REPO_ROOT
        / ".omx/research"
        / f"dqs1_local_first_queue_reroute_{_artifact_token(candidate_id)}_{stamp}.json"
    )


def _write_harvest_artifacts(
    result: Dqs1HarvestResult,
    *,
    queue_path: Path,
    prior_queue_text: str,
    stamp: str,
) -> dict[str, str | None]:
    candidate_id = str(result.harvest_record["candidate_id"])
    harvest_path = _default_harvest_path(candidate_id, stamp)
    _write_json_new(harvest_path, result.harvest_record)

    request_path = None
    if result.exact_auth_request is not None:
        request_out = _default_request_path(candidate_id, stamp)
        _write_json_new(request_out, result.exact_auth_request)
        request_path = str(request_out)

    reroute_path = None
    if result.rerouted_queue is not None:
        new_queue_text = queue_path.read_text(encoding="utf-8")
        reroute_out = _default_reroute_path(candidate_id, stamp)
        _write_json_new(
            reroute_out,
            {
                "schema": REROUTE_LEDGER_SCHEMA,
                "candidate_id": candidate_id,
                "harvest_record_path": str(harvest_path),
                "queue_path": str(queue_path),
                "previous_queue_sha256": _sha256_text(prior_queue_text)
                if prior_queue_text
                else None,
                "new_queue_sha256": _sha256_text(new_queue_text),
                "rerouted_queue_experiment_ids": [
                    str(experiment.get("id"))
                    for experiment in result.rerouted_queue.get("experiments", [])
                    if isinstance(experiment, dict)
                ],
                "source_eureka_signal_path": result.harvest_record["eureka_signal_path"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )
        reroute_path = str(reroute_out)

    return {
        "harvest_path": str(harvest_path),
        "exact_auth_request_path": request_path,
        "reroute_ledger_path": reroute_path,
    }


def _has_active_work(summary: dict[str, Any]) -> bool:
    counts = summary.get("status_counts")
    if not isinstance(counts, dict):
        return False
    return any(int(counts.get(status) or 0) > 0 for status in ("queued", "running", "failed", "blocked"))


def _compact_queue_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep autopilot output readable even when queue state has many orphans."""

    ready_steps = summary.get("ready_steps")
    status_counts = summary.get("status_counts")
    return {
        "mode": summary.get("mode"),
        "status_counts": status_counts if isinstance(status_counts, dict) else {},
        "ready_step_count": len(ready_steps) if isinstance(ready_steps, list) else 0,
        "orphaned_step_count": int(summary.get("orphaned_step_count") or 0),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--state", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-total-steps", type=int, default=64)
    parser.add_argument("--max-steps-per-worker", type=int, default=64)
    parser.add_argument(
        "--max-worker-experiments",
        type=int,
        default=1,
        help=(
            "maximum independent candidate experiments one worker round may "
            "advance concurrently; keep at 1 for legacy serial harvest/reroute"
        ),
    )
    parser.add_argument("--idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-cycles", type=int, default=1)
    parser.add_argument("--action-summary", default="latest")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--log-root", default=None)
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=40.0,
        help="fail closed before launching another worker round below this free-space floor",
    )
    parser.add_argument(
        "--retention-action",
        choices=("delete", "move"),
        default="move",
        help="post-harvest action for certified rebuildable raw/cache artifacts",
    )
    parser.add_argument(
        "--retention-cold-store-root",
        type=Path,
        action="append",
        default=[],
        help="external cold-store root required when --retention-action=move; repeat for tiered moves",
    )
    parser.add_argument(
        "--retention-min-bytes",
        type=_parse_bytes,
        default=1,
        help="minimum candidate size for post-harvest artifact retention",
    )
    parser.add_argument(
        "--include-mlx-cache-retention",
        action="store_true",
        help="also compact certified MLX scorer input caches for harvested candidates",
    )
    parser.add_argument(
        "--no-post-harvest-retention",
        dest="post_harvest_retention",
        action="store_false",
        default=True,
        help="do not execute post-harvest artifact retention",
    )
    parser.add_argument(
        "--no-cleanup-completed-scratch",
        dest="post_harvest_retention",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be >= 1")
    if args.max_total_steps < 1:
        raise SystemExit("--max-total-steps must be >= 1")
    if args.max_steps_per_worker < 1:
        raise SystemExit("--max-steps-per-worker must be >= 1")
    if args.max_worker_experiments < 1:
        raise SystemExit("--max-worker-experiments must be >= 1")

    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = REPO_ROOT / queue_path
    queue = load_queue_definition(queue_path)
    state = Path(args.state) if args.state else default_state_path(REPO_ROOT, queue["queue_id"])
    if args.execute:
        assert_canonical_state_for_execution(REPO_ROOT, queue["queue_id"], state)

    stop_signals: list[int] = []
    prior_sigint = signal.getsignal(signal.SIGINT)
    prior_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_stop(signum: int, _frame: object) -> None:
        if signum not in stop_signals:
            stop_signals.append(signum)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    rounds: list[dict[str, Any]] = []
    total_steps = 0
    candidates_harvested = 0
    stop_reason = "max_candidates_reached"
    try:
        while candidates_harvested < args.max_candidates and total_steps < args.max_total_steps:
            if stop_signals:
                stop_reason = "stop_requested"
                break
            results_root = (
                Path(args.results_root)
                if args.results_root is not None
                else REPO_ROOT / DEFAULT_RESULTS_ROOT
            )
            if not results_root.is_absolute():
                results_root = REPO_ROOT / results_root
            preflight: dict[str, Any] = {
                "free_disk_gb_before_worker": _free_disk_gb(results_root),
                "min_free_disk_gb": args.min_free_disk_gb,
            }
            if preflight["free_disk_gb_before_worker"] < args.min_free_disk_gb:
                rounds.append(
                    {
                        "queue_id": queue["queue_id"],
                        "terminal": "insufficient_free_disk",
                        "preflight": preflight,
                    }
                )
                stop_reason = "insufficient_free_disk"
                break
            queue = load_queue_definition(queue_path)
            with connect_state(state) as conn:
                initialize_queue_state(conn, queue)
                worker = run_queue_worker(
                    conn,
                    queue,
                    repo_root=REPO_ROOT,
                    execute=args.execute,
                    max_steps=min(
                        args.max_steps_per_worker,
                        args.max_total_steps - total_steps,
                    ),
                    idle_sleep_seconds=args.idle_sleep_seconds,
                    max_idle_cycles=args.max_idle_cycles,
                    allow_cloud=args.allow_cloud,
                    log_root=args.log_root,
                    stop_requested=lambda: bool(stop_signals),
                    reload_queue=lambda: load_queue_definition(queue_path),
                    max_experiments=args.max_worker_experiments,
                )
                summary = queue_summary(conn, queue)
            total_steps += int(worker.get("steps_started") or 0)
            round_record: dict[str, Any] = {
                "queue_id": queue["queue_id"],
                "preflight": preflight,
                "worker": worker,
                "summary": _compact_queue_summary(summary),
            }
            if worker.get("failure_count"):
                round_record["terminal"] = "worker_failure"
                rounds.append(round_record)
                stop_reason = "worker_failure"
                break
            if worker.get("execute") is False:
                round_record["terminal"] = "dry_run"
                rounds.append(round_record)
                stop_reason = "dry_run"
                break
            if _has_active_work(summary):
                round_record["terminal"] = "active_work_remaining"
                rounds.append(round_record)
                if total_steps >= args.max_total_steps:
                    stop_reason = "max_total_steps_reached"
                    break
                continue

            prior_queue_text = queue_path.read_text(encoding="utf-8") if queue_path.is_file() else ""
            stamp = _utc_stamp()
            batch_mode = args.max_worker_experiments > 1
            ids = candidate_experiment_ids(queue) if batch_mode else [""]
            ids = ids[: max(0, args.max_candidates - candidates_harvested)]
            harvest_summaries: list[dict[str, Any]] = []
            retention_summaries: list[dict[str, Any]] = []
            rerouted = False
            exact_request_created = False
            for candidate_id in ids:
                harvest_kwargs: dict[str, Any] = {
                    "queue_path": queue_path,
                    "repo_root": REPO_ROOT,
                    "candidate_id": candidate_id or None,
                    "timestamp": stamp,
                    "reroute_observe_only": not batch_mode,
                    "output_queue_path": None if batch_mode else queue_path,
                    "expected_output_queue_sha256": (
                        None
                        if batch_mode or not prior_queue_text
                        else _sha256_text(prior_queue_text)
                    ),
                    "action_summary": args.action_summary,
                }
                if args.results_root is not None:
                    harvest_kwargs["results_root"] = args.results_root
                harvest = build_dqs1_harvest_result(**harvest_kwargs)
                artifact_paths = _write_harvest_artifacts(
                    harvest,
                    queue_path=queue_path,
                    prior_queue_text=prior_queue_text,
                    stamp=stamp,
                )
                candidates_harvested += 1
                exact_request_created = exact_request_created or harvest.exact_auth_request is not None
                rerouted = rerouted or harvest.rerouted_queue is not None
                harvest_summaries.append(
                    {
                        "candidate_id": harvest.harvest_record["candidate_id"],
                        "recommended_action": harvest.harvest_record["recommended_action"],
                        "eureka_trigger": harvest.harvest_record["eureka_trigger"],
                        "local_score": harvest.harvest_record["local_score"],
                        "conservative_projected_contest_score": harvest.harvest_record[
                            "conservative_projected_contest_score"
                        ],
                        **artifact_paths,
                    }
                )
                if args.execute and args.post_harvest_retention:
                    retention_summaries.append(
                        _execute_candidate_artifact_retention(
                            candidate_root=_candidate_root_from_harvest(harvest, repo_root=REPO_ROOT),
                            candidate_id=str(harvest.harvest_record["candidate_id"]),
                            stamp=stamp,
                            action=args.retention_action,
                            cold_store_roots=args.retention_cold_store_root,
                            min_bytes=args.retention_min_bytes,
                            include_mlx_cache=args.include_mlx_cache_retention,
                            repo_root=REPO_ROOT,
                        )
                    )
            round_record["harvests"] = harvest_summaries
            if len(harvest_summaries) == 1:
                round_record["harvest"] = harvest_summaries[0]
            if retention_summaries:
                round_record["post_harvest_retention"] = {
                    "mode": "batch" if batch_mode else "single",
                    "candidate_count": len(retention_summaries),
                    "items": retention_summaries,
                    "executed_bytes": sum(int(row.get("executed_bytes") or 0) for row in retention_summaries),
                }
                round_record["preflight"]["free_disk_gb_after_post_harvest_retention"] = _free_disk_gb(
                    results_root
                )
            rounds.append(round_record)
            if exact_request_created:
                stop_reason = "exact_auth_anchor_request_created"
                break
            if batch_mode:
                stop_reason = "batch_harvested_waiting_for_portfolio_rebuild"
                break
            if not rerouted:
                stop_reason = "no_reroute_available"
                break
        else:
            if total_steps >= args.max_total_steps:
                stop_reason = "max_total_steps_reached"
    finally:
        signal.signal(signal.SIGINT, prior_sigint)
        signal.signal(signal.SIGTERM, prior_sigterm)

    _json_print(
        {
            "schema": AUTOPILOT_SCHEMA,
            "queue_path": str(queue_path),
            "state": str(state),
            "execute": args.execute,
            "allow_cloud": args.allow_cloud,
            "max_worker_experiments": args.max_worker_experiments,
            "stop_reason": stop_reason,
            "total_steps_started": total_steps,
            "candidates_harvested": candidates_harvested,
            "stop_signals": [signal.Signals(signum).name for signum in stop_signals],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rounds": rounds,
        }
    )
    return 2 if stop_reason in {"worker_failure"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
