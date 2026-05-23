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

from comma_lab.scheduler.dqs1_local_first_harvest import (  # noqa: E402
    DEFAULT_QUEUE_PATH,
    DEFAULT_RESULTS_ROOT,
    Dqs1HarvestResult,
    ExperimentQueueError,
    build_dqs1_harvest_result,
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
from tac.optimization.local_cpu_contest_drift import (  # noqa: E402
    local_cpu_advisory_payload_blockers,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

AUTOPILOT_SCHEMA = "dqs1_local_first_autopilot_result.v1"
REROUTE_LEDGER_SCHEMA = "dqs1_local_first_queue_reroute_record.v1"
SCRATCH_CLEANUP_SCHEMA = "dqs1_local_first_scratch_cleanup.v1"


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)


def _write_json_new(path: Path, payload: object) -> None:
    try:
        write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _json_load_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cleanup_completed_local_cpu_scratch(
    *,
    results_root: Path,
    stamp: str,
) -> dict[str, Any]:
    """Delete rebuildable inflate scratch only after advisory custody exists."""

    materialized = results_root / "materialized"
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if not materialized.is_dir():
        return {
            "schema": SCRATCH_CLEANUP_SCHEMA,
            "timestamp_utc": stamp,
            "results_root": str(results_root),
            "deleted_path_count": 0,
            "deleted_bytes": 0,
            "deleted": [],
            "skipped": [{"path": str(materialized), "reason": "materialized_dir_missing"}],
        }
    for candidate_dir in sorted(path for path in materialized.iterdir() if path.is_dir()):
        advisory_path = candidate_dir / "local_cpu_advisory.json"
        advisory = _json_load_object(advisory_path)
        if advisory is None:
            skipped.append({"path": str(candidate_dir), "reason": "local_cpu_advisory_json_missing_or_invalid"})
            continue
        blockers = local_cpu_advisory_payload_blockers(advisory)
        if blockers:
            skipped.append({"path": str(candidate_dir), "reason": "local_cpu_advisory_contract_blocked:" + ",".join(blockers)})
            continue
        work = candidate_dir / "local_cpu_advisory_work"
        for scratch_name in ("inflated", "extracted"):
            scratch = work / scratch_name
            if not scratch.exists():
                continue
            size = _path_size_bytes(scratch)
            shutil.rmtree(scratch)
            deleted.append(
                {
                    "candidate_id": candidate_dir.name,
                    "path": str(scratch),
                    "bytes": size,
                    "reason": "rebuildable_after_local_cpu_advisory_json_contract_validated",
                }
            )
    return {
        "schema": SCRATCH_CLEANUP_SCHEMA,
        "timestamp_utc": stamp,
        "results_root": str(results_root),
        "deleted_path_count": len(deleted),
        "deleted_bytes": sum(int(row["bytes"]) for row in deleted),
        "deleted": deleted,
        "skipped": skipped,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--state", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-total-steps", type=int, default=64)
    parser.add_argument("--max-steps-per-worker", type=int, default=64)
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
        "--no-cleanup-completed-scratch",
        action="store_true",
        help="keep completed local_cpu_advisory_work inflated/extracted directories",
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
            if args.execute and not args.no_cleanup_completed_scratch:
                cleanup_stamp = _utc_stamp()
                cleanup = _cleanup_completed_local_cpu_scratch(
                    results_root=results_root,
                    stamp=cleanup_stamp,
                )
                preflight["scratch_cleanup"] = {
                    "deleted_path_count": cleanup["deleted_path_count"],
                    "deleted_bytes": cleanup["deleted_bytes"],
                }
                if cleanup["deleted_path_count"]:
                    cleanup_path = _default_cleanup_path(cleanup_stamp)
                    _write_json_new(cleanup_path, cleanup)
                    preflight["scratch_cleanup_path"] = str(cleanup_path)
                preflight["free_disk_gb_after_cleanup"] = _free_disk_gb(results_root)
            free_after_cleanup = float(
                preflight.get("free_disk_gb_after_cleanup", preflight["free_disk_gb_before_worker"])
            )
            if free_after_cleanup < args.min_free_disk_gb:
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
                )
                summary = queue_summary(conn, queue)
            total_steps += int(worker.get("steps_started") or 0)
            round_record: dict[str, Any] = {
                "queue_id": queue["queue_id"],
                "preflight": preflight,
                "worker": worker,
                "summary": summary,
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
            harvest_kwargs: dict[str, Any] = {
                "queue_path": queue_path,
                "repo_root": REPO_ROOT,
                "timestamp": stamp,
                "reroute_observe_only": True,
                "output_queue_path": queue_path,
                "expected_output_queue_sha256": _sha256_text(prior_queue_text)
                if prior_queue_text
                else None,
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
            round_record["harvest"] = {
                "candidate_id": harvest.harvest_record["candidate_id"],
                "recommended_action": harvest.harvest_record["recommended_action"],
                "eureka_trigger": harvest.harvest_record["eureka_trigger"],
                "local_score": harvest.harvest_record["local_score"],
                "conservative_projected_contest_score": harvest.harvest_record[
                    "conservative_projected_contest_score"
                ],
                **artifact_paths,
            }
            rounds.append(round_record)
            if harvest.exact_auth_request is not None:
                stop_reason = "exact_auth_anchor_request_created"
                break
            if harvest.rerouted_queue is None:
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
