#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operate a declarative experiment queue with SQLite state.

The queue definition is a JSON-compatible ``.json``/``.yaml`` file. State,
telemetry, pause/freeze controls, and rewinds live in SQLite so the definition
can remain the editable source of intent while execution history stays
append-only and queryable.
"""
from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    BLOCKING_ORPHAN_STATUSES,
    ExperimentQueueError,
    apply_scheduler_runtime_policy,
    assert_canonical_state_for_execution,
    assert_no_orphaned_steps_for_execution,
    connect_state,
    connect_state_readonly,
    default_state_path,
    derive_scheduler_runtime_policy,
    initialize_queue_state,
    load_queue_definition,
    orphaned_step_rows,
    queue_performance_summary,
    queue_summary,
    ready_steps,
    reconcile_satisfied_queued_steps,
    reconcile_stale_running_steps,
    resolve_worker_max_parallel,
    retire_orphaned_steps,
    rewind_step,
    run_queue_worker,
    run_ready_step,
    set_control_mode,
)
from comma_lab.scheduler.experiment_queue_observer import (  # noqa: E402
    observe_experiment_queue,
    render_observation_markdown,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

_PLACEHOLDER_RATIONALES = {"", "n/a", "na", "none", "test", "true", "yes", "because"}
STATE_RECONCILIATION_SCHEMA = "experiment_queue_state_reconciliation.v1"


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _load(args: argparse.Namespace) -> tuple[dict, Path]:
    queue = load_queue_definition(args.queue)
    state = Path(args.state) if args.state else default_state_path(REPO_ROOT, queue["queue_id"])
    return queue, state


def _rationale_text(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) < 12 or text.lower() in _PLACEHOLDER_RATIONALES:
        raise ExperimentQueueError(f"{label} must be a specific non-placeholder rationale")
    return text


def cmd_validate(args: argparse.Namespace) -> int:
    queue, _state = _load(args)
    local_parallel, local_limits = resolve_worker_max_parallel(
        queue,
        0,
        allow_cloud=False,
    )
    cloud_parallel, cloud_limits = resolve_worker_max_parallel(
        queue,
        0,
        allow_cloud=True,
    )
    _json_print(
        {
            "valid": True,
            "queue_id": queue["queue_id"],
            "experiment_count": len(queue["experiments"]),
            "step_count": sum(len(exp["steps"]) for exp in queue["experiments"]),
            "auto_parallelism": {
                "local_only": {
                    "max_parallel": local_parallel,
                    "resource_limits": local_limits,
                },
                "with_cloud": {
                    "max_parallel": cloud_parallel,
                    "resource_limits": cloud_limits,
                },
            },
        }
    )
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        summary = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        summary = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_performance(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state_readonly(state) as conn:
        payload = queue_performance_summary(conn, queue)
    output_payload = {"state": str(state), **payload}
    if args.output is not None:
        try:
            artifact = write_json_artifact(
                args.output,
                output_payload,
                allow_overwrite=args.expected_output_sha256 is not None,
                expected_existing_sha256=args.expected_output_sha256,
            )
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        output_payload["artifact"] = {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        }
    _json_print(output_payload)
    return 0


def cmd_runtime_policy(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state_readonly(state) as conn:
        policy = derive_scheduler_runtime_policy(
            conn,
            queue,
            cpu_count=args.cpu_count,
            timeout_multiplier=args.timeout_multiplier,
            min_timeout_seconds=args.min_timeout_seconds,
            max_timeout_seconds=args.max_timeout_seconds,
        )
    payload: dict[str, object] = {"state": str(state), **policy}
    if args.policy_output is not None:
        try:
            artifact = write_json_artifact(
                args.policy_output,
                policy,
                allow_overwrite=args.expected_policy_output_sha256 is not None,
                expected_existing_sha256=args.expected_policy_output_sha256,
            )
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        payload["policy_artifact"] = {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        }
    if args.applied_queue_output is not None:
        applied = apply_scheduler_runtime_policy(
            queue,
            policy,
            apply_concurrency=not args.no_apply_concurrency,
            apply_timeouts=not args.no_apply_timeouts,
        )
        try:
            artifact = write_json_artifact(
                args.applied_queue_output,
                applied,
                allow_overwrite=args.expected_applied_queue_output_sha256 is not None,
                expected_existing_sha256=args.expected_applied_queue_output_sha256,
            )
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        payload["applied_queue_artifact"] = {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        }
    _json_print(payload)
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
    )
    if args.format == "markdown":
        if args.output is not None:
            raise ExperimentQueueError("--output is only supported for --format json")
        print(render_observation_markdown(observation), end="")
    else:
        output_payload = {"state": str(state), **observation}
        if args.output is not None:
            try:
                artifact = write_json_artifact(
                    args.output,
                    output_payload,
                    allow_overwrite=args.expected_output_sha256 is not None,
                    expected_existing_sha256=args.expected_output_sha256,
                )
            except ArtifactWriteError as exc:
                raise ExperimentQueueError(str(exc)) from exc
            output_payload["artifact"] = {
                "path": artifact.path,
                "bytes": artifact.bytes_written,
                "sha256": artifact.sha256,
            }
        _json_print(output_payload)
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(
            conn,
            queue,
            allow_cloud=args.allow_cloud,
            repo_root=REPO_ROOT,
        )
    _json_print({"state": str(state), "ready_steps": [step.to_dict() for step in ready]})
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    noncanonical_rationale = _rationale_text(
        args.noncanonical_state_rationale,
        label="--noncanonical-state-rationale",
    )
    orphaned_rationale = _rationale_text(
        args.orphaned_state_rationale,
        label="--orphaned-state-rationale",
    )
    if args.execute:
        assert_canonical_state_for_execution(
            REPO_ROOT,
            queue["queue_id"],
            state,
            allow_noncanonical_state=noncanonical_rationale is not None,
        )
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        if args.execute:
            assert_no_orphaned_steps_for_execution(
                conn,
                queue,
                allow_orphaned_state=orphaned_rationale is not None,
            )
        ready = ready_steps(
            conn,
            queue,
            allow_cloud=args.allow_cloud,
            repo_root=REPO_ROOT,
        )
        if not ready:
            _json_print({"state": str(state), "executed": False, "reason": "no_ready_steps"})
            return 0
        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=REPO_ROOT,
            execute=args.execute,
            log_root=args.log_root,
        )
    _json_print({"state": str(state), **result})
    return 0 if not result.get("executed") or result.get("succeeded") else 2


def cmd_run_worker(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    noncanonical_rationale = _rationale_text(
        args.noncanonical_state_rationale,
        label="--noncanonical-state-rationale",
    )
    orphaned_rationale = _rationale_text(
        args.orphaned_state_rationale,
        label="--orphaned-state-rationale",
    )
    if args.execute:
        assert_canonical_state_for_execution(
            REPO_ROOT,
            queue["queue_id"],
            state,
            allow_noncanonical_state=noncanonical_rationale is not None,
        )
    stop_signals: list[int] = []
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_stop(signum: int, _frame: object) -> None:
        if signum not in stop_signals:
            stop_signals.append(signum)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        with connect_state(state) as conn:
            initialize_queue_state(conn, queue)
            result = run_queue_worker(
                conn,
                queue,
                repo_root=REPO_ROOT,
                execute=args.execute,
                max_steps=args.max_steps,
                max_parallel=args.max_parallel,
                idle_sleep_seconds=args.idle_sleep_seconds,
                max_idle_cycles=args.max_idle_cycles,
                poll_interval_seconds=args.poll_interval_seconds,
                stop_policy=args.stop_policy,
                shutdown_grace_seconds=args.shutdown_grace_seconds,
                allow_cloud=args.allow_cloud,
                allow_orphaned_state=orphaned_rationale is not None,
                orphaned_state_rationale=orphaned_rationale,
                noncanonical_state_rationale=noncanonical_rationale,
                log_root=args.log_root,
                stop_requested=lambda: bool(stop_signals),
                max_experiments=args.max_experiments,
                reload_queue=None
                if args.no_reload_definition
                else lambda: load_queue_definition(args.queue),
            )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    _json_print(
        {
            "state": str(state),
            "stop_signals": [signal.Signals(signum).name for signum in stop_signals],
            **result,
        }
    )
    if args.output is not None:
        payload = {
            "state": str(state),
            "stop_signals": [signal.Signals(signum).name for signum in stop_signals],
            **result,
        }
        try:
            write_json_artifact(
                args.output,
                payload,
                allow_overwrite=args.expected_output_sha256 is not None,
                expected_existing_sha256=args.expected_output_sha256,
            )
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
    return 2 if result.get("failure_count", 0) else 0


def cmd_control(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        set_control_mode(conn, queue["queue_id"], args.mode, reason=args.reason)
        summary = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_rewind(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        rewind_step(
            conn,
            queue["queue_id"],
            args.experiment_id,
            args.step_id,
            reason=args.reason,
            queue=queue,
            cascade=not args.no_cascade,
        )
        summary = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_retire_orphans(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        retired = retire_orphaned_steps(conn, queue, reason=args.reason)
        summary = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), "retired_steps": retired, **summary})
    return 0


def cmd_reconcile_state(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    reason = _rationale_text(args.reason, label="--reason")
    assert reason is not None
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        before = queue_summary(conn, queue, repo_root=REPO_ROOT)
        blocking_before = _blocking_orphan_count(conn, queue)
        retired = retire_orphaned_steps(conn, queue, reason=reason)
        after = queue_summary(conn, queue, repo_root=REPO_ROOT)
        blocking_after = _blocking_orphan_count(conn, queue)
    payload = {
        "schema": STATE_RECONCILIATION_SCHEMA,
        "queue_id": queue["queue_id"],
        "state": str(state),
        "reason": reason,
        "blocking_orphan_count_before": blocking_before,
        "blocking_orphan_count_after": blocking_after,
        "retired_step_count": len(retired),
        "retired_steps": retired,
        "before": before,
        "after": after,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if args.output is not None:
        try:
            artifact = write_json_artifact(args.output, payload)
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        payload["artifact"] = {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        }
    _json_print(payload)
    return 0


def cmd_reconcile_satisfied(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        before = queue_summary(conn, queue, repo_root=REPO_ROOT)
        reconciled = reconcile_satisfied_queued_steps(
            conn,
            queue,
            repo_root=REPO_ROOT,
            include_failed=args.include_failed,
        )
        after = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), "before": before, "after": after, **reconciled})
    return 0


def cmd_reconcile_stale_running(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        before = queue_summary(conn, queue, repo_root=REPO_ROOT)
        reconciled = reconcile_stale_running_steps(
            conn,
            queue,
            repo_root=REPO_ROOT,
            stale_after_seconds=args.stale_after_seconds,
        )
        after = queue_summary(conn, queue, repo_root=REPO_ROOT)
    _json_print({"state": str(state), "before": before, "after": after, **reconciled})
    return 0


def _blocking_orphan_count(conn: sqlite3.Connection, queue: dict) -> int:
    return sum(
        1
        for row in orphaned_step_rows(conn, queue)
        if str(row["status"]) in BLOCKING_ORPHAN_STATUSES
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="queue definition JSON or JSON-compatible YAML")
    parser.add_argument("--state", default=None, help="SQLite state path")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("validate")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("init")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("status")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("performance", help="aggregate completed-step telemetry")
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--expected-output-sha256", default=None)
    sp.set_defaults(func=cmd_performance)

    sp = sub.add_parser(
        "runtime-policy",
        help="derive advisory scheduler runtime policy from queue telemetry",
    )
    sp.add_argument("--cpu-count", type=int, default=None)
    sp.add_argument("--timeout-multiplier", type=float, default=3.0)
    sp.add_argument("--min-timeout-seconds", type=int, default=30)
    sp.add_argument("--max-timeout-seconds", type=int, default=24 * 60 * 60)
    sp.add_argument("--policy-output", type=Path, default=None)
    sp.add_argument("--expected-policy-output-sha256", default=None)
    sp.add_argument("--applied-queue-output", type=Path, default=None)
    sp.add_argument("--expected-applied-queue-output-sha256", default=None)
    sp.add_argument("--no-apply-concurrency", action="store_true")
    sp.add_argument("--no-apply-timeouts", action="store_true")
    sp.set_defaults(func=cmd_runtime_policy)

    sp = sub.add_parser("observe", help="compact live queue telemetry and artifact view")
    sp.add_argument("--tail-lines", type=int, default=20)
    sp.add_argument("--include-orphans", action="store_true")
    sp.add_argument("--format", choices=["json", "markdown"], default="json")
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--expected-output-sha256", default=None)
    sp.set_defaults(func=cmd_observe)

    sp = sub.add_parser("next")
    sp.add_argument("--allow-cloud", action="store_true", help="include cloud resource steps")
    sp.set_defaults(func=cmd_next)

    sp = sub.add_parser("run-once")
    sp.add_argument("--execute", action="store_true", help="actually run the selected command")
    sp.add_argument("--allow-cloud", action="store_true", help="include cloud resource steps")
    sp.add_argument(
        "--noncanonical-state-rationale",
        default=None,
        help="specific audit rationale allowing execute mode with a noncanonical --state path",
    )
    sp.add_argument(
        "--orphaned-state-rationale",
        default=None,
        help="specific audit rationale allowing execute mode when SQLite retains blocking orphans",
    )
    sp.add_argument("--log-root", default=None, help="override command log root")
    sp.set_defaults(func=cmd_run_once)

    sp = sub.add_parser(
        "run-worker",
        aliases=["run-loop"],
        help="execute ready steps in a bounded worker loop",
    )
    sp.add_argument("--execute", action="store_true", help="actually run selected commands")
    sp.add_argument("--allow-cloud", action="store_true", help="include cloud resource steps")
    sp.add_argument(
        "--noncanonical-state-rationale",
        default=None,
        help="specific audit rationale allowing execute mode with a noncanonical --state path",
    )
    sp.add_argument(
        "--orphaned-state-rationale",
        default=None,
        help="specific audit rationale allowing execute mode when SQLite retains blocking orphans",
    )
    sp.add_argument("--log-root", default=None, help="override command log root")
    sp.add_argument("--max-steps", type=int, default=1, help="maximum steps to start")
    sp.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help=(
            "maximum distinct experiments to start in this worker run; useful for "
            "bounded multi-candidate fan-out"
        ),
    )
    sp.add_argument(
        "--max-parallel",
        type=int,
        default=0,
        help=(
            "maximum subprocesses to keep running concurrently; 0 auto-sums "
            "queue controls.max_concurrency for allowed resource kinds"
        ),
    )
    sp.add_argument(
        "--no-reload-definition",
        action="store_true",
        help="pin the queue definition loaded at startup instead of rereading between steps",
    )
    sp.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=5.0,
        help="sleep between idle polls before the idle limit is reached",
    )
    sp.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.25,
        help="sleep between active subprocess polls",
    )
    sp.add_argument(
        "--max-idle-cycles",
        type=int,
        default=1,
        help="maximum no-ready-step polls before stopping",
    )
    sp.add_argument(
        "--stop-policy",
        choices=["drain", "terminate"],
        default="drain",
        help="on SIGINT/SIGTERM, drain running children or terminate them",
    )
    sp.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=5.0,
        help="grace period before killing children after terminate stop-policy",
    )
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--expected-output-sha256", default=None)
    sp.set_defaults(func=cmd_run_worker)

    sp = sub.add_parser("control")
    sp.add_argument("mode", choices=["running", "paused", "frozen"])
    sp.add_argument("--reason", default="")
    sp.set_defaults(func=cmd_control)

    sp = sub.add_parser("rewind")
    sp.add_argument("experiment_id")
    sp.add_argument("step_id")
    sp.add_argument("--reason", default="")
    sp.add_argument(
        "--no-cascade",
        action="store_true",
        help="rewind only the target step; default also rewinds dependent steps",
    )
    sp.set_defaults(func=cmd_rewind)

    sp = sub.add_parser("retire-orphans")
    sp.add_argument("--reason", required=True)
    sp.set_defaults(func=cmd_retire_orphans)

    sp = sub.add_parser(
        "reconcile-state",
        help="retire stale blocking orphans and optionally write an audit artifact",
    )
    sp.add_argument("--reason", required=True)
    sp.add_argument("--output", default=None, help="append-only JSON audit artifact path")
    sp.set_defaults(func=cmd_reconcile_state)

    sp = sub.add_parser(
        "reconcile-satisfied",
        help="mark queued steps succeeded when their postconditions already pass",
    )
    sp.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "also reconcile failed steps whose postconditions already pass; "
            "for terminal negative/refusal artifacts written before a nonzero exit"
        ),
    )
    sp.set_defaults(func=cmd_reconcile_satisfied)

    sp = sub.add_parser(
        "reconcile-stale-running",
        help="fail or recover local running rows whose recorded process is gone",
    )
    sp.add_argument("--stale-after-seconds", type=float, default=300.0)
    sp.set_defaults(func=cmd_reconcile_stale_running)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except ExperimentQueueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
