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
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    ExperimentQueueError,
    assert_canonical_state_for_execution,
    assert_no_orphaned_steps_for_execution,
    connect_state,
    default_state_path,
    initialize_queue_state,
    load_queue_definition,
    queue_summary,
    ready_steps,
    retire_orphaned_steps,
    rewind_step,
    run_queue_worker,
    run_ready_step,
    set_control_mode,
)

_PLACEHOLDER_RATIONALES = {"", "n/a", "na", "none", "test", "true", "yes", "because"}


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
    _json_print(
        {
            "valid": True,
            "queue_id": queue["queue_id"],
            "experiment_count": len(queue["experiments"]),
            "step_count": sum(len(exp["steps"]) for exp in queue["experiments"]),
        }
    )
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        summary = queue_summary(conn, queue)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        summary = queue_summary(conn, queue)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, allow_cloud=args.allow_cloud)
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
        ready = ready_steps(conn, queue, allow_cloud=args.allow_cloud)
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
                idle_sleep_seconds=args.idle_sleep_seconds,
                max_idle_cycles=args.max_idle_cycles,
                allow_cloud=args.allow_cloud,
                allow_orphaned_state=orphaned_rationale is not None,
                orphaned_state_rationale=orphaned_rationale,
                noncanonical_state_rationale=noncanonical_rationale,
                log_root=args.log_root,
                stop_requested=lambda: bool(stop_signals),
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
    return 2 if result.get("failure_count", 0) else 0


def cmd_control(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        set_control_mode(conn, queue["queue_id"], args.mode, reason=args.reason)
        summary = queue_summary(conn, queue)
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
        summary = queue_summary(conn, queue)
    _json_print({"state": str(state), **summary})
    return 0


def cmd_retire_orphans(args: argparse.Namespace) -> int:
    queue, state = _load(args)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        retired = retire_orphaned_steps(conn, queue, reason=args.reason)
        summary = queue_summary(conn, queue)
    _json_print({"state": str(state), "retired_steps": retired, **summary})
    return 0


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
        "--max-idle-cycles",
        type=int,
        default=1,
        help="maximum no-ready-step polls before stopping",
    )
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
