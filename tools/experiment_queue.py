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
    connect_state,
    default_state_path,
    initialize_queue_state,
    load_queue_definition,
    queue_summary,
    ready_steps,
    rewind_step,
    run_ready_step,
    set_control_mode,
)


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _load(args: argparse.Namespace) -> tuple[dict, Path]:
    queue = load_queue_definition(args.queue)
    state = Path(args.state) if args.state else default_state_path(REPO_ROOT, queue["queue_id"])
    return queue, state


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
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
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
    sp.add_argument("--log-root", default=None, help="override command log root")
    sp.set_defaults(func=cmd_run_once)

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
