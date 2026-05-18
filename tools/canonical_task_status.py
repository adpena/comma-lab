#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator CLI for `.omx/state/canonical_task_status.jsonl`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)


def _row_table(rows) -> str:
    if not rows:
        return "(no rows)"
    widths = {
        "status": max(6, *(len(row.status) for row in rows)),
        "owner": max(5, *(len(row.owner) for row in rows)),
        "task": max(7, *(min(96, len(row.task_id)) for row in rows)),
    }
    out = [f"{'STATUS':{widths['status']}}  {'OWNER':{widths['owner']}}  TASK"]
    for row in rows:
        task = row.task_id if len(row.task_id) <= 96 else row.task_id[:93] + "..."
        out.append(f"{row.status:{widths['status']}}  {row.owner:{widths['owner']}}  {task}")
    return "\n".join(out)


def _normalize_directive_arg(value: str) -> str:
    path = Path(value)
    if path.parent == Path(".") and not value.startswith(".omx/"):
        return f".omx/research/{path.name}"
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list-pending", action="store_true")
    group.add_argument("--list-blocked", action="store_true")
    group.add_argument("--list-by-owner")
    group.add_argument("--task-history")
    group.add_argument("--directive-summary")
    group.add_argument("--json", action="store_true", help="Dump all latest rows as JSON.")
    group.add_argument("--validate", action="store_true")
    sub = parser.add_subparsers(dest="command")
    upd = sub.add_parser("update", help="Append an operator status update.")
    upd.add_argument("--task-id", required=True)
    upd.add_argument("--status", required=True)
    upd.add_argument("--actor", required=True)
    upd.add_argument("--session-id", default="operator_cli")
    upd.add_argument("--notes", default="")
    upd.add_argument("--test-status", default="pending")
    upd.add_argument("--commit-sha", action="append", default=[])
    upd.add_argument("--blocker", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.canonical_task_status import (
        latest_statuses,
        query_task_history,
        query_tasks_by_directive,
        query_tasks_by_status,
        update_status,
        validate_ledger,
    )

    if args.command == "update":
        row = update_status(
            args.task_id,
            args.status,
            actor=args.actor,
            session_id=args.session_id,
            notes=args.notes,
            commit_shas=tuple(args.commit_sha),
            test_status=args.test_status,
            blockers=tuple(args.blocker),
            repo_root=REPO,
        )
        print(json.dumps(row.to_json_obj(), indent=2, sort_keys=True))
        return 0
    if args.list_pending:
        print(_row_table(query_tasks_by_status("pending", repo_root=REPO)))
        return 0
    if args.list_blocked:
        print(_row_table(query_tasks_by_status("blocked", repo_root=REPO)))
        return 0
    if args.list_by_owner:
        rows = [row for row in latest_statuses(REPO).values() if row.owner == args.list_by_owner]
        print(_row_table(sorted(rows, key=lambda row: (row.status, row.task_id))))
        return 0
    if args.task_history:
        print(json.dumps([row.to_json_obj() for row in query_task_history(args.task_history, repo_root=REPO)], indent=2, sort_keys=True))
        return 0
    if args.directive_summary:
        print(_row_table(query_tasks_by_directive(_normalize_directive_arg(args.directive_summary), repo_root=REPO)))
        return 0
    if args.json:
        print(json.dumps([row.to_json_obj() for row in latest_statuses(REPO).values()], indent=2, sort_keys=True))
        return 0
    if args.validate:
        print(json.dumps({"rows": validate_ledger(REPO), "status": "valid"}, sort_keys=True))
        return 0
    raise SystemExit("select an action; try --list-pending or --validate")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
