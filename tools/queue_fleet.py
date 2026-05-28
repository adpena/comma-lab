#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Discover, observe, and bounded-supervise experiment_queue.v1 fleets."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.queue_fleet import (  # noqa: E402
    FALSE_AUTHORITY,
    queue_fleet_status,
    repo_rel,
    select_supervision_rows,
)
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402

FLEET_SUPERVISOR_SCHEMA = "experiment_queue_fleet_supervisor_run.v1"
FLEET_LOCK_SCHEMA = "experiment_queue_fleet_supervisor_lock.v1"
FLEET_INIT_MISSING_SCHEMA = "experiment_queue_fleet_init_missing_run.v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")


def _write_output(payload: dict[str, Any], output: Path | None, expected_sha256: str | None = None) -> None:
    if output is None:
        return
    try:
        artifact = write_json_artifact(
            output,
            payload,
            allow_overwrite=expected_sha256 is not None,
            expected_existing_sha256=expected_sha256,
        )
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    payload["artifact"] = {
        "path": artifact.path,
        "bytes": artifact.bytes_written,
        "sha256": artifact.sha256,
    }


class FleetLock:
    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = path
        self.payload = dict(payload)
        self.handle: Any | None = None

    def __enter__(self) -> FleetLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            existing = handle.read().strip()
            handle.close()
            raise ExperimentQueueError(
                f"queue fleet supervisor lock already held: {self.path}; existing={existing[:500]}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"schema": FLEET_LOCK_SCHEMA, "started_at_utc": _utc_now(), **self.payload}, sort_keys=True))
        handle.write("\n")
        handle.flush()
        self.handle = handle
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self.handle is None:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _run(command: Sequence[str]) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    payload: dict[str, Any] | None = None
    if proc.stdout.strip():
        try:
            loaded = json.loads(proc.stdout)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            payload = loaded
    return {
        "command": list(command),
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "json_payload": payload,
        "failed": bool(proc.returncode),
        **FALSE_AUTHORITY,
    }


def _fleet_status_from_args(args: argparse.Namespace) -> dict[str, Any]:
    roots = args.root or None
    return queue_fleet_status(
        REPO_ROOT,
        roots,
        state_root=args.state_root,
        max_depth=args.max_depth,
        limit=args.scan_limit,
        row_limit=args.row_limit,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
        supervisor_output_root=args.supervisor_output_root,
    )


def _format_status(payload: Mapping[str, Any]) -> str:
    lines = [
        (
            "queue fleet: "
            f"queues={payload.get('queue_count')} "
            f"actionable={payload.get('actionable_count')} "
            f"ready={payload.get('ready_to_supervise_count')} "
            f"needs_recovery={payload.get('needs_recovery_count')} "
            f"invalid={payload.get('invalid_queue_count')} "
            f"non_exec_artifacts={payload.get('non_executable_artifact_count')}"
        ),
        f"status_counts: {payload.get('status_counts')}",
    ]
    samples = payload.get("status_samples")
    if isinstance(samples, Mapping):
        lines.append(f"status_samples: {samples}")
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        for row in rows[:12]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"- {row.get('status')} {row.get('queue_id') or '-'} "
                f"counts={row.get('status_counts') or {}} path={row.get('queue_path')}"
            )
            blockers = row.get("blockers")
            if blockers:
                lines.append(f"  blockers={blockers}")
    commands = payload.get("next_supervise_commands")
    if isinstance(commands, Sequence) and commands:
        lines.append("next supervise:")
        for command in commands[:3]:
            if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
                lines.append("  " + " ".join(str(part) for part in command))
    init_commands = payload.get("next_init_commands")
    if isinstance(init_commands, Sequence) and init_commands:
        lines.append("next init:")
        for command in init_commands[:3]:
            if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
                lines.append("  " + " ".join(str(part) for part in command))
    recovery_commands = payload.get("next_recovery_commands")
    if isinstance(recovery_commands, Sequence) and recovery_commands:
        lines.append("next recovery/status:")
        for command in recovery_commands[:3]:
            if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
                lines.append("  " + " ".join(str(part) for part in command))
    lines.append("authority: local telemetry/supervision only; never score or promotion authority.")
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    payload = _fleet_status_from_args(args)
    _write_output(payload, args.output, args.expected_output_sha256)
    if args.format == "text":
        print(_format_status(payload))
    else:
        _json_print(payload)
    blocked = int(payload.get("needs_recovery_count") or 0) > 0
    return 3 if args.strict and blocked else 0


def _supervisor_command_from_row(row: Mapping[str, Any], args: argparse.Namespace, queue_output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "tools/queue_supervisor.py",
        "--queue",
        str(row["queue_path"]),
        "--state",
        str(row["state"]),
        "--output-dir",
        repo_rel(queue_output_dir, REPO_ROOT),
        "--max-ticks",
        str(args.max_ticks_per_queue),
        "--max-steps-per-tick",
        str(args.max_steps_per_tick),
        "--max-parallel",
        args.max_parallel,
        "--max-parallel-cap",
        str(args.max_parallel_cap),
        "--strict",
    ]
    if args.execute:
        command.append("--execute")
    if args.allow_cloud:
        command.append("--allow-cloud")
    if args.no_recover:
        command.append("--no-recover")
    if args.include_orphans:
        command.append("--include-orphans")
    if args.log_root:
        command.extend(["--log-root", args.log_root])
    return command


def cmd_supervise(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock_path) if args.lock_path else output_root / "fleet.lock"
    started_at = _utc_now()
    with FleetLock(
        lock_path,
        {
            "pid": str(__import__("os").getpid()),
            "output_dir": repo_rel(output_root, REPO_ROOT),
            "execute": args.execute,
        },
    ):
        initial = _fleet_status_from_args(args)
        selected = select_supervision_rows(
            initial.get("rows", []),
            include_recovery=args.include_recovery,
            max_queues=args.max_queues,
        )
        runs: list[dict[str, Any]] = []
        events_path = output_root / "fleet_events.jsonl"
        for index, row in enumerate(selected):
            queue_id = str(row.get("queue_id") or f"queue_{index}")
            queue_dir = output_root / f"{index:03d}_{queue_id}"
            command = _supervisor_command_from_row(row, args, queue_dir)
            result = {
                "schema": "experiment_queue_fleet_supervisor_child.v1",
                "index": index,
                "queue_id": queue_id,
                "queue_path": row.get("queue_path"),
                "status_before": row.get("status"),
                "started_at_utc": _utc_now(),
                "supervisor_result": _run(command),
                "finished_at_utc": _utc_now(),
                **FALSE_AUTHORITY,
            }
            runs.append(result)
            _append_jsonl(events_path, result)
            if result["supervisor_result"].get("failed") and args.stop_on_child_failure:
                break
        final = _fleet_status_from_args(args)
        payload = {
            "schema": FLEET_SUPERVISOR_SCHEMA,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "execute": args.execute,
            "output_dir": repo_rel(output_root, REPO_ROOT),
            "lock_path": repo_rel(lock_path, REPO_ROOT),
            "events_jsonl": repo_rel(events_path, REPO_ROOT),
            "selected_count": len(selected),
            "completed_child_count": len(runs),
            "failed_child_count": sum(1 for row in runs if row["supervisor_result"].get("failed")),
            "initial_status": initial,
            "final_status": final,
            "child_runs": runs,
            "allowed_use": "queue_fleet_bounded_local_supervision_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
            **FALSE_AUTHORITY,
        }
        result_path = output_root / "fleet_supervisor_result.json"
        try:
            expected = sha256_file(result_path) if result_path.exists() else None
            artifact = write_json_artifact(
                result_path,
                payload,
                allow_overwrite=expected is not None,
                expected_existing_sha256=expected,
            )
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        payload["artifact"] = {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        }
    _json_print(payload)
    return 4 if args.strict and payload["failed_child_count"] else 0


def _init_command_from_row(row: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        str(row["queue_path"]),
        "--state",
        str(row["state"]),
        "init",
    ]


def cmd_init_missing(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock_path) if args.lock_path else output_root / "fleet.lock"
    started_at = _utc_now()
    with FleetLock(
        lock_path,
        {
            "pid": str(__import__("os").getpid()),
            "output_dir": repo_rel(output_root, REPO_ROOT),
            "execute": args.execute,
            "operation": "init_missing",
        },
    ):
        initial = _fleet_status_from_args(args)
        rows = initial.get("rows", [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            rows = []
        selected = [row for row in rows if isinstance(row, Mapping) and row.get("status") == "NEEDS_INIT"]
        selected = selected[: args.max_queues]
        events_path = output_root / "fleet_init_events.jsonl"
        runs: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            command = _init_command_from_row(row)
            result: dict[str, Any] = {
                "schema": "experiment_queue_fleet_init_missing_child.v1",
                "index": index,
                "queue_id": row.get("queue_id"),
                "queue_path": row.get("queue_path"),
                "state": row.get("state"),
                "status_before": row.get("status"),
                "command": command,
                "execute": args.execute,
                **FALSE_AUTHORITY,
            }
            if args.execute:
                result["started_at_utc"] = _utc_now()
                result["init_result"] = _run(command)
                result["finished_at_utc"] = _utc_now()
            runs.append(result)
            _append_jsonl(events_path, result)
            init_result = result.get("init_result")
            if isinstance(init_result, Mapping) and init_result.get("failed") and args.stop_on_child_failure:
                break
        final = _fleet_status_from_args(args)
        failed_count = sum(
            1
            for row in runs
            if isinstance(row.get("init_result"), Mapping) and row["init_result"].get("failed")
        )
        payload = {
            "schema": FLEET_INIT_MISSING_SCHEMA,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "execute": args.execute,
            "output_dir": repo_rel(output_root, REPO_ROOT),
            "lock_path": repo_rel(lock_path, REPO_ROOT),
            "events_jsonl": repo_rel(events_path, REPO_ROOT),
            "selected_count": len(selected),
            "completed_child_count": len(runs),
            "failed_child_count": failed_count,
            "initial_status": initial,
            "final_status": final,
            "child_runs": runs,
            "allowed_use": "queue_fleet_missing_state_initialization_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
            **FALSE_AUTHORITY,
        }
        result_path = output_root / "fleet_init_missing_result.json"
        try:
            expected = sha256_file(result_path) if result_path.exists() else None
            artifact = write_json_artifact(
                result_path,
                payload,
                allow_overwrite=expected is not None,
                expected_existing_sha256=expected,
            )
        except ArtifactWriteError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        payload["artifact"] = {
            "path": artifact.path,
            "bytes": artifact.bytes_written,
            "sha256": artifact.sha256,
        }
    _json_print(payload)
    return 4 if args.strict and payload["failed_child_count"] else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", help="scan root; repeatable")
    parser.add_argument("--state-root", default=None, help="optional queue state directory override")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=80)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--tail-lines", type=int, default=0)
    parser.add_argument("--include-orphans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--supervisor-output-root",
        default=".omx/research/queue_fleet_supervisor",
        help="root used when suggesting per-queue supervisor output dirs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.add_argument("--format", choices=("json", "text"), default="json")
    status.add_argument("--output", type=Path)
    status.add_argument("--expected-output-sha256")
    status.add_argument("--strict", action="store_true")
    status.set_defaults(func=cmd_status)

    supervise = sub.add_parser("supervise")
    supervise.add_argument("--output-dir", required=True)
    supervise.add_argument("--execute", action="store_true")
    supervise.add_argument("--include-recovery", action="store_true")
    supervise.add_argument("--max-queues", type=int, default=4)
    supervise.add_argument("--max-ticks-per-queue", type=int, default=16)
    supervise.add_argument("--max-steps-per-tick", type=int, default=16)
    supervise.add_argument("--max-parallel", default="auto")
    supervise.add_argument("--max-parallel-cap", type=int, default=8)
    supervise.add_argument("--allow-cloud", action="store_true")
    supervise.add_argument("--no-recover", action="store_true")
    supervise.add_argument("--stop-on-child-failure", action=argparse.BooleanOptionalAction, default=True)
    supervise.add_argument("--log-root", default=None)
    supervise.add_argument("--lock-path", default=None)
    supervise.add_argument("--strict", action="store_true")
    supervise.set_defaults(func=cmd_supervise)

    init_missing = sub.add_parser("init-missing")
    init_missing.add_argument("--output-dir", required=True)
    init_missing.add_argument("--execute", action="store_true")
    init_missing.add_argument("--max-queues", type=int, default=8)
    init_missing.add_argument("--stop-on-child-failure", action=argparse.BooleanOptionalAction, default=True)
    init_missing.add_argument("--lock-path", default=None)
    init_missing.add_argument("--strict", action="store_true")
    init_missing.set_defaults(func=cmd_init_missing)
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
