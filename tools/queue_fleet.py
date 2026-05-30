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
FLEET_NATIVE_CONSUMER_SCHEMA = "experiment_queue_fleet_native_consumer_run.v1"
FLEET_LOCAL_DRAIN_SCHEMA = "experiment_queue_fleet_local_drain_run.v1"


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


def _fleet_status_from_args(args: argparse.Namespace, *, full_rows: bool = False) -> dict[str, Any]:
    roots = args.root or None
    return queue_fleet_status(
        REPO_ROOT,
        roots,
        state_root=args.state_root,
        max_depth=args.max_depth,
        limit=args.scan_limit,
        row_limit=None if full_rows else args.row_limit,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
        supervisor_output_root=args.supervisor_output_root,
    )


def _rows_from_status(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _compact_status_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    command_keys = (
        "next_supervise_commands",
        "next_recovery_commands",
        "next_resume_commands",
        "next_init_commands",
        "next_native_consumer_commands",
    )
    out: dict[str, Any] = {
        "schema": payload.get("schema"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "queue_count": payload.get("queue_count"),
        "candidate_path_count": payload.get("candidate_path_count"),
        "row_count": payload.get("row_count"),
        "actionable_count": payload.get("actionable_count"),
        "ready_to_supervise_count": payload.get("ready_to_supervise_count"),
        "paused_with_queued_work_count": payload.get("paused_with_queued_work_count"),
        "paused_exact_dispatch_gate_count": payload.get("paused_exact_dispatch_gate_count"),
        "needs_recovery_count": payload.get("needs_recovery_count"),
        "invalid_queue_count": payload.get("invalid_queue_count"),
        "non_executable_artifact_count": payload.get("non_executable_artifact_count"),
        "native_consumer_artifact_count": payload.get("native_consumer_artifact_count"),
        "known_native_consumer_artifact_count": payload.get("known_native_consumer_artifact_count"),
        "status_counts": payload.get("status_counts") if isinstance(payload.get("status_counts"), Mapping) else {},
        **FALSE_AUTHORITY,
    }
    for key in command_keys:
        commands = payload.get(key)
        out[f"{key}_count"] = (
            len(commands)
            if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray))
            else 0
        )
    return out


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": row.get("status"),
        "queue_id": row.get("queue_id"),
        "queue_path": row.get("queue_path"),
    }
    native_consumer = row.get("native_consumer")
    if isinstance(native_consumer, Mapping):
        out["consumer_kind"] = native_consumer.get("consumer_kind")
    for key in (
        "artifact_schema",
        "state",
        "status_counts",
        "blockers",
        "recommended_action",
        "native_consumer_command",
    ):
        value = row.get(key)
        if value not in (None, {}, [], ""):
            out[key] = value
    return out


def _compact_json_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "schema",
        "started_at_utc",
        "finished_at_utc",
        "execute",
        "output_dir",
        "artifact",
        "queue_id",
        "queue_schema",
        "source_work_queue_schema",
        "selected_count",
        "completed_child_count",
        "failed_child_count",
        "final_reason",
        "final_status_counts",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    ):
        if key in payload:
            out[key] = payload[key]
    initial_status = payload.get("initial_status")
    if isinstance(initial_status, Mapping):
        out["initial_status"] = _compact_status_snapshot(initial_status)
    final_status = payload.get("final_status")
    if isinstance(final_status, Mapping):
        out["final_status"] = _compact_status_snapshot(final_status)
    child_runs = payload.get("child_runs")
    if isinstance(child_runs, Sequence) and not isinstance(child_runs, (str, bytes, bytearray)):
        out["child_run_count"] = len(child_runs)
    return out


def _compact_run_result(result: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(result)
    json_payload = out.get("json_payload")
    if isinstance(json_payload, Mapping):
        out["json_payload"] = _compact_json_payload(json_payload)
    return out


def _select_init_rows(rows: Sequence[Mapping[str, Any]], *, max_queues: int) -> list[Mapping[str, Any]]:
    if max_queues < 0:
        raise ExperimentQueueError("max-init-queues must be >= 0")
    if max_queues == 0:
        return []
    return [row for row in rows if row.get("status") == "NEEDS_INIT"][:max_queues]


def _native_consumer_output_exists(row: Mapping[str, Any]) -> bool:
    native_consumer = row.get("native_consumer")
    if not isinstance(native_consumer, Mapping):
        return False
    for key in ("output_queue_path", "output_report_path", "closed_source_queue_path"):
        value = native_consumer.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            return True
    return False


def _format_status(payload: Mapping[str, Any]) -> str:
    lines = [
        (
            "queue fleet: "
            f"queues={payload.get('queue_count')} "
            f"actionable={payload.get('actionable_count')} "
            f"ready={payload.get('ready_to_supervise_count')} "
            f"needs_recovery={payload.get('needs_recovery_count')} "
            f"invalid={payload.get('invalid_queue_count')} "
            f"non_exec_artifacts={payload.get('non_executable_artifact_count')} "
            f"native_consumers={payload.get('known_native_consumer_artifact_count')}"
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
    native_commands = payload.get("next_native_consumer_commands")
    if isinstance(native_commands, Sequence) and native_commands:
        lines.append("next native consumers:")
        for command in native_commands[:3]:
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


def _select_native_consumer_rows(
    rows: Sequence[Any],
    *,
    artifact_schemas: Sequence[str],
    consumer_kinds: Sequence[str],
    max_artifacts: int,
) -> list[Mapping[str, Any]]:
    schema_filter = {str(item) for item in artifact_schemas if str(item).strip()}
    kind_filter = {str(item) for item in consumer_kinds if str(item).strip()}
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        command = row.get("native_consumer_command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes, bytearray)):
            continue
        native_consumer = row.get("native_consumer")
        if not isinstance(native_consumer, Mapping):
            continue
        if schema_filter and str(row.get("artifact_schema") or "") not in schema_filter:
            continue
        if kind_filter and str(native_consumer.get("consumer_kind") or "") not in kind_filter:
            continue
        selected.append(row)
        if len(selected) >= max_artifacts:
            break
    return selected


def cmd_consume_native(args: argparse.Namespace) -> int:
    if args.max_artifacts < 1:
        raise ExperimentQueueError("max-artifacts must be >= 1")
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
            "operation": "consume_native",
        },
    ):
        initial = _fleet_status_from_args(args, full_rows=True)
        rows = initial.get("rows", [])
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            rows = []
        selected = _select_native_consumer_rows(
            rows,
            artifact_schemas=args.artifact_schema or [],
            consumer_kinds=args.consumer_kind or [],
            max_artifacts=args.max_artifacts,
        )
        events_path = output_root / "fleet_native_consumer_events.jsonl"
        runs: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            native_consumer = row.get("native_consumer")
            command = [str(part) for part in row.get("native_consumer_command") or []]
            result: dict[str, Any] = {
                "schema": "experiment_queue_fleet_native_consumer_child.v1",
                "index": index,
                "queue_path": row.get("queue_path"),
                "artifact_schema": row.get("artifact_schema"),
                "consumer_kind": (
                    native_consumer.get("consumer_kind")
                    if isinstance(native_consumer, Mapping)
                    else None
                ),
                "recommended_action": row.get("recommended_action"),
                "command": command,
                "execute": args.execute,
                **FALSE_AUTHORITY,
            }
            if args.execute:
                result["started_at_utc"] = _utc_now()
                result["native_consumer_result"] = _run(command)
                result["finished_at_utc"] = _utc_now()
            runs.append(result)
            _append_jsonl(events_path, result)
            child_result = result.get("native_consumer_result")
            if isinstance(child_result, Mapping) and child_result.get("failed") and args.stop_on_child_failure:
                break
        final = _fleet_status_from_args(args, full_rows=True)
        failed_count = sum(
            1
            for row in runs
            if isinstance(row.get("native_consumer_result"), Mapping)
            and row["native_consumer_result"].get("failed")
        )
        payload = {
            "schema": FLEET_NATIVE_CONSUMER_SCHEMA,
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
            "allowed_use": "queue_fleet_bounded_native_consumer_execution_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
            **FALSE_AUTHORITY,
        }
        result_path = output_root / "fleet_native_consumer_result.json"
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


def _run_drain_phase_child(
    *,
    events_path: Path,
    cycle_index: int,
    phase: str,
    index: int,
    row: Mapping[str, Any],
    command: Sequence[str],
    execute: bool,
) -> dict[str, Any]:
    child: dict[str, Any] = {
        "schema": "experiment_queue_fleet_local_drain_child.v1",
        "cycle_index": cycle_index,
        "phase": phase,
        "index": index,
        "row": _compact_row(row),
        "command": [str(part) for part in command],
        "execute": execute,
        "failed": False,
        **FALSE_AUTHORITY,
    }
    if execute:
        child["started_at_utc"] = _utc_now()
        result = _compact_run_result(_run(command))
        child["command_result"] = result
        child["finished_at_utc"] = _utc_now()
        child["failed"] = bool(result.get("failed"))
    _append_jsonl(events_path, child)
    return child


def _drain_native_phase(
    *,
    args: argparse.Namespace,
    cycle_index: int,
    rows: Sequence[Mapping[str, Any]],
    events_path: Path,
) -> dict[str, Any]:
    selected = (
        _select_native_consumer_rows(
            rows,
            artifact_schemas=args.artifact_schema or [],
            consumer_kinds=args.consumer_kind or [],
            max_artifacts=max(1, len(rows))
            if args.skip_existing_native_outputs
            else args.max_native_artifacts,
        )
        if args.max_native_artifacts
        else []
    )
    if args.skip_existing_native_outputs:
        selected = [row for row in selected if not _native_consumer_output_exists(row)]
    selected = selected[: args.max_native_artifacts]
    children: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        command = row.get("native_consumer_command") or []
        child = _run_drain_phase_child(
            events_path=events_path,
            cycle_index=cycle_index,
            phase="native_consumer",
            index=index,
            row=row,
            command=[str(part) for part in command],
            execute=args.execute,
        )
        children.append(child)
        if child["failed"] and args.stop_on_child_failure:
            break
    return {
        "schema": "experiment_queue_fleet_local_drain_phase.v1",
        "cycle_index": cycle_index,
        "phase": "native_consumer",
        "selected_count": len(selected),
        "completed_child_count": len(children),
        "failed_child_count": sum(1 for child in children if child.get("failed")),
        "children": children,
        **FALSE_AUTHORITY,
    }


def _drain_init_phase(
    *,
    args: argparse.Namespace,
    cycle_index: int,
    rows: Sequence[Mapping[str, Any]],
    events_path: Path,
) -> dict[str, Any]:
    selected = _select_init_rows(rows, max_queues=args.max_init_queues)
    children: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        child = _run_drain_phase_child(
            events_path=events_path,
            cycle_index=cycle_index,
            phase="init_missing",
            index=index,
            row=row,
            command=_init_command_from_row(row),
            execute=args.execute,
        )
        children.append(child)
        if child["failed"] and args.stop_on_child_failure:
            break
    return {
        "schema": "experiment_queue_fleet_local_drain_phase.v1",
        "cycle_index": cycle_index,
        "phase": "init_missing",
        "selected_count": len(selected),
        "completed_child_count": len(children),
        "failed_child_count": sum(1 for child in children if child.get("failed")),
        "children": children,
        **FALSE_AUTHORITY,
    }


def _drain_supervise_phase(
    *,
    args: argparse.Namespace,
    cycle_index: int,
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    events_path: Path,
) -> dict[str, Any]:
    selected = (
        select_supervision_rows(
            rows,
            include_recovery=args.include_recovery,
            max_queues=args.max_supervise_queues,
        )
        if args.max_supervise_queues
        else []
    )
    children: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        queue_id = str(row.get("queue_id") or f"queue_{index}")
        queue_dir = output_root / f"cycle_{cycle_index:03d}" / f"{index:03d}_{queue_id}"
        child = _run_drain_phase_child(
            events_path=events_path,
            cycle_index=cycle_index,
            phase="supervise",
            index=index,
            row=row,
            command=_supervisor_command_from_row(row, args, queue_dir),
            execute=args.execute,
        )
        children.append(child)
        if child["failed"] and args.stop_on_child_failure:
            break
    return {
        "schema": "experiment_queue_fleet_local_drain_phase.v1",
        "cycle_index": cycle_index,
        "phase": "supervise",
        "selected_count": len(selected),
        "completed_child_count": len(children),
        "failed_child_count": sum(1 for child in children if child.get("failed")),
        "children": children,
        **FALSE_AUTHORITY,
    }


def cmd_drain_local(args: argparse.Namespace) -> int:
    if args.max_cycles < 1:
        raise ExperimentQueueError("max-cycles must be >= 1")
    for attr in ("max_native_artifacts", "max_init_queues", "max_supervise_queues"):
        if int(getattr(args, attr)) < 0:
            raise ExperimentQueueError(f"{attr.replace('_', '-')} must be >= 0")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock_path) if args.lock_path else output_root / "fleet.lock"
    events_path = output_root / "fleet_local_drain_events.jsonl"
    started_at = _utc_now()
    with FleetLock(
        lock_path,
        {
            "pid": str(__import__("os").getpid()),
            "output_dir": repo_rel(output_root, REPO_ROOT),
            "execute": args.execute,
            "operation": "drain_local",
        },
    ):
        initial = _fleet_status_from_args(args, full_rows=True)
        current = initial
        cycles: list[dict[str, Any]] = []
        halted_reason = ""
        for cycle_index in range(args.max_cycles):
            cycle_started_at = _utc_now()
            status_before = current
            phases: list[dict[str, Any]] = []

            native_phase = _drain_native_phase(
                args=args,
                cycle_index=cycle_index,
                rows=_rows_from_status(status_before),
                events_path=events_path,
            )
            phases.append(native_phase)
            if args.execute and native_phase["completed_child_count"]:
                current = _fleet_status_from_args(args, full_rows=True)

            native_failed = bool(native_phase["failed_child_count"])
            if native_failed and args.stop_on_child_failure:
                halted_reason = "native_consumer_child_failed"
            else:
                init_phase = _drain_init_phase(
                    args=args,
                    cycle_index=cycle_index,
                    rows=_rows_from_status(current),
                    events_path=events_path,
                )
                phases.append(init_phase)
                if args.execute and init_phase["completed_child_count"]:
                    current = _fleet_status_from_args(args, full_rows=True)

                init_failed = bool(init_phase["failed_child_count"])
                if init_failed and args.stop_on_child_failure:
                    halted_reason = "init_missing_child_failed"
                else:
                    supervise_phase = _drain_supervise_phase(
                        args=args,
                        cycle_index=cycle_index,
                        rows=_rows_from_status(current),
                        output_root=output_root,
                        events_path=events_path,
                    )
                    phases.append(supervise_phase)
                    if args.execute and supervise_phase["completed_child_count"]:
                        current = _fleet_status_from_args(args, full_rows=True)
                    if supervise_phase["failed_child_count"] and args.stop_on_child_failure:
                        halted_reason = "supervise_child_failed"

            selected_count = sum(int(phase["selected_count"]) for phase in phases)
            failed_count = sum(int(phase["failed_child_count"]) for phase in phases)
            cycle = {
                "schema": "experiment_queue_fleet_local_drain_cycle.v1",
                "cycle_index": cycle_index,
                "started_at_utc": cycle_started_at,
                "finished_at_utc": _utc_now(),
                "execute": args.execute,
                "selected_count": selected_count,
                "completed_child_count": sum(int(phase["completed_child_count"]) for phase in phases),
                "failed_child_count": failed_count,
                "status_before": _compact_status_snapshot(status_before),
                "status_after": _compact_status_snapshot(current),
                "phases": phases,
                **FALSE_AUTHORITY,
            }
            cycles.append(cycle)
            _append_jsonl(events_path, cycle)
            if halted_reason:
                break
            if selected_count == 0:
                halted_reason = "no_selectable_local_work"
                break
            if not args.execute:
                halted_reason = "plan_only"
                break
        final = _fleet_status_from_args(args, full_rows=True)
        failed_child_count = sum(int(cycle["failed_child_count"]) for cycle in cycles)
        payload = {
            "schema": FLEET_LOCAL_DRAIN_SCHEMA,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "execute": args.execute,
            "output_dir": repo_rel(output_root, REPO_ROOT),
            "lock_path": repo_rel(lock_path, REPO_ROOT),
            "events_jsonl": repo_rel(events_path, REPO_ROOT),
            "max_cycles": args.max_cycles,
            "completed_cycle_count": len(cycles),
            "failed_child_count": failed_child_count,
            "halted_reason": halted_reason or "max_cycles_exhausted",
            "initial_status": _compact_status_snapshot(initial),
            "final_status": _compact_status_snapshot(final),
            "cycles": cycles,
            "allowed_use": "queue_fleet_bounded_local_native_init_supervision_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
            **FALSE_AUTHORITY,
        }
        result_path = output_root / "fleet_local_drain_result.json"
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
    return 4 if args.strict and failed_child_count else 0


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

    consume_native = sub.add_parser("consume-native")
    consume_native.add_argument("--output-dir", required=True)
    consume_native.add_argument("--execute", action="store_true")
    consume_native.add_argument("--max-artifacts", type=int, default=4)
    consume_native.add_argument("--artifact-schema", action="append", default=[])
    consume_native.add_argument("--consumer-kind", action="append", default=[])
    consume_native.add_argument("--stop-on-child-failure", action=argparse.BooleanOptionalAction, default=True)
    consume_native.add_argument("--lock-path", default=None)
    consume_native.add_argument("--strict", action="store_true")
    consume_native.set_defaults(func=cmd_consume_native)

    init_missing = sub.add_parser("init-missing")
    init_missing.add_argument("--output-dir", required=True)
    init_missing.add_argument("--execute", action="store_true")
    init_missing.add_argument("--max-queues", type=int, default=8)
    init_missing.add_argument("--stop-on-child-failure", action=argparse.BooleanOptionalAction, default=True)
    init_missing.add_argument("--lock-path", default=None)
    init_missing.add_argument("--strict", action="store_true")
    init_missing.set_defaults(func=cmd_init_missing)

    drain_local = sub.add_parser("drain-local")
    drain_local.add_argument("--output-dir", required=True)
    drain_local.add_argument("--execute", action="store_true")
    drain_local.add_argument("--max-cycles", type=int, default=2)
    drain_local.add_argument("--max-native-artifacts", type=int, default=4)
    drain_local.add_argument("--artifact-schema", action="append", default=[])
    drain_local.add_argument("--consumer-kind", action="append", default=[])
    drain_local.add_argument("--skip-existing-native-outputs", action=argparse.BooleanOptionalAction, default=True)
    drain_local.add_argument("--max-init-queues", type=int, default=4)
    drain_local.add_argument("--max-supervise-queues", type=int, default=4)
    drain_local.add_argument("--include-recovery", action="store_true")
    drain_local.add_argument("--max-ticks-per-queue", type=int, default=16)
    drain_local.add_argument("--max-steps-per-tick", type=int, default=16)
    drain_local.add_argument("--max-parallel", default="auto")
    drain_local.add_argument("--max-parallel-cap", type=int, default=8)
    drain_local.add_argument("--allow-cloud", action="store_true")
    drain_local.add_argument("--no-recover", action="store_true")
    drain_local.add_argument("--stop-on-child-failure", action=argparse.BooleanOptionalAction, default=True)
    drain_local.add_argument("--log-root", default=None)
    drain_local.add_argument("--lock-path", default=None)
    drain_local.add_argument("--strict", action="store_true")
    drain_local.set_defaults(func=cmd_drain_local)
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
