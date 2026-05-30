#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator control surface for experiment_queue.v1 queues.

This is intentionally a thin layer over ``tools/experiment_queue.py`` and the
shared scheduler modules. The SQLite queue state remains the authority; this
tool makes that authority easy to inspect and control without inventing a
second scheduler.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    ExperimentQueueError,
    _safe_log_path_component,
    connect_state,
    default_state_path,
    initialize_queue_state,
    load_queue_definition,
    queue_summary,
    reconcile_stale_running_steps,
    rewind_step,
    set_control_mode,
)
from comma_lab.scheduler.experiment_queue_observer import observe_experiment_queue  # noqa: E402
from comma_lab.scheduler.queue_feedback_replan_policy import (  # noqa: E402
    build_queue_observation_recovery_plan,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

QUEUE_CONTROL_SCHEMA = "experiment_queue_control_surface.v1"
QUEUE_CONTROL_ACTION_SCHEMA = "experiment_queue_control_action.v1"
QUEUE_CONTROL_LOG_TAIL_SCHEMA = "experiment_queue_control_log_tail.v1"
QUEUE_CONTROL_RESUME_COMMAND_SCHEMA = "experiment_queue_resume_command.v1"
QUEUE_CONTROL_RECOVERY_SCHEMA = "experiment_queue_crash_recovery.v1"
QUEUE_CONTROL_AUTO_RECOVERY_REWINDS_SCHEMA = "experiment_queue_auto_recovery_rewinds.v1"

AUTO_RECOVERY_REWIND_ACTIONS = {"rewind_succeeded_step_with_artifact_failure"}

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _repo_rel(path: str | Path, *, repo_root: Path = REPO_ROOT) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _load_queue_and_state(queue_path: str | Path, state_path: str | Path | None) -> tuple[dict[str, Any], Path]:
    queue_file = Path(queue_path)
    if not queue_file.is_file():
        raise ExperimentQueueError(f"queue definition not found: {queue_file}")
    queue = load_queue_definition(queue_file)
    state = Path(state_path) if state_path else default_state_path(REPO_ROOT, str(queue["queue_id"]))
    return queue, state


def _command_path(path: str | Path) -> str:
    return _repo_rel(path)


def build_resume_command(
    queue_path: str | Path,
    state_path: str | Path,
    *,
    max_steps: int | None = None,
    max_parallel: int | None = None,
    allow_cloud: bool = False,
    execute: bool = True,
    no_reload_definition: bool = False,
    log_root: str | Path | None = None,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        _command_path(queue_path),
        "--state",
        _command_path(state_path),
        "run-worker",
    ]
    if execute:
        command.append("--execute")
    if allow_cloud:
        command.append("--allow-cloud")
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if max_parallel is not None:
        command.extend(["--max-parallel", str(max_parallel)])
    if no_reload_definition:
        command.append("--no-reload-definition")
    if log_root is not None:
        command.extend(["--log-root", _command_path(log_root)])
    return command


def build_control_command(queue_path: str | Path, state_path: str | Path, mode: str, *, reason: str) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        _command_path(queue_path),
        "--state",
        _command_path(state_path),
        "control",
        mode,
        "--reason",
        reason,
    ]


def build_supervise_command(
    queue_path: str | Path,
    state_path: str | Path,
    output_dir: str | Path,
    *,
    max_ticks: int = 16,
    max_steps_per_tick: int = 16,
    max_parallel: str = "auto",
    execute: bool = True,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/queue_supervisor.py",
        "--queue",
        _command_path(queue_path),
        "--state",
        _command_path(state_path),
        "--output-dir",
        _command_path(output_dir),
        "--max-ticks",
        str(max_ticks),
        "--max-steps-per-tick",
        str(max_steps_per_tick),
        "--max-parallel",
        max_parallel,
    ]
    if execute:
        command.append("--execute")
    return command


def _count(observation: Mapping[str, Any], status: str) -> int:
    counts = observation.get("status_counts")
    if not isinstance(counts, Mapping):
        return 0
    try:
        return int(counts.get(status) or 0)
    except (TypeError, ValueError):
        return 0


def _step_ids(steps: object, *, limit: int = 12) -> list[str]:
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        return []
    out: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        exp = str(step.get("experiment_id") or "")
        step_id = str(step.get("step_id") or "")
        if exp and step_id:
            out.append(f"{exp}.{step_id}")
        if len(out) >= limit:
            break
    return out


def _running_without_processes(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    running = observation.get("running_steps")
    if not isinstance(running, Sequence) or isinstance(running, (str, bytes, bytearray)):
        return out
    for step in running:
        if not isinstance(step, Mapping):
            continue
        processes = step.get("processes")
        if isinstance(processes, Sequence) and not isinstance(processes, (str, bytes, bytearray)) and processes:
            continue
        out.append(
            {
                "experiment_id": step.get("experiment_id"),
                "step_id": step.get("step_id"),
                "pid": step.get("pid"),
                "worker_run_id": step.get("worker_run_id"),
                "updated_at_utc": step.get("updated_at_utc"),
            }
        )
    return out


def automation_blockers(summary: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if summary.get("healthy") is not True:
        blockers.append("queue_observation_unhealthy")
    counts = summary.get("status_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    for status in ("failed", "blocked"):
        try:
            count = int(counts.get(status) or 0)
        except (TypeError, ValueError):
            count = 0
        if count:
            blockers.append(f"queue_has_{status}_steps:{count}")
    if int(summary.get("orphaned_step_count") or 0):
        blockers.append(f"queue_has_orphaned_steps:{summary.get('orphaned_step_count')}")
    for action in summary.get("recommended_actions") or []:
        if not isinstance(action, Mapping):
            continue
        if action.get("action") == "reconcile_stale_running":
            blockers.append("queue_has_stale_running_rows")
    return list(dict.fromkeys(blockers))


def recommend_next_actions(
    observation: Mapping[str, Any],
    *,
    queue_path: str | Path,
    state_path: str | Path,
    max_steps: int,
    max_parallel: int | None,
    allow_cloud: bool,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    mode = str(observation.get("mode") or "")
    queued = _count(observation, "queued")
    running = _count(observation, "running")
    failed = _count(observation, "failed")
    blocked = _count(observation, "blocked")
    orphaned = int(observation.get("orphaned_step_count") or 0)
    if observation.get("state_watermark", {}).get("state_missing") is True:
        actions.append(
            {
                "rank": 10,
                "action": "initialize_queue_state",
                "reason": "queue_state_missing",
                "command": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _command_path(queue_path),
                    "--state",
                    _command_path(state_path),
                    "init",
                ],
            }
        )
        return actions
    stale_running = _running_without_processes(observation)
    if stale_running:
        actions.append(
            {
                "rank": 20,
                "action": "reconcile_stale_running",
                "reason": "running_steps_have_no_matching_process",
                "affected_steps": stale_running,
                "command": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _command_path(queue_path),
                    "--state",
                    _command_path(state_path),
                    "reconcile-stale-running",
                ],
            }
        )
    if orphaned:
        actions.append(
            {
                "rank": 30,
                "action": "reconcile_orphans",
                "reason": f"state_has_{orphaned}_orphaned_steps",
                "command": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _command_path(queue_path),
                    "--state",
                    _command_path(state_path),
                    "reconcile-state",
                    "--reason",
                    "operator_reviewed_orphaned_queue_state",
                ],
            }
        )
    if mode != "running" and queued:
        actions.append(
            {
                "rank": 40,
                "action": "resume_queue",
                "reason": f"queue_mode_{mode}_with_{queued}_queued_steps",
                "command": build_control_command(
                    queue_path,
                    state_path,
                    "running",
                    reason="operator_resume_after_queue_control_status",
                ),
            }
        )
    if mode == "running" and queued:
        actions.append(
            {
                "rank": 50,
                "action": "run_worker",
                "reason": f"{queued}_queued_steps_available",
                "command": build_resume_command(
                    queue_path,
                    state_path,
                    max_steps=max_steps,
                    max_parallel=max_parallel,
                    allow_cloud=allow_cloud,
                ),
            }
        )
    if running:
        actions.append(
            {
                "rank": 60,
                "action": "observe_running",
                "reason": f"{running}_running_steps",
                "command": [
                    ".venv/bin/python",
                    "tools/queue_control.py",
                    "--queue",
                    _command_path(queue_path),
                    "--state",
                    _command_path(state_path),
                    "tail-logs",
                    "--lines",
                    "80",
                ],
            }
        )
    if failed or blocked:
        actions.append(
            {
                "rank": 70,
                "action": "inspect_failed_or_blocked",
                "reason": f"failed={failed} blocked={blocked}",
                "failed_steps": _step_ids(observation.get("failed_steps")),
                "blocked_steps": _step_ids(observation.get("blocked_steps")),
                "command": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _command_path(queue_path),
                    "--state",
                    _command_path(state_path),
                    "observe",
                    "--format",
                    "markdown",
                    "--tail-lines",
                    "80",
                ],
            }
        )
    actions.sort(key=lambda item: int(item.get("rank") or 999))
    return actions


def summarize_observation(
    observation: Mapping[str, Any],
    *,
    queue_path: str | Path,
    state_path: str | Path,
    max_steps: int = 16,
    max_parallel: int | None = None,
    allow_cloud: bool = False,
) -> dict[str, Any]:
    recommended_actions = recommend_next_actions(
        observation,
        queue_path=queue_path,
        state_path=state_path,
        max_steps=max_steps,
        max_parallel=max_parallel,
        allow_cloud=allow_cloud,
    )
    resume_command = build_resume_command(
        queue_path,
        state_path,
        max_steps=max_steps,
        max_parallel=max_parallel,
        allow_cloud=allow_cloud,
    )
    summary = {
        "schema": QUEUE_CONTROL_SCHEMA,
        "queue_id": observation.get("queue_id"),
        "queue_path": _command_path(queue_path),
        "state": _command_path(state_path),
        "mode": observation.get("mode"),
        "healthy": observation.get("healthy"),
        "status_counts": observation.get("status_counts", {}),
        "blockers": observation.get("blockers", []),
        "blocker_count": observation.get("blocker_count", 0),
        "orphaned_step_count": observation.get("orphaned_step_count", 0),
        "state_watermark": observation.get("state_watermark", {}),
        "auto_parallelism": observation.get("auto_parallelism", {}),
        "runtime_policy": _compact_runtime_policy(observation.get("runtime_policy")),
        "performance": _compact_performance(observation.get("performance")),
        "running_steps": _step_ids(observation.get("running_steps")),
        "failed_steps": _step_ids(observation.get("failed_steps")),
        "blocked_steps": _step_ids(observation.get("blocked_steps")),
        "queued_step_count": _count(observation, "queued"),
        "recommended_actions": recommended_actions,
        "resume_command": resume_command,
        "supervise_command": build_supervise_command(
            queue_path,
            state_path,
            Path(".omx") / "research" / f"queue_supervisor_{observation.get('queue_id')}",
            max_steps_per_tick=max_steps,
            max_parallel=str(max_parallel) if max_parallel is not None else "auto",
        ),
        "pause_command": build_control_command(
            queue_path,
            state_path,
            "paused",
            reason="operator_pause_from_queue_control",
        ),
        "stop_command": build_control_command(
            queue_path,
            state_path,
            "frozen",
            reason="operator_stop_freeze_from_queue_control",
        ),
        "allowed_use": "operator_queue_control_and_local_telemetry_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }
    blockers = automation_blockers(summary)
    summary["automation_verdict"] = "needs_recovery_or_operator_attention" if blockers else "safe_to_resume_or_continue"
    summary["automation_blockers"] = blockers
    return summary


def _compact_runtime_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema": value.get("schema"),
        "advisory_only": value.get("advisory_only"),
        "recommended_max_concurrency": value.get("recommended_max_concurrency", {}),
        "recommended_timeout_seconds_by_resource": value.get(
            "recommended_timeout_seconds_by_resource",
            {},
        ),
        "score_claim": value.get("score_claim"),
        "promotion_eligible": value.get("promotion_eligible"),
        "ready_for_exact_eval_dispatch": value.get("ready_for_exact_eval_dispatch"),
    }


def _compact_performance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    by_resource = value.get("by_resource_kind")
    return {
        "schema": value.get("schema"),
        "telemetry_only": value.get("telemetry_only"),
        "event_count": value.get("event_count"),
        "by_resource_kind": by_resource if isinstance(by_resource, Mapping) else {},
        "score_claim": value.get("score_claim"),
        "promotion_eligible": value.get("promotion_eligible"),
        "ready_for_exact_eval_dispatch": value.get("ready_for_exact_eval_dispatch"),
    }


def observe_for_control(
    queue_path: str | Path,
    state_path: str | Path | None,
    *,
    tail_lines: int,
    include_orphans: bool,
) -> tuple[dict[str, Any], Path]:
    queue, state = _load_queue_and_state(queue_path, state_path)
    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        tail_lines=tail_lines,
        include_orphans=include_orphans,
    )
    return observation, state


def _auto_rewind_recoverable_steps(
    conn: Any,
    queue: Mapping[str, Any],
    *,
    queue_path: str | Path,
    state_path: str | Path,
    reason: str,
    tail_lines: int,
    include_orphans: bool,
) -> dict[str, Any]:
    """Apply recovery-policy rewinds that are local, typed, and artifact-backed."""

    conn.commit()
    observation = observe_experiment_queue(
        queue,
        state_path=state_path,
        repo_root=REPO_ROOT,
        tail_lines=tail_lines,
        include_orphans=include_orphans,
    )
    plan = build_queue_observation_recovery_plan(
        observation,
        queue_path=_command_path(queue_path),
        state_path=_command_path(state_path),
        reason=reason,
    )
    rewound: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for action in plan.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        action_name = str(action.get("action") or "")
        if action.get("required") is not True or action_name not in AUTO_RECOVERY_REWIND_ACTIONS:
            continue
        experiment_id = str(action.get("experiment_id") or "")
        step_id = str(action.get("step_id") or "")
        if not experiment_id or not step_id:
            skipped.append({"action": action_name, "reason": "missing_step_identity"})
            continue
        key = (experiment_id, step_id)
        if key in seen:
            continue
        command = action.get("command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes, bytearray)):
            skipped.append({"action": action_name, "experiment_id": experiment_id, "step_id": step_id, "reason": "missing_command"})
            continue
        if "rewind" not in [str(item) for item in command]:
            skipped.append({"action": action_name, "experiment_id": experiment_id, "step_id": step_id, "reason": "non_rewind_command"})
            continue
        rewind_step(
            conn,
            str(queue["queue_id"]),
            experiment_id,
            step_id,
            reason=reason,
            queue=queue,
            cascade=True,
        )
        seen.add(key)
        rewound.append(
            {
                "action": action_name,
                "experiment_id": experiment_id,
                "step_id": step_id,
                "cascade": True,
            }
        )
    return {
        "schema": QUEUE_CONTROL_AUTO_RECOVERY_REWINDS_SCHEMA,
        "source_observation_generated_at_utc": observation.get("generated_at_utc"),
        "source_blockers": [str(item) for item in observation.get("blockers") or []],
        "recovery_plan_schema": plan.get("schema"),
        "recovery_required": plan.get("recovery_required"),
        "rewind_count": len(rewound),
        "rewound_steps": rewound,
        "skipped_count": len(skipped),
        "skipped_steps": skipped,
        "allowed_use": "local_queue_recovery_state_mutation_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }


def _write_output_if_requested(payload: dict[str, Any], output: Path | None, expected_sha256: str | None) -> None:
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


def cmd_status(args: argparse.Namespace) -> int:
    observation, state = observe_for_control(
        args.queue,
        args.state,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
    )
    summary = summarize_observation(
        observation,
        queue_path=args.queue,
        state_path=state,
        max_steps=args.max_steps,
        max_parallel=args.max_parallel,
        allow_cloud=args.allow_cloud,
    )
    payload: dict[str, Any] = summary
    if args.include_observation:
        payload = {**summary, "observation": observation}
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    if args.format == "text":
        _print_text_status(payload)
    else:
        _json_print(payload)
    return 3 if args.strict and payload.get("automation_blockers") else 0


def _print_text_status(payload: Mapping[str, Any]) -> None:
    print(f"queue: {payload.get('queue_id')} mode={payload.get('mode')} healthy={payload.get('healthy')}")
    print(f"state: {payload.get('state')}")
    print(f"counts: {payload.get('status_counts')}")
    blockers = payload.get("blockers") or []
    if blockers:
        print(f"blockers: {blockers}")
    actions = payload.get("recommended_actions")
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, bytearray)):
        for action in actions[:5]:
            if not isinstance(action, Mapping):
                continue
            command = action.get("command")
            command_text = " ".join(str(item) for item in command) if isinstance(command, Sequence) else ""
            print(f"next: {action.get('action')} - {action.get('reason')}")
            if command_text:
                print(f"  {command_text}")


def _set_mode(
    queue_path: str | Path,
    state_path: str | Path | None,
    mode: str,
    *,
    reason: str,
    tail_lines: int,
    include_orphans: bool,
    reconcile_stale_running: bool = False,
    stale_after_seconds: float = 300.0,
) -> dict[str, Any]:
    queue, state = _load_queue_and_state(queue_path, state_path)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        before = queue_summary(conn, queue, repo_root=REPO_ROOT)
        set_control_mode(conn, str(queue["queue_id"]), mode, reason=reason)
        stale_reconciliation: dict[str, Any] | None = None
        if reconcile_stale_running:
            stale_reconciliation = reconcile_stale_running_steps(
                conn,
                queue,
                repo_root=REPO_ROOT,
                stale_after_seconds=stale_after_seconds,
            )
        after = queue_summary(conn, queue, repo_root=REPO_ROOT)
    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        tail_lines=tail_lines,
        include_orphans=include_orphans,
    )
    return {
        "schema": QUEUE_CONTROL_ACTION_SCHEMA,
        "queue_id": queue["queue_id"],
        "queue_path": _command_path(queue_path),
        "state": _command_path(state),
        "requested_mode": mode,
        "reason": reason,
        "before": {
            "mode": before.get("mode"),
            "status_counts": before.get("status_counts", {}),
        },
        "after": {
            "mode": after.get("mode"),
            "status_counts": after.get("status_counts", {}),
        },
        "stale_reconciliation": stale_reconciliation,
        "observation_summary": summarize_observation(
            observation,
            queue_path=queue_path,
            state_path=state,
            max_steps=16,
            max_parallel=None,
            allow_cloud=False,
        ),
        "allowed_use": "operator_queue_control_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }


def cmd_pause(args: argparse.Namespace) -> int:
    payload = _set_mode(
        args.queue,
        args.state,
        "paused",
        reason=args.reason,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
    )
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    _json_print(payload)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    payload = _set_mode(
        args.queue,
        args.state,
        "running",
        reason=args.reason,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
        reconcile_stale_running=args.reconcile_stale_running,
        stale_after_seconds=args.stale_after_seconds,
    )
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    _json_print(payload)
    return 0


def _pid_is_current_process(pid: int) -> bool:
    return pid == os.getpid() or pid == os.getppid()


def _terminate_running_processes(
    observation: Mapping[str, Any],
    *,
    dry_run: bool,
    sig: signal.Signals = signal.SIGTERM,
) -> list[dict[str, Any]]:
    terminated: list[dict[str, Any]] = []
    running = observation.get("running_steps")
    if not isinstance(running, Sequence) or isinstance(running, (str, bytes, bytearray)):
        return terminated
    for step in running:
        if not isinstance(step, Mapping):
            continue
        raw_pid = step.get("pid")
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        row = {
            "experiment_id": step.get("experiment_id"),
            "step_id": step.get("step_id"),
            "pid": pid,
            "signal": sig.name,
            "dry_run": dry_run,
            "sent": False,
            "blockers": [],
        }
        if pid <= 1 or _pid_is_current_process(pid):
            row["blockers"].append("unsafe_pid_refused")
            terminated.append(row)
            continue
        matching_pids: set[int] = set()
        processes = step.get("processes")
        if isinstance(processes, Sequence) and not isinstance(processes, (str, bytes, bytearray)):
            for proc in processes:
                if not isinstance(proc, Mapping):
                    continue
                try:
                    matching_pids.add(int(proc.get("pid")))
                except (TypeError, ValueError):
                    continue
        if pid not in matching_pids:
            row["blockers"].append("pid_not_matched_by_live_observer_process_table")
            terminated.append(row)
            continue
        if not dry_run:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                row["blockers"].append("process_not_found")
            except PermissionError:
                row["blockers"].append("permission_denied")
            else:
                row["sent"] = True
        terminated.append(row)
    return terminated


def cmd_stop(args: argparse.Namespace) -> int:
    before_observation, state = observe_for_control(
        args.queue,
        args.state,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
    )
    termination = (
        _terminate_running_processes(
            before_observation,
            dry_run=args.terminate_dry_run,
        )
        if args.terminate_running
        else []
    )
    payload = _set_mode(
        args.queue,
        state,
        "frozen",
        reason=args.reason,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
        reconcile_stale_running=args.reconcile_stale_running,
        stale_after_seconds=args.stale_after_seconds,
    )
    payload["operator_action"] = "stop"
    payload["stop_semantics"] = (
        "set queue control mode to frozen so no new work starts; active child "
        "process termination is opt-in via --terminate-running"
    )
    payload["terminate_running"] = termination
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    _json_print(payload)
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    queue, state = _load_queue_and_state(args.queue, args.state)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        before = queue_summary(conn, queue, repo_root=REPO_ROOT)
        set_control_mode(conn, str(queue["queue_id"]), "paused", reason=args.reason)
        stale_reconciliation = reconcile_stale_running_steps(
            conn,
            queue,
            repo_root=REPO_ROOT,
            stale_after_seconds=args.stale_after_seconds,
        )
        after_reconcile = queue_summary(conn, queue, repo_root=REPO_ROOT)
        auto_recovery_rewinds = _auto_rewind_recoverable_steps(
            conn,
            queue,
            queue_path=args.queue,
            state_path=state,
            reason=args.reason,
            tail_lines=args.tail_lines,
            include_orphans=args.include_orphans,
        )
        after_auto_recovery = queue_summary(conn, queue, repo_root=REPO_ROOT)
        if args.resume_after_recovery:
            set_control_mode(
                conn,
                str(queue["queue_id"]),
                "running",
                reason=f"resume_after_recovery:{args.reason}",
            )
        after = queue_summary(conn, queue, repo_root=REPO_ROOT)
    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
    )
    summary = summarize_observation(
        observation,
        queue_path=args.queue,
        state_path=state,
        max_steps=args.max_steps,
        max_parallel=args.max_parallel,
        allow_cloud=args.allow_cloud,
    )
    recovery_blockers = [str(item) for item in summary.get("automation_blockers") or [] if str(item)]
    skipped = stale_reconciliation.get("skipped_steps")
    if isinstance(skipped, Sequence) and not isinstance(skipped, (str, bytes, bytearray)) and skipped:
        recovery_blockers.append(f"stale_reconciliation_skipped_steps:{len(skipped)}")
    payload = {
        "schema": QUEUE_CONTROL_RECOVERY_SCHEMA,
        "queue_id": queue["queue_id"],
        "queue_path": _command_path(args.queue),
        "state": _command_path(state),
        "reason": args.reason,
        "left_paused": not args.resume_after_recovery,
        "before": {
            "mode": before.get("mode"),
            "status_counts": before.get("status_counts", {}),
        },
        "after_reconcile": {
            "mode": after_reconcile.get("mode"),
            "status_counts": after_reconcile.get("status_counts", {}),
        },
        "auto_recovery_rewinds": auto_recovery_rewinds,
        "after_auto_recovery": {
            "mode": after_auto_recovery.get("mode"),
            "status_counts": after_auto_recovery.get("status_counts", {}),
        },
        "after": {
            "mode": after.get("mode"),
            "status_counts": after.get("status_counts", {}),
        },
        "stale_reconciliation": stale_reconciliation,
        "recovery_verdict": "blocked" if recovery_blockers else "ready",
        "recovery_blockers": list(dict.fromkeys(recovery_blockers)),
        "resume_command": summary["resume_command"],
        "control_summary": summary,
        "allowed_use": "operator_crash_recovery_and_resume_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    _json_print(payload)
    return 3 if args.strict and payload["recovery_blockers"] else 0


def _log_paths_from_observation(observation: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in (
        "running_steps",
        "failed_steps",
        "blocked_steps",
        "queued_steps",
        "succeeded_artifact_failure_steps",
        "succeeded_signal_steps",
    ):
        steps = observation.get(key)
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            raw_path = step.get("log_path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def _scan_log_root(queue_id: str, *, log_root: Path) -> list[Path]:
    queue_dir = log_root / _safe_log_path_component(queue_id)
    if not queue_dir.is_dir():
        return []
    return sorted(
        (path for path in queue_dir.rglob("*.log") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _tail_lines(path: Path, *, lines: int, max_bytes: int = 256_000) -> list[str]:
    if lines <= 0 or not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            try:
                handle.seek(max(0, path.stat().st_size - max_bytes), os.SEEK_SET)
            except OSError:
                handle.seek(0)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def cmd_tail_logs(args: argparse.Namespace) -> int:
    observation, state = observe_for_control(
        args.queue,
        args.state,
        tail_lines=args.lines,
        include_orphans=args.include_orphans,
    )
    queue_id = str(observation.get("queue_id") or "")
    paths = _log_paths_from_observation(observation)
    if args.scan_log_root:
        log_root = Path(args.log_root) if args.log_root else REPO_ROOT / ".omx" / "state" / "experiment_queue_logs"
        for path in _scan_log_root(queue_id, log_root=log_root):
            if path not in paths:
                paths.append(path)
    paths = paths[: args.max_files]
    payload = {
        "schema": QUEUE_CONTROL_LOG_TAIL_SCHEMA,
        "queue_id": queue_id,
        "queue_path": _command_path(args.queue),
        "state": _command_path(state),
        "line_count": args.lines,
        "log_count": len(paths),
        "logs": [
            {
                "path": _repo_rel(path),
                "bytes": path.stat().st_size,
                "tail": _tail_lines(path, lines=args.lines),
            }
            for path in paths
        ],
        "allowed_use": "operator_queue_log_inspection_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    _json_print(payload)
    return 0


def cmd_resume_command(args: argparse.Namespace) -> int:
    queue, state = _load_queue_and_state(args.queue, args.state)
    payload = {
        "schema": QUEUE_CONTROL_RESUME_COMMAND_SCHEMA,
        "queue_id": queue["queue_id"],
        "queue_path": _command_path(args.queue),
        "state": _command_path(state),
        "resume_command": build_resume_command(
            args.queue,
            state,
            max_steps=args.max_steps,
            max_parallel=args.max_parallel,
            allow_cloud=args.allow_cloud,
            execute=not args.no_execute,
            no_reload_definition=args.no_reload_definition,
            log_root=args.log_root,
        ),
        "pause_command": build_control_command(
            args.queue,
            state,
            "paused",
            reason="operator_pause_from_queue_control",
        ),
        "stop_command": build_control_command(
            args.queue,
            state,
            "frozen",
            reason="operator_stop_freeze_from_queue_control",
        ),
        "allowed_use": "operator_resume_command_generation_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }
    _write_output_if_requested(payload, args.output, args.expected_output_sha256)
    _json_print(payload)
    return 0


def _add_common_control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reason", required=True)
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-output-sha256", default=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="queue definition JSON or JSON-compatible YAML")
    parser.add_argument("--state", default=None, help="SQLite state path; defaults to the canonical queue state")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("status", help="observe queue health, telemetry, and next actions")
    sp.add_argument("--tail-lines", type=int, default=20)
    sp.add_argument("--include-orphans", action="store_true")
    sp.add_argument("--include-observation", action="store_true")
    sp.add_argument("--format", choices=["json", "text"], default="json")
    sp.add_argument("--strict", action="store_true", help="exit nonzero when recovery/operator attention is needed")
    sp.add_argument("--max-steps", type=int, default=16)
    sp.add_argument("--max-parallel", type=int, default=None)
    sp.add_argument("--allow-cloud", action="store_true")
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--expected-output-sha256", default=None)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("pause", help="pause future queue work")
    _add_common_control_args(sp)
    sp.set_defaults(func=cmd_pause)

    sp = sub.add_parser("resume", help="resume queue work")
    _add_common_control_args(sp)
    sp.add_argument("--reconcile-stale-running", action="store_true")
    sp.add_argument("--stale-after-seconds", type=float, default=300.0)
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("stop", help="freeze queue work; optional explicit child termination")
    _add_common_control_args(sp)
    sp.add_argument("--reconcile-stale-running", action="store_true")
    sp.add_argument("--stale-after-seconds", type=float, default=300.0)
    sp.add_argument("--terminate-running", action="store_true")
    sp.add_argument(
        "--terminate-dry-run",
        action="store_true",
        help="with --terminate-running, report matching PIDs without sending SIGTERM",
    )
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("recover", help="pause, reconcile stale local running rows, and emit resume plan")
    _add_common_control_args(sp)
    sp.add_argument("--stale-after-seconds", type=float, default=300.0)
    sp.add_argument("--resume-after-recovery", action="store_true")
    sp.add_argument("--strict", action="store_true", help="exit nonzero if recovery remains unsafe")
    sp.add_argument("--max-steps", type=int, default=16)
    sp.add_argument("--max-parallel", type=int, default=None)
    sp.add_argument("--allow-cloud", action="store_true")
    sp.set_defaults(func=cmd_recover)

    sp = sub.add_parser("tail-logs", help="tail queue step logs from observer state")
    sp.add_argument("--lines", type=int, default=80)
    sp.add_argument("--max-files", type=int, default=8)
    sp.add_argument("--include-orphans", action="store_true")
    sp.add_argument("--scan-log-root", action="store_true")
    sp.add_argument("--log-root", default=None)
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--expected-output-sha256", default=None)
    sp.set_defaults(func=cmd_tail_logs)

    sp = sub.add_parser("resume-command", help="emit exact command to resume a worker")
    sp.add_argument("--max-steps", type=int, default=16)
    sp.add_argument("--max-parallel", type=int, default=None)
    sp.add_argument("--allow-cloud", action="store_true")
    sp.add_argument("--no-execute", action="store_true")
    sp.add_argument("--no-reload-definition", action="store_true")
    sp.add_argument("--log-root", default=None)
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--expected-output-sha256", default=None)
    sp.set_defaults(func=cmd_resume_command)

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
