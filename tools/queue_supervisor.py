#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded self-healing supervisor for experiment_queue.v1 queues."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
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
from tac.repo_io import ArtifactWriteError, sha256_file, write_json_artifact  # noqa: E402

SUPERVISOR_SCHEMA = "experiment_queue_supervisor_run.v1"
SUPERVISOR_TICK_SCHEMA = "experiment_queue_supervisor_tick.v1"
SUPERVISOR_HEARTBEAT_SCHEMA = "experiment_queue_supervisor_heartbeat.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _load_queue_control():
    path = REPO_ROOT / "tools" / "queue_control.py"
    spec = importlib.util.spec_from_file_location("queue_control_runtime", path)
    if spec is None or spec.loader is None:
        raise ExperimentQueueError(f"cannot load queue_control module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QC = _load_queue_control()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _repo_rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")


class SupervisorLock:
    def __init__(self, path: Path, *, payload: Mapping[str, Any]) -> None:
        self.path = path
        self.payload = dict(payload)
        self.handle: Any | None = None

    def __enter__(self) -> SupervisorLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            existing = handle.read().strip()
            handle.close()
            raise ExperimentQueueError(
                f"queue supervisor lock already held: {self.path}; existing={existing[:500]}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": self._pid(), "started_at_utc": _utc_now(), **self.payload}, sort_keys=True))
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

    @staticmethod
    def _pid() -> int:
        import os

        return os.getpid()


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _status_count(summary: Mapping[str, Any], status: str) -> int:
    counts = summary.get("status_counts")
    if not isinstance(counts, Mapping):
        return 0
    return _positive_int(counts.get(status)) or 0


def _terminal(summary: Mapping[str, Any]) -> bool:
    return (
        _status_count(summary, "queued") == 0
        and _status_count(summary, "running") == 0
        and _status_count(summary, "failed") == 0
        and _status_count(summary, "blocked") == 0
    )


def _resolve_max_parallel(summary: Mapping[str, Any], *, requested: str, cap: int | None) -> int | None:
    if requested == "none":
        return None
    if requested != "auto":
        parsed = _positive_int(requested)
        if parsed is None:
            raise ExperimentQueueError("--max-parallel must be auto, none, or a positive integer")
        return min(parsed, cap) if cap else parsed
    auto = summary.get("auto_parallelism")
    local = auto.get("local_only") if isinstance(auto, Mapping) else None
    raw = local.get("max_parallel") if isinstance(local, Mapping) else None
    parsed = _positive_int(raw) or 1
    return min(parsed, cap) if cap else parsed


def _worker_command(
    queue_path: str | Path,
    state_path: str | Path,
    *,
    max_steps: int,
    max_parallel: int | None,
    allow_cloud: bool,
    execute: bool,
    log_root: str | Path | None,
    explicit_state_path: bool,
) -> list[str]:
    command = QC.build_resume_command(
        queue_path,
        state_path,
        max_steps=max_steps,
        max_parallel=max_parallel,
        allow_cloud=allow_cloud,
        execute=execute,
        log_root=log_root,
    )
    if execute and explicit_state_path:
        command.extend(
            [
                "--noncanonical-state-rationale",
                "queue_supervisor_explicit_state_path_prevents_default_worker_collision",
            ]
        )
    return command


def _run_command(command: Sequence[str], *, label: str) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        [str(item) for item in command],
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
        "label": label,
        "command": list(command),
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "json_payload": payload,
        "failed": bool(proc.returncode),
    }


def _recover(
    args: argparse.Namespace,
    *,
    state_path: Path,
    tick_dir: Path,
    tick_index: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/queue_control.py",
        "--queue",
        _repo_rel(args.queue),
        "--state",
        _repo_rel(state_path),
        "recover",
        "--reason",
        f"queue_supervisor_tick_{tick_index}_recovery",
        "--stale-after-seconds",
        str(args.stale_after_seconds),
        "--max-steps",
        str(args.max_steps_per_tick),
        "--output",
        _repo_rel(tick_dir / "recovery.json"),
    ]
    if args.include_orphans:
        command.append("--include-orphans")
    if args.allow_cloud:
        command.append("--allow-cloud")
    if args.resume_after_recovery or args.execute:
        command.append("--resume-after-recovery")
    return _run_command(command, label="recover")


def _write_heartbeat(path: Path, payload: Mapping[str, Any]) -> None:
    heartbeat = {
        "schema": SUPERVISOR_HEARTBEAT_SCHEMA,
        "updated_at_utc": _utc_now(),
        **dict(payload),
        **FALSE_AUTHORITY,
    }
    try:
        expected_sha = sha256_file(path) if path.exists() else None
        write_json_artifact(
            path,
            heartbeat,
            allow_overwrite=expected_sha is not None,
            expected_existing_sha256=expected_sha,
        )
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc


def _write_tick(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        artifact = write_json_artifact(path, payload)
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    return {
        "path": artifact.path,
        "bytes": artifact.bytes_written,
        "sha256": artifact.sha256,
    }


def _observe_summary(args: argparse.Namespace, *, state_path: Path) -> dict[str, Any]:
    observation, resolved_state = QC.observe_for_control(
        args.queue,
        state_path,
        tail_lines=args.tail_lines,
        include_orphans=args.include_orphans,
    )
    return QC.summarize_observation(
        observation,
        queue_path=args.queue,
        state_path=resolved_state,
        max_steps=args.max_steps_per_tick,
        max_parallel=None,
        allow_cloud=args.allow_cloud,
    )


def supervise_queue(args: argparse.Namespace) -> dict[str, Any]:
    queue, state_path = QC._load_queue_and_state(args.queue, args.state)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tick_log = output_dir / "ticks.jsonl"
    heartbeat_path = output_dir / "heartbeat.json"
    lock_path = Path(args.lock_path) if args.lock_path else output_dir / "supervisor.lock"
    started_at = _utc_now()
    ticks: list[dict[str, Any]] = []
    final_reason = "max_ticks_reached"
    with SupervisorLock(
        lock_path,
        payload={
            "schema": "experiment_queue_supervisor_lock.v1",
            "queue_id": queue["queue_id"],
            "queue_path": _repo_rel(args.queue),
            "state": _repo_rel(state_path),
        },
    ):
        for tick_index in range(args.max_ticks):
            tick_dir = output_dir / f"tick_{tick_index:04d}"
            tick_dir.mkdir(parents=True, exist_ok=True)
            summary = _observe_summary(args, state_path=state_path)
            max_parallel = _resolve_max_parallel(
                summary,
                requested=args.max_parallel,
                cap=args.max_parallel_cap,
            )
            action = "observe"
            recovery_result: dict[str, Any] | None = None
            worker_result: dict[str, Any] | None = None
            blockers = [str(item) for item in summary.get("automation_blockers") or [] if str(item)]
            if blockers and args.recover:
                action = "recover"
                recovery_result = _recover(args, state_path=state_path, tick_dir=tick_dir, tick_index=tick_index)
                summary = _observe_summary(args, state_path=state_path)
                blockers = [str(item) for item in summary.get("automation_blockers") or [] if str(item)]
            if blockers and args.stop_on_unsafe:
                final_reason = "unsafe_state_after_recovery" if recovery_result else "unsafe_state"
            elif _terminal(summary):
                final_reason = "terminal_queue_state"
            elif _status_count(summary, "queued") > 0:
                action = "run_worker"
                command = _worker_command(
                    args.queue,
                    state_path,
                    max_steps=args.max_steps_per_tick,
                    max_parallel=max_parallel,
                    allow_cloud=args.allow_cloud,
                    execute=args.execute,
                    log_root=args.log_root,
                    explicit_state_path=args.state is not None,
                )
                worker_result = _run_command(command, label="run_worker")
                summary = _observe_summary(args, state_path=state_path)
                if worker_result.get("returncode"):
                    final_reason = "worker_command_failed"
            else:
                final_reason = "no_actionable_work"
            tick_payload = {
                "schema": SUPERVISOR_TICK_SCHEMA,
                "queue_id": queue["queue_id"],
                "tick_index": tick_index,
                "generated_at_utc": _utc_now(),
                "action": action,
                "execute": args.execute,
                "max_parallel": max_parallel,
                "summary": summary,
                "automation_blockers": blockers,
                "recovery_result": recovery_result,
                "worker_result": worker_result,
                "final_reason_if_stopping": final_reason,
                "allowed_use": "queue_supervisor_local_execution_telemetry_only",
                "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
                **FALSE_AUTHORITY,
            }
            tick_artifact = _write_tick(tick_dir / "tick.json", tick_payload)
            tick_record = {
                "schema": SUPERVISOR_TICK_SCHEMA,
                "queue_id": queue["queue_id"],
                "tick_index": tick_index,
                "action": action,
                "final_reason_if_stopping": final_reason,
                "status_counts": summary.get("status_counts", {}),
                "automation_verdict": summary.get("automation_verdict"),
                "tick_artifact": tick_artifact,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            _append_jsonl(tick_log, tick_record)
            _write_heartbeat(
                heartbeat_path,
                {
                    "queue_id": queue["queue_id"],
                    "tick_index": tick_index,
                    "action": action,
                    "final_reason_if_stopping": final_reason,
                    "status_counts": summary.get("status_counts", {}),
                    "automation_verdict": summary.get("automation_verdict"),
                    "tick_artifact": tick_artifact,
                },
            )
            ticks.append(tick_record)
            if final_reason in {
                "unsafe_state",
                "unsafe_state_after_recovery",
                "terminal_queue_state",
                "worker_command_failed",
                "no_actionable_work",
            }:
                break
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    final_summary = _observe_summary(args, state_path=state_path)
    payload = {
        "schema": SUPERVISOR_SCHEMA,
        "queue_id": queue["queue_id"],
        "queue_path": _repo_rel(args.queue),
        "state": _repo_rel(state_path),
        "output_dir": _repo_rel(output_dir),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "execute": args.execute,
        "final_reason": final_reason,
        "tick_count": len(ticks),
        "ticks_jsonl": _repo_rel(tick_log),
        "heartbeat_path": _repo_rel(heartbeat_path),
        "lock_path": _repo_rel(lock_path),
        "last_tick": ticks[-1] if ticks else None,
        "final_summary": final_summary,
        "allowed_use": "queue_supervisor_local_execution_telemetry_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }
    result_path = output_dir / "supervisor_result.json"
    try:
        expected_sha = sha256_file(result_path) if result_path.exists() else None
        artifact = write_json_artifact(
            result_path,
            payload,
            allow_overwrite=expected_sha is not None,
            expected_existing_sha256=expected_sha,
        )
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    payload["artifact"] = {
        "path": artifact.path,
        "bytes": artifact.bytes_written,
        "sha256": artifact.sha256,
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="queue definition JSON or JSON-compatible YAML")
    parser.add_argument("--state", default=None, help="SQLite state path; defaults to canonical queue state")
    parser.add_argument("--output-dir", required=True, help="directory for heartbeat, tick, and result artifacts")
    parser.add_argument("--execute", action="store_true", help="actually run queue workers")
    parser.add_argument("--max-ticks", type=int, default=16)
    parser.add_argument("--max-steps-per-tick", type=int, default=16)
    parser.add_argument("--max-parallel", default="auto", help="auto, none, or positive integer")
    parser.add_argument("--max-parallel-cap", type=int, default=None)
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--recover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-after-recovery", action="store_true")
    parser.add_argument("--stop-on-unsafe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--lock-path", default=None)
    parser.add_argument("--strict", action="store_true", help="exit nonzero on unsafe/failing final reason")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = supervise_queue(args)
    except ExperimentQueueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    _json_print(payload)
    bad_final = payload.get("final_reason") in {
        "unsafe_state",
        "unsafe_state_after_recovery",
        "worker_command_failed",
    }
    return 3 if args.strict and bad_final else 0


if __name__ == "__main__":
    raise SystemExit(main())
