from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    _condition_passes,
    connect_state_readonly,
    queue_definition_drift,
    queue_performance_summary,
    queue_resource_kinds,
    queue_summary,
    resolve_worker_max_parallel,
)

OBSERVATION_SCHEMA = "experiment_queue_observation.v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_load_lenient(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _tail_text(path: Path, *, max_lines: int, max_bytes: int = 64_000) -> list[str]:
    if max_lines <= 0 or not path.is_file():
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
    return text.splitlines()[-max_lines:]


def _process_table() -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,etime,command"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return []
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        rows.append({"pid": parts[0], "etime": parts[1], "command": parts[2]})
    return rows


def _matching_processes(
    processes: Sequence[Mapping[str, str]],
    needles: Sequence[str],
) -> list[dict[str, str]]:
    real_needles = [needle for needle in needles if needle]
    if not real_needles:
        return []
    out: list[dict[str, str]] = []
    for proc in processes:
        command = str(proc.get("command") or "")
        if any(needle in command for needle in real_needles):
            out.append({
                "pid": str(proc.get("pid") or ""),
                "etime": str(proc.get("etime") or ""),
                "command": command,
            })
    return out


def _path_artifact_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _repo_rel(path, repo_root),
        "exists": path.exists(),
    }
    if path.exists():
        try:
            stat = path.stat()
        except OSError:
            return record
        record["bytes"] = stat.st_size
        record["mtime_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(stat.st_mtime),
        )
        payload = _json_load_lenient(path) if path.suffix == ".json" else None
        if payload is not None:
            record["json_schema"] = payload.get("schema") or payload.get("schema_version")
            for key in (
                "candidate_id",
                "canonical_score",
                "score_axis",
                "archive_sha256",
                "archive_bytes",
                "eureka",
                "margin_vs_frontier",
            ):
                if key in payload:
                    record[key] = payload[key]
            score = payload.get("score")
            if isinstance(score, Mapping):
                record["score"] = {
                    key: score.get(key)
                    for key in (
                        "canonical_score",
                        "archive_bytes",
                        "posenet",
                        "segnet",
                        "rate",
                    )
                    if key in score
                }
    return record


def _experiment_lookup(queue: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping):
            continue
        experiment_id = str(experiment.get("id") or "")
        for step in experiment.get("steps", []):
            if isinstance(step, Mapping):
                out[(experiment_id, str(step.get("id") or ""))] = step
    return out


def _expected_artifacts(
    step: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for condition in step.get("postconditions", []):
        if not isinstance(condition, Mapping):
            continue
        path_value = condition.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = repo_root / path
        record = _path_artifact_record(path, repo_root=repo_root)
        record["postcondition_type"] = condition.get("type")
        try:
            record["postcondition_passed"] = _condition_passes(
                condition,
                repo_root=repo_root,
            )
        except (
            ExperimentQueueError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            record["postcondition_passed"] = False
            record["postcondition_error"] = f"{type(exc).__name__}: {exc}"
        artifacts.append(record)
    return artifacts


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _auto_parallelism_observation(queue: Mapping[str, Any]) -> dict[str, Any]:
    local_parallel, local_limits = resolve_worker_max_parallel(queue, 0, allow_cloud=False)
    cloud_parallel, cloud_limits = resolve_worker_max_parallel(queue, 0, allow_cloud=True)
    controls = queue.get("controls")
    configured = controls.get("max_concurrency") if isinstance(controls, Mapping) else {}
    configured = configured if isinstance(configured, Mapping) else {}
    used_kinds = queue_resource_kinds(queue)
    used = set(used_kinds)
    idle_declared_resources: dict[str, int] = {}
    for kind, raw_limit in configured.items():
        kind_text = str(kind)
        limit = _positive_int(raw_limit)
        if limit is not None and kind_text not in used:
            idle_declared_resources[kind_text] = limit
    return {
        "local_only": {
            "max_parallel": local_parallel,
            "resource_limits": local_limits,
        },
        "with_cloud": {
            "max_parallel": cloud_parallel,
            "resource_limits": cloud_limits,
        },
        "used_resource_kinds": used_kinds,
        "idle_declared_resources": idle_declared_resources,
    }


def _step_observation(
    step_state: Mapping[str, Any],
    step_definition: Mapping[str, Any] | None,
    *,
    repo_root: Path,
    processes: Sequence[Mapping[str, str]],
    tail_lines: int,
) -> dict[str, Any]:
    event = step_state.get("last_event")
    event = event if isinstance(event, Mapping) else {}
    experiment_id = str(step_state.get("experiment_id") or "")
    step_id = str(step_state.get("step_id") or "")
    command = event.get("command")
    command_text = " ".join(str(part) for part in command) if isinstance(command, list) else ""
    log_path = Path(str(event.get("log_path"))) if event.get("log_path") else None
    needles = [experiment_id, step_id]
    if command_text:
        needles.append(command_text[:160])
    if log_path is not None:
        needles.append(log_path.name)
    observation: dict[str, Any] = {
        "experiment_id": experiment_id,
        "step_id": step_id,
        "status": step_state.get("status"),
        "attempts": step_state.get("attempts"),
        "updated_at_utc": step_state.get("updated_at_utc"),
        "resource_kind": None,
        "worker_run_id": event.get("worker_run_id"),
        "pid": event.get("pid"),
        "timeout_seconds": event.get("timeout_seconds"),
        "timeout_deadline_epoch_seconds": event.get("timeout_deadline_epoch_seconds"),
        "log_path": _repo_rel(log_path, repo_root) if log_path is not None else None,
        "log_exists": bool(log_path and log_path.exists()),
        "processes": _matching_processes(processes, needles),
        "expected_artifacts": [],
    }
    if step_definition is not None:
        resources = step_definition.get("resources")
        if isinstance(resources, Mapping):
            observation["resource_kind"] = resources.get("kind")
        observation["expected_artifacts"] = _expected_artifacts(
            step_definition,
            repo_root=repo_root,
        )
    if log_path is not None and log_path.exists():
        observation["log_tail"] = _tail_text(log_path, max_lines=tail_lines)
    return observation


def observe_experiment_queue(
    queue: Mapping[str, Any],
    *,
    state_path: Path,
    repo_root: Path,
    tail_lines: int = 20,
    include_orphans: bool = False,
) -> dict[str, Any]:
    """Return a compact operator-facing observation for a queue."""

    try:
        with connect_state_readonly(state_path) as conn:
            summary = queue_summary(conn, queue, repo_root=repo_root)
            definition_drift = queue_definition_drift(conn, queue)
            performance = queue_performance_summary(conn, queue)
    except ExperimentQueueError:
        summary = {
            "queue_id": str(queue["queue_id"]),
            "mode": str(queue.get("controls", {}).get("mode") or "unknown"),
            "status_counts": {},
            "step_count": 0,
            "orphaned_step_count": 0,
            "ready_steps": [],
            "steps": [],
            "orphaned_steps": [],
        }
        definition_drift = {
            "schema": "experiment_queue_definition_drift.v1",
            "read_only": True,
            "state_missing": True,
            "missing_step_count": sum(
                len(experiment.get("steps") or [])
                for experiment in queue.get("experiments", [])
                if isinstance(experiment, Mapping)
            ),
            "changed_step_count": 0,
            "missing_hash_step_count": 0,
            "missing_steps": [],
            "changed_steps": [],
            "missing_hash_steps": [],
        }
        performance = {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": str(queue["queue_id"]),
            "state_missing": True,
            "telemetry_only": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "event_count": 0,
            "by_resource_kind": {},
            "by_step": {},
        }
    lookup = _experiment_lookup(queue)
    processes = _process_table()
    active_steps: list[dict[str, Any]] = []
    for step_state in summary.get("steps", []):
        if not isinstance(step_state, Mapping):
            continue
        if step_state.get("status") not in {"running", "failed", "queued"}:
            continue
        key = (
            str(step_state.get("experiment_id") or ""),
            str(step_state.get("step_id") or ""),
        )
        active_steps.append(
            _step_observation(
                step_state,
                lookup.get(key),
                repo_root=repo_root,
                processes=processes,
                tail_lines=tail_lines,
            )
        )
    orphaned = []
    if include_orphans:
        for step_state in summary.get("orphaned_steps", []):
            if isinstance(step_state, Mapping):
                orphaned.append(
                    _step_observation(
                        step_state,
                        None,
                        repo_root=repo_root,
                        processes=processes,
                        tail_lines=0,
                    )
                )

    running = [step for step in active_steps if step.get("status") == "running"]
    failed = [step for step in active_steps if step.get("status") == "failed"]
    queued = [step for step in active_steps if step.get("status") == "queued"]
    return {
        "schema": OBSERVATION_SCHEMA,
        "generated_at_utc": _utc_now(),
        "queue_id": summary["queue_id"],
        "mode": summary["mode"],
        "state": _repo_rel(state_path, repo_root),
        "status_counts": summary.get("status_counts", {}),
        "observe_read_only": True,
        "definition_drift": definition_drift,
        "performance": performance,
        "auto_parallelism": _auto_parallelism_observation(queue),
        "step_count": summary.get("step_count"),
        "orphaned_step_count": summary.get("orphaned_step_count"),
        "ready_steps": summary.get("ready_steps", []),
        "running_steps": running,
        "failed_steps": failed,
        "queued_steps": queued,
        "orphaned_steps": orphaned,
        "suggested_commands": {
            "refresh": (
                f".venv/bin/python tools/experiment_queue.py --queue "
                f"<queue-path> observe --tail-lines {tail_lines}"
            ),
            "pause": ".venv/bin/python tools/experiment_queue.py --queue <queue-path> control paused --reason '<reason>'",
            "resume": ".venv/bin/python tools/experiment_queue.py --queue <queue-path> control running --reason '<reason>'",
            "run_worker": ".venv/bin/python tools/experiment_queue.py --queue <queue-path> run-worker --execute",
        },
    }


def render_observation_markdown(observation: Mapping[str, Any]) -> str:
    """Render a small live dashboard suitable for terminal or memo paste."""

    lines = [
        f"# Experiment Queue Observation: {observation.get('queue_id')}",
        "",
        f"- generated_at_utc: `{observation.get('generated_at_utc')}`",
        f"- mode: `{observation.get('mode')}`",
        f"- state: `{observation.get('state')}`",
        f"- status_counts: `{observation.get('status_counts')}`",
        f"- auto_parallelism: `{observation.get('auto_parallelism')}`",
        f"- performance: `{_performance_markdown_summary(observation.get('performance'))}`",
        f"- orphaned_step_count: `{observation.get('orphaned_step_count')}`",
        "",
        "| status | experiment | step | log | artifacts | processes |",
        "|---|---|---|---|---:|---:|",
    ]
    steps = [
        *list(observation.get("running_steps") or []),
        *list(observation.get("failed_steps") or []),
        *list(observation.get("queued_steps") or []),
    ]
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        artifacts = step.get("expected_artifacts") or []
        passed = sum(
            1
            for item in artifacts
            if isinstance(item, Mapping) and item.get("postcondition_passed")
        )
        log_path = step.get("log_path") or ""
        lines.append(
            "| {status} | `{exp}` | `{step}` | `{log}` | {existing}/{total} | {pids} |".format(
                status=step.get("status"),
                exp=step.get("experiment_id"),
                step=step.get("step_id"),
                log=log_path,
                existing=passed,
                total=len(artifacts),
                pids=len(step.get("processes") or []),
            )
        )
    return "\n".join(lines) + "\n"


def _performance_markdown_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "event_count": value.get("event_count"),
        "telemetry_only": value.get("telemetry_only"),
        "score_claim": value.get("score_claim"),
        "by_resource_kind": {
            str(key): {
                "runs": bucket.get("run_count"),
                "successes": bucket.get("success_count"),
                "mean_seconds": bucket.get("elapsed_seconds_mean"),
            }
            for key, bucket in dict(value.get("by_resource_kind") or {}).items()
            if isinstance(bucket, Mapping)
        },
    }


__all__ = [
    "OBSERVATION_SCHEMA",
    "observe_experiment_queue",
    "render_observation_markdown",
]
