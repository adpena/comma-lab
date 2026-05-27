# SPDX-License-Identifier: MIT
"""Queue-owned final-rate attack autoloop helpers.

The CLI in ``tools/build_frontier_final_rate_attack_queue.py`` is intentionally
thin; reusable custody, observation, and bounded follow-up execution belongs in
this module so final-rate work compounds through the same queue authority
surface instead of turning into operator copy/paste.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import sha256_bytes, write_json_artifact, write_text_artifact

from .experiment_queue import default_state_path

POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA = (
    "frontier_final_rate_attack_post_feedback_child_queue_runs.v1"
)
POST_FEEDBACK_CHILD_QUEUE_RUN_SCHEMA = (
    "frontier_final_rate_attack_post_feedback_child_queue_run.v1"
)

POST_FEEDBACK_CHILD_QUEUE_PRIORITY = (
    "operation_materializer_execution_queue",
    "targeted_component_correction_chain_materializer_execution_queue",
    "targeted_component_correction_materialization_queue",
    "operation_chain_compiler_queue",
    "targeted_component_correction_operation_chain_queue",
    "autonomous_chain_optimization_queue",
    "repair_campaign_score_queue",
    "repair_budget_waterfill_queue",
    "receiver_repair_queue",
    "targeted_component_correction_queue",
)

RunCommand = Callable[[list[str]], dict[str, Any]]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _run_command(command: list[str], *, repo_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _json_stdout_object(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    try:
        payload = json.loads(str(result.get("stdout") or ""))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _queue_id_from_path(path: Path) -> str:
    payload = _load_json_object(path)
    if payload.get("schema") != "experiment_queue.v1":
        raise ValueError(f"{path}: expected experiment_queue.v1")
    try:
        require_no_truthy_authority_fields(
            payload,
            context=f"post_feedback_child_queue:{path}",
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    queue_id = str(payload.get("queue_id") or "").strip()
    if not queue_id:
        raise ValueError(f"{path}: experiment_queue.v1 missing queue_id")
    return queue_id


def _write_command_streams(
    *,
    results: Sequence[Mapping[str, Any]],
    log_dir: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        row = {
            key: value
            for key, value in dict(result).items()
            if key not in {"stdout", "stderr"}
        }
        for stream_name in ("stdout", "stderr"):
            stream = str(result.get(stream_name) or "")
            stream_bytes = stream.encode("utf-8")
            row[f"{stream_name}_bytes"] = len(stream_bytes)
            row[f"{stream_name}_sha256"] = sha256_bytes(stream_bytes)
            if stream:
                path = log_dir / f"{index:02d}_{stream_name}.txt"
                write_text_artifact(path, stream)
                row[f"{stream_name}_path"] = _repo_rel(path, repo_root)
        compacted.append(row)
    return compacted


def select_post_feedback_child_queue_artifacts(
    artifacts: Mapping[str, Any],
    *,
    repo_root: str | Path,
    limit: int,
) -> list[dict[str, str]]:
    """Return bounded child queues from a feedback-refresh artifact map.

    Selection is intentionally keyed and ordered. Broad "find every queue file"
    behavior would make follow-up execution depend on incidental filenames,
    which is exactly the manual/ad hoc failure mode this helper is meant to
    remove.
    """

    if limit < 1:
        raise ValueError("limit must be >= 1")
    repo = Path(repo_root)
    selected: list[dict[str, str]] = []
    for artifact_key in POST_FEEDBACK_CHILD_QUEUE_PRIORITY:
        raw_path = artifacts.get(artifact_key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = _resolve_path(raw_path, repo_root=repo)
        if not path.is_file():
            continue
        selected.append(
            {
                "artifact_key": artifact_key,
                "queue_path": _repo_rel(path, repo),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def run_experiment_queue_once(
    *,
    repo_root: str | Path,
    queue_path: str | Path,
    observer_output_path: str | Path,
    max_steps: int,
    max_parallel: int,
    poll_interval_seconds: float = 0.05,
    idle_sleep_seconds: float = 0.0,
    max_idle_cycles: int = 1,
    run_command: RunCommand | None = None,
) -> dict[str, Any]:
    """Validate, initialize, run, observe, and persist one queue observation."""

    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if max_parallel < 0:
        raise ValueError("max_parallel must be >= 0")
    repo = Path(repo_root)
    queue = _resolve_path(queue_path, repo_root=repo)
    observer_path = _resolve_path(observer_output_path, repo_root=repo)
    queue_id = _queue_id_from_path(queue)
    state_path = default_state_path(repo, queue_id)
    runner = run_command or (lambda command: _run_command(command, repo_root=repo))
    queue_ref = _repo_rel(queue, repo)
    state_ref = _repo_rel(state_path, repo)
    commands = [
        [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            queue_ref,
            "validate",
        ],
        [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            queue_ref,
            "--state",
            state_ref,
            "init",
        ],
        [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            queue_ref,
            "--state",
            state_ref,
            "run-worker",
            "--execute",
            "--max-steps",
            str(max_steps),
            "--max-parallel",
            str(max_parallel),
            "--poll-interval-seconds",
            str(poll_interval_seconds),
            "--idle-sleep-seconds",
            str(idle_sleep_seconds),
            "--max-idle-cycles",
            str(max_idle_cycles),
        ],
        [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            queue_ref,
            "--state",
            state_ref,
            "observe",
            "--format",
            "json",
        ],
    ]
    results: list[dict[str, Any]] = []
    for command in commands:
        result = runner(command)
        results.append(result)
        if int(result.get("returncode") or 0) != 0:
            break

    observation: dict[str, Any] | None = None
    worker_result = _json_stdout_object(results[2] if len(results) > 2 else None)
    if len(results) == len(commands) and int(results[-1].get("returncode") or 0) == 0:
        observation = _json_stdout_object(results[-1])
        if observation is not None:
            write_json_artifact(observer_path, observation)

    command_records = _write_command_streams(
        results=results,
        log_dir=observer_path.parent / "command_logs",
        repo_root=repo,
    )
    failed_count = sum(1 for result in results if int(result.get("returncode") or 0) != 0)
    steps_started: int | None = None
    if worker_result is not None:
        raw_steps_started = worker_result.get("steps_started")
        if isinstance(raw_steps_started, int) and not isinstance(raw_steps_started, bool):
            steps_started = raw_steps_started
        else:
            step_results = worker_result.get("step_results")
            if isinstance(step_results, list):
                steps_started = len(step_results)
    queued_after = 0
    if observation is not None:
        status_counts = observation.get("status_counts")
        if isinstance(status_counts, Mapping):
            raw_queued = status_counts.get("queued")
            if isinstance(raw_queued, int) and not isinstance(raw_queued, bool):
                queued_after = raw_queued
    progress_made = None if steps_started is None else steps_started > 0
    progress_blockers = []
    if progress_made is False and queued_after > 0:
        progress_blockers.append("child_queue_worker_started_zero_steps_with_queued_work")
    return {
        "schema": POST_FEEDBACK_CHILD_QUEUE_RUN_SCHEMA,
        "queue_id": queue_id,
        "queue_path": queue_ref,
        "state_path": state_ref,
        "observer_revalidation_path": (
            _repo_rel(observer_path, repo) if observation is not None else None
        ),
        "commands": command_records,
        "failed_command_count": failed_count,
        "steps_started": steps_started,
        "progress_made": progress_made,
        "progress_blockers": progress_blockers,
        "queue_healthy": observation.get("healthy") is True if observation else False,
        "queue_status_counts": dict(observation.get("status_counts") or {}) if observation else {},
        "queue_blockers": list(observation.get("blockers") or []) if observation else [],
        "allowed_use": "bounded_local_post_feedback_queue_execution_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def execute_post_feedback_child_queues(
    *,
    repo_root: str | Path,
    feedback_artifacts: Mapping[str, Any],
    output_dir: str | Path,
    max_steps: int = 8,
    max_parallel: int = 1,
    limit: int = 4,
    poll_interval_seconds: float = 0.05,
    idle_sleep_seconds: float = 0.0,
    max_idle_cycles: int = 1,
    run_command: RunCommand | None = None,
) -> dict[str, Any]:
    """Run selected post-feedback queues and persist a single custody report."""

    repo = Path(repo_root)
    out = _resolve_path(output_dir, repo_root=repo)
    selected = select_post_feedback_child_queue_artifacts(
        feedback_artifacts,
        repo_root=repo,
        limit=limit,
    )
    runs: list[dict[str, Any]] = []
    observation_dir = out / "post_execute_feedback_child_queue_observations"
    for row in selected:
        key = row["artifact_key"]
        observer_path = observation_dir / key / "observer_revalidation.json"
        run = run_experiment_queue_once(
            repo_root=repo,
            queue_path=row["queue_path"],
            observer_output_path=observer_path,
            max_steps=max_steps,
            max_parallel=max_parallel,
            poll_interval_seconds=poll_interval_seconds,
            idle_sleep_seconds=idle_sleep_seconds,
            max_idle_cycles=max_idle_cycles,
            run_command=run_command,
        )
        run["artifact_key"] = key
        runs.append(run)
    report = {
        "schema": POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "selected_queue_count": len(selected),
        "executed_queue_count": len(runs),
        "failed_queue_count": sum(1 for run in runs if int(run.get("failed_command_count") or 0) > 0),
        "failed_command_count": sum(int(run.get("failed_command_count") or 0) for run in runs),
        "stalled_queue_count": sum(
            1 for run in runs if run.get("progress_made") is False and run.get("queue_status_counts", {}).get("queued", 0)
        ),
        "max_steps": max_steps,
        "max_parallel": max_parallel,
        "limit": limit,
        "selected_queues": selected,
        "queue_runs": runs,
        "allowed_use": "post_feedback_bounded_local_autoloop_custody_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(report, context="post_feedback_child_queue_runs")
    report_path = out / "post_execute_feedback_child_queue_runs.json"
    write_json_artifact(report_path, report)
    report["artifact_path"] = _repo_rel(report_path, repo)
    return report


__all__ = [
    "POST_FEEDBACK_CHILD_QUEUE_PRIORITY",
    "POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA",
    "execute_post_feedback_child_queues",
    "run_experiment_queue_once",
    "select_post_feedback_child_queue_artifacts",
]
