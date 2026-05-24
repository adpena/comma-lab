from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)

from .experiment_queue import (
    ExperimentQueueError,
    ReadyStep,
    _json_text,
    _lookup_step,
    _utc_now,
    assert_canonical_state_for_execution,
    assert_no_orphaned_steps_for_execution,
    claim_ready_step_for_execution,
    connect_state,
    connect_state_readonly,
    finalize_claimed_step_execution,
    initialize_queue_state,
    ready_steps,
)
from .staircase_dag import STAIRCASE_DISPATCH_PLAN_SCHEMA

SSH_EXECUTION_RESULT_SCHEMA = "staircase_ssh_execution_result.v1"
SSH_EXECUTOR_EVENT_SCHEMA = "staircase_ssh_executor_event.v1"
SSH_ARTIFACT_MOBILITY_SCHEMA = "staircase_ssh_artifact_mobility.v1"
SSH_EXECUTOR_NAMES = {"ssh_experiment_queue"}
FUTURE_SSH_EXECUTOR_NAMES = {"ssh_experiment_queue_future"}
_UNSAFE_REMOTE_ARTIFACT_CHARS = frozenset(" \t\r\n'\"`$;&|<>\\")

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SshTaskSelection:
    task: Mapping[str, Any]
    ready_step: ReadyStep | None
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "key": self.task.get("key"),
            "queue_id": self.task.get("queue_id"),
            "experiment_id": self.task.get("experiment_id"),
            "step_id": self.task.get("step_id"),
            "machine_hint": self.task.get("machine_hint"),
            "machine": dict(self.task.get("machine") or {}),
            "blockers": list(self.blockers),
        }
        if self.ready_step is not None:
            payload["ready_step"] = self.ready_step.to_dict()
        return payload


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentQueueError(f"{label} must be a non-empty string")
    return value.strip()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentQueueError(f"{label} must be an object")
    return dict(value)


def _optional_mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value, label)


def _local_git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ExperimentQueueError(f"failed to read local git HEAD: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()


def _require_matching_queue_source_ref(plan: Mapping[str, Any], queue: Mapping[str, Any]) -> None:
    refs = plan.get("source_refs")
    if not isinstance(refs, list):
        raise ExperimentQueueError("plan.source_refs must be a list")
    queue_id = str(queue["queue_id"])
    queue_hash = _sha256_json(queue)
    matches = [
        ref
        for ref in refs
        if isinstance(ref, Mapping)
        and ref.get("kind") == "experiment_queue"
        and ref.get("queue_id") == queue_id
    ]
    if not matches:
        raise ExperimentQueueError("plan.source_refs missing matching experiment_queue reference")
    for ref in matches:
        if ref.get("queue_hash") == queue_hash:
            return
    raise ExperimentQueueError("plan.source_refs experiment_queue queue_hash mismatch")


def _ssh_argv(
    *,
    ssh_binary: str,
    ssh_target: str,
    connect_timeout_seconds: int,
    remote_script: str,
) -> list[str]:
    return [
        ssh_binary,
        "-o",
        f"ConnectTimeout={connect_timeout_seconds}",
        "-o",
        "BatchMode=yes",
        ssh_target,
        remote_script,
    ]


def _ssh_transport_arg(*, ssh_binary: str, connect_timeout_seconds: int) -> str:
    return shlex.join(
        [
            ssh_binary,
            "-o",
            f"ConnectTimeout={connect_timeout_seconds}",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
        ]
    )


def build_remote_shell_command(
    *,
    remote_repo_root: str,
    command: Sequence[str],
    expected_head: str | None = None,
    require_clean: bool = False,
) -> str:
    """Return the remote shell command used to execute a queue-owned argv."""

    if not command:
        raise ExperimentQueueError("remote command argv must be non-empty")
    remote_command = shlex.join([str(part) for part in command])
    if expected_head is None:
        return f"cd {shlex.quote(remote_repo_root)} && {remote_command}"
    preflight = build_remote_git_preflight_command(
        remote_repo_root=remote_repo_root,
        expected_head=expected_head,
        require_clean=require_clean,
    )
    return f"{preflight} && {remote_command}"


def build_remote_git_preflight_command(
    *,
    remote_repo_root: str,
    expected_head: str,
    require_clean: bool,
) -> str:
    checks = [
        f"cd {shlex.quote(remote_repo_root)}",
        "test -d .git",
        f'test "$(git rev-parse HEAD)" = {shlex.quote(expected_head)}',
    ]
    if require_clean:
        checks.extend(["git diff --quiet", "git diff --cached --quiet"])
    return " && ".join(checks)


def _machine_for_task(task: Mapping[str, Any]) -> dict[str, Any]:
    machine = task.get("machine")
    if isinstance(machine, Mapping):
        return dict(machine)
    hint = task.get("machine_hint")
    return {"id": hint} if isinstance(hint, str) and hint else {}


def _task_step_hashes(task: Mapping[str, Any]) -> dict[str, str]:
    hashes = task.get("step_hashes")
    if not isinstance(hashes, Mapping):
        return {}
    return {str(key): str(value) for key, value in hashes.items() if isinstance(value, str)}


def _queue_state_writeback_blockers(task: Mapping[str, Any], *, queue_id: str) -> list[str]:
    writeback = task.get("queue_state_writeback")
    if not isinstance(writeback, Mapping):
        return ["queue_state_writeback_missing"]
    blockers: list[str] = []
    if writeback.get("queue_id") != task.get("queue_id") or writeback.get("queue_id") != queue_id:
        blockers.append("queue_state_writeback_queue_id_mismatch")
    for key in ("experiment_id", "step_id"):
        if writeback.get(key) != task.get(key):
            blockers.append(f"queue_state_writeback_{key}_mismatch")
    task_hashes = _task_step_hashes(task)
    if not task_hashes:
        blockers.append("task_step_hashes_missing")
    writeback_hashes = writeback.get("step_hashes")
    if not isinstance(writeback_hashes, Mapping):
        blockers.append("queue_state_writeback_step_hashes_missing")
    else:
        normalized_writeback_hashes = {
            str(key): str(value)
            for key, value in writeback_hashes.items()
            if isinstance(value, str)
        }
        if task_hashes and normalized_writeback_hashes != task_hashes:
            blockers.append("queue_state_writeback_step_hashes_mismatch")
    if writeback.get("required") is not True:
        blockers.append("queue_state_writeback_required_not_true")
    if writeback.get("executor_must_claim_step_before_execution") is not True:
        blockers.append("queue_state_writeback_claim_requirement_missing")
    if writeback.get("executor_must_record_terminal_step_event") is not True:
        blockers.append("queue_state_writeback_terminal_event_requirement_missing")
    terminal = writeback.get("terminal_statuses")
    terminal_set = {str(item) for item in terminal} if isinstance(terminal, list) else set()
    if not isinstance(terminal, list) or {"succeeded", "failed"} - terminal_set:
        blockers.append("queue_state_writeback_terminal_statuses_incomplete")
    if terminal_set - {"succeeded", "failed"}:
        blockers.append("queue_state_writeback_terminal_statuses_unknown")
    return blockers


def _remote_repo_root_blockers(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return ["remote_repo_root_missing"]
    if not value.strip().startswith("/"):
        return ["remote_repo_root_must_be_absolute"]
    return []


def _resolve_local_artifact_path(path_value: str, *, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _postcondition_artifact_paths(step: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for condition in [dict(condition) for condition in step.get("postconditions", [])]:
        path_value = condition.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        path = _resolve_local_artifact_path(path_value, repo_root=repo_root)
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _telemetry_artifact_paths(step: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    telemetry = step.get("telemetry")
    raw_paths = telemetry.get("artifact_paths") if isinstance(telemetry, Mapping) else None
    if not isinstance(raw_paths, list):
        return []
    seen: set[str] = set()
    paths: list[Path] = []
    for path_value in raw_paths:
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        path = _resolve_local_artifact_path(path_value, repo_root=repo_root)
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _telemetry_pullback_artifact_paths(
    step: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[Path]:
    telemetry = step.get("telemetry")
    raw_paths = (
        telemetry.get("pullback_artifact_paths")
        if isinstance(telemetry, Mapping)
        else None
    )
    if not isinstance(raw_paths, list):
        return []
    seen: set[str] = set()
    paths: list[Path] = []
    for path_value in raw_paths:
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        path = _resolve_local_artifact_path(path_value, repo_root=repo_root)
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _artifact_paths_for_pullback(step: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for path in (
        *_postcondition_artifact_paths(step, repo_root=repo_root),
        *_telemetry_pullback_artifact_paths(step, repo_root=repo_root),
    ):
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _local_visible_postcondition_blockers(
    task: Mapping[str, Any],
    step: Mapping[str, Any],
) -> list[str]:
    task_postconditions = task.get("postconditions")
    step_postconditions = step.get("postconditions")
    blockers: list[str] = []
    if not isinstance(task_postconditions, list) or not task_postconditions:
        blockers.append("task_postconditions_missing_for_ssh")
    if not isinstance(step_postconditions, list) or not step_postconditions:
        blockers.append("ssh_executor_local_visible_postcondition_required")
    return blockers


def _normalize_path_maps(
    path_maps: Mapping[str, str] | None,
    *,
    repo_root: Path,
) -> list[tuple[Path, str]]:
    normalized: list[tuple[Path, str]] = []
    for raw_local, raw_remote in dict(path_maps or {}).items():
        local_prefix = _resolve_local_artifact_path(str(raw_local), repo_root=repo_root)
        remote_prefix = str(raw_remote).strip()
        normalized_remote = remote_prefix.rstrip("/") or "/"
        _require_safe_remote_artifact_path(normalized_remote, label="artifact path map remote prefix")
        normalized.append((local_prefix.resolve(strict=False), normalized_remote))
    return sorted(normalized, key=lambda item: len(item[0].as_posix()), reverse=True)


def _remote_artifact_path_for_local(
    local_path: Path,
    *,
    path_maps: Sequence[tuple[Path, str]],
) -> str | None:
    resolved = local_path.resolve(strict=False)
    for local_prefix, remote_prefix in path_maps:
        try:
            relative = resolved.relative_to(local_prefix)
        except ValueError:
            continue
        if relative.parts:
            if remote_prefix == "/":
                return f"/{relative.as_posix()}"
            return f"{remote_prefix}/{relative.as_posix()}"
        return remote_prefix
    return None


def _require_safe_remote_artifact_path(remote_path: str, *, label: str = "remote artifact path") -> str:
    text = _require_text(remote_path, label)
    if not text.startswith("/"):
        raise ExperimentQueueError(f"{label} must be absolute")
    if any(char in _UNSAFE_REMOTE_ARTIFACT_CHARS for char in text):
        raise ExperimentQueueError(f"{label} contains unsafe shell/rsync characters")
    if any(part == ".." for part in text.split("/")):
        raise ExperimentQueueError(f"{label} must not contain '..'")
    return text


def _artifact_mobility_blockers(
    task: Mapping[str, Any],
    step: Mapping[str, Any],
    *,
    repo_root: Path,
    path_maps: Sequence[tuple[Path, str]],
    require_artifact_mobility: bool,
    artifact_shared_path_rationale: str | None,
) -> list[str]:
    step_mobility = step.get("artifact_mobility") if isinstance(step.get("artifact_mobility"), Mapping) else {}
    task_mobility = task.get("artifact_mobility") if isinstance(task.get("artifact_mobility"), Mapping) else {}
    if step_mobility and not task_mobility:
        return ["artifact_mobility_metadata_missing_from_task"]
    if step_mobility and dict(task_mobility) != dict(step_mobility):
        return ["artifact_mobility_metadata_mismatch"]
    if not require_artifact_mobility:
        return []
    if not path_maps and artifact_shared_path_rationale is None:
        return ["artifact_mobility_contract_missing"]
    if artifact_shared_path_rationale is not None:
        return []
    blockers: list[str] = []
    for path in _postcondition_artifact_paths(step, repo_root=repo_root):
        if _remote_artifact_path_for_local(path, path_maps=path_maps) is None:
            blockers.append(f"artifact_pullback_missing_for_postcondition:{path.as_posix()}")
    telemetry = step.get("telemetry")
    telemetry_pullbacks = _telemetry_pullback_artifact_paths(step, repo_root=repo_root)
    recursive = (
        isinstance(telemetry, Mapping)
        and (telemetry.get("pullback_recursive") is True or telemetry.get("recursive") is True)
    )
    recursive_cap = (
        telemetry.get("pullback_max_recursive_entries")
        if isinstance(telemetry, Mapping)
        else None
    )
    if recursive_cap is None and isinstance(telemetry, Mapping):
        recursive_cap = telemetry.get("max_recursive_entries")
    if recursive and telemetry_pullbacks and _positive_int(recursive_cap) is None:
        blockers.append("recursive_artifact_pullback_requires_entry_cap")
    if (
        isinstance(telemetry, Mapping)
        and telemetry.get("recursive") is True
        and telemetry.get("pullback_artifact_paths") is None
        and _telemetry_artifact_paths(step, repo_root=repo_root)
    ):
        blockers.append("recursive_telemetry_artifact_paths_are_not_pullback_authority")
    for path in telemetry_pullbacks:
        if _remote_artifact_path_for_local(path, path_maps=path_maps) is None:
            blockers.append(f"artifact_pullback_missing_for_telemetry:{path.as_posix()}")
    return blockers


def build_rsync_pull_command(
    *,
    rsync_binary: str,
    ssh_binary: str = "ssh",
    ssh_target: str,
    remote_path: str,
    local_path: str | Path,
    connect_timeout_seconds: int = 10,
) -> list[str]:
    remote_path = _require_safe_remote_artifact_path(remote_path)
    return [
        rsync_binary,
        "-a",
        "-e",
        _ssh_transport_arg(
            ssh_binary=ssh_binary,
            connect_timeout_seconds=connect_timeout_seconds,
        ),
        "--",
        f"{ssh_target}:{remote_path}",
        str(local_path),
    ]


def _artifact_pullbacks_for_step(
    step: Mapping[str, Any],
    *,
    repo_root: Path,
    path_maps: Sequence[tuple[Path, str]],
) -> list[dict[str, str]]:
    pullbacks: list[dict[str, str]] = []
    for local_path in _artifact_paths_for_pullback(step, repo_root=repo_root):
        remote_path = _remote_artifact_path_for_local(local_path, path_maps=path_maps)
        if remote_path is None:
            continue
        pullbacks.append(
            {
                "local_path": local_path.as_posix(),
                "remote_path": remote_path,
            }
        )
    return pullbacks


def _run_artifact_pullbacks(
    *,
    runner: Runner,
    rsync_binary: str,
    ssh_binary: str,
    ssh_target: str,
    pullbacks: Sequence[Mapping[str, str]],
    timeout_seconds: int,
    connect_timeout_seconds: int,
) -> dict[str, Any]:
    started_all = time.monotonic()
    if not pullbacks:
        return {
            "schema": SSH_ARTIFACT_MOBILITY_SCHEMA,
            "mode": "none",
            "attempted": False,
            "succeeded": True,
            "pullbacks": [],
            "returncode": 0,
            "timed_out": False,
            "execution_error": None,
            "elapsed_seconds": 0.0,
        }
    results: list[dict[str, Any]] = []
    overall_returncode = 0
    timed_out = False
    execution_error: str | None = None
    for pullback in pullbacks:
        local_path = _require_text(pullback.get("local_path"), "pullback.local_path")
        remote_path = _require_text(pullback.get("remote_path"), "pullback.remote_path")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        argv = build_rsync_pull_command(
            rsync_binary=rsync_binary,
            ssh_binary=ssh_binary,
            ssh_target=ssh_target,
            remote_path=remote_path,
            local_path=local_path,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        started = time.monotonic()
        try:
            proc = _run_remote_command(runner, argv, timeout_seconds=timeout_seconds)
            row = {
                "local_path": local_path,
                "remote_path": remote_path,
                "argv": argv,
                "returncode": int(proc.returncode),
                "timed_out": False,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": proc.stdout,
            }
            if proc.returncode != 0 and overall_returncode == 0:
                overall_returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            overall_returncode = 124
            execution_error = f"TimeoutExpired: {exc}"
            row = {
                "local_path": local_path,
                "remote_path": remote_path,
                "argv": argv,
                "returncode": 124,
                "timed_out": True,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": str(exc),
            }
        except OSError as exc:
            overall_returncode = 127
            execution_error = f"{type(exc).__name__}: {exc}"
            row = {
                "local_path": local_path,
                "remote_path": remote_path,
                "argv": argv,
                "returncode": 127,
                "timed_out": False,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": str(exc),
            }
        results.append(row)
        if overall_returncode:
            break
    return {
        "schema": SSH_ARTIFACT_MOBILITY_SCHEMA,
        "mode": "rsync_pull",
        "attempted": True,
        "succeeded": overall_returncode == 0 and not timed_out and execution_error is None,
        "pullbacks": results,
        "returncode": overall_returncode,
        "timed_out": timed_out,
        "execution_error": execution_error,
        "elapsed_seconds": time.monotonic() - started_all,
    }


def _skipped_artifact_mobility(*, mode: str, reason: str) -> dict[str, Any]:
    return {
        "schema": SSH_ARTIFACT_MOBILITY_SCHEMA,
        "mode": mode,
        "attempted": False,
        "succeeded": True,
        "pullbacks": [],
        "returncode": 0,
        "timed_out": False,
        "execution_error": None,
        "skip_reason": reason,
        "elapsed_seconds": 0.0,
    }


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _ready_step_hashes(ready: ReadyStep) -> dict[str, str]:
    return {
        "definition_hash": ready.definition_hash,
        "command_hash": ready.command_hash,
        "postcondition_hash": ready.postcondition_hash,
    }


def _make_worker_run_id(ready: ReadyStep, *, machine_id: str, ssh_target: str) -> str:
    payload = {
        "queue_id": ready.queue_id,
        "experiment_id": ready.experiment_id,
        "step_id": ready.step_id,
        "machine_id": machine_id,
        "ssh_target": ssh_target,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
    }
    return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()[:16]


def _log_path(
    *,
    repo_root: Path,
    log_root: str | Path | None,
    ready: ReadyStep,
    worker_run_id: str,
) -> Path:
    root = Path(log_root) if log_root else repo_root / ".omx" / "state" / "ssh_experiment_queue_logs"
    stamp = _utc_now().replace(":", "").replace("-", "")
    path = root / ready.queue_id / ready.experiment_id / ready.step_id / f"{stamp}_{worker_run_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_dispatch_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    try:
        require_no_truthy_authority_fields(plan, context="staircase_ssh_executor.plan")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    schema = _require_text(plan.get("schema"), "plan.schema")
    if schema != STAIRCASE_DISPATCH_PLAN_SCHEMA:
        raise ExperimentQueueError(
            f"unsupported dispatch plan schema {schema!r}; expected {STAIRCASE_DISPATCH_PLAN_SCHEMA!r}"
        )
    tasks = plan.get("dask_task_specs")
    if not isinstance(tasks, list):
        raise ExperimentQueueError("plan.dask_task_specs must be a list")
    return dict(plan)


def select_ssh_tasks(
    plan: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    ready: Sequence[ReadyStep] = (),
    machine_id: str | None = None,
    allow_future_executor: bool = False,
    remote_repo_roots: Mapping[str, str] | None = None,
    repo_root: str | Path = ".",
    artifact_path_maps: Mapping[str, str] | None = None,
    require_artifact_mobility: bool = False,
    artifact_shared_path_rationale: str | None = None,
) -> list[SshTaskSelection]:
    """Select SSH-executable staircase tasks without mutating queue state."""

    normalized_plan = _normalize_dispatch_plan(plan)
    _require_matching_queue_source_ref(normalized_plan, queue)
    queue_id = str(queue["queue_id"])
    repo = Path(repo_root)
    path_maps = _normalize_path_maps(artifact_path_maps, repo_root=repo)
    ready_by_key = {(step.experiment_id, step.step_id): step for step in ready}
    roots = dict(remote_repo_roots or {})
    selections: list[SshTaskSelection] = []
    for index, raw_task in enumerate(normalized_plan["dask_task_specs"]):
        task = _mapping(raw_task, f"dask_task_specs[{index}]")
        blockers: list[str] = []
        if task.get("queue_id") != queue_id:
            blockers.append("task_queue_id_mismatch")
        blockers.extend(_queue_state_writeback_blockers(task, queue_id=queue_id))
        experiment_id = task.get("experiment_id")
        step_id = task.get("step_id")
        if not isinstance(experiment_id, str) or not isinstance(step_id, str):
            blockers.append("task_identity_missing")
            ready_step = None
        else:
            ready_step = ready_by_key.get((experiment_id, step_id))
            if ready_step is None and ready:
                blockers.append("task_not_ready_in_canonical_queue_state")
            if ready_step is not None:
                try:
                    step = _lookup_step(queue, experiment_id, step_id)
                except ExperimentQueueError as exc:
                    blockers.append(f"queue_step_lookup_failed:{exc}")
                else:
                    blockers.extend(_local_visible_postcondition_blockers(task, step))
                    blockers.extend(
                        _artifact_mobility_blockers(
                            task,
                            step,
                            repo_root=repo,
                            path_maps=path_maps,
                            require_artifact_mobility=require_artifact_mobility,
                            artifact_shared_path_rationale=artifact_shared_path_rationale,
                        )
                    )
        machine = _machine_for_task(task)
        task_machine_id = str(machine.get("id") or task.get("machine_hint") or "")
        if machine_id is not None and task_machine_id != machine_id:
            continue
        executor = str(machine.get("executor") or "")
        if executor not in SSH_EXECUTOR_NAMES:
            if allow_future_executor and executor in FUTURE_SSH_EXECUTOR_NAMES:
                pass
            else:
                blockers.append(f"machine_executor_not_enabled:{executor or 'missing'}")
        if not isinstance(machine.get("ssh_target"), str) or not str(machine.get("ssh_target")).strip():
            blockers.append("machine_ssh_target_missing")
        remote_root = roots.get(task_machine_id) or machine.get("remote_repo_root")
        blockers.extend(_remote_repo_root_blockers(remote_root))
        resources = task.get("resources")
        slots = machine.get("slots")
        if (
            ready_step is not None
            and isinstance(slots, Mapping)
            and int(slots.get(ready_step.resource_kind) or 0) <= 0
        ):
            blockers.append(f"machine_slot_missing:{ready_step.resource_kind}")
        if ready_step is not None and isinstance(resources, Mapping):
            machine_resource_key = f"machine:{task_machine_id}"
            if int(resources.get(machine_resource_key) or 0) <= 0:
                blockers.append("task_machine_resource_missing")
        task_hashes = _task_step_hashes(task)
        if ready_step is not None and task_hashes:
            ready_hashes = _ready_step_hashes(ready_step)
            mismatches = sorted(
                key for key, value in task_hashes.items() if ready_hashes.get(key) != value
            )
            if mismatches:
                blockers.append(f"task_step_hash_mismatch:{','.join(mismatches)}")
        selections.append(SshTaskSelection(task=task, ready_step=ready_step, blockers=tuple(blockers)))
    return selections


def _run_remote_command(
    runner: Runner,
    argv: Sequence[str],
    *,
    timeout_seconds: int | None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = runner(
        list(argv),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(proc.stdout or "")
    return proc


def _run_remote_preflight(
    *,
    runner: Runner,
    ssh_binary: str,
    ssh_target: str,
    connect_timeout_seconds: int,
    remote_repo_root: str,
    expected_head: str,
    require_clean_remote_git: bool,
    dirty_remote_git_rationale: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    remote_script = build_remote_git_preflight_command(
        remote_repo_root=remote_repo_root,
        expected_head=expected_head,
        require_clean=require_clean_remote_git,
    )
    argv = _ssh_argv(
        ssh_binary=ssh_binary,
        ssh_target=ssh_target,
        connect_timeout_seconds=connect_timeout_seconds,
        remote_script=remote_script,
    )
    start = time.monotonic()
    try:
        proc = _run_remote_command(runner, argv, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "returncode": 124,
            "timed_out": True,
            "elapsed_seconds": time.monotonic() - start,
            "command": argv,
            "require_clean_remote_git": require_clean_remote_git,
            "dirty_remote_git_rationale": dirty_remote_git_rationale,
            "stdout": str(exc),
        }
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "timed_out": False,
        "elapsed_seconds": time.monotonic() - start,
        "command": argv,
        "require_clean_remote_git": require_clean_remote_git,
        "dirty_remote_git_rationale": dirty_remote_git_rationale,
        "stdout": proc.stdout,
    }


def _execute_remote_step(
    *,
    runner: Runner,
    ssh_binary: str,
    ssh_target: str,
    connect_timeout_seconds: int,
    remote_repo_root: str,
    expected_head: str,
    require_clean_remote_git: bool,
    ready: ReadyStep,
    step: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    remote_script = build_remote_shell_command(
        remote_repo_root=remote_repo_root,
        command=ready.command,
        expected_head=expected_head,
        require_clean=require_clean_remote_git,
    )
    argv = _ssh_argv(
        ssh_binary=ssh_binary,
        ssh_target=ssh_target,
        connect_timeout_seconds=connect_timeout_seconds,
        remote_script=remote_script,
    )
    timeout_seconds = int(step.get("timeout_seconds") or 0) or None
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("[ssh-experiment-queue] remote command\n")
        handle.write(shlex.join(argv))
        handle.write("\n\n")
    try:
        proc = _run_remote_command(
            runner,
            argv,
            timeout_seconds=timeout_seconds,
            log_path=log_path,
        )
        return {
            "returncode": proc.returncode,
            "timed_out": False,
            "execution_error": None,
            "elapsed_seconds": time.monotonic() - start,
            "ssh_argv": argv,
            "remote_command": remote_script,
        }
    except subprocess.TimeoutExpired as exc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ssh-experiment-queue] timeout: {exc}\n")
        return {
            "returncode": 124,
            "timed_out": True,
            "execution_error": f"TimeoutExpired: {exc}",
            "elapsed_seconds": time.monotonic() - start,
            "ssh_argv": argv,
            "remote_command": remote_script,
        }
    except OSError as exc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ssh-experiment-queue] execution error: {type(exc).__name__}: {exc}\n")
        return {
            "returncode": 127,
            "timed_out": False,
            "execution_error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - start,
            "ssh_argv": argv,
            "remote_command": remote_script,
        }


def _record_blocked_task(
    selection: SshTaskSelection,
    *,
    blockers: Sequence[str] | None = None,
) -> dict[str, Any]:
    merged = list(selection.blockers)
    if blockers:
        merged.extend(str(item) for item in blockers)
    return {
        "executed": False,
        "succeeded": False,
        "blockers": merged,
        "task": selection.to_dict(),
    }


def run_staircase_ssh_executor(
    plan: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    state_path: str | Path,
    repo_root: str | Path,
    execute: bool,
    max_steps: int = 1,
    machine_id: str | None = None,
    remote_repo_roots: Mapping[str, str] | None = None,
    allow_future_executor: bool = False,
    allow_noncanonical_state: bool = False,
    allow_orphaned_state: bool = False,
    require_clean_remote_git: bool = True,
    dirty_remote_git_rationale: str | None = None,
    ssh_binary: str = "ssh",
    ssh_connect_timeout_seconds: int = 10,
    remote_preflight_timeout_seconds: int = 20,
    log_root: str | Path | None = None,
    artifact_path_maps: Mapping[str, str] | None = None,
    require_artifact_mobility: bool = False,
    artifact_shared_path_rationale: str | None = None,
    rsync_binary: str = "rsync",
    artifact_pull_timeout_seconds: int = 300,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Execute selected staircase tasks on SSH machines while local queue state stays authoritative."""

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ExperimentQueueError("max_steps must be a non-negative integer")
    if not require_clean_remote_git:
        dirty_remote_git_rationale = _require_text(
            dirty_remote_git_rationale,
            "dirty_remote_git_rationale",
        )
    if artifact_shared_path_rationale is not None:
        artifact_shared_path_rationale = _require_text(
            artifact_shared_path_rationale,
            "artifact_shared_path_rationale",
        )
    repo = Path(repo_root)
    normalized_artifact_path_maps = _normalize_path_maps(artifact_path_maps, repo_root=repo)
    queue_id = str(queue["queue_id"])
    state = Path(state_path)
    task_results: list[dict[str, Any]] = []
    executed_count = 0
    success_count = 0
    failure_count = 0
    claim_refused_count = 0
    local_head = _local_git_head(repo)

    if not execute:
        if state.is_file():
            with connect_state_readonly(state) as conn:
                ready = ready_steps(conn, queue, allow_cloud=False, repo_root=repo)
            dry_run_state_mode = "read_only_existing_state"
        else:
            with (
                tempfile.TemporaryDirectory(prefix="ssh_executor_dry_run_") as tmp_dir,
                connect_state(Path(tmp_dir) / "state.sqlite") as conn,
            ):
                initialize_queue_state(conn, queue)
                ready = ready_steps(conn, queue, allow_cloud=False, repo_root=repo)
            dry_run_state_mode = "ephemeral_initialized_state"
        selections = select_ssh_tasks(
            plan,
            queue,
            ready=ready,
            machine_id=machine_id,
            allow_future_executor=allow_future_executor,
            remote_repo_roots=remote_repo_roots,
            repo_root=repo,
            artifact_path_maps=artifact_path_maps,
            require_artifact_mobility=require_artifact_mobility,
            artifact_shared_path_rationale=artifact_shared_path_rationale,
        )
        selected = [selection for selection in selections if not selection.blockers]
        blocked = [selection for selection in selections if selection.blockers]
        return apply_proxy_evidence_boundary(
            {
                "schema": SSH_EXECUTION_RESULT_SCHEMA,
                "queue_id": queue_id,
                "state_path": str(state),
                "dry_run_state_mode": dry_run_state_mode,
                "execute": False,
                "execution_mode": "dry_run_read_only",
                "allow_noncanonical_state": allow_noncanonical_state,
                "allow_orphaned_state": allow_orphaned_state,
                "require_clean_remote_git": require_clean_remote_git,
                "dirty_remote_git_rationale": dirty_remote_git_rationale,
                "require_artifact_mobility": require_artifact_mobility,
                "artifact_shared_path_rationale": artifact_shared_path_rationale,
                "artifact_path_map_count": len(normalized_artifact_path_maps),
                "selected_count": min(len(selected), max_steps) if max_steps else len(selected),
                "blocked_count": len(blocked),
                "selected_tasks": [selection.to_dict() for selection in selected[: max_steps or None]],
                "blocked_tasks": [selection.to_dict() for selection in blocked],
                "task_results": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            dispatch_blockers=["ssh_executor_dry_run_only"],
        )

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        assert_canonical_state_for_execution(
            repo,
            queue_id,
            state,
            allow_noncanonical_state=allow_noncanonical_state,
        )
        assert_no_orphaned_steps_for_execution(
            conn,
            queue,
            allow_orphaned_state=allow_orphaned_state,
        )
        ready = ready_steps(conn, queue, allow_cloud=False, repo_root=repo)
        selections = select_ssh_tasks(
            plan,
            queue,
            ready=ready,
            machine_id=machine_id,
            allow_future_executor=allow_future_executor,
            remote_repo_roots=remote_repo_roots,
            repo_root=repo,
            artifact_path_maps=artifact_path_maps,
            require_artifact_mobility=require_artifact_mobility,
            artifact_shared_path_rationale=artifact_shared_path_rationale,
        )
        selected = [selection for selection in selections if not selection.blockers]
        blocked = [selection for selection in selections if selection.blockers]
        for selection in blocked:
            task_results.append(_record_blocked_task(selection))
        for selection in selected:
            if max_steps and executed_count >= max_steps:
                break
            ready_step = selection.ready_step
            if ready_step is None:
                task_results.append(_record_blocked_task(selection, blockers=["ready_step_missing"]))
                continue
            machine = _machine_for_task(selection.task)
            machine_id_value = _require_text(
                machine.get("id") or selection.task.get("machine_hint"),
                "machine.id",
            )
            ssh_target = _require_text(machine.get("ssh_target"), "machine.ssh_target")
            remote_repo_root = _require_text(
                (remote_repo_roots or {}).get(machine_id_value) or machine.get("remote_repo_root"),
                "machine.remote_repo_root",
            )
            remote_root_blockers = _remote_repo_root_blockers(remote_repo_root)
            if remote_root_blockers:
                task_results.append(_record_blocked_task(selection, blockers=remote_root_blockers))
                continue
            preflight = _run_remote_preflight(
                runner=runner,
                ssh_binary=ssh_binary,
                ssh_target=ssh_target,
                connect_timeout_seconds=ssh_connect_timeout_seconds,
                remote_repo_root=remote_repo_root,
                expected_head=local_head,
                require_clean_remote_git=require_clean_remote_git,
                dirty_remote_git_rationale=dirty_remote_git_rationale,
                timeout_seconds=remote_preflight_timeout_seconds,
            )
            if not preflight["passed"]:
                task_results.append(
                    _record_blocked_task(
                        selection,
                        blockers=[f"remote_git_preflight_failed:{preflight['returncode']}"],
                    )
                    | {"remote_preflight": preflight}
                )
                continue

            step = _lookup_step(queue, ready_step.experiment_id, ready_step.step_id)
            worker_run_id = _make_worker_run_id(
                ready_step,
                machine_id=machine_id_value,
                ssh_target=ssh_target,
            )
            log_path = _log_path(
                repo_root=repo,
                log_root=log_root,
                ready=ready_step,
                worker_run_id=worker_run_id,
            )
            running_event = {
                "schema": SSH_EXECUTOR_EVENT_SCHEMA,
                "command": list(ready_step.command),
                "log_path": str(log_path),
                "resource_kind": ready_step.resource_kind,
                "worker_run_id": worker_run_id,
                "executor": "ssh_experiment_queue",
                "machine": dict(machine),
                "ssh_target": ssh_target,
                "remote_repo_root": remote_repo_root,
                "remote_preflight": preflight,
                "dirty_remote_git_rationale": dirty_remote_git_rationale,
                "require_artifact_mobility": require_artifact_mobility,
                "artifact_shared_path_rationale": artifact_shared_path_rationale,
                "timeout_seconds": int(step.get("timeout_seconds") or 0),
            }
            claim_refused_reason = claim_ready_step_for_execution(
                conn,
                queue,
                ready_step,
                event=running_event,
            )
            if claim_refused_reason:
                claim_refused_count += 1
                task_results.append(
                    {
                        "executed": False,
                        "succeeded": False,
                        "claim_refused": True,
                        "claim_refused_reason": claim_refused_reason,
                        "task": selection.to_dict(),
                        **running_event,
                    }
                )
                continue

            executed_count += 1
            remote_plus_pullback_started = time.monotonic()
            remote_result = _execute_remote_step(
                runner=runner,
                ssh_binary=ssh_binary,
                ssh_target=ssh_target,
                connect_timeout_seconds=ssh_connect_timeout_seconds,
                remote_repo_root=remote_repo_root,
                expected_head=local_head,
                require_clean_remote_git=require_clean_remote_git,
                ready=ready_step,
                step=step,
                log_path=log_path,
            )
            terminal_returncode = int(remote_result["returncode"])
            terminal_timed_out = bool(remote_result["timed_out"])
            terminal_error = remote_result["execution_error"]
            if terminal_returncode == 0 and not terminal_timed_out and terminal_error is None:
                pullbacks = _artifact_pullbacks_for_step(
                    step,
                    repo_root=repo,
                    path_maps=normalized_artifact_path_maps,
                )
                artifact_mobility = _run_artifact_pullbacks(
                    runner=runner,
                    rsync_binary=rsync_binary,
                    ssh_binary=ssh_binary,
                    ssh_target=ssh_target,
                    pullbacks=pullbacks,
                    timeout_seconds=artifact_pull_timeout_seconds,
                    connect_timeout_seconds=ssh_connect_timeout_seconds,
                )
            else:
                artifact_mobility = _skipped_artifact_mobility(
                    mode="skipped_remote_execution_failed",
                    reason="remote_command_failed_before_artifact_pullback",
                )
            if not artifact_mobility["succeeded"]:
                terminal_returncode = int(artifact_mobility["returncode"])
                terminal_timed_out = bool(artifact_mobility["timed_out"])
                terminal_error = artifact_mobility["execution_error"] or "artifact_mobility_failed"
            terminal_result = finalize_claimed_step_execution(
                conn,
                queue,
                ready_step,
                repo_root=repo,
                log_path=log_path,
                returncode=terminal_returncode,
                timed_out=terminal_timed_out,
                execution_error=terminal_error,
                elapsed_seconds=time.monotonic() - remote_plus_pullback_started,
                event={
                    **running_event,
                    **remote_result,
                    "remote_elapsed_seconds": float(remote_result["elapsed_seconds"]),
                    "artifact_mobility": artifact_mobility,
                },
            )
            if terminal_result.get("succeeded"):
                success_count += 1
            else:
                failure_count += 1
            task_results.append(
                {
                    "executed": True,
                    "task": selection.to_dict(),
                    **terminal_result,
                }
            )

    return apply_proxy_evidence_boundary(
        {
            "schema": SSH_EXECUTION_RESULT_SCHEMA,
            "queue_id": queue_id,
            "state_path": str(state),
            "execute": execute,
            "execution_mode": "serial_ssh_executor",
            "allow_noncanonical_state": allow_noncanonical_state,
            "allow_orphaned_state": allow_orphaned_state,
            "require_clean_remote_git": require_clean_remote_git,
            "dirty_remote_git_rationale": dirty_remote_git_rationale,
            "require_artifact_mobility": require_artifact_mobility,
            "artifact_shared_path_rationale": artifact_shared_path_rationale,
            "artifact_path_map_count": len(normalized_artifact_path_maps),
            "selected_count": len([result for result in task_results if not result.get("blockers")]),
            "blocked_count": len([result for result in task_results if result.get("blockers")]),
            "executed_count": executed_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "claim_refused_count": claim_refused_count,
            "task_results": task_results,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        dispatch_blockers=["ssh_executor_result_is_not_score_authority"],
    )


__all__ = [
    "FUTURE_SSH_EXECUTOR_NAMES",
    "SSH_ARTIFACT_MOBILITY_SCHEMA",
    "SSH_EXECUTION_RESULT_SCHEMA",
    "SSH_EXECUTOR_EVENT_SCHEMA",
    "SSH_EXECUTOR_NAMES",
    "SshTaskSelection",
    "build_remote_git_preflight_command",
    "build_remote_shell_command",
    "build_rsync_pull_command",
    "run_staircase_ssh_executor",
    "select_ssh_tasks",
]
