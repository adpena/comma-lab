from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    default_state_path,
    load_queue_definition,
)
from comma_lab.scheduler.experiment_queue_observer import observe_experiment_queue

QUEUE_FLEET_STATUS_SCHEMA = "experiment_queue_fleet_status.v1"
QUEUE_FLEET_ROW_SCHEMA = "experiment_queue_fleet_row.v1"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

DEFAULT_QUEUE_ROOTS = (".omx/research", "experiments/results", "configs/experiment_queues")
QUEUE_FILENAME_TOKENS = ("queue",)
QUEUE_FILE_SUFFIXES = {".json", ".yaml", ".yml"}
MAX_DEFAULT_DEPTH = 5


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_rel(path: str | Path, repo_root: str | Path) -> str:
    p = Path(path)
    root = Path(repo_root)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def state_path_for_queue(
    repo_root: str | Path,
    queue_id: str,
    *,
    state_root: str | Path | None = None,
) -> Path:
    if state_root is None:
        return default_state_path(repo_root, queue_id)
    return Path(state_root) / f"experiment_queue_{queue_id}.sqlite"


def _depth_from_root(root: Path, path: Path) -> int:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return 999_999
    return max(0, len(rel.parts) - 1)


def _candidate_name(path: Path) -> bool:
    if path.suffix.lower() not in QUEUE_FILE_SUFFIXES:
        return False
    name = path.name.lower()
    return any(token in name for token in QUEUE_FILENAME_TOKENS)


def _stable_path_order_key(path: Path) -> tuple[int, str]:
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return (-int(mtime), path.as_posix())


def discover_queue_paths(
    repo_root: str | Path,
    roots: Sequence[str | Path] | None = None,
    *,
    max_depth: int = MAX_DEFAULT_DEPTH,
    limit: int | None = None,
) -> list[Path]:
    repo = Path(repo_root)
    root_refs = roots if roots is not None else DEFAULT_QUEUE_ROOTS
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root_ref in root_refs:
        root = Path(root_ref)
        if not root.is_absolute():
            root = repo / root
        if not root.exists():
            continue
        if root.is_file():
            if _candidate_name(root):
                resolved = root.resolve(strict=False)
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if _depth_from_root(root, current) >= max_depth:
                dirnames[:] = []
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname not in {".git", "__pycache__"} and not dirname.endswith(".egg-info")
            ]
            for filename in filenames:
                path = current / filename
                if not _candidate_name(path):
                    continue
                resolved = path.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(path)
    candidates.sort(key=_stable_path_order_key)
    return candidates[:limit] if limit is not None else candidates


def _count(observation: Mapping[str, Any], status: str) -> int:
    counts = observation.get("status_counts")
    if not isinstance(counts, Mapping):
        return 0
    try:
        return int(counts.get(status) or 0)
    except (TypeError, ValueError):
        return 0


def _queue_step_count(queue: Mapping[str, Any]) -> int:
    total = 0
    for experiment in queue.get("experiments") or []:
        if isinstance(experiment, Mapping):
            steps = experiment.get("steps")
            if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes, bytearray)):
                total += len(steps)
    return total


def _queue_command(
    repo_root: str | Path,
    tool: str,
    queue_path: str | Path,
    state_path: str | Path,
    *extra: str,
) -> list[str]:
    return [
        ".venv/bin/python",
        tool,
        "--queue",
        repo_rel(queue_path, repo_root),
        "--state",
        repo_rel(state_path, repo_root),
        *extra,
    ]


def _supervisor_output_dir(
    repo_root: str | Path,
    output_root: str | Path,
    queue_id: str,
) -> str:
    digest = hashlib.sha256(queue_id.encode("utf-8")).hexdigest()[:10]
    return repo_rel(Path(output_root) / f"{queue_id}_{digest}", repo_root)


def classify_queue_observation(observation: Mapping[str, Any]) -> tuple[str, list[str]]:
    blockers = [str(item) for item in observation.get("blockers") or [] if str(item)]
    state_watermark = observation.get("state_watermark")
    state_watermark = state_watermark if isinstance(state_watermark, Mapping) else {}
    if state_watermark.get("state_missing") is True:
        return "NEEDS_INIT", ["queue_state_missing"]
    failed = _count(observation, "failed")
    blocked = _count(observation, "blocked")
    running = _count(observation, "running")
    queued = _count(observation, "queued")
    succeeded = _count(observation, "succeeded")
    orphaned = int(observation.get("orphaned_step_count") or 0)
    if observation.get("healthy") is not True:
        return "NEEDS_RECOVERY", blockers or ["queue_observation_unhealthy"]
    if failed or blocked:
        return "NEEDS_REVIEW", [f"failed={failed}", f"blocked={blocked}"]
    if orphaned:
        return "NEEDS_RECOVERY", [f"orphaned_steps={orphaned}"]
    mode = str(observation.get("mode") or "")
    if running:
        return "RUNNING", []
    if queued and mode == "running":
        return "READY_TO_SUPERVISE", []
    if queued:
        return "PAUSED_WITH_QUEUED_WORK", [f"mode={mode or 'unknown'}"]
    if succeeded:
        return "TERMINAL", []
    return "EMPTY_OR_IDLE", []


def _priority(status: str) -> int:
    return {
        "NEEDS_RECOVERY": 10,
        "NEEDS_INIT": 20,
        "READY_TO_SUPERVISE": 30,
        "RUNNING": 40,
        "PAUSED_WITH_QUEUED_WORK": 50,
        "NEEDS_REVIEW": 60,
        "TERMINAL": 70,
        "EMPTY_OR_IDLE": 80,
        "INVALID_QUEUE": 90,
    }.get(status, 999)


def queue_fleet_row(
    repo_root: str | Path,
    queue_path: str | Path,
    *,
    state_root: str | Path | None = None,
    tail_lines: int = 0,
    include_orphans: bool = True,
    supervisor_output_root: str | Path = ".omx/research/queue_fleet_supervisor",
) -> dict[str, Any]:
    repo = Path(repo_root)
    path = Path(queue_path)
    base = {
        "schema": QUEUE_FLEET_ROW_SCHEMA,
        "queue_path": repo_rel(path, repo),
        "observed_at_utc": utc_now(),
        **FALSE_AUTHORITY,
    }
    try:
        queue = load_queue_definition(path)
    except Exception as exc:
        return {
            **base,
            "status": "INVALID_QUEUE",
            "priority": _priority("INVALID_QUEUE"),
            "blockers": [f"load_queue_definition_failed:{type(exc).__name__}:{exc}"],
        }
    queue_id = str(queue["queue_id"])
    state_path = state_path_for_queue(repo, queue_id, state_root=state_root)
    try:
        observation = observe_experiment_queue(
            queue,
            state_path=state_path,
            repo_root=repo,
            tail_lines=tail_lines,
            include_orphans=include_orphans,
        )
    except Exception as exc:
        status = "NEEDS_RECOVERY"
        blockers = [f"observe_experiment_queue_failed:{type(exc).__name__}:{exc}"]
        observation = {
            "queue_id": queue_id,
            "healthy": False,
            "status_counts": {},
            "blockers": blockers,
        }
    else:
        status, blockers = classify_queue_observation(observation)
    supervise_dir = _supervisor_output_dir(repo, supervisor_output_root, queue_id)
    return {
        **base,
        "queue_id": queue_id,
        "state": repo_rel(state_path, repo),
        "state_exists": state_path.is_file(),
        "status": status,
        "priority": _priority(status),
        "mode": observation.get("mode"),
        "healthy": observation.get("healthy"),
        "status_counts": observation.get("status_counts", {}),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "orphaned_step_count": observation.get("orphaned_step_count", 0),
        "experiment_count": len(queue.get("experiments") or []),
        "step_count": _queue_step_count(queue),
        "auto_parallelism": observation.get("auto_parallelism", {}),
        "runtime_policy": observation.get("runtime_policy", {}),
        "supervisor_output_dir": supervise_dir,
        "status_command": _queue_command(
            repo,
            "tools/queue_control.py",
            path,
            state_path,
            "status",
            "--strict",
        ),
        "supervise_command": _queue_command(
            repo,
            "tools/queue_supervisor.py",
            path,
            state_path,
            "--output-dir",
            supervise_dir,
            "--execute",
        ),
    }


def queue_fleet_status(
    repo_root: str | Path,
    roots: Sequence[str | Path] | None = None,
    *,
    state_root: str | Path | None = None,
    max_depth: int = MAX_DEFAULT_DEPTH,
    limit: int = 80,
    row_limit: int | None = None,
    tail_lines: int = 0,
    include_orphans: bool = True,
    supervisor_output_root: str | Path = ".omx/research/queue_fleet_supervisor",
) -> dict[str, Any]:
    repo = Path(repo_root)
    paths = discover_queue_paths(repo, roots, max_depth=max_depth, limit=limit)
    rows = [
        queue_fleet_row(
            repo,
            path,
            state_root=state_root,
            tail_lines=tail_lines,
            include_orphans=include_orphans,
            supervisor_output_root=supervisor_output_root,
        )
        for path in paths
    ]
    rows.sort(key=lambda row: (int(row.get("priority") or 999), str(row.get("queue_path") or "")))
    counts = Counter(str(row.get("status") or "UNKNOWN") for row in rows)
    actionable = [
        row
        for row in rows
        if row.get("status")
        in {"NEEDS_RECOVERY", "NEEDS_INIT", "READY_TO_SUPERVISE", "RUNNING", "PAUSED_WITH_QUEUED_WORK"}
    ]
    visible_rows = rows[:row_limit] if row_limit is not None else rows
    return {
        "schema": QUEUE_FLEET_STATUS_SCHEMA,
        "generated_at_utc": utc_now(),
        "repo_root": str(repo),
        "scan_roots": [repo_rel(Path(root) if Path(root).is_absolute() else repo / root, repo) for root in (roots or DEFAULT_QUEUE_ROOTS)],
        "state_root": repo_rel(state_root, repo) if state_root is not None else "",
        "max_depth": max_depth,
        "candidate_path_count": len(paths),
        "queue_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "actionable_count": len(actionable),
        "ready_to_supervise_count": counts.get("READY_TO_SUPERVISE", 0),
        "needs_recovery_count": counts.get("NEEDS_RECOVERY", 0) + counts.get("NEEDS_INIT", 0),
        "row_count": len(visible_rows),
        "rows": visible_rows,
        "next_supervise_commands": [
            row["supervise_command"]
            for row in rows
            if row.get("status") == "READY_TO_SUPERVISE"
        ][:8],
        "allowed_use": "queue_fleet_local_telemetry_and_bounded_supervision_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }


def select_supervision_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_recovery: bool = False,
    max_queues: int = 4,
) -> list[Mapping[str, Any]]:
    allowed = {"READY_TO_SUPERVISE"}
    if include_recovery:
        allowed.update({"NEEDS_RECOVERY", "NEEDS_INIT"})
    selected = [row for row in rows if row.get("status") in allowed]
    selected.sort(key=lambda row: (int(row.get("priority") or 999), str(row.get("queue_path") or "")))
    if max_queues <= 0:
        raise ExperimentQueueError("max_queues must be positive")
    return selected[:max_queues]
