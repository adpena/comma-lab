from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import (
    QUEUE_SCHEMA,
    ExperimentQueueError,
    default_state_path,
    load_queue_definition,
)
from comma_lab.scheduler.experiment_queue_observer import observe_experiment_queue

QUEUE_FLEET_STATUS_SCHEMA = "experiment_queue_fleet_status.v1"
QUEUE_FLEET_ROW_SCHEMA = "experiment_queue_fleet_row.v1"
QUEUE_FLEET_NATIVE_CONSUMER_HINT_SCHEMA = "experiment_queue_fleet_native_consumer_hint.v1"
NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS = "NON_EXECUTABLE_QUEUE_ARTIFACT"
PAUSED_EXACT_DISPATCH_GATE_STATUS = "PAUSED_EXACT_DISPATCH_GATE"
MATERIALIZER_WORK_QUEUE_SCHEMA = "byte_shaving_materializer_work_queue.v1"
FRONTIER_FINAL_RATE_ATTACK_CHILD_RUNS_SCHEMA = (
    "frontier_final_rate_attack_post_feedback_child_queue_runs.v1"
)
EXPERIMENT_QUEUE_VALIDATION_REPORT_SCHEMA = "experiment_queue_validation_report.v1"
OPTIMIZER_CANDIDATE_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
OPTIMIZER_CANDIDATE_EXACT_READY_QUEUE_SCHEMA = (
    "optimizer_candidate_exact_eval_ready_queue_v1"
)
REPORT_ONLY_NATIVE_CONSUMERS = {
    "experiment_queue_observation.v1": (
        "experiment_queue_observation_report",
        "read_as_queue_observation_report_not_experiment_queue",
    ),
    "experiment_queue_performance_summary.v1": (
        "experiment_queue_performance_report",
        "read_as_queue_performance_report_not_experiment_queue",
    ),
    "experiment_queue_fleet_status.v1": (
        "experiment_queue_fleet_status_report",
        "read_as_queue_fleet_status_report_not_experiment_queue",
    ),
    "experiment_queue_summary.v1": (
        "experiment_queue_summary_report",
        "read_as_queue_summary_report_not_experiment_queue",
    ),
}
EXACT_DISPATCH_QUEUE_ID_MARKERS = ("exact_eval_dispatch", "blocked_exact_eval_dispatch")
EXACT_DISPATCH_STEP_ID_MARKERS = ("dispatch_exact_eval",)
EXACT_DISPATCH_COMMAND_MARKERS = (
    "tools/parallel_dispatch_top_k.py",
    "parallel_dispatch_top_k.py",
    "tools/claim_lane_dispatch.py",
    "claim_lane_dispatch.py",
)

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


def _raw_json_metadata(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    out: dict[str, Any] = {}
    schema = payload.get("schema")
    if isinstance(schema, str) and schema.strip():
        out["artifact_schema"] = schema
    if schema == MATERIALIZER_WORK_QUEUE_SCHEMA:
        rows = payload.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            rows = []
        materializer_rows = [row for row in rows if isinstance(row, Mapping)]
        executable_rows = [row for row in materializer_rows if row.get("executable") is True]
        out["materializer_work_row_count"] = len(materializer_rows)
        out["materializer_executable_row_count"] = len(executable_rows)
    if (
        "experiments" not in payload
        and isinstance(payload.get("queue_id"), str)
        and isinstance(payload.get("valid"), bool)
        and "experiment_count" in payload
        and "step_count" in payload
    ):
        out["artifact_schema"] = out.get("artifact_schema") or "experiment_queue_validation_report.v1"
        out["recommended_action"] = "use_as_validation_report_not_experiment_queue_supervisor"
    if not out:
        return None
    return out


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


def _slug(text: str, *, limit: int = 48) -> str:
    out = []
    previous_underscore = False
    for char in text.lower():
        if char.isalnum():
            out.append(char)
            previous_underscore = False
        elif not previous_underscore:
            out.append("_")
            previous_underscore = True
    slug = "".join(out).strip("_")
    return (slug or "queue")[:limit].strip("_") or "queue"


def _materializer_execution_queue_ref(path: Path) -> Path:
    stem = path.stem
    stem = (
        stem.replace("work_queue", "execution_queue", 1)
        if "work_queue" in stem
        else f"{stem}_execution_queue"
    )
    return path.with_name(f"{stem}.json")


def _submission_closure_dir(path: Path) -> Path:
    return path.parent / f"{path.stem}_submission_closure"


def _exact_eval_consumer_queue_ref(path: Path) -> Path:
    stem = path.stem
    stem = (
        stem.replace("exact_ready_queue", "exact_eval_consumer_queue", 1)
        if "exact_ready_queue" in stem
        else f"{stem}_exact_eval_consumer_queue"
    )
    return path.with_name(f"{stem}.json")


def _native_consumer_hint(
    repo_root: str | Path,
    path: Path,
    artifact_schema: str,
    artifact_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root)
    metadata = artifact_metadata if isinstance(artifact_metadata, Mapping) else {}
    base: dict[str, Any] = {
        "schema": QUEUE_FLEET_NATIVE_CONSUMER_HINT_SCHEMA,
        "artifact_schema": artifact_schema,
        "artifact_path": repo_rel(path, repo),
        "known_native_consumer": False,
        "consumer_kind": "unknown",
        "recommended_action": "route_to_native_consumer_not_experiment_queue_supervisor",
        "allowed_use": "local_artifact_routing_hint_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_exact_eval_authority",
        **FALSE_AUTHORITY,
    }
    if artifact_schema == MATERIALIZER_WORK_QUEUE_SCHEMA:
        executable_count = metadata.get("materializer_executable_row_count")
        if executable_count == 0:
            return {
                **base,
                "known_native_consumer": True,
                "consumer_kind": "byte_shaving_materializer_work_queue",
                "recommended_action": "preserve_blocked_materializer_work_queue_no_executable_rows",
                "ready_for_native_consumer": False,
                "blockers": ["materializer_work_queue_has_no_executable_rows"],
                "materializer_work_row_count": metadata.get("materializer_work_row_count", 0),
                "materializer_executable_row_count": 0,
            }
        output_path = _materializer_execution_queue_ref(path)
        digest = hashlib.sha256(repo_rel(path, repo).encode("utf-8")).hexdigest()[:10]
        queue_id = f"materializer_exec_{_slug(path.stem, limit=38)}_{digest}"
        if output_path.exists():
            return {
                **base,
                "known_native_consumer": True,
                "consumer_kind": "byte_shaving_materializer_work_queue",
                "recommended_action": "use_existing_materializer_execution_queue",
                "ready_for_native_consumer": False,
                "output_queue_path": repo_rel(output_path, repo),
                "output_queue_schema": QUEUE_SCHEMA,
                "output_queue_id": queue_id,
                "output_queue_exists": True,
                "materializer_work_row_count": metadata.get("materializer_work_row_count"),
                "materializer_executable_row_count": executable_count,
            }
        command = [
            ".venv/bin/python",
            "tools/build_materializer_execution_queue.py",
            "--work-queue",
            repo_rel(path, repo),
            "--queue-out",
            repo_rel(output_path, repo),
            "--queue-id",
            queue_id,
            "--local-cpu-concurrency",
            "auto",
        ]
        return {
            **base,
            "known_native_consumer": True,
            "consumer_kind": "byte_shaving_materializer_work_queue",
            "recommended_action": (
                "build_materializer_execution_queue_then_supervise_with_experiment_queue"
            ),
            "native_consumer_command": command,
            "output_queue_path": repo_rel(output_path, repo),
            "output_queue_schema": QUEUE_SCHEMA,
            "output_queue_id": queue_id,
            "ready_for_native_consumer": True,
            "materializer_work_row_count": metadata.get("materializer_work_row_count"),
            "materializer_executable_row_count": executable_count,
        }
    if artifact_schema == OPTIMIZER_CANDIDATE_QUEUE_SCHEMA:
        if path.name == "closed_source_queue.json" or "submission_closure" in path.parts:
            return {
                **base,
                "known_native_consumer": True,
                "consumer_kind": "materializer_submission_closed_source_queue",
                "recommended_action": "read_as_closed_submission_source_queue",
            }
        output_dir = _submission_closure_dir(path)
        command = [
            ".venv/bin/python",
            "tools/build_materializer_submission_closure.py",
            "--source-queue",
            repo_rel(path, repo),
            "--submission-dir-out",
            repo_rel(output_dir / "submission", repo),
            "--closed-source-queue-out",
            repo_rel(output_dir / "closed_source_queue.json", repo),
            "--closure-report-out",
            repo_rel(output_dir / "submission_closure_report.json", repo),
        ]
        return {
            **base,
            "known_native_consumer": True,
            "consumer_kind": "optimizer_candidate_submission_closure",
            "recommended_action": "build_submission_runtime_closure_before_exact_readiness",
            "native_consumer_command": command,
            "output_report_path": repo_rel(output_dir / "submission_closure_report.json", repo),
            "closed_source_queue_path": repo_rel(output_dir / "closed_source_queue.json", repo),
        }
    if artifact_schema == OPTIMIZER_CANDIDATE_EXACT_READY_QUEUE_SCHEMA:
        output_path = _exact_eval_consumer_queue_ref(path)
        output_report = output_path.with_suffix(".consumer_report.json")
        digest = hashlib.sha256(repo_rel(path, repo).encode("utf-8")).hexdigest()[:10]
        queue_id = f"exact_eval_consumer_{_slug(path.stem, limit=36)}_{digest}"
        command = [
            ".venv/bin/python",
            "tools/build_materializer_exact_eval_consumer.py",
            "--exact-ready-queue",
            repo_rel(path, repo),
            "--consumer-report-out",
            repo_rel(output_report, repo),
            "--experiment-queue-out",
            repo_rel(output_path, repo),
            "--queue-id",
            queue_id,
        ]
        return {
            **base,
            "known_native_consumer": True,
            "consumer_kind": "optimizer_candidate_exact_eval_consumer",
            "recommended_action": "build_paused_exact_eval_consumer_queue",
            "native_consumer_command": command,
            "output_queue_path": repo_rel(output_path, repo),
            "output_queue_schema": QUEUE_SCHEMA,
            "output_queue_id": queue_id,
            "output_report_path": repo_rel(output_report, repo),
        }
    if artifact_schema == FRONTIER_FINAL_RATE_ATTACK_CHILD_RUNS_SCHEMA:
        return {
            **base,
            "known_native_consumer": True,
            "consumer_kind": "frontier_final_rate_attack_child_run_manifest",
            "recommended_action": "read_as_child_queue_run_manifest_not_experiment_queue",
        }
    if artifact_schema == EXPERIMENT_QUEUE_VALIDATION_REPORT_SCHEMA:
        return {
            **base,
            "known_native_consumer": True,
            "consumer_kind": "experiment_queue_validation_report",
            "recommended_action": "use_as_validation_report_not_experiment_queue_supervisor",
        }
    if artifact_schema in REPORT_ONLY_NATIVE_CONSUMERS:
        consumer_kind, recommended_action = REPORT_ONLY_NATIVE_CONSUMERS[artifact_schema]
        return {
            **base,
            "known_native_consumer": True,
            "consumer_kind": consumer_kind,
            "recommended_action": recommended_action,
        }
    return base


def _supervisor_output_dir(
    repo_root: str | Path,
    output_root: str | Path,
    queue_id: str,
) -> str:
    digest = hashlib.sha256(queue_id.encode("utf-8")).hexdigest()[:10]
    return repo_rel(Path(output_root) / f"{queue_id}_{digest}", repo_root)


def _queue_is_exact_dispatch_gate(queue: Mapping[str, Any], queue_path: str | Path | None = None) -> bool:
    queue_id = str(queue.get("queue_id") or "").lower()
    path_text = str(queue_path or "").lower()
    if any(marker in queue_id for marker in EXACT_DISPATCH_QUEUE_ID_MARKERS):
        return True
    if "exact_eval" in path_text and "dispatch" in path_text:
        return True
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        experiment_id = str(experiment.get("id") or "").lower()
        if any(marker in experiment_id for marker in EXACT_DISPATCH_QUEUE_ID_MARKERS):
            return True
        steps = experiment.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            step_id = str(step.get("id") or "").lower()
            if any(marker in step_id for marker in EXACT_DISPATCH_STEP_ID_MARKERS):
                return True
            command = step.get("command")
            if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
                command_text = " ".join(str(part) for part in command).lower()
                if any(marker in command_text for marker in EXACT_DISPATCH_COMMAND_MARKERS):
                    return True
    return False


def classify_queue_observation(
    observation: Mapping[str, Any],
    *,
    queue: Mapping[str, Any] | None = None,
    queue_path: str | Path | None = None,
) -> tuple[str, list[str]]:
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
        if queue is not None and _queue_is_exact_dispatch_gate(queue, queue_path):
            return PAUSED_EXACT_DISPATCH_GATE_STATUS, [
                f"mode={mode or 'unknown'}",
                "exact_dispatch_gate_paused",
                "mlx_first_no_auto_cloud_dispatch",
            ]
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
        PAUSED_EXACT_DISPATCH_GATE_STATUS: 75,
        "EMPTY_OR_IDLE": 80,
        "INVALID_QUEUE": 90,
        NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS: 95,
    }.get(status, 999)


def _compact_row_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "status": row.get("status"),
        "queue_path": row.get("queue_path"),
    }
    for key in (
        "queue_id",
        "artifact_schema",
        "state",
        "state_exists",
        "blockers",
        "status_counts",
        "recommended_action",
        "ignored_for_supervision",
        "native_consumer",
        "native_consumer_command",
        "conflict_status_before",
        "identity_conflict",
    ):
        value = row.get(key)
        if value not in (None, {}, [], ""):
            sample[key] = value
    return sample


def _samples_by_status(rows: Iterable[Mapping[str, Any]], *, per_status: int = 5) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        bucket = samples.setdefault(status, [])
        if len(bucket) < per_status:
            bucket.append(_compact_row_sample(row))
    return dict(sorted(samples.items()))


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
        artifact_metadata = _raw_json_metadata(path) or {}
        artifact_schema = artifact_metadata.get("artifact_schema")
        if artifact_schema and artifact_schema != QUEUE_SCHEMA:
            status = NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS
            native_consumer = _native_consumer_hint(
                repo,
                path,
                str(artifact_schema),
                artifact_metadata=artifact_metadata,
            )
            recommended_action = artifact_metadata.get("recommended_action") or native_consumer.get(
                "recommended_action"
            )
            return {
                **base,
                "status": status,
                "priority": _priority(status),
                "artifact_schema": artifact_schema,
                "ignored_for_supervision": True,
                "recommended_action": recommended_action,
                "native_consumer": native_consumer,
                "native_consumer_command": native_consumer.get("native_consumer_command"),
                "blockers": [],
            }
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
        status, blockers = classify_queue_observation(observation, queue=queue, queue_path=path)
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
        "resume_command": _queue_command(
            repo,
            "tools/experiment_queue.py",
            path,
            state_path,
            "control",
            "running",
            "--reason",
            "queue_fleet_resume_paused_with_queued_work",
        ),
    }


def _prepend_unique(existing: Iterable[Any], additions: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in [*additions, *[str(item) for item in existing if str(item)]]:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def annotate_identity_conflicts(rows: Sequence[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        queue_id = row.get("queue_id")
        if not isinstance(queue_id, str) or not queue_id.strip():
            continue
        if row.get("status") in {"INVALID_QUEUE", NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS}:
            continue
        groups.setdefault(queue_id, []).append(row)
    for queue_id, group in groups.items():
        queue_paths = sorted({str(row.get("queue_path") or "") for row in group if row.get("queue_path")})
        if len(queue_paths) <= 1:
            continue
        states = sorted({str(row.get("state") or "") for row in group if row.get("state")})
        blockers = [f"experiment_queue_fleet_duplicate_queue_id:{queue_id}:paths={len(queue_paths)}"]
        if len(states) == 1 and states[0]:
            blockers.append(f"experiment_queue_fleet_shared_state:{states[0]}:paths={len(queue_paths)}")
        conflict = {
            "queue_id": queue_id,
            "queue_path_count": len(queue_paths),
            "queue_paths": queue_paths,
            "state_count": len(states),
            "states": states,
        }
        for row in group:
            row["conflict_status_before"] = row.get("status")
            row["identity_conflict"] = conflict
            row["status"] = "NEEDS_RECOVERY"
            row["priority"] = _priority("NEEDS_RECOVERY")
            row["recommended_action"] = "split_or_migrate_duplicate_queue_id_before_supervision"
            row["blockers"] = _prepend_unique(row.get("blockers") or [], blockers)
            row["blocker_count"] = len(row["blockers"])


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
    annotate_identity_conflicts(rows)
    rows.sort(key=lambda row: (int(row.get("priority") or 999), str(row.get("queue_path") or "")))
    counts = Counter(str(row.get("status") or "UNKNOWN") for row in rows)
    actionable = [
        row
        for row in rows
        if row.get("status")
        in {"NEEDS_RECOVERY", "NEEDS_INIT", "READY_TO_SUPERVISE", "RUNNING", "PAUSED_WITH_QUEUED_WORK"}
    ]
    visible_rows = rows[:row_limit] if row_limit is not None else rows
    recovery_statuses = {"NEEDS_RECOVERY", "NEEDS_INIT"}
    native_consumer_rows = [
        row
        for row in rows
        if row.get("status") == NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS
        and isinstance(row.get("native_consumer"), Mapping)
    ]
    known_native_consumer_rows = [
        row
        for row in native_consumer_rows
        if isinstance(row.get("native_consumer"), Mapping)
        and row["native_consumer"].get("known_native_consumer") is True
    ]
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
        "status_samples": _samples_by_status(rows),
        "actionable_count": len(actionable),
        "ready_to_supervise_count": counts.get("READY_TO_SUPERVISE", 0),
        "paused_with_queued_work_count": counts.get("PAUSED_WITH_QUEUED_WORK", 0),
        "paused_exact_dispatch_gate_count": counts.get(PAUSED_EXACT_DISPATCH_GATE_STATUS, 0),
        "needs_recovery_count": counts.get("NEEDS_RECOVERY", 0) + counts.get("NEEDS_INIT", 0),
        "invalid_queue_count": counts.get("INVALID_QUEUE", 0),
        "non_executable_artifact_count": counts.get(NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS, 0),
        "native_consumer_artifact_count": len(native_consumer_rows),
        "known_native_consumer_artifact_count": len(known_native_consumer_rows),
        "row_count": len(visible_rows),
        "rows": visible_rows,
        "next_supervise_commands": [
            row["supervise_command"]
            for row in rows
            if row.get("status") == "READY_TO_SUPERVISE"
        ][:8],
        "next_recovery_commands": [
            row["status_command"]
            for row in rows
            if row.get("status") in recovery_statuses and row.get("status_command")
        ][:8],
        "next_resume_commands": [
            row["resume_command"]
            for row in rows
            if row.get("status") == "PAUSED_WITH_QUEUED_WORK"
            and row.get("resume_command")
        ][:8],
        "next_init_commands": [
            _queue_command(
                repo,
                "tools/experiment_queue.py",
                row["queue_path"],
                row["state"],
                "init",
            )
            for row in rows
            if row.get("status") == "NEEDS_INIT" and row.get("state")
        ][:8],
        "next_native_consumer_commands": [
            row["native_consumer_command"]
            for row in rows
            if row.get("status") == NON_EXECUTABLE_QUEUE_ARTIFACT_STATUS
            and row.get("native_consumer_command")
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
