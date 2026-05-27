from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from posixpath import normpath
from typing import Any

from tac.optimization.runtime_adapter_identity import runtime_adapter_identity_blockers

QUEUE_SCHEMA = "experiment_queue.v1"
STATE_SCHEMA = "experiment_queue_state.v1"
SCHEDULER_RUNTIME_POLICY_SCHEMA = "scheduler_runtime_policy.v1"

CONTROL_MODES = {"running", "paused", "frozen"}
STEP_STATUSES = {"queued", "running", "succeeded", "failed", "blocked", "skipped"}
BLOCKING_ORPHAN_STATUSES = {"queued", "running", "blocked"}
LOCAL_RESOURCE_KINDS = {
    "local_cpu",
    "local_cuda",
    "local_io_heavy",
    "local_mlx",
    "local_mps",
    "local",
}
CLOUD_RESOURCE_KINDS = {"cloud_cpu", "cloud_gpu", "modal_cpu", "modal_gpu", "cuda_auth"}
KNOWN_RESOURCE_KINDS = LOCAL_RESOURCE_KINDS | CLOUD_RESOURCE_KINDS
SHA256_HEX = frozenset("0123456789abcdef")
LOG_PATH_COMPONENT_MAX_CHARS = 96
LOG_PATH_COMPONENT_HASH_CHARS = 16
LOG_PATH_COMPONENT_ALLOWED_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
)
DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS = (
    "score_claim_valid",
    "score_claim_eligible",
    "promotable",
    "ready_for_exact_eval_dispatch",
    "field_selection_ready_for_exact_eval_dispatch",
    "dispatch_ready",
    "exact_eval_ready",
    "exact_eval_dispatch_ready",
    "exact_cuda_auth_eval",
    "contest_cuda_auth_eval",
    "dispatch_attempted",
    "dispatch_packet_ready",
    "gpu_launched",
    "score_affecting_payload_changed",
    "charged_bits_changed",
)
SCHEDULER_RUNTIME_POLICY_FORBIDDEN_AUTHORITY_FIELDS = tuple(
    dict.fromkeys(
        (
            *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
            *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
        )
    )
)


class ExperimentQueueError(ValueError):
    """Raised when experiment queue definitions or transitions are invalid."""


@dataclass(frozen=True)
class ReadyStep:
    queue_id: str
    experiment_id: str
    step_id: str
    priority: int
    resource_kind: str
    command: tuple[str, ...]
    definition_hash: str
    command_hash: str
    postcondition_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id,
            "experiment_id": self.experiment_id,
            "step_id": self.step_id,
            "priority": self.priority,
            "resource_kind": self.resource_kind,
            "command": list(self.command),
            "step_hashes": {
                "definition_hash": self.definition_hash,
                "command_hash": self.command_hash,
                "postcondition_hash": self.postcondition_hash,
            },
        }


@dataclass
class RunningStepProcess:
    ready_step: ReadyStep
    step: dict[str, Any]
    process: subprocess.Popen[bytes]
    log_handle: Any
    log_path: Path
    started_monotonic: float
    timeout_seconds: int
    worker_run_id: str
    terminate_requested_monotonic: float | None = None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentQueueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExperimentQueueError(f"{label} must be an object")
    return dict(value)


def _string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ExperimentQueueError(f"{label} must be a list")
    return [_require_text(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _resolve_step_ref(ref: str, *, default_experiment_id: str) -> tuple[str, str]:
    """Resolve a queue dependency ref to ``(experiment_id, step_id)``."""

    text = _require_text(ref, "step reference")
    if "." not in text:
        return default_experiment_id, text
    experiment_id, step_id = text.split(".", 1)
    return (
        _require_text(experiment_id, "step reference experiment_id"),
        _require_text(step_id, "step reference step_id"),
    )


def _format_step_ref(experiment_id: str, step_id: str, *, default_experiment_id: str) -> str:
    if experiment_id == default_experiment_id:
        return step_id
    return f"{experiment_id}.{step_id}"


def _non_negative_int(value: object, label: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentQueueError(f"{label} must be a non-negative integer")
    return value


def _finite_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _resource_kind_value(value: object, label: str) -> str:
    kind = _require_text(value, label)
    if kind in KNOWN_RESOURCE_KINDS or kind.startswith("cloud_"):
        return kind
    raise ExperimentQueueError(
        f"{label} unsupported resource kind {kind!r}; known kinds are {sorted(KNOWN_RESOURCE_KINDS)}"
    )


def normalize_resource_kind(value: object, label: str = "resource_kind") -> str:
    """Return a queue-compatible resource kind or fail closed on taxonomy drift."""

    return _resource_kind_value(value, label)


def _load_json_compatible(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raise ExperimentQueueError(
                f"{path}: YAML queue files must be JSON-compatible until PyYAML is explicitly added as a dependency"
            ) from exc
        raise ExperimentQueueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: queue root must be an object")
    return payload


def _normalize_step(raw: Mapping[str, Any], *, experiment_id: str, index: int) -> dict[str, Any]:
    step_id = _require_text(raw.get("id"), f"experiments[{experiment_id}].steps[{index}].id")
    kind = _require_text(raw.get("kind", "command"), f"{step_id}.kind")
    if kind != "command":
        raise ExperimentQueueError(f"{step_id}.kind unsupported: {kind!r}")
    command = raw.get("command")
    if not isinstance(command, list) or not command:
        raise ExperimentQueueError(f"{step_id}.command must be a non-empty argv list")
    command_items = [_require_text(item, f"{step_id}.command[{idx}]") for idx, item in enumerate(command)]
    resources = _optional_mapping(raw.get("resources"), f"{step_id}.resources")
    resource_kind = _resource_kind_value(
        resources.get("kind", "local_cpu"),
        f"{step_id}.resources.kind",
    )
    requires = _string_list(raw.get("requires"), f"{step_id}.requires")
    postconditions = raw.get("postconditions", [])
    if not isinstance(postconditions, list):
        raise ExperimentQueueError(f"{step_id}.postconditions must be a list")
    normalized_postconditions: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(postconditions):
        if not isinstance(condition, Mapping):
            raise ExperimentQueueError(f"{step_id}.postconditions[{condition_index}] must be an object")
        normalized_postconditions.append(dict(condition))
    telemetry = _optional_mapping(raw.get("telemetry"), f"{step_id}.telemetry")
    telemetry_artifact_paths = _string_list(
        telemetry.get("artifact_paths"),
        f"{step_id}.telemetry.artifact_paths",
    )
    telemetry_input_artifact_paths = _string_list(
        telemetry.get("input_artifact_paths"),
        f"{step_id}.telemetry.input_artifact_paths",
    )
    telemetry_recursive = bool(telemetry.get("recursive", False))
    telemetry_max_entries = _non_negative_int(
        telemetry.get("max_recursive_entries"),
        f"{step_id}.telemetry.max_recursive_entries",
        default=256,
    )
    artifact_mobility = _optional_mapping(
        raw.get("artifact_mobility"),
        f"{step_id}.artifact_mobility",
    )
    if artifact_mobility:
        schema = artifact_mobility.get("schema", "experiment_queue_artifact_mobility.v1")
        if schema != "experiment_queue_artifact_mobility.v1":
            raise ExperimentQueueError(f"{step_id}.artifact_mobility.schema unsupported: {schema!r}")
        artifact_mobility = {**artifact_mobility, "schema": schema}
    return {
        "id": step_id,
        "kind": kind,
        "command": command_items,
        "requires": requires,
        "resources": {**resources, "kind": resource_kind},
        "postconditions": normalized_postconditions,
        "timeout_seconds": _non_negative_int(raw.get("timeout_seconds"), f"{step_id}.timeout_seconds"),
        "telemetry": {
            **telemetry,
            "artifact_paths": telemetry_artifact_paths,
            "input_artifact_paths": telemetry_input_artifact_paths,
            "recursive": telemetry_recursive,
            "max_recursive_entries": telemetry_max_entries,
            "include_postcondition_paths": bool(telemetry.get("include_postcondition_paths", True)),
        },
        "artifact_mobility": artifact_mobility,
    }


def normalize_queue_definition(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = _require_text(payload.get("schema", QUEUE_SCHEMA), "schema")
    if schema != QUEUE_SCHEMA:
        raise ExperimentQueueError(f"unsupported queue schema: {schema!r}")
    queue_id = _require_text(payload.get("queue_id"), "queue_id")
    controls = _optional_mapping(payload.get("controls"), "controls")
    mode = _require_text(controls.get("mode", "running"), "controls.mode")
    if mode not in CONTROL_MODES:
        raise ExperimentQueueError(f"controls.mode must be one of {sorted(CONTROL_MODES)}")
    max_concurrency = _optional_mapping(controls.get("max_concurrency"), "controls.max_concurrency")
    normalized_concurrency: dict[str, int] = {}
    for key, value in max_concurrency.items():
        normalized_concurrency[_resource_kind_value(key, "controls.max_concurrency key")] = _non_negative_int(
            value,
            f"controls.max_concurrency.{key}",
            default=1,
        )
    local_first = bool(controls.get("local_first", True))

    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ExperimentQueueError("experiments must be a non-empty list")
    normalized_experiments: list[dict[str, Any]] = []
    seen_experiments: set[str] = set()
    for exp_index, raw_experiment in enumerate(experiments):
        if not isinstance(raw_experiment, Mapping):
            raise ExperimentQueueError(f"experiments[{exp_index}] must be an object")
        experiment_id = _require_text(raw_experiment.get("id"), f"experiments[{exp_index}].id")
        if experiment_id in seen_experiments:
            raise ExperimentQueueError(f"duplicate experiment id: {experiment_id}")
        seen_experiments.add(experiment_id)
        status = _require_text(raw_experiment.get("status", "queued"), f"{experiment_id}.status")
        if status not in {"queued", "paused", "frozen", "disabled"}:
            raise ExperimentQueueError(f"{experiment_id}.status unsupported: {status!r}")
        priority = _non_negative_int(raw_experiment.get("priority"), f"{experiment_id}.priority", default=100)
        raw_steps = raw_experiment.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ExperimentQueueError(f"{experiment_id}.steps must be a non-empty list")
        steps = [
            _normalize_step(step, experiment_id=experiment_id, index=step_index)
            for step_index, step in enumerate(raw_steps)
            if isinstance(step, Mapping)
        ]
        if len(steps) != len(raw_steps):
            raise ExperimentQueueError(f"{experiment_id}.steps must contain only objects")
        seen_step_ids: set[str] = set()
        duplicate_step_ids: set[str] = set()
        for step in steps:
            step_id = str(step["id"])
            if step_id in seen_step_ids:
                duplicate_step_ids.add(step_id)
            seen_step_ids.add(step_id)
        if duplicate_step_ids:
            duplicates = sorted(duplicate_step_ids)
            raise ExperimentQueueError(f"{experiment_id}.steps contains duplicate step id(s): {duplicates}")
        normalized_experiments.append(
            {
                "id": experiment_id,
                "status": status,
                "priority": priority,
                "lane_id": raw_experiment.get("lane_id"),
                "tags": _string_list(raw_experiment.get("tags"), f"{experiment_id}.tags"),
                "metadata": _optional_mapping(
                    raw_experiment.get("metadata"),
                    f"{experiment_id}.metadata",
                ),
                "steps": steps,
            }
        )
    valid_step_keys = {
        (str(experiment["id"]), str(step["id"]))
        for experiment in normalized_experiments
        for step in experiment["steps"]
    }
    for experiment in normalized_experiments:
        experiment_id = str(experiment["id"])
        for step in experiment["steps"]:
            unknown: list[str] = []
            normalized_requires: list[str] = []
            for required in step["requires"]:
                required_experiment_id, required_step_id = _resolve_step_ref(
                    str(required),
                    default_experiment_id=experiment_id,
                )
                if (required_experiment_id, required_step_id) not in valid_step_keys:
                    unknown.append(
                        _format_step_ref(
                            required_experiment_id,
                            required_step_id,
                            default_experiment_id=experiment_id,
                        )
                    )
                    continue
                normalized_requires.append(
                    _format_step_ref(
                        required_experiment_id,
                        required_step_id,
                        default_experiment_id=experiment_id,
                    )
                )
            if unknown:
                raise ExperimentQueueError(f"{experiment_id}.{step['id']} requires unknown step(s): {sorted(unknown)}")
            step["requires"] = normalized_requires
    _assert_no_dependency_cycles(normalized_experiments)
    normalized: dict[str, Any] = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": {
            "mode": mode,
            "local_first": local_first,
            "max_concurrency": normalized_concurrency,
        },
        "experiments": normalized_experiments,
    }
    if "metadata" in payload:
        normalized["metadata"] = _optional_mapping(payload.get("metadata"), "metadata")
    return normalized


def _assert_no_dependency_cycles(experiments: list[dict[str, Any]]) -> None:
    by_key = {
        (str(experiment["id"]), str(step["id"])): step for experiment in experiments for step in experiment["steps"]
    }
    memo: set[tuple[str, str]] = set()
    active: set[tuple[str, str]] = set()

    def visit(key: tuple[str, str]) -> None:
        if key in memo:
            return
        if key in active:
            raise ExperimentQueueError(f"dependency cycle detected at {key[0]}.{key[1]}")
        active.add(key)
        experiment_id, _step_id = key
        for required in by_key[key].get("requires") or []:
            visit(_resolve_step_ref(str(required), default_experiment_id=experiment_id))
        active.remove(key)
        memo.add(key)

    for key in by_key:
        visit(key)


def _downstream_step_ids_from_experiments(
    experiments: Sequence[Mapping[str, Any]],
    experiment_id: str,
    step_id: str,
) -> list[tuple[str, str]]:
    step_keys = {(str(experiment["id"]), str(step["id"])) for experiment in experiments for step in experiment["steps"]}
    root = (experiment_id, step_id)
    if root not in step_keys:
        raise ExperimentQueueError(f"unknown step: {experiment_id}.{step_id}")
    requirements = {
        (str(experiment["id"]), str(step["id"])): {
            _resolve_step_ref(str(required), default_experiment_id=str(experiment["id"]))
            for required in step.get("requires") or []
        }
        for experiment in experiments
        for step in experiment["steps"]
    }
    out: list[tuple[str, str]] = []
    seen = {root}
    pending = [root]
    while pending:
        current = pending.pop(0)
        for candidate_key, required_keys in requirements.items():
            if candidate_key in seen or current not in required_keys:
                continue
            seen.add(candidate_key)
            out.append(candidate_key)
            pending.append(candidate_key)
    return out


def load_queue_definition(path: str | Path) -> dict[str, Any]:
    return normalize_queue_definition(_load_json_compatible(Path(path)))


def default_state_path(repo_root: str | Path, queue_id: str) -> Path:
    return Path(repo_root) / ".omx" / "state" / f"experiment_queue_{queue_id}.sqlite"


def assert_canonical_state_for_execution(
    repo_root: str | Path,
    queue_id: str,
    state_path: str | Path,
    *,
    allow_noncanonical_state: bool = False,
) -> None:
    if allow_noncanonical_state:
        return
    canonical = default_state_path(repo_root, queue_id).resolve()
    actual = Path(state_path).resolve()
    if actual != canonical:
        raise ExperimentQueueError(
            "executing an experiment queue with a noncanonical state path can duplicate "
            f"default-state workers for queue {queue_id!r}; use {canonical} or pass "
            "--noncanonical-state-rationale with an explicit collision-risk rationale"
        )


def connect_state(path: str | Path) -> sqlite3.Connection:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(state_path)
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    return conn


def connect_state_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing queue state without creating tables or mutating rows."""

    state_path = Path(path)
    if not state_path.is_file():
        raise ExperimentQueueError(f"queue state does not exist: {state_path}")
    conn = sqlite3.connect(f"file:{state_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS queue_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS queue_controls (
          queue_id TEXT PRIMARY KEY,
          mode TEXT NOT NULL,
          reason TEXT,
          updated_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS step_state (
          queue_id TEXT NOT NULL,
          experiment_id TEXT NOT NULL,
          step_id TEXT NOT NULL,
          status TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          updated_at_utc TEXT NOT NULL,
          last_event_json TEXT,
          PRIMARY KEY (queue_id, experiment_id, step_id)
        );
        CREATE TABLE IF NOT EXISTS queue_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_utc TEXT NOT NULL,
          queue_id TEXT NOT NULL,
          experiment_id TEXT,
          step_id TEXT,
          event_type TEXT NOT NULL,
          payload_json TEXT NOT NULL
        );
        """
    )
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(step_state)").fetchall()}
    for column in (
        "definition_hash",
        "command_hash",
        "postcondition_hash",
        "resource_kind",
    ):
        if column not in columns:
            conn.execute(f"ALTER TABLE step_state ADD COLUMN {column} TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO queue_meta(key, value) VALUES (?, ?)",
        ("schema", STATE_SCHEMA),
    )
    conn.commit()


def _step_hashes(
    step: Mapping[str, Any],
    *,
    experiment_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    command = [str(item) for item in step.get("command", [])]
    postconditions = [dict(item) for item in step.get("postconditions", [])]
    definition = {
        "id": str(step.get("id") or ""),
        "kind": str(step.get("kind") or "command"),
        "command": command,
        "requires": [str(item) for item in step.get("requires", [])],
        "resources": dict(step.get("resources") or {}),
        "postconditions": postconditions,
        "timeout_seconds": int(step.get("timeout_seconds") or 0),
        "telemetry": dict(step.get("telemetry") or {}),
        "artifact_mobility": dict(step.get("artifact_mobility") or {}),
    }
    if experiment_metadata:
        definition["experiment_metadata"] = dict(experiment_metadata)
    return {
        "definition_hash": _sha256_json(definition),
        "command_hash": _sha256_json(command),
        "postcondition_hash": _sha256_json(postconditions),
    }


def initialize_queue_state(conn: sqlite3.Connection, queue: Mapping[str, Any]) -> None:
    queue_id = str(queue["queue_id"])
    now = _utc_now()
    mode = str(queue["controls"]["mode"])
    prior_count = conn.execute(
        "SELECT COUNT(*) AS count FROM step_state WHERE queue_id = ?",
        (queue_id,),
    ).fetchone()["count"]
    _raise_on_running_definition_drift(conn, queue)
    conn.execute(
        """
        INSERT OR IGNORE INTO queue_controls(queue_id, mode, reason, updated_at_utc)
        VALUES (?, ?, ?, ?)
        """,
        (queue_id, mode, "initialized_from_queue_definition", now),
    )
    for experiment in queue["experiments"]:
        experiment_metadata = dict(experiment.get("metadata") or {})
        for step in experiment["steps"]:
            hashes = _step_hashes(step, experiment_metadata=experiment_metadata)
            row = conn.execute(
                """
                SELECT status, definition_hash, command_hash, postcondition_hash,
                       resource_kind
                FROM step_state
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (queue_id, experiment["id"], step["id"]),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO step_state(
                      queue_id, experiment_id, step_id, status, attempts,
                      updated_at_utc, last_event_json, definition_hash,
                      command_hash, postcondition_hash, resource_kind
                    )
                    VALUES (?, ?, ?, 'queued', 0, ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        queue_id,
                        experiment["id"],
                        step["id"],
                        now,
                        hashes["definition_hash"],
                        hashes["command_hash"],
                        hashes["postcondition_hash"],
                        _resource_kind(step),
                    ),
                )
                continue

            previous_hashes = {
                "definition_hash": row["definition_hash"],
                "command_hash": row["command_hash"],
                "postcondition_hash": row["postcondition_hash"],
            }
            missing_hash = any(value is None for value in previous_hashes.values())
            missing_resource_kind = row["resource_kind"] is None
            changed_hash = any(
                previous_hashes[key] is not None and previous_hashes[key] != hashes[key] for key in hashes
            )
            if (missing_hash or missing_resource_kind) and not changed_hash:
                conn.execute(
                    """
                    UPDATE step_state
                    SET definition_hash = ?, command_hash = ?, postcondition_hash = ?,
                        resource_kind = ?
                    WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                    """,
                    (
                        hashes["definition_hash"],
                        hashes["command_hash"],
                        hashes["postcondition_hash"],
                        _resource_kind(step),
                        queue_id,
                        experiment["id"],
                        step["id"],
                    ),
                )
                continue
            if not changed_hash:
                continue
            if str(row["status"]) == "running":
                raise ExperimentQueueError(
                    f"{experiment['id']}.{step['id']} definition changed while running; "
                    "wait for the step to finish, then rewind or use a fresh state"
                )
            event = {
                "previous_status": str(row["status"]),
                "definition_changed": True,
                "previous_hashes": previous_hashes,
                "new_hashes": hashes,
            }
            conn.execute(
                """
                UPDATE step_state
                SET status = 'queued', updated_at_utc = ?, last_event_json = ?,
                    definition_hash = ?, command_hash = ?, postcondition_hash = ?,
                    resource_kind = ?
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (
                    now,
                    _json_text(event),
                    hashes["definition_hash"],
                    hashes["command_hash"],
                    hashes["postcondition_hash"],
                    _resource_kind(step),
                    queue_id,
                    experiment["id"],
                    step["id"],
                ),
            )
            append_event(
                conn,
                queue_id=queue_id,
                experiment_id=str(experiment["id"]),
                step_id=str(step["id"]),
                event_type="step_definition_changed_requeued",
                payload=event,
            )
            for downstream_experiment_id, downstream_step_id in _downstream_step_ids_from_experiments(
                queue["experiments"],
                str(experiment["id"]),
                str(step["id"]),
            ):
                downstream_row = conn.execute(
                    """
                    SELECT status
                    FROM step_state
                    WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                    """,
                    (queue_id, downstream_experiment_id, downstream_step_id),
                ).fetchone()
                if downstream_row is None:
                    continue
                if str(downstream_row["status"]) == "running":
                    raise ExperimentQueueError(
                        f"{downstream_experiment_id}.{downstream_step_id} depends on changed "
                        f"{experiment['id']}.{step['id']} while running; wait for the step to finish, "
                        "then reinitialize or use a fresh state"
                    )
                downstream_event = {
                    "previous_status": str(downstream_row["status"]),
                    "definition_changed": False,
                    "upstream_definition_changed": True,
                    "upstream_experiment_id": str(experiment["id"]),
                    "upstream_step_id": str(step["id"]),
                }
                conn.execute(
                    """
                    UPDATE step_state
                    SET status = 'queued', updated_at_utc = ?, last_event_json = ?
                    WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                    """,
                    (
                        now,
                        _json_text(downstream_event),
                        queue_id,
                        downstream_experiment_id,
                        downstream_step_id,
                    ),
                )
                append_event(
                    conn,
                    queue_id=queue_id,
                    experiment_id=downstream_experiment_id,
                    step_id=downstream_step_id,
                    event_type="step_dependency_definition_changed_requeued",
                    payload=downstream_event,
                )
    if int(prior_count) == 0:
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=None,
            step_id=None,
            event_type="initialized",
            payload={"experiment_count": len(queue["experiments"])},
        )
    conn.commit()


def _raise_on_running_definition_drift(conn: sqlite3.Connection, queue: Mapping[str, Any]) -> None:
    """Fail before mutating when definition drift would invalidate a running step."""

    queue_id = str(queue["queue_id"])
    for experiment in queue["experiments"]:
        experiment_metadata = dict(experiment.get("metadata") or {})
        for step in experiment["steps"]:
            hashes = _step_hashes(step, experiment_metadata=experiment_metadata)
            row = conn.execute(
                """
                SELECT status, definition_hash, command_hash, postcondition_hash,
                       resource_kind
                FROM step_state
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (queue_id, experiment["id"], step["id"]),
            ).fetchone()
            if row is None:
                continue
            previous_hashes = {
                "definition_hash": row["definition_hash"],
                "command_hash": row["command_hash"],
                "postcondition_hash": row["postcondition_hash"],
            }
            changed_hash = any(
                previous_hashes[key] is not None and previous_hashes[key] != hashes[key] for key in hashes
            )
            if not changed_hash:
                continue
            if str(row["status"]) == "running":
                raise ExperimentQueueError(
                    f"{experiment['id']}.{step['id']} definition changed while running; "
                    "wait for the step to finish, then rewind or use a fresh state"
                )
            for downstream_experiment_id, downstream_step_id in _downstream_step_ids_from_experiments(
                queue["experiments"],
                str(experiment["id"]),
                str(step["id"]),
            ):
                downstream_row = conn.execute(
                    """
                    SELECT status
                    FROM step_state
                    WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                    """,
                    (queue_id, downstream_experiment_id, downstream_step_id),
                ).fetchone()
                if downstream_row is not None and str(downstream_row["status"]) == "running":
                    raise ExperimentQueueError(
                        f"{downstream_experiment_id}.{downstream_step_id} depends on changed "
                        f"{experiment['id']}.{step['id']} while running; wait for the step to finish, "
                        "then reinitialize or use a fresh state"
                    )


def append_event(
    conn: sqlite3.Connection,
    *,
    queue_id: str,
    experiment_id: str | None,
    step_id: str | None,
    event_type: str,
    payload: Mapping[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO queue_events(ts_utc, queue_id, experiment_id, step_id, event_type, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_utc_now(), queue_id, experiment_id, step_id, event_type, _json_text(dict(payload))),
    )


def control_mode(conn: sqlite3.Connection, queue_id: str) -> str:
    row = conn.execute(
        "SELECT mode FROM queue_controls WHERE queue_id = ?",
        (queue_id,),
    ).fetchone()
    return str(row["mode"]) if row else "running"


def set_control_mode(conn: sqlite3.Connection, queue_id: str, mode: str, *, reason: str = "") -> None:
    if mode not in CONTROL_MODES:
        raise ExperimentQueueError(f"mode must be one of {sorted(CONTROL_MODES)}")
    now = _utc_now()
    conn.execute(
        """
        INSERT INTO queue_controls(queue_id, mode, reason, updated_at_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(queue_id) DO UPDATE SET
          mode = excluded.mode,
          reason = excluded.reason,
          updated_at_utc = excluded.updated_at_utc
        """,
        (queue_id, mode, reason, now),
    )
    append_event(
        conn,
        queue_id=queue_id,
        experiment_id=None,
        step_id=None,
        event_type=f"control_{mode}",
        payload={"reason": reason},
    )
    conn.commit()


def _state_rows(conn: sqlite3.Connection, queue_id: str) -> dict[tuple[str, str], sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT experiment_id, step_id, status, attempts, updated_at_utc,
               last_event_json, definition_hash, command_hash, postcondition_hash
        FROM step_state WHERE queue_id = ?
        """,
        (queue_id,),
    ).fetchall()
    return {(str(row["experiment_id"]), str(row["step_id"])): row for row in rows}


def _resource_kind(step: Mapping[str, Any]) -> str:
    resources = step.get("resources")
    if not isinstance(resources, Mapping):
        return "local_cpu"
    return str(resources.get("kind") or "local_cpu")


def _is_cloud_resource(kind: str) -> bool:
    return kind in CLOUD_RESOURCE_KINDS or kind.startswith("cloud_")


def queue_resource_kinds(queue: Mapping[str, Any]) -> list[str]:
    """Return resource kinds consumed by steps in a queue definition."""

    return sorted(
        {
            _resource_kind(step)
            for experiment in queue.get("experiments", [])
            if isinstance(experiment, Mapping)
            for step in experiment.get("steps", [])
            if isinstance(step, Mapping)
        }
    )


def worker_resource_limits(
    queue: Mapping[str, Any],
    *,
    allow_cloud: bool = False,
) -> dict[str, int]:
    """Return positive per-resource caps that the worker may use.

    Queue definitions already carry resource-level concurrency in
    ``controls.max_concurrency``. This helper turns that declaration into the
    worker's global parallelism budget so local CPU/MLX queues can saturate
    declared resources without requiring a second hand-tuned ``--max-parallel``
    value.
    """

    configured = dict(queue.get("controls", {}).get("max_concurrency") or {})
    limits: dict[str, int] = {}
    for kind in queue_resource_kinds(queue):
        if _is_cloud_resource(kind) and not allow_cloud:
            continue
        limit = int(configured.get(kind, 1))
        if limit > 0:
            limits[kind] = limit
    return limits


def resolve_worker_max_parallel(
    queue: Mapping[str, Any],
    requested_max_parallel: int | None,
    *,
    allow_cloud: bool = False,
) -> tuple[int, dict[str, int]]:
    """Resolve worker-level process parallelism from queue resource controls.

    ``None`` or ``0`` means "auto": sum the positive resource caps visible to
    this worker, excluding cloud resources unless ``allow_cloud`` is explicit.
    A positive requested value remains an operator override.
    """

    if isinstance(requested_max_parallel, bool):
        raise ExperimentQueueError("max_parallel must be an integer")
    resource_limits = worker_resource_limits(queue, allow_cloud=allow_cloud)
    if requested_max_parallel in (None, 0):
        return max(1, sum(resource_limits.values())), resource_limits
    if requested_max_parallel < 0:
        raise ExperimentQueueError("max_parallel must be non-negative or 0 for auto")
    return requested_max_parallel, resource_limits


def ready_steps(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    allow_cloud: bool = False,
    repo_root: str | Path | None = None,
) -> list[ReadyStep]:
    queue_id = str(queue["queue_id"])
    if control_mode(conn, queue_id) != "running":
        return []
    state = _state_rows(conn, queue_id)
    repo = Path(repo_root) if repo_root is not None else None
    steps_by_key = {
        (str(experiment["id"]), str(step["id"])): step
        for experiment in queue["experiments"]
        for step in experiment["steps"]
    }
    running_by_resource: dict[str, int] = {}
    for experiment in queue["experiments"]:
        for step in experiment["steps"]:
            row = state.get((str(experiment["id"]), str(step["id"])))
            if row and row["status"] == "running":
                kind = _resource_kind(step)
                running_by_resource[kind] = running_by_resource.get(kind, 0) + 1

    max_concurrency = dict(queue["controls"].get("max_concurrency") or {})
    out: list[ReadyStep] = []
    for experiment in queue["experiments"]:
        if experiment["status"] != "queued":
            continue
        for step in experiment["steps"]:
            row = state.get((str(experiment["id"]), str(step["id"])))
            if row is None or row["status"] != "queued":
                continue
            kind = _resource_kind(step)
            if _is_cloud_resource(kind) and not allow_cloud:
                continue
            limit = int(max_concurrency.get(kind, 1))
            if running_by_resource.get(kind, 0) >= limit:
                continue
            requirements = [
                _resolve_step_ref(str(required), default_experiment_id=str(experiment["id"]))
                for required in (step.get("requires") or [])
            ]
            dependency_blocked = False
            for required_key in requirements:
                required_row = state.get(required_key)
                if required_row is None or required_row["status"] != "succeeded":
                    dependency_blocked = True
                    break
                if repo is None:
                    continue
                required_step = steps_by_key.get(required_key)
                if required_step is None:
                    dependency_blocked = True
                    break
                failed_conditions, _postcondition_errors = _evaluate_postconditions(
                    required_step,
                    repo=repo,
                )
                if failed_conditions:
                    dependency_blocked = True
                    break
            if dependency_blocked:
                continue
            hashes = _step_hashes(
                step,
                experiment_metadata=dict(experiment.get("metadata") or {}),
            )
            out.append(
                ReadyStep(
                    queue_id=queue_id,
                    experiment_id=str(experiment["id"]),
                    step_id=str(step["id"]),
                    priority=int(experiment["priority"]),
                    resource_kind=kind,
                    command=tuple(str(part) for part in step["command"]),
                    definition_hash=hashes["definition_hash"],
                    command_hash=hashes["command_hash"],
                    postcondition_hash=hashes["postcondition_hash"],
                )
            )
            running_by_resource[kind] = running_by_resource.get(kind, 0) + 1
    local_first = bool(queue["controls"].get("local_first", True))
    out.sort(
        key=lambda step: (
            step.priority,
            0 if not local_first or step.resource_kind in LOCAL_RESOURCE_KINDS else 1,
            step.experiment_id,
            step.step_id,
        )
    )
    return out


def _lookup_step(queue: Mapping[str, Any], experiment_id: str, step_id: str) -> dict[str, Any]:
    for experiment in queue["experiments"]:
        if experiment["id"] != experiment_id:
            continue
        for step in experiment["steps"]:
            if step["id"] == step_id:
                return dict(step)
    raise ExperimentQueueError(f"unknown step: {experiment_id}.{step_id}")


def _lookup_step_with_experiment_metadata(
    queue: Mapping[str, Any],
    experiment_id: str,
    step_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for experiment in queue["experiments"]:
        if experiment["id"] != experiment_id:
            continue
        for step in experiment["steps"]:
            if step["id"] == step_id:
                return dict(step), dict(experiment.get("metadata") or {})
    raise ExperimentQueueError(f"unknown step: {experiment_id}.{step_id}")


def _json_pointer(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise ExperimentQueueError(f"json key not found: {key}")
    return current


def _resolve_postcondition_path(path_value: str, *, repo_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(ch in SHA256_HEX for ch in text)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_record_valid(record: Any, *, repo_root: Path) -> bool:
    if not isinstance(record, Mapping):
        return False
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return False
    expected_sha = record.get("sha256")
    expected_bytes = _positive_int(record.get("bytes"))
    if not _is_sha256(expected_sha) or expected_bytes is None:
        return False
    path = _resolve_postcondition_path(path_value, repo_root=repo_root)
    if not path.is_file() or path.is_symlink():
        return False
    return path.stat().st_size == expected_bytes and _sha256_file(path) == str(expected_sha).lower()


def _false_authority_payload_valid(
    payload: Mapping[str, Any],
    *,
    required_false: Sequence[str],
    false_or_missing: Sequence[str],
) -> bool:
    for key in required_false:
        try:
            if _json_pointer(payload, key) is not False:
                return False
        except ExperimentQueueError:
            return False
    for key in false_or_missing:
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            continue
        if value is not False:
            return False
    return True


def _jsonl_false_authority_valid(
    condition: Mapping[str, Any],
    *,
    repo_root: Path,
) -> bool:
    rel_path = _require_text(condition.get("path"), "postcondition.path")
    path = _resolve_postcondition_path(rel_path, repo_root=repo_root)
    if not path.is_file() or path.is_symlink():
        return False
    required_false = _string_list(
        condition.get("required_false", list(DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS)),
        "postcondition.required_false",
    )
    false_or_missing = _string_list(
        condition.get(
            "false_or_missing",
            list(DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS),
        ),
        "postcondition.false_or_missing",
    )
    require_nonempty = bool(condition.get("require_nonempty", True))
    schema_equals = condition.get("schema_equals")
    seen_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return False
            if not isinstance(payload, Mapping):
                return False
            if schema_equals is not None and payload.get("schema") != schema_equals:
                return False
            if not _false_authority_payload_valid(
                payload,
                required_false=required_false,
                false_or_missing=false_or_missing,
            ):
                return False
            seen_rows += 1
    return seen_rows > 0 if require_nonempty else True


def _nonempty_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping | list | tuple | set):
        return bool(value)
    return value is not None


def _json_completion_contract(
    condition: Mapping[str, Any],
    *,
    repo_root: Path,
) -> bool:
    rel_path = _require_text(condition.get("path"), "postcondition.path")
    payload = json.loads(_resolve_postcondition_path(rel_path, repo_root=repo_root).read_text())
    if not isinstance(payload, Mapping):
        return False
    for status in _string_list(
        condition.get("forbidden_statuses", ["failed"]),
        "postcondition.forbidden_statuses",
    ):
        if payload.get("status") == status:
            return False
    required_equals = _optional_mapping(
        condition.get("required_equals"),
        "postcondition.required_equals",
    )
    for key, expected in required_equals.items():
        try:
            actual = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if actual != expected:
            return False
    for key in _string_list(condition.get("required_true"), "postcondition.required_true"):
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if value is not True:
            return False
    for key in _string_list(condition.get("required_false"), "postcondition.required_false"):
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if value is not False:
            return False
    for key in _string_list(
        condition.get("false_or_missing"),
        "postcondition.false_or_missing",
    ):
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            continue
        if value is not False:
            return False
    for key in _string_list(
        condition.get("required_nonempty"),
        "postcondition.required_nonempty",
    ):
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if not _nonempty_value(value):
            return False
    for index, raw_item in enumerate(condition.get("required_nonempty_unless_true", []) or []):
        if not isinstance(raw_item, Mapping):
            raise ExperimentQueueError(f"postcondition.required_nonempty_unless_true[{index}] must be an object")
        key = _require_text(
            raw_item.get("key"),
            f"postcondition.required_nonempty_unless_true[{index}].key",
        )
        unless_true = _require_text(
            raw_item.get("unless_true"),
            f"postcondition.required_nonempty_unless_true[{index}].unless_true",
        )
        try:
            if _json_pointer(payload, unless_true) is True:
                continue
        except ExperimentQueueError:
            pass
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if not _nonempty_value(value):
            return False
    for key in _string_list(
        condition.get("required_positive_int"),
        "postcondition.required_positive_int",
    ):
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if _positive_int(value) is None:
            return False
    for key in _string_list(
        condition.get("required_sha256"),
        "postcondition.required_sha256",
    ):
        try:
            value = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if not _is_sha256(value):
            return False
    for key in _string_list(
        condition.get("required_artifact_records"),
        "postcondition.required_artifact_records",
    ):
        try:
            record = _json_pointer(payload, key)
        except ExperimentQueueError:
            return False
        if not _artifact_record_valid(record, repo_root=repo_root):
            return False
    if bool(condition.get("required_runtime_adapter_identity")) and runtime_adapter_identity_blockers(
        payload,
        repo_root=repo_root,
        context="json_completion_contract",
    ):
        return False
    for index, raw_pair in enumerate(condition.get("required_less_than", []) or []):
        if not isinstance(raw_pair, Mapping):
            raise ExperimentQueueError(f"postcondition.required_less_than[{index}] must be an object")
        left_key = _require_text(
            raw_pair.get("left"),
            f"postcondition.required_less_than[{index}].left",
        )
        right_key = _require_text(
            raw_pair.get("right"),
            f"postcondition.required_less_than[{index}].right",
        )
        try:
            left = _finite_int(_json_pointer(payload, left_key))
            right = _finite_int(_json_pointer(payload, right_key))
        except ExperimentQueueError:
            return False
        if left is None or right is None or left >= right:
            return False
    return True


def _materializer_chain_complete(
    condition: Mapping[str, Any],
    *,
    repo_root: Path,
) -> bool:
    rel_path = _require_text(condition.get("path"), "postcondition.path")
    payload = json.loads(_resolve_postcondition_path(rel_path, repo_root=repo_root).read_text())
    if not isinstance(payload, Mapping):
        return False
    expected_schema = _require_text(condition.get("schema"), "postcondition.schema")
    if payload.get("schema") != expected_schema:
        return False
    if payload.get("byte_closed_candidate_emitted") is not True:
        return False
    if payload.get("runtime_adapter_ready") is not True:
        return False
    if payload.get("receiver_contract_satisfied") is not True:
        return False
    if payload.get("candidate_runtime_adapter_blocker_cleared") is not True:
        return False
    if bool(condition.get("forbid_readiness_blockers")) and payload.get("readiness_blockers"):
        return False
    if not _is_sha256(payload.get("candidate_archive_sha256")):
        return False
    if _positive_int(payload.get("candidate_archive_bytes")) is None:
        return False
    if not _artifact_record_valid(payload.get("candidate_archive"), repo_root=repo_root):
        return False
    require_runtime_identity = condition.get("required_runtime_adapter_identity") is not False
    if require_runtime_identity and runtime_adapter_identity_blockers(
        payload,
        repo_root=repo_root,
        context="materializer_chain_complete",
    ):
        return False
    if bool(condition.get("required_serialized_archive_saving")):
        source_bytes = _positive_int(payload.get("source_archive_bytes"))
        candidate_bytes = _positive_int(payload.get("candidate_archive_bytes"))
        if source_bytes is None or candidate_bytes is None or candidate_bytes >= source_bytes:
            return False
        delta = payload.get("serialized_archive_delta")
        if not isinstance(delta, Mapping) or delta.get("status") != "realized_saving":
            return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return False
    for record in artifacts.values():
        if not _artifact_record_valid(record, repo_root=repo_root):
            return False
    chain_steps = payload.get("chain_steps")
    if not isinstance(chain_steps, list) or not chain_steps:
        return False
    for step in chain_steps:
        if not isinstance(step, Mapping) or step.get("status") != "succeeded":
            return False
        artifact = step.get("artifact")
        if artifact is not None and not _artifact_record_valid(artifact, repo_root=repo_root):
            return False
    required_false = _string_list(
        condition.get(
            "required_false",
            list(DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS),
        ),
        "postcondition.required_false",
    )
    false_or_missing = _string_list(
        condition.get(
            "false_or_missing",
            list(DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS),
        ),
        "postcondition.false_or_missing",
    )
    return _false_authority_payload_valid(
        payload,
        required_false=required_false,
        false_or_missing=false_or_missing,
    )


def _condition_passes(condition: Mapping[str, Any], *, repo_root: Path) -> bool:
    condition_type = _require_text(condition.get("type"), "postcondition.type")
    if condition_type == "path_exists":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        return _resolve_postcondition_path(rel_path, repo_root=repo_root).exists()
    if condition_type == "json_equals":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        key = _require_text(condition.get("key"), "postcondition.key")
        expected = condition.get("equals")
        payload = json.loads(_resolve_postcondition_path(rel_path, repo_root=repo_root).read_text())
        return _json_pointer(payload, key) == expected
    if condition_type == "json_file_key_equals":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        key = _require_text(condition.get("key"), "postcondition.key")
        if "value" not in condition:
            raise ExperimentQueueError("postcondition.value is required")
        expected = condition.get("value")
        payload = json.loads(_resolve_postcondition_path(rel_path, repo_root=repo_root).read_text())
        return _json_pointer(payload, key) == expected
    if condition_type == "json_array_contains":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        key = _require_text(condition.get("key"), "postcondition.key")
        if "contains" not in condition:
            raise ExperimentQueueError("postcondition.contains is required")
        expected = condition.get("contains")
        payload = json.loads(_resolve_postcondition_path(rel_path, repo_root=repo_root).read_text())
        actual = _json_pointer(payload, key)
        return isinstance(actual, list) and expected in actual
    if condition_type == "json_false_authority":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        payload = json.loads(_resolve_postcondition_path(rel_path, repo_root=repo_root).read_text())
        if not isinstance(payload, Mapping):
            return False
        required_false = _string_list(
            condition.get("required_false", list(DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS)),
            "postcondition.required_false",
        )
        false_or_missing = _string_list(
            condition.get(
                "false_or_missing",
                list(DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS),
            ),
            "postcondition.false_or_missing",
        )
        if not _false_authority_payload_valid(
            payload,
            required_false=required_false,
            false_or_missing=false_or_missing,
        ):
            return False
        axis_key = condition.get("axis_key")
        if axis_key is not None:
            actual_axis = _json_pointer(payload, _require_text(axis_key, "postcondition.axis_key"))
            expected_axis = _require_text(condition.get("axis_equals"), "postcondition.axis_equals")
            if actual_axis != expected_axis:
                return False
        return True
    if condition_type == "jsonl_false_authority":
        return _jsonl_false_authority_valid(condition, repo_root=repo_root)
    if condition_type == "json_completion_contract":
        return _json_completion_contract(condition, repo_root=repo_root)
    if condition_type == "materializer_chain_complete":
        return _materializer_chain_complete(condition, repo_root=repo_root)
    raise ExperimentQueueError(f"unsupported postcondition type: {condition_type!r}")


def _set_step_status(
    conn: sqlite3.Connection,
    *,
    queue_id: str,
    experiment_id: str,
    step_id: str,
    status: str,
    event: Mapping[str, Any],
    increment_attempts: bool = False,
) -> None:
    if status not in STEP_STATUSES:
        raise ExperimentQueueError(f"invalid step status: {status}")
    attempts_sql = "attempts + 1" if increment_attempts else "attempts"
    conn.execute(
        f"""
        UPDATE step_state
        SET status = ?, attempts = {attempts_sql}, updated_at_utc = ?, last_event_json = ?
        WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
        """,
        (_require_text(status, "status"), _utc_now(), _json_text(dict(event)), queue_id, experiment_id, step_id),
    )
    append_event(
        conn,
        queue_id=queue_id,
        experiment_id=experiment_id,
        step_id=step_id,
        event_type=f"step_{status}",
        payload=event,
    )


def _claim_refused_event_type(reason: str) -> str:
    if reason == "not_queued":
        return "step_claim_refused"
    return f"step_claim_refused_{reason}"


def _record_step_claim_refusal(
    conn: sqlite3.Connection,
    *,
    ready: ReadyStep,
    reason: str,
    event: Mapping[str, Any],
    control_mode_value: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    refused_event = {
        **dict(event),
        "claim_refused_reason": reason,
        "control_mode": control_mode_value,
        **(dict(extra) if extra is not None else {}),
    }
    append_event(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
        event_type=_claim_refused_event_type(reason),
        payload=refused_event,
    )
    conn.commit()
    return reason


def _step_state_hashes(row: sqlite3.Row) -> dict[str, str | None]:
    return {
        "definition_hash": row["definition_hash"],
        "command_hash": row["command_hash"],
        "postcondition_hash": row["postcondition_hash"],
    }


def _step_hash_mismatches(
    row: sqlite3.Row,
    expected_hashes: Mapping[str, str],
) -> dict[str, dict[str, str | None]]:
    state_hashes = _step_state_hashes(row)
    return {
        key: {"state": state_hashes.get(key), "expected": expected_hashes[key]}
        for key in expected_hashes
        if state_hashes.get(key) != expected_hashes[key]
    }


def _running_resource_count(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    resource_kind: str,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM step_state
        WHERE queue_id = ? AND status = 'running' AND resource_kind = ?
        """,
        (str(queue["queue_id"]), resource_kind),
    ).fetchone()
    return int(row["count"] if row is not None else 0)


def _process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _last_event(row: sqlite3.Row) -> dict[str, Any]:
    raw = row["last_event_json"]
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _updated_age_seconds(row: sqlite3.Row) -> float | None:
    raw = row["updated_at_utc"]
    if not isinstance(raw, str) or not raw:
        return None
    try:
        updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(UTC) - updated).total_seconds())


def _event_positive_int(event: Mapping[str, Any], key: str) -> int | None:
    value = event.get(key)
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _event_number(event: Mapping[str, Any], key: str) -> float | None:
    value = event.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reconcile_stale_running_steps(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    repo_root: str | Path,
    stale_after_seconds: float = 300.0,
) -> dict[str, Any]:
    """Recover local ``running`` rows whose recorded worker process is gone."""

    if stale_after_seconds < 0:
        raise ExperimentQueueError("stale_after_seconds must be non-negative")
    queue_id = str(queue["queue_id"])
    repo = Path(repo_root)
    step_lookup = _step_lookup_by_key(queue)
    rows = conn.execute(
        """
        SELECT experiment_id, step_id, status, attempts, updated_at_utc,
               last_event_json, resource_kind
        FROM step_state
        WHERE queue_id = ? AND status = 'running'
        ORDER BY experiment_id, step_id
        """,
        (queue_id,),
    ).fetchall()
    inspected = 0
    recovered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    now_epoch = time.time()
    for row in rows:
        inspected += 1
        experiment_id = str(row["experiment_id"])
        step_id = str(row["step_id"])
        step = step_lookup.get((experiment_id, step_id))
        if step is None:
            skipped.append(
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "reason": "step_not_in_active_queue",
                }
            )
            continue
        resource_kind = str(row["resource_kind"] or _resource_kind(step))
        if resource_kind not in LOCAL_RESOURCE_KINDS:
            skipped.append(
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "resource_kind": resource_kind,
                    "reason": "non_local_resource",
                }
            )
            continue
        event = _last_event(row)
        pid = _event_positive_int(event, "pid")
        parent_pid = _event_positive_int(event, "parent_pid")
        child_alive = _process_alive(pid)
        parent_alive = _process_alive(parent_pid)
        age_seconds = _updated_age_seconds(row)
        timeout_deadline = _event_number(event, "timeout_deadline_epoch_seconds")
        timed_out = timeout_deadline is not None and now_epoch >= timeout_deadline
        if child_alive:
            skipped.append(
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "pid": pid,
                    "reason": "child_process_alive",
                }
            )
            continue
        if parent_alive:
            skipped.append(
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "parent_pid": parent_pid,
                    "reason": "worker_parent_alive",
                }
            )
            continue
        if pid is None and (age_seconds is None or age_seconds < stale_after_seconds):
            skipped.append(
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "age_seconds": age_seconds,
                    "reason": "missing_child_pid_within_grace",
                }
            )
            continue

        failed_conditions, postcondition_errors = _evaluate_postconditions(step, repo=repo)
        succeeded = not failed_conditions and not postcondition_errors and bool(step.get("postconditions"))
        recovery_event = {
            **event,
            "stale_running_reconciled": True,
            "previous_status": "running",
            "pid": pid,
            "parent_pid": parent_pid,
            "resource_kind": resource_kind,
            "age_seconds": age_seconds,
            "timed_out": timed_out,
            "execution_error": None if succeeded else "stale_running_process_missing",
            "failed_postconditions": failed_conditions,
            "postcondition_errors": postcondition_errors,
        }
        _set_step_status(
            conn,
            queue_id=queue_id,
            experiment_id=experiment_id,
            step_id=step_id,
            status="succeeded" if succeeded else "failed",
            event=recovery_event,
        )
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=experiment_id,
            step_id=step_id,
            event_type="step_reconciled_from_stale_running",
            payload=recovery_event,
        )
        recovered.append(
            {
                "experiment_id": experiment_id,
                "step_id": step_id,
                "status": "succeeded" if succeeded else "failed",
                "resource_kind": resource_kind,
                "pid": pid,
                "parent_pid": parent_pid,
                "timed_out": timed_out,
            }
        )
    conn.commit()
    return {
        "schema": "experiment_queue_stale_running_reconciliation.v1",
        "queue_id": queue_id,
        "inspected_running_step_count": inspected,
        "reconciled_step_count": len(recovered),
        "reconciled_steps": recovered,
        "skipped_step_count": len(skipped),
        "skipped_steps": skipped,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def claim_ready_step_for_execution(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    ready: ReadyStep,
    *,
    event: Mapping[str, Any],
) -> str | None:
    """Atomically claim a ready step, enforcing queue mode, hashes, and resources."""

    return _claim_step_running(conn, queue=queue, ready=ready, event=event)


def _claim_step_running(
    conn: sqlite3.Connection,
    *,
    queue: Mapping[str, Any],
    ready: ReadyStep,
    event: Mapping[str, Any],
) -> str | None:
    queue_id = str(queue["queue_id"])
    if ready.queue_id != queue_id:
        raise ExperimentQueueError(f"ready step queue_id {ready.queue_id!r} does not match active queue {queue_id!r}")
    step, experiment_metadata = _lookup_step_with_experiment_metadata(
        queue,
        ready.experiment_id,
        ready.step_id,
    )
    step_resource_kind = _resource_kind(step)
    expected_hashes = _step_hashes(step, experiment_metadata=experiment_metadata)
    ready_hashes = {
        "definition_hash": ready.definition_hash,
        "command_hash": ready.command_hash,
        "postcondition_hash": ready.postcondition_hash,
    }
    max_concurrency = dict(queue["controls"].get("max_concurrency") or {})
    resource_limit = int(max_concurrency.get(step_resource_kind, 1))
    started_transaction = False
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
        started_transaction = True
    try:
        mode = control_mode(conn, queue_id)
        if mode != "running":
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="control_not_running",
                event=event,
                control_mode_value=mode,
            )
        ready_hash_mismatches = {
            key: {"ready": ready_hashes[key], "active_queue": expected_hashes[key]}
            for key in expected_hashes
            if ready_hashes[key] != expected_hashes[key]
        }
        if ready_hash_mismatches:
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="ready_step_definition_mismatch",
                event=event,
                control_mode_value=mode,
                extra={
                    "ready_hashes": ready_hashes,
                    "active_queue_hashes": expected_hashes,
                    "hash_mismatches": ready_hash_mismatches,
                },
            )
        row = conn.execute(
            """
            SELECT status, attempts, definition_hash, command_hash, postcondition_hash,
                   resource_kind
            FROM step_state
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (queue_id, ready.experiment_id, ready.step_id),
        ).fetchone()
        if row is None or row["status"] != "queued":
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="not_queued",
                event=event,
                control_mode_value=mode,
                extra={"current_status": str(row["status"]) if row else None},
            )
        hash_mismatches = _step_hash_mismatches(row, expected_hashes)
        if hash_mismatches:
            reason = (
                "definition_hash_missing"
                if any(value["state"] is None for value in hash_mismatches.values())
                else "definition_changed"
            )
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason=reason,
                event=event,
                control_mode_value=mode,
                extra={
                    "expected_hashes": dict(expected_hashes),
                    "state_hashes": _step_state_hashes(row),
                    "hash_mismatches": hash_mismatches,
                },
            )
        if row["resource_kind"] != step_resource_kind:
            reason = "resource_kind_missing" if row["resource_kind"] is None else "resource_kind_changed"
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason=reason,
                event=event,
                control_mode_value=mode,
                extra={
                    "state_resource_kind": row["resource_kind"],
                    "step_resource_kind": step_resource_kind,
                },
            )
        if ready.resource_kind != step_resource_kind:
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="resource_kind_changed",
                event=event,
                control_mode_value=mode,
                extra={
                    "ready_resource_kind": ready.resource_kind,
                    "step_resource_kind": step_resource_kind,
                },
            )
        missing_dependencies: list[dict[str, str]] = []
        for required in step.get("requires") or []:
            required_experiment_id, required_step_id = _resolve_step_ref(
                str(required),
                default_experiment_id=ready.experiment_id,
            )
            required_row = conn.execute(
                """
                SELECT status
                FROM step_state
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (queue_id, required_experiment_id, required_step_id),
            ).fetchone()
            if required_row is not None and required_row["status"] == "succeeded":
                continue
            missing_dependencies.append(
                {
                    "experiment_id": required_experiment_id,
                    "step_id": required_step_id,
                    "status": str(required_row["status"]) if required_row else "missing",
                }
            )
        if missing_dependencies:
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="dependency_not_satisfied",
                event=event,
                control_mode_value=mode,
                extra={"missing_dependencies": missing_dependencies},
            )
        running_count = _running_resource_count(conn, queue, step_resource_kind)
        if resource_limit <= 0 or running_count >= resource_limit:
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="resource_limit_reached",
                event=event,
                control_mode_value=mode,
                extra={
                    "resource_kind": step_resource_kind,
                    "resource_limit": resource_limit,
                    "running_count": running_count,
                },
            )
        cursor = conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = attempts + 1,
                updated_at_utc = ?,
                last_event_json = ?
            WHERE queue_id = ?
              AND experiment_id = ?
              AND step_id = ?
              AND status = 'queued'
              AND definition_hash = ?
              AND command_hash = ?
              AND postcondition_hash = ?
              AND resource_kind = ?
            """,
            (
                _utc_now(),
                _json_text(dict(event)),
                queue_id,
                ready.experiment_id,
                ready.step_id,
                expected_hashes["definition_hash"],
                expected_hashes["command_hash"],
                expected_hashes["postcondition_hash"],
                step_resource_kind,
            ),
        )
        if cursor.rowcount != 1:
            return _record_step_claim_refusal(
                conn,
                ready=ready,
                reason="claim_lost_race",
                event=event,
                control_mode_value=mode,
            )
    except Exception:
        if started_transaction:
            conn.rollback()
        raise
    append_event(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
        event_type="step_running",
        payload=event,
    )
    conn.commit()
    return None


def run_ready_step(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    ready: ReadyStep,
    *,
    repo_root: str | Path,
    execute: bool,
    log_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root)
    step = _lookup_step(queue, ready.experiment_id, ready.step_id)
    if not execute:
        return {"executed": False, "ready_step": ready.to_dict()}

    ts = _utc_now().replace(":", "").replace("-", "")
    resolved_log_root = Path(log_root) if log_root else repo / ".omx" / "state" / "experiment_queue_logs"
    log_dir = resolved_log_root / ready.queue_id / ready.experiment_id / ready.step_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ts}.log"

    running_event = {
        "command": list(ready.command),
        "log_path": str(log_path),
        "resource_kind": ready.resource_kind,
    }
    claim_refused_reason = claim_ready_step_for_execution(
        conn,
        queue,
        ready,
        event=running_event,
    )
    if claim_refused_reason:
        return {
            "executed": False,
            "claim_refused": True,
            "claim_refused_reason": claim_refused_reason,
            "ready_step": ready.to_dict(),
            **running_event,
        }

    start = time.monotonic()
    returncode = 0
    timed_out = False
    execution_error: str | None = None
    with log_path.open("wb") as handle:
        try:
            proc = subprocess.run(
                list(ready.command),
                cwd=repo,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=int(step.get("timeout_seconds") or 0) or None,
                check=False,
                env=os.environ.copy(),
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            execution_error = f"TimeoutExpired: {exc}"
            handle.write(f"\n[experiment-queue] timeout: {exc}\n".encode())
        except OSError as exc:
            returncode = 127
            execution_error = f"{type(exc).__name__}: {exc}"
            handle.write(f"\n[experiment-queue] execution error: {execution_error}\n".encode())
    elapsed = time.monotonic() - start
    conditions = [dict(condition) for condition in step.get("postconditions", [])]
    failed_conditions: list[dict[str, Any]] = []
    postcondition_errors: list[dict[str, str]] = []
    for condition in conditions:
        try:
            if not _condition_passes(condition, repo_root=repo):
                failed_conditions.append(condition)
        except (ExperimentQueueError, OSError, json.JSONDecodeError) as exc:
            failed_conditions.append(condition)
            postcondition_errors.append({"condition": _json_text(condition), "error": f"{type(exc).__name__}: {exc}"})
    succeeded = returncode == 0 and not timed_out and not execution_error and not failed_conditions
    telemetry = _step_telemetry_event(step, repo=repo, log_path=log_path)
    event = {
        "command": list(ready.command),
        "resource_kind": ready.resource_kind,
        "returncode": returncode,
        "timed_out": timed_out,
        "execution_error": execution_error,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "telemetry": telemetry,
        "failed_postconditions": failed_conditions,
        "postcondition_errors": postcondition_errors,
    }
    _set_step_status(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
        status="succeeded" if succeeded else "failed",
        event=event,
    )
    conn.commit()
    return {"executed": True, "succeeded": succeeded, **event}


def finalize_claimed_step_execution(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    ready: ReadyStep,
    *,
    repo_root: str | Path,
    log_path: str | Path,
    returncode: int,
    timed_out: bool,
    execution_error: str | None,
    elapsed_seconds: float,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize a step already claimed by an external executor.

    External executors such as SSH/Dask bridges may launch work outside the
    local subprocess worker, but terminal queue state is still local authority.
    This helper evaluates local postconditions and records the terminal event
    through the same state transition path as the built-in worker.
    """

    step = _lookup_step(queue, ready.experiment_id, ready.step_id)
    step_hashes = _step_hashes(
        step,
        experiment_metadata=dict(
            _lookup_step_with_experiment_metadata(
                queue,
                ready.experiment_id,
                ready.step_id,
            )[1]
        ),
    )
    terminal_event_base = dict(event)
    worker_run_id = terminal_event_base.get("worker_run_id")
    if not isinstance(worker_run_id, str) or not worker_run_id.strip():
        raise ExperimentQueueError("finalize event must carry worker_run_id")
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        """
        SELECT status, definition_hash, command_hash, postcondition_hash,
               resource_kind, last_event_json
        FROM step_state
        WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
        """,
        (ready.queue_id, ready.experiment_id, ready.step_id),
    ).fetchone()
    current_event = _last_event(row) if row is not None else {}
    current_worker_run_id = current_event.get("worker_run_id")
    stale_reasons: list[str] = []
    if row is None:
        stale_reasons.append("step_state_missing")
    else:
        if row["status"] != "running":
            stale_reasons.append(f"status_not_running:{row['status']}")
        if row["definition_hash"] != step_hashes["definition_hash"]:
            stale_reasons.append("definition_hash_mismatch")
        if row["command_hash"] != step_hashes["command_hash"]:
            stale_reasons.append("command_hash_mismatch")
        if row["postcondition_hash"] != step_hashes["postcondition_hash"]:
            stale_reasons.append("postcondition_hash_mismatch")
        if row["resource_kind"] != _resource_kind(step):
            stale_reasons.append("resource_kind_mismatch")
        if current_worker_run_id != worker_run_id:
            stale_reasons.append("worker_run_id_mismatch")
    if stale_reasons:
        refusal_event = {
            **terminal_event_base,
            "returncode": int(returncode),
            "timed_out": timed_out,
            "execution_error": execution_error,
            "elapsed_seconds": float(elapsed_seconds),
            "finalize_refused": True,
            "finalize_refusal_reasons": stale_reasons,
            "current_status": str(row["status"]) if row is not None else None,
            "current_worker_run_id": current_worker_run_id,
        }
        append_event(
            conn,
            queue_id=ready.queue_id,
            experiment_id=ready.experiment_id,
            step_id=ready.step_id,
            event_type="step_finalize_refused",
            payload=refusal_event,
        )
        conn.commit()
        return {
            "executed": True,
            "succeeded": False,
            **refusal_event,
        }
    failed_conditions, postcondition_errors = _evaluate_postconditions(
        step,
        repo=Path(repo_root),
    )
    succeeded = (
        int(returncode) == 0
        and not timed_out
        and execution_error is None
        and not failed_conditions
        and not postcondition_errors
    )
    terminal_event = {
        **dict(event),
        "returncode": int(returncode),
        "timed_out": timed_out,
        "execution_error": execution_error,
        "elapsed_seconds": float(elapsed_seconds),
        "telemetry": _step_telemetry_event(
            step,
            repo=Path(repo_root),
            log_path=Path(log_path),
        ),
        "failed_postconditions": failed_conditions,
        "postcondition_errors": postcondition_errors,
    }
    _set_step_status(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
        status="succeeded" if succeeded else "failed",
        event=terminal_event,
    )
    conn.commit()
    return {"executed": True, "succeeded": succeeded, **terminal_event}


def _make_worker_run_id(ready: ReadyStep) -> str:
    payload = {
        "queue_id": ready.queue_id,
        "experiment_id": ready.experiment_id,
        "step_id": ready.step_id,
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
    }
    return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()[:16]


def _safe_log_path_component(
    value: str,
    *,
    max_chars: int = LOG_PATH_COMPONENT_MAX_CHARS,
) -> str:
    raw = str(value or "").strip() or "unnamed"
    sanitized = "".join(char if char in LOG_PATH_COMPONENT_ALLOWED_CHARS else "_" for char in raw).strip("._-")
    sanitized = sanitized or "unnamed"
    if len(sanitized) <= max_chars:
        return sanitized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:LOG_PATH_COMPONENT_HASH_CHARS]
    head_len = max(1, max_chars - LOG_PATH_COMPONENT_HASH_CHARS - 1)
    head = sanitized[:head_len].rstrip("._-") or "id"
    return f"{head}_{digest}"


def _make_step_log_path(
    *,
    repo: Path,
    log_root: str | Path | None,
    ready: ReadyStep,
    worker_run_id: str,
) -> Path:
    ts = _utc_now().replace(":", "").replace("-", "")
    resolved_log_root = Path(log_root) if log_root else repo / ".omx" / "state" / "experiment_queue_logs"
    log_dir = (
        resolved_log_root
        / _safe_log_path_component(ready.queue_id)
        / _safe_log_path_component(ready.experiment_id)
        / _safe_log_path_component(ready.step_id)
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{ts}_{worker_run_id}.log"


def _record_process_started(
    conn: sqlite3.Connection,
    *,
    ready: ReadyStep,
    event: Mapping[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE step_state
        SET status = 'running', updated_at_utc = ?, last_event_json = ?
        WHERE queue_id = ? AND experiment_id = ? AND step_id = ? AND status = 'running'
        """,
        (
            _utc_now(),
            _json_text(dict(event)),
            ready.queue_id,
            ready.experiment_id,
            ready.step_id,
        ),
    )
    append_event(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
        event_type="step_process_started",
        payload=event,
    )
    conn.commit()


def _evaluate_postconditions(
    step: Mapping[str, Any],
    *,
    repo: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    failed_conditions: list[dict[str, Any]] = []
    postcondition_errors: list[dict[str, str]] = []
    for condition in [dict(condition) for condition in step.get("postconditions", [])]:
        try:
            if not _condition_passes(condition, repo_root=repo):
                failed_conditions.append(condition)
        except (ExperimentQueueError, OSError, json.JSONDecodeError) as exc:
            failed_conditions.append(condition)
            postcondition_errors.append({"condition": _json_text(condition), "error": f"{type(exc).__name__}: {exc}"})
    return failed_conditions, postcondition_errors


def _repo_rel_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _resolve_output_artifact_path(path_value: str, *, repo: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo / path


def _directory_footprint(
    path: Path,
    *,
    max_entries: int,
) -> dict[str, Any]:
    total_bytes = 0
    entry_count = 0
    truncated = False
    for root, dirs, files in os.walk(path, followlinks=False):
        entry_count += len(dirs) + len(files)
        for name in files:
            file_path = Path(root) / name
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
        if entry_count > max_entries:
            truncated = True
            break
    return {
        "recursive_bytes": total_bytes,
        "recursive_entry_count": entry_count,
        "recursive_truncated": truncated,
        "recursive_entry_limit": max_entries,
    }


def _artifact_record(
    path: Path,
    *,
    repo: Path,
    recursive: bool,
    max_entries: int,
    source: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _repo_rel_path(path, repo),
        "source": source,
        "exists": path.exists(),
    }
    try:
        stat = path.stat()
    except OSError:
        return record
    record.update(
        {
            "bytes": stat.st_size,
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
            "is_symlink": path.is_symlink(),
            "mtime_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(stat.st_mtime),
            ),
        }
    )
    if recursive and path.is_dir():
        record.update(_directory_footprint(path, max_entries=max_entries))
    return record


def _step_artifact_telemetry(
    step: Mapping[str, Any],
    *,
    repo: Path,
) -> list[dict[str, Any]]:
    telemetry = step.get("telemetry") if isinstance(step.get("telemetry"), Mapping) else {}
    recursive = bool(telemetry.get("recursive", False))
    max_entries = int(telemetry.get("max_recursive_entries") or 256)
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []

    def add(path_value: str, *, source: str, recursive_path: bool) -> None:
        path = _resolve_output_artifact_path(path_value, repo=repo)
        try:
            key = path.resolve()
        except OSError:
            key = path.absolute()
        if key in seen:
            return
        seen.add(key)
        records.append(
            _artifact_record(
                path,
                repo=repo,
                recursive=recursive_path,
                max_entries=max_entries,
                source=source,
            )
        )

    for artifact_path in _string_list(
        telemetry.get("artifact_paths"),
        "telemetry.artifact_paths",
    ):
        add(artifact_path, source="step.telemetry.artifact_paths", recursive_path=recursive)

    if bool(telemetry.get("include_postcondition_paths", True)):
        for condition in [dict(condition) for condition in step.get("postconditions", [])]:
            path_value = condition.get("path")
            if isinstance(path_value, str) and path_value:
                add(path_value, source="step.postconditions.path", recursive_path=False)
    return records


def _step_telemetry_event(
    step: Mapping[str, Any],
    *,
    repo: Path,
    log_path: Path,
) -> dict[str, Any]:
    try:
        log_bytes = log_path.stat().st_size
    except OSError:
        log_bytes = None
    artifacts = _step_artifact_telemetry(step, repo=repo)
    return {
        "schema": "experiment_queue_step_telemetry.v1",
        "log_bytes": log_bytes,
        "artifact_records": artifacts,
        "artifact_record_count": len(artifacts),
        "recursive_truncated": any(bool(record.get("recursive_truncated")) for record in artifacts),
    }


def _start_ready_step_process(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    ready: ReadyStep,
    *,
    repo_root: str | Path,
    log_root: str | Path | None = None,
) -> tuple[RunningStepProcess | None, dict[str, Any] | None]:
    repo = Path(repo_root)
    step = _lookup_step(queue, ready.experiment_id, ready.step_id)
    worker_run_id = _make_worker_run_id(ready)
    log_path = _make_step_log_path(
        repo=repo,
        log_root=log_root,
        ready=ready,
        worker_run_id=worker_run_id,
    )
    timeout_seconds = int(step.get("timeout_seconds") or 0)
    running_event = {
        "command": list(ready.command),
        "log_path": str(log_path),
        "resource_kind": ready.resource_kind,
        "worker_run_id": worker_run_id,
        "timeout_seconds": timeout_seconds,
        "parent_pid": os.getpid(),
    }
    claim_refused_reason = claim_ready_step_for_execution(
        conn,
        queue,
        ready,
        event=running_event,
    )
    if claim_refused_reason:
        return None, {
            "executed": False,
            "claim_refused": True,
            "claim_refused_reason": claim_refused_reason,
            "ready_step": ready.to_dict(),
            **running_event,
        }

    log_handle = log_path.open("wb")
    started_monotonic = time.monotonic()
    try:
        process = subprocess.Popen(
            list(ready.command),
            cwd=repo,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except OSError as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        log_handle.write(f"\n[experiment-queue] execution error: {execution_error}\n".encode())
        log_handle.close()
        event = {
            **running_event,
            "returncode": 127,
            "timed_out": False,
            "execution_error": execution_error,
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "telemetry": _step_telemetry_event(step, repo=repo, log_path=log_path),
            "failed_postconditions": [],
            "postcondition_errors": [],
        }
        _set_step_status(
            conn,
            queue_id=ready.queue_id,
            experiment_id=ready.experiment_id,
            step_id=ready.step_id,
            status="failed",
            event=event,
        )
        conn.commit()
        return None, {"executed": True, "succeeded": False, **event}

    started_event = {
        **running_event,
        "pid": process.pid,
        "started_at_utc": _utc_now(),
        "timeout_deadline_epoch_seconds": (time.time() + timeout_seconds if timeout_seconds > 0 else None),
    }
    _record_process_started(conn, ready=ready, event=started_event)
    return (
        RunningStepProcess(
            ready_step=ready,
            step=step,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            started_monotonic=started_monotonic,
            timeout_seconds=timeout_seconds,
            worker_run_id=worker_run_id,
        ),
        None,
    )


def _finalize_running_step_process(
    conn: sqlite3.Connection,
    running: RunningStepProcess,
    *,
    repo_root: str | Path,
    shutdown_grace_seconds: float = 5.0,
) -> dict[str, Any] | None:
    ready = running.ready_step
    timed_out = False
    execution_error: str | None = None
    returncode = running.process.poll()
    now = time.monotonic()
    if (
        returncode is None
        and running.timeout_seconds > 0
        and now - running.started_monotonic >= running.timeout_seconds
    ):
        timed_out = True
        running.process.kill()
        returncode = running.process.wait()
        running.log_handle.write(f"\n[experiment-queue] timeout after {running.timeout_seconds}s\n".encode())
    if (
        returncode is None
        and running.terminate_requested_monotonic is not None
        and now - running.terminate_requested_monotonic >= max(0.0, shutdown_grace_seconds)
    ):
        running.process.kill()
        returncode = running.process.wait()
        execution_error = "terminated_after_shutdown_grace"
        running.log_handle.write(b"\n[experiment-queue] killed after shutdown grace\n")
    if returncode is None:
        return None

    try:
        running.log_handle.close()
    except OSError:
        pass
    elapsed = time.monotonic() - running.started_monotonic
    failed_conditions, postcondition_errors = _evaluate_postconditions(
        running.step,
        repo=Path(repo_root),
    )
    succeeded = returncode == 0 and not timed_out and not execution_error and not failed_conditions
    telemetry = _step_telemetry_event(
        running.step,
        repo=Path(repo_root),
        log_path=running.log_path,
    )
    event = {
        "command": list(ready.command),
        "pid": running.process.pid,
        "resource_kind": ready.resource_kind,
        "worker_run_id": running.worker_run_id,
        "returncode": 124 if timed_out else returncode,
        "timed_out": timed_out,
        "execution_error": execution_error,
        "elapsed_seconds": elapsed,
        "log_path": str(running.log_path),
        "telemetry": telemetry,
        "failed_postconditions": failed_conditions,
        "postcondition_errors": postcondition_errors,
    }
    _set_step_status(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
        status="succeeded" if succeeded else "failed",
        event=event,
    )
    conn.commit()
    return {"executed": True, "succeeded": succeeded, **event}


def _request_running_process_termination(
    running: RunningStepProcess,
    *,
    reason: str,
) -> None:
    if running.process.poll() is not None:
        return
    if running.terminate_requested_monotonic is not None:
        return
    running.terminate_requested_monotonic = time.monotonic()
    try:
        running.log_handle.write(f"\n[experiment-queue] terminate requested: {reason}\n".encode())
        running.log_handle.flush()
    except OSError:
        pass
    try:
        running.process.terminate()
    except OSError:
        return


def _active_step_keys(queue: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(experiment["id"]), str(step["id"])) for experiment in queue["experiments"] for step in experiment["steps"]
    }


def orphaned_step_rows(conn: sqlite3.Connection, queue: Mapping[str, Any]) -> list[sqlite3.Row]:
    queue_id = str(queue["queue_id"])
    active_keys = _active_step_keys(queue)
    rows = conn.execute(
        """
        SELECT experiment_id, step_id, status, attempts, updated_at_utc, last_event_json
        FROM step_state WHERE queue_id = ?
        ORDER BY experiment_id, step_id
        """,
        (queue_id,),
    ).fetchall()
    return [row for row in rows if (str(row["experiment_id"]), str(row["step_id"])) not in active_keys]


def assert_no_orphaned_steps_for_execution(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    allow_orphaned_state: bool = False,
) -> None:
    if allow_orphaned_state:
        return
    orphaned = [row for row in orphaned_step_rows(conn, queue) if str(row["status"]) in BLOCKING_ORPHAN_STATUSES]
    if orphaned:
        preview = ", ".join(f"{row['experiment_id']}.{row['step_id']}={row['status']}" for row in orphaned[:5])
        suffix = "" if len(orphaned) <= 5 else f", ... +{len(orphaned) - 5} more"
        raise ExperimentQueueError(
            f"queue {queue['queue_id']!r} state has {len(orphaned)} orphaned step row(s): "
            f"{preview}{suffix}; use a fresh canonical state, explicitly rewind/retire stale rows, "
            "or pass --orphaned-state-rationale for an isolated audited run"
        )


def retire_orphaned_steps(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    reason: str,
) -> list[dict[str, Any]]:
    queue_id = str(queue["queue_id"])
    retired: list[dict[str, Any]] = []
    now = _utc_now()
    for row in orphaned_step_rows(conn, queue):
        status = str(row["status"])
        if status not in BLOCKING_ORPHAN_STATUSES:
            continue
        if status == "running":
            raise ExperimentQueueError(
                f"refusing to retire running orphaned step {row['experiment_id']}.{row['step_id']}"
            )
        event = {
            "reason": reason,
            "previous_status": status,
            "orphaned_by_queue_definition": str(queue_id),
        }
        conn.execute(
            """
            UPDATE step_state
            SET status = 'skipped', updated_at_utc = ?, last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                now,
                _json_text(event),
                queue_id,
                row["experiment_id"],
                row["step_id"],
            ),
        )
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=str(row["experiment_id"]),
            step_id=str(row["step_id"]),
            event_type="step_retired_orphan",
            payload=event,
        )
        retired.append(
            {
                "experiment_id": str(row["experiment_id"]),
                "step_id": str(row["step_id"]),
                "previous_status": status,
                "new_status": "skipped",
            }
        )
    conn.commit()
    return retired


def run_queue_worker(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    repo_root: str | Path,
    execute: bool,
    max_steps: int,
    idle_sleep_seconds: float = 5.0,
    max_idle_cycles: int = 1,
    max_parallel: int | None = 0,
    poll_interval_seconds: float = 0.25,
    stop_policy: str = "drain",
    shutdown_grace_seconds: float = 5.0,
    allow_cloud: bool = False,
    allow_orphaned_state: bool = False,
    orphaned_state_rationale: str | None = None,
    noncanonical_state_rationale: str | None = None,
    log_root: str | Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    reload_queue: Callable[[], Mapping[str, Any]] | None = None,
    max_experiments: int | None = None,
) -> dict[str, Any]:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ExperimentQueueError("max_steps must be a non-negative integer")
    if isinstance(max_idle_cycles, bool) or not isinstance(max_idle_cycles, int) or max_idle_cycles < 0:
        raise ExperimentQueueError("max_idle_cycles must be a non-negative integer")
    if idle_sleep_seconds < 0:
        raise ExperimentQueueError("idle_sleep_seconds must be non-negative")
    if max_parallel is not None and (
        isinstance(max_parallel, bool) or not isinstance(max_parallel, int) or max_parallel < 0
    ):
        raise ExperimentQueueError("max_parallel must be a non-negative integer or None for auto")
    if poll_interval_seconds < 0:
        raise ExperimentQueueError("poll_interval_seconds must be non-negative")
    if stop_policy not in {"drain", "terminate"}:
        raise ExperimentQueueError("stop_policy must be 'drain' or 'terminate'")
    if shutdown_grace_seconds < 0:
        raise ExperimentQueueError("shutdown_grace_seconds must be non-negative")
    if max_experiments is not None and (
        isinstance(max_experiments, bool) or not isinstance(max_experiments, int) or max_experiments <= 0
    ):
        raise ExperimentQueueError("max_experiments must be a positive integer or None")

    queue_id = str(queue["queue_id"])
    active_queue: Mapping[str, Any] = queue
    requested_max_parallel = max_parallel
    max_parallel, resource_limits = resolve_worker_max_parallel(
        active_queue,
        requested_max_parallel,
        allow_cloud=allow_cloud,
    )
    stale_running_reconciliations: list[dict[str, Any]] = []

    def _active_queue() -> Mapping[str, Any]:
        nonlocal active_queue, max_parallel, resource_limits
        if reload_queue is None:
            return active_queue
        refreshed = reload_queue()
        if str(refreshed["queue_id"]) != queue_id:
            raise ExperimentQueueError(f"reloaded queue_id changed from {queue_id!r} to {refreshed['queue_id']!r}")
        initialize_queue_state(conn, refreshed)
        assert_no_orphaned_steps_for_execution(conn, refreshed, allow_orphaned_state=allow_orphaned_state)
        if execute:
            reconciliation = reconcile_stale_running_steps(
                conn,
                refreshed,
                repo_root=repo_root,
            )
            if reconciliation["reconciled_step_count"]:
                stale_running_reconciliations.append(reconciliation)
        active_queue = refreshed
        if requested_max_parallel in (None, 0):
            max_parallel, resource_limits = resolve_worker_max_parallel(
                active_queue,
                requested_max_parallel,
                allow_cloud=allow_cloud,
            )
        else:
            resource_limits = worker_resource_limits(active_queue, allow_cloud=allow_cloud)
        return active_queue

    assert_no_orphaned_steps_for_execution(conn, active_queue, allow_orphaned_state=allow_orphaned_state)
    if execute:
        reconciliation = reconcile_stale_running_steps(
            conn,
            active_queue,
            repo_root=repo_root,
        )
        if reconciliation["reconciled_step_count"]:
            stale_running_reconciliations.append(reconciliation)
    started_at = _utc_now()
    append_event(
        conn,
        queue_id=queue_id,
        experiment_id=None,
        step_id=None,
        event_type="worker_started",
        payload={
            "execute": execute,
            "max_steps": max_steps,
            "requested_max_parallel": requested_max_parallel,
            "max_parallel": max_parallel,
            "resource_limits": resource_limits,
            "max_idle_cycles": max_idle_cycles,
            "idle_sleep_seconds": idle_sleep_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "stop_policy": stop_policy,
            "shutdown_grace_seconds": shutdown_grace_seconds,
            "allow_cloud": allow_cloud,
            "allow_orphaned_state": allow_orphaned_state,
            "orphaned_state_rationale": orphaned_state_rationale,
            "noncanonical_state_rationale": noncanonical_state_rationale,
            "max_experiments": max_experiments,
            "stale_running_reconciliations": stale_running_reconciliations,
        },
    )
    conn.commit()

    worker_experiment_ids: set[str] = set()

    def _eligible_ready_steps() -> list[ReadyStep]:
        ready = ready_steps(
            conn,
            _active_queue(),
            allow_cloud=allow_cloud,
            repo_root=repo_root,
        )
        if max_experiments is None:
            return ready
        return [
            step
            for step in ready
            if step.experiment_id in worker_experiment_ids or len(worker_experiment_ids) < max_experiments
        ]

    if not execute:
        selected: list[dict[str, object]] = []
        planned = _eligible_ready_steps()
        if max_steps:
            for step in planned:
                if len(selected) >= max_steps:
                    break
                if max_experiments is not None and step.experiment_id not in worker_experiment_ids:
                    if len(worker_experiment_ids) >= max_experiments:
                        continue
                    worker_experiment_ids.add(step.experiment_id)
                selected.append(step.to_dict())
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=None,
            step_id=None,
            event_type="worker_planned",
            payload={"selected_steps": selected, "ready_step_count": len(planned)},
        )
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=None,
            step_id=None,
            event_type="worker_stopped",
            payload={"stop_reason": "dry_run", "steps_started": 0},
        )
        conn.commit()
        return {
            "schema": "experiment_queue_worker_result.v1",
            "queue_id": queue_id,
            "execute": False,
            "allow_cloud": allow_cloud,
            "max_parallel": max_parallel,
            "requested_max_parallel": requested_max_parallel,
            "resource_limits": resource_limits,
            "max_experiments": max_experiments,
            "started_experiment_ids": sorted(worker_experiment_ids),
            "orphaned_state_rationale": orphaned_state_rationale,
            "noncanonical_state_rationale": noncanonical_state_rationale,
            "started_at_utc": started_at,
            "stop_reason": "dry_run",
            "steps_started": 0,
            "success_count": 0,
            "failure_count": 0,
            "claim_refused_count": 0,
            "stale_running_reconciliations": stale_running_reconciliations,
            "idle_cycles": 0,
            "planned_steps": selected,
            "step_results": [],
        }

    step_results: list[dict[str, Any]] = []
    steps_started = 0
    success_count = 0
    failure_count = 0
    claim_refused_count = 0
    idle_cycles = 0
    stop_reason = "max_steps_reached"
    running_processes: list[RunningStepProcess] = []
    stop_seen = False

    def _notice_stop_requested() -> None:
        nonlocal stop_reason, stop_seen
        if stop_requested is None or not stop_requested() or stop_seen:
            return
        stop_reason = "stop_requested"
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=None,
            step_id=None,
            event_type="worker_stop_requested",
            payload={
                "steps_started": steps_started,
                "running_count": len(running_processes),
                "stop_policy": stop_policy,
            },
        )
        conn.commit()
        if stop_policy == "terminate":
            for running in running_processes:
                _request_running_process_termination(
                    running,
                    reason="worker_stop_requested",
                )
        stop_seen = True

    while steps_started < max_steps or running_processes:
        _notice_stop_requested()

        for running in list(running_processes):
            result = _finalize_running_step_process(
                conn,
                running,
                repo_root=repo_root,
                shutdown_grace_seconds=shutdown_grace_seconds,
            )
            if result is None:
                continue
            running_processes.remove(running)
            ready_step = running.ready_step
            step_result = {"ready_step": ready_step.to_dict(), **result}
            step_results.append(step_result)
            if result.get("succeeded"):
                success_count += 1
                continue

            failure_count += 1
            append_event(
                conn,
                queue_id=queue_id,
                experiment_id=ready_step.experiment_id,
                step_id=ready_step.step_id,
                event_type="worker_step_failed",
                payload=step_result,
            )
            conn.commit()

        _notice_stop_requested()

        if stop_seen:
            if not running_processes:
                break
            if poll_interval_seconds:
                time.sleep(poll_interval_seconds)
            continue

        if steps_started >= max_steps:
            if running_processes:
                if poll_interval_seconds:
                    time.sleep(poll_interval_seconds)
                continue
            break

        launched_this_cycle = 0
        while steps_started < max_steps and len(running_processes) < max_parallel:
            ready = _eligible_ready_steps()
            if not ready:
                break

            idle_cycles = 0
            ready_step = ready[0]
            running, immediate_result = _start_ready_step_process(
                conn,
                active_queue,
                ready_step,
                repo_root=repo_root,
                log_root=log_root,
            )
            if immediate_result is not None:
                step_result = {"ready_step": ready_step.to_dict(), **immediate_result}
                step_results.append(step_result)
                if immediate_result.get("claim_refused"):
                    claim_refused_count += 1
                    continue
                worker_experiment_ids.add(ready_step.experiment_id)
                steps_started += 1
                launched_this_cycle += 1
                failure_count += 1
                append_event(
                    conn,
                    queue_id=queue_id,
                    experiment_id=ready_step.experiment_id,
                    step_id=ready_step.step_id,
                    event_type="worker_step_failed",
                    payload=step_result,
                )
                conn.commit()
                continue
            if running is not None:
                worker_experiment_ids.add(ready_step.experiment_id)
                running_processes.append(running)
                steps_started += 1
                launched_this_cycle += 1

        if running_processes:
            if poll_interval_seconds:
                time.sleep(poll_interval_seconds)
            continue

        if launched_this_cycle == 0:
            idle_cycles += 1
            append_event(
                conn,
                queue_id=queue_id,
                experiment_id=None,
                step_id=None,
                event_type="worker_idle",
                payload={"idle_cycle": idle_cycles, "max_idle_cycles": max_idle_cycles},
            )
            conn.commit()
            if idle_cycles >= max_idle_cycles:
                stop_reason = "idle_limit_reached"
                break
            time.sleep(idle_sleep_seconds)
            continue

    append_event(
        conn,
        queue_id=queue_id,
        experiment_id=None,
        step_id=None,
        event_type="worker_stopped",
        payload={
            "stop_reason": stop_reason,
            "steps_started": steps_started,
            "requested_max_parallel": requested_max_parallel,
            "max_parallel": max_parallel,
            "resource_limits": resource_limits,
            "max_experiments": max_experiments,
            "started_experiment_ids": sorted(worker_experiment_ids),
            "success_count": success_count,
            "failure_count": failure_count,
            "claim_refused_count": claim_refused_count,
            "stale_running_reconciliations": stale_running_reconciliations,
            "idle_cycles": idle_cycles,
        },
    )
    conn.commit()
    return {
        "schema": "experiment_queue_worker_result.v1",
        "queue_id": queue_id,
        "execute": True,
        "allow_cloud": allow_cloud,
        "max_parallel": max_parallel,
        "requested_max_parallel": requested_max_parallel,
        "resource_limits": resource_limits,
        "max_experiments": max_experiments,
        "started_experiment_ids": sorted(worker_experiment_ids),
        "orphaned_state_rationale": orphaned_state_rationale,
        "noncanonical_state_rationale": noncanonical_state_rationale,
        "started_at_utc": started_at,
        "stop_reason": stop_reason,
        "steps_started": steps_started,
        "success_count": success_count,
        "failure_count": failure_count,
        "claim_refused_count": claim_refused_count,
        "stale_running_reconciliations": stale_running_reconciliations,
        "idle_cycles": idle_cycles,
        "planned_steps": [],
        "step_results": step_results,
    }


def _downstream_step_ids(
    queue: Mapping[str, Any],
    experiment_id: str,
    step_id: str,
) -> list[tuple[str, str]]:
    return _downstream_step_ids_from_experiments(
        queue["experiments"],
        experiment_id,
        step_id,
    )


def rewind_step(
    conn: sqlite3.Connection,
    queue_id: str,
    experiment_id: str,
    step_id: str,
    *,
    reason: str = "",
    queue: Mapping[str, Any] | None = None,
    cascade: bool = True,
) -> None:
    downstream = _downstream_step_ids(queue, experiment_id, step_id) if queue and cascade else []
    for current_experiment_id, current_step_id in [(experiment_id, step_id), *downstream]:
        event = {
            "reason": reason,
            "root_experiment_id": experiment_id,
            "root_step_id": step_id,
            "cascade": cascade,
            "cascade_rewound": (current_experiment_id, current_step_id) != (experiment_id, step_id),
        }
        conn.execute(
            """
            UPDATE step_state
            SET status = 'queued', updated_at_utc = ?, last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (_utc_now(), _json_text(event), queue_id, current_experiment_id, current_step_id),
        )
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=current_experiment_id,
            step_id=current_step_id,
            event_type="step_rewound",
            payload=event,
        )
    conn.commit()


def queue_summary(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    queue_id = str(queue["queue_id"])
    active_keys = _active_step_keys(queue)
    rows = conn.execute(
        """
        SELECT experiment_id, step_id, status, attempts, updated_at_utc, last_event_json
        FROM step_state WHERE queue_id = ?
        ORDER BY experiment_id, step_id
        """,
        (queue_id,),
    ).fetchall()
    active_rows = [row for row in rows if (str(row["experiment_id"]), str(row["step_id"])) in active_keys]
    orphaned_rows = [row for row in rows if (str(row["experiment_id"]), str(row["step_id"])) not in active_keys]
    by_status: dict[str, int] = {}
    for row in active_rows:
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + 1
    return {
        "schema": "experiment_queue_summary.v1",
        "queue_id": queue_id,
        "mode": control_mode(conn, queue_id),
        "step_count": len(active_rows),
        "orphaned_step_count": len(orphaned_rows),
        "status_counts": by_status,
        "ready_steps": [step.to_dict() for step in ready_steps(conn, queue, repo_root=repo_root)],
        "steps": [
            {
                "experiment_id": row["experiment_id"],
                "step_id": row["step_id"],
                "status": row["status"],
                "attempts": row["attempts"],
                "updated_at_utc": row["updated_at_utc"],
                "last_event": json.loads(row["last_event_json"]) if row["last_event_json"] else None,
            }
            for row in active_rows
        ],
        "orphaned_steps": [
            {
                "experiment_id": row["experiment_id"],
                "step_id": row["step_id"],
                "status": row["status"],
                "attempts": row["attempts"],
                "updated_at_utc": row["updated_at_utc"],
                "last_event": json.loads(row["last_event_json"]) if row["last_event_json"] else None,
            }
            for row in orphaned_rows
        ],
    }


def reconcile_satisfied_queued_steps(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Mark queued steps succeeded when their declared postconditions already pass.

    This is intentionally artifact-backed: it only advances queued steps that
    have at least one postcondition and whose current queue definition passes
    every postcondition. It refreshes definition hashes at the same time, so
    harmless timeout/resource/telemetry edits do not force redundant reruns.
    """

    repo = Path(repo_root)
    queue_id = str(queue["queue_id"])
    now = _utc_now()
    reconciled: list[dict[str, Any]] = []
    inspected = 0
    dependency_blocked: list[dict[str, Any]] = []
    for experiment in queue["experiments"]:
        experiment_metadata = dict(experiment.get("metadata") or {})
        for step in experiment["steps"]:
            row = conn.execute(
                """
                SELECT status, attempts
                FROM step_state
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (queue_id, experiment["id"], step["id"]),
            ).fetchone()
            if row is None or str(row["status"]) != "queued":
                continue
            postconditions = [dict(item) for item in step.get("postconditions", [])]
            if not postconditions:
                continue
            blockers = _dependency_reconciliation_blockers(
                conn,
                queue,
                str(experiment["id"]),
                step,
                repo=repo,
            )
            if blockers:
                dependency_blocked.append(
                    {
                        "experiment_id": str(experiment["id"]),
                        "step_id": str(step["id"]),
                        "blockers": blockers,
                    }
                )
                continue
            inspected += 1
            failed_conditions, postcondition_errors = _evaluate_postconditions(
                step,
                repo=repo,
            )
            if failed_conditions or postcondition_errors:
                continue
            hashes = _step_hashes(step, experiment_metadata=experiment_metadata)
            event = {
                "previous_status": str(row["status"]),
                "reconcile_reason": "queued_step_postconditions_already_satisfied",
                "postcondition_count": len(postconditions),
                "attempts": int(row["attempts"] or 0),
                "hashes": hashes,
            }
            conn.execute(
                """
                UPDATE step_state
                SET status = 'succeeded', updated_at_utc = ?, last_event_json = ?,
                    definition_hash = ?, command_hash = ?, postcondition_hash = ?,
                    resource_kind = ?
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (
                    now,
                    _json_text(event),
                    hashes["definition_hash"],
                    hashes["command_hash"],
                    hashes["postcondition_hash"],
                    _resource_kind(step),
                    queue_id,
                    experiment["id"],
                    step["id"],
                ),
            )
            append_event(
                conn,
                queue_id=queue_id,
                experiment_id=str(experiment["id"]),
                step_id=str(step["id"]),
                event_type="step_reconciled_succeeded_from_postconditions",
                payload=event,
            )
            reconciled.append(
                {
                    "experiment_id": str(experiment["id"]),
                    "step_id": str(step["id"]),
                    "postcondition_count": len(postconditions),
                    "previous_status": str(row["status"]),
                }
            )
    conn.commit()
    return {
        "schema": "experiment_queue_postcondition_reconciliation.v1",
        "queue_id": queue_id,
        "inspected_queued_postcondition_steps": inspected,
        "reconciled_step_count": len(reconciled),
        "reconciled_steps": reconciled,
        "dependency_blocked_step_count": len(dependency_blocked),
        "dependency_blocked_steps": dependency_blocked,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _dependency_reconciliation_blockers(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    experiment_id: str,
    step: Mapping[str, Any],
    *,
    repo: Path,
) -> list[str]:
    blockers: list[str] = []
    step_lookup = _step_lookup_by_key(queue)
    for required in step.get("requires") or []:
        dep_experiment_id, dep_step_id = _resolve_step_ref(
            str(required),
            default_experiment_id=experiment_id,
        )
        dep_row = conn.execute(
            """
            SELECT status
            FROM step_state
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (queue["queue_id"], dep_experiment_id, dep_step_id),
        ).fetchone()
        if dep_row is None:
            blockers.append(f"dependency_missing:{dep_experiment_id}.{dep_step_id}")
            continue
        if str(dep_row["status"]) != "succeeded":
            blockers.append(f"dependency_not_succeeded:{dep_experiment_id}.{dep_step_id}:{dep_row['status']}")
            continue
        dep_step = step_lookup.get((dep_experiment_id, dep_step_id))
        if dep_step is None:
            blockers.append(f"dependency_not_in_queue_definition:{dep_experiment_id}.{dep_step_id}")
            continue
        failed_conditions, postcondition_errors = _evaluate_postconditions(dep_step, repo=repo)
        if failed_conditions or postcondition_errors:
            blockers.append(f"dependency_postconditions_not_satisfied:{dep_experiment_id}.{dep_step_id}")
    return blockers


def _step_lookup_by_key(
    queue: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(experiment["id"]), str(step["id"])): step
        for experiment in queue["experiments"]
        for step in experiment["steps"]
    }


def _experiment_lookup_by_id(queue: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(experiment["id"]): experiment for experiment in queue["experiments"]}


def _metadata_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _metadata_string_list(value: object) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _performance_candidate_ids(
    experiment_id: str,
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    for key in (
        "candidate_ids",
        "source_candidate_ids",
        "source_unit_ids",
        "source_selection_ids",
    ):
        values = _metadata_string_list(metadata.get(key))
        if values:
            return tuple(values)
    for key in ("work_id", "backlog_key"):
        value = _metadata_string(metadata.get(key))
        if value is not None:
            return (value,)
    return (experiment_id,)


def _add_performance_identity(
    bucket: dict[str, Any],
    *,
    experiment_id: str,
    step_id: str,
    metadata: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> None:
    bucket.setdefault("_experiment_ids", set()).add(experiment_id)
    bucket.setdefault("_step_ids", set()).add(step_id)
    bucket.setdefault("_candidate_ids", set()).update(candidate_ids)
    for metadata_key, bucket_key in (
        ("work_id", "_work_ids"),
        ("backlog_key", "_backlog_keys"),
        ("source_unit_ids", "_source_unit_ids"),
        ("source_selection_ids", "_source_selection_ids"),
    ):
        values = _metadata_string_list(metadata.get(metadata_key))
        if values:
            bucket.setdefault(bucket_key, set()).update(values)


def _new_performance_bucket() -> dict[str, Any]:
    return {
        "run_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "resource_kind_counts": {},
        "elapsed_seconds_sum": 0.0,
        "elapsed_seconds_min": None,
        "elapsed_seconds_max": None,
        "artifact_record_count": 0,
        "artifact_record_bytes_sum": 0,
        "artifact_record_raw_bytes_sum": 0,
        "log_bytes_sum": 0,
    }


def _add_performance_sample(
    bucket: dict[str, Any],
    *,
    succeeded: bool,
    resource_kind: str,
    elapsed_seconds: float | None,
    artifact_count: int,
    artifact_bytes: int,
    artifact_raw_bytes: int,
    log_bytes: int | None,
) -> None:
    bucket["run_count"] += 1
    if succeeded:
        bucket["success_count"] += 1
    else:
        bucket["failure_count"] += 1
    resource_counts = bucket.setdefault("resource_kind_counts", {})
    resource_counts[resource_kind] = int(resource_counts.get(resource_kind, 0)) + 1
    if elapsed_seconds is not None:
        bucket["elapsed_seconds_sum"] += elapsed_seconds
        current_min = bucket["elapsed_seconds_min"]
        current_max = bucket["elapsed_seconds_max"]
        bucket["elapsed_seconds_min"] = elapsed_seconds if current_min is None else min(current_min, elapsed_seconds)
        bucket["elapsed_seconds_max"] = elapsed_seconds if current_max is None else max(current_max, elapsed_seconds)
    bucket["artifact_record_count"] += artifact_count
    bucket["artifact_record_bytes_sum"] += artifact_bytes
    bucket["artifact_record_raw_bytes_sum"] += artifact_raw_bytes
    if log_bytes is not None:
        bucket["log_bytes_sum"] += log_bytes


def _finalize_performance_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    run_count = int(bucket["run_count"])
    elapsed_sum = float(bucket["elapsed_seconds_sum"])
    resource_counts = {str(key): int(value) for key, value in dict(bucket.get("resource_kind_counts") or {}).items()}
    dominant_resource_kind = None
    if resource_counts:
        dominant_resource_kind = sorted(
            resource_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    result = {
        **{key: value for key, value in bucket.items() if not key.startswith("_")},
        "resource_kind_counts": resource_counts,
        "dominant_resource_kind": dominant_resource_kind,
        "elapsed_seconds_mean": elapsed_sum / run_count if run_count else None,
        "artifact_record_bytes_mean": (int(bucket["artifact_record_bytes_sum"]) / run_count if run_count else None),
        "artifact_record_raw_bytes_mean": (
            int(bucket["artifact_record_raw_bytes_sum"]) / run_count if run_count else None
        ),
        "log_bytes_mean": int(bucket["log_bytes_sum"]) / run_count if run_count else None,
    }
    for internal_key, public_key in (
        ("_experiment_ids", "experiment_ids"),
        ("_step_ids", "step_ids"),
        ("_candidate_ids", "candidate_ids"),
        ("_work_ids", "work_ids"),
        ("_backlog_keys", "backlog_keys"),
        ("_source_unit_ids", "source_unit_ids"),
        ("_source_selection_ids", "source_selection_ids"),
    ):
        values = bucket.get(internal_key)
        if values:
            result[public_key] = sorted(str(value) for value in values)
    return result


def _telemetry_artifact_bytes(telemetry: Mapping[str, Any]) -> tuple[int, int, int]:
    records = telemetry.get("artifact_records")
    if not isinstance(records, list):
        return 0, 0, 0
    normalized_records = [record for record in records if isinstance(record, Mapping)]
    recursive_dirs: list[str] = []
    for record in normalized_records:
        if bool(record.get("is_dir")) and _finite_int(record.get("recursive_bytes")) is not None:
            path = _normalized_telemetry_record_path(record)
            if path is not None:
                recursive_dirs.append(path)
    total = 0
    raw_total = 0
    count = 0
    for record in normalized_records:
        count += 1
        value = record.get("recursive_bytes", record.get("bytes"))
        parsed = _finite_int(value)
        if parsed is not None:
            raw_total += parsed
        path = _normalized_telemetry_record_path(record)
        if (
            path is not None
            and not bool(record.get("is_dir"))
            and any(_telemetry_path_inside(path, root) for root in recursive_dirs)
        ):
            continue
        if (
            path is not None
            and bool(record.get("is_dir"))
            and any(root != path and _telemetry_path_inside(path, root) for root in recursive_dirs)
        ):
            continue
        if parsed is not None:
            total += parsed
    return count, total, raw_total


def _normalized_telemetry_record_path(record: Mapping[str, Any]) -> str | None:
    path = record.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    return normpath(path.strip())


def _telemetry_path_inside(path: str, root: str) -> bool:
    if path == root:
        return False
    return path.startswith(f"{root.rstrip('/')}/")


def queue_performance_summary(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate completed step telemetry for acquisition and resource scheduling."""

    queue_id = str(queue["queue_id"])
    step_lookup = _step_lookup_by_key(queue)
    experiment_lookup = _experiment_lookup_by_id(queue)
    by_resource: dict[str, dict[str, Any]] = {}
    by_experiment: dict[str, dict[str, Any]] = {}
    by_step: dict[str, dict[str, Any]] = {}
    by_work_id: dict[str, dict[str, Any]] = {}
    by_backlog_key: dict[str, dict[str, Any]] = {}
    by_source_unit_id: dict[str, dict[str, Any]] = {}
    by_source_selection_id: dict[str, dict[str, Any]] = {}
    candidate_id_by_experiment: dict[str, list[str]] = {}
    work_id_by_experiment: dict[str, str] = {}
    backlog_key_by_experiment: dict[str, str] = {}
    source_unit_ids_by_experiment: dict[str, list[str]] = {}
    source_selection_ids_by_experiment: dict[str, list[str]] = {}
    rows = conn.execute(
        """
        SELECT experiment_id, step_id, event_type, payload_json
        FROM queue_events
        WHERE queue_id = ? AND event_type IN ('step_succeeded', 'step_failed')
        ORDER BY id
        """,
        (queue_id,),
    ).fetchall()
    for row in rows:
        experiment_id = str(row["experiment_id"])
        step_id = str(row["step_id"])
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        step = step_lookup.get((experiment_id, step_id), {})
        experiment = experiment_lookup.get(experiment_id, {})
        metadata = experiment.get("metadata") if isinstance(experiment.get("metadata"), Mapping) else {}
        candidate_ids = _performance_candidate_ids(experiment_id, metadata)
        candidate_id_by_experiment.setdefault(experiment_id, list(candidate_ids))
        work_id = _metadata_string(metadata.get("work_id"))
        if work_id is not None:
            work_id_by_experiment.setdefault(experiment_id, work_id)
        backlog_key = _metadata_string(metadata.get("backlog_key"))
        if backlog_key is not None:
            backlog_key_by_experiment.setdefault(experiment_id, backlog_key)
        source_unit_ids = _metadata_string_list(metadata.get("source_unit_ids"))
        if source_unit_ids:
            source_unit_ids_by_experiment.setdefault(experiment_id, source_unit_ids)
        source_selection_ids = _metadata_string_list(metadata.get("source_selection_ids"))
        if source_selection_ids:
            source_selection_ids_by_experiment.setdefault(
                experiment_id,
                source_selection_ids,
            )
        resource_kind = str(payload.get("resource_kind") or _resource_kind(step))
        telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), Mapping) else {}
        artifact_count, artifact_bytes, artifact_raw_bytes = _telemetry_artifact_bytes(telemetry)
        log_bytes = _finite_int(telemetry.get("log_bytes")) if telemetry else None
        elapsed = _finite_float(payload.get("elapsed_seconds"))
        succeeded = str(row["event_type"]) == "step_succeeded"
        for bucket_map, key in (
            (by_resource, resource_kind),
            (by_experiment, experiment_id),
            (by_step, f"{experiment_id}.{step_id}"),
        ):
            bucket = bucket_map.setdefault(key, _new_performance_bucket())
            _add_performance_identity(
                bucket,
                experiment_id=experiment_id,
                step_id=step_id,
                metadata=metadata,
                candidate_ids=candidate_ids,
            )
            _add_performance_sample(
                bucket,
                succeeded=succeeded,
                resource_kind=resource_kind,
                elapsed_seconds=elapsed,
                artifact_count=artifact_count,
                artifact_bytes=artifact_bytes,
                artifact_raw_bytes=artifact_raw_bytes,
                log_bytes=log_bytes,
            )
        for bucket_map, keys in (
            (by_work_id, [work_id] if work_id is not None else []),
            (by_backlog_key, [backlog_key] if backlog_key is not None else []),
            (by_source_unit_id, source_unit_ids),
            (by_source_selection_id, source_selection_ids),
        ):
            for key in keys:
                bucket = bucket_map.setdefault(key, _new_performance_bucket())
                _add_performance_identity(
                    bucket,
                    experiment_id=experiment_id,
                    step_id=step_id,
                    metadata=metadata,
                    candidate_ids=candidate_ids,
                )
                _add_performance_sample(
                    bucket,
                    succeeded=succeeded,
                    resource_kind=resource_kind,
                    elapsed_seconds=elapsed,
                    artifact_count=artifact_count,
                    artifact_bytes=artifact_bytes,
                    artifact_raw_bytes=artifact_raw_bytes,
                    log_bytes=log_bytes,
                )
    return {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": queue_id,
        "telemetry_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "event_count": len(rows),
        "candidate_id_by_experiment": dict(sorted(candidate_id_by_experiment.items())),
        "work_id_by_experiment": dict(sorted(work_id_by_experiment.items())),
        "backlog_key_by_experiment": dict(sorted(backlog_key_by_experiment.items())),
        "source_unit_ids_by_experiment": dict(sorted(source_unit_ids_by_experiment.items())),
        "source_selection_ids_by_experiment": dict(sorted(source_selection_ids_by_experiment.items())),
        "by_resource_kind": {key: _finalize_performance_bucket(value) for key, value in sorted(by_resource.items())},
        "by_experiment": {key: _finalize_performance_bucket(value) for key, value in sorted(by_experiment.items())},
        "by_step": {key: _finalize_performance_bucket(value) for key, value in sorted(by_step.items())},
        "by_work_id": {key: _finalize_performance_bucket(value) for key, value in sorted(by_work_id.items())},
        "by_backlog_key": {key: _finalize_performance_bucket(value) for key, value in sorted(by_backlog_key.items())},
        "by_source_unit_id": {
            key: _finalize_performance_bucket(value) for key, value in sorted(by_source_unit_id.items())
        },
        "by_source_selection_id": {
            key: _finalize_performance_bucket(value) for key, value in sorted(by_source_selection_id.items())
        },
    }


def derive_scheduler_runtime_policy(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    cpu_count: int | None = None,
    timeout_multiplier: float = 3.0,
    min_timeout_seconds: int = 30,
    max_timeout_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    """Build an advisory local runtime policy from queue telemetry.

    The policy is an execution-control hint only. It may drive local worker
    sizing and timeout envelopes, but it cannot claim score, promotion, ranking,
    kill, or exact-dispatch authority.
    """

    if isinstance(cpu_count, bool):
        raise ExperimentQueueError("cpu_count must be an integer or None")
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if cpu_count < 1:
        raise ExperimentQueueError("cpu_count must be positive")
    if timeout_multiplier <= 0:
        raise ExperimentQueueError("timeout_multiplier must be positive")
    if min_timeout_seconds < 0:
        raise ExperimentQueueError("min_timeout_seconds must be non-negative")
    if max_timeout_seconds < 1:
        raise ExperimentQueueError("max_timeout_seconds must be positive")
    if min_timeout_seconds > max_timeout_seconds:
        raise ExperimentQueueError("min_timeout_seconds must be <= max_timeout_seconds")

    queue_id = str(queue["queue_id"])
    controls = dict(queue.get("controls") or {})
    current_concurrency: dict[str, int] = {}
    for kind, limit in dict(controls.get("max_concurrency") or {}).items():
        parsed_limit = _finite_int(limit)
        if parsed_limit is not None and parsed_limit >= 0:
            current_concurrency[str(kind)] = parsed_limit
    performance = queue_performance_summary(conn, queue)
    samples = _runtime_policy_event_samples(conn, queue)
    resource_kinds = sorted(set(queue_resource_kinds(queue)) | set(samples))
    resource_policies: dict[str, dict[str, Any]] = {}
    recommended_concurrency: dict[str, int] = {}
    recommended_timeouts: dict[str, int] = {}
    for kind in resource_kinds:
        current_limit = max(0, int(current_concurrency.get(kind, 1)))
        sample_rows = samples.get(kind, [])
        elapsed_values = [
            value
            for value in (_finite_float(row.get("elapsed_seconds")) for row in sample_rows)
            if value is not None and value >= 0
        ]
        timeout_count = sum(1 for row in sample_rows if row.get("timed_out") is True)
        failure_count = sum(1 for row in sample_rows if row.get("succeeded") is not True)
        recommended_limit, concurrency_reason = _recommended_concurrency_for_resource(
            kind,
            current_limit=current_limit,
            cpu_count=cpu_count,
            timeout_count=timeout_count,
            failure_count=failure_count,
            run_count=len(sample_rows),
        )
        recommended_timeout = _recommended_timeout_for_resource(
            elapsed_values,
            timeout_count=timeout_count,
            current_timeouts=_step_timeouts_for_resource(queue, kind),
            timeout_multiplier=timeout_multiplier,
            min_timeout_seconds=min_timeout_seconds,
            max_timeout_seconds=max_timeout_seconds,
        )
        recommended_concurrency[kind] = recommended_limit
        if recommended_timeout is not None:
            recommended_timeouts[kind] = recommended_timeout
        resource_policies[kind] = {
            "resource_kind": kind,
            "current_concurrency": current_limit,
            "recommended_concurrency": recommended_limit,
            "concurrency_reason": concurrency_reason,
            "observed_run_count": len(sample_rows),
            "observed_failure_count": failure_count,
            "observed_timeout_count": timeout_count,
            "elapsed_seconds_p50": _percentile(elapsed_values, 0.50),
            "elapsed_seconds_p95": _percentile(elapsed_values, 0.95),
            "recommended_timeout_seconds": recommended_timeout,
            "timeout_multiplier": timeout_multiplier,
            "current_timeout_seconds_max": _max_or_none(_step_timeouts_for_resource(queue, kind)),
            "advisory_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_packet_ready": False,
        }

    return {
        "schema": SCHEDULER_RUNTIME_POLICY_SCHEMA,
        "queue_id": queue_id,
        "telemetry_only": True,
        "advisory_only": True,
        "generated_from_schema": performance.get("schema"),
        "input_event_count": performance.get("event_count", 0),
        "machine_capacity": {
            "local_cpu": cpu_count,
            "local_io_heavy": _local_io_heavy_capacity(cpu_count),
            "local_mlx": 1,
            "local_mps": 1,
        },
        "current_max_concurrency": current_concurrency,
        "recommended_max_concurrency": recommended_concurrency,
        "recommended_timeout_seconds_by_resource": recommended_timeouts,
        "resource_policies": resource_policies,
        "backpressure": {
            "local_io_heavy": {
                "recommended_concurrency": recommended_concurrency.get(
                    "local_io_heavy",
                    _local_io_heavy_capacity(cpu_count),
                ),
                "reason": "limit_file_tree_hashing_and_artifact_pullback_pressure",
            }
        },
        "apply_requires_explicit_call": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_packet_ready": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def apply_scheduler_runtime_policy(
    queue: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    apply_concurrency: bool = True,
    apply_timeouts: bool = True,
) -> dict[str, Any]:
    """Return a queue definition with explicit advisory runtime policy applied."""

    if policy.get("schema") != SCHEDULER_RUNTIME_POLICY_SCHEMA:
        raise ExperimentQueueError(f"runtime policy schema must be {SCHEDULER_RUNTIME_POLICY_SCHEMA}")
    violations = _truthy_authority_field_paths(
        policy,
        fields=frozenset(SCHEDULER_RUNTIME_POLICY_FORBIDDEN_AUTHORITY_FIELDS),
        path="runtime_policy",
    )
    if violations:
        raise ExperimentQueueError("runtime policy must not carry truthy authority fields: " + ", ".join(violations))
    updated = json.loads(json.dumps(queue))
    controls = updated.setdefault("controls", {})
    if apply_concurrency:
        max_concurrency = dict(controls.get("max_concurrency") or {})
        for kind, value in dict(policy.get("recommended_max_concurrency") or {}).items():
            parsed = _finite_int(value)
            if parsed is not None and parsed >= 0 and not _is_cloud_resource(str(kind)):
                max_concurrency[str(kind)] = parsed
        controls["max_concurrency"] = max_concurrency
    if apply_timeouts:
        timeout_by_resource = {
            str(kind): int(value)
            for kind, value in dict(policy.get("recommended_timeout_seconds_by_resource") or {}).items()
            if _finite_int(value) is not None and int(value) > 0
        }
        for experiment in updated.get("experiments", []):
            if not isinstance(experiment, Mapping):
                continue
            for step in experiment.get("steps", []):
                if not isinstance(step, dict):
                    continue
                kind = _resource_kind(step)
                recommended = timeout_by_resource.get(kind)
                if recommended is None:
                    continue
                current = _finite_int(step.get("timeout_seconds"))
                if current is None or current == 0 or current < recommended:
                    step["timeout_seconds"] = recommended
    return normalize_queue_definition(updated)


def _truthy_authority_field_paths(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in fields and bool(child):
                violations.append(child_path)
            violations.extend(
                _truthy_authority_field_paths(
                    child,
                    fields=fields,
                    path=child_path,
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            violations.extend(
                _truthy_authority_field_paths(
                    child,
                    fields=fields,
                    path=f"{path}[{index}]",
                )
            )
    return violations


def _runtime_policy_event_samples(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    queue_id = str(queue["queue_id"])
    rows = conn.execute(
        """
        SELECT event_type, payload_json
        FROM queue_events
        WHERE queue_id = ? AND event_type IN ('step_succeeded', 'step_failed')
        ORDER BY id
        """,
        (queue_id,),
    ).fetchall()
    samples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        kind = str(payload.get("resource_kind") or "local_cpu")
        samples.setdefault(kind, []).append(
            {
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "timed_out": payload.get("timed_out") is True,
                "succeeded": str(row["event_type"]) == "step_succeeded",
            }
        )
    return samples


def _step_timeouts_for_resource(queue: Mapping[str, Any], kind: str) -> list[int]:
    out: list[int] = []
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping):
            continue
        for step in experiment.get("steps", []):
            if not isinstance(step, Mapping) or _resource_kind(step) != kind:
                continue
            timeout = _finite_int(step.get("timeout_seconds"))
            if timeout is not None and timeout > 0:
                out.append(timeout)
    return out


def _recommended_concurrency_for_resource(
    kind: str,
    *,
    current_limit: int,
    cpu_count: int,
    timeout_count: int,
    failure_count: int,
    run_count: int,
) -> tuple[int, str]:
    if current_limit == 0:
        return 0, "resource_disabled_by_queue_control"
    if _is_cloud_resource(kind):
        return current_limit, "cloud_resource_not_locally_resized"
    if timeout_count > 0 or (run_count >= 3 and failure_count / run_count > 0.25):
        return max(1, min(current_limit, max(1, current_limit // 2))), "backoff_after_failures"
    if kind == "local_cpu":
        return max(current_limit, cpu_count), "use_detected_local_cpu_capacity"
    if kind == "local_io_heavy":
        return min(max(current_limit, 1), _local_io_heavy_capacity(cpu_count)), "cap_io_heavy_pressure"
    if kind in {"local_mlx", "local_mps", "local_cuda"}:
        return max(1, current_limit), "single_local_accelerator_default"
    return max(1, current_limit), "preserve_existing_local_limit"


def _recommended_timeout_for_resource(
    elapsed_values: Sequence[float],
    *,
    timeout_count: int,
    current_timeouts: Sequence[int],
    timeout_multiplier: float,
    min_timeout_seconds: int,
    max_timeout_seconds: int,
) -> int | None:
    observed_p95 = _percentile(elapsed_values, 0.95)
    current_max = _max_or_none(current_timeouts)
    if observed_p95 is None and current_max is None:
        return None
    base = observed_p95 * timeout_multiplier if observed_p95 is not None else 0.0
    if current_max is not None:
        base = max(base, float(current_max))
    if timeout_count and current_max is not None:
        base = max(base, float(current_max) * 2.0)
    recommended = int(base + 0.999999)
    recommended = max(min_timeout_seconds, recommended)
    return min(max_timeout_seconds, recommended)


def _percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if value >= 0)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    q = min(1.0, max(0.0, q))
    index = int((len(clean) - 1) * q + 0.999999)
    return clean[min(len(clean) - 1, index)]


def _max_or_none(values: Sequence[int]) -> int | None:
    return max(values) if values else None


def _local_io_heavy_capacity(cpu_count: int) -> int:
    return max(1, min(4, cpu_count // 4 or 1))


def queue_definition_drift(conn: sqlite3.Connection, queue: Mapping[str, Any]) -> dict[str, Any]:
    """Return queue definition drift without mutating SQLite state."""

    queue_id = str(queue["queue_id"])
    missing_steps: list[dict[str, Any]] = []
    changed_steps: list[dict[str, Any]] = []
    missing_hash_steps: list[dict[str, Any]] = []
    for experiment in queue["experiments"]:
        experiment_metadata = dict(experiment.get("metadata") or {})
        for step in experiment["steps"]:
            hashes = _step_hashes(step, experiment_metadata=experiment_metadata)
            row = conn.execute(
                """
                SELECT status, definition_hash, command_hash, postcondition_hash,
                       resource_kind
                FROM step_state
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (queue_id, experiment["id"], step["id"]),
            ).fetchone()
            identity = {
                "experiment_id": str(experiment["id"]),
                "step_id": str(step["id"]),
            }
            if row is None:
                missing_steps.append({**identity, "new_hashes": hashes})
                continue
            previous_hashes = {
                "definition_hash": row["definition_hash"],
                "command_hash": row["command_hash"],
                "postcondition_hash": row["postcondition_hash"],
            }
            missing_hash = any(value is None for value in previous_hashes.values())
            missing_resource_kind = row["resource_kind"] is None
            changed_hash = any(
                previous_hashes[key] is not None and previous_hashes[key] != hashes[key] for key in hashes
            )
            if (missing_hash or missing_resource_kind) and not changed_hash:
                missing_hash_steps.append(
                    {
                        **identity,
                        "status": str(row["status"]),
                        "previous_hashes": previous_hashes,
                        "new_hashes": hashes,
                        "previous_resource_kind": row["resource_kind"],
                        "new_resource_kind": _resource_kind(step),
                    }
                )
            elif changed_hash:
                changed_steps.append(
                    {
                        **identity,
                        "status": str(row["status"]),
                        "previous_hashes": previous_hashes,
                        "new_hashes": hashes,
                    }
                )
    return {
        "schema": "experiment_queue_definition_drift.v1",
        "read_only": True,
        "missing_step_count": len(missing_steps),
        "changed_step_count": len(changed_steps),
        "missing_hash_step_count": len(missing_hash_steps),
        "missing_steps": missing_steps,
        "changed_steps": changed_steps,
        "missing_hash_steps": missing_hash_steps,
    }


__all__ = [
    "SCHEDULER_RUNTIME_POLICY_SCHEMA",
    "ExperimentQueueError",
    "ReadyStep",
    "apply_scheduler_runtime_policy",
    "assert_canonical_state_for_execution",
    "assert_no_orphaned_steps_for_execution",
    "connect_state",
    "connect_state_readonly",
    "default_state_path",
    "derive_scheduler_runtime_policy",
    "finalize_claimed_step_execution",
    "initialize_queue_state",
    "load_queue_definition",
    "normalize_queue_definition",
    "normalize_resource_kind",
    "orphaned_step_rows",
    "queue_definition_drift",
    "queue_performance_summary",
    "queue_resource_kinds",
    "queue_summary",
    "ready_steps",
    "reconcile_satisfied_queued_steps",
    "reconcile_stale_running_steps",
    "resolve_worker_max_parallel",
    "retire_orphaned_steps",
    "rewind_step",
    "run_queue_worker",
    "run_ready_step",
    "set_control_mode",
    "worker_resource_limits",
]
