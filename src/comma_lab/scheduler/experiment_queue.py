from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUEUE_SCHEMA = "experiment_queue.v1"
STATE_SCHEMA = "experiment_queue_state.v1"

CONTROL_MODES = {"running", "paused", "frozen"}
STEP_STATUSES = {"queued", "running", "succeeded", "failed", "blocked", "skipped"}
BLOCKING_ORPHAN_STATUSES = {"queued", "running", "blocked"}
LOCAL_RESOURCE_KINDS = {"local_cpu", "local_mlx", "local_mps", "local"}
CLOUD_RESOURCE_KINDS = {"cloud_cpu", "cloud_gpu", "modal_cpu", "modal_gpu", "cuda_auth"}
KNOWN_RESOURCE_KINDS = LOCAL_RESOURCE_KINDS | CLOUD_RESOURCE_KINDS


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

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id,
            "experiment_id": self.experiment_id,
            "step_id": self.step_id,
            "priority": self.priority,
            "resource_kind": self.resource_kind,
            "command": list(self.command),
        }


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


def _non_negative_int(value: object, label: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentQueueError(f"{label} must be a non-negative integer")
    return value


def _resource_kind_value(value: object, label: str) -> str:
    kind = _require_text(value, label)
    if kind in KNOWN_RESOURCE_KINDS or kind.startswith("cloud_"):
        return kind
    raise ExperimentQueueError(
        f"{label} unsupported resource kind {kind!r}; known kinds are "
        f"{sorted(KNOWN_RESOURCE_KINDS)}"
    )


def _load_json_compatible(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raise ExperimentQueueError(
                f"{path}: YAML queue files must be JSON-compatible until PyYAML "
                "is explicitly added as a dependency"
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
            raise ExperimentQueueError(
                f"{step_id}.postconditions[{condition_index}] must be an object"
            )
        normalized_postconditions.append(dict(condition))
    return {
        "id": step_id,
        "kind": kind,
        "command": command_items,
        "requires": requires,
        "resources": {**resources, "kind": resource_kind},
        "postconditions": normalized_postconditions,
        "timeout_seconds": _non_negative_int(raw.get("timeout_seconds"), f"{step_id}.timeout_seconds"),
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
        step_ids = seen_step_ids
        if duplicate_step_ids:
            duplicates = sorted(duplicate_step_ids)
            raise ExperimentQueueError(
                f"{experiment_id}.steps contains duplicate step id(s): {duplicates}"
            )
        for step in steps:
            unknown = sorted(set(step["requires"]) - step_ids)
            if unknown:
                raise ExperimentQueueError(
                    f"{experiment_id}.{step['id']} requires unknown step(s): {unknown}"
                )
        normalized_experiments.append(
            {
                "id": experiment_id,
                "status": status,
                "priority": priority,
                "lane_id": raw_experiment.get("lane_id"),
                "tags": _string_list(raw_experiment.get("tags"), f"{experiment_id}.tags"),
                "steps": steps,
            }
        )
    return {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": {
            "mode": mode,
            "local_first": local_first,
            "max_concurrency": normalized_concurrency,
        },
        "experiments": normalized_experiments,
    }


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
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(step_state)").fetchall()
    }
    for column in ("definition_hash", "command_hash", "postcondition_hash"):
        if column not in columns:
            conn.execute(f"ALTER TABLE step_state ADD COLUMN {column} TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO queue_meta(key, value) VALUES (?, ?)",
        ("schema", STATE_SCHEMA),
    )
    conn.commit()


def _step_hashes(step: Mapping[str, Any]) -> dict[str, str]:
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
    }
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
    conn.execute(
        """
        INSERT OR IGNORE INTO queue_controls(queue_id, mode, reason, updated_at_utc)
        VALUES (?, ?, ?, ?)
        """,
        (queue_id, mode, "initialized_from_queue_definition", now),
    )
    for experiment in queue["experiments"]:
        for step in experiment["steps"]:
            hashes = _step_hashes(step)
            row = conn.execute(
                """
                SELECT status, definition_hash, command_hash, postcondition_hash
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
                      command_hash, postcondition_hash
                    )
                    VALUES (?, ?, ?, 'queued', 0, ?, NULL, ?, ?, ?)
                    """,
                    (
                        queue_id,
                        experiment["id"],
                        step["id"],
                        now,
                        hashes["definition_hash"],
                        hashes["command_hash"],
                        hashes["postcondition_hash"],
                    ),
                )
                continue

            previous_hashes = {
                "definition_hash": row["definition_hash"],
                "command_hash": row["command_hash"],
                "postcondition_hash": row["postcondition_hash"],
            }
            missing_hash = any(value is None for value in previous_hashes.values())
            changed_hash = any(
                previous_hashes[key] is not None and previous_hashes[key] != hashes[key]
                for key in hashes
            )
            if missing_hash and not changed_hash:
                conn.execute(
                    """
                    UPDATE step_state
                    SET definition_hash = ?, command_hash = ?, postcondition_hash = ?
                    WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                    """,
                    (
                        hashes["definition_hash"],
                        hashes["command_hash"],
                        hashes["postcondition_hash"],
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
                    definition_hash = ?, command_hash = ?, postcondition_hash = ?
                WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
                """,
                (
                    now,
                    _json_text(event),
                    hashes["definition_hash"],
                    hashes["command_hash"],
                    hashes["postcondition_hash"],
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
        SELECT experiment_id, step_id, status, attempts, updated_at_utc, last_event_json
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


def ready_steps(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    allow_cloud: bool = False,
) -> list[ReadyStep]:
    queue_id = str(queue["queue_id"])
    if control_mode(conn, queue_id) != "running":
        return []
    state = _state_rows(conn, queue_id)
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
            requirements = step.get("requires") or []
            if any(
                state.get((str(experiment["id"]), str(required))) is None
                or state[(str(experiment["id"]), str(required))]["status"] != "succeeded"
                for required in requirements
            ):
                continue
            out.append(
                ReadyStep(
                    queue_id=queue_id,
                    experiment_id=str(experiment["id"]),
                    step_id=str(step["id"]),
                    priority=int(experiment["priority"]),
                    resource_kind=kind,
                    command=tuple(str(part) for part in step["command"]),
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


def _json_pointer(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise ExperimentQueueError(f"json key not found: {key}")
    return current


def _condition_passes(condition: Mapping[str, Any], *, repo_root: Path) -> bool:
    condition_type = _require_text(condition.get("type"), "postcondition.type")
    if condition_type == "path_exists":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        return (repo_root / rel_path).exists()
    if condition_type == "json_equals":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        key = _require_text(condition.get("key"), "postcondition.key")
        expected = condition.get("equals")
        payload = json.loads((repo_root / rel_path).read_text())
        return _json_pointer(payload, key) == expected
    if condition_type == "json_false_authority":
        rel_path = _require_text(condition.get("path"), "postcondition.path")
        payload = json.loads((repo_root / rel_path).read_text())
        if not isinstance(payload, Mapping):
            return False
        required_false = _string_list(
            condition.get("required_false", ["score_claim", "promotion_eligible", "rank_or_kill_eligible"]),
            "postcondition.required_false",
        )
        for key in required_false:
            if _json_pointer(payload, key) is not False:
                return False
        false_or_missing = _string_list(
            condition.get(
                "false_or_missing",
                ["ready_for_exact_eval_dispatch", "dispatch_attempted", "gpu_launched"],
            ),
            "postcondition.false_or_missing",
        )
        for key in false_or_missing:
            try:
                value = _json_pointer(payload, key)
            except ExperimentQueueError:
                continue
            if value is not False:
                return False
        axis_key = condition.get("axis_key")
        if axis_key is not None:
            actual_axis = _json_pointer(payload, _require_text(axis_key, "postcondition.axis_key"))
            expected_axis = _require_text(condition.get("axis_equals"), "postcondition.axis_equals")
            if actual_axis != expected_axis:
                return False
        return True
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


def _claim_step_running(
    conn: sqlite3.Connection,
    *,
    queue_id: str,
    experiment_id: str,
    step_id: str,
    event: Mapping[str, Any],
) -> str | None:
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
          AND COALESCE(
            (SELECT mode FROM queue_controls WHERE queue_controls.queue_id = ?),
            'running'
          ) = 'running'
        """,
        (_utc_now(), _json_text(dict(event)), queue_id, experiment_id, step_id, queue_id),
    )
    if cursor.rowcount != 1:
        mode = control_mode(conn, queue_id)
        reason = "control_not_running" if mode != "running" else "not_queued"
        refused_event = {**dict(event), "claim_refused_reason": reason, "control_mode": mode}
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=experiment_id,
            step_id=step_id,
            event_type=(
                "step_claim_refused_control_not_running"
                if reason == "control_not_running"
                else "step_claim_refused"
            ),
            payload=refused_event,
        )
        conn.commit()
        return reason
    append_event(
        conn,
        queue_id=queue_id,
        experiment_id=experiment_id,
        step_id=step_id,
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

    running_event = {"command": list(ready.command), "log_path": str(log_path)}
    claim_refused_reason = _claim_step_running(
        conn,
        queue_id=ready.queue_id,
        experiment_id=ready.experiment_id,
        step_id=ready.step_id,
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
            postcondition_errors.append(
                {"condition": _json_text(condition), "error": f"{type(exc).__name__}: {exc}"}
            )
    succeeded = returncode == 0 and not timed_out and not execution_error and not failed_conditions
    event = {
        "command": list(ready.command),
        "returncode": returncode,
        "timed_out": timed_out,
        "execution_error": execution_error,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
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


def _active_step_keys(queue: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(experiment["id"]), str(step["id"]))
        for experiment in queue["experiments"]
        for step in experiment["steps"]
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
    return [
        row
        for row in rows
        if (str(row["experiment_id"]), str(row["step_id"])) not in active_keys
    ]


def assert_no_orphaned_steps_for_execution(
    conn: sqlite3.Connection,
    queue: Mapping[str, Any],
    *,
    allow_orphaned_state: bool = False,
) -> None:
    if allow_orphaned_state:
        return
    orphaned = [
        row
        for row in orphaned_step_rows(conn, queue)
        if str(row["status"]) in BLOCKING_ORPHAN_STATUSES
    ]
    if orphaned:
        preview = ", ".join(
            f"{row['experiment_id']}.{row['step_id']}={row['status']}" for row in orphaned[:5]
        )
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
                f"refusing to retire running orphaned step "
                f"{row['experiment_id']}.{row['step_id']}"
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
    allow_cloud: bool = False,
    allow_orphaned_state: bool = False,
    orphaned_state_rationale: str | None = None,
    noncanonical_state_rationale: str | None = None,
    log_root: str | Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    reload_queue: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ExperimentQueueError("max_steps must be a non-negative integer")
    if (
        isinstance(max_idle_cycles, bool)
        or not isinstance(max_idle_cycles, int)
        or max_idle_cycles < 0
    ):
        raise ExperimentQueueError("max_idle_cycles must be a non-negative integer")
    if idle_sleep_seconds < 0:
        raise ExperimentQueueError("idle_sleep_seconds must be non-negative")

    queue_id = str(queue["queue_id"])
    active_queue: Mapping[str, Any] = queue

    def _active_queue() -> Mapping[str, Any]:
        nonlocal active_queue
        if reload_queue is None:
            return active_queue
        refreshed = reload_queue()
        if str(refreshed["queue_id"]) != queue_id:
            raise ExperimentQueueError(
                f"reloaded queue_id changed from {queue_id!r} to {refreshed['queue_id']!r}"
            )
        initialize_queue_state(conn, refreshed)
        assert_no_orphaned_steps_for_execution(conn, refreshed, allow_orphaned_state=allow_orphaned_state)
        active_queue = refreshed
        return active_queue

    assert_no_orphaned_steps_for_execution(conn, active_queue, allow_orphaned_state=allow_orphaned_state)
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
            "max_idle_cycles": max_idle_cycles,
            "idle_sleep_seconds": idle_sleep_seconds,
            "allow_cloud": allow_cloud,
            "allow_orphaned_state": allow_orphaned_state,
            "orphaned_state_rationale": orphaned_state_rationale,
            "noncanonical_state_rationale": noncanonical_state_rationale,
        },
    )
    conn.commit()

    if not execute:
        planned = [
            step.to_dict()
            for step in ready_steps(conn, _active_queue(), allow_cloud=allow_cloud)
        ]
        selected = planned[:max_steps] if max_steps else []
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
            "orphaned_state_rationale": orphaned_state_rationale,
            "noncanonical_state_rationale": noncanonical_state_rationale,
            "started_at_utc": started_at,
            "stop_reason": "dry_run",
            "steps_started": 0,
            "success_count": 0,
            "failure_count": 0,
            "claim_refused_count": 0,
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

    while steps_started < max_steps:
        if stop_requested is not None and stop_requested():
            stop_reason = "stop_requested"
            append_event(
                conn,
                queue_id=queue_id,
                experiment_id=None,
                step_id=None,
                event_type="worker_stop_requested",
                payload={"steps_started": steps_started},
            )
            conn.commit()
            break

        ready = ready_steps(conn, _active_queue(), allow_cloud=allow_cloud)
        if not ready:
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

        idle_cycles = 0
        ready_step = ready[0]
        result = run_ready_step(
            conn,
            active_queue,
            ready_step,
            repo_root=repo_root,
            execute=True,
            log_root=log_root,
        )
        step_result = {"ready_step": ready_step.to_dict(), **result}
        step_results.append(step_result)
        if result.get("claim_refused"):
            claim_refused_count += 1
            continue

        steps_started += 1
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

    append_event(
        conn,
        queue_id=queue_id,
        experiment_id=None,
        step_id=None,
        event_type="worker_stopped",
        payload={
            "stop_reason": stop_reason,
            "steps_started": steps_started,
            "success_count": success_count,
            "failure_count": failure_count,
            "claim_refused_count": claim_refused_count,
            "idle_cycles": idle_cycles,
        },
    )
    conn.commit()
    return {
        "schema": "experiment_queue_worker_result.v1",
        "queue_id": queue_id,
        "execute": True,
        "allow_cloud": allow_cloud,
        "orphaned_state_rationale": orphaned_state_rationale,
        "noncanonical_state_rationale": noncanonical_state_rationale,
        "started_at_utc": started_at,
        "stop_reason": stop_reason,
        "steps_started": steps_started,
        "success_count": success_count,
        "failure_count": failure_count,
        "claim_refused_count": claim_refused_count,
        "idle_cycles": idle_cycles,
        "planned_steps": [],
        "step_results": step_results,
    }


def _downstream_step_ids(
    queue: Mapping[str, Any],
    experiment_id: str,
    step_id: str,
) -> list[str]:
    for experiment in queue["experiments"]:
        if str(experiment["id"]) != experiment_id:
            continue
        requirements = {
            str(step["id"]): {str(required) for required in step.get("requires") or []}
            for step in experiment["steps"]
        }
        out: list[str] = []
        seen = {step_id}
        pending = [step_id]
        while pending:
            current = pending.pop(0)
            for candidate_id, required_ids in requirements.items():
                if candidate_id in seen or current not in required_ids:
                    continue
                seen.add(candidate_id)
                out.append(candidate_id)
                pending.append(candidate_id)
        return out
    raise ExperimentQueueError(f"unknown experiment: {experiment_id}")


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
    for current_step_id in [step_id, *downstream]:
        event = {
            "reason": reason,
            "root_step_id": step_id,
            "cascade": cascade,
            "cascade_rewound": current_step_id != step_id,
        }
        conn.execute(
            """
            UPDATE step_state
            SET status = 'queued', updated_at_utc = ?, last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (_utc_now(), _json_text(event), queue_id, experiment_id, current_step_id),
        )
        append_event(
            conn,
            queue_id=queue_id,
            experiment_id=experiment_id,
            step_id=current_step_id,
            event_type="step_rewound",
            payload=event,
        )
    conn.commit()


def queue_summary(conn: sqlite3.Connection, queue: Mapping[str, Any]) -> dict[str, Any]:
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
    active_rows = [
        row
        for row in rows
        if (str(row["experiment_id"]), str(row["step_id"])) in active_keys
    ]
    orphaned_rows = [
        row
        for row in rows
        if (str(row["experiment_id"]), str(row["step_id"])) not in active_keys
    ]
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
        "ready_steps": [step.to_dict() for step in ready_steps(conn, queue)],
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


def queue_definition_drift(conn: sqlite3.Connection, queue: Mapping[str, Any]) -> dict[str, Any]:
    """Return queue definition drift without mutating SQLite state."""

    queue_id = str(queue["queue_id"])
    missing_steps: list[dict[str, Any]] = []
    changed_steps: list[dict[str, Any]] = []
    missing_hash_steps: list[dict[str, Any]] = []
    for experiment in queue["experiments"]:
        for step in experiment["steps"]:
            hashes = _step_hashes(step)
            row = conn.execute(
                """
                SELECT status, definition_hash, command_hash, postcondition_hash
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
            changed_hash = any(
                previous_hashes[key] is not None and previous_hashes[key] != hashes[key]
                for key in hashes
            )
            if missing_hash and not changed_hash:
                missing_hash_steps.append(
                    {
                        **identity,
                        "status": str(row["status"]),
                        "previous_hashes": previous_hashes,
                        "new_hashes": hashes,
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
    "ExperimentQueueError",
    "ReadyStep",
    "assert_canonical_state_for_execution",
    "assert_no_orphaned_steps_for_execution",
    "connect_state",
    "connect_state_readonly",
    "default_state_path",
    "initialize_queue_state",
    "load_queue_definition",
    "normalize_queue_definition",
    "orphaned_step_rows",
    "queue_definition_drift",
    "queue_summary",
    "ready_steps",
    "retire_orphaned_steps",
    "rewind_step",
    "run_queue_worker",
    "run_ready_step",
    "set_control_mode",
]
