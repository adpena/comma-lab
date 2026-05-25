from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    _condition_passes,
    connect_state_readonly,
    derive_scheduler_runtime_policy,
    queue_definition_drift,
    queue_performance_summary,
    queue_resource_kinds,
    queue_summary,
    resolve_worker_max_parallel,
)
from tac.optimization.materializer_feedback import (
    MATERIALIZER_FALSE_AUTHORITY,
    materializer_archive_delta,
)
from tac.optimization.serialized_archive_economics import SERIALIZED_ARCHIVE_DELTA_SCHEMA

OBSERVATION_SCHEMA = "experiment_queue_observation.v1"
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA = "family_agnostic_materializer_empirical_observation.v1"
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA = "family_agnostic_materializer_empirical_sweep.v1"
PR95_MLX_PACKAGE_SCHEMA = "pr95_mlx_pytorch_state_dict_to_contest_archive.v1"
PR95_MLX_LONG_TRAINING_PLAN_SCHEMA = "pr95_mlx_long_training_plan.v1"
FAMILY_AGNOSTIC_MATERIALIZER_CANDIDATE_SCHEMAS = frozenset(
    {
        "archive_section_entropy_recode_candidate.v1",
        "packet_member_merge_candidate.v1",
        "packet_member_recompress_candidate.v1",
        "packet_member_zip_header_elide_candidate.v1",
        "renderer_payload_dfl1_candidate.v1",
        "tensor_factorize_candidate.v1",
    }
)
REQUIRED_MATERIALIZER_FEEDBACK_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            out.append(
                {
                    "pid": str(proc.get("pid") or ""),
                    "etime": str(proc.get("etime") or ""),
                    "command": command,
                }
            )
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
                *MATERIALIZER_FALSE_AUTHORITY,
                "candidate_id",
                "target_kind",
                "materializer_id",
                "receiver_contract_kind",
                "receiver_contract_satisfied",
                "runtime_consumption_proof_path",
                "canonical_score",
                "score_axis",
                "archive_sha256",
                "archive_bytes",
                "eureka",
                "margin_vs_frontier",
            ):
                if key in payload:
                    record[key] = payload[key]
            if record.get("json_schema") == PR95_MLX_PACKAGE_SCHEMA:
                record["pr95_mlx_package_report"] = True
                for source_key, target_key in (
                    ("archive_zip_sha256", "archive_sha256"),
                    ("archive_zip_bytes", "archive_bytes"),
                    ("archive_member_sha256", "archive_member_sha256"),
                    ("archive_member_bytes", "archive_member_bytes"),
                    ("archive_manifest_path", "archive_manifest_path"),
                    ("input_pt_sha256", "input_pt_sha256"),
                    ("source_archive_zip_sha256", "source_archive_zip_sha256"),
                ):
                    if source_key in payload:
                        record[target_key] = payload[source_key]
                runtime_files = payload.get("runtime_files_emitted")
                if isinstance(runtime_files, Mapping):
                    record["runtime_file_count"] = len(runtime_files)
                    record["runtime_files_emitted"] = sorted(str(key) for key in runtime_files)
                refusal = payload.get("exact_readiness_refusal")
                if isinstance(refusal, Mapping):
                    blockers = [
                        str(item)
                        for item in refusal.get("blockers", [])
                        if str(item)
                    ]
                    record["exact_readiness_refusal"] = {
                        "ready": refusal.get("ready"),
                        "blockers": blockers,
                    }
                    if blockers and "readiness_blockers" not in record:
                        record["readiness_blockers"] = blockers
            if record.get("json_schema") == PR95_MLX_LONG_TRAINING_PLAN_SCHEMA:
                record["pr95_mlx_long_training_plan"] = True
                for key in (
                    "mode",
                    "lane_id",
                    "source_video_sha256",
                    "source_video_frame_count",
                    "checkpoint_root",
                    "telemetry_path",
                    "candidate_registry_count",
                    "total_epochs",
                    "smoke_mode",
                    "training_fidelity_class",
                    "training_fidelity_status",
                    "reproduction_equivalence",
                    "reproduction_claim",
                    "pr95_1to1_reproduction_claim",
                    "reproduction_equivalence_class",
                ):
                    if key in payload:
                        record[key] = payload[key]
                refusal = payload.get("exact_readiness_refusal")
                if isinstance(refusal, Mapping):
                    blockers = [
                        str(item)
                        for item in refusal.get("blockers", [])
                        if str(item)
                    ]
                    record["exact_readiness_refusal"] = {
                        "ready": refusal.get("ready"),
                        "blockers": blockers,
                    }
                    if blockers and "readiness_blockers" not in record:
                        record["readiness_blockers"] = blockers
            if isinstance(payload.get("readiness_blockers"), list):
                record["readiness_blockers"] = [str(item) for item in payload["readiness_blockers"] if str(item)]
            receiver = payload.get("receiver_verification")
            if isinstance(receiver, Mapping):
                record["receiver_verification"] = {
                    key: receiver.get(key)
                    for key in (
                        "schema",
                        "receiver_contract_kind",
                        "receiver_contract_satisfied",
                        "runtime_adapter_ready",
                        "proof_present",
                        "proof_schema",
                    )
                    if key in receiver
                }
                receiver_blockers = receiver.get("blockers")
                if isinstance(receiver_blockers, list):
                    record["receiver_verification"]["blockers"] = [
                        str(item) for item in receiver_blockers if str(item)
                    ]
            candidate_archive = payload.get("candidate_archive")
            if isinstance(candidate_archive, Mapping):
                record["candidate_archive"] = {
                    key: candidate_archive.get(key)
                    for key in ("path", "bytes", "sha256", "member_sha256")
                    if key in candidate_archive
                }
            delta = payload.get("serialized_archive_delta")
            if isinstance(delta, Mapping):
                record["serialized_archive_delta_schema"] = delta.get("schema")
                record["serialized_archive_delta_status"] = delta.get("status")
                record["serialized_archive_delta_realized_saved_bytes"] = delta.get(
                    "realized_saved_bytes"
                )
                record["serialized_archive_delta_savings_realized"] = delta.get(
                    "savings_realized"
                )
                record["serialized_archive_delta_source_archive_bytes"] = delta.get(
                    "source_archive_bytes"
                )
                record["serialized_archive_delta_candidate_archive_bytes"] = delta.get(
                    "candidate_archive_bytes"
                )
            materializer_delta = materializer_archive_delta(payload)
            if materializer_delta is not None:
                selected_key = materializer_delta.get("selected_materialization_key")
                record["materializer_delta_source"] = selected_key
                if selected_key:
                    prefix = str(selected_key)
                    if f"{prefix}_saved_bytes" not in record:
                        record[f"{prefix}_saved_bytes"] = materializer_delta.get(
                            "realized_saved_bytes"
                        )
                    if f"{prefix}_source_archive_bytes" not in record:
                        record[f"{prefix}_source_archive_bytes"] = materializer_delta.get(
                            "source_archive_bytes"
                        )
                    if f"{prefix}_candidate_archive_bytes" not in record:
                        record[f"{prefix}_candidate_archive_bytes"] = (
                            materializer_delta.get("candidate_archive_bytes")
                        )
                if record.get("serialized_archive_delta_realized_saved_bytes") is None:
                    record["serialized_archive_delta_realized_saved_bytes"] = materializer_delta.get(
                        "realized_saved_bytes"
                    )
                if record.get("serialized_archive_delta_source_archive_bytes") is None:
                    record["serialized_archive_delta_source_archive_bytes"] = materializer_delta.get(
                        "source_archive_bytes"
                    )
                if record.get("serialized_archive_delta_candidate_archive_bytes") is None:
                    record["serialized_archive_delta_candidate_archive_bytes"] = materializer_delta.get(
                        "candidate_archive_bytes"
                    )
                if record.get("serialized_archive_delta_savings_realized") is None:
                    record["serialized_archive_delta_savings_realized"] = (
                        materializer_delta.get("savings_realized")
                    )
                if record.get("serialized_archive_delta_status") is None:
                    record["serialized_archive_delta_status"] = materializer_delta.get("status")
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


def _experiment_metadata_lookup(
    queue: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping):
            continue
        metadata = experiment.get("metadata")
        out[str(experiment.get("id") or "")] = metadata if isinstance(metadata, Mapping) else {}
    return out


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return []


def _nonempty_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _extend_unique(out: list[str], values: Sequence[Any]) -> None:
    seen = set(out)
    for value in values:
        text = _nonempty_str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)


def _artifact_paths_from_telemetry(
    step: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    telemetry = step.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return []
    paths: list[str] = []
    for key in ("artifact_paths", "input_artifact_paths", "pullback_artifact_paths"):
        for raw_path in _string_list(telemetry.get(key)):
            path = Path(raw_path)
            if not path.is_absolute():
                path = repo_root / path
            _extend_unique(paths, [_repo_rel(path, repo_root)])
    return paths


def _queue_state_watermark(
    conn: sqlite3.Connection,
    *,
    queue_id: str,
    state_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    events = conn.execute(
        """
        SELECT COUNT(*) AS event_count, COALESCE(MAX(id), 0) AS max_event_id
        FROM queue_events
        WHERE queue_id = ?
        """,
        (queue_id,),
    ).fetchone()
    steps = conn.execute(
        """
        SELECT COUNT(*) AS step_state_count,
               COALESCE(MAX(updated_at_utc), '') AS max_step_updated_at_utc
        FROM step_state
        WHERE queue_id = ?
        """,
        (queue_id,),
    ).fetchone()
    control = conn.execute(
        """
        SELECT mode, updated_at_utc
        FROM queue_controls
        WHERE queue_id = ?
        """,
        (queue_id,),
    ).fetchone()
    return {
        "schema": "experiment_queue_state_watermark.v1",
        "queue_id": queue_id,
        "state_path": _repo_rel(state_path, repo_root),
        "event_count": int(events["event_count"] or 0) if events else 0,
        "max_event_id": int(events["max_event_id"] or 0) if events else 0,
        "step_state_count": int(steps["step_state_count"] or 0) if steps else 0,
        "max_step_updated_at_utc": (str(steps["max_step_updated_at_utc"] or "") if steps else ""),
        "control_mode": str(control["mode"] or "") if control else "",
        "control_updated_at_utc": str(control["updated_at_utc"] or "") if control else "",
    }


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
        if "schema_equals" in condition:
            record["postcondition_schema_equals"] = condition.get("schema_equals")
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
    experiment_metadata: Mapping[str, Any] | None,
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
        "expected_artifact_paths": [],
    }
    if step_definition is not None:
        resources = step_definition.get("resources")
        if isinstance(resources, Mapping):
            observation["resource_kind"] = resources.get("kind")
        observation["expected_artifacts"] = _expected_artifacts(
            step_definition,
            repo_root=repo_root,
        )
        _extend_unique(
            observation["expected_artifact_paths"],
            [artifact.get("path") for artifact in observation["expected_artifacts"] if isinstance(artifact, Mapping)],
        )
        _extend_unique(
            observation["expected_artifact_paths"],
            _artifact_paths_from_telemetry(step_definition, repo_root=repo_root),
        )
    metadata = experiment_metadata if isinstance(experiment_metadata, Mapping) else {}
    for key in (
        "target_kind",
        "materializer_id",
        "receiver_contract_id",
        "receiver_contract_kind",
        "unit_kind",
        "operation_family",
    ):
        text = _nonempty_str(metadata.get(key))
        if text:
            observation[key] = text
    for metadata_key, observation_key in (
        ("work_id", "work_ids"),
        ("backlog_key", "backlog_keys"),
    ):
        text = _nonempty_str(metadata.get(metadata_key))
        if text:
            observation[observation_key] = [text]
    for key in ("candidate_ids", "source_unit_ids", "source_selection_ids"):
        values = _string_list(metadata.get(key))
        if values:
            observation[key] = values
    for key in ("candidate_saved_bytes_sum", "expected_score_gain_sum"):
        if key in metadata:
            observation[key] = metadata.get(key)
    if log_path is not None and log_path.exists():
        observation["log_tail"] = _tail_text(log_path, max_lines=tail_lines)
    return observation


def _step_has_materializer_feedback_artifact(
    step_observation: Mapping[str, Any],
) -> bool:
    for artifact in step_observation.get("expected_artifacts") or []:
        if not isinstance(artifact, Mapping):
            continue
        if artifact.get("exists") is not True:
            continue
        if artifact.get("postcondition_passed") is not True:
            continue
        if not _artifact_false_authority_satisfied(artifact):
            continue
        if (
            artifact.get("postcondition_type") == "jsonl_false_authority"
            and artifact.get("postcondition_schema_equals")
            == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA
        ):
            return True
        if artifact.get("json_schema") in {
            FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
            FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
        }:
            return True
        if (
            (
                artifact.get("json_schema") in FAMILY_AGNOSTIC_MATERIALIZER_CANDIDATE_SCHEMAS
                or artifact.get("serialized_archive_delta_schema") == SERIALIZED_ARCHIVE_DELTA_SCHEMA
            )
            and _artifact_has_materializer_identity(artifact)
            and _artifact_has_materializer_delta(artifact)
        ):
            return True
    return False


def _artifact_false_authority_satisfied(artifact: Mapping[str, Any]) -> bool:
    if (
        artifact.get("postcondition_type") in {"json_false_authority", "jsonl_false_authority"}
        and artifact.get("postcondition_passed") is True
    ):
        return True
    for key in REQUIRED_MATERIALIZER_FEEDBACK_FALSE_AUTHORITY_FIELDS:
        if artifact.get(key) is not False:
            return False
    return all(
        not (key in artifact and artifact.get(key) is not False)
        for key in MATERIALIZER_FALSE_AUTHORITY
    )


def _artifact_has_materializer_identity(artifact: Mapping[str, Any]) -> bool:
    return (
        bool(artifact.get("target_kind"))
        and bool(artifact.get("materializer_id"))
        and bool(artifact.get("receiver_contract_kind"))
    )


def _artifact_has_materializer_delta(artifact: Mapping[str, Any]) -> bool:
    return bool(artifact.get("materializer_delta_source")) or any(
        artifact.get(key) is not None
        for key in (
            "serialized_archive_delta_status",
            "serialized_archive_delta_realized_saved_bytes",
            "section_recode_saved_bytes",
            "selected_compression_saved_bytes",
            "selected_merge_saved_bytes",
            "selected_payload_saved_bytes",
            "selected_elision_saved_bytes",
            "factorization_saved_bytes",
        )
    )


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
            runtime_policy = derive_scheduler_runtime_policy(conn, queue)
            state_watermark = _queue_state_watermark(
                conn,
                queue_id=str(queue["queue_id"]),
                state_path=state_path,
                repo_root=repo_root,
            )
    except (ExperimentQueueError, sqlite3.Error):
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
        runtime_policy = {
            "schema": "scheduler_runtime_policy.v1",
            "queue_id": str(queue["queue_id"]),
            "state_missing": True,
            "telemetry_only": True,
            "advisory_only": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "recommended_max_concurrency": {},
            "recommended_timeout_seconds_by_resource": {},
        }
        state_watermark = {
            "schema": "experiment_queue_state_watermark.v1",
            "queue_id": str(queue["queue_id"]),
            "state_path": _repo_rel(state_path, repo_root),
            "state_missing": True,
            "event_count": 0,
            "max_event_id": 0,
            "step_state_count": 0,
            "max_step_updated_at_utc": "",
            "control_mode": "",
            "control_updated_at_utc": "",
        }
    lookup = _experiment_lookup(queue)
    metadata_lookup = _experiment_metadata_lookup(queue)
    processes = _process_table()
    active_steps: list[dict[str, Any]] = []
    health_steps: list[dict[str, Any]] = []
    succeeded_artifact_failure_steps: list[dict[str, Any]] = []
    succeeded_artifact_steps: list[dict[str, Any]] = []
    for step_state in summary.get("steps", []):
        if not isinstance(step_state, Mapping):
            continue
        status = step_state.get("status")
        if status not in {"running", "failed", "queued", "blocked", "succeeded"}:
            continue
        key = (
            str(step_state.get("experiment_id") or ""),
            str(step_state.get("step_id") or ""),
        )
        step_observation = _step_observation(
            step_state,
            lookup.get(key),
            experiment_metadata=metadata_lookup.get(key[0]),
            repo_root=repo_root,
            processes=processes,
            tail_lines=0 if status == "succeeded" else tail_lines,
        )
        if status == "succeeded":
            if _step_has_materializer_feedback_artifact(step_observation):
                succeeded_artifact_steps.append(step_observation)
            artifacts = step_observation.get("expected_artifacts") or []
            if any(isinstance(item, Mapping) and item.get("postcondition_passed") is False for item in artifacts):
                succeeded_artifact_failure_steps.append(step_observation)
                health_steps.append(step_observation)
            continue
        active_steps.append(step_observation)
        health_steps.append(step_observation)
    orphaned = []
    if include_orphans:
        for step_state in summary.get("orphaned_steps", []):
            if isinstance(step_state, Mapping):
                orphaned.append(
                    _step_observation(
                        step_state,
                        None,
                        experiment_metadata=None,
                        repo_root=repo_root,
                        processes=processes,
                        tail_lines=0,
                    )
                )

    running = [step for step in active_steps if step.get("status") == "running"]
    failed = [step for step in active_steps if step.get("status") == "failed"]
    queued = [step for step in active_steps if step.get("status") == "queued"]
    blocked = [step for step in active_steps if step.get("status") == "blocked"]
    blockers = _observation_blockers(
        definition_drift=definition_drift,
        performance=performance,
        runtime_policy=runtime_policy,
        orphaned_step_count=summary.get("orphaned_step_count"),
        active_steps=health_steps,
    )
    return {
        "schema": OBSERVATION_SCHEMA,
        "generated_at_utc": _utc_now(),
        "queue_id": summary["queue_id"],
        "queue_sha256": _stable_json_sha256(queue),
        "mode": summary["mode"],
        "state": _repo_rel(state_path, repo_root),
        "state_watermark": state_watermark,
        "status_counts": summary.get("status_counts", {}),
        "observe_read_only": True,
        "healthy": not blockers,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "definition_drift": definition_drift,
        "performance": performance,
        "runtime_policy": runtime_policy,
        "auto_parallelism": _auto_parallelism_observation(queue),
        "step_count": summary.get("step_count"),
        "orphaned_step_count": summary.get("orphaned_step_count"),
        "ready_steps": summary.get("ready_steps", []),
        "running_steps": running,
        "failed_steps": failed,
        "queued_steps": queued,
        "blocked_steps": blocked,
        "succeeded_artifact_steps": succeeded_artifact_steps,
        "succeeded_artifact_failure_steps": succeeded_artifact_failure_steps,
        "orphaned_steps": orphaned,
        "suggested_commands": {
            "refresh": (
                f".venv/bin/python tools/experiment_queue.py --queue <queue-path> observe --tail-lines {tail_lines}"
            ),
            "pause": ".venv/bin/python tools/experiment_queue.py --queue <queue-path> control paused --reason '<reason>'",
            "resume": ".venv/bin/python tools/experiment_queue.py --queue <queue-path> control running --reason '<reason>'",
            "run_worker": ".venv/bin/python tools/experiment_queue.py --queue <queue-path> run-worker --execute",
        },
    }


def _observation_blockers(
    *,
    definition_drift: Mapping[str, Any],
    performance: Mapping[str, Any],
    runtime_policy: Mapping[str, Any],
    orphaned_step_count: Any,
    active_steps: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if definition_drift.get("state_missing") is True:
        blockers.append("experiment_queue_observation_state_missing")
    for key, blocker in (
        ("missing_step_count", "experiment_queue_observation_missing_steps"),
        ("changed_step_count", "experiment_queue_observation_changed_steps"),
        (
            "missing_hash_step_count",
            "experiment_queue_observation_missing_step_hashes",
        ),
    ):
        try:
            count = int(definition_drift.get(key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            blockers.append(f"{blocker}:{count}")
    if performance.get("state_missing") is True:
        blockers.append("experiment_queue_observation_performance_state_missing")
    if runtime_policy.get("state_missing") is True:
        blockers.append("experiment_queue_observation_runtime_policy_state_missing")
    try:
        orphan_count = int(orphaned_step_count or 0)
    except (TypeError, ValueError):
        orphan_count = 0
    if orphan_count > 0:
        blockers.append(f"experiment_queue_observation_orphaned_steps:{orphan_count}")
    failed_steps = [step for step in active_steps if isinstance(step, Mapping) and step.get("status") == "failed"]
    if failed_steps:
        blockers.append(f"experiment_queue_observation_failed_steps:{len(failed_steps)}")
    blocked_steps = [step for step in active_steps if isinstance(step, Mapping) and step.get("status") == "blocked"]
    if blocked_steps:
        blockers.append(f"experiment_queue_observation_blocked_steps:{len(blocked_steps)}")
    artifact_failures = 0
    for step in active_steps:
        status = step.get("status") if isinstance(step, Mapping) else None
        artifacts = step.get("expected_artifacts") if isinstance(step, Mapping) else None
        if not isinstance(artifacts, Sequence) or isinstance(
            artifacts,
            (str, bytes, bytearray),
        ):
            continue
        artifact_failures += sum(
            1
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            and (status == "succeeded" or artifact.get("exists") is True)
            and artifact.get("postcondition_passed") is False
        )
    if artifact_failures:
        blockers.append(f"experiment_queue_observation_artifact_postcondition_failures:{artifact_failures}")
    return blockers


def render_observation_markdown(observation: Mapping[str, Any]) -> str:
    """Render a small live dashboard suitable for terminal or memo paste."""

    lines = [
        f"# Experiment Queue Observation: {observation.get('queue_id')}",
        "",
        f"- generated_at_utc: `{observation.get('generated_at_utc')}`",
        f"- mode: `{observation.get('mode')}`",
        f"- state: `{observation.get('state')}`",
        f"- status_counts: `{observation.get('status_counts')}`",
        f"- healthy: `{observation.get('healthy')}`",
        f"- blockers: `{observation.get('blockers')}`",
        f"- auto_parallelism: `{observation.get('auto_parallelism')}`",
        f"- performance: `{_performance_markdown_summary(observation.get('performance'))}`",
        f"- runtime_policy: `{_runtime_policy_markdown_summary(observation.get('runtime_policy'))}`",
        f"- orphaned_step_count: `{observation.get('orphaned_step_count')}`",
        "",
        "| status | experiment | step | log | artifacts | processes |",
        "|---|---|---|---|---:|---:|",
    ]
    steps = [
        *list(observation.get("running_steps") or []),
        *list(observation.get("failed_steps") or []),
        *list(observation.get("queued_steps") or []),
        *list(observation.get("blocked_steps") or []),
        *list(observation.get("succeeded_artifact_failure_steps") or []),
    ]
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        artifacts = step.get("expected_artifacts") or []
        passed = sum(1 for item in artifacts if isinstance(item, Mapping) and item.get("postcondition_passed"))
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


def _runtime_policy_markdown_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        "schema": value.get("schema"),
        "advisory_only": value.get("advisory_only"),
        "score_claim": value.get("score_claim"),
        "recommended_max_concurrency": value.get("recommended_max_concurrency"),
        "recommended_timeout_seconds_by_resource": value.get("recommended_timeout_seconds_by_resource"),
    }


__all__ = [
    "OBSERVATION_SCHEMA",
    "observe_experiment_queue",
    "render_observation_markdown",
]
