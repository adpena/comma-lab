# SPDX-License-Identifier: MIT
"""Queue-owned policy records for materializer feedback replanning."""

from __future__ import annotations

import hashlib
import json
import posixpath
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import truthy_authority_field_violations

from .experiment_queue import (
    DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
    DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
    QUEUE_SCHEMA,
    ExperimentQueueError,
    normalize_queue_definition,
)

QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA = "queue_feedback_replan_policy.v1"
QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA = (
    "queue_feedback_replan_child_queue_validation.v1"
)
QUEUE_FEEDBACK_REPLAN_CONTINUATION_METADATA_SCHEMA = (
    "queue_feedback_replan_continuation_metadata.v1"
)
QUEUE_FEEDBACK_CANDIDATE_WIDENING_METADATA_SCHEMA = (
    "queue_feedback_candidate_widening_metadata.v1"
)
QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_METADATA_SCHEMA = (
    "queue_feedback_candidate_actuation_planning_metadata.v1"
)
QUEUE_OBSERVATION_RECOVERY_QUEUE_METADATA_SCHEMA = (
    "queue_observation_recovery_queue_metadata.v1"
)
QUEUE_OBSERVATION_RECOVERY_QUEUE_VALIDATION_SCHEMA = (
    "queue_observation_recovery_queue_validation.v1"
)
QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA = "queue_observation_recovery_plan.v1"
MATERIALIZER_CAMPAIGN_RUN_SCHEMA = "byte_shaving_materializer_campaign_run.v1"
QUEUE_FEEDBACK_REPLAN_CONTINUATION_EXPERIMENT_ID = (
    "queue_feedback_replan_next_materializer_iteration"
)
QUEUE_FEEDBACK_REPLAN_CONTINUATION_STEP_ID = "run_next_materializer_campaign_iteration"
QUEUE_FEEDBACK_CANDIDATE_WIDENING_EXPERIMENT_ID = (
    "queue_feedback_widen_inverse_candidate_generation"
)
QUEUE_FEEDBACK_CANDIDATE_WIDENING_STEP_ID = "widen_inverse_candidate_generation"
QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_EXPERIMENT_ID = (
    "queue_feedback_plan_widened_candidate_actuation"
)
QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID = (
    "plan_widened_inverse_action_materialization"
)
QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_STEP_ID = (
    "compile_widened_inverse_action_materializer_work"
)
QUEUE_OBSERVATION_RECOVERY_EXPERIMENT_ID = "queue_observation_recovery_actions"
QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL = (
    "tools/run_byte_shaving_materializer_campaign.py"
)
QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL = (
    "tools/build_inverse_steganalysis_action_functional.py"
)
QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_TOOL = "tools/plan_byte_shaving_campaign.py"
QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_TOOL = (
    "tools/build_byte_shaving_campaign_queue.py"
)
QUEUE_OBSERVATION_RECOVERY_TOOL = "tools/experiment_queue.py"
QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_FLAGS = frozenset(
    {
        "--allow-paid-dispatch-queue",
        "--contest-auth-eval",
        "--cuda-auth",
        "--dispatch-mode",
        "--exact-auth-eval",
        "--exact-eval",
        "--modal",
        "--provider",
        "--submit",
    }
)
QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_WRAPPERS = frozenset(
    {"bash", "sh", "zsh", "ssh", "scp", "rsync", "osascript"}
)

FORBIDDEN_TRUE_AUTHORITY_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
            *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
            "dispatch_packet_ready",
        )
    )
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
    "promotable": False,
}

ACTION_EXECUTE_FEEDBACK_FOLLOWUP = "execute_feedback_followup_queue"
ACTION_RUN_NEXT_ITERATION = "run_next_materializer_campaign_iteration"
ACTION_INSPECT_EXACT_HANDOFFS = "inspect_exact_readiness_handoffs"
ACTION_RECOVER_QUEUE_HEALTH = "recover_queue_health"
ACTION_QUEUE_OBSERVATION_MAINTENANCE = "queue_observation_maintenance"
ACTION_WIDEN_CANDIDATE_GENERATION = "widen_inverse_candidate_generation"
ACTION_BLOCKED = "blocked"
ACTION_STOP_MAX_ITERATIONS = "stop_max_iterations"
ACTION_REFUSE = "refuse_false_authority_or_schema"


def _false_authority_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.update(FALSE_AUTHORITY)
    return out


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _nonempty_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_json_file_sha256(path: str | None) -> str | None:
    path_text = _nonempty_str(path)
    if path_text is None:
        return None
    try:
        payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        payload = normalize_queue_definition(payload)
    except ExperimentQueueError:
        return None
    return _stable_json_sha256(payload)


def _file_sha256(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _source_state_watermark_from_path(
    path: str | None,
    *,
    queue_id: str | None,
) -> dict[str, Any] | None:
    path_text = _nonempty_str(path)
    queue_id_text = _nonempty_str(queue_id)
    if path_text is None or queue_id_text is None:
        return None
    try:
        conn = sqlite3.connect(f"file:{Path(path_text).resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return None
    try:
        events = conn.execute(
            """
            SELECT COUNT(*) AS event_count, COALESCE(MAX(id), 0) AS max_event_id
            FROM queue_events
            WHERE queue_id = ?
            """,
            (queue_id_text,),
        ).fetchone()
        steps = conn.execute(
            """
            SELECT COUNT(*) AS step_state_count,
                   COALESCE(MAX(updated_at_utc), '') AS max_step_updated_at_utc
            FROM step_state
            WHERE queue_id = ?
            """,
            (queue_id_text,),
        ).fetchone()
        control = conn.execute(
            """
            SELECT mode, updated_at_utc
            FROM queue_controls
            WHERE queue_id = ?
            """,
            (queue_id_text,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return {
        "schema": "experiment_queue_state_watermark.v1",
        "queue_id": queue_id_text,
        "state_path": path_text,
        "event_count": int(events["event_count"] or 0) if events else 0,
        "max_event_id": int(events["max_event_id"] or 0) if events else 0,
        "step_state_count": int(steps["step_state_count"] or 0) if steps else 0,
        "max_step_updated_at_utc": (
            str(steps["max_step_updated_at_utc"] or "") if steps else ""
        ),
        "control_mode": str(control["mode"] or "") if control else "",
        "control_updated_at_utc": str(control["updated_at_utc"] or "") if control else "",
    }


def _experiment_queue_command(
    *,
    queue_path: str,
    state_path: str,
    subcommand: Sequence[str],
) -> list[str]:
    return [
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        queue_path,
        "--state",
        state_path,
        *[str(item) for item in subcommand],
    ]


def _blocker_matches(blockers: Sequence[str], prefix: str) -> bool:
    return any(item == prefix or item.startswith(f"{prefix}:") for item in blockers)


def _step_identity(step: Mapping[str, Any]) -> tuple[str | None, str | None]:
    experiment_id = _nonempty_str(step.get("experiment_id"))
    step_id = _nonempty_str(step.get("step_id"))
    return experiment_id, step_id


def _artifact_paths_from_step(step: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for artifact in _as_list(step.get("expected_artifacts")):
        if not isinstance(artifact, Mapping):
            continue
        path = _nonempty_str(artifact.get("path"))
        if path:
            paths.append(path)
    paths.extend(_string_list(step.get("expected_artifact_paths")))
    return list(dict.fromkeys(paths))


def _recovery_action(
    *,
    action: str,
    reason: str,
    required: bool,
    command: Sequence[str] | None = None,
    step: Mapping[str, Any] | None = None,
    blocker_sources: Sequence[str] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "reason": reason,
        "required": bool(required),
        "local_only": True,
        "requires_explicit_execution": True,
        "command": None if command is None else [str(item) for item in command],
        "blocker_sources": [str(item) for item in blocker_sources],
    }
    if step is not None:
        experiment_id, step_id = _step_identity(step)
        payload.update(
            {
                "experiment_id": experiment_id,
                "step_id": step_id,
                "status": step.get("status"),
                "target_kind": _nonempty_str(step.get("target_kind")),
                "materializer_id": _nonempty_str(
                    step.get("materializer_id") or step.get("materializer")
                ),
                "receiver_contract_kind": _nonempty_str(
                    step.get("receiver_contract_kind")
                ),
                "resource_kind": _nonempty_str(step.get("resource_kind")),
                "candidate_ids": _string_list(step.get("candidate_ids")),
                "work_ids": _string_list(step.get("work_ids")),
                "backlog_keys": _string_list(step.get("backlog_keys")),
                "source_unit_ids": _string_list(step.get("source_unit_ids")),
                "source_selection_ids": _string_list(
                    step.get("source_selection_ids")
                ),
                "expected_artifact_paths": _artifact_paths_from_step(step),
            }
        )
    return _false_authority_payload(payload)


def _append_unique_action(actions: list[dict[str, Any]], action: dict[str, Any]) -> None:
    key = (
        action.get("action"),
        tuple(action.get("command") or []),
        action.get("experiment_id"),
        action.get("step_id"),
        action.get("required"),
    )
    for existing in actions:
        existing_key = (
            existing.get("action"),
            tuple(existing.get("command") or []),
            existing.get("experiment_id"),
            existing.get("step_id"),
            existing.get("required"),
        )
        if existing_key == key:
            return
    actions.append(action)


def _blocker_family(blocker: Any) -> str:
    return str(blocker or "unknown_blocker").split(":", 1)[0] or "unknown_blocker"


def _action_group_scope(action: Mapping[str, Any]) -> tuple[str, str]:
    materializer = _nonempty_str(action.get("materializer_id"))
    receiver = _nonempty_str(action.get("receiver_contract_kind"))
    target = _nonempty_str(action.get("target_kind"))
    if materializer and receiver:
        return "materializer_receiver", f"{materializer}:{receiver}"
    if materializer and target:
        return "materializer_target", f"{materializer}:{target}"
    if receiver and target:
        return "receiver_target", f"{receiver}:{target}"
    for key in (
        "source_selection_ids",
        "source_unit_ids",
        "work_ids",
        "backlog_keys",
        "expected_artifact_paths",
        "candidate_ids",
    ):
        values = _string_list(action.get(key))
        if values:
            return key, "|".join(values)
    experiment_id = _nonempty_str(action.get("experiment_id"))
    step_id = _nonempty_str(action.get("step_id"))
    if experiment_id and step_id:
        return "step", f"{experiment_id}.{step_id}"
    action_name = _nonempty_str(action.get("action")) or "unknown_action"
    return "action", action_name


def _recommended_planning_effect(actions: Sequence[Mapping[str, Any]]) -> str:
    if not any(action.get("required") is True for action in actions):
        return "advisory_maintenance_only"
    if any(action.get("command") is None for action in actions):
        return "operator_inspection_required_before_followup"
    return "block_followup_until_recovery_queue_runs"


def _queue_recovery_grouped_blockers(
    actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups_by_id: dict[str, dict[str, Any]] = {}
    actions_by_id: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        blocker_sources = _string_list(action.get("blocker_sources")) or [
            _nonempty_str(action.get("action")) or "unknown_action"
        ]
        scope_kind, scope_value = _action_group_scope(action)
        for blocker in blocker_sources:
            blocker_family = _blocker_family(blocker)
            group_id = f"{blocker_family}:{scope_kind}:{scope_value}"
            group = groups_by_id.setdefault(
                group_id,
                {
                    "schema": "queue_observation_recovery_blocker_group.v1",
                    "group_id": group_id,
                    "blocker_family": blocker_family,
                    "scope_kind": scope_kind,
                    "scope_value": scope_value,
                    "blockers": [],
                    "actions": [],
                    "action_names": [],
                    "affected_experiment_ids": [],
                    "affected_step_ids": [],
                    "statuses": [],
                    "target_kinds": [],
                    "materializer_ids": [],
                    "receiver_contract_kinds": [],
                    "resource_kinds": [],
                    "candidate_ids": [],
                    "work_ids": [],
                    "backlog_keys": [],
                    "source_unit_ids": [],
                    "source_selection_ids": [],
                    "expected_artifact_paths": [],
                },
            )
            actions_by_id.setdefault(group_id, []).append(dict(action))
            _extend_unique(group["blockers"], [blocker])
            _extend_unique(group["actions"], [action.get("action")])
            _extend_unique(group["action_names"], [action.get("action")])
            _extend_unique(group["affected_experiment_ids"], [action.get("experiment_id")])
            _extend_unique(group["affected_step_ids"], [action.get("step_id")])
            _extend_unique(group["statuses"], [action.get("status")])
            _extend_unique(group["target_kinds"], [action.get("target_kind")])
            _extend_unique(group["materializer_ids"], [action.get("materializer_id")])
            _extend_unique(
                group["receiver_contract_kinds"],
                [action.get("receiver_contract_kind")],
            )
            _extend_unique(group["resource_kinds"], [action.get("resource_kind")])
            for key in (
                "candidate_ids",
                "work_ids",
                "backlog_keys",
                "source_unit_ids",
                "source_selection_ids",
                "expected_artifact_paths",
            ):
                _extend_unique(group[key], _string_list(action.get(key)))
    grouped: list[dict[str, Any]] = []
    for group_id, group in sorted(groups_by_id.items()):
        group_actions = actions_by_id[group_id]
        count = len(group_actions)
        required_action_count = sum(
            1 for action in group_actions if action.get("required") is True
        )
        grouped.append(
            _false_authority_payload(
                {
                    **group,
                    "count": count,
                    "repeated": count > 1,
                    "required_action_count": required_action_count,
                    "maintenance_action_count": count - required_action_count,
                    "recommended_planning_effect": _recommended_planning_effect(
                        group_actions
                    ),
                    "allowed_use": "queue_recovery_grouping_and_planning_only",
                    "forbidden_use": (
                        "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                    ),
                }
            )
        )
    return grouped


def _extend_unique(out: list[str], values: Sequence[Any]) -> None:
    seen = set(out)
    for value in values:
        text = _nonempty_str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)


def build_queue_observation_recovery_plan(
    observation: Mapping[str, Any] | None,
    *,
    queue_path: str,
    state_path: str,
    reason: str = "queue observation health recovery",
) -> dict[str, Any]:
    """Convert queue observation blockers into local recovery/maintenance actions."""

    if not isinstance(observation, Mapping):
        observation = {
            "schema": "experiment_queue_observation_unavailable.v1",
            "healthy": False,
            "blockers": ["queue_observation_missing_or_invalid"],
            "blocker_count": 1,
        }
    blockers = _string_list(observation.get("blockers"))
    actions: list[dict[str, Any]] = []
    state_blockers = [
        blocker
        for blocker in blockers
        if blocker == "experiment_queue_observation_state_missing"
        or blocker.startswith("experiment_queue_observation_missing_steps:")
        or blocker.startswith("experiment_queue_observation_changed_steps:")
        or blocker.startswith("experiment_queue_observation_missing_step_hashes:")
    ]
    if state_blockers:
        _append_unique_action(
            actions,
            _recovery_action(
                action="refresh_queue_state",
                reason=reason,
                required=True,
                command=_experiment_queue_command(
                    queue_path=queue_path,
                    state_path=state_path,
                    subcommand=["init"],
                ),
                blocker_sources=state_blockers,
            ),
        )

    orphan_count = _safe_int(observation.get("orphaned_step_count"))
    orphaned_steps = [
        item for item in _as_list(observation.get("orphaned_steps")) if isinstance(item, Mapping)
    ]
    orphan_blockers = [
        item
        for item in blockers
        if item.startswith("experiment_queue_observation_orphaned_steps")
    ]
    if orphan_count and not orphaned_steps:
        _append_unique_action(
            actions,
            _recovery_action(
                action="observe_orphaned_steps",
                reason="orphan count present without identities",
                required=True,
                command=_experiment_queue_command(
                    queue_path=queue_path,
                    state_path=state_path,
                    subcommand=["observe", "--include-orphans", "--format", "json"],
                ),
                blocker_sources=orphan_blockers,
            ),
        )
    blocking_orphans = [
        step
        for step in orphaned_steps
        if str(step.get("status") or "") in {"queued", "running", "blocked"}
    ]
    if blocking_orphans:
        _append_unique_action(
            actions,
            _recovery_action(
                action="retire_blocking_orphaned_steps",
                reason=reason,
                required=True,
                command=_experiment_queue_command(
                    queue_path=queue_path,
                    state_path=state_path,
                    subcommand=["retire-orphans", "--reason", reason],
                ),
                blocker_sources=orphan_blockers,
            ),
        )
    elif orphaned_steps:
        _append_unique_action(
            actions,
            _recovery_action(
                action="record_nonblocking_orphaned_steps",
                reason="nonblocking orphaned rows retained for cleanup/audit",
                required=False,
                blocker_sources=orphan_blockers,
            ),
        )

    step_blockers = [
        item
        for item in blockers
        if _blocker_matches([item], "experiment_queue_observation_failed_steps")
        or _blocker_matches([item], "experiment_queue_observation_blocked_steps")
        or _blocker_matches(
            [item],
            "experiment_queue_observation_artifact_postcondition_failures",
        )
    ]
    for action_name, key, action_reason in (
        ("rewind_failed_step", "failed_steps", "failed queue step"),
        ("rewind_blocked_step", "blocked_steps", "blocked queue step"),
        (
            "rewind_succeeded_step_with_artifact_failure",
            "succeeded_artifact_failure_steps",
            "succeeded step has missing or corrupt declared artifact",
        ),
    ):
        for step in _as_list(observation.get(key)):
            if not isinstance(step, Mapping):
                continue
            experiment_id, step_id = _step_identity(step)
            if experiment_id is None or step_id is None:
                continue
            _append_unique_action(
                actions,
                _recovery_action(
                    action=action_name,
                    reason=action_reason,
                    required=True,
                    command=_experiment_queue_command(
                        queue_path=queue_path,
                        state_path=state_path,
                        subcommand=[
                            "rewind",
                            experiment_id,
                            step_id,
                            "--reason",
                            reason,
                        ],
                    ),
                    step=step,
                    blocker_sources=step_blockers,
                ),
            )

    if _blocker_matches(
        blockers,
        "experiment_queue_observation_artifact_postcondition_failures",
    ) and not any(
        item.get("action") == "rewind_succeeded_step_with_artifact_failure"
        for item in actions
    ):
        _append_unique_action(
            actions,
            _recovery_action(
                action="inspect_artifact_postcondition_failures",
                reason="artifact postcondition failure lacks a succeeded-step identity",
                required=True,
                blocker_sources=step_blockers,
            ),
        )

    required_actions = [item for item in actions if item.get("required") is True]
    maintenance_actions = [item for item in actions if item.get("required") is not True]
    grouped_blockers = _queue_recovery_grouped_blockers(actions)
    repeated_group_count = sum(1 for group in grouped_blockers if group["repeated"])
    queue_id = _nonempty_str(observation.get("queue_id"))
    state_watermark = (
        dict(observation.get("state_watermark"))
        if isinstance(observation.get("state_watermark"), Mapping)
        else _source_state_watermark_from_path(state_path, queue_id=queue_id)
    )
    if state_watermark is None and _blocker_matches(
        blockers,
        "experiment_queue_observation_state_missing",
    ):
        state_watermark = {
            "schema": "experiment_queue_state_watermark.v1",
            "queue_id": queue_id,
            "state_path": state_path,
            "state_missing": True,
            "event_count": 0,
            "max_event_id": 0,
            "step_state_count": 0,
            "max_step_updated_at_utc": "",
            "control_mode": "",
            "control_updated_at_utc": "",
        }
    source_queue_sha256 = _nonempty_str(observation.get("queue_sha256")) or (
        _stable_json_file_sha256(queue_path)
    )
    return _false_authority_payload(
        {
            "schema": QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA,
            "source_schema": observation.get("schema"),
            "queue_id": queue_id,
            "queue_path": queue_path,
            "queue_state_path": state_path,
            "source_queue_sha256": source_queue_sha256,
            "source_state_watermark": state_watermark,
            "source_observation_generated_at_utc": observation.get("generated_at_utc"),
            "observation_healthy": observation.get("healthy") is True,
            "observation_blockers": blockers,
            "observation_blocker_count": len(blockers),
            "recovery_required": bool(required_actions),
            "maintenance_recommended": bool(maintenance_actions),
            "required_action_count": len(required_actions),
            "maintenance_action_count": len(maintenance_actions),
            "action_count": len(actions),
            "actions": actions,
            "grouped_blocker_count": len(grouped_blockers),
            "repeated_group_count": repeated_group_count,
            "grouped_blockers": grouped_blockers,
            "queue_health_groups": grouped_blockers,
            "commands": [
                list(item["command"])
                for item in actions
                if item.get("command") is not None
            ],
            "allowed_use": "local_queue_recovery_planning_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        }
    )


def _command_arg(command_items: Sequence[str], flag: str) -> str | None:
    for index, item in enumerate(command_items):
        if item == flag:
            value_index = index + 1
            if value_index >= len(command_items):
                return None
            return _nonempty_str(command_items[value_index])
        prefix = f"{flag}="
        if item.startswith(prefix):
            return _nonempty_str(item[len(prefix) :])
    return None


def _command_arg_values(command_items: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command_items):
        if item == flag:
            value_index = index + 1
            if value_index < len(command_items):
                value = _nonempty_str(command_items[value_index])
                if value is not None:
                    values.append(value)
            continue
        prefix = f"{flag}="
        if item.startswith(prefix):
            value = _nonempty_str(item[len(prefix) :])
            if value is not None:
                values.append(value)
    return values


def _has_command_flag(command_items: Sequence[str], flag: str) -> bool:
    return bool(_command_arg_values(command_items, flag))


def _forbidden_command_flag_uses(command_items: Sequence[str]) -> list[str]:
    uses: list[str] = []
    for item in command_items:
        for flag in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_FLAGS:
            if item == flag or item.startswith(f"{flag}="):
                uses.append(item)
    return list(dict.fromkeys(uses))


def _is_local_python_command(command0: str) -> bool:
    name = posixpath.basename(command0)
    return name in {"python", "python3"} or name.startswith("python3.")


def _without_flag(command_items: Sequence[str], flag: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for item in command_items:
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            skip_next = True
            continue
        if item.startswith(f"{flag}="):
            continue
        out.append(str(item))
    return out


def _replace_or_append_flag(
    command_items: Sequence[str],
    flag: str,
    value: str,
) -> list[str]:
    out: list[str] = []
    replaced = False
    skip_next = False
    for item in command_items:
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            out.extend([flag, value])
            replaced = True
            skip_next = True
            continue
        if item.startswith(f"{flag}="):
            out.extend([flag, value])
            replaced = True
            continue
        out.append(str(item))
    if not replaced:
        out.extend([flag, value])
    return out


def _widened_json_path(path_text: str | None, *, suffix: str) -> str | None:
    path_text = _nonempty_str(path_text)
    if path_text is None:
        return None
    path = Path(path_text)
    name = path.name
    if name.endswith(".json"):
        return path.with_name(f"{name[:-5]}.{suffix}.json").as_posix()
    return path.with_name(f"{name}.{suffix}.json").as_posix()


def _inverse_action_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    prefix = "inverse_steganalysis_action_functional"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        return suffix or ".actuation"
    return f".{name}.actuation"


def _candidate_actuation_paths(widened_action_path: str) -> dict[str, str]:
    path = Path(widened_action_path)
    suffix = _inverse_action_suffix(path)
    return {
        "campaign_plan": path.with_name(
            f"byte_shaving_campaign_plan{suffix}.json"
        ).as_posix(),
        "campaign_plan_md": path.with_name(
            f"byte_shaving_campaign_plan{suffix}.md"
        ).as_posix(),
        "inverse_action_materialization_bridge": path.with_name(
            f"inverse_action_materialization_bridge{suffix}.json"
        ).as_posix(),
        "materialization": path.with_name(f"materialization{suffix}.json").as_posix(),
        "portfolio": path.with_name(f"portfolio{suffix}.json").as_posix(),
        "action_summary": path.with_name(f"action_summary{suffix}.json").as_posix(),
        "materializer_backlog": path.with_name(
            f"materializer_backlog{suffix}.json"
        ).as_posix(),
        "materializer_work_queue": path.with_name(
            f"materializer_work_queue{suffix}.json"
        ).as_posix(),
    }


def _resolve_maybe_repo_path(path_text: str | None) -> Path | None:
    path_text = _nonempty_str(path_text)
    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def _json_file_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _path_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _nearby_scorer_response_seed_dirs(
    run_summary: Mapping[str, Any],
) -> list[Path]:
    run_dir = _resolve_maybe_repo_path(run_summary.get("run_dir"))
    plan_path = _resolve_maybe_repo_path(run_summary.get("plan"))
    seed_dirs: list[Path] = []
    for seed in (run_dir, plan_path.parent if plan_path is not None else None):
        if seed is None:
            continue
        for directory in [seed, *list(seed.parents[:4])]:
            if directory not in seed_dirs:
                seed_dirs.append(directory)
    return seed_dirs


def _nearest_seed_dir_text(path: Path, seed_dirs: Sequence[Path]) -> str | None:
    try:
        resolved_path = path.resolve()
    except OSError:
        return None
    matches: list[Path] = []
    for seed in seed_dirs:
        try:
            resolved_seed = seed.resolve()
        except OSError:
            continue
        try:
            resolved_path.relative_to(resolved_seed)
        except ValueError:
            continue
        matches.append(resolved_seed)
    if not matches:
        return None
    matches.sort(key=lambda item: len(item.parts), reverse=True)
    return _path_text(matches[0])


def _scorer_response_discovery_record(
    path: Path,
    *,
    seed_dirs: Sequence[Path],
) -> dict[str, Any] | None:
    payload = _json_file_payload(path)
    if payload is None or payload.get("schema") != "scorer_response_dataset.v1":
        return None

    blockers: list[str] = []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        row_count = 0
        blockers.append("scorer_response_rows_not_list")
    else:
        row_count = len(rows)
        if row_count <= 0:
            blockers.append("scorer_response_rows_empty")
    producer = _nonempty_str(payload.get("producer"))
    if producer is None:
        blockers.append("scorer_response_producer_missing")
    sha256 = _file_sha256(path)
    if sha256 is None:
        blockers.append("scorer_response_file_sha256_unavailable")
    relation_seed = _nearest_seed_dir_text(path, seed_dirs)
    if relation_seed is None:
        blockers.append("scorer_response_not_related_to_run_or_plan")
    blockers.extend(
        f"scorer_response_truthy_authority_field:{violation}"
        for violation in truthy_authority_field_violations(
            payload,
            fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
        )
    )

    size_bytes: int | None
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = None
    return {
        "path": _path_text(path),
        "sha256": sha256,
        "bytes": size_bytes,
        "row_count": row_count,
        "producer": producer,
        "relation_seed_dir": relation_seed,
        "usable": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _discover_nearby_scorer_response_sources(
    run_summary: Mapping[str, Any],
    *,
    max_paths: int = 8,
) -> dict[str, Any]:
    candidates: list[Path] = []
    seed_dirs = _nearby_scorer_response_seed_dirs(run_summary)
    for directory in seed_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*scorer_response*.json")):
            if path not in candidates:
                candidates.append(path)
    records: list[dict[str, Any]] = []
    for path in candidates:
        record = _scorer_response_discovery_record(path, seed_dirs=seed_dirs)
        if record is None:
            continue
        records.append(record)
        if len(records) >= max_paths:
            break
    usable_records = [record for record in records if record.get("usable") is True]
    blockers: list[str] = []
    if len(usable_records) > 1:
        blockers.append(
            "candidate_widening_ambiguous_nearby_scorer_response_sources:"
            + ",".join(str(record.get("path")) for record in usable_records)
        )
    if records and not usable_records:
        blockers.append("candidate_widening_no_usable_discovered_scorer_response")
        for record in records:
            for blocker in _string_list(record.get("blockers")):
                blockers.append(
                    "candidate_widening_discovered_scorer_response_invalid:"
                    f"{record.get('path')}:{blocker}"
                )
    return {
        "records": records,
        "usable_records": usable_records,
        "paths": [str(record.get("path")) for record in records],
        "usable_paths": [str(record.get("path")) for record in usable_records],
        "blockers": list(dict.fromkeys(blockers)),
    }


def _id_fragment(value: Any) -> str:
    text = str(value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in out.split("_") if part) or "unknown"


def _normalized_posix_path(path: str) -> str:
    text = path.strip().replace("\\", "/")
    return posixpath.normpath(text)


def _path_is_under(child: str, parent: str) -> bool:
    child_norm = _normalized_posix_path(child)
    parent_norm = _normalized_posix_path(parent)
    if parent_norm in {"", "."}:
        return False
    if child_norm == parent_norm:
        return True
    return child_norm.startswith(parent_norm.rstrip("/") + "/")


def _telemetry_truncation_markers(payload: Any, *, prefix: str = "") -> list[str]:
    markers: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if "truncat" in key_text.lower() and value:
                markers.append(f"{path}=truthy")
            markers.extend(_telemetry_truncation_markers(value, prefix=path))
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            markers.extend(
                _telemetry_truncation_markers(value, prefix=f"{prefix}[{index}]")
            )
    return markers


def _exact_auth_calibration_pair_policy(
    run_summary: Mapping[str, Any],
) -> dict[str, Any]:
    request = run_summary.get("queue_feedback_replan_request")
    request = request if isinstance(request, Mapping) else {}
    packet_paths = _string_list(request.get("exact_auth_calibration_packet_paths"))
    pair = request.get("exact_auth_calibration_discovery_pair")
    pair = pair if isinstance(pair, Mapping) else None
    blockers: list[str] = []

    if packet_paths and len(packet_paths) != 2:
        blockers.append("exact_auth_calibration_requires_exactly_two_packet_paths")
    if packet_paths and pair is None:
        blockers.append("exact_auth_calibration_pair_metadata_missing")
    if pair is not None:
        required_pair_fields = (
            "archive_sha256",
            "archive_bytes",
            "n_samples",
            "runtime_content_tree_sha256",
            "contest_cpu_packet_path",
            "contest_cuda_packet_path",
        )
        for field in required_pair_fields:
            if _nonempty_str(pair.get(field)) is None:
                blockers.append(f"exact_auth_calibration_pair_field_missing:{field}")
        if _safe_int(pair.get("archive_bytes")) < 1:
            blockers.append("exact_auth_calibration_archive_bytes_invalid")
        if _safe_int(pair.get("n_samples")) < 1:
            blockers.append("exact_auth_calibration_n_samples_invalid")
        pair_paths = {
            _nonempty_str(pair.get("contest_cpu_packet_path")),
            _nonempty_str(pair.get("contest_cuda_packet_path")),
        }
        if packet_paths and pair_paths != set(packet_paths):
            blockers.append("exact_auth_calibration_pair_paths_mismatch")

    return _false_authority_payload(
        {
            "source": _nonempty_str(
                request.get("exact_auth_calibration_packet_source")
            ),
            "packet_paths": packet_paths,
            "packet_count": len(packet_paths),
            "pair": dict(pair) if pair is not None else None,
            "usable_for_feedback_trust_region": bool(packet_paths) and not blockers,
            "blockers": list(dict.fromkeys(blockers)),
            "allowed_use": "feedback_replan_trust_region_calibration_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_dispatch_authority",
        }
    )


def _queue_observation_recovery_plan_from_run(
    run_summary: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    direct = run_summary.get("queue_observation_recovery_plan")
    if isinstance(direct, Mapping):
        return direct
    request = run_summary.get("queue_feedback_replan_request")
    if isinstance(request, Mapping):
        nested = request.get("queue_observation_recovery_plan")
        if isinstance(nested, Mapping):
            return nested
    return None


def validate_feedback_followup_queue(
    child_queue: Mapping[str, Any],
    *,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """Validate a local feedback child queue before a policy may execute it."""

    blockers: list[str] = []
    warnings: list[str] = []
    controls = child_queue.get("controls")
    experiments = child_queue.get("experiments")
    step_count = 0
    command_count = 0
    output_paths: list[str] = []

    if child_queue.get("schema") != QUEUE_SCHEMA:
        blockers.append("queue_feedback_replan_followup_schema_not_experiment_queue")

    if not isinstance(controls, Mapping):
        blockers.append("queue_feedback_replan_followup_controls_missing")
        control_mode = None
        local_first = None
        max_concurrency = None
    else:
        control_mode = controls.get("mode")
        local_first = controls.get("local_first")
        max_concurrency = controls.get("max_concurrency")
        if control_mode != "paused":
            blockers.append("queue_feedback_replan_followup_control_mode_not_paused")
        if local_first is not True:
            blockers.append("queue_feedback_replan_followup_not_local_first")
        if isinstance(max_concurrency, Mapping):
            non_local_limits = sorted(
                str(key) for key in max_concurrency if str(key) != "local_cpu"
            )
            blockers.extend(
                f"queue_feedback_replan_followup_non_local_concurrency:{key}"
                for key in non_local_limits
            )
        elif max_concurrency not in ({}, None):
            blockers.append("queue_feedback_replan_followup_max_concurrency_not_mapping")

    for violation in truthy_authority_field_violations(
        child_queue,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append(
            f"queue_feedback_replan_followup_truthy_authority_field:{violation}"
        )

    if run_dir is None:
        blockers.append("queue_feedback_replan_followup_run_dir_missing")

    if not isinstance(experiments, list) or not experiments:
        blockers.append("queue_feedback_replan_followup_experiments_missing")
    else:
        for experiment_index, experiment in enumerate(experiments):
            if not isinstance(experiment, Mapping):
                blockers.append(
                    f"queue_feedback_replan_followup_experiment_not_object:{experiment_index}"
                )
                continue
            steps = experiment.get("steps")
            if not isinstance(steps, list) or not steps:
                blockers.append(
                    f"queue_feedback_replan_followup_steps_missing:{experiment_index}"
                )
                continue
            for step_index, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    blockers.append(
                        "queue_feedback_replan_followup_step_not_object:"
                        f"{experiment_index}:{step_index}"
                    )
                    continue
                step_count += 1
                resources = step.get("resources")
                if not isinstance(resources, Mapping) or resources.get("kind") != "local_cpu":
                    blockers.append(
                        "queue_feedback_replan_followup_step_not_local_cpu:"
                        f"{experiment_index}:{step_index}"
                    )
                command = step.get("command")
                if not isinstance(command, list) or not command:
                    blockers.append(
                        "queue_feedback_replan_followup_step_command_missing:"
                        f"{experiment_index}:{step_index}"
                    )
                    continue
                command_count += 1
                command_items = [str(item) for item in command]
                command0_name = posixpath.basename(command_items[0])
                if command0_name in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_WRAPPERS:
                    blockers.append(
                        "queue_feedback_replan_followup_step_command_shell_or_remote_wrapper:"
                        f"{experiment_index}:{step_index}:{command0_name}"
                    )
                if not _is_local_python_command(command_items[0]):
                    blockers.append(
                        "queue_feedback_replan_followup_step_command_not_local_python:"
                        f"{experiment_index}:{step_index}"
                    )
                if len(command_items) < 2 or command_items[1] != (
                    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL
                ):
                    blockers.append(
                        "queue_feedback_replan_followup_step_command_not_action_functional_tool:"
                        f"{experiment_index}:{step_index}"
                    )
                blockers.extend(
                    "queue_feedback_replan_followup_step_command_forbidden_flag:"
                    f"{experiment_index}:{step_index}:{flag}"
                    for flag in _forbidden_command_flag_uses(command_items)
                )
                output_path = _command_arg(command_items, "--output")
                md_path = _command_arg(command_items, "--md-out")
                if output_path is None:
                    blockers.append(
                        "queue_feedback_replan_followup_step_output_path_missing:"
                        f"{experiment_index}:{step_index}"
                    )
                else:
                    output_paths.append(output_path)
                    if run_dir is not None and not _path_is_under(output_path, run_dir):
                        blockers.append(
                            "queue_feedback_replan_followup_step_output_outside_run_dir:"
                            f"{experiment_index}:{step_index}:{output_path}"
                        )
                if md_path is not None:
                    output_paths.append(md_path)
                    if run_dir is not None and not _path_is_under(md_path, run_dir):
                        blockers.append(
                            "queue_feedback_replan_followup_step_md_outside_run_dir:"
                            f"{experiment_index}:{step_index}:{md_path}"
                        )

    return _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA,
            "queue_schema": _nonempty_str(child_queue.get("schema")),
            "queue_id": _nonempty_str(child_queue.get("queue_id")),
            "queue_sha256": _stable_json_sha256(child_queue),
            "run_dir": run_dir,
            "control_mode": control_mode,
            "local_first": local_first,
            "max_concurrency": dict(max_concurrency)
            if isinstance(max_concurrency, Mapping)
            else max_concurrency,
            "experiment_count": len(experiments) if isinstance(experiments, list) else 0,
            "step_count": step_count,
            "command_count": command_count,
            "output_paths": output_paths,
            "valid": not blockers,
            "blockers": list(dict.fromkeys(blockers)),
            "warnings": list(dict.fromkeys(warnings)),
            "allowed_use": "local_feedback_child_queue_validation_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        }
    )


def _campaign_iteration_command(
    run_summary: Mapping[str, Any],
    *,
    next_iteration_index: int,
    max_iterations: int,
) -> list[str] | None:
    plan = _nonempty_str(run_summary.get("plan"))
    action_functional = _nonempty_str(
        run_summary.get("queue_feedback_replan_followup_action_functional_path")
    )
    if plan is None or action_functional is None:
        return None
    return [
        sys.executable,
        QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL,
        "--plan",
        plan,
        "--inverse-scorer-action-functional",
        action_functional,
        "--queue-feedback-replan-policy-iteration",
        str(next_iteration_index),
        "--queue-feedback-replan-policy-max-iterations",
        str(max_iterations),
    ]


def _feedback_action_functional_summary(path: str | None) -> dict[str, Any]:
    path_text = _nonempty_str(path)
    if path_text is None:
        return {
            "path": None,
            "loaded": False,
            "dry_no_selected_cells": False,
            "blockers": [],
        }
    resolved = Path(path_text)
    if not resolved.exists():
        return {
            "path": path_text,
            "loaded": False,
            "dry_no_selected_cells": False,
            "blockers": [],
            "warnings": ["feedback_action_functional_path_not_readable"],
        }
    blockers: list[str] = []
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {
            "path": path_text,
            "loaded": False,
            "dry_no_selected_cells": False,
            "blockers": ["feedback_action_functional_json_invalid"],
        }
    if not isinstance(payload, Mapping):
        return {
            "path": path_text,
            "loaded": False,
            "dry_no_selected_cells": False,
            "blockers": ["feedback_action_functional_not_object"],
        }
    schema = _nonempty_str(payload.get("schema"))
    if schema != "inverse_steganalysis_discrete_action_functional.v1":
        blockers.append("feedback_action_functional_schema_invalid")
    blockers.extend(
        f"feedback_action_functional_truthy_authority_field:{violation}"
        for violation in truthy_authority_field_violations(
            payload,
            fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
        )
    )
    water_bucket = payload.get("water_bucket")
    water_bucket = water_bucket if isinstance(water_bucket, Mapping) else {}
    integral_totals = payload.get("integral_totals")
    integral_totals = integral_totals if isinstance(integral_totals, Mapping) else {}
    selected_count = _safe_int(water_bucket.get("selected_count"))
    cell_count = _safe_int(integral_totals.get("cell_count"))
    blocked_cell_count = _safe_int(integral_totals.get("blocked_cell_count"))
    archive_delta_blocked = _safe_int(
        integral_totals.get("materializer_archive_delta_blocked_cell_count")
    )
    feedback = payload.get("materializer_archive_delta_feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    return _false_authority_payload(
        {
            "path": path_text,
            "loaded": True,
            "schema": schema,
            "cell_count": cell_count,
            "selected_count": selected_count,
            "blocked_cell_count": blocked_cell_count,
            "materializer_archive_delta_blocked_cell_count": archive_delta_blocked,
            "selected_expected_score_gain": water_bucket.get(
                "selected_expected_score_gain"
            ),
            "materializer_archive_delta_blocks_water_bucket": (
                feedback.get("blocks_water_bucket") is True
                or archive_delta_blocked > 0
            ),
            "materializer_archive_delta_realized_saved_bytes_sum": feedback.get(
                "realized_saved_bytes_sum"
            ),
            "dry_no_selected_cells": cell_count > 0 and selected_count == 0,
            "blockers": list(dict.fromkeys(blockers)),
            "allowed_use": "local_feedback_policy_routing_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_dispatch_authority"
            ),
        }
    )


def _candidate_widening_command_template(
    run_summary: Mapping[str, Any],
    *,
    feedback_action_functional_summary: Mapping[str, Any],
) -> dict[str, Any]:
    request = run_summary.get("queue_feedback_replan_request")
    request = request if isinstance(request, Mapping) else {}
    raw_command = request.get("command_template")
    blockers: list[str] = []
    if not isinstance(raw_command, list) or not raw_command:
        return _false_authority_payload(
            {
                "command_template": None,
                "blockers": ["candidate_widening_source_command_missing"],
                "widened_output_path": None,
                "widened_md_path": None,
                "inverse_scorer_max_units": None,
                "allowed_use": "local_candidate_generation_queue_only",
                "forbidden_use": (
                    "score_claim_or_promotion_or_rank_kill_or_dispatch_authority"
                ),
            }
        )
    command_items = [str(item) for item in raw_command]
    if len(command_items) < 2:
        blockers.append("candidate_widening_source_command_too_short")
    elif command_items[1] != QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL:
        blockers.append("candidate_widening_source_command_not_action_functional_tool")
    if command_items:
        command0_name = posixpath.basename(command_items[0])
        if command0_name in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_WRAPPERS:
            blockers.append(
                f"candidate_widening_command_shell_or_remote_wrapper:{command0_name}"
            )
        if not _is_local_python_command(command_items[0]):
            blockers.append("candidate_widening_command_not_local_python")
    blockers.extend(
        f"candidate_widening_command_forbidden_flag:{flag}"
        for flag in _forbidden_command_flag_uses(command_items)
    )
    widenable_source_flags = (
        "--scorer-response",
        "--inverse-scorer-surface",
        "--byte-shaving-signal-surface",
        "--mlx-acquisition-batch",
        "--mlx-effective-spend-triage-selection",
    )
    has_widenable_source = any(
        _has_command_flag(command_items, flag) for flag in widenable_source_flags
    )
    discovered_scorer_responses: list[str] = []
    discovered_scorer_response_records: list[dict[str, Any]] = []
    discovered_scorer_response_blockers: list[str] = []
    if not has_widenable_source:
        discovery = _discover_nearby_scorer_response_sources(run_summary)
        discovered_scorer_response_records = [
            dict(record) for record in _as_list(discovery.get("records"))
        ]
        discovered_scorer_response_blockers = _string_list(discovery.get("blockers"))
        usable_scorer_responses = _string_list(discovery.get("usable_paths"))
        if discovered_scorer_response_blockers:
            blockers.extend(discovered_scorer_response_blockers)
        if len(usable_scorer_responses) == 1 and not discovered_scorer_response_blockers:
            discovered_scorer_responses = usable_scorer_responses
            for path in discovered_scorer_responses:
                command_items.extend(["--scorer-response", path])
            has_widenable_source = True
        elif not discovered_scorer_response_records:
            has_widenable_source = False
        else:
            has_widenable_source = False
    if not has_widenable_source:
        blockers.append("candidate_widening_no_widenable_source_surface")

    source_output = _nonempty_str(
        feedback_action_functional_summary.get("path")
    ) or _command_arg(command_items, "--output")
    widened_output = _widened_json_path(source_output, suffix="widened")
    if widened_output is None:
        blockers.append("candidate_widening_output_path_missing")
    widened_md = None
    if widened_output is not None:
        md_path = Path(widened_output)
        widened_md = md_path.with_suffix(".md").as_posix()

    current_units = _safe_int(_command_arg(command_items, "--inverse-scorer-max-units"))
    if current_units < 1:
        current_units = 32
    cell_count = _safe_int(feedback_action_functional_summary.get("cell_count"))
    blocked_cell_count = _safe_int(
        feedback_action_functional_summary.get("blocked_cell_count")
    )
    archive_blocked_count = _safe_int(
        feedback_action_functional_summary.get(
            "materializer_archive_delta_blocked_cell_count"
        )
    )
    widened_units = min(
        4096,
        max(
            current_units * 2,
            current_units + max(blocked_cell_count, archive_blocked_count, 1),
            cell_count * 2,
            64,
        ),
    )
    widened_max_cells = min(16384, max(4096, widened_units * 8))
    if widened_output is not None:
        command_items = _replace_or_append_flag(command_items, "--output", widened_output)
    if widened_md is not None:
        command_items = _replace_or_append_flag(command_items, "--md-out", widened_md)
    command_items = _replace_or_append_flag(
        command_items,
        "--inverse-scorer-max-units",
        str(widened_units),
    )
    command_items = _replace_or_append_flag(
        command_items,
        "--max-cells",
        str(widened_max_cells),
    )
    command_items = _without_flag(command_items, "--expected-output-sha256")
    command_items = _without_flag(command_items, "--expected-md-sha256")

    return _false_authority_payload(
        {
            "command_template": None if blockers else command_items,
            "blockers": list(dict.fromkeys(blockers)),
            "source_command_template": [str(item) for item in raw_command],
            "widened_output_path": widened_output,
            "widened_md_path": widened_md,
            "source_mode": (
                "existing_widenable_source"
                if not discovered_scorer_responses
                else "discovered_nearby_scorer_response"
            )
            if has_widenable_source
            else "missing_widenable_source",
            "discovered_scorer_response_paths": discovered_scorer_responses,
            "discovered_scorer_response_records": discovered_scorer_response_records,
            "discovered_scorer_response_blockers": discovered_scorer_response_blockers,
            "previous_inverse_scorer_max_units": current_units,
            "inverse_scorer_max_units": widened_units,
            "max_cells": widened_max_cells,
            "widening_rules": [
                "double_inverse_scorer_max_units",
                "raise_max_cells_for_local_candidate_generation",
                "write_distinct_widened_action_functional",
                "strip_expected_output_hashes",
            ],
            "allowed_use": "local_candidate_generation_queue_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_dispatch_authority"
            ),
        }
    )


def _validate_next_iteration_command(
    policy: Mapping[str, Any],
    command_items: Sequence[str],
) -> list[str]:
    blockers: list[str] = []
    if len(command_items) < 2:
        return ["queue_feedback_replan_continuation_command_too_short"]
    command0_name = posixpath.basename(command_items[0])
    if command0_name in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_WRAPPERS:
        blockers.append(
            f"queue_feedback_replan_continuation_command_shell_or_remote_wrapper:{command0_name}"
        )
    if not _is_local_python_command(command_items[0]):
        blockers.append("queue_feedback_replan_continuation_command_not_local_python")
    if command_items[1] != QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL:
        blockers.append("queue_feedback_replan_continuation_command_not_campaign_tool")
    blockers.extend(
        f"queue_feedback_replan_continuation_command_forbidden_flag:{flag}"
        for flag in _forbidden_command_flag_uses(command_items)
    )

    expected_plan = _nonempty_str(policy.get("plan_path"))
    expected_action = _nonempty_str(policy.get("feedback_action_functional_path"))
    expected_iteration = str(_safe_int(policy.get("next_iteration_index")))
    expected_max = str(_safe_int(policy.get("max_iterations")))
    if _command_arg(command_items, "--plan") != expected_plan:
        blockers.append("queue_feedback_replan_continuation_command_plan_mismatch")
    if _command_arg(command_items, "--inverse-scorer-action-functional") != expected_action:
        blockers.append(
            "queue_feedback_replan_continuation_command_action_functional_mismatch"
        )
    if _command_arg(command_items, "--queue-feedback-replan-policy-iteration") != expected_iteration:
        blockers.append("queue_feedback_replan_continuation_command_iteration_mismatch")
    if _command_arg(command_items, "--queue-feedback-replan-policy-max-iterations") != expected_max:
        blockers.append(
            "queue_feedback_replan_continuation_command_max_iterations_mismatch"
        )
    return list(dict.fromkeys(blockers))


def _recovery_queue_command_blockers(
    *,
    command_items: Sequence[str],
    action_index: int,
    recovery_plan: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if len(command_items) < 2:
        return [f"queue_observation_recovery_command_too_short:{action_index}"]
    command0_name = posixpath.basename(command_items[0])
    if command0_name in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_WRAPPERS:
        blockers.append(
            "queue_observation_recovery_command_shell_or_remote_wrapper:"
            f"{action_index}:{command0_name}"
        )
    if not _is_local_python_command(command_items[0]):
        blockers.append(
            f"queue_observation_recovery_command_not_local_python:{action_index}"
        )
    if command_items[1] != QUEUE_OBSERVATION_RECOVERY_TOOL:
        blockers.append(
            f"queue_observation_recovery_command_not_experiment_queue_tool:{action_index}"
        )
    blockers.extend(
        f"queue_observation_recovery_command_forbidden_flag:{action_index}:{flag}"
        for flag in _forbidden_command_flag_uses(command_items)
    )
    expected_queue = _nonempty_str(recovery_plan.get("queue_path"))
    expected_state = _nonempty_str(recovery_plan.get("queue_state_path"))
    if _command_arg(command_items, "--queue") != expected_queue:
        blockers.append(f"queue_observation_recovery_command_queue_mismatch:{action_index}")
    if _command_arg(command_items, "--state") != expected_state:
        blockers.append(f"queue_observation_recovery_command_state_mismatch:{action_index}")
    allowed_subcommands = {"init", "observe", "retire-orphans", "rewind"}
    if not any(item in allowed_subcommands for item in command_items[2:]):
        blockers.append(
            f"queue_observation_recovery_command_subcommand_not_allowed:{action_index}"
        )
    return list(dict.fromkeys(blockers))


def validate_queue_observation_recovery_queue(
    recovery_queue: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a paused local recovery queue before policy execution."""

    blockers: list[str] = []
    controls = recovery_queue.get("controls")
    experiments = recovery_queue.get("experiments")
    step_count = 0
    command_count = 0
    source_queue_paths: list[str] = []
    source_state_paths: list[str] = []
    expected_source_queue_sha256s: list[str] = []
    current_source_queue_sha256s: list[str] = []
    expected_source_state_watermarks: list[dict[str, Any]] = []
    current_source_state_watermarks: list[dict[str, Any]] = []

    if recovery_queue.get("schema") != QUEUE_SCHEMA:
        blockers.append("queue_observation_recovery_schema_not_experiment_queue")

    if not isinstance(controls, Mapping):
        blockers.append("queue_observation_recovery_controls_missing")
        control_mode = None
        local_first = None
        max_concurrency = None
    else:
        control_mode = controls.get("mode")
        local_first = controls.get("local_first")
        max_concurrency = controls.get("max_concurrency")
        if control_mode != "paused":
            blockers.append("queue_observation_recovery_control_mode_not_paused")
        if local_first is not True:
            blockers.append("queue_observation_recovery_not_local_first")
        if isinstance(max_concurrency, Mapping):
            non_local_limits = sorted(
                str(key) for key in max_concurrency if str(key) != "local_cpu"
            )
            blockers.extend(
                f"queue_observation_recovery_non_local_concurrency:{key}"
                for key in non_local_limits
            )
        elif max_concurrency not in ({}, None):
            blockers.append("queue_observation_recovery_max_concurrency_not_mapping")

    for violation in truthy_authority_field_violations(
        recovery_queue,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append(
            f"queue_observation_recovery_truthy_authority_field:{violation}"
        )

    if not isinstance(experiments, list) or not experiments:
        blockers.append("queue_observation_recovery_experiments_missing")
    else:
        for experiment_index, experiment in enumerate(experiments):
            if not isinstance(experiment, Mapping):
                blockers.append(
                    f"queue_observation_recovery_experiment_not_object:{experiment_index}"
                )
                continue
            if experiment.get("id") != QUEUE_OBSERVATION_RECOVERY_EXPERIMENT_ID:
                blockers.append(
                    f"queue_observation_recovery_experiment_id_unexpected:{experiment_index}"
                )
            metadata = (
                experiment.get("metadata")
                if isinstance(experiment.get("metadata"), Mapping)
                else {}
            )
            source_queue_path = _nonempty_str(metadata.get("source_queue_path"))
            source_state_path = _nonempty_str(metadata.get("source_queue_state_path"))
            source_queue_id = _nonempty_str(metadata.get("source_queue_id"))
            expected_source_queue_sha256 = _nonempty_str(
                metadata.get("expected_source_queue_sha256")
            )
            expected_source_state_watermark = metadata.get(
                "expected_source_state_watermark"
            )
            if source_queue_path is None:
                blockers.append(
                    f"queue_observation_recovery_source_queue_path_missing:{experiment_index}"
                )
            else:
                source_queue_paths.append(source_queue_path)
                current_source_queue_sha256 = _stable_json_file_sha256(
                    source_queue_path
                )
                if current_source_queue_sha256:
                    current_source_queue_sha256s.append(current_source_queue_sha256)
                if expected_source_queue_sha256 is None:
                    blockers.append(
                        "queue_observation_recovery_expected_source_queue_sha256_missing:"
                        f"{experiment_index}"
                    )
                else:
                    expected_source_queue_sha256s.append(expected_source_queue_sha256)
                    if current_source_queue_sha256 is None:
                        blockers.append(
                            "queue_observation_recovery_current_source_queue_sha256_missing:"
                            f"{experiment_index}"
                        )
                    elif current_source_queue_sha256 != expected_source_queue_sha256:
                        blockers.append(
                            "queue_observation_recovery_source_queue_sha256_drift:"
                            f"{experiment_index}"
                        )
            if source_state_path is None:
                blockers.append(
                    f"queue_observation_recovery_source_state_path_missing:{experiment_index}"
                )
            else:
                source_state_paths.append(source_state_path)
                current_source_state_watermark = _source_state_watermark_from_path(
                    source_state_path,
                    queue_id=source_queue_id,
                )
                if current_source_state_watermark is not None:
                    current_source_state_watermarks.append(current_source_state_watermark)
                if not isinstance(expected_source_state_watermark, Mapping):
                    blockers.append(
                        "queue_observation_recovery_expected_source_state_watermark_missing:"
                        f"{experiment_index}"
                    )
                else:
                    expected_watermark = dict(expected_source_state_watermark)
                    expected_source_state_watermarks.append(expected_watermark)
                    if (
                        current_source_state_watermark is None
                        and expected_watermark.get("state_missing") is True
                    ):
                        pass
                    elif current_source_state_watermark is None:
                        blockers.append(
                            "queue_observation_recovery_current_source_state_watermark_missing:"
                            f"{experiment_index}"
                        )
                    elif current_source_state_watermark != expected_watermark:
                        blockers.append(
                            "queue_observation_recovery_source_state_watermark_drift:"
                            f"{experiment_index}"
                        )

            steps = experiment.get("steps")
            if not isinstance(steps, list) or not steps:
                blockers.append(
                    f"queue_observation_recovery_steps_missing:{experiment_index}"
                )
                continue
            for step_index, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    blockers.append(
                        "queue_observation_recovery_step_not_object:"
                        f"{experiment_index}:{step_index}"
                    )
                    continue
                step_count += 1
                resources = step.get("resources")
                if not isinstance(resources, Mapping) or resources.get("kind") != "local_cpu":
                    blockers.append(
                        "queue_observation_recovery_step_not_local_cpu:"
                        f"{experiment_index}:{step_index}"
                    )
                command = step.get("command")
                if not isinstance(command, list) or not command:
                    blockers.append(
                        "queue_observation_recovery_step_command_missing:"
                        f"{experiment_index}:{step_index}"
                    )
                    continue
                command_count += 1
                command_items = [str(item) for item in command]
                command_blockers = _recovery_queue_command_blockers(
                    command_items=command_items,
                    action_index=step_index,
                    recovery_plan={
                        "queue_path": source_queue_path,
                        "queue_state_path": source_state_path,
                    },
                )
                blockers.extend(
                    "queue_observation_recovery_step_command_validation:"
                    f"{experiment_index}:{item}"
                    for item in command_blockers
                )

    return _false_authority_payload(
        {
            "schema": QUEUE_OBSERVATION_RECOVERY_QUEUE_VALIDATION_SCHEMA,
            "queue_schema": _nonempty_str(recovery_queue.get("schema")),
            "queue_id": _nonempty_str(recovery_queue.get("queue_id")),
            "queue_sha256": _stable_json_sha256(recovery_queue),
            "control_mode": control_mode,
            "local_first": local_first,
            "max_concurrency": dict(max_concurrency)
            if isinstance(max_concurrency, Mapping)
            else max_concurrency,
            "experiment_count": len(experiments) if isinstance(experiments, list) else 0,
            "step_count": step_count,
            "command_count": command_count,
            "source_queue_paths": list(dict.fromkeys(source_queue_paths)),
            "source_state_paths": list(dict.fromkeys(source_state_paths)),
            "expected_source_queue_sha256s": list(
                dict.fromkeys(expected_source_queue_sha256s)
            ),
            "current_source_queue_sha256s": list(
                dict.fromkeys(current_source_queue_sha256s)
            ),
            "expected_source_state_watermarks": expected_source_state_watermarks,
            "current_source_state_watermarks": current_source_state_watermarks,
            "valid": not blockers,
            "blockers": list(dict.fromkeys(blockers)),
            "allowed_use": "local_queue_observation_recovery_validation_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        }
    )


def build_queue_observation_recovery_queue(
    policy: Mapping[str, Any],
    *,
    lane_id: str,
    queue_id: str | None = None,
    source_policy_path: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a paused local queue for required queue-observation recovery actions."""

    blockers: list[str] = []
    lane = _nonempty_str(lane_id)
    if lane is None:
        blockers.append("queue_observation_recovery_lane_id_missing")
    if policy.get("schema") != QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA:
        blockers.append("queue_observation_recovery_policy_schema_invalid")
    if policy.get("decision") != ACTION_RECOVER_QUEUE_HEALTH:
        blockers.append("queue_observation_recovery_policy_not_recovery")
    if policy.get("ready_for_queue_health_recovery") is not True:
        blockers.append("queue_observation_recovery_policy_not_ready")
    policy_blockers = _string_list(policy.get("blockers"))
    blockers.extend(
        f"queue_observation_recovery_policy_blocker:{item}"
        for item in policy_blockers
    )
    for violation in truthy_authority_field_violations(
        policy,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append(
            f"queue_observation_recovery_policy_truthy_authority_field:{violation}"
        )

    recovery_plan = policy.get("queue_observation_recovery_plan")
    if not isinstance(recovery_plan, Mapping):
        blockers.append("queue_observation_recovery_plan_missing")
        recovery_plan = {}
    elif recovery_plan.get("schema") != QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA:
        blockers.append("queue_observation_recovery_plan_schema_invalid")
    elif recovery_plan.get("recovery_required") is not True:
        blockers.append("queue_observation_recovery_plan_not_required")
    for violation in truthy_authority_field_violations(
        recovery_plan,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append(
            f"queue_observation_recovery_plan_truthy_authority_field:{violation}"
        )

    required_actions = [
        dict(item)
        for item in _as_list(recovery_plan.get("actions"))
        if isinstance(item, Mapping) and item.get("required") is True
    ]
    if not required_actions:
        blockers.append("queue_observation_recovery_required_actions_missing")

    command_items_by_index: list[tuple[int, dict[str, Any], list[str]]] = []
    for index, action in enumerate(required_actions):
        command = action.get("command")
        action_name = _nonempty_str(action.get("action")) or f"action_{index}"
        if not isinstance(command, list) or not command:
            blockers.append(
                f"queue_observation_recovery_action_command_missing:{action_name}"
            )
            continue
        command_items = [str(item) for item in command]
        blockers.extend(
            _recovery_queue_command_blockers(
                command_items=command_items,
                action_index=index,
                recovery_plan=recovery_plan,
            )
        )
        command_items_by_index.append((index, action, command_items))

    if blockers:
        return None, list(dict.fromkeys(blockers))

    source_queue_id = _nonempty_str(policy.get("queue_id")) or "materializer_campaign"
    effective_queue_id = queue_id or f"{source_queue_id}_queue_observation_recovery"
    input_artifacts = [
        source_policy_path,
        policy.get("source_run_path"),
        policy.get("queue_path"),
        policy.get("queue_state_path"),
    ]
    metadata = _false_authority_payload(
        {
            "schema": QUEUE_OBSERVATION_RECOVERY_QUEUE_METADATA_SCHEMA,
            "source_policy_path": source_policy_path,
            "source_policy_sha256": _stable_json_sha256(policy),
            "source_run_path": policy.get("source_run_path"),
            "source_queue_id": policy.get("queue_id"),
            "source_queue_path": policy.get("queue_path"),
            "source_queue_state_path": policy.get("queue_state_path"),
            "expected_source_queue_sha256": recovery_plan.get(
                "source_queue_sha256"
            ),
            "expected_source_state_watermark": recovery_plan.get(
                "source_state_watermark"
            ),
            "source_observation_generated_at_utc": recovery_plan.get(
                "source_observation_generated_at_utc"
            ),
            "policy_decision": policy.get("decision"),
            "action": ACTION_RECOVER_QUEUE_HEALTH,
            "required_action_count": len(command_items_by_index),
            "operator_queue_state_mutation_required": True,
            "auto_execute_eligible": False,
            "allowed_use": "paused_local_queue_health_recovery_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            "dispatch_blockers": [
                "paused_recovery_queue_requires_explicit_operator_or_autopilot_resume",
                "local_queue_state_mutation_only",
                "exact_auth_eval_required_before_score_claim",
                "lane_dispatch_claim_required_before_paid_or_remote_eval",
            ],
        }
    )
    steps: list[dict[str, Any]] = []
    for index, action, command_items in command_items_by_index:
        action_name = _nonempty_str(action.get("action")) or f"action_{index}"
        step_suffix = "_".join(
            item
            for item in (
                _id_fragment(action_name),
                _id_fragment(action.get("experiment_id")),
                _id_fragment(action.get("step_id")),
            )
            if item != "unknown"
        )
        steps.append(
            {
                "id": f"recover_{index:03d}_{step_suffix}",
                "kind": "command",
                "command": command_items,
                "requires": [],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 0,
                "metadata": _false_authority_payload(
                    {
                        "action": action_name,
                        "source_experiment_id": action.get("experiment_id"),
                        "source_step_id": action.get("step_id"),
                        "reason": action.get("reason"),
                        "operator_queue_state_mutation_required": True,
                        "auto_execute_eligible": False,
                    }
                ),
                "telemetry": {
                    "artifact_paths": [],
                    "input_artifact_paths": [
                        str(item) for item in input_artifacts if item
                    ],
                    "recursive": False,
                },
            }
        )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": effective_queue_id,
            "controls": {
                "mode": "paused",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": QUEUE_OBSERVATION_RECOVERY_EXPERIMENT_ID,
                    "lane_id": lane,
                    "priority": 95,
                    "status": "queued",
                    "tags": [
                        "byte-shaving",
                        "inverse-steganalysis",
                        "queue-observation",
                        "queue-recovery",
                        "paused-followup",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": steps,
                }
            ],
        }
    )
    return queue, []


def build_queue_feedback_replan_continuation_queue(
    policy: Mapping[str, Any],
    *,
    lane_id: str,
    queue_id: str | None = None,
    source_policy_path: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a paused local queue for the next materializer iteration."""

    blockers: list[str] = []
    lane = _nonempty_str(lane_id)
    if lane is None:
        blockers.append("queue_feedback_replan_continuation_lane_id_missing")
    if policy.get("schema") != QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA:
        blockers.append("queue_feedback_replan_continuation_policy_schema_invalid")
    if policy.get("decision") != ACTION_RUN_NEXT_ITERATION:
        blockers.append("queue_feedback_replan_continuation_policy_not_next_iteration")
    if policy.get("should_continue_feedback_loop") is not True:
        blockers.append("queue_feedback_replan_continuation_policy_not_continuable")
    policy_blockers = _string_list(policy.get("blockers"))
    blockers.extend(
        f"queue_feedback_replan_continuation_policy_blocker:{item}"
        for item in policy_blockers
    )
    for violation in truthy_authority_field_violations(
        policy,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append(
            f"queue_feedback_replan_continuation_policy_truthy_authority_field:{violation}"
        )

    command = policy.get("next_iteration_command_template")
    if not isinstance(command, list) or not command:
        blockers.append("queue_feedback_replan_continuation_command_missing")
        command_items: list[str] = []
    else:
        command_items = [str(item) for item in command]
        blockers.extend(_validate_next_iteration_command(policy, command_items))

    if blockers:
        return None, list(dict.fromkeys(blockers))

    source_queue_id = _nonempty_str(policy.get("queue_id")) or "materializer_campaign"
    next_iteration = _safe_int(policy.get("next_iteration_index"))
    effective_queue_id = queue_id or f"{source_queue_id}_feedback_continue_{next_iteration}"
    input_artifacts = [
        source_policy_path,
        policy.get("source_run_path"),
        policy.get("plan_path"),
        policy.get("feedback_action_functional_path"),
        policy.get("queue_performance_summary_path"),
        policy.get("feedback_followup_queue_path"),
    ]
    metadata = _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_CONTINUATION_METADATA_SCHEMA,
            "source_policy_path": source_policy_path,
            "source_policy_sha256": _stable_json_sha256(policy),
            "source_run_path": policy.get("source_run_path"),
            "source_queue_id": policy.get("queue_id"),
            "source_queue_path": policy.get("queue_path"),
            "source_queue_state_path": policy.get("queue_state_path"),
            "policy_decision": policy.get("decision"),
            "iteration_index": policy.get("iteration_index"),
            "next_iteration_index": next_iteration,
            "max_iterations": policy.get("max_iterations"),
            "action": ACTION_RUN_NEXT_ITERATION,
            "allowed_use": "paused_local_materializer_feedback_continuation_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            "dispatch_blockers": [
                "paused_continuation_queue_requires_explicit_operator_or_autopilot_resume",
                "exact_auth_eval_required_before_score_claim",
                "lane_dispatch_claim_required_before_paid_or_remote_eval",
            ],
        }
    )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": effective_queue_id,
            "controls": {
                "mode": "paused",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": QUEUE_FEEDBACK_REPLAN_CONTINUATION_EXPERIMENT_ID,
                    "lane_id": lane,
                    "priority": 80,
                    "status": "queued",
                    "tags": [
                        "byte-shaving",
                        "inverse-steganalysis",
                        "queue-feedback",
                        "continuation",
                        "paused-followup",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": [
                        {
                            "id": QUEUE_FEEDBACK_REPLAN_CONTINUATION_STEP_ID,
                            "kind": "command",
                            "command": command_items,
                            "requires": [],
                            "resources": {"kind": "local_cpu"},
                            "timeout_seconds": 0,
                            "telemetry": {
                                "artifact_paths": [],
                                "input_artifact_paths": [
                                    str(item) for item in input_artifacts if item
                                ],
                                "recursive": False,
                            },
                        }
                    ],
                }
            ],
        }
    )
    return queue, []


def build_queue_feedback_candidate_widening_queue(
    policy: Mapping[str, Any],
    *,
    lane_id: str,
    source_policy_path: str | None = None,
    queue_id: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    if policy.get("schema") != QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA:
        blockers.append("queue_feedback_candidate_widening_policy_schema_invalid")
    if policy.get("decision") != ACTION_WIDEN_CANDIDATE_GENERATION:
        blockers.append("queue_feedback_candidate_widening_policy_not_widening")
    if policy.get("ready_for_candidate_generation_widening") is not True:
        blockers.append("queue_feedback_candidate_widening_policy_not_ready")
    if policy.get("should_continue_feedback_loop") is True:
        blockers.append("queue_feedback_candidate_widening_policy_continuable")
    if truthy_authority_field_violations(
        policy,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append("queue_feedback_candidate_widening_policy_authority_violation")

    handoff = policy.get("candidate_widening_handoff")
    handoff = handoff if isinstance(handoff, Mapping) else {}
    blockers.extend(
        f"queue_feedback_candidate_widening_handoff:{item}"
        for item in _string_list(handoff.get("blockers"))
    )
    command = handoff.get("command_template") or policy.get(
        "candidate_generation_command_template"
    )
    if not isinstance(command, list) or not command:
        blockers.append("queue_feedback_candidate_widening_command_missing")
        command_items: list[str] = []
    else:
        command_items = [str(item) for item in command]
        if len(command_items) < 2:
            blockers.append("queue_feedback_candidate_widening_command_too_short")
        else:
            command0_name = posixpath.basename(command_items[0])
            if command0_name in QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_WRAPPERS:
                blockers.append(
                    "queue_feedback_candidate_widening_command_shell_or_remote_wrapper:"
                    f"{command0_name}"
                )
            if not _is_local_python_command(command_items[0]):
                blockers.append("queue_feedback_candidate_widening_command_not_local_python")
            if command_items[1] != QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL:
                blockers.append(
                    "queue_feedback_candidate_widening_command_not_action_functional_tool"
                )
        blockers.extend(
            f"queue_feedback_candidate_widening_command_forbidden_flag:{flag}"
            for flag in _forbidden_command_flag_uses(command_items)
        )

    if blockers:
        return None, list(dict.fromkeys(blockers))

    lane = lane_id.strip()
    if not lane:
        return None, ["queue_feedback_candidate_widening_lane_id_missing"]
    source_queue_id = _nonempty_str(policy.get("queue_id")) or "materializer_campaign"
    effective_queue_id = queue_id or f"{source_queue_id}_feedback_candidate_widening"
    input_artifacts = [
        source_policy_path,
        policy.get("source_run_path"),
        policy.get("plan_path"),
        policy.get("feedback_action_functional_path"),
        policy.get("queue_performance_summary_path"),
        policy.get("feedback_followup_queue_path"),
    ]
    output_artifacts = [
        handoff.get("widened_output_path"),
        handoff.get("widened_md_path"),
    ]
    postconditions: list[dict[str, Any]] = []
    widened_output_path = _nonempty_str(handoff.get("widened_output_path"))
    widened_md_path = _nonempty_str(handoff.get("widened_md_path"))
    if widened_output_path is not None:
        postconditions.extend(
            [
                {
                    "type": "path_exists",
                    "path": widened_output_path,
                },
                {
                    "type": "json_completion_contract",
                    "path": widened_output_path,
                    "required_equals": {
                        "schema": "inverse_steganalysis_discrete_action_functional.v1"
                    },
                    "required_true": [
                        "planning_only",
                        "candidate_generation_only",
                    ],
                    "required_false": [
                        *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
                    ],
                    "false_or_missing": [
                        *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
                    ],
                },
            ]
        )
    if widened_md_path is not None:
        postconditions.append(
            {
                "type": "path_exists",
                "path": widened_md_path,
            }
        )
    metadata = _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_CANDIDATE_WIDENING_METADATA_SCHEMA,
            "source_policy_path": source_policy_path,
            "source_policy_sha256": _stable_json_sha256(policy),
            "source_run_path": policy.get("source_run_path"),
            "source_queue_id": policy.get("queue_id"),
            "source_queue_path": policy.get("queue_path"),
            "source_queue_state_path": policy.get("queue_state_path"),
            "policy_decision": policy.get("decision"),
            "action": ACTION_WIDEN_CANDIDATE_GENERATION,
            "feedback_action_functional_path": policy.get(
                "feedback_action_functional_path"
            ),
            "widened_output_path": handoff.get("widened_output_path"),
            "inverse_scorer_max_units": handoff.get("inverse_scorer_max_units"),
            "allowed_use": "paused_local_inverse_candidate_generation_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            "dispatch_blockers": [
                "paused_candidate_generation_queue_requires_explicit_operator_or_autopilot_resume",
                "generated_action_functional_is_planning_only",
                "exact_auth_eval_required_before_score_claim",
                "lane_dispatch_claim_required_before_paid_or_remote_eval",
            ],
        }
    )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": effective_queue_id,
            "controls": {
                "mode": "paused",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": QUEUE_FEEDBACK_CANDIDATE_WIDENING_EXPERIMENT_ID,
                    "lane_id": lane,
                    "priority": 85,
                    "status": "queued",
                    "tags": [
                        "byte-shaving",
                        "inverse-steganalysis",
                        "queue-feedback",
                        "candidate-widening",
                        "paused-followup",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": [
                        {
                            "id": QUEUE_FEEDBACK_CANDIDATE_WIDENING_STEP_ID,
                            "kind": "command",
                            "command": command_items,
                            "requires": [],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": postconditions,
                            "timeout_seconds": 0,
                            "telemetry": {
                                "artifact_paths": [
                                    str(item) for item in output_artifacts if item
                                ],
                                "input_artifact_paths": [
                                    str(item) for item in input_artifacts if item
                                ],
                                "pullback_artifact_paths": [
                                    str(item) for item in output_artifacts if item
                                ],
                                "recursive": False,
                            },
                        }
                    ],
                }
            ],
        }
    )
    return queue, []


def build_queue_feedback_candidate_actuation_planning_queue(
    policy: Mapping[str, Any],
    *,
    lane_id: str,
    source_policy_path: str | None = None,
    queue_id: str | None = None,
    campaign_id: str | None = None,
    max_k: int = 8,
    candidate_limit: int = 16,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a paused queue that turns widened cells into materializer work state.

    This does not execute materializers. It compiles the widened inverse-action
    surface into a byte-shaving campaign plan, inverse-action materialization
    bridge, materializer backlog, and materializer work queue. If cells still
    need a receiver/compiler transform, the work queue records that blocker as
    durable state rather than presenting a fake executable row.
    """

    blockers: list[str] = []
    if max_k < 1:
        blockers.append("queue_feedback_candidate_actuation_max_k_invalid")
    if candidate_limit < 1:
        blockers.append("queue_feedback_candidate_actuation_candidate_limit_invalid")
    if policy.get("schema") != QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA:
        blockers.append("queue_feedback_candidate_actuation_policy_schema_invalid")
    if policy.get("decision") != ACTION_WIDEN_CANDIDATE_GENERATION:
        blockers.append("queue_feedback_candidate_actuation_policy_not_widening")
    if truthy_authority_field_violations(
        policy,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    ):
        blockers.append("queue_feedback_candidate_actuation_policy_authority_violation")
    handoff = policy.get("candidate_widening_handoff")
    handoff = handoff if isinstance(handoff, Mapping) else {}
    blockers.extend(
        f"queue_feedback_candidate_actuation_handoff:{item}"
        for item in _string_list(handoff.get("blockers"))
    )
    widened_action_path = _nonempty_str(handoff.get("widened_output_path"))
    if widened_action_path is None:
        blockers.append("queue_feedback_candidate_actuation_widened_output_missing")

    lane = lane_id.strip()
    if not lane:
        blockers.append("queue_feedback_candidate_actuation_lane_id_missing")
    if blockers:
        return None, list(dict.fromkeys(blockers))

    assert widened_action_path is not None
    paths = _candidate_actuation_paths(widened_action_path)
    effective_campaign_id = (
        campaign_id
        or f"{_nonempty_str(policy.get('queue_id')) or 'feedback'}_candidate_widening_actuation"
    )
    plan_command = [
        ".venv/bin/python",
        QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_TOOL,
        "--source",
        widened_action_path,
        "--from-inverse-action-functional",
        "--output",
        paths["campaign_plan"],
        "--md-out",
        paths["campaign_plan_md"],
        "--inverse-action-materialization-bridge-out",
        paths["inverse_action_materialization_bridge"],
        "--campaign-id",
        effective_campaign_id,
        "--max-k",
        str(max_k),
    ]
    compile_command = [
        ".venv/bin/python",
        QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_TOOL,
        "--plan",
        paths["campaign_plan"],
        "--materialization-out",
        paths["materialization"],
        "--portfolio-out",
        paths["portfolio"],
        "--action-summary-out",
        paths["action_summary"],
        "--materializer-backlog-out",
        paths["materializer_backlog"],
        "--materializer-work-queue-out",
        paths["materializer_work_queue"],
        "--repo-root",
        ".",
        "--candidate-limit",
        str(candidate_limit),
        "--queue-candidate-limit",
        str(candidate_limit),
        "--local-cpu-concurrency",
        "auto",
    ]
    source_queue_id = _nonempty_str(policy.get("queue_id")) or "materializer_campaign"
    effective_queue_id = queue_id or f"{source_queue_id}_feedback_candidate_actuation"
    input_artifacts = [
        source_policy_path,
        policy.get("source_run_path"),
        policy.get("plan_path"),
        policy.get("feedback_action_functional_path"),
        widened_action_path,
        handoff.get("widened_md_path"),
    ]
    output_artifacts = [
        paths["campaign_plan"],
        paths["campaign_plan_md"],
        paths["inverse_action_materialization_bridge"],
        paths["materialization"],
        paths["portfolio"],
        paths["action_summary"],
        paths["materializer_backlog"],
        paths["materializer_work_queue"],
    ]
    metadata = _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_METADATA_SCHEMA,
            "source_policy_path": source_policy_path,
            "source_policy_sha256": _stable_json_sha256(policy),
            "source_run_path": policy.get("source_run_path"),
            "source_queue_id": policy.get("queue_id"),
            "policy_decision": policy.get("decision"),
            "action": "plan_widened_inverse_action_materializer_work",
            "widened_action_functional_path": widened_action_path,
            "campaign_plan_path": paths["campaign_plan"],
            "inverse_action_materialization_bridge_path": paths[
                "inverse_action_materialization_bridge"
            ],
            "materializer_work_queue_path": paths["materializer_work_queue"],
            "allowed_use": (
                "paused_local_candidate_widening_actuation_planning_only"
            ),
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            "dispatch_blockers": [
                "generated_materializer_work_queue_may_be_blocked",
                "compiler_required_cells_need_receiver_operation_transform",
                "materializer_execution_queue_required_before_candidate_archive",
                "exact_auth_eval_required_before_score_claim",
            ],
        }
    )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": effective_queue_id,
            "controls": {
                "mode": "paused",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_EXPERIMENT_ID,
                    "lane_id": lane,
                    "priority": 84,
                    "status": "queued",
                    "tags": [
                        "byte-shaving",
                        "inverse-steganalysis",
                        "queue-feedback",
                        "candidate-widening",
                        "actuation-planning",
                        "no-score-authority",
                    ],
                    "metadata": metadata,
                    "steps": [
                        {
                            "id": QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID,
                            "kind": "command",
                            "command": plan_command,
                            "requires": [],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {
                                    "type": "path_exists",
                                    "path": paths["campaign_plan"],
                                },
                                {
                                    "type": "json_completion_contract",
                                    "path": paths["campaign_plan"],
                                    "required_equals": {
                                        "schema": "byte_shaving_campaign_plan.v1"
                                    },
                                    "required_false": [
                                        *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
                                    ],
                                    "false_or_missing": [
                                        *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
                                    ],
                                    "required_nonempty": [
                                        "inverse_action_materialization_portfolios",
                                    ],
                                },
                                {
                                    "type": "path_exists",
                                    "path": paths[
                                        "inverse_action_materialization_bridge"
                                    ],
                                },
                                {
                                    "type": "json_completion_contract",
                                    "path": paths[
                                        "inverse_action_materialization_bridge"
                                    ],
                                    "required_equals": {
                                        "schema": (
                                            "inverse_steganalysis_water_bucket_"
                                            "materialization_bridge.v1"
                                        )
                                    },
                                    "required_false": [
                                        *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
                                    ],
                                    "false_or_missing": [
                                        *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
                                    ],
                                    "required_nonempty": [
                                        "queue_consumption",
                                        "water_bucket_materialization_portfolios",
                                    ],
                                },
                            ],
                            "timeout_seconds": 0,
                            "telemetry": {
                                "artifact_paths": [
                                    paths["campaign_plan"],
                                    paths["campaign_plan_md"],
                                    paths["inverse_action_materialization_bridge"],
                                ],
                                "input_artifact_paths": [
                                    str(item) for item in input_artifacts if item
                                ],
                                "pullback_artifact_paths": [
                                    paths["campaign_plan"],
                                    paths["campaign_plan_md"],
                                    paths["inverse_action_materialization_bridge"],
                                ],
                                "recursive": False,
                            },
                        },
                        {
                            "id": QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_STEP_ID,
                            "kind": "command",
                            "command": compile_command,
                            "requires": [
                                QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID
                            ],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {
                                    "type": "json_completion_contract",
                                    "path": paths["materialization"],
                                    "required_equals": {
                                        "schema": "byte_shaving_campaign_materialization.v1"
                                    },
                                    "required_false": [
                                        *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
                                    ],
                                    "false_or_missing": [
                                        *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
                                    ],
                                },
                                {
                                    "type": "json_completion_contract",
                                    "path": paths["materializer_backlog"],
                                    "required_equals": {
                                        "schema": "byte_shaving_materializer_backlog.v1"
                                    },
                                    "required_false": [
                                        *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
                                    ],
                                    "false_or_missing": [
                                        *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
                                    ],
                                    "required_nonempty": ["rows"],
                                    "required_positive_int": ["backlog_row_count"],
                                },
                                {
                                    "type": "json_completion_contract",
                                    "path": paths["materializer_work_queue"],
                                    "required_equals": {
                                        "schema": (
                                            "byte_shaving_materializer_work_queue.v1"
                                        )
                                    },
                                    "required_false": [
                                        *DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
                                    ],
                                    "false_or_missing": [
                                        *DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
                                    ],
                                    "required_positive_int": ["row_count"],
                                },
                            ],
                            "timeout_seconds": 0,
                            "telemetry": {
                                "artifact_paths": output_artifacts,
                                "input_artifact_paths": [
                                    paths["campaign_plan"],
                                    paths["inverse_action_materialization_bridge"],
                                    *[str(item) for item in input_artifacts if item],
                                ],
                                "pullback_artifact_paths": output_artifacts,
                                "recursive": False,
                            },
                        },
                    ],
                }
            ],
        }
    )
    return queue, []


def build_queue_feedback_replan_policy(
    run_summary: Mapping[str, Any],
    *,
    feedback_followup_queue: Mapping[str, Any] | None = None,
    source_run_path: str | None = None,
    iteration_index: int = 0,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """Return a non-authoritative bounded feedback-replan policy record.

    The policy is intentionally an advisory/local actuator surface. It may tell
    the local scheduler which safe follow-up to run next, but it never grants
    score, promotion, rank/kill, paid dispatch, or exact-eval authority.
    """

    if iteration_index < 0:
        raise ValueError("iteration_index must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    blockers: list[str] = []
    source_refusal_blockers: list[str] = []
    warnings: list[str] = []
    source_schema = _nonempty_str(run_summary.get("schema"))
    if source_schema != MATERIALIZER_CAMPAIGN_RUN_SCHEMA:
        source_refusal_blockers.append("source_run_schema_not_materializer_campaign_run")

    authority_violations = truthy_authority_field_violations(
        run_summary,
        fields=FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    )
    source_refusal_blockers.extend(
        f"source_run_truthy_authority_field:{violation}"
        for violation in authority_violations
    )
    blockers.extend(source_refusal_blockers)

    exact_handoff_count = _safe_int(run_summary.get("exact_readiness_handoff_count"))
    replan_ready = run_summary.get("queue_feedback_replan_ready") is True
    followup_queue_emitted = (
        run_summary.get("queue_feedback_replan_followup_queue_emitted") is True
    )
    followup_policy_enabled = (
        run_summary.get("queue_feedback_replan_followup_policy_enabled") is True
    )
    followup_executed = (
        run_summary.get("queue_feedback_replan_followup_executed") is True
    )
    followup_success = (
        run_summary.get("queue_feedback_replan_followup_execution_success") is True
    )
    followup_action_functional_path = _nonempty_str(
        run_summary.get("queue_feedback_replan_followup_action_functional_path")
    )
    feedback_action_functional_summary = _feedback_action_functional_summary(
        followup_action_functional_path
    )
    exact_auth_calibration_policy = _exact_auth_calibration_pair_policy(run_summary)
    queue_observation_recovery_plan = _queue_observation_recovery_plan_from_run(
        run_summary
    )
    queue_observation_recovery_required = (
        isinstance(queue_observation_recovery_plan, Mapping)
        and queue_observation_recovery_plan.get("recovery_required") is True
    )
    queue_observation_maintenance_recommended = (
        isinstance(queue_observation_recovery_plan, Mapping)
        and queue_observation_recovery_plan.get("maintenance_recommended") is True
    )

    if exact_handoff_count:
        warnings.append("exact_readiness_handoffs_available")
    if exact_auth_calibration_policy["packet_count"] == 0:
        warnings.append("exact_auth_calibration_packet_pair_absent")

    queue_validation: dict[str, Any] | None = None
    feedback_queue_artifact_not_validated = False
    if followup_queue_emitted:
        if feedback_followup_queue is None:
            feedback_queue_artifact_not_validated = True
            blockers.append("feedback_followup_queue_artifact_not_validated")
        else:
            queue_validation = validate_feedback_followup_queue(
                feedback_followup_queue,
                run_dir=_nonempty_str(run_summary.get("run_dir")),
            )
            validation_blockers = [
                f"feedback_followup_queue_validation:{item}"
                for item in _string_list(queue_validation.get("blockers"))
            ]
            blockers.extend(validation_blockers)

    telemetry_truncation_markers = _telemetry_truncation_markers(
        run_summary.get("performance")
    )
    telemetry_blockers = [
        f"queue_performance_telemetry_truncated:{item}"
        for item in telemetry_truncation_markers
    ]
    blockers.extend(telemetry_blockers)
    calibration_blockers = [
        f"exact_auth_calibration_policy:{item}"
        for item in _string_list(exact_auth_calibration_policy.get("blockers"))
    ]
    blockers.extend(calibration_blockers)
    action_functional_blockers = [
        f"feedback_action_functional:{item}"
        for item in _string_list(feedback_action_functional_summary.get("blockers"))
    ]
    blockers.extend(action_functional_blockers)
    for warning in _string_list(feedback_action_functional_summary.get("warnings")):
        warnings.append(f"feedback_action_functional:{warning}")

    if iteration_index >= max_iterations:
        decision = ACTION_STOP_MAX_ITERATIONS
        stop_reason = "max_iterations_reached"
    elif source_refusal_blockers:
        decision = ACTION_REFUSE
        stop_reason = "blocked"
    elif feedback_queue_artifact_not_validated:
        decision = ACTION_BLOCKED
        stop_reason = "feedback_followup_queue_artifact_not_validated"
    elif queue_validation is not None and queue_validation.get("valid") is not True:
        decision = ACTION_BLOCKED
        stop_reason = "feedback_followup_queue_validation_failed"
    elif telemetry_blockers:
        decision = ACTION_BLOCKED
        stop_reason = "queue_performance_telemetry_truncated"
    elif calibration_blockers:
        decision = ACTION_BLOCKED
        stop_reason = "exact_auth_calibration_policy_failed"
    elif action_functional_blockers:
        decision = ACTION_BLOCKED
        stop_reason = "feedback_action_functional_invalid"
    elif queue_observation_recovery_required:
        decision = ACTION_RECOVER_QUEUE_HEALTH
        stop_reason = None
    elif not replan_ready:
        decision = ACTION_BLOCKED
        stop_reason = "queue_feedback_replan_not_ready"
        blockers.extend(
            f"queue_feedback_replan_blocker:{item}"
            for item in _string_list(run_summary.get("queue_feedback_replan_blockers"))
        )
    elif not followup_queue_emitted:
        decision = ACTION_BLOCKED
        stop_reason = "feedback_followup_queue_not_emitted"
        blockers.extend(
            f"feedback_followup_queue_blocker:{item}"
            for item in _string_list(
                run_summary.get("queue_feedback_replan_followup_queue_blockers")
            )
        )
    elif _string_list(run_summary.get("queue_feedback_replan_followup_policy_blockers")):
        decision = ACTION_BLOCKED
        stop_reason = "feedback_followup_autopolicy_blocked"
        blockers.extend(
            f"feedback_followup_policy_blocker:{item}"
            for item in _string_list(
                run_summary.get("queue_feedback_replan_followup_policy_blockers")
            )
        )
    elif not followup_executed:
        decision = ACTION_EXECUTE_FEEDBACK_FOLLOWUP
        stop_reason = None
    elif not followup_success:
        decision = ACTION_BLOCKED
        stop_reason = "feedback_followup_execution_failed"
        execution = run_summary.get("queue_feedback_replan_followup_execution")
        if isinstance(execution, Mapping):
            blockers.extend(
                f"feedback_followup_execution_blocker:{item}"
                for item in _string_list(execution.get("blockers"))
            )
    elif followup_action_functional_path is None:
        decision = ACTION_BLOCKED
        stop_reason = "feedback_action_functional_missing"
        blockers.append("feedback_action_functional_missing")
    elif feedback_action_functional_summary.get("dry_no_selected_cells") is True:
        decision = ACTION_WIDEN_CANDIDATE_GENERATION
        stop_reason = "feedback_action_functional_dry_no_selected_cells"
    else:
        decision = ACTION_RUN_NEXT_ITERATION
        stop_reason = None

    candidate_widening_handoff = (
        _candidate_widening_command_template(
            run_summary,
            feedback_action_functional_summary=feedback_action_functional_summary,
        )
        if decision == ACTION_WIDEN_CANDIDATE_GENERATION
        else _false_authority_payload(
            {
                "command_template": None,
                "blockers": [],
                "widened_output_path": None,
                "widened_md_path": None,
                "inverse_scorer_max_units": None,
                "allowed_use": "local_candidate_generation_queue_only",
                "forbidden_use": (
                    "score_claim_or_promotion_or_rank_kill_or_dispatch_authority"
                ),
            }
        )
    )

    recommended_actions: list[dict[str, Any]] = []
    if exact_handoff_count:
        recommended_actions.append(
            {
                "action": ACTION_INSPECT_EXACT_HANDOFFS,
                "handoff_count": exact_handoff_count,
                "paths": [dict(item) for item in _as_list(run_summary.get("exact_readiness_handoff_paths"))],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    if isinstance(queue_observation_recovery_plan, Mapping):
        recovery_actions = [
            dict(item)
            for item in _as_list(queue_observation_recovery_plan.get("actions"))
            if isinstance(item, Mapping) and item.get("required") is True
        ]
        maintenance_actions = [
            dict(item)
            for item in _as_list(queue_observation_recovery_plan.get("actions"))
            if isinstance(item, Mapping) and item.get("required") is not True
        ]
        if recovery_actions:
            recommended_actions.append(
                {
                    "action": ACTION_RECOVER_QUEUE_HEALTH,
                    "required_action_count": len(recovery_actions),
                    "actions": recovery_actions,
                    "operator_queue_state_mutation_required": True,
                    "auto_execute_eligible": False,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
        if maintenance_actions:
            recommended_actions.append(
                {
                    "action": ACTION_QUEUE_OBSERVATION_MAINTENANCE,
                    "maintenance_action_count": len(maintenance_actions),
                    "actions": maintenance_actions,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    if decision == ACTION_EXECUTE_FEEDBACK_FOLLOWUP:
        recommended_actions.append(
            {
                "action": ACTION_EXECUTE_FEEDBACK_FOLLOWUP,
                "queue_path": _nonempty_str(
                    run_summary.get("queue_feedback_replan_followup_queue_path")
                ),
                "policy_enabled": followup_policy_enabled,
                "local_only": True,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    if decision == ACTION_WIDEN_CANDIDATE_GENERATION:
        recommended_actions.append(
            {
                "action": ACTION_WIDEN_CANDIDATE_GENERATION,
                "reason": stop_reason,
                "feedback_action_functional_path": followup_action_functional_path,
                "cell_count": feedback_action_functional_summary.get("cell_count"),
                "selected_count": feedback_action_functional_summary.get(
                    "selected_count"
                ),
                "blocked_cell_count": feedback_action_functional_summary.get(
                    "blocked_cell_count"
                ),
                "materializer_archive_delta_blocked_cell_count": (
                    feedback_action_functional_summary.get(
                        "materializer_archive_delta_blocked_cell_count"
                    )
                ),
                "next_gate": "widen_inverse_surface_candidate_generation",
                "candidate_generation_hints": [
                    "increase_inverse_scorer_max_units",
                    "refresh_source_inverse_scorer_surface",
                    "switch_to_compiled_receiver_transform_materializers",
                ],
                "candidate_generation_command_template": (
                    candidate_widening_handoff.get("command_template")
                ),
                "candidate_generation_blockers": candidate_widening_handoff.get(
                    "blockers"
                ),
                "local_only": True,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )

    next_iteration_index = iteration_index + 1
    next_command = None
    if decision == ACTION_RUN_NEXT_ITERATION:
        next_command = _campaign_iteration_command(
            run_summary,
            next_iteration_index=next_iteration_index,
            max_iterations=max_iterations,
        )
        if next_command is None:
            decision = ACTION_BLOCKED
            stop_reason = "next_iteration_command_missing_inputs"
            blockers.append("next_iteration_command_missing_inputs")
        else:
            recommended_actions.append(
                {
                    "action": ACTION_RUN_NEXT_ITERATION,
                    "iteration_index": next_iteration_index,
                    "command_template": next_command,
                    "requires_materializer_context_or_artifact_map": True,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )

    should_continue = decision in {
        ACTION_EXECUTE_FEEDBACK_FOLLOWUP,
        ACTION_RUN_NEXT_ITERATION,
    }
    operator_queue_state_mutation_required = decision == ACTION_RECOVER_QUEUE_HEALTH

    return _false_authority_payload(
        {
            "schema": QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA,
            "source_run_path": source_run_path,
            "source_schema": source_schema,
            "queue_id": _nonempty_str(run_summary.get("queue_id")),
            "run_dir": _nonempty_str(run_summary.get("run_dir")),
            "plan_path": _nonempty_str(run_summary.get("plan")),
            "queue_path": _nonempty_str(run_summary.get("queue_path")),
            "queue_state_path": _nonempty_str(run_summary.get("state_path")),
            "queue_performance_summary_path": _nonempty_str(
                run_summary.get("queue_performance_summary_path")
            ),
            "feedback_followup_queue_path": _nonempty_str(
                run_summary.get("queue_feedback_replan_followup_queue_path")
            ),
            "feedback_action_functional_path": followup_action_functional_path,
            "feedback_action_functional_summary": feedback_action_functional_summary,
            "candidate_widening_handoff": candidate_widening_handoff,
            "candidate_generation_command_template": (
                candidate_widening_handoff.get("command_template")
            ),
            "candidate_generation_queue_blockers": candidate_widening_handoff.get(
                "blockers"
            ),
            "feedback_followup_queue_validation": queue_validation,
            "iteration_index": iteration_index,
            "next_iteration_index": next_iteration_index,
            "max_iterations": max_iterations,
            "decision": decision,
            "stop_reason": stop_reason,
            "should_continue_feedback_loop": should_continue,
            "ready_for_feedback_replan": replan_ready,
            "ready_for_feedback_followup_execution": (
                decision == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
            ),
            "ready_for_local_materialization": (
                decision == ACTION_RUN_NEXT_ITERATION
            ),
            "ready_for_candidate_generation_widening": (
                decision == ACTION_WIDEN_CANDIDATE_GENERATION
            ),
            "exact_readiness_handoff_count": exact_handoff_count,
            "exact_auth_calibration_policy": exact_auth_calibration_policy,
            "exact_auth_calibration_usable": exact_auth_calibration_policy[
                "usable_for_feedback_trust_region"
            ],
            "queue_observation_recovery_plan": queue_observation_recovery_plan,
            "queue_observation_recovery_required": queue_observation_recovery_required,
            "queue_observation_maintenance_recommended": (
                queue_observation_maintenance_recommended
            ),
            "ready_for_queue_health_recovery": (
                decision == ACTION_RECOVER_QUEUE_HEALTH
            ),
            "operator_queue_state_mutation_required": (
                operator_queue_state_mutation_required
            ),
            "auto_execute_eligible": (
                should_continue and not operator_queue_state_mutation_required
            ),
            "feedback_followup_queue_emitted": followup_queue_emitted,
            "feedback_followup_policy_enabled": followup_policy_enabled,
            "feedback_followup_executed": followup_executed,
            "feedback_followup_execution_success": followup_success,
            "next_iteration_command_template": next_command,
            "recommended_actions": recommended_actions,
            "blockers": list(dict.fromkeys(blockers)),
            "warnings": list(dict.fromkeys(warnings)),
            "allowed_use": (
                "local_materializer_feedback_replan_policy_only"
            ),
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        }
    )


__all__ = [
    "ACTION_BLOCKED",
    "ACTION_EXECUTE_FEEDBACK_FOLLOWUP",
    "ACTION_INSPECT_EXACT_HANDOFFS",
    "ACTION_QUEUE_OBSERVATION_MAINTENANCE",
    "ACTION_RECOVER_QUEUE_HEALTH",
    "ACTION_REFUSE",
    "ACTION_RUN_NEXT_ITERATION",
    "ACTION_STOP_MAX_ITERATIONS",
    "ACTION_WIDEN_CANDIDATE_GENERATION",
    "MATERIALIZER_CAMPAIGN_RUN_SCHEMA",
    "QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_EXPERIMENT_ID",
    "QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_METADATA_SCHEMA",
    "QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID",
    "QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_TOOL",
    "QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_STEP_ID",
    "QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_TOOL",
    "QUEUE_FEEDBACK_CANDIDATE_WIDENING_EXPERIMENT_ID",
    "QUEUE_FEEDBACK_CANDIDATE_WIDENING_METADATA_SCHEMA",
    "QUEUE_FEEDBACK_CANDIDATE_WIDENING_STEP_ID",
    "QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL",
    "QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA",
    "QUEUE_FEEDBACK_REPLAN_CONTINUATION_EXPERIMENT_ID",
    "QUEUE_FEEDBACK_REPLAN_CONTINUATION_METADATA_SCHEMA",
    "QUEUE_FEEDBACK_REPLAN_CONTINUATION_STEP_ID",
    "QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_FLAGS",
    "QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL",
    "QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA",
    "QUEUE_OBSERVATION_RECOVERY_EXPERIMENT_ID",
    "QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA",
    "QUEUE_OBSERVATION_RECOVERY_QUEUE_METADATA_SCHEMA",
    "QUEUE_OBSERVATION_RECOVERY_QUEUE_VALIDATION_SCHEMA",
    "build_queue_feedback_candidate_actuation_planning_queue",
    "build_queue_feedback_candidate_widening_queue",
    "build_queue_feedback_replan_continuation_queue",
    "build_queue_feedback_replan_policy",
    "build_queue_observation_recovery_plan",
    "build_queue_observation_recovery_queue",
    "validate_feedback_followup_queue",
    "validate_queue_observation_recovery_queue",
]
