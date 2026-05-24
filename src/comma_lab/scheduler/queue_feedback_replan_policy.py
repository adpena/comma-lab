# SPDX-License-Identifier: MIT
"""Queue-owned policy records for materializer feedback replanning."""

from __future__ import annotations

import hashlib
import json
import posixpath
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.proxy_candidate_contract import truthy_authority_field_violations

from .experiment_queue import (
    DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
    DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
    QUEUE_SCHEMA,
)

QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA = "queue_feedback_replan_policy.v1"
QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA = (
    "queue_feedback_replan_child_queue_validation.v1"
)
MATERIALIZER_CAMPAIGN_RUN_SCHEMA = "byte_shaving_materializer_campaign_run.v1"
QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL = (
    "tools/build_inverse_steganalysis_action_functional.py"
)
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
        "tools/run_byte_shaving_materializer_campaign.py",
        "--plan",
        plan,
        "--inverse-scorer-action-functional",
        action_functional,
        "--queue-feedback-replan-policy-iteration",
        str(next_iteration_index),
        "--queue-feedback-replan-policy-max-iterations",
        str(max_iterations),
    ]


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

    if exact_handoff_count:
        warnings.append("exact_readiness_handoffs_available")

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
    else:
        decision = ACTION_RUN_NEXT_ITERATION
        stop_reason = None

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
            "exact_readiness_handoff_count": exact_handoff_count,
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
    "ACTION_REFUSE",
    "ACTION_RUN_NEXT_ITERATION",
    "ACTION_STOP_MAX_ITERATIONS",
    "MATERIALIZER_CAMPAIGN_RUN_SCHEMA",
    "QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL",
    "QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA",
    "QUEUE_FEEDBACK_REPLAN_FORBIDDEN_COMMAND_FLAGS",
    "QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA",
    "build_queue_feedback_replan_policy",
    "validate_feedback_followup_queue",
]
