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
from tac.optimization.repair_campaign_learning_signal import (
    build_repair_campaign_activation_plan_learning_signal_report,
)
from tac.optimization.repair_campaign_posterior import (
    append_repair_campaign_blocked_learning_signal_report,
)
from tac.repo_io import ArtifactWriteResult, sha256_bytes, write_json_artifact, write_text_artifact

from .experiment_queue import default_state_path
from .json_identity import stable_json_sha256

POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA = (
    "frontier_final_rate_attack_post_feedback_child_queue_runs.v1"
)
POST_FEEDBACK_CHILD_QUEUE_RUN_SCHEMA = (
    "frontier_final_rate_attack_post_feedback_child_queue_run.v1"
)
POST_FEEDBACK_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA = (
    "frontier_final_rate_attack_child_queue_activation_plan.v1"
)

POST_FEEDBACK_CHILD_QUEUE_PRIORITY = (
    "operation_materializer_execution_queue",
    "targeted_component_correction_chain_materializer_execution_queue",
    "targeted_component_correction_materialization_queue",
    "operation_chain_compiler_queue",
    "targeted_component_correction_operation_chain_queue",
    "autonomous_chain_optimization_queue",
    "repair_campaign_score_queue",
    "repair_posterior_acquisition_followup_queue",
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


def _observation_status_count(
    observation: Mapping[str, Any] | None,
    status: str,
) -> int:
    if observation is None:
        return 0
    status_counts = observation.get("status_counts")
    if not isinstance(status_counts, Mapping):
        return 0
    raw_count = status_counts.get(status)
    if not isinstance(raw_count, int) or isinstance(raw_count, bool):
        return 0
    return raw_count


def _strings_from(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _step_artifact_refs(step: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = step.get("telemetry")
    if not isinstance(telemetry, Mapping):
        telemetry = {}
    postcondition_paths: list[str] = []
    postcondition_types: list[str] = []
    for condition in step.get("postconditions") or []:
        if not isinstance(condition, Mapping):
            continue
        condition_type = condition.get("type")
        if isinstance(condition_type, str) and condition_type:
            postcondition_types.append(condition_type)
        path = condition.get("path")
        if isinstance(path, str) and path:
            postcondition_paths.append(path)
    return {
        "step_id": str(step.get("id") or ""),
        "requires": _strings_from(step.get("requires")),
        "resource_kind": str((step.get("resources") or {}).get("kind") or "local_cpu")
        if isinstance(step.get("resources"), Mapping)
        else "local_cpu",
        "telemetry_input_artifact_paths": _strings_from(
            telemetry.get("input_artifact_paths")
        ),
        "telemetry_output_artifact_paths": _strings_from(telemetry.get("artifact_paths")),
        "postcondition_types": sorted(set(postcondition_types)),
        "postcondition_paths": sorted(set(postcondition_paths)),
    }


def _activation_action_for_blocker(blocker: str) -> dict[str, str]:
    if "targeted_component_correction" in blocker or "component_eval" in blocker:
        return {
            "blocker": blocker,
            "activation_action": "harvest_targeted_component_response_rows",
            "evidence_surface": "targeted_component_correction_response_harvest",
        }
    if "receiver_closed" in blocker or "saved_bytes" in blocker:
        return {
            "blocker": blocker,
            "activation_action": "materialize_receiver_closed_rate_budget_credit",
            "evidence_surface": "receiver_closed_correction_budget",
        }
    if "exact_auth_eval" in blocker:
        return {
            "blocker": blocker,
            "activation_action": "route_byte_closed_candidate_to_exact_auth_eval_handoff",
            "evidence_surface": "contest_cpu_or_cuda_auth_axis_payload",
        }
    if blocker.startswith("experiment_status:"):
        return {
            "blocker": blocker,
            "activation_action": "thaw_queue_definition_after_prerequisite_evidence_lands",
            "evidence_surface": "queue_definition_status",
        }
    return {
        "blocker": blocker,
        "activation_action": "inspect_queue_owned_prerequisite",
        "evidence_surface": "experiment_metadata_or_step_postconditions",
    }


def _activation_blockers_for_experiment(experiment: Mapping[str, Any]) -> list[str]:
    metadata = experiment.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    blockers: list[str] = []
    status = str(experiment.get("status") or "queued")
    if status != "queued":
        blockers.append(f"experiment_status:{status}")
    for key in (
        "queue_actuation_blockers",
        "activation_blockers",
        "readiness_blockers",
        "blockers",
    ):
        blockers.extend(_strings_from(metadata.get(key)))
    return list(dict.fromkeys(blockers))


def _child_queue_activation_plan(
    *,
    queue_path: Path,
    queue_payload: Mapping[str, Any],
    queue_id: str,
    queue_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    blocked_experiments: list[dict[str, Any]] = []
    for experiment in queue_payload.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        status = str(experiment.get("status") or "queued")
        if status not in {"frozen", "paused", "disabled"}:
            continue
        blockers = _activation_blockers_for_experiment(experiment)
        blocked_experiments.append(
            {
                "experiment_id": str(experiment.get("id") or ""),
                "lane_id": experiment.get("lane_id"),
                "status": status,
                "tags": _strings_from(experiment.get("tags")),
                "activation_blockers": blockers,
                "activation_actions": [
                    _activation_action_for_blocker(blocker) for blocker in blockers
                ],
                "step_count": len(experiment.get("steps") or []),
                "step_evidence_refs": [
                    _step_artifact_refs(step)
                    for step in experiment.get("steps") or []
                    if isinstance(step, Mapping)
                ],
            }
        )
    activation_actions = list(
        {
            row["activation_action"]: row
            for experiment in blocked_experiments
            for row in experiment["activation_actions"]
        }.values()
    )
    return {
        "schema": POST_FEEDBACK_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "queue_id": queue_id,
        "queue_sha256": queue_sha256,
        "queue_path": _repo_rel(queue_path, repo_root),
        "blocked_experiment_count": len(blocked_experiments),
        "blocked_experiments": blocked_experiments,
        "activation_actions": activation_actions,
        "mathematical_contract": {
            "objective": "minimize_delta_score_under_rate_and_runtime_constraints",
            "state_variables": [
                "bytes_delta",
                "segnet_response",
                "posenet_response",
                "interaction_terms",
                "entropy_position",
                "receiver_runtime_identity",
            ],
            "hard_constraints": [
                "score_claim_false_until_contest_auth_axis_eval",
                "promotion_false_until_byte_closed_runtime_receiver_proof",
                "budget_spend_false_until_receiver_closed_saved_bytes_exist",
            ],
        },
        "allowed_use": "local_child_queue_activation_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _queue_definition_status_counts(queue_payload: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for experiment in queue_payload.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        status = str(experiment.get("status") or "queued")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _queue_selection_rank(status_counts: Mapping[str, Any]) -> int:
    """Prefer runnable child queues over empty/frozen advisory queues."""

    try:
        queued = int(status_counts.get("queued") or 0)
    except (TypeError, ValueError):
        queued = 0
    if queued > 0:
        return 0
    if any(status_counts.get(status) for status in ("frozen", "paused", "disabled")):
        return 2
    return 1


def _child_queue_artifact_candidates(
    artifacts: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for priority_index, artifact_key in enumerate(POST_FEEDBACK_CHILD_QUEUE_PRIORITY):
        raw_path = artifacts.get(artifact_key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = _resolve_path(raw_path, repo_root=repo_root)
        if not path.is_file():
            continue
        queue_payload = _load_json_object(path)
        if queue_payload.get("schema") != "experiment_queue.v1":
            continue
        status_counts = _queue_definition_status_counts(queue_payload)
        candidates.append(
            {
                "artifact_key": artifact_key,
                "queue_path": _repo_rel(path, repo_root),
                "priority_index": priority_index,
                "selection_rank": _queue_selection_rank(status_counts),
                "definition_status_counts": status_counts,
            }
        )
    candidates.sort(
        key=lambda row: (
            int(row["selection_rank"]),
            int(row["priority_index"]),
            str(row["artifact_key"]),
        )
    )
    return candidates


def _write_child_queue_activation_plan(
    *,
    queue_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any] | None:
    queue_payload = _load_json_object(queue_path)
    identity = _queue_identity_from_path(queue_path)
    activation_plan = _child_queue_activation_plan(
        queue_path=queue_path,
        queue_payload=queue_payload,
        queue_id=identity["queue_id"],
        queue_sha256=identity["queue_sha256"],
        repo_root=repo_root,
    )
    if not activation_plan["blocked_experiment_count"]:
        return None
    require_no_truthy_authority_fields(
        activation_plan,
        context="child_queue_activation_plan",
    )
    write_result = write_json_artifact(output_path, activation_plan)
    activation_plan["artifact_path"] = _repo_rel(output_path, repo_root)
    activation_plan["artifact_sha256"] = write_result.sha256
    activation_plan["artifact_bytes"] = write_result.bytes_written
    return activation_plan


def _write_child_queue_activation_learning_signal_report(
    *,
    activation_plan: Mapping[str, Any],
    activation_plan_path: str | Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    signal_report = build_repair_campaign_activation_plan_learning_signal_report(
        activation_plan_path=activation_plan_path,
        activation_plan=activation_plan,
        repo_root=repo_root,
    )
    require_no_truthy_authority_fields(
        signal_report,
        context="child_queue_activation_learning_signal_report",
    )
    write_result = write_json_artifact(output_path, signal_report)
    signal_report["artifact_path"] = _repo_rel(output_path, repo_root)
    signal_report["artifact_sha256"] = write_result.sha256
    signal_report["artifact_bytes"] = write_result.bytes_written
    return signal_report


def _append_activation_learning_signal_report_to_posterior(
    *,
    signal_report: Mapping[str, Any],
    signal_report_path: str | Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    posterior_path = (
        repo_root / ".omx" / "state" / "repair_campaign_stackability_posterior.jsonl"
    )
    lock_path = (
        repo_root / ".omx" / "state" / ".repair_campaign_stackability_posterior.lock"
    )
    append_report = append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=lock_path,
        repo_root=repo_root,
    )
    require_no_truthy_authority_fields(
        append_report,
        context="child_queue_activation_posterior_append_report",
    )
    write_result = write_json_artifact(output_path, append_report)
    append_report["artifact_path"] = _repo_rel(output_path, repo_root)
    append_report["artifact_sha256"] = write_result.sha256
    append_report["artifact_bytes"] = write_result.bytes_written
    return append_report


def _queue_identity_from_path(path: Path) -> dict[str, str]:
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
    return {"queue_id": queue_id, "queue_sha256": stable_json_sha256(payload)}


def _observer_revalidation(
    *,
    queue_id: str,
    queue_sha256: str,
    observation: Mapping[str, Any] | None,
    observer_path: Path,
    observer_write_result: ArtifactWriteResult | None,
    repo_root: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    observed_queue_id: str | None = None
    observed_queue_sha256: str | None = None
    observed_schema: str | None = None
    observe_read_only: bool | None = None
    if observation is None:
        blockers.append("observer_payload_missing_or_not_json_object")
    else:
        observed_schema = str(observation.get("schema") or "")
        observed_queue_id = str(observation.get("queue_id") or "")
        observed_queue_sha256 = str(observation.get("queue_sha256") or "")
        observe_read_only = observation.get("observe_read_only") is True
        if observed_schema != "experiment_queue_observation.v1":
            blockers.append("observer_schema_mismatch")
        if observed_queue_id != queue_id:
            blockers.append("observer_queue_id_mismatch")
        if not observed_queue_sha256:
            blockers.append("observer_queue_sha256_missing")
        elif observed_queue_sha256 != queue_sha256:
            blockers.append("observer_queue_sha256_mismatch")
        if observe_read_only is not True:
            blockers.append("observer_read_only_flag_missing")
    if observer_write_result is None:
        blockers.append("observer_revalidation_artifact_not_written")
    return {
        "schema": "frontier_final_rate_attack_child_queue_observer_revalidation.v1",
        "valid": not blockers,
        "blockers": blockers,
        "expected_queue_id": queue_id,
        "observed_queue_id": observed_queue_id,
        "expected_queue_sha256": queue_sha256,
        "observed_queue_sha256": observed_queue_sha256,
        "observed_schema": observed_schema,
        "observe_read_only": observe_read_only,
        "observer_revalidation_path": (
            _repo_rel(observer_path, repo_root) if observer_write_result is not None else None
        ),
        "observer_revalidation_sha256": (
            observer_write_result.sha256 if observer_write_result is not None else None
        ),
        "observer_revalidation_bytes": (
            observer_write_result.bytes_written if observer_write_result is not None else None
        ),
        "allowed_use": "local_observer_custody_revalidation_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


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
    candidates = _child_queue_artifact_candidates(artifacts, repo_root=repo)
    return [
        {
            "artifact_key": str(row["artifact_key"]),
            "queue_path": str(row["queue_path"]),
        }
        for row in candidates[:limit]
    ]


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
    queue_identity = _queue_identity_from_path(queue)
    queue_id = queue_identity["queue_id"]
    queue_sha256 = queue_identity["queue_sha256"]
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
    observer_write_result: ArtifactWriteResult | None = None
    worker_result = _json_stdout_object(results[2] if len(results) > 2 else None)
    if len(results) == len(commands) and int(results[-1].get("returncode") or 0) == 0:
        observation = _json_stdout_object(results[-1])
        if observation is not None:
            observer_write_result = write_json_artifact(observer_path, observation)

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
    queued_after = _observation_status_count(observation, "queued")
    frozen_after = _observation_status_count(observation, "frozen")
    paused_after = _observation_status_count(observation, "paused")
    disabled_after = _observation_status_count(observation, "disabled")
    activation_plan: dict[str, Any] | None = None
    activation_learning_signal_report: dict[str, Any] | None = None
    activation_posterior_append_report: dict[str, Any] | None = None
    if frozen_after or paused_after or disabled_after:
        activation_plan = _write_child_queue_activation_plan(
            queue_path=queue,
            output_path=observer_path.parent / "activation_plan.json",
            repo_root=repo,
        )
        if activation_plan is not None and activation_plan.get("artifact_path"):
            activation_learning_signal_report = (
                _write_child_queue_activation_learning_signal_report(
                    activation_plan=activation_plan,
                    activation_plan_path=str(activation_plan["artifact_path"]),
                    output_path=(
                        observer_path.parent / "activation_learning_signal_report.json"
                    ),
                    repo_root=repo,
                )
            )
            activation_posterior_append_report = (
                _append_activation_learning_signal_report_to_posterior(
                    signal_report=activation_learning_signal_report,
                    signal_report_path=str(
                        activation_learning_signal_report["artifact_path"]
                    ),
                    output_path=(
                        observer_path.parent
                        / "activation_posterior_append_report.json"
                    ),
                    repo_root=repo,
                )
            )
    progress_made = None if steps_started is None else steps_started > 0
    progress_blockers = []
    observer_revalidation = _observer_revalidation(
        queue_id=queue_id,
        queue_sha256=queue_sha256,
        observation=observation,
        observer_path=observer_path,
        observer_write_result=observer_write_result,
        repo_root=repo,
    )
    observer_revalidation_blockers = [
        str(blocker)
        for blocker in observer_revalidation.get("blockers", [])
        if str(blocker)
    ]
    if observer_revalidation_blockers:
        progress_made = False
        progress_blockers.extend(
            f"observer_revalidation:{blocker}"
            for blocker in observer_revalidation_blockers
        )
    if progress_made is False:
        if queued_after > 0:
            progress_blockers.append("child_queue_worker_started_zero_steps_with_queued_work")
        elif frozen_after > 0:
            progress_blockers.append("child_queue_remaining_work_frozen_by_definition")
        elif paused_after > 0:
            progress_blockers.append("child_queue_remaining_work_paused_by_definition")
        elif disabled_after > 0:
            progress_blockers.append("child_queue_remaining_work_disabled_by_definition")
    return {
        "schema": POST_FEEDBACK_CHILD_QUEUE_RUN_SCHEMA,
        "queue_id": queue_id,
        "queue_sha256": queue_sha256,
        "queue_path": queue_ref,
        "state_path": state_ref,
        "observer_revalidation_path": (
            _repo_rel(observer_path, repo) if observation is not None else None
        ),
        "observer_revalidation": observer_revalidation,
        "observer_revalidation_valid": observer_revalidation.get("valid") is True,
        "observer_revalidation_blockers": observer_revalidation_blockers,
        "commands": command_records,
        "failed_command_count": failed_count,
        "steps_started": steps_started,
        "progress_made": progress_made,
        "progress_blockers": progress_blockers,
        "queue_healthy": observation.get("healthy") is True if observation else False,
        "queue_status_counts": dict(observation.get("status_counts") or {}) if observation else {},
        "queue_blockers": list(observation.get("blockers") or []) if observation else [],
        "activation_plan": activation_plan,
        "activation_plan_path": activation_plan.get("artifact_path")
        if activation_plan is not None
        else None,
        "activation_plan_sha256": activation_plan.get("artifact_sha256")
        if activation_plan is not None
        else None,
        "activation_learning_signal_report": activation_learning_signal_report,
        "activation_learning_signal_report_path": activation_learning_signal_report.get(
            "artifact_path"
        )
        if activation_learning_signal_report is not None
        else None,
        "activation_learning_signal_report_sha256": activation_learning_signal_report.get(
            "artifact_sha256"
        )
        if activation_learning_signal_report is not None
        else None,
        "activation_posterior_append_report": activation_posterior_append_report,
        "activation_posterior_append_report_path": activation_posterior_append_report.get(
            "artifact_path"
        )
        if activation_posterior_append_report is not None
        else None,
        "activation_posterior_appended_count": (
            activation_posterior_append_report.get("appended_count")
            if activation_posterior_append_report is not None
            else None
        ),
        "activation_posterior_skipped_duplicate_count": (
            activation_posterior_append_report.get("skipped_duplicate_count")
            if activation_posterior_append_report is not None
            else None
        ),
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
    candidates = _child_queue_artifact_candidates(feedback_artifacts, repo_root=repo)
    selected = [
        {
            "artifact_key": str(row["artifact_key"]),
            "queue_path": str(row["queue_path"]),
        }
        for row in candidates[:limit]
    ]
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
    deferred_activation_plans: list[dict[str, Any]] = []
    for row in candidates[limit:]:
        status_counts = row.get("definition_status_counts")
        if not isinstance(status_counts, Mapping):
            continue
        if _queue_selection_rank(status_counts) != 2:
            continue
        key = str(row["artifact_key"])
        queue_path = _resolve_path(str(row["queue_path"]), repo_root=repo)
        activation_plan = _write_child_queue_activation_plan(
            queue_path=queue_path,
            output_path=observation_dir / key / "activation_plan.json",
            repo_root=repo,
        )
        if activation_plan is None:
            continue
        activation_learning_signal_report = (
            _write_child_queue_activation_learning_signal_report(
                activation_plan=activation_plan,
                activation_plan_path=str(activation_plan["artifact_path"]),
                output_path=(
                    observation_dir / key / "activation_learning_signal_report.json"
                ),
                repo_root=repo,
            )
        )
        activation_posterior_append_report = (
            _append_activation_learning_signal_report_to_posterior(
                signal_report=activation_learning_signal_report,
                signal_report_path=str(
                    activation_learning_signal_report["artifact_path"]
                ),
                output_path=(
                    observation_dir / key / "activation_posterior_append_report.json"
                ),
                repo_root=repo,
            )
        )
        deferred_activation_plans.append(
            {
                "artifact_key": key,
                "queue_path": str(row["queue_path"]),
                "definition_status_counts": dict(status_counts),
                "activation_plan_path": activation_plan.get("artifact_path"),
                "activation_plan_sha256": activation_plan.get("artifact_sha256"),
                "activation_learning_signal_report_path": (
                    activation_learning_signal_report.get("artifact_path")
                ),
                "activation_learning_signal_report_sha256": (
                    activation_learning_signal_report.get("artifact_sha256")
                ),
                "activation_posterior_append_report_path": (
                    activation_posterior_append_report.get("artifact_path")
                ),
                "activation_posterior_appended_count": (
                    activation_posterior_append_report.get("appended_count")
                ),
                "activation_posterior_skipped_duplicate_count": (
                    activation_posterior_append_report.get("skipped_duplicate_count")
                ),
                "blocked_experiment_count": activation_plan.get(
                    "blocked_experiment_count"
                ),
            }
        )
    report = {
        "schema": POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA,
        "generated_at_utc": _utc_now(),
        "selected_queue_count": len(selected),
        "executed_queue_count": len(runs),
        "failed_queue_count": sum(1 for run in runs if int(run.get("failed_command_count") or 0) > 0),
        "failed_command_count": sum(int(run.get("failed_command_count") or 0) for run in runs),
        "observer_revalidation_failed_count": sum(
            1 for run in runs if run.get("observer_revalidation_valid") is not True
        ),
        "stalled_queue_count": sum(
            1 for run in runs if run.get("progress_made") is False and run.get("queue_status_counts", {}).get("queued", 0)
        ),
        "frozen_noop_queue_count": sum(
            1
            for run in runs
            if run.get("progress_made") is False
            and not run.get("queue_status_counts", {}).get("queued", 0)
            and run.get("queue_status_counts", {}).get("frozen", 0)
        ),
        "activation_plan_count": sum(1 for run in runs if run.get("activation_plan_path")),
        "activation_learning_signal_report_count": sum(
            1 for run in runs if run.get("activation_learning_signal_report_path")
        ),
        "deferred_activation_plan_count": len(deferred_activation_plans),
        "deferred_activation_learning_signal_report_count": sum(
            1
            for plan in deferred_activation_plans
            if plan.get("activation_learning_signal_report_path")
        ),
        "selected_activation_posterior_append_report_count": sum(
            1 for run in runs if run.get("activation_posterior_append_report_path")
        ),
        "deferred_activation_posterior_append_report_count": sum(
            1
            for plan in deferred_activation_plans
            if plan.get("activation_posterior_append_report_path")
        ),
        "activation_posterior_append_report_count": sum(
            1 for run in runs if run.get("activation_posterior_append_report_path")
        )
        + sum(
            1
            for plan in deferred_activation_plans
            if plan.get("activation_posterior_append_report_path")
        ),
        "activation_posterior_appended_count": sum(
            int(run.get("activation_posterior_appended_count") or 0)
            for run in runs
        )
        + sum(
            int(plan.get("activation_posterior_appended_count") or 0)
            for plan in deferred_activation_plans
        ),
        "activation_posterior_skipped_duplicate_count": sum(
            int(run.get("activation_posterior_skipped_duplicate_count") or 0)
            for run in runs
        )
        + sum(
            int(plan.get("activation_posterior_skipped_duplicate_count") or 0)
            for plan in deferred_activation_plans
        ),
        "max_steps": max_steps,
        "max_parallel": max_parallel,
        "limit": limit,
        "selected_queues": selected,
        "deferred_activation_plans": deferred_activation_plans,
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
    "POST_FEEDBACK_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA",
    "POST_FEEDBACK_CHILD_QUEUE_PRIORITY",
    "POST_FEEDBACK_CHILD_QUEUE_RUNS_SCHEMA",
    "execute_post_feedback_child_queues",
    "run_experiment_queue_once",
    "select_post_feedback_child_queue_artifacts",
]
