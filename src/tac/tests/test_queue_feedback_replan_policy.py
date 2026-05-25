# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys

import pytest

from comma_lab.scheduler.queue_feedback_replan_policy import (
    ACTION_BLOCKED,
    ACTION_EXECUTE_FEEDBACK_FOLLOWUP,
    ACTION_INSPECT_EXACT_HANDOFFS,
    ACTION_QUEUE_OBSERVATION_MAINTENANCE,
    ACTION_RECOVER_QUEUE_HEALTH,
    ACTION_REFUSE,
    ACTION_RUN_NEXT_ITERATION,
    ACTION_STOP_MAX_ITERATIONS,
    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
    QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA,
    QUEUE_FEEDBACK_REPLAN_CONTINUATION_EXPERIMENT_ID,
    QUEUE_FEEDBACK_REPLAN_CONTINUATION_METADATA_SCHEMA,
    QUEUE_FEEDBACK_REPLAN_CONTINUATION_STEP_ID,
    QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL,
    QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA,
    QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA,
    build_queue_feedback_replan_continuation_queue,
    build_queue_feedback_replan_policy,
    build_queue_observation_recovery_plan,
    validate_feedback_followup_queue,
)


def _run_summary(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "byte_shaving_materializer_campaign_run.v1",
        "run_dir": ".omx/research/campaign",
        "plan": ".omx/research/campaign/plan.json",
        "queue_id": "campaign_queue",
        "queue_path": ".omx/research/campaign/materializer_execution_queue.json",
        "state_path": ".omx/research/campaign/materializer_execution_queue.sqlite",
        "queue_performance_summary_path": ".omx/research/campaign/queue_performance_summary.json",
        "queue_feedback_replan_ready": True,
        "queue_feedback_replan_blockers": [],
        "queue_feedback_replan_followup_queue_emitted": True,
        "queue_feedback_replan_followup_queue_path": (
            ".omx/research/campaign/queue_feedback_replan_followup_queue.json"
        ),
        "queue_feedback_replan_followup_queue_blockers": [],
        "queue_feedback_replan_followup_policy_enabled": True,
        "queue_feedback_replan_followup_policy_blockers": [],
        "queue_feedback_replan_followup_executed": False,
        "queue_feedback_replan_followup_execution_success": None,
        "exact_readiness_handoff_count": 0,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
    }
    row.update(overrides)
    return row


def _calibration_request(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "exact_auth_calibration_packet_source": "run_derived_discovery",
        "exact_auth_calibration_packet_paths": [
            ".omx/research/campaign/cpu_result_review.json",
            ".omx/research/campaign/cuda_result_review.json",
        ],
        "exact_auth_calibration_discovery_pair": {
            "archive_sha256": "a" * 64,
            "archive_bytes": 12345,
            "n_samples": 600,
            "runtime_content_tree_sha256": "b" * 64,
            "contest_cpu_packet_path": (
                ".omx/research/campaign/cpu_result_review.json"
            ),
            "contest_cuda_packet_path": (
                ".omx/research/campaign/cuda_result_review.json"
            ),
        },
    }
    row.update(overrides)
    return row


def _child_queue(command: list[str] | None = None) -> dict[str, object]:
    return {
        "schema": "experiment_queue.v1",
        "queue_id": "campaign_queue_feedback_replan",
        "controls": {
            "mode": "paused",
            "local_first": True,
            "max_concurrency": {"local_cpu": 1},
        },
        "experiments": [
            {
                "id": "queue_feedback_replan_action_functional",
                "metadata": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_packet_ready": False,
                },
                "steps": [
                    {
                        "id": "build_feedback_action_functional",
                        "kind": "command",
                        "command": command
                        or [
                            sys.executable,
                            QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                            "--output",
                            (
                                ".omx/research/campaign/"
                                "inverse_steganalysis_action_functional.feedback.json"
                            ),
                            "--md-out",
                            (
                                ".omx/research/campaign/"
                                "inverse_steganalysis_action_functional.feedback.md"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                    }
                ],
            }
        ],
    }


def test_queue_observation_helpers_are_public_package_exports() -> None:
    from comma_lab.scheduler import (
        build_queue_observation_recovery_plan as scheduler_export,
    )
    from tac.optimization import observations_from_queue_observation

    assert scheduler_export is build_queue_observation_recovery_plan
    assert callable(observations_from_queue_observation)


def test_queue_observation_recovery_plan_builds_required_local_commands() -> None:
    plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": [
                "experiment_queue_observation_failed_steps:1",
                "experiment_queue_observation_blocked_steps:1",
                "experiment_queue_observation_artifact_postcondition_failures:1",
                "experiment_queue_observation_orphaned_steps:1",
            ],
            "orphaned_step_count": 1,
            "orphaned_steps": [
                {
                    "experiment_id": "old_exp",
                    "step_id": "old_step",
                    "status": "queued",
                }
            ],
            "failed_steps": [
                {
                    "experiment_id": "exp0",
                    "step_id": "materialize",
                    "status": "failed",
                }
            ],
            "blocked_steps": [
                {
                    "experiment_id": "exp1",
                    "step_id": "harvest",
                    "status": "blocked",
                }
            ],
            "succeeded_artifact_failure_steps": [
                {
                    "experiment_id": "exp2",
                    "step_id": "handoff",
                    "status": "succeeded",
                }
            ],
        },
        queue_path="queue.json",
        state_path="queue.sqlite",
        reason="fixture recovery",
    )

    assert plan["schema"] == QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA
    assert plan["recovery_required"] is True
    assert plan["required_action_count"] == 4
    assert plan["score_claim"] is False
    actions = {item["action"]: item for item in plan["actions"]}
    assert actions["retire_blocking_orphaned_steps"]["command"] == [
        sys.executable,
        "tools/experiment_queue.py",
        "--queue",
        "queue.json",
        "--state",
        "queue.sqlite",
        "retire-orphans",
        "--reason",
        "fixture recovery",
    ]
    assert actions["rewind_failed_step"]["command"][-5:] == [
        "rewind",
        "exp0",
        "materialize",
        "--reason",
        "fixture recovery",
    ]
    assert actions["rewind_blocked_step"]["experiment_id"] == "exp1"
    assert actions["rewind_succeeded_step_with_artifact_failure"]["step_id"] == (
        "handoff"
    )


def test_feedback_replan_policy_executes_safe_followup_queue() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(),
        feedback_followup_queue=_child_queue(),
        source_run_path=".omx/research/campaign/materializer_campaign_run.json",
        iteration_index=0,
        max_iterations=3,
    )

    assert policy["schema"] == QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA
    assert policy["decision"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
    assert policy["should_continue_feedback_loop"] is True
    assert policy["ready_for_feedback_followup_execution"] is True
    assert policy["feedback_followup_queue_validation"]["valid"] is True
    assert policy["feedback_followup_queue_validation"]["schema"] == (
        QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA
    )
    assert len(policy["feedback_followup_queue_validation"]["queue_sha256"]) == 64
    assert policy["score_claim"] is False
    assert policy["ready_for_exact_eval_dispatch"] is False
    assert policy["recommended_actions"][0]["action"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
    assert policy["recommended_actions"][0]["local_only"] is True


def test_feedback_replan_policy_counts_only_paired_exact_auth_calibration() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(queue_feedback_replan_request=_calibration_request()),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
    assert policy["exact_auth_calibration_usable"] is True
    calibration = policy["exact_auth_calibration_policy"]
    assert calibration["usable_for_feedback_trust_region"] is True
    assert calibration["packet_count"] == 2
    assert calibration["pair"]["archive_bytes"] == 12345
    assert calibration["score_claim"] is False
    assert calibration["ready_for_exact_eval_dispatch"] is False


def test_feedback_replan_policy_prioritizes_queue_health_recovery() -> None:
    recovery_plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": ["experiment_queue_observation_failed_steps:1"],
            "failed_steps": [
                {
                    "experiment_id": "exp0",
                    "step_id": "materialize",
                    "status": "failed",
                }
            ],
        },
        queue_path=".omx/research/campaign/materializer_execution_queue.json",
        state_path=".omx/research/campaign/materializer_execution_queue.sqlite",
    )
    policy = build_queue_feedback_replan_policy(
        _run_summary(queue_observation_recovery_plan=recovery_plan),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_RECOVER_QUEUE_HEALTH
    assert policy["should_continue_feedback_loop"] is False
    assert policy["ready_for_queue_health_recovery"] is True
    assert policy["operator_queue_state_mutation_required"] is True
    assert policy["auto_execute_eligible"] is False
    assert policy["ready_for_feedback_followup_execution"] is False
    assert policy["queue_observation_recovery_required"] is True
    assert policy["recommended_actions"][0]["action"] == ACTION_RECOVER_QUEUE_HEALTH
    assert policy["recommended_actions"][0]["auto_execute_eligible"] is False
    assert policy["recommended_actions"][0][
        "operator_queue_state_mutation_required"
    ] is True
    assert policy["recommended_actions"][0]["actions"][0]["action"] == (
        "rewind_failed_step"
    )
    assert policy["score_claim"] is False
    assert policy["ready_for_exact_eval_dispatch"] is False
    continuation_queue, continuation_blockers = (
        build_queue_feedback_replan_continuation_queue(
            policy,
            lane_id="queue_recovery_regression",
            source_policy_path=".omx/research/campaign/queue_feedback_replan_policy.json",
        )
    )
    assert continuation_queue is None
    assert "queue_feedback_replan_continuation_policy_not_next_iteration" in (
        continuation_blockers
    )
    assert "queue_feedback_replan_continuation_policy_not_continuable" in (
        continuation_blockers
    )


def test_feedback_replan_policy_keeps_nonblocking_observation_maintenance_advisory() -> None:
    recovery_plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": ["experiment_queue_observation_orphaned_steps:1"],
            "orphaned_step_count": 1,
            "orphaned_steps": [
                {
                    "experiment_id": "old_exp",
                    "step_id": "old_step",
                    "status": "skipped",
                }
            ],
        },
        queue_path=".omx/research/campaign/materializer_execution_queue.json",
        state_path=".omx/research/campaign/materializer_execution_queue.sqlite",
    )
    policy = build_queue_feedback_replan_policy(
        _run_summary(queue_observation_recovery_plan=recovery_plan),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
    assert policy["queue_observation_recovery_required"] is False
    assert policy["queue_observation_maintenance_recommended"] is True
    assert policy["recommended_actions"][0]["action"] == (
        ACTION_QUEUE_OBSERVATION_MAINTENANCE
    )
    assert policy["recommended_actions"][1]["action"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP


def test_feedback_replan_policy_blocks_unpaired_exact_auth_calibration() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_feedback_replan_request=_calibration_request(
                exact_auth_calibration_packet_paths=[
                    ".omx/research/campaign/cpu_result_review.json",
                    ".omx/research/campaign/cuda_result_review.json",
                ],
                exact_auth_calibration_discovery_pair=None,
            )
        ),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_BLOCKED
    assert policy["stop_reason"] == "exact_auth_calibration_policy_failed"
    assert policy["exact_auth_calibration_usable"] is False
    assert (
        "exact_auth_calibration_policy:exact_auth_calibration_pair_metadata_missing"
        in policy["blockers"]
    )


def test_feedback_replan_policy_builds_next_iteration_command_after_success() -> None:
    action_path = ".omx/research/campaign/inverse_steganalysis_action_functional.feedback.json"
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=action_path,
        ),
        feedback_followup_queue=_child_queue(),
        iteration_index=1,
        max_iterations=3,
    )

    assert policy["decision"] == ACTION_RUN_NEXT_ITERATION
    assert policy["ready_for_local_materialization"] is True
    assert policy["next_iteration_index"] == 2
    assert policy["next_iteration_command_template"] == [
        sys.executable,
        QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL,
        "--plan",
        ".omx/research/campaign/plan.json",
        "--inverse-scorer-action-functional",
        action_path,
        "--queue-feedback-replan-policy-iteration",
        "2",
        "--queue-feedback-replan-policy-max-iterations",
        "3",
    ]
    assert policy["recommended_actions"][-1]["requires_materializer_context_or_artifact_map"] is True


def test_feedback_replan_policy_builds_paused_next_iteration_queue() -> None:
    action_path = ".omx/research/campaign/inverse_steganalysis_action_functional.feedback.json"
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=action_path,
        ),
        feedback_followup_queue=_child_queue(),
        iteration_index=1,
        max_iterations=3,
    )

    queue, blockers = build_queue_feedback_replan_continuation_queue(
        policy,
        lane_id="feedback_lane",
        source_policy_path=".omx/research/campaign/queue_feedback_replan_policy.json",
    )

    assert blockers == []
    assert queue is not None
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["queue_id"] == "campaign_queue_feedback_continue_2"
    assert queue["controls"] == {
        "mode": "paused",
        "local_first": True,
        "max_concurrency": {"local_cpu": 1},
    }
    experiment = queue["experiments"][0]
    assert experiment["id"] == QUEUE_FEEDBACK_REPLAN_CONTINUATION_EXPERIMENT_ID
    assert experiment["lane_id"] == "feedback_lane"
    metadata = experiment["metadata"]
    assert metadata["schema"] == QUEUE_FEEDBACK_REPLAN_CONTINUATION_METADATA_SCHEMA
    assert metadata["source_policy_path"] == (
        ".omx/research/campaign/queue_feedback_replan_policy.json"
    )
    assert metadata["source_policy_sha256"]
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert (
        "exact_auth_eval_required_before_score_claim"
        in metadata["dispatch_blockers"]
    )
    step = experiment["steps"][0]
    assert step["id"] == QUEUE_FEEDBACK_REPLAN_CONTINUATION_STEP_ID
    assert step["resources"] == {"kind": "local_cpu"}
    assert step["command"] == policy["next_iteration_command_template"]


def test_feedback_replan_continuation_queue_refuses_unsafe_policy_command() -> None:
    action_path = ".omx/research/campaign/inverse_steganalysis_action_functional.feedback.json"
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=action_path,
        ),
        feedback_followup_queue=_child_queue(),
        iteration_index=1,
        max_iterations=3,
    )
    policy = {
        **policy,
        "next_iteration_command_template": [
            sys.executable,
            QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL,
            "--plan",
            ".omx/research/campaign/plan.json",
            "--inverse-scorer-action-functional",
            action_path,
            "--queue-feedback-replan-policy-iteration",
            "2",
            "--queue-feedback-replan-policy-max-iterations",
            "3",
            "--provider=modal",
        ],
    }

    queue, blockers = build_queue_feedback_replan_continuation_queue(
        policy,
        lane_id="feedback_lane",
    )

    assert queue is None
    assert (
        "queue_feedback_replan_continuation_command_forbidden_flag:--provider=modal"
        in blockers
    )


def test_feedback_replan_policy_refuses_truthy_authority() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            score_claim=True,
            ready_for_exact_eval_dispatch=True,
            dispatch_packet_ready=True,
        ),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_REFUSE
    assert policy["should_continue_feedback_loop"] is False
    assert "source_run_truthy_authority_field:score_claim=truthy" in policy["blockers"]
    assert (
        "source_run_truthy_authority_field:ready_for_exact_eval_dispatch=truthy"
        in policy["blockers"]
    )
    assert (
        "source_run_truthy_authority_field:dispatch_packet_ready=truthy"
        in policy["blockers"]
    )


def test_feedback_replan_policy_blocks_missing_action_functional() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
        ),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_BLOCKED
    assert policy["stop_reason"] == "feedback_action_functional_missing"
    assert "feedback_action_functional_missing" in policy["blockers"]


def test_feedback_replan_policy_stops_at_iteration_limit() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(),
        feedback_followup_queue=_child_queue(),
        iteration_index=3,
        max_iterations=3,
    )

    assert policy["decision"] == ACTION_STOP_MAX_ITERATIONS
    assert policy["stop_reason"] == "max_iterations_reached"
    assert policy["should_continue_feedback_loop"] is False


def test_feedback_replan_policy_validates_bounds() -> None:
    with pytest.raises(ValueError, match="iteration_index"):
        build_queue_feedback_replan_policy(_run_summary(), iteration_index=-1)
    with pytest.raises(ValueError, match="max_iterations"):
        build_queue_feedback_replan_policy(_run_summary(), max_iterations=0)


def test_feedback_replan_policy_blocks_unvalidated_child_queue() -> None:
    policy = build_queue_feedback_replan_policy(_run_summary())

    assert policy["decision"] == ACTION_BLOCKED
    assert policy["stop_reason"] == "feedback_followup_queue_artifact_not_validated"
    assert "feedback_followup_queue_artifact_not_validated" in policy["blockers"]


def test_feedback_replan_policy_blocks_unsafe_child_queue_command() -> None:
    unsafe_queue = _child_queue(
        [
            sys.executable,
            QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
            "--output",
            ".omx/research/campaign/feedback.json",
            "--provider=modal",
        ]
    )

    policy = build_queue_feedback_replan_policy(
        _run_summary(),
        feedback_followup_queue=unsafe_queue,
    )

    assert policy["decision"] == ACTION_BLOCKED
    assert policy["stop_reason"] == "feedback_followup_queue_validation_failed"
    assert (
        "feedback_followup_queue_validation:"
        "queue_feedback_replan_followup_step_command_forbidden_flag:"
        "0:0:--provider=modal"
    ) in policy["blockers"]
    assert policy["ready_for_feedback_followup_execution"] is False


def test_feedback_replan_policy_blocks_child_output_outside_run_dir() -> None:
    unsafe_queue = _child_queue(
        [
            sys.executable,
            QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
            "--output",
            ".omx/other/feedback.json",
        ]
    )

    validation = validate_feedback_followup_queue(
        unsafe_queue,
        run_dir=".omx/research/campaign",
    )

    assert validation["valid"] is False
    assert (
        "queue_feedback_replan_followup_step_output_outside_run_dir:"
        "0:0:.omx/other/feedback.json"
    ) in validation["blockers"]


def test_feedback_replan_policy_blocks_telemetry_truncation() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            performance={
                "schema": "experiment_queue_performance_summary.v1",
                "recursive_artifact_record_truncated": True,
            }
        ),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_BLOCKED
    assert policy["stop_reason"] == "queue_performance_telemetry_truncated"
    assert (
        "queue_performance_telemetry_truncated:"
        "recursive_artifact_record_truncated=truthy"
    ) in policy["blockers"]


def test_feedback_replan_policy_keeps_exact_handoffs_separate_from_local_loop() -> None:
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            exact_readiness_handoff_count=1,
            exact_readiness_handoff_paths=[{"path": ".omx/research/campaign/handoff.json"}],
        ),
        feedback_followup_queue=_child_queue(),
    )

    assert policy["decision"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
    assert policy["recommended_actions"][0]["action"] == ACTION_INSPECT_EXACT_HANDOFFS
    assert policy["recommended_actions"][0]["ready_for_exact_eval_dispatch"] is False
    assert policy["recommended_actions"][1]["action"] == ACTION_EXECUTE_FEEDBACK_FOLLOWUP
