# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import (
    DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS,
    DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS,
    connect_state,
    initialize_queue_state,
)
from comma_lab.scheduler.queue_feedback_replan_policy import (
    ACTION_BLOCKED,
    ACTION_EXECUTE_FEEDBACK_FOLLOWUP,
    ACTION_INSPECT_EXACT_HANDOFFS,
    ACTION_QUEUE_OBSERVATION_MAINTENANCE,
    ACTION_RECOVER_QUEUE_HEALTH,
    ACTION_REFUSE,
    ACTION_RUN_NEXT_ITERATION,
    ACTION_STOP_MAX_ITERATIONS,
    ACTION_WIDEN_CANDIDATE_GENERATION,
    QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID,
    QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_EXPERIMENT_ID,
    QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_STEP_ID,
    QUEUE_FEEDBACK_CANDIDATE_WIDENING_EXPERIMENT_ID,
    QUEUE_FEEDBACK_CANDIDATE_WIDENING_METADATA_SCHEMA,
    QUEUE_FEEDBACK_CANDIDATE_WIDENING_STEP_ID,
    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
    QUEUE_FEEDBACK_REPLAN_CHILD_QUEUE_VALIDATION_SCHEMA,
    QUEUE_FEEDBACK_REPLAN_CONTINUATION_EXPERIMENT_ID,
    QUEUE_FEEDBACK_REPLAN_CONTINUATION_METADATA_SCHEMA,
    QUEUE_FEEDBACK_REPLAN_CONTINUATION_STEP_ID,
    QUEUE_FEEDBACK_REPLAN_MATERIALIZER_CAMPAIGN_TOOL,
    QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA,
    QUEUE_OBSERVATION_RECOVERY_EXPERIMENT_ID,
    QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA,
    QUEUE_OBSERVATION_RECOVERY_QUEUE_METADATA_SCHEMA,
    QUEUE_OBSERVATION_RECOVERY_QUEUE_VALIDATION_SCHEMA,
    build_queue_feedback_candidate_actuation_planning_queue,
    build_queue_feedback_candidate_widening_queue,
    build_queue_feedback_replan_continuation_queue,
    build_queue_feedback_replan_policy,
    build_queue_observation_recovery_plan,
    build_queue_observation_recovery_queue,
    validate_feedback_followup_queue,
    validate_queue_observation_recovery_queue,
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
    from comma_lab.scheduler import (
        build_queue_observation_recovery_queue as scheduler_queue_export,
    )
    from comma_lab.scheduler import (
        validate_queue_observation_recovery_queue as scheduler_queue_validator,
    )
    from tac.optimization import observations_from_queue_observation

    assert scheduler_export is build_queue_observation_recovery_plan
    assert scheduler_queue_export is build_queue_observation_recovery_queue
    assert scheduler_queue_validator is validate_queue_observation_recovery_queue
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
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer_id": "entropy_adapter",
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
    entropy_groups = [
        group
        for group in plan["grouped_blockers"]
        if group["scope_kind"] == "materializer_target"
        and group["scope_value"]
        == "entropy_adapter:archive_section_entropy_recode_v1"
    ]
    assert len(entropy_groups) == 3
    assert all(group["score_claim"] is False for group in plan["grouped_blockers"])
    assert entropy_groups[0]["target_kinds"] == [
        "archive_section_entropy_recode_v1"
    ]
    assert entropy_groups[0]["materializer_ids"] == ["entropy_adapter"]
    assert entropy_groups[0]["recommended_planning_effect"] == (
        "block_followup_until_recovery_queue_runs"
    )


def test_queue_observation_recovery_plan_groups_repeated_materializer_blockers() -> None:
    plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": [
                "experiment_queue_observation_failed_steps:2",
            ],
            "failed_steps": [
                {
                    "experiment_id": "exp0",
                    "step_id": "materialize",
                    "status": "failed",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer_id": "entropy_adapter",
                    "source_selection_ids": ["compiled_direct_selection"],
                },
                {
                    "experiment_id": "exp0",
                    "step_id": "proof",
                    "status": "failed",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer_id": "entropy_adapter",
                    "source_selection_ids": ["compiled_direct_selection"],
                }
            ],
        },
        queue_path="queue.json",
        state_path="queue.sqlite",
    )

    groups = {
        group["group_id"]: group for group in plan["grouped_blockers"]
    }
    repeated = [
        group
        for group in groups.values()
        if group["scope_kind"] == "materializer_target"
        and group["blocker_family"] == "experiment_queue_observation_failed_steps"
    ]

    assert plan["grouped_blocker_count"] == 1
    assert plan["repeated_group_count"] == 1
    assert len(repeated) == 1
    assert repeated[0]["count"] == 2
    assert repeated[0]["repeated"] is True
    assert repeated[0]["source_selection_ids"] == ["compiled_direct_selection"]
    assert repeated[0]["affected_step_ids"] == ["materialize", "proof"]
    assert repeated[0]["recommended_planning_effect"] == (
        "block_followup_until_recovery_queue_runs"
    )
    assert all(group["ready_for_exact_eval_dispatch"] is False for group in repeated)


def test_queue_observation_recovery_plan_records_receiver_negative_artifact_feedback() -> None:
    plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": [
                "experiment_queue_observation_failed_steps:1",
                "experiment_queue_observation_artifact_postcondition_failures:1",
            ],
            "failed_steps": [
                {
                    "experiment_id": "exp0",
                    "step_id": "materialize",
                    "status": "failed",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer_id": "archive_section_entropy_recode_adapter",
                    "receiver_contract_kind": (
                        "family_agnostic_archive_section_entropy_recode"
                    ),
                    "expected_artifacts": [
                        {
                            "path": "candidate_manifest.json",
                            "exists": True,
                            "postcondition_passed": False,
                            "receiver_verification": {
                                "schema": (
                                    "family_agnostic_runtime_consumption_proof_"
                                    "verification.v1"
                                ),
                                "receiver_contract_satisfied": False,
                            },
                        }
                    ],
                }
            ],
        },
        queue_path="queue.json",
        state_path="queue.sqlite",
    )

    actions = {item["action"]: item for item in plan["actions"]}
    feedback = actions["record_materializer_receiver_feedback"]

    assert "rewind_failed_step" not in actions
    assert feedback["required"] is False
    assert feedback["command"] is None
    assert feedback["score_claim"] is False
    assert feedback["ready_for_exact_eval_dispatch"] is False
    assert feedback["expected_artifact_paths"] == ["candidate_manifest.json"]
    assert (
        "materializer_receiver_verification_unsatisfied:candidate_manifest.json"
        in feedback["blocker_sources"]
    )
    receiver_groups = [
        group
        for group in plan["grouped_blockers"]
        if group["scope_kind"] == "materializer_receiver"
    ]
    assert receiver_groups
    assert receiver_groups[0]["recommended_planning_effect"] == (
        "advisory_maintenance_only"
    )
    assert receiver_groups[0]["ready_for_exact_eval_dispatch"] is False


def test_queue_observation_recovery_plan_records_archive_zip_repack_blocker_feedback() -> None:
    plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": [
                "experiment_queue_observation_artifact_postcondition_failures:1"
            ],
            "succeeded_artifact_failure_steps": [
                {
                    "experiment_id": "exp0",
                    "step_id": "archive_zip_repack",
                    "status": "succeeded",
                    "target_kind": "archive_zip_repack_v1",
                    "materializer_id": "archive_zip_repack_adapter",
                    "receiver_contract_kind": "family_agnostic_archive_zip_repack",
                    "expected_artifacts": [
                        {
                            "path": "archive_zip_repack/candidate.json",
                            "exists": True,
                            "postcondition_passed": False,
                            "json_schema": "archive_zip_repack_candidate.v1",
                            "receiver_contract_satisfied": True,
                            "readiness_blockers": ["candidate_not_rate_positive"],
                            "receiver_verification": {
                                "schema": (
                                    "family_agnostic_runtime_consumption_proof_"
                                    "verification.v1"
                                ),
                                "receiver_contract_satisfied": True,
                                "blockers": [],
                            },
                        }
                    ],
                }
            ],
        },
        queue_path="queue.json",
        state_path="queue.sqlite",
    )

    actions = {item["action"]: item for item in plan["actions"]}
    feedback = actions["record_materializer_receiver_feedback"]

    assert "rewind_succeeded_step_with_artifact_failure" not in actions
    assert feedback["required"] is False
    assert feedback["command"] is None
    assert feedback["target_kind"] == "archive_zip_repack_v1"
    assert feedback["materializer_id"] == "archive_zip_repack_adapter"
    assert feedback["receiver_contract_kind"] == "family_agnostic_archive_zip_repack"
    assert feedback["expected_artifact_paths"] == [
        "archive_zip_repack/candidate.json"
    ]
    assert (
        "materializer_readiness_blocker:candidate_not_rate_positive"
        in feedback["blocker_sources"]
    )
    receiver_groups = [
        group
        for group in plan["grouped_blockers"]
        if group["scope_kind"] == "materializer_receiver"
    ]
    assert receiver_groups
    assert receiver_groups[0]["scope_value"] == (
        "archive_zip_repack_adapter:family_agnostic_archive_zip_repack"
    )
    assert receiver_groups[0]["recommended_planning_effect"] == (
        "advisory_maintenance_only"
    )
    assert receiver_groups[0]["ready_for_exact_eval_dispatch"] is False


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


def test_feedback_replan_policy_prioritizes_queue_health_recovery(
    tmp_path: Path,
) -> None:
    source_queue_path = tmp_path / "materializer_execution_queue.json"
    source_state_path = tmp_path / "materializer_execution_queue.sqlite"
    source_queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "campaign_queue",
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {"local_cpu": 1},
        },
        "experiments": [
            {
                "id": "exp0",
                "steps": [
                    {
                        "id": "materialize",
                        "kind": "command",
                        "command": [sys.executable, "-c", "print('materialize')"],
                        "resources": {"kind": "local_cpu"},
                    }
                ],
            }
        ],
    }
    source_queue_path.write_text(json.dumps(source_queue), encoding="utf-8")
    with connect_state(source_state_path) as conn:
        initialize_queue_state(conn, source_queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T03:00:00Z'
            WHERE queue_id = 'campaign_queue'
              AND experiment_id = 'exp0'
              AND step_id = 'materialize'
            """,
            (json.dumps({"reason": "fixture failed materializer"}),),
        )
        conn.commit()
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
        queue_path=str(source_queue_path),
        state_path=str(source_state_path),
    )
    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_path=str(source_queue_path),
            state_path=str(source_state_path),
            queue_observation_recovery_plan=recovery_plan,
        ),
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
    recovery_queue, recovery_blockers = build_queue_observation_recovery_queue(
        policy,
        lane_id="queue_recovery_regression",
        source_policy_path=".omx/research/campaign/queue_feedback_replan_policy.json",
    )
    assert recovery_blockers == []
    assert recovery_queue is not None
    assert recovery_queue["schema"] == "experiment_queue.v1"
    assert recovery_queue["controls"]["mode"] == "paused"
    assert recovery_queue["controls"]["max_concurrency"] == {"local_cpu": 1}
    experiment = recovery_queue["experiments"][0]
    assert experiment["id"] == QUEUE_OBSERVATION_RECOVERY_EXPERIMENT_ID
    assert experiment["metadata"]["schema"] == (
        QUEUE_OBSERVATION_RECOVERY_QUEUE_METADATA_SCHEMA
    )
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    assert experiment["metadata"]["operator_queue_state_mutation_required"] is True
    assert experiment["metadata"]["auto_execute_eligible"] is False
    assert experiment["steps"][0]["id"] == (
        "recover_000_rewind_failed_step_exp0_materialize"
    )
    assert experiment["steps"][0]["resources"] == {"kind": "local_cpu"}
    assert experiment["steps"][0]["command"][-5:] == [
        "rewind",
        "exp0",
        "materialize",
        "--reason",
        "queue observation health recovery",
    ]
    validation = validate_queue_observation_recovery_queue(recovery_queue)
    assert validation["schema"] == QUEUE_OBSERVATION_RECOVERY_QUEUE_VALIDATION_SCHEMA
    assert validation["valid"] is True
    assert validation["command_count"] == 1
    assert validation["source_queue_paths"] == [str(source_queue_path)]
    assert validation["source_state_paths"] == [str(source_state_path)]
    assert validation["expected_source_queue_sha256s"] == (
        validation["current_source_queue_sha256s"]
    )
    assert validation["expected_source_state_watermarks"] == (
        validation["current_source_state_watermarks"]
    )
    assert validation["score_claim"] is False
    assert validation["ready_for_exact_eval_dispatch"] is False
    assert validation["blockers"] == []

    unsafe_queue = dict(recovery_queue)
    unsafe_queue["controls"] = {
        **dict(recovery_queue["controls"]),
        "mode": "running",
        "max_concurrency": {"local_cpu": 1, "modal_gpu": 1},
    }
    unsafe_queue["experiments"] = [
        {
            **experiment,
            "metadata": {
                **dict(experiment["metadata"]),
                "score_claim": True,
            },
            "steps": [
                {
                    **experiment["steps"][0],
                    "command": [
                        sys.executable,
                        "tools/experiment_queue.py",
                        "--queue",
                        "other.json",
                        "--state",
                        str(source_state_path),
                        "rewind",
                        "exp0",
                        "materialize",
                    ],
                }
            ],
        }
    ]
    unsafe_validation = validate_queue_observation_recovery_queue(unsafe_queue)
    assert unsafe_validation["valid"] is False
    assert "queue_observation_recovery_control_mode_not_paused" in unsafe_validation[
        "blockers"
    ]
    assert "queue_observation_recovery_non_local_concurrency:modal_gpu" in (
        unsafe_validation["blockers"]
    )
    assert (
        "queue_observation_recovery_truthy_authority_field:"
        "experiments[0].metadata.score_claim=truthy"
    ) in unsafe_validation["blockers"]
    assert (
        "queue_observation_recovery_step_command_validation:0:"
        "queue_observation_recovery_command_queue_mismatch:0"
    ) in unsafe_validation["blockers"]
    with connect_state(source_state_path) as conn:
        conn.execute(
            """
            UPDATE step_state
            SET updated_at_utc = '2026-05-25T03:01:00Z'
            WHERE queue_id = 'campaign_queue'
              AND experiment_id = 'exp0'
              AND step_id = 'materialize'
            """
        )
        conn.commit()
    drifted_validation = validate_queue_observation_recovery_queue(recovery_queue)
    assert drifted_validation["valid"] is False
    assert (
        "queue_observation_recovery_source_state_watermark_drift:0"
        in drifted_validation["blockers"]
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
    recovery_queue, recovery_blockers = build_queue_observation_recovery_queue(
        policy,
        lane_id="queue_recovery_regression",
    )
    assert recovery_queue is None
    assert "queue_observation_recovery_policy_not_recovery" in recovery_blockers
    assert "queue_observation_recovery_policy_not_ready" in recovery_blockers


def test_queue_observation_recovery_queue_requires_command_backed_actions() -> None:
    recovery_plan = build_queue_observation_recovery_plan(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "campaign_queue",
            "healthy": False,
            "blockers": [
                "experiment_queue_observation_artifact_postcondition_failures:1"
            ],
        },
        queue_path=".omx/research/campaign/materializer_execution_queue.json",
        state_path=".omx/research/campaign/materializer_execution_queue.sqlite",
    )
    policy = build_queue_feedback_replan_policy(
        _run_summary(queue_observation_recovery_plan=recovery_plan),
        feedback_followup_queue=_child_queue(),
    )

    recovery_queue, recovery_blockers = build_queue_observation_recovery_queue(
        policy,
        lane_id="queue_recovery_regression",
    )

    assert recovery_queue is None
    assert (
        "queue_observation_recovery_action_command_missing:"
        "inspect_artifact_postcondition_failures"
    ) in recovery_blockers


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


def test_feedback_replan_policy_routes_dry_action_functional_to_candidate_widening(
    tmp_path: Path,
) -> None:
    action_path = tmp_path / "inverse_steganalysis_action_functional.feedback.json"
    action_path.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "integral_totals": {
                    "cell_count": 1,
                    "blocked_cell_count": 1,
                    "materializer_archive_delta_blocked_cell_count": 1,
                },
                "water_bucket": {
                    "schema": "inverse_steganalysis_water_bucket_plan.v1",
                    "selected_count": 0,
                    "selected_expected_score_gain": 0.0,
                    "selected_cells": [],
                },
                "materializer_archive_delta_feedback": {
                    "schema": "inverse_steganalysis_materializer_archive_delta_feedback.v1",
                    "blocks_water_bucket": True,
                    "realized_saved_bytes_sum": -1805,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                },
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
        ),
        encoding="utf-8",
    )

    policy = build_queue_feedback_replan_policy(
        _run_summary(
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=str(action_path),
            queue_feedback_replan_request={
                "command_template": [
                    ".venv/bin/python",
                    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                    "--output",
                    str(action_path),
                    "--md-out",
                    str(action_path.with_suffix(".md")),
                    "--scorer-response",
                    "experiments/results/scorer_response.json",
                    "--inverse-scorer-max-units",
                    "16",
                    "--max-cells",
                    "128",
                    "--expected-output-sha256",
                    "0" * 64,
                ],
            },
        ),
        feedback_followup_queue=_child_queue(),
        iteration_index=1,
        max_iterations=3,
    )

    assert policy["decision"] == ACTION_WIDEN_CANDIDATE_GENERATION
    assert policy["stop_reason"] == "feedback_action_functional_dry_no_selected_cells"
    assert policy["should_continue_feedback_loop"] is False
    assert policy["ready_for_local_materialization"] is False
    assert policy["ready_for_candidate_generation_widening"] is True
    assert policy["next_iteration_command_template"] is None
    summary = policy["feedback_action_functional_summary"]
    assert summary["loaded"] is True
    assert summary["dry_no_selected_cells"] is True
    assert summary["selected_count"] == 0
    assert summary["materializer_archive_delta_blocked_cell_count"] == 1
    action = policy["recommended_actions"][-1]
    assert action["action"] == ACTION_WIDEN_CANDIDATE_GENERATION
    assert action["next_gate"] == "widen_inverse_surface_candidate_generation"
    assert "refresh_source_inverse_scorer_surface" in action["candidate_generation_hints"]
    assert action["candidate_generation_command_template"] == policy[
        "candidate_generation_command_template"
    ]
    widening = policy["candidate_widening_handoff"]
    assert widening["blockers"] == []
    assert widening["previous_inverse_scorer_max_units"] == 16
    assert widening["inverse_scorer_max_units"] == 64
    assert widening["widened_output_path"].endswith(
        "inverse_steganalysis_action_functional.feedback.widened.json"
    )
    assert "--expected-output-sha256" not in widening["command_template"]
    assert widening["command_template"][
        widening["command_template"].index("--inverse-scorer-max-units") + 1
    ] == "64"
    assert widening["source_mode"] == "existing_widenable_source"

    queue, blockers = build_queue_feedback_replan_continuation_queue(
        policy,
        lane_id="feedback_lane",
    )
    assert queue is None
    assert "queue_feedback_replan_continuation_policy_not_next_iteration" in blockers
    assert "queue_feedback_replan_continuation_policy_not_continuable" in blockers

    widening_queue, widening_blockers = build_queue_feedback_candidate_widening_queue(
        policy,
        lane_id="feedback_lane",
        source_policy_path=".omx/research/campaign/queue_feedback_replan_policy.json",
    )
    assert widening_blockers == []
    assert widening_queue is not None
    assert widening_queue["schema"] == "experiment_queue.v1"
    assert widening_queue["controls"]["mode"] == "paused"
    assert widening_queue["controls"]["max_concurrency"] == {"local_cpu": 1}
    experiment = widening_queue["experiments"][0]
    assert experiment["id"] == QUEUE_FEEDBACK_CANDIDATE_WIDENING_EXPERIMENT_ID
    assert experiment["lane_id"] == "feedback_lane"
    assert experiment["metadata"]["schema"] == QUEUE_FEEDBACK_CANDIDATE_WIDENING_METADATA_SCHEMA
    assert experiment["metadata"]["score_claim"] is False
    step = experiment["steps"][0]
    assert step["id"] == QUEUE_FEEDBACK_CANDIDATE_WIDENING_STEP_ID
    assert step["command"] == policy["candidate_generation_command_template"]
    assert step["resources"] == {"kind": "local_cpu"}
    assert step["postconditions"] == [
        {
            "type": "path_exists",
            "path": widening["widened_output_path"],
        },
        {
            "type": "json_completion_contract",
            "path": widening["widened_output_path"],
            "required_equals": {
                "schema": "inverse_steganalysis_discrete_action_functional.v1"
            },
            "required_true": [
                "planning_only",
                "candidate_generation_only",
            ],
            "required_false": list(DEFAULT_REQUIRED_FALSE_AUTHORITY_FIELDS),
            "false_or_missing": list(DEFAULT_FALSE_OR_MISSING_AUTHORITY_FIELDS),
        },
        {
            "type": "path_exists",
            "path": widening["widened_md_path"],
        },
    ]


def test_feedback_replan_policy_candidate_widening_discovers_nearby_scorer_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "campaign" / "feedback_exec"
    run_dir.mkdir(parents=True)
    action_path = run_dir / "inverse_steganalysis_action_functional.feedback.json"
    action_path.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "integral_totals": {
                    "cell_count": 1,
                    "blocked_cell_count": 1,
                    "materializer_archive_delta_blocked_cell_count": 1,
                },
                "water_bucket": {"selected_count": 0, "selected_cells": []},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    scorer_response = tmp_path / "campaign" / "scorer_response.json"
    scorer_response.write_text(
        json.dumps(
            {
                "schema": "scorer_response_dataset.v1",
                "producer": "test",
                "rows": [
                    {
                        "schema": "scorer_response_row.v1",
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    plan_path = run_dir / "byte_shaving_campaign_plan.feedback.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_campaign_plan.v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    policy = build_queue_feedback_replan_policy(
        _run_summary(
            run_dir=str(run_dir),
            plan=str(plan_path),
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=str(action_path),
            queue_feedback_replan_request={
                "command_template": [
                    ".venv/bin/python",
                    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                    "--output",
                    str(action_path),
                    "--byte-shaving-campaign-plan",
                    str(plan_path),
                ],
            },
        ),
        feedback_followup_queue=_child_queue(
            [
                sys.executable,
                QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                "--output",
                str(action_path),
                "--md-out",
                str(action_path.with_suffix(".md")),
            ]
        ),
        iteration_index=0,
        max_iterations=3,
    )

    assert policy["decision"] == ACTION_WIDEN_CANDIDATE_GENERATION
    widening = policy["candidate_widening_handoff"]
    assert widening["blockers"] == []
    assert widening["source_mode"] == "discovered_nearby_scorer_response"
    assert widening["discovered_scorer_response_paths"] == [
        "campaign/scorer_response.json"
    ]
    assert widening["discovered_scorer_response_records"] == [
        {
            "path": "campaign/scorer_response.json",
            "sha256": widening["discovered_scorer_response_records"][0]["sha256"],
            "bytes": scorer_response.stat().st_size,
            "row_count": 1,
            "producer": "test",
            "relation_seed_dir": "campaign",
            "usable": True,
            "blockers": [],
        }
    ]
    command = widening["command_template"]
    assert command[command.index("--scorer-response") + 1] == (
        "campaign/scorer_response.json"
    )


def test_feedback_replan_policy_candidate_widening_blocks_ambiguous_scorer_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "campaign" / "feedback_exec"
    run_dir.mkdir(parents=True)
    action_path = run_dir / "inverse_steganalysis_action_functional.feedback.json"
    action_path.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "integral_totals": {"cell_count": 1},
                "water_bucket": {"selected_count": 0, "selected_cells": []},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    for name in ("a_scorer_response.json", "b_scorer_response.json"):
        (tmp_path / "campaign" / name).write_text(
            json.dumps(
                {
                    "schema": "scorer_response_dataset.v1",
                    "producer": "test",
                    "rows": [
                        {
                            "schema": "scorer_response_row.v1",
                            "score_claim": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        }
                    ],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )

    policy = build_queue_feedback_replan_policy(
        _run_summary(
            run_dir=str(run_dir),
            plan=str(run_dir / "byte_shaving_campaign_plan.feedback.json"),
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=str(action_path),
            queue_feedback_replan_request={
                "command_template": [
                    ".venv/bin/python",
                    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                    "--output",
                    str(action_path),
                    "--byte-shaving-campaign-plan",
                    str(run_dir / "byte_shaving_campaign_plan.feedback.json"),
                ],
            },
        ),
        feedback_followup_queue=_child_queue(
            [
                sys.executable,
                QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                "--output",
                str(action_path),
                "--md-out",
                str(action_path.with_suffix(".md")),
            ]
        ),
    )

    assert policy["decision"] == ACTION_WIDEN_CANDIDATE_GENERATION
    assert policy["candidate_generation_command_template"] is None
    blockers = policy["candidate_generation_queue_blockers"]
    assert "candidate_widening_no_widenable_source_surface" in blockers
    assert any(
        item.startswith(
            "candidate_widening_ambiguous_nearby_scorer_response_sources:"
        )
        for item in blockers
    )


def test_feedback_replan_policy_candidate_widening_blocks_no_widenable_source(
    tmp_path: Path,
) -> None:
    action_path = tmp_path / "inverse_steganalysis_action_functional.feedback.json"
    action_path.write_text(
        json.dumps(
            {
                "schema": "inverse_steganalysis_discrete_action_functional.v1",
                "integral_totals": {"cell_count": 1},
                "water_bucket": {"selected_count": 0, "selected_cells": []},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    policy = build_queue_feedback_replan_policy(
        _run_summary(
            run_dir=str(tmp_path),
            queue_feedback_replan_followup_executed=True,
            queue_feedback_replan_followup_execution_success=True,
            queue_feedback_replan_followup_action_functional_path=str(action_path),
            queue_feedback_replan_request={
                "command_template": [
                    ".venv/bin/python",
                    QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                    "--output",
                    str(action_path),
                    "--byte-shaving-campaign-plan",
                    str(tmp_path / "plan.json"),
                ],
            },
        ),
        feedback_followup_queue=_child_queue(
            [
                sys.executable,
                QUEUE_FEEDBACK_REPLAN_ACTION_FUNCTIONAL_TOOL,
                "--output",
                str(action_path),
                "--md-out",
                str(action_path.with_suffix(".md")),
            ]
        ),
    )

    assert policy["decision"] == ACTION_WIDEN_CANDIDATE_GENERATION
    assert policy["candidate_generation_command_template"] is None
    assert "candidate_widening_no_widenable_source_surface" in policy[
        "candidate_generation_queue_blockers"
    ]
    queue, blockers = build_queue_feedback_candidate_widening_queue(
        policy,
        lane_id="feedback_lane",
    )
    assert queue is None
    assert (
        "queue_feedback_candidate_widening_handoff:"
        "candidate_widening_no_widenable_source_surface"
    ) in blockers


def test_feedback_candidate_actuation_planning_queue_compiles_widened_cells() -> None:
    policy = {
        "schema": QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA,
        "decision": ACTION_WIDEN_CANDIDATE_GENERATION,
        "queue_id": "campaign_queue",
        "source_run_path": ".omx/research/campaign/materializer_campaign_run.json",
        "plan_path": ".omx/research/campaign/byte_shaving_campaign_plan.feedback.json",
        "feedback_action_functional_path": (
            ".omx/research/campaign/inverse_steganalysis_action_functional.feedback.json"
        ),
        "candidate_widening_handoff": {
            "blockers": [],
            "widened_output_path": (
                ".omx/research/campaign/"
                "inverse_steganalysis_action_functional.feedback.widened.json"
            ),
            "widened_md_path": (
                ".omx/research/campaign/"
                "inverse_steganalysis_action_functional.feedback.widened.md"
            ),
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    queue, blockers = build_queue_feedback_candidate_actuation_planning_queue(
        policy,
        lane_id="feedback_lane",
        source_policy_path=".omx/research/campaign/queue_feedback_replan_policy.json",
    )

    assert blockers == []
    assert queue is not None
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["mode"] == "paused"
    assert queue["controls"]["max_concurrency"] == {"local_cpu": 1}
    experiment = queue["experiments"][0]
    assert (
        experiment["id"]
        == QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLANNING_EXPERIMENT_ID
    )
    assert experiment["lane_id"] == "feedback_lane"
    assert experiment["metadata"]["schema"] == (
        "queue_feedback_candidate_actuation_planning_metadata.v1"
    )
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    plan_step, compile_step = experiment["steps"]
    assert plan_step["id"] == QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID
    assert compile_step["id"] == QUEUE_FEEDBACK_CANDIDATE_ACTUATION_QUEUE_STEP_ID
    assert compile_step["requires"] == [
        QUEUE_FEEDBACK_CANDIDATE_ACTUATION_PLAN_STEP_ID
    ]
    assert plan_step["command"][:4] == [
        ".venv/bin/python",
        "tools/plan_byte_shaving_campaign.py",
        "--source",
        (
            ".omx/research/campaign/"
            "inverse_steganalysis_action_functional.feedback.widened.json"
        ),
    ]
    assert "--from-inverse-action-functional" in plan_step["command"]
    assert "--inverse-action-materialization-bridge-out" in plan_step["command"]
    assert compile_step["command"][:2] == [
        ".venv/bin/python",
        "tools/build_byte_shaving_campaign_queue.py",
    ]
    assert "--materializer-contexts-out" in compile_step["command"]
    assert "--materializer-context-default-output-root" in compile_step["command"]
    assert "--materializer-work-queue-out" in compile_step["command"]
    assert "--materializer-execution-queue-out" not in compile_step["command"]
    assert plan_step["telemetry"]["pullback_artifact_paths"] == [
        ".omx/research/campaign/byte_shaving_campaign_plan.feedback.widened.json",
        ".omx/research/campaign/byte_shaving_campaign_plan.feedback.widened.md",
        (
            ".omx/research/campaign/"
            "inverse_action_materialization_bridge.feedback.widened.json"
        ),
    ]
    assert (
        ".omx/research/campaign/materializer_contexts.feedback.widened.json"
        in compile_step["telemetry"]["pullback_artifact_paths"]
    )
    assert (
        ".omx/research/campaign/materializer_work_queue.feedback.widened.json"
        in compile_step["telemetry"]["pullback_artifact_paths"]
    )
    assert experiment["metadata"]["materializer_contexts_path"].endswith(
        "materializer_contexts.feedback.widened.json"
    )
    assert experiment["metadata"]["materializer_context_default_output_root"].endswith(
        "materializer_outputs.feedback.widened"
    )
    assert any(
        condition.get("path", "").endswith(
            "inverse_action_materialization_bridge.feedback.widened.json"
        )
        for condition in plan_step["postconditions"]
    )
    plan_contracts = [
        condition
        for condition in plan_step["postconditions"]
        if condition.get("type") == "json_completion_contract"
    ]
    assert any(
        "inverse_action_materialization_portfolios"
        in condition.get("required_nonempty", [])
        for condition in plan_contracts
    )
    assert any(
        condition.get("path", "").endswith("materializer_work_queue.feedback.widened.json")
        for condition in compile_step["postconditions"]
    )
    backlog_contracts = [
        condition
        for condition in compile_step["postconditions"]
        if condition.get("path", "").endswith("materializer_backlog.feedback.widened.json")
    ]
    assert any(
        "backlog_row_count" in condition.get("required_positive_int", [])
        for condition in backlog_contracts
    )
    context_contracts = [
        condition
        for condition in compile_step["postconditions"]
        if condition.get("path", "").endswith(
            "materializer_contexts.feedback.widened.json"
        )
    ]
    assert any(
        "row_count" in condition.get("required_positive_int", [])
        for condition in context_contracts
    )


def test_feedback_candidate_actuation_planning_queue_blocks_handoff_refusal() -> None:
    policy = {
        "schema": QUEUE_FEEDBACK_REPLAN_POLICY_SCHEMA,
        "decision": ACTION_WIDEN_CANDIDATE_GENERATION,
        "candidate_widening_handoff": {
            "blockers": ["candidate_widening_no_widenable_source_surface"],
            "widened_output_path": None,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    queue, blockers = build_queue_feedback_candidate_actuation_planning_queue(
        policy,
        lane_id="feedback_lane",
    )

    assert queue is None
    assert "queue_feedback_candidate_actuation_widened_output_missing" in blockers
    assert (
        "queue_feedback_candidate_actuation_handoff:"
        "candidate_widening_no_widenable_source_surface"
    ) in blockers


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
