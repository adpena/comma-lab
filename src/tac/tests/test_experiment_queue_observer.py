# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
)
from comma_lab.scheduler.experiment_queue_observer import (
    observe_experiment_queue,
    render_observation_markdown,
)
from comma_lab.scheduler.queue_feedback_replan_policy import (
    build_queue_observation_recovery_plan,
)


def _queue(artifact_path: Path) -> dict[str, object]:
    return {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {
            "mode": "running",
            "max_concurrency": {"local_cpu": 1, "local_mlx": 1},
        },
        "experiments": [
            {
                "id": "exp0",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "smoke",
                        "kind": "command",
                        "command": ["python", "-c", "print('hello queue')"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": artifact_path.as_posix(),
                                "key": "schema",
                                "equals": "artifact.v1",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_observer_surfaces_running_step_log_tail_and_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "artifact.v1",
                "canonical_score": 0.5,
                "serialized_archive_delta": {
                    "status": "realized_cost",
                    "realized_saved_bytes": -1764,
                    "savings_realized": False,
                    "source_archive_bytes": 178_592,
                    "candidate_archive_bytes": 180_356,
                },
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "logs" / "smoke.log"
    log_path.parent.mkdir()
    log_path.write_text("first\nsecond\n", encoding="utf-8")
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (
                json.dumps(
                    {
                        "command": ["python", "-c", "print('hello queue')"],
                        "log_path": str(log_path),
                        "pid": 12345,
                        "worker_run_id": "workerabc",
                        "timeout_seconds": 60,
                        "timeout_deadline_epoch_seconds": 1_779_494_400.0,
                    }
                ),
            ),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    running = observation["running_steps"][0]

    assert observation["schema"] == "experiment_queue_observation.v1"
    assert observation["healthy"] is True
    assert observation["blockers"] == []
    assert observation["blocker_count"] == 0
    assert observation["auto_parallelism"]["local_only"] == {
        "max_parallel": 1,
        "resource_limits": {"local_cpu": 1},
    }
    assert observation["auto_parallelism"]["idle_declared_resources"] == {"local_mlx": 1}
    assert running["pid"] == 12345
    assert running["worker_run_id"] == "workerabc"
    assert running["timeout_seconds"] == 60
    assert running["timeout_deadline_epoch_seconds"] == 1_779_494_400.0
    assert running["log_tail"] == ["second"]
    assert running["expected_artifacts"][0]["exists"] is True
    assert running["expected_artifacts"][0]["json_schema"] == "artifact.v1"
    assert running["expected_artifacts"][0]["serialized_archive_delta_status"] == "realized_cost"
    assert running["expected_artifacts"][0]["serialized_archive_delta_realized_saved_bytes"] == -1764
    assert running["expected_artifacts"][0]["serialized_archive_delta_savings_realized"] is False
    assert running["expected_artifacts"][0]["postcondition_passed"] is True
    markdown = render_observation_markdown(observation)
    assert "observer_test" in markdown
    assert "local_mlx" in markdown
    assert "smoke" in markdown
    assert "healthy" in markdown


def test_observer_marks_existing_artifact_failed_when_postcondition_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"schema": "wrong.v1", "canonical_score": 0.5}),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    artifact_record = observation["running_steps"][0]["expected_artifacts"][0]

    assert artifact_record["exists"] is True
    assert artifact_record["json_schema"] == "wrong.v1"
    assert artifact_record["postcondition_passed"] is False
    assert observation["healthy"] is False
    assert observation["blockers"] == ["experiment_queue_observation_artifact_postcondition_failures:1"]
    assert observation["blocker_count"] == 1
    markdown = render_observation_markdown(observation)
    assert "0/1" in markdown


def test_observer_reports_definition_drift_without_mutating_state(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        before = conn.execute(
            """
            SELECT status, command_hash
            FROM step_state
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """
        ).fetchone()

    changed_queue = _queue(artifact)
    changed_queue["experiments"][0]["steps"][0]["command"] = [
        "python",
        "-c",
        "print('changed')",
    ]

    observation = observe_experiment_queue(
        changed_queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["observe_read_only"] is True
    assert observation["definition_drift"]["changed_step_count"] == 1
    assert observation["healthy"] is False
    assert "experiment_queue_observation_changed_steps:1" in observation["blockers"]
    with connect_state(state) as conn:
        after = conn.execute(
            """
            SELECT status, command_hash
            FROM step_state
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """
        ).fetchone()
    assert after["status"] == before["status"]
    assert after["command_hash"] == before["command_hash"]


def test_observer_surfaces_read_only_performance_telemetry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            INSERT INTO queue_events(
                ts_utc, queue_id, experiment_id, step_id, event_type, payload_json
            )
            VALUES (
                '2026-05-23T00:00:00Z',
                'observer_test',
                'exp0',
                'smoke',
                'step_succeeded',
                ?
            )
            """,
            (
                json.dumps(
                    {
                        "resource_kind": "local_cpu",
                        "elapsed_seconds": 2.5,
                        "telemetry": {
                            "artifact_records": [
                                {
                                    "path": "artifact.json",
                                    "bytes": 42,
                                }
                            ],
                            "log_bytes": 12,
                        },
                    }
                ),
            ),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    performance = observation["performance"]
    runtime_policy = observation["runtime_policy"]

    assert performance["schema"] == "experiment_queue_performance_summary.v1"
    assert performance["telemetry_only"] is True
    assert performance["score_claim"] is False
    assert performance["by_resource_kind"]["local_cpu"]["elapsed_seconds_mean"] == 2.5
    assert performance["by_resource_kind"]["local_cpu"]["dominant_resource_kind"] == "local_cpu"
    assert runtime_policy["schema"] == "scheduler_runtime_policy.v1"
    assert runtime_policy["advisory_only"] is True
    assert runtime_policy["score_claim"] is False
    assert runtime_policy["recommended_max_concurrency"]["local_cpu"] >= 1
    assert runtime_policy["recommended_timeout_seconds_by_resource"]["local_cpu"] >= 30
    markdown = render_observation_markdown(observation)
    assert "performance" in markdown
    assert "runtime_policy" in markdown
    assert "telemetry_only" in markdown
    assert "score_claim" in markdown


def test_observer_health_marks_orphaned_steps_with_details_when_requested(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)

    changed_queue = _queue(artifact)
    changed_queue["experiments"][0]["steps"] = []

    observation = observe_experiment_queue(
        changed_queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
        include_orphans=True,
    )

    assert observation["healthy"] is False
    assert observation["orphaned_step_count"] == 1
    assert "experiment_queue_observation_orphaned_steps:1" in observation["blockers"]
    assert observation["orphaned_steps"][0]["experiment_id"] == "exp0"
    assert observation["orphaned_steps"][0]["step_id"] == "smoke"
    assert observation["orphaned_steps"][0]["expected_artifacts"] == []


def test_observer_health_marks_failed_steps(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["healthy"] is False
    assert "experiment_queue_observation_failed_steps:1" in observation["blockers"]
    assert observation["failed_steps"][0]["step_id"] == "smoke"


def test_observer_preserves_materializer_metadata_for_recovery_grouping(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "archive_section_entropy_recode_candidate.v1",
                "target_kind": "archive_section_entropy_recode_v1",
                "materializer_id": "entropy_adapter",
                "receiver_contract_kind": "archive_section_receiver_v1",
                "receiver_contract_satisfied": False,
                "readiness_blockers": [
                    "runtime_consumption_proof_not_passed",
                    "archive_section_entropy_recode_receiver_contract_not_satisfied",
                ],
                "receiver_verification": {
                    "schema": "receiver_verification.v1",
                    "receiver_contract_satisfied": False,
                    "blockers": ["runtime_consumption_proof_not_passed"],
                },
                "candidate_archive": {
                    "path": str(tmp_path / "candidate.zip"),
                    "bytes": 143,
                    "sha256": "a" * 64,
                },
                "section_recode": {
                    "source_archive_bytes": 209,
                    "candidate_archive_bytes": 143,
                    "saved_bytes": 66,
                },
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue_path = tmp_path / "queue.json"
    queue = _queue(artifact)
    queue["experiments"][0]["metadata"] = {
        "schema": "byte_shaving_materializer_execution_experiment_metadata.v1",
        "target_kind": "archive_section_entropy_recode_v1",
        "materializer_id": "entropy_adapter",
        "receiver_contract_kind": "archive_section_receiver_v1",
        "work_id": "work-17",
        "backlog_key": "backlog-3",
        "source_unit_ids": ["unit-a"],
        "source_selection_ids": ["selection-a"],
        "candidate_saved_bytes_sum": 19,
        "expected_score_gain_sum": 0.25,
    }
    queue["experiments"][0]["steps"][0]["telemetry"] = {
        "artifact_paths": [artifact.as_posix()],
    }
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T03:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    failed = observation["failed_steps"][0]

    assert len(observation["queue_sha256"]) == 64
    assert observation["state_watermark"]["schema"] == ("experiment_queue_state_watermark.v1")
    assert failed["target_kind"] == "archive_section_entropy_recode_v1"
    assert failed["materializer_id"] == "entropy_adapter"
    assert failed["receiver_contract_kind"] == "archive_section_receiver_v1"
    assert failed["work_ids"] == ["work-17"]
    assert failed["backlog_keys"] == ["backlog-3"]
    assert failed["source_unit_ids"] == ["unit-a"]
    assert failed["source_selection_ids"] == ["selection-a"]
    assert failed["expected_artifact_paths"] == ["candidate.json"]
    artifact_record = failed["expected_artifacts"][0]
    assert artifact_record["json_schema"] == "archive_section_entropy_recode_candidate.v1"
    assert artifact_record["receiver_contract_satisfied"] is False
    assert artifact_record["readiness_blockers"] == [
        "runtime_consumption_proof_not_passed",
        "archive_section_entropy_recode_receiver_contract_not_satisfied",
    ]
    assert artifact_record["receiver_verification"]["blockers"] == ["runtime_consumption_proof_not_passed"]
    assert artifact_record["candidate_archive"]["bytes"] == 143
    assert artifact_record["section_recode_saved_bytes"] == 66
    assert artifact_record["serialized_archive_delta_status"] == "realized_saving"
    assert artifact_record["serialized_archive_delta_realized_saved_bytes"] == 66
    assert artifact_record["serialized_archive_delta_savings_realized"] is True

    recovery_plan = build_queue_observation_recovery_plan(
        observation,
        queue_path=str(queue_path),
        state_path=str(state),
    )
    groups = recovery_plan["grouped_blockers"]

    assert recovery_plan["recovery_required"] is False
    assert recovery_plan["maintenance_recommended"] is True
    assert recovery_plan["actions"][0]["action"] == ("record_materializer_receiver_feedback")
    assert recovery_plan["source_queue_sha256"] == observation["queue_sha256"]
    assert recovery_plan["source_state_watermark"] == observation["state_watermark"]
    assert groups[0]["scope_kind"] == "materializer_receiver"
    assert groups[0]["scope_value"] == ("entropy_adapter:archive_section_receiver_v1")
    assert groups[0]["target_kinds"] == ["archive_section_entropy_recode_v1"]
    assert groups[0]["source_selection_ids"] == ["selection-a"]


def test_observer_derives_generic_materializer_archive_delta(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "packet_recompress.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "packet_member_recompress_candidate.v1",
                "target_kind": "packet_member_recompress_v1",
                "materializer_id": "packet_member_recompress_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_recompress",
                "receiver_contract_satisfied": True,
                "source_archive": {"bytes": 500, "sha256": "a" * 64},
                "candidate_archive": {
                    "path": str(tmp_path / "candidate.zip"),
                    "bytes": 425,
                    "sha256": "b" * 64,
                },
                "selected_compression": {
                    "compression_method": "deflated",
                    "compresslevel": 9,
                    "source_archive_bytes": 500,
                    "candidate_archive_bytes": 425,
                    "saved_bytes": 75,
                },
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    queue["experiments"][0]["steps"][0]["telemetry"] = {
        "artifact_paths": [artifact.as_posix()],
    }

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T10:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )
    artifact_record = observation["failed_steps"][0]["expected_artifacts"][0]

    assert artifact_record["materializer_delta_source"] == "selected_compression"
    assert artifact_record["selected_compression_saved_bytes"] == 75
    assert artifact_record["selected_compression_source_archive_bytes"] == 500
    assert artifact_record["selected_compression_candidate_archive_bytes"] == 425
    assert artifact_record["serialized_archive_delta_status"] == "realized_saving"
    assert artifact_record["serialized_archive_delta_realized_saved_bytes"] == 75
    assert artifact_record["serialized_archive_delta_savings_realized"] is True
    assert artifact_record["serialized_archive_delta_source_archive_bytes"] == 500
    assert artifact_record["serialized_archive_delta_candidate_archive_bytes"] == 425


def test_observer_health_marks_blocked_steps(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'blocked',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (
                json.dumps(
                    {
                        "reason": "fixture_dependency_missing",
                        "command": ["python", "-c", "print('hello queue')"],
                    }
                ),
            ),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["healthy"] is False
    assert "experiment_queue_observation_blocked_steps:1" in observation["blockers"]
    assert observation["blocked_steps"][0]["step_id"] == "smoke"


def test_observer_health_marks_succeeded_step_with_missing_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-23T00:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    failed_step = observation["succeeded_artifact_failure_steps"][0]
    failed_artifact = failed_step["expected_artifacts"][0]
    assert observation["healthy"] is False
    assert "experiment_queue_observation_artifact_postcondition_failures:1" in (observation["blockers"])
    assert failed_step["step_id"] == "smoke"
    assert failed_artifact["exists"] is False
    assert failed_artifact["postcondition_passed"] is False


def test_observer_surfaces_succeeded_materializer_feedback_artifact(
    tmp_path: Path,
) -> None:
    observation_jsonl = tmp_path / "observations.jsonl"
    observation_jsonl.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_observation.v1",
                "observation_id": "obs_packet_recompress_success",
                "candidate_id": "candidate_packet",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "materializer_sweep",
                "status": "queued",
                "priority": 1,
                "metadata": {
                    "candidate_ids": ["candidate_packet"],
                    "source_unit_ids": ["packet_payload_bin"],
                    "source_selection_ids": ["selection_packet_payload_bin"],
                },
                "steps": [
                    {
                        "id": "sweep",
                        "kind": "command",
                        "command": ["python", "tools/run_family_agnostic_materializer_sweep.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "jsonl_false_authority",
                                "path": observation_jsonl.as_posix(),
                                "schema_equals": ("family_agnostic_materializer_empirical_observation.v1"),
                                "require_nonempty": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T06:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'materializer_sweep'
              AND step_id = 'sweep'
            """,
            (json.dumps({"command": ["python", "tools/run_family_agnostic_materializer_sweep.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["healthy"] is True
    assert observation["succeeded_artifact_failure_steps"] == []
    succeeded = observation["succeeded_artifact_steps"][0]
    artifact = succeeded["expected_artifacts"][0]
    assert succeeded["candidate_ids"] == ["candidate_packet"]
    assert succeeded["source_unit_ids"] == ["packet_payload_bin"]
    assert artifact["path"] == "observations.jsonl"
    assert artifact["postcondition_type"] == "jsonl_false_authority"
    assert artifact["postcondition_schema_equals"] == (
        "family_agnostic_materializer_empirical_observation.v1"
    )
    assert artifact["postcondition_passed"] is True


def test_observer_health_marks_missing_state_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    state = tmp_path / "missing.sqlite"
    queue = _queue(artifact)

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["healthy"] is False
    assert observation["observe_read_only"] is True
    assert observation["blocker_count"] >= 1
    assert "experiment_queue_observation_state_missing" in observation["blockers"]
    assert "experiment_queue_observation_missing_steps:1" in observation["blockers"]
