# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
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
from tac.repo_io import tree_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_artifact_bytes(path: Path, data: bytes) -> dict[str, object]:
    path.write_bytes(data)
    return {
        "path": path.as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_receiver_proof(
    path: Path,
    *,
    candidate_archive: dict[str, object],
) -> dict[str, object]:
    payload = {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "candidate_archive": dict(candidate_archive),
        "candidate_archive_sha256": candidate_archive["sha256"],
        "receiver_contract_satisfied": True,
        "runtime_consumption_proof_passed": True,
        "passed": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "blockers": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
        "receiver_contract_satisfied": True,
        "runtime_adapter_ready": False,
        "proof_present": True,
        "proof_path": path.as_posix(),
        "blockers": [],
    }


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


def test_observer_surfaces_frozen_definition_status(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    queue = _queue(artifact)
    queue["experiments"][0]["status"] = "frozen"
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["healthy"] is True
    assert observation["status_counts"] == {"frozen": 1}
    assert observation["queued_steps"] == []
    assert len(observation["frozen_steps"]) == 1
    assert observation["frozen_steps"][0]["status"] == "frozen"
    assert observation["frozen_steps"][0]["stored_status"] == "queued"
    assert observation["ready_steps"] == []


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


def test_observer_revalidates_required_runtime_identity_claim(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runtime_contract.json"
    runtime_dir = tmp_path / "candidate_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    artifact.write_text(
        json.dumps(
            {
                "schema": "runtime_contract.v1",
                "runtime_dir": runtime_dir.as_posix(),
                "runtime_tree_sha256": tree_sha256(runtime_dir),
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_completion_contract",
            "path": artifact.as_posix(),
            "required_runtime_adapter_identity": True,
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T18:45:00Z'
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
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact_record = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert artifact_record["postcondition_passed"] is False
    assert (
        "json_completion_contract_runtime_identity_runtime_adapter_identity_claim_missing"
        in artifact_record["artifact_revalidation_blockers"]
    )
    assert artifact_record["artifact_revalidation"]["runtime_identity"]["valid"] is False


def test_observer_rejects_completion_contract_without_custody_proof(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "completion.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "completion_report.v1",
                "status": "succeeded",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_completion_contract",
            "path": artifact.as_posix(),
            "required_equals": {"schema": "completion_report.v1"},
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T21:15:00Z'
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
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact_record = observation["succeeded_artifact_failure_steps"][0][
        "expected_artifacts"
    ][0]
    assert artifact_record["postcondition_passed"] is False
    assert "json_completion_contract_candidate_archive_custody_missing" in (
        artifact_record["artifact_revalidation_blockers"]
    )
    required_custody = artifact_record["artifact_revalidation"][
        "required_archive_runtime_receiver_custody"
    ]
    assert required_custody["valid"] is False


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


def test_observer_surfaces_pr95_mlx_package_report_readiness(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "package_report.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "pr95_mlx_pytorch_state_dict_to_contest_archive.v1",
                "archive_zip_sha256": "a" * 64,
                "archive_zip_bytes": 230_345,
                "archive_member_sha256": "b" * 64,
                "archive_member_bytes": 230_237,
                "archive_manifest_path": "submission_dir/archive_manifest.json",
                "input_pt_sha256": "c" * 64,
                "source_archive_zip_sha256": "d" * 64,
                "runtime_files_emitted": {
                    "inflate.sh": "e" * 64,
                    "inflate.py": "f" * 64,
                    "vendored_src_model.py": "1" * 64,
                    "vendored_src_codec.py": "2" * 64,
                },
                "exact_readiness_refusal": {
                    "schema": "exact_readiness_refusal.v1",
                    "ready": False,
                    "blockers": [
                        "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
                        "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
                    ],
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_packet_ready": False,
                "promotable": False,
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
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T18:00:00Z'
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

    assert artifact_record["json_schema"] == ("pr95_mlx_pytorch_state_dict_to_contest_archive.v1")
    assert artifact_record["pr95_mlx_package_report"] is True
    assert artifact_record["archive_sha256"] == "a" * 64
    assert artifact_record["archive_bytes"] == 230_345
    assert artifact_record["archive_member_sha256"] == "b" * 64
    assert artifact_record["archive_manifest_path"] == ("submission_dir/archive_manifest.json")
    assert artifact_record["runtime_file_count"] == 4
    assert artifact_record["score_claim"] is False
    assert artifact_record["ready_for_exact_eval_dispatch"] is False
    assert artifact_record["dispatch_packet_ready"] is False
    assert artifact_record["exact_readiness_refusal"]["ready"] is False
    assert artifact_record["readiness_blockers"] == [
        "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
        "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
    ]


def test_observer_surfaces_pr95_mlx_long_training_plan(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "long_training_plan.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "pr95_mlx_long_training_plan.v1",
                "mode": "plan_only",
                "lane_id": "lane_pr95_mlx_long_training_test",
                "source_video_sha256": "a" * 64,
                "source_video_frame_count": 1200,
                "source_video_frame_count_scope": "full_video_decode",
                "max_frames": None,
                "checkpoint_root": "experiments/results/pr95_mlx_long_training",
                "telemetry_path": ".omx/state/pr95_mlx_long_training.jsonl",
                "candidate_registry_count": 6,
                "total_epochs": 3000,
                "smoke_mode": True,
                "training_fidelity_class": "rgb_frame_mse_local_mlx_research_mvp",
                "training_fidelity_status": ("local_rgb_frame_mse_mvp_not_segnet_posenet_scorer_faithful"),
                "reproduction_equivalence": False,
                "reproduction_claim": False,
                "pr95_1to1_reproduction_claim": False,
                "reproduction_equivalence_class": "not_pr95_1to1_rgb_mse_mvp",
                "exact_readiness_refusal": {
                    "schema": "exact_readiness_refusal.v1",
                    "ready": False,
                    "blockers": [
                        "local_mlx_long_training_is_research_signal_not_contest_auth_eval",
                        "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
                    ],
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
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
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T18:00:00Z'
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

    assert artifact_record["json_schema"] == "pr95_mlx_long_training_plan.v1"
    assert artifact_record["pr95_mlx_long_training_plan"] is True
    assert artifact_record["mode"] == "plan_only"
    assert artifact_record["lane_id"] == "lane_pr95_mlx_long_training_test"
    assert artifact_record["source_video_frame_count_scope"] == "full_video_decode"
    assert artifact_record["max_frames"] is None
    assert artifact_record["training_fidelity_class"] == "rgb_frame_mse_local_mlx_research_mvp"
    assert artifact_record["candidate_registry_count"] == 6
    assert artifact_record["reproduction_equivalence"] is False
    assert artifact_record["reproduction_claim"] is False
    assert artifact_record["pr95_1to1_reproduction_claim"] is False
    assert artifact_record["reproduction_equivalence_class"] == ("not_pr95_1to1_rgb_mse_mvp")
    assert artifact_record["ready_for_exact_eval_dispatch"] is False
    assert artifact_record["exact_readiness_refusal"]["ready"] is False
    assert artifact_record["readiness_blockers"] == [
        "local_mlx_long_training_is_research_signal_not_contest_auth_eval",
        "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
    ]


def test_observer_surfaces_hinton_mlx_long_training_smoke(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "hinton_smoke.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "hinton_mlx_long_training_smoke_verdict.v1",
                "mode": "executed_smoke",
                "lane_id": "lane_hinton_mlx_smoke_test",
                "operator_run_label": "hinton_kl_t2_smoke",
                "source_video_sha256": "a" * 64,
                "source_video_frame_count": 4,
                "max_frames": 4,
                "smoke_epochs": 100,
                "distillation_temperature": 2.0,
                "distillation_weight": 0.5,
                "num_classes": 5,
                "spatial_downsample_factor": 4,
                "local_training_queue_signal": "LOCAL_MLX_QUEUE_READY",
                "paid_dispatch_authorization_signal": (
                    "PAID_DISPATCH_BLOCKED_REQUIRES_CONTEST_TEACHER_AND_CPU_CUDA_AUTH_EVAL"
                ),
                "convergence_verdict": {
                    "verdict": "CONVERGES_CONSISTENTLY",
                    "initial_loss": 0.2,
                    "final_loss": 0.01,
                    "loss_reduction_percent": 95.0,
                    "oscillation_score": 0.0,
                    "smoke_epochs": 100,
                },
                "readiness_blockers": [
                    "scorer_loss_is_kl_to_mock_teacher_logits_not_contest_segnet",
                    "no_paired_cpu_cuda_auth_eval_yet",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    queue["experiments"][0]["steps"][0]["postconditions"][0]["equals"] = "hinton_mlx_long_training_smoke_verdict.v1"
    queue["experiments"][0]["steps"][0]["telemetry"] = {
        "artifact_paths": [artifact.as_posix()],
    }

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T22:00:00Z'
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

    assert artifact_record["json_schema"] == "hinton_mlx_long_training_smoke_verdict.v1"
    assert artifact_record["hinton_mlx_long_training_smoke"] is True
    assert artifact_record["local_training_queue_signal"] == "LOCAL_MLX_QUEUE_READY"
    assert artifact_record["paid_dispatch_authorization_signal"].startswith("PAID_DISPATCH_BLOCKED")
    assert artifact_record["convergence_verdict"]["verdict"] == ("CONVERGES_CONSISTENTLY")
    assert artifact_record["ready_for_exact_eval_dispatch"] is False
    assert artifact_record["readiness_blockers"] == [
        "scorer_loss_is_kl_to_mock_teacher_logits_not_contest_segnet",
        "no_paired_cpu_cuda_auth_eval_yet",
    ]

    with connect_state(state) as conn:
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T22:10:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "-c", "print('hello queue')"]}),),
        )
        conn.commit()

    succeeded_observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert len(succeeded_observation["succeeded_signal_steps"]) == 1
    assert succeeded_observation["local_training_signal_observation_count"] == 1
    signal = succeeded_observation["local_training_signal_observations"][0]
    assert signal["schema"] == "local_training_signal_observation.v1"
    assert signal["source_schema"] == "hinton_mlx_long_training_smoke_verdict.v1"
    assert signal["local_training_queue_signal"] == "LOCAL_MLX_QUEUE_READY"
    assert signal["recommended_next_action"] == ("build_contest_teacher_or_strict_surrogate_queue_before_paid_dispatch")
    assert signal["score_claim"] is False
    assert signal["promotion_eligible"] is False
    assert signal["ready_for_exact_eval_dispatch"] is False


def test_observer_refuses_local_training_signal_without_readiness_blockers(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "hinton_smoke_no_blockers.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "hinton_mlx_long_training_smoke_verdict.v1",
                "mode": "executed_smoke",
                "lane_id": "lane_hinton_mlx_smoke_test",
                "operator_run_label": "hinton_kl_t2_smoke",
                "local_training_queue_signal": "LOCAL_MLX_QUEUE_READY",
                "paid_dispatch_authorization_signal": (
                    "PAID_DISPATCH_BLOCKED_REQUIRES_CONTEST_TEACHER_AND_CPU_CUDA_AUTH_EVAL"
                ),
                "convergence_verdict": {
                    "verdict": "CONVERGES_CONSISTENTLY",
                    "initial_loss": 0.2,
                    "final_loss": 0.01,
                    "loss_reduction_percent": 95.0,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "promotable": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_false_authority",
            "path": artifact.as_posix(),
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
            ],
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T14:30:00Z'
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
        tail_lines=0,
    )

    assert observation["healthy"] is False
    assert observation["succeeded_signal_steps"] == []
    assert observation["local_training_signal_observation_count"] == 0
    artifact_record = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert any(
        "requires_readiness_blockers_or_exact_readiness_refusal" in blocker
        for blocker in artifact_record["artifact_revalidation_blockers"]
    )


def test_observer_real_pr95_queue_owns_drift_trace_and_package_artifacts(
    tmp_path: Path,
) -> None:
    queue_path = (
        REPO_ROOT
        / ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z"
        / "pr95_mlx_full_profile_queue.json"
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    stage8 = queue["experiments"][-1]
    step = stage8["steps"][0]
    artifact_paths = set(step["telemetry"]["artifact_paths"])
    expected = {
        ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/"
        "matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/"
        "pr95_mlx_pytorch_per_op_drift.json",
        ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/"
        "matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/"
        "pytorch_export_forward_parity_with_decoder_trace.json",
        ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/"
        "matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/"
        "pytorch_mlx_decoder_trace.json",
        ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/"
        "matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/"
        "pr95_packaged_submission_report.json",
    }
    assert expected <= artifact_paths

    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-25T18:00:00Z'
            WHERE queue_id = ?
              AND experiment_id = ?
              AND step_id = ?
            """,
            (
                json.dumps({"command": step["command"]}),
                queue["queue_id"],
                stage8["id"],
                step["id"],
            ),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        tail_lines=1,
    )
    records = {record["path"]: record for record in observation["running_steps"][0]["expected_artifacts"]}

    package = records[
        ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/"
        "matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/"
        "pr95_packaged_submission_report.json"
    ]
    trace = records[
        ".omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/"
        "matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/"
        "pytorch_mlx_decoder_trace.json"
    ]
    assert package["pr95_mlx_package_report"] is True
    assert package["ready_for_exact_eval_dispatch"] is False
    assert trace["json_schema"] == "pr95_hnerv_public_archive_mlx_decoder_trace.v1"
    assert trace["ready_for_exact_eval_dispatch"] is False


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
    assert artifact["postcondition_schema_equals"] == ("family_agnostic_materializer_empirical_observation.v1")
    assert artifact["postcondition_passed"] is True


def test_observer_rejects_jsonl_custody_row_with_only_proof_present_flag(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "jsonl_custody_candidate.zip",
        b"jsonl custody candidate bytes",
    )
    observation_jsonl = tmp_path / "custody_observations.jsonl"
    observation_jsonl.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody_observation.v1",
                "candidate_id": "jsonl_custody_candidate",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "receiver_verification": {
                    "schema": (
                        "family_agnostic_runtime_consumption_proof_verification.v1"
                    ),
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                    "blockers": [],
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
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
                "id": "jsonl_custody_experiment",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "observe_custody_jsonl",
                        "kind": "command",
                        "command": ["python", "emit_custody_jsonl.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "jsonl_false_authority",
                                "path": observation_jsonl.as_posix(),
                                "schema_equals": (
                                    "generic_candidate_custody_observation.v1"
                                ),
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
                updated_at_utc = '2026-05-27T18:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'jsonl_custody_experiment'
              AND step_id = 'observe_custody_jsonl'
            """,
            (json.dumps({"command": ["python", "emit_custody_jsonl.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    failed_step = observation["succeeded_artifact_failure_steps"][0]
    artifact = failed_step["expected_artifacts"][0]
    assert artifact["postcondition_type"] == "jsonl_false_authority"
    assert artifact["postcondition_passed"] is False
    assert (
        "row_1:jsonl_custody_runtime_or_receiver_proof_path_missing"
        in artifact["artifact_revalidation_blockers"]
    )
    assert "experiment_queue_observation_artifact_postcondition_failures:1" in (
        observation["blockers"]
    )


def test_observer_accepts_jsonl_custody_row_with_live_proof(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "jsonl_custody_candidate.zip",
        b"jsonl custody candidate bytes",
    )
    receiver_verification = _write_receiver_proof(
        tmp_path / "jsonl_custody_receiver_proof.json",
        candidate_archive=candidate_archive,
    )
    observation_jsonl = tmp_path / "custody_observations.jsonl"
    observation_jsonl.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody_observation.v1",
                "candidate_id": "jsonl_custody_candidate",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": receiver_verification["proof_path"],
                "receiver_verification": receiver_verification,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
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
                "id": "jsonl_custody_experiment",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "observe_custody_jsonl",
                        "kind": "command",
                        "command": ["python", "emit_custody_jsonl.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "jsonl_false_authority",
                                "path": observation_jsonl.as_posix(),
                                "schema_equals": (
                                    "generic_candidate_custody_observation.v1"
                                ),
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
                updated_at_utc = '2026-05-27T18:05:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'jsonl_custody_experiment'
              AND step_id = 'observe_custody_jsonl'
            """,
            (json.dumps({"command": ["python", "emit_custody_jsonl.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is True
    assert observation["succeeded_artifact_failure_steps"] == []
    assert observation["blockers"] == []
    assert observation["status_counts"]["succeeded"] == 1


def test_observer_rejects_jsonl_custody_row_with_stale_runtime_tree_identity(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "jsonl_stale_runtime_candidate.zip",
        b"jsonl stale runtime candidate bytes",
    )
    runtime_dir = tmp_path / "candidate_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    stale_tree_sha = "b" * 64
    proof_path = tmp_path / "jsonl_stale_runtime_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": dict(candidate_archive),
                "candidate_archive_sha256": candidate_archive["sha256"],
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_adapter_ready": True,
                "runtime_adapter_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": runtime_dir.as_posix(),
                    "runtime_tree_sha256": stale_tree_sha,
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    observation_jsonl = tmp_path / "custody_observations.jsonl"
    observation_jsonl.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody_observation.v1",
                "candidate_id": "jsonl_stale_runtime_candidate",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "runtime_consumption_proof_path": proof_path.as_posix(),
                "receiver_verification": {
                    "schema": (
                        "family_agnostic_runtime_consumption_proof_verification.v1"
                    ),
                    "receiver_contract_satisfied": True,
                    "runtime_adapter_ready": True,
                    "proof_present": True,
                    "proof_path": proof_path.as_posix(),
                    "runtime_adapter_tree_sha256": stale_tree_sha,
                    "blockers": [],
                },
                "runtime_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": runtime_dir.as_posix(),
                    "runtime_tree_sha256": stale_tree_sha,
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
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
                "id": "jsonl_custody_experiment",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "observe_custody_jsonl",
                        "kind": "command",
                        "command": ["python", "emit_custody_jsonl.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "jsonl_false_authority",
                                "path": observation_jsonl.as_posix(),
                                "schema_equals": (
                                    "generic_candidate_custody_observation.v1"
                                ),
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
                updated_at_utc = '2026-05-27T18:10:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'jsonl_custody_experiment'
              AND step_id = 'observe_custody_jsonl'
            """,
            (json.dumps({"command": ["python", "emit_custody_jsonl.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["succeeded_artifact_failure_steps"][0][
        "expected_artifacts"
    ][0]
    assert artifact["postcondition_type"] == "jsonl_false_authority"
    assert artifact["postcondition_passed"] is False
    assert any(
        "runtime_tree_sha256_mismatch" in blocker
        for blocker in artifact["artifact_revalidation_blockers"]
    )


def test_observer_surfaces_succeeded_single_family_materializer_manifests(
    tmp_path: Path,
) -> None:
    manifests = [
        (
            "archive_section_entropy_recode",
            "archive_section_entropy_recode_candidate.v1",
            "archive_section_entropy_recode_v1",
            "archive_section_entropy_recode_adapter",
            "family_agnostic_archive_section_entropy_recode",
            "section_recode",
        ),
        (
            "archive_zip_repack",
            "archive_zip_repack_candidate.v1",
            "archive_zip_repack_v1",
            "archive_zip_repack_adapter",
            "family_agnostic_archive_zip_repack",
            "selected_repack",
        ),
        (
            "packet_member_merge",
            "packet_member_merge_candidate.v1",
            "packet_member_merge_v1",
            "packet_member_merge_adapter",
            "family_agnostic_packet_member_merge",
            "selected_merge",
        ),
        (
            "packet_member_recompress",
            "packet_member_recompress_candidate.v1",
            "packet_member_recompress_v1",
            "packet_member_recompress_adapter",
            "family_agnostic_packet_member_recompress",
            "selected_compression",
        ),
        (
            "packet_member_zip_header_elide",
            "packet_member_zip_header_elide_candidate.v1",
            "packet_member_zip_header_elide_v1",
            "packet_member_zip_header_elide_adapter",
            "family_agnostic_packet_member_zip_header_elide",
            "selected_elision",
        ),
        (
            "renderer_payload_dfl1",
            "renderer_payload_dfl1_candidate.v1",
            "renderer_payload_dfl1_v1",
            "renderer_payload_dfl1_adapter",
            "source_runtime_native_renderer_payload_dfl1",
            "selected_payload",
        ),
        (
            "tensor_factorize",
            "tensor_factorize_candidate.v1",
            "tensor_factorize_v1",
            "tensor_factorize_adapter",
            "family_agnostic_tensor_factorize",
            "factorization",
        ),
    ]
    experiments: list[dict[str, object]] = []
    for index, (
        slug,
        schema,
        target_kind,
        materializer_id,
        receiver_contract_kind,
        delta_key,
    ) in enumerate(manifests, start=1):
        manifest = tmp_path / f"{slug}.json"
        candidate_archive = _write_artifact_bytes(
            tmp_path / f"{slug}_candidate.zip",
            (f"{slug}:candidate".encode() * 64)[: 900 + index],
        )
        receiver_verification = _write_receiver_proof(
            tmp_path / f"{slug}_receiver_proof.json",
            candidate_archive=candidate_archive,
        )
        manifest.write_text(
            json.dumps(
                {
                    "schema": schema,
                    "target_kind": target_kind,
                    "materializer_id": materializer_id,
                    "receiver_contract_kind": receiver_contract_kind,
                    "receiver_contract_satisfied": True,
                    "source_archive": {"bytes": 1000 + index, "sha256": f"{index}" * 64},
                    "candidate_archive": candidate_archive,
                    "runtime_consumption_proof_path": receiver_verification["proof_path"],
                    "receiver_verification": receiver_verification,
                    delta_key: {
                        "source_archive_bytes": 1000 + index,
                        "candidate_archive_bytes": candidate_archive["bytes"],
                        "saved_bytes": 100,
                    },
                    "score_claim": False,
                    "score_claim_valid": False,
                    "score_claim_eligible": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "promotable": False,
                    "ready_for_exact_eval_dispatch": False,
                    "field_selection_ready_for_exact_eval_dispatch": False,
                    "dispatch_attempted": False,
                    "gpu_launched": False,
                    "exact_cuda_auth_eval": False,
                    "contest_cuda_auth_eval": False,
                    "score_affecting_payload_changed": False,
                    "charged_bits_changed": False,
                }
            ),
            encoding="utf-8",
        )
        experiments.append(
            {
                "id": f"{slug}_experiment",
                "status": "queued",
                "priority": index,
                "metadata": {
                    "candidate_ids": [f"{slug}_candidate"],
                    "source_unit_ids": [f"{slug}_source"],
                    "source_selection_ids": [f"{slug}_selection"],
                },
                "steps": [
                    {
                        "id": "materialize",
                        "kind": "command",
                        "command": ["python", "tools/run_family_agnostic_materializer.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": manifest.as_posix(),
                                "required_false": [
                                    "score_claim",
                                    "promotion_eligible",
                                    "rank_or_kill_eligible",
                                    "ready_for_exact_eval_dispatch",
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    state = tmp_path / "queue.sqlite"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 3}},
        "experiments": experiments,
    }

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        for slug, *_ in manifests:
            conn.execute(
                """
                UPDATE step_state
                SET status = 'succeeded',
                    attempts = 1,
                    last_event_json = ?,
                    updated_at_utc = '2026-05-25T06:00:00Z'
                WHERE queue_id = 'observer_test'
                  AND experiment_id = ?
                  AND step_id = 'materialize'
                """,
                (
                    json.dumps({"command": ["python", "tools/run_family_agnostic_materializer.py"]}),
                    f"{slug}_experiment",
                ),
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
    assert len(observation["succeeded_artifact_steps"]) == len(manifests)
    artifact_by_schema = {
        step["expected_artifacts"][0]["json_schema"]: step["expected_artifacts"][0]
        for step in observation["succeeded_artifact_steps"]
    }
    assert set(artifact_by_schema) == {schema for _, schema, *_ in manifests}
    for _, schema, target_kind, materializer_id, receiver_contract_kind, delta_key in manifests:
        artifact = artifact_by_schema[schema]
        assert artifact["target_kind"] == target_kind
        assert artifact["materializer_id"] == materializer_id
        assert artifact["receiver_contract_kind"] == receiver_contract_kind
        assert artifact["materializer_delta_source"] == delta_key
        assert artifact["serialized_archive_delta_realized_saved_bytes"] == 100
        assert artifact["score_claim"] is False
        assert artifact["postcondition_passed"] is True


def test_observer_surfaces_future_materializer_with_serialized_archive_delta(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "future_materializer.json"
    candidate_archive = _write_artifact_bytes(
        tmp_path / "future_candidate.zip",
        b"f" * 1000,
    )
    receiver_verification = _write_receiver_proof(
        tmp_path / "future_receiver_proof.json",
        candidate_archive=candidate_archive,
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "future_materializer_candidate.v1",
                "target_kind": "future_byte_packer_v1",
                "materializer_id": "future_byte_packer_adapter",
                "receiver_contract_kind": "family_agnostic_future_byte_packer",
                "receiver_contract_satisfied": True,
                "source_archive": {"bytes": 1024, "sha256": "a" * 64},
                "candidate_archive": candidate_archive,
                "runtime_consumption_proof_path": receiver_verification["proof_path"],
                "receiver_verification": receiver_verification,
                "serialized_archive_delta": {
                    "schema": "serialized_archive_delta_contract.v1",
                    "source_archive_bytes": 1024,
                    "candidate_archive_bytes": candidate_archive["bytes"],
                    "archive_delta_bytes": -24,
                    "realized_saved_bytes": 24,
                    "savings_realized": True,
                    "status": "realized_saving",
                    "score_claim": False,
                },
                "score_claim": False,
                "score_claim_valid": False,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "exact_cuda_auth_eval": False,
                "contest_cuda_auth_eval": False,
                "score_affecting_payload_changed": False,
                "charged_bits_changed": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "future_materializer_experiment",
                "status": "queued",
                "priority": 1,
                "metadata": {
                    "candidate_ids": ["future_candidate"],
                    "source_unit_ids": ["future_source"],
                },
                "steps": [
                    {
                        "id": "materialize",
                        "kind": "command",
                        "command": ["python", "tools/run_family_agnostic_materializer.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": manifest.as_posix(),
                                "required_false": [
                                    "score_claim",
                                    "promotion_eligible",
                                    "rank_or_kill_eligible",
                                    "ready_for_exact_eval_dispatch",
                                ],
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
              AND experiment_id = 'future_materializer_experiment'
              AND step_id = 'materialize'
            """,
            (json.dumps({"command": ["python", "tools/run_family_agnostic_materializer.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert len(observation["succeeded_artifact_steps"]) == 1
    artifact = observation["succeeded_artifact_steps"][0]["expected_artifacts"][0]
    assert artifact["json_schema"] == "future_materializer_candidate.v1"
    assert artifact["serialized_archive_delta_schema"] == "serialized_archive_delta_contract.v1"
    assert artifact["materializer_delta_source"] == "serialized_archive_delta"
    assert artifact["serialized_archive_delta_realized_saved_bytes"] == 24
    assert artifact["score_claim"] is False


def test_observer_rejects_materializer_with_only_proof_present_flag(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "flag_only_materializer.json"
    candidate_archive = _write_artifact_bytes(
        tmp_path / "flag_only_candidate.zip",
        b"flag-only candidate bytes",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "future_materializer_candidate.v1",
                "target_kind": "future_byte_packer_v1",
                "materializer_id": "future_byte_packer_adapter",
                "receiver_contract_kind": "family_agnostic_future_byte_packer",
                "receiver_contract_satisfied": True,
                "candidate_archive": candidate_archive,
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                    "blockers": [],
                },
                "serialized_archive_delta": {
                    "schema": "serialized_archive_delta_contract.v1",
                    "source_archive_bytes": 128,
                    "candidate_archive_bytes": candidate_archive["bytes"],
                    "realized_saved_bytes": 8,
                    "savings_realized": True,
                    "status": "realized_saving",
                },
                "score_claim": False,
                "score_claim_valid": False,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "exact_cuda_auth_eval": False,
                "contest_cuda_auth_eval": False,
                "score_affecting_payload_changed": False,
                "charged_bits_changed": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "flag_only_materializer_experiment",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "materialize",
                        "kind": "command",
                        "command": ["python", "tools/run_family_agnostic_materializer.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": manifest.as_posix(),
                                "required_false": [
                                    "score_claim",
                                    "promotion_eligible",
                                    "rank_or_kill_eligible",
                                    "ready_for_exact_eval_dispatch",
                                ],
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
                updated_at_utc = '2026-05-27T15:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'flag_only_materializer_experiment'
              AND step_id = 'materialize'
            """,
            (json.dumps({"command": ["python", "tools/run_family_agnostic_materializer.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=1,
    )

    assert observation["healthy"] is False
    assert observation["succeeded_artifact_steps"] == []
    failed_step = observation["succeeded_artifact_failure_steps"][0]
    artifact = failed_step["expected_artifacts"][0]
    assert artifact["postcondition_passed"] is False
    assert "experiment_queue_observation_artifact_postcondition_failures:1" in (observation["blockers"])
    assert (
        "json_false_authority_materializer_runtime_or_receiver_proof_path_missing"
        in artifact["artifact_revalidation_blockers"]
    )


def test_observer_rejects_materializer_stale_runtime_tree_identity(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "stale_runtime_materializer.json"
    candidate_archive = _write_artifact_bytes(
        tmp_path / "stale_runtime_candidate.zip",
        b"stale runtime candidate bytes",
    )
    runtime_dir = tmp_path / "candidate_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    proof = tmp_path / "runtime_consumption_proof.json"
    stale_tree_sha = "b" * 64
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": dict(candidate_archive),
                "candidate_archive_sha256": candidate_archive["sha256"],
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_adapter_ready": True,
                "runtime_adapter_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": runtime_dir.as_posix(),
                    "runtime_tree_sha256": stale_tree_sha,
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "packet_member_merge_candidate.v1",
                "target_kind": "packet_member_merge_v1",
                "materializer_id": "packet_member_merge_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_merge",
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "candidate_archive": candidate_archive,
                "runtime_consumption_proof_path": proof.as_posix(),
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "runtime_adapter_ready": True,
                    "proof_present": True,
                    "proof_path": proof.as_posix(),
                    "runtime_adapter_tree_sha256": stale_tree_sha,
                    "blockers": [],
                },
                "packet_member_merge_receiver_runtime": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": runtime_dir.as_posix(),
                    "runtime_tree_sha256": stale_tree_sha,
                },
                "serialized_archive_delta": {
                    "schema": "serialized_archive_delta_contract.v1",
                    "source_archive_bytes": 128,
                    "candidate_archive_bytes": candidate_archive["bytes"],
                    "realized_saved_bytes": 8,
                    "savings_realized": True,
                    "status": "realized_saving",
                },
                "score_claim": False,
                "score_claim_valid": False,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "exact_cuda_auth_eval": False,
                "contest_cuda_auth_eval": False,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(manifest)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_false_authority",
            "path": manifest.as_posix(),
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
            ],
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T15:30:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "tools/run_family_agnostic_materializer.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert any("runtime_tree_sha256_mismatch" in blocker for blocker in artifact["artifact_revalidation_blockers"])


def test_observer_rejects_materializer_expected_runtime_tree_mismatch(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "expected_runtime_mismatch_materializer.json"
    candidate_archive = _write_artifact_bytes(
        tmp_path / "expected_runtime_mismatch_candidate.zip",
        b"expected runtime mismatch candidate bytes",
    )
    runtime_dir = tmp_path / "candidate_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runtime_tree_sha = tree_sha256(runtime_dir)
    stale_expected_tree_sha = "c" * 64
    proof = tmp_path / "runtime_consumption_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": dict(candidate_archive),
                "candidate_archive_sha256": candidate_archive["sha256"],
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_adapter_ready": True,
                "runtime_adapter_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": runtime_dir.as_posix(),
                    "runtime_tree_sha256": runtime_tree_sha,
                    "expected_runtime_tree_sha256": stale_expected_tree_sha,
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "packet_member_merge_candidate.v1",
                "target_kind": "packet_member_merge_v1",
                "materializer_id": "packet_member_merge_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_merge",
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "candidate_archive": candidate_archive,
                "runtime_consumption_proof_path": proof.as_posix(),
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "runtime_adapter_ready": True,
                    "proof_present": True,
                    "proof_path": proof.as_posix(),
                    "runtime_adapter_tree_sha256": runtime_tree_sha,
                    "expected_runtime_tree_sha256": stale_expected_tree_sha,
                    "blockers": [],
                },
                "packet_member_merge_receiver_runtime": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": runtime_dir.as_posix(),
                    "runtime_tree_sha256": runtime_tree_sha,
                    "expected_runtime_tree_sha256": stale_expected_tree_sha,
                },
                "serialized_archive_delta": {
                    "schema": "serialized_archive_delta_contract.v1",
                    "source_archive_bytes": 128,
                    "candidate_archive_bytes": candidate_archive["bytes"],
                    "realized_saved_bytes": 8,
                    "savings_realized": True,
                    "status": "realized_saving",
                },
                "score_claim": False,
                "score_claim_valid": False,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "exact_cuda_auth_eval": False,
                "contest_cuda_auth_eval": False,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(manifest)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_false_authority",
            "path": manifest.as_posix(),
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
            ],
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T17:30:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "tools/run_family_agnostic_materializer.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert any(
        "expected_runtime_tree_sha256_mismatch" in blocker
        for blocker in artifact["artifact_revalidation_blockers"]
    )


def test_observer_rejects_materializer_runtime_ready_without_proof_identity(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "runtime_ready_without_proof_identity.json"
    candidate_archive = _write_artifact_bytes(
        tmp_path / "runtime_ready_without_proof_identity_candidate.zip",
        b"runtime ready without proof identity candidate bytes",
    )
    runtime_dir = tmp_path / "candidate_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runtime_tree_sha = tree_sha256(runtime_dir)
    proof = tmp_path / "runtime_consumption_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "proof_kind": "fixture_receiver_without_runtime_identity",
                "target_kind": "packet_member_recompress_v1",
                "materializer_id": "packet_member_recompress_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_recompress",
                "candidate_archive": dict(candidate_archive),
                "candidate_archive_sha256": candidate_archive["sha256"],
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema": "packet_member_recompress_candidate.v1",
                "target_kind": "packet_member_recompress_v1",
                "materializer_id": "packet_member_recompress_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_recompress",
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "candidate_runtime_adapter_blocker_cleared": True,
                "candidate_runtime_dir": runtime_dir.as_posix(),
                "candidate_runtime_tree_sha256": runtime_tree_sha,
                "candidate_archive": candidate_archive,
                "runtime_consumption_proof_path": proof.as_posix(),
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "runtime_adapter_ready": True,
                    "proof_present": True,
                    "proof_path": proof.as_posix(),
                    "blockers": [],
                },
                "selected_compression": {
                    "source_archive_bytes": 128,
                    "candidate_archive_bytes": candidate_archive["bytes"],
                    "saved_bytes": 8,
                },
                "score_claim": False,
                "score_claim_valid": False,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "exact_cuda_auth_eval": False,
                "contest_cuda_auth_eval": False,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(manifest)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_false_authority",
            "path": manifest.as_posix(),
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
            ],
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T19:30:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "tools/run_family_agnostic_materializer.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert (
        "json_false_authority_materializer_proof_runtime_adapter_identity_claim_missing"
        in artifact["artifact_revalidation_blockers"]
    )


def test_observer_rejects_succeeded_generic_custody_artifact_without_proof(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "generic_candidate.zip",
        b"generic custody candidate bytes",
    )
    artifact_path = tmp_path / "generic_custody.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody.v1",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                    "blockers": [],
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact_path)
    queue["experiments"][0]["steps"][0]["postconditions"][0]["equals"] = (
        "generic_candidate_custody.v1"
    )
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T16:00:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "generic_custody.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert artifact["postcondition_passed"] is False
    assert (
        "json_equals_custody_runtime_or_receiver_proof_path_missing"
        in artifact["artifact_revalidation_blockers"]
    )


def test_observer_rejects_running_generic_custody_artifact_without_proof(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "generic_candidate.zip",
        b"generic custody candidate bytes",
    )
    artifact_path = tmp_path / "generic_custody.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody.v1",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                    "blockers": [],
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact_path)
    queue["experiments"][0]["steps"][0]["postconditions"][0]["equals"] = (
        "generic_candidate_custody.v1"
    )
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T16:02:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "generic_custody.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["running_steps"][0]["expected_artifacts"][0]
    assert artifact["postcondition_passed"] is False
    assert (
        "json_equals_custody_runtime_or_receiver_proof_path_missing"
        in artifact["artifact_revalidation_blockers"]
    )
    assert "experiment_queue_observation_artifact_postcondition_failures:1" in (
        observation["blockers"]
    )


def test_observer_rejects_receiver_contract_only_proof_as_custody(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "generic_candidate.zip",
        b"generic custody candidate bytes",
    )
    proof_path = tmp_path / "receiver_contract_only_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": dict(candidate_archive),
                "candidate_archive_sha256": candidate_archive["sha256"],
                "receiver_contract_satisfied": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "generic_custody.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody.v1",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": proof_path.as_posix(),
                "receiver_verification": {
                    "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                    "proof_path": proof_path.as_posix(),
                    "blockers": [],
                },
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact_path)
    queue["experiments"][0]["steps"][0]["postconditions"][0]["equals"] = (
        "generic_candidate_custody.v1"
    )
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T16:10:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "generic_custody.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    artifact = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert artifact["postcondition_passed"] is False
    assert (
        "json_equals_custody_proof_json_success_flag_missing"
        in artifact["artifact_revalidation_blockers"]
    )


def test_observer_accepts_succeeded_generic_custody_artifact_with_live_proof(
    tmp_path: Path,
) -> None:
    candidate_archive = _write_artifact_bytes(
        tmp_path / "generic_candidate.zip",
        b"generic custody candidate bytes",
    )
    proof = _write_receiver_proof(
        tmp_path / "generic_runtime_consumption_proof.json",
        candidate_archive=candidate_archive,
    )
    artifact_path = tmp_path / "generic_custody.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "generic_candidate_custody.v1",
                "candidate_archive": candidate_archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": proof["proof_path"],
                "receiver_verification": proof,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(artifact_path)
    queue["experiments"][0]["steps"][0]["postconditions"][0]["equals"] = (
        "generic_candidate_custody.v1"
    )
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T16:05:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "generic_custody.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is True
    assert observation["succeeded_artifact_failure_steps"] == []


def test_observer_surfaces_optimizer_candidate_queue_materializer_top_k(
    tmp_path: Path,
) -> None:
    source_queue = tmp_path / "source_queue.json"
    candidate_archive = _write_artifact_bytes(
        tmp_path / "candidate.zip",
        b"r" * 345_422,
    )
    receiver_verification = _write_receiver_proof(
        tmp_path / "optimizer_receiver_proof.json",
        candidate_archive=candidate_archive,
    )
    source_queue.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "top_k": [
                    {
                        "candidate_id": "renderer_payload_dfl1_e20295f0a662",
                        "target_kind": "renderer_payload_dfl1_v1",
                        "materializer_id": "renderer_payload_dfl1_adapter",
                        "receiver_contract_kind": ("source_runtime_native_renderer_payload_dfl1"),
                        "receiver_contract_satisfied": True,
                        "score_affecting_payload_changed": True,
                        "charged_bits_changed": True,
                        "full_frame_inflate_parity_proven": True,
                        "renderer_payload_dfl1_full_frame_inflate_parity_satisfied": (True),
                        "candidate_archive": candidate_archive,
                        "runtime_consumption_proof_path": receiver_verification["proof_path"],
                        "receiver_verification": receiver_verification,
                        "serialized_archive_delta": {
                            "schema": "serialized_archive_delta_contract.v1",
                            "status": "realized_saving",
                            "realized_saved_bytes": 380,
                            "source_archive_bytes": 345_802,
                            "candidate_archive_bytes": 345_422,
                            "savings_realized": True,
                            "score_claim": False,
                            "score_claim_valid": False,
                            "promotion_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                            "rank_or_kill_eligible": False,
                            "promotable": False,
                            "dispatch_attempted": False,
                            "gpu_launched": False,
                        },
                        "readiness_blockers": [],
                        "score_claim": False,
                        "score_claim_valid": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "rank_or_kill_eligible": False,
                        "promotable": False,
                        "dispatch_attempted": False,
                        "gpu_launched": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "observer_test",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "harvest_materializer",
                "status": "queued",
                "priority": 1,
                "metadata": {
                    "candidate_ids": ["targeted_chain:renderer_payload_dfl1_v1:001"],
                    "source_unit_ids": ["renderer_payload_unit"],
                },
                "steps": [
                    {
                        "id": "harvest",
                        "kind": "command",
                        "command": ["python", "tools/harvest_materializer_chain.py"],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_false_authority",
                                "path": source_queue.as_posix(),
                                "required_false": [
                                    "score_claim",
                                    "promotion_eligible",
                                    "rank_or_kill_eligible",
                                    "ready_for_exact_eval_dispatch",
                                ],
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
                updated_at_utc = '2026-05-26T19:10:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'harvest_materializer'
              AND step_id = 'harvest'
            """,
            (json.dumps({"command": ["python", "tools/harvest_materializer_chain.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is True
    assert len(observation["succeeded_artifact_steps"]) == 1
    artifact = observation["succeeded_artifact_steps"][0]["expected_artifacts"][0]
    rows = artifact["optimizer_candidate_queue_materializer_rows"]
    assert artifact["json_schema"] == "optimizer_candidate_queue_v1"
    assert artifact["optimizer_candidate_queue_materializer_row_count"] == 1
    assert rows[0]["optimizer_candidate_queue_row_index"] == 0
    assert rows[0]["target_kind"] == "renderer_payload_dfl1_v1"
    assert rows[0]["receiver_contract_satisfied"] is True
    assert rows[0]["materializer_score_affecting_payload_changed"] is True
    assert rows[0]["materializer_charged_bits_changed"] is True
    assert "score_affecting_payload_changed" not in rows[0]
    assert "charged_bits_changed" not in rows[0]
    assert rows[0]["serialized_archive_delta_realized_saved_bytes"] == 380
    assert rows[0]["readiness_blockers"] == []


def test_observer_refuses_optimizer_top_k_row_without_nested_revalidation(
    tmp_path: Path,
) -> None:
    source_queue = tmp_path / "source_queue.json"
    source_queue.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
                "top_k": [
                    {
                        "candidate_id": "missing_receiver_runtime",
                        "target_kind": "renderer_payload_dfl1_v1",
                        "materializer_id": "renderer_payload_dfl1_adapter",
                        "receiver_contract_kind": ("source_runtime_native_renderer_payload_dfl1"),
                        "receiver_contract_satisfied": True,
                        "candidate_archive": {
                            "path": (tmp_path / "missing_candidate.zip").as_posix(),
                            "bytes": 345_422,
                            "sha256": "e" * 64,
                        },
                        "serialized_archive_delta": {
                            "schema": "serialized_archive_delta_contract.v1",
                            "status": "realized_saving",
                            "realized_saved_bytes": 380,
                            "source_archive_bytes": 345_802,
                            "candidate_archive_bytes": 345_422,
                            "savings_realized": True,
                            "score_claim": False,
                            "score_claim_valid": False,
                            "promotion_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                            "rank_or_kill_eligible": False,
                            "promotable": False,
                            "dispatch_attempted": False,
                            "gpu_launched": False,
                        },
                        "readiness_blockers": [],
                        "score_claim": False,
                        "score_claim_valid": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "rank_or_kill_eligible": False,
                        "promotable": False,
                        "dispatch_attempted": False,
                        "gpu_launched": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "queue.sqlite"
    queue = _queue(source_queue)
    queue["experiments"][0]["steps"][0]["postconditions"] = [
        {
            "type": "json_false_authority",
            "path": source_queue.as_posix(),
            "required_false": [
                "score_claim",
                "promotion_eligible",
                "rank_or_kill_eligible",
                "ready_for_exact_eval_dispatch",
            ],
        }
    ]

    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded',
                attempts = 1,
                last_event_json = ?,
                updated_at_utc = '2026-05-27T14:25:00Z'
            WHERE queue_id = 'observer_test'
              AND experiment_id = 'exp0'
              AND step_id = 'smoke'
            """,
            (json.dumps({"command": ["python", "tools/harvest_materializer_chain.py"]}),),
        )
        conn.commit()

    observation = observe_experiment_queue(
        queue,
        state_path=state,
        repo_root=tmp_path,
        tail_lines=0,
    )

    assert observation["healthy"] is False
    assert observation["succeeded_artifact_steps"] == []
    artifact = observation["succeeded_artifact_failure_steps"][0]["expected_artifacts"][0]
    assert artifact.get("optimizer_candidate_queue_materializer_row_count") is None
    blockers = artifact["artifact_revalidation_blockers"]
    assert any("candidate_archive_file_missing" in blocker for blocker in blockers)
    assert any("runtime_or_receiver_proof_path_missing" in blocker for blocker in blockers)


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
