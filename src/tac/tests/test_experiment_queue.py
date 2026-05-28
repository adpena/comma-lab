# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.comma_lab.scheduler.experiment_queue import (
    LOG_PATH_COMPONENT_MAX_CHARS,
    SCHEDULER_RUNTIME_POLICY_SCHEMA,
    ExperimentQueueError,
    _condition_passes,
    _safe_log_path_component,
    apply_scheduler_runtime_policy,
    assert_canonical_state_for_execution,
    assert_no_orphaned_steps_for_execution,
    claim_ready_step_for_execution,
    connect_state,
    default_state_path,
    derive_scheduler_runtime_policy,
    finalize_claimed_step_execution,
    initialize_queue_state,
    normalize_queue_definition,
    queue_definition_drift,
    queue_performance_summary,
    queue_resource_kinds,
    queue_summary,
    ready_steps,
    reconcile_satisfied_queued_steps,
    reconcile_stale_running_steps,
    resolve_worker_max_parallel,
    retire_orphaned_steps,
    rewind_step,
    run_queue_worker,
    run_ready_step,
    set_control_mode,
    worker_resource_limits,
)
from tac.repo_io import tree_sha256


def _queue(tmp_path: Path) -> dict[str, object]:
    marker = tmp_path / "materialized.json"
    advisory = tmp_path / "advisory.json"
    return normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "unit_queue",
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1, "cloud_cpu": 1},
            },
            "experiments": [
                {
                    "id": "candidate_a",
                    "priority": 5,
                    "steps": [
                        {
                            "id": "materialize",
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import json, pathlib; "
                                    f"pathlib.Path({str(marker)!r}).write_text("
                                    "json.dumps({'ok': True}))"
                                ),
                            ],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {"type": "path_exists", "path": marker.name},
                                {
                                    "type": "json_equals",
                                    "path": marker.name,
                                    "key": "ok",
                                    "equals": True,
                                },
                            ],
                        },
                        {
                            "id": "local_advisory",
                            "requires": ["materialize"],
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import json, pathlib; "
                                    f"pathlib.Path({str(advisory)!r}).write_text("
                                    "json.dumps({'score_claim': False}))"
                                ),
                            ],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {
                                    "type": "json_equals",
                                    "path": advisory.name,
                                    "key": "score_claim",
                                    "equals": False,
                                }
                            ],
                        },
                        {
                            "id": "exact_anchor",
                            "requires": ["local_advisory"],
                            "command": [sys.executable, "-c", "print('anchor')"],
                            "resources": {"kind": "cloud_cpu"},
                        },
                    ],
                }
            ],
        }
    )


def test_experiment_queue_executes_local_steps_and_gates_cloud(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)

        ready = ready_steps(conn, queue)
        assert [step.step_id for step in ready] == ["materialize"]

        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        assert result["succeeded"] is True

        ready = ready_steps(conn, queue)
        assert [step.step_id for step in ready] == ["local_advisory"]
        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        assert result["succeeded"] is True

        assert ready_steps(conn, queue) == []
        assert [step.step_id for step in ready_steps(conn, queue, allow_cloud=True)] == [
            "exact_anchor"
        ]
        summary = queue_summary(conn, queue)
        assert summary["status_counts"] == {"queued": 1, "succeeded": 2}


def test_worker_log_paths_shorten_long_queue_components(tmp_path: Path) -> None:
    marker = tmp_path / "long_component_done.json"
    queue_id = "frontier_rate_attack_" + ("q" * 220)
    experiment_id = "materializer_work_" + ("x" * 260)
    step_id = "targeted_component_correction_chain_materializer_execution_" + ("s" * 220)
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": experiment_id,
                    "steps": [
                        {
                            "id": step_id,
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib; "
                                    f"pathlib.Path({str(marker)!r}).write_text('ok')"
                                ),
                            ],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {"type": "path_exists", "path": marker.name}
                            ],
                        }
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=1,
            max_parallel=1,
            log_root=tmp_path / "logs",
            idle_sleep_seconds=0,
            max_idle_cycles=0,
        )
        summary = queue_summary(conn, queue)

    log_paths = list((tmp_path / "logs").rglob("*.log"))
    assert result["success_count"] == 1
    assert result["failure_count"] == 0
    assert summary["status_counts"] == {"succeeded": 1}
    assert len(log_paths) == 1
    log_parts = log_paths[0].relative_to(tmp_path / "logs").parts[:-1]
    assert all(len(part) <= LOG_PATH_COMPONENT_MAX_CHARS for part in log_parts)
    assert log_parts == (
        _safe_log_path_component(queue_id),
        _safe_log_path_component(experiment_id),
        _safe_log_path_component(step_id),
    )
    assert experiment_id not in str(log_paths[0])


def test_experiment_queue_accepts_legacy_json_file_key_equals_postcondition(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact_dir"
    manifest = artifact_dir / "manifest.json"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "legacy_postcondition_queue",
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": "candidate_a",
                    "priority": 1,
                    "metadata": {
                        "work_id": "work_drop_pair_001",
                        "backlog_key": "pairset_drop_one:rank001",
                        "source_unit_ids": ["pair0371"],
                        "source_selection_ids": ["rank001_pair0371"],
                    },
                    "steps": [
                        {
                            "id": "write_manifest",
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import json, pathlib; "
                                    f"root = pathlib.Path({str(artifact_dir)!r}); "
                                    "root.mkdir(parents=True); "
                                    "(root / 'payload.bin').write_bytes(b'abcde'); "
                                    f"pathlib.Path({str(manifest)!r}).write_text("
                                    "json.dumps({'schema': 'chain.v1'}))"
                                ),
                            ],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {
                                    "type": "json_file_key_equals",
                                    "path": str(manifest),
                                    "key": "schema",
                                    "value": "chain.v1",
                                }
                            ],
                            "telemetry": {
                                "artifact_paths": [str(artifact_dir)],
                                "recursive": True,
                                "max_recursive_entries": 8,
                            },
                        }
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue)
        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        performance = queue_performance_summary(conn, queue)

    assert result["succeeded"] is True
    telemetry = result["telemetry"]
    assert telemetry["schema"] == "experiment_queue_step_telemetry.v1"
    assert telemetry["log_bytes"] is not None
    assert telemetry["artifact_record_count"] == 2
    records = {record["path"]: record for record in telemetry["artifact_records"]}
    assert records["artifact_dir"]["recursive_bytes"] >= 5
    assert records["artifact_dir/manifest.json"]["bytes"] > 0
    assert performance["schema"] == "experiment_queue_performance_summary.v1"
    assert performance["telemetry_only"] is True
    assert performance["score_claim"] is False
    assert performance["promotion_eligible"] is False
    assert performance["rank_or_kill_eligible"] is False
    assert performance["candidate_id_by_experiment"] == {
        "candidate_a": ["pair0371"]
    }
    assert performance["work_id_by_experiment"] == {
        "candidate_a": "work_drop_pair_001"
    }
    assert performance["backlog_key_by_experiment"] == {
        "candidate_a": "pairset_drop_one:rank001"
    }
    assert performance["source_unit_ids_by_experiment"] == {
        "candidate_a": ["pair0371"]
    }
    assert performance["by_resource_kind"]["local_cpu"]["run_count"] == 1
    assert (
        performance["by_resource_kind"]["local_cpu"]["dominant_resource_kind"]
        == "local_cpu"
    )
    assert performance["by_resource_kind"]["local_cpu"]["artifact_record_count"] == 2
    assert performance["by_resource_kind"]["local_cpu"]["artifact_record_bytes_sum"] >= 5
    assert (
        performance["by_resource_kind"]["local_cpu"]["artifact_record_raw_bytes_sum"]
        > performance["by_resource_kind"]["local_cpu"]["artifact_record_bytes_sum"]
    )
    assert (
        performance["by_step"]["candidate_a.write_manifest"]["success_count"]
        == 1
    )
    assert performance["by_step"]["candidate_a.write_manifest"]["candidate_ids"] == [
        "pair0371"
    ]
    assert performance["by_experiment"]["candidate_a"]["work_ids"] == [
        "work_drop_pair_001"
    ]
    assert performance["by_work_id"]["work_drop_pair_001"]["candidate_ids"] == [
        "pair0371"
    ]
    assert performance["by_backlog_key"]["pairset_drop_one:rank001"][
        "source_unit_ids"
    ] == ["pair0371"]
    assert performance["by_source_unit_id"]["pair0371"]["backlog_keys"] == [
        "pairset_drop_one:rank001"
    ]
    assert performance["by_source_selection_id"]["rank001_pair0371"][
        "source_unit_ids"
    ] == ["pair0371"]


def test_scheduler_runtime_policy_derives_advisory_limits_and_timeouts(
    tmp_path: Path,
) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "runtime_policy_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {
                    "local_cpu": 1,
                    "local_io_heavy": 8,
                    "local_mlx": 1,
                    "cloud_cpu": 3,
                },
            },
            "experiments": [
                {
                    "id": "cpu_exp",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "cpu_step",
                            "command": [sys.executable, "-c", "print('cpu')"],
                            "resources": {"kind": "local_cpu"},
                            "timeout_seconds": 12,
                        }
                    ],
                },
                {
                    "id": "io_exp",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "io_step",
                            "command": [sys.executable, "-c", "print('io')"],
                            "resources": {"kind": "local_io_heavy"},
                            "timeout_seconds": 50,
                        }
                    ],
                },
                {
                    "id": "mlx_exp",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "mlx_step",
                            "command": [sys.executable, "-c", "print('mlx')"],
                            "resources": {"kind": "local_mlx"},
                            "timeout_seconds": 30,
                        }
                    ],
                },
                {
                    "id": "cloud_exp",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "cloud_step",
                            "command": [sys.executable, "-c", "print('cloud')"],
                            "resources": {"kind": "cloud_cpu"},
                            "timeout_seconds": 99,
                        }
                    ],
                },
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        for event_type, experiment_id, step_id, payload in (
            (
                "step_succeeded",
                "cpu_exp",
                "cpu_step",
                {"resource_kind": "local_cpu", "elapsed_seconds": 4.0},
            ),
            (
                "step_succeeded",
                "cpu_exp",
                "cpu_step",
                {"resource_kind": "local_cpu", "elapsed_seconds": 8.0},
            ),
            (
                "step_succeeded",
                "io_exp",
                "io_step",
                {"resource_kind": "local_io_heavy", "elapsed_seconds": 7.0},
            ),
            (
                "step_failed",
                "mlx_exp",
                "mlx_step",
                {
                    "resource_kind": "local_mlx",
                    "elapsed_seconds": 31.0,
                    "timed_out": True,
                },
            ),
        ):
            conn.execute(
                """
                INSERT INTO queue_events(
                    ts_utc, queue_id, experiment_id, step_id, event_type, payload_json
                )
                VALUES ('2026-05-24T00:00:00Z', ?, ?, ?, ?, ?)
                """,
                (
                    "runtime_policy_queue",
                    experiment_id,
                    step_id,
                    event_type,
                    json.dumps(payload),
                ),
            )
        conn.commit()

        policy = derive_scheduler_runtime_policy(
            conn,
            queue,
            cpu_count=12,
            timeout_multiplier=2.0,
            min_timeout_seconds=10,
            max_timeout_seconds=120,
        )

    assert policy["schema"] == SCHEDULER_RUNTIME_POLICY_SCHEMA
    assert policy["telemetry_only"] is True
    assert policy["score_claim"] is False
    assert policy["promotion_eligible"] is False
    assert policy["ready_for_exact_eval_dispatch"] is False
    assert policy["dispatch_packet_ready"] is False
    assert policy["recommended_max_concurrency"]["local_cpu"] == 12
    assert policy["recommended_max_concurrency"]["local_io_heavy"] == 3
    assert policy["recommended_max_concurrency"]["local_mlx"] == 1
    assert policy["recommended_max_concurrency"]["cloud_cpu"] == 3
    assert policy["recommended_timeout_seconds_by_resource"]["local_cpu"] == 16
    assert policy["recommended_timeout_seconds_by_resource"]["local_io_heavy"] == 50
    assert policy["recommended_timeout_seconds_by_resource"]["local_mlx"] == 62
    assert policy["resource_policies"]["local_mlx"]["observed_timeout_count"] == 1
    assert policy["resource_policies"]["local_mlx"]["dispatch_packet_ready"] is False

    updated = apply_scheduler_runtime_policy(queue, policy)
    assert updated["controls"]["max_concurrency"]["local_cpu"] == 12
    assert updated["controls"]["max_concurrency"]["local_io_heavy"] == 3
    assert updated["controls"]["max_concurrency"]["local_mlx"] == 1
    assert updated["controls"]["max_concurrency"]["cloud_cpu"] == 3
    assert updated["experiments"][0]["steps"][0]["timeout_seconds"] == 16
    assert updated["experiments"][1]["steps"][0]["timeout_seconds"] == 50
    assert updated["experiments"][2]["steps"][0]["timeout_seconds"] == 62


def test_scheduler_runtime_policy_apply_rejects_nested_false_authority(
    tmp_path: Path,
) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "runtime_policy_false_authority_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": "cpu_exp",
                    "steps": [
                        {
                            "id": "cpu_step",
                            "command": [sys.executable, "-c", "print('cpu')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                }
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        policy = derive_scheduler_runtime_policy(conn, queue, cpu_count=4)

    poisoned = json.loads(json.dumps(policy))
    poisoned["score_claim_valid"] = True
    poisoned["resource_policies"]["local_cpu"]["gpu_launched"] = True

    with pytest.raises(ExperimentQueueError, match="truthy authority fields"):
        apply_scheduler_runtime_policy(queue, poisoned)


def test_experiment_queue_reconciles_queued_steps_with_satisfied_postconditions(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"

    def queue_with_timeout(timeout_seconds: int) -> dict[str, object]:
        return normalize_queue_definition(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "postcondition_reconcile_queue",
                "controls": {
                    "mode": "running",
                    "local_first": True,
                    "max_concurrency": {"local_cpu": 1},
                },
                "experiments": [
                    {
                        "id": "candidate_a",
                        "priority": 1,
                        "steps": [
                            {
                                "id": "produce",
                                "command": [sys.executable, "-c", "print('already done')"],
                                "resources": {"kind": "local_cpu"},
                                "timeout_seconds": timeout_seconds,
                                "postconditions": [
                                    {
                                        "type": "json_equals",
                                        "path": artifact.name,
                                        "key": "schema",
                                        "equals": "artifact.v1",
                                    }
                                ],
                            },
                            {
                                "id": "consume",
                                "requires": ["produce"],
                                "command": [sys.executable, "-c", "print('ready')"],
                                "resources": {"kind": "local_cpu"},
                            },
                        ],
                    }
                ],
            }
        )

    artifact.write_text(json.dumps({"schema": "artifact.v1"}), encoding="utf-8")
    queue_v1 = queue_with_timeout(10)
    queue_v2 = queue_with_timeout(1200)

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue_v1)
        initialize_queue_state(conn, queue_v2)
        drift_before = queue_definition_drift(conn, queue_v2)
        result = reconcile_satisfied_queued_steps(
            conn,
            queue_v2,
            repo_root=tmp_path,
        )
        summary = queue_summary(conn, queue_v2, repo_root=tmp_path)
        drift_after = queue_definition_drift(conn, queue_v2)

    assert drift_before["changed_step_count"] == 0
    assert result["schema"] == "experiment_queue_postcondition_reconciliation.v1"
    assert result["reconciled_step_count"] == 1
    assert result["reconciled_steps"][0]["step_id"] == "produce"
    assert summary["status_counts"] == {"queued": 1, "succeeded": 1}
    assert [step["step_id"] for step in summary["ready_steps"]] == ["consume"]
    assert drift_after["changed_step_count"] == 0


def test_experiment_queue_reconcile_respects_dependency_causality(
    tmp_path: Path,
) -> None:
    consumer_artifact = tmp_path / "consumer.json"
    consumer_artifact.write_text(json.dumps({"schema": "consumer.v1"}), encoding="utf-8")
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "dependency_reconcile_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "candidate",
                    "steps": [
                        {
                            "id": "produce",
                            "command": [sys.executable, "-c", "print('producer not run')"],
                            "postconditions": [{"type": "path_exists", "path": "producer.json"}],
                        },
                        {
                            "id": "consume",
                            "requires": ["produce"],
                            "command": [sys.executable, "-c", "print('consumer already done')"],
                            "postconditions": [
                                {
                                    "type": "json_equals",
                                    "path": consumer_artifact.name,
                                    "key": "schema",
                                    "equals": "consumer.v1",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = reconcile_satisfied_queued_steps(conn, queue, repo_root=tmp_path)
        summary = queue_summary(conn, queue, repo_root=tmp_path)

    assert result["reconciled_step_count"] == 0
    assert result["dependency_blocked_step_count"] == 1
    assert result["dependency_blocked_steps"][0]["step_id"] == "consume"
    assert summary["status_counts"] == {"queued": 2}


def test_experiment_queue_pause_freeze_and_rewind(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        set_control_mode(conn, "unit_queue", "paused", reason="operator")
        assert ready_steps(conn, queue) == []

        set_control_mode(conn, "unit_queue", "running", reason="resume")
        ready = ready_steps(conn, queue)
        assert [step.step_id for step in ready] == ["materialize"]

        run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        ready = ready_steps(conn, queue)
        assert [step.step_id for step in ready] == ["local_advisory"]
        run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        rewind_step(
            conn,
            "unit_queue",
            "candidate_a",
            "materialize",
            reason="rerun locality",
            queue=queue,
        )
        assert [step.step_id for step in ready_steps(conn, queue)] == ["materialize"]
        assert queue_summary(conn, queue)["status_counts"] == {"queued": 3}


def test_experiment_queue_summary_reports_definition_frozen_steps(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    queue["experiments"][0]["status"] = "frozen"
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue)
        summary = queue_summary(conn, queue)

    assert ready == []
    assert summary["status_counts"] == {"frozen": 3}
    assert {step["stored_status"] for step in summary["steps"]} == {"queued"}
    assert {step["status"] for step in summary["steps"]} == {"frozen"}
    assert {step["experiment_status"] for step in summary["steps"]} == {"frozen"}


def test_initialize_queue_state_requeues_dependents_on_upstream_definition_drift(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        for step_id in ("materialize", "local_advisory", "exact_anchor"):
            conn.execute(
                """
                UPDATE step_state
                SET status = 'succeeded'
                WHERE queue_id = 'unit_queue'
                  AND experiment_id = 'candidate_a'
                  AND step_id = ?
                """,
                (step_id,),
            )
        conn.commit()
        queue["experiments"][0]["steps"][0]["command"] = [
            sys.executable,
            "-c",
            "print('definition changed')",
        ]

        initialize_queue_state(conn, queue)
        rows = {
            row["step_id"]: row
            for row in conn.execute(
                """
                SELECT step_id, status, last_event_json
                FROM step_state
                WHERE queue_id = 'unit_queue' AND experiment_id = 'candidate_a'
                """
            ).fetchall()
        }

    assert rows["materialize"]["status"] == "queued"
    assert rows["local_advisory"]["status"] == "queued"
    assert rows["exact_anchor"]["status"] == "queued"
    assert json.loads(rows["materialize"]["last_event_json"])["definition_changed"] is True
    assert (
        json.loads(rows["local_advisory"]["last_event_json"])[
            "upstream_definition_changed"
        ]
        is True
    )


def test_experiment_queue_supports_cross_experiment_dependencies_and_rewind(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate.txt"
    candidate = tmp_path / "candidate.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "cross_dependency_queue",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 2}},
            "experiments": [
                {
                    "id": "preflight",
                    "priority": 0,
                    "steps": [
                        {
                            "id": "cleanup",
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(gate)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": gate.name}],
                        }
                    ],
                },
                {
                    "id": "candidate_a",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "materialize",
                            "requires": ["preflight.cleanup"],
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(candidate)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": candidate.name}],
                        }
                    ],
                },
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue)
        assert [(step.experiment_id, step.step_id) for step in ready] == [("preflight", "cleanup")]

        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        assert result["succeeded"] is True
        assert [(step.experiment_id, step.step_id) for step in ready_steps(conn, queue)] == [
            ("candidate_a", "materialize")
        ]

        rewind_step(
            conn,
            "cross_dependency_queue",
            "preflight",
            "cleanup",
            reason="rerun storage cleanup",
            queue=queue,
        )
        assert queue_summary(conn, queue)["status_counts"] == {"queued": 2}


def test_experiment_queue_blocks_ready_step_when_dependency_artifact_goes_stale(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate.txt"
    candidate = tmp_path / "candidate.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "stale_dependency_queue",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 2}},
            "experiments": [
                {
                    "id": "candidate_a",
                    "steps": [
                        {
                            "id": "prepare",
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(gate)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": gate.name}],
                        },
                        {
                            "id": "consume",
                            "requires": ["prepare"],
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib; "
                                    f"pathlib.Path({str(candidate)!r}).write_text('ok')"
                                ),
                            ],
                            "postconditions": [
                                {"type": "path_exists", "path": candidate.name}
                            ],
                        },
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=tmp_path)
        assert [step.step_id for step in ready] == ["prepare"]
        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        assert result["succeeded"] is True
        assert [step.step_id for step in ready_steps(conn, queue, repo_root=tmp_path)] == [
            "consume"
        ]

        gate.unlink()

        assert ready_steps(conn, queue, repo_root=tmp_path) == []
        summary = queue_summary(conn, queue, repo_root=tmp_path)
        assert summary["ready_steps"] == []


def test_experiment_queue_timeout_marks_failed_not_running(tmp_path: Path) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "timeout_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "candidate_timeout",
                    "steps": [
                        {
                            "id": "slow",
                            "command": [
                                sys.executable,
                                "-c",
                                "import time; time.sleep(5)",
                            ],
                            "timeout_seconds": 1,
                        }
                    ],
                }
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue)
        result = run_ready_step(
            conn,
            queue,
            ready[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )
        assert result["succeeded"] is False
        assert result["returncode"] == 124
        assert result["timed_out"] is True
        assert queue_summary(conn, queue)["status_counts"] == {"failed": 1}


def test_experiment_queue_rejects_shell_string_commands() -> None:
    payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "bad",
        "experiments": [
            {
                "id": "candidate",
                "steps": [{"id": "bad_step", "command": "echo nope"}],
            }
        ],
    }
    try:
        normalize_queue_definition(payload)
    except ExperimentQueueError as exc:
        assert "argv list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("queue accepted a shell string command")


def test_experiment_queue_normalizes_input_artifact_paths() -> None:
    payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "input_paths",
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {
                        "id": "materialize",
                        "command": [sys.executable, "-c", "print('ok')"],
                        "telemetry": {
                            "input_artifact_paths": ["input/a.json"],
                        },
                    }
                ],
            }
        ],
    }

    queue = normalize_queue_definition(payload)

    telemetry = queue["experiments"][0]["steps"][0]["telemetry"]
    assert telemetry["input_artifact_paths"] == ["input/a.json"]


def test_experiment_queue_preserves_queue_level_metadata() -> None:
    payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "metadata_queue",
        "metadata": {
            "schema": "unit_queue_metadata.v1",
            "blocked_work_order_ids": ["external_anchor"],
            "score_claim": False,
        },
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {
                        "id": "materialize",
                        "command": [sys.executable, "-c", "print('ok')"],
                    }
                ],
            }
        ],
    }

    queue = normalize_queue_definition(payload)

    assert queue["metadata"] == payload["metadata"]


def test_experiment_queue_rejects_malformed_input_artifact_paths() -> None:
    payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "bad_input_paths",
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {
                        "id": "materialize",
                        "command": [sys.executable, "-c", "print('ok')"],
                        "telemetry": {
                            "input_artifact_paths": "input/a.json",
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(ExperimentQueueError, match="input_artifact_paths must be a list"):
        normalize_queue_definition(payload)


def test_experiment_queue_rejects_duplicate_step_ids() -> None:
    payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "bad",
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {"id": "same", "command": [sys.executable, "-c", "print('a')"]},
                    {"id": "same", "command": [sys.executable, "-c", "print('b')"]},
                ],
            }
        ],
    }
    with pytest.raises(ExperimentQueueError, match="duplicate step id"):
        normalize_queue_definition(payload)


def test_experiment_queue_rejects_unknown_resource_kind() -> None:
    payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "bad",
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {
                        "id": "cloud_typo",
                        "command": [sys.executable, "-c", "print('should not run')"],
                        "resources": {"kind": "modal-cpu"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ExperimentQueueError, match="unsupported resource kind"):
        normalize_queue_definition(payload)


def test_experiment_queue_stale_ready_step_respects_pause(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        stale_ready = ready_steps(conn, queue)[0]
        set_control_mode(conn, "unit_queue", "paused", reason="operator")

        result = run_ready_step(
            conn,
            queue,
            stale_ready,
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )

        assert result["executed"] is False
        assert result["claim_refused"] is True
        assert result["claim_refused_reason"] == "control_not_running"
        summary = queue_summary(conn, queue)
        assert summary["status_counts"] == {"queued": 3}
        assert summary["steps"][0]["attempts"] == 0

        events = [
            row["event_type"]
            for row in conn.execute("SELECT event_type FROM queue_events ORDER BY id").fetchall()
        ]
        assert "step_claim_refused_control_not_running" in events


def test_experiment_queue_atomic_claim_respects_resource_limits_across_connections(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "queue.sqlite"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "atomic_resource_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_mlx": 1},
            },
            "experiments": [
                {
                    "id": "candidate",
                    "steps": [
                        {
                            "id": "first",
                            "resources": {"kind": "local_mlx"},
                            "command": [sys.executable, "-c", "print('first')"],
                        },
                        {
                            "id": "second",
                            "resources": {"kind": "local_mlx"},
                            "command": [sys.executable, "-c", "print('second')"],
                        },
                    ],
                }
            ],
        }
    )

    with connect_state(state_path) as conn_a, connect_state(state_path) as conn_b:
        initialize_queue_state(conn_a, queue)
        stale_ready_a = ready_steps(conn_a, queue)
        stale_ready_b = ready_steps(conn_b, queue)

        assert [step.step_id for step in stale_ready_a] == ["first"]
        assert [step.step_id for step in stale_ready_b] == ["first"]

        assert claim_ready_step_for_execution(
            conn_a,
            queue,
            stale_ready_a[0],
            event={"command": list(stale_ready_a[0].command), "test": "first_claim"},
        ) is None
        refused = claim_ready_step_for_execution(
            conn_b,
            queue,
            stale_ready_b[0],
            event={"command": list(stale_ready_b[0].command), "test": "second_claim"},
        )

        assert refused == "not_queued"
        summary = queue_summary(conn_b, queue)
        first_row = next(row for row in summary["steps"] if row["step_id"] == "first")
        second_row = next(row for row in summary["steps"] if row["step_id"] == "second")
        assert first_row["status"] == "running"
        assert first_row["attempts"] == 1
        assert second_row["status"] == "queued"
        assert second_row["attempts"] == 0


def test_experiment_queue_atomic_claim_refuses_stale_resource_capacity(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "queue.sqlite"
    queue_payload = {
        "schema": "experiment_queue.v1",
        "queue_id": "atomic_capacity_queue",
        "controls": {
            "mode": "running",
            "max_concurrency": {"local_mlx": 1},
        },
        "experiments": [
            {
                "id": "candidate",
                "steps": [
                    {
                        "id": "first",
                        "resources": {"kind": "local_mlx"},
                        "command": [sys.executable, "-c", "print('first')"],
                    },
                    {
                        "id": "second",
                        "resources": {"kind": "local_mlx"},
                        "command": [sys.executable, "-c", "print('second')"],
                    },
                ],
            }
        ],
    }
    queue = normalize_queue_definition(queue_payload)
    planning_payload = {
        **queue_payload,
        "controls": {"mode": "running", "max_concurrency": {"local_mlx": 2}},
    }
    planning_queue = normalize_queue_definition(planning_payload)

    with connect_state(state_path) as conn_a, connect_state(state_path) as conn_b:
        initialize_queue_state(conn_a, queue)
        first_ready, second_ready = ready_steps(conn_a, planning_queue)

        assert claim_ready_step_for_execution(
            conn_a,
            queue,
            first_ready,
            event={"command": list(first_ready.command), "test": "first_claim"},
        ) is None
        refused = claim_ready_step_for_execution(
            conn_b,
            queue,
            second_ready,
            event={"command": list(second_ready.command), "test": "second_claim"},
        )

        assert refused == "resource_limit_reached"
        summary = queue_summary(conn_b, queue)
        second_row = next(row for row in summary["steps"] if row["step_id"] == "second")
        assert second_row["status"] == "queued"
        assert second_row["attempts"] == 0
        events = [
            row["event_type"]
            for row in conn_b.execute(
                "SELECT event_type FROM queue_events ORDER BY id"
            ).fetchall()
        ]
        assert "step_claim_refused_resource_limit_reached" in events


def test_finalize_claimed_step_refuses_stale_terminal_state(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "materialized.json"
    marker.write_text(json.dumps({"ok": True}), encoding="utf-8")
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue)[0]
        event = {
            "command": list(ready.command),
            "worker_run_id": "worker-a",
            "test": "running_claim",
        }
        assert claim_ready_step_for_execution(conn, queue, ready, event=event) is None
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                json.dumps({"worker_run_id": "recovery-worker"}),
                ready.queue_id,
                ready.experiment_id,
                ready.step_id,
            ),
        )
        conn.commit()

        result = finalize_claimed_step_execution(
            conn,
            queue,
            ready,
            repo_root=tmp_path,
            log_path=tmp_path / "worker.log",
            returncode=0,
            timed_out=False,
            execution_error=None,
            elapsed_seconds=1.0,
            event=event,
        )

        row = conn.execute(
            """
            SELECT status, last_event_json
            FROM step_state
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (ready.queue_id, ready.experiment_id, ready.step_id),
        ).fetchone()
        events = [
            row["event_type"]
            for row in conn.execute("SELECT event_type FROM queue_events ORDER BY id").fetchall()
        ]

    assert result["succeeded"] is False
    assert result["finalize_refused"] is True
    assert "status_not_running:failed" in result["finalize_refusal_reasons"]
    assert "worker_run_id_mismatch" in result["finalize_refusal_reasons"]
    assert row["status"] == "failed"
    assert json.loads(row["last_event_json"]) == {"worker_run_id": "recovery-worker"}
    assert "step_finalize_refused" in events


def test_finalize_claimed_step_requires_matching_worker_run_id(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "materialized.json"
    marker.write_text(json.dumps({"ok": True}), encoding="utf-8")
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue)[0]
        claim_event = {
            "command": list(ready.command),
            "worker_run_id": "worker-a",
            "test": "running_claim",
        }
        assert claim_ready_step_for_execution(conn, queue, ready, event=claim_event) is None
        result = finalize_claimed_step_execution(
            conn,
            queue,
            ready,
            repo_root=tmp_path,
            log_path=tmp_path / "worker.log",
            returncode=0,
            timed_out=False,
            execution_error=None,
            elapsed_seconds=1.0,
            event={**claim_event, "worker_run_id": "worker-b"},
        )
        row = conn.execute(
            """
            SELECT status
            FROM step_state
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (ready.queue_id, ready.experiment_id, ready.step_id),
        ).fetchone()

    assert result["succeeded"] is False
    assert result["finalize_refused"] is True
    assert result["finalize_refusal_reasons"] == ["worker_run_id_mismatch"]
    assert row["status"] == "running"


def test_reconcile_stale_running_step_recovers_when_postconditions_pass(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "first.done"
    marker.write_text("ok", encoding="utf-8")
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "stale_running_recovery_queue",
            "controls": {"mode": "running", "max_concurrency": {"local_mlx": 1}},
            "experiments": [
                {
                    "id": "candidate",
                    "steps": [
                        {
                            "id": "first",
                            "resources": {"kind": "local_mlx"},
                            "command": [sys.executable, "-c", "print('first')"],
                            "postconditions": [
                                {"type": "path_exists", "path": marker.name}
                            ],
                        },
                        {
                            "id": "second",
                            "resources": {"kind": "local_mlx"},
                            "command": [sys.executable, "-c", "print('second')"],
                        },
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                resource_kind = 'local_mlx',
                last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                json.dumps({"pid": 999999, "parent_pid": 999999}),
                "stale_running_recovery_queue",
                "candidate",
                "first",
            ),
        )
        conn.commit()

        assert ready_steps(conn, queue, repo_root=tmp_path) == []
        result = reconcile_stale_running_steps(conn, queue, repo_root=tmp_path)
        summary = queue_summary(conn, queue, repo_root=tmp_path)

    assert result["reconciled_step_count"] == 1
    assert result["reconciled_steps"][0]["status"] == "succeeded"
    assert summary["status_counts"] == {"queued": 1, "succeeded": 1}
    assert summary["ready_steps"][0]["step_id"] == "second"


def test_worker_reclaims_failed_stale_running_step_before_launching_next(
    tmp_path: Path,
) -> None:
    second_marker = tmp_path / "second.done"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "stale_running_worker_queue",
            "controls": {"mode": "running", "max_concurrency": {"local_mlx": 1}},
            "experiments": [
                {
                    "id": "candidate",
                    "steps": [
                        {
                            "id": "first",
                            "resources": {"kind": "local_mlx"},
                            "command": [sys.executable, "-c", "print('first')"],
                            "postconditions": [
                                {"type": "path_exists", "path": "missing.done"}
                            ],
                        },
                        {
                            "id": "second",
                            "resources": {"kind": "local_mlx"},
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib; "
                                    f"pathlib.Path({str(second_marker)!r}).write_text('ok')"
                                ),
                            ],
                            "postconditions": [
                                {"type": "path_exists", "path": str(second_marker)}
                            ],
                        },
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                attempts = 1,
                resource_kind = 'local_mlx',
                last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                json.dumps({"pid": 999999, "parent_pid": 999999}),
                "stale_running_worker_queue",
                "candidate",
                "first",
            ),
        )
        conn.commit()

        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=1,
            max_parallel=0,
            poll_interval_seconds=0.01,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )
        summary = queue_summary(conn, queue, repo_root=tmp_path)

    assert result["success_count"] == 1
    assert result["stale_running_reconciliations"][0]["reconciled_step_count"] == 1
    assert result["stale_running_reconciliations"][0]["reconciled_steps"][0]["status"] == "failed"
    assert summary["status_counts"] == {"failed": 1, "succeeded": 1}


def test_experiment_queue_stale_ready_step_refuses_definition_drift(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "queue.sqlite"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    def queue_for(path: Path) -> dict[str, object]:
        return normalize_queue_definition(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "stale_definition_claim_queue",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "candidate",
                        "steps": [
                            {
                                "id": "same_step",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"import pathlib; pathlib.Path({str(path)!r}).write_text('ok')",
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    with connect_state(state_path) as conn_a, connect_state(state_path) as conn_b:
        original = queue_for(first)
        initialize_queue_state(conn_a, original)
        stale_ready = ready_steps(conn_a, original)[0]
        changed = queue_for(second)
        initialize_queue_state(conn_b, changed)

        result = run_ready_step(
            conn_a,
            original,
            stale_ready,
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )

        assert result["executed"] is False
        assert result["claim_refused"] is True
        assert result["claim_refused_reason"] == "definition_changed"
        summary = queue_summary(conn_a, changed)
        assert summary["status_counts"] == {"queued": 1}
        assert summary["steps"][0]["attempts"] == 0
        assert not first.exists()
        assert not second.exists()


def test_experiment_queue_claim_refuses_missing_step_hashes(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        stale_ready = ready_steps(conn, queue)[0]
        conn.execute(
            """
            UPDATE step_state
            SET command_hash = NULL
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            ("unit_queue", "candidate_a", "materialize"),
        )
        conn.commit()

        result = run_ready_step(
            conn,
            queue,
            stale_ready,
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )

        assert result["executed"] is False
        assert result["claim_refused"] is True
        assert result["claim_refused_reason"] == "definition_hash_missing"
        summary = queue_summary(conn, queue)
        first_step = next(row for row in summary["steps"] if row["step_id"] == "materialize")
        assert first_step["status"] == "queued"
        assert first_step["attempts"] == 0


def test_experiment_queue_worker_runs_bounded_local_steps(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=2,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )
        assert result["stop_reason"] == "max_steps_reached"
        assert result["steps_started"] == 2
        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert queue_summary(conn, queue)["status_counts"] == {"queued": 1, "succeeded": 2}

        events = [
            row["event_type"]
            for row in conn.execute("SELECT event_type FROM queue_events ORDER BY id").fetchall()
        ]
        assert "worker_started" in events
        assert "worker_stopped" in events


def test_experiment_queue_worker_limits_started_experiments(tmp_path: Path) -> None:
    first_a = tmp_path / "candidate_a_first.txt"
    second_a = tmp_path / "candidate_a_second.txt"
    first_b = tmp_path / "candidate_b_first.txt"
    second_b = tmp_path / "candidate_b_second.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "candidate_window_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_cpu": 1},
            },
            "experiments": [
                {
                    "id": "candidate_a",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "first",
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(first_a)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": first_a.name}],
                        },
                        {
                            "id": "second",
                            "requires": ["first"],
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(second_a)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": second_a.name}],
                        },
                    ],
                },
                {
                    "id": "candidate_b",
                    "priority": 2,
                    "steps": [
                        {
                            "id": "first",
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(first_b)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": first_b.name}],
                        },
                        {
                            "id": "second",
                            "requires": ["first"],
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(second_b)!r}).write_text('ok')",
                            ],
                            "postconditions": [{"type": "path_exists", "path": second_b.name}],
                        },
                    ],
                },
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)

        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=10,
            max_experiments=1,
            idle_sleep_seconds=0,
            max_idle_cycles=1,
            log_root=tmp_path / "logs",
        )

        assert result["stop_reason"] == "idle_limit_reached"
        assert result["steps_started"] == 2
        assert result["started_experiment_ids"] == ["candidate_a"]
        assert first_a.read_text() == "ok"
        assert second_a.read_text() == "ok"
        assert not first_b.exists()
        assert not second_b.exists()
        assert queue_summary(conn, queue)["status_counts"] == {"queued": 2, "succeeded": 2}


def test_experiment_queue_worker_runs_independent_steps_in_parallel(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "parallel_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_cpu": 2},
            },
            "experiments": [
                {
                    "id": "candidate_parallel",
                    "steps": [
                        {
                            "id": "first",
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; "
                                    "time.sleep(0.4); "
                                    f"pathlib.Path({str(first)!r}).write_text('ok')"
                                ),
                            ],
                        },
                        {
                            "id": "second",
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; "
                                    "time.sleep(0.4); "
                                    f"pathlib.Path({str(second)!r}).write_text('ok')"
                                ),
                            ],
                        },
                    ],
                }
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=2,
            max_parallel=2,
            poll_interval_seconds=0.01,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )

        assert result["max_parallel"] == 2
        assert result["steps_started"] == 2
        assert result["success_count"] == 2
        assert first.read_text() == "ok"
        assert second.read_text() == "ok"
        assert queue_summary(conn, queue)["status_counts"] == {"succeeded": 2}

        started = [
            (int(row["id"]), json.loads(row["payload_json"]))
            for row in conn.execute(
                "SELECT id, payload_json FROM queue_events WHERE event_type = 'step_process_started'"
            ).fetchall()
        ]
        assert len(started) == 2
        assert all(isinstance(event.get("pid"), int) for _id, event in started)
        assert all(event.get("worker_run_id") for _id, event in started)
        first_success_id = min(
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM queue_events WHERE event_type = 'step_succeeded'"
            ).fetchall()
        )
        assert max(event_id for event_id, _event in started) < first_success_id


def test_experiment_queue_worker_respects_resource_limits_under_parallelism(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "parallel_resource_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_mlx": 1},
            },
            "experiments": [
                {
                    "id": "candidate_parallel_resource",
                    "steps": [
                        {
                            "id": "first",
                            "resources": {"kind": "local_mlx"},
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; "
                                    "time.sleep(0.25); "
                                    f"pathlib.Path({str(first)!r}).write_text('ok')"
                                ),
                            ],
                        },
                        {
                            "id": "second",
                            "resources": {"kind": "local_mlx"},
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; "
                                    "time.sleep(0.25); "
                                    f"pathlib.Path({str(second)!r}).write_text('ok')"
                                ),
                            ],
                        },
                    ],
                }
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=2,
            max_parallel=2,
            poll_interval_seconds=0.01,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )

        assert result["success_count"] == 2
        assert first.read_text() == "ok"
        assert second.read_text() == "ok"
        rows = conn.execute(
            """
            SELECT id, step_id, event_type
            FROM queue_events
            WHERE event_type IN ('step_process_started', 'step_succeeded')
            ORDER BY id
            """
        ).fetchall()
        ordered = [(str(row["step_id"]), str(row["event_type"])) for row in rows]
        assert ordered == [
            ("first", "step_process_started"),
            ("first", "step_succeeded"),
            ("second", "step_process_started"),
            ("second", "step_succeeded"),
        ]


def test_experiment_queue_worker_auto_parallelism_sums_local_resource_limits(
    tmp_path: Path,
) -> None:
    cpu_a = tmp_path / "cpu_a.txt"
    cpu_b = tmp_path / "cpu_b.txt"
    mlx = tmp_path / "mlx.txt"
    cloud = tmp_path / "cloud.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "auto_parallel_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {
                    "local_cpu": 2,
                    "local_mlx": 1,
                    "modal_gpu": 4,
                },
            },
            "experiments": [
                {
                    "id": "candidate_auto_parallel",
                    "steps": [
                        {
                            "id": "cpu_a",
                            "resources": {"kind": "local_cpu"},
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; time.sleep(0.2); "
                                    f"pathlib.Path({str(cpu_a)!r}).write_text('ok')"
                                ),
                            ],
                        },
                        {
                            "id": "cpu_b",
                            "resources": {"kind": "local_cpu"},
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; time.sleep(0.2); "
                                    f"pathlib.Path({str(cpu_b)!r}).write_text('ok')"
                                ),
                            ],
                        },
                        {
                            "id": "mlx",
                            "resources": {"kind": "local_mlx"},
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import pathlib, time; time.sleep(0.2); "
                                    f"pathlib.Path({str(mlx)!r}).write_text('ok')"
                                ),
                            ],
                        },
                        {
                            "id": "cloud",
                            "resources": {"kind": "modal_gpu"},
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(cloud)!r}).write_text('ok')",
                            ],
                        },
                    ],
                }
            ],
        }
    )

    assert worker_resource_limits(queue) == {"local_cpu": 2, "local_mlx": 1}
    assert queue_resource_kinds(queue) == ["local_cpu", "local_mlx", "modal_gpu"]
    assert resolve_worker_max_parallel(queue, 0) == (
        3,
        {"local_cpu": 2, "local_mlx": 1},
    )
    assert resolve_worker_max_parallel(queue, 0, allow_cloud=True) == (
        7,
        {"local_cpu": 2, "local_mlx": 1, "modal_gpu": 4},
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=3,
            max_parallel=0,
            poll_interval_seconds=0.01,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )

        assert result["requested_max_parallel"] == 0
        assert result["max_parallel"] == 3
        assert result["resource_limits"] == {"local_cpu": 2, "local_mlx": 1}
        assert result["success_count"] == 3
        assert result["failure_count"] == 0
        assert cpu_a.read_text() == "ok"
        assert cpu_b.read_text() == "ok"
        assert mlx.read_text() == "ok"
        assert not cloud.exists()

        started = [
            row["step_id"]
            for row in conn.execute(
                "SELECT step_id FROM queue_events WHERE event_type = 'step_process_started'"
            ).fetchall()
        ]
        assert sorted(started) == ["cpu_a", "cpu_b", "mlx"]


def test_experiment_queue_parallel_worker_finalizes_timeout_without_blocking_success(
    tmp_path: Path,
) -> None:
    fast = tmp_path / "fast.txt"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "parallel_timeout_queue",
            "controls": {
                "mode": "running",
                "max_concurrency": {"local_cpu": 2},
            },
            "experiments": [
                {
                    "id": "candidate_parallel_timeout",
                    "steps": [
                        {
                            "id": "slow",
                            "command": [sys.executable, "-c", "import time; time.sleep(5)"],
                            "timeout_seconds": 1,
                        },
                        {
                            "id": "fast",
                            "command": [
                                sys.executable,
                                "-c",
                                f"import pathlib; pathlib.Path({str(fast)!r}).write_text('ok')",
                            ],
                        },
                    ],
                }
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=2,
            max_parallel=2,
            poll_interval_seconds=0.01,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )

        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert fast.read_text() == "ok"
        assert queue_summary(conn, queue)["status_counts"] == {"failed": 1, "succeeded": 1}
        timed_out = next(
            item
            for item in result["step_results"]
            if item["ready_step"]["step_id"] == "slow"
        )
        assert timed_out["returncode"] == 124
        assert timed_out["timed_out"] is True


def test_experiment_queue_worker_records_failure_telemetry(tmp_path: Path) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "failure_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "candidate_failure",
                    "steps": [
                        {
                            "id": "bad",
                            "command": [sys.executable, "-c", "import sys; sys.exit(7)"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                }
            ],
        }
    )
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=1,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )
        assert result["failure_count"] == 1
        assert result["step_results"][0]["returncode"] == 7
        assert queue_summary(conn, queue)["status_counts"] == {"failed": 1}

        events = [
            row["event_type"]
            for row in conn.execute("SELECT event_type FROM queue_events ORDER BY id").fetchall()
        ]
        assert "step_failed" in events
        assert "worker_step_failed" in events


def test_experiment_queue_execute_requires_canonical_state_path(tmp_path: Path) -> None:
    canonical = default_state_path(tmp_path, "unit_queue")
    assert_canonical_state_for_execution(tmp_path, "unit_queue", canonical)
    assert_canonical_state_for_execution(
        tmp_path,
        "unit_queue",
        tmp_path / "isolated.sqlite",
        allow_noncanonical_state=True,
    )

    with pytest.raises(ExperimentQueueError, match="noncanonical state path"):
        assert_canonical_state_for_execution(tmp_path, "unit_queue", tmp_path / "isolated.sqlite")


def test_experiment_queue_worker_refuses_orphaned_reroute_state(tmp_path: Path) -> None:
    first_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "reroute_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "old_candidate",
                    "steps": [{"id": "plan", "command": [sys.executable, "-c", "print('old')"]}],
                }
            ],
        }
    )
    second_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "reroute_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "new_candidate",
                    "steps": [{"id": "plan", "command": [sys.executable, "-c", "print('new')"]}],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, first_queue)
        initialize_queue_state(conn, second_queue)

        with pytest.raises(ExperimentQueueError, match="orphaned step"):
            assert_no_orphaned_steps_for_execution(conn, second_queue)
        assert_no_orphaned_steps_for_execution(conn, second_queue, allow_orphaned_state=True)

        with pytest.raises(ExperimentQueueError, match="orphaned step"):
            run_queue_worker(
                conn,
                second_queue,
                repo_root=tmp_path,
                execute=False,
                max_steps=1,
                idle_sleep_seconds=0,
                max_idle_cycles=0,
            )


def test_experiment_queue_can_retire_blocking_orphaned_steps(tmp_path: Path) -> None:
    first_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "reroute_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "old_candidate",
                    "steps": [{"id": "plan", "command": [sys.executable, "-c", "print('old')"]}],
                }
            ],
        }
    )
    second_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "reroute_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "new_candidate",
                    "steps": [{"id": "plan", "command": [sys.executable, "-c", "print('new')"]}],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, first_queue)
        initialize_queue_state(conn, second_queue)
        retired = retire_orphaned_steps(conn, second_queue, reason="rerouted")

        assert retired == [
            {
                "experiment_id": "old_candidate",
                "step_id": "plan",
                "previous_status": "queued",
                "new_status": "skipped",
            }
        ]
        assert_no_orphaned_steps_for_execution(conn, second_queue)
        summary = queue_summary(conn, second_queue)
        assert summary["orphaned_steps"][0]["status"] == "skipped"


def test_experiment_queue_cli_reconcile_state_writes_audit_artifact(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    state = tmp_path / "queue.sqlite"
    first_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "cli_reconcile",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "old_candidate",
                    "steps": [
                        {
                            "id": "plan",
                            "command": [sys.executable, "-c", "print('old')"],
                        }
                    ],
                }
            ],
        }
    )
    second_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "cli_reconcile",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "new_candidate",
                    "steps": [
                        {
                            "id": "plan",
                            "command": [sys.executable, "-c", "print('new')"],
                        }
                    ],
                }
            ],
        }
    )
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(second_queue), encoding="utf-8")
    with connect_state(state) as conn:
        initialize_queue_state(conn, first_queue)

    output = tmp_path / "reconciliation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "reconcile-state",
            "--reason",
            "unit test queue reroute state reconciliation",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "experiment_queue_state_reconciliation.v1"
    assert payload["blocking_orphan_count_before"] == 1
    assert payload["blocking_orphan_count_after"] == 0
    assert payload["retired_step_count"] == 1
    assert artifact["retired_steps"] == [
        {
            "experiment_id": "old_candidate",
            "step_id": "plan",
            "previous_status": "queued",
            "new_status": "skipped",
        }
    ]
    assert artifact["before"]["orphaned_step_count"] == 1
    assert artifact["after"]["orphaned_step_count"] == 1
    assert artifact["after"]["orphaned_steps"][0]["status"] == "skipped"
    assert artifact["after"]["status_counts"] == {"queued": 1}
    assert artifact["score_claim"] is False
    assert artifact["promotion_eligible"] is False


def test_experiment_queue_false_authority_postcondition_blocks_leaks(tmp_path: Path) -> None:
    advisory = tmp_path / "advisory.json"
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "authority_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "candidate",
                    "steps": [
                        {
                            "id": "local_advisory",
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "import json, pathlib; "
                                    f"pathlib.Path({str(advisory)!r}).write_text("
                                    "json.dumps({"
                                    "'score_claim': False, "
                                    "'promotion_eligible': 'true', "
                                    "'rank_or_kill_eligible': False, "
                                    "'ready_for_exact_eval_dispatch': 1, "
                                    "'score_axis': 'contest_cpu'"
                                    "}))"
                                ),
                            ],
                            "postconditions": [
                                {
                                    "type": "json_false_authority",
                                    "path": advisory.name,
                                    "required_false": [
                                        "score_claim",
                                        "promotion_eligible",
                                        "rank_or_kill_eligible",
                                    ],
                                    "false_or_missing": ["ready_for_exact_eval_dispatch"],
                                    "axis_key": "score_axis",
                                    "axis_equals": "cpu_advisory",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_ready_step(
            conn,
            queue,
            ready_steps(conn, queue)[0],
            repo_root=tmp_path,
            execute=True,
            log_root=tmp_path / "logs",
        )

    assert result["succeeded"] is False
    assert result["failed_postconditions"][0]["type"] == "json_false_authority"


@pytest.mark.parametrize(
    "alias_field",
    [
        "score_claim_valid",
        "promotable",
        "exact_cuda_auth_eval",
        "charged_bits_changed",
        "dispatch_packet_ready",
        "dispatch_ready",
        "exact_eval_ready",
    ],
)
def test_json_false_authority_defaults_block_truthy_alias_fields(
    tmp_path: Path,
    alias_field: str,
) -> None:
    advisory = tmp_path / "advisory.json"
    payload = {
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        alias_field: True,
    }
    advisory.write_text(json.dumps(payload), encoding="utf-8")

    condition = {
        "type": "json_false_authority",
        "path": advisory.name,
    }

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_json_false_authority_explicit_false_or_missing_override_is_exact(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "advisory.json"
    advisory.write_text(
        json.dumps(
            {
                "score_claim": False,
                "score_claim_valid": True,
            }
        ),
        encoding="utf-8",
    )

    condition = {
        "type": "json_false_authority",
        "path": advisory.name,
        "required_false": ["score_claim"],
        "false_or_missing": [],
    }

    assert _condition_passes(condition, repo_root=tmp_path) is True


def test_json_false_authority_rejects_receiver_contract_without_live_proof(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "advisory.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    advisory.write_text(
        json.dumps(
            {
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    condition = {
        "type": "json_false_authority",
        "path": advisory.name,
    }

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_json_false_authority_accepts_receiver_contract_with_live_proof(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "advisory.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    advisory.write_text(
        json.dumps(
            {
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": proof.name,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    condition = {
        "type": "json_false_authority",
        "path": advisory.name,
    }

    assert _condition_passes(condition, repo_root=tmp_path) is True


def test_json_false_authority_rejects_receiver_proof_authority_leak(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "advisory.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": True,
            }
        ),
        encoding="utf-8",
    )
    advisory.write_text(
        json.dumps(
            {
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": proof.name,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    condition = {
        "type": "json_false_authority",
        "path": advisory.name,
    }

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_json_false_authority_rejects_receiver_proof_archive_mismatch(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "advisory.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    other_archive = _postcondition_artifact(tmp_path / "other.zip", b"other")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": other_archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    advisory.write_text(
        json.dumps(
            {
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": proof.name,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    condition = {
        "type": "json_false_authority",
        "path": advisory.name,
    }

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_jsonl_false_authority_rejects_custody_row_with_only_proof_present(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "observations.jsonl"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    advisory.write_text(
        json.dumps(
            {
                "schema": "mlx_dynamic_sweep_observation.v1",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    condition = {
        "type": "jsonl_false_authority",
        "path": advisory.name,
        "schema_equals": "mlx_dynamic_sweep_observation.v1",
    }

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_json_array_contains_postcondition_checks_dotted_list_value(
    tmp_path: Path,
) -> None:
    advisory = tmp_path / "preprocess.json"
    advisory.write_text(
        json.dumps(
            {
                "exact_readiness_refusal": {
                    "ready": False,
                    "blockers": [
                        "fixture_exact_readiness_blocker",
                        "requires_exact_cpu_cuda_auth_eval_before_score_claim",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    condition = {
        "type": "json_array_contains",
        "path": advisory.name,
        "key": "exact_readiness_refusal.blockers",
        "contains": "fixture_exact_readiness_blocker",
    }
    missing = {**condition, "contains": "missing_blocker"}

    assert _condition_passes(condition, repo_root=tmp_path) is True
    assert _condition_passes(missing, repo_root=tmp_path) is False


def test_jsonl_false_authority_blocks_truthy_rows(tmp_path: Path) -> None:
    advisory = tmp_path / "observations.jsonl"
    advisory.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema": "mlx_dynamic_sweep_observation.v1",
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ),
                json.dumps(
                    {
                        "schema": "mlx_dynamic_sweep_observation.v1",
                        "score_claim": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    condition = {
        "type": "jsonl_false_authority",
        "path": advisory.name,
        "schema_equals": "mlx_dynamic_sweep_observation.v1",
    }

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_jsonl_false_authority_accepts_all_false_rows(tmp_path: Path) -> None:
    advisory = tmp_path / "observations.jsonl"
    advisory.write_text(
        json.dumps(
            {
                "schema": "mlx_dynamic_sweep_observation.v1",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    condition = {
        "type": "jsonl_false_authority",
        "path": advisory.name,
        "schema_equals": "mlx_dynamic_sweep_observation.v1",
    }

    assert _condition_passes(condition, repo_root=tmp_path) is True


def _postcondition_artifact(path: Path, payload: bytes = b"artifact") -> dict[str, object]:
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _postcondition_runtime(path: Path) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    inflate = path / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | 0o100)
    runtime_tree_sha256 = tree_sha256(path)
    return {
        "candidate_runtime_dir": str(path),
        "candidate_runtime_tree_sha256": runtime_tree_sha256,
        "expected_runtime_tree_sha256": runtime_tree_sha256,
    }


def test_json_completion_contract_required_less_than_is_strict(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    condition = {
        "type": "json_completion_contract",
        "path": manifest.name,
        "required_less_than": [
            {"left": "candidate_archive_bytes", "right": "source_archive_bytes"}
        ],
    }

    manifest.write_text(
        json.dumps({"candidate_archive_bytes": 99, "source_archive_bytes": 100}),
        encoding="utf-8",
    )
    assert _condition_passes(condition, repo_root=tmp_path) is True

    manifest.write_text(
        json.dumps({"candidate_archive_bytes": 100, "source_archive_bytes": 100}),
        encoding="utf-8",
    )
    assert _condition_passes(condition, repo_root=tmp_path) is False

    manifest.write_text(
        json.dumps({"candidate_archive_bytes": "bad", "source_archive_bytes": 100}),
        encoding="utf-8",
    )
    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_json_completion_contract_required_runtime_identity_is_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    runtime = _postcondition_runtime(tmp_path / "candidate_runtime")
    condition = {
        "type": "json_completion_contract",
        "path": manifest.name,
        "required_runtime_adapter_identity": True,
    }

    manifest.write_text(json.dumps(runtime), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is False

    manifest.write_text(
        json.dumps({"runtime_adapter_ready": True, **runtime}),
        encoding="utf-8",
    )
    assert _condition_passes(condition, repo_root=tmp_path) is True

    opt_out = dict(condition)
    opt_out["required_runtime_adapter_identity"] = False
    manifest.write_text(json.dumps(runtime), encoding="utf-8")
    assert _condition_passes(opt_out, repo_root=tmp_path) is True


def test_json_completion_contract_rejects_receiver_contract_without_live_proof(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    condition = {
        "type": "json_completion_contract",
        "path": manifest.name,
        "required_true": ["receiver_contract_satisfied"],
    }
    manifest.write_text(
        json.dumps(
            {
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_json_completion_contract_accepts_receiver_contract_with_live_proof(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    condition = {
        "type": "json_completion_contract",
        "path": manifest.name,
        "required_true": ["receiver_contract_satisfied"],
    }
    manifest.write_text(
        json.dumps(
            {
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_path": proof.name,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _condition_passes(condition, repo_root=tmp_path) is True


def test_json_completion_contract_accepts_shell_parity_as_companion_proof(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    runtime = _postcondition_runtime(tmp_path / "candidate_runtime")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "runtime_adapter_ready": True,
                "passed": True,
                **runtime,
            }
        ),
        encoding="utf-8",
    )
    parity = tmp_path / "shell_parity.json"
    parity.write_text(
        json.dumps(
            {
                "schema": "shell_inflate_parity_proof_v2",
                "full_frame_inflate_output_parity_claim": True,
                "cmp_equal": True,
                "output_sha256_match": True,
                "right": {"archive_sha256": archive["sha256"]},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    condition = {
        "type": "json_completion_contract",
        "path": manifest.name,
        "required_true": ["receiver_contract_satisfied"],
        "required_runtime_adapter_identity": True,
    }
    manifest.write_text(
        json.dumps(
            {
                "candidate_archive": archive,
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "runtime_consumption_proof_path": proof.name,
                "full_frame_inflate_parity_proof_path": parity.name,
                "receiver_verification": {
                    "receiver_contract_satisfied": True,
                    "proof_present": True,
                },
                **runtime,
            }
        ),
        encoding="utf-8",
    )

    assert _condition_passes(condition, repo_root=tmp_path) is True


def test_materializer_chain_complete_allows_downstream_readiness_blockers(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "chain.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = _postcondition_artifact(tmp_path / "receiver_proof.json", b"proof")
    runtime = _postcondition_runtime(tmp_path / "candidate_runtime")
    payload = {
        "schema": "chain.v1",
        "candidate_archive": archive,
        "candidate_archive_sha256": archive["sha256"],
        "candidate_archive_bytes": archive["bytes"],
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        **runtime,
        "readiness_blockers": [
            "candidate_inflate_output_parity_missing",
            "exact_auth_eval_required_before_score_claim",
        ],
        "artifacts": {"receiver_proof": proof},
        "chain_steps": [{"status": "succeeded", "artifact": proof}],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    condition = {
        "type": "materializer_chain_complete",
        "path": manifest.name,
        "schema": "chain.v1",
    }

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is True

    strict_condition = dict(condition)
    strict_condition["forbid_readiness_blockers"] = True
    assert _condition_passes(strict_condition, repo_root=tmp_path) is False


def test_materializer_chain_complete_defaults_block_truthy_alias_fields(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "chain.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = _postcondition_artifact(tmp_path / "receiver_proof.json", b"proof")
    runtime = _postcondition_runtime(tmp_path / "candidate_runtime")
    base_payload = {
        "schema": "chain.v1",
        "candidate_archive": archive,
        "candidate_archive_sha256": archive["sha256"],
        "candidate_archive_bytes": archive["bytes"],
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        **runtime,
        "readiness_blockers": ["exact_auth_eval_required_before_score_claim"],
        "artifacts": {"receiver_proof": proof},
        "chain_steps": [{"status": "succeeded", "artifact": proof}],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    condition = {
        "type": "materializer_chain_complete",
        "path": manifest.name,
        "schema": "chain.v1",
    }

    for alias_field in (
        "score_claim_valid",
        "promotable",
        "exact_cuda_auth_eval",
        "charged_bits_changed",
        "dispatch_ready",
        "exact_eval_ready",
    ):
        manifest.write_text(
            json.dumps({**base_payload, alias_field: True}),
            encoding="utf-8",
        )
        assert _condition_passes(condition, repo_root=tmp_path) is False, alias_field

    manifest.write_text(json.dumps(base_payload), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is True


def test_materializer_chain_complete_requires_runtime_identity_by_default(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "chain.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = _postcondition_artifact(tmp_path / "receiver_proof.json", b"proof")
    payload = {
        "schema": "chain.v1",
        "candidate_archive": archive,
        "candidate_archive_sha256": archive["sha256"],
        "candidate_archive_bytes": archive["bytes"],
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "readiness_blockers": ["exact_auth_eval_required_before_score_claim"],
        "artifacts": {"receiver_proof": proof},
        "chain_steps": [{"status": "succeeded", "artifact": proof}],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    condition = {
        "type": "materializer_chain_complete",
        "path": manifest.name,
        "schema": "chain.v1",
    }

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is False

    opt_out = dict(condition)
    opt_out["required_runtime_adapter_identity"] = False
    assert _condition_passes(opt_out, repo_root=tmp_path) is True


def test_materializer_chain_complete_explicit_false_or_missing_override_is_exact(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "chain.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = _postcondition_artifact(tmp_path / "receiver_proof.json", b"proof")
    runtime = _postcondition_runtime(tmp_path / "candidate_runtime")
    payload = {
        "schema": "chain.v1",
        "candidate_archive": archive,
        "candidate_archive_sha256": archive["sha256"],
        "candidate_archive_bytes": archive["bytes"],
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        **runtime,
        "readiness_blockers": [],
        "artifacts": {"receiver_proof": proof},
        "chain_steps": [{"status": "succeeded", "artifact": proof}],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "charged_bits_changed": True,
    }
    condition = {
        "type": "materializer_chain_complete",
        "path": manifest.name,
        "schema": "chain.v1",
        "false_or_missing": [],
    }

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is True


def test_materializer_chain_complete_requires_serialized_archive_saving_status(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "chain.json"
    archive = _postcondition_artifact(tmp_path / "candidate.zip", b"candidate")
    proof = _postcondition_artifact(tmp_path / "receiver_proof.json", b"proof")
    runtime = _postcondition_runtime(tmp_path / "candidate_runtime")
    payload = {
        "schema": "chain.v1",
        "candidate_archive": archive,
        "candidate_archive_sha256": archive["sha256"],
        "candidate_archive_bytes": archive["bytes"],
        "source_archive_bytes": int(archive["bytes"]) + 1,
        "serialized_archive_delta": {"status": "realized_saving"},
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        **runtime,
        "readiness_blockers": [],
        "artifacts": {"receiver_proof": proof},
        "chain_steps": [{"status": "succeeded", "artifact": proof}],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    condition = {
        "type": "materializer_chain_complete",
        "path": manifest.name,
        "schema": "chain.v1",
        "required_serialized_archive_saving": True,
    }

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is True

    missing_delta = dict(payload)
    missing_delta.pop("serialized_archive_delta")
    manifest.write_text(json.dumps(missing_delta), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is False

    wrong_status = dict(payload)
    wrong_status["serialized_archive_delta"] = {"status": "realized_cost"}
    manifest.write_text(json.dumps(wrong_status), encoding="utf-8")
    assert _condition_passes(condition, repo_root=tmp_path) is False


def test_experiment_queue_cli_requires_noncanonical_state_rationale(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "cli_noncanonical",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "candidate",
                        "steps": [
                            {
                                "id": "noop",
                                "command": [sys.executable, "-c", "print('noop')"],
                            }
                        ],
                    }
                ],
            }
        )
    )
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "experiment_queue.py"),
        "--queue",
        str(queue_path),
        "--state",
        str(tmp_path / "isolated.sqlite"),
        "run-worker",
        "--execute",
        "--max-steps",
        "0",
        "--idle-sleep-seconds",
        "0",
        "--max-idle-cycles",
        "0",
    ]

    refused = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    assert refused.returncode == 2
    assert "noncanonical state path" in refused.stderr

    allowed = subprocess.run(
        [*cmd, "--noncanonical-state-rationale", "isolated cli unit test state"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "isolated cli unit test state" in allowed.stdout


def test_experiment_queue_cli_run_worker_exposes_max_experiments(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    queue_path = tmp_path / "queue.json"
    state = tmp_path / ".omx" / "state" / "experiment_queue_cli_max_experiments.sqlite"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    queue_path.write_text(
        json.dumps(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "cli_max_experiments",
                "controls": {
                    "mode": "running",
                    "max_concurrency": {"local_cpu": 2},
                },
                "experiments": [
                    {
                        "id": "candidate_a",
                        "priority": 1,
                        "steps": [
                            {
                                "id": "write",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"import pathlib; pathlib.Path({str(first)!r}).write_text('a')",
                                ],
                            }
                        ],
                    },
                    {
                        "id": "candidate_b",
                        "priority": 2,
                        "steps": [
                            {
                                "id": "write",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"import pathlib; pathlib.Path({str(second)!r}).write_text('b')",
                                ],
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "run-worker",
            "--execute",
            "--max-steps",
            "2",
            "--max-parallel",
            "2",
            "--max-experiments",
            "1",
            "--idle-sleep-seconds",
            "0",
            "--max-idle-cycles",
            "0",
            "--noncanonical-state-rationale",
            "isolated cli unit test state",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["max_experiments"] == 1
    assert payload["steps_started"] == 1
    assert first.exists()
    assert not second.exists()


def test_experiment_queue_cli_performance_is_read_only_on_definition_drift(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    marker = tmp_path / "marker.txt"

    def queue_for(command_text: str) -> dict[str, object]:
        return normalize_queue_definition(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "cli_performance_readonly",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "candidate",
                        "steps": [
                            {
                                "id": "write",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"import pathlib; pathlib.Path({str(marker)!r}).write_text({command_text!r})",
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    state = tmp_path / "queue.sqlite"
    queue_v1 = queue_for("v1")
    queue_v2 = queue_for("v2")
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue_v2), encoding="utf-8")
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue_v1)
        result = run_queue_worker(
            conn,
            queue_v1,
            repo_root=tmp_path,
            execute=True,
            max_steps=1,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )
        assert result["success_count"] == 1
        before = queue_summary(conn, queue_v1)
        event_count_before = conn.execute("SELECT COUNT(*) AS count FROM queue_events").fetchone()["count"]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "performance",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with connect_state(state) as conn:
        after = queue_summary(conn, queue_v2)
        event_count_after = conn.execute("SELECT COUNT(*) AS count FROM queue_events").fetchone()["count"]
        drift = queue_definition_drift(conn, queue_v2)
    assert before["status_counts"] == {"succeeded": 1}
    assert after["status_counts"] == {"succeeded": 1}
    assert after["steps"][0]["last_event"] is not None
    assert event_count_after == event_count_before
    assert drift["changed_step_count"] == 1


def test_experiment_queue_cli_runtime_policy_writes_guarded_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "cli_runtime_policy",
            "controls": {
                "mode": "running",
                "max_concurrency": {
                    "local_cpu": 1,
                    "local_io_heavy": 8,
                    "local_mlx": 1,
                    "cloud_cpu": 3,
                },
            },
            "experiments": [
                {
                    "id": "cpu_exp",
                    "steps": [
                        {
                            "id": "cpu_step",
                            "resources": {"kind": "local_cpu"},
                            "command": [sys.executable, "-c", "print('cpu')"],
                            "timeout_seconds": 12,
                        }
                    ],
                },
                {
                    "id": "io_exp",
                    "steps": [
                        {
                            "id": "io_step",
                            "resources": {"kind": "local_io_heavy"},
                            "command": [sys.executable, "-c", "print('io')"],
                            "timeout_seconds": 50,
                        }
                    ],
                },
                {
                    "id": "mlx_exp",
                    "steps": [
                        {
                            "id": "mlx_step",
                            "resources": {"kind": "local_mlx"},
                            "command": [sys.executable, "-c", "print('mlx')"],
                            "timeout_seconds": 30,
                        }
                    ],
                },
                {
                    "id": "cloud_exp",
                    "steps": [
                        {
                            "id": "cloud_step",
                            "resources": {"kind": "cloud_cpu"},
                            "command": [sys.executable, "-c", "print('cloud')"],
                            "timeout_seconds": 99,
                        }
                    ],
                },
            ],
        }
    )
    queue_path = tmp_path / "queue.json"
    state = tmp_path / "queue.sqlite"
    policy_output = tmp_path / "runtime_policy.json"
    applied_output = tmp_path / "queue.runtime_policy_applied.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        for event_type, experiment_id, step_id, payload in (
            (
                "step_succeeded",
                "cpu_exp",
                "cpu_step",
                {"resource_kind": "local_cpu", "elapsed_seconds": 4.0},
            ),
            (
                "step_succeeded",
                "cpu_exp",
                "cpu_step",
                {"resource_kind": "local_cpu", "elapsed_seconds": 8.0},
            ),
            (
                "step_succeeded",
                "io_exp",
                "io_step",
                {"resource_kind": "local_io_heavy", "elapsed_seconds": 7.0},
            ),
            (
                "step_failed",
                "mlx_exp",
                "mlx_step",
                {
                    "resource_kind": "local_mlx",
                    "elapsed_seconds": 31.0,
                    "timed_out": True,
                },
            ),
        ):
            conn.execute(
                """
                INSERT INTO queue_events(
                    ts_utc, queue_id, experiment_id, step_id, event_type, payload_json
                )
                VALUES ('2026-05-24T00:00:00Z', ?, ?, ?, ?, ?)
                """,
                (
                    "cli_runtime_policy",
                    experiment_id,
                    step_id,
                    event_type,
                    json.dumps(payload),
                ),
            )
        before_event_count = conn.execute(
            "SELECT COUNT(*) AS count FROM queue_events"
        ).fetchone()["count"]
        conn.commit()

    command = [
        sys.executable,
        str(repo_root / "tools" / "experiment_queue.py"),
        "--queue",
        str(queue_path),
        "--state",
        str(state),
        "runtime-policy",
        "--cpu-count",
        "8",
        "--timeout-multiplier",
        "2",
        "--min-timeout-seconds",
        "10",
        "--max-timeout-seconds",
        "120",
        "--policy-output",
        str(policy_output),
        "--applied-queue-output",
        str(applied_output),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    policy = json.loads(policy_output.read_text(encoding="utf-8"))
    applied = json.loads(applied_output.read_text(encoding="utf-8"))
    assert stdout["schema"] == SCHEDULER_RUNTIME_POLICY_SCHEMA
    assert stdout["score_claim"] is False
    assert stdout["ready_for_exact_eval_dispatch"] is False
    assert stdout["policy_artifact"]["path"] == str(policy_output)
    assert stdout["applied_queue_artifact"]["path"] == str(applied_output)
    assert policy["recommended_max_concurrency"]["local_cpu"] == 8
    assert policy["recommended_max_concurrency"]["local_io_heavy"] == 2
    assert policy["recommended_max_concurrency"]["local_mlx"] == 1
    assert policy["recommended_max_concurrency"]["cloud_cpu"] == 3
    assert policy["recommended_timeout_seconds_by_resource"]["local_cpu"] == 16
    assert policy["recommended_timeout_seconds_by_resource"]["local_io_heavy"] == 50
    assert policy["recommended_timeout_seconds_by_resource"]["local_mlx"] == 62
    assert applied["controls"]["max_concurrency"]["local_cpu"] == 8
    assert applied["controls"]["max_concurrency"]["local_io_heavy"] == 2
    assert applied["controls"]["max_concurrency"]["cloud_cpu"] == 3
    assert applied["experiments"][0]["steps"][0]["timeout_seconds"] == 16
    assert applied["experiments"][2]["steps"][0]["timeout_seconds"] == 62
    with connect_state(state) as conn:
        after_event_count = conn.execute(
            "SELECT COUNT(*) AS count FROM queue_events"
        ).fetchone()["count"]
    assert after_event_count == before_event_count
    refused = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "refusing to overwrite existing artifact" in refused.stderr


def test_experiment_queue_cli_validate_reports_auto_parallelism(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "cli_auto_parallel",
                "controls": {
                    "mode": "running",
                    "max_concurrency": {
                        "local_cpu": 2,
                        "local_mlx": 1,
                        "modal_gpu": 4,
                    },
                },
                "experiments": [
                    {
                        "id": "candidate",
                        "steps": [
                            {
                                "id": "cpu",
                                "resources": {"kind": "local_cpu"},
                                "command": [sys.executable, "-c", "print('cpu')"],
                            },
                            {
                                "id": "mlx",
                                "resources": {"kind": "local_mlx"},
                                "command": [sys.executable, "-c", "print('mlx')"],
                            },
                            {
                                "id": "gpu",
                                "resources": {"kind": "modal_gpu"},
                                "command": [sys.executable, "-c", "print('gpu')"],
                            },
                        ],
                    }
                ],
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["auto_parallelism"] == {
        "local_only": {
            "max_parallel": 3,
            "resource_limits": {"local_cpu": 2, "local_mlx": 1},
        },
        "with_cloud": {
            "max_parallel": 7,
            "resource_limits": {"local_cpu": 2, "local_mlx": 1, "modal_gpu": 4},
        },
    }


def test_experiment_queue_cli_observe_and_performance_write_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    queue_path = tmp_path / "queue.json"
    state = tmp_path / "queue.sqlite"
    observe_output = tmp_path / "observation.json"
    performance_output = tmp_path / "performance.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "cli_observe_output",
                "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
                "experiments": [
                    {
                        "id": "candidate",
                        "steps": [
                            {
                                "id": "materialize",
                                "resources": {"kind": "local_cpu"},
                                "command": [sys.executable, "-c", "print('ok')"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    init = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "init",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert init.returncode == 0, init.stderr

    observe = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "observe",
            "--output",
            str(observe_output),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert observe.returncode == 0, observe.stderr
    observe_stdout = json.loads(observe.stdout)
    observe_payload = json.loads(observe_output.read_text(encoding="utf-8"))
    assert observe_stdout["artifact"]["path"] == str(observe_output)
    assert observe_payload["schema"] == "experiment_queue_observation.v1"
    assert observe_payload["status_counts"] == {"queued": 1}

    performance = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "performance",
            "--output",
            str(performance_output),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert performance.returncode == 0, performance.stderr
    performance_stdout = json.loads(performance.stdout)
    performance_payload = json.loads(performance_output.read_text(encoding="utf-8"))
    assert performance_stdout["artifact"]["path"] == str(performance_output)
    assert performance_payload["schema"] == "experiment_queue_performance_summary.v1"
    assert performance_payload["event_count"] == 0

    refused = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "observe",
            "--output",
            str(observe_output),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "refusing to overwrite existing artifact" in refused.stderr

    expected_sha = hashlib.sha256(observe_output.read_bytes()).hexdigest()
    allowed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "observe",
            "--output",
            str(observe_output),
            "--expected-output-sha256",
            expected_sha,
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr

    markdown_refused = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state),
            "observe",
            "--format",
            "markdown",
            "--output",
            str(tmp_path / "observation.md"),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert markdown_refused.returncode == 2
    assert "--output is only supported for --format json" in markdown_refused.stderr


def test_experiment_queue_worker_stops_cleanly_between_steps(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)

        def stop_after_one_success() -> bool:
            return int(queue_summary(conn, queue)["status_counts"].get("succeeded", 0)) >= 1

        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=3,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
            stop_requested=stop_after_one_success,
        )
        assert result["stop_reason"] == "stop_requested"
        assert result["steps_started"] == 1
        assert queue_summary(conn, queue)["status_counts"] == {"queued": 2, "succeeded": 1}

        events = [
            row["event_type"]
            for row in conn.execute("SELECT event_type FROM queue_events ORDER BY id").fetchall()
        ]
        assert "worker_stop_requested" in events


def test_experiment_queue_worker_can_reload_updated_definition(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    def payload(*, include_second: bool) -> dict[str, object]:
        steps: list[dict[str, object]] = [
            {
                "id": "first",
                "command": [
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(first)!r}).write_text('ok')",
                ],
            }
        ]
        if include_second:
            steps.append(
                {
                    "id": "second",
                    "requires": ["first"],
                    "command": [
                        sys.executable,
                        "-c",
                        f"import pathlib; pathlib.Path({str(second)!r}).write_text('ok')",
                    ],
                }
            )
        return {
            "schema": "experiment_queue.v1",
            "queue_id": "reload_queue",
            "controls": {"mode": "running"},
            "experiments": [{"id": "candidate_reload", "steps": steps}],
        }

    reload_calls = 0

    def reload_queue() -> dict[str, object]:
        nonlocal reload_calls
        reload_calls += 1
        return normalize_queue_definition(payload(include_second=reload_calls >= 2))

    queue = normalize_queue_definition(payload(include_second=False))
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=2,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
            reload_queue=reload_queue,
        )
        assert result["success_count"] == 2
        assert first.read_text() == "ok"
        assert second.read_text() == "ok"
        assert queue_summary(conn, reload_queue())["status_counts"] == {"succeeded": 2}


def test_experiment_queue_requeues_succeeded_step_when_definition_changes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    def queue_for(path: Path) -> dict[str, object]:
        return normalize_queue_definition(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "definition_hash_queue",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "candidate",
                        "steps": [
                            {
                                "id": "same_step",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"import pathlib; pathlib.Path({str(path)!r}).write_text('ok')",
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        queue = queue_for(first)
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=1,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )
        assert result["success_count"] == 1
        assert queue_summary(conn, queue)["status_counts"] == {"succeeded": 1}

        changed = queue_for(second)
        initialize_queue_state(conn, changed)
        summary = queue_summary(conn, changed)

        assert summary["status_counts"] == {"queued": 1}
        assert summary["ready_steps"][0]["step_id"] == "same_step"
        assert summary["steps"][0]["last_event"]["definition_changed"] is True


def test_experiment_queue_requeues_succeeded_step_when_experiment_metadata_changes(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "first.txt"

    def queue_for(source_sha: str) -> dict[str, object]:
        return normalize_queue_definition(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "metadata_hash_queue",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "candidate",
                        "metadata": {
                            "schema": "unit_metadata.v1",
                            "source_archive_sha256": source_sha,
                            "score_claim": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        },
                        "steps": [
                            {
                                "id": "same_step",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ok')",
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        queue = queue_for("a" * 64)
        initialize_queue_state(conn, queue)
        result = run_queue_worker(
            conn,
            queue,
            repo_root=tmp_path,
            execute=True,
            max_steps=1,
            idle_sleep_seconds=0,
            max_idle_cycles=0,
            log_root=tmp_path / "logs",
        )
        assert result["success_count"] == 1
        assert queue_summary(conn, queue)["status_counts"] == {"succeeded": 1}

        changed = queue_for("b" * 64)
        drift = queue_definition_drift(conn, changed)
        assert drift["changed_step_count"] == 1
        assert drift["changed_steps"][0]["step_id"] == "same_step"

        initialize_queue_state(conn, changed)
        summary = queue_summary(conn, changed)

        assert summary["status_counts"] == {"queued": 1}
        assert summary["steps"][0]["last_event"]["definition_changed"] is True


def test_experiment_queue_refuses_definition_change_while_running(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running'
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            ("unit_queue", "candidate_a", "materialize"),
        )
        conn.commit()
        changed = _queue(tmp_path)
        changed["experiments"][0]["steps"][0]["command"] = [
            sys.executable,
            "-c",
            "print('changed')",
        ]

        with pytest.raises(ExperimentQueueError, match="definition changed while running"):
            initialize_queue_state(conn, changed)


def test_experiment_queue_refuses_upstream_drift_without_partial_mutation(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded'
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            ("unit_queue", "candidate_a", "materialize"),
        )
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running'
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            ("unit_queue", "candidate_a", "local_advisory"),
        )
        conn.commit()
        event_count_before = conn.execute("SELECT COUNT(*) AS count FROM queue_events").fetchone()["count"]
        changed = _queue(tmp_path)
        changed["experiments"][0]["steps"][0]["command"] = [
            sys.executable,
            "-c",
            "print('changed upstream')",
        ]

        with pytest.raises(ExperimentQueueError, match="depends on changed"):
            initialize_queue_state(conn, changed)
        rows = {
            row["step_id"]: row
            for row in conn.execute(
                """
                SELECT step_id, status, last_event_json
                FROM step_state
                WHERE queue_id = ? AND experiment_id = ?
                """,
                ("unit_queue", "candidate_a"),
            ).fetchall()
        }
        event_count_after = conn.execute("SELECT COUNT(*) AS count FROM queue_events").fetchone()["count"]

    assert rows["materialize"]["status"] == "succeeded"
    assert rows["local_advisory"]["status"] == "running"
    assert rows["materialize"]["last_event_json"] is None
    assert event_count_after == event_count_before


def test_experiment_queue_summary_separates_orphaned_reroute_rows(
    tmp_path: Path,
) -> None:
    first_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "reroute_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "old_candidate",
                    "steps": [
                        {
                            "id": "plan",
                            "command": [sys.executable, "-c", "print('old')"],
                        }
                    ],
                }
            ],
        }
    )
    second_queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "reroute_queue",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "new_candidate",
                    "steps": [
                        {
                            "id": "plan",
                            "command": [sys.executable, "-c", "print('new')"],
                        }
                    ],
                }
            ],
        }
    )

    with connect_state(tmp_path / "queue.sqlite") as conn:
        initialize_queue_state(conn, first_queue)
        initialize_queue_state(conn, second_queue)
        summary = queue_summary(conn, second_queue)

    assert summary["step_count"] == 1
    assert summary["orphaned_step_count"] == 1
    assert summary["status_counts"] == {"queued": 1}
    assert summary["steps"][0]["experiment_id"] == "new_candidate"
    assert summary["orphaned_steps"][0]["experiment_id"] == "old_candidate"
