# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    assert_canonical_state_for_execution,
    assert_no_orphaned_steps_for_execution,
    connect_state,
    default_state_path,
    initialize_queue_state,
    normalize_queue_definition,
    queue_resource_kinds,
    queue_summary,
    ready_steps,
    resolve_worker_max_parallel,
    retire_orphaned_steps,
    rewind_step,
    run_queue_worker,
    run_ready_step,
    set_control_mode,
    worker_resource_limits,
)


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
