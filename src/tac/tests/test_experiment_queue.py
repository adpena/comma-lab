# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
    queue_summary,
    ready_steps,
    rewind_step,
    run_queue_worker,
    run_ready_step,
    set_control_mode,
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
