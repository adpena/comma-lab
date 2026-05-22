# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

from src.comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
    queue_summary,
    ready_steps,
    rewind_step,
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
