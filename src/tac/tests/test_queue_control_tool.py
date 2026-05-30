from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
    load_queue_definition,
    queue_summary,
)


def _load_queue_control():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "tools" / "queue_control.py"
    spec = importlib.util.spec_from_file_location("queue_control_tool_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _queue_file(tmp_path: Path) -> Path:
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_control_queue",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 2}},
        "experiments": [
            {
                "id": "exp",
                "steps": [
                    {
                        "id": "first",
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resources": {"kind": "local_cpu"},
                    }
                ],
            }
        ],
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")
    return path


def test_queue_control_status_is_fail_closed_and_recommends_worker(tmp_path: Path) -> None:
    qc = _load_queue_control()
    queue_path = _queue_file(tmp_path)
    state_path = tmp_path / "queue.sqlite"
    queue = load_queue_definition(queue_path)
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)

    observation, resolved_state = qc.observe_for_control(
        queue_path,
        state_path,
        tail_lines=0,
        include_orphans=False,
    )
    summary = qc.summarize_observation(
        observation,
        queue_path=queue_path,
        state_path=resolved_state,
        max_steps=4,
        max_parallel=2,
        allow_cloud=False,
    )

    assert summary["schema"] == "experiment_queue_control_surface.v1"
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["rank_or_kill_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["status_counts"] == {"queued": 1}
    assert summary["resume_command"][-4:] == ["--max-steps", "4", "--max-parallel", "2"]
    assert summary["recommended_actions"][0]["action"] == "run_worker"


def test_queue_control_pause_resume_stop_use_queue_authority(tmp_path: Path, capsys) -> None:
    qc = _load_queue_control()
    queue_path = _queue_file(tmp_path)
    state_path = tmp_path / "queue.sqlite"

    assert (
        qc.main(
            [
                "--queue",
                str(queue_path),
                "--state",
                str(state_path),
                "pause",
                "--reason",
                "unit test pause operator control",
            ]
        )
        == 0
    )
    capsys.readouterr()
    queue = load_queue_definition(queue_path)
    with connect_state(state_path) as conn:
        assert queue_summary(conn, queue)["mode"] == "paused"

    assert (
        qc.main(
            [
                "--queue",
                str(queue_path),
                "--state",
                str(state_path),
                "resume",
                "--reason",
                "unit test resume operator control",
            ]
        )
        == 0
    )
    capsys.readouterr()
    with connect_state(state_path) as conn:
        assert queue_summary(conn, queue)["mode"] == "running"

    assert (
        qc.main(
            [
                "--queue",
                str(queue_path),
                "--state",
                str(state_path),
                "stop",
                "--reason",
                "unit test stop freezes future work",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["operator_action"] == "stop"
    assert output["requested_mode"] == "frozen"
    assert output["score_claim"] is False
    with connect_state(state_path) as conn:
        assert queue_summary(conn, queue)["mode"] == "frozen"


def test_queue_control_recover_pauses_and_reconciles_stale_running(tmp_path: Path, capsys) -> None:
    qc = _load_queue_control()
    marker = tmp_path / "done.json"
    marker.write_text(json.dumps({"schema": "done.v1"}), encoding="utf-8")
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "unit_recover_queue",
                "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
                "experiments": [
                    {
                        "id": "exp",
                        "steps": [
                            {
                                "id": "stale",
                                "command": [sys.executable, "-c", "print('already done')"],
                                "resources": {"kind": "local_cpu"},
                                "postconditions": [
                                    {
                                        "type": "json_equals",
                                        "path": str(marker),
                                        "key": "schema",
                                        "equals": "done.v1",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "queue.sqlite"
    queue = load_queue_definition(queue_path)
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'running',
                last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                json.dumps({"pid": 999999, "parent_pid": 999999}),
                "unit_recover_queue",
                "exp",
                "stale",
            ),
        )
        conn.commit()

    rc = qc.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "recover",
            "--reason",
            "unit test crash recovery after worker disappearance",
            "--stale-after-seconds",
            "0",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "experiment_queue_crash_recovery.v1"
    assert payload["left_paused"] is True
    assert payload["stale_reconciliation"]["reconciled_step_count"] == 1
    assert payload["stale_reconciliation"]["reconciled_steps"][0]["status"] == "succeeded"
    assert payload["score_claim"] is False
    with connect_state(state_path) as conn:
        summary = queue_summary(conn, queue)
    assert summary["mode"] == "paused"
    assert summary["status_counts"] == {"succeeded": 1}


def test_queue_control_recover_rewinds_succeeded_missing_artifact_step(
    tmp_path: Path,
    capsys,
) -> None:
    qc = _load_queue_control()
    missing_marker = tmp_path / "missing.json"
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "unit_recover_artifact_queue",
                "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
                "experiments": [
                    {
                        "id": "exp",
                        "steps": [
                            {
                                "id": "materialize",
                                "command": [sys.executable, "-c", "print('materialize')"],
                                "resources": {"kind": "local_cpu"},
                                "postconditions": [
                                    {
                                        "type": "json_equals",
                                        "path": str(missing_marker),
                                        "key": "schema",
                                        "equals": "done.v1",
                                    }
                                ],
                            },
                            {
                                "id": "followup",
                                "command": [sys.executable, "-c", "print('followup')"],
                                "resources": {"kind": "local_cpu"},
                                "requires": ["materialize"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "queue.sqlite"
    queue = load_queue_definition(queue_path)
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'succeeded'
            WHERE queue_id = ? AND experiment_id = ? AND step_id IN (?, ?)
            """,
            ("unit_recover_artifact_queue", "exp", "materialize", "followup"),
        )
        conn.commit()

    rc = qc.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "recover",
            "--reason",
            "unit test artifact recovery",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["auto_recovery_rewinds"]["rewind_count"] == 1
    assert payload["auto_recovery_rewinds"]["rewound_steps"][0] == {
        "action": "rewind_succeeded_step_with_artifact_failure",
        "experiment_id": "exp",
        "step_id": "materialize",
        "cascade": True,
    }
    assert payload["after_auto_recovery"]["status_counts"] == {"queued": 2}
    with connect_state(state_path) as conn:
        summary = queue_summary(conn, queue)
    assert summary["mode"] == "paused"
    assert summary["status_counts"] == {"queued": 2}


def test_queue_control_missing_queue_fails_without_traceback(capsys, tmp_path: Path) -> None:
    qc = _load_queue_control()
    rc = qc.main(["--queue", str(tmp_path / "missing.json"), "status"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "FATAL: queue definition not found" in captured.err
    assert "Traceback" not in captured.err


def test_queue_control_termination_refuses_unmatched_pid() -> None:
    qc = _load_queue_control()
    rows = qc._terminate_running_processes(
        {
            "running_steps": [
                {
                    "experiment_id": "exp",
                    "step_id": "step",
                    "pid": 999999,
                    "processes": [{"pid": "123", "command": "different process"}],
                }
            ]
        },
        dry_run=False,
    )

    assert rows == [
        {
            "experiment_id": "exp",
            "step_id": "step",
            "pid": 999999,
            "signal": "SIGTERM",
            "dry_run": False,
            "sent": False,
            "blockers": ["pid_not_matched_by_live_observer_process_table"],
        }
    ]
