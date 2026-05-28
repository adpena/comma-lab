from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import connect_state, initialize_queue_state, load_queue_definition


def _load_queue_fleet_tool():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "tools" / "queue_fleet.py"
    spec = importlib.util.spec_from_file_location("queue_fleet_tool_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _queue_file(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "artifact.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_fleet_queue",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 2}},
        "experiments": [
            {
                "id": "exp",
                "steps": [
                    {
                        "id": "write_artifact",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({str(artifact)!r}).write_text("
                                "json.dumps({'schema':'fleet-done.v1'}))"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": str(artifact),
                                "key": "schema",
                                "equals": "fleet-done.v1",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")
    return path, artifact


def _init_state(queue_path: Path, state_root: Path) -> Path:
    queue = load_queue_definition(queue_path)
    state = state_root / f"experiment_queue_{queue['queue_id']}.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
    return state


def test_queue_fleet_status_discovers_ready_queue(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, _artifact = _queue_file(tmp_path)
    state_root = tmp_path / "state"
    _init_state(queue_path, state_root)

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "experiment_queue_fleet_status.v1"
    assert payload["ready_to_supervise_count"] == 1
    assert payload["rows"][0]["status"] == "READY_TO_SUPERVISE"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False


def test_queue_fleet_supervise_plan_only_preserves_queue(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, artifact = _queue_file(tmp_path)
    state_root = tmp_path / "state"
    _init_state(queue_path, state_root)

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "supervise",
            "--output-dir",
            str(tmp_path / "fleet"),
            "--max-queues",
            "1",
            "--max-ticks-per-queue",
            "1",
            "--max-steps-per-tick",
            "1",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "experiment_queue_fleet_supervisor_run.v1"
    assert payload["execute"] is False
    assert payload["selected_count"] == 1
    assert artifact.exists() is False
    assert payload["failed_child_count"] == 0
    assert payload["score_claim"] is False


def test_queue_fleet_supervise_executes_selected_queue(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, artifact = _queue_file(tmp_path)
    state_root = tmp_path / "state"
    _init_state(queue_path, state_root)

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "supervise",
            "--output-dir",
            str(tmp_path / "fleet"),
            "--execute",
            "--max-queues",
            "1",
            "--max-ticks-per-queue",
            "4",
            "--max-steps-per-tick",
            "2",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["execute"] is True
    assert payload["completed_child_count"] == 1
    assert artifact.is_file()
    child_payload = payload["child_runs"][0]["supervisor_result"]["json_payload"]
    assert child_payload["final_reason"] == "terminal_queue_state"
    assert payload["final_status"]["status_counts"]["TERMINAL"] == 1


def test_queue_fleet_lives_in_comma_lab_control_plane() -> None:
    import comma_lab.scheduler.queue_fleet as queue_fleet

    assert queue_fleet.__name__ == "comma_lab.scheduler.queue_fleet"
    assert queue_fleet.queue_fleet_status.__module__ == "comma_lab.scheduler.queue_fleet"
    assert "tac." not in queue_fleet.queue_fleet_status.__module__
