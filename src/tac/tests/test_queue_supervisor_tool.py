from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_queue_supervisor():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "tools" / "queue_supervisor.py"
    spec = importlib.util.spec_from_file_location("queue_supervisor_tool_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _queue_file(tmp_path: Path, *, mode: str = "running") -> Path:
    artifact = tmp_path / "artifact.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_supervisor_queue",
        "controls": {"mode": mode, "max_concurrency": {"local_cpu": 2}},
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
                                "json.dumps({'schema':'done.v1'}))"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": str(artifact),
                                "key": "schema",
                                "equals": "done.v1",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")
    return path


def _init_state(queue_path: Path, state_path: Path) -> None:
    from comma_lab.scheduler.experiment_queue import (
        connect_state,
        initialize_queue_state,
        load_queue_definition,
    )

    queue = load_queue_definition(queue_path)
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)


def test_queue_supervisor_executes_to_terminal_state(tmp_path: Path, capsys) -> None:
    qs = _load_queue_supervisor()
    queue_path = _queue_file(tmp_path)
    state_path = tmp_path / "queue.sqlite"
    _init_state(queue_path, state_path)
    out_dir = tmp_path / "supervisor"
    rc = qs.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "--output-dir",
            str(out_dir),
            "--execute",
            "--max-ticks",
            "4",
            "--max-steps-per-tick",
            "2",
            "--max-parallel",
            "auto",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "experiment_queue_supervisor_run.v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["final_reason"] == "terminal_queue_state"
    assert payload["tick_count"] >= 2
    assert payload["final_summary"]["status_counts"] == {"succeeded": 1}
    assert (out_dir / "heartbeat.json").is_file()
    assert (out_dir / "ticks.jsonl").is_file()


def test_queue_supervisor_plan_only_does_not_run_worker(tmp_path: Path, capsys) -> None:
    qs = _load_queue_supervisor()
    queue_path = _queue_file(tmp_path)
    state_path = tmp_path / "queue.sqlite"
    _init_state(queue_path, state_path)
    out_dir = tmp_path / "supervisor"
    rc = qs.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "--output-dir",
            str(out_dir),
            "--max-ticks",
            "1",
            "--max-steps-per-tick",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["execute"] is False
    assert payload["last_tick"]["action"] == "run_worker"
    assert payload["final_summary"]["status_counts"] == {"queued": 1}
    assert payload["last_tick"]["score_claim"] is False


def test_queue_supervisor_auto_resumes_paused_queue_when_enabled(tmp_path: Path, capsys) -> None:
    qs = _load_queue_supervisor()
    queue_path = _queue_file(tmp_path, mode="paused")
    state_path = tmp_path / "queue.sqlite"
    _init_state(queue_path, state_path)
    out_dir = tmp_path / "supervisor"
    rc = qs.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "--output-dir",
            str(out_dir),
            "--execute",
            "--auto-resume-paused",
            "--max-ticks",
            "4",
            "--max-steps-per-tick",
            "2",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["auto_resume_paused"] is True
    assert payload["final_reason"] == "terminal_queue_state"
    assert payload["final_summary"]["mode"] == "running"
    assert payload["final_summary"]["status_counts"] == {"succeeded": 1}
    first_tick = json.loads((out_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_tick["action"] == "auto_resume_then_run_worker"
    assert first_tick["score_claim"] is False


def test_queue_supervisor_refuses_paused_queue_without_auto_resume(tmp_path: Path, capsys) -> None:
    qs = _load_queue_supervisor()
    queue_path = _queue_file(tmp_path, mode="paused")
    state_path = tmp_path / "queue.sqlite"
    _init_state(queue_path, state_path)
    out_dir = tmp_path / "supervisor"
    rc = qs.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "--output-dir",
            str(out_dir),
            "--execute",
            "--max-ticks",
            "1",
            "--max-steps-per-tick",
            "2",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["final_reason"] == "paused_with_queued_work"
    assert payload["last_tick"]["action"] == "observe"
    tick_payload = json.loads((out_dir / "tick_0000" / "tick.json").read_text(encoding="utf-8"))
    assert tick_payload["worker_result"] is None
    assert payload["final_summary"]["status_counts"] == {"queued": 1}
