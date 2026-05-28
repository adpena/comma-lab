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


def _queue_file(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_supervisor_queue",
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


def test_queue_supervisor_executes_to_terminal_state(tmp_path: Path, capsys) -> None:
    qs = _load_queue_supervisor()
    queue_path = _queue_file(tmp_path)
    out_dir = tmp_path / "supervisor"
    rc = qs.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(tmp_path / "queue.sqlite"),
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
    out_dir = tmp_path / "supervisor"
    rc = qs.main(
        [
            "--queue",
            str(queue_path),
            "--state",
            str(tmp_path / "queue.sqlite"),
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
