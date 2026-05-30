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


def _queue_file(tmp_path: Path, *, mode: str = "running") -> tuple[Path, Path]:
    artifact = tmp_path / "artifact.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "unit_fleet_queue",
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


def test_queue_fleet_treats_native_work_queue_as_non_executable_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    native_queue = tmp_path / "materializer_work_queue.json"
    native_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "items": [{"id": "native-only"}],
            }
        ),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--row-limit",
            "0",
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["row_count"] == 0
    assert payload["invalid_queue_count"] == 0
    assert payload["non_executable_artifact_count"] == 1
    assert payload["needs_recovery_count"] == 0
    assert payload["next_supervise_commands"] == []
    assert payload["next_init_commands"] == []
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    assert samples[0]["artifact_schema"] == "byte_shaving_materializer_work_queue.v1"
    assert samples[0]["recommended_action"] == "route_to_native_consumer_not_experiment_queue_supervisor"


def test_queue_fleet_treats_queue_validation_report_as_non_executable_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    validation_report = tmp_path / "queue_validate.json"
    validation_report.write_text(
        json.dumps(
            {
                "valid": True,
                "queue_id": "validated_but_not_executable",
                "experiment_count": 1,
                "step_count": 3,
            }
        ),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--row-limit",
            "0",
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["invalid_queue_count"] == 0
    assert payload["non_executable_artifact_count"] == 1
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    assert samples[0]["artifact_schema"] == "experiment_queue_validation_report.v1"
    assert samples[0]["recommended_action"] == "use_as_validation_report_not_experiment_queue_supervisor"


def test_queue_fleet_still_flags_malformed_experiment_queue(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    bad_queue = tmp_path / "broken_queue.json"
    bad_queue.write_text('{"schema":"experiment_queue.v1",', encoding="utf-8")

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["invalid_queue_count"] == 1
    assert payload["non_executable_artifact_count"] == 0
    assert payload["rows"][0]["status"] == "INVALID_QUEUE"
    assert payload["rows"][0]["blockers"][0].startswith("load_queue_definition_failed:")


def test_queue_fleet_exposes_init_commands_for_missing_state(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, _artifact = _queue_file(tmp_path)
    state_root = tmp_path / "state"

    rc = tool.main(
        [
            "--root",
            str(queue_path),
            "--state-root",
            str(state_root),
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["status_counts"] == {"NEEDS_INIT": 1}
    assert payload["needs_recovery_count"] == 1
    assert payload["next_init_commands"]
    init_command = payload["next_init_commands"][0]
    assert "tools/experiment_queue.py" in init_command
    assert init_command[-1] == "init"


def test_queue_fleet_exposes_resume_commands_for_paused_work(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, _artifact = _queue_file(tmp_path, mode="paused")
    state_root = tmp_path / "state"
    _init_state(queue_path, state_root)

    rc = tool.main(
        [
            "--root",
            str(queue_path),
            "--state-root",
            str(state_root),
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["status_counts"] == {"PAUSED_WITH_QUEUED_WORK": 1}
    assert payload["ready_to_supervise_count"] == 0
    assert payload["paused_with_queued_work_count"] == 1
    assert payload["next_supervise_commands"] == []
    assert payload["next_resume_commands"]
    resume_command = payload["next_resume_commands"][0]
    assert "tools/experiment_queue.py" in resume_command
    assert resume_command[-4:] == [
        "control",
        "running",
        "--reason",
        "queue_fleet_resume_paused_with_queued_work",
    ]
    assert payload["rows"][0]["resume_command"] == resume_command
    assert payload["score_claim"] is False


def test_queue_fleet_paused_exact_dispatch_gate_is_not_actionable(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, artifact = _queue_file(tmp_path, mode="paused")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["queue_id"] = "unit_exact_eval_dispatch"
    queue["experiments"][0]["steps"][0]["id"] = "dispatch_exact_eval"
    queue["experiments"][0]["steps"][0]["command"] = [
        sys.executable,
        "tools/parallel_dispatch_top_k.py",
        "--ranked-input",
        str(artifact),
    ]
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    state_root = tmp_path / "state"
    _init_state(queue_path, state_root)

    rc = tool.main(
        [
            "--root",
            str(queue_path),
            "--state-root",
            str(state_root),
            "status",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["actionable_count"] == 0
    assert payload["paused_with_queued_work_count"] == 0
    assert payload["paused_exact_dispatch_gate_count"] == 1
    assert payload["next_resume_commands"] == []
    assert payload["rows"][0]["status"] == "PAUSED_EXACT_DISPATCH_GATE"
    assert payload["rows"][0]["score_claim"] is False


def test_queue_fleet_init_missing_initializes_missing_state(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    queue_path, _artifact = _queue_file(tmp_path)
    state_root = tmp_path / "state"

    rc = tool.main(
        [
            "--root",
            str(queue_path),
            "--state-root",
            str(state_root),
            "init-missing",
            "--output-dir",
            str(tmp_path / "fleet_init"),
            "--execute",
            "--max-queues",
            "1",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "experiment_queue_fleet_init_missing_run.v1"
    assert payload["execute"] is True
    assert payload["selected_count"] == 1
    assert payload["failed_child_count"] == 0
    assert payload["child_runs"][0]["init_result"]["returncode"] == 0
    assert payload["initial_status"]["status_counts"] == {"NEEDS_INIT": 1}
    assert payload["final_status"]["status_counts"] == {"READY_TO_SUPERVISE": 1}
    assert payload["score_claim"] is False


def test_queue_fleet_flags_duplicate_queue_id_shared_state(tmp_path: Path, capsys) -> None:
    tool = _load_queue_fleet_tool()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_queue, _first_artifact = _queue_file(first_root)
    second_queue, _second_artifact = _queue_file(second_root)
    state_root = tmp_path / "state"
    state = _init_state(first_queue, state_root)

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
    assert payload["status_counts"] == {"NEEDS_RECOVERY": 2}
    assert payload["needs_recovery_count"] == 2
    rows = sorted(payload["rows"], key=lambda row: row["queue_path"])
    assert {row["queue_path"] for row in rows} == {
        str(first_queue),
        str(second_queue),
    }
    for row in rows:
        assert row["status"] == "NEEDS_RECOVERY"
        assert row["recommended_action"] == "split_or_migrate_duplicate_queue_id_before_supervision"
        assert row["identity_conflict"]["queue_id"] == "unit_fleet_queue"
        assert row["identity_conflict"]["queue_path_count"] == 2
        assert row["identity_conflict"]["states"] == [str(state)]
        assert row["blockers"][0] == "experiment_queue_fleet_duplicate_queue_id:unit_fleet_queue:paths=2"
        assert row["blockers"][1].startswith("experiment_queue_fleet_shared_state:")


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
