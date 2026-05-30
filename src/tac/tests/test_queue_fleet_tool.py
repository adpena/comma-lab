from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import connect_state, initialize_queue_state, load_queue_definition


def _load_queue_fleet_tool():
    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    path = repo / "tools" / "queue_fleet.py"
    spec = importlib.util.spec_from_file_location("queue_fleet_tool_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_build_materializer_execution_queue_tool():
    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    path = repo / "tools" / "build_materializer_execution_queue.py"
    spec = importlib.util.spec_from_file_location(
        "build_materializer_execution_queue_tool_under_test",
        path,
    )
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
                "rows": [
                    {
                        "work_id": "native-only",
                        "executable": True,
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resource_kind": "local_cpu",
                        "postconditions": [],
                    }
                ],
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
    assert payload["native_consumer_artifact_count"] == 1
    assert payload["known_native_consumer_artifact_count"] == 1
    assert payload["needs_recovery_count"] == 0
    assert payload["next_supervise_commands"] == []
    assert payload["next_init_commands"] == []
    assert payload["next_native_consumer_commands"]
    assert "tools/build_materializer_execution_queue.py" in payload[
        "next_native_consumer_commands"
    ][0]
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    assert samples[0]["artifact_schema"] == "byte_shaving_materializer_work_queue.v1"
    assert samples[0]["ignored_for_supervision"] is True
    assert samples[0]["recommended_action"] == (
        "build_materializer_execution_queue_then_supervise_with_experiment_queue"
    )
    assert samples[0]["native_consumer"]["known_native_consumer"] is True
    assert samples[0]["native_consumer"]["score_claim"] is False
    assert samples[0]["native_consumer"]["promotion_eligible"] is False
    assert samples[0]["native_consumer"]["consumer_kind"] == (
        "byte_shaving_materializer_work_queue"
    )
    assert samples[0]["native_consumer"]["ready_for_native_consumer"] is True
    assert samples[0]["native_consumer"]["materializer_executable_row_count"] == 1
    assert samples[0]["native_consumer_command"] == payload[
        "next_native_consumer_commands"
    ][0]


def test_queue_fleet_refuses_materializer_work_queue_with_no_executable_rows(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    native_queue = tmp_path / "blocked_materializer_work_queue.json"
    native_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "rows": [
                    {
                        "work_id": "blocked_work",
                        "executable": False,
                        "blockers": ["receiver_contract_missing"],
                    }
                ],
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
    assert payload["non_executable_artifact_count"] == 1
    assert payload["known_native_consumer_artifact_count"] == 1
    assert payload["next_native_consumer_commands"] == []
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    native_consumer = samples[0]["native_consumer"]
    assert native_consumer["consumer_kind"] == "byte_shaving_materializer_work_queue"
    assert native_consumer["ready_for_native_consumer"] is False
    assert native_consumer["blockers"] == ["materializer_work_queue_has_no_executable_rows"]
    assert native_consumer["materializer_work_row_count"] == 1
    assert native_consumer["materializer_executable_row_count"] == 0
    assert "native_consumer_command" not in samples[0]


def test_queue_fleet_uses_existing_materializer_execution_queue_without_rebuild(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    work_queue = tmp_path / "materializer_work_queue.json"
    work_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "rows": [
                    {
                        "work_id": "unit_work",
                        "executable": True,
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resource_kind": "local_cpu",
                        "postconditions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    execution_queue = tmp_path / "materializer_execution_queue.json"
    execution_queue.write_text(
        json.dumps({"schema": "experiment_queue.v1", "queue_id": "existing", "experiments": []}),
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
    assert payload["next_native_consumer_commands"] == []
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    native_consumer = samples[0]["native_consumer"]
    assert native_consumer["recommended_action"] == "use_existing_materializer_execution_queue"
    assert native_consumer["ready_for_native_consumer"] is False
    assert native_consumer["output_queue_exists"] is True


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
    assert payload["known_native_consumer_artifact_count"] == 1
    assert payload["next_native_consumer_commands"] == []
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    assert samples[0]["artifact_schema"] == "experiment_queue_validation_report.v1"
    assert samples[0]["recommended_action"] == "use_as_validation_report_not_experiment_queue_supervisor"
    assert samples[0]["native_consumer"]["consumer_kind"] == "experiment_queue_validation_report"


def test_queue_fleet_routes_optimizer_candidate_queue_to_submission_closure(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    candidate_queue = tmp_path / "optimizer_candidate_queue.json"
    candidate_queue.write_text(
        json.dumps({"schema": "optimizer_candidate_queue_v1", "top_k": []}),
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
    assert payload["known_native_consumer_artifact_count"] == 1
    assert payload["next_native_consumer_commands"]
    command = payload["next_native_consumer_commands"][0]
    assert "tools/build_materializer_submission_closure.py" in command
    assert "--overwrite" in command
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    assert samples[0]["native_consumer"]["consumer_kind"] == (
        "optimizer_candidate_submission_closure"
    )
    assert samples[0]["native_consumer"]["ready_for_exact_eval_dispatch"] is False


def test_queue_fleet_routes_exact_ready_queue_to_paused_consumer_queue(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    exact_ready_queue = tmp_path / "blocked_exact_ready_queue.json"
    exact_ready_queue.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
                "rows": [],
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
    assert payload["known_native_consumer_artifact_count"] == 1
    command = payload["next_native_consumer_commands"][0]
    assert "tools/build_materializer_exact_eval_consumer.py" in command
    samples = payload["status_samples"]["NON_EXECUTABLE_QUEUE_ARTIFACT"]
    assert samples[0]["native_consumer"]["consumer_kind"] == (
        "optimizer_candidate_exact_eval_consumer"
    )
    assert samples[0]["native_consumer"]["score_claim"] is False


def test_build_materializer_execution_queue_tool_consumes_native_work_queue(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_build_materializer_execution_queue_tool()
    work_queue = tmp_path / "materializer_work_queue.json"
    queue_out = tmp_path / "materializer_execution_queue.json"
    work_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "rows": [
                    {
                        "work_id": "unit_work",
                        "executable": True,
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resource_kind": "local_cpu",
                        "postconditions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--work-queue",
            str(work_queue),
            "--queue-out",
            str(queue_out),
            "--queue-id",
            "unit_materializer_exec",
            "--local-cpu-concurrency",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    queue = json.loads(queue_out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["schema"] == "materializer_execution_queue_build_result.v1"
    assert payload["source_work_queue_schema"] == "byte_shaving_materializer_work_queue.v1"
    assert payload["queue_schema"] == "experiment_queue.v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["queue_id"] == "unit_materializer_exec"
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert queue["experiments"][0]["metadata"]["work_id"] == "unit_work"


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


def test_queue_fleet_consume_native_executes_materializer_builder(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    work_queue = tmp_path / "materializer_work_queue.json"
    execution_queue = tmp_path / "materializer_execution_queue.json"
    work_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "rows": [
                    {
                        "work_id": "unit_work",
                        "executable": True,
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resource_kind": "local_cpu",
                        "postconditions": [],
                    }
                ],
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
            "consume-native",
            "--output-dir",
            str(tmp_path / "native_run"),
            "--max-artifacts",
            "1",
            "--execute",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    queue = json.loads(execution_queue.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["schema"] == "experiment_queue_fleet_native_consumer_run.v1"
    assert payload["selected_count"] == 1
    assert payload["failed_child_count"] == 0
    assert payload["score_claim"] is False
    child = payload["child_runs"][0]
    assert child["consumer_kind"] == "byte_shaving_materializer_work_queue"
    assert child["native_consumer_result"]["returncode"] == 0
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["experiments"][0]["metadata"]["work_id"] == "unit_work"


def test_queue_fleet_drain_local_plan_records_native_init_supervise_phases(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    work_queue = tmp_path / "materializer_work_queue.json"
    work_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "rows": [
                    {
                        "work_id": "unit_work",
                        "executable": True,
                        "command": [sys.executable, "-c", "print('ok')"],
                        "resource_kind": "local_cpu",
                        "postconditions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    queue_path, _artifact = _queue_file(tmp_path)
    state_root = tmp_path / "state"
    _init_state(queue_path, state_root)

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "drain-local",
            "--output-dir",
            str(tmp_path / "fleet_drain"),
            "--max-cycles",
            "3",
            "--max-native-artifacts",
            "1",
            "--max-init-queues",
            "1",
            "--max-supervise-queues",
            "1",
            "--strict",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["schema"] == "experiment_queue_fleet_local_drain_run.v1"
    assert payload["execute"] is False
    assert payload["completed_cycle_count"] == 1
    assert payload["halted_reason"] == "plan_only"
    assert payload["score_claim"] is False
    phases = payload["cycles"][0]["phases"]
    assert [phase["phase"] for phase in phases] == [
        "native_consumer",
        "init_missing",
        "supervise",
    ]
    assert phases[0]["selected_count"] == 1
    assert phases[1]["selected_count"] == 0
    assert phases[2]["selected_count"] == 1


def test_queue_fleet_drain_local_executes_native_init_and_supervise(
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_queue_fleet_tool()
    artifact = tmp_path / "native_artifact.json"
    work_queue = tmp_path / "materializer_work_queue.json"
    execution_queue = tmp_path / "materializer_execution_queue.json"
    work_queue.write_text(
        json.dumps(
            {
                "schema": "byte_shaving_materializer_work_queue.v1",
                "rows": [
                    {
                        "work_id": "unit_work",
                        "executable": True,
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({str(artifact)!r}).write_text("
                                "json.dumps({'schema':'fleet-drain-done.v1'}))"
                            ),
                        ],
                        "resource_kind": "local_cpu",
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": str(artifact),
                                "key": "schema",
                                "equals": "fleet-drain-done.v1",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_root = tmp_path / "state"

    rc = tool.main(
        [
            "--root",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "drain-local",
            "--output-dir",
            str(tmp_path / "fleet_drain"),
            "--execute",
            "--max-cycles",
            "1",
            "--max-native-artifacts",
            "1",
            "--max-init-queues",
            "1",
            "--max-supervise-queues",
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
    assert payload["failed_child_count"] == 0
    assert payload["completed_cycle_count"] == 1
    assert execution_queue.is_file()
    assert artifact.is_file()
    phases = payload["cycles"][0]["phases"]
    assert [phase["selected_count"] for phase in phases] == [1, 1, 1]
    assert payload["final_status"]["status_counts"]["TERMINAL"] == 1
    assert payload["final_status"]["status_counts"]["NON_EXECUTABLE_QUEUE_ARTIFACT"] == 1


def test_queue_fleet_lives_in_comma_lab_control_plane() -> None:
    import comma_lab.scheduler.queue_fleet as queue_fleet

    assert queue_fleet.__name__ == "comma_lab.scheduler.queue_fleet"
    assert queue_fleet.queue_fleet_status.__module__ == "comma_lab.scheduler.queue_fleet"
    assert "tac." not in queue_fleet.queue_fleet_status.__module__
