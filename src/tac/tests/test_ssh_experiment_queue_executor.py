# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
    ready_steps,
    reconcile_stale_running_steps,
)
from comma_lab.scheduler.ssh_experiment_queue_executor import (
    build_remote_git_preflight_command,
    build_remote_shell_command,
    build_rsync_pull_command,
    build_rsync_push_command,
    run_staircase_ssh_executor,
    select_ssh_tasks,
)
from comma_lab.scheduler.staircase_dag import (
    build_staircase_dag_from_experiment_queue,
    plan_staircase_dispatch,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeSshRunner:
    def __init__(self, *, artifact: Path | None = None, create_artifact: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.artifact = artifact
        self.create_artifact = create_artifact

    def __call__(self, argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        if self.artifact is not None and self.create_artifact:
            self.artifact.write_text(
                json.dumps(
                    {
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="remote ok\n")


def _is_preflight_only(remote_script: str) -> bool:
    return (
        "git rev-parse HEAD" in remote_script
        and " && python " not in remote_script
        and " && .venv/bin/python " not in remote_script
    )


def _queue(artifact: Path, *, artifact_mobility: dict[str, Any] | None = None) -> dict[str, Any]:
    return normalize_queue_definition({
        "schema": "experiment_queue.v1",
        "queue_id": "ssh_executor_fixture",
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
        "experiments": [
            {
                "id": "candidate",
                "status": "queued",
                "priority": 1,
                "steps": [
                    {
                        "id": "materialize",
                        "kind": "command",
                        "command": [
                            "python",
                            "-c",
                            (
                                "import json, pathlib; "
                                f"pathlib.Path({str(artifact)!r}).write_text("
                                "json.dumps({'score_claim': False}))"
                            ),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "postconditions": [
                            {"type": "path_exists", "path": str(artifact)},
                            {
                                "type": "json_false_authority",
                                "path": str(artifact),
                            },
                        ],
                        "artifact_mobility": artifact_mobility or {},
                    }
                ],
            }
        ],
    })


def _parallel_queue(
    tmp_path: Path,
    *,
    max_concurrency: int = 2,
    artifact_mobility: bool = False,
    duplicate_artifact: bool = False,
) -> tuple[dict[str, Any], list[Path]]:
    artifacts = [
        tmp_path / "out" / ("shared.json" if duplicate_artifact else f"done_{index}.json")
        for index in range(2)
    ]
    mobility = (
        {
            "schema": "experiment_queue_artifact_mobility.v1",
            "mode": "pullback",
            "required": True,
        }
        if artifact_mobility
        else {}
    )
    return (
        normalize_queue_definition(
            {
                "schema": "experiment_queue.v1",
                "queue_id": "ssh_executor_fixture",
                "controls": {"mode": "running", "max_concurrency": {"local_cpu": max_concurrency}},
                "experiments": [
                    {
                        "id": f"candidate_{index}",
                        "status": "queued",
                        "priority": 1,
                        "steps": [
                            {
                                "id": "materialize",
                                "kind": "command",
                                "command": [
                                    "python",
                                    "-c",
                                    (
                                        "import json, pathlib; "
                                        f"pathlib.Path({str(artifact)!r}).write_text("
                                        "json.dumps({'score_claim': False, "
                                        "'promotion_eligible': False, "
                                        "'rank_or_kill_eligible': False, "
                                        "'ready_for_exact_eval_dispatch': False}))"
                                    ),
                                ],
                                "resources": {"kind": "local_cpu"},
                                "postconditions": [
                                    {"type": "path_exists", "path": str(artifact)},
                                    {
                                        "type": "json_false_authority",
                                        "path": str(artifact),
                                    },
                                ],
                                "artifact_mobility": dict(mobility),
                            }
                        ],
                    }
                    for index, artifact in enumerate(artifacts)
                ],
            }
        ),
        artifacts,
    )


def _sha256_json(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plan(
    queue: dict[str, Any],
    *,
    executor: str = "ssh_experiment_queue",
    slots: int = 1,
    max_nodes: int = 1,
) -> dict[str, Any]:
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="ssh_executor_fixture_dag",
        resource_pools=[
            {
                "id": "sshbox",
                "label": "SSH test worker",
                "slots": {"local_cpu": slots},
                "memory_gb": 16,
                "disk_gb": 8,
                "tags": ["ssh", "test"],
                "executor": executor,
                "ssh_target": "user@sshbox",
                "remote_repo_root": "/remote/pact",
            }
        ],
    )
    return plan_staircase_dispatch(dag, max_nodes=max_nodes)


def _step_row(state: Path) -> sqlite3.Row:
    with connect_state(state) as conn:
        return conn.execute(
            """
            SELECT status, attempts, last_event_json
            FROM step_state
            WHERE queue_id = 'ssh_executor_fixture'
              AND experiment_id = 'candidate'
              AND step_id = 'materialize'
            """
        ).fetchone()


def _step_rows(state: Path) -> list[sqlite3.Row]:
    with connect_state(state) as conn:
        return conn.execute(
            """
            SELECT experiment_id, step_id, status, attempts, last_event_json
            FROM step_state
            WHERE queue_id = 'ssh_executor_fixture'
            ORDER BY experiment_id, step_id
            """
        ).fetchall()


def test_remote_shell_command_uses_queue_argv_shape() -> None:
    command = build_remote_shell_command(
        remote_repo_root="/tmp/pact repo",
        command=["python", "-c", "print('ok')"],
    )

    assert command == "cd '/tmp/pact repo' && python -c 'print('\"'\"'ok'\"'\"')'"


def test_remote_shell_command_rechecks_git_head_in_execution_call() -> None:
    command = build_remote_shell_command(
        remote_repo_root="/tmp/pact repo",
        command=["python", "-c", "print('ok')"],
        expected_head="b" * 40,
        require_clean=True,
    )

    assert command.startswith("cd '/tmp/pact repo' && test -d .git")
    assert 'test "$(git rev-parse HEAD)" = bbbbb' in command
    assert "git diff --quiet" in command
    assert command.endswith("&& python -c 'print('\"'\"'ok'\"'\"')'")


def test_remote_git_preflight_command_checks_head_and_cleanliness() -> None:
    preflight = build_remote_git_preflight_command(
        remote_repo_root="/tmp/pact repo",
        expected_head="a" * 40,
        require_clean=True,
    )

    assert "cd '/tmp/pact repo'" in preflight
    assert "git rev-parse HEAD" in preflight
    assert "git diff --quiet" in preflight
    assert "git diff --cached --quiet" in preflight


def test_rsync_pull_command_uses_remote_source_and_local_destination() -> None:
    command = build_rsync_pull_command(
        rsync_binary="rsync",
        ssh_target="user@host",
        remote_path="/remote/pact/out/result.json",
        local_path="/local/pact/out/result.json",
    )

    assert command == [
        "rsync",
        "-a",
        "-e",
        (
            "ssh -o ConnectTimeout=10 -o BatchMode=yes -o ConnectionAttempts=1 "
            "-o ServerAliveInterval=15 -o ServerAliveCountMax=2"
        ),
        "--",
        "user@host:/remote/pact/out/result.json",
        "/local/pact/out/result.json",
    ]


def test_rsync_push_command_uses_local_source_and_remote_destination() -> None:
    command = build_rsync_push_command(
        rsync_binary="rsync",
        ssh_target="user@host",
        local_path="/local/pact/input.json",
        remote_path="/remote/pact/input.json",
    )

    assert command == [
        "rsync",
        "-a",
        "-e",
        (
            "ssh -o ConnectTimeout=10 -o BatchMode=yes -o ConnectionAttempts=1 "
            "-o ServerAliveInterval=15 -o ServerAliveCountMax=2"
        ),
        "--",
        "/local/pact/input.json",
        "user@host:/remote/pact/input.json",
    ]


def test_rsync_push_command_deletes_stale_remote_files_for_directory_inputs(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "runtime"
    local_dir.mkdir()

    command = build_rsync_push_command(
        rsync_binary="rsync",
        ssh_target="user@host",
        local_path=local_dir,
        remote_path="/remote/pact/runtime",
    )

    assert "--delete" in command
    assert command[-2:] == [
        f"{local_dir.as_posix()}/",
        "user@host:/remote/pact/runtime/",
    ]


def test_rsync_pull_command_rejects_unsafe_remote_artifact_paths() -> None:
    for remote_path in (
        "relative/out.json",
        "/remote/../out.json",
        "/remote/out bad.json",
        "/remote/out;touch.json",
        "/remote/out*.json",
        "/remote/[out].json",
        "/remote/out(1).json",
        "/",
    ):
        with pytest.raises(ExperimentQueueError):
            build_rsync_pull_command(
                rsync_binary="rsync",
                ssh_target="user@host",
                remote_path=remote_path,
                local_path="/local/out.json",
            )
        with pytest.raises(ExperimentQueueError):
            build_rsync_push_command(
                rsync_binary="rsync",
                ssh_target="user@host",
                local_path="/local/in.json",
                remote_path=remote_path,
            )


def test_rsync_push_command_rejects_directory_push_to_remote_root(
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "runtime"
    local_dir.mkdir()

    with pytest.raises(ExperimentQueueError, match="filesystem root"):
        build_rsync_push_command(
            rsync_binary="rsync",
            ssh_target="user@host",
            local_path=local_dir,
            remote_path="/",
        )


def test_select_ssh_tasks_rejects_filesystem_root_path_map(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    with pytest.raises(ExperimentQueueError, match="filesystem root"):
        select_ssh_tasks(
            plan,
            queue,
            ready=ready,
            repo_root=REPO_ROOT,
            artifact_path_maps={"/": "/remote"},
        )


def test_select_ssh_tasks_rejects_remote_root_path_map(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    with pytest.raises(ExperimentQueueError, match="filesystem root"):
        select_ssh_tasks(
            plan,
            queue,
            ready=ready,
            repo_root=REPO_ROOT,
            artifact_path_maps={tmp_path.as_posix(): "/"},
        )


def test_select_ssh_tasks_requires_writeback_hashes_remote_root_and_postconditions(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    plan["dask_task_specs"][0]["machine"]["remote_repo_root"] = "relative/path"
    plan["dask_task_specs"][0]["queue_state_writeback"]["step_hashes"]["command_hash"] = "0" * 64
    plan["dask_task_specs"][0]["postconditions"] = []
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(plan, queue, ready=ready)

    assert selections[0].ready_step is ready[0]
    assert "remote_repo_root_must_be_absolute" in selections[0].blockers
    assert "queue_state_writeback_step_hashes_mismatch" in selections[0].blockers
    assert "task_postconditions_missing_for_ssh" in selections[0].blockers


def test_select_ssh_tasks_blocks_missing_required_artifact_mobility_contract(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
        require_artifact_mobility=True,
    )

    assert "artifact_mobility_contract_missing" in selections[0].blockers


def test_select_ssh_tasks_blocks_stripped_artifact_mobility_metadata(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(
        artifact,
        artifact_mobility={
            "schema": "experiment_queue_artifact_mobility.v1",
            "mode": "pullback",
            "required": True,
        },
    )
    plan = _plan(queue)
    del plan["dask_task_specs"][0]["artifact_mobility"]
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
    )

    assert "artifact_mobility_metadata_missing_from_task" in selections[0].blockers


def test_select_ssh_tasks_requires_mapped_existing_input_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    input_artifact = tmp_path / "inputs" / "surface.json"
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = [str(input_artifact)]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
        artifact_path_maps={(tmp_path / "out").as_posix(): "/remote/out"},
        require_artifact_mobility=True,
    )

    assert f"input_artifact_missing:{input_artifact.as_posix()}" in selections[0].blockers
    assert f"artifact_push_missing_for_input:{input_artifact.as_posix()}" in selections[0].blockers


def test_select_ssh_tasks_requires_input_mobility_for_declared_inputs(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    input_artifact = tmp_path / "inputs" / "surface.json"
    input_artifact.parent.mkdir(parents=True)
    input_artifact.write_text(json.dumps({"schema": "input_fixture.v1"}), encoding="utf-8")
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = [str(input_artifact)]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
        require_artifact_mobility=False,
    )

    assert "input_artifact_mobility_contract_missing" in selections[0].blockers
    assert f"artifact_push_missing_for_input:{input_artifact.as_posix()}" in selections[0].blockers


def test_select_ssh_tasks_rejects_symlink_input_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    real_input = tmp_path / "inputs" / "real.json"
    real_input.parent.mkdir(parents=True)
    real_input.write_text(json.dumps({"schema": "input_fixture.v1"}), encoding="utf-8")
    symlink_input = tmp_path / "inputs" / "linked.json"
    symlink_input.symlink_to(real_input)
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = [str(symlink_input)]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
    )

    assert (
        f"input_artifact_symlink_forbidden:{symlink_input.as_posix()}"
        in selections[0].blockers
    )


@pytest.mark.parametrize(
    "input_artifact_paths, expected_blocker",
    [
        ("surface.json", "input_artifact_paths_must_be_list"),
        ([None], "input_artifact_paths[0]_must_be_nonempty_string"),
        ([""], "input_artifact_paths[0]_must_be_nonempty_string"),
    ],
)
def test_select_ssh_tasks_blocks_malformed_input_artifact_telemetry(
    tmp_path: Path,
    input_artifact_paths: object,
    expected_blocker: str,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = input_artifact_paths
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selections = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
    )

    assert expected_blocker in selections[0].blockers


def test_ssh_executor_blocks_remote_step_without_local_visible_postcondition(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    queue["experiments"][0]["steps"][0]["postconditions"] = []
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    runner = FakeSshRunner(artifact=artifact)

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        max_steps=2,
        allow_noncanonical_state=True,
        runner=runner,
    )

    assert runner.calls == []
    assert result["blocked_count"] == 1
    assert (
        "ssh_executor_local_visible_postcondition_required"
        in result["task_results"][0]["blockers"]
    )
    row = _step_row(state)
    assert row["status"] == "queued"
    assert row["attempts"] == 0


def test_ssh_executor_pulls_back_artifacts_before_local_postconditions(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if str(argv[0]).endswith("rsync"):
            Path(argv[-1]).write_text(
                json.dumps(
                    {
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        max_steps=2,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["success_count"] == 1
    assert any(call[0] == "rsync" for call in calls)
    row = _step_row(state)
    assert row["status"] == "succeeded"
    last_event = json.loads(row["last_event_json"])
    assert last_event["artifact_mobility"]["mode"] == "rsync_pull"
    assert last_event["artifact_mobility"]["succeeded"] is True
    assert last_event["elapsed_seconds"] >= last_event["remote_elapsed_seconds"]


def test_ssh_executor_pushes_input_artifacts_before_remote_command(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "done.json"
    input_artifact = tmp_path / "inputs" / "surface.json"
    input_artifact.parent.mkdir(parents=True)
    input_artifact.write_text(json.dumps({"schema": "input_fixture.v1"}), encoding="utf-8")
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = [str(input_artifact)]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        if remote_script.startswith("mkdir -p "):
            return subprocess.CompletedProcess(argv, 0, stdout="mkdir ok\n")
        if str(argv[0]).endswith("rsync") and str(argv[-1]).startswith("user@sshbox:"):
            return subprocess.CompletedProcess(argv, 0, stdout="push ok\n")
        if str(argv[0]).endswith("rsync"):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(
                    {
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, stdout="pull ok\n")
        return subprocess.CompletedProcess(argv, 0, stdout="remote ok\n")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["success_count"] == 1
    remote_call_index = next(
        index
        for index, call in enumerate(calls)
        if call[0] == "ssh" and "python -c" in str(call[-1])
    )
    push_call_index = next(
        index
        for index, call in enumerate(calls)
        if call[0] == "rsync" and str(call[-1]).startswith("user@sshbox:")
    )
    assert push_call_index < remote_call_index
    row = _step_row(state)
    last_event = json.loads(row["last_event_json"])
    assert last_event["input_mobility"]["mode"] == "rsync_push"
    push = last_event["input_mobility"]["pushes"][0]
    assert push["local_path"] == input_artifact.as_posix()
    assert push["local_manifest"]["path"] == input_artifact.as_posix()
    assert push["local_manifest"]["bytes"] == input_artifact.stat().st_size
    assert push["local_manifest"]["sha256"] == hashlib.sha256(
        input_artifact.read_bytes()
    ).hexdigest()
    assert push["local_manifest_sha256"] == _sha256_json(push["local_manifest"])
    assert last_event["artifact_mobility"]["mode"] == "rsync_pull"


def test_ssh_executor_records_directory_input_manifest_and_delete_push(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "done.json"
    input_dir = tmp_path / "inputs" / "runtime"
    input_dir.mkdir(parents=True)
    (input_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (input_dir / "b.bin").write_bytes(b"beta")
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = [str(input_dir)]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        if remote_script.startswith("mkdir -p "):
            return subprocess.CompletedProcess(argv, 0, stdout="mkdir ok\n")
        if str(argv[0]).endswith("rsync") and str(argv[-1]).startswith("user@sshbox:"):
            return subprocess.CompletedProcess(argv, 0, stdout="push ok\n")
        if str(argv[0]).endswith("rsync"):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(
                    {
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, stdout="pull ok\n")
        return subprocess.CompletedProcess(argv, 0, stdout="remote ok\n")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["success_count"] == 1
    push_call = next(
        call
        for call in calls
        if call[0] == "rsync" and str(call[-1]).startswith("user@sshbox:")
    )
    assert "--delete" in push_call
    row = _step_row(state)
    last_event = json.loads(row["last_event_json"])
    push = last_event["input_mobility"]["pushes"][0]
    assert push["local_manifest"]["is_dir"] is True
    assert push["local_manifest"]["recursive_entry_count"] == 2
    assert push["local_manifest"]["recursive_bytes"] == 9
    assert len(push["local_manifest"]["recursive_sha256"]) == 64


def test_ssh_executor_does_not_run_remote_command_when_input_push_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "done.json"
    input_artifact = tmp_path / "inputs" / "surface.json"
    input_artifact.parent.mkdir(parents=True)
    input_artifact.write_text(json.dumps({"schema": "input_fixture.v1"}), encoding="utf-8")
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step.setdefault("telemetry", {})["input_artifact_paths"] = [str(input_artifact)]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        if remote_script.startswith("mkdir -p "):
            return subprocess.CompletedProcess(argv, 0, stdout="mkdir ok\n")
        if str(argv[0]).endswith("rsync") and str(argv[-1]).startswith("user@sshbox:"):
            return subprocess.CompletedProcess(argv, 23, stdout="push failed\n")
        raise AssertionError(f"remote command should not run after failed input push: {argv}")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["failure_count"] == 1
    assert not artifact.exists()
    row = _step_row(state)
    assert row["status"] == "failed"
    last_event = json.loads(row["last_event_json"])
    assert last_event["input_mobility"]["mode"] == "rsync_push"
    assert last_event["input_mobility"]["succeeded"] is False
    assert last_event["remote_skipped_reason"] == "input_mobility_failed"
    assert last_event["artifact_mobility"]["mode"] == "skipped_remote_execution_failed"


def test_ssh_executor_pulls_back_mapped_telemetry_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "done.json"
    telemetry_artifact = tmp_path / "out" / "done.md"
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step["telemetry"]["artifact_paths"] = [str(artifact), str(telemetry_artifact)]
    step["telemetry"]["pullback_artifact_paths"] = [
        str(artifact),
        str(telemetry_artifact),
    ]
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if str(argv[0]).endswith("rsync"):
            target = Path(argv[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".json":
                target.write_text(
                    json.dumps(
                        {
                            "score_claim": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                target.write_text("remote telemetry\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["success_count"] == 1
    rsync_calls = [call for call in calls if call[0] == "rsync"]
    assert len(rsync_calls) == 2
    assert telemetry_artifact.read_text(encoding="utf-8") == "remote telemetry\n"
    last_event = json.loads(_step_row(state)["last_event_json"])
    assert {
        pullback["local_path"]
        for pullback in last_event["artifact_mobility"]["pullbacks"]
    } == {artifact.as_posix(), telemetry_artifact.as_posix()}


def test_parallel_ssh_executor_claims_steps_before_remote_commands_and_preserves_order(
    tmp_path: Path,
) -> None:
    queue, artifacts = _parallel_queue(tmp_path)
    plan = _plan(queue, slots=2, max_nodes=2)
    state = tmp_path / "queue.sqlite"
    active = 0
    max_active = 0
    remote_started = threading.Event()
    lock = threading.Lock()
    observed_statuses: list[str] = []
    remote_scripts: list[str] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal active, max_active
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        with lock:
            remote_scripts.append(remote_script)
            experiment_id = "candidate_0" if str(artifacts[0]) in remote_script else "candidate_1"
            with connect_state(state) as conn:
                row = conn.execute(
                    """
                    SELECT status
                    FROM step_state
                    WHERE queue_id = 'ssh_executor_fixture'
                      AND experiment_id = ?
                      AND step_id = 'materialize'
                    """,
                    (experiment_id,),
                ).fetchone()
            observed_statuses.append(str(row["status"]))
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                remote_started.set()
        remote_started.wait(timeout=1.0)
        target = artifacts[0] if str(artifacts[0]) in remote_script else artifacts[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        time.sleep(0.02 if target == artifacts[0] else 0.0)
        with lock:
            active -= 1
        return subprocess.CompletedProcess(argv, 0, stdout="remote ok\n")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        max_steps=2,
        allow_noncanonical_state=True,
        runner=runner,
    )

    assert result["execution_mode"] == "bounded_parallel_ssh_executor"
    assert result["max_workers"] == 2
    assert result["success_count"] == 2
    assert result["executed_count"] == 2
    assert max_active == 2
    assert observed_statuses == ["running", "running"]
    assert len(remote_scripts) == 2
    assert [
        row["experiment_id"]
        for row in (item["task"]["ready_step"] for item in result["task_results"])
    ] == ["candidate_0", "candidate_1"]
    assert {row["status"] for row in _step_rows(state)} == {"succeeded"}


def test_parallel_ssh_executor_does_not_duplicate_remote_command_across_instances(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
    command_started = threading.Event()
    release_command = threading.Event()
    lock = threading.Lock()
    remote_command_count = 0

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal remote_command_count
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        with lock:
            remote_command_count += 1
            command_started.set()
        release_command.wait(timeout=1.0)
        artifact.write_text(
            json.dumps(
                {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="remote ok\n")

    results: list[dict[str, Any]] = []

    def execute_once() -> None:
        results.append(
            run_staircase_ssh_executor(
                plan,
                queue,
                state_path=state,
                repo_root=REPO_ROOT,
                execute=True,
                allow_noncanonical_state=True,
                runner=runner,
            )
        )

    first = threading.Thread(target=execute_once)
    second = threading.Thread(target=execute_once)
    first.start()
    assert command_started.wait(timeout=1.0)
    second.start()
    release_command.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert remote_command_count == 1
    assert sum(result["success_count"] for result in results) == 1
    assert _step_row(state)["status"] == "succeeded"


def test_parallel_ssh_executor_blocks_duplicate_pullback_destinations(
    tmp_path: Path,
) -> None:
    queue, _artifacts = _parallel_queue(
        tmp_path,
        artifact_mobility=True,
        duplicate_artifact=True,
    )
    plan = _plan(queue, slots=2, max_nodes=2)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="should not run\n")

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        max_steps=2,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert calls == []
    assert result["blocked_count"] == 2
    assert all(
        any(
            blocker.startswith("duplicate_pullback_destination:")
            for blocker in row["blockers"]
        )
        for row in result["task_results"]
    )
    assert {row["status"] for row in _step_rows(state)} == {"queued"}


def test_parallel_ssh_executor_records_parent_pid_for_stale_recovery(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    remote_running = threading.Event()
    release_remote = threading.Event()
    recovery_result: dict[str, Any] = {}

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        remote_running.set()
        release_remote.wait(timeout=1.0)
        artifact.write_text(
            json.dumps(
                {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="remote ok\n")

    def execute_once() -> None:
        run_staircase_ssh_executor(
            plan,
            queue,
            state_path=state,
            repo_root=REPO_ROOT,
            execute=True,
            allow_noncanonical_state=True,
            runner=runner,
        )

    worker = threading.Thread(target=execute_once)
    worker.start()
    assert remote_running.wait(timeout=1.0)
    with connect_state(state) as conn:
        recovery_result.update(
            reconcile_stale_running_steps(
                conn,
                queue,
                repo_root=REPO_ROOT,
                stale_after_seconds=0,
            )
        )
    release_remote.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert recovery_result["reconciled_step_count"] == 0
    assert recovery_result["skipped_steps"][0]["reason"] == "worker_parent_alive"
    assert _step_row(state)["status"] == "succeeded"


def test_ssh_executor_does_not_pull_recursive_telemetry_without_explicit_pullback(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "done.json"
    telemetry_dir = tmp_path / "out" / "inflated"
    queue = _queue(artifact)
    step = queue["experiments"][0]["steps"][0]
    step["telemetry"]["artifact_paths"] = [str(telemetry_dir)]
    step["telemetry"]["recursive"] = True
    step["telemetry"]["max_recursive_entries"] = 512
    plan = _plan(queue)
    state = tmp_path / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        ready = ready_steps(conn, queue, repo_root=REPO_ROOT)

    selection = select_ssh_tasks(
        plan,
        queue,
        ready=ready,
        repo_root=REPO_ROOT,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
    )[0]

    assert (
        "recursive_telemetry_artifact_paths_are_not_pullback_authority"
        in selection.blockers
    )


def test_ssh_executor_fails_remote_success_when_artifact_pullback_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "missing.json"
    queue = _queue(artifact)
    state = tmp_path / "queue.sqlite"

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if str(argv[0]).endswith("rsync"):
            return subprocess.CompletedProcess(argv, 23, stdout="no such file\n")
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n")

    result = run_staircase_ssh_executor(
        _plan(queue),
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["failure_count"] == 1
    row = _step_row(state)
    assert row["status"] == "failed"
    last_event = json.loads(row["last_event_json"])
    assert last_event["returncode"] == 23
    assert last_event["artifact_mobility"]["succeeded"] is False


def test_ssh_executor_skips_pullback_when_remote_command_fails(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "out" / "missing.json"
    queue = _queue(artifact)
    state = tmp_path / "queue.sqlite"
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if str(argv[0]).endswith("rsync"):
            raise AssertionError("pullback must not run after remote command failure")
        remote_script = str(argv[-1])
        if _is_preflight_only(remote_script):
            return subprocess.CompletedProcess(argv, 0, stdout="preflight ok\n")
        return subprocess.CompletedProcess(argv, 9, stdout="remote failed\n")

    result = run_staircase_ssh_executor(
        _plan(queue),
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        artifact_path_maps={tmp_path.as_posix(): "/remote/tmp"},
        require_artifact_mobility=True,
        runner=runner,
    )

    assert result["failure_count"] == 1
    assert not any(call[0] == "rsync" for call in calls)
    row = _step_row(state)
    assert row["status"] == "failed"
    last_event = json.loads(row["last_event_json"])
    assert last_event["returncode"] == 9
    assert last_event["artifact_mobility"]["mode"] == "skipped_remote_execution_failed"


def test_ssh_executor_preflight_failure_leaves_step_queued(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    state = tmp_path / "queue.sqlite"

    def runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 42, stdout="remote dirty\n")

    result = run_staircase_ssh_executor(
        _plan(queue),
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        runner=runner,
    )

    assert result["executed_count"] == 0
    assert result["blocked_count"] == 1
    assert "remote_git_preflight_failed:42" in result["task_results"][0]["blockers"]
    row = _step_row(state)
    assert row["status"] == "queued"
    assert row["attempts"] == 0


def test_ssh_executor_dry_run_does_not_create_or_mutate_state(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    state = tmp_path / "missing.sqlite"

    result = run_staircase_ssh_executor(
        _plan(queue),
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=False,
        allow_noncanonical_state=True,
    )

    assert result["dry_run_state_mode"] == "ephemeral_initialized_state"
    assert result["selected_count"] == 1
    assert not state.exists()


def test_ssh_executor_ignores_mutated_plan_command_and_uses_queue_command(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    plan["dask_task_specs"][0]["command"] = ["rm", "-rf", "/definitely-not-authority"]
    state = tmp_path / "queue.sqlite"
    runner = FakeSshRunner(artifact=artifact)

    result = run_staircase_ssh_executor(
        plan,
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        runner=runner,
    )

    remote_scripts = [call[-1] for call in runner.calls]
    assert result["success_count"] == 1
    assert "rm -rf" not in "\n".join(remote_scripts)
    assert "done.json" in "\n".join(remote_scripts)
    assert "git rev-parse HEAD" in "\n".join(remote_scripts)
    row = _step_row(state)
    assert row["status"] == "succeeded"
    assert row["attempts"] == 1


def test_ssh_executor_does_not_trust_remote_success_without_local_postcondition(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "missing.json"
    queue = _queue(artifact)
    state = tmp_path / "queue.sqlite"
    runner = FakeSshRunner(artifact=artifact, create_artifact=False)

    result = run_staircase_ssh_executor(
        _plan(queue),
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        runner=runner,
    )

    assert result["failure_count"] == 1
    row = _step_row(state)
    assert row["status"] == "failed"
    last_event = json.loads(row["last_event_json"])
    assert last_event["returncode"] == 0
    assert last_event["failed_postconditions"]


def test_ssh_executor_blocks_future_executor_contract_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    state = tmp_path / "queue.sqlite"
    runner = FakeSshRunner(artifact=artifact)

    result = run_staircase_ssh_executor(
        _plan(queue, executor="ssh_experiment_queue_future"),
        queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        runner=runner,
    )

    assert runner.calls == []
    assert result["blocked_count"] == 1
    assert "machine_executor_not_enabled:ssh_experiment_queue_future" in result["task_results"][0]["blockers"]
    row = _step_row(state)
    assert row["status"] == "queued"
    assert row["attempts"] == 0


def test_ssh_executor_rejects_stale_plan_hash_before_ssh(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    original_queue = _queue(artifact)
    plan = _plan(original_queue)
    changed_queue = _queue(artifact)
    changed_queue["experiments"][0]["steps"][0]["command"] = ["python", "-c", "print('changed')"]
    plan["source_refs"][0]["queue_hash"] = _sha256_json(changed_queue)
    state = tmp_path / "queue.sqlite"
    runner = FakeSshRunner(artifact=artifact)

    result = run_staircase_ssh_executor(
        plan,
        changed_queue,
        state_path=state,
        repo_root=REPO_ROOT,
        execute=True,
        allow_noncanonical_state=True,
        runner=runner,
    )

    assert runner.calls == []
    assert result["blocked_count"] == 1
    assert "task_step_hash_mismatch:command_hash,definition_hash" in result["task_results"][0]["blockers"]
    row = _step_row(state)
    assert row["status"] == "queued"
    assert row["attempts"] == 0


def test_ssh_executor_cli_dry_run_is_operator_callable(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    queue_path = tmp_path / "queue.json"
    plan_path = tmp_path / "plan.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(REPO_ROOT / "tools" / "run_staircase_ssh_executor.py"),
            "--plan",
            str(plan_path),
            "--queue",
            str(queue_path),
            "--state",
            str(tmp_path / "queue.sqlite"),
            "--remote-repo-root",
            "sshbox=/remote/pact",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "staircase_ssh_execution_result.v1"
    assert payload["execute"] is False
    assert payload["selected_count"] == 1


def test_ssh_executor_cli_requires_rationale_for_dirty_remote_git(tmp_path: Path) -> None:
    artifact = tmp_path / "done.json"
    queue = _queue(artifact)
    plan = _plan(queue)
    queue_path = tmp_path / "queue.json"
    plan_path = tmp_path / "plan.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(REPO_ROOT / "tools" / "run_staircase_ssh_executor.py"),
            "--plan",
            str(plan_path),
            "--queue",
            str(queue_path),
            "--state",
            str(tmp_path / "queue.sqlite"),
            "--allow-dirty-remote-git",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--dirty-remote-git-rationale" in completed.stderr
