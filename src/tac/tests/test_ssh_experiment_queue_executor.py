# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
    ready_steps,
)
from comma_lab.scheduler.ssh_experiment_queue_executor import (
    build_remote_git_preflight_command,
    build_remote_shell_command,
    build_rsync_pull_command,
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
        if "git rev-parse HEAD" in remote_script:
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


def _sha256_json(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plan(queue: dict[str, Any], *, executor: str = "ssh_experiment_queue") -> dict[str, Any]:
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id="ssh_executor_fixture_dag",
        resource_pools=[
            {
                "id": "sshbox",
                "label": "SSH test worker",
                "slots": {"local_cpu": 1},
                "memory_gb": 16,
                "disk_gb": 8,
                "tags": ["ssh", "test"],
                "executor": executor,
                "ssh_target": "user@sshbox",
                "remote_repo_root": "/remote/pact",
            }
        ],
    )
    return plan_staircase_dispatch(dag, max_nodes=1)


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


def test_remote_shell_command_uses_queue_argv_shape() -> None:
    command = build_remote_shell_command(
        remote_repo_root="/tmp/pact repo",
        command=["python", "-c", "print('ok')"],
    )

    assert command == "cd '/tmp/pact repo' && python -c 'print('\"'\"'ok'\"'\"')'"


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
        "--",
        "user@host:/remote/pact/out/result.json",
        "/local/pact/out/result.json",
    ]


def test_rsync_pull_command_rejects_unsafe_remote_artifact_paths() -> None:
    for remote_path in (
        "relative/out.json",
        "/remote/../out.json",
        "/remote/out bad.json",
        "/remote/out;touch.json",
    ):
        with pytest.raises(ExperimentQueueError):
            build_rsync_pull_command(
                rsync_binary="rsync",
                ssh_target="user@host",
                remote_path=remote_path,
                local_path="/local/out.json",
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
        if "git rev-parse HEAD" in remote_script:
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
