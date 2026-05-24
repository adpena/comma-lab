# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
)
from tools import run_byte_shaving_materializer_campaign as runner


def test_materializer_campaign_runner_builds_queue_owned_followup_command(
    tmp_path: Path,
) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--materializer-contexts",
            str(tmp_path / "contexts.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--queue-id",
            "materializer_campaign_fixture",
            "--materializer-resource-concurrency",
            "local_mlx=2",
            "--include-storage-preflight",
            "--storage-expected-workload-root",
            str(tmp_path / "campaign" / "work"),
            "--storage-workload-subdir",
            "work",
            "--proactive-cleanup-root",
            "experiments/results",
        ]
    )

    command = runner._build_queue_command(args, run_dir=tmp_path / "campaign")

    assert command[:2] == [
        runner.sys.executable,
        "tools/build_byte_shaving_campaign_queue.py",
    ]
    assert "--include-materializer-exact-readiness-followup" in command
    assert "--materializer-execution-queue-out" in command
    assert "--materializer-resource-concurrency" in command
    assert "local_mlx=2" in command
    assert "--include-materializer-scheduler-preflight" in command
    assert "--materializer-scheduler-proactive-cleanup-execute" in command
    assert "--dispatch-mode" not in command
    assert "--allow-paid-dispatch-queue" not in command


def test_materializer_campaign_runner_emits_staircase_artifacts(tmp_path: Path) -> None:
    queue = normalize_queue_definition(
        {
            "schema": "experiment_queue.v1",
            "queue_id": "campaign_staircase_fixture",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
            "experiments": [
                {
                    "id": "candidate",
                    "priority": 1,
                    "steps": [
                        {
                            "id": "materialize",
                            "command": ["python", "-c", "print('ok')"],
                            "resources": {"kind": "local_cpu"},
                            "postconditions": [
                                {"type": "path_exists", "path": str(tmp_path / "done.json")}
                            ],
                        }
                    ],
                }
            ],
        }
    )
    queue_path = tmp_path / "queue.json"
    state_path = tmp_path / "queue.sqlite"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--emit-staircase-plan",
            "--staircase-resource-pool",
            "sshbox:local_cpu=1,memory_gb=8,disk_gb=8,"
            "executor=ssh_experiment_queue,ssh_target=user@sshbox,"
            "remote_repo_root=/remote/pact",
        ]
    )
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()

    result = runner._build_staircase_artifacts(
        args,
        run_dir=run_dir,
        execution_queue=queue_path,
        state_path=state_path,
        queue=queue,
    )

    assert result["selected_count"] == 1
    plan = json.loads((run_dir / "staircase_dispatch_plan.json").read_text(encoding="utf-8"))
    task = plan["dask_task_specs"][0]
    assert task["machine"]["executor"] == "ssh_experiment_queue"
    assert task["machine"]["remote_repo_root"] == "/remote/pact"
    assert task["queue_state_writeback"]["required"] is True
    assert plan["score_claim"] is False


def test_materializer_campaign_runner_builds_ssh_dry_run_command(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--run-dir",
            str(tmp_path / "campaign"),
            "--staircase-ssh-dry-run",
            "--staircase-ssh-machine-id",
            "sshbox",
            "--staircase-ssh-remote-repo-root",
            "sshbox=/remote/pact",
        ]
    )

    command = runner._ssh_executor_dry_run_command(
        args,
        execution_queue=tmp_path / "queue.json",
        state_path=tmp_path / "queue.sqlite",
        staircase_plan_path=tmp_path / "plan.staircase.json",
        run_dir=tmp_path / "campaign",
    )

    assert command[:2] == [
        runner.sys.executable,
        "tools/run_staircase_ssh_executor.py",
    ]
    assert "--execute" not in command
    assert "--machine-id" in command
    assert "sshbox" in command
    assert "--remote-repo-root" in command
    assert "sshbox=/remote/pact" in command


def test_materializer_campaign_runner_rejects_bad_resource_concurrency() -> None:
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=0"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=two"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu"])


def test_materializer_campaign_runner_rejects_bad_ssh_remote_root_mapping() -> None:
    with pytest.raises(SystemExit):
        runner._parse_remote_repo_roots(["sshbox"])


def test_materializer_campaign_runner_requires_json_from_ssh_dry_run() -> None:
    with pytest.raises(SystemExit, match="did not emit a JSON object"):
        runner._require_json_stdout(
            runner.CommandResult(
                command=["ssh-dry-run"],
                returncode=0,
                stdout="not-json",
                stderr="",
                elapsed_seconds=0.0,
            ),
            label="staircase SSH executor dry-run",
        )
    with pytest.raises(SystemExit, match="failed"):
        runner._require_json_stdout(
            runner.CommandResult(
                command=["ssh-dry-run"],
                returncode=2,
                stdout="",
                stderr="bad plan",
                elapsed_seconds=0.0,
            ),
            label="staircase SSH executor dry-run",
        )
