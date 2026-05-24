# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_materializer_campaign_runner_rejects_bad_resource_concurrency() -> None:
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=0"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu=two"])
    with pytest.raises(SystemExit):
        runner._parse_resource_concurrency(["local_cpu"])
