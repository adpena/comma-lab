# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from comma_lab.scheduler.experiment_queue import _step_artifact_telemetry


def test_compact_renderer_runner_artifacts_are_inferred_from_command(
    tmp_path: Path,
) -> None:
    out = tmp_path / "runs" / "hi_nerv"
    out_rel = out.relative_to(tmp_path)
    step = {
        "command": [
            "uv",
            "run",
            "python",
            "tools/run_compact_renderer_mlx_spine_runner.py",
            "--execute-family",
            "hi_nerv",
            "--output-dir",
            out.as_posix(),
        ],
        "postconditions": [],
    }

    records = _step_artifact_telemetry(step, repo=tmp_path)

    by_path = {record["path"]: record for record in records}
    assert (
        out_rel / "compact_renderer_mlx_spine_runner_startup.json"
    ).as_posix() in by_path
    assert (
        out_rel / "compact_renderer_mlx_spine_runner_report.json"
    ).as_posix() in by_path
    assert (
        out_rel / "hi_nerv_mlx_training" / "telemetry.jsonl"
    ).as_posix() in by_path
    assert (
        out_rel / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl"
    ).as_posix() in by_path
    assert all(
        record["source"] == "step.command.inferred_compact_renderer_output_artifacts"
        for record in by_path.values()
    )


def test_compact_renderer_inferred_artifacts_deduplicate_explicit_telemetry(
    tmp_path: Path,
) -> None:
    out = tmp_path / "runs" / "snerv"
    out_rel = out.relative_to(tmp_path)
    startup = out / "compact_renderer_mlx_spine_runner_startup.json"
    startup_rel = startup.relative_to(tmp_path)
    step = {
        "command": [
            "python",
            "tools/run_compact_renderer_mlx_spine_runner.py",
            "--execute-family",
            "snerv",
            "--output-dir",
            out.as_posix(),
        ],
        "telemetry": {"artifact_paths": [startup.as_posix()]},
        "postconditions": [{"path": startup.as_posix()}],
    }

    records = _step_artifact_telemetry(step, repo=tmp_path)

    paths = [record["path"] for record in records]
    assert paths.count(startup_rel.as_posix()) == 1
    assert (
        out_rel / "snerv_mlx_training" / "telemetry.jsonl"
    ).as_posix() in paths
