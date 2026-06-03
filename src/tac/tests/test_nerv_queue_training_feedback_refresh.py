# SPDX-License-Identifier: MIT
"""Tests for queue-owned NeRV training-feedback refresh."""

from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.nerv_queue_training_feedback_refresh import (
    refresh_nerv_queue_training_feedback,
    render_refresh_markdown,
    write_nerv_queue_training_feedback_refresh,
)
from tools.build_nerv_long_training_campaign_plan import _load_feedback_sources


def test_refresh_queue_training_feedback_harvests_running_hinerv_row(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    telemetry = output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl"
    telemetry.parent.mkdir(parents=True)
    rows = [
        {
            "epoch": epoch,
            "learning_rate": 2.7e-5,
            "loss_components": {
                "loss_part_pose_distill": 2.0,
                "loss_part_distill": 6.3,
                "loss_part_weighted_distill": 25.2,
            },
            "per_axis_decomposition": {
                "pose": 3.0,
                "seg": 6.4 - min(epoch, 127) * 0.0002,
            },
        }
        for epoch in range(160)
    ]
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue.json"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "test_queue",
        "experiments": [
            {
                "id": "hi_nerv_candidate_adamw",
                "steps": [
                    {
                        "id": "run_mlx_first_campaign_row",
                        "command": [
                            "uv",
                            "run",
                            "python",
                            "tools/run_compact_renderer_mlx_spine_runner.py",
                            "--execute-family",
                            "hi_nerv",
                            "--modelsize-candidate-id",
                            "hinerv_np600_test",
                            "--num-pairs",
                            "600",
                            "--output-dir",
                            output_dir.as_posix(),
                        ],
                        "telemetry": {
                            "artifact_paths": [
                                telemetry.as_posix(),
                            ],
                        },
                    },
                ],
            },
        ],
    }
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    summary = {
        "schema": "experiment_queue_summary.v1",
        "steps": [
            {
                "experiment_id": "hi_nerv_candidate_adamw",
                "step_id": "run_mlx_first_campaign_row",
                "status": "running",
            },
        ],
    }

    report = refresh_nerv_queue_training_feedback(
        queue=queue,
        queue_path=queue_path,
        queue_summary=summary,
        output_dir=tmp_path / "feedback",
    )

    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["refreshed_row_count"] == 1
    feedback = report["rows"][0]["row"]
    assert feedback["candidate_id"] == "hinerv_np600_test"
    assert feedback["training_stopped"] is False
    assert feedback["observed_segnet_distillation_weight"] == 4.0
    assert feedback["recommended_segnet_distillation_weight"] == 8.0
    assert (
        feedback["training_control_action"]
        == "checkpoint_then_supersede_with_higher_segnet_weight"
    )
    assert feedback["training_control_should_stop_current_run"] is True
    assert "hi_nerv_segnet_stagnation_telemetry_feedback" in feedback["blockers"]
    assert Path(report["rows"][0]["row_path"]).is_file()

    write_result = write_nerv_queue_training_feedback_refresh(
        report=report,
        output_json=tmp_path / "refresh.json",
        output_jsonl=tmp_path / "refresh.jsonl",
        output_md=tmp_path / "refresh.md",
    )
    assert write_result["row_count"] == 1
    assert Path(write_result["jsonl_path"]).read_text(encoding="utf-8").count("\n") == 1
    markdown = render_refresh_markdown(report)
    assert "recommended_segnet_weight: `8.0`" in markdown
    assert "pose_tail_burst: `False`" in markdown
    assert (
        "training_control_action: `checkpoint_then_supersede_with_higher_segnet_weight`"
        in markdown
    )

    loaded = _load_feedback_sources([tmp_path / "refresh.json"])
    assert len(loaded) == 1
    assert loaded[0]["schema"] == "nerv_candidate_feedback_row.v1"
    assert loaded[0]["candidate_id"] == "hinerv_np600_test"
    assert loaded[0]["recommended_segnet_distillation_weight"] == 8.0
