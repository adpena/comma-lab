# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools import build_hprc_compact_receiver_training_queue as builder
from tools.gate_archive_rate_for_local_replay import build_archive_rate_local_replay_gate


def test_hprc_campaign_queue_scales_and_gates_full_video(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    plan_path = tmp_path / "hprc_plan.json"

    exit_code = builder.main(
        [
            "--repo-root",
            repo_root.as_posix(),
            "--video-path",
            video_path.as_posix(),
            "--output",
            queue_path.as_posix(),
            "--plan-output",
            plan_path.as_posix(),
            "--campaign-pairs",
            "32",
            "--campaign-pairs",
            "128",
            "--campaign-pairs",
            "600",
            "--epochs",
            "1",
            "--timeout-seconds",
            "123",
            "--auth-frontier-score",
            "0.19",
            "--storage-tier",
            f"test={storage_root.as_posix()}",
            "--storage-expected-bytes",
            "1",
            "--storage-reserve-free-gb",
            "0",
            "--allow-local-output-dir",
        ]
    )

    assert exit_code == 0
    queue = json.loads(queue_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert plan["schema"] == builder.HPRC_TRAINING_PLAN_SUITE_SCHEMA
    assert plan["campaign_pairs"] == [32, 128, 600]
    assert len(queue["experiments"]) == 3

    partial_steps = [step["id"] for step in queue["experiments"][0]["steps"]]
    assert partial_steps == [
        "run_local_training",
        "write_hprc_campaign_followup_report",
    ]
    full_steps = [step["id"] for step in queue["experiments"][2]["steps"]]
    assert full_steps == [
        "run_local_training",
        "write_hprc_campaign_followup_report",
        "gate_archive_rate_before_local_replay",
        "run_local_cpu_replay",
        "gate_exact_cpu_after_local_replay",
    ]
    train_command = queue["experiments"][2]["steps"][0]["command"]
    assert "--skip-runtime-consumption-proof" not in train_command
    train_postconditions = queue["experiments"][2]["steps"][0]["postconditions"]
    assert any(
        condition["type"] == "path_exists"
        and condition["path"].endswith("receiver_proof/hprc_receiver_proof.json")
        for condition in train_postconditions
    )
    rate_gate = queue["experiments"][2]["steps"][2]
    assert rate_gate["on_postcondition_failure"] == "skipped"
    assert rate_gate["postconditions"][-1]["key"] == "local_replay_recommended"
    replay_step = queue["experiments"][2]["steps"][3]
    assert replay_step["requires"] == ["gate_archive_rate_before_local_replay"]
    gate_command = queue["experiments"][2]["steps"][4]["command"]
    assert "--success-on-blocked" in gate_command
    assert "--auth-frontier-score" in gate_command


def test_hprc_campaign_queue_binds_optional_z8_followups(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    z8_archive = tmp_path / "z8.bin"
    z8_archive.write_bytes(b"z8")
    z8_surface = tmp_path / "surface.npz"
    z8_surface.write_bytes(b"not-loaded-by-builder")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"

    assert (
        builder.main(
            [
                "--repo-root",
                repo_root.as_posix(),
                "--video-path",
                video_path.as_posix(),
                "--output",
                queue_path.as_posix(),
                "--campaign-pairs",
                "32",
                "--z8-archive-bin",
                z8_archive.as_posix(),
                "--z8-surface",
                z8_surface.as_posix(),
                "--storage-tier",
                f"test={storage_root.as_posix()}",
                "--storage-expected-bytes",
                "1",
                    "--storage-reserve-free-gb",
                    "0",
                    "--allow-local-output-dir",
                ]
            )
        == 0
    )

    queue = json.loads(queue_path.read_text())
    step_ids = [step["id"] for step in queue["experiments"][0]["steps"]]
    assert "build_z8_full_video_p18_p19_allocator_plan" in step_ids
    assert "materialize_z8_p18_p19_allocator_candidate" in step_ids
    materializer = next(
        step
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "materialize_z8_p18_p19_allocator_candidate"
    )
    assert "--entropy-code-quantized-details" in materializer["command"]
    assert materializer["requires"] == ["build_z8_full_video_p18_p19_allocator_plan"]


def test_archive_rate_gate_blocks_replay_when_rate_alone_loses() -> None:
    report = build_archive_rate_local_replay_gate(
        training_result={
            "artifact": {
                "archive_path": "archive.zip",
                "archive_sha256": "a" * 64,
                "archive_bytes": 1_092_409,
            }
        },
        training_result_path="result.json",
        auth_frontier_score=0.1919853363,
        local_baseline_score=None,
        min_local_improvement=0.0,
    )

    assert report["schema"] == "archive_rate_local_replay_gate.v1"
    assert report["local_replay_recommended"] is False
    assert "archive_rate_term_not_below_target_before_distortion" in report["blockers"]
    assert report["ready_for_exact_eval_dispatch"] is False
