# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import build_hprc_compact_receiver_training_queue as builder
from tools.gate_archive_rate_for_local_replay import build_archive_rate_local_replay_gate
from tools.gate_hprc_mlx_prefilter_for_local_replay import (
    build_hprc_mlx_prefilter_local_replay_gate,
)


def test_hprc_campaign_queue_scales_and_gates_full_video(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    plan_path = tmp_path / "hprc_plan.json"
    p19_artifact = repo_root / ".omx" / "research" / "p19_posenet_null_pairs.json"
    p18_artifact = repo_root / ".omx" / "research" / "p18_segnet_region_waterfill.json"
    p19_artifact.parent.mkdir(parents=True)
    p19_artifact.write_text("{}")
    p18_artifact.write_text("{}")

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
            "--hprc-rate-collapse-residual-collapse-schedule",
            "dz0_qd10",
            "--hprc-rate-collapse-p19-posenet-null-pairs",
            p19_artifact.as_posix(),
            "--hprc-rate-collapse-p18-segnet-region-waterfill",
            p18_artifact.as_posix(),
            "--hprc-rate-collapse-importance-coarsen-quantile",
            "0.9",
            "--hprc-rate-collapse-importance-selection-domain",
            "global_weighted",
            "--hprc-rate-collapse-importance-protected-spec",
            "dz0_qd1",
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
        "transcode_hprc_rate_collapse",
        "write_hprc_campaign_followup_report",
    ]
    full_steps = [step["id"] for step in queue["experiments"][2]["steps"]]
    assert full_steps == [
        "run_local_training",
        "transcode_hprc_rate_collapse",
        "write_hprc_campaign_followup_report",
        "gate_archive_rate_before_local_replay",
        "prove_hprc_rate_collapsed_receiver",
        "run_local_cpu_replay",
        "gate_exact_cpu_after_local_replay",
        "write_hprc_campaign_post_replay_report",
    ]
    train_command = queue["experiments"][2]["steps"][0]["command"]
    assert "--skip-runtime-consumption-proof" in train_command
    train_postconditions = queue["experiments"][2]["steps"][0]["postconditions"]
    assert not any(
        condition["type"] == "path_exists"
        and condition["path"].endswith("receiver_proof/hprc_receiver_proof.json")
        for condition in train_postconditions
    )
    rate_collapse = queue["experiments"][2]["steps"][1]
    assert "--skip-receiver-proof" in rate_collapse["command"]
    assert "--enable-lossy-residual-collapse" in rate_collapse["command"]
    assert "--target-rate-term" in rate_collapse["command"]
    assert rate_collapse["command"][rate_collapse["command"].index("--target-rate-term") + 1] == "0.15"
    assert "--residual-collapse-schedule" in rate_collapse["command"]
    assert (
        rate_collapse["command"][rate_collapse["command"].index("--residual-collapse-schedule") + 1]
        == "dz0_qd10"
    )
    assert "--training-backend" in train_command
    assert train_command[train_command.index("--training-backend") + 1] == "auto"
    assert queue["experiments"][2]["steps"][0]["resources"]["kind"] == "local_mlx"
    assert "--p19-posenet-null-pairs" in rate_collapse["command"]
    assert "--p18-segnet-region-waterfill" in rate_collapse["command"]
    assert "--importance-coarsen-quantile" in rate_collapse["command"]
    assert (
        rate_collapse["command"][rate_collapse["command"].index("--importance-coarsen-quantile") + 1]
        == "0.9"
    )
    assert "--importance-selection-domain" in rate_collapse["command"]
    assert (
        rate_collapse["command"][rate_collapse["command"].index("--importance-selection-domain") + 1]
        == "global_weighted"
    )
    assert "--importance-protected-spec" in rate_collapse["command"]
    assert (
        rate_collapse["command"][rate_collapse["command"].index("--importance-protected-spec") + 1]
        == "dz0_qd1"
    )
    rate_gate = queue["experiments"][2]["steps"][3]
    assert rate_gate["on_postcondition_failure"] == "skipped"
    assert rate_gate["postconditions"][-1]["key"] == "local_replay_recommended"
    assert rate_gate["requires"] == ["write_hprc_campaign_followup_report"]
    assert rate_gate["command"][rate_gate["command"].index("--min-local-improvement") + 1] == "0.04"
    proof_step = queue["experiments"][2]["steps"][4]
    assert proof_step["requires"] == ["gate_archive_rate_before_local_replay"]
    assert "--skip-receiver-proof" not in proof_step["command"]
    proof_conditions = {(row["path"], row.get("key")) for row in proof_step["postconditions"]}
    proof_path = next(
        row["path"]
        for row in proof_step["postconditions"]
        if row.get("path", "").endswith("receiver_proof/hprc_receiver_proof.json")
    )
    assert (proof_path, "receiver_contract_satisfied") in proof_conditions
    assert (proof_path, "runtime_consumption_proof_ready") in proof_conditions
    assert (proof_path, "blockers") in proof_conditions
    replay_step = queue["experiments"][2]["steps"][5]
    assert replay_step["requires"] == ["prove_hprc_rate_collapsed_receiver"]
    assert any(
        command_part.endswith("hprc_rate_collapse/best_archive_export/archive.zip")
        for command_part in replay_step["command"]
    )


def test_hprc_rate_collapse_defaults_to_p18_p19_waterfill(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    p19_artifact = repo_root / ".omx" / "research" / "p19_posenet_null_pairs.json"
    p18_artifact = repo_root / ".omx" / "research" / "p18_segnet_region_waterfill.json"
    p19_artifact.parent.mkdir(parents=True)
    p19_artifact.write_text("{}")
    p18_artifact.write_text("{}")

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
                "600",
                "--epochs",
                "1",
                "--auth-frontier-score",
                "0.19",
                "--hprc-rate-collapse-p19-posenet-null-pairs",
                p19_artifact.as_posix(),
                "--hprc-rate-collapse-p18-segnet-region-waterfill",
                p18_artifact.as_posix(),
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
    command = queue["experiments"][0]["steps"][1]["command"]
    assert command[command.index("--residual-collapse-schedule") + 1] == "dz0_qd10"
    assert command[command.index("--importance-selection-domain") + 1] == "eligible_low"
    assert command[command.index("--importance-coarsen-quantile") + 1] == "0.25"
    assert command[command.index("--importance-protected-spec") + 1] == "dz0_qd1"
    gate_command = queue["experiments"][0]["steps"][6]["command"]
    assert "--success-on-blocked" in gate_command
    assert "--auth-frontier-score" in gate_command
    post_replay = queue["experiments"][0]["steps"][7]
    assert post_replay["requires"] == ["gate_exact_cpu_after_local_replay"]
    assert any(
        command_part.endswith("hprc_queue_post_replay_report.json")
        for command_part in post_replay["command"]
    )


def test_hprc_rate_collapse_reuses_native_p18_p19_waterfill_by_default(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    p19_artifact = repo_root / ".omx" / "research" / "p19_posenet_null_pairs.json"
    p18_artifact = repo_root / ".omx" / "research" / "p18_segnet_region_waterfill.json"
    p19_artifact.parent.mkdir(parents=True)
    p19_artifact.write_text("{}")
    p18_artifact.write_text("{}")

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
                "600",
                "--epochs",
                "1",
                "--enable-native-rate-aware-hprc",
                "--native-rate-p19-posenet-null-pairs",
                p19_artifact.as_posix(),
                "--native-rate-p18-segnet-region-waterfill",
                p18_artifact.as_posix(),
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
        == 0
    )

    queue = json.loads(queue_path.read_text())
    steps = queue["experiments"][0]["steps"]
    assert [step["id"] for step in steps[:3]] == [
        "build_hprc_native_rate_residual_protection_surface",
        "run_local_training",
        "transcode_hprc_rate_collapse",
    ]
    command = steps[2]["command"]
    assert "--enable-lossy-residual-collapse" in command
    assert command[command.index("--residual-collapse-schedule") + 1] == "dz0_qd10"
    assert command[command.index("--p19-posenet-null-pairs") + 1].endswith(
        "p19_posenet_null_pairs.json"
    )
    assert command[command.index("--p18-segnet-region-waterfill") + 1].endswith(
        "p18_segnet_region_waterfill.json"
    )
    assert command[command.index("--importance-selection-domain") + 1] == "eligible_low"
    assert command[command.index("--importance-protected-spec") + 1] == "dz0_qd1"


def test_hprc_campaign_queue_refuses_missing_rate_artifacts_before_write(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"

    with pytest.raises(ValueError, match="HPRC rate-collapse P19 PoseNet-null artifact"):
        builder.main(
            [
                "--repo-root",
                repo_root.as_posix(),
                "--video-path",
                video_path.as_posix(),
                "--output",
                queue_path.as_posix(),
                "--campaign-pairs",
                "600",
                "--hprc-rate-collapse-p19-posenet-null-pairs",
                (repo_root / ".omx" / "research" / "missing_p19.json").as_posix(),
                "--storage-tier",
                f"test={storage_root.as_posix()}",
                "--storage-expected-bytes",
                "1",
                "--storage-reserve-free-gb",
                "0",
                "--allow-local-output-dir",
            ]
        )

    assert not queue_path.exists()


def test_hprc_campaign_queue_refuses_missing_native_protection_input(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"

    with pytest.raises(ValueError, match="native rate residual protection npy"):
        builder.main(
            [
                "--repo-root",
                repo_root.as_posix(),
                "--video-path",
                video_path.as_posix(),
                "--output",
                queue_path.as_posix(),
                "--campaign-pairs",
                "600",
                "--enable-native-rate-aware-hprc",
                "--native-rate-residual-protection-npy",
                (repo_root / ".omx" / "research" / "missing_residual_protection.npy").as_posix(),
                "--storage-tier",
                f"test={storage_root.as_posix()}",
                "--storage-expected-bytes",
                "1",
                "--storage-reserve-free-gb",
                "0",
                "--allow-local-output-dir",
            ]
        )

    assert not queue_path.exists()


def test_hprc_campaign_queue_refuses_native_protection_shape_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    protection = repo_root / ".omx" / "research" / "wrong_shape.npy"
    protection.parent.mkdir(parents=True)
    np.save(protection, np.zeros((64, 16, 16, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="native rate residual protection npy shape mismatch"):
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
                "--enable-native-rate-aware-hprc",
                "--native-rate-residual-protection-npy",
                protection.as_posix(),
                "--residual-grid-h",
                "24",
                "--residual-grid-w",
                "32",
                "--storage-tier",
                f"test={storage_root.as_posix()}",
                "--storage-expected-bytes",
                "1",
                "--storage-reserve-free-gb",
                "0",
                "--allow-local-output-dir",
            ]
        )

    assert not queue_path.exists()


def test_hprc_campaign_queue_can_insert_mlx_prefilter_before_cpu_replay(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    reference_cache = repo_root / "mlx_reference_cache"
    reference_cache.mkdir()
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
                "600",
                "--epochs",
                "1",
                "--auth-frontier-score",
                "0.19",
                "--enable-hprc-mlx-prefilter-before-local-replay",
                "--hprc-mlx-prefilter-reference-cache-dir",
                reference_cache.as_posix(),
                "--hprc-mlx-prefilter-max-pairs",
                "600",
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
    steps = queue["experiments"][0]["steps"]
    step_ids = [step["id"] for step in steps]
    assert "run_hprc_full_video_mlx_prefilter" in step_ids
    assert "gate_hprc_mlx_prefilter_before_local_replay" in step_ids
    prefilter = steps[step_ids.index("run_hprc_full_video_mlx_prefilter")]
    gate = steps[step_ids.index("gate_hprc_mlx_prefilter_before_local_replay")]
    replay = steps[step_ids.index("run_local_cpu_replay")]
    assert prefilter["requires"] == ["prove_hprc_rate_collapsed_receiver"]
    assert prefilter["resources"]["kind"] == "local_mlx"
    assert "--no-sections" in prefilter["command"]
    assert "--cache-materialization-mode" in prefilter["command"]
    assert "hprc-direct" in prefilter["command"]
    assert gate["requires"] == ["run_hprc_full_video_mlx_prefilter"]
    assert gate["on_postcondition_failure"] == "skipped"
    assert "--max-mlx-score-for-local-replay" in gate["command"]
    assert gate["postconditions"][-1]["key"] == "local_replay_recommended"
    assert replay["requires"] == ["gate_hprc_mlx_prefilter_before_local_replay"]


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
    assert "transcode_hprc_rate_collapse" in step_ids
    assert "build_z8_full_video_p18_p19_allocator_plan" in step_ids
    assert "materialize_z8_p18_p19_allocator_candidate" in step_ids
    materializer = next(
        step
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "materialize_z8_p18_p19_allocator_candidate"
    )
    assert "--entropy-code-quantized-details" in materializer["command"]
    assert materializer["requires"] == ["build_z8_full_video_p18_p19_allocator_plan"]


def test_hprc_campaign_queue_builds_native_rate_surface_before_training(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    p19_artifact = repo_root / ".omx" / "research" / "p19.json"
    p18_artifact = repo_root / ".omx" / "research" / "p18.json"
    p19_artifact.parent.mkdir(parents=True)
    p19_artifact.write_text("{}")
    p18_artifact.write_text("{}")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    plan_path = tmp_path / "hprc_plan.json"

    assert (
        builder.main(
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
                "--enable-native-rate-aware-hprc",
                "--native-rate-residual-prox-weight",
                "0.25",
                "--native-rate-p19-posenet-null-pairs",
                p19_artifact.as_posix(),
                "--native-rate-p18-segnet-region-waterfill",
                p18_artifact.as_posix(),
                "--disable-hprc-rate-collapse",
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
    plan = json.loads(plan_path.read_text())
    steps = queue["experiments"][0]["steps"]
    assert steps[0]["id"] == "build_hprc_native_rate_residual_protection_surface"
    assert steps[0]["resources"]["kind"] == "local_cpu"
    assert steps[1]["id"] == "run_local_training"
    assert steps[1]["requires"] == ["build_hprc_native_rate_residual_protection_surface"]
    train_command = steps[1]["command"]
    assert "--native-rate-aware" in train_command
    assert "--rate-aware-residual-protection-npy" in train_command
    protection_arg = train_command[train_command.index("--rate-aware-residual-protection-npy") + 1]
    assert protection_arg.endswith("hprc_native_rate_surface/residual_protection.npy")
    assert plan["schema"] == builder.HPRC_TRAINING_PLAN_SUITE_SCHEMA
    assert plan["native_rate_surface_steps"][0]["schema"] == "hprc_native_rate_surface_step_config.v1"


def test_hprc_campaign_queue_scales_with_mlx_prefilter_and_protected_pathway(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
    p19_artifact = repo_root / ".omx" / "research" / "p19.json"
    p18_artifact = repo_root / ".omx" / "research" / "p18.json"
    p19_artifact.parent.mkdir(parents=True)
    p19_artifact.write_text("{}")
    p18_artifact.write_text("{}")
    storage_root = tmp_path / "ssd"
    queue_path = tmp_path / "hprc_queue.json"
    plan_path = tmp_path / "hprc_plan.json"

    assert (
        builder.main(
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
                "--training-backend",
                "mlx",
                "--enable-native-rate-aware-hprc",
                "--native-rate-residual-prox-weight",
                "0.25",
                "--native-rate-p19-posenet-null-pairs",
                p19_artifact.as_posix(),
                "--native-rate-p18-segnet-region-waterfill",
                p18_artifact.as_posix(),
                "--enable-protected-residual-pathway",
                "--protected-residual-grid-h",
                "32",
                "--protected-residual-grid-w",
                "32",
                "--enable-hprc-mlx-prefilter-before-local-replay",
                "--hprc-mlx-prefilter-max-score-for-local-replay",
                "0.5",
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
        == 0
    )

    queue = json.loads(queue_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert plan["campaign_pairs"] == [32, 128, 600]
    full = queue["experiments"][2]
    step_ids = [step["id"] for step in full["steps"]]
    assert "run_hprc_full_video_mlx_prefilter" in step_ids
    assert "gate_hprc_mlx_prefilter_before_local_replay" in step_ids
    train_step = next(step for step in full["steps"] if step["id"] == "run_local_training")
    train_command = train_step["command"]
    assert "--training-backend" in train_command
    assert train_command[train_command.index("--training-backend") + 1] == "mlx"
    assert "--enable-protected-residual-pathway" in train_command
    assert train_command[train_command.index("--protected-residual-grid-h") + 1] == "32"
    assert train_command[train_command.index("--protected-residual-grid-w") + 1] == "32"
    assert "--rate-aware-residual-protection-npy" in train_command
    assert train_step["requires"] == ["build_hprc_native_rate_residual_protection_surface"]
    assert any(
        condition.get("key") == "protected_highres_residual_pathway.enabled"
        and condition.get("equals") is True
        for condition in train_step["postconditions"]
    )
    mlx_step = next(
        step for step in full["steps"] if step["id"] == "run_hprc_full_video_mlx_prefilter"
    )
    assert mlx_step["resources"]["kind"] == "local_mlx"
    assert "--max-pairs" in mlx_step["command"]
    assert mlx_step["command"][mlx_step["command"].index("--max-pairs") + 1] == "600"
    replay_step = next(step for step in full["steps"] if step["id"] == "run_local_cpu_replay")
    assert replay_step["requires"] == ["gate_hprc_mlx_prefilter_before_local_replay"]
    full_plan = plan["plans"][2]
    assert full_plan["candidate_params"]["protected_highres_residual_pathway_enabled"] is True
    assert full_plan["candidate_params"]["protected_residual_grid_h"] == 32
    assert full_plan["candidate_params"]["protected_residual_grid_w"] == 32


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


def test_hprc_mlx_prefilter_gate_blocks_cpu_replay_for_obvious_local_loser(
    tmp_path: Path,
) -> None:
    response = tmp_path / "baseline_mlx_response.json"
    response.write_text(
        json.dumps(
            {
                "canonical_score": 25.59,
                "axis_tag": "[macOS-MLX research-signal]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_hprc_mlx_prefilter_local_replay_gate(
        profile={
            "schema": "hprc_mlx_component_neutralization_profile.v1",
            "scorer_batch_pairs": 1,
            "scope_status": {"full_video": "executed"},
            "variant_rows": [
                {
                    "variant_id": "baseline",
                    "mlx_response": response.as_posix(),
                }
            ],
        },
        profile_path=tmp_path / "profile.json",
        auth_frontier_score=0.1919853363,
        local_baseline_score=None,
        min_local_improvement=0.0,
    )

    assert report["schema"] == "hprc_mlx_prefilter_local_replay_gate.v1"
    assert report["local_replay_recommended"] is False
    assert "mlx_score_not_below_target" in report["blockers"]
    assert "mlx_score_above_hard_demote_threshold" in report["blockers"]
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hprc_campaign_queue_passes_pr95_pose_guard_curriculum_preset(tmp_path: Path) -> None:
    repo_root = tmp_path
    video_path = repo_root / "upstream" / "videos" / "0.mkv"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake contest video bytes")
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
                "--epochs",
                "8",
                "--curriculum-preset",
                "hprc_pr95_pose_guard_rate_v1",
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
    train_command = queue["experiments"][0]["steps"][0]["command"]
    assert "--curriculum-preset" in train_command
    assert (
        train_command[train_command.index("--curriculum-preset") + 1]
        == "hprc_pr95_pose_guard_rate_v1"
    )
