# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.local_training_queue import build_local_training_execution_queue
from tac.local_acceleration.pr95_hnerv_mlx_contract import (
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.optimization.representation_training_probe_integration import (
    adapt_representation_training_manifest_to_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pr95_mlx_plan_only_cli_builds_queueable_local_mlx_plan(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pr95_mlx_stage8_plan"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_timing_smoke.py"),
            "--stage",
            "8",
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--synthetic-pairs",
            "4",
            "--seed",
            "23",
            "--base-channels",
            "36",
            "--output-dir",
            str(output_dir),
            "--write-byte-closed-smoke",
            "--write-pr95-public-archive-export",
            "--prove-pr95-runtime-consumption",
            "--runtime-proof-max-output-bytes",
            "7000000",
            "--write-mlx-gpu-drift-attestation",
            "--write-source-video-preprocess-smoke",
            "--source-video-path",
            "upstream/videos/0.mkv",
            "--source-video-upstream-dir",
            "upstream",
            "--source-video-pair-index",
            "0",
            "--source-video-output-hw",
            "384,512",
            "--source-video-gradient-shape",
            "1,2,8,10,3",
            "--train-on-source-video-pairs",
            "--source-video-loss-surface",
            "rgb_yuv6_mse",
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    representation_plan = json.loads(
        (output_dir / "representation_training_plan.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["schema"] == "pr95_hnerv_mlx_timing_smoke_plan_summary_v1"
    assert plan["schema"] == "representation_training_probe_plan_v1"
    assert plan["representation_family_class"] == "hnerv_variant"
    assert plan["stage_count"] == 1
    assert plan["stage_index"] == 8
    assert plan["optimizer_descriptor_id"] == "pr95_stage8_muon_adamw_mlx"
    assert PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER not in plan["dispatch_blockers"]
    assert "synthetic_targets_do_not_establish_contest_quality" not in plan[
        "dispatch_blockers"
    ]
    assert PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER not in plan[
        "dispatch_blockers"
    ]
    assert PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER in plan[
        "dispatch_blockers"
    ]
    assert PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER in plan["dispatch_blockers"]
    assert len(plan["optimizer_config_sha256"]) == 64
    assert plan["parameter_group_lr_policy_id"] == (
        "embedding_theta1_hidden_muon_adamw"
    )
    execution = plan["recommended_execution"]
    assert execution["tool"] == "tools/run_pr95_mlx_timing_smoke.py"
    assert execution["training_backend"] == "mlx"
    assert execution["resource_kind"] == "local_mlx"
    assert execution["optimizer_descriptor_id"] == "pr95_stage8_muon_adamw_mlx"
    assert "--optimizer-descriptor-id" in execution["python_command_args"]
    assert "--allow-existing-output-dir" in execution["python_command_args"]
    assert "--write-byte-closed-smoke" in execution["python_command_args"]
    assert "--write-pr95-public-archive-export" in execution["python_command_args"]
    assert "--prove-pr95-runtime-consumption" in execution["python_command_args"]
    assert "--write-mlx-gpu-drift-attestation" in execution["python_command_args"]
    assert execution["mlx_gpu_forward_drift_attestation"].endswith(
        "mlx_gpu_forward_drift_attestation.json"
    )
    assert execution["archive_export_manifest"].endswith(
        "pr95_public_archive_export.json"
    )
    assert execution["runtime_consumption_proof"].endswith(
        "runtime_consumption_proof.json"
    )
    assert execution["source_video_preprocess_smoke"].endswith(
        "source_video_preprocess_smoke.json"
    )
    assert execution["source_video_pair_indices"] == [0]
    assert "--write-source-video-preprocess-smoke" in execution["python_command_args"]
    assert "--source-video-path" in execution["python_command_args"]
    assert "upstream/videos/0.mkv" in execution["python_command_args"]
    assert "--source-video-output-hw" in execution["python_command_args"]
    assert "384,512" in execution["python_command_args"]
    assert any(
        condition["type"] == "json_equals"
        and condition["path"].endswith("runtime_consumption_proof.json")
        and condition["key"] == "runtime_consumption_proven"
        and condition["equals"] is True
        for condition in execution["extra_artifact_postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"].endswith("mlx_gpu_forward_drift_attestation.json")
        and condition["key"] == "mlx_device"
        and condition["equals"] == "gpu"
        for condition in execution["extra_artifact_postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"].endswith("source_video_preprocess_smoke.json")
        and condition["key"] == "source_video_loader_ready"
        and condition["equals"] is True
        for condition in execution["extra_artifact_postconditions"]
    )
    assert any(
        condition["type"] == "json_array_contains"
        and condition["path"].endswith("source_video_preprocess_smoke.json")
        and condition["key"] == "exact_readiness_refusal.blockers"
        and condition["contains"]
        == PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER
        for condition in execution["extra_artifact_postconditions"]
    )
    assert representation_plan["schema"] == "representation_training_probe_plan_v1"
    assert representation_plan["candidate_params"]["stage_module"] == (
        "stage8_muon_finetune"
    )
    assert representation_plan["candidate_params"]["stage_count"] == 1
    assert representation_plan["candidate_params"]["optimizer_descriptor_id"] == (
        "pr95_stage8_muon_adamw_mlx"
    )
    assert representation_plan["candidate_params"][
        "source_video_preprocess_smoke_requested"
    ] is True
    assert representation_plan["candidate_params"][
        "mlx_gpu_forward_drift_attestation_requested"
    ] is True
    assert PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER not in representation_plan[
        "dispatch_blockers"
    ]
    assert "synthetic_targets_do_not_establish_contest_quality" not in (
        representation_plan["dispatch_blockers"]
    )
    assert PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER in (
        representation_plan["dispatch_blockers"]
    )
    assert representation_plan["candidate_params"]["source_video_pair_indices"] == [0]
    assert representation_plan["candidate_params"]["source_video_output_hw"] == "384,512"
    assert representation_plan["recommended_execution"] == execution

    queue = build_local_training_execution_queue(
        [plan],
        queue_id="pr95_mlx_stage8_plan_fixture",
        repo_root=REPO_ROOT,
        local_mlx_concurrency=4,
        timeout_seconds=0,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 4
    assert queue["experiments"][0]["metadata"]["optimizer_descriptor_id"] == (
        "pr95_stage8_muon_adamw_mlx"
    )
    step = queue["experiments"][0]["steps"][0]
    assert step["resources"]["kind"] == "local_mlx"
    assert "tools/run_pr95_mlx_timing_smoke.py" in step["command"]
    assert any(
        condition["type"] == "json_equals"
        and condition["path"].endswith("representation_training_manifest.json")
        and condition["equals"] == "representation_training_probe_manifest_v1"
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"].endswith("runtime_consumption_proof.json")
        and condition["key"] == "runtime_consumption_proven"
        and condition["equals"] is True
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"].endswith("source_video_preprocess_smoke.json")
        and condition["key"] == "source_video_preprocess_ready"
        and condition["equals"] is True
        for condition in step["postconditions"]
    )

    row = adapt_representation_training_manifest_to_candidate(
        representation_plan,
        source_path=output_dir / "representation_training_plan.json",
        repo_root=REPO_ROOT,
    )
    assert row["representation_family"] == "hnerv"
    assert row["candidate_params"]["optimizer_descriptor_id"] == (
        "pr95_stage8_muon_adamw_mlx"
    )
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert "requires_lane_claim_before_dispatch" in row["dispatch_blockers"]
    assert PR95_SOURCE_VIDEO_LOADER_UNPORTED_BLOCKER not in row["dispatch_blockers"]
    assert row["candidate_params"]["source_video_preprocess_smoke_requested"] is True
    assert row["candidate_params"]["train_on_source_video_pairs"] is True
    assert validate_proxy_candidate(row) == []
