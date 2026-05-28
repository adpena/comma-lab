# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from tac.local_acceleration.pr95_hnerv_mlx import (
    FALSE_AUTHORITY,
    PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE,
    PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY,
)
from tac.local_acceleration.pr95_hnerv_mlx_contract import (
    PR95_EXPORT_FORWARD_PARITY_BLOCKER,
    PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER,
    PR95_FULL_FRAME_INFLATE_PARITY_FAILED_BLOCKER,
    PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER,
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER,
    PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER,
)
from tac.optimization.representation_training_probe_integration import (
    adapt_representation_training_manifest_to_candidate,
    validate_representation_training_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_run_pr95_mlx_timing_smoke_cli_writes_queueable_manifests(tmp_path: Path) -> None:
    output_dir = tmp_path / "pr95_mlx_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_timing_smoke.py"),
            "--stage",
            "1",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--synthetic-pairs",
            "1",
            "--seed",
            "11",
            "--base-channels",
            "4",
            "--output-dir",
            str(output_dir),
            "--write-byte-closed-smoke",
            "--write-pr95-public-archive-export",
            "--write-pytorch-export-parity",
            "--prove-pr95-runtime-consumption",
            "--runtime-proof-max-output-bytes",
            "7000000",
            "--runtime-proof-timeout-seconds",
            "180",
            "--write-pr95-full-frame-inflate-parity",
            "--full-frame-parity-max-output-bytes",
            "7000000",
            "--full-frame-parity-max-mismatch-samples",
            "5",
            "--full-frame-parity-timeout-seconds",
            "180",
            "--full-frame-parity-mlx-device",
            "cpu",
            "--full-frame-parity-conv2d-accumulation-mode",
            "fixed_fp32",
            "--full-frame-parity-conv2d-override-preset",
            "none",
            "--pytorch-export-conv2d-accumulation-mode",
            "kahan_fp32",
            "--write-source-faithful-preprocess-smoke",
            "--source-preprocess-shape",
            "1,2,8,10,3",
            "--source-preprocess-camera-hw",
            "11,13",
            "--source-preprocess-gradient-shape",
            "1,2,8,10,3",
            "--train-on-source-video-pairs",
            "--source-video-path",
            "upstream/videos/0.mkv",
            "--source-video-upstream-dir",
            "upstream",
            "--source-video-pair-index",
            "0",
            "--source-video-output-hw",
            "384,512",
            "--source-video-loss-surface",
            "rgb_yuv6_mse",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_profile = json.loads(
        (output_dir / "runtime_profile.json").read_text(encoding="utf-8")
    )
    representation = json.loads(
        (output_dir / "representation_training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    export_summary = json.loads(
        (output_dir / "pr95_public_archive_export.json").read_text(encoding="utf-8")
    )
    runtime_proof = json.loads(
        (output_dir / "runtime_consumption_proof.json").read_text(encoding="utf-8")
    )
    full_frame_parity = json.loads(
        (output_dir / "full_frame_inflate_parity_proof.json").read_text(
            encoding="utf-8"
        )
    )
    pytorch_export_parity = json.loads(
        (output_dir / "pytorch_export_forward_parity.json").read_text(
            encoding="utf-8"
        )
    )
    preprocess_smoke = json.loads(
        (output_dir / "source_faithful_preprocess_smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["ok"] is True
    assert summary["byte_closed_smoke_archive"]["member"] == "0.bin"
    assert summary["pr95_public_archive_export"]["schema"] == (
        "pr95_hnerv_archive_export.v1"
    )
    assert export_summary["runtime_consumption_proof_present"] is True
    assert export_summary["runtime_consumption_proven"] is True
    assert runtime_proof["work_dir_preserved"] is False
    assert runtime_proof["raw_output_path"] is None
    assert not (output_dir / "runtime_consumption_work").exists()
    assert (output_dir / "pr95_pytorch_state_dict.pt").is_file()
    assert pytorch_export_parity["schema"] == (
        "pr95_hnerv_mlx_pytorch_export_forward_parity.v1"
    )
    assert (
        pytorch_export_parity["pytorch_export_forward_parity_established"] is True
    )
    assert pytorch_export_parity["state_dict_pt_export"]["schema_version"] == (
        "mlx_to_pytorch_export.v1"
    )
    assert pytorch_export_parity["forward_parity"]["parity"]["passed"] is True
    assert pytorch_export_parity["conv2d_accumulation_mode"] == "kahan_fp32"
    for key in FALSE_AUTHORITY:
        assert pytorch_export_parity[key] is False
    assert (
        "pr95_archive_export_is_byte_closed_but_not_runtime_consumed"
        not in export_summary["exact_readiness_refusal"]["blockers"]
    )
    assert "requires_full_frame_inflate_parity_before_runtime_consumption_claim" not in (
        export_summary["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_FULL_FRAME_INFLATE_PARITY_FAILED_BLOCKER in (
        export_summary["exact_readiness_refusal"]["blockers"]
    )
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in (
        export_summary["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_EXPORT_FORWARD_PARITY_BLOCKER not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert "requires_pytorch_export_forward_parity_on_source_checkpoint" not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert "pytorch_export_forward_parity_is_local_probe_not_score_authority" in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_EXPORT_FORWARD_PARITY_BLOCKER not in (
        manifest["source_faithfulness_blockers"]
    )
    assert PR95_EXPORT_FORWARD_PARITY_BLOCKER not in (
        manifest["runtime_profile"]["source_faithfulness_blockers"]
    )
    assert PR95_EXPORT_FORWARD_PARITY_BLOCKER not in (
        manifest["optimizer_recipe"]["source_faithfulness_blockers"]
    )
    assert "blocker" not in manifest["pytorch_export_parity"]
    assert manifest["pytorch_export_parity"]["authority_note"] == (
        "local_mlx_pytorch_export_parity_probe_is_not_contest_auth_eval"
    )
    assert summary["runtime_consumption_proof"]["runtime_consumption_proven"] is True
    assert summary["full_frame_inflate_parity_proof"][
        "runtime_consumption_proof_passed"
    ] is True
    assert summary["full_frame_inflate_parity_proof"][
        "full_frame_inflate_parity_satisfied"
    ] is False
    full_frame_diff = summary["full_frame_inflate_parity_proof"]["diff"]
    torch_reference = summary["full_frame_inflate_parity_proof"][
        "torch_direct_reference"
    ]
    assert torch_reference["enabled"] is True
    assert torch_reference["byte_exact_with_public_inflate"] is True
    assert torch_reference["diff"]["byte_exact"] is True
    assert summary["full_frame_inflate_parity_proof"][
        "drift_localization_verdict"
    ] == "mlx_decoder_or_mlx_bridge_arithmetic_drift"
    assert full_frame_diff["max_abs_uint8"] <= 1
    assert full_frame_diff["changed_byte_count"] > 0
    assert full_frame_diff["raw_layout"] == "frames_nhwc_uint8"
    assert full_frame_diff["frame_shape_nhwc"][1:] == [874, 1164, 3]
    assert sum(full_frame_diff["per_frame_changed_byte_count"]) == (
        full_frame_diff["changed_byte_count"]
    )
    assert sum(full_frame_diff["per_channel_changed_byte_count"].values()) == (
        full_frame_diff["changed_byte_count"]
    )
    assert sum(full_frame_diff["boundary_distance_changed_byte_count"].values()) == (
        full_frame_diff["changed_byte_count"]
    )
    assert full_frame_diff["first_mismatch"]["abs_delta_uint8"] <= 1
    assert full_frame_diff["first_mismatch"]["channel_name"] in {"r", "g", "b"}
    assert full_frame_diff["mismatch_sample_count"] <= 5
    assert full_frame_diff["mismatch_sample_cap"] == 5
    assert manifest["runtime_consumption_proof_present"] is True
    assert manifest["runtime_consumption_proven"] is True
    assert manifest["full_frame_inflate_parity_proof_present"] is True
    assert manifest["full_frame_inflate_parity_satisfied"] is False
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["receiver_proof_present"] is True
    assert summary["pytorch_export_forward_parity"][
        "pytorch_export_forward_parity_established"
    ] is True
    assert summary["source_faithful_preprocess_smoke"]["source_faithful_preprocess_ready"] is True
    assert preprocess_smoke["gradient_probe"]["gradient_reachable"] is True
    assert manifest["stage_module"] == "stage1_v328_ce"
    assert manifest["training_fidelity"] == (
        PR95_MLX_TRAINING_FIDELITY_SOURCE_VIDEO_RGB_YUV6_TIMING_ONLY
    )
    assert manifest["training_loss_surface"] == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
    assert manifest["loss_surface_weights"] == {"rgb_mse": 0.5, "yuv6_mse": 0.5}
    assert manifest["target_yuv6_shape"] == [1, 2, 192, 256, 6]
    assert manifest["source_video_training"] is True
    assert manifest["source_video_target_loss_training"] is True
    assert manifest["source_faithful_training"] is False
    assert manifest["source_faithful_training_scope"] == "source_video_target_loss_only"
    assert manifest["full_pr95_source_faithful_training"] is False
    assert manifest["target_source"]["kind"] == "pr95_source_video_rgb_pairs"
    assert manifest["target_source"]["target_shape_n2chw"] == [1, 2, 3, 384, 512]
    assert manifest["synthetic_pairs"] is None
    assert manifest["training_pair_count"] == 1
    assert manifest["source_preprocess_shape"] == "1,2,8,10,3"
    assert manifest["source_preprocess_camera_hw"] == "11,13"
    assert manifest["source_preprocess_gradient_shape"] == "1,2,8,10,3"
    assert manifest["source_faithful_preprocess_smoke_path"].endswith(
        "source_faithful_preprocess_smoke.json"
    )
    assert "runtime_consumption_proof_missing" not in manifest[
        "exact_readiness_refusal"
    ]["blockers"]
    assert "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx" not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert (
        PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER
        not in manifest["exact_readiness_refusal"]["blockers"]
    )
    assert "synthetic_targets_do_not_establish_contest_quality" not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert manifest["pr95_public_archive_export"]["sha256"] == export_summary["sha256"]
    assert manifest["runtime_consumption_proof"]["raw_output_bytes"] == (
        runtime_proof["expected_raw_bytes"]
    )
    assert manifest["full_frame_inflate_parity_proof"]["public_raw_bytes"] == (
        full_frame_parity["expected_raw_bytes"]
    )
    assert manifest["full_frame_inflate_parity_proof"]["diff"]["same_shape"] is True
    assert manifest["full_frame_inflate_parity_proof"]["diff"]["first_mismatch"] == (
        full_frame_diff["first_mismatch"]
    )
    assert PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert PR95_FULL_FRAME_INFLATE_PARITY_FAILED_BLOCKER in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert manifest["pytorch_export_forward_parity"][
        "pytorch_export_forward_parity_established"
    ] is True
    assert manifest["pytorch_export_conv2d_accumulation_mode"] == "kahan_fp32"
    assert manifest["pytorch_export_state_dict_pt_path"].endswith(
        "pr95_pytorch_state_dict.pt"
    )
    assert manifest["optimizer_recipe"]["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert len(manifest["optimizer_recipe"]["parameter_group_fingerprint_sha256"]) == 64
    assert runtime_profile["training_backend"] == "mlx"
    assert runtime_profile["source_video_training"] is True
    assert runtime_profile["source_video_target_loss_training"] is True
    assert runtime_profile["source_faithful_training"] is False
    assert runtime_profile["target_source_kind"] == "pr95_source_video_rgb_pairs"
    assert runtime_profile["training_loss_surface"] == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
    assert runtime_profile["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert representation["training_recipe"]["quality_comparable"] is False
    assert representation["training_recipe"]["source_video_training"] is True
    assert representation["training_recipe"]["training_loss_surface"] == (
        PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
    )
    assert representation["candidate_params"]["source_video_training"] is True
    assert representation["candidate_params"]["training_loss_surface"] == (
        PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
    )
    assert representation["candidate_params"]["target_source_kind"] == (
        "pr95_source_video_rgb_pairs"
    )
    assert representation["candidate_params"]["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert representation["runtime_profile"]["profile_id"] == runtime_profile["profile_id"]
    assert representation["byte_closed_smoke_archive"]["sha256"] == summary[
        "byte_closed_smoke_archive"
    ]["sha256"]
    assert representation["archive_zip"]["sha256"] == export_summary["sha256"]
    assert representation["pr95_public_archive_export"]["sha256"] == export_summary["sha256"]
    assert representation["runtime_consumption_proof"]["runtime_consumption_proven"] is True
    assert representation["full_frame_inflate_parity_proof"][
        "runtime_consumption_proof_passed"
    ] is True
    assert representation["candidate_params"][
        "pr95_full_frame_inflate_parity_present"
    ] is True
    assert representation["candidate_params"][
        "pr95_full_frame_inflate_parity_satisfied"
    ] is False
    assert representation["pytorch_export_forward_parity"][
        "pytorch_export_forward_parity_established"
    ] is True
    assert PR95_EXPORT_FORWARD_PARITY_BLOCKER not in (
        representation["runtime_profile"]["source_faithfulness_blockers"]
    )
    assert "blocker" not in representation["pytorch_export_parity"]
    assert representation["candidate_params"]["pytorch_export_forward_parity_present"] is True
    assert representation["candidate_params"]["pytorch_export_forward_parity_requested"] is True
    assert representation["candidate_params"][
        "pytorch_export_conv2d_accumulation_mode"
    ] == "kahan_fp32"
    assert representation["source_faithful_preprocess_smoke"][
        "source_faithful_preprocess_ready"
    ] is True
    assert representation["candidate_params"]["source_faithful_preprocess_smoke_present"] is True
    assert representation["candidate_params"]["source_preprocess_shape"] == "1,2,8,10,3"
    validate_representation_training_manifest(representation)
    row = adapt_representation_training_manifest_to_candidate(
        representation,
        source_path=output_dir / "representation_training_manifest.json",
        repo_root=REPO_ROOT,
    )
    timing = row["consumer_payload"]["representation_training_probe"]["timing_smoke"]
    source_target = row["consumer_payload"]["representation_training_probe"][
        "source_video_training_target"
    ]
    assert timing["runtime_profile_summary"]["best_local_backend"] == "mlx"
    assert source_target["present"] is True
    assert source_target["source_video_target_loss_training"] is True
    assert source_target["training_loss_surface"] == PR95_MLX_LOSS_SURFACE_RGB_YUV6_MSE
    assert source_target["target_shape_n2chw"] == [1, 2, 3, 384, 512]
    assert "runtime_consumption_proof_missing" not in row["dispatch_blockers"]
    assert "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx" not in row[
        "dispatch_blockers"
    ]
    assert (
        PR95_YUV6_SCORER_LOSS_UNWIRED_BLOCKER not in row["dispatch_blockers"]
    )
    assert PR95_FULL_FRAME_INFLATE_PARITY_BLOCKER not in row[
        "dispatch_blockers"
    ]
    assert PR95_FULL_FRAME_INFLATE_PARITY_FAILED_BLOCKER in row[
        "dispatch_blockers"
    ]
    assert PR95_EXPORT_FORWARD_PARITY_BLOCKER not in row[
        "dispatch_blockers"
    ]
    assert "requires_pytorch_export_forward_parity_on_source_checkpoint" not in row[
        "dispatch_blockers"
    ]
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in row[
        "dispatch_blockers"
    ]
    assert PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER in row[
        "dispatch_blockers"
    ]
    assert "synthetic_targets_do_not_establish_contest_quality" not in row[
        "dispatch_blockers"
    ]
    assert PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER not in row[
        "dispatch_blockers"
    ]
    assert PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER in row[
        "dispatch_blockers"
    ]
    assert PR95_SOURCE_VIDEO_RGB_YUV6_NOT_FULL_SCORER_BLOCKER in row[
        "dispatch_blockers"
    ]
    preprocess_signal = row["consumer_payload"]["representation_training_probe"][
        "source_faithful_preprocess"
    ]
    assert preprocess_signal["present"] is True
    assert preprocess_signal["gradient_reachable"] is True
    assert preprocess_signal["exact_readiness_ready"] is False
    assert PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER in (
        preprocess_signal["exact_readiness_blockers"]
    )
    for payload in (summary, manifest, representation, preprocess_smoke):
        for key in FALSE_AUTHORITY:
            assert payload[key] is False
