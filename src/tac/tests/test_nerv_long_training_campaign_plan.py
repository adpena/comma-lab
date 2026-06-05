# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import load_queue_definition
from tac.analysis import nerv_long_training_campaign_plan as plan_module
from tac.analysis.nerv_candidate_feedback import (
    build_hinerv_archive_ladder_feedback_report,
)
from tac.analysis.nerv_long_training_campaign_plan import (
    DEFAULT_OPTIMIZER_KINDS,
    HINERV_POSE_INSTABILITY_LOW_LR_FLOOR,
    NervLongTrainingCampaignPlanError,
    build_nerv_long_training_campaign_plan,
    render_nerv_long_training_campaign_plan_markdown,
)
from tac.analysis.nerv_modelsize_budget import analyze_snerv_modelsize_candidate
from tac.substrates._shared.mlx_score_aware.adapter import (
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
)
from tools import build_nerv_long_training_campaign_plan as cli


def test_long_training_campaign_plan_builds_optimizer_matrix() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion", "adafactor"),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    assert report["schema"] == "nerv_long_training_campaign_plan.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["campaign_row_count"] == 3
    assert report["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert report["experiment_queue_id"] == "nerv_long_training_campaign_queue.v1"
    assert report["experiment_queue_experiment_count"] == 3
    assert report["launchable_local_row_count"] == 2
    assert report["snerv_lf_over_ceiling_reroute_queue"]["schema"] == (
        "snerv_lf_over_ceiling_reroute_queue.v1"
    )
    assert report["snerv_lf_over_ceiling_reroute_queue_row_count"] == 1
    reroute_row = report["snerv_lf_over_ceiling_reroute_queue"]["queue_rows"][0]
    assert reroute_row["work_order_type"] == "lf_reroute_blocker"
    assert "snerv_measured_lf_payload_report_missing" in reroute_row["blockers"]
    assert reroute_row["local_mlx_long_training_allowed"] is False
    assert report["family_counts"] == {"hi_nerv": 2, "snerv": 1}
    assert report["source_parity_contract"]["schema"] == ("nerv_source_parity_contract.v1")
    assert report["source_parity_required_for_long_training_ready"] is True
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["source_parity_nonblocking_gaps"]
    assert report["pr95_distortion_source_ready"] is True
    assert report["pr95_distortion_practices_consumed_by_rows"] is True
    assert report["pr95_distortion_practices_blockers"] == []
    assert all(
        row["pr95_distortion_practices_guard"]["launch_allowed"] is True
        for row in report["campaign_rows"]
    )
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"][
            "pr95_distortion_practices_consumed"
        ]
        is True
        for row in report["campaign_rows"]
    )

    hi_rows = [row for row in report["campaign_rows"] if row["family"] == "hi_nerv"]
    assert {row["optimizer_kind"] for row in hi_rows} == {"lion", "adafactor"}
    qat_flags = {
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        "--coder-qat-quant-residual-weight",
        "--coder-qat-magnitude-weight",
        "--coder-qat-delta-weight",
        "--coder-qat-c1a-entropy-weight",
        "--coder-qat-c1a-sigma",
        "--coder-qat-c1a-sample-size",
    }
    assert all(row["optimizer_control"]["backend"] == "mlx.optimizers" for row in hi_rows)
    assert all(row["optimizer_control"]["native_mlx_on_apple_silicon"] is True for row in hi_rows)
    assert all(row["optimizer_control"]["apple_specific_algorithm_claim"] is False for row in hi_rows)
    assert all("--optimizer-kind" in row["command_argv"] for row in hi_rows)
    assert all("--hi-nerv-optimizer-policy" in row["command_argv"] for row in hi_rows)
    assert all(row["optimizer_policy"]["requested_policy"] == "native_optimizer" for row in hi_rows)
    assert all(row["optimizer_policy"]["native_mlx_optimizer_expected"] is True for row in hi_rows)
    assert report["optimizer_control_policy"]["applies_to"] == [
        "hi_nerv_shared_mlx_scoreaware_runner_rows",
        "snerv_shared_mlx_scoreaware_long_training_rows",
    ]
    assert report["optimizer_control_policy"]["does_not_apply_to"] == []
    assert all(
        row["command_argv"][row["command_argv"].index("--hi-nerv-optimizer-policy") + 1] == "native_optimizer"
        for row in hi_rows
    )
    assert all("--segnet-distillation-objective" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--segnet-distillation-objective") + 1]
        == "boundary_argmax_hinge"
        for row in hi_rows
    )
    assert all("--segnet-direct-live-distillation-weight" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--segnet-direct-live-distillation-weight") + 1]
        == "0.25"
        for row in hi_rows
    )
    assert all("--segnet-direct-live-class-histogram-weight" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--segnet-direct-live-class-histogram-weight") + 1]
        == "0.25"
        for row in hi_rows
    )
    assert all("--segnet-direct-live-class-balanced-hinge-weight" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--segnet-direct-live-class-balanced-hinge-weight") + 1]
        == "0.5"
        for row in hi_rows
    )
    assert all("--segnet-direct-live-class-balanced-ce-weight" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--segnet-direct-live-class-balanced-ce-weight") + 1]
        == "0.25"
        for row in hi_rows
    )
    assert all("--coder-aware-qat" in row["command_argv"] for row in hi_rows)
    assert all(qat_flags.issubset(set(row["command_argv"])) for row in hi_rows)
    assert all(
        row["coder_qat_control"]["c1a_source"].startswith("PR95") and row["coder_qat_control"]["score_claim"] is False
        for row in hi_rows
    )
    assert all(row["command_argv"][row["command_argv"].index("--distillation-device") + 1] == "gpu" for row in hi_rows)
    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv["optimizer_kind"] == "pact_muon_adamw"
    assert qat_flags.issubset(set(snerv["command_argv"]))
    assert snerv["coder_qat_control"]["quant_bits"] == snerv["quant_bits"]
    assert snerv["coder_qat_control"]["c1a_entropy_weight"] == pytest.approx(1.0e-4)
    assert snerv["coder_qat_control"]["ready_for_exact_eval_dispatch"] is False
    assert snerv["optimizer_control"]["optimizer_kind"] == "pact_muon_adamw"
    assert snerv["optimizer_control"]["pact_partitioned_muon_adamw"] is True
    assert snerv["optimizer_control"]["borrowed_from_pr95"] is True
    assert snerv["optimizer_control"]["score_claim"] is False
    assert "snerv_optimizer_control_requires_learned_scoreaware_training_loop" not in snerv["blockers"]
    assert "--snerv-score-aware-long-training-epochs" in snerv["command_argv"]
    assert snerv["command_argv"][snerv["command_argv"].index("--snerv-score-aware-long-training-epochs") + 1] == "29650"
    assert snerv["command_argv"][snerv["command_argv"].index("--snerv-score-aware-long-training-optimizer") + 1] == "pact_muon_adamw"
    assert snerv["command_argv"][snerv["command_argv"].index("--segnet-distillation-weight") + 1] == "1.0"
    assert snerv["command_argv"][snerv["command_argv"].index("--pose-distillation-weight") + 1] == "1.0"
    assert "--snerv-score-aware-long-training-eval-roundtrip-ste" in snerv["command_argv"]
    assert "--snerv-score-aware-long-training-pr95-faithful-curriculum" in snerv["command_argv"]
    assert snerv["score_aware_long_training_plan"]["coder_aware_qat_bound"] is True
    assert snerv["score_aware_long_training_plan"]["pr95_faithful_curriculum_bound"] is True
    assert all(
        row["command_argv"][row["command_argv"].index("--mlx-prefilter-scorer-device") + 1] == "gpu" for row in hi_rows
    )
    assert all(
        row["command_argv"][row["command_argv"].index("--mlx-prefilter-scorer-batch-pairs") + 1] == "8"
        for row in hi_rows
    )
    assert all("--mlx-prefilter-progress-every" in row["command_argv"] for row in hi_rows)
    assert all("--telemetry-flush-interval-epochs" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--telemetry-flush-interval-epochs") + 1] == "1"
        for row in hi_rows
    )
    assert all("--scorer-input-distribution-guard-weight" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--scorer-input-distribution-guard-weight") + 1] == "2"
        for row in hi_rows
    )
    assert all("--scorer-input-contrast-floor-weight" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--scorer-input-contrast-floor-weight") + 1] == "0.5"
        for row in hi_rows
    )
    assert all("--scorer-input-contrast-floor-segnet-min-std-ratio" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--scorer-input-contrast-floor-segnet-min-std-ratio") + 1]
        == "0.6"
        for row in hi_rows
    )
    assert all("--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio" in row["command_argv"] for row in hi_rows)
    assert all(
        row["command_argv"][row["command_argv"].index("--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio") + 1]
        == "0.6"
        for row in hi_rows
    )
    assert all(row["local_mlx_launch_command_ready"] is True for row in hi_rows)
    assert all(row["local_mlx_executable"] is True for row in hi_rows)
    assert all("--auto-joint-recon-pixel-weight" not in row["command_argv"] for row in hi_rows)
    assert all("--recon-pixel-weight-path" not in row["command_argv"] for row in hi_rows)
    assert all("requires_verified_joint_p18_p19_recon_pixel_weight_artifact" in row["blockers"] for row in hi_rows)
    assert all(row["cpu_replay_ready"] is False for row in hi_rows)
    assert all(row["exact_gate_ready"] is False for row in hi_rows)
    assert all(row["score_lowering_gate"]["schema"] == "nerv_long_training_score_lowering_gate.v1" for row in hi_rows)
    assert all(
        {
            "archive_in_loop_byte_oracle",
            "byte_closed_archive_export",
            "receiver_proof",
            "full_video_local_prefilter",
            "local_cpu_replay_gate",
        }.issubset(set(row["score_lowering_gate"]["post_run_missing_requirement_ids"]))
        for row in hi_rows
    )
    assert all("hi_nerv_receiver_proof_missing" in row["blockers"] for row in hi_rows)
    assert all(row["source_parity"]["required_blockers"] == [] for row in hi_rows)
    assert all(row["source_parity"]["score_claim"] is False for row in hi_rows)
    assert all("hi_nerv_byte_closed_archive_export_missing" in row["promotion_blockers"] for row in hi_rows)
    assert all(row["experiment_queue_entry"]["status"] == "queued" for row in hi_rows)
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"][
            "queue_launch_blockers"
        ]
        == []
        for row in hi_rows
    )
    assert all(
        row["experiment_queue_entry"]["launch_authority_contract"][
            "pr95_distortion_practices_guard"
        ]
        == row["pr95_distortion_practices_guard"]
        for row in hi_rows
    )
    assert all(row["experiment_queue_entry"]["cpu_replay_ready"] is False for row in hi_rows)
    assert all(row["experiment_queue_entry"]["exact_gate_ready"] is False for row in hi_rows)
    hi_step = hi_rows[0]["experiment_queue_entry"]["steps"][0]
    assert hi_step["command"] == hi_rows[0]["command_argv"]
    assert "telemetry" in hi_step
    assert hi_step["telemetry"]["include_postcondition_paths"] is True
    assert any(
        path.endswith("compact_renderer_mlx_spine_runner_startup.json")
        for path in hi_step["telemetry"]["artifact_paths"]
    )
    assert any(path.endswith("hi_nerv_mlx_training/telemetry.jsonl") for path in hi_step["telemetry"]["artifact_paths"])
    assert any(
        path.endswith("hi_nerv_mlx_training/local_mlx_prefilter_progress.jsonl")
        for path in hi_step["telemetry"]["artifact_paths"]
    )
    assert "--planner-row-id" in hi_rows[0]["command_argv"]
    assert hi_rows[0]["command_argv"][hi_rows[0]["command_argv"].index("--planner-row-id") + 1] == hi_rows[0]["row_id"]
    assert hi_step["resources"]["kind"] == "local_mlx"
    assert {
        (condition["key"], condition.get("equals"))
        for condition in hi_step["postconditions"]
        if condition["type"] == "json_equals"
    } >= {
        ("schema", "compact_renderer_mlx_spine_runner.v1"),
        ("execute_family", "hi_nerv"),
        ("training_executed", True),
        ("score_claim", False),
        ("promotion_eligible", False),
        ("ready_for_exact_eval_dispatch", False),
    }

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["implementation_status"] == "snerv_scorer_tether_smoke_gate_blocked"
    assert snerv_row["hard_byte_ceiling_satisfied_for_long_training"] is False
    assert snerv_row["score_lowering_gate"]["command_materialized"] is False
    assert snerv_row["score_lowering_gate"]["local_mlx_executable"] is False
    assert snerv_row["score_lowering_gate"]["prelaunch_allowed"] is False
    assert snerv_row["score_lowering_gate"]["promotion_prelaunch_allowed"] is False
    assert (
        "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training"
        in snerv_row["blockers"]
    )
    assert "snerv_scorer_tether_smoke_report_missing" in snerv_row["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_missing" in snerv_row["blockers"]
    assert (
        "snerv_native_rate_pressure_in_loop_not_yet_training_authority"
        not in snerv_row["score_lowering_gate"]["prelaunch_blockers"]
    )
    assert "snerv_pr95_staged_curriculum_missing" not in snerv_row["score_lowering_gate"]["prelaunch_blockers"]
    assert "snerv_optimizer_control_requires_learned_scoreaware_training_loop" not in snerv_row["score_lowering_gate"]["launch_blockers"]
    assert snerv_row["cpu_replay_ready"] is False
    assert snerv_row["exact_gate_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"
    assert snerv_row["experiment_queue_entry"]["blocked"] is True
    launch_contract = snerv_row["experiment_queue_entry"]["launch_authority_contract"]
    assert launch_contract["schema"] == ("nerv_long_training_queue_launch_authority_contract.v1")
    assert launch_contract["queue_status_is_local_mlx_plan"] is False
    assert launch_contract["queue_status_is_runnable_plan"] is False
    assert (
        "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training"
        in launch_contract["queue_launch_blockers"]
    )
    assert (
        "snerv_optimizer_control_requires_learned_scoreaware_training_loop" not in launch_contract["queue_launch_blockers"]
    )
    assert launch_contract["queue_status_is_receiver_proof"] is False
    assert launch_contract["queue_status_is_cpu_replay_proof"] is False
    assert launch_contract["queue_status_is_exact_eval_authority"] is False
    assert launch_contract["cpu_replay_ready"] is False
    assert launch_contract["exact_gate_ready"] is False
    assert launch_contract["pr95_distortion_practices_guard"] == (
        snerv_row["pr95_distortion_practices_guard"]
    )
    assert "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only" not in snerv_row["blockers"]
    assert snerv_row["execution_epochs"] == 29_650
    assert snerv_row["current_command_is_bounded_proof_not_long_training"] is False
    assert "--snerv-scorer-loop-qat" in snerv_row["command_argv"]
    assert "--snerv-spectra-preserving-adapter" not in snerv_row["command_argv"]
    assert snerv_row["candidate"]["wavelet"] == "haar"
    assert snerv_row["source_bound_capacity_controls"]["fc_dim"] == 11
    assert snerv_row["source_bound_capacity_controls"]["emb_size"] == 2
    assert snerv_row["source_bound_capacity_controls"]["candidate_id_matches_source_controls"] is True
    assert snerv_row["source_bound_capacity_controls"]["expected_candidate_id"] == snerv_row["candidate_id"]
    assert not snerv_row["source_bound_capacity_control_blockers"]
    assert snerv_row["source_parity"]["required_blockers"] == []
    assert "source_parity:snerv_official_mfu_hfr_tub_parity_missing" in snerv_row["source_parity"]["nonblocking_gaps"]
    assert snerv_row["source_parity"]["score_claim"] is False
    assert "--snerv-model-size-adapter" in snerv_row["command_argv"]
    argv = snerv_row["command_argv"]
    assert argv[argv.index("--snerv-native-mlx-receiver-proof-timeout") + 1] == "321"
    assert argv[argv.index("--snerv-native-mlx-decoder-train-steps") + 1] == "5"
    assert argv[argv.index("--snerv-native-mlx-decoder-train-lr") + 1] == "0.00025"
    assert argv[argv.index("--snerv-native-mlx-decoder-train-ridge") + 1] == "2e-06"
    native_training = snerv_row["native_mlx_decoder_training_plan"]
    assert native_training["schema"] == "snerv_native_mlx_decoder_training_plan.v1"
    assert native_training["planned_steps"] == 5
    assert native_training["backend"] == "mlx_metal_full_batch_gradient_descent"
    assert native_training["score_claim"] is False
    assert (
        snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-model-size-adapter") + 1]
        == "snerv_fc_dim_emb_size_adapter_v1"
    )
    assert "--snerv-fc-dim" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-fc-dim") + 1] == "11"
    assert "--snerv-emb-size" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-emb-size") + 1] == "2"
    assert "--snerv-patch-radius" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-patch-radius") + 1] == "1"
    assert "--snerv-mfu-scales" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-mfu-scales") + 1] == "1,2,4"
    assert "--snerv-hfr-gain" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-hfr-gain") + 1] == "0"
    assert "--snerv-temporal-context" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-temporal-context") + 1] == "0"
    assert "--snerv-temporal-mode" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-temporal-mode") + 1] == "delta"
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--distillation-device") + 1] == "gpu"
    assert "--planner-row-id" in snerv_row["command_argv"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--planner-row-id") + 1] == snerv_row["row_id"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--epochs") + 1] == "29650"
    snerv_step = snerv_row["experiment_queue_entry"]["steps"][0]
    assert {condition["type"] for condition in snerv_step["postconditions"]} >= {"json_equals"}
    snerv_blocker_postconditions = [
        condition for condition in snerv_step["postconditions"] if condition["type"] == "json_array_contains"
    ]
    assert not snerv_blocker_postconditions

    markdown = render_nerv_long_training_campaign_plan_markdown(report)
    assert "NeRV Long-Training Campaign Plan" in markdown
    assert "hi_nerv::hinerv_tiny::lion" in markdown


def test_long_training_campaign_plan_binds_upstream_evaluate_contract() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    contract = report["upstream_evaluate_priority_contract"]
    assert contract["schema"] == "nerv_upstream_evaluate_priority_contract.v1"
    assert contract["source"] == "upstream/evaluate.py"
    assert contract["baseline_to_beat"] == (
        "full_pr95_fidelity_or_better_on_exact_upstream_evaluate_axes"
    )
    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["crux"]["segnet_frame0_direct_weight"] == 0.0
    assert contract["crux"]["segnet_frame1_direct_weight"] == 1.0
    assert contract["crux"]["pose_marginal_formula"] == "5/sqrt(10*d_pose)"
    assert contract["crux"]["canonical_rate_denominator_bytes"] == 37_545_489
    assert contract["crux"]["rate_price_per_archive_byte"] == pytest.approx(
        25 / 37_545_489
    )
    score_allocation = contract["score_allocation_contract"]
    assert score_allocation["rate"]["archive_authority"] == (
        "submission_dir/archive.zip.stat().st_size"
    )
    assert score_allocation["rate"]["raw_output_shape_bytes_are_not_rate_denominator"] == (
        1200 * 874 * 1164 * 3
    )
    assert report["upstream_evaluate_contract_consumed_by_rows"] is True

    for row in report["campaign_rows"]:
        binding = row["upstream_evaluate_score_binding"]
        assert binding["schema"] == "nerv_row_upstream_evaluate_binding.v1"
        assert binding["contract_schema"] == contract["schema"]
        assert binding["baseline_to_beat"] == contract["baseline_to_beat"]
        assert binding["rate"]["canonical_denominator_bytes"] == 37_545_489
        assert binding["rate"]["rate_price_per_archive_byte"] == pytest.approx(
            25 / 37_545_489
        )
        assert binding["rate"]["raw_output_shape_bytes_are_not_rate_denominator"] == (
            1200 * 874 * 1164 * 3
        )
        assert binding["pair_geometry"]["seq_len"] == 2
        assert binding["pair_geometry"]["public_test_pair_count"] == 600
        assert binding["pair_geometry"]["camera_size_wh"] == [1164, 874]
        assert binding["pair_geometry"]["candidate_raw_shape"] == [
            1200,
            874,
            1164,
            3,
        ]
        assert binding["segnet"]["frame_scope"] == "last_frame_only"
        assert binding["segnet"]["scored_frame_index_within_pair"] == 1
        assert binding["segnet"]["unscored_frame_index_within_pair"] == 0
        assert binding["posenet"]["frame_scope"] == "both_frames_in_pair"
        assert binding["posenet"]["scored_frame_indices_within_pair"] == [0, 1]
        assert binding["posenet"]["derivative_wrt_d_pose"] == "5/sqrt(10*d_pose)"
        assert binding["score_claim"] is False
        assert binding["promotion_eligible"] is False
        queue_entry = row["experiment_queue_entry"]
        assert queue_entry["metadata"]["upstream_evaluate_score_binding"] == binding
        launch_contract = queue_entry["launch_authority_contract"]
        assert launch_contract["upstream_evaluate_score_contract_consumed"] is True
        assert launch_contract["upstream_evaluate_score_binding"] == binding

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv["score_aware_long_training_plan"]["upstream_evaluate_score_binding"] == (
        snerv["upstream_evaluate_score_binding"]
    )


def test_long_training_campaign_plan_binds_tilde_oss_leverage_policy() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("aurora_like", "lion"),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    policy = report["tilde_oss_leverage_policy"]
    assert policy["schema"] == "nerv_tilde_oss_leverage_policy.v1"
    assert policy["score_claim"] is False
    assert policy["ready_for_exact_eval_dispatch"] is False
    assert policy["aurora"]["allowed_use"] == "optimizer_timing_convergence_smoke_only"
    assert policy["aurora"]["planner_optimizer_kind"] == "aurora_like"
    assert policy["wall_attention"]["direct_kernel_import_allowed"] is False
    assert policy["wall_attention"]["byte_charged_receiver_replay_required"] is True
    assert policy["parallax"]["official_tilde_surface"] is False
    assert policy["parallax"]["direct_runtime_import_allowed"] is False
    assert policy["parallax"]["classification"] == (
        "llm_local_linear_attention_not_video_parallax_geometry"
    )
    assert policy["direct_import_policy"]["forbidden_repos"] == [
        "Yifei-Zuo/Parallax",
        "tilde-research/wall-attention-release",
    ]
    assert report["tilde_oss_policy_consumed_by_rows"] is True

    for row in report["campaign_rows"]:
        binding = row["tilde_oss_leverage_binding"]
        assert binding["schema"] == "nerv_row_tilde_oss_binding.v1"
        assert binding["policy_schema"] == policy["schema"]
        assert binding["aurora_like_optimizer_smoke_allowed"] is True
        assert binding["parallax_direct_runtime_import_allowed"] is False
        assert binding["wall_attention_direct_kernel_import_allowed"] is False
        assert binding["pact_native_receiver_byte_charged_required"] is True
        assert binding["score_claim"] is False
        queue_entry = row["experiment_queue_entry"]
        assert queue_entry["metadata"]["tilde_oss_leverage_binding"] == binding
        launch_contract = queue_entry["launch_authority_contract"]
        assert launch_contract["tilde_oss_leverage_policy_consumed"] is True
        assert launch_contract["tilde_oss_leverage_binding"] == binding

    aurora_rows = [
        row for row in report["campaign_rows"] if row["optimizer_kind"] == "aurora_like"
    ]
    assert len(aurora_rows) == 1
    assert aurora_rows[0]["experiment_queue_entry"]["status"] == "disabled"
    assert (
        "aurora_requires_local_timing_convergence_smoke"
        in aurora_rows[0]["experiment_queue_entry"]["launch_authority_contract"][
            "queue_launch_blockers"
        ]
    )


def test_long_training_campaign_plan_binds_pr95_baseline_identity() -> None:
    pr95_identity = _pr95_baseline_identity()
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        pr95_baseline_identity=pr95_identity,
    )

    binding = report["pr95_baseline_identity_binding"]
    assert binding["schema"] == "nerv_pr95_baseline_identity_binding.v1"
    assert binding["attached"] is True
    assert binding["baseline_id"] == "pr95_public_hnerv_muon_control_arm"
    assert binding["selected_archive"]["sha256"] == "a" * 64
    assert binding["local_cpu_mlx_work_order"]["ready"] is True
    assert binding["local_cpu_mlx_work_order"]["local_cpu_axis_tag"] == (
        "[macOS-CPU advisory]"
    )
    assert binding["local_cpu_mlx_work_order"]["mlx_axis_tag"] == (
        "[macOS-MLX research-signal]"
    )
    assert binding["modal_dispatch_policy"]["modal_dispatch_allowed"] is False
    assert binding["paired_exact_eval_work_order"]["ready"] is False
    assert "modal_reserved_for_frontier_candidates" in binding[
        "paired_exact_eval_work_order"
    ]["blockers"]
    assert "pr95_contest_cpu_exact_eval_missing" in binding["blockers"]
    assert report["pr95_baseline_identity_consumed_by_rows"] is True
    assert report["score_claim"] is False

    for row in report["campaign_rows"]:
        row_binding = row["pr95_baseline_identity_binding"]
        assert row_binding == binding
        queue_entry = row["experiment_queue_entry"]
        assert queue_entry["metadata"]["pr95_baseline_identity_binding"] == binding
        launch_contract = queue_entry["launch_authority_contract"]
        assert launch_contract["pr95_baseline_identity_consumed"] is True
        assert launch_contract["pr95_baseline_identity_binding"] == binding
        assert launch_contract["score_claim"] is False

    markdown = render_nerv_long_training_campaign_plan_markdown(report)
    assert "pr95_baseline_identity_attached: `True`" in markdown
    assert f"selected_archive_sha256: `{'a' * 64}`" in markdown
    assert "local_cpu_mlx_ready: `True`" in markdown
    assert "modal_dispatch_allowed: `False`" in markdown
    assert "paired_exact_eval_ready: `False`" in markdown
    assert "pr95_contest_cpu_exact_eval_missing" in markdown


def test_long_training_campaign_plan_pr95_distortion_guard_blocks_queue_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        plan_module._experiment_launch_blockers(
            ["requires_verified_joint_p18_p19_recon_pixel_weight_artifact"]
        )
        == []
    )

    def fake_source_inventory(_: object) -> dict:
        return {
            "schema": "pr95_distortion_source_inventory.v1",
            "source_ready": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def fake_row_guard(
        row: dict,
        *,
        repo_root: object,
        source_inventory: dict | None = None,
    ) -> dict:
        family = str(row["family"])
        blocker = (
            f"{family}_pr95_distortion_"
            "scorer_preprocess_eval_roundtrip_yuv6_missing"
        )
        return {
            "schema": "pr95_distortion_practices_guard.v1",
            "family": family,
            "row_id": row.get("id"),
            "required_for_family": True,
            "source_inventory_schema": (source_inventory or {}).get("schema"),
            "launch_allowed": False,
            "practice_rows": [],
            "blockers": [blocker],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        plan_module,
        "build_pr95_distortion_source_inventory",
        fake_source_inventory,
    )
    monkeypatch.setattr(
        plan_module,
        "build_pr95_distortion_practices_row_guard",
        fake_row_guard,
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        batch_pairs=8,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    assert report["pr95_distortion_practices_consumed_by_rows"] is True
    assert set(report["pr95_distortion_practices_blockers"]) == {
        "hi_nerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing",
        "snerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing",
    }
    for row in report["campaign_rows"]:
        guard = row["pr95_distortion_practices_guard"]
        blocker = guard["blockers"][0]
        launch_contract = row["experiment_queue_entry"]["launch_authority_contract"]
        assert guard["launch_allowed"] is False
        assert blocker in row["blockers"]
        assert blocker in launch_contract["queue_launch_blockers"]
        assert launch_contract["pr95_distortion_practices_consumed"] is True
        assert launch_contract["pr95_distortion_practices_guard"] == guard
        assert row["experiment_queue_entry"]["blocked"] is True


def test_long_training_campaign_plan_source_parity_required_blocker_disables_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_source_parity_contract(**_: object) -> dict:
        return {
            "schema": "nerv_source_parity_contract.v1",
            "authority": "false_authority_source_parity_no_score_claim",
            "required_for_long_training_ready": False,
            "blockers": ["snerv_official_mfu_hfr_tub_parity_missing"],
            "nonblocking_gaps": [],
            "family_rows": [
                {"family": "hi_nerv", "long_training_ready": True, "blockers": []},
                {
                    "family": "snerv",
                    "long_training_ready": False,
                    "blockers": ["snerv_official_mfu_hfr_tub_parity_missing"],
                },
            ],
            "feature_rows": [
                {
                    "family": "snerv",
                    "feature_id": "snerv_official_mfu_hfr_tub_parity",
                    "status": "missing_or_partial",
                    "required_for_long_training": True,
                    "blockers": ["snerv_official_mfu_hfr_tub_parity_missing"],
                }
            ],
            "control_rows": [],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        plan_module,
        "build_nerv_source_parity_contract",
        fake_source_parity_contract,
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert "source_parity:snerv_official_mfu_hfr_tub_parity_missing" in snerv_row["blockers"]
    assert snerv_row["source_parity"]["required_blockers"] == [
        "source_parity:snerv_official_mfu_hfr_tub_parity_missing"
    ]
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"
    assert snerv_row["implementation_status"] == ("source_bound_capacity_controls_incomplete")


def test_long_training_campaign_plan_embeds_snerv_official_source_audit() -> None:
    audit = {
        "schema": "snerv_official_source_parity_audit.v1",
        "authority": "false_authority_source_audit_no_score_claim",
        "family": "snerv",
        "official_repo": {
            "repo_url": "https://github.com/qwertja/SNeRV",
            "root": "/Volumes/VertigoDataTier/pact/oss_sources/SNeRV",
            "head_sha": "0844a08f",
        },
        "official_source_markers_present": True,
        "local_receiver_safe_adapter_present": True,
        "official_mfu_hfr_tub_parity_proven": False,
        "official_mfu_hfr_tub_primitive_replay_binding": {
            "all_receiver_primitive_replay_proven": True,
            "all_primitive_numeric_graph_replay_proven": True,
            "all_primitive_numeric_source_fixture_replay_proven": True,
            "all_primitive_source_replay_proven": False,
            "full_stack_source_forward_replay_proven": False,
            "receiver_source_forward_replay_bound": False,
            "official_receiver_runtime_decode_contract": {
                "receiver_runtime_decode_proven": True,
                "receiver_source_forward_replay_bound": False,
                "blockers": [
                    "snerv_official_mfu_hfr_tub_source_forward_replay_missing"
                ],
            },
        },
        "official_receiver_runtime_decode_contract": {
            "receiver_runtime_decode_proven": True,
            "receiver_source_forward_replay_bound": False,
            "blockers": ["snerv_official_mfu_hfr_tub_source_forward_replay_missing"],
        },
        "blockers": ["snerv_official_mfu_hfr_tub_parity_missing"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_official_source_audit=audit,
    )

    assert report["snerv_official_source_audit_attached"] is True
    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["source_parity"]["source_audit_rows"]
    source_audit = snerv_row["source_parity"]["source_audit_rows"][0]
    assert source_audit["official_head_sha"] == "0844a08f"
    assert source_audit["official_source_markers_present"] is True
    assert source_audit["official_mfu_hfr_tub_receiver_primitives_proven"] is True
    assert source_audit["official_mfu_hfr_tub_numeric_graph_replay_proven"] is True
    assert source_audit["official_receiver_runtime_decode_proven"] is True
    assert source_audit["official_mfu_hfr_tub_parity_proven"] is False
    assert source_audit["full_stack_source_forward_replay_proven"] is False
    split = snerv_row["snerv_official_runtime_authority_split"]
    assert split["schema"] == "snerv_official_runtime_authority_split.v1"
    assert split["receiver_bound_training_evidence_usable"] is True
    assert split["full_source_forward_authority_proven"] is False
    assert split["receiver_primitive_replay_proven"] is True
    assert split["numeric_graph_replay_proven"] is True
    assert split["receiver_runtime_decode_proven"] is True
    assert split["receiver_source_forward_replay_bound"] is False
    assert split["launch_semantics"] == (
        "receiver_bound_training_allowed_but_official_source_authority_false"
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in (
        split["blockers"]
    )
    official_feature = next(
        row
        for row in snerv_row["source_parity"]["feature_status_rows"]
        if row["feature_id"] == "snerv_official_mfu_hfr_tub_parity"
    )
    assert official_feature["source_audit_rows"][0]["official_head_sha"] == "0844a08f"
    assert "source_parity:snerv_official_mfu_hfr_tub_parity_missing" in snerv_row["source_parity"]["nonblocking_gaps"]
    assert snerv_row["source_parity"]["score_claim"] is False
    queue_entry = snerv_row["experiment_queue_entry"]
    assert queue_entry["metadata"]["source_parity"]["source_audit_rows"][0][
        "official_head_sha"
    ] == "0844a08f"
    assert queue_entry["metadata"]["source_bound_capacity_controls"]["schema"] == (
        "snerv_source_bound_capacity_controls.v1"
    )
    launch_contract = queue_entry["launch_authority_contract"]
    assert launch_contract["source_parity_contract_consumed"] is True
    assert launch_contract["source_bound_capacity_controls_consumed"] is True
    assert launch_contract["source_parity"]["source_audit_rows"][0][
        "official_source_markers_present"
    ] is True
    assert launch_contract["snerv_official_runtime_authority_split"][
        "receiver_bound_training_evidence_usable"
    ] is True
    assert launch_contract["snerv_official_runtime_authority_split"][
        "full_source_forward_authority_proven"
    ] is False
    assert launch_contract["source_bound_capacity_controls"]["schema"] == (
        "snerv_source_bound_capacity_controls.v1"
    )
    assert launch_contract["score_claim"] is False
    markdown = render_nerv_long_training_campaign_plan_markdown(report)
    assert (
        "snerv_runtime_authority: "
        "`receiver_bound_training_allowed_but_official_source_authority_false`"
    ) in markdown
    assert "snerv_full_source_forward_authority: `False`" in markdown


def test_long_training_campaign_plan_blocks_legacy_snerv_ids_for_long_runs() -> None:
    snerv_budget = _snerv_budget()
    legacy = dict(snerv_budget["selected_candidates"][0])
    legacy["candidate_id"] = "snerv_np600_lv2_lfb1p5_stepb0p5_int2_symmetric_ceil36000"
    for key in (
        "wavelet",
        "fc_dim",
        "emb_size",
        "patch_radius",
        "mfu_scales",
        "hfr_gain",
        "temporal_context",
        "temporal_mode",
    ):
        legacy.pop(key, None)
    snerv_budget["selected_candidates"] = [legacy]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert "snerv_source_bound_control_missing:wavelet" in snerv_row["blockers"]
    assert "snerv_source_bound_control_missing:fc_dim" in snerv_row["blockers"]
    assert "snerv_source_bound_control_missing:temporal_mode" in snerv_row["blockers"]
    assert snerv_row["source_bound_capacity_control_blockers"]
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"


def test_long_training_campaign_plan_scrubs_nested_candidate_authority_flags() -> None:
    snerv_budget = _snerv_budget()
    candidate = dict(snerv_budget["selected_candidates"][0])
    candidate["score_claim"] = True
    candidate["metadata"] = {
        "score_claim": True,
        "nested": {"ready_for_exact_eval_dispatch": True},
    }
    candidate["section_rows"] = [{"promotion_eligible": True}]
    snerv_budget["selected_candidates"] = [candidate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert "selected_candidate_authority_flag_true:score_claim" in snerv_row[
        "blockers"
    ]
    assert "selected_candidate_authority_flag_true:metadata.score_claim" in snerv_row[
        "blockers"
    ]
    assert (
        "selected_candidate_authority_flag_true:"
        "metadata.nested.ready_for_exact_eval_dispatch"
    ) in snerv_row["blockers"]
    assert (
        "selected_candidate_authority_flag_true:section_rows[0].promotion_eligible"
        in snerv_row["blockers"]
    )
    emitted = snerv_row["candidate"]
    assert emitted["score_claim"] is False
    assert emitted["metadata"]["score_claim"] is False
    assert emitted["metadata"]["nested"]["ready_for_exact_eval_dispatch"] is False
    assert emitted["section_rows"][0]["promotion_eligible"] is False
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"


def test_long_training_campaign_plan_executes_snerv_official_temporal_mode() -> None:
    snerv_budget = _snerv_budget()
    candidate = dict(snerv_budget["selected_candidates"][0])
    candidate["temporal_context"] = 2
    candidate["temporal_mode"] = "official_haar_dwt1d_lowpass"
    candidate["candidate_id"] = (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_"
        "p1_mfu1-2-4_hfr0_t2_tmhaar1_adbase_int4_symmetric_ceil178000"
    )
    snerv_budget["selected_candidates"] = [candidate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["source_bound_capacity_controls"]["candidate_id_matches_source_controls"] is True
    assert snerv_row["source_bound_capacity_controls"]["temporal_mode"] == "official_haar_dwt1d_lowpass"
    assert not snerv_row["source_bound_capacity_control_blockers"]
    assert snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-temporal-context") + 1] == "2"
    assert (
        snerv_row["command_argv"][snerv_row["command_argv"].index("--snerv-temporal-mode") + 1]
        == "official_haar_dwt1d_lowpass"
    )
    queue_command = snerv_row["experiment_queue_entry"]["steps"][0]["command"]
    assert queue_command == snerv_row["command_argv"]
    assert queue_command[queue_command.index("--snerv-temporal-mode") + 1] == "official_haar_dwt1d_lowpass"
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["implementation_status"] == "snerv_scorer_tether_smoke_gate_blocked"
    assert snerv_row["hard_byte_ceiling_satisfied_for_long_training"] is False
    assert "snerv_scorer_tether_smoke_report_missing" in snerv_row["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_missing" in snerv_row["blockers"]
    assert "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training" in snerv_row["blockers"]


def test_long_training_campaign_plan_executes_snerv_official_mfu_hfr_tub_adapter() -> None:
    snerv_budget = _snerv_budget()
    candidate = dict(snerv_budget["selected_candidates"][0])
    candidate.update(
        {
            "candidate_id": (
                "snerv_np600_haar_lv1_lfb1_stepb1_fc11e0_"
                "p1_mfu1-2-4_hfr0_t0_adofficial_oms0p05_"
                "int2_symmetric_ceil178000"
            ),
            "levels": 1,
            "bits_per_coeff": 1.0,
            "step_map_bits_per_coeff": 1.0,
            "decoder_payload_codec": "int2_symmetric",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "emb_size": 0,
            "capacity_source": "official_snerv_modelsize",
            "modelsize_mparams": 0.05,
        }
    )
    snerv_budget["selected_candidates"] = [candidate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("pact_muon_adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    argv = snerv_row["command_argv"]
    assert "--snerv-spectra-preserving-adapter" not in argv
    assert argv[argv.index("--snerv-model-size-adapter") + 1] == (
        SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    )
    assert argv[argv.index("--modelsize-candidate-id") + 1] == candidate["candidate_id"]
    assert snerv_row["source_bound_capacity_controls"]["levels"] == 1
    assert snerv_row["source_bound_capacity_controls"]["emb_size"] == 0
    assert snerv_row["source_bound_capacity_controls"][
        "candidate_id_matches_source_controls"
    ] is True


def test_long_training_campaign_plan_blocks_snerv_id_control_mismatch() -> None:
    snerv_budget = _snerv_budget()
    mismatched = dict(snerv_budget["selected_candidates"][0])
    mismatched["candidate_id"] = str(mismatched["candidate_id"]).replace(
        "_mfu1-2-4_",
        "_mfu1-3_",
    )
    snerv_budget["selected_candidates"] = [mismatched]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert "snerv_candidate_id_source_bound_controls_mismatch" in snerv_row["blockers"]
    assert snerv_row["source_bound_capacity_controls"]["candidate_id_matches_source_controls"] is False
    assert snerv_row["source_bound_capacity_controls"]["expected_candidate_id"] != snerv_row["candidate_id"]
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"


def test_long_training_campaign_plan_binds_snerv_official_skip_high_mode_id() -> None:
    snerv_budget = _snerv_budget()
    candidate = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=1,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        official_modelsize_mparams=0.05,
        temporal_mode="official_haar_dwt1d_lowpass",
        official_skip_high_mode="channel_mean",
    ).as_dict()
    snerv_budget["selected_candidates"] = [candidate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert "snerv_candidate_id_source_bound_controls_mismatch" not in snerv_row[
        "blockers"
    ]
    assert snerv_row["source_bound_capacity_controls"][
        "candidate_id_matches_source_controls"
    ] is True
    assert snerv_row["source_bound_capacity_controls"]["official_skip_high_mode"] == (
        "channel_mean"
    )
    argv = snerv_row["command_argv"]
    assert argv[argv.index("--snerv-official-skip-high-mode") + 1] == "channel_mean"


def test_long_training_campaign_plan_scrubs_nested_candidate_authority() -> None:
    snerv_budget = _snerv_budget()
    candidate = dict(snerv_budget["selected_candidates"][0])
    candidate["score_claim"] = True
    candidate["ready_for_exact_eval_dispatch"] = True
    snerv_budget["selected_candidates"] = [candidate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["candidate"]["score_claim"] is False
    assert snerv_row["candidate"]["ready_for_exact_eval_dispatch"] is False
    assert "selected_candidate_authority_flag_true:score_claim" in snerv_row["blockers"]
    assert "selected_candidate_authority_flag_true:ready_for_exact_eval_dispatch" in (snerv_row["blockers"])
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"


def test_long_training_campaign_plan_blocks_hinerv_candidate_authority_launch(
    tmp_path: Path,
) -> None:
    hinerv_budget = _hinerv_budget()
    candidate = dict(hinerv_budget["selected_candidates"][0])
    candidate["score_claim"] = True
    candidate["promotion_eligible"] = "true"
    hinerv_budget["selected_candidates"] = [candidate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        joint_recon_weight_manifest_paths=(_joint_recon_weight_manifest(tmp_path, num_pairs=600),),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate"]["score_claim"] is False
    assert hi["candidate"]["promotion_eligible"] is False
    assert hi["local_mlx_launch_command_ready"] is False
    assert hi["implementation_status"] == "selected_candidate_authority_flags_block_launch"
    assert "selected_candidate_authority_flag_true:score_claim" in hi["blockers"]
    assert "selected_candidate_authority_flag_true:promotion_eligible" in hi["blockers"]
    assert hi["experiment_queue_entry"]["status"] == "disabled"


def test_long_training_campaign_plan_blocks_partial_hinerv_official_controls(
    tmp_path: Path,
) -> None:
    hinerv_budget = _hinerv_budget()
    partial = dict(hinerv_budget["selected_candidates"][0])
    partial.update(
        {
            "candidate_id": "hinerv_partial_hfg_only",
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": False,
        }
    )
    hinerv_budget["selected_candidates"] = [partial]
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id=partial["candidate_id"],
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill_path = tmp_path / "partial_hinerv_waterfill.json"
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        joint_recon_weight_manifest_paths=(
            _joint_recon_weight_manifest(tmp_path, num_pairs=600),
        ),
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_id"] == "hinerv_partial_hfg_only"
    assert hi["local_mlx_launch_command_ready"] is False
    assert hi["implementation_status"] == "hinerv_official_controls_required_for_launch"
    assert "hinerv_official_control_required_for_top_priority_launch" in hi[
        "blockers"
    ]
    assert "hinerv_official_convnext_blocks_not_enabled" in hi["blockers"]
    assert "hinerv_official_convnext_blocks_not_enabled" in hi[
        "source_faithfulness_controls"
    ]["target_official_control_blockers"]
    assert (
        hi["experiment_queue_entry"]["status"] == "disabled"
    )
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_prefers_rate_plausible_snerv_rows() -> None:
    snerv_budget = _snerv_budget()
    huge_over = dict(snerv_budget["selected_candidates"][0])
    huge_over.update(
        {
            "candidate_id": (
                "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t0_adbase_int2_symmetric_ceil36000"
            ),
            "hard_byte_ceiling": 36_000,
            "decoder_payload_codec": "int2_symmetric",
            "fc_dim": 9,
            "emb_size": 0,
            "decoder_feature_count": 9,
            "nominal_total_payload_bytes": 11_074_662,
            "nominal_under_ceiling": False,
            "byte_headroom": 36_000 - 11_074_662,
        }
    )
    plausible = dict(snerv_budget["selected_candidates"][0])
    plausible.update(
        {
            "candidate_id": (
                "snerv_np600_haar_lv5_lfb2_stepb0p5_fc11e2_p1_mfu1-2-4_hfr0_t0_adbase_int2_symmetric_ceil285000"
            ),
            "hard_byte_ceiling": 285_000,
            "levels": 5,
            "bits_per_coeff": 2.0,
            "decoder_payload_codec": "int2_symmetric",
            "nominal_total_payload_bytes": 231_518,
            "nominal_under_ceiling": True,
            "byte_headroom": 285_000 - 231_518,
            "fc_dim": 11,
            "emb_size": 2,
        }
    )
    snerv_budget["selected_candidates"] = [huge_over, plausible]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["candidate_id"] == plausible["candidate_id"]
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["implementation_status"] == "snerv_scorer_tether_smoke_gate_blocked"
    assert "snerv_nominal_payload_far_over_ceiling_refuse_long_training" not in snerv_row["blockers"]
    assert "snerv_scorer_tether_smoke_report_missing" in snerv_row["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_missing" in snerv_row["blockers"]


def test_long_training_campaign_plan_requires_and_accepts_snerv_nondegenerate_proof() -> None:
    candidate = dict(_snerv_budget()["selected_candidates"][0])
    candidate.update(
        {
            "candidate_id": (
                "snerv_np600_haar_lv5_lfb2_stepb0p5_fc11e2_p1_mfu1-2-4_hfr0_t0_adbase_int2_symmetric_ceil285000"
            ),
            "hard_byte_ceiling": 285_000,
            "levels": 5,
            "bits_per_coeff": 2.0,
            "decoder_payload_codec": "int2_symmetric",
            "nominal_total_payload_bytes": 231_518,
            "nominal_under_ceiling": True,
            "byte_headroom": 285_000 - 231_518,
            "fc_dim": 11,
            "emb_size": 2,
        }
    )
    tether_smoke = {
        "schema": "snerv_scorer_tether_smoke.v1",
        "created_utc": "2026-06-05T00:00:00Z",
        "steps": 2,
        "passed": True,
        "metric_summary": {
            "loss_part_distill": 0.5,
            "loss_part_pose_distill": 0.25,
            "snerv_segnet_last_frame_distill_lambda": 0.01,
            "snerv_posenet_yuv6_pair_distill_lambda": 0.01,
        },
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    renderer_feedback = {
        "schema": "nerv_candidate_feedback_row.v1",
        "family": "snerv",
        "candidate_id": candidate["candidate_id"],
        "candidate_num_pairs": 600,
        "measured_num_pairs": 16,
        "scope_matches_candidate": True,
        "feedback_ready": False,
        "snerv_renderer_nondegenerate_proof_passed": True,
        "snerv_renderer_nondegenerate_blockers": [],
        "snerv_renderer_nondegenerate_proof": {
            "schema": "snerv_renderer_nondegenerate_proof.v1",
            "min_pair_count": 16,
            "measured_num_pairs": 16,
            "scorer_tether_gate_passed": True,
            "telemetry_contract_passed": True,
            "receiver_reconstruction_verified": True,
            "target_value_domain_passed": True,
            "export_value_domain_passed": True,
            "passed": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "direct_feedback_blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(candidate),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_scorer_tether_smoke_report=tether_smoke,
        candidate_feedback_sources=(renderer_feedback,),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert snerv["snerv_renderer_nondegenerate_gate"]["passed"] is True
    assert snerv["local_mlx_launch_command_ready"] is False
    assert snerv["implementation_status"] == "snerv_scoreaware_curriculum_blocked"
    assert "snerv_scorer_loop_qat_pose_guard_not_ready" in snerv["blockers"]
    assert "snerv_native_scorer_loop_best_packet_not_materialized" in snerv[
        "blockers"
    ]
    queue_contract = snerv["experiment_queue_entry"]["launch_authority_contract"]
    assert "snerv_scorer_tether_smoke_report_missing" not in (
        queue_contract["queue_launch_blockers"]
    )
    assert "snerv_renderer_nondegenerate_smoke_missing" not in (
        queue_contract["queue_launch_blockers"]
    )
    assert "snerv_scorer_loop_qat_pose_guard_not_ready" in (
        queue_contract["queue_launch_blockers"]
    )
    assert queue_contract["snerv_renderer_nondegenerate_gate"]["passed"] is True
    assert snerv["score_claim"] is False


def test_long_training_campaign_plan_consumes_snerv_lf_recode_admission() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="spatial_delta_zigzag_leb128_lzma",
                source_packet_bytes=190_000,
                candidate_packet_bytes=160_000,
                source_lf_bytes=120_000,
                candidate_lf_bytes=90_000,
            ),
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    admission = snerv["snerv_lf_payload_recode_admission_plan"]
    argv = snerv["command_argv"]
    queue_metadata = snerv["experiment_queue_entry"]["metadata"]
    launch_contract = snerv["experiment_queue_entry"]["launch_authority_contract"]
    assert report["snerv_lf_payload_recode_source_count"] == 1
    assert admission["schema"] == "snerv_lf_payload_recode_admission_plan.v1"
    assert admission["selected_mode"] == "spatial_delta_zigzag_leb128_lzma"
    assert admission["selected_row"]["packet_byte_delta"] == -30_000
    assert admission["selected_row"]["waterline_crossed_by_recode"] is True
    assert admission["waterline_satisfied_after_selected_recode"] is True
    assert snerv["snerv_lf_payload_codec_from_admission_plan"] is None
    assert "--snerv-scorer-loop-lf-payload-codec" not in argv
    assert "snerv_lf_recode_admission_plan_false_authority" in snerv["blockers"]
    assert "not_packaged_as_contest_archive_zip" in snerv["blockers"]
    assert "full_video_scorer_replay_missing" in snerv["blockers"]
    assert queue_metadata["snerv_lf_payload_recode_admission_plan"][
        "selected_mode"
    ] == "spatial_delta_zigzag_leb128_lzma"
    assert queue_metadata["snerv_lf_payload_codec_from_admission_plan"] is None
    assert launch_contract["queue_status_is_exact_eval_authority"] is False
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_reroutes_post_recode_packet_overrun() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="auto",
                source_packet_bytes=2_347_142,
                candidate_packet_bytes=1_485_285,
                candidate_packet_header_bytes=1_346_233,
                candidate_packet_path="/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
                source_lf_bytes=879_633,
                candidate_lf_bytes=17_779,
            ),
        ),
        snerv_snar_header_grammar_profile_sources=(
            _snerv_snar_header_grammar_profile(
                packet_sha256="b" * 64,
                packet_bytes=1_485_285,
                header_bytes=1_346_233,
                metadata_json_bytes=1_345_466,
                section_total_bytes=139_052,
            ),
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    reroute_queue = report["snerv_lf_over_ceiling_reroute_queue"]
    expected_overrun = 1_485_285 - snerv["hard_byte_ceiling"]
    assert reroute_queue["local_recode_command_row_count"] == 0
    assert reroute_queue["local_executable_command_row_count"] == 1
    assert reroute_queue["queue_row_count"] == 4
    assert (
        "snerv_lf_recode_selected_mode_still_over_byte_waterline"
        in snerv["experiment_queue_entry"]["launch_authority_contract"][
            "queue_launch_blockers"
        ]
    )

    rows_by_type = {row["work_order_type"]: row for row in reroute_queue["queue_rows"]}
    recode_result = rows_by_type["lf_recode_admission_result"]
    assert recode_result["blocked"] is True
    assert recode_result["lossless_lf_recode_already_admitted"] is True
    assert recode_result["candidate_lf_payload_bytes"] == 17_779
    assert recode_result["candidate_packet_header_bytes"] == 1_346_233
    assert recode_result["snar_header_grammar_profile_attached"] is True
    assert recode_result["snar_header_grammar_profile"]["header_bytes"] == 1_346_233
    assert recode_result["snar_header_grammar_profile"]["top_metadata_contributors"][1][
        "path"
    ] == "$.metadata.lf_step_allocation_rows"
    assert recode_result["post_recode_over_waterline_bytes"] == expected_overrun
    assert recode_result["required_lf_savings_bytes"] == expected_overrun
    assert recode_result["lf_payload_can_cover_required_savings"] is False
    assert (
        "snerv_post_recode_packet_still_over_hard_byte_ceiling"
        in recode_result["blockers"]
    )
    assert (
        "snerv_post_recode_overrun_exceeds_remaining_lf_payload_bytes"
        in recode_result["blockers"]
    )
    assert (
        "snerv_post_recode_overrun_dominated_by_packet_header_bytes"
        in recode_result["blockers"]
    )
    assert (
        "snerv_snar_packet_header_grammar_rewrite_required"
        in recode_result["blockers"]
    )
    assert snerv["snerv_lf_payload_recode_admission_plan"]["verdict"] == (
        "ADMIT_LF_RECODE__POST_RECODE_PACKET_HEADER_GRAMMAR_DOMINATES"
    )
    assert (
        "attack_snerv_snar_packet_header_grammar_or_packaging_overhead"
        in snerv["snerv_lf_payload_recode_admission_plan"]["next_actions"]
    )

    header_rewrite = rows_by_type["snar_header_grammar_rewrite_materialization"]
    assert header_rewrite["blocked"] is False
    assert header_rewrite["command_argv"][:4] == [
        "uv",
        "run",
        "python",
        "tools/minimize_snerv_snar_header.py",
    ]
    assert "--packet" in header_rewrite["command_argv"]
    assert header_rewrite["command_argv"][
        header_rewrite["command_argv"].index("--packet") + 1
    ] == "/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar"
    assert "--candidate-id" in header_rewrite["command_argv"]
    assert header_rewrite["command_argv"][
        header_rewrite["command_argv"].index("--candidate-id") + 1
    ] == snerv["candidate_id"]
    assert "--wire-format" in header_rewrite["command_argv"]
    assert header_rewrite["command_argv"][
        header_rewrite["command_argv"].index("--wire-format") + 1
    ] == "snar2"
    assert "--output-packet" in header_rewrite["command_argv"]
    assert "--output-archive-zip" in header_rewrite["command_argv"]
    assert header_rewrite["command_argv"][
        header_rewrite["command_argv"].index("--output-archive-zip") + 1
    ].endswith("/archive.zip")
    assert "--output-package-dir" in header_rewrite["command_argv"]
    assert header_rewrite["command_argv"][
        header_rewrite["command_argv"].index("--output-package-dir") + 1
    ].endswith("/runtime_package")
    assert "--full-video-receiver-proof" in header_rewrite["command_argv"]
    assert "--hard-byte-ceiling" in header_rewrite["command_argv"]
    assert int(
        header_rewrite["command_argv"][
            header_rewrite["command_argv"].index("--hard-byte-ceiling") + 1
        ]
    ) == snerv["hard_byte_ceiling"]
    assert header_rewrite["dispatch_allowed"] is False
    assert header_rewrite["local_mlx_long_training_allowed"] is False
    assert (
        header_rewrite["planner_action"]
        == "run_receiver_proven_snar2_binary_header_prune_then_rerun_recode_admission"
    )
    assert header_rewrite["blockers"] == []

    representation_rows = [
        row
        for row in reroute_queue["queue_rows"]
        if row["work_order_type"] == "lf_representation_change_candidate"
    ]
    assert len(representation_rows) == 2
    assert all(row["measured_lf_payload_bytes"] == 17_779 for row in representation_rows)
    assert all(
        row["lf_payload_can_cover_required_savings"] is False
        for row in representation_rows
    )
    assert all(
        "snerv_required_savings_exceeds_measured_lf_payload_bytes" in row["blockers"]
        for row in representation_rows
    )
    assert all(
        "snerv_snar_header_grammar_rewrite_precedes_lf_representation_change"
        in row["blockers"]
        for row in representation_rows
    )


def test_long_training_campaign_plan_consumes_snerv_header_minimization_result() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="auto",
                source_packet_bytes=2_347_142,
                candidate_packet_bytes=1_485_285,
                candidate_packet_header_bytes=1_346_233,
                candidate_packet_path="/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
                source_lf_bytes=879_633,
                candidate_lf_bytes=17_779,
            ),
        ),
        snerv_snar_header_grammar_profile_sources=(
            _snerv_snar_header_grammar_profile(
                packet_sha256="b" * 64,
                packet_bytes=1_485_285,
                header_bytes=1_346_233,
                metadata_json_bytes=1_345_466,
                section_total_bytes=139_052,
            ),
        ),
        snerv_snar_header_minimization_report_sources=(
            _snerv_snar_header_minimization_report(source_packet_sha256="b" * 64),
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    reroute_queue = report["snerv_lf_over_ceiling_reroute_queue"]
    rows_by_type = {row["work_order_type"]: row for row in reroute_queue["queue_rows"]}
    result = rows_by_type["snar_header_minimization_result"]
    assert report["launchable_local_row_count"] == 1
    assert snerv["experiment_queue_entry"]["status"] == "disabled"
    assert reroute_queue["snar_header_minimization_report_count"] == 1
    assert result["blocked"] is True
    assert result["command_argv"] == []
    assert result["snar_header_minimization_report_attached"] is True
    assert result["snar_header_minimization_report"]["candidate_packet_bytes"] == 139_855
    assert result["snar_header_minimization_report"][
        "candidate_archive_zip_bytes"
    ] == 139_963
    assert result["snar_header_minimization_report"]["receiver_contract_satisfied"] is True
    assert (
        "snerv_snar_header_minimized_packet_candidate_id_binding_missing"
        in result["blockers"]
    )
    assert (
        "snerv_snar_header_minimized_packet_full_video_replay_missing"
        in result["blockers"]
    )
    assert result["score_claim"] is False
    representation_rows = [
        row
        for row in reroute_queue["queue_rows"]
        if row["work_order_type"] == "lf_representation_change_candidate"
    ]
    assert all(
        "snerv_snar_header_minimization_result_precedes_lf_representation_change"
        in row["blockers"]
        for row in representation_rows
    )


def test_long_training_campaign_plan_accepts_full_video_header_minimization_proof() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="auto",
                source_packet_bytes=2_347_142,
                candidate_packet_bytes=1_485_285,
                candidate_packet_header_bytes=1_346_233,
                candidate_packet_path="/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
                source_lf_bytes=879_633,
                candidate_lf_bytes=17_779,
            ),
        ),
        snerv_snar_header_minimization_report_sources=(
            _snerv_snar_header_minimization_report(
                source_packet_sha256="b" * 64,
                candidate_id=_snerv_candidate_id(),
                full_video_receiver_contract_satisfied=True,
            ),
        ),
    )

    reroute_queue = report["snerv_lf_over_ceiling_reroute_queue"]
    rows_by_type = {row["work_order_type"]: row for row in reroute_queue["queue_rows"]}
    result = rows_by_type["snar_header_minimization_result"]
    assert result["snar_header_minimization_report"][
        "full_video_receiver_contract_satisfied"
    ] is True
    assert (
        "snerv_snar_header_minimized_packet_full_video_replay_missing"
        not in result["blockers"]
    )
    assert (
        "snerv_snar_header_minimized_packet_candidate_id_binding_missing"
        not in result["blockers"]
    )
    assert "paired_contest_cpu_cuda_auth_eval_missing" in result["blockers"]
    assert result["score_claim"] is False


def test_long_training_campaign_plan_queues_step_map_compaction_after_snar2_proof() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="auto",
                source_packet_bytes=2_347_142,
                candidate_packet_bytes=1_485_285,
                candidate_packet_header_bytes=1_346_233,
                candidate_packet_path="/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
                source_lf_bytes=879_633,
                candidate_lf_bytes=17_779,
            ),
        ),
        snerv_snar_header_minimization_report_sources=(
            _snerv_snar_header_minimization_report(
                source_packet_sha256="b" * 64,
                candidate_id=_snerv_candidate_id(),
                full_video_receiver_contract_satisfied=True,
            ),
        ),
    )

    reroute_queue = report["snerv_lf_over_ceiling_reroute_queue"]
    rows_by_type = {row["work_order_type"]: row for row in reroute_queue["queue_rows"]}
    step_map = rows_by_type["snar_step_map_packet_compaction_materialization"]
    assert reroute_queue["local_executable_command_row_count"] == 1
    assert step_map["blocked"] is False
    assert step_map["planner_action"] == (
        "run_receiver_proven_step_map_constant_shape_partition_compaction"
    )
    assert step_map["command_argv"][:4] == [
        "uv",
        "run",
        "python",
        "tools/materialize_snerv_step_map_compaction.py",
    ]
    assert step_map["command_argv"][
        step_map["command_argv"].index("--packet") + 1
    ] == "/Volumes/VertigoDataTier/pact/snerv_test/candidate.minimized.snar"
    assert step_map["command_argv"][
        step_map["command_argv"].index("--candidate-id") + 1
    ] == _snerv_candidate_id()
    assert step_map["command_argv"][
        step_map["command_argv"].index("--wire-format") + 1
    ] == "snar2"
    assert step_map["command_argv"][
        step_map["command_argv"].index("--output-packet") + 1
    ].endswith("/candidate.stepmap.snar2")
    assert "--output-package-dir" in step_map["command_argv"]
    assert "--full-video-receiver-proof" in step_map["command_argv"]
    assert "--hard-byte-ceiling" in step_map["command_argv"]
    assert step_map["dispatch_allowed"] is False
    assert step_map["local_mlx_long_training_allowed"] is False
    assert step_map["score_claim"] is False
    representation_rows = [
        row
        for row in reroute_queue["queue_rows"]
        if row["work_order_type"] == "lf_representation_change_candidate"
    ]
    assert all(
        "snerv_step_map_packet_compaction_precedes_lf_representation_change"
        in row["blockers"]
        for row in representation_rows
    )


def test_long_training_campaign_plan_accepts_header_minimization_candidate_alias_for_step_map_queue() -> None:
    alias_candidate_id = f"native_rate_aware_training_{_snerv_candidate_id()}_los"
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="auto",
                source_packet_bytes=2_347_142,
                candidate_packet_bytes=1_485_285,
                candidate_packet_header_bytes=1_346_233,
                candidate_packet_path="/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
                source_lf_bytes=879_633,
                candidate_lf_bytes=17_779,
            ),
        ),
        snerv_snar_header_minimization_report_sources=(
            _snerv_snar_header_minimization_report(
                source_packet_sha256="b" * 64,
                candidate_id=alias_candidate_id,
                full_video_receiver_contract_satisfied=True,
            ),
        ),
    )

    reroute_queue = report["snerv_lf_over_ceiling_reroute_queue"]
    rows_by_type = {row["work_order_type"]: row for row in reroute_queue["queue_rows"]}
    result = rows_by_type["snar_header_minimization_result"]
    step_map = rows_by_type["snar_step_map_packet_compaction_materialization"]
    assert result["snar_header_minimization_report"]["candidate_binding"][
        "candidate_id"
    ] == alias_candidate_id
    assert (
        "snerv_snar_header_minimized_packet_candidate_id_binding_missing"
        not in result["blockers"]
    )
    assert step_map["blocked"] is False
    assert (
        "snerv_step_map_compaction_candidate_id_binding_missing"
        not in step_map["blockers"]
    )
    assert step_map["command_argv"][
        step_map["command_argv"].index("--candidate-id") + 1
    ] == _snerv_candidate_id()


def test_long_training_campaign_plan_prefers_candidate_bound_header_minimization() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_lf_payload_recode_sources=(
            _snerv_lf_recode_report(
                mode="auto",
                source_packet_bytes=2_347_142,
                candidate_packet_bytes=1_485_285,
                candidate_packet_header_bytes=1_346_233,
                candidate_packet_path="/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
                source_lf_bytes=879_633,
                candidate_lf_bytes=17_779,
            ),
        ),
        snerv_snar_header_minimization_report_sources=(
            _snerv_snar_header_minimization_report(
                source_packet_sha256="b" * 64,
                candidate_id="other_snerv_candidate",
                candidate_packet_bytes=141_000,
            ),
            _snerv_snar_header_minimization_report(
                source_packet_sha256="b" * 64,
                candidate_id=_snerv_candidate_id(),
                candidate_packet_bytes=139_855,
            ),
        ),
    )

    rows_by_type = {
        row["work_order_type"]: row
        for row in report["snerv_lf_over_ceiling_reroute_queue"]["queue_rows"]
    }
    result = rows_by_type["snar_header_minimization_result"]
    assert result["snar_header_minimization_report"]["candidate_packet_bytes"] == 139_855
    assert result["snar_header_minimization_report"]["candidate_binding"][
        "candidate_id"
    ] == _snerv_candidate_id()
    assert (
        "snerv_snar_header_minimized_packet_candidate_id_binding_missing"
        not in result["blockers"]
    )


def test_long_training_campaign_plan_dedupes_snerv_candidate_ids() -> None:
    snerv_budget = _snerv_budget()
    first = dict(snerv_budget["selected_candidates"][0])
    duplicate = dict(first)
    duplicate["nominal_total_payload_bytes"] = int(first["nominal_total_payload_bytes"]) - 1
    snerv_budget["selected_candidates"] = [first, duplicate]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=2,
    )

    snerv_rows = [row for row in report["campaign_rows"] if row["family"] == "snerv"]
    assert len(snerv_rows) == 1
    assert snerv_rows[0]["candidate_id"] == first["candidate_id"]


def test_long_training_campaign_plan_refuses_far_over_ceiling_snerv_long_run() -> None:
    snerv_budget = _snerv_budget()
    huge_over = dict(snerv_budget["selected_candidates"][0])
    huge_over.update(
        {
            "candidate_id": (
                "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t0_adbase_int2_symmetric_ceil36000"
            ),
            "hard_byte_ceiling": 36_000,
            "decoder_payload_codec": "int2_symmetric",
            "fc_dim": 9,
            "emb_size": 0,
            "decoder_feature_count": 9,
            "nominal_total_payload_bytes": 11_074_662,
            "nominal_under_ceiling": False,
            "byte_headroom": 36_000 - 11_074_662,
        }
    )
    snerv_budget["selected_candidates"] = [huge_over]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    snerv_row = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["implementation_status"] == "snerv_scorer_tether_smoke_gate_blocked"
    assert "snerv_scorer_tether_smoke_report_missing" in snerv_row["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_missing" in snerv_row["blockers"]
    assert "snerv_nominal_payload_far_over_ceiling_refuse_long_training" in snerv_row["blockers"]
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"


def test_long_training_campaign_plan_accepts_unique_experiment_queue_id() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        experiment_queue_id="nerv_hinerv_lr_recovery_unique_queue.v1",
    )

    assert report["experiment_queue_id"] == "nerv_hinerv_lr_recovery_unique_queue.v1"
    assert report["experiment_queue"]["queue_id"] == ("nerv_hinerv_lr_recovery_unique_queue.v1")


def test_long_training_campaign_plan_rejects_empty_experiment_queue_id() -> None:
    with pytest.raises(NervLongTrainingCampaignPlanError, match="experiment_queue_id must be non-empty"):
        build_nerv_long_training_campaign_plan(
            hinerv_modelsize_budget=_hinerv_budget(),
            snerv_modelsize_budget=_snerv_budget(),
            optimizer_kinds=("lion",),
            epochs=29_650,
            output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
            max_candidates_per_family=1,
            experiment_queue_id="",
        )


def test_long_training_campaign_plan_pins_verified_joint_recon_weight(
    tmp_path: Path,
) -> None:
    manifest = _joint_recon_weight_manifest(tmp_path, num_pairs=600)

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        joint_recon_weight_manifest_paths=(manifest,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--auto-joint-recon-pixel-weight" not in argv
    assert "--recon-pixel-weight-path" in argv
    weight_arg = argv[argv.index("--recon-pixel-weight-path") + 1]
    assert Path(weight_arg).is_file()
    assert "requires_verified_joint_p18_p19_recon_pixel_weight_artifact" not in hi["blockers"]
    artifact = hi["joint_recon_pixel_weight_artifact"]
    assert artifact["num_pairs"] == 600
    assert artifact["manifest_path"] == manifest.as_posix()
    assert artifact["score_claim"] is False
    assert report["joint_recon_weight_artifact_count"] == 1


def test_long_training_campaign_plan_attaches_hinerv_decoder_weight_waterfill(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill.json"
    waterfill = _decoder_weight_waterfill_plan(candidate_id="hinerv_tiny")
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--decoder-weight-waterfill-plan-json" not in argv
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["sha256"] == _sha256(waterfill_path)
    assert attachment["candidate_keys"] == ["hinerv_tiny"]
    assert attachment["runner_admitted"] is False
    assert attachment["runner_admission"]["mode"] == "advisory_learning_signal_only"
    refusal_reasons = attachment["runner_admission"]["refusal_reasons"]
    assert "decoder_weight_waterfill_receiver_proof_not_ready" in refusal_reasons
    assert "receiver_proof_not_satisfied" in refusal_reasons
    assert hi["experiment_queue_entry"]["metadata"]["decoder_weight_waterfill_plan"]["attached"] is True
    assert "hinerv_decoder_weight_waterfill_plan_missing" not in hi["blockers"]
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" in hi["blockers"]
    assert (
        "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted"
        in hi["score_lowering_gate"]["launch_blockers"]
    )
    assert hi["implementation_status"] == "decoder_weight_waterfill_plan_advisory_only_blocks_launch"
    assert hi["local_mlx_launch_command_ready"] is False
    assert report["decoder_weight_waterfill_source_count"] == 1
    assert report["decoder_weight_waterfill_attached_row_count"] == 1
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_admits_receiver_proven_hinerv_waterfill(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_receiver_ready.json"
    proof_path = _receiver_proof(tmp_path, archive_sha="a" * 64)
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_tiny",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill["receiver_proof_path"] = proof_path.as_posix()
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--decoder-weight-waterfill-plan-json" in argv
    assert argv[argv.index("--decoder-weight-waterfill-plan-json") + 1] == (waterfill_path.as_posix())
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["runner_admitted"] is True
    assert attachment["runner_admission"]["mode"] == ("runner_training_pressure_and_export_mutation")
    assert attachment["runner_admission"]["refusal_reasons"] == []
    assert attachment["receiver_proof_binding"]["bound"] is True
    assert attachment["receiver_proof_binding"]["proof_path"] == proof_path.as_posix()
    assert attachment["receiver_proof_binding"]["proof_archive_sha256"] == "a" * 64
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" not in hi["blockers"]


def test_long_training_campaign_plan_attaches_hinerv_archive_section_telemetry_fixture_path(
    tmp_path: Path,
) -> None:
    telemetry_path = tmp_path / "hinerv_archive_section_telemetry.json"
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv_tiny",
        "profile_ready": True,
        "archive_zip_bytes": 160_000,
        "section_payload_bytes": 159_000,
        "sections": [
            {"name": "decoder_state", "bytes": 120_000},
            {"name": "latents", "bytes": 39_000},
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")
    telemetry["_archive_section_telemetry_path"] = telemetry_path.as_posix()
    telemetry["_archive_section_telemetry_source_path"] = telemetry_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        archive_section_telemetry_sources=(telemetry,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--archive-section-telemetry-json" not in argv
    attachment = hi["archive_section_telemetry"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is False
    assert attachment["sha256"] == _sha256(telemetry_path)
    assert attachment["candidate_keys"] == ["hinerv_tiny"]
    assert attachment["section_names"] == ["decoder_state", "latents"]
    assert "hinerv_archive_section_telemetry_receiver_proof_path_missing" in (
        attachment["runner_admission"]["refusal_reasons"]
    )
    assert (
        "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted"
        in hi["blockers"]
    )
    assert report["archive_section_telemetry_source_count"] == 1
    assert report["archive_section_telemetry_row_count"] == 1
    assert report["archive_section_telemetry_attached_row_count"] == 1
    assert report["archive_section_telemetry_unattached_source_count"] == 0


def test_long_training_campaign_plan_rejects_status_only_receiver_proof(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_status_only.json"
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_tiny",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["receiver_proof_ready"] is False
    assert attachment["runner_admitted"] is False
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]
    assert "decoder_weight_waterfill_receiver_proof_path_missing" in attachment[
        "runner_admission"
    ]["refusal_reasons"]
    assert attachment["receiver_proof_binding"]["bound"] is False
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" in hi["blockers"]


def test_long_training_campaign_plan_rejects_generic_receiver_proof_string(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_generic_passed.json"
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_tiny",
        receiver_proof_status="passed",
    )
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["receiver_proof_ready"] is False
    assert attachment["runner_admitted"] is False
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]
    refusal_reasons = attachment["runner_admission"]["refusal_reasons"]
    assert "decoder_weight_waterfill_receiver_proof_not_ready" in refusal_reasons
    assert "receiver_proof_not_satisfied" in refusal_reasons
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" in hi["blockers"]


def test_long_training_campaign_plan_attaches_hinerv_archive_section_telemetry(
    tmp_path: Path,
) -> None:
    proof_path = _receiver_proof(tmp_path, archive_sha="b" * 64)
    cache_path = _receiver_cache_quality_report(tmp_path, passed=True)
    telemetry_path = tmp_path / "hi_nerv_archive_section_telemetry.json"
    telemetry = _archive_section_telemetry(
        candidate_id="hinerv_tiny",
        archive_sha="b" * 64,
        archive_zip_bytes=177_500,
        receiver_proof_path=proof_path,
        cache_quality_report_path=cache_path,
    )
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")
    telemetry["_archive_section_telemetry_path"] = telemetry_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        archive_section_telemetry_sources=(telemetry,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    assert "--archive-section-telemetry-json" in argv
    assert argv[argv.index("--archive-section-telemetry-json") + 1] == (
        telemetry_path.as_posix()
    )
    attachment = hi["archive_section_telemetry"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is True
    assert attachment["sha256"] == _sha256(telemetry_path)
    assert attachment["candidate_keys"] == ["hinerv_tiny"]
    assert attachment["section_count"] == 3
    assert attachment["section_names"] == [
        "archive_zip_overhead",
        "decoder_state",
        "latents_mid",
    ]
    assert attachment["decoder_state_section_present"] is True
    assert attachment["archive_under_hard_byte_ceiling"] is True
    assert attachment["receiver_proof_binding"]["bound"] is True
    assert attachment["receiver_cache_quality_binding"]["bound"] is True
    assert report["archive_section_telemetry_source_count"] == 1
    assert report["archive_section_telemetry_row_count"] == 1
    assert report["archive_section_telemetry_attached_row_count"] == 1
    assert report["archive_section_telemetry_unattached_source_count"] == 0
    assert "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted" not in hi[
        "blockers"
    ]
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_blocks_bad_hinerv_archive_section_telemetry(
    tmp_path: Path,
) -> None:
    proof_path = _receiver_proof(tmp_path, archive_sha="b" * 64)
    cache_path = _receiver_cache_quality_report(tmp_path, passed=False)
    telemetry_path = tmp_path / "hi_nerv_archive_section_telemetry_bad.json"
    telemetry = _archive_section_telemetry(
        candidate_id="hinerv_tiny",
        archive_sha="b" * 64,
        archive_zip_bytes=181_000,
        receiver_proof_path=proof_path,
        cache_quality_report_path=cache_path,
        cache_quality_passed=False,
        profile_ready=False,
        sections=({"name": "latents_mid", "role": "latent", "bytes": 64},),
    )
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")
    telemetry["_archive_section_telemetry_path"] = telemetry_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        archive_section_telemetry_sources=(telemetry,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["archive_section_telemetry"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is False
    assert "--archive-section-telemetry-json" not in hi["command_argv"]
    assert "hinerv_archive_section_telemetry_not_profile_ready" in attachment[
        "blockers"
    ]
    assert "hinerv_archive_section_telemetry_decoder_state_missing" in attachment[
        "blockers"
    ]
    assert "hinerv_archive_section_telemetry_archive_not_under_hard_byte_ceiling" in (
        attachment["runner_admission"]["refusal_reasons"]
    )
    assert (
        "hinerv_archive_section_telemetry_receiver_cache_quality_gate_not_passed"
        in attachment["runner_admission"]["refusal_reasons"]
    )
    assert "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted" in hi[
        "blockers"
    ]
    assert "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted" in hi[
        "score_lowering_gate"
    ]["launch_blockers"]
    assert "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted" in hi[
        "experiment_queue_entry"
    ]["launch_authority_contract"]["queue_launch_blockers"]
    assert hi["experiment_queue_entry"]["launch_authority_contract"][
        "queue_status_is_runnable_plan"
    ] is False
    assert hi["experiment_queue_entry"]["status"] == "disabled"
    assert hi["local_mlx_launch_command_ready"] is False


def test_long_training_campaign_plan_routes_hinerv_hard_pair_feedback(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_receiver_ready.json"
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_tiny",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()
    feedback = {
        "schema": "nerv_candidate_feedback_row.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv_tiny",
        "scope_matches_candidate": True,
        "measured_num_pairs": 600,
        "hard_pair_coverage": {
            "schema": "nerv_hard_pair_coverage_evidence.v1",
            "representative_distortion_evidence": True,
            "prioritized_pair_indices": [17, 4, 17, 0],
            "hard_pair_count": 3,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        joint_recon_weight_manifest_paths=(
            _joint_recon_weight_manifest(tmp_path, num_pairs=600),
        ),
        decoder_weight_waterfill_sources=(waterfill,),
        candidate_feedback_sources=(feedback,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["prioritized_pair_training"]["schema"] == (
        "nerv_prioritized_pair_training_plan.v1"
    )
    assert hi["prioritized_pair_training"]["enabled"] is True
    assert hi["prioritized_pair_training"]["pair_indices"] == [17, 4, 0]
    assert "--prioritized-pair-indices" in hi["command_argv"]
    assert (
        hi["command_argv"][hi["command_argv"].index("--prioritized-pair-indices") + 1]
        == "17,4,0"
    )
    queue = hi["experiment_queue_entry"]
    assert queue["metadata"]["prioritized_pair_training"]["pair_indices"] == [
        17,
        4,
        0,
    ]
    assert queue["steps"][0]["command"] == hi["command_argv"]
    assert queue["steps"][0]["command"][
        queue["steps"][0]["command"].index("--prioritized-pair-indices") + 1
    ] == "17,4,0"


def test_long_training_campaign_plan_blocks_invalid_hard_pair_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "scope_matches_candidate": True,
                "measured_num_pairs": 600,
                "hard_pair_coverage": {
                    "schema": "nerv_hard_pair_coverage_evidence.v1",
                    "representative_distortion_evidence": True,
                    "prioritized_pair_indices": [17, 1.9],
                    "hard_pair_count": 2,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["prioritized_pair_training"]["enabled"] is False
    assert hi["prioritized_pair_training"]["pair_indices"] == []
    assert "--prioritized-pair-indices" not in hi["command_argv"]
    assert "candidate_feedback_prioritized_pair_indices_parse_failed" in hi[
        "candidate_feedback_evidence_blockers"
    ]
    assert "candidate_feedback_prioritized_pair_indices_parse_failed" in hi[
        "blockers"
    ]
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_blocks_bare_snerv_hard_pair_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "snerv",
                "candidate_id": _snerv_candidate_id(),
                "scope_matches_candidate": True,
                "measured_num_pairs": 600,
                "hard_pair_indices": [417, 22, 417],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv["prioritized_pair_training"]["enabled"] is False
    assert snerv["prioritized_pair_training"]["pair_indices"] == []
    assert "--prioritized-pair-indices" not in snerv["command_argv"]
    assert "candidate_feedback_prioritized_pair_indices_not_launch_routable" in snerv[
        "candidate_feedback_evidence_blockers"
    ]
    assert "candidate_feedback_prioritized_pair_indices_not_launch_routable" in snerv[
        "blockers"
    ]
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_blocks_snerv_representative_hard_pair_command_routing() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "snerv",
                "candidate_id": _snerv_candidate_id(),
                "scope_matches_candidate": True,
                "measured_num_pairs": 600,
                "hard_pair_coverage": {
                    "schema": "nerv_hard_pair_coverage_evidence.v1",
                    "representative_distortion_evidence": True,
                    "prioritized_pair_indices": [417, 22, 417],
                    "hard_pair_count": 2,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv["prioritized_pair_training"]["schema"] == (
        "nerv_prioritized_pair_training_plan.v1"
    )
    assert snerv["prioritized_pair_training"]["enabled"] is False
    assert snerv["prioritized_pair_training"]["requested"] is True
    assert snerv["prioritized_pair_training"]["command_routed"] is False
    assert snerv["prioritized_pair_training"]["pair_indices"] == []
    assert snerv["prioritized_pair_training"]["requested_pair_indices"] == [417, 22]
    assert snerv["prioritized_pair_training"]["blocked_pair_indices"] == [417, 22]
    assert snerv["prioritized_pair_training"]["requested_pair_count"] == 2
    assert snerv["prioritized_pair_training"]["required_successor"] == (
        "snerv_full_video_scoreaware_trainer_with_sampler_emphasis"
    )
    assert "snerv_hardpair_indices_only_hydrated_subset_not_full_training" in snerv[
        "prioritized_pair_training"
    ]["blockers"]
    assert "snerv_hardpair_indices_only_hydrated_subset_not_full_training" in snerv[
        "blockers"
    ]
    assert "--prioritized-pair-indices" not in snerv["command_argv"]
    queue = snerv["experiment_queue_entry"]
    assert queue["metadata"]["prioritized_pair_training"]["requested_pair_indices"] == [
        417,
        22,
    ]
    assert queue["metadata"]["prioritized_pair_training"]["command_routed"] is False
    assert "snerv_hardpair_indices_only_hydrated_subset_not_full_training" in queue[
        "metadata"
    ]["prioritized_pair_training"]["blockers"]
    assert queue["steps"][0]["command"] == snerv["command_argv"]
    assert "--prioritized-pair-indices" not in queue["steps"][0]["command"]
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_blocks_pose_tail_burst_without_pair_indices() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": False,
                "pose_tail_burst_detected": True,
                "recommended_launch_mutations": [
                    "build_xray_hardpair_hitlist_from_full_video_pose_tail",
                    "launch_hard_pair_prioritized_sampler_successor",
                ],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["launch_control_feedback_ready"] is True
    assert adjustment["applied"] is False
    assert adjustment["pose_tail_burst_detected"] is True
    assert adjustment["reason"] == "pose_tail_burst_requires_prioritized_pair_indices"
    assert hi["prioritized_pair_training"]["enabled"] is False
    assert "--prioritized-pair-indices" not in hi["command_argv"]
    assert "hinerv_pose_tail_burst_requires_prioritized_pair_indices" in hi["blockers"]


def test_long_training_campaign_plan_routes_pose_tail_burst_pair_indices() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": False,
                "pose_tail_burst_detected": True,
                "hard_pair_indices": [42, 3, 42],
                "recommended_launch_mutations": [
                    "launch_hard_pair_prioritized_sampler_successor",
                ],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["prioritized_pair_training"]["enabled"] is True
    assert hi["prioritized_pair_training"]["pair_indices"] == [42, 3]
    assert "--prioritized-pair-indices" in hi["command_argv"]
    assert (
        hi["command_argv"][hi["command_argv"].index("--prioritized-pair-indices") + 1]
        == "42,3"
    )
    assert "hinerv_pose_tail_burst_requires_prioritized_pair_indices" not in hi["blockers"]


def test_long_training_campaign_plan_attaches_hinerv_waterfill_from_full_row_id(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_full_row_id.json"
    waterfill = _decoder_weight_waterfill_plan(candidate_id="hi_nerv::hinerv_tiny::lion")
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is False
    assert "hinerv_tiny" in attachment["candidate_keys"]
    assert "lion" not in attachment["candidate_keys"]
    assert "hinerv_decoder_weight_waterfill_plan_missing" not in hi["blockers"]
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" in hi["blockers"]
    assert report["decoder_weight_waterfill_attached_row_count"] == 1
    assert report["decoder_weight_waterfill_unattached_source_count"] == 0


def test_long_training_campaign_plan_attaches_hinerv_waterfill_from_group_row_id(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_group_row_id.json"
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id=("hinerv_tiny:hi_nerv_decoder_weight_waterfill:blocks.0.conv.weight")
    )
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is False
    assert "hinerv_tiny" in attachment["candidate_keys"]
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]
    assert "hinerv_decoder_weight_waterfill_plan_missing" not in hi["blockers"]
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" in hi["blockers"]
    assert report["decoder_weight_waterfill_attached_row_count"] == 1


def test_long_training_campaign_plan_records_unattached_decoder_weight_waterfill(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "decoder_weight_waterfill_wrong_candidate.json"
    waterfill = _decoder_weight_waterfill_plan(candidate_id="hinerv_wrong_shape")
    waterfill_path.write_text(json.dumps(waterfill, sort_keys=True), encoding="utf-8")
    waterfill["_decoder_weight_waterfill_plan_path"] = waterfill_path.as_posix()
    waterfill["_decoder_weight_waterfill_plan_sha256"] = _sha256(waterfill_path)
    waterfill["_decoder_weight_waterfill_source_path"] = waterfill_path.as_posix()

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root=tmp_path / "campaigns",
        max_candidates_per_family=1,
        decoder_weight_waterfill_sources=(waterfill,),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]
    assert hi["decoder_weight_waterfill_plan"]["attached"] is False
    assert "hinerv_decoder_weight_waterfill_plan_missing" in hi["blockers"]
    assert report["decoder_weight_waterfill_attached_row_count"] == 0
    assert report["decoder_weight_waterfill_unattached_source_count"] == 1
    [unattached] = report["decoder_weight_waterfill_unattached_sources"]
    assert unattached["reason"] == "no_matching_campaign_candidate_id"
    assert unattached["source_candidate_id"] == "hinerv_wrong_shape"
    assert unattached["target_candidate_ids"] == ["hinerv_tiny"]
    assert unattached["sha256"] == _sha256(waterfill_path)
    assert unattached["score_claim"] is False


def test_long_training_campaign_plan_keeps_snerv_bounded_proof_explicit() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_bounded_proof_only=True,
        snerv_bounded_proof_epochs=5,
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert report["snerv_bounded_proof_only"] is True
    assert snerv["execution_epochs"] == 5
    assert snerv["current_command_is_bounded_proof_not_long_training"] is True
    assert snerv["implementation_status"] == "snerv_scorer_tether_smoke_gate_blocked"
    assert snerv["command_argv"][snerv["command_argv"].index("--epochs") + 1] == "5"
    assert "snerv_scorer_tether_smoke_report_missing" in snerv["blockers"]
    assert "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only" in snerv["blockers"]
    assert snerv["curriculum_plan"]["training_plan"]["native_mlx_long_training_bound"] is False


def test_long_training_campaign_plan_threads_modelsize_byte_cap_feedback_paths() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(
            "/Volumes/VertigoDataTier/pact/exports/hi.json",
            "/Volumes/VertigoDataTier/pact/exports/sn.json",
        ),
    )

    assert report["modelsize_byte_cap_feedback_path_count"] == 2
    assert report["modelsize_byte_cap_feedback_paths"] == [
        "/Volumes/VertigoDataTier/pact/exports/hi.json",
        "/Volumes/VertigoDataTier/pact/exports/sn.json",
    ]
    for row in report["campaign_rows"]:
        argv = row["command_argv"]
        assert row["runner_modelsize_candidate_id"] == "auto"
        assert row["modelsize_candidate_selection_mode"] == (
            "calibrated_auto_from_modelsize_byte_cap_feedback"
        )
        assert argv[argv.index("--modelsize-candidate-id") + 1] == "auto"
        assert argv[argv.index("--hard-byte-ceiling") + 1] == str(
            row["candidate"]["hard_byte_ceiling"]
        )
        indices = [
            index
            for index, value in enumerate(argv)
            if value == "--modelsize-byte-cap-feedback-json"
        ]
        assert len(indices) == 2
        assert [argv[index + 1] for index in indices] == report[
            "modelsize_byte_cap_feedback_paths"
        ]


def test_long_training_campaign_plan_auto_bytecap_queue_ids_stay_unique(
    tmp_path: Path,
) -> None:
    first = dict(_snerv_budget()["selected_candidates"][0])
    second = dict(first)
    second["candidate_id"] = str(first["candidate_id"]).replace(
        "fc11e2",
        "fc9e0",
    )
    second["fc_dim"] = 9
    second["decoder_feature_count"] = 9
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidates((first, second)),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=2,
        modelsize_byte_cap_feedback_paths=(
            "/Volumes/VertigoDataTier/pact/exports/snerv_feedback.json",
        ),
    )

    snerv_rows = [row for row in report["campaign_rows"] if row["family"] == "snerv"]
    assert len(snerv_rows) == 2
    assert {row["runner_modelsize_candidate_id"] for row in snerv_rows} == {"auto"}
    assert len({row["row_id"] for row in snerv_rows}) == 2
    assert all(row["row_id"].startswith("snerv::auto_bytecap::") for row in snerv_rows)
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(report["experiment_queue"], sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_queue_definition(queue_path)
    experiment_ids = [exp["id"] for exp in loaded["experiments"]]
    assert len(experiment_ids) == len(set(experiment_ids))


def test_long_training_campaign_plan_blocks_snerv_auto_when_calibrated_byte_cap_is_over(
    tmp_path: Path,
) -> None:
    export = tmp_path / "snerv_export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "archive_bytes": 444_036,
                "packet_bytes": 2_347_396,
                "decoder_payload_codec": "int4_symmetric",
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
                "hard_byte_ceiling_measurement_bypass_enabled": True,
                "hard_byte_ceiling_checked_after_export": True,
                "calibrated_archive_overrun_bytes": 266_036,
                "required_nominal_payload_bytes_max": 75_705,
                "modelsize_candidate": {
                    "candidate_id": _snerv_candidate_id(),
                    "family": "snerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 188_854,
                    "decoder_payload_codec": "int4_symmetric",
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(export.as_posix(),),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert snerv["local_mlx_launch_command_ready"] is False
    assert snerv["experiment_queue_entry"]["status"] == "disabled"
    assert (
        "snerv_modelsize_auto_calibrated_byte_cap_over_ceiling"
        in snerv["blockers"]
    )
    preflight = snerv["modelsize_byte_cap_preflight"]
    assert preflight["matching_observation_count"] == 1
    assert preflight["predicted_under_hard_byte_ceiling"] is False
    assert preflight["predicted_archive_bytes"] > 178_000
    assert preflight["matching_calibrated_archive_overrun_bytes_max"] == 266_036
    assert preflight["matching_required_nominal_payload_bytes_max"] == 75_705
    assert preflight["matching_measurement_bypass_observed"] is True
    observation = preflight["matching_observations"][0]
    assert observation["calibrated_archive_overrun_bytes"] == 266_036
    assert observation["required_nominal_payload_bytes_max"] == 75_705
    assert observation["hard_byte_ceiling_measurement_bypass_enabled"] is True
    assert observation["hard_byte_ceiling_checked_after_export"] is True


def test_long_training_campaign_plan_byte_cap_preflight_reads_startup_candidate_fallback(
    tmp_path: Path,
) -> None:
    startup = tmp_path / "startup.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "modelsize_candidate": {
                    "candidate_id": "hinerv_tiny",
                    "family": "hi_nerv",
                    "hard_byte_ceiling": 178_000,
                    "nominal_total_payload_bytes": 95_000,
                    "decoder_codec": "portfolio_auto",
                },
            }
        ),
        encoding="utf-8",
    )
    export = tmp_path / "hinerv_export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "hinerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "archive_bytes": 80_000,
                "decoder_codec": "portfolio_auto",
                "receiver_proof_ready": True,
                "startup_json_path": startup.as_posix(),
            }
        ),
        encoding="utf-8",
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(export.as_posix(),),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")

    assert "hi_nerv_modelsize_byte_cap_feedback_observation_missing" not in hi[
        "blockers"
    ]
    preflight = hi["modelsize_byte_cap_preflight"]
    assert preflight["matching_observation_count"] == 1
    assert preflight["predicted_under_hard_byte_ceiling"] is True


def test_long_training_campaign_plan_promotes_hinerv_candidate_with_byte_feedback_into_limit(
    tmp_path: Path,
) -> None:
    hinerv_budget = _hinerv_budget()
    stale = dict(hinerv_budget["selected_candidates"][0])
    calibrated = {
        **stale,
        "candidate_id": "hinerv_calibrated_bytecap",
        "decoder_codec": "int2_mixed",
        "nominal_total_payload_bytes": 96_000,
        "byte_headroom": 82_000,
    }
    hinerv_budget["selected_candidates"] = [stale]
    export = tmp_path / "hinerv_export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "hinerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": calibrated["candidate_id"],
                "archive_bytes": 101_000,
                "decoder_codec": "int2_mixed",
                "receiver_proof_ready": True,
                "modelsize_candidate": calibrated,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(export.as_posix(),),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")

    assert hi["candidate"]["candidate_id"] == calibrated["candidate_id"]
    assert hi["budget_candidate_id"] == calibrated["candidate_id"]
    assert hi["runner_modelsize_candidate_id"] == "auto"
    assert hi["modelsize_candidate_selection_mode"] == (
        "calibrated_auto_from_modelsize_byte_cap_feedback"
    )
    assert (
        "hi_nerv_modelsize_byte_cap_feedback_observation_missing"
        not in hi["blockers"]
    )
    preflight = hi["modelsize_byte_cap_preflight"]
    assert preflight["matching_observation_count"] == 1
    assert preflight["predicted_archive_bytes"] == 101_000
    assert preflight["predicted_under_hard_byte_ceiling"] is True
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_consumes_snerv_binary_profile_receiver_proof_feedback(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="a" * 64,
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(scalar),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(profile.as_posix(),),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert "snerv_modelsize_byte_cap_feedback_observation_missing" not in snerv[
        "blockers"
    ]
    preflight = snerv["modelsize_byte_cap_preflight"]
    assert preflight["observation_count"] == 1
    assert preflight["matching_observation_count"] == 1
    assert preflight["predicted_archive_bytes"] == 91_445
    assert preflight["predicted_under_hard_byte_ceiling"] is True
    assert preflight["matching_observations"][0]["candidate_id"] == scalar[
        "candidate_id"
    ]
    assert preflight["matching_observations"][0]["receiver_closed"] is True
    assert preflight["score_claim"] is False
    feedback = snerv["candidate_feedback"]
    assert feedback["byte_feedback_source"] == "modelsize_byte_cap_receiver_proof"
    assert feedback["feedback_scope"] == "full600_native_file_backed_snar1_export"
    assert feedback["feedback_ready"] is True
    assert feedback["native_mlx_receiver_proof_passed"] is True
    assert feedback["native_mlx_full600_campaign_ready"] is True
    training_plan = snerv["curriculum_plan"]["training_plan"]
    assert training_plan["native_mlx_train_export_verified"] is True
    assert training_plan["native_mlx_file_backed_export_proof_passed"] is True
    assert "snerv_snar1_byte_feedback_missing" not in snerv["blockers"]
    assert "snerv_mlx_native_receiver_proof_missing_or_failed" not in snerv[
        "blockers"
    ]
    assert "snerv_mlx_native_file_backed_export_proof_missing_or_failed" not in snerv[
        "blockers"
    ]
    assert "snerv_mlx_native_full600_campaign_not_ready" not in snerv["blockers"]


def test_long_training_campaign_plan_preserves_over_ceiling_snerv_feedback_as_demotion_row(
    tmp_path: Path,
) -> None:
    observed = dict(_snerv_budget()["selected_candidates"][0])
    observed.update(
        {
            "candidate_id": "snerv_observed_receiver_proven_over_ceiling",
            "hard_byte_ceiling": 216_000,
            "nominal_total_payload_bytes": 188_854,
            "nominal_under_ceiling": True,
        }
    )
    export = tmp_path / "snerv_checkpoint_archive_export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "candidate_id": observed["candidate_id"],
                "archive_bytes": 444_828,
                "packet_bytes": 2_347_476,
                "packet_path": "/Volumes/VertigoDataTier/pact/snerv/observed_over_ceiling.snar",
                "packet_sha256": "c" * 64,
                "lf_payload_codec": "portfolio_auto",
                "packet_section_bytes": {
                    "metadata_payload": 14_400,
                    "lf_payload": 879_633,
                    "decoder_payload": 1_282,
                    "step_map_packet": 105_591,
                },
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
                "hard_byte_ceiling_measurement_bypass_enabled": True,
                "hard_byte_ceiling_checked_after_export": True,
                "calibrated_archive_overrun_bytes": 228_828,
                "required_nominal_payload_bytes_max": 91_703,
                "modelsize_candidate": observed,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(export.as_posix(),),
    )

    snerv_rows = [row for row in report["campaign_rows"] if row["family"] == "snerv"]
    assert len(snerv_rows) == 2
    selected, demoted = snerv_rows
    assert selected["candidate"]["candidate_id"] != observed["candidate_id"]
    assert selected["candidate"].get("_modelsize_feedback_demote_only") is not True
    selected_preflight = selected["modelsize_byte_cap_preflight"]
    assert selected_preflight["missing_matching_feedback_is_blocking"] is False
    assert "snerv_modelsize_byte_cap_feedback_observation_missing" not in selected[
        "blockers"
    ]
    assert demoted["candidate"]["candidate_id"] == observed["candidate_id"]
    assert demoted["local_mlx_launch_command_ready"] is False
    assert demoted["candidate"]["_modelsize_feedback_demote_only"] is True
    assert demoted["candidate"]["_modelsize_feedback_observed_archive_bytes"] == 444_828
    assert (
        demoted["candidate"]["_modelsize_feedback_archive_over_hard_byte_ceiling_bytes"]
        == 228_828
    )
    assert (
        demoted["candidate"]["_modelsize_feedback_required_nominal_payload_bytes_max"]
        == 91_703
    )
    assert demoted["candidate"]["_modelsize_feedback_measurement_bypass_enabled"] is True
    assert (
        "snerv_receiver_proven_archive_over_hard_byte_ceiling_observed_demote_only"
        in demoted["blockers"]
    )
    assert "snerv_receiver_proven_archive_over_hard_byte_ceiling" in demoted[
        "blockers"
    ]
    launch_contract = demoted["experiment_queue_entry"]["launch_authority_contract"]
    assert (
        "snerv_receiver_proven_archive_over_hard_byte_ceiling_observed_demote_only"
        in launch_contract["queue_launch_blockers"]
    )
    assert launch_contract["queue_status_is_runnable_plan"] is False
    reroute_queue = report["snerv_lf_over_ceiling_reroute_queue"]
    queue_row_ids = [row["queue_row_id"] for row in reroute_queue["queue_rows"]]
    assert len(queue_row_ids) == len(set(queue_row_ids))
    reroute_rows = {
        row["representation_candidate_id"]: row
        for row in reroute_queue["queue_rows"]
        if row["candidate_id"] == observed["candidate_id"]
    }
    assert reroute_queue["schema"] == "snerv_lf_over_ceiling_reroute_queue.v1"
    assert reroute_queue["local_recode_command_row_count"] >= 1
    recode = reroute_rows["snerv_lossless_lf_recode_probe"]
    assert recode["work_order_type"] == "lossless_lf_recode_probe"
    assert recode["blocked"] is False
    assert "tools/recode_snerv_lf_payload_archive.py" in recode["command_argv"]
    assert recode["measured_lf_payload_bytes"] == 879_633
    assert recode["required_lf_savings_bytes"] == 228_828
    assert recode["source_campaign_status"]["local_mlx_launch_command_ready"] is False
    assert observed["candidate_id"] in " ".join(recode["command_argv"])
    temporal_gate = reroute_rows["snerv_lf_temporal_tub_gate_receiver_visible"]
    assert temporal_gate["work_order_type"] == "lf_representation_change_candidate"
    assert temporal_gate["blocked"] is True
    assert temporal_gate["lf_payload_can_cover_required_savings"] is True
    assert (
        "snerv_lf_tub_temporal_gate_not_implemented"
        in temporal_gate["blockers"]
    )
    feedback = demoted["curriculum_plan"]["byte_oracle_logging"]
    assert feedback["archive_under_hard_byte_ceiling"] is False
    assert feedback["archive_over_hard_byte_ceiling_bytes"] == 228_828
    assert feedback["required_nominal_payload_bytes_max"] == 91_703
    assert feedback["calibrated_archive_overrun_bytes"] == 228_828
    assert feedback["hard_byte_ceiling_measurement_bypass_enabled"] is True


def test_long_training_campaign_plan_rejects_snerv_byte_feedback_failed_receiver_contract(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="a" * 64,
    )
    payload = json.loads(profile.read_text(encoding="utf-8"))
    proof_path = (
        Path(payload["input_path"]).parent
        / "receiver_proof"
        / "snerv_inverse_steg_receiver_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["receiver_contract_satisfied"] = False
    proof["blockers"] = ["synthetic_receiver_contract_failed"]
    proof_path.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(scalar),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(profile.as_posix(),),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert snerv["modelsize_byte_cap_preflight"]["observation_count"] == 0
    assert "snerv_modelsize_byte_cap_feedback_observation_missing" in snerv[
        "blockers"
    ]
    assert "snerv_mlx_native_full600_campaign_not_ready" in snerv["blockers"]


def test_long_training_campaign_plan_rejects_contract_only_snerv_byte_feedback(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    export = tmp_path / "snerv_contract_only_export.json"
    export.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "archive_bytes": 91_445,
                "packet_bytes": 188_000,
                "receiver_contract_satisfied": True,
                "modelsize_candidate": scalar,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(scalar),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(export.as_posix(),),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert snerv["modelsize_byte_cap_preflight"]["observation_count"] == 0
    assert "snerv_modelsize_byte_cap_feedback_observation_missing" in snerv[
        "blockers"
    ]


def test_long_training_campaign_plan_refuses_snerv_byte_feedback_from_wrong_skip_mode(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    full = _snerv_official_skip_candidate("full")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="b" * 64,
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(full),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=2,
        modelsize_byte_cap_feedback_paths=(profile.as_posix(),),
    )

    snerv = next(
        row
        for row in report["campaign_rows"]
        if row["family"] == "snerv"
        and row["candidate"]["candidate_id"] == full["candidate_id"]
    )

    assert "snerv_modelsize_byte_cap_feedback_observation_missing" in snerv[
        "blockers"
    ]
    preflight = snerv["modelsize_byte_cap_preflight"]
    assert preflight["observation_count"] == 1
    assert preflight["matching_observation_count"] == 0


def test_long_training_campaign_plan_promotes_snerv_candidate_with_byte_feedback_into_limit(
    tmp_path: Path,
) -> None:
    channel = _snerv_official_skip_candidate("channel_mean")
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="f" * 64,
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(channel),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(profile.as_posix(),),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert snerv["candidate"]["candidate_id"] == scalar["candidate_id"]
    assert "snerv_modelsize_byte_cap_feedback_observation_missing" not in snerv[
        "blockers"
    ]
    assert snerv["modelsize_byte_cap_preflight"]["matching_observation_count"] == 1


def test_long_training_campaign_cli_discovers_snerv_binary_profile_receiver_feedback(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="e" * 64,
    )

    discovered = cli._discover_modelsize_byte_cap_feedback_paths(
        [tmp_path],
        limit=8,
    )

    assert profile.resolve(strict=False) in discovered


def test_long_training_campaign_cli_discovers_candidate_feedback_rows(
    tmp_path: Path,
) -> None:
    feedback = tmp_path / "nested" / "snerv_upstream_eval_candidate_feedback_row.json"
    feedback.parent.mkdir()
    feedback.write_text(
        json.dumps(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "upstream_eval_gate",
                "feedback_scope": "full600_upstream_cpu_eval",
                "family": "snerv",
                "candidate_id": "snerv_upstream_data_only_snsa2",
                "measured_num_pairs": 600,
                "direct_feedback_blockers": ["snerv_upstream_eval_gate_score_bad"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    telemetry_feedback = (
        tmp_path
        / "telemetry"
        / "nerv_candidate_training_telemetry_feedback_row.json"
    )
    telemetry_feedback.parent.mkdir()
    telemetry_feedback.write_text(
        json.dumps(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "feedback_scope": "full600_training_telemetry",
                "family": "snerv",
                "candidate_id": "snerv_scalarmean_hardpair_successor_fix2",
                "measured_num_pairs": 600,
                "degenerate_renderer_risk_detected": True,
                "direct_feedback_blockers": [
                    "snerv_scorer_domain_tether_missing_telemetry"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    hinerv_refresh = (
        tmp_path
        / "hinerv_smoke"
        / "hinerv_smoke_comparison_candidate_feedback_refresh.json"
    )
    hinerv_refresh.parent.mkdir()
    hinerv_refresh.write_text(
        json.dumps(
            {
                "schema": "nerv_queue_training_feedback_refresh.v1",
                "rows": [
                    {
                        "experiment_id": "hinerv_directlive_smoke",
                        "step_id": "embedded_runner_candidate_feedback",
                        "status": "harvested",
                        "family": "hi_nerv",
                        "candidate_id": "hinerv_tiny",
                        "row": {
                            "schema": "nerv_candidate_feedback_row.v1",
                            "feedback_kind": "smoke_comparison_harvest",
                            "family": "hi_nerv",
                            "candidate_id": "hinerv_tiny",
                            "measured_num_pairs": 600,
                            "measured_archive_bytes": 121_000,
                            "score_claim": False,
                            "promotion_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        },
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ignored = tmp_path / "nested" / "not_candidate_feedback_row.json"
    ignored.write_text('{"schema":"other.v1"}', encoding="utf-8")

    discovered = cli._discover_candidate_feedback_paths([tmp_path], limit=8)

    assert feedback.resolve(strict=False) in discovered
    assert telemetry_feedback.resolve(strict=False) in discovered
    assert hinerv_refresh.resolve(strict=False) in discovered
    assert ignored.resolve(strict=False) not in discovered


def test_long_training_campaign_plan_rejects_partial_snerv_binary_profile_byte_feedback(
    tmp_path: Path,
) -> None:
    scalar = _snerv_official_skip_candidate("scalar_mean")
    profile = _write_snerv_binary_profile_receiver_feedback(
        tmp_path,
        candidate=scalar,
        archive_bytes=91_445,
        archive_sha256="1" * 64,
    )
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["snar1_metadata"]["n_pairs"] = 2
    profile.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    assert profile.resolve(strict=False) not in cli._discover_modelsize_byte_cap_feedback_paths(
        [tmp_path],
        limit=8,
    )
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(scalar),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        modelsize_byte_cap_feedback_paths=(profile.as_posix(),),
    )
    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert snerv["modelsize_byte_cap_preflight"]["observation_count"] == 0
    assert "snerv_modelsize_byte_cap_feedback_observation_missing" in snerv[
        "blockers"
    ]


def test_long_training_campaign_plan_consumes_passing_snerv_tether_smoke() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_scorer_tether_smoke_report={
            "schema": "snerv_scorer_tether_smoke.v1",
            "created_utc": "2026-06-05T00:00:00Z",
            "steps": 2,
            "passed": True,
            "metric_summary": {
                "loss_part_distill": 0.5,
                "loss_part_pose_distill": 0.25,
                "snerv_segnet_last_frame_distill_lambda": 0.01,
                "snerv_posenet_yuv6_pair_distill_lambda": 0.01,
            },
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert report["snerv_scorer_tether_smoke_report_attached"] is True
    assert report["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert snerv["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert (
        snerv["score_aware_long_training_plan"]["snerv_scorer_tether_smoke_gate"][
            "passed"
        ]
        is True
    )
    assert "snerv_scorer_tether_smoke_failed" not in snerv["blockers"]


def test_long_training_campaign_plan_blocks_failed_snerv_tether_smoke() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_scorer_tether_smoke_report={
            "schema": "snerv_scorer_tether_smoke.v1",
            "created_utc": "2026-06-05T00:00:00Z",
            "steps": 2,
            "passed": False,
            "metric_summary": {},
            "blockers": ["snerv_scorer_tether_smoke_lambda_inactive"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")

    assert report["snerv_scorer_tether_smoke_gate"]["passed"] is False
    assert "snerv_scorer_tether_smoke_failed" in snerv["blockers"]
    assert "snerv_scorer_tether_smoke_lambda_inactive" in snerv["blockers"]
    assert snerv["experiment_queue_entry"]["status"] == "disabled"
    assert snerv["experiment_queue_entry"]["blocked"] is True
    assert "snerv_scorer_tether_smoke_failed" in snerv[
        "experiment_queue_entry"
    ]["launch_authority_contract"]["queue_launch_blockers"]


def test_long_training_campaign_plan_consumes_candidate_feedback_sources() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "measured_archive_bytes": 111_000,
                "measured_num_pairs": 600,
            },
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "snerv",
                "candidate_id": _snerv_candidate_id(),
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "native_mlx_receiver_proof_passed": True,
                "native_mlx_full600_campaign_ready": True,
                "native_mlx_scorer_loop_qat_receiver_contract_satisfied": True,
                "native_mlx_scorer_loop_qat_ready_for_pose_guard_gate": True,
                "native_mlx_scorer_loop_qat_accepted_improvement": True,
                "native_mlx_scorer_loop_qat_best_materialized": True,
                "measured_payload_bytes": 175_000,
                "measured_archive_bytes": 176_000,
                "measured_num_pairs": 600,
            },
        ),
    )

    assert report["candidate_feedback_source_count"] == 2
    assert report["candidate_feedback_row_count"] == 2
    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    hi_curriculum = hi["curriculum_plan"]
    assert hi_curriculum["byte_oracle_logging"]["feedback_ready"] is True
    assert hi_curriculum["byte_oracle_logging"]["measured_archive_bytes"] == 111_000
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in hi["blockers"]
    assert "hi_nerv_receiver_proof_missing" in hi["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" in hi["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" in hi["blockers"]
    assert "direct_feedback_receiver_proof_file_missing" in hi["candidate_feedback"]["direct_feedback_blockers"]
    assert hi["candidate_feedback"]["measured_num_pairs"] == 600

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    snerv_curriculum = snerv["curriculum_plan"]
    assert snerv_curriculum["byte_oracle_logging"]["feedback_ready"] is True
    assert snerv_curriculum["byte_oracle_logging"]["measured_payload_bytes"] == 175_000
    assert "snerv_snar1_byte_feedback_missing" not in snerv["blockers"]
    assert "snerv_receiver_proof_missing" in snerv["blockers"]
    assert "snerv_full_video_local_prefilter_missing" in snerv["blockers"]
    assert "snerv_local_cpu_replay_gate_missing" in snerv["blockers"]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in snerv["blockers"]
    assert snerv["candidate_feedback"]["measured_archive_bytes"] == 176_000
    assert (
        "direct_feedback_native_receiver_proof_file_missing" in snerv["candidate_feedback"]["direct_feedback_blockers"]
    )
    assert "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only" not in snerv["blockers"]
    assert snerv["execution_epochs"] == 29_650


def test_long_training_campaign_plan_consumes_hinerv_archive_ladder_feedback(
    tmp_path: Path,
) -> None:
    proof = tmp_path / "receiver_proof.json"
    proof.write_text('{"runtime_consumption_proof_ready": true}\n', encoding="utf-8")
    ladder_feedback = build_hinerv_archive_ladder_feedback_report(
        archive_ladder_report={
            "schema": "hinerv_archive_size_ladder.v1",
            "num_pairs": 600,
            "archive_rows": [
                {
                    "row_id": "hinerv_tiny",
                    "archive_bytes": 45_834,
                    "archive_path": "/Volumes/VertigoDataTier/pact/hinerv_tiny/archive.zip",
                    "archive_sha256": "2" * 64,
                    "runtime_consumption_proof_ready": True,
                    "receiver_proof_path": proof.as_posix(),
                    "modelsize_candidate": {"hard_byte_ceiling": 178_000},
                }
            ],
        },
        source_report_path=tmp_path / "archive_ladder.json",
    )

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=tuple(ladder_feedback["rows"]),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_feedback"]["byte_feedback_source"] == "hinerv_archive_size_ladder"
    assert hi["candidate_feedback"]["receiver_proof_attached"] is True
    assert hi["curriculum_plan"]["byte_oracle_logging"]["feedback_ready"] is True
    assert hi["curriculum_plan"]["byte_oracle_logging"]["measured_archive_bytes"] == 45_834
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in hi["blockers"]
    assert "representative_distortion_evidence_missing" in hi["candidate_feedback_evidence_blockers"]
    assert hi["cpu_replay_ready"] is False
    assert hi["exact_gate_ready"] is False
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_consumes_hinerv_feedback_from_full_row_id() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "hi_nerv",
                "candidate_id": "hi_nerv::hinerv_tiny::lion",
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "measured_archive_bytes": 111_000,
                "measured_num_pairs": 600,
            },
        ),
    )

    assert report["candidate_feedback_source_count"] == 1
    assert report["candidate_feedback_row_count"] == 1
    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_feedback"]["candidate_id"] == "hi_nerv::hinerv_tiny::lion"
    assert hi["candidate_feedback"]["candidate_id_match"] is True
    assert "hinerv_trained_archive_byte_oracle_feedback_missing" not in hi["blockers"]
    assert "hi_nerv_receiver_proof_missing" in hi["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" in hi["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" in hi["blockers"]


def test_long_training_campaign_plan_applies_hinerv_pose_instability_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=1.0e-3,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 1.0e-3,
                "recommended_learning_rate": 3.0e-4,
                "recommended_launch_mutations": ["lower_learning_rate_from_pose_instability_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    lr = argv[argv.index("--learning-rate") + 1]
    assert lr == "0.0003"
    output_dir = argv[argv.index("--output-dir") + 1]
    output_name = Path(output_dir).name
    assert output_name.startswith("hi_nerv_hinerv_tiny_adamw_feedback")
    assert "pose_instability" in output_name
    assert "lr0.0003" in output_name
    assert "lower_learning_rate_from_pose_instability" in output_name
    assert hi["output_dir_basename"] == output_name
    assert hi["output_dir_reuse_policy"] == "fresh_feedback_mutation_path"
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["schema"] == "hinerv_feedback_launch_adjustment.v1"
    assert adjustment["applied"] is True
    assert adjustment["requested_learning_rate"] == 1.0e-3
    assert adjustment["learning_rate"] == 3.0e-4
    assert "lower_learning_rate_from_pose_instability_telemetry" in adjustment["launch_mutations"]
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert hi["candidate_feedback"]["feedback_kind"] == "training_telemetry"
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_applies_hinerv_lr9e5_recovery_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=9.0e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 9.0e-5,
                "recommended_learning_rate": 2.7e-5,
                "recommended_launch_mutations": ["lower_learning_rate_from_pose_instability_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    adjustment = hi["feedback_launch_adjustment"]
    assert HINERV_POSE_INSTABILITY_LOW_LR_FLOOR == 3.0e-5
    assert adjustment["applied"] is True
    assert adjustment["lower_learning_rate_applied"] is True
    assert adjustment["pose_protected_pathway_applied"] is False
    assert adjustment["repeated_low_lr_pose_instability"] is False
    assert adjustment["learning_rate"] == 2.7e-5
    assert adjustment["pose_distillation_loss"] == "mse"
    assert adjustment["pose_distillation_huber_delta"] is None
    assert adjustment["reason"] == ("pose_instability_recommended_lower_learning_rate")
    assert "above low_learning_rate_floor applies" in adjustment["policy_logic"]
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert "hinerv_repeated_low_lr_pose_instability_requires_pose_protected_pathway" not in hi["blockers"]
    assert hi["command_argv"][hi["command_argv"].index("--learning-rate") + 1] == ("2.7e-05")
    assert "--pose-distillation-loss" not in hi["command_argv"]
    assert "--pose-distillation-huber-delta" not in hi["command_argv"]


def test_long_training_campaign_plan_applies_hinerv_family_pose_instability_feedback() -> None:
    hinerv_budget = _hinerv_budget()
    sibling = dict(hinerv_budget["selected_candidates"][0])
    sibling["candidate_id"] = "hinerv_sibling"
    sibling["decoder_codec"] = "portfolio_auto"
    hinerv_budget["selected_candidates"] = [sibling]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=9.0e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_previous_full600",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "measured_archive_bytes": 111_000,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 9.0e-5,
                "recommended_learning_rate": 2.7e-5,
                "recommended_launch_mutations": ["lower_learning_rate_from_pose_instability_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    feedback = hi["candidate_feedback"]
    assert feedback["feedback_match_scope"] == "family_training_telemetry"
    assert feedback["candidate_id_match"] is False
    assert feedback["source_candidate_id"] == "hinerv_previous_full600"
    assert feedback["target_candidate_id"] == "hinerv_sibling"
    assert feedback["receiver_proof_attached"] is False
    assert feedback["full_video_local_prefilter_attached"] is False
    assert feedback["local_cpu_replay_gate_attached"] is False
    assert feedback["measured_archive_bytes"] is None
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["learning_rate"] == 2.7e-5
    assert hi["command_argv"][hi["command_argv"].index("--learning-rate") + 1] == ("2.7e-05")
    assert hi["curriculum_plan"]["byte_oracle_logging"]["feedback_ready"] is False
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert "hi_nerv_receiver_proof_missing" in hi["blockers"]


def test_long_training_campaign_plan_applies_hinerv_segnet_stagnation_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": False,
                "seg_stagnation_detected": True,
                "observed_learning_rate": 2.7e-5,
                "recommended_segnet_distillation_weight": 2.0,
                "recommended_launch_mutations": ["increase_segnet_distillation_weight_from_stagnation_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["segnet_weight_applied"] is True
    assert adjustment["segnet_distillation_weight"] == 2.0
    assert adjustment["reason"] == ("segnet_stagnation_recommended_higher_segnet_weight")
    assert argv[argv.index("--segnet-distillation-weight") + 1] == "2"
    assert hi["curriculum_plan"]["scorer_pressure"]["segnet_distillation_weight"] == 2.0
    output_name = Path(argv[argv.index("--output-dir") + 1]).name
    assert "seg_stagnation" in output_name
    assert "segw2" in output_name
    assert "increase_segnet_distillation_weight_from_stagnation_telemetry" in (adjustment["launch_mutations"])
    assert hi["output_dir_reuse_policy"] == "fresh_feedback_mutation_path"


def test_long_training_campaign_plan_applies_full_video_mlx_response_feedback(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "hinerv_full_video_mlx_response.json"
    receiver_proof_path = tmp_path / "hinerv_receiver_proof.json"
    local_replay_path = tmp_path / "hinerv_local_cpu_replay_gate.json"
    response_path.write_text('{"schema":"mlx_scorer_response.v1"}', encoding="utf-8")
    receiver_proof_path.write_text('{"schema":"receiver_proof.v1"}', encoding="utf-8")
    local_replay_path.write_text('{"schema":"local_cpu_replay_gate.v1"}', encoding="utf-8")
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "full_video_mlx_scorer_response",
                "full_video_mlx_feedback_schema": (
                    "nerv_full_video_mlx_scorer_feedback.v1"
                ),
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_mlx_scorer_response",
                "scope_matches_candidate": True,
                "feedback_ready": True,
                "launch_control_feedback_ready": True,
                "receiver_proof_attached": True,
                "receiver_proof_path": receiver_proof_path.as_posix(),
                "receiver_proof_sha256": sha256(receiver_proof_path.read_bytes()).hexdigest(),
                "full_video_local_prefilter_attached": True,
                "full_video_mlx_response_attached": True,
                "full_video_mlx_response_path": response_path.as_posix(),
                "full_video_mlx_response_sha256": sha256(response_path.read_bytes()).hexdigest(),
                "local_cpu_replay_gate_attached": True,
                "local_cpu_replay_gate_path": local_replay_path.as_posix(),
                "local_cpu_replay_gate_sha256": sha256(local_replay_path.read_bytes()).hexdigest(),
                "measured_archive_bytes": 122_074,
                "hard_byte_ceiling": 178_000,
                "seg_stagnation_detected": True,
                "pose_instability_detected": True,
                "pose_tail_burst_detected": True,
                "observed_segnet_distillation_weight": 2.0,
                "recommended_segnet_distillation_weight": 4.0,
                "observed_pose_distillation_weight": 1.0,
                "recommended_pose_distillation_weight": 8.0,
                "recommended_launch_mutations": [
                    "increase_segnet_distillation_weight_from_full_video_mlx_response",
                    "increase_pose_distillation_weight_from_full_video_mlx_response",
                    "treat_previous_hi_nerv_run_as_fit_failure_not_rate_negative",
                ],
                "full_video_mlx_scorer_response": {
                    "score_recomputed_from_components": 91.57,
                    "avg_segnet_dist": 0.55,
                    "avg_posenet_dist": 132.0,
                    "archive_under_hard_byte_ceiling": True,
                    "score_claim": False,
                },
                "direct_feedback_blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    adjustment = hi["feedback_launch_adjustment"]
    assert hi["candidate_feedback"]["feedback_kind"] == "full_video_mlx_scorer_response"
    assert hi["candidate_feedback"]["full_video_mlx_response_attached"] is True
    assert hi["candidate_feedback"]["receiver_proof_attached"] is True
    assert hi["candidate_feedback"]["full_video_local_prefilter_attached"] is True
    assert hi["candidate_feedback"]["local_cpu_replay_gate_attached"] is True
    assert hi["candidate_feedback"]["direct_feedback_blockers"] == []
    assert "hi_nerv_receiver_proof_missing" not in hi["blockers"]
    assert "hi_nerv_full_video_local_prefilter_missing" not in hi["blockers"]
    assert "hi_nerv_local_cpu_replay_gate_missing" not in hi["blockers"]
    assert adjustment["launch_control_feedback_ready"] is True
    assert adjustment["applied"] is True
    assert adjustment["segnet_weight_applied"] is True
    assert adjustment["pose_weight_applied"] is True
    assert adjustment["segnet_distillation_weight"] == 4.0
    assert adjustment["pose_distillation_weight"] == 8.0
    assert argv[argv.index("--segnet-distillation-weight") + 1] == "4"
    assert argv[argv.index("--pose-distillation-weight") + 1] == "8"
    assert "increase_segnet_distillation_weight_from_full_video_mlx_response" in (
        adjustment["launch_mutations"]
    )
    assert "increase_pose_distillation_weight_from_full_video_mlx_response" in (
        adjustment["launch_mutations"]
    )
    assert "hinerv_segnet_stagnation_feedback_unapplied" not in hi["blockers"]
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_reuses_snerv_full_video_rate_failure_as_context_only(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "snerv_full_video_mlx_response.json"
    receiver_proof_path = tmp_path / "snerv_receiver_proof.json"
    response_path.write_text('{"schema":"mlx_scorer_response.v1"}', encoding="utf-8")
    receiver_proof_path.write_text('{"schema":"receiver_proof.v1"}', encoding="utf-8")
    snerv_budget = _snerv_budget()
    sibling = dict(snerv_budget["selected_candidates"][0])
    sibling["candidate_id"] = (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc9e2_p1_mfu1-2-4_hfr0_t0_adbase_int4_symmetric_ceil178000"
    )
    sibling["fc_dim"] = 9
    snerv_budget["selected_candidates"] = [sibling]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=snerv_budget,
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "full_video_mlx_scorer_response",
                "full_video_mlx_feedback_schema": (
                    "nerv_full_video_mlx_scorer_feedback.v1"
                ),
                "family": "snerv",
                "candidate_id": (
                    "snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc36e0_p1_mfu1-2-4_hfr0_t1_adbase_oms0p285_int8_symmetric_ceil216000"
                ),
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_mlx_scorer_response",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "launch_control_feedback_ready": False,
                "receiver_proof_attached": True,
                "receiver_proof_path": receiver_proof_path.as_posix(),
                "receiver_proof_sha256": sha256(receiver_proof_path.read_bytes()).hexdigest(),
                "full_video_local_prefilter_attached": True,
                "full_video_mlx_response_attached": True,
                "full_video_mlx_response_path": response_path.as_posix(),
                "full_video_mlx_response_sha256": sha256(response_path.read_bytes()).hexdigest(),
                "local_cpu_replay_gate_attached": False,
                "measured_archive_bytes": 444_036,
                "hard_byte_ceiling": 178_000,
                "recommended_launch_mutations": [
                    "treat_previous_snerv_run_as_rate_failure_not_distortion_negative",
                    "switch_snerv_representation_before_more_same_modelsize_training",
                ],
                "direct_feedback_blockers": [
                    "snerv_full_video_mlx_response_archive_over_hard_byte_ceiling"
                ],
                "full_video_mlx_scorer_response": {
                    "score_recomputed_from_components": 108.61,
                    "avg_segnet_dist": 0.68,
                    "avg_posenet_dist": 162.84,
                    "archive_under_hard_byte_ceiling": False,
                    "score_claim": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["feedback_match_scope"] == "family_full_video_mlx_response_context"
    assert feedback["candidate_id_match"] is False
    assert feedback["source_candidate_id"].startswith("snerv_np600_haar_lv5")
    assert feedback["target_candidate_id"] == sibling["candidate_id"]
    assert feedback["context_only"] is True
    assert feedback["receiver_proof_attached"] is False
    assert feedback["full_video_local_prefilter_attached"] is False
    assert feedback["local_cpu_replay_gate_attached"] is False
    assert feedback["measured_archive_bytes"] is None
    assert feedback["measured_payload_bytes"] is None
    assert feedback["feedback_ready"] is False
    assert feedback["direct_feedback_blockers"] == [
        "snerv_full_video_mlx_response_archive_over_hard_byte_ceiling"
    ]
    assert feedback["score_claim"] is False
    assert (
        feedback["feedback_reuse_policy"]
        == "family_full_video_context_only_no_archive_receiver_replay_or_launch_authority"
    )
    assert "snerv_full_video_mlx_response_archive_over_hard_byte_ceiling" in (
        snerv["candidate_feedback_evidence_blockers"]
    )
    assert "treat_previous_snerv_run_as_rate_failure_not_distortion_negative" in (
        snerv["candidate_feedback"]["recommended_launch_mutations"]
    )
    assert snerv["score_claim"] is False


def test_long_training_campaign_plan_reuses_snerv_upstream_eval_gate_as_context_only() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "upstream_eval_gate",
                "feedback_scope": "full600_upstream_cpu_eval",
                "family": "snerv",
                "candidate_id": "snerv_upstream_data_only_snsa2",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "scope_matches_candidate": False,
                "context_only": True,
                "feedback_ready": False,
                "launch_control_feedback_ready": False,
                "receiver_proof_attached": False,
                "full_video_local_prefilter_attached": False,
                "local_cpu_replay_gate_attached": False,
                "measured_archive_bytes": 51_694,
                "upstream_eval_score": 90.61,
                "upstream_eval_pose": 162.09104919,
                "upstream_eval_seg": 0.50314105,
                "upstream_eval_rate": 0.00137684,
                "recommended_launch_mutations": [
                    "block_snerv_data_only_archive_as_launch_candidate_due_to_scorer_quality",
                    "require_snerv_representation_change_before_more_same_long_training",
                ],
                "direct_feedback_blockers": [
                    "snerv_upstream_eval_gate_score_bad",
                    "paired_contest_cpu_cuda_auth_eval_missing",
                    "pre_submission_compliance_gate_missing",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["feedback_match_scope"] == "family_upstream_eval_gate_context"
    assert feedback["candidate_id_match"] is False
    assert feedback["source_candidate_id"] == "snerv_upstream_data_only_snsa2"
    assert feedback["target_candidate_id"] == snerv["candidate_id"]
    assert feedback["context_only"] is True
    assert feedback["receiver_proof_attached"] is False
    assert feedback["full_video_local_prefilter_attached"] is False
    assert feedback["local_cpu_replay_gate_attached"] is False
    assert feedback["measured_archive_bytes"] is None
    assert feedback["measured_payload_bytes"] is None
    assert feedback["feedback_ready"] is False
    assert feedback["launch_control_feedback_ready"] is False
    assert (
        feedback["feedback_reuse_policy"]
        == "family_upstream_eval_context_only_no_archive_receiver_replay_or_launch_authority"
    )
    assert "snerv_upstream_eval_gate_score_bad" in (
        snerv["candidate_feedback_evidence_blockers"]
    )
    assert "require_snerv_representation_change_before_more_same_long_training" in (
        snerv["candidate_feedback"]["recommended_launch_mutations"]
    )
    assert snerv["score_claim"] is False
    assert snerv["experiment_queue_entry"]["status"] == "disabled"
    assert snerv["experiment_queue_entry"]["score_claim"] is False


def test_long_training_campaign_plan_blocks_snerv_degenerate_renderer_context() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "feedback_scope": "full600_training_telemetry",
                "family": "snerv",
                "candidate_id": "snerv_scalarmean_hardpair_successor_fix2",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "launch_control_feedback_ready": False,
                "receiver_proof_attached": False,
                "full_video_local_prefilter_attached": False,
                "local_cpu_replay_gate_attached": False,
                "training_stopped": True,
                "training_telemetry": {
                    "last_epoch": 29_649,
                    "row_count": 29_650,
                },
                "degenerate_renderer_risk_detected": True,
                "recommended_launch_mutations": [
                    "bind_snerv_posenet_yuv6_and_segnet_last_frame_distill_metrics_before_more_long_training",
                    "reject_snerv_degenerate_renderer_even_when_archive_bytes_are_frontier",
                    "preserve_snerv_snar2_snsa2_byte_layout_while_rebinding_scorer_tethers",
                ],
                "direct_feedback_blockers": [
                    "snerv_scorer_domain_tether_missing_telemetry",
                    "snerv_posenet_yuv6_pair_distill_metric_missing_telemetry",
                    "snerv_segnet_last_frame_distill_metric_missing_telemetry",
                    "snerv_scorer_domain_tether_lambda_inactive_telemetry",
                    "snerv_score_aware_long_training_dual_segnet_lambda_never_active",
                    "snerv_score_aware_long_training_dual_posenet_lambda_never_active",
                    "snerv_score_aware_long_training_telemetry_contract_failed",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["feedback_match_scope"] == (
        "family_snerv_degenerate_renderer_training_telemetry_context"
    )
    assert feedback["candidate_id_match"] is False
    assert feedback["source_candidate_id"] == "snerv_scalarmean_hardpair_successor_fix2"
    assert feedback["target_candidate_id"] == snerv["candidate_id"]
    assert feedback["context_only"] is True
    assert feedback["receiver_proof_attached"] is False
    assert feedback["full_video_local_prefilter_attached"] is False
    assert feedback["local_cpu_replay_gate_attached"] is False
    assert feedback["measured_archive_bytes"] is None
    assert feedback["measured_payload_bytes"] is None
    assert feedback["feedback_ready"] is False
    assert (
        feedback["feedback_reuse_policy"]
        == "family_snerv_degenerate_renderer_context_only_no_archive_receiver_replay_or_launch_authority"
    )
    assert "snerv_scorer_domain_tether_missing_telemetry" in (
        snerv["candidate_feedback_evidence_blockers"]
    )
    assert "snerv_scorer_domain_tether_missing_telemetry" in snerv["blockers"]
    queue_contract = snerv["experiment_queue_entry"]["launch_authority_contract"]
    assert "snerv_scorer_domain_tether_missing_telemetry" in (
        queue_contract["queue_launch_blockers"]
    )
    assert "snerv_score_aware_long_training_dual_segnet_lambda_never_active" in (
        queue_contract["queue_launch_blockers"]
    )
    assert "snerv_score_aware_long_training_telemetry_contract_failed" in (
        queue_contract["queue_launch_blockers"]
    )
    assert snerv["experiment_queue_entry"]["status"] == "disabled"
    assert snerv["score_claim"] is False


def test_long_training_campaign_plan_tether_smoke_clears_stale_tether_blockers() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_scorer_tether_smoke_report={
            "schema": "snerv_scorer_tether_smoke.v1",
            "created_utc": "2026-06-05T00:51:03Z",
            "steps": 2,
            "passed": True,
            "blockers": [],
            "metric_summary": {
                "final": {
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 1.0,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 1.0,
                },
                "step_count": 2,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "feedback_scope": "full600_training_telemetry",
                "family": "snerv",
                "candidate_id": "snerv_scalarmean_hardpair_successor_fix2",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "launch_control_feedback_ready": False,
                "receiver_proof_attached": False,
                "full_video_local_prefilter_attached": False,
                "local_cpu_replay_gate_attached": False,
                "degenerate_renderer_risk_detected": True,
                "direct_feedback_blockers": [
                    "snerv_scorer_domain_tether_missing_telemetry",
                    "snerv_posenet_yuv6_pair_distill_metric_missing_telemetry",
                    "snerv_segnet_last_frame_distill_metric_missing_telemetry",
                    "snerv_scorer_domain_tether_lambda_inactive_telemetry",
                    "snerv_score_aware_long_training_dual_segnet_lambda_never_active",
                    "snerv_score_aware_long_training_dual_posenet_lambda_never_active",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    assert snerv["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert "snerv_scorer_domain_tether_missing_telemetry" in (
        snerv["candidate_feedback_evidence_blockers_before_tether_smoke"]
    )
    assert "snerv_scorer_domain_tether_missing_telemetry" in (
        snerv["snerv_scorer_tether_smoke_suppressed_feedback_blockers"]
    )
    assert "snerv_scorer_domain_tether_missing_telemetry" not in (
        snerv["candidate_feedback_evidence_blockers"]
    )
    queue_contract = snerv["experiment_queue_entry"]["launch_authority_contract"]
    assert "snerv_scorer_domain_tether_missing_telemetry" not in (
        queue_contract["queue_launch_blockers"]
    )
    assert "snerv_scorer_domain_tether_lambda_inactive_telemetry" not in (
        queue_contract["queue_launch_blockers"]
    )
    assert snerv["score_claim"] is False


def test_long_training_campaign_plan_requires_snerv_renderer_nondegenerate_proof() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_scorer_tether_smoke_report=_passing_snerv_tether_smoke_report(),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    gate = snerv["snerv_renderer_nondegenerate_gate"]

    assert gate["required"] is True
    assert gate["proof_attached"] is False
    assert gate["passed"] is False
    assert "snerv_renderer_nondegenerate_smoke_missing" in gate["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_min16_pairs_missing" in gate["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_missing" in snerv["blockers"]
    queue_contract = snerv["experiment_queue_entry"]["launch_authority_contract"]
    assert "snerv_renderer_nondegenerate_smoke_missing" in (
        queue_contract["queue_launch_blockers"]
    )
    assert snerv["score_claim"] is False


def test_long_training_campaign_plan_consumes_passing_snerv_renderer_proof() -> None:
    candidate = dict(_snerv_budget()["selected_candidates"][0])
    candidate.update(
        {
            "nominal_total_payload_bytes": 120_000,
            "nominal_under_ceiling": True,
        }
    )
    proof = {
        "schema": "snerv_renderer_nondegenerate_proof.v1",
        "min_pair_count": 16,
        "measured_num_pairs": 16,
        "scorer_tether_gate_passed": True,
        "telemetry_contract_passed": True,
        "receiver_reconstruction_verified": True,
        "target_value_domain_passed": True,
        "export_value_domain_passed": True,
        "official_skip_high_value_domain_passed": True,
        "passed": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget_with_candidate(candidate),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        snerv_scorer_tether_smoke_report=_passing_snerv_tether_smoke_report(),
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "snerv",
                "candidate_id": candidate["candidate_id"],
                "candidate_num_pairs": 600,
                "measured_num_pairs": 16,
                "scope_matches_candidate": True,
                "snerv_renderer_nondegenerate_proof": proof,
                "snerv_renderer_nondegenerate_proof_passed": True,
                "snerv_renderer_nondegenerate_blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    gate = snerv["snerv_renderer_nondegenerate_gate"]

    assert gate["required"] is True
    assert gate["proof_attached"] is True
    assert gate["proof_passed"] is True
    assert gate["passed"] is True
    assert gate["measured_num_pairs"] == 16
    assert gate["blockers"] == []
    assert "snerv_renderer_nondegenerate_smoke_missing" not in snerv["blockers"]
    assert "snerv_renderer_nondegenerate_smoke_failed" not in snerv["blockers"]
    queue_contract = snerv["experiment_queue_entry"]["launch_authority_contract"]
    assert "snerv_renderer_nondegenerate_smoke_missing" not in (
        queue_contract["queue_launch_blockers"]
    )
    assert snerv["score_claim"] is False


def test_long_training_campaign_plan_refuses_not_ready_hinerv_launch_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "partial_advisory",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 16,
                "feedback_scope": "partial_pair_advisory",
                "scope_matches_candidate": False,
                "feedback_ready": False,
                "launch_control_feedback_ready": False,
                "pose_instability_detected": False,
                "seg_stagnation_detected": True,
                "recommended_segnet_distillation_weight": 8.0,
                "recommended_launch_mutations": ["increase_segnet_distillation_weight_from_stagnation_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    argv = hi["command_argv"]
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is False
    assert adjustment["reason"] == "feedback_not_launch_control_ready"
    assert adjustment["feedback_ready"] is False
    assert adjustment["launch_control_feedback_ready"] is False
    assert argv[argv.index("--segnet-distillation-weight") + 1] == "1"
    assert hi["curriculum_plan"]["scorer_pressure"]["segnet_distillation_weight"] == 1.0


def test_long_training_campaign_plan_prefers_newer_running_telemetry_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "training_stopped": True,
                "pose_instability_detected": True,
                "observed_learning_rate": 2.7e-5,
                "recommended_learning_rate": 8.1e-6,
                "training_telemetry": {"last_epoch": 560},
                "recommended_launch_mutations": ["lower_learning_rate_from_pose_instability_telemetry"],
            },
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "training_stopped": False,
                "pose_instability_detected": False,
                "pose_instability_recovered": True,
                "seg_stagnation_detected": True,
                "recommended_segnet_distillation_weight": 2.0,
                "training_telemetry": {"last_epoch": 938},
                "recommended_launch_mutations": ["increase_segnet_distillation_weight_from_stagnation_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    feedback = hi["candidate_feedback"]
    adjustment = hi["feedback_launch_adjustment"]
    assert feedback["training_stopped"] is False
    assert feedback["training_telemetry"]["last_epoch"] == 938
    assert adjustment["segnet_weight_applied"] is True
    assert adjustment["pose_protected_pathway_applied"] is False
    assert adjustment["reason"] == ("segnet_stagnation_recommended_higher_segnet_weight")


def test_long_training_campaign_plan_reuses_family_segnet_stagnation_feedback() -> None:
    hinerv_budget = _hinerv_budget()
    sibling = dict(hinerv_budget["selected_candidates"][0])
    sibling["candidate_id"] = "hinerv_sibling_seg"
    hinerv_budget["selected_candidates"] = [sibling]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_previous_full600_seg",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "receiver_proof_attached": True,
                "full_video_local_prefilter_attached": True,
                "local_cpu_replay_gate_attached": True,
                "measured_archive_bytes": 111_000,
                "feedback_ready": False,
                "pose_instability_detected": False,
                "seg_stagnation_detected": True,
                "recommended_segnet_distillation_weight": 2.0,
                "recommended_launch_mutations": ["increase_segnet_distillation_weight_from_stagnation_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    feedback = hi["candidate_feedback"]
    adjustment = hi["feedback_launch_adjustment"]
    assert feedback["feedback_match_scope"] == "family_training_telemetry"
    assert feedback["candidate_id_match"] is False
    assert feedback["source_candidate_id"] == "hinerv_previous_full600_seg"
    assert feedback["target_candidate_id"] == "hinerv_sibling_seg"
    assert feedback["receiver_proof_attached"] is False
    assert feedback["full_video_local_prefilter_attached"] is False
    assert feedback["local_cpu_replay_gate_attached"] is False
    assert feedback["measured_archive_bytes"] is None
    assert adjustment["applied"] is True
    assert adjustment["segnet_weight_applied"] is True
    assert adjustment["segnet_distillation_weight"] == 2.0
    assert hi["command_argv"][hi["command_argv"].index("--segnet-distillation-weight") + 1] == "2"
    assert "hinerv_segnet_stagnation_feedback_unapplied" not in hi["blockers"]
    assert "hi_nerv_receiver_proof_missing" in hi["blockers"]


def test_long_training_campaign_plan_applies_hinerv_receiver_class_survival_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=128,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "candidate_full_scope",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "launch_control_feedback_ready": True,
                "post_export_receiver_class_collapse_detected": True,
                "post_export_receiver_cache_quality_gate_passed": False,
                "post_export_receiver_segnet_candidate_occupied_class_fraction": 0.4,
                "direct_feedback_blockers": [
                    "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
                ],
                "recommended_launch_mutations": [
                    "increase_hi_nerv_receiver_class_survival_pressure",
                    (
                        "disable_hi_nerv_byte_feedback_learning_from_"
                        "receiver_collapsed_export"
                    ),
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["receiver_class_survival_applied"] is True
    assert adjustment["reason"] == "receiver_class_survival_probe_mutation_applied"
    assert "increase_hi_nerv_receiver_class_survival_pressure" in adjustment[
        "launch_mutations"
    ]
    assert (
        "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
        in hi["candidate_feedback_evidence_blockers"]
    )
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_preserves_nonlaunch_hinerv_family_telemetry_context() -> None:
    hinerv_budget = _hinerv_budget()
    sibling = dict(hinerv_budget["selected_candidates"][0])
    sibling["candidate_id"] = "hinerv_np600_ld16_ed24_dc6_int4_mixed_ceil178000_tgtmp0p05"
    hinerv_budget["selected_candidates"] = [sibling]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "training_stopped": False,
                "training_control_action": "continue_running",
                "training_control_reason": "no_live_training_replan_trigger",
                "pose_instability_detected": False,
                "pose_instability_ever_detected": True,
                "pose_instability_recovered": True,
                "seg_stagnation_detected": False,
                "seg_stagnation_relative_improvement": 0.0944,
                "observed_learning_rate": 2.7e-5,
                "observed_segnet_distillation_weight": 4.0,
                "recommended_learning_rate": None,
                "recommended_segnet_distillation_weight": None,
                "training_telemetry": {"last_epoch": 19_781, "row_count": 19_782},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_feedback"] is None
    context = hi["family_training_telemetry_context"]
    assert context["context_only"] is True
    assert context["candidate_id_match"] is False
    assert context["feedback_match_scope"] == "family_training_telemetry_context"
    assert context["source_candidate_id"] == "hinerv_np600_ld28_ed12_dc32_hfg_cnx_int4_mixed_ceil285000"
    assert context["target_candidate_id"] == sibling["candidate_id"]
    assert context["training_control_action"] == "continue_running"
    assert context["pose_instability_recovered"] is True
    assert context["seg_stagnation_detected"] is False
    assert context["receiver_proof_attached"] is False
    assert context["full_video_local_prefilter_attached"] is False
    assert context["local_cpu_replay_gate_attached"] is False
    assert context["launch_control_feedback_ready"] is False
    assert hi["feedback_launch_adjustment"]["applied"] is False
    assert hi["feedback_launch_adjustment"]["reason"] == "no_candidate_feedback"
    argv = hi["command_argv"]
    assert argv[argv.index("--learning-rate") + 1] == "2.7e-05"
    assert argv[argv.index("--segnet-distillation-weight") + 1] == "1"
    queue_metadata = hi["experiment_queue_entry"]["metadata"]
    assert queue_metadata["family_training_telemetry_context"]["context_only"] is True
    assert queue_metadata["family_training_telemetry_context"]["score_claim"] is False


def test_long_training_campaign_plan_prefers_official_hinerv_controls_after_stagnation() -> None:
    hinerv_budget = _hinerv_budget()
    generic = dict(hinerv_budget["selected_candidates"][0])
    generic.update(
        {
            "candidate_id": "hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000",
            "decoder_codec": "int8_mixed",
            "nominal_total_payload_bytes": 90_000,
            "byte_headroom": 88_000,
            "use_hierarchical_feature_grid": False,
            "use_convnext_blocks": False,
        }
    )
    official = dict(generic)
    official.update(
        {
            "candidate_id": "hinerv_np600_ld4_ed16_dc8_hfg_cnx_int2_mixed_ceil36000",
            "decoder_codec": "int2_mixed",
            "embed_dim": 16,
            "nominal_total_payload_bytes": 110_000,
            "byte_headroom": 68_000,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "local_grid_levels": 2,
            "local_grid_channels": 4,
            "convnext_mlp_ratio": 2,
            "convnext_kernel_size": 3,
        }
    )
    hinerv_budget["selected_candidates"] = [generic, official]

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": generic["candidate_id"],
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": False,
                "seg_stagnation_detected": True,
                "observed_learning_rate": 2.7e-5,
                "recommended_segnet_distillation_weight": 2.0,
                "recommended_launch_mutations": ["increase_segnet_distillation_weight_from_stagnation_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_id"] == official["candidate_id"]
    feedback = hi["candidate_feedback"]
    assert feedback["feedback_match_scope"] == "family_training_telemetry"
    assert feedback["source_candidate_id"] == generic["candidate_id"]
    assert feedback["target_candidate_id"] == official["candidate_id"]
    assert feedback["source_official_control_score"] == 0
    assert feedback["target_official_control_score"] == 2
    assert feedback["source_official_control_superseded"] is True

    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["segnet_weight_applied"] is True
    assert adjustment["official_control_superseded"] is True
    assert "switch_to_hinerv_official_feature_grid_convnext_controls" in adjustment["launch_mutations"]
    assert hi["command_argv"][hi["command_argv"].index("--modelsize-candidate-id") + 1] == official["candidate_id"]
    assert hi["command_argv"][hi["command_argv"].index("--segnet-distillation-weight") + 1] == "2"
    assert hi["source_faithfulness_controls"]["source_official_control_superseded"] is True
    metadata = hi["experiment_queue_entry"]["metadata"]
    assert metadata["source_faithfulness_controls"]["target_official_control_score"] == 2
    assert metadata["feedback_launch_adjustment"]["official_control_superseded"] is True
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_consumes_hinerv_foreground_feedback_schema() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "hinerv_training_telemetry_feedback.v1",
                "source_kind": "foreground_official_controls_proof",
                "candidate_id": "hinerv_previous_official",
                "telemetry_path": ("/Volumes/VertigoDataTier/pact/test/telemetry.jsonl"),
                "row_count": 128,
                "last_epoch": 127,
                "first_pose_axis": 62_414.0,
                "last_pose_axis": 5.51,
                "first_seg_axis": 6.36,
                "last_seg_axis": 6.21,
                "learning_rate": 2.7e-5,
                "observed_segnet_distillation_weight": 2.0,
                "pose_recovered_from_initial_spike": True,
                "segnet_still_binding": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    feedback = hi["candidate_feedback"]
    assert feedback["schema"] == "nerv_candidate_feedback_row.v1"
    assert feedback["telemetry_feedback_schema"] == ("hinerv_training_telemetry_feedback.v1")
    assert feedback["feedback_match_scope"] == "family_training_telemetry"
    assert feedback["segnet_still_binding"] is True
    assert feedback["observed_segnet_distillation_weight"] == 2.0
    assert feedback["recommended_segnet_distillation_weight"] == 4.0
    assert hi["command_argv"][hi["command_argv"].index("--segnet-distillation-weight") + 1] == "4"
    assert hi["feedback_launch_adjustment"]["segnet_weight_applied"] is True
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_keeps_foreground_pose_recovery_nonlaunch() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "hinerv_training_telemetry_feedback.v1",
                "source_kind": "foreground_official_controls_proof",
                "candidate_id": "hinerv_tiny",
                "telemetry_path": ("/Volumes/VertigoDataTier/pact/test/telemetry.jsonl"),
                "row_count": 128,
                "last_epoch": 127,
                "first_pose_axis": 62_414.0,
                "last_pose_axis": 5.51,
                "first_seg_axis": 6.36,
                "last_seg_axis": 6.35,
                "learning_rate": 2.7e-5,
                "observed_segnet_distillation_weight": 1.0,
                "pose_recovered_from_initial_spike": True,
                "segnet_still_binding": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    feedback = hi["candidate_feedback"]
    assert feedback["feedback_match_scope"] == "candidate"
    assert feedback["pose_recovered_from_initial_spike"] is True
    assert feedback["launch_control_feedback_ready"] is False
    assert hi["command_argv"][hi["command_argv"].index("--learning-rate") + 1] == "2.7e-05"
    assert hi["command_argv"][hi["command_argv"].index("--segnet-distillation-weight") + 1] == "1"
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is False
    assert adjustment["reason"] == "feedback_not_launch_control_ready"
    assert adjustment["launch_control_feedback_ready"] is False
    assert hi["output_dir_reuse_policy"] == "stable_candidate_optimizer_path"
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert "hinerv_segnet_stagnation_feedback_unapplied" not in hi["blockers"]


def test_long_training_campaign_plan_blocks_repeated_low_lr_pose_instability() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "pose_instability_detected": True,
                "observed_learning_rate": 2.7e-5,
                "recommended_learning_rate": 8.1e-6,
                "recommended_launch_mutations": ["lower_learning_rate_from_pose_instability_telemetry"],
            },
        ),
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    adjustment = hi["feedback_launch_adjustment"]
    assert adjustment["applied"] is True
    assert adjustment["pose_protected_pathway_applied"] is True
    assert adjustment["repeated_low_lr_pose_instability"] is True
    assert adjustment["learning_rate"] == 2.7e-5
    assert adjustment["low_learning_rate_floor"] == 3.0e-5
    assert adjustment["reason"] == ("repeated_pose_instability_at_low_lr_pose_protected_pathway")
    assert "switches to pose_distillation_loss=huber" in adjustment["policy_logic"]
    assert "hinerv_pose_instability_feedback_unapplied" not in hi["blockers"]
    assert "hinerv_repeated_low_lr_pose_instability_requires_pose_protected_pathway" not in hi["blockers"]
    assert "--learning-rate" in hi["command_argv"]
    assert hi["command_argv"][hi["command_argv"].index("--learning-rate") + 1] == ("2.7e-05")
    assert hi["command_argv"][hi["command_argv"].index("--pose-distillation-loss") + 1] == "huber"
    assert hi["command_argv"][hi["command_argv"].index("--pose-distillation-huber-delta") + 1] == "1"


def test_long_training_campaign_plan_consumes_partial_snerv_runner_feedback() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(_snerv_partial_compact_runner_report(),),
    )

    assert report["candidate_feedback_source_count"] == 1
    assert report["candidate_feedback_row_count"] == 1
    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["schema"] == "nerv_candidate_feedback_row.v1"
    assert feedback["candidate_id"] == _snerv_candidate_id()
    assert feedback["measured_num_pairs"] == 2
    assert feedback["scope_matches_candidate"] is False
    assert feedback["sample_generalization_small_pair_smoke_only"] is True
    assert "small_pair_distortion_smoke_only_not_representative" in snerv[
        "candidate_feedback_evidence_blockers"
    ]
    assert "partial_pair_byte_feedback_only" in snerv["blockers"]
    assert "full600_or_hardpair_distortion_replay_required" in snerv["blockers"]
    assert "snerv_archive_in_loop_byte_oracle_missing" in snerv["blockers"]
    assert "snerv_native_scorer_loop_best_packet_not_materialized" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_receiver_contract_failed" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_pose_guard_not_ready" not in snerv["blockers"]
    assert "snerv_scorer_loop_qat_no_accepted_improvement" not in snerv["blockers"]
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in snerv["blockers"]
    assert "snerv_mlx_native_file_backed_export_proof_missing_or_failed" in snerv["blockers"]
    assert "snerv_mlx_native_packet_file_missing" in snerv["blockers"]
    assert "snerv_mlx_native_full600_campaign_not_ready" in snerv["blockers"]
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_consumes_full600_snerv_native_file_backed_bytes(
    tmp_path: Path,
) -> None:
    runner = _snerv_partial_compact_runner_report()
    report_path = tmp_path / "snerv_mlx_native_train_export.json"
    packet_path = tmp_path / "packet.snar1"
    archive_path = tmp_path / "archive.zip"
    proof_path = tmp_path / "receiver_proof.json"
    report_path.write_text('{"schema":"snerv_mlx_native_train_export.v1"}', encoding="utf-8")
    packet_path.write_bytes(b"SNAR1 packet bytes")
    archive_path.write_bytes(b"archive bytes")
    proof_path.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_ready": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runner["num_pairs"] = 600
    runner["snerv_mlx_native_export"].update(
        {
            "executed": True,
            "candidate_id": _snerv_candidate_id(),
            "num_pairs": 600,
            "artifact_report_path": report_path.as_posix(),
            "packet_path": packet_path.as_posix(),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": _sha256(packet_path),
            "archive_path": archive_path.as_posix(),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256(archive_path),
            "receiver_proof_path": proof_path.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": True,
        }
    )
    runner["snerv_mlx_native_file_backed_export_evidence"] = {
        "schema": "snerv_mlx_native_train_export.v1",
        "executed": True,
        "num_pairs": 600,
        "candidate_id": _snerv_candidate_id(),
        "artifact_report_path": report_path.as_posix(),
        "packet_path": packet_path.as_posix(),
        "packet_sha256": _sha256(packet_path),
        "archive_path": archive_path.as_posix(),
        "archive_sha256": _sha256(archive_path),
        "receiver_proof_path": proof_path.as_posix(),
        "receiver_proof_passed": True,
        "receiver_contract_satisfied": True,
        "file_backed_export_proof_passed": True,
        "required_pair_file_backed_export_proof_passed": True,
        "checkpoint_trained_state_exportable": True,
        "score_aware_long_training_trained_state_exportable": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        # Deliberately only in file-backed evidence: campaign feedback must not
        # drop official authority fields when the native export block is compact.
        "packet_metadata_summary": {
            "schema": "snerv_checkpoint_packet_metadata_summary.v1",
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested": True,
            "snerv_official_mfu_hfr_tub_export_bound": True,
            "snerv_official_mfu_hfr_tub_export_bound_semantics": (
                "receiver_payload_bound_not_source_forward_parity"
            ),
            "snerv_official_mfu_hfr_tub_receiver_payload_bound": True,
            "snerv_official_mfu_hfr_tub_frame_producing_export": True,
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound": False,
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority": False,
            "checkpoint_trained_state_exportable": True,
            "score_aware_long_training_trained_state_exportable": True,
            "source_faithful_stack": False,
            "official_source_parity_blockers": [
                "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
            ],
        },
        "official_checkpoint_export_binding": {
            "schema": "snerv_official_checkpoint_export_binding.v1",
            "trained_state_exportable": True,
            "official_trained_state_exportable": True,
            "official_trained_checkpoint_mapping_manifest": {
                "schema": (
                    "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1"
                ),
                "official_trained_checkpoint_loaded": False,
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
                "official_tub_temporal_encoder_weight_mapping_proven": False,
                "component_rows": [],
                "blockers": [
                    "snerv_official_trained_checkpoint_state_dict_not_loaded",
                    "snerv_official_trained_checkpoint_source_forward_replay_missing",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        },
    }

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(runner,),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["byte_feedback_source"] == "snerv_mlx_native_file_backed_export"
    assert feedback["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert (
        feedback["snerv_official_mfu_hfr_tub_source_forward_replay_authority"]
        is False
    )
    assert feedback["snerv_official_trained_checkpoint_loaded"] is False
    assert (
        feedback["snerv_official_trained_checkpoint_state_dict_mapping_verified"]
        is False
    )
    assert feedback["snerv_trained_state_exportable"] is True
    assert feedback["snerv_checkpoint_trained_state_exportable"] is True
    assert feedback["snerv_score_aware_long_training_trained_state_exportable"] is True
    assert feedback["checkpoint_trained_state_exportable"] is True
    assert feedback["score_aware_long_training_trained_state_exportable"] is True
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in feedback[
        "snerv_official_trained_checkpoint_mapping_blockers"
    ]
    assert feedback["feedback_scope"] == "full600_native_file_backed_snar1_export"
    assert feedback["feedback_ready"] is True
    assert feedback["scope_matches_candidate"] is True
    assert feedback["measured_num_pairs"] == 600
    assert feedback["measured_payload_bytes"] == packet_path.stat().st_size
    assert feedback["measured_archive_bytes"] == archive_path.stat().st_size
    assert (
        feedback["sample_generalization_verdict"]
        == "representative_distortion_evidence_missing"
    )
    assert "representative_distortion_evidence_missing" in snerv[
        "candidate_feedback_evidence_blockers"
    ]
    training_plan = snerv["curriculum_plan"]["training_plan"]
    official_split = snerv["curriculum_plan"][
        "official_source_forward_authority_split"
    ]
    assert official_split["receiver_payload_bound"] is True
    assert official_split["receiver_bound_training_evidence_usable"] is True
    assert official_split["full_source_forward_authority_proven"] is False
    assert (
        "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
        in official_split["blockers"]
    )
    checkpoint_mapping = snerv["snerv_official_trained_checkpoint_mapping"]
    assert checkpoint_mapping["official_trained_checkpoint_loaded"] is False
    assert training_plan["native_mlx_train_export_planned"] is True
    assert training_plan["native_mlx_train_export_verified"] is True
    assert training_plan["native_mlx_scorer_loop_qat_planned"] is True
    assert training_plan["native_mlx_scorer_loop_qat_verified"] is True
    assert "partial_pair_byte_feedback_only" not in snerv["blockers"]
    assert "snerv_snar1_byte_feedback_missing" not in snerv["blockers"]
    assert "snerv_archive_in_loop_byte_oracle_missing" not in snerv["blockers"]
    assert "snerv_mlx_native_file_backed_export_proof_missing_or_failed" not in snerv["blockers"]
    assert "snerv_mlx_native_packet_file_missing" not in snerv["blockers"]
    assert "snerv_mlx_native_full600_campaign_not_ready" not in snerv["blockers"]
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in snerv["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
        in snerv["blockers"]
    )
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in snerv[
        "blockers"
    ]
    assert "snerv_full_video_local_prefilter_missing" in snerv["blockers"]
    assert "snerv_local_cpu_replay_gate_missing" in snerv["blockers"]
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_blocks_loss_worsened_snerv_native_export(
    tmp_path: Path,
) -> None:
    runner = _snerv_partial_compact_runner_report()
    report_path = tmp_path / "snerv_mlx_native_train_export.json"
    packet_path = tmp_path / "packet.snar1"
    archive_path = tmp_path / "archive.zip"
    proof_path = tmp_path / "receiver_proof.json"
    report_path.write_text('{"schema":"snerv_mlx_native_train_export.v1"}', encoding="utf-8")
    packet_path.write_bytes(b"SNAR1 packet bytes")
    archive_path.write_bytes(b"archive bytes")
    proof_path.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_ready": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rejected_training = {
        "schema": "snerv_native_mlx_hf_decoder_training.v1",
        "attempted": True,
        "requested_steps": 2,
        "executed": False,
        "accepted": False,
        "any_loss_worsened": True,
        "all_final_losses_finite": True,
        "blockers": ["snerv_native_mlx_decoder_loss_worsened"],
    }
    runner["num_pairs"] = 600
    runner["snerv_mlx_native_export"].update(
        {
            "executed": True,
            "candidate_id": _snerv_candidate_id(),
            "num_pairs": 600,
            "artifact_report_path": report_path.as_posix(),
            "packet_path": packet_path.as_posix(),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": _sha256(packet_path),
            "archive_path": archive_path.as_posix(),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256(archive_path),
            "receiver_proof_path": proof_path.as_posix(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": True,
            "native_mlx_training_executed": False,
            "native_mlx_hf_decoder_training": rejected_training,
        }
    )
    runner["snerv_mlx_native_file_backed_export_evidence"] = {
        "schema": "snerv_mlx_native_train_export.v1",
        "executed": True,
        "num_pairs": 600,
        "candidate_id": _snerv_candidate_id(),
        "artifact_report_path": report_path.as_posix(),
        "packet_path": packet_path.as_posix(),
        "packet_sha256": _sha256(packet_path),
        "archive_path": archive_path.as_posix(),
        "archive_sha256": _sha256(archive_path),
        "receiver_proof_path": proof_path.as_posix(),
        "receiver_proof_passed": True,
        "receiver_contract_satisfied": True,
        "file_backed_export_proof_passed": True,
        "required_pair_file_backed_export_proof_passed": True,
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("lion",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(runner,),
    )

    snerv = next(row for row in report["campaign_rows"] if row["family"] == "snerv")
    feedback = snerv["candidate_feedback"]
    assert feedback["byte_feedback_source"] is None
    assert feedback["feedback_ready"] is False
    assert feedback["snerv_mlx_native_training_export_guard_passed"] is False
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in feedback[
        "snerv_mlx_native_training_export_guard_blockers"
    ]
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in snerv["blockers"]
    training_plan = snerv["curriculum_plan"]["training_plan"]
    assert training_plan["native_mlx_train_export_verified"] is False
    assert training_plan["native_mlx_file_backed_export_proof_passed"] is False
    assert "snerv_mlx_native_file_backed_export_proof_missing_or_failed" in snerv["blockers"]
    assert snerv["score_claim"] is False
    assert snerv["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_plan_rejects_unknown_optimizer() -> None:
    with pytest.raises(NervLongTrainingCampaignPlanError, match="unsupported"):
        build_nerv_long_training_campaign_plan(
            hinerv_modelsize_budget=_hinerv_budget(),
            snerv_modelsize_budget=_snerv_budget(),
            optimizer_kinds=("not_a_real_optimizer",),
        )


def test_default_optimizer_kinds_cover_native_mlx_optimizer_surface() -> None:
    explicit_timing_smoke_only = {"aurora_like"}
    assert set(DEFAULT_OPTIMIZER_KINDS) == (
        set(SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS) - explicit_timing_smoke_only
    )
    assert explicit_timing_smoke_only.issubset(
        set(SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS)
    )
    assert DEFAULT_OPTIMIZER_KINDS[:5] == (
        "pact_muon_adamw",
        "adamw",
        "muon",
        "lion",
        "adamax",
    )


def test_aurora_like_optimizer_row_is_native_mlx_timing_smoke_and_fail_closed() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("aurora",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["optimizer_kind"] == "aurora_like"
    assert hi["implementation_status"] == (
        "optimizer_timing_smoke_required_before_campaign_launch"
    )
    assert hi["local_mlx_launch_command_ready"] is False
    assert hi["local_mlx_executable"] is False
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False
    assert "--optimizer-kind" in hi["command_argv"]
    assert hi["command_argv"][hi["command_argv"].index("--optimizer-kind") + 1] == "aurora_like"

    expected_launch_blockers = {"aurora_requires_local_timing_convergence_smoke"}
    assert expected_launch_blockers.issubset(set(hi["blockers"]))
    assert "aurora_not_integrated_with_mlx_score_aware_optimizer_contract" not in set(
        hi["blockers"]
    )
    assert expected_launch_blockers.issubset(
        set(hi["experiment_queue_entry"]["launch_authority_contract"]["queue_launch_blockers"])
    )
    assert hi["experiment_queue_entry"]["status"] == "disabled"
    assert hi["experiment_queue_entry"]["blocked"] is True
    assert hi["experiment_queue_entry"]["launch_authority_contract"]["queue_status_is_runnable_plan"] is False
    assert hi["score_lowering_gate"]["command_materialized"] is False
    assert hi["score_lowering_gate"]["local_mlx_executable"] is False
    assert expected_launch_blockers.issubset(
        set(hi["score_lowering_gate"]["launch_blockers"])
    )

    control = hi["optimizer_control"]
    assert control["classification"] == (
        "runnable_false_authority_timing_smoke_candidate"
    )
    assert control["backend"] == (
        "tac.substrates._shared.mlx_score_aware.adapter.AuroraLikeMlxOptimizer"
    )
    assert control["native_mlx_on_apple_silicon"] is True
    assert control["native_mlx_optimizer_object"] is True
    assert control["score_claim"] is False
    assert expected_launch_blockers.issubset(set(control["launch_blockers"]))
    assert "aurora_not_pr95_source_authority" in set(control["authority_blockers"])

    policy = hi["optimizer_policy"]
    assert policy["classification"] == (
        "runnable_false_authority_timing_smoke_candidate"
    )
    assert policy["is_plan_only_optimizer_control"] is False
    assert policy["is_timing_smoke_optimizer_control"] is True
    assert policy["native_mlx_optimizer_expected"] is True
    assert policy["pr95_faithful_curriculum_expected"] is False
    assert policy["runner_policy_if_implemented"] == "native_optimizer"

    optimizer_policy = report["optimizer_control_policy"]
    assert optimizer_policy["selected_plan_only_optimizer_kinds"] == []
    assert optimizer_policy["selected_timing_smoke_optimizer_kinds"] == ["aurora_like"]
    assert "aurora_like" in optimizer_policy["native_mlx_optimizer_kinds"]


def test_adamw_hinerv_row_is_explicit_pr95_curriculum_control() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["optimizer_kind"] == "adamw"
    assert hi["optimizer_policy"]["requested_policy"] == "pr95_curriculum"
    assert hi["optimizer_policy"]["pr95_faithful_curriculum_expected"] is True
    assert hi["optimizer_policy"]["native_mlx_optimizer_expected"] is False
    assert hi["command_argv"][hi["command_argv"].index("--hi-nerv-optimizer-policy") + 1] == "pr95_curriculum"


def test_native_muon_hinerv_row_is_not_pact_or_pr95_curriculum() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("muon",),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    hi = next(row for row in report["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["optimizer_kind"] == "muon"
    assert hi["optimizer_control"]["backend"] == "mlx.optimizers"
    assert hi["optimizer_control"]["native_mlx_optimizer_object"] is True
    assert hi["optimizer_control"]["pact_partitioned_muon_adamw"] is False
    assert hi["optimizer_policy"]["requested_policy"] == "native_optimizer"
    assert hi["optimizer_policy"]["pr95_faithful_curriculum_expected"] is False
    assert hi["optimizer_policy"]["native_mlx_optimizer_expected"] is True
    assert hi["command_argv"][hi["command_argv"].index("--hi-nerv-optimizer-policy") + 1] == "native_optimizer"


def test_pact_muon_adamw_hinerv_row_is_default_first_priority() -> None:
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("pact_muon_adamw", "adamw", "lion"),
        epochs=29_650,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )

    hi_rows = [row for row in report["campaign_rows"] if row["family"] == "hi_nerv"]
    assert hi_rows[0]["optimizer_kind"] == "pact_muon_adamw"
    assert hi_rows[0]["priority"] == 9
    assert hi_rows[0]["optimizer_control"]["backend"] == ("tac.local_acceleration.pr95_hnerv_mlx")
    assert hi_rows[0]["optimizer_control"]["borrowed_from_pr95"] is True
    assert hi_rows[0]["optimizer_control"]["original_pact_contest_adaptation"] is True
    assert hi_rows[0]["optimizer_policy"]["requested_policy"] == "pr95_curriculum"
    assert hi_rows[0]["optimizer_policy"]["pr95_faithful_curriculum_expected"] is True
    assert hi_rows[0]["optimizer_policy"]["native_mlx_optimizer_expected"] is False
    assert (
        hi_rows[0]["command_argv"][
            hi_rows[0]["command_argv"].index("--hi-nerv-optimizer-policy") + 1
        ]
        == "pr95_curriculum"
    )
    assert report["optimizer_control_policy"]["default_optimizer_kind"] == ("pact_muon_adamw")
    assert report["optimizer_control_policy"]["default_optimizer_backend"] == ("tac.local_acceleration.pr95_hnerv_mlx")


def test_build_long_training_campaign_plan_cli_writes_outputs(tmp_path: Path) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    out_md = tmp_path / "campaign.md"
    out_queue = tmp_path / "campaign_queue.json"
    out_snerv_lf_reroute_queue = tmp_path / "snerv_lf_reroute_queue.json"
    feedback_jsonl = tmp_path / "feedback.jsonl"
    waterfill_bundle = tmp_path / "hinerv_archive_ladder_waterfill.json"
    archive_section_telemetry = tmp_path / "hi_nerv_archive_section_telemetry.json"
    snerv_tether_smoke = tmp_path / "snerv_scorer_tether_smoke.json"
    proof_path = _receiver_proof(tmp_path, archive_sha="a" * 64)
    cache_quality_path = _receiver_cache_quality_report(tmp_path, passed=True)
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    feedback_jsonl.write_text(
        json.dumps(_snerv_partial_compact_runner_report(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    waterfill_bundle.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "rows": [
                    {
                        "row_id": "hinerv_tiny",
                        "archive_sha256": "a" * 64,
                        "receiver_proof_path": proof_path.as_posix(),
                        "waterfill_plan": _decoder_weight_waterfill_plan(
                            candidate_id="source_prefix:hinerv_tiny",
                            receiver_proof_status="runtime_consumption_proof_ready",
                        ),
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    archive_section_telemetry.write_text(
        json.dumps(
            _archive_section_telemetry(
                candidate_id="hinerv_tiny",
                archive_sha="a" * 64,
                archive_zip_bytes=177_000,
                receiver_proof_path=proof_path,
                cache_quality_report_path=cache_quality_path,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snerv_tether_smoke.write_text(
        json.dumps(
            {
                "schema": "snerv_scorer_tether_smoke.v1",
                "created_utc": "2026-06-05T00:00:00Z",
                "steps": 2,
                "passed": True,
                "metric_summary": {
                    "loss_part_distill": 0.5,
                    "loss_part_pose_distill": 0.25,
                    "snerv_segnet_last_frame_distill_lambda": 0.01,
                    "snerv_posenet_yuv6_pair_distill_lambda": 0.01,
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            str(hinerv),
            "--snerv-modelsize-budget",
            str(snerv),
            "--joint-recon-weight-manifest",
            str(_joint_recon_weight_manifest(tmp_path, num_pairs=600)),
            "--optimizer-kind",
            "lion",
            "--candidate-feedback-source",
            str(feedback_jsonl),
            "--decoder-weight-waterfill-source",
            str(waterfill_bundle),
            "--archive-section-telemetry-source",
            str(archive_section_telemetry),
            "--snerv-scorer-tether-smoke-report",
            str(snerv_tether_smoke),
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--output-snerv-lf-reroute-queue",
            str(out_snerv_lf_reroute_queue),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["campaign_row_count"] == 2
    assert payload["planner_row_queue_artifact_path"] == out_queue.as_posix()
    assert payload["candidate_feedback_row_count"] == 1
    assert payload["decoder_weight_waterfill_attached_row_count"] == 1
    assert payload["archive_section_telemetry_attached_row_count"] == 1
    assert payload["snerv_scorer_tether_smoke_report_attached"] is True
    assert payload["snerv_scorer_tether_smoke_gate"]["passed"] is True
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert "--planner-row-queue-artifact" in hi["command_argv"]
    assert hi["command_argv"][hi["command_argv"].index("--planner-row-queue-artifact") + 1] == out_queue.as_posix()
    assert "--recon-pixel-weight-path" in hi["command_argv"]
    assert "--decoder-weight-waterfill-plan-json" in hi["command_argv"]
    assert "--archive-section-telemetry-json" in hi["command_argv"]
    assert hi["command_argv"][hi["command_argv"].index("--archive-section-telemetry-json") + 1] == (
        archive_section_telemetry.resolve(strict=False).as_posix()
    )
    waterfill_sidecar = Path(hi["command_argv"][hi["command_argv"].index("--decoder-weight-waterfill-plan-json") + 1])
    assert waterfill_sidecar.is_file()
    assert waterfill_sidecar.parent.name == "decoder_weight_waterfill_sidecars"
    assert hi["decoder_weight_waterfill_plan"]["source_path"] == (waterfill_bundle.resolve(strict=False).as_posix())
    snerv_row = next(row for row in payload["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["candidate_feedback"]["candidate_id"] == _snerv_candidate_id()
    assert snerv_row["snerv_scorer_tether_smoke_gate"]["passed"] is True
    assert "partial_pair_byte_feedback_only" in snerv_row["blockers"]
    assert payload["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert payload["snerv_lf_over_ceiling_reroute_queue"]["schema"] == (
        "snerv_lf_over_ceiling_reroute_queue.v1"
    )
    assert payload["experiment_queue_id"] == (f"nerv_long_training_campaign_{out_json.stem}.v1")
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue == payload["experiment_queue"]
    reroute_queue = json.loads(out_snerv_lf_reroute_queue.read_text(encoding="utf-8"))
    assert reroute_queue == payload["snerv_lf_over_ceiling_reroute_queue"]
    assert reroute_queue["queue_kind"] == "planner_queue_not_training_queue"
    assert queue["experiments"][0]["steps"][0]["command"] == payload["campaign_rows"][0]["command_argv"]
    assert queue["queue_id"] == f"nerv_long_training_campaign_{out_json.stem}.v1"
    assert queue["experiments"][0]["steps"][0]["postconditions"]
    snerv_exp = next(exp for exp in queue["experiments"] if exp["family"] == "snerv")
    assert "--planner-row-queue-artifact" in snerv_exp["steps"][0]["command"]
    assert snerv_exp["steps"][0]["command"][
        snerv_exp["steps"][0]["command"].index("--planner-row-queue-artifact") + 1
    ] == out_queue.as_posix()
    assert snerv_exp["status"] == "disabled"
    assert snerv_exp["blocked"] is True
    assert "--snerv-score-aware-long-training-epochs" in snerv_exp["steps"][0]["command"]
    loaded_queue = load_queue_definition(out_queue)
    assert loaded_queue["schema"] == "experiment_queue.v1"
    loaded_snerv_exp = next(exp for exp in loaded_queue["experiments"] if exp["family"] == "snerv")
    assert loaded_snerv_exp["status"] == "disabled"
    assert loaded_snerv_exp["blocked"] is True
    assert loaded_snerv_exp["launch_authority_contract"]["queue_status_is_runnable_plan"] is False
    assert loaded_snerv_exp["steps"][0]["resources"]["kind"] == "local_mlx"
    assert out_md.read_text(encoding="utf-8").startswith("# NeRV Long-Training Campaign Plan")

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            str(hinerv),
            "--snerv-modelsize-budget",
            str(snerv),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
            "--expected-output-json-sha256",
            _sha256(out_json),
            "--expected-output-md-sha256",
            _sha256(out_md),
            "--expected-output-queue-sha256",
            _sha256(out_queue),
            "--output-snerv-lf-reroute-queue",
            str(out_snerv_lf_reroute_queue),
            "--expected-output-snerv-lf-reroute-queue-sha256",
            _sha256(out_snerv_lf_reroute_queue),
        ]
    )

    assert rc == 0


def test_build_long_training_campaign_plan_cli_extracts_waterfill_from_archive_ladder(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    waterfill_plan = tmp_path / "decoder_weight_waterfill.json"
    archive_ladder = tmp_path / "hinerv_archive_ladder.json"
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    waterfill_plan.write_text(
        json.dumps(_decoder_weight_waterfill_plan(candidate_id="hinerv_tiny")),
        encoding="utf-8",
    )
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_rows": [
                    {
                        "row_id": "hinerv_tiny",
                        "decoder_weight_waterfill_plan_path": waterfill_plan.as_posix(),
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            archive_ladder.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["decoder_weight_waterfill_source_count"] == 1
    assert payload["decoder_weight_waterfill_row_count"] == 1
    assert payload["decoder_weight_waterfill_attached_row_count"] == 1
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["decoder_weight_waterfill_plan"]["attached"] is True
    assert hi["decoder_weight_waterfill_plan"]["path"] == waterfill_plan.as_posix()
    assert hi["decoder_weight_waterfill_plan"]["receiver_proof_ready"] is False
    assert hi["decoder_weight_waterfill_plan"][
        "_archive_size_ladder_runtime_consumption_proof_ready"
    ] is True
    assert "decoder_weight_waterfill_receiver_proof_not_ready" in hi[
        "decoder_weight_waterfill_plan"
    ]["runner_admission"]["refusal_reasons"]


def test_build_long_training_campaign_plan_cli_auto_discovers_bytecap_exports(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    feedback_root = tmp_path / "exports"
    export_dir = feedback_root / "hinerv_epoch16749"
    snerv_export_dir = feedback_root / "snerv_epoch22399"
    contract_only_dir = feedback_root / "snerv_contract_only"
    export_dir.mkdir(parents=True)
    snerv_export_dir.mkdir(parents=True)
    contract_only_dir.mkdir(parents=True)
    candidate = dict(_hinerv_budget()["selected_candidates"][0])
    snerv_candidate = dict(_snerv_budget()["selected_candidates"][0])
    export_report = export_dir / "export_report.json"
    snerv_export_report = snerv_export_dir / "snerv_checkpoint_archive_export.json"
    contract_only_report = (
        contract_only_dir / "snerv_checkpoint_archive_export.json"
    )
    export_report.write_text(
        json.dumps(
            {
                "schema": "hinerv_checkpoint_archive_export.v1",
                "family": "hi_nerv",
                "candidate_id": candidate["candidate_id"],
                "archive_bytes": 214_497,
                "decoder_codec": candidate["decoder_codec"],
                "receiver_proof_ready": True,
                "modelsize_candidate": candidate,
                "hard_byte_ceilings": [178_000],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snerv_export_report.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "candidate_id": snerv_candidate["candidate_id"],
                "archive_bytes": 444_828,
                "packet_bytes": 2_347_476,
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
                "modelsize_candidate": snerv_candidate,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    contract_only_report.write_text(
        json.dumps(
            {
                "schema": "snerv_checkpoint_archive_export.v1",
                "family": "snerv",
                "candidate_id": snerv_candidate["candidate_id"],
                "archive_bytes": 91_445,
                "packet_bytes": 188_000,
                "receiver_contract_satisfied": True,
                "modelsize_candidate": snerv_candidate,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--auto-modelsize-byte-cap-feedback-root",
            feedback_root.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["modelsize_byte_cap_feedback_path_count"] == 2
    assert set(payload["modelsize_byte_cap_feedback_paths"]) == {
        export_report.resolve(strict=False).as_posix(),
        snerv_export_report.resolve(strict=False).as_posix(),
    }
    assert contract_only_report.resolve(strict=False).as_posix() not in payload[
        "modelsize_byte_cap_feedback_paths"
    ]
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["runner_modelsize_candidate_id"] == "auto"
    assert hi["local_mlx_launch_command_ready"] is False
    assert (
        "hi_nerv_modelsize_auto_calibrated_byte_cap_over_ceiling"
        in hi["blockers"]
    )
    preflight = hi["modelsize_byte_cap_preflight"]
    assert preflight["matching_observation_count"] == 1
    assert preflight["matching_observations"][0]["measured_archive_bytes"] == 214_497
    assert preflight["predicted_archive_bytes"] == 214_497
    assert preflight["predicted_under_hard_byte_ceiling"] is False


def test_build_long_training_campaign_plan_cli_selects_archive_ladder_waterfill_candidate(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    waterfill_plan = tmp_path / "decoder_weight_waterfill.json"
    archive_ladder = tmp_path / "hinerv_archive_ladder.json"
    proof_path = _receiver_proof(tmp_path, archive_sha="a" * 64)
    ladder_candidate = dict(_hinerv_budget()["selected_candidates"][0])
    ladder_candidate.update(
        {
            "candidate_id": "hinerv_ladder_receiver_backed",
            "nominal_total_payload_bytes": 88_000,
            "byte_headroom": 90_000,
        }
    )
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_ladder_receiver_backed",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill["full_video_coverage"] = True
    waterfill_plan.write_text(json.dumps(waterfill), encoding="utf-8")
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_rows": [
                    {
                        "row_id": "hinerv_ladder_receiver_backed",
                        "modelsize_candidate": ladder_candidate,
                        "decoder_weight_waterfill_plan_path": waterfill_plan.as_posix(),
                        "archive_sha256": "a" * 64,
                        "receiver_proof_path": proof_path.as_posix(),
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            archive_ladder.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_id"] == "hinerv_ladder_receiver_backed"
    assert hi["candidate"]["_candidate_source"] == (
        "decoder_weight_waterfill_modelsize_candidate"
    )
    assert hi["decoder_weight_waterfill_plan"]["attached"] is True
    assert hi["decoder_weight_waterfill_plan"]["runner_admitted"] is True
    assert hi["decoder_weight_waterfill_plan"]["receiver_proof_binding"][
        "proof_path"
    ] == proof_path.as_posix()
    assert "hinerv_decoder_weight_waterfill_plan_missing" not in hi["blockers"]
    assert payload["decoder_weight_waterfill_unattached_source_count"] == 0
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def test_build_long_training_campaign_plan_cli_preserves_archive_ladder_waterfill_custody(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    waterfill_bundle = tmp_path / "hinerv_archive_ladder_waterfill.json"
    archive_ladder = tmp_path / "hinerv_archive_size_ladder.json"
    proof_path = _receiver_proof(tmp_path, archive_sha="a" * 64)
    ladder_candidate = dict(_hinerv_budget()["selected_candidates"][0])
    ladder_candidate.update(
        {
            "candidate_id": "hinerv_ladder_waterfill_bundle",
            "nominal_total_payload_bytes": 88_000,
            "byte_headroom": 90_000,
        }
    )
    nested_waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_ladder_waterfill_bundle",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    # The real derived waterfill bundle carries proof identity at the row/ladder
    # layer, not inside the nested plan. The planner must preserve that custody.
    nested_waterfill.pop("archive_sha256")
    nested_waterfill.pop("full_video_coverage")
    nested_waterfill.pop("receiver_proof_status")
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_rows": [
                    {
                        "row_id": "hinerv_ladder_waterfill_bundle",
                        "modelsize_candidate": ladder_candidate,
                        "archive_sha256": "a" * 64,
                        "receiver_proof_path": proof_path.as_posix(),
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    waterfill_bundle.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "source_schema": "hinerv_archive_size_ladder.v1",
                "archive_ladder_report_path": archive_ladder.as_posix(),
                "full_video_coverage": True,
                "num_pairs": 600,
                "rows": [
                    {
                        "row_id": "hinerv_ladder_waterfill_bundle",
                        "archive_sha256": "a" * 64,
                        "waterfill_plan": nested_waterfill,
                        "blockers": [],
                        "saliency_replay_blockers": [],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            waterfill_bundle.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert hi["candidate_id"] == "hinerv_ladder_waterfill_bundle"
    assert attachment["attached"] is True
    assert attachment["full_video_coverage"] is True
    assert attachment["receiver_proof_ready"] is True
    assert attachment["runner_admitted"] is True
    assert attachment["receiver_proof_binding"]["archive_sha256"] == "a" * 64
    assert attachment["receiver_proof_binding"]["proof_path"] == proof_path.as_posix()
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" not in hi["blockers"]
    assert "--decoder-weight-waterfill-plan-json" in hi["command_argv"]


def test_build_long_training_campaign_plan_cli_rejects_stale_ladder_receiver_proof(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    waterfill_plan = tmp_path / "decoder_weight_waterfill.json"
    archive_ladder = tmp_path / "hinerv_archive_ladder.json"
    proof_path = _receiver_proof(tmp_path, archive_sha="b" * 64)
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_tiny",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill_plan.write_text(json.dumps(waterfill), encoding="utf-8")
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_rows": [
                    {
                        "row_id": "hinerv_tiny",
                        "decoder_weight_waterfill_plan_path": waterfill_plan.as_posix(),
                        "archive_sha256": "a" * 64,
                        "receiver_proof_path": proof_path.as_posix(),
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            archive_ladder.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["runner_admitted"] is False
    assert attachment["receiver_proof_binding"]["bound"] is False
    assert "decoder_weight_waterfill_receiver_proof_archive_sha256_mismatch" in (
        attachment["runner_admission"]["refusal_reasons"]
    )
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]


def test_build_long_training_campaign_plan_keeps_current_budget_over_stale_waterfill_sidecar(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    waterfill_plan = tmp_path / "decoder_weight_waterfill.json"
    archive_ladder = tmp_path / "hinerv_archive_ladder.json"
    current_candidate = dict(_hinerv_budget()["selected_candidates"][0])
    stale_candidate = dict(current_candidate)
    stale_candidate["decoder_codec"] = "int8_mixed"
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id=current_candidate["candidate_id"],
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill_plan.write_text(json.dumps(waterfill), encoding="utf-8")
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_rows": [
                    {
                        "row_id": current_candidate["candidate_id"],
                        "modelsize_candidate": stale_candidate,
                        "decoder_weight_waterfill_plan_path": waterfill_plan.as_posix(),
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            archive_ladder.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert hi["candidate_id"] == current_candidate["candidate_id"]
    assert hi["candidate"]["decoder_codec"] == current_candidate["decoder_codec"]
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is False
    assert "decoder_weight_waterfill_modelsize_mismatch:decoder_codec" in attachment["blockers"]
    assert "decoder_weight_waterfill_receiver_proof_not_ready" in attachment[
        "runner_admission"
    ]["refusal_reasons"]
    assert "--decoder-weight-waterfill-plan-json" not in hi["command_argv"]


def test_build_long_training_campaign_plan_cli_emits_ladder_saliency_replay_work_order(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    waterfill_plan = tmp_path / "decoder_weight_waterfill.json"
    archive_ladder = tmp_path / "hinerv_archive_ladder.json"
    ladder_candidate = dict(_hinerv_budget()["selected_candidates"][0])
    ladder_candidate.update({"candidate_id": "hinerv_ladder_needs_saliency"})
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="hinerv_ladder_needs_saliency",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill["full_video_coverage"] = False
    waterfill["blockers"] = [
        "decoder_weight_saliency_missing_for_some_groups",
        "full_video_coverage_missing",
    ]
    for row in waterfill["rows"]:
        row["blockers"] = ["decoder_weight_group_saliency_missing"]
    waterfill_plan.write_text(json.dumps(waterfill), encoding="utf-8")
    hinerv.write_text(json.dumps(_hinerv_budget()), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_rows": [
                    {
                        "row_id": "hinerv_ladder_needs_saliency",
                        "modelsize_candidate": ladder_candidate,
                        "decoder_weight_waterfill_plan_path": waterfill_plan.as_posix(),
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            archive_ladder.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["attached"] is True
    assert attachment["runner_admitted"] is False
    work_order = attachment["saliency_replay_work_order"]
    assert work_order["schema"] == "nerv_decoder_weight_saliency_replay_work_order.v1"
    assert work_order["required"] is True
    assert work_order["coverage_required"] == "full600_start0_stride1"
    assert work_order["blockers"] == []
    assert "tools/build_hinerv_decoder_weight_saliency_replay.py" in work_order[
        "saliency_replay_command_argv"
    ]
    assert "--max-pairs" in work_order["saliency_replay_command_argv"]
    assert "600" in work_order["saliency_replay_command_argv"]
    assert "tools/build_hinerv_archive_ladder_waterfill.py" in work_order[
        "waterfill_rebuild_command_argv"
    ]
    assert work_order["campaign_rebuild_hint_argv"] == []
    assert work_order["campaign_rebuild_decoder_weight_waterfill_source"] == work_order[
        "expected_output_waterfill_json"
    ]
    assert "--hinerv-modelsize-budget" in work_order["campaign_rebuild_required_inputs"]
    for argv in (
        work_order["saliency_replay_command_argv"],
        work_order["waterfill_rebuild_command_argv"],
        work_order["campaign_rebuild_hint_argv"],
    ):
        assert all("${" not in str(item) for item in argv)
        assert all(not str(item).strip().startswith("#") for item in argv)
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def test_archive_ladder_waterfill_reingest_preserves_replay_source_and_refuses_unfit_basin(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    archive_ladder = tmp_path / "hinerv_archive_ladder.json"
    waterfill_bundle = tmp_path / "hinerv_archive_ladder_waterfill.json"
    candidate = dict(_hinerv_budget()["selected_candidates"][0])
    candidate.update({"candidate_id": "hinerv_ladder_unfit"})
    waterfill = _decoder_weight_waterfill_plan(
        candidate_id="source_hinerv:hinerv_ladder_unfit",
        receiver_proof_status="runtime_consumption_proof_ready",
    )
    waterfill["full_video_coverage"] = True
    waterfill["blockers"] = [
        "decoder_weight_saliency_replay_has_blockers",
        "score_loss_proxy_outside_allocator_linearization_basin",
        "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin",
    ]
    hinerv_budget = _hinerv_budget()
    hinerv_budget["selected_candidates"] = [candidate]
    hinerv.write_text(json.dumps(hinerv_budget), encoding="utf-8")
    snerv.write_text(json.dumps(_snerv_budget()), encoding="utf-8")
    archive_ladder.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "report_path": archive_ladder.as_posix(),
                "archive_rows": [
                    {
                        "row_id": "hinerv_ladder_unfit",
                        "modelsize_candidate": candidate,
                        "runtime_consumption_proof_ready": True,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    waterfill_bundle.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "source_schema": "hinerv_archive_size_ladder.v1",
                "archive_ladder_report_path": archive_ladder.as_posix(),
                "rows": [
                    {
                        "row_id": "hinerv_ladder_unfit",
                        "waterfill_plan": waterfill,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        [
            "--hinerv-modelsize-budget",
            hinerv.as_posix(),
            "--snerv-modelsize-budget",
            snerv.as_posix(),
            "--optimizer-kind",
            "lion",
            "--epochs",
            "16",
            "--max-candidates-per-family",
            "1",
            "--decoder-weight-waterfill-source",
            waterfill_bundle.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    attachment = hi["decoder_weight_waterfill_plan"]
    assert attachment["source_path"] == archive_ladder.resolve(strict=False).as_posix()
    assert attachment["_archive_size_ladder_source_schema"] == "hinerv_archive_size_ladder.v1"
    assert attachment["runner_admitted"] is False
    refusal_reasons = attachment["runner_admission"]["refusal_reasons"]
    assert "score_loss_proxy_outside_allocator_linearization_basin" in refusal_reasons
    assert "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin" in refusal_reasons
    replay_work_order = attachment["saliency_replay_work_order"]
    assert replay_work_order["required"] is False
    recovery_work_order = attachment["allocator_basin_recovery_work_order"]
    assert recovery_work_order["schema"] == "nerv_decoder_weight_allocator_basin_recovery_work_order.v1"
    assert recovery_work_order["required"] is True
    assert recovery_work_order["candidate_id"] == "hinerv_ladder_unfit"
    assert recovery_work_order["blockers"] == []
    assert recovery_work_order["source_decoder_weight_waterfill_report_path"] == archive_ladder.resolve(
        strict=False
    ).as_posix()
    assert "--execute-family" in recovery_work_order["command_argv"]
    assert "hi_nerv" in recovery_work_order["command_argv"]
    assert "--hi-nerv-optimizer-policy" in recovery_work_order["command_argv"]
    assert "pr95_curriculum" in recovery_work_order["command_argv"]
    assert hi["local_mlx_launch_command_ready"] is False
    assert hi["score_claim"] is False
    assert hi["ready_for_exact_eval_dispatch"] is False


def _hinerv_budget() -> dict:
    return {
        "schema": "nerv_modelsize_budget.v1",
        "selected_candidates": [
            {
                "schema": "hinerv_modelsize_candidate.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "decoder_codec": "int4_mixed",
                "nominal_total_payload_bytes": 120_000,
                "nominal_under_ceiling": True,
                "use_hierarchical_feature_grid": True,
                "use_convnext_blocks": True,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_budget() -> dict:
    return {
        "schema": "snerv_modelsize_budget.v1",
        "selected_candidates": [
            {
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": (
                    "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_p1_mfu1-2-4_hfr0_t0_adbase_int4_symmetric_ceil178000"
                ),
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "wavelet": "haar",
                "levels": 2,
                "bits_per_coeff": 1.5,
                "step_map_bits_per_coeff": 0.5,
                "decoder_payload_codec": "int4_symmetric",
                "snerv_model_size_adapter": "snerv_fc_dim_emb_size_adapter_v1",
                "fc_dim": 11,
                "emb_size": 2,
                "patch_radius": 1,
                "mfu_scales": [1, 2, 4],
                "hfr_gain": 0.0,
                "temporal_context": 0,
                "temporal_mode": "delta",
                "snerv_native_mlx_receiver_proof_timeout": 321,
                "snerv_native_mlx_decoder_train_steps": 5,
                "snerv_native_mlx_decoder_train_lr": 2.5e-4,
                "snerv_native_mlx_decoder_train_ridge": 2.0e-6,
                "decoder_feature_count": 16,
                "nominal_total_payload_bytes": 190_000,
                "nominal_under_ceiling": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_budget_with_candidate(candidate: dict) -> dict:
    return _snerv_budget_with_candidates((candidate,))


def _snerv_budget_with_candidates(candidates: tuple[dict, ...]) -> dict:
    return {
        "schema": "snerv_modelsize_budget.v1",
        "selected_candidates": list(candidates),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _passing_snerv_tether_smoke_report() -> dict:
    return {
        "schema": "snerv_scorer_tether_smoke.v1",
        "created_utc": "2026-06-05T00:51:03Z",
        "steps": 2,
        "passed": True,
        "blockers": [],
        "metric_summary": {
            "final": {
                "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                "dual_ascent_lambda__snerv_segnet_last_frame_distill": 1.0,
                "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 1.0,
            },
            "step_count": 2,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_official_skip_candidate(mode: str) -> dict:
    return analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        wavelet="haar",
        levels=1,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        official_modelsize_mparams=0.05,
        emb_size=0,
        patch_radius=1,
        mfu_scales=(1, 2, 4),
        hfr_gain=0.0,
        temporal_context=0,
        temporal_mode="official_haar_dwt1d_lowpass",
        official_skip_high_mode=mode,
    ).as_dict()


def _write_snerv_binary_profile_receiver_feedback(
    tmp_path: Path,
    *,
    candidate: dict,
    archive_bytes: int,
    archive_sha256: str,
) -> Path:
    run_root = tmp_path / "snerv_run"
    package = (
        run_root
        / "snerv_mlx_native_export"
        / "native_train_export"
        / "snerv_mlx_native_archive_bound_package"
    )
    archive = package / "archive.zip"
    packet = package.parent / "snerv_mlx_native_packet.snar"
    proof = package / "receiver_proof" / "snerv_inverse_steg_receiver_proof.json"
    profile_dir = tmp_path / "binary_profile"
    startup = run_root / "compact_renderer_mlx_spine_runner_startup.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"synthetic archive bytes")
    packet.write_bytes(b"SNAR1 synthetic packet bytes")
    actual_archive_sha = _sha256(archive)
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "execute_family": "snerv",
                "modelsize_candidate": candidate,
                "modelsize_candidate_id": "auto",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(
        json.dumps(
            {
                "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
                "archive_bytes": int(archive_bytes),
                "archive_path": archive.as_posix(),
                "archive_sha256": actual_archive_sha,
                "runtime_consumption_proof_ready": True,
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_dir / "snerv_binary_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "snerv_binary_profile.v1",
                "charged_archive_bytes": int(archive_bytes),
                "input_kind": "contest_archive_zip",
                "input_path": archive.as_posix(),
                "input_sha256": actual_archive_sha,
                "snar1_metadata": {"n_pairs": int(candidate["num_pairs"])},
                "snar1_packet_bytes": packet.stat().st_size,
                "snar1_packet_sha256": _sha256(packet),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return profile


def _snerv_candidate_id() -> str:
    return str(_snerv_budget()["selected_candidates"][0]["candidate_id"])


def _snerv_lf_recode_report(
    *,
    mode: str,
    source_packet_bytes: int,
    candidate_packet_bytes: int,
    source_lf_bytes: int,
    candidate_lf_bytes: int,
    candidate_packet_header_bytes: int | None = None,
    candidate_packet_path: str | None = None,
) -> dict:
    return {
        "schema": "snerv_lf_payload_archive_recode.v1",
        "mode": mode,
        "report_path": (
            "/Volumes/VertigoDataTier/pact/reports/"
            f"snerv_lf_recode_{mode}.json"
        ),
        "source_packet": {"bytes": source_packet_bytes, "sha256": "a" * 64},
        "candidate_packet": {
            "bytes": candidate_packet_bytes,
            "sha256": "b" * 64,
            **({} if candidate_packet_path is None else {"path": candidate_packet_path}),
            **(
                {}
                if candidate_packet_header_bytes is None
                else {"header_bytes": candidate_packet_header_bytes}
            ),
        },
        "packet_byte_delta": int(candidate_packet_bytes - source_packet_bytes),
        "lf_payload": {
            "source_bytes": source_lf_bytes,
            "candidate_bytes": candidate_lf_bytes,
            "byte_delta": int(candidate_lf_bytes - source_lf_bytes),
        },
        "section_bytes": {
            "source": {
                "metadata_payload": 64,
                "lf_payload": source_lf_bytes,
                "decoder_payload": 1024,
                "step_map_packet": max(source_packet_bytes - source_lf_bytes - 1088, 0),
            },
            "candidate": {
                "metadata_payload": 64,
                "lf_payload": candidate_lf_bytes,
                "decoder_payload": 1024,
                "step_map_packet": max(
                    candidate_packet_bytes - candidate_lf_bytes - 1088,
                    0,
                ),
            },
        },
        "receiver_contract_satisfied": True,
        "runtime_consumption_proof_ready": True,
        "receiver_frame_equality_proof": {"status": "proven_exact"},
        "blockers": [
            "not_packaged_as_contest_archive_zip",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_snar_header_grammar_profile(
    *,
    packet_sha256: str,
    packet_bytes: int,
    header_bytes: int,
    metadata_json_bytes: int,
    section_total_bytes: int,
) -> dict:
    return {
        "schema": "snerv_snar_header_grammar_profile.v1",
        "input": {
            "path": "/tmp/candidate.snar",
            "kind": "snar1_packet",
            "bytes": packet_bytes,
            "sha256": packet_sha256,
        },
        "packet": {
            "bytes": packet_bytes,
            "sha256": packet_sha256,
            "schema_valid": True,
        },
        "header": {
            "bytes": header_bytes,
            "metadata_json_bytes": metadata_json_bytes,
            "metadata_top_contributor_rows": [
                {
                    "path": "$.metadata",
                    "json_bytes": metadata_json_bytes,
                    "type": "dict",
                    "item_count": 53,
                    "depth": 0,
                },
                {
                    "path": "$.metadata.lf_step_allocation_rows",
                    "json_bytes": metadata_json_bytes - 100_000,
                    "type": "list",
                    "item_count": 3600,
                    "depth": 1,
                },
            ],
        },
        "payload": {
            "section_total_bytes": section_total_bytes,
            "section_rows": [],
        },
        "hard_byte_ceiling_rows": [
            {
                "hard_byte_ceiling": 216_000,
                "packet_over_ceiling_bytes": packet_bytes - 216_000,
                "header_bytes_can_cover_overrun": True,
                "metadata_json_bytes_can_cover_overrun": True,
                "sections_alone_under_ceiling": True,
            }
        ],
        "next_actions": [
            "build_receiver_visible_snar_header_minimization_candidate"
        ],
        "blockers": ["snerv_snar_packet_header_grammar_rewrite_required"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_snar_header_minimization_report(
    *,
    source_packet_sha256: str,
    candidate_id: str | None = None,
    candidate_packet_bytes: int = 139_855,
    full_video_receiver_contract_satisfied: bool = False,
) -> dict:
    report = {
        "schema": "snerv_snar_header_minimization.v1",
        "source_packet": {
            "path": "/Volumes/VertigoDataTier/pact/snerv_test/candidate.snar",
            "bytes": 1_485_285,
            "sha256": source_packet_sha256,
            "header_bytes": 1_346_233,
        },
        "candidate_packet": {
            "path": "/Volumes/VertigoDataTier/pact/snerv_test/candidate.minimized.snar",
            "bytes": int(candidate_packet_bytes),
            "sha256": "c" * 64,
            "header_bytes": 803,
        },
        "candidate_archive_zip": {
            "path": "/Volumes/VertigoDataTier/pact/snerv_test/archive.zip",
            "bytes": 139_963,
            "sha256": "d" * 64,
            "member": "0.bin",
        },
        "packet_byte_delta": -1_345_430,
        "header_byte_delta": -1_345_430,
        "receiver_contract_satisfied": True,
        "full_video_receiver_contract_satisfied": (
            bool(full_video_receiver_contract_satisfied)
        ),
        "receiver_pair_frame_equality_proof": {
            "status": "proven_exact",
            "scope": (
                "full_video_streaming"
                if full_video_receiver_contract_satisfied
                else "sampled_pairs"
            ),
            "exact_equal": True,
        },
        "hard_byte_ceiling_rows": [
            {
                "hard_byte_ceiling": 178_000,
                "candidate_packet_under_ceiling": True,
                "candidate_archive_zip_under_ceiling": True,
                "candidate_packet_over_ceiling_bytes": 0,
                "candidate_archive_zip_over_ceiling_bytes": 0,
            }
        ],
        "blockers": [
            "snerv_snar_header_minimization_false_authority",
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if candidate_id is not None:
        report["candidate_binding"] = {
            "candidate_id": str(candidate_id),
            "binding_status": "candidate_id_and_source_packet_sha256",
            "source_packet_sha256": source_packet_sha256,
            "candidate_packet_sha256": "c" * 64,
            "candidate_id_required_for_launch_reenable": True,
        }
    return report


def _decoder_weight_waterfill_plan(
    *,
    candidate_id: str,
    receiver_proof_status: str = "missing",
) -> dict:
    receiver_ready = receiver_proof_status in {
        "runtime_consumption_proof_ready",
        "receiver_proof_valid",
        "runtime_consumption_proof_passed",
    }
    blockers = [] if receiver_ready else ["receiver_proof_not_satisfied"]
    return {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "group_count": 2,
        "full_video_coverage": True,
        "receiver_proof_status": receiver_proof_status,
        "archive_sha256": "a" * 64,
        "rows": [
            {
                "group_name": "blocks.0.conv.weight",
                "selected_bits": 4,
                "selected_action": "int4",
                "blockers": blockers,
            },
            {
                "group_name": "head_rgb_0.bias",
                "selected_bits": 2,
                "selected_action": "int2",
                "blockers": blockers,
            },
        ],
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _archive_section_telemetry(
    *,
    candidate_id: str,
    archive_sha: str = "a" * 64,
    archive_zip_bytes: int = 177_500,
    receiver_proof_path: Path | None = None,
    cache_quality_report_path: Path | None = None,
    cache_quality_passed: bool = True,
    profile_ready: bool = True,
    sections: tuple[dict[str, object], ...] = (
        {"name": "decoder_state", "role": "decoder", "bytes": 1200},
        {"name": "latents_mid", "role": "latent", "bytes": 64},
    ),
) -> dict:
    return {
        "schema": "hinerv_archive_section_telemetry.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "profile_ready": bool(profile_ready),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": int(archive_zip_bytes),
        "inner_payload_bytes": int(archive_zip_bytes) - 42,
        "section_payload_bytes": sum(int(row.get("bytes") or 0) for row in sections),
        "sections": list(sections),
        "sections_with_zip_overhead": [
            *list(sections),
            {
                "name": "archive_zip_overhead",
                "role": "container_overhead",
                "bytes": 42,
            },
        ],
        "num_pairs": 600,
        "hard_byte_ceiling": 178_000,
        "receiver_proof_status": (
            "runtime_consumption_proof_ready"
            if receiver_proof_path is not None
            else "missing"
        ),
        "receiver_proof_path": (
            receiver_proof_path.as_posix() if receiver_proof_path is not None else None
        ),
        "receiver_cache_quality_report_path": (
            cache_quality_report_path.as_posix()
            if cache_quality_report_path is not None
            else None
        ),
        "receiver_cache_quality_gate_passed": bool(cache_quality_passed),
        "receiver_cache_quality_gate_verdict": (
            "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY"
            if cache_quality_passed
            else "FIT_OR_SCALE_FAILURE"
        ),
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _receiver_cache_quality_report(root: Path, *, passed: bool) -> Path:
    out = root / f"hi_nerv_receiver_cache_quality_{int(passed)}.json"
    out.write_text(
        json.dumps(
            {
                "schema": "hi_nerv_receiver_cache_quality_report.v1",
                "quality_gate_passed": bool(passed),
                "quality_gate": {
                    "schema": "mlx_cache_quality_gate.v1",
                    "fit_gate_passed": bool(passed),
                    "verdict": (
                        "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY"
                        if passed
                        else "FIT_OR_SCALE_FAILURE"
                    ),
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out


def _pr95_baseline_identity() -> dict:
    return {
        "schema": "pr95_baseline_identity.v1",
        "baseline_id": "pr95_public_hnerv_muon_control_arm",
        "baseline_identity_reusable": True,
        "selected_reusable_candidate_archive": {
            "schema": "pr95_baseline_identity_archive_record.v1",
            "source_artifact_path": "/tmp/pr95/pr95_stage8_from_public_archive_report.json",
            "path": "/tmp/pr95/byte_closed_submission/archive.zip",
            "bytes": 178_363,
            "sha256": "a" * 64,
            "runtime_path": "/tmp/pr95/runtime",
            "runtime_tree_sha256": "b" * 64,
            "candidate_type": "pr95_stage8_public_archive_candidate",
            "reusable_identity": True,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "exact_axis_status": {
            "schema": "pr95_baseline_identity_exact_axis_status.v1",
            "contest_cpu": {
                "present": False,
                "blockers": ["pr95_contest_cpu_exact_eval_missing"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "contest_cuda": {
                "present": False,
                "blockers": ["pr95_contest_cuda_exact_eval_missing"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "matched_archive_sha256": "a" * 64,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "local_cpu_mlx_work_order": {
            "schema": "pr95_baseline_local_cpu_mlx_work_order.v1",
            "ready": True,
            "local_cpu_axis_tag": "[macOS-CPU advisory]",
            "local_cpu_command_argv": [
                "uv",
                "run",
                "python",
                "experiments/contest_auth_eval.py",
                "--archive",
                "/tmp/pr95/byte_closed_submission/archive.zip",
                "--device",
                "cpu",
            ],
            "mlx_axis_tag": "[macOS-MLX research-signal]",
            "modal_dispatch_allowed": False,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "modal_dispatch_policy": {
            "schema": "pr95_baseline_modal_dispatch_policy.v1",
            "modal_dispatch_allowed": False,
            "reason": "non_frontier_control_arm_modal_dispatch_forbidden",
            "allowed_only_for": "frontier_candidate_exact_auth_eval_after_local_cpu_mlx_gates",
            "baseline_control_arm_policy": "local_cpu_and_mlx_only",
            "forbidden_work_order_blocker": "modal_reserved_for_frontier_candidates",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "paired_exact_eval_work_order": {
            "schema": "pr95_baseline_paired_exact_eval_work_order.v1",
            "ready": False,
            "modal_dispatch_allowed": False,
            "reason": "non_frontier_control_arm_modal_dispatch_forbidden",
            "allowed_only_for": "frontier_candidate_exact_auth_eval_after_local_cpu_mlx_gates",
            "output_root": "/tmp/pr95/exact_eval",
            "command_argv": [],
            "blockers": ["modal_reserved_for_frontier_candidates"],
            "superseded_by": "pr95_baseline_local_cpu_mlx_work_order.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "blockers": [
            "pr95_contest_cpu_exact_eval_missing",
            "pr95_contest_cuda_exact_eval_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _receiver_proof(root: Path, *, archive_sha: str = "a" * 64) -> Path:
    proof = root / f"receiver_proof_{archive_sha[:8]}.json"
    proof.write_text(
        json.dumps(
            {
                "schema": "compact_receiver_proof.v1",
                "archive_sha256": archive_sha,
                "runtime_consumption_proof_ready": True,
                "runtime_consumption_proof_passed": True,
                "receiver_archive_replay_verified": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return proof


def _joint_recon_weight_manifest(root: Path, *, num_pairs: int) -> Path:
    out = root / f"joint_weight_{num_pairs}"
    out.mkdir(parents=True, exist_ok=True)
    weight = out / "joint_p18_p19_recon_pixel_weight.npz"
    weight.write_bytes(b"unit-weight-bytes")
    manifest = out / "joint_p18_p19_recon_pixel_weight_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight.as_posix(),
                "weight_sha256": _sha256(weight),
                "config": {"num_pairs": int(num_pairs)},
                "metadata": {
                    "gradient_health": {
                        "status": "pass_finite",
                        "nonfinite_count": 0,
                    },
                    "training_consumption_recommended": True,
                    "blockers": [],
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _snerv_partial_compact_runner_report() -> dict:
    candidate = dict(_snerv_budget()["selected_candidates"][0])
    return {
        "schema": "compact_renderer_mlx_spine_runner.v1",
        "execute_family": "snerv",
        "mode": "executed_snerv_archive_bound_advisory_and_exported",
        "num_pairs": 2,
        "archive_bytes": 57_892,
        "archive_sha256": "f" * 64,
        "modelsize_candidate_selection": {"candidate": candidate},
        "candidate_curriculum_plan": {
            "schema": "nerv_candidate_curriculum_plan.v1",
            "family": "snerv",
            "candidate_id": candidate["candidate_id"],
            "candidate_conditioned": True,
            "byte_oracle_logging": {
                "schema": "nerv_candidate_byte_feedback.v1",
                "candidate_id": candidate["candidate_id"],
                "candidate_num_pairs": 600,
                "measured_num_pairs": 2,
                "feedback_scope": "partial_pair_advisory",
                "scope_matches_candidate": False,
                "feedback_ready": False,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 190_000,
                "measured_payload_bytes": 10_441,
                "measured_archive_bytes": 57_892,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        },
        "snerv_mlx_native_export": {
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "native_mlx_full600_campaign_ready": False,
            "scorer_loop_qat_receiver_contract_satisfied": True,
            "scorer_loop_qat_ready_for_pose_guard_gate": True,
            "scorer_loop_qat_accepted_improvement": True,
            "scorer_loop_qat_best_materialized": True,
        },
        "snerv_binary_profile": {
            "profile_written": True,
            "verdict": "snerv_payload_lf_dominant_but_archive_under_frontier",
            "charged_archive_bytes": 57_892,
            "snar1_packet_bytes": 10_824,
            "lf_payload_bytes": 6_156,
            "lf_payload_fraction_of_packet": 0.5687,
            "blockers": [],
        },
        "blockers": ["snerv_mlx_native_full600_campaign_not_ready"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
