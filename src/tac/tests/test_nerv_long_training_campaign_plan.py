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
    assert report["launchable_local_row_count"] == 0
    assert report["family_counts"] == {"hi_nerv": 2, "snerv": 1}
    assert report["source_parity_contract"]["schema"] == ("nerv_source_parity_contract.v1")
    assert report["source_parity_required_for_long_training_ready"] is True
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["source_parity_nonblocking_gaps"]

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
    assert all(row["local_mlx_launch_command_ready"] is False for row in hi_rows)
    assert all(row["local_mlx_executable"] is False for row in hi_rows)
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
    assert all(row["experiment_queue_entry"]["status"] == "disabled" for row in hi_rows)
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
    assert snerv_row["implementation_status"] == "native_rate_aware_long_training_rate_blocked"
    assert snerv_row["hard_byte_ceiling_satisfied_for_long_training"] is False
    assert snerv_row["score_lowering_gate"]["command_materialized"] is False
    assert snerv_row["score_lowering_gate"]["local_mlx_executable"] is False
    assert snerv_row["score_lowering_gate"]["prelaunch_allowed"] is False
    assert snerv_row["score_lowering_gate"]["promotion_prelaunch_allowed"] is False
    assert (
        "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training"
        in snerv_row["blockers"]
    )
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
        "snerv_optimizer_control_requires_learned_scoreaware_training_loop" not in launch_contract["queue_launch_blockers"]
    )
    assert launch_contract["queue_status_is_receiver_proof"] is False
    assert launch_contract["queue_status_is_cpu_replay_proof"] is False
    assert launch_contract["queue_status_is_exact_eval_authority"] is False
    assert launch_contract["cpu_replay_ready"] is False
    assert launch_contract["exact_gate_ready"] is False
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
    assert source_audit["official_mfu_hfr_tub_parity_proven"] is False
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
    assert launch_contract["source_bound_capacity_controls"]["schema"] == (
        "snerv_source_bound_capacity_controls.v1"
    )
    assert launch_contract["score_claim"] is False


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
    assert snerv_row["implementation_status"] == "native_rate_aware_long_training_rate_blocked"
    assert snerv_row["hard_byte_ceiling_satisfied_for_long_training"] is False
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
    assert snerv_row["local_mlx_launch_command_ready"] is True
    assert "snerv_nominal_payload_far_over_ceiling_refuse_long_training" not in snerv_row["blockers"]


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
    assert snerv_row["implementation_status"] == ("native_rate_aware_long_training_rate_blocked")
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
    assert snerv["implementation_status"] == "bounded_native_export_scorer_loop_stage_ready"
    assert snerv["command_argv"][snerv["command_argv"].index("--epochs") + 1] == "5"
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
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
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
        "snerv_receiver_proven_archive_over_hard_byte_ceiling_observed_demote_only"
        in demoted["blockers"]
    )
    assert "snerv_receiver_proven_archive_over_hard_byte_ceiling" in demoted[
        "blockers"
    ]
    feedback = demoted["curriculum_plan"]["byte_oracle_logging"]
    assert feedback["archive_under_hard_byte_ceiling"] is False
    assert feedback["archive_over_hard_byte_ceiling_bytes"] == 228_828


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
                "observed_segnet_distillation_weight": 2.0,
                "recommended_segnet_distillation_weight": 4.0,
                "recommended_launch_mutations": [
                    "increase_segnet_distillation_weight_from_full_video_mlx_response",
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
    assert adjustment["segnet_distillation_weight"] == 4.0
    assert argv[argv.index("--segnet-distillation-weight") + 1] == "4"
    assert "increase_segnet_distillation_weight_from_full_video_mlx_response" in (
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
    assert feedback["byte_feedback_source"] == "snerv_mlx_native_file_backed_export"
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
    assert set(DEFAULT_OPTIMIZER_KINDS) == set(SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS)
    assert DEFAULT_OPTIMIZER_KINDS[:5] == (
        "pact_muon_adamw",
        "adamw",
        "muon",
        "lion",
        "adamax",
    )


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
    assert hi_rows[0]["optimizer_policy"]["requested_policy"] == "native_optimizer"
    assert report["optimizer_control_policy"]["default_optimizer_kind"] == ("pact_muon_adamw")
    assert report["optimizer_control_policy"]["default_optimizer_backend"] == ("tac.local_acceleration.pr95_hnerv_mlx")


def test_build_long_training_campaign_plan_cli_writes_outputs(tmp_path: Path) -> None:
    hinerv = tmp_path / "hinerv_budget.json"
    snerv = tmp_path / "snerv_budget.json"
    out_json = tmp_path / "campaign.json"
    out_md = tmp_path / "campaign.md"
    out_queue = tmp_path / "campaign_queue.json"
    feedback_jsonl = tmp_path / "feedback.jsonl"
    waterfill_bundle = tmp_path / "hinerv_archive_ladder_waterfill.json"
    proof_path = _receiver_proof(tmp_path, archive_sha="a" * 64)
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
            "--epochs",
            "16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-queue",
            str(out_queue),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["campaign_row_count"] == 2
    assert payload["planner_row_queue_artifact_path"] == out_queue.as_posix()
    assert payload["candidate_feedback_row_count"] == 1
    assert payload["decoder_weight_waterfill_attached_row_count"] == 1
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
    assert "--planner-row-queue-artifact" in hi["command_argv"]
    assert hi["command_argv"][hi["command_argv"].index("--planner-row-queue-artifact") + 1] == out_queue.as_posix()
    assert "--recon-pixel-weight-path" in hi["command_argv"]
    assert "--decoder-weight-waterfill-plan-json" in hi["command_argv"]
    waterfill_sidecar = Path(hi["command_argv"][hi["command_argv"].index("--decoder-weight-waterfill-plan-json") + 1])
    assert waterfill_sidecar.is_file()
    assert waterfill_sidecar.parent.name == "decoder_weight_waterfill_sidecars"
    assert hi["decoder_weight_waterfill_plan"]["source_path"] == (waterfill_bundle.resolve(strict=False).as_posix())
    snerv_row = next(row for row in payload["campaign_rows"] if row["family"] == "snerv")
    assert snerv_row["candidate_feedback"]["candidate_id"] == _snerv_candidate_id()
    assert "partial_pair_byte_feedback_only" in snerv_row["blockers"]
    assert payload["experiment_queue"]["schema"] == "experiment_queue.v1"
    assert payload["experiment_queue_id"] == (f"nerv_long_training_campaign_{out_json.stem}.v1")
    queue = json.loads(out_queue.read_text(encoding="utf-8"))
    assert queue == payload["experiment_queue"]
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
    export_dir.mkdir(parents=True)
    snerv_export_dir.mkdir(parents=True)
    candidate = dict(_hinerv_budget()["selected_candidates"][0])
    snerv_candidate = dict(_snerv_budget()["selected_candidates"][0])
    export_report = export_dir / "export_report.json"
    snerv_export_report = snerv_export_dir / "snerv_checkpoint_archive_export.json"
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
) -> dict:
    return {
        "schema": "snerv_lf_payload_archive_recode.v1",
        "mode": mode,
        "source_packet": {"bytes": source_packet_bytes, "sha256": "a" * 64},
        "candidate_packet": {"bytes": candidate_packet_bytes, "sha256": "b" * 64},
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
