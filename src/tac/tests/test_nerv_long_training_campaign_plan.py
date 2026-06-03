# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from comma_lab.scheduler.experiment_queue import load_queue_definition
from tac.analysis import nerv_long_training_campaign_plan as plan_module
from tac.analysis.nerv_long_training_campaign_plan import (
    DEFAULT_OPTIMIZER_KINDS,
    HINERV_POSE_INSTABILITY_LOW_LR_FLOOR,
    NervLongTrainingCampaignPlanError,
    build_nerv_long_training_campaign_plan,
    render_nerv_long_training_campaign_plan_markdown,
)
from tac.substrates._shared.mlx_score_aware.adapter import (
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
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
        "future_snerv_learned_scoreaware_decoder_rows_after_binding",
    ]
    assert report["optimizer_control_policy"]["does_not_apply_to"] == [
        "snerv_current_closed_form_native_export_and_scorer_loop_qat_rows"
    ]
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
    assert snerv["optimizer_kind"] is None
    assert qat_flags.issubset(set(snerv["command_argv"]))
    assert snerv["coder_qat_control"]["quant_bits"] == snerv["quant_bits"]
    assert snerv["coder_qat_control"]["c1a_entropy_weight"] == pytest.approx(1.0e-4)
    assert snerv["coder_qat_control"]["ready_for_exact_eval_dispatch"] is False
    assert snerv["optimizer_control"]["optimizer_kind"] is None
    assert snerv["optimizer_control"]["backend"] == (
        "mlx_target_hydration_numpy_closed_form_decoder_fit_plus_scorer_loop_qat"
    )
    assert snerv["optimizer_control"]["pact_muon_adamw_default_inherited"] is False
    assert snerv["optimizer_control"]["score_claim"] is False
    assert "snerv_optimizer_control_requires_learned_scoreaware_training_loop" in snerv["blockers"]
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
    assert snerv_row["local_mlx_launch_command_ready"] is True
    assert snerv_row["score_lowering_gate"]["local_mlx_executable"] is True
    assert snerv_row["score_lowering_gate"]["prelaunch_allowed"] is True
    assert snerv_row["score_lowering_gate"]["promotion_prelaunch_allowed"] is False
    assert "snerv_pr95_staged_curriculum_missing" in snerv_row["score_lowering_gate"]["prelaunch_blockers"]
    assert snerv_row["cpu_replay_ready"] is False
    assert snerv_row["exact_gate_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"
    assert snerv_row["experiment_queue_entry"]["blocked"] is True
    launch_contract = snerv_row["experiment_queue_entry"]["launch_authority_contract"]
    assert launch_contract["schema"] == ("nerv_long_training_queue_launch_authority_contract.v1")
    assert launch_contract["queue_status_is_local_mlx_plan"] is True
    assert launch_contract["queue_status_is_runnable_plan"] is False
    assert (
        "snerv_optimizer_control_requires_learned_scoreaware_training_loop" in launch_contract["queue_launch_blockers"]
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
    assert snerv_row["source_bound_capacity_control_blockers"]
    assert snerv_row["local_mlx_launch_command_ready"] is False
    assert snerv_row["experiment_queue_entry"]["status"] == "disabled"
    assert snerv_row["implementation_status"] == "source_bound_capacity_controls_incomplete"


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
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" not in hi["blockers"]
    assert report["decoder_weight_waterfill_source_count"] == 1
    assert report["decoder_weight_waterfill_attached_row_count"] == 1
    assert hi["score_claim"] is False


def test_long_training_campaign_plan_admits_receiver_proven_hinerv_waterfill(
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
    assert "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted" not in hi["blockers"]


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
    assert "partial_pair_byte_feedback_only" in snerv["blockers"]
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
    assert payload["candidate_feedback_row_count"] == 1
    assert payload["decoder_weight_waterfill_attached_row_count"] == 1
    hi = next(row for row in payload["campaign_rows"] if row["family"] == "hi_nerv")
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
    assert queue["queue_id"] == f"nerv_long_training_campaign_{out_json.stem}.v1"
    assert queue["experiments"][0]["steps"][0]["postconditions"]
    snerv_exp = next(exp for exp in queue["experiments"] if exp["family"] == "snerv")
    assert snerv_exp["blocked"] is True
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
                "decoder_feature_count": 16,
                "nominal_total_payload_bytes": 190_000,
                "nominal_under_ceiling": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_candidate_id() -> str:
    return str(_snerv_budget()["selected_candidates"][0]["candidate_id"])


def _decoder_weight_waterfill_plan(
    *,
    candidate_id: str,
    receiver_proof_status: str = "missing",
) -> dict:
    receiver_ready = receiver_proof_status in {
        "runtime_consumption_proof_ready",
        "receiver_proof_valid",
        "runtime_consumption_proof_passed",
        "satisfied",
        "valid",
        "passed",
    }
    blockers = [] if receiver_ready else ["receiver_proof_not_satisfied"]
    return {
        "schema": "nerv_decoder_weight_waterfill.v1",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "group_count": 2,
        "full_video_coverage": True,
        "receiver_proof_status": receiver_proof_status,
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
