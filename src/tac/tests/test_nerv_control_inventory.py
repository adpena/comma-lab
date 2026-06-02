# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.nerv_control_inventory import (
    NERV_CONTROL_INVENTORY_SCHEMA,
    build_nerv_control_inventory,
    render_nerv_control_inventory_markdown,
)
from tools import build_nerv_control_inventory as inventory_tool
from tools.build_nerv_control_inventory import main as inventory_tool_main

REPO = Path(__file__).resolve().parents[3]


def test_nerv_control_inventory_tracks_hi_nerv_snerv_and_cross_stack_controls() -> None:
    report = build_nerv_control_inventory(repo_root=REPO)

    assert report["schema"] == NERV_CONTROL_INVENTORY_SCHEMA
    assert report["focus_families"] == ["hi_nerv", "snerv"]
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["objective_authority"]["objective"] == "contest_auth_eval_scorer_only"
    assert "SSIM" in report["objective_authority"]["forbidden_selection_terms"]
    assert report["rate_constraint"]["constraint_id"] == "fixed_contest_byte_price"
    assert report["rate_constraint"]["contest_byte_price_score_per_byte"] > 0.0
    assert "waterfilled_int8_int4_int2_zero_allocation" in report["rate_constraint"][
        "required_large_model_escape_hatches"
    ]

    control_ids = {row["control_id"] for row in report["control_rows"]}
    assert {
        "hi_nerv_hierarchical_capacity",
        "snerv_frequency_split",
        "hnerv_modelsize_control",
        "rnerv_config_optimizer",
        "sr_nerv_lowres_receiver_axis",
        "ffnerv_flow_temporal_redundancy",
        "nervplusplus_decoder_efficiency_blocks",
        "vq_c3_cool_chic_latent_codebook",
        "inverse_steganalysis_saliency_stack",
        "master_gradient_xray_stack",
        "full_video_vjp_master_gradient_authority",
        "bitmask_and_zero_packing",
        "receiver_exact_custody_gate",
    }.issubset(control_ids)

    gap_ids = {row["gap_id"] for row in report["binding_gap_rows"]}
    assert {
        "measured_hi_nerv_modelsize_budget_ladder",
        "full_video_decoder_weight_saliency_replay_for_hi_nerv_archive_rows",
        "mlx_native_snerv_train_export",
        "decoder_weight_waterfill_plan_for_snerv_receiver_rows",
        "push_saliency_into_hi_nerv_weight_groups_and_snerv_wavelet_groups",
        "runnable_rnerv_style_config_search_over_hi_nerv_snerv_controls",
        "receiver_closed_scorer_preserving_sr_axis_for_hi_nerv_and_snerv",
        "byte_priced_flow_or_pose_support_tokens_for_hi_nerv_snerv",
        "rate_priced_nervplusplus_block_ablation_for_hi_nerv_snerv",
        "full_video_section_value_for_vq_codebook_indices",
        "full_video_vjp_bundle_as_budget_spend_prerequisite",
        "all_compact_carrier_emitters_on_shared_archive_bound_contract",
    }.issubset(gap_ids)

    source_ids = {row["id"] for row in report["upstream_sources_checked"]}
    assert {
        "comma_leaderboard_video_compression",
        "pr95_public_control",
        "hnerv_official",
        "hinerv_official",
        "snerv_official",
        "snerv_scalable_disambiguation",
        "sr_nerv_paper",
        "rnerv_paper",
        "ffnerv_paper",
        "ffnerv_project",
        "nervplusplus_paper",
        "c3_paper",
        "cool_chic_docs",
        "nvrc_paper",
        "rnerv_vinrb_oss",
        "boost_nerv_official",
        "c3_project",
    }.issubset(source_ids)
    assert report["source_review_policy"]["schema"] == "nerv_source_review_policy.v1"
    assert report["source_review_policy"]["score_claim"] is False
    hard_rules = set(report["source_review_policy"]["hard_rules"])
    assert any("SNeRV means spectra-preserving" in rule for rule in hard_rules)
    assert any("HiNeRV and HNeRV are distinct" in rule for rule in hard_rules)
    assert any("PR95 is the same-axis public control arm" in rule for rule in hard_rules)
    assert any("CPU, CUDA, and MLX observations stay separate" in rule for rule in hard_rules)
    transfers = {
        row["from_family"]: row for row in report["stack_transfer_matrix"]
    }
    assert "C1a entropy-shaping regularizer before brotli" in transfers[
        "PR95/HNeRV"
    ]["transfers"]
    assert "CPU/CUDA axis custody" in transfers["PR95/HNeRV"]["guard"]

    surfaces = report["local_binding_surfaces"]
    assert "src/tac/analysis/score_exact_saliency.py" in surfaces["scorer_and_saliency"]
    assert "src/tac/master_gradient.py" in surfaces["xray_and_master_gradient"]
    assert "src/tac/analysis/mlx_cache_quality_gate.py" in surfaces["newly_required_gates"]
    assert (
        "src/tac/substrates/_shared/mlx_score_aware/modelsize_budget_plan.py"
        in surfaces["newly_required_gates"]
    )
    assert "tools/profile_pact_nerv_selector_v3_mlx_section_value.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/analysis/nerv_decoder_weight_waterfill.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/analysis/hinerv_archive_ladder_waterfill.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/analysis/hinerv_archive_backend_drift.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/analysis/snerv_trained_ladder_waterfill.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/analysis/snerv_waterfill_mode_assignment.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "tools/build_nerv_decoder_weight_waterfill_plan.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "tools/build_hinerv_archive_ladder_waterfill.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "tools/build_hinerv_archive_backend_drift.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "tools/build_snerv_trained_ladder_waterfill.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/analysis/hinerv_decoder_weight_saliency_replay.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "tools/build_hinerv_decoder_weight_saliency_replay.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "tools/build_snerv_waterfill_mode_assignment.py" in surfaces[
        "section_value_and_codebook"
    ]
    assert "src/tac/submission_packet/paired_auth_eval.py" in surfaces[
        "receiver_and_exact_custody"
    ]
    assert "experiments/train_substrate_hi_nerv_mlx_local.py" in surfaces[
        "nerv_carriers"
    ]
    assert "experiments/train_substrate_snerv_scorer_loop_local.py" in surfaces[
        "nerv_carriers"
    ]
    assert report["runner_spend_rule"]["score_claim"] is False
    assert report["runner_policy"]["bounded_runner_must_select_from_inventory_rows"]
    ladder = report["modelsize_ladder"]
    assert ladder["schema"] == "nerv_modelsize_ladder.v1"
    assert ladder["score_claim"] is False
    ladder_families = {row["family"]: row for row in ladder["family_rows"]}
    assert {"hi_nerv", "snerv"} == set(ladder_families)
    assert ladder_families["hi_nerv"]["marginal_gates"]
    assert ladder_families["snerv"]["marginal_gates"]

    sweep = report["implementation_sweep"]
    assert sweep["status"] == "implementation_sweep_completed_false_authority"
    stack_rows = {row["family"]: row for row in sweep["stack_rows"]}
    assert {
        "hi_nerv_official_symbol_parity_map_missing",
        "hi_nerv_full600_receiver_proven_candidate_missing",
        "hi_nerv_missing_measured_config_family_ladder",
        "hi_nerv_missing_integer_bitstream_q_roundtrip",
        "hi_nerv_full_video_decoder_weight_saliency_replay_missing",
    }.issubset(set(stack_rows["hi_nerv"]["blocking_gaps"]))
    assert {
        "snerv_official_symbol_parity_map_missing",
        "snerv_scorer_loop_decoder_qat_full_video_missing",
        "snerv_missing_mfu_blocks",
        "snerv_missing_measured_fc_dim_modelsize_ladder",
    }.issubset(set(stack_rows["snerv"]["blocking_gaps"]))
    assert sweep["design_memo_index"]["hi_nerv"]["memo_count"] > 0
    assert sweep["design_memo_index"]["snerv"]["memo_count"] > 0
    assert sweep["design_memo_index"]["hi_nerv"]["memo_paths_are_complete"] is True
    assert sweep["design_memo_index"]["snerv"]["truncated"] is False
    assert sweep["design_memo_index"]["hi_nerv"]["memo_rows"][0]["sha256"]
    assert stack_rows["hi_nerv"]["official_feature_rows"]
    assert stack_rows["snerv"]["official_feature_rows"]
    assert sweep["score_claim"] is False

    markdown = render_nerv_control_inventory_markdown(report)
    assert "## Implementation Sweep" in markdown
    assert "## Model-Size Ladder" in markdown
    assert "full_video_vjp_master_gradient_authority" in markdown


def test_nerv_control_inventory_can_focus_on_snerv_plus_cross_stack_only() -> None:
    report = build_nerv_control_inventory(focus_families=("snerv",))

    applies_to = {row["applies_to"] for row in report["control_rows"]}
    assert applies_to <= {"snerv", "cross_stack"}
    assert "snerv" in applies_to
    assert "cross_stack" in applies_to
    work_order_ids = {row["work_order_id"] for row in report["recommended_next_work_orders"]}
    assert "decoder_weight_waterfill_plan_for_snerv_receiver_rows" in work_order_ids
    assert report["score_claim"] is False
    assert [row["family"] for row in report["modelsize_ladder"]["family_rows"]] == [
        "snerv"
    ]
    assert report["measured_archive_size_ladders"] == {}
    assert report["implementation_sweep"]["status"] == "repo_root_not_supplied"


def test_nerv_control_inventory_accepts_measured_hinerv_archive_size_ladder() -> None:
    archive_ladder = {
        "schema": "hinerv_archive_size_ladder.v1",
        "report_path": ".omx/research/hinerv_archive_size_ladder_fake.json",
        "output_dir": "/Volumes/VertigoDataTier/pact/fake",
        "decoder_codec": "int8_mixed",
        "row_count": 1,
        "selection_rule": "adaptive quantization and waterfilling required",
        "required_allocator_bindings": [
            "adaptive_quantization_by_decoder_weight_group",
            "waterfill_group_bits_against_fixed_contest_byte_price",
            "inverse_steg_saliency_decoder_weight_binding",
        ],
        "archive_rows": [
            {
                "row_id": "hi_nerv_local_tiny",
                "archive_bytes": 123,
                "archive_sha256": "a" * 64,
                "archive_path": "/Volumes/VertigoDataTier/pact/fake/archive.zip",
                "runtime_consumption_proof_ready": None,
                "blockers": ["receiver_proof_not_executed_for_archive_size_ladder"],
            }
        ],
        "marginal_archive_gates": [],
        "blockers": ["hinerv_archive_size_ladder_false_authority_no_nonrate_score"],
    }

    report = build_nerv_control_inventory(
        focus_families=("hi_nerv",),
        hinerv_archive_size_ladder_report=archive_ladder,
    )

    measured = report["measured_archive_size_ladders"]["hi_nerv"]
    assert measured["schema"] == "hinerv_archive_size_ladder.v1"
    assert measured["score_claim"] is False
    assert measured["row_count"] == 1
    assert measured["archive_rows"][0]["archive_bytes"] == 123
    assert "inverse_steg_saliency_decoder_weight_binding" in measured[
        "required_allocator_bindings"
    ]


def test_nerv_control_inventory_accepts_hinerv_archive_ladder_waterfill_report() -> None:
    waterfill_report = {
        "schema": "hinerv_archive_ladder_waterfill.v1",
        "report_path": ".omx/research/hinerv_archive_ladder_waterfill_fake.json",
        "row_count": 1,
        "full_video_coverage": True,
        "rows": [
            {
                "row_id": "hi_nerv_local_tiny",
                "archive_bytes": 123,
                "archive_sha256": "a" * 64,
                "state_npz_artifact_sha256": "b" * 64,
                "waterfill_summary": {
                    "group_count": 2,
                    "total_selected_byte_delta": -16,
                },
                "archive_ladder_replay_command_axis_tag": (
                    "[planning/control:false-authority]"
                ),
                "archive_ladder_replay_command_argv": [
                    ".venv/bin/python",
                    "tools/build_hinerv_archive_size_ladder.py",
                    "--row-id",
                    "hi_nerv_local_tiny",
                ],
                "archive_ladder_replay_command_hint": (
                    ".venv/bin/python tools/build_hinerv_archive_size_ladder.py "
                    "--row-id hi_nerv_local_tiny"
                ),
                "archive_ladder_replay_output_dir": (
                    "/Volumes/VertigoDataTier/pact/"
                    "hinerv_archive_ladder_waterfill_replay/hi_nerv_local_tiny"
                ),
                "blockers": ["decoder_weight_saliency_missing_for_some_groups"],
            }
        ],
        "section_value_rows": [{"row_id": "r0"}],
        "byte_price_plan": {"schema": "compact_nerv_byte_price_controller.v1"},
        "blockers": ["decoder_weight_saliency_replay_required_for_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("hi_nerv",),
        hinerv_archive_ladder_waterfill_report=waterfill_report,
    )

    measured = report["decoder_weight_waterfill_reports"]["hi_nerv"]
    assert measured["schema"] == "hinerv_archive_ladder_waterfill.v1"
    assert measured["score_claim"] is False
    assert measured["row_count"] == 1
    assert measured["section_value_row_count"] == 1
    row = measured["waterfill_rows"][0]
    assert row["archive_ladder_replay_command_axis_tag"] == (
        "[planning/control:false-authority]"
    )
    assert row["archive_ladder_replay_command_argv"] == [
        ".venv/bin/python",
        "tools/build_hinerv_archive_size_ladder.py",
        "--row-id",
        "hi_nerv_local_tiny",
    ]
    assert row["archive_ladder_replay_output_dir"].endswith(
        "/hinerv_archive_ladder_waterfill_replay/hi_nerv_local_tiny"
    )
    assert measured["waterfill_rows"][0]["waterfill_summary"]["group_count"] == 2


def test_nerv_control_inventory_accepts_hinerv_replay_actuator_report() -> None:
    replay_report = {
        "schema": "hinerv_archive_ladder_replay_actuator.v1",
        "report_path": ".omx/research/hinerv_archive_ladder_replay_actuator_fake.json",
        "execution_requested": True,
        "load_existing_requested": False,
        "row_count": 1,
        "loaded_replay_report_count": 1,
        "receiver_proof_ready_row_count": 1,
        "archive_bytes_by_row_id": {"hi_nerv_local_tiny": 134908},
        "rows": [
            {
                "row_id": "hi_nerv_local_tiny",
                "status": "executed_report_loaded_false_authority",
                "archive_bytes": 134908,
                "archive_sha256": "a" * 64,
                "archive_path": "/Volumes/VertigoDataTier/pact/tiny/archive.zip",
                "submission_dir": "/Volumes/VertigoDataTier/pact/tiny/submission",
                "spine_manifest_path": "/Volumes/VertigoDataTier/pact/tiny/spine.json",
                "receiver_proof_path": "/Volumes/VertigoDataTier/pact/tiny/proof.json",
                "decoder_weight_waterfill_plan_path": "/Volumes/VertigoDataTier/pact/tiny/plan.json",
                "replay_report_path": ".omx/research/tiny_replay.json",
                "replay_report_sha256": "b" * 64,
                "receiver_proof_ready": True,
                "archive_export_backend_counts": {"mlx": 1},
                "blockers": ["hinerv_archive_size_row_has_no_nonrate_score"],
            }
        ],
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("hi_nerv",),
        hinerv_archive_ladder_replay_actuator_report=replay_report,
    )

    measured = report["archive_ladder_replay_actuator_reports"]["hi_nerv"]
    assert measured["schema"] == "hinerv_archive_ladder_replay_actuator.v1"
    assert measured["score_claim"] is False
    assert measured["row_count"] == 1
    assert measured["receiver_proof_ready_row_count"] == 1
    row = measured["replay_rows"][0]
    assert row["archive_bytes"] == 134908
    assert row["submission_dir"].endswith("/tiny/submission")
    assert row["spine_manifest_path"].endswith("/tiny/spine.json")
    assert row["receiver_proof_path"].endswith("/tiny/proof.json")
    assert row["decoder_weight_waterfill_plan_path"].endswith("/tiny/plan.json")
    markdown = render_nerv_control_inventory_markdown(report)
    assert "## Archive Ladder Replay Actuators" in markdown


def test_nerv_control_inventory_accepts_hinerv_archive_backend_drift_report() -> None:
    drift_report = {
        "schema": "hinerv_archive_backend_drift.v1",
        "status": "local_backend_drift_within_tolerance_false_authority",
        "report_path": ".omx/research/hinerv_archive_backend_drift_fake.json",
        "reference_json_path": ".omx/research/fallback.json",
        "reference_json_sha256": "a" * 64,
        "candidate_json_path": ".omx/research/mlx.json",
        "candidate_json_sha256": "b" * 64,
        "reference_label": "pytorch_portable_fallback",
        "candidate_label": "mlx_metal",
        "row_count": 4,
        "matched_row_count": 4,
        "byte_ready_row_count": 4,
        "reference_receiver_proof_ready_row_count": 4,
        "candidate_receiver_proof_ready_row_count": 4,
        "max_abs_byte_delta_allowed": 1024,
        "max_abs_byte_delta_observed": 81,
        "sum_byte_delta_candidate_minus_reference": -194,
        "sum_rate_score_delta_candidate_minus_reference": -0.000129185,
        "within_byte_drift_tolerance": True,
        "local_dev_velocity_ready": True,
        "ready_backend_for_local_iteration": "mlx_metal",
        "blockers": ["hinerv_archive_backend_drift_local_dev_velocity_only"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("hi_nerv",),
        hinerv_archive_backend_drift_report=drift_report,
    )

    measured = report["archive_backend_drift_reports"]["hi_nerv"]
    assert measured["schema"] == "hinerv_archive_backend_drift.v1"
    assert measured["score_claim"] is False
    assert measured["local_dev_velocity_ready"] is True
    assert measured["max_abs_byte_delta_observed"] == 81
    assert measured["candidate_json_sha256"] == "b" * 64
    assert "hinerv_archive_backend_drift_local_dev_velocity_only" in measured[
        "blockers"
    ]
    markdown = render_nerv_control_inventory_markdown(report)
    assert "## Archive Backend Drift" in markdown


def test_nerv_control_inventory_accepts_snerv_trained_ladder_waterfill_report() -> None:
    waterfill_report = {
        "schema": "snerv_trained_ladder_waterfill.v1",
        "report_path": ".omx/research/snerv_trained_ladder_waterfill_fake.json",
        "row_count": 1,
        "source_status": "trained_ladder_row_blocked",
        "source_verdict": "NO_GO_HARVEST_INPUT__TRAINED_ROW_PROOF_INCOMPLETE",
        "rows": [
            {
                "row_id": "snerv_local_tiny",
                "archive_bytes": 123,
                "archive_sha256": "a" * 64,
                "archive_sha256_actual": "a" * 64,
                "receiver_codec_mode": "contest_archive_zip",
                "decoder_precision_mode": "mixed_magnitude_symmetric",
                "decoder_payload_schema": "snerv_decoder_payload.v3",
                "decoder_state_group_count": 3,
                "waterfill_summary": {
                    "group_count": 3,
                    "total_selected_byte_delta": 0,
                },
                "blockers": ["decoder_weight_saliency_missing_for_some_groups"],
            }
        ],
        "section_value_rows": [{"row_id": "r0"}],
        "blockers": ["decoder_weight_saliency_replay_required_for_authority"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("snerv",),
        snerv_trained_ladder_waterfill_report=waterfill_report,
    )

    measured = report["decoder_weight_waterfill_reports"]["snerv"]
    assert measured["schema"] == "snerv_trained_ladder_waterfill.v1"
    assert measured["score_claim"] is False
    assert measured["row_count"] == 1
    assert measured["section_value_row_count"] == 1
    assert measured["waterfill_rows"][0]["decoder_payload_schema"] == (
        "snerv_decoder_payload.v3"
    )
    assert measured["waterfill_rows"][0]["waterfill_summary"]["group_count"] == 3


def test_nerv_control_inventory_accepts_hinerv_decoder_weight_saliency_report() -> None:
    saliency_report = {
        "schema": "hinerv_decoder_weight_saliency_replay.v1",
        "report_path": ".omx/research/hinerv_decoder_weight_saliency_fake.json",
        "row_count": 1,
        "full_video_coverage": False,
        "scorer_source": "real_upstream_differentiable_scorers",
        "pair_schedule": {"max_pairs": 1, "start_pair": 0, "pair_stride": 1},
        "saliency_by_name": {"head_rgb_1.weight": 0.125},
        "saliency_rows": [
            {
                "group_name": "head_rgb_1.weight",
                "decoder_weight_saliency": 0.125,
                "score_saliency": 0.125,
            }
        ],
        "blockers": ["full_video_coverage_missing"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("hi_nerv",),
        hinerv_decoder_weight_saliency_report=saliency_report,
    )

    measured = report["decoder_weight_saliency_replays"]["hi_nerv"]
    assert measured["schema"] == "hinerv_decoder_weight_saliency_replay.v1"
    assert measured["score_claim"] is False
    assert measured["row_count"] == 1
    assert measured["full_video_coverage"] is False
    assert measured["saliency_group_count"] == 1
    assert measured["saliency_rows"][0]["decoder_weight_saliency"] == 0.125


def test_nerv_control_inventory_accepts_snerv_waterfill_mode_assignment_report() -> None:
    mode_assignment_report = {
        "schema": "snerv_waterfill_mode_assignment.v1",
        "report_path": ".omx/research/snerv_mode_assignment_fake.json",
        "row_count": 1,
        "local_advisory_probe_ready_row_count": 1,
        "receiver_mode_export_ready_row_count": 0,
        "rows": [
            {
                "row_id": "snerv_local_tiny",
                "archive_sha256": "a" * 64,
                "decoder_payload_schema": "snerv_decoder_payload.v3",
                "mode_count": 3,
                "mode_plan_cli_arg": "fp16,fp16,fp16",
                "mode_histogram": {"fp16": 3},
                "ready_for_local_advisory_probe": True,
                "ready_for_receiver_mode_export": False,
                "blockers": [
                    "fp32_protect_downgraded_to_fp16_requires_receiver_replay"
                ],
            }
        ],
        "blockers": ["mode_assignment_is_false_authority_until_receiver_replay"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("snerv",),
        snerv_waterfill_mode_assignment_report=mode_assignment_report,
    )

    measured = report["decoder_mode_assignment_reports"]["snerv"]
    assert measured["schema"] == "snerv_waterfill_mode_assignment.v1"
    assert measured["score_claim"] is False
    assert measured["row_count"] == 1
    assert measured["local_advisory_probe_ready_row_count"] == 1
    assert measured["receiver_mode_export_ready_row_count"] == 0
    assert measured["assignment_rows"][0]["mode_plan_cli_arg"] == "fp16,fp16,fp16"


def test_nerv_control_inventory_accepts_snerv_decoder_mode_probe_report() -> None:
    probe_report = {
        "schema": "snerv_decoder_mode_assignment_probe.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "n_pairs": 1,
        "levels": 1,
        "best_plan_label": "explicit_fp163",
        "best_plan_score_linf_advisory": 3.58,
        "candidates": [
            {
                "label": "magnitude_heuristic",
                "modes": None,
                "mode_assignment_source": "magnitude_heuristic",
                "mode_histogram": {"fp16": 2, "int4": 1},
                "archive_bytes_total": 550370,
                "decoder_bytes": 1028,
                "d_seg_mean_linf": 0.0139,
                "d_pose_mean_linf": 0.333,
                "score_linf": 3.585,
                "receiver_archive_replay_verified": True,
                "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"],
            },
            {
                "label": "explicit_fp163",
                "modes": ["fp16", "fp16", "fp16"],
                "mode_assignment_source": "explicit",
                "mode_histogram": {"fp16": 3},
                "archive_bytes_total": 550390,
                "decoder_bytes": 1030,
                "d_seg_mean_linf": 0.0139,
                "d_pose_mean_linf": 0.332,
                "score_linf": 3.58,
                "receiver_archive_replay_verified": True,
                "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"],
            },
        ],
        "blockers": ["macos_cpu_advisory_only"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("snerv",),
        snerv_decoder_mode_probe_report=probe_report,
    )

    measured = report["decoder_mode_probe_reports"]["snerv"]
    assert measured["schema"] == "snerv_decoder_mode_assignment_probe.v1"
    assert measured["score_claim"] is False
    assert measured["best_plan_label"] == "explicit_fp163"
    assert measured["candidate_count"] == 2
    assert measured["candidate_rows"][1]["modes"] == ["fp16", "fp16", "fp16"]


def test_nerv_control_inventory_accepts_snerv_scorer_loop_qat_report() -> None:
    qat_report = {
        "schema": "snerv_scorer_loop_qat_local_trainer.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "research_json_path": ".omx/research/snerv_scorer_loop_qat_fake.json",
        "result_path": "/Volumes/VertigoDataTier/pact/snerv/result.json",
        "result_sha256": "a" * 64,
        "n_pairs": 1,
        "levels": 1,
        "wavelet": "haar",
        "qat_bits": 8,
        "search_mode": "random_signed",
        "scorer_loop_evaluations": 3,
        "baseline_archive_bytes": 1000,
        "best_archive_bytes": 1000,
        "baseline_score_linf": 0.9,
        "best_score_linf": 0.9,
        "accepted_improvement": False,
        "ready_for_pose_guard_gate": False,
        "receiver_contract_satisfied": True,
        "blockers": ["local_smoke_only_not_full_600_pairs"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    report = build_nerv_control_inventory(
        focus_families=("snerv",),
        snerv_scorer_loop_qat_report=qat_report,
    )

    measured = report["snerv_scorer_loop_qat_reports"]["snerv"]
    assert measured["schema"] == "snerv_scorer_loop_qat_local_trainer.v1"
    assert measured["score_claim"] is False
    assert measured["n_pairs"] == 1
    assert measured["accepted_improvement"] is False
    assert measured["result_sha256"] == "a" * 64
    markdown = render_nerv_control_inventory_markdown(report)
    assert "## SNeRV Scorer-Loop QAT" in markdown


def test_build_nerv_control_inventory_cli_accepts_hinerv_waterfill_report(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "waterfill.json"
    output_json = tmp_path / "inventory.json"
    waterfill_path.write_text(
        """
        {
          "schema": "hinerv_archive_ladder_waterfill.v1",
          "report_path": ".omx/research/hinerv_archive_ladder_waterfill_fake.json",
          "row_count": 1,
          "full_video_coverage": true,
          "rows": [
            {
              "row_id": "hi_nerv_local_tiny",
              "archive_bytes": 123,
              "archive_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "state_npz_artifact_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "waterfill_summary": {"group_count": 2},
              "blockers": ["decoder_weight_saliency_missing_for_some_groups"]
            }
          ],
          "section_value_rows": [{"row_id": "r0"}],
          "byte_price_plan": {"schema": "compact_nerv_byte_price_controller.v1"},
          "blockers": ["decoder_weight_saliency_replay_required_for_authority"],
          "score_claim": false,
          "ready_for_exact_eval_dispatch": false
        }
        """,
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "hi_nerv",
            "--repo-root",
            str(REPO),
            "--hinerv-archive-ladder-waterfill-json",
            str(waterfill_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["decoder_weight_waterfill_reports"]["hi_nerv"]["row_count"] == 1
    assert payload["score_claim"] is False


def test_build_nerv_control_inventory_cli_auto_discovers_latest_waterfill_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    output_json = tmp_path / "inventory.json"
    (research / "hinerv_archive_ladder_waterfill_0001.json").write_text(
        json.dumps({"schema": "wrong"}),
        encoding="utf-8",
    )
    latest = research / "hinerv_archive_ladder_waterfill_9999.json"
    latest.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "report_path": ".omx/research/hinerv_archive_ladder_waterfill_9999.json",
                "row_count": 1,
                "full_video_coverage": True,
                "rows": [
                    {
                        "row_id": "hi_nerv_auto",
                        "archive_bytes": 321,
                        "archive_sha256": "a" * 64,
                        "state_npz_artifact_sha256": "b" * 64,
                        "waterfill_summary": {"group_count": 5},
                        "blockers": [
                            "decoder_weight_saliency_missing_for_some_groups"
                        ],
                    }
                ],
                "section_value_rows": [{"row_id": "r0"}],
                "byte_price_plan": {
                    "schema": "compact_nerv_byte_price_controller.v1"
                },
                "blockers": [
                    "decoder_weight_saliency_replay_required_for_authority"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(inventory_tool, "RESEARCH_DIR", research)

    rc = inventory_tool_main(
        [
            "--focus-family",
            "hi_nerv",
            "--repo-root",
            str(REPO),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    measured = payload["decoder_weight_waterfill_reports"]["hi_nerv"]
    assert measured["row_count"] == 1
    assert measured["waterfill_rows"][0]["row_id"] == "hi_nerv_auto"
    assert measured["waterfill_rows"][0]["waterfill_summary"]["group_count"] == 5
    assert payload["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_hinerv_replay_actuator_report(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "replay_actuator.json"
    output_json = tmp_path / "inventory.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_ladder_replay_actuator.v1",
                "report_path": ".omx/research/hinerv_archive_ladder_replay_actuator_fake.json",
                "execution_requested": True,
                "load_existing_requested": False,
                "row_count": 1,
                "loaded_replay_report_count": 1,
                "receiver_proof_ready_row_count": 1,
                "archive_bytes_by_row_id": {"hi_nerv_local_tiny": 134908},
                "rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "status": "executed_report_loaded_false_authority",
                        "archive_bytes": 134908,
                        "archive_sha256": "a" * 64,
                        "archive_path": "/Volumes/VertigoDataTier/pact/tiny/archive.zip",
                        "submission_dir": "/Volumes/VertigoDataTier/pact/tiny/submission",
                        "receiver_proof_path": "/Volumes/VertigoDataTier/pact/tiny/proof.json",
                        "receiver_proof_ready": True,
                        "blockers": ["hinerv_archive_size_row_has_no_nonrate_score"],
                    }
                ],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "hi_nerv",
            "--repo-root",
            str(REPO),
            "--hinerv-archive-ladder-replay-actuator-json",
            str(replay_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    measured = payload["archive_ladder_replay_actuator_reports"]["hi_nerv"]
    assert measured["row_count"] == 1
    assert measured["replay_rows"][0]["archive_bytes"] == 134908
    assert measured["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_hinerv_backend_drift_report(
    tmp_path: Path,
) -> None:
    drift_path = tmp_path / "backend_drift.json"
    output_json = tmp_path / "inventory.json"
    drift_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_backend_drift.v1",
                "status": "local_backend_drift_within_tolerance_false_authority",
                "report_path": ".omx/research/hinerv_archive_backend_drift_fake.json",
                "reference_json_path": ".omx/research/fallback.json",
                "reference_json_sha256": "a" * 64,
                "candidate_json_path": ".omx/research/mlx.json",
                "candidate_json_sha256": "b" * 64,
                "reference_label": "pytorch_portable_fallback",
                "candidate_label": "mlx_metal",
                "row_count": 4,
                "matched_row_count": 4,
                "byte_ready_row_count": 4,
                "reference_receiver_proof_ready_row_count": 4,
                "candidate_receiver_proof_ready_row_count": 4,
                "max_abs_byte_delta_allowed": 1024,
                "max_abs_byte_delta_observed": 81,
                "sum_byte_delta_candidate_minus_reference": -194,
                "sum_rate_score_delta_candidate_minus_reference": -0.000129185,
                "within_byte_drift_tolerance": True,
                "local_dev_velocity_ready": True,
                "ready_backend_for_local_iteration": "mlx_metal",
                "blockers": ["hinerv_archive_backend_drift_local_dev_velocity_only"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "hi_nerv",
            "--repo-root",
            str(REPO),
            "--hinerv-archive-backend-drift-json",
            str(drift_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    measured = payload["archive_backend_drift_reports"]["hi_nerv"]
    assert measured["local_dev_velocity_ready"] is True
    assert measured["max_abs_byte_delta_observed"] == 81
    assert measured["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_snerv_waterfill_report(
    tmp_path: Path,
) -> None:
    waterfill_path = tmp_path / "waterfill.json"
    output_json = tmp_path / "inventory.json"
    waterfill_path.write_text(
        """
        {
          "schema": "snerv_trained_ladder_waterfill.v1",
          "report_path": ".omx/research/snerv_trained_ladder_waterfill_fake.json",
          "row_count": 1,
          "source_status": "trained_ladder_row_blocked",
          "rows": [
            {
              "row_id": "snerv_local_tiny",
              "archive_bytes": 123,
              "archive_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "archive_sha256_actual": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "receiver_codec_mode": "contest_archive_zip",
              "decoder_precision_mode": "mixed_magnitude_symmetric",
              "decoder_payload_schema": "snerv_decoder_payload.v3",
              "decoder_state_group_count": 3,
              "waterfill_summary": {"group_count": 3},
              "blockers": ["decoder_weight_saliency_missing_for_some_groups"]
            }
          ],
          "section_value_rows": [{"row_id": "r0"}],
          "blockers": ["decoder_weight_saliency_replay_required_for_authority"],
          "score_claim": false,
          "ready_for_exact_eval_dispatch": false
        }
        """,
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "snerv",
            "--repo-root",
            str(REPO),
            "--snerv-trained-ladder-waterfill-json",
            str(waterfill_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["decoder_weight_waterfill_reports"]["snerv"]["row_count"] == 1
    assert payload["decoder_weight_waterfill_reports"]["snerv"][
        "section_value_row_count"
    ] == 1
    assert payload["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_hinerv_saliency_report(
    tmp_path: Path,
) -> None:
    saliency_path = tmp_path / "saliency.json"
    output_json = tmp_path / "inventory.json"
    saliency_path.write_text(
        """
        {
          "schema": "hinerv_decoder_weight_saliency_replay.v1",
          "report_path": ".omx/research/hinerv_decoder_weight_saliency_fake.json",
          "row_count": 1,
          "full_video_coverage": false,
          "scorer_source": "real_upstream_differentiable_scorers",
          "pair_schedule": {"max_pairs": 1, "start_pair": 0, "pair_stride": 1},
          "saliency_by_name": {"head_rgb_1.weight": 0.125},
          "saliency_rows": [
            {
              "group_name": "head_rgb_1.weight",
              "decoder_weight_saliency": 0.125,
              "score_saliency": 0.125
            }
          ],
          "blockers": ["full_video_coverage_missing"],
          "score_claim": false,
          "ready_for_exact_eval_dispatch": false
        }
        """,
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "hi_nerv",
            "--repo-root",
            str(REPO),
            "--hinerv-decoder-weight-saliency-json",
            str(saliency_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["decoder_weight_saliency_replays"]["hi_nerv"]["row_count"] == 1
    assert payload["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_snerv_mode_assignment_report(
    tmp_path: Path,
) -> None:
    mode_path = tmp_path / "modes.json"
    output_json = tmp_path / "inventory.json"
    mode_path.write_text(
        """
        {
          "schema": "snerv_waterfill_mode_assignment.v1",
          "report_path": ".omx/research/snerv_mode_assignment_fake.json",
          "row_count": 1,
          "local_advisory_probe_ready_row_count": 1,
          "receiver_mode_export_ready_row_count": 0,
          "rows": [
            {
              "row_id": "snerv_local_tiny",
              "archive_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "decoder_payload_schema": "snerv_decoder_payload.v3",
              "mode_count": 3,
              "mode_plan_cli_arg": "fp16,fp16,fp16",
              "mode_histogram": {"fp16": 3},
              "ready_for_local_advisory_probe": true,
              "ready_for_receiver_mode_export": false,
              "blockers": ["fp32_protect_downgraded_to_fp16_requires_receiver_replay"]
            }
          ],
          "blockers": ["mode_assignment_is_false_authority_until_receiver_replay"],
          "score_claim": false,
          "ready_for_exact_eval_dispatch": false
        }
        """,
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "snerv",
            "--repo-root",
            str(REPO),
            "--snerv-waterfill-mode-assignment-json",
            str(mode_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["decoder_mode_assignment_reports"]["snerv"]["row_count"] == 1
    assert payload["decoder_mode_assignment_reports"]["snerv"]["assignment_rows"][0][
        "mode_plan_cli_arg"
    ] == "fp16,fp16,fp16"
    assert payload["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_snerv_decoder_mode_probe_report(
    tmp_path: Path,
) -> None:
    probe_path = tmp_path / "probe.json"
    output_json = tmp_path / "inventory.json"
    probe_path.write_text(
        """
        {
          "schema": "snerv_decoder_mode_assignment_probe.v1",
          "axis_tag": "[macOS-CPU advisory]",
          "n_pairs": 1,
          "levels": 1,
          "best_plan_label": "explicit_fp163",
          "best_plan_score_linf_advisory": 3.58,
          "candidates": [
            {
              "label": "explicit_fp163",
              "modes": ["fp16", "fp16", "fp16"],
              "mode_assignment_source": "explicit",
              "mode_histogram": {"fp16": 3},
              "archive_bytes_total": 550390,
              "decoder_bytes": 1030,
              "d_seg_mean_linf": 0.0139,
              "d_pose_mean_linf": 0.332,
              "score_linf": 3.58,
              "receiver_archive_replay_verified": true,
              "blockers": ["paired_contest_cpu_cuda_auth_eval_missing"]
            }
          ],
          "blockers": ["macos_cpu_advisory_only"],
          "score_claim": false,
          "ready_for_exact_eval_dispatch": false
        }
        """,
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "snerv",
            "--repo-root",
            str(REPO),
            "--snerv-decoder-mode-probe-json",
            str(probe_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["decoder_mode_probe_reports"]["snerv"]["best_plan_label"] == (
        "explicit_fp163"
    )
    assert payload["decoder_mode_probe_reports"]["snerv"]["candidate_count"] == 1
    assert payload["score_claim"] is False


def test_build_nerv_control_inventory_cli_accepts_snerv_scorer_loop_qat_report(
    tmp_path: Path,
) -> None:
    qat_path = tmp_path / "snerv_qat.json"
    output_json = tmp_path / "inventory.json"
    qat_path.write_text(
        json.dumps(
            {
                "schema": "snerv_scorer_loop_qat_local_trainer.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "research_json_path": ".omx/research/snerv_qat_fake.json",
                "result_sha256": "a" * 64,
                "n_pairs": 1,
                "levels": 1,
                "wavelet": "haar",
                "qat_bits": 8,
                "search_mode": "random_signed",
                "scorer_loop_evaluations": 3,
                "baseline_score_linf": 0.9,
                "best_score_linf": 0.9,
                "accepted_improvement": False,
                "receiver_contract_satisfied": True,
                "blockers": ["local_smoke_only_not_full_600_pairs"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    rc = inventory_tool_main(
        [
            "--focus-family",
            "snerv",
            "--repo-root",
            str(REPO),
            "--snerv-scorer-loop-qat-json",
            str(qat_path),
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    measured = payload["snerv_scorer_loop_qat_reports"]["snerv"]
    assert measured["n_pairs"] == 1
    assert measured["result_sha256"] == "a" * 64
    assert measured["score_claim"] is False
