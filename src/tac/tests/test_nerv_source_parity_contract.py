# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from tac.analysis.nerv_source_parity_contract import (
    build_nerv_source_parity_contract,
    render_nerv_source_parity_markdown,
    write_nerv_source_parity_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_source_parity_contract_is_false_authority_and_family_scoped() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT)

    assert report["schema"] == "nerv_source_parity_contract.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert set(report["families"]) == {"hi_nerv", "snerv"}
    assert report["required_for_long_training_ready"] is True
    assert report["blockers"] == ()
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["nonblocking_gaps"]
    analogue_rows = {row["surface_id"]: row for row in report["analogue_risk_rows"]}
    assert analogue_rows["snerv_official_mfu_hfr_tub_numeric_primitives"][
        "insufficient_for"
    ] == "byte_closed_official_snerv_export_runtime"
    assert analogue_rows["hi_nerv_mlx_backend_drift"][
        "insufficient_for"
    ] == "contest_cpu_cuda_auth_eval_authority"
    assert analogue_rows["pr95_hnerv_mlx_control_arm"][
        "insufficient_for"
    ] == "pr95_source_faithful_control_reproduction"
    assert all(row["score_claim"] is False for row in analogue_rows.values())
    analogue_ids = {row["surface_id"] for row in report["analogue_risk_rows"]}
    assert {
        "snerv_receiver_safe_mfu_hfr_temporal_adapter",
        "snerv_official_mfu_hfr_tub_numeric_primitives",
        "snerv_local_modelsize_analogue",
        "hi_nerv_local_target_modelsize",
        "hi_nerv_mlx_backend_drift",
        "pr95_hnerv_mlx_control_arm",
    }.issubset(analogue_ids)


def test_hinerv_generic_resize_path_is_no_longer_a_source_parity_blocker() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT, families=("hi_nerv",))
    rows = {row["feature_id"]: row for row in report["feature_rows"]}

    assert rows["hi_nerv_generic_resolution_path"]["status"] == "implemented_or_bound"
    assert "hi_nerv_generic_resolution_path_missing" not in report["blockers"]


def test_hinerv_legacy_phase_a_surface_is_fail_closed_in_source_contract() -> None:
    report = build_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        families=("hi_nerv",),
    )
    rows = {row["feature_id"]: row for row in report["feature_rows"]}
    guard = rows["hi_nerv_legacy_phase_a_false_authority_guard"]

    assert guard["status"] == "implemented_or_bound"
    assert guard["required_for_long_training"] is True
    assert guard["blockers"] == ()
    assert "hi_nerv_legacy_phase_a_false_authority_guard_missing" not in report[
        "blockers"
    ]
    present_symbols = {
        symbol["symbol"]
        for symbol in guard["symbol_rows"]
        if symbol["status"] == "present"
    }
    assert {
        "LEGACY_HINERV_PHASE_A_BLOCKER",
        "legacy_hinerv_phase_a_false_authority",
    }.issubset(present_symbols)


def test_hinerv_grid_convnext_and_receiver_bitstream_pipeline_are_bound() -> None:
    report = build_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        families=("hi_nerv",),
    )
    rows = {row["feature_id"]: row for row in report["feature_rows"]}

    assert report["required_for_long_training_ready"] is True
    assert "hi_nerv_official_feature_grid_convnext_trilinear_missing" not in report["blockers"]
    assert "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline_missing" not in report["blockers"]
    assert "hi_nerv_official_torchac_entropy_coder_missing" not in report["blockers"]
    assert rows["hi_nerv_official_feature_grid_convnext_trilinear"]["status"] == ("implemented_or_bound")
    assert rows["hi_nerv_prune_quantnoise_receiver_bitstream_pipeline"]["status"] == ("implemented_or_bound")
    assert rows["hi_nerv_official_torchac_entropy_coder_parity"]["status"] == ("implemented_or_bound")
    assert rows["hi_nerv_official_torchac_entropy_coder_parity"]["required_for_long_training"] is False
    present_symbols = {
        symbol["symbol"]
        for symbol in rows["hi_nerv_prune_quantnoise_receiver_bitstream_pipeline"]["symbol_rows"]
        if symbol["status"] == "present"
    }
    assert {
        "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
        "apply_decoder_pruning",
        "apply_decoder_quant_noise",
        "measure_hi_nerv_decoder_bitstream_roundtrip",
        "select_hi_nerv_bitstream_codec_by_scorer_waterfill",
        "repack_archive_decoder_codec",
    }.issubset(present_symbols)


def test_source_parity_contract_records_insufficient_analogue_surfaces() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT)

    risks = {row["surface_id"]: row for row in report["analogue_risk_rows"]}
    assert risks["snerv_receiver_safe_mfu_hfr_temporal_adapter"][
        "insufficient_for"
    ] == "official_spectra_preserving_snerv_source_forward"
    assert "snerv_official_mfu_hfr_tub_source_forward_replay_missing" in risks[
        "snerv_receiver_safe_mfu_hfr_temporal_adapter"
    ]["remaining_blockers"]
    assert risks["snerv_official_mfu_hfr_tub_numeric_primitives"][
        "insufficient_for"
    ] == "byte_closed_official_snerv_export_runtime"
    assert risks["snerv_local_modelsize_analogue"]["insufficient_for"] == (
        "official_snerv_modelsize_authority"
    )
    assert risks["hi_nerv_local_target_modelsize"]["insufficient_for"] == (
        "official_hinerv_config_family_authority"
    )
    assert risks["hi_nerv_mlx_backend_drift"]["insufficient_for"] == (
        "contest_cpu_cuda_auth_eval_authority"
    )
    assert risks["pr95_hnerv_mlx_control_arm"]["insufficient_for"] == (
        "pr95_source_faithful_control_reproduction"
    )
    assert all(row["score_claim"] is False for row in risks.values())


def test_snerv_spectra_preserving_adapter_unblocks_training_but_not_official_parity() -> None:
    report = build_nerv_source_parity_contract(repo_root=REPO_ROOT, families=("snerv",))

    assert report["required_for_long_training_ready"] is True
    assert "snerv_official_mfu_hfr_tub_parity_missing" not in report["blockers"]
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["nonblocking_gaps"]
    assert "snerv_receiver_safe_mfu_hfr_temporal_adapter_missing" not in report["blockers"]
    assert "snerv_fc_dim_modelsize_control_missing" not in report["blockers"]
    assert "snerv_scorer_loop_decoder_qat_missing" not in report["blockers"]
    assert "snerv_lf_quant_intn_codec_missing" not in report["blockers"]
    assert "snerv_qat_receiver_codec_pricing_missing" not in report["blockers"]
    assert "snerv_official_haar_mode_missing" not in report["blockers"]
    assert "snerv_receiver_dependency_custody_missing" not in report["blockers"]
    rows = {row["feature_id"]: row for row in report["feature_rows"]}
    assert rows["snerv_scorer_loop_decoder_qat"]["status"] == "implemented_or_bound"
    assert rows["snerv_qat_receiver_codec_pricing"]["status"] == "implemented_or_bound"
    assert rows["snerv_official_haar_mode"]["status"] == "implemented_or_bound"
    assert rows["snerv_official_tub_haar_dwt1d_temporal_primitive"]["status"] == (
        "implemented_or_bound"
    )
    assert rows["snerv_official_mfu_hfr_tub_numeric_primitives"]["status"] == (
        "implemented_or_bound"
    )
    assert rows["snerv_official_mfu_hfr_tub_numeric_primitives"][
        "required_for_long_training"
    ] is True
    assert rows["snerv_receiver_dependency_custody"]["status"] == "implemented_or_bound"
    assert rows["snerv_scalable_layer_admission_policy"]["status"] == ("implemented_or_bound")
    assert rows["snerv_scalable_layer_admission_policy"]["required_for_long_training"] is False
    assert "snerv_scalable_layer_admission_policy_missing" not in report["blockers"]
    present_symbols = {
        symbol["symbol"]
        for symbol in rows["snerv_receiver_safe_mfu_hfr_temporal_adapter"]["symbol_rows"]
        if symbol["status"] == "present"
    }
    assert {
        "MultiResolutionFusionUnit",
        "HighFrequencyRestorer",
        "SnervTemporalExtension",
        "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
        "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
    }.issubset(present_symbols)
    assert rows["snerv_receiver_safe_mfu_hfr_temporal_adapter"]["status"] == ("implemented_or_bound")
    official = rows["snerv_official_mfu_hfr_tub_parity"]
    assert official["status"] == "missing_or_partial"
    assert official["required_for_long_training"] is False
    assert official["blockers"] == ("snerv_official_mfu_hfr_tub_parity_missing",)
    official_symbols = {symbol["symbol"] for symbol in official["symbol_rows"] if symbol["status"] == "present"}
    assert "SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF" not in official_symbols
    tub = rows["snerv_official_tub_haar_dwt1d_temporal_primitive"]
    tub_symbols = {
        symbol["symbol"]
        for symbol in tub["symbol_rows"]
        if symbol["status"] == "present"
    }
    assert {
        "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
        "SnervTemporalExtension.official_haar_dwt1d_lowpass_features",
    }.issubset(tub_symbols)
    assert tub["blockers"] == ()
    assert {row["status"] for row in tub["source_marker_rows"]} == {"present"}
    primitive_symbols = {
        symbol["symbol"]
        for symbol in rows["snerv_official_mfu_hfr_tub_numeric_primitives"][
            "symbol_rows"
        ]
        if symbol["status"] == "present"
    }
    assert {
        "OfficialSnervMfu",
        "conv_transpose2d_nchw",
        "OfficialHfrHeads",
        "conv2d_nchw_mlx",
        "prepare_official_tub_graph_inputs",
        "official_output2_fusion_shape",
    }.issubset(primitive_symbols)
    assert (
        "snerv_official_mfu_hfr_tub_numeric_primitives_missing"
        not in report["blockers"]
    )
    controls = {row["control_id"]: row for row in report["control_rows"]}
    assert controls["snerv_fc_dim_modelsize_control"]["status"] == ("implemented_or_declared")
    assert controls["snerv_fc_dim_modelsize_control"]["missing_markers"] == []
    assert controls["snerv_lf_stepmap_and_intN_control"]["status"] == ("implemented_or_declared")
    assert controls["snerv_lf_stepmap_and_intN_control"]["missing_markers"] == []


def test_snerv_official_source_audit_embeds_without_promoting_parity() -> None:
    audit = {
        "schema": "snerv_official_source_parity_audit.v1",
        "authority": "false_authority_source_audit_no_score_claim",
        "family": "snerv",
        "official_repo": {
            "repo_url": "https://github.com/qwertja/SNeRV",
            "root": "/Volumes/VertigoDataTier/pact/oss_sources/SNeRV",
            "head_sha": "abc123",
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

    report = build_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        families=("snerv",),
        snerv_official_source_audit=audit,
    )

    assert report["source_audits"]
    rows = {row["feature_id"]: row for row in report["feature_rows"]}
    official = rows["snerv_official_mfu_hfr_tub_parity"]
    assert official["status"] == "missing_or_partial"
    assert official["source_audit_rows"][0]["official_head_sha"] == "abc123"
    assert official["source_audit_rows"][0]["official_source_markers_present"] is True
    assert official["source_audit_rows"][0]["official_mfu_hfr_tub_parity_proven"] is False
    assert "snerv_official_mfu_hfr_tub_parity_missing" in report["nonblocking_gaps"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_source_parity_contract_writes_json_and_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "source_parity.json"
    md_path = tmp_path / "source_parity.md"

    report = write_nerv_source_parity_contract(
        repo_root=REPO_ROOT,
        output_json=json_path,
        output_md=md_path,
    )

    assert json_path.is_file()
    assert md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert "NeRV Source-Parity Contract" in md_text
    assert "Nonblocking Source Gaps" in md_text
    assert render_nerv_source_parity_markdown(report).startswith("# NeRV Source-Parity Contract")
