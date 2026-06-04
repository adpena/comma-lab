# SPDX-License-Identifier: MIT
"""Tests for NeRV-family model-size budget planning."""

from __future__ import annotations

import pytest

from tac.analysis.nerv_modelsize_budget import (
    DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS,
    MODELSIZE_RATE_AUTHORITY_SURFACE,
    SNERV_CONTEST_RECEIVER_PROFILE_ID,
    SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID,
    SNERV_OFFICIAL_MFU_HFR_TUB_LEVELS,
    NervModelSizeBudgetError,
    analyze_hinerv_modelsize_candidate,
    analyze_snerv_modelsize_candidate,
    build_hinerv_config_from_modelsize_candidate,
    build_hinerv_config_from_size_knobs,
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
    enumerate_hinerv_modelsize_candidates,
    enumerate_snerv_modelsize_candidates,
    official_nerv_oss_flag_audit,
    select_hinerv_modelsize_candidates,
    snerv_model_size_adapter_from_id_token,
    snerv_model_size_adapter_id_token,
    snerv_modelsize_control_profile,
    snerv_temporal_mode_from_id_token,
    snerv_temporal_mode_id_token,
)
from tac.substrates.hi_nerv.architecture import HinervSubstrate
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    SnervModelSizeConfig,
)


def test_hinerv_modelsize_closed_form_matches_real_module() -> None:
    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
    )
    model = HinervSubstrate(cfg)

    row = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        decoder_codec="int4_mixed",
    )

    assert row.total_trainable_params == model.num_parameters()
    assert row.mid_injection_block_index == cfg.mid_injection_block_index == 1
    assert row.fine_injection_block_index == cfg.fine_injection_block_index == 4
    assert row.latent_trainable_params == 17 * (6 + 12 + 24)
    assert row.decoder_trainable_params == (
        row.total_trainable_params - row.latent_trainable_params
    )
    assert row.latent_int16_payload_bytes == 2 * row.latent_trainable_params
    assert row.nominal_decoder_payload_bytes == (
        row.decoder_trainable_params * 4 + 7
    ) // 8
    assert row.capacity_source == "manual_local_knobs"
    assert row.target_modelsize_mparams is None
    assert row.modelsize_error_mparams is None
    assert row.score_claim is False
    assert row.ready_for_exact_eval_dispatch is False


def test_hinerv_modelsize_counts_official_grid_convnext_controls() -> None:
    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
    )
    model = HinervSubstrate(cfg)

    row = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
    )

    assert row.candidate_id == (
        "hinerv_np17_ld12_ed16_dc10_mi1fi4_hfg_cnx_lg2c4_cx2k3_"
        "int4_mixed_ceil178000"
    )
    assert row.use_hierarchical_feature_grid is True
    assert row.use_convnext_blocks is True
    assert row.mid_injection_block_index == cfg.mid_injection_block_index
    assert row.fine_injection_block_index == cfg.fine_injection_block_index
    assert row.total_trainable_params == model.num_parameters()
    assert row.decoder_trainable_params > row.latent_trainable_params
    assert row.score_claim is False


def test_hinerv_modelsize_candidate_builds_same_receiver_config() -> None:
    row = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
        mid_injection_block_index=1,
        fine_injection_block_index=4,
    )
    cfg_from_candidate = build_hinerv_config_from_modelsize_candidate(row.as_dict())
    cfg_from_knobs = build_hinerv_config_from_size_knobs(
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
        mid_injection_block_index=1,
        fine_injection_block_index=4,
    )

    assert cfg_from_candidate == cfg_from_knobs
    assert HinervSubstrate(cfg_from_candidate).num_parameters() == (
        row.total_trainable_params
    )


def test_hinerv_modelsize_candidate_id_separates_graph_controls() -> None:
    base = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=4,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
    )
    changed = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=17,
        latent_dim=12,
        embed_dim=16,
        decoder_channel=10,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=3,
        local_grid_channels=5,
        convnext_mlp_ratio=4,
        convnext_kernel_size=5,
        mid_injection_block_index=0,
        fine_injection_block_index=2,
    )

    assert base.candidate_id != changed.candidate_id
    assert "_mi1fi4_hfg_cnx_lg2c4_cx2k3_" in base.candidate_id
    assert "_mi0fi2_hfg_cnx_lg3c5_cx4k5_" in changed.candidate_id


def test_hinerv_modelsize_budget_report_is_false_authority_and_budgeted() -> None:
    report = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=6,
    )

    assert report["schema"] == "nerv_modelsize_budget.v1"
    assert report["candidate_count"] > report["selected_candidate_count"] > 0
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["budget_math"]["nominal_payload_is_not_authority"] is True
    selected = report["selected_candidates"]
    assert all(row["hard_byte_ceiling"] == 178_000 for row in selected)
    assert any(row["nominal_under_ceiling"] for row in selected)
    assert all(row["requires_archive_byte_oracle"] is True for row in selected)
    assert report["use_hierarchical_feature_grid_options"] == [False, True]
    assert report["use_convnext_blocks_options"] == [False, True]
    assert any(row["use_hierarchical_feature_grid"] for row in selected)
    assert any(row["use_convnext_blocks"] for row in selected)
    codecs = {row["decoder_codec"] for row in selected}
    assert {
        "portfolio_auto",
        "int8_mixed",
        "int7_mixed",
        "int6_mixed",
        "int4_mixed",
        "int2_mixed",
    } <= codecs
    assert "selection_strategy" in report["budget_math"]


def test_hinerv_modelsize_budget_can_emit_official_controls_only() -> None:
    report = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=(36_000, 178_000),
        num_pairs=600,
        per_ceiling_limit=4,
        official_controls_only=True,
    )

    assert report["official_controls_only"] is True
    assert report["budget_math"]["official_controls_only"] is True
    assert report["use_hierarchical_feature_grid_options"] == [True]
    assert report["use_convnext_blocks_options"] == [True]
    selected = report["selected_candidates"]
    assert selected
    assert all(row["use_hierarchical_feature_grid"] is True for row in selected)
    assert all(row["use_convnext_blocks"] is True for row in selected)
    assert all("_hfg_cnx_" in row["candidate_id"] for row in selected)
    assert all(row["score_claim"] is False for row in selected)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in selected)


def test_hinerv_target_modelsize_selects_real_capacity_rows() -> None:
    report = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=17,
        per_ceiling_limit=8,
        target_modelsize_mparams=(0.03,),
        use_hierarchical_feature_grid_options=(False, True),
        use_convnext_blocks_options=(False, True),
    )

    assert report["target_modelsize_mparams"] == [0.03]
    selected = [
        row
        for row in report["selected_candidates"]
        if row["capacity_source"] == "local_hinerv_target_modelsize"
    ]
    assert selected
    row = min(selected, key=lambda item: item["modelsize_error_mparams"])
    assert row["target_modelsize_mparams"] == 0.03
    assert row["modelsize_error_mparams"] == pytest.approx(
        abs(row["modelsize_mparams"] - 0.03)
    )
    assert row["candidate_id"].endswith("_tgtmp0p03")

    cfg = build_hinerv_config_from_size_knobs(
        num_pairs=17,
        latent_dim=row["latent_dim"],
        embed_dim=row["embed_dim"],
        decoder_channel=row["decoder_channel"],
        use_hierarchical_feature_grid=row["use_hierarchical_feature_grid"],
        use_convnext_blocks=row["use_convnext_blocks"],
        local_grid_levels=row["local_grid_levels"],
        local_grid_channels=row["local_grid_channels"],
        convnext_mlp_ratio=row["convnext_mlp_ratio"],
        convnext_kernel_size=row["convnext_kernel_size"],
    )
    assert row["total_trainable_params"] == HinervSubstrate(cfg).num_parameters()
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_hinerv_modelsize_knobs_are_real_capacity_and_codec_controls() -> None:
    small = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=8,
        embed_dim=8,
        decoder_channel=8,
        decoder_codec="int4_mixed",
    )
    wider = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=16,
        embed_dim=16,
        decoder_channel=16,
        decoder_codec="int4_mixed",
    )
    same_wider_int2 = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        latent_dim=16,
        embed_dim=16,
        decoder_channel=16,
        decoder_codec="int2_mixed",
    )

    assert wider.modelsize_mparams > small.modelsize_mparams
    assert wider.total_trainable_params > small.total_trainable_params
    assert wider.latent_int16_payload_bytes > small.latent_int16_payload_bytes
    assert wider.nominal_decoder_payload_bytes > small.nominal_decoder_payload_bytes
    assert same_wider_int2.total_trainable_params == wider.total_trainable_params
    assert same_wider_int2.latent_int16_payload_bytes == (
        wider.latent_int16_payload_bytes
    )
    assert same_wider_int2.nominal_decoder_payload_bytes < (
        wider.nominal_decoder_payload_bytes
    )
    assert same_wider_int2.nominal_rate_score < wider.nominal_rate_score


def test_hinerv_target_modelsize_rows_are_false_authority_nearest_rows() -> None:
    report = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=(36_000,),
        num_pairs=600,
        per_ceiling_limit=4,
        target_modelsize_mparams=(0.02,),
    )

    targeted = [
        row
        for row in report["selected_candidates"]
        if row["capacity_source"] == "local_hinerv_target_modelsize"
    ]
    assert targeted
    assert report["target_modelsize_mparams"] == [0.02]
    assert all(row["target_modelsize_mparams"] == 0.02 for row in targeted)
    assert all(row["modelsize_error_mparams"] is not None for row in targeted)
    assert all(row["score_claim"] is False for row in targeted)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in targeted)
    assert all("_tgtmp0p02" in row["candidate_id"] for row in targeted)
    contract = targeted[0]["modelsize_control_contract"]
    assert contract["schema"] == "nerv_modelsize_control_contract.v1"
    assert contract["family"] == "hi_nerv"
    assert contract["control_semantics"] == (
        "local_receiver_visible_grid_search_nearest_target"
    )
    assert contract["shared_target_modelsize_mparams_consumed_as"] == (
        "nearest_local_param_count_target"
    )
    assert contract["modelsize_mparams_is_official_upstream_flag"] is False
    assert contract["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert contract["authority_split"]["schema"] == (
        "nerv_modelsize_control_authority_split.v1"
    )
    assert contract["authority_split"]["modelsize_mparams_semantics"] == (
        "local_nearest_parameter_count_target"
    )
    assert contract["authority_split"]["archive_byte_authority_surface"] == (
        MODELSIZE_RATE_AUTHORITY_SURFACE
    )
    assert contract["authority_split"]["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert contract["authority_split"]["score_claim"] is False
    assert contract["nominal_payload_bytes_are_planner_prior_only"] is True
    assert contract["nominal_under_ceiling_is_not_promotion_authority"] is True
    assert (
        contract["receiver_closed_archive_bytes_required_for_under_ceiling_claim"]
        is True
    )
    assert contract["trained_archive_export_required_for_score_or_rate_claim"] is True
    assert contract["rate_authority_surface"] == MODELSIZE_RATE_AUTHORITY_SURFACE
    assert contract["mutates_receiver_visible_architecture"] is True
    assert contract["archive_bytes_authority_required"] is True
    assert all(contract[key] is True for key in MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS)


def test_hinerv_target_modelsize_demotes_over_ceiling_closeness() -> None:
    rows = enumerate_hinerv_modelsize_candidates(
        hard_byte_ceilings=(50_000,),
        num_pairs=17,
        latent_dims=(12, 24),
        embed_dims=(24, 32),
        decoder_channels=(12,),
        decoder_codecs=("int8_mixed",),
        use_hierarchical_feature_grid_options=(False,),
        use_convnext_blocks_options=(False,),
        target_modelsize_mparams=(0.05,),
    )
    targeted = [
        row for row in rows if row.capacity_source == "local_hinerv_target_modelsize"
    ]
    under = [row for row in targeted if row.nominal_under_ceiling]
    over = [row for row in targeted if not row.nominal_under_ceiling]

    assert under
    assert over
    best_under = min(
        under,
        key=lambda row: float(row.modelsize_error_mparams or 0.0),
    )
    best_over = min(
        over,
        key=lambda row: float(row.modelsize_error_mparams or 0.0),
    )
    assert best_over.modelsize_error_mparams < best_under.modelsize_error_mparams
    assert best_over.byte_headroom < 0
    assert best_under.byte_headroom >= 0

    selected = select_hinerv_modelsize_candidates(rows, per_ceiling_limit=1)

    assert selected[0].candidate_id == best_under.candidate_id
    assert selected[0].nominal_under_ceiling is True
    assert selected[0].score_claim is False
    assert best_over.candidate_id not in {row.candidate_id for row in selected}


def test_snerv_modelsize_budget_report_prices_receiver_grammar() -> None:
    row = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="db2",
        levels=3,
        bits_per_coeff=2.0,
        step_map_bits_per_coeff=1.0,
        decoder_payload_codec="int4_symmetric",
        fc_dim=11,
        emb_size=2,
        mfu_scales=(1, 3),
        hfr_gain=0.25,
        temporal_context=1,
    )

    assert row.family == "snerv"
    assert row.candidate_id == (
        "snerv_np600_db2_lv3_lfb2_stepb1_fc11e2_"
        "p1_mfu1-3_hfr0p25_t1_adbase_int4_symmetric_ceil178000"
    )
    assert row.wavelet == "db2"
    assert row.fc_dim == 11
    assert row.capacity_source == "manual_fc_dim"
    assert row.emb_size == 2
    assert row.decoder_feature_count == 15
    assert row.hf_decoder_weight_count == 3 * 3 * 15
    assert row.mfu_scales == (1, 3)
    assert row.hfr_gain == 0.25
    assert row.temporal_context == 1
    assert row.temporal_mode == "delta"
    assert row.lf_plane_count == 600 * 2 * 3
    assert row.lf_coeff_count_total == row.lf_plane_count * row.lf_coeffs_per_plane
    assert row.nominal_lf_payload_bytes == int(
        (row.lf_coeff_count_total * 2.0 + 7) // 8
    )
    assert row.requires_snAR1_archive_byte_oracle is True
    assert row.score_claim is False
    assert row.ready_for_exact_eval_dispatch is False
    row_contract = row.as_dict()["modelsize_control_contract"]
    assert row_contract["control_semantics"] == (
        "manual_receiver_visible_fc_dim_feature_basis"
    )
    assert row_contract["modelsize_mparams_is_official_upstream_flag"] is False
    assert row_contract["authority_split"]["modelsize_mparams_semantics"] == (
        "absent_or_measured_parameter_count_metadata"
    )
    assert row_contract["authority_split"]["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert row_contract["nominal_payload_bytes_are_planner_prior_only"] is True
    assert row_contract["nominal_under_ceiling_is_not_promotion_authority"] is True
    assert (
        row_contract["receiver_closed_archive_bytes_required_for_under_ceiling_claim"]
        is True
    )
    assert row_contract["mutates_receiver_visible_fc_dim"] is True
    assert row_contract["archive_bytes_authority_required"] is True

    report = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=8,
    )
    assert report["schema"] == "snerv_modelsize_budget.v1"
    assert report["wavelet"] == "haar"
    assert report["candidate_count"] > report["selected_candidate_count"] > 0
    assert report["budget_math"]["nominal_payload_is_not_authority"] is True
    assert report["budget_math"]["rate_authority_surface"] == (
        MODELSIZE_RATE_AUTHORITY_SURFACE
    )
    assert report["budget_math"]["modelsize_control_contract_required_true_fields"] == list(
        MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
    )
    selected = report["selected_candidates"]
    assert all(row["hard_byte_ceiling"] == 178_000 for row in selected)
    assert {row["levels"] for row in selected} >= {3, 4, 5}
    assert len({row["bits_per_coeff"] for row in selected}) >= 3
    assert all(row["wavelet"] == "haar" for row in selected)
    assert all(row["decoder_feature_count"] == 9 for row in selected)
    assert all(row["fc_dim"] == 9 for row in selected)
    assert all(row["emb_size"] == 0 for row in selected)
    assert report["score_claim"] is False


def test_snerv_modelsize_candidate_id_tokens_losslessly_bind_receiver_controls() -> None:
    row = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=36_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=2,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int2_symmetric",
        snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
        fc_dim=11,
        emb_size=2,
        patch_radius=3,
        mfu_scales=(1, 5),
        hfr_gain=0.375,
        temporal_context=2,
    )

    assert row.candidate_id == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_"
        "p3_mfu1-5_hfr0p375_t2_adspectra_int2_symmetric_ceil36000"
    )
    official_temporal = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=36_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=2,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int2_symmetric",
        fc_dim=11,
        emb_size=2,
        patch_radius=3,
        mfu_scales=(1, 5),
        hfr_gain=0.375,
        temporal_context=2,
        temporal_mode="official_haar_dwt1d_lowpass",
        snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
    )
    assert official_temporal.candidate_id == (
        "snerv_np600_haar_lv2_lfb1p5_stepb0p5_fc11e2_"
        "p3_mfu1-5_hfr0p375_t2_tmhaar1_adspectra_int2_symmetric_ceil36000"
    )
    assert official_temporal.temporal_mode == "official_haar_dwt1d_lowpass"
    assert official_temporal.as_dict()["modelsize_control_contract"][
        "mutates_receiver_visible_temporal_basis"
    ] is True
    assert snerv_temporal_mode_id_token("official_haar_dwt1d_lowpass") == "haar1"
    assert (
        snerv_temporal_mode_from_id_token("haar1")
        == "official_haar_dwt1d_lowpass"
    )
    official_row = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=2,
        bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int2_symmetric",
        official_modelsize_mparams=0.05,
    )
    assert "_oms0p05_" in official_row.candidate_id
    assert official_row.capacity_source == "official_snerv_modelsize"
    assert official_row.modelsize_mparams == 0.05
    assert official_row.official_modelsize_solution is not None
    official_contract = official_row.as_dict()["modelsize_control_contract"]
    assert official_contract["family"] == "snerv"
    assert official_contract["control_semantics"] == (
        "official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert official_contract["shared_target_modelsize_mparams_consumed_as"] == (
        "official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert official_contract["modelsize_mparams_is_official_upstream_flag"] is True
    assert official_contract["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert official_contract["authority_split"]["modelsize_mparams_semantics"] == (
        "official_upstream_parameter_budget_control"
    )
    assert official_contract["authority_split"][
        "same_numeric_target_can_feed_family_specific_controls"
    ] is True
    assert official_contract["authority_split"]["archive_byte_authority_surface"] == (
        MODELSIZE_RATE_AUTHORITY_SURFACE
    )
    assert official_contract["nominal_payload_bytes_are_planner_prior_only"] is True
    assert official_contract["nominal_under_ceiling_is_not_promotion_authority"] is True
    assert (
        official_contract[
            "receiver_closed_archive_bytes_required_for_under_ceiling_claim"
        ]
        is True
    )
    assert (
        official_contract["trained_archive_export_required_for_score_or_rate_claim"]
        is True
    )
    assert official_contract["mutates_receiver_visible_fc_dim"] is True
    assert official_contract["archive_bytes_authority_required"] is True
    assert all(
        official_contract[key] is True
        for key in MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
    )
    assert snerv_model_size_adapter_id_token(
        SNERV_SPECTRA_PRESERVING_ADAPTER
    ) == "spectra"
    assert snerv_model_size_adapter_from_id_token("spectra") == (
        SNERV_SPECTRA_PRESERVING_ADAPTER
    )
    assert snerv_model_size_adapter_id_token(
        SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    ) == "official"
    assert snerv_model_size_adapter_from_id_token("official") == (
        SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    )
    assert (
        snerv_model_size_adapter_id_token(
            "snerv_official_mfu_hfr_tub_primitives_adapter"
        )
        == "official"
    )
    assert SnervModelSizeConfig(
        adapter="snerv_official_mfu_hfr_tub_primitives_adapter"
    ).official_mfu_hfr_tub_numeric_primitives_requested
    custom_adapter = "snerv_custom/research:v2"
    custom_token = snerv_model_size_adapter_id_token(custom_adapter)
    assert custom_token.startswith("hx")
    assert snerv_model_size_adapter_from_id_token(custom_token) == custom_adapter
    with pytest.raises(NervModelSizeBudgetError):
        snerv_model_size_adapter_from_id_token("unknown")
    with pytest.raises(NervModelSizeBudgetError):
        snerv_model_size_adapter_from_id_token("hxnothex")


def test_snerv_fc_dim_and_emb_size_are_receiver_decoder_controls() -> None:
    base = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=4,
        bits_per_coeff=2.0,
        step_map_bits_per_coeff=1.0,
        decoder_payload_codec="int4_symmetric",
        fc_dim=9,
        emb_size=0,
    )
    larger_decoder = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=4,
        bits_per_coeff=2.0,
        step_map_bits_per_coeff=1.0,
        decoder_payload_codec="int4_symmetric",
        fc_dim=17,
        emb_size=4,
    )

    assert base.capacity_source == "manual_fc_dim"
    assert larger_decoder.capacity_source == "manual_fc_dim"
    assert larger_decoder.decoder_feature_count > base.decoder_feature_count
    assert larger_decoder.hf_decoder_weight_count > base.hf_decoder_weight_count
    assert larger_decoder.nominal_decoder_payload_bytes > (
        base.nominal_decoder_payload_bytes
    )
    assert larger_decoder.nominal_lf_payload_bytes == base.nominal_lf_payload_bytes
    assert larger_decoder.nominal_step_map_payload_bytes == (
        base.nominal_step_map_payload_bytes
    )
    assert larger_decoder.nominal_rate_score > base.nominal_rate_score


def test_snerv_official_primitives_modelsize_enumeration_uses_source_haar_j1() -> None:
    rows = enumerate_snerv_modelsize_candidates(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        levels=(2, 3, 4, 5),
        bits_per_coeffs=(1.5,),
        step_map_bits_per_coeffs=(0.5,),
        decoder_codecs=("int2_symmetric",),
        snerv_model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
        official_modelsize_mparams=(0.05,),
    )

    assert rows
    assert {row.levels for row in rows} == set(SNERV_OFFICIAL_MFU_HFR_TUB_LEVELS)
    assert all(
        row.snerv_model_size_adapter == SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
        for row in rows
    )
    assert all("_haar_lv1_" in row.candidate_id for row in rows)
    assert all(row.lf_plane_count == 1 for row in rows)
    assert all(row.lf_coeff_count_total == 1 for row in rows)
    assert all(row.bits_per_coeff == 1.0 for row in rows)
    assert all(row.step_map_bits_per_coeff == 1.0 for row in rows)
    assert all("_lfb1_stepb1_" in row.candidate_id for row in rows)
    assert all(row.nominal_lf_payload_bytes < 1_000 for row in rows)
    assert all(row.nominal_step_map_payload_bytes < 1_000 for row in rows)


def test_snerv_official_skip_high_modes_are_charged_by_storage_shape() -> None:
    rows = {
        mode: analyze_snerv_modelsize_candidate(
            hard_byte_ceiling=178_000,
            num_pairs=600,
            carrier_hw=(384, 512),
            wavelet="haar",
            levels=1,
            bits_per_coeff=1.5,
            step_map_bits_per_coeff=0.5,
            decoder_payload_codec="int8_symmetric",
            snerv_model_size_adapter="snerv_official_mfu_hfr_tub_primitives_adapter",
            official_modelsize_mparams=0.05,
            official_skip_high_mode=mode,
        )
        for mode in ("full", "shared_mean", "channel_mean", "scalar_mean")
    }

    assert rows["full"].nominal_skip_high_payload_bytes > 1_000_000_000
    assert rows["shared_mean"].nominal_skip_high_payload_bytes == 1_179_648
    assert rows["channel_mean"].nominal_skip_high_payload_bytes == 24
    assert rows["scalar_mean"].nominal_skip_high_payload_bytes == 8
    assert {row.bits_per_coeff for row in rows.values()} == {1.0}
    assert {row.step_map_bits_per_coeff for row in rows.values()} == {1.0}
    assert rows["shared_mean"].nominal_under_ceiling is False
    assert rows["channel_mean"].nominal_under_ceiling is True
    assert rows["scalar_mean"].nominal_under_ceiling is True
    assert rows["shared_mean"].nominal_decoder_payload_bytes > rows[
        "channel_mean"
    ].nominal_decoder_payload_bytes


def test_snerv_modelsize_budget_can_consume_official_modelsize_formula() -> None:
    row = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=4,
        bits_per_coeff=2.0,
        step_map_bits_per_coeff=1.0,
        decoder_payload_codec="int4_symmetric",
        official_modelsize_mparams=0.05,
        emb_size=0,
    )

    assert row.modelsize_mparams == 0.05
    assert row.capacity_source == "official_snerv_modelsize"
    assert row.modelsize_control_profile_id == DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID
    assert row.modelsize_control_profile["profile_id"] == SNERV_CONTEST_RECEIVER_PROFILE_ID
    assert row.fc_dim == 11
    assert row.official_modelsize_solution is not None
    assert row.official_modelsize_solution["schema"] == (
        "official_snerv_modelsize_to_fc_dim.v1"
    )
    assert row.official_modelsize_solution["fc_dim"] == 11
    assert row.official_modelsize_solution["score_claim"] is False
    assert row.official_modelsize_solution["ready_for_exact_eval_dispatch"] is False

    report = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=8,
        official_modelsize_mparams=(0.05,),
        fc_dims=(),
    )

    assert report["official_modelsize_mparams"] == [0.05]
    assert report["modelsize_control_profile_id"] == SNERV_CONTEST_RECEIVER_PROFILE_ID
    assert report["modelsize_control_profile"]["source"] == "pact_receiver_closed_snar1_profile"
    assert report["official_enc_strds"] == [5, 4, 2, 2, 2]
    assert report["official_dec_strds"] == [5, 4, 2, 2, 2]
    assert report["candidate_count"] > 0
    assert {
        row["official_modelsize_solution"]["fc_dim"]
        for row in report["selected_candidates"]
        if row["official_modelsize_solution"] is not None
    } == {11}
    assert all(
        row["modelsize_mparams"] == 0.05 for row in report["selected_candidates"]
    )
    assert {row["capacity_source"] for row in report["selected_candidates"]} == {
        "official_snerv_modelsize"
    }
    assert {
        row["modelsize_control_profile_id"] for row in report["selected_candidates"]
    } == {SNERV_CONTEST_RECEIVER_PROFILE_ID}


def test_snerv_official_cli_default_profile_is_explicit_and_fail_closed() -> None:
    profile = snerv_modelsize_control_profile(SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID)

    assert profile["enc_strds"] == []
    assert profile["dec_strds"] == [5, 4, 3, 2, 2]
    assert profile["modelsize_solve_supported"] is False

    report = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=8,
        official_modelsize_mparams=(0.05,),
        fc_dims=(),
        official_enc_strds=(),
        official_dec_strds=(5, 4, 3, 2, 2),
        modelsize_control_profile_id=SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID,
    )

    assert report["modelsize_control_profile_id"] == SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID
    assert report["official_enc_strds"] == []
    assert report["official_dec_strds"] == [5, 4, 3, 2, 2]
    assert report["candidate_count"] == 0
    assert report["invalid_candidate_count"] > 0
    invalid = report["invalid_candidates"][0]
    assert invalid["modelsize_control_profile_id"] == SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID
    assert invalid["modelsize_control_profile"]["blockers"] == [
        "official_cli_default_enc_strds_empty_requires_source_context"
    ]
    assert invalid["score_claim"] is False
    assert invalid["ready_for_exact_eval_dispatch"] is False


def test_snerv_modelsize_budget_records_invalid_official_modelsize_without_aborting() -> None:
    report = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=4,
        carrier_hw=(384, 512),
        wavelet="haar",
        fc_dims=(9,),
        emb_sizes=(0, 2),
        official_modelsize_mparams=(0.05,),
        temporal_context=2,
        temporal_modes=("delta", "official_haar_dwt1d_lowpass"),
    )

    assert report["schema"] == "snerv_modelsize_budget.v1"
    assert report["candidate_count"] > 0
    assert report["selected_candidate_count"] > 0
    assert report["invalid_candidate_count"] > 0
    invalid = report["invalid_candidates"][0]
    assert invalid["schema"] == "snerv_invalid_official_modelsize_candidate.v1"
    assert invalid["official_modelsize_mparams"] == 0.05
    assert invalid["emb_size"] == 2
    assert invalid["error_type"] == "SnervCarrierError"
    assert "snerv_official_modelsize_quadratic_unsatisfied" in invalid["blockers"]
    contract = invalid["modelsize_control_contract"]
    assert contract["schema"] == "nerv_modelsize_control_contract.v1"
    assert contract["control_semantics"] == (
        "invalid_official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert contract["shared_target_modelsize_mparams_consumed_as"] == (
        "official_snerv_modelsize_quadratic_fc_dim_solve"
    )
    assert contract["modelsize_mparams_is_official_upstream_flag"] is True
    assert contract["modelsize_mparams_caps_archive_zip_bytes"] is False
    assert contract["authority_split"]["modelsize_mparams_semantics"] == (
        "official_upstream_parameter_budget_control"
    )
    assert contract["authority_split"]["invalid_control_row"] is True
    assert contract["authority_split"]["ready_for_exact_eval_dispatch"] is False
    assert contract["nominal_payload_bytes_are_planner_prior_only"] is True
    assert contract["receiver_closed_archive_bytes_required_for_under_ceiling_claim"]
    assert contract["archive_bytes_authority_required"] is True
    assert contract["mutates_receiver_visible_fc_dim"] is False
    assert contract["invalid_control_row"] is True
    assert contract["control_resolution_status"] == (
        "failed_before_receiver_visible_fc_dim_candidate"
    )
    assert invalid["score_claim"] is False
    assert invalid["rank_or_kill_eligible"] is False
    assert invalid["ready_for_exact_eval_dispatch"] is False
    assert report["score_claim"] is False


def test_snerv_modelsize_selection_preserves_official_source_metadata() -> None:
    report = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=6,
        fc_dims=(11,),
        official_modelsize_mparams=(0.05,),
    )

    selected = report["selected_candidates"]
    assert selected
    assert any(row["candidate_id"].find("_fc11e0_") >= 0 for row in selected)
    assert any(
        row["modelsize_mparams"] == 0.05
        and row["capacity_source"] == "official_snerv_modelsize"
        and row["official_modelsize_solution"]["fc_dim"] == 11
        and row["modelsize_control_profile_id"]
        == DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID
        and row["modelsize_control_profile"]["profile_id"]
        == SNERV_CONTEST_RECEIVER_PROFILE_ID
        and row["modelsize_control_profile"]["source"]
        == "pact_receiver_closed_snar1_profile"
        for row in selected
        if row["candidate_id"].find("_fc11e0_") >= 0
    )


def test_snerv_modelsize_control_profiles_are_explicit_source_contracts() -> None:
    cli_default = snerv_modelsize_control_profile(SNERV_OFFICIAL_CLI_DEFAULT_PROFILE_ID)
    assert cli_default["source"] == "upstream_train_snerv_parser_defaults"
    assert cli_default["enc_strds"] == []
    assert cli_default["dec_strds"] == [5, 4, 3, 2, 2]
    assert cli_default["modelsize_solve_supported"] is False
    assert "official_cli_default_enc_strds_empty_requires_source_context" in cli_default[
        "blockers"
    ]

    contest = snerv_modelsize_control_profile(SNERV_CONTEST_RECEIVER_PROFILE_ID)
    assert contest["source"] == "pact_receiver_closed_snar1_profile"
    assert contest["source_family"] == "PACT SNeRV receiver adapter"
    assert "Not an upstream README default" in contest["source_notes"]
    assert contest["enc_strds"] == [5, 4, 2, 2, 2]
    assert contest["dec_strds"] == [5, 4, 2, 2, 2]
    assert contest["modelsize_solve_supported"] is True
    assert contest["blockers"] == []

    with pytest.raises(NervModelSizeBudgetError):
        snerv_modelsize_control_profile("not-a-profile")


def test_snerv_modelsize_manual_stride_override_is_not_labeled_official_default() -> None:
    row = analyze_snerv_modelsize_candidate(
        hard_byte_ceiling=178_000,
        num_pairs=600,
        carrier_hw=(384, 512),
        wavelet="haar",
        levels=4,
        bits_per_coeff=2.0,
        step_map_bits_per_coeff=1.0,
        decoder_payload_codec="int4_symmetric",
        official_modelsize_mparams=0.05,
        official_enc_strds=(5, 4, 2, 2, 2),
        official_dec_strds=(5, 4, 3, 2, 2),
    )

    payload = row.as_dict()
    profile = payload["modelsize_control_profile"]
    assert payload["modelsize_control_profile_id"] == "manual_stride_override"
    assert profile["base_profile_id"] == SNERV_CONTEST_RECEIVER_PROFILE_ID
    assert profile["source"] == "operator_or_tool_explicit_stride_override"
    assert profile["enc_strds"] == [5, 4, 2, 2, 2]
    assert profile["dec_strds"] == [5, 4, 3, 2, 2]
    assert profile["modelsize_solve_supported"] is True
    assert payload["modelsize_control_contract"][
        "modelsize_mparams_is_official_upstream_flag"
    ] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_official_nerv_oss_flag_audit_maps_controls_to_local_consumers() -> None:
    audit = official_nerv_oss_flag_audit()

    assert audit["schema"] == "nerv_oss_flag_audit.v1"
    assert "--modelsize" in audit["hnerv_high_ev_flags"]
    assert "--modelsize" in audit["snerv_high_ev_flags"]
    assert "--quant-level" in audit["hinerv_high_ev_flags"]
    maps = {row["control_family"]: row for row in audit["control_to_local_consumer_map"]}
    assert "archive_byte_capacity" in maps
    assert "scorer_saliency_and_inverse_steganalysis" in maps
    assert "tac.analysis.score_exact_saliency" in maps[
        "scorer_saliency_and_inverse_steganalysis"
    ]["local_consumers"]
    priors = {row["variant"]: row for row in audit["cross_variant_design_priors"]}
    assert "HNeRV / PR95-HNeRV" in priors
    assert "SR-NeRV" in priors
    assert "RNeRV / E-NeRV" in priors
    assert audit["score_claim"] is False
