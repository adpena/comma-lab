# SPDX-License-Identifier: MIT
"""Tests for NeRV-family model-size budget planning."""

from __future__ import annotations

import pytest

from tac.analysis.nerv_modelsize_budget import (
    NervModelSizeBudgetError,
    analyze_hinerv_modelsize_candidate,
    analyze_snerv_modelsize_candidate,
    build_hinerv_config_from_size_knobs,
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
    official_nerv_oss_flag_audit,
    snerv_model_size_adapter_from_id_token,
    snerv_model_size_adapter_id_token,
)
from tac.substrates.hi_nerv.architecture import HinervSubstrate
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
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
        "hinerv_np17_ld12_ed16_dc10_hfg_cnx_int4_mixed_ceil178000"
    )
    assert row.use_hierarchical_feature_grid is True
    assert row.use_convnext_blocks is True
    assert row.mid_injection_block_index == cfg.mid_injection_block_index
    assert row.fine_injection_block_index == cfg.fine_injection_block_index
    assert row.total_trainable_params == model.num_parameters()
    assert row.decoder_trainable_params > row.latent_trainable_params
    assert row.score_claim is False


def test_hinerv_modelsize_budget_report_is_false_authority_and_budgeted() -> None:
    report = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=5,
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
    assert {"portfolio_auto", "int8_mixed", "int4_mixed", "int2_mixed"} <= codecs
    assert "selection_strategy" in report["budget_math"]


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
    assert row.emb_size == 2
    assert row.decoder_feature_count == 15
    assert row.hf_decoder_weight_count == 3 * 3 * 15
    assert row.mfu_scales == (1, 3)
    assert row.hfr_gain == 0.25
    assert row.temporal_context == 1
    assert row.lf_plane_count == 600 * 2 * 3
    assert row.lf_coeff_count_total == row.lf_plane_count * row.lf_coeffs_per_plane
    assert row.nominal_lf_payload_bytes == int(
        (row.lf_coeff_count_total * 2.0 + 7) // 8
    )
    assert row.requires_snAR1_archive_byte_oracle is True
    assert row.score_claim is False
    assert row.ready_for_exact_eval_dispatch is False

    report = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        per_ceiling_limit=8,
    )
    assert report["schema"] == "snerv_modelsize_budget.v1"
    assert report["wavelet"] == "haar"
    assert report["candidate_count"] > report["selected_candidate_count"] > 0
    assert report["budget_math"]["nominal_payload_is_not_authority"] is True
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
    assert snerv_model_size_adapter_id_token(
        SNERV_SPECTRA_PRESERVING_ADAPTER
    ) == "spectra"
    assert snerv_model_size_adapter_from_id_token("spectra") == (
        SNERV_SPECTRA_PRESERVING_ADAPTER
    )
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
