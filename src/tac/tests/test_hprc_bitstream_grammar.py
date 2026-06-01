# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.substrates.hprc.archive import (
    HprcArchiveError,
    HprcPacketConfig,
    HprcSectionKind,
    pack_hprc_g1_packet,
    pack_hprc_packet,
    parse_hprc_packet,
)
from tac.substrates.hprc.bitstream_grammar import (
    HPRC_OPTIMAL_BITSTREAM_GRAMMAR_PLAN_SCHEMA,
    build_optimal_bitstream_grammar_plan,
    joint_p18_p19_saliency_contract,
)


def test_hprc_g1_roundtrip_removes_fixed_table_overhead() -> None:
    sections = {
        HprcSectionKind.DECODER_QW: b"decoder",
        HprcSectionKind.LATENTS_RC: b"latents",
        HprcSectionKind.RECEIVER_STATE: b"state",
    }
    config = HprcPacketConfig(decoder_family_id=95)

    v0 = pack_hprc_packet(sections, config=config)
    g1 = pack_hprc_g1_packet(sections, config=config)
    parsed = parse_hprc_packet(g1)

    assert len(g1) < len(v0)
    assert parsed.grammar == "hprc_g1_compact_bitmask_varint"
    assert parsed.config == config
    assert parsed.section_map() == sections
    manifest = parsed.manifest()
    assert manifest["magic"] == "HPRG"
    assert manifest["byte_accounting"]["section_table_bytes"] == 0
    assert manifest["byte_accounting"]["wrapper_overhead_bytes"] < len(sections) * 46


def test_hprc_g1_rejects_unknown_section_mask_bits() -> None:
    packet = b"HPRG" + bytes([1, 0, 0x80, 0x04])

    with pytest.raises(HprcArchiveError, match="section mask has unknown bits"):
        parse_hprc_packet(packet)


def test_optimal_bitstream_grammar_prices_sections_and_residuals() -> None:
    plan = build_optimal_bitstream_grammar_plan(
        hard_byte_ceilings=(178_000,),
        acquisition_rows=[
            {
                "family": "pr95_hnerv",
                "effective_archive_bytes": 120_000,
                "source_archive": {"path": "archive.zip", "sha256": "a" * 64},
                "section_rows": [
                    {"name": "decoder_qw", "bytes": 100_000},
                    {"name": "latents_rc", "bytes": 15_000},
                    {"name": "residual_rc", "bytes": 1000},
                ],
            }
        ],
    )

    assert plan["schema"] == HPRC_OPTIMAL_BITSTREAM_GRAMMAR_PLAN_SCHEMA
    row = plan["rows"][0]
    assert row["recommended_outer_grammar"] == "source_archive_native_single_member_zip"
    assert row["next_executable_task"] == (
        "run_full_video_value_per_byte_then_keep_only_negative_delta_residuals"
    )
    residual = next(
        section for section in row["section_rows"] if section["section"] == "residual_rc"
    )
    assert residual["admission_policy"] == (
        "admit_only_if measured_delta_nonrate_plus_rate_cost_lt_zero"
    )
    assert residual["rate_cost"] > 0


def test_optimal_bitstream_grammar_detects_saturated_and_unsaturated_entropy() -> None:
    plan = build_optimal_bitstream_grammar_plan(
        hard_byte_ceilings=(285_000,),
        acquisition_rows=[
            {
                "family": "z8_wavelet",
                "effective_archive_bytes": 200_000,
                "section_rows": [
                    {
                        "name": "decoder_qw",
                        "bytes": 100_000,
                        "entropy_profile": {
                            "coded_bytes": 100_000,
                            "empirical_floor_bytes": 99_500,
                        },
                    },
                    {
                        "name": "latents_rc",
                        "bytes": 50_000,
                        "entropy_profile": {
                            "coded_bytes": 50_000,
                            "structured_floor_bytes": 500,
                        },
                    },
                ],
            }
        ],
    )

    sections = {row["section"]: row for row in plan["rows"][0]["section_rows"]}
    assert sections["decoder_qw"]["entropy_gap"]["status"] == "entropy_saturated"
    assert sections["latents_rc"]["entropy_gap"]["status"] == "unsaturated_entropy_gap"
    assert plan["rows"][0]["next_executable_task"] == (
        "open_entropy_gap_materializer_for_unsaturated_sections"
    )
    assert plan["semantic_allocation_grammar"]["authority"]["mlx_or_proxy_rows"] == (
        "proposal_only"
    )


def test_joint_p18_p19_saliency_contract_keeps_frame_and_pair_channels_separate() -> None:
    contract = joint_p18_p19_saliency_contract()

    assert contract["do_not_collapse_channels_before_incidence_projection"] is True
    assert contract["segnet_frame_channel"]["scope"] == "frame_pixel_class_boundary"
    assert contract["posenet_pair_channel"]["scope"] == "pair_pixel_geometry"
    assert "not class-boundary-only" in contract["posenet_pair_channel"]["spatial_structure"]
    assert "sum_frame_incidence" in contract["combined_saliency"]["formula"]
    assert "sum_pair_incidence" in contract["combined_saliency"]["formula"]
