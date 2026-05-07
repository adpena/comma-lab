from __future__ import annotations

import math

from tac.hnerv_hdc2_combined_entropy import build_hdc2_combined_entropy_manifest


def test_hdc2_combined_entropy_manifest_requires_payload_and_model_reduction() -> None:
    manifest = build_hdc2_combined_entropy_manifest(
        {
            "current_frontier": {
                "archive_sha256": "a" * 64,
                "archive_bytes": 186080,
                "score": 0.20935073680571203,
            },
            "next_entropy_research_action": {
                "target_label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                "target_section": "decoder_packed_brotli",
            },
            "combined_entropy_gap_groups": [
                {
                    "target_label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "frontier_section": "decoder_packed_brotli",
                    "frontier_section_bytes": 170127,
                    "actual_replacement_bytes": 221381,
                }
            ],
        },
        {
            "entropy_overhead_target_ranking": [
                {
                    "label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "target_kind": "known_model_overhead",
                    "target_bytes": 40840,
                },
                {
                    "label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "target_kind": "known_payload_entropy_gap",
                    "target_bytes": 23979,
                },
            ]
        },
        {
            "source_stream_section_manifest": {
                "stream": {
                    "sha256": "b" * 64,
                    "decoded_raw_sha256": "c" * 64,
                }
            },
            "roundtrip_decode_validation_manifest": {
                "roundtrip_valid": True,
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "candidate_stream_sha256": "d" * 64,
                "candidate_stream_bytes": 221381,
            },
            "decoded_output_equivalence_report": {
                "decoded_output_equal": True,
                "old_new_sha256_equal": True,
            },
            "bounded_hdc2_recode_variants": [
                {
                    "variant": (
                        "mixed_range_raw_global_prev_symbol_schema_indexed_"
                        "q_streams_plus_raw_scales"
                    ),
                    "codec": "HDM2_global_prev_symbol_mixed_range_raw_schema_indexed_uint8",
                    "bytes": 208821,
                    "raw_equal": True,
                    "q_roundtrip_equal": True,
                    "scale_roundtrip_equal": True,
                    "archive_ready": False,
                    "header_bytes": 19329,
                    "mixed_payload_bytes": 189380,
                    "range_payload_bytes": 147466,
                    "raw_payload_bytes": 41914,
                    "raw_scale_bytes": 112,
                    "raw_context_count": 187,
                    "range_context_count": 60,
                    "schema_metadata_elided_vs_hdc2_bytes": 444,
                }
            ],
        },
    )

    accounting = manifest["byte_accounting"]
    candidate = manifest["bounded_candidate"]
    break_even = manifest["bounded_candidate_break_even_plan"]
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["blockers"] == ["bounded_candidate_archive_build_gate_not_passed"]
    assert accounting["net_byte_delta_now"] == 51254
    assert accounting["net_byte_delta_after_model_overhead_only"] == 10414
    assert accounting["minimum_payload_reduction_needed_after_zero_model_overhead_bytes"] == 10415
    assert accounting["net_byte_delta_after_combined_targets"] == -13565
    assert accounting["best_bounded_candidate_bytes"] == 208821
    assert accounting["bounded_candidate_reduction_vs_hdc2_bytes"] == 12560
    assert accounting["net_byte_delta_after_best_bounded_candidate"] == 38694
    assert (
        accounting["remaining_reduction_to_beat_frontier_after_best_bounded_candidate_bytes"]
        == 38695
    )
    assert candidate["score_claim"] is False
    assert candidate["planning_only"] is True
    assert candidate["ready_for_exact_eval_dispatch"] is False
    assert candidate["archive_build_gate"] is False
    assert candidate["archive_build_gate_blockers"] == [
        "candidate_stream_bytes_not_less_than_current_frontier_section_bytes"
    ]
    assert candidate["byte_reduction_vs_hdc2_bytes"] == 12560
    assert candidate["static_context_header_reduction_vs_hdc2_bytes"] == 21511
    assert candidate["payload_delta_vs_hdc2_bytes"] == 8951
    assert candidate["raw_context_count"] == 187
    assert candidate["range_payload_bytes"] == 147466
    assert candidate["raw_payload_bytes"] == 41914
    assert break_even["current_best_variant"] == candidate["variant"]
    assert break_even["remaining_reduction_to_beat_frontier_section_bytes"] == 38695
    assert break_even["archive_build_gate"] is False
    assert break_even["archive_build_gate_rule"] == (
        "candidate_stream_bytes_must_be_less_than_current_frontier_section_bytes_"
        "and_raw_equality_closed"
    )
    assert break_even["archive_build_gate_blockers"] == candidate["archive_build_gate_blockers"]
    assert break_even["required_next_reduction_sources"] == [
        "reduce_range_payload_bytes_with_better_context_or_escape_policy",
        "reduce_raw_payload_bytes_with_secondary_entropy_code_or_run_model",
        "elide_static_context_header_bytes_without_reintroducing_self_describing_schema",
    ]
    assert math.isclose(
        accounting["projected_rate_score_delta_after_combined_targets"],
        -13565 * 25.0 / 37_545_489,
    )


def test_hdc2_combined_entropy_manifest_selects_hdm3_when_byte_gate_closes() -> None:
    manifest = build_hdc2_combined_entropy_manifest(
        {
            "current_frontier": {
                "archive_sha256": "a" * 64,
                "archive_bytes": 186080,
                "score": 0.20935073680571203,
            },
            "next_entropy_research_action": {
                "target_label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                "target_section": "decoder_packed_brotli",
            },
            "combined_entropy_gap_groups": [
                {
                    "target_label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "frontier_section": "decoder_packed_brotli",
                    "frontier_section_bytes": 170127,
                    "actual_replacement_bytes": 221381,
                }
            ],
        },
        {
            "entropy_overhead_target_ranking": [
                {
                    "label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "target_kind": "known_model_overhead",
                    "target_bytes": 40840,
                },
                {
                    "label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "target_kind": "known_payload_entropy_gap",
                    "target_bytes": 23979,
                },
            ]
        },
        {
            "source_stream_section_manifest": {
                "stream": {
                    "sha256": "b" * 64,
                    "decoded_raw_sha256": "c" * 64,
                }
            },
            "roundtrip_decode_validation_manifest": {
                "roundtrip_valid": True,
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "candidate_stream_sha256": "d" * 64,
                "candidate_stream_bytes": 221381,
            },
            "decoded_output_equivalence_report": {
                "decoded_output_equal": True,
                "old_new_sha256_equal": True,
            },
            "bounded_hdc2_recode_variants": [
                {
                    "variant": (
                        "mixed_range_raw_global_prev_symbol_schema_indexed_"
                        "q_streams_plus_raw_scales"
                    ),
                    "codec": "HDM2_global_prev_symbol_mixed_range_raw_schema_indexed_uint8",
                    "bytes": 208821,
                    "raw_equal": True,
                    "q_roundtrip_equal": True,
                    "scale_roundtrip_equal": True,
                    "archive_ready": False,
                    "header_bytes": 19329,
                    "mixed_payload_bytes": 189380,
                    "range_payload_bytes": 147466,
                    "raw_payload_bytes": 41914,
                    "raw_scale_bytes": 112,
                    "raw_context_count": 187,
                    "range_context_count": 60,
                    "schema_metadata_elided_vs_hdc2_bytes": 444,
                },
                {
                    "variant": "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales",
                    "codec": "HDM3_fixed_schema_q_brotli_raw_scales",
                    "bytes": 166000,
                    "raw_equal": True,
                    "q_roundtrip_equal": True,
                    "scale_roundtrip_equal": True,
                    "archive_ready": False,
                    "header_bytes": 7,
                    "q_brotli_bytes": 165881,
                    "q_stream_bytes": 230000,
                    "raw_scale_bytes": 112,
                    "brotli_quality": 11,
                },
            ],
        },
    )

    accounting = manifest["byte_accounting"]
    candidate = manifest["bounded_candidate"]
    break_even = manifest["bounded_candidate_break_even_plan"]
    assert manifest["blockers"] == []
    assert candidate["variant"] == "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"
    assert candidate["archive_build_gate"] is True
    assert candidate["archive_build_gate_blockers"] == []
    assert candidate["raw_equality_closed"] is True
    assert candidate["net_byte_delta_vs_frontier_section"] == -4127
    assert candidate["remaining_reduction_to_beat_frontier_section_bytes"] == 0
    assert candidate["q_brotli_bytes"] == 165881
    assert candidate["payload_delta_vs_hdc2_bytes"] == -14548
    assert accounting["best_bounded_candidate_archive_build_gate"] is True
    assert accounting["net_byte_delta_after_best_bounded_candidate"] == -4127
    assert break_even["archive_build_gate"] is True
    assert break_even["archive_build_gate_blockers"] == []


def test_hdc2_combined_entropy_manifest_blocks_candidates_above_active_floor() -> None:
    manifest = build_hdc2_combined_entropy_manifest(
        {
            "current_frontier": {
                "archive_sha256": "b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f",
                "archive_bytes": 186080,
                "score": 0.20935073680571203,
            },
            "next_entropy_research_action": {
                "target_label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                "target_section": "decoder_packed_brotli",
            },
            "combined_entropy_gap_groups": [
                {
                    "target_label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "frontier_section": "decoder_packed_brotli",
                    "frontier_section_bytes": 170127,
                    "actual_replacement_bytes": 221381,
                }
            ],
        },
        {
            "entropy_overhead_target_ranking": [
                {
                    "label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "target_kind": "known_model_overhead",
                    "target_bytes": 40840,
                },
                {
                    "label": "public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts",
                    "target_kind": "known_payload_entropy_gap",
                    "target_bytes": 23979,
                },
            ]
        },
        {
            "source_stream_section_manifest": {
                "stream": {
                    "sha256": "b" * 64,
                    "decoded_raw_sha256": "c" * 64,
                }
            },
            "roundtrip_decode_validation_manifest": {
                "roundtrip_valid": True,
                "raw_equal": True,
                "q_roundtrip_equal": True,
                "scale_roundtrip_equal": True,
                "candidate_stream_sha256": "d" * 64,
                "candidate_stream_bytes": 221381,
            },
            "decoded_output_equivalence_report": {
                "decoded_output_equal": True,
                "old_new_sha256_equal": True,
            },
            "bounded_hdc2_recode_variants": [
                {
                    "variant": "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales",
                    "codec": "HDM3_fixed_schema_q_brotli_raw_scales",
                    "bytes": 170113,
                    "raw_equal": True,
                    "q_roundtrip_equal": True,
                    "scale_roundtrip_equal": True,
                    "archive_ready": False,
                    "header_bytes": 7,
                    "q_brotli_bytes": 169994,
                    "q_stream_bytes": 228958,
                    "raw_scale_bytes": 112,
                    "brotli_quality": 11,
                },
            ],
        },
        active_floor={
            "label": "pr103_on_pr106_a++",
            "archive_bytes": 185578,
            "archive_sha256": "ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce",
            "score": 0.2089810755823297,
        },
    )

    gate = manifest["active_floor_gate"]
    hdc2_gate = manifest["hdc2_direct_archive_runtime_gate"]
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "best_bounded_candidate_not_below_active_archive_floor" in manifest["blockers"]
    assert gate["active_floor_archive_bytes"] == 185578
    assert gate["direct_hdc2_projected_archive_bytes"] == 237334
    assert gate["direct_hdc2_byte_delta_vs_active_floor"] == 51756
    assert gate["best_bounded_candidate_projected_archive_bytes"] == 186066
    assert gate["best_bounded_candidate_byte_delta_vs_active_floor"] == 488
    assert gate["best_bounded_candidate_below_active_floor"] is False
    assert gate["blockers"] == [
        "not_below_active_archive_floor:185578",
        "requires_candidate_archive_below_active_floor_before_exact_eval",
    ]
    assert hdc2_gate["status"] == "fail_closed"
    assert hdc2_gate["missing_artifacts"] == [
        "hdc2_runtime_decoder_contract_with_inflate_consumer",
        "hdc2_archive_candidate_manifest_with_decoder_stream_consumed",
    ]
