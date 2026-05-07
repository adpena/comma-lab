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
    assert manifest["blockers"] == []
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
    assert candidate["byte_reduction_vs_hdc2_bytes"] == 12560
    assert candidate["static_context_header_reduction_vs_hdc2_bytes"] == 21511
    assert candidate["payload_delta_vs_hdc2_bytes"] == 8951
    assert candidate["raw_context_count"] == 187
    assert candidate["range_payload_bytes"] == 147466
    assert candidate["raw_payload_bytes"] == 41914
    assert break_even["current_best_variant"] == candidate["variant"]
    assert break_even["remaining_reduction_to_beat_frontier_section_bytes"] == 38695
    assert break_even["archive_build_gate"] == (
        "candidate_stream_bytes_must_be_less_than_current_frontier_section_bytes"
    )
    assert break_even["required_next_reduction_sources"] == [
        "reduce_range_payload_bytes_with_better_context_or_escape_policy",
        "reduce_raw_payload_bytes_with_secondary_entropy_code_or_run_model",
        "elide_static_context_header_bytes_without_reintroducing_self_describing_schema",
    ]
    assert math.isclose(
        accounting["projected_rate_score_delta_after_combined_targets"],
        -13565 * 25.0 / 37_545_489,
    )
