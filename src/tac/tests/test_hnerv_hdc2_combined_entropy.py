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
        },
    )

    accounting = manifest["byte_accounting"]
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["blockers"] == []
    assert accounting["net_byte_delta_now"] == 51254
    assert accounting["net_byte_delta_after_model_overhead_only"] == 10414
    assert accounting["minimum_payload_reduction_needed_after_zero_model_overhead_bytes"] == 10415
    assert accounting["net_byte_delta_after_combined_targets"] == -13565
    assert math.isclose(
        accounting["projected_rate_score_delta_after_combined_targets"],
        -13565 * 25.0 / 37_545_489,
    )

