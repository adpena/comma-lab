# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV rate-sweep adjudication."""

from __future__ import annotations

from tac.analysis.snerv_rate_adjudication import (
    AXIS_TAG,
    build_snerv_rate_adjudication_payload,
)

_SHA_A = "4d00368c7ae6" + "0" * 52
_SHA_B = "86f09bc76323" + "1" * 52


def test_legacy_undercharged_row_is_blocked_even_when_rate_low() -> None:
    payload = [
        {
            "levels": 3,
            "bits": 1.5,
            "archive_bytes": 151_632,
            "lf_bytes": 151_216,
            "d_seg_linf": 0.00967,
            "d_pose_linf": 0.08663,
            "score_linf": 1.999,
            "beats_frontier": True,
        }
    ]
    report = build_snerv_rate_adjudication_payload(payload)
    row = report["rows"][0]

    assert row["classification"] == "legacy_undercharged_requires_step_map_replay"
    assert row["beats_frontier_rate_only"] is True
    assert row["frontier_score_claim"] is False
    assert row["score_claim"] is False
    assert "linf_step_map_payload_missing_or_legacy_undercharged" in row["blockers"]
    assert "receiver_packet_bytes_not_contest_archive_zip_bytes" in row["blockers"]
    assert report["exact_readiness_refusal"]["ready"] is False


def test_step_map_charged_distortion_promising_row_is_rate_blocked() -> None:
    payload = {
        "four_pair_comparable_rows": [
            {
                "levels": 3,
                "config": {"target_bits_per_coeff": 1.5},
                "archive_bytes_total": 264_396,
                "lf_payload_bytes": 144_072,
                "linf_steps_payload_bytes": 119_908,
                "decoder_bytes": 224,
                "metadata_bytes": 192,
                "d_seg_mean_linf": 0.01059,
                "d_pose_mean_linf": 0.03202,
                "score_linf": 1.80134,
                "score_l2": 2.813,
            }
        ]
    }
    report = build_snerv_rate_adjudication_payload(payload)
    row = report["rows"][0]

    assert row["step_map_accounting"] == "charged_receiver_visible_payload"
    assert row["step_map_overhead_bytes"] == 119_908
    assert row["classification"] == "distortion_promising_step_map_rate_blocked"
    assert row["beats_frontier_rate_only"] is False
    assert row["axis_tag"] == AXIS_TAG


def test_step_map_charged_low_rate_pose_destroyed_row_is_not_promoted() -> None:
    payload = [
        {
            "levels": 4,
            "bits": 5.0,
            "archive_bytes_total": 143_672,
            "lf_payload_bytes": 95_416,
            "linf_steps_payload_bytes": 47_788,
            "decoder_bytes": 276,
            "metadata_bytes": 192,
            "d_seg_mean_linf": 0.02188,
            "d_pose_mean_linf": 2.30867,
            "score_linf": 7.089,
        }
    ]
    report = build_snerv_rate_adjudication_payload(payload)
    row = report["rows"][0]

    assert row["beats_frontier_rate_only"] is True
    assert row["classification"] == "rate_below_frontier_pose_or_seg_destroyed"
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_step_map_charged_sweep_without_explicit_step_field_infers_overhead() -> None:
    payload = [
        {
            "levels": 3,
            "bits": 1.5,
            "archive_bytes": 273_000,
            "lf_bytes": 151_216,
            "decoder_bytes": 224,
            "metadata_bytes": 192,
            "step_map_charged": True,
            "d_seg_linf": 0.00967,
            "d_pose_linf": 0.08663,
        }
    ]
    row = build_snerv_rate_adjudication_payload(payload)["rows"][0]

    assert row["linf_steps_payload_bytes"] == 121_368
    assert row["step_map_accounting"] == "charged_receiver_visible_payload"
    assert row["classification"] == "distortion_promising_step_map_rate_blocked"


def test_compact_step_map_codec_metadata_survives_adjudication() -> None:
    payload = [
        {
            "levels": 3,
            "archive_bytes_total": 67_444,
            "lf_payload_bytes": 58_640,
            "linf_steps_payload_bytes": 8_532,
            "linf_steps_payload_codec": "snerv_step_map_coder.v1",
            "linf_steps_coder_mode": "uniform",
            "linf_steps_coder_bins": 64,
            "linf_steps_fp32_lzma_baseline_bytes": 39_060,
            "d_seg_mean_linf": 0.00885,
            "d_pose_mean_linf": 0.00462,
        }
    ]
    row = build_snerv_rate_adjudication_payload(payload)["rows"][0]

    assert row["classification"] == "rate_promising_runtime_unclosed"
    assert row["linf_steps_payload_codec"] == "snerv_step_map_coder.v1"
    assert row["linf_steps_coder_mode"] == "uniform"
    assert row["linf_steps_coder_bins"] == 64
    assert row["linf_steps_fp32_lzma_baseline_bytes"] == 39_060
    assert row["linf_steps_payload_vs_fp32_baseline_ratio"] == 8_532 / 39_060
    assert (
        "contest_receiver_archive_parser_not_yet_wired_to_compact_step_map_packet"
        in row["blockers"]
    )


def test_adaptive_step_map_group_metadata_survives_adjudication() -> None:
    payload = [
        {
            "levels": 4,
            "archive_bytes_total": 27_540,
            "receiver_archive_packet_bytes": 27_540,
            "receiver_archive_sha256": _SHA_B,
            "receiver_archive_replay_verified": True,
            "linf_steps_payload_bytes": 2_774,
            "linf_steps_payload_codec": "snerv_step_map_coder.adaptive.v1",
            "linf_steps_coder_mode": "adaptive",
            "linf_steps_coder_bins": 0,
            "linf_steps_coder_groups": [
                {"bins": 128, "map_indices": [5]},
                {"bins": 16, "map_indices": [2, 3, 4]},
                {"bins": 0, "kind": "constant_log2_fill", "map_indices": [0, 1]},
            ],
            "d_seg_mean_linf": 0.02284,
            "d_pose_mean_linf": 2.47515,
        }
    ]
    row = build_snerv_rate_adjudication_payload(payload)["rows"][0]

    assert row["linf_steps_payload_codec"] == "snerv_step_map_coder.adaptive.v1"
    assert row["linf_steps_coder_mode"] == "adaptive"
    assert row["linf_steps_coder_bins"] == 0
    assert {group["bins"] for group in row["linf_steps_coder_groups"]} == {
        128,
        16,
        0,
    }
    report = build_snerv_rate_adjudication_payload(payload)
    assert (
        report["summary"]["actionable_next_code_move"]
        == "score_aware_stepmap_waterfill_and_decoder_fit_before_packaging"
    )


def test_receiver_archive_packet_proof_replaces_parser_blocker() -> None:
    payload = [
        {
            "levels": 4,
            "archive_bytes_total": 16_939,
            "receiver_archive_packet_bytes": 16_939,
            "receiver_archive_header_bytes": 1_028,
            "receiver_archive_sha256": _SHA_A,
            "receiver_archive_replay_verified": True,
            "lf_payload_bytes": 13_300,
            "linf_steps_payload_bytes": 2_311,
            "linf_steps_payload_codec": "snerv_step_map_coder.v1",
            "linf_steps_coder_bins": 16,
            "linf_steps_fp32_lzma_baseline_bytes": 7_472,
            "metadata_bytes": 24,
            "d_seg_mean_linf": 0.02498,
            "d_pose_mean_linf": 3.19430,
        }
    ]
    report = build_snerv_rate_adjudication_payload(payload)
    row = report["rows"][0]

    assert row["receiver_archive_packet_bytes"] == 16_939
    assert row["receiver_archive_header_bytes"] == 1_028
    assert row["receiver_archive_sha256"] == _SHA_A
    assert row["receiver_archive_replay_verified"] is True
    assert (
        "contest_receiver_archive_parser_not_yet_wired_to_compact_step_map_packet"
        not in row["blockers"]
    )
    assert (
        "full_frame_inflate_runtime_not_yet_wired_to_snerv_archive_packet"
        not in row["blockers"]
    )
    assert (
        "not_packaged_as_contest_archive_zip"
        in row["blockers"]
    )
    assert (
        report["summary"]["actionable_next_code_move"]
        == "score_aware_stepmap_waterfill_and_decoder_fit_before_packaging"
    )


def test_replay_verified_distortion_promising_row_routes_to_packaging() -> None:
    payload = [
        {
            "levels": 4,
            "archive_bytes_total": 16_939,
            "receiver_archive_packet_bytes": 16_939,
            "receiver_archive_header_bytes": 1_028,
            "receiver_archive_sha256": _SHA_A,
            "receiver_archive_replay_verified": True,
            "lf_payload_bytes": 13_300,
            "linf_steps_payload_bytes": 2_311,
            "linf_steps_payload_codec": "snerv_step_map_coder.v1",
            "linf_steps_coder_bins": 16,
            "linf_steps_fp32_lzma_baseline_bytes": 7_472,
            "metadata_bytes": 24,
            "d_seg_mean_linf": 0.008,
            "d_pose_mean_linf": 0.04,
        }
    ]
    report = build_snerv_rate_adjudication_payload(payload)

    assert report["rows"][0]["classification"] == "rate_promising_runtime_unclosed"
    assert (
        report["summary"]["actionable_next_code_move"]
        == "contest_archive_zip_packaging_full600_and_paired_cpu_cuda_replay"
    )


def test_short_receiver_archive_sha_does_not_replace_parser_blocker() -> None:
    payload = [
        {
            "levels": 4,
            "archive_bytes_total": 16_939,
            "receiver_archive_packet_bytes": 16_939,
            "receiver_archive_header_bytes": 1_028,
            "receiver_archive_sha256": "abc",
            "receiver_archive_replay_verified": True,
            "lf_payload_bytes": 13_300,
            "linf_steps_payload_bytes": 2_311,
            "linf_steps_payload_codec": "snerv_step_map_coder.v1",
            "linf_steps_coder_bins": 16,
            "d_seg_mean_linf": 0.02498,
            "d_pose_mean_linf": 3.19430,
        }
    ]
    row = build_snerv_rate_adjudication_payload(payload)["rows"][0]

    assert row["receiver_archive_sha256"] == "abc"
    assert "receiver_archive_sha256_invalid_or_missing" in row["blockers"]
    assert (
        "contest_receiver_archive_parser_not_yet_wired_to_compact_step_map_packet"
        in row["blockers"]
    )
    assert "not_packaged_as_contest_archive_zip" not in row["blockers"]


def test_archive_bytes_without_replay_do_not_replace_parser_blocker() -> None:
    payload = [
        {
            "levels": 4,
            "archive_bytes_total": 16_939,
            "receiver_archive_packet_bytes": 16_939,
            "receiver_archive_header_bytes": 1_028,
            "receiver_archive_sha256": _SHA_A,
            "lf_payload_bytes": 13_300,
            "linf_steps_payload_bytes": 2_311,
            "linf_steps_payload_codec": "snerv_step_map_coder.v1",
            "linf_steps_coder_bins": 16,
            "d_seg_mean_linf": 0.02498,
            "d_pose_mean_linf": 3.19430,
        }
    ]
    report = build_snerv_rate_adjudication_payload(payload)
    row = report["rows"][0]

    assert row["receiver_archive_replay_verified"] is False
    assert (
        "contest_receiver_archive_parser_not_yet_wired_to_compact_step_map_packet"
        in row["blockers"]
    )
    assert (
        report["summary"]["actionable_next_code_move"]
        == "compact_step_map_packet_and_snAR1_receiver_replay_closure"
    )


def test_summary_never_grants_frontier_score_authority() -> None:
    payload = [
        {
            "archive_bytes_total": 100_000,
            "linf_steps_payload_bytes": 10_000,
            "d_seg_mean_linf": 0.0,
            "d_pose_mean_linf": 0.0,
        }
    ]
    report = build_snerv_rate_adjudication_payload(payload)

    assert report["frontier_score_claim"] is False
    assert report["summary"]["any_frontier_score_claim"] is False
    assert report["exact_readiness_refusal"]["promotion_eligible"] is False
