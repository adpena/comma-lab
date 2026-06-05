# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV score-aware decoder-fit work orders."""

from __future__ import annotations

from tac.analysis.snerv_rate_adjudication import build_snerv_rate_adjudication_payload
from tac.analysis.snerv_score_aware_decoder_fit_work_order import (
    EXPECTED_NEXT,
    build_snerv_decoder_fit_work_order,
)


def test_waterfill_pose_destroyed_row_routes_to_local_decoder_fit_smoke() -> None:
    adjudication = build_snerv_rate_adjudication_payload(
        [
            {
                "levels": 4,
                "archive_bytes_total": 33_754,
                "receiver_archive_packet_bytes": 33_754,
                "receiver_archive_header_bytes": 1_233,
                "receiver_archive_sha256": "e69b773bb5e2",
                "receiver_archive_replay_verified": True,
                "lf_payload_bytes": 24_577,
                "linf_steps_payload_bytes": 7_182,
                "linf_steps_payload_codec": "snerv_step_map_coder.adaptive.v1",
                "linf_steps_coder_mode": "waterfill",
                "linf_steps_coder_groups": [
                    {"precision_label": "int4_bins16", "bins": 16, "map_indices": [1, 2, 5]},
                    {"precision_label": "int8_bins256", "bins": 256, "map_indices": [0, 3, 4]},
                ],
                "d_seg_mean_linf": 0.022644,
                "d_pose_mean_linf": 2.13907,
                "score_linf": 6.91189,
                "score_l2": 6.97034,
            }
        ]
    )
    work_order = build_snerv_decoder_fit_work_order(adjudication).as_jsonable()

    assert adjudication["summary"]["actionable_next_code_move"] == EXPECTED_NEXT
    assert work_order["ready_for_local_decoder_fit_smoke"] is True
    assert work_order["ready_for_exact_eval_dispatch"] is False
    assert work_order["score_claim"] is False
    assert work_order["promotion_eligible"] is False
    assert work_order["selected_classification"] == "rate_below_frontier_pose_or_seg_destroyed"
    assert work_order["current_step_map_mode"] == "waterfill"
    first, second = work_order["recommended_smoke_commands"]
    assert "tools/run_snerv_scorer_loop_decoder_qat_smoke.py" in first
    assert "--search-mode nes_pair_robust" in first
    assert "--byte-pressure-multiplier 8.0" in first
    assert "--max-archive-byte-growth 0" in first
    assert "--component-guard-mode score_primary" in first
    assert "--dynamic-range-repair-gains auto" in first
    assert "--progress-jsonl" in first
    assert "tools/run_snerv_inverse_steg_advisory.py" not in first
    assert "tools/build_snerv_scorer_loop_geometry.py" in second
    assert "snerv_score_aware_decoder_fit_after_work_order" in second


def test_undercharged_or_unreplayed_rows_refuse_decoder_fit_work_order() -> None:
    adjudication = build_snerv_rate_adjudication_payload(
        [
            {
                "levels": 4,
                "archive_bytes_total": 25_000,
                "receiver_archive_packet_bytes": 25_000,
                "receiver_archive_sha256": "abc",
                "receiver_archive_replay_verified": False,
                "d_seg_mean_linf": 0.03,
                "d_pose_mean_linf": 3.0,
            }
        ]
    )
    work_order = build_snerv_decoder_fit_work_order(adjudication).as_jsonable()

    assert work_order["ready_for_local_decoder_fit_smoke"] is False
    assert "no_replay_verified_low_rate_distortion_destroyed_row" in work_order["blockers"]
    assert "legacy_undercharged_step_map_payload" in work_order["blockers"]
    assert work_order["recommended_smoke_commands"] == ()


def test_distortion_promising_row_does_not_route_to_decoder_fit() -> None:
    adjudication = build_snerv_rate_adjudication_payload(
        [
            {
                "levels": 4,
                "archive_bytes_total": 33_754,
                "receiver_archive_packet_bytes": 33_754,
                "receiver_archive_sha256": "e69b773bb5e2",
                "receiver_archive_replay_verified": True,
                "linf_steps_payload_bytes": 7_182,
                "linf_steps_payload_codec": "snerv_step_map_coder.adaptive.v1",
                "linf_steps_coder_mode": "waterfill",
                "linf_steps_coder_groups": [
                    {"precision_label": "int8_bins256", "bins": 256, "map_indices": [0, 1]},
                ],
                "d_seg_mean_linf": 0.008,
                "d_pose_mean_linf": 0.04,
            }
        ]
    )
    work_order = build_snerv_decoder_fit_work_order(adjudication).as_jsonable()

    assert (
        adjudication["summary"]["actionable_next_code_move"]
        == "contest_archive_zip_packaging_full600_and_paired_cpu_cuda_replay"
    )
    assert work_order["ready_for_local_decoder_fit_smoke"] is False
    assert "adjudication_does_not_route_to_decoder_fit" in work_order["blockers"]
