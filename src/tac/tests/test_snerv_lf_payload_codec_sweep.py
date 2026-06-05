# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.snerv_lf_payload_codec_sweep import (
    SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA,
    SNERV_OFFICIAL_DUMMY_LF_PAYLOAD_CODEC_SWEEP_SCHEMA,
    build_snerv_lf_payload_codec_sweep,
    build_snerv_official_dummy_lf_payload_codec_sweep,
)
from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    CUT,
    DEMOTE,
    NERV_BYTE_PRICE_CONTROLLER_SCHEMA,
    PROTECT,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive_snar2,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tools.build_snerv_lf_payload_codec_sweep import _load_planes


def test_snerv_lf_payload_codec_sweep_is_rate_only_and_scorer_only() -> None:
    plane = np.zeros((32, 32), dtype=np.int64)
    plane[0, 0] = -1
    plane[10, 4] = 1

    report = build_snerv_lf_payload_codec_sweep(
        [plane],
        modes=("int64_lzma", "portfolio_auto", "int2"),
    )

    assert report["schema"] == SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["objective_authority"]["objective"] == "contest_auth_eval_scorer_only"
    assert "human_visual_fidelity" in report["objective_authority"][
        "forbidden_selection_terms"
    ]
    assert report["codec_proof"] == (
        "snerv_lf_quant_payload.v2_receiver_visible_exact_intn_codec"
    )
    assert report["selected_rate_only_row"]["payload_bytes"] > 0
    assert report["selected_rate_only_row"]["payload_bytes"] < report["raw_i64_bytes"]
    assert "snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay" in report[
        "blockers"
    ]
    assert report["baseline_mode"] == "int64_lzma"
    assert report["section_value_rows"]
    plan = report["byte_price_plan"]
    assert plan["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA
    assert plan["source_schema"] == SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert plan["input_row_count"] == len(report["section_value_rows"])
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert "full_video_coverage_missing" in plan["blockers"]

    by_mode = {
        row["source"]["candidate_mode"]: row for row in plan["decision_rows"]
    }
    assert by_mode["portfolio_auto"]["receiver_proof_status"] == (
        "packet_exact_only_full_archive_replay_missing"
    )
    assert "receiver_proof_not_satisfied" in by_mode["portfolio_auto"]["blockers"]
    for row in by_mode.values():
        assert row["decision"] == DEMOTE
        if row["delta_total_score"] is None:
            continue
        assert row["economic_decision"] in {CUT, PROTECT}
        if row["byte_delta"] < 0:
            assert row["economic_decision"] == CUT
        else:
            assert row["economic_decision"] == PROTECT


def test_snerv_lf_payload_codec_sweep_marks_failed_modes() -> None:
    plane = np.array([[3]], dtype=np.int64)

    report = build_snerv_lf_payload_codec_sweep([plane], modes=("int2",))
    row = report["rows"][0]

    assert row["payload_bytes"] == 0
    assert row["error"]
    assert "snerv_lf_payload_codec_mode_failed" in row["blockers"]
    plan_row = report["byte_price_plan"]["decision_rows"][0]
    assert plan_row["delta_total_score"] is None
    assert "snerv_lf_payload_codec_mode_failed" in plan_row["blockers"]


def test_snerv_lf_payload_codec_sweep_never_selects_failed_zero_byte_mode() -> None:
    plane = np.array([[3, 0], [0, 0]], dtype=np.int64)

    report = build_snerv_lf_payload_codec_sweep(
        [plane],
        modes=("int2", "int64_lzma"),
    )

    assert report["selected_rate_only_row"]["mode"] == "int64_lzma"
    assert report["selected_rate_only_row"]["payload_bytes"] > 0
    assert report["selected_rate_only_row"]["error"] is None
    assert report["failed_modes"] == [
        {
            "mode": "int2",
            "error": "signed_int2_bitpack requires values in [-2, 1]",
        }
    ]

    by_mode = {
        row["source"]["candidate_mode"]: row
        for row in report["byte_price_plan"]["decision_rows"]
    }
    assert "snerv_lf_payload_codec_mode_failed" in by_mode["int2"]["blockers"]
    assert "snerv_lf_payload_codec_mode_failed" not in by_mode["int64_lzma"][
        "blockers"
    ]


def test_lf_payload_codec_sweep_loader_labels_raw_snar2_packet(tmp_path) -> None:
    lf_planes = [
        np.arange(4, dtype=np.int64).reshape(2, 2),
        -np.arange(4, dtype=np.int64).reshape(2, 2),
    ]
    step_packet = encode_step_maps(
        [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 2.0, dtype=np.float32)],
        bins=16,
    )
    archive = pack_snerv_archive_snar2(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0, 1.0]),
        lf_payload=encode_lf_quant_payload(lf_planes),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=1)),
        step_map_packet=step_packet.packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "lf_plane_count": 2,
            "levels": 1,
            "wavelet": "haar",
            "carrier_hw": [4, 4],
        },
    )
    packet_path = tmp_path / "candidate.snar2"
    packet_path.write_bytes(archive.packet)

    loaded_planes, source = _load_planes(packet_path, None)

    assert source["kind"] == "raw_snar2_packet"
    assert source["bytes"] == len(archive.packet)
    assert source["packet_sha256"] == source["sha256"]
    for expected, actual in zip(lf_planes, loaded_planes, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_snerv_official_dummy_lf_payload_codec_sweep_prices_receiver_sections() -> None:
    report = build_snerv_official_dummy_lf_payload_codec_sweep(
        modes=("int64_lzma", "spatial_delta_zigzag_leb128", "portfolio_auto", "int2"),
        hard_byte_ceiling=285_000,
    )

    assert report["schema"] == SNERV_OFFICIAL_DUMMY_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert report["source_schema"] == SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["receiver_lf_section_role"] == (
        "snar1_required_placeholder_official_decoder_payload_renders_frames"
    )
    assert report["official_mfu_hfr_tub_decoder_payload_renders_frames"] is True
    assert report["full_level1_lf_grid_required_for_receiver_frames"] is False
    assert report["lf_plane_count"] == 1
    assert report["lf_coeff_count_total"] == 1
    assert report["metadata_payload_bytes"] == 4
    assert report["step_map_packet_bytes"] > 0
    assert report["selected_rate_only_row"]["under_hard_byte_ceiling"] is True
    assert report["selected_rate_only_row"]["receiver_section_total_bytes"] < 285_000
    assert all(
        row["receiver_section_total_bytes"] < 285_000
        for row in report["rows"]
        if not row["error"]
    )
    assert "snerv_official_dummy_lf_has_receiver_section_only_no_trained_official_payload" in (
        report["blockers"]
    )

    plan = report["byte_price_plan"]
    assert plan["source_schema"] == SNERV_OFFICIAL_DUMMY_LF_PAYLOAD_CODEC_SWEEP_SCHEMA
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert "full_video_coverage_missing" in plan["blockers"]
