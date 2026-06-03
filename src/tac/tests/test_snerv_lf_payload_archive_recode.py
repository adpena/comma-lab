# SPDX-License-Identifier: MIT
"""Tests for receiver-proof SNeRV LF payload archive recoding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.snerv_lf_payload_archive_recode import (
    build_snerv_lf_payload_archive_recode,
    build_snerv_lf_payload_recode_admission_plan,
    render_snerv_lf_payload_recode_admission_markdown,
)
from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tools.recode_snerv_lf_payload_archive import main as recode_main


def _packet(*, lf_codec: str = "int64_lzma") -> bytes:
    rng = np.random.default_rng(1591)
    lf_planes = [
        rng.integers(-8, 9, size=(4, 6), dtype=np.int64),
        rng.integers(-3, 4, size=(4, 6), dtype=np.int64),
    ]
    step_maps = [
        np.full((4, 6), 1.0 + 0.125 * idx, dtype=np.float32)
        for idx in range(len(lf_planes))
    ]
    return pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(
            lf_zero_points=[0.0 for _ in lf_planes],
        ),
        lf_payload=encode_lf_quant_payload(lf_planes, codec=lf_codec),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=1)),
        step_map_packet=encode_step_maps(step_maps, bins=4).packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 1,
            "height": 8,
            "width": 12,
            "orig_hw": [8, 12],
            "lf_plane_count": len(lf_planes),
            "levels": 1,
            "wavelet": "haar",
        },
    ).packet


def test_snerv_lf_payload_recode_preserves_receiver_lf_and_sections() -> None:
    source = _packet(lf_codec="int64_lzma")

    report, candidate = build_snerv_lf_payload_archive_recode(
        source,
        mode="spatial_delta_zigzag_leb128_lzma",
        source_packet_path="/tmp/source.snar",
        frame_proof_max_output_bytes=1,
    )

    source_decoded = unpack_snerv_archive(source)
    candidate_decoded = unpack_snerv_archive(candidate)
    assert report["schema"] == "snerv_lf_payload_archive_recode.v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["lf_planes_exact_equal"] is True
    assert report["receiver_contract_satisfied"] is True
    assert report["candidate_packet"]["bytes"] == len(candidate)
    for section in ("metadata_payload", "decoder_payload", "step_map_packet"):
        assert source_decoded.sections[section] == candidate_decoded.sections[section]
        assert report["unchanged_sections_exact"][section] is True
    for source_plane, candidate_plane in zip(
        source_decoded.decode_lf_quant_planes(),
        candidate_decoded.decode_lf_quant_planes(),
        strict=True,
    ):
        np.testing.assert_array_equal(source_plane, candidate_plane)


def test_recode_snerv_lf_payload_archive_cli_writes_matching_report(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "source.snar"
    output_packet = tmp_path / "candidate.snar"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    packet_path.write_bytes(_packet(lf_codec="int64_lzma"))

    rc = recode_main(
        [
            "--packet",
            packet_path.as_posix(),
            "--mode",
            "spatial_delta_zigzag_leb128_lzma",
            "--output-packet",
            output_packet.as_posix(),
            "--output-json",
            output_json.as_posix(),
            "--output-md",
            output_md.as_posix(),
            "--frame-proof-max-output-bytes",
            "1",
        ]
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    candidate = output_packet.read_bytes()
    assert rc == 0
    assert output_md.is_file()
    assert report["candidate_packet"]["file_matches_report"] is True
    assert report["candidate_packet"]["file_bytes"] == len(candidate)
    assert report["lf_planes_exact_equal"] is True
    assert report["receiver_contract_satisfied"] is True


def test_snerv_lf_payload_recode_admission_consumes_real_snar_recode() -> None:
    source = _packet(lf_codec="raw_i64")
    report, _candidate = build_snerv_lf_payload_archive_recode(
        source,
        mode="spatial_delta_zigzag_leb128_lzma",
        frame_proof_max_output_bytes=1,
    )
    hard_ceiling = int(report["source_packet"]["bytes"]) - 100

    plan = build_snerv_lf_payload_recode_admission_plan(
        [report],
        hard_byte_ceiling=hard_ceiling,
        candidate_id="snerv-real-snar-fixture",
        full_video_coverage=True,
    )

    row = plan["selected_row"]
    assert plan["selected_mode"] == "spatial_delta_zigzag_leb128_lzma"
    assert plan["local_planner_admitted"] is True
    assert plan["waterline_satisfied_after_selected_recode"] is True
    assert row["packet_byte_delta"] == (
        report["candidate_packet"]["bytes"] - report["source_packet"]["bytes"]
    )
    assert row["packet_rate_score_delta"] == pytest.approx(
        row["packet_byte_delta"] * plan["rate_score_per_byte"]
    )
    assert row["waterline_crossed_by_recode"] is True
    assert row["admission_decision"] == (
        "admit_lossless_lf_recode_crosses_byte_waterline"
    )
    assert row["ablation_decision"] == "no_lf_ablation_required_after_recode_waterline"
    assert plan["byte_price_plan"]["decision_rows"][0]["economic_decision"] == "cut"
    assert "paired_contest_cpu_cuda_auth_eval_missing" in plan["blockers"]


def test_snerv_lf_payload_recode_admission_prices_receiver_backed_savings() -> None:
    saving_report = {
        "schema": "snerv_lf_payload_archive_recode.v1",
        "mode": "zero_run_varint",
        "source_packet": {"bytes": 200_000, "sha256": "source-sha"},
        "candidate_packet": {"bytes": 160_000, "sha256": "candidate-sha"},
        "packet_byte_delta": -40_000,
        "lf_payload": {
            "source_bytes": 150_000,
            "candidate_bytes": 110_000,
            "byte_delta": -40_000,
        },
        "receiver_contract_satisfied": True,
        "receiver_frame_equality_proof": {"status": "proven_exact"},
        "blockers": [
            "not_packaged_as_contest_archive_zip",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    weaker_report = {
        **saving_report,
        "mode": "int16_brotli",
        "candidate_packet": {"bytes": 190_000, "sha256": "candidate-sha-2"},
        "packet_byte_delta": -10_000,
        "lf_payload": {
            "source_bytes": 150_000,
            "candidate_bytes": 140_000,
            "byte_delta": -10_000,
        },
    }

    plan = build_snerv_lf_payload_recode_admission_plan(
        [weaker_report, saving_report],
        hard_byte_ceiling=178_000,
        candidate_id="snerv-full600",
        full_video_coverage=True,
    )

    assert plan["schema"] == "snerv_lf_payload_recode_admission_plan.v1"
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["selected_mode"] == "zero_run_varint"
    assert plan["selected_row"]["waterline_crossed_by_recode"] is True
    assert plan["waterline_satisfied_after_selected_recode"] is True
    assert plan["selected_row"]["post_recode_over_waterline_bytes"] == 0
    assert (
        plan["verdict"]
        == "ADMIT_LF_RECODE__CROSSES_BYTE_WATERLINE__FALSE_AUTHORITY"
    )
    assert "snerv_lf_recode_admission_plan_false_authority" in plan["blockers"]
    assert "paired_contest_cpu_cuda_auth_eval_missing" in plan["blockers"]
    rows = {row["mode"]: row for row in plan["admission_rows"]}
    assert rows["zero_run_varint"]["local_planner_admitted"] is True
    assert rows["int16_brotli"]["post_recode_over_waterline_bytes"] == 12_000
    section_rows = {row["candidate_mode"]: row for row in plan["section_value_rows"]}
    assert section_rows["zero_run_varint"]["byte_delta"] == -40_000
    assert section_rows["zero_run_varint"]["delta_nonrate_score"] == 0.0
    assert section_rows["zero_run_varint"]["full_video_coverage"] is True
    byte_price_plan = plan["byte_price_plan"]
    assert byte_price_plan["schema"] == "compact_nerv_byte_price_controller.v1"
    assert byte_price_plan["source_schema"] == plan["schema"]
    assert byte_price_plan["score_claim"] is False
    markdown = render_snerv_lf_payload_recode_admission_markdown(plan)
    assert "zero_run_varint" in markdown
    assert "ADMIT_LF_RECODE__CROSSES_BYTE_WATERLINE" in markdown


def test_snerv_lf_payload_recode_admission_blocks_invalid_and_non_saving_rows() -> None:
    non_saving = {
        "schema": "snerv_lf_payload_archive_recode.v1",
        "mode": "raw_int64",
        "source_packet": {"bytes": 170_000, "sha256": "source-sha"},
        "candidate_packet": {"bytes": 170_128, "sha256": "candidate-sha"},
        "packet_byte_delta": 128,
        "lf_payload": {"source_bytes": 100_000, "candidate_bytes": 100_128},
        "receiver_contract_satisfied": True,
        "receiver_frame_equality_proof": {"status": "proven_exact"},
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    invalid = {
        "schema": "orphan_lf_codec_sweep.v0",
        "mode": "detached_proxy",
    }

    plan = build_snerv_lf_payload_recode_admission_plan(
        [non_saving, invalid],
        hard_byte_ceiling=178_000,
        candidate_id="snerv-full600",
    )

    assert plan["local_planner_admitted"] is False
    assert plan["selected_mode"] is None
    assert (
        plan["verdict"]
        == "NO_ADMISSIBLE_LF_RECODE__RERUN_RECEIVER_PROVEN_CODEC_SWEEP"
    )
    assert "snerv_lf_recode_no_receiver_proven_byte_saving_mode" in plan["blockers"]
    rows = {row["mode"]: row for row in plan["admission_rows"]}
    assert rows["raw_int64"]["local_planner_admitted"] is False
    assert "snerv_lf_recode_not_byte_saving" in rows["raw_int64"][
        "local_admission_blockers"
    ]
    assert rows["detached_proxy"]["local_planner_admitted"] is False
    assert "snerv_lf_recode_source_schema_invalid:orphan_lf_codec_sweep.v0" in rows[
        "detached_proxy"
    ]["local_admission_blockers"]
    assert plan["byte_price_plan"]["decision_counts"]["demote"] == 2
