# SPDX-License-Identifier: MIT
"""Tests for SNeRV scalable-layer admission accounting."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from tac.analysis.nerv_modelsize_budget import RATE_SCORE_PER_BYTE
from tac.analysis.snerv_scalable_layer_admission import (
    BASE_LAYER_ID,
    HF_LAYER_ID,
    STEP_LAYER_ID,
    build_snerv_scalable_layer_admission_report,
    render_snerv_scalable_layer_admission_markdown,
    write_snerv_scalable_layer_admission_report,
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
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder


def test_scalable_layer_admission_prices_real_snar_sections_without_overclaim(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(_snar_packet())

    report = build_snerv_scalable_layer_admission_report(input_path=packet_path)

    assert report["schema"] == "snerv_scalable_layer_admission.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["deserves_separate_scalable_layer_lane"] is False
    assert report["verdict"] == (
        "KEEP_AS_SNERV_BITSTREAM_POLICY__SECTION_VALUE_EVIDENCE_MISSING"
    )
    rows = {row["layer_id"]: row for row in report["layer_rows"]}
    assert rows[BASE_LAYER_ID]["optional_layer"] is False
    assert rows[BASE_LAYER_ID]["sections"] == ["metadata_payload", "lf_payload"]
    assert rows[BASE_LAYER_ID]["layer_bytes"] > 0
    assert rows[BASE_LAYER_ID]["byte_price_score"] == (
        rows[BASE_LAYER_ID]["layer_bytes"] * RATE_SCORE_PER_BYTE
    )
    assert rows[HF_LAYER_ID]["admission_decision"] == (
        "needs_scorer_section_value_profile"
    )
    assert rows[STEP_LAYER_ID]["admission_decision"] == (
        "needs_scorer_section_value_profile"
    )
    assert len(report["section_value_rows"]) == 2
    assert report["byte_price_plan"]["schema"] == NERV_BYTE_PRICE_CONTROLLER_SCHEMA
    assert report["byte_price_plan"]["input_row_count"] == 2
    for decision in report["byte_price_plan"]["decision_rows"]:
        assert decision["decision"] == DEMOTE
        assert "delta_nonrate_score_missing" in decision["blockers"]
    assert "snerv_scalable_layer_section_value_profile_missing" in report["blockers"]


def test_scalable_layer_admission_uses_scorer_deltas_for_optional_layers(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "archive.zip"
    packet = _snar_packet()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", packet)
    first = build_snerv_scalable_layer_admission_report(input_path=archive_path)
    rows = {row["layer_id"]: row for row in first["layer_rows"]}
    hf_price = float(rows[HF_LAYER_ID]["byte_price_score"])
    step_price = float(rows[STEP_LAYER_ID]["byte_price_score"])

    report = build_snerv_scalable_layer_admission_report(
        input_path=archive_path,
        full_video_coverage=True,
        layer_nonrate_deltas={
            HF_LAYER_ID: hf_price + 0.01,
            STEP_LAYER_ID: max(0.0, step_price - 1e-9),
        },
    )

    rows = {row["layer_id"]: row for row in report["layer_rows"]}
    assert rows[HF_LAYER_ID]["admission_decision"] == (
        "admit_layer_bytes_are_scorer_justified"
    )
    assert rows[STEP_LAYER_ID]["admission_decision"] == (
        "cut_or_receiver_generate_layer_candidate"
    )
    assert rows[STEP_LAYER_ID]["net_score_saved_if_removed"] > 0.0
    section_rows = {row["section_id"]: row for row in report["section_value_rows"]}
    assert section_rows[HF_LAYER_ID]["byte_delta"] == -rows[HF_LAYER_ID]["layer_bytes"]
    assert section_rows[STEP_LAYER_ID]["byte_delta"] == -rows[STEP_LAYER_ID]["layer_bytes"]
    assert section_rows[HF_LAYER_ID]["delta_nonrate_score"] == rows[HF_LAYER_ID][
        "measured_nonrate_score_increase_if_removed"
    ]
    assert "snerv_scalable_layer_cut_receiver_variant_not_materialized" in (
        section_rows[HF_LAYER_ID]["blockers"]
    )
    decisions = {
        row["section_id"]: row for row in report["byte_price_plan"]["decision_rows"]
    }
    assert decisions[HF_LAYER_ID]["economic_decision"] == PROTECT
    assert decisions[STEP_LAYER_ID]["economic_decision"] == CUT
    assert decisions[HF_LAYER_ID]["decision"] == DEMOTE
    assert decisions[STEP_LAYER_ID]["decision"] == DEMOTE
    assert report["section_value_profile_attached"] is True
    assert report["deserves_separate_scalable_layer_lane"] is True
    assert report["verdict"] == (
        "SPLIT_SCALABLE_LAYER_LANE_CANDIDATE__BYTE_PRICED_AND_SCORER_USEFUL"
    )


def test_scalable_layer_admission_writer_and_markdown(tmp_path: Path) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(_snar_packet())
    out_path = tmp_path / "admission.json"

    report = write_snerv_scalable_layer_admission_report(
        input_path=packet_path,
        output_path=out_path,
    )
    markdown = render_snerv_scalable_layer_admission_markdown(report)

    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == report["schema"]
    assert "SNeRV Scalable-Layer Admission" in markdown


def _snar_packet() -> bytes:
    rng = np.random.default_rng(12)
    lf_planes = [
        rng.integers(-7, 8, size=(8, 8), dtype=np.int64)
        for _ in range(4)
    ]
    step_maps = [
        np.full((8, 8), 1.0 + idx * 0.125, dtype=np.float32)
        for idx in range(len(lf_planes))
    ]
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(
            lf_zero_points=np.linspace(0.0, 1.0, len(lf_planes), dtype=np.float32),
        ),
        lf_payload=encode_lf_quant_payload(lf_planes),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=2)),
        step_map_packet=encode_step_maps(step_maps, bins=4).packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 2,
            "height": 16,
            "width": 16,
            "lf_plane_count": len(lf_planes),
            "levels": 2,
            "wavelet": "db2",
        },
    )
    return archive.packet
