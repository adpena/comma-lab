# SPDX-License-Identifier: MIT
"""Tests for receiver-proof SNeRV LF payload archive recoding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.analysis.snerv_lf_payload_archive_recode import (
    build_snerv_lf_payload_archive_recode,
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
