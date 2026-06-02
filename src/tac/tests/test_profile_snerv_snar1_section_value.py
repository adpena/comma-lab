# SPDX-License-Identifier: MIT
"""Tests for the SNeRV SNAR1 section-value profiler tool."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.repo_io import ArtifactWriteError
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tools.profile_snerv_snar1_section_value import (
    PROFILE_SCHEMA,
    build_snerv_snar1_section_value_profile,
)
from tools.profile_snerv_snar1_section_value import (
    main as profile_main,
)


def _packet() -> bytes:
    step_maps = [
        np.exp2(
            np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3) + idx * 0.1
        )
        for idx in range(6)
    ]
    decoder = HfGenerationDecoder.zeros(levels=1)
    decoder.kernels[0]["LH"][:] = 0.125
    decoder.kernels[0]["HL"][:] = -0.25
    decoder.kernels[0]["HH"][:] = 0.5
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0] * 6),
        lf_payload=encode_lf_quant_payload(
            [
                (np.arange(9, dtype=np.int64).reshape(3, 3) + idx)
                for idx in range(6)
            ],
            codec="portfolio_auto",
        ),
        decoder_payload=encode_decoder_payload(decoder),
        step_map_packet=encode_step_maps(step_maps, bins=8).packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "carrier_hw": [6, 6],
            "orig_hw": [6, 6],
            "lf_plane_count": 6,
            "levels": 1,
            "wavelet": "haar",
        },
    )
    return archive.packet


def test_snerv_snar1_section_value_profile_writes_receiver_decodable_variants(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(_packet())

    report = build_snerv_snar1_section_value_profile(
        input_path=packet_path,
        variant_output_dir=tmp_path / "variants",
        requested_sections=("decoder_payload", "step_map_packet"),
        raw_argv=["unit"],
    )

    assert report["schema"] == PROFILE_SCHEMA
    assert report["score_claim"] is False
    assert report["variant_count"] == 3
    assert len(report["section_value_rows"]) == 2
    assert "runtime_consumption_proof_missing_for_neutralized_packets" in report[
        "blockers"
    ]
    for row in report["section_value_rows"]:
        assert row["receiver_proof_status"] == "receiver_decode_only"
        assert row["runtime_consumption_proof_passed"] is False
        variant_path = Path(row["variant_packet_path"])
        assert variant_path.is_file()
        assert decode_snerv_archive_frames(variant_path.read_bytes()).shape == (
            1,
            2,
            3,
            6,
            6,
        )


def test_snerv_snar1_section_value_cli_accepts_archive_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("0.bin", _packet())
    output_json = tmp_path / "profile.json"

    assert (
        profile_main(
            [
                str(archive_path),
                "--output-json",
                str(output_json),
                "--variant-output-dir",
                str(tmp_path / "cli_variants"),
                "--sections",
                "decoder_payload",
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["input_kind"] == "archive_zip_member_0_bin"
    assert payload["variant_count"] == 2
    assert payload["section_value_rows"][0]["section_id"] == "snerv_decoder_payload"


def test_snerv_snar1_section_value_profile_reuses_only_identical_variants(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(_packet())
    variant_dir = tmp_path / "variants"

    first = build_snerv_snar1_section_value_profile(
        input_path=packet_path,
        variant_output_dir=variant_dir,
        requested_sections=("decoder_payload",),
    )
    second = build_snerv_snar1_section_value_profile(
        input_path=packet_path,
        variant_output_dir=variant_dir,
        requested_sections=("decoder_payload",),
        expected_variant_tree_present=True,
    )

    assert second["variants"][0]["packet_sha256"] == first["variants"][0]["packet_sha256"]
    (variant_dir / "baseline.snar").write_bytes(b"stale-different-packet")
    with pytest.raises(ArtifactWriteError, match="refusing to replace non-identical"):
        build_snerv_snar1_section_value_profile(
            input_path=packet_path,
            variant_output_dir=variant_dir,
            requested_sections=("decoder_payload",),
            expected_variant_tree_present=True,
        )


def test_snerv_snar1_section_value_profile_rejects_required_sections(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(_packet())

    with pytest.raises(ValueError, match="unsupported SNeRV section"):
        build_snerv_snar1_section_value_profile(
            input_path=packet_path,
            variant_output_dir=tmp_path / "variants",
            requested_sections=("lf_payload",),
        )
