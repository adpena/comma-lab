# SPDX-License-Identifier: MIT
"""Tests for SNeRV binary/package attribution profiles."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from tac.analysis.snerv_binary_profile import (
    build_snerv_binary_profile,
    write_snerv_binary_profile,
)
from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_lf_quant_payload,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    inspect_lf_quant_payload_header,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder


def _snar1_packet(*, lf_codec: str = "int64_lzma") -> bytes:
    rng = np.random.default_rng(7)
    lf_planes = [
        rng.integers(-12, 13, size=(32, 32), dtype=np.int64)
        for _ in range(8)
    ]
    step_maps = [
        np.full((32, 32), 2.0 + idx * 0.01, dtype=np.float32)
        for idx in range(len(lf_planes))
    ]
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(
            lf_zero_points=np.linspace(0.0, 1.0, len(lf_planes), dtype=np.float32),
        ),
        lf_payload=encode_lf_quant_payload(lf_planes, codec=lf_codec),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=2)),
        step_map_packet=encode_step_maps(step_maps, bins=4).packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 4,
            "height": 64,
            "width": 64,
            "lf_plane_count": len(lf_planes),
            "levels": 2,
            "wavelet": "db2",
        },
    )
    return archive.packet


def test_snerv_lf_v2_spatial_delta_payload_decodes_and_profiles(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "candidate_v2.snar"
    packet_path.write_bytes(_snar1_packet(lf_codec="spatial_delta_zigzag_leb128_lzma"))

    profile = build_snerv_binary_profile(input_path=packet_path)
    header = profile["lf_payload_header"]

    assert header["schema"] == "snerv_lf_quant_payload.v2"
    assert header["codec"] == "spatial_delta_zigzag_leb128_lzma"
    assert profile["lf_quant_profile"]["decode_status"] == "decoded"
    assert profile["lf_quant_profile"]["plane_count"] == 8
    assert profile["lf_quant_profile"]["coeff_count"] == 8 * 32 * 32


def test_snerv_lf_v2_header_inspection_is_decode_free() -> None:
    payload = encode_lf_quant_payload(
        [np.arange(16, dtype=np.int64).reshape(4, 4)],
        codec="spatial_delta_zigzag_leb128_lzma",
    )

    header = inspect_lf_quant_payload_header(payload)
    planes = decode_lf_quant_payload(payload)

    assert header["schema"] == "snerv_lf_quant_payload.v2"
    assert header["shared_shape"] == [4, 4]
    assert header["shape_count"] == 1
    assert planes[0].shape == (4, 4)
    assert int(planes[0][3, 3]) == 15


def test_snerv_binary_profile_attributes_archive_sections(tmp_path: Path) -> None:
    packet = _snar1_packet()
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", packet)
        zf.writestr("inflate.sh", "#!/usr/bin/env bash\n")

    profile = build_snerv_binary_profile(input_path=archive_path, frontier_bytes=1)

    assert profile["schema"] == "snerv_binary_profile.v1"
    assert profile["input_kind"] == "contest_archive_zip"
    assert profile["package_profile"]["zip_member_count"] == 2
    assert profile["snar1_packet_bytes"] == len(packet)
    assert profile["charged_archive_bytes"] == archive_path.stat().st_size
    assert profile["charged_rate_score"] > 0.0
    assert profile["lf_quant_profile"]["plane_count"] == 8
    assert profile["lf_quant_profile"]["coeff_count"] == 8 * 32 * 32
    assert profile["lf_quant_profile"]["section_bytes"] > 0
    assert profile["lf_quant_profile"]["order0_entropy_floor_bytes"] > 0
    assert profile["section_summary"]["largest_section"] in {
        "lf_payload",
        "step_map_packet",
        "decoder_payload",
        "metadata_payload",
    }
    assert "snerv_binary_profile_is_rate_only_not_score_authority" in profile[
        "blockers"
    ]
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False


def test_write_snerv_binary_profile_supports_raw_snar1_packet(tmp_path: Path) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(_snar1_packet())
    out_path = tmp_path / "profile.json"

    profile = write_snerv_binary_profile(input_path=packet_path, output_path=out_path)

    assert out_path.is_file()
    assert profile["input_kind"] == "raw_snar1_packet"
    assert "not_packaged_as_contest_archive_zip" in profile["blockers"]
    assert profile["package_profile"]["zip_members"] == []
