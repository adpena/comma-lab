# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from tac.analysis.snerv_snar_header_minimizer import (
    SCHEMA,
    build_snerv_snar_header_minimization,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.inflate import (
    snerv_frames_to_raw_bytes,
)
from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (
    build_snerv_receiver_archive_proof,
)
from tools import minimize_snerv_snar_header as cli


def test_snerv_snar_header_minimizer_prunes_provenance_and_preserves_receiver_raw() -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(16, 24),
        full_frame_packet=True,
    )
    decoded = unpack_snerv_archive(archive.packet)
    rich_metadata = dict(decoded.metadata)
    rich_metadata.update(
        {
            "lf_step_allocation_rows": [
                {
                    "pair": index // 6,
                    "plane": index,
                    "human_readable_label": f"plane-{index}",
                    "debug_values": [index, index + 1, index + 2],
                }
                for index in range(96)
            ],
            "step_map_coder_groups": [
                {
                    "group_name": "debug-step-map-group",
                    "log2_values": list(range(128)),
                    "shapes": [[2, 3] for _ in range(32)],
                    "map_indices": list(range(32)),
                }
            ],
            "source_pair_indices_preserved": list(range(600)),
            "score_claim": False,
        }
    )
    source_packet = _repack_with_metadata(archive.packet, rich_metadata)

    report, candidate_packet = build_snerv_snar_header_minimization(
        source_packet,
        proof_pair_indices=(0,),
        hard_byte_ceilings=(len(source_packet) - 32,),
        generated_utc="2026-06-04T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["receiver_contract_satisfied"] is True
    assert report["sections_exact_equal"] is True
    assert report["hard_byte_ceiling_rows"][0]["header_bytes_removed"] > 0
    assert report["hard_byte_ceiling_rows"][0]["source_packet_over_ceiling_bytes"] == 32
    assert report["packet_byte_delta"] < 0
    assert report["header_byte_delta"] < 0
    assert report["removed_metadata"]["json_bytes"] > report["candidate_metadata"][
        "json_bytes"
    ]
    assert "lf_step_allocation_rows" in report["removed_metadata"]["top_level_keys"]
    assert "step_map_coder_groups" in report["removed_metadata"]["top_level_keys"]
    assert "source_pair_indices_preserved" in report["removed_metadata"][
        "top_level_keys"
    ]
    candidate = unpack_snerv_archive(candidate_packet)
    assert "lf_step_allocation_rows" not in candidate.metadata
    assert "step_map_coder_groups" not in candidate.metadata
    assert "source_pair_indices_preserved" not in candidate.metadata
    assert set(candidate.metadata).issubset(
        {
        "n_pairs",
        "frames_per_pair",
        "channels",
        "lf_plane_count",
        "levels",
        "wavelet",
        "carrier_hw",
        "orig_hw",
        }
    )
    assert "carrier_hw" in candidate.metadata or "orig_hw" in candidate.metadata
    source_frames = decode_snerv_archive_frames(source_packet)
    candidate_frames = decode_snerv_archive_frames(candidate_packet)
    np.testing.assert_array_equal(candidate_frames, source_frames)
    assert snerv_frames_to_raw_bytes(candidate_frames) == snerv_frames_to_raw_bytes(
        source_frames
    )
    for row in report["section_parity_rows"]:
        assert row["bytes_exact_equal"] is True
        assert row["sha256_exact_equal"] is True


def test_minimize_snerv_snar_header_cli_accepts_zip_and_writes_deterministic_archive(
    tmp_path: Path,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        hw=(12, 16),
        full_frame_packet=True,
    )
    decoded = unpack_snerv_archive(archive.packet)
    source_packet = _repack_with_metadata(
        archive.packet,
        {
            **decoded.metadata,
            "debug_manifest": [{"index": index, "sha256": "b" * 64} for index in range(16)],
        },
    )
    source_zip = tmp_path / "source.zip"
    _write_zip(source_zip, source_packet)
    output_packet = tmp_path / "candidate.snar"
    output_zip = tmp_path / "archive.zip"
    output_json = tmp_path / "manifest.json"

    assert cli.main(
        [
            "--packet",
            str(source_zip),
            "--output-packet",
            str(output_packet),
            "--output-archive-zip",
            str(output_zip),
            "--output-json",
            str(output_json),
            "--pair-index",
            "0",
            "--hard-byte-ceiling",
            str(len(source_packet) - 8),
        ]
    ) == 0

    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    candidate_packet = output_packet.read_bytes()
    with zipfile.ZipFile(output_zip) as archive_zip:
        info = archive_zip.getinfo("0.bin")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert archive_zip.read("0.bin") == candidate_packet
    assert manifest["candidate_archive_zip"]["member"] == "0.bin"
    assert manifest["contest_compliance_contract"]["archive_zip_materialized"] is True
    assert "not_packaged_as_contest_archive_zip" not in manifest["blockers"]
    assert manifest["source_packet"]["input_kind"] == "archive_zip_member_0_bin"
    assert manifest["hard_byte_ceiling_rows"][0]["candidate_archive_zip_bytes"] == (
        output_zip.stat().st_size
    )
    assert unpack_snerv_archive(candidate_packet).metadata["carrier_hw"] == list(
        decoded.metadata["carrier_hw"]
    )


def _repack_with_metadata(packet: bytes, metadata: dict) -> bytes:
    decoded = unpack_snerv_archive(packet)
    from tac.substrates.snerv_inverse_steg_carrier.archive import pack_snerv_archive

    return pack_snerv_archive(
        metadata_payload=decoded.sections["metadata_payload"],
        lf_payload=decoded.sections["lf_payload"],
        decoder_payload=decoded.sections["decoder_payload"],
        step_map_packet=decoded.sections["step_map_packet"],
        metadata=metadata,
    ).packet


def _write_zip(path: Path, packet: bytes) -> None:
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, packet)
