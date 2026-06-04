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
    SNAR2_HEADER_BYTES,
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
        candidate_id="snerv_unit_candidate",
        proof_pair_indices=(0,),
        full_video_receiver_proof=True,
        hard_byte_ceilings=(len(source_packet) - 32,),
        generated_utc="2026-06-04T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["candidate_binding"]["candidate_id"] == "snerv_unit_candidate"
    assert (
        report["candidate_binding"]["binding_status"]
        == "candidate_id_and_source_packet_sha256"
    )
    assert (
        "snerv_snar_header_minimization_candidate_id_binding_missing"
        not in report["blockers"]
    )
    assert report["receiver_contract_satisfied"] is True
    assert report["full_video_receiver_contract_satisfied"] is True
    assert report["receiver_pair_frame_equality_proof"]["scope"] == (
        "full_video_streaming"
    )
    assert report["receiver_pair_frame_equality_proof"]["pair_count"] == (
        decoded.metadata["n_pairs"]
    )
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
    _write_zip(
        source_zip,
        source_packet,
        extra_members=(
            ("inflate.py", b"print('receiver runtime')\n"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/archive.py", b"# runtime\n"),
        ),
    )
    output_packet = tmp_path / "candidate.snar"
    output_zip = tmp_path / "archive.zip"
    output_package = tmp_path / "runtime_package"
    output_json = tmp_path / "manifest.json"

    assert cli.main(
        [
            "--packet",
            str(source_zip),
            "--candidate-id",
            "snerv_cli_candidate",
            "--output-packet",
            str(output_packet),
            "--output-archive-zip",
            str(output_zip),
            "--output-package-dir",
            str(output_package),
            "--output-json",
            str(output_json),
            "--pair-index",
            "0",
            "--full-video-receiver-proof",
            "--hard-byte-ceiling",
            str(len(source_packet) - 8),
        ]
    ) == 0

    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    candidate_packet = output_packet.read_bytes()
    with zipfile.ZipFile(output_zip) as archive_zip:
        assert archive_zip.namelist() == [
            "0.bin",
            "inflate.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/archive.py",
        ]
        info = archive_zip.getinfo("0.bin")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert archive_zip.read("0.bin") == candidate_packet
        assert archive_zip.read("inflate.py") == b"print('receiver runtime')\n"
        assert (
            archive_zip.read("src/tac/substrates/snerv_inverse_steg_carrier/archive.py")
            == b"# runtime\n"
        )
    assert manifest["candidate_archive_zip"]["member"] == "0.bin"
    assert manifest["candidate_archive_zip"]["archive_zip_kind"] in {
        "runtime_preserving_repack",
        "generated_runtime_package",
    }
    assert manifest["candidate_archive_zip_pre_package_repack"]["archive_zip_kind"] == (
        "runtime_preserving_repack"
    )
    assert (
        manifest["candidate_archive_zip_pre_package_repack"]["path"]
        == output_zip.as_posix()
    )
    assert manifest["candidate_archive_zip_pre_package_repack"]["bytes"] == (
        output_zip.stat().st_size
    )
    assert manifest["contest_compliance_contract"]["archive_zip_materialized"] is True
    assert manifest["contest_compliance_contract"]["runtime_package_materialized"] is True
    assert (
        manifest["contest_compliance_contract"]["runtime_consumption_proof_passed"]
        is True
    )
    assert manifest["runtime_package"]["receiver_proof"][
        "runtime_consumption_proof_passed"
    ] is True
    assert (output_package / "archive.zip").is_file()
    assert (
        output_package
        / "receiver_proof"
        / "snerv_inverse_steg_receiver_proof.json"
    ).is_file()
    assert "not_packaged_as_contest_archive_zip" not in manifest["blockers"]
    assert (
        "snerv_snar_header_minimization_candidate_id_binding_missing"
        not in manifest["blockers"]
    )
    assert manifest["full_video_receiver_contract_satisfied"] is True
    assert manifest["receiver_pair_frame_equality_proof"]["scope"] == (
        "full_video_streaming"
    )
    assert manifest["source_packet"]["input_kind"] == "archive_zip_member_0_bin"
    assert manifest["hard_byte_ceiling_rows"][0]["candidate_archive_zip_bytes"] == (
        manifest["candidate_archive_zip"]["bytes"]
    )
    assert unpack_snerv_archive(candidate_packet).metadata["carrier_hw"] == list(
        decoded.metadata["carrier_hw"]
    )

    parity_output_packet = tmp_path / "candidate.parity.snar"
    parity_output_package = tmp_path / "runtime_package_parity"
    parity_output_json = tmp_path / "manifest.parity.json"
    source_receiver_proof = (
        output_package
        / "receiver_proof"
        / "snerv_inverse_steg_receiver_proof.json"
    )
    assert cli.main(
        [
            "--packet",
            str(output_package / "archive.zip"),
            "--candidate-id",
            "snerv_cli_candidate",
            "--output-packet",
            str(parity_output_packet),
            "--output-package-dir",
            str(parity_output_package),
            "--output-json",
            str(parity_output_json),
            "--full-video-receiver-proof",
            "--source-receiver-proof-json",
            str(source_receiver_proof),
        ]
    ) == 0
    parity_manifest = json.loads(parity_output_json.read_text(encoding="utf-8"))
    assert parity_manifest["source_receiver_output_parity"]["status"] == "proven_exact"
    assert parity_manifest["source_receiver_output_parity"]["exact_equal"] is True
    assert (
        parity_manifest["source_receiver_output_parity"]["source_proof_schema_valid"]
        is True
    )
    assert (
        parity_manifest["source_receiver_output_parity"]["candidate_proof_schema_valid"]
        is True
    )
    assert (
        parity_manifest["source_receiver_output_parity"]["source_archive_identity"][
            "exact_equal"
        ]
        is True
    )
    assert (
        parity_manifest["source_receiver_output_parity"]["candidate_archive_identity"][
            "exact_equal"
        ]
        is True
    )
    assert parity_manifest["source_receiver_output_parity"]["source_proof_file_sha256"]
    assert (
        parity_manifest["contest_compliance_contract"][
            "source_receiver_output_parity_proven"
        ]
        is True
    )
    assert parity_manifest["full_video_receiver_contract_satisfied"] is True
    assert parity_manifest["runtime_consumption_proof_ready"] is True
    assert (
        "snerv_snar_header_minimization_receiver_proof_failed"
        not in parity_manifest["blockers"]
    )
    assert parity_manifest["score_claim"] is False
    assert parity_manifest["ready_for_exact_eval_dispatch"] is False

    mismatched_proof = json.loads(source_receiver_proof.read_text(encoding="utf-8"))
    mismatched_proof["archive_sha256"] = "0" * 64
    mismatched_source_proof = tmp_path / "mismatched_source_receiver_proof.json"
    mismatched_source_proof.write_text(
        json.dumps(mismatched_proof, sort_keys=True),
        encoding="utf-8",
    )
    mismatch_output_packet = tmp_path / "candidate.mismatch.snar"
    mismatch_output_package = tmp_path / "runtime_package_mismatch"
    mismatch_output_json = tmp_path / "manifest.mismatch.json"
    assert cli.main(
        [
            "--packet",
            str(output_package / "archive.zip"),
            "--candidate-id",
            "snerv_cli_candidate",
            "--output-packet",
            str(mismatch_output_packet),
            "--output-package-dir",
            str(mismatch_output_package),
            "--output-json",
            str(mismatch_output_json),
            "--full-video-receiver-proof",
            "--source-receiver-proof-json",
            str(mismatched_source_proof),
        ]
    ) == 0
    mismatch_manifest = json.loads(mismatch_output_json.read_text(encoding="utf-8"))
    assert mismatch_manifest["source_receiver_output_parity"]["status"] == "failed"
    assert (
        "snerv_source_receiver_proof_archive_identity_mismatch"
        in mismatch_manifest["source_receiver_output_parity"]["blockers"]
    )
    assert (
        "snerv_snar_header_minimization_full_video_receiver_proof_missing"
        not in mismatch_manifest["blockers"]
    )
    assert (
        "snerv_source_receiver_proof_archive_identity_mismatch"
        in mismatch_manifest["blockers"]
    )


def test_minimize_snerv_snar_header_cli_emits_snar2_binary_header_and_runtime_package(
    tmp_path: Path,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        wavelet="haar",
        hw=(12, 16),
        full_frame_packet=True,
    )
    decoded = unpack_snerv_archive(archive.packet)
    source_packet = _repack_with_metadata(
        archive.packet,
        {
            **decoded.metadata,
            "debug_manifest": [
                {"index": index, "sha256": "b" * 64} for index in range(16)
            ],
        },
    )
    source_path = tmp_path / "source.snar"
    source_path.write_bytes(source_packet)
    output_packet = tmp_path / "candidate.snar2"
    output_zip = tmp_path / "archive.zip"
    output_package = tmp_path / "runtime_package"
    output_json = tmp_path / "manifest.json"

    assert cli.main(
        [
            "--packet",
            str(source_path),
            "--candidate-id",
            "snerv_snar2_unit_candidate",
            "--wire-format",
            "snar2",
            "--output-packet",
            str(output_packet),
            "--output-archive-zip",
            str(output_zip),
            "--output-package-dir",
            str(output_package),
            "--output-json",
            str(output_json),
            "--full-video-receiver-proof",
            "--hard-byte-ceiling",
            str(len(source_packet) - 8),
        ]
    ) == 0

    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    candidate_packet = output_packet.read_bytes()
    assert candidate_packet.startswith(b"SNAR2")
    assert manifest["operation"] == "snar2_fixed_binary_header_receiver_metadata_prune"
    assert manifest["wire_format"] == "snar2"
    header = candidate_packet[: manifest["candidate_packet"]["header_bytes"]]
    for forbidden in (
        b"metadata_payload",
        b"lf_payload",
        b"decoder_payload",
        b"step_map_packet",
        b"debug_manifest",
        b"wavelet",
        b"schema",
    ):
        assert forbidden not in header
    assert manifest["candidate_packet"]["header_bytes"] == SNAR2_HEADER_BYTES
    assert manifest["header_byte_delta"] < 0
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["full_video_receiver_contract_satisfied"] is True
    assert manifest["runtime_package"]["receiver_proof"][
        "runtime_consumption_proof_passed"
    ] is True
    assert "snar2_no_human_readable_label_bitstream_not_implemented" not in manifest[
        "blockers"
    ]
    assert (
        "snerv_snar_header_minimization_receiver_proof_failed"
        not in manifest["blockers"]
    )
    assert (
        "snerv_snar_header_minimization_candidate_id_binding_missing"
        not in manifest["blockers"]
    )
    with zipfile.ZipFile(output_zip) as archive_zip:
        assert archive_zip.read("0.bin") == candidate_packet
    np.testing.assert_array_equal(
        decode_snerv_archive_frames(candidate_packet),
        decode_snerv_archive_frames(source_packet),
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


def _write_zip(
    path: Path,
    packet: bytes,
    *,
    extra_members: tuple[tuple[str, bytes], ...] = (),
) -> None:
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, packet)
        for name, payload in extra_members:
            extra_info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            extra_info.compress_type = zipfile.ZIP_STORED
            archive.writestr(extra_info, payload)
