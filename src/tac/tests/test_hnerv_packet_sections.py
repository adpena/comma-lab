# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.analysis.hnerv_packet_sections import (
    A2K1_MAGIC,
    CPLX1_MAGIC,
    MANIFEST_SCHEMA,
    PARSER_A2K1,
    PARSER_AUTO,
    PARSER_CPLX1,
    PARSER_PR101,
    PARSER_PR103,
    PARSER_PR106,
    PR101_DECODER_BLOB_LEN,
    PR101_LATENT_BLOB_LEN,
    HnervPacketSectionManifestError,
    build_packet_section_manifest,
    build_packet_section_manifest_batch,
    validate_packet_section_manifest,
    validate_packet_section_manifest_batch,
)
from tac.hnerv_pr103_lc_ac_schema import PUBLIC_PR103_LAYOUT
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106SidecarPacket,
    PR106_SIDECAR_FORMAT_BROTLI,
    emit_pr106_sidecar_packet,
)
from tac.repo_io import sha256_bytes

REPO = Path(__file__).resolve().parents[3]


def test_pr101_manifest_records_archive_member_and_fixed_sections(tmp_path: Path) -> None:
    payload = b"d" * PR101_DECODER_BLOB_LEN + b"l" * PR101_LATENT_BLOB_LEN + b"s" * 607
    archive = tmp_path / "pr101.zip"
    _stored_zip(archive, "x", payload)

    manifest = build_packet_section_manifest(archive, label="PR101", parser=PARSER_PR101)

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["archive"]["bytes"] == archive.stat().st_size
    assert manifest["member"]["name"] == "x"
    assert manifest["member"]["bytes"] == len(payload)
    assert manifest["parser"]["name"] == PARSER_PR101
    assert [section["name"] for section in manifest["sections"]] == [
        "decoder_compact_brotli_streams",
        "latents_raw_lzma_delta_u8",
        "sidecar_dim_delta_huffman_enum",
    ]
    assert manifest["sections"][1]["offset"] == PR101_DECODER_BLOB_LEN
    assert manifest["sections"][2]["length"] == 607
    assert validate_packet_section_manifest(manifest) == []


def test_a2k1_manifest_records_magic_len_and_pr101_sections(tmp_path: Path) -> None:
    decoder = b"a2-decoder"
    latent = b"l" * PR101_LATENT_BLOB_LEN
    sidecar = b"sidecar"
    payload = A2K1_MAGIC + len(decoder).to_bytes(4, "little") + decoder + latent + sidecar
    archive = tmp_path / "a2k1.zip"
    _stored_zip(archive, "x", payload)

    manifest = build_packet_section_manifest(archive, label="A2K1", parser=PARSER_A2K1)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_A2K1
    assert [section["name"] for section in manifest["sections"]] == [
        "a2k1_magic",
        "decoder_len_u32le",
        "decoder_blob",
        "latent_blob",
        "sidecar_blob",
    ]
    assert manifest["sections"][2]["offset"] == 8
    assert manifest["sections"][3]["offset"] == 8 + len(decoder)
    assert manifest["sections"][4]["length"] == len(sidecar)
    gate3 = manifest["parser_section_manifest"]
    assert gate3["section_names"] == [section["name"] for section in manifest["sections"]]
    assert gate3["offsets"] == [section["offset"] for section in manifest["sections"]]
    assert gate3["lengths"] == [section["length"] for section in manifest["sections"]]
    assert all(length > 0 for length in gate3["lengths"])
    assert validate_packet_section_manifest(manifest) == []


def test_cplx1_manifest_records_byte_map_json_and_op1_sections(tmp_path: Path) -> None:
    byte_maps = b'{"0":"zig"}'
    op1 = b"op1-inner"
    latent = b"l" * PR101_LATENT_BLOB_LEN
    sidecar = b"sidecar"
    section_total = 10 + len(byte_maps) + len(op1)
    payload = (
        CPLX1_MAGIC
        + section_total.to_bytes(4, "little")
        + len(byte_maps).to_bytes(2, "little")
        + byte_maps
        + op1
        + latent
        + sidecar
    )
    archive = tmp_path / "cplx1.zip"
    _stored_zip(archive, "x", payload)

    manifest = build_packet_section_manifest(archive, label="CPLX1", parser=PARSER_CPLX1)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_CPLX1
    assert [section["name"] for section in manifest["sections"]] == [
        "cplx_magic",
        "decoder_section_len_u32le",
        "byte_maps_json_len_u16le",
        "byte_maps_json",
        "op1_inner_blob",
        "latent_blob",
        "sidecar_blob",
    ]
    assert manifest["sections"][3]["offset"] == 10
    assert manifest["sections"][4]["offset"] == 10 + len(byte_maps)
    assert manifest["sections"][5]["offset"] == section_total
    gate3 = manifest["parser_section_manifest"]
    assert set(gate3) >= {
        "offsets",
        "lengths",
        "section_names",
        "section_sha256s",
        "entropy_estimates",
        "old_new_section_boundaries",
    }
    assert all(length > 0 for length in gate3["lengths"])
    assert validate_packet_section_manifest(manifest) == []


def test_a2k1_manifest_fails_closed_on_truncated_decoder_length(tmp_path: Path) -> None:
    payload = A2K1_MAGIC + (128).to_bytes(4, "little") + b"short"
    archive = tmp_path / "a2k1_truncated.zip"
    _stored_zip(archive, "x", payload)

    with pytest.raises(HnervPacketSectionManifestError, match="A2K1 payload too short"):
        build_packet_section_manifest(archive, label="A2K1", parser=PARSER_A2K1)


def test_cplx1_manifest_fails_closed_on_invalid_byte_maps_json(tmp_path: Path) -> None:
    byte_maps = b"{not-json"
    op1 = b"op1-inner"
    latent = b"l" * PR101_LATENT_BLOB_LEN
    sidecar = b"sidecar"
    section_total = 10 + len(byte_maps) + len(op1)
    payload = (
        CPLX1_MAGIC
        + section_total.to_bytes(4, "little")
        + len(byte_maps).to_bytes(2, "little")
        + byte_maps
        + op1
        + latent
        + sidecar
    )
    archive = tmp_path / "cplx1_bad_json.zip"
    _stored_zip(archive, "x", payload)

    with pytest.raises(HnervPacketSectionManifestError, match="byte_maps JSON is invalid"):
        build_packet_section_manifest(archive, label="CPLX1", parser=PARSER_CPLX1)


def test_pr106_manifest_records_len24_sections(tmp_path: Path) -> None:
    decoder = b"decoder-section"
    tail = b"latent-sidecar"
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + tail
    archive = tmp_path / "pr106.zip"
    _stored_zip(archive, "0.bin", payload)

    manifest = build_packet_section_manifest(archive, label="PR106", parser=PARSER_PR106)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_PR106
    assert [section["name"] for section in manifest["sections"]] == [
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    assert manifest["sections"][0]["length"] == 4
    assert manifest["sections"][1]["offset"] == 4
    assert manifest["sections"][2]["offset"] == 4 + len(decoder)
    assert manifest["coverage"]["covers_payload"] is True


def test_pr106_sidecar_wrapper_manifest_records_inner_sections_and_wrapper_custody(
    tmp_path: Path,
) -> None:
    decoder = b"decoder-section"
    tail = b"latent-sidecar"
    inner_payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + tail
    wrapper_payload = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=PR106_SIDECAR_FORMAT_BROTLI,
            pr106_bytes=inner_payload,
            sidecar_payload=b"opaque-sidecar",
            framing_meta=None,
        )
    )
    archive = tmp_path / "pr106_sidecar.zip"
    _stored_zip(archive, "0.bin", wrapper_payload)

    manifest = build_packet_section_manifest(
        archive,
        label="PR106 sidecar",
        parser=PARSER_AUTO,
    )

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_PR106
    assert manifest["member"]["bytes"] == len(wrapper_payload)
    assert manifest["member"]["sha256"] == sha256_bytes(wrapper_payload)
    assert manifest["parser_input"] == {
        "kind": "pr106_sidecar_inner_payload",
        "bytes": len(inner_payload),
        "sha256": sha256_bytes(inner_payload),
        "offset_base": "parser_input",
    }
    assert manifest["coverage"]["payload_bytes"] == len(inner_payload)
    assert manifest["coverage"]["covers_payload"] is True
    assert [section["name"] for section in manifest["sections"]] == [
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    wrapper = manifest["pr106_sidecar_wrapper"]
    assert wrapper["kind"] == "pr106_sidecar_wrapper"
    assert wrapper["format_id"] == "0x01"
    assert wrapper["sidecar_kind"] == "brotli_dim_delta"
    assert wrapper["outer_member_bytes"] == len(wrapper_payload)
    assert wrapper["inner_pr106_bytes"] == len(inner_payload)
    assert wrapper["sidecar_payload_bytes"] == len(b"opaque-sidecar")
    assert wrapper["score_claim"] is False
    assert validate_packet_section_manifest(manifest) == []


def test_pr103_manifest_uses_existing_lc_ac_parser(tmp_path: Path) -> None:
    payload = _pr103_payload()
    archive = tmp_path / "pr103.zip"
    _stored_zip(archive, "x", payload)

    manifest = build_packet_section_manifest(archive, label="PR103", parser=PARSER_PR103)

    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser"]["name"] == PARSER_PR103
    assert [section["name"] for section in manifest["sections"]][:3] == [
        "scales_fp16",
        "non_ac_weights_brotli",
        "ac_histograms_brotli",
    ]
    assert manifest["sections"][-1]["name"] == "sidecar_corrections_brotli"
    assert manifest["sections"][-1]["length"] == 19
    assert manifest["coverage"]["section_count"] == 8


def test_manifest_validation_rejects_tampered_section_sha(tmp_path: Path) -> None:
    decoder = b"d" * 7
    tail = b"t" * 5
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + tail
    archive = tmp_path / "pr106.zip"
    _stored_zip(archive, "0.bin", payload)
    manifest = build_packet_section_manifest(archive, label="PR106", parser=PARSER_PR106)
    manifest["sections"][1]["sha256"] = "0" * 64

    blockers = validate_packet_section_manifest(manifest)

    assert "sections_does_not_match_archive" in blockers


def test_batch_manifest_and_cli_emit_and_validate(tmp_path: Path) -> None:
    archive = tmp_path / "pr106.zip"
    decoder = b"abc"
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + b"tail"
    _stored_zip(archive, "0.bin", payload)
    out = tmp_path / "manifest.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_packet_section_manifest.py"),
            "--archive",
            f"fixture={archive}",
            "--parser",
            PARSER_PR106,
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)

    assert out.is_file()
    assert payload["score_claim"] is False
    assert payload["parser_section_gate"]["ready"] is True
    assert payload["records"][0]["parser"]["name"] == PARSER_PR106

    validate = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_packet_section_manifest.py"),
            "--validate-json",
            str(out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(validate.stdout)["ready"] is True

    direct = build_packet_section_manifest_batch([("fixture", archive, PARSER_PR106)])
    assert direct["parser_section_gate"]["ready"] is True


def test_batch_manifest_validation_rejects_false_authority_fields(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pr106.zip"
    decoder = b"abc"
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + b"tail"
    _stored_zip(archive, "0.bin", payload)
    batch = build_packet_section_manifest_batch([("fixture", archive, PARSER_PR106)])
    batch["ready_for_exact_eval_dispatch"] = True
    batch["parser_section_gate"]["score_claim"] = True
    batch["score_evidence_grade"] = "A++"
    batch["gpu_required"] = True

    blockers = validate_packet_section_manifest_batch(batch)

    assert "batch_ready_for_exact_eval_dispatch_not_false" in blockers
    assert "batch_parser_section_gate_score_claim_not_false" in blockers
    assert "batch_score_evidence_grade_not_invalid_no_score" in blockers
    assert "batch_gpu_required_not_false" in blockers


def test_single_manifest_validation_rejects_false_authority_fields(tmp_path: Path) -> None:
    archive = tmp_path / "pr106.zip"
    decoder = b"abc"
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + b"tail"
    _stored_zip(archive, "0.bin", payload)
    batch = build_packet_section_manifest_batch([("fixture", archive, PARSER_PR106)])
    manifest = batch["records"][0]
    manifest["score_evidence_grade"] = "A++"
    manifest["gpu_required"] = True

    blockers = validate_packet_section_manifest_batch(manifest)

    assert "score_evidence_grade_not_invalid_no_score" in blockers
    assert "gpu_required_not_false" in blockers


def _stored_zip(path: Path, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _pr103_payload() -> bytes:
    parts = []
    for index, (_name, length) in enumerate(PUBLIC_PR103_LAYOUT.section_specs()):
        parts.append(bytes([index + 1]) * length)
    parts.append(b"sidecar-corrections")
    return b"".join(parts)
