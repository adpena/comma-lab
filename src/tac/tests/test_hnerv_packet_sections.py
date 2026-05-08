from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.analysis.hnerv_packet_sections import (
    MANIFEST_SCHEMA,
    PARSER_PR101,
    PARSER_PR103,
    PARSER_PR106,
    PR101_DECODER_BLOB_LEN,
    PR101_LATENT_BLOB_LEN,
    build_packet_section_manifest,
    build_packet_section_manifest_batch,
    validate_packet_section_manifest,
)
from tac.hnerv_pr103_lc_ac_schema import PUBLIC_PR103_LAYOUT

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
