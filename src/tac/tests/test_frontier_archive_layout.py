from __future__ import annotations

import json
import zipfile
from pathlib import Path

import brotli

from tac.frontier_archive_layout import (
    A2K1_MAGIC,
    CPLX1_MAGIC,
    PR101_DECODER_BLOB_LEN,
    PR101_INNER_MEMBER_NAME,
    PR101_LATENT_BLOB_LEN,
    PR106_HEADER_MAGIC,
    PR106_INNER_MEMBER_NAME,
    inspect_frontier_archive_layout,
    render_frontier_archive_layout_summary,
)


def _stored_zip(path: Path, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_pr101_single_member_has_parser_proven_logical_sections(tmp_path: Path) -> None:
    decoder = b"d" * PR101_DECODER_BLOB_LEN
    latent = b"l" * PR101_LATENT_BLOB_LEN
    sidecar = b"s" * 607
    archive = tmp_path / "pr101.zip"
    _stored_zip(archive, PR101_INNER_MEMBER_NAME, decoder + latent + sidecar)

    manifest = inspect_frontier_archive_layout(archive)

    assert manifest["score_claim"] is False
    assert manifest["physical_layout"]["single_member_monolithic_packet"] is True
    assert manifest["physical_layout"]["archive_member_level_component_budgets_valid"] is False
    assert manifest["physical_layout"]["member_level_mask_budget_valid"] is False
    assert manifest["logical_layout"]["grammar"] == "pr101_fixed_offset_hnerv_microcodec"
    assert [section["name"] for section in manifest["logical_layout"]["sections"]] == [
        "decoder_blob",
        "latent_blob",
        "sidecar_blob",
    ]
    assert manifest["logical_layout"]["sections"][0]["len"] == PR101_DECODER_BLOB_LEN
    assert "do not infer mask/pose budgets" in " ".join(manifest["cautions"])


def test_a2k1_single_member_has_magic_and_variable_decoder_sections(tmp_path: Path) -> None:
    decoder = b"decoder-a2"
    latent = b"l" * PR101_LATENT_BLOB_LEN
    sidecar = b"sidecar"
    payload = A2K1_MAGIC + len(decoder).to_bytes(4, "little") + decoder + latent + sidecar
    archive = tmp_path / "a2k1.zip"
    _stored_zip(archive, PR101_INNER_MEMBER_NAME, payload)

    manifest = inspect_frontier_archive_layout(archive)

    logical = manifest["logical_layout"]
    assert logical["grammar"] == "a2k1_variable_decoder_pr101"
    assert logical["wire_format"] == "A2K1"
    assert logical["magic"] == "A2K1"
    assert logical["decoder_len_field"] == len(decoder)
    assert [section["name"] for section in logical["sections"]] == [
        "a2k1_magic",
        "decoder_len_u32le",
        "decoder_blob",
        "latent_blob",
        "sidecar_blob",
    ]
    assert logical["sections"][2]["offset"] == 8
    assert logical["sections"][3]["offset"] == 8 + len(decoder)
    assert logical["sections"][4]["len"] == len(sidecar)


def test_cplx1_single_member_has_byte_map_and_op1_sections(tmp_path: Path) -> None:
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
    _stored_zip(archive, PR101_INNER_MEMBER_NAME, payload)

    manifest = inspect_frontier_archive_layout(archive)

    logical = manifest["logical_layout"]
    assert logical["grammar"] == "cplx1_op1_byte_maps"
    assert logical["wire_format"] == "CPLX1"
    assert logical["magic"] == "CPLX"
    assert logical["decoder_section_total_field"] == section_total
    assert logical["byte_maps_json_len_field"] == len(byte_maps)
    assert [section["name"] for section in logical["sections"]] == [
        "cplx_magic",
        "decoder_section_len_u32le",
        "byte_maps_json_len_u16le",
        "byte_maps_json",
        "op1_inner_blob",
        "latent_blob",
        "sidecar_blob",
    ]
    assert logical["sections"][3]["offset"] == 10
    assert logical["sections"][4]["offset"] == 10 + len(byte_maps)
    assert logical["sections"][5]["offset"] == section_total
    assert logical["sections"][6]["len"] == len(sidecar)


def test_pr106_single_member_has_ff_parser_sections(tmp_path: Path) -> None:
    decoder = brotli.compress(b"decoder-weights")
    tail = brotli.compress(b"latent-sidecar")
    header = bytes([PR106_HEADER_MAGIC]) + len(decoder).to_bytes(3, "little")
    archive = tmp_path / "pr106.zip"
    _stored_zip(archive, PR106_INNER_MEMBER_NAME, header + decoder + tail)

    manifest = inspect_frontier_archive_layout(archive)

    assert manifest["physical_layout"]["single_member_monolithic_packet"] is True
    assert manifest["physical_layout"]["archive_member_level_component_budgets_valid"] is False
    assert manifest["logical_layout"]["grammar"] == "pr106_ff_packed_hnerv"
    assert manifest["logical_layout"]["decoder_len_field"] == len(decoder)
    assert [section["name"] for section in manifest["logical_layout"]["sections"]] == [
        "ff_header",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    summary = render_frontier_archive_layout_summary(manifest)
    assert "member_level_component_budgets_valid: False" in summary
    assert "logical grammar: pr106_ff_packed_hnerv" in summary


def test_pr106_like_invalid_brotli_streams_fail_closed(tmp_path: Path) -> None:
    decoder = b"not-brotli"
    tail = b"also-not-brotli"
    header = bytes([PR106_HEADER_MAGIC]) + len(decoder).to_bytes(3, "little")
    archive = tmp_path / "invalid_pr106.zip"
    _stored_zip(archive, PR106_INNER_MEMBER_NAME, header + decoder + tail)

    manifest = inspect_frontier_archive_layout(archive)

    assert manifest["logical_layout"] is None
    assert any("PR106-like" in caution and "ambiguous" in caution for caution in manifest["cautions"])


def test_unknown_single_member_does_not_create_logical_budget(tmp_path: Path) -> None:
    archive = tmp_path / "unknown.zip"
    _stored_zip(archive, "payload.bin", b"x" * 1024)

    manifest = inspect_frontier_archive_layout(archive)

    assert json.dumps(manifest, sort_keys=True)
    assert manifest["physical_layout"]["single_member_monolithic_packet"] is True
    assert manifest["physical_layout"]["archive_member_level_component_budgets_valid"] is False
    assert manifest["logical_layout"] is None
    assert any("No known internal grammar was proven" in caution for caution in manifest["cautions"])


def test_truncated_a2k1_magic_fails_closed_without_logical_layout(tmp_path: Path) -> None:
    payload = A2K1_MAGIC + (128).to_bytes(4, "little") + b"short"
    archive = tmp_path / "truncated_a2k1.zip"
    _stored_zip(archive, PR101_INNER_MEMBER_NAME, payload)

    manifest = inspect_frontier_archive_layout(archive)

    assert manifest["logical_layout"] is None
    assert any("No known internal grammar was proven" in caution for caution in manifest["cautions"])
