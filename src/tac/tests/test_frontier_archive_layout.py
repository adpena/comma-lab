from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tac.frontier_archive_layout import (
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


def test_pr106_single_member_has_ff_parser_sections(tmp_path: Path) -> None:
    decoder = b"b" * 17
    tail = b"t" * 11
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


def test_unknown_single_member_does_not_create_logical_budget(tmp_path: Path) -> None:
    archive = tmp_path / "unknown.zip"
    _stored_zip(archive, "payload.bin", b"x" * 1024)

    manifest = inspect_frontier_archive_layout(archive)

    assert json.dumps(manifest, sort_keys=True)
    assert manifest["physical_layout"]["single_member_monolithic_packet"] is True
    assert manifest["physical_layout"]["archive_member_level_component_budgets_valid"] is False
    assert manifest["logical_layout"] is None
    assert any("No known internal grammar was proven" in caution for caution in manifest["cautions"])
