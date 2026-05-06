from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from tac.archive_byte_profile import (
    ArchiveByteProfileError,
    build_profile_collection,
    profile_archive,
    render_markdown,
    write_outputs,
)


def test_profile_archive_reports_deterministic_member_and_group_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_member(archive, "decoder.bin", b"decoder", ZIP_STORED)
    _write_member(archive, "latents/0.bin", b"latent" * 3, ZIP_DEFLATED, append=True)

    profile = profile_archive(archive)

    assert profile["schema"] == "archive_byte_profile_v1"
    assert profile["score_claim"] is False
    assert profile["member_count"] == 2
    assert profile["archive_bytes"] == archive.stat().st_size
    assert profile["rate_term"] > 0
    assert [member["filename"] for member in profile["members"]] == [
        "decoder.bin",
        "latents/0.bin",
    ]
    assert profile["profile_by_suffix"][0]["key"] == ".bin"
    assert {row["key"] for row in profile["profile_by_top_level"]} == {"decoder.bin", "latents"}
    assert "Archive Byte Profile" in render_markdown(profile)


def test_profile_archive_rejects_zip_slip_and_duplicate_members(tmp_path: Path) -> None:
    zip_slip = tmp_path / "zipslip.zip"
    _write_member(zip_slip, "../bad", b"x", ZIP_STORED)
    with pytest.raises(ArchiveByteProfileError, match="zip-slip"):
        profile_archive(zip_slip)

    duplicate = tmp_path / "duplicate.zip"
    _write_member(duplicate, "x", b"1", ZIP_STORED)
    _write_member(duplicate, "x", b"2", ZIP_STORED, append=True)
    with pytest.raises(ArchiveByteProfileError, match="duplicate archive member"):
        profile_archive(duplicate)


def test_profile_collection_continue_on_error_and_write_outputs(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    bad = tmp_path / "bad.zip"
    json_out = tmp_path / "profile.json"
    md_out = tmp_path / "profile.md"
    _write_member(good, "x", b"payload", ZIP_STORED)
    bad.write_bytes(b"not zip")

    profile = build_profile_collection([good, bad], continue_on_error=True)
    write_outputs(profile, json_out=json_out, markdown_out=md_out)

    payload = json.loads(json_out.read_text())
    assert payload["schema"] == "archive_byte_profile_collection_v1"
    assert payload["archive_count"] == 2
    assert payload["archives"][0]["valid"] is True
    assert payload["archives"][1]["valid"] is False
    assert payload["archives"][1]["score_claim"] is False
    assert "Archive Byte Profile Collection" in md_out.read_text()


def _write_member(
    archive: Path,
    name: str,
    payload: bytes,
    method: int,
    *,
    append: bool = False,
) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = method
    info.external_attr = 0o644 << 16
    mode = "a" if append else "w"
    with ZipFile(archive, mode) as zf:
        zf.writestr(info, payload)
