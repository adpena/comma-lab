# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from tac.zipwire_archive import (
    inspect_zip_headers,
    inspect_zip_headers_python,
    inspect_zip_headers_rust,
    resolve_zipwire_binary,
)


def test_python_fallback_reports_header_only_core_fields(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_member(archive, "x", b"payload-bytes", ZIP_STORED)
    raw = archive.read_bytes()

    record = inspect_zip_headers(archive, prefer_rust=False)

    assert record["path"] == str(archive)
    assert record["bytes"] == len(raw)
    assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["zip_strict"] is True
    assert record["blockers"] == []
    assert record["member_count"] == 1
    member = record["members"][0]
    assert member["name"] == "x"
    assert member["local_header_name"] == "x"
    assert member["local_central_name_match"] is True
    assert member["header_offset"] == 0
    assert member["payload_offset"] == 31
    assert member["compress_type"] == ZIP_STORED
    assert member["compressed_bytes"] == 13
    assert member["uncompressed_bytes"] == 13
    assert member["local_header"]["compressed_bytes"] == 13


def test_python_fallback_reports_duplicate_and_local_central_mismatch(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.zip"
    _write_member(duplicate, "x", b"1", ZIP_STORED)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_member(duplicate, "x", b"2", ZIP_STORED, append=True)

    duplicate_record = inspect_zip_headers_python(duplicate)

    assert duplicate_record["zip_strict"] is False
    assert duplicate_record["duplicate_member_names"] == ["x"]
    assert "duplicate_archive_member:x" in duplicate_record["blockers"]

    mismatch = tmp_path / "mismatch.zip"
    _write_member(mismatch, "x", b"payload-bytes", ZIP_STORED)
    raw = bytearray(mismatch.read_bytes())
    raw[30] = ord("y")
    mismatch.write_bytes(raw)

    mismatch_record = inspect_zip_headers_python(mismatch)

    assert mismatch_record["zip_strict"] is False
    assert mismatch_record["members"][0]["name"] == "x"
    assert mismatch_record["members"][0]["local_header_name"] == "y"
    assert mismatch_record["members"][0]["blockers"] == ["local_central_name_mismatch"]
    assert mismatch_record["blockers"] == ["x:local_central_name_mismatch"]


def test_python_fallback_blocks_data_descriptor_flag(tmp_path: Path) -> None:
    archive = tmp_path / "descriptor.zip"
    _write_member(archive, "x", b"payload-bytes", ZIP_STORED)
    raw = bytearray(archive.read_bytes())
    raw[6] = raw[6] | 0x08
    raw[52] = raw[52] | 0x08
    archive.write_bytes(raw)

    record = inspect_zip_headers_python(archive)

    assert record["zip_strict"] is False
    assert record["members"][0]["blockers"] == ["data_descriptor_member_not_supported"]
    assert record["blockers"] == ["x:data_descriptor_member_not_supported"]


@pytest.mark.parametrize("name", ["bad\nname", "C:evil"])
def test_python_fallback_uses_submission_archive_strict_name_rules(
    tmp_path: Path,
    name: str,
) -> None:
    archive = tmp_path / "unsafe_name.zip"
    _write_member(archive, name, b"payload-bytes", ZIP_STORED)

    record = inspect_zip_headers_python(archive)

    assert record["zip_strict"] is False
    assert record["members"][0]["name"] == name
    assert any("unsafe_member_name:" in blocker for blocker in record["blockers"])


def test_rust_bridge_matches_python_core_fields_when_binary_is_present(
    tmp_path: Path,
) -> None:
    binary = resolve_zipwire_binary()
    if binary is None:
        pytest.skip("zipwire binary has not been built")

    archive = tmp_path / "archive.zip"
    _write_member(archive, "x", b"payload-bytes", ZIP_STORED)
    _write_member(archive, "nested/y", b"abcdef" * 4, ZIP_DEFLATED, append=True)

    rust_record = inspect_zip_headers_rust(archive, binary=binary)
    python_record = inspect_zip_headers_python(archive)

    assert _core_record(rust_record) == _core_record(python_record)


def test_explicit_missing_rust_binary_falls_back_to_python(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_member(archive, "x", b"payload-bytes", ZIP_STORED)

    record = inspect_zip_headers(
        archive,
        zipwire_bin=tmp_path / "missing-zipwire",
    )

    assert record["zip_strict"] is True
    assert record["members"][0]["name"] == "x"


def _core_record(record: dict) -> dict:
    return {
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "member_count": record["member_count"],
        "duplicate_member_names": record["duplicate_member_names"],
        "blockers": record["blockers"],
        "zip_strict": record["zip_strict"],
        "members": [
            {
                "name": member["name"],
                "local_header_name": member["local_header_name"],
                "local_central_name_match": member["local_central_name_match"],
                "header_offset": member["header_offset"],
                "payload_offset": member["payload_offset"],
                "compress_type": member["compress_type"],
                "compressed_bytes": member["compressed_bytes"],
                "uncompressed_bytes": member["uncompressed_bytes"],
                "crc32": member["crc32"],
                "flag_bits": member["flag_bits"],
                "blockers": member["blockers"],
                "local_header": member["local_header"],
            }
            for member in record["members"]
        ],
    }


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
