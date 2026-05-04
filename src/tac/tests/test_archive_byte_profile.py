from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tac.archive_byte_profile import (
    ArchiveByteProfileError,
    build_profile_collection,
    contest_rate_term,
    profile_archive,
    render_markdown,
    write_outputs,
)


def _write_zip(path: Path, members: list[tuple[str, bytes]], *, duplicate_names: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for index, (name, data) in enumerate(members):
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED if index % 2 == 0 else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
        if duplicate_names:
            info = zipfile.ZipInfo(members[0][0])
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, b"same-name-different-payload")


def test_profile_member_ordering_is_deterministic(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(
        archive,
        [
            ("z.bin", b"z" * 5),
            ("nested/b.txt", b"b" * 7),
            ("a.bin", b"a" * 3),
        ],
    )

    profile = profile_archive(archive)

    assert [member["name"] for member in profile["members"]] == [
        "a.bin",
        "nested/b.txt",
        "z.bin",
    ]
    assert [row["name"] for row in profile["top_contributors"]] == [
        "nested/b.txt",
        "z.bin",
        "a.bin",
    ]


def test_rate_term_uses_contest_formula(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(archive, [("p", b"payload" * 11)])

    profile = profile_archive(archive)

    assert profile["score_claim"] is False
    assert profile["evidence_grade"] == "byte_profile_only"
    assert profile["rate_term"] == round(contest_rate_term(archive.stat().st_size), 12)
    assert profile["members"][0]["rate_term"] == round(
        contest_rate_term(profile["members"][0]["compressed_size"]), 12
    )


@pytest.mark.parametrize("bad_name", ["../escape.bin", "/abs.bin", "safe/../escape.bin", "C:/abs.bin", "a\\b.bin"])
def test_profile_rejects_zip_slip_member_names(tmp_path: Path, bad_name: str) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(archive, [(bad_name, b"payload")])

    with pytest.raises(ArchiveByteProfileError, match="archive member|zip-slip|backslashes"):
        profile_archive(archive)


def test_duplicate_payload_and_member_name_detection(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    shared = b"shared-payload" * 9
    _write_zip(
        archive,
        [
            ("a.bin", shared),
            ("b.bin", shared),
        ],
        duplicate_names=True,
    )

    profile = profile_archive(archive)
    duplicates = profile["duplicate_detection"]

    assert duplicates["has_duplicate_payload_hashes"] is True
    assert duplicates["duplicate_payload_hashes"][0]["count"] == 2
    assert [member["name"] for member in duplicates["duplicate_payload_hashes"][0]["members"]] == [
        "a.bin",
        "b.bin",
    ]
    assert duplicates["has_duplicate_member_names"] is True
    assert duplicates["duplicate_member_names"] == [{"name": "a.bin", "count": 2}]


def test_extension_and_path_group_totals(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(
        archive,
        [
            ("renderer.bin", b"r" * 10),
            ("nested/masks.mkv", b"m" * 20),
            ("nested/repair.amr1.xz", b"x" * 5),
            ("README", b"n" * 3),
        ],
    )

    profile = profile_archive(archive)
    extension_totals = {row["name"]: row for row in profile["extension_totals"]}
    path_totals = {row["name"]: row for row in profile["path_group_totals"]}

    assert extension_totals[".bin"]["uncompressed_size"] == 10
    assert extension_totals[".mkv"]["uncompressed_size"] == 20
    assert extension_totals[".amr1.xz"]["uncompressed_size"] == 5
    assert extension_totals["(no_extension)"]["uncompressed_size"] == 3
    assert path_totals["nested"]["member_count"] == 2
    assert path_totals["(root)"]["member_count"] == 2


def test_collection_writes_json_and_markdown(tmp_path: Path) -> None:
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    json_out = tmp_path / "profile.json"
    markdown_out = tmp_path / "profile.md"
    _write_zip(archive_a, [("a.bin", b"same"), ("b.txt", b"other")])
    _write_zip(archive_b, [("c.bin", b"same")])

    profile = build_profile_collection([archive_a, archive_b])
    write_outputs(profile, json_out=json_out, markdown_out=markdown_out)

    loaded = json.loads(json_out.read_text())
    markdown = markdown_out.read_text()
    assert loaded["schema"] == "archive_byte_profile_collection_v1"
    assert loaded["score_claim"] is False
    assert loaded["evidence_grade"] == "byte_profile_only"
    assert loaded["cross_archive_duplicate_payload_hashes"][0]["archive_count"] == 2
    assert "# Archive Byte Profile" in markdown
    assert "This is byte attribution only" in render_markdown(profile)


def test_collection_can_record_invalid_archive(tmp_path: Path) -> None:
    valid = tmp_path / "valid.zip"
    invalid = tmp_path / "invalid.zip"
    _write_zip(valid, [("p", b"payload")])
    invalid.write_bytes(b"not a zip archive")

    with pytest.raises(ArchiveByteProfileError):
        build_profile_collection([valid, invalid])

    profile = build_profile_collection([valid, invalid], continue_on_error=True)
    markdown = render_markdown(profile)

    assert profile["archive_count"] == 2
    assert profile["invalid_archive_count"] == 1
    assert profile["invalid_archives"][0]["archive_path"] == str(invalid)
    assert profile["invalid_archives"][0]["valid_profile"] is False
    assert "Invalid Archives" in markdown
