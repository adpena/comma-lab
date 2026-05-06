from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from tac.archive_byte_profile import (
    ArchiveByteProfileError,
    build_profile_collection,
    build_candidate_diff_manifest,
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
    with pytest.warns(UserWarning, match="Duplicate name"):
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


def test_candidate_diff_manifest_classifies_archive_noop(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    _write_member(source, "x", b"payload", ZIP_STORED)
    candidate.write_bytes(source.read_bytes())

    manifest = build_candidate_diff_manifest(
        source_archive=source,
        candidate_archive=candidate,
        source_label="PR106x",
        candidate_label="noop-copy",
    )

    assert manifest["schema"] == "archive_candidate_diff_manifest_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_sha256_equal"] is True
    assert manifest["payload_sha256_multiset_equal"] is True
    assert manifest["candidate_non_noop"] is False
    assert manifest["no_op_status"] == "byte_identical_archive_noop"
    assert "candidate_is_noop" in manifest["dispatch_blockers"]
    assert "Archive Candidate Diff Manifest" in render_markdown(manifest)


def test_candidate_diff_manifest_classifies_container_reemit_noop(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    _write_member(source, "0.bin", b"same payload", ZIP_STORED)
    _write_member(candidate, "x", b"same payload", ZIP_DEFLATED)

    manifest = build_candidate_diff_manifest(source_archive=source, candidate_archive=candidate)

    assert manifest["archive_sha256_equal"] is False
    assert manifest["member_name_sets_identical"] is False
    assert manifest["payload_sha256_multiset_equal"] is True
    assert manifest["candidate_non_noop"] is False
    assert manifest["no_op_status"] == "payload_identical_container_reemit_noop"
    assert "candidate_is_noop" in manifest["dispatch_blockers"]


def test_candidate_diff_manifest_classifies_payload_change(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    _write_member(source, "x", b"source payload", ZIP_STORED)
    _write_member(candidate, "x", b"candidate payload", ZIP_STORED)

    manifest = build_candidate_diff_manifest(source_archive=source, candidate_archive=candidate)

    assert manifest["archive_sha256_equal"] is False
    assert manifest["payload_sha256_multiset_equal"] is False
    assert manifest["candidate_non_noop"] is True
    assert manifest["no_op_status"] == "non_noop_payload_changed"
    assert "candidate_is_noop" not in manifest["dispatch_blockers"]
    assert manifest["changed_members"][0]["status"] == "payload_changed"


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
