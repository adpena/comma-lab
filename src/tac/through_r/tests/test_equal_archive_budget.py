# SPDX-License-Identifier: MIT
from __future__ import annotations

import stat
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import tac.through_r.equal_archive_budget as equal_budget
from tac.through_r.equal_archive_budget import (
    FIXED_ZIP_TIMESTAMP,
    PADDING_MEMBER,
    ArchiveBudgetError,
    equalize_archive_budgets,
    file_sha256,
    verify_equal_archive_budget_receipt,
    verify_original_members_preserved,
    verify_output_tree_preserved,
    verify_rate_match_member,
    zip_member_hashes,
)


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 2, 3, 4, 4))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)


def test_equal_size_deterministic_hashes_and_original_member_bytes_unchanged(tmp_path: Path):
    left_source = tmp_path / "left_source.zip"
    right_source = tmp_path / "right_source.zip"
    left_members = {"inflate.sh": b"#!/bin/sh\nexit 0\n", "model.bin": bytes(range(64))}
    right_members = {"inflate.sh": b"#!/bin/sh\nexit 0\n", "model.bin": b"abc" * 1700}
    _write_archive(left_source, left_members)
    _write_archive(right_source, right_members)

    left_1, right_1 = tmp_path / "left_1.zip", tmp_path / "right_1.zip"
    receipt_1 = equalize_archive_budgets(left_source, right_source, left_1, right_1)
    assert receipt_1.equal_archive_bytes
    assert left_1.stat().st_size == right_1.stat().st_size == receipt_1.target_archive_bytes
    assert receipt_1.left.matched_archive_bytes == receipt_1.right.matched_archive_bytes
    assert receipt_1.left.padding_bytes != receipt_1.right.padding_bytes
    assert verify_equal_archive_budget_receipt(left_1, right_1, receipt_1)
    with pytest.raises(ArchiveBudgetError, match="canonical hash mismatch"):
        verify_equal_archive_budget_receipt(
            left_1,
            right_1,
            replace(receipt_1, target_archive_bytes=receipt_1.target_archive_bytes + 1),
        )
    assert verify_original_members_preserved(left_source, left_1) == {
        name: zip_member_hashes(left_source)[name] for name in left_members
    }
    assert verify_original_members_preserved(right_source, right_1) == {
        name: zip_member_hashes(right_source)[name] for name in right_members
    }
    with zipfile.ZipFile(left_1) as archive:
        assert archive.read("inflate.sh") == left_members["inflate.sh"]
        assert archive.read("model.bin") == left_members["model.bin"]
        info = archive.getinfo(PADDING_MEMBER)
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == FIXED_ZIP_TIMESTAMP

    # Repeating from the same exact source archives produces identical output bytes/hashes.
    left_2, right_2 = tmp_path / "left_2.zip", tmp_path / "right_2.zip"
    receipt_2 = equalize_archive_budgets(left_source, right_source, left_2, right_2)
    assert file_sha256(left_1) == file_sha256(left_2)
    assert file_sha256(right_1) == file_sha256(right_2)
    assert receipt_1.left.matched_archive_sha256 == receipt_2.left.matched_archive_sha256
    assert receipt_1.right.matched_archive_sha256 == receipt_2.right.matched_archive_sha256
    assert receipt_1.receipt_sha256 == receipt_2.receipt_sha256


def test_equal_sources_still_receive_one_empty_fixed_padding_member(tmp_path: Path):
    source = tmp_path / "source.zip"
    twin = tmp_path / "twin.zip"
    _write_archive(source, {"x": b"payload"})
    twin.write_bytes(source.read_bytes())
    left, right = tmp_path / "left.zip", tmp_path / "right.zip"
    receipt = equalize_archive_budgets(source, twin, left, right)
    assert receipt.left.padding_bytes == receipt.right.padding_bytes == 0
    assert verify_rate_match_member(left) == verify_rate_match_member(right) == 0
    assert left.read_bytes() == right.read_bytes()


def test_fail_closed_on_preexisting_or_duplicate_padding(tmp_path: Path):
    clean = tmp_path / "clean.zip"
    padded = tmp_path / "padded.zip"
    duplicate = tmp_path / "duplicate.zip"
    _write_archive(clean, {"x": b"x"})
    _write_archive(padded, {PADDING_MEMBER: b"not allowed"})
    with pytest.raises(ArchiveBudgetError, match="already contains"):
        equalize_archive_budgets(
            padded, clean, tmp_path / "a.zip", tmp_path / "b.zip"
        )
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr(PADDING_MEMBER, b"")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(PADDING_MEMBER, b"")
    with pytest.raises(ArchiveBudgetError, match="duplicate"):
        verify_rate_match_member(duplicate)
    padding_alias = tmp_path / "padding_alias.zip"
    with zipfile.ZipFile(padding_alias, "w") as archive:
        archive.writestr(f"{PADDING_MEMBER}/", b"")
    with pytest.raises(ArchiveBudgetError, match="already contains"):
        equalize_archive_budgets(
            padding_alias, clean, tmp_path / "alias_a.zip", tmp_path / "alias_b.zip"
        )


def test_fail_closed_on_malformed_zip_and_output_source_alias(tmp_path: Path):
    malformed = tmp_path / "bad.zip"
    malformed.write_bytes(b"PK\x03\x04truncated")
    source = tmp_path / "source.zip"
    other = tmp_path / "other.zip"
    _write_archive(source, {"x": b"x"})
    _write_archive(other, {"y": b"y"})
    with pytest.raises(ArchiveBudgetError, match="well-formed"):
        equalize_archive_budgets(
            malformed, other, tmp_path / "a.zip", tmp_path / "b.zip"
        )
    with pytest.raises(ArchiveBudgetError, match="never overwrites"):
        equalize_archive_budgets(source, other, source, tmp_path / "b.zip")


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "/absolute.bin",
        "../traversal.bin",
        "a/../traversal.bin",
        "a/./alias.bin",
        "a//alias.bin",
        "back\\slash.bin",
        "C:/drive.bin",
    ],
)
def test_rejects_unsafe_member_paths_before_any_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_name: str
):
    unsafe = tmp_path / "unsafe.zip"
    clean = tmp_path / "clean.zip"
    _write_archive(unsafe, {unsafe_name: b"payload"})
    _write_archive(clean, {"safe.bin": b"safe"})
    copy_calls = 0
    real_copyfile = equal_budget.shutil.copyfile

    def counted_copyfile(source, destination):
        nonlocal copy_calls
        copy_calls += 1
        return real_copyfile(source, destination)

    monkeypatch.setattr(equal_budget.shutil, "copyfile", counted_copyfile)
    with pytest.raises(ArchiveBudgetError, match=r"path|backslash|drive"):
        equalize_archive_budgets(
            unsafe, clean, tmp_path / "left.zip", tmp_path / "right.zip"
        )
    assert copy_calls == 0
    assert not (tmp_path / "left.zip").exists()
    assert not (tmp_path / "right.zip").exists()


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFIFO])
def test_rejects_symlink_and_special_entries_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_type: int
):
    unsafe = tmp_path / "unsafe_type.zip"
    clean = tmp_path / "clean.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        info = zipfile.ZipInfo("unsafe-entry")
        info.create_system = 3
        info.external_attr = (file_type | 0o777) << 16
        archive.writestr(info, b"target-or-special")
    _write_archive(clean, {"safe.bin": b"safe"})

    def forbidden_copy(*_args, **_kwargs):
        raise AssertionError("validation must happen before copy")

    monkeypatch.setattr(equal_budget.shutil, "copyfile", forbidden_copy)
    with pytest.raises(ArchiveBudgetError, match="symlink or special"):
        equalize_archive_budgets(
            unsafe, clean, tmp_path / "left.zip", tmp_path / "right.zip"
        )


def test_rejects_file_directory_and_parent_path_aliases(tmp_path: Path):
    clean = tmp_path / "clean.zip"
    _write_archive(clean, {"safe.bin": b"safe"})
    for index, names in enumerate((("node", "node/"), ("node", "node/child.bin"))):
        unsafe = tmp_path / f"alias_{index}.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            for name in names:
                archive.writestr(name, b"" if name.endswith("/") else b"payload")
        with pytest.raises(ArchiveBudgetError, match="alias"):
            equalize_archive_budgets(
                unsafe,
                clean,
                tmp_path / f"left_{index}.zip",
                tmp_path / f"right_{index}.zip",
            )


def test_validates_both_sources_before_first_snapshot_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clean = tmp_path / "clean.zip"
    unsafe_right = tmp_path / "unsafe_right.zip"
    _write_archive(clean, {"safe.bin": b"safe"})
    _write_archive(unsafe_right, {"../escape.bin": b"unsafe"})

    def forbidden_copy(*_args, **_kwargs):
        raise AssertionError("neither source may be copied before both validate")

    monkeypatch.setattr(equal_budget.shutil, "copyfile", forbidden_copy)
    with pytest.raises(ArchiveBudgetError, match="traversing"):
        equalize_archive_budgets(
            clean, unsafe_right, tmp_path / "left.zip", tmp_path / "right.zip"
        )


def test_pair_publication_failure_restores_both_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    left_source, right_source = tmp_path / "ls.zip", tmp_path / "rs.zip"
    _write_archive(left_source, {"x": b"left"})
    _write_archive(right_source, {"x": b"right" * 200})
    left_output, right_output = tmp_path / "left.zip", tmp_path / "right.zip"
    old_left, old_right = b"old-left-output", b"old-right-output"
    left_output.write_bytes(old_left)
    right_output.write_bytes(old_right)
    real_replace = equal_budget._PAIR_PUBLISH_REPLACE
    calls = 0

    def fail_second_publication(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(equal_budget, "_PAIR_PUBLISH_REPLACE", fail_second_publication)
    with pytest.raises(OSError, match="injected"):
        equalize_archive_budgets(
            left_source, right_source, left_output, right_output
        )
    assert left_output.read_bytes() == old_left
    assert right_output.read_bytes() == old_right
    assert not list(tmp_path.glob(".rate_match_*.zip"))


def test_pair_publication_failure_leaves_no_half_pair_when_outputs_were_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    left_source, right_source = tmp_path / "ls.zip", tmp_path / "rs.zip"
    _write_archive(left_source, {"x": b"left"})
    _write_archive(right_source, {"x": b"right" * 200})
    left_output, right_output = tmp_path / "left.zip", tmp_path / "right.zip"
    real_replace = equal_budget._PAIR_PUBLISH_REPLACE
    calls = 0

    def fail_second_publication(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected absent-pair failure")
        return real_replace(source, destination)

    monkeypatch.setattr(equal_budget, "_PAIR_PUBLISH_REPLACE", fail_second_publication)
    with pytest.raises(OSError, match="injected absent-pair"):
        equalize_archive_budgets(
            left_source, right_source, left_output, right_output
        )
    assert not left_output.exists()
    assert not right_output.exists()


def test_source_mutation_after_snapshot_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    left_source, right_source = tmp_path / "ls.zip", tmp_path / "rs.zip"
    _write_archive(left_source, {"x": b"left"})
    _write_archive(right_source, {"x": b"right" * 200})
    left_output, right_output = tmp_path / "left.zip", tmp_path / "right.zip"
    left_output.write_bytes(b"old-left")
    right_output.write_bytes(b"old-right")
    real_append = equal_budget._append_padding_copy
    calls = 0

    def mutate_after_snapshot(source, destination, padding_bytes):
        nonlocal calls
        real_append(source, destination, padding_bytes)
        calls += 1
        if calls == 1:
            left_source.write_bytes(left_source.read_bytes() + b"mutated")

    monkeypatch.setattr(equal_budget, "_append_padding_copy", mutate_after_snapshot)
    with pytest.raises(ArchiveBudgetError, match="changed after immutable snapshot"):
        equalize_archive_budgets(
            left_source, right_source, left_output, right_output
        )
    assert left_output.read_bytes() == b"old-left"
    assert right_output.read_bytes() == b"old-right"


def test_source_mutation_during_snapshot_copy_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    left_source, right_source = tmp_path / "ls.zip", tmp_path / "rs.zip"
    _write_archive(left_source, {"x": b"left"})
    _write_archive(right_source, {"x": b"right"})
    real_copyfile = equal_budget.shutil.copyfile
    mutated = False

    def mutate_during_copy(source, destination):
        nonlocal mutated
        result = real_copyfile(source, destination)
        if Path(source) == left_source and not mutated:
            left_source.write_bytes(left_source.read_bytes() + b"raced")
            mutated = True
        return result

    monkeypatch.setattr(equal_budget.shutil, "copyfile", mutate_during_copy)
    with pytest.raises(ArchiveBudgetError, match="changed while"):
        equalize_archive_budgets(
            left_source,
            right_source,
            tmp_path / "left.zip",
            tmp_path / "right.zip",
        )
    assert not (tmp_path / "left.zip").exists()
    assert not (tmp_path / "right.zip").exists()


def test_post_publication_destination_reverification_failure_rolls_back_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    left_source, right_source = tmp_path / "ls.zip", tmp_path / "rs.zip"
    _write_archive(left_source, {"x": b"left"})
    _write_archive(right_source, {"x": b"right" * 200})
    left_output, right_output = tmp_path / "left.zip", tmp_path / "right.zip"
    left_output.write_bytes(b"old-left")
    right_output.write_bytes(b"old-right")
    real_verify = equal_budget._verify_published_pair

    def mutate_then_verify(left, right, left_receipt, right_receipt, target_bytes):
        left.write_bytes(left.read_bytes() + b"post-publication-mutation")
        return real_verify(left, right, left_receipt, right_receipt, target_bytes)

    monkeypatch.setattr(equal_budget, "_verify_published_pair", mutate_then_verify)
    with pytest.raises(ArchiveBudgetError, match=r"size differs|hash differs"):
        equalize_archive_budgets(
            left_source, right_source, left_output, right_output
        )
    assert left_output.read_bytes() == b"old-left"
    assert right_output.read_bytes() == b"old-right"


def test_padding_mutation_and_original_member_mutation_are_detected(tmp_path: Path):
    left_source, right_source = tmp_path / "ls.zip", tmp_path / "rs.zip"
    _write_archive(left_source, {"x": b"left"})
    _write_archive(right_source, {"x": b"right" * 200})
    left, right = tmp_path / "left.zip", tmp_path / "right.zip"
    equalize_archive_budgets(left_source, right_source, left, right)

    bad_padding = tmp_path / "bad_padding.zip"
    with zipfile.ZipFile(bad_padding, "w") as archive:
        info = zipfile.ZipInfo(PADDING_MEMBER, FIXED_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, b"X")
    with pytest.raises(ArchiveBudgetError, match="all zero"):
        verify_rate_match_member(bad_padding)

    bad_member = tmp_path / "bad_member.zip"
    with zipfile.ZipFile(bad_member, "w") as archive:
        archive.writestr("x", b"changed")
        info = zipfile.ZipInfo(PADDING_MEMBER, FIXED_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, b"")
    with pytest.raises(ArchiveBudgetError, match="changed"):
        verify_original_members_preserved(left_source, bad_member)


def test_inflated_output_tree_preservation_helper(tmp_path: Path):
    before, after = tmp_path / "before", tmp_path / "after"
    (before / "frames").mkdir(parents=True)
    (after / "frames").mkdir(parents=True)
    (before / "frames" / "000.png").write_bytes(b"same frame bytes")
    (after / "frames" / "000.png").write_bytes(b"same frame bytes")
    assert verify_output_tree_preserved(before, after)
    (after / "frames" / "000.png").write_bytes(b"different")
    with pytest.raises(ArchiveBudgetError, match="changed"):
        verify_output_tree_preserved(before, after)
