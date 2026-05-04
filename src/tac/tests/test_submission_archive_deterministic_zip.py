from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import pytest

from tac.submission_archive import (
    RENDERER_COMPACT_MANIFEST,
    build_submission_archive,
    deterministic_zip_directory,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_submission_archive_is_stable_when_source_mtimes_change(tmp_path: Path) -> None:
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"r" * 12000)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"m" * 12000)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"p" * 7200)

    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"

    build_submission_archive(
        archive_a,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    for i, path in enumerate((renderer, masks, poses), start=1):
        os.utime(path, (1_900_000_000 + i, 1_900_000_000 + i))

    build_submission_archive(
        archive_b,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    assert _sha256(archive_a) == _sha256(archive_b)
    with zipfile.ZipFile(archive_a) as zf:
        assert [info.date_time for info in zf.infolist()] == [
            (1980, 1, 1, 0, 0, 0),
            (1980, 1, 1, 0, 0, 0),
            (1980, 1, 1, 0, 0, 0),
        ]
        assert [info.external_attr >> 16 for info in zf.infolist()] == [
            0o644,
            0o644,
            0o644,
        ]


def test_deterministic_zip_directory_sorts_members_and_ignores_source_mtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "b.bin").write_bytes(b"b" * 17)
    (source / "a.bin").write_bytes(b"a" * 13)

    archive_a = tmp_path / "dir_a.zip"
    archive_b = tmp_path / "dir_b.zip"
    members_a = deterministic_zip_directory(source, archive_a)

    os.utime(source / "a.bin", (1_900_000_100, 1_900_000_100))
    os.utime(source / "nested" / "b.bin", (1_900_000_200, 1_900_000_200))
    members_b = deterministic_zip_directory(source, archive_b)

    assert members_a == ["a.bin", "nested/b.bin"]
    assert members_b == ["a.bin", "nested/b.bin"]
    assert _sha256(archive_a) == _sha256(archive_b)


@pytest.mark.parametrize(
    "relpath",
    [
        ".DS_Store",
        "._renderer.bin",
        "__MACOSX/renderer.bin",
        "nested/.hidden",
    ],
)
def test_deterministic_zip_directory_flags_hidden_system_junk(
    tmp_path: Path,
    relpath: str,
) -> None:
    source = tmp_path / "source"
    junk = source / relpath
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"junk")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        deterministic_zip_directory(source, tmp_path / "archive.zip")
