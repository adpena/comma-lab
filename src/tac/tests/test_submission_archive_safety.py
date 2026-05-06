from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from tac.submission_archive import safe_extract_zip, validate_archive_member_name


def _zip(path: Path, members: list[tuple[str | zipfile.ZipInfo, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return path


def test_safe_extract_zip_allows_valid_members(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "ok.zip", [("renderer.bin", b"abcd"), ("nested/masks.mkv", b"m")])
    out = tmp_path / "out"

    names = safe_extract_zip(archive, out)

    assert names == ["renderer.bin", "nested/masks.mkv"]
    assert (out / "renderer.bin").read_bytes() == b"abcd"
    assert (out / "nested" / "masks.mkv").read_bytes() == b"m"


@pytest.mark.parametrize("name", ["../escape", "/abs", "__MACOSX/x", ".DS_Store", "ok/.hidden"])
def test_safe_extract_zip_rejects_unsafe_member_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        validate_archive_member_name(name)

    archive = _zip(tmp_path / "bad.zip", [(name, b"x")])

    with pytest.raises(ValueError):
        safe_extract_zip(archive, tmp_path / "out")


def test_validate_archive_member_name_rejects_nul_byte() -> None:
    with pytest.raises(ValueError, match="NUL"):
        validate_archive_member_name("bad\x00name")


def test_safe_extract_zip_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "dup.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("renderer.bin", b"a")
        zf.writestr("renderer.bin", b"b")

    with pytest.raises(ValueError, match="duplicate archive member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_zip_rejects_symlink_members(tmp_path: Path) -> None:
    info = zipfile.ZipInfo("renderer.bin")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _zip(tmp_path / "symlink.zip", [(info, b"target")])

    with pytest.raises(ValueError, match="symlink"):
        safe_extract_zip(archive, tmp_path / "out")
