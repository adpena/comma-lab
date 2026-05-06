from __future__ import annotations

from pathlib import Path

from tac.preflight_fs_cache import cache_stats, cached_filesystem


def test_cached_filesystem_caches_rglob_and_restores(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    first = root / "a.py"
    first.write_text("print('a')\n", encoding="utf-8")
    original_rglob = Path.rglob

    with cached_filesystem():
        assert [path.name for path in root.rglob("*.py")] == ["a.py"]
        second = root / "b.py"
        second.write_text("print('b')\n", encoding="utf-8")
        assert [path.name for path in root.rglob("*.py")] == ["a.py"]
        stats = cache_stats()
        assert stats["rglob_entries"] == 1
        assert stats["rglob_files_total"] == 1

    assert Path.rglob is original_rglob
    assert cache_stats()["rglob_entries"] == 0
    assert sorted(path.name for path in root.rglob("*.py")) == ["a.py", "b.py"]


def test_cached_filesystem_does_not_cache_reads_by_default(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text("first\n", encoding="utf-8")

    with cached_filesystem():
        assert path.read_text(encoding="utf-8") == "first\n"
        path.write_text("second\n", encoding="utf-8")
        assert path.read_text(encoding="utf-8") == "second\n"


def test_cached_filesystem_can_cache_reads_when_explicit(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    path = source_root / "sample.py"
    path.write_text("first\n", encoding="utf-8")

    with cached_filesystem(cache_reads=True):
        assert path.read_text() == "first\n"
        path.write_text("second\n", encoding="utf-8")
        assert path.read_text() == "first\n"


def test_cached_filesystem_read_cache_skips_non_source_paths(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text("first\n", encoding="utf-8")

    with cached_filesystem(cache_reads=True):
        assert path.read_text() == "first\n"
        path.write_text("second\n", encoding="utf-8")
        assert path.read_text() == "second\n"
