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


def test_cached_read_text_passes_through_broken_symlink_to_original(tmp_path: Path) -> None:
    """Round 8 R8-1 fix (2026-05-06, 85%): a broken symlink causes
    `Path.resolve()` to raise FileNotFoundError on most platforms (and
    OSError on some). Pre-R8-1, _cached_read_text/_cached_read_bytes
    would let that exception propagate from the resolve step rather
    than from the actual read attempt — confusing the operator's
    diagnostic. Post-R8-1, a broken symlink falls through to the
    original method, which raises a meaningful read-failure error.
    Pin this contract.
    """
    target = tmp_path / "missing.py"  # never created
    link = tmp_path / "src" / "broken_link.py"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        # Symlinks not available on this platform — skip silently.
        return

    with cached_filesystem(cache_reads=True):
        # Either the original raises FileNotFoundError from the read OR
        # the resolve guard falls through and the original raises. Both
        # are acceptable; what's NOT acceptable is a confusing pre-resolve
        # error message bubbling up from inside _cached_read_text.
        try:
            link.read_text()
        except (FileNotFoundError, OSError):
            pass  # expected error class; the message points at the read


def test_cached_read_bytes_passes_through_broken_symlink_to_original(tmp_path: Path) -> None:
    """Round 8 R8-1 sister test for read_bytes."""
    target = tmp_path / "missing.bin"
    link = tmp_path / "src" / "broken_link.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return

    with cached_filesystem(cache_reads=True):
        try:
            link.read_bytes()
        except (FileNotFoundError, OSError):
            pass
