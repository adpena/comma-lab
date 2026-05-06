from __future__ import annotations

from pathlib import Path

import pytest

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
    """Round 8 R8-1 + Round 9 R9-1 fix (2026-05-06): a broken symlink
    causes `Path.resolve()` to raise FileNotFoundError on most platforms
    (and OSError on some). Pre-R8-1, _cached_read_text/_cached_read_bytes
    would let that exception propagate from the resolve step rather
    than from the actual read attempt. Post-R8-1, a broken symlink
    falls through to the original method, which raises a meaningful
    read-failure error.

    R9-1 hardening: the previous R8-1 test used `try/except: pass`
    which passed unconditionally — a tautology that proved nothing.
    R9-1 uses `pytest.raises` so the test FAILS if no exception is
    raised, AND asserts the broken-symlink key is NOT cached (the
    intent of R8-1's fall-through is to avoid caching a failed
    resolution).
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
        # The read MUST raise — either the original raises FileNotFoundError
        # from the read attempt, or the resolve guard falls through and the
        # original raises. What's NOT acceptable is silent success or a
        # confusing pre-resolve error from inside _cached_read_text.
        with pytest.raises((FileNotFoundError, OSError)):
            link.read_text()
        # The broken-symlink path must NOT be cached: caching a failed
        # resolution would mean a future call returns stale state. R8-1
        # specifically falls through BEFORE the cache write to prevent
        # exactly this poisoning.
        stats = cache_stats()
        # No read_text entry should reference the broken-link path.
        # We can't introspect keys directly without exposing a fixture
        # API, but read_text_entries == 0 means nothing was cached.
        assert stats["read_text_entries"] == 0, (
            "broken-symlink read must not poison the read_text cache; "
            f"got stats={stats}"
        )


def test_cached_read_bytes_passes_through_broken_symlink_to_original(tmp_path: Path) -> None:
    """Round 8 R8-1 + Round 9 R9-1 sister test for read_bytes."""
    target = tmp_path / "missing.bin"
    link = tmp_path / "src" / "broken_link.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return

    with cached_filesystem(cache_reads=True):
        with pytest.raises((FileNotFoundError, OSError)):
            link.read_bytes()
        stats = cache_stats()
        assert stats["read_bytes_entries"] == 0, (
            "broken-symlink read must not poison the read_bytes cache; "
            f"got stats={stats}"
        )
