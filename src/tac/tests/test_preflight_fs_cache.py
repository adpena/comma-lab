from __future__ import annotations

import contextlib
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
    """Round 8 R8-1 + Round 9 R9-1 + Round 10 R10-1/R10-2 fix (2026-05-06):
    a broken symlink causes `Path.resolve()` to raise FileNotFoundError on
    most platforms. Pre-R8-1 the exception propagated from the resolve
    step; post-R8-1 the resolve guard falls through to the original method
    which raises from the read step.

    R10-1: replaced bare `return` on symlink-unavailable platforms with
    `pytest.skip` so CI surfaces the skip instead of a vacuous PASS.

    R10-2: the previous `pytest.raises` + `read_text_entries == 0`
    assertions passed whether or not R8-1's resolve guard existed (a bare
    resolve raise also bypasses the cache write). The new
    `test_cached_read_text_resolve_failure_falls_through_to_original`
    test below uses monkeypatch to differentially pin R8-1: it injects a
    distinctive marker into Path.resolve and asserts the marker does NOT
    appear in the user-visible exception, proving the fall-through swapped
    the resolve-step error for a read-step error.
    """
    target = tmp_path / "missing.py"  # never created
    link = tmp_path / "src" / "broken_link.py"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    with cached_filesystem(cache_reads=True):
        with pytest.raises((FileNotFoundError, OSError)):
            link.read_text()
        # Cache invariant (NOT a R8-1 differential pin — see R10-2 test
        # below): a failed read must not leave a cache entry behind.
        stats = cache_stats()
        assert stats["read_text_entries"] == 0, (
            "failed read must not poison the read_text cache; "
            f"got stats={stats}"
        )


def test_cached_read_bytes_passes_through_broken_symlink_to_original(tmp_path: Path) -> None:
    """Round 8 R8-1 + Round 9 R9-1 + Round 10 R10-1 sister test for read_bytes."""
    target = tmp_path / "missing.bin"
    link = tmp_path / "src" / "broken_link.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    with cached_filesystem(cache_reads=True):
        with pytest.raises((FileNotFoundError, OSError)):
            link.read_bytes()
        stats = cache_stats()
        assert stats["read_bytes_entries"] == 0, (
            "failed read must not poison the read_bytes cache; "
            f"got stats={stats}"
        )


def test_cached_read_text_resolve_failure_falls_through_to_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 10 R10-2 fix (2026-05-06, 88%): differentially pin R8-1.

    The previous broken-symlink tests asserted that a failed read raises
    AND doesn't cache. Both assertions pass with or without R8-1's
    resolve guard, because a bare unguarded `self.resolve()` raise also
    bypasses the cache write. So the prior tests did not differentially
    catch a regression that removed R8-1.

    This test injects a UNIQUE marker into Path.resolve so that if R8-1
    is reverted, the user-visible exception will contain the marker
    (proving the resolve-step error reached the caller). With R8-1, the
    fall-through routes to `_ORIGINAL_READ_TEXT(self)` which raises a
    DIFFERENT FileNotFoundError that does NOT contain the marker.
    """
    source_root = tmp_path / "src"
    source_root.mkdir()
    path = source_root / "sample.py"
    # Don't create the file — read_text will fail. But we need resolve()
    # to also fail with our marker so that a missing R8-1 guard would
    # leak the marker out.

    marker = "ROUND_10_R10_2_DISTINCTIVE_RESOLVE_MARKER"
    real_resolve = Path.resolve

    def _raising_resolve(self: Path, *args, **kwargs):
        if self == path:
            raise FileNotFoundError(marker)
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _raising_resolve)

    with cached_filesystem(cache_reads=True):
        with pytest.raises((FileNotFoundError, OSError)) as exc_info:
            path.read_text()

    # R8-1 differential pin: post-R8-1 the resolve raise is CAUGHT and
    # routed to the original read, which raises a different exception.
    # Pre-R8-1 the resolve raise leaks out and the marker appears in the
    # exception. Assert the marker does NOT leak.
    assert marker not in str(exc_info.value), (
        f"R8-1 resolve guard regressed: the resolve-step marker leaked "
        f"into the user-visible exception. Got: {exc_info.value!r}"
    )


def test_preflight_all_wraps_programmatic_codebase_scan_in_fs_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac import preflight as preflight_mod
    import tac.preflight_fs_cache as fs_cache

    class Sentinel(Exception):
        pass

    events: list[tuple[str, bool]] = []
    active = False

    @contextlib.contextmanager
    def fake_cached_filesystem(*, cache_reads: bool = False):
        nonlocal active
        events.append(("enter", cache_reads))
        active = True
        try:
            yield
        finally:
            active = False
            events.append(("exit", cache_reads))

    def fake_check_codebase_drift(*, strict: bool, verbose: bool):
        assert strict is True
        assert verbose is False
        assert active is True
        raise Sentinel("stop after first codebase check")

    monkeypatch.setattr(fs_cache, "cached_filesystem", fake_cached_filesystem)
    monkeypatch.setattr(preflight_mod, "check_codebase_drift", fake_check_codebase_drift)

    with pytest.raises(Sentinel):
        preflight_mod.preflight_all(verbose=False)

    assert events == [("enter", True), ("exit", True)]


def test_preflight_all_cache_can_be_disabled_for_diagnostic_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac import preflight as preflight_mod
    import tac.preflight_fs_cache as fs_cache

    class Sentinel(Exception):
        pass

    @contextlib.contextmanager
    def forbidden_cached_filesystem(*, cache_reads: bool = False):
        raise AssertionError("cache should not be entered")
        yield

    def fake_check_codebase_drift(*, strict: bool, verbose: bool):
        raise Sentinel("stop after first codebase check")

    monkeypatch.setattr(fs_cache, "cached_filesystem", forbidden_cached_filesystem)
    monkeypatch.setattr(preflight_mod, "check_codebase_drift", fake_check_codebase_drift)

    with pytest.raises(Sentinel):
        preflight_mod.preflight_all(verbose=False, use_fs_cache=False)


def test_preflight_all_skips_fs_cache_when_codebase_scan_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac import preflight as preflight_mod
    import tac.preflight_fs_cache as fs_cache

    @contextlib.contextmanager
    def forbidden_cached_filesystem(*, cache_reads: bool = False):
        raise AssertionError("artifactless no-codebase preflight should not cache")
        yield

    monkeypatch.setattr(fs_cache, "cached_filesystem", forbidden_cached_filesystem)

    preflight_mod.preflight_all(check_codebase=False, verbose=False)
