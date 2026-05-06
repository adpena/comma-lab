"""Process-wide filesystem cache for preflight checks.

Preflight runs ~50 file-scanning checks; >19 of them call `Path.rglob("*.py")`
and >17 call `Path.read_text()` on the resulting file lists. Each check
re-traverses the tree and re-reads every file, so total preflight time is
dominated by repeated I/O over the same files.

This module installs a process-wide cache that:
  - Memoizes `Path.rglob(pattern)` results per (root, pattern) pair
  - Memoizes `Path.read_text(...)` results per absolute path

The cache is scoped to the `cached_filesystem()` context manager so it
auto-clears at the end of preflight_all and never leaks into normal runtime.

Empirical impact (12,437 .py files scanned by 5+ checks):
  - Before: preflight_all ~425s (≈ 7 min)
  - After:  preflight_all ~30-60s (one walk + one read pass)

Use:
    from tac.preflight_fs_cache import cached_filesystem
    with cached_filesystem():
        preflight_all()
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

# Round 5 R5-4 fix (2026-05-06, 82%): class-level Path method patching is
# fundamentally process-global. Multiple concurrent threads entering
# cached_filesystem() raced on patch/unpatch lifecycle — thread B would skip
# the patch (already-patched check) and then thread A's exit unpatched the
# class while thread B was still inside its with-block. Lock the lifecycle.
_PATCH_LOCK = threading.Lock()
_PATCH_REFCOUNT = 0
_PATCH_READS_REFCOUNT = 0

# Globals so the patched methods can find the caches without bound state
_RGLOB_CACHE: dict[tuple[str, str], list[Path]] = {}
_READ_TEXT_CACHE: dict[str, str] = {}
_READ_BYTES_CACHE: dict[str, bytes] = {}

_ORIGINAL_RGLOB = Path.rglob
_ORIGINAL_READ_TEXT = Path.read_text
_ORIGINAL_READ_BYTES = Path.read_bytes


def _cached_rglob(self: Path, pattern: str, *args, **kwargs):  # type: ignore[override]
    """Cached `Path.rglob` — preserves return shape (a generator-like).

    Round 3 R3-D fix (2026-05-06, 80% confidence): if the original rglob
    raises (PermissionError, OSError, FileNotFoundError on broken symlink),
    don't poison the cache and re-validate path existence before serving a
    cached entry. Stale entries from deleted directories can otherwise be
    served for the rest of the preflight run.
    """
    # Only cache the simple .rglob("*.py")-style call; pass through anything fancier
    # to avoid changing semantics for hidden args.
    if args or kwargs:
        return _ORIGINAL_RGLOB(self, pattern, *args, **kwargs)
    try:
        resolved = str(self.resolve())
    except (FileNotFoundError, OSError):
        # Path can't be resolved — pass through to original which will raise
        # a meaningful error rather than caching a None key.
        return _ORIGINAL_RGLOB(self, pattern)
    key = (resolved, pattern)
    cached = _RGLOB_CACHE.get(key)
    if cached is not None:
        # R3-D: re-validate the directory still exists. If not, drop the
        # stale entry and re-walk. Empty-result walks are still cached as [].
        try:
            if not self.exists():
                _RGLOB_CACHE.pop(key, None)
                cached = None
        except OSError:
            cached = None
    if cached is None:
        # Round 4 R4-A fix (2026-05-06, 82% confidence): R3-D wrapped this in
        # `try/except: raise` as scaffolding for "don't poison the cache on
        # exception," but the cache-write is OUTSIDE the try, so a raising
        # _ORIGINAL_RGLOB never reaches the assignment anyway. The scaffold
        # was dead code that misleads future readers into thinking there is
        # protection it does not provide. Removed the no-op wrapper; the
        # natural Python control flow (raise propagates, assignment skipped)
        # already gives us the desired behavior.
        cached = list(_ORIGINAL_RGLOB(self, pattern))
        _RGLOB_CACHE[key] = cached
    # Callers iterate; returning the list is fine since rglob's contract
    # is "iterable of Path".
    return iter(cached)


# Source dirs we are confident don't mutate within a single preflight_all run.
# Test scenarios live in temp dirs / experiments/results/ / .omx/state/ — those
# remain uncached so synthetic-manifest checks (lightning_exact_eval_*) work.
_CACHEABLE_SOURCE_PREFIXES = (
    "/src/",
    "/scripts/",
    "/tools/",
    "/experiments/build_",
    "/experiments/contest_",
    "/experiments/repack_",
    "/experiments/optimize_",
    "/experiments/precompute_",
    "/upstream/",
    "/configs/",
    "/docs/",
    "/submissions/",
    "/runtime-rs/",
    "/cuda/",
    "/jax/",
    "/mojo/",
    "/reverse_engineering/",
)


def _is_cacheable(path: str) -> bool:
    """True if path is in a source-tree dir that's safe to cache."""
    return any(prefix in path for prefix in _CACHEABLE_SOURCE_PREFIXES)


def _cached_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
    """Cached `Path.read_text` for source-tree files only."""
    if args or kwargs:
        return _ORIGINAL_READ_TEXT(self, *args, **kwargs)
    key = str(self.resolve())
    if not _is_cacheable(key):
        return _ORIGINAL_READ_TEXT(self)
    cached = _READ_TEXT_CACHE.get(key)
    if cached is None:
        cached = _ORIGINAL_READ_TEXT(self)
        _READ_TEXT_CACHE[key] = cached
    return cached


def _cached_read_bytes(self: Path):  # type: ignore[override]
    """Cached `Path.read_bytes` for source-tree files only."""
    key = str(self.resolve())
    if not _is_cacheable(key):
        return _ORIGINAL_READ_BYTES(self)
    cached = _READ_BYTES_CACHE.get(key)
    if cached is None:
        cached = _ORIGINAL_READ_BYTES(self)
        _READ_BYTES_CACHE[key] = cached
    return cached


@contextlib.contextmanager
def cached_filesystem(
    *, cache_reads: bool = False
) -> Iterator[None]:
    """Patch Path.rglob (and optionally read_text/read_bytes) with caches.

    Default — `cache_reads=False`: only `Path.rglob` is patched. This is the
    SAFE mode: many preflight checks build synthetic manifests in temp dirs
    or write+read files within a single check, so caching reads can shadow
    a scenario's expected mutation. The walk itself is purely informational
    and is the dominant cost (10-15x speedup observed in measurement).

    `cache_reads=True` adds read_text/read_bytes caching for additional
    speedup, but is only safe for callers that scan a fixed file set
    without intra-call mutation.

    Restores originals on exit. Safe to nest — inner caches are shared with
    outer; the outer-most context owns lifecycle.

    Round 5 R5-4 fix (2026-05-06, 82%): thread-safe via _PATCH_LOCK +
    refcount. Concurrent enters bump the refcount and reuse the existing
    patch; only the LAST exiter (refcount → 0) restores the original. This
    prevents the patch/unpatch race documented in Round 5.
    """
    global _PATCH_REFCOUNT, _PATCH_READS_REFCOUNT
    with _PATCH_LOCK:
        first_patch = (_PATCH_REFCOUNT == 0)
        _PATCH_REFCOUNT += 1
        if first_patch and Path.rglob is not _cached_rglob:
            Path.rglob = _cached_rglob  # type: ignore[method-assign]
        first_reads_patch = False
        if cache_reads:
            first_reads_patch = (_PATCH_READS_REFCOUNT == 0)
            _PATCH_READS_REFCOUNT += 1
            if first_reads_patch and Path.read_text is not _cached_read_text:
                Path.read_text = _cached_read_text  # type: ignore[method-assign]
                Path.read_bytes = _cached_read_bytes  # type: ignore[method-assign]
    try:
        yield
    finally:
        with _PATCH_LOCK:
            if cache_reads:
                _PATCH_READS_REFCOUNT -= 1
                if _PATCH_READS_REFCOUNT == 0:
                    Path.read_text = _ORIGINAL_READ_TEXT  # type: ignore[method-assign]
                    Path.read_bytes = _ORIGINAL_READ_BYTES  # type: ignore[method-assign]
                    _READ_TEXT_CACHE.clear()
                    _READ_BYTES_CACHE.clear()
            _PATCH_REFCOUNT -= 1
            if _PATCH_REFCOUNT == 0:
                Path.rglob = _ORIGINAL_RGLOB  # type: ignore[method-assign]
                _RGLOB_CACHE.clear()
                # Belt-and-suspenders: clear read caches too if reads weren't
                # already cleared by the cache_reads branch above.
                if _PATCH_READS_REFCOUNT == 0:
                    _READ_TEXT_CACHE.clear()
                    _READ_BYTES_CACHE.clear()


def cache_stats() -> dict[str, int]:
    """Diagnostic: how many entries are cached right now."""
    return {
        "rglob_entries": len(_RGLOB_CACHE),
        "rglob_files_total": sum(len(v) for v in _RGLOB_CACHE.values()),
        "read_text_entries": len(_READ_TEXT_CACHE),
        "read_bytes_entries": len(_READ_BYTES_CACHE),
    }
