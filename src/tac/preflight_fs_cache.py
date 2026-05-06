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
from collections.abc import Iterator
from pathlib import Path

# Globals so the patched methods can find the caches without bound state
_RGLOB_CACHE: dict[tuple[str, str], list[Path]] = {}
_READ_TEXT_CACHE: dict[str, str] = {}
_READ_BYTES_CACHE: dict[str, bytes] = {}

_ORIGINAL_RGLOB = Path.rglob
_ORIGINAL_READ_TEXT = Path.read_text
_ORIGINAL_READ_BYTES = Path.read_bytes


def _cached_rglob(self: Path, pattern: str, *args, **kwargs):  # type: ignore[override]
    """Cached `Path.rglob` — preserves return shape (a generator-like)."""
    # Only cache the simple .rglob("*.py")-style call; pass through anything fancier
    # to avoid changing semantics for hidden args.
    if args or kwargs:
        return _ORIGINAL_RGLOB(self, pattern, *args, **kwargs)
    key = (str(self.resolve()), pattern)
    cached = _RGLOB_CACHE.get(key)
    if cached is None:
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
    """
    patched_rglob = Path.rglob is not _cached_rglob
    patched_reads = cache_reads and Path.read_text is not _cached_read_text
    if patched_rglob:
        Path.rglob = _cached_rglob  # type: ignore[method-assign]
    if patched_reads:
        Path.read_text = _cached_read_text  # type: ignore[method-assign]
        Path.read_bytes = _cached_read_bytes  # type: ignore[method-assign]
    try:
        yield
    finally:
        if patched_reads:
            Path.read_text = _ORIGINAL_READ_TEXT  # type: ignore[method-assign]
            Path.read_bytes = _ORIGINAL_READ_BYTES  # type: ignore[method-assign]
            _READ_TEXT_CACHE.clear()
            _READ_BYTES_CACHE.clear()
        if patched_rglob:
            Path.rglob = _ORIGINAL_RGLOB  # type: ignore[method-assign]
            _RGLOB_CACHE.clear()
        if patched_rglob and not patched_reads:
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
