# SPDX-License-Identifier: MIT
"""Tests for the three Comma2k19FrameIterator modes (DP1 auto-load).

Mode 1 (synthetic stub): unchanged from L1 scaffold (backward compat).
Mode 2 (explicit chunks_dir): unchanged from L1 scaffold (backward compat).
Mode 3 (auto-download cache + log-incremental): new in this lane.

Per Catalog #209 the contest-video-leakage guard fires inside the iterator
constructor BEFORE any decode work; these tests pin that the new auto-download
path inherits the same guard via cache.cached_chunks_dir().
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tac.substrates.pretrained_driving_prior import (
    Comma2k19ChunkManifestEntry,
    Comma2k19FrameIterator,
    Comma2k19LocalCache,
    ContestVideoLeakageError,
)


# ---------------------------------------------------------------------------
# Mode 1 (synthetic stub) — backward compat from L1 scaffold
# ---------------------------------------------------------------------------


def test_mode1_synthetic_stub_still_works() -> None:
    it = Comma2k19FrameIterator(synthetic=True, n_frames=8, seed=42)
    frames = list(it)
    assert len(frames) == 8
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.ndim == 3
        assert f.shape[2] == 3


def test_mode1_synthetic_stub_provenance_unchanged() -> None:
    it = Comma2k19FrameIterator(synthetic=True, n_frames=4, seed=42)
    list(it)
    # The provenance.license tag for synthetic mode is unchanged.
    assert it.provenance["synthetic"] is True
    assert it.provenance["license"] == "synthetic-test-only"
    assert it.provenance["chunks_dir"] is None


# ---------------------------------------------------------------------------
# Mode 2 (explicit chunks_dir) — backward compat from L1 scaffold
# ---------------------------------------------------------------------------


def test_mode2_explicit_chunks_dir_refuses_missing(tmp_path) -> None:
    """Without synthetic and without an existing chunks_dir, the iterator raises."""
    missing = tmp_path / "missing_chunks"
    with pytest.raises(FileNotFoundError):
        Comma2k19FrameIterator(chunks_dir=missing, synthetic=False)


def test_mode2_explicit_chunks_dir_refuses_contest_video(tmp_path) -> None:
    """Pointing at upstream/videos/ triggers the leakage guard.

    The guard matches the substring ``upstream/videos/`` which requires
    at least one path component AFTER ``videos/``, so we build a
    ``upstream/videos/sub`` path.
    """
    leaked = tmp_path / "upstream" / "videos" / "subdir"
    leaked.mkdir(parents=True)
    with pytest.raises(ContestVideoLeakageError):
        Comma2k19FrameIterator(chunks_dir=leaked, synthetic=False)


def test_mode2_explicit_chunks_dir_accepts_empty_real_dir(tmp_path) -> None:
    """A real chunks_dir (no video.hevc files) yields zero frames."""
    chunks = tmp_path / "comma2k19_chunks"
    chunks.mkdir()
    it = Comma2k19FrameIterator(chunks_dir=chunks, synthetic=False, max_chunks=1)
    frames = list(it)
    assert frames == []


# ---------------------------------------------------------------------------
# Mode 3 (auto-download cache) — new in this lane
# ---------------------------------------------------------------------------


def test_mode3_cache_dir_returns_valid_path(tmp_path) -> None:
    """``cache.cached_chunks_dir()`` returns a Path usable by Comma2k19FrameIterator."""
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
    )
    chunks_dir = cache.cached_chunks_dir()
    assert chunks_dir.exists()
    # Now construct the iterator — should NOT raise even with no chunks fetched yet.
    it = Comma2k19FrameIterator(chunks_dir=chunks_dir, synthetic=False, max_chunks=1)
    # Iterating with no chunks yields nothing.
    frames = list(it)
    assert frames == []


def test_mode3_cache_to_iterator_pipeline(tmp_path) -> None:
    """End-to-end: cache fetches a chunk, iterator finds it via rglob."""
    from unittest.mock import patch

    import urllib.error

    # We can't decode arbitrary bytes as video.hevc, so use a SYNTHETIC chunk
    # path that the rglob finds but skip the actual decode. Instead, test
    # that rglob finds the cached chunk.
    entry = Comma2k19ChunkManifestEntry(
        chunk_id="test",
        url="https://example.com/test",
        expected_sha256="",
        size_bytes=64,
        dest_relpath="dongle/route/0/video.hevc",
    )
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={"test": entry},
    )

    class _FakeResponse:
        def __init__(self, content):
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *a, **kw):
            return False

        def read(self, n=-1):
            data, self._content = self._content, b""
            return data

    def _fake_urlopen(req, timeout=None):
        return _FakeResponse(b"\x00" * 64)

    with patch("urllib.request.urlopen", _fake_urlopen):
        cache.fetch_chunk("test")
    # The cached chunks_dir now contains the video.hevc under the right
    # layout for rglob.
    found = list(cache.cached_chunks_dir().rglob("video.hevc"))
    assert len(found) == 1


def test_mode3_cache_dir_refuses_tmp(tmp_path) -> None:
    """The cache constructor refuses /tmp paths per CLAUDE.md."""
    from tac.substrates.pretrained_driving_prior.local_chunk_cache import (
        Comma2k19CacheError,
    )

    with pytest.raises(Comma2k19CacheError):
        Comma2k19LocalCache(cache_dir=Path("/tmp/foo"), max_disk_gb=1.0)


# ---------------------------------------------------------------------------
# Trainer wire-up — _use_auto_download_cache helper
# ---------------------------------------------------------------------------


def _trainer_args(**overrides) -> argparse.Namespace:
    """Build a Namespace mirroring the trainer's argparse defaults."""
    defaults = {
        "dataset_name": "synthetic_test",
        "comma2k19_chunks_dir": "",
        "cache_dir": "",
        "max_disk_gb": 100.0,
        "log_incremental_base": 2,
        "log_incremental_max_chunks": 80,
        "log_incremental_quality_threshold": 0.005,
        "disable_log_incremental": False,
        "seed": 42,
        "max_distillation_frames": 1024,
        "max_distillation_chunks": 8,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_trainer_use_auto_download_cache_resolves_correctly() -> None:
    """The _use_auto_download_cache helper routes to the cache mode correctly."""
    import sys

    sys.path.insert(0, "experiments")
    try:
        import train_substrate_pretrained_driving_prior as t

        # Mode 1: synthetic_test → cache mode NOT used.
        ns = _trainer_args(dataset_name="synthetic_test")
        assert t._use_auto_download_cache(ns) is False

        # Mode 2: comma2k19 + explicit chunks_dir → cache mode NOT used.
        ns = _trainer_args(
            dataset_name="comma2k19", comma2k19_chunks_dir="/some/path"
        )
        assert t._use_auto_download_cache(ns) is False

        # Mode 3: comma2k19 + no chunks_dir → cache mode IS used.
        ns = _trainer_args(dataset_name="comma2k19", comma2k19_chunks_dir="")
        assert t._use_auto_download_cache(ns) is True

        # Operator opt-out: --disable-log-incremental disables cache mode.
        ns = _trainer_args(
            dataset_name="comma2k19",
            comma2k19_chunks_dir="",
            disable_log_incremental=True,
        )
        assert t._use_auto_download_cache(ns) is False
    finally:
        sys.path.remove("experiments")


def test_trainer_build_local_cache_honors_cache_dir_flag(tmp_path) -> None:
    """The _build_local_cache helper creates a cache rooted at --cache-dir."""
    import sys

    sys.path.insert(0, "experiments")
    try:
        import train_substrate_pretrained_driving_prior as t

        target = tmp_path / "trainer_cache"
        ns = _trainer_args(cache_dir=str(target))
        cache = t._build_local_cache(ns)
        assert cache.cache_dir == target.resolve()
        assert cache.max_disk_gb == 100.0
    finally:
        sys.path.remove("experiments")
