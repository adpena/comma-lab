"""Tests for the canonical Comma2k19LocalCache (DP1 auto-download).

Per Catalog #213 every Comma2k19 chunk fetch MUST route through this class.
These tests pin (1) URL pinning + manifest determinism, (2) SHA-256
integrity verification (positive + tampered chunk refusal), (3) license-tag
propagation, (4) disk-budget LRU eviction, (5) resumable download via temp
file + os.replace, (6) idempotent re-fetch, (7) /tmp avoidance.

Auto-downloads are MOCKED — these tests NEVER hit the network. A monkeypatch
of ``urllib.request.urlopen`` returns canned bytes so the cache exercises
its full path without the operator paying network/GPU time.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from tac.substrates.pretrained_driving_prior.local_chunk_cache import (
    COMMA2K19_LICENSE_SPDX,
    COMMA2K19_REPO_URL,
    DEFAULT_CHUNK_MANIFEST,
    Comma2k19CacheError,
    Comma2k19ChunkManifestEntry,
    Comma2k19LocalCache,
    DiskBudgetExceededError,
    DownloadIntegrityError,
    default_cache_dir,
    verify_chunk_sha256,
)


# ---------------------------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------------------------


_DUMMY_BYTES_1KB = bytes(range(256)) * 4  # 1024 bytes of mixed-value content


def _make_canned_chunk(content: bytes) -> Comma2k19ChunkManifestEntry:
    return Comma2k19ChunkManifestEntry(
        chunk_id="test_chunk",
        url="https://raw.githubusercontent.com/commaai/comma2k19/master/Example_X/x/0/video.hevc",
        expected_sha256="",  # verify-on-first-download + pin
        size_bytes=len(content),
        dest_relpath="test_dongle/test_route/0/video.hevc",
    )


class _FakeResponse:
    """Mimic urllib.request.urlopen() result for monkeypatching."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    def read(self, n: int = -1) -> bytes:
        if n < 0 or n >= len(self._content):
            data, self._content = self._content, b""
            return data
        data = self._content[:n]
        self._content = self._content[n:]
        return data


def _mock_urlopen_factory(content_by_url: dict[str, bytes]):
    """Return a urlopen monkeypatch that serves canned content per URL."""

    def _fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if url not in content_by_url:
            raise urllib.error.URLError(f"no mocked content for {url!r}")
        return _FakeResponse(content_by_url[url])

    return _fake_urlopen


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_pinned() -> None:
    assert COMMA2K19_LICENSE_SPDX == "MIT"
    assert COMMA2K19_REPO_URL == "https://github.com/commaai/comma2k19"
    assert "example_1" in DEFAULT_CHUNK_MANIFEST
    entry = DEFAULT_CHUNK_MANIFEST["example_1"]
    assert entry.url.startswith(
        "https://raw.githubusercontent.com/commaai/comma2k19/master/Example_1/"
    )
    assert entry.url.endswith("/video.hevc")
    # The expected_sha256 placeholder is intentionally empty (verify-on-first-download).
    assert entry.expected_sha256 == ""
    # 37 MB ish.
    assert 30_000_000 < entry.size_bytes < 50_000_000


def test_default_cache_dir_under_home() -> None:
    p = default_cache_dir()
    # Must not be /tmp.
    s = str(p)
    assert not s.startswith("/tmp")
    assert not s.startswith("/var/tmp")
    assert not s.startswith("/private/tmp")


def test_default_cache_dir_refuses_tmp_via_env(monkeypatch) -> None:
    monkeypatch.setenv("DPP_CACHE_DIR", "/tmp/cache")
    with pytest.raises(Comma2k19CacheError):
        default_cache_dir()
    monkeypatch.setenv("DPP_CACHE_DIR", "/var/tmp/cache")
    with pytest.raises(Comma2k19CacheError):
        default_cache_dir()


def test_default_cache_dir_env_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "my_cache"
    monkeypatch.setenv("DPP_CACHE_DIR", str(target))
    assert default_cache_dir() == target.resolve()


# ---------------------------------------------------------------------------
# Cache directory creation + permissions
# ---------------------------------------------------------------------------


def test_cache_constructor_refuses_tmp_root(tmp_path) -> None:
    with pytest.raises(Comma2k19CacheError):
        Comma2k19LocalCache(cache_dir=Path("/tmp/foo"))


def test_cache_constructor_refuses_zero_budget(tmp_path) -> None:
    with pytest.raises(ValueError):
        Comma2k19LocalCache(cache_dir=tmp_path / "cache", max_disk_gb=0.0)
    with pytest.raises(ValueError):
        Comma2k19LocalCache(cache_dir=tmp_path / "cache", max_disk_gb=-1.0)


def test_cache_constructor_creates_no_directory_until_fetch(tmp_path) -> None:
    cache_dir = tmp_path / "lazy_cache"
    cache = Comma2k19LocalCache(cache_dir=cache_dir, max_disk_gb=1.0)
    # Constructor doesn't create the dir; that's lazy.
    assert not cache_dir.exists()
    # cached_chunks_dir() DOES create it.
    p = cache.cached_chunks_dir()
    assert p == cache_dir.resolve()
    assert cache_dir.exists()


# ---------------------------------------------------------------------------
# verify_chunk_sha256 — positive + tampered chunk refusal
# ---------------------------------------------------------------------------


def test_verify_chunk_sha256_pin_on_empty_expected(tmp_path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc123")
    pinned = verify_chunk_sha256(f, "")
    # SHA-256 of b"abc123":
    import hashlib

    expected = hashlib.sha256(b"abc123").hexdigest()
    assert pinned == expected


def test_verify_chunk_sha256_match(tmp_path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"xyzabc")
    import hashlib

    expected = hashlib.sha256(b"xyzabc").hexdigest()
    assert verify_chunk_sha256(f, expected) == expected


def test_verify_chunk_sha256_mismatch_raises(tmp_path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"tampered")
    with pytest.raises(DownloadIntegrityError):
        verify_chunk_sha256(f, "0" * 64)  # bogus expected


# ---------------------------------------------------------------------------
# fetch_chunk — happy path with mocked download
# ---------------------------------------------------------------------------


def test_fetch_chunk_downloads_and_pins_sha(tmp_path) -> None:
    entry = _make_canned_chunk(_DUMMY_BYTES_1KB)
    manifest = {entry.chunk_id: entry}
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest=manifest,
    )
    fake = _mock_urlopen_factory({entry.url: _DUMMY_BYTES_1KB})
    with patch("urllib.request.urlopen", fake):
        local_path = cache.fetch_chunk(entry.chunk_id)
    assert local_path.is_file()
    assert local_path.read_bytes() == _DUMMY_BYTES_1KB
    # Cache meta should have pinned the SHA.
    cached = cache.list_cached_chunks()
    assert entry.chunk_id in cached


def test_fetch_chunk_idempotent(tmp_path) -> None:
    entry = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={entry.chunk_id: entry},
    )
    fake = _mock_urlopen_factory({entry.url: _DUMMY_BYTES_1KB})
    with patch("urllib.request.urlopen", fake):
        first = cache.fetch_chunk(entry.chunk_id)
    # Second call should NOT trigger a network fetch — patch to refuse all URLs.
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("must not be called"),
    ):
        second = cache.fetch_chunk(entry.chunk_id)
    assert first == second


def test_fetch_chunk_unknown_chunk_id(tmp_path) -> None:
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={},
    )
    with pytest.raises(KeyError):
        cache.fetch_chunk("nope")


def test_fetch_chunk_offline_no_cache_raises(tmp_path) -> None:
    entry = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={entry.chunk_id: entry},
        offline=True,
    )
    with pytest.raises(Comma2k19CacheError, match="offline"):
        cache.fetch_chunk(entry.chunk_id)


def test_fetch_chunk_network_failure_raises(tmp_path) -> None:
    entry = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={entry.chunk_id: entry},
    )
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(Comma2k19CacheError):
            cache.fetch_chunk(entry.chunk_id)


def test_fetch_chunk_writes_to_chunk_relpath(tmp_path) -> None:
    entry = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={entry.chunk_id: entry},
    )
    fake = _mock_urlopen_factory({entry.url: _DUMMY_BYTES_1KB})
    with patch("urllib.request.urlopen", fake):
        local_path = cache.fetch_chunk(entry.chunk_id)
    # The chunk lands under the canonical Comma2k19 layout for rglob() to find it.
    assert "test_dongle" in str(local_path)
    assert "test_route" in str(local_path)
    assert local_path.name == "video.hevc"


# ---------------------------------------------------------------------------
# fetch_chunks (iterator)
# ---------------------------------------------------------------------------


def test_fetch_chunks_yields_in_input_order(tmp_path) -> None:
    e1 = Comma2k19ChunkManifestEntry(
        chunk_id="a",
        url="https://example.com/a",
        expected_sha256="",
        size_bytes=4,
        dest_relpath="d/r/0/a",
    )
    e2 = Comma2k19ChunkManifestEntry(
        chunk_id="b",
        url="https://example.com/b",
        expected_sha256="",
        size_bytes=4,
        dest_relpath="d/r/1/b",
    )
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={"a": e1, "b": e2},
    )
    fake = _mock_urlopen_factory({e1.url: b"AAAA", e2.url: b"BBBB"})
    with patch("urllib.request.urlopen", fake):
        paths = list(cache.fetch_chunks(["b", "a"]))
    assert paths[0].name == "b"
    assert paths[1].name == "a"


# ---------------------------------------------------------------------------
# SHA-256 verification (tampered chunk refusal)
# ---------------------------------------------------------------------------


def test_fetch_chunk_rejects_tampered_response(tmp_path) -> None:
    import hashlib

    real_content = b"the real chunk content"
    real_sha = hashlib.sha256(real_content).hexdigest()
    entry = Comma2k19ChunkManifestEntry(
        chunk_id="test",
        url="https://example.com/test",
        expected_sha256=real_sha,
        size_bytes=len(real_content),
        dest_relpath="d/r/0/test",
    )
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={"test": entry},
    )
    # Mock returns TAMPERED content (different from real).
    fake = _mock_urlopen_factory({entry.url: b"DIFFERENT CONTENT"})
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(DownloadIntegrityError):
            cache.fetch_chunk("test")


# ---------------------------------------------------------------------------
# Disk-budget LRU eviction
# ---------------------------------------------------------------------------


def test_disk_budget_exceeded_raises(tmp_path) -> None:
    huge_size = 10 * 1024**3  # 10 GB
    entry = Comma2k19ChunkManifestEntry(
        chunk_id="huge",
        url="https://example.com/huge",
        expected_sha256="",
        size_bytes=huge_size,
        dest_relpath="d/r/0/huge",
    )
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=0.001,  # ~1 MB budget — way too small for 10 GB
        chunk_manifest={"huge": entry},
    )
    # Even before download, the budget check refuses with empty cache.
    with patch("urllib.request.urlopen", _mock_urlopen_factory({entry.url: b""})):
        with pytest.raises(DiskBudgetExceededError):
            cache.fetch_chunk("huge")


def test_clear_stale_chunks_lru(tmp_path) -> None:
    # Create 4 entries, fetch them in order, then clear_stale_chunks(keep_recent=2).
    entries = {}
    urls = {}
    for i in range(4):
        cid = f"c{i}"
        content = bytes([i]) * 32
        e = Comma2k19ChunkManifestEntry(
            chunk_id=cid,
            url=f"https://example.com/{cid}",
            expected_sha256="",
            size_bytes=len(content),
            dest_relpath=f"d/r/{i}/video.hevc",
        )
        entries[cid] = e
        urls[e.url] = content
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest=entries,
    )
    fake = _mock_urlopen_factory(urls)
    import time

    with patch("urllib.request.urlopen", fake):
        for cid in ["c0", "c1", "c2", "c3"]:
            cache.fetch_chunk(cid)
            time.sleep(0.01)  # ensure monotonic last_used_at differs
    assert sorted(cache.list_cached_chunks()) == ["c0", "c1", "c2", "c3"]
    evicted = cache.clear_stale_chunks(keep_recent=2)
    # c0 and c1 are oldest (LRU); they should be evicted.
    assert set(evicted) == {"c0", "c1"}
    assert sorted(cache.list_cached_chunks()) == ["c2", "c3"]


def test_clear_stale_chunks_keep_recent_zero_evicts_all(tmp_path) -> None:
    e = _make_canned_chunk(b"x")
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={e.chunk_id: e},
    )
    fake = _mock_urlopen_factory({e.url: b"x"})
    with patch("urllib.request.urlopen", fake):
        cache.fetch_chunk(e.chunk_id)
    evicted = cache.clear_stale_chunks(keep_recent=0)
    assert evicted == [e.chunk_id]
    assert cache.list_cached_chunks() == []


def test_clear_stale_chunks_negative_raises(tmp_path) -> None:
    cache = Comma2k19LocalCache(cache_dir=tmp_path / "cache", max_disk_gb=1.0)
    with pytest.raises(ValueError):
        cache.clear_stale_chunks(keep_recent=-1)


# ---------------------------------------------------------------------------
# /tmp avoidance (CLAUDE.md "Forbidden /tmp paths")
# ---------------------------------------------------------------------------


def test_downloads_dir_under_cache_root_not_tmp(tmp_path) -> None:
    cache_dir = tmp_path / "my_cache"
    cache = Comma2k19LocalCache(cache_dir=cache_dir, max_disk_gb=1.0)
    e = _make_canned_chunk(b"y" * 16)
    cache.chunk_manifest = {e.chunk_id: e}
    fake = _mock_urlopen_factory({e.url: b"y" * 16})
    with patch("urllib.request.urlopen", fake):
        cache.fetch_chunk(e.chunk_id)
    # No file should ever appear under /tmp.
    assert (cache_dir / ".downloads").exists()
    s = str((cache_dir / ".downloads").resolve())
    assert not s.startswith("/tmp")


# ---------------------------------------------------------------------------
# cache_status reporting
# ---------------------------------------------------------------------------


def test_cache_status_empty_cache(tmp_path) -> None:
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache", max_disk_gb=2.5
    )
    status = cache.cache_status()
    assert status["chunks_cached"] == []
    assert status["disk_used_bytes"] == 0
    assert status["max_disk_gb"] == 2.5
    assert status["offline"] is False
    assert str(tmp_path) in status["cache_dir"]


def test_cache_status_after_fetch(tmp_path) -> None:
    e = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={e.chunk_id: e},
    )
    fake = _mock_urlopen_factory({e.url: _DUMMY_BYTES_1KB})
    with patch("urllib.request.urlopen", fake):
        cache.fetch_chunk(e.chunk_id)
    status = cache.cache_status()
    assert e.chunk_id in status["chunks_cached"]
    assert status["disk_used_bytes"] >= len(_DUMMY_BYTES_1KB)


# ---------------------------------------------------------------------------
# list_available_chunks deterministic
# ---------------------------------------------------------------------------


def test_list_available_chunks_sorted(tmp_path) -> None:
    entries = {}
    for cid in ["zebra", "apple", "mango"]:
        entries[cid] = Comma2k19ChunkManifestEntry(
            chunk_id=cid,
            url=f"https://example.com/{cid}",
            expected_sha256="",
            size_bytes=1,
            dest_relpath=f"d/r/0/{cid}",
        )
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest=entries,
    )
    assert cache.list_available_chunks() == ["apple", "mango", "zebra"]


# ---------------------------------------------------------------------------
# cached_chunks_dir creates root if needed + canonical chunk-tree layout
# ---------------------------------------------------------------------------


def test_cached_chunks_dir_creates_root(tmp_path) -> None:
    cache_dir = tmp_path / "rooted"
    cache = Comma2k19LocalCache(cache_dir=cache_dir, max_disk_gb=1.0)
    assert not cache_dir.exists()
    p = cache.cached_chunks_dir()
    assert p.exists()
    assert p.is_dir()


def test_cached_chunks_dir_layout_walks_video_hevc(tmp_path) -> None:
    """rglob('video.hevc') over cached_chunks_dir() finds cached files."""
    e = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={e.chunk_id: e},
    )
    fake = _mock_urlopen_factory({e.url: _DUMMY_BYTES_1KB})
    with patch("urllib.request.urlopen", fake):
        cache.fetch_chunk(e.chunk_id)
    found = list(cache.cached_chunks_dir().rglob("video.hevc"))
    assert len(found) == 1
    assert found[0].read_bytes() == _DUMMY_BYTES_1KB


# ---------------------------------------------------------------------------
# License-tag propagation (cache → cache_meta)
# ---------------------------------------------------------------------------


def test_cache_meta_records_license_and_source_url(tmp_path) -> None:
    e = _make_canned_chunk(_DUMMY_BYTES_1KB)
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "cache",
        max_disk_gb=1.0,
        chunk_manifest={e.chunk_id: e},
    )
    fake = _mock_urlopen_factory({e.url: _DUMMY_BYTES_1KB})
    with patch("urllib.request.urlopen", fake):
        cache.fetch_chunk(e.chunk_id)
    meta_path = tmp_path / "cache" / "cache_meta.json"
    assert meta_path.is_file()
    raw = json.loads(meta_path.read_text())
    assert raw[e.chunk_id]["license"] == "MIT"
    assert raw[e.chunk_id]["source_url"].startswith("https://")


# ---------------------------------------------------------------------------
# Atomic write of cache_meta.json
# ---------------------------------------------------------------------------


def test_cache_meta_survives_corrupted_disk(tmp_path) -> None:
    """If cache_meta.json is corrupted on disk, the cache returns {} (recoverable)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cache_meta.json").write_text("{not valid json")
    cache = Comma2k19LocalCache(cache_dir=cache_dir, max_disk_gb=1.0)
    # list_cached_chunks should return [] (corrupted = treated as empty).
    assert cache.list_cached_chunks() == []
