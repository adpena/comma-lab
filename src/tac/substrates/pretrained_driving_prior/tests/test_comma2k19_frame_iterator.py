"""Tests for the canonical Comma2k19FrameIterator (DP1 Phase 2).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L1
+ Catalog #209 routing contract: every codebook distillation MUST go
through this iterator class so :func:`check_no_contest_video_leakage`
runs BEFORE any decode work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.substrates.pretrained_driving_prior import (
    Comma2k19FrameIterator,
    ContestVideoLeakageError,
    DistillationConfig,
    distill_codebook,
)


# ---------------------------------------------------------------------------
# Synthetic-mode contract
# ---------------------------------------------------------------------------


def test_synthetic_mode_yields_uint8_HWC_frames() -> None:
    it = Comma2k19FrameIterator(synthetic=True, n_frames=8, seed=1234)
    frames = list(it)
    assert len(frames) == 8
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.ndim == 3
        assert f.shape[2] == 3


def test_synthetic_mode_is_deterministic() -> None:
    a = list(Comma2k19FrameIterator(synthetic=True, n_frames=4, seed=42))
    b = list(Comma2k19FrameIterator(synthetic=True, n_frames=4, seed=42))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_synthetic_mode_provenance_tags() -> None:
    it = Comma2k19FrameIterator(synthetic=True, n_frames=4, seed=42)
    list(it)
    assert it.provenance["synthetic"] is True
    assert it.provenance["license"] == "synthetic-test-only"
    assert it.provenance["chunks_dir"] is None
    assert it.provenance["frame_count_used"] == 4
    assert it.provenance["source_url"] is None


def test_synthetic_with_chunks_dir_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        Comma2k19FrameIterator(chunks_dir=Path("/some/path"), synthetic=True)


def test_nonsynthetic_without_chunks_dir_rejected() -> None:
    with pytest.raises(ValueError, match="invalid"):
        Comma2k19FrameIterator(chunks_dir=None, synthetic=False)


# ---------------------------------------------------------------------------
# Contest-video-leakage refusal
# ---------------------------------------------------------------------------


def test_contest_video_path_refused_in_constructor(tmp_path: Path) -> None:
    """Path containing 'upstream/videos/<file>' MUST be refused before decode."""
    leak_path = tmp_path / "upstream" / "videos" / "subdir"
    leak_path.mkdir(parents=True)
    with pytest.raises(ContestVideoLeakageError):
        Comma2k19FrameIterator(chunks_dir=leak_path)


def test_contest_video_filename_refused() -> None:
    with pytest.raises(ContestVideoLeakageError):
        Comma2k19FrameIterator(chunks_dir=Path("/some/path/0.mkv"))


def test_contest_challenge_path_refused() -> None:
    with pytest.raises(ContestVideoLeakageError):
        Comma2k19FrameIterator(
            chunks_dir=Path("/data/comma_video_compression_challenge/chunks")
        )


def test_missing_chunks_dir_raises_filenotfound(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent_chunks"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Comma2k19FrameIterator(chunks_dir=nonexistent)


# ---------------------------------------------------------------------------
# Real-mode iteration over an empty chunks dir
# ---------------------------------------------------------------------------


def test_empty_real_chunks_dir_yields_zero_frames(tmp_path: Path) -> None:
    """Empty (no video.hevc files) chunks dir yields nothing — no error."""
    chunks_dir = tmp_path / "comma2k19_chunks"
    chunks_dir.mkdir()
    it = Comma2k19FrameIterator(chunks_dir=chunks_dir, max_chunks=3)
    frames = list(it)
    assert frames == []


# ---------------------------------------------------------------------------
# Distillation integration
# ---------------------------------------------------------------------------


def test_distill_via_synthetic_iterator_produces_valid_codebook() -> None:
    """End-to-end: synthetic iterator + distill_codebook = valid codebook."""
    cfg = DistillationConfig(
        dataset_name="synthetic_test", random_seed=0xDA5C, max_frames=64
    )
    it = Comma2k19FrameIterator(synthetic=True, n_frames=64, seed=0xDA5C)
    book = distill_codebook(cfg, frames=iter(it))
    assert book is not None
    assert "license_tags" in book.metadata
    assert "synthetic-test-only" in book.metadata["license_tags"]


# ---------------------------------------------------------------------------
# Provenance bookkeeping
# ---------------------------------------------------------------------------


def test_real_mode_provenance_lists_visited_chunks(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "real_chunks"
    chunks_dir.mkdir()
    it = Comma2k19FrameIterator(chunks_dir=chunks_dir)
    list(it)
    assert it.provenance["synthetic"] is False
    assert it.provenance["license"] == "MIT"
    assert it.provenance["source_url"] == "https://github.com/commaai/comma2k19"
    assert it.provenance["chunks_dir"] == str(chunks_dir.resolve())


def test_max_chunks_and_frames_bound_provenance(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "x"
    chunks_dir.mkdir()
    it = Comma2k19FrameIterator(
        chunks_dir=chunks_dir,
        max_chunks=2,
        max_frames_per_chunk=10,
        frame_stride=2,
    )
    assert it.provenance["max_chunks"] == 2
    assert it.provenance["max_frames_per_chunk"] == 10
    assert it.provenance["frame_stride"] == 2
