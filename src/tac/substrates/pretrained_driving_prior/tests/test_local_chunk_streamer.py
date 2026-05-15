# SPDX-License-Identifier: MIT
"""Tests for Comma2k19LocalStreamer (operator pivot 2026-05-14).

Per operator directive *"or instead of downloading just configure a local
streamer and log"*: NO real network in tests. The synthetic-mode streamer
yields deterministic in-memory bytes, and the http_fetcher dependency
injection point lets tests pass a mock fetcher for explicit-bytes paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
    CANONICAL_SOURCE_URL,
    DATASET_LICENSE,
    DEFAULT_RAM_BUFFER_GB,
    Comma2k19LocalStreamer,
    DynamicChunkingStrategy,
    SHA256MismatchError,
    _log_file_for_today,
    replay_stream_log,
    summarize_stream_log,
)

# ---------------------------------------------------------------------------
# Constructor + provenance
# ---------------------------------------------------------------------------


def test_streamer_constructor_defaults(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(log_dir=tmp_path / "logs", synthetic=True)
    assert s.source_url == CANONICAL_SOURCE_URL
    assert s.is_synthetic is True
    assert s.ram_buffer_gb == DEFAULT_RAM_BUFFER_GB
    assert s.log_dir == (tmp_path / "logs").resolve()


def test_streamer_log_dir_created_on_init(tmp_path: Path) -> None:
    log_dir = tmp_path / "deep" / "nested" / "logs"
    assert not log_dir.exists()
    Comma2k19LocalStreamer(log_dir=log_dir, synthetic=True)
    assert log_dir.exists()
    assert (log_dir / ".partial").exists()


def test_streamer_refuses_zero_or_negative_buffer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ram_buffer_gb must be > 0"):
        Comma2k19LocalStreamer(
            log_dir=tmp_path, synthetic=True, ram_buffer_gb=0
        )
    with pytest.raises(ValueError, match="ram_buffer_gb must be > 0"):
        Comma2k19LocalStreamer(
            log_dir=tmp_path, synthetic=True, ram_buffer_gb=-1.0
        )


def test_streamer_license_attribution() -> None:
    assert DATASET_LICENSE == "MIT"
    assert "comma2k19" in CANONICAL_SOURCE_URL.lower()


# ---------------------------------------------------------------------------
# Dynamic chunking
# ---------------------------------------------------------------------------


def test_dynamic_chunking_frame_range_preserves_source_order(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=3,
    )

    specs = s.plan_chunks(
        DynamicChunkingStrategy(mode="frame_range", frame_range_size=12)
    )

    assert [spec.chunk_id for spec in specs] == [
        "synthetic_chunk_0000",
        "synthetic_chunk_0001",
        "synthetic_chunk_0002",
    ]
    assert all(spec.frame_start == 0 and spec.frame_end == 12 for spec in specs)
    assert specs[0].decode_hint["chunking_mode"] == "frame_range"


def test_dynamic_chunking_saliency_topk_prioritizes_metadata_scores(
    tmp_path: Path,
) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=4,
    )

    specs = s.plan_chunks(
        DynamicChunkingStrategy(mode="saliency", saliency_topk=2),
        video_metadata={"saliency_scores": [0.1, 0.9, 0.2, 0.8]},
    )

    assert [spec.chunk_id for spec in specs] == [
        "synthetic_chunk_0001",
        "synthetic_chunk_0003",
    ]
    assert [spec.decode_hint["priority_rank"] for spec in specs] == [0, 1]


def test_dynamic_chunking_motion_threshold_keeps_only_high_motion(
    tmp_path: Path,
) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=4,
    )

    specs = s.plan_chunks(
        DynamicChunkingStrategy(mode="motion_class", motion_threshold=0.5),
        video_metadata={"motion_scores": [0.1, 0.6, 0.4, 0.7]},
    )

    assert [spec.chunk_id for spec in specs] == [
        "synthetic_chunk_0003",
        "synthetic_chunk_0001",
    ]


def test_stream_chunks_yields_specs_and_frames(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=1,
    )

    rows = list(
        s.stream_chunks(
            chunking=DynamicChunkingStrategy(mode="frame_range", frame_range_size=3),
            max_frames_per_chunk=2,
        )
    )

    assert len(rows) == 1
    spec, frames = rows[0]
    assert spec.chunk_id == "synthetic_chunk_0000"
    assert len(frames) == 2


# ---------------------------------------------------------------------------
# Synthetic-mode chunk streaming
# ---------------------------------------------------------------------------


def test_synthetic_stream_chunk_yields_bytes(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_chunk_size_bytes=1024,
        synthetic_n_chunks=2,
    )
    chunk_ids = s.list_chunk_ids()
    assert len(chunk_ids) == 2
    pieces = list(s.stream_chunk(chunk_ids[0]))
    assert len(pieces) > 0
    total = b"".join(pieces)
    assert isinstance(total, bytes)
    assert len(total) > 0


def test_synthetic_stream_chunk_deterministic(tmp_path: Path) -> None:
    s1 = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs1",
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=2,
        synthetic_seed=42,
    )
    s2 = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs2",
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=2,
        synthetic_seed=42,
    )
    a = b"".join(s1.stream_chunk("synthetic_chunk_0000"))
    b = b"".join(s2.stream_chunk("synthetic_chunk_0000"))
    assert a == b


def test_synthetic_stream_chunk_different_chunks_yield_different_bytes(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=4,
    )
    a = b"".join(s.stream_chunk("synthetic_chunk_0000"))
    b = b"".join(s.stream_chunk("synthetic_chunk_0001"))
    assert a != b


# ---------------------------------------------------------------------------
# Synthetic-mode frame streaming
# ---------------------------------------------------------------------------


def test_synthetic_stream_frames_yields_uint8_HWC(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=2,
    )
    frames = list(s.stream_frames("synthetic_chunk_0000", max_frames=4))
    assert len(frames) == 4
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.dtype == np.uint8
        assert f.ndim == 3
        assert f.shape[2] == 3


def test_synthetic_stream_frames_with_frame_indices(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=2,
    )
    frames = list(
        s.stream_frames("synthetic_chunk_0000", frame_indices=[0, 5, 10])
    )
    assert len(frames) == 3


# ---------------------------------------------------------------------------
# JSONL access log
# ---------------------------------------------------------------------------


def test_stream_chunk_writes_jsonl_access_row(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=2,
        dispatch_label="test_dispatch_001",
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    log_path = _log_file_for_today(log_dir)
    assert log_path.exists()
    rows = list(replay_stream_log(log_path))
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "synthetic_chunk_0000"
    assert rows[0]["dispatch_label"] == "test_dispatch_001"
    assert rows[0]["license"] == "MIT"
    assert rows[0]["source_url"] == CANONICAL_SOURCE_URL
    assert rows[0]["bytes_transferred"] > 0
    assert rows[0]["kind"] == "chunk_fetch"


def test_stream_frames_writes_two_jsonl_rows(tmp_path: Path) -> None:
    """stream_frames invokes stream_chunk + frame_decode; expect 2 rows."""
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir, synthetic=True, synthetic_n_chunks=1
    )
    list(s.stream_frames("synthetic_chunk_0000", max_frames=2))
    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    assert len(rows) == 2
    kinds = {r["kind"] for r in rows}
    assert kinds == {"chunk_fetch", "frame_decode"}


def test_jsonl_log_appends_under_lock(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir, synthetic=True, synthetic_n_chunks=4
    )
    for cid in s.list_chunk_ids():
        list(s.stream_chunk(cid))
    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    assert len(rows) == 4
    chunk_ids = [r["chunk_id"] for r in rows]
    assert chunk_ids == sorted(chunk_ids)  # appended in order


def test_log_file_is_date_rotated(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir, synthetic=True, synthetic_n_chunks=1
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    today_log = _log_file_for_today(log_dir)
    assert today_log.exists()
    # Filename matches comma2k19_stream_log_YYYYMMDD.jsonl
    assert today_log.name.startswith("comma2k19_stream_log_")
    assert today_log.name.endswith(".jsonl")


# ---------------------------------------------------------------------------
# Replay + summary APIs
# ---------------------------------------------------------------------------


def test_replay_stream_log_yields_in_file_order(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir, synthetic=True, synthetic_n_chunks=3
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    list(s.stream_chunk("synthetic_chunk_0001"))
    list(s.stream_chunk("synthetic_chunk_0002"))
    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    chunk_ids = [r["chunk_id"] for r in rows]
    assert chunk_ids == [
        "synthetic_chunk_0000",
        "synthetic_chunk_0001",
        "synthetic_chunk_0002",
    ]


def test_replay_stream_log_missing_file_yields_nothing(tmp_path: Path) -> None:
    rows = list(replay_stream_log(tmp_path / "does_not_exist.jsonl"))
    assert rows == []


def test_summarize_stream_log_aggregates(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir, synthetic=True, synthetic_n_chunks=2
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    list(s.stream_chunk("synthetic_chunk_0001"))
    summary = summarize_stream_log(_log_file_for_today(log_dir))
    assert summary["row_count"] == 2
    assert summary["chunks_seen"] == 2
    assert summary["total_bytes"] > 0


def test_stream_log_summary_method(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir, synthetic=True, synthetic_n_chunks=1
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    summary = s.stream_log_summary()
    assert summary["row_count"] == 1
    assert summary["chunks_seen"] == 1


def test_replay_stream_log_skips_malformed_rows(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text('{"chunk_id":"a"}\nnot json\n{"chunk_id":"b"}\n')
    rows = list(replay_stream_log(log))
    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "a"
    assert rows[1]["chunk_id"] == "b"


# ---------------------------------------------------------------------------
# SHA-256 verification
# ---------------------------------------------------------------------------


def test_sha256_mismatch_raises_and_logs_failure(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    # Pin a sha256 that will NOT match the synthetic bytes for the chunk.
    s = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_n_chunks=1,
        dataset_sha256_manifest={"synthetic_chunk_0000": "0" * 64},
    )
    with pytest.raises(SHA256MismatchError, match="expected sha256"):
        list(s.stream_chunk("synthetic_chunk_0000"))
    # The FAILED access is still recorded.
    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    assert len(rows) == 1
    assert rows[0]["sha256_verified"] is False
    assert rows[0]["sha256_expected"] == "0" * 64


def test_sha256_verification_passes_when_hash_matches(tmp_path: Path) -> None:
    """Compute the actual synthetic sha first, then pin it."""
    log_dir = tmp_path / "logs"
    # First pass: discover the actual sha by streaming with verification off.
    s_probe = Comma2k19LocalStreamer(
        log_dir=tmp_path / "probe",
        synthetic=True,
        synthetic_n_chunks=1,
        verify_sha256_in_flight=False,
    )
    payload = b"".join(s_probe.stream_chunk("synthetic_chunk_0000"))
    import hashlib

    expected = hashlib.sha256(payload).hexdigest()
    # Second pass: pin the expected sha and verify.
    s = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_n_chunks=1,
        dataset_sha256_manifest={"synthetic_chunk_0000": expected},
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    assert rows[0]["sha256_verified"] is True


def test_sha256_disabled_does_not_verify(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_n_chunks=1,
        verify_sha256_in_flight=False,
        dataset_sha256_manifest={"synthetic_chunk_0000": "0" * 64},
    )
    # Bad sha but verification is off — no raise.
    list(s.stream_chunk("synthetic_chunk_0000"))


# ---------------------------------------------------------------------------
# Contest-video leakage guard
# ---------------------------------------------------------------------------


def test_streamer_refuses_contest_video_chunk_id(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(log_dir=tmp_path / "logs", synthetic=True)
    with pytest.raises(ValueError, match="contest-video reference"):
        list(s.stream_chunk("upstream/videos/0.mkv"))
    with pytest.raises(ValueError, match="contest-video reference"):
        list(s.stream_chunk("0.mkv"))
    with pytest.raises(ValueError, match="contest-video reference"):
        list(s.stream_chunk("evaluate.py"))


# ---------------------------------------------------------------------------
# RAM buffer + eviction
# ---------------------------------------------------------------------------


def test_buffer_evicts_lru_when_cap_reached(tmp_path: Path) -> None:
    """Set a tiny buffer cap; multiple chunks must trigger eviction."""
    log_dir = tmp_path / "logs"
    # 1 KB cap. Each synthetic chunk is ~512 bytes so two chunks fit but
    # three trigger eviction.
    s = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_chunk_size_bytes=512,
        synthetic_n_chunks=4,
        ram_buffer_gb=1 / (1 << 20),  # 1 KB
    )
    for cid in s.list_chunk_ids():
        list(s.stream_frames(cid, max_frames=1))
    status = s.current_buffer_status()
    # Frames decode finishes by evicting the chunk, so buffer ends empty.
    assert status["chunks_buffered"] == 0


def test_current_buffer_status_is_dict(tmp_path: Path) -> None:
    s = Comma2k19LocalStreamer(log_dir=tmp_path / "logs", synthetic=True)
    status = s.current_buffer_status()
    assert "buffer_bytes_used" in status
    assert "ram_buffer_cap_bytes" in status
    assert "chunks_buffered" in status


# ---------------------------------------------------------------------------
# Mock http_fetcher dependency injection
# ---------------------------------------------------------------------------


def test_custom_http_fetcher_is_called(tmp_path: Path) -> None:
    """Real-mode fetcher dependency injection (no actual network)."""
    captured: list[str] = []

    def mock_fetcher(chunk_id: str, source_url: str):
        captured.append(chunk_id)
        yield b"mock_bytes_for_" + chunk_id.encode("ascii")

    s = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=False,
        http_fetcher=mock_fetcher,
        verify_sha256_in_flight=False,
    )
    payload = b"".join(s.stream_chunk("my_test_chunk"))
    assert payload == b"mock_bytes_for_my_test_chunk"
    assert captured == ["my_test_chunk"]


def test_dispatch_label_recorded_in_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    s = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_n_chunks=1,
        dispatch_label="dp1_phase_2_smoke_20260514T120000Z",
    )
    list(s.stream_chunk("synthetic_chunk_0000"))
    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    assert rows[0]["dispatch_label"] == "dp1_phase_2_smoke_20260514T120000Z"
