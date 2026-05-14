# SPDX-License-Identifier: MIT
"""Tests for log_incremental_distillation_streaming (operator pivot 2026-05-14).

Streaming-mode companion to test_log_incremental_distillation. Uses
synthetic-mode Comma2k19LocalStreamer (no real network).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.substrates.pretrained_driving_prior import (
    Comma2k19LocalStreamer,
    DistillationConfig,
    LogIncrementalSchedule,
    log_incremental_distillation_streaming,
    replay_stream_log,
)


# ---------------------------------------------------------------------------
# Smoke: 1-chunk schedule end-to-end
# ---------------------------------------------------------------------------


def test_streaming_one_chunk_smoke(tmp_path: Path) -> None:
    streamer = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=1,
        synthetic_chunk_size_bytes=1024,
    )
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=1,
        quality_plateau_threshold=0.0,
        max_steps=1,
    )
    book, log = log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32
        ),
        frames_per_chunk=16,
    )
    assert book is not None
    assert len(log) == 1
    assert log[0].chunk_count == 1
    assert book.metadata.get("streaming_mode") is True


# ---------------------------------------------------------------------------
# Multi-step schedule
# ---------------------------------------------------------------------------


def test_streaming_doubling_schedule_runs_to_completion(tmp_path: Path) -> None:
    streamer = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=8,
        synthetic_chunk_size_bytes=512,
    )
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=8,
        quality_plateau_threshold=0.0,  # disable plateau early-stop
        max_steps=6,
    )
    book, log = log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32
        ),
        frames_per_chunk=16,
    )
    assert book is not None
    # Doubling schedule with 8 max chunks → [1, 2, 4, 8] = 4 steps
    assert len(log) == 4
    assert [step.chunk_count for step in log] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Schedule errors
# ---------------------------------------------------------------------------


def test_streaming_refuses_empty_chunk_list(tmp_path: Path) -> None:
    streamer = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=0,
    )
    schedule = LogIncrementalSchedule(initial_chunks=1, max_chunks=1)
    with pytest.raises(ValueError, match="has no chunk ids"):
        log_incremental_distillation_streaming(
            streamer=streamer,
            schedule=schedule,
        )


def test_streaming_with_explicit_chunk_ids(tmp_path: Path) -> None:
    streamer = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=4,
    )
    # Pass an explicit subset of chunk ids.
    explicit = ["synthetic_chunk_0000", "synthetic_chunk_0002"]
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=2,
        quality_plateau_threshold=0.0,
        max_steps=2,
    )
    book, log = log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        chunk_ids=explicit,
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32
        ),
        frames_per_chunk=16,
    )
    assert book is not None
    last_window = log[-1].provenance.get("streamed_chunk_ids_used", [])
    assert last_window == explicit


# ---------------------------------------------------------------------------
# JSONL access log integration
# ---------------------------------------------------------------------------


def test_streaming_appends_jsonl_per_chunk(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    streamer = Comma2k19LocalStreamer(
        log_dir=log_dir,
        synthetic=True,
        synthetic_n_chunks=2,
    )
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=2,
        quality_plateau_threshold=0.0,
        max_steps=2,
    )
    log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32
        ),
        frames_per_chunk=8,
    )
    from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
        _log_file_for_today,
    )

    rows = list(replay_stream_log(_log_file_for_today(log_dir)))
    # step 0: 1 chunk × 2 rows (chunk_fetch + frame_decode) = 2 rows
    # step 1: 2 chunks × 2 rows = 4 rows
    # total = 6 rows
    assert len(rows) == 6
    chunks_seen = {r["chunk_id"] for r in rows}
    assert chunks_seen == {"synthetic_chunk_0000", "synthetic_chunk_0001"}


# ---------------------------------------------------------------------------
# Provenance fields
# ---------------------------------------------------------------------------


def test_streaming_provenance_carries_streamer_metadata(tmp_path: Path) -> None:
    streamer = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=1,
    )
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=1,
        quality_plateau_threshold=0.0,
        max_steps=1,
    )
    _, log = log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32
        ),
        frames_per_chunk=8,
    )
    prov = log[0].provenance
    assert "streamer_source_url" in prov
    assert "streamer_is_synthetic" in prov
    assert "streamer_ram_buffer_gb" in prov
    assert "streamed_chunk_ids_used" in prov
    assert "frames_streamed" in prov
    assert prov["streamer_is_synthetic"] is True


# ---------------------------------------------------------------------------
# Plateau early-stop
# ---------------------------------------------------------------------------


def test_streaming_plateau_early_stop_engages(tmp_path: Path) -> None:
    """With a high plateau threshold, schedule stops early."""
    streamer = Comma2k19LocalStreamer(
        log_dir=tmp_path / "logs",
        synthetic=True,
        synthetic_n_chunks=8,
    )
    # Very high plateau threshold → guaranteed early-stop after step 2.
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=8,
        quality_plateau_threshold=1.0,
        max_steps=6,
    )

    def constant_quality_metric(_book) -> float:
        return 0.5  # constant quality → marginal = 0 → triggers plateau

    book, log = log_incremental_distillation_streaming(
        streamer=streamer,
        schedule=schedule,
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32
        ),
        quality_metric=constant_quality_metric,
        frames_per_chunk=8,
    )
    # Early-stop fires at step 2 (need step >= 2 + marginal < threshold)
    assert any(step.early_stopped for step in log)
    assert len(log) <= 3
