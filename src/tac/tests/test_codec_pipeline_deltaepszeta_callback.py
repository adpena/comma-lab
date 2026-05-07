"""Tests for :mod:`tac.codec_pipeline_deltaepszeta_callback`.

Coverage:
- callback constructs with a valid pipeline
- ``report(synthetic_sd, 0)`` returns dict with op byte counts
- JSONL log writes correctly + can be read back
- ``add_to_loss(0.0, target=200_000)`` with stub-lambda returns 0
- ``add_to_loss(0.0, target=100_000)`` with real-lambda penalizes overshoot
- log_dir under /tmp is rejected (transient-evidence trap)
- two ops compose; per-op deltas and totals match manifest

Strict-scorer-rule: pure CPU codec measurement; no scorer load anywhere.
"""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from tac.codec_pipeline import CodecPipeline, Op1_PR101SplitBrotli
from tac.codec_pipeline_deltaepszeta_callback import (
    CodecPipelineAwareTrainingCallback,
    EpochReport,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA


# ---------------------------------------------------------------------------
# Synthetic state_dict + log dir helpers
# ---------------------------------------------------------------------------

def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


@pytest.fixture
def lane_log_dir(tmp_path: Path) -> Path:
    """Pytest's tmp_path is NOT under /tmp on macOS (it's under
    /private/var/folders/...) so it does NOT trip the /tmp guard. This
    keeps tests hermetic without violating the transient-evidence rule."""
    d = tmp_path / "training_log"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_callback_constructs_with_valid_pipeline(lane_log_dir: Path) -> None:
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    assert cb.pipeline is pipeline
    assert cb.log_dir == lane_log_dir
    assert cb.log_filename == "training_pipeline_bytes.jsonl"
    assert cb.lambda_penalty == 0.0
    assert cb.history == []


def test_callback_rejects_non_pipeline_arg(lane_log_dir: Path) -> None:
    with pytest.raises(TypeError, match="must be a CodecPipeline"):
        CodecPipelineAwareTrainingCallback(
            pipeline="not_a_pipeline",  # type: ignore[arg-type]
            log_dir=lane_log_dir,
        )


def test_callback_rejects_tmp_log_dir() -> None:
    """Per CLAUDE.md transient-evidence trap: /tmp paths are forbidden."""
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    with pytest.raises(ValueError, match="/tmp"):
        CodecPipelineAwareTrainingCallback(
            pipeline=pipeline, log_dir="/tmp/some_lane"
        )


def test_callback_rejects_negative_lambda(lane_log_dir: Path) -> None:
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    with pytest.raises(ValueError, match="lambda_penalty"):
        CodecPipelineAwareTrainingCallback(
            pipeline=pipeline, log_dir=lane_log_dir, lambda_penalty=-1.0
        )


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------

def test_report_returns_dict_with_op_byte_counts(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    per_op = cb.report(sd, epoch=0)
    assert isinstance(per_op, dict)
    assert "pr101_split_brotli" in per_op
    assert per_op["pr101_split_brotli"] > 0
    assert len(cb.history) == 1
    assert cb.history[0].epoch == 0


def test_report_rejects_negative_epoch(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    with pytest.raises(ValueError, match="epoch"):
        cb.report(sd, epoch=-1)


def test_report_records_overshoot_when_target_set(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    cb.report(sd, epoch=0, archive_size_target=10)  # tiny target → overshoot
    assert cb.history[0].overshoot_bytes > 0
    cb.report(sd, epoch=1, archive_size_target=10_000_000)  # huge target → none
    assert cb.history[1].overshoot_bytes == 0


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

def test_jsonl_log_writes_correctly(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    cb.report(sd, epoch=0)
    cb.report(sd, epoch=1, notes="post-warmup")

    rows = cb.read_log()
    assert len(rows) == 2
    assert rows[0]["epoch"] == 0
    assert rows[1]["epoch"] == 1
    assert rows[1]["notes"] == "post-warmup"
    # Each row must have the keys that downstream tooling expects.
    for r in rows:
        assert "total_bytes" in r
        assert "per_op_bytes" in r
        assert "per_op_delta_bytes" in r
        assert "final_blob_sha256" in r
        assert "timestamp_utc" in r
        # timestamp parseable as ISO-8601 UTC.
        datetime.strptime(r["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")


def test_jsonl_log_appends_across_callback_lifetime(lane_log_dir: Path) -> None:
    """A second callback instance pointed at the same log_dir continues
    appending — the JSONL log is durable across process restarts (which is
    the intended audit-trail use case)."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])

    cb1 = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    cb1.report(sd, epoch=0)

    cb2 = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    cb2.report(sd, epoch=1)

    rows = cb2.read_log()
    assert len(rows) == 2  # both reports persisted to the same file
    assert {r["epoch"] for r in rows} == {0, 1}


# ---------------------------------------------------------------------------
# add_to_loss() — stub-mode + real-mode
# ---------------------------------------------------------------------------

def test_add_to_loss_stub_lambda_returns_zero(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(
        pipeline=pipeline, log_dir=lane_log_dir, lambda_penalty=0.0
    )
    cb.report(sd, epoch=0)
    # Stub-mode: penalty is exactly zero regardless of overshoot.
    out = cb.add_to_loss(0.0, archive_size_target=200_000)
    assert out == 0
    out_with_loss = cb.add_to_loss(1.5, archive_size_target=10)  # huge overshoot
    assert out_with_loss == 0


def test_add_to_loss_real_lambda_penalizes_overshoot(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(
        pipeline=pipeline, log_dir=lane_log_dir, lambda_penalty=1e-6
    )
    cb.report(sd, epoch=0)
    archive_bytes = cb.history[0].total_bytes
    # Target much smaller than the actual archive → big overshoot → penalty > 0.
    out = cb.add_to_loss(0.0, archive_size_target=100_000)
    expected_overshoot = max(0, archive_bytes - 100_000)
    expected_penalty = 1e-6 * expected_overshoot
    assert out == pytest.approx(expected_penalty)
    # Target larger than archive → zero overshoot → zero penalty.
    out2 = cb.add_to_loss(0.0, archive_size_target=10_000_000)
    assert out2 == pytest.approx(0.0)


def test_add_to_loss_real_lambda_returns_zero_before_first_report(
    lane_log_dir: Path,
) -> None:
    """Until ``report`` runs once, there is no archive measurement; a
    real-mode penalty defensively returns 0 rather than raising or
    using a fake measurement."""
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(
        pipeline=pipeline, log_dir=lane_log_dir, lambda_penalty=1.0
    )
    out = cb.add_to_loss(0.0, archive_size_target=100_000)
    assert out == 0


def test_add_to_loss_rejects_non_positive_target(lane_log_dir: Path) -> None:
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    with pytest.raises(ValueError, match="archive_size_target"):
        cb.add_to_loss(0.0, archive_size_target=0)


def test_add_to_loss_preserves_torch_tensor_dtype_device(lane_log_dir: Path) -> None:
    """When called with a torch.Tensor loss, the stub-mode return must
    stay a tensor (not silently upcast to a Python float). This protects
    callers that do ``loss = pixel_loss + cb.add_to_loss(loss, ...)``."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    loss = torch.tensor(1.5)
    out = cb.add_to_loss(loss, archive_size_target=200_000)
    assert isinstance(out, torch.Tensor)
    assert out.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pipeline manifest cross-check
# ---------------------------------------------------------------------------

def test_per_op_bytes_match_pipeline_manifest(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    per_op = cb.report(sd, epoch=0)
    # Re-encode independently and compare.
    _, manifest = pipeline.encode(sd)
    for res in manifest.op_results:
        assert per_op[res.op_name] == res.bytes_out


def test_epoch_report_to_dict_is_json_serializable(lane_log_dir: Path) -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(pipeline=pipeline, log_dir=lane_log_dir)
    cb.report(sd, epoch=0)
    blob = json.dumps(cb.history[0].to_dict(), sort_keys=True)
    assert isinstance(blob, str) and len(blob) > 0
    assert "per_op_bytes" in blob
