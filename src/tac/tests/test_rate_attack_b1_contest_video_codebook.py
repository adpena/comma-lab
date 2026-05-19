# SPDX-License-Identifier: MIT
"""Tests for B1 contest-video-as-codebook Phase 1 probes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.contest_exploits import contest_video_codebook as b1


def _frame(value: int, *, shape: tuple[int, int, int] = (8, 8, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_decode_identity_flags_hevc_not_av1_and_missing_cuda(monkeypatch, tmp_path):
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake")

    monkeypatch.setattr(
        b1,
        "inspect_video_stream",
        lambda _path: {"codec_name": "hevc", "width": 1164, "height": 874},
    )
    monkeypatch.setattr(b1, "sha256_file", lambda _path: "v" * 64)
    monkeypatch.setattr(
        b1,
        "decode_rgb_frames",
        lambda *_args, **_kwargs: [_frame(7)],
    )

    report = b1.build_b1_decode_identity_probe(
        video_path=video,
        frame_count=1,
        enable_cuda_decode=False,
        generated_at_utc="2026-05-19T00:00:00+00:00",
    )

    assert report["directive_assumption_audit"]["source_video_codec_is_av1"] is False
    assert report["directive_assumption_audit"]["source_video_codec_is_hevc"] is True
    assert "directive_av1_premise_false_actual_codec_is_not_av1" in report["blockers"]
    assert "cuda_decode_not_attempted" in report["blockers"]
    assert report["authority"]["av1_bit_identity_authority"] is False
    assert report["authority"]["score_claim"] is False
    assert report["verdict"] == "DEFER"
    assert report["blocker_status"] == "blocking"
    assert report["attempts"]["cpu"]["evaluator_dataset"] == "AVVideoDataset"


def test_decode_identity_hevc_cpu_cuda_equal_has_blocking_partial_until_directive_renamed(
    monkeypatch, tmp_path
):
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake")

    monkeypatch.setattr(
        b1,
        "inspect_video_stream",
        lambda _path: {"codec_name": "hevc", "width": 16, "height": 16},
    )
    monkeypatch.setattr(b1, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(
        b1,
        "decode_rgb_frames",
        lambda *_args, **_kwargs: [_frame(11)],
    )

    report = b1.build_b1_decode_identity_probe(
        video_path=video,
        frame_count=1,
        enable_cuda_decode=True,
    )

    assert report["comparison"]["bit_identical"] is True
    assert report["authority"]["av1_bit_identity_authority"] is False
    assert report["authority"]["actual_codec_identity_authority"] is True
    assert report["authority"]["hevc_evaluator_identity_authority"] is True
    assert report["verdict"] == "PARTIAL"
    assert report["blocker_status"] == "blocking"
    assert report["probe_outcome_kwargs"]["metric_value"] == 1.0


def test_real_contest_video_stream_is_hevc_not_av1():
    repo = Path(__file__).resolve().parents[3]
    video = repo / "upstream/videos/0.mkv"
    if not video.exists():
        pytest.skip("contest video not present")

    info = b1.inspect_video_stream(video)

    assert info["codec_name"] == "hevc"
    assert info["pix_fmt"] == "yuv420p"
    assert info["width"] == 1164
    assert info["height"] == 874


def test_decode_identity_cuda_difference_is_not_authority(monkeypatch, tmp_path):
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake")

    monkeypatch.setattr(
        b1,
        "inspect_video_stream",
        lambda _path: {"codec_name": "hevc", "width": 16, "height": 16},
    )
    monkeypatch.setattr(b1, "sha256_file", lambda _path: "a" * 64)

    def _decode(*_args, **kwargs):
        return [_frame(12 if kwargs.get("hwaccel") == "cuda" else 11)]

    monkeypatch.setattr(b1, "decode_rgb_frames", _decode)
    report = b1.build_b1_decode_identity_probe(
        video_path=video,
        frame_count=1,
        enable_cuda_decode=True,
    )

    assert report["comparison"]["bit_identical"] is False
    assert report["comparison"]["max_abs_diff"] == 1
    assert "cpu_cuda_decode_not_bit_identical" in report["blockers"]
    assert report["authority"]["av1_bit_identity_authority"] is False
    assert report["blocker_status"] == "blocking"


def test_extract_regular_patches_is_deterministic_and_bounded():
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    patches_a = b1.extract_regular_patches(
        [frame],
        patch_size=4,
        stride=2,
        max_patches=5,
    )
    patches_b = b1.extract_regular_patches(
        [frame],
        patch_size=4,
        stride=2,
        max_patches=5,
    )

    assert patches_a.shape == (5, 4 * 4 * 3)
    assert np.array_equal(patches_a, patches_b)


def test_patch_density_self_density_does_not_become_frontier_authority():
    codebook = np.stack([np.zeros(12), np.ones(12) * 10]).astype(np.float32)
    query = np.stack([np.zeros(12), np.ones(12) * 10]).astype(np.float32)

    report = b1.build_patch_density_report_from_arrays(
        codebook_patches=codebook,
        query_patches=query,
        threshold_rmse=1.0,
        query_is_rendered_frontier=False,
        video_sha256="v" * 64,
        query_source_label="heldout_upstream",
    )

    assert report["dense_at_threshold"] is True
    assert report["authority"]["density_authority"] is False
    assert "query_source_not_rendered_frontier_output" in report["blockers"]
    assert report["verdict"] == "DEFER"
    assert report["blocker_status"] == "blocking"


def test_patch_density_rendered_frontier_dense_can_proceed_without_score_claim():
    codebook = np.stack([np.zeros(12), np.ones(12) * 10]).astype(np.float32)
    query = np.stack([np.zeros(12), np.ones(12) * 10]).astype(np.float32)

    report = b1.build_patch_density_report_from_arrays(
        codebook_patches=codebook,
        query_patches=query,
        threshold_rmse=1.0,
        query_is_rendered_frontier=True,
        video_sha256="v" * 64,
        query_source_label="inflated_frontier_raw",
    )

    assert report["authority"]["density_authority"] is True
    assert report["authority"]["score_claim"] is False
    assert report["verdict"] == "PROCEED"
    assert report["probe_outcome_kwargs"]["metric_name"] == "rmse_p50"


def test_b1_probe_tools_have_help():
    repo = Path(__file__).resolve().parents[3]
    for rel in (
        "tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py",
        "tools/probe_b1_patch_distribution_density.py",
    ):
        result = subprocess.run(
            [sys.executable, str(repo / rel), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "B1" in result.stdout
