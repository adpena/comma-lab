# SPDX-License-Identifier: MIT
"""Regression coverage for SNeRV archive-backed MLX prefilter profiling."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest


def test_read_snerv_packet_from_archive_zip(tmp_path: Path) -> None:
    from tools.profile_snerv_archive_mlx_prefilter import (
        read_snerv_packet_from_archive_or_raw,
    )

    archive = tmp_path / "archive.zip"
    packet = b"SNAR1receiver-packet"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", packet)
        zf.writestr("inflate.sh", "#!/bin/sh\n")

    out, info = read_snerv_packet_from_archive_or_raw(archive)

    assert out == packet
    assert info["input_kind"] == "archive_zip"
    assert info["archive_member"] == "0.bin"
    assert info["packet_bytes"] == len(packet)


def test_read_snerv_packet_auto_detects_receiver_proof_member(tmp_path: Path) -> None:
    from tools.profile_snerv_archive_mlx_prefilter import (
        read_snerv_packet_from_archive_or_raw,
    )

    archive = tmp_path / "archive.zip"
    packet = b"SNAR2receiver-packet"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("inflate.py", "not a packet")
        zf.writestr("x", packet)

    out, info = read_snerv_packet_from_archive_or_raw(archive)

    assert out == packet
    assert info["archive_member_requested"] == "0.bin"
    assert info["archive_member"] == "x"


def test_compare_prefilter_profiles_splits_frame1_seg_and_pair_pose() -> None:
    from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
    from tools.profile_snerv_archive_mlx_prefilter import compare_prefilter_profiles

    baseline = {
        "archive_bytes": 100,
        "score_components": {
            "avg_segnet_dist": 0.10,
            "avg_posenet_dist": 4.0,
            "seg_term": 10.0,
            "pose_term": (10.0 * 4.0) ** 0.5,
            "canonical_score": 16.0 + 25.0 * 100.0 / ORIGINAL_VIDEO_BYTES,
        },
    }
    candidate = {
        "archive_bytes": 200,
        "score_components": {
            "avg_segnet_dist": 0.09,
            "avg_posenet_dist": 3.61,
            "seg_term": 9.0,
            "pose_term": (10.0 * 3.61) ** 0.5,
            "canonical_score": 15.0083275543 + 25.0 * 200.0 / ORIGINAL_VIDEO_BYTES,
        },
    }

    comparison = compare_prefilter_profiles(
        baseline_profile=baseline,
        candidate_profile=candidate,
        required_non_rate_drop=0.5,
    )

    deltas = comparison["deltas_candidate_minus_baseline"]
    assert deltas["archive_bytes"] == 100
    assert deltas["frame1_segnet_avg_dist"] == pytest.approx(-0.01)
    assert deltas["two_frame_posenet_avg_dist"] == pytest.approx(-0.39)
    assert deltas["non_rate_score"] < 0.0
    assert comparison["admission"]["observed_non_rate_drop"] > 0.5
    assert comparison["admission"]["passes_required_drop"] is True
    assert comparison["score_claim"] is False
    assert comparison["contest_lagrangian"]["segnet_domain"] == "pair frame 1 only"
    assert comparison["contest_lagrangian"]["posenet_domain"] == (
        "both frames through YUV6 pair input"
    )


def test_build_snerv_archive_prefilter_bundle_uses_receiver_frames(monkeypatch) -> None:
    mx = pytest.importorskip("mlx.core")
    import tools.profile_snerv_archive_mlx_prefilter as tool

    frames = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 2, 3, 4, 5)

    def fake_decode(packet: bytes) -> np.ndarray:
        assert packet == b"SNAR1fake"
        return frames

    def fake_targets(
        video_path: object,
        *,
        num_pairs: int,
        output_height: int,
        output_width: int,
    ):
        assert str(video_path) == "video.mkv"
        assert num_pairs == 2
        assert (output_height, output_width) == (4, 5)
        return (
            mx.zeros((2, 4, 5, 3), dtype=mx.float32),
            mx.zeros((2, 4, 5, 3), dtype=mx.float32),
        )

    monkeypatch.setattr(tool, "decode_snerv_archive_frames", fake_decode)
    monkeypatch.setattr(tool, "decode_mlx_targets", fake_targets)

    bundle, summary = tool.build_snerv_archive_prefilter_bundle(
        packet=b"SNAR1fake",
        source_video_path="video.mkv",
    )
    out = np.asarray(bundle.model(mx.array([1], dtype=mx.int32)))

    assert summary["decoded_receiver_frame_shape"] == [2, 2, 3, 4, 5]
    assert bundle.forward_convention == "call_b2chw_255"
    assert out.shape == (1, 2, 3, 4, 5)
    np.testing.assert_array_equal(out[0], frames[1])


def test_profile_manifest_paths_are_json_serializable(tmp_path: Path) -> None:
    from tools.profile_snerv_archive_mlx_prefilter import compare_prefilter_profiles

    baseline = {
        "archive_bytes": 100,
        "score_components": {
            "avg_segnet_dist": 0.0,
            "avg_posenet_dist": 0.0,
            "seg_term": 0.0,
            "pose_term": 0.0,
            "canonical_score": 0.0,
        },
    }
    candidate = {
        "archive_bytes": 100,
        "score_components": {
            "avg_segnet_dist": 0.0,
            "avg_posenet_dist": 0.0,
            "seg_term": 0.0,
            "pose_term": 0.0,
            "canonical_score": 0.0,
        },
    }
    path = tmp_path / "comparison.json"
    path.write_text(
        json.dumps(
            compare_prefilter_profiles(
                baseline_profile=baseline,
                candidate_profile=candidate,
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == (
        "snerv_skip_high_prefilter_profile_comparison.v1"
    )
