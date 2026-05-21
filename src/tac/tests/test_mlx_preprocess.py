# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.local_acceleration.mlx_preprocess import (
    CAMERA_HW,
    SEGNET_INPUT_HW,
    YUV6_INPUT_HW,
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    preprocess_scorer_inputs_from_pairs,
    write_scorer_input_cache,
)

REPO = Path(__file__).resolve().parents[3]


def test_non_overlapping_pair_indices_match_upstream_seq_len_two() -> None:
    pairs = non_overlapping_pair_indices(5)
    np.testing.assert_array_equal(pairs, np.array([[0, 1], [2, 3]], dtype=np.int64))


def test_preprocess_uses_last_frame_for_segnet_and_both_frames_for_posenet() -> None:
    pair = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
    pair[:, 1, ...] = 255
    batch = preprocess_scorer_inputs_from_pairs(pair, pair_indices=np.array([[10, 11]]))

    assert batch.segnet_last_rgb.shape == (1, 3, *SEGNET_INPUT_HW)
    assert batch.posenet_yuv6_pair.shape == (1, 12, *YUV6_INPUT_HW)
    assert batch.pair_indices.tolist() == [[10, 11]]
    np.testing.assert_allclose(batch.segnet_last_rgb, 255.0, atol=1e-4, rtol=0)

    first_frame_y = batch.posenet_yuv6_pair[:, 0:4]
    first_frame_uv = batch.posenet_yuv6_pair[:, 4:6]
    second_frame_y = batch.posenet_yuv6_pair[:, 6:10]
    second_frame_uv = batch.posenet_yuv6_pair[:, 10:12]
    np.testing.assert_allclose(first_frame_y, 0.0, atol=1e-4, rtol=0)
    np.testing.assert_allclose(first_frame_uv, 128.0, atol=1e-4, rtol=0)
    np.testing.assert_allclose(second_frame_y, 255.0, atol=1e-4, rtol=0)
    np.testing.assert_allclose(second_frame_uv, 128.0, atol=1e-4, rtol=0)


def test_write_scorer_input_cache_is_non_authoritative(tmp_path: Path) -> None:
    pair = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
    pair[:, 1, ...] = 255
    batch = preprocess_scorer_inputs_from_pairs(pair)

    manifest = write_scorer_input_cache(
        batch,
        tmp_path,
        archive_sha256="a" * 64,
        inflated_outputs_aggregate_sha256="b" * 64,
        raw_sha256="c" * 64,
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert (tmp_path / "segnet_last_rgb.npy").exists()
    assert (tmp_path / "posenet_yuv6_pair.npy").exists()
    assert (tmp_path / "pair_indices.npy").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["archive_sha256"] == "a" * 64
    assert len(saved["array_sha256"]["segnet_last_rgb"]) == 64


def test_raw_memmap_and_cli_smoke_on_default_contest_shape(tmp_path: Path) -> None:
    h, w = CAMERA_HW
    raw_path = tmp_path / "0.raw"
    frames = np.zeros((2, h, w, 3), dtype=np.uint8)
    frames[1, ...] = 255
    raw_path.write_bytes(frames.tobytes())

    mm = load_raw_video_memmap(raw_path)
    assert mm.shape == (2, h, w, 3)

    out_dir = tmp_path / "cache"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--raw",
            str(raw_path),
            "--output-dir",
            str(out_dir),
            "--archive-sha256",
            "a" * 64,
            "--inflated-outputs-aggregate-sha256",
            "b" * 64,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"pair_count": 1' in completed.stdout
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 1
    assert manifest["segnet_last_rgb_shape"] == [1, 3, *SEGNET_INPUT_HW]
    assert manifest["posenet_yuv6_pair_shape"] == [1, 12, *YUV6_INPUT_HW]
