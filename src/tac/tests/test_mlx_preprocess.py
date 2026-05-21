# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.mlx_preprocess import (
    CAMERA_HW,
    SEGNET_INPUT_HW,
    YUV6_INPUT_HW,
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    preprocess_scorer_inputs_from_pairs,
    write_scorer_input_cache,
    write_scorer_input_cache_hash_manifest_from_raw_file,
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
    assert saved["hash_domain"] == "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
    assert saved["producer_environment"]["numpy_version"] == np.__version__
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


def test_hash_only_manifest_matches_full_cache_hashes(tmp_path: Path) -> None:
    h, w = CAMERA_HW
    raw_path = tmp_path / "0.raw"
    frames = np.arange(4 * h * w * 3, dtype=np.uint32)
    frames = (frames % 251).astype(np.uint8).reshape(4, h, w, 3)
    raw_path.write_bytes(frames.tobytes())

    full_dir = tmp_path / "full"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--raw",
            str(raw_path),
            "--output-dir",
            str(full_dir),
            "--archive-sha256",
            "a" * 64,
            "--inflated-outputs-aggregate-sha256",
            "b" * 64,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    hash_manifest_path = tmp_path / "hash" / "manifest.json"
    hash_manifest = write_scorer_input_cache_hash_manifest_from_raw_file(
        raw_path,
        hash_manifest_path,
        archive_sha256="a" * 64,
        inflated_outputs_aggregate_sha256="b" * 64,
        batch_pairs=1,
    )
    full_manifest = json.loads((full_dir / "manifest.json").read_text(encoding="utf-8"))
    saved_hash_manifest = json.loads(hash_manifest_path.read_text(encoding="utf-8"))

    assert hash_manifest["hash_only"] is True
    assert hash_manifest["hash_domain"] == "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
    assert hash_manifest["producer_environment"]["numpy_version"] == np.__version__
    assert saved_hash_manifest["artifacts"] == {}
    assert hash_manifest["array_sha256"] == full_manifest["array_sha256"]


def test_hash_only_cli_writes_no_tensor_payloads(tmp_path: Path) -> None:
    h, w = CAMERA_HW
    raw_path = tmp_path / "0.raw"
    frames = np.zeros((2, h, w, 3), dtype=np.uint8)
    raw_path.write_bytes(frames.tobytes())
    out_dir = tmp_path / "hash_cli"

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
            "--hash-only",
            "--batch-pairs",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"pair_count": 1' in completed.stdout
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hash_only"] is True
    assert not (out_dir / "segnet_last_rgb.npy").exists()
    assert not (out_dir / "posenet_yuv6_pair.npy").exists()
    assert len(manifest["array_sha256"]["posenet_yuv6_pair"]) == 64


def test_video_cli_smoke_on_upstream_video_with_pair_cap(tmp_path: Path) -> None:
    video_path = REPO / "upstream" / "videos" / "0.mkv"
    if not video_path.exists():
        pytest.skip("upstream video fixture is not available")
    out_dir = tmp_path / "video_cache"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--video",
            str(video_path),
            "--output-dir",
            str(out_dir),
            "--max-pairs",
            "1",
            "--batch-pairs",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"pair_count": 1' in completed.stdout
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "video"
    assert manifest["pair_count"] == 1
    assert len(manifest["source_video_sha256"]) == 64
    assert manifest["segnet_last_rgb_shape"] == [1, 3, *SEGNET_INPUT_HW]
    assert manifest["posenet_yuv6_pair_shape"] == [1, 12, *YUV6_INPUT_HW]


def test_full_cache_cli_requires_ack_for_large_eager_tensor_surface(tmp_path: Path) -> None:
    h, w = CAMERA_HW
    raw_path = tmp_path / "0.raw"
    frames = np.zeros((4, h, w, 3), dtype=np.uint8)
    raw_path.write_bytes(frames.tobytes())

    blocked = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--raw",
            str(raw_path),
            "--output-dir",
            str(tmp_path / "blocked"),
            "--large-cache-pair-threshold",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert blocked.returncode != 0
    assert "refusing eager full MLX scorer-input tensor cache" in blocked.stderr

    allowed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--raw",
            str(raw_path),
            "--output-dir",
            str(tmp_path / "allowed"),
            "--large-cache-pair-threshold",
            "1",
            "--allow-large-tensor-cache",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"pair_count": 2' in allowed.stdout


def test_full_cache_cli_rejects_negative_max_pairs(tmp_path: Path) -> None:
    h, w = CAMERA_HW
    raw_path = tmp_path / "0.raw"
    frames = np.zeros((2, h, w, 3), dtype=np.uint8)
    raw_path.write_bytes(frames.tobytes())

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--raw",
            str(raw_path),
            "--output-dir",
            str(tmp_path / "cache"),
            "--max-pairs",
            "-1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--max-pairs must be >= 1" in completed.stderr


def test_hash_only_cli_rejects_nonpositive_batch_pairs(tmp_path: Path) -> None:
    h, w = CAMERA_HW
    raw_path = tmp_path / "0.raw"
    frames = np.zeros((2, h, w, 3), dtype=np.uint8)
    raw_path.write_bytes(frames.tobytes())

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_scorer_input_cache.py"),
            "--raw",
            str(raw_path),
            "--output-dir",
            str(tmp_path / "cache"),
            "--hash-only",
            "--batch-pairs",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--batch-pairs must be >= 1" in completed.stderr


def test_contest_auth_eval_hash_artifact_updates_provenance(tmp_path: Path) -> None:
    from experiments.contest_auth_eval import _record_scorer_input_cache_hash_artifact

    h, w = CAMERA_HW
    inflated = tmp_path / "inflated"
    inflated.mkdir()
    raw_path = inflated / "0.raw"
    frames = np.zeros((2, h, w, 3), dtype=np.uint8)
    frames[1, ...] = 255
    raw_path.write_bytes(frames.tobytes())
    video_names = tmp_path / "videos.txt"
    video_names.write_text("0.mkv\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "provenance.json").write_text("{}\n", encoding="utf-8")

    prov = {"archive_sha256": "a" * 64}
    manifest = _record_scorer_input_cache_hash_artifact(
        prov,
        work_dir,
        inflated,
        video_names,
        {"aggregate_sha256": "b" * 64},
        Path("scorer_input_cache_hashes.json"),
        batch_pairs=1,
    )

    assert manifest["pair_count"] == 1
    assert manifest["archive_sha256"] == "a" * 64
    assert manifest["inflated_outputs_aggregate_sha256"] == "b" * 64
    assert manifest["video_name"] == "0.mkv"
    saved = json.loads((work_dir / "scorer_input_cache_hashes.json").read_text())
    assert saved["hash_only"] is True
    provenance = json.loads((work_dir / "provenance.json").read_text())
    assert provenance["scorer_input_cache_hash_manifest"]["payload"]["video_name"] == "0.mkv"
