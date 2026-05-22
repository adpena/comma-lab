# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.local_acceleration.mlx_preprocess import ScorerInputBatch, write_scorer_input_cache
from tac.local_acceleration.mlx_scorer_response import (
    BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER,
    GPU_RESEARCH_SIGNAL_BLOCKER,
    build_mlx_scorer_response_payload,
    load_scorer_input_cache,
)

REPO = Path(__file__).resolve().parents[3]


def test_mlx_scorer_response_cache_cli_is_non_authoritative(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)
    output = tmp_path / "mlx_response.json"
    archive_size_bytes = 1000

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            str(archive_size_bytes),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_rate_score = 25.0 * archive_size_bytes / ORIGINAL_VIDEO_BYTES
    assert stdout["score_claim"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["n_samples"] == 1
    assert payload["avg_posenet_dist"] == 0.0
    assert payload["avg_segnet_dist"] == 0.0
    assert abs(payload["canonical_score"] - expected_rate_score) < 1.0e-12
    assert payload["cache_identity"]["pair_indices_equal"] is True


def test_load_scorer_input_cache_rejects_hash_only_manifest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hash_only"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps({"hash_only": True}) + "\n",
        encoding="utf-8",
    )

    try:
        load_scorer_input_cache(cache_dir)
    except ValueError as exc:
        assert "hash-only" in str(exc)
    else:
        raise AssertionError("hash-only cache was accepted")


def test_mlx_scorer_response_cli_can_score_deterministic_pair_window(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    seg = np.zeros((2, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((2, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)
    output = tmp_path / "mlx_response_window.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1000",
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--start-pair",
            "1",
            "--max-pairs",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["n_samples"] == 1
    assert payload["n_samples"] == 1
    assert payload["total_cache_pairs"] == 2
    assert payload["start_pair"] == 1
    assert payload["max_pairs"] == 1
    assert payload["pair_window"] == [1, 2]
    assert payload["avg_posenet_dist"] == 0.0
    assert payload["avg_segnet_dist"] == 0.0


def test_mlx_scorer_response_rejects_invalid_pair_window_before_loading() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            max_pairs=0,
        )
    except ValueError as exc:
        assert "max_pairs must be >= 1" in str(exc)
    else:
        raise AssertionError("invalid max_pairs was accepted")


def test_mlx_scorer_response_rejects_gpu_without_explicit_research_allowance() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            device_type="gpu",
        )
    except ValueError as exc:
        assert GPU_RESEARCH_SIGNAL_BLOCKER in str(exc)
        assert "allow_gpu_research_signal=True" in str(exc)
    else:
        raise AssertionError("MLX GPU scorer-response path was accepted without explicit allowance")


def test_mlx_scorer_response_rejects_non_singleton_cpu_batch_without_research_allowance() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            device_type="cpu",
            batch_pairs=2,
        )
    except ValueError as exc:
        assert BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER in str(exc)
        assert "batch_pairs=1" in str(exc)
    else:
        raise AssertionError("MLX CPU non-singleton batch was accepted without research allowance")


def test_mlx_scorer_response_cli_rejects_gpu_without_explicit_research_allowance(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
            "--device",
            "gpu",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert GPU_RESEARCH_SIGNAL_BLOCKER in completed.stderr


def test_mlx_scorer_response_rejects_non_singleton_gpu_batch_after_allowance() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            device_type="gpu",
            batch_pairs=2,
            allow_gpu_research_signal=True,
        )
    except ValueError as exc:
        assert BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER in str(exc)
        assert "batch_pairs=1" in str(exc)
    else:
        raise AssertionError("MLX GPU non-singleton batch was accepted without invariance override")


def test_mlx_scorer_response_cli_rejects_non_singleton_gpu_batch_after_allowance(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    seg = np.zeros((2, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((2, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--batch-pairs",
            "2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER in completed.stderr


def _write_test_cache(
    path: Path,
    *,
    seg: np.ndarray,
    pose: np.ndarray,
    pair_indices: np.ndarray,
) -> Path:
    batch = ScorerInputBatch(
        segnet_last_rgb=seg,
        posenet_yuv6_pair=pose,
        pair_indices=pair_indices,
        metadata={
            "schema_version": "mlx_scorer_input_cache.v1",
            "pair_count": int(pair_indices.shape[0]),
            "segnet_last_rgb_shape": list(seg.shape),
            "posenet_yuv6_pair_shape": list(pose.shape),
            "pair_indices_shape": list(pair_indices.shape),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    write_scorer_input_cache(
        batch,
        path,
        archive_sha256="a" * 64,
        inflated_outputs_aggregate_sha256="b" * 64,
        raw_sha256="c" * 64,
    )
    return path
