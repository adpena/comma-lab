# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.local_acceleration import EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_preprocess import ScorerInputBatch, write_scorer_input_cache
from tac.local_acceleration.mlx_scorer_response import GPU_RESEARCH_SIGNAL_BLOCKER
from tac.local_acceleration.mlx_scorer_torch_parity import (
    FAIL_VERDICT,
    PASS_SWEEP_VERDICT,
    PASS_VERDICT,
    MLXTorchParityThresholds,
    build_mlx_scorer_torch_parity_manifest,
    build_mlx_scorer_torch_parity_manifest_from_outputs,
    build_mlx_scorer_torch_parity_sweep_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def test_mlx_scorer_torch_parity_passes_cpu_cache_window(tmp_path: Path) -> None:
    cache_dir = _write_test_cache(tmp_path / "cache")

    manifest = build_mlx_scorer_torch_parity_manifest(
        cache_dir=cache_dir,
        repo_root=REPO,
        device_type="cpu",
        start_pair=0,
        max_pairs=1,
        run_id="fixture",
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["requires_exact_eval_before_promotion"] is True
    assert manifest["score_axis"] == EVIDENCE_TAG_MLX
    assert manifest["pair_window"] == [0, 1]
    assert manifest["n_samples"] == 1
    assert manifest["deltas"]["segnet_argmax_diff_pixels"] == 0
    assert manifest["deltas"]["posenet_component_abs_max"] <= 2.0e-5
    assert manifest["device_contract"]["gpu_research_signal_required"] is False


def test_mlx_scorer_torch_parity_rejects_gpu_without_explicit_allowance(
    tmp_path: Path,
) -> None:
    cache_dir = _write_test_cache(tmp_path / "cache")

    try:
        build_mlx_scorer_torch_parity_manifest(
            cache_dir=cache_dir,
            repo_root=REPO,
            device_type="gpu",
            start_pair=0,
            max_pairs=1,
        )
    except ValueError as exc:
        assert GPU_RESEARCH_SIGNAL_BLOCKER in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected GPU parity audit to require explicit allowance")


def test_mlx_scorer_torch_parity_fails_tight_output_thresholds() -> None:
    torch_outputs = {
        "posenet": {"pose": np.zeros((1, 12), dtype=np.float32)},
        "segnet": np.zeros((1, 2, 2, 2), dtype=np.float32),
    }
    mlx_outputs = {
        "posenet": {"pose": np.full((1, 12), 0.1, dtype=np.float32)},
        "segnet": np.ones((1, 2, 2, 2), dtype=np.float32),
    }

    manifest = build_mlx_scorer_torch_parity_manifest_from_outputs(
        torch_outputs=torch_outputs,
        mlx_outputs=mlx_outputs,
        thresholds=MLXTorchParityThresholds(
            max_posenet_output_abs_delta=1.0e-4,
            max_segnet_logit_abs_delta=1.0e-4,
            max_posenet_component_abs_delta=1.0e-6,
            max_segnet_argmax_diff_pixels=0,
        ),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert any(
        blocker.startswith("posenet_output_abs_delta_exceeds_threshold")
        for blocker in manifest["blockers"]
    )
    assert any(
        blocker.startswith("segnet_logit_abs_delta_exceeds_threshold")
        for blocker in manifest["blockers"]
    )


def test_mlx_scorer_torch_parity_counts_segnet_argmax_fraction() -> None:
    torch_seg = np.zeros((1, 2, 2, 2), dtype=np.float32)
    mlx_seg = torch_seg.copy()
    torch_seg[:, 0, :, :] = 1.0
    mlx_seg[:, 0, :, :] = 1.0
    mlx_seg[0, 0, 1, 1] = 0.0
    mlx_seg[0, 1, 1, 1] = 1.1

    manifest = build_mlx_scorer_torch_parity_manifest_from_outputs(
        torch_outputs={
            "posenet": {"pose": np.zeros((1, 12), dtype=np.float32)},
            "segnet": torch_seg,
        },
        mlx_outputs={
            "posenet": {"pose": np.zeros((1, 12), dtype=np.float32)},
            "segnet": mlx_seg,
        },
        thresholds=MLXTorchParityThresholds(
            max_segnet_logit_abs_delta=2.0,
            max_segnet_argmax_diff_pixels=0,
        ),
    )

    assert manifest["passed"] is False
    assert manifest["deltas"]["segnet_argmax_diff_pixels"] == 1
    assert manifest["deltas"]["segnet_argmax_pixel_count"] == 4
    assert manifest["deltas"]["segnet_argmax_diff_fraction"] == 0.25
    assert any(
        blocker.startswith("segnet_argmax_diff_pixels_exceeds_threshold")
        for blocker in manifest["blockers"]
    )


def test_mlx_scorer_torch_parity_cli_rejects_invalid_window_before_loading(tmp_path: Path) -> None:
    output = tmp_path / "parity.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_torch_parity.py"),
            "--cache-dir",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--max-pairs",
            "0",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "FATAL:" in completed.stderr
    assert "max_pairs must be >= 1" in completed.stderr
    assert not output.exists()


def test_mlx_scorer_torch_parity_cli_rejects_gpu_without_allowance_before_loading(
    tmp_path: Path,
) -> None:
    output = tmp_path / "parity.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_torch_parity.py"),
            "--cache-dir",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--device",
            "gpu",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert GPU_RESEARCH_SIGNAL_BLOCKER in completed.stderr
    assert not output.exists()


def test_mlx_scorer_torch_parity_sweep_passes_cpu_cache_windows(tmp_path: Path) -> None:
    cache_dir = _write_test_cache(tmp_path / "cache", pair_count=2)

    manifest = build_mlx_scorer_torch_parity_sweep_manifest(
        cache_dir=cache_dir,
        repo_root=REPO,
        device_type="cpu",
        start_pair=0,
        max_pairs=2,
        window_pairs=1,
        run_id="fixture_sweep",
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_SWEEP_VERDICT
    assert manifest["score_claim"] is False
    assert manifest["requires_exact_eval_before_promotion"] is True
    assert manifest["window_count"] == 2
    assert manifest["covered_pair_window"] == [0, 2]
    assert manifest["summary"]["failed_windows"] == 0
    assert manifest["summary"]["segnet_argmax_diff_pixels"]["max"] == 0.0
    assert manifest["summary"]["segnet_argmax_diff_fraction"]["max"] == 0.0
    assert manifest["rows"][0]["deltas"]["segnet_argmax_pixel_count"] > 0


def test_mlx_scorer_torch_parity_sweep_cli_rejects_empty_window_before_loading(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sweep.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_torch_parity_sweep.py"),
            "--cache-dir",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--window-pairs",
            "0",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "window_pairs must be >= 1" in completed.stderr
    assert not output.exists()


def _write_test_cache_with_pair_count(path: Path, *, pair_count: int) -> Path:
    pair_indices = np.asarray([[2 * idx, 2 * idx + 1] for idx in range(pair_count)], dtype=np.int64)
    seg = np.zeros((pair_count, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((pair_count, 12, 64, 80), dtype=np.float32)
    batch = ScorerInputBatch(
        segnet_last_rgb=seg,
        posenet_yuv6_pair=pose,
        pair_indices=pair_indices,
        metadata={
            "schema_version": "mlx_scorer_input_cache.v1",
            "pair_count": pair_count,
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


def _write_test_cache(path: Path, *, pair_count: int = 1) -> Path:
    return _write_test_cache_with_pair_count(path, pair_count=pair_count)
