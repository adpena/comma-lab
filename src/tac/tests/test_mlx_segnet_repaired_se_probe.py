# SPDX-License-Identifier: MIT
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.local_acceleration import EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_preprocess import ScorerInputBatch, write_scorer_input_cache
from tac.local_acceleration.mlx_scorer_response import GPU_RESEARCH_SIGNAL_BLOCKER
from tac.local_acceleration.mlx_segnet_repaired_se_probe import (
    SCHEMA_VERSION,
    VARIANTS,
    build_mlx_segnet_repaired_stage0_se_probe_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def test_mlx_segnet_repaired_stage0_se_probe_schema_and_authority(tmp_path: Path) -> None:
    cache_dir = _write_test_cache(tmp_path / "cache")

    manifest = build_mlx_segnet_repaired_stage0_se_probe_manifest(
        cache_dir=cache_dir,
        repo_root=REPO,
        device_type="cpu",
        start_pair=0,
        max_pairs=1,
        run_id="fixture_repaired_se",
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["requires_exact_eval_before_promotion"] is True
    assert manifest["score_axis"] == EVIDENCE_TAG_MLX
    assert manifest["pair_window"] == [0, 1]
    assert manifest["n_samples"] == 1
    assert [row["label"] for row in manifest["rows"]] == [item[0] for item in VARIANTS]
    for row in manifest["rows"]:
        assert row["segnet_argmax_pixel_count"] > 0
        assert row["segnet_argmax_diff_pixels"] >= 0
    assert manifest["best_variant"]["label"] in {item[0] for item in VARIANTS}


def test_mlx_segnet_repaired_stage0_se_probe_cli_rejects_gpu_without_allowance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "repaired.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "probe_mlx_segnet_repaired_stage0_se.py"),
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


def _write_test_cache(path: Path, *, pair_count: int = 1) -> Path:
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
