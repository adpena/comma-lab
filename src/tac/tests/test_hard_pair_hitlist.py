# SPDX-License-Identifier: MIT
"""Tests for scorer-response hard-pair hitlists."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tac.adaptation.hard_pair_hitlist import (
    build_hard_pair_hitlist_from_mlx_response,
)
from tools.build_hard_pair_hitlist_from_mlx_response import main as hitlist_main


def test_hard_pair_hitlist_ranks_by_contest_marginal(tmp_path: Path) -> None:
    response = _response_with_components(tmp_path)

    hitlist = build_hard_pair_hitlist_from_mlx_response(
        mlx_response=response,
        top_k=2,
    )

    assert hitlist["schema"] == "nerv_hard_pair_hitlist.v1"
    assert hitlist["pair_indices"] == [2, 3]
    assert hitlist["ranked_pairs"][0]["pair_index"] == 2
    assert hitlist["ranked_pairs"][0]["source_frame_pair"] == [33, 34]
    assert hitlist["ranked_pairs"][1]["pair_index"] == 3
    assert hitlist["ranked_pairs"][1]["source_frame_pair"] == [44, 45]
    assert hitlist["score_claim"] is False
    assert hitlist["ready_for_exact_eval_dispatch"] is False


def test_hard_pair_hitlist_cli_writes_runner_consumable_json(tmp_path: Path) -> None:
    response = _response_with_components(tmp_path)
    response_path = tmp_path / "mlx_response.json"
    output = tmp_path / "hard_pairs.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")

    assert hitlist_main(
        [
            "--mlx-response",
            response_path.as_posix(),
            "--output-json",
            output.as_posix(),
            "--top-k",
            "3",
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pair_indices"] == [2, 3, 1]
    assert payload["top_k"] == 3


def _response_with_components(tmp_path: Path) -> dict[str, object]:
    components = tmp_path / "components"
    components.mkdir()
    pose_path = components / "posenet_distortion.npy"
    seg_path = components / "segnet_distortion.npy"
    np.save(pose_path, np.array([1.0, 10.0, 5.0, 1.0], dtype=np.float32))
    np.save(seg_path, np.array([0.01, 0.01, 0.50, 0.25], dtype=np.float32))
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(
        cache / "pair_indices.npy",
        np.array([[11, 12], [22, 23], [33, 34], [44, 45]], dtype=np.int64),
    )
    return {
        "schema": "mlx_scorer_response.v1",
        "avg_posenet_dist": 4.25,
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 123,
        "response_family": "hi_nerv",
        "hardware_substrate": "MLX gpu",
        "evidence_tag": "[macOS-MLX research-signal]",
        "source_cache_run": {"candidate_cache_dir": cache.as_posix()},
        "components": {
            "artifacts": {
                "posenet_distortion": {
                    "path": pose_path.as_posix(),
                    "bytes": pose_path.stat().st_size,
                    "sha256": _sha256(pose_path),
                },
                "segnet_distortion": {
                    "path": seg_path.as_posix(),
                    "bytes": seg_path.stat().st_size,
                    "sha256": _sha256(seg_path),
                },
            }
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
