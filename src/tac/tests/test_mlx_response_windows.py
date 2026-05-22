# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.auth_eval_schema import contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_response_windows import (
    MLXResponseWindowSplitError,
    WINDOW_SPLIT_SCHEMA,
    split_mlx_scorer_response_windows,
)

REPO = Path(__file__).resolve().parents[3]


def test_split_mlx_scorer_response_windows_writes_false_authority_rows(tmp_path: Path) -> None:
    payload = _response_payload()
    index = split_mlx_scorer_response_windows(
        response_payload=payload,
        posenet_distortion=np.array([0.0, 1.0e-6, 4.0e-6], dtype=np.float32),
        segnet_distortion=np.array([0.0, 0.001, 0.002], dtype=np.float32),
        output_dir=tmp_path / "windows",
        components_dir=tmp_path / "components",
        window_pairs=1,
        prefix="candidate",
    )

    assert index["score_claim"] is False
    assert index["promotion_eligible"] is False
    assert index["window_count"] == 3
    first = json.loads(Path(index["rows"][0]["path"]).read_text(encoding="utf-8"))
    assert first["schema_version"] == "mlx_scorer_response.v1"
    assert first["score_claim"] is False
    assert first["promotion_eligible"] is False
    assert first["batch_pairs"] == 1
    assert first["pair_window"] == [10, 11]
    assert first["n_samples"] == 1
    assert first["avg_posenet_dist"] == 0.0
    assert first["avg_segnet_dist"] == 0.0
    assert first["canonical_score"] == contest_formula_score(
        seg_dist=0.0,
        pose_dist=0.0,
        archive_bytes=1000,
    )
    assert Path(first["components"]["artifacts"]["posenet_distortion"]["path"]).exists()


def test_split_mlx_scorer_response_windows_supports_strided_multi_pair_windows(
    tmp_path: Path,
) -> None:
    payload = _response_payload()

    index = split_mlx_scorer_response_windows(
        response_payload=payload,
        posenet_distortion=np.array([0.0, 1.0e-6, 4.0e-6], dtype=np.float32),
        segnet_distortion=np.array([0.0, 0.001, 0.002], dtype=np.float32),
        output_dir=tmp_path / "windows",
        window_pairs=2,
        stride_pairs=2,
        prefix="candidate",
    )

    assert index["schema_version"] == WINDOW_SPLIT_SCHEMA
    assert index["window_count"] == 2
    assert [row["pair_window"] for row in index["rows"]] == [[10, 12], [12, 13]]

    first = json.loads(Path(index["rows"][0]["path"]).read_text(encoding="utf-8"))
    assert first["batch_pairs"] == 1
    assert first["pair_window"] == [10, 12]
    assert first["n_samples"] == 2
    expected_score = contest_formula_score(
        seg_dist=float(np.mean(np.array([0.0, 0.001], dtype=np.float32), dtype=np.float64)),
        pose_dist=float(np.mean(np.array([0.0, 1.0e-6], dtype=np.float32), dtype=np.float64)),
        archive_bytes=1000,
    )
    assert first["canonical_score"] == expected_score
    assert first["score_claim"] is False
    assert first["rank_or_kill_eligible"] is False


def test_split_mlx_scorer_response_windows_rejects_non_singleton_parent(tmp_path: Path) -> None:
    payload = _response_payload()
    payload["batch_pairs"] = 2

    try:
        split_mlx_scorer_response_windows(
            response_payload=payload,
            posenet_distortion=np.zeros(3, dtype=np.float32),
            segnet_distortion=np.zeros(3, dtype=np.float32),
            output_dir=tmp_path / "windows",
            window_pairs=1,
        )
    except MLXResponseWindowSplitError as exc:
        assert "batch_pairs=1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-singleton parent was accepted")


def test_split_mlx_scorer_response_windows_cli(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    components_dir = tmp_path / "parent_components"
    components_dir.mkdir()
    pose_path = components_dir / "posenet_distortion.npy"
    seg_path = components_dir / "segnet_distortion.npy"
    np.save(pose_path, np.array([0.0, 1.0e-6, 4.0e-6], dtype=np.float32))
    np.save(seg_path, np.array([0.0, 0.001, 0.002], dtype=np.float32))
    payload = _response_payload()
    payload["components"]["artifacts"] = {
        "posenet_distortion": {"path": str(pose_path)},
        "segnet_distortion": {"path": str(seg_path)},
    }
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    index_path = tmp_path / "index.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "split_mlx_scorer_response_windows.py"),
            "--response",
            str(response_path),
            "--output-dir",
            str(tmp_path / "windows"),
            "--index-out",
            str(index_path),
            "--window-pairs",
            "1",
            "--max-windows",
            "2",
            "--prefix",
            "baseline",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"score_claim": false' in completed.stdout
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["window_count"] == 2


def _response_payload() -> dict:
    return {
        "schema_version": "mlx_scorer_response.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "hardware_substrate": "MLX cpu",
        "batch_pairs": 1,
        "start_pair": 10,
        "max_pairs": 3,
        "n_samples": 3,
        "pair_window": [10, 13],
        "elapsed_seconds": 1.0,
        "canonical_score": 0.1,
        "score_recomputed_from_components": 0.1,
        "canonical_score_source": "score_recomputed_from_components",
        "archive_size_bytes": 1000,
        "avg_posenet_dist": 0.0,
        "avg_segnet_dist": 0.0,
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "c" * 64,
        "components": {
            "posenet_shape": [3],
            "segnet_shape": [3],
            "posenet_sha256": "d" * 64,
            "segnet_sha256": "e" * 64,
            "artifacts": {},
        },
        "cache_identity": {
            "pair_indices_equal": True,
            "reference": {
                "archive_sha256": None,
                "inflated_outputs_aggregate_sha256": None,
                "raw_sha256": None,
                "array_sha256": {
                    "pair_indices": "0" * 64,
                    "posenet_yuv6_pair": "1" * 64,
                    "segnet_last_rgb": "2" * 64,
                },
            },
            "candidate": {
                "archive_sha256": "a" * 64,
                "inflated_outputs_aggregate_sha256": "b" * 64,
                "raw_sha256": "c" * 64,
                "array_sha256": {
                    "pair_indices": "0" * 64,
                    "posenet_yuv6_pair": "3" * 64,
                    "segnet_last_rgb": "4" * 64,
                },
            },
        },
        "device_contract": {
            "forbidden_uses": ["auth_eval", "score_claim", "promotion", "rank_or_kill"],
            "allowed_uses": ["local_mlx_training_gradient_shaping"],
        },
    }
