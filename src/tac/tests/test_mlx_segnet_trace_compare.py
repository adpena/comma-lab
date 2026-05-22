# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration.mlx_segnet_trace_compare import (
    compare_mlx_segnet_layer_traces,
)

REPO = Path(__file__).resolve().parents[3]


def _trace(*, name: str, max_delta: float, argmax_pixels: int) -> dict:
    return {
        "schema_version": "mlx_segnet_layer_trace.v1",
        "score_claim": False,
        "pair_window": [156, 160],
        "trace_count": 1,
        "segnet_argmax_diff_pixels": argmax_pixels,
        "drift_cliff": {
            "name": name,
            "max_abs_delta": max_delta,
            "exceeds_cliff_threshold": True,
        },
        "rows": [
            {
                "index": 0,
                "name": name,
                "shape_match": True,
                "exceeds_cliff_threshold": True,
                "max_abs_delta": max_delta,
                "mean_abs_delta": max_delta / 10.0,
                "rms_delta": max_delta / 5.0,
                "p95_abs_delta": max_delta / 2.0,
                "p99_abs_delta": max_delta / 1.5,
            }
        ],
    }


def test_compare_mlx_segnet_layer_traces_detects_worsening() -> None:
    comparison = compare_mlx_segnet_layer_traces(
        baseline=_trace(name="encoder.stage_0.block_0.bn2", max_delta=0.1, argmax_pixels=1),
        candidate=_trace(name="encoder.stage_0.block_0.bn2", max_delta=0.2, argmax_pixels=1),
        baseline_label="generic_bn",
        candidate_label="affine_bn",
    )

    assert comparison["score_claim"] is False
    assert comparison["verdict"] == "TRACE_CANDIDATE_WORSENED_DRIFT"
    assert comparison["common_row_count"] == 1
    assert comparison["worsened_rows_top"][0]["name"] == "encoder.stage_0.block_0.bn2"
    assert comparison["worsened_rows_top"][0]["max_abs_delta_change"] == 0.1


def test_compare_mlx_segnet_layer_traces_cli(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(
        json.dumps(_trace(name="encoder.stage_0.block_0.bn2", max_delta=0.1, argmax_pixels=1)),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_trace(name="encoder.stage_0.block_0.bn2", max_delta=0.05, argmax_pixels=1)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "compare_mlx_segnet_layer_traces.py"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "TRACE_CANDIDATE_IMPROVED_DRIFT" in completed.stdout
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["promotion_eligible"] is False
    assert comparison["improved_rows_top"][0]["max_abs_delta_change"] == -0.05
