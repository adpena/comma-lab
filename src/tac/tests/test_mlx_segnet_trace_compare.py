# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_response import GPU_RESEARCH_SIGNAL_BLOCKER
from tac.local_acceleration.mlx_segnet_trace_compare import (
    MLXSegNetTraceComparisonError,
    compare_mlx_segnet_layer_traces,
)

REPO = Path(__file__).resolve().parents[3]


def _trace(
    *,
    name: str,
    max_delta: float,
    argmax_pixels: int,
    device_type: str = "cpu",
    gpu_research_signal_allowed: bool = False,
) -> dict:
    return {
        "schema_version": "mlx_segnet_layer_trace.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "device_type": device_type,
        "gpu_research_signal_allowed": gpu_research_signal_allowed,
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
    assert comparison["score_claim_valid"] is False
    assert comparison["promotion_eligible"] is False
    assert comparison["rank_or_kill_eligible"] is False
    assert comparison["ready_for_exact_eval_dispatch"] is False
    assert comparison["candidate_generation_only"] is True
    assert comparison["requires_exact_eval_before_promotion"] is True
    assert comparison["evidence_tag"] == EVIDENCE_TAG_MLX
    assert comparison["score_axis"] == EVIDENCE_TAG_MLX
    assert comparison["baseline_device_type"] == "cpu"
    assert comparison["candidate_device_type"] == "cpu"
    assert comparison["device_contract"]["required_false_authority_fields"] == [
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "promotable",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ]
    assert comparison["verdict"] == "TRACE_CANDIDATE_WORSENED_DRIFT"
    assert comparison["common_row_count"] == 1
    assert comparison["worsened_rows_top"][0]["name"] == "encoder.stage_0.block_0.bn2"
    assert comparison["worsened_rows_top"][0]["max_abs_delta_change"] == 0.1


@pytest.mark.parametrize(
    "field",
    [
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    ],
)
def test_compare_mlx_segnet_layer_traces_requires_false_authority_contract(
    field: str,
) -> None:
    baseline = _trace(name="encoder.stage_0.block_0.bn2", max_delta=0.1, argmax_pixels=1)
    baseline.pop(field)

    with pytest.raises(MLXSegNetTraceComparisonError, match=f"baseline: {field}"):
        compare_mlx_segnet_layer_traces(
            baseline=baseline,
            candidate=_trace(
                name="encoder.stage_0.block_0.bn2",
                max_delta=0.2,
                argmax_pixels=1,
            ),
        )


def test_compare_mlx_segnet_layer_traces_requires_device_and_evidence_semantics() -> None:
    baseline = _trace(name="encoder.stage_0.block_0.bn2", max_delta=0.1, argmax_pixels=1)
    baseline["evidence_tag"] = "[contest-CUDA]"

    with pytest.raises(MLXSegNetTraceComparisonError, match="baseline: evidence_tag"):
        compare_mlx_segnet_layer_traces(
            baseline=baseline,
            candidate=_trace(
                name="encoder.stage_0.block_0.bn2",
                max_delta=0.2,
                argmax_pixels=1,
            ),
        )


def test_compare_mlx_segnet_layer_traces_fails_closed_for_gpu_research_signal() -> None:
    baseline = _trace(
        name="encoder.stage_0.block_0.bn2",
        max_delta=0.1,
        argmax_pixels=1,
        device_type="gpu",
        gpu_research_signal_allowed=False,
    )

    with pytest.raises(
        MLXSegNetTraceComparisonError,
        match=GPU_RESEARCH_SIGNAL_BLOCKER,
    ):
        compare_mlx_segnet_layer_traces(
            baseline=baseline,
            candidate=_trace(
                name="encoder.stage_0.block_0.bn2",
                max_delta=0.2,
                argmax_pixels=1,
            ),
        )


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
    assert comparison["rank_or_kill_eligible"] is False
    assert comparison["evidence_semantics"] == (
        "diagnostic_pytorch_vs_mlx_segnet_layer_trace_comparison_only"
    )
    assert comparison["improved_rows_top"][0]["max_abs_delta_change"] == -0.05
