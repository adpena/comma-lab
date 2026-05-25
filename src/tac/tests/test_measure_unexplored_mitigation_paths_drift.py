# SPDX-License-Identifier: MIT
"""Regression tests for the unexplored Conv2d drift mitigation measurement CLI."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import tools.measure_unexplored_mitigation_paths_drift as tool


def test_parse_mitigation_paths_expands_all() -> None:
    assert tool._parse_mitigation_paths("all") == (
        "cudnn_reference",
        "fp64",
        "kahan",
        "mlx_deterministic",
    )


def test_parse_mitigation_paths_rejects_unknown() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="unknown mitigation"):
        tool._parse_mitigation_paths("kahan,imaginary")


def test_default_shape_preset_is_single_pr95_stage() -> None:
    shapes = tool._shape_specs_for_preset("pr95-stage2")
    assert [shape["name"] for shape in shapes] == ["pr95_stage2_36_to_144_6x8"]


def test_manifest_respects_mitigation_path_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_measure_threads_1_and_2(shape_spec: dict[str, Any]) -> dict[str, Any]:
        calls.append(str(shape_spec["name"]))
        return {
            "shape_name": shape_spec["name"],
            "shape_spec": dict(shape_spec),
            "baseline_optimized_max_abs": 1.0,
            "fixed_fp32_max_abs": 0.8,
            "kahan_fp32_max_abs": 0.4,
            "fixed_fp64_max_abs": 0.5,
            "fixed_fp32_reduction_percent": 20.0,
            "kahan_fp32_reduction_percent": 60.0,
            "fixed_fp64_reduction_percent": 50.0,
            "thread_1_verdict": "FIXABLE",
            "thread_2_verdict": "FIXABLE",
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(tool, "measure_threads_1_and_2", fake_measure_threads_1_and_2)

    manifest = tool.build_active_exploration_manifest(
        run_id="unit",
        mitigation_paths=("kahan",),
        shape_preset="smoke",
    )

    assert calls == ["smoke_tiny_conv2d_4_to_5_4x4"]
    assert manifest["mitigation_paths"] == ["kahan"]
    assert manifest["shape_preset"] == "smoke"
    assert manifest["thread_1_aggregate_verdict"] == "FIXABLE"
    assert manifest["thread_2_aggregate_verdict"] == "NOT_MEASURED"
    assert (
        manifest["thread_3_mlx_deterministic_investigation"]["measurement_status"]
        == "skipped_by_mitigation_paths_filter"
    )
    assert (
        manifest["thread_4_cudnn_reference_measurement"]["measurement_status"]
        == "skipped_by_mitigation_paths_filter"
    )
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
