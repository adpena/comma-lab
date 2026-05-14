# SPDX-License-Identifier: MIT
"""Tests for the SegNet boundary smoothing GHA harvester."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "harvest_segnet_boundary_smoothing_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "harvest_segnet_boundary_smoothing_sweep", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_verdict_keep_threshold() -> None:
    tool = _load_tool()
    # Below baseline by ≥ 0.001 → KEEP
    assert tool.make_verdict(0.190, 0.19284) == "KEEP"
    assert tool.make_verdict(0.180, 0.19284) == "KEEP"
    assert tool.make_verdict(0.19184, 0.19284) == "KEEP"  # exactly -0.001
    # Just above the keep threshold (delta -0.0009) → DEFER
    assert tool.make_verdict(0.19194, 0.19284) == "DEFER"


def test_make_verdict_reject_threshold() -> None:
    tool = _load_tool()
    # Above baseline by ≥ 0.001 → REJECT
    assert tool.make_verdict(0.19384, 0.19284) == "REJECT"  # exactly +0.001
    assert tool.make_verdict(0.5, 0.19284) == "REJECT"
    # Just below the reject threshold (delta +0.0009) → DEFER
    assert tool.make_verdict(0.19374, 0.19284) == "DEFER"


def test_make_verdict_defer_band() -> None:
    tool = _load_tool()
    # Within (-0.001, +0.001) → DEFER
    assert tool.make_verdict(0.19284, 0.19284) == "DEFER"
    assert tool.make_verdict(0.19350, 0.19284) == "DEFER"
    assert tool.make_verdict(0.19250, 0.19284) == "DEFER"


def test_a1_baseline_constants() -> None:
    """A1 baseline + medal-band threshold are constants from CLAUDE.md."""
    tool = _load_tool()
    assert tool.A1_BASELINE_SCORE == 0.19284757743677347
    assert tool.A1_BASELINE_TAG == "[contest-CPU GHA Linux x86_64]"
    assert tool.PR102_SILVER_SCORE == 0.19538
    assert tool.SUB_MEDAL_BAND_THRESHOLD == 0.190


def test_parse_report_recomputes_score() -> None:
    """Verify parse_report recomputes score from components per contest formula."""
    tool = _load_tool()
    report = (
        "=== Evaluation config ===\n"
        "  device: cpu\n"
        "  submission_dir: submissions/test_sub\n"
        "=== Evaluation results over 600 samples ===\n"
        "  Average PoseNet Distortion: 0.00003351\n"
        "  Average SegNet Distortion: 0.00058211\n"
        "  Submission file size: 178,262 bytes\n"
        "  Compression Rate: 0.00474789\n"
        "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.20\n"
    )
    parsed = tool.parse_report(report, expected_submission_name="test_sub")
    assert "_error" not in parsed
    assert parsed["report_device"] == "cpu"
    assert parsed["report_submission_name"] == "test_sub"
    assert parsed["avg_segnet_dist"] == 0.00058211
    assert parsed["avg_posenet_dist"] == 0.00003351
    assert parsed["compression_rate"] == 0.00474789
    assert parsed["n_samples"] == 600
    assert parsed["submission_file_size"] == 178262
    expected_score = (
        100.0 * 0.00058211 + (10.0 * 0.00003351) ** 0.5 + 25.0 * 0.00474789
    )
    assert abs(parsed["score_recomputed"] - expected_score) < 1e-12


def test_parse_report_refuses_submission_name_mismatch() -> None:
    """Verify parse_report fails CLOSED on submission_name mismatch (custody)."""
    tool = _load_tool()
    report = (
        "=== Evaluation config ===\n"
        "  device: cpu\n"
        "  submission_dir: submissions/wrong_sub\n"
        "=== Evaluation results over 600 samples ===\n"
        "  Average PoseNet Distortion: 0.0\n"
        "  Average SegNet Distortion: 0.0\n"
        "  Submission file size: 178,262 bytes\n"
        "  Compression Rate: 0.0\n"
        "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.0\n"
    )
    parsed = tool.parse_report(report, expected_submission_name="expected_sub")
    assert "_error" in parsed
    assert "submission_name mismatch" in parsed["_error"]
    assert parsed["report_submission_name"] == "wrong_sub"
    assert parsed["expected_submission_name"] == "expected_sub"


def test_parse_report_refuses_non_cpu_device() -> None:
    """CPU-only path: fail closed if report device is not 'cpu'."""
    tool = _load_tool()
    report = (
        "=== Evaluation config ===\n"
        "  device: cuda\n"
        "  submission_dir: submissions/test_sub\n"
        "=== Evaluation results over 600 samples ===\n"
        "  Average PoseNet Distortion: 0.0\n"
        "  Average SegNet Distortion: 0.0\n"
        "  Submission file size: 178,262 bytes\n"
        "  Compression Rate: 0.0\n"
        "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.0\n"
    )
    parsed = tool.parse_report(report)
    assert "_error" in parsed
    assert "unexpected report device" in parsed["_error"]


def test_parse_report_refuses_non_600_samples() -> None:
    """Contest contract: 600 samples; refuse anything else."""
    tool = _load_tool()
    report = (
        "=== Evaluation config ===\n"
        "  device: cpu\n"
        "  submission_dir: submissions/test_sub\n"
        "=== Evaluation results over 100 samples ===\n"
        "  Average PoseNet Distortion: 0.0\n"
        "  Average SegNet Distortion: 0.0\n"
        "  Submission file size: 178,262 bytes\n"
        "  Compression Rate: 0.0\n"
        "  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.0\n"
    )
    parsed = tool.parse_report(report)
    assert "_error" in parsed
    assert "unexpected sample count" in parsed["_error"]
