# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from experiments.contest_eval import parse_report
from tac.eval.auth_eval import ReportMetrics
from tac.measurement_integrity import (
    assert_no_git_log_wc_count_consumers,
    find_git_log_wc_count_consumers,
)


def test_git_log_wc_count_consumers_are_refused() -> None:
    text = "commits=$(git log --oneline HEAD~200..HEAD | wc -l)\n"

    hits = find_git_log_wc_count_consumers(text, "script.sh")

    assert hits == ["script.sh:1: commits=$(git log --oneline HEAD~200..HEAD | wc -l)"]
    with pytest.raises(AssertionError, match="git rev-list --count"):
        assert_no_git_log_wc_count_consumers(text, "script.sh")


def test_git_rev_list_count_is_allowed() -> None:
    assert_no_git_log_wc_count_consumers(
        "commits=$(git rev-list --count HEAD~200..HEAD)\n",
        "script.sh",
    )


def test_report_metrics_best_score_prefers_recomputed_components() -> None:
    metrics = ReportMetrics(
        avg_posenet_dist=0.0004,
        avg_segnet_dist=0.001,
        compression_rate=0.004,
        final_score=0.21,
    )

    assert metrics.computed_score == pytest.approx(0.1 + math.sqrt(0.004) + 0.1)
    assert metrics.best_score == pytest.approx(metrics.computed_score)


def test_contest_eval_parse_report_treats_final_score_as_display_rounded(tmp_path) -> None:
    report = tmp_path / "report.txt"
    report.write_text(
        "\n".join(
            [
                "Average PoseNet Distortion: 0.0004",
                "Average SegNet Distortion: 0.001",
                "Submission file size: 1,234 bytes",
                "Original uncompressed size: 37,545,489 bytes",
                "Compression Rate: 0.004",
                "Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.21",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_report(report)

    expected = 0.1 + math.sqrt(0.004) + 0.1
    assert parsed["reported_final_score_display_rounded"] == 0.21
    assert parsed["score_recomputed_from_components"] == pytest.approx(expected)
    assert parsed["score"] == pytest.approx(expected)
    assert parsed["canonical_score_source"] == "score_recomputed_from_components"
