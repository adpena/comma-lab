# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.normalized_objective import (
    RATE_SCORE_PER_BYTE,
    NormalizedObjectiveError,
    compute_normalized_full_video_gain,
    require_normalized_full_video_gain,
    require_normalized_full_video_objective,
)


def test_normalized_objective_recomputes_singleton_window_units() -> None:
    raw_gain = 0.012
    normalized_gain = raw_gain / 600.0
    metrics = require_normalized_full_video_objective(
        {
            "observed_scorer_gain_vs_baseline": raw_gain,
            "source_n_samples": 1,
            "full_video_denominator": 600,
            "added_archive_bytes": 13,
            "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
            "projected_full_video_delta_vs_baseline_score": (
                13 * RATE_SCORE_PER_BYTE - normalized_gain
            ),
            "normalized_full_video_byte_budget_margin_vs_break_even": (
                normalized_gain / RATE_SCORE_PER_BYTE - 13
            ),
        }
    )

    assert metrics["normalized_gain"] == pytest.approx(normalized_gain)
    assert metrics["normalized_margin"] == pytest.approx(
        normalized_gain / RATE_SCORE_PER_BYTE - 13
    )


def test_normalized_objective_rejects_raw_gain_disguised_as_full_video() -> None:
    with pytest.raises(NormalizedObjectiveError, match="normalized_full_video_gain_mismatch"):
        require_normalized_full_video_objective(
            {
                "observed_scorer_gain_vs_baseline": 0.012,
                "source_n_samples": 1,
                "full_video_denominator": 600,
                "added_archive_bytes": 0,
                "normalized_full_video_scorer_gain_vs_baseline": 0.012,
                "projected_full_video_delta_vs_baseline_score": -0.012,
                "normalized_full_video_byte_budget_margin_vs_break_even": (
                    0.012 / RATE_SCORE_PER_BYTE
                ),
            }
        )


def test_normalized_gain_guard_rejects_alias_boundary_mismatch() -> None:
    with pytest.raises(NormalizedObjectiveError, match="normalized_full_video_gain_mismatch"):
        require_normalized_full_video_gain(
            observed_gain=0.012,
            source_n_samples=1,
            normalized_gain=0.012,
            full_video_denominator=600,
        )


def test_compute_normalized_gain_rejects_wrong_denominator() -> None:
    with pytest.raises(
        NormalizedObjectiveError,
        match="full_video_denominator_missing_or_not_contest_sample_count",
    ):
        compute_normalized_full_video_gain(
            0.01,
            1,
            full_video_denominator=1,
        )
