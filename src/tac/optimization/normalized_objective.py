# SPDX-License-Identifier: MIT
"""Canonical full-video normalization checks for MLX/window response rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES
DEFAULT_ABS_TOL = 1.0e-12


class NormalizedObjectiveError(ValueError):
    """Raised when a row's normalized full-video objective is inconsistent."""


def compute_normalized_full_video_gain(
    observed_scorer_gain: float,
    source_n_samples: int,
    *,
    full_video_denominator: int = CONTEST_EXACT_SAMPLE_COUNT,
) -> float:
    """Scale an observed sample/window scorer gain to full contest-video units."""

    if not math.isfinite(float(observed_scorer_gain)):
        raise NormalizedObjectiveError("observed_scorer_gain_not_finite")
    if source_n_samples < 1:
        raise NormalizedObjectiveError("source_n_samples_not_positive")
    if full_video_denominator != CONTEST_EXACT_SAMPLE_COUNT:
        raise NormalizedObjectiveError(
            "full_video_denominator_missing_or_not_contest_sample_count"
        )
    if source_n_samples > full_video_denominator:
        raise NormalizedObjectiveError("source_n_samples_exceeds_full_video_denominator")
    return float(observed_scorer_gain) * float(source_n_samples) / float(full_video_denominator)


def normalized_full_video_objective_metrics(
    row: Mapping[str, Any],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> tuple[dict[str, float], list[str]]:
    """Return recomputed normalized metrics plus fail-closed blocker names."""

    blockers: list[str] = []
    denominator = _as_int(row.get("full_video_denominator"))
    source_n_samples = _as_int(row.get("source_n_samples"))
    observed_gain = _as_float(row.get("observed_scorer_gain_vs_baseline"))
    added_archive_bytes = _as_float(row.get("added_archive_bytes"))
    normalized_gain = _as_float(row.get("normalized_full_video_scorer_gain_vs_baseline"))
    projected_delta = _as_float(row.get("projected_full_video_delta_vs_baseline_score"))
    normalized_margin = _as_float(
        row.get("normalized_full_video_byte_budget_margin_vs_break_even")
    )
    if denominator != CONTEST_EXACT_SAMPLE_COUNT:
        blockers.append("full_video_denominator_missing_or_not_contest_sample_count")
    if source_n_samples is None:
        blockers.append("source_n_samples_missing")
    elif source_n_samples < 1:
        blockers.append("source_n_samples_not_positive")
    elif denominator is not None and source_n_samples > denominator:
        blockers.append("source_n_samples_exceeds_full_video_denominator")
    if observed_gain is None:
        blockers.append("observed_scorer_gain_missing")
    if added_archive_bytes is None:
        blockers.append("added_archive_bytes_missing")
    if normalized_gain is None:
        blockers.append("normalized_full_video_gain_missing")
    if projected_delta is None:
        blockers.append("projected_full_video_delta_missing")
    if normalized_margin is None:
        blockers.append("normalized_full_video_margin_missing")

    recomputed_gain = 0.0
    recomputed_projected_delta = 0.0
    recomputed_break_even_bytes = 0.0
    recomputed_margin = 0.0
    rate_delta = 0.0
    if (
        denominator == CONTEST_EXACT_SAMPLE_COUNT
        and source_n_samples is not None
        and source_n_samples > 0
        and source_n_samples <= denominator
        and observed_gain is not None
    ):
        recomputed_gain = compute_normalized_full_video_gain(
            observed_gain,
            source_n_samples,
            full_video_denominator=denominator,
        )
        if normalized_gain is not None and not _close(
            normalized_gain,
            recomputed_gain,
            abs_tol=abs_tol,
        ):
            blockers.append("normalized_full_video_gain_mismatch")
    if added_archive_bytes is not None:
        rate_delta = RATE_SCORE_PER_BYTE * added_archive_bytes
        recomputed_projected_delta = rate_delta - recomputed_gain
        recomputed_break_even_bytes = recomputed_gain / RATE_SCORE_PER_BYTE
        recomputed_margin = recomputed_break_even_bytes - added_archive_bytes
        if projected_delta is not None and not _close(
            projected_delta,
            recomputed_projected_delta,
            abs_tol=abs_tol,
        ):
            blockers.append("projected_full_video_delta_mismatch")
        if normalized_margin is not None and not _close(
            normalized_margin,
            recomputed_margin,
            abs_tol=abs_tol,
        ):
            blockers.append("normalized_full_video_margin_mismatch")
    return {
        "normalized_gain": recomputed_gain,
        "projected_delta": recomputed_projected_delta,
        "break_even_added_bytes": recomputed_break_even_bytes,
        "normalized_margin": recomputed_margin,
        "rate_delta": rate_delta,
    }, blockers


def require_normalized_full_video_objective(
    row: Mapping[str, Any],
    *,
    label: str = "row",
    abs_tol: float = DEFAULT_ABS_TOL,
) -> dict[str, float]:
    """Return normalized metrics or raise if any supplied objective field lies."""

    metrics, blockers = normalized_full_video_objective_metrics(row, abs_tol=abs_tol)
    if blockers:
        raise NormalizedObjectiveError(f"{label}: " + ", ".join(blockers))
    return metrics


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _close(left: float, right: float, *, abs_tol: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=abs_tol)
