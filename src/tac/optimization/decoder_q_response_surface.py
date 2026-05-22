# SPDX-License-Identifier: MIT
"""Window-level response surface for decoder-q candidate planning."""

from __future__ import annotations

import math
from typing import Any

from tac.optimization.scorer_response_dataset import render_authority_markdown_block
from tac.optimization.scorer_response_family_delta import FAMILY_DELTA_SCHEMA

DECODER_Q_RESPONSE_SURFACE_SCHEMA = "decoder_q_response_surface_plan.v1"


class DecoderQResponseSurfaceError(ValueError):
    """Raised when a family-delta artifact cannot be converted to a surface."""


def build_decoder_q_response_surface(
    family_delta: dict[str, Any],
    *,
    improvement_threshold: float = 0.0,
    regression_threshold: float = 0.0,
    top_k: int = 16,
) -> dict[str, Any]:
    """Classify matched windows into preserve/suppress/rerank buckets."""

    if not isinstance(family_delta, dict):
        raise DecoderQResponseSurfaceError("family_delta must be a JSON object")
    if family_delta.get("schema") != FAMILY_DELTA_SCHEMA:
        raise DecoderQResponseSurfaceError("family_delta schema mismatch")
    if improvement_threshold < 0.0 or regression_threshold < 0.0:
        raise DecoderQResponseSurfaceError("thresholds must be non-negative")
    if top_k <= 0:
        raise DecoderQResponseSurfaceError("top_k must be positive")
    matched = family_delta.get("matched_rows")
    if not isinstance(matched, list):
        raise DecoderQResponseSurfaceError("family_delta matched_rows[] missing")

    rows = [
        _surface_row(
            row,
            improvement_threshold=improvement_threshold,
            regression_threshold=regression_threshold,
        )
        for row in matched
        if isinstance(row, dict)
    ]
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda row: _sort_key(row["match_key_value"]))

    preserve = [row for row in rows if row["response_class"] == "preserve_candidate_effect"]
    suppress = [row for row in rows if row["response_class"] == "suppress_or_invert_candidate_effect"]
    neutral = [row for row in rows if row["response_class"] == "neutral_or_uncertain"]
    preserve.sort(key=lambda row: float(row["candidate_minus_reference_score_delta"]))
    suppress.sort(key=lambda row: -float(row["candidate_minus_reference_score_delta"]))

    score_deltas = [float(row["candidate_minus_reference_score_delta"]) for row in rows]
    seg_deltas = [float(row["candidate_minus_reference_seg_term_delta"]) for row in rows]
    pose_deltas = [float(row["candidate_minus_reference_pose_term_delta"]) for row in rows]
    return {
        "schema": DECODER_Q_RESPONSE_SURFACE_SCHEMA,
        "producer": "tac.optimization.decoder_q_response_surface",
        "source_family_delta_schema": family_delta.get("schema"),
        "reference_family": family_delta.get("reference_family"),
        "candidate_family": family_delta.get("candidate_family"),
        "match_key": family_delta.get("match_key"),
        "thresholds": {
            "improvement_threshold": float(improvement_threshold),
            "regression_threshold": float(regression_threshold),
        },
        "summary": {
            "matched_count": len(rows),
            "preserve_candidate_effect_count": len(preserve),
            "suppress_or_invert_candidate_effect_count": len(suppress),
            "neutral_or_uncertain_count": len(neutral),
            "score_delta_mean": _mean(score_deltas),
            "score_delta_sum": sum(score_deltas),
            "preserve_gain_sum": -sum(
                float(row["candidate_minus_reference_score_delta"]) for row in preserve
            ),
            "suppress_harm_sum": sum(
                float(row["candidate_minus_reference_score_delta"]) for row in suppress
            ),
            "seg_term_delta_sum": sum(seg_deltas),
            "pose_term_delta_sum": sum(pose_deltas),
            "axis_dominance_counts": _axis_dominance_counts(rows),
        },
        "top_preserve_windows": preserve[:top_k],
        "top_suppress_windows": suppress[:top_k],
        "neutral_windows_sample": neutral[:top_k],
        "rows": rows,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "evidence_grade": family_delta.get("evidence_grade"),
        "evidence_tag": family_delta.get("evidence_tag"),
        "score_axis": family_delta.get("score_axis") or family_delta.get("evidence_tag"),
    }


def render_decoder_q_response_surface_markdown(surface: dict[str, Any]) -> str:
    summary = surface["summary"]
    lines = [
        "# Decoder-Q Response Surface",
        "",
        f"- Reference family: `{surface.get('reference_family')}`",
        f"- Candidate family: `{surface.get('candidate_family')}`",
        f"- Matched windows: `{summary['matched_count']}`",
        f"- Preserve / suppress / neutral: `{summary['preserve_candidate_effect_count']}` / `{summary['suppress_or_invert_candidate_effect_count']}` / `{summary['neutral_or_uncertain_count']}`",
        f"- Preserve gain sum: `{summary['preserve_gain_sum']}`",
        f"- Suppress harm sum: `{summary['suppress_harm_sum']}`",
        f"- Score delta sum: `{summary['score_delta_sum']}`",
        f"- Axis dominance: `{summary['axis_dominance_counts']}`",
        "",
    ]
    lines.extend(render_authority_markdown_block(surface))
    lines.extend([
        "## Preserve Candidate Effect",
        "",
    ])
    for row in surface["top_preserve_windows"]:
        lines.append(_render_row(row))
    lines.extend(["", "## Suppress Or Invert Candidate Effect", ""])
    for row in surface["top_suppress_windows"]:
        lines.append(_render_row(row))
    lines.append("")
    return "\n".join(lines)


def _surface_row(
    row: dict[str, Any],
    *,
    improvement_threshold: float,
    regression_threshold: float,
) -> dict[str, Any] | None:
    score_delta = _finite_float(row.get("candidate_minus_reference_delta_vs_baseline_score"))
    if score_delta is None:
        return None
    pose_term_delta = _term_delta(row, "pose")
    seg_term_delta = _term_delta(row, "seg")
    scorer_term_delta = _finite_float(row.get("candidate_minus_reference_scorer_term"))
    if scorer_term_delta is None:
        scorer_term_delta = pose_term_delta + seg_term_delta
    if score_delta < -float(improvement_threshold):
        response_class = "preserve_candidate_effect"
        recommended_action = "prefer_window_or_similar_axis_pattern"
        optimization_weight = -score_delta
    elif score_delta > float(regression_threshold):
        response_class = "suppress_or_invert_candidate_effect"
        recommended_action = "penalize_window_or_try_opposite_sign"
        optimization_weight = score_delta
    else:
        response_class = "neutral_or_uncertain"
        recommended_action = "low_priority_until_more_signal"
        optimization_weight = 0.0
    return {
        "match_key_value": str(row.get("match_key_value")),
        "source_pair_window": row.get("source_pair_window"),
        "reference_row_id": row.get("reference_row_id"),
        "candidate_row_id": row.get("candidate_row_id"),
        "candidate_minus_reference_score_delta": score_delta,
        "candidate_minus_reference_pose_term_delta": pose_term_delta,
        "candidate_minus_reference_seg_term_delta": seg_term_delta,
        "candidate_minus_reference_scorer_term_delta": scorer_term_delta,
        "axis_dominance": _axis_dominance(pose_term_delta, seg_term_delta),
        "response_class": response_class,
        "recommended_action": recommended_action,
        "optimization_weight": optimization_weight,
        "score_claim": False,
    }


def _term_delta(row: dict[str, Any], axis: str) -> float:
    direct = _finite_float(row.get(f"candidate_minus_reference_{axis}_term"))
    if direct is not None:
        return direct
    if axis == "seg":
        dist_delta = _finite_float(row.get("candidate_minus_reference_avg_segnet_dist"))
        return 100.0 * dist_delta if dist_delta is not None else 0.0
    if axis == "pose":
        candidate = _finite_float(row.get("candidate_avg_posenet_dist"))
        reference = _finite_float(row.get("reference_avg_posenet_dist"))
        if candidate is not None and reference is not None and candidate >= 0.0 and reference >= 0.0:
            return math.sqrt(10.0 * candidate) - math.sqrt(10.0 * reference)
    return 0.0


def _axis_dominance(pose_term_delta: float, seg_term_delta: float) -> str:
    pose_abs = abs(float(pose_term_delta))
    seg_abs = abs(float(seg_term_delta))
    if pose_abs == 0.0 and seg_abs == 0.0:
        return "none"
    if seg_abs >= 2.0 * pose_abs:
        return "seg"
    if pose_abs >= 2.0 * seg_abs:
        return "pose"
    return "mixed"


def _axis_dominance_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        axis = str(row.get("axis_dominance") or "unknown")
        counts[axis] = counts.get(axis, 0) + 1
    return counts


def _render_row(row: dict[str, Any]) -> str:
    return (
        f"- key=`{row['match_key_value']}` "
        f"class=`{row['response_class']}` "
        f"score_delta={row['candidate_minus_reference_score_delta']} "
        f"seg_term_delta={row['candidate_minus_reference_seg_term_delta']} "
        f"pose_term_delta={row['candidate_minus_reference_pose_term_delta']} "
        f"axis=`{row['axis_dominance']}`"
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _sort_key(value: Any) -> tuple[int, float | str]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    return (0, parsed)


__all__ = [
    "DECODER_Q_RESPONSE_SURFACE_SCHEMA",
    "DecoderQResponseSurfaceError",
    "build_decoder_q_response_surface",
    "render_decoder_q_response_surface_markdown",
]
