# SPDX-License-Identifier: MIT
"""Matched-family deltas for non-authoritative scorer-response datasets."""

from __future__ import annotations

import copy
import math
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.scorer_response_dataset import (
    ScorerResponseDatasetError,
    normalize_legacy_response_dataset_authority,
    render_authority_markdown_block,
)

FAMILY_DELTA_SCHEMA = "scorer_response_family_delta.v1"
DEFAULT_DELTA_FIELDS = (
    "delta_vs_baseline_score",
    "scorer_delta_vs_baseline",
    "pose_term",
    "seg_term",
    "scorer_term",
    "avg_posenet_dist",
    "avg_segnet_dist",
)


def build_family_delta(
    dataset: dict[str, Any],
    *,
    reference_family: str,
    candidate_family: str,
    match_key: str = "source_start_pair",
    fields: tuple[str, ...] = DEFAULT_DELTA_FIELDS,
    top_k: int = 12,
) -> dict[str, Any]:
    """Compare candidate-family rows against reference rows on a shared key."""

    if not reference_family:
        raise ScorerResponseDatasetError("reference_family must be non-empty")
    if not candidate_family:
        raise ScorerResponseDatasetError("candidate_family must be non-empty")
    if not match_key:
        raise ScorerResponseDatasetError("match_key must be non-empty")
    if top_k <= 0:
        raise ScorerResponseDatasetError("top_k must be positive")

    normalized = normalize_legacy_response_dataset_authority(dataset)
    rows = copy.deepcopy(normalized["rows"])
    reference_rows, reference_duplicates = _index_rows(
        rows,
        family=reference_family,
        match_key=match_key,
    )
    candidate_rows, candidate_duplicates = _index_rows(
        rows,
        family=candidate_family,
        match_key=match_key,
    )

    matched: list[dict[str, Any]] = []
    missing_reference: list[str] = []
    missing_candidate: list[str] = []
    for key in sorted(set(reference_rows) | set(candidate_rows), key=_sort_key):
        reference = reference_rows.get(key)
        candidate = candidate_rows.get(key)
        if reference is None:
            missing_reference.append(key)
            continue
        if candidate is None:
            missing_candidate.append(key)
            continue
        matched.append(
            _delta_row(
                key,
                reference=reference,
                candidate=candidate,
                fields=fields,
            )
        )

    score_field = "candidate_minus_reference_delta_vs_baseline_score"
    score_deltas = [
        float(row[score_field])
        for row in matched
        if _finite_float(row.get(score_field)) is not None
    ]
    ranked = sorted(
        matched,
        key=lambda row: (
            _finite_float(row.get(score_field))
            if _finite_float(row.get(score_field)) is not None
            else math.inf
        ),
    )
    authority = normalized.get("authority") if isinstance(normalized.get("authority"), dict) else {}
    evidence_tag = authority.get("evidence_tag") or _single_row_axis(rows)
    evidence_grade = authority.get("evidence_grade")
    if evidence_tag == EVIDENCE_TAG_MLX:
        evidence_grade = EVIDENCE_GRADE_MLX
    return {
        "schema": FAMILY_DELTA_SCHEMA,
        "producer": "tac.optimization.scorer_response_family_delta",
        "dataset_schema": normalized.get("schema"),
        "reference_family": reference_family,
        "candidate_family": candidate_family,
        "match_key": match_key,
        "fields": list(fields),
        "summary": {
            "matched_count": len(matched),
            "missing_reference_count": len(missing_reference),
            "missing_candidate_count": len(missing_candidate),
            "reference_duplicate_count": len(reference_duplicates),
            "candidate_duplicate_count": len(candidate_duplicates),
            "score_delta_mean": _mean(score_deltas),
            "score_delta_min": min(score_deltas) if score_deltas else None,
            "score_delta_max": max(score_deltas) if score_deltas else None,
            "candidate_better_count": sum(1 for value in score_deltas if value < 0.0),
            "candidate_worse_count": sum(1 for value in score_deltas if value > 0.0),
            "candidate_tie_count": sum(1 for value in score_deltas if value == 0.0),
        },
        "top_candidate_improvements": ranked[:top_k],
        "top_candidate_regressions": list(reversed(ranked[-top_k:])),
        "matched_rows": matched,
        "missing_reference_keys": missing_reference[:100],
        "missing_candidate_keys": missing_candidate[:100],
        "reference_duplicate_keys": reference_duplicates[:100],
        "candidate_duplicate_keys": candidate_duplicates[:100],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "evidence_grade": evidence_grade,
        "evidence_tag": evidence_tag,
        "score_axis": authority.get("score_axis") or evidence_tag,
    }


def render_family_delta_markdown(delta: dict[str, Any]) -> str:
    summary = delta["summary"]
    lines = [
        "# Scorer Response Family Delta",
        "",
        f"- Reference family: `{delta['reference_family']}`",
        f"- Candidate family: `{delta['candidate_family']}`",
        f"- Match key: `{delta['match_key']}`",
        f"- Matched rows: `{summary['matched_count']}`",
        f"- Candidate better / worse / tie: `{summary['candidate_better_count']}` / `{summary['candidate_worse_count']}` / `{summary['candidate_tie_count']}`",
        f"- Mean candidate-minus-reference score delta: `{summary['score_delta_mean']}`",
        f"- Min / max score delta: `{summary['score_delta_min']}` / `{summary['score_delta_max']}`",
        "",
    ]
    lines.extend(render_authority_markdown_block(delta))
    lines.extend([
        "## Top Candidate Improvements",
        "",
    ])
    for row in delta["top_candidate_improvements"]:
        lines.append(_render_delta_row(row))
    lines.extend(["", "## Top Candidate Regressions", ""])
    for row in delta["top_candidate_regressions"]:
        lines.append(_render_delta_row(row))
    lines.append("")
    return "\n".join(lines)


def _render_delta_row(row: dict[str, Any]) -> str:
    return (
        f"- key=`{row['match_key_value']}` "
        f"score_delta={row.get('candidate_minus_reference_delta_vs_baseline_score')} "
        f"pose_delta={row.get('candidate_minus_reference_avg_posenet_dist')} "
        f"seg_delta={row.get('candidate_minus_reference_avg_segnet_dist')}"
    )


def _index_rows(
    rows: list[dict[str, Any]],
    *,
    family: str,
    match_key: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        if str(row.get("family") or "") != family:
            continue
        key = _match_value(row, match_key)
        if key is None:
            continue
        if key in indexed:
            duplicates.append(key)
            indexed.pop(key, None)
            continue
        indexed[key] = row
    return indexed, duplicates


def _match_value(row: dict[str, Any], match_key: str) -> str | None:
    value = row.get(match_key)
    if value is None and match_key == "source_start_pair":
        pair_window = row.get("source_pair_window")
        if isinstance(pair_window, list) and pair_window:
            value = pair_window[0]
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _single_row_axis(rows: list[dict[str, Any]]) -> str | None:
    tags = {
        str(row.get("axis") or row.get("source_evidence_tag"))
        for row in rows
        if row.get("axis") is not None or row.get("source_evidence_tag") is not None
    }
    if len(tags) == 1:
        return next(iter(tags))
    return None


def _delta_row(
    key: str,
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "match_key_value": key,
        "reference_row_id": reference.get("row_id"),
        "candidate_row_id": candidate.get("row_id"),
        "source_pair_window": candidate.get("source_pair_window") or reference.get("source_pair_window"),
        "score_claim": False,
    }
    for field in fields:
        reference_value = _finite_float(reference.get(field))
        candidate_value = _finite_float(candidate.get(field))
        out[f"reference_{field}"] = reference_value
        out[f"candidate_{field}"] = candidate_value
        out[f"candidate_minus_reference_{field}"] = (
            None
            if reference_value is None or candidate_value is None
            else candidate_value - reference_value
        )
    return out


def _sort_key(value: str) -> tuple[int, float | str]:
    number = _finite_float(value)
    if number is not None:
        return (0, number)
    return (1, value)


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


__all__ = [
    "DEFAULT_DELTA_FIELDS",
    "FAMILY_DELTA_SCHEMA",
    "build_family_delta",
    "render_family_delta_markdown",
]
