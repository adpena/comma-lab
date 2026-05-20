# SPDX-License-Identifier: MIT
"""Score-weighted byte-importance helpers for contest archive gradients.

The master-gradient archive tensor is stored per scorer axis.  Ranking by raw
vector norm is not the same as ranking by score impact because the contest
formula weights SegNet, PoseNet, and rate differently at the operating point.
This module applies the explicit marginal coefficients before top-K selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

AXIS_ORDER: tuple[str, str, str] = ("seg", "pose", "rate")


class ByteScoreImpactError(ValueError):
    """Raised when byte score-impact inputs are malformed."""


def marginal_vector(marginal_coefficients: Mapping[str, Any]) -> "Any":
    """Return `[seg, pose, rate]` marginal coefficients as a float64 vector."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy required for byte score-impact ranking") from exc

    values: list[float] = []
    for axis in AXIS_ORDER:
        raw = marginal_coefficients.get(axis)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ByteScoreImpactError(f"missing numeric marginal coefficient for axis {axis!r}")
        value = float(raw)
        if value < 0.0:
            raise ByteScoreImpactError(f"negative marginal coefficient for axis {axis!r}: {value}")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def score_impact_matrix(
    m_archive: "Any",
    marginal_coefficients: Mapping[str, Any],
) -> "Any":
    """Return `abs(M_archive) * [dS/dseg, dS/dpose, dS/dbyte]`.

    `M_archive` must have shape `(N_bytes, 3)` with axis order
    `(seg, pose, rate)`.  The result is in score-impact units per byte/axis.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy required for byte score-impact ranking") from exc

    arr = np.asarray(m_archive, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ByteScoreImpactError(f"M_archive must have shape (N_bytes, 3); got {arr.shape}")
    return np.abs(arr) * marginal_vector(marginal_coefficients)


def rank_bytes_by_score_impact(
    m_archive: "Any",
    marginal_coefficients: Mapping[str, Any],
    *,
    k_top: int,
    axis: str | None = None,
) -> list[int]:
    """Rank archive bytes by score-impact magnitude and return top-K indices."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy required for byte score-impact ranking") from exc

    if isinstance(k_top, bool) or k_top < 0:
        raise ByteScoreImpactError(f"k_top must be a non-negative integer; got {k_top!r}")
    impact = score_impact_matrix(m_archive, marginal_coefficients)
    if axis is None:
        magnitude = impact.sum(axis=1)
    else:
        if axis not in AXIS_ORDER:
            raise ByteScoreImpactError(f"axis must be one of {AXIS_ORDER}; got {axis!r}")
        magnitude = impact[:, AXIS_ORDER.index(axis)]
    if k_top == 0:
        return []
    count = min(int(k_top), int(impact.shape[0]))
    order = np.argsort(-magnitude, kind="stable")
    return [int(index) for index in order[:count]]


def summarize_topk_score_impact(
    m_archive: "Any",
    marginal_coefficients: Mapping[str, Any],
    *,
    k_top: int,
    top_record_limit: int = 128,
) -> dict[str, Any]:
    """Summarize aggregate + per-axis score-impact mass for a top-K set."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy required for byte score-impact ranking") from exc

    impact = score_impact_matrix(m_archive, marginal_coefficients)
    indices = rank_bytes_by_score_impact(
        m_archive,
        marginal_coefficients,
        k_top=k_top,
    )
    selected = impact[indices] if indices else impact[:0]
    selected_axis_sums = selected.sum(axis=0) if indices else np.zeros(3, dtype=np.float64)
    total_axis_sums = impact.sum(axis=0)
    selected_total = float(selected_axis_sums.sum())
    total_mass = float(total_axis_sums.sum())
    dominant_axis_counts = {axis: 0 for axis in AXIS_ORDER}
    for row in selected:
        dominant_axis_counts[AXIS_ORDER[int(np.argmax(row))]] += 1

    records = []
    magnitude = impact.sum(axis=1)
    for byte_index in indices[: max(0, int(top_record_limit))]:
        axis_values = {axis: float(impact[byte_index, i]) for i, axis in enumerate(AXIS_ORDER)}
        row_total = float(magnitude[byte_index])
        records.append(
            {
                "byte_index": int(byte_index),
                "score_impact_abs_sum": row_total,
                "axis_score_impact": axis_values,
                "dominant_axis": max(axis_values, key=axis_values.get),
            }
        )

    return {
        "k_top": int(min(k_top, impact.shape[0])),
        "top_byte_indices": indices,
        "top_byte_runs": contiguous_byte_runs(indices),
        "top_record_limit": int(top_record_limit),
        "top_records": records,
        "selected_score_impact_abs_sum": selected_total,
        "total_score_impact_abs_sum": total_mass,
        "selected_total_share": (selected_total / total_mass) if total_mass else 0.0,
        "selected_axis_score_impact_abs_sum": {
            axis: float(selected_axis_sums[i]) for i, axis in enumerate(AXIS_ORDER)
        },
        "selected_axis_share_within_topk": {
            axis: (float(selected_axis_sums[i]) / selected_total) if selected_total else 0.0
            for i, axis in enumerate(AXIS_ORDER)
        },
        "total_axis_score_impact_abs_sum": {
            axis: float(total_axis_sums[i]) for i, axis in enumerate(AXIS_ORDER)
        },
        "dominant_axis_counts_within_topk": dominant_axis_counts,
    }


def contiguous_byte_runs(indices: Sequence[int]) -> list[dict[str, int]]:
    """Group byte indices into contiguous runs sorted by byte offset."""

    sorted_indices = sorted({int(index) for index in indices})
    if not sorted_indices:
        return []
    runs: list[dict[str, int]] = []
    start = prev = sorted_indices[0]
    for index in sorted_indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        runs.append({"start": start, "end": prev, "length": prev - start + 1})
        start = prev = index
    runs.append({"start": start, "end": prev, "length": prev - start + 1})
    return runs


def parse_k_values(raw: str | Sequence[int]) -> list[int]:
    """Parse comma-separated K values and return unique positive ints."""

    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        values = [int(token) for token in tokens]
    else:
        values = [int(value) for value in raw]
    out: list[int] = []
    for value in values:
        if value <= 0:
            raise ByteScoreImpactError(f"K values must be positive; got {value}")
        if value not in out:
            out.append(value)
    if not out:
        raise ByteScoreImpactError("at least one K value is required")
    return out


__all__ = [
    "AXIS_ORDER",
    "ByteScoreImpactError",
    "contiguous_byte_runs",
    "marginal_vector",
    "parse_k_values",
    "rank_bytes_by_score_impact",
    "score_impact_matrix",
    "summarize_topk_score_impact",
]
