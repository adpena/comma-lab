from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path


@dataclass(frozen=True)
class SelectionMetric:
    key: str
    direction: str = "min"

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("metric key must be non-empty")
        if self.direction not in {"min", "max"}:
            raise ValueError("metric direction must be 'min' or 'max'")


@dataclass(frozen=True)
class _PreparedCandidate:
    summary: Mapping[str, object]
    metric_values: tuple[int | float | str, ...]
    canonical_summary: str
    original_index: int


def rank_exact_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[SelectionMetric],
    exact_key: str = "exact_match",
) -> tuple[Mapping[str, object], ...]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    if not metrics:
        raise ValueError("metrics must not be empty")

    prepared: list[_PreparedCandidate] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {index} must be a mapping")
        exact_value = candidate.get(exact_key)
        if not isinstance(exact_value, bool):
            raise ValueError(f"candidate {index} must include boolean {exact_key!r}")
        if not exact_value:
            continue
        prepared.append(
            _PreparedCandidate(
                summary=candidate,
                metric_values=tuple(_metric_value(candidate, metric, index) for metric in metrics),
                canonical_summary=_canonical_summary(candidate),
                original_index=index,
            )
        )

    if not prepared:
        raise ValueError(f"at least one exact candidate is required via {exact_key!r}")

    ranked = sorted(prepared, key=cmp_to_key(lambda left, right: _compare_candidates(left, right, metrics)))
    return tuple(item.summary for item in ranked)


def select_exact_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[SelectionMetric],
    exact_key: str = "exact_match",
) -> Mapping[str, object]:
    return rank_exact_candidates(candidates, metrics=metrics, exact_key=exact_key)[0]


def _metric_value(candidate: Mapping[str, object], metric: SelectionMetric, candidate_index: int) -> int | float | str:
    if metric.key not in candidate:
        raise ValueError(f"candidate {candidate_index} is missing required metric {metric.key!r}")
    value = candidate[metric.key]
    if isinstance(value, bool):
        raise ValueError(f"candidate {candidate_index} metric {metric.key!r} must not be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"candidate {candidate_index} metric {metric.key!r} must be finite")
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"candidate {candidate_index} metric {metric.key!r} must be int, float, or str")


def _compare_candidates(
    left: _PreparedCandidate,
    right: _PreparedCandidate,
    metrics: Sequence[SelectionMetric],
) -> int:
    for metric, left_value, right_value in zip(metrics, left.metric_values, right.metric_values):
        order = _compare_metric_values(left_value, right_value, metric)
        if order != 0:
            return order

    if left.canonical_summary < right.canonical_summary:
        return -1
    if left.canonical_summary > right.canonical_summary:
        return 1
    if left.original_index < right.original_index:
        return -1
    if left.original_index > right.original_index:
        return 1
    return 0


def _compare_metric_values(left: int | float | str, right: int | float | str, metric: SelectionMetric) -> int:
    left_kind = _metric_kind(left)
    right_kind = _metric_kind(right)
    if left_kind != right_kind:
        raise ValueError(f"metric {metric.key!r} must have consistent scalar types across exact candidates")
    if left < right:
        return -1 if metric.direction == "min" else 1
    if left > right:
        return 1 if metric.direction == "min" else -1
    return 0


def _metric_kind(value: int | float | str) -> str:
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    raise TypeError(f"unsupported metric value type: {type(value)!r}")


def _canonical_summary(candidate: Mapping[str, object]) -> str:
    normalized = _normalize_jsonish(candidate)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_jsonish(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize_jsonish(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


__all__ = ["SelectionMetric", "rank_exact_candidates", "select_exact_candidate"]
