# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from operator import index
from typing import TypeVar

DEFAULT_BUCKET_KEY_FIELDS = ("token_count", "first_token", "last_token", "unique_token_count")

SegmentLike = Iterable[object]
SegmentT = TypeVar("SegmentT")


@dataclass(frozen=True)
class SegmentTokenFeatures:
    segment_index: int
    token_count: int
    first_token: int | None
    last_token: int | None
    unique_token_count: int

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.unique_token_count < 0:
            raise ValueError("unique_token_count must be non-negative")
        if self.token_count == 0:
            if self.first_token is not None or self.last_token is not None:
                raise ValueError("empty segments must not define edge tokens")
            return
        if self.first_token is None or self.last_token is None:
            raise ValueError("non-empty segments must define edge tokens")


@dataclass(frozen=True)
class BucketAssignment:
    segment_index: int
    bucket_id: int
    bucket_key: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if self.bucket_id < 0:
            raise ValueError("bucket_id must be non-negative")


@dataclass(frozen=True)
class BucketingPlan:
    key_fields: tuple[str, ...]
    bucket_keys: tuple[tuple[int | None, ...], ...]
    features: tuple[SegmentTokenFeatures, ...]
    assignments: tuple[BucketAssignment, ...]
    ordered_indices: tuple[int, ...]
    restore_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.features)
        if len(self.assignments) != size:
            raise ValueError("assignments must match feature count")
        if len(self.ordered_indices) != size or len(self.restore_indices) != size:
            raise ValueError("permutation sizes must match feature count")
        expected = list(range(size))
        if sorted(self.ordered_indices) != expected:
            raise ValueError("ordered_indices must be a permutation of segment indices")
        if sorted(self.restore_indices) != expected:
            raise ValueError("restore_indices must be a permutation of ordered positions")
        for original_index, ordered_position in enumerate(self.restore_indices):
            if self.ordered_indices[ordered_position] != original_index:
                raise ValueError("restore_indices must invert ordered_indices")


def _normalize_key_fields(key_fields: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(field).strip() for field in key_fields if str(field).strip())
    if not normalized:
        raise ValueError("key_fields must contain at least one field")
    valid_fields = SegmentTokenFeatures.__dataclass_fields__
    for field in normalized:
        if field not in valid_fields:
            raise ValueError(f"unknown bucket key field: {field}")
    return normalized


def _normalize_segment_tokens(segment_tokens: SegmentLike) -> tuple[int, ...]:
    normalized: list[int] = []
    for position, value in enumerate(segment_tokens):
        try:
            token = int(index(value))
        except TypeError as exc:
            raise TypeError(f"token at position {position} must be an integer") from exc
        if token < 0:
            raise ValueError(f"token at position {position} must be non-negative")
        normalized.append(token)
    return tuple(normalized)


def _sortable_bucket_key(bucket_key: tuple[int | None, ...]) -> tuple[tuple[int, int], ...]:
    sortable: list[tuple[int, int]] = []
    for value in bucket_key:
        if value is None:
            sortable.append((0, -1))
        else:
            sortable.append((1, int(value)))
    return tuple(sortable)


def derive_token_segment_features(
    segment_tokens: SegmentLike,
    *,
    segment_index: int,
) -> SegmentTokenFeatures:
    tokens = _normalize_segment_tokens(segment_tokens)
    if tokens:
        first_token = tokens[0]
        last_token = tokens[-1]
        unique_token_count = len(set(tokens))
    else:
        first_token = None
        last_token = None
        unique_token_count = 0
    return SegmentTokenFeatures(
        segment_index=segment_index,
        token_count=len(tokens),
        first_token=first_token,
        last_token=last_token,
        unique_token_count=unique_token_count,
    )


def assign_segments_to_buckets(
    features: Sequence[SegmentTokenFeatures],
    *,
    key_fields: Sequence[str] = DEFAULT_BUCKET_KEY_FIELDS,
) -> tuple[tuple[BucketAssignment, ...], tuple[tuple[int | None, ...], ...]]:
    normalized_fields = _normalize_key_fields(key_fields)
    segment_keys = tuple(
        tuple(getattr(feature, field) for field in normalized_fields)
        for feature in features
    )
    bucket_keys = tuple(sorted(set(segment_keys), key=_sortable_bucket_key))
    bucket_ids = {bucket_key: bucket_id for bucket_id, bucket_key in enumerate(bucket_keys)}
    assignments = tuple(
        BucketAssignment(
            segment_index=feature.segment_index,
            bucket_id=bucket_ids[bucket_key],
            bucket_key=bucket_key,
        )
        for feature, bucket_key in zip(features, segment_keys, strict=True)
    )
    return assignments, bucket_keys


def build_bucketing_plan(
    segments: Sequence[SegmentLike],
    *,
    key_fields: Sequence[str] = DEFAULT_BUCKET_KEY_FIELDS,
) -> BucketingPlan:
    features = tuple(
        derive_token_segment_features(segment, segment_index=segment_index)
        for segment_index, segment in enumerate(segments)
    )
    assignments, bucket_keys = assign_segments_to_buckets(features, key_fields=key_fields)
    ordered_indices = tuple(
        assignment.segment_index
        for assignment in sorted(assignments, key=lambda assignment: (assignment.bucket_id, assignment.segment_index))
    )
    restore = [0] * len(ordered_indices)
    for ordered_position, original_index in enumerate(ordered_indices):
        restore[original_index] = ordered_position
    return BucketingPlan(
        key_fields=_normalize_key_fields(key_fields),
        bucket_keys=bucket_keys,
        features=features,
        assignments=assignments,
        ordered_indices=ordered_indices,
        restore_indices=tuple(restore),
    )


def apply_bucketing_plan(segments: Sequence[SegmentT], plan: BucketingPlan) -> tuple[SegmentT, ...]:
    if len(segments) != len(plan.ordered_indices):
        raise ValueError("segment count must match bucketing plan")
    return tuple(segments[index] for index in plan.ordered_indices)


def restore_bucketed_segments(ordered_segments: Sequence[SegmentT], plan: BucketingPlan) -> tuple[SegmentT, ...]:
    if len(ordered_segments) != len(plan.restore_indices):
        raise ValueError("segment count must match bucketing plan")
    return tuple(ordered_segments[plan.restore_indices[index]] for index in range(len(plan.restore_indices)))


__all__ = [
    "DEFAULT_BUCKET_KEY_FIELDS",
    "BucketAssignment",
    "BucketingPlan",
    "SegmentTokenFeatures",
    "apply_bucketing_plan",
    "assign_segments_to_buckets",
    "build_bucketing_plan",
    "derive_token_segment_features",
    "restore_bucketed_segments",
]
