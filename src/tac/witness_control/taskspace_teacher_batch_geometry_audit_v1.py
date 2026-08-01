# SPDX-License-Identifier: MIT
"""Exact, chunked comparison of task-space label banks across scorer batches."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    CLASS_COUNT,
    FreshTeacherMaterializationError,
    canonical_json_bytes,
)

SCHEMA: Final = "tac.taskspace_teacher_batch_geometry_comparison.v1"
MAX_EXACT_RECORDS: Final = 4096


def compare_label_banks(
    primary: np.ndarray,
    comparison: np.ndarray,
    *,
    comparison_name: str,
) -> dict[str, Any]:
    """Compare two ``(pair,H,W)`` argmax banks without a full diff tensor."""

    if not comparison_name:
        raise FreshTeacherMaterializationError("comparison_name must be nonempty")
    if primary.ndim != 3 or comparison.ndim != 3 or primary.shape != comparison.shape:
        raise FreshTeacherMaterializationError(
            f"label-bank geometry mismatch: {primary.shape!r} != {comparison.shape!r}"
        )
    if not np.issubdtype(primary.dtype, np.integer) or not np.issubdtype(comparison.dtype, np.integer):
        raise FreshTeacherMaterializationError("label banks must have integer dtypes")

    digest = hashlib.sha256()
    exact_records: list[dict[str, Any]] = []
    first_records: list[dict[str, Any]] = []
    transitions: Counter[tuple[int, int]] = Counter()
    mismatches_by_pair: list[dict[str, int]] = []
    mismatch_count = 0
    for pair_index in range(int(primary.shape[0])):
        primary_pair = np.asarray(primary[pair_index])
        comparison_pair = np.asarray(comparison[pair_index])
        if (
            bool(np.any(primary_pair < 0))
            or bool(np.any(primary_pair >= CLASS_COUNT))
            or bool(np.any(comparison_pair < 0))
            or bool(np.any(comparison_pair >= CLASS_COUNT))
        ):
            raise FreshTeacherMaterializationError("label bank contains a class outside 0..4")
        rows, cols = np.nonzero(primary_pair != comparison_pair)
        pair_mismatches = int(rows.size)
        if pair_mismatches:
            mismatches_by_pair.append({"pair_index": pair_index, "mismatch_count": pair_mismatches})
        mismatch_count += pair_mismatches
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            primary_label = int(primary_pair[row, col])
            comparison_label = int(comparison_pair[row, col])
            record = {
                "pair_index": pair_index,
                "row": int(row),
                "col": int(col),
                "primary_label": primary_label,
                "comparison_label": comparison_label,
            }
            digest.update(canonical_json_bytes(record))
            digest.update(b"\n")
            transitions[(primary_label, comparison_label)] += 1
            if len(exact_records) < MAX_EXACT_RECORDS:
                exact_records.append(record)
            if len(first_records) < 256:
                first_records.append(record)

    total_cells = int(np.prod(primary.shape, dtype=np.int64))
    records_are_complete = mismatch_count <= MAX_EXACT_RECORDS
    return {
        "schema": SCHEMA,
        "comparison_name": comparison_name,
        "shape": [int(value) for value in primary.shape],
        "primary_dtype": str(primary.dtype),
        "comparison_dtype": str(comparison.dtype),
        "total_cells": total_cells,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": format(mismatch_count / total_cells, ".17g"),
        "mismatching_pair_count": len(mismatches_by_pair),
        "mismatches_by_pair": mismatches_by_pair,
        "transition_histogram": [
            {
                "primary_label": primary_label,
                "comparison_label": comparison_label,
                "count": count,
            }
            for (primary_label, comparison_label), count in sorted(transitions.items())
        ],
        "mismatch_records_sha256": digest.hexdigest(),
        "mismatch_records_are_complete": records_are_complete,
        "mismatch_records": exact_records if records_are_complete else first_records,
    }


def seal_batch_geometry_audit(body: Mapping[str, Any]) -> dict[str, Any]:
    if "audit_sha256" in body:
        raise FreshTeacherMaterializationError("batch-geometry audit body is already sealed")
    sealed = dict(body)
    sealed["audit_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return sealed


__all__ = [
    "MAX_EXACT_RECORDS",
    "SCHEMA",
    "compare_label_banks",
    "seal_batch_geometry_audit",
]
