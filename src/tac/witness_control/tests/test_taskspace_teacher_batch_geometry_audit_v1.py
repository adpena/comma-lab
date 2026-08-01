# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    FreshTeacherMaterializationError,
)
from tac.witness_control.taskspace_teacher_batch_geometry_audit_v1 import (
    compare_label_banks,
    seal_batch_geometry_audit,
)


def test_comparison_records_exact_coordinates_transitions_and_digest() -> None:
    primary = np.zeros((2, 2, 3), dtype=np.uint8)
    comparison = primary.copy()
    primary[0, 1, 2] = 4
    comparison[1, 0, 1] = 2

    row = compare_label_banks(primary, comparison, comparison_name="batch4")

    assert row["mismatch_count"] == 2
    assert row["mismatching_pair_count"] == 2
    assert row["mismatch_records_are_complete"] is True
    assert row["mismatch_records"] == [
        {"pair_index": 0, "row": 1, "col": 2, "primary_label": 4, "comparison_label": 0},
        {"pair_index": 1, "row": 0, "col": 1, "primary_label": 0, "comparison_label": 2},
    ]
    assert len(row["mismatch_records_sha256"]) == 64
    assert row["transition_histogram"] == [
        {"primary_label": 0, "comparison_label": 2, "count": 1},
        {"primary_label": 4, "comparison_label": 0, "count": 1},
    ]


def test_comparison_refuses_geometry_or_label_domain_drift() -> None:
    primary = np.zeros((1, 2, 2), dtype=np.uint8)
    with pytest.raises(FreshTeacherMaterializationError, match="geometry mismatch"):
        compare_label_banks(primary, np.zeros((1, 3, 2), dtype=np.uint8), comparison_name="bad")
    invalid = primary.copy()
    invalid[0, 0, 0] = 5
    with pytest.raises(FreshTeacherMaterializationError, match=r"outside 0\.\.4"):
        compare_label_banks(primary, invalid, comparison_name="bad")


def test_sealed_audit_is_deterministic_and_refuses_reseal() -> None:
    first = seal_batch_geometry_audit({"schema": "test", "value": 1})
    second = seal_batch_geometry_audit({"value": 1, "schema": "test"})
    assert first == second
    with pytest.raises(FreshTeacherMaterializationError, match="already sealed"):
        seal_batch_geometry_audit(first)
