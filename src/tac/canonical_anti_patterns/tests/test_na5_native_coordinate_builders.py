# SPDX-License-Identifier: MIT
"""Tests for NA5 native-coordinate anti-pattern builders."""
from __future__ import annotations

from pathlib import Path

from tac.canonical_anti_patterns import (
    PARADIGM_PROVENANCE,
    SEVERITY_CRITICAL,
    build_all_na5_native_coordinate_anti_patterns,
    build_lossy_projection_shipped_expecting_decode_realization_v1,
    populate_na5_native_coordinate_anti_patterns,
)
from tac.canonical_anti_patterns.registry import (
    query_anti_patterns,
    query_anti_patterns_by_substrate,
)


def test_na5_native_coordinate_builder_constructs_valid_anti_pattern() -> None:
    anti_patterns = build_all_na5_native_coordinate_anti_patterns()
    assert [ap.anti_pattern_id for ap in anti_patterns] == [
        "lossy_projection_shipped_expecting_decode_realization_v1"
    ]
    ap = anti_patterns[0]
    assert len(ap.description) <= 600
    assert ap.provenance.promotion_eligible is False
    assert ap.provenance.score_claim_valid is False


def test_lossy_projection_band_preserves_addendum_8_counts() -> None:
    ap = build_lossy_projection_shipped_expecting_decode_realization_v1()
    assert ap.paradigm_class == PARADIGM_PROVENANCE
    assert ap.severity == SEVERITY_CRITICAL
    assert ap.falsification_band["measured_instances_total"] == 8.0
    assert ap.falsification_band["dead_projection_instances"] == 6.0
    assert ap.falsification_band["native_coordinate_success_instances"] == 2.0
    assert "native DOF" in ap.canonical_unwind_path
    assert "payload_schema_declared_before_run" in ap.forbidden_pattern_predicate


def test_populate_na5_native_coordinate_anti_patterns_uses_locked_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anti_patterns.jsonl"
    lock_path = tmp_path / "anti_patterns.jsonl.lock"
    populate_na5_native_coordinate_anti_patterns(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        notes="temp native-coordinate registry test",
    )

    anti_patterns = query_anti_patterns(path=path)
    assert {ap.anti_pattern_id for ap in anti_patterns} == {
        "lossy_projection_shipped_expecting_decode_realization_v1"
    }

    matches = query_anti_patterns_by_substrate("native", path=path)
    assert [ap.anti_pattern_id for ap in matches] == [
        "lossy_projection_shipped_expecting_decode_realization_v1"
    ]
