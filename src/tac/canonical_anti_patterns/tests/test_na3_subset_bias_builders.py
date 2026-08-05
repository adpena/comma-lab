# SPDX-License-Identifier: MIT
"""Tests for NA3 subset-bias canonical anti-pattern builders."""
from __future__ import annotations

from pathlib import Path

from tac.canonical_anti_patterns import (
    PARADIGM_DIAGNOSIS,
    PARADIGM_RIGOR_LOSS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    build_all_na3_subset_bias_anti_patterns,
    build_prefix_bias_sign_inversion_pose_axis_v1,
    build_subset_default_silent_under_sampling_v1,
    populate_na3_subset_bias_anti_patterns,
)
from tac.canonical_anti_patterns.registry import (
    query_anti_patterns,
    query_anti_patterns_by_substrate,
)


def test_na3_subset_bias_builders_construct_valid_anti_patterns() -> None:
    anti_patterns = build_all_na3_subset_bias_anti_patterns()
    ids = {ap.anti_pattern_id for ap in anti_patterns}
    assert ids == {
        "prefix_bias_sign_inversion_pose_axis_v1",
        "subset_default_silent_under_sampling_v1",
    }
    assert all(len(ap.description) <= 600 for ap in anti_patterns)
    assert all(ap.provenance.promotion_eligible is False for ap in anti_patterns)
    assert all(ap.provenance.score_claim_valid is False for ap in anti_patterns)


def test_prefix_bias_sign_inversion_band_is_axis_specific() -> None:
    ap = build_prefix_bias_sign_inversion_pose_axis_v1()
    assert ap.paradigm_class == PARADIGM_DIAGNOSIS
    assert ap.severity == SEVERITY_HIGH
    assert ap.falsification_band["population_pairs"] == 600.0
    assert ap.falsification_band["pose_prefix_ratio_n24"] == 2.535475579649216
    assert ap.falsification_band["pose_prefix_ratio_n96"] == 4.206770932037034
    assert "pose" in ap.forbidden_pattern_predicate
    assert "INSTANCE_ON_PREFIX" in ap.canonical_unwind_path


def test_subset_default_under_sampling_band_requires_explicit_provenance() -> None:
    ap = build_subset_default_silent_under_sampling_v1()
    assert ap.paradigm_class == PARADIGM_RIGOR_LOSS
    assert ap.severity == SEVERITY_CRITICAL
    assert ap.falsification_band["slice_tool_count"] == 110.0
    assert ap.falsification_band["representative_selector_count"] == 0.0
    assert ap.falsification_band["silent_same_line_candidate_fraction"] == 0.715
    assert "selection_provenance.mode IS NULL" in ap.forbidden_pattern_predicate


def test_populate_na3_subset_bias_anti_patterns_uses_locked_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anti_patterns.jsonl"
    lock_path = tmp_path / "anti_patterns.jsonl.lock"
    populate_na3_subset_bias_anti_patterns(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        notes="temp registry test",
    )

    anti_patterns = query_anti_patterns(path=path)
    assert {ap.anti_pattern_id for ap in anti_patterns} == {
        "prefix_bias_sign_inversion_pose_axis_v1",
        "subset_default_silent_under_sampling_v1",
    }

    subset_matches = query_anti_patterns_by_substrate("subset_selection", path=path)
    assert {ap.anti_pattern_id for ap in subset_matches} == {
        "prefix_bias_sign_inversion_pose_axis_v1",
        "subset_default_silent_under_sampling_v1",
    }
