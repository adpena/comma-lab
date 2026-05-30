# SPDX-License-Identifier: MIT
"""Tests for canonical Fridrich-school pattern inventory."""

from __future__ import annotations

import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    FRIDRICH_GROUP_ATTRIBUTION,
    FRIDRICH_SCHOOL_ATTRIBUTION,
    FridrichSchoolCanonicalPatternRow,
    YOUSFI_GITHUB_HOMEPAGE,
    build_fridrich_school_canonical_patterns_inventory,
)


def test_inventory_returns_7_canonical_patterns() -> None:
    """Per the Phase A research: 7 canonical patterns extracted."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    assert len(rows) == 7


def test_inventory_pattern_ids_all_distinct() -> None:
    """Slot EEE: every pattern_id MUST be unique."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    ids = [r.pattern_id for r in rows]
    assert len(set(ids)) == len(ids), f"Duplicate pattern_ids: {ids}"


def test_inventory_canonical_pattern_ids_present() -> None:
    """All 7 canonical pattern names from Phase A research."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    ids = {r.pattern_id for r in rows}
    expected = {
        "alice_vs_eve_adversarial_loop",
        "lclsmr_linear_steganalysis_detector",
        "efficientnet_steganalysis_surgery",
        "onehot_jpeg_steganalysis",
        "comma10k_baseline_lineage",
        "syndrome_trellis_coding_filler",
        "fusion_detector_ensemble",
    }
    assert ids == expected


def test_inventory_rows_have_5_axis_classification() -> None:
    """Every row MUST declare 5-axis adaptation per Catalog #303 sister discipline."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    required_axes = {"contest", "problem_space", "math", "data", "video"}
    for row in rows:
        missing = required_axes - set(row.five_axis_classification.keys())
        assert not missing, (
            f"Row {row.pattern_id} missing axes: {missing}"
        )


def test_inventory_rows_have_yousfi_fridrich_axis_cross_ref() -> None:
    """Every row MUST cross-reference the Yousfi-Fridrich cascade."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    for row in rows:
        assert row.cross_reference_yousfi_fridrich_axis
        assert "axis" in row.cross_reference_yousfi_fridrich_axis.lower()


def test_inventory_rows_have_score_lowering_ev_estimate() -> None:
    """Every row MUST declare predicted ΔS band per Catalog #296."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    for row in rows:
        assert row.pr_score_lowering_ev_estimate
        # MUST cite either FORMALIZATION_PENDING / INFRASTRUCTURE-ONLY / DISPATCH-BUDGET
        assert any(
            tok in row.pr_score_lowering_ev_estimate
            for tok in [
                "FORMALIZATION_PENDING",
                "INFRASTRUCTURE-ONLY",
                "DISPATCH-BUDGET",
            ]
        )


def test_inventory_rows_have_upstream_source() -> None:
    """Every row MUST cite upstream source (file:line OR repo URL OR paper DOI)."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    for row in rows:
        assert row.upstream_source
        assert row.upstream_author
        assert row.upstream_paper_or_repo
        assert row.last_updated_or_published


def test_inventory_rows_have_tac_module() -> None:
    """Every row MUST cite the tac module that ports the canonical pattern."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    for row in rows:
        assert row.tac_module.startswith(
            "tac.composition.fridrich_school_inverse_steganalysis_patterns"
        )


def test_yousfi_github_homepage_canonical() -> None:
    """Canonical Yousfi GitHub homepage URL."""
    assert YOUSFI_GITHUB_HOMEPAGE == "https://github.com/YassineYousfi"


def test_fridrich_school_attribution_non_empty() -> None:
    """Attribution text is substantive (not placeholder)."""
    assert FRIDRICH_SCHOOL_ATTRIBUTION
    assert len(FRIDRICH_SCHOOL_ATTRIBUTION) > 100
    assert "Yassine Yousfi" in FRIDRICH_SCHOOL_ATTRIBUTION
    assert "DDE Lab" in FRIDRICH_SCHOOL_ATTRIBUTION


def test_fridrich_group_attribution_lists_all_canonical_students() -> None:
    """Group attribution names all canonical Fridrich PhD students cited."""
    assert "Tomas Filler" in FRIDRICH_GROUP_ATTRIBUTION
    assert "Tomas Pevny" in FRIDRICH_GROUP_ATTRIBUTION
    assert "Vojtech Holub" in FRIDRICH_GROUP_ATTRIBUTION
    assert "Vahid Sedighi" in FRIDRICH_GROUP_ATTRIBUTION
    assert "Mehdi Boroumand" in FRIDRICH_GROUP_ATTRIBUTION
    assert "Jan Kodovsky" in FRIDRICH_GROUP_ATTRIBUTION
    assert "Yassine Yousfi" in FRIDRICH_GROUP_ATTRIBUTION


def test_row_dataclass_frozen() -> None:
    """FridrichSchoolCanonicalPatternRow is frozen (immutable)."""
    rows = build_fridrich_school_canonical_patterns_inventory()
    row = rows[0]
    with pytest.raises((AttributeError, Exception)):
        row.pattern_id = "modified"  # type: ignore[misc]
