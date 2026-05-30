# SPDX-License-Identifier: MIT
"""Tests for canonical ALASKA pattern inventory + cross-reference matrix."""

from __future__ import annotations

import pytest

from tac.composition.alaska_inverse_steganalysis_patterns import (
    ALASKA_ORIGIN_PAPER_CITATION,
    ALASKA_REPO_ATTRIBUTION,
    AlaskaCanonicalPatternRow,
    build_alaska_canonical_patterns_inventory,
)


def test_attribution_cites_dde_lab_binghamton() -> None:
    """Per Yousfi LICENSE.md upstream: must cite DDE Lab + Binghamton."""
    assert "DDE Lab" in ALASKA_REPO_ATTRIBUTION
    assert "Binghamton University" in ALASKA_REPO_ATTRIBUTION
    assert "external/alaska_yousfi/" in ALASKA_REPO_ATTRIBUTION


def test_paper_citation_canonical() -> None:
    """Per Yousfi README.md upstream: canonical paper citation."""
    assert "Yousfi" in ALASKA_ORIGIN_PAPER_CITATION
    assert "Butora" in ALASKA_ORIGIN_PAPER_CITATION
    assert "Fridrich" in ALASKA_ORIGIN_PAPER_CITATION
    assert "Giboulot" in ALASKA_ORIGIN_PAPER_CITATION
    assert "2019" in ALASKA_ORIGIN_PAPER_CITATION
    assert "10.1145/3335203.3335727" in ALASKA_ORIGIN_PAPER_CITATION


def test_inventory_returns_6_patterns() -> None:
    """6 canonical patterns extracted from ALASKA."""
    inv = build_alaska_canonical_patterns_inventory()
    assert len(inv) == 6


def test_inventory_all_rows_typed() -> None:
    inv = build_alaska_canonical_patterns_inventory()
    for row in inv:
        assert isinstance(row, AlaskaCanonicalPatternRow)


def test_inventory_pattern_ids_distinct() -> None:
    """Slot EEE substantive-distinctness: 6 distinct pattern_id values."""
    inv = build_alaska_canonical_patterns_inventory()
    ids = [row.pattern_id for row in inv]
    assert len(set(ids)) == 6


def test_inventory_canonical_pattern_ids_present() -> None:
    """6 canonical patterns enumerated by name."""
    inv = build_alaska_canonical_patterns_inventory()
    ids = {row.pattern_id for row in inv}
    assert ids == {
        "color_separation_5_branch",
        "pair_constraint_batch",
        "multi_scheme_dirichlet_prior",
        "detector_aware_iterative_training",
        "cmd_per_image_4_stat_discrimination",
        "warm_start_single_to_multi_branch",
    }


def test_inventory_5_axis_classification_complete() -> None:
    """Per CLAUDE.md "15-item-audit 1:1 fidelity with documented adaptations"
    standing directive: EVERY pattern MUST classify all 5 axes."""
    inv = build_alaska_canonical_patterns_inventory()
    required = {"contest", "problem_space", "math", "data", "video"}
    for row in inv:
        assert set(row.five_axis_classification.keys()) == required, (
            f"pattern_id={row.pattern_id} missing axes: "
            f"{required - set(row.five_axis_classification.keys())}"
        )


def test_inventory_each_row_cites_upstream_source() -> None:
    """Every row MUST cite the upstream alaska clone path + file:line range."""
    inv = build_alaska_canonical_patterns_inventory()
    for row in inv:
        assert "external/alaska_yousfi/" in row.upstream_source, (
            f"row {row.pattern_id} upstream_source={row.upstream_source!r} "
            f"missing canonical clone path"
        )


def test_inventory_each_row_points_to_tac_module() -> None:
    """Every row MUST point to a tac module path."""
    inv = build_alaska_canonical_patterns_inventory()
    for row in inv:
        assert row.tac_module.startswith(
            "tac.composition.alaska_inverse_steganalysis_patterns"
        )


def test_inventory_each_row_has_cross_reference_axis() -> None:
    """Every row MUST cite a Yousfi-Fridrich-cascade axis cross-reference."""
    inv = build_alaska_canonical_patterns_inventory()
    for row in inv:
        assert "Axis" in row.cross_reference_yousfi_fridrich_axis


def test_inventory_each_row_has_pr_ev_estimate() -> None:
    """Every row MUST estimate predicted ΔS band or carry FORMALIZATION_PENDING."""
    inv = build_alaska_canonical_patterns_inventory()
    for row in inv:
        ev = row.pr_score_lowering_ev_estimate
        assert any(token in ev for token in ("[", "FORMALIZATION_PENDING", "EV-NOT-SCORE"))


def test_inventory_classification_substantively_varied() -> None:
    """Slot EEE substantive-distinctness: across 6 patterns, the
    classifications MUST NOT all be identical (would indicate fake
    classification).
    """
    inv = build_alaska_canonical_patterns_inventory()
    # Collapse each row's 5-axis classification to a frozenset of (axis, classification) tuples
    frozen = {
        frozenset(row.five_axis_classification.items()) for row in inv
    }
    # At least 5 of 6 must be unique (allowing 1 accidental match)
    assert len(frozen) >= 5, (
        f"5-axis classifications too uniform across patterns: {len(frozen)}/6 distinct"
    )
