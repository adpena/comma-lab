# SPDX-License-Identifier: MIT
"""Isolated tests for realization_necessity_preimage_20260715 (locked-registry pattern)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.realization_necessity_preimage_20260715 import (
    CAMERA_SUPPORT_FRAC,
    EQUATION_ID,
    H_LADDER_EDGE_BYTES_N600,
    K_LADDER_EDGE_BYTES_N600_EPS1,
    K_OVER_H,
    build_realization_necessity_preimage_per_stratum_v1,
    necessity_split,
    stratum_rate_floor_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3].parent
ART_DIR = REPO_ROOT / "experiments" / "results" / "necessity_solver_20260715"


def test_rate_floor_law_basic() -> None:
    # 800 cracks at 1.5 bits/step + 2 curves at (17.585 + 2) bits overhead
    got = stratum_rate_floor_bytes(800, 1.5, 2)
    assert got == pytest.approx((800 * 1.5 + 2 * 19.585) / 8.0)


def test_rate_floor_rejects_negative() -> None:
    with pytest.raises(ValueError):
        stratum_rate_floor_bytes(-1, 1.0, 0)


def test_necessity_split_sums_to_one() -> None:
    s = necessity_split()
    assert sum(s.values()) == pytest.approx(1.0, abs=2e-5)
    assert s["strict_necessary"] < 0.05  # bounded by the boundary annulus
    assert s["free_certified"] == pytest.approx(0.227, abs=1e-3)


def test_k_ladder_below_h_ladder() -> None:
    assert K_LADDER_EDGE_BYTES_N600_EPS1 < H_LADDER_EDGE_BYTES_N600
    assert 0.0 < K_OVER_H < 1.0


def test_equation_builds_and_validates() -> None:
    eq = build_realization_necessity_preimage_per_stratum_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.domain_of_validity["research_only"] is True
    assert eq.domain_of_validity["score_claim"] is False
    assert len(eq.empirical_anchors) == 3


def test_constants_match_measured_artifacts() -> None:
    """The module constants must equal the on-disk measured artifacts (no drift)."""
    floors_path = ART_DIR / "floors.json"
    asupport_path = ART_DIR / "asupport.json"
    if not floors_path.exists() or not asupport_path.exists():
        pytest.skip("necessity solver artifacts not present on this checkout")
    floors = json.loads(floors_path.read_text())
    asup = json.loads(asupport_path.read_text())
    assert floors["edges_H_ladder_total_bytes_n600"] == pytest.approx(
        H_LADDER_EDGE_BYTES_N600, rel=1e-6
    )
    assert floors["edges_K_ladder_bytes_n600_eps1"] == pytest.approx(
        K_LADDER_EDGE_BYTES_N600_EPS1, rel=1e-6
    )
    for k_mod, k_art in (
        ("saddle", "saddle"),
        ("edge", "edge"),
        ("cell_loose_membership_only", "cell_loose"),
    ):
        assert CAMERA_SUPPORT_FRAC[k_mod] == pytest.approx(
            asup["camera_frac_mean"][k_art], abs=1e-6
        )
    assert CAMERA_SUPPORT_FRAC["certified_free_zero_weight"] == pytest.approx(
        asup["camera_zero_weight_frac_certified"], abs=1e-6
    )
