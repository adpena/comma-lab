# SPDX-License-Identifier: MIT
"""Focused tests for the SPD-cone water-filled pose-section codec rate canonical equation."""
from __future__ import annotations

import numpy as np

from tac.canonical_equations.pose_spd_cone_waterfill_rate_20260710 import (
    EQUATION_ID,
    build_pose_spd_cone_waterfill_rate_v1 as build_eq,
    hilbert_projective_distance,
)


def test_equation_builds_and_validates() -> None:
    eq = build_eq()
    assert eq.equation_id == EQUATION_ID == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 1
    a = eq.empirical_anchors[0]
    # the MEASURED Pareto win on the real pose section (matched-MSE, fewer bytes)
    assert a.empirical_output["spd_matched_mse_bytes"] < a.inputs["baseline_bytes"]
    assert a.empirical_output["byte_fraction_saved"] > 0.15
    # non-orphan contract: producer (the codec) + consumers (the archive build/parse path)
    assert eq.canonical_producers and eq.canonical_consumers
    assert any("build_archive_with_pose" in c for c in eq.canonical_consumers)
    # advisory / non-promotable provenance (rate law on the section, NOT a d_pose/score claim)
    assert eq.provenance.promotion_eligible is False


def test_hilbert_distance_matches_anchor_and_is_monotone() -> None:
    eq = build_eq()
    eigs = eq.empirical_anchors[0].inputs["cov_eigenvalues"]
    dH = hilbert_projective_distance(eigs)
    assert abs(dH - eq.empirical_anchors[0].inputs["hilbert_projective_distance"]) < 0.05
    # anisotropy monotonicity: wider spectrum -> larger d_H
    assert hilbert_projective_distance([100.0, 1.0]) > hilbert_projective_distance([2.0, 1.0])
    assert hilbert_projective_distance([1.0, 1.0]) == 0.0
    # degenerate (all filtered) -> 0
    assert hilbert_projective_distance(np.zeros(3)) == 0.0
