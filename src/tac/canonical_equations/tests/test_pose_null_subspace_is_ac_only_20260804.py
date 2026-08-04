# SPDX-License-Identifier: MIT
"""Tests for the AC-only pose-null law (ddm_lr2, 2026-08-04)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.pose_null_subspace_is_ac_only_20260804 import (
    EQUATION_ID,
    ac_energy_fraction,
    build_pose_null_subspace_is_ac_only_v1,
    dc_projection_residual,
)


def test_dc_projection_residual_is_zero_to_fp_noise():
    assert dc_projection_residual() < 1e-6


def test_dc_projection_residual_covers_random_constants():
    assert dc_projection_residual(n_random=64, seed=7) < 1e-6


def test_constant_has_zero_ac_energy():
    for c in (np.array([1.0, 0.0, 0.0]), np.array([-37.0, 250.0, 4.0])):
        assert ac_energy_fraction(np.tile(c, 4)) < 1e-12


def test_random_vector_has_partial_ac_energy():
    rng = np.random.default_rng(3)
    frac = ac_energy_fraction(rng.normal(size=12))
    assert 0.0 < frac < 1.0


def test_null_space_vector_has_full_ac_energy():
    # Build a vector IN the null space via the projector itself; its fraction must be ~1.
    from tac.canonical_equations.pose_null_subspace_is_ac_only_20260804 import (
        _constraint_matrix,
    )

    a = _constraint_matrix()
    proj = np.eye(12) - np.linalg.pinv(a) @ a
    rng = np.random.default_rng(5)
    d = proj @ rng.normal(size=12)
    assert ac_energy_fraction(d) == pytest.approx(1.0, abs=1e-9)


def test_zero_vector_reports_zero():
    assert ac_energy_fraction(np.zeros(12)) == 0.0


def test_equation_builds_with_anchor_and_consumers():
    eq = build_pose_null_subspace_is_ac_only_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.empirical_anchors and eq.empirical_anchors[0].residual == 0.0
    assert len(eq.canonical_consumers) >= 3
    assert eq.domain_of_validity["score_claim"] is False
