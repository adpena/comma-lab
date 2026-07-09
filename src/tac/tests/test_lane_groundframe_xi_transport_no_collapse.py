# SPDX-License-Identifier: MIT
"""Tests for the lane ground-frame xi-transport NO-COLLAPSE canonical equation (FEED-v8-lane-xi).

EQUATIONS leg of a MEASURED NEGATIVE (a NO-GO): ego-advected predictive coding LOSES to the identity
temporal delta on a ground-frame-canonicalized lane chart. DSL leg is N/A (a measurement of a negative
lever -- no trainer lever / launch / curriculum change), so these tests lock ONLY the equations leg:
the equation builds, is non-orphan, carries both accounting-mode anchors with the memo's MEASURED
numbers, and registers into the canonical registry idempotently. All $0, no GPU, no mlx.
"""
from __future__ import annotations

import pytest

from tac.canonical_equations import (
    CanonicalEquation,
    build_lane_groundframe_xi_transport_no_collapse_v1,
    get_equation_by_id,
    populate_lane_groundframe_xi_transport_no_collapse_equation,
)
from tac.canonical_equations.equation import VERIFIED_VIA_EMPIRICAL_ANCHOR
from tac.canonical_equations.lane_groundframe_xi_transport_no_collapse_20260709 import (
    EQUATION_ID,
)


# --- build + structural invariants ------------------------------------------
def test_builds_and_is_valid_canonical_equation():
    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    assert isinstance(eq, CanonicalEquation)
    assert eq.equation_id == EQUATION_ID == "lane_groundframe_xi_transport_no_collapse_v1"


def test_no_orphan_producers_and_consumers():
    # producer/consumer non-orphan invariant (CanonicalEquation.__post_init__ also enforces).
    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    assert eq.canonical_producers and eq.canonical_consumers
    # the two REAL producer modules cited by the memo.
    assert "tac.boundary_math.ego_xi_trajectory" in eq.canonical_producers
    assert "tac.boundary_math.analytic_lane_render_band" in eq.canonical_producers


def test_producer_modules_are_importable():
    # the cited producer surfaces exist (not invented paths).
    import importlib

    importlib.import_module("tac.boundary_math.ego_xi_trajectory")
    m = importlib.import_module("tac.boundary_math.analytic_lane_render_band")
    assert hasattr(m, "serialize_lane_band_rd3")  # the LBND3 coder surface the callable cites


def test_advisory_non_promotable_and_formulation_scope():
    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    dov = eq.domain_of_validity
    assert dov["promotion_eligible"] is False
    assert dov["measurement_axis"] == ["macOS-CPU advisory"]
    assert "FORMULATION" in dov["verdict_scope"]  # not FAMILY / not PARADIGM


# --- the two accounting-mode anchors carry the MEASURED numbers -------------
def test_two_empirical_anchors_verified_status():
    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    assert len(eq.empirical_anchors) == 2
    for a in eq.empirical_anchors:
        assert a.empirical_verification_status == VERIFIED_VIA_EMPIRICAL_ANCHOR
        assert a.residual == 0.0  # MEASURED, not predicted


def test_isolated_payload_anchor_identity_beats_best_xi():
    # the decisive n600 numbers (isolated lane payload): identity < every xi on BOTH bytes and L1.
    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    a = next(a for a in eq.empirical_anchors if "isolated_payload" in a.anchor_id)
    out = a.empirical_output
    assert out["identity_bytes"] == 41_085
    assert out["identity_L1"] == 7_584_060
    assert out["xi_affine_bytes"] == 42_017 > out["identity_bytes"]        # bytes: xi larger
    assert out["xi_affine_L1"] == 8_252_100 > out["identity_L1"]           # L1: xi larger
    assert out["xi_laneopt_L1"] == 9_983_228 > out["identity_L1"]          # achievable-floor still worse


def test_full_blob_anchor_every_xi_arm_worse():
    eq = build_lane_groundframe_xi_transport_no_collapse_v1()
    a = next(a for a in eq.empirical_anchors if "full_blob" in a.anchor_id)
    out = a.empirical_output
    base = out["lbnd2_baseline_S"]
    assert base == pytest.approx(0.02750)
    for key in ("xi_affine_S", "xi_geometry_S", "xi_laneopt_S"):
        assert out[key] > base  # every xi arm enlarges S


# --- registration (registry grep count) -------------------------------------
def test_populate_registers_into_registry(tmp_path):
    path = tmp_path / "canonical_equations_registry.jsonl"
    lock = tmp_path / "canonical_equations_registry.lock"
    eq = populate_lane_groundframe_xi_transport_no_collapse_equation(
        path=path, lock_path=lock, agent="test", subagent_id="test-xi-nocollapse",
    )
    assert eq.equation_id == EQUATION_ID
    # idempotent APPEND-ONLY: latest-row-wins readback returns the registered equation.
    got = get_equation_by_id(EQUATION_ID, path=path)
    assert got is not None
    assert got.equation_id == EQUATION_ID
    assert len(got.empirical_anchors) == 2
