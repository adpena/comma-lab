# SPDX-License-Identifier: MIT
"""Focused tests for the #205 STORE-NOTHING-but-xi pose-carrier rate/d_pose canonical equation."""
from __future__ import annotations

import pytest

from tac.canonical_equations.store_nothing_pose_carrier_rate_dpose_20260702 import (
    EQUATION_ID,
    build_store_nothing_pose_carrier_rate_collapse_vs_dpose_v1 as build_eq,
    store_nothing_marginal_bytes,
    store_nothing_rate_term,
)


def test_equation_builds_and_validates() -> None:
    eq = build_eq()
    assert eq.equation_id == EQUATION_ID == eq.equation_id.lower()
    # 3rd anchor appended 2026-07-07: the se3_bspline first-firing rate-vs-error curve
    # (spline direct replacement DEAD; spline-predictive residual floor 2182 B vs 3200 B table).
    # 4th anchor appended 2026-07-07: the REAL residual coder — delta_res (no spline) wins at
    # 2714 B vs 3200 B (-486 B, bit-identical); spline predictor MEASURED DEAD as a rate lever.
    assert len(eq.empirical_anchors) == 4
    assert eq.empirical_anchors[2].anchor_id == (
        "se3_bspline_xi_rate_error_curve_first_firing_n600_20260707")
    assert eq.empirical_anchors[3].anchor_id == (
        "xi_spline_residual_real_coder_delta_res_wins_n600_20260707")
    coder_emp = eq.empirical_anchors[3].empirical_output
    assert coder_emp["delta_res_no_spline_bytes"] == 2714
    assert coder_emp["bit_identical_to_shipped_table"] is True
    assert coder_emp["delta_res_no_spline_bytes"] < coder_emp["baseline_delta_ar_bytes"]
    # non-orphan contract: producers AND consumers present
    assert eq.canonical_producers and eq.canonical_consumers
    assert "tools/levelset_byte_close_and_eval.py" in eq.canonical_producers
    # the DSL PoseGauge + the launch config are declared consumers
    assert "tac.witness_dsl.gauge" in eq.canonical_consumers
    assert "tac.witness_autoconfig" in eq.canonical_consumers


def test_store_nothing_marginal_bytes_xi_only() -> None:
    # byte-optimal xi-only = n_pairs * 6 dims * fp16 (2 B) = 12 B/pair.
    assert store_nothing_marginal_bytes(600) == 7200
    assert store_nothing_marginal_bytes(6) == 72


def test_store_nothing_rate_term_is_near_zero_marginal() -> None:
    # 25 * 7200 / 37_545_489 ~= 0.0048 -- ~0 marginal vs the stored-keyframe rate.
    assert store_nothing_rate_term(600) == pytest.approx(0.004794, abs=1e-5)


def test_rate_anchor_records_the_measured_bit_exact_collapse() -> None:
    eq = build_eq()
    rate_anchor = eq.empirical_anchors[0]
    emp = rate_anchor.empirical_output
    # the byte-close MEASURED (BIT-EXACT) section collapse + frame0 bit-exactness
    assert emp["section_bytes_warp_real_luma_ds4"] == 697941
    assert emp["section_bytes_store_nothing"] == 1049
    assert emp["frame0_decode_max_abs_uint8_diff"] == 0  # BIT-EXACT
    assert emp["rate_term_store_nothing"] < emp["rate_term_warp_real_luma_ds4"]


def test_dpose_anchor_is_preresidual_and_upper_bounded() -> None:
    eq = build_eq()
    dpose_anchor = eq.empirical_anchors[1]
    emp = dpose_anchor.empirical_output
    # Track B classmean proxy pre-residual: lf0 is the upper bound; +lf24 beats the full real keyframe.
    assert emp["dpose_classmean_lf0"] == pytest.approx(4.975)
    assert emp["dpose_classmean_lf24"] < emp["dpose_full_real_keyframe"]
    # honest: the post-residual d_pose is #205-gated (no borrowed number)
    assert "#205-gated" in emp["verdict"]
