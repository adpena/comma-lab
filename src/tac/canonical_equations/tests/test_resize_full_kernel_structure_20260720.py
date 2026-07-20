# SPDX-License-Identifier: MIT
"""Tests for the full-resize-kernel canonical equation."""

from __future__ import annotations

from tac.canonical_equations.resize_full_kernel_structure_20260720 import (
    EQUATION_ID,
    build_separable_resize_full_kernel_direct_sum_v1,
    full_resize_kernel_direct_sum,
)


def test_law_returns_exact_contest_dimensions():
    law = full_resize_kernel_direct_sum()
    assert law["equation_id"] == EQUATION_ID
    assert law["full_nullity_per_channel"] == 820_728
    assert law["old_zero_weight_nullity_per_channel"] == 230_904
    assert law["orthogonal_direct_sum"] is True
    assert law["score_claim"] is False


def test_equation_is_nonpromotable_and_routes_consumers():
    equation = build_separable_resize_full_kernel_direct_sum_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["exact_resize_numerator_equal"] is True
    assert anchor.empirical_output["selected_name"] == "old_zero_weight_mask"
    assert anchor.empirical_output["score_claim"] is False
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert "R1 d_B preimage-cell compiler" in equation.canonical_consumers
    assert equation.provenance.promotion_eligible is False
