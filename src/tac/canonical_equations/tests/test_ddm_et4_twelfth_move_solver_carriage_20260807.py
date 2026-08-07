# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_et4_twelfth_move_solver_carriage_20260807 import (
    EQUATION_ID,
    W_BREAK_EVEN_BYTES_PER_FLIP,
    build_ddm_et4_twelfth_move_solver_carriage_split_v1,
    et4_solver_carriage_split,
    et4_solver_carriage_split_from_receipts,
)


def test_w_break_even_matches_campaign_constant() -> None:
    assert W_BREAK_EVEN_BYTES_PER_FLIP == pytest.approx(1.27310821533, rel=0, abs=1e-11)


def test_et4_receipts_recompute_solver_reach_and_carriage_failure() -> None:
    split = et4_solver_carriage_split_from_receipts()
    assert split["solver_reduces_d_seg"] is True
    assert split["patch_b_per_flip"] == pytest.approx(144.1435978646778)
    assert split["patch_over_break_even_ratio"] == pytest.approx(113.22179538923534)
    assert split["nnz_per_flip"] == pytest.approx(122.7733391228832)


def test_helper_rejects_zero_flip_fake() -> None:
    with pytest.raises(ValueError, match="net_flip_reduction"):
        et4_solver_carriage_split(
            net_flip_reduction=0,
            patch_compressed_bytes=1,
            patch_nnz=1,
            baseline_d_seg=0.1,
            realized_d_seg=0.0,
            archive_delta_bytes=1,
        )


def test_equation_builds_non_promoting_anchor() -> None:
    eq = build_ddm_et4_twelfth_move_solver_carriage_split_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.domain_of_validity["score_claim"] is False
    assert eq.provenance.promotion_eligible is False
    assert eq.empirical_anchors[0].empirical_output["pointer_moved"] is False
