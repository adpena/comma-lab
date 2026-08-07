# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_et5_carriage_family_fold_20260807 import (
    EQUATION_ID,
    build_ddm_et5_restricted_carriage_family_fold_v1,
    restricted_carriage_fold_decision,
    restricted_carriage_fold_decision_from_receipt,
)


def test_et5_receipt_folds_with_zero_waterfill() -> None:
    decision = restricted_carriage_fold_decision_from_receipt()
    assert decision["folded"] is True
    assert decision["selected_count"] == 0
    assert decision["over_waterline_ratio"] == pytest.approx(66.35415552642162)


def test_et5_helper_routes_eligible_candidate_to_materialization() -> None:
    decision = restricted_carriage_fold_decision(
        best_b_per_full_patch_flip=1.0,
        waterline_b_per_flip=1.27310821533,
        waterfill_selected_count=3,
        realization_ran=False,
    )
    assert decision["folded"] is False
    assert "materialize" in str(decision["owed_reopen_condition"])


def test_equation_builds_instance_scope_negative() -> None:
    eq = build_ddm_et5_restricted_carriage_family_fold_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.empirical_anchors[0].empirical_output["waterfill_selected_count"] == 0
    assert eq.domain_of_validity["score_claim"] is False
    assert eq.provenance.score_claim_valid is False
