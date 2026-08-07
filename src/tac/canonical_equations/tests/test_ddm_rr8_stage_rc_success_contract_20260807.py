# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.canonical_equations.ddm_rr8_stage_rc_success_contract_20260807 import (
    EQUATION_ID,
    build_ddm_rr8_stage_rc_success_contract_v1,
    stage_chain_success,
)


def test_done_rc_zero_does_not_hide_failed_final_stage() -> None:
    status = stage_chain_success(
        detached_done_rc=0,
        stage_rcs={"shard_0": 0, "shard_1": 0, "shard_2": 0, "final": 1},
    )
    assert status["success"] is False
    assert status["false_success_if_done_only"] is True
    assert status["failed_stages"] == ("final",)


def test_all_stage_rcs_zero_is_success() -> None:
    status = stage_chain_success(detached_done_rc=0, stage_rcs={"a": 0, "b": 0})
    assert status["success"] is True
    assert status["failed_stages"] == ()


def test_equation_builds_source_inspection_anchor() -> None:
    eq = build_ddm_rr8_stage_rc_success_contract_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.empirical_anchors[0].empirical_output["false_success_receipt_observed"] is True
    assert eq.domain_of_validity["score_claim"] is False
