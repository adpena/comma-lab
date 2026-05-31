# SPDX-License-Identifier: MIT
"""Tests for the provenance-clean predictive-coding stack-of-stacks contract."""

from __future__ import annotations

import pytest

from tac.substrates.predictive_coding_stack_of_stacks import (
    CANONICAL_STACK_MEMBER_IDS,
    PREDICTIVE_CODING_STACK_MEMBER_ROW_SCHEMA,
    PREDICTIVE_CODING_STACK_OF_STACKS_SCHEMA,
    PredictiveCodingStackError,
    build_predictive_coding_stack_of_stacks_plan,
)


def test_default_stack_uses_only_provenance_clean_members() -> None:
    plan = build_predictive_coding_stack_of_stacks_plan()

    assert plan["schema"] == PREDICTIVE_CODING_STACK_OF_STACKS_SCHEMA
    assert tuple(plan["canonical_member_ids"]) == CANONICAL_STACK_MEMBER_IDS
    assert plan["provenance_clean"] is True
    assert plan["compound_c_leakage_detected"] is False
    assert plan["stack_executable"] is True
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert all(
        row["schema"] == PREDICTIVE_CODING_STACK_MEMBER_ROW_SCHEMA
        for row in plan["member_rows"]
    )
    assert {row["member_id"] for row in plan["member_rows"]} == set(
        CANONICAL_STACK_MEMBER_IDS
    )


def test_strict_stack_rejects_falsified_compound_c_leakage() -> None:
    plan = build_predictive_coding_stack_of_stacks_plan(
        ["z8", "pact_nerv_compound_c", "z6"],
    )

    assert plan["provenance_clean"] is False
    assert plan["compound_c_leakage_detected"] is True
    assert plan["stack_executable"] is False
    assert "falsified_stack_member_rejected:compound_c" in plan["blockers"]
    assert plan["canonical_member_ids"] == [
        "z8_hierarchical_predictive_coding",
        "z6_v2_cargo_cult_unwind",
    ]

    with pytest.raises(PredictiveCodingStackError, match="compound_c"):
        build_predictive_coding_stack_of_stacks_plan(
            ["z8", "compound_c"],
            strict=True,
        )


def test_exact_bridge_requirement_blocks_archive_bytes_only_members() -> None:
    plan = build_predictive_coding_stack_of_stacks_plan(
        ["z8", "dreamer", "z7", "z6", "z4"],
        require_archive_bound_bridge=True,
    )

    assert plan["provenance_clean"] is True
    assert plan["archive_bound_bridge_complete"] is False
    assert plan["stack_executable"] is False
    assert "z8_hierarchical_predictive_coding_archive_bound_bridge_missing" in (
        plan["blockers"]
    )
    assert "dreamer_v3_rssm_archive_bound_bridge_missing" in plan["blockers"]
    bridge_ready = {
        row["member_id"]: row["archive_bound_bridge_ready"]
        for row in plan["member_rows"]
    }
    assert bridge_ready["z7_mamba2"] is True
    assert bridge_ready["z6_v2_cargo_cult_unwind"] is True
    assert bridge_ready["z4_atick_redlich"] is True
