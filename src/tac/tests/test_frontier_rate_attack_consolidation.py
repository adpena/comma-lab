# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from comma_lab.scheduler.frontier_rate_attack_consolidation import (
    FRONTIER_RATE_ATTACK_CONSOLIDATION_SCHEMA,
    build_frontier_rate_attack_consolidation_audit,
    render_frontier_rate_attack_consolidation_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_frontier_rate_attack_consolidation_audit_covers_compiler_layers() -> None:
    audit = build_frontier_rate_attack_consolidation_audit(REPO_ROOT)

    assert audit["schema"] == FRONTIER_RATE_ATTACK_CONSOLIDATION_SCHEMA
    assert audit["status"] == "PASS"
    assert audit["score_claim"] is False
    assert audit["promotion_eligible"] is False
    assert audit["ready_for_exact_eval_dispatch"] is False
    assert audit["forbidden_parallel_surfaces"] == []
    assert {row["layer_id"] for row in audit["score_program_layers"]} == {
        "action_candidates",
        "entropy_grammar",
        "payload_and_residual_basis",
    }
    assert all(row["covered"] is True for row in audit["score_program_layers"])
    assert all(row["exists"] is True for row in audit["required_sources"])
    assert all(row["count"] > 0 for row in audit["state_surfaces"])
    assert audit["score_program_dag"]["score_claim"] is False
    assert audit["score_program_dag"]["ready_for_exact_eval_dispatch"] is False
    assert {
        (edge["from"], edge["to"])
        for edge in audit["score_program_dag"]["edges"]
    } >= {
        ("oracle_target_fiber", "action_candidates"),
        ("entropy_grammar", "receiver_parseback_replay"),
        ("receiver_parseback_replay", "exact_eval_handoff"),
    }
    for row in audit["score_program_layers"]:
        assert row["adapter_count"] == len(row["target_kinds"])
        assert row["executable_candidate_archive_count"] > 0
        assert row["receiver_contract_count"] == len(row["target_kinds"])
        assert all(
            adapter["receiver_contract_bound"] is True
            for adapter in row["adapter_rows"]
        )


def test_frontier_rate_attack_consolidation_render_names_formal_and_legacy_surface() -> None:
    audit = build_frontier_rate_attack_consolidation_audit(REPO_ROOT)
    text = render_frontier_rate_attack_consolidation_audit(audit)

    assert "frontier_final_rate_attack_materializer_stack" in text
    assert "score_program_compiler_over_frozen_evaluator_quotient" in text
    assert "action_candidates: covered" in text
    assert "exec_archive=" in text
    assert "receiver_contracts=" in text
    assert "dag:" in text
    assert "entropy_grammar: covered" in text
    assert "payload_and_residual_basis: covered" in text
