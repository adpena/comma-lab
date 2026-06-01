# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import pr101_optimal_grammar_campaign_consumer as consumer
from tac.packet_compiler.pr101_per_tensor_grammar_solver import (
    PR101_GRAMMAR_CAMPAIGN_SUMMARY_SCHEMA,
    contest_rate_term,
)


def _summary(
    *,
    verdict: str = "grouped_positive_consumed_by_archive_overhead",
    archive_delta: int = 3,
    archive_rate_positive: bool = False,
    runtime_compatible: bool = True,
    receiver_work_justified: bool = False,
) -> dict[str, object]:
    return {
        "schema": PR101_GRAMMAR_CAMPAIGN_SUMMARY_SCHEMA,
        "campaign_id": "fixture",
        "verdict": verdict,
        "next_action": "fixture_next_action",
        "rate_axis": {
            "archive_zip_delta_bytes": archive_delta,
            "archive_rate_positive": archive_rate_positive,
            "grouped_saved_bytes_vs_current_stock": 1,
            "saturation_status": "entropy_saturated",
        },
        "artifact_status": {
            "runtime_tree_compatible_with_archive_layout": runtime_compatible,
        },
        "planner_feedback": {
            "grouped_positive": True,
            "receiver_adapter_work_justified": receiver_work_justified,
            "grammar_payoff_is_substrate_conditional": True,
        },
        "blockers": ["full_frame_inflate_parity_missing"],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_pr101_optimal_grammar_campaign_consumer_contract() -> None:
    reg = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.pr101_optimal_grammar_campaign_consumer",
    )

    assert reg.contract_compliant is True
    assert consumer.CONSUMER_NAME == "pr101_optimal_grammar_campaign_consumer"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert consumer.update_from_anchor({"anchor": "ignored"}) is None


def test_consumer_demotes_archive_overhead_without_score_authority() -> None:
    verdict = consumer.consume_candidate(_summary())

    assert verdict["schema"] == "pr101_optimal_grammar_campaign_consumer_result.v1"
    assert verdict["planner_action"] == (
        "record_negative_rate_posterior_and_demote_format_churn"
    )
    assert verdict["demotion_recommended"] is True
    assert verdict["local_replay_recommended"] is False
    assert verdict["candidate_archive_bytes_delta"] == 3
    assert verdict["rate_delta_score_if_components_unchanged"] == contest_rate_term(3)
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["score_claim"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False
    assert "full_frame_inflate_parity_missing" in verdict["blockers"]


def test_consumer_routes_archive_positive_runtime_ready_summary_to_local_replay() -> None:
    verdict = consumer.consume_candidate(
        _summary(
            verdict="grouped_positive_runtime_ready_for_local_replay_gate",
            archive_delta=-7,
            archive_rate_positive=True,
            runtime_compatible=True,
        )
    )

    assert verdict["planner_action"] == "run_full_frame_inflate_parity_and_local_replay_gate"
    assert verdict["local_replay_recommended"] is True
    assert verdict["demotion_recommended"] is False
    assert verdict["receiver_adapter_work_justified"] is False
    assert verdict["candidate_archive_bytes_delta"] == -7
    assert verdict["rate_delta_score_if_components_unchanged"] == contest_rate_term(-7)
    assert verdict["promotable"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_consumer_preserves_false_authority_and_schema_blockers() -> None:
    payload = _summary()
    payload["schema"] = "wrong"
    payload["ready_for_exact_eval_dispatch"] = True

    verdict = consumer.consume_candidate(payload)

    assert verdict["local_replay_recommended"] is False
    assert "campaign_summary_schema_mismatch" in verdict["blockers"]
    assert "ready_for_exact_eval_dispatch_overclaimed" in verdict["blockers"]
    assert verdict["ready_for_exact_eval_dispatch"] is False
