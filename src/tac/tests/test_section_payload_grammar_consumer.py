# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.archive_byte_profile import contest_rate_term
from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import section_payload_grammar_consumer as consumer
from tac.packet_compiler.section_payload_grammar_optimizer import (
    SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
)


def _report(
    *,
    status: str = "entropy_saturated",
    saved: int = 37,
    over_floor: float = 1.02,
    grouped_saved: int = 0,
) -> dict[str, object]:
    grouped_delta = -int(grouped_saved)
    return {
        "schema": SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
        "campaign_id": "fixture",
        "section_count": 2,
        "source_payload_manifest": {
            "source_kind": "single_member_zip_archive",
            "section_count": 2,
        },
        "byte_accounting": {
            "selected_isolated_section_bytes": 1000,
            "baseline_isolated_section_bytes": 1000 + saved,
            "selected_saved_bytes_vs_baseline": saved,
            "selected_over_floor_ratio": over_floor,
        },
        "saturation_diagnostic": {"status": status},
        "grouped_brotli_order_diagnostic": {
            "schema": "section_payload_grouped_brotli_order_diagnostic.v1",
            "selected_grouped_brotli_bytes": 900,
            "selected_isolated_section_bytes": 900 + int(grouped_saved),
            "identity_grouped_brotli_bytes": 900 + int(grouped_saved),
            "grouped_delta_bytes_vs_identity": grouped_delta,
            "grouped_saved_bytes_vs_identity": int(grouped_saved),
            "grouped_delta_bytes_vs_selected_isolated": grouped_delta,
            "grouped_saved_bytes_vs_selected_isolated": int(grouped_saved),
        },
        "planner_feedback": {
            "operation_hint_count": 2,
            "rate_positive_hint_count": 1,
        },
        "blockers": ["byte_closed_archive_not_materialized"],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_section_payload_grammar_consumer_contract() -> None:
    reg = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.section_payload_grammar_consumer",
    )

    assert reg.contract_compliant is True
    assert consumer.CONSUMER_NAME == "section_payload_grammar_consumer"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert consumer.update_from_anchor({"anchor": "ignored"}) is None


def test_consumer_demotes_saturated_section_payload_without_score_authority() -> None:
    verdict = consumer.consume_candidate(_report())

    assert verdict["schema"] == "section_payload_grammar_consumer_result.v1"
    assert verdict["planner_action"] == (
        "record_section_payload_saturation_and_demote_format_churn"
    )
    assert verdict["demotion_recommended"] is True
    assert verdict["receiver_work_justified"] is False
    assert verdict["grouped_receiver_work_justified"] is False
    assert verdict["selected_saved_bytes_vs_baseline"] == 37
    assert verdict["rate_delta_score_if_components_unchanged"] == contest_rate_term(-37)
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["score_claim"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False
    assert "byte_closed_archive_not_materialized" in verdict["blockers"]


def test_consumer_routes_unsaturated_section_payload_to_receiver_binding() -> None:
    verdict = consumer.consume_candidate(
        _report(status="unsaturated_entropy_gap", saved=4096, over_floor=1.5)
    )

    assert verdict["planner_action"] == (
        "bind_section_receiver_and_materialize_byte_closed_archive"
    )
    assert verdict["receiver_work_justified"] is True
    assert verdict["grouped_receiver_work_justified"] is False
    assert verdict["demotion_recommended"] is False
    assert verdict["selected_over_floor_ratio"] == 1.5
    assert verdict["rate_delta_score_if_components_unchanged"] == contest_rate_term(-4096)
    assert verdict["promotable"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_consumer_routes_grouped_section_order_savings_to_receiver_binding() -> None:
    verdict = consumer.consume_candidate(
        _report(status="entropy_saturated", saved=0, grouped_saved=23)
    )

    assert verdict["planner_action"] == (
        "bind_section_receiver_and_materialize_grouped_brotli_archive"
    )
    assert verdict["receiver_work_justified"] is True
    assert verdict["grouped_receiver_work_justified"] is True
    assert verdict["demotion_recommended"] is False
    assert verdict["grouped_saved_bytes_vs_identity"] == 23
    assert verdict["grouped_delta_bytes_vs_identity"] == -23
    assert verdict["grouped_saved_bytes_vs_selected_isolated"] == 23
    assert verdict["grouped_delta_bytes_vs_selected_isolated"] == -23
    assert verdict["grouped_rate_delta_score_if_components_unchanged"] == contest_rate_term(-23)
    assert verdict["score_claim"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_consumer_preserves_schema_and_false_authority_blockers() -> None:
    payload = _report()
    payload["schema"] = "wrong"
    payload["ready_for_exact_eval_dispatch"] = True

    verdict = consumer.consume_candidate(payload)

    assert "section_payload_grammar_schema_mismatch" in verdict["blockers"]
    assert "ready_for_exact_eval_dispatch_overclaimed" in verdict["blockers"]
    assert verdict["ready_for_exact_eval_dispatch"] is False
    assert verdict["score_claim"] is False
