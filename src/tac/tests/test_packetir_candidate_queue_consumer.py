# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import packetir_candidate_queue_consumer as consumer
from tac.packet_compiler.pr101_fec6_candidate_queue import (
    PR101_FEC6_CANDIDATE_QUEUE_SCHEMA,
)


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "probe",
        "candidate_kind": "selector_entropy_recode_probe",
        "blockers": ["packet_candidate_not_materialized"],
        "consumer_surfaces": ["runtime", "queue"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_packetir_candidate_queue_consumer_contract() -> None:
    reg = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.packetir_candidate_queue_consumer",
    )

    assert reg.contract_compliant is True
    assert consumer.CONSUMER_NAME == "packetir_candidate_queue_consumer"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert consumer.update_from_anchor({"anchor": "ignored"}) is None


def test_consume_candidate_is_observability_only() -> None:
    verdict = consumer.consume_candidate(_candidate())

    assert verdict["candidate_id"] == "probe"
    assert verdict["eligible"] is False
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["axis_tag"] == "[predicted]"
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False
    assert "packet_candidate_not_materialized" in verdict["blockers"]
    assert "tac.packet_compiler" in verdict["rationale"]


def test_consume_candidate_rejects_overclaim() -> None:
    candidate = _candidate()
    candidate["ready_for_exact_eval_dispatch"] = True

    verdict = consumer.consume_candidate(candidate)

    assert verdict["eligible"] is False
    assert "ready_for_exact_eval_dispatch_overclaimed" in verdict["blockers"]
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_consume_queue_validates_schema_and_candidates() -> None:
    queue = {
        "schema": PR101_FEC6_CANDIDATE_QUEUE_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidates": [_candidate()],
    }

    result = consumer.consume_queue(queue)

    assert result["schema"] == "packetir_candidate_queue_consumer_result_v2"
    assert result["candidate_count"] == 1
    assert result["consumed_candidate_count"] == 1
    assert result["eligible_candidate_count"] == 0
    assert result["score_claim"] is False
    assert result["axis_tag"] == "[predicted]"
    assert result["candidate_verdicts"][0]["candidate_id"] == "probe"


def test_consume_queue_rejects_queue_overclaim() -> None:
    queue = {
        "schema": "wrong",
        "score_claim": True,
        "candidates": [],
    }

    result = consumer.consume_queue(queue)

    assert "queue_schema_mismatch" in result["blockers"]
    assert "score_claim_overclaimed" in result["blockers"]
    assert result["score_claim"] is False
