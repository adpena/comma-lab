# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.cathedral.consumer_contract import HookNumber
from tac.cathedral_consumers import (
    distilled_scorer_surrogate_canonical_equation_consumer as consumer,
)


def _row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "scorer_response_row.v1",
        "row_id": "row-a",
        "candidate_id": "pds_stage1",
        "family": consumer.ROW_FAMILY,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "delta_vs_baseline_score": -0.01,
        "scorer_delta_vs_baseline": -0.02,
        "added_archive_bytes": -1,
    }
    payload.update(overrides)
    return payload


def test_consumer_contract_markers() -> None:
    assert consumer.CONSUMER_NAME == "distilled_scorer_surrogate_canonical_equation_consumer"
    assert consumer.CONSUMES_SCORER_RESPONSE_DATASET is True
    assert consumer.CONSUMES_MASTER_GRADIENT_ANCHORS is True
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert callable(consumer.update_from_anchor)
    assert callable(consumer.consume_candidate)


def test_update_from_anchor_noops_for_supported_inputs() -> None:
    consumer.update_from_anchor(None)
    consumer.update_from_anchor({"family": consumer.ROW_FAMILY})
    consumer.update_from_anchor({"archive_sha256": "a" * 64, "substrate_id": "fec6"})


def test_consume_candidate_routes_distilled_rows_non_promotably() -> None:
    result = consumer.consume_candidate({"scorer_response_dataset": {"rows": [_row()]}})

    assert result["consumer_signal_kind"] == "distilled_scorer_surrogate_routing"
    assert result["canonical_equation_id"] == consumer.CANONICAL_EQUATION_ID
    assert result["row_count"] == 1
    assert result["row_ids"] == ["row-a"]
    assert result["best_total_row"]["delta_vs_baseline_score"] == -0.01
    assert result["best_scorer_row"]["scorer_delta_vs_baseline"] == -0.02
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["score_claim"] is False
    assert result["score_claim_valid"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["promotable"] is False


def test_consume_candidate_refuses_authority_true_row() -> None:
    result = consumer.consume_candidate({"rows": [_row(score_claim=True)]})

    assert result["consumer_signal_kind"] == "distilled_scorer_surrogate_authority_blocked"
    assert result["blocked_field"] == "score_claim"
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["score_claim"] is False
    assert result["score_claim_valid"] is False
    assert result["promotable"] is False


def test_consume_candidate_no_signal_for_unrelated_candidate() -> None:
    result = consumer.consume_candidate({"rows": [{"family": "decoder_q"}]})

    assert result["consumer_signal_kind"] == "distilled_scorer_surrogate_absent"
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
