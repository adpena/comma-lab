# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.cathedral.consumer_contract import HookNumber
from tac.cathedral_consumers import (
    distilled_scorer_surrogate_canonical_equation_consumer as consumer,
)
from tac.optimization.scorer_response_dataset import RATE_SCORE_PER_BYTE


def _row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "scorer_response_row.v1",
        "row_id": "row-a",
        "candidate_id": "pds_stage1",
        "family": consumer.ROW_FAMILY,
        "score_claim": False,
        "score_claim_valid": False,
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
    assert result["planning_target_accessor"] == "scorer_response_planning_value_for_target"
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["score_claim"] is False
    assert result["score_claim_valid"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["promotable"] is False


def test_consume_candidate_uses_normalized_mlx_planning_target() -> None:
    row = _mlx_distilled_row(projected_delta=0.001, raw_delta=-10.0)

    result = consumer.consume_candidate({"scorer_response_dataset": {"rows": [row]}})

    assert result["consumer_signal_kind"] == "distilled_scorer_surrogate_routing"
    assert result["improved_total_score_count"] == 0
    assert result["improved_scorer_term_count"] == 0
    assert result["best_total_row"]["delta_vs_baseline_score"] == 0.001
    assert result["best_total_row"]["planning_target_accessor"] == (
        "scorer_response_planning_value_for_target"
    )
    assert result["best_scorer_row"]["scorer_delta_vs_baseline"] == 0.001


def test_consume_candidate_blocks_mlx_missing_normalized_objective() -> None:
    row = _row(
        source_schema="mlx_scorer_response.v1",
        delta_vs_baseline_score=-10.0,
        scorer_delta_vs_baseline=-10.0,
    )

    result = consumer.consume_candidate({"scorer_response_dataset": {"rows": [row]}})

    assert result["consumer_signal_kind"] == "distilled_scorer_surrogate_authority_blocked"
    assert result["blocked_field"] == "delta_vs_baseline_score"
    assert "missing normalized full-video objective" in result["blocked_error"]
    assert result["score_claim_valid"] is False


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


def _mlx_distilled_row(*, projected_delta: float, raw_delta: float) -> dict[str, object]:
    normalized_gain = -float(projected_delta)
    return _row(
        source_schema="mlx_scorer_response.v1",
        delta_vs_baseline_score=float(raw_delta),
        scorer_delta_vs_baseline=float(raw_delta),
        observed_scorer_gain_vs_baseline=normalized_gain,
        added_archive_bytes=0,
        source_n_samples=600,
        full_video_denominator=600,
        normalized_full_video_scorer_gain_vs_baseline=normalized_gain,
        projected_full_video_delta_vs_baseline_score=float(projected_delta),
        break_even_added_bytes_from_normalized_full_video_gain=(
            normalized_gain / RATE_SCORE_PER_BYTE
        ),
        normalized_full_video_byte_budget_margin_vs_break_even=(
            normalized_gain / RATE_SCORE_PER_BYTE
        ),
    )
