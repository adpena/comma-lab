"""Adversarial tests for the N600 pair-held-out lambda ranker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.ddm_lambda_ranker import (
    ADMISSION_NDCG_AT_4,
    PAIR_COUNT,
    SCHEMA,
    _fold_id,
    build_n600_lambda_ranker_receipt,
    ndcg_at_k,
    spearman_rho,
)

REPO = Path(__file__).resolve().parents[3]


def test_rank_metrics_have_exact_direction_and_top_k_semantics() -> None:
    relevance = [0.0, 3.0, 2.0, 1.0]
    assert spearman_rho(relevance, relevance) == pytest.approx(1.0)
    assert spearman_rho([-value for value in relevance], relevance) == pytest.approx(-1.0)
    assert ndcg_at_k(relevance, relevance, k=4) == pytest.approx(1.0)
    assert ndcg_at_k([3.0, 0.0, 1.0, 2.0], relevance, k=1) == pytest.approx(0.0)


def test_outer_fold_rule_is_deterministic_complete_and_nontrivial() -> None:
    first = [_fold_id(pair_id) for pair_id in range(PAIR_COUNT)]
    second = [_fold_id(pair_id) for pair_id in range(PAIR_COUNT)]
    assert first == second
    assert set(first) == set(range(5))
    assert sum(first.count(fold) for fold in range(5)) == PAIR_COUNT
    assert min(first.count(fold) for fold in range(5)) >= 100


def test_full_n600_receipt_is_pair_held_out_advisory_and_deterministic() -> None:
    first = build_n600_lambda_ranker_receipt(REPO)
    second = build_n600_lambda_ranker_receipt(REPO)
    assert first["schema"] == SCHEMA
    assert first["content_sha256"] == second["content_sha256"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["population"]["joined_pairs"] == PAIR_COUNT
    assert sum(first["population"]["fold_counts"].values()) == PAIR_COUNT
    assert len(first["pair_rankings"]) == PAIR_COUNT
    assert {row["pair_id"] for row in first["pair_rankings"]} == set(range(PAIR_COUNT))
    assert all(row["schema"] == "ddm_lambda_ranker_oof_pair_row.v1" for row in first["pair_rankings"])
    assert all(row["learned_form_tag"] == "[advisory-heuristic]" for row in first["pair_rankings"])
    assert first["score_claim"] is False
    assert first["actuation"] == "NONE"
    assert first["promotion_eligible"] is False
    assert first["main_landing_review_required"] is True
    assert first["j8f_blocker_preserved"] is True
    assert "BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY" in first["blocker_ids"]

    selected = first["selected_model"]
    assert selected["metrics"]["heldout_only"] is True
    assert selected["metrics"]["n_pairs"] == PAIR_COUNT
    assert first["admission_gate"]["passed"] == (
        selected["metrics"]["ndcg_at_4"] >= ADMISSION_NDCG_AT_4
    )
    assert all(
        candidate["metrics"]["heldout_only"] is True
        for candidate in first["model_race"]
        if candidate["metrics"] is not None
    )


def test_oracle_surfaces_and_requested_error_slices_are_explicit() -> None:
    receipt = build_n600_lambda_ranker_receipt(REPO)
    lineage = receipt["source_lineage"]
    assert lineage["margin_fisher_oracle"]["freshness"] == "FRESH"
    assert lineage["margin_fisher_oracle"]["surface_counts"] == {
        "bucket_rows": 1200,
        "direct_blocks": 25,
        "direct_pair_ids": 15,
    }
    assert lineage["pose_tube_oracle"]["surface_counts"]["pair_rows"] == 600
    assert lineage["stationarity_oracle"]["surface_counts"]["strata"] == 5
    assert {row["dimension"] for row in receipt["ranking_error_slices"]} == {
        "stratum",
        "g4_class",
        "margin_decile",
        "pair_hardness_decile",
    }
    assert receipt["innovations"]["status"] == "MEASURED_HELDOUT_INNOVATIONS"
    assert receipt["rudin_explanation"]["status"] == (
        "REUSED_CANONICAL_FALLING_RULE_LIST"
    )


def test_precision_refuses_fake_pair_order_and_self_checks_preserve_nulls() -> None:
    receipt = build_n600_lambda_ranker_receipt(REPO)
    precision = next(
        row
        for row in receipt["self_checks"]
        if row["check_id"] == "wallace_mml_pair_precision"
    )
    assert precision["status"] == "PARTIAL_TYPED"
    assert precision["value"] == {"pair_intervals": 15, "required": 600}
    assert sum(
        row["pair_order_status"] == "UNRANKED_PRECISION_OWED"
        for row in receipt["pair_rankings"]
    ) == 585
    checks = {row["check_id"]: row for row in receipt["self_checks"]}
    assert checks["pontryagin_bellman_adjacent_lambda_residual"]["value"] is None
    assert checks["rd1_organ_dual_consistency"]["value"] is None
    assert checks["rd1_organ_dual_consistency"]["status"] == (
        "AWAITING_NON_NULL_MATCHED_RD1_DUALS"
    )
    assert receipt["source_lineage"]["rd1_dual_authority"][
        "typed_dual_rows"
    ] == 162
    assert receipt["source_lineage"]["rd1_dual_authority"][
        "actionable_dual_rows"
    ] == 0
    assert checks["compression_progress_per_effort"]["value"] is None
