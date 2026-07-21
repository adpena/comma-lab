# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools.costate_organ_v3_backtest import build_backtest

REPO = Path(__file__).resolve().parents[2]


def test_fixed_n24_backtest_reports_every_stage_metric_and_ablation() -> None:
    result = build_backtest(
        v2_receipt=REPO / ".omx/research/costate_organ_v2_exact_anchor_backtest_20260721.json",
        r1b7_receipt=REPO / ".omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json",
        registry=REPO / ".omx/state/canonical_equations_registry.jsonl",
        created_utc="2026-07-21T02:25:46Z",
        bootstrap_replicates=100,
    )
    assert result["n_rows"] == 24
    assert len(result["row_ids"]) == len(set(result["row_ids"])) == 24
    assert result["learned_parameters"] == 0
    assert result["actuation"] == "NONE"
    assert not result["score_claim"] and not result["pointer_changed"]
    assert set(result["stages"]) == {
        "v2_baseline",
        "graded_realizability",
        "pool_interaction",
        "target_denoising",
        "receipt_emission",
    }
    required_metrics = {
        "spearman",
        "weighted_spearman",
        "top8_precision",
        "decision_ndcg_at_8",
        "tie_pairs",
    }
    assert all(required_metrics <= set(stage["metrics"]) for stage in result["stages"].values())
    assert set(result["ablations"]) == {
        "graded_realizability",
        "pool_interaction",
        "ema_delag",
        "apparatus_weighting",
        "receipt_emission",
    }
    assert result["source_custody"]["registry_bytes_unchanged"]
    assert result["pool_interactions"]["shared_pool_row_count"] > 0


def test_every_delta_has_bootstrap_ci_and_noise_verdict() -> None:
    result = build_backtest(
        v2_receipt=REPO / ".omx/research/costate_organ_v2_exact_anchor_backtest_20260721.json",
        r1b7_receipt=REPO / ".omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json",
        registry=REPO / ".omx/state/canonical_equations_registry.jsonl",
        created_utc="2026-07-21T02:25:46Z",
        bootstrap_replicates=100,
    )
    for comparison in result["comparisons"].values():
        for metric in comparison.values():
            assert len(metric["ci95"]) == 2
            assert metric["replicates_requested"] == 100
            assert metric["noise_band_verdict"] in {
                "POSITIVE_OUTSIDE_CI_NOISE",
                "NEGATIVE_OUTSIDE_CI_NOISE",
                "INSIDE_CI_NOISE_NO_IMPROVEMENT_CLAIM",
            }
    identity = result["comparisons"]["target_denoising_to_receipt_emission"]
    assert all(metric["delta"] == 0.0 and metric["ci95"] == [0.0, 0.0] for metric in identity.values())


def test_current_m1_repo_receipt_has_no_realized_costate_row() -> None:
    receipt = json.loads((REPO / ".omx/research/m1_band_manifest_n600_20260720T235322Z.json").read_text())
    assert "costate_realized_delta" not in receipt
    assert receipt["verdict"] == "BANDARTIFACT_LOAD_PASS_RECEIVER_FIRE_BLOCKED"
