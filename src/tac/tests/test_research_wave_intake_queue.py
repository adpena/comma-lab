# SPDX-License-Identifier: MIT
"""Tests for the guarded research-wave intake queue."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from research_wave_intake_queue import (  # noqa: E402
    build_intake_payload,
    parse_top5_rows,
)


TOP5_FIXTURE = """---
review_id: comprehensive_research_wave_20260518
lane_id: lane_deep_research_wave_20260518
research_only: true
---

### TOP-5 substrate reformulations from new evidence

| # | Substrate | Reformulation | Predicted delta S band | First-principles citation | Approx cost |
|---|---|---|---|---|---|
| 1 | **TT5L V2** (foveation+LAPose+world-model) | VGGT + foveation stack | `[-0.020, -0.008]` over PR101 frontier => `[0.172, 0.184]` [contest-CPU] | DreamerV3 + VGGT | ~$15-25 Modal A100 |
| 2 | **Z7-as-GRU** | Mamba-style selective state-space | `[-0.025, -0.008]` => `[0.167, 0.184]` [contest-CPU] | Mamba-2 | ~$20-30 Modal A100 |

## next section
"""


def test_parse_top5_rows_extracts_markdown_table() -> None:
    rows = parse_top5_rows(TOP5_FIXTURE)
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"
    assert rows[0]["substrate"].startswith("TT5L V2")
    assert "VGGT" in rows[0]["reformulation"]


def test_build_intake_payload_preserves_false_authority_and_readiness_join() -> None:
    readiness = {
        "time_traveler_l5_autonomy": {
            "readiness_verdict": "NEEDS_FIX",
            "blocking_issues": ["CATALOG_315_council_proceed_with_revisions"],
            "recipe_path": ".omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml",
        }
    }
    payload = build_intake_payload(
        TOP5_FIXTURE,
        repo_root=REPO_ROOT,
        source_memo_path=Path(".omx/research/comprehensive_research_wave_20260518.md"),
        readiness_by_substrate=readiness,
        created_utc="2026-05-18T05:30:00Z",
    )

    assert payload["schema"] == "research_wave_intake_queue_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["current_readiness_join_count"] == 1
    assert payload["research_priority_order"] == [
        "time_traveler_l5_autonomy",
        "time_traveler_l5_z7_lstm_predictive_coding",
    ]
    assert payload["actionable_priority_order"] == []
    assert payload["dispatch_actionable_priority_order"] == []
    assert "provider-dispatch actionable only" in payload[
        "actionable_priority_order_semantics"
    ]

    tt5l = payload["candidates"][0]
    assert tt5l["substrate_id"] == "time_traveler_l5_autonomy"
    assert tt5l["predicted_delta_s_band"] == [-0.02, -0.008]
    assert tt5l["predicted_frontier_score_band"] == [0.172, 0.184]
    assert tt5l["score_axis"] == "contest-CPU"
    assert tt5l["current_readiness_verdict"] == "NEEDS_FIX"
    assert tt5l["recommended_next_gate"] == (
        "resolve_modal_billing_or_lightning_doctor_env_then_stage_manifest_and_claim"
    )
    assert "current_readiness_has_blocking_issues" in tt5l["blockers"]
    assert "prediction_axis_is_contest-CPU_not_contest_cuda" in tt5l["blockers"]

    z7 = payload["candidates"][1]
    assert z7["substrate_id"] == "time_traveler_l5_z7_lstm_predictive_coding"
    assert z7["current_readiness_verdict"] is None
    assert "no_current_asymptotic_readiness_entry" in z7["blockers"]


def test_build_intake_payload_maps_dp1_after_l1_noop_probe() -> None:
    text = """---
review_id: comprehensive_research_wave_20260518
lane_id: lane_deep_research_wave_20260518
research_only: true
---

### TOP-5 substrate reformulations from new evidence

| # | Substrate | Reformulation | Predicted delta S band | First-principles citation | Approx cost |
|---|---|---|---|---|---|
| 4 | DP1 + PR101 composition | Stack DP1 onto PR101 | `[-0.012, -0.004]` => `[0.180, 0.188]` [contest-CPU] | DP1 prior | ~$10-15 Modal A100 |
"""
    payload = build_intake_payload(
        text,
        repo_root=REPO_ROOT,
        source_memo_path=Path(".omx/research/comprehensive_research_wave_20260518.md"),
        readiness_by_substrate={
            "dp1_pr101_composition": {
                "readiness_verdict": "DEFER",
                "blocking_issues": ["CATALOG_240_FULL_MAIN_BLOCKED"],
                "recipe_path": ".omx/operator_authorize_recipes/substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch.yaml",
            }
        },
        created_utc="2026-05-18T06:30:00Z",
    )

    dp1 = payload["candidates"][0]
    assert dp1["substrate_id"] == "dp1_pr101_composition"
    assert dp1["current_readiness_verdict"] == "DEFER"
    assert dp1["recommended_next_gate"] == (
        "run_full_frame_parity_or_path2_lambda_prior_disambiguator_after_l1_noop_probe"
    )
    assert "current_readiness_has_blocking_issues" in dp1["blockers"]


def test_build_intake_payload_maps_atw_after_byte_closed_probe() -> None:
    text = """---
review_id: comprehensive_research_wave_20260518
lane_id: lane_deep_research_wave_20260518
research_only: true
---

### TOP-5 substrate reformulations from new evidence

| # | Substrate | Reformulation | Predicted delta S band | First-principles citation | Approx cost |
|---|---|---|---|---|---|
| 3 | ATW V2-1 | Richer side-info channel | `[-0.015, -0.005]` => `[0.178, 0.188]` [contest-CPU] | Atick-Tishby-Wyner | ~$5-7 CPU probe |
"""
    payload = build_intake_payload(
        text,
        repo_root=REPO_ROOT,
        source_memo_path=Path(".omx/research/comprehensive_research_wave_20260518.md"),
        readiness_by_substrate={
            "atw_codec_v2": {
                "readiness_verdict": "DEFER",
                "blocking_issues": ["CATALOG_313_PROBE_BLOCKING"],
                "recipe_path": ".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml",
            }
        },
        created_utc="2026-05-18T06:40:00Z",
    )

    atw = payload["candidates"][0]
    assert atw["substrate_id"] == "atw_codec_v2"
    assert atw["current_readiness_verdict"] == "DEFER"
    assert atw["recommended_next_gate"] == (
        "design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe"
    )
    assert "current_readiness_has_blocking_issues" in atw["blockers"]


def test_build_intake_payload_blocks_non_lowering_or_malformed_bands() -> None:
    text = """---
review_id: comprehensive_research_wave_20260518
lane_id: lane_deep_research_wave_20260518
research_only: true
---

### TOP-5 substrate reformulations from new evidence

| # | Substrate | Reformulation | Predicted delta S band | First-principles citation | Approx cost |
|---|---|---|---|---|---|
| 1 | TT5L V2 | malformed band | TBD [contest-CPU] | VGGT | ~$1 |
| 2 | Z7-as-GRU | positive regression band | `[0.001, 0.004]` => `[0.193, 0.196]` [contest-CPU] | Mamba | ~$1 |
"""
    payload = build_intake_payload(
        text,
        repo_root=REPO_ROOT,
        source_memo_path=Path(".omx/research/comprehensive_research_wave_20260518.md"),
        readiness_by_substrate={},
        created_utc="2026-05-18T09:00:00Z",
    )

    malformed, positive = payload["candidates"]
    assert payload["research_priority_order"] == []
    assert payload["actionable_priority_order"] == []
    assert malformed["prediction_band_valid_for_queue"] is False
    assert "predicted_delta_s_band_missing_or_malformed" in malformed["blockers"]
    assert "predicted_frontier_score_band_missing_or_malformed" in malformed[
        "blockers"
    ]
    assert positive["prediction_band_valid_for_queue"] is False
    assert "predicted_delta_s_band_not_strictly_score_lowering" in positive[
        "blockers"
    ]
    assert positive["score_claim"] is False
    assert positive["ready_for_paid_dispatch"] is False


def test_build_intake_payload_keeps_cuda_ready_row_dispatch_actionable() -> None:
    text = """---
review_id: comprehensive_research_wave_20260518
lane_id: lane_deep_research_wave_20260518
research_only: true
---

### TOP-5 substrate reformulations from new evidence

| # | Substrate | Reformulation | Predicted delta S band | First-principles citation | Approx cost |
|---|---|---|---|---|---|
| 1 | Z7-as-GRU | contest CUDA ready hypothesis | `[-0.010, -0.004]` => `[0.190, 0.196]` [contest-CUDA] | Mamba | ~$1 |
"""
    payload = build_intake_payload(
        text,
        repo_root=REPO_ROOT,
        source_memo_path=Path(".omx/research/comprehensive_research_wave_20260518.md"),
        readiness_by_substrate={
            "time_traveler_l5_z7_lstm_predictive_coding": {
                "readiness_verdict": "READY",
                "blocking_issues": [],
                "recipe_path": ".omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml",
            }
        },
        created_utc="2026-05-18T09:10:00Z",
    )

    assert payload["research_priority_order"] == [
        "time_traveler_l5_z7_lstm_predictive_coding"
    ]
    assert payload["actionable_priority_order"] == [
        "time_traveler_l5_z7_lstm_predictive_coding"
    ]
