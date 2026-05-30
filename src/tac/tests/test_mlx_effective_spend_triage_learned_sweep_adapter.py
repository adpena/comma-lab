# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.mlx_dynamic_learned_sweep import (
    build_mlx_dynamic_learned_sweep_plan,
)
from tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter import (
    REFUSAL_SCHEMA,
    SCHEMA,
    MLXEffectiveSpendTriageLearnedSweepAdapterError,
    build_mlx_effective_spend_triage_learned_sweep_candidates,
)
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE

REPO_ROOT = Path(__file__).resolve().parents[3]
INCUMBENT_SCORE = 0.1920513168811056


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _selection() -> dict[str, object]:
    normalized_gain = 0.012 / 600.0
    return {
        "schema": "mlx_effective_spend_triage_candidate_selection.v1",
        **_false_authority(),
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "allowed_use": (
            "candidate_generation_filter_after_strict_effective_mlx_spend_triage_gate"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "gates": {
            "effective_mlx_spend_triage_gate": {
                "schema": "ll_effective_mlx_spend_triage_gate.v1",
                "status": "strict_pass",
                "mlx_exact_eval_spend_triage_allowed": True,
                "allowed_use": (
                    "local_exact_eval_spend_triage_filter_after_all_mlx_and_dataset_gates"
                ),
            },
            "response_validation_status": "passed",
            "torch_parity_status": "strict_pass",
            "score_calibration_status": "strict_pass",
            "production_contract_status": "strict_pass",
        },
        "selection_policy": {
            "top_k": 1,
            "families": ["mlx_decoder_q"],
            "min_observed_gain": 0.00001,
            "planning_value_scope": "normalized_full_video",
        },
        "summary": {
            "selected_count": 1,
            "eligible_row_count": 1,
            "dataset_row_count": 1,
        },
        "selected_rows": [
            {
                "schema": "mlx_effective_spend_triage_candidate_row.v1",
                **_false_authority(),
                "candidate_generation_only": True,
                "archive_materialization_required": True,
                "requires_exact_auth_eval_before_score_claim": True,
                "candidate_id": "mlx_scorer_response:window:501:502",
                "row_id": "window_501_502",
                "family": "mlx_decoder_q",
                "source_schema": "mlx_scorer_response.v1",
                "source_evidence_grade": EVIDENCE_GRADE_MLX,
                "source_evidence_tag": EVIDENCE_TAG_MLX,
                "canonical_provenance": {
                    "measurement_axis": EVIDENCE_TAG_MLX,
                    "hardware_substrate": "macos_arm64_mlx",
                    "evidence_grade": "macos_mlx_research_signal",
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                },
                "rank": 1,
                "observed_delta_vs_baseline_score": -0.012,
                "observed_scorer_delta_vs_baseline": -0.012,
                "observed_scorer_gain_vs_baseline": 0.012,
                "projected_full_video_delta_vs_baseline_score": -normalized_gain,
                "full_video_denominator": 600,
                "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
                "break_even_added_bytes_from_normalized_full_video_gain": (
                    normalized_gain / RATE_SCORE_PER_BYTE
                ),
                "normalized_full_video_byte_budget_margin_vs_break_even": (
                    normalized_gain / RATE_SCORE_PER_BYTE
                ),
                "predicted_delta_vs_baseline_score": 0.0004,
                "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
                "selection_basis": "normalized_full_video_mlx_singleton_response_gain",
                "selection_planning_value_scope": "normalized_full_video",
                "pair_indices": [501, 502],
                "source_pair_window": [501, 502],
                "byte_budget_margin_vs_break_even": 2500.0,
                "added_archive_bytes": 0,
                "source_n_samples": 1,
                "source_batch_pairs": 1,
            }
        ],
    }


def test_adapter_emits_quality_evidence_consumed_by_dynamic_sweep() -> None:
    payload = build_mlx_effective_spend_triage_learned_sweep_candidates(
        _selection(),
        incumbent_score=INCUMBENT_SCORE,
        top_k=1,
    )

    assert payload["schema"] == SCHEMA
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    candidate = payload["candidates"][0]
    assert candidate["candidate_id"] == "mlx_scorer_response:window:501:502"
    assert candidate["predicted_score_mean"] == pytest.approx(
        INCUMBENT_SCORE - 0.00002
    )
    assert candidate["non_authoritative_mlx_window_gain_sum"] == pytest.approx(0.012)
    assert candidate["quality_evidence"]["gate_statuses"] == {
        "calibration": "strict_pass",
        "parity": "strict_pass",
        "production_contract": "strict_pass",
        "effective_spend_triage": "strict_pass",
    }
    assert candidate["quality_evidence"]["source_evidence_tag"] == EVIDENCE_TAG_MLX
    assert candidate["quality_evidence"]["canonical_provenance"]["evidence_grade"] == "macos_mlx_research_signal"

    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[payload],
        top_k=1,
    )

    row = plan["ranked_sweep_rows"][0]
    assert row["candidate_id"] == "mlx_scorer_response:window:501:502"
    assert row["score_claim"] is False
    assert row["prediction_source"] == "mlx_effective_spend_triage_quality_adapter"


def test_adapter_rejects_legacy_observed_window_selection_basis() -> None:
    selection = _selection()
    selection["selected_rows"][0]["selection_basis"] = (
        "observed_strict_gated_mlx_singleton_response_gain"
    )

    with pytest.raises(
        MLXEffectiveSpendTriageLearnedSweepAdapterError,
        match="selection_basis",
    ):
        build_mlx_effective_spend_triage_learned_sweep_candidates(
            selection,
            incumbent_score=INCUMBENT_SCORE,
        )


def test_adapter_rejects_failed_strict_gate() -> None:
    selection = _selection()
    selection["gates"]["score_calibration_status"] = "blocked"

    with pytest.raises(
        MLXEffectiveSpendTriageLearnedSweepAdapterError,
        match="calibration",
    ):
        build_mlx_effective_spend_triage_learned_sweep_candidates(
            selection,
            incumbent_score=INCUMBENT_SCORE,
        )


def test_adapter_rejects_truthy_row_authority() -> None:
    selection = _selection()
    selection["selected_rows"][0]["score_claim"] = True

    with pytest.raises(
        MLXEffectiveSpendTriageLearnedSweepAdapterError,
        match="score_claim",
    ):
        build_mlx_effective_spend_triage_learned_sweep_candidates(
            selection,
            incumbent_score=INCUMBENT_SCORE,
        )


def test_adapter_rejects_selection_rows_missing_mlx_source_evidence() -> None:
    selection = _selection()
    selection["selected_rows"][0].pop("source_evidence_grade")

    with pytest.raises(
        MLXEffectiveSpendTriageLearnedSweepAdapterError,
        match="source_evidence_grade",
    ):
        build_mlx_effective_spend_triage_learned_sweep_candidates(
            selection,
            incumbent_score=INCUMBENT_SCORE,
        )


def test_adapter_cli_writes_payload_and_dynamic_sweep_cli_consumes_it(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.json"
    payload_path = tmp_path / "learned_sweep_candidates.json"
    plan_path = tmp_path / "learned_sweep_plan.json"
    selection_path.write_text(json.dumps(_selection(), sort_keys=True), encoding="utf-8")

    adapt = subprocess.run(
        [
            sys.executable,
            "tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py",
            "--selection",
            str(selection_path),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--top-k",
            "1",
            "--json-out",
            str(payload_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert adapt.returncode == 0, adapt.stderr
    assert json.loads(payload_path.read_text(encoding="utf-8"))["schema"] == SCHEMA

    plan = subprocess.run(
        [
            sys.executable,
            "tools/plan_mlx_dynamic_learned_sweep.py",
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--candidate-payload",
            str(payload_path),
            "--top-k",
            "1",
            "--json-out",
            str(plan_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan_path.read_text(encoding="utf-8"))["score_claim"] is False


def test_adapter_cli_failure_json_records_legacy_refusal(tmp_path: Path) -> None:
    selection = _selection()
    selection["selected_rows"][0]["selection_basis"] = (
        "observed_strict_gated_mlx_singleton_response_gain"
    )
    selection_path = tmp_path / "legacy_selection.json"
    payload_path = tmp_path / "should_not_exist.json"
    refusal_path = tmp_path / "legacy_refusal.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py",
            "--selection",
            str(selection_path),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--json-out",
            str(payload_path),
            "--failure-json-out",
            str(refusal_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert not payload_path.exists()
    refusal = json.loads(refusal_path.read_text(encoding="utf-8"))
    assert refusal["schema"] == REFUSAL_SCHEMA
    assert refusal["score_claim"] is False
    assert "selection_basis" in refusal["error"]


def test_adapter_cli_refuses_silent_overwrite_and_requires_expected_hash(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.json"
    payload_path = tmp_path / "learned_sweep_candidates.json"
    selection_path.write_text(json.dumps(_selection(), sort_keys=True), encoding="utf-8")
    payload_path.write_text('{"old": true}\n', encoding="utf-8")

    refused = subprocess.run(
        [
            sys.executable,
            "tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py",
            "--selection",
            str(selection_path),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--json-out",
            str(payload_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "refusing to overwrite" in refused.stderr

    existing_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    overwritten = subprocess.run(
        [
            sys.executable,
            "tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py",
            "--selection",
            str(selection_path),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--json-out",
            str(payload_path),
            "--allow-overwrite",
            "--expected-output-sha256",
            existing_sha,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert overwritten.returncode == 0, overwritten.stderr
    assert json.loads(payload_path.read_text(encoding="utf-8"))["schema"] == SCHEMA
