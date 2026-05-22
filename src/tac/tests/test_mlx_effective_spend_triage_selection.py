# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.mlx_effective_spend_triage_selection import (
    MLXEffectiveSpendTriageSelectionError,
    build_mlx_effective_spend_triage_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _row(
    row_id: str,
    *,
    gain: float,
    family: str = "mlx_decoder_q",
    predicted: float = 1.0e-4,
    score_claim: bool = False,
) -> dict:
    return {
        "schema": "scorer_response_row.v1",
        **_false_authority(),
        "score_claim": score_claim,
        "row_id": row_id,
        "family": family,
        "candidate_id": f"mlx_scorer_response:window:{row_id}",
        "axis": "[macOS-MLX research-signal]",
        "source_evidence_grade": "macOS-MLX-research-signal",
        "source_evidence_tag": "[macOS-MLX research-signal]",
        "source_schema": "mlx_scorer_response.v1",
        "source_batch_pairs": 1,
        "source_n_samples": 1,
        "source_pair_window": [10, 11],
        "pair_indices": [10, 11],
        "source_path": f"candidate_pair_{row_id}.json",
        "window_baseline_source_path": f"baseline_pair_{row_id}.json",
        "delta_vs_baseline_score": -gain,
        "scorer_delta_vs_baseline": -gain,
        "observed_scorer_gain_vs_baseline": gain,
        "byte_budget_margin_vs_break_even": gain * 1000.0,
        "break_even_added_bytes_from_scorer_gain": gain * 1000.0,
        "added_archive_bytes": 0,
        "ll_predicted_delta_vs_baseline_score": predicted,
        "archive_sha256": "a" * 64,
        "raw_sha256": "b" * 64,
        "source_inflated_outputs_aggregate_sha256": "c" * 64,
        "source_candidate_cache_array_sha256": {"pair_indices": "d" * 64},
        "source_reference_cache_array_sha256": {"pair_indices": "e" * 64},
        "window_baseline_candidate_cache_array_sha256": {"pair_indices": "f" * 64},
        "window_baseline_reference_cache_array_sha256": {"pair_indices": "1" * 64},
    }


def _dataset() -> dict:
    return {
        "schema": "scorer_response_dataset.v1",
        **_false_authority(),
        "authority": _false_authority(),
        "rows": [
            _row("best", gain=0.002, predicted=0.0003),
            _row("second", gain=0.001, predicted=-0.0001),
            _row("weak", gain=0.000001),
            _row("fec6", gain=0.0, family="mlx_fec6_auth_parent"),
        ],
    }


def _plan(*, effective_status: str = "strict_pass") -> dict:
    authority = _false_authority()
    return {
        "schema": "ll_scorer_response_next_probe_plan.v1",
        **authority,
        "response_validation_gate": {
            **authority,
            "status": "passed",
            "passed": True,
        },
        "mlx_torch_parity_sweep_gate": {
            **authority,
            "status": "strict_pass",
            "mlx_rows_allowed_for_planner": True,
        },
        "mlx_score_calibration_gate": {
            **authority,
            "status": "strict_pass",
            "mlx_spend_triage_allowed": True,
            "summary": {
                "recommended_min_mlx_gap_for_spend_triage": 0.00001,
            },
        },
        "mlx_production_contract_gate": {
            **authority,
            "status": "strict_pass",
            "mlx_spend_triage_allowed": True,
        },
        "effective_mlx_spend_triage_gate": {
            **authority,
            "schema": "ll_effective_mlx_spend_triage_gate.v1",
            "status": effective_status,
            "candidate_generation_only": True,
            "mlx_exact_eval_spend_triage_allowed": effective_status == "strict_pass",
            "allowed_use": (
                "local_exact_eval_spend_triage_filter_after_all_mlx_and_dataset_gates"
            ),
        },
    }


def test_selection_uses_observed_strict_gated_rows_not_positive_predictions() -> None:
    selection = build_mlx_effective_spend_triage_selection(
        _dataset(),
        _plan(),
        top_k=2,
        families=["mlx_decoder_q"],
    )

    assert selection["score_claim"] is False
    assert selection["ready_for_exact_eval_dispatch"] is False
    assert selection["archive_materialization_required"] is True
    assert selection["summary"]["eligible_row_count"] == 2
    assert selection["summary"]["selected_count"] == 2
    assert selection["summary"]["prediction_disagree_selected_count"] == 1
    assert [row["row_id"] for row in selection["selected_rows"]] == [
        "best",
        "second",
    ]
    assert selection["selected_rows"][0]["selection_basis"] == (
        "observed_strict_gated_mlx_singleton_response_gain"
    )


def test_selection_can_require_negative_predictions_when_requested() -> None:
    selection = build_mlx_effective_spend_triage_selection(
        _dataset(),
        _plan(),
        top_k=4,
        families=["mlx_decoder_q"],
        require_prediction_negative=True,
    )

    assert selection["summary"]["eligible_row_count"] == 1
    assert selection["selected_rows"][0]["row_id"] == "second"


def test_selection_blocks_non_oof_or_failed_effective_gate() -> None:
    with pytest.raises(MLXEffectiveSpendTriageSelectionError, match="must be strict_pass"):
        build_mlx_effective_spend_triage_selection(
            _dataset(),
            _plan(effective_status="blocked"),
        )


def test_selection_blocks_rows_with_score_authority() -> None:
    dataset = _dataset()
    dataset["rows"][0]["score_claim"] = True

    selection = build_mlx_effective_spend_triage_selection(dataset, _plan(), top_k=4)

    assert [row["row_id"] for row in selection["selected_rows"]] == ["second"]
    assert selection["summary"]["rejection_counts"]["score_claim_not_false"] == 1


def test_selection_cli_writes_false_authority_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    plan_path = tmp_path / "plan.json"
    json_out = tmp_path / "selection.json"
    md_out = tmp_path / "selection.md"
    dataset_path.write_text(json.dumps(_dataset()), encoding="utf-8")
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "select_mlx_effective_spend_triage_candidates.py"),
            "--dataset",
            str(dataset_path),
            "--plan",
            str(plan_path),
            "--family",
            "mlx_decoder_q",
            "--top-k",
            "1",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["selected_count"] == 1
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["selected_rows"][0]["row_id"] == "best"
    assert payload["source_artifacts"]["dataset"]["sha256"]
    assert "Required Next Step" in md_out.read_text(encoding="utf-8")
