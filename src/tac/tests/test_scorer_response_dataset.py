from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES
from tac.optimization.decoder_q_response_surface import (
    build_decoder_q_response_surface,
    render_decoder_q_response_surface_markdown,
)
from tac.optimization.scorer_response_dataset import (
    RATE_SCORE_PER_BYTE,
    ResponseBaseline,
    ScorerResponseDatasetError,
    build_decoder_q_surface_advisory_gate,
    build_magic_codec_seed_boundary,
    build_mlx_production_contract_gate,
    build_mlx_score_calibration_gate,
    build_mlx_torch_parity_sweep_gate,
    build_next_probe_plan,
    build_null_byte_priority_weights,
    build_response_dataset,
    build_scorer_response_consumer_routing,
    build_scorer_response_validation_gate,
    build_windowed_mlx_response_dataset,
    merge_scorer_response_datasets,
    normalize_legacy_response_dataset_authority,
    refresh_mlx_scorer_response_source_identity,
    render_markdown,
    render_next_probe_plan_markdown,
    render_validation_gate_markdown,
)
from tac.optimization.scorer_response_family_delta import (
    build_family_delta,
    render_family_delta_markdown,
)
from tac.optimization.scorer_response_prediction import (
    STRUCTURED_FEATURE_SET,
    attach_out_of_fold_linear_predictions,
)
from tac.optimization.scorer_response_structural_features import (
    attach_structural_features,
)

REPO = Path(__file__).resolve().parents[3]


def _advisory(score: float, archive_bytes: int, pose: float, seg: float) -> dict:
    return {
        "canonical_score": score,
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "axis": "[macOS-CPU advisory test]",
        "archive": {"sha256": "a" * 64, "bytes": archive_bytes},
        "raw": {"sha256": "b" * 64},
    }


def _mlx_response_payload() -> dict:
    return {
        "schema_version": "mlx_scorer_response.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "hardware_substrate": {"device_type": "cpu"},
        "batch_pairs": 2,
        "start_pair": 16,
        "max_pairs": 4,
        "n_samples": 4,
        "pair_window": [16, 20],
        "elapsed_seconds": 3.0,
        "canonical_score": 0.900001,
        "score_recomputed_from_components": 0.900001,
        "canonical_score_source": "score_recomputed_from_components",
        "archive_size_bytes": 110,
        "avg_posenet_dist": 0.004,
        "avg_segnet_dist": 0.010,
        "archive_sha256": "a" * 64,
        "raw_sha256": "b" * 64,
        "components": {
            "posenet_sha256": "p" * 64,
            "segnet_sha256": "s" * 64,
        },
        "cache_identity": {
            "pair_indices_equal": True,
            "reference": {
                "archive_sha256": None,
                "inflated_outputs_aggregate_sha256": None,
                "raw_sha256": None,
                "array_sha256": {
                    "pair_indices": "0" * 64,
                    "posenet_yuv6_pair": "1" * 64,
                    "segnet_last_rgb": "2" * 64,
                },
            },
            "candidate": {
                "archive_sha256": "a" * 64,
                "inflated_outputs_aggregate_sha256": "e" * 64,
                "raw_sha256": "b" * 64,
                "array_sha256": {
                    "pair_indices": "0" * 64,
                    "posenet_yuv6_pair": "3" * 64,
                    "segnet_last_rgb": "4" * 64,
                },
            },
        },
        "device_contract": {
            "forbidden_uses": [
                "auth_eval",
                "score_claim",
                "promotion",
                "rank_or_kill",
            ],
            "allowed_uses": ["local_mlx_training_gradient_shaping"],
        },
    }


def _mlx_parity_sweep_payload(*, passed: bool = True) -> dict:
    blockers = [] if passed else ["window_failed:index=39:pair_window=[156, 160]"]
    failed_windows = 0 if passed else 1
    segnet_argmax_pixels = 0.0 if passed else 1.0
    segnet_argmax_fraction = 0.0 if passed else 1.2715657552083333e-06
    return {
        "schema_version": "mlx_scorer_torch_parity_sweep.v1",
        "run_id": "unit",
        "verdict": (
            "PASS_MLX_TORCH_SCORER_PARITY_SWEEP"
            if passed
            else "FAIL_MLX_TORCH_SCORER_PARITY_SWEEP"
        ),
        "passed": passed,
        "blockers": blockers,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "device_type": "cpu",
        "window_count": 75,
        "covered_pair_window": [0, 300],
        "summary": {
            "passed_windows": 75 - failed_windows,
            "failed_windows": failed_windows,
            "posenet_output_abs_max": {"max": 7.62939453125e-06},
            "posenet_component_abs_max": {"max": 9.713470479344455e-12},
            "segnet_logit_abs_max": {"max": 0.0007913112640380859},
            "segnet_argmax_diff_pixels": {"max": segnet_argmax_pixels},
            "segnet_argmax_diff_fraction": {"max": segnet_argmax_fraction},
            "segnet_argmax_mismatch_pixels_total": 0 if passed else 1,
        },
        "rows": [
            {
                "index": 39,
                "passed": passed,
                "verdict": (
                    "PASS_MLX_TORCH_SCORER_PARITY"
                    if passed
                    else "FAIL_MLX_TORCH_SCORER_PARITY"
                ),
                "blockers": [] if passed else ["segnet_argmax_diff_pixels_exceeds_threshold:1>0"],
                "pair_window": [156, 160],
                "deltas": {
                    "posenet_output_abs_max": 7.62939453125e-06,
                    "posenet_component_abs_max": 9.713470479344455e-12,
                    "segnet_logit_abs_max": 0.0007913112640380859,
                    "segnet_argmax_diff_pixels": int(segnet_argmax_pixels),
                    "segnet_argmax_diff_fraction": segnet_argmax_fraction,
                },
            }
        ],
    }


def _mlx_score_calibration_payload(*, uncertain_count: int = 0) -> dict:
    certified_count = 1 if uncertain_count == 0 else 0
    return {
        "schema_version": "mlx_score_calibration.v1",
        "run_id": "unit",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "decision_policy": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "decision_safety_factor": 5.0,
            "calibration_uncertainty_basis": "mlx_minus_cpu_max_abs",
            "calibration_uncertainty_score": 1.0e-5,
            "recommended_min_mlx_gap_for_spend_triage": 5.0e-5,
            "allowed_use": "local_spend_triage_only_after_strict_auth_axis_calibration",
            "forbidden_use": "score_claim_or_rank_or_kill_or_promotion",
        },
        "summary": {
            "mlx_spend_triage_pairwise_certified_count": certified_count,
            "mlx_spend_triage_pairwise_uncertain_count": uncertain_count,
            "mlx_spend_triage_pairwise_total_count": certified_count + uncertain_count,
            "recommended_min_mlx_gap_for_spend_triage": 5.0e-5,
            "calibration_uncertainty_score": 1.0e-5,
        },
    }


def _mlx_production_contract_payload(*, passed: bool = True) -> dict:
    blockers = [] if passed else ["score_calibration_manifest_not_supplied"]
    return {
        "schema_version": "mlx_scorer_production_contract.v2",
        "gate_set_version": "mlx_scorer_production_gate_set.v2.cache_auth_torch_profile",
        "run_id": "unit",
        "passed": passed,
        "advisory_passed": passed,
        "verdict": (
            "PASS_MLX_SCORER_PRODUCTION_CONTRACT"
            if passed
            else "FAIL_MLX_SCORER_PRODUCTION_CONTRACT"
        ),
        "blockers": blockers,
        "warnings": ["batch_invariance_not_required_for_singleton_response"] if passed else [],
        "production_deployment_role": "local_mlx_scorer_acceleration_non_authoritative",
        "score_authority": False,
        "contest_authority": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "score_axis": EVIDENCE_TAG_MLX,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "response_summary": {
            "schema_version": "mlx_scorer_response.v1",
            "hardware_substrate": "MLX cpu",
            "batch_pairs": 1,
            "n_samples": 600,
            "pair_window": [0, 600],
            "archive_sha256": "a" * 64,
            "inflated_outputs_aggregate_sha256": "e" * 64,
        },
        "required_gates": {
            "cache_identity": passed,
            "cache_auth_audit": passed,
            "torch_parity": passed,
            "reference_torch_parity": passed,
            "profile_stability": passed,
            "batch_invariance": False,
            "batch_invariance_policy_requested": True,
            "score_calibration": passed,
            "strict_gate_policy": passed,
        },
        "authority_status": "non-authoritative local MLX production signal",
    }


def _mlx_production_contract_for_window(
    *,
    run_id: str,
    pair_window: list[int],
    archive_sha256: str = "a" * 64,
    inflated_outputs_aggregate_sha256: str = "e" * 64,
    response_run_id: str | None = None,
    candidate_cache_array_sha256: dict | None = None,
    reference_cache_array_sha256: dict | None = None,
    posenet_sha256: str | None = None,
    segnet_sha256: str | None = None,
    passed: bool = True,
) -> dict:
    payload = json.loads(json.dumps(_mlx_production_contract_payload(passed=passed)))
    payload["run_id"] = run_id
    payload["response_summary"]["pair_window"] = pair_window
    payload["response_summary"]["archive_sha256"] = archive_sha256
    payload["response_summary"]["inflated_outputs_aggregate_sha256"] = (
        inflated_outputs_aggregate_sha256
    )
    if response_run_id is not None:
        payload["response_summary"]["response_run_id"] = response_run_id
    if candidate_cache_array_sha256 is not None:
        payload["response_summary"]["candidate_cache_array_sha256"] = (
            candidate_cache_array_sha256
        )
    if reference_cache_array_sha256 is not None:
        payload["response_summary"]["reference_cache_array_sha256"] = (
            reference_cache_array_sha256
        )
    if posenet_sha256 is not None:
        payload["response_summary"]["posenet_sha256"] = posenet_sha256
    if segnet_sha256 is not None:
        payload["response_summary"]["segnet_sha256"] = segnet_sha256
    return payload


def _mlx_production_contract_bundle(*contracts: dict) -> dict:
    return {
        "schema": "mlx_scorer_production_contract_bundle.v1",
        "producer": "unit-test",
        "run_id": "unit-bundle",
        "passed": True,
        "verdict": "PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE",
        "score_authority": False,
        "contest_authority": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "score_axis": EVIDENCE_TAG_MLX,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "contracts": list(contracts),
    }


def _mlx_response_row(
    *,
    row_id: str,
    pair_window: list[int],
    archive_sha256: str = "a" * 64,
    inflated_outputs_aggregate_sha256: str = "e" * 64,
    source_run_id: str | None = None,
    candidate_cache_array_sha256: dict | None = None,
    reference_cache_array_sha256: dict | None = None,
    posenet_sha256: str | None = None,
    segnet_sha256: str | None = None,
) -> dict:
    row = {
        "schema": "scorer_response_row.v1",
        "row_id": row_id,
        "family": "mlx_scorer_response",
        "delta_vs_baseline_score": 1.0e-6,
        "scorer_delta_vs_baseline": 1.0e-6,
        "byte_budget_margin_vs_break_even": None,
        "archive_sha256": archive_sha256,
        "source_inflated_outputs_aggregate_sha256": (
            inflated_outputs_aggregate_sha256
        ),
        "source_batch_pairs": 1,
        "source_n_samples": 600,
        "source_pair_window": pair_window,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    if source_run_id is not None:
        row["source_run_id"] = source_run_id
    if candidate_cache_array_sha256 is not None:
        row["source_candidate_cache_array_sha256"] = candidate_cache_array_sha256
    if reference_cache_array_sha256 is not None:
        row["source_reference_cache_array_sha256"] = reference_cache_array_sha256
    if posenet_sha256 is not None:
        row["source_posenet_sha256"] = posenet_sha256
    if segnet_sha256 is not None:
        row["source_segnet_sha256"] = segnet_sha256
    return row


def _attach_mlx_identity_to_rows(dataset: dict) -> dict:
    for row in dataset["rows"]:
        if row.get("family") != "mlx_scorer_response":
            continue
        row["archive_sha256"] = "a" * 64
        row["source_inflated_outputs_aggregate_sha256"] = "e" * 64
        row["source_batch_pairs"] = 1
        row["source_n_samples"] = 600
        row["source_pair_window"] = [0, 600]
    return dataset


def test_build_response_dataset_normalizes_single_candidate(tmp_path) -> None:
    path = tmp_path / "scorer_gradient.json"
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "summary": {
            "component": "pose",
            "pair_indices": [7],
            "changed_pixel_count": 3,
            "changed_byte_count": 8,
            "packed_bytes": 12,
            "delta_vs_baseline_score": 0.1,
        },
        "candidate": {
            "advisory_eval": _advisory(1.25, 110, 0.004, 0.010),
            "inputs": {"target_raw_sha256": "c" * 64},
            "plan": {"selected_gain_sum": 5.5, "n_kept": 3},
            "local_pair_evals": [
                {
                    "delta": {"pose_dist_delta": -0.25, "seg_dist_delta": 0.0},
                    "worse_or_null": False,
                }
            ],
        },
        "authority": {"score_claim": False, "promotion_blockers": ["advisory"]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    baseline = ResponseBaseline(score=1.0, archive_bytes=100)

    dataset = build_response_dataset([path], baseline=baseline)

    assert dataset["score_claim"] is False
    assert dataset["rank_or_kill_eligible"] is False
    assert dataset["promotable"] is False
    assert dataset["authority"]["rank_or_kill_eligible"] is False
    assert dataset["authority"]["promotable"] is False
    assert dataset["summary"]["row_count"] == 1
    row = dataset["rows"][0]
    assert row["rank_or_kill_eligible"] is False
    assert row["promotable"] is False
    assert row["family"] == "scorer_gradient_sparse_residual"
    assert row["delta_vs_baseline_score"] == 0.25
    expected_rate_delta = 25.0 * 10.0 / CONTEST_UNCOMPRESSED_BYTES
    assert math.isclose(row["rate_delta_vs_baseline"], expected_rate_delta)
    assert math.isclose(row["scorer_delta_vs_baseline"], 0.25 - expected_rate_delta)
    assert row["added_archive_bytes"] == 10
    assert math.isclose(row["required_scorer_gain_for_added_bytes"], expected_rate_delta)
    assert row["observed_scorer_gain_vs_baseline"] == 0.0
    assert math.isclose(row["scorer_gain_shortfall_to_break_even"], expected_rate_delta)
    assert row["break_even_added_bytes_from_scorer_gain"] is None
    assert row["byte_budget_margin_vs_break_even"] is None
    assert row["local_pose_delta_sum"] == -0.25
    assert row["target_raw_sha256"] == "c" * 64
    assert row["holdout_fold"] in {0, 1, 2, 3, 4}


def test_build_response_dataset_accepts_direct_mlx_scorer_response_payload(tmp_path) -> None:
    path = tmp_path / "mlx_response.json"
    path.write_text(json.dumps(_mlx_response_payload()), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )

    assert dataset["score_claim"] is False
    assert dataset["summary"]["family_counts"]["mlx_scorer_response"] == 1
    row = dataset["rows"][0]
    assert row["family"] == "mlx_scorer_response"
    assert row["axis"] == EVIDENCE_TAG_MLX
    assert row["authority_source_score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["advisory_score_report_derived"] == 0.900001
    assert row["archive_bytes"] == 110
    assert row["added_archive_bytes"] == 10
    assert row["source_schema"] == "mlx_scorer_response.v1"
    assert row["source_evidence_grade"] == EVIDENCE_GRADE_MLX
    assert row["source_batch_pairs"] == 2
    assert row["source_pair_window"] == [16, 20]
    assert row["source_posenet_sha256"] == "p" * 64


def test_build_response_dataset_honors_explicit_mlx_response_family(tmp_path) -> None:
    path = tmp_path / "mlx_decoder_q_response.json"
    payload = _mlx_response_payload()
    payload["response_family"] = "mlx_decoder_q"
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )

    assert dataset["summary"]["family_counts"] == {"mlx_decoder_q": 1}
    assert dataset["rows"][0]["family"] == "mlx_decoder_q"


def test_build_response_dataset_rejects_invalid_mlx_response_family(tmp_path) -> None:
    path = tmp_path / "mlx_response.json"
    payload = _mlx_response_payload()
    payload["response_family"] = "Decoder Q"
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert "response_family may contain only lowercase" in dataset["skipped"][0]["reason"]


def test_next_probe_plan_prioritizes_mlx_response_harvest() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    "archive_sha256": "a" * 64,
                    "source_inflated_outputs_aggregate_sha256": "e" * 64,
                    "source_batch_pairs": 1,
                    "source_n_samples": 600,
                    "source_pair_window": [0, 600],
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
        mlx_production_contract=_mlx_production_contract_payload(),
    )

    assert plan["score_claim"] is False
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "strict_pass"
    assert plan["mlx_score_calibration_gate"]["status"] == "strict_pass"
    assert plan["mlx_production_contract_gate"]["status"] == "strict_pass"
    assert plan["probes"][0]["probe_id"] == "ll_mlx_cpu_stable_response_harvest"
    assert plan["probes"][0]["input_rows"] == ["mlx-row-1"]
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_score_calibration"
        not in rules
    )
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        not in rules
    )


def test_mlx_torch_parity_sweep_gate_blocks_failed_strict_sweep() -> None:
    gate = build_mlx_torch_parity_sweep_gate(_mlx_parity_sweep_payload(passed=False))

    assert gate["score_claim"] is False
    assert gate["status"] == "blocked"
    assert gate["mlx_rows_allowed_for_planner"] is False
    assert gate["summary"]["failed_windows"] == 1
    assert gate["summary"]["segnet_argmax_diff_pixels_max"] == 1.0
    assert gate["failed_rows"][0]["pair_window"] == [156, 160]


def test_mlx_score_calibration_gate_passes_certified_decision_band() -> None:
    gate = build_mlx_score_calibration_gate(_mlx_score_calibration_payload())

    assert gate["score_claim"] is False
    assert gate["status"] == "strict_pass"
    assert gate["mlx_spend_triage_allowed"] is True
    assert gate["summary"]["recommended_min_mlx_gap_for_spend_triage"] == 5.0e-5


def test_mlx_production_contract_gate_passes_strict_bundle() -> None:
    gate = build_mlx_production_contract_gate(_mlx_production_contract_payload())

    assert gate["score_claim"] is False
    assert gate["status"] == "strict_pass"
    assert gate["mlx_spend_triage_allowed"] is True
    assert gate["summary"]["batch_pairs"] == 1
    assert gate["summary"]["n_samples"] == 600
    assert gate["summary"]["required_gates"]["score_calibration"] is True


def test_mlx_production_contract_gate_blocks_failed_bundle() -> None:
    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_payload(passed=False)
    )

    assert gate["status"] == "blocked"
    assert gate["mlx_spend_triage_allowed"] is False
    assert "mlx_production_contract_not_passed" in gate["blockers"]
    assert (
        "mlx_production_contract_required_gate_score_calibration_not_true"
        in gate["blockers"]
    )


def test_mlx_production_contract_gate_blocks_advisory_contract_without_blockers() -> None:
    payload = _mlx_production_contract_payload()
    payload["passed"] = False
    payload["verdict"] = "ADVISORY_MLX_SCORER_DEV_CONTRACT"
    payload["blockers"] = []
    payload["warnings"] = ["production_required_gate_policy_bypassed"]
    payload["required_gates"]["strict_gate_policy"] = False

    gate = build_mlx_production_contract_gate(payload)

    assert gate["status"] == "blocked"
    assert "mlx_production_contract_not_passed" in gate["blockers"]
    assert "mlx_production_contract_verdict_not_pass" in gate["blockers"]


def test_mlx_production_contract_gate_rejects_false_authority_breach() -> None:
    payload = _mlx_production_contract_payload()
    payload["score_claim"] = True

    with pytest.raises(
        ScorerResponseDatasetError,
        match="MLX production contract score_claim must be false",
    ):
        build_mlx_production_contract_gate(payload)


def test_mlx_production_contract_gate_blocks_row_identity_mismatch() -> None:
    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_payload(),
        rows=[
            {
                "row_id": "mlx-row-1",
                "archive_sha256": "b" * 64,
                "source_inflated_outputs_aggregate_sha256": "e" * 64,
                "source_batch_pairs": 1,
                "source_n_samples": 600,
                "source_pair_window": [0, 600],
            }
        ],
    )

    assert gate["status"] == "blocked"
    assert "mlx_production_contract_row_archive_sha256_mismatch:mlx-row-1" in gate[
        "blockers"
    ]


def test_mlx_production_contract_bundle_gate_requires_every_mlx_row_match() -> None:
    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
            ),
            _mlx_production_contract_for_window(
                run_id="window-600-1200",
                pair_window=[600, 1200],
            ),
        ),
        rows=[
            _mlx_response_row(row_id="mlx-row-1", pair_window=[0, 600]),
            _mlx_response_row(row_id="mlx-row-2", pair_window=[600, 1200]),
        ],
    )

    assert gate["source_schema"] == "mlx_scorer_production_contract_bundle.v1"
    assert gate["status"] == "strict_pass"
    assert gate["mlx_spend_triage_allowed"] is True
    assert gate["summary"]["contract_count"] == 2
    assert gate["summary"]["strict_contract_count"] == 2
    assert gate["summary"]["row_count"] == 2
    assert gate["summary"]["matched_row_count"] == 2
    assert gate["summary"]["unmatched_row_ids"] == []


def test_mlx_production_contract_bundle_gate_blocks_uncovered_mlx_row() -> None:
    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
            ),
        ),
        rows=[
            _mlx_response_row(row_id="mlx-row-1", pair_window=[0, 600]),
            _mlx_response_row(row_id="mlx-row-2", pair_window=[600, 1200]),
        ],
    )

    assert gate["status"] == "blocked"
    assert gate["mlx_spend_triage_allowed"] is False
    assert "mlx-row-2" in gate["summary"]["unmatched_row_ids"]
    assert "mlx_production_contract_bundle_row_unmatched:mlx-row-2" in gate[
        "blockers"
    ]


def test_mlx_production_contract_bundle_gate_parent_window_covers_child_rows() -> None:
    cache_hashes = {
        "pair_indices": "0" * 64,
        "posenet_yuv6_pair": "1" * 64,
        "segnet_last_rgb": "2" * 64,
    }
    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
                candidate_cache_array_sha256=cache_hashes,
                reference_cache_array_sha256=cache_hashes,
                posenet_sha256="p" * 64,
                segnet_sha256="s" * 64,
            ),
        ),
        rows=[
            _mlx_response_row(
                row_id="mlx-row-1",
                pair_window=[218, 219],
                candidate_cache_array_sha256=cache_hashes,
                reference_cache_array_sha256=cache_hashes,
                posenet_sha256="a" * 64,
                segnet_sha256="b" * 64,
            )
        ],
    )

    assert gate["status"] == "strict_pass"
    assert gate["summary"]["row_count"] == 1
    assert gate["summary"]["matched_row_count"] == 1
    assert gate["summary"]["parent_window_matched_row_count"] == 1
    assert gate["summary"]["unmatched_row_count"] == 0


def test_mlx_production_contract_bundle_gate_parent_window_requires_cache_identity() -> None:
    contract_cache = {
        "pair_indices": "0" * 64,
        "posenet_yuv6_pair": "1" * 64,
        "segnet_last_rgb": "2" * 64,
    }
    row_cache = {
        "pair_indices": "0" * 64,
        "posenet_yuv6_pair": "3" * 64,
        "segnet_last_rgb": "2" * 64,
    }
    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
                candidate_cache_array_sha256=contract_cache,
                reference_cache_array_sha256=contract_cache,
            ),
        ),
        rows=[
            _mlx_response_row(
                row_id="mlx-row-1",
                pair_window=[218, 219],
                candidate_cache_array_sha256=row_cache,
                reference_cache_array_sha256=contract_cache,
            )
        ],
    )

    assert gate["status"] == "blocked"
    assert gate["summary"]["matched_row_count"] == 0
    assert gate["summary"]["parent_window_matched_row_count"] == 0
    assert gate["summary"]["unmatched_row_count"] == 1
    assert "mlx_production_contract_bundle_row_unmatched:mlx-row-1" in gate[
        "blockers"
    ]


def test_mlx_production_contract_bundle_gate_blocks_failed_bundle_verdict() -> None:
    bundle = _mlx_production_contract_bundle(
        _mlx_production_contract_for_window(
            run_id="window-0-600",
            pair_window=[0, 600],
        ),
    )
    bundle["passed"] = False
    bundle["verdict"] = "FAIL_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE"

    gate = build_mlx_production_contract_gate(
        bundle,
        rows=[_mlx_response_row(row_id="mlx-row-1", pair_window=[0, 600])],
    )

    assert gate["status"] == "blocked"
    assert gate["source_passed"] is False
    assert "mlx_production_contract_bundle_not_passed" in gate["blockers"]
    assert "mlx_production_contract_bundle_verdict_not_pass" in gate["blockers"]


def test_mlx_production_contract_bundle_gate_blocks_cache_detail_mismatch() -> None:
    contract_cache = {
        "pair_indices": "0" * 64,
        "posenet_yuv6_pair": "1" * 64,
        "segnet_last_rgb": "2" * 64,
    }
    row_cache = {
        "pair_indices": "0" * 64,
        "posenet_yuv6_pair": "3" * 64,
        "segnet_last_rgb": "2" * 64,
    }

    gate = build_mlx_production_contract_gate(
        _mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
                response_run_id="response-a",
                candidate_cache_array_sha256=contract_cache,
                reference_cache_array_sha256=contract_cache,
                posenet_sha256="p" * 64,
                segnet_sha256="s" * 64,
            ),
        ),
        rows=[
            _mlx_response_row(
                row_id="mlx-row-1",
                pair_window=[0, 600],
                source_run_id="response-a",
                candidate_cache_array_sha256=row_cache,
                reference_cache_array_sha256=contract_cache,
                posenet_sha256="p" * 64,
                segnet_sha256="s" * 64,
            )
        ],
    )

    assert gate["status"] == "blocked"
    assert "mlx_production_contract_bundle_row_detail_mismatch:mlx-row-1" in gate[
        "blockers"
    ]
    assert (
        "mlx_production_contract_row_candidate_cache_array_sha256_mismatch:mlx-row-1"
        in gate["blockers"]
    )


def test_next_probe_plan_accepts_mlx_production_contract_bundle() -> None:
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 2},
            "rows": [
                _mlx_response_row(row_id="mlx-row-1", pair_window=[0, 600]),
                _mlx_response_row(row_id="mlx-row-2", pair_window=[600, 1200]),
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
        mlx_production_contract=_mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
            ),
            _mlx_production_contract_for_window(
                run_id="window-600-1200",
                pair_window=[600, 1200],
            ),
        ),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert plan["mlx_production_contract_gate"]["status"] == "strict_pass"
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        not in rules
    )
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_failed_production_contract"
        not in rules
    )
    assert plan["effective_mlx_spend_triage_gate"]["status"] == "blocked"
    assert (
        "response_validation_gate_not_passed"
        in plan["effective_mlx_spend_triage_gate"]["blockers"]
    )
    assert (
        plan["effective_mlx_spend_triage_gate"][
            "mlx_exact_eval_spend_triage_allowed"
        ]
        is False
    )
    assert plan["probes"][0]["probe_id"] == "ll_mlx_cpu_stable_response_harvest"
    assert plan["probes"][0]["input_rows"] == ["mlx-row-1", "mlx-row-2"]


def test_next_probe_plan_effective_mlx_spend_triage_gate_passes_only_after_dataset_validation() -> None:
    dataset = _attach_mlx_identity_to_rows(
        _validation_dataset(
            families=("mlx_scorer_response", "decoder_q"),
            rows_per_fold=5,
            include_prediction=True,
        )
    )

    plan = build_next_probe_plan(
        dataset,
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
        mlx_production_contract=_mlx_production_contract_payload(),
    )

    assert plan["response_validation_gate"]["status"] == "passed"
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "strict_pass"
    assert plan["mlx_score_calibration_gate"]["status"] == "strict_pass"
    assert plan["mlx_production_contract_gate"]["status"] == "strict_pass"
    gate = plan["effective_mlx_spend_triage_gate"]
    assert gate["status"] == "strict_pass"
    assert gate["mlx_exact_eval_spend_triage_allowed"] is True
    assert gate["blockers"] == []
    assert gate["summary"]["mlx_row_count"] == 25
    assert gate["summary"]["production_contract_row_count"] == 25
    assert gate["summary"]["production_contract_matched_row_count"] == 25
    assert gate["summary"]["production_contract_unmatched_row_count"] == 0
    assert gate["summary"]["production_contract_unmatched_row_ids_sample"] == []
    assert gate["summary"]["production_contract_blockers_sample"] == []


def test_next_probe_plan_effective_mlx_spend_triage_gate_surfaces_contract_coverage_blocker() -> None:
    dataset = _attach_mlx_identity_to_rows(
        _validation_dataset(
            families=("mlx_scorer_response", "decoder_q"),
            rows_per_fold=5,
            include_prediction=True,
        )
    )

    plan = build_next_probe_plan(
        dataset,
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
        mlx_production_contract=_mlx_production_contract_bundle(
            _mlx_production_contract_for_window(
                run_id="window-600-1200",
                pair_window=[600, 1200],
            ),
        ),
    )

    assert plan["response_validation_gate"]["status"] == "passed"
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "strict_pass"
    assert plan["mlx_score_calibration_gate"]["status"] == "strict_pass"
    assert plan["mlx_production_contract_gate"]["status"] == "blocked"
    gate = plan["effective_mlx_spend_triage_gate"]
    assert gate["status"] == "blocked"
    assert gate["mlx_exact_eval_spend_triage_allowed"] is False
    assert "mlx_production_contract_gate_not_strict_pass" in gate["blockers"]
    summary = gate["summary"]
    assert summary["mlx_row_count"] == 25
    assert summary["response_validation_status"] == "passed"
    assert summary["torch_parity_status"] == "strict_pass"
    assert summary["score_calibration_status"] == "strict_pass"
    assert summary["production_contract_status"] == "blocked"
    assert summary["production_contract_row_count"] == 25
    assert summary["production_contract_matched_row_count"] == 0
    assert summary["production_contract_unmatched_row_count"] == 25
    assert summary["production_contract_unmatched_row_ids_sample"] == [
        "mlx_scorer_response-0-0",
        "mlx_scorer_response-0-1",
        "mlx_scorer_response-0-2",
        "mlx_scorer_response-0-3",
        "mlx_scorer_response-0-4",
        "mlx_scorer_response-1-0",
        "mlx_scorer_response-1-1",
        "mlx_scorer_response-1-2",
    ]
    assert "mlx_production_contract_bundle_row_unmatched:mlx_scorer_response-0-0" in (
        summary["production_contract_blockers_sample"]
    )
    rendered = render_next_probe_plan_markdown(plan)
    assert "- Production contract rows: `25`" in rendered
    assert "- Production contract matched rows: `0`" in rendered
    assert "- Production contract unmatched rows: `25`" in rendered


def test_next_probe_plan_requires_mlx_parity_sweep_for_mlx_rows() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        }
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_use_mlx_rows_without_torch_parity_sweep" in rules
    assert plan["mlx_torch_parity_sweep_gate"] is None
    assert plan["probes"][0]["probe_id"] == "ll_mlx_torch_parity_sweep_required"
    assert "ll_mlx_cpu_stable_response_harvest" not in {
        probe["probe_id"] for probe in plan["probes"]
    }
    assert "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_score_calibration" in rules
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        in rules
    )


def test_next_probe_plan_attaches_passing_mlx_score_calibration_gate() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_score_calibration" not in rules
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_uncertain_calibration"
        not in rules
    )
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        in rules
    )
    assert plan["mlx_score_calibration_gate"]["status"] == "strict_pass"
    assert (
        plan["probes"][0]["mlx_score_calibration_gate"]["summary"][
            "recommended_min_mlx_gap_for_spend_triage"
        ]
        == 5.0e-5
    )


def test_next_probe_plan_gates_explicit_mlx_response_family_by_source_schema() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "pr101_pose_axis_strict_calibration",
                    "source_schema": "mlx_scorer_response.v1",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        in rules
    )
    assert plan["probes"][0]["input_rows"] == ["mlx-row-1"]


def test_next_probe_plan_blocks_uncertain_mlx_score_calibration_gate() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(uncertain_count=1),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_uncertain_calibration"
        in rules
    )
    assert plan["mlx_score_calibration_gate"]["status"] == "blocked"
    assert plan["mlx_score_calibration_gate"]["mlx_spend_triage_allowed"] is False


def test_next_probe_plan_blocks_failed_mlx_production_contract() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=True),
        mlx_score_calibration=_mlx_score_calibration_payload(),
        mlx_production_contract=_mlx_production_contract_payload(passed=False),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_failed_production_contract"
        in rules
    )
    assert plan["mlx_production_contract_gate"]["status"] == "blocked"
    assert plan["mlx_production_contract_gate"]["mlx_spend_triage_allowed"] is False


def test_next_probe_plan_blocks_failed_mlx_parity_sweep_without_override() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=False),
    )

    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "blocked"
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_use_mlx_rows_after_failed_strict_parity_sweep" in rules
    assert plan["probes"][0]["probe_id"] == "ll_mlx_torch_parity_repair_or_override"
    assert "ll_mlx_cpu_stable_response_harvest" not in {
        probe["probe_id"] for probe in plan["probes"]
    }


def test_next_probe_plan_allows_failed_mlx_parity_sweep_with_research_override() -> None:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    plan = build_next_probe_plan(
        {
            "schema": "scorer_response_dataset.v1",
            "summary": {"row_count": 1},
            "rows": [
                {
                    "schema": "scorer_response_row.v1",
                    "row_id": "mlx-row-1",
                    "family": "mlx_scorer_response",
                    "delta_vs_baseline_score": 1.0e-6,
                    "scorer_delta_vs_baseline": 1.0e-6,
                    "byte_budget_margin_vs_break_even": None,
                    **false_authority,
                }
            ],
        },
        mlx_torch_parity_sweep=_mlx_parity_sweep_payload(passed=False),
        allow_mlx_parity_research_signal_override=True,
    )

    gate = plan["mlx_torch_parity_sweep_gate"]
    assert gate["status"] == "research_signal_override"
    assert gate["score_claim"] is False
    assert gate["ready_for_exact_eval_dispatch"] is False
    assert plan["probes"][0]["probe_id"] == "ll_mlx_cpu_stable_response_harvest"
    assert plan["probes"][0]["mlx_torch_parity_gate"]["status"] == "research_signal_override"
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        in rules
    )


def test_build_windowed_mlx_response_dataset_uses_matching_window_baseline(tmp_path) -> None:
    baseline = _mlx_response_payload()
    baseline["canonical_score"] = 0.9
    baseline["score_recomputed_from_components"] = 0.9
    baseline["archive_size_bytes"] = 100
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    candidate = _mlx_response_payload()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    dataset = build_windowed_mlx_response_dataset(
        candidate_paths=[candidate_path],
        baseline_paths=[baseline_path],
    )

    assert dataset["score_claim"] is False
    assert dataset["baseline"]["mode"] == "per_window_mlx_response"
    assert dataset["summary"]["row_count"] == 1
    row = dataset["rows"][0]
    assert row["family"] == "mlx_scorer_response"
    assert row["window_baseline_source_path"] == str(baseline_path)
    assert row["window_baseline_key"] == "start=16:max=4:window=16-20"
    assert row["window_baseline_score"] == 0.9
    assert row["window_baseline_archive_bytes"] == 100
    assert row["window_baseline_avg_posenet_dist"] == 0.004
    assert row["window_baseline_avg_segnet_dist"] == 0.010
    assert math.isclose(row["window_baseline_pose_term"], math.sqrt(0.04))
    assert row["window_baseline_seg_term"] == 1.0
    assert row["window_baseline_scorer_term"] == 1.2
    assert row["delta_vs_baseline_score"] == 0.0000010000000000287557
    assert row["added_archive_bytes"] == 10
    rendered = render_markdown(dataset)
    assert EVIDENCE_TAG_MLX in rendered
    assert "Ready for exact-eval dispatch: `False`" in rendered


def test_build_windowed_mlx_response_dataset_accepts_reference_cache_baseline_identity(
    tmp_path,
) -> None:
    baseline = _mlx_response_payload()
    baseline["canonical_score"] = 0.9
    baseline["score_recomputed_from_components"] = 0.9
    baseline["archive_sha256"] = None
    baseline["inflated_outputs_aggregate_sha256"] = None
    baseline["raw_sha256"] = None
    baseline["cache_identity"]["candidate"]["archive_sha256"] = None
    baseline["cache_identity"]["candidate"]["inflated_outputs_aggregate_sha256"] = None
    baseline["cache_identity"]["candidate"]["raw_sha256"] = None
    baseline_path = tmp_path / "baseline_reference_cache.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    candidate = _mlx_response_payload()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    dataset = build_windowed_mlx_response_dataset(
        candidate_paths=[candidate_path],
        baseline_paths=[baseline_path],
    )

    assert dataset["summary"]["row_count"] == 1
    assert dataset["skipped"] == []


def test_refresh_mlx_scorer_response_source_identity_restores_missing_fields(
    tmp_path,
) -> None:
    source_path = tmp_path / "candidate.json"
    source_path.write_text(json.dumps(_mlx_response_payload()), encoding="utf-8")
    dataset = build_response_dataset(
        [source_path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )
    row = dataset["rows"][0]
    row.pop("source_inflated_outputs_aggregate_sha256")
    row.pop("source_candidate_cache_array_sha256")
    row.pop("source_reference_cache_array_sha256")
    row.pop("source_posenet_sha256")
    row.pop("source_segnet_sha256")

    refreshed = refresh_mlx_scorer_response_source_identity(dataset)

    refresh = refreshed["source_identity_refresh"]
    assert refresh["passed"] is True
    assert refresh["mlx_row_count"] == 1
    assert refresh["refreshed_row_count"] == 1
    assert refresh["updated_row_count"] == 1
    refreshed_row = refreshed["rows"][0]
    assert refreshed_row["source_inflated_outputs_aggregate_sha256"] == "e" * 64
    assert refreshed_row["source_candidate_cache_array_sha256"] == {
        "pair_indices": "0" * 64,
        "posenet_yuv6_pair": "3" * 64,
        "segnet_last_rgb": "4" * 64,
    }
    assert refreshed["summary"]["mlx_source_identity_refresh_passed"] is True


def test_refresh_mlx_scorer_response_source_identity_blocks_mismatch(tmp_path) -> None:
    source_path = tmp_path / "candidate.json"
    source_path.write_text(json.dumps(_mlx_response_payload()), encoding="utf-8")
    dataset = build_response_dataset(
        [source_path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )
    dataset["rows"][0]["archive_sha256"] = "b" * 64

    refreshed = refresh_mlx_scorer_response_source_identity(dataset)

    refresh = refreshed["source_identity_refresh"]
    assert refresh["passed"] is False
    assert (
        f"source_identity_field_mismatch:{dataset['rows'][0]['row_id']}:archive_sha256"
        in refresh["blockers"]
    )


def test_build_windowed_mlx_response_dataset_skips_missing_window_baseline(tmp_path) -> None:
    baseline = _mlx_response_payload()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    candidate = _mlx_response_payload()
    candidate["start_pair"] = 20
    candidate["pair_window"] = [20, 24]
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    dataset = build_windowed_mlx_response_dataset(
        candidate_paths=[candidate_path],
        baseline_paths=[baseline_path],
    )

    assert dataset["rows"] == []
    assert "no matching baseline window" in dataset["skipped"][0]["reason"]


def test_build_windowed_mlx_response_dataset_rejects_duplicate_baseline_window(
    tmp_path,
) -> None:
    baseline_a = _mlx_response_payload()
    baseline_a_path = tmp_path / "baseline_a.json"
    baseline_a_path.write_text(json.dumps(baseline_a), encoding="utf-8")
    baseline_b = _mlx_response_payload()
    baseline_b["canonical_score"] = 0.91
    baseline_b["score_recomputed_from_components"] = 0.91
    baseline_b_path = tmp_path / "baseline_b.json"
    baseline_b_path.write_text(json.dumps(baseline_b), encoding="utf-8")

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_mlx_response_payload()), encoding="utf-8")

    dataset = build_windowed_mlx_response_dataset(
        candidate_paths=[candidate_path],
        baseline_paths=[baseline_a_path, baseline_b_path],
    )

    assert dataset["rows"] == []
    assert any("duplicate baseline window" in item["reason"] for item in dataset["skipped"])
    assert any("ambiguous duplicate baseline window" in item["reason"] for item in dataset["skipped"])


def test_build_mlx_window_response_dataset_cli(tmp_path) -> None:
    baseline = _mlx_response_payload()
    baseline["canonical_score"] = 0.9
    baseline["score_recomputed_from_components"] = 0.9
    baseline["archive_size_bytes"] = 100
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_mlx_response_payload()), encoding="utf-8")
    json_out = tmp_path / "dataset.json"
    md_out = tmp_path / "dataset.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_window_response_dataset.py"),
            "--candidate-response",
            str(candidate_path),
            "--baseline-response",
            str(baseline_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"score_claim": false' in completed.stdout
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["row_count"] == 1
    assert "mlx_scorer_response" in md_out.read_text(encoding="utf-8")


def test_build_mlx_window_response_dataset_cli_fails_empty_by_default(tmp_path) -> None:
    baseline = _mlx_response_payload()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    candidate = _mlx_response_payload()
    candidate["start_pair"] = 20
    candidate["pair_window"] = [20, 24]
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_window_response_dataset.py"),
            "--candidate-response",
            str(candidate_path),
            "--baseline-response",
            str(baseline_path),
            "--json-out",
            str(tmp_path / "dataset.json"),
        ],
        cwd=REPO,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "no usable MLX window response rows produced" in completed.stderr


def test_build_response_dataset_rejects_mlx_scorer_response_false_authority_breach(
    tmp_path,
) -> None:
    path = tmp_path / "mlx_response.json"
    payload = _mlx_response_payload()
    payload["score_claim"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert "score_claim must be explicit false" in dataset["skipped"][0]["reason"]


def test_build_response_dataset_rejects_mlx_response_without_exact_eval_flag(
    tmp_path,
) -> None:
    path = tmp_path / "mlx_response.json"
    payload = _mlx_response_payload()
    payload.pop("requires_exact_eval_before_promotion")
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert "requires_exact_eval_before_promotion must be true" in dataset["skipped"][0]["reason"]


def test_build_response_dataset_rejects_mlx_scorer_response_missing_cache_identity(
    tmp_path,
) -> None:
    path = tmp_path / "mlx_response.json"
    payload = _mlx_response_payload()
    payload.pop("cache_identity")
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=0.9, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert "cache_identity must be an object" in dataset["skipped"][0]["reason"]


def test_build_response_dataset_rejects_source_score_claim_true(tmp_path) -> None:
    path = tmp_path / "source_score_claim.json"
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(1.25, 110, 0.004, 0.010)},
        "authority": {"score_claim": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=1.0, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert dataset["skipped"]
    assert "source score_claim must be false" in dataset["skipped"][0]["reason"]


def test_build_response_dataset_requires_explicit_source_score_claim_false(
    tmp_path,
) -> None:
    path = tmp_path / "missing_source_authority.json"
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(1.25, 110, 0.004, 0.010)},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=1.0, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert dataset["skipped"]
    assert "source score_claim must be explicit false" in dataset["skipped"][0]["reason"]


def test_build_response_dataset_rejects_present_optional_authority_true(
    tmp_path,
) -> None:
    path = tmp_path / "ready_true.json"
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(1.25, 110, 0.004, 0.010)},
        "authority": {
            "score_claim": False,
            "ready_for_exact_eval_dispatch": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=1.0, archive_bytes=100),
    )

    assert dataset["rows"] == []
    assert dataset["skipped"]
    assert "ready_for_exact_eval_dispatch must be false" in dataset["skipped"][0]["reason"]


def test_build_response_dataset_distilled_vs_direct_rows_are_opt_in(tmp_path) -> None:
    path = tmp_path / "distilled_vs_direct.json"
    payload = {
        "schema": "distilled_vs_direct_scorer_paired_smoke.v1",
        "producer": "pact_nerv_distilled_scorer_stage1",
        "smoke_kind": "distilled_vs_direct_scorer_paired_smoke",
        "candidate": {
            "candidate_id": "pds_stage1_smoke",
            "advisory_eval": _advisory(0.99, 99, 0.003, 0.009),
            "summary": {"component": "distilled_scorer"},
        },
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    baseline = ResponseBaseline(score=1.0, archive_bytes=100)

    skipped = build_response_dataset([path], baseline=baseline)
    included = build_response_dataset(
        [path],
        baseline=baseline,
        include_distilled_vs_direct_rows=True,
    )

    assert skipped["rows"] == []
    assert "requires include_distilled_vs_direct_rows" in skipped["skipped"][0]["reason"]
    assert included["rows"][0]["family"] == "distilled_vs_direct_scorer_paired_smoke"
    assert included["rows"][0]["candidate_id"] == "pds_stage1_smoke"


def test_scorer_response_consumer_routing_invokes_opt_in_consumers(tmp_path) -> None:
    path = tmp_path / "distilled_vs_direct.json"
    payload = {
        "schema": "distilled_vs_direct_scorer_paired_smoke.v1",
        "producer": "pact_nerv_distilled_scorer_stage1",
        "smoke_kind": "distilled_vs_direct_scorer_paired_smoke",
        "candidate": {
            "candidate_id": "pds_stage1_smoke",
            "advisory_eval": _advisory(0.99, 99, 0.003, 0.009),
        },
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=1.0, archive_bytes=100),
        include_distilled_vs_direct_rows=True,
    )

    def _consume_candidate(row: dict) -> dict:
        assert row["family"] == "distilled_vs_direct_scorer_paired_smoke"
        return {
            "consumer_signal_kind": "unit_distilled_route",
            "predicted_delta_adjustment": 0.0,
            "axis_tag": "[predicted]",
            "rationale": "unit route",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        }

    consumer = SimpleNamespace(
        __name__="unit.consumer",
        CONSUMER_NAME="unit_distilled_consumer",
        CONSUMER_VERSION="0.1",
        CONSUMES_SCORER_RESPONSE_DATASET=True,
        consume_candidate=_consume_candidate,
    )
    routing = build_scorer_response_consumer_routing(
        dataset,
        consumer_modules=[consumer],
    )

    assert routing["schema"] == "scorer_response_dataset_consumer_routing.v1"
    assert routing["score_claim"] is False
    assert routing["score_claim_valid"] is False
    assert routing["consumer_count"] == 1
    assert routing["verdict_count"] == 1
    assert routing["verdicts"][0]["consumer_signal_kind"] == "unit_distilled_route"
    assert routing["verdicts"][0]["score_claim"] is False
    assert routing["verdicts"][0]["promotable"] is False


def _current_response_dataset_payload() -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return {
        "schema": "scorer_response_dataset.v1",
        "producer": "test",
        **false_authority,
        "authority": {
            **false_authority,
            "evidence_grade": "macOS-CPU advisory response dataset",
        },
        "summary": {"row_count": 1},
        "rows": [
            {
                "schema": "scorer_response_row.v1",
                "row_id": "row-a",
                **false_authority,
                "authority_source_score_claim": False,
                "advisory_score_report_derived": 1.0,
                "delta_vs_baseline_score": 0.0,
            }
        ],
    }


def _validation_dataset(
    *,
    families: tuple[str, ...],
    rows_per_fold: int,
    include_prediction: bool = False,
) -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    rows = []
    for family_index, family in enumerate(families):
        for fold in range(5):
            for offset in range(rows_per_fold):
                value = float(family_index * 100 + fold * rows_per_fold + offset)
                row = {
                    "schema": "scorer_response_row.v1",
                    "row_id": f"{family}-{fold}-{offset}",
                    "candidate_id": f"{family}-{fold}-{offset}",
                    **false_authority,
                    "authority_source_score_claim": False,
                    "family": family,
                    "axis": "[unit-test response-signal]",
                    "holdout_fold": fold,
                    "delta_vs_baseline_score": value,
                    "scorer_delta_vs_baseline": value,
                    "byte_budget_margin_vs_break_even": None,
                    "archive_bytes": 100,
                }
                if include_prediction:
                    row["predicted_delta_vs_baseline_score"] = value
                rows.append(row)
    return {
        "schema": "scorer_response_dataset.v1",
        "producer": "test",
        **false_authority,
        "authority": {
            **false_authority,
            "evidence_grade": "macOS-CPU advisory response dataset",
        },
        "summary": {
            "row_count": len(rows),
            "family_counts": dict.fromkeys(families, rows_per_fold * 5),
        },
        "rows": rows,
    }


def test_normalize_legacy_response_dataset_authority_backfills_extended_fields_only() -> None:
    payload = _current_response_dataset_payload()
    payload.pop("rank_or_kill_eligible")
    payload.pop("promotable")
    payload["authority"].pop("rank_or_kill_eligible")
    payload["authority"].pop("promotable")
    payload["rows"][0].pop("rank_or_kill_eligible")
    payload["rows"][0].pop("promotable")

    normalized = normalize_legacy_response_dataset_authority(
        payload,
        source_label="historical_pr110",
    )

    assert normalized["rank_or_kill_eligible"] is False
    assert normalized["promotable"] is False
    assert normalized["authority"]["rank_or_kill_eligible"] is False
    assert normalized["rows"][0]["promotable"] is False
    metadata = normalized["authority_normalization"]
    assert metadata["score_claim"] is False
    assert metadata["source_label"] == "historical_pr110"
    assert metadata["backfilled_missing_false_field_count"] == 6
    assert {
        (item["label"], item["field"])
        for item in metadata["backfilled_missing_false_fields"]
    } == {
        ("scorer-response dataset", "rank_or_kill_eligible"),
        ("scorer-response dataset", "promotable"),
        ("scorer-response dataset authority", "rank_or_kill_eligible"),
        ("scorer-response dataset authority", "promotable"),
        ("scorer-response row 0", "rank_or_kill_eligible"),
        ("scorer-response row 0", "promotable"),
    }


def test_normalize_legacy_response_dataset_authority_refuses_core_or_source_ambiguity() -> None:
    missing_core = _current_response_dataset_payload()
    missing_core.pop("score_claim")
    try:
        normalize_legacy_response_dataset_authority(missing_core)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing core authority rejection")

    source_claim = _current_response_dataset_payload()
    source_claim["rows"][0]["authority_source_score_claim"] = "true"
    try:
        normalize_legacy_response_dataset_authority(source_claim)
    except ScorerResponseDatasetError as exc:
        assert "authority_source_score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected source score-claim rejection")


def test_scorer_response_validation_gate_blocks_single_family_without_predictions() -> None:
    dataset = _validation_dataset(families=("mlx_scorer_response",), rows_per_fold=10)

    gate = build_scorer_response_validation_gate(dataset)

    assert gate["score_claim"] is False
    assert gate["status"] == "blocked"
    assert "family_count_below_min:1<2" in gate["blockers"]
    assert "families_with_required_folds_below_min:1<2" in gate["blockers"]
    assert "no_prediction_fields_present" in gate["blockers"]
    assert gate["coverage"]["row_count"] == 50
    rendered = render_validation_gate_markdown(gate)
    assert "Scorer Response Validation Gate" in rendered
    assert "## Authority" in rendered
    assert "Promotion eligible: `False`" in rendered


def test_scorer_response_validation_gate_passes_family_diverse_heldout_predictions() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "decoder_q"),
        rows_per_fold=5,
        include_prediction=True,
    )

    gate = build_scorer_response_validation_gate(dataset)

    assert gate["status"] == "passed"
    assert gate["blockers"] == []
    assert set(gate["coverage"]["families_with_required_folds"]) == {
        "decoder_q",
        "mlx_scorer_response",
    }
    assert gate["passing_prediction_fields"] == ["predicted_delta_vs_baseline_score"]


def test_out_of_fold_linear_predictions_can_satisfy_validation_gate() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "mlx_decoder_q"),
        rows_per_fold=8,
    )
    for row in dataset["rows"]:
        family_offset = 0.002 if row["family"] == "mlx_decoder_q" else 0.0
        fold = int(row["holdout_fold"])
        offset = int(row["candidate_id"].split("-")[-1])
        pair_start = fold * 8 + offset
        row["source_pair_window"] = [pair_start, pair_start + 1]
        row["delta_vs_baseline_score"] = 0.1 + 0.003 * pair_start + family_offset

    predicted = attach_out_of_fold_linear_predictions(dataset)
    gate = build_scorer_response_validation_gate(predicted)

    assert predicted["score_claim"] is False
    assert predicted["prediction_fit"]["score_claim"] is False
    assert predicted["prediction_fit"]["feature_set"] == "pair_family_archive_linear_v1"
    assert gate["status"] == "passed"
    assert gate["passing_prediction_fields"] == ["ll_predicted_delta_vs_baseline_score"]


def test_structural_features_feed_oof_predictions_without_authority() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "mlx_decoder_q"),
        rows_per_fold=8,
    )
    frame_axis = np.zeros((16, 3), dtype=np.float64)
    for pair in range(8):
        frame_axis[2 * pair, 1] = float(pair + 1)
        frame_axis[2 * pair + 1, 0] = float(10 + pair)
        frame_axis[2 * pair + 1, 1] = float(pair + 1)
    frame_decomposition = {
        "schema": "per_frame_decomposition_segnet_per_frame_posenet_per_pair_v1",
        "axis_labels": ["seg", "pose", "rate"],
    }
    mutation_manifest = {
        "archive_bin_bytes": 200,
        "mutation_row": {
            "mutation_id": "unit-decoder-q",
            "mutation": {
                "delta": 1,
                "q_offset": 0,
                "tensor_name": "rgb_1.weight",
            },
            "op3v3_target_evidence": {
                "score_impact_abs_sum": 0.25,
                "axis_share": {"seg": 0.75, "pose": 0.25, "rate": 0.0},
                "top_byte_count": 4,
                "approx_compressed_range": {"start": 20, "length": 10},
            },
        },
    }
    for row in dataset["rows"]:
        fold = int(row["holdout_fold"])
        offset = int(row["candidate_id"].split("-")[-1])
        pair_start = fold * 8 + offset
        row["source_pair_window"] = [pair_start, pair_start + 1]

    enriched = attach_structural_features(
        dataset,
        frame_axis_l1=frame_axis,
        frame_decomposition=frame_decomposition,
        decoder_q_mutation_manifest=mutation_manifest,
    )
    for row in enriched["rows"]:
        family_offset = 0.002 if row["family"] == "mlx_decoder_q" else 0.0
        row["delta_vs_baseline_score"] = (
            0.1
            + 0.0005 * row["diagnostic_total_pair_l1"]
            + family_offset
        )

    predicted = attach_out_of_fold_linear_predictions(enriched)
    gate = build_scorer_response_validation_gate(predicted)

    decoder_row = next(row for row in predicted["rows"] if row["family"] == "mlx_decoder_q")
    baseline_row = next(row for row in predicted["rows"] if row["family"] == "mlx_scorer_response")
    assert predicted["score_claim"] is False
    assert predicted["structural_features"]["score_claim"] is False
    assert predicted["structural_features"]["feature_write_counts"]["decoder_q_score_impact_abs_sum"] == 80
    assert predicted["structural_features"]["feature_nonzero_counts"]["decoder_q_score_impact_abs_sum"] == 40
    assert predicted["prediction_fit"]["feature_set"] == STRUCTURED_FEATURE_SET
    assert "diagnostic_total_pair_l1" in predicted["prediction_fit"]["feature_names"]
    assert "decoder_q_score_impact_abs_sum" in predicted["prediction_fit"]["feature_names"]
    assert decoder_row["decoder_q_score_impact_abs_sum"] == 0.25
    assert baseline_row["decoder_q_score_impact_abs_sum"] == 0.0
    assert gate["status"] == "passed"


def test_family_delta_matches_rows_by_source_start_pair() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "mlx_decoder_q"),
        rows_per_fold=3,
    )
    for row in dataset["rows"]:
        fold = int(row["holdout_fold"])
        offset = int(row["candidate_id"].split("-")[-1])
        pair_start = fold * 3 + offset
        row["source_pair_window"] = [pair_start, pair_start + 1]
        if row["family"] == "mlx_scorer_response":
            row["delta_vs_baseline_score"] = 0.1 + 0.001 * pair_start
            row["avg_posenet_dist"] = 0.01
            row["avg_segnet_dist"] = 0.02
        else:
            row["delta_vs_baseline_score"] = 0.2 + 0.001 * pair_start
            row["avg_posenet_dist"] = 0.011
            row["avg_segnet_dist"] = 0.019

    delta = build_family_delta(
        dataset,
        reference_family="mlx_scorer_response",
        candidate_family="mlx_decoder_q",
        top_k=2,
    )

    assert delta["score_claim"] is False
    assert delta["summary"]["matched_count"] == 15
    assert delta["summary"]["candidate_worse_count"] == 15
    assert math.isclose(delta["summary"]["score_delta_mean"], 0.1)
    assert math.isclose(
        delta["matched_rows"][0]["candidate_minus_reference_avg_posenet_dist"],
        0.001,
    )
    assert math.isclose(
        delta["matched_rows"][0]["candidate_minus_reference_avg_segnet_dist"],
        -0.001,
    )
    rendered = render_family_delta_markdown(delta)
    assert "Scorer Response Family Delta" in rendered
    assert "## Authority" in rendered
    assert "Ready for exact-eval dispatch: `False`" in rendered


def test_decoder_q_response_surface_classifies_preserve_and_suppress_windows() -> None:
    family_delta = {
        "schema": "scorer_response_family_delta.v1",
        "reference_family": "mlx_scorer_response",
        "candidate_family": "mlx_decoder_q",
        "match_key": "source_start_pair",
        "matched_rows": [
            {
                "match_key_value": "1",
                "candidate_minus_reference_delta_vs_baseline_score": -0.002,
                "candidate_minus_reference_seg_term": -0.003,
                "candidate_minus_reference_pose_term": 0.001,
            },
            {
                "match_key_value": "2",
                "candidate_minus_reference_delta_vs_baseline_score": 0.004,
                "candidate_minus_reference_seg_term": 0.004,
                "candidate_minus_reference_pose_term": 0.0,
            },
            {
                "match_key_value": "3",
                "candidate_minus_reference_delta_vs_baseline_score": 0.0,
                "candidate_minus_reference_seg_term": 0.0,
                "candidate_minus_reference_pose_term": 0.0,
            },
        ],
    }

    surface = build_decoder_q_response_surface(family_delta, top_k=2)

    assert surface["score_claim"] is False
    assert surface["summary"]["matched_count"] == 3
    assert surface["summary"]["preserve_candidate_effect_count"] == 1
    assert surface["summary"]["suppress_or_invert_candidate_effect_count"] == 1
    assert surface["summary"]["neutral_or_uncertain_count"] == 1
    assert surface["summary"]["preserve_gain_sum"] == 0.002
    assert surface["summary"]["suppress_harm_sum"] == 0.004
    assert surface["top_preserve_windows"][0]["recommended_action"] == "prefer_window_or_similar_axis_pattern"
    assert surface["top_suppress_windows"][0]["recommended_action"] == "penalize_window_or_try_opposite_sign"
    rendered = render_decoder_q_response_surface_markdown(surface)
    assert "Decoder-Q Response Surface" in rendered
    assert "## Authority" in rendered
    assert "Rank/kill eligible: `False`" in rendered


def test_next_probe_plan_can_prioritize_decoder_q_response_surface() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "mlx_decoder_q"),
        rows_per_fold=5,
        include_prediction=True,
    )
    surface = {
        "schema": "decoder_q_response_surface_plan.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "summary": {
            "matched_count": 10,
            "preserve_candidate_effect_count": 3,
            "suppress_or_invert_candidate_effect_count": 7,
            "neutral_or_uncertain_count": 0,
            "preserve_gain_sum": 0.01,
            "suppress_harm_sum": 0.05,
            "axis_dominance_counts": {"seg": 10},
        },
        "top_preserve_windows": [{"match_key_value": "1"}],
        "top_suppress_windows": [{"match_key_value": "2"}],
    }

    plan = build_next_probe_plan(dataset, decoder_q_response_surface=surface)

    assert plan["decoder_q_response_surface_summary"]["matched_count"] == 10
    assert plan["probes"][0]["probe_id"] == "ll_decoder_q_window_signed_response_surface"
    assert plan["probes"][0]["response_surface_summary"]["suppress_harm_sum"] == 0.05
    assert plan["probes"][0]["top_preserve_windows"] == [{"match_key_value": "1"}]
    assert "do_not_dispatch_decoder_q_response_surface_without_advisory_sign_calibration" in {
        item["rule"] for item in plan["prohibitions"]
    }


def test_decoder_q_surface_advisory_gate_blocks_all_regressing_surface_candidates() -> None:
    gate = build_decoder_q_surface_advisory_gate(
        _decoder_q_surface_advisory_batch(deltas=(0.00043, 0.00048, 0.00053))
    )

    assert gate["score_claim"] is False
    assert gate["status"] == "blocked"
    assert gate["decoder_q_surface_exact_eval_allowed"] is False
    assert gate["summary"]["surface_guided_candidate_count"] == 3
    assert gate["summary"]["improving_surface_guided_candidate_count"] == 0
    assert (
        "decoder_q_surface_advisory_surface_guided_all_non_improving"
        in gate["blockers"]
    )
    labels = gate["signed_calibration_labels"]
    assert labels["schema"] == "decoder_q_surface_sign_calibration_labels.v1"
    assert labels["score_claim"] is False
    assert labels["summary"]["label_count"] == 3
    assert labels["summary"]["sign_mismatch_count"] == 3
    assert labels["summary"]["all_labels_regressed"] is True
    assert labels["labels"][0]["recommended_atom_action"] == "suppress_same_sign_try_inverse"
    assert labels["labels"][0]["atom_mutation_keys"] == [
        {"tensor_name": "decoder.weight", "q_offset": 0, "delta": 1}
    ]


def test_next_probe_plan_repairs_decoder_q_surface_after_advisory_regression() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "mlx_decoder_q"),
        rows_per_fold=5,
        include_prediction=True,
    )
    surface = _decoder_q_surface_payload()

    plan = build_next_probe_plan(
        dataset,
        decoder_q_response_surface=surface,
        decoder_q_surface_advisory_batch=_decoder_q_surface_advisory_batch(
            deltas=(0.00043, 0.00048, 0.00053),
        ),
    )

    assert plan["decoder_q_surface_advisory_gate"]["status"] == "blocked"
    assert plan["probes"][0]["probe_id"] == "ll_decoder_q_surface_sign_calibration_repair"
    assert "do_not_dispatch_decoder_q_response_surface_after_advisory_regression" in {
        item["rule"] for item in plan["prohibitions"]
    }


def test_next_probe_plan_allows_decoder_q_surface_after_advisory_improvement() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "mlx_decoder_q"),
        rows_per_fold=5,
        include_prediction=True,
    )
    surface = _decoder_q_surface_payload()

    plan = build_next_probe_plan(
        dataset,
        decoder_q_response_surface=surface,
        decoder_q_surface_advisory_batch=_decoder_q_surface_advisory_batch(
            deltas=(-0.00010, 0.00002, 0.00003),
        ),
    )

    assert plan["decoder_q_surface_advisory_gate"]["status"] == "strict_pass"
    assert plan["probes"][0]["probe_id"] == "ll_decoder_q_window_signed_response_surface"
    assert "do_not_dispatch_decoder_q_response_surface_after_advisory_regression" not in {
        item["rule"] for item in plan["prohibitions"]
    }


def test_decoder_q_surface_advisory_gate_blocks_non_fixed_surface_candidates() -> None:
    batch = _decoder_q_surface_advisory_batch(deltas=(-0.00010,))
    manifest = batch["candidates"][0]["mutation_manifest"]
    manifest["fixed_length_runtime_compatible"] = False
    manifest["length_delta"] = 8

    gate = build_decoder_q_surface_advisory_gate(batch)

    assert gate["status"] == "blocked"
    assert gate["summary"]["fixed_length_surface_guided_candidate_count"] == 0
    assert (
        "decoder_q_surface_advisory_has_no_fixed_length_surface_guided_candidates"
        in gate["blockers"]
    )


def test_decoder_q_surface_advisory_gate_blocks_stale_delta_sign() -> None:
    batch = _decoder_q_surface_advisory_batch(deltas=(-0.00010,))
    batch["candidates"][0]["advisory_eval"]["canonical_score"] = (
        batch["inputs"]["baseline_score"] + 0.00020
    )

    gate = build_decoder_q_surface_advisory_gate(batch)

    assert gate["status"] == "blocked"
    assert math.isclose(
        gate["summary"]["best_surface_guided_delta_vs_baseline_score"],
        0.00020,
    )
    assert "decoder_q_surface_advisory_delta_mismatch" in gate["blockers"]


def test_decoder_q_surface_advisory_gate_rejects_wrong_producer_and_nested_authority() -> None:
    wrong_producer = _decoder_q_surface_advisory_batch(deltas=(-0.00010,))
    wrong_producer["producer"] = "unit-test"
    try:
        build_decoder_q_surface_advisory_gate(wrong_producer)
    except ScorerResponseDatasetError as exc:
        assert "producer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong producer was accepted")

    nested_claim = _decoder_q_surface_advisory_batch(deltas=(-0.00010,))
    nested_claim["candidates"][0]["advisory_eval"]["score_claim"] = True
    try:
        build_decoder_q_surface_advisory_gate(nested_claim)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nested advisory score claim was accepted")

    nested_objective_claim = _decoder_q_surface_advisory_batch(deltas=(-0.00010,))
    nested_objective_claim["candidates"][0]["mutation_manifest"]["response_surface_objective"][
        "ready_for_exact_eval_dispatch"
    ] = True
    try:
        build_decoder_q_surface_advisory_gate(nested_objective_claim)
    except ScorerResponseDatasetError as exc:
        assert "ready_for_exact_eval_dispatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nested objective dispatch authority was accepted")


def _decoder_q_surface_payload() -> dict:
    return {
        "schema": "decoder_q_response_surface_plan.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "summary": {
            "matched_count": 10,
            "preserve_candidate_effect_count": 3,
            "suppress_or_invert_candidate_effect_count": 7,
            "neutral_or_uncertain_count": 0,
            "preserve_gain_sum": 0.01,
            "suppress_harm_sum": 0.05,
            "axis_dominance_counts": {"seg": 10},
        },
        "top_preserve_windows": [{"match_key_value": "1"}],
        "top_suppress_windows": [{"match_key_value": "2"}],
    }


def _decoder_q_surface_advisory_batch(*, deltas: tuple[float, ...]) -> dict:
    baseline = 0.19206142414659494
    candidates = []
    for index, delta in enumerate(deltas):
        candidates.append(
            {
                "candidate_id": f"surface-{index}",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "delta_vs_baseline_score": delta,
                "mutation_manifest": {
                    "bucket": "response_surface_guided",
                    "edit_budget": index + 1,
                    "atoms": [
                        {
                            "mutation": {
                                "tensor_name": "decoder.weight",
                                "q_offset": index,
                                "delta": 1,
                            }
                        }
                    ],
                    "fixed_length_runtime_compatible": True,
                    "length_delta": 0,
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "rank_or_kill_eligible": False,
                    "promotable": False,
                    "response_surface_objective": {
                        "proxy_priority_sum": float(index + 1),
                        "strategy": "suppress_or_invert_regressions_first",
                        "preferred_direction": "suppress",
                        "dominant_axis": "seg",
                        "score_claim": False,
                        "score_claim_valid": False,
                        "promotion_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                        "rank_or_kill_eligible": False,
                        "promotable": False,
                    },
                },
                "raw_comparison": {
                    "changed_frame_count": 600,
                    "byte_delta_summary": {"changed_byte_count": 1000 + index},
                },
                "advisory_eval": {
                    "returncode": 0,
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "rank_or_kill_eligible": False,
                    "promotable": False,
                    "canonical_score": baseline + delta,
                    "avg_segnet_dist": 0.00056,
                    "avg_posenet_dist": 0.000029,
                    "archive_size_bytes": 178517,
                },
            }
        )
    best = min(candidates, key=lambda row: row["delta_vs_baseline_score"])
    return {
        "schema": "fec6_decoder_q_candidate_advisory_batch_v1",
        "producer": "tools/run_decoder_q_candidate_advisory_batch.py",
        "authority": {
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
        },
        "inputs": {"baseline_score": baseline},
        "summary": {
            "candidate_count": len(candidates),
            "advisory_success_count": len(candidates),
            "visible_change_count": len(candidates),
            "best_candidate_id": best["candidate_id"],
            "best_score": baseline + best["delta_vs_baseline_score"],
            "best_delta_vs_baseline_score": best["delta_vs_baseline_score"],
        },
        "candidates": candidates,
    }


def test_scorer_response_validation_gate_blocks_mixed_axis_targets() -> None:
    dataset = _validation_dataset(
        families=("mlx_scorer_response", "decoder_q"),
        rows_per_fold=5,
        include_prediction=True,
    )
    for row in dataset["rows"]:
        if row["family"] == "decoder_q":
            row["axis"] = "[macOS-CPU advisory decoder-q]"

    gate = build_scorer_response_validation_gate(dataset)

    assert gate["status"] == "blocked"
    assert any(blocker.startswith("mixed_axis_targets") for blocker in gate["blockers"])
    assert gate["passing_prediction_fields"] == ["predicted_delta_vs_baseline_score"]


def test_validate_scorer_response_dataset_cli_writes_gate(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(_validation_dataset(families=("mlx_scorer_response",), rows_per_fold=10)),
        encoding="utf-8",
    )
    json_out = tmp_path / "gate.json"
    md_out = tmp_path / "gate.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "validate_scorer_response_dataset.py"),
            "--dataset",
            str(dataset_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"score_claim": false' in completed.stdout
    gate = json.loads(json_out.read_text(encoding="utf-8"))
    assert gate["status"] == "blocked"
    assert "no_prediction_fields_present" in gate["blockers"]
    assert "Scorer Response Validation Gate" in md_out.read_text(encoding="utf-8")


def test_validate_scorer_response_dataset_cli_require_pass_blocks_after_writes(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(_validation_dataset(families=("mlx_scorer_response",), rows_per_fold=10)),
        encoding="utf-8",
    )
    json_out = tmp_path / "gate.json"
    md_out = tmp_path / "gate.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "validate_scorer_response_dataset.py"),
            "--dataset",
            str(dataset_path),
            "--require-pass",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["passed"] is False
    gate = json.loads(json_out.read_text(encoding="utf-8"))
    assert gate["status"] == "blocked"
    assert "no_prediction_fields_present" in gate["blockers"]
    assert "Scorer Response Validation Gate" in md_out.read_text(encoding="utf-8")


def test_validate_scorer_response_dataset_cli_require_pass_allows_passed_gate(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            _validation_dataset(
                families=("mlx_scorer_response", "decoder_q"),
                rows_per_fold=5,
                include_prediction=True,
            )
        ),
        encoding="utf-8",
    )
    json_out = tmp_path / "gate.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "validate_scorer_response_dataset.py"),
            "--dataset",
            str(dataset_path),
            "--require-pass",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["passed"] is True
    gate = json.loads(json_out.read_text(encoding="utf-8"))
    assert gate["status"] == "passed"
    assert gate["blockers"] == []


def test_merge_scorer_response_datasets_preserves_false_authority() -> None:
    mlx = _validation_dataset(families=("mlx_scorer_response",), rows_per_fold=1)
    decoder_q = _validation_dataset(families=("decoder_q",), rows_per_fold=1)

    merged = merge_scorer_response_datasets([("mlx.json", mlx), ("decoder_q.json", decoder_q)])

    assert merged["score_claim"] is False
    assert merged["summary"]["row_count"] == 10
    assert merged["summary"]["family_counts"] == {
        "decoder_q": 5,
        "mlx_scorer_response": 5,
    }
    assert {row["source_dataset"] for row in merged["rows"]} == {"mlx.json", "decoder_q.json"}


def test_merge_scorer_response_datasets_rejects_duplicate_rows() -> None:
    dataset = _validation_dataset(families=("decoder_q",), rows_per_fold=1)

    try:
        merge_scorer_response_datasets([("a.json", dataset), ("b.json", dataset)])
    except ScorerResponseDatasetError as exc:
        assert "duplicate row_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate rows were accepted")


def test_merge_scorer_response_datasets_cli(tmp_path) -> None:
    mlx_path = tmp_path / "mlx.json"
    decoder_path = tmp_path / "decoder_q.json"
    mlx_path.write_text(
        json.dumps(_validation_dataset(families=("mlx_scorer_response",), rows_per_fold=1)),
        encoding="utf-8",
    )
    decoder_path.write_text(
        json.dumps(_validation_dataset(families=("decoder_q",), rows_per_fold=1)),
        encoding="utf-8",
    )
    json_out = tmp_path / "merged.json"
    md_out = tmp_path / "merged.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "merge_scorer_response_datasets.py"),
            "--input",
            str(mlx_path),
            "--input",
            str(decoder_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"score_claim": false' in completed.stdout
    merged = json.loads(json_out.read_text(encoding="utf-8"))
    assert merged["summary"]["family_counts"]["decoder_q"] == 5
    assert "Scorer Response Dataset" in md_out.read_text(encoding="utf-8")


def test_build_response_dataset_normalizes_candidate_list_and_correlations(tmp_path) -> None:
    path = tmp_path / "op3_summary.json"
    payload = {
        "schema": "op3v3_decoder_q_advisory_batch.v1",
        "producer": "tools/run_decoder_q_candidate_advisory_batch.py",
        "candidates": [
            {
                "candidate_id": f"c{i}",
                "advisory_eval": _advisory(1.0 + i * 0.1, 100 + i, 0.001 + i * 0.001, 0.01),
            }
            for i in range(4)
        ],
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset(
        [path],
        baseline=ResponseBaseline(score=1.0, archive_bytes=100),
    )

    assert dataset["summary"]["row_count"] == 4
    assert dataset["summary"]["family_counts"] == {"decoder_q": 4}
    assert dataset["feature_correlations"]
    md = render_markdown(dataset)
    assert "Scorer Response Dataset" in md
    assert "## Authority" in md
    assert "Score claim valid: `False`" in md
    assert "`decoder_q`: 4" in md


def test_response_dataset_computes_break_even_bytes_for_scorer_gain(tmp_path) -> None:
    path = tmp_path / "gain.json"
    score = 1.0 - RATE_SCORE_PER_BYTE
    payload = {
        "schema": "sparse_residual_oracle_smoke.v1",
        "candidate": {
            "advisory_eval": _advisory(score, 102, 0.001, 0.01),
            "plan": {"packed_bytes": 2},
        },
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))
    row = dataset["rows"][0]

    assert row["added_archive_bytes"] == 2
    assert math.isclose(row["rate_delta_vs_baseline"], 2.0 * RATE_SCORE_PER_BYTE)
    assert math.isclose(row["scorer_delta_vs_baseline"], -3.0 * RATE_SCORE_PER_BYTE)
    assert math.isclose(row["observed_scorer_gain_vs_baseline"], 3.0 * RATE_SCORE_PER_BYTE)
    assert math.isclose(row["required_scorer_gain_for_added_bytes"], 2.0 * RATE_SCORE_PER_BYTE)
    assert row["scorer_gain_shortfall_to_break_even"] == 0.0
    assert math.isclose(row["break_even_added_bytes_from_scorer_gain"], 3.0)
    assert math.isclose(row["byte_budget_margin_vs_break_even"], 1.0)
    assert dataset["summary"]["best_byte_budget_margin"]["byte_budget_margin_vs_break_even"] > 0


def test_next_probe_plan_blocks_overbudget_coordinate_residual(tmp_path) -> None:
    path = tmp_path / "overbudget.json"
    score = 1.0 + RATE_SCORE_PER_BYTE
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(score, 103, 0.001, 0.01)},
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    plan = build_next_probe_plan(dataset)

    assert plan["score_claim"] is False
    assert plan["prohibitions"][0]["rule"] == "do_not_widen_coordinate_sparse_residual_sidecar"
    assert plan["probes"][0]["probe_id"] == "ll_byte_neutral_decoder_q_response_model"
    rendered = render_next_probe_plan_markdown(plan)
    assert "Next-Probe" in rendered
    assert "## Authority" in rendered
    assert "Promotable: `False`" in rendered


def _null_byte_matrix() -> dict:
    return {
        "schema": "null_byte_master_gradient_probe_matrix_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "n_anchors_probed_ok": 2,
        "top5_replacement_candidates": [
            {
                "substrate_label": "smaller_null_budget",
                "codec_family": "hnerv_family",
                "scored_archive_sha256": "b" * 64,
                "axis": "[contest-CUDA]",
                "anchor_index": 2,
                "n_null_bytes": 128,
                "null_fraction": 0.1,
                "predicted_delta_s_per_seed_budget": {"K=16": -0.0001},
            },
            {
                "substrate_label": "larger_null_budget",
                "codec_family": "hnerv_family",
                "scored_archive_sha256": "a" * 64,
                "axis": "[contest-CUDA]",
                "anchor_index": 1,
                "n_null_bytes": 256,
                "null_fraction": 0.2,
                "predicted_delta_s_per_seed_budget": {"K=16": -0.0002},
            },
        ],
    }


def _pair4_seed_boundary_smoke() -> dict:
    return {
        "smoke_label": "wave_3_magic_codec_pair_4_procedural_seed_orthogonality_smoke",
        "smoke_pair_id": "pair_4_magic_codec_x_procedural_codebook_seed_bytes",
        "cascade_verdict": "PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "n_canonical_reversible_ordering_rows": 30,
        "n_canonical_reversible_ordering_rows_raw_seed_dominates": 30,
        "min_canonical_reversible_best_nonraw_delta_vs_raw_bytes": 4,
        "ordering_dimension": {
            "reversible_free_orderings": ["identity", "reverse"],
            "non_free_control_orderings": ["sorted_ascending"],
        },
        "codec_dimensions": {"raw_seed": True, "brotli_q11_seed_bytes": True},
    }


def test_magic_codec_seed_boundary_normalizes_pair4_smoke() -> None:
    boundary = build_magic_codec_seed_boundary(_pair4_seed_boundary_smoke())

    assert boundary["schema"] == "ll_magic_codec_seed_boundary.v1"
    assert boundary["score_claim"] is False
    assert boundary["score_claim_valid"] is False
    assert boundary["rank_or_kill_eligible"] is False
    assert boundary["promotable"] is False
    assert boundary["boundary_validated_raw_seed_dominates"] is True
    assert boundary["n_canonical_reversible_ordering_rows"] == 30
    assert boundary["min_canonical_reversible_best_nonraw_delta_vs_raw_bytes"] == 4


def test_magic_codec_seed_boundary_rejects_promotional_smoke() -> None:
    bad = _pair4_seed_boundary_smoke()
    bad["promotion_eligible"] = True
    try:
        build_magic_codec_seed_boundary(bad)
    except ScorerResponseDatasetError as exc:
        assert "promotion_eligible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected promotional boundary rejection")

    score_claim = _pair4_seed_boundary_smoke()
    score_claim["score_claim"] = True
    try:
        build_magic_codec_seed_boundary(score_claim)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected score-claim boundary rejection")


def test_null_byte_priority_weights_sort_by_predicted_delta() -> None:
    weights = build_null_byte_priority_weights(_null_byte_matrix())

    assert weights["score_claim"] is False
    assert weights["rank_or_kill_eligible"] is False
    assert weights["promotable"] is False
    assert weights["summary"]["candidate_count"] == 2
    assert weights["priority_rows"][0]["substrate_label"] == "larger_null_budget"
    assert weights["priority_rows"][0]["priority_weight"] == 0.0002
    assert weights["priority_rows"][0]["priority_weight_units"] == "absolute_predicted_score_delta"
    assert 0.0 < weights["priority_rows"][0]["ll_sampling_weight"] < 1.0


def test_null_byte_priority_weights_reject_legacy_missing_false_authority_keys_by_default() -> None:
    matrix = _null_byte_matrix()
    matrix.pop("promotion_eligible")
    matrix.pop("rank_or_kill_eligible")

    try:
        build_null_byte_priority_weights(matrix)
    except ScorerResponseDatasetError as exc:
        assert "must be explicit false" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing authority rejection")


def test_null_byte_priority_weights_accept_legacy_missing_false_authority_keys_with_flag() -> None:
    matrix = _null_byte_matrix()
    matrix.pop("promotion_eligible")
    matrix.pop("rank_or_kill_eligible")
    matrix.pop("ready_for_exact_eval_dispatch")

    weights = build_null_byte_priority_weights(
        matrix,
        allow_legacy_missing_authority=True,
    )

    assert weights["score_claim"] is False
    assert weights["promotion_eligible"] is False
    assert weights["ready_for_exact_eval_dispatch"] is False
    assert set(weights["legacy_missing_authority_fields_accepted"]) == {
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    }


def test_next_probe_plan_consumes_null_byte_matrix_as_first_probe(tmp_path) -> None:
    path = tmp_path / "overbudget.json"
    score = 1.0 + RATE_SCORE_PER_BYTE
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(score, 103, 0.001, 0.01)},
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    plan = build_next_probe_plan(dataset, null_byte_matrix=_null_byte_matrix())

    assert plan["null_byte_priority_weights"]["schema"] == "ll_null_byte_priority_weights.v1"
    assert plan["probes"][0]["probe_id"] == "ll_null_byte_procedural_codebook_candidates"
    assert plan["probes"][0]["priority"] == 1
    assert plan["probes"][1]["probe_id"] == "ll_byte_neutral_decoder_q_response_model"
    assert "CandidateModificationSpec" in plan["probes"][0]["acceptance_gate"]
    assert "Null-Byte Matrix Priority" in render_next_probe_plan_markdown(plan)


def test_next_probe_plan_consumes_pair4_seed_boundary_as_prohibition(tmp_path) -> None:
    path = tmp_path / "overbudget.json"
    score = 1.0 + RATE_SCORE_PER_BYTE
    payload = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "candidate": {"advisory_eval": _advisory(score, 103, 0.001, 0.01)},
        "authority": {"score_claim": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    plan = build_next_probe_plan(
        dataset,
        null_byte_matrix=_null_byte_matrix(),
        magic_codec_seed_boundary_smoke=_pair4_seed_boundary_smoke(),
    )

    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in rules
    assert plan["magic_codec_seed_boundary"]["boundary_validated_raw_seed_dominates"] is True
    assert "keep seeds raw" in plan["probes"][0]["rationale"]
    rendered = render_next_probe_plan_markdown(plan)
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in rendered


def test_null_byte_matrix_fail_closed_on_promotional_or_missing_k() -> None:
    bad = _null_byte_matrix()
    bad["score_claim"] = True
    try:
        build_null_byte_priority_weights(bad)
    except ScorerResponseDatasetError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected promotional matrix rejection")

    missing_authority = _null_byte_matrix()
    missing_authority.pop("ready_for_exact_eval_dispatch")
    try:
        build_next_probe_plan(
            {"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []},
            null_byte_matrix=missing_authority,
        )
    except ScorerResponseDatasetError as exc:
        assert "ready_for_exact_eval_dispatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing ready_for_exact_eval_dispatch rejection")

    missing_k = _null_byte_matrix()
    missing_k["top5_replacement_candidates"][0]["predicted_delta_s_per_seed_budget"] = {"K=32": -0.1}
    try:
        build_null_byte_priority_weights(missing_k)
    except ScorerResponseDatasetError as exc:
        assert "K=16" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing K rejection")


def test_build_response_dataset_skips_non_advisory_rows(tmp_path) -> None:
    path = tmp_path / "skip.json"
    path.write_text(
        json.dumps({"candidate": {"advisory_eval": {"skipped": True, "reason": "local_veto"}}}),
        encoding="utf-8",
    )

    dataset = build_response_dataset([path], baseline=ResponseBaseline(score=1.0, archive_bytes=100))

    assert dataset["summary"]["row_count"] == 0
    assert dataset["skipped"][0]["reason"].endswith("no usable advisory row")
