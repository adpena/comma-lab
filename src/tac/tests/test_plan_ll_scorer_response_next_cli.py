from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mlx_dataset() -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return {
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
                **_normalized_objective_fields(
                    observed_scorer_gain=0.002,
                    source_n_samples=600,
                    added_archive_bytes=0,
                ),
                **false_authority,
            }
        ],
    }


def _mlx_dataset_two_windows() -> dict:
    payload = _mlx_dataset()
    payload["summary"]["row_count"] = 2
    second = dict(payload["rows"][0])
    second["row_id"] = "mlx-row-2"
    second["source_pair_window"] = [600, 1200]
    payload["rows"].append(second)
    return payload


def _validated_mlx_dataset() -> dict:
    false_authority = {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    rows = []
    for family in ("mlx_scorer_response", "decoder_q"):
        for fold in range(5):
            for offset in range(5):
                pair_rank = fold * 5 + offset
                value = (
                    -0.002 - 0.00001 * pair_rank
                    if family == "mlx_scorer_response"
                    else float(100 + pair_rank)
                )
                row = {
                    "schema": "scorer_response_row.v1",
                    "row_id": f"{family}-{fold}-{offset}",
                    "family": family,
                    "axis": "[macOS-MLX research-signal]",
                    "holdout_fold": fold,
                    "delta_vs_baseline_score": value,
                    "scorer_delta_vs_baseline": value,
                    "predicted_delta_vs_baseline_score": value,
                    "byte_budget_margin_vs_break_even": None,
                    "authority_source_score_claim": False,
                    **false_authority,
                }
                if family == "mlx_scorer_response":
                    row.update(
                        {
                            "archive_sha256": "a" * 64,
                            "source_inflated_outputs_aggregate_sha256": "e" * 64,
                            "source_batch_pairs": 1,
                            "source_n_samples": 600,
                            "source_pair_window": [0, 600],
                            **_normalized_objective_fields(
                                observed_scorer_gain=0.002 + 0.00001 * pair_rank,
                                source_n_samples=600,
                                added_archive_bytes=0,
                            ),
                        }
                    )
                rows.append(row)
    return {
        "schema": "scorer_response_dataset.v1",
        "summary": {"row_count": len(rows)},
        "authority": {
            **false_authority,
            "evidence_grade": "macOS-MLX-research-signal",
            "evidence_tag": "[macOS-MLX research-signal]",
            "score_axis": "[macOS-MLX research-signal]",
        },
        **false_authority,
        "rows": rows,
    }


def _normalized_objective_fields(
    *,
    observed_scorer_gain: float,
    source_n_samples: int,
    added_archive_bytes: int,
) -> dict:
    normalized_gain = (
        float(observed_scorer_gain)
        * float(source_n_samples)
        / float(CONTEST_EXACT_SAMPLE_COUNT)
    )
    rate_delta = float(added_archive_bytes) * RATE_SCORE_PER_BYTE
    return {
        "full_video_denominator": CONTEST_EXACT_SAMPLE_COUNT,
        "observed_scorer_gain_vs_baseline": float(observed_scorer_gain),
        "added_archive_bytes": int(added_archive_bytes),
        "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
        "projected_full_video_delta_vs_baseline_score": rate_delta - normalized_gain,
        "break_even_added_bytes_from_normalized_full_video_gain": (
            normalized_gain / RATE_SCORE_PER_BYTE
        ),
        "normalized_full_video_byte_budget_margin_vs_break_even": (
            normalized_gain / RATE_SCORE_PER_BYTE - float(added_archive_bytes)
        ),
    }


def _failed_mlx_parity_sweep() -> dict:
    return {
        "schema_version": "mlx_scorer_torch_parity_sweep.v1",
        "run_id": "unit",
        "verdict": "FAIL_MLX_TORCH_SCORER_PARITY_SWEEP",
        "passed": False,
        "blockers": ["window_failed:index=39:pair_window=[156, 160]"],
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "score_axis": "[macOS-MLX research-signal]",
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
            "passed_windows": 74,
            "failed_windows": 1,
            "posenet_output_abs_max": {"max": 7.62939453125e-06},
            "posenet_component_abs_max": {"max": 9.713470479344455e-12},
            "segnet_logit_abs_max": {"max": 0.0007913112640380859},
            "segnet_argmax_diff_pixels": {"max": 1.0},
            "segnet_argmax_diff_fraction": {"max": 1.2715657552083333e-06},
            "segnet_argmax_mismatch_pixels_total": 1,
        },
        "rows": [
            {
                "index": 39,
                "passed": False,
                "verdict": "FAIL_MLX_TORCH_SCORER_PARITY",
                "blockers": ["segnet_argmax_diff_pixels_exceeds_threshold:1>0"],
                "pair_window": [156, 160],
                "deltas": {
                    "posenet_output_abs_max": 7.62939453125e-06,
                    "posenet_component_abs_max": 9.713470479344455e-12,
                    "segnet_logit_abs_max": 0.0007913112640380859,
                    "segnet_argmax_diff_pixels": 1,
                    "segnet_argmax_diff_fraction": 1.2715657552083333e-06,
                },
            }
        ],
    }


def _passing_mlx_parity_sweep() -> dict:
    payload = _failed_mlx_parity_sweep()
    payload["verdict"] = "PASS_MLX_TORCH_SCORER_PARITY_SWEEP"
    payload["passed"] = True
    payload["blockers"] = []
    payload["summary"]["passed_windows"] = 75
    payload["summary"]["failed_windows"] = 0
    payload["summary"]["segnet_argmax_diff_pixels"]["max"] = 0.0
    payload["summary"]["segnet_argmax_diff_fraction"]["max"] = 0.0
    payload["summary"]["segnet_argmax_mismatch_pixels_total"] = 0
    payload["rows"][0]["passed"] = True
    payload["rows"][0]["verdict"] = "PASS_MLX_TORCH_SCORER_PARITY"
    payload["rows"][0]["blockers"] = []
    payload["rows"][0]["deltas"]["segnet_argmax_diff_pixels"] = 0
    payload["rows"][0]["deltas"]["segnet_argmax_diff_fraction"] = 0.0
    return payload


def _passing_mlx_score_calibration() -> dict:
    return {
        "schema_version": "mlx_score_calibration.v1",
        "run_id": "unit",
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
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
            "mlx_spend_triage_pairwise_certified_count": 1,
            "mlx_spend_triage_pairwise_uncertain_count": 0,
            "mlx_spend_triage_pairwise_total_count": 1,
            "recommended_min_mlx_gap_for_spend_triage": 5.0e-5,
            "calibration_uncertainty_score": 1.0e-5,
        },
    }


def _passing_mlx_production_contract() -> dict:
    return {
        "schema_version": "mlx_scorer_production_contract.v2",
        "gate_set_version": "mlx_scorer_production_gate_set.v2.cache_auth_torch_profile",
        "run_id": "unit",
        "passed": True,
        "advisory_passed": True,
        "verdict": "PASS_MLX_SCORER_PRODUCTION_CONTRACT",
        "blockers": [],
        "warnings": ["batch_invariance_not_required_for_singleton_response"],
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
        "score_axis": "[macOS-MLX research-signal]",
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
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
            "cache_identity": True,
            "cache_auth_audit": True,
            "torch_parity": True,
            "reference_torch_parity": True,
            "profile_stability": True,
            "batch_invariance": False,
            "batch_invariance_policy_requested": True,
            "score_calibration": True,
            "strict_gate_policy": True,
        },
        "authority_status": "non-authoritative local MLX production signal",
    }


def _passing_mlx_production_contract_for_window(
    *,
    run_id: str,
    pair_window: list[int],
) -> dict:
    payload = json.loads(json.dumps(_passing_mlx_production_contract()))
    payload["run_id"] = run_id
    payload["response_summary"]["pair_window"] = pair_window
    return payload


def test_plan_ll_scorer_response_next_cli_accepts_null_byte_matrix(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    matrix_path = tmp_path / "null_byte_matrix.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "null_byte_master_gradient_probe_matrix_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "axis_tag": "[predicted]",
                "n_anchors_probed_ok": 1,
                "top5_replacement_candidates": [
                    {
                        "substrate_label": "fec6",
                        "codec_family": "hnerv_family",
                        "scored_archive_sha256": "a" * 64,
                        "axis": "[contest-CUDA]",
                        "anchor_index": 1,
                        "n_null_bytes": 16292,
                        "null_fraction": 0.091,
                        "predicted_delta_s_per_seed_budget": {"K=16": -0.0108375},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
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

    assert "ll_null_byte_procedural_codebook_candidates" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["probes"][0]["probe_id"] == "ll_null_byte_procedural_codebook_candidates"
    assert plan["probes"][0]["null_byte_priority_rows"][0]["priority_weight"] == 0.0108375
    assert "Null-Byte Matrix Priority" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_accepts_legacy_null_byte_matrix_only_with_flag(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    matrix_path = tmp_path / "null_byte_matrix.json"
    json_out = tmp_path / "plan.json"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "null_byte_master_gradient_probe_matrix_v1",
                "score_claim": False,
                "promotable": False,
                "axis_tag": "[predicted]",
                "n_anchors_probed_ok": 1,
                "top5_replacement_candidates": [
                    {
                        "substrate_label": "legacy_fec6",
                        "codec_family": "hnerv_family",
                        "scored_archive_sha256": "a" * 64,
                        "axis": "[contest-CUDA]",
                        "anchor_index": 1,
                        "n_null_bytes": 16292,
                        "null_fraction": 0.091,
                        "predicted_delta_s_per_seed_budget": {"K=16": -0.0108375},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rejected = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode == 2
    assert "promotion_eligible must be explicit false" in rejected.stderr

    accepted = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--null-byte-matrix",
            str(matrix_path),
            "--allow-legacy-null-byte-matrix-missing-authority",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "ll_null_byte_procedural_codebook_candidates" in accepted.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert set(
        plan["null_byte_priority_weights"]["legacy_missing_authority_fields_accepted"]
    ) == {
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
    }


def test_plan_ll_scorer_response_next_cli_accepts_pair4_seed_boundary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    boundary_path = tmp_path / "pair4_boundary.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )
    boundary_path.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--magic-codec-seed-boundary-smoke",
            str(boundary_path),
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

    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["magic_codec_seed_boundary"]["boundary_validated_raw_seed_dominates"] is True
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in rules
    assert "do_not_wrap_procedural_seed_bytes_with_magic_codec" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_blocks_failed_mlx_parity_without_override(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_failed_mlx_parity_sweep()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
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

    assert "do_not_use_mlx_rows_after_failed_strict_parity_sweep" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "blocked"
    assert plan["probes"][0]["probe_id"] == "ll_mlx_torch_parity_repair_or_override"
    assert "ll_mlx_cpu_stable_response_harvest" not in {
        probe["probe_id"] for probe in plan["probes"]
    }
    assert "MLX Torch Parity Gate" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_accepts_mlx_production_contract(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    calibration_path = tmp_path / "calibration.json"
    production_path = tmp_path / "production_contract.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_passing_mlx_parity_sweep()), encoding="utf-8")
    calibration_path.write_text(
        json.dumps(_passing_mlx_score_calibration()),
        encoding="utf-8",
    )
    production_path.write_text(
        json.dumps(_passing_mlx_production_contract()),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--mlx-score-calibration",
            str(calibration_path),
            "--mlx-production-contract",
            str(production_path),
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

    assert "ll_mlx_cpu_stable_response_harvest" in completed.stdout
    assert "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract" not in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["mlx_production_contract_gate"]["status"] == "strict_pass"
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        not in rules
    )
    assert "MLX Production Contract Gate" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_accepts_mlx_contract_bundle(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    calibration_path = tmp_path / "calibration.json"
    production_a_path = tmp_path / "production_contract_a.json"
    production_b_path = tmp_path / "production_contract_b.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset_two_windows()), encoding="utf-8")
    parity_path.write_text(json.dumps(_passing_mlx_parity_sweep()), encoding="utf-8")
    calibration_path.write_text(
        json.dumps(_passing_mlx_score_calibration()),
        encoding="utf-8",
    )
    production_a_path.write_text(
        json.dumps(
            _passing_mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
            )
        ),
        encoding="utf-8",
    )
    production_b_path.write_text(
        json.dumps(
            _passing_mlx_production_contract_for_window(
                run_id="window-600-1200",
                pair_window=[600, 1200],
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--mlx-score-calibration",
            str(calibration_path),
            "--mlx-production-contract",
            str(production_a_path),
            "--mlx-production-contract",
            str(production_b_path),
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

    assert "ll_mlx_cpu_stable_response_harvest" in completed.stdout
    assert "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract" not in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    gate = plan["mlx_production_contract_gate"]
    assert gate["source_schema"] == "mlx_scorer_production_contract_bundle.v1"
    assert gate["status"] == "strict_pass"
    assert gate["summary"]["contract_count"] == 2
    assert gate["summary"]["matched_row_count"] == 2
    assert gate["summary"]["unmatched_row_ids"] == []
    rules = {item["rule"] for item in plan["prohibitions"]}
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        not in rules
    )
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_failed_production_contract"
        not in rules
    )
    assert "MLX Production Contract Gate" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_requires_effective_mlx_gate_blocks(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    calibration_path = tmp_path / "calibration.json"
    production_path = tmp_path / "production_contract.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_passing_mlx_parity_sweep()), encoding="utf-8")
    calibration_path.write_text(
        json.dumps(_passing_mlx_score_calibration()),
        encoding="utf-8",
    )
    production_path.write_text(
        json.dumps(_passing_mlx_production_contract()),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--mlx-score-calibration",
            str(calibration_path),
            "--mlx-production-contract",
            str(production_path),
            "--required-spend-triage-family",
            "mlx_scorer_response",
            "--require-effective-mlx-spend-triage",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["effective_mlx_spend_triage_allowed"] is False
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    gate = plan["effective_mlx_spend_triage_gate"]
    assert gate["status"] == "blocked"
    assert "response_validation_gate_not_passed" in gate["blockers"]
    assert "Effective MLX Spend Triage Gate" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_requires_effective_mlx_gate_passes(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    calibration_path = tmp_path / "calibration.json"
    production_path = tmp_path / "production_contract.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_validated_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_passing_mlx_parity_sweep()), encoding="utf-8")
    calibration_path.write_text(
        json.dumps(_passing_mlx_score_calibration()),
        encoding="utf-8",
    )
    production_path.write_text(
        json.dumps(_passing_mlx_production_contract()),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--mlx-score-calibration",
            str(calibration_path),
            "--mlx-production-contract",
            str(production_path),
            "--required-spend-triage-family",
            "mlx_scorer_response",
            "--require-effective-mlx-spend-triage",
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
    assert stdout_payload["effective_mlx_spend_triage_allowed"] is True
    assert (
        stdout_payload["effective_mlx_spend_triage_gate"][
            "mlx_exact_eval_spend_triage_allowed"
        ]
        is True
    )
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    gate = plan["effective_mlx_spend_triage_gate"]
    assert gate["status"] == "strict_pass"
    assert gate["mlx_exact_eval_spend_triage_allowed"] is True


def test_plan_ll_scorer_response_next_cli_blocks_uncovered_mlx_contract_bundle_row(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    calibration_path = tmp_path / "calibration.json"
    production_a_path = tmp_path / "production_contract_a.json"
    production_b_path = tmp_path / "production_contract_b.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset_two_windows()), encoding="utf-8")
    parity_path.write_text(json.dumps(_passing_mlx_parity_sweep()), encoding="utf-8")
    calibration_path.write_text(
        json.dumps(_passing_mlx_score_calibration()),
        encoding="utf-8",
    )
    production_a_path.write_text(
        json.dumps(
            _passing_mlx_production_contract_for_window(
                run_id="window-0-600",
                pair_window=[0, 600],
            )
        ),
        encoding="utf-8",
    )
    production_b_path.write_text(
        json.dumps(
            _passing_mlx_production_contract_for_window(
                run_id="window-1200-1800",
                pair_window=[1200, 1800],
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--mlx-score-calibration",
            str(calibration_path),
            "--mlx-production-contract",
            str(production_a_path),
            "--mlx-production-contract",
            str(production_b_path),
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

    assert "do_not_use_mlx_rows_for_exact_eval_spend_triage_after_failed_production_contract" in completed.stdout
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    gate = plan["mlx_production_contract_gate"]
    assert gate["source_schema"] == "mlx_scorer_production_contract_bundle.v1"
    assert gate["status"] == "blocked"
    assert "mlx-row-2" in gate["summary"]["unmatched_row_ids"]
    assert "MLX Production Contract Gate" in md_out.read_text(encoding="utf-8")


def test_plan_ll_scorer_response_next_cli_allows_failed_mlx_parity_with_override(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    parity_path = tmp_path / "parity.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    dataset_path.write_text(json.dumps(_mlx_dataset()), encoding="utf-8")
    parity_path.write_text(json.dumps(_failed_mlx_parity_sweep()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-torch-parity-sweep",
            str(parity_path),
            "--allow-mlx-parity-research-signal-override",
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

    assert "ll_mlx_cpu_stable_response_harvest" in completed.stdout
    assert (
        "do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract"
        in completed.stdout
    )
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["mlx_torch_parity_sweep_gate"]["status"] == "research_signal_override"
    assert plan["probes"][0]["probe_id"] == "ll_mlx_cpu_stable_response_harvest"
    assert plan["probes"][0]["mlx_torch_parity_gate"]["score_claim"] is False
    assert "research_signal_override" in md_out.read_text(encoding="utf-8")
