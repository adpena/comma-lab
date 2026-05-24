# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.inverse_steganalysis_acquisition import (
    ACTION_FUNCTIONAL_SCHEMA,
    CONTEST_RATE_SCORE_PER_BYTE,
    action_atoms_from_inverse_scorer_surface,
    build_discrete_scorer_action_functional,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.optimization.scorer_inverse_decision_surface import (
    build_inverse_scorer_decision_surface,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_inverse_steganalysis_action_functional.py"


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _scorer_response_dataset(path: Path) -> None:
    false_authority = _false_authority()
    path.write_text(
        json.dumps(
            {
                "schema": "scorer_response_dataset.v1",
                "producer": "test",
                **false_authority,
                "authority": {
                    **false_authority,
                    "evidence_grade": "macOS-MLX research-signal",
                },
                "summary": {"row_count": 1},
                "rows": [
                    {
                        "schema": "scorer_response_row.v1",
                        "row_id": "inverse-row-a",
                        "candidate_id": "inverse-row-a",
                        "family": "decoder_q",
                        **false_authority,
                        "authority_source_score_claim": False,
                        "delta_vs_baseline_score": -0.0001,
                        "scorer_delta_vs_baseline": 0.0,
                        "observed_scorer_gain_vs_baseline": 0.0,
                        "added_archive_bytes": -32,
                        "byte_budget_margin_vs_break_even": 32.0,
                        "source_pair_window": [7, 8],
                        "diagnostic_seg_share": 0.15,
                        "diagnostic_pose_share": 0.85,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _mlx_selection(path: Path) -> None:
    false_authority = _false_authority()
    path.write_text(
        json.dumps(
            {
                "schema": "mlx_effective_spend_triage_candidate_selection.v1",
                **false_authority,
                "candidate_generation_only": True,
                "archive_materialization_required": True,
                "requires_exact_auth_eval_before_score_claim": True,
                "allowed_use": (
                    "candidate_generation_filter_after_strict_effective_mlx_spend_triage_gate"
                ),
                "evidence_grade": "macOS-MLX-research-signal",
                "evidence_tag": "[macOS-MLX research-signal]",
                "score_axis": "[macOS-MLX research-signal]",
                "gates": {
                    "effective_mlx_spend_triage_gate": {
                        "schema": "ll_effective_mlx_spend_triage_gate.v1",
                        "status": "strict_pass",
                        "mlx_exact_eval_spend_triage_allowed": True,
                        "allowed_use": (
                            "local_exact_eval_spend_triage_filter_after_all_gates"
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
                    "gate_spend_triage_allowed_families": ["mlx_decoder_q"],
                    "min_observed_gain": 0.00001,
                    "prediction_field": "ll_predicted_delta_vs_baseline_score",
                    "require_prediction_negative": False,
                    "require_singleton_windows": True,
                    "planning_value_accessor": (
                        "scorer_response_planning_value_for_target"
                    ),
                    "planning_value_scope": "normalized_full_video",
                },
                "summary": {
                    "dataset_row_count": 1,
                    "eligible_row_count": 1,
                    "selected_count": 1,
                },
                "selected_rows": [
                    {
                        "schema": "mlx_effective_spend_triage_candidate_row.v1",
                        **false_authority,
                        "rank": 1,
                        "candidate_generation_only": True,
                        "archive_materialization_required": True,
                        "requires_exact_auth_eval_before_score_claim": True,
                        "selection_basis": (
                            "normalized_full_video_mlx_singleton_response_gain"
                        ),
                        "selection_planning_value_accessor": (
                            "scorer_response_planning_value_for_target"
                        ),
                        "selection_planning_value_scope": "normalized_full_video",
                        "row_id": "best",
                        "family": "mlx_decoder_q",
                        "candidate_id": "mlx_scorer_response:window:best",
                        "pair_indices": [10, 11],
                        "source_pair_window": [10, 11],
                        "source_path": "candidate_pair_best.json",
                        "window_baseline_source_path": "baseline_pair_best.json",
                        "archive_sha256": "a" * 64,
                        "raw_sha256": "b" * 64,
                        "source_inflated_outputs_aggregate_sha256": "c" * 64,
                        "source_candidate_cache_array_sha256": {
                            "pair_indices": "d" * 64
                        },
                        "source_reference_cache_array_sha256": {
                            "pair_indices": "e" * 64
                        },
                        "window_baseline_candidate_cache_array_sha256": {
                            "pair_indices": "f" * 64
                        },
                        "window_baseline_reference_cache_array_sha256": {
                            "pair_indices": "1" * 64
                        },
                        "observed_scorer_gain_vs_baseline": 0.012,
                        "full_video_denominator": 600,
                        "normalized_full_video_scorer_gain_vs_baseline": 0.00002,
                        "projected_full_video_delta_vs_baseline_score": -0.00002,
                        "break_even_added_bytes_from_normalized_full_video_gain": (
                            0.00002 / CONTEST_RATE_SCORE_PER_BYTE
                        ),
                        "normalized_full_video_byte_budget_margin_vs_break_even": (
                            0.00002 / CONTEST_RATE_SCORE_PER_BYTE
                        ),
                        "added_archive_bytes": 0,
                        "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
                        "prediction_field": "ll_predicted_delta_vs_baseline_score",
                        "predicted_delta_vs_baseline_score": -0.000015,
                        "prediction_agrees_with_observed_gain": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _review_packet(axis: str, *, score: float, baseline: float, aggregate: str) -> dict[str, object]:
    exact_cuda = axis == "contest_cuda"
    return {
        "schema": "tac_result_review_packet_v1",
        "tool": "tools/build_result_review_packet.py",
        "technique": "ias1_runtime_parity_top4",
        "lane_id": f"lane_{axis}",
        "job_id": f"job_{axis}",
        "source_json_path": f"experiments/results/{axis}/contest_auth_eval.json",
        "source_json_sha256": "1" * 64,
        "score_claim": False,
        "score_axis": axis,
        "score_claim_valid": exact_cuda,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "family_falsified": False,
        "method_family_retired": False,
        "measured_config_status": (
            "measured_config_retired" if exact_cuda else "contest_cpu_result_reviewed"
        ),
        "failure_class": "legitimate_score_regression_or_component_collapse",
        "baseline_score": baseline,
        "canonical_score": score,
        "exact_cuda_evidence": exact_cuda,
        "exact_cpu_evidence": not exact_cuda,
        "custody": {
            "archive_bytes": 181_232,
            "archive_sha256": "2" * 64,
            "device": "cuda" if exact_cuda else "cpu",
            "gpu_model": "Tesla T4" if exact_cuda else "",
            "n_samples": 600,
            "inflate_script": "/tmp/submission/inflate.sh",
            "command": [],
        },
        "dispatch_claim_state": {
            "terminal_status_recorded": True,
            "latest_status": f"completed_{axis}",
        },
        "runtime_custody": {
            "runtime_manifest_present": True,
            "runtime_tree_sha256": ("3" if exact_cuda else "4") * 64,
            "runtime_content_tree_sha256": "5" * 64,
            "runtime_file_count": 12,
            "runtime_files_listed": True,
            "payload_closure_fields_present": True,
            "inflate_script_sha256": "6" * 64,
            "inflated_output_manifest_sha256": ("7" if exact_cuda else "8") * 64,
            "inflated_output_aggregate_sha256": aggregate,
        },
        "score_recomputation": {
            "available": True,
            "matches_reported": True,
            "avg_segnet_dist": 0.0006,
            "avg_posenet_dist": 0.00003,
            "rate_term": 0.12067494979223735,
            "recomputed_score": score,
            "reported_score": score,
            "abs_delta_vs_reported": 0.0,
        },
        "engineering_forensic_audit": {
            "schema": "engineering_forensic_audit_v1",
            "custody_reviewed": True,
            "axis_reviewed": True,
            "runtime_config_reviewed": True,
            "archive_runtime_closure_reviewed": True,
            "score_formula_reviewed": True,
            "dispatch_claim_reviewed": True,
            "engineering_or_config_bug_found": False,
            "audit_blockers": [],
            "classification_after_audit": (
                "measured_config_retired_only"
                if exact_cuda
                else "contest_cpu_axis_reviewed_cuda_pending"
            ),
            "dead_or_family_falsification_allowed": False,
            "measured_config_retirement_allowed": exact_cuda,
        },
        "reactivation_criteria": ["byte-closed implementation change required"],
    }


def test_inverse_surface_cells_become_action_atoms_and_water_buckets(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    _scorer_response_dataset(scorer)
    surface = build_inverse_scorer_decision_surface(
        json.loads(scorer.read_text(encoding="utf-8")),
        source_label="scorer.json",
    )

    atoms = action_atoms_from_inverse_scorer_surface(surface, resource_kind="local_mlx")
    action = build_discrete_scorer_action_functional(
        atoms,
        total_byte_budget=64,
        lambda_rate=CONTEST_RATE_SCORE_PER_BYTE,
    )

    assert atoms[0]["scope_axis"] == "pairs"
    assert atoms[0]["pair_indices"] == [7, 8]
    assert atoms[0]["component"] == "posenet"
    assert atoms[0]["predicted_rate_gain"] == pytest.approx(
        CONTEST_RATE_SCORE_PER_BYTE * 32
    )
    assert action["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert action["water_bucket"]["selected_count"] == 1
    assert action["water_bucket"]["selected_cells"][0]["atom_id"] == atoms[0]["atom_id"]
    assert action["score_claim"] is False
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert action[key] is value


def test_cli_builds_inverse_action_functional_from_scorer_response(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    performance = tmp_path / "performance.json"
    runtime_identity = tmp_path / "runtime_identity.json"
    cache_identity = tmp_path / "cache_identity.json"
    output = tmp_path / "action.json"
    md_out = tmp_path / "action.md"
    _scorer_response_dataset(scorer)
    performance.write_text(
        json.dumps(
            {
                "schema": "experiment_queue_performance_summary.v1",
                "queue_id": "inverse_action_queue",
                "telemetry_only": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "event_count": 1,
                "candidate_id_by_experiment": {
                    "inverse-row-a": ["inverse-row-a"]
                },
                "by_resource_kind": {},
                "by_step": {
                    "inverse-row-a.materialize": {
                        "run_count": 1,
                        "success_count": 1,
                        "failure_count": 0,
                        "resource_kind_counts": {"local_mlx": 1},
                        "dominant_resource_kind": "local_mlx",
                        "elapsed_seconds_mean": 2.25,
                        "artifact_record_bytes_mean": 4096,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runtime_identity.write_text(
        json.dumps(
            {
                "runtime_tree_sha256": "d" * 64,
                "scorer_version": "local_scheduler.v1",
            }
        ),
        encoding="utf-8",
    )
    cache_identity.write_text(
        json.dumps({"cache_sha256": "e" * 64}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--queue-performance-summary",
            str(performance),
            "--queue-performance-runtime-identity",
            str(runtime_identity),
            "--queue-performance-cache-identity",
            str(cache_identity),
            "--output",
            str(output),
            "--md-out",
            str(md_out),
            "--repo-root",
            str(tmp_path),
            "--total-byte-budget",
            "64",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    action = json.loads(output.read_text(encoding="utf-8"))
    assert "score_claim=false" in result.stdout
    assert action["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert action["integral_totals"]["cell_count"] == 1
    assert action["water_bucket"]["selected_count"] == 1
    assert action["cells"][0]["best_observation_id"] == (
        "queue_perf_inverse_action_queue_inverse_row_a_materialize"
    )
    assert action["cells"][0]["priority"]["elapsed_seconds"] == 2.25
    assert action["cells"][0]["priority"]["artifact_bytes"] == 4096
    assert "Selected Water Buckets" in md_out.read_text(encoding="utf-8")


def test_cli_accepts_mlx_effective_spend_triage_selection(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    output = tmp_path / "action.json"
    _mlx_selection(selection)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--mlx-effective-spend-triage-selection",
            str(selection),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--total-byte-budget",
            "64",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    action = json.loads(output.read_text(encoding="utf-8"))
    cell = action["cells"][0]
    assert "score_claim=false" in result.stdout
    assert action["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert action["integral_totals"]["cell_count"] == 1
    assert action["water_bucket"]["selected_count"] == 1
    assert cell["source_provenance"]["selection_source_path"] == "selection.json"
    assert cell["source_provenance"]["source_row_id"] == "best"
    assert cell["candidate_generation_only"] is True
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert cell[key] is value
        assert action[key] is value


def test_cli_rejects_malformed_mlx_effective_spend_triage_selection(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    output = tmp_path / "action.json"
    _mlx_selection(selection)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["selected_rows"][0]["archive_sha256"] = "not-sha"
    selection.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--mlx-effective-spend-triage-selection",
            str(selection),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--total-byte-budget",
            "64",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "archive_sha256 must be sha256 hex" in result.stderr


def test_cli_accepts_paired_exact_auth_calibration_packets(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.json"
    cpu_packet = tmp_path / "cpu_review.json"
    cuda_packet = tmp_path / "cuda_review.json"
    output = tmp_path / "action.json"
    _scorer_response_dataset(scorer)
    cpu_packet.write_text(
        json.dumps(
            _review_packet(
                "contest_cpu",
                score=0.1938,
                baseline=0.1920,
                aggregate="a" * 64,
            )
        ),
        encoding="utf-8",
    )
    cuda_packet.write_text(
        json.dumps(
            _review_packet(
                "contest_cuda",
                score=0.228,
                baseline=0.205,
                aggregate="b" * 64,
            )
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--candidate-id",
            "ias1_runtime_parity_top4",
            "--exact-auth-calibration-packet",
            str(cpu_packet),
            "--exact-auth-calibration-packet",
            str(cuda_packet),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--total-byte-budget",
            "64",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    action = json.loads(output.read_text(encoding="utf-8"))
    assert action["water_bucket"]["selected_count"] == 0
    assert action["cells"][0]["best_observation_id"].startswith(
        "exact_auth_calibration_ias1_runtime_parity_top4_"
    )
    assert action["cells"][0]["priority"]["expected_score_gain"] == 0.0
    assert action["cells"][0]["priority"]["calibration_penalty"] == pytest.approx(
        (0.1938 - 0.1920) + (0.228 - 0.205)
    )
    assert action["observation_feedback"]["exact_auth_calibration_count"] == 1
    assert action["score_claim"] is False
    assert action["rank_or_kill_eligible"] is False


def test_cli_requires_identity_for_queue_performance_summary(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.json"
    performance = tmp_path / "performance.json"
    output = tmp_path / "action.json"
    _scorer_response_dataset(scorer)
    performance.write_text(
        json.dumps(
            {
                "schema": "experiment_queue_performance_summary.v1",
                "queue_id": "inverse_action_queue",
                "event_count": 0,
                "by_resource_kind": {},
                "by_step": {},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--queue-performance-summary",
            str(performance),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--queue-performance-runtime-identity" in result.stderr


def test_cli_requires_candidate_identity_for_queue_performance_summary(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    performance = tmp_path / "performance.json"
    runtime_identity = tmp_path / "runtime_identity.json"
    cache_identity = tmp_path / "cache_identity.json"
    output = tmp_path / "action.json"
    _scorer_response_dataset(scorer)
    performance.write_text(
        json.dumps(
            {
                "schema": "experiment_queue_performance_summary.v1",
                "queue_id": "inverse_action_queue",
                "telemetry_only": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "event_count": 1,
                "by_resource_kind": {},
                "by_step": {
                    "legacy.materialize": {
                        "run_count": 1,
                        "success_count": 1,
                        "failure_count": 0,
                        "elapsed_seconds_mean": 2.25,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runtime_identity.write_text(
        json.dumps(
            {
                "runtime_tree_sha256": "d" * 64,
                "scorer_version": "local_scheduler.v1",
            }
        ),
        encoding="utf-8",
    )
    cache_identity.write_text(
        json.dumps({"cache_sha256": "e" * 64}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--queue-performance-summary",
            str(performance),
            "--queue-performance-runtime-identity",
            str(runtime_identity),
            "--queue-performance-cache-identity",
            str(cache_identity),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing candidate_id_by_experiment" in result.stderr
