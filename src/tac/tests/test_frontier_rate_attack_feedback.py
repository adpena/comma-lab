# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from comma_lab.scheduler.frontier_rate_attack_feedback import (
    AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
    AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA,
    AUTONOMOUS_CHAIN_WORK_ORDER_SCHEMA,
    BYTE_RANGE_STAGE_INPUTS_SCHEMA,
    FEEDBACK_REFRESH_SCHEMA,
    LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA,
    OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA,
    OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
    OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
    OPERATION_MATERIALIZER_BRIDGE_SCHEMA,
    OPERATION_PORTFOLIO_SCHEMA,
    OPERATION_PORTFOLIO_TAXONOMY_SCHEMA,
    OPERATOR_ACTION_LEDGER_SCHEMA,
    OPERATOR_ACTION_TERM_SCHEMA,
    RATE_BUDGET_PRESERVATION_PLAN_SCHEMA,
    RATE_BUDGET_PRESERVATION_ROW_SCHEMA,
    RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA,
    RECEIVER_REPAIR_BACKLOG_SCHEMA,
    RECEIVER_REPAIR_ROW_SCHEMA,
    RECEIVER_REPAIR_WORK_ORDER_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
    REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA,
    REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA,
    REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_CHAIN_MATERIALIZER_HANDOFF_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA,
    TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA,
    FrontierRateAttackFeedbackError,
    build_frontier_autonomous_chain_optimization,
    build_frontier_autonomous_chain_optimization_queue,
    build_frontier_autonomous_chain_work_order,
    build_frontier_byte_range_stage_inputs,
    build_frontier_operation_chain_compiler_queue,
    build_frontier_operation_chain_compiler_stage_plan,
    build_frontier_rate_attack_feedback_refresh,
    build_frontier_receiver_repair_work_order,
    build_frontier_repair_budget_materialization_execution_report,
    build_frontier_repair_budget_materialization_plan,
    build_frontier_repair_budget_materializer_binding_report,
    build_frontier_repair_budget_waterfill_queue,
    build_frontier_repair_budget_waterfill_work_order,
    build_frontier_targeted_component_correction_acquisition,
    build_frontier_targeted_component_correction_chain_materializer_handoff,
    build_frontier_targeted_component_correction_chain_work_orders,
    build_frontier_targeted_component_correction_materialization_queue,
    build_frontier_targeted_component_correction_materialization_request,
    build_frontier_targeted_component_correction_materialization_requests,
    build_frontier_targeted_component_correction_queue,
    build_frontier_targeted_component_correction_response_harvest,
    build_frontier_targeted_component_correction_response_harvest_from_artifacts,
    build_frontier_targeted_component_correction_work_order,
    build_frontier_targeted_drop_many_stage_inputs,
    build_receiver_closed_correction_budget,
    discover_local_cpu_eureka_planning_signals,
    discover_materializer_feedback_payloads,
)
from comma_lab.scheduler.frontier_rate_attack_feedback_cycle import (
    AUTOPILOT_RESULT_SCHEMA,
    FrontierRateAttackFeedbackCycleError,
    discover_dqs1_drop_many_greedy_verdict_paths,
    harvest_paths_from_autopilot_payload,
    select_pairset_acquisition_for_harvests,
    write_frontier_refresh_artifacts,
    write_pairset_component_marginal_feedback_bundle,
    write_targeted_component_correction_post_auxiliary_artifacts,
)
from tac.fec6_selector_operator_space import FEC6_FIXED_K16_MODE_IDS
from tac.optimization.repair_dynamics_palette_probe import (
    REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


AUTHORITY_KEYS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _false_authority() -> dict[str, object]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _advisory(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    raw_sha256: str,
    runtime_sha256: str,
    seg: float = 0.055,
    pose: float = 0.017,
) -> dict[str, object]:
    rate = archive_size_bytes / 10_000_000.0
    score = seg + pose + rate
    return {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "score_seg_contribution": seg,
        "score_pose_contribution": pose,
        "score_rate_contribution": rate,
        "archive_size_bytes": archive_size_bytes,
        "provenance": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": runtime_sha256,
            },
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": raw_sha256,
                },
            },
        },
    }


def _action(candidate_id: str, rank: int) -> dict[str, object]:
    return {
        **_false_authority(),
        "candidate_id": candidate_id,
        "operator_action_rank": rank,
        "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
    }


def _row(candidate_id: str, selected_pair_indices: list[int]) -> dict[str, object]:
    return {
        **_false_authority(),
        "candidate_id": candidate_id,
        "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
        "source_metadata": {
            "selected_pair_count": len(selected_pair_indices),
            "selected_pair_indices": selected_pair_indices,
        },
    }


def _write_action_summary(repo: Path) -> Path:
    summary_dir = repo / "experiments" / "results" / "cross_family_candidate_portfolio"
    portfolio = summary_dir / "portfolio.json"
    candidate_rows = [
        ("pairset_drop_one_rank023_pair0440", [1, 2, 440]),
        ("pairset_drop_one_rank024_pair0112", [1, 2, 112]),
        ("pairset_drop_one_rank025_pair0233", [1, 2, 233]),
    ]
    _write_json(
        portfolio,
        {"operator_action_rows": [_row(candidate, pairs) for candidate, pairs in candidate_rows]},
    )
    return _write_json(
        summary_dir / "action_summary.json",
        {
            **_false_authority(),
            "schema": "cross_family_candidate_portfolio_action_summary.v1",
            "json_out": str(portfolio),
            "top_operator_actions": [
                _action(candidate, rank)
                for rank, (candidate, _pairs) in enumerate(candidate_rows, start=1)
            ],
        },
    )


def _materializer_observation(
    *,
    observation_id: str,
    target_kind: str,
    saved_bytes: int,
    receiver_contract_satisfied: bool,
    rate_positive: bool,
) -> dict[str, object]:
    return {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        **_false_authority(),
        "observation_id": observation_id,
        "candidate_id": observation_id,
        "target_kind": target_kind,
        "materializer_id": f"{target_kind}_adapter",
        "saved_bytes": saved_bytes,
        "rate_positive": rate_positive,
        "savings_realized": rate_positive,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "inflate_parity_satisfied": False,
        "recommended_planner_action": (
            "keep_rate_positive_candidate_for_inflate_parity_gate"
            if rate_positive
            else "demote_matching_archive_class_for_materializer"
        ),
        "readiness_blockers": [] if receiver_contract_satisfied else ["runtime_adapter_missing"],
    }


def _eureka_signal(
    *,
    candidate_id: str = "pairset_drop_two_r013_009_p0327_0459",
    projected_contest_score: float = 0.1920291170981836,
    conservative_projected_contest_score: float = 0.1920321170981836,
    local_score: float = 0.19203961709818362,
    archive_sha256: str = "d" * 64,
) -> dict[str, object]:
    return {
        "schema": "local_cpu_contest_drift_eureka_signal.v1",
        **_false_authority(),
        "auth_frontier_score": 0.19202828295713675,
        "authority": "false_authority_exact_eval_spend_trigger_only",
        "bias_local_minus_contest": 0.000010500000000010501,
        "calibration_anchor_count": 4,
        "calibration_confidence": "stable_core",
        "candidate_archive_sha256": archive_sha256,
        "candidate_id": candidate_id,
        "candidate_trust_region": "dqs1_fec6_like_same_archive_segnet_rounding",
        "candidate_trust_region_blockers": [],
        "candidate_trust_region_matches_calibration": True,
        "conservative_projected_contest_score": conservative_projected_contest_score,
        "dispatch_blockers": [
            "optimizer_candidate_queue_is_planning_only",
            "requires_exact_eval_readiness_gate",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
            "requires_non_proxy_score_evidence_before_promotion",
            "eureka_signal_is_not_score_authority",
            "exact_contest_cpu_eval_required_before_frontier_claim",
        ],
        "eureka_margin": 0.19202828295713675 - conservative_projected_contest_score,
        "eureka_trigger": False,
        "guard_band": 0.000003,
        "local_axis": "macOS-CPU advisory",
        "local_score": local_score,
        "projected_contest_score": projected_contest_score,
        "recommended_action": "observe_only",
        "source_artifact": (
            "experiments/results/pareto_gap_uleb/materialized/"
            f"{candidate_id}/local_cpu_advisory.json"
        ),
        "target_axis": "contest-CPU",
        "target_modes": ["contest_exact_eval_planning"],
        "trust_region": "dqs1_fec6_like_same_archive_segnet_rounding",
    }


def _write_materializer_feedback(root: Path) -> Path:
    sweep = _write_json(
        root / "receiver_smoke" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [
                _materializer_observation(
                    observation_id="zip_header_elide_receiver_positive",
                    target_kind="packet_member_zip_header_elide_v1",
                    saved_bytes=52,
                    receiver_contract_satisfied=True,
                    rate_positive=True,
                ),
                _materializer_observation(
                    observation_id="packet_member_merge_receiver_positive",
                    target_kind="packet_member_merge_v1",
                    saved_bytes=258,
                    receiver_contract_satisfied=True,
                    rate_positive=True,
                ),
                _materializer_observation(
                    observation_id="renderer_payload_dfl1_receiver_positive",
                    target_kind="renderer_payload_dfl1_v1",
                    saved_bytes=380,
                    receiver_contract_satisfied=True,
                    rate_positive=True,
                ),
                _materializer_observation(
                    observation_id="packet_member_recompress_no_delta",
                    target_kind="packet_member_recompress_v1",
                    saved_bytes=0,
                    receiver_contract_satisfied=True,
                    rate_positive=False,
                ),
            ],
        },
    )
    repo = root.parent
    exact_handoff_dir = root / "receiver_smoke" / "exact_eval_handoff"
    exact_readiness_report = _write_json(
        exact_handoff_dir / "receiver_smoke_candidate.exact_readiness_report.json",
        {
            "schema": "optimizer_candidate_exact_eval_readiness_report_v1",
            **_false_authority(),
            "candidate_id": "receiver_smoke_candidate",
            "ready_for_exact_eval_dispatch": False,
            "blockers": [
                "archive_manifest_missing",
                "runtime_tree_sha256_missing",
            ],
        },
    )
    source_queue = _write_json(
        exact_handoff_dir / "source_queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            **_false_authority(),
            "top_k": [
                {
                    **_false_authority(),
                    "candidate_id": "receiver_smoke_candidate",
                    "target_kind": "packet_member_zip_header_elide_v1",
                    "candidate_archive_path": (
                        "experiments/results/receiver_smoke/candidate.zip"
                    ),
                    "candidate_archive_sha256": "a" * 64,
                    "source_archive_path": (
                        "experiments/results/receiver_smoke/source.zip"
                    ),
                    "source_archive_sha256": "b" * 64,
                    "source_manifest_path": (
                        "experiments/results/receiver_smoke/source_manifest.json"
                    ),
                    "runtime_consumption_proof_status": "present",
                    "runtime_consumption_proof_path": (
                        "experiments/results/receiver_smoke/runtime_consumption_proof.json"
                    ),
                    "receiver_contract_kind": "packet_member_zip_header_elide_v1",
                    "receiver_contract_satisfied": True,
                    "runtime_adapter_ready": True,
                    "readiness_blockers": [
                        "archive_manifest_missing",
                        "runtime_tree_sha256_missing",
                    ],
                }
            ],
            "dispatch_ready": [],
        },
    )
    _write_json(
        root / "receiver_smoke" / "exact_eval_handoff" / "exact_readiness_bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            **_false_authority(),
            "source_queue_path": source_queue.relative_to(repo).as_posix(),
            "candidate_count": 1,
            "ready_candidate_count": 0,
            "blocked_candidate_count": 1,
            "dispatch_blockers": [
                "bridge_report_is_not_dispatch_authority",
                "requires_exact_eval_readiness_gate",
            ],
            "rows": [
                {
                    **_false_authority(),
                    "candidate_id": "receiver_smoke_candidate",
                    "blockers": [
                        "archive_manifest_missing",
                        "runtime_tree_sha256_missing",
                    ],
                    "exact_readiness_report_path": exact_readiness_report.relative_to(
                        repo
                    ).as_posix(),
                    "exact_ready_queue_path": None,
                }
            ],
        },
    )
    _write_jsonl(
        root / "tensor_probe" / "observations.jsonl",
        [
            _materializer_observation(
                observation_id="zip_header_elide_receiver_positive",
                target_kind="packet_member_zip_header_elide_v1",
                saved_bytes=52,
                receiver_contract_satisfied=True,
                rate_positive=True,
            ),
            _materializer_observation(
                observation_id="tensor_factorize_receiver_missing",
                target_kind="tensor_factorize_v1",
                saved_bytes=7,
                receiver_contract_satisfied=False,
                rate_positive=True,
            )
        ],
    )
    return sweep


def _write_receiver_closed_budget_signal(
    repo: Path,
    *,
    results_root: Path,
    candidate_id: str = "packet_member_zip_header_elide_544c5f580ec2",
    target_kind: str = "packet_member_zip_header_elide_v1",
    saved_bytes: int = 156,
    bridge_blockers: list[str] | None = None,
) -> Path:
    repair_dir = (
        results_root
        / "frontier_operation_portfolio"
        / "frontier_receiver_repair"
        / f"receiver_repair_{target_kind}_unit"
    )
    closure_dir = repair_dir / "submission_closure"
    closure_report = _write_json(
        closure_dir / "submission_closure_report.json",
        {
            "schema": "materializer_submission_runtime_closure_report.v1",
            **_false_authority(),
            "candidate_id": candidate_id,
            "target_kind": target_kind,
            "archive_sha256": "5" * 64,
            "archive_bytes": 345646,
            "closed_source_queue_path": (
                closure_dir / "closed_source_queue.json"
            ).relative_to(repo).as_posix(),
            "submission_dir": (closure_dir / "submission").relative_to(repo).as_posix(),
            "saved_bytes_at_risk": saved_bytes,
            "targeted_correction_budget_signal": {
                "freed_bytes_require_receiver_and_exact_readiness_before_spend": True,
                "saved_bytes_at_risk": saved_bytes,
                **_false_authority(),
            },
            "allowed_use": "exact_readiness_static_submission_closure_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        },
    )
    _write_json(
        repair_dir / "exact_readiness_bridge" / "exact_readiness_bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            **_false_authority(),
            "source_queue_path": (
                closure_dir / "closed_source_queue.json"
            ).relative_to(repo).as_posix(),
            "candidate_count": 1,
            "ready_candidate_count": 0,
            "blocked_candidate_count": 1,
            "dispatch_blockers": [
                "bridge_report_is_not_dispatch_authority",
                "requires_exact_eval_readiness_gate",
                "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
                "requires_non_proxy_score_evidence_before_promotion",
                "optimizer_candidate_queue_is_planning_only",
            ],
            "rows": [
                {
                    **_false_authority(),
                    "candidate_id": candidate_id,
                    "blockers": bridge_blockers
                    if bridge_blockers is not None
                    else [
                        (
                            "above_active_floor_archive_bytes_without_operator_override:"
                            "345646>185578, active_score_frontier=0.206316386616; "
                            "above rate-only byte floor"
                        )
                    ],
                    "exact_readiness_report_path": (
                        repair_dir
                        / "exact_readiness_bridge"
                        / "exact_readiness"
                        / f"{candidate_id}.exact_readiness_report.json"
                    ).relative_to(repo).as_posix(),
                    "exact_ready_queue_path": None,
                    "readiness_verdict": "blocked",
                }
            ],
        },
    )
    return closure_report


def _pair_frame_geometry_lattice(
    *,
    candidate_id: str = "pairset_geometry_lowimpact_k003_habcdef1234",
    selected_pair_indices: list[int] | None = None,
) -> dict[str, object]:
    selected = selected_pair_indices or [1, 2, 112, 233, 440]
    return {
        "schema": "pair_frame_scorer_geometry_lattice.v1",
        **_false_authority(),
        "summary": {
            "queue_executable_request_count": 1,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "queue_executable_pairset_drop_requests": [
            {
                "schema": "pair_frame_geometry_queue_executable_drop_request.v1",
                **_false_authority(),
                "candidate_id": candidate_id,
                "selector_kind": "pair_frame_geometry_low_impact_drop_many",
                "dropped_pair_indices": [3, 4, 5],
                "selected_pair_indices": selected,
                "selected_pair_count": len(selected),
                "geometry_covered_dropped_pair_count": 3,
                "geometry_coverage": 1.0,
                "queue_executable": True,
                "queue_family": "dqs1_pairset_local_first",
                "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
                "allowed_use": "queue_executable_local_dqs1_pairset_drop_probe_only",
                "forbidden_use": "score_claim_or_dispatch_or_promotion_authority",
            }
        ],
    }


def _dqs1_observation_row(
    *,
    candidate_id: str = "pairset_drop_one_rank023_pair0440",
    raw_output_or_cache_sha256: str = "c" * 64,
    family: str = "decoder_q_pairset_drop_one",
    segnet_delta: float = -0.0001,
    posenet_delta: float = 0.0,
    rate_delta: float = -0.00002,
    score_delta: float = -0.00012,
    archive_byte_delta: int = -4,
    selected_pair_indices: list[int] | None = None,
) -> dict[str, object]:
    selected_pairs = selected_pair_indices or [1, 2, 440]
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        **_false_authority(),
        "candidate_id": candidate_id,
        "source_schema": "dqs1_local_first_harvest.v1",
        "sweep_config_id": "dqs1_local_first_macos_cpu_advisory",
        "optimization_pass_id": "local_cpu_advisory_harvest",
        "family": family,
        "observed_axis": "macos_cpu_advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "observed_score_or_delta": 0.1919,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "raw_output_or_cache_sha256": raw_output_or_cache_sha256,
        "component_deltas": {
            "segnet_delta": segnet_delta,
            "posenet_delta": posenet_delta,
            "rate_delta": rate_delta,
        },
        "score_delta_vs_baseline": score_delta,
        "archive_byte_delta_vs_baseline": archive_byte_delta,
        "selected_pair_indices": selected_pairs,
    }


def _drop_many_greedy_negative_verdict() -> dict[str, object]:
    return {
        "schema": "dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1",
        **_false_authority(),
        "captured_at_utc": "2026-05-25T15:30:00Z",
        "lane_id": "lane_dqs1_drop_many_build_1c_fixture",  # FAKE_LANE_OK:test_fixture_or_docstring_or_dict_key_reference_to_lane_token_lane_dqs1_drop_many_build_1c_fixture_NOT_a_real_lane_registry_pre_registration_per_catalog_126_false_positive_per_comprehensive_bug_audit_cascade_20260526
        "build_1c_final_verdict": (
            "NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES"
        ),
        "build_1c_final_verdict_reason": (
            "empirical K>1 sisters regress vs the K=1 drop-one anchor"
        ),
        "greedy_top_k_sweep": [
            {
                "k": 1,
                "selected_pair_indices": [371],
                "cumulative_predicted_delta_vs_base": -6.6e-7,
            },
            {
                "k": 2,
                "selected_pair_indices": [327, 371],
                "cumulative_predicted_delta_vs_base": -1.3e-6,
            },
        ],
        "canonical_equation_candidate_refinement": {
            "candidate_id": "dqs1_drop_many_greedy_independent_pair_ordering_v1",
            "refinement_field_proposed": {
                "empirical_k1_best_drop_one_pair_index": 371,
                "empirical_k1_best_drop_one_delta_vs_base": -6.6e-7,
                "greedy_verdict_class": (
                    "NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES"
                ),
            },
        },
        "catalog_313_probe_outcomes_row": {
            "probe_id": "dqs1_drop_many_build_1c_fixture",
            "verdict": "DEFER",
            "status": "blocking",
        },
    }


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key in AUTHORITY_KEYS:
        assert payload[key] is False


def _operation_chain_stage_plan_payload() -> dict[str, object]:
    return {
        "schema": OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA,
        **_false_authority(),
        "source_operation_id": "chain_registered_multisurface_materializer_program",
        "source_operation_family": "registered_multisurface_materializer_chain",
        "chain_targets": ["byte_range_entropy_recode_v1"],
        "targeted_correction_budget": {
            "schema": "frontier_rate_attack_targeted_correction_budget_summary.v1",
            **_false_authority(),
            "active": True,
            "receiver_closed_materializer_saved_bytes_total": 156,
        },
        "stage_rows": [
            {
                "schema": "frontier_rate_attack_operation_chain_stage_row.v1",
                **_false_authority(),
                "stage_index": 1,
                "stage_id": "payload_grammar_and_entropy",
                "targets": ["byte_range_entropy_recode_v1"],
                "required_before_execution": [
                    "schema_manifest",
                    "beam_probe_reports",
                    "source_runtime_dir",
                ],
                "stage_ready_for_execution": False,
                "blockers": [
                    "payload_grammar_and_entropy_requires:schema_manifest",
                    "payload_grammar_and_entropy_requires:beam_probe_reports",
                    "payload_grammar_and_entropy_requires:source_runtime_dir",
                ],
            }
        ],
        "blockers": [
            "operation_chain_stage_plan_requires_materializer_context_binding",
            "operation_chain_stage_plan_requires_single_runtime_consumption_proof",
        ],
        "execution_ready": False,
    }


def _targeted_drop_many_stage_plan_payload() -> dict[str, object]:
    return {
        "schema": OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA,
        **_false_authority(),
        "source_operation_id": "targeted_component_chain_unit",
        "source_operation_family": (
            "targeted_component_correction_receiver_consumed_multi_op_chain"
        ),
        "chain_targets": [
            "drop_within_selected_set_masked_boundary",
            "inverse_scorer_cell_basis_expansion",
            "pose_stable_pair_frame_motion_correction",
            "full_video_batch_residual_budget_reallocation",
        ],
        "targeted_correction_budget": {
            "schema": "frontier_rate_attack_targeted_chain_budget.v1",
            **_false_authority(),
            "saved_bytes_budget": 258,
            "estimated_receiver_closed_rate_credit_score_units": 0.00017,
            "budget_spend_allowed": False,
        },
        "stage_rows": [
            {
                "schema": "frontier_rate_attack_operation_chain_stage_row.v1",
                **_false_authority(),
                "stage_index": 1,
                "stage_id": "scorer_sensitive_operation_selection",
                "targets": [
                    "drop_within_selected_set_masked_boundary",
                    "inverse_scorer_cell_basis_expansion",
                    "pose_stable_pair_frame_motion_correction",
                    "full_video_batch_residual_budget_reallocation",
                ],
                "required_before_execution": [
                    "paired_cpu_mlx_delta_model",
                    "master_gradient_or_component_marginal_model",
                    "chain_synergy_antagonism_model",
                ],
                "stage_ready_for_execution": False,
                "blockers": [
                    (
                        "scorer_sensitive_operation_selection_requires:"
                        "paired_cpu_mlx_delta_model"
                    )
                ],
            }
        ],
        "blockers": [
            "operation_chain_stage_plan_requires_materializer_context_binding",
            "operation_chain_stage_plan_requires_single_runtime_consumption_proof",
        ],
        "execution_ready": False,
    }


def test_byte_range_stage_inputs_bind_existing_receiver_chain_context(
    tmp_path: Path,
) -> None:
    source_archive = _write_json(tmp_path / "manifest_source.json", {"ok": True})
    source_archive.write_bytes(b"PK\x05\x06" + b"\0" * 18)
    schema_manifest = _write_json(
        tmp_path / "schema_manifest.json",
        {
            "source_archive": {
                "path": source_archive.as_posix(),
                "member_name": "x",
            },
            **_false_authority(),
        },
    )
    beam_probe = _write_json(tmp_path / "stem_weight_beam_probe.json", {"ok": True})
    combo_report = _write_json(tmp_path / "global_combo_probe.json", {"ok": True})
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.py").write_text("BR_LEN = 1\n", encoding="utf-8")
    (runtime_dir / "inflate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    chain_output_dir = tmp_path / "byte_range_chain"
    stage_plan = _operation_chain_stage_plan_payload()

    payload = build_frontier_byte_range_stage_inputs(
        repo_root=tmp_path,
        operation_chain_stage_plan=stage_plan,
        schema_manifest=schema_manifest,
        beam_probe_reports=(beam_probe,),
        source_runtime_dir=runtime_dir,
        source_archive=source_archive,
        global_combo_report=combo_report,
        chain_output_dir=chain_output_dir,
    )

    assert payload["schema"] == BYTE_RANGE_STAGE_INPUTS_SCHEMA
    _assert_false_authority(payload)
    assert payload["local_chain_queueable"] is True
    assert payload["exact_execution_ready"] is False
    assert payload["budget_spend_allowed"] is False
    assert payload["materializer_context"]["context_blockers"] == []
    assert payload["materializer_context"]["schema_manifest"] == (
        "schema_manifest.json"
    )
    assert payload["materializer_context"]["beam_probe_reports"] == [
        "stem_weight_beam_probe.json"
    ]
    assert payload["materializer_context"]["global_combo_report"] == (
        "global_combo_probe.json"
    )
    assert payload["materializer_context"]["member_name"] == "x"
    assert payload["local_chain_command"][:2] == [
        ".venv/bin/python",
        "tools/run_byte_range_entropy_recode_chain.py",
    ]
    assert "--source-runtime-dir" in payload["local_chain_command"]
    assert "byte_range_stage_requires_receiver_proof_after_local_chain" in (
        payload["blockers"]
    )
    assert payload["rate_budget_policy"]["budget_spend_allowed"] is False
    assert "segnet_boundary_repair" in payload["rate_budget_policy"]["freed_bytes_can_fund"]

    stage_plan_path = _write_json(tmp_path / "stage_plan.json", stage_plan)
    cli_out = tmp_path / "stage_inputs_cli.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_byte_range_stage_inputs.py",
            "--operation-chain-stage-plan",
            str(stage_plan_path),
            "--stage-inputs-out",
            str(cli_out),
            "--schema-manifest",
            str(schema_manifest),
            "--beam-probe-report",
            str(beam_probe),
            "--source-runtime-dir",
            str(runtime_dir),
            "--source-archive",
            str(source_archive),
            "--global-combo-report",
            str(combo_report),
            "--chain-output-dir",
            str(tmp_path / "byte_range_chain_cli"),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cli_payload = json.loads(result.stdout)
    assert cli_payload["local_chain_queueable"] is True
    written = json.loads(cli_out.read_text(encoding="utf-8"))
    assert written["schema"] == BYTE_RANGE_STAGE_INPUTS_SCHEMA
    assert written["ready_for_exact_eval_dispatch"] is False


def test_targeted_drop_many_stage_inputs_bind_existing_eureka_geometry(
    tmp_path: Path,
) -> None:
    stage_plan = _targeted_drop_many_stage_plan_payload()

    payload = build_frontier_targeted_drop_many_stage_inputs(
        repo_root=REPO_ROOT,
        operation_chain_stage_plan=stage_plan,
        output_dir=tmp_path / "targeted_drop_many",
    )

    assert payload["schema"] == TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA
    _assert_false_authority(payload)
    assert payload["target_present"] is True
    assert payload["local_plan_queueable"] is True
    assert payload["exact_execution_ready"] is False
    assert payload["budget_spend_allowed"] is False
    assert "drop_within_selected_set_masked_boundary" in payload[
        "selected_family_targets"
    ]
    assert payload["selector_pareto_summary"]["candidate_count"] >= 1
    assert payload["pair_frame_geometry_lattice_summary"][
        "queue_executable_request_count"
    ] >= 1
    assert payload["local_plan_command"][:2] == [
        ".venv/bin/python",
        "tools/plan_decoder_q_pairset_acquisition.py",
    ]
    assert "--drop-many-counts" in payload["local_plan_command"]
    assert payload["rate_budget_policy"]["budget_spend_allowed"] is False

    stage_plan_path = _write_json(tmp_path / "stage_plan.json", stage_plan)
    cli_out = tmp_path / "targeted_drop_many_stage_inputs.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_targeted_drop_many_stage_inputs.py",
            "--operation-chain-stage-plan",
            str(stage_plan_path),
            "--stage-inputs-out",
            str(cli_out),
            "--output-dir",
            str(tmp_path / "targeted_drop_many_cli"),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cli_payload = json.loads(result.stdout)
    assert cli_payload["local_plan_queueable"] is True
    written = json.loads(cli_out.read_text(encoding="utf-8"))
    assert written["schema"] == TARGETED_DROP_MANY_STAGE_INPUTS_SCHEMA
    assert written["ready_for_exact_eval_dispatch"] is False


def test_byte_range_stage_inputs_infer_target_bound_single_member_name(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate_archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("__packet_member_merge_v1.bin", b"payload")
    runtime_dir = tmp_path / "submission"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    stage_plan = _operation_chain_stage_plan_payload()
    stage_plan["source_operation_family"] = (
        "targeted_component_correction_receiver_consumed_multi_op_chain"
    )
    stage_plan["targeted_correction_budget"] = {
        "schema": "frontier_rate_attack_targeted_chain_budget.v1",
        **_false_authority(),
        "receiver_runtime_binding_context": {
            "schema": (
                "frontier_rate_attack_targeted_component_receiver_runtime_binding.v1"
            ),
            **_false_authority(),
            "candidate_archive_path": archive.as_posix(),
            "candidate_submission_dir": runtime_dir.as_posix(),
            "binding_complete_for_component_eval": False,
            "binding_complete_for_reference_eval": False,
        },
    }

    payload = build_frontier_byte_range_stage_inputs(
        repo_root=tmp_path,
        operation_chain_stage_plan=stage_plan,
        chain_output_dir=tmp_path / "targeted_byte_range_chain",
    )

    context = payload["materializer_context"]
    assert payload["local_chain_queueable"] is False
    assert context["default_pr103_context_disabled"] is True
    assert context["source_archive"] == "candidate_archive.zip"
    assert context["member_name"] == "__packet_member_merge_v1.bin"
    assert context["member_name_inference"]["status"] == "inferred"
    assert context["member_name_inference"]["inference_rule"] == (
        "strict_single_member_zip"
    )
    assert "byte_range_stage_missing:member_name" not in context["context_blockers"]
    assert "byte_range_stage_missing:schema_manifest" in context["context_blockers"]
    assert "byte_range_stage_missing:beam_probe_reports" in context[
        "context_blockers"
    ]
    assert "byte_range_stage_missing:source_runtime_dir" not in context[
        "context_blockers"
    ]


def _write_default_byte_range_chain_context(repo: Path) -> None:
    source_archive = repo / "source_archive.zip"
    source_archive.write_bytes(b"PK\x05\x06" + b"\0" * 18)
    _write_json(
        repo
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json",
        {
            "source_archive": {
                "path": source_archive.relative_to(repo).as_posix(),
                "member_name": "x",
            },
            **_false_authority(),
        },
    )
    _write_json(
        repo
        / ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/stem_weight_beam_probe_mid32.json",
        {"schema": "fixture_beam_probe.v1", **_false_authority()},
    )
    runtime = repo / "submissions/hnerv_lc_ac"
    runtime.mkdir(parents=True)
    (runtime / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")
    (runtime / "inflate.sh").write_text("#!/bin/sh\n", encoding="utf-8")


def _byte_range_operation_chain_work_orders() -> dict[str, object]:
    return {
        "schema": OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
        **_false_authority(),
        "work_orders": [
            {
                "schema": OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
                **_false_authority(),
                "source_operation_id": "chain_registered_multisurface_materializer_program",
                "source_operation_family": "registered_multisurface_materializer_chain",
                "chain_targets": ["byte_range_entropy_recode_v1"],
                "stage_plan": [
                    {
                        "stage": "payload_grammar_and_entropy",
                        "targets": ["byte_range_entropy_recode_v1"],
                        "required_before_execution": [
                            "schema_manifest",
                            "beam_probe_reports",
                            "source_runtime_dir",
                        ],
                    }
                ],
                "targeted_correction_budget": {
                    "schema": "frontier_rate_attack_targeted_correction_budget_summary.v1",
                    **_false_authority(),
                    "active": True,
                    "receiver_closed_materializer_saved_bytes_total": 16,
                },
            }
        ],
    }


def test_operation_chain_queue_wires_byte_range_harvest_closure_and_readiness(
    tmp_path: Path,
) -> None:
    _write_default_byte_range_chain_context(tmp_path)
    work_orders = _byte_range_operation_chain_work_orders()
    work_orders_path = _write_json(tmp_path / "operation_chain_work_orders.json", work_orders)

    queue = build_frontier_operation_chain_compiler_queue(
        repo_root=tmp_path,
        operation_chain_compiler_work_orders=work_orders,
        operation_chain_compiler_work_orders_path=work_orders_path,
        results_root=tmp_path / "results",
        queue_id="byte_range_handoff_queue",
    )

    experiment = queue["experiments"][0]
    step_ids = [step["id"] for step in experiment["steps"]]
    assert step_ids == [
        "emit_operation_chain_stage_plan",
        "emit_byte_range_stage_inputs",
        "run_byte_range_entropy_recode_chain",
        "harvest_byte_range_entropy_recode_chain",
        "build_byte_range_submission_closure",
        "run_byte_range_exact_readiness_bridge",
    ]
    assert experiment["steps"][1]["requires"] == ["emit_operation_chain_stage_plan"]
    harvest_step = experiment["steps"][3]
    run_chain_step = experiment["steps"][2]
    assert "--overwrite" in run_chain_step["command"]
    assert run_chain_step["requires"] == ["emit_byte_range_stage_inputs"]
    assert harvest_step["command"][1] == "tools/harvest_materializer_chain_candidates.py"
    assert "--require-accepted" in harvest_step["command"]
    assert harvest_step["requires"] == ["run_byte_range_entropy_recode_chain"]
    closure_step = experiment["steps"][4]
    assert closure_step["command"][1] == "tools/build_materializer_submission_closure.py"
    assert closure_step["requires"] == ["harvest_byte_range_entropy_recode_chain"]
    bridge_step = experiment["steps"][5]
    assert bridge_step["command"][1] == "tools/run_materializer_exact_readiness_bridge.py"
    assert bridge_step["requires"] == ["build_byte_range_submission_closure"]
    assert "--force-recompute" in bridge_step["command"]
    assert any(
        postcondition["type"] == "json_false_authority"
        for postcondition in harvest_step["postconditions"]
    )
    assert any(
        postcondition["type"] == "json_false_authority"
        for postcondition in closure_step["postconditions"]
    )
    assert any(
        postcondition["type"] == "json_false_authority"
        for postcondition in bridge_step["postconditions"]
    )
    metadata = experiment["metadata"]
    assert metadata["byte_range_exact_readiness_handoff_enabled"] is True
    assert metadata["byte_range_harvest_source_queue_path"].endswith(
        "exact_eval_handoff/source_queue.json"
    )
    assert metadata["byte_range_submission_closure_report_path"].endswith(
        "submission_closure/submission_closure_report.json"
    )
    assert metadata["byte_range_exact_readiness_bridge_report_path"].endswith(
        "exact_eval_handoff/exact_readiness_bridge_report.json"
    )
    assert metadata["byte_range_rate_budget_policy"]["budget_spend_allowed"] is False
    _assert_false_authority(metadata)


def test_materializer_feedback_default_discovery_scans_research_candidates_only(
    tmp_path: Path,
) -> None:
    research_root = tmp_path / ".omx" / "research"
    for index in range(8):
        (research_root / f"memo_{index}.md").parent.mkdir(parents=True, exist_ok=True)
        (research_root / f"memo_{index}.md").write_text("not json feedback\n")
    _write_materializer_feedback(research_root / "frontier_artifacts")

    payloads, source_paths, discovery = discover_materializer_feedback_payloads(
        repo_root=tmp_path,
        max_files_per_root=4,
    )

    assert len(payloads) == 2
    assert len(source_paths) == 2
    assert discovery["frontier_artifact_roots"] == [".omx/research"]
    assert discovery["discovered_feedback_count"] == 2
    assert discovery["scanned_candidate_path_count"] == 2
    assert {
        target
        for row in discovery["discovered_feedback"]
        for target in row["target_kinds"]
    } == {
        "packet_member_zip_header_elide_v1",
        "packet_member_merge_v1",
        "packet_member_recompress_v1",
        "renderer_payload_dfl1_v1",
        "tensor_factorize_v1",
    }
    _assert_false_authority(discovery)


def test_materializer_feedback_discovery_accepts_queue_observation_top_k(
    tmp_path: Path,
) -> None:
    queue_observation = _write_json(
        tmp_path / "materializer_queue_observation.json",
        {
            "schema": "experiment_queue_observation.v1",
            **_false_authority(),
            "queue_id": "materializer_queue",
            "healthy": True,
            "succeeded_artifact_steps": [
                {
                    "experiment_id": "renderer_payload_dfl1",
                    "step_id": "harvest_materializer_chains",
                    "status": "succeeded",
                    "resource_kind": "local_cpu",
                    "source_unit_ids": ["renderer_payload_unit"],
                    "source_selection_ids": ["renderer_payload_selection"],
                    "expected_artifacts": [
                        {
                            "path": "source_queue.json",
                            "json_schema": "optimizer_candidate_queue_v1",
                            "optimizer_candidate_queue_materializer_row_count": 1,
                            "optimizer_candidate_queue_materializer_rows": [
                                {
                                    "candidate_id": (
                                        "renderer_payload_dfl1_e20295f0a662"
                                    ),
                                    "target_kind": "renderer_payload_dfl1_v1",
                                    "materializer_id": (
                                        "renderer_payload_dfl1_adapter"
                                    ),
                                    "receiver_contract_kind": (
                                        "source_runtime_native_renderer_payload_dfl1"
                                    ),
                                    "receiver_contract_satisfied": True,
                                    "score_affecting_payload_changed": True,
                                    "charged_bits_changed": True,
                                    "candidate_archive": {
                                        "bytes": 345_422,
                                        "sha256": "e" * 64,
                                    },
                                    "serialized_archive_delta": {
                                        "schema": (
                                            "serialized_archive_delta_contract.v1"
                                        ),
                                        **_false_authority(),
                                        "status": "realized_saving",
                                        "realized_saved_bytes": 380,
                                        "source_archive_bytes": 345_802,
                                        "candidate_archive_bytes": 345_422,
                                        "savings_realized": True,
                                    },
                                    "readiness_blockers": [],
                                    **_false_authority(),
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    payloads, source_paths, discovery = discover_materializer_feedback_payloads(
        repo_root=tmp_path,
        materializer_feedback_paths=(queue_observation,),
    )
    observations = payloads[0]["observations"]

    assert len(payloads) == 1
    assert source_paths == ("materializer_queue_observation.json",)
    assert discovery["discovered_feedback_count"] == 1
    assert observations[0]["target_kind"] == "renderer_payload_dfl1_v1"
    assert observations[0]["saved_bytes"] == 380
    assert observations[0]["rate_positive"] is True
    assert observations[0]["receiver_contract_satisfied"] is True
    assert observations[0]["source_unit_ids"] == ["renderer_payload_unit"]
    assert observations[0]["score_claim"] is False


def test_frontier_feedback_binds_materializer_context_hints_into_work_queue(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    source_archive = tmp_path / "source.zip"
    source_archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    section_manifest = _write_json(tmp_path / "sections.json", {"sections": []})
    candidate_archive = artifact_root / "outputs" / "candidate.zip"
    candidate_manifest = artifact_root / "outputs" / "candidate.json"
    runtime_proof = _write_json(
        artifact_root / "outputs" / "candidate.runtime_consumption_proof.json",
        {
            "schema": "archive_section_entropy_recode_runtime_consumption_proof.v1",
            **_false_authority(),
            "receiver_contract_satisfied": False,
            "proof_status": "failed",
            "blockers": ["section_length_changed_requires_runtime_consumption_proof"],
        },
    )
    _write_json(
        artifact_root / "receiver_negative" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [
                {
                    "schema": "family_agnostic_materializer_empirical_observation.v1",
                    **_false_authority(),
                    "observation_id": "archive_section_recode_context_hint",
                    "candidate_id": "archive_section_recode_context_hint",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer_id": "archive_section_entropy_recode_adapter",
                    "source_archive_path": source_archive.as_posix(),
                    "expected_artifact_paths": [
                        candidate_manifest.as_posix(),
                        candidate_archive.as_posix(),
                        runtime_proof.as_posix(),
                        source_archive.as_posix(),
                        section_manifest.as_posix(),
                    ],
                    "candidate_archive_path": candidate_archive.as_posix(),
                    "manifest_path": candidate_manifest.as_posix(),
                    "runtime_consumption_proof_path": runtime_proof.as_posix(),
                    "saved_bytes": 66,
                    "rate_positive": False,
                    "savings_realized": False,
                    "receiver_contract_satisfied": False,
                    "inflate_parity_satisfied": False,
                    "readiness_blockers": [
                        "section_length_changed_requires_runtime_consumption_proof",
                        "runtime_consumption_proof_not_passed",
                    ],
                }
            ],
        },
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=None,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_context_hints_unit",
        candidate_limit=1,
    )

    bridge = report["operation_materializer_bridge"]
    backlog_row = bridge["materializer_backlog"]["rows"][0]
    assert backlog_row["target_kind"] == "archive_section_entropy_recode_v1"
    params = backlog_row["operation_params"]
    assert params["archive_path"] == source_archive.as_posix()
    assert params["source_archive"] == source_archive.as_posix()
    assert params["section_manifest"] == section_manifest.as_posix()
    assert "runtime_consumption_proof" not in params
    assert params["observed_runtime_consumption_proof_path"] == runtime_proof.as_posix()
    assert params["feedback_readiness_blockers"] == [
        "section_length_changed_requires_runtime_consumption_proof",
        "runtime_consumption_proof_not_passed",
    ]

    context_row = bridge["materializer_contexts"]["rows"][0]
    context = context_row["context"]
    assert context["archive_path"] == source_archive.as_posix()
    assert context["section_manifest"] == section_manifest.as_posix()
    assert "materializer_context_missing:archive_path" not in (
        context_row["context_blockers"]
    )
    assert "materializer_context_missing:section_manifest" not in (
        context_row["context_blockers"]
    )
    assert "runtime_consumption_proof" not in context
    assert "runtime_consumption_proof_missing_hint_ignored" not in context

    work_row = bridge["materializer_work_queue"]["rows"][0]
    assert work_row["executable"] is True
    assert work_row["score_claim"] is False
    assert work_row["ready_for_exact_eval_dispatch"] is False
    assert "materializer_context_missing:archive_path" not in (
        work_row["materialization_blockers"]
    )
    assert "materializer_context_missing:section_manifest" not in (
        work_row["materialization_blockers"]
    )
    assert "--archive-path" in work_row["command"]
    assert "--section-manifest" in work_row["command"]


def test_frontier_feedback_compiler_discovers_materializers_and_refreshes_dqs1_queue(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        dqs1_observation_paths=(dqs1_observations,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_unit",
        candidate_limit=2,
        raw_retention_execute=True,
        raw_retention_cold_store_roots=(str(tmp_path / "cold_store"),),
    )

    assert report["schema"] == FEEDBACK_REFRESH_SCHEMA
    _assert_false_authority(report)
    assert report["materializer_feedback_payload_count"] == 2
    assert report["dqs1_observation_count"] == 1
    assert report["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    discovery = report["discovery"]
    assert discovery["discovered_feedback_count"] == 2
    assert discovery["duplicate_observation_count"] == 1
    discovered_targets = {
        target
        for row in discovery["discovered_feedback"]
        for target in row["target_kinds"]
    }
    assert discovered_targets == {
        "packet_member_zip_header_elide_v1",
        "packet_member_merge_v1",
        "packet_member_recompress_v1",
        "renderer_payload_dfl1_v1",
        "tensor_factorize_v1",
    }
    bridge = report["materializer_feedback_bridge"]
    assert bridge["materializer_observation_count"] == 5
    assert bridge["planned_dqs1_candidate_count"] == 2
    assert bridge["observed_dqs1_candidate_count"] == 1
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert bridge["recommended_next_action"] == (
        "materializer_receiver_positive_followup_before_dqs1_switch"
    )
    operation_portfolio = report["operation_portfolio"]
    assert operation_portfolio["schema"] == OPERATION_PORTFOLIO_SCHEMA
    _assert_false_authority(operation_portfolio)
    taxonomy = operation_portfolio["taxonomy"]
    assert taxonomy["schema"] == OPERATION_PORTFOLIO_TAXONOMY_SCHEMA
    assert "byte_range_entropy_recode_v1" in {
        row["operation_id"].removeprefix("materializer_backlog_")
        for row in taxonomy["registered_missing_materializers"]
    }
    assert "missing_class_range_ans_packet_compiler_target" in {
        row["operation_id"] for row in taxonomy["missing_entire_classes"]
    }
    _assert_false_authority(taxonomy)
    assert operation_portfolio["queue_executable_operation_count"] >= 2
    assert operation_portfolio["followup_signal_operation_count"] >= 6
    assert {"bit", "byte", "packet_member", "pair", "frame", "scorer_axis"}.issubset(
        set(operation_portfolio["operation_level_counts"])
    )
    assert operation_portfolio["component_behavior_summary"]["active"] is True
    assert operation_portfolio["component_behavior_summary"]["best_candidate_id"] == (
        "pairset_drop_one_rank023_pair0440"
    )
    correction_budget = operation_portfolio["targeted_correction_budget_summary"]
    assert correction_budget["active"] is True
    assert correction_budget["local_drop_saved_bytes_max"] == 4
    assert correction_budget["local_drop_rate_credit_score_units_max"] == 0.00002
    assert correction_budget["materializer_rate_positive_saved_bytes_total"] > 0
    assert correction_budget["receiver_closed_materializer_saved_bytes_total"] == 156
    assert correction_budget["receiver_closed_rate_budget_planning_active"] is True
    assert "some_materializer_saved_bytes_still_require_receiver_runtime_proof_before_spend" in (
        correction_budget["blockers"]
    )
    assert correction_budget["score_claim"] is False
    assert correction_budget["ready_for_exact_eval_dispatch"] is False
    rate_plan = operation_portfolio["rate_budget_preservation_plan"]
    assert rate_plan["schema"] == RATE_BUDGET_PRESERVATION_PLAN_SCHEMA
    _assert_false_authority(rate_plan)
    assert rate_plan["active"] is True
    assert rate_plan["rate_only_candidate_count"] >= 1
    assert rate_plan["rate_only_saved_bytes_total"] >= 156
    assert rate_plan["action_functional"]["schema"] == (
        "frontier_rate_attack_operator_action_functional.v1"
    )
    assert "operator_synergy_or_antagonism_terms" in rate_plan[
        "action_functional"
    ]["state_variables"]
    assert "lagrangian_waterfill" in rate_plan["action_functional"]["discrete_solver"]
    assert rate_plan["action_functional"]["operator_action_ledger_schema"] == (
        OPERATOR_ACTION_LEDGER_SCHEMA
    )
    assert rate_plan["operator_action_ledger"]["schema"] == OPERATOR_ACTION_LEDGER_SCHEMA
    assert (
        rate_plan["operator_action_ledger"]["term_schema"]
        == OPERATOR_ACTION_TERM_SCHEMA
    )
    first_action_term = rate_plan["operator_action_ledger"]["terms"][0]
    assert first_action_term["schema"] == OPERATOR_ACTION_TERM_SCHEMA
    assert first_action_term["T_i"]["archive_byte_delta_vs_baseline"] < 0
    assert "receiver_consumes_materialized_runtime_output" in (
        first_action_term["legal_runtime_constraints"]
    )
    assert rate_plan["cumulative_rate_attack"][
        "preserve_cumulative_rate_only_archive"
    ] is True
    assert rate_plan["cumulative_rate_attack"]["operator_action_term_count"] == (
        rate_plan["operator_action_ledger"]["term_count"]
    )
    assert rate_plan["cumulative_rate_attack"][
        "emit_cumulative_rate_only_before_any_distortion_spend"
    ] is True
    assert rate_plan["waterfill_solver"]["rebrotli_default_after_rate_attack"] is True
    assert rate_plan["waterfill_solver"]["budget_spend_allowed"] is False
    assert {
        "dqs1_component_rate_budget",
        "receiver_closed_materializer_rate_budget",
    }.issubset(set(rate_plan["source_kinds"]))
    assert all(
        row["schema"] == RATE_BUDGET_PRESERVATION_ROW_SCHEMA
        and row["preserve_as_rate_only_candidate"] is True
        and row["rate_only_archive_preservation_required"] is True
        for row in rate_plan["rows"]
    )
    targeted_queue = build_frontier_targeted_component_correction_queue(
        repo_root=tmp_path,
        targeted_component_correction_acquisition=report[
            "targeted_component_correction_acquisition"
        ],
        targeted_component_correction_acquisition_path=(
            tmp_path / "targeted_component_correction_acquisition.json"
        ),
        results_root=str(results_root),
        queue_id="targeted_family_round_robin_unit",
        candidate_limit=4,
        include_mlx_response=False,
    )
    assert targeted_queue is not None
    assert targeted_queue["selection_policy"]["policy"] == (
        "bounded_candidate_family_round_robin"
    )
    assert targeted_queue["selection_policy"]["selected_row_count"] == 4
    assert targeted_queue["selection_policy"][
        "selected_correction_family_count"
    ] >= 4
    assert len(targeted_queue["experiments"]) == 1
    assert (
        targeted_queue["experiments"][0]["metadata"]["selected_acquisition_count"]
        == 4
    )
    assert (
        targeted_queue["experiments"][0]["metadata"]["shared_component_response_reuse"]
        is True
    )
    local_cpu_step = next(
        step
        for step in targeted_queue["experiments"][0]["steps"]
        if step["id"] == "local_cpu_component_advisory"
    )
    assert "--scorer-input-cache-hashes-out" in local_cpu_step["command"]
    assert (
        "--allow-scorer-input-cache-artifact-output-outside-work-dir"
        in local_cpu_step["command"]
    )
    assert {
        request["correction_family"]
        for experiment in targeted_queue["experiments"]
        for request in experiment["metadata"]["correction_requests"]
    } >= {
        "segnet_posenet_waterfill_region_repair",
        "drop_within_selected_set_masked_boundary",
        "inverse_scorer_cell_basis_expansion",
    }
    assert all(
        experiment["metadata"]["selection_policy"]["budget_spend_allowed"] is False
        for experiment in targeted_queue["experiments"]
    )
    registered_chain = next(
        row
        for row in operation_portfolio["rows"]
        if row["operation_id"] == "chain_registered_multisurface_materializer_program"
    )
    assert registered_chain["queue_executable"] is False
    assert registered_chain["followup_signal"] is True
    assert registered_chain["queue_consumer"] == (
        "frontier_materializer_chain_acquisition_queue"
    )
    assert {
        "byte_range_entropy_recode_v1",
        "archive_section_header_elide_v1",
        "archive_section_reorder_v1",
        "tensor_quantize_v1",
        "tensor_prune_v1",
        "tensor_shared_codebook_v1",
    }.issubset(set(registered_chain["evidence_summary"]["chain_targets"]))
    assert registered_chain["evidence_summary"]["queue_work_order_schema"] == (
        "frontier_rate_attack_operation_chain_compiler_work_order.v1"
    )
    assert "single_receiver_adapter_amortizes_multiple_archive_ops" in (
        registered_chain["evidence_summary"]["synergy_terms_to_measure"]
    )
    assert "chain_requires_single_composed_receiver_runtime_consumption_proof" in (
        registered_chain["blockers"]
    )
    operation_materializer = report["operation_materializer_bridge"]
    assert operation_materializer["schema"] == OPERATION_MATERIALIZER_BRIDGE_SCHEMA
    _assert_false_authority(operation_materializer)
    assert operation_materializer["bridge_row_count"] >= 6
    assert operation_materializer["materializer_backlog_row_count"] == 2
    assert operation_materializer["work_queue_row_count"] == 2
    assert operation_materializer["blocked_work_row_count"] == 2
    assert operation_materializer["executable_work_row_count"] == 0
    assert operation_materializer["materializer_backlog"]["schema"] == (
        "byte_shaving_materializer_backlog.v1"
    )
    assert operation_materializer["selected_materializer_targets"] == [
        "renderer_payload_dfl1_v1",
        "packet_member_merge_v1",
    ]
    registered_chain_bridge = next(
        row
        for row in operation_materializer["rows"]
        if row["source_operation_id"]
        == "chain_registered_multisurface_materializer_program"
    )
    assert registered_chain_bridge["chain_compiler_work_order"]["schema"] == (
        "frontier_rate_attack_operation_chain_compiler_work_order.v1"
    )
    assert "operation_portfolio_chain_requires_chain_compiler_work_order" in (
        registered_chain_bridge["blockers"]
    )
    assert "operation_portfolio_row_has_no_registered_materializer_target" not in (
        registered_chain_bridge["blockers"]
    )
    assert registered_chain_bridge["score_claim"] is False
    assert registered_chain_bridge["ready_for_exact_eval_dispatch"] is False
    chain_work_orders_payload = {
        "schema": "frontier_rate_attack_operation_chain_compiler_work_orders.v1",
        "work_orders": [
            {
                **registered_chain_bridge["chain_compiler_work_order"],
                "source_bridge_blockers": registered_chain_bridge["blockers"],
            }
        ],
        **_false_authority(),
    }
    chain_stage_plan = build_frontier_operation_chain_compiler_stage_plan(
        operation_chain_compiler_work_orders=chain_work_orders_payload,
        source_operation_id="chain_registered_multisurface_materializer_program",
    )
    assert chain_stage_plan["schema"] == OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA
    assert chain_stage_plan["execution_ready"] is False
    assert chain_stage_plan["stage_count"] == 4
    assert "single_composed_receiver_runtime_consumption_proof" in (
        chain_stage_plan["missing_contracts"]
    )
    assert "frontier_targeted_component_correction_queue" in {
        handoff["queue_consumer"] for handoff in chain_stage_plan["queue_handoffs"]
    }
    chain_work_orders_path = tmp_path / "operation_chain_compiler_work_orders.json"
    _write_json(chain_work_orders_path, chain_work_orders_payload)
    chain_queue = build_frontier_operation_chain_compiler_queue(
        repo_root=tmp_path,
        operation_chain_compiler_work_orders=chain_work_orders_payload,
        operation_chain_compiler_work_orders_path=chain_work_orders_path,
        results_root=tmp_path / "results",
        queue_id="chain_queue_unit",
    )
    assert chain_queue is not None
    assert chain_queue["schema"] == "experiment_queue.v1"
    assert chain_queue["experiments"][0]["metadata"]["execution_ready"] is False
    assert chain_queue["experiments"][0]["metadata"][
        "byte_range_local_chain_queueable"
    ] is False
    assert chain_queue["experiments"][0]["metadata"][
        "byte_range_stage_inputs_path"
    ].endswith("byte_range_stage_inputs.json")
    assert chain_queue["experiments"][0]["metadata"][
        "byte_range_rate_budget_policy"
    ]["budget_spend_allowed"] is False
    _assert_false_authority(chain_queue["experiments"][0]["metadata"])
    assert chain_queue["experiments"][0]["steps"][0]["postconditions"][0][
        "equals"
    ] == OPERATION_CHAIN_COMPILER_STAGE_PLAN_SCHEMA
    assert chain_queue["experiments"][0]["steps"][1]["command"][1] == (
        "tools/build_frontier_byte_range_stage_inputs.py"
    )
    assert chain_queue["experiments"][0]["steps"][1]["postconditions"][0][
        "equals"
    ] == BYTE_RANGE_STAGE_INPUTS_SCHEMA
    assert all(
        row["source_selection_samples"][0]["selection_kind"] == "materializer_feedback"
        for row in operation_materializer["materializer_backlog"]["rows"]
    )
    assert operation_materializer["materializer_contexts"]["schema"] == (
        "byte_shaving_materializer_contexts.v1"
    )
    assert operation_materializer["materializer_work_queue"]["schema"] == (
        "byte_shaving_materializer_work_queue.v1"
    )
    assert "packet_member_merge_v1" in set(
        operation_materializer["top_materializer_targets"]
    )
    assert all(
        row["score_claim"] is False
        and row["ready_for_exact_eval_dispatch"] is False
        for row in operation_materializer["rows"]
    )
    receiver_closed_budget = report["receiver_closed_correction_budget"]
    assert receiver_closed_budget["schema"] == RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA
    _assert_false_authority(receiver_closed_budget)
    assert receiver_closed_budget["active"] is True
    assert receiver_closed_budget["receiver_closed_candidate_count"] == 1
    assert receiver_closed_budget["receiver_closed_saved_bytes_total"] == 156
    assert receiver_closed_budget["duplicate_closure_report_count"] == 0
    assert receiver_closed_budget["rows"][0]["release_to_targeted_correction_planning"] is True
    assert receiver_closed_budget["rows"][0]["ready_for_budget_spend"] is False
    assert "active_rate_floor_override_required_before_exact_dispatch" in (
        receiver_closed_budget["blockers"]
    )
    targeted_component = report["targeted_component_correction_acquisition"]
    assert (
        targeted_component["schema"]
        == TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA
    )
    _assert_false_authority(targeted_component)
    assert targeted_component["active"] is True
    assert targeted_component["receiver_closed_saved_bytes_total"] == 156
    assert targeted_component["queue_actionable_acquisition_count"] >= 5
    assert {
        "pair",
        "frame",
        "region",
        "boundary",
        "batch",
        "full_video",
    }.issubset(set(targeted_component["targeted_dimensions"]))
    assert targeted_component["best_component_behavior_candidate_id"] == (
        "pairset_drop_one_rank023_pair0440"
    )
    assert "component_eval_required_before_budget_spend" in targeted_component[
        "blockers"
    ]
    assert targeted_component["ready_for_exact_eval_dispatch"] is False
    assert any(
        row["correction_family"] == "drop_within_selected_set_masked_boundary"
        and row["queue_actionable"] is True
        and row["ready_for_budget_spend"] is False
        and row["budget_spend_allowed"] is False
        and "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
        in row["forbidden_use"]
        for row in targeted_component["rows"]
    )
    first_acquisition_id = targeted_component["top_acquisition_ids"][0]
    correction_work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=targeted_component,
        acquisition_id=first_acquisition_id,
    )
    assert (
        correction_work_order["schema"]
        == TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA
    )
    _assert_false_authority(correction_work_order)
    assert correction_work_order["budget_spend_gate"]["ready_for_budget_spend"] is False
    assert correction_work_order["budget_spend_gate"]["budget_spend_allowed"] is False
    assert correction_work_order["command_hints"]
    assert correction_work_order["lagrangian_acceptance_rule"][
        "component_eval_required"
    ] is True
    assert operation_portfolio["top_operation_ids"][0] == (
        "dqs1_component_coupled_pair_batch_expansion"
    )
    assert operation_portfolio["top_queue_executable_operation_ids"][0] == (
        "dqs1_component_coupled_pair_batch_expansion"
    )
    chain_row = next(
        row
        for row in operation_portfolio["rows"]
        if row["operation_id"] == "chain_dfl1_merge_header_elide_minimal_envelope"
    )
    assert chain_row["queue_executable"] is False
    assert chain_row["followup_signal"] is True
    assert chain_row["evidence_summary"]["chain_targets"] == [
        "renderer_payload_dfl1_v1",
        "packet_member_merge_v1",
        "packet_member_zip_header_elide_v1",
    ]
    chain_bridge = chain_row["evidence_summary"]["exact_readiness_bridge_summary"]
    assert chain_bridge["bridge_report_count"] == 1
    assert chain_bridge["ready_candidate_count"] == 0
    assert "archive_manifest_missing" in chain_bridge["top_blockers"]
    assert "chain_exact_readiness_bridges_have_no_ready_candidate" in chain_row[
        "blockers"
    ]
    merge_row = next(
        row
        for row in operation_portfolio["rows"]
        if row["operation_id"] == "materializer_packet_member_merge_v1"
    )
    merge_bridge = merge_row["evidence_summary"]["exact_readiness_bridge"]
    assert merge_bridge["bridge_report_count"] == 1
    assert "exact_readiness_bridge:archive_manifest_missing" in merge_row["blockers"]
    receiver_repair_backlog = report["receiver_repair_backlog"]
    assert receiver_repair_backlog["schema"] == RECEIVER_REPAIR_BACKLOG_SCHEMA
    _assert_false_authority(receiver_repair_backlog)
    assert receiver_repair_backlog["row_count"] >= 4
    assert receiver_repair_backlog["queue_actionable_repair_count"] >= 2
    assert receiver_repair_backlog["materializer_rate_positive_saved_bytes_total"] > 0
    assert receiver_repair_backlog["targeted_correction_budget_active"] is True
    assert "submission_runtime_manifest_closure" in receiver_repair_backlog[
        "repair_family_counts"
    ]
    assert "authority_gate" in receiver_repair_backlog["repair_family_counts"]
    assert "submission_runtime_manifest_closure" in receiver_repair_backlog[
        "top_repair_families"
    ]
    assert any(
        row["source_operation_id"] == "materializer_backlog_byte_range_entropy_recode_v1"
        and row["queue_actionable"] is True
        and row["bridge_report_paths"] == []
        for row in receiver_repair_backlog["rows"]
    )
    first_repair = next(
        row for row in receiver_repair_backlog["rows"] if row["bridge_report_paths"]
    )
    assert first_repair["schema"] == RECEIVER_REPAIR_ROW_SCHEMA
    _assert_false_authority(first_repair)
    assert first_repair["source_operation_id"] in {
        "chain_dfl1_merge_header_elide_minimal_envelope",
        "materializer_packet_member_merge_v1",
        "materializer_packet_member_zip_header_elide_v1",
        "materializer_renderer_payload_dfl1_v1",
    }
    assert first_repair["bridge_report_paths"]
    assert first_repair["candidate_ids"] == ["receiver_smoke_candidate"]
    assert first_repair["correction_budget_context"][
        "materializer_rate_positive_saved_bytes_total"
    ] == receiver_repair_backlog["materializer_rate_positive_saved_bytes_total"]
    work_order = build_frontier_receiver_repair_work_order(
        repo_root=tmp_path,
        receiver_repair_backlog=receiver_repair_backlog,
        repair_id=first_repair["repair_id"],
    )
    assert work_order["schema"] == RECEIVER_REPAIR_WORK_ORDER_SCHEMA
    _assert_false_authority(work_order)
    assert work_order["bridge_report_paths"] == first_repair["bridge_report_paths"]
    assert work_order["bridge_details"]["bridge_reports"]
    assert work_order["bridge_details"]["source_queue_paths"] == [
        "frontier_artifacts/receiver_smoke/exact_eval_handoff/source_queue.json"
    ]
    assert work_order["bridge_details"]["candidate_rows"]
    assert work_order["budget_spend_gate"][
        "ready_for_targeted_correction_budget_spend"
    ] is False
    assert work_order["command_hints"]
    recompress_row = next(
        row
        for row in operation_portfolio["rows"]
        if row["operation_id"] == "materializer_packet_member_recompress_v1"
    )
    assert recompress_row["queue_executable"] is False
    assert recompress_row["followup_signal"] is True
    assert recompress_row["suppression_keys"]
    assert "same_archive_target_member_negative_rate_feedback" in recompress_row[
        "blockers"
    ]
    queue = report["queue"]
    assert queue["queue_id"] == "frontier_feedback_unit"
    assert len(queue["experiments"]) == 2
    first_steps = {step["id"]: step for step in queue["experiments"][0]["steps"]}
    raw_retention_command = first_steps["plan_raw_artifact_retention"]["command"]
    assert "--execute" in raw_retention_command
    assert "--cold-store-root" in raw_retention_command
    assert all(
        experiment["metadata"]["materializer_feedback_bridge"] == bridge
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"]["operation_count"]
        == operation_portfolio["operation_count"]
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"]["top_operation_ids"]
        == operation_portfolio["top_operation_ids"]
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"][
            "top_queue_executable_operation_ids"
        ]
        == operation_portfolio["top_queue_executable_operation_ids"]
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"][
            "top_followup_signal_operation_ids"
        ]
        == operation_portfolio["top_followup_signal_operation_ids"]
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"][
            "followup_signal_operation_count"
        ]
        == operation_portfolio["followup_signal_operation_count"]
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"][
            "targeted_correction_budget_active"
        ]
        is True
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_operation_portfolio"][
            "receiver_closed_correction_budget_active"
        ]
        is True
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_receiver_closed_correction_budget"][
            "receiver_closed_saved_bytes_total"
        ]
        == 156
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"][
            "frontier_targeted_component_correction_acquisition"
        ]["active"]
        is True
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"][
            "frontier_targeted_component_correction_acquisition"
        ]["receiver_closed_saved_bytes_total"]
        == 156
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_receiver_repair_backlog"]["row_count"]
        == receiver_repair_backlog["row_count"]
        for experiment in queue["experiments"]
    )
    assert all(
        experiment["metadata"]["frontier_receiver_repair_backlog"][
            "targeted_correction_budget_active"
        ]
        is True
        for experiment in queue["experiments"]
    )
    assert report["queue_summary"]["experiment_count"] == 2
    assert report["queue_summary"]["score_claim"] is False


def test_rate_budget_preservation_keeps_rate_only_floor_for_distortion_regressions(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [
            _dqs1_observation_row(
                candidate_id="pairset_drop_many_rate_floor_regression",
                family="decoder_q_pairset_drop_many",
                segnet_delta=0.0002,
                posenet_delta=0.0001,
                rate_delta=-0.00002,
                score_delta=0.00028,
                archive_byte_delta=-4,
                selected_pair_indices=[1, 2, 112, 233, 371],
            )
        ],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        dqs1_observation_paths=(dqs1_observations,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_rate_preserve_unit",
        candidate_limit=2,
    )

    rate_plan = report["operation_portfolio"]["rate_budget_preservation_plan"]
    assert rate_plan["schema"] == RATE_BUDGET_PRESERVATION_PLAN_SCHEMA
    _assert_false_authority(rate_plan)
    assert rate_plan["active"] is True
    assert rate_plan["rate_positive_distortion_regression_count"] == 1
    assert rate_plan["rate_positive_distortion_regression_candidate_ids"] == [
        "pairset_drop_many_rate_floor_regression"
    ]
    dqs1_row = next(
        row
        for row in rate_plan["rows"]
        if row["candidate_id"] == "pairset_drop_many_rate_floor_regression"
    )
    assert dqs1_row["schema"] == RATE_BUDGET_PRESERVATION_ROW_SCHEMA
    _assert_false_authority(dqs1_row)
    assert dqs1_row["source_kind"] == "dqs1_component_rate_budget"
    assert dqs1_row["saved_bytes"] == 4
    assert dqs1_row["distortion_debt_score_units"] == pytest.approx(0.0003)
    assert dqs1_row["budget_reinvestment_candidate"] is True
    assert dqs1_row["preserve_as_rate_only_candidate"] is True
    assert dqs1_row["minimum_repair_score_units_to_beat_rate_only_floor"] == (
        pytest.approx(0.00028)
    )
    assert rate_plan["cumulative_rate_attack"][
        "emit_cumulative_rate_only_before_any_distortion_spend"
    ] is True
    assert rate_plan["action_functional"]["objective"] == (
        "minimize_S_under_receiver_and_exact_readiness_constraints"
    )
    assert rate_plan["operator_action_ledger"]["schema"] == OPERATOR_ACTION_LEDGER_SCHEMA
    assert rate_plan["operator_action_ledger"]["terms"][0]["schema"] == (
        OPERATOR_ACTION_TERM_SCHEMA
    )
    assert rate_plan["waterfill_solver"]["acceptance_rule"].startswith(
        "emit_rate_only_archive_first"
    )
    assert report["rate_budget_preservation_plan"] == rate_plan
    assert report["queue"]["experiments"][0]["metadata"][
        "frontier_operation_portfolio"
    ]["rate_positive_distortion_regression_count"] == 1


def test_targeted_component_correction_response_harvest_measures_lagrangian(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_response_harvest_unit",
        candidate_limit=2,
    )
    targeted_component = report["targeted_component_correction_acquisition"]
    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=targeted_component,
        acquisition_id=targeted_component["top_acquisition_ids"][0],
    )
    local_cpu_advisory = {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "canonical_score": 0.2,
        "archive_size_bytes": 345658,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
        "component_deltas": {
            "segnet_delta": -0.00012,
            "posenet_delta": 0.00001,
            "archive_byte_delta_vs_receiver_closed_candidate": 12,
        },
    }
    mlx_response = {
        "schema_version": "mlx_scorer_response.v1",
        **_false_authority(),
        "score_axis": "[macOS-MLX research-signal]",
        "response_family": "targeted_component_correction_receiver_closed_budget",
        "canonical_score": 0.2,
        "archive_size_bytes": 345658,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
        "component_deltas": {
            "segnet_delta": -0.00010,
            "posenet_delta": 0.00001,
            "archive_byte_delta_vs_receiver_closed_candidate": 12,
        },
    }

    row = build_frontier_targeted_component_correction_response_harvest_from_artifacts(
        work_order=work_order,
        local_cpu_advisory=local_cpu_advisory,
        local_mlx_response=mlx_response,
        work_order_path="work_order.json",
        local_cpu_advisory_path="local_cpu_advisory.json",
        local_mlx_response_path="mlx_scorer_response.json",
        response_artifact_path="component_correction_response_harvest.json",
    )
    harvest = build_frontier_targeted_component_correction_response_harvest(
        repo_root=tmp_path,
        response_rows=(row,),
    )

    assert harvest["schema"] == TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA
    _assert_false_authority(harvest)
    assert harvest["row_count"] == 1
    assert harvest["local_acquisition_recommended_count"] == 1
    assert harvest["ready_for_budget_spend_count"] == 0
    assert row["negative_measured_lagrangian_delta"] is True
    assert row["local_acquisition_recommended"] is True
    assert row["budget_spend_allowed"] is False
    assert row["measured_lagrangian_delta_score_units"] < 0.0
    assert "exact_axis_component_response_required_before_budget_spend" in row[
        "budget_spend_blockers"
    ]


def test_targeted_component_response_harvest_derives_paired_local_cpu_deltas(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_paired_response_delta_unit",
        candidate_limit=2,
    )
    targeted_component = report["targeted_component_correction_acquisition"]
    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=targeted_component,
        acquisition_id=targeted_component["top_acquisition_ids"][0],
    )
    candidate_advisory = {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "archive_size_bytes": 345_544,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.0008,
    }
    reference_advisory = {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "archive_size_bytes": 345_802,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
    }

    row = build_frontier_targeted_component_correction_response_harvest_from_artifacts(
        work_order=work_order,
        local_cpu_advisory=candidate_advisory,
        reference_local_cpu_advisory=reference_advisory,
        work_order_path="work_order.json",
        local_cpu_advisory_path="candidate_local_cpu_advisory.json",
        reference_local_cpu_advisory_path="reference_local_cpu_advisory.json",
        response_artifact_path="component_correction_response_harvest.json",
    )

    _assert_false_authority(row)
    assert row["local_cpu_component_terms"]["segnet_delta_score_units"] == pytest.approx(
        -0.02
    )
    assert row["local_cpu_component_terms"]["posenet_delta_score_units"] == pytest.approx(
        0.0
    )
    assert row["local_cpu_component_terms"]["correction_rate_delta_score_units"] == 0.0
    assert row["local_cpu_paired_reference_terms"][
        "receiver_closed_archive_byte_delta_vs_reference"
    ] == -258
    assert row["local_cpu_score_delta_summary"][
        "component_delta_score_units"
    ] == pytest.approx(-0.02)
    _assert_false_authority(row["local_cpu_score_delta_summary"])
    assert row["local_cpu_score_delta_summary"][
        "receiver_closed_rate_delta_score_units"
    ] == pytest.approx(-0.0001717916099055202)
    assert row["local_cpu_score_delta_summary"][
        "receiver_closed_total_delta_score_units"
    ] == pytest.approx(-0.020171791609905523)
    assert row["negative_measured_lagrangian_delta"] is True
    assert row["local_acquisition_recommended"] is True
    assert row["reference_local_cpu_advisory_path"] == "reference_local_cpu_advisory.json"
    assert (
        "paired_reference_local_cpu_advisory_required_for_component_delta"
        not in row["budget_spend_blockers"]
    )
    assert "exact_axis_component_response_required_before_budget_spend" in row[
        "budget_spend_blockers"
    ]


def test_targeted_component_response_harvest_derives_paired_local_mlx_deltas(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_paired_mlx_response_delta_unit",
        candidate_limit=2,
    )
    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=report[
            "targeted_component_correction_acquisition"
        ],
        acquisition_id=report["targeted_component_correction_acquisition"][
            "top_acquisition_ids"
        ][0],
    )
    candidate_advisory = {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "archive_size_bytes": 345_544,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.0008,
    }
    reference_advisory = {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "archive_size_bytes": 345_802,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
    }
    candidate_mlx = {
        "schema_version": "mlx_scorer_response.v1",
        **_false_authority(),
        "score_axis": "[macOS-MLX research-signal]",
        "response_family": "targeted_component_correction_receiver_closed_budget",
        "archive_size_bytes": 345_544,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.0007,
    }
    reference_mlx = {
        "schema_version": "mlx_scorer_response.v1",
        **_false_authority(),
        "score_axis": "[macOS-MLX research-signal]",
        "response_family": "targeted_component_correction_receiver_closed_reference",
        "archive_size_bytes": 345_802,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
    }

    row = build_frontier_targeted_component_correction_response_harvest_from_artifacts(
        work_order=work_order,
        local_cpu_advisory=candidate_advisory,
        reference_local_cpu_advisory=reference_advisory,
        local_mlx_response=candidate_mlx,
        reference_local_mlx_response=reference_mlx,
        work_order_path="work_order.json",
        local_cpu_advisory_path="candidate_local_cpu_advisory.json",
        reference_local_cpu_advisory_path="reference_local_cpu_advisory.json",
        local_mlx_response_path="candidate_mlx.json",
        reference_local_mlx_response_path="reference_mlx.json",
        response_artifact_path="component_correction_response_harvest.json",
    )

    _assert_false_authority(row)
    assert row["local_mlx_component_terms"]["segnet_delta_score_units"] == pytest.approx(
        -0.03
    )
    assert row["local_mlx_component_terms"]["posenet_delta_score_units"] == pytest.approx(
        0.0
    )
    assert row["local_mlx_paired_reference_terms"][
        "receiver_closed_archive_byte_delta_vs_reference"
    ] == -258
    assert row["local_mlx_score_delta_summary"][
        "component_delta_score_units"
    ] == pytest.approx(-0.03)
    _assert_false_authority(row["local_mlx_score_delta_summary"])
    assert row["local_mlx_score_delta_summary"][
        "receiver_closed_rate_delta_score_units"
    ] == pytest.approx(-0.0001717916099055202)
    assert row["local_mlx_score_delta_summary"][
        "receiver_closed_total_delta_score_units"
    ] == pytest.approx(-0.03017179160990552)
    assert row["local_mlx_vs_local_cpu_drift_terms"][
        "mlx_minus_local_cpu_segnet_score_units"
    ] == pytest.approx(-0.01)
    assert row["local_mlx_vs_local_cpu_paired_delta_drift_terms"][
        "mlx_minus_local_cpu_segnet_delta_score_units"
    ] == pytest.approx(-0.01)
    assert row["local_mlx_vs_local_cpu_paired_delta_drift_terms"][
        "mlx_minus_local_cpu_paired_lagrangian_delta_score_units"
    ] == pytest.approx(-0.01)
    assert row["reference_local_mlx_response_path"] == "reference_mlx.json"
    assert "local_mlx_component_delta_missing" not in row["budget_spend_blockers"]
    harvest = build_frontier_targeted_component_correction_response_harvest(
        repo_root=tmp_path,
        response_rows=(row,),
    )
    assert harvest["mlx_cpu_drift_summary"]["absolute_score_drift_max_abs"] == (
        pytest.approx(0.01)
    )
    assert harvest["mlx_cpu_drift_summary"][
        "paired_lagrangian_delta_drift_max_abs"
    ] == pytest.approx(0.01)
    requests = build_frontier_targeted_component_correction_materialization_requests(
        targeted_component_correction_response_harvest=harvest,
        candidate_limit=1,
        family_limit_per_candidate=1,
    )
    request = requests["rows"][0]
    assert request["best_local_mlx_score_delta_summary"][
        "receiver_closed_total_delta_score_units"
    ] == pytest.approx(-0.03017179160990552)
    _assert_false_authority(request["best_local_mlx_score_delta_summary"])
    assert request["materializer_chain_basis"][0]["local_mlx_score_delta_summary"][
        "receiver_closed_rate_delta_score_units"
    ] == pytest.approx(-0.0001717916099055202)
    _assert_false_authority(
        request["materializer_chain_basis"][0]["local_mlx_score_delta_summary"]
    )


def test_targeted_component_response_harvest_cli_accepts_reference_advisory(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)
    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_paired_response_cli_unit",
        candidate_limit=1,
    )
    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=report[
            "targeted_component_correction_acquisition"
        ],
        acquisition_id=report["targeted_component_correction_acquisition"][
            "top_acquisition_ids"
        ][0],
    )
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    candidate_path = _write_json(
        tmp_path / "candidate_local_cpu_advisory.json",
        {
            "schema_version": "contest_auth_eval_result.v1",
            **_false_authority(),
            "score_axis": "cpu_advisory",
            "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
            "archive_size_bytes": 345_544,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.0008,
        },
    )
    reference_path = _write_json(
        tmp_path / "reference_local_cpu_advisory.json",
        {
            "schema_version": "contest_auth_eval_result.v1",
            **_false_authority(),
            "score_axis": "cpu_advisory",
            "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
            "archive_size_bytes": 345_802,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.001,
        },
    )
    output = tmp_path / "component_correction_response_harvest.json"
    materialization_requests = tmp_path / "materialization_requests.json"
    materialization_queue = tmp_path / "materialization_queue.json"
    operation_chain_work_orders = tmp_path / "operation_chain_work_orders.json"
    operation_chain_queue = tmp_path / "operation_chain_queue.json"
    operation_chain_handoff = tmp_path / "operation_chain_handoff.json"
    operation_chain_work_queue = tmp_path / "operation_chain_work_queue.json"
    operation_chain_execution_queue = tmp_path / "operation_chain_execution_queue.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/harvest_frontier_targeted_component_correction_response.py",
            "--work-order",
            str(work_order_path),
            "--local-cpu-advisory",
            str(candidate_path),
            "--reference-local-cpu-advisory",
            str(reference_path),
            "--output",
            str(output),
            "--materialization-requests-output",
            str(materialization_requests),
            "--materialization-queue-output",
            str(materialization_queue),
            "--materialization-queue-id",
            "targeted_component_materialization_cli_unit",
            "--operation-chain-work-orders-output",
            str(operation_chain_work_orders),
            "--operation-chain-queue-output",
            str(operation_chain_queue),
            "--operation-chain-queue-id",
            "targeted_component_operation_chain_cli_unit",
            "--operation-chain-materializer-handoff-output",
            str(operation_chain_handoff),
            "--operation-chain-materializer-work-queue-output",
            str(operation_chain_work_queue),
            "--operation-chain-materializer-execution-queue-output",
            str(operation_chain_execution_queue),
            "--operation-chain-materializer-execution-queue-id",
            "targeted_component_chain_materializer_execution_cli_unit",
            "--results-root",
            str(tmp_path / "results"),
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["materialization_request_count"] == 1
    assert stdout["materialization_queue_experiment_count"] == 1
    assert stdout["operation_chain_work_order_count"] == 1
    assert stdout["operation_chain_queue_experiment_count"] == 1
    assert stdout["operation_chain_materializer_handoff_work_rows"] >= 1
    assert stdout["operation_chain_materializer_work_queue_executable_rows"] >= 1
    assert (
        stdout["operation_chain_materializer_execution_queue_experiment_count"]
        == 1
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA
    _assert_false_authority(payload)
    assert payload["row_count"] == 1
    assert payload["local_acquisition_recommended_count"] == 1
    row = payload["rows"][0]
    assert row["local_cpu_component_terms"]["segnet_delta_score_units"] == pytest.approx(
        -0.02
    )
    assert row["reference_local_cpu_advisory_path"].endswith(
        "reference_local_cpu_advisory.json"
    )
    requests = json.loads(materialization_requests.read_text(encoding="utf-8"))
    assert (
        requests["schema"]
        == TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA
    )
    assert requests["row_count"] == 1
    _assert_false_authority(requests)
    request = requests["rows"][0]
    assert (
        request["schema"]
        == TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA
    )
    assert request["ready_for_materializer_execution"] is False
    queue = json.loads(materialization_queue.read_text(encoding="utf-8"))
    assert len(queue["experiments"]) == 1
    assert queue["experiments"][0]["metadata"]["ready_for_budget_spend"] is False
    assert queue["materialization_request_summary"]["score_claim"] is False
    chain_work_orders = json.loads(
        operation_chain_work_orders.read_text(encoding="utf-8")
    )
    assert chain_work_orders["schema"] == OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA
    assert chain_work_orders["work_order_count"] == 1
    _assert_false_authority(chain_work_orders)
    chain_budget = chain_work_orders["work_orders"][0]["targeted_correction_budget"]
    _assert_false_authority(chain_budget)
    assert chain_budget["best_local_cpu_score_delta_summary"][
        "receiver_closed_rate_delta_score_units"
    ] == pytest.approx(-0.0001717916099055202)
    _assert_false_authority(chain_budget["best_local_cpu_score_delta_summary"])
    assert chain_budget["paired_delta_basis"][0]["local_cpu_score_delta_summary"][
        "receiver_closed_total_delta_score_units"
    ] == pytest.approx(-0.020171791609905523)
    _assert_false_authority(
        chain_budget["paired_delta_basis"][0]["local_cpu_score_delta_summary"]
    )
    chain_queue = json.loads(operation_chain_queue.read_text(encoding="utf-8"))
    assert len(chain_queue["experiments"]) == 1
    assert chain_queue["experiments"][0]["metadata"]["execution_ready"] is False
    assert chain_queue["experiments"][0]["metadata"]["budget_spend_allowed"] is False
    handoff = json.loads(operation_chain_handoff.read_text(encoding="utf-8"))
    assert (
        handoff["schema"]
        == TARGETED_COMPONENT_CORRECTION_CHAIN_MATERIALIZER_HANDOFF_SCHEMA
    )
    _assert_false_authority(handoff)
    assert "packet_member_merge_v1" in handoff["registered_chain_targets"]
    handoff_budget = handoff["materializer_backlog"]["rows"][0][
        "frontier_operation_portfolio_row"
    ]["evidence_summary"]["targeted_correction_budget"]
    assert handoff_budget["best_local_cpu_score_delta_summary"][
        "receiver_closed_rate_delta_score_units"
    ] == pytest.approx(-0.0001717916099055202)
    _assert_false_authority(handoff_budget["best_local_cpu_score_delta_summary"])
    params_budget = handoff["materializer_backlog"]["rows"][0]["operation_params"][
        "targeted_correction_budget"
    ]
    assert params_budget["paired_delta_basis"][0]["local_cpu_score_delta_summary"][
        "receiver_closed_archive_byte_delta_vs_reference"
    ] == -258
    work_queue = json.loads(operation_chain_work_queue.read_text(encoding="utf-8"))
    assert work_queue["schema"] == "byte_shaving_materializer_work_queue.v1"
    _assert_false_authority(work_queue)
    assert work_queue["executable_row_count"] >= 1
    execution_queue = json.loads(
        operation_chain_execution_queue.read_text(encoding="utf-8")
    )
    assert execution_queue["schema"] == "experiment_queue.v1"
    _assert_false_authority(execution_queue)
    assert len(execution_queue["experiments"]) == 1


def test_targeted_component_response_harvest_expands_grouped_request_metadata(
    tmp_path: Path,
) -> None:
    queue = {
        "schema": "unit_targeted_component_correction_queue.v1",
        "experiments": [
            {
                "metadata": {
                    "candidate_id": "candidate_grouped",
                    "saved_bytes_budget": 258,
                    "estimated_rate_credit_score_units": 0.00000258,
                    "correction_requests": [
                        {
                            "acquisition_id": "request_region",
                            "candidate_id": "candidate_grouped",
                            "correction_family": (
                                "segnet_posenet_waterfill_region_repair"
                            ),
                            "operation_levels": [
                                "pixel",
                                "region",
                                "boundary",
                                "frame",
                            ],
                            "targeted_dimensions": ["region", "boundary"],
                            "work_order_path": "missing/request_region_work_order.json",
                            "local_cpu_advisory_path": (
                                "missing/request_region_local_cpu.json"
                            ),
                            "component_correction_response_harvest_path": (
                                "missing/request_region_response.json"
                            ),
                        },
                        {
                            "acquisition_id": "request_batch",
                            "candidate_id": "candidate_grouped",
                            "correction_family": (
                                "full_video_batch_residual_budget_reallocation"
                            ),
                            "operation_levels": ["pair", "batch", "full_video"],
                            "targeted_dimensions": ["pair", "batch", "full_video"],
                            "work_order_path": "missing/request_batch_work_order.json",
                            "local_cpu_advisory_path": (
                                "missing/request_batch_local_cpu.json"
                            ),
                            "component_correction_response_harvest_path": (
                                "missing/request_batch_response.json"
                            ),
                        },
                    ],
                    **_false_authority(),
                },
            }
        ],
        **_false_authority(),
    }

    harvest = build_frontier_targeted_component_correction_response_harvest(
        repo_root=tmp_path,
        targeted_component_correction_queue=queue,
    )

    assert harvest["schema"] == TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA
    _assert_false_authority(harvest)
    assert harvest["row_count"] == 2
    assert harvest["ready_for_budget_spend_count"] == 0
    assert {row["acquisition_id"] for row in harvest["rows"]} == {
        "request_region",
        "request_batch",
    }
    assert {row["correction_family"] for row in harvest["rows"]} == {
        "segnet_posenet_waterfill_region_repair",
        "full_video_batch_residual_budget_reallocation",
    }
    assert all(
        "targeted_component_correction_response_artifacts_missing"
        in row["budget_spend_blockers"]
        for row in harvest["rows"]
    )
    grouped_levels = {
        level for row in harvest["rows"] for level in row["operation_levels"]
    }
    assert {"pixel", "region", "boundary", "pair", "batch", "full_video"}.issubset(
        grouped_levels
    )


def test_targeted_component_correction_materialization_requests_group_responses(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_materialization_request_unit",
        candidate_limit=3,
    )
    targeted_component = report["targeted_component_correction_acquisition"]
    work_orders = [
        build_frontier_targeted_component_correction_work_order(
            targeted_component_correction_acquisition=targeted_component,
            acquisition_id=acquisition_id,
        )
        for acquisition_id in targeted_component["top_acquisition_ids"][:2]
    ]
    local_cpu_advisory = {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "component_deltas": {
            "segnet_delta": -0.00012,
            "posenet_delta": 0.00001,
            "archive_byte_delta_vs_receiver_closed_candidate": 12,
        },
    }
    response_rows = [
        build_frontier_targeted_component_correction_response_harvest_from_artifacts(
            work_order=work_order,
            local_cpu_advisory=local_cpu_advisory,
            work_order_path=f"work_order_{index}.json",
            local_cpu_advisory_path=f"local_cpu_advisory_{index}.json",
            response_artifact_path=f"component_correction_response_harvest_{index}.json",
        )
        for index, work_order in enumerate(work_orders, start=1)
    ]
    harvest = build_frontier_targeted_component_correction_response_harvest(
        repo_root=tmp_path,
        response_rows=response_rows,
    )

    requests = build_frontier_targeted_component_correction_materialization_requests(
        targeted_component_correction_response_harvest=harvest,
        candidate_limit=1,
    )

    assert (
        requests["schema"]
        == TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA
    )
    _assert_false_authority(requests)
    assert requests["accepted_response_count"] == 2
    assert requests["row_count"] == 1
    assert requests["ready_for_budget_spend_count"] == 0
    request = requests["rows"][0]
    assert (
        request["schema"]
        == TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA
    )
    _assert_false_authority(request)
    assert request["accepted_response_count"] == 2
    assert set(request["accepted_correction_families"]) == {
        row["correction_family"] for row in response_rows
    }
    assert {"pixel", "region", "boundary", "frame", "pair"}.issubset(
        set(request["operation_levels"])
    )
    assert "receiver_consumed_correction_materializer_missing" in request[
        "budget_spend_blockers"
    ]
    assert request["ready_for_materializer_execution"] is False
    assert request["ready_for_budget_spend"] is False
    assert request["budget_spend_allowed"] is False
    assert request["receiver_materialization_contract"][
        "parser_only_or_planner_only_signal_is_insufficient"
    ] is True

    selected = build_frontier_targeted_component_correction_materialization_request(
        targeted_component_correction_response_harvest=harvest,
        materialization_request_id=request["materialization_request_id"],
        candidate_limit=1,
    )
    assert selected["materialization_request_id"] == request["materialization_request_id"]
    _assert_false_authority(selected)

    harvest_path = tmp_path / "targeted_component_correction_response_harvest.json"
    harvest_path.write_text(json.dumps(harvest), encoding="utf-8")
    queue = build_frontier_targeted_component_correction_materialization_queue(
        repo_root=REPO_ROOT,
        targeted_component_correction_response_harvest=harvest,
        targeted_component_correction_response_harvest_path=harvest_path,
        results_root=tmp_path / "results",
        queue_id="frontier_feedback_materialization_request_unit",
        candidate_limit=1,
    )
    assert queue is not None
    assert len(queue["experiments"]) == 1
    step = queue["experiments"][0]["steps"][0]
    assert (
        step["command"][1]
        == "tools/build_frontier_targeted_component_correction_materialization_request.py"
    )
    result = subprocess.run(
        [sys.executable, *step["command"][1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["bytes_written"] > 0
    request_path = Path(
        step["command"][step["command"].index("--request-out") + 1]
    )
    materialized_request = json.loads(request_path.read_text(encoding="utf-8"))
    assert (
        materialized_request["schema"]
        == TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUEST_ROW_SCHEMA
    )
    _assert_false_authority(materialized_request)

    chain_work_orders = (
        build_frontier_targeted_component_correction_chain_work_orders(
            targeted_component_correction_materialization_requests=requests,
            request_limit=1,
        )
    )
    assert chain_work_orders["schema"] == OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA
    _assert_false_authority(chain_work_orders)
    assert chain_work_orders["work_order_count"] == 1
    chain_work_order = chain_work_orders["work_orders"][0]
    _assert_false_authority(chain_work_order)
    assert chain_work_order["source_materialization_request_id"] == request[
        "materialization_request_id"
    ]
    runtime_binding = chain_work_order["targeted_correction_budget"][
        "receiver_runtime_binding_context"
    ]
    assert runtime_binding["binding_complete_for_component_eval"] is True
    assert runtime_binding["candidate_archive_path"]
    assert chain_work_order["targeted_correction_budget"]["candidate_archive_path"]
    assert "byte_range_entropy_recode_v1" in chain_work_order["chain_targets"]
    assert "packet_member_merge_v1" in chain_work_order["chain_targets"]
    assert "segnet_component_response" in chain_work_order["chain_targets"]
    assert {
        stage["stage"] for stage in chain_work_order["stage_plan"]
    } == {
        "scorer_sensitive_operation_selection",
        "receiver_consumed_correction_synthesis",
        "payload_grammar_and_entropy",
        "component_guarded_budget_replay",
    }
    assert (
        "single_composed_receiver_runtime_consumption_proof"
        in chain_work_order["required_before_execution"]
    )

    chain_work_orders_path = tmp_path / "component_chain_work_orders.json"
    chain_work_orders_path.write_text(json.dumps(chain_work_orders), encoding="utf-8")
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row(candidate_id="pairset_drop_many_k012_h1ecc99d178")],
    )
    chain_queue = build_frontier_operation_chain_compiler_queue(
        repo_root=REPO_ROOT,
        operation_chain_compiler_work_orders=chain_work_orders,
        operation_chain_compiler_work_orders_path=chain_work_orders_path,
        results_root=tmp_path / "results",
        queue_id="frontier_feedback_targeted_component_chain_unit",
        candidate_limit=1,
        dqs1_observation_source_paths=(dqs1_observations,),
    )
    assert chain_queue is not None
    _assert_false_authority(chain_queue["experiments"][0]["metadata"])
    assert len(chain_queue["experiments"]) == 1
    assert chain_queue["experiments"][0]["steps"][0]["command"][1] == (
        "tools/build_frontier_operation_chain_stage_plan.py"
    )
    assert isinstance(
        chain_queue["experiments"][0]["metadata"][
            "byte_range_local_chain_queueable"
        ],
        bool,
    )
    assert chain_queue["experiments"][0]["metadata"][
        "byte_range_local_chain_queueable"
    ] is False
    assert chain_queue["experiments"][0]["metadata"]["chain_target_count"] >= 6
    chain_stage_plan = build_frontier_operation_chain_compiler_stage_plan(
        operation_chain_compiler_work_orders=chain_work_orders,
        source_operation_id=chain_work_order["source_operation_id"],
    )
    byte_range_inputs = build_frontier_byte_range_stage_inputs(
        repo_root=REPO_ROOT,
        operation_chain_stage_plan=chain_stage_plan,
        chain_output_dir=tmp_path / "targeted_byte_range_chain",
    )
    assert byte_range_inputs["local_chain_queueable"] is False
    assert byte_range_inputs["materializer_context"][
        "default_pr103_context_disabled"
    ] is True
    assert (
        "byte_range_stage_default_pr103_context_disabled_for_target_bound_chain"
        in byte_range_inputs["materializer_context"]["context_blockers"]
    )
    assert "public_pr103_intake" not in byte_range_inputs["materializer_context"][
        "source_archive"
    ]
    assert len(chain_queue["experiments"][0]["steps"]) == 6
    assert {
        step["id"] for step in chain_queue["experiments"][0]["steps"]
    } == {
        "emit_operation_chain_stage_plan",
        "emit_byte_range_stage_inputs",
        "emit_targeted_drop_many_stage_inputs",
        "run_targeted_drop_many_pairset_acquisition",
        "build_targeted_drop_many_dqs1_followup_queue",
        "validate_targeted_drop_many_dqs1_followup_queue",
    }
    targeted_drop_many_step = next(
        step
        for step in chain_queue["experiments"][0]["steps"]
        if step["id"] == "emit_targeted_drop_many_stage_inputs"
    )
    assert targeted_drop_many_step["requires"] == ["emit_operation_chain_stage_plan"]
    targeted_drop_many_run = next(
        step
        for step in chain_queue["experiments"][0]["steps"]
        if step["id"] == "run_targeted_drop_many_pairset_acquisition"
    )
    assert targeted_drop_many_run["requires"] == [
        "emit_targeted_drop_many_stage_inputs"
    ]
    targeted_dqs1_build = next(
        step
        for step in chain_queue["experiments"][0]["steps"]
        if step["id"] == "build_targeted_drop_many_dqs1_followup_queue"
    )
    assert targeted_dqs1_build["requires"] == [
        "run_targeted_drop_many_pairset_acquisition"
    ]
    assert "--pairset-acquisition" in targeted_dqs1_build["command"]
    selector_kind_indices = [
        index
        for index, value in enumerate(targeted_dqs1_build["command"])
        if value == "--selector-kind"
    ]
    assert [
        targeted_dqs1_build["command"][index + 1]
        for index in selector_kind_indices
    ] == [
        "drop_many_beam_pairwise_interaction_waterfill",
        "pair_frame_geometry_low_impact_drop_many",
    ]
    assert "--selected-pairset-acquisition-out" in targeted_dqs1_build["command"]
    assert "--materializer-feedback-bridge-out" in targeted_dqs1_build["command"]
    assert "--eureka-run-id" in targeted_dqs1_build["command"]
    assert targeted_dqs1_build["command"][
        targeted_dqs1_build["command"].index("--eureka-run-id") + 1
    ].endswith("targeted_drop_many_dqs1_followup")
    observation_indices = [
        index
        for index, value in enumerate(targeted_dqs1_build["command"])
        if value == "--dqs1-observation-jsonl"
    ]
    assert [targeted_dqs1_build["command"][index + 1] for index in observation_indices] == [
        dqs1_observations.as_posix()
    ]
    targeted_dqs1_validate = next(
        step
        for step in chain_queue["experiments"][0]["steps"]
        if step["id"] == "validate_targeted_drop_many_dqs1_followup_queue"
    )
    assert targeted_dqs1_validate["requires"] == [
        "build_targeted_drop_many_dqs1_followup_queue"
    ]
    assert chain_queue["experiments"][0]["metadata"][
        "targeted_drop_many_local_plan_queueable"
    ] is True
    assert chain_queue["experiments"][0]["metadata"][
        "targeted_drop_many_dqs1_followup_queue_enabled"
    ] is True
    assert chain_queue["experiments"][0]["metadata"][
        "targeted_drop_many_dqs1_followup_queue_path"
    ].endswith("targeted_drop_many_dqs1_followup_queue.json")
    assert chain_queue["experiments"][0]["metadata"][
        "targeted_drop_many_dqs1_selector_kind_allowlist"
    ] == [
        "drop_many_beam_pairwise_interaction_waterfill",
        "pair_frame_geometry_low_impact_drop_many",
    ]
    assert chain_queue["experiments"][0]["metadata"][
        "targeted_drop_many_dqs1_observation_source_paths"
    ] == [dqs1_observations.as_posix()]
    assert "drop_within_selected_set_masked_boundary" in chain_queue["experiments"][0][
        "metadata"
    ]["targeted_drop_many_selected_family_targets"]
    handoff = build_frontier_targeted_component_correction_chain_materializer_handoff(
        repo_root=REPO_ROOT,
        targeted_component_correction_chain_work_orders=chain_work_orders,
        default_output_root=tmp_path / "chain_materializers",
    )
    assert (
        handoff["schema"]
        == TARGETED_COMPONENT_CORRECTION_CHAIN_MATERIALIZER_HANDOFF_SCHEMA
    )
    _assert_false_authority(handoff)
    assert "packet_member_merge_v1" in handoff["registered_chain_targets"]
    assert "packet_member_zip_header_elide_v1" in handoff[
        "registered_chain_targets"
    ]
    assert {
        "archive_section_header_elide_v1",
        "packet_member_reorder_v1",
        "tensor_quantize_v1",
        "tensor_prune_v1",
        "tensor_shared_codebook_v1",
        "inverse_steganalysis_high_level_operation_set_v1",
    }.issubset(set(handoff["registered_chain_targets"]))
    assert "segnet_component_response" in handoff["unregistered_chain_targets"]
    assert handoff["work_queue_row_count"] == handoff[
        "registered_chain_target_count"
    ]
    assert handoff["context_closure_plan_count"] == handoff[
        "registered_chain_target_count"
    ]
    assert handoff["materializer_work_queue"]["schema"] == (
        "byte_shaving_materializer_work_queue.v1"
    )
    assert handoff["materializer_work_queue"]["blocked_row_count"] >= 1
    context_closure_by_target = {
        plan["target_kind"]: plan for plan in handoff["context_closure_plans"]
    }
    packet_merge_plan = context_closure_by_target["packet_member_merge_v1"]
    _assert_false_authority(packet_merge_plan)
    assert packet_merge_plan["schema"] == (
        "frontier_rate_attack_targeted_component_chain_materializer_context_closure_plan.v1"
    )
    assert "merge_contract" in packet_merge_plan["missing_context_fields"]
    assert "packet_member_merge_source_runtime_dir" in packet_merge_plan[
        "provided_context_fields"
    ]
    assert packet_merge_plan["receiver_proof_request"][
        "parser_only_proof_rejected"
    ] is True
    dfl1_plan = context_closure_by_target["renderer_payload_dfl1_v1"]
    assert dfl1_plan["missing_context_fields"] == []
    if dfl1_plan["context_blockers"]:
        assert dfl1_plan["ready_for_materializer_execution"] is False
        assert any(
            blocker.startswith(
                "renderer_payload_dfl1_source_archive_missing_required_members"
            )
            for blocker in dfl1_plan["context_blockers"]
        )
    else:
        assert dfl1_plan["ready_for_materializer_execution"] is True
    assert {
        "renderer_payload_dfl1_full_frame_file_list_or_entries",
        "renderer_payload_dfl1_expected_full_frame_file_list_sha256",
        "renderer_payload_dfl1_expected_full_frame_entry_count",
        "renderer_payload_dfl1_full_frame_file_list_source",
    }.issubset(set(dfl1_plan["provided_context_fields"]))
    tensor_quantize_plan = context_closure_by_target["tensor_quantize_v1"]
    assert "quantization_contract" in tensor_quantize_plan[
        "missing_context_fields"
    ]
    assert "runtime_consumption_proof" in tensor_quantize_plan[
        "missing_context_fields"
    ]
    reorder_plan = context_closure_by_target["packet_member_reorder_v1"]
    assert "member_order_contract" in reorder_plan["missing_context_fields"]
    packet_merge_rows = [
        row
        for row in handoff["materializer_work_queue"]["rows"]
        if row["target_kind"] == "packet_member_merge_v1"
    ]
    assert packet_merge_rows
    assert packet_merge_rows[0]["materializer_context_closure_plan"][
        "target_kind"
    ] == "packet_member_merge_v1"
    assert "materializer_context_missing:merge_contract" in packet_merge_rows[0][
        "materialization_blockers"
    ]
    dfl1_rows = [
        row
        for row in handoff["materializer_work_queue"]["rows"]
        if row["target_kind"] == "renderer_payload_dfl1_v1"
    ]
    assert dfl1_rows
    assert dfl1_rows[0]["executable"] is (not bool(dfl1_plan["context_blockers"]))
    assert dfl1_rows[0]["renderer_payload_dfl1_parity_context"][
        "file_list"
    ] == "upstream/public_test_video_names.txt"
    assert dfl1_rows[0]["renderer_payload_dfl1_parity_context"][
        "expected_full_frame_entry_count"
    ] == 1
    assert len(
        dfl1_rows[0]["renderer_payload_dfl1_parity_context"][
            "expected_full_frame_file_list_sha256"
        ]
    ) == 64
    packet_reorder_rows = [
        row
        for row in handoff["materializer_work_queue"]["rows"]
        if row["target_kind"] == "packet_member_reorder_v1"
    ]
    assert packet_reorder_rows
    assert packet_reorder_rows[0]["telemetry"]["receiver_contract_work_order"][
        "schema"
    ] == "packet_member_receiver_contract_work_order.v1"
    tensor_quantize_rows = [
        row
        for row in handoff["materializer_work_queue"]["rows"]
        if row["target_kind"] == "tensor_quantize_v1"
    ]
    assert tensor_quantize_rows
    assert tensor_quantize_rows[0]["telemetry"]["receiver_contract_work_order"][
        "schema"
    ] == "tensor_receiver_contract_work_order.v1"
    bridge_only_autonomous = build_frontier_autonomous_chain_optimization(
        operation_portfolio=report["operation_portfolio"],
        operation_materializer_bridge=report["operation_materializer_bridge"],
        chain_limit=3,
    )
    assert bridge_only_autonomous["registered_target_count"] == 0
    assert all(
        action["id"] != "bind_targeted_chain_materializer_contexts"
        for row in bridge_only_autonomous["rows"]
        for action in row["scheduler_actions"]
    )
    autonomous = build_frontier_autonomous_chain_optimization(
        operation_portfolio=report["operation_portfolio"],
        operation_materializer_bridge=report["operation_materializer_bridge"],
        targeted_component_correction_chain_materializer_handoff=handoff,
        chain_limit=3,
    )
    assert autonomous["schema"] == AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA
    _assert_false_authority(autonomous)
    assert autonomous["chain_count"] >= 2
    assert "global_many_op_rate_distortion_receiver_campaign" in autonomous[
        "top_chain_ids"
    ]
    assert {"packet_member", "archive_section", "tensor"}.issubset(
        set(autonomous["target_classes"])
    )
    first_chain = autonomous["rows"][0]
    assert first_chain["schema"] == AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA
    _assert_false_authority(first_chain)
    assert first_chain["registered_chain_target_count"] == handoff[
        "registered_chain_target_count"
    ]
    assert {"pixel", "region", "boundary", "frame", "pair", "batch"}.issubset(
        set(first_chain["operation_levels"])
    )
    assert first_chain["materializer_target_count"] >= 6
    assert first_chain["repair_budget_waterfill_plan"]["allocator"] == (
        "measured_component_marginal_waterfill_over_segnet_posenet_rate_budget"
    )
    assert first_chain["repair_budget_waterfill_plan"]["component_axes"] == [
        "segnet",
        "posenet",
    ]
    assert first_chain["repair_budget_waterfill_plan"]["operator_action_ledger"][
        "schema"
    ] == OPERATOR_ACTION_LEDGER_SCHEMA
    assert first_chain["repair_budget_waterfill_plan"]["waterfill_policy"][
        "repair_allocation_action_term_schema"
    ] == REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
    assert "fit_segnet_posenet_repair_waterfill_policy" in {
        action["id"] for action in first_chain["scheduler_actions"]
    }
    bind_actions = [
        action
        for row in autonomous["rows"]
        for action in row["scheduler_actions"]
        if action["id"] == "bind_targeted_chain_materializer_contexts"
    ]
    assert bind_actions
    assert all(
        action["target_count"] == handoff["registered_chain_target_count"]
        for action in bind_actions
    )
    emitted_artifact_keys = {
        "operation_materializer_execution_queue",
        "repair_budget_waterfill_queue",
        "receiver_repair_queue",
        "targeted_component_correction_chain_materializer_execution_queue",
        "targeted_component_correction_operation_chain_queue",
    }
    emitted_source_keys = {
        "operation_materializer_work_queue",
        "targeted_component_correction_chain_materializer_work_queue",
        "materializer_chain_exact_readiness_bridge",
    }
    for row in autonomous["rows"]:
        for action in row["scheduler_actions"]:
            if action.get("advisory_only"):
                assert "queue_artifact_key" not in action
                assert action["source_artifact_key"] in emitted_source_keys
            else:
                assert action["queue_artifact_key"] in emitted_artifact_keys
    work_order = build_frontier_autonomous_chain_work_order(
        autonomous_chain_optimization=autonomous,
        chain_id=first_chain["chain_id"],
    )
    assert work_order["schema"] == AUTONOMOUS_CHAIN_WORK_ORDER_SCHEMA
    _assert_false_authority(work_order)
    assert work_order["pipeline_placement"]["rate_attack_owner"] == (
        "encoder_materializer_and_archive_builder"
    )
    assert work_order["pipeline_placement"]["receiver_owner"] == (
        "deterministic_inflate_runtime_adapter_only"
    )
    assert {"encoder_materializer", "encoder_repair_allocator"}.issubset(
        {stage["pipeline_side"] for stage in work_order["pipeline_stages"]}
    )
    assert "receiver_decode_only" in {
        stage["pipeline_side"] for stage in work_order["pipeline_stages"]
    }
    assert work_order["local_queue_action_count"] >= 1
    assert work_order["advisory_action_count"] >= 1
    assert work_order["repair_budget_waterfill_plan"]["budget_spend_allowed"] is False

    autonomous_path = _write_json(tmp_path / "autonomous_chain.json", autonomous)
    receiver_budget_path = _write_json(
        tmp_path / "receiver_closed_correction_budget.json",
        report["receiver_closed_correction_budget"],
    )
    repair_waterfill_work_order = build_frontier_repair_budget_waterfill_work_order(
        autonomous_chain_optimization=autonomous,
        chain_id=first_chain["chain_id"],
        targeted_component_correction_response_harvest=harvest,
        receiver_closed_correction_budget=report["receiver_closed_correction_budget"],
        autonomous_chain_optimization_path=autonomous_path,
        targeted_component_correction_response_harvest_path=harvest_path,
        receiver_closed_correction_budget_path=receiver_budget_path,
    )
    assert (
        repair_waterfill_work_order["schema"]
        == REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA
    )
    _assert_false_authority(repair_waterfill_work_order)
    assert repair_waterfill_work_order["pipeline_side"] == "encoder_repair_allocator"
    assert repair_waterfill_work_order["rate_budget_preservation_plan"]["schema"] == (
        RATE_BUDGET_PRESERVATION_PLAN_SCHEMA
    )
    assert repair_waterfill_work_order["operator_action_ledger"]["schema"] == (
        OPERATOR_ACTION_LEDGER_SCHEMA
    )
    assert repair_waterfill_work_order["action_functional_lineage"][
        "upstream_rate_budget_preservation_schema"
    ] == RATE_BUDGET_PRESERVATION_PLAN_SCHEMA
    assert repair_waterfill_work_order["action_functional_lineage"][
        "upstream_operator_action_ledger_schema"
    ] == OPERATOR_ACTION_LEDGER_SCHEMA
    assert repair_waterfill_work_order["action_functional_lineage"][
        "repair_allocation_action_term_schema"
    ] == REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
    assert repair_waterfill_work_order["action_functional_lineage"][
        "upstream_action_functional_schema"
    ] == "frontier_rate_attack_operator_action_functional.v1"
    assert repair_waterfill_work_order["action_functional_lineage"][
        "new_parallel_action_functional_created"
    ] is False
    assert repair_waterfill_work_order["preservation_contract"][
        "emit_rate_only_floor_archive_before_repair_archive"
    ] is True
    assert repair_waterfill_work_order["accepted_response_count"] == 2
    assert repair_waterfill_work_order["allocation_row_count"] == 2
    first_allocation = repair_waterfill_work_order["allocation_rows"][0]
    assert first_allocation["allocation_action_term"]["schema"] == (
        REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
    )
    assert first_allocation["allocation_action_term"]["T_i"][
        "allocated_repair_bytes"
    ] == first_allocation["proposed_encoder_repair_bytes"]
    assert repair_waterfill_work_order["budget_spend_allowed"] is False
    assert repair_waterfill_work_order["ready_for_exact_eval_dispatch"] is False
    assert "exact_axis_component_response_required_before_budget_spend" in (
        repair_waterfill_work_order["blockers"]
    )
    materialization_plan = build_frontier_repair_budget_materialization_plan(
        repair_budget_waterfill_work_order=repair_waterfill_work_order,
        repair_budget_waterfill_work_order_path=tmp_path
        / "repair_budget_waterfill_work_order.json",
    )
    assert materialization_plan["schema"] == REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA
    _assert_false_authority(materialization_plan)
    assert materialization_plan["candidate_archive_materialized"] is False
    assert materialization_plan["rate_only_floor_preserved_before_repair_spend"] is True
    assert materialization_plan["spent_budget_candidates_are_children_of_rate_only_floor"]
    parent_row = materialization_plan["candidate_chain_rows"][0]
    assert parent_row["schema"] == REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA
    assert parent_row["candidate_kind"] == "rate_only_floor_parent"
    assert parent_row["operator_action_ledger_schema"] == OPERATOR_ACTION_LEDGER_SCHEMA
    assert parent_row["operator_action_terms"][0]["schema"] == OPERATOR_ACTION_TERM_SCHEMA
    assert parent_row["candidate_archive_materialized"] is False
    child_rows = [
        row
        for row in materialization_plan["candidate_chain_rows"][1:]
        if row["candidate_kind"] == "spent_budget_repair_child"
    ]
    cascade_rows = [
        row
        for row in materialization_plan["candidate_chain_rows"][1:]
        if row["candidate_kind"] == "structural_repair_cascade_probe"
    ]
    assert len(child_rows) == repair_waterfill_work_order["allocation_row_count"]
    assert materialization_plan["structural_repair_cascade_candidate_count"] == 1
    assert cascade_rows[0]["cascade_id"] == (
        "cascade_c_posenet_null_segnet_region_selector_codec"
    )
    assert cascade_rows[0]["source_relation"] == "PR110-OPT-5+7+10+12_UNTOUCHED"
    cascade_mechanisms = {
        item["mechanism_id"]
        for item in cascade_rows[0]["cascade_opportunity"]["canonical_mechanisms"]
    }
    assert {
        "uniward_textured_region_undetectability",
        "detector_informed_embedding",
        "square_root_law_capacity",
        "cnn_blind_spot_texture_and_dct_statistics",
    }.issubset(cascade_mechanisms)
    assert "segnet_logit_margin_or_detector_margin" in cascade_rows[0][
        "cascade_opportunity"
    ]["required_probe_measurements"]
    assert all(
        row["parent_candidate_chain_id"] == parent_row["candidate_chain_id"]
        for row in [*child_rows, *cascade_rows]
    )
    assert all(row["candidate_kind"] == "spent_budget_repair_child" for row in child_rows)
    assert all(row["budget_spend_allowed"] is False for row in child_rows)
    assert child_rows[0]["allocation_action_term"]["schema"] == (
        REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
    )
    binding_report = build_frontier_repair_budget_materializer_binding_report(
        repo_root=REPO_ROOT,
        repair_budget_materialization_plan=materialization_plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
    )
    assert binding_report["schema"] == REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA
    assert binding_report["candidate_archive_materialized"] is False
    assert binding_report["ready_for_exact_eval_dispatch"] is False
    _assert_false_authority(binding_report)
    execution_report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=materialization_plan,
        repair_budget_materialization_plan_path=tmp_path
        / "repair_budget_materialization_plan.json",
        materializer_binding_report=binding_report,
        materializer_binding_report_path=tmp_path
        / "repair_budget_materializer_binding_report.json",
    )
    assert execution_report["schema"] == (
        REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    )
    assert execution_report["execution_rows"][0]["schema"] == (
        REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA
    )
    assert execution_report["ready_for_exact_eval_dispatch"] is False
    repair_dynamics_prior = {
        "schema": "frontier_rate_attack_repair_dynamics_palette_prior.v1",
        **_false_authority(),
        "source": "unit_test_repair_waterfill_palette",
        "palette_modes": list(FEC6_FIXED_K16_MODE_IDS),
    }
    repair_waterfill_queue = build_frontier_repair_budget_waterfill_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        targeted_component_correction_response_harvest=harvest,
        targeted_component_correction_response_harvest_path=harvest_path,
        receiver_closed_correction_budget=report["receiver_closed_correction_budget"],
        receiver_closed_correction_budget_path=receiver_budget_path,
        repair_dynamics_palette_prior=repair_dynamics_prior,
        results_root=tmp_path / "results",
        queue_id="frontier_repair_waterfill_unit",
        chain_limit=1,
    )
    assert repair_waterfill_queue is not None
    assert repair_waterfill_queue["schema"] == "experiment_queue.v1"
    repair_waterfill_experiment = repair_waterfill_queue["experiments"][0]
    assert repair_waterfill_experiment["status"] == "queued"
    assert repair_waterfill_experiment["metadata"]["pipeline_side"] == (
        "encoder_repair_allocator"
    )
    assert (
        repair_waterfill_experiment["metadata"]["candidate_archive_materialized"]
        is False
    )
    assert repair_waterfill_experiment["metadata"]["operator_action_ledger_schema"] == (
        OPERATOR_ACTION_LEDGER_SCHEMA
    )
    assert repair_waterfill_experiment["metadata"][
        "repair_allocation_action_term_schema"
    ] == REPAIR_BUDGET_WATERFILL_ALLOCATION_ACTION_TERM_SCHEMA
    assert repair_waterfill_experiment["metadata"]["repair_dynamics_prior_active"] is True
    assert repair_waterfill_experiment["metadata"]["repair_dynamics_palette_prior"][
        "palette_modes"
    ] == list(FEC6_FIXED_K16_MODE_IDS)
    _assert_false_authority(repair_waterfill_experiment["metadata"])
    repair_step = repair_waterfill_experiment["steps"][0]
    assert repair_step["command"][1] == (
        "tools/build_frontier_repair_budget_waterfill_work_order.py"
    )
    result = subprocess.run(
        [sys.executable, *repair_step["command"][1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    repair_payload = json.loads(result.stdout)
    assert repair_payload["budget_spend_allowed"] is False
    repair_work_order_path = Path(
        repair_step["command"][repair_step["command"].index("--work-order-out") + 1]
    )
    repair_materialized = json.loads(
        (REPO_ROOT / repair_work_order_path).read_text(encoding="utf-8")
    )
    assert repair_materialized["schema"] == REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA
    _assert_false_authority(repair_materialized)
    materialization_step = repair_waterfill_experiment["steps"][1]
    assert materialization_step["command"][1] == (
        "tools/build_frontier_repair_budget_materialization_plan.py"
    )
    result = subprocess.run(
        [sys.executable, *materialization_step["command"][1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    materialization_payload = json.loads(result.stdout)
    assert materialization_payload["candidate_archive_materialized"] is False
    materialization_plan_path = Path(
        materialization_step["command"][
            materialization_step["command"].index("--materialization-plan-out") + 1
        ]
    )
    materialization_materialized = json.loads(
        (REPO_ROOT / materialization_plan_path).read_text(encoding="utf-8")
    )
    assert (
        materialization_materialized["schema"]
        == REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA
    )
    assert materialization_materialized["parent_candidate_chain_id"] == (
        parent_row["candidate_chain_id"]
    )
    assert materialization_materialized["candidate_archive_materialized"] is False
    _assert_false_authority(materialization_materialized)
    binding_step = repair_waterfill_experiment["steps"][2]
    assert binding_step["command"][1] == (
        "tools/build_frontier_repair_budget_materializer_binding_report.py"
    )
    assert binding_step["command"].count("--repair-palette-mode") == len(
        FEC6_FIXED_K16_MODE_IDS
    )
    result = subprocess.run(
        [sys.executable, *binding_step["command"][1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    binding_payload = json.loads(result.stdout)
    assert binding_payload["candidate_archive_materialized"] is False
    binding_report_path = Path(
        binding_step["command"][
            binding_step["command"].index("--binding-report-out") + 1
        ]
    )
    binding_materialized = json.loads(
        (REPO_ROOT / binding_report_path).read_text(encoding="utf-8")
    )
    assert (
        binding_materialized["schema"]
        == REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA
    )
    assert binding_materialized["candidate_archive_materialized"] is False
    assert binding_materialized["repair_dynamics_palette_prior"]["mode_count"] == 16
    assert (
        "frame0_palette_modes_are_first_class_repair_operators"
        in binding_materialized["repair_dynamics_palette_prior"]["repair_waterfill_hints"]
    )
    _assert_false_authority(binding_materialized)
    execution_step = repair_waterfill_experiment["steps"][3]
    assert execution_step["command"][1] == (
        "tools/build_frontier_repair_budget_materialization_execution_report.py"
    )
    assert "--materializer-binding-report" in execution_step["command"]
    result = subprocess.run(
        [sys.executable, *execution_step["command"][1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    execution_payload = json.loads(result.stdout)
    assert execution_payload["ready_for_exact_eval_dispatch"] is False
    execution_report_path = Path(
        execution_step["command"][
            execution_step["command"].index("--execution-report-out") + 1
        ]
    )
    execution_materialized = json.loads(
        (REPO_ROOT / execution_report_path).read_text(encoding="utf-8")
    )
    assert (
        execution_materialized["schema"]
        == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    )
    assert execution_materialized["candidate_archive_materialized"] is False
    _assert_false_authority(execution_materialized)
    chain_queue_path = _write_json(
        tmp_path / "targeted_component_correction_operation_chain_queue.json",
        chain_queue,
    )
    repair_waterfill_queue_path = _write_json(
        tmp_path / "repair_budget_waterfill_queue.json",
        repair_waterfill_queue,
    )
    autonomous_queue = build_frontier_autonomous_chain_optimization_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        artifact_paths_by_key={
            "operation_materializer_execution_queue": chain_queue_path,
            "repair_budget_waterfill_queue": repair_waterfill_queue_path,
            "receiver_repair_queue": chain_queue_path,
            "targeted_component_correction_operation_chain_queue": chain_queue_path,
            "targeted_component_correction_chain_materializer_execution_queue": (
                chain_queue_path
            ),
        },
        results_root=tmp_path / "results",
        queue_id="frontier_autonomous_chain_unit",
        chain_limit=2,
    )
    assert autonomous_queue is not None
    assert autonomous_queue["schema"] == "experiment_queue.v1"
    assert len(autonomous_queue["experiments"]) >= 1
    autonomous_experiment = autonomous_queue["experiments"][0]
    assert autonomous_experiment["status"] == "queued"
    _assert_false_authority(autonomous_experiment["metadata"])
    assert autonomous_experiment["metadata"]["queue_actuation_ready"] is True
    assert autonomous_experiment["metadata"]["missing_queue_artifact_keys"] == []
    assert autonomous_experiment["metadata"]["child_queue_artifact_paths"]
    assert isinstance(
        autonomous_experiment["metadata"]["post_repair_refresh_planned"], bool
    )
    assert autonomous_experiment["steps"][0]["command"][1] == (
        "tools/build_frontier_autonomous_chain_work_order.py"
    )
    assert "--child-queue-artifact-path" in autonomous_experiment["steps"][0]["command"]
    assert "encoder_materializer" in {
        stage["pipeline_side"]
        for stage in autonomous_experiment["metadata"]["pipeline_stages"]
    }
    assert any(
        step["id"].startswith("validate_") for step in autonomous_experiment["steps"]
    )
    assert any(
        step["id"].startswith("run_") and step["command"][4] == "run-worker"
        for step in autonomous_experiment["steps"]
    )
    blocked_repair_queue = build_frontier_repair_budget_waterfill_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        results_root=tmp_path / "results",
        queue_id="frontier_repair_waterfill_blocked_unit",
        chain_limit=1,
    )
    assert blocked_repair_queue is not None
    blocked_repair_experiment = blocked_repair_queue["experiments"][0]
    assert blocked_repair_experiment["status"] == "frozen"
    assert blocked_repair_experiment["metadata"]["queue_actuation_ready"] is False
    assert "targeted_component_correction_response_harvest" in (
        blocked_repair_experiment["metadata"]["missing_prerequisite_artifact_keys"]
    )
    _assert_false_authority(blocked_repair_experiment["metadata"])
    blocked_repair_queue_path = _write_json(
        tmp_path / "blocked_repair_budget_waterfill_queue.json",
        blocked_repair_queue,
    )
    blocked_child_autonomous_queue = build_frontier_autonomous_chain_optimization_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        artifact_paths_by_key={
            "operation_materializer_execution_queue": chain_queue_path,
            "repair_budget_waterfill_queue": blocked_repair_queue_path,
            "receiver_repair_queue": chain_queue_path,
            "targeted_component_correction_operation_chain_queue": chain_queue_path,
            "targeted_component_correction_chain_materializer_execution_queue": (
                chain_queue_path
            ),
        },
        results_root=tmp_path / "results",
        queue_id="frontier_autonomous_chain_blocked_child_unit",
        chain_limit=1,
    )
    assert blocked_child_autonomous_queue is not None
    blocked_child_experiment = blocked_child_autonomous_queue["experiments"][0]
    assert blocked_child_experiment["status"] == "frozen"
    blocked_child_metadata = blocked_child_experiment["metadata"]
    assert blocked_child_metadata["missing_queue_artifact_keys"] == []
    assert blocked_child_metadata["blocked_child_queue_artifact_keys"] == [
        "repair_budget_waterfill_queue"
    ]
    assert (
        "child_queue_not_runnable:repair_budget_waterfill_queue"
        in blocked_child_metadata["queue_actuation_blockers"]
    )
    repair_child_health = blocked_child_metadata["child_queue_health_by_key"][
        "repair_budget_waterfill_queue"
    ]
    assert repair_child_health["queued_experiment_count"] == 0
    assert repair_child_health["frozen_experiment_count"] == 1
    assert "child_queue_has_no_queued_experiments" in repair_child_health["blockers"]
    repair_refresh_autonomous = json.loads(json.dumps(autonomous))
    repair_refresh_row = repair_refresh_autonomous["rows"][0]
    repair_refresh_actions = repair_refresh_row["scheduler_actions"]
    if not any(
        action["id"] == "fill_receiver_runtime_proof_requests"
        for action in repair_refresh_actions
    ):
        repair_refresh_actions.append(
            {
                **_false_authority(),
                "id": "fill_receiver_runtime_proof_requests",
                "queue_artifact_key": "receiver_repair_queue",
                "purpose": (
                    "unit_test_receiver_repair_before_targeted_materializer_refresh"
                ),
                "bounded_local_execution": True,
                "advisory_only": False,
                "requires_exact_auth_before_score_claim": True,
            }
        )
    bind_action = next(
        (
            action
            for action in repair_refresh_actions
            if action["id"] == "bind_targeted_chain_materializer_contexts"
        ),
        None,
    )
    if bind_action is None:
        bind_action = {
            **_false_authority(),
            "id": "bind_targeted_chain_materializer_contexts",
            "purpose": "unit_test_blocked_targeted_chain_materializer_refresh",
            "bounded_local_execution": True,
            "requires_exact_auth_before_score_claim": True,
        }
        repair_refresh_actions.append(bind_action)
    bind_action.pop("queue_artifact_key", None)
    bind_action["source_artifact_key"] = (
        "targeted_component_correction_chain_materializer_work_queue"
    )
    bind_action["advisory_only"] = True
    bind_action["advisory_reason"] = "unit test blocked receiver context"
    repair_refresh_queue = build_frontier_autonomous_chain_optimization_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=repair_refresh_autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        artifact_paths_by_key={
            "receiver_repair_queue": chain_queue_path,
            "targeted_component_correction_operation_chain_queue": chain_queue_path,
        },
        results_root=tmp_path / "results",
        queue_id="frontier_autonomous_chain_repair_refresh_unit",
        chain_limit=1,
    )
    assert repair_refresh_queue is not None
    repair_refresh_experiment = repair_refresh_queue["experiments"][0]
    assert repair_refresh_experiment["metadata"]["post_repair_refresh_planned"] is True
    assert any(
        step["id"] == "refresh_after_receiver_repair_for_targeted_materializers"
        for step in repair_refresh_experiment["steps"]
    )
    missing_child_queue = build_frontier_autonomous_chain_optimization_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        artifact_paths_by_key={"receiver_repair_queue": chain_queue_path},
        results_root=tmp_path / "results",
        queue_id="frontier_autonomous_chain_missing_child_unit",
        chain_limit=1,
    )
    assert missing_child_queue is not None
    missing_experiment = missing_child_queue["experiments"][0]
    assert missing_experiment["status"] == "frozen"
    assert missing_experiment["metadata"]["queue_actuation_ready"] is False
    assert missing_experiment["metadata"]["missing_queue_artifact_keys"]
    assert (
        "--missing-queue-artifact-key"
        in missing_experiment["steps"][0]["command"]
    )
    assert first_chain["repair_budget_waterfill_plan"][
        "budget_spend_allowed"
    ] is False
    assert first_chain["ready_for_exact_eval_dispatch"] is False
    assert (
        "many_op_plan_to_component_replay_and_exact_readiness_bridge"
        in first_chain["next_queue_edges"]
    )


def test_targeted_dfl1_binds_member_compatible_source_archive(
    tmp_path: Path,
) -> None:
    upstream_dir = tmp_path / "upstream"
    upstream_dir.mkdir()
    (upstream_dir / "public_test_video_names.txt").write_text(
        "0.mkv\n",
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "source_runtime"
    runtime_dir.mkdir()
    source_archive = tmp_path / "source.zip"
    with zipfile.ZipFile(source_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("renderer.bin", b"renderer")
        archive.writestr("masks.mkv", b"masks")
        archive.writestr("optimized_poses.pt", b"poses")
    merged_candidate = tmp_path / "merged_candidate.zip"
    with zipfile.ZipFile(
        merged_candidate,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("__packet_member_merge_v1.bin", b"merged")
    work_orders = {
        "schema": OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
        **_false_authority(),
        "work_order_count": 1,
        "work_orders": [
            {
                "schema": OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
                **_false_authority(),
                "source_operation_id": "dfl1_source_selection_unit",
                "source_materialization_request_id": "request_dfl1_source_selection",
                "chain_targets": ["renderer_payload_dfl1_v1"],
                "source_bridge_blockers": [],
                "targeted_correction_budget": {
                    "schema": "frontier_rate_attack_targeted_chain_budget.v1",
                    **_false_authority(),
                    "saved_bytes_budget": 1,
                    "source_archive_path": source_archive.as_posix(),
                    "candidate_archive_path": merged_candidate.as_posix(),
                    "source_submission_dir": runtime_dir.as_posix(),
                    "candidate_submission_dir": (tmp_path / "candidate_runtime").as_posix(),
                    "receiver_runtime_binding_context": {
                        "schema": (
                            "frontier_rate_attack_targeted_component_receiver_runtime_binding.v1"
                        ),
                        **_false_authority(),
                        "source_archive_path": source_archive.as_posix(),
                        "candidate_archive_path": merged_candidate.as_posix(),
                        "source_submission_dir": runtime_dir.as_posix(),
                        "candidate_submission_dir": (tmp_path / "candidate_runtime").as_posix(),
                    },
                },
            }
        ],
    }

    handoff = build_frontier_targeted_component_correction_chain_materializer_handoff(
        repo_root=tmp_path,
        targeted_component_correction_chain_work_orders=work_orders,
        default_output_root=tmp_path / "out",
    )

    plan = handoff["context_closure_plans"][0]
    assert plan["target_kind"] == "renderer_payload_dfl1_v1"
    assert plan["missing_context_fields"] == []
    assert plan["context_blockers"] == []
    assert plan["ready_for_materializer_execution"] is True
    params = handoff["materializer_backlog"]["rows"][0]["operation_params"]
    assert params["archive_path"] == source_archive.as_posix()
    assert params["source_archive"] == source_archive.as_posix()
    assert params["renderer_payload_dfl1_source_archive_role"] == (
        "source_archive_path"
    )
    assert params["renderer_payload_dfl1_source_runtime_dir"] == runtime_dir.as_posix()
    assert params["renderer_payload_dfl1_candidate_runtime_dir"] == runtime_dir.as_posix()
    row = handoff["materializer_work_queue"]["rows"][0]
    assert row["executable"] is True
    assert row["command"][
        row["command"].index("--archive-path") + 1
    ] == source_archive.as_posix()
    assert row["renderer_payload_dfl1_parity_context"]["source_runtime_dir"] == (
        runtime_dir.as_posix()
    )
    assert row["renderer_payload_dfl1_parity_context"]["candidate_runtime_dir"] == (
        runtime_dir.as_posix()
    )


def test_post_auxiliary_targeted_component_refresh_reharvests_into_chain(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_post_auxiliary_refresh_unit",
        candidate_limit=1,
    )
    targeted_component = report["targeted_component_correction_acquisition"]
    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=targeted_component,
        acquisition_id=targeted_component["top_acquisition_ids"][0],
    )
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    candidate_advisory_path = _write_json(
        tmp_path / "candidate_local_cpu_advisory.json",
        {
            "schema_version": "contest_auth_eval_result.v1",
            **_false_authority(),
            "score_axis": "cpu_advisory",
            "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
            "archive_size_bytes": 345_544,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.0008,
        },
    )
    reference_advisory_path = _write_json(
        tmp_path / "reference_local_cpu_advisory.json",
        {
            "schema_version": "contest_auth_eval_result.v1",
            **_false_authority(),
            "score_axis": "cpu_advisory",
            "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
            "archive_size_bytes": 345_802,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.001,
        },
    )
    response_path = tmp_path / "component_correction_response_harvest.json"
    queue = {
        "schema": "experiment_queue.v1",
        **_false_authority(),
        "queue_id": "post_auxiliary_targeted_component_queue_unit",
        "controls": {"max_concurrency": {"local_io_heavy": 1}},
        "experiments": [
            {
                "id": "targeted_component_response_unit",
                "metadata": {
                    **_false_authority(),
                    "correction_requests": [
                        {
                            "acquisition_id": work_order["acquisition_id"],
                            "candidate_id": work_order["candidate_id"],
                            "correction_family": work_order["correction_family"],
                            "operation_levels": work_order["operation_levels"],
                            "targeted_dimensions": work_order["targeted_dimensions"],
                            "saved_bytes_budget": work_order["saved_bytes_budget"],
                            "estimated_rate_credit_score_units": (
                                work_order[
                                    "estimated_rate_credit_score_units"
                                ]
                            ),
                            "work_order_path": work_order_path.as_posix(),
                            "local_cpu_advisory_path": (
                                candidate_advisory_path.as_posix()
                            ),
                            "reference_local_cpu_advisory_path": (
                                reference_advisory_path.as_posix()
                            ),
                            "component_correction_response_harvest_path": (
                                response_path.as_posix()
                            ),
                        }
                    ],
                },
                "steps": [],
            }
        ],
    }
    queue_path = _write_json(tmp_path / "targeted_component_queue.json", queue)

    summary = write_targeted_component_correction_post_auxiliary_artifacts(
        output_dir=tmp_path / "post_auxiliary_refresh",
        targeted_component_correction_queue=queue,
        targeted_component_correction_queue_path=queue_path,
        repo_root=tmp_path,
        results_root=results_root,
        queue_id="post_auxiliary_refresh_unit",
        candidate_limit=1,
    )

    _assert_false_authority(summary)
    assert summary["response_harvest_row_count"] == 1
    assert summary["response_harvest_local_acquisition_recommended_count"] == 1
    assert summary["materialization_request_row_count"] == 1
    assert summary["operation_chain_work_order_count"] == 1
    assert summary["chain_materializer_handoff_work_queue_row_count"] >= 1
    for artifact_path in summary["artifacts"].values():
        assert (tmp_path / artifact_path).is_file()


def test_frontier_feedback_discovers_dqs1_observation_jsonls_from_artifact_roots(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / ".omx" / "research"
    _write_jsonl(
        artifact_root / "dqs1_local_first_harvest_observations_20260525T010203Z.jsonl",
        [_dqs1_observation_row()],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_dqs1_discovery_unit",
        candidate_limit=1,
    )

    discovery = report["dqs1_observation_discovery"]
    assert discovery["schema"] == "frontier_rate_attack_dqs1_observation_discovery.v1"
    assert discovery["active"] is True
    assert discovery["discovered_observation_count"] == 1
    assert report["dqs1_observation_count"] == 1
    assert report["dqs1_observation_source_paths"] == [
        ".omx/research/dqs1_local_first_harvest_observations_20260525T010203Z.jsonl"
    ]
    assert report["selected_candidate_ids"] == ["pairset_drop_one_rank024_pair0112"]
    _assert_false_authority(discovery)
    assert report["ready_for_exact_eval_dispatch"] is False


def test_receiver_closed_correction_budget_refuses_static_runtime_gaps(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(
        tmp_path,
        results_root=results_root,
        bridge_blockers=[
            "archive_manifest_missing",
            "runtime_tree_sha256_missing",
        ],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(results_root),
        queue_id="frontier_feedback_static_gap_unit",
        candidate_limit=1,
    )

    receiver_budget = report["receiver_closed_correction_budget"]
    assert receiver_budget["schema"] == RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA
    _assert_false_authority(receiver_budget)
    assert receiver_budget["active"] is False
    assert receiver_budget["receiver_closed_candidate_count"] == 0
    assert receiver_budget["receiver_closed_saved_bytes_total"] == 0
    assert receiver_budget["blocked_candidate_count"] == 1
    assert receiver_budget["rows"][0]["release_to_targeted_correction_planning"] is False
    assert "archive_manifest_missing" in receiver_budget["rows"][0]["critical_blockers"]
    correction_budget = report["operation_portfolio"]["targeted_correction_budget_summary"]
    assert correction_budget["receiver_closed_materializer_saved_bytes_total"] == 0
    assert "materializer_saved_bytes_require_receiver_runtime_proof_before_spend" in (
        correction_budget["blockers"]
    )


def test_receiver_closed_correction_budget_pairs_handoff_root_bridge(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    candidate_id = "byte_range_entropy_recode_8460014d7085"
    handoff = (
        results_root
        / "frontier_operation_chain_compiler"
        / "byte_range_handoff_queue"
        / "chain_registered_multisurface_materializer_program"
        / "byte_range_entropy_recode_chain"
        / "exact_eval_handoff"
    )
    closure_report = handoff / "submission_closure" / "submission" / candidate_id / (
        "submission_closure_report.json"
    )
    _write_json(
        closure_report,
        {
            "schema": "materializer_submission_runtime_closure_report.v1",
            **_false_authority(),
            "candidate_id": candidate_id,
            "target_kind": "byte_range_entropy_recode_v1",
            "archive_sha256": "8" * 64,
            "archive_bytes": 178207,
            "closed_source_queue_path": (
                handoff / "submission_closure" / "closed_source_queue.json"
            ).relative_to(tmp_path).as_posix(),
            "submission_dir": closure_report.parent.relative_to(tmp_path).as_posix(),
            "saved_bytes_at_risk": 16,
            "targeted_correction_budget_signal": {
                "freed_bytes_require_receiver_and_exact_readiness_before_spend": True,
                "saved_bytes_at_risk": 16,
                **_false_authority(),
            },
            "allowed_use": "exact_readiness_static_submission_closure_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        },
    )
    _write_json(
        handoff / "exact_readiness_bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            **_false_authority(),
            "source_queue_path": (
                handoff / "submission_closure" / "closed_source_queue.json"
            ).relative_to(tmp_path).as_posix(),
            "candidate_count": 1,
            "ready_candidate_count": 0,
            "blocked_candidate_count": 1,
            "dispatch_blockers": ["bridge_report_is_not_dispatch_authority"],
            "rows": [
                {
                    **_false_authority(),
                    "candidate_id": candidate_id,
                    "readiness_verdict": "blocked",
                    "blockers": [
                        (
                            "unknown_uncleared_source_dispatch_blocker:"
                            "full_frame_render_output_parity_missing"
                        )
                    ],
                }
            ],
        },
    )

    budget = build_receiver_closed_correction_budget(
        repo_root=tmp_path,
        results_root=results_root,
    )

    row = next(item for item in budget["rows"] if item["candidate_id"] == candidate_id)
    assert row["paired_exact_readiness_bridge_report_path"].endswith(
        "exact_eval_handoff/exact_readiness_bridge_report.json"
    )
    assert "paired_exact_readiness_bridge_report_missing" not in row["critical_blockers"]
    assert (
        "unknown_uncleared_source_dispatch_blocker:full_frame_render_output_parity_missing"
        in row["critical_blockers"]
    )
    assert row["receiver_closed"] is False
    assert row["release_to_targeted_correction_planning"] is False


def test_targeted_component_queue_carries_receiver_closed_reference_eval(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    source_dir = tmp_path / "source_submission"
    source_dir.mkdir()
    source_archive = _write_json(source_dir / "archive_correct.zip", {"fixture": True})
    source_inflate = source_dir / "inflate.sh"
    source_inflate.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    candidate_runtime_dir = tmp_path / "candidate_runtime"
    candidate_runtime_dir.mkdir()
    (candidate_runtime_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )
    candidate_id = "packet_member_merge_reference_eval_unit"
    closure_dir = (
        results_root
        / "frontier_final_rate_attack"
        / "reference_eval_unit"
        / "submission_closure"
    )
    source_queue = closure_dir / "closed_source_queue.json"
    closure_report = closure_dir / "submission_closure_report.json"
    _write_json(
        source_queue,
        {
            "schema": "optimizer_candidate_queue.v1",
            **_false_authority(),
            "top_k_forensic": [
                {
                    **_false_authority(),
                    "candidate_id": candidate_id,
                    "candidate_archive_sha256": "8" * 64,
                    "source_archive_path": source_archive.relative_to(tmp_path).as_posix(),
                    "source_archive_sha256": "7" * 64,
                    "source_archive_bytes": 345_802,
                    "packet_member_merge_source_runtime_dir": (
                        source_dir.relative_to(tmp_path).as_posix()
                    ),
                }
            ],
        },
    )
    _write_json(
        closure_report,
        {
            "schema": "materializer_submission_runtime_closure_report.v1",
            **_false_authority(),
            "candidate_id": candidate_id,
            "target_kind": "packet_member_merge_v1",
            "archive_sha256": "8" * 64,
            "archive_bytes": 345_544,
            "closed_source_queue_path": source_queue.relative_to(tmp_path).as_posix(),
            "submission_dir": (closure_dir / "submission").relative_to(tmp_path).as_posix(),
            "source_runtime_dir": candidate_runtime_dir.relative_to(tmp_path).as_posix(),
            "saved_bytes_at_risk": 258,
            "targeted_correction_budget_signal": {
                "freed_bytes_require_receiver_and_exact_readiness_before_spend": True,
                "saved_bytes_at_risk": 258,
                **_false_authority(),
            },
            "allowed_use": "exact_readiness_static_submission_closure_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        },
    )
    _write_json(
        closure_dir.parent / "exact_readiness_bridge" / "exact_readiness_bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            **_false_authority(),
            "candidate_count": 1,
            "ready_candidate_count": 0,
            "blocked_candidate_count": 1,
            "dispatch_blockers": [
                "bridge_report_is_not_dispatch_authority",
                "requires_exact_eval_readiness_gate",
            ],
            "rows": [
                {
                    **_false_authority(),
                    "candidate_id": candidate_id,
                    "readiness_verdict": "blocked",
                    "blockers": [],
                }
            ],
        },
    )

    budget = build_receiver_closed_correction_budget(
        repo_root=tmp_path,
        results_root=results_root,
    )
    budget_row = next(row for row in budget["rows"] if row["candidate_id"] == candidate_id)
    assert budget_row["receiver_closed"] is True
    assert budget_row["source_archive_path"] == "source_submission/archive_correct.zip"
    assert budget_row["source_inflate_sh_path"] == "source_submission/inflate.sh"

    acquisition = build_frontier_targeted_component_correction_acquisition(
        operation_portfolio={
            "schema": OPERATION_PORTFOLIO_SCHEMA,
            **_false_authority(),
            "component_behavior_summary": {
                "schema": "frontier_rate_attack_component_behavior_summary.v1",
                **_false_authority(),
                "active": True,
            },
            "master_gradient_summary": {
                "schema": "frontier_rate_attack_master_gradient_summary.v1",
                **_false_authority(),
                "active": True,
            },
            "targeted_correction_budget_summary": {
                "schema": "frontier_rate_attack_targeted_correction_budget_summary.v1",
                **_false_authority(),
            },
        },
        receiver_closed_correction_budget=budget,
    )
    acquisition_path = _write_json(tmp_path / "targeted_acquisition.json", acquisition)
    queue = build_frontier_targeted_component_correction_queue(
        repo_root=tmp_path,
        targeted_component_correction_acquisition=acquisition,
        targeted_component_correction_acquisition_path=acquisition_path,
        results_root=tmp_path / "queue_results",
        queue_id="targeted_component_reference_eval_unit",
        candidate_limit=1,
    )

    assert queue is not None
    experiment = queue["experiments"][0]
    step_ids = [step["id"] for step in experiment["steps"]]
    assert "local_cpu_reference_advisory" in step_ids
    assert "build_reference_mlx_component_cache" in step_ids
    assert "reference_local_mlx_component_response" in step_ids
    steps_by_id = {step["id"]: step for step in experiment["steps"]}
    reference_step = steps_by_id["local_cpu_reference_advisory"]
    assert (
        "--allow-scorer-input-cache-artifact-output-outside-work-dir"
        in reference_step["command"]
    )
    assert "--reuse-valid-json-out" in reference_step["command"]
    assert any(
        condition.get("path", "").endswith("reference_scorer_input_cache_hashes.json")
        and condition.get("key") == "schema_version"
        for condition in reference_step["postconditions"]
    )
    assert not any(
        path.endswith("reference_local_cpu_advisory_work")
        for path in reference_step["telemetry"]["artifact_paths"]
    )
    assert reference_step["telemetry"].get("recursive") is not True
    component_step = steps_by_id["local_cpu_component_advisory"]
    assert (
        "--allow-scorer-input-cache-artifact-output-outside-work-dir"
        in component_step["command"]
    )
    assert "--reuse-valid-json-out" in component_step["command"]
    assert any(
        condition.get("path", "").endswith("scorer_input_cache_hashes.json")
        and condition.get("key") == "schema_version"
        for condition in component_step["postconditions"]
    )
    assert not any(
        path.endswith("local_cpu_advisory_work")
        for path in component_step["telemetry"]["artifact_paths"]
    )
    assert component_step["telemetry"].get("recursive") is not True
    assert experiment["metadata"]["reference_component_eval_available"] is True
    assert "--reuse-valid-cache" in steps_by_id["build_mlx_component_cache"]["command"]
    assert (
        "--reuse-valid-cache"
        in steps_by_id["build_reference_mlx_component_cache"]["command"]
    )
    assert experiment["metadata"]["reference_archive_path"] == (
        "source_submission/archive_correct.zip"
    )
    harvest_steps = [
        step
        for step in experiment["steps"]
        if step["id"].startswith("harvest_targeted_component_correction_response")
    ]
    assert harvest_steps
    assert "--reference-local-cpu-advisory" in harvest_steps[0]["command"]
    assert "--reference-local-mlx-response" in harvest_steps[0]["command"]
    assert "local_cpu_reference_advisory" in harvest_steps[0]["requires"]
    assert "reference_local_mlx_component_response" in harvest_steps[0]["requires"]
    request = experiment["metadata"]["correction_requests"][0]
    assert "shared_candidate_component_response" in request["shared_component_response_dir"]
    assert "shared_reference_component_response" in (
        request["reference_shared_component_response_dir"]
    )
    assert request["reference_local_cpu_advisory_path"].endswith(
        "reference_local_cpu_advisory.json"
    )
    assert request["reference_local_mlx_response_path"].endswith(
        "reference_mlx_scorer_response.json"
    )
    assert experiment["metadata"]["reference_local_cpu_advisory_path"].startswith(
        "queue_results/frontier_targeted_component_correction/"
        "shared_reference_component_response/"
    )


def test_targeted_component_queue_recovers_reference_eval_from_closure_report(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    source_dir = tmp_path / "source_submission"
    source_dir.mkdir()
    source_archive = _write_json(source_dir / "archive_correct.zip", {"fixture": True})
    (source_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    candidate_id = "packet_member_merge_reference_recovery_unit"
    closure_dir = (
        results_root
        / "frontier_final_rate_attack"
        / "reference_recovery_unit"
        / "submission_closure"
    )
    source_queue = closure_dir / "closed_source_queue.json"
    closure_report = closure_dir / "submission_closure_report.json"
    _write_json(
        source_queue,
        {
            "schema": "optimizer_candidate_queue.v1",
            **_false_authority(),
            "top_k_forensic": [
                {
                    **_false_authority(),
                    "candidate_id": candidate_id,
                    "candidate_archive_sha256": "8" * 64,
                    "source_archive_path": source_archive.relative_to(tmp_path).as_posix(),
                    "source_archive_sha256": "7" * 64,
                    "source_archive_bytes": 345_802,
                    "packet_member_merge_source_runtime_dir": (
                        source_dir.relative_to(tmp_path).as_posix()
                    ),
                }
            ],
        },
    )
    _write_json(
        closure_report,
        {
            "schema": "materializer_submission_runtime_closure_report.v1",
            **_false_authority(),
            "candidate_id": candidate_id,
            "target_kind": "packet_member_merge_v1",
            "archive_sha256": "8" * 64,
            "archive_bytes": 345_544,
            "closed_source_queue_path": source_queue.relative_to(tmp_path).as_posix(),
            "submission_dir": (closure_dir / "submission").relative_to(tmp_path).as_posix(),
            "saved_bytes_at_risk": 258,
            "allowed_use": "exact_readiness_static_submission_closure_only",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
        },
    )
    acquisition = {
        "schema": TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA,
        "active": True,
        "row_count": 1,
        "queue_actionable_acquisition_count": 1,
        "rows": [
            {
                "acquisition_id": "component_reference_recovery_row",
                "candidate_id": candidate_id,
                "correction_family": "repair_dynamics_frame0_palette_interaction_waterfill",
                "target_kind": "packet_member_merge_v1",
                "operation_levels": ["frame", "pair"],
                "targeted_dimensions": ["frame", "pair"],
                "saved_bytes_budget": 258,
                "estimated_rate_credit_score_units": 0.0001717916099055202,
                "submission_dir": "submissions/candidate_reference_recovery",
                "archive_path": "submissions/candidate_reference_recovery/archive.zip",
                "archive_sha256": "8" * 64,
                "inflate_sh_path": "submissions/candidate_reference_recovery/inflate.sh",
                "closure_report_path": closure_report.relative_to(tmp_path).as_posix(),
                "reference_component_eval_context": {},
                "queue_actionable": True,
                "priority_score": 1.0,
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }
    acquisition_path = _write_json(tmp_path / "targeted_acquisition.json", acquisition)

    queue = build_frontier_targeted_component_correction_queue(
        repo_root=tmp_path,
        targeted_component_correction_acquisition=acquisition,
        targeted_component_correction_acquisition_path=acquisition_path,
        results_root=tmp_path / "queue_results",
        queue_id="component_reference_recovery_unit",
        candidate_limit=1,
    )
    assert queue is not None
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["reference_component_eval_available"] is True
    assert experiment["metadata"]["reference_archive_path"] == (
        "source_submission/archive_correct.zip"
    )
    assert experiment["metadata"]["reference_inflate_sh_path"] == (
        "source_submission/inflate.sh"
    )
    request = experiment["metadata"]["correction_requests"][0]
    assert request["reference_component_eval_context"][
        "reference_context_recovery_mode"
    ] == "receiver_closure_source_reference_context"
    step_ids = [step["id"] for step in experiment["steps"]]
    assert "local_cpu_reference_advisory" in step_ids
    assert "reference_local_mlx_component_response" in step_ids

    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=acquisition,
        acquisition_id="component_reference_recovery_row",
        repo_root=tmp_path,
    )
    assert work_order["source_archive_path"] == "source_submission/archive_correct.zip"
    assert work_order["source_inflate_sh_path"] == "source_submission/inflate.sh"
    assert work_order["reference_component_eval_context"][
        "reference_context_recovery_source"
    ].endswith("submission_closure_report.json")


def test_targeted_component_queue_imports_false_authority_component_response_cache(
    tmp_path: Path,
) -> None:
    acquisition = {
        "schema": TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA,
        "active": True,
        "row_count": 1,
        "queue_actionable_acquisition_count": 1,
        "rows": [
            {
                "acquisition_id": "component_cache_import_row",
                "candidate_id": "candidate_cache_import",
                "correction_family": "segnet_posenet_waterfill_region_repair",
                "target_kind": "renderer_payload_dfl1_v1",
                "operation_levels": ["region"],
                "targeted_dimensions": ["segnet", "posenet"],
                "saved_bytes_budget": 512,
                "estimated_rate_credit_score_units": 0.000001,
                "submission_dir": "submissions/candidate_cache_import",
                "archive_path": "submissions/candidate_cache_import/archive.zip",
                "archive_sha256": "c" * 64,
                "inflate_sh_path": "submissions/candidate_cache_import/inflate.sh",
                "source_archive_path": "source_submission/archive_correct.zip",
                "source_archive_sha256": "d" * 64,
                "source_inflate_sh_path": "source_submission/inflate.sh",
                "queue_actionable": True,
                "priority_score": 1.0,
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }
    acquisition_path = _write_json(
        tmp_path / "targeted_acquisition.json",
        acquisition,
    )
    probe_queue = build_frontier_targeted_component_correction_queue(
        repo_root=tmp_path,
        targeted_component_correction_acquisition=acquisition,
        targeted_component_correction_acquisition_path=acquisition_path,
        results_root=tmp_path / "probe_results",
        queue_id="component_cache_probe",
        candidate_limit=1,
        include_mlx_response=False,
    )
    assert probe_queue is not None
    probe_metadata = probe_queue["experiments"][0]["metadata"]
    candidate_cache_key = probe_metadata["candidate_cache_key"]
    reference_cache_key = probe_metadata["reference_cache_key"]
    cache_root = tmp_path / "component_cache_root"
    candidate_cache_dir = (
        cache_root / "shared_candidate_component_response" / candidate_cache_key
    )
    reference_cache_dir = (
        cache_root / "shared_reference_component_response" / reference_cache_key
    )
    candidate_cache_dir.mkdir(parents=True)
    reference_cache_dir.mkdir(parents=True)
    advisory = {
        "schema": "unit_local_cpu_advisory.v1",
        "score_axis": "cpu_advisory",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "archive_size_bytes": 123456,
        **_false_authority(),
    }
    hashes = {
        "schema_version": "mlx_scorer_input_cache_hashes.v1",
        **_false_authority(),
    }
    _write_json(candidate_cache_dir / "local_cpu_advisory.json", advisory)
    _write_json(candidate_cache_dir / "scorer_input_cache_hashes.json", hashes)
    _write_json(reference_cache_dir / "reference_local_cpu_advisory.json", advisory)
    _write_json(
        reference_cache_dir / "reference_scorer_input_cache_hashes.json",
        hashes,
    )
    candidate_mlx_cache = candidate_cache_dir / "mlx_scorer_input_cache"
    reference_mlx_cache = reference_cache_dir / "reference_mlx_scorer_input_cache"
    candidate_mlx_cache.mkdir()
    reference_mlx_cache.mkdir()
    _write_json(candidate_mlx_cache / "manifest.json", {"passed": True, **_false_authority()})
    _write_json(
        reference_mlx_cache / "manifest.json",
        {"passed": True, **_false_authority()},
    )
    _write_json(
        candidate_cache_dir / "mlx_scorer_input_cache_audit.json",
        {"passed": True, **_false_authority()},
    )
    _write_json(
        reference_cache_dir / "reference_mlx_scorer_input_cache_audit.json",
        {"passed": True, **_false_authority()},
    )

    queue = build_frontier_targeted_component_correction_queue(
        repo_root=tmp_path,
        targeted_component_correction_acquisition=acquisition,
        targeted_component_correction_acquisition_path=acquisition_path,
        results_root=tmp_path / "queue_results",
        queue_id="component_cache_import_unit",
        candidate_limit=1,
        component_response_cache_roots=[cache_root],
    )

    assert queue is not None
    experiment = queue["experiments"][0]
    steps_by_id = {step["id"]: step for step in experiment["steps"]}
    assert "import_local_cpu_component_advisory_cache" in steps_by_id
    assert "import_local_cpu_reference_advisory_cache" in steps_by_id
    assert "local_cpu_component_advisory" not in steps_by_id
    assert "local_cpu_reference_advisory" not in steps_by_id
    assert "reuse_mlx_component_cache" in steps_by_id
    assert "reuse_reference_mlx_component_cache" in steps_by_id
    assert "build_mlx_component_cache" not in steps_by_id
    assert "build_reference_mlx_component_cache" not in steps_by_id
    assert steps_by_id["reuse_mlx_component_cache"]["requires"] == [
        "import_local_cpu_component_advisory_cache"
    ]
    assert steps_by_id["reuse_reference_mlx_component_cache"]["requires"] == [
        "import_local_cpu_reference_advisory_cache"
    ]
    assert steps_by_id["local_mlx_component_response"]["requires"] == [
        "reuse_mlx_component_cache"
    ]
    assert steps_by_id["reference_local_mlx_component_response"]["requires"] == [
        "reuse_reference_mlx_component_cache"
    ]
    component_step = steps_by_id["import_local_cpu_component_advisory_cache"]
    reference_step = steps_by_id["import_local_cpu_reference_advisory_cache"]
    assert component_step["command"][1] == "tools/import_frontier_component_response_cache.py"
    assert reference_step["command"][1] == "tools/import_frontier_component_response_cache.py"
    assert component_step["requires"] == ["emit_targeted_component_correction_work_order"]
    assert reference_step["requires"] == ["emit_targeted_component_correction_work_order"]
    request = experiment["metadata"]["correction_requests"][0]
    assert request["local_cpu_advisory_reuse_mode"] == (
        "import_false_authority_component_response_cache"
    )
    assert request["reference_local_cpu_advisory_reuse_mode"] == (
        "import_false_authority_component_response_cache"
    )
    assert request["local_mlx_cache_reuse_mode"] == (
        "reuse_false_authority_mlx_scorer_input_cache"
    )
    assert request["reference_local_mlx_cache_reuse_mode"] == (
        "reuse_false_authority_mlx_scorer_input_cache"
    )


def test_targeted_component_queue_limits_repair_dynamics_probe_to_repair_families(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_repair_probe_gate"
    base_row = {
        "candidate_id": candidate_id,
        "target_kind": "packet_member_merge_v1",
        "operation_levels": ["frame", "pair"],
        "targeted_dimensions": ["segnet", "posenet"],
        "saved_bytes_budget": 258,
        "estimated_rate_credit_score_units": 0.0001717916099055202,
        "submission_dir": "submissions/candidate_repair_probe_gate",
        "archive_path": "submissions/candidate_repair_probe_gate/archive.zip",
        "archive_sha256": "e" * 64,
        "inflate_sh_path": "submissions/candidate_repair_probe_gate/inflate.sh",
        "source_archive_path": "submissions/source/archive_correct.zip",
        "source_archive_sha256": "f" * 64,
        "source_inflate_sh_path": "submissions/source/inflate.sh",
        "repair_dynamics_prior_active": True,
        "queue_actionable": True,
        "priority_score": 1.0,
        **_false_authority(),
    }
    acquisition = {
        "schema": TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA,
        "active": True,
        "row_count": 2,
        "queue_actionable_acquisition_count": 2,
        "rows": [
            {
                **base_row,
                "acquisition_id": "repair_probe_required_row",
                "correction_family": (
                    "repair_dynamics_frame0_palette_interaction_waterfill"
                ),
            },
            {
                **base_row,
                "acquisition_id": "waterfill_without_palette_probe_row",
                "correction_family": "segnet_posenet_waterfill_region_repair",
            },
        ],
        **_false_authority(),
    }
    acquisition_path = _write_json(
        tmp_path / "targeted_acquisition.json",
        acquisition,
    )

    queue = build_frontier_targeted_component_correction_queue(
        repo_root=tmp_path,
        targeted_component_correction_acquisition=acquisition,
        targeted_component_correction_acquisition_path=acquisition_path,
        results_root=tmp_path / "queue_results",
        queue_id="repair_probe_gate_unit",
        candidate_limit=2,
        include_mlx_response=False,
    )

    assert queue is not None
    experiment = queue["experiments"][0]
    probe_steps = [
        step
        for step in experiment["steps"]
        if step["id"].startswith("emit_repair_dynamics_palette_probe_matrix")
    ]
    assert [step["id"] for step in probe_steps] == [
        "emit_repair_dynamics_palette_probe_matrix_01"
    ]
    requests = {
        request["acquisition_id"]: request
        for request in experiment["metadata"]["correction_requests"]
    }
    repair_request = requests["repair_probe_required_row"]
    waterfill_request = requests["waterfill_without_palette_probe_row"]
    assert repair_request["repair_dynamics_prior_active"] is True
    assert repair_request["repair_dynamics_probe_required"] is True
    assert repair_request["repair_dynamics_palette_probe_matrix_path"] is not None
    assert waterfill_request["repair_dynamics_prior_active"] is True
    assert waterfill_request["repair_dynamics_probe_required"] is False
    assert waterfill_request["repair_dynamics_palette_probe_matrix_path"] is None


def test_targeted_component_acquisition_uses_explicit_repair_dynamics_palette_prior() -> None:
    operation_portfolio = {
        "schema": OPERATION_PORTFOLIO_SCHEMA,
        **_false_authority(),
        "component_behavior_summary": {
            "schema": "frontier_rate_attack_component_behavior_summary.v1",
            **_false_authority(),
            "active": True,
            "best_candidate_id": "pr110_palette_control",
        },
        "master_gradient_summary": {
            "schema": "frontier_rate_attack_master_gradient_summary.v1",
            **_false_authority(),
            "active": True,
        },
        "targeted_correction_budget_summary": {
            "schema": "frontier_rate_attack_targeted_correction_budget_summary.v1",
            **_false_authority(),
        },
    }
    receiver_budget = {
        "schema": RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA,
        **_false_authority(),
        "active": True,
        "rows": [
            {
                **_false_authority(),
                "candidate_id": "pr110_rate_win",
                "target_kind": "renderer_payload_dfl1_v1",
                "receiver_closed": True,
                "saved_bytes_at_risk": 320,
                "submission_dir": "submissions/pr110_rate_win",
                "source_archive_path": "submissions/pr110/archive.zip",
                "source_inflate_sh_path": "submissions/pr110/inflate.sh",
                "correction_budget_gate": "receiver_closed_rate_budget",
                "active_rate_floor_blocked": False,
            }
        ],
    }
    prior = {
        "schema": "frontier_rate_attack_repair_dynamics_palette_prior.v1",
        **_false_authority(),
        "source": "unit_test_pr110_k16",
        "palette_modes": list(FEC6_FIXED_K16_MODE_IDS),
        "allowed_use": "repair_dynamics_prior_for_local_waterfill_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
    }

    acquisition = build_frontier_targeted_component_correction_acquisition(
        operation_portfolio=operation_portfolio,
        receiver_closed_correction_budget=receiver_budget,
        repair_dynamics_palette_prior=prior,
    )

    assert acquisition["schema"] == TARGETED_COMPONENT_CORRECTION_ACQUISITION_SCHEMA
    _assert_false_authority(acquisition)
    assert acquisition["repair_dynamics_prior_active"] is True
    assert acquisition["repair_dynamics_palette_prior"]["mode_count"] == 16
    assert acquisition["repair_dynamics_palette_prior"]["zero_frame1_modes"] is True
    assert acquisition["repair_dynamics_palette_probe_count"] >= 3
    assert "repair_dynamics_palette_probe_required_before_budget_spend" in (
        acquisition["blockers"]
    )
    families = {row["correction_family"] for row in acquisition["rows"]}
    assert "repair_dynamics_frame0_palette_interaction_waterfill" in families
    assert "repair_dynamics_chroma_luma_bias_basis_expansion" in families
    assert "repair_dynamics_frame1_counterfactual_null_probe" in families
    row = next(
        row
        for row in acquisition["rows"]
        if row["correction_family"]
        == "repair_dynamics_frame0_palette_interaction_waterfill"
    )
    _assert_false_authority(row)
    assert "repair_dynamics_palette_prior" in row["prior_status"]["available_priors"]
    assert row["repair_dynamics_context"]["zero_frame1_modes"] is True
    assert row["budget_spend_allowed"] is False

    work_order = build_frontier_targeted_component_correction_work_order(
        targeted_component_correction_acquisition=acquisition,
        acquisition_id=row["acquisition_id"],
    )
    _assert_false_authority(work_order)
    assert work_order["repair_dynamics_prior_active"] is True
    assert work_order["repair_dynamics_palette_prior"]["mode_count"] == 16
    assert work_order["budget_spend_gate"]["budget_spend_allowed"] is False
    hint_ids = {hint["action_id"] for hint in work_order["command_hints"]}
    assert "build_repair_dynamics_palette_probe_matrix" in hint_ids


def test_targeted_component_acquisition_does_not_default_to_pr110_palette() -> None:
    acquisition = build_frontier_targeted_component_correction_acquisition(
        operation_portfolio={
            "schema": OPERATION_PORTFOLIO_SCHEMA,
            **_false_authority(),
            "component_behavior_summary": {
                "schema": "frontier_rate_attack_component_behavior_summary.v1",
                **_false_authority(),
                "active": True,
            },
        },
        receiver_closed_correction_budget={
            "schema": RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA,
            **_false_authority(),
            "active": True,
            "rows": [
                {
                    **_false_authority(),
                    "candidate_id": "generic_rate_win",
                    "target_kind": "packet_member_merge_v1",
                    "receiver_closed": True,
                    "saved_bytes_at_risk": 128,
                    "submission_dir": "submissions/generic_rate_win",
                }
            ],
        },
    )

    assert acquisition["repair_dynamics_prior_active"] is False
    assert acquisition["repair_dynamics_palette_prior"] == {}
    assert acquisition["repair_dynamics_palette_probe_count"] == 0
    assert not any(
        str(row["correction_family"]).startswith("repair_dynamics_")
        for row in acquisition["rows"]
    )


def test_targeted_component_acquisition_keeps_mixed_frame_palette_open() -> None:
    acquisition = build_frontier_targeted_component_correction_acquisition(
        operation_portfolio={
            "schema": OPERATION_PORTFOLIO_SCHEMA,
            **_false_authority(),
            "component_behavior_summary": {
                "schema": "frontier_rate_attack_component_behavior_summary.v1",
                **_false_authority(),
                "active": True,
            },
        },
        receiver_closed_correction_budget={
            "schema": RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA,
            **_false_authority(),
            "active": True,
            "rows": [
                {
                    **_false_authority(),
                    "candidate_id": "mixed_frame_palette_rate_win",
                    "target_kind": "packet_member_merge_v1",
                    "receiver_closed": True,
                    "saved_bytes_at_risk": 128,
                    "submission_dir": "submissions/mixed_frame_palette_rate_win",
                }
            ],
        },
        repair_dynamics_palette_prior={
            "schema": "frontier_rate_attack_repair_dynamics_palette_prior.v1",
            **_false_authority(),
            "source": "unit_test_mixed_frame_palette",
            "palette_modes": [
                "none",
                "frame0_luma_bias_+1",
                "frame1_luma_bias_+1",
            ],
        },
    )

    assert acquisition["repair_dynamics_prior_active"] is True
    assert acquisition["repair_dynamics_palette_prior"]["zero_frame1_modes"] is False
    families = {row["correction_family"] for row in acquisition["rows"]}
    assert "repair_dynamics_frame0_palette_interaction_waterfill" in families
    assert "repair_dynamics_frame1_counterfactual_null_probe" not in families


def test_targeted_component_acquisition_refuses_truthy_repair_dynamics_authority() -> None:
    with pytest.raises(ValueError, match="forbidden truthy authority fields"):
        build_frontier_targeted_component_correction_acquisition(
            operation_portfolio={
                "schema": OPERATION_PORTFOLIO_SCHEMA,
                **_false_authority(),
                "component_behavior_summary": {
                    "schema": "frontier_rate_attack_component_behavior_summary.v1",
                    **_false_authority(),
                    "active": True,
                },
            },
            receiver_closed_correction_budget={
                "schema": RECEIVER_CLOSED_CORRECTION_BUDGET_SCHEMA,
                **_false_authority(),
                "active": True,
                "rows": [
                    {
                        **_false_authority(),
                        "candidate_id": "bad_prior_rate_win",
                        "target_kind": "packet_member_merge_v1",
                        "receiver_closed": True,
                        "saved_bytes_at_risk": 128,
                        "submission_dir": "submissions/bad_prior_rate_win",
                    }
                ],
            },
            repair_dynamics_palette_prior={
                "schema": "frontier_rate_attack_repair_dynamics_palette_prior.v1",
                **_false_authority(),
                "palette_modes": ["none"],
                "score_claim": True,
            },
        )


def test_frontier_feedback_compiler_turns_eureka_near_misses_into_beyond_drop_two_hints(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    eureka_path = _write_json(
        artifact_root
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        _eureka_signal(),
    )

    discovery = discover_local_cpu_eureka_planning_signals(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
    )

    assert discovery["schema"] == LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA
    assert discovery["active"] is True
    assert discovery["signal_count"] == 1
    assert discovery["candidate_family_counts"] == {"decoder_q_pairset_drop_two": 1}
    assert discovery["planner_hint_count"] == 1
    hint = discovery["planner_hints"][0]
    assert hint["hint_id"] == "dqs1_expand_beyond_drop_two_near_boundary"
    assert "learned_multi_drop" in hint["recommended_candidate_families"]
    assert "drop_many_beam_pairwise_interaction_waterfill" in hint[
        "recommended_candidate_families"
    ]
    assert "inverse_scorer_action_surface" in hint["planner_consumers"]
    _assert_false_authority(hint)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_unit",
        candidate_limit=1,
    )

    eureka = report["local_cpu_eureka_planning"]
    assert eureka["signal_rows"][0]["path"] == eureka_path.relative_to(tmp_path).as_posix()
    assert eureka["planner_hints"][0]["hint_id"] == (
        "dqs1_expand_beyond_drop_two_near_boundary"
    )
    experiment = report["queue"]["experiments"][0]
    assert experiment["metadata"]["frontier_feedback_eureka_planning"][
        "planner_hint_count"
    ] == 1
    operation_portfolio = report["operation_portfolio"]
    assert operation_portfolio["schema"] == OPERATION_PORTFOLIO_SCHEMA
    operation_ids = set(operation_portfolio["top_operation_ids"])
    assert "eureka_learned_multi_drop" in {
        row["operation_id"] for row in operation_portfolio["rows"]
    }
    assert "eureka_drop_many_beam_pairwise_interaction_waterfill" in operation_ids
    assert operation_portfolio["component_behavior_summary"]["active"] is False
    assert report["queue"]["experiments"][0]["metadata"][
        "frontier_operation_portfolio"
    ]["operation_count"] == operation_portfolio["operation_count"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False

    artifacts = write_frontier_refresh_artifacts(
        output_dir=tmp_path / "refresh_artifacts",
        report=report,
        repo_root=tmp_path,
    )
    assert artifacts["local_cpu_eureka_planning"].endswith(
        "local_cpu_eureka_planning.json"
    )
    assert artifacts["operation_portfolio"].endswith("operation_portfolio.json")
    assert artifacts["rate_budget_preservation_plan"].endswith(
        "rate_budget_preservation_plan.json"
    )
    assert artifacts["operation_chain_compiler_work_orders"].endswith(
        "operation_chain_compiler_work_orders.json"
    )
    assert artifacts["operation_chain_compiler_queue"].endswith(
        "operation_chain_compiler_queue.json"
    )
    assert artifacts["receiver_repair_backlog"].endswith(
        "receiver_repair_backlog.json"
    )
    chain_work_orders = json.loads(
        (tmp_path / artifacts["operation_chain_compiler_work_orders"]).read_text(
            encoding="utf-8"
        )
    )
    assert chain_work_orders["schema"] == (
        "frontier_rate_attack_operation_chain_compiler_work_orders.v1"
    )
    assert chain_work_orders["work_order_count"] >= 1
    assert {
        "per_stage_materializer_contexts",
        "single_composed_receiver_runtime_consumption_proof",
        "chain_exact_readiness_bridge",
        "targeted_component_budget_spend_gate",
    }.issubset(
        set(chain_work_orders["work_orders"][0]["required_before_execution"])
    )
    _assert_false_authority(chain_work_orders)
    chain_queue_payload = json.loads(
        (tmp_path / artifacts["operation_chain_compiler_queue"]).read_text(
            encoding="utf-8"
        )
    )
    assert chain_queue_payload["schema"] == "experiment_queue.v1"
    assert chain_queue_payload["experiments"][0]["steps"][0]["command"][1] == (
        "tools/build_frontier_operation_chain_stage_plan.py"
    )
    _assert_false_authority(chain_queue_payload["experiments"][0]["metadata"])
    artifact_payload = json.loads(
        (tmp_path / artifacts["local_cpu_eureka_planning"]).read_text(
            encoding="utf-8"
        )
    )
    assert artifact_payload["planner_hint_count"] == 1
    rate_plan_payload = json.loads(
        (tmp_path / artifacts["rate_budget_preservation_plan"]).read_text(
            encoding="utf-8"
        )
    )
    assert rate_plan_payload["schema"] == RATE_BUDGET_PRESERVATION_PLAN_SCHEMA
    _assert_false_authority(rate_plan_payload)


def test_frontier_feedback_eureka_default_discovers_research_root(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    research_root = tmp_path / ".omx" / "research"
    _write_json(
        research_root
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        _eureka_signal(),
    )

    discovery = discover_local_cpu_eureka_planning_signals(repo_root=tmp_path)

    assert discovery["active"] is True
    assert discovery["frontier_artifact_roots"] == [".omx/research"]
    assert discovery["signal_count"] == 1
    assert discovery["planner_hint_count"] == 1
    _assert_false_authority(discovery)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_default_eureka_unit",
        candidate_limit=1,
    )

    eureka = report["local_cpu_eureka_planning"]
    assert eureka["active"] is True
    assert eureka["signal_count"] == 1
    assert report["queue"]["experiments"][0]["metadata"][
        "frontier_feedback_eureka_planning"
    ]["planner_hint_count"] == 1
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_component_marginal_bundle_auto_discovers_drop_many_greedy_verdict(
    tmp_path: Path,
) -> None:
    acquisition = _write_json(
        tmp_path / "pairset_acquisition.json",
        {
            "schema": "decoder_q_pairset_acquisition.v1",
            **_false_authority(),
            "candidates": [
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank023_pair0440",
                    "selector_id": "pairset_drop_one_rank023_pair0440",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 23,
                    "predicted_score_mean": 0.1919,
                    "selected_pair_indices": [1, 2, 440],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 23,
                        "dropped_pair_index": 440,
                    },
                },
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_many_fixture_k004",
                    "selector_id": "pairset_drop_many_fixture_k004",
                    "selector_kind": "drop_many_beam_pairwise_interaction_waterfill",
                    "acquisition_rank": 24,
                    "predicted_score_mean": 0.1918,
                    "selected_pair_indices": [1, 2],
                    "selected_pair_count": 2,
                    "acquisition_operation": {
                        "op": "drop_many",
                        "dropped_pair_indices": [112, 233, 371, 440],
                    },
                },
            ],
        },
    )
    observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )
    _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "dqs1_drop_many_build_1c_greedy_heuristic_fixture"
        / "verdict.json",
        _drop_many_greedy_negative_verdict(),
    )
    out = tmp_path / "component_bundle"

    bundle = write_pairset_component_marginal_feedback_bundle(
        repo_root=tmp_path,
        pairset_acquisition_paths=(acquisition,),
        observation_paths=(observations,),
        incumbent_score=0.19202,
        incumbent_scores_by_axis={"macos_cpu_advisory": 0.19202},
        output_dir=out,
        top_k=8,
        top_actions=8,
    )

    assert bundle["drop_many_greedy_verdict_count"] == 1
    assert bundle["drop_many_greedy_verdict_discovery"]["active"] is True
    summary = json.loads((out / "action_summary.json").read_text(encoding="utf-8"))
    assert summary["drop_many_greedy_verdict_model"]["active"] is True
    portfolio = json.loads((out / "portfolio.json").read_text(encoding="utf-8"))
    independent = next(
        row
        for row in portfolio["operator_action_rows"]
        if row["candidate_id"] == "pairset_drop_many_fixture_k004"
    )
    assert independent["operator_next_action"] == (
        "hold_independent_drop_many_until_interaction_or_component_model"
    )
    assert (
        "drop_many_independent_greedy_deferred_requires_interaction_or_component_model"
        in independent["source_dispatch_blockers"]
    )
    assert independent["score_claim"] is False
    assert independent["ready_for_exact_eval_dispatch"] is False


def test_drop_many_greedy_discovery_uses_bounded_known_verdict_pattern(
    tmp_path: Path,
) -> None:
    verdict = _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "dqs1_drop_many_build_1c_greedy_heuristic_fixture"
        / "verdict.json",
        _drop_many_greedy_negative_verdict(),
    )
    _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "unrelated"
        / "very"
        / "deep"
        / "dqs1_drop_many_accidental_verdict.json",
        _drop_many_greedy_negative_verdict(),
    )

    discovery = discover_dqs1_drop_many_greedy_verdict_paths(repo_root=tmp_path)

    assert discovery["active"] is True
    assert discovery["discovered_verdict_paths"] == [
        verdict.relative_to(tmp_path).as_posix()
    ]
    assert discovery["refusal_count"] == 0
    assert discovery["score_claim"] is False
    assert discovery["ready_for_exact_eval_dispatch"] is False


def test_frontier_feedback_compiler_promotes_pair_frame_geometry_requests_to_queue(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    lattice_path = _write_json(
        tmp_path / "frontier_artifacts" / "pair_frame_scorer_geometry_lattice.json",
        _pair_frame_geometry_lattice(),
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        pair_frame_geometry_paths=(lattice_path,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_pair_frame_unit",
        candidate_limit=2,
    )

    assert report["pair_frame_geometry_queue_request_count"] == 1
    discovery = report["pair_frame_geometry_discovery"]
    assert discovery["queue_executable_request_count"] == 1
    assert discovery["discovered_lattices"][0]["candidate_ids"] == [
        "pairset_geometry_lowimpact_k003_habcdef1234"
    ]
    assert report["selected_candidate_ids"] == [
        "pairset_geometry_lowimpact_k003_habcdef1234",
        "pairset_drop_one_rank023_pair0440",
    ]
    geometry_experiment = report["queue"]["experiments"][0]
    assert geometry_experiment["metadata"]["source_metadata"][
        "queue_source_kind"
    ] == "pair_frame_scorer_geometry_lattice"
    selected_acquisition = report["selected_pairset_acquisition"]
    assert selected_acquisition["schema"] == "dqs1_selected_pairset_acquisition.v1"
    assert selected_acquisition["candidate_count"] == 2
    assert selected_acquisition["candidates"][0]["candidate_id"] == (
        "pairset_geometry_lowimpact_k003_habcdef1234"
    )
    assert selected_acquisition["candidates"][0]["selector_kind"] == (
        "pair_frame_geometry_low_impact_drop_many"
    )
    assert geometry_experiment["metadata"]["selected_pair_indices"] == [
        1,
        2,
        112,
        233,
        440,
    ]
    plan_packet = {
        step["id"]: step["command"] for step in geometry_experiment["steps"]
    }["plan_packet"]
    assert plan_packet[-1] == "1,2,112,233,440"
    artifacts = write_frontier_refresh_artifacts(
        output_dir=tmp_path / "refresh_artifacts",
        report=report,
        repo_root=tmp_path,
    )
    assert artifacts["dqs1_selected_pairset_acquisition"].endswith(
        "dqs1_selected_pairset_acquisition.json"
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_frontier_feedback_default_discovers_pair_frame_geometry_lattice(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    lattice_path = _write_json(
        tmp_path
        / ".omx"
        / "research"
        / "pair_frame_geometry_run"
        / "pair_frame_scorer_geometry_lattice.json",
        _pair_frame_geometry_lattice(
            candidate_id="pairset_geometry_lowimpact_k004_h1234567890",
            selected_pair_indices=[1, 2, 112, 233],
        ),
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_pair_frame_default",
        candidate_limit=1,
    )

    discovery = report["pair_frame_geometry_discovery"]
    assert discovery["frontier_artifact_roots"] == [".omx/research"]
    assert discovery["explicit_pair_frame_geometry_paths"] == []
    assert discovery["queue_executable_request_count"] == 1
    assert discovery["discovered_lattices"][0]["path"] == (
        lattice_path.relative_to(tmp_path).as_posix()
    )
    assert report["selected_candidate_ids"] == [
        "pairset_geometry_lowimpact_k004_h1234567890"
    ]
    assert report["queue"]["experiments"][0]["metadata"]["source_metadata"][
        "queue_source_kind"
    ] == "pair_frame_scorer_geometry_lattice"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_frontier_feedback_pair_frame_geometry_discovery_rejects_authority_leak(
    tmp_path: Path,
) -> None:
    payload = _pair_frame_geometry_lattice()
    request = payload["queue_executable_pairset_drop_requests"][0]  # type: ignore[index]
    assert isinstance(request, dict)
    request["score_claim"] = True
    lattice_path = _write_json(
        tmp_path / "pair_frame_scorer_geometry_lattice.json",
        payload,
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="score_claim"):
        build_frontier_rate_attack_feedback_refresh(
            repo_root=tmp_path,
            pair_frame_geometry_paths=(lattice_path,),
        )


def test_frontier_feedback_eureka_discovery_rejects_authority_leak(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    payload = _eureka_signal()
    payload["score_claim"] = True
    _write_json(
        artifact_root
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        payload,
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="score_claim"):
        discover_local_cpu_eureka_planning_signals(
            repo_root=tmp_path,
            frontier_artifact_roots=(artifact_root,),
        )


def test_feedback_refresh_ignores_stale_eureka_authority_in_root_scan(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    payload = _eureka_signal()
    payload["score_claim"] = True
    _write_json(
        artifact_root
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        payload,
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        local_cpu_eureka_roots=(artifact_root,),
        action_summary_path=None,
    )

    eureka = report["local_cpu_eureka_planning"]
    assert eureka["signal_count"] == 0
    assert len(eureka["ignored_signal_candidates"]) == 1
    assert "score_claim" in eureka["ignored_signal_candidates"][0]["reason"]
    assert report["operation_portfolio"]["score_claim"] is False


def test_frontier_feedback_compiler_fails_closed_on_truthy_authority(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    bad = _materializer_observation(
        observation_id="bad_authority",
        target_kind="packet_member_recompress_v1",
        saved_bytes=1,
        receiver_contract_satisfied=True,
        rate_positive=True,
    )
    bad["score_claim"] = True
    _write_json(
        artifact_root / "bad" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [bad],
        },
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="forbidden truthy"):
        build_frontier_rate_attack_feedback_refresh(
            repo_root=tmp_path,
            frontier_artifact_roots=(artifact_root,),
        )


def test_feedback_cycle_prefers_selected_pairset_acquisition_for_geometry_harvest(
    tmp_path: Path,
) -> None:
    candidate_id = "pairset_geometry_lowimpact_k003_habcdef1234"
    harvest = _write_json(
        tmp_path / ".omx" / "research" / "dqs1_local_first_harvest.json",
        {
            "schema": "dqs1_local_first_harvest.v1",
            **_false_authority(),
            "candidate_id": candidate_id,
            "local_cpu_advisory_path": "results/geometry/local_cpu_advisory.json",
            "harvested_at_utc": "20260525T010203Z",
            "authority": "false_authority_dqs1_local_first_harvest",
        },
    )
    preferred = _write_json(
        tmp_path / "refresh" / "dqs1_selected_pairset_acquisition.json",
        {
            "schema": "dqs1_selected_pairset_acquisition.v1",
            **_false_authority(),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "acquisition_id": candidate_id,
                    "selector_id": candidate_id,
                    "selector_kind": "pair_frame_geometry_low_impact_drop_many",
                    "selected_pair_indices": [1, 2, 112],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "pair_frame_geometry_low_impact_drop_many"
                    },
                    **_false_authority(),
                }
            ],
        },
    )
    fallback = _write_json(
        tmp_path / "pairset_acquisition.json",
        {"schema": "decoder_q_pairset_acquisition.v1", "candidates": []},
    )

    selected = select_pairset_acquisition_for_harvests(
        harvest_paths=(harvest,),
        repo_root=tmp_path,
        preferred_pairset_acquisition_path=preferred,
        fallback_pairset_acquisition_path=fallback,
    )

    assert selected == preferred.resolve()


def test_frontier_feedback_cli_writes_valid_followup_queue(tmp_path: Path) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_rate_attack_feedback_refresh.py",
            "--action-summary",
            str(action_summary),
            "--frontier-artifact-root",
            str(artifact_root),
            "--dqs1-observation-jsonl",
            str(dqs1_observations),
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(results_root),
            "--queue-id",
            "frontier_feedback_cli_unit",
            "--candidate-limit",
            "2",
            "--repair-palette",
            "fec6-fixed-k16",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    assert payload["receiver_repair_backlog_summary"]["row_count"] >= 4
    assert payload["receiver_repair_backlog_summary"][
        "queue_actionable_repair_count"
    ] >= 2
    assert payload["receiver_closed_correction_budget_summary"][
        "receiver_closed_saved_bytes_total"
    ] == 156
    assert payload["repair_dynamics_prior_summary"][
        "repair_dynamics_palette_prior_present"
    ] is True
    assert payload["repair_dynamics_prior_summary"]["repair_dynamics_prior_defaulted"] is False
    assert payload["repair_dynamics_prior_summary"]["mode_count"] == 16
    assert payload["repair_dynamics_prior_summary"]["zero_frame1_modes"] is True
    assert payload["targeted_component_correction_acquisition_summary"][
        "receiver_closed_saved_bytes_total"
    ] == 156
    assert payload["targeted_component_correction_acquisition_summary"][
        "queue_actionable_acquisition_count"
    ] >= 5
    assert payload["targeted_component_correction_acquisition_summary"][
        "repair_dynamics_prior_active"
    ] is True
    assert payload["targeted_component_correction_acquisition_summary"][
        "repair_dynamics_palette_probe_count"
    ] >= 3
    assert payload[
        "targeted_component_correction_materialization_request_summary"
    ]["row_count"] == 0
    assert payload[
        "targeted_component_correction_materialization_request_summary"
    ]["ready_for_budget_spend_count"] == 0
    assert payload[
        "targeted_component_correction_operation_chain_summary"
    ]["work_order_count"] == 0
    assert payload[
        "targeted_component_correction_chain_materializer_handoff_summary"
    ]["work_queue_row_count"] == 0
    assert payload["autonomous_chain_optimization_summary"]["chain_count"] >= 1
    assert "global_many_op_rate_distortion_receiver_campaign" in payload[
        "autonomous_chain_optimization_summary"
    ]["top_chain_ids"]
    queue_path = output_dir / "dqs1_followup_queue.json"
    bridge_path = output_dir / "materializer_feedback_bridge.json"
    receiver_repair_backlog_path = output_dir / "receiver_repair_backlog.json"
    receiver_closed_budget_path = output_dir / "receiver_closed_correction_budget.json"
    receiver_repair_queue_path = output_dir / "receiver_repair_queue.json"
    repair_dynamics_prior_path = output_dir / "repair_dynamics_palette_prior.json"
    targeted_component_acquisition_path = (
        output_dir / "targeted_component_correction_acquisition.json"
    )
    targeted_component_queue_path = (
        output_dir / "targeted_component_correction_queue.json"
    )
    targeted_component_response_harvest_path = (
        output_dir / "targeted_component_correction_response_harvest.json"
    )
    targeted_component_materialization_requests_path = (
        output_dir / "targeted_component_correction_materialization_requests.json"
    )
    targeted_component_operation_chain_work_orders_path = (
        output_dir / "targeted_component_correction_operation_chain_work_orders.json"
    )
    targeted_component_chain_materializer_handoff_path = (
        output_dir / "targeted_component_correction_chain_materializer_handoff.json"
    )
    targeted_component_chain_materializer_work_queue_path = (
        output_dir / "targeted_component_correction_chain_materializer_work_queue.json"
    )
    autonomous_chain_optimization_path = output_dir / "autonomous_chain_optimization.json"
    repair_budget_waterfill_queue_path = output_dir / "repair_budget_waterfill_queue.json"
    autonomous_chain_optimization_queue_path = (
        output_dir / "autonomous_chain_optimization_queue.json"
    )
    report_path = output_dir / "feedback_refresh_report.json"
    assert queue_path.exists()
    assert bridge_path.exists()
    assert receiver_repair_backlog_path.exists()
    assert receiver_closed_budget_path.exists()
    assert receiver_repair_queue_path.exists()
    assert repair_dynamics_prior_path.exists()
    assert targeted_component_acquisition_path.exists()
    assert targeted_component_queue_path.exists()
    assert targeted_component_response_harvest_path.exists()
    assert targeted_component_materialization_requests_path.exists()
    assert targeted_component_operation_chain_work_orders_path.exists()
    assert targeted_component_chain_materializer_handoff_path.exists()
    assert targeted_component_chain_materializer_work_queue_path.exists()
    assert autonomous_chain_optimization_path.exists()
    assert repair_budget_waterfill_queue_path.exists()
    assert autonomous_chain_optimization_queue_path.exists()
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifacts"]["dqs1_followup_queue"].endswith("dqs1_followup_queue.json")
    assert report["artifacts"]["receiver_repair_backlog"].endswith(
        "receiver_repair_backlog.json"
    )
    assert report["artifacts"]["receiver_repair_queue"].endswith(
        "receiver_repair_queue.json"
    )
    assert report["artifacts"]["receiver_closed_correction_budget"].endswith(
        "receiver_closed_correction_budget.json"
    )
    assert report["artifacts"]["repair_dynamics_palette_prior"].endswith(
        "repair_dynamics_palette_prior.json"
    )
    assert report["artifacts"]["targeted_component_correction_acquisition"].endswith(
        "targeted_component_correction_acquisition.json"
    )
    assert report["artifacts"]["targeted_component_correction_queue"].endswith(
        "targeted_component_correction_queue.json"
    )
    targeted_queue = json.loads(
        targeted_component_queue_path.read_text(encoding="utf-8")
    )
    repair_probe_steps = [
        step
        for experiment in targeted_queue["experiments"]
        for step in experiment["steps"]
        if step["id"].startswith("emit_repair_dynamics_palette_probe_matrix")
    ]
    assert repair_probe_steps
    assert repair_probe_steps[0]["command"][1] == (
        "tools/build_repair_dynamics_palette_probe_matrix.py"
    )
    assert repair_probe_steps[0]["postconditions"][0]["equals"] == (
        REPAIR_DYNAMICS_PALETTE_PROBE_MATRIX_SCHEMA
    )
    assert report["artifacts"][
        "targeted_component_correction_response_harvest"
    ].endswith("targeted_component_correction_response_harvest.json")
    assert report["artifacts"][
        "targeted_component_correction_materialization_requests"
    ].endswith("targeted_component_correction_materialization_requests.json")
    assert report["artifacts"][
        "targeted_component_correction_operation_chain_work_orders"
    ].endswith("targeted_component_correction_operation_chain_work_orders.json")
    assert report["artifacts"][
        "targeted_component_correction_chain_materializer_handoff"
    ].endswith("targeted_component_correction_chain_materializer_handoff.json")
    assert report["artifacts"][
        "targeted_component_correction_chain_materializer_work_queue"
    ].endswith("targeted_component_correction_chain_materializer_work_queue.json")
    assert report["artifacts"]["autonomous_chain_optimization"].endswith(
        "autonomous_chain_optimization.json"
    )
    assert report["artifacts"]["repair_budget_waterfill_queue"].endswith(
        "repair_budget_waterfill_queue.json"
    )
    assert report["artifacts"]["autonomous_chain_optimization_queue"].endswith(
        "autonomous_chain_optimization_queue.json"
    )
    assert report["artifacts"]["operation_materializer_bridge"].endswith(
        "operation_materializer_bridge.json"
    )
    assert report["artifacts"]["operation_materializer_backlog"].endswith(
        "operation_materializer_backlog.json"
    )
    assert report["artifacts"]["operation_materializer_contexts"].endswith(
        "operation_materializer_contexts.json"
    )
    assert report["artifacts"]["operation_materializer_work_queue"].endswith(
        "operation_materializer_work_queue.json"
    )
    assert report["artifacts"]["operation_chain_compiler_work_orders"].endswith(
        "operation_chain_compiler_work_orders.json"
    )
    assert report["artifacts"]["operation_chain_compiler_queue"].endswith(
        "operation_chain_compiler_queue.json"
    )
    assert report["operator_commands"]["validate_followup_queue"][0] == ".venv/bin/python"
    assert (
        report["operator_commands"]["validate_receiver_repair_queue"][0]
        == ".venv/bin/python"
    )
    assert report["operator_commands"]["init_receiver_repair_queue"] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["receiver_repair_queue"],
        "init",
    ]
    assert report["operator_commands"]["status_receiver_repair_queue"][-1] == "status"
    bounded_receiver_run = report["operator_commands"][
        "run_receiver_repair_queue_bounded_local"
    ]
    assert bounded_receiver_run[:4] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["receiver_repair_queue"],
    ]
    assert bounded_receiver_run[4:] == [
        "run-worker",
        "--execute",
        "--max-steps",
        "12",
        "--max-experiments",
        "4",
        "--max-parallel",
        "2",
    ]
    assert (
        report["operator_commands"]["validate_operation_chain_compiler_queue"][0]
        == ".venv/bin/python"
    )
    bounded_chain_run = report["operator_commands"][
        "run_operation_chain_compiler_queue_bounded_local"
    ]
    assert bounded_chain_run[:4] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["operation_chain_compiler_queue"],
    ]
    assert bounded_chain_run[4:] == [
        "run-worker",
        "--execute",
        "--max-steps",
        "16",
        "--max-experiments",
        "2",
        "--max-parallel",
        "2",
    ]
    assert (
        report["operator_commands"][
            "validate_targeted_component_correction_queue"
        ][0]
        == ".venv/bin/python"
    )
    assert report["operator_commands"]["inspect_autonomous_chain_optimization"] == [
        ".venv/bin/python",
        "-m",
        "json.tool",
        report["artifacts"]["autonomous_chain_optimization"],
    ]
    assert report["operator_commands"]["validate_repair_budget_waterfill_queue"] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["repair_budget_waterfill_queue"],
        "validate",
    ]
    repair_waterfill_run = report["operator_commands"][
        "run_repair_budget_waterfill_queue_bounded_local"
    ]
    assert repair_waterfill_run[:4] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["repair_budget_waterfill_queue"],
    ]
    assert report["operator_commands"]["validate_autonomous_chain_optimization_queue"] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["autonomous_chain_optimization_queue"],
        "validate",
    ]
    autonomous_run = report["operator_commands"][
        "run_autonomous_chain_optimization_queue_bounded_local"
    ]
    assert autonomous_run[:4] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["autonomous_chain_optimization_queue"],
    ]
    autonomous_artifact = json.loads(
        autonomous_chain_optimization_path.read_text(encoding="utf-8")
    )
    assert autonomous_artifact["schema"] == AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA
    _assert_false_authority(autonomous_artifact)
    autonomous_queue_artifact = json.loads(
        autonomous_chain_optimization_queue_path.read_text(encoding="utf-8")
    )
    repair_waterfill_queue_artifact = json.loads(
        repair_budget_waterfill_queue_path.read_text(encoding="utf-8")
    )
    assert repair_waterfill_queue_artifact["schema"] == "experiment_queue.v1"
    assert repair_waterfill_queue_artifact["experiments"][0]["metadata"][
        "pipeline_side"
    ] == "encoder_repair_allocator"
    _assert_false_authority(
        repair_waterfill_queue_artifact["experiments"][0]["metadata"]
    )
    assert autonomous_queue_artifact["schema"] == "experiment_queue.v1"
    autonomous_queue_metadata = autonomous_queue_artifact["experiments"][0]["metadata"]
    _assert_false_authority(autonomous_queue_metadata)
    if autonomous_queue_artifact["experiments"][0]["status"] == "queued":
        assert autonomous_queue_metadata["queue_actuation_ready"] is True
        assert autonomous_queue_metadata["missing_queue_artifact_keys"] == []
        assert autonomous_queue_metadata["child_queue_artifact_paths"]
    else:
        assert autonomous_queue_artifact["experiments"][0]["status"] == "frozen"
        assert autonomous_queue_metadata["queue_actuation_ready"] is False
        assert autonomous_queue_metadata["queue_actuation_blockers"]
    assert autonomous_queue_artifact["experiments"][0]["steps"][0]["command"][1] == (
        "tools/build_frontier_autonomous_chain_work_order.py"
    )
    if autonomous_queue_metadata["child_queue_artifact_paths"]:
        assert any(
            step["id"].startswith("validate_")
            for step in autonomous_queue_artifact["experiments"][0]["steps"]
        )
        assert any(
            step["id"].startswith("run_") and step["command"][4] == "run-worker"
            for step in autonomous_queue_artifact["experiments"][0]["steps"]
        )
    followup_queue = json.loads(queue_path.read_text(encoding="utf-8"))
    autonomous_metadata = followup_queue["experiments"][0]["metadata"][
        "frontier_autonomous_chain_optimization"
    ]
    assert autonomous_metadata["chain_count"] == autonomous_artifact["chain_count"]
    assert autonomous_metadata["registered_target_count"] == autonomous_artifact[
        "registered_target_count"
    ]
    assert autonomous_artifact["rows"][0]["repair_budget_waterfill_plan"][
        "exact_auth_eval_required_before_score_claim"
    ] is True
    assert report["operator_commands"]["init_targeted_component_correction_queue"] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["targeted_component_correction_queue"],
        "init",
    ]
    assert (
        report["operator_commands"]["status_targeted_component_correction_queue"][-1]
        == "status"
    )
    bounded_targeted_run = report["operator_commands"][
        "run_targeted_component_correction_queue_bounded_local"
    ]
    assert bounded_targeted_run[:4] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        report["artifacts"]["targeted_component_correction_queue"],
    ]
    assert bounded_targeted_run[4:] == [
        "run-worker",
        "--execute",
        "--max-steps",
        "21",
        "--max-experiments",
        "2",
        "--max-parallel",
        "3",
    ]
    assert (
        report["operator_commands"][
            "inspect_targeted_component_correction_response_harvest"
        ][0]
        == ".venv/bin/python"
    )
    assert (
        report["operator_commands"][
            "inspect_targeted_component_correction_materialization_requests"
        ][0]
        == ".venv/bin/python"
    )
    assert (
        report["operator_commands"][
            "inspect_targeted_component_correction_operation_chain_work_orders"
        ][0]
        == ".venv/bin/python"
    )
    assert (
        report["operator_commands"][
            "inspect_targeted_component_correction_chain_materializer_handoff"
        ][0]
        == ".venv/bin/python"
    )

    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr
    repair_validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(receiver_repair_queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repair_validate.returncode == 0, repair_validate.stderr
    correction_validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(targeted_component_queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert correction_validate.returncode == 0, correction_validate.stderr
    autonomous_queue_validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(autonomous_chain_optimization_queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert autonomous_queue_validate.returncode == 0, autonomous_queue_validate.stderr
    targeted_component_queue = json.loads(
        targeted_component_queue_path.read_text(encoding="utf-8")
    )
    targeted_component_response_harvest = json.loads(
        targeted_component_response_harvest_path.read_text(encoding="utf-8")
    )
    targeted_component_materialization_requests = json.loads(
        targeted_component_materialization_requests_path.read_text(encoding="utf-8")
    )
    assert targeted_component_response_harvest["schema"] == (
        TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA
    )
    assert targeted_component_response_harvest["row_count"] == 2
    assert targeted_component_response_harvest["ready_for_budget_spend_count"] == 0
    assert targeted_component_materialization_requests["schema"] == (
        TARGETED_COMPONENT_CORRECTION_MATERIALIZATION_REQUESTS_SCHEMA
    )
    assert targeted_component_materialization_requests["row_count"] == 0
    assert targeted_component_materialization_requests[
        "ready_for_budget_spend_count"
    ] == 0
    assert len(targeted_component_queue["experiments"]) == 1
    assert targeted_component_queue["selection_policy"]["policy"] == (
        "bounded_candidate_family_round_robin"
    )
    assert targeted_component_queue["selection_policy"]["selected_row_count"] == 2
    assert (
        targeted_component_queue["experiments"][0]["metadata"][
            "selected_acquisition_count"
        ]
        == 2
    )
    assert (
        targeted_component_queue["experiments"][0]["metadata"][
            "shared_component_response_reuse"
        ]
        is True
    )
    selected_correction_families = {
        request["correction_family"]
        for experiment in targeted_component_queue["experiments"]
        for request in experiment["metadata"]["correction_requests"]
    }
    assert selected_correction_families == {
        "repair_dynamics_frame0_palette_interaction_waterfill",
        "repair_dynamics_chroma_luma_bias_basis_expansion",
    }
    first_correction_step = targeted_component_queue["experiments"][0]["steps"][0]
    assert (
        first_correction_step["command"][1]
        == "tools/build_frontier_targeted_component_correction_work_order.py"
    )
    correction_work_order_command = [
        sys.executable,
        *first_correction_step["command"][1:],
    ]
    correction_work_order_first = subprocess.run(
        correction_work_order_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert correction_work_order_first.returncode == 0, correction_work_order_first.stderr
    correction_work_order_payload_first = json.loads(correction_work_order_first.stdout)
    assert correction_work_order_payload_first["bytes_written"] > 0
    assert (
        correction_work_order_payload_first["skipped_identical_existing_artifact"]
        is False
    )
    correction_work_order_path = Path(
        first_correction_step["command"][
            first_correction_step["command"].index("--work-order-out") + 1
        ]
    )
    if not correction_work_order_path.is_absolute():
        correction_work_order_path = REPO_ROOT / correction_work_order_path
    correction_work_order = json.loads(
        correction_work_order_path.read_text(encoding="utf-8")
    )
    assert (
        correction_work_order["schema"]
        == TARGETED_COMPONENT_CORRECTION_WORK_ORDER_SCHEMA
    )
    _assert_false_authority(correction_work_order)
    assert correction_work_order["repair_dynamics_prior_active"] is True
    assert correction_work_order["repair_dynamics_palette_prior"]["mode_count"] == 16
    repair_dynamics_hint = next(
        hint
        for hint in correction_work_order["command_hints"]
        if hint["action_id"] == "build_repair_dynamics_palette_probe_matrix"
    )
    assert "tools/build_repair_dynamics_palette_probe_matrix.py" in (
        repair_dynamics_hint["command_template"]
    )
    assert "--matrix-out" in repair_dynamics_hint["command_template"]
    assert (
        repair_dynamics_hint["output_contract"]
        == "false_authority_repair_dynamics_palette_probe_matrix"
    )
    assert correction_work_order["budget_spend_gate"]["budget_spend_allowed"] is False
    step_ids = [
        step["id"] for step in targeted_component_queue["experiments"][0]["steps"]
    ]
    assert step_ids[-1].startswith("harvest_targeted_component_correction_response_")
    work_order_paths = {
        request["work_order_path"]
        for experiment in targeted_component_queue["experiments"]
        for request in experiment["metadata"]["correction_requests"]
    }
    assert len(work_order_paths) == targeted_component_queue["selection_policy"][
        "selected_row_count"
    ]
    assert len(correction_work_order["candidate_family_rows"]) >= 5
    correction_work_order_retry = subprocess.run(
        correction_work_order_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert correction_work_order_retry.returncode == 0, correction_work_order_retry.stderr
    correction_work_order_payload_retry = json.loads(correction_work_order_retry.stdout)
    assert correction_work_order_payload_retry["bytes_written"] == 0
    assert (
        correction_work_order_payload_retry["skipped_identical_existing_artifact"]
        is True
    )
    repair_queue = json.loads(receiver_repair_queue_path.read_text(encoding="utf-8"))
    selected_sources = [
        experiment["metadata"]["source_operation_id"]
        for experiment in repair_queue["experiments"]
    ]
    actionable_sources = {
        row["source_operation_id"]
        for row in json.loads(
            receiver_repair_backlog_path.read_text(encoding="utf-8")
        )["rows"]
        if row["queue_actionable"] is True
    }
    if len(actionable_sources) >= len(selected_sources):
        assert len(selected_sources) == len(set(selected_sources))
    bridge_steps = [
        step
        for experiment in repair_queue["experiments"]
        for step in experiment["steps"]
        if str(step["id"]).startswith("run_exact_readiness_bridge")
    ]
    closure_steps = [
        step
        for experiment in repair_queue["experiments"]
        for step in experiment["steps"]
        if str(step["id"]).startswith("build_submission_runtime_closure")
    ]
    assert closure_steps
    closure_step = closure_steps[0]
    assert closure_step["command"][1] == "tools/build_materializer_submission_closure.py"
    assert "--closed-source-queue-out" in closure_step["command"]
    assert "--submission-dir-out" in closure_step["command"]
    assert "--closure-report-out" in closure_step["command"]
    assert any(
        postcondition["type"] == "json_false_authority"
        for postcondition in closure_step["postconditions"]
    )
    closed_queue_false_authority = [
        postcondition
        for postcondition in closure_step["postconditions"]
        if postcondition["type"] == "json_false_authority"
        and str(postcondition["path"]).endswith("closed_source_queue.json")
    ]
    assert closed_queue_false_authority
    assert "dispatch_ready" not in closed_queue_false_authority[0]["false_or_missing"]
    assert bridge_steps
    bridge_step = bridge_steps[0]
    assert bridge_step["command"][1] == "tools/run_materializer_exact_readiness_bridge.py"
    assert "--source-queue" in bridge_step["command"]
    bridge_source_queue = bridge_step["command"][
        bridge_step["command"].index("--source-queue") + 1
    ]
    assert bridge_source_queue.endswith("submission_closure/closed_source_queue.json")
    assert "--bridge-report-out" in bridge_step["command"]
    assert "--overwrite" in bridge_step["command"]
    assert "--force-recompute" in bridge_step["command"]
    assert "emit_receiver_repair_work_order" in bridge_step["requires"]
    assert closure_step["id"] in bridge_step["requires"]
    assert any(
        postcondition["type"] == "json_false_authority"
        for postcondition in bridge_step["postconditions"]
    )
    assert any(
        postcondition["type"] == "json_equals"
        and postcondition["key"] == "ready_for_exact_eval_dispatch"
        and postcondition["equals"] is False
        for postcondition in bridge_step["postconditions"]
    )
    assert any(
        experiment["metadata"]["exact_readiness_bridge_step_count"] > 0
        and experiment["metadata"]["submission_closure_step_count"] > 0
        and experiment["metadata"]["source_queue_paths"]
        for experiment in repair_queue["experiments"]
    )
    first_step = repair_queue["experiments"][0]["steps"][0]
    assert first_step["command"][1] == "tools/build_frontier_receiver_repair_work_order.py"
    work_order_command = [sys.executable, *first_step["command"][1:]]
    work_order_first = subprocess.run(
        work_order_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert work_order_first.returncode == 0, work_order_first.stderr
    work_order_payload_first = json.loads(work_order_first.stdout)
    assert work_order_payload_first["bytes_written"] > 0
    assert work_order_payload_first["skipped_identical_existing_artifact"] is False
    work_order_path = Path(
        first_step["command"][first_step["command"].index("--work-order-out") + 1]
    )
    if not work_order_path.is_absolute():
        work_order_path = REPO_ROOT / work_order_path
    assert work_order_path.exists()

    work_order_retry = subprocess.run(
        work_order_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert work_order_retry.returncode == 0, work_order_retry.stderr
    work_order_payload_retry = json.loads(work_order_retry.stdout)
    assert work_order_payload_retry["bytes_written"] == 0
    assert work_order_payload_retry["skipped_identical_existing_artifact"] is True


def test_frontier_feedback_cycle_harvests_batch_and_refreshes_queue(tmp_path: Path) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    results_root = tmp_path / "results"
    _write_receiver_closed_budget_signal(tmp_path, results_root=results_root)
    baseline = _write_json(
        tmp_path / "baseline.json",
        _advisory(
            archive_sha256="1" * 64,
            archive_size_bytes=178_592,
            raw_sha256="2" * 64,
            runtime_sha256="3" * 64,
        ),
    )
    advisory = _write_json(
        tmp_path / "candidate" / "local_cpu_advisory.json",
        _advisory(
            archive_sha256="4" * 64,
            archive_size_bytes=178_559,
            raw_sha256="5" * 64,
            runtime_sha256="6" * 64,
            seg=0.054,
        ),
    )
    harvest = _write_json(
        tmp_path / ".omx" / "research" / "dqs1_local_first_harvest.json",
        {
            "schema": "dqs1_local_first_harvest.v1",
            **_false_authority(),
            "candidate_id": "pairset_drop_one_rank023_pair0440",
            "candidate_archive_sha256": "4" * 64,
            "local_cpu_advisory_path": str(advisory),
            "local_score": 0.191,
            "projected_contest_score": 0.190,
            "conservative_projected_contest_score": 0.192,
            "recommended_action": "observe_only",
            "eureka_trigger": False,
            "eureka_margin": -1e-6,
            "authority": "false_authority_dqs1_local_first_harvest",
            "harvested_at_utc": "20260525T010203Z",
            "dispatch_blockers": ["exact_cpu_cuda_auth_eval_required"],
        },
    )
    _write_json(
        tmp_path
        / ".omx"
        / "research"
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        _eureka_signal(),
    )
    acquisition = _write_json(
        tmp_path / "pairset_acquisition.json",
        {
            "schema": "decoder_q_pairset_acquisition.v1",
            **_false_authority(),
            "candidates": [
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank023_pair0440",
                    "selector_id": "pairset_drop_one_rank023_pair0440",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 23,
                    "predicted_score_mean": 0.1919,
                    "selected_pair_indices": [1, 2, 440],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 23,
                        "dropped_pair_index": 440,
                    },
                },
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank024_pair0112",
                    "selector_id": "pairset_drop_one_rank024_pair0112",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 24,
                    "predicted_score_mean": 0.19191,
                    "selected_pair_indices": [1, 2, 112],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 24,
                        "dropped_pair_index": 112,
                    },
                },
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank025_pair0233",
                    "selector_id": "pairset_drop_one_rank025_pair0233",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 25,
                    "predicted_score_mean": 0.19192,
                    "selected_pair_indices": [1, 2, 233],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 25,
                        "dropped_pair_index": 233,
                    },
                },
            ],
        },
    )
    autopilot_result = _write_json(
        tmp_path / "autopilot_result.json",
        {
            "schema": AUTOPILOT_RESULT_SCHEMA,
            **_false_authority(),
            "queue_path": str(tmp_path / "initial_queue.json"),
            "execute": True,
            "stop_reason": "batch_harvested_waiting_for_portfolio_rebuild",
            "total_steps_started": 6,
            "candidates_harvested": 1,
            "rounds": [
                {
                    "harvests": [
                        {
                            **_false_authority(),
                            "candidate_id": "pairset_drop_one_rank023_pair0440",
                            "harvest_path": str(harvest),
                        }
                    ]
                }
            ],
        },
    )
    output_dir = tmp_path / "cycle_out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_frontier_rate_attack_feedback_cycle.py",
            "--action-summary",
            str(action_summary),
            "--frontier-artifact-root",
            str(artifact_root),
            "--frontier-artifact-root",
            str(tmp_path / ".omx" / "research"),
            "--autopilot-result-json",
            str(autopilot_result),
            "--pairset-acquisition",
            str(acquisition),
            "--baseline-advisory",
            str(baseline),
            "--baseline-archive-size-bytes",
            "178560",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(results_root),
            "--queue-id",
            "frontier_feedback_cycle_unit",
            "--post-harvest-queue-id",
            "frontier_feedback_cycle_unit_post",
            "--candidate-limit",
            "2",
            "--execute-auxiliary-queues",
            "--auxiliary-queue-max-steps",
            "1",
            "--auxiliary-queue-max-experiments",
            "1",
            "--auxiliary-queue-max-parallel",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["harvest_path_count"] == 1
    assert payload["initial_selected_candidate_ids"] == [
        "pairset_drop_one_rank023_pair0440",
        "pairset_drop_one_rank024_pair0112",
    ]
    assert payload["post_harvest_selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    assert payload["initial_auxiliary_queue_execution_summary"]["queue_count"] >= 3
    assert payload["initial_auxiliary_queue_execution_summary"][
        "executed_queue_count"
    ] >= 1
    assert payload["initial_auxiliary_queue_execution_summary"][
        "failed_queue_count"
    ] == 0
    assert payload[
        "initial_post_auxiliary_targeted_component_refresh_summary"
    ]["response_harvest_row_count"] >= 1
    assert payload[
        "initial_post_auxiliary_targeted_component_refresh_summary"
    ]["materialization_request_row_count"] == 0
    assert payload[
        "initial_targeted_component_correction_chain_materializer_handoff_summary"
    ]["work_queue_row_count"] == 0
    assert payload["initial_autonomous_chain_optimization_summary"][
        "chain_count"
    ] >= 1
    assert "global_many_op_rate_distortion_receiver_campaign" in payload[
        "initial_autonomous_chain_optimization_summary"
    ]["top_chain_ids"]
    assert payload["initial_targeted_drop_many_dqs1_child_queue_paths"] == []
    observation_jsonl = Path(payload["observation_jsonl"])
    if not observation_jsonl.is_absolute():
        observation_jsonl = REPO_ROOT / observation_jsonl
    assert observation_jsonl.exists()
    cycle_report = json.loads(
        (output_dir / "frontier_rate_attack_feedback_cycle.json").read_text(
            encoding="utf-8"
        )
    )
    assert cycle_report["schema"] == "frontier_rate_attack_feedback_cycle.v1"
    assert cycle_report["post_followup_eureka_planning"]["payload"][
        "signal_count"
    ] == 1
    assert (
        output_dir / "post_followup_local_cpu_eureka_planning.json"
    ).is_file()
    initial_artifacts = cycle_report["initial_refresh"]["artifacts"]
    assert initial_artifacts["targeted_component_correction_acquisition"].endswith(
        "targeted_component_correction_acquisition.json"
    )
    assert initial_artifacts["targeted_component_correction_queue"].endswith(
        "targeted_component_correction_queue.json"
    )
    assert initial_artifacts[
        "targeted_component_correction_response_harvest"
    ].endswith("targeted_component_correction_response_harvest.json")
    assert initial_artifacts[
        "targeted_component_correction_materialization_requests"
    ].endswith("targeted_component_correction_materialization_requests.json")
    assert initial_artifacts[
        "targeted_component_correction_operation_chain_work_orders"
    ].endswith("targeted_component_correction_operation_chain_work_orders.json")
    assert initial_artifacts[
        "targeted_component_correction_chain_materializer_handoff"
    ].endswith("targeted_component_correction_chain_materializer_handoff.json")
    assert initial_artifacts["autonomous_chain_optimization"].endswith(
        "autonomous_chain_optimization.json"
    )
    assert initial_artifacts["repair_budget_waterfill_queue"].endswith(
        "repair_budget_waterfill_queue.json"
    )
    initial_feedback_report = json.loads(
        (REPO_ROOT / initial_artifacts["feedback_refresh_report"]).read_text(
            encoding="utf-8"
        )
    )
    initial_dqs1_queue = json.loads(
        (REPO_ROOT / initial_feedback_report["artifacts"]["dqs1_followup_queue"]).read_text(
            encoding="utf-8"
        )
    )
    assert initial_dqs1_queue["controls"]["max_concurrency"]["local_io_heavy"] <= 1
    assert (
        initial_feedback_report["operator_commands"][
            "run_targeted_component_correction_queue_bounded_local"
        ][4]
        == "run-worker"
    )
    assert cycle_report["initial_refresh"]["auxiliary_queue_execution"][
        "execute_auxiliary_queues"
    ] is True
    assert cycle_report["initial_refresh"]["auxiliary_queue_execution"][
        "failed_queue_count"
    ] == 0
    post_auxiliary_refresh = cycle_report["initial_refresh"][
        "post_auxiliary_targeted_component_refresh"
    ]
    assert post_auxiliary_refresh["schema"] == (
        "frontier_rate_attack_post_auxiliary_targeted_component_refresh.v1"
    )
    _assert_false_authority(post_auxiliary_refresh)
    assert post_auxiliary_refresh["response_harvest_row_count"] >= 1
    assert post_auxiliary_refresh["materialization_request_row_count"] == 0
    assert post_auxiliary_refresh["artifacts"][
        "targeted_component_correction_response_harvest"
    ].endswith("post_auxiliary_targeted_component_correction_response_harvest.json")
    assert {
        row["artifact_key"]
        for row in cycle_report["initial_refresh"]["auxiliary_queue_execution"][
            "rows"
        ]
    }.issuperset(
        {
            "receiver_repair_queue",
            "operation_chain_compiler_queue",
            "targeted_component_correction_queue",
        }
    )
    assert (
        initial_feedback_report["operator_commands"][
            "run_receiver_repair_queue_bounded_local"
        ][4:]
        == [
            "run-worker",
            "--execute",
            "--max-steps",
            "12",
            "--max-experiments",
            "4",
            "--max-parallel",
            "2",
        ]
    )
    assert (
        initial_feedback_report["operator_commands"][
            "run_operation_chain_compiler_queue_bounded_local"
        ][4:]
        == [
            "run-worker",
            "--execute",
            "--max-steps",
            "16",
            "--max-experiments",
            "2",
            "--max-parallel",
            "2",
        ]
    )
    assert initial_artifacts["operation_materializer_bridge"].endswith(
        "operation_materializer_bridge.json"
    )
    assert initial_artifacts["operation_materializer_backlog"].endswith(
        "operation_materializer_backlog.json"
    )
    assert initial_artifacts["operation_materializer_contexts"].endswith(
        "operation_materializer_contexts.json"
    )
    assert initial_artifacts["operation_materializer_work_queue"].endswith(
        "operation_materializer_work_queue.json"
    )
    assert initial_artifacts["operation_chain_compiler_work_orders"].endswith(
        "operation_chain_compiler_work_orders.json"
    )
    assert initial_artifacts["operation_chain_compiler_queue"].endswith(
        "operation_chain_compiler_queue.json"
    )
    assert initial_artifacts[
        "targeted_component_correction_chain_materializer_work_queue"
    ].endswith("targeted_component_correction_chain_materializer_work_queue.json")
    assert initial_artifacts["repair_budget_waterfill_queue"].endswith(
        "repair_budget_waterfill_queue.json"
    )
    assert initial_artifacts["autonomous_chain_optimization_queue"].endswith(
        "autonomous_chain_optimization_queue.json"
    )
    operation_chain_orders = json.loads(
        (
            REPO_ROOT / initial_artifacts["operation_chain_compiler_work_orders"]
        ).read_text(encoding="utf-8")
    )
    assert operation_chain_orders["schema"] == (
        "frontier_rate_attack_operation_chain_compiler_work_orders.v1"
    )
    _assert_false_authority(operation_chain_orders)
    operation_chain_queue = json.loads(
        (
            REPO_ROOT / initial_artifacts["operation_chain_compiler_queue"]
        ).read_text(encoding="utf-8")
    )
    assert operation_chain_queue["schema"] == "experiment_queue.v1"
    _assert_false_authority(operation_chain_queue["experiments"][0]["metadata"])
    operation_work_queue = json.loads(
        (
            REPO_ROOT / initial_artifacts["operation_materializer_work_queue"]
        ).read_text(encoding="utf-8")
    )
    assert operation_work_queue["schema"] == "byte_shaving_materializer_work_queue.v1"
    assert operation_work_queue["row_count"] >= 2
    assert operation_work_queue["blocked_row_count"] >= 1
    targeted_validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            initial_artifacts["targeted_component_correction_queue"],
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert targeted_validate.returncode == 0, targeted_validate.stderr
    targeted_queue = json.loads(
        (
            REPO_ROOT / initial_artifacts["targeted_component_correction_queue"]
        ).read_text(encoding="utf-8")
    )
    assert targeted_queue["experiments"]
    assert targeted_queue["experiments"][0]["steps"][0]["command"][1] == (
        "tools/build_frontier_targeted_component_correction_work_order.py"
    )
    targeted_response_harvest = json.loads(
        (
            REPO_ROOT
            / initial_artifacts["targeted_component_correction_response_harvest"]
        ).read_text(encoding="utf-8")
    )
    assert targeted_response_harvest["schema"] == (
        "frontier_rate_attack_targeted_component_correction_response_harvest.v1"
    )
    _assert_false_authority(targeted_response_harvest)
    targeted_materialization_requests = json.loads(
        (
            REPO_ROOT
            / initial_artifacts[
                "targeted_component_correction_materialization_requests"
            ]
        ).read_text(encoding="utf-8")
    )
    assert targeted_materialization_requests["schema"] == (
        "frontier_rate_attack_targeted_component_correction_materialization_requests.v1"
    )
    _assert_false_authority(targeted_materialization_requests)
    targeted_operation_chain_work_orders = json.loads(
        (
            REPO_ROOT
            / initial_artifacts[
                "targeted_component_correction_operation_chain_work_orders"
            ]
        ).read_text(encoding="utf-8")
    )
    assert targeted_operation_chain_work_orders["schema"] == (
        "frontier_rate_attack_operation_chain_compiler_work_orders.v1"
    )
    _assert_false_authority(targeted_operation_chain_work_orders)
    targeted_chain_handoff = json.loads(
        (
            REPO_ROOT
            / initial_artifacts[
                "targeted_component_correction_chain_materializer_handoff"
            ]
        ).read_text(encoding="utf-8")
    )
    assert targeted_chain_handoff["schema"] == (
        "frontier_rate_attack_targeted_component_correction_chain_materializer_handoff.v1"
    )
    _assert_false_authority(targeted_chain_handoff)
    assert targeted_chain_handoff["work_queue_row_count"] == 0
    initial_repair_waterfill_queue = json.loads(
        (
            REPO_ROOT / initial_artifacts["repair_budget_waterfill_queue"]
        ).read_text(encoding="utf-8")
    )
    assert initial_repair_waterfill_queue["schema"] == "experiment_queue.v1"
    assert initial_repair_waterfill_queue["experiments"][0]["metadata"][
        "pipeline_side"
    ] == "encoder_repair_allocator"
    _assert_false_authority(
        initial_repair_waterfill_queue["experiments"][0]["metadata"]
    )
    initial_autonomous_chain = json.loads(
        (
            REPO_ROOT / initial_artifacts["autonomous_chain_optimization"]
        ).read_text(encoding="utf-8")
    )
    assert initial_autonomous_chain["schema"] == AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA
    _assert_false_authority(initial_autonomous_chain)
    initial_autonomous_metadata = initial_dqs1_queue["experiments"][0]["metadata"][
        "frontier_autonomous_chain_optimization"
    ]
    assert initial_autonomous_metadata["chain_count"] == initial_autonomous_chain[
        "chain_count"
    ]
    assert initial_autonomous_metadata["registered_target_count"] == (
        initial_autonomous_chain["registered_target_count"]
    )
    assert "fit_segnet_posenet_repair_waterfill_policy" in {
        action["id"]
        for row in initial_autonomous_chain["rows"]
        for action in row["scheduler_actions"]
    }
    initial_autonomous_queue = json.loads(
        (
            REPO_ROOT / initial_artifacts["autonomous_chain_optimization_queue"]
        ).read_text(encoding="utf-8")
    )
    assert initial_autonomous_queue["schema"] == "experiment_queue.v1"
    initial_autonomous_queue_experiment = initial_autonomous_queue["experiments"][0]
    initial_autonomous_queue_metadata = initial_autonomous_queue_experiment["metadata"]
    _assert_false_authority(initial_autonomous_queue_metadata)
    if initial_autonomous_queue_experiment["status"] == "queued":
        assert initial_autonomous_queue_metadata["queue_actuation_ready"] is True
        assert initial_autonomous_queue_metadata["missing_queue_artifact_keys"] == []
        assert initial_autonomous_queue_metadata["child_queue_artifact_paths"]
        assert any(
            step["id"].startswith("validate_")
            for step in initial_autonomous_queue_experiment["steps"]
        )
        assert any(
            step["id"].startswith("run_") and step["command"][4] == "run-worker"
            for step in initial_autonomous_queue_experiment["steps"]
        )
    else:
        assert initial_autonomous_queue_experiment["status"] == "frozen"
        assert initial_autonomous_queue_metadata["queue_actuation_ready"] is False
        assert initial_autonomous_queue_metadata["queue_actuation_blockers"]
    assert (
        initial_feedback_report["operator_commands"][
            "inspect_targeted_component_correction_chain_materializer_handoff"
        ][0]
        == ".venv/bin/python"
    )
    assert (
        initial_feedback_report["operator_commands"][
            "inspect_autonomous_chain_optimization"
        ][0]
        == ".venv/bin/python"
    )
    assert (
        initial_feedback_report["operator_commands"][
            "validate_repair_budget_waterfill_queue"
        ][0]
        == ".venv/bin/python"
    )
    assert (
        initial_feedback_report["operator_commands"][
            "validate_autonomous_chain_optimization_queue"
        ][0]
        == ".venv/bin/python"
    )
    component = cycle_report["post_harvest_refresh"]["pairset_component_marginal"]
    assert component["schema"] == "frontier_rate_attack_pairset_component_marginal_bundle.v1"
    assert component["active"] is True
    assert component["training_row_count"] == 1
    assert cycle_report["post_harvest_refresh"]["artifacts"][
        "feedback_refresh_report"
    ].endswith("feedback_refresh_report.json")
    assert cycle_report["post_harvest_refresh"]["queue_summary"]["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    assert "post_harvest_component_marginal_refresh/action_summary.json" in (
        cycle_report["post_harvest_refresh"]["artifacts"]["feedback_refresh_report"]
        or ""
    ) or cycle_report["post_harvest_refresh"]["pairset_component_marginal"][
        "action_summary_json"
    ].endswith("post_harvest_component_marginal_refresh/action_summary.json")
    assert cycle_report["post_harvest_refresh"]["queue_validate"]["valid"] is True
    assert "dynamic_observation_jsonl_to_refreshed_dqs1_queue" in cycle_report[
        "integration_edges"
    ]
    assert "dynamic_observation_jsonl_to_pairset_component_marginal_model" in cycle_report[
        "integration_edges"
    ]
    assert (
        "exact_readiness_bridge_to_receiver_repair_backlog_and_correction_budget"
        in cycle_report["integration_edges"]
    )
    assert (
        "receiver_closed_correction_acquisition_to_local_component_correction_queue"
        in cycle_report["integration_edges"]
    )
    assert (
        "targeted_component_correction_queue_to_response_harvest_and_materialization_requests"
        in cycle_report["integration_edges"]
    )
    assert (
        "targeted_component_materialization_requests_to_operation_chain_queue"
        in cycle_report["integration_edges"]
    )
    assert (
        "targeted_component_operation_chain_to_materializer_handoff"
        in cycle_report["integration_edges"]
    )
    assert (
        "targeted_operation_chain_queue_to_targeted_drop_many_child_queue"
        in cycle_report["integration_edges"]
    )
    assert (
        "autonomous_chain_optimization_to_queue_owned_many_op_plan"
        in cycle_report["integration_edges"]
    )
    assert (
        "receiver_closed_rate_budget_to_encoder_repair_waterfill_queue"
        in cycle_report["integration_edges"]
    )
    assert (
        "autonomous_chain_optimization_to_local_child_queue_actuation"
        in cycle_report["integration_edges"]
    )
    assert (
        "many_op_plan_to_component_replay_and_exact_readiness_bridge"
        in cycle_report["integration_edges"]
    )
    assert (
        "bounded_auxiliary_queue_artifacts_to_local_execution_trace"
        in cycle_report["integration_edges"]
    )
    assert (
        "bounded_auxiliary_targeted_response_reharvest_to_materialization_chain"
        in cycle_report["integration_edges"]
    )
    assert (
        "operation_portfolio_to_materializer_backlog_context_work_queue"
        in cycle_report["integration_edges"]
    )


def test_frontier_feedback_cycle_refuses_truthy_autopilot_authority(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": AUTOPILOT_RESULT_SCHEMA,
        **_false_authority(),
        "score_claim": True,
        "rounds": [],
    }

    with pytest.raises(FrontierRateAttackFeedbackCycleError, match="forbidden truthy"):
        harvest_paths_from_autopilot_payload(payload, repo_root=tmp_path)
