# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "all_lanes_operator_briefing_gate_test",
        ALL_LANES,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_briefing_payload() -> dict[str, object]:
    return {
        "dispatch_claim_summary": {
            "active_count": 0,
            "stale_nonterminal_count": 0,
            "unparsable_timestamp_count": 0,
            "invalid_lane_id_count": 0,
        },
        "dispatch_claim_historical_summary": {
            "unparsable_timestamp_count": 0,
            "invalid_lane_id_count": 0,
        },
        "dispatch_readiness": {
            "schema": "pact.operator_dispatch_readiness.v1",
            "phase_1_exact_eval_packets": {"status": "PENDING"},
            "phase_6d_frontier_feedback_cycle": {
                "status": "PENDING",
                "reason": "no frontier feedback cycle or queue refresh artifact found",
                "cycle_report_count": 0,
                "refresh_report_count": 0,
                "ready_local_execution_count": 0,
                "post_harvest_queue_count": 0,
                "next_command": (
                    ".venv/bin/python "
                    "tools/run_frontier_rate_attack_feedback_cycle.py "
                    "--frontier-artifact-root .omx/research --candidate-limit 4"
                ),
            },
        },
        "frontier_feedback_cycle": {
            "schema": "pact.frontier_feedback_cycle_summary.v1",
            "cycle_tool": "tools/run_frontier_rate_attack_feedback_cycle.py",
            "cycle_tool_exists": True,
            "status": "PENDING",
            "reason": "no frontier feedback cycle or queue refresh artifact found",
            "cycle_report_count": 0,
            "refresh_report_count": 0,
            "ready_local_execution_count": 0,
            "post_harvest_queue_count": 0,
            "latest_cycle": {},
            "latest_refresh": {},
            "error_count": 0,
            "next_command": (
                ".venv/bin/python "
                "tools/run_frontier_rate_attack_feedback_cycle.py "
                "--frontier-artifact-root .omx/research --candidate-limit 4"
            ),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
        "exact_eval_packets": [],
        "non_dispatchable_readiness_artifacts": [
            {
                "kind": "fixture_blocked_readiness",
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
            }
        ],
        "l5_v2_frontier_readiness": {
            "schema": "pact.l5_v2_frontier_readiness.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "measurement_schedule_score_claim": False,
            "measurement_schedule_promotion_eligible": False,
            "measurement_schedule_ready_for_exact_eval_dispatch": False,
            "asymptotic_pursuit_candidate_count": 1,
            "asymptotic_pursuit_candidates": [
                {
                    "candidate_id": "z6_z7_z8_predictive_coding_world_models",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "ready_for_paid_dispatch": False,
                    "ready_for_l1_scaffold_dispatch": False,
                    "ready_for_l1_build": True,
                    "ready_for_l1_build_semantics": (
                        "ready_to_start_l1_scaffold_work_only_not_scaffold_ready"
                    ),
                    "local_ledger_present": True,
                    "local_ledger_sha256": "a" * 64,
                    "lane_registry_registered": True,
                    "l5_v2_asymptotic_next_action_status": {
                        "schema": "l5_v2_asymptotic_next_action_status_v1",
                        "candidate_id": (
                            "z6_z7_z8_predictive_coding_world_models"
                        ),
                        "lane_id": (
                            "lane_time_traveler_l5_z6_z7_z8_predictive_"
                            "coding_world_models_scoping_design_20260516"
                        ),
                        "ledger_present": True,
                        "ledger_sha256": "a" * 64,
                        "lane_registry_registered": True,
                        "canonical_replacement_lane_id": "",
                        "canonical_replacement_lane_registered": False,
                        "expected_first_artifact_status": [],
                        "expected_first_artifacts_all_present": False,
                        "next_prerequisite_status": {
                            "status": "pending",
                            "ready_for_l1_build": True,
                            "ready_for_l1_scaffold_dispatch": False,
                        },
                        "ready_for_l1_build_semantics": (
                            "ready_to_start_l1_scaffold_work_only_not_"
                            "scaffold_ready"
                        ),
                    },
                }
            ],
            "l5_v2_asymptotic_next_action_status": [
                {
                    "schema": "l5_v2_asymptotic_next_action_status_v1",
                    "candidate_id": "z6_z7_z8_predictive_coding_world_models",
                    "lane_id": (
                        "lane_time_traveler_l5_z6_z7_z8_predictive_"
                        "coding_world_models_scoping_design_20260516"
                    ),
                    "ledger_present": True,
                    "ledger_sha256": "a" * 64,
                    "lane_registry_registered": True,
                    "canonical_replacement_lane_id": "",
                    "canonical_replacement_lane_registered": False,
                    "expected_first_artifact_status": [],
                    "expected_first_artifacts_all_present": False,
                    "next_prerequisite_status": {
                        "status": "pending",
                        "ready_for_l1_build": True,
                        "ready_for_l1_scaffold_dispatch": False,
                    },
                    "ready_for_l1_build_semantics": (
                        "ready_to_start_l1_scaffold_work_only_not_scaffold_ready"
                    ),
                }
            ],
            "tt5l_campaign_readiness": {
                "schema": "l5_v2_tt5l_campaign_readiness_v1",
                "non_pr106_staircase_priority": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dykstra_feasibility_artifact_valid": False,
                "dykstra_feasibility_status": {
                    "schema": "l5_v2_tt5l_dykstra_feasibility_status_v1",
                    "artifact_valid": False,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "blockers": ["tt5l_dykstra_feasibility_artifact_missing"],
                },
                "sideinfo_gate_evidence_valid": False,
                "probe_gate_evidence_valid": False,
                "paired_axis_plan_evidence_valid": False,
                "sideinfo_effect_curve_allowed": False,
                "first_anchor_timing_smoke_allowed": False,
                "next_non_pr106_l5_action": {
                    "action_id": "run_tt5l_dykstra_score_axis_sanity",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": ["tt5l_dykstra_feasibility_artifact_missing"],
            },
            "target_rows_are_fail_fast_only": True,
            "next_exact_eval_target_count": 1,
            "next_exact_eval_targets_sample": [
                {
                    "lane_id": "pr106_packetir_fixture_contest_cpu",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
                    "command_template": (
                        ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
                        "--archive a.zip --expected-runtime-tree-sha256 auto "
                        "--skip-axis-if-promotable-anchor-exists"
                    ),
                    "dispatch_status": (
                        "requires_claim_lane_dispatch_before_provider_launch"
                    ),
                }
            ],
        },
    }


def test_operator_briefing_dispatch_gate_accepts_blocked_score_plausible_rows() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [
            {
                "lane_id": "lane_pr106_stacked",
                "one_liner": "--env STACKED_LATENT_ARCHIVE=<path/to/archive.zip>",
                "gate_condition": "requires sister empirical landings",
                "score_target_routing": {"active": True},
                "dispatch_routing": {
                    "active": False,
                    "status": "dispatch_gate_blocked",
                    "blockers": [
                        "gate_condition_not_satisfied",
                        "operator_one_liner_has_unresolved_placeholders",
                    ],
                },
                "ready_for_operator_dispatch": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "active_composition_lanes": [],
    }

    assert module._operator_briefing_dispatch_failures(payload) == []


def test_operator_briefing_dispatch_gate_rejects_placeholder_active_row() -> None:
    module = _load_all_lanes_module()
    row = {
        "lane_id": "lane_pr106_stacked",
        "one_liner": "--env STACKED_LATENT_ARCHIVE=<path/to/archive.zip>",
        "dispatch_routing": {"active": True, "status": "dispatch_active", "blockers": []},
        "ready_for_operator_dispatch": True,
        "ready_for_exact_eval_dispatch": False,
    }
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [row],
        "active_composition_lanes": [row],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "composition_lanes:lane_pr106_stacked:active_with_operator_placeholder" in failures
    assert "active_composition_lanes:lane_pr106_stacked:active_row_has_placeholder" in failures


def test_operator_briefing_dispatch_gate_rejects_active_list_mismatch() -> None:
    module = _load_all_lanes_module()
    row = {
        "lane_id": "lane_pr106_stacked",
        "one_liner": "echo ready",
        "dispatch_routing": {"active": False, "status": "dispatch_gate_blocked", "blockers": ["x"]},
        "ready_for_operator_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
    }
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [row],
        "active_composition_lanes": [row],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "active_composition_lanes:lane_pr106_stacked:active_row_not_dispatch_active" in failures
    assert "active_composition_lanes:lane_pr106_stacked:active_row_not_operator_ready" in failures
    assert any(
        failure.startswith("active_composition_lanes_does_not_match_dispatch_routing")
        for failure in failures
    )


def test_operator_briefing_dispatch_gate_rejects_invalid_claim_summary() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "dispatch_claim_summary": {
            "active_count": 0,
            "stale_nonterminal_count": 0,
            "unparsable_timestamp_count": 0,
            "invalid_lane_id_count": 1,
        },
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "dispatch_claim_summary:invalid_lane_id_count:1" in failures


def test_operator_briefing_dispatch_gate_rejects_inverse_scorer_false_readiness() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    payload["non_dispatchable_readiness_artifacts"] = [
        {
            "kind": "inverse_scorer_cell_candidate_chain",
            "receiver_contract_satisfied": True,
            "inflate_parity_satisfied": False,
            "ready_for_exact_eval_dispatch": True,
            "score_claim": True,
            "promotion_eligible": True,
            "rank_or_kill_eligible": True,
            "dispatch_blockers": ["exact_auth_eval_required_before_score_claim"],
        }
    ]

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "non_dispatchable_readiness_artifacts:"
        "inverse_scorer_cell_candidate_chain:ready_for_exact_eval_dispatch_not_false"
    ) in failures
    assert (
        "non_dispatchable_readiness_artifacts:"
        "inverse_scorer_cell_candidate_chain:score_claim_not_false"
    ) in failures
    assert (
        "non_dispatchable_readiness_artifacts:"
        "inverse_scorer_cell_candidate_chain:promotion_eligible_true"
    ) in failures
    assert (
        "non_dispatchable_readiness_artifacts:"
        "inverse_scorer_cell_candidate_chain:rank_or_kill_eligible_true"
    ) in failures
    assert (
        "non_dispatchable_readiness_artifacts:"
        "inverse_scorer_cell_candidate_chain:missing_parity_blocker"
    ) in failures


def test_operator_briefing_dispatch_gate_requires_inverse_scorer_exact_auth_blocker() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    payload["non_dispatchable_readiness_artifacts"] = [
        {
            "kind": "inverse_scorer_cell_candidate_chain",
            "receiver_contract_satisfied": True,
            "inflate_parity_satisfied": True,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "dispatch_blockers": [],
        }
    ]

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "non_dispatchable_readiness_artifacts:"
        "inverse_scorer_cell_candidate_chain:missing_exact_auth_blocker"
    ) in failures


def test_operator_briefing_dispatch_gate_rejects_missing_structured_readiness() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "dispatch_readiness": {
            "schema": "pact.operator_dispatch_readiness.v1",
            "phase_1_exact_eval_packets": {"status": "READY"},
        },
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
        "exact_eval_packets": [
            {
                "lane_id": "terminal_packet",
                "dispatch_action": "terminal_exact_eval_evidence_stop",
                "repeat_dispatch_allowed": False,
                "ready_for_submit": False,
                "commands": {},
            }
        ],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "dispatch_readiness:phase_1_ready_while_all_exact_packets_blocked" in failures


def test_operator_briefing_dispatch_gate_rejects_l5_authority_leak() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    l5["ready_for_exact_eval_dispatch"] = True
    l5["l5_ready_for_score_or_rank_dispatch"] = True
    l5["l5_ready_for_dispatch"] = True
    l5["target_rows_are_fail_fast_only"] = False
    l5["next_exact_eval_targets_sample"] = [
        {
            "lane_id": "pr106_packetir_bad",
            "score_claim": True,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": True,
            "dispatch_status": "ready",
        }
    ]
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "l5_v2_frontier_readiness:ready_for_exact_eval_dispatch_not_false" in failures
    assert (
        "l5_v2_frontier_readiness:"
        "l5_ready_for_score_or_rank_dispatch_true_without_top_level_authority"
        in failures
    )
    assert (
        "l5_v2_frontier_readiness:"
        "l5_ready_for_dispatch_true_without_top_level_authority"
        in failures
    )
    assert "l5_v2_frontier_readiness:target_rows_not_fail_fast_only" in failures
    assert "l5_v2_frontier_readiness:target_0:score_claim_not_false" in failures
    assert (
        "l5_v2_frontier_readiness:target_0:ready_for_exact_eval_dispatch_not_false"
        in failures
    )
    assert (
        "l5_v2_frontier_readiness:target_0:dispatch_status_not_claim_gated"
        in failures
    )
    assert (
        "l5_v2_frontier_readiness:target_0:paired_dispatch_tool_not_canonical"
        in failures
    )


def test_operator_briefing_dispatch_gate_rejects_l5_missing_tt5l_campaign() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    l5.pop("tt5l_campaign_readiness")
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:tt5l_campaign_missing_or_not_object"
        in failures
    )


def test_operator_briefing_dispatch_gate_rejects_tt5l_timing_without_probe_plan() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    tt5l = dict(l5["tt5l_campaign_readiness"])  # type: ignore[index]
    tt5l["dykstra_feasibility_artifact_valid"] = True
    tt5l["dykstra_feasibility_status"] = {
        **tt5l["dykstra_feasibility_status"],  # type: ignore[arg-type]
        "artifact_valid": True,
    }
    tt5l["sideinfo_gate_evidence_valid"] = True
    tt5l["sideinfo_effect_curve_allowed"] = True
    tt5l["first_anchor_timing_smoke_allowed"] = True
    tt5l["next_non_pr106_l5_action"] = {
        "action_id": "materialize_tt5l_exact_or_diagnostic_anchor_pair",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    l5["tt5l_campaign_readiness"] = tt5l
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:"
        "tt5l_timing_smoke_without_dykstra_move_level_sideinfo_probe_paired_axis_plan"
        in failures
    )
    assert "l5_v2_frontier_readiness:tt5l_dykstra_status_validity_mismatch" not in failures


def test_operator_briefing_dispatch_gate_allows_tt5l_timing_before_artifact() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    tt5l = dict(l5["tt5l_campaign_readiness"])  # type: ignore[index]
    tt5l["dykstra_feasibility_artifact_valid"] = True
    tt5l["dykstra_feasibility_status"] = {
        **tt5l["dykstra_feasibility_status"],  # type: ignore[arg-type]
        "artifact_valid": True,
    }
    tt5l["move_level_feasibility_artifact_valid"] = True
    tt5l["sideinfo_gate_evidence_valid"] = True
    tt5l["probe_gate_evidence_valid"] = True
    tt5l["paired_axis_plan_evidence_valid"] = True
    tt5l["sideinfo_effect_curve_allowed"] = True
    tt5l["sideinfo_effect_curve_artifact_valid"] = True
    tt5l["first_anchor_timing_smoke_artifact_valid"] = False
    tt5l["first_anchor_timing_smoke_allowed"] = True
    tt5l["next_non_pr106_l5_action"] = {
        "action_id": "materialize_tt5l_first_anchor_timing_smoke_artifact",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    l5["tt5l_campaign_readiness"] = tt5l
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert not any("tt5l_timing_smoke_without" in failure for failure in failures)
    assert "l5_v2_frontier_readiness:tt5l_dykstra_status_validity_mismatch" not in failures


def test_operator_briefing_dispatch_gate_rejects_l5_gate_probe_without_tt5l_preconditions() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    l5["l5_ready_for_gate_probe_dispatch"] = True
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:"
        "gate_probe_dispatch_without_tt5l_cargo_cult_preconditions"
        in failures
    )


def test_operator_briefing_dispatch_gate_rejects_sideinfo_curve_without_dykstra() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    tt5l = dict(l5["tt5l_campaign_readiness"])  # type: ignore[index]
    tt5l["sideinfo_effect_curve_allowed"] = True
    l5["tt5l_campaign_readiness"] = tt5l
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:"
        "tt5l_sideinfo_effect_curve_without_dykstra_move_level_and_sideinfo"
        in failures
    )


def test_operator_briefing_dispatch_gate_rejects_tt5l_pr106_next_action() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    tt5l = dict(l5["tt5l_campaign_readiness"])  # type: ignore[index]
    tt5l["next_non_pr106_l5_action"] = {
        "action_id": "PR106_packetir_local_minimum_retry",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    l5["tt5l_campaign_readiness"] = tt5l
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:tt5l_next_action_mentions_pr106" in failures
    )
    assert (
        "l5_v2_frontier_readiness:tt5l_missing_dykstra_not_first_action"
        in failures
    )


def test_operator_briefing_dispatch_gate_rejects_l5_single_axis_modal_leak() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    l5["next_exact_eval_targets_sample"] = [
        {
            "lane_id": "pr106_packetir_bad",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "paired_dispatch_tool": "experiments/modal_auth_eval_cpu.py",
            "command_template": (
                ".venv/bin/modal run experiments/modal_auth_eval_cpu.py "
                "--expected-runtime-tree-sha256 "
                "<AXIS_SPECIFIC_MODAL_UPLOADED_RUNTIME_TREE_SHA256>"
            ),
            "dispatch_status": "requires_claim_lane_dispatch_before_provider_launch",
        }
    ]
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:target_0:paired_dispatch_tool_not_canonical"
        in failures
    )
    assert (
        "l5_v2_frontier_readiness:"
        "target_0:axis_specific_runtime_tree_placeholder_leak"
    ) in failures
    assert (
        "l5_v2_frontier_readiness:target_0:single_axis_modal_entrypoint_leak"
        in failures
    )


def test_operator_briefing_dispatch_gate_rejects_l5_matrix_sha_mismatch() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    l5["next_exact_eval_target_count"] = 0
    l5["next_exact_eval_targets_sample"] = []
    l5["packetir_matrix_dispatch_targets_suppressed"] = True
    l5["blockers"] = ["l5_v2_packetir_matrix_artifact_sha_mismatch"]
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:l5_v2_packetir_matrix_artifact_sha_mismatch"
        in failures
    )


def test_operator_briefing_dispatch_gate_checks_full_l5_target_list() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])  # type: ignore[index]
    safe = {
        "lane_id": "pr106_packetir_safe",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
        "command_template": (
            ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
            "--archive a.zip --expected-runtime-tree-sha256 auto "
            "--skip-axis-if-promotable-anchor-exists"
        ),
        "dispatch_status": "requires_claim_lane_dispatch_before_provider_launch",
    }
    unsafe = {
        **safe,
        "lane_id": "pr106_packetir_hidden_bad",
        "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
        "command_template": (
            ".venv/bin/modal run experiments/modal_auth_eval.py "
            "--archive a.zip"
        ),
    }
    l5["next_exact_eval_target_count"] = 2
    l5["next_exact_eval_targets_sample"] = [safe]
    l5["next_exact_eval_targets"] = [safe, unsafe]
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:target_1:single_axis_modal_entrypoint_leak"
        in failures
    )
    assert (
        "l5_v2_frontier_readiness:target_1:paired_dispatch_command_missing:"
        "tools/dispatch_modal_paired_auth_eval.py"
    ) in failures


def test_operator_briefing_dispatch_gate_rejects_asymptotic_false_authority() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])
    bad = dict(l5["asymptotic_pursuit_candidates"][0])
    bad["ready_for_exact_eval_dispatch"] = True
    bad["lane_registry_registered"] = False
    bad["ready_for_l1_build_semantics"] = "ambiguous"
    bad_status = dict(bad["l5_v2_asymptotic_next_action_status"])
    bad_status["lane_registry_registered"] = False
    bad_status["canonical_replacement_lane_registered"] = False
    bad["l5_v2_asymptotic_next_action_status"] = bad_status
    l5["asymptotic_pursuit_candidates"] = [bad]
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:asymptotic_candidate:"
        "z6_z7_z8_predictive_coding_world_models:"
        "ready_for_exact_eval_dispatch_not_false"
    ) in failures
    assert (
        "l5_v2_frontier_readiness:asymptotic_candidate:"
        "z6_z7_z8_predictive_coding_world_models:lane_registry_missing"
    ) in failures
    assert (
        "l5_v2_frontier_readiness:asymptotic_candidate:"
        "z6_z7_z8_predictive_coding_world_models:"
        "next_action_status_lane_registry_missing"
    ) in failures
    assert (
        "l5_v2_frontier_readiness:asymptotic_candidate:"
        "z6_z7_z8_predictive_coding_world_models:l1_semantics_missing"
    ) in failures


def test_operator_briefing_dispatch_gate_rejects_completed_asymptotic_action_ready() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }
    l5 = dict(payload["l5_v2_frontier_readiness"])
    bad = dict(l5["asymptotic_pursuit_candidates"][0])
    bad["ready_for_l1_build"] = False
    bad["l1_scaffold_present"] = True
    bad["recommended_next_action_completed_or_superseded"] = True
    bad["ready_for_recommended_next_action"] = True
    bad["ready_for_l1_build_semantics"] = (
        "ready_to_start_l1_scaffold_work_only_not_scaffold_ready"
    )
    l5["asymptotic_pursuit_candidates"] = [bad]
    payload["l5_v2_frontier_readiness"] = l5

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "l5_v2_frontier_readiness:asymptotic_candidate:"
        "z6_z7_z8_predictive_coding_world_models:"
        "completed_l1_semantics_invalid"
    ) in failures
    assert (
        "l5_v2_frontier_readiness:asymptotic_candidate:"
        "z6_z7_z8_predictive_coding_world_models:completed_action_still_ready"
    ) in failures


def test_operator_briefing_dispatch_gate_rejects_terminal_packet_commands() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
        "exact_eval_packets": [
            {
                "lane_id": "terminal_packet",
                "terminal_exact_eval_evidence_blockers": ["same_lane_terminal_negative"],
                "repeat_dispatch_allowed": True,
                "ready_for_submit": True,
                "commands": {"claim": "claim", "submit": "submit", "harvest": "harvest"},
            }
        ],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "exact_eval_packets:terminal_packet:terminal_evidence_not_suppressing_repeat_dispatch"
        in failures
    )
    assert "exact_eval_packets:terminal_packet:terminal_evidence_ready_for_submit" in failures
    assert "exact_eval_packets:terminal_packet:terminal_evidence_commands_not_suppressed" in failures


def test_operator_briefing_xray_gate_requires_visible_false_authority_tools() -> None:
    module = _load_all_lanes_module()
    payload = {
        "xray_tools": [
            {
                "tool": tool,
                "tool_exists": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": ["diagnostic_tool_no_score_or_dispatch_authority"],
            }
            for tool in sorted(module._EXPECTED_XRAY_TOOLS)
        ]
    }

    assert module._operator_briefing_xray_failures(payload) == []


def test_operator_briefing_xray_gate_rejects_missing_and_authoritative_rows() -> None:
    module = _load_all_lanes_module()
    first_tool = sorted(module._EXPECTED_XRAY_TOOLS)[0]
    payload = {
        "xray_tools": [
            {
                "tool": first_tool,
                "tool_exists": False,
                "score_claim": True,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": [],
            }
        ]
    }

    failures = module._operator_briefing_xray_failures(payload)

    assert f"xray_tools:{first_tool}:tool_exists_not_true" in failures
    assert f"xray_tools:{first_tool}:score_claim_not_false" in failures
    assert f"xray_tools:{first_tool}:missing_dispatch_blocker" in failures
    assert any(failure.startswith("xray_tools_missing_expected:") for failure in failures)


def test_terminal_substrate_claims_missing_evidence_detects_pr95plus_gap() -> None:
    module = _load_all_lanes_module()
    rows = [
        {
            "timestamp_utc": "2026-05-13T22:08:18Z",
            "platform": "modal",
            "lane_id": "lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513",
            "instance_job_id": (
                "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch_"
                "20260513T215933Z__smoke__100ep"
            ),
            "status": "failed_modal_training_rc_13",
        }
    ]

    missing = module._terminal_substrate_claims_missing_evidence(rows, set())

    assert missing
    assert "lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513" in missing[0]


def test_terminal_substrate_claims_require_exact_job_status_coverage() -> None:
    module = _load_all_lanes_module()
    rows = [
        {
            "timestamp_utc": "2026-05-13T22:08:18Z",
            "platform": "modal",
            "lane_id": "lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513",
            "instance_job_id": (
                "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch_"
                "20260513T215933Z__smoke__100ep"
            ),
            "status": "failed_modal_training_rc_13",
        }
    ]

    lane_only = {
        (
            "lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513",
            "different_job_same_lane",
            "failed_modal_training_rc_13",
        )
    }
    exact = {
        (
            "lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513",
            "substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch_"
            "20260513T215933Z__smoke__100ep",
            "failed_modal_training_rc_13",
        )
    }

    assert module._terminal_substrate_claims_missing_evidence(rows, lane_only)
    assert module._terminal_substrate_claims_missing_evidence(rows, exact) == []


def test_terminal_substrate_claims_missing_evidence_detects_refused_dispatch() -> None:
    module = _load_all_lanes_module()
    rows = [
        {
            "timestamp_utc": "2026-05-13T20:59:07Z",
            "platform": "modal",
            "lane_id": "lane_time_traveler_l5_autonomy_substrate_20260513",
            "instance_job_id": "pending_smoke_t4",
            "status": "refused_dispatch_operator_authorize_yN_gate_not_bypassable_from_subprocess",
        }
    ]

    missing = module._terminal_substrate_claims_missing_evidence(rows, set())

    assert missing
    assert "refused_dispatch_operator_authorize_yN_gate_not_bypassable_from_subprocess" in missing[0]


def test_terminal_substrate_claims_missing_evidence_detects_non_modal_substrate_gap() -> None:
    module = _load_all_lanes_module()
    rows = [
        {
            "timestamp_utc": "2026-05-13T20:59:07Z",
            "platform": "lightning",
            "lane_id": "lane_substrate_boundary_atoms_20260513",
            "instance_job_id": "substrate_boundary_atoms_lightning_job",
            "status": "failed_lightning_runtime_deps",
        }
    ]

    missing = module._terminal_substrate_claims_missing_evidence(rows, set())

    assert missing
    assert "platform" not in missing[0]


def test_terminal_substrate_claims_ignores_nonterminal_smoke_dispatched() -> None:
    module = _load_all_lanes_module()
    rows = [
        {
            "timestamp_utc": "2026-05-13T20:55:45Z",
            "platform": "modal",
            "lane_id": "lane_time_traveler_l5_autonomy_substrate_20260513",
            "instance_job_id": "pending_smoke_t4",
            "status": "smoke_dispatched",
        }
    ]

    assert module._terminal_substrate_claims_missing_evidence(rows, set()) == []


def test_terminal_claim_coverage_from_jsonl_reads_exact_claims(tmp_path: Path) -> None:
    module = _load_all_lanes_module()
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        '{"covered_terminal_claims":[{'
        '"lane_id":"lane_a",'
        '"instance_job_id":"job_a",'
        '"status":"failed_modal_training_rc_13"'
        '}],'
        '"covered_terminal_lane_ids":["lane_a"]}\n'
        '{"covered_terminal_lane_ids":["lane_b"]}\n',
        encoding="utf-8",
    )

    assert module._terminal_claim_coverage_from_jsonl(evidence) == {
        ("lane_a", "job_a", "failed_modal_training_rc_13")
    }


def test_hlm1_frontier_prose_guard_allows_nonpromotional_reference(tmp_path: Path) -> None:
    module = _load_all_lanes_module()
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "ok.md").write_text(
        "HLM1 is a non-promotional reference, not the active frontier.\n",
        encoding="utf-8",
    )
    (research / "bad.md").write_text(
        "HLM1 is the current exact frontier used by optimizer routing.\n",
        encoding="utf-8",
    )
    (research / "bad_paragraph.md").write_text(
        "HLM1 at 0.206 is preserved.\n"
        "It is the current local exact floor.\n",
        encoding="utf-8",
    )
    (research / "handoff.md").write_text(
        "HLM1 is a non-promotional reference only.\n"
        "HDM4 is the active exact dispatch frontier.\n",
        encoding="utf-8",
    )
    (research / "historical.md").write_text(
        "Superseded historical note.\n\n"
        "HLM1 is the current exact frontier used by optimizer routing.\n",
        encoding="utf-8",
    )

    violations = module._hlm1_frontier_prose_violations(tmp_path)

    assert len(violations) == 2
    assert any("bad.md:1" in violation for violation in violations)
    assert any("bad_paragraph.md:2" in violation for violation in violations)
    assert not any("handoff.md" in violation for violation in violations)
