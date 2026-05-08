from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from tools.build_cross_paradigm_frontier_inventory import build_inventory

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location("all_lanes_preflight_geometry_test", ALL_LANES)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_lanes_cross_paradigm_gate_accepts_fail_closed_geometry_contracts() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)

    assert module._geometry_feedback_inventory_failures(payload) == []
    assert module._cross_paradigm_queue_routing_failures(
        payload["frontier_action_queue"]
    ) == []


def test_all_lanes_cross_paradigm_gate_rejects_lost_first_tranche_route() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)
    queue = [
        row
        for row in payload["frontier_action_queue"]
        if row["key"] != "categorical_qma9_clade_spade_openpilot"
    ]
    queue.append(
        next(
            row
            for row in payload["frontier_action_queue"]
            if row["key"] == "categorical_qma9_clade_spade_openpilot"
        )
    )

    failures = module._cross_paradigm_queue_routing_failures(queue)

    assert (
        "first_tranche_missing_required_score_path_row(s): categorical_qma9_clade_spade_openpilot"
        in failures
    )


def test_all_lanes_cross_paradigm_gate_pins_promoted_pr103_anchor() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)

    assert module._cross_paradigm_pr103_anchor_failures(payload["rows"]) == []

    row = next(
        row
        for row in payload["rows"]
        if row["key"] == "hnerv_pr103_pr106_ac_repack_runtime_closure"
    )
    row["score_snapshot"]["archive_bytes"] = 186239

    assert module._cross_paradigm_pr103_anchor_failures(payload["rows"]) == [
        "anchor_archive_bytes_drift: 186239"
    ]

    row["score_snapshot"]["archive_bytes"] = 185578
    row["score_snapshot"]["runtime_tree_sha256"] = "deadbeef"

    assert module._cross_paradigm_pr103_anchor_failures(payload["rows"]) == [
        "anchor_runtime_tree_sha256_drift: expected "
        "'54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6', "
        "got 'deadbeef'"
    ]

    row["score_snapshot"]["runtime_tree_sha256"] = (
        "54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6"
    )
    row["score_snapshot"]["compliance_passed"] = False
    row["score_snapshot"]["compliance_failed_checks"] = ["auth_eval_promotable_stamp"]

    assert module._cross_paradigm_pr103_anchor_failures(payload["rows"])[:2] == [
        "anchor_contest_final_compliance_not_passed",
        "anchor_contest_final_failed_checks: auth_eval_promotable_stamp",
    ]


def test_all_lanes_cross_paradigm_gate_rejects_missing_geometry_contract() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)
    row = next(row for row in payload["rows"] if row["key"] == "raft_radial_openpilot_pose")
    row.pop("geometry_feedback_contract")

    failures = module._geometry_feedback_inventory_failures(payload)

    assert "raft_radial_openpilot_pose: geometry_feedback_contract_missing" in failures


def test_all_lanes_cross_paradigm_gate_rejects_geometry_dispatch_ready_drift() -> None:
    module = _load_all_lanes_module()
    payload = build_inventory(repo_root=REPO)
    row = next(row for row in payload["rows"] if row["key"] == "lapose_motion_atom_allocator")
    row["geometry_feedback_contract"]["ready_for_exact_eval_dispatch"] = True

    failures = module._geometry_feedback_inventory_failures(payload)

    assert (
        "lapose_motion_atom_allocator: "
        "geometry_feedback_contract_ready_for_exact_eval_dispatch_false"
    ) in failures


def test_all_lanes_frontier_layout_gate_accepts_monolithic_logical_sections() -> None:
    module = _load_all_lanes_module()
    payload = {
        "score_claim": False,
        "evidence_grade": "empirical_archive_layout_cpu_no_score",
        "runs": [
            {
                "archive_path": "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
                "score_claim": False,
                "physical_layout": {
                    "single_member_monolithic_packet": True,
                    "archive_member_level_component_budgets_valid": False,
                    "member_level_mask_budget_valid": False,
                    "member_level_pose_budget_valid": False,
                    "members": [{"name": "x"}],
                },
                "logical_layout": {
                    "grammar": "pr101_fixed_offset_hnerv_microcodec",
                    "sections": [{"name": "decoder_blob"}],
                },
            },
            {
                "archive_path": "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip",
                "score_claim": False,
                "physical_layout": {
                    "single_member_monolithic_packet": True,
                    "archive_member_level_component_budgets_valid": False,
                    "member_level_mask_budget_valid": False,
                    "member_level_pose_budget_valid": False,
                    "members": [{"name": "0.bin"}],
                },
                "logical_layout": {
                    "grammar": "pr106_ff_packed_hnerv",
                    "sections": [{"name": "decoder_packed_brotli"}],
                },
            },
        ],
    }

    assert module._frontier_monolithic_layout_failures(payload) == []


def test_all_lanes_frontier_layout_gate_rejects_member_budget_regression() -> None:
    module = _load_all_lanes_module()
    payload = {
        "score_claim": False,
        "evidence_grade": "empirical_archive_layout_cpu_no_score",
        "runs": [
            {
                "archive_path": "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
                "score_claim": False,
                "physical_layout": {
                    "single_member_monolithic_packet": True,
                    "archive_member_level_component_budgets_valid": True,
                    "member_level_mask_budget_valid": True,
                    "member_level_pose_budget_valid": False,
                    "members": [{"name": "x"}],
                },
                "logical_layout": {
                    "grammar": "pr101_fixed_offset_hnerv_microcodec",
                    "sections": [{"name": "decoder_blob"}],
                },
            },
            {
                "archive_path": "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip",
                "score_claim": False,
                "physical_layout": {
                    "single_member_monolithic_packet": True,
                    "archive_member_level_component_budgets_valid": False,
                    "member_level_mask_budget_valid": False,
                    "member_level_pose_budget_valid": False,
                    "members": [{"name": "0.bin"}],
                },
                "logical_layout": {
                    "grammar": "pr106_ff_packed_hnerv",
                    "sections": [{"name": "decoder_packed_brotli"}],
                },
            },
        ],
    }

    failures = module._frontier_monolithic_layout_failures(payload)

    assert "layout_run_0_member_level_component_budgets_not_rejected" in failures
    assert "layout_run_0_member_level_mask_budget_not_rejected" in failures


def test_all_lanes_frontier_layout_gate_rejects_swapped_pr_family_members() -> None:
    module = _load_all_lanes_module()
    payload = {
        "score_claim": False,
        "evidence_grade": "empirical_archive_layout_cpu_no_score",
        "runs": [
            {
                "archive_path": "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
                "score_claim": False,
                "physical_layout": {
                    "single_member_monolithic_packet": True,
                    "archive_member_level_component_budgets_valid": False,
                    "member_level_mask_budget_valid": False,
                    "member_level_pose_budget_valid": False,
                    "members": [{"name": "0.bin"}],
                },
                "logical_layout": {
                    "grammar": "pr101_fixed_offset_hnerv_microcodec",
                    "sections": [{"name": "decoder_blob"}],
                },
            },
            {
                "archive_path": "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip",
                "score_claim": False,
                "physical_layout": {
                    "single_member_monolithic_packet": True,
                    "archive_member_level_component_budgets_valid": False,
                    "member_level_mask_budget_valid": False,
                    "member_level_pose_budget_valid": False,
                    "members": [{"name": "x"}],
                },
                "logical_layout": {
                    "grammar": "pr106_ff_packed_hnerv",
                    "sections": [{"name": "decoder_packed_brotli"}],
                },
            },
        ],
    }

    failures = module._frontier_monolithic_layout_failures(payload)

    assert "layout_run_0_pr101_member_name_must_be_x" in failures
    assert "layout_run_1_pr106_member_name_must_be_0.bin" in failures
