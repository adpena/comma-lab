from __future__ import annotations

import json
from pathlib import Path

from tac.frontier_rows import FRONTIER_ROW_FIELDS, FRONTIER_ROW_SCHEMA
from tac.geometry_feedback_readiness import (
    GEOMETRY_FEEDBACK_ROADMAP_KEYS,
    UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER,
)
from tools.build_cross_paradigm_frontier_inventory import (
    STATIC_ROWS,
    build_inventory,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[3]


def test_cross_paradigm_inventory_is_deterministic_and_non_dispatching() -> None:
    first = build_inventory(repo_root=REPO)
    second = build_inventory(repo_root=REPO)

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["score_claim"] is False
    assert first["dispatch_attempted"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert first["frontier_action_queue_count"] == first["row_count"]
    assert first["frontier_action_queue"]
    assert all(row["score_claim"] is False for row in first["frontier_action_queue"])
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in first["frontier_action_queue"]
    )
    assert "requires_exact_cuda_auth_eval" in first["dispatch_blockers"]


def test_cross_paradigm_inventory_pins_required_score_path_rows() -> None:
    payload = build_inventory(repo_root=REPO)
    rows = {row["key"]: row for row in payload["rows"]}

    for key in (
        "hnerv_pr103_pr106_ac_repack_runtime_closure",
        "categorical_qma9_clade_spade_openpilot",
        "lapose_motion_atom_allocator",
        "meta_lagrangian_cross_paradigm_allocator",
        "telescopic_foveation_field",
        "hnerv_per_tensor_context_entropy",
    ):
        assert key in rows
        assert rows[key]["score_claim"] is False
        assert rows[key]["ready_for_exact_eval_dispatch"] is False
        assert rows[key]["next_patch"]
        assert rows[key]["blockers"]

    pr103 = rows["hnerv_pr103_pr106_ac_repack_runtime_closure"]
    assert pr103["action_class"] == "maintain_exact_eval_anchor_and_pivot"
    assert pr103["priority_tier"] == 900
    assert pr103["status"] == "exact_cuda_a++_anchor_promoted"
    assert pr103["role"] == "current_exact_rate_anchor"
    assert pr103["score_snapshot"]["score"] == 0.2089810755823297
    assert pr103["score_snapshot"]["report_reconstructed_score"] == 0.20898105277982337
    assert pr103["score_snapshot"]["score_basis"] == (
        "auth_eval_report_components_plus_exact_archive_bytes"
    )
    assert pr103["score_snapshot"]["anchor_proof_schema"] == (
        "pre_submission_compliance_anchor_proof_v1"
    )
    assert pr103["score_snapshot"]["compliance_passed"] is True
    assert pr103["score_snapshot"]["compliance_failed_checks"] == []
    assert pr103["score_snapshot"]["compliance_check_count"] > 0
    assert pr103["score_snapshot"]["archive_bytes"] == 185578
    assert (
        pr103["score_snapshot"]["archive_sha256"]
        == "ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce"
    )
    assert "entropy_coding" in pr103["paradigms"]
    assert "src/tac/pr103_pr106_runtime_closure.py" in pr103["code_paths"]
    assert "submissions/pr103_pr106_final_runtime/inflate.py" in pr103["code_paths"]
    assert "tools/prove_pr103_pr106_runtime_closure.py" in pr103["code_paths"]
    assert "tools/prove_pr103_pr106_final_runtime_packet.py" in pr103["code_paths"]
    assert (
        "experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json"
        in pr103["evidence_paths"]
    )
    assert (
        "experiments/results/pr103_repack_pr106_standalone_20260507/final_runtime_packet_proof.json"
        in pr103["evidence_paths"]
    )
    assert (
        "experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.static.json"
        in pr103["evidence_paths"]
    )
    assert (
        "experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json"
        in pr103["evidence_paths"]
    )
    assert pr103["score_snapshot"]["path"] == (
        "experiments/results/pr103_repack_pr106_standalone_20260507/"
        "pre_submission_compliance.contest_final.json"
    )
    assert (
        ".omx/research/pr103_pr106_ac_repack_exact_eval_20260507_codex.md"
        in pr103["evidence_paths"]
    )
    assert "185578 bytes" in pr103["next_patch"]
    assert any("completed anchor" in blocker for blocker in pr103["blockers"])

    categorical = rows["categorical_qma9_clade_spade_openpilot"]
    assert "categorical_masks" in categorical["paradigms"]
    assert "openpilot_priors" in categorical["paradigms"]
    assert categorical["action_class"] == "build_byte_closed_categorical_candidate"
    assert categorical["priority_tier"] == 10
    assert categorical["status"] == "byte_closed_local_candidate_artifact_landed_blocked_on_parity"
    assert "src/tac/categorical_candidate_readiness.py" in categorical["code_paths"]
    assert "src/tac/categorical_candidate_runtime_skeleton.py" in categorical["code_paths"]
    assert (
        "src/tac/categorical_openpilot_mask_prior_contract.py"
        in categorical["code_paths"]
    )
    assert "src/tac/categorical_payload_candidate.py" in categorical["code_paths"]
    assert "src/tac/pr91_hpm1_readiness.py" in categorical["code_paths"]
    assert "src/tac/pr91_hpm1_runtime_contract.py" in categorical["code_paths"]
    assert "tools/audit_categorical_candidate_readiness.py" in categorical["code_paths"]
    assert "tools/audit_pr91_hpm1_readiness.py" in categorical["code_paths"]
    assert "tools/audit_pr91_hpm1_runtime_contract.py" in categorical["code_paths"]
    assert "tools/build_categorical_candidate_fixture.py" in categorical["code_paths"]
    assert "tools/build_categorical_candidate_payload.py" in categorical["code_paths"]
    assert (
        "experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json"
        in categorical["evidence_paths"]
    )
    assert (
        "experiments/results/pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json"
        in categorical["evidence_paths"]
    )
    assert (
        "experiments/results/categorical_openpilot_payload_candidate_20260506_codex/readiness.json"
        in categorical["evidence_paths"]
    )
    assert (
        ".omx/research/pr91_hpm1_phase_major_failure_classification_20260507_codex.json"
        in categorical["evidence_paths"]
    )
    assert (
        ".omx/research/pr91_hpm1_submitted_prefix_token_recovery_tile_major_20260507_codex.json"
        in categorical["evidence_paths"]
    )
    assert (
        ".omx/research/pr91_hpm1_next_row_suffix_scan_tile_major_20260507_codex.json"
        in categorical["evidence_paths"]
    )
    assert "phase-major and tile-major failure classifications" in categorical["next_patch"]
    assert any("15989 symbols" in blocker for blocker in categorical["blockers"])
    assert any("8274 symbols" in blocker for blocker in categorical["blockers"])
    assert any("1134 remaining rows" in blocker for blocker in categorical["blockers"])
    assert any("range-state grammar" in blocker for blocker in categorical["blockers"])

    entropy = rows["hnerv_per_tensor_context_entropy"]
    assert "src/tac/optimization/entropy_codec_gap_audit.py" in entropy["code_paths"]
    assert "tools/audit_entropy_codec_gap.py" in entropy["code_paths"]

    wr01 = rows["hnerv_wavelet_wr01_apply"]
    assert "src/tac/hnerv_wavelet_compress_time_harness.py" in wr01["code_paths"]
    assert "tools/build_hnerv_wavelet_compress_time_harness.py" in wr01["code_paths"]
    assert "compress-time harness" in wr01["next_patch"]

    sensitivity = rows["sensitivity_omega_w_v3"]
    assert "src/tac/neural_weight_codec_sensitivity.py" in sensitivity["code_paths"]
    assert (
        ".omx/research/nwcs_beta_encoding_loop_greenup_20260507_codex.md"
        in sensitivity["evidence_paths"]
    )
    assert "deterministic NWCS stream manifest" in sensitivity["next_patch"]

    lowlevel = rows["hnerv_lowlevel_brotli_repack"]
    assert (
        "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/result.json"
        in lowlevel["evidence_paths"]
    )
    assert "PR106x lgblock16 -1B" in lowlevel["next_patch"]

    lapose = rows["lapose_motion_atom_allocator"]
    assert lapose["role"] == "proposal_allocator"
    assert "meta_lagrangian" in lapose["paradigms"]
    assert lapose["action_class"] == "calibrate_planning_signal_and_attach_archive_consumer"

    allocator = rows["meta_lagrangian_cross_paradigm_allocator"]
    assert allocator["status"] == "field_acquisition_ranker_landed_planning_only"
    assert "src/tac/optimization/field_equation_planner.py" in allocator["code_paths"]
    assert "src/tac/optimization/bayesian_experimental_design.py" in allocator["code_paths"]
    assert "tools/build_field_equation_plan.py" in allocator["code_paths"]
    assert (
        ".omx/research/field_acquisition_ranking_20260507_codex.md"
        in allocator["evidence_paths"]
    )
    assert "field_acquisition_ranking" in allocator["next_patch"]
    assert any("zero design-ready rows" in blocker for blocker in allocator["blockers"])


def test_cross_paradigm_inventory_geometry_feedback_contracts_fail_closed() -> None:
    payload = build_inventory(repo_root=REPO)
    rows = {row["key"]: row for row in payload["rows"]}

    assert "geometry_feedback_requires_charged_runtime_consumer" in payload["dispatch_blockers"]
    for key in GEOMETRY_FEEDBACK_ROADMAP_KEYS:
        contract = rows[key]["geometry_feedback_contract"]
        assert contract["score_claim"] is False
        assert contract["dispatch_attempted"] is False
        assert contract["ready_for_exact_eval_dispatch"] is False
        assert contract["charged_runtime_consumed"] is False
        assert UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER in contract["dispatch_blockers"]
        assert UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER in rows[key]["blockers"]


def test_cross_paradigm_inventory_action_queue_routes_next_tranche() -> None:
    payload = build_inventory(repo_root=REPO)
    queue = payload["frontier_action_queue"]

    assert [row["key"] for row in queue[:5]] == [
        "categorical_qma9_clade_spade_openpilot",
        "joint_admm_balle_arithmetic_stack",
        "hnerv_per_tensor_context_entropy",
        "telescopic_foveation_field",
        "lapose_motion_atom_allocator",
    ]
    assert queue[0]["action_class"] == "build_byte_closed_categorical_candidate"
    assert queue[1]["action_class"] == "wire_jcsp_submission_runtime_consumer"
    assert queue[-1]["action_class"] == "maintain_exact_eval_anchor_and_pivot"
    assert payload["action_class_counts"]["maintain_exact_eval_anchor_and_pivot"] == 1
    assert payload["action_class_counts"]["wire_jcsp_submission_runtime_consumer"] == 1


def test_cross_paradigm_inventory_emits_comparable_frontier_rows() -> None:
    payload = build_inventory(repo_root=REPO)

    assert payload["frontier_row_schema"] == FRONTIER_ROW_SCHEMA
    assert payload["frontier_row_fields"] == list(FRONTIER_ROW_FIELDS)
    assert payload["frontier_row_count"] == payload["row_count"]
    assert len(payload["frontier_rows"]) == payload["row_count"]
    for row, frontier_row in zip(payload["rows"], payload["frontier_rows"], strict=True):
        assert row["frontier_row"] == frontier_row
        assert list(frontier_row) == list(FRONTIER_ROW_FIELDS)
        assert frontier_row["schema"] == FRONTIER_ROW_SCHEMA
        assert frontier_row["source_tool"] == "tools/build_cross_paradigm_frontier_inventory.py"
        assert frontier_row["candidate_id"] == row["key"]
        assert frontier_row["family_group"] == row["key"]
        assert frontier_row["pareto_scope"] == row["key"]
        assert frontier_row["paradigms"] == row["paradigms"]
        assert frontier_row["score_claim"] is False
        assert frontier_row["dispatch_attempted"] is False
        assert frontier_row["candidate_static_preflight_ready"] is False
        assert frontier_row["ready_for_exact_eval_dispatch"] is False
        if row["key"] == "hnerv_pr103_pr106_ac_repack_runtime_closure":
            assert frontier_row["pareto_eligible"] is True
            assert frontier_row["pareto_frontier"] is True
            assert frontier_row["planning_priority_rankable"] is True
        else:
            assert frontier_row["pareto_eligible"] is False
            assert frontier_row["pareto_frontier"] is False
            assert frontier_row["planning_priority_rankable"] is False
        assert "exact_cuda_auth_eval" in frontier_row["next_required_proof"]
    pr103 = next(
        row for row in payload["frontier_rows"] if row["key"] == "hnerv_pr103_pr106_ac_repack_runtime_closure"
    )
    assert pr103["score_evidence_rankable"] is True
    assert pr103["score_evidence_path"].endswith(
        "pre_submission_compliance.contest_final.json"
    )
    assert pr103["exact_score"] == 0.2089810755823297
    assert pr103["archive_bytes"] == 185578
    assert pr103["archive_sha256"] == "ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce"


def test_cross_paradigm_inventory_paths_are_current_on_main() -> None:
    payload = build_inventory(repo_root=REPO)

    assert payload["row_count"] == len(STATIC_ROWS)
    assert payload["missing_code_path_count"] == 0
    assert payload["missing_evidence_path_count"] == 0
    for row in payload["rows"]:
        assert row["path_audit"]["code"]["missing"] == []
        assert row["path_audit"]["evidence"]["missing"] == []


def test_cross_paradigm_inventory_markdown_is_operator_briefing() -> None:
    payload = build_inventory(repo_root=REPO)
    markdown = render_markdown(payload)

    assert "Cross-Paradigm Frontier Inventory" in markdown
    assert "Inventory-only orchestration artifact" in markdown
    assert "`categorical_qma9_clade_spade_openpilot`" in markdown
    assert "`meta_lagrangian_cross_paradigm_allocator`" in markdown
    assert "`build_byte_closed_categorical_candidate`" in markdown


def test_checked_in_cross_paradigm_inventory_markdown_matches_generator() -> None:
    payload = build_inventory(repo_root=REPO)
    expected = render_markdown(payload)
    checked_in = (
        REPO / ".omx/research/cross_paradigm_frontier_inventory_20260506_codex.md"
    ).read_text(encoding="utf-8")

    assert checked_in == expected
