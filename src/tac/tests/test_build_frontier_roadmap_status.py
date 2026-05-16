# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from tac.hnerv_entropy_frontier_selector import (
    ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES,
    RATE_ONLY_FLOOR_BLOCKER_PREFIX,
)
from tools.build_frontier_roadmap_status import (
    DEFAULT_PACKET_MANIFEST_GLOBS,
    GLOBAL_ROADMAP_BUILD_PREFLIGHT_PATHS,
    build_roadmap_status,
    dirty_paths_for_row,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[3]


def test_dirty_paths_for_row_matches_exact_and_nested_paths() -> None:
    row = {
        "code_paths": ["src/tac/hnerv_wavelet_apply_transform.py"],
        "evidence_paths": ["experiments/results/example/manifest.json"],
    }

    assert dirty_paths_for_row(
        row,
        [
            "src/tac/hnerv_wavelet_apply_transform.py",
            "experiments/results/example",
            "src/tac/unrelated.py",
        ],
    ) == [
        "experiments/results/example",
        "src/tac/hnerv_wavelet_apply_transform.py",
    ]


def test_dirty_paths_for_row_blocks_global_roadmap_modules_without_row_code_paths() -> None:
    row = {
        "code_paths": [],
        "evidence_paths": [],
    }

    assert dirty_paths_for_row(
        row,
        ["experiments/preflight_candidate_manifest_dispatch_readiness.py"],
    ) == ["experiments/preflight_candidate_manifest_dispatch_readiness.py"]
    for path in GLOBAL_ROADMAP_BUILD_PREFLIGHT_PATHS:
        assert path in dirty_paths_for_row(row, [path])


def test_frontier_roadmap_status_is_non_dispatching_and_dirty_aware() -> None:
    payload = build_roadmap_status(
        repo_root=REPO,
        dirty_paths=[
            "src/tac/hnerv_wavelet_apply_transform.py",
            "src/tac/sensitivity_map/__init__.py",
        ],
    )
    rows = {row["key"]: row for row in payload["rows"]}

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["row_count"] == 13
    assert payload["dirty_path_count"] == 2
    assert "requires_exact_cuda_auth_eval" in payload["dispatch_blockers"]
    assert payload["next_comprehensive_tranche"]["score_claim"] is False
    assert payload["next_comprehensive_tranche"]["dispatch_attempted"] is False
    assert payload["next_comprehensive_tranche"]["ready_for_exact_eval_dispatch"] is False
    assert payload["next_comprehensive_tranche"]["schema"] == "next_comprehensive_tranche_v1"
    assert payload["next_comprehensive_tranche"]["effective_dispatch_candidate_id"] is None
    assert payload["scorer_surface_shaking_plan"]["score_claim"] is False
    assert payload["scorer_surface_shaking_plan"]["ready_for_exact_eval_dispatch"] is False
    assert (
        "pixel_lsb_threshold_probe"
        in payload["next_comprehensive_tranche"]["candidate_pools"]["scorer_surface_proxy_atoms"]
    )
    assert payload["next_comprehensive_tranche"]["workstreams"][0]["id"] == (
        "scorer_surface_shaking"
    )
    assert payload["next_comprehensive_tranche"]["workstreams"][0]["all_keys_safe_to_touch_now"] is True
    assert (
        payload["next_comprehensive_tranche"]["selected_candidate_dispatch_status"][
            "selected_row_dispatchable"
        ]
        is False
    )
    assert "requires_lane_dispatch_claim" in payload["next_comprehensive_tranche"]["dispatch_blockers"]

    wr01 = rows["hnerv_wavelet_wr01_apply"]
    assert wr01["role"] == "stacker_scorer_changing"
    assert wr01["missing_code_path_count"] == 0
    assert wr01["missing_evidence_path_count"] == 0
    assert wr01["readiness_stage"] == "blocked_by_dirty_worktree"
    assert wr01["safe_to_touch_now"] is False
    assert wr01["dirty_path_blockers"] == ["src/tac/hnerv_wavelet_apply_transform.py"]

    sensitivity = rows["sensitivity_omega_w_v3"]
    assert sensitivity["readiness_stage"] == "blocked_by_dirty_worktree"
    assert sensitivity["safe_to_touch_now"] is False
    assert sensitivity["dirty_path_blockers"] == ["src/tac/sensitivity_map/__init__.py"]

    categorical = rows["categorical_qma9_clade_spade_openpilot"]
    assert categorical["safe_to_touch_now"] is True
    assert categorical["readiness_stage"] == "needs_byte_closed_candidate_or_fixture"

    jcsp = rows["joint_admm_balle_arithmetic_stack"]
    assert jcsp["action_class"] == "prove_jcsp_runtime_parity_and_charged_stack"
    assert jcsp["readiness_stage"] == "needs_byte_closed_candidate_or_fixture"
    assert jcsp["safe_to_touch_now"] is True
    assert "real AQ rawvideo JCSP runtime bridge" in jcsp["next_patch"]
    assert (
        "submission runtime detects but refuses jcsp.bin consumption"
        not in jcsp["blockers"]
    )
    assert any("raw-output parity proof missing" in blocker for blocker in jcsp["blockers"])

    tranche = payload["next_comprehensive_tranche"]
    pools = tranche["candidate_pools"]
    assert "hnerv_wavelet_wr01_apply" not in pools["exact_eval_or_review"]
    assert "categorical_qma9_clade_spade_openpilot" in pools["needs_byte_closed_candidate"]
    workstreams = {stream["id"]: stream for stream in tranche["workstreams"]}
    assert workstreams["scorer_changing_mask_payload"]["keys"] == [
        "hnerv_wavelet_wr01_apply",
        "categorical_qma9_clade_spade_openpilot",
        "cmg3_predictive_mask_grammar",
    ]
    assert workstreams["scorer_changing_mask_payload"]["dirty_blocked_keys"] == [
        "hnerv_wavelet_wr01_apply",
    ]
    assert workstreams["scorer_changing_mask_payload"]["unblocked_keys"] == [
        "categorical_qma9_clade_spade_openpilot",
        "cmg3_predictive_mask_grammar",
    ]
    assert workstreams["scorer_changing_mask_payload"]["all_keys_safe_to_touch_now"] is False
    assert "no remote/GPU dispatch without an active lane claim" in tranche["global_acceptance_gates"]
    assert any("no rate-only exact-eval spend" in gate for gate in tranche["global_acceptance_gates"])


def test_frontier_roadmap_status_markdown_is_operator_briefing() -> None:
    payload = build_roadmap_status(repo_root=REPO, dirty_paths=[])
    markdown = render_markdown(payload)

    assert "Frontier Roadmap Status" in markdown
    assert "Live-safe operator roadmap" in markdown
    assert "Next Comprehensive Tranche" in markdown
    assert "candidate_static_preflight_ready_count" in markdown
    assert "field_selection_ready_candidate_packet_count" in markdown
    assert "selected_candidate_dispatchable" in markdown
    assert "effective_dispatch_candidate" in markdown
    assert "selected_candidate_frontier_reason" in markdown
    assert "selected_candidate_exact_blocker_count" in markdown
    assert "selected_candidate_next_local_non_gpu_step" in markdown
    assert "selected_candidate_claim_blockers" in markdown
    assert "selected_candidate_static_refresh_status" in markdown
    assert "selected_candidate_approval_blockers" in markdown
    assert "Scorer Surface Shaking" in markdown
    assert "Planning-only local CPU search surface" in markdown
    assert "pixel_lsb_threshold_probe" in markdown
    assert "renderer_training_score_surface_loop" in markdown
    assert "Candidate Packets" in markdown
    assert "verify_lightning_env" in markdown
    assert "missing_operator_exact_cuda_approval" in markdown
    assert "dirty-blocked keys" in markdown
    assert "`rate_frontier_closure`" in markdown
    assert "`field_meta_selection`" in markdown
    assert "`hnerv_wavelet_wr01_apply`" in markdown
    assert "`categorical_qma9_clade_spade_openpilot`" in markdown
    assert "`stacker_scorer_changing`" in markdown
    assert "evidence" in markdown
    assert "next_unblocked_keys" in markdown


def test_frontier_roadmap_status_discovers_default_packet_manifests() -> None:
    payload = build_roadmap_status(repo_root=REPO, dirty_paths=[])

    packet_selection = payload["next_comprehensive_tranche"][
        "field_meta_candidate_packet_selection"
    ]
    candidate_ids = {row["candidate_id"] for row in packet_selection["rows"]}

    assert DEFAULT_PACKET_MANIFEST_GLOBS == (
        "experiments/results/**/wr01_exact_eval_packet.json",
        "experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_*/hnerv_lowlevel_exact_eval_packet.json",
        "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_*/hnerv_lowlevel_exact_eval_packet.json",
        "experiments/results/categorical_openpilot_payload_candidate*/candidate.json",
        "experiments/results/hnerv_hdm3_archive_candidate_*/hdm3_archive_candidate_manifest.json",
        "experiments/results/hnerv_hdm3_entropy_packet_*/hdc2_combined_entropy_reduction_manifest.json",
        "experiments/results/frontier_hidden_gem_routing_*/hidden_gem_readiness.json",
        "experiments/results/cross_paradigm_atom_ledger_*/ledger.json",
        "experiments/results/field_equation_plan_*/plan.json",
        "experiments/results/**/field_meta_selection*.json",
    )
    assert "wr01_apply_pr106x_half" in candidate_ids
    assert "pr106_q10_151byte_brotli" in candidate_ids
    assert "pr106x_lgblock16_1byte_brotli" in candidate_ids
    assert "categorical_openpilot_hpm1_payload_candidate" in candidate_ids
    assert "pr106x_hdm3_decoder_recode_14byte" in candidate_ids
    assert "hnerv_hdc2_hdm3_combined_entropy_target" in candidate_ids
    assert "hidden_gem_readiness_registry" in candidate_ids
    assert "cross_paradigm_atom_ledger_meta_adapter" in candidate_ids
    assert "field_equation_plan_meta_adapter" in candidate_ids
    assert "field_meta_selection_report" in candidate_ids
    assert packet_selection["candidate_count"] >= 3
    assert packet_selection["candidate_static_preflight_ready_count"] == 0
    assert packet_selection["pareto_summary"]["frontier_count"] == 0
    assert packet_selection["report_blockers"] == []
    wr01 = next(
        row
        for row in packet_selection["rows"]
        if row["candidate_id"] == "wr01_apply_pr106x_half"
    )
    assert wr01["archive_proof"]["byte_closed"] is True
    assert wr01["runtime_proof"]["runtime_closed"] is True
    assert wr01["strict_candidate_preflight_ready"] is False
    assert wr01["candidate_static_preflight_ready"] is False
    assert wr01["selection_decision"] == "strict_candidate_preflight_refused"
    assert "strict_candidate_preflight_not_ready" in wr01["candidate_blockers"]
    assert "strict:dispatch_unlocked_false" in wr01["candidate_blockers"]
    assert wr01["operator_next_steps_summary"]["schema"] == "packet_operator_next_steps_summary_v1"
    assert wr01["operator_next_steps_summary"]["packet_operator_next_steps_schema"] == (
        "wr01_operator_next_steps_v1"
    )
    assert wr01["operator_next_steps_summary"]["static_refresh_status"] == "passed"
    assert wr01["next_local_non_gpu_action"]["id"] == "verify_lightning_env"
    assert wr01["operator_claim_blockers"] == ["missing_active_lane_dispatch_claim"]
    assert wr01["operator_refresh_blockers"] == []
    assert wr01["operator_approval_blockers"] == []
    assert "missing_lightning_environment" in wr01["operator_current_blockers"]
    assert "missing_env:LIGHTNING_SSH_TARGET" in wr01["operator_environment_blockers"]
    q10 = next(
        row
        for row in packet_selection["rows"]
        if row["candidate_id"] == "pr106_q10_151byte_brotli"
    )
    assert q10["family_group"] == "hnerv_lowlevel_brotli_repack"
    assert q10["byte_delta"] == -151
    assert q10["archive_proof"]["byte_closed"] is True
    assert q10["runtime_proof"]["runtime_closed"] is True
    assert q10["candidate_static_preflight_ready"] is False
    assert q10["pareto_frontier"] is False
    assert q10["selection_decision"] == "rate_only_candidate_above_active_pr103_pr106_floor"
    dispatch_status = payload["next_comprehensive_tranche"]["selected_candidate_dispatch_status"]
    assert dispatch_status["candidate_id"] == "pr106_q10_151byte_brotli"
    assert dispatch_status["selected_row_dispatchable"] is False
    assert dispatch_status["effective_dispatch_candidate_id"] is None
    assert dispatch_status["exact_dispatch_blocker_count"] > 0
    assert q10["active_rate_only_floor_policy"]["active_floor_archive_bytes"] == (
        ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES
    )
    assert q10["active_rate_only_floor_policy"]["beats_active_floor"] is False
    assert f"{RATE_ONLY_FLOOR_BLOCKER_PREFIX}:{ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES}" in q10[
        "candidate_blockers"
    ]
    assert "missing_active_lane_dispatch_claim" in q10["candidate_blockers"]
    assert "claim:dispatch_claim_check_missing" in q10["candidate_blockers"]
    assert "rate_only_candidate_below_185578_byte_floor_or_scorer_changing_stack_path" in q10[
        "next_required_proof"
    ]
    assert "matching_active_level2_lane_claim_for_manifest_lane_and_job" in q10[
        "next_required_proof"
    ]
    assert "lightning_or_remote_exact_eval_environment_available" in q10["next_required_proof"]
    assert q10["operator_next_steps_summary"]["static_refresh_status"] == "passed"
    lgblock16 = next(
        row
        for row in packet_selection["rows"]
        if row["candidate_id"] == "pr106x_lgblock16_1byte_brotli"
    )
    assert lgblock16["family_group"] == "hnerv_lowlevel_brotli_repack"
    assert lgblock16["byte_delta"] == -1
    assert lgblock16["archive_proof"]["byte_closed"] is True
    assert lgblock16["runtime_proof"]["runtime_closed"] is True
    assert lgblock16["candidate_static_preflight_ready"] is False
    assert lgblock16["pareto_frontier"] is False
    assert lgblock16["pareto_dominated_by"] == []
    assert lgblock16["selection_decision"] == "rate_only_candidate_above_active_pr103_pr106_floor"
    assert f"{RATE_ONLY_FLOOR_BLOCKER_PREFIX}:{ACTIVE_RATE_ONLY_FLOOR_ARCHIVE_BYTES}" in lgblock16[
        "candidate_blockers"
    ]
    assert "missing_active_lane_dispatch_claim" in lgblock16["candidate_blockers"]
    assert "claim:dispatch_claim_check_missing" in lgblock16["candidate_blockers"]
    assert lgblock16["operator_next_steps_summary"]["static_refresh_status"] == "passed"
    assert lgblock16["operator_next_steps_summary"]["static_refresh_schema"] == (
        "hnerv_lowlevel_static_compliance_refresh_v1"
    )
    assert lgblock16["operator_next_steps_summary"]["static_refresh_source"] == (
        "refreshes.static_compliance"
    )
    assert "pareto_ineligible_for_field_selection" in lgblock16["exact_dispatch_blockers"]["blockers"]


def test_frontier_roadmap_status_consumes_field_meta_packet_manifests(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"field meta packet archive")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": "generic_packet",
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
                "lane_id": "lane_generic_packet",  # FAKE_LANE_OK:test-fixture lane_id
                "job_name": "job_generic_packet",
                "dispatch_unlocked": True,
                "ready_for_exact_eval_dispatch_claim": True,
                "family_group": "fixture_packet_family",
                "pareto_scope": "fixture_packet_family",
                "interaction_assumptions": ["fixture_first_order_packet"],
                "archive": {
                    "path": archive.as_posix(),
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "bytes": archive.stat().st_size,
                },
                "fixed_runtime_preflight": {
                    "ready_for_fixed_runtime_exact_eval": True,
                    "runtime_tree_sha256": "c" * 64,
                    "remaining_blockers": [],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_roadmap_status(
        repo_root=REPO,
        dirty_paths=[],
        packet_manifest_paths=[manifest],
    )

    packet_selection = payload["next_comprehensive_tranche"][
        "field_meta_candidate_packet_selection"
    ]
    assert packet_selection["candidate_count"] == 1
    assert packet_selection["candidate_local_preflight_ready_count"] == 1
    assert packet_selection["candidate_static_preflight_ready_count"] == 1
    assert packet_selection["ready_candidate_count"] == 0
    assert packet_selection["field_selection_ready_for_exact_eval_dispatch_count"] == 0
    assert packet_selection["pareto_summary"]["frontier_count"] == 1
    assert packet_selection["kkt_ready_for_field_planning_count"] == 0
    assert packet_selection["selected_candidate"]["candidate_id"] == "generic_packet"
    assert packet_selection["selected_candidate"]["strict_candidate_preflight_ready"] is True
    assert packet_selection["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert packet_selection["selected_candidate"]["kkt_ready_for_field_planning"] is False
    assert "kkt:kkt_proof_or_admm_result_missing" in packet_selection["selected_candidate"]["kkt_blockers"]
    assert (
        "kkt:kkt_proof_or_admm_result_missing"
        in packet_selection["selected_candidate"]["exact_dispatch_blockers"]["blockers"]
    )
    assert (
        packet_selection["selected_candidate"]["non_dominated_frontier_reason"]["reason"]
        == "non_dominated_within_pareto_scope"
    )
    assert packet_selection["selected_candidate"]["selection_decision"] == "static_candidate_acquire_kkt_and_lane_claim_before_dispatch"
    assert packet_selection["selected_candidate"]["ready_for_exact_eval_dispatch"] is False
    dispatch_status = payload["next_comprehensive_tranche"]["selected_candidate_dispatch_status"]
    assert dispatch_status["candidate_id"] == "generic_packet"
    assert dispatch_status["selected_row_dispatchable"] is False
    assert dispatch_status["effective_dispatch_candidate_id"] is None
    assert "kkt:kkt_proof_or_admm_result_missing" in dispatch_status["dispatch_blockers"]
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_frontier_roadmap_status_passes_dirty_paths_to_packet_selector(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="dirty_packet",
        code_paths=["src/tac/optimization/dirty_packet_owner.py"],
    )

    payload = build_roadmap_status(
        repo_root=REPO,
        dirty_paths=["src/tac/optimization/dirty_packet_owner.py"],
        packet_manifest_paths=[manifest],
    )

    packet_selection = payload["next_comprehensive_tranche"][
        "field_meta_candidate_packet_selection"
    ]
    row = packet_selection["rows"][0]
    assert packet_selection["dirty_blocked_candidate_count"] == 1
    assert row["candidate_id"] == "dirty_packet"
    assert row["dirty_blocked"] is True
    assert row["candidate_static_preflight_ready"] is False
    assert row["selection_decision"] == "refused_dirty_worktree_overlap"


def _packet_manifest(
    root: Path,
    *,
    candidate_id: str,
    code_paths: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"field meta packet archive")
    payload = {
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
        "lane_id": f"lane_{candidate_id}",
        "job_name": f"job_{candidate_id}",
        "dispatch_unlocked": True,
        "ready_for_exact_eval_dispatch_claim": True,
        "family_group": "fixture_packet_family",
        "pareto_scope": "fixture_packet_family",
        "interaction_assumptions": ["fixture_first_order_packet"],
        "archive": {
            "path": archive.as_posix(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bytes": archive.stat().st_size,
        },
        "fixed_runtime_preflight": {
            "ready_for_fixed_runtime_exact_eval": True,
            "runtime_tree_sha256": "c" * 64,
            "remaining_blockers": [],
        },
    }
    if code_paths is not None:
        payload["code_paths"] = code_paths
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
