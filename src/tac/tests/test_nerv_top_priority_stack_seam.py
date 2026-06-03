# SPDX-License-Identifier: MIT
"""Tests for the fail-closed NeRV top-priority stack seam."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from tac.analysis.nerv_top_priority_stack_seam import (
    FULL_STACK_COMPONENTS,
    SCHEMA,
    build_nerv_top_priority_stack_seam,
    build_source_faithfulness_matrix,
    discover_dispatch_blockers,
)


def test_top_priority_seam_is_fail_closed_and_orders_carriers(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    upstream = _upstream(tmp_path)
    intake = _pr95_intake(tmp_path)
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "| 2026-06-02T03:22:22Z | codex | "
        "lane_pr101_storage_order_len24_exact_cpu_20260601 | modal | job | "
        "2026-06-02T06:17:34Z | active_modal_cpu_auth_eval_pending_recovery_poll | |\n",
        encoding="utf-8",
    )

    payload = build_nerv_top_priority_stack_seam(
        repo_root=repo,
        upstream_repo_dir=upstream,
        pr95_intake_root=intake,
        active_claims_path=claims,
        pr95_pr_metadata={
            "url": "https://github.com/commaai/comma_video_compression_challenge/pull/95",
            "title": "hnerv_muon submission (0.20)",
            "state": "MERGED",
            "headRefOid": _git_head(upstream),
            "headRefName": "hnerv_muon",
        },
        oss_source_metadata={
            "snerv": {
                "repo_url": "https://github.com/qwertja/SNeRV.git",
                "head_sha": "0844a08f9591eea9625f8b961ed91d08030e06d1",
                "audit_root": "/Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/SNeRV",
            },
            "hinerv": {
                "repo_url": "https://github.com/hmkx/HiNeRV.git",
                "head_sha": "fdb92ec22492246f800621dfd454f6a5c62ab75b",
                "audit_root": "/Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/HiNeRV",
            },
            "hnerv_pr95_control": {
                "repo_url": "https://github.com/haochen-rye/HNeRV.git",
                "head_sha": "4872129c8d004a25477e0c1ffbbff4ba71943ad5",
                "audit_root": "/Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/HNeRV",
            },
        },
        generated_utc="2026-06-02T03:40:00+00:00",
    )

    assert payload["schema"] == SCHEMA
    assert payload["go_no_go_verdict"] == (
        "GO_LOCAL_STACK_OPTIMIZATION__NO_GO_PRODUCTION_HARDENED_OR_EXACT_CLAIM"
    )
    assert payload["top_priority_carriers"] == ["snerv", "hinerv"]
    assert payload["priority_policy"]["individually_fractally_optimized_full_stacks"]
    assert payload["priority_policy"][
        "shared_synergy_surfaces_do_not_collapse_carrier_specific_work"
    ]
    assert payload["full_stack_priority"]["components"] == list(FULL_STACK_COMPONENTS)
    assert payload["baseline_to_beat"] == "pr95_hnerv_muon"
    assert payload["baseline"]["archive"]["bytes"] > 0
    assert payload["baseline"]["submission"] == "hnerv_muon"
    assert payload["baseline"]["blockers"] == []
    assert payload["blocked_dispatch"] is True
    assert payload["score_claim"] is False
    assert payload["frontier_score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["exact_or_full_video_launched"] is False
    assert payload["production_hardened_claim"] is False
    assert payload["source_faithful_stack_claim"] is False
    assert payload["priority_policy"]["no_fake_implementations_allowed"] is True
    assert payload["modelsize_archive_budget_policy"]["verdict"] == (
        "NO_GO_FULL_LEVERAGE_UNTIL_MODEL_SIZE_TO_ARCHIVE_BYTES_CURVE_EXISTS"
    )
    assert "--modelsize" in payload["modelsize_archive_budget_policy"][
        "official_controls_to_bind"
    ]
    assert payload["modelsize_archive_budget_policy"]["contest_inversion_target"][
        "byte_caps_to_sweep"
    ] == [36_000, 72_000, 120_000, 150_000, 178_417]
    assert "nerv_modelsize_to_archive_bytes_curve_missing" in payload[
        "modelsize_archive_budget_policy"
    ]["production_blockers"]
    assert (
        "official_modelsize_flag_not_receiver_closed_under_contest_byte_caps"
        in payload["modelsize_archive_budget_policy"]["production_blockers"]
    )
    assert "official SNeRV --modelsize/fc_dim solving is bound" in payload[
        "modelsize_archive_budget_policy"
    ]["current_gap"]["snerv"]
    assert "0/2/4/6/7/8/16/32 decoder-waterfill actions" in payload[
        "modelsize_archive_budget_policy"
    ]["current_gap"]["hi_nerv"]
    scorer_policy = payload["scorer_domain_control_policy"]
    assert scorer_policy["schema"] == "nerv_scorer_domain_control_policy.v1"
    assert scorer_policy["scorer_domains"]["segnet"]["frame_indices_per_pair"] == [1]
    assert scorer_policy["scorer_domains"]["posenet"]["frame_indices_per_pair"] == [
        0,
        1,
    ]
    assert scorer_policy["scorer_domains"]["rate"]["water_level_fixed_by_contest"] is True
    assert "src/tac/master_gradient.py" in scorer_policy["reusable_surfaces_to_consume"]
    assert "tools/cathedral_autopilot.py" in scorer_policy["reusable_surfaces_to_consume"]
    assert "nerv_scorer_domain_controls_not_bound_to_both_snerv_and_hinerv_train_export" in payload[
        "blockers"
    ]
    modelsize_policy = payload["optimal_modelsize_control_policy"]
    assert modelsize_policy["schema"] == "nerv_optimal_modelsize_control_policy.v1"
    component_budgets = modelsize_policy["fine_grained_component_budget_controls"]
    assert "lf_plane_payload" in component_budgets["snerv_sections"]
    assert "hierarchical_feature_grids" in component_budgets["hinerv_sections"]
    assert "decoder_weight_sections" in component_budgets["shared_sections"]
    assert "int1_int2_int4_int6_int7_int8" in component_budgets["per_section_actions"]
    assert modelsize_policy["different_for_snerv"]["main_risk"].startswith(
        "explicit LF storage"
    )
    assert modelsize_policy["different_for_hinerv"]["main_risk"].startswith(
        "decoder/grid weights"
    )
    assert "contest_optimal_modelsize_controller_not_yet_bound_to_training_loop" in payload[
        "blockers"
    ]
    assert payload["source_faithfulness"]["policy"][
        "bad_scores_from_non_source_faithful_stacks_are_bug_signals"
    ]
    assert payload["source_faithfulness"]["production_hardened_claim"] is False
    official = {
        row["stack_id"]: row
        for row in payload["source_faithfulness"]["official_sources"]
    }
    assert official["snerv"]["observed_source"]["head_sha"] == (
        "0844a08f9591eea9625f8b961ed91d08030e06d1"
    )
    assert official["snerv"]["parity_status"] == (
        "oss_snapshot_observed_not_yet_contest_parity_proven"
    )
    local_audit = {
        row["stack_id"]: row
        for row in payload["source_faithfulness"]["local_implementation_audit"]
    }
    assert local_audit["snerv"]["status"] == (
        "simplified_contest_adapter_not_source_faithful"
    )
    assert "official_MFU_multi_resolution_fusion_blocks" in local_audit["snerv"][
        "missing_source_features"
    ]
    assert "official_modelsize_fc_dim_parameter_budget_solver" in local_audit[
        "snerv"
    ]["implemented_features"]
    assert "official_SNeRV_T_Haar_DWT1D_temporal_primitive" in local_audit[
        "snerv"
    ]["implemented_features"]
    assert "official_modelsize_fc_dim_parameter_budget_solver" not in local_audit[
        "snerv"
    ]["missing_source_features"]
    assert "official_SNeRV_T_temporal_neighbor_path" not in local_audit[
        "snerv"
    ]["missing_source_features"]
    assert "official_SNeRV_T_full_neighbor_TUB_training_path" in local_audit[
        "snerv"
    ]["missing_source_features"]
    assert "invalid_official_modelsize_rows_preserved_false_authority" in local_audit[
        "snerv"
    ]["implemented_features"]
    assert local_audit["hinerv"]["status"] == "l0_sketch_not_source_faithful"
    assert "official_hierarchical_feature_grid_encoding" in local_audit["hinerv"][
        "missing_source_features"
    ]
    assert "decoder_waterfill_action_lattice_0_2_4_6_7_8_16_32" in local_audit[
        "hinerv"
    ]["implemented_features"]
    assert "quant_noise_action_lattice_2_4_6_7_8" in local_audit["hinerv"][
        "implemented_features"
    ]
    assert "official_prune_quant_ste_torchac_pipeline_parity" in local_audit[
        "hinerv"
    ]["missing_source_features"]
    assert payload["synergy_enhancers"][0]["enhancer_id"] == (
        "sr_nerv_trained_scorer_aware"
    )
    assert all(
        enhancer["not_a_standalone_carrier_stack"]
        for enhancer in payload["synergy_enhancers"]
    )
    work_orders = {
        order["stack_id"]: order["work_order"]
        for order in payload["fractal_work_orders"]
    }
    assert set(work_orders) == {"snerv", "hinerv"}
    assert [item["component"] for item in work_orders["snerv"]] == list(
        FULL_STACK_COMPONENTS
    )
    assert [item["component"] for item in work_orders["hinerv"]] == list(
        FULL_STACK_COMPONENTS
    )
    snerv_archive = next(
        item for item in work_orders["snerv"] if item["component"] == "archive_grammar"
    )
    assert snerv_archive["requires_receiver_byte_accounting"] is True
    assert snerv_archive["promotion_authority"] is False
    assert "mixed decoder modes" in snerv_archive["next_action"]
    hinerv_allocator = next(
        item for item in work_orders["hinerv"] if item["component"] == "allocator"
    )
    assert "joint P18/P19" in hinerv_allocator["next_action"]
    snerv_qat = next(
        action
        for action in payload["next_local_actions"]
        if action["id"] == "snerv_pair_robust_decoder_qat_continuation"
    )
    assert "--search-mode nes_pair_robust" in snerv_qat["command"]
    assert "--byte-pressure-multiplier 8.0" in snerv_qat["command"]
    assert "--max-archive-byte-growth 0" in snerv_qat["command"]
    assert "--seg-slack 0.00005" in snerv_qat["command"]
    assert "--pose-hard-guard" not in snerv_qat["command"]
    assert (
        "pr101_cpu_recovery_pending_blocks_new_exact_or_full_video"
        in payload["dispatch_blockers"]
    )
    assert (
        "full_600_byte_closed_receiver_proof_missing_for_snerv_and_hinerv"
        in payload["blockers"]
    )
    assert "snerv_simplified_contest_adapter_not_source_faithful" in payload[
        "blockers"
    ]
    assert "hinerv_l0_sketch_not_source_faithful" in payload["blockers"]


def test_missing_pr95_intake_blocks_baseline_authority(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    upstream = _upstream(tmp_path)
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text("", encoding="utf-8")

    payload = build_nerv_top_priority_stack_seam(
        repo_root=repo,
        upstream_repo_dir=upstream,
        pr95_intake_root=tmp_path / "missing_pr95",
        active_claims_path=claims,
        generated_utc="2026-06-02T03:40:00+00:00",
    )

    assert payload["baseline"]["archive"] is None
    assert "pr95_public_intake_root_missing" in payload["baseline"]["blockers"]
    assert "pr95_public_intake_root_missing" in payload["blockers"]
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_source_faithfulness_matrix_blocks_missing_oss_and_sketch_claims() -> None:
    payload = build_source_faithfulness_matrix({})

    assert payload["verdict"].startswith("NO_GO_PRODUCTION_HARDENED_CLAIM")
    assert payload["production_hardened_claim"] is False
    assert payload["source_faithful_stack_claim"] is False
    assert payload["policy"]["minimal_or_sketch_adapters_are_local_only"] is True
    assert "snerv_official_oss_snapshot_missing" in payload["blockers"]
    assert "hinerv_official_oss_snapshot_missing" in payload["blockers"]
    assert "hnerv_pr95_control_official_oss_snapshot_missing" in payload["blockers"]
    assert "snerv_simplified_contest_adapter_not_source_faithful" in payload[
        "blockers"
    ]
    assert "hinerv_l0_sketch_not_source_faithful" in payload["blockers"]


def test_nongit_upstream_blocks_baseline_authority(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    upstream = tmp_path / "plain_upstream"
    upstream.mkdir()
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text("", encoding="utf-8")

    payload = build_nerv_top_priority_stack_seam(
        repo_root=repo,
        upstream_repo_dir=upstream,
        pr95_intake_root=_pr95_intake(tmp_path),
        active_claims_path=claims,
        generated_utc="2026-06-02T03:40:00+00:00",
    )

    assert "upstream_repo_git_head_missing" in payload["baseline"]["blockers"]
    assert "upstream_repo_git_remote_origin_missing" in payload["baseline"]["blockers"]
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_pr95_metadata_absence_or_head_mismatch_blocks_baseline_authority(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    upstream = _upstream(tmp_path)
    intake = _pr95_intake(tmp_path)
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text("", encoding="utf-8")

    missing = build_nerv_top_priority_stack_seam(
        repo_root=repo,
        upstream_repo_dir=upstream,
        pr95_intake_root=intake,
        active_claims_path=claims,
        generated_utc="2026-06-02T03:40:00+00:00",
    )
    assert "pr95_pr_metadata_missing" in missing["baseline"]["blockers"]
    assert "pr95_pr_head_sha_missing" in missing["baseline"]["blockers"]
    assert "pr95_pr_head_ref_missing" in missing["baseline"]["blockers"]

    mismatched = build_nerv_top_priority_stack_seam(
        repo_root=repo,
        upstream_repo_dir=upstream,
        pr95_intake_root=intake,
        active_claims_path=claims,
        pr95_pr_metadata={
            "url": "https://github.com/commaai/comma_video_compression_challenge/pull/95",
            "title": "hnerv_muon submission (0.20)",
            "state": "MERGED",
            "headRefOid": "9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9",
            "headRefName": "hnerv_muon",
        },
        generated_utc="2026-06-02T03:40:00+00:00",
    )
    assert "upstream_repo_head_sha_mismatch_pr95_metadata" in mismatched[
        "baseline"
    ]["blockers"]
    assert "upstream_repo_head_sha_mismatch_pr95_metadata" in mismatched["blockers"]


def test_dispatch_blocker_parser_finds_named_and_generic_claims(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | "
        "predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-06-02T03:22:22Z | codex | "
        "lane_pr101_storage_order_len24_exact_cpu_20260601 | modal | job | "
        "2026-06-02T06:17:34Z | active_modal_cpu_auth_eval_pending_recovery_poll | |\n"
        "| 2026-05-31T14:31:04Z | claude | "
        "lane_z5_rao_ballard_paired_cuda_ratification_wave2a_20260531_contest_cuda | "
        "modal | job | 2026-05-31T17:31:04Z | active_modal_auth_eval_spawned | |\n"
        "| 2026-06-02T04:00:00Z | codex | lane_other_exact_eval | lightning | "
        "job | 2026-06-02T05:00:00Z | active_dispatching | exact eval |\n"
        "| 2026-06-02T04:20:00Z | codex | lane_closed_exact_eval | lightning | "
        "job | | completed_score_0_2 | exact eval terminal |\n"
        "| 2026-06-02T04:05:00Z | codex | lane_closed_exact_eval | lightning | "
        "job | 2026-06-02T05:00:00Z | active_dispatching | exact eval older |\n"
        "| 2026-06-02T04:01:00Z | codex | lane_old_exact_eval | lightning | "
        "job | | completed_score_0_2 | exact eval terminal |\n"
        "| 2026-05-20T04:00:00Z | codex | lane_stale_exact_eval | lightning | "
        "job | 2026-05-20T05:00:00Z | active_dispatching | exact eval stale |\n",
        encoding="utf-8",
    )

    blockers = discover_dispatch_blockers(
        claims,
        now_utc="2026-06-02T04:30:00+00:00",
    )

    assert "pr101_cpu_recovery_pending_blocks_new_exact_or_full_video" in blockers
    assert "z5_rao_ballard_modal_claims_still_need_terminal_adjudication" in blockers
    assert "active_exact_or_full_video_claim:lane_other_exact_eval" in blockers
    assert "active_exact_or_full_video_claim:lane_closed_exact_eval" not in blockers
    assert "active_exact_or_full_video_claim:lane_old_exact_eval" not in blockers
    assert "active_exact_or_full_video_claim:lane_stale_exact_eval" not in blockers


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "prove_pr95_public_archive_runtime_consumption.py").write_text(
        "# proof tool\n", encoding="utf-8"
    )
    return repo


def _upstream(tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "codex@example.invalid"],
        cwd=upstream,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex"],
        cwd=upstream,
        check=True,
        capture_output=True,
    )
    (upstream / "README.md").write_text("upstream\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=upstream, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=upstream,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/commaai/comma_video_compression_challenge.git",
        ],
        cwd=upstream,
        check=True,
        capture_output=True,
    )
    return upstream


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pr95_intake(tmp_path: Path) -> Path:
    root = tmp_path / "public_pr95_intake_20260505_auto"
    submission = root / "source" / "submissions" / "hnerv_muon"
    (submission / "src" / "stages").mkdir(parents=True)
    with zipfile.ZipFile(root / "archive.zip", "w") as zf:
        zf.writestr("0.bin", b"pr95")
    for rel in (
        "inflate.sh",
        "inflate.py",
        "src/model.py",
        "src/codec.py",
        "src/optim.py",
        "src/stages/stage8_muon_finetune.py",
    ):
        path = submission / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# runtime\n", encoding="utf-8")
    return root
