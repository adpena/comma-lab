# SPDX-License-Identifier: MIT
"""Tests for the Candidate 4c no-spend launch packet doctor."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import verify_candidate4c_launch_packet as doctor  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_single_member_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", payload)


def _paid_command(audit_rel: str, sentinel_hash: str = "abc123") -> str:
    return (
        "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 "
        f"OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:{sentinel_hash} "
        f"OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON={audit_rel} "
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 "
        "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=1.250 "
        ".venv/bin/python tools/run_modal_smoke_before_full.py "
        "--recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch "
        "--operator-handle codex:z6_v2_candidate_4c_scorer_logit"
    )


def _write_recipe(tmp_path: Path, *, dispatch_enabled: bool = True) -> None:
    blockers = (
        "[]"
        if dispatch_enabled
        else (
            "\n  - "
            "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required"
        )
    )
    text = (
        "schema_version: 1\n"
        "name: substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch\n"
        "lane_id: lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518\n"
        "research_only: false\n"
        f"dispatch_enabled: {str(dispatch_enabled).lower()}\n"
        f"dispatch_blockers: {blockers}\n"
        "smoke_only: true\n"
        "smoke_validation_contract: training_artifact_v1\n"
        "target_modes:\n"
        "  - contest_one_video_replay\n"
    )
    path = (
        tmp_path
        / ".omx/operator_authorize_recipes/"
        "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_repo(tmp_path: Path, *, dispatch_enabled: bool = True) -> Path:
    _write_recipe(tmp_path, dispatch_enabled=dispatch_enabled)
    audit_rel = (
        ".omx/state/catalog202_sentinel_cleanliness/"
        "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T000000Z.json"
    )
    probe_rel = ".omx/research/z6_probe.json"
    _write_json(
        tmp_path / audit_rel,
        {
            "ready_for_catalog202_audit_backed_dirty_sentinel_attestation": True,
            "sentinel_set_sha256": "abc123",
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
        },
    )
    _write_json(
        tmp_path / probe_rel,
        {
            "verdict": "pending_paired_exact_eval_json",
            "score_claim": False,
            "promotion_eligible": False,
            "inflate_output_comparison": {
                "runtime_output_changed": True,
                "runtime_custody": {"aggregate_sha256": "runtime-sha"},
                "full_output_tree": {"aggregate_sha256": "full-sha"},
                "identity_output_tree": {"aggregate_sha256": "identity-sha"},
                "output_root": "experiments/results/z6/output-proof",
                "total_byte_differences": 22253,
            },
        },
    )
    _write_json(
        tmp_path / ".omx/state/asymptotic_pursuit/dispatch_queue_20260518T000001Z.json",
        {
            "top_ready_substrate": doctor.SUBSTRATE_ID,
            "top_ready_audit_backed_paid_launch_command": _paid_command(audit_rel),
            "current_worktree_dirty_path_count": 3,
            "immediately_runnable_paid_dispatch_count": 1,
            "top_ready_paid_launch_missing_preconditions": [],
            "top_immediately_runnable_paid_launch_command": _paid_command(audit_rel),
            "dispatch_sequence": [
                {
                    "substrate_id": doctor.SUBSTRATE_ID,
                    "ready_for_paid_dispatch": True,
                    "immediately_runnable_paid_launch": True,
                    "paid_launch_missing_preconditions": [],
                    "local_identity_disambiguator_probe": {
                        "path": probe_rel,
                        "verdict": "pending_paired_exact_eval_json",
                        "runtime_output_changed": True,
                        "custody": {
                            "runtime_custody_aggregate_sha256": "runtime-sha",
                            "full_output_aggregate_sha256": "full-sha",
                            "identity_output_aggregate_sha256": "identity-sha",
                            "output_root": "experiments/results/z6/output-proof",
                            "total_byte_differences": 22253,
                        },
                        "blockers": [],
                    },
                    "operator_session_authorization": {
                        "catalog202_dirty_tree_attestation": {
                            "required_for_paid_dispatch": True,
                            "satisfied_in_current_environment": True,
                            "dirty_sentinel_audit_required": True,
                            "latest_sentinel_audit_matches_current": True,
                            "env_sentinel_audit_matches_current": True,
                            "current_sentinel_snapshot_valid": True,
                            "latest_sentinel_audit": {
                                "path": audit_rel,
                                "dirty_sentinel_path_count": 1,
                                "sentinel_set_sha256": "abc123",
                                "ready_for_catalog202_audit_backed_dirty_sentinel_attestation": True,
                            },
                            "env_sentinel_audit": {
                                "path": audit_rel,
                                "dirty_sentinel_path_count": 1,
                                "sentinel_set_sha256": "abc123",
                                "ready_for_catalog202_audit_backed_dirty_sentinel_attestation": True,
                            }
                        }
                    },
                }
            ],
        },
    )
    _write_json(
        tmp_path
        / ".omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T000002Z.json",
        {
            "verdict": "approve",
            "findings": [],
            "cache_hit": False,
            "cache_key": "fixture",
            "invoked_at_utc": "2026-05-18T00:00:02Z",
            "elapsed_sec": 1.0,
        },
    )
    return tmp_path


def _write_full600_probe_pair(repo: Path) -> None:
    run_dir = repo / "experiments/results/candidate4c_full600_fixture"
    full_zip = run_dir / "archive.zip"
    identity_zip = run_dir / "archive_identity_predictor_disambiguator.zip"
    _write_single_member_zip(full_zip, b"full-payload")
    _write_single_member_zip(identity_zip, b"identity-payload")
    submission_dir = run_dir / "submission_dir"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    probe_path = repo / ".omx/research/z6_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["source_archives"] = [
        {
            "mode": "full_film_predictor",
            "identity_predictor": False,
            "num_pairs": 600,
            "zip_path": "experiments/results/candidate4c_full600_fixture/archive.zip",
            "zip_sha256": "f" * 64,
        },
        {
            "mode": "identity_predictor",
            "identity_predictor": True,
            "num_pairs": 600,
            "zip_path": (
                "experiments/results/candidate4c_full600_fixture/"
                "archive_identity_predictor_disambiguator.zip"
            ),
            "zip_sha256": "e" * 64,
        },
    ]
    probe_path.write_text(json.dumps(probe), encoding="utf-8")


def _runner(stdout_by_name: dict[str, str]):
    def run(cmd, cwd, env=None):
        command = " ".join(cmd)
        if "claim_lane_dispatch.py summary" in command:
            stdout = stdout_by_name.get(
                "claim",
                "CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=1",
            )
        elif "validate_dispatch_required_inputs.py" in command:
            stdout = stdout_by_name.get("required_inputs", "OK")
        elif "run_modal_smoke_before_full.py" in command:
            stdout = stdout_by_name.get("smoke_dry_run", "--dry-run")
        elif "tools/operator_authorize.py" in command:
            stdout = stdout_by_name.get("operator_dry_run", "--dry-run")
        elif "_whole_tree_clean_check_bypass_active" in command:
            stdout = stdout_by_name.get("bypass_probe", "True")
        else:
            stdout = "OK"
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    return run


def test_build_packet_ready_when_all_no_spend_checks_green(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is True
    assert payload["paid_training_launch_in_scope"] is True
    assert payload["current_mode"] == "paid_training_launch_surface"
    assert payload["provider_dispatch_attempted"] is False
    assert payload["lane_claim_opened"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["active_lane_claims_clean"] is True
    assert payload["queue_immediate_launch_ready"] is True
    assert payload["queue_immediate_launch_blockers"] == []
    assert payload["local_identity_disambiguator_probe_ready"] is True
    assert payload["local_identity_disambiguator_probe_blockers"] == []
    assert payload["local_identity_disambiguator_probe"]["custody"][
        "runtime_custody_aggregate_sha256"
    ] == "runtime-sha"
    assert payload["next_paid_command_ready"] is True
    assert payload["next_paid_command_blockers"] == []
    assert payload["catalog202_audit_backed_bypass_probe_accepted"] is True
    assert payload["codex_pre_dispatch_review_ready"] is True
    assert payload["codex_pre_dispatch_review_blockers"] == []
    assert {check["name"] for check in payload["checks"]} == {
        "lane_claim_summary",
        "required_input_validation",
        "smoke_before_full_dry_run",
        "operator_authorize_dry_run",
        "catalog202_audit_backed_bypass_probe",
    }


def test_build_packet_classifies_diagnostic_only_exact_eval_handoff_mode(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, dispatch_enabled=False)
    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["paid_training_launch_in_scope"] is False
    assert payload["current_mode"] == "diagnostic_only_exact_eval_handoff_required"
    assert payload["diagnostic_smoke_dry_run_ready"] is True
    assert payload["catalog202_audit_backed_bypass_probe_accepted"] is True
    bypass_check = next(
        check
        for check in payload["checks"]
        if check["name"] == "catalog202_audit_backed_bypass_probe"
    )
    assert bypass_check["skipped"] is True
    assert payload["next_paid_command_ready"] is False
    assert "candidate4c_paid_training_launch_not_in_scope_recipe_dispatch_disabled" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_recipe_dispatch_disabled_exact_eval_handoff_required" in payload[
        "result_review_blockers"
    ]
    assert payload["exact_eval_handoff"]["ready_for_exact_eval_handoff"] is False
    assert payload["exact_eval_handoff"]["provider_dispatch_attempted"] is False
    assert payload["exact_eval_handoff"]["lane_claim_opened"] is False
    assert payload["exact_eval_handoff"]["score_claim"] is False
    assert payload["exact_eval_handoff"]["promotion_eligible"] is False
    assert payload["exact_eval_handoff"]["modal_commands_after_full_600_pair_packet"] == {}
    assert "<harvested_full_600_pair_run_dir>/archive.zip" in payload[
        "exact_eval_handoff"
    ]["modal_command_templates_after_full_600_pair_packet"][
        "full_paired_contest_cpu_cuda"
    ]
    assert "tools/dispatch_modal_paired_auth_eval.py" in payload[
        "exact_eval_handoff"
    ]["modal_command_templates_after_full_600_pair_packet"][
        "full_paired_contest_cpu_cuda"
    ]
    assert "<full_archive_sha256_prefix>" in payload[
        "exact_eval_handoff"
    ]["modal_command_templates_after_full_600_pair_packet"][
        "full_paired_contest_cpu_cuda"
    ]
    assert "<identity_archive_sha256_prefix>" in payload[
        "exact_eval_handoff"
    ]["modal_command_templates_after_full_600_pair_packet"][
        "identity_paired_contest_cpu_cuda"
    ]
    assert "candidate4c_exact_handoff_latest_archive_pair_not_600_pairs" in payload[
        "result_review_blockers"
    ]


def test_exact_eval_handoff_uses_canonical_paired_dispatcher(
    tmp_path: Path,
) -> None:
    """Exact-eval handoff must expose paired CPU/CUDA dispatch, not single-axis wrappers."""
    repo = _fixture_repo(tmp_path, dispatch_enabled=False)
    _write_full600_probe_pair(repo)

    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))
    handoff = payload["exact_eval_handoff"]

    assert handoff["ready_for_exact_eval_handoff"] is True
    assert handoff["latest_pair_count"] == 600
    assert handoff["blockers"] == []
    assert handoff["modal_plan_commands_after_full_600_pair_packet"].keys() == {
        "full_paired_contest_cpu_cuda",
        "identity_paired_contest_cpu_cuda",
    }
    assert handoff["modal_commands_after_full_600_pair_packet"].keys() == {
        "full_paired_contest_cpu_cuda",
        "identity_paired_contest_cpu_cuda",
    }
    for command in handoff["modal_commands_after_full_600_pair_packet"].values():
        assert "tools/dispatch_modal_paired_auth_eval.py" in command
        assert "--execute" in command
        assert "--expected-runtime-tree-sha256 auto" in command
        assert "--skip-axis-if-promotable-anchor-exists" in command
        assert "experiments/modal_auth_eval.py" not in command
        assert "experiments/modal_auth_eval_cpu.py" not in command

    assert "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_full" in (
        handoff["modal_commands_after_full_600_pair_packet"][
            "full_paired_contest_cpu_cuda"
        ]
    )
    assert "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518_identity" in (
        handoff["modal_commands_after_full_600_pair_packet"][
            "identity_paired_contest_cpu_cuda"
        ]
    )


def test_build_packet_refuses_queue_row_that_is_not_immediately_runnable(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    queue_path = next((repo / ".omx/state/asymptotic_pursuit").glob("*.json"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    missing = [
        "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch"
    ]
    queue["immediately_runnable_paid_dispatch_count"] = 0
    queue["top_ready_paid_launch_missing_preconditions"] = missing
    queue["top_immediately_runnable_paid_launch_command"] = None
    row = queue["dispatch_sequence"][0]
    row["immediately_runnable_paid_launch"] = False
    row["paid_launch_missing_preconditions"] = missing
    row["operator_session_authorization"]["catalog202_dirty_tree_attestation"][
        "satisfied_in_current_environment"
    ] = False
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["queue_immediate_launch_ready"] is False
    assert "candidate4c_queue_row_not_immediately_runnable" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_queue_top_ready_paid_launch_preconditions_missing" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_queue_no_immediately_runnable_paid_dispatch" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_queue_catalog202_env_attestation_not_satisfied" in payload[
        "result_review_blockers"
    ]


def test_build_packet_refuses_catalog202_bypass_probe_stdout_false(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    payload = doctor.build_packet(
        repo_root=repo,
        runner=_runner({"bypass_probe": "False\n"}),
    )

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["catalog202_audit_backed_bypass_probe_accepted"] is False
    probe_check = next(
        check
        for check in payload["checks"]
        if check["name"] == "catalog202_audit_backed_bypass_probe"
    )
    assert probe_check["returncode"] == 0
    assert probe_check["ok"] is False
    assert "candidate4c_catalog202_audit_backed_bypass_probe_not_true" in payload[
        "result_review_blockers"
    ]


def test_build_packet_refuses_blocking_codex_pre_dispatch_review(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path)
    review_path = (
        repo
        / ".omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T000003Z.json"
    )
    _write_json(
        review_path,
        {
            "verdict": "needs-attention",
            "findings": [],
            "raw_output_excerpt": (
                "Verdict: needs-attention\n"
                "Findings:\n"
                "- [high] contest-CUDA smoke contract is incompatible"
            ),
            "cache_hit": False,
            "cache_key": "blocking",
            "invoked_at_utc": "2026-05-18T00:00:03Z",
            "elapsed_sec": 2.0,
        },
    )

    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["codex_pre_dispatch_review_ready"] is False
    assert payload["codex_pre_dispatch_review"]["path"].endswith(
        "candidate4c_codex_pre_dispatch_review_20260518T000003Z.json"
    )
    assert payload["codex_pre_dispatch_review"]["verdict"] == "needs-attention"
    assert payload["codex_pre_dispatch_review"]["findings"] == [
        "- [high] contest-CUDA smoke contract is incompatible"
    ]
    assert "candidate4c_codex_pre_dispatch_review_blocking_needs-attention" in payload[
        "result_review_blockers"
    ]


def test_build_packet_refuses_active_claims(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    payload = doctor.build_packet(
        repo_root=repo,
        runner=_runner({"claim": "CLAIM_SUMMARY active=1 stale_nonterminal=0"}),
    )

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["active_lane_claims_clean"] is False
    assert payload["result_review_blockers"] == [
        "candidate4c_no_spend_launch_packet_checks_not_all_green"
    ]


def test_build_packet_refuses_missing_disambiguator_custody(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    queue_path = next((repo / ".omx/state/asymptotic_pursuit").glob("*.json"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["dispatch_sequence"][0]["local_identity_disambiguator_probe"]["custody"] = {
        "runtime_custody_aggregate_sha256": "runtime-sha",
        "total_byte_differences": 22253,
    }
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["local_identity_disambiguator_probe_ready"] is False
    assert "candidate4c_local_identity_disambiguator_full_output_aggregate_sha256_missing" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_local_identity_disambiguator_identity_output_aggregate_sha256_missing" in payload[
        "result_review_blockers"
    ]


def test_build_packet_refuses_stale_paid_command_audit_hash(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    queue_path = next((repo / ".omx/state/asymptotic_pursuit").glob("*.json"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["top_ready_audit_backed_paid_launch_command"] = _paid_command(
        ".omx/state/catalog202_sentinel_cleanliness/stale.json",
        sentinel_hash="old-hash",
    )
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["next_paid_command_ready"] is False
    assert "candidate4c_next_paid_command_missing_catalog202_audit_env" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_next_paid_command_missing_catalog202_sentinel_hash" in payload[
        "result_review_blockers"
    ]


def test_build_packet_refuses_stale_disambiguator_artifact(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    probe_path = repo / ".omx/research/z6_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["inflate_output_comparison"]["runtime_output_changed"] = False
    probe["inflate_output_comparison"]["full_output_tree"][
        "aggregate_sha256"
    ] = "stale-full-sha"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")

    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))

    assert payload["ready_for_operator_paid_execution"] is False
    assert payload["local_identity_disambiguator_probe_ready"] is False
    assert "candidate4c_local_identity_disambiguator_artifact_runtime_output_not_changed" in payload[
        "result_review_blockers"
    ]
    assert "candidate4c_local_identity_disambiguator_full_output_aggregate_sha256_artifact_mismatch" in payload[
        "result_review_blockers"
    ]


def test_write_artifact_preserves_false_authority_flags(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    payload = doctor.build_packet(repo_root=repo, runner=_runner({}))
    path = doctor.write_artifact(payload, repo_root=repo)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.startswith("candidate4c_no_spend_launch_packet_")
    assert persisted["score_claim"] is False
    assert persisted["promotion_eligible"] is False
    assert persisted["provider_dispatch_attempted"] is False
    assert persisted["lane_claim_opened"] is False
