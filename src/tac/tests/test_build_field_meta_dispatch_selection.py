from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tac.frontier_rows import FRONTIER_ROW_FIELDS, FRONTIER_ROW_SCHEMA
from tools.build_field_meta_dispatch_selection import build_selection_report

REPO = Path(__file__).resolve().parents[3]


def test_field_meta_selector_reports_static_ready_but_not_dispatch_ready_without_claim(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="closed_candidate",
        lane_id="lane_closed_candidate",
        job_name="job_closed_candidate",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_static_preflight_ready"] is True
    assert report["candidate_static_preflight_ready_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_candidate_count"] == 0
    assert row["candidate_static_preflight_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["strict_candidate_preflight_ready"] is True
    assert row["strict_candidate_static_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["dispatch_identity_proof"]["status"] == "passed"
    assert row["dispatch_claim_proof"]["checked"] is False
    assert "missing_active_lane_dispatch_claim" in row["candidate_blockers"]
    assert row["candidate_preflight_next_required_proof"] == [
        "matching_active_level2_lane_claim_for_manifest_lane_and_job",
    ]
    assert row["next_required_proof"] == [
        "matching_active_level2_lane_claim_for_manifest_lane_and_job",
        "passed_kkt_proof_or_converged_admm_waterline_result",
    ]
    assert row["selection_decision"] == "static_candidate_acquire_kkt_and_lane_claim_before_dispatch"


def test_field_meta_selector_summarizes_operator_next_steps_without_dispatch(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="operator_packet",
        lane_id="lane_operator_packet",
        job_name="job_operator_packet",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    refresh_command = ".venv/bin/python tools/build_packet.py --json-out packet.json"
    verify_command = ".venv/bin/python -c 'import os; assert os.environ.get(\"LIGHTNING_SSH_TARGET\")'"
    payload.update(
        {
            "missing_env": ["LIGHTNING_SSH_TARGET"],
            "operator_approved_exact_cuda": False,
            "lane_claim_preflight": {
                "active_claim_present": False,
                "claims_path_exists": True,
                "conflict_present": False,
            },
            "static_compliance_refresh": {
                "command": refresh_command,
                "returncode": 0,
            },
            "operator_next_steps": {
                "schema": "fixture_operator_next_steps_v1",
                "copy_safe": True,
                "must_run_in_order": True,
                "first_remote_gpu_step": "submit_exact_cuda",
                "current_blockers": [
                    "missing_lightning_environment",
                    "missing_active_lane_dispatch_claim",
                    "missing_operator_exact_cuda_approval",
                ],
                "steps": [
                    {
                        "id": "verify_lightning_env",
                        "order": 1,
                        "copy_safe_command": verify_command,
                        "dispatches_remote_gpu": False,
                        "writes_repo_state": False,
                    },
                    {
                        "id": "submit_exact_cuda",
                        "order": 2,
                        "copy_safe_command": "scripts/lightning_exact_eval_repro.py --submit",
                        "dispatches_remote_gpu": True,
                        "writes_repo_state": True,
                    },
                ],
            },
        }
    )
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    summary = row["operator_next_steps_summary"]
    assert row["score_claim"] is False
    assert row["dispatch_attempted"] is False
    assert summary["packet_operator_next_steps_schema"] == "fixture_operator_next_steps_v1"
    assert summary["first_remote_gpu_step"] == "submit_exact_cuda"
    assert summary["local_non_gpu_step_ids"] == ["verify_lightning_env"]
    assert summary["remote_gpu_step_ids"] == ["submit_exact_cuda"]
    assert row["next_local_non_gpu_action"]["id"] == "verify_lightning_env"
    assert row["next_local_non_gpu_command"] == verify_command
    assert row["operator_claim_blockers"] == ["missing_active_lane_dispatch_claim"]
    assert row["operator_refresh_blockers"] == []
    assert row["operator_next_steps_summary"]["static_refresh_status"] == "passed"
    assert row["operator_next_steps_summary"]["static_refresh_command"] == refresh_command
    assert row["operator_next_steps_summary"]["static_refresh_source"] == "static_compliance_refresh"
    assert row["operator_next_steps_summary"]["static_refresh_schema"] == ""
    assert row["operator_approval_blockers"] == ["missing_operator_exact_cuda_approval"]
    assert row["operator_environment_blockers"] == [
        "missing_lightning_environment",
        "missing_env:LIGHTNING_SSH_TARGET",
    ]
    assert "missing_lightning_environment" in row["exact_dispatch_blockers"]["blockers"]
    assert (
        "missing_env:LIGHTNING_SSH_TARGET"
        in row["exact_dispatch_blockers"]["blockers"]
    )
    assert row["exact_dispatch_blockers"]["blocker_categories"]["environment"] == [
        "missing_lightning_environment",
        "missing_env:LIGHTNING_SSH_TARGET",
    ]
    assert (
        "lightning_or_remote_exact_eval_environment_available"
        in row["exact_dispatch_blockers"]["next_required_proof"]
    )
    assert row["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_records_operator_approval_context_without_unlocking_dispatch(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="approval_context_candidate",
        lane_id="lane_approval_context_candidate",
        job_name="job_approval_context_candidate",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["operator_next_steps"] = {
        "schema": "fixture_operator_next_steps_v1",
        "copy_safe": True,
        "current_submit_blockers": [
            "missing_operator_exact_cuda_approval",
            "missing_active_lane_dispatch_claim",
        ],
        "steps": [],
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        operator_approved_exact_cuda=True,
    )

    row = report["rows"][0]
    assert report["operator_approval_state"]["approved"] is True
    assert report["operator_approval_state"]["dispatch_unlocked_by_approval"] is False
    assert report["adversarial_gate_policy"]["score_claim_requires_exact_cuda_and_adversarial_review"] is True
    assert row["operator_approval_state"]["source"] == "selector_context_operator_approved_exact_cuda"
    assert row["operator_approval_blockers"] == []
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "missing_active_lane_dispatch_claim" in row["field_selection_blockers"]
    assert "requires_adversarial_review_before_score_claim" in report["dispatch_blockers"]


def test_field_meta_selector_ingests_hnerv_lowlevel_result_manifest_as_local_candidate() -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/result.json"
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_count"] == 1
    assert report["candidate_static_preflight_ready_count"] == 0
    assert report["ready_candidate_count"] == 0
    assert row["candidate_id"] == "pr106x_lgblock16_lowlevel_brotli_1byte"
    assert row["family_group"] == "hnerv_lowlevel_brotli_repack"
    assert row["evidence_grade"] == (
        "empirical local archive candidate; raw-equivalence audit only until exact CUDA"
    )
    assert row["interaction_assumptions"] == [
        "rate_only_raw_equivalent_brotli_repack",
        "component_deltas_require_exact_cuda_confirmation",
    ]
    assert row["field_interaction_contract"]["status"] == "passed"
    assert row["byte_delta"] == -1
    assert row["expected_total_score_delta"] == -6.65859e-7
    assert row["score_claim"] is False
    assert row["dispatch_attempted"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["archive_proof"]["byte_closed"] is True
    assert row["archive_proof"]["path"] == (
        "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/"
        "pr106x_lowlevel_brotli_hnerv_brotli_repack_candidate.zip"
    )
    assert row["runtime_proof"]["runtime_closed"] is True
    assert "strict_candidate_preflight_not_ready" in row["candidate_blockers"]
    assert "strict:ready_for_exact_eval_dispatch_false" in row["candidate_blockers"]
    assert "missing_dispatch_identity_for_lane_claim" in row["exact_dispatch_blockers"]["blockers"]
    assert "kkt:kkt_proof_or_admm_result_missing" in row["exact_dispatch_blockers"]["blockers"]
    assert row["selection_decision"] == "strict_candidate_preflight_refused"


def test_field_meta_selector_uses_hnerv_rate_only_packet_delta(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="pr106_q10_151byte_brotli",
        family_group="hnerv_lowlevel_brotli_repack",
        pareto_scope="hnerv_rate_only_exact_archive",
        lane_id="pr106_q10_151byte_brotli",
        job_name="exact_eval_pr106_q10_151byte_brotli_20260507",
        expected_score_delta=0.0,
        byte_delta=-151,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("expected_total_score_delta")
    payload["family"] = "hnerv_lowlevel_brotli_repack"
    payload["expected_total_score_delta_rate_only"] = -0.00010054470192144788
    payload.pop("interaction_assumptions")
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["byte_delta"] == -151
    assert row["expected_total_score_delta"] == -0.00010054470192144788
    assert row["field_selection_score_delta"] == -0.000100544702
    assert row["rate_only_delta_proof"]["status"] == "passed"
    assert row["rate_only_delta_proof"]["byte_delta"] == -151
    assert row["rate_only_delta_proof"]["expected_total_score_delta_source"] == (
        "expected_total_score_delta_rate_only"
    )
    assert row["interaction_assumptions"] == [
        "rate_only_raw_equivalent_brotli_repack",
        "component_deltas_require_exact_cuda_confirmation",
    ]
    assert row["field_interaction_contract"]["status"] == "passed"
    assert "field_interaction_contract_blocked" not in row["exact_dispatch_blockers"]["blockers"]


def test_field_meta_selector_normalizes_apogee_intn_forensic_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "apogee_int6_archive.zip"
    info = zipfile.ZipInfo("0.bin")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, bytes([0xA6]) + b"fixture-apogee-int6")
    metadata = {
        "archive_path": archive.as_posix(),
        "archive_size_bytes": archive.stat().st_size,
        "bits": 6,
        "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "delta_bytes": -15789,
        "dispatch_blockers": [
            "missing_contest_faithful_distortion_model",
            "missing_scorer_basin_parity_gate",
            "byte_only_prediction_not_score_evidence",
        ],
        "distortion_model_status": "missing",
        "rate_component_score_delta_vs_pr106": -0.010513247010845963,
        "ready_for_exact_eval_dispatch": False,
        "score_affecting_payload_changed": True,
        "score_claim": False,
        "scorer_basin_parity_status": "missing",
        "source_archive_sha256": "3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58",
    }
    manifest = tmp_path / "repack_metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["candidate_id"] == "apogee_int6"
    assert row["family_group"] == "apogee_intN"
    assert row["archive_proof"]["byte_closed"] is True
    assert row["archive_proof"]["sha256_expected"] == metadata["candidate_archive_sha256"]
    assert row["byte_delta"] == -15789
    assert row["expected_total_score_delta"] == -0.010513247010845963
    assert row["proxy_row"] is True
    assert row["candidate_static_preflight_ready"] is False
    assert "strict:ready_for_exact_eval_dispatch_false" in row["candidate_blockers"]
    assert "strict:missing_distortion_model_gate" in row["candidate_blockers"]
    assert row["rate_only_delta_proof"]["status"] == "not_applicable"


def test_field_meta_selector_ingests_apogee_parity_evidence_as_calibration(tmp_path: Path) -> None:
    archive = tmp_path / "apogee_int6_archive.zip"
    info = zipfile.ZipInfo("0.bin")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, bytes([0xA6]) + b"fixture-apogee-int6")
    evidence = tmp_path / "parity_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "evidence_semantics": "scorer_basin_parity_gate",
                "parity_report": {
                    "pose_dist_lossless": 0.0001678224216448143,
                    "pose_dist_quantized": 0.00027574982959777117,
                    "seg_dist_delta": 0.0009618123876862228,
                },
                "ready_for_exact_eval_dispatch": True,
                "scorer_basin_parity_status": "pass",
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "archive_path": archive.as_posix(),
        "archive_size_bytes": archive.stat().st_size,
        "bits": 6,
        "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "delta_bytes": -15789,
        "rate_component_score_delta_vs_pr106": -0.010513247010845963,
        "readiness_evidence_json": evidence.as_posix(),
        "ready_for_exact_eval_dispatch": False,
        "score_affecting_payload_changed": True,
        "score_claim": False,
        "source_archive_sha256": "3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58",
    }
    manifest = tmp_path / "repack_metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["readiness_evidence_semantics"] == "scorer_basin_parity_gate"
    assert row["scorer_basin_parity_status"] == "pass"
    assert row["proxy_row"] is False
    assert row["pareto_scope"] == "apogee_intN_calibration_component_penalty"
    assert row["expected_total_score_delta"] > 0.09
    assert row["calibration_expected_total_score_delta"] == row["expected_total_score_delta"]
    assert row["readiness_component_penalty_overwhelms_rate_gain"] is True
    assert row["score_lowering_evidence"] is False
    assert row["expected_total_score_delta_source"] == (
        "rate_component_score_delta_vs_pr106_plus_readiness_component_penalty"
    )
    assert "readiness_component_penalty_overwhelms_rate_gain" in row["candidate_blockers"]


def test_field_meta_selector_summarizes_nested_static_compliance_refresh() -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/"
        "hnerv_lowlevel_exact_eval_packet.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    static_refresh = payload["refreshes"]["static_compliance"]

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest], dirty_paths=[])

    row = report["rows"][0]
    summary = row["operator_next_steps_summary"]
    assert summary["static_refresh_status"] == "passed"
    assert summary["static_refresh_command"] == static_refresh["command"]
    assert summary["static_refresh_schema"] == "hnerv_lowlevel_static_compliance_refresh_v1"
    assert summary["static_refresh_source"] == "refreshes.static_compliance"
    assert row["operator_refresh_blockers"] == []
    assert row["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_requires_lane_and_job_identity_for_static_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="identity_missing_candidate",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_local_preflight_ready"] is True
    assert report["candidate_local_preflight_ready_count"] == 1
    assert report["candidate_static_preflight_ready"] is False
    assert report["candidate_static_preflight_ready_count"] == 0
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["ready_candidate_count"] == 0
    assert row["strict_candidate_preflight_ready"] is True
    assert row["candidate_local_preflight_ready"] is True
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["dispatch_identity_proof"]["status"] == "blocked"
    assert row["dispatch_identity_proof"]["blockers"] == [
        "dispatch_lane_id_missing",
        "dispatch_instance_job_id_missing",
    ]
    assert "missing_dispatch_identity_for_lane_claim" in row["candidate_blockers"]
    assert row["candidate_preflight_next_required_proof"] == [
        "manifest_lane_id_and_instance_job_id_for_level2_claim",
    ]
    assert row["next_required_proof"] == [
        "manifest_lane_id_and_instance_job_id_for_level2_claim",
        "passed_kkt_proof_or_converged_admm_waterline_result",
        "pareto_eligible_static_ready_non_proxy_candidate",
    ]


def test_field_meta_selector_honors_packet_static_blockers(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="packet_static_blocked",
        lane_id="lane_packet_static_blocked",
        job_name="job_packet_static_blocked",
        static_packet_ready=False,
        static_blockers=[
            "pre_submission_compliance_failed",
            "dry_run_queue_payload_section_diff_mismatch",
        ],
    )

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        now_utc="2026-05-06T10:00:00Z",
    )

    row = report["selected_candidate"]
    assert row["candidate_id"] == "packet_static_blocked"
    assert row["candidate_local_preflight_ready"] is False
    assert row["candidate_static_preflight_ready"] is False
    assert row["pareto_eligible"] is False
    assert row["selection_decision"] == "strict_candidate_preflight_refused"
    assert "packet_static_preflight_not_ready" in row["static_candidate_blockers"]
    assert "packet_static:pre_submission_compliance_failed" in row["static_candidate_blockers"]
    assert (
        "packet_static:dry_run_queue_payload_section_diff_mismatch"
        in row["exact_dispatch_blockers"]["blockers"]
    )


def test_field_meta_selector_requires_matching_active_claim_for_dispatch_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="claimed_candidate",
        lane_id="lane_claimed_candidate",
        job_name="job_claimed_candidate",
        kkt_proof=_kkt_proof(),
    )
    claims = _claims_file(tmp_path, lane_id="lane_claimed_candidate", job_name="job_claimed_candidate")

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        claims_path=claims,
        now_utc="2026-05-06T12:00:00Z",
    )

    row = report["rows"][0]
    assert report["candidate_static_preflight_ready_count"] == 1
    assert report["ready_candidate_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is True
    assert report["field_selection_ready_for_exact_eval_dispatch"] is True
    assert report["field_selection_ready_for_exact_eval_dispatch_count"] == 1
    assert row["candidate_static_preflight_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["field_selection_ready_for_exact_eval_dispatch"] is True
    assert row["exact_dispatch_blockers"]["blocker_count"] == 0
    assert row["dispatch_claim_proof"]["checked"] is True
    assert row["dispatch_claim_proof"]["active_lane_claim"] is True
    assert row["dispatch_claim_proof"]["active_claim"]["instance_job_id"] == "job_claimed_candidate"
    assert row["next_required_proof"] == [
        "exact_cuda_auth_eval_on_selected_archive_bytes",
    ]


def test_field_meta_selector_keeps_active_claim_blocked_by_missing_kkt_for_field_selection(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="claimed_no_kkt",
        lane_id="lane_claimed_no_kkt",
        job_name="job_claimed_no_kkt",
    )
    claims = _claims_file(tmp_path, lane_id="lane_claimed_no_kkt", job_name="job_claimed_no_kkt")

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[manifest],
        claims_path=claims,
        now_utc="2026-05-06T12:00:00Z",
    )

    row = report["rows"][0]
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["field_selection_ready_for_exact_eval_dispatch"] is False
    assert report["field_selection_ready_for_exact_eval_dispatch_count"] == 0
    assert "kkt:kkt_proof_or_admm_result_missing" in row["exact_dispatch_blockers"]["blockers"]
    assert (
        "passed_kkt_proof_or_converged_admm_waterline_result"
        in row["exact_dispatch_blockers"]["next_required_proof"]
    )


def test_field_meta_selector_rejects_non_zip_archive_even_with_matching_sha(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="not_a_zip",
        lane_id="lane_not_a_zip",
        job_name="job_not_a_zip",
        valid_zip=False,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["candidate_static_preflight_ready"] is False
    assert row["archive_proof"]["byte_closed"] is False
    assert row["archive_proof"]["zip_custody"]["status"] == "blocked"
    assert "archive:zip:archive_zip_unreadable" in row["candidate_blockers"]
    assert row["candidate_preflight_next_required_proof"] == [
        "local_archive_file_with_matching_sha256_and_bytes"
    ]
    assert row["next_required_proof"] == [
        "local_archive_file_with_matching_sha256_and_bytes",
        "passed_kkt_proof_or_converged_admm_waterline_result",
        "pareto_eligible_static_ready_non_proxy_candidate",
    ]


def test_field_meta_selector_rejects_packet_without_runtime_tree_even_when_strict_preflight_passes(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="missing_runtime_tree",
        runtime_tree_sha256=None,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["strict_candidate_preflight_ready"] is True
    assert row["candidate_static_preflight_ready"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "missing_byte_closed_runtime_proof" in row["candidate_blockers"]
    assert row["candidate_preflight_next_required_proof"] == [
        "runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract"
    ]
    assert row["next_required_proof"] == [
        "runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract",
        "passed_kkt_proof_or_converged_admm_waterline_result",
        "pareto_eligible_static_ready_non_proxy_candidate",
    ]


def test_field_meta_selector_never_overrides_strict_candidate_preflight(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="strict_blocked_candidate",
        dispatch_gate="planning_only/no_remote_dispatch",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert row["strict_candidate_preflight_ready"] is False
    assert row["candidate_static_preflight_ready"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "strict_candidate_preflight_not_ready" in row["candidate_blockers"]
    assert any(
        blocker.startswith("strict:dispatch_gate_blocked")
        for blocker in row["candidate_blockers"]
    )


def test_field_meta_selector_surfaces_packet_static_blockers(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="wr01_static_blocked",
        lane_id="lane_wr01_static_blocked",
        job_name="job_wr01_static_blocked",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "dispatch_gate": "blocked_static_packet_ready_until_static_blockers_clear",
            "dispatch_unlocked": False,
            "ready_for_exact_eval_dispatch_claim": False,
            "static_packet_ready": False,
            "static_blockers": [
                "pre_submission_compliance_failed",
                "dry_run_queue_payload_section_diff_mismatch",
            ],
        }
    )
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["candidate_local_preflight_ready"] is False
    assert report["candidate_static_preflight_ready"] is False
    assert row["strict_candidate_preflight_ready"] is False
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is True
    assert "strict_candidate_preflight_not_ready" in row["candidate_blockers"]
    assert "strict:static_packet_ready_false" in row["candidate_blockers"]
    assert "strict:static_blockers_reported" in row["candidate_blockers"]
    assert row["selection_decision"] == "strict_candidate_preflight_refused"
    assert row["non_dominated_frontier_reason"]["status"] == "ineligible"
    assert row["pareto_eligibility_blockers"] == ["candidate_static_preflight_not_ready"]
    assert row["candidate_preflight_next_required_proof"] == [
        "passing_experiments/preflight_candidate_manifest_dispatch_readiness.py"
    ]
    assert row["next_required_proof"] == [
        "passing_experiments/preflight_candidate_manifest_dispatch_readiness.py",
        "passed_kkt_proof_or_converged_admm_waterline_result",
        "pareto_eligible_static_ready_non_proxy_candidate",
    ]


def test_field_meta_selector_is_deterministic_and_orders_static_ready_packets_first(
    tmp_path: Path,
) -> None:
    blocked = _packet_manifest(
        tmp_path / "blocked",
        candidate_id="blocked",
        runtime_tree_sha256=None,
    )
    ready = _packet_manifest(
        tmp_path / "ready",
        candidate_id="ready",
        expected_score_delta=-0.0002,
        lane_id="lane_ready",
        job_name="job_ready",
    )

    first = build_selection_report(repo_root=REPO, manifest_paths=[blocked, ready])
    second = build_selection_report(repo_root=REPO, manifest_paths=[ready, blocked])

    assert first == second
    assert first["selected_candidate"]["candidate_id"] == "ready"
    assert [row["candidate_id"] for row in first["rows"]] == ["ready", "blocked"]
    assert first["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert first["selected_candidate"]["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_exposes_pareto_kkt_and_information_gain(
    tmp_path: Path,
) -> None:
    common = {
        "lane_id": "lane_pareto",
        "job_name": "job_pareto",
        "family_group": "hnerv_rate_recode",
        "pareto_scope": "hnerv_rate_recode",
        "interaction_assumptions": ["first_order_volterra_recode"],
        "interaction_model": "first_order_volterra_rate_recode",
        "volterra_order": 1,
        "volterra_terms": ["rate_only_linear_section"],
        "kkt_proof": _kkt_proof(),
    }
    dominator = _packet_manifest(
        tmp_path / "dominator",
        candidate_id="dominator",
        expected_score_delta=-0.0003,
        byte_delta=-30,
        expected_seg_dist_delta=-0.0002,
        expected_pose_dist_delta=-0.00002,
        expected_information_gain_nats=0.2,
        **common,
    )
    dominated = _packet_manifest(
        tmp_path / "dominated",
        candidate_id="dominated",
        expected_score_delta=-0.0001,
        byte_delta=-10,
        expected_seg_dist_delta=-0.0001,
        expected_pose_dist_delta=-0.00001,
        expected_information_gain_nats=0.9,
        **common,
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[dominated, dominator])

    assert report["pareto_summary"]["frontier_count"] == 1
    assert report["pareto_summary"]["dominated_count"] == 1
    assert report["kkt_ready_for_field_planning_count"] == 2
    assert report["selected_candidate"]["candidate_id"] == "dominator"
    first = report["rows"][0]
    assert first["pareto_frontier"] is True
    assert first["field_interaction_contract"]["status"] == "passed"
    assert first["field_interaction_contract"]["assumptions"] == ["first_order_volterra_recode"]
    assert first["field_interaction_contract"]["interaction_model"] == "first_order_volterra_rate_recode"
    assert first["field_interaction_contract"]["volterra_order"] == 1
    assert first["field_interaction_contract"]["volterra_terms"] == ["rate_only_linear_section"]
    assert first["non_dominated_frontier_reason"]["schema"] == "non_dominated_frontier_reason_v1"
    assert first["non_dominated_frontier_reason"]["status"] == "non_dominated"
    assert first["non_dominated_frontier_reason"]["reason"] == "non_dominated_within_pareto_scope"
    assert first["kkt_ready_for_field_planning"] is True
    assert first["lexicographic_feasibility_tuple"][6] is True
    assert first["field_selection_score_delta"] == first["expected_total_score_delta"]
    assert first["expected_information_gain_nats"] == 0.2
    assert first["exact_dispatch_blockers"]["schema"] == "exact_dispatch_blockers_v1"
    assert "missing_active_lane_dispatch_claim" in first["exact_dispatch_blockers"]["blockers"]
    second = report["rows"][1]
    assert second["candidate_id"] == "dominated"
    assert second["pareto_frontier"] is False
    assert second["pareto_dominated_by"] == ["dominator"]
    assert second["non_dominated_frontier_reason"]["status"] == "dominated"
    assert second["non_dominated_frontier_reason"]["dominated_by"] == ["dominator"]
    assert second["selection_penalty_terms"]["pareto_dominated_packet"] > 0.0
    assert "pareto_dominated_within_scope" in second["exact_dispatch_blockers"]["blockers"]


def test_field_meta_selector_requires_real_kkt_proof_before_kkt_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="no_kkt_proof",
        lane_id="lane_no_kkt_proof",
        job_name="job_no_kkt_proof",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["kkt_ready_for_field_planning_count"] == 0
    assert row["candidate_static_preflight_ready"] is True
    assert row["kkt_ready_for_field_planning"] is False
    assert row["kkt_proof"]["status"] == "blocked"
    assert "kkt:kkt_proof_or_admm_result_missing" in row["kkt_blockers"]
    assert row["field_interaction_contract"]["status"] == "passed"
    assert "kkt:kkt_proof_or_admm_result_missing" in row["exact_dispatch_blockers"]["blockers"]
    assert row["selection_penalty_terms"]["kkt_not_ready_for_field_planning"] > 0.0
    assert row["field_selection_score_delta"] == row["expected_total_score_delta"]


def test_field_meta_selector_accepts_converged_admm_result_as_kkt_proof(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="admm_kkt_ready",
        lane_id="lane_admm_kkt_ready",
        job_name="job_admm_kkt_ready",
        admm_result={
            "converged": True,
            "waterline_kkt_residual": 0.002,
            "kkt_waterline_satisfied": True,
        },
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["kkt_ready_for_field_planning_count"] == 1
    assert row["kkt_ready_for_field_planning"] is True
    assert row["kkt_proof"]["kind"] == "admm_result"
    assert row["lexicographic_feasibility_tuple"][6] is True


def test_field_meta_selector_penalizes_dirty_packet_over_raw_delta(
    tmp_path: Path,
) -> None:
    dirty = _packet_manifest(
        tmp_path / "dirty",
        candidate_id="dirty_raw_best",
        lane_id="lane_dirty_raw_best",
        job_name="job_dirty_raw_best",
        expected_score_delta=-10.0,
        code_paths=["src/tac/optimization/dirty_packet_owner.py"],
        interaction_assumptions=["dirty_candidate_pending_other_worker"],
    )
    clean = _packet_manifest(
        tmp_path / "clean",
        candidate_id="clean_weaker",
        lane_id="lane_clean_weaker",
        job_name="job_clean_weaker",
        expected_score_delta=-0.0001,
        interaction_assumptions=["clean_candidate_first_order"],
    )

    report = build_selection_report(
        repo_root=REPO,
        manifest_paths=[dirty, clean],
        dirty_paths=["src/tac/optimization/dirty_packet_owner.py"],
    )

    assert report["dirty_blocked_candidate_count"] == 1
    assert report["selected_candidate"]["candidate_id"] == "clean_weaker"
    dirty_row = next(row for row in report["rows"] if row["candidate_id"] == "dirty_raw_best")
    assert dirty_row["dirty_blocked"] is True
    assert dirty_row["candidate_static_preflight_ready"] is False
    assert dirty_row["selection_decision"] == "refused_dirty_worktree_overlap"
    assert dirty_row["selection_penalty_terms"]["dirty_path_blocked"] > 0.0
    assert dirty_row["field_selection_score_delta"] == dirty_row["expected_total_score_delta"]
    assert dirty_row["next_required_proof"][0] == "clean_or_isolate_dirty_candidate_paths_before_selection"


def test_field_meta_selector_ranks_q10_over_forensic_cross_paradigm_packets(
    tmp_path: Path,
) -> None:
    q10 = _packet_manifest(
        tmp_path / "q10",
        candidate_id="pr106_q10_151byte_brotli",
        family_group="hnerv_lowlevel_brotli_repack",
        pareto_scope="hnerv_rate_only_exact_archive",
        lane_id="pr106_q10_151byte_brotli",
        job_name="exact_eval_pr106_q10_151byte_brotli_20260507",
        byte_delta=-151,
        expected_score_delta=0.0,
        interaction_assumptions=["rate_only_raw_equivalent_brotli_repack"],
    )
    q10_payload = json.loads(q10.read_text(encoding="utf-8"))
    q10_payload.pop("expected_total_score_delta")
    q10_payload["family"] = "hnerv_lowlevel_brotli_repack"
    q10_payload["expected_total_score_delta_rate_only"] = -0.00010054470192144788
    q10.write_text(json.dumps(q10_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    apogee = _packet_manifest(
        tmp_path / "apogee",
        candidate_id="apogee_int5_forensic",
        family_group="apogee_intN",
        pareto_scope="apogee_intN",
        lane_id="apogee_int5_forensic",
        job_name="forensic_apogee_int5",
        expected_score_delta=-0.02,
        byte_delta=-31_000,
        interaction_assumptions=["forensic_byte_only_no_distortion_model"],
    )
    wave_omega = _packet_manifest(
        tmp_path / "wave_omega",
        candidate_id="wave_omega_proxy",
        family_group="wave_omega",
        pareto_scope="wave_omega",
        lane_id="wave_omega_proxy",
        job_name="forensic_wave_omega",
        expected_score_delta=-0.01,
        byte_delta=-20_000,
        interaction_assumptions=["planning_proxy_wave_omega"],
    )
    dezeta = _packet_manifest(
        tmp_path / "dezeta",
        candidate_id="paradigm_dezeta_proxy",
        family_group="paradigm_dezeta",
        pareto_scope="paradigm_dezeta",
        lane_id="paradigm_dezeta_proxy",
        job_name="forensic_paradigm_dezeta",
        expected_score_delta=-0.03,
        byte_delta=-45_000,
        runtime_tree_sha256=None,
        interaction_assumptions=["planning_proxy_dezeta"],
    )
    for manifest in (apogee, wave_omega, dezeta):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["evidence_grade"] = "planning_proxy"
        payload["proxy_row"] = True
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[apogee, wave_omega, q10, dezeta])

    assert report["selected_candidate"]["candidate_id"] == "pr106_q10_151byte_brotli"
    assert report["adversarial_gate_policy"]["pareto_dominated_packets_sort_behind_non_dominated_static_ready_packets"] is True
    q10_row = report["rows"][0]
    assert q10_row["rate_only_delta_proof"]["status"] == "passed"
    assert q10_row["pareto_frontier"] is True
    assert q10_row["selection_decision"] == "static_candidate_acquire_kkt_and_lane_claim_before_dispatch"
    blocked = {row["candidate_id"]: row for row in report["rows"][1:]}
    assert blocked["apogee_int5_forensic"]["proxy_row"] is True
    assert blocked["wave_omega_proxy"]["proxy_row"] is True
    assert blocked["paradigm_dezeta_proxy"]["runtime_proof"]["runtime_closed"] is False
    assert all(row["pareto_eligible"] is False for row in blocked.values())


def test_field_meta_pareto_does_not_false_dominate_across_scopes(tmp_path: Path) -> None:
    a = _packet_manifest(
        tmp_path / "a",
        candidate_id="scope_a_candidate",
        family_group="scope_a",
        pareto_scope="scope_a",
        lane_id="scope_a_candidate",
        job_name="job_scope_a_candidate",
        expected_score_delta=-0.01,
        byte_delta=-100,
        kkt_proof=_kkt_proof(),
    )
    b = _packet_manifest(
        tmp_path / "b",
        candidate_id="scope_b_candidate",
        family_group="scope_b",
        pareto_scope="scope_b",
        lane_id="scope_b_candidate",
        job_name="job_scope_b_candidate",
        expected_score_delta=-0.001,
        byte_delta=-1,
        kkt_proof=_kkt_proof(),
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[a, b])

    rows = {row["candidate_id"]: row for row in report["rows"]}
    assert rows["scope_a_candidate"]["pareto_frontier"] is True
    assert rows["scope_b_candidate"]["pareto_frontier"] is True
    assert rows["scope_a_candidate"]["pareto_dominated_by"] == []
    assert rows["scope_b_candidate"]["pareto_dominated_by"] == []
    assert report["pareto_summary"]["frontier_count"] == 2


def test_field_meta_selector_blocks_rate_only_delta_that_mismatches_byte_term(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="bad_rate_only_delta",
        lane_id="lane_bad_rate_only_delta",
        job_name="job_bad_rate_only_delta",
        family_group="hnerv_lowlevel_brotli_repack",
        pareto_scope="hnerv_lowlevel_brotli_repack",
        byte_delta=-30,
        expected_score_delta=-0.5,
        interaction_assumptions=["rate_only_raw_equivalent_brotli_repack"],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("expected_total_score_delta")
    payload["expected_total_score_delta_rate_only"] = -0.5
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["rate_only_delta_proof"]["status"] == "blocked"
    assert "rate_only_score_delta_mismatch" in row["rate_only_delta_proof"]["blockers"]
    assert "rate_only_score_delta_mismatch" in row["candidate_blockers"]
    assert row["candidate_static_preflight_ready"] is False
    assert row["pareto_eligible"] is False
    assert "rate_only_score_delta_reconciles_to_official_byte_rate_term" in row["next_required_proof"]


def test_field_meta_selector_exposes_closed_ingestion_contract_without_score_evidence(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="closed_not_score_evidence",
        lane_id="lane_closed_not_score_evidence",
        job_name="job_closed_not_score_evidence",
        kkt_proof=_kkt_proof(),
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["field_meta_ingestion_ready_count"] == 1
    assert report["score_evidence_rankable_count"] == 0
    assert row["field_meta_ingestion_contract"]["schema"] == "field_meta_ingestion_contract_v1"
    assert row["field_meta_ingestion_contract"]["local_field_meta_ingestion_ready"] is True
    assert row["field_meta_ingestion_contract"]["dispatch_ingestion_ready"] is True
    assert row["score_evidence_contract"]["score_evidence_rankable"] is False
    assert row["score_evidence_contract"]["planning_priority_rankable"] is True
    assert "missing_exact_cuda_positive_score_evidence" in row["score_evidence_contract"]["blockers"]
    assert row["score_lowering_evidence"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_emits_comparable_frontier_rows(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="categorical_frontier_fixture",
        family_group="categorical_qma9_clade_spade_openpilot",
        pareto_scope="categorical_mask_runtime",
        lane_id="lane_categorical_frontier_fixture",
        job_name="job_categorical_frontier_fixture",
        kkt_proof=_kkt_proof(),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["paradigms"] = ["categorical_masks", "openpilot_priors", "meta_lagrangian"]
    payload["role"] = "replacement_or_mask_stacker"
    payload["action_class"] = "build_byte_closed_categorical_candidate"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    frontier_row = row["frontier_row"]
    assert report["frontier_row_schema"] == FRONTIER_ROW_SCHEMA
    assert report["frontier_row_fields"] == list(FRONTIER_ROW_FIELDS)
    assert report["frontier_row_count"] == report["candidate_count"] == 1
    assert report["frontier_rows"] == [frontier_row]
    assert list(frontier_row) == list(FRONTIER_ROW_FIELDS)
    assert frontier_row["schema"] == FRONTIER_ROW_SCHEMA
    assert frontier_row["source_tool"] == "tools/build_field_meta_dispatch_selection.py"
    assert frontier_row["source_path"] == row["manifest_path"]
    assert frontier_row["candidate_id"] == row["candidate_id"]
    assert frontier_row["family_group"] == "categorical_qma9_clade_spade_openpilot"
    assert frontier_row["pareto_scope"] == "categorical_mask_runtime"
    assert frontier_row["paradigms"] == [
        "categorical_masks",
        "openpilot_priors",
        "meta_lagrangian",
    ]
    assert frontier_row["role"] == "replacement_or_mask_stacker"
    assert frontier_row["action_class"] == "build_byte_closed_categorical_candidate"
    assert frontier_row["candidate_static_preflight_ready"] is True
    assert frontier_row["pareto_eligible"] is True
    assert frontier_row["pareto_frontier"] is True
    assert frontier_row["ready_for_exact_eval_dispatch"] is False
    assert frontier_row["score_claim"] is False
    assert frontier_row["dispatch_attempted"] is False
    assert frontier_row["expected_total_score_delta"] == row["expected_total_score_delta"]
    assert "missing_active_lane_dispatch_claim" in frontier_row["blockers"]
    assert (
        "matching_active_level2_lane_claim_for_manifest_lane_and_job"
        in frontier_row["next_required_proof"]
    )


def test_field_meta_selector_refuses_planning_packet_as_score_evidence_even_if_source_claims_it(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="planning_claimed_score_evidence",
        lane_id="lane_planning_claimed_score_evidence",
        job_name="job_planning_claimed_score_evidence",
        kkt_proof=_kkt_proof(),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["planning_only"] = True
    payload["proxy_row"] = True
    payload["evidence_grade"] = "planning_proxy"
    payload["score_lowering_evidence"] = True
    payload["exact_positive_cuda_evidence"] = True
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["source_score_lowering_evidence"] is True
    assert row["score_lowering_evidence"] is False
    assert row["score_evidence_rankable"] is False
    assert row["planning_priority_rankable"] is False
    assert row["field_meta_ingestion_contract"]["local_field_meta_ingestion_ready"] is True
    assert row["field_meta_ingestion_contract"]["dispatch_ingestion_ready"] is False
    assert "planning_only_packet_not_dispatch_ready" in row["field_meta_ingestion_contract"]["dispatch_blockers"]
    assert "planning_or_proxy_packet_not_score_evidence" in row["score_evidence_contract"]["blockers"]
    assert "proxy_or_planning_evidence_grade_not_score_evidence" in row["score_evidence_contract"]["blockers"]
    assert row["pareto_eligible"] is False
    assert "planning_or_proxy_packet" in row["pareto_eligibility_blockers"]
    assert row["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_normalizes_hdm3_and_pr101_rate_recode_manifests(
    tmp_path: Path,
) -> None:
    hdm3_archive = _zip_fixture(tmp_path / "hdm3.zip", "x", b"hdm3")
    pr101_archive = _zip_fixture(tmp_path / "pr101.zip", "x", b"pr101")
    hdm3_manifest = tmp_path / "hdm3_manifest.json"
    hdm3_manifest.write_text(
        json.dumps(
            {
                "tool": "tac.hnerv_hdm3_archive_candidate.build_hdm3_archive_candidate",
                "candidate_archive_path": hdm3_archive.as_posix(),
                "candidate_archive_sha256": hashlib.sha256(hdm3_archive.read_bytes()).hexdigest(),
                "candidate_archive_bytes": hdm3_archive.stat().st_size,
                "source_archive_bytes": hdm3_archive.stat().st_size + 14,
                "candidate_rate_score_delta_if_runtime_supported_and_components_equal": -9.322025e-6,
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    proof = tmp_path / "runtime_adapter_proof.with_tool_run.json"
    proof.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": hashlib.sha256(hdm3_archive.read_bytes()).hexdigest(),
                "ready_for_public_runtime_inflate": True,
                "inflate_output_parity_proven_by_payload_identity": True,
                "remaining_dispatch_blockers": [
                    "strict_pre_submission_compliance_json_missing",
                    "lane_dispatch_claim_missing",
                    "exact_cuda_auth_eval_missing",
                ],
                "score_claim": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pr101_manifest = tmp_path / "pr101_manifest.json"
    pr101_manifest.write_text(
        json.dumps(
            {
                "tool": "tac.hnerv_pr101_schema_packer.build_pr101_schema_archive_candidate",
                "candidate_archive_path": pr101_archive.as_posix(),
                "candidate_archive_sha256": hashlib.sha256(pr101_archive.read_bytes()).hexdigest(),
                "candidate_archive_bytes": pr101_archive.stat().st_size,
                "source_archive_bytes": pr101_archive.stat().st_size + 36,
                "candidate_rate_score_delta_if_runtime_supported_and_components_equal": -2.3970922e-5,
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[hdm3_manifest, pr101_manifest])

    rows = {row["candidate_id"]: row for row in report["rows"]}
    hdm3 = rows["pr106x_hdm3_decoder_recode_14byte"]
    assert hdm3["family_group"] == "hnerv_decoder_entropy_recode"
    assert hdm3["pareto_scope"] == "hnerv_rate_only_exact_archive"
    assert hdm3["byte_delta"] == -14
    assert hdm3["expected_total_score_delta"] == -9.322025e-6
    assert hdm3["rate_only_delta_proof"]["status"] == "passed"
    assert hdm3["archive_proof"]["byte_closed"] is True
    assert hdm3["runtime_proof"]["runtime_closed"] is False
    assert hdm3["readiness_evidence_semantics"] == "hdm3_runtime_adapter_payload_identity"
    assert "hdm3_runtime_adapter_archive_parity_proof_missing" not in hdm3["candidate_blockers"]
    assert hdm3["frontier_row"]["family"] == "hnerv_hdm3_decoder_entropy_recode"
    pr101 = rows["pr106x_pr101_schema_f32_recode_36byte"]
    assert pr101["family"] == "hnerv_pr101_schema_decoder_recode"
    assert pr101["byte_delta"] == -36
    assert pr101["rate_only_delta_proof"]["status"] == "passed"
    assert "fp16_scale_probe_is_scorer_changing_and_not_rate_only" in pr101["interaction_assumptions"]
    assert pr101["archive_proof"]["byte_closed"] is True
    assert pr101["ready_for_exact_eval_dispatch"] is False


def test_field_meta_selector_normalizes_categorical_and_entropy_planning_manifests(
    tmp_path: Path,
) -> None:
    categorical_archive = _zip_fixture(tmp_path / "categorical.zip", "categorical_payload.bin", b"labels")
    categorical_manifest = tmp_path / "categorical_summary.json"
    categorical_manifest.write_text(
        json.dumps(
            {
                "kind": "categorical_byte_closed_local_candidate_build",
                "archive_bytes": categorical_archive.stat().st_size,
                "archive_sha256": hashlib.sha256(categorical_archive.read_bytes()).hexdigest(),
                "paths": {"archive": categorical_archive.as_posix()},
                "payload_source": {"source_archive_bytes": categorical_archive.stat().st_size + 52_679},
                "readiness_blockers": ["decode_reencode_full_decode_not_proven"],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    entropy_manifest = tmp_path / "entropy.json"
    entropy_manifest.write_text(
        json.dumps(
            {
                "tool": "tac.hnerv_hdc2_combined_entropy",
                "planning_only": True,
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
                "byte_accounting": {
                    "net_byte_delta_after_combined_targets": -13_565,
                    "projected_rate_score_delta_after_combined_targets": -0.009032376699102255,
                },
                "target": {
                    "frontier_archive_bytes": 186_080,
                    "frontier_archive_sha256": "a" * 64,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[categorical_manifest, entropy_manifest])

    rows = {row["candidate_id"]: row for row in report["rows"]}
    categorical = rows["categorical_openpilot_hpm1_payload_candidate"]
    assert categorical["family_group"] == "categorical_selfcompression_mask_payload"
    assert categorical["byte_delta"] == -52_679
    assert categorical["archive_proof"]["byte_closed"] is True
    assert categorical["proxy_row"] is False
    assert categorical["planning_priority_rankable"] is False
    assert "decode_reencode_identity_required_before_dispatch" in categorical["interaction_assumptions"]
    entropy = rows["hnerv_hdc2_hdm3_combined_entropy_target"]
    assert entropy["proxy_row"] is True
    assert entropy["planning_priority_rankable"] is False
    assert entropy["byte_delta"] == -13_565
    assert entropy["expected_total_score_delta"] == -0.009032376699102255
    assert "planning_or_proxy_packet" in entropy["pareto_eligibility_blockers"]


def test_field_meta_selector_normalizes_pr102_zero_byte_runtime_tuning_custody(
    tmp_path: Path,
) -> None:
    archive = _zip_fixture(tmp_path / "pr102.zip", "0.bin", b"runtime-tuning")
    manifest = tmp_path / "pr102_custody.json"
    manifest.write_text(
        json.dumps(
            {
                "tool": "tools/audit_pr102_zero_byte_tuning_custody.py",
                "correct_pr102_archive": {
                    "path": archive.as_posix(),
                    "bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
                "zero_byte_runtime_contract": {
                    "archive_byte_delta": 0,
                    "archive_payload_unchanged_from_pr100": True,
                },
                "dispatch_blockers": ["pr102_exact_cuda_replay_missing"],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert row["candidate_id"] == "pr102_zero_byte_runtime_tuning"
    assert row["family_group"] == "hnerv_runtime_tuning"
    assert row["byte_delta"] == 0
    assert row["expected_total_score_delta"] == 0.0
    assert row["archive_proof"]["byte_closed"] is True
    assert row["runtime_proof"]["runtime_closed"] is False
    assert row["proxy_row"] is False
    assert row["pareto_eligible"] is False
    assert "zero_byte_runtime_tuning_no_archive_delta" in row["interaction_assumptions"]
    assert "pr102_exact_cuda_replay_missing" in row["candidate_blockers"]


def test_field_meta_selector_does_not_ingest_unclosed_runtime_as_dispatch_ready(
    tmp_path: Path,
) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="unclosed_runtime_packet",
        lane_id="lane_unclosed_runtime_packet",
        job_name="job_unclosed_runtime_packet",
        runtime_tree_sha256=None,
        kkt_proof=_kkt_proof(),
    )

    report = build_selection_report(repo_root=REPO, manifest_paths=[manifest])

    row = report["rows"][0]
    assert report["field_meta_ingestion_ready_count"] == 0
    assert row["field_meta_ingestion_contract"]["local_field_meta_ingestion_ready"] is False
    assert row["field_meta_ingestion_contract"]["dispatch_ingestion_ready"] is False
    assert "runtime_tree_closure_proof_missing" in row["field_meta_ingestion_contract"]["local_blockers"]
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["field_selection_ready_for_exact_eval_dispatch"] is False
    assert row["pareto_eligible"] is False
    assert "field_meta_ingestion_contract_not_ready" in row["pareto_eligibility_blockers"]


def test_build_field_meta_dispatch_selection_cli_writes_json(tmp_path: Path) -> None:
    manifest = _packet_manifest(
        tmp_path,
        candidate_id="cli_candidate",
        lane_id="lane_cli_candidate",
        job_name="job_cli_candidate",
    )
    out = tmp_path / "selection.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_field_meta_dispatch_selection.py"),
            "--manifest",
            str(manifest),
            "--json-out",
            str(out),
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    assert payload["selected_candidate"]["candidate_id"] == "cli_candidate"
    assert payload["candidate_static_preflight_ready_count"] == 1
    assert payload["ready_candidate_count"] == 0
    assert payload["selected_candidate"]["candidate_static_preflight_ready"] is True
    assert payload["selected_candidate"]["ready_for_exact_eval_dispatch"] is False


def _zip_fixture(path: Path, member: str, payload: bytes) -> Path:
    info = zipfile.ZipInfo(member)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)
    return path


def _packet_manifest(
    root: Path,
    *,
    candidate_id: str,
    runtime_tree_sha256: str | None = "b" * 64,
    dispatch_gate: str = "eligible_for_cuda_auth_eval_after_lane_claim",
    expected_score_delta: float = -0.0001,
    byte_delta: int = -9,
    expected_seg_dist_delta: float = 0.0,
    expected_pose_dist_delta: float = 0.0,
    expected_information_gain_nats: float = 0.1,
    family_group: str = "fixture_family",
    pareto_scope: str = "fixture_family",
    interaction_assumptions: list[str] | None = None,
    interaction_model: str | None = None,
    volterra_order: int | None = None,
    volterra_terms: list[str] | None = None,
    code_paths: list[str] | None = None,
    evidence_paths: list[str] | None = None,
    kkt_proof: dict[str, object] | None = None,
    admm_result: dict[str, object] | None = None,
    lane_id: str | None = None,
    job_name: str | None = None,
    static_packet_ready: bool | None = None,
    static_blockers: list[str] | None = None,
    valid_zip: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.zip"
    if valid_zip:
        info = zipfile.ZipInfo("x")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 0o644 << 16
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(info, f"{candidate_id} archive bytes".encode())
    else:
        archive.write_bytes(f"{candidate_id} archive bytes".encode())
    fixed_runtime = {
        "ready_for_fixed_runtime_exact_eval": True,
        "remaining_blockers": [],
    }
    if runtime_tree_sha256 is not None:
        fixed_runtime["runtime_tree_sha256"] = runtime_tree_sha256
    payload = {
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_attempted": False,
        "dispatch_gate": dispatch_gate,
        "dispatch_unlocked": True,
        "ready_for_exact_eval_dispatch_claim": True,
        "family_group": family_group,
        "pareto_scope": pareto_scope,
        "interaction_assumptions": interaction_assumptions or ["fixture_first_order"],
        "archive": {
            "path": archive.as_posix(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bytes": archive.stat().st_size,
        },
        "fixed_runtime_preflight": fixed_runtime,
        "byte_delta": byte_delta,
        "expected_total_score_delta": expected_score_delta,
        "expected_seg_dist_delta": expected_seg_dist_delta,
        "expected_pose_dist_delta": expected_pose_dist_delta,
        "expected_information_gain_nats": expected_information_gain_nats,
    }
    if static_packet_ready is not None:
        payload["static_packet_ready"] = static_packet_ready
    if static_blockers is not None:
        payload["static_blockers"] = static_blockers
    if interaction_model is not None:
        payload["interaction_model"] = interaction_model
    if volterra_order is not None:
        payload["volterra_order"] = volterra_order
    if volterra_terms is not None:
        payload["volterra_terms"] = volterra_terms
    if code_paths is not None:
        payload["code_paths"] = code_paths
    if evidence_paths is not None:
        payload["evidence_paths"] = evidence_paths
    if kkt_proof is not None:
        payload["kkt_proof"] = kkt_proof
    if admm_result is not None:
        payload["admm_result"] = admm_result
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if job_name is not None:
        payload["job_name"] = job_name
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _kkt_proof() -> dict[str, object]:
    return {
        "status": "passed",
        "stationarity_residual": 0.0,
        "stationarity_tolerance": 0.001,
    }


def _claims_file(root: Path, *, lane_id: str, job_name: str) -> Path:
    claims = root / "active_lane_dispatch_claims.md"
    claims.write_text(
        "\n".join(
            [
                "# Active lane dispatch claims - test fixture",
                "",
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                f"| 2026-05-06T11:55:00Z | codex:test | {lane_id} | lightning | {job_name} | 2026-05-06T12:30:00Z | active_exact_eval | test claim |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return claims
