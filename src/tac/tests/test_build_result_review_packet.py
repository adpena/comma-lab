# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_result_review_packet.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_result_review_packet", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _auth_eval_payload() -> dict:
    archive_bytes = 156_404
    seg = 0.00186125
    pose = 0.00037762
    score = 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * archive_bytes / 37_545_489
    return {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_bytes,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": "a" * 64,
            "archive_size_bytes": archive_bytes,
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "inflate_script": "/remote/submission/inflate.sh",
            "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": "b" * 64,
                "runtime_content_tree_sha256": "c" * 64,
                "runtime_file_count": 2,
                "files": [{"relative_path": "inflate.sh"}, {"relative_path": "inflate.py"}],
                "external_dependency_roots": [],
            },
            "inflate_script_sha256": "d" * 64,
            "inflated_output_manifest": {
                "sha256": "e" * 64,
                "payload": {"aggregate_sha256": "f" * 64},
            },
        },
    }


def test_builds_negative_exact_cuda_review_packet(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "contest_auth_eval.json"
    source.write_text(json.dumps(_auth_eval_payload()), encoding="utf-8")
    claims = tmp_path / "active_lane_dispatch_claims.md"
    claims.write_text(
        "\n".join([
            "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
            "|---|---|---|---|---|---|---|---|",
            "| 2026-05-08T03:30:36Z | claude_lab | lossy_coarsening_analytical_cuda | lightning | job |  | completed_score_0.351719 | artifact=contest_auth_eval.json |",
        ]),
        encoding="utf-8",
    )

    packet = tool.build_packet(
        auth_eval_json=source,
        technique="lossy_coarsening_analytical",
        lane_id="lossy_coarsening_analytical_cuda",
        job_id="job",
        baseline_score=0.20898105277982337,
        reactivation_criteria=["requires retrain or scorer-aware loss"],
        reviewer="test",
        dispatch_claims_path=claims,
    )

    assert packet["schema"] == "tac_result_review_packet_v1"
    assert packet["measured_config_status"] == "measured_config_retired"
    assert packet["family_falsified"] is False
    assert packet["rank_or_kill_eligible"] is False
    assert packet["exact_cuda_evidence"] is True
    assert packet["score_axis"] == "contest_cuda"
    assert packet["score_claim_valid"] is True
    assert packet["score_recomputation"]["matches_reported"] is True
    assert packet["runtime_custody"]["payload_closure_fields_present"] is True
    assert packet["runtime_custody"]["runtime_content_tree_sha256"] == "c" * 64
    assert packet["runtime_custody"]["inflate_script_sha256"] == "d" * 64
    assert packet["runtime_custody"]["inflated_output_aggregate_sha256"] == "f" * 64
    assert packet["engineering_forensic_audit"] == {
        "schema": "engineering_forensic_audit_v1",
        "custody_reviewed": True,
        "axis_reviewed": True,
        "runtime_config_reviewed": True,
        "archive_runtime_closure_reviewed": True,
        "score_formula_reviewed": True,
        "dispatch_claim_reviewed": True,
        "engineering_or_config_bug_found": False,
        "audit_blockers": [],
        "classification_after_audit": "measured_config_retired_only",
        "dead_or_family_falsification_allowed": False,
        "measured_config_retirement_allowed": True,
    }
    assert packet["dispatch_claim_state"]["matching_claim_count"] == 1
    assert packet["dispatch_claim_state"]["latest_status"] == "completed_score_0.351719"
    assert packet["dispatch_claim_state"]["terminal_status_recorded"] is True

    row = tool.evidence_row_from_packet(
        packet,
        review_packet_path=Path(".omx/research/review.json"),
        timestamp_utc="2026-05-08T00:00:00Z",
    )
    assert row["evidence_grade"] == "[contest-CUDA A-negative]"
    assert row["score_axis"] == "contest_cuda"
    assert row["score_claim_valid"] is True
    assert row["exact_cuda_evidence"] is True
    assert row["contest_dispatch_verdict"] == "measured_config_retired_exact_cuda_negative"
    assert row["empirical_archive_bytes"] == 156_404
    assert row["score_contest_cuda"] == packet["canonical_score"]
    assert row["rate"] == packet["score_recomputation"]["rate_term"]
    assert row["rate_term"] == packet["score_recomputation"]["rate_term"]
    assert row["runtime_content_tree_sha256"] == "c" * 64
    assert row["inflate_script_sha256"] == "d" * 64
    assert row["inflated_output_aggregate_sha256"] == "f" * 64
    assert row["archive_rate_ratio"] == pytest.approx(
        packet["score_recomputation"]["rate_term"] / 25.0
    )
    assert row["family_falsified"] is False
    assert row["method_family_retired"] is False
    assert row["engineering_forensic_audit"]["engineering_or_config_bug_found"] is False
    assert row["engineering_forensic_audit"]["measured_config_retirement_allowed"] is True
    assert "reactivation_required_before_new_dispatch" in row["dispatch_blockers"]
    assert row["exact_result_review_packet"] == ".omx/research/review.json"


def test_requires_reactivation_criteria_for_negative_exact_cuda(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "contest_auth_eval.json"
    source.write_text(json.dumps(_auth_eval_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="reactivation criteria"):
        tool.build_packet(
            auth_eval_json=source,
            technique="lossy_coarsening_analytical",
            lane_id="lossy_coarsening_analytical_cuda",
            job_id="job",
            baseline_score=0.20898105277982337,
            reactivation_criteria=[],
            reviewer="test",
            dispatch_claims_path=None,
        )


def test_exact_cuda_without_baseline_does_not_claim_not_negative(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "contest_auth_eval.json"
    source.write_text(json.dumps(_auth_eval_payload()), encoding="utf-8")

    packet = tool.build_packet(
        auth_eval_json=source,
        technique="selector_probe",
        lane_id="selector_probe_cuda",
        job_id="job",
        baseline_score=None,
        reactivation_criteria=[],
        reviewer="test",
        dispatch_claims_path=None,
    )

    assert packet["measured_config_status"] == "exact_cuda_result_reviewed"
    assert packet["failure_class"] == "exact_cuda_result_reviewed_baseline_missing"
    assert packet["baseline_score"] is None
    assert packet["score_claim_valid"] is True
    assert (
        packet["engineering_forensic_audit"]["classification_after_audit"]
        == "exact_cuda_result_reviewed_no_negative_status_change"
    )


def test_proxy_or_cpu_packet_stays_non_rankable(tmp_path: Path) -> None:
    tool = _load_tool()
    payload = _auth_eval_payload()
    payload["provenance"]["device"] = "cpu"
    payload["provenance"]["gpu_model"] = ""
    source = tmp_path / "contest_auth_eval.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    packet = tool.build_packet(
        auth_eval_json=source,
        technique="proxy_lane",
        lane_id="proxy_lane",
        job_id="local",
        baseline_score=0.1,
        reactivation_criteria=[],
        reviewer="test",
        dispatch_claims_path=None,
    )

    assert packet["measured_config_status"] == "proxy_or_non_cuda_review_only"
    assert packet["exact_cuda_evidence"] is False
    assert packet["promotion_eligible"] is False
    assert packet["method_family_retired"] is False
    assert (
        packet["engineering_forensic_audit"]["classification_after_audit"]
        == "non_cuda_review_only"
    )


def test_contest_cpu_packet_is_reviewed_as_public_axis_not_cuda_promotion(tmp_path: Path) -> None:
    tool = _load_tool()
    payload = _auth_eval_payload()
    payload["evidence_grade"] = "contest-CPU-1to1"
    payload["lane_tag"] = "[contest-CPU]"
    payload["hardware"] = "github-actions-ubuntu-latest-x86_64"
    payload["runner_os_release"] = "Image: ubuntu-24.04"
    payload["provenance"]["device"] = "cpu"
    payload["provenance"]["gpu_model"] = ""
    source = tmp_path / "contest_cpu_eval.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    packet = tool.build_packet(
        auth_eval_json=source,
        technique="a1_score_gradient",
        lane_id="a1_cpu",
        job_id="gha",
        baseline_score=0.2,
        reactivation_criteria=[],
        reviewer="test",
        dispatch_claims_path=None,
    )

    assert packet["measured_config_status"] == "contest_cpu_result_reviewed"
    assert packet["exact_cuda_evidence"] is False
    assert packet["exact_cpu_evidence"] is True
    assert packet["cpu_leaderboard_reproduction_eligible"] is True
    assert packet["promotion_eligible"] is False
    assert (
        packet["engineering_forensic_audit"]["classification_after_audit"]
        == "contest_cpu_axis_reviewed_cuda_pending"
    )

    row = tool.evidence_row_from_packet(
        packet,
        review_packet_path=Path(".omx/research/cpu_review.json"),
        timestamp_utc="2026-05-09T00:00:00Z",
    )

    assert row["evidence_grade"] == "[contest-CPU reviewed]"
    assert row["score_contest_cpu"] == packet["canonical_score"]
    assert row["score_contest_cuda"] is None
    assert "contest_cuda_pending_for_internal_promotion" in row["dispatch_blockers"]


def test_negative_exact_cuda_with_engineering_blocker_stays_indeterminate(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    payload = _auth_eval_payload()
    payload["provenance"].pop("inflate_runtime_manifest")
    source = tmp_path / "contest_auth_eval.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    packet = tool.build_packet(
        auth_eval_json=source,
        technique="lossy_coarsening_analytical",
        lane_id="lossy_coarsening_analytical_cuda",
        job_id="job",
        baseline_score=0.20898105277982337,
        reactivation_criteria=["requires runtime closure review"],
        reviewer="test",
        dispatch_claims_path=None,
    )

    assert packet["measured_config_status"] == "indeterminate_engineering_or_config_blocker"
    assert packet["failure_class"] == "indeterminate_engineering_or_config_blocker"
    audit = packet["engineering_forensic_audit"]
    assert audit["engineering_or_config_bug_found"] is True
    assert audit["measured_config_retirement_allowed"] is False
    assert "runtime_manifest_missing" in audit["audit_blockers"]
    assert "terminal_dispatch_claim_missing_for_negative_cuda" in audit["audit_blockers"]

    row = tool.evidence_row_from_packet(
        packet,
        review_packet_path=Path(".omx/research/review.json"),
        timestamp_utc="2026-05-09T00:00:00Z",
    )
    assert row["evidence_grade"] == "[contest-CUDA reviewed]"
    assert row["contest_dispatch_verdict"] == "indeterminate_engineering_or_config_blocker"
    assert row["rank_or_kill_eligible"] is False
