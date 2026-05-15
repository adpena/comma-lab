"""Regression tests for advisory-negative evidence handling in cathedral autopilot."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "cathedral_autopilot.py"


def _load_autopilot():
    spec = importlib.util.spec_from_file_location(
        "cathedral_autopilot_under_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_macos_cpu_advisory_negative_enters_validation_queue_not_ranking() -> None:
    autopilot = _load_autopilot()
    evidence = [
        autopilot.TechniqueEvidence(
            technique="cross_paradigm_admm_continuous_k_plus_op1_finalizer",
            empirical_archive_bytes=153_513,
            empirical_score=0.32844434076752543,
            empirical_d_seg=0.00188570,
            empirical_d_pose=0.00014180,
            evidence_grade="[macOS-CPU advisory negative]",
            evidence_semantics=(
                "non_contest_cpu_auth_eval_advisory_measured_config_negative"
            ),
            device_axis="macos_cpu_advisory",
            archive_sha256="7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897",
            runtime_tree_sha256="4a3fdcb6fbe8aed4263b283da89a96ec6f0dff8dba1efdcd3811fda5228ecdea",
            hardware="Apple Silicon macOS CPU advisory; not contest-CPU",
            sample_count=600,
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            cuda_eval_worth_testing=False,
            proxy_row=True,
            contest_dispatch_verdict=(
                "measured_config_retired_macos_cpu_advisory_negative_not_score_evidence"
            ),
            measured_config_status="measured_config_retired_macos_cpu_advisory_negative",
            family_falsified=False,
            method_family_retired=False,
            falsification_scope="measured_configuration_only_macos_cpu_advisory",
            dispatch_blockers=[
                "macos_cpu_advisory_not_contest_cpu",
                "missing_exact_cuda_auth_eval",
                "reactivation_required_before_new_dispatch",
            ],
            reactivation_criteria=[
                "replace rel_err-only allocation with scorer-aware allocation",
            ],
        )
    ]

    plan = autopilot.build_plan(
        d_seg=0.00067082,
        d_pose=3.36e-05,
        archive_bytes=185_578,
        target_score=0.190,
        label="test_current",
        prior_evidence=evidence,
        include_axis_priorities=False,
    )
    payload = asdict(plan)

    report = payload["evidence_semantics_report"]
    assert report["device_axis_report"]["macos_cpu_advisory_count"] == 1
    assert report["contest_cpu_score_claim_row_count"] == 0
    assert report["unknown_exact_negative_row_count"] == 0

    queue_rows = [
        row for row in payload["validation_queue"]
        if row["technique"] == "cross_paradigm_admm_continuous_k_plus_op1_finalizer"
    ]
    assert len(queue_rows) == 1
    row = queue_rows[0]
    assert row["queue_source"] == "unknown_evidence_candidate"
    assert row["validation_status"] == "unknown_not_cuda_worth_testing_until_reactivated"
    assert row["potential_score_delta_if_validated"] == 0.0
    assert row["score_claim"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "reactivation_required_before_new_dispatch" in row["dispatch_blockers"]


def test_autopilot_rank_or_kill_requires_explicit_true() -> None:
    autopilot = _load_autopilot()

    row = autopilot._attach_fail_closed_dispatch_fields(
        {
            "name": "stale_promotable_without_rank_evidence",
            "score_claim": True,
            "score_claim_valid": True,
            "promotion_eligible": True,
            "ready_for_exact_eval_dispatch": True,
        }
    )

    assert row["score_claim"] is True
    assert row["score_claim_valid"] is True
    assert row["promotion_eligible"] is True
    assert row["rank_or_kill_eligible"] is False


def test_contest_cpu_claim_counter_does_not_count_linux_cuda_rows() -> None:
    autopilot = _load_autopilot()
    archive_sha = "a" * 64
    runtime_sha = "b" * 64
    common = {
        "empirical_archive_bytes": 185_578,
        "empirical_score": 0.1966,
        "empirical_d_seg": 0.00067,
        "empirical_d_pose": 3.36e-05,
        "archive_sha256": archive_sha,
        "runtime_tree_sha256": runtime_sha,
        "sample_count": 600,
        "score_claim": True,
        "rank_or_kill_eligible": True,
        "exact_eval_command": "python experiments/contest_auth_eval.py --samples 600",
        "log_path": "reports/exact_eval.log",
    }
    cuda_row = autopilot.TechniqueEvidence(
        technique="linux_cuda_row",
        evidence_grade="[contest-CUDA]",
        device_axis="contest_cuda",
        hardware="Linux x86_64 T4 [contest-CUDA]",
        score_contest_cuda=0.1966,
        **common,
    )
    cpu_row = autopilot.TechniqueEvidence(
        technique="linux_cpu_row",
        evidence_grade="[contest-CPU]",
        device_axis="contest_cpu",
        hardware="GitHub Actions Ubuntu-24.04 Linux x86_64 CPU [contest-CPU]",
        score_contest_cpu=0.1966,
        **common,
    )

    report = autopilot.summarize_evidence_semantics(
        [cuda_row, cpu_row],
        catalogs=[],
    )

    assert report["contest_cpu_score_claim_row_count"] == 1
