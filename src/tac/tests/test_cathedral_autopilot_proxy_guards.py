from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_tool_module():
    path = REPO / "tools" / "cathedral_autopilot.py"
    spec = importlib.util.spec_from_file_location("cathedral_autopilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_autopilot_rejects_spoofed_mps_promotability_booleans() -> None:
    tool = _load_tool_module()
    evidence = tool.TechniqueEvidence(
        technique="arch_shrink_mps",
        empirical_archive_bytes=100_000,
        evidence_grade="[MPS-research-signal]",
        evidence_marker="[MPS-research-signal]",
        evidence_semantics="mps_proxy_curve_shape_only",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        dispatch_blockers=[],
        source="[MPS-research-signal] local proxy",
    )

    assert tool._is_explicitly_promotable_evidence(evidence) is False


def test_autopilot_requires_exact_cuda_for_promotable_evidence() -> None:
    tool = _load_tool_module()
    exact = tool.TechniqueEvidence(
        technique="exact_anchor",
        empirical_archive_bytes=100_000,
        evidence_grade="[contest-CUDA]",
        evidence_semantics="contest_cuda_exact_eval_positive",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        dispatch_blockers=[],
        source="contest_auth_eval.json",
    )
    cpu = tool.TechniqueEvidence(
        technique="cpu_anchor",
        empirical_archive_bytes=100_000,
        evidence_grade="[contest-CPU]",
        evidence_semantics="contest_cpu_exact_eval_positive",
        device_axis="contest_cpu",
        hardware="ubuntu-24.04 linux x86_64 github-actions",
        score_claim=True,
        promotion_eligible=True,
        rank_or_kill_eligible=True,
        ready_for_exact_eval_dispatch=True,
        dispatch_blockers=[],
        source="github-actions contest_auth_eval.json",
    )

    assert tool._is_explicitly_promotable_evidence(exact) is True
    assert tool._is_explicitly_promotable_evidence(cpu) is False
    assert tool._is_explicitly_contest_cpu_evidence(cpu) is True


def test_exact_negative_supersedes_proxy_byte_anchor() -> None:
    tool = _load_tool_module()
    catalog = [{
        "name": "lossy_coarsening_analytical",
        "predicted_archive_bytes": 180_000,
        "cost_dollars": 5.0,
        "cost_hours": 2.0,
    }]
    evidence = [
        tool.TechniqueEvidence(
            technique="lossy_coarsening_analytical",
            empirical_archive_bytes=156_344,
            evidence_grade="[MPS-research-signal]",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            dispatch_blockers=["missing_exact_cuda_auth_eval"],
            source="[MPS-research-signal] byte-only proxy",
        ),
        tool.TechniqueEvidence(
            technique="lossy_coarsening_analytical",
            empirical_archive_bytes=156_404,
            empirical_score=0.351718793322788,
            score_contest_cuda=0.351718793322788,
            evidence_grade="[contest-CUDA A-negative]",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            contest_dispatch_verdict="measured_config_retired_exact_cuda_negative",
            measured_config_status="measured_config_retired",
            family_falsified=False,
            method_family_retired=False,
            falsification_scope="measured_config_only",
            reactivation_criteria=[
                "retrain under scorer-aware loss",
                "prove byte-closed runtime packet",
            ],
            exact_result_review_path=(
                ".omx/research/"
                "lossy_coarsening_exact_cuda_result_review_20260508_codex.json"
            ),
            dispatch_blockers=["reactivation_required_before_new_dispatch"],
            source="auth_eval_work/contest_auth_eval.json",
        ),
    ]

    updated = tool.update_catalog_from_evidence(
        catalog,
        evidence,
        log_warnings=False,
    )

    row = updated[0]
    assert row["predicted_archive_bytes"] == 180_000
    assert row["exact_negative_evidence_n"] == 1
    assert row["measured_config_retired"] is True
    assert row["family_falsified"] is False
    assert row["method_family_retired"] is False
    assert row["measured_config_retired_only"] is True
    assert row["exact_negative_classification"] == "measured_config_retired_only"
    assert row["exact_negative_falsification_scopes"] == ["measured_config_only"]
    assert row["reactivation_criteria"] == [
        "retrain under scorer-aware loss",
        "prove byte-closed runtime packet",
    ]
    assert row["retired_from_active_ranking"] is True
    assert row["empirical_anchor_promotable"] is False
    assert row["rank_or_kill_eligible"] is False
    assert "exact-negative-N1" in row["evidence_grade"]
    assert (
        "exact_negative_result_requires_reactivation_before_empirical_byte_anchor"
        in row["dispatch_blockers"]
    )

    ranked = tool._rank_techniques(
        updated,
        d_seg=0.0,
        d_pose=0.0,
        current_archive_bytes=200_000,
        current_score=1.0,
        target_score=None,
    )
    assert ranked[0]["retired_from_active_ranking"] is True
    assert ranked[0]["predicted_score_delta"] == 0.0


def test_lossy_coarsening_builtin_catalog_classifies_reviewed_negative() -> None:
    tool = _load_tool_module()
    evidence = [
        tool.TechniqueEvidence(
            technique="lossy_coarsening_analytical",
            empirical_archive_bytes=156_344,
            evidence_grade="[MPS-research-signal]",
            evidence_semantics="mps_or_cpu_byte_roundtrip_proxy_no_score",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            dispatch_blockers=[
                "missing_exact_cuda_auth_eval_on_lossy_decoder",
                "no_runtime_dequantize_path_built",
            ],
            source="[MPS-research-signal] byte proxy manifest",
        ),
        tool.TechniqueEvidence(
            technique="lossy_coarsening_analytical",
            empirical_archive_bytes=156_404,
            evidence_grade="[contest-CUDA]",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            contest_dispatch_verdict="completed",
            source="[contest-CUDA] initial harvester row without review semantics",
        ),
        tool.TechniqueEvidence(
            technique="lossy_coarsening_analytical",
            empirical_archive_bytes=156_404,
            empirical_score=0.351718793322788,
            score_contest_cuda=0.351718793322788,
            evidence_grade="[contest-CUDA A-negative]",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            contest_dispatch_verdict="measured_config_retired_exact_cuda_negative",
            measured_config_status="measured_config_retired",
            family_falsified=False,
            method_family_retired=False,
            falsification_scope="measured_config_only_per_tensor_K_budget_0.05",
            reactivation_criteria=[
                "retrain or jointly optimize under scorer-aware loss",
                "prove byte-closed runtime packet below the active anchor",
            ],
            exact_result_review_path=(
                ".omx/research/"
                "lossy_coarsening_exact_cuda_result_review_20260508_codex.json"
            ),
            dispatch_blockers=[
                "measured_config_retired_exact_cuda_negative",
                "reactivation_required_before_new_dispatch",
            ],
            source="[contest-CUDA A-negative] reviewed auth eval",
        ),
    ]

    plan = tool.build_plan(
        d_seg=0.0,
        d_pose=0.0,
        archive_bytes=200_000,
        prior_evidence=evidence,
        include_axis_priorities=False,
    )

    report = plan.evidence_semantics_report
    assert "lossy_coarsening_analytical" in report[
        "cataloged_exact_negative_techniques"
    ]
    assert "lossy_coarsening_analytical" in report[
        "active_ranking_blocked_techniques"
    ]
    assert all(
        row["technique"] != "lossy_coarsening_analytical"
        for row in report["unknown_evidence_techniques"]
    )

    row = next(
        item for item in plan.arch_technique_ranking
        if item["name"] == "lossy_coarsening_analytical"
    )
    assert row["predicted_archive_bytes"] == 156_344
    assert row["predicted_score_delta"] == 0.0
    assert row["active_ranking_blocked"] is True
    assert row["retired_from_active_ranking"] is True
    assert row["measured_config_retired_only"] is True
    assert row["exact_negative_classification"] == "measured_config_retired_only"
    assert row["family_falsified"] is False
    assert row["method_family_retired"] is False
    assert row["supporting_non_promotable_evidence_n"] == 2
    assert row["exact_negative_falsification_scopes"] == [
        "measured_config_only_per_tensor_K_budget_0.05"
    ]
    assert row["reactivation_criteria"] == [
        "retrain or jointly optimize under scorer-aware loss",
        "prove byte-closed runtime packet below the active anchor",
    ]


def test_proxy_byte_anchor_cannot_dominate_active_ranking() -> None:
    tool = _load_tool_module()
    catalog = [{
        "name": "byte_only_proxy_lane",
        "predicted_archive_bytes": 190_000,
        "cost_dollars": 1.0,
        "cost_hours": 1.0,
    }, {
        "name": "unblocked_zero_delta_lane",
        "predicted_archive_bytes": 250_000,
        "cost_dollars": 100.0,
        "cost_hours": 1.0,
    }]
    evidence = [
        tool.TechniqueEvidence(
            technique="byte_only_proxy_lane",
            empirical_archive_bytes=50_000,
            evidence_grade="[CPU-prep byte-only]",
            evidence_semantics="cpu_byte_anchor_no_score",
            score_claim=False,
            promotion_eligible=False,
            rank_or_kill_eligible=False,
            ready_for_exact_eval_dispatch=False,
            dispatch_blockers=["missing_exact_cuda_auth_eval"],
            source="[CPU-prep byte-only] local manifest",
        ),
    ]

    updated = tool.update_catalog_from_evidence(
        catalog,
        evidence,
        log_warnings=False,
    )
    row = next(r for r in updated if r["name"] == "byte_only_proxy_lane")
    assert row["predicted_archive_bytes"] == 50_000
    assert row["empirical_anchor_promotable"] is False
    assert row["active_ranking_blocked"] is True

    ranked = tool._rank_techniques(
        updated,
        d_seg=0.0,
        d_pose=0.0,
        current_archive_bytes=200_000,
        current_score=1.0,
        target_score=None,
    )
    assert ranked[-1]["name"] == "byte_only_proxy_lane"
    assert ranked[-1]["active_ranking_blocked"] is True
    assert ranked[-1]["predicted_score_delta"] == 0.0


def test_build_plan_threads_cpu_rank_axis_into_rankings() -> None:
    tool = _load_tool_module()

    plan = tool.build_plan(
        d_seg=0.00067623,
        d_pose=0.00017198,
        archive_bytes=178_981,
        target_score=0.19,
        include_axis_priorities=False,
        rank_axis="cpu",
        current_score_axis="cuda",
        architecture_class="hnerv_ft_microcodec",
    )

    assert plan.operator_state["rank_axis"] == "cpu"
    assert plan.operator_state["architecture_class"] == "hnerv_ft_microcodec"
    assert all(
        row["rank_axis"] == "cpu"
        for row in plan.encoder_technique_ranking + plan.arch_technique_ranking
    )
    assert plan.recommended_top_3 == sorted(
        plan.encoder_technique_ranking + plan.arch_technique_ranking,
        key=lambda r: (
            r["active_ranking_blocked"],
            -r["primary_score_delta"],
            r["cost_dollars"],
        ),
    )[:3]


def test_unknown_evidence_techniques_are_reported_not_ranked() -> None:
    tool = _load_tool_module()
    evidence = [
        tool.TechniqueEvidence(
            technique="unmodeled_proxy_codec",
            empirical_archive_bytes=42_000,
            evidence_grade="[CPU-prep byte-only]",
            source="[CPU-prep byte-only] local manifest",
        ),
        tool.TechniqueEvidence(
            technique="unmodeled_exact_negative_codec",
            empirical_archive_bytes=40_000,
            evidence_grade="[contest-CUDA A-negative]",
            contest_dispatch_verdict="measured_config_retired_exact_cuda_negative",
            measured_config_status="measured_config_retired",
            source="[contest-CUDA A-negative] auth eval",
        ),
    ]

    plan = tool.build_plan(
        d_seg=0.0,
        d_pose=0.0,
        archive_bytes=200_000,
        prior_evidence=evidence,
        include_axis_priorities=False,
    )

    report = plan.evidence_semantics_report
    assert report["unknown_evidence_row_count"] == 2
    assert report["unknown_evidence_technique_count"] == 2
    assert report["unknown_exact_negative_row_count"] == 1
    unknown_names = {
        row["technique"] for row in report["unknown_evidence_techniques"]
    }
    assert unknown_names == {
        "unmodeled_proxy_codec",
        "unmodeled_exact_negative_codec",
    }
    ranked_names = {
        row["name"]
        for row in (
            plan.encoder_technique_ranking + plan.arch_technique_ranking
        )
    }
    assert "unmodeled_proxy_codec" not in ranked_names
    assert "unmodeled_exact_negative_codec" not in ranked_names
    assert "unknown technique" in " ".join(plan.notes)


def test_device_axis_report_requires_paired_archive_runtime_before_priority() -> None:
    tool = _load_tool_module()
    evidence = [
        tool.TechniqueEvidence(
            technique="apogee_pr107",
            empirical_score=0.19664189,
            empirical_d_seg=0.000589,
            empirical_d_pose=0.0000358,
            empirical_archive_bytes=178_637,
            evidence_grade="[macOS-CPU advisory only]",
            device_axis="macOS-CPU",
            archive_sha256="abc",
            runtime_tree_sha256="runtime",
            substrate_class="hnerv",
            source="local M5 advisory",
        )
    ]

    report = tool.summarize_device_axis_evidence(evidence)

    assert report["priority_status"] == "insufficient_paired_axis_evidence"
    assert report["paired_comparison_count"] == 0
    assert report["macos_cpu_advisory_count"] == 1
    assert report["unpaired_device_axis_evidence"][0]["reason"] == (
        "macos_cpu_research_proxy_needs_linux_contest_cpu_promotion"
    )
    assert "Do not assign CPU-vs-GPU" in report["policy"]
    assert "PR107 GHA-vs-M5" in report["policy"]


def test_device_axis_report_learns_per_substrate_profile_from_paired_axes() -> None:
    tool = _load_tool_module()
    evidence = [
        tool.TechniqueEvidence(
            technique="pr102",
            empirical_score=0.1953761765,
            score_contest_cpu=0.1953761765,
            empirical_d_seg=0.00057599,
            empirical_d_pose=0.00003460,
            empirical_archive_bytes=178_981,
            evidence_grade="[contest-CPU]",
            device_axis="contest_cpu",
            archive_sha256="same-archive",
            runtime_tree_sha256="same-runtime",
            substrate_class="hnerv_lc_v2",
            decoder_pose_ratio_cuda_over_cpu=1.25,
            network_pose_ratio_cuda_over_cpu=4.0,
            source="cpu json",
        ),
        tool.TechniqueEvidence(
            technique="pr102",
            empirical_score=0.2283908312,
            score_contest_cuda=0.2283908312,
            empirical_d_seg=0.00067565,
            empirical_d_pose=0.00017347,
            empirical_archive_bytes=178_981,
            evidence_grade="[contest-CUDA]",
            device_axis="contest_cuda",
            archive_sha256="same-archive",
            runtime_tree_sha256="same-runtime",
            substrate_class="hnerv_lc_v2",
            source="cuda json",
        ),
    ]

    report = tool.summarize_device_axis_evidence(evidence)

    assert report["priority_status"] == "paired_archive_specific_diagnostics_available"
    assert report["paired_comparison_count"] == 1
    pair = report["paired_comparisons"][0]
    assert abs(pair["pose_ratio_cuda_over_cpu"] - 5.013583815) < 1e-6
    assert abs(pair["seg_ratio_cuda_over_cpu"] - 1.173023837) < 1e-6
    profiles = report["substrate_class_profiles"]
    assert profiles[0]["substrate_class"] == "hnerv_lc_v2"
    assert profiles[0]["posterior_status"] == "needs_more_anchors"
    assert profiles[0]["decoder_pose_ratio_mean_cuda_over_cpu"] == 1.25
    assert profiles[0]["network_pose_ratio_mean_cuda_over_cpu"] == 4.0


def test_device_axis_report_rejects_conflicting_duplicate_axis_rows() -> None:
    tool = _load_tool_module()
    common = {
        "technique": "pr102",
        "archive_sha256": "same-archive",
        "runtime_tree_sha256": "same-runtime",
        "substrate_class": "hnerv_lc_v2",
    }
    evidence = [
        tool.TechniqueEvidence(
            **common,
            empirical_score=0.195,
            score_contest_cpu=0.195,
            empirical_d_seg=0.00057,
            empirical_d_pose=0.000034,
            evidence_grade="[contest-CPU]",
            device_axis="contest_cpu",
            source="cpu json a",
        ),
        tool.TechniqueEvidence(
            **common,
            empirical_score=0.196,
            score_contest_cpu=0.196,
            empirical_d_seg=0.00058,
            empirical_d_pose=0.000035,
            evidence_grade="[contest-CPU]",
            device_axis="contest_cpu",
            source="cpu json b",
        ),
        tool.TechniqueEvidence(
            **common,
            empirical_score=0.228,
            score_contest_cuda=0.228,
            empirical_d_seg=0.00067,
            empirical_d_pose=0.00017,
            evidence_grade="[contest-CUDA]",
            device_axis="contest_cuda",
            source="cuda json",
        ),
    ]

    report = tool.summarize_device_axis_evidence(evidence)

    assert report["paired_comparison_count"] == 0
    assert report["priority_status"] == "insufficient_paired_axis_evidence"
    assert report["unpaired_device_axis_evidence"][0]["reason"] == (
        "conflicting_duplicate_axis_rows_same_archive_runtime"
    )
