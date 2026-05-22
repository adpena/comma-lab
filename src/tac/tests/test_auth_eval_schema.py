# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from tac.auth_eval_schema import (
    auth_eval_completion_summary,
    contest_formula_score,
    eval_device,
    eval_metric_summary,
    main,
    required_contest_auth_axis_payload_blockers,
    required_contest_cpu_evidence_blockers,
    required_contest_cuda_evidence_blockers,
    required_exact_eval_metric_blockers,
)


def test_eval_metric_summary_prefers_canonical_auth_eval_schema() -> None:
    score = contest_formula_score(
        seg_dist=0.000665,
        pose_dist=0.00017099,
        archive_bytes=178392,
    )
    metrics = eval_metric_summary(
        {
            "canonical_score": score,
            "score_recomputed_from_components": score,
            "final_score": 0.23,
            "avg_posenet_dist": 0.00017099,
            "avg_segnet_dist": 0.000665,
            "score_rate_contribution": 25 * 178392 / 37_545_489,
            "rate_unscaled": 178392 / 37_545_489,
            "archive_size_bytes": 178392,
            "n_samples": 600,
            "canonical_score_source": "score_recomputed_from_components",
        }
    )

    assert metrics["score"] == score
    assert metrics["pose_avg"] == 0.00017099
    assert metrics["seg_avg"] == 0.000665
    assert metrics["rate"] == 25 * 178392 / 37_545_489
    assert metrics["rate_unscaled"] == 178392 / 37_545_489
    assert metrics["archive_size_bytes"] == 178392
    assert metrics["n_samples"] == 600
    assert metrics["canonical_score_source"] == "score_recomputed_from_components"
    assert required_exact_eval_metric_blockers(metrics, expected_archive_bytes=178392) == []


def test_eval_metric_summary_keeps_legacy_score_components_compatible() -> None:
    metrics = eval_metric_summary(
        {
            "score": "0.31",
            "archive_bytes": "1234",
            "score_components": {
                "pose": "0.0002",
                "seg": 0.0006,
                "rate": 0.119,
                "rate_unscaled": 0.00476,
            },
        }
    )

    assert metrics["score"] == 0.31
    assert metrics["pose_avg"] == 0.0002
    assert metrics["seg_avg"] == 0.0006
    assert metrics["rate"] == 0.119
    assert metrics["rate_unscaled"] == 0.00476
    assert metrics["archive_size_bytes"] == 1234
    assert "canonical_score_source_not_recomputed_from_components" in (
        required_exact_eval_metric_blockers(metrics)
    )


def test_required_exact_eval_metric_blockers_fail_closed_on_null_score_fields() -> None:
    metrics = eval_metric_summary(
        {
            "final_score": 0.23,
            "archive_size_bytes": 100,
        }
    )

    blockers = required_exact_eval_metric_blockers(
        metrics,
        expected_archive_bytes=101,
        expected_n_samples=600,
    )

    assert "pose_avg_missing" in blockers
    assert "seg_avg_missing" in blockers
    assert "rate_unscaled_missing" in blockers
    assert "canonical_score_source_not_recomputed_from_components" in blockers
    assert "archive_size_bytes_mismatch:manifest=100:actual=101" in blockers
    assert "n_samples_missing" in blockers


def test_required_exact_eval_metric_blockers_catches_partial_sample_count() -> None:
    score = contest_formula_score(
        seg_dist=0.000665,
        pose_dist=0.00017099,
        archive_bytes=178392,
    )
    metrics = eval_metric_summary(
        {
            "canonical_score": score,
            "avg_posenet_dist": 0.00017099,
            "avg_segnet_dist": 0.000665,
            "rate_unscaled": 178392 / 37_545_489,
            "archive_size_bytes": 178392,
            "n_samples": 64,
            "canonical_score_source": "score_recomputed_from_components",
        }
    )

    assert required_exact_eval_metric_blockers(metrics, expected_n_samples=600) == [
        "n_samples_mismatch:manifest=64:expected=600"
    ]


def test_required_exact_eval_metric_blockers_rejects_non_finite_scores() -> None:
    metrics = eval_metric_summary(
        {
            "canonical_score": "nan",
            "avg_posenet_dist": "inf",
            "avg_segnet_dist": 0.000665,
            "rate_unscaled": 178392 / 37_545_489,
            "archive_size_bytes": 178392,
            "n_samples": 600,
            "canonical_score_source": "score_recomputed_from_components",
        }
    )

    blockers = required_exact_eval_metric_blockers(metrics, expected_n_samples=600)

    assert "score_missing" in blockers
    assert "pose_avg_missing" in blockers


def test_required_exact_eval_metric_blockers_rejects_score_formula_mismatch() -> None:
    metrics = eval_metric_summary(
        {
            "canonical_score": 0.123,
            "score_recomputed_from_components": 0.123,
            "avg_posenet_dist": 0.00017099,
            "avg_segnet_dist": 0.000665,
            "rate_unscaled": 178392 / 37_545_489,
            "archive_size_bytes": 178392,
            "n_samples": 600,
            "canonical_score_source": "score_recomputed_from_components",
        }
    )

    blockers = required_exact_eval_metric_blockers(metrics, expected_n_samples=600)

    assert any(
        blocker.startswith("score_component_formula_mismatch")
        for blocker in blockers
    )


def test_required_exact_eval_metric_blockers_rejects_rate_bytes_mismatch() -> None:
    score = contest_formula_score(
        seg_dist=0.000665,
        pose_dist=0.00017099,
        archive_bytes=178392,
    )
    metrics = eval_metric_summary(
        {
            "canonical_score": score,
            "score_recomputed_from_components": score,
            "avg_posenet_dist": 0.00017099,
            "avg_segnet_dist": 0.000665,
            "rate_unscaled": 0.00474835,
            "archive_size_bytes": 178392,
            "n_samples": 600,
            "canonical_score_source": "score_recomputed_from_components",
        }
    )

    blockers = required_exact_eval_metric_blockers(metrics, expected_n_samples=600)

    assert any(
        blocker.startswith("rate_unscaled_archive_bytes_mismatch")
        for blocker in blockers
    )


def _contest_cuda_payload(**overrides):
    score = contest_formula_score(
        seg_dist=0.000665,
        pose_dist=0.00017099,
        archive_bytes=178392,
    )
    payload = {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "avg_posenet_dist": 0.00017099,
        "avg_segnet_dist": 0.000665,
        "score_rate_contribution": 25 * 178392 / 37_545_489,
        "rate_unscaled": 178392 / 37_545_489,
        "archive_size_bytes": 178392,
        "n_samples": 600,
        "canonical_score_source": "score_recomputed_from_components",
        "lane_tag": "[contest-CUDA]",
        "score_axis": "contest_cuda",
        "evidence_semantics": "contest_cuda_exact_auth_eval",
        "score_claim_valid": True,
        "promotion_eligible": False,
        "provenance": {
            "device": "cuda",
            "gpu_t4_match": True,
        },
    }
    payload.update(overrides)
    return payload


def _strict_contest_cuda_payload(**overrides):
    payload = _contest_cuda_payload(
        evidence_grade="contest-CUDA",
        exact_cuda_eval_complete=True,
        score_claim=True,
        rank_or_kill_eligible=False,
    )
    payload.update(overrides)
    return payload


def _strict_contest_cpu_payload(**overrides):
    score = contest_formula_score(
        seg_dist=0.000665,
        pose_dist=0.00017099,
        archive_bytes=178392,
    )
    payload = {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "avg_posenet_dist": 0.00017099,
        "avg_segnet_dist": 0.000665,
        "score_rate_contribution": 25 * 178392 / 37_545_489,
        "rate_unscaled": 178392 / 37_545_489,
        "archive_size_bytes": 178392,
        "n_samples": 600,
        "canonical_score_source": "score_recomputed_from_components",
        "evidence_grade": "contest-CPU",
        "lane_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        "evidence_semantics": "public_leaderboard_cpu_reproduction",
        "exact_cuda_eval_complete": False,
        "score_claim": True,
        "score_claim_valid": True,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "cpu_leaderboard_reproduction_eligible": True,
        "provenance": {
            "device": "cpu",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
    }
    payload.update(overrides)
    return payload


def test_required_contest_cuda_evidence_blockers_accepts_t4_full_sample_eval() -> None:
    payload = _contest_cuda_payload()
    metrics = eval_metric_summary(payload)

    assert required_contest_cuda_evidence_blockers(
        payload,
        metrics,
        expected_archive_bytes=178392,
        expected_n_samples=600,
    ) == []


def test_required_contest_cpu_evidence_blockers_accepts_linux_x86_full_sample_eval() -> None:
    payload = _strict_contest_cpu_payload()
    metrics = eval_metric_summary(payload)

    assert required_contest_cpu_evidence_blockers(
        payload,
        metrics,
        expected_archive_bytes=178392,
        expected_n_samples=600,
    ) == []


def test_required_contest_cpu_evidence_blockers_rejects_darwin_arm64() -> None:
    payload = _strict_contest_cpu_payload(
        provenance={
            "device": "cpu",
            "platform_system": "Darwin",
            "platform_machine": "arm64",
        },
    )
    metrics = eval_metric_summary(payload)

    blockers = required_contest_cpu_evidence_blockers(payload, metrics)

    assert "contest_cpu_platform_system_not_linux" in blockers
    assert "contest_cpu_platform_machine_not_x86_64" in blockers


def test_required_contest_auth_axis_payload_blockers_accept_strict_cuda_and_cpu() -> None:
    cuda_payload = _strict_contest_cuda_payload()
    cpu_payload = _strict_contest_cpu_payload()

    assert required_contest_auth_axis_payload_blockers(
        cuda_payload,
        eval_metric_summary(cuda_payload),
    ) == []
    assert required_contest_auth_axis_payload_blockers(
        cpu_payload,
        eval_metric_summary(cpu_payload),
    ) == []


def test_required_contest_auth_axis_payload_blockers_rejects_forged_cuda_labels() -> None:
    payload = _strict_contest_cuda_payload(
        exact_cuda_eval_complete=False,
        score_claim_valid=False,
        provenance={"device": "mps", "gpu_t4_match": True},
        diagnostic_blockers=["synthetic_diagnostic_blocker"],
    )
    metrics = eval_metric_summary(payload)

    blockers = required_contest_auth_axis_payload_blockers(payload, metrics)

    assert "diagnostic_blockers_present" in blockers
    assert "device_not_cuda" in blockers
    assert "exact_cuda_eval_complete_not_true" in blockers
    assert "score_claim_valid_not_true" in blockers


def test_required_contest_auth_axis_payload_blockers_rejects_forged_cpu_labels() -> None:
    payload = _strict_contest_cpu_payload(
        provenance={
            "device": "cpu",
            "platform_system": "Darwin",
            "platform_machine": "arm64",
        },
        cpu_leaderboard_reproduction_eligible=False,
    )
    metrics = eval_metric_summary(payload)

    blockers = required_contest_auth_axis_payload_blockers(payload, metrics)

    assert "contest_cpu_platform_system_not_linux" in blockers
    assert "contest_cpu_platform_machine_not_x86_64" in blockers
    assert "cpu_leaderboard_reproduction_eligible_not_true" in blockers


def test_required_contest_cuda_evidence_blockers_rejects_cpu_axis_even_with_metrics() -> None:
    payload = _contest_cuda_payload(
        lane_tag="[contest-CPU]",
        score_axis="contest_cpu",
        evidence_semantics="public_leaderboard_cpu_reproduction",
        score_claim_valid=False,
        promotion_eligible=False,
        provenance={"device": "cpu", "platform_system": "Linux", "platform_machine": "x86_64"},
    )
    metrics = eval_metric_summary(payload)

    blockers = required_contest_cuda_evidence_blockers(payload, metrics, expected_n_samples=600)

    assert "device_not_cuda" in blockers
    assert "contest_cuda_hardware_not_t4_or_documented_equivalent" in blockers
    assert "evidence_tag_not_contest_cuda" in blockers
    assert not any(blocker.startswith("lane_") for blocker in blockers)
    assert "score_axis_not_contest_cuda" in blockers
    assert "evidence_semantics_not_contest_cuda_exact_auth_eval" in blockers
    assert "score_claim_valid_not_true" in blockers
    assert "promotion_eligible_not_true" not in blockers


def test_eval_device_prefers_actual_device_over_requested_device() -> None:
    assert (
        eval_device(
            {
                "device": "cuda",
                "requested_device": "cuda",
                "actual_device": "cpu",
            }
        )
        == "cpu"
    )
    assert (
        eval_device(
            {
                "device": "cuda",
                "provenance": {"actual_device": "mps"},
            }
        )
        == "mps"
    )


def test_required_contest_cuda_evidence_blockers_do_not_require_promotion_eligibility() -> None:
    payload = _contest_cuda_payload(
        promotion_eligible=False,
        rank_or_kill_eligible=False,
    )
    metrics = eval_metric_summary(payload)

    assert required_contest_cuda_evidence_blockers(
        payload,
        metrics,
        expected_archive_bytes=178392,
        expected_n_samples=600,
    ) == []


def test_required_contest_cuda_evidence_blockers_rejects_cuda_without_t4_or_equivalence() -> None:
    payload = _contest_cuda_payload(provenance={"device": "cuda", "gpu_t4_match": False})
    metrics = eval_metric_summary(payload)

    blockers = required_contest_cuda_evidence_blockers(payload, metrics, expected_n_samples=600)

    assert blockers == ["contest_cuda_hardware_not_t4_or_documented_equivalent"]


def test_required_contest_cuda_evidence_blockers_accepts_documented_equivalent_cuda() -> None:
    payload = _contest_cuda_payload(
        provenance={
            "device": "cuda",
            "gpu_t4_match": False,
            "contest_equivalent_hardware": True,
            "contest_equivalent_hardware_note": "operator-approved equivalent CUDA runner",
        },
    )
    metrics = eval_metric_summary(payload)

    assert required_contest_cuda_evidence_blockers(payload, metrics, expected_n_samples=600) == []


def test_auth_eval_completion_summary_uses_evaluator_evidence_fields() -> None:
    payload = _contest_cuda_payload(
        evidence_grade="B",
        lane_tag="[diagnostic-auth-eval]",
        score_axis="diagnostic_cuda",
        evidence_semantics="diagnostic_auth_eval_non_promotable",
        score_claim=False,
        score_claim_valid=False,
        provenance={"device": "cuda", "gpu_model": "Tesla P100", "gpu_t4_match": False},
    )

    summary = auth_eval_completion_summary(payload)

    assert summary["score"] == payload["score_recomputed_from_components"]
    assert summary["lane_tag"] == "[diagnostic-auth-eval]"
    assert summary["score_axis"] == "diagnostic_cuda"
    assert summary["score_claim"] is False
    assert summary["score_claim_valid"] is False
    assert summary["device"] == "cuda"
    assert summary["gpu_model"] == "Tesla P100"
    assert summary["gpu_t4_match"] is False


def test_auth_eval_completion_summary_cli_outputs_json_and_fields(tmp_path) -> None:
    payload = _contest_cuda_payload(
        evidence_grade="B",
        lane_tag="[diagnostic-auth-eval]",
        score_axis="diagnostic_cuda",
        evidence_semantics="diagnostic_auth_eval_non_promotable",
        score_claim=False,
        score_claim_valid=False,
    )
    path = tmp_path / "contest_auth_eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["completion-summary", str(path)]) == 0
    summary = json.loads(out.getvalue())

    assert summary["score"] == payload["score_recomputed_from_components"]
    assert summary["score_axis"] == "diagnostic_cuda"
    assert summary["score_claim"] is False

    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["completion-summary", str(path), "--field", "score"]) == 0

    assert float(out.getvalue().strip()) == payload["score_recomputed_from_components"]
