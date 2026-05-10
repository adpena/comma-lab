from __future__ import annotations

from tac.auth_eval_schema import (
    contest_formula_score,
    eval_metric_summary,
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


def test_required_contest_cuda_evidence_blockers_accepts_t4_full_sample_eval() -> None:
    payload = _contest_cuda_payload()
    metrics = eval_metric_summary(payload)

    assert required_contest_cuda_evidence_blockers(
        payload,
        metrics,
        expected_archive_bytes=178392,
        expected_n_samples=600,
    ) == []


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
