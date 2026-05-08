from __future__ import annotations

from tac.optimization.candidate_evidence_contract import (
    is_promotable_exact_cuda_evidence,
    promotable_exact_cuda_evidence_blockers,
)


def test_promotable_exact_cuda_evidence_requires_archive_and_runtime_custody() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "empirical_score": 0.18,
        "evidence_grade": "[contest-CUDA]",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "dispatch_blockers": [],
    }

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert is_promotable_exact_cuda_evidence(row) is False
    assert "archive_sha256_required" in blockers
    assert "runtime_tree_sha256_required" in blockers


def test_promotable_exact_cuda_evidence_passes_with_full_contract() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "score_contest_cuda": 0.18,
        "evidence_grade": "[contest-CUDA]",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "dispatch_blockers": [],
    }

    assert promotable_exact_cuda_evidence_blockers(row) == []
    assert is_promotable_exact_cuda_evidence(row) is True


def test_negative_or_proxy_marked_rows_never_promote() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "score_contest_cuda": 0.35,
        "evidence_grade": "[contest-CUDA A-negative]",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "dispatch_blockers": [],
        "contest_dispatch_verdict": "measured_config_retired_exact_cuda_negative",
    }

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert is_promotable_exact_cuda_evidence(row) is False
    assert "negative_or_retired_result_not_promotable" in blockers


def test_generic_empirical_score_is_not_cuda_score_custody() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "empirical_score": 0.18,
        "evidence_grade": "[contest-CUDA]",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "dispatch_blockers": [],
    }

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert is_promotable_exact_cuda_evidence(row) is False
    assert "contest_cuda_score_required" in blockers


def test_non_finite_cuda_score_is_not_promotable() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "score_contest_cuda": float("nan"),
        "evidence_grade": "[contest-CUDA]",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "dispatch_blockers": [],
    }

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert is_promotable_exact_cuda_evidence(row) is False
    assert "contest_cuda_score_required" in blockers
