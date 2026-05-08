from __future__ import annotations

import math

from tac.optimization.candidate_evidence_contract import (
    is_promotable_exact_cuda_evidence,
    promotable_exact_cuda_evidence_blockers,
)

CONTEST_N_BYTES = 37_545_489


def _full_exact_cuda_row(**overrides):
    archive_bytes = 100_000
    rate_term = 25.0 * archive_bytes / CONTEST_N_BYTES
    recomputed_score = 100.0 * 0.0005 + math.sqrt(10.0 * 0.00003) + rate_term
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": archive_bytes,
        "score_contest_cuda": recomputed_score,
        "evidence_grade": "[contest-CUDA]",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "sample_count": 1199,
        "seg_distortion": 0.0005,
        "pose_distortion": 0.00003,
        "rate_term": rate_term,
        "recomputed_score": recomputed_score,
        "exact_eval_command": "python experiments/contest_auth_eval.py --device cuda",
        "hardware": "Lightning T4",
        "log_path": "external:contest_auth_eval.log",
        "dispatch_claim_status": f"completed_score_{recomputed_score:.6f}",
        "dispatch_blockers": [],
    }
    row.update(overrides)
    return row


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
    row = _full_exact_cuda_row()

    assert promotable_exact_cuda_evidence_blockers(row) == []
    assert is_promotable_exact_cuda_evidence(row) is True


def test_promotable_exact_cuda_evidence_recomputes_formula() -> None:
    row = _full_exact_cuda_row(recomputed_score=0.18)

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert "recomputed_score_formula_mismatch" in blockers
    assert "contest_cuda_score_recomputed_score_mismatch" in blockers
    assert is_promotable_exact_cuda_evidence(row) is False


def test_promotable_exact_cuda_evidence_rejects_bad_rate_term() -> None:
    row = _full_exact_cuda_row(rate_term=0.066)

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert "rate_term_formula_mismatch" in blockers
    assert is_promotable_exact_cuda_evidence(row) is False


def test_minimal_exact_cuda_row_with_shas_still_requires_gate8_custody() -> None:
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

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert "sample_count_ge_600_required" in blockers
    assert "seg_component_required" in blockers
    assert "pose_component_required" in blockers
    assert "rate_component_required" in blockers
    assert "recomputed_score_required" in blockers
    assert "exact_eval_command_required" in blockers
    assert "hardware_required" in blockers
    assert "log_path_required" in blockers
    assert "dispatch_claim_status_required" in blockers
    assert is_promotable_exact_cuda_evidence(row) is False


def test_free_form_source_path_proxy_words_do_not_block_exact_cuda() -> None:
    row = _full_exact_cuda_row(
        contest_cuda_score=0.18,
        source=(
            "experiments/results/predicted_from_r2_component_response/"
            "contest_auth_eval.json"
        ),
    )

    assert promotable_exact_cuda_evidence_blockers(row) == []
    assert is_promotable_exact_cuda_evidence(row) is True


def test_bracketed_source_proxy_label_still_blocks_promotability() -> None:
    row = _full_exact_cuda_row(
        contest_cuda_score=0.18,
        source="[MPS-research-signal] local proxy artifact",
    )

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert "proxy_or_planning_marker_not_promotable" in blockers
    assert is_promotable_exact_cuda_evidence(row) is False


def test_string_booleans_are_schema_invalid_and_not_promotable() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "score_contest_cuda": 0.18,
        "evidence_grade": "[contest-CUDA]",
        "score_claim": "false",
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "archive_sha256": "a" * 64,
        "runtime_tree_sha256": "b" * 64,
        "dispatch_blockers": [],
    }

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert is_promotable_exact_cuda_evidence(row) is False
    assert "invalid_evidence_schema_non_promotable" in blockers
    assert "invalid_evidence_schema_boolean:score_claim" in blockers
    assert "score_claim_true_required" in blockers


def test_negated_exact_cuda_requirement_text_is_not_positive_marker() -> None:
    row = {
        "technique": "candidate",
        "empirical_archive_bytes": 100_000,
        "score_contest_cuda": 0.18,
        "evidence_semantics": "requires_exact_cuda_auth_eval_before_any_score_use",
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
    assert "exact_cuda_evidence_marker_required" in blockers


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


def test_negative_pose_component_fails_closed_instead_of_crashing() -> None:
    row = _full_exact_cuda_row(pose_distortion=-0.00003)

    blockers = promotable_exact_cuda_evidence_blockers(row)

    assert is_promotable_exact_cuda_evidence(row) is False
    assert "pose_component_nonnegative_required" in blockers
