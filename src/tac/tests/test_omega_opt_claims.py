from __future__ import annotations

from tac.omega_opt_claims import (
    FAIL_CLOSED_FIELDS,
    OMEGA_OPT_CLAIMS,
    has_exact_1to1_anchor,
    omega_opt_claim_manifest,
    omega_opt_claim_rows,
    validate_omega_opt_claim_table,
    validate_omega_opt_ledger_text,
    validate_omega_opt_row,
)


def test_canonical_omega_opt_claim_table_has_all_fail_closed_levels() -> None:
    manifest = omega_opt_claim_manifest()

    assert manifest["claim_count"] == 8
    assert len(manifest["claims"]) == 8
    assert [claim.predicted_score for claim in OMEGA_OPT_CLAIMS] == [
        0.130,
        0.115,
        0.110,
        0.105,
        0.100,
        0.095,
        0.092,
        0.090,
    ]
    for row in manifest["claims"]:
        for field in FAIL_CLOSED_FIELDS:
            assert row[field] is False
        assert row["promotion_allowed"] is False
        assert row["dispatchable"] is False
        assert row["requires_exact_1to1_anchor"] is True

    assert validate_omega_opt_claim_table(manifest["claims"]) == []


def test_unanchored_omega_opt_evidence_cannot_set_promotion_flags() -> None:
    row = {
        "technique": "omega_opt_score_feedback_meta_pass",
        "evidence_grade": "prediction",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "source": "Omega-OPT score-feedback predicted 0.095, no archive",
    }

    findings = validate_omega_opt_row(row)

    assert "score_claim_must_be_false_without_exact_1to1_anchor" in findings
    assert "promotion_eligible_must_be_false_without_exact_1to1_anchor" in findings
    assert "rank_or_kill_eligible_must_be_false_without_exact_1to1_anchor" in findings
    assert "ready_for_exact_eval_dispatch_must_be_false_without_exact_1to1_anchor" in findings


def test_exact_1to1_anchor_is_the_only_path_to_true_flags() -> None:
    row = {
        "claim_id": "omega_opt_linear_stack",
        "evidence_grade": "A++",
        "score_claim": True,
        "promotion_eligible": True,
        "rank_or_kill_eligible": True,
        "ready_for_exact_eval_dispatch": True,
        "exact_archive_sha256": "a" * 64,
        "exact_archive_bytes": 12345,
        "contest_auth_eval_json": "experiments/results/run/contest_auth_eval.json",
        "one_to_one_anchor_artifact": "experiments/results/run/one_to_one_anchor.json",
    }

    assert has_exact_1to1_anchor(row) is True
    assert validate_omega_opt_row(row) == []


def test_ledger_text_requires_every_claim_and_fail_closed_fields() -> None:
    rows = omega_opt_claim_rows()
    text = "\n".join(
        " | ".join(
            [
                row["claim_id"],
                "Next 1:1 test",
                "score_claim=false",
                "promotion_eligible=false",
                "rank_or_kill_eligible=false",
                "ready_for_exact_eval_dispatch=false",
            ]
        )
        for row in rows
    )

    assert validate_omega_opt_ledger_text(text) == []
    assert validate_omega_opt_ledger_text(text.replace("omega_opt_linear_stack", "")) == [
        "omega_opt_linear_stack: ledger_missing_token:omega_opt_linear_stack"
    ]
