from __future__ import annotations

import pytest

from tac.audit_contract import AuditReport, audit_exit_code


def test_audit_report_json_contract_never_claims_score_or_dispatch_by_default() -> None:
    report = AuditReport(
        audit="example_gate",
        readiness_key="ready_for_example",
        ready=True,
        summary={"row_count": 3},
        metadata={"artifact": "relative/path.json"},
    )

    payload = report.to_dict()

    assert payload == {
        "artifact": "relative/path.json",
        "audit": "example_gate",
        "blockers": [],
        "dispatch_attempted": False,
        "ready_for_example": True,
        "score_claim": False,
        "summary": {"row_count": 3},
    }
    assert audit_exit_code(report) == 0


def test_audit_report_failure_text_and_exit_code() -> None:
    report = AuditReport(
        audit="example_gate",
        readiness_key="ready_for_example",
        ready=False,
        blockers=("missing_artifact", "stale_metadata"),
    )

    assert audit_exit_code(report) == 2
    assert report.render_text() == (
        "example gate: FAIL\n"
        "  - missing_artifact\n"
        "  - stale_metadata"
    )


def test_audit_report_metadata_cannot_override_safety_fields() -> None:
    with pytest.raises(ValueError, match="score_claim"):
        AuditReport(
            audit="example_gate",
            readiness_key="ready_for_example",
            ready=True,
            score_claim=False,
            metadata={"score_claim": True},
        )

    with pytest.raises(ValueError, match="ready_for_example"):
        AuditReport(
            audit="example_gate",
            readiness_key="ready_for_example",
            ready=True,
            metadata={"ready_for_example": False},
        )
