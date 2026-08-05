from __future__ import annotations

import tools.au1_measurement_integrity_audit as au1


def test_git_subject_positive_control_933_is_caught() -> None:
    subject = (
        "ddm_tz1: archive token-sweep RATE attack harness + byte-only leg landed "
        "[no-triality] [p0-ledger-ok] - reproduces bs2/gk2 L=14 smevr -23,655 B "
        "AND corrects to LIVE ix2 -24,605 B (form conflation)"
    )
    split = au1.split_subject_on_refutation(subject)
    assert split is not None
    title, body = split
    rows = au1.body_refutation_different_number(
        source="git_log:.omx/state/canonical_task_status.jsonl:eb3d47ce90",
        source_kind="task_ledger_git_subject",
        title=title,
        body=body,
    )
    assert rows
    serialized = au1.json_dump_row(rows[0])
    assert "23,655" in serialized
    assert "24,605" in serialized


def test_absent_receipt_931_is_not_caught_without_refutation() -> None:
    title = "#931 absent-receipt check"
    body = "No receipt was found in the searched scope. This row has no corrected numeric claim."
    rows = au1.body_refutation_different_number(
        source=".omx/research/example.md",
        source_kind="research_memo",
        title=title,
        body=body,
    )
    assert rows == []


def test_decode_read_seed_has_required_denominator_and_positive_controls() -> None:
    rows, summary = au1.decode_read_coverage_rows()
    assert summary["reads_enumerated"] == 25
    assert summary["reads_classified"] == 25
    assert summary["positive_control_933_pm1"] is True
    assert summary["positive_control_933_l16"] is True
    controls = {row["positive_control"]: row for row in rows if row["positive_control"]}
    assert controls["#933_pm1_clamp"]["classification"] == "NONE"
    assert controls["#933_L16"]["classification"] == "NONE"


def test_correction_rows_pick_adjacent_numeric_refutation() -> None:
    text = "\n".join(
        [
            "# Memo",
            "Headline had 23,655 B.",
            "Corrected value is -24,605 B after form-conflation audit.",
        ]
    )
    rows = au1.correction_rows_for_text(".omx/research/memo.md", text)
    assert rows
    assert rows[0]["refuted_value"] == "23,655"
    assert rows[0]["corrected_value"] == "-24,605"
