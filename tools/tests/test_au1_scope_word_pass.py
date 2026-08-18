"""Controls for the au1 scope-word pass — the closure-names-wrong-object detector.

The instrument asks one question of every closure sentence: *which object did this unit
measure, and is that the object the sentence names?*  It is WARN-ONLY, so its job is to be
readable, not to block; that makes both directions mandatory.  A detector that fires on the
house dialect ("byte-closed", "closed-form") is one nobody reads, and a detector that fires
on nothing is one nobody needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from au1_measurement_integrity_audit import (
    scope_vs_object_detector,
    scope_word_rows,
)


def _flag(text: str) -> list[dict[str, object]]:
    rows, _ = scope_word_rows(source="<test>", text=text)
    return rows


def _scoped(text: str) -> int:
    _, scoped = scope_word_rows(source="<test>", text=text)
    return scoped


# --------------------------------------------------------------------------------------
# RED — the disease, in the exact shape that cost a day
# --------------------------------------------------------------------------------------


def test_red_the_na9_instance_four_sentence_is_caught() -> None:
    rows = _flag("The coder axis is CLOSED on hv1.")

    assert len(rows) == 1
    assert rows[0]["closure_words"] == ["closed"]
    assert rows[0]["wide_object_words"] == ["axis"]
    assert "which object did this unit MEASURE" in str(rows[0]["question"])
    assert rows[0]["advisory_only"] is True


def test_red_the_other_three_measured_instances_are_caught() -> None:
    assert _flag("The carrier family's ceiling is EXHAUSTED.")
    assert _flag("This paradigm is DEAD.")
    assert _flag("The whole class is retired.")


# --------------------------------------------------------------------------------------
# GREEN — the honest form must pass, or the gauge is uninformative
# --------------------------------------------------------------------------------------


def test_green_the_same_claim_correctly_scoped_does_not_fire() -> None:
    text = "The coder-swap axis is CLOSED given fixed probabilities."

    assert _flag(text) == []
    assert _scoped(text) == 1  # counted as a pass, not silently dropped


def test_green_a_scope_ladder_word_is_enough_to_pass() -> None:
    for qualifier in ("at INSTANCE scope", "at FORMULATION scope", "measured on hv1 only"):
        assert _flag(f"The carrier family is closed {qualifier}.") == []


# --------------------------------------------------------------------------------------
# The house dialect must not trip it (measured: 382 + 257 false positives before this)
# --------------------------------------------------------------------------------------


def test_our_own_idioms_are_not_closures() -> None:
    assert _flag("The candidate is byte-closed at 180,601 B across the whole family.") == []
    assert _flag("Canonical equation #26 closed form applies to every class in the family.") == []
    assert _flag("The gate is fail-closed for this family.") == []
    assert _flag("We missed the deadline for the family run.") == []


def test_a_distant_closure_and_object_are_two_clauses_not_one_claim() -> None:
    far = "the axis was measured " + "x" * 80 + " and a separate run was closed"
    assert _flag(far) == []


def test_headers_rules_and_quotations_are_not_this_memo_asserting() -> None:
    assert _flag("# The coder axis is CLOSED") == []
    assert _flag("> The coder axis is CLOSED") == []
    assert _flag("|--- the axis is closed ---|") == []


# --------------------------------------------------------------------------------------
# The pass reports its denominators and carries its own controls
# --------------------------------------------------------------------------------------


def test_detector_summary_reports_both_sides(tmp_path: Path) -> None:
    (tmp_path / "wide.md").write_text("The coder axis is CLOSED on hv1.\n")
    (tmp_path / "scoped.md").write_text("The coder-swap axis is CLOSED given fixed probabilities.\n")

    rows, summary = scope_vs_object_detector(tmp_path)

    assert summary["memos_scanned"] == 2
    assert summary["rows_emitted"] == 1
    assert summary["sources_flagged"] == 1
    # The honest denominator: a reader can tell "clean" from "did not look".
    assert summary["closure_sentences_already_scoped"] == 1
    assert summary["advisory_only"] is True


def test_detector_carries_its_own_positive_and_negative_controls(tmp_path: Path) -> None:
    """A gauge whose known-positive is invisible is not a gauge."""
    _, summary = scope_vs_object_detector(tmp_path)

    assert summary["positive_control_caught"] is True
    assert summary["negative_control_caught"] is False
    assert summary["negative_control_recognised_as_scoped"] is True


def test_empty_corpus_reports_zero_rather_than_looking_clean(tmp_path: Path) -> None:
    rows, summary = scope_vs_object_detector(tmp_path)

    assert rows == []
    assert summary["memos_scanned"] == 0
    assert summary["closure_sentences_already_scoped"] == 0
