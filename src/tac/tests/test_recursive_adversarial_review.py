# SPDX-License-Identifier: MIT
"""Minimal-viable tests for tac.recursive_adversarial_review.

Covers: dataclass invariants, bundle_id determinism, append/load roundtrip,
counter semantics (clean/reset/SEAL), strict-load corruption raise, query
helpers. Sister 4-proc spawn-pool concurrent-append stress test deferred to
sister test file per Catalog #245 pattern.
"""
from __future__ import annotations

import json
import uuid

import pytest

from tac.recursive_adversarial_review import (
    CANONICAL_AXES,
    SEAL_THRESHOLD,
    VALID_SEVERITIES,
    VALID_VERDICTS,
    RecursiveReviewLedgerCorruptError,
    RecursiveReviewRound,
    ReviewFinding,
    append_round_locked,
    clean_pass_counter_for_bundle,
    compute_bundle_id,
    compute_scope_content_sha256,
    load_rounds_lenient,
    load_rounds_strict,
    query_unresolved_critical_findings,
    update_from_anchor,
)


def _finding(sev="CRITICAL", fid="C-1") -> ReviewFinding:
    return ReviewFinding(
        finding_id=fid,
        axis="assumption_challenge",
        severity=sev,
        member="Contrarian",
        description="x",
        recommended_fix="y",
    )


def _round(
    *,
    counter_before=0,
    counter_after=0,
    findings=(),
    scope_content_sha256="0" * 64,
) -> RecursiveReviewRound:
    return RecursiveReviewRound(
        review_id=uuid.uuid4().hex[:12],
        bundle_id="abcdef1234567890",
        scope_paths=("a.md", "b.md"),
        scope_content_sha256=scope_content_sha256,
        round_number=1,
        council_rotation="Z_fresh_eyes",
        council_attendees=("Karpathy", "Contrarian"),
        findings=tuple(findings),
        verdict="PROCEED_WITH_REVISIONS" if findings else "PROCEED",
        counter_before=counter_before,
        counter_after=counter_after,
        reviewed_at_utc="2026-05-17T00:00:00+00:00",
        reviewer_agent="test",
    )


# --- canonical sets pinned ---


def test_axes_pinned_to_eight():
    assert len(CANONICAL_AXES) == 8


def test_assumption_challenge_axis_present():
    assert "assumption_challenge" in CANONICAL_AXES


def test_seal_threshold_three():
    assert SEAL_THRESHOLD == 3


def test_verdicts_canonical():
    assert "PROCEED_WITH_REVISIONS" in VALID_VERDICTS


def test_severities_include_confirms():
    assert "CONFIRMS" in VALID_SEVERITIES


# --- dataclass invariants ---


def test_finding_rejects_bad_axis():
    with pytest.raises(ValueError, match="canonical 8-axis set"):
        ReviewFinding(
            finding_id="C-1",
            axis="bogus_axis",
            severity="CRITICAL",
            member="x",
            description="x",
            recommended_fix="y",
        )


def test_finding_rejects_bad_severity():
    with pytest.raises(ValueError):
        ReviewFinding(
            finding_id="C-1",
            axis="call_sites",
            severity="EXTREME",
            member="x",
            description="x",
            recommended_fix="y",
        )


def test_finding_rejects_empty_description():
    with pytest.raises(ValueError):
        ReviewFinding(
            finding_id="C-1",
            axis="call_sites",
            severity="CRITICAL",
            member="x",
            description="",
            recommended_fix="y",
        )


def test_round_rejects_bad_verdict():
    with pytest.raises(ValueError, match="verdict"):
        RecursiveReviewRound(
            review_id="x",
            bundle_id="x",
            scope_paths=("a.md",),
            scope_content_sha256="0" * 64,
            round_number=1,
            council_rotation="Z",
            council_attendees=("x",),
            findings=(),
            verdict="MAYBE",
            counter_before=0,
            counter_after=1,
            reviewed_at_utc="2026-05-17",
            reviewer_agent="x",
        )


def test_round_clean_pass_advances_counter():
    rec = _round(counter_before=2, counter_after=3, findings=())
    assert rec.counter_after == 3


def test_round_with_critical_must_reset_counter():
    rec = _round(counter_before=2, counter_after=0, findings=(_finding(),))
    assert rec.counter_after == 0


def test_round_with_findings_but_nonzero_counter_rejected():
    with pytest.raises(ValueError, match="reset to 0"):
        _round(counter_before=2, counter_after=3, findings=(_finding(),))


def test_round_clean_but_counter_not_advanced_rejected():
    with pytest.raises(ValueError, match="counter_before \\+ 1"):
        _round(counter_before=2, counter_after=2, findings=())


def test_confirms_only_round_is_clean():
    rec = _round(counter_before=0, counter_after=1, findings=(_finding(sev="CONFIRMS"),))
    assert rec.counter_after == 1


# --- bundle id determinism ---


def test_bundle_id_deterministic():
    a = compute_bundle_id(["a.md", "b.md", "c.md"])
    b = compute_bundle_id(["c.md", "a.md", "b.md"])
    assert a == b  # sort-invariant


def test_bundle_id_distinct_on_path_change():
    a = compute_bundle_id(["a.md", "b.md"])
    b = compute_bundle_id(["a.md", "b.md", "c.md"])
    assert a != b


def test_scope_content_sha_changes_on_content_change(tmp_path):
    (tmp_path / "a.md").write_text("hello")
    sha1 = compute_scope_content_sha256(["a.md"], repo_root=tmp_path)
    (tmp_path / "a.md").write_text("world")
    sha2 = compute_scope_content_sha256(["a.md"], repo_root=tmp_path)
    assert sha1 != sha2


def test_scope_content_sha_handles_missing(tmp_path):
    sha = compute_scope_content_sha256(["does_not_exist.md"], repo_root=tmp_path)
    assert len(sha) == 64  # still a sha; encodes MISSING sentinel


# --- ledger roundtrip ---


def test_append_and_load_roundtrip(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rec = _round(counter_before=0, counter_after=1, findings=())
    append_round_locked(rec, path=ledger, lock_path=tmp_path / ".lock")
    rows = load_rounds_lenient(path=ledger)
    assert len(rows) == 1
    assert rows[0]["bundle_id"] == "abcdef1234567890"
    assert rows[0]["counter_after"] == 1
    assert "written_at_utc" in rows[0]
    assert "schema_version" in rows[0]


def test_append_preserves_byte_stable_json(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rec = _round(counter_before=0, counter_after=1, findings=())
    append_round_locked(rec, path=ledger, lock_path=tmp_path / ".lock")
    text = ledger.read_text()
    # sort_keys=True ensures determinism
    parsed = json.loads(text.strip())
    assert json.dumps(parsed, sort_keys=True) + "\n" == text


# --- counter queries ---


def test_clean_pass_counter_empty_ledger(tmp_path):
    counter = clean_pass_counter_for_bundle("abc", path=tmp_path / "ledger.jsonl")
    assert counter == 0


def test_clean_pass_counter_reads_latest(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / ".lock"
    append_round_locked(_round(counter_before=0, counter_after=1), path=ledger, lock_path=lock)
    append_round_locked(
        RecursiveReviewRound(
            review_id="x",
            bundle_id="abcdef1234567890",
            scope_paths=("a.md", "b.md"),
            scope_content_sha256="0" * 64,
            round_number=2,
            council_rotation="Y",
            council_attendees=("x",),
            findings=(),
            verdict="PROCEED",
            counter_before=1,
            counter_after=2,
            reviewed_at_utc="2026-05-17T01:00:00+00:00",
            reviewer_agent="test",
        ),
        path=ledger,
        lock_path=lock,
    )
    counter = clean_pass_counter_for_bundle("abcdef1234567890", path=ledger)
    assert counter == 2


def test_clean_pass_counter_resets_when_current_content_sha_changes(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_round_locked(
        _round(counter_before=0, counter_after=1, scope_content_sha256="0" * 64),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )

    assert clean_pass_counter_for_bundle("abcdef1234567890", path=ledger) == 1
    assert (
        clean_pass_counter_for_bundle(
            "abcdef1234567890",
            path=ledger,
            scope_content_sha256="1" * 64,
        )
        == 0
    )


def test_append_rejects_stale_counter_after_content_change(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / ".lock"
    append_round_locked(
        _round(counter_before=0, counter_after=1, scope_content_sha256="0" * 64),
        path=ledger,
        lock_path=lock,
    )

    with pytest.raises(ValueError, match="content changed"):
        append_round_locked(
            _round(counter_before=1, counter_after=2, scope_content_sha256="1" * 64),
            path=ledger,
            lock_path=lock,
        )


def test_append_allows_clean_counter_restart_after_content_change(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / ".lock"
    append_round_locked(
        _round(counter_before=0, counter_after=1, scope_content_sha256="0" * 64),
        path=ledger,
        lock_path=lock,
    )
    append_round_locked(
        _round(counter_before=0, counter_after=1, scope_content_sha256="1" * 64),
        path=ledger,
        lock_path=lock,
    )

    rows = load_rounds_lenient(path=ledger)
    assert len(rows) == 2
    assert rows[-1]["scope_content_sha256"] == "1" * 64


def test_query_unresolved_critical_findings(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_round_locked(
        _round(counter_before=0, counter_after=0, findings=(_finding(),)),
        path=ledger,
        lock_path=tmp_path / ".lock",
    )
    critical = query_unresolved_critical_findings("abcdef1234567890", path=ledger)
    assert len(critical) == 1


def test_sealed_bundle_returns_no_unresolved(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / ".lock"
    # 3 clean rounds => SEALED
    for i in range(3):
        append_round_locked(
            RecursiveReviewRound(
                review_id=f"r{i}",
                bundle_id="abcdef1234567890",
                scope_paths=("a.md", "b.md"),
                scope_content_sha256="0" * 64,
                round_number=i + 1,
                council_rotation="Z",
                council_attendees=("x",),
                findings=(),
                verdict="PROCEED",
                counter_before=i,
                counter_after=i + 1,
                reviewed_at_utc=f"2026-05-17T0{i}:00:00+00:00",
                reviewer_agent="test",
            ),
            path=ledger,
            lock_path=lock,
        )
    assert clean_pass_counter_for_bundle("abcdef1234567890", path=ledger) == 3
    assert query_unresolved_critical_findings("abcdef1234567890", path=ledger) == []


# --- strict load corruption raise ---


def test_strict_load_quarantines_malformed_json(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("not-json\n")
    with pytest.raises(RecursiveReviewLedgerCorruptError):
        load_rounds_strict(path=ledger)
    # quarantined; original moved
    assert not ledger.exists()


def test_strict_load_handles_empty(tmp_path):
    rows = load_rounds_strict(path=tmp_path / "missing.jsonl")
    assert rows == []


# --- Catalog #265 canonical-contract alias ---


def test_update_from_anchor_persists(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    anchor = {
        "bundle_id": "abcdef1234567890",
        "scope_paths": ["a.md", "b.md"],
        "scope_content_sha256": "0" * 64,
        "round_number": 1,
        "council_rotation": "Z",
        "council_attendees": ["x"],
        "findings": [],
        "verdict": "PROCEED",
        "counter_before": 0,
        "counter_after": 1,
    }
    update_from_anchor(anchor, path=ledger)
    assert len(load_rounds_lenient(path=ledger)) == 1
