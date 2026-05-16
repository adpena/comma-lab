# SPDX-License-Identifier: MIT
"""Tests for the council continual-learning canonical helper (Catalog #300 sister).

Mirrors the test pattern of:
* :mod:`src.tac.tests.test_modal_call_id_ledger` (Catalog #245 sister)
* :mod:`src.tac.tests.test_codex_round78_check_131_harden_lowercase_and_atomic_replace`
  (Catalog #131 sister)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from tac.council_continual_learning import (
    CouncilDeliberationRecord,
    CouncilPosteriorCorruptError,
    CouncilRecordValidationError,
    CouncilTier,
    DEFERRED_RETROSPECTIVE_WINDOW_DAYS,
    RIGOR_DOMINANT_THRESHOLD,
    SCHEMA_VERSION,
    VALID_MISSION_CONTRIBUTIONS,
    VALID_TIERS,
    VALID_VERDICTS,
    append_council_anchor,
    compute_deferred_retrospective_due_utc,
    is_rigor_dominant,
    load_council_anchors,
    load_council_anchors_strict,
    query_anchors_by_topic,
    query_assumption_classification_history,
    query_dissent_history,
    query_due_retrospectives,
    query_mission_contribution_distribution,
    query_overrides,
    update_from_anchor,
)


def _record(
    *,
    deliberation_id: str = "test_deliberation_001",
    topic: str = "test topic",
    tier: str = CouncilTier.T2,
    attendees=("Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"),
    quorum: bool = True,
    verdict: str = "PROCEED",
    dissent=(),
    adv_verdict=({"assumption": "EMA decay 0.997", "classification": "HARD-EARNED", "rationale": "PR101 empirical"},),
    decisions=("op-routable #1",),
    related=(),
    # Mission-alignment fields (operator binding directive 2026-05-16):
    # required at T2+ per CLAUDE.md "Mission alignment — non-negotiable"
    # operational consequence 5. Default to apparatus_maintenance (the
    # most common historical pattern; safe default for fixture rows that
    # don't otherwise care about the value).
    predicted_mission_contribution: str | None = "apparatus_maintenance",
    override_invoked: bool = False,
    override_rationale: str | None = None,
    deferred_substrate_retrospective_due_utc: str | None = None,
    deferred_substrate_id: str | None = None,
) -> CouncilDeliberationRecord:
    # T1 fixtures default to no mission contribution (T1 is exempt). If the
    # caller passes tier=T1 with the default predicted_mission_contribution,
    # silently clear it so the fixture matches the T1 contract.
    effective_mission = predicted_mission_contribution
    if tier == CouncilTier.T1 and predicted_mission_contribution == "apparatus_maintenance":
        effective_mission = None
    return CouncilDeliberationRecord(
        deliberation_id=deliberation_id,
        topic=topic,
        council_tier=tier,
        council_attendees=tuple(attendees),
        council_quorum_met=quorum,
        council_verdict=verdict,
        council_dissent=tuple(dissent),
        council_assumption_adversary_verdict=tuple(adv_verdict),
        council_decisions_recorded=tuple(decisions),
        related_deliberation_ids=tuple(related),
        predicted_mission_contribution=effective_mission,
        override_invoked=override_invoked,
        override_rationale=override_rationale,
        deferred_substrate_retrospective_due_utc=deferred_substrate_retrospective_due_utc,
        deferred_substrate_id=deferred_substrate_id,
    )


# ─────────────────────── Schema validation ────────────────────────────


def test_schema_version_canonical():
    assert SCHEMA_VERSION == "council_deliberation_posterior_v1"


def test_valid_tiers_canonical_set():
    assert VALID_TIERS == frozenset({"T1", "T2", "T3", "T4"})


def test_valid_verdicts_includes_canonical_outcomes():
    assert "PROCEED" in VALID_VERDICTS
    assert "PROCEED_WITH_REVISIONS" in VALID_VERDICTS
    assert "DEFER_PENDING_EVIDENCE" in VALID_VERDICTS
    assert "REFUSE" in VALID_VERDICTS
    assert "ESCALATE_TO_OPERATOR" in VALID_VERDICTS
    assert "ESCALATE_TO_HIGHER_TIER" in VALID_VERDICTS


# ─────────────────────── append_council_anchor ────────────────────────────


def test_append_writes_row(tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    lock = tmp_path / ".council.lock"
    append_council_anchor(_record(), posterior_path=posterior, lock_path=lock)
    assert posterior.exists()
    rows = posterior.read_text().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["deliberation_id"] == "test_deliberation_001"
    assert payload["council_tier"] == "T2"
    assert payload["schema"] == SCHEMA_VERSION


def test_append_persists_attendees_and_dissent(tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    lock = tmp_path / ".council.lock"
    record = _record(
        attendees=("Shannon", "Dykstra"),
        dissent=({"member": "Dykstra", "verbatim": "feasibility region too narrow"},),
    )
    append_council_anchor(record, posterior_path=posterior, lock_path=lock)
    rows = posterior.read_text().splitlines()
    payload = json.loads(rows[0])
    assert payload["council_attendees"] == ["Shannon", "Dykstra"]
    assert payload["council_dissent"][0]["member"] == "Dykstra"


def test_append_rejects_empty_deliberation_id(tmp_path: Path):
    with pytest.raises(ValueError, match="deliberation_id"):
        append_council_anchor(
            _record(deliberation_id=""),
            posterior_path=tmp_path / "c.jsonl",
            lock_path=tmp_path / ".c.lock",
        )


def test_append_rejects_newline_in_deliberation_id(tmp_path: Path):
    with pytest.raises(ValueError, match="newlines"):
        append_council_anchor(
            _record(deliberation_id="bad\nid"),
            posterior_path=tmp_path / "c.jsonl",
            lock_path=tmp_path / ".c.lock",
        )


def test_append_rejects_invalid_tier(tmp_path: Path):
    with pytest.raises(ValueError, match="council_tier"):
        append_council_anchor(
            _record(tier="T5"),
            posterior_path=tmp_path / "c.jsonl",
            lock_path=tmp_path / ".c.lock",
        )


def test_append_rejects_invalid_verdict(tmp_path: Path):
    with pytest.raises(ValueError, match="council_verdict"):
        append_council_anchor(
            _record(verdict="MAYBE"),
            posterior_path=tmp_path / "c.jsonl",
            lock_path=tmp_path / ".c.lock",
        )


def test_append_rejects_empty_attendees(tmp_path: Path):
    with pytest.raises(ValueError, match="attendees"):
        append_council_anchor(
            _record(attendees=()),
            posterior_path=tmp_path / "c.jsonl",
            lock_path=tmp_path / ".c.lock",
        )


def test_append_t2_requires_assumption_adversary_verdict(tmp_path: Path):
    with pytest.raises(ValueError, match="assumption_adversary"):
        append_council_anchor(
            _record(tier=CouncilTier.T2, adv_verdict=()),
            posterior_path=tmp_path / "c.jsonl",
            lock_path=tmp_path / ".c.lock",
        )


def test_append_t1_allows_empty_assumption_adversary_verdict(tmp_path: Path):
    # T1 working groups may skip the Assumption-Adversary block (T1 is
    # bounded-scope recommendation, not binding decision).
    posterior = tmp_path / "c.jsonl"
    append_council_anchor(
        _record(tier=CouncilTier.T1, adv_verdict=()),
        posterior_path=posterior,
        lock_path=tmp_path / ".c.lock",
    )
    assert posterior.exists()


def test_append_multiple_rows(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    for i in range(5):
        append_council_anchor(
            _record(deliberation_id=f"delib_{i:03d}"),
            posterior_path=posterior,
            lock_path=lock,
        )
    rows = posterior.read_text().splitlines()
    assert len(rows) == 5


# ─────────────────────── load (lenient) ────────────────────────────


def test_load_empty_when_missing(tmp_path: Path):
    posterior = tmp_path / "missing.jsonl"
    assert load_council_anchors(posterior_path=posterior) == []


def test_load_round_trip(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    record = _record(deliberation_id="round_trip_001")
    append_council_anchor(record, posterior_path=posterior, lock_path=tmp_path / ".c.lock")
    loaded = load_council_anchors(posterior_path=posterior)
    assert len(loaded) == 1
    assert loaded[0].deliberation_id == "round_trip_001"
    assert loaded[0].council_tier == "T2"


def test_load_skips_malformed_line(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    posterior.write_text("not_json\n" + json.dumps({
        "schema": SCHEMA_VERSION,
        "deliberation_id": "good_one",
        "topic": "t",
        "council_tier": "T2",
        "council_attendees": ["S"],
        "council_quorum_met": True,
        "council_verdict": "PROCEED",
        "council_dissent": [],
        "council_assumption_adversary_verdict": [{"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"}],
        "council_decisions_recorded": [],
        "related_deliberation_ids": [],
        "event_type": "dispatched",
    }) + "\n")
    loaded = load_council_anchors(posterior_path=posterior)
    assert len(loaded) == 1
    assert loaded[0].deliberation_id == "good_one"


def test_load_skips_wrong_schema(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    posterior.write_text(json.dumps({"schema": "v_other", "deliberation_id": "x"}) + "\n")
    assert load_council_anchors(posterior_path=posterior) == []


# ─────────────────────── load_strict ────────────────────────────


def test_load_strict_empty_when_missing(tmp_path: Path):
    posterior = tmp_path / "missing.jsonl"
    assert load_council_anchors_strict(posterior_path=posterior) == []


def test_load_strict_raises_on_malformed(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    posterior.write_text("not_json\n")
    with pytest.raises(CouncilPosteriorCorruptError):
        load_council_anchors_strict(posterior_path=posterior)


def test_load_strict_raises_on_non_dict_root(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    posterior.write_text("[1, 2, 3]\n")
    with pytest.raises(CouncilPosteriorCorruptError):
        load_council_anchors_strict(posterior_path=posterior)


def test_load_strict_clean_round_trip(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    append_council_anchor(_record(), posterior_path=posterior, lock_path=tmp_path / ".c.lock")
    assert len(load_council_anchors_strict(posterior_path=posterior)) == 1


# ─────────────────────── query helpers ────────────────────────────


def test_query_by_topic(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    append_council_anchor(
        _record(deliberation_id="d1", topic="hierarchy v2 landing"),
        posterior_path=posterior, lock_path=lock,
    )
    append_council_anchor(
        _record(deliberation_id="d2", topic="unrelated work"),
        posterior_path=posterior, lock_path=lock,
    )
    hits = query_anchors_by_topic("hierarchy", posterior_path=posterior)
    assert len(hits) == 1
    assert hits[0].deliberation_id == "d1"


def test_query_by_topic_case_insensitive(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    append_council_anchor(
        _record(topic="UPPER CASE TOPIC"),
        posterior_path=posterior, lock_path=tmp_path / ".c.lock",
    )
    assert len(query_anchors_by_topic("upper case", posterior_path=posterior)) == 1


def test_query_dissent_history(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    append_council_anchor(
        _record(
            deliberation_id="d1",
            dissent=({"member": "Contrarian", "verbatim": "this is lazy consensus"},),
        ),
        posterior_path=posterior, lock_path=lock,
    )
    append_council_anchor(
        _record(
            deliberation_id="d2",
            dissent=({"member": "Contrarian", "verbatim": "argument lacks rigor"},),
        ),
        posterior_path=posterior, lock_path=lock,
    )
    hits = query_dissent_history("Contrarian", posterior_path=posterior)
    assert len(hits) == 2
    assert all("Contrarian" in str(d.get("member", "")) for _, d in hits)


def test_query_assumption_classification_history(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    append_council_anchor(
        _record(
            deliberation_id="d1",
            adv_verdict=({"assumption": "EMA decay 0.997", "classification": "HARD-EARNED", "rationale": "PR101"},),
        ),
        posterior_path=posterior, lock_path=lock,
    )
    append_council_anchor(
        _record(
            deliberation_id="d2",
            adv_verdict=({"assumption": "EMA decay 0.997", "classification": "HARD-EARNED", "rationale": "Quantizr"},),
        ),
        posterior_path=posterior, lock_path=lock,
    )
    hits = query_assumption_classification_history("EMA decay", posterior_path=posterior)
    assert len(hits) == 2


def test_query_empty_substring_returns_empty(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    append_council_anchor(_record(), posterior_path=posterior, lock_path=tmp_path / ".c.lock")
    assert query_anchors_by_topic("", posterior_path=posterior) == []
    assert query_dissent_history("", posterior_path=posterior) == []
    assert query_assumption_classification_history("", posterior_path=posterior) == []


# ─────────────────────── update_from_anchor alias ────────────────────────────


def test_update_from_anchor_aliases_append(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    update_from_anchor(_record(deliberation_id="alias_001"), posterior_path=posterior, lock_path=tmp_path / ".c.lock")
    assert posterior.exists()
    loaded = load_council_anchors(posterior_path=posterior)
    assert len(loaded) == 1
    assert loaded[0].deliberation_id == "alias_001"


# ─────────────────────── 4-process concurrent append ────────────────────────────


def _worker_append(args):
    posterior_path, lock_path, worker_id = args
    from tac.council_continual_learning import (
        CouncilDeliberationRecord,
        CouncilTier,
        append_council_anchor,
    )

    for i in range(5):
        record = CouncilDeliberationRecord(
            deliberation_id=f"worker{worker_id}_row{i}",
            topic=f"worker {worker_id} row {i}",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon", "Dykstra"),
            council_quorum_met=True,
            council_verdict="PROCEED",
            council_assumption_adversary_verdict=(
                {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
            ),
            # Required at T2+ per mission-alignment binding directive 2026-05-16.
            predicted_mission_contribution="apparatus_maintenance",
        )
        append_council_anchor(
            record,
            posterior_path=Path(posterior_path),
            lock_path=Path(lock_path),
        )
    return worker_id


def test_4_proc_spawn_pool_concurrent_append(tmp_path: Path):
    posterior = tmp_path / "concurrent.jsonl"
    lock = tmp_path / ".concurrent.lock"
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(
            _worker_append,
            [(str(posterior), str(lock), w) for w in range(4)],
        )
    assert sorted(results) == [0, 1, 2, 3]
    rows = posterior.read_text().splitlines()
    # Each of 4 workers wrote 5 rows = 20 total; lock guarantees no torn writes.
    assert len(rows) == 20
    # Every row must be valid JSON with a unique deliberation_id.
    ids = set()
    for r in rows:
        payload = json.loads(r)
        ids.add(payload["deliberation_id"])
    assert len(ids) == 20


# ───────────────────────────────────────────────────────────────────
# Mission-alignment tests (operator binding directive 2026-05-16).
# Per CLAUDE.md "Mission alignment — non-negotiable" subsection of
# "Council hierarchy: 4-tier protocol".
# ───────────────────────────────────────────────────────────────────


def test_valid_mission_contributions_canonical_set():
    assert VALID_MISSION_CONTRIBUTIONS == frozenset({
        "frontier_breaking",
        "frontier_protecting",
        "rigor_overhead",
        "apparatus_maintenance",
        "mission_questioned",
    })


def test_rigor_dominant_threshold_is_60_percent():
    assert RIGOR_DOMINANT_THRESHOLD == 0.60


def test_deferred_retrospective_window_is_30_days():
    assert DEFERRED_RETROSPECTIVE_WINDOW_DAYS == 30


def test_t2_record_requires_predicted_mission_contribution():
    with pytest.raises(CouncilRecordValidationError, match="predicted_mission_contribution"):
        CouncilDeliberationRecord(
            deliberation_id="t2_no_mission",
            topic="t",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon",),
            council_quorum_met=True,
            council_verdict="PROCEED",
            council_assumption_adversary_verdict=(
                {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
            ),
            # predicted_mission_contribution intentionally None — must raise.
        )


def test_t1_record_allows_none_predicted_mission_contribution():
    # T1 working groups don't make binding decisions; mission forecast is exempt.
    rec = CouncilDeliberationRecord(
        deliberation_id="t1_no_mission",
        topic="t",
        council_tier=CouncilTier.T1,
        council_attendees=("Shannon",),
        council_quorum_met=True,
        council_verdict="PROCEED",
    )
    assert rec.predicted_mission_contribution is None


def test_record_rejects_invalid_mission_contribution():
    with pytest.raises(CouncilRecordValidationError, match="predicted_mission_contribution"):
        CouncilDeliberationRecord(
            deliberation_id="bad_mission",
            topic="t",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon",),
            council_quorum_met=True,
            council_verdict="PROCEED",
            council_assumption_adversary_verdict=(
                {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
            ),
            predicted_mission_contribution="bogus_category",
        )


def test_override_invoked_true_requires_rationale():
    with pytest.raises(CouncilRecordValidationError, match="override_rationale"):
        CouncilDeliberationRecord(
            deliberation_id="bare_override",
            topic="t",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon",),
            council_quorum_met=True,
            council_verdict="PROCEED",
            council_assumption_adversary_verdict=(
                {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
            ),
            predicted_mission_contribution="frontier_breaking",
            override_invoked=True,
            # override_rationale intentionally None — must raise.
        )


def test_override_with_empty_rationale_rejected():
    with pytest.raises(CouncilRecordValidationError, match="override_rationale"):
        CouncilDeliberationRecord(
            deliberation_id="empty_override",
            topic="t",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon",),
            council_quorum_met=True,
            council_verdict="PROCEED",
            council_assumption_adversary_verdict=(
                {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
            ),
            predicted_mission_contribution="frontier_breaking",
            override_invoked=True,
            override_rationale="   ",   # whitespace-only
        )


def test_override_with_rationale_constructs():
    rec = CouncilDeliberationRecord(
        deliberation_id="good_override",
        topic="t",
        council_tier=CouncilTier.T2,
        council_attendees=("Shannon",),
        council_quorum_met=True,
        council_verdict="PROCEED",
        council_assumption_adversary_verdict=(
            {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
        ),
        predicted_mission_contribution="frontier_breaking",
        override_invoked=True,
        override_rationale="operator verbatim: leaderboard moved, skip sextet pact for this dispatch",
    )
    assert rec.override_invoked is True
    assert "leaderboard" in (rec.override_rationale or "")


def test_deferred_retrospective_field_pairing_required():
    with pytest.raises(CouncilRecordValidationError, match="deferred_substrate_id"):
        CouncilDeliberationRecord(
            deliberation_id="unpaired_retro",
            topic="t",
            council_tier=CouncilTier.T2,
            council_attendees=("Shannon",),
            council_quorum_met=True,
            council_verdict="DEFER_PENDING_EVIDENCE",
            council_assumption_adversary_verdict=(
                {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
            ),
            predicted_mission_contribution="frontier_protecting",
            deferred_substrate_retrospective_due_utc="2026-06-15T15:00:00+00:00",
            # deferred_substrate_id intentionally None — must raise.
        )


def test_deferred_retrospective_paired_fields_construct():
    rec = CouncilDeliberationRecord(
        deliberation_id="paired_retro",
        topic="t",
        council_tier=CouncilTier.T2,
        council_attendees=("Shannon",),
        council_quorum_met=True,
        council_verdict="DEFER_PENDING_EVIDENCE",
        council_assumption_adversary_verdict=(
            {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
        ),
        predicted_mission_contribution="frontier_protecting",
        deferred_substrate_retrospective_due_utc="2026-06-15T15:00:00+00:00",
        deferred_substrate_id="lane_some_deferred_substrate_20260516",
    )
    assert rec.deferred_substrate_id is not None
    assert rec.deferred_substrate_retrospective_due_utc is not None


def test_compute_deferred_retrospective_due_utc_adds_30_days():
    base = "2026-05-16T15:00:00+00:00"
    due = compute_deferred_retrospective_due_utc(base)
    # 30 days later -> 2026-06-15T15:00:00+00:00
    assert due.startswith("2026-06-15T15:00:00")


def test_compute_deferred_retrospective_due_utc_default_uses_now():
    due = compute_deferred_retrospective_due_utc()
    # Should not raise; should be ~30 days in the future.
    import datetime as _dt
    parsed = _dt.datetime.fromisoformat(due.replace("Z", "+00:00"))
    now = _dt.datetime.now(_dt.UTC)
    delta = parsed - now
    assert _dt.timedelta(days=29, hours=23) < delta < _dt.timedelta(days=30, hours=1)


def test_append_persists_mission_alignment_fields(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    rec = _record(
        deliberation_id="mission_persist",
        predicted_mission_contribution="frontier_breaking",
        override_invoked=True,
        override_rationale="operator verbatim test",
        deferred_substrate_retrospective_due_utc="2026-06-15T15:00:00+00:00",
        deferred_substrate_id="lane_some_substrate",
    )
    append_council_anchor(rec, posterior_path=posterior, lock_path=lock)
    loaded = load_council_anchors(posterior_path=posterior)
    assert len(loaded) == 1
    a = loaded[0]
    assert a.predicted_mission_contribution == "frontier_breaking"
    assert a.override_invoked is True
    assert a.override_rationale == "operator verbatim test"
    assert a.deferred_substrate_id == "lane_some_substrate"
    assert a.deferred_substrate_retrospective_due_utc == "2026-06-15T15:00:00+00:00"


def test_legacy_row_loads_with_apparatus_maintenance_backfill_default(tmp_path: Path):
    """Legacy rows (pre-mission-alignment) lack the field; loader backfills T2+."""
    posterior = tmp_path / "c.jsonl"
    legacy = {
        "schema": SCHEMA_VERSION,
        "deliberation_id": "legacy_one",
        "topic": "legacy",
        "council_tier": "T3",
        "council_attendees": ["S"],
        "council_quorum_met": True,
        "council_verdict": "PROCEED",
        "council_dissent": [],
        "council_assumption_adversary_verdict": [
            {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
        ],
        "council_decisions_recorded": [],
        "related_deliberation_ids": [],
        "event_type": "dispatched",
        "written_at_utc": "2026-05-15T00:00:00+00:00",
        # No predicted_mission_contribution / override_invoked / override_rationale.
    }
    posterior.write_text(json.dumps(legacy) + "\n")
    loaded = load_council_anchors(posterior_path=posterior)
    assert len(loaded) == 1
    # Backfill default for T2+ legacy rows: apparatus_maintenance.
    assert loaded[0].predicted_mission_contribution == "apparatus_maintenance"
    assert loaded[0].override_invoked is False
    assert loaded[0].override_rationale is None


def test_query_overrides_returns_only_overrides(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    append_council_anchor(
        _record(deliberation_id="no_override", override_invoked=False),
        posterior_path=posterior, lock_path=lock,
    )
    append_council_anchor(
        _record(
            deliberation_id="yes_override",
            override_invoked=True,
            override_rationale="operator verbatim",
        ),
        posterior_path=posterior, lock_path=lock,
    )
    hits = query_overrides(posterior_path=posterior)
    assert len(hits) == 1
    assert hits[0].deliberation_id == "yes_override"


def test_query_overrides_since_utc_filters(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    # Write an old override and a new override.
    import datetime as _dt
    old_ts = "2026-04-01T00:00:00+00:00"
    new_ts = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    payload_old = {
        "schema": SCHEMA_VERSION,
        "deliberation_id": "old_override",
        "topic": "t",
        "council_tier": "T2",
        "council_attendees": ["S"],
        "council_quorum_met": True,
        "council_verdict": "PROCEED",
        "council_dissent": [],
        "council_assumption_adversary_verdict": [
            {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"},
        ],
        "council_decisions_recorded": [],
        "related_deliberation_ids": [],
        "event_type": "dispatched",
        "predicted_mission_contribution": "frontier_breaking",
        "override_invoked": True,
        "override_rationale": "old",
        "written_at_utc": old_ts,
    }
    posterior.write_text(json.dumps(payload_old) + "\n")
    append_council_anchor(
        _record(
            deliberation_id="new_override",
            override_invoked=True,
            override_rationale="new",
        ),
        posterior_path=posterior, lock_path=lock,
    )
    hits = query_overrides(since_utc="2026-05-01T00:00:00+00:00", posterior_path=posterior)
    # Only new_override should match the cutoff.
    ids = {h.deliberation_id for h in hits}
    assert "new_override" in ids
    assert "old_override" not in ids


def test_query_due_retrospectives_returns_overdue(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    # Write a deliberation whose retrospective is due in the past.
    past_due = "2026-04-15T00:00:00+00:00"
    rec = _record(
        deliberation_id="overdue_substrate",
        verdict="DEFER_PENDING_EVIDENCE",
        deferred_substrate_retrospective_due_utc=past_due,
        deferred_substrate_id="lane_some_deferred_substrate",
    )
    append_council_anchor(rec, posterior_path=posterior, lock_path=lock)
    # Write a deliberation whose retrospective is due in the future.
    future_due = "2027-05-16T00:00:00+00:00"
    rec_future = _record(
        deliberation_id="future_substrate",
        verdict="DEFER_PENDING_EVIDENCE",
        deferred_substrate_retrospective_due_utc=future_due,
        deferred_substrate_id="lane_future_deferred_substrate",
    )
    append_council_anchor(rec_future, posterior_path=posterior, lock_path=lock)
    due = query_due_retrospectives(
        as_of_utc="2026-05-16T00:00:00+00:00",
        posterior_path=posterior,
    )
    ids = {h.deliberation_id for h in due}
    assert "overdue_substrate" in ids
    assert "future_substrate" not in ids


def test_query_mission_contribution_distribution(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    append_council_anchor(
        _record(deliberation_id="d1", predicted_mission_contribution="frontier_breaking"),
        posterior_path=posterior, lock_path=lock,
    )
    append_council_anchor(
        _record(deliberation_id="d2", predicted_mission_contribution="frontier_breaking"),
        posterior_path=posterior, lock_path=lock,
    )
    append_council_anchor(
        _record(deliberation_id="d3", predicted_mission_contribution="apparatus_maintenance"),
        posterior_path=posterior, lock_path=lock,
    )
    dist = query_mission_contribution_distribution(posterior_path=posterior)
    assert dist["frontier_breaking"] == 2
    assert dist["apparatus_maintenance"] == 1
    assert dist["mission_questioned"] == 0


def test_is_rigor_dominant_true_when_overhead_exceeds_threshold(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    # 7 apparatus_maintenance + 3 frontier_breaking = 70% rigor → True.
    for i in range(7):
        append_council_anchor(
            _record(
                deliberation_id=f"app_{i}",
                predicted_mission_contribution="apparatus_maintenance",
            ),
            posterior_path=posterior, lock_path=lock,
        )
    for i in range(3):
        append_council_anchor(
            _record(
                deliberation_id=f"fb_{i}",
                predicted_mission_contribution="frontier_breaking",
            ),
            posterior_path=posterior, lock_path=lock,
        )
    assert is_rigor_dominant(posterior_path=posterior) is True


def test_is_rigor_dominant_false_when_frontier_breaking_dominates(tmp_path: Path):
    posterior = tmp_path / "c.jsonl"
    lock = tmp_path / ".c.lock"
    # 2 apparatus_maintenance + 8 frontier_breaking = 20% rigor → False.
    for i in range(2):
        append_council_anchor(
            _record(
                deliberation_id=f"app_{i}",
                predicted_mission_contribution="apparatus_maintenance",
            ),
            posterior_path=posterior, lock_path=lock,
        )
    for i in range(8):
        append_council_anchor(
            _record(
                deliberation_id=f"fb_{i}",
                predicted_mission_contribution="frontier_breaking",
            ),
            posterior_path=posterior, lock_path=lock,
        )
    assert is_rigor_dominant(posterior_path=posterior) is False


def test_is_rigor_dominant_false_when_no_anchors(tmp_path: Path):
    posterior = tmp_path / "missing.jsonl"
    # No file → no data → False (not rigor-dominant by absence).
    assert is_rigor_dominant(posterior_path=posterior) is False
