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
    CouncilTier,
    SCHEMA_VERSION,
    VALID_TIERS,
    VALID_VERDICTS,
    append_council_anchor,
    load_council_anchors,
    load_council_anchors_strict,
    query_anchors_by_topic,
    query_assumption_classification_history,
    query_dissent_history,
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
) -> CouncilDeliberationRecord:
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
