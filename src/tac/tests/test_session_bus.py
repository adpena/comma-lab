# SPDX-License-Identifier: MIT
"""Tests for the session-bus coordination pair (task #388).

Coverage: locked concurrent appends (two real processes) · events_since filtering (offset +
timestamp + kinds) · staleness_check hit/miss/payload-subjects/since · partial-line JSONL
corruption tolerance · recovery report on synthetic progress rows · register/heartbeat/
complete round-trip · fail-open bulletin producer in review_counter · CLI smoke.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tac.session_bus import bulletin as B
from tac.session_bus import recovery_manifest as rm

_SRC_DIR = Path(B.__file__).resolve().parents[2]  # .../src


@pytest.fixture()
def bpaths(tmp_path):
    return tmp_path / "session_events.jsonl", tmp_path / ".session_events.jsonl.lock"


# ─────────────────────────── bulletin: post + validate ───────────────────────────


def test_post_and_read_roundtrip(bpaths):
    path, lock = bpaths
    rec = B.post_event(
        B.EVENT_GATE_RULED, "#386", {"verdict": "RULED"}, "agentA",
        bulletin_path=path, lock_path=lock,
    )
    assert rec["kind"] == B.EVENT_GATE_RULED
    assert rec["subject"] == "#386"
    rows = B.read_events(bulletin_path=path)
    assert len(rows) == 1
    assert rows[0]["payload"] == {"verdict": "RULED"}
    assert rows[0]["schema"] == B.SCHEMA_VERSION


def test_post_event_rejects_bad_kind(bpaths):
    path, lock = bpaths
    with pytest.raises(B.SessionBusError):
        B.post_event("not_a_kind", "s", {}, "a", bulletin_path=path, lock_path=lock)


def test_post_event_rejects_empty_subject(bpaths):
    path, lock = bpaths
    with pytest.raises(B.SessionBusError):
        B.post_event(B.EVENT_MEMO_LANDED, "  ", {}, "a", bulletin_path=path, lock_path=lock)


def test_post_event_rejects_non_dict_payload(bpaths):
    path, lock = bpaths
    with pytest.raises(B.SessionBusError):
        B.post_event(
            B.EVENT_MEMO_LANDED, "s", ["not", "a", "dict"], "a",
            bulletin_path=path, lock_path=lock,
        )


def test_post_event_rejects_non_serializable_payload(bpaths):
    path, lock = bpaths
    with pytest.raises(B.SessionBusError):
        B.post_event(
            B.EVENT_MEMO_LANDED, "s", {"bad": {1, 2, 3}}, "a",
            bulletin_path=path, lock_path=lock,
        )


def test_post_event_rejects_empty_agent_label(bpaths):
    path, lock = bpaths
    with pytest.raises(B.SessionBusError):
        B.post_event(B.EVENT_MEMO_LANDED, "s", {}, "", bulletin_path=path, lock_path=lock)


# ─────────────────────────── bulletin: env override + disable (#389 r2) ───────────────────────────


def test_env_path_override_used_when_no_explicit_path(tmp_path, monkeypatch):
    """No explicit ``bulletin_path`` → the env var redirects the default (child-inheritable)."""
    ev = tmp_path / "env_store.jsonl"
    monkeypatch.setenv(B.BULLETIN_PATH_ENV, str(ev))
    monkeypatch.setenv(B.BULLETIN_LOCK_ENV, str(tmp_path / ".env_store.lock"))
    B.post_event(B.EVENT_MEMO_LANDED, "s", {}, "a")  # no explicit path
    assert B.event_count(bulletin_path=ev) == 1


def test_explicit_path_beats_env_override(tmp_path, monkeypatch):
    """Explicit call arg > env override > module default."""
    ev = tmp_path / "env_store.jsonl"
    explicit = tmp_path / "explicit_store.jsonl"
    monkeypatch.setenv(B.BULLETIN_PATH_ENV, str(ev))
    B.post_event(
        B.EVENT_MEMO_LANDED, "s", {}, "a",
        bulletin_path=explicit, lock_path=tmp_path / ".explicit.lock",
    )
    assert B.event_count(bulletin_path=explicit) == 1
    assert not ev.exists()  # env path untouched — explicit won


def test_disable_env_mutes_default_write_but_returns_record(tmp_path, monkeypatch):
    """DISABLE hard-mutes a default/env-path write (child-inheritable) but still validates
    + returns the record; nothing is written to the (monkeypatched) default store."""
    default = tmp_path / "would_be_default.jsonl"
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_PATH", default)
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_LOCK_PATH", tmp_path / ".def.lock")
    monkeypatch.setenv(B.BULLETIN_DISABLE_ENV, "1")
    rec = B.post_event(B.EVENT_MEMO_LANDED, "s", {"k": 1}, "a")  # no explicit path
    assert rec["kind"] == B.EVENT_MEMO_LANDED  # record built + returned
    assert not default.exists()  # muted: nothing written
    # validation still fires even when muted:
    with pytest.raises(B.SessionBusError):
        B.post_event("not_a_kind", "s", {}, "a")


def test_disable_env_does_not_mute_explicit_path(tmp_path, monkeypatch):
    """An explicit destination is honored even under DISABLE — the caller chose it on purpose."""
    explicit = tmp_path / "explicit.jsonl"
    monkeypatch.setenv(B.BULLETIN_DISABLE_ENV, "true")
    B.post_event(
        B.EVENT_MEMO_LANDED, "s", {}, "a",
        bulletin_path=explicit, lock_path=tmp_path / ".ex.lock",
    )
    assert B.event_count(bulletin_path=explicit) == 1


# ─────────────────────────── bulletin: events_since ───────────────────────────


def _seed(path, lock, n):
    for i in range(n):
        B.post_event(
            B.EVENT_SPEC_EDITED, f"subj-{i}", {"i": i}, "seed",
            bulletin_path=path, lock_path=lock,
        )


def test_events_since_none_returns_all(bpaths):
    path, lock = bpaths
    _seed(path, lock, 4)
    assert len(B.events_since(None, bulletin_path=path)) == 4


def test_events_since_int_offset(bpaths):
    path, lock = bpaths
    _seed(path, lock, 5)
    tail = B.events_since(2, bulletin_path=path)
    assert [r["subject"] for r in tail] == ["subj-2", "subj-3", "subj-4"]


def test_events_since_int_offset_negative_raises(bpaths):
    path, lock = bpaths
    _seed(path, lock, 1)
    with pytest.raises(B.SessionBusError):
        B.events_since(-1, bulletin_path=path)


def test_events_since_bool_marker_raises(bpaths):
    path, lock = bpaths
    _seed(path, lock, 1)
    with pytest.raises(B.SessionBusError):
        B.events_since(True, bulletin_path=path)


def test_events_since_timestamp_filter(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_MEMO_LANDED, "old", {}, "a", bulletin_path=path, lock_path=lock)
    rows = B.read_events(bulletin_path=path)
    cutoff = rows[0]["written_at_utc"]
    B.post_event(B.EVENT_MEMO_LANDED, "new", {}, "a", bulletin_path=path, lock_path=lock)
    after = B.events_since(cutoff, bulletin_path=path)
    assert [r["subject"] for r in after] == ["new"]


def test_events_since_kinds_filter(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_GATE_RULED, "g", {}, "a", bulletin_path=path, lock_path=lock)
    B.post_event(B.EVENT_MEMO_LANDED, "m", {}, "a", bulletin_path=path, lock_path=lock)
    only_gate = B.events_since(None, kinds=[B.EVENT_GATE_RULED], bulletin_path=path)
    assert [r["subject"] for r in only_gate] == ["g"]


def test_event_count(bpaths):
    path, lock = bpaths
    _seed(path, lock, 3)
    assert B.event_count(bulletin_path=path) == 3


# ─────────────────────────── bulletin: corruption tolerance ───────────────────────────


def test_read_events_skips_partial_line_corruption(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_MEMO_LANDED, "good", {}, "a", bulletin_path=path, lock_path=lock)
    # Simulate a process killed mid-append: a torn (unterminated, invalid) trailing line.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"schema": "session_bus_event_v1", "kind": "memo_lan')  # no newline, torn
    rows = B.read_events(bulletin_path=path)
    assert len(rows) == 1
    assert rows[0]["subject"] == "good"


def test_read_events_skips_wrong_schema(bpaths):
    path, lock = bpaths
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": "some_other_v9", "kind": "x"}) + "\n")
    B.post_event(B.EVENT_MEMO_LANDED, "good", {}, "a", bulletin_path=path, lock_path=lock)
    rows = B.read_events(bulletin_path=path)
    assert [r["subject"] for r in rows] == ["good"]


# ─────────────────────────── bulletin: staleness_check ───────────────────────────


def test_staleness_check_hit(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_GATE_RULED, "#386", {}, "sibling", bulletin_path=path, lock_path=lock)
    report = B.staleness_check(["b_c", "flip", "#386"], bulletin_path=path)
    assert report.stale is True
    assert len(report.events) == 1
    assert "#386" in report.matched_subjects
    assert "STALE" in report.describe()


def test_staleness_check_miss(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_GATE_RULED, "unrelated_topic", {}, "s", bulletin_path=path, lock_path=lock)
    report = B.staleness_check(["b_c", "#386"], bulletin_path=path)
    assert report.stale is False
    assert report.events == ()
    assert "FRESH" in report.describe()


def test_staleness_check_payload_subjects_match(bpaths):
    path, lock = bpaths
    B.post_event(
        B.EVENT_VERDICT_LANDED, "SYNTHESIS_v3",
        {"subjects": ["b_c", "flip_weighted"]}, "s",
        bulletin_path=path, lock_path=lock,
    )
    report = B.staleness_check(["flip_weighted"], bulletin_path=path)
    assert report.stale is True


def test_staleness_check_since_offset_excludes_earlier(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_GATE_RULED, "#386", {}, "s", bulletin_path=path, lock_path=lock)
    offset = B.event_count(bulletin_path=path)  # I have seen everything up to here
    # No new #386 events after offset:
    B.post_event(B.EVENT_MEMO_LANDED, "other", {}, "s", bulletin_path=path, lock_path=lock)
    report = B.staleness_check(["#386"], since=offset, bulletin_path=path)
    assert report.stale is False
    # But a NEW #386 event after the offset flips it stale:
    B.post_event(B.EVENT_GATE_RULED, "#386", {}, "s", bulletin_path=path, lock_path=lock)
    report2 = B.staleness_check(["#386"], since=offset, bulletin_path=path)
    assert report2.stale is True


def test_staleness_check_empty_subjects_raises(bpaths):
    path, lock = bpaths
    with pytest.raises(B.SessionBusError):
        B.staleness_check([], bulletin_path=path)


def test_staleness_check_kinds_filter(bpaths):
    path, lock = bpaths
    B.post_event(B.EVENT_MEMO_LANDED, "#386", {}, "s", bulletin_path=path, lock_path=lock)
    # A memo about #386 exists, but we only care about gate_ruled events:
    report = B.staleness_check(["#386"], kinds=[B.EVENT_GATE_RULED], bulletin_path=path)
    assert report.stale is False


# ─────────────────────────── bulletin: concurrent locked appends ───────────────────────────

_WORKER_SRC = """
import sys
from pathlib import Path
from tac.session_bus.bulletin import post_event, EVENT_MEMO_LANDED
path, lock, label, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
for i in range(n):
    post_event(EVENT_MEMO_LANDED, label + "-" + str(i), {"i": i}, label,
               bulletin_path=Path(path), lock_path=Path(lock))
"""


def test_concurrent_appends_two_procs(bpaths):
    path, lock = bpaths
    env = {**os.environ, "PYTHONPATH": str(_SRC_DIR)}
    n = 30
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER_SRC, str(path), str(lock), label, str(n)],
            env=env,
        )
        for label in ("procA", "procB")
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0
    rows = B.read_events(bulletin_path=path)
    # No lost rows under contention, and every line is valid JSON (no interleaved/torn write).
    assert len(rows) == 2 * n
    labels = [r["agent_label"] for r in rows]
    assert labels.count("procA") == n
    assert labels.count("procB") == n


# ─────────────────────────── recovery_manifest ───────────────────────────


@pytest.fixture()
def tool_tmp(tmp_path, monkeypatch):
    """Redirect the checkpoint tool's store to a temp file so tests never touch the real one.

    #389: register_inflight / complete also post fail-open lifecycle bulletin events, so
    redirect the bulletin store too — otherwise these tests would pollute the live
    ``.omx/state/session_events.jsonl`` feed that a concurrent seal agent reads.
    """
    tool = rm._checkpoint_tool()
    monkeypatch.setattr(tool, "STATE_DIR", tmp_path)
    monkeypatch.setattr(tool, "JSONL_PATH", tmp_path / "subagent_progress.jsonl")
    monkeypatch.setattr(tool, "LOCK_PATH", tmp_path / ".subagent_progress.lock")
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_PATH", tmp_path / "session_events.jsonl")
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_LOCK_PATH", tmp_path / ".session_events.lock")
    return tmp_path


def test_register_heartbeat_complete_roundtrip(tool_tmp):
    rm.register_inflight(
        "AGENT-X",
        respawn_context="task #388; spec at docs/foo.md; build bar.py",
        expected_outputs=["src/tac/bar.py"],
        files_touched=["src/tac/bar.py"],
        next_action="write tests",
        lane_id="lane_388",
    )
    rm.heartbeat("AGENT-X", 2, files_touched=["src/tac/bar.py"], next_action="ruff")
    # Far-future 'now' → still in-progress and stale → appears in report.
    future = datetime.now(UTC) + timedelta(hours=1)
    entries = rm.recover_report(now=future)
    assert [e.subagent_id for e in entries] == ["AGENT-X"]
    assert entries[0].respawn_context.startswith("task #388")
    rm.complete("AGENT-X", files_touched=["src/tac/bar.py"])
    assert rm.recover_report(now=future) == []


def test_register_inflight_rejects_empty_respawn_context(tool_tmp):
    with pytest.raises(ValueError):
        rm.register_inflight("AGENT-Y", respawn_context="   ")


def _write_progress(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_recover_report_synthetic_rows(tool_tmp):
    tool = rm._checkpoint_tool()
    now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)
    old = (now - timedelta(seconds=1000)).isoformat()
    recent = (now - timedelta(seconds=10)).isoformat()
    rows = [
        {"subagent_id": "DONE", "status": "complete", "step": "complete",
         "written_at_utc": old},
        {"subagent_id": "RECENT", "status": "in_progress", "step": 3,
         "written_at_utc": recent},
        {"subagent_id": "STALE", "status": "in_progress", "step": 5,
         "written_at_utc": old, "respawn_context": "resume me",
         "files_touched": ["a.py"], "next_action": "finish", "expected_outputs": ["a.py"]},
        {"subagent_id": "BADTS", "status": "in_progress", "step": 1,
         "written_at_utc": "not-a-timestamp"},
    ]
    _write_progress(tool.JSONL_PATH, rows)
    entries = rm.recover_report(stale_after_seconds=300, now=now)
    ids = {e.subagent_id for e in entries}
    assert ids == {"STALE", "BADTS"}  # DONE excluded (complete), RECENT excluded (fresh)
    stale = next(e for e in entries if e.subagent_id == "STALE")
    assert stale.respawn_context == "resume me"
    assert stale.expected_outputs == ("a.py",)


def test_recover_report_latest_supersession(tool_tmp):
    tool = rm._checkpoint_tool()
    now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)
    old = (now - timedelta(seconds=1000)).isoformat()
    rows = [
        {"subagent_id": "A", "status": "in_progress", "step": 1, "written_at_utc": old},
        {"subagent_id": "A", "status": "complete", "step": "complete", "written_at_utc": old},
    ]
    _write_progress(tool.JSONL_PATH, rows)
    # Latest record for A is complete → excluded even though an earlier one was in_progress.
    assert rm.recover_report(now=now) == []


def test_recover_report_sort_most_stale_first(tool_tmp):
    tool = rm._checkpoint_tool()
    now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)
    rows = [
        {"subagent_id": "LESS", "status": "in_progress", "step": 1,
         "written_at_utc": (now - timedelta(seconds=400)).isoformat()},
        {"subagent_id": "MORE", "status": "in_progress", "step": 1,
         "written_at_utc": (now - timedelta(seconds=9000)).isoformat()},
    ]
    _write_progress(tool.JSONL_PATH, rows)
    entries = rm.recover_report(stale_after_seconds=300, now=now)
    assert [e.subagent_id for e in entries] == ["MORE", "LESS"]


def test_recovery_entry_render_contains_context(tool_tmp):
    entry = rm.RecoveryEntry(
        subagent_id="AGENT-Z",
        last_step=4,
        last_status="in_progress",
        last_heartbeat_utc="2026-07-09T12:00:00+00:00",
        age_seconds=1234.0,
        respawn_context="PASTE-ME task #388",
        expected_outputs=("out.py",),
        files_touched=("f.py",),
        next_action="finish tests",
        lane_id="lane_388",
    )
    text = entry.render()
    assert "RESPAWN: AGENT-Z" in text
    assert "PASTE-ME task #388" in text
    assert "Expected outputs: out.py" in text
    assert "Next action: finish tests" in text


def test_render_report_empty():
    assert "no in-flight agents" in rm.render_report([])


# ─────────────────────────── review_counter producer wire-in ───────────────────────────


def test_record_round_posts_bulletin_event(tmp_path, monkeypatch):
    from tac import review_counter as rc

    ledger = tmp_path / "review_counter.jsonl"
    lock = tmp_path / ".review_counter.jsonl.lock"
    bpath = tmp_path / "session_events.jsonl"
    block = tmp_path / ".session_events.jsonl.lock"
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_PATH", bpath)
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_LOCK_PATH", block)

    rc.record_round(
        "SYNTHESIS_v3", 1, 0, rc.VERDICT_CLEAN,
        reviewed_commits=["abc123"], ledger_path=ledger, lock_path=lock,
    )
    events = B.read_events(bulletin_path=bpath)
    assert len(events) == 1
    assert events[0]["kind"] == B.EVENT_VERDICT_LANDED
    assert events[0]["subject"] == "SYNTHESIS_v3"
    assert events[0]["payload"]["verdict"] == rc.VERDICT_CLEAN


def test_record_round_failopen_when_bulletin_raises(tmp_path, monkeypatch):
    from tac import review_counter as rc

    ledger = tmp_path / "review_counter.jsonl"
    lock = tmp_path / ".review_counter.jsonl.lock"

    def _boom(*a, **k):
        raise RuntimeError("bulletin exploded")

    monkeypatch.setattr(B, "post_event", _boom)
    # The counter write MUST still succeed and the ledger row MUST land.
    rec = rc.record_round(
        "SURF", 1, 0, rc.VERDICT_CLEAN,
        reviewed_commits=["deadbeef"], ledger_path=ledger, lock_path=lock,
    )
    assert rec.surface_id == "SURF"
    state = rc.current_state("SURF", ledger_path=ledger)
    assert state.consecutive_clean == 1


# ─────────────────────────── CLI smoke ───────────────────────────


def test_session_recover_cli(tmp_path, monkeypatch):
    import tools.session_recover as cli  # imported via sys.path insert in the CLI module

    tool = rm._checkpoint_tool()
    monkeypatch.setattr(tool, "STATE_DIR", tmp_path)
    monkeypatch.setattr(tool, "JSONL_PATH", tmp_path / "subagent_progress.jsonl")
    monkeypatch.setattr(tool, "LOCK_PATH", tmp_path / ".subagent_progress.lock")
    # #389: register/complete post fail-open lifecycle bulletin events — isolate that
    # store too so the CLI test never pollutes the live feed.
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_PATH", tmp_path / "session_events.jsonl")
    monkeypatch.setattr(B, "DEFAULT_BULLETIN_LOCK_PATH", tmp_path / ".session_events.lock")

    assert cli.main(["register", "--subagent-id", "CLI-A",
                     "--respawn-context", "task #388 resume", "--step", "1"]) == 0
    assert cli.main(["report", "--stale-after-seconds", "0"]) == 0
    assert cli.main(["complete", "--subagent-id", "CLI-A"]) == 0
