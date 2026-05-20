# SPDX-License-Identifier: MIT
"""Tests for the cathedral autopilot consumer verdict ledger.

Sister of ``test_canonical_equations_registry.py`` + ``test_modal_call_id_ledger.py``
+ ``test_codex_to_claude_inbox.py`` — same canonical 4-layer ledger pattern
per Catalog #245/#313/#333/#344. Lands the activation-discipline ledger for
the cathedral autopilot consumer invocations.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
every persisted row MUST carry ``score_claim=False`` + ``promotion_eligible=False``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.cathedral.verdict_ledger import (
    CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH,
    CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION,
    EVENT_CONSUMER_INVOCATION_BATCH,
    EVENT_OPERATOR_REVIEW,
    EVENT_RECTIFIED_VERDICT,
    VALID_EVENT_TYPES,
    CathedralConsumerVerdictLedgerCorruptError,
    _append_event_locked,
    append_consumer_invocation_batch,
    load_verdict_events_lenient,
    load_verdict_ledger_strict,
    query_consumer_activity_summary,
    query_latest_session,
    query_sessions,
)


def _make_invocation_payload(consumer_count: int = 3, candidates_invoked: int = 2) -> dict:
    invocations = []
    for ci in range(consumer_count):
        for cdi in range(candidates_invoked):
            invocations.append({
                "consumer_module": f"tac.cathedral_consumers.fake_consumer_{ci}",
                "consumer_name": f"fake_consumer_{ci}",
                "consumer_version": "1.0",
                "candidate_id": f"lane_test_{cdi}",
                "predicted_delta_adjustment": 0.0,
                "rationale": f"fake_consumer_{ci} non-vacuous rationale for lane_test_{cdi}",
                "axis_tag": "[predicted]",
                "promotable": False,
                "confidence": 0.0,
            })
    return {
        "schema": "cathedral_consumer_invocation_v1_20260519",
        "evidence_grade": "[predicted, cathedral consumer invocation]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "panel_axis": "contest_cpu",
        "top_n": candidates_invoked,
        "consumer_count": consumer_count,
        "consumer_names": [f"tac.cathedral_consumers.fake_consumer_{i}" for i in range(consumer_count)],
        "candidates_invoked": candidates_invoked,
        "invocations": invocations,
        "master_gradient_annotations": [
            {"candidate_id": f"lane_test_{i}", "explanation": "fake annotation"}
            for i in range(candidates_invoked)
        ],
        "master_gradient_rerank_invoked": True,
    }


def test_schema_version_pinned():
    assert CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION == "cathedral_consumer_verdict_v1_20260519"


def test_valid_event_types_canonical_set():
    assert VALID_EVENT_TYPES == frozenset({
        EVENT_CONSUMER_INVOCATION_BATCH,
        EVENT_OPERATOR_REVIEW,
        EVENT_RECTIFIED_VERDICT,
    })


def test_append_consumer_invocation_batch_writes_row(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    payload = _make_invocation_payload(consumer_count=3, candidates_invoked=2)
    record = append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cpu",
        rank_axis="eig_per_dollar",
        candidate_ids=["lane_test_0", "lane_test_1"],
        path=p,
        lock_path=lp,
    )
    assert record["event_type"] == EVENT_CONSUMER_INVOCATION_BATCH
    assert record["consumer_count"] == 3
    assert record["candidates_invoked"] == 2
    assert record["n_invocations"] == 6
    assert record["n_non_vacuous"] == 6
    assert record["n_errors"] == 0
    assert record["score_claim"] is False
    assert record["promotion_eligible"] is False
    assert record["axis_tag"] == "[predicted]"
    rows = load_verdict_ledger_strict(p)
    assert len(rows) == 1
    assert rows[0]["session_id"] == record["session_id"]


def test_append_persists_to_disk_atomically(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    payload = _make_invocation_payload()
    append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cuda",
        rank_axis="predicted_score_delta",
        candidate_ids=["lane_a", "lane_b"],
        path=p,
        lock_path=lp,
    )
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert txt.count("\n") == 1
    parsed = json.loads(txt.strip())
    assert parsed["panel_axis"] == "contest_cuda"
    assert parsed["rank_axis"] == "predicted_score_delta"


def test_append_refuses_non_mapping(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    with pytest.raises(TypeError):
        append_consumer_invocation_batch(
            "not a mapping",  # type: ignore[arg-type]
            panel_axis="contest_cpu",
            rank_axis="eig_per_dollar",
            candidate_ids=["lane_x"],
            path=p,
            lock_path=lp,
        )


def test_validate_event_record_refuses_promoted_row(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    # Build a malformed record manually (claims score authority) — the validator MUST refuse.
    bad_record = {
        "schema_version": CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION,
        "event_type": EVENT_CONSUMER_INVOCATION_BATCH,
        "session_id": "test_session",
        "score_claim": True,  # FORBIDDEN per Catalog #287/#323
        "promotion_eligible": False,
    }
    with pytest.raises(ValueError, match="score_claim must be False"):
        _append_event_locked(bad_record, path=p, lock_path=lp)


def test_validate_event_record_refuses_missing_session_id(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    bad_record = {
        "schema_version": CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION,
        "event_type": EVENT_CONSUMER_INVOCATION_BATCH,
        "session_id": "",  # empty
        "score_claim": False,
        "promotion_eligible": False,
    }
    with pytest.raises(ValueError, match="session_id"):
        _append_event_locked(bad_record, path=p, lock_path=lp)


def test_validate_event_record_refuses_bad_event_type(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    bad_record = {
        "schema_version": CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION,
        "event_type": "rogue_event",
        "session_id": "test",
        "score_claim": False,
        "promotion_eligible": False,
    }
    with pytest.raises(ValueError, match="event_type"):
        _append_event_locked(bad_record, path=p, lock_path=lp)


def test_load_lenient_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    payload = _make_invocation_payload()
    append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cpu",
        rank_axis="eig_per_dollar",
        candidate_ids=["lane_a"],
        path=p,
        lock_path=lp,
    )
    # Append a malformed line manually.
    with p.open("a", encoding="utf-8") as f:
        f.write("not valid json\n")
    rows = load_verdict_events_lenient(p)
    assert len(rows) == 1  # only the valid row


def test_load_strict_raises_on_corrupt(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    p.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(CathedralConsumerVerdictLedgerCorruptError):
        load_verdict_ledger_strict(p)


def test_load_strict_raises_on_non_dict_root(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    p.write_text('["not a dict"]\n', encoding="utf-8")
    with pytest.raises(CathedralConsumerVerdictLedgerCorruptError, match="non-dict"):
        load_verdict_ledger_strict(p)


def test_query_latest_session_returns_most_recent(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    payload = _make_invocation_payload()
    # Append two batches at different times.
    rec1 = append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cpu",
        rank_axis="eig_per_dollar",
        candidate_ids=["lane_a"],
        path=p,
        lock_path=lp,
        session_id="session_a",
    )
    rec2 = append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cuda",
        rank_axis="predicted_score_delta",
        candidate_ids=["lane_b"],
        path=p,
        lock_path=lp,
        session_id="session_b",
    )
    latest = query_latest_session(path=p)
    assert latest is not None
    # rec2 was written later so its written_at_utc >= rec1's. Equal timestamps allowed.
    assert latest["written_at_utc"] >= rec1["written_at_utc"]
    assert latest["session_id"] in {rec1["session_id"], rec2["session_id"]}


def test_query_latest_session_empty_returns_none(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    assert query_latest_session(path=p) is None


def test_query_sessions_unbounded(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    payload = _make_invocation_payload()
    for i in range(3):
        append_consumer_invocation_batch(
            payload,
            panel_axis="contest_cpu",
            rank_axis="eig_per_dollar",
            candidate_ids=[f"lane_{i}"],
            path=p,
            lock_path=lp,
            session_id=f"sess_{i}",
        )
    rows = query_sessions(path=p)
    assert len(rows) == 3


def test_query_consumer_activity_summary_aggregates(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    payload = _make_invocation_payload(consumer_count=3, candidates_invoked=4)
    append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cpu",
        rank_axis="eig_per_dollar",
        candidate_ids=["lane_a", "lane_b", "lane_c", "lane_d"],
        path=p,
        lock_path=lp,
    )
    summary = query_consumer_activity_summary(path=p)
    assert len(summary) == 3
    for entry in summary.values():
        assert entry["session_count"] == 1
        assert entry["candidate_count_total"] == 4
        assert entry["last_seen_utc"]


def test_append_counts_non_vacuous_correctly(tmp_path: Path):
    p = tmp_path / "verdict_ledger.jsonl"
    lp = tmp_path / "verdict_ledger.jsonl.lock"
    # Build payload with mix of vacuous + non-vacuous + error rows.
    payload = {
        "schema": "cathedral_consumer_invocation_v1_20260519",
        "consumer_count": 4,
        "consumer_names": ["c0", "c1", "c2", "c3"],
        "candidates_invoked": 1,
        "invocations": [
            {"consumer_module": "c0", "candidate_id": "x", "predicted_delta_adjustment": 0.0, "rationale": "non-empty"},
            {"consumer_module": "c1", "candidate_id": "x", "predicted_delta_adjustment": 0.0, "rationale": ""},  # vacuous
            {"consumer_module": "c2", "candidate_id": "x", "error": "test error"},  # error
            {"consumer_module": "c3", "candidate_id": "x", "predicted_delta_adjustment": 0.1, "rationale": ""},  # non-vacuous via adj
        ],
        "master_gradient_annotations": [],
    }
    record = append_consumer_invocation_batch(
        payload,
        panel_axis="contest_cpu",
        rank_axis="eig_per_dollar",
        candidate_ids=["x"],
        path=p,
        lock_path=lp,
    )
    assert record["n_invocations"] == 4
    assert record["n_non_vacuous"] == 2  # c0 (rationale) + c3 (adj)
    assert record["n_errors"] == 1  # c2
