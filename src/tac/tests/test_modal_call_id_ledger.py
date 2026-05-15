# SPDX-License-Identifier: MIT
"""Tests for the canonical Modal call_id ledger.

Mirrors the ``tac.deploy.lightning.active_jobs_state`` test pattern. Covers:
- schema validation
- atomic append (transactional .tmp + os.replace)
- fcntl-locked concurrency (multiprocessing 4-proc stress test)
- strict-load corrupt detection (CallIdLedgerCorruptError)
- query helpers (by_call_id / by_lane / unharvested / post_utc)
- full call_id lifecycle (dispatch -> harvest -> failure events)
- public API contract
"""

from __future__ import annotations

import datetime as _dt
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from tac.deploy.modal.call_id_ledger import (
    EVENT_DISPATCHED,
    EVENT_STALE,
    SCHEMA_VERSION,
    STATUS_DISPATCHED,
    STATUS_FAILED,
    STATUS_HARVESTED,
    STATUS_STALE,
    TERMINAL_STATUSES,
    VALID_EVENT_TYPES,
    VALID_STATUSES,
    CallIdLedgerCorruptError,
    latest_status_by_call_id,
    load_call_ids,
    load_call_ids_strict,
    query_all_post_utc,
    query_by_call_id,
    query_by_lane,
    query_unharvested,
    register_dispatched_call_id,
    update_call_id_outcome,
)

# ─────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────


def test_schema_version_is_pinned():
    assert SCHEMA_VERSION == 1


def test_valid_event_types_canonical_set():
    assert {
        "dispatched",
        "harvested",
        "failed",
        "stale",
        "manually_terminated",
    } == VALID_EVENT_TYPES


def test_valid_statuses_match_event_types():
    assert VALID_STATUSES == VALID_EVENT_TYPES


def test_terminal_statuses_subset():
    assert TERMINAL_STATUSES.issubset(VALID_STATUSES)
    assert "dispatched" not in TERMINAL_STATUSES


# ─────────────────────────────────────────────────────────────────────────
# register_dispatched_call_id — happy path + validation
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_ledger(tmp_path):
    """Fresh ledger path per test."""
    return tmp_path / "modal_call_id_ledger.jsonl"


@pytest.fixture
def tmp_lock(tmp_path):
    return tmp_path / "modal_call_id_ledger.jsonl.lock"


def test_register_dispatched_call_id_writes_row(tmp_ledger, tmp_lock):
    rec = register_dispatched_call_id(
        call_id="fc-test-001",
        lane_id="lane_demo_substrate_20260515",
        label="substrate_demo_modal_t4_dispatch_20260515T140000Z__smoke",
        gpu="T4",
        expected_cost_usd=0.30,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    assert rec["call_id"] == "fc-test-001"
    assert rec["event_type"] == EVENT_DISPATCHED
    assert rec["status"] == STATUS_DISPATCHED
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["expected_cost_usd"] == 0.30
    assert rec["gpu"] == "T4"
    assert rec["written_at_utc"]  # set by helper
    assert rec["written_pid"] == os.getpid()


def test_register_dispatched_call_id_persists_to_disk(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-test-002",
        lane_id="lane_x",
        label="x_dispatch",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 1
    assert rows[0]["call_id"] == "fc-test-002"


def test_register_dispatched_call_id_rejects_empty_call_id(tmp_ledger, tmp_lock):
    with pytest.raises(ValueError, match="call_id"):
        register_dispatched_call_id(
            call_id="",
            lane_id="lane_x",
            label="x",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )


def test_register_dispatched_call_id_rejects_empty_lane_id(tmp_ledger, tmp_lock):
    with pytest.raises(ValueError, match="lane_id"):
        register_dispatched_call_id(
            call_id="fc-x",
            lane_id="",
            label="x",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )


def test_register_dispatched_call_id_rejects_empty_label(tmp_ledger, tmp_lock):
    with pytest.raises(ValueError, match="label"):
        register_dispatched_call_id(
            call_id="fc-x",
            lane_id="lane_x",
            label="",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )


def test_register_dispatched_call_id_rejects_newline_in_id(tmp_ledger, tmp_lock):
    with pytest.raises(ValueError, match="newlines"):
        register_dispatched_call_id(
            call_id="fc-bad\nid",
            lane_id="lane_x",
            label="x",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )


def test_register_extra_kwargs_attached(tmp_ledger, tmp_lock):
    rec = register_dispatched_call_id(
        call_id="fc-ex",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
        operator_note="urgent canary",
    )
    assert rec["operator_note"] == "urgent canary"


def test_register_extra_kwargs_cannot_overwrite_reserved(tmp_ledger, tmp_lock):
    with pytest.raises(ValueError, match="collides with a reserved"):
        register_dispatched_call_id(
            call_id="fc-ex",
            lane_id="lane_x",
            label="x",
            path=tmp_ledger,
            lock_path=tmp_lock,
            schema_version=99,  # collides
        )


# ─────────────────────────────────────────────────────────────────────────
# update_call_id_outcome — happy path + validation
# ─────────────────────────────────────────────────────────────────────────


def test_update_call_id_outcome_appends_new_row(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-life-001",
        lane_id="lane_y",
        label="y",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-life-001",
        status=STATUS_HARVESTED,
        rc=0,
        elapsed_seconds=120.5,
        score=0.18,
        score_axis="contest_cuda",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 2
    assert rows[0]["status"] == STATUS_DISPATCHED
    assert rows[1]["status"] == STATUS_HARVESTED
    assert rows[1]["score"] == 0.18


def test_update_outcome_rejects_invalid_status(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-y",
        lane_id="lane_y",
        label="y",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    with pytest.raises(ValueError, match="status must be one of"):
        update_call_id_outcome(
            call_id="fc-y",
            status="bogus_status",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )


def test_update_outcome_event_type_overrides_status(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-ev",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rec = update_call_id_outcome(
        call_id="fc-ev",
        status=STATUS_FAILED,
        event_type=EVENT_STALE,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    assert rec["status"] == STATUS_FAILED
    assert rec["event_type"] == EVENT_STALE


def test_update_outcome_does_not_mutate_dispatched_row(tmp_ledger, tmp_lock):
    """Per CLAUDE.md HISTORICAL_PROVENANCE — outcomes are NEW rows, not mutations."""
    register_dispatched_call_id(
        call_id="fc-immut",
        lane_id="lane_z",
        label="z",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-immut",
        status=STATUS_HARVESTED,
        rc=0,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 2  # NOT 1 — original row preserved
    assert rows[0]["event_type"] == EVENT_DISPATCHED
    assert rows[0]["status"] == STATUS_DISPATCHED  # unchanged


# ─────────────────────────────────────────────────────────────────────────
# load_call_ids — lenient
# ─────────────────────────────────────────────────────────────────────────


def test_load_call_ids_returns_empty_when_missing(tmp_path):
    p = tmp_path / "missing.jsonl"
    assert load_call_ids(p) == []


def test_load_call_ids_skips_malformed_lines(tmp_path):
    p = tmp_path / "with_bad.jsonl"
    p.write_text(
        '{"schema_version": 1, "call_id": "fc-1", "event_type": "dispatched", "status": "dispatched"}\n'
        "{not json}\n"
        '{"schema_version": 1, "call_id": "fc-2", "event_type": "dispatched", "status": "dispatched"}\n'
    )
    rows = load_call_ids(p)
    assert len(rows) == 2
    assert {r["call_id"] for r in rows} == {"fc-1", "fc-2"}


def test_load_call_ids_skips_non_dict_rows(tmp_path):
    p = tmp_path / "nondict.jsonl"
    p.write_text('{"call_id": "fc-1"}\n["not", "a", "dict"]\n')
    rows = load_call_ids(p)
    assert len(rows) == 1


# ─────────────────────────────────────────────────────────────────────────
# load_call_ids_strict — Catalog #138 strict-load discipline
# ─────────────────────────────────────────────────────────────────────────


def test_load_strict_returns_empty_when_missing(tmp_path):
    p = tmp_path / "missing.jsonl"
    assert load_call_ids_strict(p) == []


def test_load_strict_raises_on_malformed_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(CallIdLedgerCorruptError, match="invalid JSON"):
        load_call_ids_strict(p)


def test_load_strict_raises_on_non_dict_root(tmp_path):
    p = tmp_path / "list.jsonl"
    p.write_text('["not", "a", "dict"]\n')
    with pytest.raises(CallIdLedgerCorruptError, match="non-dict root"):
        load_call_ids_strict(p)


def test_load_strict_clean_file_returns_rows(tmp_path):
    p = tmp_path / "clean.jsonl"
    p.write_text('{"call_id": "fc-x", "event_type": "dispatched"}\n')
    rows = load_call_ids_strict(p)
    assert len(rows) == 1
    assert rows[0]["call_id"] == "fc-x"


# ─────────────────────────────────────────────────────────────────────────
# Quarantine on corrupt — append refused
# ─────────────────────────────────────────────────────────────────────────


def test_register_quarantines_corrupt_and_refuses(tmp_ledger, tmp_lock):
    tmp_ledger.parent.mkdir(parents=True, exist_ok=True)
    tmp_ledger.write_text("{garbage}\n")
    with pytest.raises(CallIdLedgerCorruptError, match="quarantined"):
        register_dispatched_call_id(
            call_id="fc-after-corrupt",
            lane_id="lane_x",
            label="x",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )
    # corrupt file moved aside; canonical path now absent
    assert not tmp_ledger.exists()
    quarantines = list(tmp_ledger.parent.glob("modal_call_id_ledger.jsonl.corrupt.*"))
    assert len(quarantines) == 1


# ─────────────────────────────────────────────────────────────────────────
# Atomic write — no .tmp leakage
# ─────────────────────────────────────────────────────────────────────────


def test_atomic_write_leaves_no_tmp_after_success(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-atomic",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    leftovers = list(tmp_ledger.parent.glob("modal_call_id_ledger.jsonl.tmp.*"))
    assert leftovers == []


# ─────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────


def test_query_by_call_id_returns_chronological(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-multi",
        lane_id="lane_a",
        label="a",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-multi",
        status=STATUS_HARVESTED,
        rc=0,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = query_by_call_id("fc-multi", path=tmp_ledger)
    assert len(rows) == 2
    assert rows[0]["status"] == STATUS_DISPATCHED
    assert rows[1]["status"] == STATUS_HARVESTED


def test_query_by_lane_returns_all(tmp_ledger, tmp_lock):
    for cid in ["fc-l1", "fc-l2", "fc-l3"]:
        register_dispatched_call_id(
            call_id=cid,
            lane_id="lane_shared",
            label=f"label_{cid}",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )
    register_dispatched_call_id(
        call_id="fc-other",
        lane_id="lane_other",
        label="other",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = query_by_lane("lane_shared", path=tmp_ledger)
    assert len(rows) == 3
    assert {r["call_id"] for r in rows} == {"fc-l1", "fc-l2", "fc-l3"}


def test_query_all_post_utc_filters_correctly(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-old",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    pivot = _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    time.sleep(0.01)
    register_dispatched_call_id(
        call_id="fc-new",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = query_all_post_utc(pivot, path=tmp_ledger)
    cids = {r["call_id"] for r in rows}
    assert "fc-new" in cids


def test_query_unharvested_returns_only_dispatched(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-pend",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    register_dispatched_call_id(
        call_id="fc-done",
        lane_id="lane_y",
        label="y",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-done",
        status=STATUS_HARVESTED,
        rc=0,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    rows = query_unharvested(path=tmp_ledger)
    cids = {r["call_id"] for r in rows}
    assert cids == {"fc-pend"}


def test_query_unharvested_age_filter(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-young",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    # younger than 60s = excluded
    rows = query_unharvested(older_than_seconds=60, path=tmp_ledger)
    assert rows == []
    # without age filter = included
    rows = query_unharvested(older_than_seconds=0, path=tmp_ledger)
    assert len(rows) == 1


def test_latest_status_by_call_id(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-x",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-x",
        status=STATUS_FAILED,
        rc=1,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    statuses = latest_status_by_call_id(path=tmp_ledger)
    assert statuses["fc-x"] == STATUS_FAILED


# ─────────────────────────────────────────────────────────────────────────
# Concurrency stress test — 4-process spawn pool
# ─────────────────────────────────────────────────────────────────────────


def _stress_worker(args):
    """Append N rows from one process to a shared ledger."""
    ledger_path, lock_path, worker_id, n_rows = args
    for i in range(n_rows):
        register_dispatched_call_id(
            call_id=f"fc-w{worker_id}-r{i}",
            lane_id=f"lane_w{worker_id}",
            label=f"label_w{worker_id}_r{i}",
            path=Path(ledger_path),
            lock_path=Path(lock_path),
        )
    return worker_id


def test_concurrent_appends_under_fcntl_lock_no_lost_rows(tmp_path):
    """4 processes each append 5 rows -> all 20 rows survive."""
    ledger = tmp_path / "stress_ledger.jsonl"
    lock = tmp_path / "stress_ledger.jsonl.lock"
    n_workers = 4
    n_rows = 5
    ctx = mp.get_context("spawn")
    args = [(str(ledger), str(lock), w, n_rows) for w in range(n_workers)]
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_stress_worker, args)
    assert sorted(results) == list(range(n_workers))
    rows = load_call_ids(ledger)
    assert len(rows) == n_workers * n_rows
    cids = {r["call_id"] for r in rows}
    expected = {f"fc-w{w}-r{r}" for w in range(n_workers) for r in range(n_rows)}
    assert cids == expected


# ─────────────────────────────────────────────────────────────────────────
# Full lifecycle smoke
# ─────────────────────────────────────────────────────────────────────────


def test_full_call_id_lifecycle_dispatch_harvest_failure_stale(tmp_ledger, tmp_lock):
    """Smoke: 3 distinct call_ids each go through different terminal events."""
    # Dispatch 3 calls
    for cid in ["fc-success", "fc-failure", "fc-stale"]:
        register_dispatched_call_id(
            call_id=cid,
            lane_id="lane_smoke",
            label=f"label_{cid}",
            path=tmp_ledger,
            lock_path=tmp_lock,
        )
    # fc-success gets harvested
    update_call_id_outcome(
        call_id="fc-success",
        status=STATUS_HARVESTED,
        rc=0,
        score=0.19,
        score_axis="contest_cuda",
        archive_sha256="a" * 64,
        archive_bytes=300_000,
        evidence_grade="contest-CUDA",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    # fc-failure rc=1
    update_call_id_outcome(
        call_id="fc-failure",
        status=STATUS_FAILED,
        rc=1,
        elapsed_seconds=12.5,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    # fc-stale expired
    update_call_id_outcome(
        call_id="fc-stale",
        status=STATUS_STALE,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    statuses = latest_status_by_call_id(path=tmp_ledger)
    assert statuses == {
        "fc-success": STATUS_HARVESTED,
        "fc-failure": STATUS_FAILED,
        "fc-stale": STATUS_STALE,
    }
    unharvested = query_unharvested(path=tmp_ledger)
    assert unharvested == []  # all 3 reached terminal status


# ─────────────────────────────────────────────────────────────────────────
# JSONL byte-stable format (sort_keys=True)
# ─────────────────────────────────────────────────────────────────────────


def test_jsonl_rows_have_sorted_keys(tmp_ledger, tmp_lock):
    register_dispatched_call_id(
        call_id="fc-sort",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    text = tmp_ledger.read_text()
    line = text.strip()
    parsed = json.loads(line)
    re_serialized = json.dumps(parsed, sort_keys=True)
    assert line == re_serialized
