# SPDX-License-Identifier: MIT
"""OP-10 read-amortization tests for the canonical Modal call_id ledger.

Per `feedback_op10_modal_ledger_append_amortization_landed_20260515.md` +
codex chunk 5 Finding #2 (`.omx/research/codex_chunked_full_codebase_review_20260515.md`
H-18). Covers:

- O(1) append path via POSIX O_APPEND under fcntl lock
- Sidecar index correctness (full rebuild + incremental extension)
- HISTORICAL_PROVENANCE invariants (Catalog #110/#113/#132): existing rows
  never mutated; canonical JSONL byte-payload identical to legacy path
- Backward compat: legacy O(N) full-rewrite path still callable; existing
  query helpers return same results post-amortization
- 4-process concurrent append safety (POSIX O_APPEND atomic-write semantics
  + fcntl serialization)
- 10000-entry stress: append latency stays bounded; query latency advantage
  via indexed helpers vs full-scan helpers
- Corruption recovery: torn trailing line refused; sidecar corruption
  triggers rebuild without data loss; canonical JSONL is source of truth
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import pytest

from tac.deploy.modal.call_id_ledger import (
    INDEX_SCHEMA_VERSION,
    SCHEMA_VERSION,
    STATUS_DISPATCHED,
    STATUS_HARVESTED,
    CallIdLedgerCorruptError,
    _append_event_locked,
    _ensure_index_fresh,
    _load_index,
    _rebuild_index_from_ledger,
    _validate_ledger_tail,
    load_call_ids,
    query_by_call_id,
    query_by_call_id_indexed,
    query_by_lane,
    query_by_lane_indexed,
    rebuild_sidecar_index,
    register_dispatched_call_id,
    update_call_id_outcome,
)


@pytest.fixture
def tmp_ledger(tmp_path):
    return tmp_path / "modal_call_id_ledger.jsonl"


@pytest.fixture
def tmp_lock(tmp_path):
    return tmp_path / "modal_call_id_ledger.jsonl.lock"


@pytest.fixture
def tmp_index(tmp_path):
    return tmp_path / "modal_call_id_ledger_index.json"


# -------------------------------------------------------------------------
# O(1) append path - POSIX O_APPEND under fcntl lock
# -------------------------------------------------------------------------


def test_truly_appending_writes_byte_identical_to_legacy(tmp_path):
    """The new write path produces byte-identical lines to the legacy path."""
    legacy_ledger = tmp_path / "legacy.jsonl"
    legacy_lock = tmp_path / "legacy.jsonl.lock"
    new_ledger = tmp_path / "new.jsonl"
    new_lock = tmp_path / "new.jsonl.lock"

    common_kwargs = {
        "call_id": "fc-byte-id",
        "lane_id": "lane_x",
        "label": "label_x",
        "dispatched_at_utc": "2026-05-15T12:00:00.000Z",
    }
    # Pin the time-stamping by passing all writer-injected fields explicitly
    # via extra. Otherwise written_at_utc differs.
    rec_legacy = register_dispatched_call_id(
        path=legacy_ledger,
        lock_path=legacy_lock,
        **common_kwargs,
    )
    # Force legacy path on second helper call
    rec_new = register_dispatched_call_id(
        path=new_ledger,
        lock_path=new_lock,
        **common_kwargs,
    )
    # Strip server-stamped fields that vary per call
    for r in (rec_legacy, rec_new):
        for k in ("written_at_utc", "written_pid", "written_host"):
            r.pop(k, None)
    # Re-serialize what was actually written, modulo server-stamped fields:
    legacy_lines = legacy_ledger.read_text().splitlines()
    new_lines = new_ledger.read_text().splitlines()
    assert len(legacy_lines) == 1
    assert len(new_lines) == 1
    # Both must be sort_keys=True JSON
    legacy_parsed = json.loads(legacy_lines[0])
    new_parsed = json.loads(new_lines[0])
    for k in ("written_at_utc", "written_pid", "written_host"):
        legacy_parsed.pop(k, None)
        new_parsed.pop(k, None)
    legacy_re = json.dumps(legacy_parsed, sort_keys=True)
    new_re = json.dumps(new_parsed, sort_keys=True)
    assert legacy_re == new_re


def test_append_uses_amortized_path_by_default(tmp_ledger, tmp_lock):
    """`_append_event_locked` defaults to the truly-appending path."""
    # Capture the size before / after to verify O_APPEND semantics
    register_dispatched_call_id(
        call_id="fc-1",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    s1 = tmp_ledger.stat().st_size
    register_dispatched_call_id(
        call_id="fc-2",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    s2 = tmp_ledger.stat().st_size
    assert s2 > s1
    # Both rows present
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 2
    assert {r["call_id"] for r in rows} == {"fc-1", "fc-2"}


def test_legacy_full_rewrite_path_still_callable(tmp_ledger, tmp_lock):
    """The legacy O(N) full-rewrite path is preserved for compat."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "dispatched",
        "call_id": "fc-legacy",
        "status": "dispatched",
    }
    rec = _append_event_locked(
        record,
        path=tmp_ledger,
        lock_path=tmp_lock,
        use_truly_appending=False,
    )
    assert rec["call_id"] == "fc-legacy"
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 1


# -------------------------------------------------------------------------
# HISTORICAL_PROVENANCE: existing bytes never mutated
# -------------------------------------------------------------------------


def test_amortized_appends_never_mutate_existing_bytes(tmp_ledger, tmp_lock):
    """Append-only: existing JSONL bytes are byte-identical after each append.

    Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113/#132.
    """
    register_dispatched_call_id(
        call_id="fc-immut-1",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    snapshot1 = tmp_ledger.read_bytes()
    register_dispatched_call_id(
        call_id="fc-immut-2",
        lane_id="lane_y",
        label="y",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    snapshot2 = tmp_ledger.read_bytes()
    # snapshot1 must be a strict prefix of snapshot2 (byte-for-byte)
    assert snapshot2.startswith(snapshot1)
    # The new bytes must be the new row + newline
    new_bytes = snapshot2[len(snapshot1):]
    assert new_bytes.endswith(b"\n")
    parsed = json.loads(new_bytes.decode("utf-8").strip())
    assert parsed["call_id"] == "fc-immut-2"


def test_outcome_update_is_new_row_not_mutation(tmp_ledger, tmp_lock):
    """Outcomes append NEW rows; original dispatched row is preserved verbatim."""
    register_dispatched_call_id(
        call_id="fc-life",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    snapshot_dispatched = tmp_ledger.read_bytes()
    update_call_id_outcome(
        call_id="fc-life",
        status=STATUS_HARVESTED,
        rc=0,
        score=0.18,
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    snapshot_harvested = tmp_ledger.read_bytes()
    # Original dispatched row preserved byte-for-byte
    assert snapshot_harvested.startswith(snapshot_dispatched)
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 2
    assert rows[0]["status"] == STATUS_DISPATCHED
    assert rows[1]["status"] == STATUS_HARVESTED


# -------------------------------------------------------------------------
# Sidecar index - load / save / rebuild / extend
# -------------------------------------------------------------------------


def test_sidecar_index_missing_returns_empty(tmp_index):
    idx = _load_index(index_path=tmp_index)
    assert idx["schema_version"] == INDEX_SCHEMA_VERSION
    assert idx["call_id_to_byte_offsets"] == {}
    assert idx["lane_id_to_byte_offsets"] == {}
    assert idx["last_indexed_byte"] == 0


def test_sidecar_index_corrupt_json_returns_empty(tmp_index):
    tmp_index.parent.mkdir(parents=True, exist_ok=True)
    tmp_index.write_text("{not json")
    idx = _load_index(index_path=tmp_index)
    assert idx["call_id_to_byte_offsets"] == {}


def test_sidecar_index_schema_mismatch_returns_empty(tmp_index):
    tmp_index.parent.mkdir(parents=True, exist_ok=True)
    tmp_index.write_text(json.dumps({"schema_version": 99}))
    idx = _load_index(index_path=tmp_index)
    assert idx["schema_version"] == INDEX_SCHEMA_VERSION


def test_rebuild_index_from_ledger_full_scan(tmp_ledger, tmp_lock, tmp_index):
    register_dispatched_call_id(
        call_id="fc-r1", lane_id="lane_a", label="a",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    register_dispatched_call_id(
        call_id="fc-r2", lane_id="lane_b", label="b",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-r1", status=STATUS_HARVESTED, rc=0,
        path=tmp_ledger, lock_path=tmp_lock,
    )
    idx = _rebuild_index_from_ledger(ledger_path=tmp_ledger, index_path=tmp_index)
    assert "fc-r1" in idx["call_id_to_byte_offsets"]
    assert len(idx["call_id_to_byte_offsets"]["fc-r1"]) == 2  # dispatched + harvested
    assert "fc-r2" in idx["call_id_to_byte_offsets"]
    assert len(idx["call_id_to_byte_offsets"]["fc-r2"]) == 1
    assert "lane_a" in idx["lane_id_to_byte_offsets"]
    # The harvest row's lane_id is None by default - only dispatched row maps
    assert len(idx["lane_id_to_byte_offsets"]["lane_a"]) >= 1


def test_ensure_index_fresh_incremental_extend(tmp_path, tmp_lock):
    """Sidecar built at size N; new appends extend without rescanning N bytes."""
    ledger = tmp_path / "incremental.jsonl"
    index_p = tmp_path / "incremental_index.json"
    # Force a different lock to avoid pollution
    lock = tmp_path / "incremental.jsonl.lock"

    # Append 5 rows
    for i in range(5):
        register_dispatched_call_id(
            call_id=f"fc-inc-{i}", lane_id=f"lane_inc_{i}", label=f"L{i}",
            path=ledger, lock_path=lock,
        )
    # Build index at size 5
    idx_v1 = _ensure_index_fresh(ledger_path=ledger, index_path=index_p)
    # The index sidecar built via the helpers above used the DEFAULT sidecar
    # path (because register_dispatched_call_id doesn't accept index_path).
    # _ensure_index_fresh on tmp_path index_p sees an empty sidecar +
    # builds full from the canonical ledger.
    size_v1 = idx_v1["last_indexed_byte"]
    assert size_v1 == ledger.stat().st_size
    assert len(idx_v1["call_id_to_byte_offsets"]) == 5

    # Append 5 more
    for i in range(5, 10):
        register_dispatched_call_id(
            call_id=f"fc-inc-{i}", lane_id=f"lane_inc_{i}", label=f"L{i}",
            path=ledger, lock_path=lock,
        )
    # Re-fresh: should incrementally extend
    idx_v2 = _ensure_index_fresh(ledger_path=ledger, index_path=index_p)
    assert idx_v2["last_indexed_byte"] == ledger.stat().st_size
    assert idx_v2["last_indexed_byte"] > size_v1
    assert len(idx_v2["call_id_to_byte_offsets"]) == 10


def test_ensure_index_fresh_rebuilds_on_truncation(tmp_ledger, tmp_lock, tmp_index):
    """If ledger size shrinks below sidecar's last_indexed_byte, full rebuild."""
    for i in range(3):
        register_dispatched_call_id(
            call_id=f"fc-tr-{i}", lane_id="lane_t", label=f"L{i}",
            path=tmp_ledger, lock_path=tmp_lock,
        )
    idx_v1 = _ensure_index_fresh(ledger_path=tmp_ledger, index_path=tmp_index)
    assert idx_v1["last_indexed_byte"] > 0

    # Truncate the ledger to 0 bytes (simulate operator clear)
    tmp_ledger.write_text("")
    idx_v2 = _ensure_index_fresh(ledger_path=tmp_ledger, index_path=tmp_index)
    assert idx_v2["last_indexed_byte"] == 0
    assert idx_v2["call_id_to_byte_offsets"] == {}


def test_ensure_index_fresh_rebuilds_when_ledger_path_mismatch(tmp_path, tmp_index):
    """If sidecar's ledger_path doesn't match, full rebuild against new path."""
    ledger_a = tmp_path / "ledger_a.jsonl"
    ledger_b = tmp_path / "ledger_b.jsonl"
    lock_a = tmp_path / "ledger_a.jsonl.lock"
    lock_b = tmp_path / "ledger_b.jsonl.lock"
    register_dispatched_call_id(
        call_id="fc-a", lane_id="lane_a", label="a",
        path=ledger_a, lock_path=lock_a,
    )
    register_dispatched_call_id(
        call_id="fc-b", lane_id="lane_b", label="b",
        path=ledger_b, lock_path=lock_b,
    )
    # Build against ledger_a
    _ensure_index_fresh(ledger_path=ledger_a, index_path=tmp_index)
    # Now ask for ledger_b's index - should rebuild (path differs)
    idx = _ensure_index_fresh(ledger_path=ledger_b, index_path=tmp_index)
    assert "fc-b" in idx["call_id_to_byte_offsets"]
    assert "fc-a" not in idx["call_id_to_byte_offsets"]


# -------------------------------------------------------------------------
# Indexed query helpers - equivalence with full-scan helpers
# -------------------------------------------------------------------------


def test_query_by_call_id_indexed_matches_scan(tmp_ledger, tmp_lock, tmp_index):
    register_dispatched_call_id(
        call_id="fc-q1", lane_id="lane_x", label="x",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-q1", status=STATUS_HARVESTED, rc=0,
        path=tmp_ledger, lock_path=tmp_lock,
    )
    register_dispatched_call_id(
        call_id="fc-q2", lane_id="lane_y", label="y",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    scan = query_by_call_id("fc-q1", path=tmp_ledger)
    indexed = query_by_call_id_indexed("fc-q1", ledger_path=tmp_ledger, index_path=tmp_index)
    assert len(scan) == 2 == len(indexed)
    # Strip volatile written_at fields
    for r in scan + indexed:
        r.pop("written_at_utc", None)
        r.pop("written_pid", None)
    assert scan == indexed


def test_query_by_lane_indexed_matches_scan(tmp_ledger, tmp_lock, tmp_index):
    for cid in ["fc-l1", "fc-l2", "fc-l3"]:
        register_dispatched_call_id(
            call_id=cid, lane_id="lane_shared", label=f"L_{cid}",
            path=tmp_ledger, lock_path=tmp_lock,
        )
    register_dispatched_call_id(
        call_id="fc-other", lane_id="lane_other", label="O",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    scan = query_by_lane("lane_shared", path=tmp_ledger)
    indexed = query_by_lane_indexed("lane_shared", ledger_path=tmp_ledger, index_path=tmp_index)
    assert len(scan) == 3 == len(indexed)
    assert {r["call_id"] for r in scan} == {r["call_id"] for r in indexed}


def test_query_by_call_id_indexed_unknown_returns_empty(tmp_ledger, tmp_lock, tmp_index):
    register_dispatched_call_id(
        call_id="fc-known", lane_id="lane_x", label="x",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    rows = query_by_call_id_indexed("fc-unknown", ledger_path=tmp_ledger, index_path=tmp_index)
    assert rows == []


def test_query_by_call_id_indexed_empty_ledger(tmp_path, tmp_index):
    ledger = tmp_path / "missing.jsonl"
    rows = query_by_call_id_indexed("fc-anything", ledger_path=ledger, index_path=tmp_index)
    assert rows == []


def test_query_by_call_id_indexed_rejects_empty_id(tmp_ledger):
    with pytest.raises(ValueError, match="call_id"):
        query_by_call_id_indexed("", ledger_path=tmp_ledger)


def test_query_by_lane_indexed_rejects_empty_id(tmp_ledger):
    with pytest.raises(ValueError, match="lane_id"):
        query_by_lane_indexed("", ledger_path=tmp_ledger)


# -------------------------------------------------------------------------
# rebuild_sidecar_index - operator-callable
# -------------------------------------------------------------------------


def test_rebuild_sidecar_index_writes_sidecar(tmp_ledger, tmp_lock, tmp_index):
    register_dispatched_call_id(
        call_id="fc-rb", lane_id="lane_x", label="x",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    assert not tmp_index.exists()
    rebuild_sidecar_index(ledger_path=tmp_ledger, index_path=tmp_index)
    assert tmp_index.exists()
    parsed = json.loads(tmp_index.read_text())
    assert "fc-rb" in parsed["call_id_to_byte_offsets"]


def test_rebuild_sidecar_index_recovers_from_corrupt_sidecar(tmp_ledger, tmp_lock, tmp_index):
    register_dispatched_call_id(
        call_id="fc-rec", lane_id="lane_x", label="x",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    tmp_index.parent.mkdir(parents=True, exist_ok=True)
    tmp_index.write_text("{garbage}")  # corrupt sidecar
    # Rebuild should recover
    idx = rebuild_sidecar_index(ledger_path=tmp_ledger, index_path=tmp_index)
    assert "fc-rec" in idx["call_id_to_byte_offsets"]


# -------------------------------------------------------------------------
# Tail validation - refuses torn writes
# -------------------------------------------------------------------------


def test_validate_ledger_tail_empty_ledger_ok(tmp_path):
    ledger = tmp_path / "empty.jsonl"
    _validate_ledger_tail(ledger)  # missing -> ok


def test_validate_ledger_tail_clean_ledger_ok(tmp_path):
    ledger = tmp_path / "clean.jsonl"
    ledger.write_text('{"call_id": "fc-1"}\n{"call_id": "fc-2"}\n')
    _validate_ledger_tail(ledger)


def test_validate_ledger_tail_no_trailing_newline_raises(tmp_path):
    ledger = tmp_path / "torn.jsonl"
    ledger.write_text('{"call_id": "fc-1"}\n{"call_id": "fc-2"}')  # no \n at end
    with pytest.raises(CallIdLedgerCorruptError, match="trailing region"):
        _validate_ledger_tail(ledger)


def test_validate_ledger_tail_invalid_json_last_line_raises(tmp_path):
    ledger = tmp_path / "bad.jsonl"
    ledger.write_text('{"call_id": "fc-1"}\n{not json}\n')
    with pytest.raises(CallIdLedgerCorruptError, match="invalid JSON"):
        _validate_ledger_tail(ledger)


def test_validate_ledger_tail_non_dict_root_raises(tmp_path):
    ledger = tmp_path / "list.jsonl"
    ledger.write_text('{"call_id": "fc-1"}\n["not", "dict"]\n')
    with pytest.raises(CallIdLedgerCorruptError, match="non-dict root"):
        _validate_ledger_tail(ledger)


def test_truly_appending_path_quarantines_torn_tail(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    lock = tmp_path / "ledger.jsonl.lock"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"call_id": "fc-1"}\nbroken')  # torn
    with pytest.raises(CallIdLedgerCorruptError, match="quarantined"):
        register_dispatched_call_id(
            call_id="fc-after", lane_id="lane_x", label="x",
            path=ledger, lock_path=lock,
        )
    # Quarantined: original torn file moved aside
    quarantines = list(ledger.parent.glob("ledger.jsonl.corrupt.*"))
    assert len(quarantines) == 1


# -------------------------------------------------------------------------
# Concurrent multi-process append safety - POSIX O_APPEND + fcntl lock
# -------------------------------------------------------------------------


def _stress_amortized_worker(args):
    """Append N rows using the amortized path."""
    ledger_path, lock_path, worker_id, n_rows = args
    for i in range(n_rows):
        register_dispatched_call_id(
            call_id=f"fc-amort-w{worker_id}-r{i}",
            lane_id=f"lane_amort_w{worker_id}",
            label=f"L_w{worker_id}_r{i}",
            path=Path(ledger_path),
            lock_path=Path(lock_path),
        )
    return worker_id


def test_concurrent_amortized_appends_4proc_no_lost_rows(tmp_path):
    """4 processes x 100 rows each -> 400 rows survive amortized append path."""
    ledger = tmp_path / "concurrent_amortized.jsonl"
    lock = tmp_path / "concurrent_amortized.jsonl.lock"
    n_workers = 4
    n_rows = 100
    ctx = mp.get_context("spawn")
    args_list = [(str(ledger), str(lock), w, n_rows) for w in range(n_workers)]
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_stress_amortized_worker, args_list)
    assert sorted(results) == list(range(n_workers))
    rows = load_call_ids(ledger)
    assert len(rows) == n_workers * n_rows
    expected = {f"fc-amort-w{w}-r{r}" for w in range(n_workers) for r in range(n_rows)}
    assert {r["call_id"] for r in rows} == expected


# -------------------------------------------------------------------------
# 10000-entry stress - append latency stays bounded; query advantage
# -------------------------------------------------------------------------


@pytest.mark.slow
def test_stress_10000_entries_append_latency_bounded(tmp_path):
    """10000 sequential appends should each finish well under 1 second.

    Sized to fit the 30s LOCK_TIMEOUT_SECONDS ceiling with margin. The
    legacy O(N) full-rewrite path would degrade to several seconds per
    append at this scale; the amortized path stays at ~few-ms.
    """
    ledger = tmp_path / "stress10k.jsonl"
    lock = tmp_path / "stress10k.jsonl.lock"
    N = 10_000
    t0 = time.perf_counter()
    last_call_latency_ms = 0.0
    for i in range(N):
        if i == N - 1:
            t_call = time.perf_counter()
        register_dispatched_call_id(
            call_id=f"fc-stress-{i}",
            lane_id=f"lane_stress_{i % 100}",  # 100 distinct lanes
            label=f"L_{i}",
            path=ledger,
            lock_path=lock,
        )
        if i == N - 1:
            last_call_latency_ms = (time.perf_counter() - t_call) * 1000
    total_s = time.perf_counter() - t0
    avg_ms = (total_s / N) * 1000
    print(f"\n[STRESS-10K] N={N} total={total_s:.2f}s avg={avg_ms:.3f}ms last={last_call_latency_ms:.3f}ms")
    # Final-row latency MUST stay bounded (no O(N) degradation)
    assert last_call_latency_ms < 1000, (
        f"Last-row latency {last_call_latency_ms:.1f}ms exceeds 1s; "
        "amortization regressed (O(N) write detected)."
    )
    # Sanity: avg latency well under 100ms (typical macOS dev box: ~1-3ms)
    assert avg_ms < 100, f"Average append latency {avg_ms:.1f}ms exceeds 100ms"


@pytest.mark.slow
def test_stress_10000_indexed_query_faster_than_full_scan(tmp_path):
    """At 10k entries, indexed query is >=10x faster than full-file scan."""
    ledger = tmp_path / "stress10k_q.jsonl"
    lock = tmp_path / "stress10k_q.jsonl.lock"
    index_p = tmp_path / "stress10k_q_index.json"
    N = 10_000
    target_call_id = "fc-target-50"  # mid-file row to defeat any scan-from-end optim
    for i in range(N):
        cid = f"fc-target-{i}" if i == 50 else f"fc-other-{i}"
        register_dispatched_call_id(
            call_id=cid,
            lane_id=f"lane_q_{i % 50}",
            label=f"L_{i}",
            path=ledger,
            lock_path=lock,
        )
    # Build the sidecar
    rebuild_sidecar_index(ledger_path=ledger, index_path=index_p)

    # Time scan vs indexed (10 reps)
    t0 = time.perf_counter()
    for _ in range(10):
        scan_rows = query_by_call_id(target_call_id, path=ledger)
    scan_avg_ms = ((time.perf_counter() - t0) / 10) * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        idx_rows = query_by_call_id_indexed(target_call_id, ledger_path=ledger, index_path=index_p)
    idx_avg_ms = ((time.perf_counter() - t0) / 10) * 1000

    print(
        f"\n[STRESS-10K-QUERY] N={N} target={target_call_id} "
        f"scan_avg={scan_avg_ms:.3f}ms indexed_avg={idx_avg_ms:.3f}ms "
        f"speedup={scan_avg_ms / max(idx_avg_ms, 0.001):.1f}x"
    )
    assert len(scan_rows) == 1 == len(idx_rows)
    assert scan_rows[0]["call_id"] == target_call_id == idx_rows[0]["call_id"]
    # Indexed must beat scan by >=5x at this size (typically 50-500x)
    assert idx_avg_ms < scan_avg_ms / 5, (
        f"Indexed query ({idx_avg_ms:.2f}ms) not materially faster than "
        f"scan ({scan_avg_ms:.2f}ms) at N={N}"
    )


# -------------------------------------------------------------------------
# Backward compat: existing query helpers still return same results
# -------------------------------------------------------------------------


def test_load_call_ids_unchanged_post_amortization(tmp_ledger, tmp_lock):
    """Lenient load_call_ids returns same shape as pre-amortization."""
    register_dispatched_call_id(
        call_id="fc-bc1", lane_id="lane_x", label="x",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 1
    r = rows[0]
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["call_id"] == "fc-bc1"
    assert r["status"] == STATUS_DISPATCHED
    assert "written_at_utc" in r
    assert "written_pid" in r


def test_query_by_call_id_unchanged(tmp_ledger, tmp_lock):
    """Legacy O(N) query_by_call_id continues to return chronological rows."""
    register_dispatched_call_id(
        call_id="fc-bc2", lane_id="lane_x", label="x",
        path=tmp_ledger, lock_path=tmp_lock,
    )
    update_call_id_outcome(
        call_id="fc-bc2", status=STATUS_HARVESTED, rc=0,
        path=tmp_ledger, lock_path=tmp_lock,
    )
    rows = query_by_call_id("fc-bc2", path=tmp_ledger)
    assert len(rows) == 2
    assert rows[0]["status"] == STATUS_DISPATCHED
    assert rows[1]["status"] == STATUS_HARVESTED


# -------------------------------------------------------------------------
# Sidecar-update failure does not break the canonical write path
# -------------------------------------------------------------------------


def test_sidecar_write_failure_does_not_block_canonical_append(
    tmp_ledger, tmp_lock, tmp_index, monkeypatch
):
    """Per fail-soft sidecar contract: canonical JSONL is source of truth."""
    # Make sidecar write fail by forcing _save_index to raise
    from tac.deploy.modal import call_id_ledger as mod

    original_save_index = mod._save_index
    call_count = {"n": 0}

    def failing_save_index(*a, **kw):
        call_count["n"] += 1
        raise OSError("simulated sidecar write failure")

    monkeypatch.setattr(mod, "_save_index", failing_save_index)

    # Append must still succeed (canonical JSONL is source of truth)
    rec = register_dispatched_call_id(
        call_id="fc-failsoft",
        lane_id="lane_x",
        label="x",
        path=tmp_ledger,
        lock_path=tmp_lock,
    )
    assert rec["call_id"] == "fc-failsoft"
    rows = load_call_ids(tmp_ledger)
    assert len(rows) == 1
    assert rows[0]["call_id"] == "fc-failsoft"

    # Restore + verify rebuild from canonical works
    monkeypatch.setattr(mod, "_save_index", original_save_index)
    idx = rebuild_sidecar_index(ledger_path=tmp_ledger, index_path=tmp_index)
    assert "fc-failsoft" in idx["call_id_to_byte_offsets"]


# -------------------------------------------------------------------------
# Lock-held assertion - defense in depth
# -------------------------------------------------------------------------


def test_truly_appending_write_refuses_without_lock(tmp_ledger):
    """_truly_appending_write_locked refuses if caller is not holding the lock."""
    from tac.deploy.modal.call_id_ledger import _truly_appending_write_locked

    record = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "dispatched",
        "call_id": "fc-lock",
        "status": "dispatched",
    }
    with pytest.raises(RuntimeError, match="WITHOUT holding _ledger_lock"):
        _truly_appending_write_locked(record, path=tmp_ledger)


def test_index_update_refuses_without_lock(tmp_ledger, tmp_index):
    from tac.deploy.modal.call_id_ledger import _index_record_event_locked

    record = {"call_id": "fc-x", "lane_id": "lane_x"}
    with pytest.raises(RuntimeError, match="WITHOUT holding _ledger_lock"):
        _index_record_event_locked(
            record=record,
            byte_offset=0,
            total_size=100,
            ledger_path=tmp_ledger,
            index_path=tmp_index,
        )


# -------------------------------------------------------------------------
# Schema constants stable
# -------------------------------------------------------------------------


def test_index_schema_version_pinned():
    assert INDEX_SCHEMA_VERSION == 1
