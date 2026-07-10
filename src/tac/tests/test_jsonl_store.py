"""Tests for the canonical fcntl-locked JSONL append helper (tac.jsonl_store) — the shared
implementation that replaces the byte-identical ``_append_locked_jsonl`` copies formerly
duplicated in ``tac.witness_dsl.activation_ledger`` and ``tac.witness_dsl.curriculum_candidate_pool``
(per ``.omx/research/hardcode_duplication_audit_witness_stack_20260710.md`` finding #4)."""
from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

import pytest

from tac.jsonl_store import append_locked_jsonl


@pytest.fixture
def store(tmp_path):
    return tmp_path / "sub" / "store.jsonl"


# --- basic append behavior ---------------------------------------------------
def test_append_creates_file_and_parent_dir(store):
    assert not store.parent.exists()
    append_locked_jsonl(store, {"a": 1})
    assert store.exists()
    assert store.parent.exists()


def test_append_writes_one_json_line(store):
    append_locked_jsonl(store, {"a": 1, "b": 2})
    lines = store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row == {"a": 1, "b": 2}


def test_append_ends_with_newline(store):
    append_locked_jsonl(store, {"a": 1})
    content = store.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert content.count("\n") == 1


def test_multiple_appends_preserve_prior_rows_in_order(store):
    append_locked_jsonl(store, {"n": 1})
    append_locked_jsonl(store, {"n": 2})
    append_locked_jsonl(store, {"n": 3})
    rows = [json.loads(ln) for ln in store.read_text(encoding="utf-8").splitlines()]
    assert [r["n"] for r in rows] == [1, 2, 3]


def test_append_does_not_truncate_existing_content(store):
    append_locked_jsonl(store, {"n": 1})
    size_after_first = store.stat().st_size
    append_locked_jsonl(store, {"n": 2})
    assert store.stat().st_size > size_after_first
    rows = [json.loads(ln) for ln in store.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


# --- sort_keys behavior -------------------------------------------------------
def test_sort_keys_default_true_orders_keys_alphabetically(store):
    append_locked_jsonl(store, {"z": 1, "a": 2, "m": 3})
    line = store.read_text(encoding="utf-8").splitlines()[0]
    # sort_keys=True -> alphabetical key order in the raw JSON text.
    assert line.index('"a"') < line.index('"m"') < line.index('"z"')


def test_sort_keys_false_preserves_insertion_order(store):
    append_locked_jsonl(store, {"z": 1, "a": 2, "m": 3}, sort_keys=False)
    line = store.read_text(encoding="utf-8").splitlines()[0]
    assert line.index('"z"') < line.index('"a"') < line.index('"m"')


# --- round-trip / value fidelity ----------------------------------------------
def test_round_trip_preserves_nested_structures(store):
    row = {"lever": "X", "nested": {"a": [1, 2, 3], "b": None}, "flag": True}
    append_locked_jsonl(store, row)
    read_back = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert read_back == row


def test_round_trip_preserves_float_and_unicode(store):
    row = {"est_delta_s": 0.0037, "notes": "café — witness ξ"}
    append_locked_jsonl(store, row)
    read_back = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert read_back == row


# --- edge cases ----------------------------------------------------------------
def test_append_empty_dict(store):
    append_locked_jsonl(store, {})
    read_back = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert read_back == {}


def test_append_accepts_path_like_string_via_Path_wrapper(tmp_path):
    p = Path(str(tmp_path / "str_path.jsonl"))
    append_locked_jsonl(p, {"n": 1})
    assert p.exists()


def test_parent_dir_already_exists_is_fine(store):
    store.parent.mkdir(parents=True, exist_ok=True)
    append_locked_jsonl(store, {"n": 1})
    assert store.exists()


# --- concurrency / atomicity (the whole point of the fcntl lock) ---------------
def test_concurrent_appends_from_threads_all_survive_no_interleave(store):
    """N concurrent threads each append their own row; every row must land as its OWN clean JSON
    line (no torn/interleaved writes), and all N rows must be present — proving the lock actually
    serializes the critical section rather than merely existing as decoration."""
    n = 40
    barrier = threading.Barrier(n)

    def _write(i: int) -> None:
        barrier.wait()  # maximize contention: everyone hits the lock at ~the same instant
        append_locked_jsonl(store, {"i": i, "payload": "x" * 50})

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_write, range(n)))

    lines = store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    # every single line must parse cleanly (a torn/interleaved write would corrupt a line here)
    parsed = [json.loads(ln) for ln in lines]
    assert sorted(r["i"] for r in parsed) == list(range(n))


def _write_one_row(args: tuple[str, int]) -> None:
    """Module-level (picklable) worker for the cross-process atomicity test below."""
    store_path, i = args
    append_locked_jsonl(Path(store_path), {"i": i})


def test_concurrent_appends_from_processes_all_survive(tmp_path):
    """Cross-process atomicity: the fcntl lock must serialize writers from SEPARATE processes
    (the actual failure mode the lock exists to prevent — a thread-only test cannot prove this
    because threads share the GIL and a Python-level write already looks atomic without any lock)."""
    store = tmp_path / "proc_store.jsonl"
    n = 8

    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as ex:
        list(ex.map(_write_one_row, [(str(store), i) for i in range(n)]))

    lines = store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    parsed = [json.loads(ln) for ln in lines]
    assert sorted(r["i"] for r in parsed) == list(range(n))
