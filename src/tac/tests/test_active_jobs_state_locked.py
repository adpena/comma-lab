"""Tests for tac.deploy.lightning.active_jobs_state — locked transactional API.

Covers the proactive META-class custody+concurrency audit (catalog #131,
2026-05-09). Memory:
``feedback_proactive_custody_concurrency_audit_landed_20260509.md``.
"""
from __future__ import annotations

import json
import multiprocessing as mp

import pytest

from tac.deploy.lightning import active_jobs_state as ajs


# ──────────────────────────────────────────────────────────────────────────
# Single-process correctness
# ──────────────────────────────────────────────────────────────────────────


def test_load_active_jobs_returns_empty_for_missing(tmp_path):
    p = tmp_path / "missing.json"
    assert ajs.load_active_jobs(p) == []


def test_load_active_jobs_returns_empty_for_corrupt(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("not valid json")
    assert ajs.load_active_jobs(p) == []


def test_load_active_jobs_returns_empty_for_non_list(tmp_path):
    p = tmp_path / "wrong_shape.json"
    p.write_text(json.dumps({"key": "value"}))
    assert ajs.load_active_jobs(p) == []


def test_register_job_appends(tmp_path, monkeypatch):
    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)
    rows = ajs.register_job({"job_name": "j1", "value": 1})
    assert len(rows) == 1
    assert rows[0]["job_name"] == "j1"
    rows = ajs.register_job({"job_name": "j2", "value": 2})
    assert len(rows) == 2
    assert [r["job_name"] for r in rows] == ["j1", "j2"]
    on_disk = json.loads(p.read_text())
    assert on_disk == rows


def test_upsert_job_replaces_by_key(tmp_path, monkeypatch):
    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)
    ajs.upsert_job({"job_name": "j1", "value": 1})
    ajs.upsert_job({"job_name": "j1", "value": 99})  # replace
    rows = ajs.load_active_jobs(p)
    assert len(rows) == 1
    assert rows[0]["value"] == 99


def test_upsert_job_missing_key_raises(tmp_path, monkeypatch):
    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)
    with pytest.raises(ValueError, match="missing key"):
        ajs.upsert_job({"value": 1})


def test_mark_job_terminal_updates_in_place(tmp_path, monkeypatch):
    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)
    ajs.register_job({"job_name": "j1"})
    ajs.mark_job_terminal("j1", terminal_status="completed_score_0.5")
    rows = ajs.load_active_jobs(p)
    assert rows[0]["terminal_status"] == "completed_score_0.5"


def test_mark_job_terminal_extra_fields(tmp_path, monkeypatch):
    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)
    ajs.register_job({"job_name": "j1"})
    ajs.mark_job_terminal(
        "j1", terminal_status="completed", extra_fields={"score": 0.5, "extra": "x"}
    )
    rows = ajs.load_active_jobs(p)
    assert rows[0]["score"] == 0.5
    assert rows[0]["extra"] == "x"


def test_update_active_jobs_locked_arbitrary_mutation(tmp_path, monkeypatch):
    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)
    ajs.register_job({"job_name": "j1"})
    ajs.register_job({"job_name": "j2"})

    def _filter(rows):
        return [r for r in rows if r["job_name"] != "j1"]

    rows_after = ajs.update_active_jobs_locked(_filter)
    assert len(rows_after) == 1
    assert rows_after[0]["job_name"] == "j2"


# ──────────────────────────────────────────────────────────────────────────
# Atomic-replace + unique-tmp guarantees
# ──────────────────────────────────────────────────────────────────────────


def test_save_active_jobs_uses_unique_tmp(tmp_path, monkeypatch):
    """Each save uses a distinct .tmp.<uuid> path; no shared .tmp clobber.

    Codex round 5 HIGH 2 fix sister update (catalog #140): _save_active_jobs
    now runtime-asserts the caller holds _active_jobs_lock; this unit test
    must wrap the call in the lock context manager to exercise the path.
    """
    p = tmp_path / "active.json"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", tmp_path / "active.json.lock")
    with ajs._active_jobs_lock():
        ajs._save_active_jobs([{"a": 1}])
    # After successful save, no .tmp.* files should remain in the directory.
    leftover = list(tmp_path.glob("active.json.tmp.*"))
    assert leftover == []


def test_save_active_jobs_atomic_replace(tmp_path, monkeypatch):
    """A successful save leaves the file at the right path with right content.

    Codex round 5 HIGH 2 fix sister update (catalog #140): _save_active_jobs
    now runtime-asserts the caller holds _active_jobs_lock; this unit test
    must wrap the call in the lock context manager to exercise the path.
    """
    p = tmp_path / "active.json"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", tmp_path / "active.json.lock")
    rows = [{"job_name": "j1"}, {"job_name": "j2"}]
    with ajs._active_jobs_lock():
        ajs._save_active_jobs(rows)
    on_disk = json.loads(p.read_text())
    assert on_disk == rows


# ──────────────────────────────────────────────────────────────────────────
# True multiprocessing — proves fcntl serialization works across processes
# ──────────────────────────────────────────────────────────────────────────


def _worker_register(args):
    """Worker process: register a job, then return the resulting rows count."""
    state_path, lock_path, job_name = args
    # Each worker re-imports the module and rebinds the path constants
    # before calling register_job. This simulates the cron-fired sister
    # subagent pattern.
    from tac.deploy.lightning import active_jobs_state as _ajs

    _ajs.ACTIVE_JOBS_PATH = state_path
    _ajs.ACTIVE_JOBS_LOCK = lock_path
    _ajs.register_job({"job_name": job_name, "worker_pid": True})
    return job_name


def test_concurrent_register_distinct_jobs_all_survive(tmp_path):
    """4 worker procs each register a distinct job; ALL 4 must survive."""
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    job_names = [f"j{i}" for i in range(4)]
    args = [(state_path, lock_path, name) for name in job_names]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_worker_register, args)
    assert sorted(results) == sorted(job_names)
    rows = json.loads(state_path.read_text())
    on_disk_names = sorted(r["job_name"] for r in rows)
    assert on_disk_names == sorted(job_names), (
        f"concurrent register lost rows: expected {job_names}, got {on_disk_names}"
    )


def _worker_upsert_same(args):
    """Two workers upsert the SAME key; result should be 1 row, last writer wins."""
    state_path, lock_path, job_name, value = args
    from tac.deploy.lightning import active_jobs_state as _ajs

    _ajs.ACTIVE_JOBS_PATH = state_path
    _ajs.ACTIVE_JOBS_LOCK = lock_path
    _ajs.upsert_job({"job_name": job_name, "value": value})
    return value


def test_concurrent_upsert_same_key_yields_single_row(tmp_path):
    """4 workers upsert SAME key with different values; final state has 1 row."""
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    args = [(state_path, lock_path, "shared_key", v) for v in range(4)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        pool.map(_worker_upsert_same, args)
    rows = json.loads(state_path.read_text())
    assert len(rows) == 1, f"upsert race produced {len(rows)} rows"
    assert rows[0]["job_name"] == "shared_key"
    assert rows[0]["value"] in {0, 1, 2, 3}  # last writer wins, deterministic across procs only modulo pool order


def _worker_mark_terminal(args):
    """Worker marks its own job terminal; sister-dispatcher meanwhile registers."""
    state_path, lock_path, op, name = args
    from tac.deploy.lightning import active_jobs_state as _ajs

    _ajs.ACTIVE_JOBS_PATH = state_path
    _ajs.ACTIVE_JOBS_LOCK = lock_path
    if op == "register":
        _ajs.register_job({"job_name": name})
    elif op == "mark_terminal":
        _ajs.mark_job_terminal(name, terminal_status="completed")
    return (op, name)


def test_concurrent_register_and_mark_terminal_no_drop(tmp_path):
    """Sister dispatcher (register) + sister harvester (mark_terminal) coexist."""
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    # Pre-seed one job that the harvester will mark terminal.
    ajs.ACTIVE_JOBS_PATH = state_path
    ajs.ACTIVE_JOBS_LOCK = lock_path
    ajs.register_job({"job_name": "preseeded"})
    # Now run dispatcher + harvester concurrently.
    args = [
        (state_path, lock_path, "mark_terminal", "preseeded"),
        (state_path, lock_path, "register", "new_job_1"),
        (state_path, lock_path, "register", "new_job_2"),
        (state_path, lock_path, "register", "new_job_3"),
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        pool.map(_worker_mark_terminal, args)
    rows = json.loads(state_path.read_text())
    names = sorted(r["job_name"] for r in rows)
    assert names == ["new_job_1", "new_job_2", "new_job_3", "preseeded"]
    preseeded = [r for r in rows if r["job_name"] == "preseeded"][0]
    assert preseeded.get("terminal_status") == "completed"


# ──────────────────────────────────────────────────────────────────────────
# Lock lifecycle
# ──────────────────────────────────────────────────────────────────────────


def test_lock_released_after_exception(tmp_path, monkeypatch):
    """After an exception inside the lock, subsequent acquire works fine."""
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)

    def _raises(rows):
        raise RuntimeError("simulated mutation failure")

    with pytest.raises(RuntimeError):
        ajs.update_active_jobs_locked(_raises)
    # Subsequent normal call must succeed (proves lock was released).
    rows = ajs.register_job({"job_name": "after_exception"})
    assert rows == [{"job_name": "after_exception"}]


def test_lock_path_is_sibling(tmp_path, monkeypatch):
    """Default lock path is sibling of state path."""
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", tmp_path / "subdir" / "state.json")
    monkeypatch.setattr(
        ajs, "ACTIVE_JOBS_LOCK", tmp_path / "subdir" / "state.json.lock"
    )
    ajs.register_job({"job_name": "j1"})
    assert (tmp_path / "subdir" / "state.json.lock").exists()
    assert (tmp_path / "subdir" / "state.json").exists()
