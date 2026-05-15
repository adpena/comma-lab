# SPDX-License-Identifier: MIT
"""OP-7 fix tests (codex chunk 5, 2026-05-15): thread-safety of
``_active_jobs_lock_depth`` in ``tac.deploy.lightning.active_jobs_state``.

Pre-fix the depth counter was a module-level int shared across threads.
Two threads in the same process could observe ``depth>0`` simultaneously
and both skip the fcntl acquire (the 0->1 transition), then run the
critical section concurrently and silently drop each other's
register/upsert/mark_terminal calls.

Post-fix the depth counter is ``threading.local()``; each thread has
its own counter. Re-entry within the SAME thread still short-circuits
(no deadlock); a DIFFERENT thread sees its own depth=0 and proceeds to
fcntl-acquire. ``fcntl.flock(LOCK_EX)`` blocks the second thread until
the first releases.

Sister of Catalog #131 (`check_no_bare_writes_to_shared_state`) +
Catalog #140 (`check_state_writers_own_their_lock_end_to_end`).

Memory: feedback_op4_7_8_batch_paired_env_lock_depth_n_artifacts_landed_20260515.md.
"""
from __future__ import annotations

import threading
import time

from tac.deploy.lightning import active_jobs_state as ajs

# ──────────────────────────────────────────────────────────────────────────
# Helper-API regression
# ──────────────────────────────────────────────────────────────────────────


def test_op7_default_lock_depth_zero_in_fresh_thread():
    """A fresh thread sees thread-local depth = 0 even if other threads
    have set their own depth."""

    seen: list[int] = []

    def reader_thread() -> None:
        seen.append(ajs._get_active_jobs_lock_depth())

    # Set main-thread depth artificially.
    ajs._set_active_jobs_lock_depth(5)
    try:
        t = threading.Thread(target=reader_thread)
        t.start()
        t.join()
        assert seen == [0], "fresh thread MUST observe its own depth=0"
        # Main thread retains its own counter.
        assert ajs._get_active_jobs_lock_depth() == 5
    finally:
        ajs._set_active_jobs_lock_depth(0)


def test_op7_active_jobs_lock_held_is_thread_local():
    """``_active_jobs_lock_held()`` queries THIS thread's counter."""

    sibling_held: list[bool] = []

    def sibling_thread() -> None:
        sibling_held.append(ajs._active_jobs_lock_held())

    ajs._set_active_jobs_lock_depth(3)
    try:
        assert ajs._active_jobs_lock_held() is True
        t = threading.Thread(target=sibling_thread)
        t.start()
        t.join()
        # Sibling MUST NOT see main's depth.
        assert sibling_held == [False]
    finally:
        ajs._set_active_jobs_lock_depth(0)


# ──────────────────────────────────────────────────────────────────────────
# Lock-context-manager re-entry within the same thread
# ──────────────────────────────────────────────────────────────────────────


def test_op7_same_thread_reentry_short_circuits(tmp_path, monkeypatch):
    """Same-thread re-entry MUST be counted (depth>1) without re-fcntl."""

    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)

    with ajs._active_jobs_lock(lock):
        assert ajs._get_active_jobs_lock_depth() == 1
        assert ajs._active_jobs_lock_held() is True
        with ajs._active_jobs_lock(lock):
            assert ajs._get_active_jobs_lock_depth() == 2
            with ajs._active_jobs_lock(lock):
                assert ajs._get_active_jobs_lock_depth() == 3
            assert ajs._get_active_jobs_lock_depth() == 2
        assert ajs._get_active_jobs_lock_depth() == 1
    # After exiting all nested levels, depth returns to 0 in this thread.
    assert ajs._get_active_jobs_lock_depth() == 0
    assert ajs._active_jobs_lock_held() is False


# ──────────────────────────────────────────────────────────────────────────
# Lock-held assertion respects thread boundaries (Catalog #140 sister)
# ──────────────────────────────────────────────────────────────────────────


def test_op7_save_active_jobs_refuses_bare_call_from_sibling_thread(
    tmp_path, monkeypatch
):
    """A sibling thread that enters ``_save_active_jobs`` WITHOUT first
    acquiring the lock MUST be refused with RuntimeError per Catalog #140
    contract — even when another thread holds the lock."""

    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)

    sibling_exception: list[BaseException] = []
    sibling_started = threading.Event()
    main_can_release = threading.Event()

    def sibling() -> None:
        sibling_started.set()
        try:
            ajs._save_active_jobs([{"job_name": "from_sibling"}])
        except BaseException as exc:
            sibling_exception.append(exc)

    with ajs._active_jobs_lock(lock):
        # Main thread holds the lock; spawn sibling that tries to bare-write.
        t = threading.Thread(target=sibling)
        t.start()
        sibling_started.wait(timeout=5.0)
        # Give sibling time to attempt the bare write.
        time.sleep(0.05)
        main_can_release.set()
        t.join(timeout=5.0)

    # Sibling MUST be refused per Catalog #140 + OP-7 thread-local discipline.
    assert len(sibling_exception) == 1, (
        "Sibling thread should have raised RuntimeError; instead got: "
        f"{sibling_exception}"
    )
    msg = str(sibling_exception[0])
    assert isinstance(sibling_exception[0], RuntimeError)
    assert "_save_active_jobs called WITHOUT holding _active_jobs_lock" in msg


# ──────────────────────────────────────────────────────────────────────────
# Multi-thread stress: concurrent register_job invocations from N threads
# all survive (no silent drops) per OP-7 thread-local discipline.
# ──────────────────────────────────────────────────────────────────────────


def test_op7_multi_thread_register_no_silent_drops(tmp_path, monkeypatch):
    """4-thread × 5-row stress: 20 register_job invocations across 4 threads
    must ALL survive in the final active-jobs file. Pre-fix, threads sharing
    the module-level int counter could bypass the fcntl acquire and silently
    drop each other's appends."""

    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)

    n_threads = 4
    rows_per_thread = 5
    barrier = threading.Barrier(n_threads)

    def worker(thread_id: int) -> None:
        barrier.wait()  # release all threads at once
        for i in range(rows_per_thread):
            ajs.register_job({"job_name": f"t{thread_id}_r{i}", "thread": thread_id})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    final_rows = ajs.load_active_jobs(p)
    assert len(final_rows) == n_threads * rows_per_thread, (
        f"Expected {n_threads * rows_per_thread} rows; got {len(final_rows)}. "
        "OP-7 regression: thread-local depth counter not applied → "
        "concurrent threads dropped each other's appends."
    )
    seen_names = {r["job_name"] for r in final_rows}
    expected_names = {
        f"t{tid}_r{rid}" for tid in range(n_threads) for rid in range(rows_per_thread)
    }
    assert seen_names == expected_names, (
        "Some rows were dropped; OP-7 thread-local discipline not enforced."
    )


def test_op7_concurrent_threads_serialize_at_fcntl(tmp_path, monkeypatch):
    """Two threads entering ``_active_jobs_lock`` MUST serialize: while
    thread A holds the lock, thread B blocks at fcntl until A releases.

    Verifies via timestamp ordering that critical sections do not overlap.
    """

    p = tmp_path / "active.json"
    lock = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", p)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock)

    timeline: list[tuple[str, str, float]] = []
    timeline_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(thread_id: str, hold_seconds: float) -> None:
        barrier.wait()
        with ajs._active_jobs_lock(lock):
            with timeline_lock:
                timeline.append((thread_id, "enter", time.monotonic()))
            time.sleep(hold_seconds)
            with timeline_lock:
                timeline.append((thread_id, "exit", time.monotonic()))

    t1 = threading.Thread(target=worker, args=("A", 0.1))
    t2 = threading.Thread(target=worker, args=("B", 0.1))
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Sort by timestamp to get global ordering.
    timeline_sorted = sorted(timeline, key=lambda r: r[2])
    # Critical sections must be non-overlapping: pattern is
    # [(X, enter), (X, exit), (Y, enter), (Y, exit)]
    assert len(timeline_sorted) == 4
    first_thread = timeline_sorted[0][0]
    assert timeline_sorted[0] == (first_thread, "enter", timeline_sorted[0][2])
    assert timeline_sorted[1][0] == first_thread
    assert timeline_sorted[1][1] == "exit"
    second_thread = timeline_sorted[2][0]
    assert second_thread != first_thread
    assert timeline_sorted[2][1] == "enter"
    assert timeline_sorted[3] == (second_thread, "exit", timeline_sorted[3][2])


# ──────────────────────────────────────────────────────────────────────────
# Module-level constant rename regression (no global int counter)
# ──────────────────────────────────────────────────────────────────────────


def test_op7_no_global_int_counter_remains():
    """Regression guard: ``_active_jobs_lock_depth`` MUST be a
    threading.local instance, NOT a module-level int.

    Catches accidental revert that would re-introduce the OP-7 bug class.
    """

    # The module-level attribute is now ``_active_jobs_lock_depth_tls``
    # of type threading.local; the bare ``_active_jobs_lock_depth`` int
    # MUST NOT exist (or if it does, MUST be a threading.local).
    bare = getattr(ajs, "_active_jobs_lock_depth", None)
    if bare is not None:
        # If the bare name still exists, it MUST be a threading.local
        # — never a plain int.
        assert isinstance(bare, threading.local), (
            "OP-7 regression: _active_jobs_lock_depth is a plain int again. "
            "Restore threading.local() per OP-7 fix."
        )
    tls = getattr(ajs, "_active_jobs_lock_depth_tls", None)
    assert tls is not None, "OP-7 fix removed the canonical TLS attribute"
    assert isinstance(tls, threading.local)
