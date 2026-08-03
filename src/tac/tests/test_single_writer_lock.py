# SPDX-License-Identifier: MIT
"""ddm_sm1 2026-08-03 — the single-writer guard on resumable append-only stores.

EMPIRICAL ANCHOR (this is a fixed incident, not a hypothetical). Two copies of
``tools/sb1_seg_batch.py qa03`` wrote one JSONL concurrently and produced 28 rows /
24 unique with a broken pair-state chain. The trigger was an instrument defect: the
agent harness reported the job as ``failed exit code 144`` (SIGURG) while the Python
child kept running, so relaunching -- the correct response to "your job died" --
created the second writer.

The procedural guard ("check the process table first") was then written down AND
FOLLOWED, and still returned a false negative twice: ``ps`` came back empty while the
process was demonstrably alive and 17 minutes into its run. This lock is what actually
caught it. That is the design argument these tests protect: liveness must be enforced
by something that does not depend on a reader correctly interpreting a signal.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from tac.single_writer_lock import (
    LOCK_BASENAME,
    SingleWriterLockError,
    read_lock_holder,
    single_writer_lock,
)


def test_second_writer_in_the_same_process_is_refused(tmp_path: Path):
    with (
        single_writer_lock(tmp_path, label="first"),
        pytest.raises(SingleWriterLockError),
        single_writer_lock(tmp_path, label="second"),
    ):
        pytest.fail("a second writer was admitted")


def test_lock_is_released_and_reacquirable(tmp_path: Path):
    with single_writer_lock(tmp_path, label="a"):
        pass
    with single_writer_lock(tmp_path, label="b"):
        pass  # no exception


def test_refusal_names_the_holder_and_says_how_to_check(tmp_path: Path):
    """A refusal a reader cannot act on becomes a --force flag within the week."""
    with (
        single_writer_lock(tmp_path, label="holder-job"),
        pytest.raises(SingleWriterLockError) as exc,
        single_writer_lock(tmp_path, label="contender"),
    ):
        pass
    msg = str(exc.value)
    assert "holder-job" in msg
    assert str(os.getpid()) in msg
    assert "ps " in msg, "must tell the operator how to verify liveness"
    assert "not evidence the job died" in msg, "must state the law that caused the incident"


def test_holder_record_identifies_the_owner(tmp_path: Path):
    with single_writer_lock(tmp_path, label="my-job"):
        rec = read_lock_holder(tmp_path)
    assert rec is not None
    assert rec["label"] == "my-job"
    assert rec["pid"] == os.getpid()
    assert rec["host"] and rec["started_utc"]


def test_holder_record_is_diagnostic_only_not_authority(tmp_path: Path):
    """A stale record must NOT block a new writer -- flock is the authority.

    kill -9 leaves the record behind. If presence of the file were the test, every
    hard-killed job would permanently poison its own output directory and the guard
    would be removed the first time that happened.
    """
    (tmp_path / LOCK_BASENAME).write_text(json.dumps(
        {"label": "ghost", "pid": 999999, "host": "gone", "started_utc": "2026-01-01T00:00:00Z"}))
    with single_writer_lock(tmp_path, label="live"):
        assert read_lock_holder(tmp_path)["label"] == "live"


def test_unreadable_holder_record_does_not_crash_acquisition(tmp_path: Path):
    (tmp_path / LOCK_BASENAME).write_text("{not json")
    assert read_lock_holder(tmp_path) is None
    with single_writer_lock(tmp_path, label="live"):
        pass


def test_skip_bypasses_entirely_and_yields_none(tmp_path: Path):
    with single_writer_lock(tmp_path, label="x", skip=True) as p:
        assert p is None
    assert not (tmp_path / LOCK_BASENAME).exists()


def test_creates_the_directory_if_absent(tmp_path: Path):
    d = tmp_path / "nested" / "out"
    with single_writer_lock(d, label="x"):
        assert d.is_dir()


def _child_tries_to_lock(d: str, q) -> None:  # pragma: no cover - runs in a subprocess
    try:
        with single_writer_lock(Path(d), label="child"):
            q.put("ACQUIRED")
    except SingleWriterLockError:
        q.put("REFUSED")
    except Exception as exc:
        q.put(f"ERROR:{exc!r}")


def test_a_genuinely_separate_process_is_refused(tmp_path: Path):
    """The real incident was cross-process; an in-process test would not have caught it."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with single_writer_lock(tmp_path, label="parent"):
        p = ctx.Process(target=_child_tries_to_lock, args=(str(tmp_path), q))
        p.start()
        p.join(60)
        assert q.get(timeout=10) == "REFUSED"


def test_separate_process_acquires_once_the_holder_exits(tmp_path: Path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with single_writer_lock(tmp_path, label="parent"):
        pass  # released
    p = ctx.Process(target=_child_tries_to_lock, args=(str(tmp_path), q))
    p.start()
    p.join(60)
    assert q.get(timeout=10) == "ACQUIRED"


def test_lock_released_even_when_the_body_raises(tmp_path: Path):
    with pytest.raises(ValueError), single_writer_lock(tmp_path, label="boom"):
        raise ValueError("body failed")
    with single_writer_lock(tmp_path, label="after"):
        pass


def test_both_seg_solver_entrypoints_take_the_lock():
    """Regression on the WIRING: an unused guard is the orphan-grade defect (m55)."""
    root = Path(__file__).resolve().parents[3]
    for rel in ("tools/sb1_seg_batch.py", "tools/sm1_seg_search_probe.py"):
        src = (root / rel).read_text()
        assert "single_writer_lock(" in src, f"{rel} does not acquire the lock"
        assert "--skip-writer-lock" in src, f"{rel} lacks the documented bypass"
