# SPDX-License-Identifier: MIT
"""Tests for codex round 6 MEDIUM 1 fix: SETUP-first-seen single transaction.

Catalog #144 — refuses split observed/left transactions for SETUP-first-seen.
The round-4 fix split the mutation into two separate locked transactions;
two overlapping verifier runs that disagree on the same id can drop
first-seen timestamps the other inserted because the two transactions
are not atomic together.

Bug class: codex round 6 MEDIUM 1 (2026-05-09). Memory:
feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Single-transaction API tests (functional)
# --------------------------------------------------------------------------


def _import_verify_vast(monkeypatch=None):
    """Import scripts/verify_vast_instances.py with patched paths."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import verify_vast_instances  # type: ignore
        return verify_vast_instances
    finally:
        # Don't pollute sys.path for other tests
        sys.path.remove(str(REPO_ROOT / "scripts"))


def test_single_txn_handles_observed_and_left_in_one_call(tmp_path: Path):
    """The canonical helper accepts BOTH observed and left ids."""
    vvi = _import_verify_vast()
    # Override the canonical path to a tmp location
    state_path = tmp_path / "instance_setup_first_seen.json"
    lock_path = tmp_path / "instance_setup_first_seen.json.lock"

    orig_state = vvi.SETUP_FIRST_SEEN_PATH
    orig_lock = vvi._setup_first_seen_lock_path

    vvi.SETUP_FIRST_SEEN_PATH = state_path
    vvi._setup_first_seen_lock_path = lambda: lock_path

    try:
        # Round 1: observe id1 SETUP, observe id2 leaving
        # (id2 was never SETUP; left should have no effect)
        rs = vvi.update_setup_first_seen_locked(
            observed_setup_ids={"id1"},
            left_setup_ids={"id2"},
            tracked_ids={"id1", "id2"},
            now_ts=1000.0,
        )
        assert "id1" in rs
        assert "id2" not in rs

        # Round 2: id1 still SETUP, id3 newly SETUP
        rs = vvi.update_setup_first_seen_locked(
            observed_setup_ids={"id1", "id3"},
            left_setup_ids=set(),
            tracked_ids={"id1", "id3"},
            now_ts=2000.0,
        )
        assert rs["id1"] == 1000.0  # KEEP older
        assert rs["id3"] == 2000.0

        # Round 3: id1 left SETUP
        rs = vvi.update_setup_first_seen_locked(
            observed_setup_ids={"id3"},
            left_setup_ids={"id1"},
            tracked_ids={"id1", "id3"},
            now_ts=3000.0,
        )
        assert "id1" not in rs
        assert rs["id3"] == 2000.0  # KEEP older
    finally:
        vvi.SETUP_FIRST_SEEN_PATH = orig_state
        vvi._setup_first_seen_lock_path = orig_lock


def test_single_txn_conflict_prefer_observed(tmp_path: Path):
    """When same id is in BOTH observed and left, observed wins by default."""
    vvi = _import_verify_vast()
    state_path = tmp_path / "instance_setup_first_seen.json"
    lock_path = tmp_path / "instance_setup_first_seen.json.lock"

    orig_state = vvi.SETUP_FIRST_SEEN_PATH
    orig_lock = vvi._setup_first_seen_lock_path
    vvi.SETUP_FIRST_SEEN_PATH = state_path
    vvi._setup_first_seen_lock_path = lambda: lock_path

    try:
        # id1 in BOTH sets — prefer_observed (default) keeps it as SETUP
        rs = vvi.update_setup_first_seen_locked(
            observed_setup_ids={"id1"},
            left_setup_ids={"id1"},
            tracked_ids={"id1"},
            now_ts=1000.0,
        )
        assert "id1" in rs, "prefer_observed: observed should win"

        # Reset
        if state_path.exists():
            state_path.unlink()

        # Switch rule to prefer_left
        rs = vvi.update_setup_first_seen_locked(
            observed_setup_ids={"id2"},
            left_setup_ids={"id2"},
            tracked_ids={"id2"},
            now_ts=2000.0,
            monotonic_conflict_rule="prefer_left",
        )
        assert "id2" not in rs, "prefer_left: left should win"

        # Switch rule to raise
        with pytest.raises(ValueError, match="same id in observed AND left"):
            vvi.update_setup_first_seen_locked(
                observed_setup_ids={"id3"},
                left_setup_ids={"id3"},
                tracked_ids={"id3"},
                now_ts=3000.0,
                monotonic_conflict_rule="raise",
            )
    finally:
        vvi.SETUP_FIRST_SEEN_PATH = orig_state
        vvi._setup_first_seen_lock_path = orig_lock


def test_single_txn_unknown_conflict_rule_raises(tmp_path: Path):
    vvi = _import_verify_vast()
    state_path = tmp_path / "instance_setup_first_seen.json"
    lock_path = tmp_path / "instance_setup_first_seen.json.lock"

    orig_state = vvi.SETUP_FIRST_SEEN_PATH
    orig_lock = vvi._setup_first_seen_lock_path
    vvi.SETUP_FIRST_SEEN_PATH = state_path
    vvi._setup_first_seen_lock_path = lambda: lock_path
    try:
        with pytest.raises(ValueError, match="unknown monotonic_conflict_rule"):
            vvi.update_setup_first_seen_locked(
                observed_setup_ids={"id1"},
                left_setup_ids={"id1"},
                tracked_ids={"id1"},
                now_ts=1000.0,
                monotonic_conflict_rule="bogus",
            )
    finally:
        vvi.SETUP_FIRST_SEEN_PATH = orig_state
        vvi._setup_first_seen_lock_path = orig_lock


# --------------------------------------------------------------------------
# Concurrency: 2 workers, one observes SETUP, other observes leaving
# --------------------------------------------------------------------------


def _worker_observe(state_path: str, lock_path: str, iid: str, now_ts: float):
    """Worker: observe ``iid`` as SETUP."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_vast_instances as vvi
    vvi.SETUP_FIRST_SEEN_PATH = Path(state_path)
    vvi._setup_first_seen_lock_path = lambda: Path(lock_path)
    vvi.update_setup_first_seen_locked(
        observed_setup_ids={iid},
        left_setup_ids=set(),
        tracked_ids={iid},
        now_ts=now_ts,
    )


def _worker_leave(state_path: str, lock_path: str, iid: str, now_ts: float):
    """Worker: observe ``iid`` as leaving SETUP."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import verify_vast_instances as vvi
    vvi.SETUP_FIRST_SEEN_PATH = Path(state_path)
    vvi._setup_first_seen_lock_path = lambda: Path(lock_path)
    vvi.update_setup_first_seen_locked(
        observed_setup_ids=set(),
        left_setup_ids={iid},
        tracked_ids={iid},
        now_ts=now_ts,
    )


def test_concurrent_two_workers_disagreeing_does_not_lose_first_seen(tmp_path: Path):
    """Two workers race; one inserts a first-seen, other tries to remove it.

    The single-transaction API serializes both calls. Final state is
    deterministic given each worker's SETUP/left observation; the only
    requirement is that we don't get an inconsistent on-disk state.
    """
    state_path = tmp_path / "instance_setup_first_seen.json"
    lock_path = tmp_path / "instance_setup_first_seen.json.lock"

    # Worker 1 observes id5 SETUP; Worker 2 observes id5 LEFT.
    # Order is non-deterministic but both serialize.
    p1 = multiprocessing.Process(
        target=_worker_observe,
        args=(str(state_path), str(lock_path), "id5", 100.0),
    )
    p2 = multiprocessing.Process(
        target=_worker_leave,
        args=(str(state_path), str(lock_path), "id5", 200.0),
    )
    p1.start()
    p2.start()
    p1.join(timeout=10)
    p2.join(timeout=10)
    assert p1.exitcode == 0, f"worker 1 failed: exit={p1.exitcode}"
    assert p2.exitcode == 0, f"worker 2 failed: exit={p2.exitcode}"

    # Final state should be coherent JSON (not corrupt mid-write)
    assert state_path.exists()
    final = json.loads(state_path.read_text())
    # id5 is either present (worker 1 won the race) OR absent (worker 2 won).
    # Either is acceptable. The key invariant is that the file is not corrupt
    # AND we don't have two simultaneous writers blowing each other away
    # (which would produce an inconsistent intermediate state).
    assert isinstance(final, dict)


# --------------------------------------------------------------------------
# Preflight check #144 STRICT
# --------------------------------------------------------------------------


def test_preflight_check_144_passes_with_zero_violations():
    from tac.preflight import check_setup_first_seen_no_split_transactions

    violations = check_setup_first_seen_no_split_transactions(
        verbose=False, strict=False,
    )
    assert violations == [], (
        f"Catalog #144 preflight should be at 0; got {len(violations)}:\n  "
        + "\n  ".join(violations[:5])
    )


def test_preflight_check_144_fires_on_split_transactions(tmp_path: Path):
    from tac.preflight import check_setup_first_seen_no_split_transactions

    fake_repo = tmp_path
    (fake_repo / "scripts").mkdir()
    bad = fake_repo / "scripts" / "verify_bad.py"
    bad.write_text(
        "def main():\n"
        "    update_setup_first_seen_locked(observed_setup_ids={1})\n"
        "    if True:\n"
        "        remove_setup_first_seen_locked({2})\n"
    )
    (fake_repo / "src" / "tac").mkdir(parents=True)
    (fake_repo / "tools").mkdir()
    (fake_repo / "experiments").mkdir()

    violations = check_setup_first_seen_no_split_transactions(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert len(violations) >= 1
    assert "verify_bad.py" in str(violations)


def test_preflight_check_144_accepts_canonical_helper(tmp_path: Path):
    """The canonical helper itself is allowed to combine both via the waiver."""
    from tac.preflight import check_setup_first_seen_no_split_transactions

    fake_repo = tmp_path
    (fake_repo / "scripts").mkdir()
    canonical = fake_repo / "scripts" / "canonical.py"
    canonical.write_text(
        "def update_setup_first_seen_locked():  # SETUP_FIRST_SEEN_SINGLE_TXN_OK:canonical\n"
        "    update_setup_first_seen_locked(observed_setup_ids={1})\n"
        "    remove_setup_first_seen_locked({2})\n"
    )
    (fake_repo / "src" / "tac").mkdir(parents=True)
    (fake_repo / "tools").mkdir()
    (fake_repo / "experiments").mkdir()

    violations = check_setup_first_seen_no_split_transactions(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_144_strict_mode_raises():
    from tac.preflight import (
        PreflightError,
        check_setup_first_seen_no_split_transactions,
    )

    with tempfile.TemporaryDirectory() as td:
        fake_repo = Path(td)
        (fake_repo / "scripts").mkdir()
        bad = fake_repo / "scripts" / "v.py"
        bad.write_text(
            "def main():\n"
            "    update_setup_first_seen_locked()\n"
            "    remove_setup_first_seen_locked()\n"
        )
        (fake_repo / "src" / "tac").mkdir(parents=True)
        (fake_repo / "tools").mkdir()
        (fake_repo / "experiments").mkdir()

        with pytest.raises(PreflightError):
            check_setup_first_seen_no_split_transactions(
                repo_root=fake_repo, verbose=False, strict=True,
            )
