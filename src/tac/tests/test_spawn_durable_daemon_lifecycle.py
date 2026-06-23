# SPDX-License-Identifier: MIT
"""NO-FAKE lifecycle tests for tools/spawn_durable_daemon.py.

Headline (the bug class this PERMANENTLY fixes): a detached daemon's process
GROUP must be killed as a whole so the worker AND its children die together —
NO ORPHAN. These tests spawn REAL throwaway dummy daemons (a python parent that
spawns a child subprocess then loops) and assert that --stop kills BOTH the
parent and the child. If --stop only killed the group LEADER and orphaned the
child, ``test_stop_kills_whole_group_no_orphan`` would FAIL.

Every dummy daemon here is labeled ``test_*`` and torn down in a finally block;
none of the live lab infra (jump run, dashboard renderer, http.server,
cloudflared, monitor) is touched.
"""
from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HELPER_PATH = _REPO_ROOT / "tools" / "spawn_durable_daemon.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("spawn_durable_daemon_under_test", str(_HELPER_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


SDD = _load_helper()


@pytest.fixture()
def isolated_registry(tmp_path, monkeypatch):
    """Point the helper's registry + lock at a throwaway dir (no live infra)."""
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(SDD, "_REGISTRY_PATH", reg, raising=True)
    monkeypatch.setattr(SDD, "_REGISTRY_LOCK", lock, raising=True)
    SDD._registry_lock_depth = 0
    return reg, lock


def _pid_alive(pid: int) -> bool:
    # Delegate to the module's reap-aware probe so own-child zombies (the
    # in-process test artifact) are reported as dead, not falsely alive.
    return SDD._pid_alive(pid)


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def _wait_file_nonempty(path: Path, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.read_text():
            return True
        time.sleep(0.05)
    return path.exists() and bool(path.read_text())


# Dummy worker: a parent python that spawns a child python subprocess (so the
# process GROUP has >1 member), writes both pids to CHILD_PID_FILE, then loops.
_DUMMY_WORKER = r"""
import os, subprocess, sys, time
child_pid_file = os.environ["CHILD_PID_FILE"]
# Spawn a child that just sleeps in a loop — it inherits our process group.
child = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.2)"])
with open(child_pid_file, "w") as f:
    f.write("%d\n%d\n" % (os.getpid(), child.pid))
    f.flush()
while True:
    time.sleep(0.2)
"""


def _spawn_dummy(SDD_mod, tmp_path, label: str):
    """Spawn the dummy daemon via the helper; return (parent_pid, child_pid)."""
    log = tmp_path / f"{label}.log"
    # Unique per spawn so a re-spawn (upsert test) never reads a stale pids file.
    child_pid_file = tmp_path / f"{label}.{time.time_ns()}.pids"
    os.environ["CHILD_PID_FILE"] = str(child_pid_file)
    try:
        argv = ["--log", str(log), "--label", label, "--", sys.executable, "-c", _DUMMY_WORKER]
        rc = SDD_mod.main(argv)
        assert rc == 0
        assert _wait_file_nonempty(child_pid_file, timeout=8.0), "dummy worker never wrote its pids"
        parent_pid, child_pid = (int(x) for x in child_pid_file.read_text().split())
        return parent_pid, child_pid
    finally:
        os.environ.pop("CHILD_PID_FILE", None)


# --------------------------------------------------------------------------
# HEADLINE: no-orphan
# --------------------------------------------------------------------------
def test_stop_kills_whole_group_no_orphan(isolated_registry, tmp_path):
    label = "test_no_orphan"
    parent_pid = child_pid = None
    try:
        parent_pid, child_pid = _spawn_dummy(SDD, tmp_path, label)
        assert _pid_alive(parent_pid), "parent should be alive after spawn"
        assert _pid_alive(child_pid), "child should be alive after spawn"

        rc = SDD._do_stop(label)
        assert rc == 0, "stop should succeed"

        # The decisive assertions: BOTH dead. If stop orphaned the child, the
        # second assertion fails.
        assert _wait_dead(parent_pid), "parent must be dead after --stop"
        assert _wait_dead(child_pid), "child must be dead after --stop (NO ORPHAN)"
    finally:
        for pid in (child_pid, parent_pid):
            if pid and _pid_alive(pid):
                with pytest.raises(Exception) if False else _suppress():
                    os.kill(pid, signal.SIGKILL)


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return True


def test_child_outlives_leader_only_kill_but_dies_on_group_kill(isolated_registry, tmp_path):
    """Sanity-of-mechanism: killing ONLY the leader pid (the naive/buggy path)
    leaves the child alive — proving the orphan is real — then the group kill
    via --stop reaps it. This documents WHY --stop must use killpg."""
    label = "test_orphan_proof"
    parent_pid = child_pid = None
    try:
        parent_pid, child_pid = _spawn_dummy(SDD, tmp_path, label)
        # Naive kill of just the leader pid (NOT the group):
        os.kill(parent_pid, signal.SIGKILL)
        # The helper Popen'd the leader in-process, so reap its zombie so the
        # liveness probe is truthful (in production the launcher has already
        # exited and init reaps the leader — this reap is a test-only artifact).
        end = time.time() + 5.0
        while time.time() < end and _pid_alive(parent_pid):
            SDD._try_reap(parent_pid)
            time.sleep(0.05)
        assert _wait_dead(parent_pid)
        # The child is now an orphan and STILL ALIVE (reparented to init):
        assert _pid_alive(child_pid), "child should be orphaned & alive after leader-only kill"
        # Now the canonical --stop group-kill reaps it (pgid == old leader pid).
        rc = SDD._do_stop(label)
        assert rc == 0
        assert _wait_dead(child_pid), "group kill must reap the orphan"
    finally:
        for pid in (child_pid, parent_pid):
            if pid and _pid_alive(pid):
                with _suppress():
                    os.kill(pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# Registry round-trip + status transition
# --------------------------------------------------------------------------
def test_registry_round_trip_start_then_stop(isolated_registry, tmp_path):
    reg, _lock = isolated_registry
    label = "test_round_trip"
    parent_pid = child_pid = None
    try:
        parent_pid, child_pid = _spawn_dummy(SDD, tmp_path, label)
        rows = SDD._load_registry()
        entry = [r for r in rows if r["label"] == label]
        assert len(entry) == 1
        r = entry[0]
        assert r["status"] == "running"
        assert r["pid"] == parent_pid
        assert r["pgid"] == parent_pid  # session leader: pgid == pid
        assert r["log"].endswith(f"{label}.log")
        assert r.get("started_utc")
        assert r["cwd"] == os.getcwd()

        SDD._do_stop(label)
        rows2 = SDD._load_registry()
        r2 = next(x for x in rows2 if x["label"] == label)
        assert r2["status"] == "stopped"
        assert "stopped_utc" in r2
    finally:
        for pid in (child_pid, parent_pid):
            if pid and _pid_alive(pid):
                with _suppress():
                    os.kill(pid, signal.SIGKILL)


def test_status_reports_live_then_dead(isolated_registry, tmp_path, capsys):
    label = "test_status"
    parent_pid = child_pid = None
    try:
        parent_pid, child_pid = _spawn_dummy(SDD, tmp_path, label)
        SDD._do_status()
        out = capsys.readouterr().out
        assert label in out
        assert "actual=LIVE" in out

        SDD._do_stop(label)
        SDD._do_status()
        out2 = capsys.readouterr().out
        assert label in out2
        assert "actual=DEAD" in out2
    finally:
        for pid in (child_pid, parent_pid):
            if pid and _pid_alive(pid):
                with _suppress():
                    os.kill(pid, signal.SIGKILL)


def test_start_upserts_same_label(isolated_registry, tmp_path):
    """Re-running the same label replaces its row (no duplicate)."""
    label = "test_upsert"
    pids = []
    try:
        p1, c1 = _spawn_dummy(SDD, tmp_path, label)
        pids += [p1, c1]
        # Stop the first so we don't leak it, then re-spawn same label.
        SDD._do_stop(label)
        p2, c2 = _spawn_dummy(SDD, tmp_path, label)
        pids += [p2, c2]
        rows = SDD._load_registry()
        entries = [r for r in rows if r["label"] == label]
        assert len(entries) == 1, "same label must upsert, not duplicate"
        assert entries[0]["pid"] == p2
        assert entries[0]["status"] == "running"
        SDD._do_stop(label)
    finally:
        for pid in pids:
            if pid and _pid_alive(pid):
                with _suppress():
                    os.kill(pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_stop_unknown_label_returns_1(isolated_registry):
    rc = SDD._do_stop("test_does_not_exist")
    assert rc == 1


def test_stop_already_stopped_is_noop_rc0(isolated_registry, tmp_path):
    label = "test_already_stopped"
    parent_pid = child_pid = None
    try:
        parent_pid, child_pid = _spawn_dummy(SDD, tmp_path, label)
        assert SDD._do_stop(label) == 0
        # second stop: already marked stopped -> noop rc 0
        assert SDD._do_stop(label) == 0
    finally:
        for pid in (child_pid, parent_pid):
            if pid and _pid_alive(pid):
                with _suppress():
                    os.kill(pid, signal.SIGKILL)


def test_status_empty_registry(isolated_registry, capsys):
    SDD._do_status()
    out = capsys.readouterr().out
    assert "registry empty" in out


def test_start_without_label_synthesizes(isolated_registry, tmp_path):
    log = tmp_path / "nolabel.log"
    pids = []
    try:
        rc = SDD.main(["--log", str(log), "--", sys.executable, "-c", "import time\nwhile True: time.sleep(0.2)"])
        assert rc == 0
        rows = SDD._load_registry()
        assert len(rows) == 1
        label = rows[0]["label"]
        assert label.startswith("python") or "pid" in label
        pids.append(rows[0]["pid"])
        SDD._do_stop(label)
    finally:
        for pid in pids:
            if pid and _pid_alive(pid):
                with _suppress():
                    os.kill(pid, signal.SIGKILL)


def test_start_requires_log():
    rc = SDD.main([])
    assert rc == 2


def test_start_no_command_returns_2(tmp_path):
    rc = SDD.main(["--log", str(tmp_path / "x.log"), "--label", "test_nocmd"])
    assert rc == 2


def test_save_registry_outside_lock_raises(isolated_registry):
    SDD._registry_lock_depth = 0
    with pytest.raises(RuntimeError):
        SDD._save_registry_atomic([{"label": "x"}])


def test_load_registry_lenient_on_corrupt(isolated_registry):
    reg, _lock = isolated_registry
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text("{not json", encoding="utf-8")
    assert SDD._load_registry() == []
    reg.write_text('{"a": 1}', encoding="utf-8")  # non-list root
    assert SDD._load_registry() == []


def test_pgid_alive_false_for_dead(isolated_registry):
    assert SDD._pgid_alive(0) is False
    assert SDD._pid_alive(0) is False
