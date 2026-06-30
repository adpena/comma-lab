# SPDX-License-Identifier: MIT
"""NO-SILENT-FAILURE launch-path hardening tests (operator 2026-06-30:
"Ensure all errors provide detailed debugging information no silent failures").

Covers the bug class: ``spawn_durable_daemon`` printed a healthy
"pid=... (detached session)" line + returned 0 even though the ``safe_run`` child
had ALREADY died at exec (an unquoted shell expansion folded the whole command
into a single argv[0] -> ``failed to start <cmd[0]>``). The optimistic success
report MASKED the immediate failure.

These tests prove:
  * ``spawn_durable_daemon._do_start`` now VERIFIES the child survived exec — a
    dead child -> nonzero exit + detailed debug; a live child -> success.
  * ``safe_run`` spawn-failure messages carry len(cmd) / cmd[:3] / cwd and the
    collapse-to-one-arg HINT.

Every spawned process here uses a THROWAWAY label + a trivial sleep/false command
in a tmp dir; none of the live lab infra (n600 training, dashboard, tunnel) is
touched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, str(_REPO / rel))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


sd = _load("tac_spawn_durable_daemon_under_test_nsf", "tools/spawn_durable_daemon.py")
sr = _load("tac_safe_run_under_test_nsf", "tools/safe_run.py")


def _start_args(**kw):
    base = dict(
        cmd=kw.pop("cmd"),
        log=kw.pop("log"),
        label=kw.pop("label", "test_throwaway"),
        skip_mem_preflight=True,
        min_free_gb=0.0,
        projected_gb=0.0,
        rss_cap_mb=None,
        walltime_cap_s=None,
        verify_s=kw.pop("verify_s", 2.0),
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ───────────────────────── unit: _verify_child_survived ─────────────────────────
class _FakePoll:
    """Minimal Popen stand-in with a scripted poll() sequence + pid."""

    def __init__(self, codes, pid=10_000_001):
        self._codes = list(codes)
        self.pid = pid

    def poll(self):
        if self._codes:
            return self._codes.pop(0)
        return None


def test_verify_running_child_is_ok(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("[durable-daemon] launching: foo\n")
    # poll() always None -> still running -> healthy daemon
    proc = _FakePoll([None, None, None])
    ok, info = sd._verify_child_survived(proc, pgid=proc.pid, log_path=str(log), verify_s=0.3)
    assert ok is True
    assert info["code"] is None


def test_verify_nonzero_exit_is_failure(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("safe_run.py: failed to start 'whatever'\n")
    proc = _FakePoll([125])  # exited rc=125 during the window
    ok, info = sd._verify_child_survived(proc, pgid=proc.pid, log_path=str(log), verify_s=1.0)
    assert ok is False
    assert info["code"] == 125
    assert "exited rc=125" in info["exit_str"]


def test_verify_clean_exit_with_failure_marker_is_failure(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("Traceback (most recent call last):\n  boom\n")
    proc = _FakePoll([0])  # rc=0 BUT log shows a failure -> contradiction -> fail
    ok, info = sd._verify_child_survived(proc, pgid=proc.pid, log_path=str(log), verify_s=1.0)
    assert ok is False
    assert info["failures"]


def test_verify_clean_exit_no_failure_is_ok(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("[durable-daemon] launching: /bin/true\nall good\n")
    proc = _FakePoll([0])
    ok, info = sd._verify_child_survived(proc, pgid=proc.pid, log_path=str(log), verify_s=1.0)
    assert ok is True


def test_scan_log_for_failures_detects_markers(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(
        "normal line\n"
        "safe_run.py: failed to start 'a b c': command not found\n"
        "another normal line\n"
    )
    hits = sd._scan_log_for_failures(str(log))
    assert len(hits) == 1
    assert "failed to start" in hits[0]


def test_decode_exit_signal_and_code():
    assert sd._decode_exit(None) == "still running"
    assert "rc=3" in sd._decode_exit(3)
    assert "signal" in sd._decode_exit(-signal.SIGKILL)


# ───────────────────────── integration: _do_start verification ─────────────────────────
def test_do_start_direct_exec_failure_returns_nonzero_with_detail(tmp_path, monkeypatch, capsys):
    """Collapsed argv[0] (unquoted shell expansion) with NO safe_run wrapper:
    _do_start's own Popen raises -> MUST return nonzero + detailed debug (cmd[0],
    cwd, HINT) instead of a traceback or a phantom-healthy line."""
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    log = tmp_path / "dead.log"
    a = _start_args(
        cmd=["--", "this is one collapsed command --with --flags"],
        log=str(log), label="test_dead_direct", verify_s=2.0,
    )
    rc = sd._do_start(a)
    assert rc == 4, "exec-failed launch must EXIT NONZERO, not crash or report healthy"
    err = capsys.readouterr().err
    assert "failed to start child" in err
    assert "cmd[0]" in err and "cwd" in err and "HINT" in err
    assert "test_dead_direct" in err
    # the failure is ALSO written into the daemon log (no silent failure)
    assert "ERROR: failed to start child" in log.read_text()


def test_do_start_safe_run_wrapped_dead_child_returns_nonzero(tmp_path, monkeypatch, capsys):
    """The EXACT production bug: spawn wraps a bad command in safe_run; safe_run's
    Popen of the bad inner cmd fails -> safe_run exits 125. spawn's verification
    MUST catch the dead child -> nonzero + 'DID NOT survive' + registry stopped."""
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    log = tmp_path / "wrapdead.log"
    a = _start_args(
        cmd=["--", "/nonexistent/binary_xyz_throwaway"],
        log=str(log), label="test_dead_wrapped", verify_s=3.0,
        rss_cap_mb=4000, walltime_cap_s=120,  # forces safe_run wrap (layer 3)
    )
    rc = sd._do_start(a)
    assert rc == 4, "safe_run-wrapped dead child must EXIT NONZERO"
    err = capsys.readouterr().err
    assert "DID NOT survive" in err
    assert "cmd[0]" in err and "cwd" in err and "log tail" in err
    assert "test_dead_wrapped" in err
    rows = json.loads(reg.read_text())
    rec = next(r for r in rows if r["label"] == "test_dead_wrapped")
    assert rec["status"] == "stopped"
    assert "died_at_exec" in rec.get("stopped_reason", "")


def test_do_start_live_child_succeeds(tmp_path, monkeypatch, capsys):
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    log = tmp_path / "live.log"
    a = _start_args(cmd=["--", "/bin/sleep", "30"], log=str(log),
                    label="test_live_throwaway", verify_s=1.5)
    pid = None
    try:
        rc = sd._do_start(a)
        assert rc == 0, "a genuinely live daemon must report success"
        out = capsys.readouterr().out
        assert "VERIFIED alive" in out
        rows = json.loads(reg.read_text())
        rec = next(r for r in rows if r["label"] == "test_live_throwaway")
        assert rec["status"] == "running"
        pid = int(rec["pid"])
        assert sd._pid_alive(pid)
    finally:
        if pid:
            import contextlib
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)


def test_do_start_verify_zero_skips_wait_back_compat(tmp_path, monkeypatch):
    """verify_s<=0 returns instantly (back-compat); a live child still succeeds."""
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    log = tmp_path / "skip.log"
    a = _start_args(cmd=["--", "/bin/sleep", "20"], log=str(log),
                    label="test_skip_throwaway", verify_s=0.0)
    pid = None
    t0 = time.time()
    try:
        rc = sd._do_start(a)
        assert rc == 0
        assert time.time() - t0 < 1.5, "verify_s=0 must not block"
        rows = json.loads(reg.read_text())
        pid = int(next(r for r in rows if r["label"] == "test_skip_throwaway")["pid"])
    finally:
        if pid:
            import contextlib
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)


# ───────────────────────── safe_run detail ─────────────────────────
def test_safe_run_spawn_debug_collapse_hint():
    d = sr._spawn_debug(["this is one big collapsed command"])
    assert "len(cmd)=1" in d
    assert "collapsed" in d.lower()
    assert "cwd=" in d


def test_safe_run_spawn_debug_no_hint_for_normal_argv():
    d = sr._spawn_debug(["/bin/echo", "hi"])
    assert "len(cmd)=2" in d
    assert "collapsed" not in d.lower()


def test_safe_run_command_not_found_detailed(capsys):
    rc = sr.main(["--", "this is one collapsed command --with --flags"])
    assert rc == sr.EXIT_SPAWN
    err = capsys.readouterr().err
    assert "failed to start" in err
    assert "len(cmd)=1" in err
    assert "cwd=" in err
    assert "collapsed" in err.lower()


def test_safe_run_no_command_detailed(capsys):
    rc = sr.main(["--rss-mb", "100"])  # no command after options
    assert rc == sr.EXIT_SPAWN
    err = capsys.readouterr().err
    assert "no command given" in err
    assert "cwd=" in err


def test_safe_run_valid_quick_command_succeeds(capsys):
    rc = sr.main(["--quiet", "--", "/usr/bin/true"])
    assert rc == 0
