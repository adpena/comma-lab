"""Task #525 — permanent codex spawn-kill fix: reaper-immune detached spawns.

ROOT CAUSE (MEASURED 2026-07-17): the operator fleet launchd agent
``com.vertigo.claude-code-reaper`` (60s cadence, GRACE=300s) SIGTERMs any process
matching ``\\b(claude|codex)\\b`` with no controlling terminal (ps tty == "??")
and dead stdin. The 2026-07-16 headless-launch change (ff0f884b35) removed the
Terminal TTY that had incidentally protected 150 prior arms → every codex arm
died rc=143 at ~5:20-5:55. Fix: the shared detached-spawn core allocates a
controlling pty via script(1) so the arm chain is classified as a live terminal
session (the reaper's own documented exclusion class).
"""
from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import pytest

from tools import codex_delegate, spawn_durable_daemon


# ---------------------------------------------------------------------------
# _pty_wrap — pure command construction
# ---------------------------------------------------------------------------

def test_pty_wrap_darwin_uses_bsd_script_form():
    out = spawn_durable_daemon._pty_wrap(["bash", "x.sh"], platform="darwin")
    assert out == ["/usr/bin/script", "-q", "/dev/null", "bash", "x.sh"]


def test_pty_wrap_linux_uses_util_linux_c_form():
    out = spawn_durable_daemon._pty_wrap(["bash", "x.sh"], platform="linux")
    assert out[0] == "script"
    assert "-c" in out
    # command is a single shell-joined string; transcript file is last
    assert out[out.index("-c") + 1] == "bash x.sh"
    assert out[-1] == "/dev/null"


def test_pty_wrap_defaults_to_current_platform():
    out = spawn_durable_daemon._pty_wrap(["true"])
    if sys.platform == "darwin":
        assert out[0] == "/usr/bin/script"
    else:
        assert out[0] == "script"


def test_pty_wrap_preserves_argument_order():
    out = spawn_durable_daemon._pty_wrap(["a", "b", "c"], platform="darwin")
    assert out[-3:] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# spawn_detached_verified — the shared core (real spawns; macOS + linux safe)
# ---------------------------------------------------------------------------

def test_spawn_detached_child_is_session_leader(tmp_path):
    log = tmp_path / "d.log"
    proc, pgid, ok, vinfo = spawn_durable_daemon.spawn_detached_verified(
        ["/bin/sleep", "2"], log, verify_s=0.5)
    try:
        assert ok, vinfo
        assert pgid == proc.pid  # setsid: own session/pgroup leader
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()


def test_spawn_detached_exec_failure_raises_oserror(tmp_path):
    log = tmp_path / "d.log"
    with pytest.raises((OSError, FileNotFoundError)):
        spawn_durable_daemon.spawn_detached_verified(
            ["/nonexistent/binary/xyz525"], log, verify_s=0.1)


def test_spawn_detached_early_nonzero_exit_is_not_ok(tmp_path):
    log = tmp_path / "d.log"
    proc, pgid, ok, vinfo = spawn_durable_daemon.spawn_detached_verified(
        ["/bin/bash", "-c", "exit 7"], log, verify_s=1.5)
    assert not ok
    assert vinfo["code"] == 7


def test_spawn_detached_clean_fast_exit_is_ok(tmp_path):
    log = tmp_path / "d.log"
    proc, pgid, ok, vinfo = spawn_durable_daemon.spawn_detached_verified(
        ["/usr/bin/true"], log, verify_s=1.0)
    assert ok, vinfo


def test_spawn_detached_verify_zero_skips_wait(tmp_path):
    log = tmp_path / "d.log"
    t0 = time.time()
    proc, pgid, ok, vinfo = spawn_durable_daemon.spawn_detached_verified(
        ["/bin/sleep", "1"], log, verify_s=0.0)
    assert time.time() - t0 < 0.9
    assert ok
    proc.terminate()


@pytest.mark.skipif(sys.platform != "darwin", reason="pty/tty semantics verified on macOS")
def test_spawn_detached_with_pty_child_has_controlling_tty(tmp_path):
    """THE fix invariant: with_pty=True gives the child chain a controlling
    terminal (ps tty != '??'), which exempts it from every reaper phase."""
    log = tmp_path / "d.log"
    proc, pgid, ok, vinfo = spawn_durable_daemon.spawn_detached_verified(
        ["/bin/bash", "-c", "ps -o tty= -p $$; sleep 0.2"], log,
        with_pty=True, verify_s=2.5)
    assert ok, vinfo
    for _ in range(40):
        text = log.read_text(errors="replace") if log.exists() else ""
        if "tty" in text:
            break
        time.sleep(0.1)
    assert "tty" in text and "??" not in text, f"child had no controlling tty: {text!r}"


@pytest.mark.skipif(sys.platform != "darwin", reason="pty/tty semantics verified on macOS")
def test_spawn_detached_without_pty_child_has_no_tty(tmp_path):
    """Counterfactual: the default (no pty) child shows tty '??' — the exact
    pre-fix reap signature. Guards against the test above passing vacuously."""
    log = tmp_path / "d.log"
    proc, pgid, ok, vinfo = spawn_durable_daemon.spawn_detached_verified(
        ["/bin/bash", "-c", "ps -o tty= -p $$"], log, with_pty=False, verify_s=2.5)
    assert ok, vinfo
    for _ in range(40):
        text = log.read_text(errors="replace") if log.exists() else ""
        if text.strip():
            break
        time.sleep(0.1)
    assert "??" in text, f"expected no controlling tty, got: {text!r}"


# ---------------------------------------------------------------------------
# codex_delegate wiring — the arm launch path uses the shared core with a pty
# ---------------------------------------------------------------------------

def _run_delegate_launch(tmp_path, monkeypatch, extra_args=()):
    """Drive codex_delegate.main() with the spawn core captured (no real launch)."""
    calls = {}

    def _fake_spawn(cmd, log_path, *, with_pty=False, verify_s=3.0, env=None, cwd=None):
        calls["cmd"] = list(cmd)
        calls["with_pty"] = with_pty
        calls["env"] = dict(env or {})
        calls["verify_s"] = verify_s

        class _P:  # minimal proc stub
            pid = 4242
        return _P(), 4242, True, {"code": None, "exit_str": "still running"}

    monkeypatch.setattr(spawn_durable_daemon, "spawn_detached_verified", _fake_spawn)
    monkeypatch.setattr(codex_delegate, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(codex_delegate, "EVENTS", tmp_path / "runs" / "codex_events.log")
    monkeypatch.setattr(codex_delegate, "LEDGER", tmp_path / "state" / "codex_delegations.jsonl")
    monkeypatch.setattr(codex_delegate, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(codex_delegate, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(codex_delegate, "WORKTREE_DIR", tmp_path / "worktrees")
    monkeypatch.setattr(codex_delegate, "_live_labels", lambda: [])
    monkeypatch.setattr(codex_delegate, "_uncommitted_pile_size", lambda: 0)
    monkeypatch.setattr(codex_delegate, "_live_nonisolated_writer_count", lambda: 0)
    prompt = tmp_path / "p.txt"
    prompt.write_text("tiny", encoding="utf-8")
    rc = codex_delegate.main([
        "--label", "unit525", "--prompt-file", str(prompt), *extra_args])
    return rc, calls


def test_delegate_launch_uses_pty_by_default(tmp_path, monkeypatch):
    rc, calls = _run_delegate_launch(tmp_path, monkeypatch)
    assert rc == 0
    assert calls["with_pty"] is True
    assert calls["cmd"][0] == "bash"
    assert calls["cmd"][1].endswith(".sh")


def test_delegate_launch_no_pty_flag_disables_wrap(tmp_path, monkeypatch):
    rc, calls = _run_delegate_launch(tmp_path, monkeypatch, extra_args=("--no-pty",))
    assert rc == 0
    assert calls["with_pty"] is False


def test_delegate_launch_sets_term_dumb_for_log_hygiene(tmp_path, monkeypatch):
    rc, calls = _run_delegate_launch(tmp_path, monkeypatch)
    assert calls["env"].get("TERM") == "dumb"


def test_delegate_no_launch_never_touches_spawn_core(tmp_path, monkeypatch):
    rc, calls = _run_delegate_launch(tmp_path, monkeypatch, extra_args=("--no-launch",))
    assert rc == 0
    assert "cmd" not in calls  # spawn core untouched


def test_delegate_source_has_no_bare_popen_launch_path():
    """The single-impl invariant: codex_delegate must not keep a second bare
    Popen(start_new_session=...) spawn path beside the shared core."""
    src = Path(codex_delegate.__file__).read_text(encoding="utf-8")
    assert "spawn_detached_verified" in src
    assert "subprocess.Popen" not in src, (
        "codex_delegate grew a second bare Popen spawn path; use "
        "spawn_durable_daemon.spawn_detached_verified (task #525 single impl)")


def test_delegate_launch_verifies_alive(tmp_path, monkeypatch):
    rc, calls = _run_delegate_launch(tmp_path, monkeypatch)
    assert calls["verify_s"] and calls["verify_s"] > 0


def test_durable_daemon_cli_exposes_with_pty_flag():
    src = Path(spawn_durable_daemon.__file__).read_text(encoding="utf-8")
    assert '"--with-pty"' in src
    # default OFF: byte-identical daemon behavior when unset
    assert 'getattr(a, "with_pty", False)' in src
