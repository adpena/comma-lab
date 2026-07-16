"""Tests for check_retry_without_descendant_check (#512 preflight half).

Flags tools/ + scripts/ .py files that perform a DETACHED spawn
(setsid/start_new_session/preexec_fn) in a retry/respawn path with NO psutil
descendant / same-out-dir guard — the orphan-spawn duplicate class (2026-07-15).
The runtime cure (a guard IN tools/launch_witness_run.py) is DEFERRED to
post-#507-launch; this is the static warn-only detector only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_retry_without_descendant_check,
)


def _mk(root: Path, sub: str, name: str, body: str) -> None:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


_DETACHED_RESPAWN_NOGUARD = '''\
"""A respawn helper that detach-spawns a trainer."""
import subprocess

def respawn(out_dir):
    return subprocess.Popen(
        ["bash", "launch.sh", "--out-dir", out_dir],
        start_new_session=True,
    )
'''

_DETACHED_RETRY_LAUNCH_NOGUARD = '''\
"""Retry wrapper that relaunches a detached job."""
import os, subprocess

def run_with_retry(cmd):
    for attempt in range(3):  # retry loop
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid)  # spawn/launch
        if proc.wait() == 0:
            return 0
    return 1
'''

_DETACHED_RESPAWN_WITH_GUARD = '''\
"""A respawn helper that DOES enumerate descendants first."""
import subprocess, psutil

def respawn(out_dir):
    for p in psutil.process_iter(["cmdline"]):
        if out_dir in " ".join(p.info["cmdline"] or []):
            raise SystemExit("live writer already at out-dir")
    return subprocess.Popen(["bash", "launch.sh"], start_new_session=True)
'''

_NO_DETACH_RESPAWN = '''\
"""A respawn helper that spawns an ATTACHED child process."""
import subprocess

def respawn():
    return subprocess.Popen(["python", "worker.py"])  # relaunch, but attached
'''

_DETACHED_NO_RETRY = '''\
"""A one-shot detached launcher."""
import subprocess

def launch_once():
    return subprocess.Popen(["bash", "run.sh"], start_new_session=True)
'''

_RETRY_HTTP_ONLY = '''\
"""HTTP client with retry — no detached spawn at all."""
import requests

def fetch(url):
    for attempt in range(3):  # retry
        r = requests.get(url)
        if r.ok:
            return r
'''

_DETACHED_RESPAWN_WAIVED = '''\
"""A respawn helper waived deliberately."""
import subprocess

def respawn():  # RETRY_WITHOUT_DESCENDANT_CHECK_OK: supervisor tracks pgid itself
    return subprocess.Popen(["bash", "launch.sh"], start_new_session=True)
'''

_DETACHED_RESPAWN_PLACEHOLDER_WAIVER = '''\
"""A respawn helper with a placeholder waiver (must NOT self-suppress)."""
import subprocess

def respawn():  # RETRY_WITHOUT_DESCENDANT_CHECK_OK:<rationale>
    return subprocess.Popen(["bash", "launch.sh"], start_new_session=True)
'''


# ---- positive ---------------------------------------------------------------

def test_detached_respawn_noguard_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "respawn_helper.py", _DETACHED_RESPAWN_NOGUARD)
    violations = check_retry_without_descendant_check(repo_root=tmp_path)
    assert len(violations) == 1
    assert "tools/respawn_helper.py" in violations[0]
    assert "descendant" in violations[0]


def test_detached_retry_launch_noguard_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "retry_launch.py", _DETACHED_RETRY_LAUNCH_NOGUARD)
    violations = check_retry_without_descendant_check(repo_root=tmp_path)
    assert len(violations) == 1


def test_scripts_scope_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "scripts", "relaunch_thing.py", _DETACHED_RESPAWN_NOGUARD)
    violations = check_retry_without_descendant_check(repo_root=tmp_path)
    assert len(violations) == 1
    assert "scripts/relaunch_thing.py" in violations[0]


def test_strict_raises(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "respawn_helper.py", _DETACHED_RESPAWN_NOGUARD)
    with pytest.raises(PreflightError):
        check_retry_without_descendant_check(repo_root=tmp_path, strict=True)


def test_placeholder_waiver_still_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "ph.py", _DETACHED_RESPAWN_PLACEHOLDER_WAIVER)
    violations = check_retry_without_descendant_check(repo_root=tmp_path)
    assert len(violations) == 1


# ---- negative ---------------------------------------------------------------

def test_guarded_respawn_not_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "guarded.py", _DETACHED_RESPAWN_WITH_GUARD)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_non_detached_respawn_not_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "attached.py", _NO_DETACH_RESPAWN)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_detached_but_no_retry_not_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "oneshot.py", _DETACHED_NO_RETRY)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_http_retry_only_not_flagged(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "httpclient.py", _RETRY_HTTP_ONLY)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_waiver_suppresses(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "waived.py", _DETACHED_RESPAWN_WAIVED)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_out_of_scope_dir_not_scanned(tmp_path: Path) -> None:
    _mk(tmp_path, "experiments", "respawn.py", _DETACHED_RESPAWN_NOGUARD)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_missing_dirs_fail_open(tmp_path: Path) -> None:
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_multiple_files_mixed(tmp_path: Path) -> None:
    _mk(tmp_path, "tools", "bad.py", _DETACHED_RESPAWN_NOGUARD)
    _mk(tmp_path, "tools", "good.py", _DETACHED_RESPAWN_WITH_GUARD)
    _mk(tmp_path, "scripts", "bad2.py", _DETACHED_RETRY_LAUNCH_NOGUARD)
    violations = check_retry_without_descendant_check(repo_root=tmp_path)
    assert len(violations) == 2
    joined = "\n".join(violations)
    assert "tools/bad.py" in joined and "scripts/bad2.py" in joined
    assert "tools/good.py" not in joined


def test_cmdline_guard_token_recognized(tmp_path: Path) -> None:
    body = _DETACHED_RESPAWN_NOGUARD.replace(
        "start_new_session=True,",
        "start_new_session=True,\n    )\n    # after: check p.cmdline() of children\n    if False: p.cmdline()")
    _mk(tmp_path, "tools", "cmdlineguard.py", body)
    assert check_retry_without_descendant_check(repo_root=tmp_path) == []


def test_verbose_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _mk(tmp_path, "tools", "bad.py", _DETACHED_RESPAWN_NOGUARD)
    check_retry_without_descendant_check(repo_root=tmp_path, verbose=True)
    out = capsys.readouterr().out
    assert "check_retry_without_descendant_check" in out


def test_live_repo_does_not_raise() -> None:
    result = check_retry_without_descendant_check(strict=False)
    assert isinstance(result, list)
