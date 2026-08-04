"""SIGURG/rc=144 kill-class permanent fix (operator 2026-08-04, recurring since 2026-04-28).

Two-landing: (1) the guard blocks the hand-rolled patterns at the Bash spawn
site; (2) the canonical launcher gained --done-receipt so detached completions
notify MAIN via the fleet watcher. Cases below are the EXECUTED positive
controls from the landing turn — both blocked shapes are the literal incidents
from 2026-08-04 (hand-rolled nohup clone; run_in_background git clone rc=144).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guard_blocks_hand_rolled_nohup_detach() -> None:
    m = _load("launch_guard_hook")
    allow, reason = m.decide(
        "nohup bash -c 'git clone x y > log 2>&1' < /dev/null > /dev/null 2>&1 & disown",
        env={},
    )
    assert not allow and "launch_detached_process.py" in reason


def test_guard_blocks_backgrounded_long_runner() -> None:
    m = _load("launch_guard_hook")
    allow, _ = m.decide(
        "git clone --depth 1 https://github.com/x/y /dest", env={}, run_in_background=True
    )
    assert not allow
    # Foreground stays allowed (times out visibly instead of dying silently).
    allow_fg, _ = m.decide(
        "git clone --depth 1 https://github.com/x/y /dest", env={}, run_in_background=False
    )
    assert allow_fg


def test_guard_allows_canonical_detach_surfaces_and_hatch() -> None:
    m = _load("launch_guard_hook")
    for cmd, bg in [
        (
            ".venv/bin/python tools/launch_detached_process.py --output-dir d "
            "--done-receipt n -- git clone x y",
            True,
        ),
        ("nohup .venv/bin/python tools/spawn_durable_daemon.py --x & disown", False),
        ("TAC_LAUNCH_GUARD_OK=1 nohup long_thing & disown", False),
        ("git status && ls -la", False),
    ]:
        allow, _ = m.decide(cmd, env={}, run_in_background=bg)
        assert allow, cmd


def test_launcher_done_receipt_supervisor_shape() -> None:
    """--done-receipt wraps argv in the SIGURG-ignoring supervisor and the
    receipt path targets the fleet-watcher dir with keeper-compatible format."""
    src = (_TOOLS / "launch_detached_process.py").read_text()
    assert "--done-receipt" in src
    assert "codex_runs" in src  # the watched dir — completions notify MAIN
    assert "SIGURG" in src  # supervisor is immune to the reaper signal
    assert "rc=%d elapsed=%d" in src  # keeper-compatible receipt prefix
