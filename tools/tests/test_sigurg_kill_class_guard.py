"""SIGURG/rc=144 kill-class permanent fix (operator 2026-08-04, recurring since 2026-04-28).

Two-landing: (1) the guard blocks the hand-rolled patterns at the Bash spawn
site; (2) the canonical launcher gained --done-receipt so detached completions
notify MAIN via the fleet watcher. Cases below are the EXECUTED positive
controls from the landing turn — both blocked shapes are the literal incidents
from 2026-08-04 (hand-rolled nohup clone; run_in_background git clone rc=144).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
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
    receipt path targets the fleet-watcher dir with structured launch identity."""
    src = (_TOOLS / "launch_detached_process.py").read_text()
    assert "--done-receipt" in src
    assert "codex_runs" in src  # the watched dir — completions notify MAIN
    assert "SIGURG" in src  # supervisor is immune to the reaper signal
    assert 'DONE_RECEIPT_SCHEMA = "detached_local_process_done.v2"' in src
    assert '"manifest_path"' in src
    assert '"monotonic_launch_counter"' in src


def test_launcher_adjudicates_child_nonzero_rc_during_verify_window(tmp_path: Path) -> None:
    """An immediate child failure is reported once and tagged for suppression."""
    repo = _TOOLS.parent
    name = f"bl1_receipt_{os.getpid()}_{time.time_ns()}"
    done = repo / ".omx" / "tmp" / "codex_runs" / f"{name}.done"
    done.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(_TOOLS / "launch_detached_process.py"),
                "--output-dir",
                str(tmp_path / "launch"),
                "--cwd",
                str(repo),
                "--done-receipt",
                name,
                "--",
                sys.executable,
                "-c",
                "import sys; sys.exit(7)",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 7, proc.stderr
        deadline = time.time() + 10
        while time.time() < deadline and not done.exists():
            time.sleep(0.05)
        assert done.exists(), proc.stdout
        receipt = json.loads(done.read_text(encoding="utf-8"))
        assert receipt["rc"] == 7
        assert receipt["adjudicated_at_launch"] is True
        assert set(receipt["launch_id"]) == {
            "manifest_path",
            "pid",
            "monotonic_launch_counter",
        }
    finally:
        done.unlink(missing_ok=True)
        done.with_name(done.name + ".consumed.json").unlink(missing_ok=True)
