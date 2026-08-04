"""Keeper context-exhaustion RELAY — operator directive 2026-08-04.

Arms must exceed one context window and run autonomously (se1 died rc=1 at
562s on the exact signature pinned here). The keeper relaunches codex with a
fresh context + disk-state continuation header, capped, with a no-progress
guard. Positive control executed at landing: fake codex dying with the
signature in gen 1 relayed and succeeded in gen 2 (done receipt ``gen=2``).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import codex_arm_queue as q  # noqa: E402

SE1_SIGNATURE = "ran out of room in the model's context window"


def _src() -> str:
    return q.keeper_source("relayarm", ".omx/tmp/codex_runs/relayarm_prompt.md")


def test_generated_keeper_compiles():
    compile(_src(), "keeper", "exec")


def test_keeper_carries_relay_contract():
    src = _src()
    # The apostrophe is backslash-escaped in the generated source text; the
    # behavioral test below proves the full signature matches at runtime.
    assert "ran out of room in the model" in src, "the REAL se1 death signature must be matched"
    assert "MAX_GEN = 12" in src, "generation cap missing (runaway guard)"
    assert "CONTINUATION generation" in src, "relay header missing"
    assert "progressed" in src, "no-progress relay guard missing"
    assert "gen=%d" in src, "done receipt must record the generation count"


def test_done_receipt_prefix_backward_compatible():
    # Existing .done parsers key on the leading rc=/signal= token; gen is appended.
    src = _src()
    assert "'rc=%d elapsed=%d gen=%d\\n'" in src


def test_spawn_command_reaper_shape_unchanged():
    cmd = q.spawn_command("relayarm", ".omx/tmp/codex_runs/relayarm_prompt.md")
    assert not re.search(r"\bcodex\b", cmd), "reaper-shape invariant broken"
    assert not re.search(r"\bclaude\b", cmd)


def test_relay_fires_and_second_generation_succeeds(tmp_path):
    # Behavioral twin of the landing-time positive control, isolated in tmp.
    src = re.sub(
        r"ARGV_PREFIX = \[.*?\]",
        "ARGV_PREFIX = ['python3', 'fake_codex.py']",
        _src(),
        flags=re.S,
    )
    (tmp_path / ".omx/tmp/codex_runs").mkdir(parents=True)
    (tmp_path / ".omx/research").mkdir(parents=True)
    (tmp_path / "fake_codex.py").write_text(
        "import os, sys\n"
        "if not os.path.exists('.gen_marker'):\n"
        "    open('.gen_marker', 'w').write('1')\n"
        f"    print(\"ERROR: Codex {SE1_SIGNATURE}.\")\n"
        "    sys.exit(1)\n"
        "print('FINAL from: ' + sys.argv[-1][:40])\n"
        "sys.exit(0)\n"
    )
    (tmp_path / "keeper_test.py").write_text(src)
    r = subprocess.run(
        ["python3", "keeper_test.py"], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    done = (tmp_path / ".omx/tmp/codex_runs/relayarm.done").read_text()
    log = (tmp_path / ".omx/tmp/codex_runs/relayarm.log").read_text()
    assert r.returncode == 0
    assert "rc=0" in done and "gen=2" in done
    assert "CONTINUATION generation 2" in log


def test_non_exhaustion_failure_does_not_relay(tmp_path):
    src = re.sub(
        r"ARGV_PREFIX = \[.*?\]",
        "ARGV_PREFIX = ['python3', 'fake_codex.py']",
        _src(),
        flags=re.S,
    )
    (tmp_path / ".omx/tmp/codex_runs").mkdir(parents=True)
    (tmp_path / ".omx/research").mkdir(parents=True)
    (tmp_path / "fake_codex.py").write_text(
        "import sys\nprint('ERROR: some unrelated failure')\nsys.exit(3)\n"
    )
    (tmp_path / "keeper_test.py").write_text(src)
    r = subprocess.run(
        ["python3", "keeper_test.py"], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    done = (tmp_path / ".omx/tmp/codex_runs/relayarm.done").read_text()
    assert r.returncode == 3
    assert "rc=3" in done and "gen=1" in done


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
