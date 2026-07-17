"""Tests for check_codex_exec_spawn_paths_are_reaper_immune (task #525, warn-only).

Bug class: a codex-exec spawn path outside the reaper-immune pty core leaves the
spawned codex with no controlling terminal; the fleet claude-code-reaper launchd
agent SIGTERMs it at ~5min (measured 2026-07-17: 10/10 arm kills at 5m18-5m55).
"""
from __future__ import annotations

import pytest

from tac.preflight import PreflightError, check_codex_exec_spawn_paths_are_reaper_immune


def _mk(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


_VIOLATING = (
    "import subprocess\n"
    "launcher = 'codex exec --skip-git-repo-check -m gpt-5.6-sol'\n"
    "subprocess.Popen(['bash', '-c', launcher])\n"
)

_IMMUNE = (
    "from spawn_durable_daemon import spawn_detached_verified\n"
    "launcher = 'codex exec -m gpt-5.6-sol'\n"
    "spawn_detached_verified(['bash', '-c', launcher], 'l.log', with_pty=True)\n"
)


def test_flags_bare_popen_codex_spawn(tmp_path):
    _mk(tmp_path, "tools/bad_spawner.py", _VIOLATING)
    v = check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path)
    assert len(v) == 1
    assert "bad_spawner.py" in v[0]
    assert "spawn_detached_verified" in v[0]


def test_allows_immune_core_route(tmp_path):
    _mk(tmp_path, "tools/good_spawner.py", _IMMUNE)
    assert check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path) == []


def test_allows_codex_exec_string_without_spawning(tmp_path):
    _mk(tmp_path, "tools/doc_only.py", "TEMPLATE = 'codex exec ...'  # docs/template only\n")
    assert check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path) == []


def test_allows_popen_without_codex(tmp_path):
    _mk(tmp_path, "tools/other.py", "import subprocess\nsubprocess.Popen(['ls'])\n")
    assert check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path) == []


def test_waiver_with_real_rationale_respected(tmp_path):
    _mk(tmp_path, "tools/waived.py",
        _VIOLATING + "# CODEX_SPAWN_REAPER_IMMUNE_OK:foreground-only smoke, exits in <60s\n")
    assert check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path) == []


def test_placeholder_waiver_rejected(tmp_path):
    _mk(tmp_path, "tools/fake_waiver.py",
        _VIOLATING + "# CODEX_SPAWN_REAPER_IMMUNE_OK:<rationale>\n")
    v = check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path)
    assert len(v) == 1


def test_empty_waiver_rejected(tmp_path):
    _mk(tmp_path, "tools/empty_waiver.py",
        _VIOLATING + "# CODEX_SPAWN_REAPER_IMMUNE_OK:\n")
    v = check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path)
    assert len(v) == 1


def test_tests_and_intake_dirs_excluded(tmp_path):
    _mk(tmp_path, "tools/tests/test_x.py", _VIOLATING)
    _mk(tmp_path, "experiments/results/public_pr95_intake_x/clone.py", _VIOLATING)
    assert check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path) == []


def test_osascript_launch_counts_as_spawning(tmp_path):
    _mk(tmp_path, "tools/osa.py",
        "cmd = 'codex exec -m x'\nos.system('osascript -e ...')  # osascript launch\n")
    v = check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path)
    assert len(v) == 1


def test_strict_mode_raises(tmp_path):
    _mk(tmp_path, "tools/bad_spawner.py", _VIOLATING)
    with pytest.raises(PreflightError):
        check_codex_exec_spawn_paths_are_reaper_immune(repo_root=tmp_path, strict=True)


def test_live_repo_count_is_zero():
    """Strict-flip precondition custody: the live repo has no violating spawn path
    (codex_delegate routes through the shared immune core as of task #525 C1)."""
    assert check_codex_exec_spawn_paths_are_reaper_immune() == []
