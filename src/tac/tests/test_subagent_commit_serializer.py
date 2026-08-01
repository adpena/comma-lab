# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO / "tools" / "subagent_commit_serializer.py"
    spec = importlib.util.spec_from_file_location("_subagent_commit_serializer", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def test_refresh_real_index_after_temp_commit_clears_stale_status(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "codex@example.invalid")
    _git(tmp_path, "config", "user.name", "Codex")
    target = tmp_path / "f.txt"
    target.write_text("old\n", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-m", "base")

    target.write_text("new\n", encoding="utf-8")
    alt_index = tmp_path.parent / f"{tmp_path.name}-alt-index"
    env = {**os.environ, "GIT_INDEX_FILE": str(alt_index)}
    _git(tmp_path, "read-tree", "HEAD", env=env)
    _git(tmp_path, "add", "--", "f.txt", env=env)
    _git(tmp_path, "commit", "-m", "alt", env=env)

    assert _git(tmp_path, "status", "--short")

    mod._refresh_real_index_after_temp_commit(["f.txt"], repo_root=tmp_path)

    assert _git(tmp_path, "status", "--short") == ""


def test_review_gate_override_rejects_python_but_allows_state_docs() -> None:
    mod = _load_module()
    with mock.patch.dict(os.environ, {"REVIEW_GATE_OVERRIDE": "1"}, clear=False):
        assert mod._review_gate_override_python_targets(
            ["notes.md", ".omx/state/ledger.jsonl"], no_stage=False
        ) == []
        assert mod._review_gate_override_python_targets(
            ["notes.md", "tools/fix.py"], no_stage=False
        ) == ["tools/fix.py"]


def test_review_gate_override_no_stage_inspects_real_staged_targets() -> None:
    mod = _load_module()
    staged = subprocess.CompletedProcess(
        args=["git"], returncode=0, stdout="notes.md\nsrc/tac/live.py\n", stderr=""
    )
    with (
        mock.patch.dict(os.environ, {"REVIEW_GATE_OVERRIDE": "1"}, clear=False),
        mock.patch.object(mod.subprocess, "run", return_value=staged),
    ):
        assert mod._review_gate_override_python_targets([], no_stage=True) == [
            "src/tac/live.py"
        ]


# ---------------------------------------------------------------------------
# Lock patience (task #854).
#
# The retired literal was 120s, derived from a hook that ran ~5-10s. MEASURED over
# .omx/state/commit-serializer.log: p50 `commit_seconds` 3.2s and p90 7.8s (n=9112) —
# the median never moved — while p99 reached 161.0s and max 468.0s after the hook grew a
# CI-blind pytest step. Result on 2026-08-01: 7 of 60 attempts (11.7%) died with
# outcome=lock_timeout, every one a healthy commit behind a healthy sibling.
# ---------------------------------------------------------------------------
def test_lock_patience_is_derived_from_the_hook_bound_not_a_literal() -> None:
    mod = _load_module()
    assert mod.DEFAULT_TIMEOUT_SECONDS == max(
        mod._hook_wall_clock_bound_seconds(),
        mod.MAX_CONCURRENT_COMMITTERS * mod.MEASURED_P99_HOLD_SECONDS,
    )


def test_lock_patience_covers_the_largest_wait_ever_recorded() -> None:
    # 1021.93s is the largest SUCCESSFUL lock wait in the ledger (n=9833 waits). A
    # patience below it would have failed a commit that in fact went on to succeed.
    mod = _load_module()
    assert mod.DEFAULT_TIMEOUT_SECONDS > 1022
    # and it must exceed the 120s that produced the measured starvation
    assert mod.DEFAULT_TIMEOUT_SECONDS > 120


def test_lock_patience_follows_the_hook_env_so_the_two_cannot_drift() -> None:
    # The child `git commit` inherits this env, so a raised CI-blind ceiling means a
    # longer hook — and the patience has to move with it. This is the coupling whose
    # ABSENCE was the starvation mechanism.
    mod = _load_module()
    with mock.patch.dict(
        os.environ, {"PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS": "1500"}, clear=False
    ):
        raised = mod.default_timeout_seconds()
    with mock.patch.dict(
        os.environ, {"PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS": "60"}, clear=False
    ):
        lowered = mod.default_timeout_seconds()
    assert raised > lowered
    # ...and the queue term still floors it when the hook bound is small
    assert lowered >= mod.MAX_CONCURRENT_COMMITTERS * mod.MEASURED_P99_HOLD_SECONDS


def test_hook_bound_falls_back_rather_than_raising_when_the_hook_is_unreadable() -> None:
    # A hook we cannot import must never stop a commit.
    mod = _load_module()
    with mock.patch.object(
        mod.importlib.util, "spec_from_file_location", side_effect=OSError("boom")
    ):
        assert mod._hook_wall_clock_bound_seconds() == mod._FALLBACK_HOOK_BOUND_SECONDS


def test_lock_wait_is_narrated_not_silent() -> None:
    # Patience without narration is indistinguishable from a hang.
    src = (REPO / "tools" / "subagent_commit_serializer.py").read_text(encoding="utf-8")
    body = src.split("def _acquire_lock", 1)[1].split("\ndef ", 1)[0]
    assert "LOCK_WAIT_PROGRESS_SECONDS" in body
    assert "waiting for" in body
