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
