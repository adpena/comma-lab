# SPDX-License-Identifier: MIT
"""WORKTREE-AWARE FIX (2026-07-17) tests for the subagent commit serializer.

Incident: ``REPO_ROOT = Path(__file__).resolve().parent.parent`` hardcoded the
serializer to the MAIN checkout, so a commit attempted from inside a git
WORKTREE silently staged main's copy of the file (the worktree copy was never
seen). The caller's post-edit ``--expected-content-sha256`` then could never
match (spurious rc=4), making worktree commits via the serializer impossible.
This surfaced live 2026-07-17 committing SPEC_v10 §13.13 from a worktree.

The fix: ``_resolve_effective_repo_root`` resolves the root from
``--repo-root`` / ``$SUBAGENT_SERIALIZER_REPO_ROOT`` / ``git rev-parse
--show-toplevel`` (CWD) / the ``__file__`` fallback, and ``main()`` rebinds the
REPO_ROOT / LOCK_PATH / LOG_PATH globals before any git op. A worktree has its
OWN index, so the anti-commit-swap guarantee (per-index race) is preserved.

ALL tests run against throwaway git repos under tmp_path — NEVER the real repo.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_SERIALIZER_PATH = REPO / "tools" / "subagent_commit_serializer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_scs_worktree_aware", _SERIALIZER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )


def _make_repo_with_worktree(base: Path) -> tuple[Path, Path]:
    """Return (main_checkout, linked_worktree_on_branch 'wtbr')."""
    main_co = base / "main_co"
    main_co.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["init"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "WorktreeTest"],
        ["config", "commit.gpgsign", "false"],
    ):
        assert _git(main_co, *cmd).returncode == 0
    (main_co / ".omx" / "state").mkdir(parents=True, exist_ok=True)
    (main_co / "seed.txt").write_bytes(b"seed\n")
    assert _git(main_co, "add", ".").returncode == 0
    assert _git(main_co, "commit", "-m", "init").returncode == 0
    wt = base / "wt_branch"
    assert _git(main_co, "worktree", "add", str(wt), "-b", "wtbr").returncode == 0
    (wt / ".omx" / "state").mkdir(parents=True, exist_ok=True)
    return main_co, wt


def _run_serializer(cwd: Path, argv: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["REVIEW_GATE_OVERRIDE"] = "1"  # non-.py content in these tests
    return subprocess.run(
        [sys.executable, str(_SERIALIZER_PATH), *argv],
        cwd=cwd, capture_output=True, text=True, check=False, env=env,
    )


# ---------------------------------------------------------------------------
# 1. Unit: the resolver honors explicit > env > cwd > __file__ fallback.
# ---------------------------------------------------------------------------
def test_resolve_prefers_explicit_repo_root(tmp_path):
    mod = _load_module()
    main_co, wt = _make_repo_with_worktree(tmp_path)
    # explicit wins even when CWD is elsewhere
    assert mod._resolve_effective_repo_root(str(wt)) == wt.resolve()
    assert mod._resolve_effective_repo_root(str(main_co)) == main_co.resolve()


def test_resolve_env_var(tmp_path, monkeypatch):
    mod = _load_module()
    _main, wt = _make_repo_with_worktree(tmp_path)
    monkeypatch.setenv("SUBAGENT_SERIALIZER_REPO_ROOT", str(wt))
    assert mod._resolve_effective_repo_root(None) == wt.resolve()


def test_resolve_falls_back_to_module_root_outside_git(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.delenv("SUBAGENT_SERIALIZER_REPO_ROOT", raising=False)
    non_git = tmp_path / "not_a_repo"
    non_git.mkdir()
    monkeypatch.chdir(non_git)
    # No git tree, no explicit, no env -> the __file__ checkout.
    assert mod._resolve_effective_repo_root(None) == mod.REPO_ROOT


def test_resolve_bad_explicit_ignored(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.delenv("SUBAGENT_SERIALIZER_REPO_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)  # tmp_path itself is not a git root
    # A path with no .git is not a valid root -> ignored, falls through.
    got = mod._resolve_effective_repo_root(str(tmp_path / "does_not_exist"))
    assert got == mod.REPO_ROOT


# ---------------------------------------------------------------------------
# 2. Functional: a commit from inside a worktree lands on the WORKTREE branch
#    (auto-detect via CWD), NOT the main checkout — the regression this fixes.
# ---------------------------------------------------------------------------
def test_commit_from_worktree_cwd_lands_on_worktree_branch(tmp_path):
    main_co, wt = _make_repo_with_worktree(tmp_path)
    f = wt / "wtfile.txt"
    f.write_bytes(b"worktree-only edit\n")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()

    proc = _run_serializer(wt, [
        "--message", "worktree functional test [no-triality]",
        "--files", "wtfile.txt",
        "--expected-content-sha256", f"wtfile.txt={sha}",
    ])
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    # The worktree branch HEAD has the commit + the file.
    assert _git(wt, "cat-file", "-e", "HEAD:wtfile.txt").returncode == 0
    head_msg = _git(wt, "log", "-1", "--format=%s").stdout
    assert "worktree functional test" in head_msg

    # The MAIN checkout must NOT have the file (no leak across working trees).
    tracked_on_main = _git(main_co, "ls-files").stdout
    assert "wtfile.txt" not in tracked_on_main


# ---------------------------------------------------------------------------
# 3. Functional: --repo-root explicit override from an unrelated CWD.
# ---------------------------------------------------------------------------
def test_commit_via_explicit_repo_root_flag(tmp_path):
    main_co, wt = _make_repo_with_worktree(tmp_path)
    f = wt / "explicit.txt"
    f.write_bytes(b"via --repo-root\n")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()

    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    proc = _run_serializer(unrelated, [
        "--repo-root", str(wt),
        "--message", "explicit repo-root test [no-triality]",
        "--files", "explicit.txt",
        "--expected-content-sha256", f"explicit.txt={sha}",
    ])
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _git(wt, "cat-file", "-e", "HEAD:explicit.txt").returncode == 0
    assert "explicit.txt" not in _git(main_co, "ls-files").stdout


# ---------------------------------------------------------------------------
# 4. Backward-compat: committing from the MAIN checkout still targets main
#    (the pre-fix behavior for the common case is unchanged).
# ---------------------------------------------------------------------------
def test_commit_from_main_checkout_targets_main(tmp_path):
    main_co, _wt = _make_repo_with_worktree(tmp_path)
    f = main_co / "mainfile.txt"
    f.write_bytes(b"main edit\n")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()

    proc = _run_serializer(main_co, [
        "--message", "main checkout test [no-triality]",
        "--files", "mainfile.txt",
        "--expected-content-sha256", f"mainfile.txt={sha}",
    ])
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _git(main_co, "cat-file", "-e", "HEAD:mainfile.txt").returncode == 0
