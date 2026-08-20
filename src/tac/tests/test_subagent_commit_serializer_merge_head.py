# SPDX-License-Identifier: MIT
"""FIX-MERGE-HEAD: refuse to commit while a `git merge --no-commit` is open.

The ddm_oc2 incident (2026-08-20): a merge was left open across tool calls. A
concurrent serializer commit then landed while `.git/MERGE_HEAD` existed, so git
recorded the merge branch as a SECOND PARENT of a commit that staged only the
serializer caller's own files. History claimed the branch was merged, `git merge`
afterwards reported "Already up to date", and 7,637 insertions were nearly lost
silently.

`test_unguarded_commit_really_does_fabricate_a_false_second_parent` is the
POSITIVE CONTROL: it reproduces the incident with a bare `git commit` and proves
the branch content is absent from the resulting merge commit's tree. Without it
the refusal tests below could pass against a hazard that does not exist.

Measured class population over this repo's history at the time of the fix:
3 of 311 merge commits carry the signature (100% of the branch's changed files
absent from the merge tree).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SERIALIZER = REPO / "tools" / "subagent_commit_serializer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_subagent_commit_serializer_mergehead", SERIALIZER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc


def _scratch_repo_with_open_merge(tmp_path: Path) -> Path:
    """A throwaway repo mid-`git merge --no-commit`, plus one unrelated edit.

    Shape of the incident: `feature` adds `branch_only.txt`; `main` meanwhile has
    an unrelated file the caller wants to commit.
    """

    repo = tmp_path / "scratch"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ad1@test.invalid")
    _git(repo, "config", "user.name", "ad1 test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / ".omx" / "state").mkdir(parents=True)
    (repo / "seed.txt").write_text("seed\n")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-q", "-m", "seed")

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "branch_only.txt").write_text("content the merge must not lose\n")
    _git(repo, "add", "branch_only.txt")
    _git(repo, "commit", "-q", "-m", "branch work")

    _git(repo, "checkout", "-q", "main")
    (repo / "main_only.txt").write_text("unrelated main-side edit\n")
    _git(repo, "add", "main_only.txt")
    _git(repo, "commit", "-q", "-m", "main work")

    # Open the merge and DO NOT commit it — the incident window.
    proc = subprocess.run(
        ["git", "merge", "--no-commit", "--no-ff", "feature"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (repo / ".git" / "MERGE_HEAD").is_file()
    return repo


def _run_serializer(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["SUBAGENT_SERIALIZER_REPO_ROOT"] = str(repo)
    env["REVIEW_GATE_OVERRIDE"] = "1"
    env["SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE"] = "1"
    env["SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE"] = "scratch repo unit test"
    return subprocess.run(
        [sys.executable, str(SERIALIZER), *args],
        cwd=repo, env=env, capture_output=True, text=True, check=False, timeout=300,
    )


def test_unguarded_commit_really_does_fabricate_a_false_second_parent(
    tmp_path: Path,
) -> None:
    """POSITIVE CONTROL: the hazard is real, so the refusals below are not theatre."""

    repo = _scratch_repo_with_open_merge(tmp_path)
    # Stage ONLY an unrelated file, exactly as a serializer caller would.
    (repo / "my_own.txt").write_text("my own work\n")
    _git(repo, "rm", "-q", "--cached", "branch_only.txt")
    (repo / "branch_only.txt").unlink()
    _git(repo, "add", "my_own.txt")
    _git(repo, "commit", "-q", "-m", "my own commit, nothing to do with the merge")

    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    parents = _git(repo, "rev-list", "--parents", "-n", "1", head).stdout.split()
    assert len(parents) == 3, "git did NOT record a second parent — hazard absent"

    # The branch's file is absent from the merge commit's tree ...
    tree = _git(repo, "ls-tree", "-r", "--name-only", head).stdout.split()
    assert "branch_only.txt" not in tree
    # ... yet git now considers the branch fully merged. This is the silent loss.
    already = subprocess.run(
        ["git", "merge", "feature"], cwd=repo, capture_output=True, text=True, check=False,
    )
    assert "Already up to date" in already.stdout


def test_merge_in_progress_detects_and_clears(tmp_path: Path) -> None:
    mod = _load_module()
    repo = _scratch_repo_with_open_merge(tmp_path)
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = repo
    try:
        state = mod.merge_in_progress()
        assert state is not None
        sha, mtime = state
        assert sha == _git(repo, "rev-parse", "feature").stdout.strip()
        assert mtime > 0
        _git(repo, "merge", "--abort")
        assert mod.merge_in_progress() is None
    finally:
        mod.REPO_ROOT = old_root


def test_serializer_refuses_rc16_while_a_merge_is_open(tmp_path: Path) -> None:
    repo = _scratch_repo_with_open_merge(tmp_path)
    (repo / "my_own.txt").write_text("my own work\n")
    proc = _run_serializer(
        repo, "--message", "unrelated work", "--files", "my_own.txt",
    )
    assert proc.returncode == 16, proc.stdout + proc.stderr
    assert "merge is IN PROGRESS" in proc.stderr
    assert _git(repo, "rev-parse", "feature").stdout.strip() in proc.stderr
    # `git merge --no-commit` leaves the branch file staged in the REAL index,
    # but the serializer commits from a temp index built from HEAD, so the
    # message must name the branch's file count rather than imply the real
    # index protects anything.
    assert "changed 1 file(s)" in proc.stderr
    assert "TEMP INDEX" in proc.stderr
    # Refusal must not have committed anything.
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == _git(
        repo, "rev-parse", "main",
    ).stdout.strip()


def test_merge_commit_flag_requires_no_stage(tmp_path: Path) -> None:
    repo = _scratch_repo_with_open_merge(tmp_path)
    (repo / "my_own.txt").write_text("my own work\n")
    proc = _run_serializer(
        repo, "--message", "merge", "--files", "my_own.txt", "--merge-commit",
    )
    assert proc.returncode == 16, proc.stdout + proc.stderr
    assert "requires --no-stage" in proc.stderr


def test_merge_commit_refuses_when_staged_set_misses_branch_files(
    tmp_path: Path,
) -> None:
    repo = _scratch_repo_with_open_merge(tmp_path)
    # Unstage the branch's contribution: the exact incident shape.
    _git(repo, "rm", "-q", "--cached", "branch_only.txt")
    proc = _run_serializer(
        repo, "--message", "merge feature", "--merge-commit", "--no-stage",
    )
    assert proc.returncode == 16, proc.stdout + proc.stderr
    assert "does NOT cover the merge" in proc.stderr
    assert "branch_only.txt" in proc.stderr


def test_merge_commit_succeeds_when_the_whole_merge_is_staged(tmp_path: Path) -> None:
    repo = _scratch_repo_with_open_merge(tmp_path)
    # `--no-stage` still requires every staged path to be declared (rc=15), so a
    # merge commit declares the merge result. The two guards compose.
    proc = _run_serializer(
        repo, "--message", "merge feature into main",
        "--merge-commit", "--no-stage", "--files", "branch_only.txt",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    parents = _git(repo, "rev-list", "--parents", "-n", "1", head).stdout.split()
    assert len(parents) == 3
    tree = _git(repo, "ls-tree", "-r", "--name-only", head).stdout.split()
    assert "branch_only.txt" in tree, "the merge landed without its content"


def test_no_merge_open_is_unaffected(tmp_path: Path) -> None:
    """Regression guard: the common path must not be disturbed."""

    repo = _scratch_repo_with_open_merge(tmp_path)
    _git(repo, "merge", "--abort")
    (repo / "my_own.txt").write_text("my own work\n")
    proc = _run_serializer(
        repo, "--message", "ordinary commit", "--files", "my_own.txt",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    parents = _git(repo, "rev-list", "--parents", "-n", "1", head).stdout.split()
    assert len(parents) == 2, "an ordinary commit must have exactly one parent"


@pytest.mark.parametrize("subcommand", ["--merge-commit"])
def test_flag_is_documented_in_help(subcommand: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(SERIALIZER), "--help"],
        cwd=REPO, capture_output=True, text=True, check=False, timeout=120,
    )
    assert proc.returncode == 0
    assert subcommand in proc.stdout
