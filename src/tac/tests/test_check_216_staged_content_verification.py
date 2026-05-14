# SPDX-License-Identifier: MIT
"""Catalog #216 P1 fix — subagent commit serializer staged-content verification.

Bug class anchor 2026-05-14: D4-OOM-FIX commit ``5d0ec061d`` (Catalog #218
self-protect) absorbed FIX-HARDEN-OPT's Catalog #215 edits to
``src/tac/preflight.py``. Both subagents edited the same file; D4's
``git add src/tac/preflight.py`` packaged BOTH sets of edits because both
were already in the working tree at the time of the staging step. The
pre-lock + post-lock check (Catalog #157) saw stable content because BOTH
edits were present before either subagent took its pre-lock snapshot.

The fix (Catalog #216): after ``git add <files>``, hash the file's STAGED
content (via ``git cat-file blob $(git ls-files --stage <file>)``) and
verify it matches the caller's ``--expected-content-sha256``. Refuse with
rc=5 on mismatch (separate from rc=4 = pre-lock working-tree mismatch).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from tools.subagent_commit_serializer import (
    _hash_staged_files,
    _staged_content_check,
)


def _init_test_repo(tmp_path: Path) -> Path:
    """Initialize a tiny git repo in tmp_path. Returns the repo root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "commit.gpgsign", "false"],
        check=True,
    )
    # Need an initial commit so the index is usable.
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True
    )
    return repo


def _stage_file(
    repo: Path, rel: str, content: str, env: dict | None = None
) -> dict:
    """Write rel with content; `git add`; return env (the env passed in)."""
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    e = dict(env or os.environ)
    subprocess.run(
        ["git", "-C", str(repo), "add", "--", rel],
        check=True,
        env=e,
    )
    return e


def _sha256_of_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -------- _hash_staged_files --------


def test_hash_staged_files_returns_hash_for_staged_file(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "foo.txt", "hello\n")
    result = _hash_staged_files(["foo.txt"], env)
    assert result["foo.txt"] == _sha256_of_string("hello\n")


def test_hash_staged_files_handles_unstaged_file(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    # File doesn't exist; not staged.
    result = _hash_staged_files(["nonexistent.txt"], dict(os.environ))
    assert result["nonexistent.txt"] == "NOT_STAGED"


def test_hash_staged_files_multi_file(tmp_path: Path, monkeypatch) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "a.txt", "AAA\n")
    _stage_file(repo, "b.txt", "BBB\n", env=env)
    result = _hash_staged_files(["a.txt", "b.txt"], env)
    assert result["a.txt"] == _sha256_of_string("AAA\n")
    assert result["b.txt"] == _sha256_of_string("BBB\n")


def test_hash_staged_files_reflects_index_not_working_tree(
    tmp_path: Path, monkeypatch
) -> None:
    """If working tree diverges from index, hash returns INDEX content."""
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "foo.txt", "staged_content\n")
    # Now modify working tree without staging.
    (repo / "foo.txt").write_text("working_tree_modified\n", encoding="utf-8")
    result = _hash_staged_files(["foo.txt"], env)
    # Should still match the STAGED content, not the working tree.
    assert result["foo.txt"] == _sha256_of_string("staged_content\n")


# -------- _staged_content_check --------


def test_staged_content_check_clean_when_matches(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "foo.txt", "hello\n")
    expected = {"foo.txt": _sha256_of_string("hello\n")}
    diffs = _staged_content_check(expected, env)
    assert diffs == {}


def test_staged_content_check_reports_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "foo.txt", "actual_content\n")
    expected = {"foo.txt": _sha256_of_string("declared_content\n")}
    diffs = _staged_content_check(expected, env)
    assert "foo.txt" in diffs
    want, got = diffs["foo.txt"]
    assert want == _sha256_of_string("declared_content\n")
    assert got == _sha256_of_string("actual_content\n")


def test_staged_content_check_empty_expected_returns_empty(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    diffs = _staged_content_check({}, dict(os.environ))
    assert diffs == {}


def test_staged_content_check_simulates_absorbed_edits_race(
    tmp_path: Path, monkeypatch
) -> None:
    """Empirical anchor: 5d0ec061d absorbed Catalog #215 edits.

    Subagent A edits foo.txt -> "A_edit". Subagent B edits foo.txt ->
    "A_edit\nB_edit" (working tree contains BOTH edits because subagent A's
    edit landed on disk before B started). B declares sha of "B_edit" alone
    (its mental model of what its work should produce). B's `git add`
    stages the merged content. The Catalog #216 check correctly refuses B.
    """
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    # B's "what I intended to commit" sha: only B's edits.
    declared_sha = _sha256_of_string("B_only_edit\n")
    # Reality: working tree contains A's edits AND B's edits.
    env = _stage_file(repo, "foo.txt", "A_edit\nB_only_edit\n")
    expected = {"foo.txt": declared_sha}
    diffs = _staged_content_check(expected, env)
    assert "foo.txt" in diffs
    want, got = diffs["foo.txt"]
    assert want == declared_sha
    assert got == _sha256_of_string("A_edit\nB_only_edit\n")


def test_staged_content_check_missing_file_reports_not_staged(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    expected = {"nonexistent.txt": _sha256_of_string("anything")}
    diffs = _staged_content_check(expected, dict(os.environ))
    assert "nonexistent.txt" in diffs
    want, got = diffs["nonexistent.txt"]
    assert got == "NOT_STAGED"


def test_staged_content_check_multiple_files_partial_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "a.txt", "AAA\n")
    _stage_file(repo, "b.txt", "BBB_ACTUAL\n", env=env)
    expected = {
        "a.txt": _sha256_of_string("AAA\n"),  # matches
        "b.txt": _sha256_of_string("BBB_DECLARED\n"),  # mismatch
    }
    diffs = _staged_content_check(expected, env)
    assert "a.txt" not in diffs
    assert "b.txt" in diffs


def test_staged_content_check_temp_index_isolated(
    tmp_path: Path, monkeypatch
) -> None:
    """Using a custom GIT_INDEX_FILE reads ONLY the custom index, not real."""
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    # Stage foo.txt into the REAL index.
    env_real = _stage_file(repo, "foo.txt", "real_content\n")
    # Now create a custom temp index and stage DIFFERENT content there.
    temp_index = tmp_path / "temp.idx"
    # Initialize the temp index by copying from the real one (matches
    # _make_temp_index behavior).
    subprocess.run(
        ["cp", str(repo / ".git" / "index"), str(temp_index)],
        check=True,
    )
    env_temp = {**env_real, "GIT_INDEX_FILE": str(temp_index)}
    # Modify the working tree.
    (repo / "foo.txt").write_text("temp_index_content\n", encoding="utf-8")
    # Stage to the TEMP index.
    subprocess.run(
        ["git", "-C", str(repo), "add", "--", "foo.txt"],
        check=True,
        env=env_temp,
    )
    # The temp index now has the new content; the real index has the old.
    result_temp = _hash_staged_files(["foo.txt"], env_temp)
    assert result_temp["foo.txt"] == _sha256_of_string("temp_index_content\n")
    # Read the real index by removing GIT_INDEX_FILE.
    env_real_only = {k: v for k, v in env_temp.items() if k != "GIT_INDEX_FILE"}
    result_real = _hash_staged_files(["foo.txt"], env_real_only)
    assert result_real["foo.txt"] == _sha256_of_string("real_content\n")


def test_staged_content_check_handles_files_with_subpath(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "subdir/nested.txt", "nested_content\n")
    expected = {"subdir/nested.txt": _sha256_of_string("nested_content\n")}
    diffs = _staged_content_check(expected, env)
    assert diffs == {}


def test_staged_content_check_idempotent_double_call(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _init_test_repo(tmp_path)
    monkeypatch.setattr(
        "tools.subagent_commit_serializer.REPO_ROOT", repo
    )
    env = _stage_file(repo, "foo.txt", "content\n")
    expected = {"foo.txt": _sha256_of_string("content\n")}
    once = _staged_content_check(expected, env)
    twice = _staged_content_check(expected, env)
    assert once == twice == {}
