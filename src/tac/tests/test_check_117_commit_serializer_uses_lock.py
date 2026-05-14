# SPDX-License-Identifier: MIT
"""Catalog #117 commit-serializer usage regressions."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.preflight import (
    _check_117_serializer_sha_matches_commit,
    check_subagent_commit_serializer_uses_lock,
)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(
        repo,
        "commit",
        "--allow-empty",
        "-m",
        "initial",
        "-m",
        "# NO_SERIALIZER_OK:test-fixture-bootstrap",
    )
    (repo / ".omx" / "state").mkdir(parents=True)


def _commit(repo: Path, subject: str) -> str:
    _run_git(repo, "commit", "--allow-empty", "-m", subject)
    return _run_git(repo, "rev-parse", "HEAD")


def _append_serializer_row(
    repo: Path,
    *,
    sha: str,
    prefix_len: int = 9,
    commit_rc: int = 0,
    outcome: str = "committed",
) -> None:
    log = repo / ".omx" / "state" / "commit-serializer.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "commit_rc": commit_rc,
                    "head_after": sha[:prefix_len],
                    "outcome": outcome,
                }
            )
            + "\n"
        )


def test_check_117_matches_nine_char_serializer_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha = _commit(repo, "subagent serialized commit")
    _append_serializer_row(repo, sha=sha, prefix_len=9)

    violations = check_subagent_commit_serializer_uses_lock(
        repo_root=repo,
        last_n_commits=5,
    )

    assert violations == []


def test_check_117_failed_serializer_attempt_does_not_bless_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha = _commit(repo, "commit created before serializer reached git commit")
    _append_serializer_row(repo, sha=sha, prefix_len=9, commit_rc=1, outcome="commit_failed")

    violations = check_subagent_commit_serializer_uses_lock(
        repo_root=repo,
        last_n_commits=5,
    )

    assert len(violations) == 1
    assert sha[:7] in violations[0]


def test_check_117_rejects_nonmatching_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha = _commit(repo, "subagent commit without matching serializer row")
    bad_prefix = ("0" if sha[0] != "0" else "1") + sha[1:9]
    _append_serializer_row(repo, sha=bad_prefix + sha[9:], prefix_len=9)

    violations = check_subagent_commit_serializer_uses_lock(
        repo_root=repo,
        last_n_commits=5,
    )

    assert len(violations) == 1
    assert sha[:7] in violations[0]


def test_check_117_prefix_match_boundaries() -> None:
    sha = "abcdef1234567890abcdef1234567890abcdef12"
    assert _check_117_serializer_sha_matches_commit(sha, "abcdef123")
    assert _check_117_serializer_sha_matches_commit(sha, sha)
    assert not _check_117_serializer_sha_matches_commit(sha, "abcdef")
    assert not _check_117_serializer_sha_matches_commit(sha, "abcdee123")
    assert not _check_117_serializer_sha_matches_commit(sha, "not-a-sha")
