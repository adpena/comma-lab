# SPDX-License-Identifier: MIT
"""Regression tests for Catalog #234 and #235 meta commit gates."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.preflight import (
    check_no_sha_prefix_length_mismatch_comparisons,
    check_subagent_commit_bodies_not_empty,
)


def _git(repo: Path, *args: str) -> str:
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
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / ".omx" / "state").mkdir(parents=True)


def _commit(repo: Path, subject: str, body: str = "") -> str:
    args = ["commit", "--allow-empty", "-m", subject]
    if body:
        args.extend(["-m", body])
    _git(repo, *args)
    return _git(repo, "rev-parse", "HEAD")


def _append_serializer_row(repo: Path, sha: str, started_at_utc: str) -> None:
    log = repo / ".omx" / "state" / "commit-serializer.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "outcome": "committed",
                    "commit_rc": 0,
                    "head_after": sha[:9],
                    "started_at_utc": started_at_utc,
                }
            )
            + "\n"
        )


def test_check_234_flags_post_cutoff_subject_only_subagent_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    sha = _commit(repo, "subject only")
    _append_serializer_row(repo, sha, "9999-12-31T00:00:00Z")

    violations = check_subagent_commit_bodies_not_empty(
        repo_root=repo,
        last_n_commits=5,
    )

    assert len(violations) == 1
    assert sha[:10] in violations[0]
    assert "EMPTY body" in violations[0]


def test_check_234_accepts_marker_body_and_exempts_pre_cutoff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    old_sha = _commit(repo, "legacy subject only")
    _append_serializer_row(repo, old_sha, "2026-05-14T01:00:00Z")
    good_sha = _commit(
        repo,
        "subagent body present",
        "Lane: lane_fixture\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
    )
    _append_serializer_row(repo, good_sha, "9999-12-31T00:00:00Z")

    assert check_subagent_commit_bodies_not_empty(
        repo_root=repo,
        last_n_commits=5,
    ) == []


def test_check_235_flags_literal_prefix_length_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target = repo / "src" / "tac"
    target.mkdir(parents=True)
    (target / "preflight.py").write_text(
        "def bad(short_sha, seen_hashes):\n"
        "    return short_sha[:7] in seen_hashes\n",
        encoding="utf-8",
    )

    violations = check_no_sha_prefix_length_mismatch_comparisons(repo_root=repo)

    assert len(violations) == 1
    assert "[:7] in seen_hashes" in violations[0]


def test_check_235_allows_same_line_waiver(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target = repo / "src" / "tac"
    target.mkdir(parents=True)
    (target / "preflight.py").write_text(
        "def fixture(short_sha, seen_hashes):\n"
        "    return short_sha[:7] in seen_hashes  # SHA_PREFIX_LENGTH_MISMATCH_OK:test fixture\n",
        encoding="utf-8",
    )

    assert check_no_sha_prefix_length_mismatch_comparisons(repo_root=repo) == []
