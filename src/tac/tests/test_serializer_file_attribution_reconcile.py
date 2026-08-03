# SPDX-License-Identifier: MIT
"""FIX-ATTRIBUTION (task #911): the serializer must COMPARE requested vs recorded files.

THE INCIDENT (2026-08-02, verified at source): `git show --stat 06fa0ad37d` records
`experiments/train_tr1_partition_renderer_mlx.py | 215 +++...` inside a commit whose own
message says "every edit is one % -> %% inside a help string". A sibling arm's 215 lines
were absorbed. The serializer printed `files=6`; git recorded 5. Two components emitted
contradictory counts in the same run and nothing compared them.

WHY EVERY EXISTING CHECK PASSED — the projection is wrong, not the arithmetic.
`_post_commit_content_check` asks "is the declared CONTENT at HEAD?" When a sibling has
already committed byte-identical content, the HEAD blob matches and the check passes BY
CONSTRUCTION. It is structurally incapable of asking "did MY commit put it there?"

These tests cover the complement. Warn-only by design: `requested - recorded` has a benign
cause (a requested file whose content already equals HEAD yields no diff), so refusing
would block honest commits. The cure is that the two sets are COMPARED at all.
"""
from __future__ import annotations

import subprocess

import pytest

from tools import subagent_commit_serializer as scs


def _git(repo, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    )
    return proc.stdout


@pytest.fixture()
def tmp_repo(tmp_path):
    """A real git repo — no mocking of git itself, per measure-don't-assume."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    return repo


def _commit(repo, name: str, body: str, msg: str) -> None:
    (repo / name).write_text(body)
    _git(repo, "add", "--", name)
    _git(repo, "commit", "-q", "-m", msg)


def test_helper_matches_git_on_the_real_repo(monkeypatch):
    """NEGATIVE CONTROL + DENOMINATOR: on the live repo the helper must equal
    `git show` exactly, and must be NON-EMPTY — an empty return would make every
    downstream comparison vacuously clean (the vacuity genus this arm audits)."""
    got = scs._files_recorded_by_head_commit()
    assert got is not None, "helper failed open on the live repo"
    raw = subprocess.run(
        ["git", "show", "--pretty=format:", "--name-only", "HEAD"],
        cwd=scs.REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    expected = {ln.strip() for ln in raw.splitlines() if ln.strip()}
    assert got == expected
    assert got, "HEAD recorded zero files — denominator is vacuous, not clean"


def test_positive_control_absorbed_file_is_detected(tmp_repo, monkeypatch):
    """POSITIVE CONTROL — the incident's exact shape.

    A sibling commits `absorbed.py`. We then commit `mine.py` while still BELIEVING
    we are committing both. `requested - recorded` must surface `absorbed.py`.
    """
    monkeypatch.setattr(scs, "REPO_ROOT", tmp_repo)
    _commit(tmp_repo, "seed.txt", "seed\n", "seed")
    _commit(tmp_repo, "absorbed.py", "x = 1\n", "sibling absorbs it")
    _commit(tmp_repo, "mine.py", "y = 2\n", "my landing")

    recorded = scs._files_recorded_by_head_commit()
    assert recorded == {"mine.py"}

    requested = {"mine.py", "absorbed.py"}
    absent = requested - recorded
    assert absent == {"absorbed.py"}, "the absorbed file must be surfaced"
    assert len(requested) != len(recorded), (
        "the count mismatch (6 vs 5 in the real incident) must be visible"
    )


def test_negative_control_clean_commit_reconciles(tmp_repo, monkeypatch):
    """NEGATIVE CONTROL — an honest multi-file commit must produce NO mismatch
    in either direction. A check that fires on clean commits is worse than none."""
    monkeypatch.setattr(scs, "REPO_ROOT", tmp_repo)
    _commit(tmp_repo, "seed.txt", "seed\n", "seed")
    for n in ("a.py", "b.py"):
        (tmp_repo / n).write_text(f"# {n}\n")
    _git(tmp_repo, "add", "--", "a.py", "b.py")
    _git(tmp_repo, "commit", "-q", "-m", "honest two-file landing")

    recorded = scs._files_recorded_by_head_commit()
    requested = {"a.py", "b.py"}
    assert recorded == requested
    assert not (requested - recorded) and not (recorded - requested)


def test_reverse_direction_we_absorbed_a_sibling(tmp_repo, monkeypatch):
    """The OTHER direction (the 2026-04-29 anchor incident): our commit carries a
    file we never requested. `recorded - requested` must surface it."""
    monkeypatch.setattr(scs, "REPO_ROOT", tmp_repo)
    _commit(tmp_repo, "seed.txt", "seed\n", "seed")
    for n in ("mine.py", "siblings.py"):
        (tmp_repo / n).write_text(f"# {n}\n")
    _git(tmp_repo, "add", "-A")          # the forbidden whole-tree add
    _git(tmp_repo, "commit", "-q", "-m", "swept a sibling in")

    recorded = scs._files_recorded_by_head_commit()
    requested = {"mine.py"}
    assert (recorded - requested) == {"siblings.py"}


def test_helper_fails_open_not_closed(tmp_path, monkeypatch):
    """FAIL-OPEN: a non-repo path must return None, never raise. This diagnostic
    may never be the reason a real commit dies."""
    monkeypatch.setattr(scs, "REPO_ROOT", tmp_path)
    assert scs._files_recorded_by_head_commit() is None
