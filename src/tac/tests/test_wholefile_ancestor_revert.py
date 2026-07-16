"""Tests for check_no_wholefile_ancestor_revert (#511 — stale-base merge clobber).

The gate detects the stale-base whole-file merge-clobber signature over the staged
diff: a src/tac|tools module whose staged (index) blob is byte-identical to a
strictly-older committed version of the file (an ancestor blob, not HEAD) while the
diff from HEAD is a full-file replacement (>90% of HEAD lines removed, >100 churn).
It is STATIC + FAIL-OPEN over a real git repo.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_wholefile_ancestor_revert,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgsign", "false")
    return root


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _commit(root: Path, rel: str, text: str, msg: str) -> None:
    _write(root, rel, text)
    _git(root, "add", "--", rel)
    _git(root, "commit", "-q", "-m", msg)


_OLD = "\n".join(f"# old line {i}" for i in range(200)) + "\n"
_NEW = "\n".join(f"# NEW content line {i}" for i in range(200)) + "\n"


def _repo_with_revert(tmp_path: Path, rel: str = "src/tac/mod.py") -> Path:
    """Commit OLD then NEW to `rel`, then STAGE a whole-file revert back to OLD."""
    root = _init_repo(tmp_path)
    _commit(root, rel, _OLD, "A: old")      # ancestor blob
    _commit(root, rel, _NEW, "B: new")      # HEAD blob
    _write(root, rel, _OLD)                  # revert working tree to OLD
    _git(root, "add", "--", rel)             # stage the revert
    return root


# ---- positive (the clobber signature fires) --------------------------------

def test_staged_wholefile_revert_flagged(tmp_path: Path) -> None:
    root = _repo_with_revert(tmp_path)
    violations = check_no_wholefile_ancestor_revert(repo_root=root)
    assert len(violations) == 1
    assert "src/tac/mod.py" in violations[0]
    assert "ancestor blob" in violations[0]


def test_flagged_names_matching_ancestor_commit(tmp_path: Path) -> None:
    root = _repo_with_revert(tmp_path)
    # the matched commit is commit A (the ancestor with OLD content)
    log = subprocess.run(
        ["git", "-C", str(root), "log", "--format=%H %s"],
        capture_output=True, text=True, check=True).stdout
    commit_a = next(ln.split()[0] for ln in log.splitlines() if "A: old" in ln)
    violations = check_no_wholefile_ancestor_revert(repo_root=root)
    assert commit_a[:12] in violations[0]


def test_tools_scope_flagged(tmp_path: Path) -> None:
    root = _repo_with_revert(tmp_path, rel="tools/helper.py")
    violations = check_no_wholefile_ancestor_revert(repo_root=root)
    assert len(violations) == 1
    assert "tools/helper.py" in violations[0]


def test_strict_raises_on_revert(tmp_path: Path) -> None:
    root = _repo_with_revert(tmp_path)
    with pytest.raises(PreflightError):
        check_no_wholefile_ancestor_revert(repo_root=root, strict=True)


# ---- negative (must NOT fire) ----------------------------------------------

def test_clean_new_content_not_flagged(tmp_path: Path) -> None:
    """Staged content that is NOT any ancestor blob is a normal edit -> no flag."""
    root = _init_repo(tmp_path)
    _commit(root, "src/tac/mod.py", _OLD, "A")
    _commit(root, "src/tac/mod.py", _NEW, "B")
    unique = "\n".join(f"# unique fresh {i}" for i in range(200)) + "\n"
    _write(root, "src/tac/mod.py", unique)
    _git(root, "add", "--", "src/tac/mod.py")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_small_churn_revert_not_flagged(tmp_path: Path) -> None:
    """A revert under the churn threshold is not a whole-file clobber."""
    root = _init_repo(tmp_path)
    small_old = "\n".join(f"l{i}" for i in range(10)) + "\n"
    small_new = "\n".join(f"N{i}" for i in range(10)) + "\n"
    _commit(root, "src/tac/mod.py", small_old, "A")
    _commit(root, "src/tac/mod.py", small_new, "B")
    _write(root, "src/tac/mod.py", small_old)
    _git(root, "add", "--", "src/tac/mod.py")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_partial_edit_not_flagged(tmp_path: Path) -> None:
    """Editing a few lines (not a full-file replacement) is not the signature."""
    root = _init_repo(tmp_path)
    _commit(root, "src/tac/mod.py", _OLD, "A")
    _commit(root, "src/tac/mod.py", _NEW, "B")
    # tweak two lines of NEW -> mostly-preserved, not a full replacement, and
    # the result is not an ancestor blob either.
    edited = _NEW.replace("# NEW content line 0", "# tweaked 0").replace(
        "# NEW content line 1", "# tweaked 1")
    _write(root, "src/tac/mod.py", edited)
    _git(root, "add", "--", "src/tac/mod.py")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_out_of_scope_path_not_flagged(tmp_path: Path) -> None:
    """A whole-file revert of a docs/ file is out of scope."""
    root = _repo_with_revert(tmp_path, rel="docs/thing.md")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_no_staged_changes_not_flagged(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    _commit(root, "src/tac/mod.py", _OLD, "A")
    _commit(root, "src/tac/mod.py", _NEW, "B")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_new_file_no_head_version_not_flagged(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    _commit(root, "src/tac/other.py", _OLD, "A")
    _write(root, "src/tac/fresh.py", _NEW)
    _git(root, "add", "--", "src/tac/fresh.py")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_not_a_git_repo_fail_open(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert check_no_wholefile_ancestor_revert(repo_root=plain) == []


def test_waiver_suppresses(tmp_path: Path) -> None:
    """A same-line waiver in the staged content honors an intentional rollback."""
    root = _init_repo(tmp_path)
    old_waived = _OLD + "# WHOLEFILE_ANCESTOR_REVERT_OK:deliberate rollback of bad landing\n"
    _commit(root, "src/tac/mod.py", old_waived, "A")
    _commit(root, "src/tac/mod.py", _NEW, "B")
    _write(root, "src/tac/mod.py", old_waived)
    _git(root, "add", "--", "src/tac/mod.py")
    assert check_no_wholefile_ancestor_revert(repo_root=root) == []


def test_placeholder_waiver_rejected(tmp_path: Path) -> None:
    """A placeholder <rationale> waiver must NOT self-suppress."""
    root = _init_repo(tmp_path)
    old_ph = _OLD + "# WHOLEFILE_ANCESTOR_REVERT_OK:<rationale>\n"
    _commit(root, "src/tac/mod.py", old_ph, "A")
    _commit(root, "src/tac/mod.py", _NEW, "B")
    _write(root, "src/tac/mod.py", old_ph)
    _git(root, "add", "--", "src/tac/mod.py")
    violations = check_no_wholefile_ancestor_revert(repo_root=root)
    assert len(violations) == 1


def test_verbose_ok_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _init_repo(tmp_path)
    _commit(root, "src/tac/mod.py", _OLD, "A")
    check_no_wholefile_ancestor_revert(repo_root=root, verbose=True)
    out = capsys.readouterr().out
    assert "check_no_wholefile_ancestor_revert" in out


def test_live_repo_does_not_raise() -> None:
    """The real repo must not raise in warn-only mode (whatever the staged state)."""
    result = check_no_wholefile_ancestor_revert(strict=False)
    assert isinstance(result, list)


def test_three_commit_history_matches_older_ancestor(tmp_path: Path) -> None:
    """Revert to the OLDEST of three versions is still flagged (ancestor blob)."""
    root = _init_repo(tmp_path)
    _commit(root, "src/tac/mod.py", _OLD, "A: oldest")
    mid = "\n".join(f"# mid {i}" for i in range(200)) + "\n"
    _commit(root, "src/tac/mod.py", mid, "B: mid")
    _commit(root, "src/tac/mod.py", _NEW, "C: head")
    _write(root, "src/tac/mod.py", _OLD)  # revert all the way back to A
    _git(root, "add", "--", "src/tac/mod.py")
    violations = check_no_wholefile_ancestor_revert(repo_root=root)
    assert len(violations) == 1
