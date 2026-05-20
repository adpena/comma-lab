# SPDX-License-Identifier: MIT
"""Tests for ``tac.commit_safety.pre_write_sister_check`` + the CLI helper
at ``tools/check_sister_files_recently_landed.py``.

Per CLAUDE.md "Subagent coherence-by-default" + "Bugs must be permanently
fixed AND self-protected against" non-negotiables + the
WAVE-3-PRE-WRITE-SISTER-ACTIVITY-CHECK-HELPER 2026-05-20 lane closing the
NERV-FAMILY-L0-BUILD empirical anchor (~30 min wasted duplicating sister
commit ``18b0beed6``'s ego_nerv + e_nerv + nervdc trainers because PV did
not run ``git log -- <target>`` before first Write).
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Tests run from the repo root; ensure src/ is importable like the rest
# of the canonical test suite.
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.commit_safety.pre_write_sister_check import (  # noqa: E402
    DEFAULT_LOOKBACK_HOURS,
    STAND_DOWN_FILE_OVERLAP_THRESHOLD,
    SisterRecentlyLandedVerdict,
    check_sister_files_recently_landed,
)


# ── Fixtures ─────────────────────────────────────────────────────────────
def _make_test_repo(tmp_path: Path) -> tuple[Path, dict]:
    """Initialize a minimal git repo for sister-commit simulation.

    Returns (repo_root, env-dict). The env-dict has GIT_AUTHOR_*, etc. set
    so commits land without operator config.
    """
    repo = tmp_path / "test_repo"
    repo.mkdir()

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "TestAuthor",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "TestAuthor",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
    }
    # Empty global gitconfig prevents user config from interfering.
    (tmp_path / "gitconfig").write_text("")

    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"],
        cwd=repo, env=env, check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "init"],
        cwd=repo, env=env, check=True,
    )
    return repo, env


def _commit_file(
    repo: Path,
    env: dict,
    rel_path: str,
    content: str,
    *,
    message: str,
    extra_body: str = "",
    backdate_hours: float = 0.0,
) -> str:
    """Create file, commit it, return short sha. Optionally backdate."""
    full = repo / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    subprocess.run(["git", "add", rel_path], cwd=repo, env=env, check=True)
    full_message = message
    if extra_body:
        full_message = f"{message}\n\n{extra_body}"

    commit_env = dict(env)
    if backdate_hours > 0:
        when = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=backdate_hours)
        # ISO 8601 with timezone for git
        when_str = when.strftime("%Y-%m-%dT%H:%M:%S%z")
        commit_env["GIT_AUTHOR_DATE"] = when_str
        commit_env["GIT_COMMITTER_DATE"] = when_str

    subprocess.run(
        ["git", "commit", "-q", "-m", full_message],
        cwd=repo, env=commit_env, check=True,
    )
    sha_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo, env=env, capture_output=True, text=True, check=True,
    )
    return sha_result.stdout.strip()


# ── Helper unit tests ────────────────────────────────────────────────────
class TestCheckSisterFilesRecentlyLanded:
    """Helper unit tests covering all recommendation paths + edge cases."""

    def test_clean_repo_no_sister_activity_proceeds(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        verdict = check_sister_files_recently_landed(
            ["src/tac/foo.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.recommendation == "PROCEED"
        assert verdict.sister_commits == ()
        assert verdict.file_to_sister_commits == ()
        assert verdict.has_sister_activity() is False
        assert verdict.lookback_hours == 6
        assert verdict.target_files == ("src/tac/foo.py",)

    def test_sister_commit_on_target_file_in_lookback_stands_down(self, tmp_path):
        """The canonical NERV-FAMILY-L0-BUILD anchor regression at unit scale."""
        repo, env = _make_test_repo(tmp_path)
        sha = _commit_file(
            repo, env,
            "experiments/train_substrate_ego_nerv.py",
            "# ego_nerv L0 SCAFFOLD\n",
            message="substrates: land ego_nerv L0 SCAFFOLD per BUILD-1 NeRV-trio queue fill",
            backdate_hours=4.5,  # NERV-FAMILY-L0-BUILD anchor delay
        )
        verdict = check_sister_files_recently_landed(
            ["experiments/train_substrate_ego_nerv.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.recommendation == "STAND_DOWN_DUPLICATE"
        assert len(verdict.sister_commits) == 1
        assert verdict.sister_commits[0][0] == sha
        assert verdict.file_to_sister_commits == (
            ("experiments/train_substrate_ego_nerv.py", (sha,)),
        )
        assert verdict.has_sister_activity() is True
        assert "STAND DOWN" in verdict.rationale

    def test_sister_commit_outside_lookback_window_proceeds(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        # Sister commit landed 10 hours ago; lookback 6 hours → out of window.
        _commit_file(
            repo, env,
            "src/tac/foo.py",
            "# stale\n",
            message="ancient commit",
            backdate_hours=10.0,
        )
        verdict = check_sister_files_recently_landed(
            ["src/tac/foo.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.recommendation == "PROCEED"
        assert verdict.sister_commits == ()
        assert "no sister commits" in verdict.rationale.lower()

    def test_own_subagent_id_filter_via_co_authored_by(self, tmp_path):
        """Commits whose body mentions own_subagent_id should be filtered."""
        repo, env = _make_test_repo(tmp_path)
        # Sister commit (no own_subagent_id mention) → would normally STAND_DOWN.
        _commit_file(
            repo, env,
            "src/tac/foo.py",
            "# sister\n",
            message="sister: real work",
            extra_body="Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
            backdate_hours=1.0,
        )
        # Caller's own commit on a different file with their subagent_id in body.
        _commit_file(
            repo, env,
            "src/tac/bar.py",
            "# my work\n",
            message="my work",
            extra_body=(
                "Subagent: wave-3-pre-write-sister-activity-check-helper-20260520\n"
                "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
            ),
            backdate_hours=0.5,
        )
        # When checking foo.py with own_subagent_id, foo.py was touched by
        # sister (NOT us) → STAND_DOWN.
        verdict = check_sister_files_recently_landed(
            ["src/tac/foo.py"],
            repo_root=repo,
            lookback_hours=6,
            own_subagent_id="wave-3-pre-write-sister-activity-check-helper-20260520",
        )
        assert verdict.recommendation == "STAND_DOWN_DUPLICATE"
        # When checking bar.py (which WE touched), own-commit filter removes
        # the row → PROCEED.
        verdict_self = check_sister_files_recently_landed(
            ["src/tac/bar.py"],
            repo_root=repo,
            lookback_hours=6,
            own_subagent_id="wave-3-pre-write-sister-activity-check-helper-20260520",
        )
        assert verdict_self.recommendation == "PROCEED"

    def test_multi_file_overlap_aggregation_stand_down(self, tmp_path):
        """Single sister commit touching multiple target files → STAND_DOWN_DUPLICATE."""
        repo, env = _make_test_repo(tmp_path)
        # ONE commit touching THREE files (the NERV-FAMILY-L0-BUILD pattern).
        for rel in (
            "experiments/train_substrate_ego_nerv.py",
            "experiments/train_substrate_e_nerv.py",
            "experiments/train_substrate_nervdc.py",
        ):
            full = repo / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text("# scaffold\n")
            subprocess.run(["git", "add", rel], cwd=repo, env=env, check=True)
        sha = _commit_file_no_create(
            repo, env,
            message="substrates: land 3 L0 SCAFFOLD NeRV trainers per BUILD-1",
            backdate_hours=4.5,
        )
        verdict = check_sister_files_recently_landed(
            [
                "experiments/train_substrate_ego_nerv.py",
                "experiments/train_substrate_e_nerv.py",
                "experiments/train_substrate_nervdc.py",
            ],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.recommendation == "STAND_DOWN_DUPLICATE"
        # One unique sister sha (matches NERV-FAMILY-L0-BUILD: 1 commit, 3 files).
        assert len(verdict.sister_commits) == 1
        assert verdict.sister_commits[0][0] == sha
        # All 3 files have overlap entries.
        assert len(verdict.file_to_sister_commits) == 3

    def test_partial_overlap_multiple_sisters_wait_and_reassess(self, tmp_path):
        """Multiple sisters touching some-but-not-all target files → WAIT_AND_REASSESS."""
        repo, env = _make_test_repo(tmp_path)
        # Sister A touches file 1 only.
        _commit_file(
            repo, env,
            "src/tac/a.py", "# a\n",
            message="sister-a work",
            backdate_hours=1.0,
        )
        # Sister B touches file 2 only.
        _commit_file(
            repo, env,
            "src/tac/b.py", "# b\n",
            message="sister-b work",
            backdate_hours=2.0,
        )
        # Caller targets 4 files. Sister coverage: 2 of 4 (50%) but split
        # across 2 different sisters AND no single sister covers majority.
        verdict = check_sister_files_recently_landed(
            ["src/tac/a.py", "src/tac/b.py", "src/tac/c.py", "src/tac/d.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        # 2-of-4 file overlap; max single-sister overlap = 1 < ceil(4/2) = 2.
        # So: WAIT_AND_REASSESS.
        assert verdict.recommendation == "WAIT_AND_REASSESS"
        assert len(verdict.sister_commits) == 2
        assert len(verdict.file_to_sister_commits) == 2
        assert "re-read sister landing memos" in verdict.rationale

    def test_corrupt_git_repo_raises_value_error(self, tmp_path):
        """Helper fail-closed when not a git repo per Catalog #229 PV discipline."""
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        with pytest.raises(ValueError, match="git log failed"):
            check_sister_files_recently_landed(
                ["src/tac/foo.py"],
                repo_root=non_git,
                lookback_hours=6,
            )

    def test_empty_files_list_proceeds(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        verdict = check_sister_files_recently_landed(
            [],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.recommendation == "PROCEED"
        assert verdict.sister_commits == ()
        assert "no target files" in verdict.rationale.lower()

    def test_sister_recently_landed_verdict_frozen_invariant(self, tmp_path):
        """The dataclass is frozen → attempts to mutate raise FrozenInstanceError."""
        repo, env = _make_test_repo(tmp_path)
        verdict = check_sister_files_recently_landed(
            ["src/tac/foo.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            verdict.recommendation = "STAND_DOWN_DUPLICATE"  # type: ignore[misc]

    def test_files_accepts_path_objects(self, tmp_path):
        """Sequence[str | Path] — Path objects should be normalized."""
        repo, env = _make_test_repo(tmp_path)
        verdict = check_sister_files_recently_landed(
            [Path("src/tac/foo.py"), "src/tac/bar.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.recommendation == "PROCEED"
        assert verdict.target_files == ("src/tac/bar.py", "src/tac/foo.py")

    def test_files_rejects_non_string_non_path(self):
        """TypeError on non-str / non-Path elements."""
        with pytest.raises(TypeError, match="files must be Sequence"):
            check_sister_files_recently_landed(
                [123, "src/tac/foo.py"],  # type: ignore[list-item]
                lookback_hours=6,
            )

    def test_files_dedups_and_sorts(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        verdict = check_sister_files_recently_landed(
            ["src/tac/b.py", "src/tac/a.py", "src/tac/b.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert verdict.target_files == ("src/tac/a.py", "src/tac/b.py")


def _commit_file_no_create(
    repo: Path,
    env: dict,
    *,
    message: str,
    extra_body: str = "",
    backdate_hours: float = 0.0,
) -> str:
    """Commit already-staged files (multi-file commits)."""
    full_message = message
    if extra_body:
        full_message = f"{message}\n\n{extra_body}"
    commit_env = dict(env)
    if backdate_hours > 0:
        when = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=backdate_hours)
        when_str = when.strftime("%Y-%m-%dT%H:%M:%S%z")
        commit_env["GIT_AUTHOR_DATE"] = when_str
        commit_env["GIT_COMMITTER_DATE"] = when_str
    subprocess.run(
        ["git", "commit", "-q", "-m", full_message],
        cwd=repo, env=commit_env, check=True,
    )
    sha_result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo, env=env, capture_output=True, text=True, check=True,
    )
    return sha_result.stdout.strip()


# ── CLI tests ────────────────────────────────────────────────────────────
def _run_cli(
    repo: Path,
    env: dict,
    files: list[str],
    *,
    lookback_hours: int = 6,
    own_subagent_id: str | None = None,
) -> subprocess.CompletedProcess:
    """Invoke the canonical CLI helper as a subprocess."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "check_sister_files_recently_landed.py"),
        "--files", *files,
        "--lookback-hours", str(lookback_hours),
        "--repo-root", str(repo),
    ]
    if own_subagent_id:
        cmd.extend(["--own-subagent-id", own_subagent_id])
    return subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)


class TestCLI:
    def test_cli_proceeds_on_clean_repo(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        result = _run_cli(repo, env, ["src/tac/foo.py"])
        assert result.returncode == 0, (
            f"expected rc=0 PROCEED on clean repo; "
            f"got rc={result.returncode}, stderr={result.stderr}"
        )
        assert "OK" in result.stdout
        assert "PROCEED" in result.stdout

    def test_cli_rc8_on_synthetic_sister_commit(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        _commit_file(
            repo, env,
            "experiments/train_substrate_ego_nerv.py",
            "# scaffold\n",
            message="sister: land L0 scaffold",
            backdate_hours=2.0,
        )
        result = _run_cli(repo, env, ["experiments/train_substrate_ego_nerv.py"])
        assert result.returncode == 8, (
            f"expected rc=8 STAND_DOWN; got rc={result.returncode}, "
            f"stderr={result.stderr}"
        )
        assert "STAND_DOWN_DUPLICATE" in result.stderr
        assert "STAND DOWN" in result.stderr

    def test_cli_rc9_on_ambiguous_overlap(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        # Two sisters, each touching one of four targets.
        _commit_file(repo, env, "src/tac/a.py", "# a\n",
                     message="sister-a", backdate_hours=1.0)
        _commit_file(repo, env, "src/tac/b.py", "# b\n",
                     message="sister-b", backdate_hours=2.0)
        result = _run_cli(
            repo, env,
            ["src/tac/a.py", "src/tac/b.py", "src/tac/c.py", "src/tac/d.py"],
        )
        assert result.returncode == 9, (
            f"expected rc=9 WAIT_AND_REASSESS; got rc={result.returncode}, "
            f"stderr={result.stderr}"
        )
        assert "WAIT_AND_REASSESS" in result.stderr

    def test_cli_rc2_on_bad_repo_root(self, tmp_path):
        """CLI surfaces ValueError as rc=2."""
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = str(tmp_path / "gitconfig")
        (tmp_path / "gitconfig").write_text("")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "check_sister_files_recently_landed.py"),
            "--files", "src/tac/foo.py",
            "--repo-root", str(non_git),
        ]
        result = subprocess.run(
            cmd, cwd=non_git, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert "ERROR" in result.stderr

    def test_cli_files_from_stdin(self, tmp_path):
        repo, env = _make_test_repo(tmp_path)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "check_sister_files_recently_landed.py"),
            "--files-from-stdin",
            "--repo-root", str(repo),
        ]
        result = subprocess.run(
            cmd, cwd=repo, env=env,
            input="src/tac/foo.py\nsrc/tac/bar.py\n",
            capture_output=True, text=True,
        )
        assert result.returncode == 0


# ── Empirical anchor regression test ─────────────────────────────────────
class TestNervFamilyL0BuildAnchorRegression:
    """Replay the empirical NERV-FAMILY-L0-BUILD stand-down bug class.

    Per the task brief: "synthetic sister commit landing
    experiments/train_substrate_ego_nerv.py → helper returns STAND_DOWN_DUPLICATE
    within 6h window → NERV-FAMILY-L0-BUILD would have caught this BEFORE Write."
    """

    def test_nerv_family_l0_build_anchor_would_have_caught_duplication(
        self, tmp_path,
    ):
        """The bug-class regression: with this helper, NERV-FAMILY-L0-BUILD
        would have run the canonical PRE-WRITE check, seen sister commit
        ``18b0beed6``'s overlap, and stood down BEFORE the wasted Writes.
        """
        repo, env = _make_test_repo(tmp_path)

        # Simulate the BUILD-1 NeRV-trio sister commit (the real one was
        # 18b0beed6 @ 2026-05-20T08:05:36-05:00). One commit, three files,
        # ~4.5h before the duplicate dispatch.
        target_files = (
            "experiments/train_substrate_ego_nerv.py",
            "experiments/train_substrate_e_nerv.py",
            "experiments/train_substrate_nervdc.py",
        )
        for rel in target_files:
            full = repo / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(
                "# SPDX-License-Identifier: MIT\n"
                "# L0 SCAFFOLD per BUILD-1 NeRV-trio queue fill 2026-05-20\n"
            )
            subprocess.run(["git", "add", rel], cwd=repo, env=env, check=True)
        sister_sha = _commit_file_no_create(
            repo, env,
            message=(
                "substrates: land 3 L0 SCAFFOLD NeRV trainers "
                "(ego_nerv/e_nerv/nervdc) per BUILD-1 NeRV-trio queue fill "
                "2026-05-20"
            ),
            extra_body=(
                "Adds 3 canonical-pattern substrate trainers mirroring tc_nerv\n"
                "for the missing NeRV-family variants per Catalog #220/#240/#315.\n"
                "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
            ),
            backdate_hours=4.5,
        )

        # The NERV-FAMILY-L0-BUILD subagent would dispatch with the same
        # target files (the canonical "missing NeRV-family variants" list)
        # and a NEW subagent_id (its own).
        verdict = check_sister_files_recently_landed(
            list(target_files),
            repo_root=repo,
            lookback_hours=6,
            own_subagent_id="wave-3-nerv-family-l0-build-20260520",
        )

        # The structural protection fires: STAND_DOWN_DUPLICATE.
        assert verdict.recommendation == "STAND_DOWN_DUPLICATE", (
            f"NERV-FAMILY-L0-BUILD anchor REGRESSION: helper should return "
            f"STAND_DOWN_DUPLICATE but returned {verdict.recommendation!r}. "
            f"This is the empirical bug class the helper extincts."
        )
        # Exactly ONE sister commit (the BUILD-1 NeRV-trio commit).
        assert len(verdict.sister_commits) == 1
        assert verdict.sister_commits[0][0] == sister_sha
        # All 3 target files have overlap entries.
        assert len(verdict.file_to_sister_commits) == 3
        # The rationale should mention STAND DOWN.
        assert "STAND DOWN" in verdict.rationale
        # And explicitly cite Subagent coherence-by-default OR PV discipline.
        assert (
            "Subagent coherence-by-default" in verdict.rationale
            or "Catalog #229" in verdict.rationale
        )

    def test_anchor_canonical_recommendation_taxonomy(self, tmp_path):
        """The 3-recommendation taxonomy is structurally complete for the anchor."""
        repo, env = _make_test_repo(tmp_path)
        # Sister landed nothing → PROCEED.
        v_proceed = check_sister_files_recently_landed(
            ["experiments/train_substrate_ego_nerv.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert v_proceed.recommendation == "PROCEED"

        # Sister landed exact target → STAND_DOWN_DUPLICATE.
        _commit_file(
            repo, env,
            "experiments/train_substrate_ego_nerv.py",
            "# sister\n",
            message="sister: ego_nerv L0 scaffold",
            backdate_hours=2.0,
        )
        v_stand_down = check_sister_files_recently_landed(
            ["experiments/train_substrate_ego_nerv.py"],
            repo_root=repo,
            lookback_hours=6,
        )
        assert v_stand_down.recommendation == "STAND_DOWN_DUPLICATE"
