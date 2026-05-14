# SPDX-License-Identifier: MIT
"""Tests for Catalog #230 — bulk-rewrite sister-subagent ownership-map gate.

Per `feedback_editor_vs_editor_collision_patterns_in_parallel_waves_20260514.md`.

Refuses subagent commits whose body mentions a bulk-op token (SPDX header
sweep / mass refactor / bulk rewrite / etc.) WITHOUT referencing the
sister-subagent ownership map / disjoint-scope marker.

Same-line waiver: ``# BULK_REWRITE_OWNERSHIP_WAIVED:<reason>`` on a body
line.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_230_BULK_OP_TOKENS,
    _CHECK_230_OWNERSHIP_MAP_TOKENS,
    _CHECK_230_WAIVER_RE,
    _check_230_iter_recent_commits,
    _check_230_load_serializer_shas,
    check_bulk_rewrite_respects_sister_subagent_ownership_map,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake_git_repo(tmp_path):
    """Build a fake git repo with a commit-serializer log."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(repo), check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(repo), check=True,
    )
    # Initial empty commit
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "initial"],
        cwd=str(repo), check=True,
    )
    # Set up commit-serializer log
    (repo / ".omx" / "state").mkdir(parents=True)
    return repo


def _make_subagent_commit(repo, message):
    """Create a commit in the fake repo and append its SHA to the
    serializer log."""
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", message],
        cwd=str(repo), check=True,
    )
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo), capture_output=True, text=True, check=True,
    )
    sha = out.stdout.strip()
    # Append to serializer log
    log = repo / ".omx" / "state" / "commit-serializer.log"
    with log.open("a") as f:
        f.write(json.dumps({"commit_sha": sha[:9], "sha": sha}) + "\n")
    return sha


# ─── Live repo regression guard ──────────────────────────────────────


def test_live_repo_zero_violations():
    """Live repo should be clean per Strict-flip atomicity rule."""
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert v == []


# ─── Positive: bulk-op commit without ownership map flagged ──────────


def test_bulk_op_without_ownership_flagged(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Big SPDX header sweep across all .py files",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "SPDX header sweep" in v[0].lower() or "spdx" in v[0].lower()
    assert "Catalog #230" in v[0]


def test_mass_refactor_without_ownership_flagged(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Mass refactor of all build_*.py tools to canonical pattern",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert len(v) == 1


def test_bulk_rewrite_without_ownership_flagged(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Bulk-rewrite of all SPDX headers under src/",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert len(v) == 1


# ─── Acceptance: ownership-map reference ─────────────────────────────


def test_ownership_map_reference_satisfies(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "SPDX header sweep\n\n"
        "Recursive R2 ownership map honored. Sister-subagent territory "
        "excluded.",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert v == []


def test_owns_marker_satisfies(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Mass refactor\n\n"
        "OWNS: src/tac/myslice/ (disjoint scope)",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert v == []


def test_path_prefix_exclusion_satisfies(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Bulk rewrite\n\n"
        "Path-prefix exclusion: experiments/results/, _intake_.",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert v == []


# ─── Waiver acceptance ────────────────────────────────────────────────


def test_same_line_waiver_accepted(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Bulk import refactor\n\n"
        "# BULK_REWRITE_OWNERSHIP_WAIVED:operator-direct-no-sister-active",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert v == []


def test_placeholder_waiver_rejected(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Bulk rewrite\n\n"
        "# BULK_REWRITE_OWNERSHIP_WAIVED:<reason>",
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert len(v) == 1


# ─── Out-of-scope: non-subagent commits exempt ───────────────────────


def test_non_subagent_commit_exempt(fake_git_repo):
    """Operator-direct commits (not in serializer log) are exempt."""
    # Create a commit but DO NOT add to serializer log
    subprocess.run(
        ["git", "commit", "--allow-empty",
         "-m", "Bulk rewrite by operator without sister-subagent map"],
        cwd=str(fake_git_repo), check=True,
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert v == []


# ─── Strict mode ──────────────────────────────────────────────────────


def test_strict_raises_on_violation(fake_git_repo):
    _make_subagent_commit(
        fake_git_repo,
        "Wide-edit across all substrate trainers",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_bulk_rewrite_respects_sister_subagent_ownership_map(
            repo_root=fake_git_repo, strict=True, verbose=False
        )
    assert "Catalog #230" in str(exc_info.value)


def test_strict_silent_on_clean(fake_git_repo):
    """No bulk-op commits = clean."""
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=True, verbose=False
    )
    assert v == []


# ─── Edge cases ───────────────────────────────────────────────────────


def test_no_serializer_log_no_op(tmp_path):
    """If serializer log absent, gate is no-op."""
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@x.com"],
        cwd=str(tmp_path), check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=str(tmp_path), check=True
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "Bulk rewrite"],
        cwd=str(tmp_path), check=True,
    )
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_serializer_loader_handles_malformed_log(tmp_path):
    """Malformed JSON lines in serializer log are skipped gracefully."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".omx" / "state").mkdir(parents=True)
    (repo / ".omx" / "state" / "commit-serializer.log").write_text(
        "not json\n"
        '{"commit_sha": "abc123def"}\n'
        "broken {\n"
    )
    shas = _check_230_load_serializer_shas(repo, 50)
    assert "abc123def" in shas


def test_iter_recent_commits_handles_no_git(tmp_path):
    """Non-git directory returns empty list."""
    out = _check_230_iter_recent_commits(tmp_path, 50)
    assert out == []


def test_bulk_op_tokens_constant():
    """The bulk-op tokens include the canonical empirical anchors."""
    assert "spdx header sweep" in _CHECK_230_BULK_OP_TOKENS
    assert "mass refactor" in _CHECK_230_BULK_OP_TOKENS
    assert "bulk rewrite" in _CHECK_230_BULK_OP_TOKENS


def test_ownership_map_tokens_constant():
    """The ownership-map tokens include canonical references."""
    assert "recursive R2" in _CHECK_230_OWNERSHIP_MAP_TOKENS
    assert "OWNS:" in _CHECK_230_OWNERSHIP_MAP_TOKENS
    assert "sister-subagent" in _CHECK_230_OWNERSHIP_MAP_TOKENS


def test_waiver_regex_well_formed():
    assert _CHECK_230_WAIVER_RE.search(
        "# BULK_REWRITE_OWNERSHIP_WAIVED:real-reason"
    )
    assert not _CHECK_230_WAIVER_RE.search(
        "# BULK_REWRITE_OWNERSHIP_WAIVED:<reason>"
    )


def test_verbose_output(fake_git_repo, capsys):
    _make_subagent_commit(fake_git_repo, "normal commit")
    check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "bulk-rewrite-ownership" in captured.out


def test_string_repo_root_accepted(fake_git_repo):
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=str(fake_git_repo), strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_preflight_all_wiring_warn_only():
    """Wired into preflight_all() with strict=False."""
    from tac import preflight as pf
    source = Path(pf.__file__).read_text(encoding="utf-8")
    callsite_idx = source.find(
        "lambda: check_bulk_rewrite_respects_sister_subagent_ownership_map("
    )
    assert callsite_idx > 0
    window = source[callsite_idx : callsite_idx + 200]
    assert "strict=False" in window


def test_multiple_violations_aggregated(fake_git_repo):
    """Multiple bulk-op commits without ownership are all flagged."""
    _make_subagent_commit(fake_git_repo, "SPDX header sweep")
    _make_subagent_commit(fake_git_repo, "Mass refactor of code")
    _make_subagent_commit(fake_git_repo, "Bulk-edit refactor")
    v = check_bulk_rewrite_respects_sister_subagent_ownership_map(
        repo_root=fake_git_repo, strict=False, verbose=False
    )
    assert len(v) == 3
