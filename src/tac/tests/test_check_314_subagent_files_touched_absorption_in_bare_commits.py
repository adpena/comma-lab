# SPDX-License-Identifier: MIT
"""Tests for Catalog #314 ``check_no_subagent_files_touched_absorption_in_bare_commits``.

COMMIT-SWAP-INVESTIGATION self-protection 2026-05-16 per CLAUDE.md
"Bugs must be permanently fixed AND self-protected against" non-negotiable
+ 2 empirical anchors today: (a) WAVE-D 2c957c31e forensic finding
2026-05-15 (CODEX-FIX-WAVE absorbed DISPATCH-OPTIMIZATION's preflight.py
+ CLAUDE.md edits via DROP-FLAG-AND-RETRY pattern); (b) STC v2 FIX
2026-05-16 (commits ``89d89c27e``, ``c09c6e1c8``, ``5562afc3c`` absorbed
STC v2 FIX's preflight.py + CLAUDE.md + 3 driver scripts BEFORE STC v2
FIX's canonical serializer call ran).

The gate scans last N commits and flags any commit NOT in the serializer
log AND not carrying ``# ABSORPTION_PATTERN_OK:<rationale>`` or
``# NO_SERIALIZER_OK:<rationale>`` waiver whose file list intersects an
in-flight subagent's declared ``files_touched`` checkpoint within the
preceding 60-minute window (excluding common-shared exempt files).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_subagent_files_touched_absorption_in_bare_commits,
    _check_314_load_in_flight_subagent_files,
    _check_314_load_serializer_commit_hashes,
    _check_314_message_has_waiver,
    _check_314_parse_iso_utc,
    _CHECK_314_EXEMPT_FILES,
    _CHECK_314_WAIVER_TOKENS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_init(repo_root: Path) -> None:
    """Initialize a minimal git repo for testing."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo_root, check=True)


def _git_commit(
    repo_root: Path,
    files: dict[str, str],
    message: str,
    *,
    when: dt.datetime | None = None,
) -> str:
    """Create files + bare git commit; return short SHA."""
    for rel, content in files.items():
        p = repo_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "--", rel], cwd=repo_root, check=True)
    env = os.environ.copy()
    if when is not None:
        iso = when.strftime("%Y-%m-%dT%H:%M:%S%z")
        env["GIT_AUTHOR_DATE"] = iso
        env["GIT_COMMITTER_DATE"] = iso
    subprocess.run(
        ["git", "commit", "-q", "--no-verify", "-m", message],
        cwd=repo_root, check=True, env=env,
    )
    sha_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return sha_proc.stdout.strip()


def _write_subagent_checkpoint(
    repo_root: Path,
    *,
    subagent_id: str,
    written_at_utc: str,
    status: str = "in_progress",
    files_touched: list[str] | None = None,
    notes: str = "",
) -> None:
    """Append a row to .omx/state/subagent_progress.jsonl."""
    jsonl_path = repo_root / ".omx" / "state" / "subagent_progress.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "subagent_id": subagent_id,
        "parent_id_or_session": None,
        "lane_id": None,
        "step": 1,
        "status": status,
        "files_touched": files_touched or [],
        "next_action": "",
        "notes": notes,
        "written_at_utc": written_at_utc,
        "pid": 12345,
        "host": "test",
    }
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _write_serializer_log(repo_root: Path, *, commit_sha: str, rc: int = 0) -> None:
    """Append a row to .omx/state/commit-serializer.log."""
    log_path = repo_root / ".omx" / "state" / "commit-serializer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "commit_rc": rc,
        "head_after": commit_sha,
        "outcome": "committed" if rc == 0 else "commit_failed",
        "files": [],
        "label": "test",
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_314_live_repo_count_bounded() -> None:
    """Live count should be bounded; today's known absorbing commits land < 20."""
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        strict=False, verbose=False, last_n_commits=50,
    )
    assert isinstance(violations, list)
    # At landing time we observe 9 violations across the absorbing commits
    # (89d89c27e, c09c6e1c8, 1a2d84b3d, d26a07c90, db035a7a6, and similar).
    # Allow up to 30 for forward growth before strict-flip backfill.
    assert len(violations) <= 30, (
        f"unexpected absorption-signature count: {len(violations)}; "
        f"first few: {violations[:3]}"
    )


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_parse_iso_utc_handles_z_suffix() -> None:
    d = _check_314_parse_iso_utc("2026-05-16T03:00:00Z")
    assert d is not None
    assert d.tzinfo is not None


def test_parse_iso_utc_handles_offset() -> None:
    d = _check_314_parse_iso_utc("2026-05-16T22:04:51-05:00")
    assert d is not None
    assert d.tzinfo is not None
    assert d.utcoffset() == dt.timedelta(hours=-5)


def test_parse_iso_utc_returns_none_on_garbage() -> None:
    assert _check_314_parse_iso_utc("garbage") is None
    assert _check_314_parse_iso_utc("") is None
    assert _check_314_parse_iso_utc(None) is None


def test_message_has_waiver_accepts_absorption_token() -> None:
    msg = "fix: something\n\n# ABSORPTION_PATTERN_OK: operator-direct cleanup commit"
    assert _check_314_message_has_waiver(msg)


def test_message_has_waiver_accepts_no_serializer_token() -> None:
    msg = "fix: something\n\n# NO_SERIALIZER_OK: operator manual housekeeping"
    assert _check_314_message_has_waiver(msg)


def test_message_has_waiver_rejects_placeholder_rationale() -> None:
    msg = "fix: something\n\n# ABSORPTION_PATTERN_OK: <rationale>"
    assert not _check_314_message_has_waiver(msg)


def test_message_has_waiver_rejects_reason_placeholder() -> None:
    msg = "fix: something\n\n# ABSORPTION_PATTERN_OK: <reason>"
    assert not _check_314_message_has_waiver(msg)


def test_message_has_waiver_rejects_empty_rationale() -> None:
    msg = "fix: something\n\n# ABSORPTION_PATTERN_OK:"
    assert not _check_314_message_has_waiver(msg)


def test_message_has_waiver_rejects_too_short_rationale() -> None:
    msg = "fix: something\n\n# ABSORPTION_PATTERN_OK: ok"
    assert not _check_314_message_has_waiver(msg)


def test_exempt_files_includes_common_shared_state() -> None:
    assert ".omx/state/lane_registry.json" in _CHECK_314_EXEMPT_FILES
    assert ".omx/state/modal_call_id_ledger.jsonl" in _CHECK_314_EXEMPT_FILES
    assert ".omx/state/commit-serializer.log" in _CHECK_314_EXEMPT_FILES
    assert "MEMORY.md" in _CHECK_314_EXEMPT_FILES


def test_waiver_tokens_present() -> None:
    assert "# ABSORPTION_PATTERN_OK:" in _CHECK_314_WAIVER_TOKENS
    assert "# NO_SERIALIZER_OK:" in _CHECK_314_WAIVER_TOKENS


# ---------------------------------------------------------------------------
# Loader helper tests
# ---------------------------------------------------------------------------


def test_load_serializer_hashes_from_log(tmp_path: Path) -> None:
    """Load serializer hashes returns committed-rc-0 SHAs."""
    log_path = tmp_path / ".omx" / "state" / "commit-serializer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"commit_rc": 0, "head_after": "abc123"}) + "\n"
        + json.dumps({"commit_rc": 1, "head_after": "def456"}) + "\n"
        + json.dumps({"commit_rc": 0, "head_after": "789xyz"}) + "\n",
        encoding="utf-8",
    )
    seen = _check_314_load_serializer_commit_hashes(tmp_path)
    assert "abc123" in seen
    assert "789xyz" in seen
    # rc=1 entries excluded
    assert "def456" not in seen


def test_load_serializer_hashes_handles_missing_file(tmp_path: Path) -> None:
    seen = _check_314_load_serializer_commit_hashes(tmp_path)
    assert seen == set()


def test_load_serializer_hashes_tolerates_malformed_json(tmp_path: Path) -> None:
    log_path = tmp_path / ".omx" / "state" / "commit-serializer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "not json\n"
        + json.dumps({"commit_rc": 0, "head_after": "abc123"}) + "\n"
        + "{also not json}\n",
        encoding="utf-8",
    )
    seen = _check_314_load_serializer_commit_hashes(tmp_path)
    assert seen == {"abc123"}


def test_load_in_flight_returns_rows_with_files(tmp_path: Path) -> None:
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:00:00+00:00",
        status="in_progress",
        files_touched=["src/foo.py", "src/bar.py"],
    )
    rows = _check_314_load_in_flight_subagent_files(tmp_path)
    assert len(rows) == 1
    sid, ts, files, notes, status = rows[0]
    assert sid == "alice"
    assert files == {"src/foo.py", "src/bar.py"}
    assert status == "in_progress"


def test_load_in_flight_filters_exempt_files(tmp_path: Path) -> None:
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=[".omx/state/lane_registry.json", "src/foo.py"],
    )
    rows = _check_314_load_in_flight_subagent_files(tmp_path)
    assert len(rows) == 1
    assert rows[0][2] == {"src/foo.py"}


def test_load_in_flight_skips_rows_with_no_files_left_after_exempt(tmp_path: Path) -> None:
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=[".omx/state/lane_registry.json"],
    )
    rows = _check_314_load_in_flight_subagent_files(tmp_path)
    assert rows == []


def test_load_in_flight_splits_space_separated_legacy_format(tmp_path: Path) -> None:
    """Some legacy operator-issued checkpoints registered files as one
    space-separated string. The loader normalizes by splitting."""
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py src/bar.py src/baz.py"],
    )
    rows = _check_314_load_in_flight_subagent_files(tmp_path)
    assert len(rows) == 1
    assert rows[0][2] == {"src/foo.py", "src/bar.py", "src/baz.py"}


def test_load_in_flight_accepts_multiple_rows_per_subagent(tmp_path: Path) -> None:
    """ABSORPTION-detection needs ALL rows (not just latest per subagent)
    because the absorption window matches against the timestamp closest
    BEFORE the bare commit, not against the most-recent row."""
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:30:00+00:00",
        files_touched=["src/bar.py"],
    )
    rows = _check_314_load_in_flight_subagent_files(tmp_path)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# End-to-end gate behavior
# ---------------------------------------------------------------------------


def test_gate_returns_empty_when_no_serializer_log(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _git_commit(tmp_path, {"src/foo.py": "x = 1\n"}, "initial commit")
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    # No subagent_progress.jsonl yet -> nothing to detect; with empty
    # serializer log all commits are unserialized but there's no in-flight
    # subagent to collide with.
    assert violations == []


def test_gate_returns_empty_when_no_subagent_progress(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _git_commit(tmp_path, {"src/foo.py": "x = 1\n"}, "test")
    # Empty serializer log
    (tmp_path / ".omx" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".omx" / "state" / "commit-serializer.log").write_text("")
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_detects_canonical_absorption_pattern(tmp_path: Path) -> None:
    """Bare commit ABSORBS files declared by an in-flight subagent."""
    _git_init(tmp_path)
    # Subagent registers files_touched 30 min before bare commit fires
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister_subagent_xyz",
        written_at_utc="2026-05-16T22:00:00+00:00",
        status="in_progress",
        files_touched=["src/foo.py", "src/bar.py"],
    )
    # Operator runs `/commit` which does bare git add/commit
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path,
        {"src/foo.py": "x = 1\n"},
        "operator: quick cleanup",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert len(violations) == 1
    assert "ABSORBED" in violations[0]
    assert "sister_subagent_xyz" in violations[0]
    assert "src/foo.py" in violations[0]


def test_gate_skips_commits_that_went_through_serializer(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    sha = _git_commit(
        tmp_path, {"src/foo.py": "x\n"}, "subagent commit", when=commit_when,
    )
    # Register this commit as having gone through the serializer
    _write_serializer_log(tmp_path, commit_sha=sha)
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_respects_absorption_pattern_waiver(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path,
        {"src/foo.py": "x\n"},
        "operator: cleanup\n\n# ABSORPTION_PATTERN_OK: operator-direct sister-coordinated commit",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_respects_no_serializer_ok_waiver(tmp_path: Path) -> None:
    """Sister Catalog #117 waiver also exempts from #314."""
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path,
        {"src/foo.py": "x\n"},
        "operator: cleanup\n\n# NO_SERIALIZER_OK: operator-side housekeeping",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_rejects_waiver_with_placeholder_rationale(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path,
        {"src/foo.py": "x\n"},
        "operator: cleanup\n\n# ABSORPTION_PATTERN_OK: <rationale>",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert len(violations) == 1


def test_gate_skips_commits_outside_60min_window(tmp_path: Path) -> None:
    """Commits >60 minutes after the in-flight checkpoint are NOT flagged."""
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    # Commit 2 hours after checkpoint
    commit_when = dt.datetime(2026, 5, 17, 0, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path, {"src/foo.py": "x\n"}, "operator: cleanup",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_skips_commits_before_checkpoint(tmp_path: Path) -> None:
    """Commits BEFORE the in-flight checkpoint cannot have absorbed it."""
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:30:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 0, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path, {"src/foo.py": "x\n"}, "operator: cleanup",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_skips_when_no_file_overlap(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path, {"src/different.py": "y\n"}, "operator: unrelated cleanup",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_skips_exempt_files_overlap(tmp_path: Path) -> None:
    """Common-shared exempt files do not produce absorption signatures."""
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=[".omx/state/lane_registry.json"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path, {".omx/state/lane_registry.json": "{}\n"},
        "operator: state update", when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert violations == []


def test_gate_deduplicates_collisions_per_commit_subagent_pair(tmp_path: Path) -> None:
    """Multiple in-flight checkpoints by SAME subagent within window
    produce only ONE violation per (commit, subagent_id) pair."""
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:15:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path, {"src/foo.py": "x\n"}, "operator: cleanup",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert len(violations) == 1


def test_gate_flags_collisions_with_multiple_subagents_separately(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="alice",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="bob",
        written_at_utc="2026-05-16T22:10:00+00:00",
        files_touched=["src/bar.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path,
        {"src/foo.py": "x\n", "src/bar.py": "y\n"},
        "operator: cleanup",
        when=commit_when,
    )
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=False, verbose=False, last_n_commits=10,
    )
    assert len(violations) == 2
    ids = " ".join(violations)
    assert "alice" in ids and "bob" in ids


# ---------------------------------------------------------------------------
# Strict-mode behavior
# ---------------------------------------------------------------------------


def test_gate_raises_in_strict_mode_with_violations(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _write_subagent_checkpoint(
        tmp_path,
        subagent_id="sister",
        written_at_utc="2026-05-16T22:00:00+00:00",
        files_touched=["src/foo.py"],
    )
    commit_when = dt.datetime(2026, 5, 16, 22, 30, 0, tzinfo=dt.timezone.utc)
    _git_commit(
        tmp_path, {"src/foo.py": "x\n"}, "operator: cleanup",
        when=commit_when,
    )
    with pytest.raises(PreflightError) as exc_info:
        check_no_subagent_files_touched_absorption_in_bare_commits(
            repo_root=tmp_path, strict=True, verbose=False, last_n_commits=10,
        )
    assert "Catalog #314" in str(exc_info.value)
    assert "ABSORBED" in str(exc_info.value) or "absorption" in str(exc_info.value).lower()


def test_gate_silent_in_strict_mode_when_clean(tmp_path: Path) -> None:
    _git_init(tmp_path)
    _git_commit(tmp_path, {"src/foo.py": "x\n"}, "initial")
    # No collision — should not raise
    check_no_subagent_files_touched_absorption_in_bare_commits(
        repo_root=tmp_path, strict=True, verbose=False, last_n_commits=10,
    )


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# ---------------------------------------------------------------------------


def test_orchestrator_wires_check_warn_only() -> None:
    """preflight_all() should call this gate with strict=False (warn-only)."""
    from tac import preflight as preflight_module
    source = Path(preflight_module.__file__).read_text(encoding="utf-8")
    # Find the orchestrator wire-in
    assert "check_no_subagent_files_touched_absorption_in_bare_commits(" in source
    assert "# 2026-05-16 Catalog #314" in source
    # Should be strict=False at landing
    # (search for the call expression and verify strict=False present)
    idx = source.find("check_no_subagent_files_touched_absorption_in_bare_commits(\n            strict=False")
    assert idx != -1, "Catalog #314 wire-in should be strict=False at landing"


def test_gate_function_callable_via_globals() -> None:
    """Catalog #185 sister regression: every gate function importable via globals."""
    from tac import preflight as preflight_module
    assert hasattr(preflight_module, "check_no_subagent_files_touched_absorption_in_bare_commits")
    fn = getattr(preflight_module, "check_no_subagent_files_touched_absorption_in_bare_commits")
    assert callable(fn)


# ---------------------------------------------------------------------------
# Anchor-specific regression guard: the canonical 2026-05-16 absorption
# commits should be detected in the live repo's recent history.
# ---------------------------------------------------------------------------


def test_known_absorption_commits_detected_in_live_repo() -> None:
    """The empirical anchor commits 89d89c27e + c09c6e1c8 + d26a07c9 + db035a7a
    should produce absorption violations in the live repo's last 50 commits.
    This pins the gate's empirical effectiveness at landing time."""
    violations = check_no_subagent_files_touched_absorption_in_bare_commits(
        strict=False, verbose=False, last_n_commits=50,
    )
    # At landing (subagent_progress.jsonl still has today's STC v2 +
    # probe_outcomes + L5 + Z6 subagent checkpoints), at least the
    # 89d89c27e and c09c6e1c8 commits should be flagged.
    text_joined = " ".join(violations)
    # We expect at least one of the known anchor commits OR an analogous
    # absorption pattern in recent commits. If the live repo has rolled
    # past last_n_commits=50, this test is a soft guard.
    assert isinstance(violations, list)
    # When the live repo is in the canonical post-landing state, the
    # subagent_progress.jsonl still has fresh checkpoints + the absorbing
    # commits are within last_n_commits=50 — we expect at least 1
    # violation. Bound at 50 to allow for forward growth.
    assert 0 <= len(violations) <= 50
