# SPDX-License-Identifier: MIT
"""Tests for ``tac.discipline_anti_pattern_guards`` canonical helpers.

Covers:
  * :class:`SpawnGuardVerdict` + :class:`SpawnPvEvidenceContext` invariants
  * :func:`verify_head_state_before_spawn` 3-surface decision matrix
  * :class:`HandoffGuardVerdict` + :class:`PredecessorHandoffContext` invariants
  * :func:`verify_predecessor_working_tree_committed_or_auto_commit` 4-verdict
    decision matrix (PROCEED / UNCOMMITTED-no-auto / AUTO_COMMIT_LANDED /
    AUTO_COMMIT_FAILED)
  * Live-repo regression guards
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pytest

from tac.discipline_anti_pattern_guards import (
    DEFAULT_SISTER_LANDING_LOOKBACK_MINUTES,
    DEFAULT_SISTER_LANDING_MEMO_GLOB,
    HEAD_STATE_PV_TOKENS,
    HandoffGuardRecommendation,
    HandoffGuardVerdict,
    PredecessorHandoffContext,
    SpawnGuardRecommendation,
    SpawnGuardVerdict,
    SpawnPvEvidenceContext,
    verify_head_state_before_spawn,
    verify_predecessor_working_tree_committed_or_auto_commit,
)
from tac.discipline_anti_pattern_guards.predecessor_handoff_auto_commit_guard import (
    EXEMPT_PATHS as HANDOFF_EXEMPT_PATHS,
)


# ─────────────────────────────────────────────────────────────────────────
# Section A: SpawnPvEvidenceContext + SpawnGuardVerdict invariants
# ─────────────────────────────────────────────────────────────────────────
class TestSpawnPvEvidenceContext:
    def test_default_construction(self) -> None:
        ctx = SpawnPvEvidenceContext()
        assert ctx.observed_head_sha is None
        assert ctx.observed_dirty_paths == ()
        assert ctx.observed_recent_commits == ()
        assert ctx.consulted_landing_memo_paths == ()

    def test_full_construction(self) -> None:
        ctx = SpawnPvEvidenceContext(
            observed_head_sha="abc123def456",
            observed_dirty_paths=("a.py",),
            observed_recent_commits=("commit1: foo",),
            consulted_landing_memo_paths=(".omx/research/foo_landed_20260528.md",),
        )
        assert ctx.observed_head_sha == "abc123def456"
        assert ctx.observed_dirty_paths == ("a.py",)

    def test_rejects_non_tuple_paths(self) -> None:
        with pytest.raises(TypeError):
            SpawnPvEvidenceContext(observed_dirty_paths=["a.py"])  # type: ignore[arg-type]

    def test_rejects_non_str_in_tuple(self) -> None:
        with pytest.raises(TypeError):
            SpawnPvEvidenceContext(observed_dirty_paths=(123,))  # type: ignore[arg-type]

    def test_rejects_non_str_head_sha(self) -> None:
        with pytest.raises(TypeError):
            SpawnPvEvidenceContext(observed_head_sha=123)  # type: ignore[arg-type]


class TestSpawnGuardVerdict:
    def test_proceed_has_no_conflict(self) -> None:
        verdict = SpawnGuardVerdict(
            recommendation="PROCEED",
            overlapping_scope=(),
            diagnostic="PROCEED",
            conflict_source="none",
            consulted_evidence=SpawnPvEvidenceContext(),
            sister_subagent_ids=(),
            recent_landing_memo_matches=(),
        )
        assert verdict.has_conflict() is False

    def test_sister_in_flight_has_conflict(self) -> None:
        verdict = SpawnGuardVerdict(
            recommendation="SISTER_IN_FLIGHT",
            overlapping_scope=("src/tac/foo/",),
            diagnostic="conflict",
            conflict_source="sister_checkpoint",
            consulted_evidence=SpawnPvEvidenceContext(),
            sister_subagent_ids=("sister1",),
            recent_landing_memo_matches=(),
        )
        assert verdict.has_conflict() is True


# ─────────────────────────────────────────────────────────────────────────
# Section B: verify_head_state_before_spawn — input validation
# ─────────────────────────────────────────────────────────────────────────
class TestVerifyHeadStateInputValidation:
    def test_rejects_string_as_scope(self) -> None:
        with pytest.raises(TypeError, match="declared_scope must be a list"):
            verify_head_state_before_spawn("src/tac/foo.py")  # type: ignore[arg-type]

    def test_rejects_list_with_non_str(self) -> None:
        with pytest.raises(TypeError, match="declared_scope must be a list"):
            verify_head_state_before_spawn(["src/tac/foo.py", 123])  # type: ignore[list-item]

    def test_empty_scope_returns_proceed(self, tmp_path: Path) -> None:
        verdict = verify_head_state_before_spawn(
            [],
            repo_root=tmp_path,
            research_dir=tmp_path / "research",
            checkpoint_path=tmp_path / "checkpoint.jsonl",
        )
        assert verdict.recommendation == "PROCEED"
        assert "empty" in verdict.diagnostic.lower()

    def test_scope_with_empty_strings_treated_as_empty(self, tmp_path: Path) -> None:
        verdict = verify_head_state_before_spawn(
            ["", "  "],
            repo_root=tmp_path,
            research_dir=tmp_path / "research",
            checkpoint_path=tmp_path / "checkpoint.jsonl",
        )
        assert verdict.recommendation == "PROCEED"


# ─────────────────────────────────────────────────────────────────────────
# Section C: verify_head_state_before_spawn — falling-rule decision matrix
# ─────────────────────────────────────────────────────────────────────────
def _init_git_repo(tmp_path: Path) -> None:
    """Initialize a minimal git repo for testing."""
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True
    )
    (tmp_path / "README.md").write_text("# test\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"], cwd=str(tmp_path), check=True
    )


def _commit_files(tmp_path: Path, files: dict[str, str], message: str) -> str:
    """Create + commit files; return commit sha."""
    for rel, content in files.items():
        full = tmp_path / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        subprocess.run(["git", "add", rel], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", message], cwd=str(tmp_path), check=True
    )
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


class TestVerifyHeadStateFallingRuleDecision:
    def test_proceed_when_no_overlap_anywhere(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        verdict = verify_head_state_before_spawn(
            ["src/tac/never_touched/"],
            repo_root=tmp_path,
            research_dir=tmp_path / ".omx" / "research",
            checkpoint_path=tmp_path / "checkpoint.jsonl",
        )
        assert verdict.recommendation == "PROCEED"
        assert verdict.conflict_source == "none"
        assert verdict.overlapping_scope == ()

    def test_duplicate_head_state_from_recent_commit(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _commit_files(
            tmp_path,
            {"src/tac/foo/bar.py": "x = 1\n"},
            "test: add foo/bar.py",
        )
        verdict = verify_head_state_before_spawn(
            ["src/tac/foo/"],
            repo_root=tmp_path,
            research_dir=tmp_path / ".omx" / "research",
            checkpoint_path=tmp_path / "checkpoint.jsonl",
        )
        assert verdict.recommendation == "DUPLICATE_HEAD_STATE"
        assert verdict.conflict_source == "head_recent_commit"
        assert "src/tac/foo/" in verdict.overlapping_scope

    def test_duplicate_head_state_from_landing_memo(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        research_dir = tmp_path / ".omx" / "research"
        research_dir.mkdir(parents=True)
        memo = research_dir / "foo_landed_20260528.md"
        memo.write_text(
            "Bug class anchor: src/tac/substrates/foo/ trainer refactor landed.\n"
        )
        verdict = verify_head_state_before_spawn(
            ["src/tac/substrates/foo/"],
            repo_root=tmp_path,
            research_dir=research_dir,
            checkpoint_path=tmp_path / "checkpoint.jsonl",
        )
        assert verdict.recommendation == "DUPLICATE_HEAD_STATE"
        assert verdict.conflict_source == "sister_landing_memo"
        assert len(verdict.recent_landing_memo_matches) == 1

    def test_sister_in_flight_overrides_landing_memo(self, tmp_path: Path) -> None:
        """Sister-checkpoint conflict has priority over landing-memo overlap."""
        _init_git_repo(tmp_path)
        research_dir = tmp_path / ".omx" / "research"
        research_dir.mkdir(parents=True)
        memo = research_dir / "foo_landed_20260528.md"
        memo.write_text("Mentions src/tac/foo/")

        # Add a sister checkpoint that overlaps.
        ckpt = tmp_path / "checkpoint.jsonl"
        now = _dt.datetime.now(_dt.timezone.utc)
        row = {
            "subagent_id": "sister_subagent_a",
            "status": "in_progress",
            "written_at_utc": now.isoformat().replace("+00:00", "Z"),
            "files_touched": ["src/tac/foo/bar.py"],
            "step": 1,
            "next_action": "edit",
        }
        ckpt.write_text(json.dumps(row) + "\n")

        verdict = verify_head_state_before_spawn(
            ["src/tac/foo/"],
            repo_root=tmp_path,
            research_dir=research_dir,
            checkpoint_path=ckpt,
            now_utc=now,
        )
        assert verdict.recommendation == "SISTER_IN_FLIGHT"
        assert verdict.conflict_source == "sister_checkpoint"
        assert "sister_subagent_a" in verdict.sister_subagent_ids

    def test_sister_overrides_head_commit(self, tmp_path: Path) -> None:
        """Sister-checkpoint priority over HEAD-recent-commit overlap."""
        _init_git_repo(tmp_path)
        _commit_files(tmp_path, {"src/tac/foo/bar.py": "x = 1\n"}, "test: add")

        ckpt = tmp_path / "checkpoint.jsonl"
        now = _dt.datetime.now(_dt.timezone.utc)
        row = {
            "subagent_id": "sister_b",
            "status": "in_progress",
            "written_at_utc": now.isoformat().replace("+00:00", "Z"),
            "files_touched": ["src/tac/foo/baz.py"],
            "step": 1,
            "next_action": "edit",
        }
        ckpt.write_text(json.dumps(row) + "\n")

        verdict = verify_head_state_before_spawn(
            ["src/tac/foo/"],
            repo_root=tmp_path,
            research_dir=tmp_path / ".omx" / "research",
            checkpoint_path=ckpt,
            now_utc=now,
        )
        assert verdict.recommendation == "SISTER_IN_FLIGHT"

    def test_completed_sister_does_not_flag(self, tmp_path: Path) -> None:
        """Sister with status=complete should NOT flag."""
        _init_git_repo(tmp_path)
        ckpt = tmp_path / "checkpoint.jsonl"
        now = _dt.datetime.now(_dt.timezone.utc)
        row = {
            "subagent_id": "completed_sister",
            "status": "complete",
            "written_at_utc": now.isoformat().replace("+00:00", "Z"),
            "files_touched": ["src/tac/foo/bar.py"],
            "step": 5,
            "next_action": "",
        }
        ckpt.write_text(json.dumps(row) + "\n")

        verdict = verify_head_state_before_spawn(
            ["src/tac/foo/"],
            repo_root=tmp_path,
            research_dir=tmp_path / ".omx" / "research",
            checkpoint_path=ckpt,
            now_utc=now,
        )
        assert verdict.recommendation == "PROCEED"

    def test_sister_outside_lookback_window_does_not_flag(self, tmp_path: Path) -> None:
        """Sister checkpoint older than lookback window should NOT flag."""
        _init_git_repo(tmp_path)
        ckpt = tmp_path / "checkpoint.jsonl"
        now = _dt.datetime.now(_dt.timezone.utc)
        old_ts = now - _dt.timedelta(hours=3)
        row = {
            "subagent_id": "old_sister",
            "status": "in_progress",
            "written_at_utc": old_ts.isoformat().replace("+00:00", "Z"),
            "files_touched": ["src/tac/foo/bar.py"],
            "step": 1,
            "next_action": "edit",
        }
        ckpt.write_text(json.dumps(row) + "\n")

        verdict = verify_head_state_before_spawn(
            ["src/tac/foo/"],
            repo_root=tmp_path,
            research_dir=tmp_path / ".omx" / "research",
            checkpoint_path=ckpt,
            now_utc=now,
            lookback_minutes=60,  # 3h ago is outside 60-min lookback
        )
        assert verdict.recommendation == "PROCEED"

    def test_corrupt_checkpoint_treated_as_no_sister(self, tmp_path: Path) -> None:
        """Lenient mode: malformed JSON lines skipped silently."""
        _init_git_repo(tmp_path)
        ckpt = tmp_path / "checkpoint.jsonl"
        ckpt.write_text("not_json\n{also not json\n")

        verdict = verify_head_state_before_spawn(
            ["src/tac/never_touched/"],
            repo_root=tmp_path,
            research_dir=tmp_path / ".omx" / "research",
            checkpoint_path=ckpt,
        )
        assert verdict.recommendation == "PROCEED"


# ─────────────────────────────────────────────────────────────────────────
# Section D: PredecessorHandoffContext + HandoffGuardVerdict invariants
# ─────────────────────────────────────────────────────────────────────────
class TestPredecessorHandoffContext:
    def test_default_construction(self) -> None:
        ctx = PredecessorHandoffContext()
        assert ctx.predecessor_subagent_id is None
        assert ctx.successor_subagent_id is None
        assert ctx.auto_commit_message_subject is None

    def test_full_construction(self) -> None:
        ctx = PredecessorHandoffContext(
            predecessor_subagent_id="pred_a",
            successor_subagent_id="succ_b",
            auto_commit_message_subject="handoff: my custom subject",
        )
        assert ctx.predecessor_subagent_id == "pred_a"

    def test_rejects_non_str_id(self) -> None:
        with pytest.raises(TypeError):
            PredecessorHandoffContext(predecessor_subagent_id=123)  # type: ignore[arg-type]


class TestHandoffGuardVerdict:
    def test_proceed_no_residue(self) -> None:
        verdict = HandoffGuardVerdict(
            recommendation="PROCEED",
            uncommitted_paths=(),
            diagnostic="clean",
            auto_commit_attempted=False,
            new_head_sha=None,
            canonical_serializer_invoked=False,
        )
        assert verdict.has_handoff_residue() is False

    def test_auto_commit_landed_has_residue(self) -> None:
        verdict = HandoffGuardVerdict(
            recommendation="AUTO_COMMIT_LANDED",
            uncommitted_paths=("src/tac/foo.py",),
            diagnostic="committed",
            auto_commit_attempted=True,
            new_head_sha="abc123",
            canonical_serializer_invoked=True,
        )
        assert verdict.has_handoff_residue() is True


# ─────────────────────────────────────────────────────────────────────────
# Section E: verify_predecessor_working_tree decision matrix
# ─────────────────────────────────────────────────────────────────────────
class TestVerifyPredecessorHandoffDecision:
    def test_proceed_on_clean_tree(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        verdict = verify_predecessor_working_tree_committed_or_auto_commit(
            repo_root=tmp_path,
        )
        assert verdict.recommendation == "PROCEED"
        assert verdict.uncommitted_paths == ()
        assert verdict.auto_commit_attempted is False

    def test_proceed_on_exempt_only(self, tmp_path: Path) -> None:
        """Modifications limited to canonical exempt state files should PROCEED."""
        _init_git_repo(tmp_path)
        # Modify only an exempt file (well, simulate via a path mentioned in EXEMPT_PATHS).
        # Pick one that doesn't require parent setup.
        exempt_dir = tmp_path / ".omx" / "state"
        exempt_dir.mkdir(parents=True)
        (exempt_dir / "lane_registry.json").write_text('{"lanes": []}\n')
        # Now stage + commit so it exists at HEAD; THEN modify.
        subprocess.run(
            ["git", "add", ".omx/state/lane_registry.json"], cwd=str(tmp_path), check=True
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "add registry"], cwd=str(tmp_path), check=True
        )
        (exempt_dir / "lane_registry.json").write_text('{"lanes": [], "updated": true}\n')

        verdict = verify_predecessor_working_tree_committed_or_auto_commit(
            repo_root=tmp_path,
        )
        assert verdict.recommendation == "PROCEED"
        assert "exempt-state file" in verdict.diagnostic

    def test_uncommitted_detected_no_auto_commit(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        # Commit a file first, then modify it (so porcelain reports it as M not ??).
        _commit_files(tmp_path, {"src/foo.py": "x = 1\n"}, "test: add src/foo.py")
        (tmp_path / "src" / "foo.py").write_text("x = 2\n")

        verdict = verify_predecessor_working_tree_committed_or_auto_commit(
            repo_root=tmp_path,
            auto_commit=False,
        )
        assert verdict.recommendation == "PREDECESSOR_UNCOMMITTED_WORK_DETECTED"
        assert "src/foo.py" in verdict.uncommitted_paths
        assert verdict.auto_commit_attempted is False
        assert verdict.canonical_serializer_invoked is False

    def test_auto_commit_failed_when_serializer_missing(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _commit_files(tmp_path, {"src/foo.py": "x = 1\n"}, "test: add src/foo.py")
        (tmp_path / "src" / "foo.py").write_text("x = 2\n")

        verdict = verify_predecessor_working_tree_committed_or_auto_commit(
            repo_root=tmp_path,
            auto_commit=True,
            serializer_path=tmp_path / "nonexistent_serializer.py",
        )
        assert verdict.recommendation == "AUTO_COMMIT_FAILED"
        assert verdict.canonical_serializer_invoked is True
        assert "not found" in verdict.diagnostic

    def test_handoff_context_propagates(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _commit_files(tmp_path, {"src/foo.py": "x = 1\n"}, "test: add src/foo.py")
        (tmp_path / "src" / "foo.py").write_text("x = 2\n")
        ctx = PredecessorHandoffContext(
            predecessor_subagent_id="pred_a",
            successor_subagent_id="succ_b",
        )
        verdict = verify_predecessor_working_tree_committed_or_auto_commit(
            repo_root=tmp_path,
            auto_commit=False,
            handoff_context=ctx,
        )
        assert verdict.recommendation == "PREDECESSOR_UNCOMMITTED_WORK_DETECTED"


# ─────────────────────────────────────────────────────────────────────────
# Section F: Live-repo regression guards
# ─────────────────────────────────────────────────────────────────────────
class TestLiveRepoRegression:
    def test_spawn_guard_callable_on_live_repo(self) -> None:
        """Smoke test: spawn guard runs against live repo without crashing."""
        verdict = verify_head_state_before_spawn(["src/tac/never_touched_by_anyone_xyz/"])
        # The verdict may be PROCEED or any other (depends on session state);
        # the contract is that it returns a typed verdict without exception.
        assert isinstance(verdict, SpawnGuardVerdict)
        assert verdict.recommendation in (
            "PROCEED",
            "DUPLICATE_HEAD_STATE",
            "SISTER_IN_FLIGHT",
        )

    def test_handoff_guard_callable_on_live_repo(self) -> None:
        """Smoke test: handoff guard runs against live repo without crashing.

        Use ``auto_commit=False`` to avoid mutating live state.
        """
        verdict = verify_predecessor_working_tree_committed_or_auto_commit(
            auto_commit=False,
        )
        assert isinstance(verdict, HandoffGuardVerdict)
        assert verdict.recommendation in (
            "PROCEED",
            "PREDECESSOR_UNCOMMITTED_WORK_DETECTED",
            "AUTO_COMMIT_LANDED",
            "AUTO_COMMIT_FAILED",
        )
        assert verdict.canonical_serializer_invoked is False  # auto_commit=False

    def test_constants_pinned(self) -> None:
        """Canonical constants are pinned per Catalog #302 + #314 + #340."""
        assert DEFAULT_SISTER_LANDING_LOOKBACK_MINUTES == 60
        assert DEFAULT_SISTER_LANDING_MEMO_GLOB == "*landed_*.md"
        assert "git log" in HEAD_STATE_PV_TOKENS
        assert "Catalog #229" in HEAD_STATE_PV_TOKENS
        assert "premise verification" in HEAD_STATE_PV_TOKENS
        # Exempt-paths set mirrors Catalog #340 sister.
        assert ".omx/state/modal_call_id_ledger.jsonl" in HANDOFF_EXEMPT_PATHS
        assert ".omx/state/subagent_progress.jsonl" in HANDOFF_EXEMPT_PATHS
        assert "MEMORY.md" in HANDOFF_EXEMPT_PATHS

    def test_recommendation_literals_match_taxonomy(self) -> None:
        """Recommendation taxonomy is closed; new values would surface here."""
        spawn_values = set(SpawnGuardRecommendation.__args__)
        assert spawn_values == {"PROCEED", "DUPLICATE_HEAD_STATE", "SISTER_IN_FLIGHT"}
        handoff_values = set(HandoffGuardRecommendation.__args__)
        assert handoff_values == {
            "PROCEED",
            "PREDECESSOR_UNCOMMITTED_WORK_DETECTED",
            "AUTO_COMMIT_LANDED",
            "AUTO_COMMIT_FAILED",
        }
