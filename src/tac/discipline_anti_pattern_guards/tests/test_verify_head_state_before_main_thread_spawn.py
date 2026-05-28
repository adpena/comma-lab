# SPDX-License-Identifier: MIT
"""Tests for the canonical main-thread spawn-decision PV helper.

Sister-test of ``tests/test_subagent_spawn_head_pv_guard.py`` at the
PARENT-MAIN-THREAD spawn-decision surface (Catalog #378 sister of
Catalog #376).
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pytest

from tac.discipline_anti_pattern_guards.main_thread_spawn_decision_pv_guard import (
    DEFAULT_LOOKBACK_MINUTES,
    DEFAULT_RECENT_COMMIT_LIMIT,
    MainThreadSpawnGuardVerdict,
    WAIT_AND_RETRY_THRESHOLD_MINUTES,
    _layer1_head_state_commit_overlap,
    _layer2_existing_file_overlap,
    _layer3_canonical_equation_overlap,
    _layer4_sister_checkpoint_overlap,
    _normalize_scope_token,
    _path_overlaps,
    verify_head_state_before_main_thread_spawn,
)


# ── Helper unit tests ──────────────────────────────────────────────────


def test_normalize_scope_token_strips_whitespace_and_trailing_slash() -> None:
    assert _normalize_scope_token("  foo/bar/  ") == "foo/bar"
    assert _normalize_scope_token("foo") == "foo"
    assert _normalize_scope_token("") == ""
    assert _normalize_scope_token("/") == ""


def test_path_overlaps_exact_match() -> None:
    assert _path_overlaps("src/tac/foo.py", "src/tac/foo.py") is True


def test_path_overlaps_declared_is_dir_prefix_of_candidate() -> None:
    assert _path_overlaps("src/tac/foo/", "src/tac/foo/bar.py") is True
    assert _path_overlaps("src/tac/foo", "src/tac/foo/bar.py") is True


def test_path_overlaps_candidate_is_dir_prefix_of_declared() -> None:
    assert _path_overlaps("src/tac/foo/bar.py", "src/tac/foo") is True


def test_path_overlaps_disjoint() -> None:
    assert _path_overlaps("src/tac/foo.py", "src/tac/bar.py") is False
    # Substring non-prefix should NOT match (e.g. "fo" vs "foo")
    assert _path_overlaps("fo", "foo") is False


def test_path_overlaps_empty_inputs() -> None:
    assert _path_overlaps("", "anything") is False
    assert _path_overlaps("anything", "") is False


# ── Verdict dataclass unit tests ──────────────────────────────────────


def test_verdict_proceed_construction() -> None:
    v = MainThreadSpawnGuardVerdict(
        recommendation="PROCEED",
        conflict_source="none",
        overlapping_scope=(),
        diagnostic="ok",
    )
    assert v.is_proceed is True
    assert v.is_abort is False


def test_verdict_abort_construction() -> None:
    v = MainThreadSpawnGuardVerdict(
        recommendation="ABORT_DUPLICATE_WORK_ON_DISK",
        conflict_source="head_recent_commit",
        overlapping_scope=("src/tac/foo.py",),
        diagnostic="duplicate work",
        cited_commits=("abc123def",),
    )
    assert v.is_proceed is False
    assert v.is_abort is True


def test_verdict_invalid_recommendation_raises() -> None:
    with pytest.raises(ValueError, match="recommendation must be one of"):
        MainThreadSpawnGuardVerdict(
            recommendation="UNKNOWN_REC",  # type: ignore[arg-type]
            conflict_source="none",
            overlapping_scope=(),
            diagnostic="",
        )


def test_verdict_invalid_conflict_source_raises() -> None:
    with pytest.raises(ValueError, match="conflict_source must be one of"):
        MainThreadSpawnGuardVerdict(
            recommendation="PROCEED",
            conflict_source="bogus_source",
            overlapping_scope=(),
            diagnostic="",
        )


def test_verdict_non_tuple_fields_raise() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        MainThreadSpawnGuardVerdict(
            recommendation="PROCEED",
            conflict_source="none",
            overlapping_scope=["not", "a", "tuple"],  # type: ignore[arg-type]
            diagnostic="",
        )


def test_verdict_non_str_diagnostic_raises() -> None:
    with pytest.raises(TypeError, match="diagnostic must be a str"):
        MainThreadSpawnGuardVerdict(
            recommendation="PROCEED",
            conflict_source="none",
            overlapping_scope=(),
            diagnostic=123,  # type: ignore[arg-type]
        )


# ── Layer 1: HEAD-state commit grep ───────────────────────────────────


def test_layer1_returns_empty_when_no_overlap(tmp_path: Path) -> None:
    """When git log has no overlap with declared_scope, return empty."""
    # tmp_path is not a git repo; the helper returns [] silently on
    # git failure (per CLAUDE.md fail-OPEN at this layer).
    overlaps = _layer1_head_state_commit_overlap(
        ["nonexistent/scope/xyz"],
        repo_root=tmp_path,
        recent_commit_limit=10,
    )
    assert overlaps == []


# ── Layer 2: Existing-file Glob scan ──────────────────────────────────


def test_layer2_returns_empty_when_no_overlap(tmp_path: Path) -> None:
    """When git ls-files has no overlap, return empty."""
    overlaps = _layer2_existing_file_overlap(
        ["nonexistent/scope/xyz"],
        repo_root=tmp_path,
    )
    assert overlaps == []


def test_layer2_returns_empty_when_no_declared_scope(tmp_path: Path) -> None:
    overlaps = _layer2_existing_file_overlap([], repo_root=tmp_path)
    assert overlaps == []


# ── Layer 3: Canonical equation registry query ────────────────────────


def test_layer3_returns_empty_when_registry_missing(tmp_path: Path) -> None:
    registry = tmp_path / "missing_registry.jsonl"
    overlaps = _layer3_canonical_equation_overlap(
        ["main_thread_spawn"],
        registry_path=registry,
    )
    assert overlaps == []


def test_layer3_matches_existing_equation_id(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    # Synthesize a registry row with the canonical schema.
    registry.write_text(
        json.dumps({
            "equation_id": "main_thread_spawn_pv_gap_pre_catalog_376_extension_v1",
            "name": "Main-thread spawn-decision PV gap",
            "one_line_summary": "predicts STAND_DOWN rate when PV unwired",
        }) + "\n",
        encoding="utf-8",
    )
    overlaps = _layer3_canonical_equation_overlap(
        ["main_thread_spawn"],
        registry_path=registry,
    )
    assert len(overlaps) == 1
    assert overlaps[0][1] == "main_thread_spawn_pv_gap_pre_catalog_376_extension_v1"


def test_layer3_matches_one_line_summary(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({
            "equation_id": "some_other_id_v1",
            "name": "irrelevant",
            "one_line_summary": "predicts STAND_DOWN rate when main_thread_spawn PV unwired",
        }) + "\n",
        encoding="utf-8",
    )
    overlaps = _layer3_canonical_equation_overlap(
        ["main_thread_spawn"],
        registry_path=registry,
    )
    assert len(overlaps) == 1


def test_layer3_no_match_when_disjoint(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({
            "equation_id": "brotli_cascade_v1",
            "name": "irrelevant",
            "one_line_summary": "irrelevant",
        }) + "\n",
        encoding="utf-8",
    )
    overlaps = _layer3_canonical_equation_overlap(
        ["dykstra_pareto"],
        registry_path=registry,
    )
    assert overlaps == []


def test_layer3_skips_malformed_rows(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "not valid JSON\n"
        + json.dumps({"equation_id": "real_id_v1", "name": "main_thread"}) + "\n"
        + "{\n"  # invalid JSON
        + json.dumps({"no_equation_id": "field"}) + "\n",
        encoding="utf-8",
    )
    overlaps = _layer3_canonical_equation_overlap(
        ["main_thread"],
        registry_path=registry,
    )
    assert len(overlaps) == 1


# ── Layer 4: Sister-subagent checkpoint scan ──────────────────────────


def _write_checkpoint(
    path: Path,
    *,
    subagent_id: str,
    written_at_utc: str,
    files_touched: list[str],
    status: str = "in_progress",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "subagent_id": subagent_id,
        "written_at_utc": written_at_utc,
        "files_touched": files_touched,
        "status": status,
        "step": 1,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def test_layer4_returns_empty_when_no_checkpoint_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    overlaps = _layer4_sister_checkpoint_overlap(
        ["src/tac/foo.py"],
        checkpoint_path=path,
        lookback_minutes=60,
        now_utc=now,
    )
    assert overlaps == []


def test_layer4_detects_sister_in_window(tmp_path: Path) -> None:
    path = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    # Sister is 5 min ago (within 60-min lookback).
    _write_checkpoint(
        path,
        subagent_id="sister_xyz",
        written_at_utc="2026-05-28T11:55:00+00:00",
        files_touched=["src/tac/foo.py", "src/tac/bar.py"],
    )
    overlaps = _layer4_sister_checkpoint_overlap(
        ["src/tac/foo.py"],
        checkpoint_path=path,
        lookback_minutes=60,
        now_utc=now,
    )
    assert len(overlaps) == 1
    sid, ts, paths = overlaps[0]
    assert sid == "sister_xyz"
    assert "src/tac/foo.py" in paths


def test_layer4_excludes_caller_subagent(tmp_path: Path) -> None:
    path = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    _write_checkpoint(
        path,
        subagent_id="my_self",
        written_at_utc="2026-05-28T11:55:00+00:00",
        files_touched=["src/tac/foo.py"],
    )
    overlaps = _layer4_sister_checkpoint_overlap(
        ["src/tac/foo.py"],
        checkpoint_path=path,
        lookback_minutes=60,
        now_utc=now,
        current_subagent_id="my_self",
    )
    assert overlaps == []


def test_layer4_skips_pre_cutoff_sisters(tmp_path: Path) -> None:
    path = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    # Sister is 90 min ago (outside 60-min lookback).
    _write_checkpoint(
        path,
        subagent_id="old_sister",
        written_at_utc="2026-05-28T10:30:00+00:00",
        files_touched=["src/tac/foo.py"],
    )
    overlaps = _layer4_sister_checkpoint_overlap(
        ["src/tac/foo.py"],
        checkpoint_path=path,
        lookback_minutes=60,
        now_utc=now,
    )
    assert overlaps == []


def test_layer4_skips_completed_sisters(tmp_path: Path) -> None:
    path = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    _write_checkpoint(
        path,
        subagent_id="done_sister",
        written_at_utc="2026-05-28T11:55:00+00:00",
        files_touched=["src/tac/foo.py"],
        status="complete",
    )
    overlaps = _layer4_sister_checkpoint_overlap(
        ["src/tac/foo.py"],
        checkpoint_path=path,
        lookback_minutes=60,
        now_utc=now,
    )
    assert overlaps == []


def test_layer4_latest_checkpoint_wins(tmp_path: Path) -> None:
    """When sister has multiple checkpoints, only the LATEST counts."""
    path = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    _write_checkpoint(
        path,
        subagent_id="sister_xyz",
        written_at_utc="2026-05-28T11:00:00+00:00",
        files_touched=["src/tac/foo.py"],
        status="in_progress",
    )
    _write_checkpoint(
        path,
        subagent_id="sister_xyz",
        written_at_utc="2026-05-28T11:55:00+00:00",
        files_touched=["src/tac/bar.py"],  # LATER files
        status="in_progress",
    )
    # Declared scope overlaps the LATER files only.
    overlaps = _layer4_sister_checkpoint_overlap(
        ["src/tac/bar.py"],
        checkpoint_path=path,
        lookback_minutes=60,
        now_utc=now,
    )
    assert len(overlaps) == 1
    sid, ts, paths = overlaps[0]
    assert "src/tac/bar.py" in paths


# ── verify_head_state_before_main_thread_spawn end-to-end ────────────


def test_proceed_when_declared_scope_empty(tmp_path: Path) -> None:
    v = verify_head_state_before_main_thread_spawn(
        declared_scope=[],
        repo_root=tmp_path,
        checkpoint_path=tmp_path / "missing.jsonl",
        canonical_equations_registry_path=tmp_path / "missing.jsonl",
    )
    assert v.is_proceed
    assert v.conflict_source == "none"


def test_proceed_when_no_conflict_any_layer(tmp_path: Path) -> None:
    v = verify_head_state_before_main_thread_spawn(
        declared_scope=["nonexistent_scope_xyz_unique_2026"],
        repo_root=tmp_path,
        checkpoint_path=tmp_path / "missing.jsonl",
        canonical_equations_registry_path=tmp_path / "missing.jsonl",
    )
    assert v.is_proceed
    assert v.conflict_source == "none"


def test_abort_sister_in_flight_when_sister_recent_but_below_wait_threshold(
    tmp_path: Path,
) -> None:
    """Sister is in flight but checkpoint is OLDER than wait threshold:
    returns ABORT_SISTER_IN_FLIGHT (sister may be stalled)."""
    checkpoint = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    # Sister is 30 min ago, MORE than the 10-min WAIT_AND_RETRY threshold.
    _write_checkpoint(
        checkpoint,
        subagent_id="sister_xyz",
        written_at_utc="2026-05-28T11:30:00+00:00",
        files_touched=["src/tac/foo.py"],
    )
    v = verify_head_state_before_main_thread_spawn(
        declared_scope=["src/tac/foo.py"],
        repo_root=tmp_path,
        checkpoint_path=checkpoint,
        canonical_equations_registry_path=tmp_path / "missing.jsonl",
        now_utc=now,
        wait_and_retry_threshold_minutes=10,
    )
    assert v.recommendation == "ABORT_SISTER_IN_FLIGHT"
    assert v.conflict_source == "sister_checkpoint"
    assert "sister_xyz" in v.cited_sister_subagent_ids


def test_wait_and_retry_when_sister_actively_working(tmp_path: Path) -> None:
    """Sister checkpoint is MORE RECENT than wait threshold:
    returns WAIT_AND_RETRY (actively working)."""
    checkpoint = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    # Sister is 5 min ago, LESS than the 10-min WAIT_AND_RETRY threshold.
    _write_checkpoint(
        checkpoint,
        subagent_id="active_sister",
        written_at_utc="2026-05-28T11:55:00+00:00",
        files_touched=["src/tac/foo.py"],
    )
    v = verify_head_state_before_main_thread_spawn(
        declared_scope=["src/tac/foo.py"],
        repo_root=tmp_path,
        checkpoint_path=checkpoint,
        canonical_equations_registry_path=tmp_path / "missing.jsonl",
        now_utc=now,
        wait_and_retry_threshold_minutes=10,
    )
    assert v.recommendation == "WAIT_AND_RETRY"
    assert v.conflict_source == "sister_checkpoint"


def test_abort_duplicate_work_when_canonical_equation_matches(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({
            "equation_id": "frontier_score_predictor_v1",
            "name": "Frontier",
            "one_line_summary": "predicts main_thread_spawn behavior",
        }) + "\n",
        encoding="utf-8",
    )
    v = verify_head_state_before_main_thread_spawn(
        declared_scope=["main_thread_spawn"],
        repo_root=tmp_path,
        checkpoint_path=tmp_path / "missing_checkpoint.jsonl",
        canonical_equations_registry_path=registry,
    )
    assert v.recommendation == "ABORT_DUPLICATE_WORK_ON_DISK"
    assert v.conflict_source == "canonical_equation"
    assert "frontier_score_predictor_v1" in v.cited_equation_ids


def test_sister_layer_takes_priority_over_equation_layer(tmp_path: Path) -> None:
    """When BOTH sister-in-flight AND canonical equation match, sister wins."""
    checkpoint = tmp_path / "subagent_progress.jsonl"
    now = _dt.datetime(2026, 5, 28, 12, 0, 0, tzinfo=_dt.timezone.utc)
    # Use a path-component scope token so the sister-overlap match fires
    # via _path_overlaps (path-component boundary semantics).
    _write_checkpoint(
        checkpoint,
        subagent_id="sister_xyz",
        written_at_utc="2026-05-28T11:55:00+00:00",
        files_touched=["src/tac/main_thread_spawn"],
    )
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps({
            "equation_id": "main_thread_spawn_pv_gap_v1",
            "name": "x",
            "one_line_summary": "y mentions src/tac/main_thread_spawn",
        }) + "\n",
        encoding="utf-8",
    )
    v = verify_head_state_before_main_thread_spawn(
        declared_scope=["src/tac/main_thread_spawn"],
        repo_root=tmp_path,
        checkpoint_path=checkpoint,
        canonical_equations_registry_path=registry,
        now_utc=now,
    )
    # Sister-in-flight wins per Layer-4 priority.
    assert v.conflict_source == "sister_checkpoint"


def test_invalid_declared_scope_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="declared_scope must be a list of strings"):
        verify_head_state_before_main_thread_spawn(declared_scope="not a list")  # type: ignore[arg-type]


def test_invalid_declared_scope_member_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="declared_scope must be a list of strings"):
        verify_head_state_before_main_thread_spawn(declared_scope=["ok", 42])  # type: ignore[list-item]


# ── Live-repo regression guard ────────────────────────────────────────


def test_live_repo_helper_importable_from_public_api() -> None:
    """Catalog #335 sister discipline: public API includes the helper."""
    from tac.discipline_anti_pattern_guards import (
        verify_head_state_before_main_thread_spawn as live_helper,
        MainThreadSpawnGuardVerdict as live_verdict,
    )
    assert live_helper is verify_head_state_before_main_thread_spawn
    assert live_verdict is MainThreadSpawnGuardVerdict


def test_live_repo_helper_handles_empty_scope_safely() -> None:
    """Live-repo invariant: empty scope always PROCEEDS."""
    v = verify_head_state_before_main_thread_spawn(declared_scope=[])
    assert v.is_proceed
