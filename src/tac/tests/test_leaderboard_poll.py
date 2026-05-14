# SPDX-License-Identifier: MIT
"""Tests for tools/leaderboard_poll.py — uses a recorded README fixture.

The fixture (`fixtures/leaderboard_readme_20260505.md`) is a real snapshot of
the upstream challenge README captured 2026-05-05. The test exercises:

  - leaderboard block extraction
  - row parsing -> structured entries
  - score-column hashing (cosmetic-edit-stable)
  - state save / load round-trip
  - change detection: same-fixture replay = no change
  - change detection: mutated score column = change

Per CLAUDE.md: every test added must actually pass.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "leaderboard_poll.py"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "leaderboard_readme_20260505.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("leaderboard_poll", TOOL_PATH)
    assert spec and spec.loader, f"could not load module from {TOOL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard_poll"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lp():
    return _load_module()


@pytest.fixture(scope="module")
def fixture_readme():
    assert FIXTURE_PATH.is_file(), f"fixture missing: {FIXTURE_PATH}"
    return FIXTURE_PATH.read_text()


def test_extract_block_finds_markers(lp, fixture_readme):
    block = lp.extract_leaderboard_block(fixture_readme)
    assert block.startswith("<!-- TABLE-START -->")
    assert block.endswith("<!-- TABLE-END -->")
    assert "<table" in block
    assert "</table>" in block


def test_extract_block_raises_when_markers_missing(lp):
    with pytest.raises(ValueError, match="leaderboard markers missing"):
        lp.extract_leaderboard_block("# README without markers")


def test_parse_returns_nonempty_entries(lp, fixture_readme):
    block = lp.extract_leaderboard_block(fixture_readme)
    entries = lp.parse_leaderboard_entries(block)
    assert len(entries) >= 5, f"expected >=5 entries, got {len(entries)}"
    # Sanity: every entry has a numeric score and a 1-indexed rank.
    for i, e in enumerate(entries, start=1):
        assert e.rank == i, f"rank not contiguous at {i}"
        assert isinstance(e.score, float)
        assert 0.0 <= e.score < 100.0
        assert e.name  # non-empty


def test_top_3_matches_known_snapshot(lp, fixture_readme):
    """The 2026-05-05 snapshot top-3 starts with hnerv_ft_microcodec @ 0.193."""
    block = lp.extract_leaderboard_block(fixture_readme)
    entries = lp.parse_leaderboard_entries(block)
    assert len(entries) >= 3
    top = entries[:3]
    assert top[0].score == pytest.approx(0.193)
    assert "hnerv" in top[0].name.lower()
    # PR number extracted (hnerv_ft_microcodec is PR #101 in the snapshot)
    assert top[0].pr_number == 101


def test_score_column_hash_is_stable_across_calls(lp, fixture_readme):
    block = lp.extract_leaderboard_block(fixture_readme)
    entries = lp.parse_leaderboard_entries(block)
    h1 = lp.hash_score_column(entries)
    h2 = lp.hash_score_column(entries)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_score_column_hash_ignores_cosmetic_edits(lp, fixture_readme):
    """Cosmetic edits (whitespace/emoji in name column) must NOT change the hash."""
    block = lp.extract_leaderboard_block(fixture_readme)
    entries = lp.parse_leaderboard_entries(block)
    h_orig = lp.hash_score_column(entries)

    # Mutate names only — score column unchanged
    mutated = [
        lp.LeaderboardEntry(
            rank=e.rank, score=e.score, name=e.name + " (edited)",
            pr_url=e.pr_url, pr_number=e.pr_number,
        )
        for e in entries
    ]
    h_mut = lp.hash_score_column(mutated)
    assert h_mut == h_orig


def test_score_column_hash_changes_when_score_changes(lp, fixture_readme):
    block = lp.extract_leaderboard_block(fixture_readme)
    entries = lp.parse_leaderboard_entries(block)
    h_orig = lp.hash_score_column(entries)

    # Mutate first entry's score
    mutated = list(entries)
    first = mutated[0]
    mutated[0] = lp.LeaderboardEntry(
        rank=first.rank, score=first.score - 0.001, name=first.name,
        pr_url=first.pr_url, pr_number=first.pr_number,
    )
    h_mut = lp.hash_score_column(mutated)
    assert h_mut != h_orig


def test_build_state_from_readme_roundtrip(lp, fixture_readme, tmp_path):
    state = lp.build_state_from_readme(fixture_readme)
    assert state.n_entries >= 5
    assert len(state.score_column_hash) == 64
    assert len(state.top_3) == 3

    out = tmp_path / "leaderboard_state.json"
    lp.save_state(state, out)
    assert out.is_file()

    loaded = lp.load_state(out)
    assert loaded is not None
    assert loaded.score_column_hash == state.score_column_hash
    assert loaded.n_entries == state.n_entries
    assert loaded.top_3 == state.top_3


def test_load_state_returns_none_when_missing(lp, tmp_path):
    assert lp.load_state(tmp_path / "does_not_exist.json") is None


def test_replay_same_readme_is_unchanged(lp, fixture_readme, tmp_path):
    """Same README -> same state hash -> no change detected."""
    s1 = lp.build_state_from_readme(fixture_readme)
    s2 = lp.build_state_from_readme(fixture_readme)
    # captured_utc differs, but score_column_hash MUST match
    assert s1.score_column_hash == s2.score_column_hash
    assert s1.top_3 == s2.top_3


def test_append_change_writes_jsonl(lp, fixture_readme, tmp_path):
    s_prev = lp.build_state_from_readme(fixture_readme)
    # Construct a "next" state with one mutated score
    block = lp.extract_leaderboard_block(fixture_readme)
    entries = lp.parse_leaderboard_entries(block)
    entries[0] = lp.LeaderboardEntry(
        rank=entries[0].rank, score=0.150, name="new_leader",
        pr_url=entries[0].pr_url, pr_number=entries[0].pr_number,
    )
    s_next = lp.LeaderboardState(
        score_column_hash=lp.hash_score_column(entries),
        captured_utc="2026-05-06T12:00:00Z",
        n_entries=len(entries),
        top_3=[
            {"rank": e.rank, "score": e.score, "name": e.name, "pr_url": e.pr_url}
            for e in entries[:3]
        ],
    )
    jsonl = tmp_path / "changes.jsonl"
    lp.append_change(s_prev, s_next, jsonl)
    assert jsonl.is_file()
    rec = json.loads(jsonl.read_text().strip())
    assert rec["prev_hash"] == s_prev.score_column_hash
    assert rec["curr_hash"] == s_next.score_column_hash
    assert rec["curr_top_3"][0]["name"] == "new_leader"


def test_touch_race_flag_creates_file(lp, tmp_path):
    flag = tmp_path / "RACE_MODE_ACTIVE.flag"
    assert not flag.exists()
    lp.touch_race_flag(flag)
    assert flag.is_file()
    # Touching again does not raise
    lp.touch_race_flag(flag)
    assert flag.is_file()
