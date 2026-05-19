# SPDX-License-Identifier: MIT
"""Tests for `tools/audit_memory_file_freshness.py` per-category rotation map.

Per CLAUDE.md "Memory file rotation discipline" non-negotiable + Wave 2C #8
finding (2026-05-19): uniform 60d rotation over-rotates fast-moving
categories (catalog gates / fix-wave findings) and under-rotates slow-moving
ones (project state). The per-category map at PER_CATEGORY_STALE_DAYS is the
canonical structural fix.

Lane: lane_hardening_license_plus_memory_hygiene_20260519.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Import the tool module directly (it lives under tools/, not src/tac).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_memory_file_freshness.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "audit_memory_file_freshness", _TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool_module()


def _set_mtime(path: Path, days_ago: float) -> None:
    """Backdate a file's mtime by N days."""
    target = time.time() - days_ago * 86400.0
    os.utime(path, (target, target))


def _write_memo(memory_dir: Path, name: str, days_ago: float = 0.0) -> Path:
    p = memory_dir / name
    p.write_text(f"# {name}\n\nMemo body.\n", encoding="utf-8")
    if days_ago > 0:
        _set_mtime(p, days_ago)
    return p


def test_per_category_stale_days_map_pinned(tool):
    """The canonical map must include the 5 Wave 2C #8 entries with the
    documented thresholds. Iteration order must place more-specific
    prefixes (catalog_) BEFORE general ones (feedback_)."""
    cmap = tool.PER_CATEGORY_STALE_DAYS
    assert cmap["catalog_"] == 7
    assert cmap["fix_wave_"] == 12
    assert cmap["codex_"] == 21
    assert cmap["project_"] == 90
    assert cmap["feedback_"] == 60
    keys = list(cmap.keys())
    # catalog_ + fix_wave_ + codex_ + project_ MUST appear before feedback_
    # so feedback_catalog_*.md classifies as catalog_ not feedback_.
    assert keys.index("catalog_") < keys.index("feedback_")
    assert keys.index("fix_wave_") < keys.index("feedback_")
    assert keys.index("codex_") < keys.index("feedback_")
    assert keys.index("project_") < keys.index("feedback_")


def test_category_lookup_helper_returns_substring_match(tool):
    days, category = tool._category_stale_days(
        "catalog_270_landed_20260515.md", default_days=60,
    )
    assert days == 7
    assert category == "catalog_"


def test_category_lookup_helper_prefers_first_match_on_overlap(tool):
    """`feedback_catalog_270_landed_*.md` matches BOTH catalog_ and
    feedback_; catalog_ comes first in iteration order so the file
    classifies as catalog_ (7d) NOT feedback_ (60d)."""
    days, category = tool._category_stale_days(
        "feedback_catalog_270_landed_20260515.md", default_days=60,
    )
    assert days == 7
    assert category == "catalog_"


def test_category_lookup_helper_falls_through_to_default(tool):
    days, category = tool._category_stale_days(
        "MEMORY.md", default_days=60,
    )
    assert days == 60
    assert category is None


def test_category_lookup_helper_honors_custom_map(tool):
    custom = {"unique_prefix_": 3}
    days, category = tool._category_stale_days(
        "unique_prefix_test.md",
        default_days=60,
        category_map=custom,
    )
    assert days == 3
    assert category == "unique_prefix_"


def test_audit_classifies_catalog_file_as_stale_at_8d(tool, tmp_path):
    """A catalog memo 8 days old MUST be flagged (7d window) even though
    it is well under the 60d global default."""
    _write_memo(tmp_path, "catalog_270_landed.md", days_ago=8.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = [row["filename"] for row in audit["stale_by_age"]]
    assert "catalog_270_landed.md" in names
    row = next(r for r in audit["stale_by_age"]
               if r["filename"] == "catalog_270_landed.md")
    assert row["category"] == "catalog_"
    assert row["threshold_days"] == 7


def test_audit_does_not_classify_catalog_file_at_5d_as_stale(tool, tmp_path):
    """5 days old < 7d catalog window → NOT flagged."""
    _write_memo(tmp_path, "catalog_270_landed.md", days_ago=5.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = [row["filename"] for row in audit["stale_by_age"]]
    assert "catalog_270_landed.md" not in names


def test_audit_honors_project_90d_for_old_project_file(tool, tmp_path):
    """A project_*.md 75d old is NOT stale (window is 90d)."""
    _write_memo(tmp_path, "project_lane_g_v3.md", days_ago=75.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = [row["filename"] for row in audit["stale_by_age"]]
    assert "project_lane_g_v3.md" not in names


def test_audit_flags_project_file_past_90d_window(tool, tmp_path):
    """A project_*.md 95d old IS stale (window is 90d)."""
    _write_memo(tmp_path, "project_lane_g_v3.md", days_ago=95.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = [row["filename"] for row in audit["stale_by_age"]]
    assert "project_lane_g_v3.md" in names
    row = next(r for r in audit["stale_by_age"]
               if r["filename"] == "project_lane_g_v3.md")
    assert row["category"] == "project_"
    assert row["threshold_days"] == 90


def test_audit_default_fallback_for_unmatched_filename(tool, tmp_path):
    """Files matching NO category fall through to the 60d global default."""
    _write_memo(tmp_path, "random_notes.md", days_ago=70.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = [row["filename"] for row in audit["stale_by_age"]]
    assert "random_notes.md" in names
    row = next(r for r in audit["stale_by_age"]
               if r["filename"] == "random_notes.md")
    assert row["category"] is None
    assert row["threshold_days"] == 60


def test_audit_disable_category_map_uses_uniform_default(tool, tmp_path):
    """Passing category_map={} disables per-category and falls back to
    uniform stale_days for ALL files."""
    _write_memo(tmp_path, "catalog_270_landed.md", days_ago=8.0)
    audit = tool.audit_memory_files(
        memory_dir=tmp_path, stale_days=60, category_map={},
    )
    names = [row["filename"] for row in audit["stale_by_age"]]
    # 8 days < 60d uniform → NOT flagged under disabled mode
    assert "catalog_270_landed.md" not in names


def test_audit_summary_exposes_category_map_and_breakdown(tool, tmp_path):
    """The summary dict must surface the active category map AND a
    per-category count of how many flagged rows fell in each bucket."""
    _write_memo(tmp_path, "catalog_a.md", days_ago=10.0)
    _write_memo(tmp_path, "catalog_b.md", days_ago=12.0)
    _write_memo(tmp_path, "fix_wave_a.md", days_ago=20.0)
    _write_memo(tmp_path, "feedback_long_tail.md", days_ago=80.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    summary = audit["summary"]
    # The map is echoed back so downstream consumers can audit it.
    assert summary["category_map"]["catalog_"] == 7
    assert summary["category_map"]["fix_wave_"] == 12
    by_cat = summary["stale_by_category_count"]
    assert by_cat.get("catalog_") == 2
    assert by_cat.get("fix_wave_") == 1
    assert by_cat.get("feedback_") == 1


def test_audit_respects_superseded_by_marker_within_category(tool, tmp_path):
    """A file with `superseded_by:` marker is NOT flagged even if past
    its category window."""
    p = tmp_path / "catalog_270_landed.md"
    p.write_text(
        "# catalog_270\n\nsuperseded_by: catalog_271_landed.md\n",
        encoding="utf-8",
    )
    _set_mtime(p, 30.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = [row["filename"] for row in audit["stale_by_age"]]
    assert "catalog_270_landed.md" not in names


def test_audit_missing_memory_dir_returns_error(tool, tmp_path):
    missing = tmp_path / "does_not_exist"
    audit = tool.audit_memory_files(memory_dir=missing, stale_days=60)
    assert "error" in audit["summary"]


def test_audit_handles_mixed_categories_correctly(tool, tmp_path):
    """End-to-end: 5 files across 5 categories at different ages.
    Each should classify against its own window."""
    # All 14d old: catalog_ flagged (>7d), fix_wave_ flagged (>12d),
    # codex_ NOT (<21d), project_ NOT (<90d), feedback_ NOT (<60d).
    _write_memo(tmp_path, "catalog_a.md", days_ago=14.0)
    _write_memo(tmp_path, "fix_wave_b.md", days_ago=14.0)
    _write_memo(tmp_path, "codex_c.md", days_ago=14.0)
    _write_memo(tmp_path, "project_d.md", days_ago=14.0)
    _write_memo(tmp_path, "feedback_e.md", days_ago=14.0)
    audit = tool.audit_memory_files(memory_dir=tmp_path, stale_days=60)
    names = {row["filename"] for row in audit["stale_by_age"]}
    assert "catalog_a.md" in names
    assert "fix_wave_b.md" in names
    assert "codex_c.md" not in names
    assert "project_d.md" not in names
    assert "feedback_e.md" not in names
