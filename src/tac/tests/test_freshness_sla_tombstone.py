# SPDX-License-Identifier: MIT
"""Tests for the #569 P0-3 freshness-SLA + tombstone extension of
tools/audit_memory_file_freshness.py.

Pins: (1) last_validated / superseded_by frontmatter parsing, (2) the
past_freshness_window SLA surface (last_validated OR mtime fallback), (3)
tombstones are surfaced and EXCLUDED from freshness + stale-by-age, and (4)
the registry-freshness pass (latest-row-wins + tombstone rows).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_memory_file_freshness as af  # noqa: E402

_NOW = datetime(2026, 7, 19, tzinfo=timezone.utc)


def _write(p: Path, body: str, *, mtime_days_ago: float | None = None) -> Path:
    p.write_text(body, encoding="utf-8")
    if mtime_days_ago is not None:
        ts = _NOW.timestamp() - mtime_days_ago * 86400.0
        os.utime(p, (ts, ts))
    return p


# ---------------------------------------------------------------- parsers ----
def test_parse_last_validated_and_superseded_target() -> None:
    text = '---\nname: x\nlast_validated: 2026-05-01\nsuperseded_by: newer.md\n---\nbody\n'
    lv = af._last_validated(text)
    assert lv is not None and lv.date().isoformat() == "2026-05-01"
    assert af._superseded_target(text) == "newer.md"


def test_parse_absent_returns_none() -> None:
    assert af._last_validated("no frontmatter here") is None
    assert af._superseded_target("no frontmatter here") is None


# ---------------------------------------------------------------- freshness --
def test_past_freshness_window_by_last_validated(tmp_path: Path) -> None:
    _write(tmp_path / "a.md", "---\nname: a\nlast_validated: 2026-01-01\n---\nbody\n")
    _write(tmp_path / "b.md", "---\nname: b\nlast_validated: 2026-07-10\n---\nbody\n")
    audit = af.audit_memory_files(tmp_path, freshness_days=90, now=_NOW)
    flagged = {r["filename"] for r in audit["past_freshness_window"]}
    assert "a.md" in flagged  # ~200 days since validation > 90
    assert "b.md" not in flagged  # 9 days since validation
    a_row = next(r for r in audit["past_freshness_window"] if r["filename"] == "a.md")
    assert a_row["validation_source"] == "last_validated"


def test_freshness_mtime_fallback(tmp_path: Path) -> None:
    # no last_validated frontmatter -> falls back to mtime age.
    _write(tmp_path / "old.md", "---\nname: old\n---\nbody\n", mtime_days_ago=200)
    _write(tmp_path / "new.md", "---\nname: new\n---\nbody\n", mtime_days_ago=5)
    audit = af.audit_memory_files(tmp_path, freshness_days=90, now=_NOW)
    rows = {r["filename"]: r for r in audit["past_freshness_window"]}
    assert "old.md" in rows
    assert rows["old.md"]["validation_source"] == "mtime_fallback"
    assert "new.md" not in rows


def test_tombstone_excluded_from_freshness_and_stale(tmp_path: Path) -> None:
    # a superseded file that is also old by mtime + validation.
    _write(
        tmp_path / "retired.md",
        "---\nname: retired\nlast_validated: 2026-01-01\nsuperseded_by: fresh.md\n---\nbody\n",
        mtime_days_ago=300,
    )
    audit = af.audit_memory_files(tmp_path, stale_days=60, freshness_days=90, now=_NOW)
    tomb = {r["filename"]: r for r in audit["tombstoned"]}
    assert "retired.md" in tomb
    assert tomb["retired.md"]["superseded_by"] == "fresh.md"
    # tombstoned rows are NOT freshness violations and NOT stale-by-age.
    assert "retired.md" not in {r["filename"] for r in audit["past_freshness_window"]}
    assert "retired.md" not in {r["filename"] for r in audit["stale_by_age"]}


def test_summary_counts_present(tmp_path: Path) -> None:
    _write(tmp_path / "a.md", "---\nname: a\nlast_validated: 2026-01-01\n---\nbody\n")
    _write(tmp_path / "t.md", "---\nname: t\nsuperseded_by: a.md\n---\nbody\n")
    audit = af.audit_memory_files(tmp_path, freshness_days=90, now=_NOW)
    s = audit["summary"]
    assert s["past_freshness_window_count"] == 1
    assert s["tombstoned_count"] == 1
    assert s["freshness_days_threshold"] == 90


# ---------------------------------------------------------------- registry ---
def test_registry_freshness_latest_row_wins_and_tombstone(tmp_path: Path) -> None:
    reg = tmp_path / "reg.jsonl"
    rows = [
        {"task_id": "#100", "written_at_utc": "2026-01-01T00:00:00Z"},   # stale, superseded later
        {"task_id": "#100", "written_at_utc": "2026-07-15T00:00:00Z"},   # fresh (latest wins)
        {"task_id": "#200", "written_at_utc": "2026-01-05T00:00:00Z"},   # stale
        {"task_id": "#300", "written_at_utc": "2026-02-01T00:00:00Z", "superseded_by": "#301"},
    ]
    reg.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    audit = af.audit_registry_freshness(reg, freshness_days=90, now=_NOW)
    stale_ids = {r["id"] for r in audit["past_freshness_window"]}
    assert stale_ids == {"#200"}  # #100 fresh (latest row), #300 tombstoned
    tomb_ids = {r["id"] for r in audit["tombstoned"]}
    assert tomb_ids == {"#300"}
    assert audit["summary"]["total_ids"] == 3


def test_registry_missing_file(tmp_path: Path) -> None:
    audit = af.audit_registry_freshness(tmp_path / "nope.jsonl", now=_NOW)
    assert "error" in audit["summary"]


def test_registry_report_renders(tmp_path: Path) -> None:
    reg = tmp_path / "reg.jsonl"
    reg.write_text(json.dumps({"task_id": "#9", "written_at_utc": "2026-01-01T00:00:00Z"}) + "\n",
                   encoding="utf-8")
    audit = af.audit_registry_freshness(reg, freshness_days=90, now=_NOW)
    text = af._format_registry_report(audit)
    assert "Registry freshness audit" in text
    assert "#9" in text
