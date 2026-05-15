# SPDX-License-Identifier: MIT
"""Tests for tools/view_pending_council_decisions.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "view_pending_council_decisions.py"


def _load_module():
    name = "view_pending_council_decisions"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    return path


def test_load_rows_handles_empty_file(tmp_path, mod):
    p = _write_jsonl(tmp_path / "q.jsonl", [])
    assert mod.load_rows(p) == []


def test_load_rows_skips_blank_lines(tmp_path, mod):
    p = tmp_path / "q.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n')
    rows = mod.load_rows(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_load_rows_raises_on_missing_file(tmp_path, mod):
    with pytest.raises(FileNotFoundError):
        mod.load_rows(tmp_path / "nope.jsonl")


def test_load_rows_raises_on_malformed_jsonl(tmp_path, mod):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\nnot-json\n')
    with pytest.raises(json.JSONDecodeError):
        mod.load_rows(p)


def test_filter_rows_default_status_pending_council(mod):
    rows = [
        {"decision_id": "a", "status": "pending_council"},
        {"decision_id": "b", "status": "resolved"},
    ]
    out = mod.filter_rows(rows)
    assert [r["decision_id"] for r in out] == ["a"]


def test_filter_rows_status_all_returns_everything(mod):
    rows = [
        {"decision_id": "a", "status": "pending_council"},
        {"decision_id": "b", "status": "resolved"},
        {"decision_id": "c", "status": "deferred"},
    ]
    out = mod.filter_rows(rows, status="all")
    assert len(out) == 3


def test_filter_rows_decision_id_substring(mod):
    rows = [
        {"decision_id": "oss_push_x", "status": "pending_council"},
        {"decision_id": "c1_route", "status": "pending_council"},
        {"decision_id": "oss_announce", "status": "pending_council"},
    ]
    out = mod.filter_rows(rows, decision_id="oss_")
    assert sorted(r["decision_id"] for r in out) == ["oss_announce", "oss_push_x"]


def test_group_by_priority_buckets_unknown(mod):
    rows = [
        {"council_priority": "HIGH"},
        {"council_priority": "MEDIUM"},
        {"council_priority": "LOW"},
        {"council_priority": "BOGUS"},
        {},  # missing -> UNKNOWN
    ]
    g = mod.group_by_priority(rows)
    assert len(g["HIGH"]) == 1
    assert len(g["MEDIUM"]) == 1
    assert len(g["LOW"]) == 1
    assert len(g["UNKNOWN"]) == 2  # BOGUS + missing


def test_format_row_includes_options_and_resolution(mod):
    row = {
        "decision_id": "x_y",
        "title": "Title T",
        "status": "resolved",
        "source_lane": "lane_z",
        "cost_usd": 0,
        "blocking": "nothing",
        "queued_utc": "2026-05-15T00:00:00Z",
        "options": ["opt-1", "opt-2"],
        "resolution": "PROCEED option 1",
        "resolved_by": "council_omnibus",
    }
    out = mod.format_row(row)
    assert "x_y" in out
    assert "Title T" in out
    assert "opt-1" in out and "opt-2" in out
    assert "PROCEED option 1" in out
    assert "council_omnibus" in out


def test_render_handles_no_matching_rows(mod):
    out = mod.render({}, total=0)
    assert "no rows match filter" in out


def test_render_groups_in_priority_order(mod):
    rows = [
        {"decision_id": "h1", "council_priority": "HIGH", "title": "h1t"},
        {"decision_id": "l1", "council_priority": "LOW", "title": "l1t"},
        {"decision_id": "m1", "council_priority": "MEDIUM", "title": "m1t"},
    ]
    g = mod.group_by_priority(rows)
    out = mod.render(g, total=3)
    pos_h = out.index("h1")
    pos_m = out.index("m1")
    pos_l = out.index("l1")
    assert pos_h < pos_m < pos_l


def test_main_json_emits_filtered_array(tmp_path, mod, capsys):
    rows = [
        {"decision_id": "a", "status": "pending_council", "council_priority": "HIGH"},
        {"decision_id": "b", "status": "resolved", "council_priority": "LOW"},
    ]
    p = _write_jsonl(tmp_path / "q.jsonl", rows)
    rc = mod.main(["--queue-path", str(p), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["decision_id"] == "a"


def test_main_returns_2_on_missing_queue(tmp_path, mod, capsys):
    rc = mod.main(["--queue-path", str(tmp_path / "nope.jsonl")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "queue file not found" in err


def test_main_handles_live_repo_queue_smoke():
    """Smoke-test against the live queue file if it exists; never hard-fail."""
    live = REPO_ROOT / ".omx" / "state" / "pending_council_design_decisions.jsonl"
    if not live.exists():
        pytest.skip("live queue not present")
    mod = _load_module()
    rc = mod.main(["--queue-path", str(live), "--json"])
    assert rc == 0
