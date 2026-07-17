"""Tests for check_codex_findings_memos_consumed (ARM F consumption audit,
2026-07-17) — the read-path detector for the landed-findings-nobody-consumed
orphan class (operator: "super poisonous bug class").

All tests run against a fabricated tmp repo root — no live .omx state.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from tac.preflight import (
    PreflightError,
    _codex_findings_arm_label,
    check_codex_findings_memos_consumed,
)

NOW = 1_800_000_000.0  # fixed clock for deterministic freshness math


def _mk_repo(tmp_path):
    research = tmp_path / ".omx" / "research"
    state = tmp_path / ".omx" / "state"
    research.mkdir(parents=True)
    state.mkdir(parents=True)
    return research, state


def _write(path, text, *, age_seconds=60.0):
    path.write_text(text, encoding="utf-8")
    mtime = NOW - age_seconds
    os.utime(path, (mtime, mtime))
    return path


# ---- label extraction ------------------------------------------------------
@pytest.mark.parametrize("filename,expected", [
    ("codex_findings_curvelet_throughR_p0_20260715_codex.md", "curvelet_throughR_p0"),
    ("codex_findings_jrd_coeff_prefix_20260712T224139Z_codex.md", "jrd_coeff_prefix"),
    ("codex_findings_harvest_held_catalog406_20260715_codex.md", "harvest_held_catalog406"),
    ("codex_findings_no_fourier_basis_20260715_codex.md", "no_fourier_basis"),
    ("codex_findings_some_arm_20260714.md", "some_arm"),
])
def test_arm_label_extraction(filename, expected):
    assert _codex_findings_arm_label(filename) == expected


# ---- orphan detection ------------------------------------------------------
def test_fresh_orphan_memo_flagged(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_lonely_arm_20260717_codex.md",
           "# findings\nnobody routed these\n")
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert len(v) == 1
    assert "lonely_arm" in v[0]
    assert "consumer surface" in v[0]


def test_stale_orphan_memo_not_flagged(tmp_path):
    # older than the 3-day grace window ⇒ outside this gate's jurisdiction
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_old_arm_20260601_codex.md",
           "# findings\n", age_seconds=10 * 24 * 3600)
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_dag_feed_companion_filename_clears(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_routed_arm_20260717_codex.md", "# findings\n")
    _write(research / "routed_arm_DAG_FEED_20260717.md", "decision recorded\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_ledger_consumed_by_receipt_clears(tmp_path):
    research, state = _mk_repo(tmp_path)
    _write(research / "codex_findings_ledger_arm_20260717_codex.md", "# findings\n")
    row = {"label": "ledger_arm_20260717", "status": "reviewed_committed",
           "consumed_by": ".omx/research/some_feed.md"}
    (state / "codex_landing_ledger.jsonl").write_text(json.dumps(row) + "\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_ledger_row_without_consumed_by_does_not_clear(tmp_path):
    # a bare custody disposition is NOT consumption — the proven orphan class
    research, state = _mk_repo(tmp_path)
    _write(research / "codex_findings_custody_arm_20260717_codex.md", "# findings\n")
    row = {"label": "custody_arm_20260717", "status": "reviewed_committed"}
    (state / "codex_landing_ledger.jsonl").write_text(json.dumps(row) + "\n")
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert len(v) == 1


def test_p0_ledger_content_clears(tmp_path):
    research, state = _mk_repo(tmp_path)
    _write(research / "codex_findings_p0_arm_xyz_20260717_codex.md", "# findings\n")
    (state / "operator_p0_ledger.jsonl").write_text(
        json.dumps({"p0_id": "p0_999", "notes": "absorb p0_arm_xyz findings"}) + "\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_recent_research_content_clears(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_content_arm_20260717_codex.md", "# findings\n")
    _write(research / "sub015_DAG_main.md",
           "### FEED-x — content_arm verdict NO-GO, reason recorded\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_stale_research_content_does_not_clear(tmp_path):
    # a consumer file older than the 30-day scan window is not read
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_ghost_arm_20260717_codex.md", "# findings\n")
    _write(research / "ancient_notes.md", "ghost_arm mentioned long ago\n",
           age_seconds=60 * 24 * 3600)
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert len(v) == 1


def test_sibling_codex_memo_mention_does_not_clear(tmp_path):
    # codex_findings_* / codex_session_summary_* are producer surfaces, not
    # consumer surfaces — a sister codex memo citing the arm is not consumption
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_sister_cited_arm_20260717_codex.md", "# f\n")
    _write(research / "codex_session_summary_20260717_codex.md",
           "consulted sister_cited_arm findings\n")
    _write(research / "codex_findings_other_20260717_codex.md",
           "stores consulted: sister_cited_arm\nCODEX_FINDINGS_CONSUMPTION_WAIVED: unit fixture\n")
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert [x for x in v if "sister_cited_arm" in x]


def test_in_memo_waiver_clears(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_waived_arm_20260717_codex.md",
           "# findings\nCODEX_FINDINGS_CONSUMPTION_WAIVED: pure-mechanical arm, "
           "no findings beyond the cherry-picked commits\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


@pytest.mark.parametrize("bad", ["<rationale>", "TBD", "n/a", ""])
def test_placeholder_waiver_rejected(tmp_path, bad):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_fakewaive_arm_20260717_codex.md",
           f"# findings\nCODEX_FINDINGS_CONSUMPTION_WAIVED: {bad}\n")
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert len(v) == 1


def test_strict_raises(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_strict_arm_20260717_codex.md", "# findings\n")
    with pytest.raises(PreflightError):
        check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW, strict=True)


def test_missing_research_dir_is_ok(tmp_path):
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_default_now_is_wallclock(tmp_path):
    # smoke: now=None path uses time.time() without exploding
    research, _ = _mk_repo(tmp_path)
    p = research / "codex_findings_wall_arm_20260717_codex.md"
    p.write_text("# findings\n")
    os.utime(p, (time.time() - 60, time.time() - 60))
    v = check_codex_findings_memos_consumed(repo_root=tmp_path)
    assert len(v) == 1
