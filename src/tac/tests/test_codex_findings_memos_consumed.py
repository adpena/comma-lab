"""Tests for check_codex_findings_memos_consumed (ARM F consumption audit,
2026-07-17) — the read-path detector for the landed-findings-nobody-consumed
orphan class (operator: "super poisonous bug class").

All tests run against a fabricated tmp repo root — no live .omx state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from tac.preflight import (
    PreflightError,
    _codex_findings_arm_label,
    check_codex_findings_memos_consumed,
)

# Fixed clock = 2026-07-18T00:00:00Z. ddm_rg5 (#825): producer age now comes from the memo's
# FILENAME DATE STAMP, not its mtime (mtime is a property of the checkout, not of the work — it
# is what made this gate scan 0 of 1,260 memos while reporting LIVE COUNT 0). The clock therefore
# has to sit in the same era as the fixture filenames, which are stamped 2026-07-17 / 2026-06-01.
# The previous value (1.8e9 = 2027-01-15) was ~6 months AHEAD of every fixture name, which the
# mtime axis hid entirely. ``_write(age_seconds=...)`` still drives mtime, which still governs the
# CONSUMER-side scan window — that axis is unchanged and the tests below still exercise it.
NOW = 1_784_332_800.0  # 2026-07-18T00:00:00Z


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
    # filename-stamped 2026-06-01, i.e. 47 days before NOW ⇒ outside the 30-day routing window
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_old_arm_20260601_codex.md",
           "# findings\n", age_seconds=10 * 24 * 3600)
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_producer_age_comes_from_filename_not_mtime(tmp_path):
    """THE regression guard for the vacuity: a freshly-touched file with an OLD name is stale.

    On the mtime axis this memo reads as 60 seconds old and would be scanned; on the honest axis
    its name says 2026-06-01, 47 days back, so it is out of the routing window. The mirror case
    (old mtime, in-window name) must still be scanned. Together these pin the axis itself, which
    is the property no previous test asserted — and its absence is why a checkout-dependent
    window survived review while scanning 0 of 1,260 memos.
    """
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_freshly_touched_20260601_codex.md",
           "# findings\n", age_seconds=60.0)
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []

    _write(research / "codex_findings_old_mtime_arm_20260717_codex.md",
           "# findings\n", age_seconds=365 * 24 * 3600)
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert len(v) == 1
    assert "old_mtime_arm" in v[0]


def test_unstamped_filename_falls_back_to_mtime(tmp_path):
    """A memo whose name carries no date stamp must not silently drop out of scope."""
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_nostamp_arm.md", "# findings\n", age_seconds=60.0)
    v = check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW)
    assert len(v) == 1
    assert "nostamp_arm" in v[0]

    _write(research / "codex_findings_nostamp_arm.md", "# findings\n",
           age_seconds=90 * 24 * 3600)
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_verbose_reports_scan_scope(capsys, tmp_path):
    """"OK" must never be printable without the scope it was computed over.

    The hollow gate printed ``OK (0 fresh memo(s) scanned)`` — technically honest, and read by
    every reviewer as "clean". The verdict now carries in-window count, total, and window width.
    """
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_scoped_arm_20260717_codex.md", "# findings\n")
    _write(research / "codex_findings_way_old_arm_20260101_codex.md", "# findings\n")
    check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW, verbose=True)
    out = capsys.readouterr().out
    assert "1 in-window memo(s) scanned of 2 total" in out
    assert "window 30d by filename date" in out


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


# ---- consumption-audit memo family carve-out --------------------------------
def test_audit_memo_family_not_scanned_as_producer(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_consumption_audit_20260717.md",
           "# audit memo — adjudication surface, not a findings memo\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_audit_memo_content_counts_as_consumer(tmp_path):
    research, _ = _mk_repo(tmp_path)
    _write(research / "codex_findings_audited_arm_20260717_codex.md", "# findings\n")
    _write(research / "codex_findings_consumption_audit_20260717.md",
           "ORPHAN routed: audited_arm -> reactivation route recorded here\n")
    assert check_codex_findings_memos_consumed(repo_root=tmp_path, now=NOW) == []


def test_date_axis_is_year_agnostic():
    """No one-year fuse: a 2027-stamped memo must still resolve on the filename axis.

    Both regexes originally hardcoded ``2026``. Left alone, on 2027-01-01 the stamp regex would
    fall back to mtime (re-acquiring the exact vacuity #825 fixes) and the label regex would stop
    stripping the stamp, so every label would match nothing and every memo would report as an
    orphan forever. A gate with an expiry date is a gate that will be ignored on the day it goes
    off, so the axis is pinned here rather than trusted.
    """
    import datetime as dt

    from tac.preflight import _codex_findings_memo_age_seconds

    assert _codex_findings_arm_label("codex_findings_future_arm_20270301_codex.md") == "future_arm"
    now = dt.datetime(2027, 3, 2, tzinfo=dt.UTC).timestamp()
    age = _codex_findings_memo_age_seconds(Path("codex_findings_future_arm_20270301_codex.md"), now)
    assert age is not None and 0 < age < 2 * 24 * 3600


def test_impossible_date_stamp_falls_back_instead_of_being_accepted(tmp_path):
    """``_20261332`` is not a date; strptime decides validity, not the pattern."""
    from tac.preflight import _codex_findings_memo_age_seconds

    research, _ = _mk_repo(tmp_path)
    memo = _write(research / "codex_findings_bad_stamp_20261332_codex.md", "# f\n",
                  age_seconds=60.0)
    age = _codex_findings_memo_age_seconds(memo, NOW)
    assert age is not None and age < 3600, "must fall back to mtime, not accept a fake date"
