# SPDX-License-Identifier: MIT
"""Tests for the SELF-FOLLOWING + STALENESS-HONEST level-set dashboard surfaces
(tools/render_levelset_dashboard.py, landed 2026-06-27).

Extincts the bug class where the dashboard rendered a DEAD run's log as if live
for hours: the watched log is now RE-RESOLVED each cycle to the newest
VERDICT-bearing file matching --log-glob (so the dashboard never self-follows
its own non-verdict daemon log), and a staleness banner keyed to the WATCHED
LOG's real mtime makes a stopped/crashed source impossible to miss.

Covers the pure (matplotlib-free where possible) helpers:
  * _has_verdict / _resolve_watched_log -- newest-verdict-log resolution,
    --log override, no-match, non-verdict-log exclusion, mtime tie-break.
  * _staleness / _fmt_age -- fresh-vs-stale threshold via fabricated old mtime.
  * _detect_switch -- first-resolution note vs switch-on-new-log note.
  * _write_html -- the banner is present + correctly LIVE vs STALE in the html.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.render_levelset_dashboard import (  # noqa: E402
    _detect_switch,
    _fmt_age,
    _has_verdict,
    _resolve_watched_log,
    _staleness,
    _write_html,
)

_VERDICT = '{"stage": "verdict", "epoch": %d, "d_seg": 0.004, "d_pose": 0.0009, "blob_bytes": 1234, "implied_S": 0.21}'


def _write_log(path: Path, *, epochs=(0, 1, 2), verdict=True, mtime=None) -> Path:
    lines = []
    if verdict:
        lines = [_VERDICT % e for e in epochs]
    else:
        lines = ['{"stage": "dashboard", "rendered": true}']
    path.write_text("\n".join(lines) + "\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# ---------------------------------------------------------------------------
# _has_verdict
# ---------------------------------------------------------------------------
def test_has_verdict_true_for_verdict_log(tmp_path):
    p = _write_log(tmp_path / "levelset_run.log")
    assert _has_verdict(p) is True


def test_has_verdict_false_for_non_verdict_log(tmp_path):
    # the dashboard's OWN daemon log -- no verdict lines -> must be excluded
    p = _write_log(tmp_path / "levelset_dashboard_daemon.log", verdict=False)
    assert _has_verdict(p) is False


def test_has_verdict_false_for_missing(tmp_path):
    assert _has_verdict(tmp_path / "nope.log") is False


# ---------------------------------------------------------------------------
# _resolve_watched_log
# ---------------------------------------------------------------------------
def test_resolve_picks_newest_verdict_log(tmp_path):
    old = _write_log(tmp_path / "levelset_old.log", mtime=1000)
    new = _write_log(tmp_path / "levelset_new.log", mtime=2000)
    resolved = _resolve_watched_log(None, str(tmp_path / "levelset_*.log"))
    assert resolved == new
    assert resolved != old


def test_resolve_switches_when_newer_log_appears(tmp_path):
    a = _write_log(tmp_path / "levelset_a.log", mtime=1000)
    assert _resolve_watched_log(None, str(tmp_path / "levelset_*.log")) == a
    # a brand-new run starts -> newer mtime verdict log -> auto-switch
    b = _write_log(tmp_path / "levelset_b.log", mtime=5000)
    assert _resolve_watched_log(None, str(tmp_path / "levelset_*.log")) == b


def test_resolve_excludes_newer_non_verdict_log(tmp_path):
    # THE confounder: dashboard daemon log has the NEWEST mtime but no verdicts.
    real = _write_log(tmp_path / "levelset_run.log", mtime=1000)
    _write_log(tmp_path / "levelset_dashboard_daemon.log", verdict=False, mtime=9999)
    resolved = _resolve_watched_log(None, str(tmp_path / "levelset_*.log"))
    assert resolved == real  # never self-follows the (newer) non-verdict log


def test_resolve_no_match_returns_none(tmp_path):
    assert _resolve_watched_log(None, str(tmp_path / "levelset_*.log")) is None


def test_resolve_single_match(tmp_path):
    only = _write_log(tmp_path / "levelset_only.log", mtime=1000)
    assert _resolve_watched_log(None, str(tmp_path / "levelset_*.log")) == only


def test_resolve_log_override_wins_and_is_verbatim(tmp_path):
    # --log explicit override (back-compat) returns the path verbatim, ignoring
    # the glob -- even when the override has no verdicts yet / does not exist.
    _write_log(tmp_path / "levelset_glob.log", mtime=9999)
    override = tmp_path / "explicit_chosen.log"
    resolved = _resolve_watched_log(str(override), str(tmp_path / "levelset_*.log"))
    assert resolved == override


def test_resolve_mtime_tie_breaks_deterministically(tmp_path):
    a = _write_log(tmp_path / "levelset_a.log", mtime=3000)
    b = _write_log(tmp_path / "levelset_b.log", mtime=3000)
    resolved = _resolve_watched_log(None, str(tmp_path / "levelset_*.log"))
    # equal mtime -> lexicographically-last name wins, deterministically
    assert resolved == b
    assert resolved != a


# ---------------------------------------------------------------------------
# _staleness + _fmt_age
# ---------------------------------------------------------------------------
def test_staleness_missing(tmp_path):
    st = _staleness(None, stale_min=5)
    assert st["state"] == "missing"
    assert st["age_s"] is None


def test_staleness_fresh_is_live(tmp_path):
    p = _write_log(tmp_path / "levelset_fresh.log", mtime=time.time())
    st = _staleness(p, stale_min=5)
    assert st["state"] == "live"
    assert st["age_s"] is not None and st["age_s"] < 60


def test_staleness_old_is_stale(tmp_path):
    # fabricated old mtime: 20 min ago, threshold 5 min -> STALE
    p = _write_log(tmp_path / "levelset_old.log", mtime=time.time() - 20 * 60)
    st = _staleness(p, stale_min=5)
    assert st["state"] == "stale"
    assert st["age_s"] > 5 * 60


def test_fmt_age_units():
    assert _fmt_age(None) == "?"
    assert _fmt_age(30) == "30s"
    assert _fmt_age(120).endswith("m")
    assert _fmt_age(2 * 3600).endswith("h")


# ---------------------------------------------------------------------------
# _detect_switch
# ---------------------------------------------------------------------------
def test_detect_switch_first_resolution(tmp_path):
    note = _detect_switch(None, tmp_path / "levelset_a.log", "2026-06-27T00:00:00Z")
    assert note is not None and "following" in note and "levelset_a.log" in note
    assert "switched" not in note


def test_detect_switch_unchanged_returns_none(tmp_path):
    note = _detect_switch("levelset_a.log", tmp_path / "levelset_a.log", "x")
    assert note is None


def test_detect_switch_on_new_log(tmp_path):
    note = _detect_switch("levelset_a.log", tmp_path / "levelset_b.log", "2026-06-27T01:02:03Z")
    assert note is not None and "switched" in note and "levelset_b.log" in note


def test_detect_switch_none_watched():
    assert _detect_switch("levelset_a.log", None, "x") is None


# ---------------------------------------------------------------------------
# _write_html banner end-to-end (matplotlib PNG is tiny stub bytes here)
# ---------------------------------------------------------------------------
def test_write_html_live_banner(tmp_path):
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 5, "d_seg": 0.004, "d_pose": 0.0009, "blob_bytes": 1, "implied_S": 0.21}]
    stale = {"state": "live", "age_s": 12.0, "mtime": time.time()}
    _write_html(out, b"PNG", rows, 30, watched=Path("levelset_live.log"),
                stale=stale, switched_note="▶ following levelset_live.log",
                log_glob=".omx/tmp/levelset_*.log")
    html = out.read_text()
    assert "● live" in html
    assert "STALE" not in html
    assert "levelset_live.log" in html
    assert "pointer UNMOVED 0.19110" in html  # footer preserved


def test_write_html_stale_banner(tmp_path):
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 5, "d_seg": 0.004, "d_pose": 0.0009, "blob_bytes": 1, "implied_S": 0.21}]
    stale = {"state": "stale", "age_s": 20 * 60.0, "mtime": time.time() - 20 * 60}
    _write_html(out, b"PNG", rows, 30, watched=Path("levelset_dead.log"),
                stale=stale, switched_note=None, log_glob=".omx/tmp/levelset_*.log")
    html = out.read_text()
    assert "⚠ STALE" in html
    assert "may be STOPPED/crashed" in html
    assert "● live" not in html


def test_write_html_missing_banner(tmp_path):
    out = tmp_path / "dash" / "index.html"
    stale = {"state": "missing", "age_s": None, "mtime": None}
    _write_html(out, b"PNG", [], 30, watched=None, stale=stale, switched_note=None,
                log_glob=".omx/tmp/levelset_*.log")
    html = out.read_text()
    assert "no run log found" in html


def test_write_html_backcompat_no_staleness_args(tmp_path):
    # existing callers that pass only (out, png, rows, refresh) still work
    out = tmp_path / "dash" / "index.html"
    _write_html(out, b"PNG", [], 30)
    assert out.exists()
    assert "auto-refresh 30s" in out.read_text()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
