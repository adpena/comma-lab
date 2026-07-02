# SPDX-License-Identifier: MIT
"""Tests for the SELF-FOLLOWING + SELF-CALIBRATING-STALENESS level-set dashboard
(tools/render_levelset_dashboard.py, landed 2026-06-27; staleness self-calibrated
FEED-ge 2026-06-27).

Extincts TWO bug classes:
  1. The dashboard rendered a DEAD run's log as if live for hours: the watched
     log is RE-RESOLVED each cycle to the newest VERDICT-bearing file matching
     --log-glob (never self-follows its own non-verdict daemon log).
  2. A hand-set fixed minute threshold (--stale-min 45 band-aid) false-alarmed
     "STOPPED" on a HEALTHY run whose verdict clusters are spaced minutes apart.
     The live/stale JUDGMENT is now DATA-DERIVED from the OBSERVED verdict
     cadence (median inter-arrival of new verdict epochs), with a labeled
     warming-up prior until calibrated and a >2.5×-cadence STALE trigger.

Covers the pure (matplotlib-free) helpers:
  * _has_verdict / _resolve_watched_log -- newest-verdict-log resolution.
  * _staleness / _fmt_age -- the file-mtime LOG-ACTIVITY signal (secondary).
  * _detect_switch -- first-resolution note vs switch-on-new-log note.
  * _verdict_ts_epoch / _arrival_times / _measure_cadence / _epoch_spacing /
    _compute_liveness / _cfg_from_args -- the self-calibrating cadence logic:
    warmup (<3 samples), live within cadence, stale >2.5×, embedded-ts instant
    calibration, --stale-min hard-floor override, verdict-age vs log-activity.
  * _banner_html / _write_html -- banner states + the two honest recency facts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.render_levelset_dashboard import (  # noqa: E402
    _arrival_times,
    _cfg_from_args,
    _ckpt_disk_footprint,
    _collect_run_info,
    _compute_liveness,
    _detail_line,
    _detect_switch,
    _epoch_spacing,
    _first_verdict_ts,
    _fmt_age,
    _fmt_bytes,
    _has_verdict,
    _load_cadence_state,
    _measure_cadence,
    _parse_checkpoints,
    _perf_env_status,
    _pill_html,
    _read_best_json,
    _resolve_watched_log,
    _resume_status,
    _run_dir_for,
    _run_info_html,
    _save_cadence_state,
    _stage_progress,
    _staleness,
    _status_line,
    _throughput_eta,
    _verdict_ts_epoch,
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
# _verdict_ts_epoch (forward-compat embedded-timestamp parsing)
# ---------------------------------------------------------------------------
def test_verdict_ts_epoch_parses_iso_z():
    e = _verdict_ts_epoch({"ts": "2026-06-27T20:00:00Z"})
    assert e is not None and e > 0


def test_verdict_ts_epoch_none_when_absent():
    assert _verdict_ts_epoch({"epoch": 5}) is None
    assert _verdict_ts_epoch({"ts": ""}) is None
    assert _verdict_ts_epoch({"ts": "not-a-date"}) is None


# ---------------------------------------------------------------------------
# _epoch_spacing (data-derived verdict-epoch spacing for the next estimate)
# ---------------------------------------------------------------------------
def test_epoch_spacing_median_diff():
    rows = [{"epoch": e} for e in (475, 500, 525, 550)]
    assert _epoch_spacing(rows, default=99) == 25


def test_epoch_spacing_default_when_single():
    assert _epoch_spacing([{"epoch": 5}], default=25) == 25


# ---------------------------------------------------------------------------
# _measure_cadence + _arrival_times
# ---------------------------------------------------------------------------
def test_arrival_times_embedded_ts_all_measured():
    # embedded ts -> arrival read directly; ALL verdicts are measured samples
    t0 = 1_700_000_000.0
    rows = [{"epoch": e, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + i * 1000))}
            for i, e in enumerate((0, 25, 50, 75))]
    sub = {}
    arrivals, baseline = _arrival_times(rows, sub, now=t0 + 3100)
    assert [a[2] for a in arrivals] == ["ts"] * 4  # all source=ts
    cfg = {"min_samples": 3, "prior_s": 1080.0}
    cad, src, n = _measure_cadence(arrivals, baseline, cfg)
    assert src == "measured" and n == 4
    assert abs(cad - 1000.0) < 1e-6  # median inter-arrival from the ts spacing


def test_arrival_times_self_observed_excludes_baseline():
    # no ts -> epochs present at launch are BASELINE (latched at 'now'), and must
    # NOT count as measured inter-arrivals; only later-observed epochs do.
    t0 = 1_700_000_000.0
    sub = {}
    rows0 = [{"epoch": e} for e in (475, 500, 525)]
    _arrival_times(rows0, sub, now=t0)  # first observation -> baseline {475,500,525}
    assert set(sub["baseline_epochs"]) == {475, 500, 525}
    cfg = {"min_gaps": 1}
    arrivals0, baseline = _arrival_times(rows0, sub, now=t0)
    cad0, src0, n0 = _measure_cadence(arrivals0, baseline, cfg)
    # baseline epochs (present at launch, no ts) are NOT measured gaps -> "calibrating"
    # with NO fabricated cadence number (the retired hardcoded 18m prior is gone).
    assert src0 == "calibrating" and cad0 is None and n0 == 0


def test_measure_cadence_becomes_measured_after_three_new_arrivals():
    cfg = {"min_samples": 3, "prior_s": 1080.0}
    sub = {
        "first_observation_at": 1000.0,
        "baseline_epochs": [0],
        "first_seen": {"0": 1000.0, "25": 1600.0, "50": 2200.0, "75": 2800.0},
    }
    rows = [{"epoch": e} for e in (0, 25, 50, 75)]
    arrivals, baseline = _arrival_times(rows, sub, now=2900.0)
    cad, src, n = _measure_cadence(arrivals, baseline, cfg)
    assert src == "measured"  # 3 post-baseline arrivals (25,50,75)
    assert n == 3
    assert abs(cad - 600.0) < 1e-6  # median of [600, 600]


# ---------------------------------------------------------------------------
# _compute_liveness -- the headline DATA-DERIVED judgment
# ---------------------------------------------------------------------------
_CFG = {"k": 2.5, "floor_s": 600.0, "prior_s": 1080.0, "min_samples": 3, "default_spacing": 25}


def _seg_rows(epochs):
    return [{"epoch": e, "d_seg": 0.004, "seg_form": "tau_softplus"} for e in epochs]


def test_liveness_missing_when_no_log():
    live = _compute_liveness([], None, time.time(), {}, _CFG)
    assert live["kind"] == "missing"
    assert live["verdict_age_s"] is None


def test_liveness_fresh_launch_reads_live_calibrating():
    # fresh self-observed run (all epochs baseline) on a recently-written log reads
    # LIVE (NOT a separate alarming state); cadence is the labeled prior.
    now = 1000.0
    live = _compute_liveness(_seg_rows((475, 500, 525)), watched_mtime=now - 60,
                             now=now, sub={}, cfg=_CFG)
    assert live["kind"] == "live"
    assert live["calibrating"] is True
    assert live["cadence_source"] == "calibrating"  # no gap yet -> calibrating, NOT a prior
    assert live["cadence_s"] is None               # never a fabricated number
    assert live["next_epoch"] == 550  # last + spacing(25)
    # baseline epoch -> verdict_age falls back to log mtime (consistency invariant)
    assert live["verdict_age_s"] >= live["log_age_s"]


def test_liveness_healthy_4m_since_verdict_at_18m_cadence_is_live():
    # THE coordinator regression: verdict 4m ago at a MEASURED ~18m cadence MUST be
    # LIVE (threshold = max(10m, 2.5*18m) = 45m; 4m << 45m). The old logic tripped
    # STALE far below its own displayed threshold — this guards that exact failure.
    t0 = 1000.0
    sub = {"first_observation_at": t0, "baseline_epochs": [0],
           "first_seen": {"0": t0, "25": t0 + 1080, "50": t0 + 2160, "75": t0 + 3240}}
    now = t0 + 3240 + 240  # 4 min after the last verdict (ep75)
    live = _compute_liveness(_seg_rows((0, 25, 50, 75)), watched_mtime=now - 5,
                             now=now, sub=sub, cfg=_CFG)
    assert live["kind"] == "live"
    assert live["cadence_source"] == "measured"
    assert abs(live["cadence_s"] - 1080.0) < 1e-6
    assert abs(live["verdict_age_s"] - 240.0) < 1e-6
    assert live["verdict_age_s"] < live["threshold_s"]  # 240 < 2700


def test_liveness_live_within_cadence_self_observed():
    sub = {"first_observation_at": 1000.0, "baseline_epochs": [0],
           "first_seen": {"0": 1000.0, "25": 1600.0, "50": 2200.0, "75": 2800.0}}
    now = 2900.0  # 100s after last verdict; measured cadence 600s -> threshold 1500s
    live = _compute_liveness(_seg_rows((0, 25, 50, 75)), watched_mtime=now - 5,
                             now=now, sub=sub, cfg=_CFG)
    assert live["kind"] == "live"
    assert live["cadence_source"] == "measured"
    assert live["calibrating"] is False
    assert abs(live["cadence_s"] - 600.0) < 1e-6
    assert abs(live["verdict_age_s"] - 100.0) < 1e-6


def test_liveness_stale_beyond_2p5x_cadence_AND_log_quiet():
    # DOUBLE-GATED stale: verdict overdue past K×cadence AND the log file ALSO quiet
    # (genuinely hung). measured cadence 600 -> threshold max(600,2.5*600)=1500.
    sub = {"first_observation_at": 1000.0, "baseline_epochs": [0],
           "first_seen": {"0": 1000.0, "25": 1600.0, "50": 2200.0, "75": 2800.0}}
    now = 2800.0 + 1600.0  # verdict 1600s old (> 1500 threshold)
    # log ALSO quiet (mtime 1600s old > log_threshold = cadence 600) -> genuinely hung
    live = _compute_liveness(_seg_rows((0, 25, 50, 75)), watched_mtime=now - 1600,
                             now=now, sub=sub, cfg=_CFG)
    assert live["kind"] == "stale"
    assert live["verdict_age_s"] > live["threshold_s"]
    assert live["log_age_s"] > live["log_threshold_s"]


def test_liveness_overdue_but_log_fresh_is_NOT_stale():
    # THE false-stale fix: verdict overdue past K×cadence but the LOG is being actively
    # written (checkpoints / in-flight async eval) -> NOT hung -> NOT stale. The old
    # logic flagged stale on verdict_age ALONE; the double-gate keeps a live run live.
    sub = {"first_observation_at": 1000.0, "baseline_epochs": [0],
           "first_seen": {"0": 1000.0, "25": 1600.0, "50": 2200.0, "75": 2800.0}}
    now = 2800.0 + 1600.0  # verdict 1600s old (overdue) ...
    live = _compute_liveness(_seg_rows((0, 25, 50, 75)), watched_mtime=now - 20,  # ... but log fresh
                             now=now, sub=sub, cfg=_CFG)
    assert live["kind"] == "live"  # process is writing -> alive


def test_liveness_calibrating_never_time_stales_no_fabricated_threshold():
    # NEW CONTRACT (operator "Telemetry accuracy vital"): while CALIBRATING (no measured
    # gap yet) we CANNOT know the cadence, so we NEVER flag stale on a fabricated time
    # threshold (the retired hardcoded prior/floor that mis-judged). A dead process in
    # this window surfaces via the SEPARATE training-alive signal (meta.training_alive),
    # not via an invented minute bound. So even a long-quiet calibrating log reads 'live'
    # (warming) from the time-telemetry's view — honest "I can't time-judge yet".
    now = 5000.0
    live = _compute_liveness(_seg_rows((475, 500)), watched_mtime=now - 3000,  # 50m quiet
                             now=now, sub={}, cfg=_CFG)
    assert live["kind"] == "live"
    assert live["calibrating"] is True
    assert live["cadence_s"] is None      # no fabricated number
    assert live["threshold_s"] is None    # no fabricated stale threshold while calibrating


def test_liveness_calibrating_healthy_quiet_stretch_not_stale():
    # THE false-alarm regression guard (FEED-gf live bug): a HEALTHY ~14-min quiet
    # stretch BETWEEN verdict clusters while still calibrating must NOT read STALE.
    # The old bare-10-min floor false-alarmed here on a live run; the threshold is
    # now max(floor, K*prior) ≈ 45m so a 14-min gap stays LIVE.
    now = 5000.0
    live = _compute_liveness(_seg_rows((475, 500)), watched_mtime=now - 14 * 60,
                             now=now, sub={}, cfg=_CFG)
    assert live["kind"] == "live"  # 840s < threshold 2700s
    assert live["calibrating"] is True


def test_liveness_embedded_ts_calibrates_instantly_live():
    t0 = 1_700_000_000.0
    rows = [{"epoch": e, "d_seg": 0.004, "seg_form": "tau_softplus",
             "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + i * 1000))}
            for i, e in enumerate((0, 25, 50, 75))]
    now = t0 + 3000 + 120  # 120s after last verdict
    live = _compute_liveness(rows, watched_mtime=now - 5, now=now, sub={}, cfg=_CFG)
    assert live["kind"] == "live"  # ts gives instant measured cadence, no warmup
    assert live["cadence_source"] == "measured"
    assert live["calibrating"] is False


def test_liveness_verdict_age_consistent_with_log_age():
    # the two honest facts: verdict recency >= log activity ALWAYS (a verdict IS a
    # log write, so it can never be fresher than the file mtime). Post-launch
    # observed epoch -> exact real recency; here the log was written more recently
    # (a later checkpoint line), so verdict_age (300s) > log_age (5s) but consistent.
    sub = {"first_observation_at": 1000.0, "baseline_epochs": [0],
           "first_seen": {"0": 1000.0, "25": 1600.0, "50": 2200.0, "75": 2800.0}}
    now = 3100.0  # log just written (mtime now-5), last verdict observed 300s ago
    live = _compute_liveness(_seg_rows((0, 25, 50, 75)), watched_mtime=now - 5,
                             now=now, sub=sub, cfg=_CFG)
    assert abs(live["verdict_age_s"] - 300.0) < 1e-6  # verdict recency (real)
    assert abs(live["log_age_s"] - 5.0) < 1e-6        # file-mtime activity (distinct)
    assert live["verdict_age_s"] >= live["log_age_s"]  # invariant: never inconsistent


# ---------------------------------------------------------------------------
# _cfg_from_args -- --stale-min HARD FLOOR override
# ---------------------------------------------------------------------------
def _args(**kw):
    base = dict(stale_min=None, stale_floor_min=10.0, cadence_k=2.5, cadence_prior_min=18.0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cfg_default_floor_is_stale_floor_min():
    cfg = _cfg_from_args(_args())
    assert cfg["floor_s"] == 10.0 * 60.0  # self-calibrated default floor


def test_cfg_stale_min_overrides_floor():
    cfg = _cfg_from_args(_args(stale_min=45.0))
    assert cfg["floor_s"] == 45.0 * 60.0  # hard floor override


def test_stale_min_hard_floor_keeps_live_when_default_would_stale():
    sub = {"first_observation_at": 1000.0, "baseline_epochs": [0],
           "first_seen": {"0": 1000.0, "25": 1600.0, "50": 2200.0, "75": 2800.0}}
    now = 2800.0 + 1600.0  # verdict 1600s old; measured cadence 600s; log ALSO quiet (1600s)
    rows = _seg_rows((0, 25, 50, 75))
    # default floor (10m=600s): threshold max(600, 1500)=1500 -> 1600 > 1500 + log quiet -> STALE
    default_cfg = _cfg_from_args(_args())
    assert _compute_liveness(rows, now - 1600, now, dict(sub), default_cfg)["kind"] == "stale"
    # hard floor override (45m=2700s): threshold max(2700, 1500)=2700 -> 1600 < 2700 -> LIVE
    override_cfg = _cfg_from_args(_args(stale_min=45.0))
    assert _compute_liveness(rows, now - 1600, now, dict(sub), override_cfg)["kind"] == "live"


# ---------------------------------------------------------------------------
# _load_cadence_state / _save_cadence_state -- restart-survival, per-log isolation
# ---------------------------------------------------------------------------
def test_cadence_state_roundtrip_per_log_name(tmp_path):
    p = tmp_path / "cadence.json"
    alls, sub = _load_cadence_state(str(p), "levelset_run_a.log")
    assert sub == {}
    sub["first_seen"] = {"25": 123.0}
    _save_cadence_state(str(p), alls, "levelset_run_a.log", sub)
    # same log name -> calibration survives a restart
    _, sub_a = _load_cadence_state(str(p), "levelset_run_a.log")
    assert sub_a["first_seen"] == {"25": 123.0}
    # a DIFFERENT (new run) log name -> fresh, never inherits the dead run's cadence
    _, sub_b = _load_cadence_state(str(p), "levelset_run_b.log")
    assert sub_b == {}


def test_cadence_state_missing_file_is_empty(tmp_path):
    alls, sub = _load_cadence_state(str(tmp_path / "nope.json"), "x.log")
    assert alls == {} and sub == {}


# ---------------------------------------------------------------------------
# _pill_html / _status_line / _detail_line -- clean minimal chrome
# ---------------------------------------------------------------------------
def test_pill_live():
    p = _pill_html({"kind": "live", "calibrating": False})
    assert "● live" in p and "pill live" in p


def test_pill_warming_up_when_calibrating():
    p = _pill_html({"kind": "live", "calibrating": True})
    assert "◐ warming up" in p and "pill warm" in p  # calm calibrating state, not red


def test_pill_stale():
    p = _pill_html({"kind": "stale", "calibrating": False})
    assert "⚠ stale" in p and "pill stale" in p


def test_pill_missing():
    p = _pill_html({"kind": "missing"})
    assert "no log" in p


def test_status_line_live_compact():
    rows = _seg_rows((725,))
    live = {"kind": "live", "calibrating": True, "last_epoch": 725,
            "next_epoch": 750, "next_eta_s": 16 * 60.0, "verdict_age_s": 120.0}
    s = _status_line(rows, live, tau=300, l7=900)
    assert "tau stage" in s and "ep725" in s and "next verdict ~16" in s.replace(".0", "")
    assert "—" not in s  # no stale text on a live run


def test_status_line_stale_says_stopped():
    rows = _seg_rows((725,))
    live = {"kind": "stale", "last_epoch": 725, "verdict_age_s": 50 * 60.0}
    s = _status_line(rows, live, tau=300, l7=900)
    assert "no verdict in" in s and "likely stopped" in s


def test_detail_line_both_facts_and_cadence():
    live = {"kind": "live", "calibrating": True, "verdict_age_s": 108.0,
            "log_age_s": 108.0, "cadence_s": 1080.0}
    d = _detail_line(live)
    assert "verdict" in d and "log" in d and "cadence ~18m (calibrating)" in d


# ---------------------------------------------------------------------------
# _write_html end-to-end (matplotlib PNG is tiny stub bytes here)
# ---------------------------------------------------------------------------
def test_write_html_live_pill_and_footer(tmp_path):
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 5, "d_seg": 0.004, "d_pose": 0.0009, "blob_bytes": 1,
             "implied_S": 0.21, "seg_form": "tau_softplus"}]
    live = {"kind": "live", "calibrating": False, "last_epoch": 5, "verdict_age_s": 12.0,
            "log_age_s": 12.0, "cadence_s": 1080.0, "cadence_source": "measured",
            "next_epoch": 30, "next_eta_s": 900.0, "threshold_s": 2700.0}
    _write_html(out, b"PNG", rows, 30, watched=Path("levelset_live.log"), live=live,
                log_glob=".omx/tmp/levelset_*.log")
    html = out.read_text()
    assert "● live" in html and "pill live" in html
    assert "⚠ stale" not in html
    assert "Level-Set Witness" in html              # clean title
    assert "pointer 0.19110" in html                # minimal footer
    assert "NON-PROMOTABLE" in html                  # advisory tag preserved
    assert "levelset_live.log" in html              # watched name in footer


def test_write_html_stale_pill(tmp_path):
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 5, "d_seg": 0.004, "d_pose": 0.0009, "blob_bytes": 1, "implied_S": 0.21}]
    live = {"kind": "stale", "calibrating": False, "last_epoch": 5, "verdict_age_s": 50 * 60.0,
            "log_age_s": 50 * 60.0, "cadence_s": 1080.0, "cadence_source": "measured",
            "next_epoch": 30, "next_eta_s": 0.0, "threshold_s": 2700.0}
    _write_html(out, b"PNG", rows, 30, watched=Path("levelset_dead.log"), live=live,
                log_glob=".omx/tmp/levelset_*.log")
    html = out.read_text()
    assert "⚠ stale" in html and "pill stale" in html
    assert "likely stopped" in html
    assert "pill live" not in html


def test_write_html_missing_pill(tmp_path):
    out = tmp_path / "dash" / "index.html"
    live = {"kind": "missing", "calibrating": True, "last_epoch": None, "verdict_age_s": None,
            "log_age_s": None, "cadence_s": None, "cadence_source": None,
            "next_epoch": None, "next_eta_s": None, "threshold_s": None}
    _write_html(out, b"PNG", [], 30, watched=None, live=live,
                log_glob=".omx/tmp/levelset_*.log")
    html = out.read_text()
    assert "no run log found" in html and "no log" in html


def test_write_html_backcompat_no_staleness_args(tmp_path):
    # existing callers that pass only (out, png, rows, refresh) still work
    out = tmp_path / "dash" / "index.html"
    _write_html(out, b"PNG", [], 30)
    assert out.exists()
    assert "refresh 30s" in out.read_text()


def test_write_html_headline_clean(tmp_path):
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 500, "d_seg": 0.0045, "d_pose": 0.0008, "blob_bytes": 74000,
             "implied_S": 0.59, "seg_form": "tau_softplus"},
            {"epoch": 525, "d_seg": 0.0044, "d_pose": 0.0007, "blob_bytes": 73000,
             "implied_S": 0.57, "seg_form": "tau_softplus"}]
    live = {"kind": "live", "calibrating": False, "last_epoch": 525, "verdict_age_s": 30.0,
            "log_age_s": 30.0, "cadence_s": 1080.0, "cadence_source": "measured",
            "next_epoch": 550, "next_eta_s": 900.0, "threshold_s": 2700.0}
    _write_html(out, b"PNG", rows, 30, watched=Path("levelset_run.log"), live=live,
                tau=300, l7=900, goal_dseg=0.00112)
    html = out.read_text()
    assert "0.00440" in html               # big d_seg headline value
    assert "goal &lt;0.00112" in html      # compact goal note
    assert "tau stage" in html and "ep525" in html  # one-line status
    assert "stage shading" not in html     # dense legend removed (clean)


# ===========================================================================
# #205 MULTI-DAY RUN OBSERVABILITY readers (stage progress · best · throughput ·
# checkpoint ledger · resumability · perf fast-path · config) — ALL REAL telemetry.
# ===========================================================================
import json as _json  # noqa: E402

_CKPT_BEST = '{"stage": "checkpoint", "kind": "best", "epoch": 675, "d_seg": 0.00412, "prev_best": 0.0044, "path": "levelset_witness_ema_BEST.npz"}'
_CKPT_STAGE = '{"stage": "checkpoint", "kind": "stage_transition", "epoch": 300, "ema_latest": "levelset_witness_ema_mlx.npz", "resume_latest": "levelset_resume_state.npz", "has_opt": true, "ema_preserved": "levelset_ckpt_stageCE_ep300.npz", "resume_preserved": "levelset_resume_stageCE_ep300.npz"}'
_CKPT_INTRA = '{"stage": "checkpoint", "kind": "intra_stage", "epoch": 600, "ema_latest": "levelset_witness_ema_mlx.npz", "resume_latest": "levelset_resume_state.npz", "has_opt": true}'
_PERF_ACTIVE = '{"stage": "custom_grouped_backward", "active": true, "reason": "env", "path": "MLXCustomKernelStridedGroupedConvAdapter"}'
_PERF_INACTIVE = '{"stage": "custom_grouped_backward", "active": false, "reason": "TAC_MLX_CUSTOM_GROUPED_BACKWARD unset"}'


def _write_run_log(path: Path, extra_lines=(), epochs=(0, 25, 50)) -> Path:
    lines = [_VERDICT % e for e in epochs] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# _fmt_bytes
# ---------------------------------------------------------------------------
def test_fmt_bytes_units():
    assert _fmt_bytes(None) == "?"
    assert _fmt_bytes(512) == "512B"
    assert _fmt_bytes(2048).endswith("KB")
    assert _fmt_bytes(5 * 1024 ** 2).endswith("MB")
    assert _fmt_bytes(3 * 1024 ** 3).endswith("GB")


# ---------------------------------------------------------------------------
# _run_dir_for
# ---------------------------------------------------------------------------
def test_run_dir_arg_wins(tmp_path):
    assert _run_dir_for(tmp_path / "a" / "run.log", "x/*.log", str(tmp_path / "explicit")) \
        == Path(tmp_path / "explicit")


def test_run_dir_from_watched_parent(tmp_path):
    w = tmp_path / "myrun" / "run.log"
    assert _run_dir_for(w, None, None) == (tmp_path / "myrun")


def test_run_dir_from_glob_dir():
    assert _run_dir_for(None, "runs/abc/*.log", None) == Path("runs/abc")


def test_run_dir_none_when_unresolvable():
    assert _run_dir_for(None, "*.log", None) is None
    assert _run_dir_for(None, None, None) is None


# ---------------------------------------------------------------------------
# _read_best_json
# ---------------------------------------------------------------------------
def test_read_best_json_present(tmp_path):
    (tmp_path / "levelset_best.json").write_text(
        '{"d_seg": 0.0041, "epoch": 675, "path": "levelset_witness_ema_BEST.npz", "ts": "x"}')
    d = _read_best_json(tmp_path)
    assert d and d["d_seg"] == 0.0041 and d["epoch"] == 675


def test_read_best_json_absent_and_malformed(tmp_path):
    assert _read_best_json(tmp_path) is None       # no file
    assert _read_best_json(None) is None
    (tmp_path / "levelset_best.json").write_text("{ not json")
    assert _read_best_json(tmp_path) is None        # malformed -> None (never raises)


# ---------------------------------------------------------------------------
# _parse_checkpoints
# ---------------------------------------------------------------------------
def test_parse_checkpoints_ledger(tmp_path):
    log = _write_run_log(tmp_path / "run.log", extra_lines=[_CKPT_STAGE, _CKPT_INTRA, _CKPT_BEST])
    ck = _parse_checkpoints(log)
    assert ck["best"]["d_seg"] == 0.00412
    assert ck["counts"] == {"stage_transition": 1, "intra_stage": 1, "best": 1}
    # one PRESERVED stage-encoded ckpt (the stage_transition carries ema_preserved)
    assert len(ck["preserved"]) == 1
    assert ck["preserved"][0]["ema"] == "levelset_ckpt_stageCE_ep300.npz"
    assert ck["preserved"][0]["epoch"] == 300
    # last non-best resumable ckpt line = the intra_stage (ep600)
    assert ck["last"]["epoch"] == 600
    assert ck["last_intra"]["epoch"] == 600


def test_parse_checkpoints_none_and_empty(tmp_path):
    assert _parse_checkpoints(None)["best"] is None
    empty = _write_run_log(tmp_path / "run.log")  # verdicts only, no checkpoint lines
    ck = _parse_checkpoints(empty)
    assert ck["best"] is None and ck["preserved"] == [] and ck["counts"] == {}


def test_parse_checkpoints_skips_malformed_line(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(_CKPT_BEST + "\n" + '{"stage": "checkpoint", "kind": "intra_stage"' + "\n")  # truncated
    ck = _parse_checkpoints(p)
    assert ck["best"] is not None  # good line parsed; bad line skipped (no crash)


# ---------------------------------------------------------------------------
# _ckpt_disk_footprint
# ---------------------------------------------------------------------------
def test_ckpt_disk_footprint(tmp_path):
    (tmp_path / "levelset_witness_ema_mlx.npz").write_bytes(b"\0" * 2048)
    (tmp_path / "levelset_resume_state.npz").write_bytes(b"\0" * 4096)
    (tmp_path / "unrelated.txt").write_bytes(b"\0" * 999)
    fp = _ckpt_disk_footprint(tmp_path)
    assert fp["count"] == 2 and fp["total_bytes"] == 6144
    assert fp["biggest"][0][1] == 4096  # largest first


def test_ckpt_disk_footprint_none():
    fp = _ckpt_disk_footprint(None)
    assert fp["count"] == 0 and fp["total_bytes"] == 0


# ---------------------------------------------------------------------------
# _resume_status
# ---------------------------------------------------------------------------
def test_resume_status_resumable_via_resume_npz(tmp_path):
    (tmp_path / "levelset_resume_state.npz").write_bytes(b"\0" * 10)
    ck = {"last": {"epoch": 600}}
    r = _resume_status(tmp_path, ck, now=time.time())
    assert r["resumable"] is True and r["resume_npz"] is True and r["resume_epoch"] == 600
    assert r["resume_age_s"] is not None


def test_resume_status_resumable_via_deploy_npz_only(tmp_path):
    (tmp_path / "levelset_witness_ema_mlx.npz").write_bytes(b"\0" * 10)
    r = _resume_status(tmp_path, {}, now=time.time())
    assert r["resumable"] is True and r["deploy_npz"] is True and r["resume_npz"] is False


def test_resume_status_not_resumable_when_no_ckpt(tmp_path):
    r = _resume_status(tmp_path, {}, now=time.time())
    assert r["resumable"] is False
    assert _resume_status(None, {})["resumable"] is False


# ---------------------------------------------------------------------------
# _perf_env_status
# ---------------------------------------------------------------------------
def test_perf_env_active(tmp_path):
    log = _write_run_log(tmp_path / "run.log", extra_lines=[_PERF_ACTIVE])
    assert _perf_env_status(log)["active"] is True


def test_perf_env_inactive(tmp_path):
    log = _write_run_log(tmp_path / "run.log", extra_lines=[_PERF_INACTIVE])
    st = _perf_env_status(log)
    assert st["active"] is False and "unset" in st["reason"]


def test_perf_env_not_seen(tmp_path):
    log = _write_run_log(tmp_path / "run.log")  # no perf line yet
    assert _perf_env_status(log)["active"] is None
    assert _perf_env_status(None)["active"] is None


# ---------------------------------------------------------------------------
# _first_verdict_ts
# ---------------------------------------------------------------------------
def test_first_verdict_ts():
    rows = [{"epoch": 0}, {"epoch": 25, "ts": "2026-07-01T00:00:00Z"},
            {"epoch": 50, "ts": "2026-07-01T01:00:00Z"}]
    assert abs(_first_verdict_ts(rows) - _verdict_ts_epoch({"ts": "2026-07-01T00:00:00Z"})) < 1e-6
    assert _first_verdict_ts([{"epoch": 0}]) is None  # no ts anywhere


# ---------------------------------------------------------------------------
# _stage_progress
# ---------------------------------------------------------------------------
_SCHED = {"epochs": 1500, "eval_every": 25, "tau_start": 300, "l7_start": 800, "muon_start": 1100}


def test_stage_progress_ce():
    sp = _stage_progress(150, "ce", _SCHED)
    assert sp["stage"] == "CE"
    assert abs(sp["stage_pct"] - 0.5) < 1e-6      # ep150 of [0,300)
    assert abs(sp["total_pct"] - 0.1) < 1e-6      # ep150 of 1500
    assert sp["next_name"] == "tau" and sp["next_epoch"] == 300
    assert sp["muon_active"] is False


def test_stage_progress_tau_from_seg_form():
    sp = _stage_progress(550, "tau_softplus", _SCHED)
    assert sp["stage"] == "tau"
    assert abs(sp["stage_pct"] - 0.5) < 1e-6      # ep550 of [300,800)
    assert sp["next_name"] == "l7" and sp["next_epoch"] == 800


def test_stage_progress_l7_with_muon_overlay():
    sp = _stage_progress(1200, "l7_softplus", _SCHED)
    assert sp["stage"] == "l7"
    assert sp["muon_active"] is True             # ep1200 >= muon_start 1100
    assert sp["next_name"] == "end" and sp["next_epoch"] == 1500


def test_stage_progress_derives_stage_when_seg_form_absent():
    # no seg_form on the row -> derive from schedule boundaries
    assert _stage_progress(150, None, _SCHED)["stage"] == "CE"
    assert _stage_progress(500, None, _SCHED)["stage"] == "tau"
    assert _stage_progress(900, None, _SCHED)["stage"] == "l7"


def test_stage_progress_none_epoch():
    sp = _stage_progress(None, None, _SCHED)
    assert sp["stage"] is None and sp["stage_pct"] is None


def test_stage_progress_empty_schedule():
    sp = _stage_progress(150, "ce", {})
    assert sp["stage"] == "CE"           # seg_form still names it
    assert sp["stage_pct"] is None       # no boundaries -> no % (never fabricated)
    assert sp["total_pct"] is None


# ---------------------------------------------------------------------------
# _throughput_eta
# ---------------------------------------------------------------------------
def test_throughput_eta_measured():
    live = {"cadence_s": 2612.5, "cadence_source": "measured", "last_epoch": 550}
    sp = {"next_epoch": 800}
    now = 1_700_000_000.0
    thr = _throughput_eta(live, _SCHED, sp, first_ts=now - 3600 * 26, now=now)
    # s_per_epoch = cadence / eval_every = 2612.5 / 25 = 104.5
    assert abs(thr["s_per_epoch"] - 104.5) < 1e-6
    assert abs(thr["epochs_per_hour"] - (3600 / 104.5)) < 1e-3
    assert abs(thr["elapsed_s"] - 3600 * 26) < 1.0
    # eta to next boundary ep800: (800-550)*104.5
    assert abs(thr["eta_next_boundary_s"] - 250 * 104.5) < 1e-3
    # eta to done ep1500: (1500-550)*104.5
    assert abs(thr["eta_total_s"] - 950 * 104.5) < 1e-3


def test_throughput_eta_calibrating_gives_none():
    live = {"cadence_s": None, "cadence_source": "calibrating", "last_epoch": 100}
    thr = _throughput_eta(live, _SCHED, {"next_epoch": 300}, first_ts=None)
    assert thr["s_per_epoch"] is None and thr["eta_total_s"] is None
    assert thr["elapsed_s"] is None  # no first_ts


# ---------------------------------------------------------------------------
# _run_info_html
# ---------------------------------------------------------------------------
def test_run_info_html_empty():
    assert _run_info_html(None) == ""
    assert _run_info_html({}) == ""


def test_run_info_html_renders_all_cards():
    info = {
        "schedule": {"epochs": 1500, "eval_every": 25, "tau_start": 300, "l7_start": 800,
                     "muon_start": 1100, "__ckpt_every": 100},
        "stage_prog": {"stage": "tau", "stage_pct": 0.5, "total_pct": 0.45, "epochs": 1500,
                       "next_name": "l7", "next_epoch": 800, "muon_active": False,
                       "__last_epoch": 550},
        "best": {"d_seg": 0.00412, "epoch": 540, "path": "levelset_witness_ema_BEST.npz"},
        "checkpoints": {"counts": {"intra_stage": 5, "stage_transition": 1},
                        "preserved": [{"epoch": 300, "ema": "x.npz"}]},
        "resume": {"resumable": True, "resume_epoch": 600, "resume_npz": True,
                   "resume_age_s": 120.0, "deploy_npz": True, "deploy_age_s": 120.0},
        "throughput": {"s_per_epoch": 104.5, "epochs_per_hour": 34.4, "elapsed_s": 3600 * 26,
                       "eta_next_boundary_s": 26125.0, "eta_total_s": 99275.0},
        "perf": {"active": True, "reason": "env"},
        "disk": {"count": 4, "total_bytes": 600 * 1024},
        "config": {"source": "launch.sh", "flags": {"w-seg": "100.0", "w-pose": "0.0",
                                                     "activation": "hosc", "mod-dim": "32",
                                                     "hidden-dim": "96", "seed": "0",
                                                     "num-pairs": "600"}},
    }
    html = _run_info_html(info)
    for needle in ("curriculum stage", "best d_seg", "throughput", "checkpoints",
                   "resumability", "MLX fast path", "config",
                   "tau", "0.00412", "104.5s/ep", "✓ resumable", "✓ active",
                   "w_seg 100.0", "act hosc", "ckpt-every 100", "launch.sh"):
        assert needle in html, f"MISSING: {needle}"


def test_run_info_html_unknowns_render_gracefully():
    # a fresh/partial run: stage unknown, no best, calibrating throughput, perf not seen
    info = {"schedule": {}, "stage_prog": {"stage": None, "__last_epoch": None},
            "best": None, "checkpoints": {"counts": {}, "preserved": []},
            "resume": {"resumable": False}, "throughput": {}, "perf": {"active": None},
            "disk": {}, "config": {"source": "none", "flags": {}}}
    html = _run_info_html(info)
    assert "no finite best yet" in html
    assert "calibrating" in html            # throughput card
    assert "✗ not yet" in html              # resumability
    assert "grouped-backward line not seen yet" in html
    assert "?" in html                       # unknown config values render '?'


# ---------------------------------------------------------------------------
# _collect_run_info end-to-end (REAL run-dir layout)
# ---------------------------------------------------------------------------
def test_collect_run_info_end_to_end(tmp_path):
    run = tmp_path / "run205"
    run.mkdir()
    (run / "launch.sh").write_text(
        "#!/bin/bash\n.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \\\n"
        "  --out-dir " + str(run) + " --epochs 1500 --eval-every 25 --ckpt-every 100 --curriculum \\\n"
        "  --tau-softplus-start-epoch 300 --l7-start-epoch 800 --muon-start-epoch 1100 \\\n"
        "  --activation hosc --mod-dim 32 --hidden-dim 96 --w-seg 100.0 --w-pose 0.0 --seed 0 --num-pairs 600\n")
    log = run / "run.log"
    _write_run_log(log, extra_lines=[_PERF_ACTIVE, _CKPT_STAGE, _CKPT_INTRA, _CKPT_BEST],
                   epochs=(0, 25, 50))
    (run / "levelset_best.json").write_text(
        '{"d_seg": 0.00412, "epoch": 675, "path": "levelset_witness_ema_BEST.npz", "ts": "x"}')
    (run / "levelset_resume_state.npz").write_bytes(b"\0" * 4096)
    (run / "levelset_witness_ema_mlx.npz").write_bytes(b"\0" * 2048)
    live = {"cadence_s": 2612.5, "cadence_source": "measured", "last_epoch": 50}
    rows = [{"epoch": 0, "seg_form": "ce"}, {"epoch": 25, "seg_form": "ce"},
            {"epoch": 50, "seg_form": "ce", "ts": "2026-07-01T00:00:00Z"}]
    info = _collect_run_info(log, str(run / "*.log"), str(run), rows, live, now=time.time())
    assert info["config"]["source"] == "launch.sh"
    assert info["schedule"]["tau_start"] == 300 and info["schedule"]["__ckpt_every"] == 100
    assert info["stage_prog"]["stage"] == "CE"
    assert info["best"]["d_seg"] == 0.00412
    assert info["perf"]["active"] is True
    assert info["resume"]["resumable"] is True
    assert info["disk"]["count"] == 2
    assert abs(info["throughput"]["s_per_epoch"] - 104.5) < 1e-6
    # and the whole thing renders into HTML without error
    html = _run_info_html(info)
    assert "CE" in html and "✓ active" in html


def test_collect_run_info_never_crashes_on_empty(tmp_path):
    # no launch.sh, no artifacts, no rows -> graceful empty-ish info (no exception)
    info = _collect_run_info(None, None, None, [], {"last_epoch": None}, now=time.time())
    assert info["stage_prog"]["stage"] is None
    assert info["best"] is None
    assert _run_info_html(info) != ""  # still renders the (unknown) cards


# ---------------------------------------------------------------------------
# _write_html WITH run_info (the #205 strip) + back-compat WITHOUT it
# ---------------------------------------------------------------------------
def test_write_html_with_run_info_strip(tmp_path):
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 550, "d_seg": 0.0044, "d_pose": 0.0009, "blob_bytes": 122000,
             "implied_S": 0.53, "seg_form": "tau_softplus"}]
    live = {"kind": "live", "calibrating": False, "last_epoch": 550, "verdict_age_s": 30.0,
            "log_age_s": 30.0, "cadence_s": 2612.5, "cadence_source": "measured",
            "next_epoch": 575, "next_eta_s": 900.0, "threshold_s": 5225.0}
    run_info = {
        "schedule": {"epochs": 1500, "eval_every": 25, "tau_start": 300, "l7_start": 800,
                     "muon_start": 1100, "__ckpt_every": 100},
        "stage_prog": {"stage": "tau", "stage_pct": 0.5, "total_pct": 0.37, "epochs": 1500,
                       "next_name": "l7", "next_epoch": 800, "muon_active": False,
                       "__last_epoch": 550},
        "best": {"d_seg": 0.00412, "epoch": 540, "path": "levelset_witness_ema_BEST.npz"},
        "checkpoints": {"counts": {"intra_stage": 5}, "preserved": [{"epoch": 300}]},
        "resume": {"resumable": True, "resume_epoch": 500, "resume_npz": True,
                   "resume_age_s": 60.0, "deploy_npz": True, "deploy_age_s": 60.0},
        "throughput": {"s_per_epoch": 104.5, "epochs_per_hour": 34.4, "elapsed_s": 3600 * 26,
                       "eta_next_boundary_s": 26125.0, "eta_total_s": 99275.0},
        "perf": {"active": True}, "disk": {"count": 4, "total_bytes": 500 * 1024},
        "config": {"source": "launch.sh", "flags": {"w-seg": "100.0", "activation": "hosc"}},
    }
    _write_html(out, b"PNG", rows, 30, watched=Path("run.log"), live=live,
                tau=300, l7=800, goal_dseg=0.00112, run_info=run_info)
    html = out.read_text()
    assert "grid" in html and "card" in html          # the strip rendered
    assert "curriculum stage" in html and "0.00412" in html
    assert "MLX fast path" in html and "✓ active" in html
    assert "Level-Set Witness" in html                # existing chrome preserved
    assert "pointer 0.19110" in html


def test_write_html_without_run_info_is_unchanged(tmp_path):
    # back-compat: no run_info -> NO grid section (existing single-page layout intact)
    out = tmp_path / "dash" / "index.html"
    rows = [{"epoch": 5, "d_seg": 0.004, "d_pose": 0.0009, "blob_bytes": 1, "implied_S": 0.21}]
    _write_html(out, b"PNG", rows, 30, watched=Path("run.log"))
    html = out.read_text()
    assert 'class="grid"' not in html   # no strip when run_info omitted
    assert "Level-Set Witness" in html  # page still renders


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
