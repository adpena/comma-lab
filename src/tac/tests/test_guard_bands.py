# SPDX-License-Identifier: MIT
"""Tests for the GRADUATED GUARD BANDS (BUILD #294 piece C) in tools/system_memory_governor.py.

Guards: band classification thresholds (green <85% / yellow 85-90% / red >=90% of the envelope);
synthetic RSS traces walking the bands; NEVER-KILL invariants (the action vocabulary is closed and
contains no kill; control-plane / non-eligible jobs are never actuated); defer-to-throttle when the
system pressure ladder is engaged (backstop BEHIND the throttle, never a replacement); the #246
efficacy-gap REPORT (measured pause-scope vs run RSS in the decision reason); and the tracked-units
-> true-GiB conversion (#205 memory mine §1)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import system_memory_governor as gov  # noqa: E402

ENV = gov.band_envelope_gib(128.0)   # 0.85 * 128 = 108.8


def _job(*, rss=50.0, paused=False, eligible=True, leader=True, pid=100, pgid=100,
         label="levelset_witness_run", priority=60):
    return gov.TrackedJob(label=label, pid=pid, pgid=pgid, cmd="python train_levelset_witness.py",
                          priority=priority, projected_peak_gib=70.0, current_rss_gib=rss,
                          paused=paused, throttle_eligible=eligible, own_group_leader=leader)


# ── classification ──────────────────────────────────────────────────────────────────────────────
def test_envelope_is_085_of_total():
    assert abs(ENV - 108.8) < 1e-9


def test_classify_green_below_85pct():
    assert gov.classify_guard_band(0.0, ENV) == gov.BAND_GREEN
    assert gov.classify_guard_band(0.849 * ENV, ENV) == gov.BAND_GREEN


def test_classify_yellow_between_85_and_90pct():
    assert gov.classify_guard_band(0.85 * ENV, ENV) == gov.BAND_YELLOW
    assert gov.classify_guard_band(0.899 * ENV, ENV) == gov.BAND_YELLOW


def test_classify_red_at_and_above_90pct():
    assert gov.classify_guard_band(0.90 * ENV, ENV) == gov.BAND_RED
    assert gov.classify_guard_band(2.0 * ENV, ENV) == gov.BAND_RED


def test_classify_degenerate_envelope_is_red():
    assert gov.classify_guard_band(1.0, 0.0) == gov.BAND_RED


def test_band_thresholds_in_true_gib_on_128_box():
    # yellow at 92.48 true GiB, red at 97.92 true GiB
    assert gov.classify_guard_band(92.0, ENV) == gov.BAND_GREEN
    assert gov.classify_guard_band(93.0, ENV) == gov.BAND_YELLOW
    assert gov.classify_guard_band(98.0, ENV) == gov.BAND_RED


# ── synthetic RSS trace walking the bands ───────────────────────────────────────────────────────
def test_synthetic_trace_walks_green_yellow_red():
    trace = [50.0, 80.0, 92.0, 93.5, 96.0, 98.5, 105.0]
    bands = [gov.classify_guard_band(r, ENV) for r in trace]
    assert bands == ["green", "green", "green", "yellow", "yellow", "red", "red"]


def test_synthetic_trace_decisions_never_kill():
    """Walk a rising-then-falling trace through the full decision fn: NO kill-class action ever."""
    job = _job()
    trace = [40.0, 70.0, 93.0, 99.0, 110.0, 99.0, 93.0, 50.0]
    for rss in trace:
        band = gov.classify_guard_band(rss, ENV)
        d = gov.decide_guard_band_action(band=band, job=job, pressure_class="normal",
                                         actual_rss_gib=rss, envelope_gib=ENV,
                                         checkpoint_age_s=60.0)
        assert d.action in gov.GUARD_BAND_ACTIONS
        assert "kill" not in d.action
        # the only permitted mention of SIGKILL is the negation ("NEVER SIGKILL")
        assert "sigkill" not in d.reason.lower().replace("never sigkill", "")


def test_action_vocabulary_is_closed_and_kill_free():
    assert set(gov.GUARD_BAND_ACTIONS) == {"none", "alert_prep", "already_paused",
                                           "defer_to_throttle", "alert_no_target", "clean_pause"}
    assert not any("kill" in a for a in gov.GUARD_BAND_ACTIONS)


# ── per-band decisions ──────────────────────────────────────────────────────────────────────────
def _decide(band, *, job=None, pressure="normal", rss=100.0, ck=60.0, scope=None):
    return gov.decide_guard_band_action(band=band, job=job, pressure_class=pressure,
                                        actual_rss_gib=rss, envelope_gib=ENV,
                                        checkpoint_age_s=ck, pause_scope_rss_gib=scope)


def test_green_is_none():
    d = _decide(gov.BAND_GREEN, job=_job(), rss=50.0)
    assert d.action == "none"


def test_yellow_is_alert_prep_with_checkpoint_freshness():
    d = _decide(gov.BAND_YELLOW, job=_job(), rss=94.0, ck=120.0)
    assert d.action == "alert_prep"
    assert d.checkpoint_fresh is True
    assert "FRESH" in d.reason


def test_yellow_flags_stale_checkpoint():
    d = _decide(gov.BAND_YELLOW, job=_job(), rss=94.0, ck=7200.0)
    assert d.action == "alert_prep"
    assert d.checkpoint_fresh is False
    assert "STALE" in d.reason


def test_red_clean_pause_when_eligible_and_pressure_normal():
    d = _decide(gov.BAND_RED, job=_job(), rss=100.0)
    assert d.action == "clean_pause"
    assert "NEVER SIGKILL" in d.reason


def test_red_defers_to_throttle_under_warn_pressure():
    """Backstop BEHIND the throttle: when the pressure ladder is engaged, the throttle owns
    actuation and the band only alerts."""
    for pressure in ("warn", "critical"):
        d = _decide(gov.BAND_RED, job=_job(), pressure=pressure, rss=100.0)
        assert d.action == "defer_to_throttle"
        assert "throttle" in d.reason


def test_red_never_targets_non_eligible_control_plane_protected_job():
    d = _decide(gov.BAND_RED, job=_job(eligible=False), rss=100.0)
    assert d.action == "alert_no_target"
    assert "NOT throttle-eligible" in d.reason


def test_red_no_job_alerts_only():
    d = _decide(gov.BAND_RED, job=None, rss=100.0)
    assert d.action == "alert_no_target"


def test_red_already_paused_no_double_actuation():
    d = _decide(gov.BAND_RED, job=_job(paused=True), rss=100.0)
    assert d.action == "already_paused"


def test_red_reports_246_efficacy_gap_when_pause_scope_is_small():
    """The pause SIGSTOPs the wrapper's pgid; on the current launch tree the memory-bearing
    trainer lives in a child group (governor-review F1 / task #246) — REPORTED, not fixed."""
    d = _decide(gov.BAND_RED, job=_job(), rss=100.0, scope=0.02)
    assert d.action == "clean_pause"
    assert "#246 EFFICACY GAP" in d.reason
    assert d.efficacy_gap_gib is not None and d.efficacy_gap_gib > 99.0


def test_red_no_gap_note_when_pause_scope_covers_the_run():
    d = _decide(gov.BAND_RED, job=_job(), rss=100.0, scope=99.0)
    assert d.action == "clean_pause"
    assert "#246" not in d.reason


# ── units + run selection ───────────────────────────────────────────────────────────────────────
def test_tracked_units_conversion_constant_matches_mine():
    # mine §1: 65.27 units == 62.25 true GiB (ps cross-validated)
    assert abs(gov.TRACKED_RSS_UNITS_TO_GIB - 0.95367431640625) < 1e-12
    assert abs(65.27 * gov.TRACKED_RSS_UNITS_TO_GIB - 62.25) < 0.02


def test_select_band_run_picks_largest_own_leader_only():
    """The wrapper custody group (own leader, descendant-inclusive RSS) is the run; the trainer
    child row (not own leader) is skipped — no double count (mine §7 / F2)."""
    wrapper = _job(rss=65.0, pid=29126, pgid=29126, leader=True)
    child = _job(rss=64.9, pid=29129, pgid=29127, leader=False, eligible=False, label="pid29129")
    small = _job(rss=0.04, pid=19895, pgid=19895, leader=True, label="memory_blackbox")
    assert gov.select_band_run([wrapper, child, small]) == wrapper


def test_select_band_run_none_when_no_leaders():
    child = _job(leader=False)
    assert gov.select_band_run([child]) is None


# ── pause actuator is the reversible SIGSTOP path (structural) ──────────────────────────────────
def test_clean_pause_actuator_is_pause_job_sigstop(monkeypatch):
    """band_tick's apply path routes through the EXISTING reversible pause_job (SIGSTOP) —
    structurally no other signal is reachable from the band code."""
    import inspect

    src = inspect.getsource(gov.band_tick)
    assert "pause_job" in src
    assert "SIGKILL" not in src and "SIGTERM" not in src
    band_src = inspect.getsource(gov.decide_guard_band_action)
    assert "SIGKILL" not in band_src.replace("NEVER SIGKILL", "")


def test_band_row_append_is_lock_guarded(tmp_path):
    ledger = tmp_path / "bands.jsonl"
    gov.append_band_row({"kind": "guard_band", "x": 1}, ledger)
    gov.append_band_row({"kind": "guard_band", "x": 2}, ledger)
    import json

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["x"] for r in rows] == [1, 2]
    assert all("ts" in r for r in rows)


def test_checkpoint_age_seconds(tmp_path):
    assert gov.checkpoint_age_seconds(tmp_path) is None
    (tmp_path / "levelset_resume_state.npz").write_bytes(b"x")
    age = gov.checkpoint_age_seconds(tmp_path)
    assert age is not None and 0.0 <= age < 60.0
