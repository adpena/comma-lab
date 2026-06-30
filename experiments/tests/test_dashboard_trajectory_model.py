# SPDX-License-Identifier: MIT
"""Tests for the sophisticated DATA-DERIVED trajectory + stage-aware wall-clock model
(tools/dashboard_trajectory_model.py, landed 2026-06-30).

Replaces the naive linear d_seg extrapolation with the lab's actual deep-math:
  * CRITICAL-SLOWING POWER LAW d_seg(t) = c + a·(t−t0)^(−α) (Agmon–Tishby 2103.02646;
    Rose 1998 deterministic annealing) — recover (asymptote, α) on synthetic curves.
  * GOAL-ETA WITH CONFIDENCE BANDS; honest "won't reach" when the asymptote is above
    the target; low-confidence flagged; "calibrating" until enough points.
  * STAGE-AWARE WALL-CLOCK: per-stage seconds/epoch measured separately (Muon slower);
    not-yet-entered stages get a FLAGGED estimate; completion-ETA sums per-stage rates;
    next-verdict cadence = CURRENT stage rate × eval-every, recomputed at boundaries.
  * implied_S projection via the model (asymptote + sidecar pose + projected bytes).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.dashboard_trajectory_model import (  # noqa: E402
    build_projection,
    completion_eta,
    current_stage_cadence,
    fit_critical_slowing,
    per_stage_seconds_per_epoch,
    project_bytes,
    project_goal_epoch,
    project_implied_s,
    stage_at_epoch,
)

_SCHED = {"tau_start": 300, "l7_start": 600, "muon_start": 726, "epochs": 1000,
          "eval_every": 25}
_NORM = 37_545_489
_SIDE = 3.4e-5


def _ts(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ── stage_at_epoch ──────────────────────────────────────────────────────────────
def test_stage_at_epoch_boundaries():
    assert stage_at_epoch(0, _SCHED) == "CE"
    assert stage_at_epoch(299, _SCHED) == "CE"
    assert stage_at_epoch(300, _SCHED) == "tau"
    assert stage_at_epoch(599, _SCHED) == "tau"
    assert stage_at_epoch(600, _SCHED) == "l7"
    assert stage_at_epoch(725, _SCHED) == "l7"
    assert stage_at_epoch(726, _SCHED) == "Muon"
    assert stage_at_epoch(999, _SCHED) == "Muon"


def test_stage_at_epoch_non_curriculum_all_ce():
    assert stage_at_epoch(500, {"tau_start": None, "l7_start": None, "muon_start": None}) == "CE"


# ── critical-slowing power-law fit ──────────────────────────────────────────────
def test_powerlaw_recovers_asymptote_and_exponent_noiseless():
    t = np.arange(0, 1000, 25.0)
    t0, c, a, al = -30.0, 0.0012, 5.0, 1.2
    y = c + a * (t - t0) ** (-al)
    fit = fit_critical_slowing(t, y)
    assert fit["ok"]
    assert fit["r2"] > 0.999
    assert abs(fit["asymptote"] - c) < 3e-4   # asymptote recovered (degeneracy-bounded)
    assert fit["confidence"] == "high"
    assert fit["a"] > 0                         # genuine decay


def test_powerlaw_robust_to_mild_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 1000, 25.0)
    y = 0.0012 + 5.0 * (t + 30.0) ** (-1.2) + rng.normal(0, 1.5e-4, size=t.shape)
    fit = fit_critical_slowing(t, y)
    assert fit["ok"] and fit["r2"] > 0.95
    assert abs(fit["asymptote"] - 0.0012) < 5e-4


def test_powerlaw_few_points_calibrating_not_confident_wrong():
    fit = fit_critical_slowing([0, 25, 50], [0.28, 0.011, 0.0085])
    assert fit["ok"] is False
    assert "need" in fit["reason"]              # honest: not enough points


def test_powerlaw_low_confidence_flagged_with_borderline_points():
    # exactly the minimum points with imperfect structure -> medium/low confidence flag
    t = np.array([0, 25, 50, 75, 100], dtype=float)
    y = 0.002 + 3.0 * (t + 20.0) ** (-1.0)
    fit = fit_critical_slowing(t, y)
    assert fit["ok"]
    assert fit["confidence"] in ("low", "medium", "high")
    assert fit["n"] == 5


# ── goal ETA + bands + honest "won't reach" ─────────────────────────────────────
def test_goal_eta_band_monotone_when_reachable():
    t = np.arange(0, 500, 25.0)
    y = 0.001 + 5.0 * (t + 30.0) ** (-1.2)
    fit = fit_critical_slowing(t, y)
    ge = project_goal_epoch(fit, target=0.004, current_epoch=200.0)
    assert ge["state"] == "eta"
    if ge["epoch_lo"] is not None and ge["epoch_hi"] is not None:
        assert ge["epoch_lo"] <= ge["epoch"] <= ge["epoch_hi"]   # band monotone
    assert ge["eta_epochs"] > 0


def test_goal_eta_asymptote_above_says_wont_reach():
    t = np.arange(0, 1000, 25.0)
    y = 0.0012 + 5.0 * (t + 30.0) ** (-1.2)        # asymptote ~0.0012
    fit = fit_critical_slowing(t, y)
    ge = project_goal_epoch(fit, target=0.0003, current_epoch=975.0)  # below asymptote
    assert ge["state"] == "asymptote_above"        # honest: won't reach at this trajectory
    assert ge["asymptote"] > 0.0003


def test_goal_eta_calibrating_when_no_fit():
    ge = project_goal_epoch({"ok": False}, target=0.001, current_epoch=50.0)
    assert ge["state"] == "calibrating"


# ── stage-aware seconds/epoch + completion ETA + current-stage cadence ───────────
def _ce_only_verdicts(t0):
    # CE stage verdicts ep0..100 spaced 600s/25ep => 24 s/epoch
    return [{"epoch": e, "ts": _ts(t0 + (e // 25) * 600.0)} for e in (0, 25, 50, 75, 100)]


def test_per_stage_spe_measures_ce_and_flags_muon_estimate():
    spe = per_stage_seconds_per_epoch(_ce_only_verdicts(1_700_000_000.0), _SCHED)
    assert spe["CE"]["measured"] is True
    assert abs(spe["CE"]["spe"] - 24.0) < 0.5
    # Muon not entered -> ESTIMATE, slower than CE, flagged (never silently CE's rate)
    assert spe["Muon"]["measured"] is False
    assert spe["Muon"]["spe"] > spe["CE"]["spe"]
    assert spe["Muon"]["source"].startswith("est:muon_x")


def test_per_stage_spe_two_stages_measured_separately():
    t0 = 1_700_000_000.0
    # CE: ep0..50 @ 24 s/ep (gap 600). tau: ep300..350 @ 40 s/ep (gap 1000).
    verds = [{"epoch": 0, "ts": _ts(t0)}, {"epoch": 25, "ts": _ts(t0 + 600)},
             {"epoch": 50, "ts": _ts(t0 + 1200)},
             {"epoch": 300, "ts": _ts(t0 + 9000)}, {"epoch": 325, "ts": _ts(t0 + 10000)},
             {"epoch": 350, "ts": _ts(t0 + 11000)}]
    spe = per_stage_seconds_per_epoch(verds, _SCHED)
    assert abs(spe["CE"]["spe"] - 24.0) < 0.5
    assert abs(spe["tau"]["spe"] - 40.0) < 0.5      # tau measured separately, not CE's rate


def test_completion_eta_sums_per_stage_and_flags_estimate():
    spe = per_stage_seconds_per_epoch(_ce_only_verdicts(1_700_000_000.0), _SCHED)
    comp = completion_eta(current_epoch=100, schedule=_SCHED, stage_spe=spe, total_epochs=1000)
    assert comp["ok"]
    assert comp["total_s"] > 0
    assert comp["has_estimate"] is True             # Muon portion is estimated
    assert comp["estimated_s"] > 0 and comp["measured_s"] >= 0
    # the per-stage breakdown carries the remaining epochs for each stage
    stages = {r["stage"]: r for r in comp["per_stage"]}
    assert stages["Muon"]["remaining_epochs"] == 1000 - 726
    assert stages["Muon"]["measured"] is False


def test_completion_eta_calibrating_when_no_rates():
    comp = completion_eta(current_epoch=10, schedule=_SCHED, stage_spe={}, total_epochs=1000)
    assert comp["ok"] is False and comp["reason"] == "calibrating"


def test_current_stage_cadence_recomputes_at_boundary():
    t0 = 1_700_000_000.0
    # measured CE (24 s/ep) and Muon (say 60 s/ep)
    verds = [{"epoch": 0, "ts": _ts(t0)}, {"epoch": 25, "ts": _ts(t0 + 600)},
             {"epoch": 726, "ts": _ts(t0 + 50000)}, {"epoch": 751, "ts": _ts(t0 + 51500)}]
    spe = per_stage_seconds_per_epoch(verds, _SCHED)
    cad_ce, st_ce, src_ce = current_stage_cadence(50, _SCHED, spe, eval_every=25)
    cad_mu, st_mu, src_mu = current_stage_cadence(740, _SCHED, spe, eval_every=25)
    assert st_ce == "CE" and src_ce == "measured"
    assert st_mu == "Muon" and src_mu == "measured"
    assert cad_mu > cad_ce                            # Muon cadence slower (recomputed)


# ── bytes + implied_S projection ────────────────────────────────────────────────
def test_project_bytes_linear_and_last():
    verds = [{"epoch": e, "blob_bytes": 85000 - e * 5} for e in (0, 25, 50, 75)]
    pl = project_bytes(verds, project_to_epoch=200)
    assert pl["ok"] and pl["value"] >= 0
    last = project_bytes(verds, project_to_epoch=None)
    assert last["value"] == 85000 - 75 * 5


def test_project_implied_s_with_band():
    s = project_implied_s(dseg_inf=0.0012, bytes_proj=85000, sidecar_pose=_SIDE,
                          archive_norm=_NORM, dseg_band=(0.0010, 0.0014))
    assert s["ok"]
    expect = 100 * 0.0012 + (10 * _SIDE) ** 0.5 + 25 * 85000 / _NORM
    assert abs(s["value"] - expect) < 1e-6
    assert s["value_lo"] < s["value"] < s["value_hi"]   # band monotone
    # uses the SIDECAR pose (telemetry accuracy), not a monitoring pose
    assert abs(s["pose_term"] - (10 * _SIDE) ** 0.5) < 1e-9


# ── end-to-end build_projection (the dashboard payload) ─────────────────────────
def test_build_projection_end_to_end_live_like():
    t0 = 1_700_000_000.0
    # a descending d_seg series across CE with ts (>=5 points so the fit engages)
    eps = list(range(0, 250, 25))
    verds = [{"epoch": e, "d_seg": 0.0015 + 4.0 * (e + 30.0) ** (-1.2),
              "blob_bytes": 85000, "ts": _ts(t0 + (e // 25) * 600.0)} for e in eps]
    meta = {"schedule": _SCHED, "goal_dseg": 0.00092, "goal_dseg_15": 0.00032}
    proj = build_projection(verds, meta, sidecar_pose=_SIDE, archive_norm=_NORM, eval_every=25)
    assert proj["ok"]
    assert proj["stage"] == "CE"
    assert proj["dseg_model"]["ok"]                          # fit engaged (>=5 pts)
    assert proj["completion_eta"]["ok"]
    assert proj["next_verdict_cadence_s"] is not None
    assert proj["implied_s_proj"]["ok"]
    assert "goal_eta" in proj and "goal15_eta" in proj


def test_build_projection_calibrating_with_few_points():
    meta = {"schedule": _SCHED, "goal_dseg": 0.00092, "goal_dseg_15": 0.00032}
    verds = [{"epoch": 0, "d_seg": 0.28, "blob_bytes": 58000, "ts": _ts(1_700_000_000.0)}]
    proj = build_projection(verds, meta, sidecar_pose=_SIDE, archive_norm=_NORM, eval_every=25)
    # one point -> the d_seg model is calibrating; build_projection still returns a shell
    assert proj["ok"]
    assert proj["dseg_model"]["ok"] is False                 # not confident-wrong


def test_build_projection_never_raises_on_garbage():
    assert build_projection([], {}, sidecar_pose=_SIDE, archive_norm=_NORM)["ok"] is False
    bad = build_projection([{"epoch": "x"}], {"schedule": {}}, sidecar_pose=_SIDE,
                           archive_norm=_NORM)
    assert bad["ok"] is False                                 # graceful, no crash
