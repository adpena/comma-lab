# SPDX-License-Identifier: MIT
"""Telemetry-accuracy hardcode-fix tests for the level-set dashboard cadence/ETA math
(tools/render_levelset_dashboard.py, fix landed 2026-06-30).

Extincts the "hardcoded garbage" bug class (operator 2026-06-30): the cadence/liveness
used a HARDCODED 18-min prior (and a 10-min floor) that false-flagged the live n600 run
as stale/slow and showed wrong ETAs, because the real n600 inter-verdict cadence is
~43 min (104.5 s/epoch × eval-every-25 ≈ 2613 s). The fix derives ALL time-telemetry
from the run's OWN measured data and says "calibrating" when there isn't enough yet.

Covers the binding guarantees:
  * cadence is MEASURED from >=1 inter-verdict gap (>=2 verdicts) — never an 18m prior.
  * stale = K × MEASURED cadence (+ async grace), DOUBLE-GATED on log activity — not a
    hardcoded minute floor.
  * "calibrating" (no number) before the first gap; never a false stale.
  * an in-flight async verdict (measured verdict_async_done grace) is not flagged stale.
  * n200 AND n600 each measure their OWN cadence (no hardcoded 18m for either).
  * next-verdict ETA = last_verdict + measured cadence (honest countdown).
  * the retired _CADENCE_PRIOR_MIN / _STALE_FLOOR_MIN are inert (no time value drives a
    judgment); bootstrap estimate (eval_every × spe) is labeled, not a constant.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.render_levelset_dashboard import (  # noqa: E402
    _ASYNC_GRACE_FALLBACK_S,
    _CADENCE_PRIOR_MIN,
    _MIN_CADENCE_GAPS,
    _STALE_FLOOR_MIN,
    _async_grace_s,
    _cfg_from_args,
    _compute_liveness,
    _measure_cadence,
)


def _ts(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _ts_rows(epochs, t0, gap_s, dseg=0.01):
    """verdict rows with embedded ts spaced gap_s apart (the durable trainer format)."""
    return [{"epoch": e, "d_seg": dseg, "ts": _ts(t0 + i * gap_s)}
            for i, e in enumerate(epochs)]


def _args(**kw):
    base = dict(stale_min=None, stale_floor_min=_STALE_FLOOR_MIN, cadence_k=2.0)
    base.update(kw)
    return argparse.Namespace(**base)


# ── the retired hardcoded constants are INERT ──────────────────────────────────
def test_retired_prior_constant_is_inert():
    # the 18-min "hardcoded garbage" is gone: the constant exists only for back-compat
    # and carries NO time value (0), so it can never drive a cadence/threshold again.
    assert _CADENCE_PRIOR_MIN == 0.0
    assert _STALE_FLOOR_MIN == 0.0           # no hardcoded floor by default
    assert _MIN_CADENCE_GAPS == 1            # >=1 gap => measured


def test_cfg_has_no_hardcoded_prior_and_no_default_floor():
    cfg = _cfg_from_args(_args())
    assert "prior_s" not in cfg              # no fabricated cadence prior
    assert cfg["floor_s"] == 0.0             # no hardcoded minute floor by default
    assert cfg["min_gaps"] == 1
    assert cfg["k"] == 2.0


# ── cadence is MEASURED from >=1 gap (never an 18m prior) ───────────────────────
def test_cadence_measured_from_one_gap_n600():
    # n600: 2 verdicts, one ~43-min gap -> measured 2613s, NOT the retired 18m prior.
    t0 = 1_700_000_000.0
    rows = _ts_rows((0, 25), t0, gap_s=2613.0)
    now = t0 + 2613.0 + 600.0
    live = _compute_liveness(rows, watched_mtime=now - 5, now=now, sub={},
                             cfg=_cfg_from_args(_args()), async_grace_s=240.0)
    assert live["cadence_source"] == "measured"
    assert abs(live["cadence_s"] - 2613.0) < 1.0     # the REAL n600 cadence
    assert abs(live["cadence_s"] - 18 * 60.0) > 60.0  # NOT 18 minutes
    assert live["kind"] == "live"
    assert live["calibrating"] is False


def test_n200_and_n600_each_measure_their_own_cadence_no_hardcoded_18():
    # generality: two runs with DIFFERENT real cadences each report their OWN, neither
    # falling back to a shared hardcoded 18-min constant.
    t0 = 1_700_000_000.0
    n200 = _ts_rows((0, 25, 50), t0, gap_s=1500.0)   # 25 min cadence
    n600 = _ts_rows((0, 25, 50), t0, gap_s=2613.0)   # 43.5 min cadence
    cfg = _cfg_from_args(_args())
    l200 = _compute_liveness(n200, t0 + 3000 + 100, t0 + 3000 + 105, {}, cfg, async_grace_s=200.0)
    l600 = _compute_liveness(n600, t0 + 5226 + 100, t0 + 5226 + 105, {}, cfg, async_grace_s=200.0)
    assert abs(l200["cadence_s"] - 1500.0) < 1.0
    assert abs(l600["cadence_s"] - 2613.0) < 1.0
    assert l200["cadence_s"] != l600["cadence_s"]     # data-derived, not a shared constant


# ── "calibrating" before any gap (no fabricated number, no false stale) ─────────
def test_calibrating_before_first_gap_no_number_no_stale():
    t0 = 1_700_000_000.0
    rows = _ts_rows((0,), t0, gap_s=0.0)              # a single verdict -> no gap yet
    # even with the log quiet for an hour, calibrating never time-stales (honest)
    live = _compute_liveness(rows, watched_mtime=t0 + 9999, now=t0 + 9999 + 3600,
                             sub={}, cfg=_cfg_from_args(_args()), async_grace_s=240.0)
    assert live["cadence_s"] is None
    assert live["cadence_source"] == "calibrating"
    assert live["threshold_s"] is None               # no fabricated stale threshold
    assert live["kind"] == "live"


def test_measure_cadence_one_gap_is_measured():
    # the unit guarantee: a single positive gap is enough to be "measured".
    arrivals = [(0, 100.0, "ts"), (25, 100.0 + 2613.0, "ts")]
    cad, src, n = _measure_cadence(arrivals, baseline=set(), cfg={"min_gaps": 1})
    assert src == "measured" and abs(cad - 2613.0) < 1e-6 and n == 2


# ── stale = K × MEASURED cadence, double-gated; async-in-flight not stale ────────
def test_stale_threshold_is_k_times_measured_cadence_not_hardcoded():
    t0 = 1_700_000_000.0
    rows = _ts_rows((0, 25, 50), t0, gap_s=2613.0)    # measured cadence 2613s
    now = t0 + 5226.0 + 10.0
    live = _compute_liveness(rows, watched_mtime=now - 5, now=now, sub={},
                             cfg=_cfg_from_args(_args()), async_grace_s=240.0)
    # threshold = K(2.0) × cadence + async grace; purely data-derived
    assert abs(live["threshold_s"] - (2.0 * 2613.0 + 240.0)) < 1e-6
    assert abs(live["log_threshold_s"] - (2613.0 + 240.0)) < 1e-6


def test_async_in_flight_not_flagged_stale():
    # verdict overdue past K×cadence, but the log is freshly written (async eval running
    # / checkpoints) -> NOT hung -> NOT stale. The measured async grace widens the bound.
    t0 = 1_700_000_000.0
    rows = _ts_rows((0, 25, 50), t0, gap_s=600.0)     # measured cadence 600s
    now = t0 + 1200.0 + 1700.0                          # verdict 1700s old (> 2×600+grace? grace 240 -> thr 1440)
    live = _compute_liveness(rows, watched_mtime=now - 15, now=now, sub={},
                             cfg=_cfg_from_args(_args()), async_grace_s=240.0)
    assert live["kind"] == "live"                       # log fresh -> alive (async/ckpt writing)


def test_genuinely_hung_is_stale_when_log_also_quiet():
    t0 = 1_700_000_000.0
    rows = _ts_rows((0, 25, 50), t0, gap_s=600.0)
    now = t0 + 1200.0 + 2000.0                          # verdict 2000s old
    live = _compute_liveness(rows, watched_mtime=now - 2000, now=now, sub={},  # log ALSO 2000s quiet
                             cfg=_cfg_from_args(_args()), async_grace_s=240.0)
    assert live["kind"] == "stale"
    assert live["verdict_age_s"] > live["threshold_s"]
    assert live["log_age_s"] > live["log_threshold_s"]


# ── async grace is MEASURED from the log (not a constant) ───────────────────────
def test_async_grace_measured_from_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        '{"stage": "verdict_async_done", "epoch": 25, "secs": 240.1}\n'
        '{"stage": "verdict_async_done", "epoch": 50, "secs": 203.6}\n'
        '{"stage": "verdict", "epoch": 50, "d_seg": 0.008}\n')
    assert abs(_async_grace_s(log) - 240.1) < 1e-6     # MAX observed async secs


def test_async_grace_zero_when_absent(tmp_path):
    log = tmp_path / "run.log"
    log.write_text('{"stage": "verdict", "epoch": 0, "d_seg": 0.2}\n')
    assert _async_grace_s(log) == _ASYNC_GRACE_FALLBACK_S == 0.0
    assert _async_grace_s(None) == 0.0                  # never raises on missing log


# ── next-verdict ETA = last_verdict + measured cadence (honest countdown) ────────
def test_next_eta_is_last_plus_cadence_countdown():
    t0 = 1_700_000_000.0
    rows = _ts_rows((0, 25), t0, gap_s=2613.0)
    now = t0 + 2613.0 + 600.0                            # 10 min after the 2nd verdict
    live = _compute_liveness(rows, watched_mtime=now - 5, now=now, sub={},
                             cfg=_cfg_from_args(_args()), async_grace_s=0.0)
    # ETA = cadence - verdict_age = 2613 - 600 = 2013s (honest ~33 min, not "now")
    assert abs(live["next_eta_s"] - (2613.0 - 600.0)) < 2.0
    assert live["next_epoch"] == 50                      # last(25) + spacing(25)


# ── bootstrap ESTIMATE path (eval_every × spe) is labeled, not a hardcoded prior ──
def test_bootstrap_estimate_labeled_when_spe_available_else_calibrating():
    t0 = 1_700_000_000.0
    one = _ts_rows((0,), t0, gap_s=0.0)                 # 1 verdict, no measurable gap
    # with an independent seconds/epoch (e.g. a per-epoch heartbeat) -> labeled estimate
    cfg_est = _cfg_from_args(_args(seconds_per_epoch=104.5, eval_every=25))
    live = _compute_liveness(one, watched_mtime=t0 + 100, now=t0 + 105, sub={},
                             cfg=cfg_est, async_grace_s=0.0)
    assert live["cadence_source"] == "estimate"
    assert abs(live["cadence_s"] - 104.5 * 25) < 1e-6
    assert live["calibrating"] is True                  # estimate is still 'not measured'
    # without any spe -> honest calibrating, no number
    cfg_plain = _cfg_from_args(_args())
    live2 = _compute_liveness(one, watched_mtime=t0 + 100, now=t0 + 105, sub={},
                              cfg=cfg_plain, async_grace_s=0.0)
    assert live2["cadence_source"] == "calibrating" and live2["cadence_s"] is None


# ── stage-aware PREFERRED cadence override (current stage rate wins) ─────────────
def test_preferred_stage_cadence_overrides_gap_median():
    # server feeds the CURRENT stage's measured rate; it wins over the gap median so a
    # slow (Muon) stage is not judged against a fast (CE) cadence.
    t0 = 1_700_000_000.0
    rows = _ts_rows((600, 625, 650), t0, gap_s=600.0)   # gap-median would be 600s
    cfg = _cfg_from_args(_args(preferred_cadence_s=1800.0, preferred_cadence_source="measured"))
    now = t0 + 1200.0 + 100.0
    live = _compute_liveness(rows, watched_mtime=now - 5, now=now, sub={}, cfg=cfg,
                             async_grace_s=0.0)
    assert abs(live["cadence_s"] - 1800.0) < 1e-6       # stage-aware rate, not 600
    assert live["cadence_source"] == "measured"
