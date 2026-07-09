"""Tests for the VERDICT-TREND / TRAIN-VERDICT-DECOUPLING alarm (operator-catch 2026-07-09).

Acceptance backtest (the reason this module exists): the alarm MUST fire DECOUPLING on
the LIVE run's actual verdict rows at ep100 (where the shadow controller wrongly said
CONVERGING) and MUST stay SILENT on the mod32cap descending baseline. Both are recorded
as fixtures below, transcribed verbatim from the real run.logs.
"""
from __future__ import annotations

import json

import pytest

from tac.witness_control import (
    NO_ALARM,
    RISING_VERDICT,
    TRAIN_VERDICT_DECOUPLING,
    format_verdict_trend_line,
    verdict_trend_alarm,
)
from tac.witness_control.verdict_trend_alarm import (
    RISE_REL_EPS,
    RISING_VERDICT_UNIDENTIFIABLE,
)

# ── FIXTURE 1: the LIVE run (levelset_n600_witness_20260709T105312Z), unify_tau stage,
# ep25-100 verbatim. d_seg best@ep50 then RISES to ep100; ep_loss DESCENDS throughout
# (the decoupling). This is the exact trajectory the operator flagged. ──
LIVE_UNIFY_TAU = [
    {"stage": "verdict", "seg_form": "unify_tau", "epoch": 25, "d_seg": 0.036619,
     "ep_loss": 468.618,
     "d_seg_by_class": [0.11799404, 0.21643435, 0.01492465, 0.02999675, 0.00069395]},
    {"stage": "verdict", "seg_form": "unify_tau", "epoch": 50, "d_seg": 0.03244,
     "ep_loss": 439.394,
     "d_seg_by_class": [0.09725846, 0.34855836, 0.01514204, 0.01626213, 0.00040675]},
    {"stage": "verdict", "seg_form": "unify_tau", "epoch": 75, "d_seg": 0.033791,
     "ep_loss": 429.971,
     "d_seg_by_class": [0.10603284, 0.36536599, 0.01369163, 0.01154538, 0.00037068]},
    {"stage": "verdict", "seg_form": "unify_tau", "epoch": 100, "d_seg": 0.034048,
     "ep_loss": 425.409,
     "d_seg_by_class": [0.1075422, 0.38057654, 0.01334114, 0.01050657, 0.00038492]},
]

# ── FIXTURE 2: mod32cap (levelset_n600_witness_mod32cap_20260706T115554Z), ce stage,
# ep25-125 verbatim. Monotone DESCENDING d_seg + ep_loss (the council-designed clean
# baseline). MUST stay silent. ──
MOD32CAP_CE = [
    {"stage": "verdict", "seg_form": "ce", "epoch": 25, "d_seg": 0.009288, "ep_loss": 572.318},
    {"stage": "verdict", "seg_form": "ce", "epoch": 50, "d_seg": 0.007248, "ep_loss": 492.561},
    {"stage": "verdict", "seg_form": "ce", "epoch": 75, "d_seg": 0.006354, "ep_loss": 456.962},
    {"stage": "verdict", "seg_form": "ce", "epoch": 100, "d_seg": 0.005856, "ep_loss": 434.53},
    {"stage": "verdict", "seg_form": "ce", "epoch": 125, "d_seg": 0.005519, "ep_loss": 422.688},
]

# ── FIXTURE 3: mod32cap tau_softplus terminal plateau (ep600-725 verbatim). d_seg has
# a tiny terminal wiggle that is monotone-up over the last 3, but its rel-slope is BELOW
# the calibrated flat gate -> plateau jitter, MUST stay silent (the magnitude-vs-slope
# distinction: a magnitude threshold would false-fire here; the rel-slope gate does not). ──
MOD32CAP_TAU_TAIL = [
    {"stage": "verdict", "seg_form": "tau_softplus", "epoch": 600, "d_seg": 0.003372, "ep_loss": 19.846},
    {"stage": "verdict", "seg_form": "tau_softplus", "epoch": 625, "d_seg": 0.003393, "ep_loss": 19.841},
    {"stage": "verdict", "seg_form": "tau_softplus", "epoch": 650, "d_seg": 0.003366, "ep_loss": 19.25},
    {"stage": "verdict", "seg_form": "tau_softplus", "epoch": 675, "d_seg": 0.003376, "ep_loss": 18.935},
    {"stage": "verdict", "seg_form": "tau_softplus", "epoch": 700, "d_seg": 0.003407, "ep_loss": 18.727},
    {"stage": "verdict", "seg_form": "tau_softplus", "epoch": 725, "d_seg": 0.003414, "ep_loss": 18.583},
]


# ─────────────────────────── FIRE (the acceptance test) ───────────────────────────
def test_live_run_fires_train_verdict_decoupling_at_ep100():
    a = verdict_trend_alarm(LIVE_UNIFY_TAU)
    assert a.fired()
    assert a.classification == TRAIN_VERDICT_DECOUPLING
    assert a.train_loss_decoupled is True          # ep_loss descending while verdict rising
    assert a.ema_lag_plausible is False            # persisted 2 verdicts + material -> real
    assert a.best_epoch == 50 and a.epoch_latest == 100
    assert a.best_stale_verdicts == 2
    assert a.rise_abs > 0
    # DERIVED material gate: the rise-window rel-slope is above the calibrated flat gate.
    assert a.rel_slope_per_ep > RISE_REL_EPS
    # relative-significance framing (never absolute): rise is a few % of the remaining gap.
    assert a.rise_rel_gap > 0.01


def test_live_run_train_loss_slope_is_negative_over_the_rise_window():
    a = verdict_trend_alarm(LIVE_UNIFY_TAU)
    assert a.train_loss_slope_per_ep is not None
    assert a.train_loss_slope_per_ep < 0.0         # train seg-loss DESCENDING (the decoupling)


# ─────────────────────────── SILENT (mod32cap) ───────────────────────────
def test_mod32cap_descending_stays_silent():
    a = verdict_trend_alarm(MOD32CAP_CE)
    assert not a.fired()
    assert a.classification == NO_ALARM
    assert a.per_class_alarms == ()                # no d_seg_by_class -> nothing fabricated


def test_mod32cap_terminal_plateau_wiggle_stays_silent_below_rel_slope_gate():
    """The magnitude-vs-slope distinction: this tail is monotone-up over the last 3 rows
    with a non-trivial rise_rel_gap, but its rel-slope is below the flat gate -> SILENT."""
    a = verdict_trend_alarm(MOD32CAP_TAU_TAIL)
    assert not a.fired()
    assert a.classification == NO_ALARM
    assert a.rel_slope_per_ep <= RISE_REL_EPS       # below the calibrated flat gate


# ─────────────────────────── PER-CLASS ───────────────────────────
def test_live_run_per_class_lane_fires():
    a = verdict_trend_alarm(LIVE_UNIFY_TAU)
    names = {r["class_name"] for r in a.per_class_alarms}
    assert "Lane" in names                          # the class the operator named (0.349->0.381)
    lane = next(r for r in a.per_class_alarms if r["class_name"] == "Lane")
    assert lane["d_seg_latest"] > lane["d_seg_best"]
    assert lane["rise_rel_value"] > 0.1             # Lane rose ~76% of its value
    # descending classes (Undrivable/Movable) must NOT be in the rising set.
    assert "Undrivable" not in names and "Movable" not in names


def test_per_class_unidentifiable_without_class_breakdown():
    a = verdict_trend_alarm(MOD32CAP_CE)
    assert a.per_class_alarms == ()                 # never fabricated from scalar d_seg


# ─────────────────────────── EMA-LAG vs DECOUPLED disambiguation ───────────────────────────
def test_ema_lag_plausible_for_early_just_crossed_small_rise():
    """A rise that JUST crossed (best_stale==k), very early, below the material gap floor
    is flagged EMA-lag-plausible (soft RISING), NOT hard DECOUPLING — even if loss decoupled."""
    rows = [
        {"stage": "verdict", "seg_form": "s", "epoch": 0, "d_seg": 0.040, "ep_loss": 500},
        {"stage": "verdict", "seg_form": "s", "epoch": 25, "d_seg": 0.030, "ep_loss": 480},  # best
        {"stage": "verdict", "seg_form": "s", "epoch": 50, "d_seg": 0.03024, "ep_loss": 460},  # tiny rise
    ]
    a = verdict_trend_alarm(rows, k=1)
    assert a.fired()
    assert a.classification == RISING_VERDICT        # NOT decoupling despite loss falling
    assert a.ema_lag_plausible is True
    assert a.rise_rel_gap < 0.01                     # below the material floor


def test_persisted_material_rise_is_not_ema_lag():
    """The live case (persisted 2 verdicts, material magnitude) is NOT dismissed as EMA-lag."""
    a = verdict_trend_alarm(LIVE_UNIFY_TAU)
    assert a.ema_lag_plausible is False


# ─────────────────────────── coupling / monotone / gates ───────────────────────────
def test_rising_with_rising_loss_is_not_decoupling():
    """Verdict rising WHILE train loss ALSO rises is RISING_VERDICT, not DECOUPLING
    (the decoupling class requires the loss to be descending/flat)."""
    rows = [
        {"stage": "verdict", "seg_form": "s", "epoch": 0, "d_seg": 0.030, "ep_loss": 400},   # best
        {"stage": "verdict", "seg_form": "s", "epoch": 25, "d_seg": 0.033, "ep_loss": 420},
        {"stage": "verdict", "seg_form": "s", "epoch": 50, "d_seg": 0.036, "ep_loss": 440},  # loss RISING too
    ]
    a = verdict_trend_alarm(rows, k=2)
    assert a.fired()
    assert a.classification == RISING_VERDICT
    assert a.train_loss_decoupled is False


def test_non_monotone_recent_window_stays_silent():
    """A lone up-spike bracketed by a drop (not monotone over the last k+1) stays silent."""
    rows = [
        {"stage": "verdict", "seg_form": "s", "epoch": 0, "d_seg": 0.030, "ep_loss": 400},   # best
        {"stage": "verdict", "seg_form": "s", "epoch": 25, "d_seg": 0.050, "ep_loss": 390},  # spike up
        {"stage": "verdict", "seg_form": "s", "epoch": 50, "d_seg": 0.031, "ep_loss": 380},  # back down
    ]
    a = verdict_trend_alarm(rows, k=2)
    assert not a.fired()
    assert a.classification == NO_ALARM


def test_flat_wiggle_below_rel_slope_gate_stays_silent():
    """Synthetic converged plateau: monotone-up but tiny rel-slope < gate -> silent."""
    rows = [
        {"stage": "verdict", "seg_form": "s", "epoch": 0, "d_seg": 0.003000, "ep_loss": 20},  # best
        {"stage": "verdict", "seg_form": "s", "epoch": 25, "d_seg": 0.003001, "ep_loss": 19},
        {"stage": "verdict", "seg_form": "s", "epoch": 50, "d_seg": 0.003002, "ep_loss": 18},
    ]
    a = verdict_trend_alarm(rows, k=2)
    assert not a.fired()
    assert a.rel_slope_per_ep <= RISE_REL_EPS


# ─────────────────────────── fail-open / determinism / row shape ───────────────────────────
def test_empty_and_short_are_unidentifiable_not_crash():
    assert verdict_trend_alarm([]).classification == RISING_VERDICT_UNIDENTIFIABLE
    assert verdict_trend_alarm(LIVE_UNIFY_TAU[:1]).classification == RISING_VERDICT_UNIDENTIFIABLE
    # <k+1 same-stage rows is UNIDENTIFIABLE, never a fired alarm.
    assert not verdict_trend_alarm(LIVE_UNIFY_TAU[:2], k=2).fired()


def test_deterministic_same_input_same_verdict():
    a = verdict_trend_alarm(LIVE_UNIFY_TAU)
    b = verdict_trend_alarm(LIVE_UNIFY_TAU)
    assert a.to_dict() == b.to_dict()


def test_confound_alarm_row_shape_matches_l1_convention():
    a = verdict_trend_alarm(LIVE_UNIFY_TAU)
    row = a.to_confound_alarm_row()
    assert row["stage"] == "confound_alarm"          # matches trainer L1 convention
    assert row["alarm"] == "verdict_rising_decoupling"
    assert "Lane" in row["per_class_rising"]
    assert "NON-PROMOTABLE" in row["axis"]            # advisory, never a score
    json.dumps(row)                                  # must be JSON-serializable


def test_explicit_loss_rows_override_ep_loss():
    """When a separate loss series is passed, it (not verdict ep_loss) drives decoupling."""
    # verdict rows WITHOUT ep_loss; pass an explicit descending seg_loss series.
    verds = [{"stage": "verdict", "seg_form": "s", "epoch": e, "d_seg": d}
             for e, d in [(0, 0.030), (25, 0.033), (50, 0.036)]]
    loss = [{"epoch": e, "seg_loss": v} for e, v in [(0, 400), (25, 380), (50, 360)]]
    a = verdict_trend_alarm(verds, loss, k=2)
    assert a.classification == TRAIN_VERDICT_DECOUPLING
    assert a.train_loss_slope_per_ep < 0.0


def test_format_line_smoke():
    fire = format_verdict_trend_line(verdict_trend_alarm(LIVE_UNIFY_TAU))
    assert "DECOUPLING" in fire and "Lane" in fire
    silent = format_verdict_trend_line(verdict_trend_alarm(MOD32CAP_CE))
    assert "NO-ALARM" in silent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
