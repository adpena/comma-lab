"""Tests for ``dashboard_trajectory_model.build_fresh_run_projection`` (2026-07-04).

The fresh-run A/B projection must be MEASURED-ANCHORED and honestly labeled: the #205
erosion continuation uses the MEASURED slopes (+4.68e-5/ep onset, +6.0e-6/ep steady);
the fresh seeded curve below the CE floor is a HYPOTHESIS band (labeled), never a
fabricated convergence promise. Both curves share one grid and split EXACTLY at the
tau/MCF divergence epoch so the A/B is visually falsifiable when real data lands.

Run: ``.venv/bin/python -m pytest tools/test_dashboard_fresh_run_projection.py``
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dashboard_trajectory_model as dtm  # noqa: E402

_SCHED = {"tau_start": 300, "l7_start": 1000, "muon_start": 726, "epochs": 1000}
_GOAL = 0.00112


def _proj(sched=None, goal=_GOAL, **kw):
    return dtm.build_fresh_run_projection(sched or dict(_SCHED), goal, **kw)


def test_basic_shape_and_divergence():
    p = _proj()
    assert p["ok"] is True
    assert p["divergence_epoch"] == 300
    n = len(p["epochs"])
    assert len(p["eroding_205"]) == n == len(p["fresh_lo"]) == len(p["fresh_hi"])
    assert p["epochs"][0] == 0 and p["epochs"][-1] == 1000
    assert all(v > 0 for v in p["eroding_205"] + p["fresh_lo"] + p["fresh_hi"])


def test_identical_ce_phase_before_divergence():
    p = _proj()
    for ep, e, lo, hi in zip(p["epochs"], p["eroding_205"], p["fresh_lo"], p["fresh_hi"]):
        if ep < 300:
            assert e == lo == hi, f"curves must be identical pre-divergence (ep{ep})"


def test_ce_phase_passes_measured_anchors():
    p = _proj()
    a = dtm.FRESH_RUN_ANCHORS
    by_ep = dict(zip(p["epochs"], p["eroding_205"]))
    assert abs(by_ep[0] - a["ce_transient"]["ep0_dseg"]) < 1e-9      # ep0 0.746 [measured]
    assert abs(by_ep[25] - a["ce_transient"]["ep25_dseg"]) < 1e-9    # ep25 0.010 [measured]
    # CE floor reached AT the tau boundary
    assert abs(by_ep[300] - a["ce_floor"]["best"]) < 1e-9


def test_eroding_slopes_match_measured_205():
    p = _proj()
    a = dtm.FRESH_RUN_ANCHORS["tau_erosion"]
    by_ep = dict(zip(p["epochs"], p["eroding_205"]))
    ce_best = dtm.FRESH_RUN_ANCHORS["ce_floor"]["best"]
    # onset window (first 25 ep): +4.68e-5/ep
    assert abs(by_ep[325] - (ce_best + a["onset_slope"] * 25)) < 1e-9
    # steady after: +6.0e-6/ep
    expect_450 = ce_best + a["onset_slope"] * 25 + a["steady_slope"] * (450 - 325)
    assert abs(by_ep[450] - expect_450) < 1e-9
    # monotone nondecreasing post-divergence (erosion never "recovers" in the model)
    post = [v for ep, v in zip(p["epochs"], p["eroding_205"]) if ep >= 300]
    assert all(b >= a2 - 1e-12 for a2, b in zip(post, post[1:]))


def test_fresh_band_descends_to_hypothesis_floor_and_upper_holds_ce():
    p = _proj()
    a = dtm.FRESH_RUN_ANCHORS
    post_lo = [v for ep, v in zip(p["epochs"], p["fresh_lo"]) if ep >= 300]
    assert all(b <= a2 + 1e-12 for a2, b in zip(post_lo, post_lo[1:]))  # nonincreasing
    # lower edge ends at max(goal, ce_lo - seed_gain); with these anchors goal binds
    seed_gain = (a["seed_delta"]["lane_fn_unseeded"] - a["seed_delta"]["lane_fn_seeded"])
    expect_floor = max(_GOAL, a["ce_floor"]["lo"] - seed_gain)
    assert abs(p["fresh_floor"] - expect_floor) < 1e-12
    assert abs(post_lo[-1] - expect_floor) < 1e-9
    # upper edge = the CE-floor no-improvement bound
    post_hi = {ep: v for ep, v in zip(p["epochs"], p["fresh_hi"]) if ep >= 300}
    assert all(abs(v - a["ce_floor"]["hi"]) < 1e-12 for v in post_hi.values())
    # the A/B: by end-of-run the fresh lower edge sits far below the eroding curve
    assert post_lo[-1] < p["eroding_205"][-1]


def test_goal_clamp_when_goal_above_seeded_floor():
    p = _proj(goal=0.004)  # a goal ABOVE ce_lo - seed_gain -> the goal clamps the band
    assert abs(p["fresh_floor"] - 0.004) < 1e-12


def test_labels_are_honest():
    p = _proj()
    assert "HYPOTHESIS" in p["labels"]["fresh_band"]
    assert "projected/advisory" in p["labels"]["fresh_band"]
    assert "measured" in p["labels"]["eroding_205"]
    assert "NON-PROMOTABLE" in p["authority"] and "0.19110" in p["authority"]
    # every anchor carries an explicit [measured]/[projected] label
    for k, a in p["anchors"].items():
        assert "label" in a and a["label"].startswith("["), k


def test_defaults_and_degenerate_schedules():
    # no tau_start -> default 300; no epochs -> default 1000
    p = _proj(sched={})
    assert p["divergence_epoch"] == 300 and p["epochs"][-1] == 1000
    # total <= tau is forced past the boundary (never an empty post-tau segment)
    p2 = _proj(sched={"tau_start": 300, "epochs": 200})
    assert p2["epochs"][-1] > 300


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL PASS")
