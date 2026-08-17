"""Guard the ddm_a1s FO-1 adjudicator against silent drift.

`_a1_verdict` converts the measured alpha ladder into the row MAIN acts on, so it is the
highest-blast-radius function in the stage: a wrong branch here turns a neutral result into a
"LIVE" zero-byte candidate, or buries a real one.  These tests pin the bands to the values the
sr1 FIRE-ORDER pre-registered BEFORE any alpha > 0 was scored, and exercise every branch,
including the mixed case the order did not enumerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

sr1 = pytest.importorskip("ddm_sr1_manufactured_seg_recovery")


def _ladder(**overrides: int) -> dict:
    """A ladder pinned at the control, with the named alphas overridden."""
    flips = dict.fromkeys(sr1.A1_ALPHAS, sr1.A1_CONTROL_FLIPS)
    for key, value in overrides.items():
        flips[float(key.removeprefix("a").replace("p", "."))] = value
    return flips


def test_bands_match_the_pre_registered_fire_order():
    """The four constants ARE the verdict; they must never be re-derived downward."""
    assert sr1.A1_CONTROL_FLIPS == 34_938
    assert sr1.A1_LIVE_BELOW == 33_251
    # +-1% of the control, inclusive, in whole flips.
    assert sr1.A1_NEUTRAL_LO == 34_589 and sr1.A1_CONTROL_FLIPS * 0.99 <= sr1.A1_NEUTRAL_LO
    assert sr1.A1_NEUTRAL_HI == 35_287 and sr1.A1_CONTROL_FLIPS * 1.01 >= sr1.A1_NEUTRAL_HI
    # The LIVE bar is exactly 5% of rt1's manufactured round trip.
    assert int(0.05 * sr1.RT1_ROUND_TRIP_FLIPS) == sr1.A1_CONTROL_FLIPS - sr1.A1_LIVE_BELOW
    assert sr1.A1_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert sr1.A1_PIN_THREADS == 8


def test_live_fires_only_strictly_below_the_bar():
    assert sr1._a1_verdict(_ladder(a0p5=33_250))["verdict"] == "LIVE"
    # Exactly at the bar is NOT live -- the order says "< 33,251".
    assert sr1._a1_verdict(_ladder(a0p5=33_251))["verdict"] != "LIVE"


def test_live_reports_the_recovered_share_and_gap_arithmetic():
    out = sr1._a1_verdict(_ladder(a0p75=30_000))
    assert out["verdict"] == "LIVE"
    assert out["best_alpha"] == 0.75
    assert out["flips_recovered_vs_control"] == 4_938
    assert out["delta_S_seg"] < 0.0  # a recovery lowers S
    assert out["share_of_round_trip_recovered"] == pytest.approx(
        4_938 / sr1.RT1_ROUND_TRIP_FLIPS)
    assert out["share_of_gap_closed"] == pytest.approx(
        4_938 * sr1.SEG_DS_PER_FLIP / sr1.GAP_S)


def test_neutral_requires_every_alpha_inside_the_band():
    out = sr1._a1_verdict(_ladder(a0p25=34_600, a1=35_200))
    assert out["verdict"] == "CLOSED_NEUTRAL"
    assert out["verdict_scope"].startswith("FORMULATION")
    # One alpha outside the band is no longer neutral.
    assert sr1._a1_verdict(_ladder(a1=35_400))["verdict"] != "CLOSED_NEUTRAL"


def test_harmful_requires_every_positive_alpha_above_the_band():
    flips = {0.0: sr1.A1_CONTROL_FLIPS, 0.25: 36_000, 0.5: 37_000,
             0.75: 38_000, 1.0: 40_000}
    out = sr1._a1_verdict(flips)
    assert out["verdict"] == "CLOSED_HARMFUL"
    assert out["delta_S_seg"] > 0.0


def test_mixed_ladder_is_reported_as_indeterminate_not_forced_into_a_band():
    """Some alphas harmful, some neutral: the order enumerated no band, so say so."""
    flips = {0.0: sr1.A1_CONTROL_FLIPS, 0.25: 34_900, 0.5: 36_500,
             0.75: 37_000, 1.0: 38_000}
    assert sr1._a1_verdict(flips)["verdict"] == "INDETERMINATE_MIXED"


def test_alpha_zero_never_decides_the_verdict():
    """alpha = 0 is the control, not a treatment: it must be excluded from `best`."""
    out = sr1._a1_verdict(_ladder(a0p25=34_900))
    assert out["best_alpha"] == 0.25
    assert out["best_flips"] == 34_900


def test_ties_resolve_to_the_smaller_alpha():
    """A tie should report the weaker, less clipping-exposed setting."""
    out = sr1._a1_verdict(_ladder(a0p25=33_000, a0p75=33_000))
    assert out["best_alpha"] == 0.25
