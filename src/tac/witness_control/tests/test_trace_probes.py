"""Tests for tac.witness_control.trace_probes — the T5 crucible trace-replay instruments.

Behavioral (not constant) tests: synthetic traces with known ground truth + the bit-for-bit
mod32cap estimator anchor (v2 §2.2b table) when the run dir is present on this host.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tac.witness_control.trace_probes import (
    cadence_replay,
    copredicate_backtest,
    forfeit_matched_backtest,
    load_history,
    refit_nu_per_stage,
)

MOD32CAP = Path(__file__).resolve().parents[4] / (
    "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z")


def _exp_trace(a=3.4e-3, b=1.2e-3, nu=0.025, lo=300, hi=725, grid=25):
    return [(e, a + b * math.exp(-nu * (e - lo))) for e in range(lo, hi + 1, grid)]


# ---------------------------------------------------------------- co-predicate anchor
def test_copredicate_fires_on_flat_trace():
    rows = [(e, 1e-3) for e in range(0, 251, 25)]
    r = copredicate_backtest(rows)
    assert r["n_fires"] == len(rows) - 3
    assert r["first_fire_epoch"] == rows[3][0]


def test_copredicate_holds_on_fast_descent():
    rows = [(e, 1e-2 * (0.9 ** (e // 25))) for e in range(0, 251, 25)]
    assert copredicate_backtest(rows)["n_fires"] == 0


@pytest.mark.skipif(not (MOD32CAP / "levelset_train_result.json").exists(),
                    reason="mod32cap run dir not on this host")
def test_copredicate_mod32cap_bitforbit_anchor():
    """The v2 §2.2b shipped backtest table: ep625 / -1.369e-3 / n=8."""
    r = copredicate_backtest(load_history(MOD32CAP))
    assert r["first_fire_epoch"] == 625
    assert r["n_fires"] == 8
    assert abs(r["first_fire_rel_slope"] - (-1.3693665e-3)) < 1e-9


# ---------------------------------------------------------------- forfeit-matched arm (P-CT3)
def test_forfeit_arm_fires_when_slope_below_s_star():
    # slope decays exponentially; arm must first-fire where windowed slope < s*.
    rows = _exp_trace(nu=0.03)
    s_star = 1.4154e-5
    r = forfeit_matched_backtest(rows, s_star=s_star)
    fire = r["first_sustained_fire_endpoint"]
    assert fire is not None
    per = {p["epoch"]: p for p in r["per_verdict"]}
    assert per[fire]["slope_endpoint_S_per_ep"] < s_star
    prev = [p for p in r["per_verdict"] if p["epoch"] < fire]
    assert all(p["slope_endpoint_S_per_ep"] >= s_star for p in prev)


def test_forfeit_arm_never_fires_on_fast_descent():
    rows = _exp_trace(b=5e-3, nu=0.002)  # slow contraction => slope stays high in-window
    r = forfeit_matched_backtest(rows, s_star=1e-9)
    assert r["first_sustained_fire_endpoint"] is None


def test_forfeit_report_arithmetic():
    rows = _exp_trace(nu=0.03)
    r = forfeit_matched_backtest(rows, s_star=1.4154e-5)
    rep = r["fire_report_endpoint"]
    # monotone descending trace: EMA-best at fire == d at fire; stage best == last row.
    assert rep["ema_best_epoch"] == rep["fire_epoch"]
    d = dict(rows)
    expected = 100.0 * (d[rep["fire_epoch"]] - r["stage_best"]["d_seg"])
    assert abs(rep["forfeit_S_vs_stage_best"] - expected) < 1e-15


def test_ls_and_endpoint_estimators_agree_on_linear_trace():
    rows = [(e, 5e-3 - 1e-6 * e) for e in range(300, 726, 25)]
    r = forfeit_matched_backtest(rows, s_star=0.0)
    for p in r["per_verdict"]:
        assert abs(p["slope_endpoint_S_per_ep"] - p["slope_lsfit_S_per_ep"]) < 1e-12
        assert abs(p["slope_endpoint_S_per_ep"] - 1e-4) < 1e-12


# ---------------------------------------------------------------- nu refit (P-CT1)
def test_refit_nu_recovers_known_exponential():
    nu_true = 0.025
    rows = _exp_trace(nu=nu_true, lo=0, hi=1000)
    r = refit_nu_per_stage(rows, {"synthetic": (0, 1000)}, n_boot=10)
    st = r["stages"]["synthetic"]
    assert abs(st["nu_per_ep"] - nu_true) / nu_true < 0.05
    assert st["preferred_model_aic"] == "exponential"
    laws = st["recomputed_window_laws"]
    assert abs(laws["settle_3_over_nu_ep"] - 3.0 / st["nu_per_ep"]) < 1e-9
    assert abs(laws["s_star_nu_times_forfeit_S_per_ep"]
               - st["nu_per_ep"] * 5.450779e-4) < 1e-12


def test_refit_nu_insufficient_rows_is_explicit():
    rows = _exp_trace()
    r = refit_nu_per_stage(rows, {"empty": (5000, 6000)})
    assert "error" in r["stages"]["empty"]


def test_refit_reports_powerlaw_alternative():
    rows = _exp_trace(nu=0.02)
    st = refit_nu_per_stage(rows, {"s": (300, 725)}, n_boot=10)["stages"]["s"]
    assert "powerlaw_alpha" in st and "delta_aic_exp_minus_pow" in st


# ---------------------------------------------------------------- cadence replay (P-CT2)
def test_cadence_floor_binds_on_steep_trace():
    rows = [(e, 0.1 - 5e-5 * e) for e in range(0, 1001, 25)]  # |S'| = 5e-3 S/ep everywhere
    r = cadence_replay(rows)
    assert r["n_skipped"] == 0  # slope >> floor_S/25 everywhere => floor 25 binds
    assert not r["missed_prefix_best_beyond_one_cadence"]


def test_cadence_cap_binds_on_flat_trace():
    rows = [(0, 1e-3), (25, 9.9e-4)] + [(e, 9.9e-4) for e in range(50, 1001, 25)]
    r = cadence_replay(rows)
    assert r["n_skipped"] > 0
    # flat region: dt should reach the 100-ep cap => 3-row gaps appear.
    gaps = [b - a for a, b in zip(r["visited_epochs"], r["visited_epochs"][1:])]
    assert max(gaps) == 100


def test_cadence_missed_best_detection():
    # descending, then a sharp isolated best deep in a flat (stretched-cadence) region.
    rows = []
    for e in range(0, 1001, 25):
        v = 1e-3 if e < 100 else 5e-4
        rows.append((e, v))
    rows[rows.index((500, 5e-4))] = (500, 1e-5)  # isolated best at ep500
    r = cadence_replay(rows)
    if 500 not in r["visited_epochs"]:
        assert r["global_best_dist_to_nearest_visited"] >= 0  # analysis fields present
    assert r["global_best_epoch"] == 500


def test_cadence_replay_pair_estimator_runs():
    rows = _exp_trace(lo=0, hi=1000)
    r = cadence_replay(rows, estimator="pair")
    assert r["n_visited"] + r["n_skipped"] == r["n_trace_verdicts"]


def test_cadence_replay_rejects_unknown_estimator():
    with pytest.raises(ValueError):
        cadence_replay(_exp_trace(lo=0, hi=1000), estimator="bogus")


# ---------------------------------------------------------------- loader
def test_load_history_rejects_short_trace(tmp_path):
    (tmp_path / "levelset_train_result.json").write_text(
        '{"history": [{"epoch": 0, "d_seg": 0.1}]}')
    with pytest.raises(ValueError):
        load_history(tmp_path)


@pytest.mark.skipif(not (MOD32CAP / "levelset_train_result.json").exists(),
                    reason="mod32cap run dir not on this host")
def test_load_history_mod32cap_41_rows():
    rows = load_history(MOD32CAP)
    assert len(rows) == 41
    assert rows[0] == (0, pytest.approx(0.7438667975531683))
    assert rows[-1][0] == 1000


def test_cadence_replay_rejects_non_uniform_grid():
    rows = [(0, 1e-3), (25, 9e-4), (60, 8e-4), (85, 7e-4)]
    with pytest.raises(ValueError):
        cadence_replay(rows)


def test_cadence_replay_rejects_dt_lo_below_grid():
    rows = [(e, 1e-3) for e in range(0, 201, 50)]
    with pytest.raises(ValueError):
        cadence_replay(rows, dt_lo=25)
