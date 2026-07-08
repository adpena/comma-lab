"""Tests for the cross-series dynamics analyzer (task #312 Phase C).

Correctness of the interaction math is proven on SYNTHETIC series with known lead/lag +
sign; the run-dir loader is proven on a tiny fabricated run.log so no live run is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control import dynamics_analyzer as da


def test_pearson_perfect_and_degenerate():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert da.pearson(x, x) == pytest.approx(1.0)
    assert da.pearson(x, -x) == pytest.approx(-1.0)
    # constant series -> no linear relationship -> 0.0 (fail-safe, not nan)
    assert da.pearson(x, np.array([5.0, 5.0, 5.0, 5.0])) == 0.0
    assert da.pearson(np.array([1.0]), np.array([1.0])) == 0.0


def test_best_lag_recovers_known_lead():
    # y is x delayed by 3 epochs => x LEADS y => positive lag 3.
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(120))
    y = np.empty_like(x)
    y[3:] = x[:-3]
    y[:3] = x[0]
    lag, corr = da.best_lag_correlation(x, y, max_lag=8)
    assert lag == 3
    assert corr > 0.95


def test_best_lag_sign_and_reverse():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(120))
    y = np.empty_like(x)
    y[4:] = x[:-4]
    y[:4] = x[0]
    # x~y gives +4; the reverse y~x must give the negative lag -4 (y lags x).
    lag_fwd, _ = da.best_lag_correlation(x, y, max_lag=8)
    lag_rev, _ = da.best_lag_correlation(y, x, max_lag=8)
    assert lag_fwd == 4
    assert lag_rev == -4


def test_best_lag_short_series_safe():
    lag, corr = da.best_lag_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0]), max_lag=4)
    assert (lag, corr) == (0, 0.0)


def test_align_series_forward_fill_union_grid():
    smap = {
        "a": [(0, 1.0), (2, 3.0), (4, 5.0)],
        "b": [(1, 10.0), (3, 30.0)],
    }
    grid, aligned = da.align_series(smap)
    assert list(grid) == [0, 1, 2, 3, 4]
    # a forward-fills: ep0=1, ep1=1(carry), ep2=3, ep3=3(carry), ep4=5
    assert list(aligned["a"]) == [1.0, 1.0, 3.0, 3.0, 5.0]
    # b has no obs before ep1 => nan at ep0, then fills
    assert np.isnan(aligned["b"][0])
    assert list(aligned["b"][1:]) == [10.0, 10.0, 30.0, 30.0]


def test_finite_overlap_masks_nan():
    a = np.array([np.nan, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, np.nan, 3.0])
    fa, fb = da._finite_overlap(a, b)
    assert list(fa) == [1.0, 3.0]
    assert list(fb) == [1.0, 3.0]


def test_windowed_interaction_stability_high_for_consistent_pair():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(200))
    y = 2.0 * x + rng.standard_normal(200) * 0.01  # tight positive coupling, no lag
    pi = da.windowed_interaction(x, y, "x", "y", window=40, max_lag=6)
    assert pi.lag == 0
    assert pi.correlation > 0.99
    assert pi.n_windows >= 2
    assert pi.stability > 0.8


def test_windowed_interaction_low_stability_for_flipping_pair():
    # first half positively correlated, second half negatively -> unstable.
    n = 160
    t = np.arange(n)
    x = np.sin(t / 5.0)
    y = np.concatenate([np.sin(t[:80] / 5.0), -np.sin(t[80:] / 5.0)])
    pi = da.windowed_interaction(x, y, "x", "y", window=40, max_lag=4)
    assert pi.stability < 0.6


def test_pair_interaction_row_schema_and_lead_label():
    pi = da.PairInteraction(a="hosc_beta", b="d_seg", lag=3, correlation=0.7, n=50,
                            window=40, n_windows=2, stability=0.6)
    row = pi.to_row()
    assert row["pair"] == ["hosc_beta", "d_seg"]
    assert row["lead"] == "hosc_beta"  # lag>0 => a leads
    assert set(row) >= {"pair", "lag", "correlation", "stability", "n_windows", "lead"}
    assert da.PairInteraction("a", "b", -2, 0.5, 10, 40, 1).to_row()["lead"] == "b"
    assert da.PairInteraction("a", "b", 0, 0.5, 10, 40, 1).to_row()["lead"] == "simultaneous"


def _write_fake_run(tmp_path: Path, n: int = 120) -> Path:
    """A minimal run.log: seg loss and d_seg both descend; a gnorm that tracks seg; a
    schedule knob softmax_temp that leads d_seg by a few epochs."""
    log = tmp_path / "run.log"
    lines = []
    for ep in range(1, n + 1):
        seg = 300.0 * np.exp(-ep / 40.0) + 1.0
        gnorm = seg * 0.5 + 2.0
        temp = 1.0 - 0.5 * (ep / n)  # anneal
        lines.append(json.dumps({
            "stage": "loss_terms", "ep": ep, "accum_batch": 0,
            "terms": {"seg": seg, "pose": 0.0, "eikonal": 1e-3 * ep, "length": 0.0},
            "total": seg + 1e-3 * ep, "gnorm": gnorm, "softmax_temp": temp,
        }))
        if ep % 10 == 0:  # sparse verdicts, keyed by "epoch"
            d_seg = 0.5 * np.exp(-ep / 50.0) + 0.004
            lines.append(json.dumps({
                "stage": "verdict", "epoch": ep, "d_seg": d_seg, "d_pose": 0.1,
                "implied_S": 100 * d_seg, "blob_bytes": 80000,
            }))
    log.write_text("\n".join(lines) + "\n")
    return tmp_path


def test_load_series_parses_stages_and_prunes_dead(tmp_path):
    rd = _write_fake_run(tmp_path)
    series = da.load_series(rd)
    assert "term:seg" in series
    assert "d_seg" in series
    assert "gnorm" in series
    assert "softmax_temp" in series
    # pose is constant 0.0 across the run -> pruned (no interaction signal)
    assert "term:pose" not in series
    # verdicts keyed by "epoch" resolved
    assert len(series["d_seg"]) >= 5


def test_analyze_end_to_end_finds_seg_gnorm_coupling(tmp_path):
    rd = _write_fake_run(tmp_path)
    rep = da.analyze(rd, window=30, max_lag=6, min_abs_corr=0.3)
    assert rep.n_series >= 4
    obj = rep.to_obj()
    assert obj["axis"] == da.AXIS_TAG
    assert "NON-PROMOTABLE" in obj["axis"]
    assert "0.19110" in obj["pointer"]
    pairs = {tuple(sorted(r["pair"])) for r in rep.interactions}
    # seg loss and gnorm are built to track -> must surface
    assert ("gnorm", "term:seg") in pairs or ("gnorm", "loss_total") in pairs


def test_recommendation_rows_match_shadow_schema(tmp_path):
    rd = _write_fake_run(tmp_path)
    rep = da.analyze(rd, window=30, max_lag=6)
    for r in rep.recommendations:
        assert set(r) >= {"action", "predicted_dS", "rationale", "evidence", "interaction"}
        assert r["predicted_dS"] is None  # synergy is directional, never a fake ΔS
        assert isinstance(r["evidence"], list) and r["evidence"]
        assert "ADVISORY" in r["rationale"]


def test_analyze_missing_run_dir_is_graceful(tmp_path):
    rep = da.analyze(tmp_path / "nonexistent")
    assert rep.n_series == 0
    assert rep.interactions == []
    assert rep.recommendations == []


def test_slope_helper():
    smap = {"d_seg": [(i, 1.0 - 0.01 * i) for i in range(20)]}
    s = da._slope(smap, "d_seg")
    assert s is not None and s == pytest.approx(-0.01, abs=1e-6)
    assert da._slope(smap, "absent") is None
