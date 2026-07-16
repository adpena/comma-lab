# SPDX-License-Identifier: MIT
"""Tests for the rolling-average rate-proxy telemetry producer (#408 · FEED-ratetelemetry)."""
from __future__ import annotations

import pytest

from tac.witness_control import rate_rolling_telemetry as rrt


def test_rolling_mean_windowed():
    assert rrt.rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], window=3) == pytest.approx(4.0)
    assert rrt.rolling_mean([1.0, 2.0, 3.0], window=0) == pytest.approx(2.0)


def test_rolling_mean_empty_raises():
    with pytest.raises(ValueError):
        rrt.rolling_mean([], window=5)


def test_rolling_mean_nonfinite_fail_loud():
    with pytest.raises(ValueError):
        rrt.rolling_mean([1.0, float("nan"), 3.0], window=3)


def test_signal_within_when_flat():
    series = [100.0] * 30
    sig, run = rrt.classify_drift_signal(series, window=5)
    assert sig == rrt.SIGNAL_WITHIN
    assert run == 0


def test_signal_within_when_trending_down():
    # rate trending DOWN is healthy -> never a signal
    series = [100.0 - i for i in range(30)]
    sig, _ = rrt.classify_drift_signal(series, window=5)
    assert sig == rrt.SIGNAL_WITHIN


def test_signal_within_small_fluctuation_inside_band():
    # +/-1% jitter around a flat mean stays inside the 2% band
    series = [100.0 + (1.0 if i % 2 else -1.0) for i in range(40)]
    sig, _ = rrt.classify_drift_signal(series, window=5, band_eps=0.02)
    assert sig == rrt.SIGNAL_WITHIN


def test_signal_drifting_up_then_sustained():
    # a single above-band step is DRIFTING_UP; a long steady climb is SUSTAINED_GROWTH
    flat = [100.0] * 10
    ramp = [100.0 * (1.10 ** i) for i in range(1, 12)]  # +10%/step sustained climb
    sig_sustained, run = rrt.classify_drift_signal(flat + ramp, window=3, sustain_count=3)
    assert sig_sustained == rrt.SIGNAL_SUSTAINED_GROWTH
    assert run >= 3


def test_signal_drifting_up_short_run():
    # exactly below sustain_count consecutive above-band steps -> DRIFTING_UP not SUSTAINED
    flat = [100.0] * 12
    small_bump = [130.0, 131.0]  # one window of growth then it stops climbing
    sig, run = rrt.classify_drift_signal(flat + small_bump, window=3, sustain_count=5)
    assert sig in (rrt.SIGNAL_WITHIN, rrt.SIGNAL_DRIFTING_UP)
    assert sig != rrt.SIGNAL_SUSTAINED_GROWTH


def test_signal_insufficient_history():
    sig, run = rrt.classify_drift_signal([1.0, 2.0, 3.0], window=5)
    assert sig == rrt.SIGNAL_WITHIN
    assert run == 0


def test_row_basic_fields():
    series = [100.0] * 20
    row = rrt.rate_rolling_row(42, series, window=5)
    assert row["schema"] == rrt.RATE_ROLLING_SCHEMA
    assert row["stage"] == "rate_rolling"
    assert row["ep"] == 42
    assert row["rolling_avg"] == pytest.approx(100.0)
    assert row["instant"] == pytest.approx(100.0)
    assert row["drift_signal"] == rrt.SIGNAL_WITHIN
    assert row["informs_only"] is True
    assert "proxy_tail" in row


def test_row_empty_series_raises():
    with pytest.raises(ValueError):
        rrt.rate_rolling_row(0, [], window=5)


def test_row_never_emits_kill_signal():
    # exhaustive: no proxy series produces a signal outside the graduated soft set
    for climb in (1.0, 1.05, 1.2, 2.0):
        series = [100.0 * (climb ** i) for i in range(30)]
        row = rrt.rate_rolling_row(1, series, window=5)
        assert row["drift_signal"] in rrt.SIGNAL_STATES
        assert row["informs_only"] is True


def test_baseline_rel_from_t0():
    base = rrt.RateRollingBaseline(epoch=0, rolling_mean=100.0)
    series = [110.0] * 20
    row = rrt.rate_rolling_row(50, series, window=5, baseline=base)
    assert row["rel_from_t0"] == pytest.approx(0.10)
    assert row["baseline_epoch"] == 0


def test_baseline_roundtrip_resume():
    series = [100.0 + i for i in range(20)]
    row = rrt.rate_rolling_row(7, series, window=5)
    base = rrt.baseline_from_row(row)
    assert base.epoch == 7
    assert base.rolling_mean == pytest.approx(row["rolling_avg"])
    # proxy_tail persisted for continuous rolling across resume
    assert len(base.proxy_tail) == len(row["proxy_tail"])


def test_baseline_from_row_wrong_schema():
    with pytest.raises(ValueError):
        rrt.baseline_from_row({"schema": "other.v1", "ep": 0, "rolling_avg": 1.0})


def test_baseline_bad_values_fail_loud():
    with pytest.raises(ValueError):
        rrt.RateRollingBaseline(epoch=-1, rolling_mean=1.0)
    with pytest.raises(ValueError):
        rrt.RateRollingBaseline(epoch=0, rolling_mean=float("inf"))


def test_proxy_tail_bounded():
    series = [float(i) for i in range(200)]
    row = rrt.rate_rolling_row(1, series, window=5)
    # tail keeps 2*window for continuous resume, not the whole history
    assert len(row["proxy_tail"]) == 10


def test_negative_band_eps_raises():
    with pytest.raises(ValueError):
        rrt.classify_drift_signal([1.0] * 20, band_eps=-0.1)
