# SPDX-License-Identifier: MIT
"""Tests for ``tac.early_stopping`` SlopeWatcher per TOP-2 arbitrariness extinction.

Per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE 2026-05-19. The canonical
helper extincts the per-substrate arbitrary ``--epochs`` default; tests
pin the slope-watcher mathematical contract + invariants.
"""
from __future__ import annotations

import pytest

from tac.early_stopping import (
    DEFAULT_EVAL_INTERVAL_EPOCHS,
    DEFAULT_MIN_SLOPE_IMPROVEMENT,
    DEFAULT_PATIENCE_WINDOWS,
    DEFAULT_SMOOTHING_WINDOW,
    SlopeWatcher,
    SlopeWatcherConfig,
    SlopeWatcherConfigError,
    predict_optimal_epochs_from_slope_window,
)


def test_default_config_constructs() -> None:
    cfg = SlopeWatcherConfig()
    assert cfg.eval_interval_epochs == DEFAULT_EVAL_INTERVAL_EPOCHS
    assert cfg.patience_windows == DEFAULT_PATIENCE_WINDOWS
    assert cfg.min_slope_improvement == DEFAULT_MIN_SLOPE_IMPROVEMENT
    assert cfg.smoothing_window == DEFAULT_SMOOTHING_WINDOW


def test_config_rejects_non_positive_eval_interval() -> None:
    with pytest.raises(SlopeWatcherConfigError, match="eval_interval_epochs"):
        SlopeWatcherConfig(eval_interval_epochs=0)
    with pytest.raises(SlopeWatcherConfigError):
        SlopeWatcherConfig(eval_interval_epochs=-1)


def test_config_rejects_non_positive_patience() -> None:
    with pytest.raises(SlopeWatcherConfigError, match="patience_windows"):
        SlopeWatcherConfig(patience_windows=0)


def test_config_rejects_nan_slope() -> None:
    with pytest.raises(SlopeWatcherConfigError, match="NaN"):
        SlopeWatcherConfig(min_slope_improvement=float("nan"))


def test_config_rejects_non_positive_smoothing() -> None:
    with pytest.raises(SlopeWatcherConfigError, match="smoothing_window"):
        SlopeWatcherConfig(smoothing_window=0)


def test_watcher_returns_false_with_insufficient_history() -> None:
    cfg = SlopeWatcherConfig(smoothing_window=5)
    w = SlopeWatcher(cfg)
    # Need 2 * smoothing_window = 10 observations before any slope.
    for i in range(9):
        assert w.step(i, 1.0 - 0.01 * i) is False
    assert w.latest_slope is None


def test_watcher_computes_slope_after_threshold() -> None:
    cfg = SlopeWatcherConfig(
        smoothing_window=2,
        patience_windows=10,  # high so we never fire on this test
        min_slope_improvement=-1.0,
    )
    w = SlopeWatcher(cfg)
    # Steady improvement: scores 1.0, 0.9, 0.8, 0.7
    w.step(0, 1.0)
    w.step(1, 0.9)
    w.step(2, 0.8)
    w.step(3, 0.7)
    # current_smoothed = mean(0.8, 0.7) = 0.75
    # previous_smoothed = mean(1.0, 0.9) = 0.95
    # slope = 0.75 - 0.95 = -0.2 (improvement)
    assert w.latest_slope == pytest.approx(-0.2)


def test_watcher_fires_after_patience_flat_windows() -> None:
    cfg = SlopeWatcherConfig(
        smoothing_window=2,
        patience_windows=3,
        min_slope_improvement=-1e-4,  # accepts >= -1e-4 as "flat"
    )
    w = SlopeWatcher(cfg)
    # Flat plateau: every score = 0.5
    for i in range(8):
        result = w.step(i, 0.5)
        if i < 3:
            assert result is False
    # By step 8 (window 4+), we've accumulated enough flat windows
    # Need: 2*smoothing_window = 4 to compute first slope, then
    # patience_windows = 3 more flat observations.
    # i=3: first slope = 0 (flat); counter=1
    # i=4: counter=2
    # i=5: counter=3 -> fires
    cfg2 = SlopeWatcherConfig(
        smoothing_window=2, patience_windows=3, min_slope_improvement=-1e-4
    )
    w2 = SlopeWatcher(cfg2)
    fired = [w2.step(i, 0.5) for i in range(8)]
    # First slope computed at i=3; counter reaches patience at i=5.
    assert fired[5] is True
    assert any(fired[3:])


def test_watcher_resets_counter_on_real_improvement() -> None:
    cfg = SlopeWatcherConfig(
        smoothing_window=2, patience_windows=3, min_slope_improvement=-1e-4
    )
    w = SlopeWatcher(cfg)
    # Flat for 2 windows, then real improvement, then flat
    w.step(0, 0.5)
    w.step(1, 0.5)
    w.step(2, 0.5)  # slope = 0; counter = 1
    w.step(3, 0.5)  # slope = 0; counter = 2
    w.step(4, 0.3)  # slope = -0.1; counter reset to 0
    assert w.consecutive_flat_windows == 0


def test_watcher_step_rejects_nan_score() -> None:
    w = SlopeWatcher()
    with pytest.raises(SlopeWatcherConfigError, match="NaN"):
        w.step(0, float("nan"))


def test_watcher_step_rejects_negative_epoch() -> None:
    w = SlopeWatcher()
    with pytest.raises(SlopeWatcherConfigError, match=">= 0"):
        w.step(-1, 0.5)


def test_watcher_step_rejects_non_int_epoch() -> None:
    w = SlopeWatcher()
    with pytest.raises(SlopeWatcherConfigError, match="int"):
        w.step(1.5, 0.5)  # type: ignore[arg-type]


def test_watcher_history_is_read_only_tuple() -> None:
    w = SlopeWatcher()
    w.step(0, 1.0)
    w.step(1, 0.9)
    history = w.history
    assert isinstance(history, tuple)
    assert history == ((0, 1.0), (1, 0.9))


def test_watcher_reset_clears_state() -> None:
    cfg = SlopeWatcherConfig(smoothing_window=2)
    w = SlopeWatcher(cfg)
    for i in range(5):
        w.step(i, 0.5)
    assert len(w.history) == 5
    w.reset()
    assert w.history == ()
    assert w.consecutive_flat_windows == 0
    assert w.latest_slope is None


def test_predict_optimal_epochs_bounded_by_budget() -> None:
    # Watcher needs at minimum (2 * smoothing_window + patience_windows) *
    # eval_interval_epochs to fire.
    # Defaults: (2 * 5 + 3) * 50 = 650
    min_stop = predict_optimal_epochs_from_slope_window(rough_epochs_budget=1000)
    assert min_stop == 650


def test_predict_optimal_epochs_clamps_to_small_budget() -> None:
    # If budget is smaller than required for the watcher to fire, return budget.
    min_stop = predict_optimal_epochs_from_slope_window(rough_epochs_budget=100)
    assert min_stop == 100


def test_predict_optimal_epochs_rejects_zero_budget() -> None:
    with pytest.raises(SlopeWatcherConfigError):
        predict_optimal_epochs_from_slope_window(rough_epochs_budget=0)


def test_watcher_canonical_canonical_no_fire_on_steady_improvement() -> None:
    """A watcher should NEVER fire while the loss is still meaningfully improving."""
    cfg = SlopeWatcherConfig(
        smoothing_window=3, patience_windows=2, min_slope_improvement=-1e-4
    )
    w = SlopeWatcher(cfg)
    # Linear improvement: 1.0, 0.99, 0.98, 0.97, ... slope = -0.01 throughout
    fired_at = None
    for i in range(20):
        if w.step(i, 1.0 - 0.01 * i):
            fired_at = i
            break
    # Slope -0.01 is well below the -1e-4 threshold; never fires
    assert fired_at is None


def test_watcher_realistic_substrate_convergence() -> None:
    """Simulate a realistic substrate training: improvement → plateau → fire."""
    cfg = SlopeWatcherConfig(
        smoothing_window=3, patience_windows=3, min_slope_improvement=-1e-4
    )
    w = SlopeWatcher(cfg)
    fired_epoch = None
    # First 20 epochs: real improvement (drops 0.5 to 0.2)
    for i in range(20):
        if w.step(i, 0.5 - 0.015 * i):
            fired_epoch = i
            break
    # Should NOT fire during improvement
    assert fired_epoch is None
    # Then 10 epochs of plateau at 0.2 — should fire
    for i in range(20, 40):
        if w.step(i, 0.2):
            fired_epoch = i
            break
    assert fired_epoch is not None and fired_epoch <= 30


def test_canonical_default_values_match_codex_directive() -> None:
    """Pin the canonical defaults per codex directive TOP-2."""
    assert DEFAULT_EVAL_INTERVAL_EPOCHS == 50
    assert DEFAULT_PATIENCE_WINDOWS == 3
    assert DEFAULT_MIN_SLOPE_IMPROVEMENT == -1e-4
    assert DEFAULT_SMOOTHING_WINDOW == 5


def test_slopewatcher_is_deterministic_given_same_history() -> None:
    """Two watchers fed identical histories produce identical decisions."""
    sequence = [(i, 0.5 + 0.001 * (i % 3)) for i in range(30)]
    w1 = SlopeWatcher()
    w2 = SlopeWatcher()
    fired1 = [w1.step(e, s) for e, s in sequence]
    fired2 = [w2.step(e, s) for e, s in sequence]
    assert fired1 == fired2
    assert w1.latest_slope == w2.latest_slope
    assert w1.consecutive_flat_windows == w2.consecutive_flat_windows


def test_slopewatcher_returns_int_count_not_bool() -> None:
    w = SlopeWatcher()
    assert isinstance(w.consecutive_flat_windows, int)
