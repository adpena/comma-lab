# SPDX-License-Identifier: MIT
"""Canonical early-stopping SlopeWatcher per ARBITRARINESS-EXTINCTION TOP-2.

Per codex routing directive
``codex_routing_directive_arbitrariness_extinction_top2_epochs_per_substrate_early_stopping_20260518.md``
+ slot 22 codex findings review
``codex_findings_review_top_findings_and_operator_routable_recommendations_20260519.md``
+ operator blanket approval 2026-05-19 verbatim *"all operator decisions
and approval granted and provided fuly and completely"*.

Bug class
=========

Per-substrate ``--epochs`` defaults are wildly arbitrary (1, 100, 200,
1000, 2000 across substrates). Combined with TOP-9 (early-stopping
patience undeclared), most trainers run to full ``args.epochs`` regardless
of convergence. Predicted ΔS impact: ``[-0.006, -0.001]``. Cost envelope:
``$0`` (NET-NEGATIVE — saves money by halting before paid GPU wall-clock
continues past the convergence knee).

Resolution path: ``experimental``  — Prechelt 1998 "Early Stopping --
But When?" GL_α / UP_K / PQ_α canonical slope-based criteria. We pick a
strict slope-watcher: track the val-score slope over a window; stop when
the slope falls below an absolute improvement threshold for ``patience_windows``
consecutive windows.

Mathematical contract
=====================

Given a sequence of (epoch_i, val_score_i) observations spaced by
``eval_interval_epochs``, smooth the last ``smoothing_window`` scores by a
plain mean to remove single-eval noise, then compute the slope between
the smoothed score and the previous-window's smoothed score:

    smoothed_t = mean(val_score_{t-W+1..t})
    slope_t = smoothed_t - smoothed_{t-eval_interval_epochs}

If ``slope_t >= min_slope_improvement`` for ``patience_windows`` consecutive
windows, the watcher returns ``stop=True``. Otherwise the watcher returns
``stop=False`` and continues.

Note on sign convention: ``min_slope_improvement`` is the MAXIMUM allowed
slope value still considered "improving". Because the contest score is to
be MINIMIZED, a negative slope = improvement. The canonical default
``min_slope_improvement = -1e-4`` per codex's directive means "a slope
> -1e-4 = effectively flat = stop after ``patience_windows`` such
observations".

6-hook wire-in
==============

Per Catalog #125 ("Subagent coherence-by-default" non-negotiable):

- Hook 1 (sensitivity-map): N/A (helper does not produce a sensitivity
  contribution; consumed by Hook 4 cathedral autopilot ranker via the
  sister ``tac.cathedral_consumers.early_stopping_consumer`` package).
- Hook 2 (Pareto constraint): N/A.
- Hook 3 (bit-allocator): N/A.
- Hook 4 (cathedral autopilot dispatch): ACTIVE via
  ``tac.cathedral_consumers.early_stopping_consumer`` (Catalog #335
  auto-discovery paradigm).
- Hook 5 (continual-learning posterior): ACTIVE via
  ``tac.canonical_equations`` registration as
  ``convergence_slope_early_stop_v1`` (predicted-vs-empirical residual
  refits as substrates land convergence-knee anchors).
- Hook 6 (probe-disambiguator): ACTIVE — the slope-watcher IS the
  canonical disambiguator between "continue training" vs "halt — slope
  flat".

Cross-references
================

- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE"
- CLAUDE.md "EMA — NON-NEGOTIABLE, HIGHEST EMPHASIS" (sister formula
  derivation; the EMA-decay-from-total-steps formula lives at
  ``tac.training.EMA.decay_from_total_steps``)
- Catalog #323 canonical Provenance umbrella
- Catalog #335 cathedral consumer auto-ingest paradigm

Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
dataclass + explicit invariants + zero hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Canonical Polyak/Prechelt default per the codex directive.
DEFAULT_EVAL_INTERVAL_EPOCHS: Final[int] = 50
DEFAULT_PATIENCE_WINDOWS: Final[int] = 3
DEFAULT_MIN_SLOPE_IMPROVEMENT: Final[float] = -1e-4
DEFAULT_SMOOTHING_WINDOW: Final[int] = 5


class SlopeWatcherConfigError(ValueError):
    """Raised when SlopeWatcherConfig violates invariants."""


@dataclass(frozen=True)
class SlopeWatcherConfig:
    """Canonical slope-watcher hyperparameters.

    Per the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode
    (CLAUDE.md non-negotiable) substrates MAY fork this config when their
    convergence-knee dynamics warrant a different ``smoothing_window`` /
    ``patience_windows``; the canonical defaults are derived from
    Polyak averaging window analysis + Prechelt 1998 PQ_α.
    """

    eval_interval_epochs: int = DEFAULT_EVAL_INTERVAL_EPOCHS
    patience_windows: int = DEFAULT_PATIENCE_WINDOWS
    min_slope_improvement: float = DEFAULT_MIN_SLOPE_IMPROVEMENT
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW

    def __post_init__(self) -> None:
        if not isinstance(self.eval_interval_epochs, int) or self.eval_interval_epochs <= 0:
            raise SlopeWatcherConfigError(
                f"eval_interval_epochs must be a positive int; got {self.eval_interval_epochs!r}"
            )
        if not isinstance(self.patience_windows, int) or self.patience_windows <= 0:
            raise SlopeWatcherConfigError(
                f"patience_windows must be a positive int; got {self.patience_windows!r}"
            )
        if not isinstance(self.min_slope_improvement, (int, float)):
            raise SlopeWatcherConfigError(
                f"min_slope_improvement must be numeric; got {type(self.min_slope_improvement).__name__}"
            )
        if self.min_slope_improvement != self.min_slope_improvement:  # NaN check
            raise SlopeWatcherConfigError("min_slope_improvement must not be NaN")
        if not isinstance(self.smoothing_window, int) or self.smoothing_window <= 0:
            raise SlopeWatcherConfigError(
                f"smoothing_window must be a positive int; got {self.smoothing_window!r}"
            )


class SlopeWatcher:
    """Canonical slope-watcher per Prechelt 1998 PQ_α discipline.

    Usage (per the operator-routable directive):

        from tac.early_stopping import SlopeWatcher, SlopeWatcherConfig

        watcher = SlopeWatcher(SlopeWatcherConfig())
        for epoch in range(args.epochs):
            train_one_epoch(...)
            if epoch % watcher.config.eval_interval_epochs == 0:
                val_score = evaluate(...)
                if watcher.step(epoch, val_score):
                    print(f"[early-stopping] stopped at epoch={epoch} per SlopeWatcher")
                    break

    The watcher is **stateful** (tracks the rolling history of
    eval scores + the consecutive-window-with-slope-above-threshold
    counter). It is **deterministic** (given the same sequence of
    (epoch, val_score) calls it returns the same stop decision).
    """

    def __init__(self, config: SlopeWatcherConfig | None = None) -> None:
        self._config: SlopeWatcherConfig = config or SlopeWatcherConfig()
        # History of (epoch, val_score) observations, in call order.
        self._history: list[tuple[int, float]] = []
        # Consecutive eligible windows where slope >= threshold (= "flat or worsening").
        self._consecutive_flat_windows: int = 0
        # Latest computed slope (or None if not enough history yet).
        self._latest_slope: float | None = None

    @property
    def config(self) -> SlopeWatcherConfig:
        return self._config

    @property
    def history(self) -> tuple[tuple[int, float], ...]:
        """Read-only view of observation history."""
        return tuple(self._history)

    @property
    def latest_slope(self) -> float | None:
        return self._latest_slope

    @property
    def consecutive_flat_windows(self) -> int:
        return self._consecutive_flat_windows

    def step(self, epoch: int, val_score: float) -> bool:
        """Record an observation; return ``True`` if training should stop.

        Per the canonical contract:

        1. Append ``(epoch, val_score)`` to history.
        2. If history has < ``smoothing_window`` + ``eval_interval_epochs``
           worth of windows, return ``False`` (not enough data).
        3. Compute the smoothed score over the last ``smoothing_window``
           observations.
        4. Compute the previous-window smoothed score.
        5. Compute the slope = current - previous.
        6. If slope >= ``min_slope_improvement``, increment the
           consecutive-flat counter; else reset to 0.
        7. Return ``True`` iff consecutive >= ``patience_windows``.
        """
        if not isinstance(epoch, int):
            raise SlopeWatcherConfigError(
                f"epoch must be int; got {type(epoch).__name__}"
            )
        if epoch < 0:
            raise SlopeWatcherConfigError(f"epoch must be >= 0; got {epoch}")
        if not isinstance(val_score, (int, float)):
            raise SlopeWatcherConfigError(
                f"val_score must be numeric; got {type(val_score).__name__}"
            )
        if val_score != val_score:  # NaN
            raise SlopeWatcherConfigError("val_score must not be NaN")
        self._history.append((epoch, float(val_score)))

        cfg = self._config
        # Need at least 2 * smoothing_window observations to compute one slope.
        required = 2 * cfg.smoothing_window
        if len(self._history) < required:
            self._latest_slope = None
            return False

        # Smoothed mean of last smoothing_window val_scores.
        scores = [score for _, score in self._history]
        current_smoothed = sum(scores[-cfg.smoothing_window:]) / cfg.smoothing_window
        previous_smoothed = (
            sum(scores[-2 * cfg.smoothing_window : -cfg.smoothing_window])
            / cfg.smoothing_window
        )
        slope = current_smoothed - previous_smoothed
        self._latest_slope = slope

        if slope >= cfg.min_slope_improvement:
            self._consecutive_flat_windows += 1
        else:
            self._consecutive_flat_windows = 0

        return self._consecutive_flat_windows >= cfg.patience_windows

    def reset(self) -> None:
        """Reset to a fresh state (history + counters cleared)."""
        self._history.clear()
        self._consecutive_flat_windows = 0
        self._latest_slope = None


def predict_optimal_epochs_from_slope_window(
    *,
    rough_epochs_budget: int,
    eval_interval_epochs: int = DEFAULT_EVAL_INTERVAL_EPOCHS,
    patience_windows: int = DEFAULT_PATIENCE_WINDOWS,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> int:
    """Closed-form lower bound on stop-epoch given watcher config + budget.

    The slope-watcher cannot fire BEFORE the first
    ``2 * smoothing_window * eval_interval_epochs`` epochs of observations
    are collected (slope window requires 2 smoothed windows). Then it
    fires after at least ``patience_windows * eval_interval_epochs``
    additional flat observations.

    Returns the MIN stop-epoch (in epochs) for the worst-case flat
    trajectory, clamped to ``rough_epochs_budget``.
    """
    if rough_epochs_budget <= 0:
        raise SlopeWatcherConfigError(
            f"rough_epochs_budget must be > 0; got {rough_epochs_budget}"
        )
    min_stop = (2 * smoothing_window + patience_windows) * eval_interval_epochs
    return min(min_stop, rough_epochs_budget)


__all__ = [
    "DEFAULT_EVAL_INTERVAL_EPOCHS",
    "DEFAULT_PATIENCE_WINDOWS",
    "DEFAULT_MIN_SLOPE_IMPROVEMENT",
    "DEFAULT_SMOOTHING_WINDOW",
    "SlopeWatcherConfig",
    "SlopeWatcherConfigError",
    "SlopeWatcher",
    "predict_optimal_epochs_from_slope_window",
]
