# SPDX-License-Identifier: MIT
"""Trajectory-derived stopping and adaptive depth allocation.

This module is deliberately scorer-free.  It consumes objective values already
written by a solver receipt and converts projected objective gain into contest
score units using a caller-supplied exchange rate.  Safety caps remain caps:
when one binds the stop reason says so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0
TRAJECTORY_STOPPING_LAW_REF: Final = "trajectory_derived_stopping_law_v1"

FitKind = Literal["geometric", "power_law", "last_k_slope"]
StopReason = Literal[
    "converged_projected",
    "marginal_below_bar",
    "safety_bound_REPORTED",
    "continue_projected",
]


class TrajectoryStoppingError(ValueError):
    """Fail-closed malformed trajectory, fit, stop-law, or allocation input."""


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One recorded objective value at a solver compute coordinate."""

    compute: float
    objective: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.compute) or self.compute < 0.0:
            raise TrajectoryStoppingError("compute must be finite and non-negative")
        if not math.isfinite(self.objective):
            raise TrajectoryStoppingError("objective must be finite")


@dataclass(frozen=True, slots=True)
class TrajectoryStopConfig:
    """Stop law in score units, with caps treated only as safety bounds."""

    score_units_per_objective: float
    marginal_score_gain_per_compute: float
    min_fit_r2: float = 0.95
    min_fit_points: int = 5
    fallback_window_points: int = 4
    objective_floor: float = 0.0
    law_ref: str = TRAJECTORY_STOPPING_LAW_REF

    def __post_init__(self) -> None:
        if not math.isfinite(self.score_units_per_objective) or self.score_units_per_objective <= 0.0:
            raise TrajectoryStoppingError("score_units_per_objective must be positive")
        if (
            not math.isfinite(self.marginal_score_gain_per_compute)
            or self.marginal_score_gain_per_compute <= 0.0
        ):
            raise TrajectoryStoppingError("marginal_score_gain_per_compute must be positive")
        if not 0.0 <= self.min_fit_r2 <= 1.0:
            raise TrajectoryStoppingError("min_fit_r2 must be in [0,1]")
        if self.min_fit_points < 3:
            raise TrajectoryStoppingError("min_fit_points must be at least 3")
        if self.fallback_window_points < 2:
            raise TrajectoryStoppingError("fallback_window_points must be at least 2")
        if not math.isfinite(self.objective_floor):
            raise TrajectoryStoppingError("objective_floor must be finite")
        if not self.law_ref:
            raise TrajectoryStoppingError("law_ref must be nonempty")


@dataclass(frozen=True, slots=True)
class DecayFit:
    kind: FitKind
    r2: float
    asymptote: float | None
    intercept: float
    slope: float
    origin_compute: float
    last_compute: float
    current_objective: float
    fit_quality_flag: str

    def predict_objective(self, compute: float) -> float:
        if compute < self.origin_compute:
            raise TrajectoryStoppingError("cannot extrapolate before the fit origin")
        if self.kind == "geometric":
            residual = math.exp(self.intercept + self.slope * (compute - self.origin_compute))
            return float((self.asymptote or 0.0) + residual)
        if self.kind == "power_law":
            x = max(compute - self.origin_compute + 1.0, 1.0e-12)
            residual = math.exp(self.intercept + self.slope * math.log(x))
            return float((self.asymptote or 0.0) + residual)
        # last-k slope is local and linear, clipped at zero remaining gain.
        return max(0.0, self.current_objective - self.slope * (compute - self.last_compute))

    @property
    def projected_remaining_objective_gain(self) -> float:
        if self.asymptote is None:
            return math.inf
        return max(0.0, self.current_objective - self.asymptote)

    def projected_marginal_objective_gain_per_compute(self, compute: float | None = None) -> float:
        x = self.last_compute if compute is None else compute
        if self.kind == "geometric":
            residual = max(self.predict_objective(x) - (self.asymptote or 0.0), 0.0)
            return max(0.0, -self.slope * residual)
        if self.kind == "power_law":
            shifted = max(x - self.origin_compute + 1.0, 1.0e-12)
            residual = max(self.predict_objective(x) - (self.asymptote or 0.0), 0.0)
            return max(0.0, -self.slope * residual / shifted)
        return max(0.0, self.slope)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "r2": self.r2,
            "asymptote": self.asymptote,
            "intercept": self.intercept,
            "slope": self.slope,
            "origin_compute": self.origin_compute,
            "last_compute": self.last_compute,
            "current_objective": self.current_objective,
            "fit_quality_flag": self.fit_quality_flag,
            "projected_remaining_objective_gain": self.projected_remaining_objective_gain,
            "projected_marginal_objective_gain_per_compute": (
                self.projected_marginal_objective_gain_per_compute()
            ),
        }


@dataclass(frozen=True, slots=True)
class StopDecision:
    should_stop: bool
    stop_reason: StopReason
    law_ref: str
    n_points: int
    current_compute: float
    current_objective: float
    selected_fit: DecayFit | None
    candidate_fits: tuple[DecayFit, ...]
    marginal_score_gain_per_compute: float
    projected_remaining_score_gain: float | None
    threshold_score_gain_per_compute: float
    safety_bound_compute: float | None
    bound_reported: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
            "law_ref": self.law_ref,
            "n_points": self.n_points,
            "current_compute": self.current_compute,
            "current_objective": self.current_objective,
            "selected_fit": None if self.selected_fit is None else self.selected_fit.to_payload(),
            "candidate_fits": [fit.to_payload() for fit in self.candidate_fits],
            "marginal_score_gain_per_compute": self.marginal_score_gain_per_compute,
            "projected_remaining_score_gain": self.projected_remaining_score_gain,
            "threshold_score_gain_per_compute": self.threshold_score_gain_per_compute,
            "safety_bound_compute": self.safety_bound_compute,
            "bound_reported": self.bound_reported,
        }


@dataclass(frozen=True, slots=True)
class ProjectionInterval:
    target_compute: float
    objective_low: float
    objective_high: float
    fits_used: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_compute": self.target_compute,
            "objective_low": self.objective_low,
            "objective_high": self.objective_high,
            "fits_used": list(self.fits_used),
        }


@dataclass(frozen=True, slots=True)
class DepthAllocation:
    item_id: str
    extra_compute: int
    projected_remaining_score_gain: float
    stop_reason: StopReason
    safety_bound_reported: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "extra_compute": self.extra_compute,
            "projected_remaining_score_gain": self.projected_remaining_score_gain,
            "stop_reason": self.stop_reason,
            "safety_bound_reported": self.safety_bound_reported,
        }


def byte_score_units() -> float:
    """Contest score cost of one counted archive byte."""

    return RATE_SCORE_PER_BYTE


def seg_flip_score_units(*, n_pairs: int = 600, height: int = 384, width: int = 512) -> float:
    """Contest score gain from fixing one SegNet argmax pixel in one pair."""

    if n_pairs <= 0 or height <= 0 or width <= 0:
        raise TrajectoryStoppingError("n_pairs, height, and width must be positive")
    return 100.0 / float(n_pairs * height * width)


def gap_fraction_score_bar(*, score_gap: float, fraction: float, compute_units: float) -> float:
    """Derive a marginal S/unit bar from an explicit fraction of the campaign gap."""

    if not math.isfinite(score_gap) or score_gap <= 0.0:
        raise TrajectoryStoppingError("score_gap must be positive")
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise TrajectoryStoppingError("fraction must be positive")
    if not math.isfinite(compute_units) or compute_units <= 0.0:
        raise TrajectoryStoppingError("compute_units must be positive")
    return score_gap * fraction / compute_units


def _points(points: Sequence[TrajectoryPoint | Mapping[str, Any]]) -> tuple[TrajectoryPoint, ...]:
    out: list[TrajectoryPoint] = []
    for item in points:
        if isinstance(item, TrajectoryPoint):
            point = item
        else:
            point = TrajectoryPoint(
                compute=float(item.get("compute", item.get("step", item.get("iteration")))),
                objective=float(item["objective"]),
            )
        out.append(point)
    if len(out) < 2:
        raise TrajectoryStoppingError("at least two trajectory points are required")
    out.sort(key=lambda p: p.compute)
    for prev, cur in zip(out, out[1:]):
        if cur.compute <= prev.compute:
            raise TrajectoryStoppingError("trajectory compute coordinates must be strictly increasing")
    return tuple(out)


def _r2(y: Sequence[float], pred: Sequence[float]) -> float:
    ybar = sum(y) / len(y)
    sst = sum((v - ybar) ** 2 for v in y)
    if sst <= 0.0:
        return 0.0
    rss = sum((a - b) ** 2 for a, b in zip(y, pred))
    return max(0.0, 1.0 - rss / sst)


def _linear_regression(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    den = sum((v - xbar) ** 2 for v in x)
    if den <= 0.0:
        raise TrajectoryStoppingError("degenerate fit x coordinates")
    slope = sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / den
    intercept = ybar - slope * xbar
    return float(intercept), float(slope)


def _candidate_asymptotes(y: Sequence[float], objective_floor: float) -> tuple[float, ...]:
    current = y[-1]
    total_drop = max(y[0] - current, abs(y[0]) * 1.0e-6, 1.0e-9)
    eps = max(abs(current), abs(y[0]), 1.0) * 1.0e-12
    factors = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0)
    out = {max(objective_floor, current - total_drop * factor) for factor in factors}
    out.add(objective_floor)
    return tuple(sorted(a for a in out if a < min(y) - eps))


def _fit_decay_kind(points: tuple[TrajectoryPoint, ...], kind: Literal["geometric", "power_law"], cfg: TrajectoryStopConfig) -> DecayFit | None:
    if len(points) < cfg.min_fit_points:
        return None
    t = [p.compute for p in points]
    y = [p.objective for p in points]
    origin = t[0]
    best: DecayFit | None = None
    for asymptote in _candidate_asymptotes(y, cfg.objective_floor):
        residuals = [v - asymptote for v in y]
        if any(r <= 0.0 or not math.isfinite(r) for r in residuals):
            continue
        log_y = [math.log(r) for r in residuals]
        if kind == "geometric":
            x = [v - origin for v in t]
        else:
            x = [math.log(v - origin + 1.0) for v in t]
        try:
            intercept, slope = _linear_regression(x, log_y)
        except TrajectoryStoppingError:
            continue
        if slope >= 0.0:
            continue
        if kind == "geometric":
            pred = [asymptote + math.exp(intercept + slope * (v - origin)) for v in t]
        else:
            pred = [asymptote + math.exp(intercept + slope * math.log(v - origin + 1.0)) for v in t]
        r2 = _r2(y, pred)
        fit = DecayFit(
            kind=kind,
            r2=r2,
            asymptote=float(asymptote),
            intercept=intercept,
            slope=slope,
            origin_compute=origin,
            last_compute=t[-1],
            current_objective=y[-1],
            fit_quality_flag=("accepted" if r2 >= cfg.min_fit_r2 else "low_r2"),
        )
        if best is None or fit.r2 > best.r2:
            best = fit
    return best


def fit_decay_models(
    points: Sequence[TrajectoryPoint | Mapping[str, Any]],
    config: TrajectoryStopConfig,
) -> tuple[DecayFit, ...]:
    """Fit geometric and power-law tails; low-quality fits are returned as flagged rows."""

    pts = _points(points)
    fits = [
        fit
        for fit in (
            _fit_decay_kind(pts, "geometric", config),
            _fit_decay_kind(pts, "power_law", config),
        )
        if fit is not None
    ]
    return tuple(sorted(fits, key=lambda f: (f.r2, f.kind), reverse=True))


def _fallback_fit(points: tuple[TrajectoryPoint, ...], cfg: TrajectoryStopConfig) -> DecayFit:
    window = points[-min(len(points), cfg.fallback_window_points) :]
    first, last = window[0], window[-1]
    gain = max(0.0, first.objective - last.objective)
    span = max(last.compute - first.compute, 1.0e-12)
    slope = gain / span
    return DecayFit(
        kind="last_k_slope",
        r2=0.0,
        asymptote=None,
        intercept=0.0,
        slope=slope,
        origin_compute=first.compute,
        last_compute=last.compute,
        current_objective=last.objective,
        fit_quality_flag="fallback_last_k_slope",
    )


def evaluate_trajectory_stop(
    points: Sequence[TrajectoryPoint | Mapping[str, Any]],
    config: TrajectoryStopConfig,
    *,
    safety_bound_compute: float | None = None,
) -> StopDecision:
    """Return the typed stop decision for a recorded objective trajectory."""

    pts = _points(points)
    current = pts[-1]
    if safety_bound_compute is not None and (
        not math.isfinite(safety_bound_compute) or safety_bound_compute < current.compute
    ):
        raise TrajectoryStoppingError("safety_bound_compute must be finite and >= current compute")
    fits = fit_decay_models(pts, config)
    accepted = tuple(fit for fit in fits if fit.r2 >= config.min_fit_r2)
    selected = accepted[0] if accepted else _fallback_fit(pts, config)
    marginal_score = (
        selected.projected_marginal_objective_gain_per_compute()
        * config.score_units_per_objective
    )
    remaining_score: float | None
    if math.isfinite(selected.projected_remaining_objective_gain):
        remaining_score = selected.projected_remaining_objective_gain * config.score_units_per_objective
    else:
        remaining_score = None
    last_interval = pts[-1].compute - pts[-2].compute
    one_interval_bar = config.marginal_score_gain_per_compute * max(last_interval, 1.0e-12)

    reason: StopReason = "continue_projected"
    should_stop = False
    bound_reported = False
    if remaining_score is not None and remaining_score <= one_interval_bar:
        reason = "converged_projected"
        should_stop = True
    elif marginal_score < config.marginal_score_gain_per_compute:
        reason = "marginal_below_bar"
        should_stop = True
    elif safety_bound_compute is not None and current.compute >= safety_bound_compute:
        reason = "safety_bound_REPORTED"
        should_stop = True
        bound_reported = True

    return StopDecision(
        should_stop=should_stop,
        stop_reason=reason,
        law_ref=config.law_ref,
        n_points=len(pts),
        current_compute=current.compute,
        current_objective=current.objective,
        selected_fit=selected,
        candidate_fits=fits,
        marginal_score_gain_per_compute=marginal_score,
        projected_remaining_score_gain=remaining_score,
        threshold_score_gain_per_compute=config.marginal_score_gain_per_compute,
        safety_bound_compute=safety_bound_compute,
        bound_reported=bound_reported,
    )


def projection_interval(
    points: Sequence[TrajectoryPoint | Mapping[str, Any]],
    config: TrajectoryStopConfig,
    *,
    target_compute: float,
) -> ProjectionInterval:
    """Project an objective interval from every honest high-quality decay fit."""

    pts = _points(points)
    if target_compute < pts[-1].compute:
        raise TrajectoryStoppingError("target_compute must be >= the last observed compute")
    fits = tuple(fit for fit in fit_decay_models(pts, config) if fit.r2 >= config.min_fit_r2)
    if not fits:
        fits = (_fallback_fit(pts, config),)
    preds = [fit.predict_objective(target_compute) for fit in fits]
    return ProjectionInterval(
        target_compute=float(target_compute),
        objective_low=float(min(preds)),
        objective_high=float(max(preds)),
        fits_used=tuple(fit.kind for fit in fits),
    )


def allocate_adaptive_depths(
    decisions: Mapping[str, StopDecision],
    *,
    total_extra_compute: int,
    safety_cap_per_item: int,
) -> tuple[DepthAllocation, ...]:
    """Allocate recursion depth by projected remaining score gain.

    This is the discrete waterfill recipient for callers that know the fleet
    budget but want to retire uniform caps.  Items that have already stopped
    semantically receive zero; caps only clip the allocation and report.
    """

    if total_extra_compute < 0:
        raise TrajectoryStoppingError("total_extra_compute must be non-negative")
    if safety_cap_per_item < 0:
        raise TrajectoryStoppingError("safety_cap_per_item must be non-negative")
    gains: dict[str, float] = {}
    for item_id, decision in decisions.items():
        if not item_id:
            raise TrajectoryStoppingError("item ids must be nonempty")
        if decision.stop_reason in {"converged_projected", "marginal_below_bar"}:
            gains[item_id] = 0.0
        else:
            gains[item_id] = max(0.0, float(decision.projected_remaining_score_gain or 0.0))
    total_gain = sum(gains.values())
    raw: dict[str, float] = {}
    if total_extra_compute > 0 and total_gain > 0.0:
        raw = {item: total_extra_compute * gain / total_gain for item, gain in gains.items()}
    else:
        raw = {item: 0.0 for item in gains}
    alloc = {item: min(safety_cap_per_item, int(math.floor(value))) for item, value in raw.items()}
    remaining = total_extra_compute - sum(alloc.values())
    order = sorted(gains, key=lambda item: (raw[item] - math.floor(raw[item]), gains[item], item), reverse=True)
    while remaining > 0 and order:
        progressed = False
        for item in order:
            if gains[item] <= 0.0 or alloc[item] >= safety_cap_per_item:
                continue
            alloc[item] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return tuple(
        DepthAllocation(
            item_id=item,
            extra_compute=alloc[item],
            projected_remaining_score_gain=gains[item],
            stop_reason=decisions[item].stop_reason,
            safety_bound_reported=alloc[item] >= safety_cap_per_item and gains[item] > 0.0,
        )
        for item in sorted(decisions)
    )


__all__ = [
    "RATE_SCORE_PER_BYTE",
    "TRAJECTORY_STOPPING_LAW_REF",
    "DecayFit",
    "DepthAllocation",
    "ProjectionInterval",
    "StopDecision",
    "StopReason",
    "TrajectoryPoint",
    "TrajectoryStopConfig",
    "TrajectoryStoppingError",
    "allocate_adaptive_depths",
    "byte_score_units",
    "evaluate_trajectory_stop",
    "fit_decay_models",
    "gap_fraction_score_bar",
    "projection_interval",
    "seg_flip_score_units",
]
