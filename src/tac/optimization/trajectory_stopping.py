# SPDX-License-Identifier: MIT
"""Trajectory-derived stopping and adaptive depth allocation.

This module is deliberately scorer-free.  It consumes objective values already
written by a solver receipt and converts projected objective gain into contest
score units using a caller-supplied exchange rate.  Safety caps remain caps:
when one binds the stop reason says so.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0
TRAJECTORY_STOPPING_LAW_REF: Final = "trajectory_derived_stopping_law_v1"

FitKind = Literal["geometric", "power_law", "last_k_slope"]
CapStopReason = Literal["converged", "cap_bound", "failed"]
CapBoundKind = Literal["steps", "wall_clock_seconds"]
StopReason = Literal[
    "converged_projected",
    "marginal_below_bar",
    "safety_bound_REPORTED",
    "continue_projected",
]
TrainingStopAction = Literal[
    "INSUFFICIENT_HISTORY",
    "CONTINUE",
    "STOP_CONVERGED",
    "ROLLBACK_OR_RETREAT",
    "QUEUE_RESUME",
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
        if not math.isfinite(self.marginal_score_gain_per_compute) or self.marginal_score_gain_per_compute <= 0.0:
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
            "projected_marginal_objective_gain_per_compute": (self.projected_marginal_objective_gain_per_compute()),
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
class CapStopReceipt:
    """Typed cap/convergence receipt for solvers that do not use trajectory fits."""

    stop_reason: CapStopReason
    steps_run: int
    cap: int | None
    still_descending: bool | None
    bound_kind: CapBoundKind = "steps"
    bound_value: float | None = None
    observed_value: float | None = None

    def __post_init__(self) -> None:
        if self.steps_run < 0:
            raise TrajectoryStoppingError("steps_run must be non-negative")
        if self.cap is not None and self.cap < 0:
            raise TrajectoryStoppingError("cap must be non-negative when supplied")
        if self.bound_kind not in {"steps", "wall_clock_seconds"}:
            raise TrajectoryStoppingError("unknown cap bound_kind")
        if self.bound_value is not None and (not math.isfinite(self.bound_value) or self.bound_value < 0.0):
            raise TrajectoryStoppingError("bound_value must be finite and non-negative")
        if self.observed_value is not None and (not math.isfinite(self.observed_value) or self.observed_value < 0.0):
            raise TrajectoryStoppingError("observed_value must be finite and non-negative")
        if self.stop_reason == "cap_bound":
            if self.bound_kind == "steps":
                if self.cap is None:
                    raise TrajectoryStoppingError("step cap_bound receipts must record cap")
                if self.steps_run < self.cap:
                    raise TrajectoryStoppingError("step cap_bound receipts require steps_run >= cap")
            else:
                if self.bound_value is None or self.observed_value is None:
                    raise TrajectoryStoppingError(
                        "wall-clock cap_bound receipts require bound_value and observed_value"
                    )
                if self.observed_value < self.bound_value:
                    raise TrajectoryStoppingError("wall-clock cap_bound receipts require observed_value >= bound_value")
            if self.still_descending is None:
                raise TrajectoryStoppingError("cap_bound receipts must record still_descending")
        if self.stop_reason == "converged" and self.still_descending:
            raise TrajectoryStoppingError("converged receipts cannot be still_descending")

    def to_payload(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "steps_run": self.steps_run,
            "cap": self.cap,
            "still_descending": self.still_descending,
            "bound_kind": self.bound_kind,
            "bound_value": self.bound_value,
            "observed_value": self.observed_value,
        }


def build_cap_stop_receipt(
    *,
    stop_reason: CapStopReason,
    steps_run: int,
    cap: int | None,
    still_descending: bool | None,
    bound_kind: CapBoundKind = "steps",
    bound_value: float | None = None,
    observed_value: float | None = None,
) -> CapStopReceipt:
    """Build the canonical small stop-reason payload for capped solvers."""

    return CapStopReceipt(
        stop_reason=stop_reason,
        steps_run=steps_run,
        cap=cap,
        still_descending=still_descending,
        bound_kind=bound_kind,
        bound_value=bound_value,
        observed_value=observed_value,
    )


# Detection threshold provenance (constants-are-poison discipline): 2.0 sigma is the
# two-sided standard-normal ~95.4% detection convention, NOT a fitted constant.
# Operating anchors from the 2026-08-05 TP1 adjudications that this codifies: the
# acted-on extension fired at 2.44 sigma (w2 tail-20) and the strongest censored
# signal measured 6.23 sigma (w2 tail-40) while the run's own 5-epoch interval label
# said FLAT; the OFF arm's past-minimum ascent measured ~4 sigma. Sites with a
# measured noise floor of their own may override per call.
TAIL_SLOPE_SIGMA_THRESHOLD: Final = 2.0
# Default fit windows: the long window buys slope sensitivity (~T^1.5 in SNR), the
# short window buys recency; (40, 20) is the TP1 window-boundary practice pair
# (gate cadence 5 -> 8/4 points minimum). Units are the caller's step axis.
TAIL_SLOPE_DEFAULT_SPANS: Final = (40.0, 20.0)


@dataclass(frozen=True, slots=True)
class StaircaseStopConfig:
    """Extra evidence required before a smooth-fit stop can close a staircase.

    ``event_free_horizon_compute`` and ``event_score_delta`` are supplied by the
    caller's run geometry.  Loss flatness and decision resolution are estimated
    from the recorded tail itself, so a borrowed absolute loss epsilon is not
    introduced here.
    """

    min_eval_rows: int
    window_rows: int
    event_free_horizon_compute: float
    event_score_delta: float
    creep_score_per_compute: float
    sustained_erosion_windows: int
    sigma_threshold: float = TAIL_SLOPE_SIGMA_THRESHOLD

    def __post_init__(self) -> None:
        if self.min_eval_rows < 2:
            raise TrajectoryStoppingError("min_eval_rows must be at least 2")
        if self.window_rows < 3:
            raise TrajectoryStoppingError("window_rows must be at least 3")
        if self.min_eval_rows < self.window_rows:
            raise TrajectoryStoppingError("min_eval_rows must cover window_rows")
        for name, value in (
            ("event_free_horizon_compute", self.event_free_horizon_compute),
            ("event_score_delta", self.event_score_delta),
            ("creep_score_per_compute", self.creep_score_per_compute),
            ("sigma_threshold", self.sigma_threshold),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise TrajectoryStoppingError(f"{name} must be finite and positive")
        if self.sustained_erosion_windows < 1:
            raise TrajectoryStoppingError("sustained_erosion_windows must be positive")


@dataclass(frozen=True, slots=True)
class StaircaseStopDecision:
    """Typed production action layered on the canonical trajectory fit."""

    action: TrainingStopAction
    should_halt: bool
    trajectory_decision: StopDecision | None
    n_points: int
    event_free_compute: float
    event_free_horizon_compute: float
    loss_slope_per_compute: float | None
    loss_slope_se: float | None
    loss_flat: bool
    decision_noise_upper_bound_score_per_compute: float | None
    decision_noise_resolved: bool
    liveness_clear: bool
    erosion_sustained: bool
    boundary_kind: CapBoundKind | None
    blockers: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "should_halt": self.should_halt,
            "trajectory_decision": (
                None if self.trajectory_decision is None else self.trajectory_decision.to_payload()
            ),
            "n_points": self.n_points,
            "event_free_compute": self.event_free_compute,
            "event_free_horizon_compute": self.event_free_horizon_compute,
            "loss_slope_per_compute": self.loss_slope_per_compute,
            "loss_slope_se": self.loss_slope_se,
            "loss_flat": self.loss_flat,
            "decision_noise_upper_bound_score_per_compute": (self.decision_noise_upper_bound_score_per_compute),
            "decision_noise_resolved": self.decision_noise_resolved,
            "liveness_clear": self.liveness_clear,
            "erosion_sustained": self.erosion_sustained,
            "boundary_kind": self.boundary_kind,
            "blockers": list(self.blockers),
        }


TailSlopeVerdictKind = Literal[
    "censored_still_descending",
    "ascending_past_min",
    "converged_plateau",
]


@dataclass(frozen=True, slots=True)
class TailSlopeSpanFit:
    """One linear tail fit: objective vs step over the trailing ``span`` steps."""

    span: float
    n_points: int
    slope: float
    slope_se: float
    sigma: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "span": self.span,
            "n_points": self.n_points,
            "slope": self.slope,
            "slope_se": self.slope_se,
            "sigma": self.sigma,
        }


@dataclass(frozen=True, slots=True)
class TailSlopeVerdict:
    """Amendment-3 censored-cap adjudication at a window/cap boundary.

    The MEASURED tail-slope fits are the stopping authority; short-window
    per-gate classifier labels are not (they censored a real 6.2-sigma descent
    as FLAT twice on 2026-08-05).  ``censored_still_descending`` means the
    window cap cut a live descent (endpoint = censored, warm continuation owed);
    ``ascending_past_min`` means the endpoint is NOT the best state (adoption
    must use the recorded minimum, never the endpoint); ``converged_plateau``
    means no tail slope clears the detection threshold.
    """

    verdict: TailSlopeVerdictKind
    fits: tuple[TailSlopeSpanFit, ...]
    sigma_threshold: float
    min_step: float
    min_value: float
    end_step: float
    end_value: float
    endpoint_is_min: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "fits": [f.to_payload() for f in self.fits],
            "sigma_threshold": self.sigma_threshold,
            "min_step": self.min_step,
            "min_value": self.min_value,
            "end_step": self.end_step,
            "end_value": self.end_value,
            "endpoint_is_min": self.endpoint_is_min,
            "authority_note": (
                "measured tail-slope fit is the boundary authority; "
                "interval classifier labels are advisory (#874/#935 "
                "censored-cap genus, amendment-3 2026-08-05)"
            ),
        }


def _tail_fit(x: Sequence[float], y: Sequence[float], span: float) -> TailSlopeSpanFit | None:
    lo = x[-1] - span
    xs = [a for a in x if a >= lo]
    ys = [b for a, b in zip(x, y, strict=False) if a >= lo]
    if len(xs) < 3:
        return None
    intercept, slope = _linear_regression(xs, ys)
    xbar = sum(xs) / len(xs)
    sxx = sum((a - xbar) ** 2 for a in xs)
    pred = [intercept + slope * a for a in xs]
    resid = [b - p for b, p in zip(ys, pred, strict=False)]
    dof = max(1, len(xs) - 2)
    resid_sd = math.sqrt(sum(r * r for r in resid) / dof)
    se = resid_sd / math.sqrt(sxx) if sxx > 0 else 0.0
    if se > 0:
        sigma = abs(slope) / se
    else:
        sigma = math.inf if slope != 0.0 else 0.0
    return TailSlopeSpanFit(
        span=float(span), n_points=len(xs), slope=float(slope), slope_se=float(se), sigma=float(sigma)
    )


def adjudicate_tail_slope(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    spans: Sequence[float] = TAIL_SLOPE_DEFAULT_SPANS,
    sigma_threshold: float = TAIL_SLOPE_SIGMA_THRESHOLD,
) -> TailSlopeVerdict:
    """Adjudicate a trajectory at its window/cap boundary by MEASURED tail slopes.

    Decision rule (the 2026-08-05 amendment-3 practice, made canonical):
      1. any tail-span fit descending beyond ``sigma_threshold``
         -> ``censored_still_descending`` (the cap is censoring a live descent);
      2. else the shortest fitted span ascending beyond threshold while the
         endpoint sits above the trajectory minimum
         -> ``ascending_past_min`` (adopt the minimum, never the endpoint);
      3. else -> ``converged_plateau``.
    Requires >= 3 points; spans with < 3 points are dropped, and if every span
    drops, one fit over the full trajectory is used instead.
    """

    if len(steps) != len(values):
        raise TrajectoryStoppingError("steps and values must be equal length")
    if len(steps) < 3:
        raise TrajectoryStoppingError("tail-slope adjudication requires >= 3 points")
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    x = [float(steps[i]) for i in order]
    y = [float(values[i]) for i in order]
    fits = [f for f in (_tail_fit(x, y, s) for s in spans) if f is not None]
    if not fits:
        full = _tail_fit(x, y, x[-1] - x[0])
        if full is None:
            raise TrajectoryStoppingError("no fittable tail span (degenerate steps)")
        fits = [full]
    i_min = min(range(len(y)), key=lambda i: y[i])
    endpoint_is_min = i_min == len(y) - 1
    descending = any(f.slope < 0 and f.sigma >= sigma_threshold for f in fits)
    shortest = min(fits, key=lambda f: f.span)
    ascending = shortest.slope > 0 and shortest.sigma >= sigma_threshold and not endpoint_is_min
    if descending:
        verdict: TailSlopeVerdictKind = "censored_still_descending"
    elif ascending:
        verdict = "ascending_past_min"
    else:
        verdict = "converged_plateau"
    return TailSlopeVerdict(
        verdict=verdict,
        fits=tuple(fits),
        sigma_threshold=float(sigma_threshold),
        min_step=x[i_min],
        min_value=y[i_min],
        end_step=x[-1],
        end_value=y[-1],
        endpoint_is_min=endpoint_is_min,
    )


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
    for prev, cur in itertools.pairwise(out):
        if cur.compute <= prev.compute:
            raise TrajectoryStoppingError("trajectory compute coordinates must be strictly increasing")
    return tuple(out)


def _r2(y: Sequence[float], pred: Sequence[float]) -> float:
    ybar = sum(y) / len(y)
    sst = sum((v - ybar) ** 2 for v in y)
    if sst <= 0.0:
        return 0.0
    rss = sum((a - b) ** 2 for a, b in zip(y, pred, strict=False))
    return max(0.0, 1.0 - rss / sst)


def _linear_regression(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    den = sum((v - xbar) ** 2 for v in x)
    if den <= 0.0:
        raise TrajectoryStoppingError("degenerate fit x coordinates")
    slope = sum((a - xbar) * (b - ybar) for a, b in zip(x, y, strict=False)) / den
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


def _fit_decay_kind(
    points: tuple[TrajectoryPoint, ...], kind: Literal["geometric", "power_law"], cfg: TrajectoryStopConfig
) -> DecayFit | None:
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
    marginal_score = selected.projected_marginal_objective_gain_per_compute() * config.score_units_per_objective
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


def _slope_and_se(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    if len(x) != len(y) or len(x) < 3:
        raise TrajectoryStoppingError("slope evidence requires at least three paired rows")
    intercept, slope = _linear_regression(x, y)
    xbar = sum(x) / len(x)
    sxx = sum((value - xbar) ** 2 for value in x)
    pred = [intercept + slope * value for value in x]
    dof = max(1, len(x) - 2)
    residual_sd = math.sqrt(sum((actual - fitted) ** 2 for actual, fitted in zip(y, pred, strict=False)) / dof)
    slope_se = residual_sd / math.sqrt(sxx) if sxx > 0.0 else 0.0
    return float(slope), float(slope_se)


def _window_erosion_sustained(
    rows: Sequence[Mapping[str, Any]],
    config: StaircaseStopConfig,
) -> bool:
    required = config.window_rows + config.sustained_erosion_windows - 1
    if len(rows) < required:
        return False
    for offset in range(config.sustained_erosion_windows):
        end = len(rows) - offset
        window = rows[end - config.window_rows : end]
        x = [float(row["step"]) for row in window]
        objective = [float(row["objective_S"]) for row in window]
        losses = [float(row["loss"]) for row in window]
        objective_slope, _ = _slope_and_se(x, objective)
        loss_slope, _ = _slope_and_se(x, losses)
        if not (objective_slope > config.creep_score_per_compute and loss_slope < 0.0):
            return False
    return True


def evaluate_staircase_aware_stop(
    rows: Sequence[Mapping[str, Any]],
    trajectory_config: TrajectoryStopConfig,
    staircase_config: StaircaseStopConfig,
    *,
    safety_bound_compute: float | None = None,
    boundary_kind: CapBoundKind | None = None,
) -> StaircaseStopDecision:
    """Evaluate one production training action from durable per-eval rows.

    A smooth-fit semantic stop is only admitted after the run has remained free
    of a score-lattice event for the caller-derived horizon, the recorded loss
    tail is statistically flat, the trajectory's own slope-noise upper bound is
    below the marginal bar, and the weight-update liveness counters advance.
    A safety or wall-clock boundary reports ``QUEUE_RESUME`` when any of those
    gates remains unresolved.
    """

    if boundary_kind not in {None, "steps", "wall_clock_seconds"}:
        raise TrajectoryStoppingError("unknown boundary_kind")
    normalized = [dict(row) for row in rows]
    required = {"step", "objective_S", "loss", "weights_stepped", "accepted_batch_fraction"}
    for row in normalized:
        missing = required - set(row)
        if missing:
            raise TrajectoryStoppingError(f"staircase row missing required fields: {sorted(missing)}")
    normalized.sort(key=lambda row: float(row["step"]))
    for previous, current in itertools.pairwise(normalized):
        if float(current["step"]) <= float(previous["step"]):
            raise TrajectoryStoppingError("staircase steps must be strictly increasing")

    if normalized:
        event_origin = float(normalized[0]["step"])
        previous_objective = float(normalized[0]["objective_S"])
        for row in normalized[1:]:
            objective = float(row["objective_S"])
            if abs(objective - previous_objective) >= staircase_config.event_score_delta:
                event_origin = float(row["step"])
            previous_objective = objective
        event_free_compute = float(normalized[-1]["step"]) - event_origin
    else:
        event_free_compute = 0.0

    if len(normalized) < staircase_config.min_eval_rows:
        action: TrainingStopAction = "QUEUE_RESUME" if boundary_kind is not None else "INSUFFICIENT_HISTORY"
        return StaircaseStopDecision(
            action=action,
            should_halt=boundary_kind is not None,
            trajectory_decision=None,
            n_points=len(normalized),
            event_free_compute=event_free_compute,
            event_free_horizon_compute=staircase_config.event_free_horizon_compute,
            loss_slope_per_compute=None,
            loss_slope_se=None,
            loss_flat=False,
            decision_noise_upper_bound_score_per_compute=None,
            decision_noise_resolved=False,
            liveness_clear=False,
            erosion_sustained=False,
            boundary_kind=boundary_kind,
            blockers=("min_eval_rows_not_met",),
        )

    trajectory = evaluate_trajectory_stop(
        [TrajectoryPoint(float(row["step"]), float(row["objective_S"])) for row in normalized],
        trajectory_config,
        safety_bound_compute=safety_bound_compute,
    )
    tail = normalized[-staircase_config.window_rows :]
    x = [float(row["step"]) for row in tail]
    objective = [float(row["objective_S"]) for row in tail]
    losses = [float(row["loss"]) for row in tail]
    loss_slope, loss_slope_se = _slope_and_se(x, losses)
    _, objective_slope_se = _slope_and_se(x, objective)
    loss_flat = abs(loss_slope) <= staircase_config.sigma_threshold * loss_slope_se
    noise_upper = staircase_config.sigma_threshold * objective_slope_se
    noise_resolved = noise_upper < trajectory_config.marginal_score_gain_per_compute
    liveness_clear = all(
        math.isfinite(float(row["accepted_batch_fraction"])) and 0.0 < float(row["accepted_batch_fraction"]) <= 1.0
        for row in tail
    ) and all(
        int(current["weights_stepped"]) > int(previous["weights_stepped"])
        for previous, current in itertools.pairwise(tail)
    )
    erosion_sustained = _window_erosion_sustained(normalized, staircase_config)
    event_horizon_clear = event_free_compute >= staircase_config.event_free_horizon_compute
    semantic_candidate = trajectory.stop_reason in {
        "converged_projected",
        "marginal_below_bar",
    }

    blockers: list[str] = []
    if not event_horizon_clear:
        blockers.append("event_free_horizon_not_met")
    if not loss_flat:
        blockers.append("loss_tail_not_flat")
    if not noise_resolved:
        blockers.append("decision_noise_overlaps_marginal_bar")
    if not liveness_clear:
        blockers.append("weight_update_liveness_not_clear")

    if erosion_sustained:
        action = "ROLLBACK_OR_RETREAT"
    elif semantic_candidate and not blockers:
        action = "STOP_CONVERGED"
    elif boundary_kind is not None or trajectory.stop_reason == "safety_bound_REPORTED":
        action = "QUEUE_RESUME"
    else:
        action = "CONTINUE"
    return StaircaseStopDecision(
        action=action,
        should_halt=action in {"STOP_CONVERGED", "ROLLBACK_OR_RETREAT", "QUEUE_RESUME"},
        trajectory_decision=trajectory,
        n_points=len(normalized),
        event_free_compute=event_free_compute,
        event_free_horizon_compute=staircase_config.event_free_horizon_compute,
        loss_slope_per_compute=loss_slope,
        loss_slope_se=loss_slope_se,
        loss_flat=loss_flat,
        decision_noise_upper_bound_score_per_compute=noise_upper,
        decision_noise_resolved=noise_resolved,
        liveness_clear=liveness_clear,
        erosion_sustained=erosion_sustained,
        boundary_kind=boundary_kind,
        blockers=tuple(blockers),
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
    "TAIL_SLOPE_DEFAULT_SPANS",
    "TAIL_SLOPE_SIGMA_THRESHOLD",
    "TRAJECTORY_STOPPING_LAW_REF",
    "CapBoundKind",
    "CapStopReason",
    "CapStopReceipt",
    "DecayFit",
    "DepthAllocation",
    "ProjectionInterval",
    "StaircaseStopConfig",
    "StaircaseStopDecision",
    "StopDecision",
    "StopReason",
    "TrainingStopAction",
    "TrajectoryPoint",
    "TrajectoryStopConfig",
    "TrajectoryStoppingError",
    "allocate_adaptive_depths",
    "build_cap_stop_receipt",
    "byte_score_units",
    "evaluate_staircase_aware_stop",
    "evaluate_trajectory_stop",
    "fit_decay_models",
    "gap_fraction_score_bar",
    "projection_interval",
    "seg_flip_score_units",
]
