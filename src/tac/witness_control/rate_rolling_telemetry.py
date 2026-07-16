# SPDX-License-Identifier: MIT
"""Rolling-average rate-proxy telemetry producer (#408/#404 · operator FEED-ratetelemetry).

Operator directive (verbatim, 2026-07-15): rate is *non-monotonic* during training
(the witness reallocates capacity + re-quantizes weights, so the byte-close estimate
wobbles and may TREND DOWN late). Telemetry must therefore:

  (1) report a ROLLING AVERAGE of the rate proxy (windowed mean / EMA of
      ``weight_entropy_bits`` + periodic byte-close ``archive_bytes`` estimate;
      reuse the ``costate_estimator`` window=5 pattern), NOT the instantaneous value;
  (2) TOLERATE small growth + fluctuation within a band (no signal inside +/-eps of
      the rolling mean);
  (3) raise a GRADUATED SOFT signal only on SUSTAINED rolling growth beyond the band
      (``WITHIN`` -> ``DRIFTING_UP`` -> ``SUSTAINED_GROWTH``), which INFORMS the
      costate controller / operator -- it **NEVER kills the run**.

This is score-neutral read-only observability (defaults ON per the "off is a tracked
queue" orphan-signal reconciliation), consistent with the spike-guard median-freeze
lesson (never kill on a transient), the confound immune-system L1 (loud-not-halt for
soft signals), and "guard NEVER kills the control-plane".

PRODUCER half only: pure functions over the trainer's already-materialized rate-proxy
series. It owns no optimizer/EMA/model/scorer state and allocates nothing persistent.
Trainer emission-site wiring is QUEUED behind the live dry-start (pid 31576) -- see
``tac.witness_dsl.constants_telemetry_build_wave_20260715.TRAINER_WIREIN_QUEUE``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

RATE_ROLLING_SCHEMA = "witness_rate_rolling.v1"

# Graduated soft-signal states (ordered by severity; NONE is a kill by construction).
SIGNAL_WITHIN = "WITHIN"
SIGNAL_DRIFTING_UP = "DRIFTING_UP"
SIGNAL_SUSTAINED_GROWTH = "SUSTAINED_GROWTH"
SIGNAL_STATES: tuple[str, ...] = (SIGNAL_WITHIN, SIGNAL_DRIFTING_UP, SIGNAL_SUSTAINED_GROWTH)

#: local-fit window (matches costate_estimator DEFAULT_LOCAL_WINDOW / the canonical
#: monitor window=5). The rate proxy is non-monotonic, so the LOCAL rolling mean is
#: the honest series to compare against, never the instantaneous value.
DEFAULT_WINDOW = 5

#: relative tolerance band (+/-eps) around the reference rolling mean. Within this band
#: NO signal fires -- small growth + fluctuation is expected and tolerated.
DEFAULT_BAND_EPS = 0.02

#: how many consecutive above-band rolling steps promote DRIFTING_UP -> SUSTAINED_GROWTH.
#: SUSTAINED is a slow, deliberate escalation (never a transient), mirroring the
#: spike-guard median-freeze "never kill on a transient" lesson.
DEFAULT_SUSTAIN_COUNT = 3


def _finite(value: float, *, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"rate_rolling_telemetry: non-finite {name}={value!r} -- fail loud")
    return v


def rolling_mean(values: Sequence[float], window: int = DEFAULT_WINDOW) -> float:
    """Mean of the last ``window`` values (``window<=0`` -> mean of all). Fail-loud on NaN/Inf.

    Empty input raises (the caller must have at least one proxy sample before emitting).
    """
    if not values:
        raise ValueError("rate_rolling_telemetry.rolling_mean: empty rate-proxy series")
    tail = list(values[-int(window):]) if (window and window > 0) else list(values)
    finite = [_finite(v, name="rate_proxy") for v in tail]
    return float(sum(finite) / len(finite))


@dataclass(frozen=True)
class RateRollingBaseline:
    """The t0 rolling mean the drift ratio is measured against (resume-safe).

    Capture at the FIRST emission of a run (or restore from the first emitted row via
    :func:`baseline_from_row` after a resume) so ``rel_from_t0`` is stable across
    crash-resume -- never re-anchored mid-run.
    """

    epoch: int
    rolling_mean: float = 0.0
    #: persisted proxy tail so the rolling mean + sustained-growth count resume
    #: continuously across a crash boundary (never a discontinuity).
    proxy_tail: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if int(self.epoch) < 0:
            raise ValueError(f"RateRollingBaseline.epoch must be >= 0, got {self.epoch!r}")
        rm = float(self.rolling_mean)
        if not math.isfinite(rm) or rm < 0.0:
            raise ValueError(f"RateRollingBaseline: bad rolling_mean {self.rolling_mean!r}")
        for v in self.proxy_tail:
            if not math.isfinite(float(v)):
                raise ValueError(f"RateRollingBaseline: non-finite proxy_tail entry {v!r}")


def baseline_from_row(row: Mapping[str, object]) -> RateRollingBaseline:
    """Rebuild the t0 baseline from a previously emitted row (crash-resume path)."""
    if row.get("schema") != RATE_ROLLING_SCHEMA:
        raise ValueError(
            f"baseline_from_row: row schema {row.get('schema')!r} != {RATE_ROLLING_SCHEMA!r}"
        )
    tail_obj = row.get("proxy_tail")
    tail = tuple(float(v) for v in tail_obj) if isinstance(tail_obj, (list, tuple)) else ()
    return RateRollingBaseline(
        epoch=int(row["ep"]),
        rolling_mean=float(row["rolling_avg"]),
        proxy_tail=tail,
    )


def classify_drift_signal(
    values: Sequence[float],
    *,
    window: int = DEFAULT_WINDOW,
    band_eps: float = DEFAULT_BAND_EPS,
    sustain_count: int = DEFAULT_SUSTAIN_COUNT,
) -> tuple[str, int]:
    """Graduated soft signal from a rate-proxy series (NEVER a kill).

    The reference is the rolling mean lagged by one ``window`` (the mean of the window
    ENDING one window back). The current rolling mean is compared to it:

      * within ``+/-band_eps`` of the reference          -> ``WITHIN``
      * above ``+band_eps`` but not yet sustained         -> ``DRIFTING_UP``
      * above ``+band_eps`` for ``sustain_count`` steps   -> ``SUSTAINED_GROWTH``

    Growth BELOW the band (rate trending DOWN) is expected + healthy -> ``WITHIN``.

    Returns ``(signal, above_band_run_len)``. Insufficient history -> ``(WITHIN, 0)``.
    """
    if band_eps < 0.0:
        raise ValueError(f"classify_drift_signal: band_eps must be >= 0, got {band_eps!r}")
    w = int(window) if (window and window > 0) else DEFAULT_WINDOW
    n = len(values)
    if n < 2 * w:
        # not enough history for a lagged-reference comparison -> no signal
        return SIGNAL_WITHIN, 0

    def _above_band_at(end: int) -> bool:
        # rolling mean of the window ENDING at index `end` (exclusive) vs the window
        # ending one window earlier.
        cur = rolling_mean(values[end - w:end], window=w)
        ref = rolling_mean(values[end - 2 * w:end - w], window=w)
        if ref <= 0.0:
            return False
        return (cur / ref - 1.0) > band_eps

    # count the consecutive above-band run ending at the latest sample
    run = 0
    end = n
    while end - 2 * w >= 0 and _above_band_at(end):
        run += 1
        end -= 1
    if run == 0:
        return SIGNAL_WITHIN, 0
    if run >= int(sustain_count):
        return SIGNAL_SUSTAINED_GROWTH, run
    return SIGNAL_DRIFTING_UP, run


def rate_rolling_row(
    epoch: int,
    proxy_series: Sequence[float],
    *,
    window: int = DEFAULT_WINDOW,
    band_eps: float = DEFAULT_BAND_EPS,
    sustain_count: int = DEFAULT_SUSTAIN_COUNT,
    baseline: RateRollingBaseline | None = None,
) -> dict[str, object]:
    """One emission-ready ``rate_rolling`` telemetry row (score-neutral, read-only).

    ``proxy_series`` is the run's rate-proxy history (``weight_entropy_bits`` +
    periodic byte-close ``archive_bytes`` estimate), most-recent last. On resume the
    trainer prepends ``baseline.proxy_tail`` so the rolling mean + sustained-growth
    run length are continuous across the crash boundary.

    Fields: ``rolling_avg`` (windowed mean), ``instant`` (latest raw proxy, for
    contrast), ``drift_signal`` (WITHIN/DRIFTING_UP/SUSTAINED_GROWTH -- INFORMS never
    kills), ``above_band_run`` (consecutive above-band steps), ``rel_from_t0`` (vs the
    resume-safe baseline rolling mean), and a persisted ``proxy_tail`` for resume.

    The trainer wire-in emits this at verdict cadence adjacent to the weight_norm row
    and routes it through the same run.log JSON stream every telemetry row uses. It is
    advisory: the costate controller / operator READS ``drift_signal``; no code path
    halts, aborts, reverts, or clamps on it.
    """
    if not proxy_series:
        raise ValueError("rate_rolling_row: empty proxy_series")
    series = [_finite(v, name="rate_proxy") for v in proxy_series]
    avg = rolling_mean(series, window=window)
    signal, run = classify_drift_signal(
        series, window=window, band_eps=band_eps, sustain_count=sustain_count
    )
    keep = int(window) if (window and window > 0) else DEFAULT_WINDOW
    proxy_tail = tuple(series[-(2 * keep):])
    row: dict[str, object] = {
        "stage": "rate_rolling",
        "schema": RATE_ROLLING_SCHEMA,
        "ep": int(epoch),
        "rolling_avg": avg,
        "instant": series[-1],
        "window": keep,
        "band_eps": float(band_eps),
        "drift_signal": signal,
        "above_band_run": run,
        # advisory-only contract: this signal NEVER halts/aborts/reverts the run.
        "informs_only": True,
        "proxy_tail": proxy_tail,
    }
    if baseline is not None and baseline.rolling_mean > 0.0:
        row["rel_from_t0"] = avg / baseline.rolling_mean - 1.0
        row["baseline_epoch"] = int(baseline.epoch)
    return row


__all__ = [
    "RATE_ROLLING_SCHEMA",
    "SIGNAL_WITHIN",
    "SIGNAL_DRIFTING_UP",
    "SIGNAL_SUSTAINED_GROWTH",
    "SIGNAL_STATES",
    "DEFAULT_WINDOW",
    "DEFAULT_BAND_EPS",
    "DEFAULT_SUSTAIN_COUNT",
    "RateRollingBaseline",
    "baseline_from_row",
    "rolling_mean",
    "classify_drift_signal",
    "rate_rolling_row",
]
