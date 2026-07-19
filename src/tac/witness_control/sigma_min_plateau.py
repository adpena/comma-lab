"""Pose-finish CONDITIONING gate — ROLLING-SLOPE plateau on DE-NOISED σ_min (the A-1 FIX).

Operator binding (ORCHESTRATION_LEDGER §205-211, verbatim): *"pose must not be fired for joint
descent until optimal it needs dseg to be sufficiently conditioned first."* Pose-finish engagement is a
d_seg-CONDITIONING EVENT, never an epoch. This module is the DETECTOR that decides when that event has
occurred, read off the ξ→PoseNet Jacobian conditioning σ_min(J_ξ) telemetry (``jacobian_basin`` sibling).

Requirements source: ``.omx/research/t5_crucible2/SYNTHESIS_v3_v752_20260709.md`` §A.4 (the REPAIRED
pose-gate spec), folding P4-M1 + the P5 v3 AMENDMENTs 1-3. The design has THREE prior-broken ancestors —
implement the SEALED form, not them:

  * A-1 BREAK: the ABSOLUTE ``σ* = √(C/(δ_seg·λ_min(F)))`` floor is UNREACHABLE (RED-TEAM MEASURED
    ≥14.14 vs measured σ_min ≤ 0.063 — 2-6 orders). σ* is DEMOTED to ADVISORY telemetry only
    (``sigma_star_advisory``), NEVER a firing conjunct.
  * v2 FORMULATION: the exp-asymptote FIT plateau (``σ_min ≥ σ_min(∞)·(1−δ)``) is DEGENERATE on the only
    real σ_min series (P4-M1 MEASURED: 31 rows, CV-0.21 noisy OSCILLATION with a RISING drift, s_inf CI
    ±891%, covariance non-estimable). NOT resurrected here.
  * R1-seal fix: hysteresis floor 3 (⌈W_settle⌉).

The SEALED PRIMARY criterion (v3 AMENDMENT-1): a scale-free ROLLING-MEAN-SLOPE ≈ 0 over a SETTLE WINDOW,
computed on a NOISE-REDUCED (EMA-smoothed) σ_min series. The detector fires when the smoothed series STOPS
TRENDING (|rolling rel-slope| within a near-zero band) for ``hysteresis`` consecutive windows AND the slope
is NON-RISING — so it does NOT fire while σ_min is still RISING (conditioning still improving), the right
call on the stopped run. A NOISE / FIT-QUALITY GUARD (the fitted slope's own rel-stderr must resolve to
within the flat band) falls back to ship-banked-R1 on a degenerate signal: never fire on noise, never fire
spuriously on flat noise.

DISCIPLINE:
  * PURE MATH + dataclasses. No IO (except the read-only ``load_sigma_min_series_from_jsonl`` backtest
    helper), no trainer imports, no MLX, no actuation. Deterministic: same series -> same verdict.
  * SCORE-NEUTRAL as an OBSERVER (reads σ_min telemetry, mutates nothing). The ENGAGE decision it feeds is
    a score-affecting lever (``--pose-finish-engage-on sigma_min_plateau``, DEFAULT-OFF); the detector
    itself never writes weights/loss/optimizer/RNG/EMA.
  * NON-PROMOTABLE: σ_min is an MLX-advisory quantity; this gate governs TIMING, not a score. Every row is
    axis-tagged ``[macOS-MLX advisory] NON-PROMOTABLE``.
  * NEVER-BLOCK-LAUNCH: an untrusted (canary-fail) / degenerate / never-fired gate ⇒ ship the banked R1
    dxi (``should_ship_banked_r1``) + a LOUD disengaged alarm — pose-DISENGAGED is a valid terminal state,
    never a run-blocker (SYNTHESIS §A.4 Repair 2b/4).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tac.witness_control.costate_estimator import (
    DEFAULT_STALL_REL_EPS,
    MIN_ROWS_FOR_SLOPE,
    slope_with_stderr,
)

# --------------------------------------------------------------------------- #
# DERIVED constants (value-provenance ladder — no bare literals).
# --------------------------------------------------------------------------- #
#: W_settle = 4.6·τ_EMA = 2.6 ep (SYNTHESIS §A.5, S3-derived) → ⌈W_settle⌉ = 3, the R1-seal hysteresis
#: floor. DERIVED-AT-CONFIG. The single S3 quantity all three window knobs below descend from.
W_SETTLE_CEIL: int = 3
#: EMA de-noise span in σ_min POINTS = ⌈W_settle⌉ (S3). α = 2/(span+1) = 0.5 (pandas-ewm span convention).
#: Cuts the MEASURED CV-0.21 oscillation (P4-M1) BEFORE any statistic. DERIVED-AT-CONFIG (S3 §A.5).
DEFAULT_EMA_SPAN: int = W_SETTLE_CEIL
#: Rolling |rel-slope| band ("≈ 0"): the campaign's ONE calibrated FLAT gate
#: (``costate_estimator.DEFAULT_STALL_REL_EPS`` = 3e-4 per-epoch |slope|/level), reused-for-coherence
#: EXACTLY as ``verdict_trend_alarm`` reuses it (partitions the space: ≤3e-4 = flat; >3e-4 = trending).
#: MEASURED-ANCHOR (costate_estimator), reused-for-coherence.
DEFAULT_FLAT_REL_BAND: float = DEFAULT_STALL_REL_EPS
#: # σ_min points per rolling slope = ⌈W_settle⌉ = 3 = ``MIN_ROWS_FOR_SLOPE`` (the minimum for a defined
#: OLS slope stderr, df ≥ 1). DERIVED-AT-CONFIG (S3 W_settle ∧ the stderr df floor).
DEFAULT_SETTLE_WINDOW: int = max(W_SETTLE_CEIL, MIN_ROWS_FOR_SLOPE)
#: consecutive flat windows required = ⌈W_settle⌉ floor 3 (SYNTHESIS §A.4 hysteresis + R1-seal fix).
#: DERIVED-AT-CONFIG.
DEFAULT_HYSTERESIS: int = W_SETTLE_CEIL
#: cap on the persisted σ_min series length (resume sidecar). Far exceeds any settle+hysteresis need
#: (~150 cadenced points over a 2500-ep run); bounds the additive checkpoint key. DERIVED (bound, not a
#: tuning knob).
_MAX_PERSISTED_POINTS: int = 512

# ── classification sentinels ──
PLATEAU_FIRED = "PLATEAU_FIRED"              # de-noised σ_min stopped trending, hysteresis met, guard OK
NOT_PLATEAUED = "NOT_PLATEAUED"              # still trending / not enough consecutive flat windows
DEGENERATE_GUARD_TRIPPED = "DEGENERATE_GUARD_TRIPPED"   # cannot distinguish plateau from oscillation → ship banked R1
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"      # fewer than settle_window + hysteresis - 1 points
# ── crest sentinels (SPEC_v10 §13.2 fire-on-crest alternative; arm B 2026-07-17) ──
CREST_FIRED = "CREST_FIRED"                  # a RESOLVED rising phase, then slope sign-change held for hysteresis
NOT_CRESTED = "NOT_CRESTED"                  # no resolved rising phase yet / still rising / hysteresis unmet

#: engage-detector modes: 'plateau' (the SEALED incumbent) / 'crest' (smoothed σ_min slope
#: SIGN-CHANGE = conditioning peak; SPEC_v10 §13.2 — live c2 anchor: σ_min 0.0010→0.0068 still
#: climbing +15%/ep at ep798, so a plateau gate keeps waiting while a crest gate fires the moment
#: conditioning stops improving after a genuine rising phase).
MODE_PLATEAU = "plateau"
MODE_CREST = "crest"

#: the resume-registry key prefix for the detector's persisted state (must start with '__').
RESUME_PREFIX = "__posegate_"


@dataclass(frozen=True)
class SigmaMinPlateauConfig:
    """The rolling-slope plateau detector knobs (SYNTHESIS §A.4/§B pose block). Every default DERIVED
    (see the module constants); each is a config knob so the launcher can tune per the DENSIFY amendment
    without a code edit. ``validate()`` fail-closes on a nonsensical config."""

    ema_span: int = DEFAULT_EMA_SPAN
    flat_rel_band: float = DEFAULT_FLAT_REL_BAND
    settle_window: int = DEFAULT_SETTLE_WINDOW
    hysteresis: int = DEFAULT_HYSTERESIS

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.ema_span < 1:
            problems.append(f"SigmaMinPlateauConfig: ema_span must be >= 1, got {self.ema_span}")
        if not (self.flat_rel_band > 0.0):
            problems.append(
                f"SigmaMinPlateauConfig: flat_rel_band must be > 0, got {self.flat_rel_band}")
        if self.settle_window < MIN_ROWS_FOR_SLOPE:
            problems.append(
                f"SigmaMinPlateauConfig: settle_window must be >= {MIN_ROWS_FOR_SLOPE} (a defined OLS "
                f"stderr needs df >= 1), got {self.settle_window}")
        if self.hysteresis < 1:
            problems.append(f"SigmaMinPlateauConfig: hysteresis must be >= 1, got {self.hysteresis}")
        return problems

    @property
    def min_points(self) -> int:
        """Points needed to evaluate ``hysteresis`` consecutive length-``settle_window`` windows (slid by
        one): ``settle_window + hysteresis - 1``."""
        return int(self.settle_window) + int(self.hysteresis) - 1


@dataclass(frozen=True)
class SigmaMinPlateauVerdict:
    """The detector's decision over the current de-noised σ_min series. Every field MEASURED from the
    series (no fabricated number). ``fired()`` is the ENGAGE signal; ``should_ship_banked_r1()`` is the
    degenerate fall-back."""

    classification: str
    n_points: int
    settle_window: int
    hysteresis: int
    consecutive_flat: int          # how many of the last `hysteresis` windows read flat+non-rising
    latest_rel_slope_per_ep: float | None
    latest_rel_stderr_per_ep: float | None
    flat_rel_band: float
    smoothed_sigma_min_latest: float | None
    epoch_latest: int | None
    latched_fired_epoch: int | None   # >=0 once latched (monotone engagement)
    sigma_star_advisory: float | None   # ADVISORY sideband ONLY — never a firing conjunct (A-1)
    reason: str
    evidence: tuple[str, ...]

    def fired(self) -> bool:
        # PLATEAU_FIRED (incumbent) or CREST_FIRED (SPEC_v10 §13.2 fire-on-crest) — both are the
        # ENGAGE signal; which one a run can produce is fixed by the detector's single mode.
        return self.classification in (PLATEAU_FIRED, CREST_FIRED)

    def should_ship_banked_r1(self) -> bool:
        """True when the signal is DEGENERATE (guard tripped) — the caller ships the banked R1 dxi and
        does NOT engage the conditioning gate (SYNTHESIS §A.4 Repair 2b: never fire on a degenerate
        signal). A NOT_PLATEAUED / INSUFFICIENT verdict is 'not yet', NOT a fall-back."""
        return self.classification == DEGENERATE_GUARD_TRIPPED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        return d


# --------------------------------------------------------------------------- #
# Pure de-noise + slope primitives.
# --------------------------------------------------------------------------- #
def ema_smooth(values: Sequence[float], span: int) -> list[float]:
    """EMA over the σ_min POINT sequence (point-indexed, not epoch-indexed), span-convention
    ``α = 2/(span+1)``. Cuts the measured CV-0.21 oscillation BEFORE any statistic. Deterministic; the
    first point seeds the shadow (no look-ahead)."""
    a = 2.0 / (float(max(1, int(span))) + 1.0)
    out: list[float] = []
    s: float | None = None
    for v in values:
        fv = float(v)
        s = fv if s is None else a * fv + (1.0 - a) * s
        out.append(float(s))
    return out


def _rel_slope_and_stderr(eps: Sequence[float], ys: Sequence[float]) -> tuple[float, float | None]:
    """(rel-slope, rel-stderr) per epoch over (eps, ys): slope/|level| and stderr/|level|. ``level`` is
    the window mean. Returns (0.0, None) when the level ≈ 0 (never div-by-0) or the stderr is undefined."""
    fit = slope_with_stderr([float(e) for e in eps], [float(y) for y in ys])
    level = (sum(float(y) for y in ys) / len(ys)) if ys else 0.0
    if not (abs(level) > 0.0):
        return 0.0, None
    rel_slope = fit.slope / abs(level)
    rel_stderr = (fit.stderr / abs(level)) if fit.stderr is not None else None
    return float(rel_slope), (float(rel_stderr) if rel_stderr is not None else None)


def sigma_star_advisory(c_pose_grad_cap: float, delta_seg: float,
                        lambda_min_f: float | None) -> float | None:
    """The ABSOLUTE conditioning floor ``σ* = √(C/(δ_seg·λ_min(F)))`` (A-1). ADVISORY sideband ONLY —
    RED-TEAM MEASURED ≥14.14 vs σ_min ≤ 0.063 (unreachable by 2-6 orders), so it is logged as a WATCH
    note and is NEVER a firing conjunct. Returns None when ``λ_min(F)`` is unknown (the annulus probe is
    OFF the launch-blocking path — SYNTHESIS §A.4)."""
    if lambda_min_f is None or not (lambda_min_f > 0.0) or not (delta_seg > 0.0):
        return None
    return float(math.sqrt(float(c_pose_grad_cap) / (float(delta_seg) * float(lambda_min_f))))


# --------------------------------------------------------------------------- #
# The core detector (PURE over the series + the latch).
# --------------------------------------------------------------------------- #
def evaluate_plateau(
    eps: Sequence[float], smins: Sequence[float], cfg: SigmaMinPlateauConfig,
    *, latched_fired_epoch: int = -1, sigma_star: float | None = None,
) -> SigmaMinPlateauVerdict:
    """The SEALED rolling-slope plateau detector (PURE). ``eps``/``smins`` are the raw σ_min series
    (epoch, median σ_min). ``latched_fired_epoch`` >= 0 short-circuits to a LATCHED fire (monotone
    engagement). ``sigma_star`` is the advisory sideband (never gates)."""
    eps = [float(e) for e in eps]
    raw = [float(s) for s in smins]
    n = len(raw)
    latched = latched_fired_epoch is not None and latched_fired_epoch >= 0

    if latched:
        sm = ema_smooth(raw, cfg.ema_span) if raw else []
        return SigmaMinPlateauVerdict(
            classification=PLATEAU_FIRED, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=cfg.hysteresis,
            latest_rel_slope_per_ep=None, latest_rel_stderr_per_ep=None,
            flat_rel_band=cfg.flat_rel_band, smoothed_sigma_min_latest=(sm[-1] if sm else None),
            epoch_latest=(int(eps[-1]) if eps else None), latched_fired_epoch=int(latched_fired_epoch),
            sigma_star_advisory=sigma_star,
            reason=f"pose-finish conditioning gate LATCHED at ep{latched_fired_epoch} (monotone engagement)",
            evidence=(f"latched fire @ep{latched_fired_epoch}; σ_min series now {n} points",))

    if n < cfg.min_points:
        return SigmaMinPlateauVerdict(
            classification=INSUFFICIENT_DATA, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=0, latest_rel_slope_per_ep=None,
            latest_rel_stderr_per_ep=None, flat_rel_band=cfg.flat_rel_band,
            smoothed_sigma_min_latest=(ema_smooth(raw, cfg.ema_span)[-1] if raw else None),
            epoch_latest=(int(eps[-1]) if eps else None), latched_fired_epoch=None,
            sigma_star_advisory=sigma_star,
            reason=(f"only {n} σ_min point(s); need >= {cfg.min_points} "
                    f"(settle_window {cfg.settle_window} + hysteresis {cfg.hysteresis} - 1) — wait/densify"),
            evidence=(f"{n} σ_min points, min_points={cfg.min_points}",))

    sm = ema_smooth(raw, cfg.ema_span)
    w = int(cfg.settle_window)
    band = float(cfg.flat_rel_band)

    # Evaluate the flat+non-rising predicate at the last `hysteresis` window END positions (windows slid
    # by one): window k ends at index (n-1-k), spans [end-w+1 .. end]. consecutive_flat counts how many
    # of the most-recent `hysteresis` windows read flat AND non-rising.
    flats: list[bool] = []
    latest_rel_slope: float | None = None
    latest_rel_stderr: float | None = None
    for k in range(int(cfg.hysteresis)):
        end = (n - 1) - k
        lo = end - w + 1
        we, wy = eps[lo:end + 1], sm[lo:end + 1]
        rel_slope, rel_stderr = _rel_slope_and_stderr(we, wy)
        if k == 0:
            latest_rel_slope, latest_rel_stderr = rel_slope, rel_stderr
        # flat = |rel-slope| within the band (stopped trending); non-rising = rel-slope <= +band (not a
        # material positive/rising trend, so a still-RISING σ_min does NOT read as plateaued).
        flat = (abs(rel_slope) <= band) and (rel_slope <= band)
        flats.append(bool(flat))
    consecutive_flat = 0
    for f in flats:                       # flats[0] = latest window; count the leading run
        if f:
            consecutive_flat += 1
        else:
            break

    # NOISE / FIT-QUALITY GUARD (SYNTHESIS §A.4): the LATEST window's fitted slope must be RESOLVED to
    # within the flat band (rel-stderr <= band). If it is not, we cannot distinguish a plateau from an
    # oscillation → DEGENERATE → ship banked R1. Derived (reuses the flat band; no new constant).
    guard_degenerate = (latest_rel_stderr is None) or (latest_rel_stderr > band)

    ev = (
        f"n={n} σ_min points; EMA span {cfg.ema_span} (α={2.0 / (cfg.ema_span + 1.0):.3f}); "
        f"latest de-noised σ_min={sm[-1]:.6g}@ep{int(eps[-1])}",
        f"latest-window rel-slope {(latest_rel_slope if latest_rel_slope is not None else float('nan')):+.2e}/ep "
        f"vs flat band {band:.1e}; rel-stderr "
        f"{('%.2e' % latest_rel_stderr) if latest_rel_stderr is not None else 'undefined'} "
        f"(guard: rel-stderr must be <= band to trust the plateau)",
        f"consecutive flat+non-rising windows = {consecutive_flat}/{cfg.hysteresis}",
    )

    if guard_degenerate:
        return SigmaMinPlateauVerdict(
            classification=DEGENERATE_GUARD_TRIPPED, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=consecutive_flat,
            latest_rel_slope_per_ep=latest_rel_slope, latest_rel_stderr_per_ep=latest_rel_stderr,
            flat_rel_band=band, smoothed_sigma_min_latest=sm[-1], epoch_latest=int(eps[-1]),
            latched_fired_epoch=None, sigma_star_advisory=sigma_star,
            reason=("NOISE/FIT-QUALITY GUARD tripped: the de-noised σ_min slope cannot be resolved to "
                    f"within the flat band (rel-stderr "
                    f"{('%.2e' % latest_rel_stderr) if latest_rel_stderr is not None else 'undefined'} "
                    f"> band {band:.1e}) — CANNOT distinguish plateau from oscillation. FALL BACK to "
                    "ship-banked-R1 (never fire on a degenerate signal)."),
            evidence=ev)

    if consecutive_flat >= int(cfg.hysteresis):
        return SigmaMinPlateauVerdict(
            classification=PLATEAU_FIRED, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=consecutive_flat,
            latest_rel_slope_per_ep=latest_rel_slope, latest_rel_stderr_per_ep=latest_rel_stderr,
            flat_rel_band=band, smoothed_sigma_min_latest=sm[-1], epoch_latest=int(eps[-1]),
            latched_fired_epoch=None, sigma_star_advisory=sigma_star,
            reason=(f"de-noised σ_min PLATEAUED: |rolling rel-slope| <= {band:.1e} AND non-rising for "
                    f"{consecutive_flat} >= {cfg.hysteresis} consecutive settle windows — trunk "
                    "conditioning has STOPPED IMPROVING (sufficiently conditioned). ENGAGE pose-finish."),
            evidence=ev)

    return SigmaMinPlateauVerdict(
        classification=NOT_PLATEAUED, n_points=n, settle_window=cfg.settle_window,
        hysteresis=cfg.hysteresis, consecutive_flat=consecutive_flat,
        latest_rel_slope_per_ep=latest_rel_slope, latest_rel_stderr_per_ep=latest_rel_stderr,
        flat_rel_band=band, smoothed_sigma_min_latest=sm[-1], epoch_latest=int(eps[-1]),
        latched_fired_epoch=None, sigma_star_advisory=sigma_star,
        reason=(f"not plateaued: only {consecutive_flat}/{cfg.hysteresis} consecutive flat+non-rising "
                f"windows (latest rel-slope "
                f"{(latest_rel_slope if latest_rel_slope is not None else float('nan')):+.2e}/ep vs band "
                f"{band:.1e}) — σ_min still trending (conditioning still improving = not-yet-plateaued)."),
        evidence=ev)


def evaluate_crest(
    eps: Sequence[float], smins: Sequence[float], cfg: SigmaMinPlateauConfig,
    *, latched_fired_epoch: int = -1, sigma_star: float | None = None,
) -> SigmaMinPlateauVerdict:
    """The FIRE-ON-CREST detector (SPEC_v10 §13.2; PURE): smoothed σ_min ROLLING-SLOPE SIGN-CHANGE
    = the conditioning PEAK, with hysteresis.

    Fires when BOTH hold on the EMA-smoothed series (same de-noise + window machinery as the sealed
    plateau detector — no new constants):

      1. a RESOLVED RISING phase was seen in some earlier window (rel-slope > ``flat_rel_band`` AND
         the fitted slope's rel-stderr <= the slope, i.e. the rise is real, not noise) — so a series
         that was never improving cannot crest-fire (distinct from the plateau detector, which DOES
         fire on flat-from-start);
      2. the last ``hysteresis`` consecutive windows all read NON-RISING (rel-slope <= band — the
         slope sign has changed / stopped being positive; a DECLINING post-peak series still fires,
         which is exactly the crest semantics: the peak has passed).

    NOISE GUARD (derived, one-sided; reuses the flat band): the LATEST window must be resolved
    non-rising — ``rel_slope + rel_stderr <= band``. If the upper confidence bound cannot rule out a
    rising slope, the signal is DEGENERATE → ship banked R1 (never fire on noise), mirroring the
    plateau guard's contract. Deterministic: same series -> same verdict.
    """
    eps = [float(e) for e in eps]
    raw = [float(s) for s in smins]
    n = len(raw)
    latched = latched_fired_epoch is not None and latched_fired_epoch >= 0

    if latched:
        sm = ema_smooth(raw, cfg.ema_span) if raw else []
        return SigmaMinPlateauVerdict(
            classification=CREST_FIRED, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=cfg.hysteresis,
            latest_rel_slope_per_ep=None, latest_rel_stderr_per_ep=None,
            flat_rel_band=cfg.flat_rel_band, smoothed_sigma_min_latest=(sm[-1] if sm else None),
            epoch_latest=(int(eps[-1]) if eps else None), latched_fired_epoch=int(latched_fired_epoch),
            sigma_star_advisory=sigma_star,
            reason=f"pose-finish crest gate LATCHED at ep{latched_fired_epoch} (monotone engagement)",
            evidence=(f"latched crest fire @ep{latched_fired_epoch}; σ_min series now {n} points",))

    # need enough points for the trailing hysteresis windows PLUS at least one earlier window that
    # could carry the rising phase (one extra slide) — derived from the same window machinery.
    min_points = cfg.min_points + 1
    if n < min_points:
        return SigmaMinPlateauVerdict(
            classification=INSUFFICIENT_DATA, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=0, latest_rel_slope_per_ep=None,
            latest_rel_stderr_per_ep=None, flat_rel_band=cfg.flat_rel_band,
            smoothed_sigma_min_latest=(ema_smooth(raw, cfg.ema_span)[-1] if raw else None),
            epoch_latest=(int(eps[-1]) if eps else None), latched_fired_epoch=None,
            sigma_star_advisory=sigma_star,
            reason=(f"only {n} σ_min point(s); crest needs >= {min_points} "
                    f"(settle_window {cfg.settle_window} + hysteresis {cfg.hysteresis} - 1 + 1 "
                    "pre-crest window) — wait/densify"),
            evidence=(f"{n} σ_min points, crest min_points={min_points}",))

    sm = ema_smooth(raw, cfg.ema_span)
    w = int(cfg.settle_window)
    band = float(cfg.flat_rel_band)

    # rolling rel-slopes at EVERY window end position (windows slide by one; end index w-1 .. n-1).
    slopes: list[tuple[int, float, float | None]] = []   # (end_idx, rel_slope, rel_stderr)
    for end in range(w - 1, n):
        lo = end - w + 1
        rel_slope, rel_stderr = _rel_slope_and_stderr(eps[lo:end + 1], sm[lo:end + 1])
        slopes.append((end, rel_slope, rel_stderr))

    # trailing `hysteresis` windows = the crest-hold candidates; everything BEFORE them may carry
    # the rising phase.
    trailing = slopes[-int(cfg.hysteresis):]
    earlier = slopes[:-int(cfg.hysteresis)]
    rising_seen = any(
        (rs > band) and (se is not None) and (se <= rs) for _, rs, se in earlier)
    latest_end, latest_rel_slope, latest_rel_stderr = trailing[-1]

    consecutive_nonrising = 0
    for _, rs, _se in reversed(trailing):
        if rs <= band:
            consecutive_nonrising += 1
        else:
            break

    # one-sided resolution guard on the LATEST window: upper bound must not admit a rising slope.
    guard_degenerate = (latest_rel_stderr is None) or (latest_rel_slope + latest_rel_stderr > band
                                                       and latest_rel_slope <= band)

    ev = (
        f"n={n} σ_min points; EMA span {cfg.ema_span}; latest de-noised σ_min={sm[-1]:.6g}"
        f"@ep{int(eps[-1])}",
        f"rising phase resolved earlier: {rising_seen} "
        f"(needs rel-slope > {band:.1e} with rel-stderr <= slope in a pre-crest window)",
        f"latest-window rel-slope {latest_rel_slope:+.2e}/ep, rel-stderr "
        f"{('%.2e' % latest_rel_stderr) if latest_rel_stderr is not None else 'undefined'}; "
        f"consecutive non-rising windows = {consecutive_nonrising}/{cfg.hysteresis}",
    )

    if guard_degenerate:
        return SigmaMinPlateauVerdict(
            classification=DEGENERATE_GUARD_TRIPPED, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=consecutive_nonrising,
            latest_rel_slope_per_ep=latest_rel_slope, latest_rel_stderr_per_ep=latest_rel_stderr,
            flat_rel_band=band, smoothed_sigma_min_latest=sm[-1], epoch_latest=int(eps[-1]),
            latched_fired_epoch=None, sigma_star_advisory=sigma_star,
            reason=("CREST NOISE GUARD tripped: the latest window's slope upper confidence bound "
                    f"(rel-slope {latest_rel_slope:+.2e} + rel-stderr "
                    f"{('%.2e' % latest_rel_stderr) if latest_rel_stderr is not None else 'undefined'}) "
                    f"cannot rule out a rising slope (> band {band:.1e}) — CANNOT distinguish a crest "
                    "from oscillation. FALL BACK to ship-banked-R1 (never fire on a degenerate signal)."),
            evidence=ev)

    if rising_seen and consecutive_nonrising >= int(cfg.hysteresis):
        return SigmaMinPlateauVerdict(
            classification=CREST_FIRED, n_points=n, settle_window=cfg.settle_window,
            hysteresis=cfg.hysteresis, consecutive_flat=consecutive_nonrising,
            latest_rel_slope_per_ep=latest_rel_slope, latest_rel_stderr_per_ep=latest_rel_stderr,
            flat_rel_band=band, smoothed_sigma_min_latest=sm[-1], epoch_latest=int(eps[-1]),
            latched_fired_epoch=None, sigma_star_advisory=sigma_star,
            reason=(f"σ_min CRESTED: a resolved rising phase, then non-rising slope for "
                    f"{consecutive_nonrising} >= {cfg.hysteresis} consecutive settle windows — the "
                    "conditioning PEAK has passed (slope sign-change). ENGAGE pose-finish."),
            evidence=ev)

    if not rising_seen:
        reason = ("not crested: NO resolved rising phase yet (crest requires conditioning to have "
                  "genuinely IMPROVED before the sign-change — flat-from-start never crest-fires)")
    else:
        reason = (f"not crested: only {consecutive_nonrising}/{cfg.hysteresis} consecutive non-rising "
                  f"windows (latest rel-slope {latest_rel_slope:+.2e}/ep vs band {band:.1e}) — σ_min "
                  "still rising (conditioning still improving)")
    return SigmaMinPlateauVerdict(
        classification=NOT_CRESTED, n_points=n, settle_window=cfg.settle_window,
        hysteresis=cfg.hysteresis, consecutive_flat=consecutive_nonrising,
        latest_rel_slope_per_ep=latest_rel_slope, latest_rel_stderr_per_ep=latest_rel_stderr,
        flat_rel_band=band, smoothed_sigma_min_latest=sm[-1], epoch_latest=int(eps[-1]),
        latched_fired_epoch=None, sigma_star_advisory=sigma_star, reason=reason, evidence=ev)


# --------------------------------------------------------------------------- #
# The stateful detector (Resumable — persists the σ_min series + the fire latch).
# --------------------------------------------------------------------------- #
class SigmaMinPlateauDetector:
    """Accumulates the σ_min series across the run and reports the LATCHED conditioning-gate decision.

    Resumable (registered ONLY when the ``sigma_min_plateau`` engage lever is ON — the OBSERVER path does
    not register, keeping the incumbent checkpoint sidecar byte-identical). The latch makes engagement
    MONOTONE: once fired, ``fired()`` stays True even if σ_min later wiggles up (the caller's
    ``_pose_finish_on`` latch is also monotone; this keeps the two in agreement across a resume)."""

    def __init__(self, cfg: SigmaMinPlateauConfig,
                 *, c_pose_grad_cap: float = 25.0, delta_seg: float = 0.5,
                 lambda_min_f: float | None = None, mode: str = MODE_PLATEAU) -> None:
        probs = cfg.validate()
        if probs:
            raise ValueError("; ".join(probs))
        if str(mode) not in (MODE_PLATEAU, MODE_CREST):
            raise ValueError(
                f"SigmaMinPlateauDetector: mode must be {MODE_PLATEAU!r} or {MODE_CREST!r}, got {mode!r}")
        self.cfg = cfg
        #: which event class this detector fires on: 'plateau' (sealed incumbent, DEFAULT — the
        #: OFF-path is byte-identical) or 'crest' (SPEC_v10 §13.2 fire-on-crest). One mode per run.
        self.mode = str(mode)
        self._eps: list[int] = []
        self._smins: list[float] = []
        self._fired_epoch: int = -1
        # advisory-sideband inputs (A-1: σ* is logged, never gates)
        self._c = float(c_pose_grad_cap)
        self._delta_seg = float(delta_seg)
        self._lambda_min_f = lambda_min_f

    # -- observation (mutates the series; idempotent on non-increasing epochs) --
    def observe(self, epoch: int, sigma_min: float) -> None:
        """Append (epoch, median σ_min). Idempotent guard: a non-increasing epoch (a re-run/backwards
        epoch after resume) is SKIPPED so a resume cannot double-count. Non-finite σ_min is skipped."""
        e = int(epoch)
        if self._eps and e <= self._eps[-1]:
            return
        if not math.isfinite(float(sigma_min)):
            return
        self._eps.append(e)
        self._smins.append(float(sigma_min))
        # bound the persisted series (resume-key size); the detector only reads the recent window anyway.
        if len(self._eps) > _MAX_PERSISTED_POINTS:
            self._eps = self._eps[-_MAX_PERSISTED_POINTS:]
            self._smins = self._smins[-_MAX_PERSISTED_POINTS:]

    def _sigma_star(self) -> float | None:
        return sigma_star_advisory(self._c, self._delta_seg, self._lambda_min_f)

    def verdict(self) -> SigmaMinPlateauVerdict:
        """PURE evaluation over the current series + latch (reports LATCHED if already fired).
        Dispatches on the detector's single ``mode`` (plateau incumbent / crest §13.2)."""
        _eval = evaluate_crest if self.mode == MODE_CREST else evaluate_plateau
        return _eval(self._eps, self._smins, self.cfg,
                     latched_fired_epoch=self._fired_epoch, sigma_star=self._sigma_star())

    def latch_if_fired(self, epoch: int) -> bool:
        """If the current verdict fires AND we are not yet latched, latch at ``epoch`` and return True.
        Monotone: never un-latches. DEGENERATE/NOT_PLATEAUED never latch (never fire on a bad signal)."""
        if self._fired_epoch >= 0:
            return False
        v = self.verdict()
        if v.fired():
            self._fired_epoch = int(epoch)
            return True
        return False

    def fired(self) -> bool:
        return self._fired_epoch >= 0

    def should_ship_banked_r1(self) -> bool:
        """DEGENERATE fall-back, DELEGATED to the current verdict (P0-2 interface fix 2026-07-10:
        the trainer's backstop guard called this on the DETECTOR but it existed only on
        :class:`SigmaMinPlateauVerdict` — an AttributeError that KILLED the first v752 launch at ep1,
        run ``levelset_v752_pilot_20260710T154100Z``. The detector now mirrors the verdict's decision
        surface so either object satisfies the guard's contract)."""
        return self.verdict().should_ship_banked_r1()

    @property
    def classification(self) -> str:
        """The current verdict's classification (delegated; see :meth:`should_ship_banked_r1`)."""
        return self.verdict().classification

    @property
    def fired_epoch(self) -> int:
        return self._fired_epoch

    @property
    def n_points(self) -> int:
        return len(self._eps)

    # -- Resumable protocol (write/restore the series + the latch) --
    def state_arrays(self, prefix: str) -> dict[str, Any]:
        """Persist the σ_min series + the fire latch. Non-empty (>=1 point or a latch) => keys written =>
        the sidecar carries them; an empty un-fired detector writes NOTHING (byte-identical)."""
        if not self._eps and self._fired_epoch < 0:
            return {}
        return {
            prefix + "eps": np.asarray(self._eps, dtype=np.int64),
            prefix + "smins": np.asarray(self._smins, dtype=np.float64),
            prefix + "fired_epoch": np.asarray(int(self._fired_epoch), dtype=np.int64),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict) -> bool:
        """Restore the series + latch from the parsed sidecar cfg. Returns True iff state was restored
        (the eps key is present). Missing keys => fresh detector (deterministic going forward)."""
        key = prefix + "eps"
        if key not in cfg:
            return False
        eps = np.asarray(cfg[prefix + "eps"]).reshape(-1)
        smins = np.asarray(cfg[prefix + "smins"]).reshape(-1)
        self._eps = [int(x) for x in eps.tolist()]
        self._smins = [float(x) for x in smins.tolist()]
        fe = cfg.get(prefix + "fired_epoch")
        self._fired_epoch = int(np.asarray(fe).reshape(-1)[0]) if fe is not None else -1
        return True


# --------------------------------------------------------------------------- #
# $0 CANARY controls (confound L3 positive/negative controls) + real backtest.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanaryResult:
    """Outcome of the $0 pre-launch canary suite. ``passed`` iff the NEGATIVE control does NOT fire AND
    the SYNTHETIC POSITIVE control DOES fire — a real UNIT TEST of the detector, satisfiable at $0
    (SYNTHESIS §A.4 Repair 2). ``passed`` False ⇒ the gate is UNTRUSTED ⇒ ship-banked-R1 DISENGAGED."""

    passed: bool
    negative_fired: bool           # must be False (rising σ_min must NOT read as plateaued)
    positive_fired: bool           # must be True (clean plateau must fire)
    negative_classification: str
    positive_classification: str
    reason: str
    evidence: tuple[str, ...] = field(default_factory=tuple)


def synthetic_plateau_series(
    n_rise: int = 8, n_flat: int = 16, plateau: float = 0.05, decay_ep: float = 20.0,
    ep_step: int = 4, noise: float = 0.0, seed: int = 0,
) -> list[tuple[int, float]]:
    """A clean relaxation-THEN-FLAT σ_min curve (the SYNTHETIC POSITIVE control): ``n_rise`` points of
    relaxation ``plateau·(1 − exp(−ep/decay_ep))`` followed by ``n_flat`` GENUINELY FLAT points at
    ``plateau`` (+ tiny ``noise``). The long flat tail is what a real conditioning plateau looks like —
    it out-runs the EMA-smoothing lag so ``hysteresis`` consecutive windows read flat → the detector MUST
    fire. (An exponential that is still *approaching* the asymptote has a nonzero rel-slope and correctly
    does NOT read as plateaued — hence the explicit flat tail here.)"""
    rng = np.random.default_rng(int(seed))
    out: list[tuple[int, float]] = []
    ep = 0
    for i in range(int(n_rise)):
        base = float(plateau) * (1.0 - math.exp(-ep / float(decay_ep)))
        w = float(noise) * float(plateau) * float(rng.standard_normal()) if noise > 0 else 0.0
        out.append((ep, max(0.0, base + w)))
        ep += int(ep_step)
    for _ in range(int(n_flat)):
        w = float(noise) * float(plateau) * float(rng.standard_normal()) if noise > 0 else 0.0
        out.append((ep, max(0.0, float(plateau) + w)))
        ep += int(ep_step)
    return out


def synthetic_rising_series(
    n: int = 31, start: float = 0.02, drift_per_ep: float = 3.0e-4, ep_step: int = 4,
    cv: float = 0.21, seed: int = 1,
) -> list[tuple[int, float]]:
    """A RISING-drift σ_min curve with CV-0.21 multiplicative noise (the NEGATIVE control) — reproduces
    the P4-M1 MEASURED signature of the stopped run's 31 σ_min rows (a noisy OSCILLATION with a RISING
    drift, NOT a relaxation curve). The detector must NOT fire on it (σ_min still rising = still
    conditioning, not plateaued)."""
    rng = np.random.default_rng(int(seed))
    out: list[tuple[int, float]] = []
    for i in range(int(n)):
        ep = i * int(ep_step)
        base = float(start) + float(drift_per_ep) * ep
        val = base * (1.0 + float(cv) * float(rng.standard_normal()))
        out.append((ep, max(1e-6, val)))
    return out


def run_detector_on_series(
    series: Sequence[tuple[int, float]], cfg: SigmaMinPlateauConfig | None = None,
    *, mode: str = MODE_PLATEAU,
) -> SigmaMinPlateauVerdict:
    """Feed a full (epoch, σ_min) series through a fresh detector and return the FINAL verdict (no
    latching — the un-latched classification the series earns on its own)."""
    cfg = cfg or SigmaMinPlateauConfig()
    det = SigmaMinPlateauDetector(cfg, mode=mode)
    for ep, s in series:
        det.observe(int(ep), float(s))
    return det.verdict()


def synthetic_rise_then_decline_series(
    n_rise: int = 10, n_decline: int = 10, peak: float = 0.06, ep_step: int = 4,
) -> list[tuple[int, float]]:
    """A clean rise-to-PEAK-then-DECLINE σ_min curve (the crest POSITIVE control): the plateau
    detector correctly never fires on it (the tail is trending DOWN, not flat), but the crest
    detector MUST (the sign-change is the event). Deterministic (no noise)."""
    out: list[tuple[int, float]] = []
    ep = 0
    for i in range(int(n_rise)):
        out.append((ep, float(peak) * (i + 1) / float(n_rise)))
        ep += int(ep_step)
    for i in range(int(n_decline)):
        out.append((ep, float(peak) * (1.0 - 0.4 * (i + 1) / float(n_decline))))
        ep += int(ep_step)
    return out


def crest_canary_suite(cfg: SigmaMinPlateauConfig | None = None) -> CanaryResult:
    """$0 canary controls for the FIRE-ON-CREST mode (SPEC_v10 §13.2), mirroring
    :func:`canary_suite`'s contract: NEGATIVE (a still-RISING σ_min — the live c2 signature — must
    NOT crest-fire) + POSITIVE (a clean rise-then-decline peak MUST crest-fire). FAIL ⇒ the crest
    gate is UNTRUSTED ⇒ ship-banked-R1 DISENGAGED (never fire on an untrusted instrument)."""
    cfg = cfg or SigmaMinPlateauConfig()
    neg = run_detector_on_series(synthetic_rising_series(), cfg, mode=MODE_CREST)
    pos = run_detector_on_series(synthetic_rise_then_decline_series(), cfg, mode=MODE_CREST)
    negative_fired = neg.fired()
    positive_fired = pos.fired()
    passed = (not negative_fired) and positive_fired
    reason = ("crest canary PASS: rising σ_min does NOT crest-fire and a rise-then-decline peak "
              "DOES — detector trustworthy"
              if passed else
              "crest canary FAIL: " + (
                  "rising σ_min CREST-FIRED (false positive) " if negative_fired else "") + (
                  "rise-then-decline peak did NOT crest-fire (false negative) "
                  if not positive_fired else "")
              + "→ gate UNTRUSTED → ship-banked-R1 DISENGAGED")
    return CanaryResult(
        passed=passed, negative_fired=negative_fired, positive_fired=positive_fired,
        negative_classification=neg.classification, positive_classification=pos.classification,
        reason=reason,
        evidence=(f"negative(rising)={neg.classification} [{neg.reason[:80]}]",
                  f"positive(rise-then-decline)={pos.classification} [{pos.reason[:80]}]"))


def canary_suite(cfg: SigmaMinPlateauConfig | None = None) -> CanaryResult:
    """Run the $0 negative + synthetic-positive controls (SYNTHESIS §A.4 Repair 2). PASS iff the rising
    negative does NOT fire AND the clean-plateau positive DOES. A FAIL means the detector cannot be
    trusted to gate → the caller DISENGAGES (ship banked R1)."""
    cfg = cfg or SigmaMinPlateauConfig()
    neg = run_detector_on_series(synthetic_rising_series(), cfg)
    pos = run_detector_on_series(synthetic_plateau_series(), cfg)
    negative_fired = neg.fired()
    positive_fired = pos.fired()
    passed = (not negative_fired) and positive_fired
    reason = ("canary PASS: rising σ_min does NOT fire and a clean plateau DOES — detector trustworthy"
              if passed else
              "canary FAIL: " + (
                  "rising σ_min FIRED (false positive) " if negative_fired else "") + (
                  "clean plateau did NOT fire (false negative) " if not positive_fired else "")
              + "→ gate UNTRUSTED → ship-banked-R1 DISENGAGED")
    return CanaryResult(
        passed=passed, negative_fired=negative_fired, positive_fired=positive_fired,
        negative_classification=neg.classification, positive_classification=pos.classification,
        reason=reason,
        evidence=(f"negative(rising)={neg.classification} [{neg.reason[:80]}]",
                  f"positive(plateau)={pos.classification} [{pos.reason[:80]}]"))


def load_sigma_min_series_from_jsonl(path: str | Path) -> list[tuple[int, float]]:
    """READ-ONLY backtest helper: read a run's ``jacobian_basin`` telemetry rows from a JSONL/run.log
    and return the (epoch, median_sigma_min) series in order. Never writes. Fail-soft: a missing file /
    unparseable line yields the rows found so far (empty if none). Used by the OPTIONAL real-series
    backtest canary (run dirs are READ-ONLY)."""
    p = Path(path)
    out: list[tuple[int, float]] = []
    if not p.is_file():
        return out
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or '"jacobian_basin"' not in line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if row.get("stage") != "jacobian_basin":
            continue
        ep = row.get("epoch")
        sm = row.get("median_sigma_min")
        if isinstance(ep, (int, float)) and isinstance(sm, (int, float)):
            out.append((int(ep), float(sm)))
    out.sort(key=lambda t: t[0])
    return out


# --------------------------------------------------------------------------- #
# Telemetry / alarm formatting (LOUD disengaged alarm + digest surfacing).
# --------------------------------------------------------------------------- #
def gate_observer_row(verdict: SigmaMinPlateauVerdict, epoch: int, seg_form: str,
                      *, engage_mode: str, actuated: bool) -> dict:
    """A score-neutral OBSERVER telemetry row for the conditioning gate (default-ON observability)."""
    return {
        "stage": "pose_finish_conditioning_gate", "epoch": int(epoch), "seg_form": str(seg_form),
        "engage_mode": str(engage_mode), "classification": verdict.classification,
        "consecutive_flat": verdict.consecutive_flat, "hysteresis": verdict.hysteresis,
        "settle_window": verdict.settle_window,
        "latest_rel_slope_per_ep": verdict.latest_rel_slope_per_ep,
        "latest_rel_stderr_per_ep": verdict.latest_rel_stderr_per_ep,
        "flat_rel_band": verdict.flat_rel_band,
        "smoothed_sigma_min_latest": verdict.smoothed_sigma_min_latest,
        "sigma_star_advisory": verdict.sigma_star_advisory,   # A-1: WATCH sideband only, never a conjunct
        "n_points": verdict.n_points, "fired": verdict.fired(),
        "should_ship_banked_r1": verdict.should_ship_banked_r1(), "actuated": bool(actuated),
        "axis": "[macOS-MLX advisory] NON-PROMOTABLE",
    }


def disengaged_alarm_row(epoch: int, *, reason: str, canary_passed: bool | None,
                         n_points: int) -> dict:
    """The LOUD end-of-run pose-DISENGAGED alarm (typed ``confound_alarm`` row, digest-surfaced). Fires
    when the run ships pose on the banked R1 floor because the conditioning gate never engaged — a
    pose-limited frontier, never silent (SYNTHESIS §A.4 Repair 4)."""
    return {
        "stage": "confound_alarm", "alarm": "pose_finish_disengaged_shipped_banked_r1",
        "epoch": int(epoch), "reason": str(reason), "canary_passed": canary_passed,
        "n_sigma_min_points": int(n_points),
        "note": ("pose-finish conditioning gate never engaged → shipped the banked R1 dxi (0.001610, "
                 "7.2KB) on the seg-converged EMA ckpt. A pose-limited frontier — the in-basin finish "
                 "was an OPTIONAL improvement over a banked floor, never a launch dependency."),
        "axis": "[macOS-MLX advisory] NON-PROMOTABLE",
    }


def resolve_pose_finish_engage(*, cond_fired: bool, backstop_hit: bool,
                               degenerate: bool) -> tuple[bool, bool]:
    """P0-2 MINIMUM-VIABLE backstop repair (fresh-eyes advisory 2026-07-10; operator-directed): the
    ``--pose-finish-start-epoch`` epoch backstop must NOT override a DEGENERATE
    (``should_ship_banked_r1``) classification — the sealed fallback contract (SPEC_v752 §183-187)
    says a degenerate conditioning signal ships the banked R1 artifact DISENGAGED; the prior
    ``cond_fired OR ep >= backstop`` boolean silently reinterpreted "banked fallback" as "start joint
    descent anyway" (observed live: owed16v2 arm, 8 DEGENERATE_GUARD_TRIPPED rows with
    should_ship_banked_r1=true at the same epochs the backstop would have engaged).

    Returns ``(engage, banked_r1_selected)``:
      * a REAL conditioning fire always engages (the latch is monotone and only latches on a real
        plateau — a fired detector is by definition non-degenerate at the latch);
      * the backstop engages ONLY when the detector state is NOT degenerate (healthy-but-never-fired
        at the backstop epoch = the fail-safe engage, unchanged);
      * degenerate at the backstop epoch ⇒ NO engage + ``banked_r1_selected=True`` (the caller emits
        the loud typed row ONCE and ships the banked R1 dxi path).
    The FULL ARMED→ENGAGED→ACCEPTED/REGRESSED_ROLLBACK/BANKED_COMPLETE_ARTIFACT state machine is the
    owed end state (advisory P0-2); this is the minimum-viable guard. PURE / unit-tested."""
    if cond_fired:
        return True, False
    if backstop_hit and degenerate:
        return False, True
    return bool(backstop_hit), False


def backstop_banked_r1_row(epoch: int, *, backstop_epoch: int, classification: str | None,
                           n_points: int) -> dict:
    """The LOUD typed row for the P0-2 banked-R1 backstop decision: the epoch backstop was reached
    while the conditioning signal is DEGENERATE ⇒ pose-finish stays DISENGAGED and the banked R1 dxi
    path is selected (never a silent reinterpretation). Emitted ONCE at the decision epoch."""
    return {
        "stage": "confound_alarm", "alarm": "pose_finish_backstop_overridden_banked_r1",
        "epoch": int(epoch), "backstop_epoch": int(backstop_epoch),
        "classification": classification, "should_ship_banked_r1": True,
        "n_sigma_min_points": int(n_points),
        "note": ("--pose-finish-start-epoch backstop reached while the σ_min conditioning signal is "
                 "DEGENERATE (guard tripped: plateau indistinguishable from oscillation) → per the "
                 "sealed fallback contract (SPEC_v752 §183-187) pose-finish stays DISENGAGED and the "
                 "banked R1 dxi (0.001610, 7.2KB) ships on the seg-converged EMA. The backstop may "
                 "force a DECISION; it may not reinterpret 'banked fallback' as 'engage anyway'."),
        "axis": "[macOS-MLX advisory] NON-PROMOTABLE",
    }


def format_gate_line(row: dict) -> str:
    """One-line digest formatter for a pose-gate observer / disengaged-alarm row."""
    if row.get("alarm") == "pose_finish_disengaged_shipped_banked_r1":
        return (f"pose-gate: ep{row.get('epoch')} DISENGAGED → shipped banked R1 "
                f"(canary_passed={row.get('canary_passed')}, {row.get('n_sigma_min_points')} σ_min pts) "
                "[advisory NON-PROMOTABLE]")
    cls = row.get("classification")
    return (f"pose-gate: ep{row.get('epoch')} {cls} "
            f"({row.get('consecutive_flat')}/{row.get('hysteresis')} flat windows, "
            f"σ_min~{row.get('smoothed_sigma_min_latest')}) "
            f"{'ACTUATED' if row.get('actuated') else 'observer'} [advisory NON-PROMOTABLE]")


def scan_run_for_pose_gate(run_dir: str | Path) -> dict | None:
    """READ-ONLY: scan a run's ``run.log`` for the LATEST pose-gate row worth surfacing — the DISENGAGED
    alarm if present, else the latest conditioning-gate observer row. Returns the row dict (or None).
    Fail-soft (any error → None). Consumed by ``tools/costate_digest.section_pose_conditioning_gate``."""
    p = Path(run_dir) / "run.log"
    if not p.is_file():
        return None
    disengaged: dict | None = None
    latest_obs: dict | None = None
    try:
        for line in p.read_text(errors="ignore").splitlines():
            line = line.strip()
            if '"pose_finish_disengaged_shipped_banked_r1"' in line:
                try:
                    disengaged = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
            elif '"pose_finish_conditioning_gate"' in line:
                try:
                    latest_obs = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return None
    return disengaged or latest_obs


__all__ = [
    "W_SETTLE_CEIL", "DEFAULT_EMA_SPAN", "DEFAULT_FLAT_REL_BAND", "DEFAULT_SETTLE_WINDOW",
    "DEFAULT_HYSTERESIS", "RESUME_PREFIX",
    "PLATEAU_FIRED", "NOT_PLATEAUED", "DEGENERATE_GUARD_TRIPPED", "INSUFFICIENT_DATA",
    "CREST_FIRED", "NOT_CRESTED", "MODE_PLATEAU", "MODE_CREST",
    "SigmaMinPlateauConfig", "SigmaMinPlateauVerdict", "SigmaMinPlateauDetector",
    "ema_smooth", "sigma_star_advisory", "evaluate_plateau", "evaluate_crest",
    "CanaryResult", "synthetic_plateau_series", "synthetic_rising_series",
    "synthetic_rise_then_decline_series", "crest_canary_suite",
    "run_detector_on_series", "canary_suite", "load_sigma_min_series_from_jsonl",
    "gate_observer_row", "disengaged_alarm_row", "format_gate_line", "scan_run_for_pose_gate",
]
