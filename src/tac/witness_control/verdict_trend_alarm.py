"""VERDICT-TREND / TRAIN-VERDICT-DECOUPLING alarm (operator catch 2026-07-09).

The gap this closes (score-neutral, read-only, DEFAULTS ON):

    Operator (verbatim): "I thought the costate controller and event detection was
    smart enough to respond to and detect a trajectory like dseg is on which seems
    weird and bad and like more CE isn't going to help."

The live run's ADVISORY verdict d_seg ROSE (0.0324@ep50 -> 0.0340@ep100; Lane class
0.349 -> 0.381) while the training seg-loss DESCENDED, and the shadow controller
classified CONVERGING with "(none identifiable)" recs. Why the miss is a
CLASSIFICATION bug, not a sensing bug: ``tools/witness_control_monitor.classify_
trajectory`` fits a least-squares d_seg slope over the last ``window`` same-stage
verdicts; that window still spans the early CE drop (ep25 0.0366 -> ep50 0.0324),
so the *net* window slope reads negative and the final ``else`` branch labels the
run "healthy descent (dV/dt < 0)" -- even though the last THREE verdicts are
monotone-RISING. A FLAT-or-NET-DOWN window slope reads as CONVERGING/PLATEAU. This
is the sibling of the #315 binding-term-stall FALSE-GREEN (a FLAT binding term reads
as PLATEAU); here a *net-down-but-recently-rising* binding term reads as CONVERGING.

This module adds the missing sensor. It is Layer-1 observability per CLAUDE.md
"Confound self-protection": a typed ``confound_alarm`` row that turns the silent
rising-verdict artifact LOUD. It reads verdict rows + a train-loss series and
NEVER touches weights/bytes/d_seg/d_pose -> byte-identity preserved by construction
-> it DEFAULTS ON (CLAUDE.md "'Off' is a tracked queue": score-neutral telemetry is
not gate-able). CONTAINMENT: advisory only; no actuation surface.

Two alarm classes (falling severity):

  * ``TRAIN_VERDICT_DECOUPLING`` -- the verdict d_seg is materially RISING over >=k
    consecutive verdicts WHILE the train seg-loss slope over the SAME window is <= 0.
    This is the "more-of-the-same-gradient won't help" signature the operator named:
    the optimizer is descending the training surrogate but the argmax verdict is
    getting WORSE (surrogate<->verdict decoupling).
  * ``RISING_VERDICT`` -- the verdict d_seg is materially rising over >=k verdicts,
    but the train loss is NOT (cleanly) decoupled (loss flat/rising too, or the
    rise is a plausible EMA-shadow catch-up transient). Still surfaced (rising is
    never "converging"), lower severity.

The two DERIVED thresholds (no guessed constants):

  * ``rise_rel_eps`` = ``DEFAULT_STALL_REL_EPS`` = 3e-4 per-epoch |slope|/level. This
    is the SAME calibrated gate the #315 binding-term-stall detector uses to call a
    term "flat" (``costate_estimator.DEFAULT_STALL_REL_EPS``): binding-stall fires on
    ``|rel_slope| <= 3e-4`` (flat); THIS alarm fires on ``rel_slope > +3e-4`` (a
    material RISE). They partition the trajectory space -- reusing the one calibrated
    constant keeps them coherent and non-overlapping. Empirical anchors: the live
    run's rise window (ep50->100) sits at rel_slope ~ +9.6e-4/ep (>> 3e-4 -> fires);
    mod32cap's converged terminal wiggle (ep675->725) sits at ~ +2.2e-4/ep
    (< 3e-4 -> stays SILENT, correctly read as plateau jitter, not a rise).
  * ``decouple_rel_gap_floor`` = 0.01 -- the EMA-lag-vs-decoupled magnitude floor,
    stated in the relative-significance frame (rise / remaining-gap-to-target, NEVER
    absolute -- CLAUDE.md "relative-not-absolute-significance-near-goal"). 1% of the
    remaining descent is a conservative material floor (well below the 13-27% "weak
    levers still matter" anchor). At the live ep100 anchor the sustained rise is
    +0.001608 d_seg = +4.9% of the rise-start value AND +4.85% of the remaining gap
    to the 0.00092 lane-band target -- an order above the floor, so it is a real
    DECOUPLING, not an EMA-shadow lag transient.

The EMA-lag disambiguation (mechanism-derived, from memory
``verdict-d_pose RISE while training-pose falls = EMA-shadow lag``): an EMA shadow
lags live weights and its verdict bump resolves within ~1 verdict interval; a rise
that persists >=k verdicts (>= k * eval_every epochs) is beyond that catch-up window
=> NOT lag. So ``ema_lag_plausible`` is True only when the rise JUST crossed
(best_stale == k), the run is very early (<= k+2 same-stage verdicts), AND the
magnitude is below the floor. Otherwise a persisted+material rise is a real trend.

Pure math + dataclasses. No IO, no trainer imports, no actuation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from tac.witness_control.costate_estimator import (
    DEFAULT_STALL_REL_EPS,
    slope_with_stderr,
)

# ── classification sentinels (falling severity) ──
NO_ALARM = "NO_ALARM"
RISING_VERDICT = "RISING_VERDICT"
TRAIN_VERDICT_DECOUPLING = "TRAIN_VERDICT_DECOUPLING"
RISING_VERDICT_UNIDENTIFIABLE = "RISING_VERDICT_UNIDENTIFIABLE"

#: per-epoch |slope|/level ABOVE which a trend counts as a MATERIAL RISE. The SAME
#: calibrated constant the #315 binding-term-stall detector uses for "flat"
#: (partitions the space: <=3e-4 = flat/plateau [binding-stall's lane];
#: >+3e-4 = materially rising [this alarm's lane]).
RISE_REL_EPS = DEFAULT_STALL_REL_EPS
#: EMA-lag-vs-decoupled magnitude floor, in the relative-significance frame
#: (rise / remaining-gap-to-target). 1% of the remaining descent = a conservative
#: material floor per CLAUDE.md "relative-not-absolute-significance-near-goal".
DECOUPLE_REL_GAP_FLOOR = 0.01
#: default binding d_seg target (openpilot lane-band need; memory L71 ~0.00087-0.00092)
DEFAULT_GAP_TARGET = 0.00092
#: floor so remaining-gap never divides by ~0 when the value sits at/below target.
_GAP_FLOOR = 1e-6
#: canonical comma10k SegNet class order (do NOT re-derive by luma-sort; memory L80)
_CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


@dataclass(frozen=True)
class VerdictTrendAlarm:
    """Whether the ADVISORY verdict d_seg is materially RISING (and, if so, whether
    the train seg-loss is DECOUPLED from it) over the latest within-stage window.

    Every field is MEASURED from the verdict rows; NO fabricated ΔS. ``fired()`` is
    True for the two rising classes; ``NO_ALARM`` / ``*_UNIDENTIFIABLE`` are silent.
    """

    classification: str            # NO_ALARM | RISING_VERDICT | TRAIN_VERDICT_DECOUPLING | RISING_VERDICT_UNIDENTIFIABLE
    stage: str
    n_stage: int                   # same-stage verdict rows seen
    k: int                         # persistence threshold (consecutive verdicts)
    epoch_latest: int | None
    d_seg_latest: float | None
    d_seg_best: float | None
    best_epoch: int | None
    best_stale_verdicts: int       # verdicts since the best (rise persistence)
    monotone_recent: bool          # last k+1 same-stage d_seg non-decreasing
    rise_abs: float                # d_seg_latest − d_seg_best (>0 = rose above best)
    rise_rel_value: float          # rise_abs / d_seg_best (the "+4.9%")
    rise_rel_gap: float            # rise_abs / remaining-gap-to-target (the significance)
    remaining_gap: float
    gap_target: float
    rel_slope_per_ep: float        # d_seg rel-slope over the rise window (material gate)
    rise_rel_eps: float            # the material gate value used
    train_loss_slope_per_ep: float | None   # over the SAME window (None = no loss series)
    train_loss_decoupled: bool     # loss slope <= 0 while verdict rising (the decoupling)
    ema_lag_plausible: bool        # rise looks like an EMA-shadow catch-up transient
    per_class_alarms: tuple[dict, ...]       # per-class rising rows (may be empty)
    reason: str
    evidence: tuple[str, ...]

    def fired(self) -> bool:
        return self.classification in (RISING_VERDICT, TRAIN_VERDICT_DECOUPLING)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        d["per_class_alarms"] = [dict(r) for r in self.per_class_alarms]
        return d

    def to_confound_alarm_row(self) -> dict:
        """Typed ``confound_alarm`` telemetry row (LOUD), matching the trainer's L1
        convention ``{"stage": "confound_alarm", "alarm": <kind>, ...}``. Advisory;
        never halts training (CONTAINMENT)."""
        return {
            "stage": "confound_alarm",
            "alarm": ("verdict_rising_decoupling"
                      if self.classification == TRAIN_VERDICT_DECOUPLING
                      else "verdict_trend_rising"),
            "classification": self.classification,
            "seg_form": self.stage,
            "ep": self.epoch_latest,
            "d_seg": self.d_seg_latest,
            "d_seg_best": self.d_seg_best,
            "best_stale_verdicts": self.best_stale_verdicts,
            "rise_rel_value": round(self.rise_rel_value, 5),
            "rise_rel_gap": round(self.rise_rel_gap, 5),
            "rel_slope_per_ep": self.rel_slope_per_ep,
            "train_loss_decoupled": self.train_loss_decoupled,
            "ema_lag_plausible": self.ema_lag_plausible,
            "per_class_rising": [r["class_name"] for r in self.per_class_alarms],
            "reason": self.reason,
            "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        }


def _latest_stage_rows(verdicts: list[dict], channel: str = "d_seg") -> tuple[str, list[dict]]:
    """Same-stage rows (latest ``seg_form``) carrying ``channel`` + a numeric epoch."""
    rows = [v for v in verdicts
            if isinstance(v.get(channel), (int, float))
            and isinstance(v.get("epoch"), (int, float))]
    if not rows:
        return "", []
    latest_stage = str(rows[-1].get("seg_form", ""))
    return latest_stage, [v for v in rows if str(v.get("seg_form", "")) == latest_stage]


def _rel_slope_per_ep(eps: list[float], ys: list[float]) -> float:
    """slope/|level| per epoch over (eps, ys). 0.0 when level ≈ 0 (never div-by-0)."""
    fit = slope_with_stderr(eps, ys)
    level = (sum(ys) / len(ys)) if ys else 0.0
    return (fit.slope / abs(level)) if abs(level) > 0.0 else 0.0


def _non_decreasing(seq: list[float]) -> bool:
    return all(b >= a for a, b in zip(seq, seq[1:], strict=False))


def _train_loss_slope_over_window(
    verdicts_window: list[dict], loss_rows: list[dict] | None,
    ep_lo: float, ep_hi: float,
) -> tuple[float | None, str]:
    """Train-loss slope over [ep_lo, ep_hi]. Prefers an explicit ``loss_rows`` series
    (each row {epoch, seg_loss|loss|ep_loss}); else uses the verdict rows' ``ep_loss``
    field (the total training loss -- seg-dominated via ``--w-seg 100``, so a faithful
    train-seg-loss surrogate). Returns (slope_per_ep | None, source-label)."""
    def _series(rows: list[dict], keys: tuple[str, ...]) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for r in rows:
            e = r.get("epoch")
            if not isinstance(e, (int, float)) or not (ep_lo <= float(e) <= ep_hi):
                continue
            val = next((r[k] for k in keys if isinstance(r.get(k), (int, float))), None)
            if val is not None:
                pts.append((float(e), float(val)))
        return pts

    if loss_rows:
        pts = _series(loss_rows, ("seg_loss", "loss", "ep_loss"))
        src = "explicit loss_rows series"
    else:
        pts = _series(verdicts_window, ("ep_loss",))
        src = "verdict ep_loss field (total loss, seg-dominated via w_seg=100)"
    if len(pts) < 2:
        return None, src + " (insufficient points for a slope)"
    fit = slope_with_stderr([p[0] for p in pts], [p[1] for p in pts])
    return fit.slope, src


def _rising_channel(
    rows: list[dict], channel: str, *, k: int, rise_rel_eps: float,
) -> dict | None:
    """Core rising test on ONE channel's same-stage series. Returns a metrics dict
    when the channel is materially rising over >=k consecutive verdicts, else None.

    Fires iff (all MEASURED, no guess): latest > best (rose above best) AND best_stale
    >= k (persisted >=k verdicts) AND the last k+1 values are non-decreasing (recent
    monotone rise, not a lone spike) AND the rel-slope over the rise window (best->
    latest) > rise_rel_eps (MATERIAL, above the calibrated flat gate)."""
    pts = [(float(v["epoch"]), float(v[channel])) for v in rows
           if isinstance(v.get(channel), (int, float))
           and isinstance(v.get("epoch"), (int, float))]
    n = len(pts)
    if n < k + 1:
        return None
    ys = [p[1] for p in pts]
    eps = [p[0] for p in pts]
    best_idx = min(range(n), key=lambda i: ys[i])
    best_stale = (n - 1) - best_idx
    rise_abs = ys[-1] - ys[best_idx]
    if rise_abs <= 0.0:                       # at/below best -> not rising (converging)
        return None
    if best_stale < k:                        # rise has not persisted >=k verdicts
        return None
    if not _non_decreasing(ys[-(k + 1):]):    # recent window not monotone up
        return None
    win_eps = eps[best_idx:]
    win_ys = ys[best_idx:]
    rel_slope = _rel_slope_per_ep(win_eps, win_ys)
    if rel_slope <= rise_rel_eps:             # below the calibrated flat gate -> plateau jitter
        return None
    return {
        "n": n, "best_idx": best_idx, "best_stale": best_stale,
        "best_epoch": int(eps[best_idx]), "epoch_latest": int(eps[-1]),
        "value_latest": ys[-1], "value_best": ys[best_idx], "rise_abs": rise_abs,
        "rel_slope_per_ep": rel_slope, "win_ep_lo": win_eps[0], "win_ep_hi": win_eps[-1],
    }


def _per_class_alarms(rows: list[dict], *, k: int, rise_rel_eps: float,
                      gap_target: float) -> list[dict]:
    """Run the rising test on each d_seg_by_class channel present in the same-stage
    rows. Returns one row per RISING class (may be empty). UNIDENTIFIABLE when no row
    carries ``d_seg_by_class`` -> empty list (never fabricated from scalar d_seg)."""
    n_classes = 0
    for v in rows:
        bc = v.get("d_seg_by_class")
        if isinstance(bc, (list, tuple)):
            n_classes = max(n_classes, len(bc))
    out: list[dict] = []
    for c in range(n_classes):
        series = [{"epoch": v["epoch"], "_c": bc[c]}
                  for v in rows
                  if isinstance((bc := v.get("d_seg_by_class")), (list, tuple))
                  and len(bc) > c and isinstance(bc[c], (int, float))
                  and isinstance(v.get("epoch"), (int, float))]
        m = _rising_channel(series, "_c", k=k, rise_rel_eps=rise_rel_eps)
        if m is None:
            continue
        remaining = max(m["value_latest"] - gap_target, _GAP_FLOOR)
        name = _CLASS_NAMES[c] if c < len(_CLASS_NAMES) else f"class{c}"
        out.append({
            "class_id": c, "class_name": name,
            "d_seg_latest": m["value_latest"], "d_seg_best": m["value_best"],
            "best_epoch": m["best_epoch"], "best_stale_verdicts": m["best_stale"],
            "rise_abs": m["rise_abs"],
            "rise_rel_value": (m["rise_abs"] / m["value_best"]) if m["value_best"] else 0.0,
            "rise_rel_gap": m["rise_abs"] / remaining,
            "rel_slope_per_ep": m["rel_slope_per_ep"],
            "reason": (f"class {c} ({name}) d_seg rising {m['value_best']:.5f}@ep"
                       f"{m['best_epoch']} -> {m['value_latest']:.5f}@ep{m['epoch_latest']} "
                       f"over {m['best_stale']} verdict(s) (rel-slope {m['rel_slope_per_ep']:+.2e}/ep "
                       f"> flat gate {rise_rel_eps:.0e})"),
        })
    return out


def verdict_trend_alarm(
    verdicts: list[dict], loss_rows: list[dict] | None = None, *,
    k: int = 2, gap_target: float = DEFAULT_GAP_TARGET,
    rise_rel_eps: float = RISE_REL_EPS,
    decouple_rel_gap_floor: float = DECOUPLE_REL_GAP_FLOOR,
) -> VerdictTrendAlarm:
    """Detect a materially RISING advisory verdict d_seg (and TRAIN-VERDICT DECOUPLING)
    over the latest within-stage window -- the false-green ``classify_trajectory`` misses.

    ``verdicts``: the trainer's ``{"stage":"verdict", ...}`` rows (must carry ``d_seg`` +
    ``epoch``; ``ep_loss`` and ``d_seg_by_class`` used when present). ``loss_rows``:
    optional explicit train-loss series; when None the verdict rows' ``ep_loss`` is the
    train-seg-loss surrogate. ``k``: consecutive-verdict persistence threshold. See the
    module docstring for the two DERIVED thresholds. Pure: same rows -> same verdict."""
    k = max(1, int(k))
    stage, rows = _latest_stage_rows(verdicts, "d_seg")
    if len(rows) < k + 1:
        return VerdictTrendAlarm(
            classification=RISING_VERDICT_UNIDENTIFIABLE, stage=stage, n_stage=len(rows),
            k=k, epoch_latest=(int(rows[-1]["epoch"]) if rows else None),
            d_seg_latest=(float(rows[-1]["d_seg"]) if rows else None),
            d_seg_best=None, best_epoch=None, best_stale_verdicts=0, monotone_recent=False,
            rise_abs=0.0, rise_rel_value=0.0, rise_rel_gap=0.0, remaining_gap=0.0,
            gap_target=gap_target, rel_slope_per_ep=0.0, rise_rel_eps=rise_rel_eps,
            train_loss_slope_per_ep=None, train_loss_decoupled=False, ema_lag_plausible=False,
            per_class_alarms=(),
            reason=f"only {len(rows)} same-stage verdict row(s) (need >= {k + 1}) — wait for more evals",
            evidence=(f"stage={stage!r}, {len(rows)} usable d_seg rows",))

    ys = [float(v["d_seg"]) for v in rows]
    eps = [float(v["epoch"]) for v in rows]
    n = len(rows)
    best_idx = min(range(n), key=lambda i: ys[i])
    best_stale = (n - 1) - best_idx
    rise_abs = ys[-1] - ys[best_idx]
    monotone_recent = _non_decreasing(ys[-(k + 1):])
    win_eps, win_ys = eps[best_idx:], ys[best_idx:]
    rel_slope = _rel_slope_per_ep(win_eps, win_ys)
    remaining_gap = max(ys[-1] - gap_target, _GAP_FLOOR)
    rise_rel_value = (rise_abs / ys[best_idx]) if ys[best_idx] > 0 else 0.0
    rise_rel_gap = (rise_abs / remaining_gap) if rise_abs > 0 else 0.0

    per_class = tuple(_per_class_alarms(rows, k=k, rise_rel_eps=rise_rel_eps,
                                        gap_target=gap_target))

    core = _rising_channel(rows, "d_seg", k=k, rise_rel_eps=rise_rel_eps)
    if core is None:
        # not a material rise -> silent (converging / plateau / jitter is another lane).
        why = ("d_seg at/below best (not rising)" if rise_abs <= 0 else
               (f"rise not persisted >= {k} verdicts (best_stale={best_stale})"
                if best_stale < k else
                ("recent window not monotone-up" if not monotone_recent else
                 f"rel-slope {rel_slope:+.2e}/ep below flat gate {rise_rel_eps:.0e} "
                 "(plateau jitter, not a material rise)")))
        return VerdictTrendAlarm(
            classification=NO_ALARM, stage=stage, n_stage=n, k=k,
            epoch_latest=int(eps[-1]), d_seg_latest=ys[-1], d_seg_best=ys[best_idx],
            best_epoch=int(eps[best_idx]), best_stale_verdicts=best_stale,
            monotone_recent=monotone_recent, rise_abs=rise_abs,
            rise_rel_value=rise_rel_value, rise_rel_gap=rise_rel_gap,
            remaining_gap=remaining_gap, gap_target=gap_target,
            rel_slope_per_ep=rel_slope, rise_rel_eps=rise_rel_eps,
            train_loss_slope_per_ep=None, train_loss_decoupled=False,
            ema_lag_plausible=False, per_class_alarms=per_class,
            reason="no material rising-verdict trend: " + why,
            evidence=(f"d_seg {ys[best_idx]:.6f}@ep{int(eps[best_idx])} (best) -> "
                      f"{ys[-1]:.6f}@ep{int(eps[-1])} [stage={stage}]",
                      f"rise-window rel-slope {rel_slope:+.2e}/ep vs flat gate {rise_rel_eps:.0e}"))

    # material rise confirmed. Is the train loss DECOUPLED (descending while verdict rises)?
    loss_slope, loss_src = _train_loss_slope_over_window(
        rows[best_idx:], loss_rows, core["win_ep_lo"], core["win_ep_hi"])
    loss_decoupled = (loss_slope is not None) and (loss_slope <= 0.0)

    # EMA-lag disambiguation (mechanism-derived): only when the rise JUST crossed
    # (best_stale == k), the run is very early (<= k+2 same-stage verdicts), AND the
    # magnitude is below the material floor. Else a persisted+material rise is real.
    ema_lag_plausible = (best_stale == k and n <= k + 2
                         and rise_rel_gap < decouple_rel_gap_floor)

    if ema_lag_plausible:
        cls = RISING_VERDICT
        reason = (f"verdict d_seg RISING {ys[best_idx]:.5f}@ep{int(eps[best_idx])} -> "
                  f"{ys[-1]:.5f}@ep{int(eps[-1])} (+{rise_rel_value * 100:.1f}% of value, "
                  f"+{rise_rel_gap * 100:.1f}% of remaining gap) but JUST crossed + early + "
                  "below the material floor — POSSIBLE EMA-shadow catch-up transient. WATCH; "
                  "re-check next verdict before treating as a trend.")
    elif loss_decoupled:
        cls = TRAIN_VERDICT_DECOUPLING
        reason = (f"TRAIN<->VERDICT DECOUPLING: verdict d_seg RISING "
                  f"{ys[best_idx]:.5f}@ep{int(eps[best_idx])} -> {ys[-1]:.5f}@ep{int(eps[-1])} "
                  f"(+{rise_rel_gap * 100:.1f}% of remaining gap to {gap_target:g}) over "
                  f"{best_stale} verdict(s) WHILE the train seg-loss DESCENDS "
                  f"({loss_slope:+.3e}/ep). More of the same gradient is NOT lowering the argmax "
                  "verdict — the scalar 'converging/plateau' green is FALSE. Do NOT treat as "
                  "healthy descent; investigate the curriculum/stage (CE won't help here).")
    else:
        cls = RISING_VERDICT
        reason = (f"verdict d_seg RISING {ys[best_idx]:.5f}@ep{int(eps[best_idx])} -> "
                  f"{ys[-1]:.5f}@ep{int(eps[-1])} (+{rise_rel_gap * 100:.1f}% of remaining gap) "
                  f"over {best_stale} verdict(s); train loss not cleanly decoupled "
                  f"(slope={loss_slope}). Rising is NOT converging — surface, do not advance/stop "
                  "on a 'converging' green.")

    ev = (f"d_seg {ys[best_idx]:.6f}@ep{int(eps[best_idx])} (best) -> {ys[-1]:.6f}@ep"
          f"{int(eps[-1])} [stage={stage}], rise-window rel-slope {rel_slope:+.2e}/ep "
          f"(> flat gate {rise_rel_eps:.0e})",
          f"rise +{rise_abs:.6f} = +{rise_rel_value * 100:.1f}% of value = "
          f"+{rise_rel_gap * 100:.2f}% of remaining gap ({remaining_gap:.6f} to {gap_target:g})",
          f"train-loss slope over the same window = "
          f"{('%+.3e/ep' % loss_slope) if loss_slope is not None else 'unavailable'} "
          f"[{loss_src}]",
          f"per-class rising: {', '.join(r['class_name'] for r in per_class) or 'none/UNIDENTIFIABLE'}")
    return VerdictTrendAlarm(
        classification=cls, stage=stage, n_stage=n, k=k, epoch_latest=int(eps[-1]),
        d_seg_latest=ys[-1], d_seg_best=ys[best_idx], best_epoch=int(eps[best_idx]),
        best_stale_verdicts=best_stale, monotone_recent=monotone_recent, rise_abs=rise_abs,
        rise_rel_value=rise_rel_value, rise_rel_gap=rise_rel_gap, remaining_gap=remaining_gap,
        gap_target=gap_target, rel_slope_per_ep=rel_slope, rise_rel_eps=rise_rel_eps,
        train_loss_slope_per_ep=loss_slope, train_loss_decoupled=loss_decoupled,
        ema_lag_plausible=ema_lag_plausible, per_class_alarms=per_class,
        reason=reason, evidence=ev)


def format_verdict_trend_line(alarm: VerdictTrendAlarm | dict) -> str:
    """One-line digest formatter (unit-testable without touching real state)."""
    a = alarm.to_dict() if isinstance(alarm, VerdictTrendAlarm) else alarm
    cls = str(a.get("classification"))
    ep = a.get("epoch_latest")
    if cls == RISING_VERDICT_UNIDENTIFIABLE:
        return f"verdict-trend: ep{ep} UNIDENTIFIABLE ({a.get('reason', '')[:70]}) [advisory]"
    if cls == NO_ALARM:
        return (f"verdict-trend: ep{ep} d_seg {a.get('d_seg_latest'):.5f} NO-ALARM "
                "(not materially rising) [advisory]")
    pcs = a.get("per_class_alarms") or []
    pc = (" per-class↑: " + ",".join(f"{r['class_name']}({r['rise_rel_value'] * 100:.0f}%)"
                                     for r in pcs)) if pcs else ""
    tag = ("TRAIN↓/VERDICT↑ DECOUPLING" if cls == TRAIN_VERDICT_DECOUPLING
           else ("RISING (EMA-lag?)" if a.get("ema_lag_plausible") else "RISING-VERDICT"))
    return (f"verdict-trend: ep{ep} d_seg {a.get('d_seg_best'):.5f}→{a.get('d_seg_latest'):.5f} "
            f"{tag} +{a.get('rise_rel_gap', 0) * 100:.1f}% of remaining gap over "
            f"{a.get('best_stale_verdicts')} verdict(s){pc} [advisory NON-PROMOTABLE]")


# --------------------------------------------------------------------------- #
# P4 CANARY suite (design_philosophies_eightfold P4 "no meter without a canary"): a $0
# positive + negative control that PROVES this meter can see a decoupling before its
# readings gate anything. Pattern mirrors sigma_min_plateau.canary_suite. This meter is
# ADVISORY-only (never a score authority), but P4 binds every measurement surface.
# --------------------------------------------------------------------------- #
def synthetic_decoupling_verdicts(
    *, best_dseg: float = 0.010, n_rise: int = 6, rise_step: float = 0.002,
    ep_step: int = 4, loss_start: float = 400.0, loss_step: float = -18.0,
    seg_form: str = "tau_softplus",
) -> list[dict]:
    """The SYNTHETIC POSITIVE control: verdict d_seg descends to a best, then RISES for ``n_rise``
    verdicts WHILE the train seg-loss (``ep_loss``) keeps DESCENDING — a textbook TRAIN<->VERDICT
    DECOUPLING the meter MUST fire on. Two lead-in descent rows establish the best; the rise is
    persisted + material (well above DECOUPLE_REL_GAP_FLOOR) so it is not the EMA-lag transient lane."""
    rows: list[dict] = []
    ep = 0
    loss = loss_start
    # descent to the best
    for d in (best_dseg + 0.010, best_dseg + 0.005, best_dseg):
        rows.append({"stage": "verdict", "seg_form": seg_form, "epoch": ep, "d_seg": d, "ep_loss": loss})
        ep += ep_step
        loss += loss_step
    # persisted material rise while loss keeps descending
    for i in range(1, n_rise + 1):
        rows.append({"stage": "verdict", "seg_form": seg_form, "epoch": ep,
                     "d_seg": best_dseg + i * rise_step, "ep_loss": loss})
        ep += ep_step
        loss += loss_step
    return rows


def synthetic_codescending_verdicts(
    *, start_dseg: float = 0.020, step: float = -0.0016, n: int = 9, ep_step: int = 4,
    loss_start: float = 400.0, loss_step: float = -18.0, seg_form: str = "tau_softplus",
) -> list[dict]:
    """The NEGATIVE control: verdict d_seg descends monotonically WHILE ep_loss also descends (a clean,
    co-descending, converging run). The meter must NOT fire (nothing is rising / decoupling)."""
    rows: list[dict] = []
    for i in range(n):
        rows.append({"stage": "verdict", "seg_form": seg_form, "epoch": i * ep_step,
                     "d_seg": max(1e-4, start_dseg + i * step), "ep_loss": loss_start + i * loss_step})
    return rows


@dataclass(frozen=True)
class VerdictTrendCanaryResult:
    """Outcome of the $0 verdict-trend canary. ``passed`` iff the POSITIVE (decoupling) control fires
    TRAIN_VERDICT_DECOUPLING AND the NEGATIVE (co-descending) control does NOT fire — the P4 proof that
    this meter can distinguish a real decoupling from a clean converging run before it gates anything."""

    passed: bool
    positive_fired_decoupling: bool     # must be True
    negative_fired: bool                # must be False
    positive_classification: str
    negative_classification: str
    reason: str


def canary_suite() -> VerdictTrendCanaryResult:
    """Run the $0 positive + negative controls for the verdict-trend alarm (P4 known-effect canary)."""
    pos = verdict_trend_alarm(synthetic_decoupling_verdicts())
    neg = verdict_trend_alarm(synthetic_codescending_verdicts())
    pos_ok = pos.classification == TRAIN_VERDICT_DECOUPLING
    neg_quiet = not neg.fired()
    passed = pos_ok and neg_quiet
    reason = ("canary PASS: synthetic decoupling FIRES TRAIN_VERDICT_DECOUPLING and a clean "
              "co-descending run stays quiet — meter trustworthy"
              if passed else
              "canary FAIL: " + ("" if pos_ok else "decoupling positive control did NOT fire; ")
              + ("" if neg_quiet else "co-descending negative control FIRED (false positive); ")
              + "→ meter UNTRUSTED")
    return VerdictTrendCanaryResult(
        passed=passed, positive_fired_decoupling=pos_ok, negative_fired=neg.fired(),
        positive_classification=pos.classification, negative_classification=neg.classification,
        reason=reason)
