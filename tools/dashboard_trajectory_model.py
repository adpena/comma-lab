#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sophisticated, DATA-DERIVED trajectory + wall-clock telemetry for the level-set
witness dashboard — pure numpy, fully testable, ZERO hardcoded time/score constants.

Why this exists (operator 2026-06-30, relayed): the prior dashboard projection was a
naive linear extrapolation of d_seg, and the wall-clock telemetry carried CE's
seconds/epoch across the (slower) Muon stage. Both are telemetry-accuracy violations
("confident-wrong is the worst failure"). This module replaces them with the lab's
actual deep-math understanding, and FLAGS every estimate / low-confidence fit.

Two pillars, both pure functions on the live verdict series + the run's own schedule:

1. d_seg DESCENT MODEL — CRITICAL-SLOWING POWER LAW, not linear/exp. The curriculum is
   deterministic annealing (Rose 1998, F = D − T·H) descending an Arimoto–Blahut class;
   near the rate-distortion topological transition the tail is critical-slowing
   (Agmon–Benger–Ordentlich–Tishby ISIT 2021, arXiv:2103.02646):
       d_seg(t) ≈ c + a·(t − t0)^(−α),   c = projected asymptote d_seg_∞, α = exponent.
   ``fit_critical_slowing`` fits (c, a, α, t0) robustly (grid over the nonlinear (α, t0),
   closed-form linear LS for (c, a)), returns R²/confidence and FLAGS low-confidence
   (few points / early / high residual). Goal-ETA uses this model WITH a band, and says
   "won't reach at current trajectory" honestly when the asymptote is above the target.

2. STAGE-AWARE WALL-CLOCK — epochs in different curriculum stages take different wall
   time (Muon's Newton–Schulz orthogonalization is slower than CE). ``per_stage_seconds_
   per_epoch`` MEASURES seconds/epoch per stage from the verdict timestamps; stages not
   yet entered get a clearly-FLAGGED estimate (Muon ≈ k× the slowest measured AdamW-stage
   rate). ``completion_eta`` sums the right per-stage rate over the remaining epochs;
   ``current_stage_cadence`` is the CURRENT stage's rate × eval-every, recomputed each
   tick so it never carries a stale-stage rate across a boundary.

Grounding memo: ``.omx/research/post_muon_application_plan_optimal_form_20260630T1710Z.md``
(critical-slowing / annealing) + the #188 trajectory-dynamics line. Generalizes to ANY
run (n200 / n600 / residual-INR): all inputs come from the run's verdict series + its
own meta.schedule; nothing is n600-special-cased.

AUTHORITY: advisory ([macOS-MLX] NON-PROMOTABLE). The contest score is byte-closed on
contest-CPU/CUDA; the frontier pointer is 0.19110 and UNMOVED. A projection is a MEANS.
"""
from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any

import numpy as np

# Canonical curriculum stage order (CE → tau → l7 → Muon). The Muon stage is an
# OPTIMIZER switch (Newton–Schulz), the others are loss-form switches.
STAGE_ORDER: tuple[str, ...] = ("CE", "tau", "l7", "Muon")

# Dimensionless POLICY priors (NOT time values). Muon epochs are slower than the AdamW
# stages; before Muon is entered we estimate its rate as this multiple of the slowest
# MEASURED AdamW-stage rate, and FLAG it as an estimate. Tunable, clearly labeled.
_MUON_SLOWDOWN_PRIOR = 1.6
# Minimum verdict points before the power-law fit is even attempted / trusted.
_MIN_FIT_POINTS = 5
_HIGH_CONF_POINTS = 8
_HIGH_CONF_R2 = 0.90
_MED_CONF_R2 = 0.75


# ────────────────────────── timestamp + stage helpers ──────────────────────────
def parse_ts(v: Any) -> float | None:
    """Parse an embedded UTC ``ts`` string into epoch seconds. None when absent/bad."""
    t = v.get("ts") if isinstance(v, dict) else v
    if not isinstance(t, str) or not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def stage_at_epoch(epoch: float, schedule: dict) -> str:
    """Stage name for ``epoch`` from the run's OWN schedule boundaries
    (``tau_start`` / ``l7_start`` / ``muon_start``; any may be None when that stage
    does not exist for this run). Generalizes to non-curriculum runs (all None → CE)."""
    tau = schedule.get("tau_start")
    l7 = schedule.get("l7_start")
    muon = schedule.get("muon_start")
    if muon is not None and epoch >= muon:
        return "Muon"
    if l7 is not None and epoch >= l7:
        return "l7"
    if tau is not None and epoch >= tau:
        return "tau"
    return "CE"


def _stage_segments(schedule: dict, total_epochs: int) -> list[tuple[str, int, int]]:
    """Ordered NON-OVERLAPPING ``[(stage, start, end)]`` epoch segments [start, end)
    covering [0, total_epochs), derived from the PRECEDENCE-correct ``stage_at_epoch``.

    Deriving the windows from ``stage_at_epoch`` (rather than a static CE→tau→l7→Muon
    boundary chain) makes them robust to OUT-OF-ORDER or DISABLED boundaries. E.g. when
    ``muon_start`` (726) < a disabled ``l7_start`` (1000), the old naive chain gave tau
    ``[300, 1000)`` OVERLAPPING Muon ``[726, 1000)``; here tau is correctly ``[300, 726)``
    and the unreachable l7 window collapses away. Consecutive same-stage windows merge.
    """
    if total_epochs is None or int(total_epochs) <= 0:
        return []
    total = int(total_epochs)
    # candidate step points: 0, total, and each enabled in-range stage boundary
    points = {0, total}
    for key in ("tau_start", "l7_start", "muon_start"):
        b = schedule.get(key)
        if b is not None and 0 < int(b) < total:
            points.add(int(b))
    ordered = sorted(points)
    segs: list[tuple[str, int, int]] = []
    for a, b in itertools.pairwise(ordered):
        if b <= a:
            continue
        stage = stage_at_epoch(a, schedule)  # precedence-correct (Muon > l7 > tau > CE)
        if segs and segs[-1][0] == stage:
            segs[-1] = (stage, segs[-1][1], b)  # merge a disabled/shadowed boundary
        else:
            segs.append((stage, a, b))
    return segs


# ────────────────────────── stage-aware wall-clock ──────────────────────────
def per_stage_seconds_per_epoch(verdicts: list[dict], schedule: dict,
                                muon_slowdown: float = _MUON_SLOWDOWN_PRIOR) -> dict:
    """MEASURED seconds/epoch per curriculum stage, from consecutive verdict ``ts``
    gaps assigned to the stage of the interval MIDPOINT.

    Returns ``{stage: {"spe": float, "measured": bool, "n_intervals": int, "source": str}}``
    for every stage in the schedule. Stages with >=1 timed interval are ``measured``;
    stages NOT yet entered get a clearly-FLAGGED estimate:
      * "Muon" -> ``muon_slowdown`` × the SLOWEST measured AdamW-stage (CE/tau/l7) rate
        ("est:muon_xNN" source) — NEVER silently CE's rate.
      * other un-entered stage -> the nearest measured stage's rate ("est:carry").
      * no measured data at all -> empty (caller shows "calibrating").

    Pure: only the run's own verdict ts + schedule; no hardcoded times.
    """
    # build (epoch, ts) for verdicts that carry a usable timestamp
    timed = sorted(
        [(int(v["epoch"]), parse_ts(v)) for v in verdicts
         if isinstance(v, dict) and isinstance(v.get("epoch"), (int, float)) and parse_ts(v) is not None],
        key=lambda t: t[0],
    )
    measured: dict[str, list[float]] = {}
    for (e0, t0), (e1, t1) in zip(timed, timed[1:]):
        de = e1 - e0
        dt = t1 - t0
        if de <= 0 or dt <= 0:
            continue
        spe = dt / de
        mid = (e0 + e1) / 2.0
        st = stage_at_epoch(mid, schedule)
        measured.setdefault(st, []).append(spe)

    out: dict[str, dict] = {}
    # measured stages first
    for st, vals in measured.items():
        out[st] = {"spe": float(np.median(vals)), "measured": True,
                   "n_intervals": len(vals), "source": "measured"}

    # which stages exist for this run
    segs = _stage_segments(schedule, total_epochs=int(schedule.get("epochs") or 0) or 10**9)
    existing = [name for (name, _a, _b) in segs] or list(STAGE_ORDER)

    adamw_rates = [out[s]["spe"] for s in ("CE", "tau", "l7") if s in out]
    slowest_adamw = max(adamw_rates) if adamw_rates else None
    any_measured = [out[s]["spe"] for s in out]

    for st in existing:
        if st in out:
            continue
        if st == "Muon" and slowest_adamw is not None:
            out[st] = {"spe": float(slowest_adamw * muon_slowdown), "measured": False,
                       "n_intervals": 0, "source": f"est:muon_x{muon_slowdown:g}"}
        elif any_measured:
            # carry the nearest measured rate (the slowest, conservative) — flagged
            out[st] = {"spe": float(max(any_measured)), "measured": False,
                       "n_intervals": 0, "source": "est:carry"}
        # else: leave absent -> caller treats as calibrating
    return out


def current_stage_cadence(current_epoch: float, schedule: dict, stage_spe: dict,
                          eval_every: int) -> tuple[float | None, str, str]:
    """The CURRENT stage's verdict cadence = stage_spe[current_stage] × eval_every,
    recomputed from ``current_epoch`` each call (so it never carries a stale-stage
    rate across a boundary). Returns ``(cadence_s_or_None, stage, source)`` where
    source ∈ {"measured","estimate","calibrating"}."""
    if not eval_every or eval_every <= 0:
        return None, stage_at_epoch(current_epoch, schedule), "calibrating"
    st = stage_at_epoch(current_epoch, schedule)
    info = stage_spe.get(st)
    if not info or not info.get("spe"):
        return None, st, "calibrating"
    src = "measured" if info.get("measured") else "estimate"
    return float(info["spe"]) * float(eval_every), st, src


def completion_eta(current_epoch: float, schedule: dict, stage_spe: dict,
                   total_epochs: int) -> dict:
    """ETA to ``total_epochs`` = Σ over remaining epochs of THAT epoch's stage rate
    (measured where available, FLAGGED estimate for not-yet-entered stages).

    Returns ``{"total_s","measured_s","estimated_s","has_estimate","ok",
    "per_stage":[{stage,remaining_epochs,spe,measured,seconds}], "reason"?}``.
    ``ok=False`` (reason "calibrating") when no stage rate is known yet."""
    if not total_epochs or total_epochs <= current_epoch:
        return {"ok": True, "total_s": 0.0, "measured_s": 0.0, "estimated_s": 0.0,
                "has_estimate": False, "per_stage": []}
    segs = _stage_segments(schedule, int(total_epochs))
    if not segs:
        segs = [("CE", 0, int(total_epochs))]
    per_stage: list[dict] = []
    total = meas = est = 0.0
    any_rate = False
    for name, a, b in segs:
        lo = max(a, int(np.ceil(current_epoch)))
        hi = b
        remaining = hi - lo
        if remaining <= 0:
            continue
        info = stage_spe.get(name)
        if not info or not info.get("spe"):
            # unknown rate for a remaining stage -> contributes an unflagged gap; mark
            per_stage.append({"stage": name, "remaining_epochs": int(remaining),
                              "spe": None, "measured": False, "seconds": None})
            continue
        any_rate = True
        secs = float(info["spe"]) * remaining
        total += secs
        if info.get("measured"):
            meas += secs
        else:
            est += secs
        per_stage.append({"stage": name, "remaining_epochs": int(remaining),
                          "spe": float(info["spe"]), "measured": bool(info.get("measured")),
                          "seconds": secs})
    if not any_rate:
        return {"ok": False, "reason": "calibrating", "per_stage": per_stage}
    return {"ok": True, "total_s": total, "measured_s": meas, "estimated_s": est,
            "has_estimate": est > 0.0, "per_stage": per_stage}


# ────────────────────────── d_seg critical-slowing model ──────────────────────────
def fit_critical_slowing(epochs, dseg, min_points: int = _MIN_FIT_POINTS) -> dict:
    """Fit the critical-slowing power law ``d_seg(t) = c + a·(t − t0)^(−α)`` robustly.

    Method: grid over the NONLINEAR params (α ∈ [0.15, 3.0], t0 < first epoch), and for
    each grid point solve the LINEAR least squares for (c, a) on the design
    ``y = c + a·x``, ``x = (t − t0)^(−α)``. Pick the min-SSE solution with ``a > 0``
    (a genuine decay toward the asymptote ``c = d_seg_∞``).

    Returns ``{"ok","asymptote","a","alpha","t0","r2","residual_std","n","confidence",
    "reason"}``. ``ok=False`` (with reason) when there are too few points OR no decaying
    fit exists — the dashboard then shows "calibrating", never a confident-wrong curve."""
    e = np.asarray(epochs, dtype=float)
    y = np.asarray(dseg, dtype=float)
    m = np.isfinite(e) & np.isfinite(y)
    e, y = e[m], y[m]
    order = np.argsort(e)
    e, y = e[order], y[order]
    # de-dup identical epochs (keep last)
    if e.size > 1:
        keep = np.concatenate([np.diff(e) != 0, [True]])
        e, y = e[keep], y[keep]
    n = int(e.size)
    if n < min_points:
        return {"ok": False, "reason": f"need >= {min_points} points (have {n})", "n": n}

    first, last = float(e[0]), float(e[-1])
    span = max(last - first, 1.0)
    sst = float(np.sum((y - y.mean()) ** 2)) or 1e-30

    def _grid_best(alphas, d_offsets):
        """Min-SSE (c, a) over a grid of the nonlinear (α, offset d = first − t0);
        (c, a) solved in closed form per grid point. Requires a > 0 (decay)."""
        bst = None
        for d0 in d_offsets:
            if d0 <= 0:
                continue
            t0 = first - d0
            base = e - t0  # all > 0 by construction
            for al in alphas:
                x = base ** (-al)
                A = np.column_stack([np.ones_like(x), x])
                try:
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                except Exception:
                    continue
                c, a = float(coef[0]), float(coef[1])
                if a <= 0:  # require decay toward the asymptote
                    continue
                sse = float(np.sum((y - (c + a * x)) ** 2))
                if bst is None or sse < bst["sse"]:
                    bst = {"sse": sse, "c": c, "a": a, "alpha": float(al), "d0": float(d0),
                           "t0": float(t0)}
        return bst

    # Two-stage search: coarse grid, then REFINE around the coarse optimum to lift the
    # (α, t0, c) degeneracy (a coarse-only grid leaves the projected asymptote biased).
    best = _grid_best(np.linspace(0.15, 3.0, 32),
                      np.geomspace(max(span * 0.02, 0.5), span * 50.0, 28))
    if best is None:
        return {"ok": False, "reason": "no decaying power-law fit (curve not descending)", "n": n}
    for _ in range(2):  # successive local refinements
        a_lo, a_hi = max(0.1, best["alpha"] * 0.6), best["alpha"] * 1.6
        d_lo, d_hi = best["d0"] * 0.4, best["d0"] * 2.5
        ref = _grid_best(np.linspace(a_lo, a_hi, 31),
                         np.geomspace(max(d_lo, 1e-3), d_hi, 31))
        if ref is not None and ref["sse"] <= best["sse"]:
            best = ref
    best["resid_std"] = float(np.sqrt(best["sse"] / max(n - 2, 1)))

    r2 = 1.0 - best["sse"] / sst
    if n >= _HIGH_CONF_POINTS and r2 >= _HIGH_CONF_R2:
        conf = "high"
    elif n >= min_points and r2 >= _MED_CONF_R2:
        conf = "medium"
    else:
        conf = "low"
    # asymptote can come out slightly negative from LS noise; clamp the REPORTED floor
    # at 0 (a probability/disagreement can't be < 0) but keep the raw c for projection.
    return {"ok": True, "asymptote": max(best["c"], 0.0), "asymptote_raw": best["c"],
            "a": best["a"], "alpha": best["alpha"], "t0": best["t0"],
            "r2": float(r2), "residual_std": best["resid_std"], "n": n, "confidence": conf}


def _crosses_epoch(c: float, a: float, alpha: float, t0: float, target: float) -> float | None:
    """Epoch t where c + a·(t − t0)^(−α) = target, or None if unreachable
    (target <= asymptote for a decaying curve)."""
    base = target - c
    if a <= 0 or base <= 0:
        return None
    return t0 + (a / base) ** (1.0 / alpha)


def project_goal_epoch(fit: dict, target: float, current_epoch: float) -> dict:
    """Project the epoch where d_seg crosses ``target`` using the fitted model, WITH a
    confidence band from the fit's residual std on the asymptote.

    Returns one of:
      * ``{"state":"calibrating"}``                          — fit not available
      * ``{"state":"asymptote_above", "asymptote":c}``       — won't reach (honest)
      * ``{"state":"reached"}``                              — already at/below target
      * ``{"state":"eta", "epoch", "eta_epochs", "epoch_lo","epoch_hi","band_note"}``
    Never a false-precise single number: the band reflects the asymptote uncertainty,
    and if the upper asymptote is above target we say so in ``band_note``.
    """
    if not fit.get("ok"):
        return {"state": "calibrating"}
    c = fit["asymptote_raw"]
    a, al, t0 = fit["a"], fit["alpha"], fit["t0"]
    sigma = fit.get("residual_std", 0.0) or 0.0
    if target >= c and (c + a * (max(current_epoch - t0, 1e-9)) ** (-al)) <= target:
        # current model value already at/below target
        return {"state": "reached", "asymptote": max(c, 0.0)}
    if c >= target:
        return {"state": "asymptote_above", "asymptote": max(c, 0.0)}
    t_mid = _crosses_epoch(c, a, al, t0, target)
    if t_mid is None or t_mid <= current_epoch:
        if t_mid is not None and t_mid <= current_epoch:
            return {"state": "reached", "asymptote": max(c, 0.0)}
        return {"state": "asymptote_above", "asymptote": max(c, 0.0)}
    # band: vary the asymptote by ±sigma (the dominant uncertainty for the crossing).
    # lower asymptote (c - sigma) -> reaches target SOONER; higher (c + sigma) -> LATER
    # or never. epoch_lo (optimistic) uses c - sigma, epoch_hi (pessimistic) uses c + sigma.
    band_note = ""
    t_lo = _crosses_epoch(c - sigma, a, al, t0, target)
    t_hi = _crosses_epoch(c + sigma, a, al, t0, target)
    epoch_lo = t_lo if (t_lo is not None and t_lo > current_epoch) else t_mid
    if t_hi is None or (c + sigma) >= target:
        epoch_hi = None
        band_note = "upper band: asymptote within 1σ is ABOVE target (may not reach)"
    else:
        epoch_hi = t_hi
    return {"state": "eta", "epoch": float(t_mid), "eta_epochs": float(t_mid - current_epoch),
            "epoch_lo": (float(epoch_lo) if epoch_lo is not None else None),
            "epoch_hi": (float(epoch_hi) if epoch_hi is not None else None),
            "asymptote": max(c, 0.0), "band_note": band_note}


def fit_linear_trend(epochs, vals) -> dict:
    """Simple robust linear fit ``v = m·e + b`` over recent points (for the bytes
    trend). Returns ``{"ok","slope","intercept","last","n"}``; ok=False if < 2 points."""
    e = np.asarray(epochs, dtype=float)
    v = np.asarray(vals, dtype=float)
    m = np.isfinite(e) & np.isfinite(v)
    e, v = e[m], v[m]
    if e.size < 2:
        return {"ok": False, "n": int(e.size), "last": (float(v[-1]) if v.size else None)}
    A = np.column_stack([e, np.ones_like(e)])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return {"ok": True, "slope": float(coef[0]), "intercept": float(coef[1]),
            "last": float(v[-1]), "n": int(e.size)}


def project_bytes(verdicts: list[dict], project_to_epoch: float | None) -> dict:
    """Project the LEARNED-payload bytes at ``project_to_epoch`` from the live blob_bytes
    trend (linear, clamped >= 0). Falls back to the last value. Pure."""
    pts = [(v["epoch"], v.get("blob_bytes")) for v in verdicts
           if isinstance(v, dict) and isinstance(v.get("epoch"), (int, float))
           and isinstance(v.get("blob_bytes"), (int, float))]
    if not pts:
        return {"ok": False, "value": None}
    eps = [p[0] for p in pts]
    bys = [p[1] for p in pts]
    last = float(bys[-1])
    if project_to_epoch is None:
        return {"ok": True, "value": last, "source": "last"}
    fit = fit_linear_trend(eps[-min(len(eps), 8):], bys[-min(len(bys), 8):])
    if not fit.get("ok"):
        return {"ok": True, "value": last, "source": "last"}
    proj = fit["slope"] * float(project_to_epoch) + fit["intercept"]
    return {"ok": True, "value": max(0.0, float(proj)), "source": "linear", "last": last}


def project_implied_s(dseg_inf: float | None, bytes_proj: float | None,
                      sidecar_pose: float, archive_norm: float,
                      dseg_band: tuple[float | None, float | None] = (None, None)) -> dict:
    """Projected final implied_S from the MODEL:
        S = 100·d_seg_∞ + sqrt(10·sidecar_pose) + 25·bytes/archive_norm.
    Pose rides the SOLVED stored sidecar (telemetry accuracy — not the monitoring pose).
    Returns ``{"ok","value","value_lo","value_hi"}`` with the band propagated from the
    d_seg asymptote band when provided."""
    if dseg_inf is None or bytes_proj is None or not archive_norm:
        return {"ok": False, "value": None}
    pose_term = float(np.sqrt(10.0 * sidecar_pose))
    rate_term = 25.0 * float(bytes_proj) / float(archive_norm)

    def _s(ds):
        return 100.0 * float(ds) + pose_term + rate_term

    lo, hi = dseg_band
    return {"ok": True, "value": _s(dseg_inf),
            "value_lo": (_s(max(lo, 0.0)) if lo is not None else None),
            "value_hi": (_s(hi) if hi is not None else None),
            "pose_term": pose_term, "rate_term": rate_term}


# ────────────────────────── top-level projection builder ──────────────────────────
# ────────────────────── fresh-run A/B projection (measured-anchored) ──────────────────────
# DASHBOARD PASS 2026-07-04: the fresh SEEDED run's projection band, grounded ONLY in
# MEASURED anchors from the #205 real log (+ the field-level paint-seed measurement),
# with the below-CE-floor segment explicitly a HYPOTHESIS band ([projected/advisory]).
# Sources: FEED-04i (transition analysis, MEASURED on the #205 run.log) + the master
# lever ledger + witness_converged_to_flicker_floor (CE-best 0.004964@ep225; popout
# floor 0.00520). NO fabricated convergence promise: every anchor carries its label.
FRESH_RUN_ANCHORS: dict = {
    "ce_transient": {"label": "[measured #205]", "ep0_dseg": 0.745683, "ep25_dseg": 0.010299,
                     "note": "72x CE collapse (FiLM code-differentiation transient)"},
    "ce_floor": {"label": "[measured #205]", "lo": 0.0047, "hi": 0.0052,
                 "best": 0.004964, "best_epoch": 225, "popout_floor": 0.00520},
    "tau_erosion": {"label": "[measured #205]", "onset_slope": 4.68e-5, "steady_slope": 6.0e-6,
                    "onset_window_ep": 25, "net_rise_frac_by_ep450": 0.404,
                    "note": "d_seg RISING while ep_loss fell 148->130 (decoupling)"},
    "seed_delta": {"label": "[measured field-level]", "lane_fn_seeded": 0.00211,
                   "lane_fn_unseeded": 0.00583,
                   "note": "paint-seed lane_FN at init (field-level, pre-training)"},
    "fresh_tau": {"label": "[projected/advisory HYPOTHESIS]",
                  "note": "continued descent below the CE floor — the seed-fix thesis; "
                          "falsifiable when real fresh-run data lands"},
}


def build_fresh_run_projection(schedule: dict, goal_dseg: float, *,
                               anchors: dict | None = None,
                               total_epochs: int | None = None) -> dict:
    """The A/B projection pair the dashboard renders so the seed-fix thesis is VISUALLY
    falsifiable when real data lands:

      * ``eroding_205`` — the #205 MEASURED trajectory shape: CE transient (72x collapse,
        measured points) -> CE floor -> tau-stage EROSION at the MEASURED slopes
        (+4.68e-5/ep onset for the onset window, then +6.0e-6/ep steady). Continuation
        past the measured range is the measured-slope extrapolation (labeled).
      * ``fresh_lo`` / ``fresh_hi`` — the fresh SEEDED run's band: identical CE shape
        (same transient physics), then from the tau boundary a HYPOTHESIS band easing
        from the CE floor toward ``max(goal_dseg, ce_floor - seed_lane_FN_delta)``
        (lower edge) while the upper edge holds the CE floor (no-improvement bound).
        The below-CE-floor segment is a HYPOTHESIS — labeled, never a promise.

    The divergence point (where the two curves split) is the tau/MCF onset from the
    run's OWN schedule (``tau_start``; default 300). Pure; never raises for sane input.
    All series share the same ``epochs`` grid; values clamped > 0.
    """
    a = anchors or FRESH_RUN_ANCHORS
    tau = int(schedule.get("tau_start") or 300)
    total = int(total_epochs or schedule.get("epochs") or 1000)
    total = max(total, tau + 1)
    ce = a["ce_transient"]
    fl = a["ce_floor"]
    er = a["tau_erosion"]
    sd = a["seed_delta"]

    ep0, ep25 = float(ce["ep0_dseg"]), float(ce["ep25_dseg"])
    ce_best = float(fl["best"])
    ce_lo, ce_hi = float(fl["lo"]), float(fl["hi"])
    onset, steady = float(er["onset_slope"]), float(er["steady_slope"])
    onset_w = int(er["onset_window_ep"])
    seed_gain = max(float(sd["lane_fn_unseeded"]) - float(sd["lane_fn_seeded"]), 0.0)
    fresh_floor = max(float(goal_dseg), ce_lo - seed_gain)

    step = max(1, total // 400)
    grid = set(range(0, total + 1, step))
    # always include the measured-anchor epochs + boundaries so the rendered curve
    # passes EXACTLY through the anchors (and tests can assert them).
    grid.update(e for e in (0, 25, tau, tau + onset_w, total) if 0 <= e <= total)
    epochs = sorted(grid)

    def _ce_value(t: float) -> float:
        """CE-phase shape through the MEASURED points (log-space piecewise):
        (0, ep0) -> (25, ep25) -> (tau, ce_best)."""
        if t <= 0:
            return ep0
        if t <= 25:
            f = t / 25.0
            return float(np.exp(np.log(ep0) + f * (np.log(ep25) - np.log(ep0))))
        if t >= tau:
            return ce_best
        f = (t - 25.0) / max(tau - 25.0, 1.0)
        return float(np.exp(np.log(ep25) + f * (np.log(ce_best) - np.log(ep25))))

    eroding, f_lo, f_hi = [], [], []
    for t in epochs:
        if t < tau:
            v = _ce_value(t)
            eroding.append(v)
            f_lo.append(v)
            f_hi.append(v)
        else:
            dt = t - tau
            # #205: measured erosion slopes (onset window, then steady)
            rise = onset * min(dt, onset_w) + steady * max(dt - onset_w, 0)
            eroding.append(ce_best + rise)
            # fresh band: upper = CE floor held (no-improvement bound); lower eases
            # cosine from ce_best toward the hypothesis floor across the tau stage.
            span = max(total - tau, 1)
            f = 0.5 * (1.0 - float(np.cos(np.pi * min(dt / span, 1.0))))
            f_lo.append(max(ce_best + f * (fresh_floor - ce_best), 1e-9))
            f_hi.append(ce_hi)
    return {
        "ok": True,
        "epochs": epochs,
        "eroding_205": eroding,
        "fresh_lo": f_lo,
        "fresh_hi": f_hi,
        "divergence_epoch": tau,
        "fresh_floor": fresh_floor,
        "anchors": a,
        "labels": {
            "eroding_205": "#205 measured erosion (+4.68e-5/ep onset, +6.0e-6/ep steady) "
                           "[measured slopes; continuation extrapolated]",
            "fresh_band": "fresh SEEDED run — HYPOTHESIS band [projected/advisory; "
                          "falsifiable when real data lands]",
            "ce_phase": "CE phase [measured #205 shape: 72x collapse -> floor ~0.0047-0.0052]",
        },
        "authority": "[macOS-MLX advisory · NON-PROMOTABLE] · pointer 0.19110 UNMOVED",
    }


def build_projection(trajectory: list[dict], meta: dict, *, sidecar_pose: float,
                     archive_norm: float, eval_every: int | None = None) -> dict:
    """Assemble the full DATA-DERIVED projection the dashboard renders. ALL inputs come
    from the live verdict trajectory + the run's own ``meta.schedule``; nothing is
    n600-special-cased and nothing is a hardcoded time/score constant. Every estimate is
    flagged; low-confidence fits surface as 'calibrating'. Never raises (returns a
    ``{"ok":False,"reason":...}`` shell on any failure — telemetry must not crash the UI)."""
    try:
        sched = (meta or {}).get("schedule", {}) or {}
        total_epochs = int(sched.get("epochs") or 0)
        ee = int(eval_every or sched.get("eval_every") or 0) or None
        verdicts = [v for v in (trajectory or []) if isinstance(v, dict)
                    and isinstance(v.get("epoch"), (int, float))]
        if not verdicts:
            return {"ok": False, "reason": "no verdicts yet (calibrating)"}
        current_epoch = float(max(v["epoch"] for v in verdicts))

        # --- stage-aware wall-clock ---
        stage_spe = per_stage_seconds_per_epoch(verdicts, sched)
        cad_s, cur_stage, cad_src = current_stage_cadence(current_epoch, sched, stage_spe, ee or 0)
        comp = completion_eta(current_epoch, sched, stage_spe, total_epochs) if total_epochs else {"ok": False, "reason": "no total epochs"}

        # --- d_seg critical-slowing model ---
        eps = [v["epoch"] for v in verdicts if isinstance(v.get("d_seg"), (int, float))]
        dsg = [v["d_seg"] for v in verdicts if isinstance(v.get("d_seg"), (int, float))]
        fit = fit_critical_slowing(eps, dsg)

        goal = (meta or {}).get("goal_dseg")
        goal15 = (meta or {}).get("goal_dseg_15")
        goal_eta = project_goal_epoch(fit, float(goal), current_epoch) if (fit.get("ok") and goal) else {"state": "calibrating"}
        goal15_eta = project_goal_epoch(fit, float(goal15), current_epoch) if (fit.get("ok") and goal15) else {"state": "calibrating"}

        # --- implied_S projection via the model ---
        dseg_inf = fit.get("asymptote") if fit.get("ok") else None
        # d_seg band from residual std on the asymptote
        band = (None, None)
        if fit.get("ok"):
            sig = fit.get("residual_std", 0.0) or 0.0
            band = (max(fit["asymptote"] - sig, 0.0), fit["asymptote"] + sig)
        # project bytes to the goal-crossing epoch if known, else end of run, else last
        proj_to = None
        if goal_eta.get("state") == "eta":
            proj_to = goal_eta.get("epoch")
        elif total_epochs:
            proj_to = float(total_epochs)
        bytes_proj = project_bytes(verdicts, proj_to)
        s_proj = project_implied_s(dseg_inf, bytes_proj.get("value"), sidecar_pose,
                                   archive_norm, dseg_band=band) if dseg_inf is not None else {"ok": False}

        return {
            "ok": True,
            "current_epoch": current_epoch,
            "stage": cur_stage,
            "next_verdict_cadence_s": cad_s,
            "next_verdict_cadence_source": cad_src,
            "stage_seconds_per_epoch": stage_spe,
            "completion_eta": comp,
            "dseg_model": fit,
            "goal_eta": goal_eta,
            "goal15_eta": goal15_eta,
            "bytes_proj": bytes_proj,
            "implied_s_proj": s_proj,
        }
    except Exception as exc:  # never crash the live telemetry
        return {"ok": False, "reason": f"projection error: {exc}"}
