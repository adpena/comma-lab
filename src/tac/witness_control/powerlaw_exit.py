"""Power-law plateau/exit detector — the weak-KAM late-descent shape (§4 of the viscosity hunt).

Theory (viscosity_theory_alignment_hunt_20260707.md §4, weak KAM / Lax–Oleinik rates): away from
the Aubry-set analog convergence is exponential; ON it (the binding lane class: pinned margins,
zero effective mobility, no strict descent below tau*ln5) the rate is power-law O(1/t)-class.
Consequence: an EXPONENTIAL-window plateau detector fires EARLY on the binding class — declaring
exhaustion while a 1/t tail still pays ("meat left on the bone"). The fix implemented here:

    d_seg(t) = a + b * t^(-alpha)        (power-law tail; t = epochs since window start, >= 1)
    vs
    d_seg(t) = a + b * exp(-t / tau_e)   (the exponential alternative)

fit per class (or on the total when per-class telemetry is absent), compared by AIC, and the exit
rule is on the EXTRAPOLATED REMAINING MEAT — d(t_now) - d(horizon) under the preferred model —
not on a window slope.

Everything here is READ-ONLY + advisory (like the whole ``witness_control`` package): it never
mutates a run, never claims a score. Numbers derived from trajectories are [macOS advisory]
research signals; the pointer moves only via upstream/evaluate.py.

Fitting is deterministic (profile least squares on a log-spaced grid + golden-section refine; no
RNG except the seeded bootstrap) so the same trajectory always yields the same fit — the
deterministic-reproducibility spine applied to the detector itself.

Consumers:
* ``tools/fit_powerlaw_plateau_detector.py`` — the $0 retro-fit CLI (long900 + live-run logs).
* ``tac.witness_control.producer_bridge._powerlaw_exit_signal`` — the costate SENSE producer.
* A future ``ExitEvent(criterion="powerlaw_meat", ...)`` in ``tac.witness_dsl.curriculum_dsl``
  would call :func:`powerlaw_meat_exit` (gap-kind criterion, like ``marginal_dseg_floor``: no
  trainer support yet; conservative compile = fixed boundary + TrainerSupportGap). NOT wired
  here — two sibling agents own curriculum_dsl.py this wave; see the solver-pack memo.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "TailFit",
    "fit_exponential_tail",
    "fit_power_law_tail",
    "fit_tail_models",
    "powerlaw_meat_exit",
    "remaining_meat",
]


@dataclass(frozen=True)
class TailFit:
    """One fitted tail model d(t) = a + b*basis(t; p) on a (window-relative) trajectory.

    ``model`` is "power_law" (basis t^-alpha, p=alpha) or "exponential" (basis exp(-t/tau), p=tau).
    ``a`` is clamped to >= 0 (d_seg is non-negative; when the unconstrained LS intercept is
    negative we refit with a=0 and k=2 free parameters). ``aic`` = n*ln(rss/n) + 2k.
    """

    model: str
    a: float
    b: float
    p: float           # alpha (power_law) or tau_e (exponential)
    rss: float
    n: int
    k: int
    aic: float
    t0: float          # the epoch mapped to t=1 (window origin; alpha depends on this choice)

    def predict(self, epoch: float) -> float:
        t = max(epoch - self.t0 + 1.0, 1e-9)
        if self.model == "power_law":
            return self.a + self.b * t ** (-self.p)
        return self.a + self.b * math.exp(-t / self.p)

    def to_dict(self) -> dict:
        d = {"model": self.model, "a": self.a, "b": self.b, "rss": self.rss,
             "n": self.n, "k": self.k, "aic": self.aic, "t0": self.t0}
        d["alpha" if self.model == "power_law" else "tau_e"] = self.p
        return d


def _ls_ab(ts: list[float], ys: list[float], basis: list[float]) -> tuple[float, float, float]:
    """Closed-form LS for y = a + b*basis; returns (a, b, rss). If the unconstrained intercept is
    negative, refit with a=0 (d_seg >= 0)."""
    n = len(ys)
    sb = sum(basis)
    sy = sum(ys)
    sbb = sum(x * x for x in basis)
    sby = sum(x * y for x, y in zip(basis, ys))
    det = n * sbb - sb * sb
    if abs(det) < 1e-30:
        a, b = sy / n, 0.0
    else:
        b = (n * sby - sb * sy) / det
        a = (sy - b * sb) / n
    if a < 0.0:
        # constrained refit: a=0, b = <basis,y>/<basis,basis>
        a = 0.0
        b = (sby / sbb) if sbb > 0 else 0.0
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(basis, ys))
    return a, b, rss


def _profile_fit(ts: list[float], ys: list[float], model: str,
                 grid: list[float]) -> TailFit:
    """Profile least squares: grid over the nonlinear parameter, closed-form (a,b) per grid point,
    golden-section refine around the best. Deterministic."""

    def basis_of(p: float) -> list[float]:
        if model == "power_law":
            return [t ** (-p) for t in ts]
        return [math.exp(-t / p) for t in ts]

    def rss_of(p: float) -> float:
        return _ls_ab(ts, ys, basis_of(p))[2]

    best_p = min(grid, key=rss_of)
    # golden-section refine between grid neighbors of the best point
    i = grid.index(best_p)
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        if hi - lo < 1e-6 * max(1.0, hi):
            break
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        if rss_of(c) < rss_of(d):
            hi = d
        else:
            lo = c
    p = 0.5 * (lo + hi)
    a, b, rss = _ls_ab(ts, ys, basis_of(p))
    n = len(ys)
    k = 3 if a > 0.0 else 2
    # AIC with the Gaussian-RSS form; guard rss=0 (perfect fit on tiny n)
    aic = n * math.log(max(rss, 1e-30) / n) + 2 * k
    return TailFit(model=model, a=a, b=b, p=p, rss=rss, n=n, k=k, aic=aic, t0=0.0)


def _log_grid(lo: float, hi: float, num: int) -> list[float]:
    step = (math.log(hi) - math.log(lo)) / (num - 1)
    return [math.exp(math.log(lo) + i * step) for i in range(num)]


def fit_power_law_tail(epochs: list[float], values: list[float],
                       *, alpha_lo: float = 0.05, alpha_hi: float = 4.0,
                       grid_points: int = 60) -> TailFit:
    """Fit d(t) = a + b*t^(-alpha), t = epoch - epochs[0] + 1 (window-relative; alpha depends on
    this origin choice — deterministic and recorded via ``t0``)."""
    t0 = float(epochs[0])
    ts = [float(e) - t0 + 1.0 for e in epochs]
    fit = _profile_fit(ts, [float(v) for v in values], "power_law",
                       _log_grid(alpha_lo, alpha_hi, grid_points))
    return TailFit(**{**fit.__dict__, "t0": t0})


def fit_exponential_tail(epochs: list[float], values: list[float],
                         *, grid_points: int = 60) -> TailFit:
    """Fit d(t) = a + b*exp(-t/tau_e) on the same window-relative t axis."""
    t0 = float(epochs[0])
    ts = [float(e) - t0 + 1.0 for e in epochs]
    span = max(ts[-1] - ts[0], 2.0)
    fit = _profile_fit(ts, [float(v) for v in values], "exponential",
                       _log_grid(max(span * 5e-3, 0.5), span * 10.0, grid_points))
    return TailFit(**{**fit.__dict__, "t0": t0})


def _bootstrap_alpha_ci(epochs: list[float], values: list[float], *,
                        n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on alpha (resample points with replacement; seeded/deterministic).
    Requires >= 4 distinct epochs per resample; degenerate resamples are skipped."""
    rng = random.Random(seed)
    n = len(epochs)
    alphas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        es = [epochs[i] for i in idx]
        if len(set(es)) < 4:
            continue
        order = sorted(range(len(es)), key=lambda j: es[j])
        es_s = [es[j] for j in order]
        vs_s = [values[idx[j]] for j in order]
        try:
            alphas.append(fit_power_law_tail(es_s, vs_s, grid_points=30).p)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    if len(alphas) < 20:
        return (float("nan"), float("nan"))
    alphas.sort()
    lo = alphas[max(int(0.025 * len(alphas)) - 1, 0)]
    hi = alphas[min(int(0.975 * len(alphas)), len(alphas) - 1)]
    return (lo, hi)


def fit_tail_models(epochs: list[float], values: list[float], *,
                    n_boot: int = 200, seed: int = 0) -> dict:
    """Fit BOTH models on one series, compare by AIC, bootstrap the alpha CI.

    Returns a plain dict (JSONL-safe): {power_law: {...}, exponential: {...}, preferred_model,
    delta_aic (exp - pow; positive => power law preferred), alpha, alpha_ci}.
    """
    if len(epochs) != len(values):
        raise ValueError("epochs and values length mismatch")
    if len(epochs) < 4:
        raise ValueError(f"need >= 4 points to fit two 3-parameter tail models, got {len(epochs)}")
    pairs = sorted(zip((float(e) for e in epochs), (float(v) for v in values)))
    es = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    pw = fit_power_law_tail(es, vs)
    ex = fit_exponential_tail(es, vs)
    ci = _bootstrap_alpha_ci(es, vs, n_boot=n_boot, seed=seed)
    return {
        "power_law": pw.to_dict(),
        "exponential": ex.to_dict(),
        "preferred_model": "power_law" if pw.aic <= ex.aic else "exponential",
        "delta_aic_exp_minus_pow": ex.aic - pw.aic,
        "alpha": pw.p,
        "alpha_ci95": [ci[0], ci[1]],
        "n_points": len(es),
        "window_epochs": [es[0], es[-1]],
    }


def remaining_meat(fit: TailFit, t_now_epoch: float, horizon_epoch: float) -> float:
    """Extrapolated remaining meat = d(t_now) - d(horizon) under the fitted model (the §4 exit
    quantity: exit on this, not on a window slope). Negative when the fit is rising."""
    return fit.predict(t_now_epoch) - fit.predict(horizon_epoch)


def powerlaw_meat_exit(trajectory, *, horizon_epochs: float = 300.0,
                       meat_floor: float = 1e-4, min_points: int = 8,
                       n_boot: int = 200, seed: int = 0) -> dict:
    """The exit-rule callable the costate/shadow controller (and a future
    ``ExitEvent(criterion="powerlaw_meat")``) consumes.

    Input: a per-class trajectory ``{class_name: [(epoch, d_seg), ...], ...}`` — a bare list of
    (epoch, value) pairs is accepted and treated as ``{"total": ...}`` (the honest fallback while
    per-class d_seg telemetry is an apparatus gap; see the solver-pack memo duty-to-measure).

    Output (top-level keys per the schedule-criterion contract):
      ``exhausted`` (bool) — True only when EVERY fittable class's extrapolated remaining meat to
      ``t_now + horizon_epochs`` is below ``meat_floor`` under its AIC-preferred model. Fail-safe
      direction: insufficient/unfittable data => NOT exhausted (never declare exhaustion on a bad
      measurement — the confound-L3 verdict-clearance discipline).
      ``remaining_meat_estimate`` (float) — the MAX remaining meat across classes (the binding
      class is the one still paying).
      ``alpha`` (float) — the binding class's power-law exponent.
      ``ci`` ([lo, hi]) — the binding class's bootstrap alpha CI.
      plus ``binding_class``, ``per_class`` (full fit detail), ``reason``.
    """
    if not isinstance(trajectory, dict):
        trajectory = {"total": list(trajectory)}
    per_class: dict[str, dict] = {}
    meats: dict[str, float] = {}
    any_unfittable = False
    for cls, series in trajectory.items():
        series = sorted((float(e), float(v)) for e, v in series)
        if len(series) < min_points:
            per_class[cls] = {"fit": None,
                              "reason": f"insufficient points ({len(series)} < {min_points})"}
            any_unfittable = True
            continue
        es = [p[0] for p in series]
        vs = [p[1] for p in series]
        fits = fit_tail_models(es, vs, n_boot=n_boot, seed=seed)
        pref = fits["preferred_model"]
        fd = fits[pref]
        fit = TailFit(model=fd["model"], a=fd["a"], b=fd["b"],
                      p=fd.get("alpha", fd.get("tau_e")), rss=fd["rss"], n=fd["n"],
                      k=fd["k"], aic=fd["aic"], t0=fd["t0"])
        t_now = es[-1]
        meat = remaining_meat(fit, t_now, t_now + horizon_epochs)
        meats[cls] = meat
        per_class[cls] = {"fit": fits, "remaining_meat_to_horizon": meat,
                          "t_now": t_now, "horizon": t_now + horizon_epochs,
                          "reason": "fitted"}
    if not meats:
        return {"exhausted": False, "remaining_meat_estimate": float("nan"),
                "alpha": float("nan"), "ci": [float("nan"), float("nan")],
                "binding_class": None, "per_class": per_class,
                "reason": "no fittable class (insufficient telemetry) — fail-safe: NOT exhausted"}
    binding = max(meats, key=lambda c: meats[c])
    bfit = per_class[binding]["fit"]
    exhausted = (not any_unfittable) and all(m < meat_floor for m in meats.values())
    return {
        "exhausted": exhausted,
        "remaining_meat_estimate": meats[binding],
        "alpha": bfit["alpha"],
        "ci": bfit["alpha_ci95"],
        "binding_class": binding,
        "per_class": per_class,
        "reason": ("all fittable classes below meat_floor"
                   if exhausted else
                   f"binding class {binding!r} remaining meat {meats[binding]:.3g} "
                   f">= floor {meat_floor:.3g}" if meats[binding] >= meat_floor else
                   "some classes unfittable (fail-safe: NOT exhausted)"),
    }
