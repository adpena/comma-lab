#!/usr/bin/env python3
"""Fit E-AXIS convergence models to the mined d_seg(epochs) dataset.

Reads the JSON produced by ``tools/build_convergence_dataset.py`` and fits, per
run with enough points, two candidate convergence laws to ``d_seg(E)``:

  1. exponential-to-floor:  d_seg(E) = floor + (a0 - floor) * exp(-E / tau)
  2. power-law-to-floor:     d_seg(E) = floor + c * E**(-p)

It extracts, per run: the fitted asymptote (floor), the convergence time-constant
(tau or power exponent p), epochs-to-reach-target-d_seg (interpolated AND
model-extrapolated, clearly separated), and a per-config "epochs to basin" metric.
It then groups runs by config to surface WHICH levers speed convergence
(optimizer/stage, n_pairs, base_channels, taper, lever set).

Authority: [advisory] NON-PROMOTABLE. All extrapolations beyond the measured
epoch span are explicitly tagged ``extrapolated: true``. Per the existence-proof
cross-check, NO run's final d_seg is treated as a floor unless the fit's asymptote
is itself reported as the model floor (a model artifact, also labelled). The PR95
existence proof (d_seg 5.6e-4 @ 29,650 ep) is carried through as the cross-check
anchor: any fitted asymptote ABOVE 5.6e-4 is NOT a physical floor.

Usage:
    .venv/bin/python tools/fit_convergence_models.py \
        --dataset .omx/research/eaxis_convergence_dataset_20260623.json \
        --out .omx/research/eaxis_convergence_models_20260623.json --summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

PR95_DSEG_FLOOR = 5.6e-4  # existence proof; nothing above this is a physical floor
DEFAULT_TARGETS = [0.05, 0.01, 0.005, 0.003, 0.0025, 0.002]
MIN_POINTS_FOR_FIT = 6


def _exp_to_floor(E, floor, amp, tau):
    return floor + amp * np.exp(-E / tau)


def _pow_to_floor(E, floor, c, p):
    return floor + c * np.power(np.maximum(E, 1.0), -p)


def _r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _fit_exp(E, y):
    if not _HAVE_SCIPY:
        return None
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)
    a0 = float(y[0])
    floor0 = float(min(y.min(), a0 * 0.5))
    span = max(E.max() - E.min(), 1.0)
    try:
        popt, _ = curve_fit(
            _exp_to_floor, E, y,
            p0=[max(floor0, 1e-6), max(a0 - floor0, 1e-6), span / 3.0],
            bounds=([0.0, 0.0, 1.0], [max(y.max(), 1.0), 10.0, span * 50]),
            maxfev=20000,
        )
        yhat = _exp_to_floor(E, *popt)
        return {"model": "exp_to_floor", "floor": float(popt[0]),
                "amp": float(popt[1]), "tau": float(popt[2]),
                "r2": _r2(y, yhat)}
    except Exception:
        return None


def _fit_pow(E, y):
    if not _HAVE_SCIPY:
        return None
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)
    a0 = float(y[0])
    floor0 = float(min(y.min() * 0.5, a0 * 0.1))
    try:
        popt, _ = curve_fit(
            _pow_to_floor, E, y,
            p0=[max(floor0, 1e-6), max(a0, 1e-3), 0.5],
            bounds=([0.0, 0.0, 0.01], [max(y.max(), 1.0), 1e6, 5.0]),
            maxfev=20000,
        )
        yhat = _pow_to_floor(E, *popt)
        return {"model": "pow_to_floor", "floor": float(popt[0]),
                "c": float(popt[1]), "p": float(popt[2]),
                "r2": _r2(y, yhat)}
    except Exception:
        return None


def _epochs_to_target_interp(E, y, target):
    """First measured epoch at/below target (interpolated between brackets)."""
    E = list(E)
    y = list(y)
    for i in range(len(y)):
        if y[i] <= target:
            if i == 0:
                return {"epochs": float(E[0]), "extrapolated": False,
                        "method": "first_point_already_below"}
            # linear interp between i-1 and i
            e0, e1 = E[i - 1], E[i]
            y0, y1 = y[i - 1], y[i]
            if y0 == y1:
                return {"epochs": float(e1), "extrapolated": False, "method": "interp"}
            frac = (y0 - target) / (y0 - y1)
            return {"epochs": float(e0 + frac * (e1 - e0)),
                    "extrapolated": False, "method": "interp"}
    return None  # never reached in measured range


def _epochs_to_target_model(fit, target):
    """Solve the fitted model for the epoch reaching target (extrapolation-aware)."""
    if fit is None:
        return None
    floor = fit["floor"]
    if target <= floor:
        return {"epochs": None, "extrapolated": True,
                "method": "unreachable_target_below_model_floor",
                "model_floor": floor}
    if fit["model"] == "exp_to_floor":
        amp, tau = fit["amp"], fit["tau"]
        # target = floor + amp*exp(-E/tau) -> E = -tau*ln((target-floor)/amp)
        arg = (target - floor) / amp
        if arg <= 0:
            return None
        E = -tau * math.log(arg)
        return {"epochs": float(max(E, 0.0)), "extrapolated": True, "method": "exp_solve"}
    else:
        c, p = fit["c"], fit["p"]
        # target = floor + c*E^-p -> E = (c/(target-floor))^(1/p)
        base = c / (target - floor)
        if base <= 0:
            return None
        E = base ** (1.0 / p)
        return {"epochs": float(E), "extrapolated": True, "method": "pow_solve"}


def _stage_summary(points):
    """Return ordered list of (stage_name, first_epoch, first_dseg) transitions."""
    seen = []
    last = None
    for p in points:
        sn = p.get("stage_name")
        if sn != last:
            seen.append({"stage": sn, "epoch": p.get("global_epoch"),
                         "d_seg": p.get("d_seg")})
            last = sn
    return seen


def _config_key(run):
    cfg = run["config"]
    bc = cfg.get("base_channels", "?")
    npairs = cfg.get("n_pairs", "?")
    levers = cfg.get("levers")
    lever_tag = "noflag"
    if isinstance(levers, dict):
        active = [k.replace("lever", "L").split("_")[0] for k, v in levers.items()
                  if v and not str(v).lower() == "false"]
        lever_tag = "+".join(sorted(active)) if active else "none"
    return f"bc{bc}_n{npairs}_{lever_tag}"


def fit_run(run):
    points = [p for p in run["points"] if p.get("d_seg") is not None]
    if len(points) < MIN_POINTS_FOR_FIT:
        # still record basic interp targets for short runs
        short = True
    else:
        short = False
    E = [p["global_epoch"] for p in points]
    y = [p["d_seg"] for p in points]
    # dedupe / monotone-x guard: collapse same epoch keeping min d_seg
    by_ep = {}
    for e, v in zip(E, y):
        by_ep[e] = min(v, by_ep.get(e, v))
    E = sorted(by_ep)
    y = [by_ep[e] for e in E]

    exp_fit = _fit_exp(E, y) if not short else None
    pow_fit = _fit_pow(E, y) if not short else None
    best_fit = None
    if exp_fit and pow_fit:
        best_fit = exp_fit if exp_fit["r2"] >= pow_fit["r2"] else pow_fit
    else:
        best_fit = exp_fit or pow_fit

    targets = {}
    for t in DEFAULT_TARGETS:
        interp = _epochs_to_target_interp(E, y, t)
        model = _epochs_to_target_model(best_fit, t)
        targets[str(t)] = {"measured": interp, "model": model}

    # convergence rate metric: epochs for d_seg to halve from first->measured min
    half = None
    if y:
        y0 = y[0]
        thalf = _epochs_to_target_interp(E, y, y0 / 2.0)
        if thalf:
            half = thalf["epochs"] - E[0]

    return {
        "run_id": run["run_id"],
        "config_key": _config_key(run),
        "config": run["config"],
        "schema": run["schema"],
        "n_points": len(points),
        "epoch_span": [E[0], E[-1]] if E else None,
        "final_d_seg": y[-1] if y else None,
        "min_d_seg": min(y) if y else None,
        "wall_clock_s": run.get("wall_clock_s"),
        "exp_fit": exp_fit,
        "pow_fit": pow_fit,
        "best_fit": best_fit,
        "best_fit_floor_is_physical": (
            best_fit is not None and best_fit["floor"] <= PR95_DSEG_FLOOR
        ),
        "epochs_to_halve_dseg": half,
        "epochs_to_target": targets,
        "stage_transitions": _stage_summary(points),
    }


def build_models(dataset):
    fits = [fit_run(r) for r in dataset["runs"]]
    # group by config to surface lever effects
    groups = {}
    for f in fits:
        groups.setdefault(f["config_key"], []).append({
            "run_id": f["run_id"],
            "min_d_seg": f["min_d_seg"],
            "epoch_span": f["epoch_span"],
            "epochs_to_halve_dseg": f["epochs_to_halve_dseg"],
            "n_points": f["n_points"],
        })
    return {
        "schema_version": 1,
        "authority": "[advisory] NON-PROMOTABLE; convergence fits over MINED curves. "
                     "All 'model' epoch-to-target are EXTRAPOLATIONS (extrapolated:true). "
                     "No fitted floor above PR95 5.6e-4 is a physical floor.",
        "existence_proof_dseg_floor": PR95_DSEG_FLOOR,
        "generated_by": "tools/fit_convergence_models.py",
        "n_runs_fit": len(fits),
        "config_groups": groups,
        "fits": fits,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=".omx/research/eaxis_convergence_dataset_20260623.json")
    ap.add_argument("--out", default=".omx/research/eaxis_convergence_models_20260623.json")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    ds_path = (REPO_ROOT / args.dataset).resolve()
    if not ds_path.exists():
        print(f"dataset not found: {ds_path}", file=sys.stderr)
        return 2
    dataset = json.loads(ds_path.read_text())
    models = build_models(dataset)
    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(models, indent=2))
    print(f"wrote {out_path} : {models['n_runs_fit']} fits, "
          f"{len(models['config_groups'])} config groups")

    if args.summary:
        print("\n=== best long fits (>=20 pts), sorted by min_d_seg ===")
        longfits = [f for f in models["fits"] if f["n_points"] >= 20]
        longfits.sort(key=lambda f: (f["min_d_seg"] if f["min_d_seg"] is not None else 9))
        for f in longfits[:15]:
            bf = f["best_fit"]
            tag = ""
            if bf:
                if bf["model"] == "exp_to_floor":
                    tag = f"exp floor={bf['floor']:.2e} tau={bf['tau']:.0f} r2={bf['r2']:.3f}"
                else:
                    tag = f"pow floor={bf['floor']:.2e} p={bf['p']:.2f} r2={bf['r2']:.3f}"
            print(f"  min_dseg={f['min_d_seg']:.4e}  ep{f['epoch_span']}  {tag}")
            print(f"      {f['run_id']}  [{f['config_key']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
