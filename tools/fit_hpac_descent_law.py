#!/usr/bin/env python3
"""Fit descent laws to HPAC trainer eval trajectories and derive stopping epochs.

The equations-leg instrument for the rx2 GPU-race extension decision
(wc2 memo §5, operator 2026-08-14 "more epochs ... once we confirm working and
understand the trajectory better", recalling the EventGated / canonical-
equations discipline: derive LAWS from observed behavior, never latch epoch
constants; stage skeletons -> event/law-driven continuation).

Reads the trainer's JSON-line eval rows (epoch/phase/bpp/estimated_joint_bytes)
from run logs, fits two candidate law forms per phase:

  exp_floor : y(t) = y_inf + A * exp(-(t - t0) / tau)
  power     : y(t) = y_inf + B * (t - t0 + 1) ** (-alpha)

selects by SSE (equal parameter count -> SSE ordering == AIC ordering), and
derives N* = the epoch where the fitted remaining gain to the asymptote drops
below byte bars expressed in canonical score bands (1 band = 3.5e-6 S on the
T4 axis = 3.5e-6 * 37_545_489 / 25 = 5.256 B on the rate term).

Honesty labels (cross-regime constant-transfer law, memory m21/m22):
  - The CONTINUOUS-phase law projects directly to longer runs (same regime,
    same entry state: the archive-derived init).
  - The QAT-phase law is ENTRY-STATE-CONDITIONAL: a longer run enters QAT from
    a deeper continuous endpoint, so the fitted QAT law is reported as a
    measured delta with that caveat, never silently transferred.

Output: printed table + a JSON receipt (the fit consumes retained logs; the
receipt records source paths + shas so the fit is re-derivable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# 1 canonical band (±3.5e-6 S, T4 n600) expressed as archive bytes on the rate
# term: S_rate = 25 * B / 37_545_489  =>  B per S = 37_545_489 / 25.
RATE_DENOMINATOR = 37_545_489
BAND_S = 3.5e-6
BAND_BYTES = BAND_S * RATE_DENOMINATOR / 25.0  # ~5.256 B


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in obj and "estimated_joint_bytes" in obj and "phase" in obj:
                rows.append(obj)
    # Deduplicate on (epoch, phase) keeping the LAST occurrence (resume rewrites).
    seen: dict = {}
    for obj in rows:
        seen[(obj["epoch"], obj["phase"])] = obj
    return sorted(seen.values(), key=lambda o: o["epoch"])


def _exp_floor(t, y_inf, amp, tau):
    return y_inf + amp * np.exp(-t / tau)


def _power(t, y_inf, amp, alpha):
    return y_inf + amp * np.power(t + 1.0, -alpha)


def fit_phase(epochs: np.ndarray, values: np.ndarray) -> dict:
    """Fit both law forms to one phase's (epoch, value) points; pick by SSE."""
    t = epochs - epochs.min()
    y = values.astype(float)
    span = max(y.max() - y.min(), 1e-9)
    fits = {}
    candidates = {
        "exp_floor": (
            _exp_floor,
            [y.min() - 0.1 * span, span, max(t.max(), 1.0) / 2.0],
            ([-np.inf, 0.0, 1e-3], [y.min(), np.inf, np.inf]),
        ),
        "power": (
            _power,
            [y.min() - 0.1 * span, span, 0.5],
            ([-np.inf, 0.0, 1e-3], [y.min(), np.inf, np.inf]),
        ),
    }
    for name, (fn, p0, bounds) in candidates.items():
        try:
            popt, _ = curve_fit(fn, t, y, p0=p0, bounds=bounds, maxfev=20000)
            resid = y - fn(t, *popt)
            fits[name] = {
                "params": [float(v) for v in popt],
                "sse": float(np.sum(resid**2)),
                "rms": float(np.sqrt(np.mean(resid**2))),
            }
        except Exception as exc:  # a failed fit is a report, not a crash
            fits[name] = {"error": str(exc)}
    ok = {k: v for k, v in fits.items() if "sse" in v}
    best = min(ok, key=lambda k: ok[k]["sse"]) if ok else None
    return {"fits": fits, "best": best, "t0": float(epochs.min()), "n_points": int(len(y))}


def remaining_gain(fit: dict, t: float) -> float:
    """Fitted remaining descent (bytes) from relative epoch t to the asymptote."""
    name = fit["best"]
    p = fit["fits"][name]["params"]
    if name == "exp_floor":
        return p[1] * math.exp(-t / p[2])
    return p[1] * (t + 1.0) ** (-p[2])


def derive_n_star(fit: dict, bars_bytes: list[float], t_max: float = 100000.0) -> dict:
    """Smallest relative epoch where remaining gain < bar, per bar."""
    out = {}
    for bar in bars_bytes:
        name = fit["best"]
        p = fit["fits"][name]["params"]
        if name == "exp_floor":
            t_star = -p[2] * math.log(bar / p[1]) if p[1] > bar else 0.0
        else:
            t_star = (p[1] / bar) ** (1.0 / p[2]) - 1.0 if p[1] > bar else 0.0
        out[f"bar_{bar:.3f}B"] = float(min(max(t_star, 0.0), t_max))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", action="append", required=True, type=Path,
                    help="trainer run.log / JSON-lines path (repeatable)")
    ap.add_argument("--metric", default="estimated_joint_bytes",
                    choices=["estimated_joint_bytes", "bpp", "top1_error"])
    ap.add_argument("--out", type=Path, required=True, help="JSON receipt path")
    ap.add_argument("--project-epochs", default="60,120,240,480,960",
                    help="comma list of total-epoch budgets to project (qat_fraction 0.5 assumed)")
    args = ap.parse_args()

    rows = load_rows(args.log)
    if not rows:
        print("no eval rows found", file=sys.stderr)
        return 2

    receipt: dict = {
        "schema": "hpac_descent_law_fit.v1",
        "metric": args.metric,
        "band_bytes": BAND_BYTES,
        "sources": [{"path": str(p), "sha256": _sha256(p)} for p in args.log],
        "phases": {},
        "authority": "[research-signal fit over retained trainer eval rows; "
                     "no score claim; projections are projections, not measurements]",
    }

    by_phase: dict[str, list[dict]] = {}
    for obj in rows:
        by_phase.setdefault(obj["phase"], []).append(obj)

    for phase, objs in by_phase.items():
        epochs = np.array([o["epoch"] for o in objs], dtype=float)
        values = np.array([o[args.metric] for o in objs], dtype=float)
        fit = fit_phase(epochs, values)
        entry = {"fit": fit, "first_epoch": float(epochs.min()),
                 "last_epoch": float(epochs.max()),
                 "first_value": float(values[0]), "last_value": float(values[-1])}
        if fit["best"]:
            entry["n_star_relative"] = derive_n_star(
                fit, bars_bytes=[BAND_BYTES, 10 * BAND_BYTES])
            entry["asymptote"] = fit["fits"][fit["best"]]["params"][0]
        if phase != "continuous":
            entry["transfer_label"] = (
                "ENTRY_STATE_CONDITIONAL: fitted at this run's continuous "
                "endpoint; does NOT transfer to a deeper entry state "
                "(cross-regime constant-transfer law)")
        receipt["phases"][phase] = entry

    cont = receipt["phases"].get("continuous")
    if cont and cont["fit"]["best"]:
        proj = {}
        for n_total in [int(x) for x in args.project_epochs.split(",")]:
            n_cont = n_total // 2  # qat_fraction 0.5
            t_rel = n_cont - cont["first_epoch"]
            proj[str(n_total)] = {
                "continuous_epochs": n_cont,
                "projected_continuous_value": float(
                    cont["asymptote"] + remaining_gain(cont["fit"], t_rel)),
                "remaining_gain_bytes_at_phase_exit": float(
                    remaining_gain(cont["fit"], t_rel)),
            }
        receipt["projection_table_total_epochs"] = proj
        receipt["projection_note"] = (
            "continuous-phase law only; QAT-phase gain from the measured run is "
            "additive-CONDITIONAL (see transfer_label). Rate ΔS per byte = 25/37,545,489.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
