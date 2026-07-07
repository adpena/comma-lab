#!/usr/bin/env python
"""$0 retro-fit of the power-law plateau/exit detector on logged witness trajectories (§4 of
viscosity_theory_alignment_hunt_20260707.md, weak-KAM Lax–Oleinik rates).

Fits d_seg(t) = a + b*t^(-alpha) vs d_seg(t) = a + b*exp(-t/tau) per stage window on:
  (i)  the long900 trajectory (witcap_kd_c1_long900 — single-stage KD run, 19 eval points), and
  (ii) the live mod32cap run's verdict telemetry (CE 0-299 / tau 300-725 / Muon 726+ windows),

compares by AIC, bootstraps the alpha CI, and evaluates the extrapolated-remaining-meat exit rule
(:func:`tac.witness_control.powerlaw_exit.powerlaw_meat_exit`).

Pre-registered check (hunt §4): alpha_lane < alpha_road. HONEST STATUS: per-class d_seg
trajectories are NOT logged by either run (verdict rows carry total d_seg only; the annulus
sidecar has 2 per-class snapshots) — the check is reported UNRESOLVED with the apparatus gap
named. The library + producer signal are per-class-capable the moment a run logs per-class rows.

All numbers [macOS-CPU advisory] — research signal, never a score; pointer 0.19110 moves only via
upstream/evaluate.py. Read-only on the live run dir. Peak RSS trivially small (parses two logs).

Usage:
  .venv/bin/python tools/fit_powerlaw_plateau_detector.py \
      --out-dir experiments/results/solver_pack_20260707/powerlaw_detector
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.witness_control.powerlaw_exit import fit_tail_models, powerlaw_meat_exit  # noqa: E402

LONG900_LOG = REPO / "experiments/results/witness_capstone_deepmath_20260625/kd_c1_long900.log"
MOD32CAP_LOG = Path("/Users/adpena/Projects/pact/.omx/tmp/levelset_mod32cap_20260706T115614Z.log")
MOD32CAP_RUN = REPO / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"

ADVISORY = "[macOS-CPU advisory] NON-PROMOTABLE research-signal; pointer 0.19110 UNMOVED"


def _parse_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_long900_trajectory(path: Path) -> list[tuple[float, float]]:
    """(epoch, d_seg) from the long900 durable log (plain JSON lines with epoch/d_seg keys)."""
    out = []
    for r in _parse_jsonl_rows(path):
        if "epoch" in r and "d_seg" in r and "stage" not in r:
            out.append((float(r["epoch"]), float(r["d_seg"])))
    return sorted(out)


def load_mod32cap_windows(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Stage-windowed (epoch, d_seg) series from the live run's verdict telemetry.

    Windows per the run's curriculum: CE 0-299, tau_softplus 300-725, Muon finisher 726+ (the
    muon_finisher_switch row fires at ep726; seg_form stays tau_softplus — the switch is the
    OPTIMIZER, so the Muon window is cut on epoch, not seg_form). Epoch-0 init verdict excluded.
    """
    ce, tau, muon = [], [], []
    for r in _parse_jsonl_rows(path):
        if r.get("stage") != "verdict":
            continue
        ep = float(r["epoch"])
        d = float(r["d_seg"])
        if ep <= 0:
            continue  # epoch-0 init verdict (random init) is not part of any descent window
        if ep <= 299:
            ce.append((ep, d))
        elif ep <= 725:
            tau.append((ep, d))
        else:
            muon.append((ep, d))
    return {"CE_ep1_299": sorted(ce), "tau_softplus_ep300_725": sorted(tau),
            "muon_finisher_ep726_plus": sorted(muon)}


def analyze_window(name: str, series: list[tuple[float, float]], *,
                   horizon: float, meat_floor: float, seed: int = 0) -> dict:
    out: dict = {"window": name, "n_points": len(series),
                 "epochs": [series[0][0], series[-1][0]] if series else None}
    if len(series) < 4:
        out["status"] = f"SKIPPED (insufficient points: {len(series)} < 4)"
        return out
    es = [p[0] for p in series]
    vs = [p[1] for p in series]
    fits = fit_tail_models(es, vs, seed=seed)
    out["fits"] = fits
    out["exit_rule"] = powerlaw_meat_exit({name: series}, horizon_epochs=horizon,
                                          meat_floor=meat_floor, min_points=4, seed=seed)
    # drop the nested per_class duplicate of fits to keep the JSON small
    out["exit_rule"]["per_class"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "fit"}
        for k, v in out["exit_rule"]["per_class"].items()}
    out["status"] = "FITTED"
    if len(series) < 8:
        out["status"] = "FITTED_LOW_CONFIDENCE (n < 8; 3-param fits on few points)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long900-log", type=Path, default=LONG900_LOG)
    ap.add_argument("--mod32cap-log", type=Path, default=MOD32CAP_LOG)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/results/solver_pack_20260707/powerlaw_detector")
    ap.add_argument("--horizon-epochs", type=float, default=300.0,
                    help="extrapolated-remaining-meat horizon (epochs past t_now)")
    ap.add_argument("--meat-floor", type=float, default=1e-4,
                    help="exit when extrapolated remaining d_seg meat to horizon < floor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results: dict = {
        "tool": "tools/fit_powerlaw_plateau_detector.py",
        "axis": ADVISORY,
        "theory": ("weak KAM / Lax-Oleinik: exponential descent off the Aubry-set analog, "
                   "power-law O(1/t)-class ON it (the binding lane class) — "
                   "viscosity_theory_alignment_hunt_20260707.md §4"),
        "exit_rule": {"horizon_epochs": args.horizon_epochs, "meat_floor": args.meat_floor,
                      "seed": args.seed,
                      "callable": "tac.witness_control.powerlaw_exit:powerlaw_meat_exit"},
        "windows": [],
    }

    # (i) long900 — single-stage KD run
    if args.long900_log.exists():
        traj = load_long900_trajectory(args.long900_log)
        results["windows"].append(
            analyze_window("long900_kd_c1_ep1_900", traj,
                           horizon=args.horizon_epochs, meat_floor=args.meat_floor,
                           seed=args.seed))
    else:
        results["windows"].append({"window": "long900_kd_c1_ep1_900",
                                   "status": f"MISSING LOG {args.long900_log}"})

    # (ii) live mod32cap — per stage window
    if args.mod32cap_log.exists():
        for name, series in load_mod32cap_windows(args.mod32cap_log).items():
            results["windows"].append(
                analyze_window(f"mod32cap_{name}", series,
                               horizon=args.horizon_epochs, meat_floor=args.meat_floor,
                               seed=args.seed))
    else:
        results["windows"].append({"window": "mod32cap", "status":
                                   f"MISSING LOG {args.mod32cap_log}"})

    # the pre-registered per-class check — honest apparatus-gap report
    results["preregistered_check_alpha_lane_lt_alpha_road"] = {
        "status": "UNRESOLVED_APPARATUS_GAP",
        "reason": ("per-class d_seg trajectories are NOT logged by either run: verdict telemetry "
                   "carries total d_seg only; the annulus sidecar holds 2 per-class snapshots "
                   "(ep299/ep300) — insufficient for a 3-parameter tail fit. "
                   "powerlaw_meat_exit() and the producer signal accept per-class trajectories "
                   "the moment a run logs them (duty-to-measure: per-class d_seg verdict rows)."),
        "trainer_support_gap": ("per-class d_seg in the verdict row (5 floats/verdict; "
                                "score-neutral read-only telemetry => defaults ON per the "
                                "default-off-is-orphaned-signal rule)"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "powerlaw_fits.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=1)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")
    for w in results["windows"]:
        if "fits" not in w:
            print(f"  {w['window']}: {w['status']}")
            continue
        f = w["fits"]
        er = w["exit_rule"]
        print(f"  {w['window']}: n={f['n_points']} alpha={f['alpha']:.3f} "
              f"ci95=[{f['alpha_ci95'][0]:.3f},{f['alpha_ci95'][1]:.3f}] "
              f"preferred={f['preferred_model']} dAIC(exp-pow)={f['delta_aic_exp_minus_pow']:.2f} "
              f"meat_to_+{args.horizon_epochs:.0f}ep={er['remaining_meat_estimate']:.3g} "
              f"exhausted={er['exhausted']} [{w['status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
