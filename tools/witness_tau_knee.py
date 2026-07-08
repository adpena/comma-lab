#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""P-TAU2 instrument: flip-mass-vs-margin-threshold knee -> implied f_target + fixed-point tau*.

LAW + BAND PROVENANCE (T5 crucible, DRAFT_OPTIMAL_STACK_v6 SS1.4a + SS7c): v6's tau*
convention is the FIXED POINT of mass(m < tau*.ln5) = f_target (Maslov error budget),
with f_target deferred to the named probe P-TAU2. Per SS7c, P-TAU2 is a REPORTING probe
(derives a constant): the launch fail-safe constant ``--softmax-temp-end 0.31`` stands
regardless; the tau_end LIVE law promotes only after f_target lands. The LIVE f_target is
a run-1 conversion-rate measurement between two live tau-samples (SS1.4a: "NOT derivable
from a static snapshot"); THIS tool measures the $0 STATIC prior: the KNEE of the
flip-mass CDF mu(m) on the TRUE GT-cache margin axis -- the point where the marginal
conversion rate d(mu)/dm collapses to the sweep-average rate.

KNEE CRITERION (pre-registered in probe_tau2_dither_20260708.md BEFORE computation):
PRIMARY = Kneedle max-chord-deviation (Satopaa et al. 2011) at endpoint m_hi = q99 of the
flip mass: knee = argmax_m [mu_norm - m_norm], whose argmax condition is
mu'(m_knee) = mu(m_hi)/m_hi (marginal return = sweep-average return -- derived, not
eyeballed). SECONDARY robustness = max curvature of the smoothed normalized CDF;
endpoint robustness {q95, q99, max}. Metric math: tac.witness_annulus_metrics
(flip_margin_values / flip_margin_cdf / flip_mass_knee_analysis; pure numpy, tested).

MARGIN-FIELD TRAP (inherited from tools/witness_tau_mq_confirm.py): the maps-npz
``gt_margin`` key is the WITNESS margin-toward-GT (<=0 at flips, tautological). The
margin axis here ALWAYS comes from a GT cache's ``margins`` member (>=0,
witness-independent).

AUTHORITY (NO-FAKE): consumes the SAME rendered witness-argmax maps the tau-CONFIRM
instrument produced through the REAL render-through-R + frozen CPU-torch SegNet path.
Every row is [macOS-numpy advisory . NON-PROMOTABLE]; advisory subset; NO score claims.

Usage:
  .venv/bin/python tools/witness_tau_knee.py \\
      --maps BEST_ep650=.omx/research/t5_crucible/artifacts/tau_mq_maps/maps_BEST_ep650.npz \\
      --maps END_ep1000=.omx/research/t5_crucible/artifacts/tau_mq_maps/maps_END_ep1000.npz \\
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
      --launch-tau 0.31 --out-json OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_annulus_metrics import (  # noqa: E402
    ADVISORY,
    flip_margin_values,
    flip_mass_knee_analysis,
)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cache_subset(cache_path: Path, n_pairs_avail: int, num_pairs: int):
    """(vpairs, lstars, margins) strided subset from a GT cache (same recipe as tau_mq_confirm)."""
    z = np.load(cache_path, mmap_mode="r")
    p_total = int(z["lstars"].shape[0])
    p_use = min(num_pairs, p_total)
    stride = max(1, p_use // max(n_pairs_avail, 1))
    vp = list(range(0, p_use, stride))[:n_pairs_avail]
    ls = np.stack([np.asarray(z["lstars"][i]) for i in vp]).astype(np.int64)
    mg = np.stack([np.asarray(z["margins"][i], np.float32) for i in vp])
    return vp, ls, mg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", action="append", required=True,
                    help="NAME=maps.npz (witness argmax maps; repeat). Margin axis NEVER read from these.")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--grid-n", type=int, default=2001)
    ap.add_argument("--endpoints", default="95,99,100",
                    help="flip-mass quantile endpoints (pct) for m_hi. default 95,99,100")
    ap.add_argument("--launch-tau", type=float, default=0.31,
                    help="the v6 launch constant to test against the knee band. default 0.31")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    endpoints = tuple(float(x) / 100.0 for x in args.endpoints.split(","))
    t0 = time.time()
    results = []
    for spec in args.maps:
        name, p = spec.split("=", 1)
        z = np.load(Path(p), allow_pickle=False)
        wa = np.asarray(z["argmax"])
        vp, ls, mg = _cache_subset(args.gt_cache, int(wa.shape[0]), args.num_pairs)
        if ls.shape != wa.shape:
            raise SystemExit(f"[{name}] gt subset {ls.shape} != maps argmax {wa.shape} (stride mismatch?)")
        fm = flip_margin_values(wa, ls, mg)
        tbl = flip_mass_knee_analysis(fm, endpoints=endpoints, grid_n=args.grid_n)
        if tbl.get("empty"):
            print(f"[{_utc()}] [{name}] ZERO flips — no knee (skipped)", flush=True)
            continue
        primary = next((r for r in tbl["rows"] if r["primary"]), None)
        results.append({
            "name": name, "source": p,
            "epoch": int(z["epoch"]) if "epoch" in z.files else -1,
            "softmax_temp": float(z["softmax_temp"]) if "softmax_temp" in z.files else float("nan"),
            "pairs": int(wa.shape[0]), "vpairs": vp,
            "knee": tbl, "primary_row": primary,
        })
        if primary is not None:
            print(f"[{_utc()}] [{name}] n_flips={tbl['n_flips']} PRIMARY kneedle@q99: "
                  f"m_knee={primary['m_knee']:.6f} f_target={primary['f_target']:.6f} "
                  f"tau*={primary['tau_star']:.6f} band={tbl['tau_star_band']}", flush=True)
        else:
            print(f"[{_utc()}] [{name}] n_flips={tbl['n_flips']} (no q99 endpoint -> no PRIMARY row) "
                  f"band={tbl['tau_star_band']}", flush=True)

    all_taus = [r["knee"]["tau_star_band"] for r in results]
    if not all_taus:
        raise SystemExit("no legs produced a knee (all-empty flip populations)")
    band = [min(b[0] for b in all_taus), max(b[1] for b in all_taus)]
    stands = bool(band[0] <= args.launch_tau <= band[1])
    summary = {
        "advisory": ADVISORY,
        "generated": _utc(),
        "probe": "P-TAU2 (static knee prior for f_target; v6 SS7c reporting probe)",
        "criterion": "PRIMARY kneedle@q99 (marginal return = sweep-average return); "
                     "SECONDARY max-curvature; endpoints " + args.endpoints,
        "launch_tau": args.launch_tau,
        "knee_band_tau_star": band,
        "launch_tau_within_band": stands,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    out = json.dumps(summary, indent=2, default=float)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(out, encoding="utf-8")
        print(f"[{_utc()}] wrote {args.out_json}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
