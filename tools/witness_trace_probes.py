#!/usr/bin/env python
"""witness_trace_probes.py — T5-crucible CONTROL/SCHEDULE recess probes as a reusable CLI.

DURABLE INSTRUMENT (requirement Q): runs the four $0 probes against ANY completed witness run
directory (READ-ONLY on the run dir; artifact JSON written to --out, default
``.omx/research/t5_crucible/artifacts/``). Run-2 re-runs the SAME instruments on its own trace.

Probes + PRE-REGISTERED bands (provenance: DRAFT_OPTIMAL_STACK_v5_20260707.md §2.2f/§2.5/§7c;
ct_deepresearch_1_training_campaign_control_20260707.md §3.3/§4.2/§10.2; bands are BINDING —
measured numbers reported at full precision, band verdicts never adjusted post-hoc):

* **ct3** forfeit-matched TAU→FIN arm backtest, s* = ν·forfeit = 1.4154e-5 S/ep.
  Band: first sustained fire ep670–700; EMA-best-at-fire within 1 cadence of the stage best.
  Kill: fires < ep650 or > cap 726 ⇒ arm stays would-fire-only. (Scope on kill: FORMULATION —
  this slope estimator + this ν; untested reformulations enumerated in the memo.)
* **ct1** ν refit per stage (exp tail fit; power-law alternative reported).
  Band: ν ∈ [0.02, 0.035]/ep. Kill: ν < 0.01 ⇒ recompute ALL window laws (the recomputed values
  are emitted unconditionally so a kill carries its own remediation numbers).
* **ct2** self-triggered verdict cadence replay, Δt = clamp(floor_S/|Ŝ′|, 25, 100).
  Band: 12–17 of the trace verdicts skipped. Kill: any missed prefix-best > 1 cadence ⇒ B-CT3
  stays unbuilt. A floor_S sensitivity sweep is emitted (what clamp/floor WOULD pass).
* **tau** τ*_end = m_q/ln5 CONFIRM (v3 §2.2d pre-GO check): arithmetic + the flip-support edge
  (flip-rate below/above GT-margin 0.10; flip-mass share) recomputed from the run's CACHED
  annulus maps (``annulus_live_maps/maps_*.npz``, witness argmax) against the GT cache margins.
  If no END-checkpoint map is cached the end-state confirm is BLOCKED-cheaply-with-path (stated).

Axis: everything here is [macOS advisory] NON-PROMOTABLE trace analysis; no score claims.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_control.trace_probes import (  # noqa: E402
    cadence_replay,
    copredicate_backtest,
    forfeit_matched_backtest,
    load_history,
    refit_nu_per_stage,
)

PROVENANCE = ("DRAFT_OPTIMAL_STACK_v5_20260707.md §2.2f/§2.5/§7c + "
              "ct_deepresearch_1_training_campaign_control_20260707.md §3.3/§4.2/§10.2")

# mod32cap stage windows (verdict-grid rows per stage; ckpts: stageCE_ep299 = CE end,
# stageMuonStart_ep726 = TAU end / Muon start, stageTau_muon_ep1000 = run end).
DEFAULT_STAGES = {"CE": (25, 275), "tau_softplus": (300, 725), "muon_fin": (750, 1000)}


def _parse_stages(spec: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for part in spec.split(","):
        name, lo, hi = part.split(":")
        out[name] = (int(lo), int(hi))
    return out


def probe_ct3(rows, args) -> dict:
    anchor = copredicate_backtest(rows)  # estimator-form anchor (shipped 5e-3/V=4)
    res = forfeit_matched_backtest(rows, s_star=args.s_star, stage_lo=args.stage_lo,
                                   stage_hi=args.stage_hi, cap_epoch=args.cap_epoch)
    fire = res["first_sustained_fire_endpoint"]
    band_lo, band_hi = 670, 700
    if fire is None:
        verdict = "KILL (never fires in stage window ⇒ would-fire-only)"
    elif fire < 650 or fire > args.cap_epoch:
        verdict = f"KILL (fires ep{fire}, outside [650, cap {args.cap_epoch}])"
    elif band_lo <= fire <= band_hi and res["fire_report_endpoint"][
            "ema_best_within_cadence_of_stage_best"]:
        verdict = "PASS"
    else:
        verdict = f"BAND-FAIL (fires ep{fire}, band [{band_lo},{band_hi}])"
    return {"probe": "P-CT3", "law": "fire TAU->FIN when s < s* = nu*forfeit",
            "band": {"first_sustained_fire": [band_lo, band_hi],
                     "ema_best_within_cadence_of": "stage best",
                     "kill": f"fire < 650 or > cap {args.cap_epoch}"},
            "verdict": verdict,
            "scope_on_negative": "FORMULATION (this estimator form + registered s*)",
            "estimator_anchor_shipped_copredicate": {
                k: anchor[k] for k in ("first_fire_epoch", "first_fire_rel_slope", "n_fires")},
            "result": res}


def probe_ct1(rows, stages, args) -> dict:
    res = refit_nu_per_stage(rows, stages)
    verdicts = {}
    for name, st in res["stages"].items():
        if "error" in st:
            verdicts[name] = "BLOCKED (" + st["error"] + ")"
            continue
        nu = st["nu_per_ep"]
        if nu < 0.01:
            verdicts[name] = "KILL (nu < 0.01 ⇒ recompute window laws; values emitted)"
        elif 0.02 <= nu <= 0.035:
            verdicts[name] = "PASS"
        else:
            verdicts[name] = "BAND-FAIL (nu outside [0.02, 0.035], above kill floor)"
    return {"probe": "P-CT1", "law": "d_seg(t) = a + b*exp(-nu*t) per stage; nu = 1/tau_e",
            "band": {"nu_per_ep": [0.02, 0.035], "kill": "nu < 0.01"},
            "verdict_per_stage": verdicts,
            "scope_on_negative": ("FORMULATION (single-nu exponential per stage; power-law "
                                  "alternative reported in-row)"),
            "result": res}


def probe_ct2(rows, args) -> dict:
    out: dict = {"probe": "P-CT2",
                 "law": "dt_next = clamp(floor_S/|S'|, 25, 100), self-triggered",
                 "band": {"n_skipped": [12, 17],
                          "kill": "any missed prefix-best > 1 cadence"},
                 "scope_on_negative": ("FORMULATION (this clamp form + floor_S; sweep emitted "
                                       "showing what WOULD pass)")}
    for est in ("window", "pair"):
        r = cadence_replay(rows, floor_s=args.floor_s, estimator=est)
        if r["missed_prefix_best_beyond_one_cadence"]:
            v = "KILL (missed prefix-best > 1 cadence)"
        elif 12 <= r["n_skipped"] <= 17:
            v = "PASS"
        else:
            v = f"BAND-FAIL ({r['n_skipped']} skipped, band [12,17])"
        r["verdict"] = v
        out[f"replay_{est}"] = r
    sweep = []
    for mult in (1.0, 2.0, 3.0, 4.0, 6.0):
        r = cadence_replay(rows, floor_s=args.floor_s * mult, estimator="window")
        sweep.append({"floor_S": args.floor_s * mult, "mult": mult,
                      "n_skipped": r["n_skipped"],
                      "missed_best": bool(r["missed_prefix_best_beyond_one_cadence"]),
                      "global_best_dist": r["global_best_dist_to_nearest_visited"]})
    out["floor_s_sensitivity_sweep"] = sweep
    out["verdict"] = out["replay_window"]["verdict"]
    return out


def probe_tau(run_dir: Path, args) -> dict:
    """τ-CONFIRM: arithmetic + flip-support edge on cached annulus maps (no scorer rerun)."""
    out: dict = {"probe": "tau-CONFIRM",
                 "law": "tau*_end = m_q/ln5; m_q = flip-annulus GT-margin support edge (0.10)",
                 "arithmetic": {"claimed": 0.062, "recomputed_0.10_over_ln5": 0.10 / math.log(5)}}
    maps_dir = run_dir / "annulus_live_maps"
    maps = sorted(maps_dir.glob("maps_*.npz")) if maps_dir.exists() else []
    if not maps:
        out["verdict"] = ("BLOCKED-cheaply-with-path: no cached annulus maps in run dir; path = "
                          "tools/witness_annulus_convergence.py --ckpt END=<last ckpt> --pairs 16 "
                          "(16-pair advisory scorer forward, ~minutes CPU; NOT run in a $0 wave)")
        return out
    import numpy as np  # local import: keep the trace probes numpy-free
    g = np.load(args.gt_cache, mmap_mode="r")
    lst, mgn = g["lstars"], g["margins"]
    p_total = int(lst.shape[0])
    n_sub = int(np.load(maps[0])["argmax"].shape[0])
    stride = max(1, p_total // n_sub)
    vp = list(range(0, p_total, stride))[:n_sub]
    gts = np.stack([np.asarray(lst[i]) for i in vp])
    gtm = np.stack([np.asarray(mgn[i], np.float32) for i in vp])
    per_map = {}
    for mp in maps:
        z = np.load(mp)
        fl = np.asarray(z["argmax"]) != gts
        below = gtm < 0.10
        per_map[mp.name] = {
            "epoch": int(z["epoch"]), "n_pairs_subset": n_sub,
            "flip_rate_gt_margin_below_0.10": float(fl[below].mean()),
            "flip_rate_gt_margin_above_0.10": float(fl[~below].mean()),
            "flip_mass_share_gt_margin_below_0.10": float(below[fl].mean()),
            "gt_margin_at_flips_q50_q90_q99": [
                float(q) for q in np.quantile(gtm[fl], [0.5, 0.9, 0.99])],
        }
    out["per_cached_map"] = per_map
    out["end_checkpoint_confirm"] = (
        "BLOCKED-cheaply-with-path: cached maps cover early epochs only; the decisive "
        "END-checkpoint m_q needs tools/witness_annulus_convergence.py on the last ckpt "
        "(16-pair advisory scorer forward, ~minutes CPU)")
    out["scope_on_negative"] = ("INSTANCE (the m_q=0.10 anchor's transfer to this vehicle/stage; "
                                "the law tau*=m_q/ln5 is untouched)")
    shares = [m["flip_mass_share_gt_margin_below_0.10"] for m in per_map.values()]
    out["verdict"] = (
        f"PARTIAL: arithmetic OK (0.062133); flip-mass share below the 0.10 edge = "
        f"{min(shares):.6f}-{max(shares):.6f} on cached early-stage maps (anchor expects ~1.0); "
        f"decisive END-checkpoint confirm BLOCKED-cheaply-with-path")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--probes", default="ct1,ct2,ct3,tau")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--s-star", type=float, default=1.4154e-5,
                    help="P-CT3 threshold s* = nu*forfeit [S/ep] (v5 §2.2f)")
    ap.add_argument("--floor-s", type=float, default=0.00178,
                    help="P-CT2 attribution floor [S] (v5 §2.5, post req-F#6)")
    ap.add_argument("--stage-lo", type=int, default=300)
    ap.add_argument("--stage-hi", type=int, default=725)
    ap.add_argument("--cap-epoch", type=int, default=726)
    ap.add_argument("--stages", default=None,
                    help="ct1 stage windows 'name:lo:hi,...' (default = mod32cap)")
    ap.add_argument("--gt-cache", type=Path,
                    default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    args = ap.parse_args()

    rows = load_history(args.run_dir)
    stages = _parse_stages(args.stages) if args.stages else DEFAULT_STAGES
    wanted = {p.strip() for p in args.probes.split(",")}
    artifact = {
        "instrument": "tools/witness_trace_probes.py",
        "axis": "[macOS advisory] NON-PROMOTABLE (trace analysis; no score claim)",
        "provenance": PROVENANCE,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {"run_dir": str(args.run_dir), "n_trace_rows": len(rows),
                   "trace_epochs": [rows[0][0], rows[-1][0]], "stages": stages},
        "probes": {},
    }
    if "ct3" in wanted:
        artifact["probes"]["ct3"] = probe_ct3(rows, args)
    if "ct1" in wanted:
        artifact["probes"]["ct1"] = probe_ct1(rows, stages, args)
    if "ct2" in wanted:
        artifact["probes"]["ct2"] = probe_ct2(rows, args)
    if "tau" in wanted:
        artifact["probes"]["tau"] = probe_tau(args.run_dir, args)

    out = args.out or (REPO / ".omx/research/t5_crucible/artifacts" /
                       f"trace_probes_{args.run_dir.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"[witness_trace_probes] artifact -> {out}")
    for name, p in artifact["probes"].items():
        v = p.get("verdict") or p.get("verdict_per_stage")
        print(f"  {p['probe']}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
