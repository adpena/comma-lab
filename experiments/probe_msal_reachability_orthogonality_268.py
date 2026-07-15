#!/usr/bin/env python3
"""PROBE #268 — margin-saliency REACHABILITY orthogonality ($0, disjoint, optimal-form).

Sweep Arm A drain, 2026-07-14. NON-owned surface (NOT train_levelset_witness*).

Question this probe answers (the OWED decision for the msal_uni_sr_reachability_268
curriculum-pool candidate, gate "ZERO build: gt_n600_sR.npz READY; A/B on the fragile
annulus band"):

  LEVER-4 (margin-saliency) multiplies the fragility weight sal = exp(-margin/tau)
  by the cached through-R reachability map S_R when --margin-saliency-reachability is
  on. Whether that MULTIPLY adds information depends on whether S_R is ORTHOGONAL to
  the fragility weight it multiplies:

    * If S_R strongly tracks fragility (high +Spearman(S_R, sal), i.e. S_R is just
      "small GT margin again"), the reachability multiply is REDUNDANT with the
      weight it multiplies -> the training A/B is predicted marginal.
    * If S_R is ~uncorrelated with fragility, it selects a DIFFERENT subset of the
      annulus (fragile-AND-reachable) -> real new signal -> ROUTE the training A/B.
    * The trainer already MEASURED texture-proxy-vs-S_R Pearson = -0.033 (INERT);
      this probe measures the ORTHOGONAL axis: S_R vs the *fragility* weight and
      vs raw margin, which motivates whether reachability replaces a merely-inert
      texture weight with a genuinely-informative one.

AUTHORITY: advisory [macOS-CPU research-signal] on the FROZEN GT caches. This is a
correlation of two cached GT fields; it is NOT a score claim and NOT a training result.
The verdict here can only ROUTE / de-prioritise the training A/B, never adopt/kill it
(verdict-scope: FORMULATION at most; a training A/B is the real arbiter).

Inputs (frozen GT caches, read-only):
  --gt-cache     experiments/results/mlx_fleet_gt_cache/gt_n600.npz      (margins, lstars)
  --sr-cache     experiments/results/mlx_fleet_gt_cache/gt_n600_sR.npz   (sR)

Defaults mirror the trainer: tau=0.5 (--margin-saliency-tau), band=1.0 (annulus |margin|<band).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Trainer LEVER-4 defaults (experiments/train_levelset_witness_realized_through_R_mlx.py L13561).
DEFAULT_TAU = 0.5
DEFAULT_BAND = 1.0
_ADVISORY = ("[macOS-CPU advisory research-signal] correlation of frozen GT caches; "
             "NOT a score claim, NOT a training result; can only route/de-prioritise the "
             "training A/B (verdict-scope FORMULATION).")


def _per_pair_stats(margin: np.ndarray, sR: np.ndarray, tau: float, band: float) -> dict | None:
    """One pair. margin,sR are (H,W) float32. Returns None if the annulus is degenerate."""
    ann = margin < band                      # fragile annulus (small GT top1-top2 margin)
    n_ann = int(ann.sum())
    if n_ann < 32:
        return None
    m = margin[ann].astype(np.float64)
    r = sR[ann].astype(np.float64)
    sal = np.exp(-m / tau)                    # the exact fragility weight LEVER-4 multiplies
    # S_R vs the fragility weight it multiplies (the redundancy axis).
    if float(r.std()) < 1e-12 or float(sal.std()) < 1e-12:
        rho_sal = 0.0
    else:
        rho_sal = float(spearmanr(r, sal).statistic)
    # S_R vs raw margin (negative rho_margin == S_R high where margin small == tracks fragility).
    if float(r.std()) < 1e-12 or float(m.std()) < 1e-12:
        rho_margin = 0.0
    else:
        rho_margin = float(spearmanr(r, m).statistic)
    # Concentration: what fraction of TOTAL S_R mass (whole frame) lands inside the annulus.
    tot = float(sR.astype(np.float64).sum())
    mass_in_ann = float(sR[ann].astype(np.float64).sum()) / tot if tot > 1e-12 else 0.0
    frac_area_ann = n_ann / float(margin.size)
    # Dynamic range of S_R WITHIN the annulus: guards against "orthogonal" laundering "S_R is
    # flat here" (a flat S_R would give rho~0 AND carry zero targeting information -> NOT a route).
    r_mean = float(r.mean())
    r_std = float(r.std())
    r_cv = (r_std / r_mean) if r_mean > 1e-9 else 0.0          # coefficient of variation
    frac_hi = float((r > 0.5).mean())                          # fraction of annulus with S_R>0.5
    return {
        "rho_sR_vs_sal": rho_sal,
        "rho_sR_vs_margin": rho_margin,
        "sR_mass_frac_in_annulus": mass_in_ann,
        "annulus_area_frac": frac_area_ann,
        "sR_ann_mean": r_mean,
        "sR_ann_cv": r_cv,
        "sR_ann_frac_hi": frac_hi,
        "n_ann": n_ann,
    }


def run_probe(gt_cache: Path, sr_cache: Path, tau: float = DEFAULT_TAU,
              band: float = DEFAULT_BAND, num_pairs: int | None = None) -> dict:
    """Stream per-pair; aggregate honest per-pair distribution. n600-scale by default."""
    zc = np.load(gt_cache, allow_pickle=False, mmap_mode="r")
    zs = np.load(sr_cache, allow_pickle=False, mmap_mode="r")
    margins = zc["margins"]                   # (P,H,W) f32 (mmap: sliced per pair, no balloon)
    sR = zs["sR"]                             # (P,H,W) f32
    P = int(margins.shape[0])
    if num_pairs is not None:
        P = min(P, int(num_pairs))
    P = min(P, int(sR.shape[0]))
    rows = []
    for pi in range(P):
        st = _per_pair_stats(np.asarray(margins[pi]), np.asarray(sR[pi]), tau, band)
        if st is not None:
            rows.append(st)
    if not rows:
        raise RuntimeError("no valid pairs (all annuli degenerate)")

    def _agg(key: str) -> dict:
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std()),
                "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                "p90": float(np.percentile(v, 90))}

    rho_sal = _agg("rho_sR_vs_sal")
    rho_margin = _agg("rho_sR_vs_margin")
    mass = _agg("sR_mass_frac_in_annulus")
    area = _agg("annulus_area_frac")
    sR_ann_cv = _agg("sR_ann_cv")
    sR_ann_frac_hi = _agg("sR_ann_frac_hi")
    sR_ann_mean = _agg("sR_ann_mean")

    # Decision guard (attack-your-own-conclusion, §6): "orthogonal" only ROUTES if S_R ALSO has
    # real dynamic range inside the annulus. A flat S_R gives rho~0 AND carries zero targeting
    # information -> that is UNINFORMATIVE, not a route. Require cv (coeff of variation) > 0.3.
    absrho = abs(rho_sal["mean"])
    informative = sR_ann_cv["mean"] > 0.3
    if not informative:
        verdict = "UNINFORMATIVE_FLAT_IN_ANNULUS"
        rationale = ("S_R has near-zero dynamic range inside the fragile annulus (cv<=0.3) -> the "
                     "orthogonality is flatness, not signal -> do NOT route on this basis.")
    elif absrho < 0.15:
        verdict = "ORTHOGONAL_ADDS_SIGNAL"
        rationale = ("S_R is ~uncorrelated with the fragility weight it multiplies AND has real "
                     "annulus dynamic range -> reachability selects a DIFFERENT annulus subset "
                     "(fragile-AND-reachable) -> ROUTE the training A/B.")
    elif rho_sal["mean"] >= 0.15:
        verdict = "REDUNDANT_TRACKS_FRAGILITY"
        rationale = ("S_R positively tracks the fragility weight -> the multiply is near-redundant "
                     "-> training A/B predicted marginal; de-prioritise vs orthogonal levers.")
    else:
        verdict = "ANTI_CORRELATED_SUSPICIOUS"
        rationale = ("S_R is negatively correlated with fragility -> it DE-emphasises the fragile "
                     "band -> re-derive the S_R build before routing.")

    return {
        "probe": "msal_reachability_orthogonality_268",
        "authority": _ADVISORY,
        "gt_cache": str(gt_cache), "sr_cache": str(sr_cache),
        "tau": tau, "band": band, "n_pairs_used": len(rows), "n_pairs_requested": P,
        "spearman_sR_vs_fragility_weight": rho_sal,
        "spearman_sR_vs_margin": rho_margin,
        "sR_mass_frac_in_annulus": mass,
        "annulus_area_frac": area,
        "sR_ann_mean": sR_ann_mean,
        "sR_ann_cv": sR_ann_cv,
        "sR_ann_frac_hi": sR_ann_frac_hi,
        "verdict": verdict,
        "rationale": rationale,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--gt-cache", type=Path,
                    default=root / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--sr-cache", type=Path,
                    default=root / "experiments/results/mlx_fleet_gt_cache/gt_n600_sR.npz")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--band", type=float, default=DEFAULT_BAND)
    ap.add_argument("--num-pairs", type=int, default=None, help="cap pairs (default: all = n600)")
    args = ap.parse_args()
    out = run_probe(args.gt_cache, args.sr_cache, args.tau, args.band, args.num_pairs)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main()
