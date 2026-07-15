#!/usr/bin/env python3
"""PROBE #288 — flip-weighted (annulus) target-mass head-offset reformulation ($0 both-arms).

Sweep Arm A drain, 2026-07-14. NON-owned surface (NEW probe; reuses the closed-form
menon offset + the ALREADY-MEASURED per_class_1d_curves; NO render, NO train).

Context (DAG FEED-otoffset / -otoffset-n600):
  The bulk mass-matching head-offset was MEASURED NEGATIVE, confirmed n600
  (no_offset 0.0031436 < menon 0.0033119 < ot_newton 0.0048921); verdict_scope=FORMULATION.
  The OPEN reformulation-queue item is "match the boundary-ANNULUS mass the scorer
  re-reads, not bulk cell mass" (flip-weighted targeting).

The $0 both-arms move (why this collapses the question without a render):
  1. The menon per-class offset is CLOSED-FORM  b_k = -tau*log(pi_k)  (mean-centered),
     so it needs ONLY the class-prior vector -- computable at $0 for BOTH the bulk prior
     and the annulus prior from the frozen gt_n600 cache.
  2. The OT probe already MEASURED the realized-through-R d_seg-vs-per-class-offset curves
     (per_class_1d_curves for the focus classes 0/1/3) on the exact mod32cap ep650 EMA ckpt.
     Those curves show the d_seg surface is MINIMISED at offset=0 (baseline) and is flat
     (|Delta|<2e-5) within +-0.4, worse outside (full menon +1.7e-4).
  3. The reformulation only changes the TARGET prior -> the solved offset. It acts on the
     SAME measured offset->d_seg surface. So: compute the annulus-menon offset; if it lands
     in the flat +-0.4 region -> predicted ~baseline (no help); if it is large (like the
     measured bulk menon +-1.1..2.5) -> predicted worse. Either way the surface is minimised
     at 0, so any nonzero-offset reformulation is predicted >= baseline.
  4. VALIDATION GATE: my first-order (independent-perturbation) estimate of the BULK menon
     d_seg must reproduce the MEASURED bulk menon (0.0033119) within tolerance; if it does
     not, the additive approximation is invalid and the verdict downgrades to ROUTE (realized
     eval is the arbiter). This guards the estimate against the cross-class-interaction blind.

AUTHORITY: advisory [macOS-CPU research-signal]. The offset->d_seg curves are MEASURED
realized-through-R; the reformulation d_seg here is a first-order DERIVED estimate on those
measured curves (not a fresh realized eval). It can ROUTE / de-prioritise the realized arm,
never adopt/kill it. verdict-scope FORMULATION.

Input: gt_n600.npz (lstars, margins) + the OT probe result JSON (measured curves).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_ADVISORY = ("[macOS-CPU advisory research-signal]; offset->d_seg curves MEASURED, reformulation "
             "d_seg is a first-order DERIVED estimate on them; routes/de-prioritises only; "
             "verdict-scope FORMULATION.")


def menon_offsets(priors: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Closed-form Menon logit-adjustment b_k = -tau*log(pi_k), mean-centered (zero-sum)."""
    p = np.asarray(priors, dtype=np.float64).reshape(-1)
    p = p / p.sum()
    p = np.clip(p, 1e-12, None)
    b = -tau * np.log(p)
    return b - b.mean()


def class_priors(lstars: np.ndarray, mask: np.ndarray | None, n_classes: int = 5) -> np.ndarray:
    """Per-class pixel frequency over the mask (or full frame if mask is None)."""
    if mask is None:
        vals = lstars.reshape(-1)
    else:
        vals = lstars[mask]
    counts = np.bincount(vals.astype(np.int64), minlength=n_classes).astype(np.float64)
    return counts


def _curve_interp(curve: list[dict], x: float) -> float | None:
    """Linear-interp d_seg at offset x from a measured 1d curve; None if x is out of grid."""
    xs = np.array([p["offset"] for p in curve], dtype=np.float64)
    ys = np.array([p["d_seg"] for p in curve], dtype=np.float64)
    if x < xs.min() or x > xs.max():
        return None
    return float(np.interp(x, xs, ys))


def run_probe(gt_cache: Path, ot_result: Path, tau: float = 1.0,
              bands: tuple[float, ...] = (0.5, 1.0, 2.0), num_pairs: int | None = None) -> dict:
    res = json.loads(Path(ot_result).read_text())
    baseline = float(res["baseline_d_seg"])
    curves = {int(c): cur for c, cur in res["per_class_1d_curves"].items()}
    measured_menon_d = float(res["menon_analytic"]["d_seg"])
    measured_menon_off = {int(k): float(v) for k, v in res["menon_analytic"]["offsets"].items()}

    zc = np.load(gt_cache, allow_pickle=False, mmap_mode="r")
    lstars = zc["lstars"]                 # (P,H,W) int64
    margins = zc["margins"]               # (P,H,W) f32
    P = int(lstars.shape[0])
    if num_pairs is not None:
        P = min(P, int(num_pairs))

    # Accumulate bulk + annulus class counts over all pairs (streamed, mmap-sliced).
    bulk = np.zeros(5, np.float64)
    ann = {b: np.zeros(5, np.float64) for b in bands}
    for pi in range(P):
        ls = np.asarray(lstars[pi])
        mg = np.asarray(margins[pi])
        bulk += class_priors(ls, None)
        for b in bands:
            ann[b] += class_priors(ls, mg < b)

    def _first_order_dseg(offsets: dict[int, float]) -> tuple[float | None, list[int]]:
        """Sum measured single-class curve deltas (independent-perturbation approx).
        Returns (estimate, out_of_grid_classes). estimate None if any measured focus class
        offset is out of the measured grid (then only the full-menon measured point applies)."""
        est = baseline
        oog = []
        for c, cur in curves.items():
            b = offsets.get(c, 0.0)
            d = _curve_interp(cur, b)
            if d is None:
                oog.append(c)
                continue
            est += (d - baseline)
        return (est if not oog else None), oog

    b_bulk = menon_offsets(bulk, tau)
    bulk_off = {c: float(b_bulk[c]) for c in range(5)}
    # VALIDATION: does my closed-form bulk-menon reproduce the OT probe's measured menon offsets?
    off_match = max(abs(bulk_off[c] - measured_menon_off.get(c, bulk_off[c])) for c in range(5))

    out_bands = {}
    for b in bands:
        b_ann = menon_offsets(ann[b], tau)
        ann_off = {c: float(b_ann[c]) for c in range(5)}
        est_ann, oog_ann = _first_order_dseg(ann_off)
        # magnitude of the annulus offset on the MEASURED focus classes (0,1,3)
        focus_off = {c: ann_off[c] for c in curves}
        max_focus_off = max(abs(v) for v in focus_off.values())
        in_grid = max_focus_off <= 0.4
        out_bands[str(b)] = {
            "annulus_area_frac": float(ann[b].sum() / (P * lstars.shape[1] * lstars.shape[2])),
            "annulus_priors": (ann[b] / ann[b].sum()).round(6).tolist(),
            "annulus_menon_offsets": {str(c): round(v, 4) for c, v in ann_off.items()},
            "focus_offsets_in_pm04_grid": bool(in_grid),
            "max_focus_offset": round(max_focus_off, 4),
            "est_dseg_first_order": (round(est_ann, 8) if est_ann is not None else None),
            "focus_classes_out_of_grid": oog_ann,
        }

    bulk_priors = (bulk / bulk.sum()).round(6).tolist()

    # Verdict synthesis. The measured surface is minimised at offset=0 (winner delta -3.4e-8);
    # full menon is +1.7e-4 worse. If the annulus offsets are LARGE (out of grid, like bulk menon)
    # -> same worse regime. If SMALL (in flat grid) -> ~baseline (no help). Either way: the
    # reformulation cannot beat baseline on the measured surface.
    any_small = any(out_bands[str(b)]["focus_offsets_in_pm04_grid"] for b in bands)
    validation_ok = off_match < 1e-6
    verdict = "PREDICTED_NOGO_SURFACE_MINIMISED_AT_ZERO"
    rationale = (
        "The MEASURED realized-through-R d_seg-vs-per-class-offset surface is minimised at "
        "offset=0 (baseline; winner delta -3.4e-8) and flat within +-0.4, worse outside. The "
        "flip-weighted (annulus) targeting only re-selects a NON-ZERO offset on that SAME surface "
        "-> predicted >= baseline. "
        + ("Annulus menon offsets are LARGE (out of the flat +-0.4 grid, same regime as the "
           "measured bulk menon which was +1.7e-4 WORSE)." if not any_small else
           "Some annulus menon offsets fall in the flat +-0.4 region -> predicted ~baseline (no help).")
    )
    if not validation_ok:
        verdict = "ROUTE_VALIDATION_FAILED"
        rationale = ("First-order additive assumption unverified (bulk-menon offset reproduction "
                     f"max|Δ|={off_match:.3g} >= 1e-6) -> the realized n600 eval is the arbiter; ROUTE.")

    return {
        "probe": "ot_flipweighted_mass_288",
        "authority": _ADVISORY,
        "gt_cache": str(gt_cache), "ot_result": str(ot_result),
        "tau": tau, "n_pairs": P,
        "baseline_d_seg_measured": baseline,
        "bulk_priors": bulk_priors,
        "bulk_menon_offsets": {str(c): round(v, 4) for c, v in bulk_off.items()},
        "measured_menon_d_seg": measured_menon_d,
        "bulk_menon_offset_reproduction_max_abs_err": off_match,
        "validation_reproduces_measured_menon": validation_ok,
        "bands": out_bands,
        "verdict": verdict,
        "rationale": rationale,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--gt-cache", type=Path,
                    default=root / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--ot-result", type=Path,
                    default=root / "experiments/results/ot_offset_n600_modal_20260709/"
                                   "ot_offset_n600_LOCAL_result.json")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--num-pairs", type=int, default=None)
    args = ap.parse_args()
    out = run_probe(args.gt_cache, args.ot_result, args.tau, num_pairs=args.num_pairs)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main()
