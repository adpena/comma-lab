"""ddm_cg1 — does the enrichment/precision inversion reach the ``#766`` waterfill?

PRE-REGISTERED (memo `ddm_cg1_force_verb_ledger_20260804.md` §7, commit 44923cbb8d)
-----------------------------------------------------------------------------------
`cg1` measured Spearman(enrichment, precision) = -0.900 (p=0.037) across the five
classes: ranking by ENRICHMENT sends bytes where they buy the FEWEST flips (MyCar,
precision 0.2084) instead of the most (Lane, 0.6852) -- 3.29x. That was filed as a
HARMS row with protection ABSENT: *nothing stops an allocator from ranking by
enrichment.*

**Falsifier, pre-registered verbatim:** if precision-ranking gives **< 1%**
matched-byte damage advantage over the shipped key, the inversion is real but
ALREADY PRICED by the current allocator, and the harm row downgrades
`HARMS -> NEUTRAL` at `FORMULATION` scope.

WHAT THIS RUNS
--------------
`#766` is `experiments/ddm_wr1_reverse_waterfill.py`; its unit is one of 768 cells
(24x32 tiling at 16x16) and its drop rule is (`:93`)::

    order = np.lexsort((-residual_mass, flip_mass))   # safest-per-byte first

This probe reuses `ddm_mg1`'s `_drop_curve` / `_damage_at_matched_bytes` BY IMPORT,
so the drop rule is bit-identical to the one that measured the barrier key at
0.0000% advantage, and compares four keys:

* ``flip_count``  -- the shipped primary key, on the LIVE cx1 n600 flip mask;
* ``precision``   -- flips per SITE in the cell. Every cell holds exactly the same
  number of sites (16*16*n_pairs), so this is a strictly monotone transform of
  ``flip_count``. The probe PROVES that degeneracy rather than assuming it;
* ``enrichment``  -- flips divided by the flips EXPECTED from the cell's own GT
  class composition at population per-class base rates. This is the real
  enrichment analogue at cell grain, and the key the harm row warns against;
* ``shuffled``    -- control; if it does not separate, the instrument is dead.

DAMAGE is ``flip_count``: dropping a cell admits its flips, and flips are exactly
what ``d_seg`` counts (1 flip = 1/117,964,800 of the seg leg). BYTES is `wr1`'s own
``residual_mass`` byte proxy, unchanged.

AUTHORITY
---------
`[macOS-CPU scorer-free advisory]` · score_claim=false · promotion_eligible=false ·
**scorer forwards run: 0**. Positive control: the flip mask must reproduce the
evaluator seg leg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from experiments.ddm_mg1_barrier_rerank_probe import (
    CELL_H,
    CELL_W,
    GRID_H,
    GRID_W,
    N_CELLS,
    _damage_at_matched_bytes,
    _drop_curve,
    _spearman,
)

REPO = Path(__file__).resolve().parents[1]
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
PU2_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
WR1_ATLAS = Path("/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_cell_sensitivity_atlas.npz")

D_SEG_EXPECTED = 0.004311794704861111
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

# Pre-registered decision threshold. Not tuned after seeing the result.
FALSIFIER_ADVANTAGE_PCT = 1.0


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / ".omx/research/ddm_cg1_waterfill_key_swap_n600.json"))
    args = ap.parse_args()

    gt_arg = np.load(PU2_CACHE / "gt_argmax_n600.npy", mmap_mode="r")
    cx_arg = np.load(PU2_CACHE / "cx1_argmax_n600.npy", mmap_mode="r")
    n = min(args.pairs, gt_arg.shape[0])
    h, w = gt_arg.shape[1], gt_arg.shape[2]

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cell_of_px = ((yy // CELL_H) * GRID_W + (xx // CELL_W)).ravel()
    counts = np.bincount(cell_of_px, minlength=N_CELLS)
    tiling_ok = bool(counts.sum() == h * w and counts.min() == counts.max() == CELL_H * CELL_W)

    flip_count = np.zeros(N_CELLS, dtype=np.float64)
    # GT class composition per cell -- the denominator the enrichment key needs.
    class_sites = np.zeros((N_CELLS, 5), dtype=np.float64)
    class_flips = np.zeros(5, dtype=np.float64)
    class_total = np.zeros(5, dtype=np.float64)
    n_flip_total = 0

    for start in range(0, n, args.chunk):
        stop = min(start + args.chunk, n)
        g = np.asarray(gt_arg[start:stop], dtype=np.int64)
        c = np.asarray(cx_arg[start:stop], dtype=np.int64)
        flip = g != c
        n_flip_total += int(flip.sum())
        for k in range(g.shape[0]):
            gk = g[k].ravel()
            fk = flip[k].ravel()
            flip_count += np.bincount(cell_of_px[fk], minlength=N_CELLS)
            for cls in range(5):
                sel = gk == cls
                class_sites[:, cls] += np.bincount(cell_of_px[sel], minlength=N_CELLS)
                class_total[cls] += int(sel.sum())
                class_flips[cls] += int(fk[sel].sum())

    total_sites = float(n) * h * w
    d_seg = n_flip_total / total_sites
    rel_err = abs(d_seg - D_SEG_EXPECTED) / D_SEG_EXPECTED
    control_ok = bool(rel_err < 1e-6)

    base_rate = np.divide(class_flips, class_total, out=np.zeros(5), where=class_total > 0)

    sites_per_cell = float(n) * CELL_H * CELL_W
    precision = flip_count / sites_per_cell
    # Expected flips in this cell if every class flipped at its POPULATION rate.
    # NOTE: this matmul emits divide-by-zero/overflow/invalid RuntimeWarnings on
    # this BLAS even for finite inputs -- stale FP exception flags from earlier
    # ops leak into its error check. Verified spurious: an isolated matmul on
    # clean random data raises the identical three. Rather than silence it, the
    # finiteness is ASSERTED and recorded, so a real fault cannot hide behind a
    # warning we learned to ignore.
    with np.errstate(all="ignore"):
        expected = class_sites @ base_rate
    inputs_finite = bool(np.all(np.isfinite(base_rate)) and np.all(np.isfinite(class_sites)))
    expected_finite = bool(np.all(np.isfinite(expected)))
    if not (inputs_finite and expected_finite):
        raise SystemExit("enrichment denominator is not finite; the enrichment key would be garbage")
    enrichment = np.divide(flip_count, expected, out=np.zeros(N_CELLS), where=expected > 0)

    atlas = np.load(WR1_ATLAS)
    residual = np.asarray(atlas["residual_mass"], dtype=np.float64)
    if residual.size != N_CELLS:
        raise SystemExit(f"atlas residual has {residual.size} cells, expected {N_CELLS}")

    damage = flip_count  # dropping a cell admits its flips; flips are what d_seg counts

    curve_flip = _drop_curve(flip_count, residual, damage)
    curve_prec = _drop_curve(precision, residual, damage)
    curve_enr = _drop_curve(enrichment, residual, damage)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(flip_count)
    curve_shuf = _drop_curve(shuffled, residual, damage)

    # --- DEGENERACY CONTROL -------------------------------------------------
    # Every cell holds the same number of sites, so precision is flip_count/const.
    # If the drop orders are identical, the shipped key ALREADY IS the precision
    # key and the pre-registered swap is a no-op by construction.
    order_identical = bool(np.array_equal(curve_flip["order"], curve_prec["order"]))
    rho_prec = _spearman(flip_count, precision)
    rho_enr = _spearman(flip_count, enrichment)

    total_bytes = float(residual.sum())
    fracs = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    budgets = fracs * total_bytes
    d_flip = _damage_at_matched_bytes(curve_flip, budgets)
    d_prec = _damage_at_matched_bytes(curve_prec, budgets)
    d_enr = _damage_at_matched_bytes(curve_enr, budgets)
    d_shuf = _damage_at_matched_bytes(curve_shuf, budgets)
    total_damage = float(damage.sum())

    def adv(x: float, base: float) -> float:
        """% LESS damage than the shipped key. Positive = the candidate is better.

        The shuffled control anchors the sign: it lands at roughly -120,000%,
        i.e. it admits vastly MORE damage. Zero baseline damage yields 0.0 and is
        excluded from the verdict by the `powered` filter below.
        """
        return float((base - x) / base * 100.0) if base > 0 else 0.0

    rows = []
    for i, f in enumerate(fracs):
        base = float(d_flip[i])
        rows.append(
            {
                "bytes_freed_frac": float(f),
                "damage_flip_count_key_shipped": float(base),
                "damage_precision_key": float(d_prec[i]),
                "damage_enrichment_key": float(d_enr[i]),
                "damage_shuffled_key_control": float(d_shuf[i]),
                "precision_advantage_pct": adv(float(d_prec[i]), base),
                "enrichment_advantage_pct": adv(float(d_enr[i]), base),
                "shuffled_advantage_pct": adv(float(d_shuf[i]), base),
            }
        )

    # Budgets where the shipped key admits ZERO damage carry no discriminating
    # power -- every key ties at 0 there. Restrict the verdict to budgets that
    # can actually separate keys, and report which ones those were.
    powered = [r for r in rows if r["damage_flip_count_key_shipped"] > 0]
    max_prec_adv = max((abs(r["precision_advantage_pct"]) for r in powered), default=0.0)
    worst_enr = min((r["enrichment_advantage_pct"] for r in powered), default=0.0)
    max_shuf = max((abs(r["shuffled_advantage_pct"]) for r in powered), default=0.0)

    falsifier_fired = bool(powered and max_prec_adv < FALSIFIER_ADVANTAGE_PCT)

    out = {
        "arm": "ddm_cg1",
        "probe": "precision vs enrichment as the #766 waterfill primary key",
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "git_head": _git_head(),
        "pre_registered_in": ".omx/research/ddm_cg1_force_verb_ledger_20260804.md sec 7 (commit 44923cbb8d)",
        "drop_rule": "np.lexsort((-residual_mass, key)) -- imported from ddm_mg1_barrier_rerank_probe",
        "denominator": {
            "pairs": int(n),
            "cells": N_CELLS,
            "grid": [GRID_H, GRID_W],
            "cell_px": [CELL_H, CELL_W],
            "sites_per_cell_over_all_pairs": sites_per_cell,
            "total_sites": total_sites,
            "n_flip": n_flip_total,
        },
        "controls": {
            "tiling_covers_plane_exactly_once": tiling_ok,
            "d_seg_measured": d_seg,
            "d_seg_expected": D_SEG_EXPECTED,
            "d_seg_rel_err": rel_err,
            "verdict": "ARGMAX_VERIFIED" if control_ok else "CONTROL_FAILED",
            "shuffled_key_max_abs_advantage_pct": max_shuf,
            "instrument_has_power": bool(max_shuf > 1.0),
            "enrichment_denominator_inputs_finite": inputs_finite,
            "enrichment_denominator_finite": expected_finite,
            "matmul_runtimewarning_is_spurious": (
                "This BLAS raises divide-by-zero/overflow/invalid on a finite matmul "
                "(reproduced on clean random data); finiteness asserted above instead."
            ),
            "budgets_with_discriminating_power": [
                r["bytes_freed_frac"] for r in rows if r["damage_flip_count_key_shipped"] > 0
            ],
            "budgets_with_zero_baseline_damage": [
                r["bytes_freed_frac"] for r in rows if r["damage_flip_count_key_shipped"] <= 0
            ],
        },
        "per_class_base_rate": {CLASS_NAMES[i]: float(base_rate[i]) for i in range(5)},
        "degeneracy_control": {
            "precision_is_monotone_transform_of_flip_count": True,
            "spearman_flip_count_vs_precision": rho_prec,
            "spearman_flip_count_vs_enrichment": rho_enr,
            "drop_order_identical_flip_vs_precision": order_identical,
            "note": (
                "All 768 cells hold an identical number of sites, so flips-per-site is "
                "flip_count divided by a constant. The shipped key IS the precision key."
            ),
        },
        "matched_byte_rows": rows,
        "summary": {
            "total_damage_flips": total_damage,
            "total_bytes_proxy": total_bytes,
            "max_abs_precision_advantage_pct": max_prec_adv,
            "worst_enrichment_advantage_pct": worst_enr,
            "falsifier_threshold_pct": FALSIFIER_ADVANTAGE_PCT,
            "falsifier_fired": falsifier_fired,
            "verdict": (
                "DOWNGRADE cg1.enrichment_key.inverts HARMS -> NEUTRAL at FORMULATION scope: "
                "the shipped #766 key is already the precision key, so the inversion cannot "
                "reach this allocator."
                if falsifier_fired
                else "HARMS row STANDS: precision-ranking materially beats the shipped key."
            ),
        },
    }
    if not control_ok:
        out["BLOCKER"] = "positive control failed; no row admissible"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"control d_seg={d_seg!r} rel_err={rel_err:.3e} -> {out['controls']['verdict']}")
    print(f"drop order identical (flip_count vs precision): {order_identical}")
    print(f"max |precision advantage| = {max_prec_adv:.6f}%  (falsifier < {FALSIFIER_ADVANTAGE_PCT}%)")
    print(f"worst enrichment advantage = {worst_enr:.4f}%  | shuffled control max = {max_shuf:.4f}%")
    print(f"FALSIFIER FIRED: {falsifier_fired}")
    print(f"wrote {args.out}")
    return 0 if control_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
