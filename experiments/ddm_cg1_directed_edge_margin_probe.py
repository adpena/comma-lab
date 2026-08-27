#!/usr/bin/env python3
"""ddm_cg1(r) task #809 — per-DIRECTED-SIDE repair-cost (GT-margin) distributions on live cx1 flips.

The open cell this fills (stated denominator, m50):
  * ddm_cg1 (predecessor) measured margin bins per GT CLASS (5 rows).
  * ddm_pu2 measured directed flip COUNTS per side (20 sides), no margins.
  * ddm_mg1 refuted the per-side-AVERAGE barrier as a ranking key and named repair
    cost as a PER-SITE quantity (|m|) — untested per side.
  * ddm_ph4 measured a 10.05x DIRECTED barrier asymmetry (Road->Lane 51.26 vs
    Lane->Road 5.10) on hg1's depth-binned averages.
  This probe measures, per directed side (gt_class a -> rendered_class b), the
  PER-SITE GT-margin distribution over the realized flips of the live vehicle
  (cx1, n600): count, mean, median, p90, and the shallow-band share. That is the
  site-granular price account behind the "flip-count objectives buy Lane erasure
  at a discount" HARMS row: d_seg pays a uniform 1/N per flip on every side,
  while the repair depth per side differs by the ratios measured here.

Axis: [macOS-CPU scorer-free advisory]. Zero SegNet/PoseNet forwards. All inputs
are cached artifacts. score_claim=false, promotion_eligible=false.

Controls (fail-closed):
  1. total flips MUST equal pu2's 508,640 and d_seg MUST reproduce the evaluator
     row 0.004311794704861111 (rel err < 1e-9 vs recompute).
  2. the 5x5 directed confusion matrix MUST match pu2's receipt EXACTLY
     (cross-instrument, independently computed here from the raw argmax caches).
  3. GT margins MUST be non-negative everywhere (self-margin property).
  4. per-side counts MUST sum to total flips (partition check).

Inputs:
  /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy
  /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy
  /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_directed_flip_receipt.json
  experiments/results/mlx_fleet_gt_cache/gt_n600.npz  (margins)

Output:
  .omx/research/ddm_cg1_directed_edge_margin_n600.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
GT_NPZ = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT = REPO / ".omx/research/ddm_cg1_directed_edge_margin_n600.json"

CLASSES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
D_SEG_EXPECTED = 0.004311794704861111
TOTAL_FLIPS_EXPECTED = 508_640
SHALLOW_BAND = 0.096  # mg1's lowest band edge (42.51% precision band)


def fail(msg: str) -> None:
    print(f"CONTROL FAILED: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    gt = np.load(CACHE / "gt_argmax_n600.npy")   # (600, 384, 512) uint8
    cx = np.load(CACHE / "cx1_argmax_n600.npy")  # (600, 384, 512) uint8
    with np.load(GT_NPZ) as z:
        margins = z["margins"].astype(np.float64)  # (600, 384, 512) GT self-margin
    receipt = json.loads((CACHE / "cx1_directed_flip_receipt.json").read_text())

    if gt.shape != cx.shape or gt.shape != margins.shape:
        fail(f"shape mismatch gt={gt.shape} cx={cx.shape} margins={margins.shape}")

    # Control 3: GT self-margin is non-negative everywhere.
    neg = int((margins < 0).sum())
    if neg != 0:
        fail(f"{neg} negative GT margins — not the self-margin field")

    flip = gt != cx
    total_flips = int(flip.sum())
    d_seg = total_flips / flip.size
    # Control 1: reproduce the evaluator row.
    if total_flips != TOTAL_FLIPS_EXPECTED:
        fail(f"total flips {total_flips} != {TOTAL_FLIPS_EXPECTED}")
    if abs(d_seg - D_SEG_EXPECTED) / D_SEG_EXPECTED > 1e-9:
        fail(f"d_seg {d_seg} != {D_SEG_EXPECTED}")

    # Independent 5x5 confusion matrix over flip sites (and diagonal for context).
    conf = np.zeros((5, 5), dtype=np.int64)
    gt_f = gt[flip].astype(np.int64)
    cx_f = cx[flip].astype(np.int64)
    m_f = margins[flip]
    np.add.at(conf, (gt_f, cx_f), 1)
    diag = np.zeros(5, dtype=np.int64)
    for c in range(5):
        diag[c] = int(((gt == c) & ~flip).sum())
    conf_full = conf.copy()
    conf_full[np.arange(5), np.arange(5)] = diag

    # Control 2: match pu2's receipt exactly.
    pu2_conf = np.array(receipt["confusion_matrix_gt_by_rendered"], dtype=np.int64)
    if not np.array_equal(conf_full, pu2_conf):
        fail("confusion matrix does not match pu2's receipt")

    # Control 4: partition.
    if int(conf.sum()) != total_flips:
        fail(f"directed sides sum {int(conf.sum())} != {total_flips}")

    # Per-directed-side margin distributions.
    sides = []
    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            n = int(conf[a, b])
            row: dict = {
                "side": f"{CLASSES[a]}->{CLASSES[b]}",
                "gt_idx": a,
                "rendered_idx": b,
                "flips": n,
                "pct_of_all_flips": 100.0 * n / total_flips,
            }
            if n > 0:
                m = m_f[(gt_f == a) & (cx_f == b)]
                q = np.quantile(m, [0.5, 0.9])
                row.update(
                    mean_gt_margin=float(m.mean()),
                    median_gt_margin=float(q[0]),
                    p90_gt_margin=float(q[1]),
                    max_gt_margin=float(m.max()),
                    margin_mass=float(m.sum()),
                    shallow_share_lt_0p096=float((m < SHALLOW_BAND).mean()),
                    deep_share_ge_1=float((m >= 1.0).mean()),
                )
            sides.append(row)
    sides.sort(key=lambda r: -r["flips"])

    # Per unordered edge: count asymmetry vs PER-SITE margin-depth asymmetry.
    edges = []
    for a in range(5):
        for b in range(a + 1, 5):
            n_ab, n_ba = int(conf[a, b]), int(conf[b, a])
            n_tot = n_ab + n_ba
            if n_tot == 0:
                continue
            m_ab = m_f[(gt_f == a) & (cx_f == b)]
            m_ba = m_f[(gt_f == b) & (cx_f == a)]
            mean_ab = float(m_ab.mean()) if n_ab else None
            mean_ba = float(m_ba.mean()) if n_ba else None
            edge = {
                "edge": f"{CLASSES[a]}<->{CLASSES[b]}",
                "flips": n_tot,
                "pct_of_all_flips": 100.0 * n_tot / total_flips,
                f"{CLASSES[a]}->{CLASSES[b]}_flips": n_ab,
                f"{CLASSES[b]}->{CLASSES[a]}_flips": n_ba,
                "count_asymmetry": (max(n_ab, n_ba) / min(n_ab, n_ba)) if min(n_ab, n_ba) else None,
                "dominant_direction": f"{CLASSES[a]}->{CLASSES[b]}" if n_ab >= n_ba else f"{CLASSES[b]}->{CLASSES[a]}",
                f"mean_margin_{CLASSES[a]}->{CLASSES[b]}": mean_ab,
                f"mean_margin_{CLASSES[b]}->{CLASSES[a]}": mean_ba,
                # per-site depth asymmetry: how much deeper the flips run in one
                # direction than the other (mean GT margin ratio). d_seg prices
                # both directions at exactly 1 flip each.
                "per_site_depth_asymmetry": (
                    max(mean_ab, mean_ba) / min(mean_ab, mean_ba)
                    if mean_ab and mean_ba and min(mean_ab, mean_ba) > 0
                    else None
                ),
                "deeper_direction": (
                    (f"{CLASSES[a]}->{CLASSES[b]}" if mean_ab >= mean_ba else f"{CLASSES[b]}->{CLASSES[a]}")
                    if mean_ab is not None and mean_ba is not None
                    else None
                ),
            }
            edges.append(edge)
    edges.sort(key=lambda r: -r["flips"])

    git_head = subprocess.run(  # subprocess-no-check-OK: git-head provenance capture; empty-on-failure is visible in the receipt
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()

    out = {
        "schema": "ddm_cg1_directed_edge_margin.v1",
        "arm": "ddm_cg1r (task #809)",
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "git_head": git_head,
        "question": (
            "per DIRECTED side, how deep (GT self-margin) do the live vehicle's realized "
            "flips run? d_seg pays a uniform 1/N per flip on every side; the per-site "
            "repair depth measured here is what that uniform price ignores."
        ),
        "inputs": {
            "gt_argmax": str(CACHE / "gt_argmax_n600.npy"),
            "cx1_argmax": str(CACHE / "cx1_argmax_n600.npy"),
            "margins": str(GT_NPZ) + ":margins",
            "pu2_receipt": str(CACHE / "cx1_directed_flip_receipt.json"),
        },
        "denominator": {
            "pairs": 600,
            "sites": int(flip.size),
            "flips": total_flips,
            "directed_sides_possible": 20,
            "directed_sides_with_flips": sum(1 for s in sides if s["flips"] > 0),
        },
        "controls": {
            "d_seg_recomputed": d_seg,
            "d_seg_expected": D_SEG_EXPECTED,
            "confusion_matrix_matches_pu2_receipt_exactly": True,
            "gt_margin_negative_sites": neg,
            "directed_sides_sum_to_total_flips": True,
            "verdict": "ALL_CONTROLS_PASS",
        },
        "population_reference": {
            "mean_gt_margin_on_flips": float(m_f.mean()),
            "median_gt_margin_on_flips": float(np.quantile(m_f, 0.5)),
            "mg1_mean_on_flips_published": 0.3010,
        },
        "directed_sides": sides,
        "unordered_edges": edges,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    print(f"controls: {out['controls']['verdict']}")
    for e in edges[:5]:
        print(
            f"{e['edge']:24s} flips={e['flips']:7d} count_asym={e['count_asymmetry']:.2f} "
            f"dom={e['dominant_direction']:22s} depth_asym={e['per_site_depth_asymmetry']:.3f} "
            f"deeper={e['deeper_direction']}"
        )


if __name__ == "__main__":
    main()
