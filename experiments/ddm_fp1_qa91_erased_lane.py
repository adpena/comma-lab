# SPDX-License-Identifier: MIT
"""ddm_fp1 P5 (folded) — QA91 erased-lane-component analyzer ($0, NO scorer).

vh1 row 12: size distribution of the erased Lane (class-1) GT components at the burn
endpoint vs the #315 critical-nucleus (pi_1 >~ 5) law + the birth-curve trend, from
EXISTING telemetry custody only (no new SegNet pass).  Verdict: sub-nucleus / GT-flicker
mass (un-recoverable, #141-class) vs super-nucleus recoverable mass (-> birth-plateau
event key).  RELATIVE SIGNIFICANCE: Lane holds ~26% of endpoint d_seg; state the
recoverable fraction's dS explicitly.

Custody (reads only): gt_n600 lstars + ddm_bc1 burn_out/telemetry.jsonl topology rows.
Pointer 0.1910828242 [contest-CPU] UNMOVED; [macOS-CPU advisory]; no score claim.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage

GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
BURN_TEL = "/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/telemetry.jsonl"
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/qa91_erased_lane.json")
LANE = 1
NUCLEUS_PX = 5  # #315 pi_1 >~ 5 critical-nucleus threshold (sub-nucleus <= 5px)
NUM_PAIRS = 600


def main() -> int:
    z = zipfile.ZipFile(GT_CACHE)
    with z.open("lstars.npy") as f:
        lstars = np.lib.format.read_array(f)[:NUM_PAIRS]  # (600,384,512) int64

    # --- GT Lane component size distribution (aggregate over n600) ---
    sizes: list[int] = []
    per_pair_counts = []
    struct = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    for i in range(NUM_PAIRS):
        lab, n = ndimage.label(lstars[i] == LANE, structure=struct)
        if n:
            s = np.bincount(lab.ravel())[1:]  # skip background
            sizes.extend(int(x) for x in s)
        per_pair_counts.append(int(n))
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    total_gt_comp = int(sizes_arr.size)
    lane_px = int((lstars == LANE).sum())

    sub = sizes_arr <= NUCLEUS_PX
    sub_count = int(sub.sum())
    sup_count = int((~sub).sum())
    sub_area = int(sizes_arr[sub].sum())
    sup_area = int(sizes_arr[~sub].sum())

    # --- burn endpoint topology + birth-curve trend ---
    rows = [json.loads(l) for l in open(BURN_TEL)]
    topo = [(r["epoch"], r["topology_per_class"]) for r in rows if "topology_per_class" in r]
    topo.sort()
    ep0, t0 = topo[0]
    epN, tN = topo[-1]
    betti_gt_lane = tN["betti0_gt"][LANE]
    betti_real_start = t0["betti0_realized"][LANE]
    betti_real_end = tN["betti0_realized"][LANE]
    erased_end = tN["gt_components_erased"][LANE]
    smallest_surviving = tN["smallest_surviving_gt_component_px"][LANE]
    birth_trend = [(e, t["betti0_realized"][LANE]) for e, t in topo]
    # birth-plateau: is betti0_realized[Lane] still rising at the end? slope over last 5 gates
    tail = [v for _, v in birth_trend[-5:]]
    tail_slope = float((tail[-1] - tail[0]) / max(1, len(tail) - 1)) if len(tail) >= 2 else 0.0

    # --- relative significance ---
    # endpoint per-class d_seg Lane (QA24/burn endpoint, vh1 row 11): 0.001377 (26% of 0.005277)
    lane_dseg_endpoint = 0.001377
    lane_seg_S = 100.0 * lane_dseg_endpoint  # S-units
    # Recoverable-fraction proxy: erased components are the flip carriers; the super-nucleus
    # AREA fraction of GT lane components bounds the recoverable (non-GT-flicker) portion.
    frac_area_sub = sub_area / max(1, lane_px)
    frac_area_sup = sup_area / max(1, lane_px)
    # dS bound of recoverable Lane mass = lane_seg_S * (super-nucleus area fraction of erased mass).
    # Under the size-agnostic upper bound (all erased mass recoverable) vs the #315 lower bound
    # (only super-nucleus recoverable): report BOTH.
    dS_lane_total = lane_seg_S
    dS_recoverable_upper = lane_seg_S  # if all erased Lane flips were recoverable
    dS_recoverable_super = lane_seg_S * frac_area_sup  # #315: only super-nucleus

    # --- verdict ---
    if frac_area_sub >= 0.80:
        verdict = ("CORPUS_CLOSES_GT_FLICKER: >=80% of GT Lane AREA is sub-nucleus (<=5px); "
                   "erased mass is dominated by GT-noise-floor flicker (#141-class, un-recoverable). "
                   "birth-lever corpus (#323/#315) stays non-reactivated.")
    elif frac_area_sup >= 0.20:
        verdict = ("BIRTH_PLATEAU_KEY_CANDIDATE: material super-nucleus (>5px) erased Lane mass "
                   "exists; a birth-plateau event key (lane betti0_realized slope->0 while "
                   "above-nucleus erasure persists) is warranted in the BR-A/BR-C schedule.")
    else:
        verdict = ("MIXED: super-nucleus area fraction in (0,0.20); recoverable Lane dS is small "
                   "but nonzero; birth key OPTIONAL, gated on cost.")

    receipt = {
        "schema": "ddm_fp1_qa91_erased_lane.v1",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False, "scorer_used": False,
        "gt_lane_components_total_n600": total_gt_comp,
        "gt_lane_pixels_total": lane_px,
        "component_size_stats_px": {
            "min": int(sizes_arr.min()), "p50": int(np.median(sizes_arr)),
            "p90": int(np.percentile(sizes_arr, 90)), "max": int(sizes_arr.max()),
            "mean": round(float(sizes_arr.mean()), 2),
        },
        "nucleus_threshold_px": NUCLEUS_PX,
        "sub_nucleus_count": sub_count, "super_nucleus_count": sup_count,
        "sub_nucleus_count_frac": round(sub_count / max(1, total_gt_comp), 4),
        "sub_nucleus_area_frac": round(frac_area_sub, 4),
        "super_nucleus_area_frac": round(frac_area_sup, 4),
        "burn_endpoint_topology": {
            "betti0_gt_lane": betti_gt_lane,
            "betti0_realized_lane_start_ep{}".format(ep0): betti_real_start,
            "betti0_realized_lane_end_ep{}".format(epN): betti_real_end,
            "gt_components_erased_lane": erased_end,
            "smallest_surviving_gt_component_px": smallest_surviving,
        },
        "birth_curve_trend_lane": [[int(e), int(v)] for e, v in birth_trend],
        "birth_tail_slope_comp_per_gate": round(tail_slope, 3),
        "relative_significance": {
            "lane_dseg_endpoint": lane_dseg_endpoint,
            "lane_seg_S_units": round(lane_seg_S, 5),
            "lane_frac_of_endpoint_dseg": 0.26,
            "dS_lane_total": round(dS_lane_total, 5),
            "dS_recoverable_upper_all_erased": round(dS_recoverable_upper, 5),
            "dS_recoverable_super_nucleus_only": round(dS_recoverable_super, 5),
        },
        "verdict": verdict,
        "method_note": ("GT Lane component sizes are EXACT (scipy 8-conn label on n600 GT). "
                        "Per-component ERASURE labels would need the endpoint realized argmax "
                        "(one scorer pass, NOT run here per the $0 fold); the super-nucleus AREA "
                        "fraction bounds recoverable mass under the #315 nucleus law."),
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print(json.dumps({k: receipt[k] for k in (
        "gt_lane_components_total_n600", "component_size_stats_px", "sub_nucleus_count_frac",
        "sub_nucleus_area_frac", "super_nucleus_area_frac", "relative_significance",
        "verdict")}, indent=1))
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
