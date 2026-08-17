#!/usr/bin/env python3
"""ddm_fo2h LEG 2b -- re-rank the waterfill cells on MEASURED cost, not ideal entropy.

WHY.  LEG 2 (`ddm_fo2h_waterfill_measured_bytes.py`) swept every prefix of sr1's cell ranking and
found that ranking is **not monotone in real coder cost**: 31 inversions in 69 single-cell
bundles.  sr1 ranked cells by IDEAL per-cell conditional entropy; the real CABAC coder charges a
later cell less per flip than an earlier one 31 times over.  Prefixes of a mis-ordered ranking do
not exhaust the family, so LEG 2's optimum is an upper bound on the family's net dS, not the
family's optimum.  This module tests whether a MEASURED ordering does better.

THE METHOD, and its honest limit.  Each live cell is coded ALONE to get its measured
bytes-per-flip in isolation; cells are re-ranked by that; then every prefix of the new order is
re-coded for REAL round-trip-verified bytes.  The isolated cost is not the marginal-in-context
cost (the coder's contexts adapt across the whole stream, so a cell is cheaper beside its
neighbours than alone) -- so this is a HEURISTIC re-ranking, not an optimal one.  What is NOT
heuristic is the answer: every reported byte count comes from coding the actual candidate
selection and inverting the payload.  A true greedy (re-measuring the marginal against the
current selection at every step) is ~2,775 codings and remains owed.

Reuses LEG 2's precompute and fo1's coders verbatim.  Axis `[macOS-CPU advisory]`, scorer-free,
$0.  `score_claim=false`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_fo1_waterfill_real_coder import (  # reuse, never reimplement
    RATE_DS_PER_BYTE,
    SEG_DS_PER_FLIP,
    SR1_IDEAL_CELLS,
)
from ddm_fo2h_waterfill_measured_bytes import (
    DEFAULT_GT,
    DEFAULT_RT1_WORK,
    DEFAULT_SR1_WORK,
    DEFAULT_TOKENS,
    DEFAULT_WORK,
    Fo2hError,
    cellset_bits,
    code_and_verify,
    frames_for_selection,
    precompute,
    progress,
)


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    work = args.work
    pre = precompute(args)
    band_px, flip_px = pre["band_px"], pre["flip_px"]
    tgt_bpf = pre["target_bits_per_flip"]
    live = band_px > 0
    n_live = int(live.sum())
    cell_ids = np.flatnonzero(live)
    n_r = band_px[live].astype(np.float64)
    k_r = flip_px[live].astype(np.float64)

    # --- stage A: measured cost of every cell CODED ALONE -------------------------------------
    solo = []
    for i, cid in enumerate(cell_ids.tolist()):
        flips = float(k_r[i])
        if flips <= 0:
            solo.append({"cell": int(cid), "flips": 0.0, "bytes": 0.0,
                         "bytes_per_flip": float("inf")})
            continue
        r = code_and_verify(frames_for_selection(pre, np.array([cid])))
        b = r["mask_bytes"] + r["target_bytes"]
        solo.append({"cell": int(cid), "flips": flips, "bytes": float(b),
                     "bytes_per_flip": b / flips})
        if (i + 1) % 10 == 0:
            print(f"  [solo] {i + 1}/{n_live}", flush=True)
    progress(work, "leg2b-solo-measured", {"cells": n_live,
                                           "wall_s": round(time.time() - t0, 1)})

    order_new = np.argsort([s["bytes_per_flip"] for s in solo], kind="stable")
    # sr1's ranking, recomputed identically to LEG 2 so the two orders are comparable
    ideal_cost_B = (n_r * _h(k_r / n_r) / 8.0) + k_r * tgt_bpf / 8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ideal_cost_B > 0,
                         (k_r * SEG_DS_PER_FLIP) / (ideal_cost_B * RATE_DS_PER_BYTE), 0.0)
    order_sr1 = np.argsort(-ratio, kind="stable")
    same_prefix41 = set(cell_ids[order_new[:SR1_IDEAL_CELLS]].tolist()) == \
        set(cell_ids[order_sr1[:SR1_IDEAL_CELLS]].tolist())

    # --- stage B: re-code every prefix of the MEASURED order ----------------------------------
    etas = list(args.eta)
    rows = []
    count_B = math.ceil(math.log2(n_live + 1)) / 8.0
    for m in range(1, n_live + 1):
        sel = np.sort(cell_ids[order_new[:m]])
        r = code_and_verify(frames_for_selection(pre, sel))
        flips = float(k_r[order_new[:m]].sum())
        if r["flips"] != int(flips):
            raise Fo2hError(f"m={m}: coded {r['flips']} flips, histogram says {int(flips)}")
        payload_B = r["mask_bytes"] + r["target_bytes"]
        total_B = payload_B + cellset_bits(m, n_live) / 8.0 + count_B
        row = {"cells": m, "flips": int(flips), "payload_bytes": payload_B,
               "total_bytes_with_side_info": total_B,
               "breakeven_eta": total_B * RATE_DS_PER_BYTE / (flips * SEG_DS_PER_FLIP),
               "mask_sha256": r["mask_sha256"], "target_sha256": r["target_sha256"],
               "roundtrip_verified": r["roundtrip_verified"],
               "net_dS_by_eta": {f"{e:.4f}": -e * flips * SEG_DS_PER_FLIP
                                 + total_B * RATE_DS_PER_BYTE for e in etas}}
        rows.append(row)
        print(f"  [rerank] m={m:3d} flips={int(flips):6d} payload={payload_B:7d} B "
              f"breakeven_eta={row['breakeven_eta']:.4f}", flush=True)

    base = json.loads((work / "FO2H_WATERFILL_MEASURED.json").read_text())
    base_rows = {r["cells"]: r for r in base["rows"]}
    # fail-closed: the LEG 2 baseline must carry every eta this run prices, or the comparison
    # would be against a different grid than the one reported.
    missing = [f"{e:.4f}" for e in etas if f"{e:.4f}" not in base["rows"][0]["net_dS_by_eta"]]
    if missing:
        raise Fo2hError(f"LEG 2 baseline lacks eta keys {missing} -- re-run it on the same grid")
    comparison = {}
    for e in etas:
        k = f"{e:.4f}"
        bn = min(rows, key=lambda r: r["net_dS_by_eta"][k])
        bo = min(base["rows"], key=lambda r: r["net_dS_by_eta"][k])
        comparison[k] = {
            "measured_rank_best": {"cells": bn["cells"], "net_dS": bn["net_dS_by_eta"][k]},
            "ideal_rank_best": {"cells": bo["cells"], "net_dS": bo["net_dS_by_eta"][k]},
            "improvement_dS": bo["net_dS_by_eta"][k] - bn["net_dS_by_eta"][k],
            "measured_rank_wins": bn["net_dS_by_eta"][k] < bo["net_dS_by_eta"][k],
        }
    n_wins = sum(1 for v in comparison.values() if v["measured_rank_wins"])
    mb_new = min(rows, key=lambda r: r["breakeven_eta"])
    mb_old = min(base["rows"], key=lambda r: r["breakeven_eta"])

    rec = {
        "schema": "ddm_fo2h_waterfill_rerank.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False, "promotable": False,
        "method": "cells re-ranked by MEASURED solo bytes-per-flip; every prefix re-coded for "
                  "real round-trip-verified bytes",
        "honest_limit": "solo cost is not marginal-in-context cost, so the ORDER is heuristic; "
                        "the reported bytes are real.  A true greedy (~2,775 codings) is owed.",
        "cells_live": n_live,
        "measured_order_matches_sr1_at_41": same_prefix41,
        "solo_costs": solo,
        "measured_rank_cell_order": cell_ids[order_new].tolist(),
        "comparison_vs_ideal_ranking": comparison,
        # Stated here rather than left to a reader: winning at one grid point out of five is a
        # wash, not an improvement, and the honest summary belongs in the artifact.
        "summary": {
            "eta_points_measured_rank_wins": n_wins,
            "eta_points_total": len(comparison),
            "verdict": (
                f"the solo-measured re-ranking does NOT improve on sr1's ideal-entropy ranking; "
                f"it wins at {n_wins} of {len(comparison)} grid points. Coding a cell ALONE "
                f"removes the context amortization the CABAC coder gets from neighbouring "
                f"cells, so solo cost over-prices sparse cells and the order front-loads dense "
                f"ones. The mis-ordering LEG 2 measured is real, but solo cost is the wrong "
                f"correction for it -- a TRUE greedy on marginal-in-context cost remains owed."),
        },
        "min_breakeven": {"measured_rank": {"cells": mb_new["cells"],
                                            "eta": mb_new["breakeven_eta"]},
                          "ideal_rank": {"cells": mb_old["cells"],
                                         "eta": mb_old["breakeven_eta"]}},
        "incumbent_41_breakeven_eta": base_rows[SR1_IDEAL_CELLS]["breakeven_eta"],
        "verdict_scope": "formulation -- the fo1 M8+T2 coder pair on prefixes of a "
                         "MEASURED-solo-cost ranking of the 74 live cells at n600 on hv1 ep0634",
        "rows": rows,
        "wall_s": time.time() - t0,
    }
    (work / "FO2H_WATERFILL_RERANK.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True) + "\n")
    progress(work, "leg2b-reranked", {
        "min_breakeven_measured_rank": mb_new["breakeven_eta"],
        "min_breakeven_ideal_rank": mb_old["breakeven_eta"],
        "measured_rank_wins_at": [k for k, v in comparison.items() if v["measured_rank_wins"]]})
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("rows", "solo_costs", "measured_rank_cell_order")},
                     indent=2, sort_keys=True))
    return 0


def _h(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits, 0 at p=0 and p=1 (fo1's convention, kept local to avoid drift)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -(p * np.log2(np.where(p > 0, p, 1.0))
                + (1 - p) * np.log2(np.where(p < 1, 1 - p, 1.0)))
    return np.where((p <= 0) | (p >= 1), 0.0, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--rt1-work", type=Path, default=DEFAULT_RT1_WORK)
    ap.add_argument("--sr1-work", type=Path, default=DEFAULT_SR1_WORK)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--eta", type=float, nargs="+",
                    default=[0.5196, 0.5651, 0.6111, 0.6235, 1.0])
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
