# no-argparse-OK: single optional positional (chunk dir, pinned default); no flags to discover
"""Aggregate ddm_fd1 S0 box-solve probe chunks into the n600 band receipt.

Read-only aggregation of `chunk_boxsolve_*.json` (schema ddm_rp1_rangeA_cell_probe_chunk.v2).
[macOS-CPU frozen-scorer advisory]; score_claim=false; pointer 0.1910828242 UNMOVED.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

N_SITES = 384 * 512


def main() -> int:
    chunk_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/chunks"
    rows = []
    files = sorted(glob.glob(f"{chunk_dir}/chunk_boxsolve_*.json"))
    for path in files:
        with open(path) as handle:
            rows.extend(json.load(handle)["per_pair"])
    n = len(rows)
    c0 = np.array([p["c0_dseg_vs_lstars"] for p in rows])
    c1 = np.array([p["c1a_dseg"] for p in rows])
    hold = np.array([p["c1a_cell_hold_flips_vs_c0"] for p in rows]) / N_SITES
    dp0 = np.array([p["c0_dpose"] for p in rows])
    dp1 = np.array([p["c1a_dpose"] for p in rows])
    pre_f = np.array([p["pre_margin_flipped_mean"] for p in rows if p["pre_margin_flipped_mean"] is not None])
    pre_h = np.array([p["pre_margin_held_mean"] for p in rows if p["pre_margin_held_mean"] is not None])
    per_class = {str(c): int(sum(p["per_class_gt_flips"][str(c)] for p in rows)) for c in range(5)}
    out = {
        "schema": "ddm_fd1_s0_boxsolve_band.v1",
        "pairs": n,
        "chunks": len(files),
        "c0_dseg_vs_lstars_mean": float(c0.mean()),
        "c1_dseg_vs_lstars_mean": float(c1.mean()),
        "c1_over_c0_ratio": float(c1.mean() / c0.mean()),
        "c1_cell_hold_flip_rate_mean": float(hold.mean()),
        "c1_cell_hold_flip_rate_p90": float(np.percentile(hold, 90)),
        "c1_cell_hold_flip_rate_max": float(hold.max()),
        "c0_dpose_mean": float(dp0.mean()),
        "c1_dpose_mean": float(dp1.mean()),
        "pre_margin_flipped_mean": float(pre_f.mean()),
        "pre_margin_held_mean": float(pre_h.mean()),
        "margin_gap_ratio": float(pre_h.mean() / pre_f.mean()),
        "per_class_c1_flips_vs_lstars": per_class,
        "gt_substrate_reference": {"c1_dseg": 3.6296e-4, "cell_hold_equals_lstars_flips": True},
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 UNMOVED",
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
