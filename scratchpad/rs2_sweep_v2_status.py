#!/usr/bin/env python
"""ddm_rs2 — read the drive sweep's state from its RECEIPTS, never from the process table.

This is the production caller for `tac.optimization.sweep_durability`, and it is the
instrument that would have prevented three wrong calls in one session: `pgrep -f` said
ALIVE for minutes after a real death (it matched the watcher shells' own command lines),
and a lagging log tail said DEAD twice for a job that had already finished n600.

It also aggregates whatever groups ARE on disk, so a killed sweep still yields its
completed work -- the point of per-unit receipts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from tac.optimization.sweep_durability import job_state, resumable_units

SSD = Path("/Volumes/VertigoDataTier/pact")
OUT = SSD / "ddm_rs2_20260803" / "rs2_drive_sweep_v2"
GROUPS = OUT / "groups"
PITCH = 6
NCELL = 24 * 32


def main() -> int:
    # the sweep writes `g_dr_dc.npz`; normalise to the helper's `u_*` receipt convention
    for p in GROUPS.glob("g_*.npz"):
        alias = GROUPS / ("u_" + p.name[2:])
        if not alias.exists():
            alias.write_bytes(b"")
    units = [(dr, dc) for dr in range(PITCH) for dc in range(PITCH)]
    todo, done = resumable_units(GROUPS, units)
    st = job_state(GROUPS, expected_units=len(units))

    drive = np.zeros(NCELL)
    px = None
    rf = np.zeros((NCELL, 4))
    leaks, covered = [], 0
    for p in sorted(GROUPS.glob("g_*.npz")):
        z = np.load(p)
        ids = z["cell_ids"].astype(int)
        drive[ids] = z["drive_L1"]
        rf[ids] = z["rf"]
        px = z["px_over"] if px is None else px
        leaks.append(float(z["leak_L1"][0]))
        covered += len(ids)

    rep = {
        "axis": "[byte-closed, scorer-free]", "score_claim": False,
        "promotion_eligible": False,
        "job_state": st,
        "groups_done": len(done), "groups_todo": len(todo),
        "todo_units": [list(u) for u in todo],
        "cells_covered": covered,
        "leak_L1_range": [min(leaks), max(leaks)] if leaks else None,
        "total_drive_over_covered": float(drive.sum()),
    }
    if covered:
        h = rf[:, 1] - rf[:, 0] + 1
        w = rf[:, 3] - rf[:, 2] + 1
        m = (rf[:, 1] >= 0) & (drive > 0)
        rep["measured_receptive_field"] = {
            "cells": int(m.sum()),
            "rows_min": float(h[m].min()), "rows_median": float(np.median(h[m])),
            "rows_max": float(h[m].max()),
            "cols_min": float(w[m].min()), "cols_median": float(np.median(w[m])),
            "cols_max": float(w[m].max()),
            "area_median_px": float(np.median(h[m] * w[m])),
            "support_ratio_median_vs_16x16_tile": float(np.median(h[m] * w[m])) / 256.0,
            "pilot_single_cell_anchor_bbox": "84 x 82",
        }
        d = drive[m]
        rep["drive_over_covered_cells"] = {
            "min": float(d.min()), "median": float(np.median(d)),
            "max": float(d.max()), "spread": float(d.max() / d.min()),
            "n_zero": int((drive[drive >= 0] == 0).sum() - (NCELL - covered)),
        }
    (OUT / "status.json").write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    np.save(OUT / "cell_drive_partial.npy", drive)
    print(json.dumps(rep, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
