# SPDX-License-Identifier: MIT
"""ddm_ma1 - the conditional-entropy FLOOR of the within-miss sector.

``ddm_fx1`` priced this sector at a **1,247.19 B ceiling**, which is the
perfect-model bound: it assumes a model that names the miss class at zero cost.
That is the right number for "how big is the sector" and the WRONG number for
"how much can be taken out of it", and the gap between the two is the whole
question a follow-on arm needs answered before it spends a day here.

This is ``ddm_fx2``'s R6 argument applied to the sector: for a candidate context,
compute the KT-smoothed conditional entropy of the miss class, which lower-bounds
EVERY model that sees only that context -- count-ratio, mixing, or anything else.
Reported in-sample (hindsight-optimal, so genuinely a bound) and held-out (fit on
frames [0,300), scored on [300,600)), because ``ddm_fx2`` §6 measured this field
to be strongly non-stationary and a static fit is not what an online coder gets.

The measured online model is quoted alongside, so the three numbers -- online
achieved, hindsight floor, perfect-model ceiling -- can be read together instead
of the ceiling being mistaken for available headroom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ma1_race_within_miss import (  # noqa: E402
    CELLS,
    KT_ALPHA,
    NUM_CLASSES,
    WITHIN_MISS_CEILING_BYTES,
)

SECTOR_DEFAULT = "/Volumes/APDataStore/pact/ddm_ma1/retained/miss_sector_n600.npz"


def _code_bytes(cell: np.ndarray, token: np.ndarray, n_cells: int,
                fit: np.ndarray, score: np.ndarray) -> float:
    """KT-smoothed static per-cell law fitted on ``fit``, scored on ``score``."""
    counts = np.zeros((n_cells, NUM_CLASSES), dtype=np.float64)
    np.add.at(counts, (cell[fit], token[fit]), 1.0)
    counts += KT_ALPHA
    probability = counts / counts.sum(axis=1, keepdims=True)
    chosen = np.maximum(probability[cell[score], token[score]], 1e-300)
    return float(-np.log2(chosen).sum()) / 8.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sector", default=SECTOR_DEFAULT)
    ap.add_argument("--out", default="/Volumes/APDataStore/pact/ddm_ma1/probe/sector_floor.json")
    args = ap.parse_args()

    with np.load(args.sector) as z:
        d = {k: z[k] for k in z.files}
    token = d["token"].astype(np.int64)
    frame = d["frame"].astype(np.int64)
    n = token.size
    everything = np.arange(n)
    first_half = np.flatnonzero(frame < 300)
    second_half = np.flatnonzero(frame >= 300)

    names = [
        "none", "arg", "arg_bnd", "arg_ubin8", "arg_nbup", "arg_nbmode",
        "arg_nbmode_prev1", "nb3", "arg_nb3", "nb3_prev1", "arg_nb3_prev1",
        "nb4", "arg_nb4_prev1",
    ]
    rows = []
    for name in names:
        n_cells, rule = CELLS[name]
        cell = rule(d)
        in_sample = _code_bytes(cell, token, n_cells, everything, everything)
        held_out = _code_bytes(cell, token, n_cells, first_half, second_half)
        used = int(np.unique(cell).size)
        rows.append({
            "context": name,
            "cells": n_cells,
            "cells_used": used,
            "records_per_used_cell": n / max(used, 1),
            "in_sample_floor_bytes": in_sample,
            "held_out_bytes_second_half": held_out,
        })
        print(f"{name:20s} cells {n_cells:6d} used {used:6d}  "
              f"in-sample {in_sample:9.2f} B   held-out(2nd half) {held_out:9.2f} B")

    # The second half alone under the shipped prior, so held-out is comparable.
    row64 = d["row64"]
    one_minus = d["one_minus"]
    rel = np.maximum(row64[np.arange(n), token] / one_minus, 1e-300)
    prior_bytes_all = float(-np.log2(rel).sum()) / 8.0
    prior_bytes_2nd = float(-np.log2(rel[second_half]).sum()) / 8.0

    # The bound that actually governs THIS arm's model class.  The categorical
    # floors above replace the prior and are 4.5x worse than it, because the
    # prior is per-position and continuous while a cell table is neither.  The
    # live model MULTIPLIES the prior instead, so its bound is the best per-cell
    # multiplier fitted in hindsight on the whole sector -- an upper bound on any
    # online version of the same class.
    arg = d["arg"].astype(np.int64)
    other = np.ones((n, NUM_CLASSES), dtype=bool)
    other[np.arange(n), arg] = False
    rel_prior = np.where(other, row64 / one_minus[:, None], 0.0)
    mult_rows = []
    for name in ("arg", "arg_nbup", "arg_nbmode_prev1", "arg_nb3", "nb3_prev1",
                 "arg_nb3_prev1", "arg_nb4_prev1"):
        n_cells, rule = CELLS[name]
        cell = rule(d)
        counts = np.zeros((n_cells, NUM_CLASSES))
        expect = np.zeros((n_cells, NUM_CLASSES))
        np.add.at(counts, (cell, token), 1.0)
        np.add.at(expect, (cell[:, None], np.arange(NUM_CLASSES)[None, :]), rel_prior)
        m = (counts + KT_ALPHA) / (expect + KT_ALPHA)
        np.clip(m, 1.0 / 16.0, 16.0, out=m)
        w = np.where(other, rel_prior * m[cell], 0.0)
        big_w = w.sum(axis=1)
        big_s = rel_prior.sum(axis=1)
        coded = np.maximum(w[np.arange(n), token] * (big_s / np.maximum(big_w, 1e-300)), 1e-300)
        got = float(-np.log2(coded).sum()) / 8.0
        mult_rows.append({
            "context": name,
            "hindsight_multiplicative_bytes": got,
            "hindsight_gain_vs_prior_bytes": got - prior_bytes_all,
        })
        print(f"  hindsight multiplicative {name:18s} {got:9.2f} B   "
              f"gain {got - prior_bytes_all:+8.2f} B")

    out = {
        "sector": args.sector,
        "records": int(n),
        "perfect_model_ceiling_bytes": WITHIN_MISS_CEILING_BYTES,
        "shipped_prior_bytes_all": prior_bytes_all,
        "shipped_prior_bytes_second_half": prior_bytes_2nd,
        "rows": rows,
        "hindsight_multiplicative_rows": mult_rows,
        "online_measured_gain_bytes": -104.584,
    }
    best = min(rows, key=lambda r: r["in_sample_floor_bytes"])
    out["best_in_sample"] = best
    print()
    print(f"shipped prior over the whole sector : {prior_bytes_all:9.2f} B "
          f"(= the ddm_fx1 ceiling, {WITHIN_MISS_CEILING_BYTES:.2f} B)")
    print(f"best hindsight floor ({best['context']}) : "
          f"{best['in_sample_floor_bytes']:9.2f} B")
    print(f"  => hindsight-reachable at best      : "
          f"{prior_bytes_all - best['in_sample_floor_bytes']:9.2f} B")
    print(f"shipped prior, 2nd half only        : {prior_bytes_2nd:9.2f} B")
    for r in rows:
        r["held_out_delta_vs_prior_bytes"] = r["held_out_bytes_second_half"] - prior_bytes_2nd

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
