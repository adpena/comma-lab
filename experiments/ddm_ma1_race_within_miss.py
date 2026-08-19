# SPDX-License-Identifier: MIT
"""ddm_ma1 - race probability models for the WITHIN-MISS relative law.

THE TARGET.  ``ddm_rr4``'s transport is rank-one: ``coding_row`` scales every
non-argmax column by the single scalar ``(1 - q) / one_minus``, so the relative
law inside the miss sector is exactly the neural prior's ``r_k = p_k / (1 - p_max)``
and is never adapted.  ``ddm_fx1`` priced that sector at **1,247.19 B** (the
perfect-model ceiling) and left it unbuilt; ``ddm_fx2`` re-confirmed it and named
it the largest remaining priced target on the token stream.

THE MODEL.  A multiplicative reweight of the miss-sector relative law, learned by
the SAME Krichevsky-Trofimov count-ratio ``ddm_rr4`` uses for the hit event,
lifted from the hit axis to the class axis::

    M[c, k] = (n[c, k] + KT) / (e[c, k] + KT)          observed / prior-expected
    w_k     = row64[k] * M[c, k]      (k != arg)
    W = sum_k w_k       S = sum_k row64[k]
    coded_k = w_k * (S / W) * (1 - q) / one_minus

``n`` counts observed miss classes; ``e`` accumulates the prior's own relative
mass ``r_k`` over the same population, in FIXED POINT so the sum is
order-independent (``ddm_rr4``'s ``phat_q`` discipline).  Both are folded in
``observe``, which the receiver reaches only after decoding, so the model is
causal.  The ``S / W`` factor preserves the sector's total mass, which is what
makes the identity control exact.

THE IDENTITY CONTROL, and it is exact by construction: with ``M == 1`` everywhere
``w_k == row64[k]`` bitwise, so ``W == S`` bitwise, ``S / W == 1.0`` exactly, and
the coded row is the shipped row.  A cold cell (below ``min_count``) emits
``M = 1`` and therefore ships HPAC's own relative law, exactly as ``ddm_rr4``'s
cold contexts emit exactly HPAC.

WHY THE SECTOR IS A SUFFICIENT SURFACE.  Verified at source: ``coding_row`` uses
one scalar for all non-argmax columns and ``observe`` folds only
``hit = decoded == arg``.  So a within-miss model perturbs neither the hit-event
code length nor the hit-event model's trajectory, and ``delta(total) ==
delta(within-miss)`` EXACTLY.  This race therefore reports the same number a full
117,964,800-position replay would, at 1/500th the cost -- and the full replay is
still run as the confirming control before any candidate is proposed.

NO INHERITED CONSTANTS.  ``ddm_rr4``'s ``MIN_COUNT = 32`` and its ``[2^-4, 2^4]``
odds clamp were derived for a 51,200-cell context over 117.9M positions.  This
sector has ~200 cells over 223,694 records -- a different regime, so both are
SWEPT here rather than carried (``cross_regime_constant_transfer`` genus).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NUM_CLASSES = 5
UNKNOWN = 5
KT_ALPHA = 0.5
PHAT_SCALE = 1 << 30
SECTOR_DEFAULT = "/Volumes/APDataStore/pact/ddm_ma1/retained/miss_sector_n600.npz"

# ddm_fx1 §5 / re-measured by ddm_ma1's extractor at delta -0.00218 B.
WITHIN_MISS_CEILING_BYTES = 1247.1878241883985


# --------------------------------------------------------------------------
# Cell definitions.  Every feature is causal and free at the receiver: it is
# either an input plane it already holds (boundary), a function of the prior it
# already computes (arg, ubin), or an already-decoded token (neighbours, prev).
# --------------------------------------------------------------------------

def _ubin8(d: dict) -> np.ndarray:
    return (d["ubin"].astype(np.int64) // 8).clip(0, 7)


def _nbmode(d: dict) -> np.ndarray:
    """Modal class among the four causal neighbours; UNKNOWN if none decoded."""
    stack = np.stack(
        [d[f"nb_{n}"].astype(np.int64) for n in ("up", "upright", "left", "upleft")], axis=1
    )
    counts = np.zeros((stack.shape[0], NUM_CLASSES + 1), dtype=np.int16)
    for c in range(NUM_CLASSES + 1):
        counts[:, c] = (stack == c).sum(axis=1)
    counts[:, UNKNOWN] = 0  # never let "unknown" win the vote
    best = counts.argmax(axis=1)
    return np.where(counts.max(axis=1) == 0, UNKNOWN, best).astype(np.int64)


CELLS: dict[str, tuple[int, object]] = {
    "arg": (5, lambda d: d["arg"].astype(np.int64)),
    "arg_bnd": (25, lambda d: d["arg"].astype(np.int64) * 5 + d["boundary"].astype(np.int64)),
    "arg_ubin8": (40, lambda d: d["arg"].astype(np.int64) * 8 + _ubin8(d)),
    "arg_bnd_ubin8": (
        200,
        lambda d: (d["arg"].astype(np.int64) * 5 + d["boundary"].astype(np.int64)) * 8 + _ubin8(d),
    ),
    "arg_nbup": (30, lambda d: d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)),
    "arg_nbmode": (30, lambda d: d["arg"].astype(np.int64) * 6 + _nbmode(d)),
    "arg_nbmode_bnd": (
        150,
        lambda d: (d["arg"].astype(np.int64) * 6 + _nbmode(d)) * 5 + d["boundary"].astype(np.int64),
    ),
    "arg_prev1": (30, lambda d: d["arg"].astype(np.int64) * 6 + d["prev1"].astype(np.int64)),
    "arg_nbmode_prev1": (
        180,
        lambda d: (d["arg"].astype(np.int64) * 6 + _nbmode(d)) * 6 + d["prev1"].astype(np.int64),
    ),
    "arg_agree": (20, lambda d: (d["arg"].astype(np.int64) * 2 + d["agree1"].astype(np.int64)) * 2
                  + d["agree2"].astype(np.int64)),
    "arg_run": (40, lambda d: d["arg"].astype(np.int64) * 8 + d["run"].astype(np.int64)),
    "none": (1, lambda d: np.zeros(d["arg"].size, dtype=np.int64)),
    # --- richer cells.  An ONLINE model pays its own learning cost, so an
    # over-rich context shows up directly as a worse measured delta: the code
    # length IS the held-out check (ddm_fx2 §6 makes the same argument).
    "arg_nbup_prev1": (
        180,
        lambda d: (d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
        + d["prev1"].astype(np.int64),
    ),
    "arg_nbup_nbleft": (
        180,
        lambda d: (d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
        + d["nb_left"].astype(np.int64),
    ),
    "arg_nbmode_prev1_bnd": (
        900,
        lambda d: ((d["arg"].astype(np.int64) * 6 + _nbmode(d)) * 6
                   + d["prev1"].astype(np.int64)) * 5 + d["boundary"].astype(np.int64),
    ),
    "arg_nbmode_prev1_ubin8": (
        1440,
        lambda d: ((d["arg"].astype(np.int64) * 6 + _nbmode(d)) * 6
                   + d["prev1"].astype(np.int64)) * 8 + _ubin8(d),
    ),
    "arg_nbup_nbleft_prev1": (
        1080,
        lambda d: ((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                   + d["nb_left"].astype(np.int64)) * 6 + d["prev1"].astype(np.int64),
    ),
    "arg_nbup_nbleft_nbupright": (
        1080,
        lambda d: ((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                   + d["nb_left"].astype(np.int64)) * 6 + d["nb_upright"].astype(np.int64),
    ),
    "arg_nbmode_nbup_prev1": (
        1080,
        lambda d: ((d["arg"].astype(np.int64) * 6 + _nbmode(d)) * 6
                   + d["nb_up"].astype(np.int64)) * 6 + d["prev1"].astype(np.int64),
    ),
    "arg_nbup_prev1_prev2": (
        1080,
        lambda d: ((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                   + d["prev1"].astype(np.int64)) * 6 + d["prev2"].astype(np.int64),
    ),
    # --- the full causal template ddm_fx2 R1 established (up, up-right, left,
    # up-left), with and without the argmax and the temporal leg.
    "nb4": (
        1296,
        lambda d: ((d["nb_up"].astype(np.int64) * 6 + d["nb_upright"].astype(np.int64)) * 6
                   + d["nb_left"].astype(np.int64)) * 6 + d["nb_upleft"].astype(np.int64),
    ),
    "arg_nb4": (
        6480,
        lambda d: (((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                    + d["nb_upright"].astype(np.int64)) * 6
                   + d["nb_left"].astype(np.int64)) * 6 + d["nb_upleft"].astype(np.int64),
    ),
    "nb4_prev1": (
        7776,
        lambda d: (((d["nb_up"].astype(np.int64) * 6 + d["nb_upright"].astype(np.int64)) * 6
                    + d["nb_left"].astype(np.int64)) * 6 + d["nb_upleft"].astype(np.int64)) * 6
        + d["prev1"].astype(np.int64),
    ),
    "arg_nb3_prev1": (
        6480,
        lambda d: (((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                    + d["nb_upright"].astype(np.int64)) * 6
                   + d["nb_left"].astype(np.int64)) * 6 + d["prev1"].astype(np.int64),
    ),
    "nb3": (
        216,
        lambda d: (d["nb_up"].astype(np.int64) * 6 + d["nb_upright"].astype(np.int64)) * 6
        + d["nb_left"].astype(np.int64),
    ),
    "arg_nb3": (
        1080,
        lambda d: ((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                   + d["nb_upright"].astype(np.int64)) * 6 + d["nb_left"].astype(np.int64),
    ),
    "nb3_prev1": (
        1296,
        lambda d: ((d["nb_up"].astype(np.int64) * 6 + d["nb_upright"].astype(np.int64)) * 6
                   + d["nb_left"].astype(np.int64)) * 6 + d["prev1"].astype(np.int64),
    ),
    "arg_nb4_prev1": (
        38880,
        lambda d: ((((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                     + d["nb_upright"].astype(np.int64)) * 6
                    + d["nb_left"].astype(np.int64)) * 6
                   + d["nb_upleft"].astype(np.int64)) * 6 + d["prev1"].astype(np.int64),
    ),
    "arg_nb3_prev1_bnd": (
        32400,
        lambda d: ((((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                     + d["nb_upright"].astype(np.int64)) * 6
                    + d["nb_left"].astype(np.int64)) * 6
                   + d["prev1"].astype(np.int64)) * 5 + d["boundary"].astype(np.int64),
    ),
    "arg_nb3_prev1_prev2": (
        38880,
        lambda d: ((((d["arg"].astype(np.int64) * 6 + d["nb_up"].astype(np.int64)) * 6
                     + d["nb_upright"].astype(np.int64)) * 6
                    + d["nb_left"].astype(np.int64)) * 6
                   + d["prev1"].astype(np.int64)) * 6 + d["prev2"].astype(np.int64),
    ),
}


# --------------------------------------------------------------------------
# Relational features: per-(record, class) indices.  These express "is class k
# present in my already-decoded neighbourhood?" -- structure the cell axis
# cannot reach, because it must be keyed by the CANDIDATE class k.
# --------------------------------------------------------------------------

def _rel_none(d: dict) -> tuple[int, np.ndarray]:
    n = d["arg"].size
    return 1, np.zeros((n, NUM_CLASSES), dtype=np.int64)


def _nb_count(d: dict) -> np.ndarray:
    """count[i, k] = how many of the four causal neighbours decoded to class k."""
    n = d["arg"].size
    out = np.zeros((n, NUM_CLASSES), dtype=np.int64)
    for name in ("up", "upright", "left", "upleft"):
        v = d[f"nb_{name}"].astype(np.int64)
        known = v < NUM_CLASSES
        np.add.at(out, (np.flatnonzero(known), v[known]), 1)
    return out


def _rel_nbcount(d: dict) -> tuple[int, np.ndarray]:
    return 5, _nb_count(d).clip(0, 4)


def _rel_nbany(d: dict) -> tuple[int, np.ndarray]:
    return 2, (_nb_count(d) > 0).astype(np.int64)


def _rel_nbany_arg(d: dict) -> tuple[int, np.ndarray]:
    present = (_nb_count(d) > 0).astype(np.int64)
    return 10, d["arg"].astype(np.int64)[:, None] * 2 + present


def _rel_nbcount_prev(d: dict) -> tuple[int, np.ndarray]:
    present = _nb_count(d).clip(0, 4)
    prev = d["prev1"].astype(np.int64)[:, None] == np.arange(NUM_CLASSES)[None, :]
    return 10, present * 2 + prev.astype(np.int64)


def _rel_prev(d: dict) -> tuple[int, np.ndarray]:
    prev = d["prev1"].astype(np.int64)[:, None] == np.arange(NUM_CLASSES)[None, :]
    return 2, prev.astype(np.int64)


RELATIONAL: dict[str, object] = {
    "none": _rel_none,
    "nbany": _rel_nbany,
    "nbcount": _rel_nbcount,
    "nbany_arg": _rel_nbany_arg,
    "prev": _rel_prev,
    "nbcount_prev": _rel_nbcount_prev,
}


def run_model(
    d: dict,
    *,
    cell_name: str,
    rel_name: str,
    min_count: int,
    clamp_low: float,
    clamp_high: float,
) -> dict:
    """Online, group-batched replay of the within-miss model over the sector."""
    n_cells, cell_rule = CELLS[cell_name]
    cell = cell_rule(d)
    n_rel, rel_index = RELATIONAL[rel_name](d)

    row = d["row64"]
    one_minus = d["one_minus"]
    arg = d["arg"].astype(np.int64)
    token = d["token"].astype(np.int64)
    n = arg.size

    other = np.ones((n, NUM_CLASSES), dtype=bool)
    other[np.arange(n), arg] = False
    rel_prior = np.where(other, row / one_minus[:, None], 0.0)
    rel_q = np.rint(rel_prior * PHAT_SCALE).astype(np.int64)

    # Both tables learn the SAME ratio -- observed count over prior-expected mass.
    # The cell table is keyed by (context, class); the relational table by a
    # per-(record, class) value, so it can express "class k is present in my
    # decoded neighbourhood" -- structure no cell keying can reach.
    counts_cell = np.zeros((n_cells, NUM_CLASSES), dtype=np.int64)
    expect_cell = np.zeros((n_cells, NUM_CLASSES), dtype=np.int64)
    seen_cell = np.zeros(n_cells, dtype=np.int64)
    counts_rel = np.zeros(n_rel, dtype=np.int64)
    expect_rel = np.zeros(n_rel, dtype=np.int64)
    seen_rel = np.zeros(n_rel, dtype=np.int64)

    # Group-batched: the shipped driver folds a whole causal group at once, so a
    # record may not see its own group-mates' statistics.
    key = d["frame"].astype(np.int64) * 4096 + d["group"].astype(np.int64)
    edges = np.flatnonzero(np.diff(key)) + 1
    blocks = np.split(np.arange(n), edges)

    lanes = np.arange(NUM_CLASSES)[None, :]
    delta_bits = 0.0
    applied = 0
    started = time.time()

    for block in blocks:
        c = cell[block]
        r = rel_index[block]
        rp = rel_prior[block]
        oth = other[block]
        tok = token[block]
        idx = np.arange(block.size)

        m = np.ones((block.size, NUM_CLASSES), dtype=np.float64)
        warm_c = seen_cell[c] >= min_count
        if warm_c.any():
            ratio = (counts_cell[c] + KT_ALPHA) / (expect_cell[c] / PHAT_SCALE + KT_ALPHA)
            m = np.where(warm_c[:, None], ratio, m)
        if n_rel > 1:
            warm_r = seen_rel[r] >= min_count
            if warm_r.any():
                ratio = (counts_rel[r] + KT_ALPHA) / (expect_rel[r] / PHAT_SCALE + KT_ALPHA)
                m = m * np.where(warm_r, ratio, 1.0)
        np.clip(m, clamp_low, clamp_high, out=m)
        m = np.where(oth, m, 1.0)

        active = np.any(m != 1.0, axis=1)
        if active.any():
            w = np.where(oth, rp * m, 0.0)
            big_w = w.sum(axis=1)
            big_s = np.where(oth, rp, 0.0).sum(axis=1)
            base = np.maximum(rp[idx, tok], 1e-300)
            corrected = np.maximum(w[idx, tok] * (big_s / np.maximum(big_w, 1e-300)), 1e-300)
            per_record = np.where(active, -np.log2(corrected) + np.log2(base), 0.0)
            delta_bits += float(per_record.sum())
            applied += int(active.sum())

        np.add.at(counts_cell, (c, tok), 1)
        np.add.at(expect_cell, (c[:, None], lanes), rel_q[block])
        np.add.at(seen_cell, c, 1)
        if n_rel > 1:
            np.add.at(counts_rel, r[idx, tok], 1)
            np.add.at(expect_rel, r, np.where(oth, rel_q[block], 0))
            np.add.at(seen_rel, r[oth], 1)

    return {
        "cell": cell_name,
        "cells": n_cells,
        "relational": rel_name,
        "rel_values": n_rel,
        "min_count": min_count,
        "clamp_low": clamp_low,
        "clamp_high": clamp_high,
        "delta_bytes": delta_bits / 8.0,
        "records": int(n),
        "records_corrected": applied,
        "sector_bytes_before": WITHIN_MISS_CEILING_BYTES,
        "sector_bytes_after": WITHIN_MISS_CEILING_BYTES + delta_bits / 8.0,
        "fraction_of_ceiling_pct": -100.0 * (delta_bits / 8.0) / WITHIN_MISS_CEILING_BYTES,
        "elapsed_s": time.time() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sector", default=SECTOR_DEFAULT)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--relational", default="none")
    ap.add_argument("--min-count", default="32")
    ap.add_argument("--clamp", default="16.0")
    ap.add_argument("--out", default="/Volumes/APDataStore/pact/ddm_ma1/race/within_miss_race.json")
    args = ap.parse_args()

    with np.load(args.sector) as z:
        d = {k: z[k] for k in z.files}
    print(f"sector: {d['arg'].size:,} miss records, "
          f"{WITHIN_MISS_CEILING_BYTES:.5f} B before")

    cell_names = list(CELLS) if args.cells == "all" else args.cells.split(",")
    rel_names = list(RELATIONAL) if args.relational == "all" else args.relational.split(",")
    min_counts = [int(x) for x in str(args.min_count).split(",")]
    clamps = [float(x) for x in str(args.clamp).split(",")]

    rows = []
    for cell_name, rel_name, mc, cl in itertools.product(
        cell_names, rel_names, min_counts, clamps
    ):
        row = run_model(
            d, cell_name=cell_name, rel_name=rel_name,
            min_count=mc, clamp_low=1.0 / cl, clamp_high=cl,
        )
        rows.append(row)
        print(
            f"{cell_name:20s} rel={rel_name:14s} mc={mc:<5d} clamp={cl:<6.3g} "
            f"{row['delta_bytes']:+10.3f} B  ({row['fraction_of_ceiling_pct']:5.2f}% of ceiling)"
            f"  [{row['elapsed_s']:.1f}s]"
        )

    rows.sort(key=lambda r: r["delta_bytes"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sector": args.sector,
        "sector_sha256": hashlib.sha256(Path(args.sector).read_bytes()).hexdigest(),
        "within_miss_ceiling_bytes": WITHIN_MISS_CEILING_BYTES,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nbest: {rows[0]['cell']} / rel={rows[0]['relational']} "
          f"mc={rows[0]['min_count']} clamp={rows[0]['clamp_high']} -> {rows[0]['delta_bytes']:+.3f} B")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
