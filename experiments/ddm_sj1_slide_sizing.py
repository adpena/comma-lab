"""ddm_sj1 -- SIZE a two-cell SLIDE proposal family against the measured residual.

WHY THIS FAMILY, AND WHY NOW
----------------------------
The pass-4 census (memo section 21) measured the residual as **99.56% a ONE-PIXEL boundary
displacement** on a codim-1 curve, and measured that **89.46% of it has already been
proposed on and REJECTED** by three passes of the single-cell family.  Those two facts
together are a specific, falsifiable claim about what the single-cell family cannot
express -- not that the cells are unexplored, but that the available move shape is wrong
for them.

A single move WIDENS one class by one token: it changes the class balance across the edge.
The displacement the census measures does not need widening, it needs TRANSLATION.  So the
slide changes TWO tokens straddling the site -- advancing the GT class one step on one side
while retreating it one step on the other -- which moves the token boundary without
changing its width.  That is the move a one-token family provably cannot express, and it is
distinct from jg1's block/dilation moves, which set a whole neighbourhood to one class.

PRICE, HONESTLY
---------------
Two changed tokens per proposal, so break-even is ~2x the single-cell break-even per
repaired cell: at pass 5's projected 9.7916 bits/token that is **~19.58 bits per repaired
cell**.  A slide therefore pays only where it repairs >= 2 cells, or where the second token
is one the coder charges near zero.  The second condition is REAL but is not measured here:
per-token -log2 p is direction-dependent ([[fs2]]) and average != marginal ([[fs3]], 2.24x),
so it is settled by a real re-encode, never by a model.  This sizing decides the MECHANISM
question only -- does the shape repair refused cells at all -- and hands the price question
to the encoder if and only if the mechanism clears.

PRE-REGISTERED STOP RULE
------------------------
The family is DEAD, and is not included in pass 5, unless BOTH hold on the 12 seeded pairs:
  * it repairs **>= 5%** of the REFUSED cells (cells the single-cell family proposed on and
    rejected), and
  * accepted slides average **>= 1.5 cells per slide**.
Anything less cannot clear a 2x break-even, and the pass-5 plan stays singles-only.

AUTHORITY
---------
The same instrument as every other row on this arm: ``argmax_for_tokens`` -- the receiver's
own ``jg1.render_frame1`` at batch 1 followed by the frozen CPU SegNet through the
evaluator's preprocess -- on the DALI GT lineage.  Acceptance is the REALIZED whole-pair
flip count, exactly as the production engine judges a single: local attribution only orders
the search, the composite re-render is the verdict.  ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_sj1_multipass_token_predistortion as sj1  # noqa: E402

#: The eight directions a boundary can slide along.
SLIDE_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),
)


def enumerate_slides(
    tokens: np.ndarray, argmax: np.ndarray, gt: np.ndarray
) -> list[dict[str, Any]]:
    """Two-token translations of the token boundary at every flipped cell.

    For a flipped site ``s`` reading ``o`` where GT says ``g``, and for each direction
    ``d``: advance ``g`` into ``s + d`` and retreat it at ``s - d`` by writing ``o`` there.
    The pair of writes translates the boundary by one token along ``d`` while leaving the
    number of ``g`` tokens across the cut unchanged -- the degree of freedom a single-token
    move cannot reach.

    DEDUPED on the concrete two-position write, because adjacent flipped cells sharing a GT
    class generate the same physical slide and realizing it twice would double-count it.
    """
    ys, xs = np.nonzero(argmax != gt)
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    h, w = gt.shape
    for site_y, site_x in zip(ys.tolist(), xs.tolist(), strict=True):
        g = int(gt[site_y, site_x])
        o = int(argmax[site_y, site_x])
        if g == o:
            continue
        for dy, dx in SLIDE_DIRECTIONS:
            ay, ax = site_y + dy, site_x + dx
            by, bx = site_y - dy, site_x - dx
            if not (0 <= ay < h and 0 <= ax < w and 0 <= by < h and 0 <= bx < w):
                continue
            if int(tokens[ay, ax]) == g and int(tokens[by, bx]) == o:
                continue  # already in the slid configuration; nothing to write
            key = (ay, ax, g, by, bx, o)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "site_y": site_y, "site_x": site_x,
                    "ay": ay, "ax": ax, "advance_class": g,
                    "by": by, "bx": bx, "retreat_class": o,
                    "direction": [dy, dx],
                }
            )
    return out


def size_pair(
    body: sj1.Body,
    pair: int,
    tokens: np.ndarray,
    refused: np.ndarray,
    max_slides: int,
) -> dict[str, Any]:
    """Greedy realized descent over the slide family for ONE pair."""
    tokens = np.ascontiguousarray(tokens)
    gt = body.gt[pair]
    argmax = sj1.argmax_for_tokens(body, tokens, pair)
    flips_before = int((argmax != gt).sum())
    slides = enumerate_slides(tokens, argmax, gt)
    accepted: list[dict[str, Any]] = []
    evaluations = 0
    repaired_refused = 0
    for slide in slides:
        if evaluations >= max_slides:
            break
        if argmax[slide["site_y"], slide["site_x"]] == gt[slide["site_y"], slide["site_x"]]:
            continue  # the far field of an earlier accepted slide already fixed it
        trial = tokens.copy()
        trial[slide["ay"], slide["ax"]] = slide["advance_class"]
        trial[slide["by"], slide["bx"]] = slide["retreat_class"]
        if np.array_equal(trial, tokens):
            continue
        evaluations += 1
        new_argmax = sj1.argmax_for_tokens(body, trial, pair)
        new_flips = int((new_argmax != gt).sum())
        if new_flips >= flips_before:
            continue
        fixed = (argmax != gt) & (new_argmax == gt)
        repaired_refused += int((fixed & refused).sum())
        slide = dict(slide)
        slide["repaired"] = flips_before - new_flips
        slide["repaired_refused"] = int((fixed & refused).sum())
        accepted.append(slide)
        tokens, argmax, flips_before = trial, new_argmax, new_flips
    changed = int((tokens != np.ascontiguousarray(body.tokens[pair])).sum())
    return {
        "pair": pair,
        "flips_after": int((argmax != gt).sum()),
        "slides_enumerated": len(slides),
        "slides_evaluated": evaluations,
        "slides_accepted": len(accepted),
        "cells_repaired": int(sum(s["repaired"] for s in accepted)),
        "refused_cells_repaired": repaired_refused,
        "refused_cells_present": int(refused.sum()),
        "tokens_changed_vs_pair_start": changed,
        "accepted": accepted,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", type=Path, required=True, help="the SHIPPED token field")
    ap.add_argument("--argmax", type=Path, required=True, help="the SHIPPED parse-back argmax")
    ap.add_argument("--base-argmax", type=Path, required=True, help="the body as this arm found it")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--max-slides-per-pair", type=int, default=400)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sj1._set_threads(args.threads)
    body = sj1.load_body(with_raw=False)
    shipped = np.load(args.argmax)
    base = np.load(args.base_argmax)
    blob = np.load(args.field, allow_pickle=False)
    field = {int(k): np.asarray(blob[k], dtype=np.uint8) for k in blob.files}

    indices = sj1.up2.select_pairs(args.pairs, args.seed)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for pair in indices.tolist():
        gt = body.gt[pair]
        # REFUSED = still flipped on the shipped bytes AND flipped from the very start, so
        # all three single-cell passes enumerated proposals on it and rejected them.
        refused = (shipped[pair] != gt) & (base[pair] != gt)
        row = size_pair(
            body, int(pair), field.get(int(pair), body.tokens[pair]), refused,
            args.max_slides_per_pair,
        )
        rows.append(row)
        print(
            f"  pair {pair:3d}: refused {row['refused_cells_present']:3d} | "
            f"slides {row['slides_evaluated']:4d} eval, {row['slides_accepted']:3d} accepted | "
            f"cells {row['cells_repaired']:3d} (refused {row['refused_cells_repaired']:3d})",
            flush=True,
        )

    ref_present = sum(r["refused_cells_present"] for r in rows)
    ref_fixed = sum(r["refused_cells_repaired"] for r in rows)
    accepted = sum(r["slides_accepted"] for r in rows)
    cells = sum(r["cells_repaired"] for r in rows)
    frac = ref_fixed / ref_present if ref_present else 0.0
    per_slide = cells / accepted if accepted else 0.0
    verdict = "PROCEED" if (frac >= 0.05 and per_slide >= 1.5) else "DEAD"
    report = {
        "schema": "ddm_sj1_slide_sizing.v1",
        "pairs": [int(p) for p in indices],
        "seed": args.seed,
        "refused_cells_present": ref_present,
        "refused_cells_repaired": ref_fixed,
        "refused_fraction_repaired": frac,
        "slides_enumerated": sum(r["slides_enumerated"] for r in rows),
        "slides_evaluated": sum(r["slides_evaluated"] for r in rows),
        "slides_accepted": accepted,
        "cells_repaired": cells,
        "cells_per_accepted_slide": per_slide,
        "tokens_per_slide": 2,
        "break_even_bits_per_repaired_cell_at_pass5_price": 2 * 9.7916,
        "stop_rule": ">= 5% of refused cells repaired AND >= 1.5 cells per accepted slide",
        "verdict": verdict,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"\n  REFUSED cells present {ref_present:,} | repaired by a slide {ref_fixed:,} "
        f"= {frac*100:.2f}% (bar 5%)\n"
        f"  accepted slides {accepted:,} | cells {cells:,} | "
        f"{per_slide:.3f} cells/slide (bar 1.5)\n"
        f"  VERDICT {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
