#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_p4x (#920) — the connectivity-grammar control for the existence primitive.

Reproduces, from the cached argmax corpus and with ZERO scorer forwards, the
finding that ``gt2``'s published word grammar is **4-CONNECTED** while the
existence primitive's design prescribes **8-CONNECTED** components.

Why this script exists as a committed artifact rather than a one-off: the
per-word statistics the whole P4 row is priced against (Lane 58.23% annihilation,
ANNIHILATE:BIRTH 16.4:1) are NOT connectivity-invariant, so any capture fraction
quoted in "words" is meaningless until its grammar is named.  The S-arithmetic IS
invariant, and this script prints both so a reader can never conflate them.

Controls (all must PASS or the primitive is defending a grammar nobody measured):
  C1  4-connected component counts reproduce gt2 EXACTLY (Lane 16,581; Movable 2,207)
  C2  4-connected annihilated-word counts reproduce gt2 EXACTLY (9,655; 361)
  C3  4-connected ANNIHILATE pixel mass reproduces gt2 EXACTLY (47,226; 8,180)
  C4  GT pixel totals match gt2 EXACTLY (690,639; 1,460,325) -- proves same corpus

Authority: ``[macOS-CPU advisory]``; ``score_claim=False``; ``promotable=False``;
zero scorer forwards; pointer ``0.1910828242`` [contest-CPU] UNMOVED.

Usage:
    .venv/bin/python tools/ddm_p4x_connectivity_control.py [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from tac.optimization import existence_hinge as eh  # noqa: E402

ARGMAX_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
GT_PATH = ARGMAX_DIR / "gt_argmax_n600.npy"
CX_PATH = ARGMAX_DIR / "cx1_argmax_n600.npy"

#: gt2's published values, for the exact-match controls.  4-CONNECTED grammar.
GT2_EXPECTED = {
    "Lane": {"gt_px": 690639, "components": 16581, "annihilate_components": 9655,
             "annihilate_px": 47226},
    "Movable": {"gt_px": 1460325, "components": 2207, "annihilate_components": 361,
                "annihilate_px": 8180},
}
#: gt2's own ANNIHILATE threshold: a GT component retaining <5% of its pixels.
SURVIVAL_THRESHOLD = 0.05
CLASSES = ((eh.LANE, "Lane"), (eh.MOVABLE, "Movable"))


def measure(connectivity: int) -> dict:
    """Per-class component / annihilation statistics under one connectivity."""
    gt = np.load(GT_PATH, mmap_mode="r")
    cx = np.load(CX_PATH, mmap_mode="r")
    n_frames = int(gt.shape[0])
    out: dict[str, dict] = {}
    for cid, name in CLASSES:
        n_comp = n_ann = ann_px = gt_px = 0
        for i in range(n_frames):
            g = np.asarray(gt[i])
            c = np.asarray(cx[i])
            mask = g == cid
            if not mask.any():
                continue
            gt_px += int(mask.sum())
            lab, n = eh._label_components(mask, connectivity)
            n_comp += n
            flat = lab.reshape(-1)
            kept = (c == cid).reshape(-1)
            tot = np.bincount(flat, minlength=n + 1)[1:]
            surv = (np.bincount(flat[kept], minlength=n + 1)[1:]
                    if kept.any() else np.zeros(n, dtype=np.int64))
            frac = surv / np.maximum(tot, 1)
            ann = frac < SURVIVAL_THRESHOLD
            n_ann += int(ann.sum())
            ann_px += int(tot[ann].sum())
        out[name] = {
            "gt_px": gt_px,
            "components": n_comp,
            "annihilate_components": n_ann,
            "annihilate_px": ann_px,
            "annihilation_rate_of_words": (n_ann / n_comp) if n_comp else 0.0,
            "annihilate_ceiling_s": ann_px * eh.S_PER_FLIP,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None, help="write the verdict JSON here")
    ap.add_argument("--gap", type=float, default=0.6189279,
                    help="gap denominator for the %%-of-gap column")
    args = ap.parse_args(argv)

    for p in (GT_PATH, CX_PATH):
        if not p.exists():
            print(f"REFUSE: missing cached corpus {p}", file=sys.stderr)
            return 2

    t0 = time.time()
    res = {str(k): measure(k) for k in (eh.CONNECTIVITY_4, eh.CONNECTIVITY_8)}
    elapsed = time.time() - t0

    four = res[str(eh.CONNECTIVITY_4)]
    controls: dict[str, bool] = {}
    for name, exp in GT2_EXPECTED.items():
        got = four[name]
        for key in ("gt_px", "components", "annihilate_components", "annihilate_px"):
            controls[f"{name}.{key}"] = int(got[key]) == int(exp[key])
    all_pass = all(controls.values())

    print(f"ddm_p4x connectivity control  ({elapsed:.1f}s, 0 scorer forwards)\n")
    hdr = f"{'class':9s} {'conn':5s} {'comps':>7s} {'annih':>6s} {'rate':>7s} {'ann_px':>7s} {'S':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for conn in (eh.CONNECTIVITY_4, eh.CONNECTIVITY_8):
        for _, name in CLASSES:
            r = res[str(conn)][name]
            print(f"{name:9s} {conn:5d} {r['components']:7d} {r['annihilate_components']:6d} "
                  f"{r['annihilation_rate_of_words']:7.4f} {r['annihilate_px']:7d} "
                  f"{r['annihilate_ceiling_s']:9.6f}")

    print("\nEXACT-MATCH CONTROLS vs gt2 (4-connected grammar):")
    for k, ok in sorted(controls.items()):
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")

    ceil = {c: sum(res[str(c)][n]["annihilate_ceiling_s"] for _, n in CLASSES)
            for c in (eh.CONNECTIVITY_4, eh.CONNECTIVITY_8)}
    print("\nProtected-class ANNIHILATE ceiling (Lane + Movable):")
    for c in (eh.CONNECTIVITY_4, eh.CONNECTIVITY_8):
        print(f"  {c}-conn: {ceil[c]:.6f} S = {100 * ceil[c] / args.gap:.3f}% of gap {args.gap}")
    print("\nVERDICT: gt2's grammar is "
          f"{'4-CONNECTED (all controls exact)' if all_pass else 'NOT REPRODUCED -- investigate'}. "
          "\n  Per-WORD rates are grammar-dependent and must never be quoted across grammars."
          "\n  S-arithmetic is grammar-invariant per pixel and safe to quote once named.")

    verdict = {
        "arm": "ddm_p4x", "task": 920,
        "axis": "[macOS-CPU advisory]", "score_claim": False, "promotable": False,
        "scorer_forwards_run": 0,
        "substrate": f"{GT_PATH.name} + {CX_PATH.name}",
        "survival_threshold": SURVIVAL_THRESHOLD,
        "by_connectivity": res,
        "gt2_expected_4conn": GT2_EXPECTED,
        "exact_match_controls": controls,
        "all_controls_pass": all_pass,
        "protected_ceiling_s": {str(k): v for k, v in ceil.items()},
        "gap_denominator": args.gap,
        "elapsed_s": elapsed,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
