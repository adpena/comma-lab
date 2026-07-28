#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_sp1 R3 — BASE-TRANSFER H CONTRACT: H(flip|ctx) + support bytes on a NEW base. GATED.

MEANS. pointer 0.19108 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE. This is the
READY-TO-FIRE tool for the gc5 B1 rung (a better base shrinks the flip support BEFORE coding). It
is GATED on sc1's seeded-base per-pair argmax masks — DO NOT invent them; the tool refuses to run
until the real base-argmax cache exists. NO-FAKE: it recomputes the flip field (base_argmax != GT
lstar) on the REAL provided base + the REAL GT, then codes the support with the SAME #307 contour
coder + LZMA race that R1 measured on the copy base, and fires the SAME falsifier band.

THE CONTRACT (fire the moment sc1's base lands):
  input  : --base-argmax  (a) a dir of per-pair .npz each with key 'argmax' (P,384,512) uint8,
                           or (b) a single .npz with key 'argmax' (P,384,512) uint8.
           --gt-cache      the mlx_fleet gt cache (key 'lstars').
  compute: flip = base_argmax != lstar  (support geometry); code with contour(counts+anchor+chain)
           + packbits-LZMA1-x9e race; report REAL support bytes + round-trip proof.
  FALSIFIER (B1 rung): copy-base support (R1 measured) = 444,394 B contour / 421,366 B LZMA, and
           the lossy-optimal S_support = 0.280. B1 HELPS only if the new base's support beats that:
             new support < ~250 KB           -> B1 opens the sub-bar floor (was DEAD on copy base)
             250 KB <= new support < copy     -> B1 helps but floor still bar-bound; re-run lossy curve
             new support >= copy-base support -> B1 does NOT help; base transfer rejected (typed scope)
  The tool prints the verdict and writes the JSON; NO scorer, NO render (sc1 owns that slot).

WHY GATED: sc1's base masks are produced by a seeded base (a different predictor than copy=f0). Only
sc1's arm materializes them. Until then this tool has no legitimate input and exits rc=3 (GATED),
never fabricates a base.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# REUSE R1's measured coder machinery verbatim (contour + LZMA + falsifier band).
from ddm_sp1_contour_support_coder import (
    FALSIFIER_DEAD,
    FALSIFIER_STRENGTHEN,
    _contour_measure,
    _lzma_support_bytes,
)

# copy-base R1 measured anchors (the thing B1 must beat)
COPY_BASE_CONTOUR_SUPPORT = 444_394
COPY_BASE_LZMA_SUPPORT = 421_366
COPY_BASE_LOSSY_OPT_S = 0.280


def _load_base_argmax(path: Path, n: int) -> np.ndarray:
    """Load sc1's per-pair base argmax (P,384,512) uint8. Dir-of-npz or single-npz. GATED."""
    if path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no .npz in base-argmax dir {path}")
        arrs = []
        for f in files:
            d = np.load(str(f))
            key = "argmax" if "argmax" in d else next(iter(d.keys()))
            arrs.append(np.asarray(d[key], np.uint8))
        base = np.concatenate(arrs, axis=0)
    else:
        d = np.load(str(path))
        key = "argmax" if "argmax" in d else next(iter(d.keys()))
        base = np.asarray(d[key], np.uint8)
    return base[:n]


def _verdict_b1(new_support: int) -> str:
    if new_support < FALSIFIER_STRENGTHEN:
        return "B1_OPENS_FLOOR(<150KB)"
    if new_support < FALSIFIER_DEAD:
        return "B1_HELPS_BAR_BOUND(150-250KB)"
    if new_support < COPY_BASE_LZMA_SUPPORT:
        return "B1_HELPS_STILL_BAR_BOUND(250KB-copybase)"
    return "B1_DOES_NOT_HELP(>=copy_base)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-argmax", required=True,
                    help="sc1 seeded-base per-pair argmax (dir of *.npz['argmax'] or single .npz). GATED.")
    ap.add_argument("--gt-cache",
                    default="/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_path = Path(args.base_argmax)
    if not base_path.exists():
        print(f"[sp1-R3] GATED: sc1 base-argmax not found at {base_path} — DO NOT fabricate. "
              "This tool fires only on the REAL seeded base. Exiting rc=3.", flush=True)
        sys.exit(3)

    t0 = time.time()
    base = _load_base_argmax(base_path, args.n)
    gt = np.load(args.gt_cache)
    lstars = gt["lstars"][: args.n].astype(np.uint8)
    P = min(base.shape[0], lstars.shape[0])
    base, lstars = base[:P], lstars[:P]
    flips = [base[i] != lstars[i] for i in range(P)]
    classes = [lstars[i].astype(np.int64) for i in range(P)]
    total_flips = int(sum(int(f.sum()) for f in flips))
    total_sites = P * flips[0].size
    print(f"[sp1-R3] base={base_path.name} P={P} flips={total_flips} "
          f"frac={total_flips/total_sites:.8f} ({time.time()-t0:.0f}s)", flush=True)

    lzma_support = _lzma_support_bytes(flips)
    row = _contour_measure(flips, classes, "new_base_contour")
    new_support = min(row["contour_support_bytes"], lzma_support)
    best_coder = "contour" if row["contour_support_bytes"] <= lzma_support else "lzma"
    verdict = _verdict_b1(new_support)

    result = {
        "schema": "ddm_sp1_base_transfer_h_contract.v1",
        "task": "gc5 B1 rung — base-transfer support H on a NEW seeded base (GATED fire)",
        "evidence_axis": ("[macOS-CPU advisory] NON-PROMOTABLE lossless coder bytes over a NEW-base "
                          "flip field; NOT a byte-closed evaluate.py row; pointer 0.19108 UNMOVED"),
        "utc": datetime.now(UTC).isoformat(),
        "base_source": str(base_path),
        "n_pairs": P,
        "total_flips": total_flips,
        "support_fraction": total_flips / total_sites,
        "new_base_contour_support_bytes": row["contour_support_bytes"],
        "new_base_lzma_support_bytes": lzma_support,
        "new_base_best_support_bytes": new_support,
        "new_base_best_coder": best_coder,
        "new_base_contour_lossless_roundtrip": row["lossless_roundtrip"],
        "copy_base_anchors": {
            "contour_support_bytes": COPY_BASE_CONTOUR_SUPPORT,
            "lzma_support_bytes": COPY_BASE_LZMA_SUPPORT,
            "lossy_optimal_S_support": COPY_BASE_LOSSY_OPT_S,
        },
        "falsifier": {
            "verdict": verdict,
            "new_support_vs_copy_ratio": round(new_support / COPY_BASE_LZMA_SUPPORT, 4),
            "support_saved_vs_copy_bytes": COPY_BASE_LZMA_SUPPORT - new_support,
            "verdict_scope": "FORMULATION: explicit support stream on this base + this coder race",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[sp1-R3] VERDICT: {verdict} (new support={new_support} B via {best_coder} vs "
          f"copy-base LZMA {COPY_BASE_LZMA_SUPPORT} B); wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
