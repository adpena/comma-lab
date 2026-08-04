"""ddm_cg1 probe: does the free per-site margin key REACH every class, or is its
headline precision a Road/Undrivable statistic?

WHY THIS EXISTS
---------------
`ddm_mg1` measured the frozen head's scalar margin (GT self-margin, top1 - top2)
as a per-SITE allocation key and reported 98.1x enrichment / 42.51% precision in
the lowest bin. Those are POPULATION numbers over a mixture in which Undrivable
covers 49.5% of the plane and Lane covers 0.59% -- an 85x area spread.

Per the standing lesson (`m88`: a prefix of a skewed population is a different
population), a population precision computed over that mixture can be almost
entirely a Road/Undrivable statistic while saying nothing about Lane -- and
Lane->Road alone is 36.30% of every flip we make (the single largest directed
side). If the free key does not reach Lane, then the class label is NOT redundant
with the margin key, and `ddm_cg1`'s ledger claim "class/edge is a PRIOR on sites"
becomes operational rather than decorative.

WHAT THIS DOES NOT DO
---------------------
It does not re-measure the margin key's population behaviour (mg1 owns that), the
per-side barrier (hg1), depth-by-side (hg1), per-frame concentration (dd1), or
per-mechanism prices (wf2). It conditions mg1's key on GT class -- nothing else.

AUTHORITY
---------
[macOS-CPU scorer-free advisory] score_claim=false promotion_eligible=false
rank_or_kill_eligible=false. ZERO scorer forwards: every input is a cached array.
Positive control: the flip mask must reproduce the evaluator's seg leg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

# The evaluator's own seg leg for the live cx1 vehicle, reproduced by ddm_pu2 and
# independently by ddm_mg1. Our flip mask MUST reproduce this or every row below
# is measuring something other than the scored quantity.
D_SEG_EXPECTED = 0.004311794704861111

DEFAULT_ARGMAX_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
)
DEFAULT_GT_NPZ = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)

# Bin edges chosen to CONTAIN mg1's two published bins verbatim -- [0, 0.096) and
# [4, 8) -- so our per-class numbers are directly comparable to its population
# numbers rather than being a differently-binned near-miss.
MARGIN_BIN_EDGES = (0.0, 0.096, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf)

# Global rank budgets: sort ALL sites by ascending margin and take the first N.
# This is the operational question a global byte budget actually asks.
RANK_BUDGETS = (10_000, 100_000, 500_000, 1_000_000, 5_000_000)


def _sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            if limit is not None and read + len(chunk) > limit:
                h.update(chunk[: limit - read])
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parent.parent
        ).strip()
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    ap.add_argument("--gt-npz", type=Path, default=DEFAULT_GT_NPZ)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt_path = args.argmax_cache / "gt_argmax_n600.npy"
    cx1_path = args.argmax_cache / "cx1_argmax_n600.npy"

    gt = np.load(gt_path, mmap_mode="r")
    cx1 = np.load(cx1_path, mmap_mode="r")
    n_pairs, h, w = gt.shape
    total_sites = int(n_pairs) * int(h) * int(w)

    # margins is the GT SELF-margin (top1 - top2 of the frozen head on the GT
    # frame). mg1's control established it is strictly positive everywhere, i.e.
    # it measures how fragile the GT label is at a site -- an encode-side prior,
    # never a decode-time quantity.
    with np.load(args.gt_npz) as z:
        margins = np.asarray(z["margins"], dtype=np.float32)
    if margins.shape != gt.shape:
        raise SystemExit(f"margin/argmax shape mismatch: {margins.shape} vs {gt.shape}")

    gt_flat = np.asarray(gt, dtype=np.uint8).reshape(-1)
    cx1_flat = np.asarray(cx1, dtype=np.uint8).reshape(-1)
    m_flat = margins.reshape(-1)
    del margins

    flip = gt_flat != cx1_flat
    n_flip = int(flip.sum())
    d_seg = n_flip / total_sites

    # ---- POSITIVE CONTROL -------------------------------------------------
    # If this fails, the flip mask is not the scored quantity and nothing below
    # is admissible. Fail closed rather than emitting a plausible-looking row.
    rel_err = abs(d_seg - D_SEG_EXPECTED) / D_SEG_EXPECTED
    control_ok = bool(rel_err < 1e-6)
    control_margin_negative = int((m_flat < 0).sum())

    # ---- per-class base rates ---------------------------------------------
    per_class: dict[str, dict] = {}
    for c, name in enumerate(CLASS_NAMES):
        sel = gt_flat == c
        n_sites_c = int(sel.sum())
        n_flip_c = int(flip[sel].sum())
        per_class[name] = {
            "gt_class_index": c,
            "sites": n_sites_c,
            "flips": n_flip_c,
            "area_share_of_plane": n_sites_c / total_sites,
            "share_of_all_flips": (n_flip_c / n_flip) if n_flip else 0.0,
            # base rate = P(flip | this class). This is the denominator every
            # enrichment below is measured against -- a class's own base rate,
            # NOT the population base rate. Using the population rate would
            # manufacture enrichment for whichever class is simply worse.
            "base_flip_rate": (n_flip_c / n_sites_c) if n_sites_c else 0.0,
        }

    # ---- margin bins, conditioned on GT class ------------------------------
    edges = np.asarray(MARGIN_BIN_EDGES, dtype=np.float64)
    bin_idx = np.digitize(m_flat, edges[1:-1], right=False)

    for c, name in enumerate(CLASS_NAMES):
        sel_c = gt_flat == c
        base = per_class[name]["base_flip_rate"]
        rows = []
        for b in range(len(edges) - 1):
            sel = sel_c & (bin_idx == b)
            n_sites_b = int(sel.sum())
            n_flip_b = int(flip[sel].sum())
            prec = (n_flip_b / n_sites_b) if n_sites_b else 0.0
            rows.append(
                {
                    "bin": f"[{edges[b]:g}, {edges[b + 1]:g})",
                    "sites": n_sites_b,
                    "flips": n_flip_b,
                    "precision": prec,
                    "enrichment_vs_own_base": (prec / base) if base else 0.0,
                    "share_of_class_flips": (
                        n_flip_b / per_class[name]["flips"]
                    )
                    if per_class[name]["flips"]
                    else 0.0,
                }
            )
        per_class[name]["margin_bins"] = rows

    # ---- population bins (mg1 comparability check) -------------------------
    pop_base = n_flip / total_sites
    population_bins = []
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        n_sites_b = int(sel.sum())
        n_flip_b = int(flip[sel].sum())
        prec = (n_flip_b / n_sites_b) if n_sites_b else 0.0
        population_bins.append(
            {
                "bin": f"[{edges[b]:g}, {edges[b + 1]:g})",
                "sites": n_sites_b,
                "flips": n_flip_b,
                "precision": prec,
                "enrichment_vs_population_base": (prec / pop_base) if pop_base else 0.0,
            }
        )

    # ---- GLOBAL rank budgets: does a global key reach every class? ---------
    # A byte budget is spent GLOBALLY. So the operational question is not "is
    # this class's low-margin bin precise" but "when the global key spends N
    # sites, how many of THIS class's flips did it buy?"
    order = np.argsort(m_flat, kind="stable")
    gt_ordered = gt_flat[order]
    flip_ordered = flip[order]
    del order

    budgets = []
    for nbudget in RANK_BUDGETS:
        if nbudget > total_sites:
            continue
        head_gt = gt_ordered[:nbudget]
        head_flip = flip_ordered[:nbudget]
        entry = {"budget_sites": int(nbudget), "per_class": {}}
        entry["captured_flips_total"] = int(head_flip.sum())
        entry["precision_total"] = float(head_flip.sum()) / nbudget
        for c, name in enumerate(CLASS_NAMES):
            selc = head_gt == c
            picked = int(selc.sum())
            captured = int(head_flip[selc].sum())
            entry["per_class"][name] = {
                "sites_picked": picked,
                "flips_captured": captured,
                "precision_within_class": (captured / picked) if picked else 0.0,
                # THE decisive number: of this class's total flips, what
                # fraction did a GLOBAL budget of N sites actually buy?
                "recall_of_class_flips": (
                    captured / per_class[name]["flips"]
                )
                if per_class[name]["flips"]
                else 0.0,
                "share_of_budget": picked / nbudget,
            }
        budgets.append(entry)

    out = {
        "arm": "ddm_cg1",
        "probe": "per-class reach of the free per-site GT-margin key (mg1's key, conditioned on class)",
        "axis": "[macOS-CPU scorer-free advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "git_head": _git_head(),
        "inputs": {
            "gt_argmax": str(gt_path),
            "gt_argmax_sha256_first_64MiB": _sha256(gt_path, 64 << 20),
            "cx1_argmax": str(cx1_path),
            "cx1_argmax_sha256_first_64MiB": _sha256(cx1_path, 64 << 20),
            "gt_npz": str(args.gt_npz),
        },
        "denominator": {
            "pairs": int(n_pairs),
            "h": int(h),
            "w": int(w),
            "total_sites": total_sites,
            "n_flip": n_flip,
        },
        "positive_control": {
            "d_seg_measured": d_seg,
            "d_seg_expected": D_SEG_EXPECTED,
            "rel_err": rel_err,
            "verdict": "ARGMAX_VERIFIED" if control_ok else "CONTROL_FAILED",
            "gt_margin_negative_sites": control_margin_negative,
            "gt_margin_sign_note": "MUST be 0; confirms margins is the GT self-margin",
        },
        "population_margin_bins": population_bins,
        "per_class": per_class,
        "global_rank_budgets": budgets,
    }

    if not control_ok:
        out["BLOCKER"] = (
            "positive control failed: flip mask does not reproduce the evaluator seg leg; "
            "no row in this file is admissible"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"control: d_seg={d_seg!r} rel_err={rel_err:.3e} -> {out['positive_control']['verdict']}")
    print(f"gt_margin_negative_sites={control_margin_negative} (must be 0)")
    print(f"wrote {args.out}")
    return 0 if control_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
