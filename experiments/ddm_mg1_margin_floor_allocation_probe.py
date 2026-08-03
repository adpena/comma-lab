"""ddm_mg1 — does raising ``margin_floor`` aim the seg gradient at the separatrix?

Scorer-free. Reads two already-cached n600 artifacts and answers three questions
that decide whether ``ddm_rt2``'s C1 ("raise ``margin_floor`` toward the measured
p5% = 2.0582") is the cure it is advertised as:

Q1  CONTROL. Does ``pu2``'s independently recomputed GT argmax agree with the
    canonical cached ``lstars``? If not, every cross-tab below is meaningless and
    the probe must refuse rather than report.

Q2  STRUCTURAL. A site is a realized flip iff ``argmax != target`` iff
    ``margin = target_logit - max_competing_logit < 0``. The hinge
    ``relu(floor - margin)`` is therefore ACTIVE on every flipped site for ANY
    ``floor > 0``. So raising the floor cannot recruit one additional flipped
    site; every site it recruits has ``margin >= floor_old > 0``, i.e. is a site
    the scorer ALREADY GETS RIGHT. Verified numerically on the real hinge.

Q3  ALLOCATION. Is the frozen head's own margin field a valid selector for where
    THIS vehicle actually fails? Cross-tabulate the GT-reference margin against
    the realized cx1 flip mask over all 117,964,800 sites. If flips concentrate
    at small GT margin, margin is a valid allocation key and the cure is a
    weight/shape change at fixed support. If they do not, the whole
    margin-floor cure family is mis-aimed and must be said so.

Inputs (both already on disk; no scorer, no training):
  experiments/results/mlx_fleet_gt_cache/gt_n600.npz      -> lstars, margins
  /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/
      gt_argmax_n600.npy, cx1_argmax_n600.npy             -> realized flip mask

Outputs a JSON receipt with the full denominator on every count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
PU2_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")

# The live floor and rt2's proposed target, plus a grid that exposes the limit.
FLOOR_GRID = [0.01, 0.05, 0.1, 0.2, 0.3552, 0.5, 1.0, 2.0582, 4.0, 5.8934, 10.0, 100.0]
# Margin bins for the cross-tab. Chosen to straddle 0 (the separatrix) and the
# L7 cross-hardware drift band (~0.096).
MARGIN_BINS = [
    -np.inf, -4.0, -2.0, -1.0, -0.5, -0.096, 0.0,
    0.096, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, np.inf,
]


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _hinge_numeric_check() -> dict:
    """Q2, executed rather than asserted.

    Builds a tiny explicit margin vector spanning the flip boundary and evaluates
    the REAL hinge value/derivative at two floors. The derivative of
    ``mean(relu(f - m))`` w.r.t. ``m_i`` is ``-1/N`` when ``m_i < f`` and ``0``
    otherwise -- flat, i.e. independent of how far below the floor the site sits.
    """
    m = np.array([-3.0, -0.5, -1e-6, 0.0, 1e-6, 0.05, 0.099, 0.5, 2.0, 5.9], dtype=np.float64)
    out = {}
    for f in (0.1, 2.0582):
        active = m < f
        # d/dm_i of mean(relu(f - m)) is -1/N on active sites, 0 elsewhere.
        grad = np.where(active, -1.0 / m.size, 0.0)
        out[f"floor_{f}"] = {
            "value": float(np.mean(np.maximum(f - m, 0.0))),
            "active_sites": int(active.sum()),
            "grad_per_active_site": float(-1.0 / m.size),
            "grad_absmax": float(np.abs(grad).max()),
            "distinct_nonzero_grad_magnitudes": int(
                np.unique(np.abs(grad[grad != 0.0]).round(15)).size
            ),
            "flipped_sites_active": int((m[m < 0.0] < f).sum()),
            "flipped_sites_total": int((m < 0.0).sum()),
        }
    # The limit: as floor -> inf the relu never clips, so the hinge degenerates to
    # ``floor - mean(margin)`` -- a flat-weight BULK mean-margin objective.
    big = 1.0e6
    out["limit_check"] = {
        "floor": big,
        "hinge_value": float(np.mean(np.maximum(big - m, 0.0))),
        "floor_minus_mean_margin": float(big - m.mean()),
        "abs_diff": float(abs(np.mean(np.maximum(big - m, 0.0)) - (big - m.mean()))),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument(
        "--out",
        default=str(REPO / ".omx/research/ddm_mg1_margin_floor_allocation_n600.json"),
    )
    args = ap.parse_args()

    gt_arg = np.load(PU2_CACHE / "gt_argmax_n600.npy", mmap_mode="r")
    cx_arg = np.load(PU2_CACHE / "cx1_argmax_n600.npy", mmap_mode="r")
    npz = np.load(GT_CACHE)
    lstars = npz["lstars"]
    margins = npz["margins"]
    if margins.ndim == 4:  # (B,1,H,W) -> (B,H,W)
        margins = margins[:, 0]

    n = min(args.pairs, gt_arg.shape[0], margins.shape[0], lstars.shape[0])
    # The pu2 cache is a resumable ``open_memmap(mode="w+")`` file: pairs the run
    # never reached are all-zero, NOT valid argmax planes. Counting them would
    # both inflate the denominator and silently drop their flips. Detect and
    # truncate to the written prefix, and RECORD the truncation.
    written = n
    for p in range(n):
        if not np.asarray(gt_arg[p]).any() and not np.asarray(cx_arg[p]).any():
            written = p
            break
    truncated_pairs = n - written
    n = written
    h, w = margins.shape[1], margins.shape[2]
    sites_per_pair = h * w
    total_sites = n * sites_per_pair

    # --- accumulators -----------------------------------------------------
    control_disagree = 0
    n_flip = 0
    hist_flip = np.zeros(len(MARGIN_BINS) - 1, dtype=np.int64)
    hist_ok = np.zeros(len(MARGIN_BINS) - 1, dtype=np.int64)
    # per-floor: sites with GT margin < floor, and how many of them are flips
    floor_active = np.zeros(len(FLOOR_GRID), dtype=np.int64)
    floor_active_flip = np.zeros(len(FLOOR_GRID), dtype=np.int64)
    # hinge value on the GT-reference margins (rt2's number, re-derived here)
    hinge_sum = np.zeros(len(FLOOR_GRID), dtype=np.float64)
    margin_neg = 0
    flip_margin_sum = 0.0
    ok_margin_sum = 0.0
    pairs_seen = 0

    for start in range(0, n, args.chunk):
        stop = min(start + args.chunk, n)
        g = np.asarray(gt_arg[start:stop], dtype=np.int64)
        c = np.asarray(cx_arg[start:stop], dtype=np.int64)
        ls = np.asarray(lstars[start:stop], dtype=np.int64)
        m = np.asarray(margins[start:stop], dtype=np.float64)

        control_disagree += int((g != ls).sum())
        flip = g != c
        n_flip += int(flip.sum())
        margin_neg += int((m < 0.0).sum())

        mf = m[flip]
        mo = m[~flip]
        flip_margin_sum += float(mf.sum())
        ok_margin_sum += float(mo.sum())
        hist_flip += np.histogram(mf, bins=MARGIN_BINS)[0]
        hist_ok += np.histogram(mo, bins=MARGIN_BINS)[0]

        for i, f in enumerate(FLOOR_GRID):
            act = m < f
            floor_active[i] += int(act.sum())
            floor_active_flip[i] += int((act & flip).sum())
            hinge_sum[i] += float(np.maximum(f - m, 0.0).sum())

        pairs_seen = stop
        print(f"[mg1] pairs {stop}/{n}", flush=True)

    control_ok = control_disagree == 0
    d_seg = n_flip / total_sites

    rows = []
    for i, f in enumerate(FLOOR_GRID):
        act = int(floor_active[i])
        actflip = int(floor_active_flip[i])
        rows.append({
            "floor": f,
            "gt_margin_active_sites": act,
            "gt_margin_active_fraction": act / total_sites,
            "of_which_realized_flips": actflip,
            "flip_share_of_active": (actflip / act) if act else None,
            "flip_coverage": actflip / n_flip if n_flip else None,
            "hinge_value_on_gt_margins": hinge_sum[i] / total_sites,
            "hinge_contribution_at_w0.05": 0.05 * hinge_sum[i] / total_sites,
        })

    bins_out = []
    for j in range(len(MARGIN_BINS) - 1):
        lo, hi = MARGIN_BINS[j], MARGIN_BINS[j + 1]
        tot = int(hist_flip[j] + hist_ok[j])
        bins_out.append({
            "lo": None if np.isneginf(lo) else float(lo),
            "hi": None if np.isposinf(hi) else float(hi),
            "sites": tot,
            "flips": int(hist_flip[j]),
            "flip_rate_in_bin": (int(hist_flip[j]) / tot) if tot else None,
            "share_of_all_flips": (int(hist_flip[j]) / n_flip) if n_flip else None,
            "share_of_all_sites": tot / total_sites,
        })

    receipt = {
        "arm": "ddm_mg1",
        "git_head": _git_head(),
        "axis": "[macOS-CPU scorer-free advisory] score_claim=false promotable=false",
        "inputs": {
            "gt_cache": str(GT_CACHE),
            "pu2_argmax_cache": str(PU2_CACHE),
        },
        "denominator": {
            "pairs": pairs_seen,
            "h": h, "w": w,
            "sites_per_pair": sites_per_pair,
            "total_sites": total_sites,
            "pu2_cache_unwritten_pairs_excluded": truncated_pairs,
            "pu2_cache_note": (
                "pairs beyond the written prefix are all-zero memmap fill, not "
                "argmax planes; excluded from BOTH numerator and denominator"
            ),
        },
        "control_q1_pu2_gt_vs_lstars": {
            "disagreeing_sites": control_disagree,
            "agree": control_ok,
            "note": "MUST be 0; a nonzero value invalidates every cross-tab below",
        },
        "realized_flips": {
            "n_flip": n_flip,
            "d_seg": d_seg,
            "seg_leg_100x": 100.0 * d_seg,
        },
        "control_gt_margin_sign": {
            "gt_margin_negative_sites": margin_neg,
            "note": "MUST be 0; confirms `margins` is the GT-reference (self) margin",
        },
        "mean_gt_margin": {
            "on_realized_flip_sites": flip_margin_sum / n_flip if n_flip else None,
            "on_correct_sites": ok_margin_sum / (total_sites - n_flip),
        },
        "q2_hinge_numeric": _hinge_numeric_check(),
        "q3_floor_grid": rows,
        "q3_margin_bins": bins_out,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in
                      ("denominator", "control_q1_pu2_gt_vs_lstars", "realized_flips",
                       "control_gt_margin_sign", "mean_gt_margin")}, indent=2))
    if not control_ok:
        print("[mg1] CONTROL FAILED — cross-tab REFUSED as evidence", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
