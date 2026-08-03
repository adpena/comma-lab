"""ddm_mg1 — F4: is the barrier integral a ranking key the ``#766`` waterfill is missing?

``ddm_hg1`` measured that ``d_seg`` prices ``Lane->Road`` erasure and ``Road``
boundary-nudge identically per flip while their **barrier integrals** differ
**11.4x** (5.10 vs 33.26-58.39), and registered F4: *"Ranking ``#766`` units by
barrier integral beats ranking by flip count at matched bytes"*, with the
pre-registered kill *"no separation -- the barrier is not the missing signal
either."*

``#766`` is ``experiments/ddm_wr1_reverse_waterfill.py``. Its unit is one of
**768 cells** (a 24x32 tiling of the 384x512 scorer plane at 16x16), and its
drop order is (``:93``)::

    order = np.lexsort((-residual_mass, flip_mass))   # safest-per-byte first

i.e. **flip count ascending is already the primary key** (``ddm_rs2``), with the
byte proxy only as a tie-break. F4 therefore asks whether replacing that primary
key with a barrier-weighted one *reorders the drop*.

This probe answers it scorer-free and end-to-end:

* per-cell ``flip_count``   -- the shipped key, recomputed on the LIVE cx1 n600
  flip mask (not the 2026-07-29 ``ru1`` atlas, which is a different snapshot);
* per-cell ``barrier_mass`` -- sum over that cell's flips of the hg1 barrier
  integral of the directed side ``gt_class -> cx1_class`` the flip belongs to;
* per-cell ``margin_mass``  -- sum of the GT-reference margin over the cell's
  flips (a per-site repair-proximity proxy, and the object that links hg1's
  barrier to the seg leg's margin hinge);
* the drop curves of all three keys against the SAME byte proxy
  (``residual_mass`` from ``wr1``'s own cell atlas), compared at matched bytes.

Controls that would expose a dead instrument:
* the barrier integrals are RE-DERIVED from the depth profiles and checked
  against hg1's published table rather than copied;
* the cell tiling is checked to cover the plane exactly once;
* a shuffled-key control gives the separation expected from noise alone.
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
HG1_PROFILE = REPO / ".omx/research/ddm_hg1_signed_depth_profile_n600.json"
WR1_ATLAS = Path("/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_cell_sensitivity_atlas.npz")

CELL_H, CELL_W = 16, 16
GRID_H, GRID_W = 24, 32          # 384/16, 512/16
N_CELLS = GRID_H * GRID_W        # 768, exactly wr1's unit count

# hg1's published barrier table (memo section 4) -- used ONLY as a control on the
# re-derivation, never as the input.
HG1_PUBLISHED = {
    "2->0": 51.81, "4->0": 55.29, "2->3": 37.91, "0->1": 51.26, "0->4": 58.39,
    "0->2": 33.26, "0->3": 49.51, "3->0": 55.04, "1->0": 5.10, "3->2": 51.31,
    "4->1": 58.17,
}


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy-free)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def _derive_barriers() -> tuple[dict[str, float], dict]:
    """Barrier integral = sum of mean_margin over the depth profile, re-derived."""
    payload = json.loads(HG1_PROFILE.read_text())
    profiles = payload["directed_profiles"]
    barriers: dict[str, float] = {}
    npx: dict[str, int] = {}
    for side, rec in profiles.items():
        pts = [p for p in rec["profile"] if p is not None]
        if not pts:
            continue
        # hg1's barrier TRUNCATES at the depth where the population falls below
        # 1% of the depth-1 population -- that truncation is the whole point of
        # the Lane->Road result (5.10). Summing all depths instead gives 12.48
        # for 1->0 and destroys the 11.4x spread F4 is about. Re-derived, not
        # copied; the control below checks it against hg1's published table.
        n1 = float(pts[0]["n"])
        barriers[side] = float(sum(p["mean_margin"] for p in pts if float(p["n"]) >= 0.01 * n1))
        npx[side] = int(rec["n_pixels"])
    check = []
    for side, published in HG1_PUBLISHED.items():
        got = barriers.get(side)
        check.append({
            "side": side,
            "published": published,
            "rederived": None if got is None else round(got, 4),
            "abs_diff": None if got is None else round(abs(got - published), 4),
        })
    worst = max((c["abs_diff"] for c in check if c["abs_diff"] is not None), default=None)
    return barriers, {
        "rows": check,
        "worst_abs_diff": worst,
        "agrees_within_0.01": (worst is not None and worst < 0.01),
        "formula": "barrier = sum(mean_margin_d for d where n_d >= 0.01 * n_depth1)",
        "n_pixels_by_side": npx,
    }


def _drop_curve(key: np.ndarray, residual: np.ndarray, damage: np.ndarray) -> dict:
    """wr1's drop rule under an arbitrary primary key.

    ``order = lexsort((-residual, key))`` -- safest-per-byte first. Returns the
    cumulative bytes freed and the cumulative DAMAGE admitted along that order.
    """
    order = np.lexsort((-residual, key))
    cum_bytes = np.cumsum(residual[order])
    cum_damage = np.cumsum(damage[order])
    return {"order": order, "cum_bytes": cum_bytes, "cum_damage": cum_damage}


def _damage_at_matched_bytes(curve: dict, budgets: np.ndarray) -> np.ndarray:
    """Damage admitted once ``budget`` byte-proxy units have been freed."""
    idx = np.searchsorted(curve["cum_bytes"], budgets, side="left")
    idx = np.clip(idx, 0, curve["cum_damage"].size - 1)
    return curve["cum_damage"][idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / ".omx/research/ddm_mg1_barrier_rerank_n600.json"))
    args = ap.parse_args()

    barriers, barrier_control = _derive_barriers()

    gt_arg = np.load(PU2_CACHE / "gt_argmax_n600.npy", mmap_mode="r")
    cx_arg = np.load(PU2_CACHE / "cx1_argmax_n600.npy", mmap_mode="r")
    npz = np.load(GT_CACHE)
    margins = npz["margins"]
    if margins.ndim == 4:
        margins = margins[:, 0]
    n = min(args.pairs, gt_arg.shape[0], margins.shape[0])
    written = n
    for p in range(n):
        if not np.asarray(gt_arg[p]).any() and not np.asarray(cx_arg[p]).any():
            written = p
            break
    truncated = n - written
    n = written

    h, w = margins.shape[1], margins.shape[2]
    if (h, w) != (GRID_H * CELL_H, GRID_W * CELL_W):
        raise SystemExit(f"plane {h}x{w} does not tile into {GRID_H}x{GRID_W} cells of {CELL_H}x{CELL_W}")
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cell_of_px = ((yy // CELL_H) * GRID_W + (xx // CELL_W)).ravel()
    tiling_control = {
        "cells": int(cell_of_px.max() + 1),
        "expected_cells": N_CELLS,
        "px_per_cell_min": int(np.bincount(cell_of_px, minlength=N_CELLS).min()),
        "px_per_cell_max": int(np.bincount(cell_of_px, minlength=N_CELLS).max()),
        "covers_plane_exactly_once": bool(
            np.bincount(cell_of_px, minlength=N_CELLS).sum() == h * w
            and np.bincount(cell_of_px, minlength=N_CELLS).min() == CELL_H * CELL_W
        ),
    }

    flip_count = np.zeros(N_CELLS, dtype=np.float64)
    barrier_mass = np.zeros(N_CELLS, dtype=np.float64)
    margin_mass = np.zeros(N_CELLS, dtype=np.float64)
    # Per-PIXEL accumulators (summed over pairs) so the same measurement can be
    # block-reduced to ANY cell grain afterwards. wr1's 16x16 grain must come
    # out of these identically -- that is the grain sweep's own control.
    flip_map = np.zeros(h * w, dtype=np.float64)
    barrier_map = np.zeros(h * w, dtype=np.float64)
    side_flips: dict[str, int] = {}
    side_missing = 0
    n_flip_total = 0

    for start in range(0, n, args.chunk):
        stop = min(start + args.chunk, n)
        g = np.asarray(gt_arg[start:stop], dtype=np.int64)
        c = np.asarray(cx_arg[start:stop], dtype=np.int64)
        m = np.asarray(margins[start:stop], dtype=np.float64)
        flip = g != c
        n_flip_total += int(flip.sum())
        for k in range(g.shape[0]):
            fk = flip[k].ravel()
            if not fk.any():
                continue
            cells = cell_of_px[fk]
            gk = g[k].ravel()[fk]
            ck = c[k].ravel()[fk]
            mk = m[k].ravel()[fk]
            flip_count += np.bincount(cells, minlength=N_CELLS)
            margin_mass += np.bincount(cells, weights=mk, minlength=N_CELLS)
            # barrier weight per flip = barrier of its directed side
            bw = np.zeros(cells.size, dtype=np.float64)
            for a in range(5):
                for b in range(5):
                    if a == b:
                        continue
                    sel = (gk == a) & (ck == b)
                    cnt = int(sel.sum())
                    if not cnt:
                        continue
                    side = f"{a}->{b}"
                    side_flips[side] = side_flips.get(side, 0) + cnt
                    # A side with no published depth profile gets NaN so the
                    # control below can COUNT what it dropped, rather than
                    # silently weighting it 0.
                    bw[sel] = barriers.get(side, np.nan)
            miss = int(np.isnan(bw).sum())
            side_missing += miss
            if miss:
                bw = np.nan_to_num(bw, nan=0.0)
            barrier_mass += np.bincount(cells, weights=bw, minlength=N_CELLS)
            px = np.flatnonzero(fk)
            np.add.at(flip_map, px, 1.0)
            np.add.at(barrier_map, px, bw)
        print(f"[mg1-F4] pairs {stop}/{n}", flush=True)

    atlas = np.load(WR1_ATLAS)
    residual = np.asarray(atlas["residual_mass"], dtype=np.float64)
    wr1_flip_mass = np.asarray(atlas["flip_mass"], dtype=np.float64)

    # --- rankings ---------------------------------------------------------
    rho_flip_barrier = _spearman(flip_count, barrier_mass)
    rho_flip_margin = _spearman(flip_count, margin_mass)
    rho_flip_wr1 = _spearman(flip_count, wr1_flip_mass)
    rho_flip_resid = _spearman(flip_count, residual)

    # --- matched-byte drop curves ----------------------------------------
    # DAMAGE is always measured in the barrier currency (hg1's claim), so the
    # question is: does ranking BY barrier admit less barrier damage than
    # ranking by flip count, at the same bytes freed?
    curve_flip = _drop_curve(flip_count, residual, barrier_mass)
    curve_barrier = _drop_curve(barrier_mass, residual, barrier_mass)
    rng = np.random.default_rng(args.seed)
    shuffled = flip_count.copy()
    rng.shuffle(shuffled)
    curve_shuf = _drop_curve(shuffled, residual, barrier_mass)

    total_bytes = float(residual.sum())
    budgets = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]) * total_bytes
    d_flip = _damage_at_matched_bytes(curve_flip, budgets)
    d_barrier = _damage_at_matched_bytes(curve_barrier, budgets)
    d_shuf = _damage_at_matched_bytes(curve_shuf, budgets)
    total_damage = float(barrier_mass.sum())

    rows = []
    for i, _b in enumerate(budgets):
        rows.append({
            "byte_budget_fraction": float(budgets[i] / total_bytes),
            "damage_flip_count_key": float(d_flip[i]),
            "damage_barrier_key": float(d_barrier[i]),
            "damage_shuffled_key_control": float(d_shuf[i]),
            "barrier_key_advantage_abs": float(d_flip[i] - d_barrier[i]),
            "barrier_key_advantage_pct_of_total": float(
                100.0 * (d_flip[i] - d_barrier[i]) / total_damage) if total_damage else None,
            "shuffle_control_penalty_pct_of_total": float(
                100.0 * (d_shuf[i] - d_flip[i]) / total_damage) if total_damage else None,
        })

    # order overlap along the drop
    overlaps = []
    for k in (32, 64, 128, 256, 384, 512):
        a = set(curve_flip["order"][:k].tolist())
        b = set(curve_barrier["order"][:k].tolist())
        overlaps.append({"first_k_cells": k, "jaccard": len(a & b) / len(a | b),
                         "overlap_fraction": len(a & b) / k})

    # --- grain law --------------------------------------------------------
    # F4's null at wr1's 16x16 grain could be an artifact of AGGREGATION: a
    # per-flip weight of bounded spread cannot reorder cells whose flip COUNTS
    # span three orders of magnitude. Block-reduce the per-pixel maps to finer
    # grains and find where (if anywhere) the barrier starts to reorder.
    fm2 = flip_map.reshape(h, w)
    bm2 = barrier_map.reshape(h, w)
    grain_rows = []
    for g in (1, 2, 4, 8, 16):
        gh, gw = h // g, w // g
        fc = fm2.reshape(gh, g, gw, g).sum(axis=(1, 3)).ravel()
        bmass = bm2.reshape(gh, g, gw, g).sum(axis=(1, 3)).ravel()
        nz = fc > 0
        spread = float(fc.max() / fc[nz].min()) if nz.any() else None
        # VACUITY GUARD: zero-flip cells are tied at 0 under BOTH keys (barrier
        # mass is a sum over flips), so any overlap statistic computed over the
        # full cell set is dominated by that tie block and reports ~100% no
        # matter what the keys do. At 1x1 the zero block is 78% of all cells.
        # The only cells where the keys CAN differ are the flip-bearing ones, so
        # every ranking statistic below is restricted to them and the
        # denominator is reported.
        fcz, bmz = fc[nz], bmass[nz]
        o_f = np.argsort(fcz, kind="mergesort")
        o_b = np.argsort(bmz, kind="mergesort")
        half = max(1, fcz.size // 2)
        grain_rows.append({
            "cell_px": g * g,
            "n_cells": int(fc.size),
            "zero_flip_cells": int((~nz).sum()),
            "flip_bearing_cells": int(nz.sum()),
            "mean_flips_per_nonzero_cell": float(fcz.mean()) if nz.any() else None,
            "flip_count_spread_max_over_min_nonzero": spread,
            "spearman_flip_vs_barrier_ALL_cells_TIE_INFLATED": _spearman(fc, bmass),
            "spearman_flip_vs_barrier_flip_bearing_only": _spearman(fcz, bmz),
            "first_half_drop_overlap_flip_bearing_only": float(
                len(set(o_f[:half].tolist()) & set(o_b[:half].tolist())) / half),
            "overlap_denominator_cells": int(half),
        })

    receipt = {
        "arm": "ddm_mg1",
        "probe": "F4 barrier-integral re-rank of the #766 waterfill",
        "git_head": _git_head(),
        "axis": "[macOS-CPU scorer-free advisory] score_claim=false promotable=false",
        "denominator": {
            "pairs": n, "unwritten_pairs_excluded": truncated,
            "plane": [h, w], "cells": N_CELLS,
            "flips_total_live_cx1": n_flip_total,
            "flips_in_wr1_2026_07_29_atlas": int(wr1_flip_mass.sum()),
            "note": "wr1's atlas is a DIFFERENT (2026-07-29 ru1) snapshot; the live "
                    "cx1 flip mask is recomputed here and used for all mg1 keys",
        },
        "control_barrier_rederivation": barrier_control,
        "control_cell_tiling": tiling_control,
        "control_sides_without_a_published_barrier": {
            "flips_dropped_to_zero_weight": side_missing,
            "fraction_of_flips": side_missing / n_flip_total if n_flip_total else None,
        },
        "side_flip_counts_live_cx1": dict(sorted(side_flips.items(), key=lambda kv: -kv[1])),
        "barriers_rederived": {k: round(v, 4) for k, v in sorted(barriers.items())},
        "rank_correlations_spearman": {
            "flip_count_vs_barrier_mass": rho_flip_barrier,
            "flip_count_vs_margin_mass": rho_flip_margin,
            "flip_count_vs_wr1_atlas_flip_mass": rho_flip_wr1,
            "flip_count_vs_residual_bytes": rho_flip_resid,
        },
        "key_spreads": {
            "flip_count_max_over_min_nonzero": float(
                flip_count.max() / flip_count[flip_count > 0].min()) if (flip_count > 0).any() else None,
            "flip_count_zero_cells": int((flip_count == 0).sum()),
            "barrier_spread_across_sides": float(
                max(barriers.values()) / min(barriers.values())) if barriers else None,
        },
        "grain_law": {
            "note": "grain 256 px MUST reproduce the 768-cell numbers above (control)",
            "rows": grain_rows,
        },
        "matched_byte_drop": rows,
        "drop_order_overlap": overlaps,
        "totals": {"total_byte_proxy": total_bytes, "total_barrier_damage": total_damage},
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in (
        "denominator", "control_barrier_rederivation", "control_cell_tiling",
        "rank_correlations_spearman", "key_spreads")}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
