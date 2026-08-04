# SPDX-License-Identifier: MIT
"""ddm_ph1 step 1 --- the F7 (TEMPORAL/PHASE) reach ceiling.

THE QUESTION.  ``ddm_pc2`` measured that phase-faithfulness is a **per-pair POSITIONAL
DOF** and that ``tokens_delta`` carries AMPLITUDE only.  That names a deficit but does not
size it.  This probe sizes it: **how much of the realized seg residual is removable by a
bounded-DOF positional (phase) correction, and how much requires shape/topology change?**
That fraction is family F7's reach ceiling; the complement is F7's hard wall.

WHY THIS IS EXACT AND $0 (no scorer pass).  The residual argmax field is recoverable
without re-running the scorer:

    R == G everywhere EXCEPT at the ru1 atlas flip rows, where R == realized_class.

``atlas_flat.npz`` stores every one of the 458,738 flips with ``(pair, y, x, gt_class,
realized_class)``, so ``R`` is reconstructed bit-exactly from the cached GT ``lstars`` plus
the atlas.  Positive control: the reconstruction must reproduce the atlas flip count and
d_seg exactly (absdiff 0), and it must FAIL if either input is perturbed (mutation check,
per ``si1`` --- a green that cannot go red is not a control).

THE EXACTNESS TRICK (this is what makes the sweep affordable and honest).  For a shift with
|dy|,|dx| <= RMAX, a pixel (y,x) can NEVER change its agreement state if ``R`` is constant
on the (2*RMAX+1) box around it AND R(y,x) == G(y,x).  Proof: R(y+dy,x+dx) == R(y,x) ==
G(y,x).  So the flip count under any such shift is fully determined by the "active band"
    M = (boxmax(R) != boxmin(R)) | (R != G)
and evaluating only on M is EXACT, not an approximation.  Every count this module reports
is a full-field count.

THE MODEL LADDER (each rung is a receiver that costs DOF, priced against W B/flip).
  * ``rung0``  no correction --- the measured baseline (must reproduce 458,738).
  * ``rungA``  ONE integer 2-vector per pair (global translation).      2 params/pair.
  * ``rungB``  ONE integer 2-vector per connected component of M.       2 params/component.
  * ``rungC``  the ORACLE per-pixel-neighbourhood bound: each pixel independently takes its
               best shift.  Not a shippable receiver --- it is the *floor* of the whole
               translation family, so ``rung0 - rungC`` is the family's absolute reach and
               everything left under rungC is SHAPE, unreachable by any translation model.

INSTRUMENT CAPACITY (LAW A --- a negative measures the instrument unless proven otherwise).
The sweep can only see translations of magnitude <= RMAX.  The module therefore reports
``frac_optima_on_boundary``: the fraction of fitted optima sitting on the |shift| == RMAX
rim.  If that fraction is material the ceiling is INSTRUMENT-scoped, not family-scoped, and
the module says so in its verdict rather than reporting a clean number.

GAUGE TEST (``sm1`` --- what does the metric read if the cure is applied and nothing
changes?).  If the residual were pure shape, every rung collapses onto rung0 and reach is
0.  If it were pure global phase, rungA reaches ~1.  The metric therefore distinguishes the
two hypotheses by construction rather than confirming whichever was assumed.

AXIS.  ``[macOS-CPU advisory]`` NON-PROMOTABLE.  score_claim=false.  This is a reduction
over cached arrays; it moves no pointer and fires no scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

SCHEMA = "ddm_ph1_phase_mass_reach.v1"
AXIS_TAG = "[macOS-CPU advisory]"

# The five comma10k canonical class indices, in the order the trained SegNet emits
# (CLAUDE.md: NEVER re-derive this by luma-sorting -- that gives the wrong order).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

# Scored lattice: 600 pairs x the SegNet argmax lattice (H=384, W=512).
N_PAIRS_FULL = 600
H, W_PIX = 384, 512
SCORED_PIXELS = N_PAIRS_FULL * H * W_PIX

# Seg<->rate exchange rate, B per conceded argmax flip.  Verified at source against
# tac.canonical_equations gap_decomposition_against_floor: 4 * rate_denominator /
# (600*512*384) = 4 * 37_545_489 / 117_964_800.  Passed in explicitly so a changed rate
# denominator (Catalog #812: evaluate.py sums upstream/videos/ DYNAMICALLY) cannot be
# silently inherited.
DEFAULT_RATE_DENOMINATOR_BYTES = 37_545_489


def seg_rate_exchange_bytes_per_flip(rate_denominator_bytes: int) -> float:
    """W: bytes whose rate cost equals one conceded argmax flip's seg cost."""
    return 4.0 * float(rate_denominator_bytes) / float(SCORED_PIXELS)


@dataclass(frozen=True)
class RungResult:
    """One rung of the translation ladder, with its DOF cost made explicit."""

    name: str
    flips: int
    dof_params: int
    reach_frac: float  # (rung0 - flips) / rung0
    d_seg: float
    delta_s_seg: float


def _sha256_head(path: Path, n_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()


def load_fields(lstars_path: Path, atlas_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (G, R, provenance).  R is reconstructed bit-exactly from G + atlas."""
    gt = np.load(lstars_path)
    if gt.dtype != np.uint8:
        gt = gt.astype(np.uint8)
    if gt.shape != (N_PAIRS_FULL, H, W_PIX):
        raise ValueError(f"lstars shape {gt.shape} != {(N_PAIRS_FULL, H, W_PIX)}")

    atlas = np.load(atlas_path)
    pair = atlas["pair"].astype(np.int64)
    yy = atlas["y"].astype(np.int64)
    xx = atlas["x"].astype(np.int64)
    realized = atlas["realized_class"].astype(np.uint8)
    gt_cls = atlas["gt_class"].astype(np.uint8)

    # Apparatus validity: the atlas's own gt_class must agree with the cached GT it claims
    # to be typed against.  If these disagree the two artifacts are from different runs and
    # nothing downstream is interpretable.
    gt_at_flips = gt[pair, yy, xx]
    n_gt_mismatch = int((gt_at_flips != gt_cls).sum())

    realized_field = gt.copy()
    realized_field[pair, yy, xx] = realized

    prov = {
        "lstars_path": str(lstars_path),
        "atlas_path": str(atlas_path),
        "atlas_sha256_head1MiB": _sha256_head(atlas_path),
        "n_atlas_rows": int(pair.size),
        "atlas_gt_class_vs_cached_lstars_mismatches": n_gt_mismatch,
        "atlas_y_min": int(yy.min()),
        "atlas_y_max": int(yy.max()),
        "atlas_x_min": int(xx.min()),
        "atlas_x_max": int(xx.max()),
    }
    return gt, realized_field, prov


def gt_row_band_structure(gt: np.ndarray) -> dict:
    """Measure how much of the lattice carries ANY task content.

    This is not decoration: if GT is a constant slab over a row range, that range pins the
    global phase (a global translation instantly manufactures flips at the slab seam), which
    is a mechanism-level constraint on the whole F7 family.
    """
    per_row_classes = [np.unique(gt[:, y, :]) for y in range(H)]
    varying = np.array([len(u) > 1 for u in per_row_classes])
    idx = np.flatnonzero(varying)
    first, last = (int(idx[0]), int(idx[-1])) if idx.size else (-1, -1)
    return {
        "first_varying_row": first,
        "last_varying_row": last,
        "n_varying_rows": int(varying.sum()),
        "frac_rows_varying": float(varying.mean()),
        "constant_top_class": int(per_row_classes[0][0]) if len(per_row_classes[0]) == 1 else -1,
        "constant_bottom_class": (
            int(per_row_classes[-1][0]) if len(per_row_classes[-1]) == 1 else -1
        ),
        "frac_pixels_in_varying_rows": float(varying.sum()) / float(H),
    }


def _active_mask(realized: np.ndarray, gt: np.ndarray, rmax: int) -> np.ndarray:
    """Pixels whose agreement state CAN change under some shift with |d| <= rmax.

    Exact: outside this mask R is locally constant AND already equal to G, so any such
    shift leaves it agreeing.
    """
    size = 2 * rmax + 1
    hi = ndimage.maximum_filter(realized, size=size, mode="nearest")
    lo = ndimage.minimum_filter(realized, size=size, mode="nearest")
    return (hi != lo) | (realized != gt)


def _shifted_gather(realized: np.ndarray, ys: np.ndarray, xs: np.ndarray, dy: int, dx: int):
    """R at edge-clamped shifted coordinates -- the boundary rule a real warp receiver uses."""
    yy = np.clip(ys + dy, 0, H - 1)
    xx = np.clip(xs + dx, 0, W_PIX - 1)
    return realized[yy, xx]


def _cell_normal_allowed(
    ys: np.ndarray,
    xs: np.ndarray,
    lab: np.ndarray,
    ncell: int,
    shifts: list[tuple[int, int]],
    tol_px: float,
) -> np.ndarray:
    """allowed[cell, shift]: restrict each cell's offset to its own boundary NORMAL.

    WHY THIS IS FREE TO ADDRESS.  The direction is computed from the boundary geometry of the
    DECODED field, which the receiver already holds -- so the encoder transmits only the
    scalar magnitude along it, not the 2-D vector.  That is the 'rank-into-receiver-field'
    idea applied to direction: the receiver derives WHERE-and-WHICH-WAY, the stream carries
    HOW-FAR.

    The tangent is the principal axis of the cell's band-pixel coordinates (a boundary is
    locally a curve, so PCA of its support recovers its direction); the normal is its
    perpendicular.  A shift is allowed when its TANGENTIAL component is under ``tol_px``.
    The zero shift is always allowed, so this rung can never score worse than 'no correction'.
    """
    del tol_px  # superseded: the allowed set is now the exact 1-DOF ladder round(m * normal)
    allowed = np.zeros((ncell, len(shifts)), dtype=bool)
    shift_index = {s: i for i, s in enumerate(shifts)}
    rmax = max(abs(s[0]) for s in shifts)
    order = np.argsort(lab, kind="stable")
    lab_s, ys_s, xs_s = lab[order], ys[order].astype(np.float64), xs[order].astype(np.float64)
    bounds = np.searchsorted(lab_s, np.arange(ncell + 1))
    zero_i = shift_index[(0, 0)]
    for c in range(ncell):
        a, b = bounds[c], bounds[c + 1]
        allowed[c, zero_i] = True
        if b - a < 2:
            continue
        cy, cx = ys_s[a:b], xs_s[a:b]
        cy = cy - cy.mean()
        cx = cx - cx.mean()
        # Principal axis via the 2x2 scatter matrix (closed form, no SVD needed).
        sxx, syy, sxy = float((cx * cx).sum()), float((cy * cy).sum()), float((cx * cy).sum())
        theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
        ty, tx = np.sin(theta), np.cos(theta)  # unit tangent
        ny, nx = -tx, ty  # unit normal
        # EXACTLY one scalar DOF: the offset is round(m * normal) for integer m.  This is the
        # honest 1-DOF alphabet (2*rmax+1 symbols), not an angular tolerance band -- the
        # earlier tolerance form rejected large oblique shifts and understated the rung.
        for m in range(-rmax, rmax + 1):
            key = (round(m * ny), round(m * nx))
            j = shift_index.get(key)
            if j is not None:
                allowed[c, j] = True
    return allowed


def _block_labels(ys: np.ndarray, xs: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Label band pixels by which KxK grid cell they fall in."""
    ncol = (W_PIX + k - 1) // k
    nrow = (H + k - 1) // k
    return (ys // k) * ncol + (xs // k), nrow * ncol


def sweep_pair(
    gt_p: np.ndarray,
    realized_p: np.ndarray,
    rmax: int,
    block_sizes: tuple[int, ...],
    normal_blocks: tuple[int, ...] = (),
    normal_tol_px: float = 0.75,
) -> dict:
    """Run the full translation sweep for one pair over every partition granularity.

    All partitions share ONE gather per shift (the dominant cost), so the whole ladder is
    measured in a single pass.  Every count is a full-field count (the band restriction is
    exact, see the module docstring).
    """
    mask = _active_mask(realized_p, gt_p, rmax)
    ys, xs = np.nonzero(mask)
    n_band = ys.size
    gt_band = gt_p[ys, xs]
    if n_band == 0:
        return {"n_band": 0, "rung0": 0, "partitions": {}, "edges_base": {}}

    # Partition ladder.  'component' = connected components of the active band (a region
    # whose boundary moves together); 'block<K>' = a fixed KxK grid, which is what a
    # shippable per-region offset stream would actually address.
    labels_cc, n_cc = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int64))
    partitions: dict[str, tuple[np.ndarray, int]] = {
        "component": (labels_cc[ys, xs].astype(np.int64), n_cc + 1)
    }
    for k in block_sizes:
        partitions[f"block{k}"] = _block_labels(ys, xs, k)
    # 'global' = one cell for the whole pair == the whole-frame rigid model gt2x refuted.
    partitions["global"] = (np.zeros(n_band, dtype=np.int64), 1)

    shifts = [(dy, dx) for dy in range(-rmax, rmax + 1) for dx in range(-rmax, rmax + 1)]
    zero_idx = shifts.index((0, 0))

    best: dict[str, np.ndarray] = {}
    best_shift_idx: dict[str, np.ndarray] = {}
    for name, (_lab, ncell) in partitions.items():
        best[name] = np.full(ncell, np.iinfo(np.int64).max, dtype=np.int64)
        best_shift_idx[name] = np.zeros(ncell, dtype=np.int32)

    # Full per-cell x per-shift tables, kept only for the block sizes that also get the
    # normal-restricted variant (the restriction is a mask over shifts, so it needs the table).
    full_tab: dict[str, np.ndarray] = {
        f"block{k}": np.zeros((len(shifts), partitions[f"block{k}"][1]), dtype=np.int64)
        for k in normal_blocks
        if f"block{k}" in partitions
    }

    pixel_always_flip = np.ones(n_band, dtype=bool)
    rung0 = -1
    for si, (dy, dx) in enumerate(shifts):
        got = _shifted_gather(realized_p, ys, xs, dy, dx)
        isflip = got != gt_band
        if si == zero_idx:
            rung0 = int(isflip.sum())
        pixel_always_flip &= isflip
        for name, (lab, ncell) in partitions.items():
            cur = np.bincount(lab[isflip], minlength=ncell) if isflip.any() else np.zeros(ncell, np.int64)
            if name in full_tab:
                full_tab[name][si] = cur
            better = cur < best[name]
            best_shift_idx[name][better] = si
            np.minimum(best[name], cur, out=best[name])

    shift_arr = np.array(shifts, dtype=np.int8)
    out_parts: dict[str, dict] = {}
    offsets: dict[str, np.ndarray] = {}
    for name, (lab, ncell) in partitions.items():
        occupied = np.bincount(lab, minlength=ncell) > 0
        n_occ = int(occupied.sum())
        chosen = best_shift_idx[name][occupied]
        sh = np.array([shifts[i] for i in chosen], dtype=np.int64) if n_occ else np.zeros((0, 2))
        on_rim = float((np.abs(sh).max(axis=1) == rmax).mean()) if n_occ else 0.0
        n_zero = int((np.abs(sh).max(axis=1) == 0).sum()) if n_occ else 0
        out_parts[name] = {
            "flips": int(best[name][occupied].sum()),
            "n_cells_occupied": n_occ,
            "frac_on_rim": on_rim,
            "frac_cells_choosing_zero_shift": (n_zero / n_occ) if n_occ else 0.0,
        }
        # The ACTUAL symbol stream a receiver would have to be told, in grid raster order.
        # Unoccupied cells carry the zero offset (nothing to correct there), which is what a
        # real encoder would emit and is what makes the stream compressible.
        if name.startswith("block"):
            full = np.zeros((ncell, 2), dtype=np.int8)
            full[occupied] = shift_arr[best_shift_idx[name][occupied]]
            offsets[name] = full

    # --- normal-restricted variant: 1 receiver-derived DOF per cell instead of 2 ----------
    # Estimate the normal from the TRUE argmax discontinuity pixels, not the rmax-dilated
    # band: the band is thickened by rmax, which destroys the local curve structure PCA needs
    # at 8-16 px cell scale (an 8 px cell's dilated band fills the cell and its principal axis
    # is noise).  Cells with too few true-boundary pixels keep the full 2-DOF alphabet rather
    # than being handed a fabricated direction.
    bnd = np.zeros_like(realized_p, dtype=bool)
    bnd[:-1, :] |= realized_p[:-1, :] != realized_p[1:, :]
    bnd[1:, :] |= realized_p[:-1, :] != realized_p[1:, :]
    bnd[:, :-1] |= realized_p[:, :-1] != realized_p[:, 1:]
    bnd[:, 1:] |= realized_p[:, :-1] != realized_p[:, 1:]
    on_bnd = bnd[ys, xs]
    for name, tab in full_tab.items():
        lab, ncell = partitions[name]
        allowed = _cell_normal_allowed(
            ys[on_bnd], xs[on_bnd], lab[on_bnd], ncell, shifts, normal_tol_px
        )
        masked = np.where(allowed.T, tab, np.iinfo(np.int64).max)
        occupied = np.bincount(lab, minlength=ncell) > 0
        n_occ = int(occupied.sum())
        out_parts[f"{name}_normal"] = {
            "flips": int(masked.min(axis=0)[occupied].sum()),
            "n_cells_occupied": n_occ,
            "frac_on_rim": 0.0,
            "frac_cells_choosing_zero_shift": 0.0,
            "mean_allowed_alphabet": float(allowed[occupied].sum(axis=1).mean()) if n_occ else 0.0,
        }

    # Per-EDGE baseline decomposition (m91: decompose per EDGE, never per class).
    base_flip = gt_band != realized_p[ys, xs]
    edges_base: dict[str, int] = {}
    if base_flip.any():
        g = gt_band[base_flip].astype(np.int64)
        r = realized_p[ys, xs][base_flip].astype(np.int64)
        for code, n in zip(*np.unique(g * 5 + r, return_counts=True), strict=True):
            edges_base[f"{CLASS_NAMES[code // 5]}->{CLASS_NAMES[code % 5]}"] = int(n)

    return {
        "n_band": int(n_band),
        "n_components": int(n_cc),
        "rung0": rung0,
        "rungC_per_pixel_oracle": int(pixel_always_flip.sum()),
        "partitions": out_parts,
        "edges_base": edges_base,
        "_offsets": offsets,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lstars", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--block-sizes", type=str, default="64,32,16,8,4")
    ap.add_argument("--normal-blocks", type=str, default="",
                    help="block sizes that also get the normal-restricted (1-DOF) rung")
    ap.add_argument("--normal-tol-px", type=float, default=0.75)
    ap.add_argument("--pairs", type=int, default=0, help="0 = all 600")
    ap.add_argument(
        "--selection-mode",
        required=True,
        choices=("all", "stratified", "prefix"),
        help="REQUIRED. 'prefix' is a DIFFERENT POPULATION (m88/m96) and is refused for verdicts.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-denominator-bytes", type=int, default=DEFAULT_RATE_DENOMINATOR_BYTES)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--dump-offsets",
        type=Path,
        default=None,
        help="npz path for the per-cell offset symbol streams (input to the real-coder race).",
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    gt, realized, prov = load_fields(args.lstars, args.atlas)
    band = gt_row_band_structure(gt)

    # ---- positive control + mutation check (si1) -------------------------------------
    flips_recon = int((gt != realized).sum())
    ctrl = {
        "reconstructed_flip_count": flips_recon,
        "atlas_flip_count": prov["n_atlas_rows"],
        "absdiff": abs(flips_recon - prov["n_atlas_rows"]),
        "d_seg_reconstructed": flips_recon / SCORED_PIXELS,
    }
    # Mutation check: perturb one pixel; the control MUST go red.
    mutated = realized.copy()
    py, px = band["first_varying_row"], 0
    mutated[0, py, px] = (mutated[0, py, px] + 1) % 5
    ctrl["mutation_check_flip_count"] = int((gt != mutated).sum())
    ctrl["mutation_check_went_red"] = ctrl["mutation_check_flip_count"] != flips_recon
    del mutated

    if args.pairs and args.pairs < N_PAIRS_FULL:
        if args.selection_mode == "prefix":
            sel = np.arange(args.pairs)
        else:
            rng = np.random.default_rng(args.seed)
            sel = np.sort(rng.choice(N_PAIRS_FULL, size=args.pairs, replace=False))
    else:
        sel = np.arange(N_PAIRS_FULL)

    block_sizes = tuple(int(k) for k in args.block_sizes.split(",") if k.strip())
    normal_blocks = tuple(int(k) for k in args.normal_blocks.split(",") if k.strip())
    per_pair: list[dict] = []
    offset_stacks: dict[str, list[np.ndarray]] = {f"block{k}": [] for k in block_sizes}
    for k, p in enumerate(sel):
        s = sweep_pair(gt[p], realized[p], args.rmax, block_sizes, normal_blocks, args.normal_tol_px)
        s["pair"] = int(p)
        for nm, arr in s.pop("_offsets", {}).items():
            offset_stacks[nm].append(arr)
        per_pair.append(s)
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{sel.size} pairs  ({time.time() - t0:.0f}s)", flush=True)

    if args.dump_offsets:
        args.dump_offsets.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump_offsets,
            **{nm: np.stack(v) for nm, v in offset_stacks.items() if v},
            pairs=sel.astype(np.int32),
        )

    rung0_tot = int(sum(d["rung0"] for d in per_pair))
    n_band_tot = int(sum(d["n_band"] for d in per_pair))
    scored = sel.size * H * W_PIX
    w = seg_rate_exchange_bytes_per_flip(args.rate_denominator_bytes)
    # Raw cost of ONE offset symbol from the sweep alphabet, before any entropy coding.
    # Reported alongside a coded estimate so the ladder is priced, not asserted.
    n_alpha = (2 * args.rmax + 1) ** 2
    raw_bytes_per_cell = np.log2(n_alpha) / 8.0

    part_names = (["global", "component"] + [f"block{k}" for k in block_sizes]
                  + [f"block{k}_normal" for k in normal_blocks])
    rungs: list[dict] = []
    for name in part_names:
        flips = int(sum(d["partitions"][name]["flips"] for d in per_pair))
        cells = int(sum(d["partitions"][name]["n_cells_occupied"] for d in per_pair))
        removed = rung0_tot - flips
        cost_raw = cells * raw_bytes_per_cell
        rungs.append(
            {
                "name": name,
                "flips": flips,
                "flips_removed": removed,
                "reach_frac": removed / rung0_tot if rung0_tot else 0.0,
                "n_cells": cells,
                "d_seg": flips / scored,
                "delta_s_seg": 100.0 * removed / scored,
                "raw_offset_bytes": cost_raw,
                "budget_bytes_at_W": removed * w,
                "pays_raw": bool(cost_raw < removed * w),
                "raw_bytes_per_flip_removed": (cost_raw / removed) if removed else float("inf"),
                "frac_cells_choosing_zero_shift": float(
                    np.mean([d["partitions"][name]["frac_cells_choosing_zero_shift"] for d in per_pair])
                ),
                "frac_on_rim": float(
                    np.mean([d["partitions"][name]["frac_on_rim"] for d in per_pair])
                ),
            }
        )
    oracle_flips = int(sum(d["rungC_per_pixel_oracle"] for d in per_pair))
    rungs.append(
        {
            "name": "per_pixel_oracle_NOT_SHIPPABLE",
            "flips": oracle_flips,
            "flips_removed": rung0_tot - oracle_flips,
            "reach_frac": (rung0_tot - oracle_flips) / rung0_tot if rung0_tot else 0.0,
            "n_cells": n_band_tot,
            "d_seg": oracle_flips / scored,
            "delta_s_seg": 100.0 * (rung0_tot - oracle_flips) / scored,
            "raw_offset_bytes": n_band_tot * raw_bytes_per_cell,
            "budget_bytes_at_W": (rung0_tot - oracle_flips) * w,
            "pays_raw": False,
            "raw_bytes_per_flip_removed": (
                n_band_tot * raw_bytes_per_cell / (rung0_tot - oracle_flips)
                if rung0_tot - oracle_flips
                else float("inf")
            ),
            "frac_cells_choosing_zero_shift": 0.0,
            "frac_on_rim": 0.0,
            "caveat": (
                "NEAR-TAUTOLOGICAL BOUND, not a lever: it asserts only that a correct-class "
                "pixel exists within rmax, which is implied by 93.9% of flips lying on a GT "
                "boundary (ru1 receipt). Its offset alphabet (log2(n_alpha) bits) is also "
                "WIDER than simply transmitting the class (log2(5)=2.32 bits), so it is "
                "dominated by direct class transmission at every pixel. Quoted as the "
                "translation family's ceiling, never as achievable headroom."
            ),
        }
    )

    edges_base: dict[str, int] = {}
    for d in per_pair:
        for kk, vv in d["edges_base"].items():
            edges_base[kk] = edges_base.get(kk, 0) + vv
    edges_sorted = dict(sorted(edges_base.items(), key=lambda kv: -kv[1]))

    rim_A = float(np.mean([d["partitions"]["global"]["frac_on_rim"] for d in per_pair]))
    rim_B = float(np.mean([d["partitions"]["component"]["frac_on_rim"] for d in per_pair]))

    receipt = {
        "schema": SCHEMA,
        "evidence_axis": AXIS_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "n_pairs": int(sel.size),
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "rmax": args.rmax,
        "provenance": prov,
        "gt_row_band_structure": band,
        "positive_control": ctrl,
        "instrument_capacity": {
            "rungA_frac_optima_on_rim": rim_A,
            "rungB_frac_optima_on_rim": rim_B,
            "note": (
                "optima on the |shift|==rmax rim are CENSORED by the sweep window; a material "
                "fraction makes the ceiling INSTRUMENT-scoped, not family-scoped (LAW A)."
            ),
        },
        "seg_rate_exchange_bytes_per_flip": w,
        "rate_denominator_bytes": args.rate_denominator_bytes,
        "raw_bytes_per_offset_symbol": raw_bytes_per_cell,
        "rung0_total_flips": rung0_tot,
        "rungs": rungs,
        "edges_baseline": edges_sorted,
        "per_pair": per_pair,
        "elapsed_sec": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1))

    print(f"\n== ddm_ph1 phase-mass reach ceiling  (n={sel.size}, mode={args.selection_mode}) ==")
    print(f"GT varying rows: {band['first_varying_row']}..{band['last_varying_row']} "
          f"({band['frac_rows_varying'] * 100:.1f}% of lattice)")
    print(f"positive control: recon={ctrl['reconstructed_flip_count']} "
          f"atlas={ctrl['atlas_flip_count']} absdiff={ctrl['absdiff']} "
          f"mutation_went_red={ctrl['mutation_check_went_red']}")
    print(f"{'partition':>28s} {'flips':>9s} {'reach%':>7s} {'cells':>8s} "
          f"{'rawB':>10s} {'budgetB':>10s} {'B/flip':>8s} pays?")
    for r in rungs:
        print(f"  {r['name']:>26s} {r['flips']:9d} {r['reach_frac'] * 100:7.2f} "
              f"{r['n_cells']:8d} {r['raw_offset_bytes']:10.0f} {r['budget_bytes_at_W']:10.0f} "
              f"{r['raw_bytes_per_flip_removed']:8.2f} {'YES' if r['pays_raw'] else 'no'}")
    print(f"instrument: global on-rim {rim_A * 100:.1f}%  component on-rim {rim_B * 100:.1f}%")
    print(f"top edges (baseline): "
          f"{list(edges_sorted.items())[:4]}")
    print(f"W = {w:.10f} B/flip   ->  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
