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
from dataclasses import asdict, dataclass
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


def sweep_pair(
    gt_p: np.ndarray, realized_p: np.ndarray, rmax: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run the full translation sweep for one pair.

    Returns (per-pair summary, per-component-per-shift flip counts, component sizes).
    """
    mask = _active_mask(realized_p, gt_p, rmax)
    ys, xs = np.nonzero(mask)
    n_band = ys.size
    gt_band = gt_p[ys, xs]

    # Connected components of the active band define the regional partition for rungB.
    # The partition is fixed at zero shift, so per-component minima compose into a valid
    # receiver (each component carries its own offset).
    labels, n_comp = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int64))
    lab_band = labels[ys, xs].astype(np.int64)
    comp_sizes = np.bincount(lab_band, minlength=n_comp + 1)

    shifts = [(dy, dx) for dy in range(-rmax, rmax + 1) for dx in range(-rmax, rmax + 1)]
    n_shift = len(shifts)
    comp_flips = np.zeros((n_shift, n_comp + 1), dtype=np.int64)
    total_flips = np.zeros(n_shift, dtype=np.int64)
    # rungC: per-pixel best over shifts.  Track a running per-pixel AND of "is a flip".
    pixel_always_flip = np.ones(n_band, dtype=bool)

    for si, (dy, dx) in enumerate(shifts):
        got = _shifted_gather(realized_p, ys, xs, dy, dx)
        isflip = got != gt_band
        total_flips[si] = int(isflip.sum())
        if isflip.any():
            comp_flips[si] = np.bincount(lab_band[isflip], minlength=n_comp + 1)
        pixel_always_flip &= isflip

    zero_idx = shifts.index((0, 0))
    rung0 = int(total_flips[zero_idx])
    a_idx = int(np.argmin(total_flips))
    rungA = int(total_flips[a_idx])
    best_per_comp = comp_flips.min(axis=0)
    rungB = int(best_per_comp[1:].sum())  # label 0 is background (outside band)
    rungC = int(pixel_always_flip.sum())

    a_dy, a_dx = shifts[a_idx]
    b_arg = comp_flips[:, 1:].argmin(axis=0) if n_comp else np.zeros(0, dtype=np.int64)
    b_shifts = np.array([shifts[i] for i in b_arg], dtype=np.int64) if n_comp else np.zeros((0, 2))

    def _on_rim(sh: np.ndarray) -> float:
        if sh.size == 0:
            return 0.0
        return float((np.abs(sh).max(axis=1) == rmax).mean())

    summary = {
        "n_band": int(n_band),
        "n_components": int(n_comp),
        "rung0": rung0,
        "rungA": rungA,
        "rungB": rungB,
        "rungC": rungC,
        "rungA_shift": [int(a_dy), int(a_dx)],
        "rungA_on_rim": bool(max(abs(a_dy), abs(a_dx)) == rmax),
        "rungB_frac_on_rim": _on_rim(b_shifts),
    }
    return summary, b_shifts, comp_sizes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lstars", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--rmax", type=int, default=5)
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

    per_pair: list[dict] = []
    for k, p in enumerate(sel):
        s, _bsh, _cs = sweep_pair(gt[p], realized[p], args.rmax)
        s["pair"] = int(p)
        per_pair.append(s)
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{sel.size} pairs  ({time.time() - t0:.0f}s)", flush=True)

    tot = {k: int(sum(d[k] for d in per_pair)) for k in ("rung0", "rungA", "rungB", "rungC")}
    n_band_tot = int(sum(d["n_band"] for d in per_pair))
    n_comp_tot = int(sum(d["n_components"] for d in per_pair))
    scored = sel.size * H * W_PIX
    w = seg_rate_exchange_bytes_per_flip(args.rate_denominator_bytes)

    def mk(name: str, flips: int, dof: int) -> RungResult:
        d_seg = flips / scored
        return RungResult(
            name=name,
            flips=flips,
            dof_params=dof,
            reach_frac=(tot["rung0"] - flips) / tot["rung0"] if tot["rung0"] else 0.0,
            d_seg=d_seg,
            delta_s_seg=100.0 * (tot["rung0"] - flips) / scored,
        )

    rungs = [
        mk("rung0_no_correction", tot["rung0"], 0),
        mk("rungA_global_per_pair", tot["rungA"], 2 * sel.size),
        mk("rungB_per_component", tot["rungB"], 2 * n_comp_tot),
        mk("rungC_per_pixel_oracle", tot["rungC"], 2 * n_band_tot),
    ]

    rim_A = float(np.mean([d["rungA_on_rim"] for d in per_pair]))
    rim_B = float(np.mean([d["rungB_frac_on_rim"] for d in per_pair]))

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
        "rungs": [asdict(r) for r in rungs],
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
    for r in rungs:
        print(f"  {r.name:26s} flips={r.flips:9d}  reach={r.reach_frac * 100:6.2f}%  "
              f"dof={r.dof_params:8d}  dS_seg={r.delta_s_seg:+.5f}")
    print(f"instrument: rungA on-rim {rim_A * 100:.1f}%  rungB on-rim {rim_B * 100:.1f}%")
    print(f"W = {w:.10f} B/flip   ->  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
