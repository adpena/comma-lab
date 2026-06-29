#!/usr/bin/env python3
"""Measure the ORACLE polynomial lane-shape FLOOR for the frozen-SegNet lane class.

Context (see CLAUDE.md "THE CURRENT FRONTIER ... WITNESS CAPSTONE" + the v2 grok):
the contest score is ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``.
``d_seg`` is the per-frame disagreement rate of a FROZEN SegNet argmax (5 classes,
canonical comma10k order ``[Road, Lane, Undrivable, Movable, MyCar]`` -> Lane is
class index 1). The v2 binding term is the LANE-SURVIVAL residual: F4 reduced
"sub-0.15" to ``trained-through-R lane d_seg <= 1.23e-3`` (and "sub-0.19" to
``<= 1.63e-3``).

The operator hypothesis under test: openpilot represents lanes natively as
degree-4 polynomials (``POLY_PATH_DEGREE = 4``, source-confirmed in
``.omx/research/comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md``),
so a handful of polynomial coefficients per lane line might reproduce the frozen
SegNet's argmax-lane class cheaply.

This tool measures the NECESSARY-CONDITION FLOOR for that hypothesis, at $0, with
NO openpilot and NO GPU. It fits low-order polynomials DIRECTLY to the cached
frozen-SegNet lane argmax (``lstars`` in the GT cache) -- i.e. an ORACLE fit, the
BEST any polynomial-shape carrier could do. It then measures the symmetric
disagreement (XOR) between the oracle-polynomial lane mask and the true SegNet
lane mask, which is EXACTLY the lane contribution to ``d_seg`` under perfect
substitution of the polynomial lane for the true lane.

Interpretation (NO-FAKE, advisory):
  * The result is a FLOOR: openpilot's own lane model can only be WORSE (it
    predicts the physical lane, which may disagree with SegNet's argmax), and any
    quantized/byte-limited carrier can only be worse than the oracle.
  * The result is SHAPE-ONLY, NOT through-R. The R operator
    (render -> bicubic up 874 -> uint8 -> bilinear down 384 -> argmax) adds a
    separate SURVIVAL/texture wall on thin lane strokes (the eval-roundtrip memo).
    A passing shape floor is NECESSARY but NOT SUFFICIENT for sub-0.15.
  * If the oracle floor is ALREADY above 1.23e-3, the polynomial-shape approach is
    dead regardless of openpilot -- a decisive negative.

Authority: [macOS research-signal] / advisory. score_claim=false. promotable=false.
Pointer UNMOVED (contest-CPU 0.19110). This is a MEANS measurement, not a score.

Usage:
    .venv/bin/python tools/measure_lane_polynomial_shape_floor.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n96.npz \
        --degree 4 --json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage  # available (scipy 1.17.1)

LANE_CLASS = 1  # canonical comma10k order [Road, Lane, Undrivable, Movable, MyCar]
SUB015_LANE_DSEG = 1.23e-3  # F4: sub-0.15 <=> trained-through-R lane d_seg <= this
SUB019_LANE_DSEG = 1.63e-3  # F4: sub-0.19 <=> trained-through-R lane d_seg <= this


@dataclass
class FloorResult:
    n_frames: int
    grid_h: int
    grid_w: int
    degree: int
    min_component_pixels: int
    dash_bridge_rows: int
    lane_area_frac_mean: float  # = d_seg if ALL lane pixels were lost ("drop lanes")
    d_seg_lane_floor_xor: float  # oracle polynomial-shape floor (the headline)
    iou_mean: float  # IoU(true lane, oracle poly lane)
    centerline_residual_px_median: float  # median |lane pixel - fitted curve| (px)
    components_per_frame_mean: float
    coeffs_per_frame_mean: float
    bytes_per_frame_est: float  # rough COUNTED-byte estimate for the poly carrier
    bytes_600_frames_est: float
    passes_sub015: bool
    passes_sub019: bool


def _fit_component(
    rows: np.ndarray, cols: np.ndarray, degree: int
) -> tuple[np.ndarray, bool, int]:
    """Fit a polynomial to one lane component.

    Returns (coeffs, fit_is_col_of_row, eff_degree). If row-extent >= col-extent
    we fit ``col = poly(row)`` (near-vertical strokes, the common case); otherwise
    ``row = poly(col)`` (near-horizontal strokes, e.g. distant lanes by the
    vanishing point). Degree is capped by the number of distinct samples.
    """
    row_ext = rows.max() - rows.min()
    col_ext = cols.max() - cols.min()
    fit_col_of_row = row_ext >= col_ext
    if fit_col_of_row:
        x, y = rows.astype(np.float64), cols.astype(np.float64)
    else:
        x, y = cols.astype(np.float64), rows.astype(np.float64)
    eff_deg = int(min(degree, max(1, np.unique(x).size - 1)))
    coeffs = np.polyfit(x, y, eff_deg)
    return coeffs, fit_col_of_row, eff_deg


def _rasterize_component(
    coeffs: np.ndarray,
    fit_col_of_row: bool,
    x_min: int,
    x_max: int,
    half_width: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Rasterize a fitted lane curve onto a fresh (H, W) bool mask at half_width."""
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    xs = np.arange(x_min, x_max + 1)
    ys = np.polyval(coeffs, xs.astype(np.float64))
    ys = np.rint(ys).astype(np.int64)
    for x, y in zip(xs, ys):
        lo, hi = y - half_width, y + half_width
        if fit_col_of_row:  # x is row, y is col
            if 0 <= x < grid_h:
                c0, c1 = max(0, lo), min(grid_w - 1, hi)
                if c0 <= c1:
                    mask[x, c0 : c1 + 1] = True
        else:  # x is col, y is row
            if 0 <= x < grid_w:
                r0, r1 = max(0, lo), min(grid_h - 1, hi)
                if r0 <= r1:
                    mask[r0 : r1 + 1, x] = True
    return mask


def measure(
    lstars: np.ndarray,
    degree: int = 4,
    min_component_pixels: int = 12,
    dash_bridge_rows: int = 25,
    max_half_width: int = 4,
) -> FloorResult:
    """Measure the oracle polynomial lane-shape floor over all frames in lstars."""
    n, gh, gw = lstars.shape
    total_px = gh * gw

    xor_fracs: list[float] = []
    ious: list[float] = []
    lane_fracs: list[float] = []
    residuals: list[float] = []
    comp_counts: list[int] = []
    coeff_counts: list[int] = []

    # vertical structuring element to bridge dashed lane segments of the same line
    bridge = np.ones((dash_bridge_rows, 1), dtype=bool)

    for i in range(n):
        lane = lstars[i] == LANE_CLASS
        lane_fracs.append(float(lane.mean()))
        if lane.sum() == 0:
            xor_fracs.append(0.0)
            ious.append(1.0)
            comp_counts.append(0)
            coeff_counts.append(0)
            continue

        # bridge dashes, then label connected components of the bridged mask
        bridged = ndimage.binary_closing(lane, structure=bridge)
        labels, nlab = ndimage.label(bridged)

        poly_lane = np.zeros_like(lane)
        n_comp = 0
        n_coeff = 0
        for lab in range(1, nlab + 1):
            comp_region = labels == lab
            # fit to the ORIGINAL (unbridged) lane pixels inside this component
            comp_lane = lane & comp_region
            rr, cc = np.where(comp_lane)
            if rr.size < min_component_pixels:
                continue
            coeffs, fcor, eff_deg = _fit_component(rr, cc, degree)
            n_comp += 1
            n_coeff += eff_deg + 1

            # centerline residual diagnostic
            if fcor:
                pred = np.polyval(coeffs, rr.astype(np.float64))
                residuals.append(float(np.median(np.abs(pred - cc))))
                x_min, x_max = int(rr.min()), int(rr.max())
            else:
                pred = np.polyval(coeffs, cc.astype(np.float64))
                residuals.append(float(np.median(np.abs(pred - rr))))
                x_min, x_max = int(cc.min()), int(cc.max())

            # ORACLE half-width: pick width minimizing this component's local XOR
            best_hw, best_xor = 0, None
            best_mask = None
            # local bbox for fair per-component XOR
            r0, r1 = int(rr.min()), int(rr.max())
            c0, c1 = int(cc.min()), int(cc.max())
            pad = max_half_width + 1
            br0, br1 = max(0, r0 - pad), min(gh, r1 + pad + 1)
            bc0, bc1 = max(0, c0 - pad), min(gw, c1 + pad + 1)
            true_local = comp_lane[br0:br1, bc0:bc1]
            for hw in range(0, max_half_width + 1):
                m = _rasterize_component(coeffs, fcor, x_min, x_max, hw, gh, gw)
                local = m[br0:br1, bc0:bc1]
                xv = int(np.logical_xor(true_local, local).sum())
                if best_xor is None or xv < best_xor:
                    best_xor, best_hw, best_mask = xv, hw, m
            poly_lane |= best_mask

        comp_counts.append(n_comp)
        coeff_counts.append(n_coeff)

        xor = np.logical_xor(lane, poly_lane)
        xor_fracs.append(float(xor.sum()) / total_px)
        inter = np.logical_and(lane, poly_lane).sum()
        union = np.logical_or(lane, poly_lane).sum()
        ious.append(float(inter) / float(union) if union else 1.0)

    # byte estimate for the COUNTED polynomial carrier (per frame):
    #   per component: (eff_deg+1) coeffs @ ~12 bits + 1 half-width nibble (4 bits)
    #   + 2 endpoints @ ~9 bits each. Rough entropy-coded estimate.
    coeffs_pf = float(np.mean(coeff_counts))
    comps_pf = float(np.mean(comp_counts))
    bits_pf = coeffs_pf * 12.0 + comps_pf * (4.0 + 18.0)
    bytes_pf = bits_pf / 8.0

    d_seg_floor = float(np.mean(xor_fracs))
    return FloorResult(
        n_frames=n,
        grid_h=gh,
        grid_w=gw,
        degree=degree,
        min_component_pixels=min_component_pixels,
        dash_bridge_rows=dash_bridge_rows,
        lane_area_frac_mean=float(np.mean(lane_fracs)),
        d_seg_lane_floor_xor=d_seg_floor,
        iou_mean=float(np.mean(ious)),
        centerline_residual_px_median=(
            float(np.median(residuals)) if residuals else 0.0
        ),
        components_per_frame_mean=comps_pf,
        coeffs_per_frame_mean=coeffs_pf,
        bytes_per_frame_est=bytes_pf,
        bytes_600_frames_est=bytes_pf * 600.0,
        passes_sub015=d_seg_floor <= SUB015_LANE_DSEG,
        passes_sub019=d_seg_floor <= SUB019_LANE_DSEG,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gt-cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz",
        help="path to GT cache npz containing 'lstars' (frozen SegNet argmax).",
    )
    ap.add_argument("--degree", type=int, default=4, help="max polynomial degree.")
    ap.add_argument("--min-component-pixels", type=int, default=12)
    ap.add_argument("--dash-bridge-rows", type=int, default=25)
    ap.add_argument("--max-half-width", type=int, default=4)
    ap.add_argument(
        "--sweep-degree",
        action="store_true",
        help="also report degrees 1..4 to show sensitivity.",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON.")
    args = ap.parse_args()

    d = np.load(args.gt_cache)
    if "lstars" not in d:
        raise SystemExit(f"{args.gt_cache} has no 'lstars' key; keys={list(d.keys())}")
    lstars = d["lstars"]

    degrees = [1, 2, 3, 4] if args.sweep_degree else [args.degree]
    results = {}
    for deg in degrees:
        r = measure(
            lstars,
            degree=deg,
            min_component_pixels=args.min_component_pixels,
            dash_bridge_rows=args.dash_bridge_rows,
            max_half_width=args.max_half_width,
        )
        results[deg] = r

    if args.json:
        print(
            json.dumps(
                {
                    "gt_cache": args.gt_cache,
                    "lane_class_index": LANE_CLASS,
                    "sub015_lane_dseg_threshold": SUB015_LANE_DSEG,
                    "sub019_lane_dseg_threshold": SUB019_LANE_DSEG,
                    "authority": "[macOS research-signal] advisory; score_claim=false",
                    "results_by_degree": {str(k): asdict(v) for k, v in results.items()},
                },
                indent=2,
            )
        )
        return

    for deg, r in results.items():
        print(f"\n=== degree {deg} (oracle polynomial lane-shape floor) ===")
        print(f"  frames                       : {r.n_frames} @ {r.grid_h}x{r.grid_w}")
        print(f"  lane area frac (drop-lanes d_seg): {r.lane_area_frac_mean:.6f}")
        print(f"  d_seg lane FLOOR (oracle XOR): {r.d_seg_lane_floor_xor:.6f}")
        print(f"  IoU(true, poly) mean         : {r.iou_mean:.4f}")
        print(f"  centerline residual (px med) : {r.centerline_residual_px_median:.3f}")
        print(f"  components/frame mean         : {r.components_per_frame_mean:.2f}")
        print(f"  coeffs/frame mean            : {r.coeffs_per_frame_mean:.2f}")
        print(f"  COUNTED bytes/frame (est)    : {r.bytes_per_frame_est:.1f}")
        print(f"  COUNTED bytes/600 (est)      : {r.bytes_600_frames_est:.0f}")
        print(f"  sub-0.15 floor (<= {SUB015_LANE_DSEG:.2e}) : {r.passes_sub015}")
        print(f"  sub-0.19 floor (<= {SUB019_LANE_DSEG:.2e}) : {r.passes_sub019}")
    print(
        "\nNOTE: FLOOR is SHAPE-ONLY (no R) + ORACLE (fit to target). Through-R "
        "survival is a SEPARATE wall. openpilot's own lanes can only be WORSE. "
        "Advisory; pointer UNMOVED 0.19110."
    )


if __name__ == "__main__":
    main()
