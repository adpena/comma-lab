#!/usr/bin/env python3
"""POLYNOMIAL-FILL SURVIVAL PROBE — does a CONTINUOUS per-region polynomial gradient fill
beat the FLAT-COLOUR survival wall that capped the curve-core, and at what byte cost?

THE SYNTHESIS of the operator directive (polynomials + polytopes + photoshop-layers): instead
of painting each class-region a FLAT colour (the survival wall: realized d_seg floored ~0.0067
because EfficientNet-B2 keys on TEXTURE, so a flat region lands OUTSIDE the per-pixel argmax
polytope at boundary pixels), paint each connected region with a CONTINUOUS 2D POLYNOMIAL per
RGB channel:

    c(x, y) = sum_{i+j<=k} a_ij * x^i * y^j          (a per-region photoshop "gradient fill")

fit the coefficients by LEAST-SQUARES to the GT RGB within the region (the ORACLE fit: "the best
a polynomial of this order can do"). Sweep order k=0..6:
  - k=0 == constant == per-region mean colour == the FLAT-COLOUR survival-wall baseline (MUST
    reproduce realized d_seg ~0.0067 as a harness sanity check);
  - k=1 == linear/planar gradient, k=2 quadratic, k=3 cubic, ... up to k=6 (or saturation).

THE DECISIVE QUESTION:
  Does realized d_seg drop from ~0.0067 (k=0 flat) toward ~0.0006 (frontier) as k rises, and is
  there a (k, rate) with realized < 0.0012 AND rate < 0.05 AND S < 0.15?
    GREEN -> a cheap continuous-polynomial representation beats the survival wall -> a real sub-0.15
             path (photoshop-polynomial vehicle); spec it.
    AMBER -> d_seg drops meaningfully but needs high k (expensive); quantify the order<->d_seg<->byte
             curve (the partial win + where it plateaus).
    RED   -> even high-order polynomial fill caps near 0.0067 -> the texture SegNet needs is NOT
             low-order-polynomial (it is HF/non-smooth) -> only a full learned texture-decoder
             survives; the continuous-representation family is closed.

WHY THIS IS FAITHFUL (NO-FAKE, measurement-first):
  - REAL frozen contest SegNet (CPU AUTHORITY; NEVER MPS for the score). This probe needs NO
    gradient -- the polynomial fit is a closed-form linear least-squares on CPU -- so NO MPS at all.
  - REAL GT argmax L* (the `seg` field of the capstone GT cache = SegNet argmax on GT frame1).
  - The EXACT eval roundtrip (camera-res bicubic-874 -> bilinear-384 -> round) inside the realized
    d_seg measurement (the survival check the flat fill FAILED).
  - realized d_seg is the AUTHORITY: argmax-flip-rate of the HARD polynomial-filled frame THROUGH
    the roundtrip + real SegNet vs L*. The CE/residual fit is a diagnostic, NEVER the verdict.
  - Guards BOTH false-GREEN (the factored-LF degenerate-fit + flat-NCA cautionary tales: a low
    fit-loss is not a low realized d_seg) AND false-RED (k=0 MUST reproduce the known flat floor).

This SUBSUMES a continuous-texture NCA re-run: it maps the WHOLE order<->d_seg<->bytes curve
interpretably (a polynomial of order k is the canonical smooth continuous-texture family; if even
k=6 caps at the wall, the continuous-SMOOTH family is closed and only a learned HF decoder survives).

Connects to tasks #137/#138 (road<->lane / openpilot geometric prior): a per-region gradient fill
is exactly the geometric-prior representation those tasks want; this gate measures whether it
survives the scorer at byte-cheap order.

ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.
$0, CPU only, no paid GPU, no PR; this probe must never promote itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(UPSTREAM))

GT_TARGETS = REPO / "experiments/results/capstone_gt_targets_cache/gt_targets_n16.pt"

# --- campaign measured anchors (advisory, for the GREEN/AMBER/RED thresholds) ----
FRONTIER_DSEG = 0.00257  # the frontier's own d_seg (the bar to beat on the d_seg axis)
SUB015_DSEG = 0.0006  # ~sub-0.15-grade d_seg (the target the fill must approach)
FLAT_COLOUR_SURVIVAL_WALL_DSEG = 0.00673  # the curve-core's k=0-equivalent flat floor (mp128)
GREEN_DSEG_THRESHOLD = 0.0012  # decisively past the frontier d_seg, heading sub-0.15
B0 = 37_545_489  # contest archive normalizer
HELD_POSE = 0.00034  # frontier trunk d_pose (held; the fill targets seg only)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# Polynomial design matrix + least-squares fill
# ===========================================================================
def n_coeffs(k: int) -> int:
    """Number of 2D polynomial coefficients of total degree <= k: (k+1)(k+2)/2."""
    return (k + 1) * (k + 2) // 2


def _poly_exponents(k: int):
    """The (i, j) exponent pairs for a 2D polynomial of total degree <= k, in a stable order."""
    exps = []
    for deg in range(k + 1):
        for i in range(deg + 1):
            j = deg - i
            exps.append((i, j))
    return exps  # len == n_coeffs(k)


def _design_matrix(rows, cols, k, H, W):
    """Vandermonde-style design matrix for normalized coords. rows/cols are int pixel coords.

    Coords are normalized to [-1, 1] over the FULL frame (not per-region) so the polynomial
    basis is well-conditioned and the same basis is shared across regions (a clean, stable
    photoshop-gradient parameterization). Returns (n_pix, n_coeffs(k)) float64.
    """
    import numpy as np

    x = (cols.astype(np.float64) / (W - 1)) * 2.0 - 1.0  # [-1, 1]
    y = (rows.astype(np.float64) / (H - 1)) * 2.0 - 1.0  # [-1, 1]
    exps = _poly_exponents(k)
    cols_list = []
    for i, j in exps:
        cols_list.append((x**i) * (y**j))
    return np.stack(cols_list, axis=1)  # (n_pix, n_coeffs)


def polynomial_fill_frame(L, frame_rgb, k, n_classes=5, min_region_px=1):
    """Build the polynomial-gradient-filled RGB frame (H, W, 3) for the GT argmax partition L*.

    For EACH connected region of L*, fit a per-RGB-channel 2D polynomial of order k by ordinary
    least-squares to the GT RGB inside the region, then evaluate it over the region's pixels.
    k=0 reduces to the per-region MEAN colour (== the flat-colour survival-wall baseline).

    Returns (filled_rgb_HW3 float, total_coeffs, n_regions, mean_residual). total_coeffs is the
    byte driver (n_regions * n_coeffs(k) * 3 channels). mean_residual is the per-pixel RGB L2
    residual of the polynomial fit (diagnostic: how much GT texture order-k captures).
    """
    import numpy as np

    from tac.boundary_math.partition import connected_components

    H, W = L.shape
    region_of, regions = connected_components(L, n_classes=n_classes)
    filled = np.zeros((H, W, 3), dtype=np.float64)
    nc = n_coeffs(k)
    total_coeffs = 0
    n_reg = 0
    sq_resid_sum = 0.0
    n_pix_total = 0

    for _rid, reg in regions.items():
        rows = reg.coords[0]
        cols = reg.coords[1]
        npix = rows.size
        gt = frame_rgb[rows, cols]  # (npix, 3) GT colours inside the region

        if npix < min_region_px:
            # degenerate: paint the region's mean colour, count 1 coeff (constant) * 3.
            mean = gt.mean(axis=0) if npix > 0 else np.zeros(3)
            filled[rows, cols] = mean
            total_coeffs += 1 * 3
            n_reg += 1
            continue

        if k == 0 or npix < nc:
            # constant fill (or under-determined: fewer pixels than coeffs -> drop to mean).
            mean = gt.mean(axis=0)
            filled[rows, cols] = mean
            total_coeffs += 1 * 3  # constant == 1 coeff per channel
            pred = np.broadcast_to(mean, gt.shape)
        else:
            A = _design_matrix(rows, cols, k, H, W)  # (npix, nc)
            # least-squares per channel (rcond trims tiny singular values -> robust to a
            # rank-deficient region, e.g. a thin/collinear strip where high-order monomials
            # are degenerate). solve all 3 channels at once.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                coef, _res, _rank, _sv = np.linalg.lstsq(A, gt, rcond=1e-8)  # (nc, 3)
                pred = A @ coef  # (npix, 3)
            # NUMERICAL GUARD (NO-FAKE): a degenerate fit can produce non-finite / wildly
            # out-of-range predictions; that is a fit failure, not a real fill -> fall back to
            # the per-region mean (the k=0 colour) so the measured d_seg reflects a VALID frame,
            # never NaN garbage. (On the real 384x512 regions this rarely fires; it makes the
            # toy/thin-region cases honest.)
            mean = gt.mean(axis=0)
            bad = ~np.isfinite(pred).all(axis=1)
            if bad.any():
                pred[bad] = mean
            # an absurd extrapolation outside a generous colour range is also a fit failure
            absurd = (np.abs(pred) > 1e4).any(axis=1)
            if absurd.any():
                pred[absurd] = mean
            filled[rows, cols] = pred
            total_coeffs += nc * 3

        sq_resid_sum += float(np.sum((pred - gt) ** 2))
        n_pix_total += npix
        n_reg += 1

    filled = np.clip(filled, 0.0, 255.0)
    mean_residual = math.sqrt(sq_resid_sum / max(1, n_pix_total * 3))
    return filled, int(total_coeffs), int(n_reg), float(mean_residual)


# ===========================================================================
# EXACT eval chain (mirror the curve gate / driver.py)
# ===========================================================================
def _eval_roundtrip_t(frame_chw, ste=False):
    """uint8 eval roundtrip on a (3,384,512) float frame: bicubic-up 874x1164 ->
    bilinear-down 384x512 -> clamp -> round. (ste kept for parity; this probe uses ste=False.)"""
    import torch.nn.functional as F

    x = frame_chw.unsqueeze(0)  # (1,3,384,512)
    up = F.interpolate(x, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    clamped = down.clamp(0, 255)
    if ste:
        rounded = clamped.round()
        return clamped + (rounded - clamped).detach()
    return clamped.round()


def _segnet_argmax_of_frame(segnet, frame_chw):
    """Real SegNet argmax (H,W) of a (3,384,512) float[0,255] frame, through the exact
    preprocess (degenerate pair, last frame == this frame)."""
    import torch

    pair = torch.stack([frame_chw, frame_chw], dim=0).unsqueeze(0)  # (1,2,3,384,512)
    seg_in = segnet.preprocess_input(pair)
    logits = segnet(seg_in)
    return logits.argmax(dim=1)[0]


# ===========================================================================
# Byte cost of the polynomial coeffs
# ===========================================================================
def polynomial_param_bytes(total_coeffs, coeff_bits=12):
    """Advisory byte cost of the quantized polynomial coeffs.

    - total_coeffs = n_regions * n_coeffs(k) * 3 channels (already counts the 3 RGB channels).
      Higher-order coeffs have smaller dynamic range but we use a uniform conservative bit budget.
    - coeff_bits: 12 bits/coeff (the constant term needs ~8b for [0,255]; gradient coeffs need a
      few more for fractional slopes -- 12b is a conservative uniform budget, entropy-coded down).
    - the geometry (the partition itself) is NOT counted here: this probe uses the GT ORACLE
      partition to ISOLATE the fill question (per the directive: partition (i) = oracle geometry).
      `fill_only` is reported explicitly so the framing is honest.

    Across 600 frames the partition+coeffs are quasi-static (geometric-solve identity residual
    ~0.33px); we report BOTH the per-frame full cost AND the quasi-static amortized cost
    (gamma0 once + small per-frame delta), matching the curve gate's byte model.
    """
    packed_factor = 0.55  # advisory entropy-coding factor (dense-raster LZMA measured family)
    per_frame_bytes = total_coeffs * coeff_bits * packed_factor / 8.0
    delta_frac = 0.10  # quasi-static per-frame delta (near-static partition)
    # total over 600: gamma0 stored once (full) + 10% per-frame delta * 600
    total_600_full = per_frame_bytes * 600.0
    total_600_amort = per_frame_bytes + per_frame_bytes * delta_frac * 600.0
    return {
        "total_coeffs": int(total_coeffs),
        "coeff_bits": coeff_bits,
        "per_frame_bytes_full": per_frame_bytes,
        "total_600_full_bytes": total_600_full,
        "total_600_amortized_bytes": total_600_amort,
        "fill_only_note": "geometry (partition) NOT counted; oracle GT partition isolates the fill",
    }


def rate_from_total_bytes(total_bytes):
    return 25.0 * total_bytes / B0


# ===========================================================================
# Measure ONE polynomial order on ONE frame (closed-form fill, real SegNet, exact roundtrip)
# ===========================================================================
def measure_polynomial_order(L, frame_rgb, segnet_cpu, k, n_classes=5):
    """Closed-form polynomial fill at order k, then measure geometric + realized d_seg.

    Returns geometric d_seg (no roundtrip), realized d_seg (through the EXACT uint8 roundtrip +
    real SegNet), boundary/interior flips, byte cost, S projection, fit residual.
    """
    import numpy as np
    import torch
    from scipy import ndimage

    H, W = L.shape
    t0 = time.time()

    # ---- closed-form polynomial gradient fill (the photoshop gradient per region) ----
    filled_hw3, total_coeffs, n_reg, mean_resid = polynomial_fill_frame(L, frame_rgb, k, n_classes)
    filled_chw = torch.from_numpy(filled_hw3).float().permute(2, 0, 1)  # (3,H,W)

    with torch.no_grad():
        # (a) geometric d_seg (NO roundtrip): the polynomial-filled frame's SegNet argmax vs L*
        geo_argmax = _segnet_argmax_of_frame(segnet_cpu, filled_chw).cpu().numpy()
        geo_dseg = float((geo_argmax != L).mean())
        # (b) REALIZED d_seg: THROUGH the EXACT uint8 roundtrip (the survival check)
        rt_hard = _eval_roundtrip_t(filled_chw, ste=False)[0]  # (3,384,512) uint8 roundtrip
        real_argmax = _segnet_argmax_of_frame(segnet_cpu, rt_hard).cpu().numpy()
        real_dseg = float((real_argmax != L).mean())

        # boundary-band vs interior realized flip (where the survival wall concentrates)
        bmask = np.zeros((H, W), dtype=bool)
        bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
        bmask[:, 1:] |= L[:, :-1] != L[:, 1:]
        bmask[:-1, :] |= L[:-1, :] != L[1:, :]
        bmask[1:, :] |= L[:-1, :] != L[1:, :]
        band = ndimage.binary_dilation(bmask, iterations=1)
        boundary_flip = (
            float((real_argmax[band] != L[band]).mean()) if band.any() else float("nan")
        )
        interior = ~band
        interior_flip = (
            float((real_argmax[interior] != L[interior]).mean())
            if interior.any()
            else float("nan")
        )

    bytes_info = polynomial_param_bytes(total_coeffs)
    rate_amort = rate_from_total_bytes(bytes_info["total_600_amortized_bytes"])
    rate_full = rate_from_total_bytes(bytes_info["total_600_full_bytes"])
    s_amort = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_amort
    s_full = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_full

    return {
        "k": int(k),
        "n_coeffs_per_region": int(n_coeffs(k)),
        "n_regions": int(n_reg),
        "total_coeffs": int(total_coeffs),
        "fit_mean_residual_rgb_l2": mean_resid,
        # geometric_dseg = polynomial-filled frame's SegNet argmax (NO roundtrip) vs L*
        "geometric_dseg": geo_dseg,
        # realized_dseg = THROUGH the exact uint8 roundtrip (the survival-checked AUTHORITY)
        "realized_dseg": real_dseg,
        "boundary_band_flip": boundary_flip,
        "interior_flip": interior_flip,
        "bytes": bytes_info,
        "rate_amortized": rate_amort,
        "rate_full": rate_full,
        "S_projected_amortized": s_amort,
        "S_projected_full": s_full,
        "realized_dseg_x_frontier": real_dseg / FRONTIER_DSEG,
        "realized_dseg_x_flat_wall": real_dseg / FLAT_COLOUR_SURVIVAL_WALL_DSEG,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "experiments/results/polynomial_fill_survival_probe"),
    )
    ap.add_argument(
        "--orders",
        default="0,1,2,3,4,5,6",
        help="comma-sep polynomial orders k to sweep",
    )
    ap.add_argument("--n-frames", type=int, default=3, help="GT frames to measure over")
    ap.add_argument(
        "--timing-smoke",
        action="store_true",
        help="single order (k=2), 1 frame, to calibrate s/order then exit",
    )
    ap.add_argument(
        "--verdict-only",
        action="store_true",
        help="skip measurement; recompute the verdict + result JSON from existing gate_state rows",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "gate_state.json"

    if args.verdict_only:
        if not state_path.exists():
            print("[verdict-only] no gate_state.json; nothing to finalize")
            return 1
        state = json.loads(state_path.read_text())
        if not state.get("rows"):
            print("[verdict-only] gate_state.json has no rows")
            return 1
        n_frames = max((r.get("n_frames", 0) for r in state["rows"].values()), default=0)
        return _finalize_verdict(state, state_path, args, n_frames, verdict_only=True)

    import torch

    from tac.boundary_math.seg_core import decode_gt_frame1_pairs
    from tac.scorer import load_default_segnet

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded prior rows: {list(state.get('rows', {}).keys())}")
    state.setdefault("rows", {})

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    segnet_cpu = load_default_segnet(str(UPSTREAM), device="cpu")  # CPU AUTHORITY (no MPS)

    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"].numpy()  # (16, 384, 512) int64 = L*
    n_avail = seg_targets.shape[0]
    n_frames = min(args.n_frames, n_avail)

    # decode the matching GT RGB frames (the LS fit targets) via the EXACT path
    import torch.nn.functional as F

    gt_frames = {}
    for pidx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_frames):
        t = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0)
        rs = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False)
        gt_frames[pidx] = rs[0].permute(1, 2, 0).numpy()  # (384,512,3) float[0,255]
        if len(gt_frames) >= n_frames:
            break

    if args.timing_smoke:
        print("[timing-smoke] k=2, 1 frame ...")
        L = seg_targets[0]
        fr = gt_frames.get(0)
        r = measure_polynomial_order(L, fr, segnet_cpu, 2)
        print(
            f"[timing-smoke] {r['elapsed_s']:.1f}s/order n_reg={r['n_regions']} "
            f"coeffs={r['total_coeffs']} geo={r['geometric_dseg']:.5f} "
            f"realized={r['realized_dseg']:.5f} bnd_flip={r['boundary_band_flip']:.3f} "
            f"resid={r['fit_mean_residual_rgb_l2']:.1f} rate={r['rate_amortized']:.5f} "
            f"S~{r['S_projected_amortized']:.3f}"
        )
        n_orders = len(args.orders.split(","))
        print(
            f"   full sweep ~ {r['elapsed_s'] * n_orders * n_frames / 60:.1f} min "
            f"({n_orders} orders x {n_frames} frames)"
        )
        return 0

    orders = [int(x) for x in args.orders.split(",") if x.strip()]

    for k in orders:
        key = f"k{k}"
        if key in state["rows"]:
            print(f"[resume] {key} already done; skip")
            continue
        print(f"\n=== POLYNOMIAL ORDER {key} ({n_coeffs(k)} coeffs/region/channel) ===")
        per_frame = []
        for fidx in range(n_frames):
            L = seg_targets[fidx]
            fr = gt_frames.get(fidx)
            if fr is None:
                continue
            r = measure_polynomial_order(L, fr, segnet_cpu, k)
            per_frame.append(r)
            print(
                f"   frame{fidx}: n_reg={r['n_regions']:4d} coeffs={r['total_coeffs']:6d} "
                f"geo={r['geometric_dseg']:.5f} realized={r['realized_dseg']:.5f} "
                f"bnd_flip={r['boundary_band_flip']:.3f} resid={r['fit_mean_residual_rgb_l2']:.1f} "
                f"rate={r['rate_amortized']:.5f} S~{r['S_projected_amortized']:.3f} {r['elapsed_s']:.0f}s"
            )
        if not per_frame:
            continue

        def avg(kk, _pf=per_frame):  # bind loop var (avoid late-binding closure)
            vals = [
                p[kk]
                for p in _pf
                if p[kk] is not None and not (isinstance(p[kk], float) and math.isnan(p[kk]))
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        row = {
            "k": k,
            "n_coeffs_per_region": n_coeffs(k),
            "n_frames": len(per_frame),
            "avg_n_regions": avg("n_regions"),
            "avg_total_coeffs": avg("total_coeffs"),
            "avg_fit_residual": avg("fit_mean_residual_rgb_l2"),
            "avg_geometric_dseg": avg("geometric_dseg"),
            "avg_realized_dseg": avg("realized_dseg"),
            "avg_boundary_band_flip": avg("boundary_band_flip"),
            "avg_interior_flip": avg("interior_flip"),
            "avg_rate_amortized": avg("rate_amortized"),
            "avg_rate_full": avg("rate_full"),
            "avg_S_projected_amortized": avg("S_projected_amortized"),
            "avg_S_projected_full": avg("S_projected_full"),
            "per_frame": per_frame,
        }
        state["rows"][key] = row
        save_state()
        print(
            f"   -> {key}: avg_realized_dseg={row['avg_realized_dseg']:.5f} "
            f"({row['avg_realized_dseg']/FRONTIER_DSEG:.1f}x frontier, "
            f"{row['avg_realized_dseg']/FLAT_COLOUR_SURVIVAL_WALL_DSEG:.2f}x flat-wall) "
            f"avg_rate_amort={row['avg_rate_amortized']:.5f} "
            f"avg_S~{row['avg_S_projected_amortized']:.3f}"
        )

    return _finalize_verdict(state, state_path, args, n_frames, verdict_only=False)


def _finalize_verdict(state, state_path, args, n_frames, verdict_only=False):
    """MEASUREMENT-FIRST verdict from the measured rows + write the result JSON.

    realized d_seg through the chain is the AUTHORITY; the fit residual is a surrogate, never
    the verdict. Guards false-GREEN (low residual != low realized d_seg) AND false-RED
    (k=0 must reproduce the known flat-colour floor ~0.0067).
    """
    rows = state["rows"]
    if not rows:
        print("[error] no rows produced")
        return 1

    best_realized = min(r["avg_realized_dseg"] for r in rows.values())
    best_geometric = min(r["avg_geometric_dseg"] for r in rows.values())
    best_S = min(r["avg_S_projected_amortized"] for r in rows.values())
    best_row = min(rows.values(), key=lambda r: r["avg_S_projected_amortized"])

    cheap_rows = [r for r in rows.values() if r["avg_rate_amortized"] < 0.05]
    cheap_and_low = [r for r in cheap_rows if r["avg_realized_dseg"] < GREEN_DSEG_THRESHOLD]
    sub015_rows = [
        r
        for r in rows.values()
        if r["avg_rate_amortized"] < 0.05 and r["avg_S_projected_amortized"] < 0.15
    ]

    # k=0 sanity: does the constant fill reproduce the known flat-colour survival wall?
    k0_row = rows.get("k0")
    k0_realized = k0_row["avg_realized_dseg"] if k0_row else None
    k0_reproduces_flat_wall = (
        k0_realized is not None
        and abs(k0_realized - FLAT_COLOUR_SURVIVAL_WALL_DSEG) < 0.0025
    )

    # the d_seg descent across order: does realized drop meaningfully as k rises?
    realized_by_k = {
        int(r["k"]): r["avg_realized_dseg"] for r in rows.values()
    }
    realized_drop_from_k0 = (
        (k0_realized - best_realized) if k0_realized is not None else None
    )
    meaningful_drop = (
        realized_drop_from_k0 is not None and realized_drop_from_k0 > 0.0015
    )  # > ~1.5e-3 absolute reduction = a real survival improvement

    if sub015_rows and cheap_and_low:
        verdict = "GREEN_POLYNOMIAL_FILL_REACHES_SUB015_BYTE_CHEAP"
    elif best_realized < GREEN_DSEG_THRESHOLD:
        verdict = "AMBER_LOW_DSEG_BUT_NOT_SUB015_OR_BYTE_CHEAP"
    elif meaningful_drop:
        # d_seg drops with order but does not clear the GREEN threshold byte-cheaply
        verdict = "AMBER_POLYNOMIAL_FILL_IMPROVES_BUT_PLATEAUS_ABOVE_SUB015"
    elif best_realized >= 0.9 * FLAT_COLOUR_SURVIVAL_WALL_DSEG:
        verdict = "RED_POLYNOMIAL_FILL_CAPS_AT_FLAT_COLOUR_SURVIVAL_WALL"
    else:
        verdict = "RED_POLYNOMIAL_FILL_CAPS_ABOVE_SUB015"

    # mechanism diagnosis
    if not k0_reproduces_flat_wall and k0_realized is not None:
        sanity = (
            f"WARNING: k=0 realized d_seg {k0_realized:.5f} does NOT reproduce the known flat "
            f"survival wall {FLAT_COLOUR_SURVIVAL_WALL_DSEG:.5f} (|delta|>0.0025) -- harness "
            f"sanity check soft-failed; interpret with care (possible n-frame/partition diff)"
        )
    else:
        sanity = (
            f"k=0 realized d_seg reproduces the flat survival wall "
            f"({k0_realized:.5f} ~ {FLAT_COLOUR_SURVIVAL_WALL_DSEG:.5f}) -- harness sane"
            if k0_realized is not None
            else "k=0 not measured (cannot sanity-check the flat baseline)"
        )

    if best_realized < GREEN_DSEG_THRESHOLD:
        mechanism = (
            "CONTINUOUS POLYNOMIAL GRADIENT BEATS THE FLAT SURVIVAL WALL: a smooth per-region "
            "gradient lands inside the per-pixel argmax polytope where a flat colour did not."
        )
    elif meaningful_drop:
        mechanism = (
            f"PARTIAL: polynomial gradient REDUCES realized d_seg by {realized_drop_from_k0:.5f} "
            f"from k=0 (flat) but PLATEAUS above sub-0.15. SegNet's required texture is partly "
            f"low-order-smooth (gradient helps) but has HF/non-smooth structure a polynomial of "
            f"order<={max(int(r['k']) for r in rows.values())} cannot capture."
        )
    else:
        mechanism = (
            "FLAT-WALL CONFIRMED ACROSS ORDERS: even high-order polynomial fill caps at the "
            "flat-colour survival wall -> the texture SegNet keys on is NOT low-order-polynomial "
            "(it is HF/non-smooth) -> the continuous-SMOOTH representation family is closed; only "
            "a full learned (HF) texture-decoder survives the scorer."
        )

    result_json = REPO / ".omx/research" / f"polynomial_fill_survival_gate_{_now()}.json"
    payload = {
        "schema": "polynomial_fill_survival_gate.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_polynomial_fill_survival_gate.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "finalized_verdict_only": bool(verdict_only),
        "the_question": (
            "Does a CONTINUOUS per-region polynomial gradient fill beat the FLAT-COLOUR survival "
            "wall (realized d_seg ~0.0067) that capped the curve-core, and at what byte cost? Is "
            "there a (k, rate) with realized < 0.0012 AND rate < 0.05 AND S < 0.15? GREEN -> the "
            "photoshop-polynomial sub-0.15 path; AMBER -> the order<->d_seg<->byte curve (partial "
            "win + plateau); RED -> the continuous-smooth family is closed (only an HF decoder survives)."
        ),
        "thresholds": {
            "frontier_dseg": FRONTIER_DSEG,
            "sub015_dseg": SUB015_DSEG,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
            "flat_colour_survival_wall_dseg": FLAT_COLOUR_SURVIVAL_WALL_DSEG,
            "byte_cheap_rate_threshold": 0.05,
            "S_target": 0.15,
        },
        "method": {
            "representation": (
                "Per connected region of the GT argmax partition L*, a per-RGB-channel 2D "
                "polynomial c(x,y)=sum_{i+j<=k} a_ij x^i y^j, coefficients fit by CLOSED-FORM "
                "least-squares to the GT RGB inside the region (oracle fit). k=0 == per-region "
                "mean colour == the flat-colour survival baseline. Partition = oracle GT geometry "
                "(isolates the FILL question per the directive; geometry bytes NOT counted)."
            ),
            "dseg_metric": (
                "realized = argmax-flip-rate of the HARD polynomial-filled frame THROUGH the EXACT "
                "uint8 roundtrip (bicubic-874 -> bilinear-384 -> round) + real frozen SegNet vs L* "
                "(the survival-checked AUTHORITY). geometric = SegNet argmax of the filled frame "
                "with NO roundtrip (the pre-survival SegNet match)."
            ),
            "byte_cost": (
                "quantized polynomial coeffs (n_regions * n_coeffs(k) * 3 channels * 12 bits), "
                "entropy-coded (0.55 factor), quasi-static amortized (gamma0 once + 10% per-frame "
                "delta over 600). FILL-ONLY: the partition geometry is the oracle and NOT counted."
            ),
            "compute": "CPU only, closed-form least-squares, NO gradient, NO MPS",
            "authority_device": "cpu",
            "n_frames": n_frames,
        },
        "rows": rows,
        "realized_by_order": realized_by_k,
        "best_realized_dseg": best_realized,
        "best_geometric_dseg": best_geometric,
        "best_realized_dseg_x_frontier": best_realized / FRONTIER_DSEG,
        "best_realized_dseg_x_flat_wall": best_realized / FLAT_COLOUR_SURVIVAL_WALL_DSEG,
        "best_S_projected_amortized": best_S,
        "best_row_key": f"k{best_row['k']}",
        "k0_realized_dseg": k0_realized,
        "k0_reproduces_flat_wall": bool(k0_reproduces_flat_wall),
        "realized_drop_from_k0": realized_drop_from_k0,
        "meaningful_drop_with_order": bool(meaningful_drop),
        "n_byte_cheap_rows": len(cheap_rows),
        "n_byte_cheap_AND_low_dseg_rows": len(cheap_and_low),
        "n_sub015_rows": len(sub015_rows),
        "harness_sanity": sanity,
        "mechanism": mechanism,
        "verdict": verdict,
        "verdict_basis": (
            "MEASUREMENT-FIRST: driven by the realized d_seg of the HARD polynomial-filled frame "
            "THROUGH the real SegNet + exact uint8 roundtrip (the survival check), NOT the LS fit "
            "residual (a surrogate). A low fit residual with high realized d_seg = the survival "
            "wall (the texture SegNet needs is not low-order-smooth)."
        ),
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))
    state["final_verdict"] = verdict
    state["result_json"] = str(result_json.relative_to(REPO))
    state_path.write_text(json.dumps(state, indent=2))

    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}")
    print(
        f"[best realized d_seg] {best_realized:.5f} ({best_realized/FRONTIER_DSEG:.1f}x frontier, "
        f"{best_realized/FLAT_COLOUR_SURVIVAL_WALL_DSEG:.2f}x flat survival wall)"
    )
    print(f"[best projected S] {best_S:.4f}  (best row {payload['best_row_key']})")
    print(f"[k=0 sanity] {sanity}")
    print(f"[mechanism] {mechanism}")
    print(f"[VERDICT] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
