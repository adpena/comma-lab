#!/usr/bin/env python
"""ddm_ua1 — is the GT/output loader asymmetry an exploitable lever?

upstream/evaluate.py loads the two sides of the comparison through DIFFERENT
dataset classes (VERIFIED_VIA_SOURCE_INSPECTION, upstream/evaluate.py:58,67):

  GROUND TRUTH : DaliVideoDataset (cuda) / AVVideoDataset (cpu)
                 -> hevc decode -> yuv420p planes -> frame_utils.yuv420_to_rgb
                 -> uint8 (874,1164,3)            [frame_utils.py:159-183]
  OUR OUTPUT   : TensorVideoDataset -> np.memmap raw uint8 (N,874,1164,3)
                 -> NO codec, NO yuv420, NO chroma subsampling
                                                  [frame_utils.py:218-253]

So GT is the image of a constrained source; we are unconstrained.  This probe
measures HOW constrained, and whether the surplus buys anything through the
frozen scorers.

WHAT THE SOURCE ACTUALLY SAYS (diffed against the briefing):
  * LUMA IS NOT SUBSAMPLED.  Only U,V are 2x2.  The constraint is a chroma
    BANDWIDTH constraint plus a gamut constraint, not a blanket "half the bits".
  * GT IS uint8 (`.round().to(torch.uint8)`, frame_utils.py:183) -- both sides
    live in the same 256^3 cube.  The asymmetry is about WHICH points are
    reachable, not about dtype/precision.
  * yuv420_to_rgb is the CPU REIMPLEMENTATION of nvdec; the DALI path never
    calls it.  GT-on-CUDA is nvdec hardware output.

THE ALGEBRA (DERIVED, verified numerically in stage D)
  BT.601 limited-range decode is an INVERTIBLE affine map per camera pixel:
      yf = .299r + .587g + .114b            (exactly the BT.601 luma equation)
      uf = (b - yf)/1.772 , vf = (r - yf)/1.402
  so RGB <-> YUV is a bijection.  GT-reachability of an integer RGB triple is
  therefore EXACTLY the question "does the implied (y,u,v) fit in the box, with
  y on the integer luma lattice?"  Luma is quantised to 256 levels spaced
  255/219 = 1.1644 apart in yf, while chroma is effectively continuous (the
  bilinear upsample blends neighbours at 1/16 granularity), which is what makes
  the reachable set a union of sheets rather than the full cube.

Stages
  A  loader verify on a real frame (plane shapes, dtypes, DOF accounting)
  B  EXACT enumeration of the GT-reachable subset of the 256^3 RGB cube
  C  real-GT-frame occupancy (positive control: must be 100% reachable)
  D  numerical check that D o decode collapses to (Ybar, Ubar, Vbar)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

# BT.601 limited-range decode constants, transcribed from frame_utils.py:176-182.
Y_OFF, Y_GAIN = 16.0, 255.0 / 219.0
C_OFF, C_GAIN = 128.0, 255.0 / 224.0
K_RV = 1.402
K_GU, K_GV = 0.344136, 0.714136
K_BU = 1.772

# Forward luma weights implied by the inverse of the decode matrix (DERIVED).
W_R, W_G, W_B = 0.299, 0.587, 0.114

CAMERA_H, CAMERA_W = 874, 1164


def decode_yuv_to_rgb_float(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unclamped, unrounded BT.601 limited-range decode (float64).

    Mirrors frame_utils.yuv420_to_rgb lines 176-182 exactly, minus the final
    clamp/round so callers can inspect gamut excursions.
    """
    yf = (y - Y_OFF) * Y_GAIN
    uf = (u - C_OFF) * C_GAIN
    vf = (v - C_OFF) * C_GAIN
    r = yf + K_RV * vf
    g = yf - K_GU * uf - K_GV * vf
    b = yf + K_BU * uf
    return np.stack([r, g, b], axis=-1)


def rgb_to_yuv_exact(rgb: np.ndarray) -> np.ndarray:
    """Exact inverse of decode_yuv_to_rgb_float (DERIVED; asserted in stage D)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yf = W_R * r + W_G * g + W_B * b
    uf = (b - yf) / K_BU
    vf = (r - yf) / K_RV
    y = yf / Y_GAIN + Y_OFF
    u = uf / C_GAIN + C_OFF
    v = vf / C_GAIN + C_OFF
    return np.stack([y, u, v], axis=-1)


# --------------------------------------------------------------------------
# Stage B: exact gamut enumeration
# --------------------------------------------------------------------------

def _target_interval(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Acceptable pre-clamp value interval so that round(clamp(x)) == t.

    clamp saturates at 0 and 255, so those two target values accept a half-open
    ray rather than a +/-0.5 window.  Everything else is [t-0.5, t+0.5].
    """
    lo = np.where(t <= 0, -np.inf, t - 0.5)
    hi = np.where(t >= 255, np.inf, t + 0.5)
    return lo, hi


def gt_reachable_mask_for_r(r_val: int, chroma_continuous: bool,
                            y_window: int = 4) -> np.ndarray:
    """Exact reachability of every (r_val, g, b) integer triple, shape (256,256).

    For a fixed integer luma level y (hence yf), vf is pinned by the r target and
    uf by the b target; the g target is then the single residual constraint.
    Because g depends monotonically on both uf and vf, feasibility is an interval
    overlap test -- no search, no sampling.

    chroma_continuous=True models the real GT (bilinear-upsampled chroma takes
    fractional values); False models a pure un-blended chroma sample.
    """
    gg, bb = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    gg = gg.astype(np.float64)
    bb = bb.astype(np.float64)
    rr = np.full_like(gg, float(r_val))

    r_lo, r_hi = _target_interval(rr)
    g_lo, g_hi = _target_interval(gg)
    b_lo, b_hi = _target_interval(bb)

    # chroma box in the uf/vf coordinate (u,v in [0,255] uint8 range)
    c_lo = (0.0 - C_OFF) * C_GAIN
    c_hi = (255.0 - C_OFF) * C_GAIN

    feasible = np.zeros(gg.shape, dtype=bool)

    # Only luma levels near the implied luma can work: an offset delta in yf
    # moves g by (1 + K_GU/K_BU + K_GV/K_RV)*delta ~= 1.7036*delta, and the
    # lattice spacing is 1.1644, so a small window is provably sufficient.
    yf_ideal = W_R * rr + W_G * gg + W_B * bb
    y_ideal = yf_ideal / Y_GAIN + Y_OFF
    y_center = np.clip(np.rint(y_ideal), 0, 255).astype(np.int32)

    for off in range(-y_window, y_window + 1):
        y_try = np.clip(y_center + off, 0, 255)
        yf = (y_try.astype(np.float64) - Y_OFF) * Y_GAIN

        # vf interval from the r constraint, uf interval from the b constraint
        vf_lo = (r_lo - yf) / K_RV
        vf_hi = (r_hi - yf) / K_RV
        uf_lo = (b_lo - yf) / K_BU
        uf_hi = (b_hi - yf) / K_BU

        vf_lo = np.maximum(vf_lo, c_lo)
        vf_hi = np.minimum(vf_hi, c_hi)
        uf_lo = np.maximum(uf_lo, c_lo)
        uf_hi = np.minimum(uf_hi, c_hi)

        if not chroma_continuous:
            # chroma must be an integer u,v: snap the interval to integer u,v
            u_lo_i = np.ceil(uf_lo / C_GAIN + C_OFF - 1e-9)
            u_hi_i = np.floor(uf_hi / C_GAIN + C_OFF + 1e-9)
            v_lo_i = np.ceil(vf_lo / C_GAIN + C_OFF - 1e-9)
            v_hi_i = np.floor(vf_hi / C_GAIN + C_OFF + 1e-9)
            ok_box = (u_lo_i <= u_hi_i) & (v_lo_i <= v_hi_i)
            uf_lo = (u_lo_i - C_OFF) * C_GAIN
            uf_hi = (u_hi_i - C_OFF) * C_GAIN
            vf_lo = (v_lo_i - C_OFF) * C_GAIN
            vf_hi = (v_hi_i - C_OFF) * C_GAIN
        else:
            ok_box = (uf_lo <= uf_hi) & (vf_lo <= vf_hi)

        # g = yf - K_GU*uf - K_GV*vf is decreasing in uf and vf
        g_max = yf - K_GU * uf_lo - K_GV * vf_lo
        g_min = yf - K_GU * uf_hi - K_GV * vf_hi
        ok = ok_box & (g_max >= g_lo) & (g_min <= g_hi)
        feasible |= ok

    return feasible


def stage_b(chroma_continuous: bool = True) -> dict:
    total = 0
    interior = 0
    interior_total = 0
    for r_val in range(256):
        m = gt_reachable_mask_for_r(r_val, chroma_continuous)
        total += int(m.sum())
        if 0 < r_val < 255:
            interior += int(m[1:255, 1:255].sum())
            interior_total += 254 * 254
    return {
        "chroma_model": "continuous" if chroma_continuous else "integer",
        "reachable_triples": total,
        "cube_triples": 256 ** 3,
        "reachable_fraction_full_cube": total / 256 ** 3,
        "reachable_interior": interior,
        "interior_total": interior_total,
        "reachable_fraction_interior": interior / interior_total,
    }


# --------------------------------------------------------------------------
# Stage A / C: real frames
# --------------------------------------------------------------------------

def decode_real_frames(video: Path, n_frames: int) -> tuple[np.ndarray, list[dict]]:
    """Decode via the CANONICAL upstream path only (frame_utils.yuv420_to_rgb)."""
    import av  # noqa: PLC0415
    from frame_utils import yuv420_to_rgb  # noqa: PLC0415

    container = av.open(str(video))
    stream = container.streams.video[0]
    rgbs, plane_meta = [], []
    for frame in container.decode(stream):
        if len(rgbs) >= n_frames:
            break
        plane_meta.append({
            "format": frame.format.name,
            "height": frame.height,
            "width": frame.width,
            "plane_shapes": [
                [frame.height, frame.planes[0].line_size],
                [frame.height // 2, frame.planes[1].line_size],
                [frame.height // 2, frame.planes[2].line_size],
            ],
        })
        rgbs.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return np.stack(rgbs), plane_meta


def gamut_report(rgb_u8: np.ndarray) -> dict:
    """Per-pixel GT-gamut diagnostics for an arbitrary uint8 camera image."""
    yuv = rgb_to_yuv_exact(rgb_u8.astype(np.float64))
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    box_ok = ((y >= 0) & (y <= 255) & (u >= 0) & (u <= 255) & (v >= 0) & (v <= 255))
    # luma-lattice residual: how far the implied y is from an integer
    y_resid = np.abs(y - np.rint(y))
    n = y.size
    return {
        "pixels": int(n),
        "in_yuv_box_frac": float(box_ok.mean()),
        "y_out_of_box_frac": float(((y < 0) | (y > 255)).mean()),
        "u_out_of_box_frac": float(((u < 0) | (u > 255)).mean()),
        "v_out_of_box_frac": float(((v < 0) | (v > 255)).mean()),
        "luma_lattice_resid_mean": float(y_resid.mean()),
        "luma_lattice_resid_p99": float(np.percentile(y_resid, 99)),
        "luma_lattice_resid_max": float(y_resid.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=REPO / "upstream" / "videos" / "0.mkv")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / ".omx" / "research" / "ddm_ua1_manifold.json")
    ap.add_argument("--skip-enumeration", action="store_true")
    args = ap.parse_args()

    report: dict = {}

    # ---- Stage A: loader verify + DOF accounting -------------------------
    rgbs, plane_meta = decode_real_frames(args.video, args.frames)
    luma_scalars = CAMERA_H * CAMERA_W
    chroma_scalars = 2 * (CAMERA_H // 2) * (CAMERA_W // 2)
    ours_scalars = 3 * CAMERA_H * CAMERA_W
    report["stage_a_loader"] = {
        "gt_decode_path": "AVVideoDataset -> frame_utils.yuv420_to_rgb",
        "gt_rgb_shape": list(rgbs.shape),
        "gt_rgb_dtype": str(rgbs.dtype),
        "gt_pixfmt": plane_meta[0]["format"],
        "gt_plane_shapes_yuv": plane_meta[0]["plane_shapes"],
        "gt_source_scalars_luma": luma_scalars,
        "gt_source_scalars_chroma": chroma_scalars,
        "gt_source_scalars_total": luma_scalars + chroma_scalars,
        "our_source_scalars_total": ours_scalars,
        "gt_bits_per_camera_pixel": 8 * (luma_scalars + chroma_scalars) / (CAMERA_H * CAMERA_W),
        "our_bits_per_camera_pixel": 24.0,
        "dof_ratio_ours_over_gt": ours_scalars / (luma_scalars + chroma_scalars),
    }

    # ---- Stage D: bijection + D-collapse check ---------------------------
    rng = np.random.default_rng(1234)
    y_t = rng.integers(0, 256, size=4096).astype(np.float64)
    u_t = rng.uniform(0, 255, size=4096)
    v_t = rng.uniform(0, 255, size=4096)
    fwd = decode_yuv_to_rgb_float(y_t, u_t, v_t)
    back = rgb_to_yuv_exact(fwd)
    report["stage_d_bijection"] = {
        "max_abs_roundtrip_err_y": float(np.abs(back[..., 0] - y_t).max()),
        "max_abs_roundtrip_err_u": float(np.abs(back[..., 1] - u_t).max()),
        "max_abs_roundtrip_err_v": float(np.abs(back[..., 2] - v_t).max()),
    }
    # D-collapse: a weighted average of decoded RGB equals the decode of the
    # weighted-average YUV (exact, since decode is affine) -- so the scorer sees
    # the GT source ONLY through (Ybar, Ubar, Vbar).
    w = rng.dirichlet(np.ones(4), size=1024)
    ys = rng.integers(0, 256, size=(1024, 4)).astype(np.float64)
    us = rng.uniform(0, 255, size=(1024, 4))
    vs = rng.uniform(0, 255, size=(1024, 4))
    rgb_each = decode_yuv_to_rgb_float(ys, us, vs)
    avg_of_decode = (w[..., None] * rgb_each).sum(axis=1)
    decode_of_avg = decode_yuv_to_rgb_float((w * ys).sum(1), (w * us).sum(1), (w * vs).sum(1))
    report["stage_d_collapse"] = {
        "claim": "D(decode(y,u,v)) == decode(D y, D u, D v) when no channel clamps",
        "max_abs_err": float(np.abs(avg_of_decode - decode_of_avg).max()),
    }

    # ---- Stage C: real GT frame occupancy (POSITIVE CONTROL) -------------
    report["stage_c_real_gt_frames"] = [gamut_report(rgbs[i]) for i in range(rgbs.shape[0])]

    # ---- Stage B: exact gamut enumeration --------------------------------
    if not args.skip_enumeration:
        report["stage_b_gamut_continuous_chroma"] = stage_b(chroma_continuous=True)
        report["stage_b_gamut_integer_chroma"] = stage_b(chroma_continuous=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
