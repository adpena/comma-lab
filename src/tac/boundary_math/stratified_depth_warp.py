"""Morse-Smale-stratified parallax warp (task #365 prototype, ADVISORY).

The store-nothing pose carrier synthesizes frame0 = inverse-warp(witness frame0 render, ground
homography H(xi)). That plane-only warp reproduces ground-plane flow EXACTLY and sets depth parallax
gamma(p) = 0 everywhere else. This module adds a per-region parallax term SUPPORTED on the argmax
partition's off-plane cells (movable + undrivable-below-horizon), so the synthesized frame0 can carry
the finite-depth parallax the global homography cannot (design memo
``pose_taskspace_native_morse_smale_depth_warp_design_20260708.md`` sec 3, D1 field form).

BIT-PARITY GUARANTEE: with ``extra_flow`` all-zero (or ``offplane_mask`` all-False) this reduces
OP-FOR-OP to ``warp_real_luma_frame0.warp_frame0_native_numpy`` (same fp64 Hinv @ grid inverse-sample
+ persist fallback + round/clip), so the A0 control is a strict special case. NumPy fp64 is the
authority (deterministic, host-portable). ADVISORY / NON-PROMOTABLE — nothing here is a contest score
until byte-closed through upstream/evaluate.py.
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    homography_from_xi_numpy,
)


def offplane_mask_from_partition(argmax: np.ndarray, native_hw: tuple[int, int],
                                 *, horizon_frac: float = 0.5) -> np.ndarray:
    """Boolean (H,W) mask of OFF-PLANE finite-depth pixels from a class-argmax partition.

    Canonical comma10k order (CLAUDE.md L80, MEASURED): 0=Road 1=Lane 2=Undrivable(incl sky)
    3=Movable 4=MyCar/hood. Off-plane finite depth = movable(3) + undrivable(2) BELOW the horizon
    (vertical structure); ground(0,1) uses H(xi) exactly, sky (2 above horizon) has parallax ~0, hood
    (4) is static near-field. ``argmax`` may be at any resolution -> nearest-resampled to native_hw.
    """
    H, W = int(native_hw[0]), int(native_hw[1])
    a = np.asarray(argmax)
    if a.shape != (H, W):  # nearest-neighbour resample to native camera grid
        ys = (np.linspace(0, a.shape[0] - 1, H)).round().astype(np.int64)
        xs = (np.linspace(0, a.shape[1] - 1, W)).round().astype(np.int64)
        a = a[ys][:, xs]
    hz = int(horizon_frac * H)
    rows = np.arange(H)[:, None]
    return (a == 3) | ((a == 2) & (rows >= hz))


def stratified_warp_native(
    src_hwc: np.ndarray, xi: np.ndarray, geom: GroundHomographyGeom,
    *, offplane_mask: np.ndarray | None = None, extra_flow: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse-warp ``src`` by H(xi), adding a per-pixel source-coord offset ``extra_flow`` on the
    off-plane mask (the depth-parallax steering term). ``extra_flow`` is (H,W,2) [du,dv] added to the
    homography source coords for masked pixels (unit: source pixels). Returns fp64 (H,W,3) [0,255]."""
    src = np.asarray(src_hwc, dtype=np.float64)
    Hh, Ww, C = src.shape
    if (Hh, Ww) != geom.native_hw:
        raise ValueError(f"src native {(Hh, Ww)} != geom.native_hw {geom.native_hw}")
    flat = src.reshape(-1, C)
    H = homography_from_xi_numpy(xi, geom)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ geom.grid  # (3, HW) fp64
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    if offplane_mask is not None and extra_flow is not None:
        m = np.asarray(offplane_mask, bool).reshape(-1)
        ef = np.asarray(extra_flow, np.float64).reshape(-1, 2)
        su = su + np.where(m, ef[:, 0], 0.0)
        sv = sv + np.where(m, ef[:, 1], 0.0)
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]
    Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]
    Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat)  # persist fallback
    return out.reshape(Hh, Ww, C)


def stratified_warp_uint8(src_hwc, xi, geom, *, offplane_mask=None, extra_flow=None) -> np.ndarray:
    f = stratified_warp_native(src_hwc, xi, geom, offplane_mask=offplane_mask, extra_flow=extra_flow)
    return np.clip(np.round(f), 0.0, 255.0).astype(np.uint8)


def affine_extra_flow(native_hw: tuple[int, int], params: np.ndarray) -> np.ndarray:
    """Build a (H,W,2) affine source-coord offset field from 6 params [a,b,c,d,e,f]:
    du = a*(x_n) + b*(y_n) + e ; dv = c*(x_n) + d*(y_n) + f, with x_n,y_n in [-1,1] normalized coords.
    The depth-parallax steering DOF for A2+ (a low-order per-region flow the homography cannot make)."""
    H, W = int(native_hw[0]), int(native_hw[1])
    a, b, c, d, e, f = [float(p) for p in params]
    xn = (np.arange(W) / (W - 1) * 2 - 1)[None, :] * np.ones((H, 1))
    yn = (np.arange(H) / (H - 1) * 2 - 1)[:, None] * np.ones((1, W))
    du = a * xn + b * yn + e
    dv = c * xn + d * yn + f
    return np.stack([du, dv], axis=-1)
