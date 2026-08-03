"""ddm_bp2 — the scorer-blind camera pixels as a frame_0 (pose-only) actuator.

THE STRUCTURE (standing, MEASURED; see ``ddm_ll1_window_solve``)
---------------------------------------------------------------
The scorer downsample ``D = F.interpolate(x,(384,512),'bilinear')`` point-samples
disjoint 2x2 windows (stride 2.276 > 2, ``antialias=False``), so **230,904 camera
pixels (22.70%) are read by NEITHER scorer**.  Both SegNet and PoseNet consume the
identical ``D`` (``PoseNet.preprocess_input`` interpolates BEFORE ``rgb_to_yuv6``).

THE ACTUATOR CLAIM THIS MODULE MEASURES (task #401 composed with the pose axis)
------------------------------------------------------------------------------
SegNet reads ``x[:, -1, ...]`` — frame_1 ONLY.  The v4d vehicle renders frame_1 and
then produces frame_0 by WARPING frame_1's *camera* raster.  So a change to
frame_1's blind pixels:

  * is EXACTLY invisible to SegNet (blind through D)                -> zero seg cost
  * is EXACTLY invisible to PoseNet's frame_1 half (same D)
  * DOES reach PoseNet's frame_0 half, through the warp              -> a pose lever

This module supplies the linear algebra that makes the coupling measurable: an
exact tap decomposition of the vendored ``pfs1_warp_receiver.warp_rgb`` and its
adjoint, so the influence of every camera pixel on the frame_0 scorer plane can be
computed in closed form instead of probed one pixel at a time.

GENERIC ALGORITHM, ZERO COUNTED BYTES: the geometry is derived from the two frozen
shapes plus the per-pair homography the receiver already computes.  Nothing here is
video-derived (rule 118).  This module MEASURES; it ships nothing.
"""

from __future__ import annotations

from typing import Final

import numpy as np

CAMERA_H: Final = 874
CAMERA_W: Final = 1164
SEG_H: Final = 384
SEG_W: Final = 512


def warp_taps(
    homography: np.ndarray,
    target_grid: np.ndarray,
    *,
    height: int = CAMERA_H,
    width: int = CAMERA_W,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact tap decomposition of ``pfs1_warp_receiver.warp_rgb``.

    Returns ``(idx, w, valid)`` with ``idx``/``w`` of shape ``(4, height*width)``:
    output pixel ``q`` is ``sum_k w[k,q] * flat_src[idx[k,q]]``.  Reproduces the
    receiver's INVALID branch (``out = flat`` — the identity, NOT a warp read) as a
    single unit tap ``q -> q``, so row sums are 1 everywhere and the operator is
    exactly the receiver's.

    Verified against ``warp_rgb`` on real decoded frames: max abs error 8.5e-14
    (fp64 round-off), row-sum error 1.1e-16.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        h_inv = np.linalg.inv(homography)
        src_h = h_inv @ target_grid
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (
        np.isfinite(su)
        & np.isfinite(sv)
        & (z > 0)
        & (su >= 0)
        & (su <= width - 1)
        & (sv >= 0)
        & (sv <= height - 1)
    )
    su_c = np.clip(su, 0.0, width - 1)
    sv_c = np.clip(sv, 0.0, height - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = su_c - x0
    wy = sv_c - y0
    idx = np.stack([y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1])
    w = np.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy])
    q = np.arange(height * width, dtype=np.int64)
    zeros = np.zeros_like(wx)
    ones = np.ones_like(wx)
    idx = np.where(valid[None, :], idx, np.stack([q, q, q, q]))
    w = np.where(valid[None, :], w, np.stack([ones, zeros, zeros, zeros]))
    return idx, w, valid


def apply_taps(idx: np.ndarray, w: np.ndarray, src_hwc: np.ndarray) -> np.ndarray:
    """Forward: ``M @ src``.  ``src_hwc`` (H,W,C) -> (H,W,C)."""
    h, width, c = src_hwc.shape
    flat = src_hwc.reshape(-1, c).astype(np.float64)
    out = np.zeros((h * width, c), dtype=np.float64)
    for k in range(idx.shape[0]):
        out += w[k][:, None] * flat[idx[k]]
    return out.reshape(h, width, c)


def adjoint_taps(
    idx: np.ndarray,
    w: np.ndarray,
    cotangent_hwc: np.ndarray,
    *,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Adjoint: ``M^T @ v``.  ``cotangent_hwc`` (H,W,C) -> (H,W,C) in SOURCE space.

    Scatter-add of ``w * v`` back onto the tap indices — the exact transpose of
    :func:`apply_taps`, which is what turns a frame_0 cotangent into a per-camera-
    pixel sensitivity of frame_1.

    ``height``/``width`` DEFAULT TO THE COTANGENT'S OWN SHAPE (this operator is
    square: camera raster -> camera raster).  They previously defaulted to the
    camera constants, which meant a small cotangent silently produced a full-size,
    mostly-zero result with no error — a wrong answer that never raises.  Passing
    them explicitly is still allowed and is now cross-checked.
    """
    if cotangent_hwc.ndim != 3:
        raise ValueError(f"cotangent must be (H,W,C), got {cotangent_hwc.shape}")
    src_h, src_w, c = cotangent_hwc.shape
    height = src_h if height is None else height
    width = src_w if width is None else width
    if (height, width) != (src_h, src_w):
        raise ValueError(
            f"adjoint shape {(height, width)} disagrees with cotangent {(src_h, src_w)}"
        )
    if idx.shape[1] != height * width:
        raise ValueError(
            f"taps cover {idx.shape[1]} pixels but cotangent has {height * width}"
        )
    flat_v = cotangent_hwc.reshape(-1, c).astype(np.float64)
    out = np.zeros((height * width, c), dtype=np.float64)
    for k in range(idx.shape[0]):
        for ch in range(c):
            out[:, ch] += np.bincount(
                idx[k], weights=w[k] * flat_v[:, ch], minlength=height * width
            )
    return out.reshape(height, width, c)


def d_column_weights() -> np.ndarray:
    """(H,W) float64 — total ``D`` weight each camera pixel carries into the scorer.

    Derived from the REAL torch operator by one autograd pass (``D`` is linear with
    non-negative weights, so a camera pixel is blind iff its column sum is exactly
    zero).  Independent of any hand re-derivation of the bilinear geometry.
    """
    import torch
    import torch.nn.functional as functional

    x = torch.zeros(1, 1, CAMERA_H, CAMERA_W, dtype=torch.float64, requires_grad=True)
    functional.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear").sum().backward()
    if x.grad is None:  # never an assert: this guards an AUTHORITY quantity
        raise RuntimeError("autograd produced no column weights for D")
    return x.grad[0, 0].numpy().copy()


def v4d_pair_taps(decoder, pair: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(idx, w, valid) of the FULL v4d frame_0-from-frame_1 linear map, pre-``a``.

    Composes exactly what ``inflate_runner_v4d.Decoder.f0`` does, in the same order:
    the per-pair selector (single-plane vs static two-plane far/ground row split),
    and the rung-A rolling-shutter row blend ``(1-alpha)*W(1-beta/2) +
    alpha*W(1+beta/2)`` when the per-pair beta magnitude is non-zero.  The
    photometric ``f0 := a*warp + b`` is affine and is applied by the caller (only
    ``a`` scales the Jacobian).

    ``decoder`` is an ``inflate_runner_v4d.Decoder``; passed in rather than imported
    because the receiver lives in the vendored gate substrate, not in ``tac``.
    """
    from pfs1_warp_receiver import pose_to_homography

    pose = decoder.p_best[pair]
    s_t = float(decoder.st_vals[decoder.st_idx[pair]])
    sel = int(decoder.sel[pair])
    beta_mag = float(decoder.beta_mags[int(decoder.beta_idx[pair])])
    # Shape comes from the DECODER's own row mask, never from the module constants:
    # the receiver is the authority on its raster size, and deriving it keeps this
    # function testable at a size that does not need the 874x1164 grid.
    height, width = decoder._far.shape
    far = decoder._far.reshape(-1)

    def _sel_taps(rot: float):
        h_g = pose_to_homography(pose, decoder.K, decoder.Kinv, s_t, rot, 0.0)
        idx_g, w_g, v_g = warp_taps(h_g, decoder.grid, height=height, width=width)
        if sel == 0:
            return idx_g, w_g, v_g
        h_f = pose_to_homography(pose, decoder.K, decoder.Kinv, 0.0, rot, 0.0)
        idx_f, w_f, v_f = warp_taps(h_f, decoder.grid, height=height, width=width)
        return (
            np.where(far[None, :], idx_f, idx_g),
            np.where(far[None, :], w_f, w_g),
            np.where(far, v_f, v_g),
        )

    if beta_mag == 0.0:
        return _sel_taps(1.0)
    beta = beta_mag * (1.0 if pose[5] >= 0.0 else -1.0)
    idx_a, w_a, v_a = _sel_taps(1.0 - beta / 2.0)
    idx_b, w_b, v_b = _sel_taps(1.0 + beta / 2.0)
    alpha = np.repeat(decoder._alpha[:, 0, 0], width)
    return (
        np.concatenate([idx_a, idx_b]),
        np.concatenate([(1.0 - alpha)[None, :] * w_a, alpha[None, :] * w_b]),
        v_a | v_b,
    )


def blind_influence_mass(
    idx: np.ndarray,
    w: np.ndarray,
    d_weights: np.ndarray,
    blind: np.ndarray,
    *,
    photometric_a: float = 1.0,
) -> dict[str, float]:
    """How much of frame_0's scorer-visible signal is sourced from BLIND frame_1 px.

    ``influence[b] = |a| * sum_q M[q,b] * d_weights[q]`` — the exact column mass of
    the composed operator ``D . (a*M)``.  Because both ``M`` rows and ``D`` columns
    have unit/non-negative structure, ``total`` equals ``|a| * d_weights.sum()`` =
    ``|a| * 196608``, which the caller can use as a closure check.
    """
    flat_d = d_weights.reshape(-1)
    infl = np.zeros(flat_d.size, dtype=np.float64)
    for k in range(idx.shape[0]):
        infl += np.bincount(idx[k], weights=w[k] * flat_d, minlength=flat_d.size)
    infl *= abs(photometric_a)
    flat_blind = blind.reshape(-1)
    blind_infl = infl[flat_blind]
    total = float(infl.sum())
    active = blind_infl > 1e-12
    return {
        "total_mass": total,
        "blind_mass": float(blind_infl.sum()),
        "blind_mass_frac": float(blind_infl.sum() / total) if total else 0.0,
        # An empty blind set is a legal (degenerate) input -- a reduction over it
        # must return 0.0, never raise, so callers can sweep masks without guards.
        "blind_px_active_frac": float(active.mean()) if blind_infl.size else 0.0,
        "blind_influence_max": float(blind_infl.max()) if blind_infl.size else 0.0,
        "blind_influence_mean_active": float(blind_infl[active].mean())
        if active.any()
        else 0.0,
    }
