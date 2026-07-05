# SPDX-License-Identifier: MIT
"""Exact column norms of the LINEAR R-chain (bicubic-up 384x512 -> 874x1164, then
bilinear-down -> 384x512): the theta-independent, pair-independent, video-independent
GEOMETRY factor of the LEVER-4 reachability weight S_R (#268).

DERIVATION (exact, closed form)
-------------------------------
Under the identity-gradient STE convention for the uint8 round (the convention the
trainer, the contest-R torch reference ``tools/precompute_sR_reachability.py::_R_torch``,
and the MLX fused-R all use: ``q = clipped + (round(clipped) - clipped).detach()``), the
Jacobian of R w.r.t. the witness-grid frame is EXACTLY the linear resample chain

    dR/dx = D = B_down @ C_up            (clamp[0,255] treated as identity; see CAVEAT)

Both resamplers are SEPARABLE: with 1-D operators ``C_v (874x384)``, ``C_h (1164x512)``
(bicubic, align_corners=False, A=-0.75) and ``B_v (384x874)``, ``B_h (512x1164)``
(bilinear, align_corners=False), the 2-D operator on images is the Kronecker product

    D = D_v (x) D_h,   D_v = B_v @ C_v  (384x384),   D_h = B_h @ C_h  (512x512).

The L1 column norm of a Kronecker product factorizes EXACTLY:

    || column_(r,c) of D ||_1 = sum_ij |D_v[i,r] * D_h[j,c]|
                              = (sum_i |D_v[i,r]|) * (sum_j |D_h[j,c]|)
                              = a_v[r] * a_h[c]

so the per-pixel "how much can a unit witness-grid perturbation at (r,c) move the
scorer's input" gain map is the OUTER PRODUCT of two 1-D profiles — a SINGLE static
(384,512) map, independent of the pair, the video content, and theta. That is the
"beautiful simplification" answer for the R-CHAIN factor: YES, it is one static map.

WHAT THIS MAP IS NOT (honest scope — measured 2026-07-05, see the #268 memo)
----------------------------------------------------------------------------
The FULL reachability S_R = |(w . dmargin/dRx) @ D| composes this geometry factor with
the frozen-SegNet margin Jacobian, which IS content/pair-dependent. The canonical cached
per-pair ``sR`` (``tools/precompute_sR_reachability.py``) remains the trainer's weight;
this module supplies (a) the exact numpy REFERENCE for the linear-chain part (parity
target for the torch/MLX chains), (b) the measured verdict on how much of ``sR`` the
static geometry factor explains (answer: very little — the SegNet content term
dominates; the geometry ripple is a small-amplitude resample-lattice Moire), and
(c) a cacheable exact gain map for any future lever that needs the pure R-chain gain.

CAVEAT (stated, not hidden): the camera-res ``clamp(0,255)`` is data-dependent and is
treated as identity here (valid at every non-saturated camera pixel; the saturated
fraction on real GT frames is measured and reported in the #268 memo/probe, not assumed).

Axis: pure geometry/math — no score claim. Pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import hashlib

import numpy as np

CAMERA_HW = (874, 1164)
SEG_HW = (384, 512)
_BICUBIC_A = -0.75  # torch F.interpolate(mode="bicubic") cubic-convolution constant


def _cubic_weight(t: np.ndarray, a: float = _BICUBIC_A) -> np.ndarray:
    """Cubic convolution kernel W(t) (Keys), torch's A=-0.75 convention. |t|<=2 support."""
    t = np.abs(t)
    w = np.zeros_like(t)
    m1 = t <= 1.0
    m2 = (t > 1.0) & (t < 2.0)
    w[m1] = (a + 2.0) * t[m1] ** 3 - (a + 3.0) * t[m1] ** 2 + 1.0
    w[m2] = a * t[m2] ** 3 - 5.0 * a * t[m2] ** 2 + 8.0 * a * t[m2] - 4.0 * a
    return w


def bicubic_resample_matrix_1d(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) 1-D bicubic resample matrix, torch align_corners=False semantics:
    src = (dst+0.5)*(n_in/n_out) - 0.5 (UNclamped for cubic); 4 taps at floor(src)-1..+2,
    tap indices border-clamped into [0, n_in-1] (replicate padding accumulates weight)."""
    scale = n_in / n_out
    M = np.zeros((n_out, n_in), dtype=np.float64)
    dst = np.arange(n_out, dtype=np.float64)
    src = (dst + 0.5) * scale - 0.5
    i0 = np.floor(src).astype(np.int64)
    t = src - i0
    for k in range(-1, 3):
        w = _cubic_weight(k - t)  # weight of tap i0+k at fractional offset t
        idx = np.clip(i0 + k, 0, n_in - 1)
        np.add.at(M, (np.arange(n_out), idx), w)
    return M


def bilinear_resample_matrix_1d(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) 1-D bilinear resample matrix, torch align_corners=False semantics:
    src = max((dst+0.5)*(n_in/n_out) - 0.5, 0) (clamped low for linear modes); 2 taps
    floor/floor+1, upper tap index clamped to n_in-1."""
    scale = n_in / n_out
    M = np.zeros((n_out, n_in), dtype=np.float64)
    dst = np.arange(n_out, dtype=np.float64)
    src = np.maximum((dst + 0.5) * scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    frac = src - i0
    np.add.at(M, (np.arange(n_out), i0), 1.0 - frac)
    np.add.at(M, (np.arange(n_out), i1), frac)
    return M


def rchain_1d_operator(n_seg: int, n_cam: int) -> np.ndarray:
    """The composed 1-D chain D = bilinear_down(n_cam->n_seg) @ bicubic_up(n_seg->n_cam),
    shape (n_seg, n_seg). Exact dense float64."""
    C_up = bicubic_resample_matrix_1d(n_seg, n_cam)   # (n_cam, n_seg)
    B_dn = bilinear_resample_matrix_1d(n_cam, n_seg)  # (n_seg, n_cam)
    # np.errstate: Apple-Accelerate BLAS spuriously raises FP flags on these sparse-band
    # float64 matmuls; the products are exact/finite (guarded by the test suite's finiteness
    # assertion + torch-parity at ~1e-14: test_sr_reachability_weight.py).
    with np.errstate(all="ignore"):
        return B_dn @ C_up


def rchain_column_l1_profiles(
    seg_hw: tuple[int, int] = SEG_HW, camera_hw: tuple[int, int] = CAMERA_HW
) -> tuple[np.ndarray, np.ndarray]:
    """(a_v (H,), a_h (W,)) = exact L1 column norms of the composed 1-D chains per axis."""
    D_v = rchain_1d_operator(seg_hw[0], camera_hw[0])
    D_h = rchain_1d_operator(seg_hw[1], camera_hw[1])
    return np.abs(D_v).sum(axis=0), np.abs(D_h).sum(axis=0)


def rchain_column_l1_map(
    seg_hw: tuple[int, int] = SEG_HW, camera_hw: tuple[int, int] = CAMERA_HW
) -> np.ndarray:
    """THE static geometry gain map S_geo (H,W) float64 = outer(a_v, a_h): exact L1 column
    norm of the full 2-D linear R-chain Jacobian at every witness-grid pixel. Pair-,
    theta-, and video-INDEPENDENT (fixed resample geometry)."""
    a_v, a_h = rchain_column_l1_profiles(seg_hw, camera_hw)
    return np.outer(a_v, a_h)


def rchain_signed_colsum_map(
    seg_hw: tuple[int, int] = SEG_HW, camera_hw: tuple[int, int] = CAMERA_HW
) -> np.ndarray:
    """Outer product of the SIGNED column sums (H,W). Equals the VJP of an all-ones
    cotangent through the (unclamped, STE-identity) up->down chain — the torch-parity
    surface (autograd cannot produce |entries|, but it produces exactly this)."""
    D_v = rchain_1d_operator(seg_hw[0], camera_hw[0])
    D_h = rchain_1d_operator(seg_hw[1], camera_hw[1])
    return np.outer(D_v.sum(axis=0), D_h.sum(axis=0))


def rchain_map_sha256(seg_hw: tuple[int, int] = SEG_HW,
                      camera_hw: tuple[int, int] = CAMERA_HW) -> str:
    """Deterministic provenance sha of the float64 S_geo bytes (C-order)."""
    m = np.ascontiguousarray(rchain_column_l1_map(seg_hw, camera_hw))
    return hashlib.sha256(m.tobytes()).hexdigest()
